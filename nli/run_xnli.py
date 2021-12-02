import logging
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from transformers import AdamW, get_linear_schedule_with_warmup, get_constant_schedule
import util
import time
from tensorize import XnliDataProcessor
from os.path import join
from datetime import datetime
import sys
import pickle
from model_xnli import TransformerXnli
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.optim.lr_scheduler import LambdaLR
from sklearn.metrics import classification_report

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger()


class XnliRunner:
    def __init__(self, config_name, gpu_id=0, seed=None):
        self.name = config_name
        self.name_suffix = datetime.now().strftime('%b%d_%H-%M-%S')
        self.gpu_id = gpu_id
        self.seed = seed

        # Set up config
        self.config = util.initialize_config(config_name)

        # Set up logger
        log_path = join(self.config['log_dir'], 'log_' + self.name_suffix + '.txt')
        logger.addHandler(logging.FileHandler(log_path, 'a'))
        logger.info(f'Log file path: {log_path}')

        # Set up seed
        if seed:
            util.set_seed(seed)

        # Set up device
        self.device = torch.device('cpu' if gpu_id is None else f'cuda:{gpu_id}')

        # Set up data
        self.data = XnliDataProcessor(self.config)

    def initialize_model(self, saved_suffix=None, config_name=None):
        num_labels = len(self.data.get_labels())
        model = TransformerXnli(self.config, num_labels)
        if saved_suffix:
            self.load_model_checkpoint(model, saved_suffix, config_name)
        if self.config['freeze_emb']:
            model.freeze_emb()
        return model

    def prepare_inputs(self, batch, with_labels=True):
        inputs = {
            'input_ids': batch[0],
            'attention_mask': batch[1],
            'token_type_ids': batch[2],
            'lang_ids': batch[3]
        }
        if with_labels:
            inputs['labels'] = batch[-1]
        return inputs

    def train(self, model):
        conf = self.config
        logger.info(conf)
        epochs, batch_size, grad_accum = conf['num_epochs'], conf['batch_size'], conf['gradient_accumulation_steps']

        model.to(self.device)

        # Set up tensorboard
        tb_path = join(conf['tb_dir'], self.name + '_' + self.name_suffix)
        tb_writer = SummaryWriter(tb_path, flush_secs=30)
        logger.info(f'Tensorboard summary path: {tb_path}')

        # Set up data
        train_dataset = self.data.get_data('train', 'en', only_dataset=True)
        dev_dataset = self.data.get_data('dev', 'en', only_dataset=True)
        train_dataloader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=batch_size, drop_last=False)

        # Set up optimizer and scheduler
        total_update_steps = len(train_dataloader) * epochs // grad_accum
        optimizer = self.get_optimizer(model)
        scheduler = self.get_scheduler(optimizer, total_update_steps)
        trained_params = model.parameters()

        # Start training
        logger.info('*******************Training*******************')
        logger.info('Num samples: %d' % len(train_dataset))
        logger.info('Num epochs: %d' % epochs)
        logger.info('Gradient accumulation steps: %d' % grad_accum)
        logger.info('Total update steps: %d' % total_update_steps)

        loss_during_accum = []  # To compute effective loss at each update
        loss_during_report = 0.0  # Effective loss during logging step
        loss_history = []  # Full history of effective loss; length equals total update steps
        max_acc = 0
        start_time = time.time()
        model.zero_grad()
        for epo in range(epochs):
            for batch in train_dataloader:
                model.train()
                inputs = self.prepare_inputs(batch, with_labels=True)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                loss, _ = model(**inputs)
                if grad_accum > 1:
                    loss /= grad_accum
                loss.backward()
                loss_during_accum.append(loss.item())

                # Update
                if len(loss_during_accum) % grad_accum == 0:
                    if conf['max_grad_norm']:
                        torch.nn.utils.clip_grad_norm_(trained_params, conf['max_grad_norm'])
                    optimizer.step()
                    model.zero_grad()
                    scheduler.step()

                    # Compute effective loss
                    effective_loss = np.sum(loss_during_accum).item()
                    loss_during_accum = []
                    loss_during_report += effective_loss
                    loss_history.append(effective_loss)

                    # Report
                    if len(loss_history) % conf['report_frequency'] == 0:
                        # Show avg loss during last report interval
                        avg_loss = loss_during_report / conf['report_frequency']
                        loss_during_report = 0.0
                        end_time = time.time()
                        logger.info('Step %d: avg loss %.2f; steps/sec %.2f' %
                                    (len(loss_history), avg_loss, conf['report_frequency'] / (end_time - start_time)))
                        start_time = end_time

                        tb_writer.add_scalar('Training_Loss', avg_loss, len(loss_history))
                        tb_writer.add_scalar('Learning_Rate_Bert', scheduler.get_last_lr()[0], len(loss_history))

                    # Evaluate
                    if len(loss_history) > 0 and len(loss_history) % conf['eval_frequency'] == 0:
                        metrics, _, _, _, _ = self.evaluate(model, dev_dataset, len(loss_history), tb_writer, use_un_probs=False)
                        if metrics['acc'] > max_acc:
                            max_acc = metrics['acc']
                            self.save_model_checkpoint(model, len(loss_history))
                        logger.info(f'Eval max acc: {max_acc:.4f}')
                        start_time = time.time()

        logger.info('**********Finished training**********')
        logger.info('Actual update steps: %d' % len(loss_history))

        # Eval at the end
        metrics, _, _, _, _ = self.evaluate(model, dev_dataset, len(loss_history), tb_writer, use_un_probs=False)
        if metrics['acc'] > max_acc:
            max_acc = metrics['acc']
            self.save_model_checkpoint(model, len(loss_history))
        logger.info(f'Eval max acc: {max_acc:.4f}')

        # Wrap up
        tb_writer.close()
        return loss_history

    def evaluate(self, model, dataset, step=0, tb_writer=None, output_results_file=None, dropout=False,
                 use_un_probs=None, print_report=False):
        conf = self.config
        logger.info(f'Step {step}: evaluating on {len(dataset)} samples...')
        dataloader = DataLoader(dataset, sampler=SequentialSampler(dataset), batch_size=conf['eval_batch_size'])

        # Get results
        model.eval()
        if dropout:
            model.train()
        model.to(self.device)
        all_logits, all_labels, all_un = [], [], []
        for batch in dataloader:
            inputs = self.prepare_inputs(batch, with_labels=False)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                logits, un = model(**inputs)
            all_logits.append(logits.detach().cpu())
            all_labels.append(batch[-1])
            all_un.append(None if un is None else un.detach().cpu())

        all_logits = torch.cat(all_logits, dim=0).numpy()
        all_labels = torch.cat(all_labels, dim=0).numpy()
        all_un = None if all_un[0] is None else torch.cat(all_un, dim=0).numpy()
        results = all_logits, all_labels, all_un

        if output_results_file:
            with open(output_results_file, 'wb') as f:
                pickle.dump(results, f)

        # Evaluate
        metrics, preds, labels, probs, (logits, un) = self.evaluate_from_results(results, use_un_probs=use_un_probs,
                                                                                 print_report=print_report)
        if tb_writer:
            for name, val in metrics.items():
                tb_writer.add_scalar(f'Train_Eval_{name}', val, step)
        return metrics, preds, labels, probs, (logits, un)

    def evaluate_from_results(self, results, use_un_probs=None, print_report=False):
        conf = self.config
        use_un_probs = conf['use_un_probs'] if use_un_probs is None else use_un_probs
        all_logits, all_labels, all_un = results

        if use_un_probs and all_un is not None:
            all_probs = TransformerXnli.get_probs(all_logits, un=all_un, as_regression=conf['un_as_regression'], mc=20)
        else:
            all_probs = TransformerXnli.get_probs(all_logits, evi_un=conf['evi_un'])
        all_preds = all_probs.argmax(axis=-1)

        metrics = {'acc': util.compute_acc(all_preds, all_labels)}

        if print_report:
            logger.info(f'\n{classification_report(all_labels, all_preds, target_names=self.data.get_labels())}')
        return metrics, all_preds, all_labels, all_probs, (all_logits, all_un)

    def get_optimizer(self, model):
        no_decay = ['bias', 'LayerNorm.weight']
        grouped_param = [
            {
                'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                'weight_decay': self.config['adam_weight_decay']
            }, {
                'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0
            }
        ]
        optimizer = AdamW(grouped_param, lr=self.config['bert_learning_rate'], eps=self.config['adam_eps'])
        return optimizer

    def get_scheduler(self, optimizer, total_update_steps):
        if self.config['model_type'] == 'mt5':
            # scheduler = get_constant_schedule(optimizer)
            cooldown_start = int(total_update_steps * 0.7)

            def lr_lambda(current_step: int):
                return 1 if current_step < cooldown_start else 0.3

            return LambdaLR(optimizer, lr_lambda, -1)
        else:
            warmup_steps = int(total_update_steps * self.config['warmup_ratio'])
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                        num_training_steps=total_update_steps)
        return scheduler

    def save_model_checkpoint(self, model, step):
        path_ckpt = join(self.config['log_dir'], f'model_{self.name_suffix}.bin')
        torch.save(model.state_dict(), path_ckpt)
        logger.info('Saved model to %s' % path_ckpt)

    def load_model_checkpoint(self, model, suffix, config_name=None):
        if config_name is None:
            path_ckpt = join(self.config['log_dir'], f'model_{suffix}.bin')
        else:
            path_ckpt = join(self.config['log_root'], config_name, f'model_{suffix}.bin')
        model.load_state_dict(torch.load(path_ckpt, map_location=torch.device('cpu')), strict=True)
        logger.info('Loaded model from %s' % path_ckpt)


if __name__ == '__main__':
    # Train
    config_name, gpu_id = sys.argv[1], int(sys.argv[2])
    runner = XnliRunner(config_name, gpu_id)
    model = runner.initialize_model()
    runner.train(model)
