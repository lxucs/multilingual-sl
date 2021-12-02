import logging
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from transformers import AdamW, get_linear_schedule_with_warmup, get_constant_schedule
import util
import time
from tensorize_tag import TagDataProcessor
from os.path import join
from datetime import datetime
import sys
from collections import defaultdict
import pickle
import random
import util_tag
from model_tag import TransformerTag
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, Subset, ConcatDataset
from torch.optim.lr_scheduler import LambdaLR
from seqeval.metrics import classification_report
from seqeval.metrics.sequence_labeling import precision_recall_fscore_support
from selection import get_selection_prelim

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger()


class TagRunner:
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
        self.data = TagDataProcessor(self.config)

    def initialize_model(self, saved_suffix=None, config_name=None):
        num_labels = len(self.data.get_labels(self.config['dataset_name']))
        model = TransformerTag(self.config, num_labels)
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

    def sl_select_dataset(self, model, lang, selected_indices, selected_labels, dev_dataset_by_lang,
                          criterion='max_prob', shortcut_1st_itr=True):
        assert lang != 'en'
        higher_is_better = util.if_higher_is_better(criterion)

        conf = self.config
        dataset_name = conf['dataset_name']
        dataset = dev_dataset_by_lang[lang]

        # Get dataset for prediction
        prev_indices = {idx for indices in selected_indices[:-1] for idx in indices[lang]}
        remaining_indices = list(set(range(len(dataset))).difference(prev_indices))
        idx_to_i = {idx: i for i, idx in enumerate(remaining_indices)}
        top_k = min(max(int(len(dataset) * conf['sl_top_k_ratio']), 1), len(remaining_indices))
        threshold = conf['sl_selection_threshold']

        # Predict
        if shortcut_1st_itr and len(selected_indices) == 1:  # First itr pred is always same from warm-start
            other = '_nsp' if not conf['use_un_probs'] else ''
            result_path = join(conf['log_root'], conf['init_config_name'], 'selection', conf['init_suffix'],
                               f'results_{dataset_name}_{lang}{other}.bin')  # SelectionAnalyzer.evaluate()
            with open(result_path, 'rb') as f:
                results = pickle.load(f)
                _, predictions, _, probs, raw_pred = results[:5]  # Order should match initial remaining_indices
                logits = results[5] if len(results) > 5 else None
            logger.info(f'Shortcut first iteration prediction from {result_path}')
        else:
            _, predictions, _, probs, raw_pred, logits = self.evaluate_simple(model, Subset(dataset, remaining_indices),
                                                                              only_predict=True, get_raw=True,
                                                                              dataset_name=dataset_name, lang=lang)
        # Backward compatibility
        tag_un = None
        if isinstance(probs, tuple):
            probs, tag_un = probs

        # Get new selection indices
        if dataset_name == 'panx':
            # Get selection prelim: type_scores
            type_scores, idx_more_than_one_type = get_selection_prelim(dataset_name, remaining_indices, predictions,
                                                                       probs, tag_un, logits, criterion)
            # Get indices with top entity scores per type
            selected_idx, all_idx = self._get_selected_indices_per_type(type_scores, higher_is_better, top_k, threshold)
            # Get final indices; final idx must be either in only one or all types
            final_idx = self._get_final_selected_indices(selected_idx, all_idx, idx_more_than_one_type)
        else:
            raise ValueError(dataset_name)

        # Update selected_indices
        selected_indices[-1][lang] += final_idx
        # Update dataset with predicted tags for self-training
        final_idx_i = [idx_to_i[idx] for idx in final_idx]
        dataset.tensors[-1][torch.LongTensor(final_idx)] = torch.as_tensor(raw_pred[final_idx_i], dtype=torch.long)
        selected_labels[-1][lang] = raw_pred[final_idx_i]  # Keep SL state

        return len(final_idx)

    def _get_selected_indices_per_type(self, type_scores, higher_is_better, top_k=None, threshold=None):
        # Get indices with top entity scores per type
        selected_idx = defaultdict(list)  # Top idx per type
        all_idx = defaultdict(list)  # All idx per type
        for t, entity_scores in type_scores.items():
            all_idx[t] = {idx for idx, _ in type_scores[t]}
            sorted_entity_scores = sorted(entity_scores, key=lambda tup: tup[1], reverse=higher_is_better)
            sorted_idx = [idx for idx, score in sorted_entity_scores]
            sorted_scores = [score for idx, score in sorted_entity_scores]
            # Identify num selection
            num_selection = len(sorted_idx)
            if threshold:
                if higher_is_better:
                    num_selection = len(entity_scores) - np.searchsorted(sorted_scores[::-1], threshold, side='left')
                else:
                    num_selection = np.searchsorted(sorted_scores, threshold, side='right')
            if top_k:
                if top_k > num_selection and threshold:
                    logger.info(f'Throttle selection by threshold: {top_k} to {num_selection}')
                else:
                    num_selection = top_k
            selected_idx[t] = sorted_idx[:num_selection]
        return selected_idx, all_idx

    def _get_final_selected_indices(self, selected_idx, all_idx, idx_more_than_one_type):
        final_idx = []  # Final instance indices to select; must be either in only one or all types
        types = tuple(selected_idx.keys())
        merged_selected_idx = set().union(*[indices for t, indices in selected_idx.items()])
        idx_more_than_one_type = set(idx_more_than_one_type)
        for idx in merged_selected_idx:
            if idx not in idx_more_than_one_type:
                final_idx.append(idx)
            else:
                valid = True
                for t in types:
                    if idx in all_idx[t] and idx not in selected_idx[t]:
                        valid = False
                        break
                if valid:
                    final_idx.append(idx)
        return final_idx

    def _get_sl_train_dataset(self, en_train_dataset, dev_dataset_by_lang, selected_indices):
        train_dataset = {}
        for lang in util.langs[self.config['dataset_name']]:
            if lang == 'en':
                num_en = int(len(en_train_dataset) * self.config['sl_en_ratio'])
                train_dataset[lang] = Subset(en_train_dataset, random.sample(range(len(en_train_dataset)), k=num_en))
            else:
                if self.config['sl_lang_ratio']:
                    curr_indices = selected_indices[-1][lang]
                    prev_indices = [idx for indices in selected_indices[:-1] for idx in indices[lang]]
                    all_indices = curr_indices + random.sample(prev_indices,
                                    k=min(len(prev_indices), int(len(curr_indices) * self.config['sl_lang_ratio'])))
                else:
                    all_indices = [idx for indices in selected_indices for idx in indices[lang]]
                train_dataset[lang] = Subset(dev_dataset_by_lang[lang], all_indices)
        train_dataset = ConcatDataset([ds for lang, ds in train_dataset.items()])
        return train_dataset

    def train_full(self, model, state_suffix=None):
        conf = self.config
        logger.info(conf)

        # Set up tensorboard
        tb_path = join(conf['tb_dir'], self.name + '_' + self.name_suffix)
        tb_writer = SummaryWriter(tb_path, flush_secs=30)
        logger.info(f'Tensorboard summary path: {tb_path}')

        # Set up data
        dataset_name = conf['dataset_name']
        en_train_dataset = self.data.get_data(dataset_name, 'train', 'en', only_dataset=True)
        en_dev_dataset = self.data.get_data(dataset_name, 'dev', 'en', only_dataset=True)
        dev_dataset_by_lang = {lang: self.data.get_data(dataset_name, 'dev', lang, only_dataset=True)
                               for lang in util.langs[dataset_name] if lang != 'en'}

        # Initialize SL states
        if state_suffix is None:
            itr, selected_indices, selected_labels = 1, [], []  # itr = 1 if warm-start
            loss_history = []  # Full history of effective loss; length equals total update steps
            max_eval_scores = []
        else:
            loss_history, itr, selected_indices, selected_labels, max_eval_scores = self.load_sl_state(state_suffix)
            for lang in util.langs[dataset_name]:
                if lang == 'en':
                    continue
                all_indices = [idx for indices in selected_indices for idx in indices[lang]]
                all_labels = np.concatenate([labels[lang] for labels in selected_labels], axis=0)
                dev_dataset_by_lang[lang].tensors[-1][torch.LongTensor(all_indices)] = torch.as_tensor(all_labels, dtype=torch.long)

        # Start iterative training
        while itr < conf['sl_max_itr']:
            logger.info('=' * 20 + f'SL Iteration {itr}' + '=' * 20)

            train_dataset, epochs = None, None  # For current training iteration
            if itr == 0:
                train_dataset, epochs = en_train_dataset, conf['num_epochs']
            else:
                epochs = conf['sl_num_epochs']
                # Select new training data; update selected_indices in sl_select_dataset()
                num_new_selected = 0
                selected_indices.append(defaultdict(list))
                selected_labels.append({})
                for lang in util.langs[dataset_name]:
                    if lang != 'en':
                        num_new_selected += self.sl_select_dataset(model, lang, selected_indices, selected_labels,
                                                                   dev_dataset_by_lang, criterion=conf['sl_criterion'])
                logger.info(f'Num newly selected examples: {num_new_selected}')
                # Make new training dataset
                train_dataset = self._get_sl_train_dataset(en_train_dataset, dev_dataset_by_lang, selected_indices)

            # Train
            max_eval_score = self.train_single(model, train_dataset, en_dev_dataset, loss_history, epochs, tb_writer)
            max_eval_scores.append(max_eval_score)
            itr += 1

            # Save SL state
            self.save_sl_state(loss_history, itr, selected_indices, selected_labels, max_eval_scores)

        # Wrap up
        tb_writer.close()
        logger.info('max_eval_scores for each itr: ' + '\t'.join([f'{s: .4f}' for s in max_eval_scores]))
        logger.info('Finished SL')
        return loss_history, max_eval_scores, selected_indices, selected_labels

    def train_single(self, model, train_dataset, eval_dataset, loss_history, epochs=None, tb_writer=None):
        conf = self.config
        epochs = epochs or conf['num_epochs']
        batch_size, grad_accum = conf['batch_size'], conf['gradient_accumulation_steps']

        model.to(self.device)

        # Set up data
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
        max_eval_score = 0
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
                        metrics, _, _, _, _, _ = self.evaluate_simple(model, eval_dataset, len(loss_history), use_un_probs=False,
                                                                      tb_writer=tb_writer, dataset_name=conf['dataset_name'])
                        if metrics['eval_score'] > max_eval_score:
                            max_eval_score = metrics['eval_score']
                            # self.save_model_checkpoint(model, len(loss_history))
                        logger.info(f'Max eval score: {max_eval_score:.4f}')
                        start_time = time.time()

        logger.info('**********Finished training**********')
        logger.info('Actual update steps: %d' % len(loss_history))

        # Eval at the end
        metrics, _, _, _, _, _ = self.evaluate_simple(model, eval_dataset, len(loss_history), use_un_probs=False,
                                                      tb_writer=tb_writer, dataset_name=conf['dataset_name'])
        if metrics['eval_score'] > max_eval_score:
            max_eval_score = metrics['eval_score']
        self.save_model_checkpoint(model, len(loss_history))
        logger.info(f'Max eval score: {max_eval_score:.4f}')
        return max_eval_score

    def predict_w_hidden(self, model, dataset, indices, dataset_name=None, lang=None):
        dataset = Subset(dataset, indices)
        logger.info(f'Predicting on {len(dataset)} samples...')
        dataloader = DataLoader(dataset, sampler=SequentialSampler(dataset), batch_size=32)
        label_map = {i: label for i, label in enumerate(self.data.get_labels(dataset_name))}

        model.eval()
        model.to(self.device)
        all_hiddens = []
        idx_to_range, total_tags = {}, 0
        for batch_i, batch in enumerate(dataloader):
            inputs = self.prepare_inputs(batch, with_labels=False)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            inputs['output_hidden'] = True

            with torch.no_grad():
                batch_logits, batch_tag_un, batch_hiddens = model(**inputs)
            batch_logits = batch_logits.detach().cpu().numpy()
            batch_labels = batch[-1].numpy()
            batch_hiddens = batch_hiddens.detach().cpu().numpy()
            for i in range(batch_logits.shape[0]):
                mask = batch_labels[i] != util.pad_token_label_id
                hiddens = batch_hiddens[i][mask]
                logits = batch_logits[i][mask]
                preds = [label_map[label_id] for label_id in logits.argmax(axis=-1).tolist()]

                entities = util_tag.get_entities(preds)
                range_s = total_tags
                for entity in entities:
                    all_hiddens.append(hiddens[entity[1]:entity[2] + 1])
                    total_tags += (entity[2] + 1 - entity[1])
                range_e = total_tags  # Exclusive
                idx_to_range[indices[len(idx_to_range)]] = (range_s, range_e)

        all_hiddens = np.concatenate(all_hiddens, axis=0)
        assert all_hiddens.shape[0] == total_tags
        assert len(idx_to_range) == len(indices)
        return all_hiddens, idx_to_range

    def evaluate_simple(self, model, dataset, step=0, only_predict=False, get_raw=False,
                        tb_writer=None, dataset_name=None, lang=None, use_un_probs=None, print_report=False):
        """
        Simple one-pass evaluation; use F1 score as final eval score.
        :returns (metrics, predictions, golds, probs); metrics is None when only_predict
        """
        conf = self.config
        use_un_probs = conf['use_un_probs'] if use_un_probs is None else use_un_probs
        logger.info(f'Step {step}: evaluating on {len(dataset)} samples...')
        dataloader = DataLoader(dataset, sampler=SequentialSampler(dataset), batch_size=conf['eval_batch_size'])

        # Get results
        model.eval()
        model.to(self.device)
        all_logits, all_labels, all_tag_un = [], [], []
        for batch_i, batch in enumerate(dataloader):
            inputs = self.prepare_inputs(batch, with_labels=False)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                logits, tag_un = model(**inputs)
            all_logits.append(logits.detach().cpu())
            all_labels.append(batch[-1])
            all_tag_un.append(None if tag_un is None else tag_un.detach().cpu())

        all_logits = torch.cat(all_logits, dim=0).numpy()
        all_labels = torch.cat(all_labels, dim=0).numpy()
        all_tag_un = None if all_tag_un[0] is None else torch.cat(all_tag_un, dim=0).numpy()

        # Get predictions
        label_map = {i: label for i, label in enumerate(self.data.get_labels(dataset_name))}
        tag_un, logits, probs, predictions, golds = [], [], [], [], []
        for i in range(all_logits.shape[0]):
            mask = all_labels[i] != util.pad_token_label_id
            logits.append(all_logits[i][mask])
            if all_tag_un is not None:
                tag_un.append(all_tag_un[i][mask])
            if use_un_probs and all_tag_un is not None:
                probs.append(TransformerTag.get_probs(logits[-1], tag_un=tag_un[-1],
                                                      as_regression=conf['tag_un_as_regression'], mc=20))
            else:
                probs.append(TransformerTag.get_probs(logits[-1], evi_un=conf['evi_un']))
            predictions.append([label_map[label_id] for label_id in probs[-1].argmax(axis=-1).tolist()])
            golds.append([label_map[label_id] for label_id in all_labels[i][mask].tolist()])

        raw_pred = None
        if get_raw:
            raw_pred = all_logits.argmax(axis=-1)
            mask = (all_labels != util.pad_token_label_id).astype(int)
            raw_pred = raw_pred * mask + (1 - mask) * util.pad_token_label_id

        # Get metrics
        if only_predict:
            return None, predictions, golds, (probs, tag_un or None), raw_pred, logits  # Align with end return
        p, r, f, _ = precision_recall_fscore_support(golds, predictions, beta=1, average='micro')  # Weighted average
        eval_score = f  # Make F1 as the final eval score
        metrics = {'f1': f, 'precision': p, 'recall': r, 'eval_score': eval_score}

        if print_report:
            logger.info(f'\n{classification_report(golds, predictions)}')
        if tb_writer:
            for name, val in metrics.items():
                tb_writer.add_scalar(f'Train_Eval_{name}', val, step)
        return metrics, predictions, golds, (probs, tag_un or None), raw_pred, logits

    def evaluate_simple_all_langs(self, model, itr=0, tb_writer=None, use_un_probs=None, print_report=False):
        use_un_probs = False if use_un_probs is None else use_un_probs  # For eval metrics, use_un_probs has no effect
        all_f1, dataset_name = [], self.config['dataset_name']
        for lang in util.langs[dataset_name]:
            dataset = self.data.get_data(dataset_name, 'test', lang, only_dataset=True)
            metrics, _, _, _, _, _ = self.evaluate_simple(model, dataset, dataset_name=dataset_name, lang=lang,
                                                          use_un_probs=use_un_probs, print_report=print_report)
            all_f1.append(metrics['f1'])

        avg_f1 = sum(all_f1) / len(all_f1)
        if tb_writer:
            tb_writer.add_scalar(f'Train_Eval_All_Langs', avg_f1, itr)

        logger.info('Eval all langs (test):')
        logger.info('\n'.join(util.print_all_scores(all_f1, 'f1', with_en=True, latex_scale=100)))
        return avg_f1, all_f1

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
        model.load_state_dict(torch.load(path_ckpt, map_location=torch.device('cpu')), strict=False)
        logger.info('Loaded model from %s' % path_ckpt)

    def save_sl_state(self, *to_save):
        path = join(self.config['log_dir'], f'sl_state_{self.name_suffix}.bin')
        torch.save(to_save, path)
        logger.info('Saved SL state to %s' % path)

    def load_sl_state(self, suffix, config_name=None):
        if config_name is None:
            path = join(self.config['log_dir'], f'sl_state_{suffix}.bin')
        else:
            path = join(self.config['log_root'], config_name, f'sl_state_{suffix}.bin')
        sl_state = torch.load(path)
        logger.info('Loaded SL state from %s' % path)
        return sl_state


if __name__ == '__main__':
    # Train SL
    config_name, gpu_id = sys.argv[1], int(sys.argv[2])
    runner = TagRunner(config_name, gpu_id)
    model = runner.initialize_model(runner.config['init_suffix'], runner.config['init_config_name'])
    runner.train_full(model)
