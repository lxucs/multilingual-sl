from os.path import join
from run_tag import TagRunner
import os
import sys
import json
from util import langs
import util
import pickle
from selection import get_scoring_func
from model_tag import TransformerTag
from sklearn.metrics import roc_auc_score
import numpy as np


class Evaluator:
    """ Use with run_tag.py for evaluation """
    def __init__(self, config_name, saved_suffix, gpu_id):
        self.saved_suffix = saved_suffix
        self.runner = TagRunner(config_name, gpu_id)
        self.model = None

        self.output_dir = join(self.runner.config['log_dir'], 'results', saved_suffix)
        os.makedirs(self.output_dir, exist_ok=True)

    def get_model(self):
        if self.model is None:
            self.model = self.runner.initialize_model(self.saved_suffix)
        return self.model

    def get_output_result_file(self, dataset_name, lang, eval_dev=False):
        partition = '_dev' if eval_dev else ''
        return join(self.output_dir, f'results_{dataset_name}{partition}_{lang}.bin')

    def get_output_prediction_file(self, dataset_name, lang, eval_dev=False):
        partition = '_dev' if eval_dev else ''
        return join(self.output_dir, f'prediction_{dataset_name}{partition}_{lang}.tsv')

    def get_output_metrics_file(self, dataset_name, lang, eval_dev=False):
        partition = '_dev' if eval_dev else ''
        return join(self.output_dir, f'metrics_{dataset_name}{partition}_{lang}.json')

    def evaluate_task(self, dataset_name, use_un_probs=None, eval_dev=False):
        assert dataset_name in langs

        all_f1 = []
        for lang in langs[dataset_name]:
            examples, _, dataset = self.runner.data.get_data(dataset_name, 'dev' if eval_dev else 'test', lang, only_dataset=False)
            result_path = self.get_output_result_file(dataset_name, lang, eval_dev)

            if os.path.exists(result_path):
                with open(result_path, 'rb') as f:
                    results = pickle.load(f)
                metrics, predictions, _, _, _ = self.runner.evaluate_from_results(results, dataset_name, lang,
                                                                                  use_un_probs=use_un_probs,
                                                                                  print_report=True)
            else:
                metrics, predictions, _, _, _ = self.runner.evaluate(
                    self.get_model(), dataset, output_results_file=result_path, dataset_name=dataset_name, lang=lang,
                    use_un_probs=use_un_probs, print_report=True)

            # Save predictions and metrics
            self.runner.save_predictions(examples, predictions, self.get_output_prediction_file(dataset_name, lang, eval_dev))
            with open(self.get_output_metrics_file(dataset_name, lang, eval_dev), 'w') as f:
                json.dump(metrics, f)

            all_f1.append(metrics['f1'])
            print(f'Metrics for {dataset_name}-{lang}')
            for name, val in metrics.items():
                print(f'{name}: {val:.4f}')
            print('-' * 20)

        print('-' * 20)
        print('\n'.join(util.print_all_scores(all_f1, 'f1', with_en=True, latex_scale=100)))

    def evaluate_uncertainty(self, dataset_name, eval_dev=False):
        all_auc = []
        conf = self.runner.config
        use_un_probs = True
        label_map = {i: label for i, label in enumerate(self.runner.data.get_labels(dataset_name))}
        scoring_func = get_scoring_func(conf['sl_criterion'])

        for lang in langs[dataset_name]:
            result_path = self.get_output_result_file(dataset_name, lang, eval_dev)
            with open(result_path, 'rb') as f:
                results = pickle.load(f)

            all_logits, all_labels, all_tag_un = results
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
            tag_un = tag_un or ([None] * len(probs))

            un_labels, un_scores = [], []
            for inst_uns, inst_logits, inst_probs, inst_preds, inst_golds in zip(tag_un, logits, probs, predictions, golds):
                for i in range(len(inst_golds)):
                    # if inst_golds[i] == 'O' and inst_preds[i] == 'O':
                    if inst_golds[i] == 'O':
                        continue
                    un_labels.append(inst_golds[i] == inst_preds[i])
                    un_scores.append(scoring_func(inst_probs, inst_uns, inst_logits, (None, i, i, None)).item())
            auroc = roc_auc_score(un_labels, -np.array(un_scores))
            print(f'AUROC for {lang}: {auroc:.4f}')
            all_auc.append(auroc)
        avg_auc = sum(all_auc) / len(all_auc)
        print(f'Avg AUROC: {avg_auc:.4f}')
        print('\n'.join(util.print_all_scores(all_auc, 'auc', with_en=True, latex_scale=100)))


if __name__ == '__main__':
    config_name, saved_suffix, gpu_id, dataset_name = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
    evaluator = Evaluator(config_name, saved_suffix, gpu_id)
    evaluator.evaluate_task(dataset_name, use_un_probs=False)
