from os.path import join
from run_xnli import XnliRunner
import os
import sys
import util
import pickle
from selection import get_selection_prelim
from sklearn.metrics import roc_auc_score


class Evaluator:
    def __init__(self, config_name, saved_suffix, gpu_id):
        self.saved_suffix = saved_suffix
        self.runner = XnliRunner(config_name, gpu_id)
        self.conf = self.runner.config
        self.model = None

        self.output_dir = join(self.conf['log_dir'], 'results', saved_suffix)
        os.makedirs(self.output_dir, exist_ok=True)

    def get_model(self):
        if self.model is None:
            self.model = self.runner.initialize_model(saved_suffix=self.saved_suffix)
        return self.model

    def get_output_result_file(self, lang, partition):
        return join(self.output_dir, f'results_{partition}_{lang}.bin')

    def get_output_prediction_file(self, lang, partition):
        return join(self.output_dir, f'predictions_{partition}_{lang}.tsv')

    def get_output_metrics_file(self, lang, partition):
        return join(self.output_dir, f'metrics_{partition}_{lang}.json')

    def evaluate_task(self, use_un_probs=None, eval_dev=False):
        all_acc, partition = [], 'dev' if eval_dev else 'test'
        for lang in util.langs:
            dataset = self.runner.data.get_data(partition, lang, only_dataset=True)
            result_path = self.get_output_result_file(lang, partition)

            if os.path.exists(result_path):
                with open(result_path, 'rb') as f:
                    results = pickle.load(f)
                metrics, _, _, _, _ = self.runner.evaluate_from_results(results, use_un_probs=use_un_probs, print_report=True)
            else:
                metrics, _, _, _, _ = self.runner.evaluate(self.get_model(), dataset, output_results_file=result_path,
                                                           use_un_probs=use_un_probs, print_report=True)
            all_acc.append(metrics['acc'])
            print(f'Metrics for {lang}:')
            for name, val in metrics.items():
                print(f'{name}: {val:.4f}')
            print('-' * 20)

        print('-' * 20)
        print('\n'.join(util.print_all_scores(all_acc, 'acc', with_en=True, latex_scale=100)))

    def evaluate_uncertainty(self, eval_dev=False):
        all_auc, partition = [], 'dev' if eval_dev else 'test'
        for lang in util.langs:
            result_path = self.get_output_result_file(lang, partition)
            with open(result_path, 'rb') as f:
                results = pickle.load(f)

            metrics, all_preds, all_labels, all_probs, (all_logits, all_un) = self.runner.evaluate_from_results(results,
                                                                        use_un_probs=True, print_report=False)
            uncertainty = get_selection_prelim(all_preds, all_probs, all_logits, all_un, self.conf['sl_criterion'])
            auroc = roc_auc_score(all_preds == all_labels, -uncertainty)  # No - if max prob
            print(f'AUROC for {lang}: {auroc:.4f}')
            all_auc.append(auroc)
        avg_auc = sum(all_auc) / len(all_auc)
        print(f'Avg AUROC: {avg_auc:.4f}')
        print('\n'.join(util.print_all_scores(all_auc, 'auc', with_en=True, latex_scale=100)))


if __name__ == '__main__':
    config_name, saved_suffix, gpu_id = sys.argv[1], sys.argv[2], int(sys.argv[3])
    evaluator = Evaluator(config_name, saved_suffix, gpu_id)
    evaluator.evaluate_task(use_un_probs=True)  # use_un_probs has no effect for eval metrics (argmax)
    # evaluator.evaluate_uncertainty()
