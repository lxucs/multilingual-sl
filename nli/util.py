from os import makedirs
from os.path import join
import numpy as np
import pyhocon
import logging
import torch
import random
from transformers import BertTokenizer, XLMRobertaTokenizer, T5Tokenizer
from torch.nn import CrossEntropyLoss

logger = logging.getLogger(__name__)

langs = 'en ar bg de el es fr hi ru sw th tr ur vi zh'.split()
lang_to_id = {lang: i for i, lang in enumerate(langs)}


pad_token_label_id = CrossEntropyLoss().ignore_index
assert pad_token_label_id < 0


def flatten(l):
    return [item for sublist in l for item in sublist]


def if_higher_is_better(criterion):
    assert criterion in ['max_prob', 'entropy', 'var', 'vacuity', 'dissonance', 'custom']
    return True if criterion == 'max_prob' else False


def compute_acc(preds, golds):
    return (preds == golds).mean().item()


def compute_softmax(scores):
    """ axis: -1 """
    exp_scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    probs = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
    return probs


def compute_entropy(probs):
    num_cls = probs.shape[1]
    return -probs * (np.log(probs) / np.log(num_cls))


def initialize_config(config_name, create_dir=True):
    logger.info("Experiment: {}".format(config_name))

    config = pyhocon.ConfigFactory.parse_file("experiments.conf")[config_name]

    config['log_dir'] = join(config["log_root"], config_name)
    config['tb_dir'] = join(config['log_root'], 'tensorboard')
    if create_dir:
        makedirs(config['log_dir'], exist_ok=True)
        makedirs(config['tb_dir'], exist_ok=True)

    logger.info(pyhocon.HOCONConverter.convert(config, "hocon"))
    return config


def set_seed(seed, set_gpu=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if set_gpu and torch.cuda.is_available():
        # Necessary for reproducibility; lower performance
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.cuda.manual_seed_all(seed)


def print_all_scores(all_scores, score_name, with_en, latex_scale=None):
    print_str = []

    avg_score = sum(all_scores) / len(all_scores)
    print_str.append(f'Avg {score_name}: {avg_score:.4f}')

    print_str.append('LaTex table:')
    all_scores = all_scores if with_en else ([0] + all_scores)  # If without en, set 0
    all_scores = [f'{s * latex_scale :.1f}' if latex_scale else f'{s:.2f}' for s in all_scores]
    print_str.append(' & '.join(all_scores))

    return print_str


def set_seed(seed, set_gpu=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if set_gpu and torch.cuda.is_available():
        # Necessary for reproducibility; lower performance
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.cuda.manual_seed_all(seed)
    logger.info('Random seed is set to %d' % seed)


def get_bert_tokenizer(config):
    # Avoid using fast tokenization
    if config['model_type'] == 'mt5':
        return T5Tokenizer.from_pretrained(config['pretrained'])
    elif config['model_type'] == 'bert':
        return BertTokenizer.from_pretrained(config['pretrained'])
    elif config['model_type'] == 'xlmr':
        return XLMRobertaTokenizer.from_pretrained(config['pretrained'])
    else:
        raise ValueError('Unknown model type')
