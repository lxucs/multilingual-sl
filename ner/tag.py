import logging
from io import open
from transformers import XLMTokenizer
import torch
from torch.utils.data.dataset import TensorDataset
import util

logger = logging.getLogger(__name__)


class InputExample(object):
    def __init__(self, guid, words, labels, langs=None):
        self.guid = guid
        self.words = words
        self.labels = labels
        self.langs = langs  # One langID per word for backward compatibility


class InputFeatures(object):
    def __init__(self, input_ids, input_mask, segment_ids, label_ids, langs=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids
        self.langs = langs  # Just use one langID per instance


def read_examples_from_file(file_path, lang='en', lang2id=None):
    examples = []
    guid_index = 1
    lang_id = lang2id[lang] if lang2id else 0

    with open(file_path, encoding="utf-8") as f:
        words, labels, langs = [], [], []
        for line in f:
            if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                # End of an example
                if word:
                    example = InputExample(guid="%s-%d" % (lang, guid_index),
                                           words=words, labels=labels, langs=langs)
                    examples.append(example)
                    guid_index += 1
                    words, labels, langs = [], [], []
            else:
                splits = line.split("\t")
                word = splits[0]
                label = splits[-1].strip() if len(splits) > 1 else 'O'

                words.append(word)
                labels.append(label)
                langs.append(lang_id)
        if words:
            examples.append(InputExample(guid="%s-%d" % (lang, guid_index),
                                         words=words, labels=labels, langs=langs))
    return examples


def convert_examples_to_features(examples,
                                 label_list,
                                 max_seq_length,
                                 tokenizer,
                                 config,
                                 pad_token_label_id=-1,
                                 lang='en',
                                 lang2id=None,
                                 return_dataset=True):
    label_map = {label: i for i, label in enumerate(label_list)}
    lang_id = lang2id[lang] if lang2id else 0

    # [CLS] + tokens + [SEP] for single sentence; assume one SEP at the end, same segment id, right padding
    # Currently not supporting other input format such as XLNet
    cls_token = tokenizer.cls_token if config['model_type'] not in ['mt5'] else None
    sep_token = tokenizer.sep_token if config['model_type'] not in ['mt5'] else tokenizer.eos_token
    max_num_tokens = max_seq_length - (tokenizer.model_max_length - tokenizer.max_len_single_sentence)
    assert (max_seq_length - max_num_tokens) == int(cls_token is not None) + 1
    assert tokenizer.padding_side == 'right'

    def process_one_feature(tokens, label_ids):
        input_tokens, input_label_ids = [], []
        # CLS
        if cls_token is not None:
            input_tokens.append(cls_token)
            input_label_ids.append(pad_token_label_id)
        # Tokens
        input_tokens += tokens
        input_label_ids += label_ids
        # SEP
        input_tokens.append(sep_token)
        input_label_ids.append(pad_token_label_id)

        input_ids = tokenizer.convert_tokens_to_ids(input_tokens)
        attention_mask = [1] * len(input_ids)
        token_type_ids = [0] * len(input_ids)

        pad_len = max_seq_length - len(input_ids)
        input_ids += [tokenizer.pad_token_id] * pad_len
        attention_mask += [0] * pad_len
        token_type_ids += [tokenizer.pad_token_type_id] * pad_len
        input_label_ids += [pad_token_label_id] * pad_len

        return InputFeatures(input_ids=input_ids, input_mask=attention_mask,
                             segment_ids=token_type_ids, label_ids=input_label_ids, langs=lang_id)

    features = []
    for (ex_index, example) in enumerate(examples):
        # Build features per example
        tokens, label_ids = [], []
        for word, label in zip(example.words, example.labels):
            if isinstance(tokenizer, XLMTokenizer):
                word_tokens = tokenizer.tokenize(word, lang=lang)
            else:
                word_tokens = tokenizer.tokenize(word)
            if len(word) != 0 and len(word_tokens) == 0:
                word_tokens = [tokenizer.unk_token]

            if len(tokens) + len(word_tokens) > max_num_tokens:
                features.append(process_one_feature(tokens, label_ids))
                tokens, label_ids = [], []

            tokens += word_tokens
            # Use the real label id for the first token of the word, and padding ids for the remaining tokens
            label_ids += ([label_map[label]] + [pad_token_label_id] * (len(word_tokens) - 1))

        if tokens:
            features.append(process_one_feature(tokens, label_ids))

    # Build dataset
    if return_dataset:
        all_input_ids = torch.LongTensor([f.input_ids for f in features])
        all_attention_masks = torch.LongTensor([f.input_mask for f in features])
        all_token_type_ids = torch.LongTensor([f.segment_ids for f in features])
        all_lang_ids = torch.LongTensor([f.langs for f in features])
        all_label_ids = torch.LongTensor([f.label_ids for f in features])
        dataset = TensorDataset(all_input_ids, all_attention_masks, all_token_type_ids, all_lang_ids, all_label_ids)
        return features, dataset
    else:
        return features
