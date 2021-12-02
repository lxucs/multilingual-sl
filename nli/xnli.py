import torch
import logging
from torch.utils.data import TensorDataset

logger = logging.getLogger(__name__)


class InputExample(object):
    def __init__(self, guid, text_a, text_b=None, label=None, lang=None):
        self.guid = guid
        self.text_a = text_a  # Str
        self.text_b = text_b  # Str
        self.label = label  # Str
        self.lang = lang


class InputFeatures(object):
    def __init__(self, input_ids, attention_mask=None, token_type_ids=None, label=None, lang=None):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.label = label  # ID
        self.lang = lang  # ID


def read_examples_from_file(data_file, lang):
    examples = []
    with open(data_file, encoding="utf-8") as f:
        for idx, line in enumerate(f):
            guid = f'{lang}-{idx}'
            text_a, text_b, label = line.strip().split('\t')
            assert isinstance(text_a, str) and isinstance(text_b, str) and isinstance(label, str)
            examples.append(InputExample(guid, text_a, text_b, label, lang))
    return examples


def convert_examples_to_features(
        examples,
        tokenizer,
        max_length,
        label_list=None,
        lang2id=None,
        return_dataset=True
):
    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        inputs = tokenizer.encode_plus(example.text_a, example.text_b, add_special_tokens=True, padding='max_length',
                                       truncation=True, max_length=max_length, return_token_type_ids=True,
                                       return_attention_mask=True, return_overflowing_tokens=False)
        input_ids, attention_mask, token_type_ids = inputs["input_ids"], inputs['attention_mask'], inputs["token_type_ids"]
        lang = lang2id[example.lang]
        label = label_map[example.label]

        features.append(InputFeatures(input_ids, attention_mask, token_type_ids, label, lang))

    if return_dataset:
        all_input_ids = torch.LongTensor([f.input_ids for f in features])
        all_attention_mask = torch.LongTensor([f.attention_mask for f in features])
        all_token_type_ids = torch.LongTensor([f.token_type_ids for f in features])
        all_labels = torch.LongTensor([f.label for f in features])
        all_langs = torch.LongTensor([f.lang for f in features])
        return features, TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_langs, all_labels)
    return features
