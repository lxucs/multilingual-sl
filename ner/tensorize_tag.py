import util
from os.path import join
import os
import pickle
import logging
from tag import read_examples_from_file, convert_examples_to_features

logger = logging.getLogger(__name__)


class TagDataProcessor:
    def __init__(self, config):
        self.config = config
        self.tokenizer = None  # Lazy loading

        self.max_seg_len = config['max_segment_len']

    def get_tokenizer(self):
        if self.tokenizer is None:
            self.tokenizer = util.get_bert_tokenizer(self.config)
        return self.tokenizer

    def get_labels(self, dataset_name):
        path = join(self.config['download_dir'], dataset_name, 'labels.txt')
        with open(path, "r") as f:
            labels = f.read().splitlines()
        labels = [label.strip() for label in labels]
        if "O" not in labels:
            labels = ["O"] + labels
        return labels

    def _get_data(self, dataset_name, partition, lang, data_dir, data_file):
        cache_feature_path = self.get_cache_feature_path(dataset_name, partition, lang)
        cache_dataset_path = self.get_cache_dataset_path(dataset_name, partition, lang)
        if os.path.exists(cache_feature_path) and os.path.exists(cache_dataset_path):
            with open(cache_feature_path, 'rb') as f:
                examples, features = pickle.load(f)
            with open(cache_dataset_path, 'rb') as f:
                dataset = pickle.load(f)
            to_return = (examples, features, dataset)
            logger.info(f'Loaded features and dataset from cache for {partition}-{lang}')
        else:
            logger.info(f'Getting {dataset_name}-{partition}-{lang}; results will be cached')
            examples = read_examples_from_file(join(data_dir, data_file), lang, lang2id=util.lang_to_id[dataset_name])
            features, dataset = convert_examples_to_features(examples, self.get_labels(dataset_name),
                                                             self.max_seg_len, self.get_tokenizer(), self.config,
                                                             pad_token_label_id=util.pad_token_label_id,
                                                             lang=lang, lang2id=util.lang_to_id[dataset_name],
                                                             return_dataset=True)
            with open(cache_feature_path, 'wb') as f:
                pickle.dump((examples, features), f, protocol=4)
            with open(cache_dataset_path, 'wb') as f:
                pickle.dump(dataset, f, protocol=4)
            logger.info('Saved features and dataset to cache')
            to_return = (examples, features, dataset)
        return to_return

    def get_data(self, dataset_name, partition, lang, only_dataset=False):
        assert dataset_name in ['panx']
        cache_dataset_path = self.get_cache_dataset_path(dataset_name, partition, lang)
        if dataset_name == 'panx':
            data_dir = join(self.config['download_dir'], dataset_name)
            data_file = f'{partition}-{lang}.tsv'
        else:
            data_dir, data_file = None, None

        if only_dataset and os.path.exists(cache_dataset_path):
            with open(cache_dataset_path, 'rb') as f:
                dataset = pickle.load(f)
            logger.info(f'Loaded dataset from cache for {partition}-{lang}')
            return dataset

        examples, features, dataset = self._get_data(dataset_name, partition, lang, data_dir, data_file)
        return dataset if only_dataset else (examples, features, dataset)

    def get_cache_feature_path(self, dataset_name, partition, lang):
        cache_dir = join(self.config['data_dir'], 'cache')
        os.makedirs(cache_dir, exist_ok=True)
        model_type = self.config['model_type']
        other = '.w_lang'
        cache_name = f'{dataset_name}.{partition}.{lang}.{self.max_seg_len}.{model_type}{other}'
        cache_path = join(cache_dir, f'{cache_name}.bin')
        return cache_path

    def get_cache_dataset_path(self, dataset_name, partition, lang):
        cache_path = self.get_cache_feature_path(dataset_name, partition, lang)
        cache_path = cache_path[:-4] + '.dataset'
        return cache_path
