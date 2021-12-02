import util
from os.path import join
import os
import pickle
import logging
from xnli import read_examples_from_file, convert_examples_to_features

logger = logging.getLogger(__name__)


class XnliDataProcessor:
    def __init__(self, config):
        self.config = config
        self.tokenizer = None  # Lazy loading

        self.max_seg_len = config['max_segment_len']
        self.dataset_name = 'xnli'
        self.labels = ["contradiction", "entailment", "neutral"]

    def get_tokenizer(self):
        if self.tokenizer is None:
            self.tokenizer = util.get_bert_tokenizer(self.config)
        return self.tokenizer

    def get_labels(self):
        return self.labels

    def _get_data(self, partition, lang, data_file):
        cache_feature_path = self.get_cache_feature_path(partition, lang)
        cache_dataset_path = self.get_cache_dataset_path(partition, lang)
        if os.path.exists(cache_feature_path) and os.path.exists(cache_dataset_path):
            with open(cache_feature_path, 'rb') as f:
                examples, features = pickle.load(f)
            with open(cache_dataset_path, 'rb') as f:
                dataset = pickle.load(f)
            to_return = (examples, features, dataset)
            logger.info(f'Loaded features and dataset from cache for {partition}-{lang}')
        else:
            logger.info(f'Getting {partition}-{lang}; results will be cached')
            examples = read_examples_from_file(data_file, lang)
            features, dataset = convert_examples_to_features(examples, self.get_tokenizer(), self.max_seg_len,
                                                             self.get_labels(), util.lang_to_id, return_dataset=True)
            with open(cache_feature_path, 'wb') as f:
                pickle.dump((examples, features), f, protocol=4)
            with open(cache_dataset_path, 'wb') as f:
                pickle.dump(dataset, f, protocol=4)
            logger.info('Saved features and dataset to cache')
            to_return = (examples, features, dataset)
        return to_return

    def get_data(self, partition, lang, only_dataset=False):
        cache_dataset_path = self.get_cache_dataset_path(partition, lang)
        data_file = join(self.config['download_dir'], self.dataset_name, f'{partition}-{lang}.tsv')

        if only_dataset and os.path.exists(cache_dataset_path):
            with open(cache_dataset_path, 'rb') as f:
                dataset = pickle.load(f)
            logger.info(f'Loaded dataset from cache for {partition}-{lang}')
            return dataset

        examples, features, dataset = self._get_data(partition, lang, data_file)
        return dataset if only_dataset else (examples, features, dataset)

    def get_cache_feature_path(self, partition, lang):
        cache_dir = join(self.config['data_dir'], 'cache')
        os.makedirs(cache_dir, exist_ok=True)
        model_type = self.config['model_type']
        cache_name = f'{partition}.{lang}.{self.max_seg_len}.{model_type}'
        cache_path = join(cache_dir, f'{cache_name}.bin')
        return cache_path

    def get_cache_dataset_path(self, partition, lang):
        cache_path = self.get_cache_feature_path(partition, lang)
        cache_path = cache_path[:-4] + '.dataset'
        return cache_path
