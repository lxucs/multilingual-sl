from transformers.models.bert.modeling_bert import BertModel
from transformers.models.xlm_roberta.modeling_xlm_roberta import XLMRobertaModel
from transformers.models.mt5.modeling_mt5 import MT5EncoderModel
from torch import nn
import torch.nn.functional as F
import logging
import torch
import torch.nn.init as init
import util
import math
import numpy as np
from numpy.random import default_rng

logger = logging.getLogger(__name__)


def get_seq_encoder(config):
    if config['model_type'] == 'bert':
        return BertModel.from_pretrained(config['pretrained'])
    elif config['model_type'] == 'xlmr':
        return XLMRobertaModel.from_pretrained(config['pretrained'])
    elif config['model_type'] == 'mt5':
        return MT5EncoderModel.from_pretrained(config['pretrained'])
    else:
        raise ValueError(config['model_type'])


class TransformerTag(nn.Module):
    rng = default_rng()

    def __init__(self, config, num_labels):
        super().__init__()
        assert not config['evi_un'] or not any([config['lang_un'], config['tag_un']])

        self.config = config
        self.num_labels = num_labels

        self.seq_encoder = get_seq_encoder(config)
        self.seq_config = self.seq_encoder.config
        self.seq_hidden_size = self.seq_config.hidden_size
        if config['dim_reduce']:
            self.seq_hidden_size = self.seq_config.hidden_size // config['dim_reduce']
            self.tag_dim_reduce = self.make_linear(self.seq_config.hidden_size, self.seq_hidden_size, bias=False)

        self.dropout = nn.Dropout(config['dropout_rate'])
        self.tag_outputs = nn.Linear(self.seq_hidden_size, num_labels)

        self.tag_un_output = nn.Linear(self.seq_hidden_size, 1)
        self.mc = 20

        self.emb_lang_un = self.make_emb(len(util.lang_to_id[config['dataset_name']]), 1)

    def make_emb(self, dict_size, dim_size, std=None):
        emb = nn.Embedding(dict_size, dim_size)
        if std:
            init.normal_(emb.weight, std=std)
        return emb

    def make_linear(self, in_features, out_features, bias=True, std=None):
        linear = nn.Linear(in_features, out_features, bias)
        if std:
            init.normal_(linear.weight, std=std)
        if bias:
            init.zeros_(linear.bias)
        return linear

    def freeze_emb(self):
        for param in self.seq_encoder.embeddings.parameters():
            param.requires_grad = False
        logger.info('Froze encoder embedding')

    @classmethod
    def get_probs(cls, logits, tag_un=None, as_regression=False, mc=20, evi_un=False):
        """ Per instance, after masked """
        # Evidential un
        if evi_un:
            evidence = logits
            alpha = evidence + 1
            S = alpha.sum(axis=-1, keepdims=True)
            probs = alpha / S
            return probs
        # No un
        if tag_un is None:
            return util.compute_softmax(logits)
        # Un as softmax temperature
        if as_regression:
            logits *= np.exp(-np.expand_dims(tag_un, axis=-1))
            return util.compute_softmax(logits)
        # Un as Gaussian noise on logits
        tag_std = np.exp(tag_un).reshape(tag_un.shape + (1, 1))
        gaussian_samples = cls.rng.standard_normal((logits.shape[0], mc, logits.shape[-1]))
        tag_std = tag_std * gaussian_samples
        logits_sampled = np.expand_dims(logits, axis=-2) + tag_std
        probs = util.compute_softmax(logits_sampled).mean(axis=1, keepdims=False)
        return probs

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, lang_ids=None, labels=None,
                output_hidden=False):
        conf, batch_size, seq_len = self.config, input_ids.shape[0], input_ids.shape[1]

        inputs = {'input_ids': input_ids, 'attention_mask': attention_mask,
                  'output_attentions': False, 'output_hidden_states': False, 'return_dict': False}
        if conf['model_type'] not in ['mt5']:
            inputs['token_type_ids'] = token_type_ids
        outputs = self.seq_encoder(**inputs)
        sequence_output = outputs[0]
        if conf['dim_reduce']:
            sequence_output = self.tag_dim_reduce(self.dropout(sequence_output))

        logits = self.tag_outputs(self.dropout(sequence_output))
        logits = (F.elu(logits) + 1) if conf['evi_un'] else logits

        tag_un = self.tag_un_output(sequence_output).squeeze(-1) if conf['tag_un'] else None

        # Get loss
        loss = None
        if labels is not None:
            # Heteroscedastic un per input token
            if conf['tag_un'] and not conf['tag_un_as_regression']:  # Kendall'17
                tag_std = torch.exp(tag_un).unsqueeze(-1).unsqueeze(-1)
                gaussian_samples = torch.normal(0, 1, size=(logits.size()[:-1] + (self.mc,) + logits.size()[-1:])).to(tag_std.device)
                gaussian_samples = tag_std * gaussian_samples
                logits_sampled = logits.unsqueeze(-2) + gaussian_samples

                nll_fct = nn.CrossEntropyLoss(reduction='none')
                nll = nll_fct(logits_sampled.view(-1, logits_sampled.shape[-1]),
                              labels.unsqueeze(-1).repeat(1, 1, self.mc).view(-1)).view(
                    logits_sampled.size()[:-1])
                nll_sampled = -torch.logsumexp(-nll, dim=-1, keepdim=False) + math.log(self.mc)
                loss = nll_sampled
            elif conf['tag_un']:  # Kendall'18; similar to lang_un
                loss_fct = nn.CrossEntropyLoss(reduction='none')
                loss = loss_fct(logits.view(-1, logits.shape[-1]), labels.view(-1)).view(batch_size, -1) 
                loss = loss * torch.exp(-tag_un) + tag_un / 2

            # Homoscedastic un per lang/task
            if conf['lang_un']:
                if loss is None:
                    loss_fct = nn.CrossEntropyLoss(reduction='none')
                    loss = loss_fct(logits.view(-1, logits.shape[-1]), labels.view(-1)).view(batch_size, -1)
                lang_un = self.emb_lang_un(lang_ids)
                loss = loss * torch.exp(-lang_un) + lang_un / 2 

            # Evidential un
            if conf['evi_un']:
                evidence = logits
                alpha = evidence + 1
                S = alpha.sum(dim=-1, keepdim=True)
                probs = alpha / S
                onehot_labels = F.one_hot(torch.where(labels >= 0, labels, 0), logits.shape[-1])
                loss_mse = (onehot_labels - probs).pow(2)
                loss_var = probs * (1 - probs) / (S + 1)
                loss = (loss_mse + loss_var).sum(dim=-1, keepdim=False)

            if loss is None:  # Normal loss without un
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, logits.shape[-1]), labels.view(-1))
            else:  # Loss with un
                loss = loss[labels != util.pad_token_label_id].mean()  # Padding should have ignore_index label

        total_output = (logits, tag_un, sequence_output) if output_hidden else (logits, tag_un)
        return total_output if loss is None else (loss, total_output)
