import torch
import torch.nn.functional as F
from onir import rankers


@rankers.register('vanilla_transformer')
class VanillaTransformer(rankers.Ranker):
    """
    Implementation of the Vanilla BERT model from:
      > Sean MacAvaney, Andrew Yates, Arman Cohan, and Nazli Goharian. 2019. CEDR: Contextualized
      > Embeddings for Document Ranking. In SIGIR.
    Should be used with a transformer vocab, e.g., BertVocab.
    """

    @staticmethod
    def default_config():
        result = rankers.Ranker.default_config()
        result.update({
            'combine': 'linear', # one of linear, prob
            'outputs': 1,
        })
        return result

    def __init__(self, vocab, config, logger, random):
        super().__init__(config, random)
        self.logger = logger
        self.vocab = vocab
        self.encoder = vocab.encoder()
        assert 'cls' in self.encoder.enc_spec()['joint_fields'] or self._cls_by_query_tok(), \
               "VanillaBert must be used with a vocab that supports CLS encoding (e.g., BertVocab)"
        if self.encoder.static():
            logger.warn("It's usually bad to use VanillaBert with non-trainable embeddings. "
                        "Consider setting `vocab.train=True`")
        self.dropout = torch.nn.Dropout(0.1) # self.encoder.bert.config.hidden_dropout_prob
        if self.config['combine'] == 'linear':
            self.ranker = torch.nn.Linear(self.encoder.dim(), self.config['outputs'])
        elif self.config['combine'] in ('prob', 'logprob'):
            self.ranker = torch.nn.Linear(self.encoder.dim(), 2)
        else:
            raise ValueError('unsupported combine={combine}'.format(**self.config))

    def input_spec(self):
        result = super().input_spec()
        result['fields'].update({'query_tok', 'query_len', 'doc_tok', 'doc_len'})
        result['qlen_mode'] = 'max'
        result['dlen_mode'] = 'max'
        return result

    def path_segment(self):
        result = '{name}_{qlen}q_{dlen}d'.format(name=self.name, **self.config)
        if self.config['combine'] == 'linear':
            if self.config['outputs'] > 1:
                result += '_{combine}-{outputs}'.format(**self.config)
        else:
            result += '_{combine}'.format(**self.config)
        if self.config['add_runscore']:
            result += '_addrun'
        return result

    def _forward(self, **inputs):
        pooled_output = self.encoder.enc_query_doc(**inputs)['cls'][-1]
        pooled_output = self.dropout(pooled_output)
        result = self.ranker(pooled_output)
        if self.config['combine'] == 'prob':
            result = result.softmax(dim=1)[:, 1]
        elif self.config['combine'] == 'logprob':
            result = result.log_softmax(dim=1)[:, 1]
        return result
