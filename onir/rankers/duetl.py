import torch
from torch import nn
from onir import rankers, modules


@rankers.register('duetl')
class DuetL(rankers.Ranker):
    """
    Implementation of the local variant of the Duet model from:
      > Bhaskar Mitra, Fernando Diaz, and Nick Craswell. 2016. Learning to Match using Local and
      > Distributed Representations of Text for Web Search. In WWW.
    """

    @staticmethod
    def default_config():
        result = rankers.Ranker.default_config()
        result.update({
            'nfilters': 300
        })
        return result

    def __init__(self, vocab, config, logger, random):
        super().__init__(config, random)
        self.logger = logger
        self.encoder = vocab.encoder()
        self.simmat = modules.InteractionMatrix()
        self.conv = nn.Conv2d(1, config['nfilters'], (1, config['dlen']))
        self.combine1 = nn.Linear(config['qlen'] * config['nfilters'], 300)
        self.combine2 = nn.Linear(300, 300)
        self.dropout = nn.Dropout(0.2)
        self.combine3 = nn.Linear(300, 1)

    def input_spec(self):
        result = super().input_spec()
        result['fields'].update({'query_tok', 'doc_tok', 'query_len', 'doc_len'})
        # TODO: possible changes to qlen_mode and dlen_mode
        return result

    def _forward(self, **inputs):
        BATCH = inputs['query_tok'].shape[0]
        result = self.simmat.encode_query_doc(self.encoder, **inputs)
        result = result.reshape(BATCH, 1, self.config['qlen'], self.config['dlen'])
        result = torch.tanh(self.conv(result))
        result = result.reshape(BATCH, self.config['qlen'] * self.config['nfilters'])
        result = torch.tanh(self.combine1(result))
        result = torch.tanh(self.combine2(result))
        result = self.dropout(result)
        result = torch.tanh(self.combine3(result))
        return result

    def path_segment(self):
        result = '{name}_{qlen}q_{dlen}d_{nfilters}'.format(name=self.name, **self.config)
        if self.config['add_runscore']:
            result += '_addrun'
        return result
