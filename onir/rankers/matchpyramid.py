import torch
from torch import nn
from torch.nn import functional as F
from onir import rankers, util, modules


@rankers.register('matchpyramid')
class MatchPyramid(rankers.Ranker):
    """
    Implementation of the MatchPyramid model for ranking from:
      > Liang Pang, Yanyan Lan, Jiafeng Guo, Jun Xu, and Xueqi Cheng. 2016. A Study of MatchPyramid
      > Models on Ad-hoc Retrieval. In NeuIR @ SIGIR.
    """

    @staticmethod
    def default_config():
        result = rankers.Ranker.default_config()
        result.update({
            'nfilters': 8,
            'combine': 'dense128',
            'pool_q': 4,
            'pool_d': 10,
            'conv_q': 3,
            'conv_d': 3
        })
        return result

    def __init__(self, vocab, config, logger, random):
        super().__init__(config, random)
        self.config = config
        self.logger = logger
        self.encoder = vocab.encoder()
        self.simmat = modules.InteractionMatrix()
        self.conv = nn.Conv2d(self.encoder.emb_views(), config['nfilters'], (config['conv_q'], config['conv_d']))
        self.pool = nn.MaxPool2d((config['pool_q'], config['pool_d']))
        if config['combine'] == 'dense128':
            self.combine1 = nn.Linear((config['qlen']//config['pool_q'])*(config['dlen']//config['pool_d']-1)*config['nfilters'], 128)
            self.combine2 = nn.Linear(128, 1)

    def input_spec(self):
        result = super().input_spec()
        result['fields'].update({'query_tok', 'doc_tok', 'query_len', 'doc_len'})
        return result

    def _forward(self, **inputs):
        BATCH, QLEN, DLEN = *inputs['query_tok'].shape, inputs['doc_tok'].shape[1]
        simmat = self.simmat.encode_query_doc(self.encoder, **inputs)
        scores = F.relu(self.conv(simmat.reshape(BATCH, 1, QLEN, DLEN)))
        scores = self.pool(scores)
        scores = F.relu(self.combine1(scores.reshape(BATCH, -1)))
        scores = self.combine2(scores)
        return scores

    def path_segment(self):
        result = '{name}_{qlen}q_{dlen}d_conv-{conv_q}-{conv_d}_pool-{pool_q}-{pool_d}_filters-{nfilters}_{combine}'.format(name=self.name, **self.config)
        if self.config['add_runscore']:
            result += '_addrun'
        return result
