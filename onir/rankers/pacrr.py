import torch
from torch import nn
from torch.nn import functional as F
from onir import rankers, util, modules


@rankers.register('pacrr')
class Pacrr(rankers.Ranker):
    """
    Implementation of the PACRR model from:
      > Kai Hui, Andrew Yates, Klaus Berberich, and Gerard de Melo. 2017. PACRR: A Position-Aware
      > Neural IR Model for Relevance Matching. In EMNLP.

    Some features included from CO-PACRR (e.g., shuf):
      > Kai Hui, Andrew Yates, Klaus Berberich, and Gerard de Melo. 2018. Co-PACRR: A Context-Aware
      > Neural IR Model for Ad-hoc Retrieval. In WSDM.
    """

    @staticmethod
    def default_config():
        result = rankers.Ranker.default_config()
        result.update({
            'mingram': 1,
            'maxgram': 3,
            'nfilters': 32,
            'idf': True,
            'kmax': 2,
            'shuf': False,
            'combine': 'dense32',
        })
        return result

    def __init__(self, vocab, config, logger, random):
        super().__init__(config, random)
        self.config = config
        self.logger = logger
        self.encoder = vocab.encoder()
        self.simmat = modules.InteractionMatrix()
        channels = self.encoder.emb_views()
        self.ngrams = nn.ModuleList()
        for ng in range(config['mingram'], config['maxgram']+1):
            self.ngrams.append(ConvMax2d(ng, config['nfilters'], k=config['kmax'], channels=channels))
        if config['combine'] == 'dense32':
            num_feats = len(self.ngrams)*config['kmax'] + (1 if config['idf'] else 0)
            self.combination = DenseCombination([config['qlen'] * num_feats, 32, 32, 1], config['shuf'])
        elif config['combine'] == 'sum':
            self.combination = SumCombination(len(self.ngrams)*config['kmax'] + (1 if config['idf'] else 0))
        elif config['combine'] == 'sumnorm':
            self.combination = SumCombination(len(self.ngrams)*config['kmax'] + (1 if config['idf'] else 0), False)
        else:
            raise ValueError('unknown combine `{combine}`'.format(**config))
        self.path = util.path_model(self)

    def input_spec(self):
        result = super().input_spec()
        result['fields'].update({'query_tok', 'doc_tok', 'query_len', 'doc_len'})
        if self.config['idf']:
            result['fields'].add('query_idf')
        result['dlen_mode'] = 'max' # max pooling means you don't need a fixed dlen
        result['dlen_min'] = self.config['kmax']
        return result

    def _forward(self, **inputs):
        simmat = self.simmat.encode_query_doc(self.encoder, **inputs)
        conv_pool = self.conv_pool(simmat, inputs)
        return self.combination(conv_pool, inputs['query_len'])

    def conv_pool(self, simmat, inputs):
        scores = [ng(simmat, inputs['query_len']) for ng in self.ngrams]
        if self.config['idf']:
            idfs = inputs['query_idf'].reshape(*inputs['query_idf'].shape, 1).softmax(dim=1)
            scores.append(idfs)
        return torch.cat(scores, dim=2)

    def path_segment(self):
        result = '{name}_{qlen}q_{dlen}d_ng-{mingram}-{maxgram}_k-{kmax}'.format(name=self.name, **self.config)
        if self.config['nfilters'] != 32:
            result += '_filters-{nfilters}'.format(**self.config)
        if self.config['combine'] != 'dense32':
            result += '_{combine}'.format(**self.config)
        if self.config['idf']:
            result += '_idf'
        if self.config['shuf']:
            result += '_shuf'
        if self.config['add_runscore']:
            result += '_addrun'
        return result


class ConvMax2d(nn.Module):

    def __init__(self, shape, n_filters, k, channels=1):
        super(ConvMax2d, self).__init__()
        self.shape = shape
        if shape != 1:
            self.pad = nn.ConstantPad2d((0, shape-1, 0, shape-1), 0)
        else:
            self.pad = None
        self.conv = nn.Conv2d(channels, n_filters, shape)
        self.activation = nn.ReLU()
        self.k = k
        self.shape = shape
        self.channels = channels

    def forward(self, simmat, query_len):
        assert len(simmat.size()) == 4, \
            f"incorrect shape (expected (BATCH, CHANNELS, QLEN, DLEN), got {tuple(simmat.size())})"
        batch, channels, qlen, dlen = tuple(simmat.size())
        assert channels == self.channels, 'expected {} channels; got {}'.format(self.channels, channels)
        if self.pad:
            simmat = self.pad(simmat)
        conv = self.activation(self.conv(simmat))
        top_filters, _ = conv.max(dim=1)
        top_docs, _ = top_filters.topk(self.k, dim=2)
        result = top_docs.reshape(batch, qlen, self.k)
        return result


class DenseCombination(nn.Module):

    def __init__(self, layers, shuf):
        super().__init__()
        self.layers = layers
        self.hidden = []
        self.shuf = shuf
        for din, dout in zip(layers[:-1], layers[1:]):
            self.hidden.append(torch.nn.Linear(din, dout))
        self.hidden = nn.ModuleList(self.hidden)

    def forward(self, x, *args):
        if self.shuf:
            permute = torch.randperm(x.shape[1]).to(x.device)
            x = x[:, permute]
        x = x.reshape(-1, self.layers[0])
        for i, hidden in enumerate(self.hidden):
            if i != 0:
                x = F.relu(x)
            x = hidden(x)
        return x


class SumCombination(nn.Module):

    def __init__(self, dim_in, normalize=True):
        super(SumCombination, self).__init__()
        self.conv = nn.Conv1d(dim_in, 1, 1)
        self.normalize = normalize

    def forward(self, x, qlen):
        scores = self.conv(x.permute(0, 2, 1))[:, :, 0]
        if self.normalize:
            scores = scores.sum(dim=1) / qlen.type_as(scores)
        return scores
