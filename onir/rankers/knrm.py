import torch
from torch import nn
from onir import rankers, modules


@rankers.register('knrm')
class Knrm(rankers.Ranker):
    """
    Implementation of the K-NRM model from:
      > Chenyan Xiong, Zhuyun Dai, James P. Callan, Zhiyuan Liu, and Russell Power. 2017. End-to-End
      > Neural Ad-hoc Ranking with Kernel Pooling. In SIGIR.
    """

    @staticmethod
    def default_config():
        result = rankers.Ranker.default_config()
        result.update({
            'mus': '-0.9,-0.7,-0.5,-0.3,-0.1,0.1,0.3,0.5,0.7,0.9,1.0',
            'sigmas': '0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.001',
            'grad_kernels': True
        })
        return result

    def __init__(self, vocab, config, logger, random):
        super().__init__(config, random)
        self.logger = logger
        self.encoder = vocab.encoder()
        self.simmat = modules.InteractionMatrix()
        self.kernels = modules.RbfKernelBank.from_strs(config['mus'], config['sigmas'], dim=1, requires_grad=config['grad_kernels'])
        self.combine = nn.Linear(self.kernels.count() * self.encoder.emb_views(), 1)

    def input_spec(self):
        result = super().input_spec()
        result['fields'].update({'query_tok', 'doc_tok', 'query_len', 'doc_len'})
        # combination does not enforce strict lengths for doc or query
        result['qlen_mode'] = 'max'
        result['dlen_mode'] = 'max'
        return result

    def _forward(self, **inputs):
        simmat = self.simmat.encode_query_doc(self.encoder, **inputs)
        kernel_scores = self.kernel_pool(simmat)
        result = self.combine(kernel_scores) # linear combination over kernels
        return result

    def kernel_pool(self, simmat):
        kernels = self.kernels(simmat)
        BATCH, KERNELS, VIEWS, QLEN, DLEN = kernels.shape
        kernels = kernels.reshape(BATCH, KERNELS * VIEWS, QLEN, DLEN)
        result = kernels.sum(dim=3) # sum over document
        simmat = simmat.reshape(BATCH, 1, VIEWS, QLEN, DLEN) \
                       .expand(BATCH, KERNELS, VIEWS, QLEN, DLEN) \
                       .reshape(BATCH, KERNELS * VIEWS, QLEN, DLEN)
        mask = (simmat.sum(dim=3) != 0.) # which query terms are not padding?
        result = torch.where(mask, (result + 1e-6).log(), mask.float())
        result = result.sum(dim=2) # sum over query terms
        return result

    def path_segment(self):
        result = '{name}_{qlen}q_{dlen}d'.format(name=self.name, **self.config)
        if not self.config['grad_kernels']:
            result += '_gradkernels'
        if self.config['add_runscore']:
            result += '_addrun'
        return result
