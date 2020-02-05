import torch
from torch import nn
from onir import rankers, modules
from onir.vocab import wordvec_vocab


@rankers.register('conv_knrm')
class ConvKnrm(rankers.Ranker):
    """
    Implementation of the ConvKNRM model from:
      > Zhuyun Dai, Chenyan Xiong, Jamie Callan, and Zhiyuan Liu. 2018. Convolutional Neural
      > Networks for Soft-Matching N-Grams in Ad-hoc Search. In WSDM.
    """

    @staticmethod
    def default_config():
        result = rankers.Ranker.default_config()
        result.update({
            'mus': '-0.9,-0.7,-0.5,-0.3,-0.1,0.1,0.3,0.5,0.7,0.9,1.0',
            'sigmas': '0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.001',
            'grad_kernels': True,
            'max_ngram': 3,
            'crossmatch': True,
            'conv_filters': 128,
            'combine_channels': False,
            'pretrained_kernels': False,
        })
        return result

    def __init__(self, vocab, config, logger, random):
        super().__init__(config, random)
        self.logger = logger
        self.embed = vocab.encoder()
        self.simmat = modules.InteractionMatrix()
        self.padding, self.convs = nn.ModuleList(), nn.ModuleList()
        for conv_size in range(1, config['max_ngram'] + 1):
            if conv_size > 1:
                self.padding.append(nn.ConstantPad1d((0, conv_size-1), 0))
            else:
                self.padding.append(nn.Sequential()) # identity
            self.convs.append(nn.ModuleList())
            if self.config['combine_channels']:
                self.convs[-1].append(nn.Conv1d(self.embed.dim() * self.embed.emb_views(), config['conv_filters'], conv_size))
            else:
                for _ in range(self.embed.emb_views()):
                    self.convs[-1].append(nn.Conv1d(self.embed.dim(), config['conv_filters'], conv_size))
        if self.config['pretrained_kernels']:
            kernels = wordvec_vocab._SOURCES[vocab.config['source']][vocab.config['variant']](logger, get_kernels=True)
            for conv, weight, bias in zip(self.convs, *kernels):
                conv[0].weight.data = torch.from_numpy(weight).float()
                conv[0].bias.data = torch.from_numpy(bias).float()
        self.kernels = modules.RbfKernelBank.from_strs(config['mus'], config['sigmas'], dim=1, requires_grad=config['grad_kernels'])
        channels = config['max_ngram'] ** 2 if config['crossmatch'] else config['max_ngram']
        if not self.config['combine_channels']:
            channels *= self.embed.emb_views() ** 2 if config['crossmatch'] else self.embed.emb_views()
        self.combine = nn.Linear(self.kernels.count() * channels, 1)

    def input_spec(self):
        result = super().input_spec()
        result['fields'].update({'query_tok', 'doc_tok', 'query_len', 'doc_len'})
        # TODO: possible changes to qlen_mode and dlen_mode
        return result

    def _forward(self, **inputs):
        # BATCH, QLEN, DELN, EMB = *inputs['query_tok'].shape, inputs['doc_tok'].shape[1], self.embed.size
        enc = self.embed.enc_query_doc(**inputs)
        a_embed, b_embed = enc['query'], enc['doc']
        if isinstance(a_embed, list):
            a_embed = torch.stack(a_embed, dim=2)
            b_embed = torch.stack(b_embed, dim=2)
        a_reps, b_reps = [], []
        if self.config['combine_channels']:
            a_embed = a_embed.reshape(a_embed.shape[0], a_embed.shape[1], -1)
            b_embed = b_embed.reshape(b_embed.shape[0], b_embed.shape[1], -1)
            for pad, conv in zip(self.padding, self.convs):
                a_reps.append(conv[0](pad(a_embed.permute(0, 2, 1))).permute(0, 2, 1))
                b_reps.append(conv[0](pad(b_embed.permute(0, 2, 1))).permute(0, 2, 1))
        else:
            if len(a_embed.shape) == 4:
                a_embed = [a_embed[:, i, :, :] for i in range(a_embed.shape[1])]
            else:
                a_embed = [a_embed]
            if len(b_embed.shape) == 4:
                b_embed = [b_embed[:, i, :, :] for i in range(b_embed.shape[1])]
            else:
                b_embed = [b_embed]
            for layer, (a_emb, b_emb) in enumerate(zip(a_embed, b_embed)):
                for pad, conv in zip(self.padding, self.convs):
                    a_reps.append(conv[layer](pad(a_emb.permute(0, 2, 1))).permute(0, 2, 1))
                    b_reps.append(conv[layer](pad(b_emb.permute(0, 2, 1))).permute(0, 2, 1))
        simmats = []
        if self.config['crossmatch']:
            for a_rep in a_reps:
                for b_rep in b_reps:
                    simmats.append(self.simmat(a_rep, b_rep, inputs['query_tok'], inputs['doc_tok']))
        else:
            for a_rep, b_rep in zip(a_reps, b_reps):
                simmats.append(self.simmat(a_rep, b_rep, inputs['query_tok'], inputs['doc_tok']))
        simmats = torch.cat(simmats, dim=1)
        kernels = self.kernels(simmats)
        BATCH, KERNELS, VIEWS, QLEN, DLEN = kernels.shape
        kernels = kernels.reshape(BATCH, KERNELS * VIEWS, QLEN, DLEN)
        simmats = simmats.reshape(BATCH, 1, VIEWS, QLEN, DLEN) \
                         .expand(BATCH, KERNELS, VIEWS, QLEN, DLEN) \
                         .reshape(BATCH, KERNELS * VIEWS, QLEN, DLEN)
        result = kernels.sum(dim=3) # sum over document
        mask = (simmats.sum(dim=3) != 0.) # which query terms are not padding?
        result = torch.where(mask, (result + 1e-6).log(), mask.float())
        result = result.sum(dim=2) # sum over query terms
        result = self.combine(result)
        return result

    def path_segment(self):
        result = '{name}_{qlen}q_{dlen}d_ng{max_ngram}_conv{conv_filters}'.format(name=self.name, **self.config)
        if not self.config['grad_kernels']:
            result += '_nogradkernels'
        if not self.config['crossmatch']:
            result += '_nocrossmatch'
        if self.config['combine_channels']:
            result += '_combinechannels'
        if self.config['pretrained_kernels']:
            result += '_pretrainedkernels'
        if self.config['add_runscore']:
            result += '_addrun'
        return result
