import sys
import torch
import torch.nn.functional as F
import onir
from onir import trainers, spec, util


@trainers.register('pairwise-cl')
class PairwiseClTrainer(trainers.Trainer):
    """
    Pairwise curriculum learning trainer from:

    > Sean MacAvaney, Franco Maria Nardini, Raffaele Perego, Nicola Tonellotto, Nazli Goharian, and
    > Ophir Frieder. Training Curricula for Open Domain Answer Re-Ranking. SIGIR 2020.
    """
    @staticmethod
    def default_config():
        result = trainers.Trainer.default_config()
        result.update({
            'lossfn': onir.config.Choices(['softmax', 'cross_entropy', 'nogueira_cross_entropy', 'hinge']),
            'pos_source': onir.config.Choices(['intersect', 'qrels']),
            'neg_source': onir.config.Choices(['run', 'qrels', 'union']),
            'sampling': onir.config.Choices(['query', 'qrel']),
            'pos_minrel': 1,
            'unjudged_rel': 0,
            'num_neg': 1,
            'margin': 0.,
            'curriculum': onir.config.Choices(['reciprank', 'kdescore', 'normscore']),
            'eoc_epoch': 20,
            'anti': False,
        })
        return result

    def __init__(self, config, ranker, logger, train_ds, vocab, random):
        super().__init__(config, ranker, vocab, train_ds, logger, random)
        self.loss_fn = {
            'softmax': self.softmax,
            'nogueira_cross_entropy': self.nogueira_cross_entropy,
            'cross_entropy': self.cross_entropy,
            'hinge': self.hinge
        }[config['lossfn']]
        self.dataset = train_ds
        self.input_spec = ranker.input_spec()
        self.iter_fields = self.input_spec['fields'] | {'runscore'}
        if self.config['curriculum'] == 'reciprank':
            self.iter_fields = self.iter_fields | {'rank'}
        if self.config['curriculum'] == 'kdescore':
            self.iter_fields = self.iter_fields | {'kdescore'}
        if self.config['curriculum'] == 'normscore':
            self.iter_fields = self.iter_fields | {'normscore'}
        self.train_iter_core = onir.datasets.pair_iter(
            train_ds,
            fields=self.iter_fields,
            pos_source=self.config['pos_source'],
            neg_source=self.config['neg_source'],
            sampling=self.config['sampling'],
            pos_minrel=self.config['pos_minrel'],
            unjudged_rel=self.config['unjudged_rel'],
            num_neg=self.config['num_neg'],
            random=self.random,
            inf=True)
        self.train_iter = util.background(self.iter_batches(self.train_iter_core))
        self.numneg = config['num_neg']
        self.notified_end_of_curriculum = False

    def path_segment(self):
        path = super().path_segment()
        pos = 'pos-{pos_source}-{sampling}'.format(**self.config)
        if self.config['pos_minrel'] != 1:
            pos += '-minrel{pos_minrel}'.format(**self.config)
        neg = 'neg-{neg_source}'.format(**self.config)
        if self.config['unjudged_rel'] != 0:
            neg += '-unjudged{unjudged_rel}'.format(**self.config)
        if self.config['num_neg'] != 1:
            neg += '-numneg{num_neg}'.format(**self.config)
        loss = self.config['lossfn']
        if loss == 'hinge':
            loss += '-{margin}'.format(**self.config)
        result = 'pairwise-cl_{path}_{loss}_{pos}_{neg}_{curriculum}-{eoc_epoch}'.format(**self.config, loss=loss, pos=pos, neg=neg, path=path)
        if self.config['anti']:
            result += '-anti'
        if self.config['gpu'] and not self.config['gpu_determ']:
            result += '_nondet'
        return result

    def iter_batches(self, it):
        while True: # breaks on StopIteration
            input_data = {}
            for _, record in zip(range(self.batch_size), it):
                for k, v in record.items():
                    assert len(v) == self.numneg + 1
                    for seq in v:
                        input_data.setdefault(k, []).append(seq)
            input_data = spec.apply_spec_batch(input_data, self.input_spec, self.device)
            yield input_data

    def train_batch(self):
        input_data = next(self.train_iter)
        rel_scores = self.ranker(**input_data)
        if torch.isnan(rel_scores).any() or torch.isinf(rel_scores).any():
            self.logger.error('nan or inf relevance score detected. Aborting.')
            sys.exit(1)
        rel_scores_by_record = rel_scores.reshape(self.batch_size, self.numneg + 1, -1)
        run_scores_by_record = input_data['runscore'].reshape(self.batch_size, self.numneg + 1)
        loss = self.loss_fn(rel_scores_by_record)
        losses = {'data': loss}
        loss_weights = {'data': 1.}

        if self.epoch < self.config['eoc_epoch'] or self.config['eoc_epoch'] <= 0:
            self.notified_end_of_curriculum = False
            weight = None
            if self.config['curriculum'] == 'reciprank':
                weight = (1. / (input_data['rank'].float()))
                weight = weight.reshape(self.batch_size, self.config['num_neg'] + 1)
                w = []
                for i in range(self.config['num_neg']):
                    w.append((weight[:, 0] - weight[:, i+1] + 1.) / 2.)
                weight = torch.stack(w)
                weight, _ = torch.min(weight, dim=0)
                if self.config['anti']:
                    weight = 1. - weight
                weight = weight + 1e-8 # smooth
            elif self.config['curriculum'] == 'normscore':
                weight = input_data['normscore'].reshape(self.batch_size, self.config['num_neg'] + 1)
                w = []
                for i in range(self.config['num_neg']):
                    w.append((weight[:, 0] - weight[:, i+1] + 1.) / 2.)
                weight = torch.stack(w)
                weight, _ = torch.min(weight, dim=0)
                if self.config['anti']:
                    weight = 1. - weight
                weight = weight + 1e-8 # smooth
            elif self.config['curriculum'] == 'kdescore':
                if self.config['num_neg'] > 1:
                    raise ValueError('num_neg > 1 not yet supported for kedscore')
                weight = input_data['kdescore']
                weight = weight.reshape(self.batch_size, self.config['num_neg'] + 1)
                weight = (weight[:, 0] - weight[:, 1] + 1.) / 2.
                if self.config['anti']:
                    weight = 1. - weight
                weight = weight + 1e-8 # smooth
            else:
                raise ValueError('unknown curriculum')
            if self.config['eoc_epoch'] > 0:
                progress = self.epoch / self.config['eoc_epoch']
            else:
                progress = 0.
            weight = weight * (1 - progress) + progress
            losses['data'] = losses['data'] * weight.mean()
        elif not self.notified_end_of_curriculum:
            self.logger.info('end of curriculum')
            self.notified_end_of_curriculum = True

        return {
            'losses': losses,
            'loss_weights': loss_weights,
            'acc': self.acc(rel_scores_by_record),
            'unsup_acc': self.acc(run_scores_by_record)
        }

    def fast_forward(self, record_count):
        self._fast_forward(self.train_iter_core, self.iter_fields, record_count)

    @staticmethod
    def cross_entropy(rel_scores_by_record):
        target = torch.zeros(rel_scores_by_record.shape[0]).long().to(rel_scores_by_record.device)
        return F.cross_entropy(rel_scores_by_record, target, reduction='mean')

    @staticmethod
    def softmax(rel_scores_by_record):
        return torch.mean(1. - F.softmax(rel_scores_by_record, dim=1)[:, 0])

    def hinge(self, rel_scores_by_record):
        return F.relu(self.config['margin'] - rel_scores_by_record[:, :1] + rel_scores_by_record[:, 1:]).mean()

    @staticmethod
    def nogueira_cross_entropy(rel_scores_by_record):
        """
        cross entropy loss formulation for BERT from:
         > Rodrigo Nogueira and Kyunghyun Cho. 2019.Passage re-ranking with bert. ArXiv,
         > abs/1901.04085.
        """
        log_probs = -rel_scores_by_record.log_softmax(dim=2)
        return (log_probs[:, 0, 0] + log_probs[:, 1, 1]).mean()

    @staticmethod
    def acc(scores_by_record):
        count = scores_by_record.shape[0] * (scores_by_record.shape[1] - 1)
        return (scores_by_record[:, :1] > scores_by_record[:, 1:]).sum().float() / count
