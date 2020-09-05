import sys
import torch
import torch.nn.functional as F
import onir
from onir import trainers, spec, datasets


@trainers.register('pointwise-cl')
class PointwiseClTrainer(trainers.Trainer):
    """
    Pointwise curriculum learning trainer from:

    > Sean MacAvaney, Franco Maria Nardini, Raffaele Perego, Nicola Tonellotto, Nazli Goharian, and
    > Ophir Frieder. Training Curricula for Open Domain Answer Re-Ranking. SIGIR 2020.
    """
    @staticmethod
    def default_config():
        result = trainers.Trainer.default_config()
        result.update({
            'source': 'run',
            'lossfn': 'mse',
            'minrel': -999,
            'curriculum': onir.config.Choices(['reciprank', 'kdescore', 'normscore']),
            'eoc_epoch': 20,
            'anti': False,
        })
        return result

    def __init__(self, config, ranker, vocab, logger, train_ds, random):
        super().__init__(config, ranker, vocab, train_ds, logger, random)
        self.dataset = train_ds
        self.input_spec = ranker.input_spec()
        self.iter_fields = self.input_spec['fields'] | {'relscore'}
        if config['curriculum'] == 'reciprank':
            self.iter_fields = self.iter_fields | {'rank'}
        if self.config['curriculum'] == 'kdescore':
            self.iter_fields = self.iter_fields | {'kdescore'}
        if self.config['curriculum'] == 'normscore':
            self.iter_fields = self.iter_fields | {'normscore'}
        self.train_iter_core = datasets.record_iter(train_ds,
                                                    fields=self.iter_fields,
                                                    source=self.config['source'],
                                                    minrel=None if self.config['minrel'] == -999 else self.config['minrel'],
                                                    shuf=True,
                                                    random=self.random,
                                                    inf=True)
        self.train_iter = self.iter_batches(self.train_iter_core)

    def path_segment(self):
        path = super().path_segment()
        result = 'pointwise-cl_{path}_{lossfn}_{curriculum}-{eoc_epoch}'.format(**self.config, path=path)
        if self.config['anti']:
            result += '_anti'
        if self.config['source'] != 'qrels':
            result += '_' + self.config['source']
        if self.config['minrel'] != -999:
            result += '_minrel-{minrel}'.format(**self.config)
        if self.config['gpu'] and not self.config['gpu_determ']:
            result += '_nondet'
        return result

    def iter_batches(self, it):
        while True: # breaks on StopIteration
            input_data = {}
            for _, record in zip(range(self.batch_size), it):
                for k, seq in record.items():
                    input_data.setdefault(k, []).append(seq)
            input_data = spec.apply_spec_batch(input_data, self.input_spec, self.device)
            yield input_data

    def train_batch(self):
        input_data = next(self.train_iter)
        rel_scores = self.ranker(**input_data)
        if torch.isnan(rel_scores).any() or torch.isinf(rel_scores).any():
            self.logger.error('nan or inf relevance score detected. Aborting.')
            sys.exit(1)
        target_relscores = input_data['relscore'].float()
        target_relscores[target_relscores == -999.] = 0. # replace -999 with non-relevant score
        if self.config["lossfn"] == 'mse':
            loss = F.mse_loss(rel_scores.flatten(), target_relscores)
        elif self.config["lossfn"] == 'l1':
            loss = F.l1_loss(rel_scores.flatten(), target_relscores)
        elif self.config["lossfn"] == 'smoothl1':
            loss = F.smooth_l1_loss(rel_scores.flatten(), target_relscores)
        elif self.config['lossfn'] == 'cross_entropy':
            loss = -torch.where(target_relscores > 0, rel_scores.flatten(), 1 - rel_scores.flatten()).log()
            loss = loss.mean()
        elif self.config['lossfn'] == 'cross_entropy_logits':
            assert len(rel_scores.shape) == 2
            assert rel_scores.shape[1] == 2
            log_probs = -rel_scores.log_softmax(dim=1)
            one_hot = torch.tensor([[1., 0.] if tar > 0 else [0., 1.] for tar in target_relscores], device=rel_scores.device)
            loss = (log_probs * one_hot).sum(dim=1).mean()
        elif self.config['lossfn'] == 'softmax':
            assert len(rel_scores.shape) == 2
            assert rel_scores.shape[1] == 2
            probs = rel_scores.softmax(dim=1)
            one_hot = torch.tensor([[0., 1.] if tar > 0 else [1., 0.] for tar in target_relscores], device=rel_scores.device)
            loss = (probs * one_hot).sum(dim=1).mean()
        else:
            raise ValueError(f'unknown lossfn `{self.config["lossfn"]}`')
        losses = {'data': loss}
        loss_weights = {'data': 1.}
        if self.epoch < self.config['eoc_epoch']:
            weight = None
            if self.config['curriculum'] == 'reciprank':
                is_pos = (input_data['relscore'] > 0)
                weight = (1. / (input_data['rank'].float()))
                if self.config['anti']:
                    weight = 1. - weight
                weight = torch.where(is_pos, weight, 1. - weight)
                weight = weight + 1e-8 # smooth
            elif self.config['curriculum'] == 'normscore':
                is_pos = (input_data['relscore'] > 0)
                weight = input_data['normscore'].float()
                if self.config['anti']:
                    weight = 1. - weight
                weight = torch.where(is_pos, weight, 1. - weight)
                weight = weight + 1e-8 # smooth
            elif self.config['curriculum'] == 'kdescore':
                is_pos = (input_data['relscore'] > 0)
                weight = input_data['kdescore']
                if self.config['anti']:
                    weight = 1. - weight
                weight = torch.where(is_pos, weight, 1. - weight)
                weight = weight + 1e-8 # smooth
            else:
                raise ValueError('unknown curriculum')
            progress = self.epoch / self.config['eoc_epoch']
            weight = weight * (1 - progress) + progress
            losses['data'] = losses['data'] * weight.mean()
        return {
            'losses': losses,
            'loss_weights': loss_weights,
        }

    def fast_forward(self, record_count):
        self._fast_forward(self.train_iter_core, self.iter_fields, record_count)
