from tqdm import tqdm
import torch
from onir import util, trainers
from onir.interfaces import apex


class Trainer:
    name = None

    @staticmethod
    def default_config():
        return {
            'batch_size': 16,
            'batches_per_epoch': 32,
            'grad_acc_batch': 0,
            'optimizer': 'adam',
            'lr': 0.001,
            'gpu': True,
            'gpu_determ': True,
            'encoder_lr': 0.
        }

    def __init__(self, config, ranker, vocab, train_ds, logger, random):
        self.config = config
        self.ranker = ranker
        self.vocab = vocab
        self.logger = logger
        self.dataset = train_ds
        self.random = random

        self.batch_size = self.config['batch_size']
        if self.config['grad_acc_batch'] > 0:
            assert self.config['batch_size'] % self.config['grad_acc_batch'] == 0, \
                "batch_size must be a multiple of grad_acc_batch"
            self.batch_size = self.config['grad_acc_batch']

        self.device = util.device(self.config, self.logger)

    def iter_train(self, only_cached=False):
        epoch = -1
        base_path = util.path_model_trainer(self.ranker, self.vocab, self, self.dataset)
        context = {
            'epoch': epoch,
            'batch_size': self.config['batch_size'],
            'batches_per_epoch': self.config['batches_per_epoch'],
            'num_microbatches': 1 if self.config['grad_acc_batch'] == 0 else self.config['batch_size'] // self.config['grad_acc_batch'],
            'device': self.device,
            'base_path': base_path,
        }

        files = trainers.misc.PathManager(base_path)
        self.logger.info(f'train path: {base_path}')

        b_count = context['batches_per_epoch'] * context['num_microbatches'] * self.batch_size

        ranker = self.ranker.to(self.device)
        optimizer = self.create_optimizer()

        if f'{epoch}.p' not in files['weights']:
            ranker.save(files['weights'][f'{epoch}.p'])
        if f'{epoch}.p' not in files['optimizer']:
            torch.save(optimizer.state_dict(), files['optimizer'][f'{epoch}.p'])

        context.update({
            'ranker': lambda: ranker,
            'ranker_path': files['weights'][f'{epoch}.p'],
            'optimizer': lambda: optimizer,
            'optimizer_path': files['optimizer'][f'{epoch}.p'],
        })

        yield context # before training

        while True:
            context = dict(context)

            epoch = context['epoch'] = context['epoch'] + 1
            if epoch in files['complete.tsv']:
                context.update({
                    'loss': files['loss.txt'][epoch],
                    'data_loss': files['data_loss.txt'][epoch],
                    'losses': {},
                    'acc': files['acc.tsv'][epoch],
                    'unsup_acc': files['unsup_acc.tsv'][epoch],
                    'ranker': _load_ranker(ranker, files['weights'][f'{epoch}.p']),
                    'ranker_path': files['weights'][f'{epoch}.p'],
                    'optimizer': _load_optimizer(optimizer, files['optimizer'][f'{epoch}.p']),
                    'optimizer_path': files['optimizer'][f'{epoch}.p'],
                    'cached': True,
                })
                if not only_cached:
                    self.fast_forward(b_count) # skip this epoch
                yield context
                continue

            if only_cached:
                break # no more cached

            # forward to previous versions (if needed)
            ranker = context['ranker']()
            optimizer = context['optimizer']()
            ranker.train()

            context.update({
                'loss': 0.0,
                'losses': {},
                'acc': 0.0,
                'unsup_acc': 0.0,
            })

            with tqdm(leave=False, total=b_count, ncols=100, desc=f'train {epoch}') as pbar:
                for b in range(context['batches_per_epoch']):
                    for _ in range(context['num_microbatches']):
                        self.epoch = epoch
                        train_batch_result = self.train_batch()
                        losses = train_batch_result['losses']
                        loss_weights = train_batch_result['loss_weights']
                        acc = train_batch_result.get('acc')
                        unsup_acc = train_batch_result.get('unsup_acc')

                        loss = sum(losses[k] * loss_weights.get(k, 1.) for k in losses) / context['num_microbatches']

                        context['loss'] += loss.item()
                        for lname, lvalue in losses.items():
                            context['losses'].setdefault(lname, 0.)
                            context['losses'][lname] += lvalue.item() / context['num_microbatches']

                        if acc is not None:
                            context['acc'] += acc.item() / context['num_microbatches']
                        if unsup_acc is not None:
                            context['unsup_acc'] += unsup_acc.item() / context['num_microbatches']

                        if loss.grad_fn is not None:
                            if hasattr(optimizer, 'backward'):
                                optimizer.backward(loss)
                            else:
                                loss.backward()
                        else:
                            self.logger.warn('loss has no grad_fn; skipping batch')
                        pbar.update(self.batch_size)

                    postfix = {
                        'loss': context['loss'] / (b + 1),
                    }
                    for lname, lvalue in context['losses'].items():
                        if lname in loss_weights and loss_weights[lname] != 1.:
                            postfix[f'{lname}({loss_weights[lname]})'] = lvalue / (b + 1)
                        else:
                            postfix[lname] = lvalue / (b + 1)

                    if postfix['loss'] == postfix['data']:
                        del postfix['data']

                    pbar.set_postfix(postfix)
                    optimizer.step()
                    optimizer.zero_grad()

            context.update({
                'ranker': lambda: ranker,
                'ranker_path': files['weights'][f'{epoch}.p'],
                'optimizer': lambda: optimizer,
                'optimizer_path': files['optimizer'][f'{epoch}.p'],
                'loss': context['loss'] / context['batches_per_epoch'],
                'losses': {k: v / context['batches_per_epoch'] for k, v in context['losses'].items()},
                'acc': context['acc'] / context['batches_per_epoch'],
                'unsup_acc': context['unsup_acc'] / context['batches_per_epoch'],
                'cached': False,
            })

            # save stuff
            ranker.save(files['weights'][f'{epoch}.p'])
            torch.save(optimizer.state_dict(), files['optimizer'][f'{epoch}.p'])
            files['loss.txt'][epoch] = context['loss']
            for lname, lvalue in context['losses'].items():
                files[f'loss_{lname}.txt'][epoch] = lvalue
            files['acc.tsv'][epoch] = context['acc']
            files['unsup_acc.tsv'][epoch] = context['unsup_acc']
            files['complete.tsv'][epoch] = 1 # mark as completed

            yield context

    def create_optimizer(self):
        params = self.ranker.named_parameters()
        # remove non-grad parameters
        params = [(k, v) for k, v in params if v.requires_grad]
        # split encoder params and non-encoder params to support encoder_lr setting
        encoder_params = {'params': [v for k, v in params if k.startswith('encoder.')]}
        non_encoder_params = {'params': [v for k, v in params if not k.startswith('encoder.')]}
        if self.config['encoder_lr'] > 0:
            encoder_params['lr'] = self.config['encoder_lr']
        params = []
        if non_encoder_params['params']:
            params.append(non_encoder_params)
        if encoder_params['params']:
            params.append(encoder_params)

        # build the optmizer
        return {
            'adam': lambda p: torch.optim.Adam(p, lr=self.config['lr']),
            'fusedadam': lambda p: apex.FusedAdam(p, lr=self.config['lr']),
            'fp16fusedadam': lambda p: apex.FP16_Optimizer(apex.FusedAdam(p, lr=self.config['lr']), dynamic_loss_scale=True),
        }[self.config['optimizer']](params)

    def path_segment(self):
        grad_acc = 'x{grad_acc_batch}'.format(**self.config) if self.config['grad_acc_batch'] > 0 else ''
        result = '{batches_per_epoch}x{batch_size}{grad_acc}_{optimizer}-{lr}'.format(**self.config, grad_acc=grad_acc)
        if self.config['encoder_lr'] > 0:
            result += '_encoderlr-{}'.format(self.config['encoder_lr'])
        return result

    def train_batch(self):
        raise NotImplementedError()

    def fast_forward(self, record_count):
        raise NotImplementedError()

    def _fast_forward(self, train_it, fields, record_count):
        # Since the train_it holds a refernece to fields, we can greatly speed up the "fast forward"
        # process by temporarily clearing the requested fields (meaning that the train iterator
        # should simply find the records/pairs to return, but yield an empty payload).
        orig_fields = set(fields)
        try:
            fields.clear()
            for _ in zip(range(record_count), train_it):
                pass
        finally:
            fields.update(orig_fields)


def _load_optimizer(optimizer, state_path):
    def _wrapped():
        optimizer.load_state_dict(torch.load(state_path))
        return optimizer
    return _wrapped


def _load_ranker(ranker, state_path):
    def _wrapped():
        ranker.load(state_path)
        return ranker
    return _wrapped
