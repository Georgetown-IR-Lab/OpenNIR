import os
import json
from onir import util, pipelines
import onir


@pipelines.register('default')
class DefaultPipeline(pipelines.BasePipeline):
    name = None

    @staticmethod
    def default_config():
        return {
            'max_epoch': 1000,
            'early_stop': 20,
            'warmup': -1,
            'val_metric': 'map',
            'purge_weights': True,
            'test': False,
            'initial_eval': False,
            'skip_ds_init': False,
            'only_cached': False,
        }

    def __init__(self, config, trainer, valid_pred, test_pred, logger):
        super().__init__(config, logger)
        self.trainer = trainer
        self.valid_pred = valid_pred
        self.test_pred = test_pred

    def run(self):
        validator = self.valid_pred.pred_ctxt()

        top_epoch, top_value, top_train_ctxt, top_valid_ctxt = None, None, None, None
        prev_train_ctxt = None

        file_output = {
            'ranker': self.trainer.ranker.path_segment(),
            'vocab': self.trainer.vocab.path_segment(),
            'trainer': self.trainer.path_segment(),
            'dataset': self.trainer.dataset.path_segment(),
            'valid_ds': self.valid_pred.dataset.path_segment(),
            'validation_metric': self.config['val_metric'],
            'logfile': util.path_log()
        }

        # initialize dataset(s)
        if not self.config['skip_ds_init']:
            self.trainer.dataset.init(force=False)
            self.valid_pred.dataset.init(force=False)
            if self.config['test']:
                self.test_pred.dataset.init(force=False)

        for train_ctxt in self.trainer.iter_train(only_cached=self.config['only_cached']):

            if prev_train_ctxt is not None and top_epoch is not None and prev_train_ctxt is not top_train_ctxt:
                self._purge_weights(prev_train_ctxt)

            if train_ctxt['epoch'] >= 0 and not self.config['only_cached']:
                message = self._build_train_msg(train_ctxt)

                if train_ctxt['cached']:
                    self.logger.debug(f'[train] [cached] {message}')
                else:
                    self.logger.debug(f'[train] {message}')

            if train_ctxt['epoch'] == -1 and not self.config['initial_eval']:
                continue

            valid_ctxt = dict(validator(train_ctxt))

            message = self._build_valid_msg(valid_ctxt)

            if valid_ctxt['epoch'] >= self.config['warmup']:
                if self.config['val_metric'] == '':
                    top_epoch = valid_ctxt['epoch']
                    top_train_ctxt = train_ctxt
                    top_valid_ctxt = valid_ctxt
                elif top_value is None or valid_ctxt['metrics'][self.config['val_metric']] > top_value:
                    message += ' <---'
                    top_epoch = valid_ctxt['epoch']
                    top_value = valid_ctxt['metrics'][self.config['val_metric']]
                    if top_train_ctxt is not None:
                        self._purge_weights(top_train_ctxt)
                    top_train_ctxt = train_ctxt
                    top_valid_ctxt = valid_ctxt
            else:
                if prev_train_ctxt is not None:
                    self._purge_weights(prev_train_ctxt)

            if not self.config['only_cached']:
                if valid_ctxt['cached']:
                    self.logger.debug(f'[valid] [cached] {message}')
                else:
                    self.logger.info(f'[valid] {message}')

            if top_epoch is not None:
                epochs_since_imp = valid_ctxt['epoch'] - top_epoch
                if self.config['early_stop'] > 0 and epochs_since_imp >= self.config['early_stop']:
                    self.logger.warn('stopping after epoch {epoch} ({early_stop} epochs with no '
                                     'improvement to {val_metric})'.format(**valid_ctxt, **self.config))
                    break

            if train_ctxt['epoch'] >= self.config['max_epoch']:
                self.logger.warn('stopping after epoch {max_epoch} (max_epoch)'.format(**self.config))
                break

            prev_train_ctxt = train_ctxt

        self.logger.info('top validation epoch={} {}={}'.format(top_epoch, self.config['val_metric'], top_value))

        file_output.update({
            'valid_epoch': top_epoch,
            'valid_run': top_valid_ctxt['run_path'],
            'valid_metrics': top_valid_ctxt['metrics'],
        })

        if self.config['test']:
            top_train_ctxt['ranker'] = onir.trainers.base._load_ranker(top_train_ctxt['ranker'](), top_train_ctxt['ranker_path'])

            with self.logger.duration('testing'):
                test_ctxt = self.test_pred.run(top_train_ctxt)

            file_output.update({
                'test_ds': self.test_pred.dataset.path_segment(),
                'test_run': test_ctxt['run_path'],
                'test_metrics': test_ctxt['metrics'],
            })

        with open(util.path_modelspace() + '/val_test.jsonl', 'at') as f:
            json.dump(file_output, f)
            f.write('\n')

        self.logger.info('valid run at {}'.format(valid_ctxt['run_path']))
        if self.config['test']:
            self.logger.info('test run at {}'.format(test_ctxt['run_path']))
        self.logger.info('valid ' + self._build_valid_msg(top_valid_ctxt))
        if self.config['test']:
            self.logger.info('test  ' + self._build_valid_msg(test_ctxt))

    def _build_train_msg(self, ctxt):
        delta_acc = ctxt['acc'] - ctxt['unsup_acc']
        msg_pt1 = 'epoch={epoch} loss={loss:.4f}'.format(**ctxt)
        msg_pt2 = 'acc={acc:.4f} unsup_acc={unsup_acc:.4f} ' \
                  'delta_acc={delta_acc:.4f}'.format(**ctxt, delta_acc=delta_acc)
        losses = ''
        if ctxt['losses'] and ({'data'} != ctxt['losses'].keys() or ctxt['losses']['data'] != ctxt['loss']):
            losses = []
            for lname, lvalue in ctxt['losses'].items():
                losses.append(f'{lname}={lvalue:.4f}')
            losses = ' '.join(losses)
            losses = f' ({losses})'
        return f'{msg_pt1}{losses} {msg_pt2}'


    def _build_valid_msg(self, ctxt):
        message = ['epoch=' + str(ctxt['epoch'])]
        for metric, value in sorted(ctxt['metrics'].items()):
            message.append('{}={:.4f}'.format(metric, value))
            if metric == self.config['val_metric']:
                message[-1] = '[' + message[-1] + ']'
        return ' '.join(message)

    def _purge_weights(self, ctxt):
        if self.config['purge_weights']:
            if os.path.exists(ctxt['ranker_path']):
                os.remove(ctxt['ranker_path'])
            if os.path.exists(ctxt['optimizer_path']):
                os.remove(ctxt['optimizer_path'])
