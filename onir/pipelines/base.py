import os
from onir import util


class BasePipeline:
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger

    def _load_ranker_weights(self, ranker, vocab, trainer, predictor, dataset):
        if self.config.get('file_path'):
            self.logger.info('loading model weights by path')
            self._load_ranker_weights_file_path(ranker, self.config['file_path'])
        elif self.config.get('epoch'):
            self.logger.info('loading model weights by epoch')
            self._load_ranker_weights_epoch(ranker, vocab, trainer, dataset)
        elif self.config.get('val_metric'):
            self.logger.info('loading model weights through validation')
            self._load_ranker_weights_validation(ranker, trainer, predictor)
        else:
            raise ValueError('Unable to find checkpoint from which to load weights. Please provide '
                             'either pipeline.file_path pipeline.epoch or pipeline.val_metric')
        if self.config.get('gpu'):
            self.logger.info('moving model to GPU')
            ranker.cuda()

    def _load_ranker_weights_file_path(self, ranker, path):
        ranker.load(path)
        ranker.eval()
        # TODO: any warnings about missing or extra args?

    def _load_ranker_weights_epoch(self, ranker, vocab, trainer, dataset):
        epcoh = self.config['epoch']
        base_path = util.path_model_trainer(ranker, vocab, trainer, dataset)
        weight_path = os.path.join(base_path, 'weights', f'{epcoh}.p')
        self._load_ranker_weights_file_path(ranker, weight_path)

    def _load_ranker_weights_validation(self, ranker, trainer, predictor):
        pred = predictor.pred_ctxt()
        val_metric = self.config['val_metric']

        top_epoch, top_value, top_train_ctxt = None, None, None

        for train_ctxt in trainer.iter_train(only_cached=True):
            valid_ctxt = dict(pred(train_ctxt))

            if top_value is None or valid_ctxt['metrics'][val_metric] > top_value:
                top_epoch = valid_ctxt['epoch']
                top_value = valid_ctxt['metrics'][val_metric]
                top_train_ctxt = train_ctxt

        self.logger.info(f'using epoch={top_epoch} ({val_metric}={top_value} on validation)')

        self._load_ranker_weights_file_path(ranker, top_train_ctxt['ranker_path'])
