import os
from onir import pipelines, util, metrics


@pipelines.register('tune_rerank_threshold')
class TuneRerankThreshold(pipelines.BasePipeline):
    @staticmethod
    def default_config():
        return {
            'file_path': '',
            'epoch': '',
            'val_metric': '',
            'overwrite': False,
            'gpu': True,
            'gpu_determ': True,
        }

    def __init__(self, config, ranker, vocab, train_ds, trainer, valid_ds, valid_pred, test_ds, test_pred, logger):
        super().__init__(config, logger)
        self.ranker = ranker
        self.vocab = vocab
        self.trainer = trainer
        self.train_ds = train_ds
        self.valid_ds = valid_ds
        self.valid_pred = valid_pred
        self.test_ds = test_ds
        self.test_pred = test_pred

    def run(self):
        self._load_ranker_weights(self.ranker, self.vocab, self.trainer, self.valid_pred, self.train_ds)
        device = util.device(self.config, self.logger)

        unsupervised_run = self.test_ds.run_dict()
        supervised_run = self.test_pred.rerank_dict(self.ranker, device)
        unsupervised_run_indexed = run_indexed(unsupervised_run)
        qrels = self.test_ds.qrels()
        measures = list(self.test_pred.config['measures'].split(','))

        top_threshold, top_metric = 0, 0
        for threshold in range(1, self.test_ds.config['ranktopk'] + 1):
            threshold_run = rerank_cutoff(threshold, unsupervised_run_indexed, supervised_run)
            metrics_by_query = metrics.calc(qrels, threshold_run, set(measures))
            metrics_mean = metrics.mean(metrics_by_query)
            message = ' '.join([f'threshold={threshold}'] + [f'{k}={v:.4f}' for k, v in metrics_mean.items()])
            if metrics_mean[measures[0]] > top_metric:
                top_threshold, top_metric = threshold, metrics_mean[measures[0]]
                message += ' <--'
            self.logger.debug(message)
        self.logger.info('top_threshold={} {}={:.4f}'.format(top_threshold, measures[0], top_metric))


def run_indexed(run):
    result = {}
    for qid in run:
        result[qid] = sorted(run[qid].items(), key=lambda x: x[1], reverse=True)
        result[qid] = [did for did, score in result[qid]]
    return result


def rerank_cutoff(threshold, initial_run_index, run):
    result = {}
    for qid in initial_run_index:
        result[qid] = dict((did, run[qid][did]) for did in initial_run_index[qid][:threshold])
        min_score = min(result[qid].values())
        result[qid].update((did, min_score-idx-1) for idx, did in enumerate(initial_run_index[qid][threshold:]))
    return result
