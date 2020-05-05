import os
import json
import torch
import onir
from onir import util, spec, predictors, datasets
from onir.interfaces import trec, plaintext


@predictors.register('reranker')
class Reranker(predictors.BasePredictor):
    name = None

    @staticmethod
    def default_config():
        return {
            'batch_size': 64,
            'gpu': True,
            'gpu_determ': True,
            'preload': False,
            'measures': 'map,ndcg,p@20,ndcg@20,mrr',
            'source': 'run'
        }

    def __init__(self, config, ranker, trainer, dataset, vocab, logger, random):
        self.config = config
        self.ranker = ranker
        self.trainer = trainer
        self.dataset = dataset
        self.logger = logger
        self.vocab = vocab
        self.random = random
        self.input_spec = ranker.input_spec()

    def _iter_batches(self, device):
        fields = set(self.input_spec['fields']) | {'query_id', 'doc_id'}
        it = datasets.record_iter(self.dataset,
                                  fields=fields,
                                  source=self.config['source'],
                                  minrel=None,
                                  shuf=False,
                                  random=self.random,
                                  inf=False)
        for batch_items in util.chunked(it, self.config['batch_size']):
            batch = {}
            for record in batch_items:
                for k, seq in record.items():
                    batch.setdefault(k, []).append(seq)
            batch = spec.apply_spec_batch(batch, self.input_spec, device)
            # ship 'em
            yield batch

    def _preload_batches(self, device):
        with self.logger.duration('loading evaluation data'):
            batches = list(self.logger.pbar(self._iter_batches(device), desc='preloading eval data (batches)'))
        while True:
            yield batches

    def _reload_batches(self, device):
        while True:
            it = self._iter_batches(device)
            yield it

    def pred_ctxt(self):
        device = util.device(self.config, self.logger)

        if self.config['preload']:
            datasource = self._preload_batches(device)
        else:
            datasource = self._reload_batches(device)

        return PredictorContext(self, datasource, device)

    def iter_scores(self, ranker, datasource, device):
        if ranker.name == 'trivial' and not ranker.config['neg'] and not ranker.config['qsum'] and not ranker.config['max']:
            for qid, values in self.dataset.run().items():
                for did, score in values.items():
                    yield qid, did, score
            return
        if ranker.name == 'trivial' and not ranker.config['neg'] and not ranker.config['qsum'] and ranker.config['max']:
            qrels = self.dataset.qrels()
            for qid, values in self.dataset.run().items():
                q_qrels = qrels.get(qid, {})
                for did in values:
                    yield qid, did, q_qrels.get(did, -1)
            return
        with torch.no_grad():
            ranker.eval()
            ds = next(datasource, None)
            total = None
            if isinstance(ds, list):
                total = sum(len(d['query_id']) for d in ds)
            elif self.config['source'] == 'run':
                total = sum(len(v) for v in self.dataset.run().values())
            elif self.config['source'] == 'qrels':
                total = sum(len(v) for v in self.dataset.qrels().values())
            with self.logger.pbar_raw(total=total, desc='pred', quiet=True) as pbar:
                for batch in util.background(ds):
                    batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
                    rel_scores = self.ranker(**batch).cpu()
                    if len(rel_scores.shape) == 2:
                        rel_scores = rel_scores[:, 0]
                    triples = list(zip(batch['query_id'], batch['doc_id'], rel_scores))
                    for qid, did, score in triples:
                        yield qid, did, score.item()
                    pbar.update(len(batch['query_id']))

    def rerank_dict(self, ranker, device):
        datasource = self._reload_batches(device)
        result = {}
        for qid, did, score in self.iter_scores(ranker, datasource, device):
            result.setdefault(qid, {})[did] = score
        return result


class PredictorContext:
    def __init__(self, pred, datasource, device):
        self.pred = pred
        self.datasource = datasource
        self.device = device

    def __call__(self, ctxt):
        cached = True
        epoch = ctxt['epoch']
        base_path = os.path.join(ctxt['base_path'], self.pred.dataset.path_segment())
        os.makedirs(os.path.join(base_path, 'runs'), exist_ok=True)
        with open(os.path.join(base_path, 'config.json'), 'wt') as f:
            json.dump(self.pred.dataset.config, f)
        run_path = os.path.join(base_path, 'runs', f'{epoch}.run')
        if os.path.exists(run_path):
            run = trec.read_run_dict(run_path)
        else:
            run = {}
            ranker = ctxt['ranker']().to(self.device)
            this_qid = None
            these_docs = {}
            with util.finialized_file(run_path, 'wt') as f:
                for qid, did, score in self.pred.iter_scores(ranker, self.datasource, self.device):
                    if qid != this_qid:
                        if this_qid is not None:
                            trec.write_run_dict(f, {this_qid: these_docs})
                        this_qid = qid
                        these_docs = {}
                    these_docs[did] = score
                if this_qid is not None:
                    trec.write_run_dict(f, {this_qid: these_docs})
            cached = False

        result = {
            'epoch': epoch,
            'run': run,
            'run_path': run_path,
            'base_path': base_path,
            'cached': cached
        }

        result['metrics'] = {m: None for m in self.pred.config['measures'].split(',') if m}
        result['metrics_by_query'] = {m: None for m in result['metrics']}

        missing_metrics = self.load_metrics(result)

        if missing_metrics:
            measures = set(missing_metrics)
            result['cached'] = False
            qrels = self.pred.dataset.qrels()
            calculated_metrics = onir.metrics.calc(qrels, run_path, measures)
            result['metrics_by_query'].update(calculated_metrics)
            result['metrics'].update(onir.metrics.mean(calculated_metrics))
            self.write_missing_metrics(result, missing_metrics)

        try:
            if ctxt['ranker']().config.get('add_runscore'):
                result['metrics']['runscore_alpha'] = torch.sigmoid(ctxt['ranker']().runscore_alpha).item()
                rs_alpha_f = os.path.join(ctxt['base_path'], 'runscore_alpha.txt')
                with open(rs_alpha_f, 'at') as f:
                    plaintext.write_tsv(rs_alpha_f, [(str(epoch), str(result['metrics']['runscore_alpha']))])
        except FileNotFoundError:
            pass # model may no longer exist, ignore

        return result

    def load_metrics(self, ctxt):
        missing = set()
        epoch = ctxt['epoch']
        for metric in list(ctxt['metrics']):
            path_agg = os.path.join(ctxt['base_path'], metric, 'agg.txt')
            path_epoch = os.path.join(ctxt['base_path'], metric, f'{epoch}.txt')
            if os.path.exists(path_agg) and os.path.exists(path_epoch):
                ctxt['metrics'][metric] = [float(v) for k, v in plaintext.read_tsv(path_agg) if int(k) == epoch][0]
                ctxt['metrics_by_query'][metric] = {k: float(v) for k, v in plaintext.read_tsv(path_epoch)}
            else:
                missing.add(metric)
        return missing

    def write_missing_metrics(self, ctxt, missing_metrics):
        epoch = ctxt['epoch']
        for metric in missing_metrics:
            os.makedirs(os.path.join(ctxt['base_path'], metric), exist_ok=True)
            path_agg = os.path.join(ctxt['base_path'], metric, 'agg.txt')
            path_epoch = os.path.join(ctxt['base_path'], metric, f'{epoch}.txt')
            with open(path_agg, 'at') as f:
                plaintext.write_tsv(f, [(str(epoch), str(ctxt['metrics'][metric]))])
            plaintext.write_tsv(path_epoch, ctxt['metrics_by_query'][metric].items())
