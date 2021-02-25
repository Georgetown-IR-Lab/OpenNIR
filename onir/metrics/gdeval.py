import os
import subprocess
from onir import metrics as _metrics


class GdevalMetrics(_metrics.BaseMetrics):
    QRELS_FORMAT = 'file'
    RUN_FORMAT = 'file'

    def supports(self, metric):
        metric = _metrics.Metric.parse(metric)
        if metric is None:
            return False
        return metric.name in ('ndcg', 'err') and len(metric.args) == 0 and metric.cutoff > 0

    def calc_metrics(self, qrelsf, runf, metrics, verbose=False):
        cutoffs = {}
        for metric in metrics:
            metric = _metrics.Metric.parse(metric)
            assert metric is not None
            cutoffs.setdefault(metric.cutoff, set()).add((metric.name, metric))

        results = {}

        for cutoff in cutoffs:
            results = self._gdeval(qrelsf, runf, cutoff)
            for m_name, metric in cutoffs[cutoff]:
                results[str(metric)] = results[m_name]
        return results

    def _gdeval(self, qrelsf, runf, cutoff):
        """
        Runs gdeval.pl on the given run/qrels pair
        """
        # adapted from searchivarius/AccurateLuceneBM25/scripts/eval_output_gdeval.py
        perlf = os.path.join(os.path.dirname(__file__), os.pardir, 'resources', 'gdeval.pl')
        output = subprocess.check_output([perlf, qrelsf, runf, str(cutoff)])
        output = output.decode().replace('\t', ' ').split('\n')
        result = {'ndcg': {}, 'err': {}}
        for i, s in enumerate(output):
            if s == '' or i == 0:
                continue
            arr = s.split(',')
            assert len(arr) == 4
            _, qid, ndcg, err = arr
            result['ndcg'][qid] = float(ndcg)
            result['err'][qid] = float(err)
        return result
