import re
import pytrec_eval
from onir import metrics as _metrics
Metric = _metrics.Metric


OPT_REL = r'(_rel-(?P<rel>\d+))?'
OPT_GAIN = r'(_gain-(?P<gains>\d+=\d+(:\d+=\d+)+))?'
CUT = r'@(?P<par>\d+)'


# Defines metrics available in trec_eval and supported by this module.
# The key defines the a regular expression that matches a given metric string. The value is the
# pytrec_eval name (which may contain the parameter value from the regex). The relevance level
# parameter is supported by some metrics, and indicated with metric_l-X, where X is the minimum
# value considered relevant (e.g., p-l-4@5 is precision at 5 of documents graded at least a
# relevance of 4).
PTE_METRIC_MAP = {
    rf'^mrr{OPT_REL}$': 'recip_rank',
    rf'^rprec{OPT_REL}$': 'Rprec',
    rf'^map{OPT_REL}$': 'map',
    rf'^map{OPT_REL}{CUT}$': 'map_cut_{par}',
    rf'^ndcg{OPT_GAIN}$': 'ndcg',
    rf'^ndcg{OPT_GAIN}{CUT}$': 'ndcg_cut_{par}',
    rf'^p{OPT_REL}{CUT}$': 'P_{par}',
    rf'^r{OPT_REL}{CUT}$': 'recall_{par}',
}


class PyTrecEvalMetrics(_metrics.BaseMetrics):
    """
    Faster than TrecEvalMetrics (by using the native pytrec_eval package).
    """
    QRELS_FORMAT = 'dict'
    RUN_FORMAT = 'dict'

    def supports(self, metric):
        metric = _metrics.Metric.parse(metric)
        if metric is None:
            return False
        metric = str(metric)
        for exp in PTE_METRIC_MAP:
            if re.match(exp, metric):
                return True
        return False

    def calc_metrics(self, qrels_dict, run_dict, metrics, verbose=False):
        rel_args = {}
        for metric in metrics:
            for exp in PTE_METRIC_MAP:
                match = re.match(exp, str(metric))
                if match:
                    params = match.groupdict()
                    rel, gains = int(params.get('rel') or '1'), params.get('gain')
                    rel_args.setdefault((rel, gains), {})
                    metric_name = PTE_METRIC_MAP[exp].format(**params)
                    rel_args[rel, gains][metric_name] = str(metric)
                    break
        result = {}
        for (rel, gains), measures in rel_args.items():
            these_qrels = self._apply_gains(qrels_dict, gains)
            evaluator = pytrec_eval.RelevanceEvaluator(these_qrels, measures.keys(), relevance_level=rel)
            pte_results = evaluator.evaluate(run_dict)
            # translate and filter this to the output format
            for pte_name, onir_name in measures.items():
                result[onir_name] = {}
                for qid in pte_results:
                    result[onir_name][qid] = pte_results[qid][pte_name]
        return result

    def _apply_gains(self, qrels, gains):
        if gains:
            # apply custom gains
            gain_map = [g.split('=') for g in gains.split(':')]
            gain_map = {int(k): int(v) for k, v in gain_map}
            new_qrels = {}
            for qid in qrels:
                new_qrels[qid] = {}
                for did in qrels[qid]:
                    new_qrels[qid][did] = gain_map.get(qrels[qid][did], qrels[qid][did])
            qrels = new_qrels
        return qrels
