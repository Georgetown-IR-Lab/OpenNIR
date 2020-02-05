import pytrec_eval
from onir import metrics as _metrics
Metric = _metrics.Metric


# pytrec_eval uses a fixed set of metrics (i.e., it doesn't support specifying the parameters of
# parametrized metrics). Thus, I'll use a simple mapping here, where the key is the onir metric
# and the value is a tuple of (1) the metric name passed to evaluate, and (2) the parametrized
# metric name returned from evaluate.
PTE_METRIC_MAP = {
    Metric('map'): ('map', 'map'),
    Metric('map', cutoff=5): ('map_cut', 'map_cut_5'),
    Metric('map', cutoff=10): ('map_cut', 'map_cut_10'),
    Metric('map', cutoff=15): ('map_cut', 'map_cut_15'),
    Metric('map', cutoff=20): ('map_cut', 'map_cut_20'),
    Metric('map', cutoff=30): ('map_cut', 'map_cut_30'),
    Metric('map', cutoff=100): ('map_cut', 'map_cut_100'),
    Metric('map', cutoff=200): ('map_cut', 'map_cut_200'),
    Metric('map', cutoff=500): ('map_cut', 'map_cut_500'),
    Metric('map', cutoff=1000): ('map_cut', 'map_cut_1000'),
    Metric('rprec'): ('Rprec', 'Rprec'),
    Metric('mrr'): ('recip_rank', 'recip_rank'),
    Metric('p', cutoff=5): ('P', 'P_5'),
    Metric('p', cutoff=10): ('P', 'P_10'),
    Metric('p', cutoff=15): ('P', 'P_15'),
    Metric('p', cutoff=20): ('P', 'P_20'),
    Metric('p', cutoff=30): ('P', 'P_30'),
    Metric('p', cutoff=100): ('P', 'P_100'),
    Metric('p', cutoff=200): ('P', 'P_200'),
    Metric('p', cutoff=500): ('P', 'P_500'),
    Metric('p', cutoff=1000): ('P', 'P_1000'),
    Metric('ndcg'): ('ndcg', 'ndcg'),
    Metric('ndcg', cutoff=5): ('ndcg_cut', 'ndcg_cut_5'),
    Metric('ndcg', cutoff=10): ('ndcg_cut', 'ndcg_cut_10'),
    Metric('ndcg', cutoff=15): ('ndcg_cut', 'ndcg_cut_15'),
    Metric('ndcg', cutoff=20): ('ndcg_cut', 'ndcg_cut_20'),
    Metric('ndcg', cutoff=30): ('ndcg_cut', 'ndcg_cut_30'),
    Metric('ndcg', cutoff=100): ('ndcg_cut', 'ndcg_cut_100'),
    Metric('ndcg', cutoff=200): ('ndcg_cut', 'ndcg_cut_200'),
    Metric('ndcg', cutoff=500): ('ndcg_cut', 'ndcg_cut_500'),
    Metric('ndcg', cutoff=1000): ('ndcg_cut', 'ndcg_cut_1000'),
}


class PyTrecEvalMetrics(_metrics.BaseMetrics):
    """
    Faster than TrecEvalMetrics (by using the native pytrec_eval package), but doesn't support all
    features, e.g., setting relevance level for metrics like MAP/Rprec, etc. or settings specific
    cutoff thresholds for metrics like P@X or ndcg@X. For that, use TrecEvalMetrics.
    """
    QRELS_FORMAT = 'dict'
    RUN_FORMAT = 'dict'

    def supports(self, metric):
        metric = _metrics.Metric.parse(metric)
        return metric in PTE_METRIC_MAP

    def calc_metrics(self, qrels_dict, run_dict, metrics, verbose=False):
        metric_for_pte = {PTE_METRIC_MAP[m][0] for m in metrics}
        evaluator = pytrec_eval.RelevanceEvaluator(qrels_dict, metric_for_pte)
        pte_results = evaluator.evaluate(run_dict)
        # translate and filter this to the output format
        result = {}
        for m in metrics:
            result[m] = {}
            for qid in pte_results:
                result[str(m)][qid] = pte_results[qid][PTE_METRIC_MAP[m][1]]
        return result
