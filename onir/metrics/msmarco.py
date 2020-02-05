from onir import metrics as _metrics
from onir.interfaces import msmarco_eval


class MsMarcoMetrics(_metrics.BaseMetrics):
    QRELS_FORMAT = 'dict'
    RUN_FORMAT = 'dict'

    def supports(self, metric):
        metric = _metrics.Metric.parse(metric)
        if metric is None:
            return False
        check_args = len(metric.args.keys() - {'rel'}) == 0 # optinal rel arg
        check_cutoff = (metric.cutoff is not None and metric.cutoff > 0)
        return metric.name == 'mrr' and check_args and check_cutoff

    def calc_metrics(self, qrels, run, metrics, verbose=False):

        result = {}
        sorted_run = {q: list(sorted(run[q].items(), key=lambda x: (-x[1], x[0]))) for q in run}
        sorted_run = {q: [did for did, _ in v] for q, v in sorted_run.items()}

        for metric in metrics:
            result[metric] = {}
            m_metric = _metrics.Metric.parse(metric)
            m_cutoff = m_metric.cutoff
            if 'rel' in m_metric.args:
                m_rel = int(m_metric.args['rel'])
                m_qrels = {qid: {did: score for did, score in dids.items() if score >= m_rel} for qid, dids in qrels.items()}
            else:
                m_qrels = qrels
            msmarco_result = msmarco_eval.compute_metrics(m_qrels, sorted_run, max_rank=m_cutoff)
            result[str(metric)] = msmarco_result[f'MRR @{m_cutoff} by query']

        return result
