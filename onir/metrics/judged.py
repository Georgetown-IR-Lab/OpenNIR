from onir import metrics as _metrics


class JudgedMetrics(_metrics.BaseMetrics):
    QRELS_FORMAT = 'dict'
    RUN_FORMAT = 'dict'

    def supports(self, metric):
        metric = _metrics.Metric.parse(metric)
        if metric is None:
            return False
        return metric.name in ('judged',) and len(metric.args) == 0 and metric.cutoff > 0

    def calc_metrics(self, qrels, run, metrics, verbose=False):
        result = {}
        sorted_run = {q: list(sorted(run[q].items(), key=lambda x: (-x[1], x[0]))) for q in run}

        for metric in metrics:
            result[metric] = {}
            m_cutoff = _metrics.Metric.parse(metric).cutoff
            for qid in run:
                qid_qrels = qrels.get(qid, {})
                judged_c = sum(did in qid_qrels for did, _ in sorted_run[qid][:m_cutoff])
                result[str(metric)][qid] = judged_c / m_cutoff

        return result
