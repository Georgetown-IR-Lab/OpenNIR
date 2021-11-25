import os
from onir import metrics as _metrics
import ir_measures

class IrMeasuresMetrics(_metrics.BaseMetrics):
    QRELS_FORMAT = 'dict'
    RUN_FORMAT = 'dict'

    def supports(self, metric):
        try:
            measure = ir_measures.parse_measure(str(metric))
            return True
        except ValueError:
            return False
        except NameError:
            return False

    def calc_metrics(self, qrels, run, metrics, verbose=False):
        measures = {ir_measures.parse_measure(str(m)): str(m) for m in metrics}
        results = {}
        for metric in ir_measures.iter_calc(list(measures), qrels, run):
            measure = measures[metric.measure]
            if measure not in results:
                results[measure] = {}
            results[measure][metric.query_id] = metric.value
        return results
