from typing import Dict
from onir.metrics.base import BaseMetrics, Metric
from onir.metrics.fallback import FallbackMetrics
from onir.metrics.gdeval import GdevalMetrics
from onir.metrics.judged import JudgedMetrics
from onir.metrics.msmarco import MsMarcoMetrics
from onir.metrics.pytreceval import PyTrecEvalMetrics
from onir.metrics.treceval import TrecEvalMetrics


primary = FallbackMetrics([
    MsMarcoMetrics(),
    PyTrecEvalMetrics(),
    TrecEvalMetrics(),
    JudgedMetrics(),
    GdevalMetrics()
])
calc = primary.calc_metrics


def mean(metrics: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    result = {}
    for m_name, values_by_query in metrics.items():
        result[m_name] = sum(values_by_query.values()) / len(values_by_query)
    return result
