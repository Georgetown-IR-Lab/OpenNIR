from typing import Optional, List, Union, Callable, TextIO, Iterable
from tempfile import NamedTemporaryFile
import onir
from onir import metrics as _metrics
from onir.interfaces import trec


logger = onir.log.easy()


class FallbackMetrics(_metrics.BaseMetrics):
    def __init__(self, metrics_priority_list: List[_metrics.BaseMetrics]):
        self.metrics_priority_list = metrics_priority_list

    def supports(self, metric: Union[str, _metrics.Metric]) -> bool:
        metric = _metrics.Metric.parse(metric)
        for m in self.metrics_priority_list:
            if m.supports(metric):
                return True
        return False

    def calc_metrics(self,
                     qrels: Union[str, dict],
                     run: Union[str, dict],
                     metrics: Iterable[Union[str, _metrics.Metric]],
                     verbose: bool = False) -> dict:
        metrics = set(map(_metrics.Metric.parse, metrics))
        result = {}

        qrels = self.FormatManager(qrels, trec.read_qrels_dict, trec.write_qrels_dict_)
        run = self.FormatManager(run, trec.read_run_dict, trec.write_run_dict_)

        try:
            for metric_source in self.metrics_priority_list:
                these_metrics = set(filter(metric_source.supports, metrics))
                if these_metrics:
                    if verbose:
                        logger.debug(f'using {metric_source} for {these_metrics}')
                    these_qrels = qrels.in_format(metric_source.QRELS_FORMAT)
                    these_run = run.in_format(metric_source.RUN_FORMAT)
                    these_values = metric_source.calc_metrics(these_qrels, these_run, these_metrics)
                    for metric in these_values:
                        metrics.discard(metric)
                        result[metric] = these_values[metric]
                else:
                    if verbose:
                        logger.debug(f'no metrics supported by {metric_source}')
                if not metrics:
                    if verbose:
                        logger.debug(f'all metrics covered')
                    break
            if metrics:
                logger.warn(f'missing support for metrics: {metrics}')
            result = {str(k): v for k, v in result.items()}
            return result
        finally:
            qrels.close()
            run.close()

    class FormatManager:
        def __init__(self, item: Union[str, dict], to_dict: Callable[[TextIO], dict], to_file: Callable[[dict, TextIO], None]):
            self.item = item
            if isinstance(self.item, str):
                self.file = self.item
                self.dict = None
            elif isinstance(self.item, dict):
                self.file = None
                self.dict = self.item
            else:
                raise ValueError(f'unsupported format {type(item)}')
            self.tmp_file = None
            self.to_dict = to_dict
            self.to_file = to_file

        def in_format(self, fmt: Optional[str]) -> Union[str, dict]:
            if fmt is None:
                return self.item
            if fmt == 'dict':
                if self.dict is None:
                    with open(self.file, 'rt') as f:
                        self.dict = self.to_dict(f)
                return self.dict
            if fmt == 'file':
                if self.file is None:
                    self.tmp_file = NamedTemporaryFile('w+t')
                    self.to_file(self.dict, self.tmp_file)
                    self.file = self.tmp_file.name
                return self.file
            raise ValueError(f'unsuported fmt={fmt}')

        def close(self) -> None:
            if self.tmp_file is not None:
                self.tmp_file.close()
