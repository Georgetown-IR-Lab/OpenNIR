from typing import Optional, Mapping, Union, Any, Iterable
import re


class Metric:
    def __init__(self, name: str, args: Mapping[str, str] = None, cutoff: Optional[int] = None):
        if args is None:
            args = {}
        self._name = name
        self._args_frzn = tuple(sorted(args.items()))
        self._cutoff = cutoff

    @property
    def name(self) -> str:
        return self._name

    @property
    def args(self) -> Mapping[str, str]:
        return dict(self._args_frzn)

    @property
    def cutoff(self):
        return self._cutoff

    @classmethod
    def parse(cls, metric: Union[str, 'Metric']) -> 'Metric':
        if isinstance(metric, Metric):
            return metric
        # metric format: name_opt1-val1_opt2-val2@cutoff
        match = re.match(r'^([^-_@]+)((_[^-_@]+-[^-_@]+)*)?(@([^-_@]+))?$', metric)
        name = match.group(1)
        _ = match.group(2)
        settings_text = match.group(3)
        _ = match.group(4)
        cutoff = match.group(5)
        if cutoff:
            cutoff = int(cutoff)
        args = {}
        if settings_text:
            for setting in settings_text[1:].split('_'):
                key, value = setting.split('-')
                args[key] = value
        return cls(name, args, cutoff)

    @classmethod
    def cannonical(cls, metric: str) -> str:
        return str(cls.parse(metric))

    def __str__(self) -> str:
        result = self.name
        args = '_'.join(['-'.join(p) for p in self._args_frzn])
        if args != '':
            result += f'_{args}'
        if self.cutoff is not None:
            result += f'@{self.cutoff}'
        return result

    def __repr__(self) -> str:
        return str(self)

    def __hash__(self) -> int:
        return hash(str(self))

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, Metric):
            other = str(other)
        return isinstance(other, str) and other == str(self)

    def __ne__(self, other: Any) -> bool:
        return not self == other


class BaseMetrics:
    QRELS_FORMAT = None
    RUN_FORMAT = None

    def supports(self, metric):
        raise NotImplementedError()

    def calc_metrics(self, qrels, run, metrics: Iterable[Union[str, Metric]], verbose: bool = False):
        raise NotImplementedError()
