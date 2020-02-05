import sys
import types
from onir import log


_logger = log.easy()


# hack for pylint
FP16_Optimizer = lambda *args, **kwargs: None
FusedAdam = lambda *args, **kwargs: None


class ApexWrapper(types.ModuleType):
    @property
    def _apex(self):
        try:
            import apex
            return apex
        except ImportError:
            _logger.warn('Module apex not installed. Please see <https://github.com/NVIDIA/apex>')
            raise

    @property
    def FP16_Optimizer(self):
        return self._apex.optimizers.FP16_Optimizer

    @property
    def FusedAdam(self):
        return self._apex.optimizers.FusedAdam


sys.modules[__name__].__class__ = ApexWrapper
