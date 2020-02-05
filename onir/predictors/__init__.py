# pylint: disable=C0413
from onir import util
registry = util.Registry(default='reranker')
register = registry.register

from onir.predictors.base import BasePredictor
from onir.predictors.reranker import Reranker
