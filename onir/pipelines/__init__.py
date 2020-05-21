# pylint: disable=C0413
from onir import util
registry = util.Registry(default='default')
register = registry.register

from onir.pipelines.base import BasePipeline
from onir.pipelines.default import DefaultPipeline
from onir.pipelines.grid_search import GridSearchPipeline
from onir.pipelines.extract_bert_weights import ExtractBertWeights
from onir.pipelines.epic_vectorize import EpicVectorize
from onir.pipelines.tune_rerank_threshold import TuneRerankThreshold
from onir.pipelines.epic_predict import EpicPredictionPipeline
