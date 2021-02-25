# pylint: disable=C0413
from onir import util
registry = util.Registry(default='flex')
register = registry.register

from onir.datasets.base import Dataset
from onir.datasets.index_backed import IndexBackedDataset, LazyDataRecord
from onir.datasets.multilingual_dataset import MultilingualDataset
from onir.datasets import antique, base, car, index_backed, msmarco, random, robust, flex, wikir, nyt, covid
from onir.datasets import trec_arabic, trec_mandarin, trec_spanish
from onir.datasets import irds

# Default iteration functions over datasets
from onir.datasets.query_iter import QueryIter as query_iter
from onir.datasets.doc_iter import DocIter as doc_iter
from onir.datasets.pair_iter import pair_iter
from onir.datasets.record_iter import record_iter, run_iter, qrels_iter, pos_qrels_iter
