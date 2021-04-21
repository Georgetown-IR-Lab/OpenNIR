import os
import io
import itertools
import gzip
import tarfile
import zipfile
import contextlib
import functools
from tqdm import tqdm
from pytools import memoize_method
import pandas as pd
import ir_datasets
import onir
from onir import util, datasets, indices
from onir.interfaces import trec, plaintext


def sanitize_path(s):
    return s.replace('/', '--')


@datasets.register('irds')
class IrdsDataset(datasets.IndexBackedDataset):
    @staticmethod
    def default_config():
        result = datasets.IndexBackedDataset.default_config()
        result.update({
            'ds': '', # used as shortcut
            'doc_fields': '', # used as shortcut
            'query_fields': '', # used as shortcut
            'docs_ds': '',
            'docs_index_fields': '',
            'docs_rerank_fields': '',
            'queries_ds': '',
            'queries_index_fields': '',
            'queries_rerank_fields': '',
            'rankfn': onir.config.Ranker(),
            'ranktopk': 100,
        })
        return result

    def __init__(self, config, logger, vocab):
        super().__init__(config, logger, vocab)
        if config['ds']:
            ds = ir_datasets.load(config['ds'])
            if not config['docs_ds']:
                # HACK: find "parent" dataset that contains same docs handler so we don't re-build the index for the same collection
                segments = config['ds'].split('/')
                docs_handler = ds.docs_handler()
                parent_docs_ds = config['ds']
                while len(segments) > 1:
                    segments = segments[:-1]
                    parent_ds = ir_datasets.load('/'.join(segments))
                    if parent_ds.has_docs() and parent_ds.docs_handler() == docs_handler:
                        parent_docs_ds = '/'.join(segments)
                config['docs_ds'] = parent_docs_ds
            if not config['queries_ds']:
                config['queries_ds'] = config['ds']

        if config['doc_fields']:
            if not config['docs_index_fields']:
                config['docs_index_fields'] = config['doc_fields']
            if not config['docs_rerank_fields']:
                config['docs_rerank_fields'] = config['doc_fields']

        if config['query_fields']:
            if not config['queries_index_fields']:
                config['queries_index_fields'] = config['query_fields']
            if not config['queries_rerank_fields']:
                config['queries_rerank_fields'] = config['query_fields']

        self.docs_ds = ir_datasets.load(config['docs_ds'])
        self.queries_ds = ir_datasets.load(config['queries_ds'])

        assert self.docs_ds.has_docs()
        assert self.queries_ds.has_queries()

        if not config['docs_index_fields']:
            config['docs_index_fields'] = ','.join(self.docs_ds.docs_cls()._fields[1:])
            self.logger.info('auto-filled docs_index_fields as {docs_index_fields}'.format(**config))
        if not config['docs_rerank_fields']:
            config['docs_rerank_fields'] = ','.join(self.docs_ds.docs_cls()._fields[1:])
            self.logger.info('auto-filled docs_rerank_fields as {docs_rerank_fields}'.format(**config))
        if not config['queries_index_fields']:
            config['queries_index_fields'] = ','.join(self.queries_ds.queries_cls()._fields[1:])
            self.logger.info('auto-filled queries_index_fields as {queries_index_fields}'.format(**config))
        if not config['queries_rerank_fields']:
            config['queries_rerank_fields'] = ','.join(self.queries_ds.queries_cls()._fields[1:])
            self.logger.info('auto-filled queries_rerank_fields as {queries_rerank_fields}'.format(**config))

        base_path = os.path.join(util.path_dataset(self), sanitize_path(self.config['docs_ds']))
        os.makedirs(base_path, exist_ok=True)
        real_anserini_path = os.path.join(base_path, 'anserini.porter.{docs_index_fields}'.format(**self.config))
        os.makedirs(real_anserini_path, exist_ok=True)
        virtual_anserini_path = '{}.{}'.format(real_anserini_path, sanitize_path(config['queries_ds']))
        if not os.path.exists(virtual_anserini_path):
            os.symlink(real_anserini_path, virtual_anserini_path, target_is_directory=True)
        self.index = indices.AnseriniIndex(virtual_anserini_path, stemmer='porter')
        self.doc_store = indices.IrdsDocstore(self.docs_ds.docs_store(), config['docs_rerank_fields'])

    def _get_docstore(self):
        return self.doc_store

    def _get_index(self, record):
        return self.index

    def _get_index_for_batchsearch(self):
        return self.index

    @memoize_method
    def qrels(self, fmt='dict'):
        if fmt == 'dict':
            return self.queries_ds.qrels_dict()
        if fmt == 'df':
            df = pd.DataFrame(self.queries_ds.qrels_iter())
            df = df.rename(columns={'query_id': 'qid', 'doc_id': 'did', 'relevance': 'score'})
            return df
        raise RuntimeError(f'unsupported fmt={fmt}')

    @memoize_method
    def load_queries(self) -> dict:
        queries_cls = self.queries_ds.queries_cls()
        fields = self.config['queries_rerank_fields'].split(',')
        assert all(f in queries_cls._fields for f in fields)
        field_idxs = [queries_cls._fields.index(f) for f in fields]
        return {q.query_id: '\n'.join(q[i] for i in field_idxs) for q in self.queries_ds.queries_iter()}

    @memoize_method
    def _load_queries_base(self, subset):
        # HACK: this subtly only gets called for runs in this impl. Use queries_index_fields instead here.
        queries_cls = self.queries_ds.queries_cls()
        fields = self.config['queries_index_fields'].split(',')
        assert all(f in queries_cls._fields for f in fields)
        field_idxs = [queries_cls._fields.index(f) for f in fields]
        return {q.query_id: ' '.join(q[i] for i in field_idxs).replace('\n', ' ') for q in self.queries_ds.queries_iter()}

    def path_segment(self):
        return '__'.join([
            super().path_segment(),
            sanitize_path(self.config["docs_ds"]),
            self.config['docs_index_fields'],
            self.config['docs_rerank_fields'],
            sanitize_path(self.config["queries_ds"]),
            self.config['queries_index_fields'],
            self.config['queries_rerank_fields']])

    def init(self, force=False):
        if not self.index.built() or force:
            doc_it = self._init_iter_collection()
            doc_it = self.logger.pbar(doc_it, 'docs')
            self.index.build(doc_it)
        # Attempt to grab everything (without wasting too many resources).
        # This isn't really a guarantee we have everything, but it should work in most cases.
        next(self.docs_ds.docs_iter())
        next(self.queries_ds.queries_iter())
        next(self.queries_ds.qrels_iter())

    def _init_iter_collection(self):
        docs_cls = self.docs_ds.docs_cls()
        fields = self.config['docs_index_fields'].split(',')
        assert all(f in docs_cls._fields for f in fields)
        field_idxs = [docs_cls._fields.index(f) for f in fields]
        for doc in self.docs_ds.docs_iter():
            yield indices.RawDoc(doc.doc_id, '\n'.join(str(doc[i]) for i in field_idxs))
