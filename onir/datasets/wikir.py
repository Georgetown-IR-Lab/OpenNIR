import os
import io
import zipfile
from pytools import memoize_method
from onir import util, datasets, indices
from onir.interfaces import trec, plaintext


_SOURCES = {
    '1k': 'https://zenodo.org/record/3565761/files/wikIR1k.zip?download=1',
    '59k': 'https://zenodo.org/record/3557342/files/wikIR59k.zip?download=1',
}


@datasets.register('wikir')
class WikirDataset(datasets.IndexBackedDataset):
    """
    Interface to the wikIR dataset.
     > Jibril Frej, Didier Schwab, Jean-Pierre Chevallet. 2019.  WIKIR: A Python toolkit for
     > building a large-scale Wikipedia-based English Information Retrieval Dataset.
    """
    DUA = """Will begin downloading wikIR dataset.
Please confirm you agree to the authors' data usage stipulations found at
https://github.com/getalp/wikIR"""

    @staticmethod
    def default_config():
        result = datasets.IndexBackedDataset.default_config()
        result.update({
            'collection': '1k',
            'subset': 'train',
            'rankfn': 'bm25',
            'ranktopk': 100,
            'special': '', # one of "", "mspairs", "msrun", "validrun"
        })
        return result

    def __init__(self, config, logger, vocab):
        super().__init__(config, logger, vocab)
        base_path = util.path_dataset(self)
        self.index1k = indices.AnseriniIndex(os.path.join(base_path, 'anserini.1k'), stemmer='none')
        self.index1k_stem = indices.AnseriniIndex(os.path.join(base_path, 'anserini.1k.porter'), stemmer='porter')
        self.index59k = indices.AnseriniIndex(os.path.join(base_path, 'anserini.59k'), stemmer='none')
        self.index59k_stem = indices.AnseriniIndex(os.path.join(base_path, 'anserini.59k.porter'), stemmer='porter')
        self.docstore1k = indices.SqliteDocstore(os.path.join(base_path, 'docs.1k.sqllite'))
        self.docstore59k = indices.SqliteDocstore(os.path.join(base_path, 'docs.59k.sqllite'))

    def _get_index(self, record):
        if self.config['collection'] == '1k':
            return self.index1k
        if self.config['collection'] == '59k':
            return self.index59k
        raise ValueError('unsupported collection')

    def _get_docstore(self):
        if self.config['collection'] == '1k':
            return self.docstore1k
        if self.config['collection'] == '59k':
            return self.docstore59k
        raise ValueError('unsupported collection')

    def _get_index_for_batchsearch(self):
        if self.config['collection'] == '1k':
            return self.index1k_stem
        if self.config['collection'] == '59k':
            return self.index59k_stem
        raise ValueError('unsupported collection')

    def qrels(self, fmt='dict'):
        return self._load_qrels(self.config['subset'], fmt=fmt)

    @memoize_method
    def _load_qrels(self, subset, fmt):
        with self.logger.duration('loading qrels'):
            base_path = util.path_dataset(self)
            path = os.path.join(base_path, f'{subset}.{self.config["collection"]}.qrels')
            self.logger.info(path)
            return trec.read_qrels_fmt(path, fmt)

    def load_queries(self) -> dict:
        return self._load_queries_base(self.config['subset'])

    @memoize_method
    def _load_queries_base(self, subset):
        with self.logger.duration('loading queries'):
            base_path = util.path_dataset(self)
            path = os.path.join(base_path, f'{subset}.{self.config["collection"]}.queries')
            return dict(plaintext.read_tsv(path))

    def path_segment(self):
        result = super().path_segment()
        result += '_{collection}'.format(**self.config)
        return result

    def init(self, force=False):
        self._init_collection('1k', force)
        self._init_collection('59k', force)

    def _init_collection(self, collection, force=False):
        base_path = util.path_dataset(self)
        if collection == '1k':
            idxs = [self.index1k, self.index1k_stem, self.docstore1k]
        elif collection == '59k':
            idxs = [self.index59k, self.index59k_stem, self.docstore59k]
        else:
            raise ValueError(f'unsupported collection {collection}')

        query_files = {
            f'wikIR{collection}/training/queries.csv': os.path.join(base_path, f'train.{collection}.queries'),
            f'wikIR{collection}/validation/queries.csv': os.path.join(base_path, f'dev.{collection}.queries'),
            f'wikIR{collection}/test/queries.csv': os.path.join(base_path, f'test.{collection}.queries')
        }

        qrels_files = {
            f'wikIR{collection}/training/qrels': os.path.join(base_path, f'train.{collection}.qrels'),
            f'wikIR{collection}/validation/qrels': os.path.join(base_path, f'dev.{collection}.qrels'),
            f'wikIR{collection}/test/qrels': os.path.join(base_path, f'test.{collection}.qrels')
        }

        theirbm25_files = {
            f'wikIR{collection}/training/BM25.res': os.path.join(base_path, f'train.{collection}.theirbm25'),
            f'wikIR{collection}/validation/BM25.res': os.path.join(base_path, f'dev.{collection}.theirbm25'),
            f'wikIR{collection}/test/BM25.res': os.path.join(base_path, f'test.{collection}.theirbm25')
        }

        if not force and \
           all(i.built() for i in idxs) and \
           all(os.path.exists(f) for f in query_files.values()) and \
           all(os.path.exists(f) for f in qrels_files.values()) and \
           all(os.path.exists(f) for f in theirbm25_files.values()):
            return

        if not self._confirm_dua():
            return

        with util.download_tmp(_SOURCES[collection]) as f:
            with zipfile.ZipFile(f) as zipf:
                doc_iter = self._init_iter_collection(zipf, collection)
                self._init_indices_parallel(idxs, doc_iter, force)

                for zqueryf, queryf in query_files.items():
                    if force or not os.path.exists(queryf):
                        with zipf.open(zqueryf) as f, open(queryf, 'wt') as out:
                            f = io.TextIOWrapper(f)
                            f.readline() # head
                            for qid, text in plaintext.read_sv(f, ','):
                                plaintext.write_tsv(out, [[qid, text]])

                for zqrelf, qrelf in qrels_files.items():
                    if force or not os.path.exists(qrelf):
                        with zipf.open(zqrelf) as f, open(qrelf, 'wt') as out:
                            f = io.TextIOWrapper(f)
                            plaintext.write_sv(out, plaintext.read_tsv(f), ' ')

                for zbm25, bm25 in theirbm25_files.items():
                    if force or not os.path.exists(bm25):
                        with zipf.open(zbm25) as f, open(bm25, 'wb') as out:
                            out.write(f.read())

    def _init_iter_collection(self, zipf, collection):
        with zipf.open(f'wikIR{collection}/documents.csv') as f:
            f = io.TextIOWrapper(f)
            f.readline() # head
            for did, text in self.logger.pbar(plaintext.read_sv(f, ','), desc='documents'):
                yield indices.RawDoc(did, text)
