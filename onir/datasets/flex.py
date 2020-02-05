import os
import sys
import contextlib
import itertools
import functools
from glob import glob
from tqdm import tqdm
from pytools import memoize_method
from onir import util, datasets, indices
from onir.interfaces import plaintext, trec


@datasets.register('flex')
class FlexDataset(datasets.IndexBackedDataset):
    """
    A flexible dataset that uses plain-text data sources. Useful for experimentation and in cases
    where no special functionality is needed for a particular dataset.
    """

    @staticmethod
    def default_config():
        result = datasets.IndexBackedDataset.default_config()
        result.update({
            'subset': 'dummy'
        })
        return result

    def __init__(self, config, logger, vocab):
        super().__init__(config, logger, vocab)
        base_path = os.path.join(util.path_dataset(self), config['subset'])
        os.makedirs(base_path, exist_ok=True)
        self.index = indices.AnseriniIndex(os.path.join(base_path, 'anserini'), stemmer='none')
        self.index_stem = indices.AnseriniIndex(os.path.join(base_path, 'anserini.porter'), stemmer='porter')
        self.doc_store = indices.SqliteDocstore(os.path.join(base_path, 'docs.sqllite'))

    def collection_path_segment(self):
        return '{name}_{subset}'.format(name=self.name, **self.config)

    def qrels(self, fmt='dict'):
        return self._load_qrels(self.config['subset'], fmt)

    @memoize_method
    def _load_qrels(self, subset, fmt):
        qrels_path = os.path.join(util.path_dataset(self), subset, 'qrels.txt')
        return trec.read_qrels_fmt(qrels_path, fmt)

    def _get_index(self, record):
        return self.index

    def _get_docstore(self):
        return self.doc_store

    def _get_index_for_batchsearch(self):
        return self.index_stem

    @memoize_method
    def _load_queries_base(self, subset):
        queries_path = os.path.join(util.path_dataset(self), subset, 'queries.tsv')
        return {qid: qtext for qid, qtext in plaintext.read_tsv(queries_path)}

    def init(self, force=False):
        base_dir = os.path.join(util.path_dataset(self), self.config['subset'])

        if self.config['subset'] == 'dummy':
            datafile = os.path.join(base_dir, 'datafile.tsv')
            qrels = os.path.join(base_dir, 'qrels.txt')
            if not os.path.exists(datafile):
                os.symlink(os.path.abspath('etc/dummy_datafile.tsv'), datafile)
            if not os.path.exists(qrels):
                os.symlink(os.path.abspath('etc/dummy_qrels.txt'), qrels)

        needs_datafile = []
        if force or not self.index.built():
            needs_datafile.append(lambda it: self.index.build(indices.RawDoc(did, txt) for t, did, txt in it if t == 'doc'))

        if force or not self.index_stem.built():
            needs_datafile.append(lambda it: self.index_stem.build(indices.RawDoc(did, txt) for t, did, txt in it if t == 'doc'))

        if force or not self.doc_store.built():
            needs_datafile.append(lambda it: self.doc_store.build(indices.RawDoc(did, txt) for t, did, txt in it if t == 'doc'))

        query_file = os.path.join(base_dir, 'queries.tsv')
        if force or not os.path.exists(query_file):
            needs_datafile.append(lambda it: plaintext.write_tsv(query_file, ((qid, txt) for t, qid, txt in it if t == 'query')))

        if needs_datafile:
            df_glob = os.path.join(base_dir, 'datafile*.tsv')
            datafiles = glob(df_glob)
            while not datafiles:
                c = util.confirm(f'No data files found. Please move/link data files to {df_glob}.\n'
                                  'Data files should contain both queries and documents in the '
                                  'following format (one per line):\n'
                                  '[query|doc] [TAB] [qid/did] [TAB] [text]')
                if not c:
                    sys.exit(1)
                datafiles = glob(df_glob)
            main_iter = itertools.chain(*(plaintext.read_tsv(df) for df in datafiles))
            main_iter = tqdm(main_iter, desc='reading datafiles')
            iters = util.blocking_tee(main_iter, len(needs_datafile))
            with contextlib.ExitStack() as stack:
                for fn, it in zip(needs_datafile, iters):
                    stack.enter_context(util.CtxtThread(functools.partial(fn, it)))

        qrels_file = os.path.join(base_dir, 'qrels.txt')
        while not os.path.exists(qrels_file):
            c = util.confirm(f'No qrels file found. Please move/link qrels file to {qrels_file}.\n'
                              'Qrels file should be in the TREC format:\n'
                              '[qid] [SPACE] Q0 [SPACE] [did] [SPACE] [score]')
            if not c:
                sys.exit(1)
