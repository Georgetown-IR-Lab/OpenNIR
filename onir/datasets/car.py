import os
from pathlib import Path
from pytools import memoize_method
from onir import util, datasets, indices
from onir.interfaces import trec, car, plaintext


_SOURCES = {
    'corpus': 'http://trec-car.cs.unh.edu/datareleases/v1.5/paragraphcorpus-v1.5.tar.xz',
    'train': 'http://trec-car.cs.unh.edu/datareleases/v1.5/train-v1.5.tar.xz',
    'test': 'http://trec-car.cs.unh.edu/datareleases/v1.5/benchmarkY1test.public-v1.5.tar.xz',
    'test-qrels': 'http://trec-car.cs.unh.edu/datareleases/v1.5/trec-car-2017-qrels.tar.gz',
    'test200': 'http://trec-car.cs.unh.edu/datareleases/v1.5/test200-v1.5.tar.xz',
}


@datasets.register('car')
class CarV1Dataset(datasets.IndexBackedDataset):
    """
    An interface to the TREC Complex Answer Retrieval (CAR) dataset.
     > Laura Dietz, Ben Gamari, Jeff Dalton, and Nick Craswell. 2017. TREC Complex Answer Retrieval
     > Overview. In TREC.
    """
    DUA = """Will begin downloading CAR dataset.
Please confirm you agree to the authors' data usage stipulations found at
http://trec-car.cs.unh.edu/"""

    @staticmethod
    def default_config():
        result = datasets.IndexBackedDataset.default_config()
        result.update({
            'subset': 'train-f0',
            'rankfn': 'bm25',
            'ranktopk': 100,
            'rel': 'auto'
        })
        return result

    def __init__(self, config, logger, vocab):
        super().__init__(config, logger, vocab)
        base_path = util.path_dataset(self)
        self.index = indices.AnseriniIndex(os.path.join(base_path, 'anserini'), stemmer='none')
        self.index_stem = indices.AnseriniIndex(os.path.join(base_path, 'anserini.porter'), stemmer='porter')
        self.doc_store = indices.SqliteDocstore(os.path.join(base_path, 'docs.sqlite'))

    def path_segment(self):
        result = '{name}_{subset}_{rankfn}.{ranktopk}'.format(name=self.name, **self.config)
        if self.config['rel'] != 'auto':
            result += '_{rel}'.format(**self.config)
        return result

    def _get_index(self, record):
        return self.index

    def _get_docstore(self):
        return self.doc_store

    def _get_index_for_batchsearch(self):
        return self.index_stem

    def qrels(self, fmt='dict'):
        return self._load_qrels(self.config['subset'], fmt)

    @memoize_method
    def _load_qrels(self, subset, fmt):
        with self.logger.duration('loading qrels'):
            path = util.path_dataset(self)
            rel = self.config['rel']
            qrel_file = os.path.join(path, f'{subset}.{rel}.qrels')
            return trec.read_qrels_fmt(qrel_file, fmt)

    @memoize_method
    def _load_queries_base(self, subset):
        path = os.path.join(util.path_dataset(self), f'{subset}.queries.tsv')
        result = {}
        for cols in plaintext.read_tsv(path):
            result[cols[0]] = ' '.join(cols[1:])
        return result

    def init(self, force=False):
        base_path = util.path_dataset(self)
        base = Path(base_path)

        # DOCUMENT COLLECTION
        idx = [self.index, self.index_stem, self.doc_store]
        self._init_indices_parallel(idx, self._init_doc_iter(), force)

        # TRAIN

        files = {}
        files.update({
            base / f'train-f{f}.auto.qrels': f'train/train.fold{f}.cbor.hierarchical.qrels' for f in range(5)
        })
        files.update({
            base / f'train-f{f}.queries.tsv': f'train/train.fold{f}.cbor.outlines' for f in range(5)
        })
        if force or not all(f.exists() for f in files) and self._confirm_dua():
            with util.download_tmp(_SOURCES['train'], tarf=True) as f:
                for member in f:
                    for f_out, f_in in files.items():
                        if member.name == f_in:
                            if f_out.suffix == '.qrels':
                                self._init_file_copy(f.extractfile(member), f_out, force)
                            elif f_out.suffix == '.tsv':
                                self._init_queryfile(f.extractfile(member), f_out, force)

        # TEST

        files = {
            base / 'test.queries.tsv': 'benchmarkY1test.public/test.benchmarkY1test.cbor.outlines'
        }
        if force or not all(f.exists() for f in files) and self._confirm_dua():
            with util.download_tmp(_SOURCES['test'], tarf=True) as f:
                for f_out, f_in in files.items():
                    self._init_queryfile(f.extractfile(f_in), f_out, force)

        files = {
            base / 'test.auto.qrels': 'TREC_CAR_2017_qrels/automatic.benchmarkY1test.cbor.hierarchical.qrels',
            base /'test.manual.qrels': 'TREC_CAR_2017_qrels/manual.benchmarkY1test.cbor.hierarchical.qrels',
        }
        if force or not all(f.exists() for f in files) and self._confirm_dua():
            with util.download_tmp(_SOURCES['test-qrels'], tarf=True) as f:
                for f_out, f_in in files.items():
                    self._init_file_copy(f.extractfile(f_in), f_out, force)

        # TEST200

        files = {
            base / 'test200.auto.qrels': 'test200/train.test200.cbor.hierarchical.qrels',
            base / 'test200.queries.tsv': 'test200/train.test200.cbor.outlines',
        }
        if force or not all(f.exists() for f in files) and self._confirm_dua():
            with util.download_tmp(_SOURCES['test200'], tarf=True) as f:
                for f_out, f_in in files.items():
                    if f_out.suffix == '.qrels':
                        self._init_file_copy(f.extractfile(f_in), f_out, force)
                    elif f_out.suffix == '.tsv':
                        self._init_queryfile(f.extractfile(f_in), f_out, force)


    def _init_doc_iter(self):
        with util.download_tmp(_SOURCES['corpus'], tarf=True) as f:
            cbor_file = f.extractfile('paragraphcorpus/paragraphcorpus.cbor')
            for did, text in self.logger.pbar(car.iter_paras(cbor_file), desc='documents'):
                yield indices.RawDoc(did, text)

    def _init_queryfile(self, in_stream, out_path, force=False):
        if force or not os.path.exists(out_path):
            with util.finialized_file(out_path, 'wt') as out:
                with self.logger.duration(f'extracting to {out_path}'):
                    for qid, headings in car.iter_queries(in_stream):
                        plaintext.write_tsv(out, [(qid, *headings)])

    def _init_file_copy(self, f_in, f_out, force=False):
        if force or not os.path.exists(f_out):
            with util.finialized_file(f_out, 'wb') as f:
                with self.logger.duration(f'extracting to {f_out}'):
                    for block in iter(lambda: f_in.read(2048), b''):
                        f.write(block)
