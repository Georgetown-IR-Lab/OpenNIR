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
import onir
from onir import util, datasets, indices
from onir.interfaces import trec, plaintext


_SOURCES = {
    'collection': 'https://msmarco.blob.core.windows.net/msmarcoranking/collection.tar.gz',
    'queries': 'https://msmarco.blob.core.windows.net/msmarcoranking/queries.tar.gz',
    'train-qrels': 'https://msmarco.blob.core.windows.net/msmarcoranking/qrels.train.tsv',
    'dev-qrels': 'https://msmarco.blob.core.windows.net/msmarcoranking/qrels.dev.tsv',
    # 'qidpidtriples.train.full': 'https://msmarco.blob.core.windows.net/msmarcoranking/qidpidtriples.train.full.tar.gz',
    # seems the qidpidtriples.train.full link is broken... I'll host a mirror until they fix
    'qidpidtriples.train.full': 'https://macavaney.us/misc/qidpidtriples.train.full.tar.gz',
    'train.msrun': 'https://msmarco.blob.core.windows.net/msmarcoranking/top1000.train.tar.gz',
    'dev.msrun': 'https://msmarco.blob.core.windows.net/msmarcoranking/top1000.dev.tar.gz',
    'eval.msrun': 'https://msmarco.blob.core.windows.net/msmarcoranking/top1000.eval.tar.gz',
    'trec2019.queries': 'https://msmarco.blob.core.windows.net/msmarcoranking/msmarco-test2019-queries.tsv.gz',
    'trec2019.msrun': 'https://msmarco.blob.core.windows.net/msmarcoranking/msmarco-passagetest2019-top1000.tsv.gz',
    'trec2019.qrels': 'https://trec.nist.gov/data/deep/2019qrels-pass.txt',
    'doctttttquery-predictions': 'https://storage.googleapis.com/doctttttquery_git/predicted_queries_topk_sampling.zip',
}

_HASHES = {
    'collection': "87dd01826da3e2ad45447ba5af577628",
    'queries': "c177b2795d5f2dcc524cf00fcd973be1",
    'train-qrels': "733fb9fe12d93e497f7289409316eccf",
    'dev-qrels': "9157ccaeaa8227f91722ba5770787b16",
    'qidpidtriples.train.full': "215a5204288820672f5e9451d9e202c5",
    'train.msrun': "d99fdbd5b2ea84af8aa23194a3263052",
    'dev.msrun': "8c140662bdf123a98fbfe3bb174c5831",
    'eval.msrun': "73778cd99f6e0632d12d0b5731b20a02",
    'trec2019.queries': "eda71eccbe4d251af83150abe065368c",
    'trec2019.msrun': "ec9e012746aa9763c7ff10b3336a3ce1",
    'trec2019.qrels': "2f4be390198da108f6845c822e5ada14",
    'doctttttquery-predictions': "8bb33ac317e76385d5047322db9b9c34",
}

MINI_DEV = {'484694', '836399', '683975', '428803', '1035062', '723895', '267447', '325379', '582244', '148817', '44209', '1180950', '424238', '683835', '701002', '1076878', '289809', '161771', '807419', '530982', '600298', '33974', '673484', '1039805', '610697', '465983', '171424', '1143723', '811440', '230149', '23861', '96621', '266814', '48946', '906755', '1142254', '813639', '302427', '1183962', '889417', '252956', '245327', '822507', '627304', '835624', '1147010', '818560', '1054229', '598875', '725206', '811871', '454136', '47069', '390042', '982640', '1174500', '816213', '1011280', '368335', '674542', '839790', '270629', '777692', '906062', '543764', '829102', '417947', '318166', '84031', '45682', '1160562', '626816', '181315', '451331', '337653', '156190', '365221', '117722', '908661', '611484', '144656', '728947', '350999', '812153', '149680', '648435', '274580', '867810', '101999', '890661', '17316', '763438', '685333', '210018', '600923', '1143316', '445800', '951737', '1155651', '304696', '958626', '1043094', '798480', '548097', '828870', '241538', '337392', '594253', '1047678', '237264', '538851', '126690', '979598', '707766', '1160366', '123055', '499590', '866943', '18892', '93927', '456604', '560884', '370753', '424562', '912736', '155244', '797512', '584995', '540814', '200926', '286184', '905213', '380420', '81305', '749773', '850038', '942745', '68689', '823104', '723061', '107110', '951412', '1157093', '218549', '929871', '728549', '30937', '910837', '622378', '1150980', '806991', '247142', '55840', '37575', '99395', '231236', '409162', '629357', '1158250', '686443', '1017755', '1024864', '1185054', '1170117', '267344', '971695', '503706', '981588', '709783', '147180', '309550', '315643', '836817', '14509', '56157', '490796', '743569', '695967', '1169364', '113187', '293255', '859268', '782494', '381815', '865665', '791137', '105299', '737381', '479590', '1162915', '655989', '292309', '948017', '1183237', '542489', '933450', '782052', '45084', '377501', '708154'}


@datasets.register('msmarco')
class MsmarcoDataset(datasets.IndexBackedDataset):
    """
    Interface to the MS-MARCO ranking dataset.
     > Tri Nguyen, Mir Rosenberg, Xia Song, Jianfeng Gao, Saurabh Tiwary, RanganMajumder, and Li
     > Deng. 2016.  MS MARCO: A Human Generated MAchineReading COmprehension Dataset. InCoCo@NIPS.
    """
    DUA = """Will begin downloading MS-MARCO dataset.
Please confirm you agree to the authors' data usage stipulations found at
http://www.msmarco.org/dataset.aspx"""

    @staticmethod
    def default_config():
        result = datasets.IndexBackedDataset.default_config()
        result.update({
            'subset': onir.config.Choices(['train', 'train10', 'train_med', 'dev', 'minidev', 'judgeddev', 'eval', 'trec2019', 'judgedtrec2019']),
            'rankfn': onir.config.Ranker(),
            'ranktopk': 100,
            'special': onir.config.Choices(['', 'mspairs', 'msrun', 'validrun']),
            'index': onir.config.Choices(['default', 'doctttttquery']),
            'init_skip_train10': True,
            'init_skip_train_med': True,
            'init_skip_msrun': True,
            'init_skip_doctttttquery': True,
            # validrun made with `cat ~/data/onir/datasets/msmarco/anserini.porter.minidev.runs/bm25.1000.run ~/data/onir/datasets/msmarco/minidev.qrels | awk 'NF==4||(NF==6&&($4==1||$4==2||$4==3||$4==4||$4==6||$4==8||$4==11||$4==16||$4==22||$4==31||$4==43||$4==60||$4==83||$4==116||$4==162||$4==227||$4==316||$4==441||$4==616||$4==859)){print $1, $3}' | sort | uniq`
        })
        return result

    def __init__(self, config, logger, vocab):
        super().__init__(config, logger, vocab)
        base_path = util.path_dataset(self)
        self.index_stem = indices.AnseriniIndex(os.path.join(base_path, 'anserini.porter'), stemmer='porter')
        self.index_doctttttquery_stem = indices.AnseriniIndex(os.path.join(base_path, 'anserini.doctttttquery.porter'), stemmer='porter')
        self.doc_store = indices.SqliteDocstore(os.path.join(base_path, 'docs.sqllite'))

    def _get_docstore(self):
        return self.doc_store

    def _get_index(self, record):
        return self.index_stem

    def _get_index_for_batchsearch(self):
        return {
            'default': self.index_stem,
            'doctttttquery': self.index_doctttttquery_stem,
        }[self.config['index']]

    def qrels(self, fmt='dict'):
        return self._load_qrels(self.config['subset'], fmt=fmt)

    @memoize_method
    def _load_qrels(self, subset, fmt):
        with self.logger.duration('loading qrels'):
            base_path = util.path_dataset(self)
            path = os.path.join(base_path, f'{subset}.qrels')
            return trec.read_qrels_fmt(path, fmt)

    def load_queries(self) -> dict:
        return self._load_queries_base(self.config['subset'])

    @memoize_method
    def _load_queries_base(self, subset):
        base_path = util.path_dataset(self)
        path = os.path.join(base_path, f'{subset}.queries.tsv')
        return dict(self.logger.pbar(plaintext.read_tsv(path), desc='loading queries'))

    def pair_iter(self, fields, pos_source='intersect', neg_source='run', sampling='query', pos_minrel=1, unjudged_rel=0, num_neg=1, random=None, inf=False):
        special = self.config['special']
        if special == '':
            raise NotImplementedError
        assert pos_minrel == 1, f"{special} only supports pos_minrel=1"
        assert unjudged_rel == 0, f"{special} only supports unjudged_rel=1"
        assert num_neg == 1, f"{special} only supports num_neg=1"
        assert self.config['subset'] in ('train', 'train10'), f"{special} only supported with subset=train[10]"
        self.logger.warn(f'Using {special}; ingoring pair_iter arguments pos_source={pos_source} neg_source={neg_source} sampling={sampling}')
        first = True
        while first or inf:
            first = False
            if special == 'mspairs':
                f = gzip.open(os.path.join(util.path_dataset(self), '{subset}.mspairs.gz'.format(**self.config)), 'rt')
            else:
                raise ValueError(f'unsupported special={special}')
            with f:
                for qid, pos_did, neg_did in plaintext.read_tsv(f):
                    if qid in MINI_DEV:
                        continue
                    result = {f: [] for f in fields}
                    for did in [pos_did, neg_did]:
                        record = self.build_record(fields, query_id=qid, doc_id=did)
                        for f in fields:
                            result[f].append(record[f])
                    yield result

    def record_iter(self, fields, source, minrel=None, shuf=True, random=None, inf=False, run_threshold=None):
        special = self.config['special']
        if special == '':
            raise NotImplementedError
        assert minrel is None or minrel < 1
        if source != 'run':
            self.logger.warn(f'Using special={special}; ingoring record_iter arguments source={source}')
        if run_threshold is not None:
            self.logger.warn(f'Using special={special}; ingoring record_iter arguments run_threshold={run_threshold}')
        first = True
        while first or inf:
            first = False
            if special == 'mspairs':
                f = gzip.open(os.path.join(util.path_dataset(self), '{subset}.mspairs.gz'.format(**self.config)), 'rt')
                it = plaintext.read_tsv(f)
                fields = fields - {'relscore'} # don't request relscore from typical channels (i.e., qrels) because we already know and this is faster.
            elif special == 'msrun':
                f = os.path.join(util.path_dataset(self), '{subset}.msrun'.format(**self.config))
                it = ((qid, did) for qid, did, rank, score in trec.read_run(f))
            elif special == 'validrun':
                f = os.path.join(util.path_dataset(self), '{subset}.validrun'.format(**self.config))
                it = plaintext.read_sv(f, ' ')
            else:
                raise ValueError(f'unsupported special={special}')
            if shuf:
                if special in ('msrun', 'mspairs'):
                    self.logger.warn(f'ignoring shuf=True with special={special}')
                else:
                    it = list(it)
                    random.shuffle(it)
            for cols in it:
                if len(cols) == 3:
                    qid, pos_did, neg_did = cols
                    dids = [pos_did, neg_did] if (minrel is None or minrel <= 0) else [pos_did]
                    if qid in MINI_DEV:
                        continue
                elif len(cols) == 2:
                    qid, did = cols
                    dids = [did]
                for did in dids:
                    record = self.build_record(fields, query_id=qid, doc_id=did)
                    result = {f: record[f] for f in fields}
                    if len(cols) == 3:
                        result['relscore'] = (1 if did == pos_did else 0)
                    yield result

    def path_segment(self):
        result = super().path_segment()
        if self.config['special'] != '':
            result += '_{special}'.format(**self.config)
        if self.config['index'] != 'default':
            result += '_{index}'.format(**self.config)
        return result

    def init(self, force=False):
        idxs = [self.index_stem, self.doc_store]
        self._init_indices_parallel(idxs, self._init_iter_collection(), force)
        if not self.config['init_skip_doctttttquery']:
            self._init_indices_parallel([self.index_doctttttquery_stem], self._init_doctttttquery_iter(), force)

        base_path = util.path_dataset(self)

        needs_queries = []
        if force or not os.path.exists(os.path.join(base_path, 'train.queries.tsv')):
            needs_queries.append(lambda it: plaintext.write_tsv(os.path.join(base_path, 'train.queries.tsv'), ((qid, txt) for file, qid, txt in it if file == 'queries.train.tsv' and qid not in MINI_DEV)))
        if force or not os.path.exists(os.path.join(base_path, 'minidev.queries.tsv')):
            needs_queries.append(lambda it: plaintext.write_tsv(os.path.join(base_path, 'minidev.queries.tsv'), ((qid, txt) for file, qid, txt in it if file == 'queries.train.tsv' and qid in MINI_DEV)))
        if force or not os.path.exists(os.path.join(base_path, 'dev.queries.tsv')):
            needs_queries.append(lambda it: plaintext.write_tsv(os.path.join(base_path, 'dev.queries.tsv'), ((qid, txt) for file, qid, txt in it if file == 'queries.dev.tsv')))
        if force or not os.path.exists(os.path.join(base_path, 'eval.queries.tsv')):
            needs_queries.append(lambda it: plaintext.write_tsv(os.path.join(base_path, 'eval.queries.tsv'), ((qid, txt) for file, qid, txt in it if file == 'queries.eval.tsv')))

        if needs_queries and self._confirm_dua():
            with util.download_tmp(_SOURCES['queries'], expected_md5=_HASHES['queries']) as f, \
                 tarfile.open(fileobj=f) as tarf, \
                 contextlib.ExitStack() as ctxt:
                def _extr_subf(subf):
                    for qid, txt in plaintext.read_tsv(io.TextIOWrapper(tarf.extractfile(subf))):
                        yield subf, qid, txt
                query_iter = [_extr_subf('queries.train.tsv'), _extr_subf('queries.dev.tsv'), _extr_subf('queries.eval.tsv')]
                query_iter = tqdm(itertools.chain(*query_iter), desc='queries')
                query_iters = util.blocking_tee(query_iter, len(needs_queries))
                for fn, it in zip(needs_queries, query_iters):
                    ctxt.enter_context(util.CtxtThread(functools.partial(fn, it)))

        file = os.path.join(base_path, 'train.qrels')
        if (force or not os.path.exists(file)) and self._confirm_dua():
            stream = util.download_stream(_SOURCES['train-qrels'], 'utf8', expected_md5=_HASHES['train-qrels'])
            with util.finialized_file(file, 'wt') as out:
                for qid, _, did, score in plaintext.read_tsv(stream):
                    if qid not in MINI_DEV:
                        trec.write_qrels(out, [(qid, did, score)])

        file = os.path.join(base_path, 'minidev.qrels')
        if (force or not os.path.exists(file)) and self._confirm_dua():
            with util.finialized_file(file, 'wt') as out:
                for qid, did, score in trec.read_qrels(os.path.join(base_path, 'train.qrels')):
                    if qid in MINI_DEV:
                        trec.write_qrels(out, [(qid, did, score)])

        file = os.path.join(base_path, 'dev.qrels')
        if (force or not os.path.exists(file)) and self._confirm_dua():
            stream = util.download_stream(_SOURCES['dev-qrels'], 'utf8', expected_md5=_HASHES['dev-qrels'])
            with util.finialized_file(file, 'wt') as out:
                for qid, _, did, score in plaintext.read_tsv(stream):
                    trec.write_qrels(out, [(qid, did, score)])

        file = os.path.join(base_path, 'train.mspairs.gz')
        if not os.path.exists(file) and os.path.exists(os.path.join(base_path, 'qidpidtriples.train.full')):
            # legacy
            os.rename(os.path.join(base_path, 'qidpidtriples.train.full'), file)
        if (force or not os.path.exists(file)) and self._confirm_dua():
            util.download(_SOURCES['qidpidtriples.train.full'], file, expected_md5=_HASHES['qidpidtriples.train.full'])

        if not self.config['init_skip_msrun']:
            for file_name, subf in [('dev.msrun', 'top1000.dev'), ('eval.msrun', 'top1000.eval'), ('train.msrun', 'top1000.train.txt')]:
                file = os.path.join(base_path, file_name)
                if (force or not os.path.exists(file)) and self._confirm_dua():
                    run = {}
                    with util.download_tmp(_SOURCES[file_name], expected_md5=_HASHES[file_name]) as f, \
                         tarfile.open(fileobj=f) as tarf:
                        for qid, did, _, _ in tqdm(plaintext.read_tsv(io.TextIOWrapper(tarf.extractfile(subf)))):
                            if qid not in run:
                                run[qid] = {}
                            run[qid][did] = 0.
                    if file_name == 'train.msrun':
                        minidev = {qid: dids for qid, dids in run.items() if qid in MINI_DEV}
                        with self.logger.duration('writing minidev.msrun'):
                            trec.write_run_dict(os.path.join(base_path, 'minidev.msrun'), minidev)
                        run = {qid: dids for qid, dids in run.items() if qid not in MINI_DEV}
                    with self.logger.duration(f'writing {file_name}'):
                        trec.write_run_dict(file, run)

        query_path = os.path.join(base_path, 'trec2019.queries.tsv')
        if (force or not os.path.exists(query_path)) and self._confirm_dua():
            stream = util.download_stream(_SOURCES['trec2019.queries'], 'utf8', expected_md5=_HASHES['trec2019.queries'])
            plaintext.write_tsv(query_path, plaintext.read_tsv(stream))
        msrun_path = os.path.join(base_path, 'trec2019.msrun')
        if (force or not os.path.exists(msrun_path)) and self._confirm_dua():
            run = {}
            with util.download_stream(_SOURCES['trec2019.msrun'], 'utf8', expected_md5=_HASHES['trec2019.msrun']) as stream:
                for qid, did, _, _ in plaintext.read_tsv(stream):
                    if qid not in run:
                        run[qid] = {}
                    run[qid][did] = 0.
            with util.finialized_file(msrun_path, 'wt') as f:
                trec.write_run_dict(f, run)

        qrels_path = os.path.join(base_path, 'trec2019.qrels')
        if not os.path.exists(qrels_path) and self._confirm_dua():
            util.download(_SOURCES['trec2019.qrels'], qrels_path, expected_md5=_HASHES['trec2019.qrels'])
        qrels_path = os.path.join(base_path, 'judgedtrec2019.qrels')
        if not os.path.exists(qrels_path):
            os.symlink('trec2019.qrels', qrels_path)
        query_path = os.path.join(base_path, 'judgedtrec2019.queries.tsv')
        judged_qids = util.Lazy(lambda: trec.read_qrels_dict(qrels_path).keys())
        if (force or not os.path.exists(query_path)):
            with util.finialized_file(query_path, 'wt') as f:
                for qid, qtext in plaintext.read_tsv(os.path.join(base_path, 'trec2019.queries.tsv')):
                    if qid in judged_qids():
                        plaintext.write_tsv(f, [(qid, qtext)])
        msrun_path = os.path.join(base_path, 'judgedtrec2019.msrun')
        if (force or not os.path.exists(msrun_path)) and self._confirm_dua():
            with util.finialized_file(msrun_path, 'wt') as f:
                for qid, dids in trec.read_run_dict(os.path.join(base_path, 'trec2019.msrun')).items():
                    if qid in judged_qids():
                        trec.write_run_dict(f, {qid: dids})

        # A subset of dev that only contains queries that have relevance judgments
        judgeddev_path = os.path.join(base_path, 'judgeddev')
        judged_qids = util.Lazy(lambda: trec.read_qrels_dict(os.path.join(base_path, 'dev.qrels')).keys())
        if not os.path.exists(f'{judgeddev_path}.qrels'):
            os.symlink('dev.qrels', f'{judgeddev_path}.qrels')
        if not os.path.exists(f'{judgeddev_path}.queries.tsv'):
            with util.finialized_file(f'{judgeddev_path}.queries.tsv', 'wt') as f:
                for qid, qtext in plaintext.read_tsv(os.path.join(base_path, 'dev.queries.tsv')):
                    if qid in judged_qids():
                        plaintext.write_tsv(f, [(qid, qtext)])
        if not self.config['init_skip_msrun']:
            if not os.path.exists(f'{judgeddev_path}.msrun'):
                with util.finialized_file(f'{judgeddev_path}.msrun', 'wt') as f:
                    for qid, dids in trec.read_run_dict(os.path.join(base_path, 'dev.msrun')).items():
                        if qid in judged_qids():
                            trec.write_run_dict(f, {qid: dids})

        if not self.config['init_skip_train10']:
            file = os.path.join(base_path, 'train10.queries.tsv')
            if not os.path.exists(file):
                with util.finialized_file(file, 'wt') as fout:
                    for qid, qtext in self.logger.pbar(plaintext.read_tsv(os.path.join(base_path, 'train.queries.tsv')), desc='filtering queries for train10'):
                        if int(qid) % 10 == 0:
                            plaintext.write_tsv(fout, [(qid, qtext)])

            file = os.path.join(base_path, 'train10.qrels')
            if not os.path.exists(file):
                with util.finialized_file(file, 'wt') as fout, open(os.path.join(base_path, 'train.qrels'), 'rt') as fin:
                    for line in self.logger.pbar(fin, desc='filtering qrels for train10'):
                        qid = line.split()[0]
                        if int(qid) % 10 == 0:
                            fout.write(line)

            if not self.config['init_skip_msrun']:
                file = os.path.join(base_path, 'train10.msrun')
                if not os.path.exists(file):
                    with util.finialized_file(file, 'wt') as fout, open(os.path.join(base_path, 'train.msrun'), 'rt') as fin:
                        for line in self.logger.pbar(fin, desc='filtering msrun for train10'):
                            qid = line.split()[0]
                            if int(qid) % 10 == 0:
                                fout.write(line)

            file = os.path.join(base_path, 'train10.mspairs.gz')
            if not os.path.exists(file):
                with gzip.open(file, 'wt') as fout, gzip.open(os.path.join(base_path, 'train.mspairs.gz'), 'rt') as fin:
                    for qid, did1, did2 in self.logger.pbar(plaintext.read_tsv(fin), desc='filtering mspairs for train10'):
                        if int(qid) % 10 == 0:
                            plaintext.write_tsv(fout, [(qid, did1, did2)])

        if not self.config['init_skip_train_med']:
            med_qids = util.Lazy(lambda: {qid.strip() for qid in util.download_stream('https://raw.githubusercontent.com/Georgetown-IR-Lab/covid-neural-ir/master/med-msmarco-train.txt', 'utf8', expected_md5="dc5199de7d4a872c361f89f08b1163ef")})
            file = os.path.join(base_path, 'train_med.queries.tsv')
            if not os.path.exists(file):
                with util.finialized_file(file, 'wt') as fout:
                    for qid, qtext in self.logger.pbar(plaintext.read_tsv(os.path.join(base_path, 'train.queries.tsv')), desc='filtering queries for train_med'):
                        if qid in med_qids():
                            plaintext.write_tsv(fout, [(qid, qtext)])

            file = os.path.join(base_path, 'train_med.qrels')
            if not os.path.exists(file):
                with util.finialized_file(file, 'wt') as fout, open(os.path.join(base_path, 'train.qrels'), 'rt') as fin:
                    for line in self.logger.pbar(fin, desc='filtering qrels for train_med'):
                        qid = line.split()[0]
                        if qid in med_qids():
                            fout.write(line)

            if not self.config['init_skip_msrun']:
                file = os.path.join(base_path, 'train_med.msrun')
                if not os.path.exists(file):
                    with util.finialized_file(file, 'wt') as fout, open(os.path.join(base_path, 'train.msrun'), 'rt') as fin:
                        for line in self.logger.pbar(fin, desc='filtering msrun for train_med'):
                            qid = line.split()[0]
                            if qid in med_qids():
                                fout.write(line)

            file = os.path.join(base_path, 'train_med.mspairs.gz')
            if not os.path.exists(file):
                with gzip.open(file, 'wt') as fout, gzip.open(os.path.join(base_path, 'train.mspairs.gz'), 'rt') as fin:
                    for qid, did1, did2 in self.logger.pbar(plaintext.read_tsv(fin), desc='filtering mspairs for train_med'):
                        if qid in med_qids():
                            plaintext.write_tsv(fout, [(qid, did1, did2)])

    def _init_iter_collection(self):
        with util.download_tmp(_SOURCES['collection'], expected_md5=_HASHES['collection']) as f:
            with tarfile.open(fileobj=f) as tarf:
                collection_stream = io.TextIOWrapper(tarf.extractfile('collection.tsv'))
                for did, text in self.logger.pbar(plaintext.read_tsv(collection_stream), desc='documents'):
                    yield indices.RawDoc(did, text)

    def _init_doctttttquery_iter(self):
        with util.download_tmp(_SOURCES['doctttttquery-predictions'], expected_md5=_HASHES['doctttttquery-predictions']) as f1, \
             util.download_tmp(_SOURCES['collection'], expected_md5=_HASHES['collection']) as f2:
            with zipfile.ZipFile(f1) as zipf, tarfile.open(fileobj=f2) as tarf:
                collection_stream = io.TextIOWrapper(tarf.extractfile('collection.tsv'))
                d5_iter = self._init_doctttttquery_zipf_iter(zipf)
                for (did, text), d5text in self.logger.pbar(zip(plaintext.read_tsv(collection_stream), d5_iter), desc='documents'):
                    yield indices.RawDoc(did, f'{text} {d5text}')

    def _init_doctttttquery_zipf_iter(self, zipf):
        FILE_FMT = 'predicted_queries_topk_sample{:03d}.txt{:03d}-1004000'
        file_num = 0
        NUM_PREDS = 40 # following convention in <https://github.com/castorini/docTTTTTquery>: predicted_queries_topk_sample0[0-3]?.txt${i}-1004000
        subfiles = [io.TextIOWrapper(zipf.open(FILE_FMT.format(i, file_num))) for i in range(NUM_PREDS)]
        while True:
            try:
                yield ' '.join([next(f).rstrip() for f in subfiles])
            except StopIteration:
                file_num += 1
                subfiles = [io.TextIOWrapper(zipf.open(FILE_FMT.format(i, file_num))) for i in range(NUM_PREDS)]
