import os
from pytools import memoize_method
from onir import datasets, util, indices
from onir.interfaces import trec, plaintext


# from <https://github.com/faneshion/DRMM/blob/9d348640ef8a56a8c1f2fa0754fe87d8bb5785bd/NN4IR.cpp>
FOLDS = {
    'f1': {'302', '303', '309', '316', '317', '319', '323', '331', '336', '341', '356', '357', '370', '373', '378', '381', '383', '392', '394', '406', '410', '411', '414', '426', '428', '433', '447', '448', '601', '607', '608', '612', '617', '619', '635', '641', '642', '646', '647', '654', '656', '662', '665', '669', '670', '679', '684', '690', '692', '700'},
    'f2': {'301', '308', '312', '322', '327', '328', '338', '343', '348', '349', '352', '360', '364', '365', '369', '371', '374', '386', '390', '397', '403', '419', '422', '423', '424', '432', '434', '440', '446', '602', '604', '611', '623', '624', '627', '632', '638', '643', '651', '652', '663', '674', '675', '678', '680', '683', '688', '689', '695', '698'},
    'f3': {'306', '307', '313', '321', '324', '326', '334', '347', '351', '354', '358', '361', '362', '363', '376', '380', '382', '396', '404', '413', '415', '417', '427', '436', '437', '439', '444', '445', '449', '450', '603', '605', '606', '614', '620', '622', '626', '628', '631', '637', '644', '648', '661', '664', '666', '671', '677', '685', '687', '693'},
    'f4': {'320', '325', '330', '332', '335', '337', '342', '344', '350', '355', '368', '377', '379', '387', '393', '398', '402', '405', '407', '408', '412', '420', '421', '425', '430', '431', '435', '438', '616', '618', '625', '630', '633', '636', '639', '649', '650', '653', '655', '657', '659', '667', '668', '672', '673', '676', '682', '686', '691', '697'},
    'f5': {'304', '305', '310', '311', '314', '315', '318', '329', '333', '339', '340', '345', '346', '353', '359', '366', '367', '372', '375', '384', '385', '388', '389', '391', '395', '399', '400', '401', '409', '416', '418', '429', '441', '442', '443', '609', '610', '613', '615', '621', '629', '634', '640', '645', '658', '660', '681', '694', '696', '699'}
}

_ALL = set.union(*FOLDS.values())
_FOLD_IDS = list(sorted(FOLDS.keys()))
for i in range(len(FOLDS)):
    FOLDS['tr' + _FOLD_IDS[i]] = _ALL - FOLDS[_FOLD_IDS[i]] - FOLDS[_FOLD_IDS[i-1]]
    FOLDS['va' + _FOLD_IDS[i]] = FOLDS[_FOLD_IDS[i-1]]
FOLDS['all'] = _ALL


_FILES = {
    'index': dict(url='https://git.uwaterloo.ca/jimmylin/anserini-indexes/raw/master/index-robust04-20191213.tar.gz', expected_sha256="dddb81f16d70ea6b9b0f94d6d6b888ed2ef827109a14ca21fd82b2acd6cbd450"),
    'queries': dict(url='https://trec.nist.gov/data/robust/04.testset.gz', expected_sha256="df769c0b455970168a4dd0dfb36f0d809d102d1bfcda566511e980c2355ce77f"),
    'qrels': dict(url='https://trec.nist.gov/data/robust/qrels.robust2004.txt', expected_sha256="f8f2c972d3c710d85daa7ead02daf4ffe2bbe3647c9f3904500182f43ddbf4c3"),
}


@datasets.register('robust')
class RobustDataset(datasets.IndexBackedDataset):
    """
    Interface to the TREC Robust 2004 dataset.
     > Ellen M. Voorhees. 2004. Overview of TREC 2004. In TREC.
    """
    DUA = """Will begin downloading Robust04 dataset.
Please confirm you agree to the authors' data usage stipulations found at
https://trec.nist.gov/data/cd45/index.html"""

    @staticmethod
    def default_config():
        result = datasets.IndexBackedDataset.default_config()
        result.update({
            'subset': 'all',
            'ranktopk': 100
        })
        return result

    def __init__(self, config, logger, vocab):
        super().__init__(config, logger, vocab)
        base_path = util.path_dataset(self)
        self.index = indices.AnseriniIndex(os.path.join(base_path, 'anserini'), stemmer='none')
        self.index_stem = indices.AnseriniIndex(os.path.join(base_path, 'anserini.porter'), stemmer='porter')
        self.doc_store = indices.SqliteDocstore(os.path.join(base_path, 'docs.sqllite'))

    def _get_index(self, record):
        return self.index

    def _get_docstore(self):
        return self.doc_store

    def _get_index_for_batchsearch(self):
        return self.index_stem

    @memoize_method
    def _load_queries_base(self, subset):
        topics = self._load_topics()
        result = {}
        for qid in FOLDS[subset]:
            result[qid] = topics[qid]
        return result

    def qrels(self, fmt='dict'):
        return self._load_qrels(self.config['subset'], fmt)

    @memoize_method
    def _load_qrels(self, subset, fmt):
        return trec.read_qrels_fmt(os.path.join(util.path_dataset(self), f'{subset}.qrels'), fmt)

    @memoize_method
    def _load_topics(self):
        result = {}
        for item, qid, text in plaintext.read_tsv(os.path.join(util.path_dataset(self), 'topics.txt')):
            if item == 'topic':
                result[qid] = text
        return result

    def init(self, force=False):
        base_path = util.path_dataset(self)
        idxs = [self.index, self.index_stem, self.doc_store]
        self._init_indices_parallel(idxs, self._init_iter_collection(), force)

        qrels_file = os.path.join(base_path, 'qrels.robust2004.txt')
        if (force or not os.path.exists(qrels_file)) and self._confirm_dua():
            util.download(**_FILES['qrels'], file_name=qrels_file)

        for fold in FOLDS:
            fold_qrels_file = os.path.join(base_path, f'{fold}.qrels')
            if (force or not os.path.exists(fold_qrels_file)):
                all_qrels = trec.read_qrels_dict(qrels_file)
                fold_qrels = {qid: dids for qid, dids in all_qrels.items() if qid in FOLDS[fold]}
                trec.write_qrels_dict(fold_qrels_file, fold_qrels)

        query_file = os.path.join(base_path, 'topics.txt')
        if (force or not os.path.exists(query_file)) and self._confirm_dua():
            query_file_stream = util.download_stream(**_FILES['queries'], encoding='utf8')
            plaintext.write_tsv(query_file, trec.parse_query_format(query_file_stream))

    def _init_iter_collection(self):
        # Using the trick here from capreolus, pulling document content out of public index:
        # <https://github.com/capreolus-ir/capreolus/blob/d6ae210b24c32ff817f615370a9af37b06d2da89/capreolus/collection/robust04.yaml#L15>
        with util.download_tmp(**_FILES['index']) as f:
            fd = f'{f.name}.d'
            util.extract_tarball(f.name, fd, self.logger, reset_permissions=True)
            index = indices.AnseriniIndex(f'{fd}/index-robust04-20191213')
            for did in self.logger.pbar(index.docids(), desc='documents'):
                raw_doc = index.get_raw(did)
                yield indices.RawDoc(did, raw_doc)
