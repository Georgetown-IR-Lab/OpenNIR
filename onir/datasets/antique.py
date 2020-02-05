import os
from pytools import memoize_method
from onir import datasets, util, indices
from onir.interfaces import trec, plaintext


# from `cat antique-train-queries.txt | gshuf | head -n200 | awk '{print $1}'`
VALIDATION_QIDS = {'1158088', '4032777', '1583099', '263783', '4237144', '1097878', '114758', '1211877', '1188438', '2689609', '1191621', '2571912', '1471877', '2961191', '2630860', '4092472', '3178012', '358253', '3913653', '844617', '2764765', '212427', '220575', '11706', '4069320', '3280274', '3159749', '4217473', '4042061', '1037897', '103298', '332662', '752633', '2704', '3635284', '2235825', '3651236', '2155390', '3752394', '2008456', '98438', '511835', '1647624', '3884772', '1536937', '544869', '66151', '2678635', '963523', '1881436', '993601', '3608433', '2048278', '3124162', '1907320', '1970273', '2891885', '2858043', '189364', '397709', '3470651', '3885753', '1933929', '94629', '2500918', '1708787', '2492366', '17665', '278043', '643630', '1727343', '196651', '3731489', '2910592', '1144768', '2573745', '546552', '1341602', '317469', '2735795', '1251077', '3507499', '3374970', '1034050', '1246269', '2901754', '2137263', '1295284', '2180502', '406082', '1443637', '2620488', '3118286', '3814583', '3738877', '684633', '2094435', '242701', '2613648', '2942624', '1495234', '1440810', '2421078', '961127', '595342', '363519', '4048305', '485408', '2573803', '3104841', '3626847', '727663', '3961', '4287367', '2112535', '913424', '1514356', '1512776', '937635', '1321784', '1582044', '1467322', '461995', '884643', '4338583', '2550445', '4165672', '1016750', '1184520', '3152714', '3617468', '3172166', '4031702', '2534994', '2035638', '404359', '1398838', '4183127', '2418824', '2439070', '2632334', '4262151', '3841762', '4400543', '2147417', '514804', '1423289', '2041828', '2776069', '1458676', '3407617', '1450678', '1978816', '2466898', '1607303', '2175167', '772988', '1289770', '3382182', '3690922', '1051346', '344029', '2357505', '1907847', '2587810', '3272207', '2522067', '1107012', '554539', '489705', '3652886', '4287894', '4387641', '1727879', '348777', '566364', '2678484', '4450252', '986260', '4336509', '3824106', '2169746', '2700836', '3495304', '3083719', '126182', '1607924', '1485589', '3211282', '2546730', '2897078', '3556937', '2113006', '929821', '2306533', '2543919', '1639607', '3958214', '2677193', '763189'}


@datasets.register('antique')
class AntiqueDataset(datasets.IndexBackedDataset):
    """
    Interface to the ANTIQUE dataset.
     > Helia Hashemi, Mohammad Aliannejadi, Hamed Zamani, and W. Bruce Croft. 2019. ANTIQUE: A
     > Non-Factoid Question Answering Benchmark.ArXiv(2019).
    """
    DUA = """Will begin downloading ANTIQUE dataset.
Please confirm you agree to the authors' data usage stipulations found at
https://ciir.cs.umass.edu/downloads/Antique/readme.txt"""

    @staticmethod
    def default_config():
        result = datasets.IndexBackedDataset.default_config()
        result.update({
            'subset': 'train'
        })
        return result

    def __init__(self, config, logger, vocab):
        super().__init__(config, logger, vocab)
        base_path = util.path_dataset(self)
        self.index = indices.AnseriniIndex(os.path.join(base_path, 'anserini'), stemmer='none')
        self.index_stem = indices.AnseriniIndex(os.path.join(base_path, 'anserini.porter'), stemmer='porter')
        self.doc_store = indices.SqliteDocstore(os.path.join(base_path, 'docs.sqlite'))

    def qrels(self, fmt='dict'):
        return self._load_qrels(self.config['subset'], fmt)

    @memoize_method
    def _load_qrels(self, subset, fmt):
        return trec.read_qrels_fmt(os.path.join(util.path_dataset(self), f'{subset}.qrels.txt'), fmt)

    def _get_index(self, record):
        return self.index

    def _get_docstore(self):
        return self.doc_store

    def _get_index_for_batchsearch(self):
        return self.index_stem

    @memoize_method
    def _load_queries_base(self, subset):
        result = {}
        f = os.path.join(util.path_dataset(self), f'{subset}.queries.txt')
        for qid, text in plaintext.read_tsv(f):
            result[qid] = text
        return result

    def init(self, force=False):
        idxs = [self.index, self.index_stem, self.doc_store]
        self._init_indices_parallel(idxs, self._init_iter_collection(), force)

        train_qrels = os.path.join(util.path_dataset(self), 'train.qrels.txt')
        valid_qrels = os.path.join(util.path_dataset(self), 'valid.qrels.txt')
        if (force or not os.path.exists(train_qrels) or not os.path.exists(valid_qrels)) and self._confirm_dua():
            source_stream = util.download_stream('https://ciir.cs.umass.edu/downloads/Antique/antique-train.qrel', encoding='utf8')
            with util.finialized_file(train_qrels, 'wt') as tf, \
                 util.finialized_file(valid_qrels, 'wt') as vf:
                for line in source_stream:
                    cols = line.strip().split()
                    if cols[0] in VALIDATION_QIDS:
                        vf.write(' '.join(cols) + '\n')
                    else:
                        tf.write(' '.join(cols) + '\n')

        train_queries = os.path.join(util.path_dataset(self), 'train.queries.txt')
        valid_queries = os.path.join(util.path_dataset(self), 'valid.queries.txt')
        if (force or not os.path.exists(train_queries) or not os.path.exists(valid_queries)) and self._confirm_dua():
            source_stream = util.download_stream('https://ciir.cs.umass.edu/downloads/Antique/antique-train-queries.txt', encoding='utf8')
            train, valid = [], []
            for cols in plaintext.read_tsv(source_stream):
                if cols[0] in VALIDATION_QIDS:
                    valid.append(cols)
                else:
                    train.append(cols)
            plaintext.write_tsv(train_queries, train)
            plaintext.write_tsv(valid_queries, valid)

        test_qrels = os.path.join(util.path_dataset(self), 'test.qrels.txt')
        if (force or not os.path.exists(test_qrels)) and self._confirm_dua():
            util.download('https://ciir.cs.umass.edu/downloads/Antique/antique-test.qrel', test_qrels)

        test_queries = os.path.join(util.path_dataset(self), 'test.queries.txt')
        if (force or not os.path.exists(test_queries)) and self._confirm_dua():
            util.download('https://ciir.cs.umass.edu/downloads/Antique/antique-test-queries.txt', test_queries)

    def _init_iter_collection(self):
        strm = util.download_stream('https://ciir.cs.umass.edu/downloads/Antique/antique-collection.txt', 'utf8')
        for did, text in plaintext.read_tsv(strm):
            yield indices.RawDoc(did, text)
