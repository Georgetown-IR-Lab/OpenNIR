import os
from onir import util, datasets, indices


@datasets.register('trec_mandarin')
class TrecMandarinDataset(datasets.MultilingualDataset):
    """
    Interface to the TREC Mandarin datasets. Used for cross-lingual experiments in:
     > Sean MacAvaney, Luca Soldaini, Nazli Goharian. Teaching a New Dog Old Tricks: Resurrecting
     > Multilingual Retrieval Using Zero-shot Learning. In ECIR 2020.
    """
    DUA = """Will begin downloading TREC Mandarin dataset.
You first need a copy of LDC2000T52, which can be found at:
https://catalog.ldc.upenn.edu/LDC2000T52

Then copy/link document collection to:
{ds_path}/trec_mandarin

Please confirm you agree to the authors' data usage stipulations found at
https://catalog.ldc.upenn.edu/LDC2000T52
and
https://trec.nist.gov/data/testq_noneng.html"""

    @staticmethod
    def default_config():
        result = datasets.MultilingualDataset.default_config()
        result.update({
            'subset': 'trec5',
            'ranktopk': 1000,
            'querysource': 'topic'
        })
        return result

    def __init__(self, config, vocab, logger):
        super().__init__(config, logger, vocab)
        self.index_mandarin = indices.AnseriniIndex(os.path.join(util.path_dataset(self), 'anserini.zh'), lang=self._lang())
        self.doc_store = indices.SqliteDocstore(os.path.join(util.path_dataset(self), 'docs.sqlite'))

    def _get_docstore(self):
        return self.doc_store

    def _lang(self):
        return 'zh'

    def _get_index_for_batchsearch(self):
        return self.index_mandarin

    def init(self, force=False):
        self._init_topics(
            subset='trec5',
            topic_files=['https://trec.nist.gov/data/topics_noneng/topics.CH1-CH28.chinese.english.gz'],
            qid_prefix='CH',
            encoding="GBK",
            xml_prefix='C-',
            expected_md5="9ce885d36e8642d4114f40e7008e5b8a",
            force=force)

        self._init_topics(
            subset='trec6',
            topic_files=['https://trec.nist.gov/data/topics_noneng/topics.CH29-CH54.chinese.english.gz'],
            qid_prefix='CH',
            encoding="GBK",
            xml_prefix='C-',
            expected_md5="c3a58ec59e55c162fdc3e3a9c5e9b8a7",
            force=force)

        self._init_qrels(
            subset='trec5',
            qrels_files=['https://trec.nist.gov/data/qrels_noneng/qrels.1-28.chinese.gz'],
            expected_md5="73693083d75ef323fca2a218604b41ac",
            force=force)

        self._init_qrels(
            subset='trec6',
            qrels_files=['https://trec.nist.gov/data/qrels_noneng/qrels.trec6.29-54.chinese.gz'],
            expected_md5="675ab2f14fad9017d646d052c0b35c46",
            force=force)

        self._init_indices_parallel(
            indices=[self.index_mandarin, self.doc_store],
            doc_iter=self._init_collection_iter(
                doc_paths=['trec_mandarin/xinhua', 'trec_mandarin/peoples-daily'],
                encoding="GB18030"),
            force=force)
