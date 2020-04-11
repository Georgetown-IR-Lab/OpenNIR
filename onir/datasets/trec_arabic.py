import os
from onir import util, datasets, indices


@datasets.register('trec_arabic')
class TrecArabicDataset(datasets.MultilingualDataset):
    """
    Interface to the TREC Arabic datasets. Used for cross-lingual experiments in:
     > Sean MacAvaney, Luca Soldaini, Nazli Goharian. Teaching a New Dog Old Tricks: Resurrecting
     > Multilingual Retrieval Using Zero-shot Learning. In ECIR 2020.
    """
    DUA = """Will begin downloading TREC Arabic dataset.
You first need a copy of LDC2001T55, which can be found at:
https://catalog.ldc.upenn.edu/LDC2001T55

Then copy/link document collection to:
{ds_path}/arabic_newswire_a

Please confirm you agree to the authors' data usage stipulations found at
https://catalog.ldc.upenn.edu/LDC2001T55
and
https://trec.nist.gov/data/testq_noneng.html"""

    @staticmethod
    def default_config():
        result = datasets.MultilingualDataset.default_config()
        result.update({
            'subset': '2001',
            'ranktopk': 1000,
            'querysource': 'topic'
        })
        return result

    def __init__(self, config, vocab, logger):
        super().__init__(config, logger, vocab)
        self.index_arabic = indices.AnseriniIndex(os.path.join(util.path_dataset(self), 'anserini.ar'), lang=self._lang())
        self.doc_store = indices.SqliteDocstore(os.path.join(util.path_dataset(self), 'docs.sqlite'))

    def _get_docstore(self):
        return self.doc_store

    def _lang(self):
        return 'ar'

    def _get_index_for_batchsearch(self):
        return self.index_arabic

    def init(self, force=False):
        self._init_topics(
            subset='2001',
            topic_files=['https://trec.nist.gov/data/topics_noneng/arabic_topics.txt'],
            qid_prefix='AR',
            encoding="ISO-8859-6",
            expected_md5="a3d78c379056a080fe40a59a341496b8",
            force=force)

        self._init_topics(
            subset='2002',
            topic_files=['https://trec.nist.gov/data/topics_noneng/CL.topics.arabic.trec11.txt'],
            qid_prefix='AR',
            encoding="ISO-8859-6",
            expected_md5="f75a6164d794bab66509f1e818612363",
            force=force)

        self._init_qrels(
            subset='2001',
            qrels_files=['https://trec.nist.gov/data/qrels_noneng/xlingual_t10qrels.txt'],
            expected_md5="5951e2f0bf72df9f93fc32b93e3a7fde",
            force=force)

        self._init_qrels(
            subset='2002',
            qrels_files=['https://trec.nist.gov/data/qrels_noneng/qrels.trec11.xlingual.txt'],
            expected_md5="40f25e1e98101e27d081685cbdc390ef",
            force=force)

        self._init_indices_parallel(
            indices=[self.index_arabic, self.doc_store],
            doc_iter=self._init_collection_iter(
                doc_paths=['arabic_newswire_a/transcripts/'],
                encoding="utf8"),
            force=force)
