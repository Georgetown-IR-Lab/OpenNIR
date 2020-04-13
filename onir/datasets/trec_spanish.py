import os
from onir import util, datasets, indices


@datasets.register('trec_spanish')
class TrecSpanishDataset(datasets.MultilingualDataset):
    """
    Interface to the TREC Spanish datasets. Used for cross-lingual experiments in:
     > Sean MacAvaney, Luca Soldaini, Nazli Goharian. Teaching a New Dog Old Tricks: Resurrecting
     > Multilingual Retrieval Using Zero-shot Learning. In ECIR 2020.
    """
    DUA = """Will begin downloading TREC Spanish dataset.
You first need a copy of LDC2000T51, which can be found at:
https://catalog.ldc.upenn.edu/LDC2000T51

Then copy/link document collection to:
{ds_path}/trec_spanish

Please confirm you agree to the authors' data usage stipulations found at
https://catalog.ldc.upenn.edu/LDC2000T51
and
https://trec.nist.gov/data/testq_noneng.html"""

    @staticmethod
    def default_config():
        result = datasets.MultilingualDataset.default_config()
        result.update({
            'subset': 'trec3',
            'ranktopk': 1000,
            'querysource': 'topic'
        })
        return result

    def __init__(self, config, vocab, logger):
        super().__init__(config, logger, vocab)
        self.index_spanish = indices.AnseriniIndex(os.path.join(util.path_dataset(self), 'anserini.es'), lang=self._lang())
        self.doc_store = indices.SqliteDocstore(os.path.join(util.path_dataset(self), 'docs.sqlite'))

    def _get_docstore(self):
        return self.doc_store

    def _lang(self):
        return 'es'

    def _get_index_for_batchsearch(self):
        return self.index_spanish

    def init(self, force=False):
        self._init_topics(
            subset='trec3',
            topic_files=['https://trec.nist.gov/data/topics_noneng/topics.SP1-SP25.spanish.english.gz'],
            qid_prefix='SP',
            encoding="ISO-8859-1",
            xml_prefix='',
            expected_md5="22eea4a5c131db9cc4a431235f6a0573",
            force=force)

        self._init_topics(
            subset='trec4',
            topic_files=['https://trec.nist.gov/data/topics_noneng/topics.SP26-SP50.spanish.english.gz'],
            qid_prefix='SP',
            encoding="ISO-8859-1",
            xml_prefix='',
            expected_md5="dfd9685cce559e33ab397c1878a6a1f8",
            force=force)

        self._init_qrels(
            subset='trec3',
            qrels_files=['https://trec.nist.gov/data/qrels_noneng/qrels.1-25.spanish.gz'],
            expected_md5="e1703487f43fb7ea30b87a0f14ccb5ce",
            force=force)

        self._init_qrels(
            subset='trec4',
            qrels_files=['https://trec.nist.gov/data/qrels_noneng/qrels.26-50.spanish.gz'],
            expected_md5="f2540f9fb83433ca8ef9503671136498",
            force=force)

        self._init_indices_parallel(
            indices=[self.index_spanish, self.doc_store],
            doc_iter=self._init_collection_iter(
                doc_paths=['trec_spanish/infosel_data'],
                encoding="ISO-8859-1"),
            force=force)
