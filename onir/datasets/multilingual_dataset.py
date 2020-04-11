import os
import itertools
from pytools import memoize_method
from onir import util, datasets
from onir.interfaces import trec, plaintext


class MultilingualDataset(datasets.IndexBackedDataset):
    """
    Abstract class used for cross-lingual experiments in:
     > Sean MacAvaney, Luca Soldaini, Nazli Goharian. Teaching a New Dog Old Tricks: Resurrecting
     > Multilingual Retrieval Using Zero-shot Learning. In ECIR 2020.
    """

    @staticmethod
    def default_config():
        result = datasets.IndexBackedDataset.default_config()
        result.update({
            'subset': '',
            'ranktopk': 1000,
            'querysource': 'topic',
        })
        return result

    def path_segment(self):
        result = '{name}_{rankfn}.{ranktopk}_{subset}'.format(**self.config, name=self.name)
        if self.config['querysource'] != 'topic':
            result += '_{querysource}'.format(**self.config)
        return result

    def _lang(self):
        raise NotImplementedError()

    @memoize_method
    def _load_queries_base(self, subset):
        querysource = self.config['querysource']
        query_path = os.path.join(util.path_dataset(self), f'{subset}.topics')
        return {qid: text for t, qid, text in plaintext.read_tsv(query_path) if t == querysource}

    def qrels(self, fmt='dict'):
        return self._load_qrels(self.config['subset'], fmt=fmt)

    @memoize_method
    def _load_qrels(self, subset, fmt):
        qrels_path = os.path.join(util.path_dataset(self), f'{subset}.qrels')
        return trec.read_qrels_fmt(qrels_path, fmt)

    def _init_topics(self, subset, topic_files, qid_prefix=None, encoding=None, xml_prefix=None, force=False, expected_md5=None):
        topicf = os.path.join(util.path_dataset(self), f'{subset}.topics')
        if (force or not os.path.exists(topicf)) and self._confirm_dua():
            topics = []
            for topic_file in topic_files:
                topic_file_stream = util.download_stream(topic_file, encoding, expected_md5=expected_md5)
                for t, qid, text in trec.parse_query_format(topic_file_stream, xml_prefix):
                    if qid_prefix is not None:
                        qid = qid.replace(qid_prefix, '')
                    topics.append((t, qid, text))
            plaintext.write_tsv(topicf, topics)

    def _init_qrels(self, subset, qrels_files, force=False, expected_md5=None):
        qrelsf = os.path.join(util.path_dataset(self), f'{subset}.qrels')
        if (force or not os.path.exists(qrelsf)) and self._confirm_dua():
            qrels = itertools.chain(*(trec.read_qrels(util.download_stream(f, 'utf8', expected_md5=expected_md5)) for f in qrels_files))
            trec.write_qrels(qrelsf, qrels)

    def _init_collection_iter(self, doc_paths, encoding):
        doc_paths = (os.path.join(util.path_dataset(self), p) for p in doc_paths)
        doc_iter = itertools.chain(*(trec.parse_doc_format(p, encoding) for p in doc_paths))
        doc_iter = self.logger.pbar(doc_iter, desc='documents')
        return doc_iter
