import os
import tempfile
import unittest
from onir import indices
from onir.interfaces import plaintext


class TestMetrics(unittest.TestCase):

    def test_build(self):
        df = plaintext.read_tsv('etc/dummy_datafile.tsv')
        docs = [indices.RawDoc(did, dtext) for t, did, dtext in df if t == 'doc']
        with tempfile.TemporaryDirectory() as tmpdir:
            idxs = [
                (indices.AnseriniIndex(os.path.join(tmpdir, 'anserini')), False),
                (indices.AnseriniIndex(os.path.join(tmpdir, 'anserini.rawdocs'), store_raw_docs=True), True),
                (indices.SqliteDocstore(os.path.join(tmpdir, 'sqlite')), True),
            ]
            for index, check_raw_docs in idxs:
                with self.subTest(index=index):
                    self.assertFalse(index.built())
                    index.build(iter(docs))
                    self.assertTrue(index.built())
                    self.assertEqual(index.num_docs(), len(docs))
                    if check_raw_docs:
                        for doc in docs:
                            self.assertEqual(index.get_raw(doc.did), doc.data['text'])

    def test_batch_query(self):
        df = list(plaintext.read_tsv('etc/dummy_datafile.tsv'))
        docs = [indices.RawDoc(did, dtext) for t, did, dtext in df if t == 'doc']
        queries = [(qid, qtext) for t, qid, qtext in df if t == 'query']
        with tempfile.TemporaryDirectory() as tmpdir:
            idxs = [
                indices.AnseriniIndex(os.path.join(tmpdir, 'anserini')),
            ]
            models = [
                'bm25', 'bm25_k1-1.5', 'bm25_b-0.2', 'bm25_k1-1.6_b-0.8',
                'bm25_rm3', 'bm25_rm3_k1-1.5', 'bm25_rm3_b-0.2', 'bm25_rm3_k1-1.6_b-0.8',
                'bm25_rm3_rm3.fbTerms-2_rm3.fbDocs-2', 'bm25_rm3_rm3.fbTerms-2_rm3.fbDocs-2_k1-1.5',
                'bm25_rm3_rm3.fbTerms-2_rm3.fbDocs-2_b-0.2', 'bm25_rm3_rm3.fbTerms-2_rm3.fbDocs-2_k1-1.6_b-0.8',
                'ql', 'ql_mu-0.4',
                'sdm', 'sdm_uw-0.3_ow-0.2_tw-0.5',
            ]
            for index in idxs:
                index.build(docs)
                for model in models:
                    with self.subTest(index=index, model=model):
                        index.batch_query(queries, model, topk=10)
                        index.batch_query(queries, model, topk=10, quiet=True)


if __name__ == '__main__':
    unittest.main()
