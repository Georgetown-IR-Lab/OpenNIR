import random
import unittest
import onir
from onir.interfaces import trec


class TestMetrics(unittest.TestCase):

    def test_p(self):
        self._test_cases([
            {
                'measure': 'p@1',
                'qrels': {'B': 1},
                'run': {'A': 1, 'B': 0},
                'expected': 0.0
            },
            {
                'measure': 'p@1',
                'qrels': {'A': 1},
                'run': {'A': 1, 'B': 0},
                'expected': 1.0
            },
            {
                'measure': 'p_rel-1@1',
                'qrels': {'A': 1},
                'run': {'A': 1, 'B': 0},
                'expected': 1.0
            },
            {
                'measure': 'p_rel-2@1',
                'qrels': {'A': 1},
                'run': {'A': 1, 'B': 0},
                'expected': 0.0
            },
            {
                'measure': 'p_rel-2@1',
                'qrels': {'A': 2},
                'run': {'A': 1, 'B': 0},
                'expected': 1.0
            },
            {
                'measure': 'p@2',
                'qrels': {'A': 1},
                'run': {'A': 1, 'B': 0},
                'expected': 0.5
            },
            {
                'measure': 'p@2',
                'qrels': {'A': 1, 'B': 1},
                'run': {'A': 1, 'B': 0},
                'expected': 1.0
            },
        ])

    def test_r(self):
        self._test_cases([
            {
                'measure': 'r@1',
                'qrels': {'A': 1, 'B': 0},
                'run': {'B': 1},
                'expected': 0.0
            },
            {
                'measure': 'r@1',
                'qrels': {'A': 1, 'B': 0},
                'run': {'A': 1},
                'expected': 1.0
            },
            {
                'measure': 'r_rel-1@1',
                'qrels': {'A': 1, 'B': 0},
                'run': {'A': 1},
                'expected': 1.0
            },
            {
                'measure': 'r_rel-2@1',
                'qrels': {'A': 1, 'B': 0},
                'run': {'A': 1},
                'expected': 0.0
            },
            {
                'measure': 'r_rel-2@1',
                'qrels': {'A': 2, 'B': 0},
                'run': {'A': 1},
                'expected': 1.0
            },
            {
                'measure': 'r@2',
                'qrels': {'A': 1, 'B': 1},
                'run': {'A': 1},
                'expected': 0.5
            },
            {
                'measure': 'r@2',
                'qrels': {'A': 1, 'B': 1},
                'run': {'A': 1, 'B': 1},
                'expected': 1.0
            },
        ])

    def test_treceval_equals_pytreceval(self):
        rng = random.Random(42)
        DOCS = list('ABCEDFGHIJKLMNOPQRSTUVWXYZ')
        TE = onir.metrics.TrecEvalMetrics()
        PTE = onir.metrics.PyTrecEvalMetrics()
        MEASURES = ['p@10', 'p_rel-2@10', 'r@10', 'r_rel-2@10', 'ndcg', 'mrr', 'mrr_rel-2', 'map', 'map_rel-2', 'rprec', 'rprec_rel-2', 'ndcg@10']
        for _ in range(100):
            self._test_same([{
                'measures': MEASURES,
                'qrels': {doc: rng.choice([-1, 0, 1, 2, 3]) for doc in rng.sample(DOCS, 10)},
                'run': {doc: round(rng.uniform(-5, 5), 2) for doc in rng.sample(DOCS, 20)},
                'a': TE,
                'b': PTE,
            }])

    def _test_cases(self, cases):
        for case in cases:
            measure = onir.metrics.Metric(case['measure'])
            qrels = onir.metrics.FallbackMetrics.FormatManager({'0': case['qrels']}, trec.read_qrels_dict, trec.write_qrels_dict_)
            run = onir.metrics.FallbackMetrics.FormatManager({'0': case['run']}, trec.read_run_dict, trec.write_run_dict_)
            for metric_provider in onir.metrics.primary.metrics_priority_list:
                if metric_provider.supports(measure):
                    with self.subTest(metric_provider=metric_provider, **case):
                        these_qrels = qrels.in_format(metric_provider.QRELS_FORMAT)
                        these_run = run.in_format(metric_provider.RUN_FORMAT)
                        result = metric_provider.calc_metrics(these_qrels, these_run, [measure])
                        self.assertTrue(str(measure) in result)
                        self.assertTrue('0' in result[str(measure)])
                        self.assertEqual(result[str(measure)]['0'], case['expected'])

    def _test_same(self, cases):
        for case in cases:
            measures = [onir.metrics.Metric(m) for m in case['measures']]
            qrels = onir.metrics.FallbackMetrics.FormatManager({'0': case['qrels']}, trec.read_qrels_dict, trec.write_qrels_dict_)
            run = onir.metrics.FallbackMetrics.FormatManager({'0': case['run']}, trec.read_run_dict, trec.write_run_dict_)
            a, b = case['a'], case['b']
            result_a = a.calc_metrics(qrels.in_format(a.QRELS_FORMAT), run.in_format(a.RUN_FORMAT), measures)
            result_b = b.calc_metrics(qrels.in_format(b.QRELS_FORMAT), run.in_format(b.RUN_FORMAT), measures)
            for measure in measures:
                with self.subTest(**case, measure=[measure]):
                    self.assertAlmostEqual(result_a[str(measure)]['0'], result_b[str(measure)]['0'], places=3)



if __name__ == '__main__':
    unittest.main()
