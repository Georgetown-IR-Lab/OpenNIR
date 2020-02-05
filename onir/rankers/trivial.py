import torch
from torch import nn
from onir import rankers


@rankers.register('trivial')
class Trivial(rankers.Ranker):
    """
    Trivial ranker, which just returns the initial ranking score. Used for comparisions against
    neural ranking approaches.

    Options allow the score to be inverted (neg), the individual query term scores to be summed by
    the ranker itself (qsum), and to use the manual relevance assessment instead of the run score,
    representing an optimal re-ranker (max).
    """

    @staticmethod
    def default_config():
        result = rankers.Ranker.default_config()
        del result['qlen']
        del result['dlen']
        del result['add_runscore']
        result.update({
            'neg': False,
            'qsum': False,
            'max': False,
        })
        return result

    def __init__(self, config, random):
        super().__init__(config, random)
        # dummy parameter
        self._nil = nn.Parameter(torch.tensor(0.))

    def path_segment(self):
        result = self.name
        if self.config['max']:
            result += '_max'
        elif self.config['qsum']:
            result += '_qsum'
        if self.config['neg']:
            result += '_neg'
        return result

    def input_spec(self):
        fields = set()
        if self.config['max']:
            fields.add('relscore')
        elif self.config['qsum']:
            fields.add('query_score')
        else:
            fields.add('runscore')
        return {
            'fields': fields,
            'qlen_mode': 'max',
            'dlen_mode': 'max',
            'qlen': 9999999,
            'dlen': 9999999
        }

    def _forward(self, **inputs):
        if self.config['max']:
            result = inputs['relscore']
        elif self.config['qsum']:
            result = inputs['query_score'].sum(dim=1)
        else:
            result = inputs['runscore']
        if self.config['neg']:
            result = -1. * result
        return result + self._nil
