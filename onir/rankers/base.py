import torch
from torch import nn


class Ranker(nn.Module):
    name = None

    @staticmethod
    def default_config():
        return {
            'qlen': 20,
            'dlen': 2000,
            'add_runscore': False
        }

    def __init__(self, config, random):
        super().__init__()
        self.config = config
        self.random = random
        seed = random.randint((2**32)-1)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        if self.config.get('add_runscore'):
            self.runscore_alpha = torch.nn.Parameter(torch.full((1, ), -1.))

    def input_spec(self):
        # qlen_mode and dlen_mode possible values:
        # 'strict': query/document must be exactly this length
        # 'max': query/document can be at most this length
        result = {
            'fields': set(),
            'qlen': self.config['qlen'],
            'qlen_mode': 'strict',
            'dlen': self.config['dlen'],
            'dlen_mode': 'strict'
        }
        if self.config.get('add_runscore'):
            result['fields'].add('runscore')
        return result

    def forward(self, **inputs):
        result = self._forward(**inputs)
        if len(result.shape) == 2 and result.shape[1] == 1:
            result = result.reshape(result.shape[0])
        if self.config.get('add_runscore'):
            alpha = torch.sigmoid(self.runscore_alpha)
            result = alpha * result + (1 - alpha) * inputs['runscore']
        return result

    def _forward(self, **inputs):
        raise NotImplementedError

    def path_segment(self):
        raise NotImplementedError

    def save(self, path):
        state = self.state_dict(keep_vars=True)
        for key in list(state):
            if state[key].requires_grad:
                state[key] = state[key].data
            else:
                del state[key]
        torch.save(state, path)

    def load(self, path):
        self.load_state_dict(torch.load(path), strict=False)
