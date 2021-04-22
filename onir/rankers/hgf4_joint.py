import torch
import torch.nn.functional as F
import transformers
from onir import rankers


@rankers.register('hgf4_joint')
class HuggingfaceTransformers4JointRanker(rankers.Ranker):
    @staticmethod
    def default_config():
        result = rankers.Ranker.default_config()
        result.update({
            'model': 'bert-base-uncased', # a model name registered in huggingface transformers
            'norm': 'none',
            'outputs': 1,
        })
        return result

    def __init__(self, config, logger, random):
        super().__init__(config, random)
        self.logger = logger
        assert str(transformers.__version__).startswith('4.'), "hgf4_joint only supports transformers library 4.x"
        self.model_config = transformers.AutoConfig.from_pretrained(self.config['model'])
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.config['model'])
        self.model = transformers.AutoModelForSequenceClassification.from_pretrained(self.config['model'])
        assert self.model_config.num_labels >= config['outputs']
        self._nil = torch.nn.Parameter(torch.tensor(0.))

    def input_spec(self):
        result = super().input_spec()
        result['fields'].update({'query_rawtext', 'doc_rawtext'})
        return result

    def path_segment(self):
        result = '{name}_{model}_{outputs}'.format(name=self.name, model=self.config['model'].replace('/', '-'), outputs=self.config['outputs'])
        if self.config['add_runscore']:
            result += '_addrun'
        return result

    def _forward(self, **inputs):
        model_inputs = self.tokenizer(inputs['query_rawtext'], inputs['doc_rawtext'], padding=True, truncation='only_second', return_tensors='pt')
        model_inputs = {k: v.to(self._nil.device) for k, v in model_inputs.items()}
        result = self.model(**model_inputs).logits
        if self.config['norm'] == 'none':
            return result[:, :self.config['outputs']]
        elif self.config['norm'] == 'softmax':
            result = F.softmax(result, dim=1)
            return result[:, 0]
        elif self.config['norm'] == 'softmax-2':
            result = F.softmax(result, dim=1)
            return result[:, 1]
        raise ValueError('unsupported norm={norm}'.format(**self.config))

    def load_hgf_checkpoint(self, checkpoint_path):
        with self.logger.duration('loading BERT weights from {}'.format(checkpoint_path)):
            weights = torch.load(checkpoint_path)
            if 'classifier.weight' not in weights and 'cls.seq_relationship.weight' in weights:
                weights['classifier.weight'] = weights['cls.seq_relationship.weight']
            if 'classifier.bias' not in weights and 'cls.seq_relationship.bias' in weights:
                weights['classifier.bias'] = weights['cls.seq_relationship.bias']
            result = self.model.load_state_dict(weights, strict=False)
        print(result)
