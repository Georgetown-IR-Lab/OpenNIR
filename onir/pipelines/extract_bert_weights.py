import os
from collections import OrderedDict
import torch
from onir import pipelines, util


@pipelines.register('extract_bert_weights')
class ExtractBertWeights(pipelines.BasePipeline):
    @staticmethod
    def default_config():
        return {
            'file_path': '',
            'epoch': '',
            'val_metric': '',
            'bert_weights': '',
            'overwrite': False,
        }

    def __init__(self, config, ranker, vocab, train_ds, trainer, valid_pred, logger):
        super().__init__(config, logger)
        self.ranker = ranker
        self.vocab = vocab
        self.trainer = trainer
        self.train_ds = train_ds
        self.valid_pred = valid_pred

    def run(self):
        if self.config['bert_weights'] == '':
            raise ValueError('must provide pipeline.bert_weights setting (name of weights file)')
        weight_file = os.path.join(util.path_vocab(self.vocab), self.config['bert_weights'])
        if os.path.exists(weight_file) and not self.config['overwrite']:
            raise ValueError(f'{weight_file} already exists. Please rename pipeline.bert_weights or set pipeline.overwrite=True')
        self._load_ranker_weights(self.ranker, self.vocab, self.trainer, self.valid_pred, self.train_ds)
        old_sate_dict = self.ranker.state_dict()
        new_state_dict = OrderedDict()
        for key in old_sate_dict:
            if key.startswith('encoder.bert.'):
                new_state_dict[key[len('encoder.bert.'):]] = old_sate_dict[key]
        torch.save(new_state_dict, weight_file)
        self.logger.info(f'new BERT sate dict saved to {weight_file}')
