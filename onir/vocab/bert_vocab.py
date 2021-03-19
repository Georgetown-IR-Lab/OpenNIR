import os
import tempfile
import torch
from pytorch_transformers import BertTokenizer
from onir.interfaces import bert_models
from onir import vocab, util, config
from onir.modules import CustomBertModelWrapper
import tokenizers as tk


@vocab.register('bert')
class BertVocab(vocab.Vocab):
    @staticmethod
    def default_config():
        return {
            'bert_base': 'bert-base-uncased',
            'bert_weights': '',     # TODO: merge bert_base and bert_weights somehow, better integrate fine-tuning BERT into pipeline
            'layer': -1, # all layers
            'last_layer': False,
            'train': False,
            'encoding': config.Choices(['joint', 'sep']),
        }

    def __init__(self, config, logger):
        super().__init__(config, logger)
        bert_model = bert_models.get_model(config['bert_base'], self.logger)
        self.tokenizer = BertTokenizer.from_pretrained(bert_model)
        # HACK! Until the transformers library adopts tokenizers, save and re-load vocab
        with tempfile.TemporaryDirectory() as d:
            self.tokenizer.save_vocabulary(d)
            # this tokenizer is ~4x faster as the BertTokenizer, per my measurements
            self.tokenizer = tk.BertWordPieceTokenizer(os.path.join(d, 'vocab.txt'))

    def tokenize(self, text):
        # return self.tokenizer.tokenize(text)
        return self.tokenizer.encode(text).tokens[1:-1] # removes leading [CLS] and trailing [SEP]

    def char_ranges(self, text):
        return self.tokenizer.encode(text).offsets[1:-1]

    def tok2id(self, tok):
        # return self.tokenizer.vocab[tok]
        return self.tokenizer.token_to_id(tok)

    def id2tok(self, idx):
        if torch.is_tensor(idx):
            if len(idx.shape) == 0:
                return self.id2tok(idx.item())
            return [self.id2tok(x) for x in idx]
        # return self.tokenizer.ids_to_tokens[idx]
        return self.tokenizer.id_to_token(idx)

    def encoder(self):
        return {
            'joint': JointBertEncoder,
            'sep': SepBertEncoder,
        }[self.config['encoding']](self)

    def path_segment(self):
        result = '{name}_{bert_base}'.format(name=self.name, **self.config)
        if self.config['bert_weights']:
            result += '_{}'.format(self.config['bert_weights'])
        if self.config['last_layer']:
            if self.config['layer'] != -1:
                result += '_{}last'.format(self.config['layer'])
            else:
                result += '_last'
        elif self.config['layer'] != -1:
            result += '_{}only'.format(self.config['layer'])
        if self.config['train']:
            result += '_tr'
        if self.config['encoding'] != 'joint':
            result += '_{encoding}'.format(**self.config)
        return result

    def lexicon_path_segment(self):
        return 'bert_{bert_base}'.format(**self.config)

    def lexicon_size(self) -> int:
        return self.tokenizer._tokenizer.get_vocab_size()


class BaseBertEncoder(vocab.VocabEncoder):

    def __init__(self, vocabulary):
        super().__init__(vocabulary)
        layer = vocabulary.config['layer']
        if layer == -1:
            layer = None
        bert_model = bert_models.get_model(vocabulary.config['bert_base'], vocabulary.logger)
        self.bert = CustomBertModelWrapper.from_pretrained(bert_model, depth=layer)
        if vocabulary.config['bert_weights']:
            weight_path = os.path.join(util.path_vocab(vocabulary), vocabulary.config['bert_weights'])
            with vocabulary.logger.duration('loading BERT weights from {}'.format(weight_path)):
                self.bert.load_state_dict(torch.load(weight_path), strict=False)
        self.CLS = vocabulary.tok2id('[CLS]')
        self.SEP = vocabulary.tok2id('[SEP]')
        self.bert.set_trainable(vocabulary.config['train'])

    def _enc_spec(self) -> dict:
        return {
            'dim': self.bert.config.hidden_size,
            'views': 1 if self.vocab.config['last_layer'] else self.bert.depth + 1,
            'static': not self.vocab.config['train']
        }


class SepBertEncoder(BaseBertEncoder):

    def forward(self, in_toks, lens):
        results, _ = self._forward(in_toks, lens)
        return results

    def _forward(self, in_toks, lens=None, seg_id=0):
        if lens is None:
            # if no lens provided, assume all are full length, I guess... not great
            lens = torch.full_like(in_toks[:, 0], in_toks.shape[1])
        maxlen = self.bert.config.max_position_embeddings
        MAX_TOK_LEN = maxlen - 2 # -2 for [CLS] and [SEP]
        toks, _ = util.subbatch(in_toks, MAX_TOK_LEN)
        mask = util.lens2mask(lens, in_toks.shape[1])
        mask, _ = util.subbatch(mask, MAX_TOK_LEN)
        toks = torch.cat([torch.full_like(toks[:, :1], self.CLS), toks], dim=1)
        toks = torch.cat([toks, torch.full_like(toks[:, :1], self.SEP)], dim=1)
        ONES = torch.ones_like(mask[:, :1])
        mask = torch.cat([ONES, mask, ONES], dim=1)
        segment_ids = torch.full_like(toks, seg_id)
        # Change -1 padding to 0-padding (will be masked)
        toks = torch.where(toks == -1, torch.zeros_like(toks), toks)
        result = self.bert(toks, segment_ids, mask)
        if not self.vocab.config['last_layer']:
            cls_result = [r[:, 0] for r in result]
            result = [r[:, 1:-1, :] for r in result]
            result = [util.un_subbatch(r, in_toks, MAX_TOK_LEN) for r in result]
        else:
            BATCH = in_toks.shape[0]
            result = result[-1]
            cls_output = result[:, 0]
            cls_result = []
            for i in range(cls_output.shape[0] // BATCH):
                cls_result.append(cls_output[i*BATCH:(i+1)*BATCH])
            cls_result = torch.stack(cls_result, dim=2).mean(dim=2)
            result = result[:, 1:-1, :]
            result = util.un_subbatch(result, in_toks, MAX_TOK_LEN)
        return result, cls_result

    def enc_query_doc(self, **inputs):
        result = {}
        if 'query_tok' in inputs and 'query_len' in inputs:
            query_results, query_cls = self._forward(inputs['query_tok'], inputs['query_len'], seg_id=0)
            result.update({
                'query': query_results,
                'query_cls': query_cls
            })
        if 'doc_tok' in inputs and 'doc_len' in inputs:
            doc_results, doc_cls = self._forward(inputs['doc_tok'], inputs['doc_len'], seg_id=1)
            result.update({
                'doc': doc_results,
                'doc_cls': doc_cls
            })
        return result

    def _enc_spec(self) -> dict:
        result = super()._enc_spec()
        result.update({
            'joint_fields': ['query', 'doc', 'cls_query', 'cls_doc']
        })
        return result


class JointBertEncoder(BaseBertEncoder):

    def enc_query_doc(self, **inputs):
        query_tok, query_len = inputs['query_tok'], inputs['query_len']
        doc_tok, doc_len = inputs['doc_tok'], inputs['doc_len']
        BATCH, QLEN = query_tok.shape
        maxlen = self.bert.config.max_position_embeddings
        MAX_DOC_TOK_LEN = maxlen - QLEN - 3 # -3 [CLS] and 2x[SEP]

        doc_toks, sbcount = util.subbatch(doc_tok, MAX_DOC_TOK_LEN)
        doc_mask = util.lens2mask(doc_len, doc_tok.shape[1])
        doc_mask, _ = util.subbatch(doc_mask, MAX_DOC_TOK_LEN)

        query_toks = torch.cat([query_tok] * sbcount, dim=0)
        query_mask = util.lens2mask(query_len, query_toks.shape[1])
        query_mask = torch.cat([query_mask] * sbcount, dim=0)

        CLSS = torch.full_like(query_toks[:, :1], self.CLS)
        SEPS = torch.full_like(query_toks[:, :1], self.SEP)
        ONES = torch.ones_like(query_mask[:, :1])
        NILS = torch.zeros_like(query_mask[:, :1])
        toks = torch.cat([CLSS, query_toks, SEPS, doc_toks, SEPS], dim=1)
        mask = torch.cat([ONES, query_mask, ONES, doc_mask, ONES], dim=1)
        segment_ids = torch.cat([NILS] * (2 + QLEN) + [ONES] * (1 + doc_toks.shape[1]), dim=1)

        # Change -1 padding to 0-padding (will be masked)
        toks = torch.where(toks == -1, torch.zeros_like(toks), toks)

        result = self.bert(toks, segment_ids, mask)

        # extract relevant subsequences for query and doc
        query_results = [r[:BATCH, 1:QLEN+1] for r in result]
        doc_results = [r[:, QLEN+2:-1] for r in result]
        doc_results = [util.un_subbatch(r, doc_tok, MAX_DOC_TOK_LEN) for r in doc_results]

        cls_results = []
        for layer in range(len(result)):
            cls_output = result[layer][:, 0]
            cls_result = []
            for i in range(cls_output.shape[0] // BATCH):
                cls_result.append(cls_output[i*BATCH:(i+1)*BATCH])
            cls_result = torch.stack(cls_result, dim=2).mean(dim=2)
            cls_results.append(cls_result)

        if self.vocab.config['last_layer']:
            query_results = query_results[-1]
            doc_results = doc_results[-1]
            cls_results = cls_results[-1]

        return {
            'query': query_results,
            'doc': doc_results,
            'cls': cls_results
        }

    def _enc_spec(self) -> dict:
        result = super()._enc_spec()
        result.update({
            'supports_forward': False,
            'joint_fields': ['query', 'doc', 'cls']
        })
        return result
