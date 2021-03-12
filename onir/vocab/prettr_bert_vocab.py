import os
import tempfile
import torch
from pytorch_transformers import BertTokenizer
from onir.interfaces import bert_models
from onir import vocab, util, config
from onir.modules import PrettrBertModel
import tokenizers as tk


@vocab.register('prettr_bert')
class PrettrBertVocab(vocab.Vocab):
    @staticmethod
    def default_config():
        return {
            'bert_base': 'bert-base-uncased',
            'bert_weights': '',
            'join_layer': 0, # all layers
            'compress_size': 0, # disable
            'compress_fp16': False,
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
        return PrettrBertEncoder(self)

    def path_segment(self):
        result = '{name}_{bert_base}'.format(name=self.name, **self.config)
        if self.config['bert_weights']:
            result += '_{}'.format(self.config['bert_weights'])
        if self.config['join_layer'] != 0:
            result += '_join-{}'.format(self.config['join_layer'])
        if self.config['compress_size'] != 0:
            result += '_compress-{}'.format(self.config['compress_size'])
        if self.config['compress_fp16'] != 0:
            result += '_compress-fp16'
        return result

    def lexicon_path_segment(self):
        return 'bert_{bert_base}'.format(**self.config)

    def lexicon_size(self) -> int:
        return self.tokenizer._tokenizer.get_vocab_size()


class PrettrBertEncoder(vocab.VocabEncoder):

    def __init__(self, vocabulary):
        super().__init__(vocabulary)
        bert_model = bert_models.get_model(vocabulary.config['bert_base'], vocabulary.logger)
        self.bert = PrettrBertModel.from_pretrained(bert_model,
            join_layer=vocabulary.config['join_layer'],
            compress_size=vocabulary.config['compress_size'],
            compress_fp16=vocabulary.config['compress_fp16'])
        if vocabulary.config['bert_weights']:
            weight_path = os.path.join(util.path_vocab(vocabulary), vocabulary.config['bert_weights'])
            with vocabulary.logger.duration('loading BERT weights from {}'.format(weight_path)):
                _, unexpected = self.bert.load_state_dict(torch.load(weight_path), strict=False)
                if unexpected:
                    vocabulary.logger.warn('Unexpected keys found when loading {}: {}. Be sure it'
                                           'is properly prefixed (e.g., without bert.)'.format(vocabulary.config['bert_weights'], unexpected))
        self.CLS = vocabulary.tok2id('[CLS]')
        self.SEP = vocabulary.tok2id('[SEP]')

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

        return {
            'query': query_results,
            'doc': doc_results,
            'cls': cls_results
        }

    def _enc_spec(self) -> dict:
        return {
            'dim': self.bert.config.hidden_size,
            'views': self.bert.config.num_hidden_layers + 1,
            'static': False,
            'supports_forward': False,
            'joint_fields': ['query', 'doc', 'cls']
        }
