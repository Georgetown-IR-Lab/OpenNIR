import torch
import torch.nn.functional as F
from onir import rankers, util


@rankers.register('epic')
class EpicRanker(rankers.Ranker):
    """
    Implementation of the EPIC model from:
      > Sean MacAvaney, Franco Maria Nardini, Raffaele Perego, Nicola Tonellotto,
      > Nazli Goharian, and Ophir Frieder. 2020. Expansion via Prediction of Importance with
      > Contextualization. In SIGIR.
    """
    @staticmethod
    def default_config():
        result = rankers.Ranker.default_config()
        result.update({
            'qlen': 500,
            'toks': 100,
        })
        return result

    def __init__(self, vocab, config, logger, random):
        super().__init__(config, random)
        self.config['qlen'] = 500 # always override config
        self.logger = logger
        self.vocab = vocab
        self.encoder = vocab.encoder()
        if self.encoder.static():
            logger.warn("It's usually bad to use BERTRanker with non-trainable embeddings. "
                        "Consider setting `vocab.train=True`")
        self.dropout = torch.nn.Dropout(0.1) # self.encoder.bert.config.hidden_dropout_prob
        self.query_salience = torch.nn.Linear(self.encoder.dim(), 1)
        self.doc_salience = torch.nn.Linear(self.encoder.dim(), 1)
        self.activ = lambda x: (1. + F.softplus(x)).log()
        self._nil = torch.nn.Parameter(torch.zeros(1))
        self.doc_quality = torch.nn.Linear(self.encoder.dim(), 1)

    def input_spec(self):
        result = super().input_spec()
        result['fields'].update({'query_tok', 'query_len', 'query_text', 'doc_tok', 'doc_len', 'doc_text'})
        result['qlen_mode'] = 'max'
        result['dlen_mode'] = 'max'
        return result

    def path_segment(self):
        result = '{name}'.format(name=self.name, **self.config)
        if self.config['toks'] != 100:
            result += '_{toks}toks'.format(**self.config)
        return result

    def _forward(self, **inputs):
        encoded = self.encoder.enc_query_doc(**inputs)
        query_vecs = self.query_full_vector(encoded['query'], inputs['query_len'], inputs['query_tok'])
        doc_vecs, _ = self.doc_full_vector(encoded['doc'], inputs['doc_len'], encoded['doc_cls'])

        return self.similarity(query_vecs, doc_vecs)

    def doc_vectors(self, dense=False, **inputs):
        encoded = self.encoder.enc_query_doc(**inputs)
        full_vecs, doc_qlty = self.doc_full_vector(encoded['doc'], inputs['doc_len'], encoded['doc_cls'])
        if dense:
            return full_vecs
        result = []
        for doc, qlty in zip(full_vecs, doc_qlty):
            scores, idxs = doc.topk(self.config['toks'])
            toks = self.vocab.id2tok(idxs)
            result.append(dict(zip(toks, scores)))
            if qlty is not None:
                result[-1]['__applied_quality__'] = qlty
        return result

    def query_vectors(self, dense=False, **inputs):
        encoded = self.encoder.enc_query_doc(**inputs)
        if dense:
            return self.query_full_vector(encoded['query'], inputs['query_len'], inputs['query_tok'], dense=False)
        tok_sals = self._query_salience(encoded['query'], inputs['query_len'], inputs['query_tok'])
        result = []
        for qsals, qtoks in zip(tok_sals.detach().cpu(), inputs['query_tok'].detach().cpu()):
            result.append({})
            for qsal, qtok in zip(qsals, qtoks):
                if qtok != -1:
                    result[-1][self.vocab.id2tok(qtok.item())] = qsal
        return result

    def doc_full_vector(self, doc_tok_reps, doc_len, doc_cls):
        tok_salience = self.doc_salience(self.dropout(doc_tok_reps))
        tok_salience = self.activ(tok_salience)
        exp_raw = self.encoder.bert.cls.predictions(doc_tok_reps)
        mask = util.lens2mask(doc_len, exp_raw.shape[1])
        exp = self.activ(exp_raw)
        exp = exp * tok_salience * mask.unsqueeze(2).float()
        exp, _ = exp.max(dim=1)
        qlty = torch.sigmoid(self.doc_quality(doc_cls))
        exp = qlty * exp
        qlty = qlty.reshape(doc_cls.shape[0])
        return exp, qlty

    def query_full_vector(self, query_tok_reps, query_len, query_tok, dense=True):
        tok_salience = self._query_salience(query_tok_reps, query_len, query_tok)
        idx0 = torch.arange(tok_salience.shape[0], device=tok_salience.device).reshape(tok_salience.shape[0], 1).expand(tok_salience.shape[0], tok_salience.shape[1]).flatten()
        idx1 = query_tok.flatten()
        idx1[idx1 == -1] = 0

        s = torch.Size([query_tok.shape[0], self.vocab.lexicon_size()])
        result = torch.sparse.FloatTensor(torch.stack((idx0, idx1)), tok_salience.flatten(), s)
        if dense:
            result = result.to_dense()
        return result

    def _query_salience(self, query_tok_reps, query_len, query_tok):
        inputs = self.dropout(query_tok_reps)
        tok_salience = self.query_salience(inputs)
        tok_salience = self.activ(tok_salience).squeeze(2)
        mask = util.lens2mask(query_len, query_tok.shape[1])
        tok_salience = tok_salience * mask.float()
        return tok_salience

    def similarity(self, query_vecs, doc_vecs):
        return (query_vecs * doc_vecs).sum(dim=1)

    def similarity_inference(self, query_vecs, doc_vecs):
        return torch.sparse.mm(doc_vecs, query_vecs.to_dense().transpose(1, 0)).squeeze(1)
