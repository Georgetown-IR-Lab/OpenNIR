import torch
from onir import rankers


@rankers.register('cedr_pacrr')
class CedrPacrr(rankers.pacrr.Pacrr):
    """
    Implementation of CEDR for the PACRR model described in:
      > Sean MacAvaney, Andrew Yates, Arman Cohan, and Nazli Goharian. 2019. CEDR: Contextualized
      > Embeddings for Document Ranking. In SIGIR.
    Should be used with a model first trained using Vanilla BERT.
    """

    def __init__(self, vocab, config, logger, random):
        super().__init__(vocab, config, logger, random)
        enc = self.encoder
        assert 'cls' in enc.enc_spec()['joint_fields'], \
               "CedrPacrr requires a vocabulary that supports CLS encoding, e.g., BertVocab"
        assert isinstance(self.combination, rankers.pacrr.DenseCombination), \
               "CedrPacrr only supports dense combination"
        new_layers = list(self.combination.layers)
        new_layers[0] += enc.dim()
        self.combination = rankers.pacrr.DenseCombination(new_layers, self.combination.shuf)

    def _forward(self, **inputs):
        rep = self.encoder.enc_query_doc(**inputs)
        simmat = self.simmat(rep['query'], rep['doc'], inputs['query_tok'], inputs['doc_tok'])
        BAT = simmat.shape[0]
        conv_pool = self.conv_pool(simmat, inputs)
        conv_pool = conv_pool.reshape(BAT, -1)
        all_scores = torch.cat([conv_pool, rep['cls'][-1]], dim=1)
        return self.combination(all_scores, inputs['query_len'])
