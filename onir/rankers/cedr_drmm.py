import torch
from torch import nn
from onir import rankers


@rankers.register('cedr_drmm')
class CedrDrmm(rankers.drmm.Drmm):
    """
    Implementation of CEDR for the DRMM model described in:
      > Sean MacAvaney, Andrew Yates, Arman Cohan, and Nazli Goharian. 2019. CEDR: Contextualized
      > Embeddings for Document Ranking. In SIGIR.
    Should be used with a model first trained using Vanilla BERT.
    """

    def __init__(self, vocab, config, logger, random):
        super().__init__(vocab, config, logger, random)
        enc = self.encoder
        assert 'cls' in enc.enc_spec()['joint_fields'], \
               "CedrDrmm requires a vocabulary that supports CLS encoding, e.g., BertVocab"
        self.hidden_1 = nn.Linear(self.hidden_1.in_features + enc.dim(), self.hidden_1.out_features)

    def _forward(self, **inputs):
        rep = self.encoder.enc_query_doc(**inputs)
        simmat = self.simmat(rep['query'], rep['doc'], inputs['query_tok'], inputs['doc_tok'])
        qterm_features = self.histogram_pool(simmat, inputs)
        BAT, QLEN, _ = qterm_features.shape
        cls_reps = rep['cls'][-1].reshape(BAT, 1, -1).expand(BAT, QLEN, -1)
        qterm_features = torch.cat([qterm_features, cls_reps], dim=2)
        qterm_features = qterm_features.reshape(BAT * QLEN, -1)
        qterm_scores = self.hidden_2(torch.relu(self.hidden_1(qterm_features))).reshape(BAT, QLEN)
        return self.combine(qterm_scores, inputs['query_idf'])
