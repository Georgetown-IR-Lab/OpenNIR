import torch
from torch import nn
from onir import rankers


@rankers.register('cedr_knrm')
class CedrKnrm(rankers.knrm.Knrm):
    """
    Implementation of CEDR for the KNRM model described in:
      > Sean MacAvaney, Andrew Yates, Arman Cohan, and Nazli Goharian. 2019. CEDR: Contextualized
      > Embeddings for Document Ranking. In SIGIR.
    Should be used with a model first trained using Vanilla BERT.
    """

    def __init__(self, vocab, config, logger, random):
        super().__init__(vocab, config, logger, random)
        enc = self.encoder
        assert 'cls' in enc.enc_spec()['joint_fields'], \
               "CedrKnrm requires a vocabulary that supports CLS encoding, e.g., BertVocab"
        self.combine = nn.Linear(self.combine.in_features + enc.dim(), self.combine.out_features)

    def _forward(self, **inputs):
        rep = self.encoder.enc_query_doc(**inputs)
        simmat = self.simmat(rep['query'], rep['doc'], inputs['query_tok'], inputs['doc_tok'])
        kernel_scores = self.kernel_pool(simmat)
        all_scores = torch.cat([kernel_scores, rep['cls'][-1]], dim=1)
        return self.combine(all_scores)
