import torch
from torch import nn


def binmat(a, b, padding=None):
    BAT, A, B = a.shape[0], a.shape[1], b.shape[1]
    a = a.reshape(BAT, A, 1)
    b = b.reshape(BAT, 1, B)
    result = (a == b)
    if padding is not None:
        result = result & (a != padding) & (b != padding)
    return result.float()


def cos_simmat(a, b, amask=None, bmask=None):
    BAT, A, B = a.shape[0], a.shape[1], b.shape[1]
    a_denom = a.norm(p=2, dim=2).reshape(BAT, A, 1) + 1e-9 # avoid 0div
    b_denom = b.norm(p=2, dim=2).reshape(BAT, 1, B) + 1e-9 # avoid 0div
    result = a.bmm(b.permute(0, 2, 1)) / (a_denom * b_denom)
    if amask is not None:
        result = result * amask.reshape(BAT, A, 1)
    if bmask is not None:
        result = result * bmask.reshape(BAT, 1, B)
    return result


class InteractionMatrix(nn.Module):

    def __init__(self, padding=-1):
        super().__init__()
        self.padding = padding

    def forward(self, a_embed, b_embed, a_tok, b_tok):
        wrap_list = lambda x: x if isinstance(x, list) else [x]

        a_embed = wrap_list(a_embed)
        b_embed = wrap_list(b_embed)

        BAT, A, B = a_embed[0].shape[0], a_embed[0].shape[1], b_embed[0].shape[1]

        simmats = []

        for a_emb, b_emb in zip(a_embed, b_embed):
            if a_emb.dtype is torch.long and len(a_emb.shape) == 2 and \
               b_emb.dtype is torch.long and len(b_emb.shape) == 2:
                # binary matrix
                simmats.append(binmat(a_emb, b_emb, padding=self.padding))
            else:
                # cosine similarity matrix
                a_mask = (a_tok.reshape(BAT, A, 1) != self.padding).float()
                b_mask = (b_tok.reshape(BAT, 1, B) != self.padding).float()
                simmats.append(cos_simmat(a_emb, b_emb, a_mask, b_mask))
        return torch.stack(simmats, dim=1)

    def encode_query_doc(self, encoder, **inputs):
        enc = encoder.enc_query_doc(**inputs)
        return self(enc['query'], enc['doc'], inputs['query_tok'], inputs['doc_tok'])
