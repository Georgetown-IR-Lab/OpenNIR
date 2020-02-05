import re
from torch import nn


class VocabEncoder(nn.Module):
    """
    Encodes batches of id sequences
    """
    def __init__(self, vocabulary):
        super().__init__()
        self.vocab = vocabulary

    def forward(self, toks, lens=None):
        """
        Returns embeddings for the given toks.
        toks: token IDs (shape: [batch, maxlen])
        lens: lengths of each item (shape: [batch])
        """
        raise NotImplementedError

    def enc_query_doc(self, **inputs):
        """
        Returns encoded versions of the query and document from general **inputs dict
        Requires query_tok, doc_tok, query_len, and doc_len.
        May be overwritten in subclass to provide contextualized representation, e.g.
        joinly modeling query and document representations in BERT.
        """
        return {
            'query': self(inputs['query_tok'], inputs['query_len']),
            'doc': self(inputs['doc_tok'], inputs['doc_len'])
        }

    def enc_spec(self) -> dict:
        """
        Returns various characteristics about the output format of this encoder.
        """
        result = {'supports_forward': True, 'joint_fields': ['query', 'doc']}
        result.update(self._enc_spec())
        REQUIRED_KEYS = {'views': int, 'dim': int, 'static': bool, 'supports_forward': bool,
                         'joint_fields': list}
        for key, t in REQUIRED_KEYS.items():
            assert key in result, f"missing key `{key}`"
            assert isinstance(result[key], t), f"key `{key}` not of type {t}"
        return result

    def _enc_spec(self) -> dict:
        raise NotImplementedError()

    def emb_views(self) -> int:
        """
        Returns how many "views" are returned by the embedding layer.
        Most have 1, but sometimes it's useful to return multiple, e.g., BERT's multiple layers
        """
        return self.enc_spec()['views']

    def dim(self) -> int:
        """
        Returns the number of dimensions of the embedding
        """
        return self.enc_spec()['dim']

    def static(self) -> bool:
        """
        Returns True if the representations are static, i.e., not trained. Otherwise False.
        This allows models to know when caching is appropriate.
        """
        return self.enc_spec()['static']


class Vocab:
    """
    Represents a vocabulary and corresponding neural encoding technique (e.g., embedding)
    """
    name = None

    @staticmethod
    def default_config():
        """
        Configuration for vocabulary
        """
        return {}

    def __init__(self, config, logger):
        self.config = config
        self.logger = logger

    def tokenize(self, text):
        """
        Meant to be overwritten in to provide vocab-specific tokenization when necessary
        e.g., BERT's WordPiece tokenization
        """
        text = text.lower()
        text = re.sub(r'[^a-z0-9]', ' ', text)
        return text.split()

    def tok2id(self, tok: str) -> int:
        """
        Converts a token to an integer id
        """
        raise NotImplementedError

    def id2tok(self, idx: int) -> str:
        """
        Converts an integer id to a token
        """
        raise NotImplementedError

    def path_segment(self) -> str:
        """
        Human-readable and FS-safe path segment for storing stuff related to this vocab on disk
        """
        raise NotImplementedError

    def lexicon_path_segment(self) -> str:
        """
        Human-readable and FS-safe path segment for storing stuff related only to model inputs
        (i.e., lexicon, but not weights).
        """
        raise NotImplementedError

    def encoder(self) -> VocabEncoder:
        """
        Encodes batches of id sequences
        """
        raise NotImplementedError

    def lexicon_size(self) -> int:
        """
        Returns the number of items in the lexicon
        """
        raise NotImplementedError
