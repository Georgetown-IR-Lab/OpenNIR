import os
import pickle
import hashlib
import numpy as np
import torch
from torch import nn
from onir import vocab, util, config
from onir.interfaces import wordvec


_SOURCES = {
    'fasttext': {
        'wiki-news-300d-1M': wordvec.zip_handler('https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M.vec.zip'),
        'crawl-300d-2M': wordvec.zip_handler('https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M.vec.zip'),
    },
    'glove': {
        'cc-42b-300d': wordvec.zip_handler('http://nlp.stanford.edu/data/glove.42B.300d.zip', ext='.txt'),
        'cc-840b-300d': wordvec.zip_handler('http://nlp.stanford.edu/data/glove.840B.300d.zip', ext='.txt')
    },
    'convknrm': {
        'knrm-bing': wordvec.convknrm_handler('http://boston.lti.cs.cmu.edu/appendices/WSDM2018-ConvKNRM/K-NRM/bing/'),
        'knrm-sogou': wordvec.convknrm_handler('http://boston.lti.cs.cmu.edu/appendices/WSDM2018-ConvKNRM/K-NRM/sogou/'),
        'convknrm-bing': wordvec.convknrm_handler('http://boston.lti.cs.cmu.edu/appendices/WSDM2018-ConvKNRM/Conv-KNRM/bing/'),
        'convknrm-sogou': wordvec.convknrm_handler('http://boston.lti.cs.cmu.edu/appendices/WSDM2018-ConvKNRM/Conv-KNRM/sogou/')
    },
    'bionlp': {
        'pubmed-pmc': wordvec.gensim_w2v_handler('http://evexdb.org/pmresources/vec-space-models/PubMed-and-PMC-w2v.bin')
    },
    'nil': wordvec.nil_handler
}


@vocab.register('wordvec')
class WordvecVocab(vocab.Vocab):
    """
    A word vector vocabulary that supports standard pre-trained word vectors
    """
    @staticmethod
    def default_config():
        return {
            'source': config.Choices(['fasttext', 'glove', 'convknrm', 'bionlp', 'nil']),
            'variant': 'wiki-news-300d-1M',
            'train': False
        }

    def __init__(self, config, logger, random):
        super().__init__(config, logger)
        self.random = random
        path = util.path_vocab(self)
        cache_path = os.path.join(path, '{source}-{variant}.p'.format(**self.config))
        if not os.path.exists(cache_path):
            fn = _SOURCES[self.config['source']]
            if isinstance(fn, dict):
                fn = fn[self.config['variant']]
            self._terms, self._weights = fn(self.logger)
            with logger.duration(f'writing cached at {cache_path}'):
                with open(cache_path, 'wb') as f:
                    pickle.dump((self._terms, self._weights), f, protocol=4)
        else:
            with logger.duration(f'reading cached at {cache_path}'):
                with open(cache_path, 'rb') as f:
                    self._terms, self._weights = pickle.load(f)
        self._term2idx = {t: i for i, t in enumerate(self._terms)}

    def tok2id(self, tok):
        return self._term2idx[tok]

    def id2tok(self, idx):
        return self._terms[idx]

    def path_segment(self):
        result = '{name}_{source}_{variant}'.format(name=self.name, **self.config)
        if self.config['train']:
            result += '_tr'
        return result

    def encoder(self):
        return WordvecEncoder(self)

    def lexicon_path_segment(self):
        return '{source}_{variant}'.format(**self.config)

    def lexicon_size(self) -> int:
        return len(self._terms)


@vocab.register('wordvec_unk')
class WordvecUnkVocab(WordvecVocab):
    """
    A vocabulary in which all unknown terns are given the same token (UNK; 0), with random weights
    """
    def __init__(self, config, logger, random):
        super().__init__(config, logger, random)
        self._terms = [None] + self._terms
        for term in self._term2idx:
            self._term2idx[term] += 1
        unk_weights = random.normal(scale=0.5, size=(1, self._weights.shape[1]))
        self._weights = np.concatenate([unk_weights, self._weights])

    def tok2id(self, tok):
        return self._term2idx.get(tok, 0)

    def lexicon_path_segment(self):
        return '{base}_unk'.format(base=super().lexicon_path_segment())

    def lexicon_size(self) -> int:
        return len(self._terms) + 1


@vocab.register('wordvec_hash')
class WordvecHashVocab(WordvecVocab):
    """
    A vocabulary in which all unknown terms are assigned a position in a flexible cache based on
    their hash value. Each position is assigned its own random weight.
    """
    @staticmethod
    def default_config():
        result = WordvecVocab.default_config().copy()
        result.update({
            'hashspace': 1000,
            'init_stddev': 0.5,
            'log_miss': False
        })
        return result

    def __init__(self, config, logger, random):
        super().__init__(config, logger, random)
        self._hashspace = config['hashspace']
        hash_weights = random.normal(scale=config['init_stddev'],
                                     size=(self._hashspace, self._weights.shape[1]))
        self._weights = np.concatenate([self._weights, hash_weights])

    def tok2id(self, tok):
        try:
            return super().tok2id(tok)
        except KeyError:
            if self.config['log_miss']:
                self.logger.debug(f'vocab miss {tok}')
            # NOTE: use md5 hash (or similar) here because hash() is not consistent across runs
            item = tok.encode()
            item_hash = int(hashlib.md5(item).hexdigest(), 16)
            item_hash_pos = item_hash % self._hashspace
            return len(self._terms) + item_hash_pos

    def lexicon_path_segment(self):
        return '{base}_hash{hashspace}'.format(**self.config, base=super().lexicon_path_segment())

    def lexicon_size(self) -> int:
        return len(self._terms) + self.config['hashspace']


class WordvecEncoder(vocab.VocabEncoder):

    def __init__(self, vocabulary):
        super().__init__(vocabulary)
        matrix = vocabulary._weights
        self.size = matrix.shape[1]
        matrix = np.concatenate([np.zeros((1, self.size)), matrix]) # add padding record (-1)
        self.embed = nn.Embedding.from_pretrained(torch.from_numpy(matrix.astype(np.float32)), freeze=not vocabulary.config['train'])

    def _enc_spec(self) -> dict:
        return {
            'dim': self.size,
            'views': 1,
            'static': True,
            'vocab_size': self.embed.weight.shape[0]
        }

    def forward(self, toks, lens=None):
        # lens ignored
        return self.embed(toks + 1) # +1 to handle padding at position -1
