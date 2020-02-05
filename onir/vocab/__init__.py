# pylint: disable=C0413
from onir import util
registry = util.Registry(default='wordvec_hash')
register = registry.register


from onir.vocab.base import Vocab, VocabEncoder
from onir.vocab.trivial_vocab import TrivialVocab
from onir.vocab.bert_vocab import BertVocab
from onir.vocab.wordvec_vocab import WordvecVocab, WordvecUnkVocab, WordvecHashVocab
