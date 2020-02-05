from onir import vocab


@vocab.register('trivial')
class TrivialVocab(vocab.Vocab):
    @staticmethod
    def default_config():
        return {}

    def tok2id(self, tok):
        return -1

    def id2tok(self, idx):
        return None

    def path_segment(self):
        result = '{name}'.format(name=self.name, **self.config)
        return result

    def lexicon_path_segment(self):
        return 'trivial'

    def encoder(self):
        raise NotImplementedError()
