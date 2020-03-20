

class BaseIndex:
    def built(self) -> bool:
        raise NotImplementedError

    def build(self, doc_iter):
        raise NotImplementedError

    def num_docs(self) -> bool:
        raise NotImplementedError
