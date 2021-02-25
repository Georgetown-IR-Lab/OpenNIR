import os
import onir
from onir import indices


_logger = onir.log.easy()


class IrdsDocstore(indices.BaseIndex):
    def __init__(self, docstore, fields):
        self._docstore = docstore
        self._fields = fields.split(',')

    def built(self):
        return True

    def build(self, documents, replace=False):
        raise NotImplementedError()

    def get_raw(self, did):
        doc = self._docstore.get(did)
        result = []
        for field in self._fields:
            fidx = doc._fields.index(field)
            result.append(doc[fidx])
        return '\n'.join(result)

    def path(self):
        raise NotImplementedError()

    def docids(self):
        for doc in self._docstore:
            yield doc.doc_id

    def num_docs(self):
        return self._docstore.count()
