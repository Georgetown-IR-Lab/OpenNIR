import os
import onir
from onir.interfaces.sqlite import Sqlite2KeyDict


FIELD_RAW = '__raw__'


_logger = onir.log.easy()


class SqliteDocstore:
    def __init__(self, path, field='text'):
        self._path = path
        self._field = field
        self._sql = None

    def sql(self):
        if self._sql is None:
            self._sql = Sqlite2KeyDict(self._path, tablename=self._field, autocommit=False)
        return self._sql

    def built(self):
        return os.path.exists(self._path)

    def build(self, documents, replace=False):
        if not replace and self.built():
            return
        if self._sql:
            self._sql.close()
            self._sql = None
        MAX_BATCH = 1000
        tmp_sql = Sqlite2KeyDict(f'{self._path}.tmp', tablename=self._field, autocommit=False)
        with _logger.duration(f'building {self._path}'):
            batch = []
            for doc in documents:
                batch.append((FIELD_RAW, doc.did, doc.data[self._field]))
                if len(batch) >= MAX_BATCH:
                    tmp_sql.update(batch)
                    tmp_sql.commit()
                    batch = []
            if batch:
                tmp_sql.update(batch)
                tmp_sql.commit()
            os.replace(f'{self._path}.tmp', self._path)

    def get_raw(self, did):
        return self.sql()[FIELD_RAW, did]

    def path(self):
        return self._path

    def get_tokenized(self, did, vocabulary):
        lexicon = vocabulary.lexicon_path_segment()
        if (lexicon, did) not in self.sql():
            doc_tok = vocabulary.tokenize(self.get_raw(did))
            doc_tok = [vocabulary.tok2id(tok) for tok in doc_tok]
            self.sql()[lexicon, did] = doc_tok
            self.sql().commit()
            return doc_tok
        return self.sql()[lexicon, did]

    def docids(self):
        if self.built():
            for did in self.sql().iterkey2s():
                if did != '__built__': # legacy
                    yield did

    def num_docs(self):
        if self.built():
            return self.sql().countkey2s()
