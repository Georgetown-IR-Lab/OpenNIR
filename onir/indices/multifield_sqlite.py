import os
import onir
from onir.interfaces.sqlite import Sqlite2KeyDict


_logger = onir.log.easy()


class MultifieldSqliteDocstore:
    def __init__(self, path, primary_field='text'):
        self._path = path
        self._primary_field = primary_field
        self._sql = None

    def sql(self):
        if self._sql is None:
            self._sql = Sqlite2KeyDict(self._path, tablename='default', autocommit=False)
        return self._sql

    def built(self):
        return os.path.exists(self._path)

    def build(self, documents, replace=False, all_fields=None):
        # all_fields is ignored
        if not replace and self.built():
            return
        if self._sql:
            self._sql.close()
            self._sql = None
        MAX_BATCH = 50_000
        tmp_sql = Sqlite2KeyDict(f'{self._path}.tmp', tablename='default', autocommit=False)
        with _logger.duration(f'building {self._path}'):
            batch = []
            for doc in documents:
                for field in doc.data:
                    batch.append((field, doc.did, doc.data[field]))
                if len(batch) >= MAX_BATCH:
                    tmp_sql.update(batch)
                    tmp_sql.commit()
                    batch = []
            if batch:
                tmp_sql.update(batch)
                tmp_sql.commit()
            os.replace(f'{self._path}.tmp', self._path)

    def get_raw(self, did, field=None):
        return self.sql()[field or self._primary_field, did]

    def get_raws(self, dids, field=None):
        return self.sql().lookup(field or self._primary_field, dids)

    def path(self):
        return self._path

    def docids(self):
        if self.built():
            yield from self.sql().iterkey2s()

    def num_docs(self):
        if self.built():
            return self.sql().countkey2s()
