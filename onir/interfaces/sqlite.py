import sqlitedict


class Sqlite2KeyDict(sqlitedict.SqliteDict):
    """
    Adapated from sqlitedict.SqliteDict with support for two keys
    """
    VALID_FLAGS = ['c', 'r', 'w', 'n']

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tablename = f'{self.tablename}-2'

        MAKE_TABLE = f'CREATE TABLE IF NOT EXISTS "{self.tablename}" (key1 TEXT, key2 TEXT, value BLOB, PRIMARY KEY (key1, key2))'
        self.conn = self._new_conn()
        self.conn.execute(MAKE_TABLE)
        self.conn.commit()

    def __str__(self):
        return f'Sqlite2KeyDict({self.filename})'

    def iterkeys(self):
        GET_KEYS = f'SELECT key1, key2 FROM "{self.tablename}" ORDER BY rowid'
        for key1, key2 in self.conn.select(GET_KEYS):
            yield key1, key2

    def iterkey1s(self):
        GET_KEYS = f'SELECT DISTINCT key1 FROM "{self.tablename}" ORDER BY rowid'
        for key in self.conn.select(GET_KEYS):
            yield key[0]

    def countkey1s(self):
        LEN_KEYS = f'SELECT COUNT(DISTINCT key1) FROM "{self.tablename}"'
        return list(self.conn.select(LEN_KEYS))[0][0]

    def iterkey2s(self):
        GET_KEYS = f'SELECT DISTINCT key2 FROM "{self.tablename}" ORDER BY rowid'
        for key in self.conn.select(GET_KEYS):
            yield key[0]

    def countkey2s(self):
        LEN_KEYS = f'SELECT COUNT(DISTINCT key2) FROM "{self.tablename}"'
        return list(self.conn.select(LEN_KEYS))[0][0]

    def itervalues(self):
        GET_VALUES = f'SELECT value FROM "{self.tablename}" ORDER BY rowid'
        for value in self.conn.select(GET_VALUES):
            yield self.decode(value[0])

    def iteritems(self):
        GET_ITEMS = f'SELECT key1, key2, value FROM "{self.tablename}" ORDER BY rowid'
        for key1, key2, value in self.conn.select(GET_ITEMS):
            yield key1, key2, self.decode(value)

    def key1s(self):
        return self.iterkey1s()

    def key2s(self):
        return self.iterkey2s()

    def __contains__(self, key):
        assert isinstance(key, tuple) and len(key) == 2
        key1, key2 = key
        if key1 is not Ellipsis and key2 is Ellipsis:
            HAS_ITEM = 'SELECT 1 FROM "%s" WHERE key1 = ?' % self.tablename
            return self.conn.select_one(HAS_ITEM, (key1,)) is not None
        if key1 is Ellipsis and key2 is not Ellipsis:
            HAS_ITEM = 'SELECT 1 FROM "%s" WHERE key2 = ?' % self.tablename
            return self.conn.select_one(HAS_ITEM, (key2,)) is not None
        if key1 is not Ellipsis and key2 is not Ellipsis:
            HAS_ITEM = 'SELECT 1 FROM "%s" WHERE key1 = ? AND key2 = ?' % self.tablename
            return self.conn.select_one(HAS_ITEM, (key1, key2)) is not None
        raise ValueError('must provide at least one key')

    def __getitem__(self, key):
        assert isinstance(key, tuple) and len(key) == 2
        key1, key2 = key
        GET_ITEM = 'SELECT value FROM "%s" WHERE key1 = ? AND key2 = ?' % self.tablename
        item = self.conn.select_one(GET_ITEM, (key1, key2))
        if item is None:
            raise KeyError(key)
        return self.decode(item[0])

    def __setitem__(self, key, value):
        if self.flag == 'r':
            raise RuntimeError('Refusing to write to read-only SqliteDict')

        assert isinstance(key, tuple) and len(key) == 2
        key1, key2 = key

        ADD_ITEM = 'REPLACE INTO "%s" (key1, key2, value) VALUES (?,?,?)' % self.tablename
        self.conn.execute(ADD_ITEM, (key1, key2, self.encode(value)))

    def __delitem__(self, key):
        if self.flag == 'r':
            raise RuntimeError('Refusing to delete from read-only SqliteDict')

        if key not in self:
            raise KeyError(key)
        assert isinstance(key, tuple) and len(key) == 2
        key1, key2 = key
        if key1 is not Ellipsis and key2 is Ellipsis:
            DEL_ITEM = 'DELETE FROM "%s" WHERE key1 = ?' % self.tablename
            del_args = (key1,)
        elif key1 is Ellipsis and key2 is not Ellipsis:
            DEL_ITEM = 'DELETE FROM "%s" WHERE key2 = ?' % self.tablename
            del_args = (key2,)
        elif key1 is not Ellipsis and key2 is not Ellipsis:
            DEL_ITEM = 'DELETE FROM "%s" WHERE key1 = ? AND key2 = ?' % self.tablename
            del_args = (key1, key2)
        else:
            raise ValueError('must provide at least one key')
        self.conn.execute(DEL_ITEM, del_args)

    def lookup_value(self, value):
        GET_ITEM = 'SELECT key1, key2 FROM "%s" WHERE value = ?' % self.tablename
        item = self.conn.select_one(GET_ITEM, (value, ))
        return item

    def update(self, items):
        if self.flag == 'r':
            raise RuntimeError('Refusing to update read-only SqliteDict')

        try:
            items = items.items()
        except AttributeError:
            pass
        items = [(k1, k2, self.encode(v)) for k1, k2, v in items]

        UPDATE_ITEMS = 'REPLACE INTO "%s" (key1, key2, value) VALUES (?, ?, ?)' % self.tablename
        self.conn.executemany(UPDATE_ITEMS, items)
