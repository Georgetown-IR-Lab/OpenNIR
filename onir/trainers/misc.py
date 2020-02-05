import os
from onir.interfaces import plaintext

class TsvFileManager:
    def __init__(self, path):
        self.path = path
        if os.path.exists(path):
            self.content = {int(e): float(v) for e, v in plaintext.read_tsv(path)}
        else:
            self.content = {}

    def __getitem__(self, epoch):
        return self.content.get(epoch, -999)

    def __contains__(self, epoch):
        return epoch in self.content

    def __setitem__(self, epoch, value):
        self.content[epoch] = float(value)
        with open(self.path, 'at') as f:
            plaintext.write_tsv(f, [(str(epoch), str(value))])

class DirectoryManager:
    def __init__(self, base_path):
        self.base_path = base_path
        os.makedirs(base_path, exist_ok=True)

    def __getitem__(self, path):
        return os.path.join(self.base_path, path)

    def __contains__(self, path):
        return os.path.exists(self[path])

    def open(self, path, mode='rb'):
        with open(self[path], mode) as f:
            return f


class PathManager:
    def __init__(self, base_dir):
        self.base_dir = base_dir
        self.cache = {}

    def __getitem__(self, key):
        if key not in self.cache:
            self.cache[key] = self.load(key)
        return self.cache[key]

    def load(self, key):
        if key.endswith('.txt') or key.endswith('.tsv'):
            return TsvFileManager(os.path.join(self.base_dir, key))
        return DirectoryManager(os.path.join(self.base_dir, key))
