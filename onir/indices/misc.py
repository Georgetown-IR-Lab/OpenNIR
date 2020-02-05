class RawDoc:
    def __init__(self, did, text=None, **fields):
        self.did = did
        self.data = {}
        if text is not None:
            self.data['text'] = text
        self.data.update(fields)
        if 'all' not in self.data:
            self.data['all'] = '\n\n'.join(self.data.values())
