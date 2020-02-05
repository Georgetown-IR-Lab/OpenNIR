from onir import util


@util.allow_redefinition_iter
def doc_iter(dataset,
             fields: set,
             shuf: bool = False,
             random=None
            ):
    dids = dataset.all_doc_ids()
    if shuf:
        dids = list(dids)
        random.shuffle(dids)
    for did in dids:
        record = dataset.build_record(fields, doc_id=did)
        yield {f: record[f] for f in fields}

class DocIter:
    def __init__(self, dataset, fields, shuf=False, random=None):
        self.it = doc_iter(dataset, fields, shuf, random)
        self.consumed = 0
        self.len = dataset.num_docs()

    def __next__(self):
        self.consumed += 1
        return next(self.it)

    def __iter__(self):
        return self

    def __length_hint__(self):
        return max(self.len - self.consumed, 0)
