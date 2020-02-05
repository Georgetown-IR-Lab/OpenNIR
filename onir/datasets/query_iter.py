from onir import util


@util.allow_redefinition_iter
def query_iter(dataset,
               fields: set,
               shuf: bool = False,
               random=None
              ):
    qids = dataset.all_query_ids()
    if shuf:
        qids = list(qids)
        random.shuffle(qids)
    for qid in qids:
        record = dataset.build_record(fields, query_id=qid)
        yield {f: record[f] for f in fields}


class QueryIter:
    def __init__(self, dataset, fields, shuf=False, random=None):
        self.it = query_iter(dataset, fields, shuf, random)
        self.consumed = 0
        self.len = dataset.num_queries()

    def __next__(self):
        self.consumed += 1
        return next(self.it)

    def __iter__(self):
        return self

    def __length_hint__(self):
        return max(self.len - self.consumed, 0)
