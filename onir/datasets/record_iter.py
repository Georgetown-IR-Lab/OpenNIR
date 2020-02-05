import pandas as pd
from onir import util


@util.allow_redefinition_iter
def record_iter(dataset,
                fields: set,
                source: str, # one of ['run', 'qrels']
                minrel: int = None, # integer indicating the minimum relevance score, or None for unfiltered
                shuf: bool = True,
                random=None,
                inf: bool = False
               ):
    run_fn = util.Lazy(lambda: dataset.run('df'))
    qrels_fn = util.Lazy(lambda: dataset.qrels('df'))

    if source == 'run':
        src = run_fn()
        if minrel is not None:
            src = src[src['score'] >= minrel]
            src = pd.merge(src, qrels_fn(), on=['qid', 'did'], suffixes=('_run', '_qrels'))
            src = src[src['score_qrels'] >= minrel]
    elif source == 'qrels':
        src = qrels_fn()
        if minrel is not None:
            src = src[src['score'] >= minrel]
    else:
        raise ValueError(f'unsupported source {source}')
    src = src.filter(items=['qid', 'did'])

    it = record_iter_sample(dataset, src, shuf, random, inf)

    for qid, did in it:
        record = dataset.build_record(fields, query_id=qid, doc_id=did)
        yield {f: record[f] for f in fields}


@util.allow_redefinition_iter
def run_iter(dataset, fields, minrel=None, shuf=False, random=None, inf=False):
    return record_iter(dataset, fields, 'run', minrel, shuf, random, inf)


@util.allow_redefinition_iter
def qrels_iter(dataset, fields, minrel=None, shuf=False, random=None, inf=False):
    return record_iter(dataset, fields, 'qrels', minrel, shuf, random, inf)


@util.allow_redefinition_iter
def pos_qrels_iter(dataset, fields, minrel=1, shuf=False, random=None, inf=False):
    return record_iter(dataset, fields, 'qrels', minrel, shuf, random, inf)


@util.allow_redefinition_iter
def record_iter_sample(dataset, cand, shuf, random, inf=False):
    first = True
    while first or inf:
        first = False
        if shuf:
            cand = cand.sample(frac=1., random_state=random)
        for _, sample in cand.iterrows():
            yield sample['qid'], sample['did']
