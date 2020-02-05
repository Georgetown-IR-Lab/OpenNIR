import pandas as pd
import numpy as np
from onir import util
from onir.interfaces import trec


@util.allow_redefinition_iter
def pair_iter(dataset,
              fields: set,
              pos_source: str = 'intersect', # one of ['qrels', 'intersect']
              neg_source: str = 'run', # one of ['run', 'qrels', 'union']
              sampling: str = 'query', # one of ['query', 'qrel']
              pos_minrel: int = 1,
              unjudged_rel: int = 0,
              num_neg: int = 1,
              random=None,
              inf: bool = False
             ):
    qrels_fn = util.Lazy(lambda: dataset.qrels(fmt='df'))
    run_fn = util.Lazy(lambda: dataset.run(fmt='df'))

    pos_candidates = {
        'qrels': pair_iter_pos_candidates_qrels,
        'intersect': pair_iter_pos_candidates_intersect,
    }[pos_source](dataset, qrels_fn, run_fn, pos_minrel)
    assert len(pos_candidates.index) > 0

    neg_candidates = {
        'run': pair_iter_neg_candidates_run,
        'qrels': pair_iter_neg_candidates_qrels,
        'union': pair_iter_neg_candidates_union,
    }[neg_source](dataset, qrels_fn, run_fn, unjudged_rel)
    neg_candidates = neg_candidates.set_index('qid')
    neg_candidates.sort_index(inplace=True)
    assert len(neg_candidates.index) > 0

    pos_iter = {
        'query': pair_iter_sample_by_query,
        'qrel': pair_iter_sample_by_qrel
    }[sampling](dataset, pos_candidates, random, inf)

    for qid, pos_did, score in pos_iter:
        negs = pair_iter_filter_neg(dataset, neg_candidates, qid, pos_did, score)
        if len(negs.index) < num_neg:
            # not enough negative documents for this positive sample
            dataset.logger.debug('not enough negs')
            continue
        dids = [pos_did]
        negs = negs['did'].sample(n=num_neg, random_state=random)
        for did in negs:
            dids.append(did)

        result = {f: [] for f in fields}
        for did in dids:
            record = dataset.build_record(fields, query_id=qid, doc_id=did)
            for f in fields:
                result[f].append(record[f])
        yield result


@util.allow_redefinition_iter
def pair_iter_sample_by_query(dataset, candidates, random, inf):
    first = True
    candidates = candidates.set_index('qid')
    candidates.sort_index(inplace=True)
    qids = candidates.index.unique()
    while inf or first:
        first = False
        seq = random.permutation(len(qids))
        for i in seq:
            qid = qids[i]
            samples = candidates.loc[qid:qid]
            samples = samples.sample(n=1, random_state=random)
            for _, sample in samples.iterrows():
                yield qid, sample['did'], sample['score']


@util.allow_redefinition_iter
def pair_iter_sample_by_qrel(dataset, candidates, random, inf):
    if inf:
        while True:
            sample = candidates.sample(n=1, random_state=random).iloc[0]
            yield sample['qid'], sample['did'], sample['score']
    else:
        samples = candidates.sample(frac=1., random_state=random)
        for _, sample in samples.iterrows():
            yield sample['qid'], sample['did'], sample['score']


@util.allow_redefinition
def pair_iter_neg_candidates_run(dataset, qrels_fn, run_fn, unjudged_rel):
    cand = pd.merge(qrels_fn(), run_fn(), how='right', on=['qid', 'did'], suffixes=('', '_del'))
    cand = cand.filter(items=['qid', 'did', 'score'])
    cand.loc[cand['score'].isnull(), 'score'] = unjudged_rel
    return cand

@util.allow_redefinition
def pair_iter_neg_candidates_qrels(dataset, qrels_fn, run_fn, unjudged_rel):
    return qrels_fn()

@util.allow_redefinition
def pair_iter_neg_candidates_union(dataset, qrels_fn, run_fn, unjudged_rel):
    cand = pd.merge(qrels_fn(), run_fn(), how='outer', on=['qid', 'did'], suffixes=('', '_del'))
    cand = cand.filter(items=['qid', 'did', 'score'])
    cand.loc[cand['score'].isnull(), 'score'] = unjudged_rel
    return cand


@util.allow_redefinition
def pair_iter_pos_candidates_qrels(dataset, qrels_fn, run_fn, pos_minrel):
    return qrels_fn()[qrels_fn()['score'] >= pos_minrel]

@util.allow_redefinition
def pair_iter_pos_candidates_intersect(dataset, qrels_fn, run_fn, pos_minrel):
    cand = pair_iter_pos_candidates_qrels(dataset, qrels_fn, run_fn, pos_minrel)
    cand = pd.merge(cand, run_fn(), on=['qid', 'did'], suffixes=('', '_del'))
    cand = cand.filter(items=['qid', 'did', 'score'])
    return cand


@util.allow_redefinition
def pair_iter_filter_neg(dataset, neg_candidates, qid, did, score):
    # negs = neg_candidates
    negs = neg_candidates.loc[qid:qid]
    negs = negs[negs['score'] < score]
    # negs = negs[negs.index < score]
    return negs
