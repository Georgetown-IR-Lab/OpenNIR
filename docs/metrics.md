## Metrics


### Overview

OpenNIR supports a variety of standard information retrieval metrics by interfacing with tools such
as `trec_eval` and `gdeval.pl`.

#### Naming of metrics

Because the naming conventions of metrics in these tools is
inconsistent, we made a [new convention](https://xkcd.com/927/) of specifying the metric names and
parameters: `metric_par1-val1_par2-val2@cutoff`. Note that parameters are not necessarily applicable
to all metrics, and are sometimes optional. For instance, nDCG optionally supports a cutoff, but
does not support the ranking cutoff parameter.

#### Usage in the standard pipeline

The metrics that you want to calculate for each validaiton/test epcoh are specified with
`valid_pred.measures=metric1,metric2,...` and `test_pred.measures=...`. To choose the primary
metric for validation, use `pipeline.val_metric=metric`.

#### API

Metrics can be calcualted by using the `onir.metric.calc(qrels, run, metrics)` function.

Inputs:
 - `qrels`: Path to TREC-style qrels file, or dictionary in form of `{qid: {did: rel_score}}`
 - `run`: Path to TREC-style run file, or dictionary in form of `{qid: {did: rank_score}}`
 - `metrics`: Iterable of metric names to calculate

Outputs:
 - Dicitonary containing metric values by query in the form `{metric: {qid: score}}`

The the mean values across a collection of queries can be calculated using `onir.metrics.mean(vals)`

#### Command line tool

`scripts/eval qrels_file run_file metric1 [metric2, ...] [-v] [-j] [-n] [-q]`

 - `qrels_file`: TREC-formatted query relevance file
 - `run_file`: TREC-formatted run file
 - `metric1 metric2 ...`: metric names to run
 - `-v`: verbose output
 - `-j` json-formatted output
 - `-n` no summary output
 - `-q` output by query

### Supported metrics

#### Precision @ k

`p[_rel-r]@k`, e.g., `p@10`, `p_rel-4@1`

 - `r`: (optional) minimum relevance level (defualt 1)
 - `k`: (required) ranking cutoff threshold

#### Mean Average Precision

`map[_rel-r][@k]`, e.g., `map`, `map@100`, `map_rel-4@100`

 - `r`: (optional) minimum relevance level (defualt 1)
 - `k`: (optional) ranking cutoff threshold

#### R-Precision

`rprec[_rel-r]`, e.g., `rprec`, `rprec_rel-4`

 - `r`: (optional) minimum relevance level (defualt 1)

#### Mean Reciprocal Rank

`mrr[_rel-r]`, e.g., `mrr`, `mrr_rel-4`

 - `r`: (optional) minimum relevance level (defualt 1)

#### Expected Reciprocal Rank

`err@k`, e.g., `err@20`

 - `k`: (required) ranking cutoff threshold

#### Normalized Discounted Cumulative Gain

`ndcg[@k]`, e.g., `ndcg`, `ndcg@20`

 - `k`: (optional) ranking cutoff threshold

#### Judged @ k

`judged@k`, e.g., `judged@20`

Number of judged documents (any relevance level) at level `k`

 - `k`: (required) ranking cutoff threshold


### Metirc Providers


#### `PyTrecEvalMetrics`

From the `pytrec-eval` python package ([link](https://github.com/cvangysel/pytrec_eval)). Interfaces
direclty with the `trec_eval` code, making it more efficient than spawning a `trec_eval` subprocess,
and allowing better support accross platforms.

Intorduction paper of `pytrec_eval`: Van Gysel, Christophe, and Maarten de Rijke. "Pytrec\_eval:
An Extremely Fast Python Interface to trec\_eval." SIGIR 2018.

Supported metrics: `map`, `rprec`, `mrr`, `p@5,10,15,20,30,100,200,500,1000`,
`ndcg`, `ndcg@5,10,15,20,30,100,200,500,1000`, `map@5,10,15,20,30,100,200,500,1000`.
Note: [does not support custom cutoff thresholds](https://github.com/cvangysel/pytrec_eval/issues/12),
or relevance levels (`_rel-r`).


#### `TrecEvalMetrics`

Starts `trec_eval` (link)[https://trec.nist.gov/trec_eval/] as a sub-process. Not supported by all
platforms, but does support more metrics and features than `PyTrecEvalMetrics`.

Supported metrics: `mrr[_rel-r]`, `rprec[_rel-r]`, `map[_rel-r][@k]`, `ndcg[@k]`, `p[_rel-r]@k`


#### `GdevalMetrics`

Starts `gdeval.pl` (link)[https://trec.nist.gov/data/web/12/eval-README.txt] as a sub-process.
Requires perl in system PATH.

Supported metrics: `ndcg@k`, `err@k`. Note: does not support relevance levels.


#### `JudgedMetrics`

Calcuates the percentage of documents that are judged in the top `k` ranked documents per query
(in python).

Only supports `judged@k`.
