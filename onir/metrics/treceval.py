import re
import subprocess
import tempfile
from contextlib import ExitStack
import onir
from onir import metrics as _metrics
from onir.interfaces import trec

logger = onir.log.easy()


OPT_REL = r'(_rel-(?P<rel>\d+))?'
OPT_GAIN = r'(_gain-(?P<gains>\d+=\d+(:\d+=\d+)+))?'
CUT = r'@(?P<par>\d+)'

# Defines metrics available in trec_eval and supported by this module.
# The key defines the a regular expression that matches a given metric string. The value is a tuple
# that defines the input name to trec_eval (if has <par>, this value gets appended using trec_eval
# conventions), and the trec_eval output name (which may contain the parameter value from the regex)
# The relevance level parameter is supported by some metrics, and indicated with metric_l-X, where X
# is the minimum value considered relevant (e.g., p-l-4@5 is precision at 5 of documents graded at
# least a relevance of 4).
TE_METRICS = {
    rf'^mrr{OPT_REL}$': ('recip_rank', 'recip_rank'),
    rf'^rprec{OPT_REL}$': ('Rprec', 'Rprec'),
    rf'^map{OPT_REL}$': ('map', 'map'),
    rf'^map{OPT_REL}{CUT}$': ('map_cut', 'map_cut_{par}'),
    rf'^ndcg{OPT_GAIN}$': ('ndcg', 'ndcg'),
    rf'^ndcg{OPT_GAIN}{CUT}$': ('ndcg_cut.', 'ndcg_cut_{par}'),
    rf'^p{OPT_REL}{CUT}$': ('P.', 'P_{par}'),
    rf'^r{OPT_REL}{CUT}$': ('recall.', 'recall_{par}'),
    # TODO: support more metrics by adding them here
}


class TrecEvalMetrics(_metrics.BaseMetrics):
    """
    Slower than PyTrecEvalMetrics (needs to spawn subprocess), but supports a wider variety of
    features, e.g., setting relevance level for metrics like MAP/Rprec, etc. or settings specific
    cutoff thresholds for metrics like P@X or ndcg@X.
    """
    QRELS_FORMAT = 'file'
    RUN_FORMAT = 'file'

    def supports(self, metric):
        metric = _metrics.Metric.parse(metric)
        if metric is None:
            return False
        metric = str(metric)
        for exp in TE_METRICS:
            if re.match(exp, metric):
                return True
        return False

    def calc_metrics(self, qrelsf, runf, metrics, verbose=False):
        rel_args = {}
        output_map = {}
        for metric in metrics:
            for exp in TE_METRICS:
                match = re.match(exp, str(metric))
                if match:
                    params = match.groupdict()
                    rel, par, gain = params.get('rel'), params.get('par'), params.get('gain')
                    rel_args.setdefault((rel, gain), {})
                    in_name, out_name = TE_METRICS[exp]
                    if in_name not in rel_args[rel, gain]:
                        rel_args[rel, gain][in_name] = in_name
                        if par is not None:
                            rel_args[rel, gain][in_name] += par
                    elif par is not None:
                        rel_args[rel, gain][in_name] += f',{par}'
                    output_map[metric] = (rel, gain, out_name.format(**params))
                    break

        full_trec_eval_result = {}
        for rel_level, gains in rel_args:
            with ExitStack() as stack:
                if verbose:
                    logger.debug(f'running trec_eval (rel_level={rel_level} gains={gains})')
                if gains:
                    gain_map = [g.split('=') for g in gains.split(':')]
                    gain_map = {int(k): int(v) for k, v in gain_map}
                    tmpf = tempfile.NamedTemporaryFile()
                    stack.enter_context(tmpf)
                    qrels = trec.read_qrels_dict(qrelsf)
                    for qid in qrels:
                        for did in qrels[qid]:
                            qrels[qid][did] = gain_map.get(qrels[qid][did], qrels[qid][did])
                    trec.write_run_dict(tmpf, qrels)
                    qrelsf_run = tmpf.name
                else:
                    qrelsf_run = qrelsf
                trec_eval_result = self._treceval(qrelsf_run, runf, rel_args[rel_level, gains].values(), rel_level)
                for k, v in trec_eval_result.items():
                    full_trec_eval_result[rel_level, gains, k] = v

        results = {}
        for onir_metric, te_metric in output_map.items():
            results[str(onir_metric)] = full_trec_eval_result[te_metric]

        return results

    def _treceval(self, qrelsf, runf, metric_args, rel_level=None):
        """
        Runs trec_eval on the given run/qrels pair
        """
        trec_eval_f = 'bin/trec_eval' # TODO: ensure correct path by building based on module path?
        args = [trec_eval_f, '-q', '-c', '-n']
        if rel_level is not None:
            args.append(f'-l{rel_level}')
        for metric_arg in metric_args:
            args += ['-m', metric_arg]
        args += [qrelsf, runf]
        try:
            output = subprocess.check_output(args).decode()
        except:
            import pdb; pdb.set_trace()
            pass
        output = output.replace('\t', ' ').split('\n')
        result = {}
        for line in output:
            if not line:
                continue
            m, qid, v = line.split()
            result.setdefault(m, {})[qid] = float(v)
        return result
