import math
import io
import tempfile
import inspect
import torch
import json
import tarfile
from collections import namedtuple
from typing import Union
import pandas as pd
import numpy as np
import pyterrier
import onir

_logger = onir.log.easy()


class PtValidator:
    def __init__(self, pt_transformer, va_run, va_qrels, config):
        self.pt_transformer = pt_transformer
        self.va_run = va_run
        self.va_qrels = va_qrels
        self.config = config

    def validate(self):
        with _logger.duration('validation'):
            res = self.pt_transformer.transform(self.va_run)
            metrics = pyterrier.Utils.evaluate(res, self.va_qrels, metrics=self.config['va_metrics'])
            primary_metric = metrics[self.config['va_metrics'][0]]
            return primary_metric, metrics


class PtPairTrainerBase:
    def __init__(self, pt_transformer, config):
        self.pt_transformer = pt_transformer
        self.config = config
        self._batch_iter = None
        params = self.pt_transformer.ranker.named_parameters()
        params = [v for k, v in params if v.requires_grad]
        self.optimizer = torch.optim.Adam(params, lr=config['learning_rate'])

    def lossfn(self, outputs):
        outputs = outputs.reshape(-1, 2)
        losses = outputs.softmax(dim=1)[:, 1]
        losses = losses.sum() / outputs.shape[0]
        return losses

    def train(self):
        device = onir.util.device(self.config, _logger)
        ranker = self.pt_transformer.ranker.to(device).train()
        total_loss = 0.
        count = 0
        with _logger.duration('training'), _logger.pbar_raw(desc='train pairs', tqdm=pyterrier.tqdm, total=self.config['train_count']) as pbar:
            while count < self.config['train_count']:
                if self._batch_iter is None:
                    self._batch_iter = self.batch_iter(device)
                for batch_count, batch in self._batch_iter:
                    outputs = ranker(**batch)
                    loss = self.lossfn(outputs)
                    loss.backward()
                    total_loss += loss.cpu().item()
                    count += batch_count
                    pbar.update(batch_count)
                    pbar.set_postfix({'loss': total_loss / count})
                    if count >= self.config['train_count']:
                        break
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                else:
                    # self._batch_iter is consumed; tell it to start over
                    self._batch_iter = None
        return total_loss / count

    def batch_iter(self, device):
        batch_size = self.config['batch_size'] // 2 # we're working with pairs here
        input_spec = self.pt_transformer.ranker.input_spec()
        fields = input_spec['fields']
        batch = {f: [] for f in fields}
        batch_count = 0
        for pair in self._pair_iter():
            query_text = self.pt_transformer.vocab.tokenize(pair.query)
            query_tok = [self.pt_transformer.vocab.tok2id(t) for t in query_text]
            doc_pos_text = self.pt_transformer.vocab.tokenize(pair.text_pos)
            doc_pos_tok = [self.pt_transformer.vocab.tok2id(t) for t in doc_pos_text]
            doc_neg_text = self.pt_transformer.vocab.tokenize(pair.text_neg)
            doc_neg_tok = [self.pt_transformer.vocab.tok2id(t) for t in doc_neg_text]
            for f in fields:
                if f == 'doc_id':
                    batch[f].append(rec.docno_pos)
                    batch[f].append(rec.docno_neg)
                elif f == 'query_id':
                    batch[f].append(pair.qid)
                    batch[f].append(pair.qid)
                elif f == 'query_rawtext':
                    batch[f].append(rec.query)
                    batch[f].append(rec.query)
                elif f == 'doc_rawtext':
                    batch[f].append(rec.text_pos)
                    batch[f].append(rec.text_neg)
                elif f == 'query_text':
                    batch[f].append(query_text)
                    batch[f].append(query_text)
                elif f == 'doc_text':
                    batch[f].append(doc_pos_text)
                    batch[f].append(doc_neg_text)
                elif f == 'query_tok':
                    batch[f].append(query_tok)
                    batch[f].append(query_tok)
                elif f == 'doc_tok':
                    batch[f].append(doc_pos_tok)
                    batch[f].append(doc_neg_tok)
                elif f == 'query_len':
                    batch[f].append(len(query_tok))
                    batch[f].append(len(query_tok))
                elif f == 'doc_len':
                    batch[f].append(len(doc_pos_tok))
                    batch[f].append(len(doc_neg_tok))
                else:
                    raise ValueError(f'unsupported field {f}')
            batch_count += 1
            if batch_count == batch_size:
                yield batch_count, onir.spec.apply_spec_batch(batch, input_spec, device)
                batch = {f: [] for f in fields}
                batch_count = 0
        if batch_count > 0:
            yield batch_count, onir.spec.apply_spec_batch(batch, input_spec, device)

    def _pair_iter(self):
        raise NotImplementedError()


class PtPairTrainer(PtPairTrainerBase):
    def __init__(self, pt_transformer, tr_pairs, config):
        super().__init__(pt_transformer, config)
        self.tr_pairs = tr_pairs
        self.rng = np.random.RandomState(config['random_seed'])

    def _pair_iter(self):
        if isinstance(self.tr_pairs, pd.DataFrame):
            df = self.tr_pairs.sample(frac=1, random_state=self.rng)
            return df.itertuples()
        return self.tr_pairs


class PtTopicQrelsTrainer(PtPairTrainerBase):
    def __init__(self, pt_transformer, tr_topics, tr_qrels, tr_run, config):
        super().__init__(pt_transformer, config)
        self.topic_lookup = {t.qid: t.query for t in tr_topics.itertuples()}
        self.doc_lookup = {t.docno: t.text for t in tr_run.itertuples()}
        self.tr_qrels = tr_qrels.rename(columns={'docno': 'did', 'label': 'score'})
        self.tr_run = tr_run.rename(columns={'docno': 'did'})
        self.rng = np.random.RandomState(config['random_seed'])
        self.logger = _logger # mock datast

    def _pair_iter(self):
        pair_it = onir.datasets.pair_iter(
            self,
            {'query_id', 'query_text', 'doc_id', 'doc_text'},
            pos_source=self.config['train_pos_source'],
            neg_source=self.config['train_neg_source'],
            sampling=self.config['train_sampling'],
            pos_minrel=self.config['train_pos_minrel'],
            unjudged_rel=self.config['train_unjudged_rel'],
            num_neg=1, # this implementation only supports 1 pos and 1 neg (for now)
            random=self.rng,
            inf=True)
        for records in pair_it:
            assert records['query_id'][0] == records['query_id'][1]
            yield TrainPair(
                records['query_id'][0],
                records['query_text'][0],
                records['doc_id'][0],
                records['doc_text'][0],
                records['doc_id'][1],
                records['doc_text'][1])
        return self.tr_pairs

    # mock dataset
    def qrels(self, fmt):
        assert fmt == 'df'
        return self.tr_qrels

    # mock dataset
    def run(self, fmt):
        assert fmt == 'df'
        return self.tr_run

    # mock dataset
    def build_record(self, fields, **args):
        query_text = self.topic_lookup[args['query_id']]
        doc_text = self.doc_lookup[args['doc_id']]
        return {
            'query_id': args['query_id'],
            'query_text': query_text,
            'doc_id': args['doc_id'],
            'doc_text': doc_text,
        }







class OpenNIRPyterrierReRanker(pyterrier.transformer.TransformerBase):
    """
    Provides an interface for OpenNIR-style ranking models to act
    as re-rankers in pyterrier pipelines.
    """
    def __init__(self,
        ranker: Union[str, onir.rankers.Ranker],
        vocab: Union[str, onir.vocab.Vocab] = None,
        ranker_config: dict = None,
        vocab_config: dict = None,
        weights: str = None,
        config: dict = None):
        if vocab is not None:
            if isinstance(vocab, str):
                vocab = _inject(
                    onir.vocab.registry.registered[vocab],
                    {'config': vocab_config})
            else:
                assert ranker_config is None, "cannot provide vocab_config with instantiated vocab"

        if isinstance(ranker, str):
            ranker = _inject(
                onir.rankers.registry.registered[ranker],
                {'config': ranker_config, 'vocab': vocab})
        else:
            assert ranker_config is None, "cannot provide ranker_config with instantiated ranker"

        if weights:
            ranker.load(weights)
        self.vocab = vocab
        self.ranker = ranker
        # defualt config
        self.config = {
            'batch_size': 4,
            'gpu': True,
            'gpu_determ': True,
            'random_seed': 42,
            'pre_validate': True,
            'max_train_it': 500,
            'patience': 20,
            'va_metrics': ['map', 'ndcg', 'P_10'],
            'train_count': 1024,
            'learning_rate': 1e-3,
            'train_pos_source': 'intersect', # one of ['qrels', 'intersect']
            'train_neg_source': 'run', # one of ['run', 'qrels', 'union']
            'train_sampling': 'query', # one of ['query', 'qrel']
            'train_pos_minrel': 1,
            'train_unjudged_rel': 0,
        }

        # apply overrides
        if config is not None:
            self.config.update(config)

    def transform(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        scores = []
        device = onir.util.device(self.config, _logger)
        with torch.no_grad():
            ranker = self.ranker.to(device).eval()
            batches = self._iter_batches(dataframe, device)
            for count, batch in _logger.pbar(batches, desc='batches', tqdm=pyterrier.tqdm, total=math.ceil(len(dataframe) / self.config['batch_size'])):
                batch_scores = ranker(**batch)
                while len(batch_scores.shape) > 1:
                    batch_scores = batch_scores[..., 0]
                assert batch_scores.shape[0] == count
                scores.append(batch_scores.cpu().numpy())
        dataframe['score'] = np.concatenate(scores)
        return dataframe

    def fit(self,
            tr_topics=None,
            tr_qrels=None,
            tr_run=None,
            tr_pairs=None,
            va_run=None,
            va_qrels=None) -> pd.DataFrame:
        if tr_pairs is not None:
            assert tr_topics is None and tr_qrels is None, "tr_topics and tr_qrels shouldn't be provided if tr_pairs is given"
            trainer = PtPairTrainer(self, tr_pairs, self.config)
        elif tr_topics is not None and tr_qrels is not None:
            trainer = PtTopicQrelsTrainer(self, tr_topics, tr_qrels, tr_run, self.config)
        else:
            raise RuntimeError('Must provide either tr_pairs or (tr_topics and tr_qrels)')
        if va_run is None and va_qrels is None:
            validator = None # No validation -- just do a single stage of ranking
        elif va_run is not None and va_qrels is not None:
            validator = PtValidator(self, va_run, va_qrels, self.config)
        else:
            raise RuntimeError('Must provide either both va_run and va_qrels or neither')

        best_valid, best_it = 0, -1
        if validator and self.config['pre_validate']:
            valid_score, valid_scores = validator.validate()
            _logger.info(f'pre-validation: {valid_score:.4f}')
            if valid_score > best_valid:
                best_valid = valid_score

        train_it = 0
        output = []
        with tempfile.TemporaryDirectory() as tmpdir:
            if validator is not None:
                self.ranker.save(f'{tmpdir}/best-model.pt')
            while train_it < self.config['max_train_it']:
                train_loss = trainer.train()
                _logger.info(f'training   it={train_it} loss={train_loss:.4f}')
                if validator:
                    valid_score, valid_scores = validator.validate()
                    if valid_score > best_valid:
                        best_valid = valid_score
                        best_it = train_it
                        mark = ' <--'
                        self.ranker.save(f'{tmpdir}/best-model.pt')
                    else:
                        mark = ''
                    valid_out = ' '.join([f'{m}={valid_scores[m]:.4f}' for m in self.config['va_metrics']])
                    _logger.info(f'validation it={train_it} {valid_out}{mark}')
                    output.append({'it': train_it, 'loss': train_loss, **valid_scores})
                else:
                    # no validation
                    output.append({'it': train_it, 'loss': train_loss})
                    break
                if best_it - train_it >= self.config['patience']:
                    _logger.info(f'early stopping; model reverting back to it={best_it}')
                    self.ranker.load(f'{tmpdir}/best-model.pt')
                    break
        return pd.DataFrame(output)

    def _iter_batches(self, dataframe, device):
        batch_size = self.config['batch_size']
        input_spec = self.ranker.input_spec()
        fields = input_spec['fields']
        batch = {f: [] for f in fields}
        batch_count = 0
        last_qid = None
        for rec in dataframe.itertuples():
            if rec.qid != last_qid:
                query_text = self.vocab.tokenize(rec.query)
                query_tok = [self.vocab.tok2id(t) for t in query_text]
                last_qid = rec.qid
            doc_text = self.vocab.tokenize(rec.text)
            doc_tok = [self.vocab.tok2id(t) for t in doc_text]
            for f in fields:
                if f == 'doc_id':
                    batch[f].append(rec.docno)
                elif f == 'query_id':
                    batch[f].append(rec.qid)
                elif f == 'query_rawtext':
                    batch[f].append(rec.query)
                elif f == 'doc_rawtext':
                    batch[f].append(rec.text)
                elif f == 'query_text':
                    batch[f].append(query_text)
                elif f == 'doc_text':
                    batch[f].append(doc_text)
                elif f == 'query_tok':
                    batch[f].append(query_tok)
                elif f == 'doc_tok':
                    batch[f].append(doc_tok)
                elif f == 'query_len':
                    batch[f].append(len(query_tok))
                elif f == 'doc_len':
                    batch[f].append(len(doc_tok))
                else:
                    raise ValueError(f'unsupported field {f}')
            batch_count += 1
            if batch_count == self.config['batch_size']:
                yield batch_count, onir.spec.apply_spec_batch(batch, input_spec, device)
                batch = {f: [] for f in fields}
                batch_count = 0
        if batch_count > 0:
            yield batch_count, onir.spec.apply_spec_batch(batch, input_spec, device)

    def to_checkpoint(self, checkpoint_file: str):
        with tarfile.open(checkpoint_file, 'w:gz') as tarf:
            record = tarfile.TarInfo('ranker.json')
            data = dict(self.ranker.config)
            data[''] = self.ranker.name
            data = json.dumps(data).encode()
            record.size = len(data)
            tarf.addfile(record, io.BytesIO(data))

            record = tarfile.TarInfo('vocab.json')
            data = dict(self.vocab.config)
            data[''] = self.vocab.name
            data = json.dumps(data).encode()
            record.size = len(data)
            tarf.addfile(record, io.BytesIO(data))

            record = tarfile.TarInfo('config.json')
            data = json.dumps(self.config).encode()
            record.size = len(data)
            tarf.addfile(record, io.BytesIO(data))

            record = tarfile.TarInfo('weights.p')
            data = io.BytesIO()
            self.ranker.save(data)
            record.size = data.tell()
            data.seek(0)
            tarf.addfile(record, data)

    @staticmethod
    def from_checkpoint(checkpoint_file: str, config: dict = None):
        # A checkpoint file is a .tar.gz file that contains the following:
        #  - vocab.json  (config file including key '' that indicates the name)
        #  - ranker.json  (config file including key '' that indicates the name)
        #  - config.json  (optional pyterrier transformer config file)
        #  - weights.p   (pytorch weights for the trained model)
        # TODO: make a utility that automatically builds such a file from ONIR runs
        ranker_config, vocab_config, weights = None, None, None
        with tarfile.open(checkpoint_file, 'r') as tarf:
            # TODO: support URLS
            for record in tarf:
                if record.name == 'vocab.json':
                    vocab_config = json.load(tarf.extractfile(record))
                elif record.name == 'ranker.json':
                    ranker_config = json.load(tarf.extractfile(record))
                elif record.name == 'config.json':
                    cfg = json.load(tarf.extractfile(record))
                    if config is None:
                        config = cfg
                    else:
                        cfg.update(config)
                        config = cfg
                elif record.name == 'weights.p':
                    weights = tarf.extractfile(record)
                else:
                    _logger.warn(f'unexpected file in checkpoint: {record.name}')
            assert ranker_config is not None, "ranker.json missing"
            assert vocab_config is not None, "vocab.json missing"
            assert weights is not None, "weights.p missing"
            if config is None: # missing config is OK
                config = {}
            ranker_name = ranker_config['']
            vocab_name = vocab_config['']
            del ranker_config['']
            del vocab_config['']
            return OpenNIRPyterrierReRanker(ranker_name, vocab_name, ranker_config, vocab_config, weights, config=config)


def _inject(cls, context={}):
    spec = inspect.getargspec(cls.__init__)
    args = []
    for arg in spec.args:
        if arg == 'self':
            continue
        elif arg == 'config':
            config = onir.config.apply_config(None, context.get('config') or {}, cls)
            args.append(config)
        elif arg in context:
            args.append(context[arg])
        elif arg == 'random':
            args.append(np.random.RandomState(42)) # TODO: make configurable
        elif arg == 'logger':
            args.append(onir.log.Logger(cls.__name__))
        else:
            raise ValueError(f'cannot match argument `{arg}` for `{cls}`')
    return cls(*args)


# an easier-to-remember alias: onir.pt.reranker
reranker = OpenNIRPyterrierReRanker


# column definitions for training with tr_pairs
TrainPair = namedtuple('TrainPair', ['qid', 'query', 'docno_pos', 'text_pos', 'docno_neg', 'text_neg'])


if __name__ == '__main__':
    # rr = reranker('knrm', 'wordvec')
    # rr = reranker.from_checkpoint('test.tar.gz')
    rr = reranker.from_checkpoint('epic.msmarco.tar.gz', {'learning_rate': 1e-5})
    df = pd.DataFrame([
        {"qid": '1', 'docno': '1', 'query': 'the quick brown fox', 'text': 'the quick brown fox jumps over the lazy dog'},
        {"qid": '1', 'docno': '2', 'query': 'the quick brown fox', 'text': 'the brown fox jumps over the lazy dog'},
        {"qid": '1', 'docno': '3', 'query': 'the quick brown fox', 'text': 'the quick fox jumps over the lazy dog'},
        {"qid": '1', 'docno': '4', 'query': 'the quick brown fox', 'text': 'the quick brown jumps over the lazy dog'},
        {"qid": '1', 'docno': '5', 'query': 'the quick brown fox', 'text': 'quick brown fox jumps over the lazy dog'},
        {"qid": '1', 'docno': '6', 'query': 'the quick brown fox', 'text': 'the quick brown fox'},
        {"qid": '1', 'docno': '7', 'query': 'the quick brown fox', 'text': 'over lazy the dog jumps the fox quick brown'},
        {"qid": '1', 'docno': '8', 'query': 'the quick brown fox', 'text': 'the quick brown fox jumps over lazy dog'},
        {"qid": '1', 'docno': '9', 'query': 'the quick brown fox', 'text': 'the quick brown fox jumps'},
        {"qid": '1', 'docno': '1', 'query': 'the quick brown fox', 'text': 'the quick brown fox jumps over the lazy dog'},
    ])
    result = rr.transform(df)
    # rr.to_checkpoint('test.tar.gz')
    def tr_pairs():
        import ir_datasets
        ds = ir_datasets.load('msmarco-passage/train')
        queries = {q.query_id: q for q in ds.queries_iter()}
        docstore = ds.docs_store()
        for scoreddoc in ds.docpairs_iter():
            yield TrainPair(
                scoreddoc.query_id,
                queries[scoreddoc.query_id].text,
                scoreddoc.doc_id_a,
                docstore.get(scoreddoc.doc_id_a).text,
                scoreddoc.doc_id_b,
                docstore.get(scoreddoc.doc_id_b).text)
    fit_result = rr.fit(tr_pairs=tr_pairs())
    import pdb; pdb.set_trace()
    pass
