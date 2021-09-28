import sys
import os
import hashlib
import math
import io
import tempfile
import inspect
import torch
import json
import tarfile
import transformers
from pathlib import Path
from collections import namedtuple, Counter, OrderedDict
from typing import Union, Dict, Tuple
import pandas as pd
import numpy as np

# should come before importing onir
os.environ['ONIR_IGNORE_ARGV'] = 'true' # don't process command line arguments (they come from jupyter)
os.environ['ONIR_PBAR_COLS'] = '' # no ncols for tqdm

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
    def __init__(self, pt_transformer, tr_run, tr_qrels, config):
        super().__init__(pt_transformer, config)
        self.topic_lookup = {t.qid: t.query for t in tr_run.itertuples()}
        self.doc_lookup = {t.docno: getattr(t, self.pt_transformer.text_field) for t in tr_run.itertuples()}
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



class OpenNIRPyterrierReRanker(pyterrier.transformer.EstimatorBase):
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
        config: dict = None,
        text_field: str = 'text'):
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

        self.text_field = text_field

        # apply overrides
        if config is not None:
            self.config.update(config)

    def transform(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Scores query-document pairs using this model.
        """
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
            tr_run: pd.DataFrame = None,
            tr_qrels: pd.DataFrame = None,
            va_run: pd.DataFrame = None,
            va_qrels: pd.DataFrame = None,
            *,
            tr_pairs: pd.DataFrame = None) -> pd.DataFrame:
        """
        Trains this model on either pairs generated from tr_run and tr_qrels or explicitly provided with tr_pairs.
        """
        if tr_pairs is not None:
            assert tr_run is None and tr_qrels is None, "tr_run and tr_qrels shouldn't be provided if tr_pairs is given"
            trainer = PtPairTrainer(self, tr_pairs, self.config)
        elif tr_run is not None and tr_qrels is not None:
            trainer = PtTopicQrelsTrainer(self, tr_run, tr_qrels, self.config)
        else:
            raise RuntimeError('Must provide either tr_pairs or (tr_run and tr_qrels)')
        if va_run is None and va_qrels is None:
            validator = None # No validation -- just do a single stage of training
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
                if train_it - best_it >= self.config['patience']:
                    _logger.info(f'early stopping; model reverting back to it={best_it}')
                    self.ranker.load(f'{tmpdir}/best-model.pt')
                    break
                train_it += 1
        return pd.DataFrame(output)

    def explain_diffir(self, dataframe: pd.DataFrame) -> Dict[str, Dict[str, Dict[str, Tuple[int, int, float]]]]:
        """
        Produces a diffir weights object for each query/doc in the given dataframe.
        When saved as JSON, this object is suitable for highlighting functionality in
        the diffir tool <https://github.com/capreolus-ir/diffir>.

        Not supported by all models.
        """
        assert hasattr(self.ranker, 'explain_diffir'), f"model {self.ranker} does not support explain_diffir"
        ranker = self.ranker.to(onir.util.device(self.config, _logger)).eval()

        result = {}
        for rec in _logger.pbar(dataframe.itertuples(), total=len(dataframe)):
            weights = ranker.explain_diffir(rec.query, getattr(rec, self.text_field))
            result.setdefault(rec.qid, {})[rec.docno] = {self.text_field: weights}
        return result

    def _iter_batches(self, dataframe, device, fields=None, skip_empty_docs=False):
        batch_size = self.config['batch_size']
        input_spec = self.ranker.input_spec()
        if fields is None:
            fields = input_spec['fields']
        batch = {f: [] for f in fields}
        batch_count = 0
        last_qid = None
        # support either df or iter[dict]s
        if isinstance(dataframe, pd.DataFrame):
            dataframe = [rec._asdict() for rec in dataframe.itertuples()]
        for rec in dataframe:
            if 'qid' in rec:
                if rec['qid'] != last_qid:
                    if self.vocab:
                        query_text = self.vocab.tokenize(rec['query'])
                        query_tok = [self.vocab.tok2id(t) for t in query_text]
                    else:
                        query_text, query_tok = None, None
                    last_qid = rec['qid']
            else:
                query_text, query_tok = None, None
            if self.text_field in rec:
                if self.vocab:
                    doc_rawtext = rec[self.text_field]
                    doc_text = self.vocab.tokenize(doc_rawtext)
                    doc_tok = [self.vocab.tok2id(t) for t in doc_text]
                else:
                    doc_text, doc_tok = None, None
                if skip_empty_docs and len(doc_tok) == 0:
                    continue
            else:
                doc_text, doc_tok = None, None
            for f in fields:
                if f == 'doc_id':
                    batch[f].append(rec['docno'])
                elif f == 'query_id':
                    batch[f].append(rec['qid'])
                elif f == 'query_rawtext':
                    batch[f].append(rec['query'])
                elif f == 'doc_rawtext':
                    batch[f].append(doc_rawtext)
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
        """
        Saves this model as a checkpoint that can be loaded with .from_checkpoint()
        """
        with tarfile.open(checkpoint_file, 'w:gz') as tarf:
            record = tarfile.TarInfo('ranker.json')
            data = dict(self.ranker.config)
            data[''] = self.ranker.name
            data = json.dumps(data).encode()
            record.size = len(data)
            tarf.addfile(record, io.BytesIO(data))

            if self.vocab:
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

    @classmethod
    def from_checkpoint(cls, checkpoint_file: str, config: dict = None, expected_md5: str = None, text_field: str = 'text', **kwargs):
        """
        Loads a reranker object from a checkpoint file. checkpoint_file can either be a file path
        on the filesystem, or a URL to download (optionally verified with expected_md5).
        """
        # A checkpoint file is a .tar.gz file that contains the following:
        #  - vocab.json  (config file including key '' that indicates the name)
        #  - ranker.json  (config file including key '' that indicates the name)
        #  - config.json  (optional pyterrier transformer config file)
        #  - weights.p   (pytorch weights for the trained model)
        # TODO: make a utility that automatically builds such a file from ONIR runs
        ranker_config, vocab_config, weights = None, None, None
        if checkpoint_file.startswith('http://') or checkpoint_file.startswith('https://'):
            # Download
            output_path = os.path.join(onir.util.get_working(), 'model_checkpoints')
            os.makedirs(output_path, exist_ok=True)
            output_path = os.path.join(output_path, hashlib.md5(checkpoint_file.encode()).hexdigest())
            if not os.path.exists(output_path):
                onir.util.download(checkpoint_file, output_path, expected_md5=expected_md5)
            else:
                _logger.info(f'using cached checkpoint: {output_path}')
            checkpoint_file = output_path
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
            assert weights is not None, "weights.p missing"
            if config is None: # missing config is OK
                config = {}
            ranker_name = ranker_config['']
            vocab_name = vocab_config[''] if vocab_config else None
            del ranker_config['']
            if vocab_config:
                del vocab_config['']
            return cls(ranker_name, vocab_name, ranker_config, vocab_config, weights, config=config, text_field=text_field, **kwargs)

    def __repr__(self):
        if self.vocab:
            return f'onir({self.ranker.name},{self.vocab.name})'
        return f'onir({self.ranker.name})'


    def __str__(self):
        return repr(self)


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



class IndexedEpic(OpenNIRPyterrierReRanker):
    """
    Provides an interface for a building a direct index for an EPIC model.
    """
    def __init__(self, *args, index_path=None, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.ranker.name == 'epic'
        self.index_path = Path(index_path)
        if 'epic_prune' not in self.config:
            self.config['epic_prune'] = 1000 # default pruning value

    def fit(self, docs_iter, fields=('text',)):
        return self.index(docs_iter, fields)

    def index(self, docs_iter, fields=('text',), replace=False):
        assert self.index_path is not None, "you must supply index_path to run .index()"
        if not replace:
            assert not (self.index_path / '_built').exists(), "index already built (use replace=True to replace)"
        else:
            if (self.index_path / '_built').exists():
                (self.index_path / '_built').unlink()
        self.index_path.mkdir(parents=True, exist_ok=True)
        PRUNE = self.config['epic_prune']
        LEX_SIZE = self.vocab.lexicon_size()
        with (self.index_path / 'data').open('wb') as out_data, \
             (self.index_path / 'dids').open('wt') as out_dids:
            for did, vec in self._iter_docvecs(docs_iter, fields):
                vec = vec.half().cpu().numpy()
                idxs = np.argpartition(vec, LEX_SIZE - PRUNE)[-PRUNE:].astype(np.int16)
                idxs.sort()
                vals = vec[idxs]
                out_data.write(idxs.tobytes())
                out_data.write(vals.tobytes())
                out_dids.write(f'{did}\n')
        (self.index_path / '_built').touch()
        return self

    def _iter_docvecs(self, docs_iter, fields):
        device = onir.util.device(self.config, _logger)
        def transform_doc(doc):
            doc['text'] = ' '.join(doc[f] for f in fields)
            return doc
        docs_iter = map(transform_doc, docs_iter)
        with torch.no_grad():
            ranker = self.ranker.to(device).eval()
            doc_fields = {f for f in self.ranker.input_spec()['fields'] if f.startswith('doc_')} | {'doc_id'}
            batches = self._iter_batches(docs_iter, device, doc_fields, skip_empty_docs=True)
            for count, batch in batches:
                doc_vectors = ranker.doc_vectors(dense=True, **batch)
                yield from zip(batch['doc_id'], doc_vectors)

    def reranker(self) -> 'EpicIndexedReRanker':
        """
        Returns an EpicIndexedReRanker for this (built) index, which can be used to score documents.
        """
        assert (self.index_path / '_built').exists(), "index does not exist; use .index(docs_iter) to build index"
        return EpicIndexedReRanker(ranker=self.ranker, vocab=self.vocab, config=self.config, index_path=self.index_path)

    def doc_vectors(self, doc_ids: Union[str, list]) -> Dict[str, Dict[str, float]]:
        """
        Looks up pre-computed EPIC document vectors for the provided document ID(s).
        """
        doc_ids = [doc_ids] if isinstance(doc_ids, str) else list(doc_ids)

        result = {}

        for did, (tids, vals) in zip(doc_ids, self.reranker()._iter_docs_from_index(doc_ids)):
            result = Counter()
            for tid, val in zip(tids, vals):
                result[self.vocab.id2tok(tid)] = val
            result[did] = OrderedDict(result.most_common())
        return result


_msg_text_shown = False
_msg_latency_shown = False

class EpicIndexedReRanker(OpenNIRPyterrierReRanker):
    """
    Provides an interface for using a direct index from an EPIC model to score documents.
    """
    def __init__(self, *args, index_path=None, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.ranker.name == 'epic'
        self.index_path = Path(index_path)

    def fit(self, *args, **kwargs):
        raise NotImplementedError('fit not supported for indexed EPIC reranker')


    def transform(self, dataframe: pd.DataFrame = None) -> pd.DataFrame:
        global _msg_text_shown
        global _msg_latency_shown
        # TODO: only show these messages once?
        if not _msg_text_shown:
            if 'text' in dataframe:
                _logger.info("text field spotted in dataframe; you don't need this when using epic from indexed docs. (This message is only shown once.)")
                _msg_text_shown = True
        if not _msg_latency_shown:
            _logger.info("This EPIC transformer shouldn't be used to calculate query latency. It computes "
                         "query vectors batches (rather than individually), and doesn't do this work in parallel "
                         "with first-stage retrieval. For thise operations, use the epic pipeline in OpenNIR. "
                         "(This message is only shown once.)")
            _msg_latency_shown = True

        def query_data_iter(df):
            prev_qid = None
            for rec in df.itertuples():
                if rec.qid != prev_qid:
                    yield {'qid': rec.qid, 'query': rec.query}
                    prev_qid = rec.qid
        qvec_iter = self._iter_qvecs(query_data_iter(dataframe))
        this_qid, this_qvec = next(qvec_iter)

        LEXICON_SIZE = self.vocab.lexicon_size()

        doc_iter = self._iter_docs_from_index(d.docno for d in dataframe.itertuples())
        
        scores = []
        with open(os.path.join(self.index_path, 'data'), 'rb') as fdata:
            # TODO: optmize order that the index is traversed
            for rec, (tids, vals) in _logger.pbar(zip(dataframe.itertuples(), doc_iter), desc='records', tqdm=pyterrier.tqdm, total=len(dataframe)):
                while rec.qid != this_qid: # should only ever be 0 or 1 iteration
                    this_qid, this_qvec = next(qvec_iter)
                    this_qvec = this_qvec.cpu()
                tids = torch.from_numpy(tids)
                vals = torch.from_numpy(vals)
                dvec = torch.sparse.FloatTensor(torch.stack([torch.zeros_like(tids), tids]).long(), vals.float(), torch.Size([1, LEXICON_SIZE]))
                score = (this_qvec.cpu() * dvec[0]).values().sum().item()
                scores.append(score)
        dataframe['score'] = scores
        return dataframe

    def _iter_qvecs(self, query_iter):
        device = onir.util.device(self.config, _logger)
        with torch.no_grad():
            ranker = self.ranker.to(device).eval()
            query_fields = {f for f in self.ranker.input_spec()['fields'] if f.startswith('query_')} | {'query_id'}
            batches = self._iter_batches(query_iter, device, query_fields)
            for count, batch in batches:
                query_vectors = ranker.query_vectors(dense=True, **batch) # bad arg name, not actually dense, just a tensor (rather than dict)
                yield from zip(batch['query_id'], query_vectors)

    def _iter_docs_from_index(self, did_iter):
        DTYPE_S = 2 # all values are 2-byte (int16 and f2)
        PRUNE = self.config['epic_prune']
        REC_SIZE = PRUNE * DTYPE_S * 2 # pos + vals

        # TODO: cache did2idx
        did2idx = {did.strip(): i for i, did in enumerate((self.index_path / 'dids').open('rt'))}

        with (self.index_path / 'data').open('rb') as fdata:
            for did in did_iter:
                assert did in did2idx, f"docno {did} not found in index"
                idx = did2idx[did]
                fdata.seek(REC_SIZE * idx)
                data = fdata.read(REC_SIZE)
                tids = np.frombuffer(data[:REC_SIZE//2], dtype=np.int16)
                vals = np.frombuffer(data[REC_SIZE//2:], dtype='f2')
                yield tids, vals


# easier-to-remember aliases: onir.pt.reranker and onir.pt.indexed_epic
reranker = OpenNIRPyterrierReRanker
indexed_epic = IndexedEpic


# column definitions for training with tr_pairs
TrainPair = namedtuple('TrainPair', ['qid', 'query', 'docno_pos', 'text_pos', 'docno_neg', 'text_neg'])
