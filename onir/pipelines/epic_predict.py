import os
import mmap
from multiprocessing.pool import ThreadPool
import numpy as np
import torch
import onir
from onir import pipelines, util, spec
from onir.interfaces import plaintext, trec

logger = onir.log.easy()


@pipelines.register('epic_predict')
class EpicPredictionPipeline(pipelines.BasePipeline):
    @staticmethod
    def default_config():
        return {
            'gpu': True,
            'gpu_determ': True,
            'file_path': '',
            'epoch': '',
            'val_metric': '',
            'queries': '',
            'prune': 1000,
            'dvec_file': '',
            'dvec_inmem': True,
            'rerank': True,
            'rerank_threshold': -1,
            'batch_size': 16,
            'mode': util.config.Choices(['predict', 'time']),
            'warmup': 1000,
            'count': 1000,
            'output': ''
        }

    def __init__(self, config, ranker, vocab, train_ds, trainer, valid_pred, test_ds):
        super().__init__(config, logger)
        self.ranker = ranker
        self.vocab = vocab
        self.trainer = trainer
        self.train_ds = train_ds
        self.valid_pred = valid_pred
        self.test_ds = test_ds
        self.timer = None

    def run(self):
        if self.config['queries']:
            logger.debug('loading queries from {queries}'.format(**self.config))
            query_iter = plaintext.read_tsv(self.config['queries'])
        else:
            logger.debug('loading queries test_ds')
            query_iter = self.test_ds.all_queries_raw()

        if self.config['rerank']:
            if not self.config['dvec_file']:
                raise ValueError('must provide dvec_file')
            self._load_ranker_weights(self.ranker, self.vocab, self.trainer, self.valid_pred, self.train_ds)
            self.ranker.eval()
            input_spec = self.ranker.input_spec()
            fields = {f for f in input_spec['fields'] if f.startswith('query_')}
            device = util.device(self.config, logger)
            vocab_size = self.vocab.lexicon_size()
            num_docs = self.test_ds.num_docs()
            dvec_cache = EpicCacheReader(self.config['dvec_file'], self.config['prune'], num_docs, vocab_size, self.config['dvec_inmem'], self.config['gpu'])
        else:
            pass # only do initial retrieval

        self.timer = util.DurationTimer(gpu_sync=self.config['gpu'])
        with torch.no_grad():
            if self.config['mode'] == 'time':
                self.time(query_iter, dvec_cache, fields, input_spec, device)
            if self.config['mode'] == 'predict':
                self.predict(query_iter, dvec_cache, fields, input_spec, device)

    def time(self, query_iter, dvec_cache, fields, input_spec, device):
        index = self.test_ds._get_index_for_batchsearch()
        timer_count = 0
        with ThreadPool() as pool:
            for i, (qid, query) in enumerate(query_iter):
                timer_count += 1
                with self.timer.time('total'):
                    if self.config['rerank']:
                        record = self.test_ds.build_record(fields, query_rawtext=query)
                        res = pool.apply_async(lambda: self.query_vecs([record], input_spec, fields, device))
                        run = self.initial_retrieval(index, query)
                        dids = [y[0] for y in sorted(run.items(), key=lambda x: (x[1], x[0]), reverse=True)][:self.config['rerank_threshold']]
                        dvecs = self.load_dvecs(dids, record['query_tok'], dvec_cache)
                        qvec = res.get()
                        run = self.rerank(dids, qvec, dvecs, dvec_cache)
                    else:
                        run = self.initial_retrieval(index, query)
                if timer_count % 100 == 0:
                    if i <= self.config['warmup']:
                        logger.debug(f'warmup {timer_count} {self.timer.scaled_str(timer_count)}')
                    else:
                        logger.debug(f'{timer_count} {self.timer.scaled_str(timer_count)}')
                if i + 1 == self.config['warmup']:
                    logger.debug(f'warmup finished {self.timer.scaled_str(timer_count)}')
                    self.timer = util.DurationTimer(gpu_sync=self.config['gpu'])
                    timer_count = 0
                if i >= self.config['warmup'] and timer_count == self.config['count']:
                    logger.info(f'finished {self.timer.scaled_str(timer_count)}')
                    break
            logger.debug(f'{timer_count} {self.timer.scaled_str(timer_count)}')

    def predict(self, query_iter, dvec_cache, fields, input_spec, device):
        if self.config['output']:
            output_file = open(self.config['output'], 'wt')
        result = {}
        query_iter = logger.pbar(query_iter, desc='queries')
        initial_runs = self.test_ds.run()
        batch = []
        for batch in self.batched(query_iter):
            qids, queries = zip(*batch)

            records = [self.test_ds.build_record(fields, query_rawtext=q) for q in queries]

            qvecs = self.query_vecs(records, input_spec, fields, device)
            for i, qid in enumerate(qids):
                run = initial_runs.get(qid, {})
                dids = [y[0] for y in sorted(run.items(), key=lambda x: (x[1], x[0]), reverse=True)][:self.config['rerank_threshold']]
                dvecs = self.load_dvecs(dids, record['query_tok'], dvec_cache)
                run = self.rerank(dids, qvecs[i].unsqueeze(0).coalesce(), dvecs, dvec_cache)
                if self.config['output']:
                    trec.write_run_dict(output_file, {qid: run})
                result[qid] = run

        metrics = onir.metrics.calc(self.test_ds.qrels(), result, {self.config['val_metric']})
        for metric, mean in onir.metrics.mean(metrics).items():
            logger.info(f'{metric}={mean:.4f}')

    def batched(self, query_iter):
        batch = []
        for qid, query in query_iter:
            batch.append((qid, query))
            if len(batch) == self.config['batch_size']:
                yield batch
                batch = []
        if batch:
            yield batch

    def initial_retrieval(self, index, query):
        with self.timer.time('initial_retrieval'):
            result = index.query_simplesearcher(query, self.test_ds.config['rankfn'], self.test_ds.config['ranktopk'])
        return result

    def query_vecs(self, records, input_spec, fields, device):
        with self.timer.time('query_vectors'):
            records = {f: [r[f] for r in records] for f in fields}
            batch = spec.apply_spec_batch(records, input_spec, device)
            result = self.ranker.query_vectors(dense=True, **batch)
            result = result.coalesce()
        return result

    def load_dvecs(self, dids, dims, dvec_cache):
        with self.timer.time('doc_vec_lookup'):
            dids = [int(did) for did in dids]
            dims = np.array(dims)
            dims.sort()
            dvecs = dvec_cache.lookup(dids, dims, full_vecs=True)
        return dvecs

    def rerank(self, dids, qvec, dvecs, dvec_cache):
        with self.timer.time('rerank'):
            result = {}
            doc_scores = self.ranker.similarity_inference(qvec, dvecs)
            for did, score in zip(dids, doc_scores.cpu()):
                result[str(did)] = score.item()
        return result


class EpicCacheReader:
    def __init__(self, file, prune, num_docs, vocab_size, in_memory, gpu):
        self.file = file = open(file, 'rb')
        mmp = mmap.mmap(file.fileno(), 0, flags=mmap.MAP_PRIVATE, prot=mmap.PROT_READ)
        if in_memory:
            # file will be read in in full sequentially
            os.posix_fadvise(file.fileno(), 0, 0, os.POSIX_FADV_SEQUENTIAL)
        else:
            # file will be read randomly as needed
            os.posix_fadvise(file.fileno(), 0, 0, os.POSIX_FADV_RANDOM)
        if prune != 0:
            S_DTYPE = 2
            if in_memory:
                mmp = np.empty((num_docs, prune), dtype='i2'), np.empty((num_docs, prune), dtype='f2')
                for did in logger.pbar(range(num_docs), desc='loading dvecs'):
                    try:
                        mmp[0][did] = np.frombuffer(file.read(prune*S_DTYPE), dtype='i2')
                        mmp[1][did] = np.frombuffer(file.read(prune*S_DTYPE), dtype='f2')
                    except ValueError:
                        pass
                file.close()
            self.lookup = self.dvec_lookup_pruned
        else:
            if in_memory:
                mmp = file.read()
                file.close()
            self.lookup = self.dvec_lookup_unpruned

        self.prune = prune
        self.num_docs = num_docs
        self.vocab_size = vocab_size
        self.mmp = mmp
        self.gpu = gpu
        self.in_memory = in_memory

    def close(self):
        self.file.close()

    def dvec_lookup_pruned(self, dids, dims, full_vecs=False):
        S_DTYPE = 2
        result_idxs = [[], []]
        result_vals = []

        if full_vecs:
            for i, did in enumerate(dids):
                if isinstance(self.mmp, tuple):
                    idxs = self.mmp[0][did]
                    vals = self.mmp[1][did]
                else:
                    pos = self.prune * did * S_DTYPE * 2
                    idxs = np.frombuffer(self.mmp[pos:pos+self.prune*S_DTYPE], dtype=np.int16)
                    pos += self.prune * S_DTYPE
                    vals = np.frombuffer(self.mmp[pos:pos+self.prune*S_DTYPE], dtype='f2')
                result_idxs[0].extend([i] * idxs.shape[0])
                result_idxs[1].extend(idxs)
                result_vals.extend(vals)
        else:
            for i, did in enumerate(dids):
                if isinstance(self.mmp, tuple):
                    idxs = self.mmp[0][did]
                    vals = self.mmp[1][did]
                else:
                    pos = self.prune * did * S_DTYPE * 2
                    idxs = np.frombuffer(self.mmp[pos:pos+self.prune*S_DTYPE], dtype=np.int16)
                    pos += self.prune * S_DTYPE
                    vals = np.frombuffer(self.mmp[pos:pos+self.prune*S_DTYPE], dtype='f2')
                match_points = np.searchsorted(idxs, dims)
                for mp, dim in zip(match_points, dims):
                    if mp < idxs.shape[0] and idxs[mp] == dim:
                        result_idxs[0].append(i)
                        result_idxs[1].append(dim)
                        result_vals.append(vals[mp])
        result_idxs = torch.tensor(result_idxs).long()
        result_vals = torch.tensor(result_vals).float()
        result_size = torch.Size([len(dids), self.vocab_size])
        result = torch.sparse.FloatTensor(result_idxs, result_vals, result_size)
        if self.gpu:
            result = result.cuda()
        return result

    def dvec_lookup_unpruned(self, dids, dims):
        S_DTYPE = 2
        result_idxs = [[], []]
        result_vals = []

        for i, did in enumerate(dids):
            for dim in dims:
                pos = self.vocab_size * did * S_DTYPE + dim * S_DTYPE
                try:
                    val = np.frombuffer(self.mmp[pos:pos+S_DTYPE], dtype='f2')[0]
                except IndexError as e:
                    logger.warn(e)
                    val = 0.
                result_idxs[0].append(i)
                result_idxs[1].append(dim)
                result_vals.append(val)
        result_idxs = torch.tensor(result_idxs).long()
        result_vals = torch.tensor(result_vals).float()
        result_size = torch.Size([len(dids), self.vocab_size])
        result = torch.sparse.FloatTensor(result_idxs, result_vals, result_size)
        if self.gpu:
            result = result.cuda()
        return result
