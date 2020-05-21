import os
import torch
import numpy as np
import onir
from onir import pipelines, util, spec


@pipelines.register('epic_vectorize')
class EpicVectorize(pipelines.BasePipeline):
    @staticmethod
    def default_config():
        return {
            'file_path': '',
            'epoch': '',
            'val_metric': '',
            'output_vecs': '',
            'overwrite': False,
            'gpu': True,
            'gpu_determ': True,
            'prune': 1000,
            'batch_size': 8,
        }

    def __init__(self, config, ranker, vocab, train_ds, trainer, valid_pred, test_ds, logger):
        super().__init__(config, logger)
        self.ranker = ranker
        self.vocab = vocab
        self.trainer = trainer
        self.train_ds = train_ds
        self.valid_pred = valid_pred
        self.test_ds = test_ds

    def run(self):
        if self.config['output_vecs'] == '':
            raise ValueError('must provide pipeline.output_vecs setting (name of weights file)')
        vecs_file = self.config['output_vecs']
        if os.path.exists(vecs_file) and not self.config['overwrite']:
            raise ValueError(f'{vecs_file} already exists. Please rename pipeline.output_vecs or set pipeline.overwrite=True')
        self._load_ranker_weights(self.ranker, self.vocab, self.trainer, self.valid_pred, self.train_ds)

        device = util.device(self.config, self.logger)
        doc_iter = self._iter_doc_vectors(self.ranker, device)
        docno = 0
        PRUNE = self.config['prune']
        LEX_SIZE = self.vocab.lexicon_size()
        with util.finialized_file(vecs_file, 'wb') as outf:
            doc_iter = self.logger.pbar(doc_iter, desc='documents', total=self.test_ds.num_docs())
            for did, dvec in doc_iter:
                assert docno == int(did) # these must go in sequence! Only works for MS-MARCO dataset. TODO: include some DID-to-docno mapping
                if PRUNE == 0:
                    outf.write(dvec.tobytes())
                else:
                    idxs = np.argpartition(dvec, LEX_SIZE - PRUNE)[-PRUNE:].astype(np.int16)
                    idxs.sort()
                    vals = dvec[idxs]
                    outf.write(idxs.tobytes())
                    outf.write(vals.tobytes())
                docno += 1

    def _iter_doc_vectors(self, ranker, device):
        ranker.eval()
        with torch.no_grad():
            for batch in self._batch_iter_docs(ranker, device):
                try:
                    doc_vectors = ranker.doc_vectors(dense=True, **batch)
                    for did, vec in zip(batch['doc_id'], doc_vectors):
                        yield did, vec.half().cpu().numpy()
                except RuntimeError as e:
                    if 'out of memory' not in str(e):
                        raise
                    self.logger.warn(e)
                    self.logger.warn("falling back to batch_size=1 for this minibatch")
                    for i, did in enumerate(batch['doc_id']):
                        mini_batch = {k: v[i:i+1] for k, v in batch.items()}
                        try:
                            vec = ranker.doc_vectors(**mini_batch)[0]
                            yield did, vec.half().cpu().numpy()
                        except RuntimeError as e:
                            if 'out of memory' not in str(e):
                                raise
                            self.logger.warn(e)
                            self.logger.warn(f"failed with batch_size=1; using empty vector for {did}")
                            yield did, torch.zeros(self.vocab.lexicon_size()).numpy()

    def _batch_iter_docs(self, ranker, device):
        input_spec = ranker.input_spec()
        fields = {f for f in input_spec['fields'] if f.startswith('doc_')} | {'doc_id'}

        records = []
        for record in onir.datasets.doc_iter(self.test_ds, fields):
            records.append(record)
            if len(records) == self.config['batch_size']:
                records = {f: [r[f] for r in records] for f in fields}
                yield spec.apply_spec_batch(records, input_spec, device)
                records = []
