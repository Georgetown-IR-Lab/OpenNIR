import os
import tempfile
import zipfile
import numpy as np
from gensim.models.keyedvectors import KeyedVectors
from onir import util
from onir.interfaces import plaintext



def nil_handler(_):
    terms = ['']
    weights = np.zeros((1, 1))
    return terms, weights


def zip_handler(url, ext='', expected_md5=None):
    def wrapped(logger):
        with tempfile.TemporaryDirectory() as p:
            path = os.path.join(p, 'download')
            with logger.duration(f'downloading {url}'):
                util.download(url, path, expected_md5=expected_md5)
            with logger.duration('extracting vecs'):
                with zipfile.ZipFile(path) as z:
                    zip_file_name = url.split('/')[-1][:-4] + ext
                    out_file_name = os.path.join(p, zip_file_name)
                    z.extract(zip_file_name, p)
            with logger.duration(f'loading vecs into memory'):
                terms = []
                weights = []
                with open(out_file_name, 'rt') as f:
                    for line in f:
                        cols = line.split()
                        if len(cols) == 2:
                            continue # First line includes # of terms and dim, we can ignore
                        if weights and len(weights[0]) != len(cols) - 1:
                            logger.warn(f'problem parsing line, skipping {line[:20]}...')
                        else:
                            terms.append(cols[0])
                            weights.append([float(c) for c in cols[1:]])
                weights = np.array(weights)
                return terms, weights
    return wrapped


def convknrm_handler(base_url):
    def wrapped(logger, get_kernels=False):
        with tempfile.TemporaryDirectory() as p:
            if not get_kernels:
                vocab_path = os.path.join(p, 'vocab')
                with logger.duration(f'downloading {base_url}vocab'):
                    util.download(base_url + 'vocab', vocab_path)
                with logger.duration(f'reading vocab'):
                    v = {}
                    for term, idx in plaintext.read_tsv(vocab_path):
                        v[int(idx)] = term
                    terms = [None] * (max(v.keys()) + 1)
                    for idx, term in v.items():
                        terms[idx] = term
                embedding_path = os.path.join(p, 'embedding')
                with logger.duration(f'downloading {base_url}embedding'):
                    util.download(base_url + 'embedding', embedding_path)
                with logger.duration(f'reading embedding'):
                    weights = None
                    for values in plaintext.read_sv(embedding_path, sep=' '):
                        if len(values) == 2:
                            weights = np.ndarray((int(values[0]), int(values[1])))
                        else:
                            idx, values = values[0], values[1:]
                            weights[int(idx)] = [float(v) for v in values]
                return terms, weights
            else: # get_kernels
                w, b = [], []
                for f in range(1, 4):
                    url = f'{base_url}filter{f}'
                    path = os.path.join(p, f'filter{f}')
                    with logger.duration(f'downloading {url}'):
                        util.download(url, path)
                    with logger.duration(f'reading filter{f}'):
                        weights, biases = None, None
                        for i, values in enumerate(plaintext.read_sv(path, sep=' ')):
                            if i == 0:
                                weights = np.ndarray((int(values[0]) * int(values[1]), int(values[2])))
                            elif i == 1:
                                biases = np.array([float(v) for v in values])
                            else:
                                weights[:, i-2] = [float(v) for v in values if v]
                    weights = weights.reshape(f, -1, weights.shape[1])
                    weights = np.transpose(weights, (2, 1, 0))
                    w.append(weights)
                    b.append(biases)
                return w, b
    return wrapped


def gensim_w2v_handler(url):
    def wrapped(logger):
        with tempfile.TemporaryDirectory() as p:
            vocab_path = os.path.join(p, 'vocab')
            with logger.duration(f'downloading {url}'):
                util.download(url, vocab_path)
            with logger.duration(f'loading binary {vocab_path}'):
                vectors = KeyedVectors.load_word2vec_format(vocab_path, binary=True)
            vocab_path += '.txt'
            with logger.duration(f'saving text {vocab_path}'):
                vectors.save_word2vec_format(vocab_path)
            with logger.duration(f'reading embedding'):
                weights = None
                terms = []
                for i, values in enumerate(plaintext.read_sv(vocab_path, sep=' ')):
                    if i == 0:
                        weights = np.ndarray((int(values[0]), int(values[1])))
                    else:
                        term, values = values[0], values[1:]
                        terms.append(term)
                        weights[i-1] = [float(v) for v in values]
            return terms, np.array(weights)
    return wrapped
