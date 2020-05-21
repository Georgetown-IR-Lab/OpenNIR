import os
import sys
import io
import shutil
import tempfile
import json
import math
import zlib
import random
import tarfile
import time
from glob import glob
from contextlib import contextmanager
import torch
import requests
from tqdm import tqdm
import numpy as np
from scipy.stats import gaussian_kde
from onir import config, log
from onir.util.concurrency import safe_thread_count, blocking_tee, background, CtxtThread, iter_noop, Lazy
from onir.util.download import download, download_stream, download_iter, download_if_needed, download_tmp
from onir.util.matheval import matheval



def get_working():
    base = config.args()['data_dir']
    base = os.path.expanduser(base)
    os.makedirs(base, exist_ok=True)
    return base


def path_dataset(dataset):
    path = os.path.join(get_working(), 'datasets', dataset.name)
    os.makedirs(path, exist_ok=True)
    return path


def path_dataset_records(dataset, record_method):
    path = os.path.join(get_working(), 'datasets', 'records', record_method, dataset.lexicon_path_segment())
    if not os.path.exists(path):
        os.makedirs(path)
        with open(os.path.join(path, 'config.json'), 'wt') as f:
            json.dump(dataset.config, f)
    return path


def path_modelspace():
    modelspace = config.args()['modelspace']
    path = os.path.join(get_working(), 'models', modelspace)
    os.makedirs(path, exist_ok=True)
    return path


def path_model(ranker):
    modelspace = config.args()['modelspace']
    path = os.path.join(get_working(), 'models', modelspace, ranker.path_segment())
    if not os.path.exists(path):
        os.makedirs(path)
        with open(os.path.join(path, 'config.json'), 'wt') as f:
            json.dump(ranker.config, f)
        with open(os.path.join(path, 'structure.txt'), 'wt') as f:
            f.write(repr(ranker))
    return path


def path_model_trainer(ranker, vocab, trainer, dataset):
    path = os.path.join(path_model(ranker), vocab.path_segment(), trainer.path_segment(), dataset.path_segment())
    if not os.path.exists(path):
        os.makedirs(path)
        with open(os.path.join(path_model(ranker), vocab.path_segment(), 'config.json'), 'wt') as f:
            json.dump(vocab.config, f)
        with open(os.path.join(path_model(ranker), vocab.path_segment(), trainer.path_segment(), 'config.json'), 'wt') as f:
            json.dump(trainer.config, f)
        with open(os.path.join(path_model(ranker), vocab.path_segment(), trainer.path_segment(), dataset.path_segment(), 'config.json'), 'wt') as f:
            json.dump(dataset.config, f)
    return path

def path_model_trainer_pred(ranker, vocab, trainer, dataset, valid_ds):
    path = os.path.join(path_model_trainer(ranker, vocab, trainer, dataset), valid_ds.path_segment())
    if not os.path.exists(path):
        os.makedirs(path)
        with open(os.path.join(path, 'config.json'), 'wt') as f:
            json.dump(valid_ds.config, f)
    return path

def path_log():
    path = os.path.join(get_working(), 'logs')
    os.makedirs(path, exist_ok=True)
    path = os.path.join(path, config.args()['runid'] + '.log')
    return path


def path_vocab(vocab):
    path = os.path.join(get_working(), 'vocab', vocab.name)
    os.makedirs(path, exist_ok=True)
    return path


def chunked_list(items, num):
    assert isinstance(items, list)
    for i in range(0, len(items), num):
        yield items[i:i+num]


def chunked(items, num):
    if isinstance(items, list):
        # more efficient version for lists
        yield from chunked_list(items, num)
    else:
        chunk = []
        for item in items:
            chunk.append(item)
            if len(chunk) == num:
                yield chunk
                chunk = []
        if chunk:
            yield chunk


def lens2mask(lens, size):
    mask = []
    for l in lens.cpu():
        l = l.item()
        mask.append(([1] * l) + ([0] * (size - l)))
    return torch.tensor(mask, device=lens.device).long()


def select_indices(collection, indices):
    if isinstance(collection, list):
        return [collection[i] for i in indices]
    return collection[indices]


class TqdmFile(io.BufferedReader):
    def __init__(self, path, *args, **kwargs):
        message = None
        if 'message' in kwargs:
            message = kwargs['message']
            del kwargs['message']
        self.pbar = tqdm(desc=message, unit='B', unit_scale=True, ncols=80)
        io.BufferedReader.__init__(self, io.FileIO(path, *args, **kwargs))

    def read(self, size):
        result = io.BufferedReader.read(self, size)
        self.pbar.update(len(result))
        return result

    def seek(self, offset, whence=0):
        io.BufferedReader.seek(self, offset, whence)

    def close(self, *args, **kwargs):
        self.pbar.close()
        return io.BufferedReader.close(self, *args, **kwargs)


def extract_tarball(file_name, destination, logger=None, reset_permissions=False, ignore_root=False):
    try:
        with tarfile.open(fileobj=TqdmFile(file_name, message='extracting')) as tar:
            members = list(tar.getmembers())
            if ignore_root:
                for m in members:
                    m.name = os.path.join(*os.path.split(m.name)[1:])
            tar.extractall(path=destination, members=members)
        if reset_permissions:
            for root, dirs, files in os.walk(destination):
                for momo in dirs:
                    os.chmod(os.path.join(root, momo), 0o755)
                for momo in files:
                    os.chmod(os.path.join(root, momo), 0o644)
    except:
        if logger:
            logger.warn('error encoutered; removing {}'.format(destination))
        shutil.rmtree(destination)
        raise


class DurationTimer:
    def __init__(self, gpu_sync=False, enabled=True):
        self.gpu_sync = gpu_sync
        self.enabled = enabled
        self.durations = {}

    @contextmanager
    def time(self, label=None):
        if self.enabled:
            if self.gpu_sync and torch.cuda.is_available():
                torch.cuda.synchronize()
            t = time.time()
            yield
            if self.gpu_sync and torch.cuda.is_available():
                torch.cuda.synchronize()
            d = time.time() - t
            if label not in self.durations:
                self.durations[label] = 0.
            self.durations[label] += d
        else:
            yield

    @contextmanager
    def enable(self, enabled=True):
        prev_status = self.enabled
        self.enabled = enabled
        yield
        self.enabled = prev_status

    def __str__(self):
        counts = ' '.join(f'{k}={v:.2f}s' for k, v in self.durations.items())
        return f'<DurationTimer {counts}>'

    def scaled_str(self, scale):
        counts = ' '.join(f'{k}={format_interval(v/scale)}' for k, v in self.durations.items())
        return f'<DurationTimer {counts}>'


class HierTimer(DurationTimer):
    def __init__(self, gpu_sync=False, enabled=True):
        super().__init__(gpu_sync, enabled)
        self.stack = []

    @contextmanager
    def time(self, label, skip_num=False):
        if self.enabled:
            if self.stack and not skip_num:
                self.stack += [[f'{self.stack[-1][1]:02d}-{label}', 0]]
                self.stack[-2][1] += 1
            else:
                self.stack += [[f'{label}', 0]]
            label = ':'.join([s[0] for s in self.stack])
            with super().time(label):
                yield
            self.stack = self.stack[:-1]
        else:
            yield


def subbatch(toks, maxlen):
    _, DLEN = toks.shape[:2]
    SUBBATCH = math.ceil(DLEN / maxlen)
    S = math.ceil(DLEN / SUBBATCH) if SUBBATCH > 0 else 0 # minimize the size given the number of subbatch
    stack = []
    if SUBBATCH == 1:
        return toks, SUBBATCH
    else:
        for s in range(SUBBATCH):
            stack.append(toks[:, s*S:(s+1)*S])
            if stack[-1].shape[1] != S:
                nulls = torch.zeros_like(toks[:, :S - stack[-1].shape[1]])
                stack[-1] = torch.cat([stack[-1], nulls], dim=1)
        return torch.cat(stack, dim=0), SUBBATCH


def un_subbatch(embed, toks, maxlen):
    BATCH, DLEN = toks.shape[:2]
    SUBBATCH = math.ceil(DLEN / maxlen)
    if SUBBATCH == 1:
        return embed
    else:
        embed_stack = []
        for b in range(SUBBATCH):
            embed_stack.append(embed[b*BATCH:(b+1)*BATCH])
        embed = torch.cat(embed_stack, dim=1)
        embed = embed[:, :DLEN]
        return embed


def confirm(message):
    sys.stdout.write(f'\n{message}\n')
    while True:
        sys.stdout.write('\nanswer [yes/no] ')
        answer = input().strip().lower()
        if answer in ('yes', 'y'):
            return True
        elif answer in ('no', 'n'):
            return False


def allow_redefinition(fn):
    def wrapped(obj, *args, **kwargs):
        if hasattr(obj, fn.__name__):
            try:
                return getattr(obj, fn.__name__)(*args, **kwargs)
            except NotImplementedError:
                pass # fall back on default if not implemented
        return fn(obj, *args, **kwargs)
    return wrapped


def allow_redefinition_iter(fn):
    def wrapped(obj, *args, **kwargs):
        if hasattr(obj, fn.__name__):
            try:
                yield from getattr(obj, fn.__name__)(*args, **kwargs)
            except NotImplementedError:
                # fall back on default if not implemented
                yield from fn(obj, *args, **kwargs)
        else:
            yield from fn(obj, *args, **kwargs)
    return wrapped


def device(config, logger=None):
    device = torch.device('cpu')
    if config['gpu']:
        if not torch.cuda.is_available():
            logger.error('gpu=True, but CUDA is not available. Falling back on CPU.')
        else:
            if config['gpu_determ']:
                if logger is not None:
                    logger.debug('using GPU (deterministic)')
            else:
                if logger is not None:
                    logger.debug('using GPU (non-deterministic)')
            device = torch.device('cuda')
            torch.backends.cudnn.deterministic = config['gpu_determ']
    return device


@contextmanager
def finialized_file(path, mode):
    try:
        with open(f'{path}.tmp', mode) as f:
            yield f
        os.replace(f'{path}.tmp', path)
    except:
        try:
            os.remove(f'{path}.tmp')
        except:
            pass # ignore
        raise


def path_to_stream(arg=0, mode='rt'):
    def wrapped(fn):
        def inner(*args, **kwargs):
            fn_or_stream = args[arg]
            if not hasattr(fn_or_stream, 'read'):
                with open(fn_or_stream, mode) as f:
                    args = (*args[:arg], f, *args[arg+1:])
                    return fn(*args, **kwargs)
            else:
                return fn(*args, **kwargs)
        return inner
    return wrapped



class Registry:
    def __init__(self, default: str = None):
        self.registered = {}
        self.default = default

    def register(self, name):
        rgstr = self
        def wrapped(fn):
            rgstr.registered[name] = fn
            fn.name = name
            return fn
        return wrapped

    def copy(self, default=None):
        result = Registry(default=default or self.default)
        result.registered = dict(self.registered)
        return result

def format_interval(t):
    # adapted from tqdm.format_interval, but with better support for short durations (under 1min)
    mins, s = divmod(t, 60)
    h, m = divmod(int(mins), 60)
    if h:
        return '{0:d}:{1:02d}:{2:02.0f}'.format(h, m, s)
    if m:
        return '{0:02d}:{1:02.0f}'.format(m, s)
    if s >= 1:
        return '{0:.2f}s'.format(s)
    return '{0:.0f}ms'.format(s*1000)
