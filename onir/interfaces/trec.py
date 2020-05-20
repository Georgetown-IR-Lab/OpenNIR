import os
import re
import gzip
import tempfile
import subprocess
from typing import Iterator, Tuple, Dict
from collections.abc import Iterable as IterableAbc
from multiprocessing import Pool
from contextlib import contextmanager
import pandas as pd
from tqdm import tqdm
from bs4 import BeautifulSoup
from onir.interfaces import plaintext
from onir import indices, util


class Qrels:
    def __init__(self, data, sep=None):
        self._data = {}
        self._sep = sep
        self._iter = None
        if isinstance(data, str):
            self._data['trec'] = data
        elif isinstance(data, dict):
            self._data['dict'] = data
        elif isinstance(data, pd.DataFrame):
            self._data['df'] = data
        elif isinstance(data, IterableAbc):
            if hasattr(data, 'read'):
                if hasattr(data, 'name') and os.path.isfile(data.name): # file object
                    self._data['trec'] = data.name
                else:
                    it = (line.split() for line in data)
                    it = ((qid, did, int(score)) for qid, _, did, score in it)
                    self._iter = it
            else:
                self._iter = iter(data)


    def __iter__(self):
        if self._iter is not None:
            it = self._iter
            self._iter = None # will consume iterator
            return it
        if 'df' in self._data:
            return self._data['df']
        if 'dict' in self._data:
            return ((qid, did, score) for qid, docs in self._data['dict'].items() \
                                      for did, score in docs.items())
        if 'trec' in self._data and os.path.exists(self._data['trec']):
            return self._iter_file(self._data['trec'])
        raise ValueError('cannot iterate')

    def trec(self):
        if 'trec' not in self._data or not os.path.exists(self._data['trec']):
            raise ValueError('no file for qrels')
        return self._data['trec']

    def dict(self):
        if 'dict' not in self._data:
            result = {}
            for qid, did, score in iter(self):
                if qid not in result:
                    result[qid] = {}
                result[qid][did] = score
            self._data['dict'] = result
        return self._data['dict']

    def df(self):
        if 'df' not in self._data:
            self._data['df'] = pd.DataFrame(iter(self), columns=['qid', 'did', 'score'])
        return self._data['df']

    def get(self, fmt):
        return {
            'trec': self.trec,
            'dict': self.dict,
            'df': self.df,
        }[fmt]()

    def _iter_file(self, file, sep=None):
        for qid, _, docid, score in plaintext.read_sv(file, sep=(sep or self._sep)):
            yield qid, docid, int(score)

    def save_file(self, path, link_ok=True, sep=None):
        if link_ok and 'trec' in self._data and os.path.exists(self._data['file']):
            os.symlink(self._data['trec'], path)
        else:
            it = ((qid, '0', did, score) for qid, did, score in iter(self))
            plaintext.write_sv(path, it, sep=(sep or self._sep))


def read_qrels(file: str, sep: str = None) -> Iterator[Tuple[str, str, int]]:
    return iter(Qrels(file, sep))


def read_qrels_dict(file: str, sep: str = None) -> Dict[str, Dict[str, int]]:
    return Qrels(file, sep).dict()


def read_qrels_df(file: str, sep: str = None) -> pd.DataFrame:
    return Qrels(file, sep).df()


def read_qrels_fmt(file: str, fmt: str, sep: str = None):
    return Qrels(file, sep).get(fmt)


def write_qrels(file, qrels_iter, sep=' '):
    Qrels(qrels_iter, sep).save_file(file)


def write_qrels_dict(file, qrels_dict, sep=' '):
    Qrels(qrels_dict, sep).save_file(file)


def write_qrels_dict_(qrels_dict, file):
    return write_qrels_dict(file, qrels_dict)


def read_run(file):
    """
    Reads query-document run file

    Args:
        file (str|Stream) file path (str) or stream (Stream) to read run from

    Returns:
        iter<tuple<str,str,int,float>> -- iterator with <qid, docid, rank, score>
    """
    for qid, _, docid, rank, score, _ in plaintext.read_sv(file, sep=' '):
        yield qid, docid, int(rank), float(score)


def read_run_dict(file, top=None):
    """
    Reads query-document run file and returns results as a dictionary

    Args:
        file (str|Stream) file path (str) or stream (Stream) to read run from
        top (int|None) return only at most this many records per run (highest-scoring)

    Returns:
        dict<str,dict<str,float>> -- dictionary like {qid: {docid: score}}
    """
    result = {}
    for qid, docid, rank, score in read_run(file):
        if top is None or rank <= top:
            result.setdefault(qid, {})[docid] = score
    return result


def read_run_df(file, top=None):
    return pd.DataFrame(((qid, did, score) for qid, did, rank, score in tqdm(read_run(file), leave=False) if top is None or rank <= top), columns=['qid', 'did', 'score'])


def read_run_fmt(file, fmt, top=None):
    """
    Reads query-document run file and returns results in the specified format

    Args:
        file (str|Stream) file path (str) or stream (Stream) to read qrels from
        fmt (str) format to read run as from {'trec', 'dict', 'df'}

    Returns:
        depends on fmt:
          fmt == 'trec': str representing a file that contains the qrels in trec format
          fmt == 'dict': dict<str,dict<str,int>> dictionary like {qid: {docid: score}}
          fmt == 'df': pd.DataFrame with qid, did, and score columns
    """
    return {
        'trec': lambda file: file.name if hasattr(file, 'name') else file,
        'dict': lambda file: read_run_dict(file, top),
        'df': lambda file: read_run_df(file, top),
    }[fmt](file)


def read_sample_dict(file):
    result = {}
    for qid, _, docid, cat, rel in plaintext.read_sv(file, sep=' '):
        result.setdefault(qid, {})[docid] = (cat, int(rel))
    return result


def write_run_dict(file, run_dict, runid='run'):
    """
    Writes a query-document run dictionary to the given file

    Args:
        file (str|Stream) file path (str) or stream (Stream) to read to
        run_dict (dict<str<dict<str,float>>) run scores of format {qid: {docid: score}}
        runid (str, optional) run name to output (optional)
    """
    def run_iter():
        for qid in run_dict:
            for i, (docid, score) in enumerate(sorted(run_dict[qid].items(), key=lambda x: (-x[1], x[0]))):
                yield qid, 'Q0', docid, i+1, score, runid
    plaintext.write_sv(file, run_iter(), sep=' ')


def write_run_dict_(run_dict, file, runid='run'):
    return write_run_dict(file, run_dict, runid)


def write_sample_dict(sample_dict, file):
    def sample_iter():
        for qid in sample_dict:
            for did in sample_dict[qid]:
                cat, rel = sample_dict[qid][did]
                yield qid, "0", did, cat, rel
    plaintext.write_sv(file, sample_iter(), sep=' ')


def dict2df(query_dict: dict) -> pd.DataFrame:
    data = {'qid': [], 'did': [], 'score': []}
    for qid, doc_dict in sorted(query_dict.items()):
        for did, score in sorted(doc_dict.items()):
            data['qid'].append(qid)
            data['did'].append(did)
            data['score'].append(score)
    return pd.DataFrame(data)


@contextmanager
def _write_or_file(file_or_data, writefn):
    if isinstance(file_or_data, str):
        yield file_or_data
    else:
        with tempfile.NamedTemporaryFile('wt') as f:
            writefn(file_or_data, f)
            yield f.name


def _strip_html(doc_text):
    return BeautifulSoup(doc_text, "html.parser").get_text()


def parse_doc_format(path, encoding="ISO-8859-1"):
    files = list(_parse_doc_format_files(path))
    args = [(f, encoding) for f in files]
    for docs in map(_parse_doc_file, args):
        yield from docs

def _parse_doc_format_files(path):
    if os.path.isdir(path):
        for file in os.listdir(path):
            yield from _parse_doc_format_files(os.path.join(path, file))
    else:
        yield path


DOC_TEXT_TAGS = ["<TEXT>", "<HEADLINE>", "<TITLE>", "<HL>", "<HEAD>", "<TTL>", "<DD>", "<DATE>", "<LP>", "<LEADPARA>"]
DOC_TEXT_END_TAGS = ["</TEXT>", "</HEADLINE>", "</TITLE>", "</HL>", "</HEAD>", "</TTL>", "</DD>", "</DATE>", "</LP>", "</LEADPARA>"]

# Adapted from https://github.com/castorini/anserini/blob/master/src/main/java/io/anserini/collection/TrecCollection.java
def _parse_doc_file(args):
    path, encoding = args
    docs = []
    if path.endswith('.gz'):
        open_fn = gzip.open
    else:
        open_fn = open
    with open_fn(path, 'rt', encoding=encoding, errors='replace') as file:
        docid = None
        doc_text = ''
        tag_no = None
        while file:
            line = next(file, StopIteration)
            if line is StopIteration:
                break
            if line.startswith('<DOC ') or line.startswith('<DOC>'):
                match = re.match(r".*id=\"([^\"]+)\".*", line)
                if match:
                    docid = match.group(1)
            elif line.startswith('<DOCNO>'):
                while '</DOCNO>' not in line:
                    l = next(file, StopIteration)
                    if l is StopIteration:
                        break
                    line += l
                docid = line.replace('<DOCNO>', '').replace('</DOCNO>', '').strip()
            elif line.startswith('</DOC>'):
                assert docid is not None
                docs.append(indices.RawDoc(docid, _strip_html(doc_text)))
                docid = None
                doc_text = ''
                tag_no = None
            elif tag_no is not None:
                doc_text += line
                if line.startswith(DOC_TEXT_END_TAGS[tag_no]):
                    tag_no = None
            else:
                for i, tag in enumerate(DOC_TEXT_TAGS):
                    if line.startswith(tag):
                        tag_no = i
                        doc_text += line
                        break
    return docs

def parse_query_format(file, xml_prefix=None):
    if xml_prefix is None:
        xml_prefix = ''
    if hasattr(file, 'read'):
        num, title, desc, narr, reading = None, None, None, None, None
        for line in file:
            if line.startswith('**'):
                continue # translation comment in older formats (e.g., TREC 3 Spanish track)
            elif line.startswith('</top>'):
                if title is not None:
                    yield 'topic', num, title.replace('\t', ' ').strip()
                if desc is not None:
                    yield 'desc', num, desc.replace('\t', ' ').strip()
                if narr is not None:
                    yield 'narr', num, narr.replace('\t', ' ').strip()
                num, title, desc, narr, reading = None, None, None, None, None
            elif line.startswith('<num>'):
                num = line[len('<num>'):].replace('Number:', '').strip()
                reading = None
            elif line.startswith(f'<{xml_prefix}title>'):
                title = line[len(f'<{xml_prefix}title>'):].strip()
                if title == '':
                    reading = 'title'
                else:
                    reading = None
            elif line.startswith(f'<{xml_prefix}desc>'):
                desc = ''
                reading = 'desc'
            elif line.startswith(f'<{xml_prefix}narr>'):
                narr = ''
                reading = 'narr'
            elif reading == 'desc':
                desc += line.strip() + ' '
            elif reading == 'narr':
                narr += line.strip() + ' '
            elif reading == 'title':
                title += line.strip() + ' '
    else:
        with open(file, 'rt') as f:
            yield from parse_query_format(f)
