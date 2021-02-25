import os
import math
import json
import tempfile
import itertools
import time
import re
import shutil
import threading
import contextlib
from functools import lru_cache
from pytools import memoize_method
import onir
from onir.interfaces import trec
from onir.interfaces.java import J
from onir import indices

logger = onir.log.easy()


J.register(jars=[J.locate_anserini_jar()], defs=dict(
    # [L]ucene
    L_FSDirectory='org.apache.lucene.store.FSDirectory',
    L_DirectoryReader='org.apache.lucene.index.DirectoryReader',
    L_Term='org.apache.lucene.index.Term',
    L_IndexSearcher='org.apache.lucene.search.IndexSearcher',
    L_BM25Similarity='org.apache.lucene.search.similarities.BM25Similarity',
    L_ClassicSimilarity='org.apache.lucene.search.similarities.ClassicSimilarity',
    L_LMDirichletSimilarity='org.apache.lucene.search.similarities.LMDirichletSimilarity',
    L_QueryParser='org.apache.lucene.queryparser.flexible.standard.StandardQueryParser',
    L_QueryParserUtil='org.apache.lucene.queryparser.flexible.standard.QueryParserUtil',
    L_StandardAnalyzer='org.apache.lucene.analysis.standard.StandardAnalyzer',
    L_EnglishAnalyzer='org.apache.lucene.analysis.en.EnglishAnalyzer',
    L_CharArraySet='org.apache.lucene.analysis.CharArraySet',
    L_MultiFields='org.apache.lucene.index.MultiFields',

    # [A]nserini
    A_IndexCollection='io.anserini.index.IndexCollection',
    A_IndexArgs='io.anserini.index.IndexArgs',
    A_IndexUtils='io.anserini.index.IndexUtils',
    A_LuceneDocumentGenerator='io.anserini.index.generator.LuceneDocumentGenerator',
    A_SearchCollection='io.anserini.search.SearchCollection',
    A_SearchArgs='io.anserini.search.SearchArgs',
    A_DefaultEnglishAnalyzer='io.anserini.analysis.DefaultEnglishAnalyzer',
    A_SimpleSearcher='io.anserini.search.SimpleSearcher',
    A_AnalyzerUtils='io.anserini.analysis.AnalyzerUtils',

    # [M]isc
    M_CmdLineParser='org.kohsuke.args4j.CmdLineParser',
))


def _surpress_log(java_class, levels=('DEBUG', 'INFO')):
    re_levels = r'|'.join([re.escape(l) for l in levels])
    re_java_class = re.escape(java_class)
    regex = rf'({re_levels}) {re_java_class}'
    def wrapped(log_line):
        return re.search(regex, log_line) is None
    return wrapped


def pbar_bq_listener(pbar):
    def wrapped(log_line):
        match = re.search(r'INFO io.anserini.search.SearchCollection \[pool-.*-thread-.*\] ([0-9]+) queries processed', log_line)
        if match:
            count = int(match.group(1))
            pbar.update(count - pbar.n)
            return False
    return wrapped


class AnseriniIndex(indices.BaseIndex):
    """
    Interface to an Anserini index.
    """
    def __init__(self, path, keep_stops=False, stemmer='porter', field='text', store_raw_docs=False, lang='en'):
        self._path = path
        os.makedirs(path, exist_ok=True)
        self._settings_path = os.path.join(path, 'settings.json')
        if os.path.exists(self._settings_path):
            self._load_settings()
            assert self._settings['keep_stops'] == keep_stops
            assert self._settings['stemmer'] == stemmer
            assert self._settings['field'] == field
            assert self._settings['store_raw_docs'] == store_raw_docs
            assert self._settings['lang'] == lang
        else:
            self._settings = {
                'keep_stops': keep_stops,
                'stemmer': stemmer,
                'field': field,
                'built': False,
                'store_raw_docs': store_raw_docs,
                'lang': lang,
            }
            self._dump_settings()

    def _dump_settings(self):
        with open(self._settings_path, 'wt') as f:
            json.dump(self._settings, f)

    def _load_settings(self):
        with open(self._settings_path, 'rt') as f:
            self._settings = json.load(f)
            # Apply defaults for legacy support
            if 'store_raw_docs' not in self._settings:
                self._settings['store_raw_docs'] = False
            if 'lang' not in self._settings:
                self._settings['lang'] = 'en'

    def built(self):
        self._load_settings()
        return self._settings['built']

    def num_docs(self):
        return self._reader().numDocs()

    def docids(self):
        index_utils = self._get_index_utils()
        for i in range(self.num_docs()):
            yield index_utils.convertLuceneDocidToDocid(i)

    def get_raw(self, did):
        return self._get_index_utils().getRawDocument(did)

    def path(self):
        return self._path

    @memoize_method
    def _reader(self):
        return J.L_DirectoryReader.open(J.L_FSDirectory.open(J.File(self._path).toPath()))

    @memoize_method
    def _searcher(self):
        return J.L_IndexSearcher(self._reader().getContext())

    @memoize_method
    def term2idf(self, term):
        term = J.A_AnalyzerUtils.analyze(self._get_stemmed_analyzer(), term).toArray()
        if term:
            df = self._reader().docFreq(J.L_Term(J.A_IndexArgs.CONTENTS, term[0]))
            return math.log((self._reader().numDocs() + 1) / (df + 1))
        return 0. # stop word; very common

    @memoize_method
    def term2idf_unstemmed(self, term):
        term = J.A_AnalyzerUtils.analyze(self._get_analyzer(), term).toArray()
        if len(term) == 1:
            df = self._reader().docFreq(J.L_Term(J.A_IndexArgs.CONTENTS, term[0]))
            return math.log((self._reader().numDocs() + 1) / (df + 1))
        return 0. # stop word; very common

    @memoize_method
    def collection_stats(self):
        return self._searcher().collectionStatistics(J.A_IndexArgs.CONTENTS)

    def document_vector(self, did):
        result = {}
        ldid = self._get_index_utils().convertDocidToLuceneDocid(did)
        vec = self._reader().getTermVector(ldid, J.A_IndexArgs.CONTENTS)
        it = vec.iterator()
        while it.next():
            result[it.term().utf8ToString()] = it.totalTermFreq()
        return result

    def avg_dl(self):
        cs = self.collection_stats()
        return cs.sumTotalTermFreq() / cs.docCount()

    @memoize_method
    def _get_index_utils(self):
        return J.A_IndexUtils(self._path)

    @lru_cache(maxsize=16)
    def get_doc(self, did):
        ldid = self._get_index_utils().convertDocidToLuceneDocid(did)
        if ldid == -1:
            return ["a"] # hack -- missing doc
        return self._get_index_utils().getTransformedDocument(did) or ["a"]

    @memoize_method
    def _get_analyzer(self):
        return J.L_StandardAnalyzer(J.L_CharArraySet(0, False))

    @memoize_method
    def _get_stemmed_analyzer(self):
        return J.A_DefaultEnglishAnalyzer.newStemmingInstance(self._settings['stemmer'], J.L_CharArraySet(0, False))

    def tokenize(self, text):
        result = J.A_AnalyzerUtils.analyze(self._get_analyzer(), text).toArray()
        # mostly good, just gonna split off contractions
        result = list(itertools.chain(*(x.split("'") for x in result)))
        return result

    def iter_terms(self):
        field = J.L_MultiFields.getFields(self._reader())
        it = field.terms(J.A_IndexArgs.CONTENTS).iterator()
        while it.next():
            yield {
                'term': it.term().utf8ToString(),
                'df': it.docFreq(),
                'cf': it.totalTermFreq(),
            }


    @memoize_method
    def _model(self, model):
        if model == 'randomqrels':
            return self._model('bm25_k1-0.6_b-0.5')
            # return self._model('bm25_k1-1.2_b-0.4')
        if model.startswith('bm25'):
            k1, b = 0.9, 0.4
            for k, v in [arg.split('-') for arg in model.split('_')[1:]]:
                if k == 'k1':
                    k1 = float(v)
                elif k == 'b':
                    b = float(v)
                else:
                    raise ValueError(f'unknown bm25 parameter {k}={v}')
            return J.L_BM25Similarity(k1, b)
        elif model == 'vsm':
            return J.L_ClassicSimilarity()
        elif model == 'ql':
            mu = 1000.
            for k, v in [arg.split('-') for arg in model.split('_')[1:]]:
                if k == 'mu':
                    mu = float(v)
                else:
                    raise ValueError(f'unknown ql parameter {k}={v}')
            return J.L_LMDirichletSimilarity(mu)
        raise ValueError(f'unknown model {model}')

    @memoize_method
    def get_query_doc_scores(self, query, did, model, skip_invividual=False):
        sim = self._model(model)
        self._searcher().setSimilarity(sim)
        ldid = self._get_index_utils().convertDocidToLuceneDocid(did)
        if ldid == -1:
            return -999. * len(query), [-999.] * len(query)
        analyzer = self._get_stemmed_analyzer()
        query = list(itertools.chain(*[J.A_AnalyzerUtils.analyze(analyzer, t).toArray() for t in query]))
        if not skip_invividual:
            result = []
            for q in query:
                q = _anserini_escape(q, J)
                lquery = J.L_QueryParser().parse(q, J.A_IndexArgs.CONTENTS)
                explain = self._searcher().explain(lquery, ldid)
                result.append(explain.getValue().doubleValue())
            return sum(result), result
        lquery = J.L_QueryParser().parse(_anserini_escape(' '.join(query), J), J.A_IndexArgs.CONTENTS)
        explain = self._searcher().explain(lquery, ldid)
        return explain.getValue()

    def get_query_doc_scores_batch(self, query, dids, model):
        sim = self._model(model)
        self._searcher().setSimilarity(sim)
        ldids = {self._get_index_utils().convertDocidToLuceneDocid(did): did for did in dids}
        analyzer = self._get_stemmed_analyzer()
        query = J.A_AnalyzerUtils.analyze(analyzer, query).toArray()
        query = ' '.join(_anserini_escape(q, J) for q in query)
        docs = ' '.join(f'{J.A_IndexArgs.ID}:{did}' for did in dids)
        lquery = J.L_QueryParser().parse(f'({query}) AND ({docs})', J.A_IndexArgs.CONTENTS)
        result = {}
        search_results = self._searcher().search(lquery, len(dids))
        for top_doc in search_results.scoreDocs:
            result[ldids[top_doc.doc]] = top_doc.score
        del search_results
        return result


    def build(self, doc_iter, replace=False, optimize=True, store_term_weights=False):
        with logger.duration(f'building {self._path}'), J.listen_java_log(_surpress_log('io.anserini.index.IndexCollection')):
            thread_count = onir.util.safe_thread_count()
            with tempfile.TemporaryDirectory() as d:
                if self._settings['built']:
                    if replace:
                        logger.warn(f'removing index: {self._path}')
                        shutil.rmtree(self._path)
                    else:
                        logger.warn(f'adding to existing index: {self._path}')
                fifos = []
                for t in range(thread_count):
                    fifo = os.path.join(d, f'{t}.json')
                    os.mkfifo(fifo)
                    fifos.append(fifo)
                index_args = J.A_IndexArgs()
                index_args.collectionClass = 'JsonCollection'
                index_args.generatorClass = 'LuceneDocumentGenerator'
                index_args.threads = thread_count
                index_args.input = d
                index_args.index = self._path
                index_args.storePositions = True
                index_args.storeDocvectors = True
                index_args.storeRawDocs = self._settings['store_raw_docs']
                index_args.storeTermWeights = store_term_weights
                index_args.keepStopwords = self._settings['keep_stops']
                index_args.stemmer = self._settings['stemmer']
                index_args.language = self._settings['lang']
                index_args.optimize = optimize
                indexer = J.A_IndexCollection(index_args)
                thread = threading.Thread(target=indexer.run)
                thread.start()
                time.sleep(1) # give it some time to start up, otherwise fails due to race condition
                for i, doc in enumerate(doc_iter):
                    f = fifos[hash(i) % thread_count]
                    if isinstance(f, str):
                        f = open(f, 'wt')
                        fifos[hash(i) % thread_count] = f
                    json.dump({'id': doc.did, 'contents': doc.data[self._settings['field']]}, f)
                    f.write('\n')
                for f in fifos:
                    if not isinstance(f, str):
                        f.close()
                    else:
                        with open(f, 'wt'):
                            pass # open and close to indicate file is done
                logger.debug('waiting to join')
                thread.join()
                self._settings['built'] = True
                self._dump_settings()

    def build_jsoup(self, path, replace=False, optimize=True):
        with logger.duration(f'building {self._path}'):
            if self._settings['built']:
                if replace:
                    logger.warn(f'removing index: {self._path}')
                    shutil.rmtree(self._path)
                else:
                    logger.warn(f'adding to existing index: {self._path}')
            thread_count = onir.util.safe_thread_count()
            index_args = J.A_IndexArgs()
            index_args.collectionClass = 'TrecCollection'
            index_args.generatorClass = 'JsoupGenerator'
            index_args.threads = thread_count
            index_args.input = path
            index_args.index = self._path
            index_args.storePositions = True
            index_args.storeDocvectors = True
            index_args.storeRawDocs = True
            index_args.storeTransformedDocs = True
            index_args.keepStopwords = self._settings['keep_stops']
            index_args.stemmer = self._settings['stemmer']
            index_args.optimize = optimize
            indexer = J.A_IndexCollection(index_args)
            thread = threading.Thread(target=indexer.run)
            thread.start()
            thread.join()
            self._settings['built'] = True
            self._dump_settings()

    def query(self, query, model, topk, destf=None, quiet=False):
        return self.batch_query([('0', query)], model, topk, destf=destf, quiet=quiet)['0']

    def batch_query(self, queries, model, topk, destf=None, quiet=False):
        THREADS = onir.util.safe_thread_count()
        query_file_splits = 1000
        if hasattr(queries, '__len__'):
            if len(queries) < THREADS:
                THREADS = len(queries)
                query_file_splits = 1
            elif len(queries) < THREADS * 10:
                query_file_splits = ((len(queries)+1) // THREADS)
            elif len(queries) < THREADS * 100:
                query_file_splits = ((len(queries)+1) // (THREADS * 10))
            else:
                query_file_splits = ((len(queries)+1) // (THREADS * 100))
        with tempfile.TemporaryDirectory() as topic_d, tempfile.TemporaryDirectory() as run_d:
            run_f = os.path.join(run_d, 'run')
            topic_files = []
            current_file = None
            total_topics = 0
            for i, (qid, text) in enumerate(queries):
                topic_file = '{}/{}.queries'.format(topic_d, i // query_file_splits)
                if current_file is None or current_file.name != topic_file:
                    if current_file is not None:
                        topic_files.append(current_file.name)
                        current_file.close()
                    current_file = open(topic_file, 'wt')
                current_file.write(f'{qid}\t{text}\n')
                total_topics += 1
            if current_file is not None:
                topic_files.append(current_file.name)
            current_file.close()
            args = J.A_SearchArgs()
            parser = J.M_CmdLineParser(args)
            arg_args = [
                '-index', self._path,
                '-topics', *topic_files,
                '-output', run_f,
                '-topicreader', 'TsvString',
                '-threads', str(THREADS),
                '-hits', str(topk),
                '-language', self._settings['lang'],
            ]
            if model.startswith('bm25'):
                arg_args.append('-bm25')
                model_args = [arg.split('-', 1) for arg in model.split('_')[1:]]
                for arg in model_args:
                    if len(arg) == 1:
                        k, v = arg[0], None
                    elif len(arg) == 2:
                        k, v = arg
                    if k == 'k1':
                        arg_args.append('-bm25.k1')
                        arg_args.append(v)
                    elif k == 'b':
                        arg_args.append('-bm25.b')
                        arg_args.append(v)
                    elif k == 'rm3':
                        arg_args.append('-rm3')
                    elif k == 'rm3.fbTerms':
                        arg_args.append('-rm3.fbTerms')
                        arg_args.append(v)
                    elif k == 'rm3.fbDocs':
                        arg_args.append('-rm3.fbDocs')
                        arg_args.append(v)
                    else:
                        raise ValueError(f'unknown bm25 parameter {arg}')
            elif model.startswith('ql'):
                arg_args.append('-qld')
                model_args = [arg.split('-', 1) for arg in model.split('_')[1:]]
                for arg in model_args:
                    if len(arg) == 1:
                        k, v = arg[0], None
                    elif len(arg) == 2:
                        k, v = arg
                    if k == 'mu':
                        arg_args.append('-qld.mu')
                        arg_args.append(v)
                    else:
                        raise ValueError(f'unknown ql parameter {arg}')
            elif model.startswith('sdm'):
                arg_args.append('-sdm')
                arg_args.append('-qld')
                model_args = [arg.split('-', 1) for arg in model.split('_')[1:]]
                for arg in model_args:
                    if len(arg) == 1:
                        k, v = arg[0], None
                    elif len(arg) == 2:
                        k, v = arg
                    if k == 'mu':
                        arg_args.append('-qld.mu')
                        arg_args.append(v)
                    elif k == 'tw':
                        arg_args.append('-sdm.tw')
                        arg_args.append(v)
                    elif k == 'ow':
                        arg_args.append('-sdm.ow')
                        arg_args.append(v)
                    elif k == 'uw':
                        arg_args.append('-sdm.uw')
                        arg_args.append(v)
                    else:
                        raise ValueError(f'unknown sdm parameter {arg}')
            else:
                raise ValueError(f'unknown model {model}')
            parser.parseArgument(*arg_args)
            with contextlib.ExitStack() as stack:
                stack.enter_context(J.listen_java_log(_surpress_log('io.anserini.search.SearchCollection')))
                if not quiet:
                    pbar = stack.enter_context(logger.pbar_raw(desc=f'batch_query ({model})', total=total_topics))
                    stack.enter_context(J.listen_java_log(pbar_bq_listener(pbar)))
                searcher = J.A_SearchCollection(args)
                searcher.runTopics()
                searcher.close()
            if destf:
                shutil.copy(run_f, destf)
            else:
                return trec.read_run_dict(run_f)

    @memoize_method
    def simple_searcher(self, model):
        result = J.A_SimpleSearcher(self._path)
        if model.startswith('bm25'):
            k1, b = 0.9, 0.4
            model_args = [arg.split('-', 1) for arg in model.split('_')[1:]]
            for arg in model_args:
                if len(arg) == 1:
                    k, v = arg[0], None
                elif len(arg) == 2:
                    k, v = arg
                if k == 'k1':
                    k1 = v
                elif k == 'b':
                    b = v
                else:
                    raise ValueError(f'unknown bm25 parameter {arg}')
            result.setBM25Similarity(k1, b)
        else:
            raise ValueError(f'unsupported model {model}')
        return result

    def query_simplesearcher(self, query, model, topk):
        searcher = self.simple_searcher(model)
        result = {}
        for record in searcher.search(query, topk):
            result[record.docid] = record.score
        return result



def _anserini_escape(text, J):
    text = J.L_QueryParserUtil.escape(text)
    text = text.replace('<', '\\<')
    text = text.replace('>', '\\>')
    text = text.replace('=', '\\=')
    return text
