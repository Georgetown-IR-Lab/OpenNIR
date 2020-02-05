import sys
import logging
import operator
from contextlib import contextmanager
from time import time
from cached_property import cached_property
import colorlog
from tqdm import tqdm
from onir import util


_tqdm_status = True


class TqdmHandler(logging.StreamHandler):
    def __init__(self):
        logging.StreamHandler.__init__(self)
        
    def emit(self, record):
        msg = self.format(record)
        if _tqdm_status:
            try:
                tqdm.write(msg)
            except AttributeError:
                sys.stderr.write(msg)
        else:
            sys.stderr.write(msg)


LOGGER_LEVELS = {
    'CRITICAL': 50,
    'DEBUG': 10,
    'ERROR': 40,
    'FATAL': 50,
    'INFO': 20,
    'NOTSET': 0,
    'WARN': 30,
    'WARNING': 30,
}


_logger_cache = {}


class Logger:
    def __init__(self, name):
        self.name = name

    @cached_property
    def logger(self):
        code = colorlog.escape_codes['thin_white']
        console_formatter = colorlog.ColoredFormatter(
            f'{code}[%(asctime)s][%(name)s][%(levelname)s] %(reset)s%(log_color)s%(message)s')
        file_formatter = logging.Formatter('[%(asctime)s][%(name)s][%(levelname)s] %(message)s')
        handler = TqdmHandler()
        handler.setFormatter(console_formatter)
        if self.name not in _logger_cache:
            logger = logging.getLogger(self.name)
            logger.addHandler(handler)
            file_handler = logging.FileHandler(util.path_log())
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)
            logger.propagate = False
            logger.setLevel(logging.DEBUG)
            _logger_cache[self.name] = logger
        return _logger_cache[self.name]

    def debug(self, text, **kwargs):
        self.logger.debug(text, **kwargs)

    def info(self, text, **kwargs):
        self.logger.info(text, **kwargs)

    def warn(self, text, **kwargs):
        self.logger.warn(text, **kwargs)

    def error(self, text, **kwargs):
        self.logger.error(text, **kwargs)

    def critical(self, text, **kwargs):
        self.logger.critical(text, **kwargs)

    def log(self, level, text, **kwargs):
        self.logger.log(LOGGER_LEVELS.get(level, 50), text, **kwargs)

    def pbar(self, it, *args, **kwargs):
        level = kwargs.pop('level', 'DEBUG')
        quiet = kwargs.pop('quiet', False)
        if 'ncols' not in kwargs:
            kwargs['ncols'] = 80
        if 'desc' in kwargs:
            if not quiet:
                self.log(level, '[starting] {desc}'.format(**kwargs))
            if 'leave' not in kwargs:
                kwargs['leave'] = False
        if 'total' not in kwargs and hasattr(it, '__len__'):
            kwargs['total'] = len(it)
        elif 'total' not in kwargs and hasattr(it, '__length_hint__'):
            kwargs['total'] = operator.length_hint(it)
        if 'smoothing' not in kwargs:
            kwargs['smoothing'] = 0. # disable smoothing by default; mean over entire life of pbar
        pbar = tqdm(it, *args, **kwargs)
        yield from pbar
        if not quiet:
            pbar.bar_format = '{desc}: [{elapsed}] [{n_fmt}it] [{rate_fmt}]'
            self.log(level, '[finished] ' + str(pbar))

    @contextmanager
    def pbar_raw(self, *args, **kwargs):
        level = kwargs.pop('level', 'DEBUG')
        quiet = kwargs.pop('quiet', False)
        if 'total' not in kwargs and 'total_from' in kwargs:
            total_from = kwargs.pop('total_from')
            if hasattr(total_from, '__len__'):
                kwargs['total'] = len(total_from)
            elif hasattr(total_from, '__length_hint__'):
                kwargs['total'] = operator.length_hint(total_from)
            else:
                raise ValueError('total_from does not have __len__ or __length_hint__')
        if 'ncols' not in kwargs:
            kwargs['ncols'] = 80
        if 'desc' in kwargs:
            if not quiet:
                self.log(level, '[starting] {desc}'.format(**kwargs))
            if 'leave' not in kwargs:
                kwargs['leave'] = False
        if 'smoothing' not in kwargs:
            kwargs['smoothing'] = 0. # disable smoothing by default; mean over entire life of pbar
        with tqdm(*args, **kwargs) as pbar:
            yield pbar
            if not quiet:
                pbar.bar_format = '{desc}: [{elapsed}] [{n_fmt}it] [{rate_fmt}]'
                self.log(level, '[finished] ' + str(pbar))

    @contextmanager
    def duration(self, message, level='DEBUG'):
        t = time()
        self.logger.log(LOGGER_LEVELS[level], f'[starting] {message}')
        yield
        output_duration = format_interval(time() - t)
        self.logger.log(LOGGER_LEVELS[level], f'[finished] {message} [{output_duration}]')


def easy():
    """
    Returns a logger with the caller's __name__
    """
    import inspect
    try:
        frame = inspect.stack()[1] # caller
        module = inspect.getmodule(frame[0])
        return Logger(module.__name__)
    except IndexError:
        return Logger('UNKNOWN')


def disable_tqdm():
    global _tqdm_status
    _tqdm_status = False

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


# override tqdm.format_interval
tqdm.format_interval = format_interval
