import sys
import bdb

logger = None

def handle_exception(exc_type, exc_value, exc_traceback):
    if logger is None:
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    if issubclass(exc_type, KeyboardInterrupt):
        logger.debug("Keyboard Interrupt", exc_info=(exc_type, exc_value, exc_traceback))
    elif issubclass(exc_type, bdb.BdbQuit):
        logger.debug("Quit", exc_info=(exc_type, exc_value, exc_traceback))
    else:
        logger.critical("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))

sys.excepthook = handle_exception

from onir import log

logger = log.Logger('onir')

from onir import util, injector, metrics, datasets, interfaces, rankers, config, trainers, predictors, vocab, pipelines
