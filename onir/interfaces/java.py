import os
import atexit
import shutil
import tempfile
from contextlib import contextmanager
import threading
import onir

logger = onir.log.easy()


class _JavaInterface:
    def __init__(self):
        self._autoclass = None
        self._defs = {}
        self._cache = {}
        self._jars = []
        self._log_listeners = []

    def register(self, jars=None, defs=None):
        if jars is not None:
            if self._autoclass is None:
                for jar in jars:
                    self._jars.append(jar)
            else:
                raise RuntimeError('Cannot add JAR after jnius has been initialized.')
        if defs is not None:
            for n, path in defs.items():
                if n in self._defs and self._defs[n] != path:
                    logger.warn(f'{n} already defined as {self._defs[n]}. Not replacing with {path}')
                else:
                    self._defs[n] = path

    def add_log_listener(self, func):
        self._log_listeners.insert(0, func)

    @contextmanager
    def listen_java_log(self, func):
        self._log_listeners.insert(0, func)
        yield
        self._log_listeners.pop(0)

    def __getattr__(self, key):
        if self._autoclass is None:
            self.initialize()
        if key not in self._cache:
            self._cache[key] = self._autoclass(self._defs[key])
        return self._cache[key]

    def initialize(self):
        if self._autoclass is not None:
            logger.debug('jnius already initialized')
            return
        with logger.duration('initializing jnius'):
            log_fifo, l4j12_file, l4j24_file = self._init_java_logger_interface()
            import jnius_config
            jnius_config.set_classpath(*self._jars, os.path.dirname(l4j12_file))
            jnius_config.add_options(f'-Dlog4j.configuration={l4j12_file}')
            jnius_config.add_options(f'-Dlog4j.configurationFile={l4j24_file}')
            from jnius import autoclass
            self._autoclass = autoclass
            # self.PropertyConfigurator.configure(l4j12_file)
            for key, path in self._defs.items():
                self._cache[key] = self._autoclass(path)

    def _java_log_listen(self, fifo):
        while True:
            with open(fifo) as f:
                buf = ''
                for line in f:
                    if line.rstrip() == '':
                        buf = buf.rstrip()
                        for listener in self._log_listeners:
                            result = listener(buf)
                            if result == False:
                                break
                        buf = ''
                    else:
                        buf += line

    def _init_java_logger_interface(self):
        base_tmp = tempfile.mkdtemp()
        atexit.register(shutil.rmtree, base_tmp)
        l4j12_config_file = os.path.join(base_tmp, 'log4j.properties')
        l4j24_config_file = os.path.join(base_tmp, 'log4j24.xml')
        log_fifo = os.path.join(base_tmp, 'log_interface.fifo')
        os.mkfifo(log_fifo)
        log_thread = threading.Thread(target=self._java_log_listen, args=(log_fifo,), daemon=True)
        log_thread.start()
        with open(l4j12_config_file, 'wt') as f:
            f.write(f'''
log4j.rootLogger=fifo

log4j.appender.fifo=org.apache.log4j.FileAppender
log4j.appender.fifo.fileName={log_fifo}

log4j.appender.fifo.layout=org.apache.log4j.PatternLayout
log4j.appender.fifo.layout.ConversionPattern=%p %c [%t] %m%n%n
''')


        with open(l4j24_config_file, 'wt') as f:
            f.write(f'''<?xml version="1.0" encoding="UTF-8"?>
<Configuration>
  <Appenders>
    <File name="LogFifo" fileName="{log_fifo}">
      <PatternLayout>
        <Pattern>%p %c [%t] %m%n%n</Pattern>
      </PatternLayout>
    </File>
  </Appenders>
  <Loggers>
    <Root level="all">
      <AppenderRef ref="LogFifo"/>
    </Root>
  </Loggers>
</Configuration>
''')
        return log_fifo, l4j12_config_file, l4j24_config_file

J = _JavaInterface()

J.register(defs=dict(
    # Core
    System='java.lang.System',
    Array='java.lang.reflect.Array',
    File='java.io.File',
    # PropertyConfigurator='org.apache.log4j.PropertyConfigurator',
))

def onir_java_logger(log_line):
    if len(log_line.split(' ')) > 2:
        level, src, message = log_line.split(' ', 2)
        onir.log.Logger('java:' + src).log(level, message)
    else:
        onir.log.Logger('java:???').debug(log_line)
    return True

J.add_log_listener(onir_java_logger)
