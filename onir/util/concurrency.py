import os
from typing import Tuple, Iterable
from threading import Thread, Event, Lock


__all__ = ['safe_thread_count', 'blocking_tee', 'background']


def safe_thread_count(pct=0.8):
    # TODO: maybe a smarter way of doing this?
    return int(os.cpu_count() * pct)


def blocking_tee(it: Iterable, n: int) -> Tuple[Tuple]:
    """
    Functions similar to itertools.tee, though rather than caching the iterator value, blocks the
    current thread until all tee'd iterators have next'd the current item from the iterator.
    """
    ctrl = _BlockingTeeControllerThread(it, n)
    sub_iterators = tuple(_blocking_tee_iter(ctrl, ev) for ev in ctrl.ev_value_available)
    return sub_iterators


def background(it: Iterable) -> Iterable:
    """
    Runs the provided iterable in the background.
    This is a special case of blocking_tee, in which n=1
    """
    return blocking_tee(it, n=1)[0]


class CtxtThread(Thread):
    def __init__(self, fn):
        super().__init__(target=fn)

    def __enter__(self):
        self.start()

    def __exit__(self, t, value, traceback):
        self.join()

def iter_noop(it: Iterable):
    for _ in it:
        pass

def _blocking_tee_iter(ctrl, ev_available):
    ctrl.start_if_needed()
    value = None
    while True:
        ev_available.wait()
        ev_available.clear()
        if ctrl.ex:
            if isinstance(ctrl.ex, StopIteration):
                break
            else:
                raise ctrl.ex
        value = ctrl.value
        ctrl.notify_grabbed()
        if value is not StopIteration:
            yield value


class _BlockingTeeControllerThread(Thread):
    def __init__(self, it, n):
        super().__init__(daemon=True)
        self.ev_value_available = [Event() for _ in range(n)]
        self.value = None
        self.grabs_remaining = n
        self.grabs_remaining_lock = Lock()
        self.ev_all_grabbed = Event()
        self.start_lock = Lock()
        self._has_started = False
        self._it = iter(it)
        self._n = n
        self.ex = None

    def run(self):
        while True:
            try:
                self.value = next(self._it)
            except Exception as ex:
                self.ex = ex
                raise
            finally:
                self.ev_all_grabbed.clear()
                with self.grabs_remaining_lock:
                    self.grabs_remaining = self._n
                for ev in self.ev_value_available:
                    ev.set()
                self.ev_all_grabbed.wait()

    def start_if_needed(self):
        with self.start_lock:
            if not self._has_started:
                self._has_started = True
                self.start()

    def notify_grabbed(self):
        with self.grabs_remaining_lock:
            self.grabs_remaining -= 1
            if self.grabs_remaining == 0:
                self.ev_all_grabbed.set()


class Lazy:
    def __init__(self, fn):
        self._lock = Lock()
        self._fn = fn
        self._loaded = False
        self._result = None

    def __call__(self):
        if not self._loaded:
            with self._lock:
                if not self._loaded: # repeat condition from above in thread-safe way
                    self._result = self._fn()
                    self._loaded = True
        return self._result

    @property
    def is_loaded(self):
        return self._loaded
