import random
import threading

from tnfr import rng as rng_mod
from tnfr.constants import DEFAULTS
from tnfr.rng import clear_rng_cache, make_rng


def test_make_rng_thread_safety(monkeypatch):
    monkeypatch.setattr(rng_mod, "DEFAULTS", dict(DEFAULTS))
    monkeypatch.setitem(rng_mod.DEFAULTS, "JITTER_CACHE_SIZE", 4)
    clear_rng_cache()

    results = []
    lock = threading.Lock()

    def worker():
        rng = make_rng(123, 456)
        seq = [rng.random() for _ in range(3)]
        with lock:
            results.append(seq)

    threads = [threading.Thread(target=worker) for _ in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(results) == 20
    assert all(seq == results[0] for seq in results)


def test_no_lock_when_cache_disabled(monkeypatch):
    monkeypatch.setattr(rng_mod, "_CACHE_MAXSIZE", 0)

    class FailLock:
        def __enter__(self):  # pragma: no cover - failing ensures no lock
            raise AssertionError("lock should not be acquired")

        def __exit__(self, _exc_type, _exc, _tb):
            pass

    monkeypatch.setattr(rng_mod, "_RNG_LOCK", FailLock())
    rng = rng_mod.make_rng(123, 456)
    assert isinstance(rng, random.Random)
