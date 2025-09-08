import threading

from tnfr import rng as rng_mod
from tnfr.rng import get_rng
from tnfr.constants import DEFAULTS


def test_get_rng_thread_safety(monkeypatch):
    monkeypatch.setattr(rng_mod, "DEFAULTS", dict(DEFAULTS))
    monkeypatch.setitem(rng_mod.DEFAULTS, "JITTER_CACHE_SIZE", 4)
    get_rng.cache_clear()

    results = []
    errors = []
    lock = threading.Lock()

    def worker():
        try:
            rng = get_rng(123, 456)
            seq = [rng.random() for _ in range(3)]
            with lock:
                results.append(seq)
        except Exception as e:  # pragma: no cover - should not happen
            with lock:
                errors.append(e)

    threads = [threading.Thread(target=worker) for _ in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    assert all(seq == results[0] for seq in results)
