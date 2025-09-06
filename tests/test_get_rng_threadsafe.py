import threading

from tnfr import rng as rng_mod
from tnfr.rng import get_rng
from tnfr.constants import DEFAULTS


def test_get_rng_thread_safety(monkeypatch):
    monkeypatch.setattr(rng_mod, "DEFAULTS", dict(DEFAULTS))
    monkeypatch.setitem(rng_mod.DEFAULTS, "JITTER_CACHE_SIZE", 4)
    get_rng.cache_clear()
    errors = []

    def worker(idx):
        try:
            rng = get_rng(123, idx)
            rng.random()
        except Exception as e:  # pragma: no cover - should not happen
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
