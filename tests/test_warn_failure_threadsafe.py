import threading
import warnings

import tnfr.import_utils as import_utils
from tnfr.import_utils import _WARNED_LOCK, _WARNED_MODULES, _IMPORT_STATE, _warn_failure


def test_warn_failure_thread_safety(monkeypatch):
    """Ensure _warn_failure can run concurrently without races."""

    monkeypatch.setattr(import_utils, "_FAILED_IMPORT_PRUNE_INTERVAL", 0.0)

    with _WARNED_LOCK:
        _WARNED_MODULES.clear()
    with _IMPORT_STATE.lock:
        _IMPORT_STATE.clear()
        _IMPORT_STATE.last_prune = 0.0

    errors: list[Exception] = []
    lock = threading.Lock()

    def worker() -> None:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                _warn_failure("mod", None, ImportError("boom"))
        except Exception as e:  # pragma: no cover - should not happen
            with lock:
                errors.append(e)

    threads = [threading.Thread(target=worker) for _ in range(32)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    with _WARNED_LOCK:
        assert len(_WARNED_MODULES) == 1 and "mod" in _WARNED_MODULES
    assert _IMPORT_STATE.last_prune > 0.0

