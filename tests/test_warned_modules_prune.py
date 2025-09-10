import warnings
import time

import tnfr.import_utils as import_utils
from tnfr.import_utils import (
    _warn_failure,
    _WARNED_MODULES,
    _WARNED_LOCK,
    _FAILED_IMPORT_MAX_AGE,
)


def test_warned_modules_pruning(monkeypatch):
    with _WARNED_LOCK:
        _WARNED_MODULES.clear()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _warn_failure("mod", None, ImportError("boom"))
    with _WARNED_LOCK:
        assert "mod" in _WARNED_MODULES
        _WARNED_MODULES["mod"] = time.monotonic() - (_FAILED_IMPORT_MAX_AGE + 1)
    monkeypatch.setattr(import_utils._IMPORT_STATE, "last_prune", 0.0)
    monkeypatch.setattr(import_utils, "_FAILED_IMPORT_PRUNE_INTERVAL", 0.0)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _warn_failure("mod", None, ImportError("boom"))
        assert len(w) == 1
    with _WARNED_LOCK:
        assert "mod" in _WARNED_MODULES
