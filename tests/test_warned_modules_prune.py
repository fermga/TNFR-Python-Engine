import warnings
import time

from tnfr.import_utils import (
    _warn_failure,
    _WARNED_MODULES,
    _WARNED_LOCK,
    _FAILED_IMPORT_MAX_AGE,
    prune_failed_imports,
)


def test_warned_modules_pruning():
    with _WARNED_LOCK:
        _WARNED_MODULES.clear()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _warn_failure("mod", None, ImportError("boom"))
        _warn_failure("mod", None, ImportError("boom"))
    with _WARNED_LOCK:
        assert "mod" in _WARNED_MODULES
        _WARNED_MODULES["mod"] = time.monotonic() - (_FAILED_IMPORT_MAX_AGE + 1)
    prune_failed_imports()
    with _WARNED_LOCK:
        assert "mod" not in _WARNED_MODULES
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _warn_failure("mod", None, ImportError("boom"))
        assert len(w) == 1
