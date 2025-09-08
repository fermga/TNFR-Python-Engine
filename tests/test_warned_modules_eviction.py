import warnings
from tnfr.import_utils import (
    _warn_failure,
    _WARNED_MODULES,
    _WARNED_LIMIT,
    _WARNED_LOCK,
    _WARNED_QUEUE,
)


def test_warned_modules_eviction():
    with _WARNED_LOCK:
        _WARNED_MODULES.clear()
        _WARNED_QUEUE.clear()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i in range(_WARNED_LIMIT + 5):
            _warn_failure(f"mod{i}", None, ImportError("boom"))
    with _WARNED_LOCK:
        assert len(_WARNED_MODULES) == _WARNED_LIMIT
        assert "mod0" not in _WARNED_MODULES
        assert f"mod{_WARNED_LIMIT + 4}" in _WARNED_MODULES
