import logging
import warnings

from tnfr.import_utils import _warn_failure, _WARNED_MODULES, _WARNED_LOCK, _WARNED_QUEUE


def _clear_warned():
    with _WARNED_LOCK:
        _WARNED_MODULES.clear()
        _WARNED_QUEUE.clear()


def test_warn_failure_warns_only(caplog):
    _clear_warned()
    with caplog.at_level(logging.WARNING):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _warn_failure("mod_warn", None, ImportError("boom"))
        assert len(w) == 1
        assert not caplog.records


def test_warn_failure_logs_only(caplog):
    _clear_warned()
    with caplog.at_level(logging.WARNING):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _warn_failure("mod_log", None, ImportError("boom"), emit="log")
        assert len(w) == 0
        assert len(caplog.records) == 1
        assert "mod_log" in caplog.records[0].message


def test_warn_failure_both(caplog):
    _clear_warned()
    with caplog.at_level(logging.WARNING):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _warn_failure("mod_both", None, ImportError("boom"), emit="both")
        assert len(w) == 1
        assert len(caplog.records) == 1
        assert "mod_both" in caplog.records[0].message
