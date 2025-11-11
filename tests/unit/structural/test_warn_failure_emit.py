import logging
import warnings

import tnfr.utils.init as utils_init
from tnfr.utils.init import IMPORT_LOG, _warn_failure


def _clear_warned():
    IMPORT_LOG.clear()


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


def test_warn_failure_uses_emit_map():
    called: list[str] = []

    def fake_warn(msg: str) -> None:
        called.append(msg)

    _clear_warned()
    original = utils_init.EMIT_MAP["warn"]
    utils_init.EMIT_MAP["warn"] = fake_warn
    try:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _warn_failure("mod_emit_map", None, ImportError("boom"))
        assert not w
        assert called == ["Failed to import module 'mod_emit_map': boom"]
    finally:
        utils_init.EMIT_MAP["warn"] = original
        _clear_warned()
