import logging
import warnings

from tnfr.import_utils import _warn_failure, _WARNED_MODULES, _WARNED_LOCK


def _clear_warned():
    with _WARNED_LOCK:
        _WARNED_MODULES.clear()


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
    from tnfr import import_utils

    called: list[str] = []

    def fake_warn(msg: str) -> None:
        called.append(msg)

    _clear_warned()
    original = import_utils.EMIT_MAP["warn"]
    import_utils.EMIT_MAP["warn"] = fake_warn
    try:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _warn_failure("mod_emit_map", None, ImportError("boom"))
        assert not w
        assert called == ["Failed to import module 'mod_emit_map': boom"]
    finally:
        import_utils.EMIT_MAP["warn"] = original
        _clear_warned()
