import types
import importlib
import logging
from cachetools import TTLCache

import tnfr.import_utils as import_utils
from tnfr.import_utils import (
    cached_import,
    prune_failed_imports,
    _IMPORT_STATE,
    clear_cached_imports,
)


def test_cached_import_clears_failures(monkeypatch):
    calls = {"n": 0}

    def fake_import(name):
        calls["n"] += 1
        if calls["n"] == 1:
            raise ImportError("boom")
        return types.SimpleNamespace(value=1)

    monkeypatch.setattr(importlib, "import_module", fake_import)
    clear_cached_imports()
    assert cached_import("fake_mod") is None
    assert "fake_mod" in _IMPORT_STATE
    clear_cached_imports()
    result = cached_import("fake_mod")
    assert result is not None
    assert "fake_mod" not in _IMPORT_STATE


def test_warns_once_then_debug(monkeypatch, caplog):
    def fake_import(name):
        raise ImportError("boom")

    monkeypatch.setattr(importlib, "import_module", fake_import)

    stacklevels: list[int] = []

    def fake_warn(msg, category=None, stacklevel=1):
        stacklevels.append(stacklevel)

    monkeypatch.setattr(import_utils.warnings, "warn", fake_warn)
    clear_cached_imports()
    with caplog.at_level(logging.DEBUG, logger=import_utils.logger.name):
        cached_import("fake_mod", "attr1")
        cached_import("fake_mod", "attr2")
    clear_cached_imports()
    records = [
        r.levelno for r in caplog.records if r.name == import_utils.logger.name
    ]
    assert records == [logging.DEBUG]
    assert stacklevels == [2]


def test_cached_import_removes_entry_on_success(monkeypatch):
    calls = {"n": 0}

    def fake_import(name):
        calls["n"] += 1
        if calls["n"] == 1:
            raise ImportError("boom")
        return types.SimpleNamespace(value=1)

    monkeypatch.setattr(importlib, "import_module", fake_import)
    clear_cached_imports()
    assert cached_import("fake_mod") is None
    assert "fake_mod" in _IMPORT_STATE
    cached_import.cache_clear()  # retry without clearing failure registry
    result = cached_import("fake_mod")
    assert result is not None
    assert "fake_mod" not in _IMPORT_STATE


def test_cached_import_handles_distinct_fallbacks(monkeypatch):
    def fake_import(name):
        raise ImportError("boom")

    monkeypatch.setattr(importlib, "import_module", fake_import)
    clear_cached_imports()
    fb1: list[str] = []
    fb2: dict[str, int] = {}
    assert (cached_import("fake_mod") or fb1) is fb1
    assert (cached_import("fake_mod") or fb2) is fb2


def test_record_prunes_expired_entries(monkeypatch):
    state = import_utils._IMPORT_STATE
    with state.lock:
        monkeypatch.setattr(
            import_utils._IMPORT_STATE,
            "failed",
            TTLCache(state.limit, 10, timer=lambda: import_utils.time.monotonic()),
        )
    times = iter([0.0, 11.0])
    monkeypatch.setattr(
        import_utils.time, "monotonic", lambda: next(times, 11.0)
    )
    with state.lock:
        state.record("old")
        state.record("new")
    assert "old" not in state.failed
    assert "new" in state.failed


def test_prune_failed_imports(monkeypatch):
    state = import_utils._IMPORT_STATE
    with state.lock:
        monkeypatch.setattr(
            import_utils._IMPORT_STATE,
            "failed",
            TTLCache(state.limit, 10, timer=lambda: import_utils.time.monotonic()),
        )
    times = iter([0.0, 20.0, 20.0])
    monkeypatch.setattr(import_utils.time, "monotonic", lambda: next(times))
    with state.lock:
        state.record("stale")
    prune_failed_imports()
    assert "stale" not in state.failed


def test_failure_log_bounded_without_frequent_prune(monkeypatch):
    state = import_utils._IMPORT_STATE
    with state.lock:
        monkeypatch.setattr(
            import_utils._IMPORT_STATE,
            "failed",
            TTLCache(3, state.max_age, timer=lambda: import_utils.time.monotonic()),
        )
    monkeypatch.setattr(import_utils, "_FAILED_IMPORT_PRUNE_INTERVAL", 10.0)
    monkeypatch.setattr(import_utils._IMPORT_STATE, "last_prune", 0.0)
    monkeypatch.setattr(import_utils.time, "monotonic", lambda: 1.0)

    calls = {"n": 0}

    def fake_prune() -> None:
        calls["n"] += 1

    monkeypatch.setattr(import_utils, "prune_failed_imports", fake_prune)

    def fake_import(name):
        raise ImportError("boom")

    monkeypatch.setattr(importlib, "import_module", fake_import)
    clear_cached_imports()
    for i in range(10):
        cached_import(f"fake_mod{i}")
    assert calls["n"] == 0
    assert len(state.failed) <= 3
