import importlib
import types
from cachetools import TTLCache

import tnfr.import_utils as import_utils
from tnfr.import_utils import cached_import, prune_failed_imports, _IMPORT_STATE


def reset() -> None:
    """Clear caches and failure records."""
    cached_import.cache_clear()
    prune_failed_imports()


def test_cached_import_clears_failures(monkeypatch):
    calls = {"n": 0}

    def fake_import(_name):
        calls["n"] += 1
        if calls["n"] == 1:
            raise ImportError("boom")
        return types.SimpleNamespace(value=1)

    monkeypatch.setattr(importlib, "import_module", fake_import)
    reset()
    assert cached_import("fake_mod") is None
    assert "fake_mod" in _IMPORT_STATE
    reset()
    result = cached_import("fake_mod")
    assert result is not None
    assert "fake_mod" not in _IMPORT_STATE


def test_warns_once_then_debug(monkeypatch):
    def fake_import(_name):
        raise ImportError("boom")

    monkeypatch.setattr(importlib, "import_module", fake_import)

    stacklevels: list[int] = []

    def fake_warn(_msg, _category=None, stacklevel=1):
        stacklevels.append(stacklevel)

    monkeypatch.setattr(import_utils.warnings, "warn", fake_warn)
    reset()
    cached_import("fake_mod", attr="attr1")
    cached_import("fake_mod", attr="attr2")
    reset()
    assert stacklevels == [2]


def test_cached_import_handles_distinct_fallbacks(monkeypatch):
    def fake_import(_name):
        raise ImportError("boom")

    monkeypatch.setattr(importlib, "import_module", fake_import)
    reset()
    fb1: list[str] = []
    fb2: dict[str, int] = {}
    assert cached_import("fake_mod", fallback=fb1) is fb1
    assert cached_import("fake_mod", fallback=fb2) is fb2


def test_cache_ttl(monkeypatch):
    calls = {"n": 0}

    def fake_import(_name):
        calls["n"] += 1
        return types.SimpleNamespace()

    monkeypatch.setattr(importlib, "import_module", fake_import)
    times = [0.0]
    cache = TTLCache(16, 1, timer=lambda: times[0])
    cached_import("fake_mod", cache=cache)
    cached_import("fake_mod", cache=cache)
    assert calls["n"] == 1
    times[0] = 2.0
    cached_import("fake_mod", cache=cache)
    assert calls["n"] == 2


def test_record_prunes_expired_entries(monkeypatch):
    state = import_utils._IMPORT_STATE
    with state.lock:
        monkeypatch.setattr(
            import_utils._IMPORT_STATE,
            "failed",
            TTLCache(state.limit, 10, timer=lambda: import_utils.time.monotonic()),
        )
    now = [0.0]
    monkeypatch.setattr(import_utils.time, "monotonic", lambda: now[0])
    with state.lock:
        state.record("old")
        now[0] = 11.0
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

    def fake_import(_name):
        raise ImportError("boom")

    monkeypatch.setattr(importlib, "import_module", fake_import)
    reset()
    for i in range(10):
        cached_import(f"fake_mod{i}")
    assert calls["n"] == 0
    assert len(state.failed) <= 3

