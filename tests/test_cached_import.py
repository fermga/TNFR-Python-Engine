import importlib
import types

import pytest
from cachetools import TTLCache

import tnfr.import_utils as import_utils
from tnfr.import_utils import cached_import, prune_failed_imports, _IMPORT_STATE


pytestmark = pytest.mark.usefixtures("reset_cached_import")


def test_cached_import_clears_failures(monkeypatch, reset_cached_import):
    calls = {"n": 0}

    def fake_import(_name):
        calls["n"] += 1
        if calls["n"] == 1:
            raise ImportError("boom")
        return types.SimpleNamespace(value=1)

    monkeypatch.setattr(importlib, "import_module", fake_import)
    reset_cached_import()
    assert cached_import("fake_mod") is None
    with _IMPORT_STATE.lock:
        assert "fake_mod" in _IMPORT_STATE.failed
        assert "fake_mod" in _IMPORT_STATE.warned
    reset_cached_import()
    result = cached_import("fake_mod")
    assert result is not None
    with _IMPORT_STATE.lock:
        assert "fake_mod" not in _IMPORT_STATE.failed
        assert "fake_mod" not in _IMPORT_STATE.warned


def test_warns_once_then_debug(monkeypatch, reset_cached_import):
    def fake_import(_name):
        raise ImportError("boom")

    monkeypatch.setattr(importlib, "import_module", fake_import)

    stacklevels: list[int] = []

    def fake_warn(_msg, _category=None, stacklevel=1):
        stacklevels.append(stacklevel)

    monkeypatch.setattr(import_utils.warnings, "warn", fake_warn)
    reset_cached_import()
    cached_import("fake_mod", attr="attr1")
    cached_import("fake_mod", attr="attr2")
    reset_cached_import()
    assert stacklevels == [2]


def test_cached_import_handles_distinct_fallbacks(monkeypatch, reset_cached_import):
    def fake_import(_name):
        raise ImportError("boom")

    monkeypatch.setattr(importlib, "import_module", fake_import)
    reset_cached_import()
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


def test_failure_log_respects_limit(monkeypatch, reset_cached_import):
    state = import_utils._IMPORT_STATE
    reset_cached_import()
    monkeypatch.setattr(import_utils._IMPORT_STATE, "limit", 3)

    def fake_import(_name):
        raise ImportError("boom")

    monkeypatch.setattr(importlib, "import_module", fake_import)
    for i in range(10):
        cached_import(f"fake_mod{i}")

    with state.lock:
        assert len(state.failed) <= 3

    reset_cached_import()

