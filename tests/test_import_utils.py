import importlib
import sys
import types

import pytest

import tnfr.import_utils as import_utils
from tnfr.import_utils import _IMPORT_STATE, cached_import, prune_failed_imports


pytestmark = pytest.mark.usefixtures("reset_cached_import")


# -- Attribute and fallback handling ---------------------------------------------------------


def test_cached_import_attribute_and_fallback(monkeypatch, reset_cached_import):
    reset_cached_import()
    fake_mod = types.SimpleNamespace(value=5)
    monkeypatch.setitem(sys.modules, "fake_mod", fake_mod)
    assert cached_import("fake_mod", attr="value") == 5
    reset_cached_import()
    monkeypatch.delitem(sys.modules, "fake_mod")
    assert cached_import("fake_mod", attr="value", fallback=1) == 1


def test_cached_import_handles_distinct_fallbacks(monkeypatch, reset_cached_import):
    def fake_import(_name):
        raise ImportError("boom")

    monkeypatch.setattr(importlib, "import_module", fake_import)
    reset_cached_import()
    fb1: list[str] = []
    fb2: dict[str, int] = {}
    assert cached_import("fake_mod", fallback=fb1) is fb1
    assert cached_import("fake_mod", fallback=fb2) is fb2


# -- Cache behavior and locking -------------------------------------------------------------


def test_cached_import_uses_cache(monkeypatch, reset_cached_import):
    reset_cached_import()
    calls = {"n": 0}

    def fake_import(_name):
        calls["n"] += 1
        return types.SimpleNamespace()

    monkeypatch.setattr(importlib, "import_module", fake_import)
    cached_import("fake_mod")
    cached_import("fake_mod")
    assert calls["n"] == 1


# -- Failure recovery and logging -----------------------------------------------------------


def test_cache_clear_and_prune_reset_all(monkeypatch, reset_cached_import):
    reset_cached_import()

    def fake_import(_name):
        raise ImportError("boom")

    monkeypatch.setattr(importlib, "import_module", fake_import)
    cached_import("fake_mod")
    with _IMPORT_STATE.lock:
        assert "fake_mod" in _IMPORT_STATE.failed
        assert "fake_mod" in _IMPORT_STATE.warned

    prune_failed_imports()
    with _IMPORT_STATE.lock:
        assert not _IMPORT_STATE.failed
        assert not _IMPORT_STATE.warned

    calls = {"n": 0}

    def success_import(_name):
        calls["n"] += 1
        return types.SimpleNamespace()

    monkeypatch.setattr(importlib, "import_module", success_import)
    cached_import.cache_clear()
    cached_import("fake_mod")
    cached_import("fake_mod")
    assert calls["n"] == 1
    cached_import.cache_clear()
    cached_import("fake_mod")
    assert calls["n"] == 2


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
