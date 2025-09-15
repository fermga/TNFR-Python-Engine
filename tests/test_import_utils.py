import importlib
import sys
import types

from cachetools import TTLCache
import tnfr.import_utils as import_utils
from tnfr.import_utils import (
    cached_import,
    prune_failed_imports,
    _IMPORT_CACHE,
    _IMPORT_STATE,
    _WARNED_STATE,
)


def reset() -> None:
    cached_import.cache_clear()
    prune_failed_imports()


def test_cached_import_attribute_and_fallback(monkeypatch):
    reset()
    fake_mod = types.SimpleNamespace(value=5)
    monkeypatch.setitem(sys.modules, "fake_mod", fake_mod)
    assert cached_import("fake_mod", attr="value") == 5
    reset()
    monkeypatch.delitem(sys.modules, "fake_mod")
    assert cached_import("fake_mod", attr="value", fallback=1) == 1


def test_cached_import_uses_cache(monkeypatch):
    reset()
    calls = {"n": 0}

    def fake_import(_name):
        calls["n"] += 1
        return types.SimpleNamespace()

    monkeypatch.setattr(importlib, "import_module", fake_import)
    cached_import("fake_mod")
    cached_import("fake_mod")
    assert calls["n"] == 1


def test_cached_import_uses_provided_lock(monkeypatch):
    reset()
    calls = {"lock": 0}
    orig_lock = import_utils.threading.Lock

    def fake_lock():
        calls["lock"] += 1
        return orig_lock()

    monkeypatch.setattr(import_utils.threading, "Lock", fake_lock)
    cache = TTLCache(16, 1)
    lock = orig_lock()
    fake_mod = types.ModuleType("fake_mod")
    monkeypatch.setitem(sys.modules, "fake_mod", fake_mod)
    cached_import("fake_mod", cache=cache, lock=lock)
    assert calls["lock"] == 0


def test_cached_import_uses_shared_lock_when_missing(monkeypatch):
    reset()
    calls = {"lock": 0}
    orig_lock = import_utils.threading.Lock

    def fake_lock():
        calls["lock"] += 1
        return orig_lock()

    monkeypatch.setattr(import_utils.threading, "Lock", fake_lock)
    cache = TTLCache(16, 1)
    fake_mod = types.ModuleType("fake_mod")
    monkeypatch.setitem(sys.modules, "fake_mod", fake_mod)
    cached_import("fake_mod", cache=cache)
    assert calls["lock"] == 0


def test_cache_clear_and_prune_reset_all(monkeypatch):
    reset()

    def fake_import(_name):
        raise ImportError("boom")

    monkeypatch.setattr(importlib, "import_module", fake_import)
    cached_import("fake_mod")
    with _IMPORT_STATE.lock, _WARNED_STATE.lock:
        assert "fake_mod" in _IMPORT_STATE.failed
        assert "fake_mod" in _WARNED_STATE.failed
    assert _IMPORT_CACHE
    cached_import.cache_clear()
    monkeypatch.setattr(import_utils.time, "monotonic", lambda: 1e9)
    prune_failed_imports()
    with _IMPORT_STATE.lock, _WARNED_STATE.lock:
        assert not _IMPORT_STATE.failed
        assert not _WARNED_STATE.failed
    assert not _IMPORT_CACHE
