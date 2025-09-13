import importlib
import sys
import types

import pytest
from tnfr.import_utils import cached_import, optional_import, prune_failed_imports


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

    def fake_import(name):
        calls["n"] += 1
        return types.SimpleNamespace()

    monkeypatch.setattr(importlib, "import_module", fake_import)
    cached_import("fake_mod")
    cached_import("fake_mod")
    assert calls["n"] == 1


def test_optional_import_wrapper(monkeypatch):
    reset()
    fallback = object()
    with pytest.warns(DeprecationWarning):
        assert optional_import("fake_mod", fallback) is fallback
    fake_mod = types.ModuleType("fake_mod")
    monkeypatch.setitem(sys.modules, "fake_mod", fake_mod)
    reset()
    with pytest.warns(DeprecationWarning):
        assert optional_import("fake_mod") is fake_mod
