import sys
import types
import importlib

from tnfr.import_utils import (
    clear_optional_import_cache,
    cached_import,
)


def test_cached_import_success_and_failure(monkeypatch):
    clear_optional_import_cache()
    fallback = object()
    # module missing -> fallback returned
    assert cached_import("fake_mod", fallback=fallback) is fallback
    # insert fake module
    fake_mod = types.ModuleType("fake_mod")
    monkeypatch.setitem(sys.modules, "fake_mod", fake_mod)
    # cached failure still returns fallback
    assert cached_import("fake_mod", fallback=fallback) is fallback
    # clearing cache allows successful import
    clear_optional_import_cache()
    assert cached_import("fake_mod") is fake_mod


def test_cached_import_attribute_and_fallback(monkeypatch):
    clear_optional_import_cache()
    fake_mod = types.SimpleNamespace(value=5)
    monkeypatch.setitem(sys.modules, "fake_mod", fake_mod)
    assert cached_import("fake_mod", attr="value") == 5
    clear_optional_import_cache()
    monkeypatch.delitem(sys.modules, "fake_mod")
    assert cached_import("fake_mod", attr="value", fallback=1) == 1


def test_cached_import_uses_cache(monkeypatch):
    clear_optional_import_cache()
    calls = {"n": 0}

    def fake_import(name):
        calls["n"] += 1
        return types.SimpleNamespace()

    monkeypatch.setattr(importlib, "import_module", fake_import)
    cached_import("fake_mod")
    cached_import("fake_mod")
    assert calls["n"] == 1
