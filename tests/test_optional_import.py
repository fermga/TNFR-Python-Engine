import types
import importlib

import tnfr.import_utils as import_utils
from tnfr.import_utils import (
    optional_import,
    _IMPORT_STATE,
    _optional_import_cache_clear,
)


def test_optional_import_clears_failures(monkeypatch):
    calls = {"n": 0}

    def fake_import(name):
        calls["n"] += 1
        if calls["n"] == 1:
            raise ImportError("boom")
        return types.SimpleNamespace(value=1)

    monkeypatch.setattr(importlib, "import_module", fake_import)
    optional_import.cache_clear()
    assert optional_import("fake_mod") is None
    assert "fake_mod" in _IMPORT_STATE
    optional_import.cache_clear()
    result = optional_import("fake_mod")
    assert result is not None
    assert "fake_mod" not in _IMPORT_STATE


def test_warns_with_stacklevel_2_only_once(monkeypatch):
    def fake_import(name):
        raise ImportError("boom")

    monkeypatch.setattr(importlib, "import_module", fake_import)

    stacklevels = []

    def fake_warn(msg, category=None, stacklevel=1):
        stacklevels.append(stacklevel)

    monkeypatch.setattr(import_utils.warnings, "warn", fake_warn)
    optional_import.cache_clear()
    optional_import("fake_mod.attr1")
    optional_import("fake_mod.attr2")
    optional_import.cache_clear()
    assert stacklevels == [2, 1]


def test_optional_import_removes_entry_on_success(monkeypatch):
    calls = {"n": 0}

    def fake_import(name):
        calls["n"] += 1
        if calls["n"] == 1:
            raise ImportError("boom")
        return types.SimpleNamespace(value=1)

    monkeypatch.setattr(importlib, "import_module", fake_import)
    _optional_import_cache_clear()
    assert optional_import("fake_mod") is None
    assert "fake_mod" in _IMPORT_STATE
    _optional_import_cache_clear()  # retry without clearing failure registry
    result = optional_import("fake_mod")
    assert result is not None
    assert "fake_mod" not in _IMPORT_STATE
