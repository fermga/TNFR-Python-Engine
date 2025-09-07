import types
import importlib
import logging

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


def test_warns_once_then_debug(monkeypatch, caplog):
    def fake_import(name):
        raise ImportError("boom")

    monkeypatch.setattr(importlib, "import_module", fake_import)

    stacklevels: list[int] = []

    def fake_warn(msg, category=None, stacklevel=1):
        stacklevels.append(stacklevel)

    monkeypatch.setattr(import_utils.warnings, "warn", fake_warn)
    optional_import.cache_clear()
    with caplog.at_level(logging.DEBUG, logger=import_utils.logger.name):
        optional_import("fake_mod.attr1")
        optional_import("fake_mod.attr2")
    optional_import.cache_clear()
    records = [r.levelno for r in caplog.records if r.name == import_utils.logger.name]
    assert records == [logging.WARNING, logging.DEBUG]
    assert stacklevels == [2]


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
