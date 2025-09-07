import types
import importlib

from tnfr.import_utils import optional_import, _FAILED_IMPORTS


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
    assert "fake_mod" in _FAILED_IMPORTS
    optional_import.cache_clear()
    result = optional_import("fake_mod")
    assert result is not None
    assert "fake_mod" not in _FAILED_IMPORTS
