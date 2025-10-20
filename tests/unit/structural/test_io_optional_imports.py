import importlib
import types

import pytest

from tnfr import io as io_mod
from tnfr.utils import LazyImportProxy, cached_import


@pytest.fixture
def fresh_io():
    cached_import.cache_clear()
    module = importlib.reload(io_mod)
    yield module
    cached_import.cache_clear()
    importlib.reload(io_mod)


def test_io_optional_imports_are_lazy_proxies(fresh_io):
    assert isinstance(fresh_io.tomllib, LazyImportProxy)
    assert isinstance(fresh_io._TOML_LOADS, LazyImportProxy)
    assert isinstance(fresh_io.yaml, LazyImportProxy)
    assert isinstance(fresh_io._YAML_SAFE_LOAD, LazyImportProxy)


def test_yaml_safe_load_proxy_uses_fallback(monkeypatch, fresh_io):
    calls = {"yaml": 0}

    original_import_module = importlib.import_module

    def fake_import(name, package=None):
        if name == "yaml":
            calls["yaml"] += 1
            raise ImportError("missing yaml")
        return original_import_module(name, package)

    monkeypatch.setattr(importlib, "import_module", fake_import)

    loader = fresh_io._YAML_SAFE_LOAD
    with pytest.raises(ImportError):
        loader("value: 1")
    assert calls["yaml"] == 1


def test_toml_loads_proxy_falls_back_to_tomli(monkeypatch, fresh_io):
    original_import_module = importlib.import_module

    tomli_module = types.SimpleNamespace(
        loads=lambda text: {"content": text},
    )

    def fake_import(name, package=None):
        if name == "tomllib":
            raise ImportError("missing tomllib")
        if name == "tomli":
            return tomli_module
        return original_import_module(name, package)

    monkeypatch.setattr(importlib, "import_module", fake_import)

    loader = fresh_io._TOML_LOADS
    assert loader("num = 1") == {"content": "num = 1"}
