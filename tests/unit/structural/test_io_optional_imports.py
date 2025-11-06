import importlib
import types

import pytest

from tests.utils import clear_test_module
from tnfr.utils import (
    LazyImportProxy,
    cached_import,
)  # noqa: F401 - used in tests via fresh_io


@pytest.fixture
def fresh_io():
    cached_import.cache_clear()
    # Re-import module instead of reload to handle test isolation
    clear_test_module("tnfr.utils.io")
    import tnfr.utils.io as module  # noqa: F401 - testing module reload

    yield module
    cached_import.cache_clear()
    # Cleanup
    clear_test_module("tnfr.utils.io")
    import tnfr.utils.io  # Re-import for next test  # noqa: F401


def test_io_optional_imports_are_lazy_proxies(fresh_io):
    # Must import LazyImportProxy inside test to get fresh class reference.
    # When test_version_resolution clears sys.modules, the module-level import
    # becomes stale - it references the OLD class, while fresh_io contains
    # instances of the NEW class created after re-import.
    from tnfr.utils import LazyImportProxy

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
