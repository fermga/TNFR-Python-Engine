import gc
import importlib
import sys
import types
import weakref

import pytest

import tnfr.utils as utils_pkg
import tnfr.utils.init as import_utils
from tnfr.utils import cached_import, prune_failed_imports, warm_cached_import


pytestmark = pytest.mark.usefixtures("reset_cached_import")


# -- Package re-export checks ---------------------------------------------------------------


def test_utils_package_shares_import_state(reset_cached_import):
    reset_cached_import()
    assert utils_pkg._IMPORT_STATE is import_utils._IMPORT_STATE
    assert utils_pkg.IMPORT_LOG is import_utils.IMPORT_LOG
    utils_pkg.prune_failed_imports()
    assert utils_pkg._IMPORT_STATE is import_utils._IMPORT_STATE


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


def test_cached_import_failure_is_cached(monkeypatch, reset_cached_import):
    reset_cached_import()
    calls = {"n": 0}

    def fake_import(_name):
        calls["n"] += 1
        raise ImportError("boom")

    monkeypatch.setattr(importlib, "import_module", fake_import)
    cached_import("fake_mod")
    cached_import("fake_mod")
    assert calls["n"] == 1

    prune_failed_imports()
    cached_import("fake_mod")
    assert calls["n"] == 2


# -- Lazy import handling -------------------------------------------------------------------


def test_cached_import_lazy_defers_until_used(monkeypatch, reset_cached_import):
    reset_cached_import()
    calls = {"n": 0}
    module = types.SimpleNamespace(value=42)

    def fake_import(_name):
        calls["n"] += 1
        return module

    monkeypatch.setattr(importlib, "import_module", fake_import)
    proxy = cached_import("fake_mod", lazy=True)
    assert isinstance(proxy, import_utils.LazyImportProxy)
    assert calls["n"] == 0
    assert proxy.value == 42
    assert calls["n"] == 1
    again = cached_import("fake_mod", lazy=True)
    assert again is module
    assert cached_import("fake_mod") is module


def test_cached_import_allows_garbage_collection(monkeypatch, reset_cached_import):
    reset_cached_import()

    class Token:
        pass

    module = types.ModuleType("gc_mod")
    module.target = Token()
    monkeypatch.setitem(sys.modules, "gc_mod", module)

    result = cached_import("gc_mod", attr="target")
    ref = weakref.ref(result)
    assert ref() is module.target

    again = cached_import("gc_mod", attr="target")
    assert again is result

    del module.target
    monkeypatch.delitem(sys.modules, "gc_mod")
    del result, again
    gc.collect()

    assert ref() is None


def test_cached_import_lazy_records_failure_on_use(monkeypatch, reset_cached_import):
    reset_cached_import()
    calls = {"n": 0}

    def fake_import(_name):
        calls["n"] += 1
        raise ImportError("boom")

    monkeypatch.setattr(importlib, "import_module", fake_import)
    proxy = cached_import("fake_mod", lazy=True)
    assert calls["n"] == 0
    assert isinstance(proxy, import_utils.LazyImportProxy)
    assert bool(proxy) is False
    assert calls["n"] == 1
    with import_utils._IMPORT_STATE.lock:
        assert "fake_mod" in import_utils._IMPORT_STATE.failed


def test_cached_import_lazy_honours_fallback(monkeypatch, reset_cached_import):
    reset_cached_import()
    calls = {"n": 0}

    def fake_import(_name):
        calls["n"] += 1
        raise ImportError("boom")

    monkeypatch.setattr(importlib, "import_module", fake_import)
    fallback = object()
    proxy = cached_import("fake_mod", fallback=fallback, lazy=True)
    assert isinstance(proxy, import_utils.LazyImportProxy)
    assert calls["n"] == 0
    assert proxy.resolve() is fallback
    assert calls["n"] == 1
    with import_utils._IMPORT_STATE.lock:
        assert "fake_mod" in import_utils._IMPORT_STATE.failed


# -- Failure recovery and logging -----------------------------------------------------------


def test_cache_clear_and_prune_reset_all(monkeypatch, reset_cached_import):
    reset_cached_import()

    def fake_import(_name):
        raise ImportError("boom")

    monkeypatch.setattr(importlib, "import_module", fake_import)
    cached_import("fake_mod")
    with import_utils._IMPORT_STATE.lock:
        assert "fake_mod" in import_utils._IMPORT_STATE.failed
        assert "fake_mod" in import_utils._IMPORT_STATE.warned

    prune_failed_imports()
    with import_utils._IMPORT_STATE.lock:
        assert not import_utils._IMPORT_STATE.failed
        assert not import_utils._IMPORT_STATE.warned

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
    with import_utils._IMPORT_STATE.lock:
        assert "fake_mod" in import_utils._IMPORT_STATE.failed
        assert "fake_mod" in import_utils._IMPORT_STATE.warned
    reset_cached_import()
    result = cached_import("fake_mod")
    assert result is not None
    with import_utils._IMPORT_STATE.lock:
        assert "fake_mod" not in import_utils._IMPORT_STATE.failed
        assert "fake_mod" not in import_utils._IMPORT_STATE.warned


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
# -- Warm import helper ----------------------------------------------------------------------


def test_warm_cached_import_populates_cache(monkeypatch, reset_cached_import):
    reset_cached_import()
    calls = {"n": 0}
    module = types.SimpleNamespace(token=object())

    def fake_import(_name):
        calls["n"] += 1
        return module

    monkeypatch.setattr(importlib, "import_module", fake_import)

    result = import_utils.warm_cached_import("fake_mod")
    assert result is module
    assert calls["n"] == 1

    cached = cached_import("fake_mod")
    assert cached is module
    assert calls["n"] == 1


def test_warm_cached_import_lazy_defers_until_used(monkeypatch, reset_cached_import):
    reset_cached_import()
    calls = {"n": 0}
    module = types.SimpleNamespace(value=42)

    def fake_import(_name):
        calls["n"] += 1
        return module

    monkeypatch.setattr(importlib, "import_module", fake_import)

    proxy = warm_cached_import("lazy_mod", lazy=True)
    assert isinstance(proxy, import_utils.LazyImportProxy)
    assert calls["n"] == 0

    assert proxy.value == 42
    assert calls["n"] == 1
    assert cached_import("lazy_mod") is module
    assert calls["n"] == 1


def test_warm_cached_import_can_resolve_lazy(monkeypatch, reset_cached_import):
    reset_cached_import()
    calls = {"n": 0}
    module = types.SimpleNamespace(value=99)

    def fake_import(_name):
        calls["n"] += 1
        return module

    monkeypatch.setattr(importlib, "import_module", fake_import)

    result = warm_cached_import("lazy_mod", lazy=True, resolve=True)
    assert result is module
    assert calls["n"] == 1
    assert cached_import("lazy_mod") is module
    assert calls["n"] == 1


def test_warm_cached_import_rejects_resolve_without_lazy(reset_cached_import):
    reset_cached_import()
    with pytest.raises(ValueError):
        warm_cached_import("fake_mod", resolve=True)


def test_warm_cached_import_failure_is_idempotent(monkeypatch, reset_cached_import):
    reset_cached_import()
    calls = {"n": 0}

    def fake_import(_name):
        calls["n"] += 1
        raise ImportError("boom")

    monkeypatch.setattr(importlib, "import_module", fake_import)

    assert import_utils.warm_cached_import("missing_mod") is None
    assert calls["n"] == 1

    with import_utils._IMPORT_STATE.lock:
        assert "missing_mod" in import_utils._IMPORT_STATE.failed

    again = warm_cached_import("missing_mod")
    assert again is None
    assert calls["n"] == 1


def test_warm_cached_import_handles_multiple_specs(monkeypatch, reset_cached_import):
    reset_cached_import()
    modules = {
        "pkg_a": types.SimpleNamespace(value=1),
        "pkg_b": types.SimpleNamespace(value=2),
    }

    def fake_import(name):
        return modules[name]

    monkeypatch.setattr(importlib, "import_module", fake_import)

    results = warm_cached_import("pkg_a", ("pkg_b", "value"))
    assert results == {
        "pkg_a": modules["pkg_a"],
        "pkg_b.value": 2,
    }
