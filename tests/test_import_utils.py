import sys
import types

from tnfr.import_utils import clear_cached_imports, cached_import


def test_cached_import_success_and_failure(monkeypatch):
    clear_cached_imports()
    fallback = object()
    # module missing -> fallback returned
    assert (cached_import("fake_mod") or fallback) is fallback
    # insert fake module
    fake_mod = types.ModuleType("fake_mod")
    monkeypatch.setitem(sys.modules, "fake_mod", fake_mod)
    # previously failed import succeeds after module becomes available
    assert cached_import("fake_mod") is fake_mod
    # clearing cache does not change successful import
    clear_cached_imports()
    assert cached_import("fake_mod") is fake_mod
