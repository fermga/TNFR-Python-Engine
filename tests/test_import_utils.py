import sys
import types

from tnfr.import_utils import clear_optional_import_cache, optional_import


def test_optional_import_success_and_failure(monkeypatch):
    clear_optional_import_cache()
    fallback = object()
    # module missing -> fallback returned
    assert optional_import("fake_mod", fallback) is fallback
    # insert fake module
    fake_mod = types.ModuleType("fake_mod")
    monkeypatch.setitem(sys.modules, "fake_mod", fake_mod)
    # cached failure still returns fallback
    assert optional_import("fake_mod", fallback) is fallback
    # clearing cache allows successful import
    clear_optional_import_cache()
    assert optional_import("fake_mod") is fake_mod
