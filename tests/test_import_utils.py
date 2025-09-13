import sys
import types

from tnfr.import_utils import clear_cached_imports, cached_import


def test_cached_import_success_and_failure(monkeypatch):
    clear_cached_imports()
    # module missing -> None returned
    assert cached_import("fake_mod") is None
    # insert fake module
    fake_mod = types.ModuleType("fake_mod")
    monkeypatch.setitem(sys.modules, "fake_mod", fake_mod)
    # cached failure still returns None
    assert cached_import("fake_mod") is None
    # clearing cache allows successful import
    clear_cached_imports()
    assert cached_import("fake_mod") is fake_mod
