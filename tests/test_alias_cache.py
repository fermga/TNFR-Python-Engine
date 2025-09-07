import pytest

from tnfr.alias import alias_get, alias_set, _validate_aliases


def test_alias_get_uses_cache():
    d = {"b": "1"}
    _validate_aliases.cache_clear()
    alias_get(d, ("a", "b"), int)
    info1 = _validate_aliases.cache_info()
    alias_get(d, ("a", "b"), int)
    info2 = _validate_aliases.cache_info()
    assert info2.hits == info1.hits + 1
    assert info2.misses == info1.misses


def test_alias_set_uses_cache():
    d = {}
    _validate_aliases.cache_clear()
    alias_set(d, ("x", "y"), int, "5")
    info1 = _validate_aliases.cache_info()
    alias_set(d, ("x", "y"), int, "6")
    info2 = _validate_aliases.cache_info()
    assert info2.hits == info1.hits + 1
    assert info2.misses == info1.misses
