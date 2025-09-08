from tnfr.alias import AliasAccessor, _validate_aliases


def test_get_attr_uses_cache():
    d = {"b": "1"}
    _validate_aliases.cache_clear()
    acc = AliasAccessor(int)
    acc.get(d, ("a", "b"))
    info1 = _validate_aliases.cache_info()
    acc.get(d, ("a", "b"))
    info2 = _validate_aliases.cache_info()
    assert info2.hits == info1.hits + 1
    assert info2.misses == info1.misses


def test_set_attr_uses_cache():
    d = {}
    _validate_aliases.cache_clear()
    acc = AliasAccessor(int)
    acc.set(d, ("x", "y"), "5")
    info1 = _validate_aliases.cache_info()
    acc.set(d, ("x", "y"), "6")
    info2 = _validate_aliases.cache_info()
    assert info2.hits == info1.hits + 1
    assert info2.misses == info1.misses
