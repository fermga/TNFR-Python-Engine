from tnfr.alias import AliasAccessor, _alias_cache


def test_get_attr_uses_cache():
    d = {"b": "1"}
    acc = AliasAccessor(int)
    acc.get(d, ("a", "b"))  # build cache
    _alias_cache.cache_clear()
    acc.get(d, ("a", "b"))
    info1 = _alias_cache.cache_info()
    acc.get(d, ("a", "b"))
    info2 = _alias_cache.cache_info()
    assert info2.hits == info1.hits + 1
    assert info2.misses == info1.misses


def test_set_attr_uses_cache():
    d = {}
    acc = AliasAccessor(int)
    acc.set(d, ("x", "y"), "5")  # build cache
    _alias_cache.cache_clear()
    acc.set(d, ("x", "y"), "5")
    info1 = _alias_cache.cache_info()
    acc.set(d, ("x", "y"), "6")
    info2 = _alias_cache.cache_info()
    assert info2.hits == info1.hits + 1
    assert info2.misses == info1.misses
