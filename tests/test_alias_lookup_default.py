from tnfr.helpers import alias_lookup


def test_alias_lookup_default_none_returns_none():
    d = {}
    result = alias_lookup(d, ["x"], int, default=None)
    assert result is None
    assert d == {}

