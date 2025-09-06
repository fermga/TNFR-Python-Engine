"""Pruebas de alias get default."""

from tnfr.alias import alias_get


def test_alias_get_default_none_returns_none():
    d = {}
    result = alias_get(d, ["x"], int, default=None)
    assert result is None
    assert d == {}
