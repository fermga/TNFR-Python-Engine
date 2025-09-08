"""Pruebas de ``AliasAccessor.get`` con valores por defecto."""

from tnfr.alias import AliasAccessor


def test_get_attr_default_none_returns_none():
    d = {}
    acc = AliasAccessor(int)
    result = acc.get(d, ("x",), default=None)
    assert result is None
    assert d == {}
