"""Pruebas de alias get default."""

from tnfr.helpers import alias_get, _validate_aliases


def test_alias_get_default_none_returns_none():
    d = {}
    aliases = _validate_aliases(("x",))
    result = alias_get(d, aliases, int, default=None)
    assert result is None
    assert d == {}
