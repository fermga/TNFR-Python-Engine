"""Pruebas de alias_set para conversiones nulas."""

from tnfr.alias import alias_set


def test_alias_set_allows_none_conversion():
    """alias_set debe permitir valores ``None``."""
    d = {}
    alias_set(d, ["x"], lambda v: None, 123)
    assert "x" in d and d["x"] is None
