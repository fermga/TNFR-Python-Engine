"""Pruebas de ``set_attr`` para conversiones nulas."""

from tnfr.alias import set_attr


def test_set_attr_allows_none_conversion():
    """``set_attr`` debe permitir valores ``None``."""
    d = {}
    set_attr(d, ("x",), 123, conv=lambda v: None)
    assert "x" in d and d["x"] is None
