"""Unit test for ``set_attr_generic`` ensuring ``None`` conversions are accepted."""

from tnfr.alias import set_attr_generic


def test_set_attr_allows_none_conversion():
    """``set_attr_generic`` must allow ``None`` values."""
    d = {}
    set_attr_generic(d, ("x",), 123, conv=lambda v: None)
    assert "x" in d and d["x"] is None
