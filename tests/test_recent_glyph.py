"""Pruebas de recent_glyph."""

from tnfr.glyph_history import push_glyph, recent_glyph
from tnfr.constants import ALIAS_EPI_KIND


def _make_node(history, current=None, window=10):
    nd = {}
    for g in history:
        push_glyph(nd, g, window)
    if current is not None:
        nd[ALIAS_EPI_KIND[0]] = current
    return nd


def test_recent_glyph_window_one():
    nd = _make_node(["Y"], current="X")
    assert not recent_glyph(nd, "X", 1)
    assert recent_glyph(nd, "Y", 1)


def test_recent_glyph_history_lookup():
    nd = _make_node(["A", "B"], current="C")
    assert recent_glyph(nd, "B", 2)
    assert not recent_glyph(nd, "A", 2)
    assert recent_glyph(nd, "A", 3)
