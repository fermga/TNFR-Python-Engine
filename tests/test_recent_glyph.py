"""Pruebas de recent_glyph."""

from tnfr.glyph_history import push_glyph, recent_glyph
from tnfr.constants import ALIAS_EPI_KIND
import time
import pytest


def _make_node(history, current=None, window=10):
    nd = {}
    for g in history:
        push_glyph(nd, g, window)
    if current is not None:
        nd[ALIAS_EPI_KIND[0]] = current
    return nd


def test_recent_glyph_window_one():
    nd = _make_node(["Y"], current="X")
    assert not recent_glyph(nd, "X", window=1)
    assert recent_glyph(nd, "Y", window=1)


def test_recent_glyph_window_zero():
    nd = _make_node(["A", "B"], current="B")
    assert not recent_glyph(nd, "B", window=0)


def test_recent_glyph_window_negative():
    nd = _make_node(["A", "B"], current="B")
    assert not recent_glyph(nd, "B", window=-1)


def test_recent_glyph_history_lookup():
    nd = _make_node(["A", "B"], current="C")
    assert recent_glyph(nd, "B", window=2)
    assert recent_glyph(nd, "A", window=2)
    assert recent_glyph(nd, "A", window=3)


@pytest.mark.slow
def test_recent_glyph_benchmark():
    nd = _make_node([str(i) for i in range(1000)], window=1000)
    start = time.perf_counter()
    for _ in range(1000):
        recent_glyph(nd, "999", window=1000)
    duration = time.perf_counter() - start
    assert duration < 0.1
