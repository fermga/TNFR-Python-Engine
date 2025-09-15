"""Pruebas de recent_glyph."""

import time

import pytest

import tnfr.glyph_history as glyph_history
from tnfr.constants import get_aliases

ALIAS_EPI_KIND = get_aliases("EPI_KIND")


def _make_node(history, current=None, window=10):
    nd = {}
    for g in history:
        glyph_history.push_glyph(nd, g, window)
    if current is not None:
        nd[ALIAS_EPI_KIND[0]] = current
    return nd


def test_recent_glyph_window_one():
    nd = _make_node(["Y"], current="X")
    assert not glyph_history.recent_glyph(nd, "X", window=1)
    assert glyph_history.recent_glyph(nd, "Y", window=1)


def test_recent_glyph_window_zero():
    nd = _make_node(["A", "B"], current="B")
    assert not glyph_history.recent_glyph(nd, "B", window=0)


def test_recent_glyph_window_zero_does_not_create_history():
    nd = {}
    assert not glyph_history.recent_glyph(nd, "B", window=0)
    assert "glyph_history" not in nd


def test_recent_glyph_window_negative():
    nd = _make_node(["A", "B"], current="B")
    with pytest.raises(ValueError):
        glyph_history.recent_glyph(nd, "B", window=-1)


def test_recent_glyph_history_lookup():
    nd = _make_node(["A", "B"], current="C")
    assert glyph_history.recent_glyph(nd, "B", window=2)
    assert glyph_history.recent_glyph(nd, "A", window=2)
    assert glyph_history.recent_glyph(nd, "A", window=3)


def test_recent_glyph_discards_non_iterable_history():
    nd = {"glyph_history": 1}  # type: ignore[assignment]
    assert not glyph_history.recent_glyph(nd, "A", window=1)
    assert list(nd["glyph_history"]) == []


@pytest.mark.slow
def test_recent_glyph_benchmark():
    nd = _make_node([str(i) for i in range(1000)], window=1000)
    start = time.perf_counter()
    for _ in range(1000):
        glyph_history.recent_glyph(nd, "999", window=1000)
    duration = time.perf_counter() - start
    assert duration < 0.1
