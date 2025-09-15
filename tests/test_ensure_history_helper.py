from collections import deque

from tnfr.glyph_history import _ensure_history


def test_ensure_history_skips_zero():
    nd: dict[str, object] = {}
    w, hist = _ensure_history(nd, 0)
    assert w == 0
    assert hist is None
    assert "glyph_history" not in nd


def test_ensure_history_creates_zero_when_requested():
    nd: dict[str, object] = {}
    w, hist = _ensure_history(nd, 0, create_zero=True)
    assert w == 0
    assert isinstance(hist, deque)
    assert hist.maxlen == 0
    assert "glyph_history" in nd


def test_ensure_history_positive_window():
    nd: dict[str, object] = {}
    w, hist = _ensure_history(nd, 2)
    assert w == 2
    assert isinstance(hist, deque)
    assert hist.maxlen == 2
    hist.append("A")
    assert list(nd["glyph_history"]) == ["A"]


def test_ensure_history_discards_string_input():
    nd: dict[str, object] = {"glyph_history": "ABC"}
    _, hist = _ensure_history(nd, 2)
    assert isinstance(hist, deque)
    assert list(hist) == []


def test_ensure_history_accepts_iterable_input():
    nd: dict[str, object] = {"glyph_history": ["A", "B"]}
    _, hist = _ensure_history(nd, 2)
    assert list(hist) == ["A", "B"]
