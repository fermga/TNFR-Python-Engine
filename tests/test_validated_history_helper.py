from collections import deque

from tnfr.glyph_history import _validated_history


def test_validated_history_skips_zero():
    nd: dict[str, object] = {}
    w, hist = _validated_history(nd, 0)
    assert w == 0
    assert hist is None
    assert "glyph_history" not in nd


def test_validated_history_creates_zero_when_requested():
    nd: dict[str, object] = {}
    w, hist = _validated_history(nd, 0, create_zero=True)
    assert w == 0
    assert isinstance(hist, deque)
    assert hist.maxlen == 0
    assert "glyph_history" in nd


def test_validated_history_positive_window():
    nd: dict[str, object] = {}
    w, hist = _validated_history(nd, 2)
    assert w == 2
    assert isinstance(hist, deque)
    assert hist.maxlen == 2
    hist.append("A")
    assert list(nd["glyph_history"]) == ["A"]
