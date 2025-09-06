"""Tests for push_glyph window handling."""

import pytest

from tnfr.glyph_history import push_glyph


def test_push_glyph_negative_window():
    nd = {}
    with pytest.raises(ValueError, match="window must be >= 0"):
        push_glyph(nd, "A", window=-1)


def test_push_glyph_zero_window():
    nd = {}
    push_glyph(nd, "A", window=0)
    assert list(nd["glyph_history"]) == []
    push_glyph(nd, "B", window=0)
    assert list(nd["glyph_history"]) == []


def test_push_glyph_positive_window():
    nd = {}
    push_glyph(nd, "A", window=2)
    push_glyph(nd, "B", window=2)
    assert list(nd["glyph_history"]) == ["A", "B"]
    push_glyph(nd, "C", window=2)
    assert list(nd["glyph_history"]) == ["B", "C"]


def test_push_glyph_accepts_list_history():
    nd = {"glyph_history": ["A"]}
    push_glyph(nd, "B", window=2)
    assert list(nd["glyph_history"]) == ["A", "B"]
