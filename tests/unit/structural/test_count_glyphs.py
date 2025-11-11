"""Unit tests for glyph counting helpers across history windows."""

from collections import Counter, deque

import pytest

from tnfr.glyph_history import count_glyphs


def test_count_glyphs_last_only_and_window(graph_canon):
    G = graph_canon()
    G.add_node(0, glyph_history=deque(["A", "B"]))
    G.add_node(1)

    last = count_glyphs(G, last_only=True)
    assert last == Counter({"B": 1})

    recent = count_glyphs(G, window=2)
    assert recent == Counter({"A": 1, "B": 1})


def test_count_glyphs_non_positive_window(graph_canon):
    G = graph_canon()
    G.add_node(0, glyph_history=deque(["A", "B"]))
    G.add_node(1, glyph_history=deque(["C"]))

    assert count_glyphs(G, window=0) == Counter()
    with pytest.raises(ValueError):
        count_glyphs(G, window=-1)
