"""Pruebas de history maxlen."""

from collections import deque

import pytest

from tnfr.constants import attach_defaults
from tnfr.glyph_history import ensure_history, HistoryDict


def test_history_maxlen_and_cleanup(graph_canon):
    G = graph_canon()
    G.add_node(0)
    attach_defaults(G)
    G.graph["HISTORY_MAXLEN"] = 2

    hist = ensure_history(G)
    hist.setdefault("a", []).append(1)
    hist.setdefault("b", []).append(2)
    hist.setdefault("c", []).append(3)

    # trigger cleanup
    ensure_history(G)
    assert len(hist) == 2

    series = hist.setdefault("series", [])
    series.extend([1, 2, 3])
    assert isinstance(series, deque)
    assert series.maxlen == 2
    assert series == deque([2, 3], maxlen=2)


def test_history_least_used_removed(graph_canon):
    G = graph_canon()
    G.add_node(0)
    attach_defaults(G)
    G.graph["HISTORY_MAXLEN"] = 2

    hist = ensure_history(G)
    hist.setdefault("a", [])  # length 0 but will be accessed
    hist.setdefault("b", []).append(1)
    hist.setdefault("c", []).append(1)
    # use "a" several times
    _ = hist.get_increment("a")
    _ = hist.get_increment("a")

    # trigger cleanup
    ensure_history(G)
    assert len(hist) == 2
    assert "a" in hist


def test_history_trim_uses_pop_least_used(graph_canon):
    G = graph_canon()
    G.add_node(0)
    attach_defaults(G)
    G.graph["HISTORY_MAXLEN"] = 2

    hist = ensure_history(G)
    for key in ["a", "b", "c", "d"]:
        hist.setdefault(key, []).append(1)
    _ = hist.get_increment("a")
    _ = hist.get_increment("a")
    _ = hist.get_increment("b")

    ensure_history(G)
    assert set(hist.keys()) == {"a", "b"}


def test_history_maxlen_override_respected(graph_canon):
    G = graph_canon()
    G.add_node(0)
    attach_defaults(G)

    hist = ensure_history(G)
    assert not isinstance(hist, HistoryDict)

    G.graph["HISTORY_MAXLEN"] = 3
    hist = ensure_history(G)
    assert isinstance(hist, HistoryDict)
    assert hist._maxlen == 3


def test_history_not_trimmed_when_equal_maxlen(graph_canon):
    G = graph_canon()
    G.add_node(0)
    attach_defaults(G)
    G.graph["HISTORY_MAXLEN"] = 2

    hist = ensure_history(G)
    hist.setdefault("a", []).append(1)
    hist.setdefault("b", []).append(2)

    ensure_history(G)
    assert len(hist) == 2
    assert set(hist.keys()) == {"a", "b"}


def test_history_not_trimmed_when_below_maxlen(graph_canon):
    G = graph_canon()
    G.add_node(0)
    attach_defaults(G)
    G.graph["HISTORY_MAXLEN"] = 2

    hist = ensure_history(G)
    hist.setdefault("a", []).append(1)

    ensure_history(G)
    assert len(hist) == 1
    assert "a" in hist


def test_history_negative_maxlen_raises(graph_canon):
    G = graph_canon()
    G.add_node(0)
    attach_defaults(G)
    G.graph["HISTORY_MAXLEN"] = -1
    with pytest.raises(ValueError):
        ensure_history(G)
