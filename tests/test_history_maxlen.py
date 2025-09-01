"""Pruebas de history maxlen."""
from collections import deque

from tnfr.constants import attach_defaults
from tnfr.helpers import ensure_history


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
    assert list(series) == [2, 3]


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
    _ = hist["a"]
    _ = hist["a"]

    # trigger cleanup
    ensure_history(G)
    assert len(hist) == 2
    assert "a" in hist
