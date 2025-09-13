"""Combinatorial tests for `_resolve_edge_ops`."""

import networkx as nx
import pytest

from tnfr.node import _resolve_edge_ops, EdgeStrategy, _EDGE_OPS


def dummy_exists(*_):
    return False


def dummy_set(*_):
    return None


@pytest.mark.parametrize(
    "graph,strategy,use_cb,expected",
    [
        (nx.Graph(), None, False, EdgeStrategy.NX),
        (nx.Graph(), EdgeStrategy.NX, False, EdgeStrategy.NX),
        (nx.Graph(), EdgeStrategy.TNFR, False, EdgeStrategy.TNFR),
        (nx.Graph(), None, True, EdgeStrategy.NX),
        (nx.Graph(), EdgeStrategy.NX, True, EdgeStrategy.NX),
        (nx.Graph(), EdgeStrategy.TNFR, True, EdgeStrategy.TNFR),
        ({}, None, False, EdgeStrategy.TNFR),
        ({}, EdgeStrategy.TNFR, False, EdgeStrategy.TNFR),
        ({}, EdgeStrategy.NX, False, EdgeStrategy.NX),
        ({}, None, True, EdgeStrategy.TNFR),
        ({}, EdgeStrategy.TNFR, True, EdgeStrategy.TNFR),
        ({}, EdgeStrategy.NX, True, EdgeStrategy.NX),
    ],
)
def test_resolve_edge_ops_combinations(graph, strategy, use_cb, expected):
    exists_cb = dummy_exists if use_cb else None
    set_cb = dummy_set if use_cb else None
    exists_fn, set_fn, resolved = _resolve_edge_ops(graph, strategy, exists_cb, set_cb)
    assert resolved is expected
    if use_cb:
        assert exists_fn is dummy_exists
        assert set_fn is dummy_set
    else:
        assert (exists_fn, set_fn) == _EDGE_OPS[expected]
