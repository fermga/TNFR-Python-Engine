import math
import networkx as nx
import pytest

from tnfr.node import NodeNX


def _build_nodes():
    graph = nx.Graph()
    graph.add_nodes_from([0, 1])
    return graph, NodeNX(graph, 0), NodeNX(graph, 1)


def test_add_edge_stores_weight():
    graph, a, b = _build_nodes()

    a.add_edge(b, weight=2.5)

    assert a.has_edge(b)
    assert pytest.approx(graph[0][1]["weight"]) == 2.5


def test_add_edge_preserves_weight_by_default():
    graph, a, b = _build_nodes()
    a.add_edge(b, weight=1.0)

    a.add_edge(b, weight=2.0)

    assert pytest.approx(graph[0][1]["weight"]) == 1.0


def test_add_edge_overwrite_allows_update():
    graph, a, b = _build_nodes()
    a.add_edge(b, weight=1.0)

    a.add_edge(b, weight=2.0, overwrite=True)

    assert pytest.approx(graph[0][1]["weight"]) == 2.0


def test_add_edge_rejects_negative_weight():
    graph, a, b = _build_nodes()

    with pytest.raises(ValueError):
        a.add_edge(b, weight=-1.0)

    assert not a.has_edge(b)


def test_add_edge_rejects_non_finite_weight():
    graph, a, b = _build_nodes()

    for weight in (math.nan, math.inf, -math.inf):
        with pytest.raises(ValueError):
            a.add_edge(b, weight=weight)
        assert not a.has_edge(b)
