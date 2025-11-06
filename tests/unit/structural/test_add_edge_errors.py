import networkx as nx
import pytest

from tnfr.node import add_edge


def test_add_edge_negative_weight_checked_before_graph_support():
    with pytest.raises(ValueError, match="non-negative"):
        add_edge({}, 1, 2, -1.0)


def test_add_edge_self_loop_skips_graph_validation():
    # Should not raise even though the container is not a networkx graph
    add_edge({}, 1, 1, 1.0)


def test_add_edge_requires_networkx_graph():
    with pytest.raises(TypeError):
        add_edge(object(), 1, 2, 1.0)


def test_add_edge_rejects_non_finite_weight():
    graph = nx.Graph()
    graph.add_nodes_from([1, 2])
    with pytest.raises(ValueError, match="finite number"):
        add_edge(graph, 1, 2, float("nan"))
