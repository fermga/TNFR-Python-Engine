import networkx as nx
import pytest

from tnfr.node import add_edge


def test_add_edge_adds_weight_and_version():
    graph = nx.Graph()
    graph.add_nodes_from([1, 2])

    add_edge(graph, 1, 2, 2.5)

    assert graph.has_edge(1, 2)
    assert pytest.approx(graph[1][2]["weight"]) == 2.5
    assert graph.graph.get("_edge_version", 0) >= 1


def test_add_edge_preserves_weight_by_default():
    graph = nx.Graph()
    graph.add_nodes_from([1, 2])
    add_edge(graph, 1, 2, 1.0)

    first_version = graph.graph.get("_edge_version")

    add_edge(graph, 1, 2, 2.0)

    assert pytest.approx(graph[1][2]["weight"]) == 1.0
    assert graph.graph.get("_edge_version") == first_version


def test_add_edge_overwrite_updates_weight():
    graph = nx.Graph()
    graph.add_nodes_from([1, 2])
    add_edge(graph, 1, 2, 1.0)
    first_version = graph.graph.get("_edge_version")

    add_edge(graph, 1, 2, 2.0, overwrite=True)

    assert pytest.approx(graph[1][2]["weight"]) == 2.0
    assert graph.graph.get("_edge_version") == first_version + 1


def test_add_edge_self_loop_ignored():
    graph = nx.Graph()
    graph.add_node(1)

    add_edge(graph, 1, 1, 1.0)

    assert graph.number_of_edges() == 0
    assert "_edge_version" not in graph.graph


def test_add_edge_rejects_non_graph_type():
    with pytest.raises(TypeError):
        add_edge({}, 1, 2, 1.0)
