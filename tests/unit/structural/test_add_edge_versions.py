import networkx as nx
import pytest

from tnfr.node import add_edge


def test_add_edge_does_not_increment_when_skipping_update():
    graph = nx.Graph()
    graph.add_nodes_from([1, 2])
    add_edge(graph, 1, 2, 1.0)
    version = graph.graph.get("_edge_version")

    add_edge(graph, 1, 2, 3.0)

    assert pytest.approx(graph[1][2]["weight"]) == 1.0
    assert graph.graph.get("_edge_version") == version


def test_add_edge_overwrite_triggers_version_increment():
    graph = nx.Graph()
    graph.add_nodes_from([1, 2])
    add_edge(graph, 1, 2, 1.0)
    version = graph.graph.get("_edge_version")

    add_edge(graph, 1, 2, 2.0, overwrite=True)

    assert pytest.approx(graph[1][2]["weight"]) == 2.0
    assert graph.graph.get("_edge_version") == version + 1
