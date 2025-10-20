import networkx as nx

from tnfr.node import NodoNX


def test_all_nodes_returns_wrappers(graph_canon):
    graph = graph_canon()
    graph.add_nodes_from([0, 1])

    a = NodoNX(graph, 0)
    nodes_from_a = tuple(a.all_nodes())

    assert {node.n for node in nodes_from_a} == {0, 1}
    assert all(isinstance(node, NodoNX) for node in nodes_from_a)


def test_all_nodes_respects_cached_list():
    graph = nx.Graph()
    graph.add_nodes_from([0, 1])
    graph.graph["_all_nodes"] = ("custom",)

    node = NodoNX(graph, 0)

    assert node.all_nodes() == ("custom",)
