"""Pruebas de node all nodes."""

from tnfr.node import NodoTNFR, NodoNX


def test_all_nodes_respects_manual_override():
    a = NodoTNFR()
    b = NodoTNFR()
    graph = {"_all_nodes": [a, b]}
    a.graph = graph
    b.graph = graph

    assert set(a.all_nodes()) == {a, b}
    assert set(b.all_nodes()) == {a, b}


def test_all_nodes_tnfr_graph():
    graph = {}
    a = NodoTNFR(graph=graph)
    b = NodoTNFR(graph=graph)

    assert set(a.all_nodes()) == {a, b}
    assert set(b.all_nodes()) == {a, b}


def test_all_nodes_nodonx(graph_canon):
    G = graph_canon()
    G.add_nodes_from([0, 1])
    a = NodoNX(G, 0)
    b = NodoNX(G, 1)

    nodes_from_a = tuple(a.all_nodes())
    nodes_from_b = tuple(b.all_nodes())

    assert {n.n for n in nodes_from_a} == {0, 1}
    assert {n.n for n in nodes_from_b} == {0, 1}
    assert all(isinstance(n, NodoNX) for n in nodes_from_a)
