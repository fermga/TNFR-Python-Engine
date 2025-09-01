"""Pruebas de node all nodes."""
from tnfr.node import NodoTNFR


def test_all_nodes_returns_full_list():
    a = NodoTNFR()
    b = NodoTNFR()
    graph = {"_all_nodes": [a, b]}
    a.graph = graph
    b.graph = graph

    assert set(a.all_nodes()) == {a, b}
    assert set(b.all_nodes()) == {a, b}

