"""Pruebas del offset de ``NodoTNFR``."""

from tnfr.node import NodoTNFR, NodoNX
from tnfr.helpers.node_cache import ensure_node_offset_map


def test_offset_non_zero_with_mapping(graph_canon):
    G = graph_canon()
    a = NodoTNFR()
    b = NodoTNFR()
    G.add_nodes_from([a, b])
    ensure_node_offset_map(G)
    a.graph = G
    b.graph = G
    assert b.offset() != 0


def test_nodonx_offset_non_zero(graph_canon):
    G = graph_canon()
    G.add_nodes_from([0, 1])
    ensure_node_offset_map(G)
    b = NodoNX(G, 1)
    assert b.offset() != 0
