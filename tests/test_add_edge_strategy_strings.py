"""Tests for :func:`add_edge` accepting string strategies."""

import math
import networkx as nx
import pytest

from tnfr.node import add_edge, EdgeStrategy, NodoTNFR


def test_add_edge_string_strategy_nx():
    G_enum = nx.Graph()
    G_enum.add_nodes_from([1, 2])
    add_edge(G_enum, 1, 2, 2.5, strategy=EdgeStrategy.NX)

    G_str = nx.Graph()
    G_str.add_nodes_from([1, 2])
    add_edge(G_str, 1, 2, 2.5, strategy="nx")

    assert list(G_enum.edges(data=True)) == list(G_str.edges(data=True))


def test_add_edge_string_strategy_tnfr():
    g_enum = {}
    a_enum = NodoTNFR(graph=g_enum)
    b_enum = NodoTNFR(graph=g_enum)
    add_edge(g_enum, a_enum, b_enum, 2.5, strategy=EdgeStrategy.TNFR)

    g_str = {}
    a_str = NodoTNFR(graph=g_str)
    b_str = NodoTNFR(graph=g_str)
    add_edge(g_str, a_str, b_str, 2.5, strategy="tnfr")

    assert a_enum.has_edge(b_enum)
    assert b_enum.has_edge(a_enum)
    assert a_str.has_edge(b_str)
    assert b_str.has_edge(a_str)
    assert math.isclose(a_enum._neighbors[b_enum], a_str._neighbors[b_str])
    assert math.isclose(b_enum._neighbors[a_enum], b_str._neighbors[a_str])


def test_add_edge_invalid_string_strategy():
    with pytest.raises(ValueError, match="Invalid edge strategy:"):
        add_edge({}, 1, 2, 1.0, strategy="invalid")
