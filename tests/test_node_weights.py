"""Pruebas de node weights."""

import math
import pytest
import networkx as nx
from tnfr.node import NodoTNFR, NodoNX


def test_add_edge_stores_weight():
    a = NodoTNFR()
    b = NodoTNFR()
    a.add_edge(b, weight=2.5)
    assert a.has_edge(b)
    assert b.has_edge(a)
    assert math.isclose(a.edge_weight(b), 2.5)
    assert math.isclose(b.edge_weight(a), 2.5)


def test_missing_edge_returns_zero():
    a = NodoTNFR()
    b = NodoTNFR()
    assert not a.has_edge(b)
    assert a.edge_weight(b) == 0.0


def test_add_edge_preserves_weight_by_default():
    a = NodoTNFR()
    b = NodoTNFR()
    a.add_edge(b, weight=1.0)
    a.add_edge(b, weight=2.0)
    assert math.isclose(a.edge_weight(b), 1.0)
    assert math.isclose(b.edge_weight(a), 1.0)


def test_add_edge_overwrite():
    a = NodoTNFR()
    b = NodoTNFR()
    a.add_edge(b, weight=1.0)
    a.add_edge(b, weight=2.0, overwrite=True)
    assert math.isclose(a.edge_weight(b), 2.0)
    assert math.isclose(b.edge_weight(a), 2.0)


def test_add_edge_rejects_negative_weight():
    a = NodoTNFR()
    b = NodoTNFR()
    with pytest.raises(ValueError):
        a.add_edge(b, weight=-1.0)
    assert not a.has_edge(b)
    assert not b.has_edge(a)


def test_add_edge_rejects_negative_weight_nx():
    G = nx.Graph()
    G.add_node(0)
    G.add_node(1)
    a = NodoNX(G, 0)
    b = NodoNX(G, 1)
    with pytest.raises(ValueError):
        a.add_edge(b, weight=-0.5)
    assert not a.has_edge(b)


def test_add_edge_rejects_negative_weight_existing_edge():
    a = NodoTNFR()
    b = NodoTNFR()
    a.add_edge(b, weight=1.0)
    with pytest.raises(ValueError):
        a.add_edge(b, weight=-2.0)
    assert math.isclose(a.edge_weight(b), 1.0)
    assert math.isclose(b.edge_weight(a), 1.0)


def test_add_edge_rejects_negative_weight_existing_edge_nx():
    G = nx.Graph()
    G.add_node(0)
    G.add_node(1)
    a = NodoNX(G, 0)
    b = NodoNX(G, 1)
    a.add_edge(b, weight=1.0)
    with pytest.raises(ValueError):
        a.add_edge(b, weight=-2.0)
    assert math.isclose(G[0][1]["weight"], 1.0)
