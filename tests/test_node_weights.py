import math
from tnfr.node import NodoTNFR


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
