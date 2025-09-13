"""Tests for add_edge callback validation."""

import pytest
from tnfr.node import add_edge, _validate_callbacks, NodoTNFR, EdgeStrategy


def test_validate_callbacks_requires_callback_pair():
    with pytest.raises(ValueError):
        _validate_callbacks(lambda *_: False, None)
    with pytest.raises(ValueError):
        _validate_callbacks(None, lambda *_: None)


def test_validate_callbacks_requires_callables():
    with pytest.raises(TypeError):
        _validate_callbacks(object(), lambda *_: None)
    with pytest.raises(TypeError):
        _validate_callbacks(lambda *_: False, object())


def test_add_edge_validates_callbacks():
    with pytest.raises(ValueError):
        add_edge({}, 1, 2, 1.0, False, exists_cb=lambda *_: False)
    with pytest.raises(ValueError):
        add_edge({}, 1, 2, 1.0, False, set_cb=lambda *_: None)
    with pytest.raises(TypeError):
        add_edge({}, 1, 2, 1.0, False, exists_cb=object(), set_cb=lambda *_: None)
    with pytest.raises(TypeError):
        add_edge({}, 1, 2, 1.0, False, exists_cb=lambda *_: False, set_cb=object())


def test_add_edge_validates_same_graph_for_tnfr_nodes():
    n1 = NodoTNFR()
    n2 = NodoTNFR()
    with pytest.raises(ValueError):
        add_edge(n1.graph, n1, n2, 1.0, strategy=EdgeStrategy.TNFR)


@pytest.mark.parametrize("strategy", [None, EdgeStrategy.TNFR, "tnfr"])
def test_add_edge_mixed_graph_with_callbacks(strategy):
    g1 = {}
    g2 = {}
    n1 = NodoTNFR(graph=g1)
    n2 = NodoTNFR(graph=g2)
    calls = []

    def exists_cb(*args, **kwargs):
        calls.append("exists")
        return False

    def set_cb(*args, **kwargs):
        calls.append("set")

    with pytest.raises(ValueError):
        add_edge(g1, n1, n2, 1.0, strategy=strategy, exists_cb=exists_cb, set_cb=set_cb)

    assert calls == []


def test_add_edge_checks_weight_before_callbacks():
    with pytest.raises(ValueError, match="Edge weight must be non-negative"):
        add_edge({}, 1, 2, -1.0, exists_cb=lambda *_: False)


def test_add_edge_self_connection_skips_callback_validation():
    # Should not raise even though callbacks are invalid
    add_edge({}, 1, 1, 1.0, exists_cb=lambda *_: False)
