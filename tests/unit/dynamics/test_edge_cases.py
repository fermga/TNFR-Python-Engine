"""Pruebas de edge cases."""

import networkx as nx
import pytest

from tnfr.node import NodoNX
from tnfr.operators import apply_glyph_obj
from tnfr.types import Glyph

from tnfr.dynamics import (
    default_compute_delta_nfr,
    update_epi_via_nodal_equation,
)


def test_empty_graph_handling(graph_canon):
    G = graph_canon()
    default_compute_delta_nfr(G)
    update_epi_via_nodal_equation(G)  # should not raise


def test_sigma_vector_from_graph_empty_graph(graph_canon):
    G = graph_canon()
    from tnfr.sense import sigma_vector_from_graph

    sv = sigma_vector_from_graph(G)
    assert sv == {"x": 0.0, "y": 0.0, "mag": 0.0, "angle": 0.0, "n": 0}


def test_update_epi_invalid_dt(graph_canon):
    G = graph_canon()
    G.add_node(1)
    with pytest.raises(ValueError):
        update_epi_via_nodal_equation(G, dt=-0.1)
    with pytest.raises(TypeError):
        update_epi_via_nodal_equation(G, dt="bad")


def test_dnfr_weights_normalization(graph_canon):
    G = graph_canon()
    G.graph["DNFR_WEIGHTS"] = {"phase": -1, "epi": -1, "vf": -1}
    default_compute_delta_nfr(G)
    weights = G.graph["_DNFR_META"]["weights_norm"]
    cache = G.graph.get("_dnfr_weights")
    assert pytest.approx(weights["phase"], rel=1e-6) == 0.25
    assert pytest.approx(weights["epi"], rel=1e-6) == 0.25
    assert pytest.approx(weights["vf"], rel=1e-6) == 0.25
    assert pytest.approx(weights["topo"], rel=1e-6) == 0.25
    assert cache == weights


def _build_isolated_node(value: float = 0.0) -> NodoNX:
    graph = nx.Graph()
    graph.add_node(0)
    node = NodoNX(graph, 0)
    node.EPI = value
    return node


def test_op_en_sets_epi_kind_on_isolated_node():
    node = _build_isolated_node(1.0)
    apply_glyph_obj(node, "EN")
    assert pytest.approx(node.EPI) == 1.0
    assert node.epi_kind == Glyph.EN.value


def test_apply_glyph_invalid_glyph_raises_and_logs():
    node = _build_isolated_node()
    node.graph["history"] = {}
    with pytest.raises(ValueError):
        apply_glyph_obj(node, "NO_EXISTE")
    events = node.graph["history"].get("events")
    assert events and events[-1][0] == "warn"
    assert "glyph desconocido" in events[-1][1]["msg"]
