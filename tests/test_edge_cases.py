import pytest
import networkx as nx
from tnfr.node import NodoTNFR
from tnfr.operators import op_EN
from tnfr.types import Glyph

from tnfr.dynamics import default_compute_delta_nfr, update_epi_via_nodal_equation


def test_empty_graph_handling():
    G = nx.Graph()
    default_compute_delta_nfr(G)
    update_epi_via_nodal_equation(G)  # should not raise


def test_sigma_vector_global_empty_graph():
    G = nx.Graph()
    from tnfr.sense import sigma_vector_global

    sv = sigma_vector_global(G)
    assert sv == {"x": 0.0, "y": 0.0, "mag": 0.0, "angle": 0.0, "n": 0}


def test_update_epi_invalid_dt():
    G = nx.Graph()
    G.add_node(1)
    with pytest.raises(ValueError):
        update_epi_via_nodal_equation(G, dt=-0.1)
    with pytest.raises(TypeError):
        update_epi_via_nodal_equation(G, dt="bad")


def test_dnfr_weights_normalization():
    G = nx.Graph()
    G.graph["DNFR_WEIGHTS"] = {"phase": -1, "epi": -1, "vf": -1}
    default_compute_delta_nfr(G)
    weights = G.graph["_DNFR_META"]["weights_norm"]
    cache = G.graph.get("_dnfr_weights")
    assert pytest.approx(weights["phase"], rel=1e-6) == 1/3
    assert pytest.approx(weights["epi"], rel=1e-6) == 1/3
    assert pytest.approx(weights["vf"], rel=1e-6) == 1/3
    assert cache == weights


def test_op_en_sets_epi_kind_on_isolated_node():
    node = NodoTNFR(EPI=1.0)
    op_EN(node)
    assert node.EPI == 1.0
    assert node.epi_kind == Glyph.EN.value


def test_aplicar_glifo_invalid_glifo_raises_and_logs():
    node = NodoTNFR()
    node.graph["history"] = {}
    with pytest.raises(ValueError):
        node.aplicar_glifo("NO_EXISTE")
    events = node.graph["history"].get("events")
    assert events and events[-1][0] == "warn"
    assert "glifo desconocido" in events[-1][1]["msg"]
