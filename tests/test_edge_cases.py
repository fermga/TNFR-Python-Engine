import pytest
import networkx as nx

from tnfr.dynamics import default_compute_delta_nfr, update_epi_via_nodal_equation


def test_empty_graph_handling():
    G = nx.Graph()
    default_compute_delta_nfr(G)
    update_epi_via_nodal_equation(G)  # should not raise


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
    assert pytest.approx(weights["phase"], rel=1e-6) == 1/3
    assert pytest.approx(weights["epi"], rel=1e-6) == 1/3
    assert pytest.approx(weights["vf"], rel=1e-6) == 1/3
