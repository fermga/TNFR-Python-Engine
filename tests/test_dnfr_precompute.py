"""Pruebas de dnfr precompute."""

import pytest
import networkx as nx

from tnfr.dynamics import (
    _prepare_dnfr_data,
    _compute_dnfr,
)
from tnfr.constants import get_aliases
from tnfr.alias import set_attr, collect_attr, get_attr

ALIAS_THETA = get_aliases("THETA")
ALIAS_EPI = get_aliases("EPI")
ALIAS_VF = get_aliases("VF")
ALIAS_DNFR = get_aliases("DNFR")


def _setup_graph():
    G = nx.path_graph(5)
    for n in G.nodes:
        set_attr(G.nodes[n], ALIAS_THETA, 0.1 * (n + 1))
        set_attr(G.nodes[n], ALIAS_EPI, 0.2 * (n + 1))
        set_attr(G.nodes[n], ALIAS_VF, 0.3 * (n + 1))
    G.graph["DNFR_WEIGHTS"] = {
        "phase": 0.4,
        "epi": 0.3,
        "vf": 0.2,
        "topo": 0.1,
    }
    return G


def test_strategies_share_precomputed_data():
    pytest.importorskip("numpy")
    G = _setup_graph()
    G.graph["vectorized_dnfr"] = True
    data = _prepare_dnfr_data(G)
    _compute_dnfr(G, data, use_numpy=False)
    dnfr_loop = collect_attr(G, G.nodes, ALIAS_DNFR, 0.0)
    for n in G.nodes:
        set_attr(G.nodes[n], ALIAS_DNFR, 0.0)
    _compute_dnfr(G, data, use_numpy=True)
    dnfr_vec = collect_attr(G, G.nodes, ALIAS_DNFR, 0.0)
    assert dnfr_loop == pytest.approx(dnfr_vec)


def test_prepare_dnfr_numpy_vectors_match_aliases():
    pytest.importorskip("numpy")
    G = _setup_graph()
    G.graph["vectorized_dnfr"] = True
    data = _prepare_dnfr_data(G)
    assert data["A"] is not None
    assert data["theta_np"] is not None
    assert data["epi_np"] is not None
    assert data["vf_np"] is not None
    assert data["cos_theta_np"] is not None
    assert data["sin_theta_np"] is not None
    assert data["deg_array"] is not None

    theta_expected = [get_attr(G.nodes[n], ALIAS_THETA, 0.0) for n in G.nodes]
    epi_expected = [get_attr(G.nodes[n], ALIAS_EPI, 0.0) for n in G.nodes]
    vf_expected = [get_attr(G.nodes[n], ALIAS_VF, 0.0) for n in G.nodes]
    deg_expected = [float(G.degree(n)) for n in G.nodes]

    assert data["theta_np"].tolist() == pytest.approx(theta_expected)
    assert data["epi_np"].tolist() == pytest.approx(epi_expected)
    assert data["vf_np"].tolist() == pytest.approx(vf_expected)
    assert data["deg_array"].tolist() == pytest.approx(deg_expected)

    # Ensure we can reuse the prepared data for the vectorised computation
    _compute_dnfr(G, data, use_numpy=True)
    dnfr_vec = collect_attr(G, G.nodes, ALIAS_DNFR, 0.0)
    assert all(isinstance(val, float) for val in dnfr_vec)
