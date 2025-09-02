"""Pruebas de dnfr precompute."""
import pytest
import networkx as nx

from tnfr.dynamics import _prepare_dnfr_data, _compute_dnfr_numpy, _compute_dnfr_loops
from tnfr.constants import ALIAS_THETA, ALIAS_EPI, ALIAS_VF, ALIAS_DNFR
from tnfr.helpers import get_attr


def _setup_graph():
    G = nx.path_graph(5)
    for n in G.nodes:
        G.nodes[n][ALIAS_THETA] = 0.1 * (n + 1)
        G.nodes[n][ALIAS_EPI] = 0.2 * (n + 1)
        G.nodes[n][ALIAS_VF] = 0.3 * (n + 1)
    G.graph["DNFR_WEIGHTS"] = {"phase": 0.4, "epi": 0.3, "vf": 0.2, "topo": 0.1}
    return G


def test_strategies_share_precomputed_data():
    pytest.importorskip("numpy")
    G = _setup_graph()
    data = _prepare_dnfr_data(G)
    _compute_dnfr_loops(G, data)
    dnfr_loop = [get_attr(G.nodes[n], ALIAS_DNFR, 0.0) for n in G.nodes]
    expected = [0.1, -0.05, 0.0, -0.05, 0.1]
    assert dnfr_loop == pytest.approx(expected)
    _compute_dnfr_numpy(G, data)
    dnfr_vec = [get_attr(G.nodes[n], ALIAS_DNFR, 0.0) for n in G.nodes]
    assert dnfr_vec == pytest.approx(expected)
    assert dnfr_loop == pytest.approx(dnfr_vec)
