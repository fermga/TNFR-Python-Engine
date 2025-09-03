"""Pruebas de dynamics vectorized."""

import pytest
import networkx as nx

from tnfr.dynamics import default_compute_delta_nfr
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


@pytest.mark.parametrize("vectorized", [False, True])
def test_default_compute_delta_nfr_paths(vectorized):
    if vectorized:
        pytest.importorskip("numpy")
    G = _setup_graph()
    G.graph["vectorized_dnfr"] = vectorized
    default_compute_delta_nfr(G)
    dnfr = [get_attr(G.nodes[n], ALIAS_DNFR, 0.0) for n in G.nodes]
    assert dnfr == pytest.approx([0.1, -0.05, 0.0, -0.05, 0.1])
