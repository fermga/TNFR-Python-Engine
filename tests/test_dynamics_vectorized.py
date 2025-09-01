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


def _run(G, vectorized):
    G.graph["vectorized_dnfr"] = vectorized
    default_compute_delta_nfr(G)
    return [get_attr(G.nodes[n], ALIAS_DNFR, 0.0) for n in G.nodes]


def test_vectorized_equals_loop():
    pytest.importorskip("numpy")
    G1 = _setup_graph()
    G2 = _setup_graph()
    dnfr_loop = _run(G1, False)
    dnfr_vec = _run(G2, True)
    assert dnfr_loop == pytest.approx(dnfr_vec)
