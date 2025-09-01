"""Pruebas de dnfr cache."""
import pytest
import networkx as nx

from tnfr.dynamics import default_compute_delta_nfr
from tnfr.constants import ALIAS_THETA, ALIAS_EPI, ALIAS_VF, ALIAS_DNFR
from tnfr.helpers import get_attr


def _setup_graph():
    G = nx.path_graph(3)
    for n in G.nodes:
        G.nodes[n][ALIAS_THETA[0]] = 0.1 * (n + 1)
        G.nodes[n][ALIAS_EPI[0]] = 0.2 * (n + 1)
        G.nodes[n][ALIAS_VF[0]] = 0.3 * (n + 1)
    G.graph["DNFR_WEIGHTS"] = {"phase": 1.0, "epi": 0.0, "vf": 0.0, "topo": 0.0}
    return G


@pytest.mark.parametrize("vectorized", [False, True])
def test_cache_invalidated_on_graph_change(vectorized):
    if vectorized:
        pytest.importorskip("numpy")

    G = _setup_graph()
    G.graph["vectorized_dnfr"] = vectorized
    default_compute_delta_nfr(G, cache_size=2)
    assert len(G.graph.get("_dnfr_cache", {})) == 1
    before = [get_attr(G.nodes[n], ALIAS_DNFR, 0.0) for n in G.nodes]

    G.add_edge(2, 3)  # Cambia número de nodos y aristas
    G.graph["vectorized_dnfr"] = vectorized
    default_compute_delta_nfr(G, cache_size=2)
    assert len(G.graph.get("_dnfr_cache", {})) == 2
    after = [get_attr(G.nodes[n], ALIAS_DNFR, 0.0) for n in G.nodes]

    assert len(after) == 4
    assert before[2] != pytest.approx(after[2])

    G.add_edge(3, 4)
    G.graph["vectorized_dnfr"] = vectorized
    default_compute_delta_nfr(G, cache_size=2)
    assert len(G.graph.get("_dnfr_cache", {})) == 2


def test_cache_is_per_graph():
    G1 = _setup_graph()
    G2 = _setup_graph()
    default_compute_delta_nfr(G1)
    default_compute_delta_nfr(G2)
    assert G1.graph["_dnfr_cache"] is not G2.graph["_dnfr_cache"]
    assert len(G1.graph["_dnfr_cache"]) == 1
    assert len(G2.graph["_dnfr_cache"]) == 1

