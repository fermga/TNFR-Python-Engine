"""Pruebas de dnfr cache."""

import pytest
import networkx as nx

from tnfr.dynamics import default_compute_delta_nfr
from tnfr.constants import ALIAS_THETA, ALIAS_EPI, ALIAS_VF
from tnfr.helpers import increment_edge_version, cached_nodes_and_A


def _setup_graph():
    G = nx.path_graph(3)
    for n in G.nodes:
        G.nodes[n][ALIAS_THETA[0]] = 0.1 * (n + 1)
        G.nodes[n][ALIAS_EPI[0]] = 0.2 * (n + 1)
        G.nodes[n][ALIAS_VF[0]] = 0.3 * (n + 1)
    G.graph["DNFR_WEIGHTS"] = {
        "phase": 1.0,
        "epi": 0.0,
        "vf": 0.0,
        "topo": 0.0,
    }
    return G


@pytest.mark.parametrize("vectorized", [False, True])
def test_cache_invalidated_on_graph_change(vectorized):
    if vectorized:
        pytest.importorskip("numpy")

    G = _setup_graph()
    G.graph["vectorized_dnfr"] = vectorized
    default_compute_delta_nfr(G, cache_size=2)
    nodes1, _ = cached_nodes_and_A(G, cache_size=2)

    G.add_edge(2, 3)  # Cambia número de nodos y aristas
    increment_edge_version(G)
    G.graph["vectorized_dnfr"] = vectorized
    default_compute_delta_nfr(G, cache_size=2)
    nodes2, _ = cached_nodes_and_A(G, cache_size=2)

    assert len(nodes2) == 4
    assert nodes1 is not nodes2

    G.add_edge(3, 4)
    increment_edge_version(G)
    G.graph["vectorized_dnfr"] = vectorized
    default_compute_delta_nfr(G, cache_size=2)
    nodes3, _ = cached_nodes_and_A(G, cache_size=2)
    assert nodes3 is not nodes2


def test_cache_is_per_graph():
    G1 = _setup_graph()
    G2 = _setup_graph()
    default_compute_delta_nfr(G1)
    default_compute_delta_nfr(G2)
    nodes1, _ = cached_nodes_and_A(G1)
    nodes2, _ = cached_nodes_and_A(G2)
    assert nodes1 is not nodes2


def test_cache_invalidated_on_node_rename():
    G = _setup_graph()
    default_compute_delta_nfr(G)
    assert set(G.nodes) == {0, 1, 2}

    nx.relabel_nodes(G, {2: 9}, copy=False)
    default_compute_delta_nfr(G)

    assert set(G.nodes) == {0, 1, 9}
