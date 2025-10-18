"""Pruebas de dynamics vectorized."""

import pytest
import networkx as nx

from tnfr.dynamics import default_compute_delta_nfr
from tnfr.constants import get_aliases
from tnfr.alias import set_attr, collect_attr

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


@pytest.mark.parametrize("vectorized", [False, True])
def test_default_compute_delta_nfr_paths(vectorized):
    if vectorized:
        pytest.importorskip("numpy")
    G = _setup_graph()
    G.graph["vectorized_dnfr"] = vectorized
    default_compute_delta_nfr(G)
    dnfr = collect_attr(G, G.nodes, ALIAS_DNFR, 0.0)
    assert len(dnfr) == 5


def _build_weighted_graph(factory, n_nodes: int, topo_weight: float):
    G = factory(n_nodes)
    for idx, node in enumerate(G.nodes):
        set_attr(G.nodes[node], ALIAS_THETA, 0.15 * (idx + 1))
        set_attr(G.nodes[node], ALIAS_EPI, 0.05 * (idx + 2))
        set_attr(G.nodes[node], ALIAS_VF, 0.12 * (idx + 3))
    G.graph["DNFR_WEIGHTS"] = {
        "phase": 0.35,
        "epi": 0.25,
        "vf": 0.3,
        "topo": topo_weight,
    }
    return G


@pytest.mark.parametrize("factory", [nx.path_graph, nx.complete_graph])
@pytest.mark.parametrize("topo_weight", [0.0, 0.4])
def test_vectorized_matches_reference(factory, topo_weight):
    np = pytest.importorskip("numpy")
    del np  # only needed to guarantee NumPy availability

    G_list = _build_weighted_graph(factory, 6, topo_weight)
    G_vec = _build_weighted_graph(factory, 6, topo_weight)

    default_compute_delta_nfr(G_list)

    G_vec.graph["vectorized_dnfr"] = True
    default_compute_delta_nfr(G_vec)

    dnfr_list = collect_attr(G_list, G_list.nodes, ALIAS_DNFR, 0.0)
    dnfr_vec = collect_attr(G_vec, G_vec.nodes, ALIAS_DNFR, 0.0)
    assert dnfr_vec == pytest.approx(dnfr_list)
    assert G_vec.graph.get("_DNFR_META") == G_list.graph.get("_DNFR_META")
    assert G_vec.graph.get("_dnfr_hook_name") == G_list.graph.get("_dnfr_hook_name")
