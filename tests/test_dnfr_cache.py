"""Pruebas de dnfr cache."""

import pytest
import networkx as nx

from tnfr.dynamics import default_compute_delta_nfr
from tnfr.constants import THETA_PRIMARY, EPI_PRIMARY, VF_PRIMARY
from tnfr.helpers.cache import increment_edge_version, cached_nodes_and_A


def _counting_trig(monkeypatch):
    import math

    cos_calls = {"n": 0}
    sin_calls = {"n": 0}
    orig_cos = math.cos
    orig_sin = math.sin

    def cos_wrapper(x):
        cos_calls["n"] += 1
        return orig_cos(x)

    def sin_wrapper(x):
        sin_calls["n"] += 1
        return orig_sin(x)

    monkeypatch.setattr(math, "cos", cos_wrapper)
    monkeypatch.setattr(math, "sin", sin_wrapper)
    return cos_calls, sin_calls


def _setup_graph():
    G = nx.path_graph(3)
    for n in G.nodes:
        G.nodes[n][THETA_PRIMARY] = 0.1 * (n + 1)
        G.nodes[n][EPI_PRIMARY] = 0.2 * (n + 1)
        G.nodes[n][VF_PRIMARY] = 0.3 * (n + 1)
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


def test_prepare_dnfr_data_reuses_cache(monkeypatch):
    cos_calls, sin_calls = _counting_trig(monkeypatch)
    G = _setup_graph()
    default_compute_delta_nfr(G)

    cos_first = cos_calls["n"]
    sin_first = sin_calls["n"]

    # Subsequent call without modifications should reuse cached trig values
    default_compute_delta_nfr(G)
    assert cos_calls["n"] == cos_first
    assert sin_calls["n"] == sin_first
