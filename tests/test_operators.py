"""Pruebas de operators."""
from tnfr.node import NodoNX
from tnfr.operators import random_jitter, clear_rng_cache
import tnfr.operators as operators
from tnfr.operators import op_UM
from tnfr.constants import attach_defaults
import networkx as nx
import pytest


def test_random_jitter_deterministic_with_and_without_cache(graph_canon):
    clear_rng_cache()
    G = graph_canon()
    G.add_node(0)
    n0 = NodoNX(G, 0)

    # Without explicit cache: uses global LRU cache and advances sequence
    j1 = random_jitter(n0, 0.5)
    j2 = random_jitter(n0, 0.5)
    assert j1 != j2

    # Clearing the LRU cache reproduces the deterministic sequence
    clear_rng_cache()
    j3 = random_jitter(n0, 0.5)
    j4 = random_jitter(n0, 0.5)
    assert [j3, j4] == [j1, j2]

    # With explicit cache: reproduces deterministic sequence independently
    cache = {}
    j5 = random_jitter(n0, 0.5, cache)
    j6 = random_jitter(n0, 0.5, cache)
    assert j5 == j1
    assert j6 == j2

    # Replaying with a fresh cache reproduces the sequence
    cache2 = {}
    seq = [random_jitter(n0, 0.5, cache2) for _ in range(2)]
    assert seq == [j5, j6]


def test_random_jitter_zero_amplitude(graph_canon):
    G = graph_canon()
    G.add_node(0)
    n0 = NodoNX(G, 0)
    assert random_jitter(n0, 0.0) == 0.0


def test_random_jitter_negative_amplitude(graph_canon):
    G = graph_canon()
    G.add_node(0)
    n0 = NodoNX(G, 0)
    with pytest.raises(ValueError):
        random_jitter(n0, -0.1)


def test_rng_cache_lru_purge(graph_canon):
    clear_rng_cache()
    G = graph_canon()
    G.graph["JITTER_CACHE_SIZE"] = 2
    G.add_nodes_from(range(3))
    n0, n1, n2 = [NodoNX(G, i) for i in range(3)]

    j0 = random_jitter(n0, 0.5)
    j1 = random_jitter(n1, 0.5)
    j2 = random_jitter(n2, 0.5)

    j1b = random_jitter(n1, 0.5)
    assert j1b != j1

    j0b = random_jitter(n0, 0.5)
    assert j0b == j0

    cache = operators._rng_cache[G]
    seed = int(G.graph.get("RANDOM_SEED", 0))
    p0 = (seed, operators._node_offset(G, 0))
    p1 = (seed, operators._node_offset(G, 1))
    p2 = (seed, operators._node_offset(G, 2))
    assert p0 in cache and p1 in cache and p2 not in cache


def test_rng_cache_expires_after_graph_gc(graph_canon):
    import gc

    clear_rng_cache()
    G = graph_canon()
    G.add_node(0)
    n0 = NodoNX(G, 0)
    random_jitter(n0, 0.5)
    assert len(operators._rng_cache) == 1

    del n0
    del G
    gc.collect()

    assert len(operators._rng_cache) == 0


def test_um_candidate_subset_proximity():
    G = nx.Graph()
    attach_defaults(G)
    for i, th in enumerate([0.0, 0.1, 0.2, 1.0]):
        G.add_node(i, **{"θ": th, "EPI": 0.5, "Si": 0.5})

    G.graph["UM_FUNCTIONAL_LINKS"] = True
    G.graph["UM_COMPAT_THRESHOLD"] = -1.0
    G.graph["UM_CANDIDATE_COUNT"] = 2
    G.graph["UM_CANDIDATE_MODE"] = "proximity"

    op_UM(G, 0)

    assert G.has_edge(0, 1)
    assert G.has_edge(0, 2)
    assert not G.has_edge(0, 3)
