from tnfr.node import NodoNX
from tnfr.operators import random_jitter, _GLOBAL_JITTER_CACHE
from tnfr.operators import op_UM
from tnfr.constants import attach_defaults
import networkx as nx


def test_random_jitter_deterministic_with_and_without_cache(graph_canon):
    _GLOBAL_JITTER_CACHE.clear()
    G = graph_canon()
    G.add_node(0)
    n0 = NodoNX(G, 0)

    # Without explicit cache: uses global cache and advances sequence
    j1 = random_jitter(n0, 0.5)
    j2 = random_jitter(n0, 0.5)
    assert j1 != j2

    # With explicit cache: reproduces deterministic sequence
    cache = {}
    j3 = random_jitter(n0, 0.5, cache)
    j4 = random_jitter(n0, 0.5, cache)
    assert j3 == j1
    assert j4 == j2

    # Replaying with a fresh cache reproduces the sequence
    cache2 = {}
    seq = [random_jitter(n0, 0.5, cache2) for _ in range(2)]
    assert seq == [j3, j4]


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
