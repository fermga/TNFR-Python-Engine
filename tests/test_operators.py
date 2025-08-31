from tnfr.node import NodoNX
from tnfr.operators import random_jitter


def test_random_jitter_deterministic_with_and_without_cache(graph_canon):
    G = graph_canon()
    G.add_node(0)
    n0 = NodoNX(G, 0)

    # Without cache
    j1 = random_jitter(n0, 0.5)
    j2 = random_jitter(n0, 0.5)
    assert j1 == j2

    # With explicit cache
    cache = {}
    j3 = random_jitter(n0, 0.5, cache)
    j4 = random_jitter(n0, 0.5, cache)
    assert j3 == j4 == j1
