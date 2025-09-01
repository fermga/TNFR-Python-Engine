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

    # With explicit cache: sequence should be deterministic
    cache = {}
    j3 = random_jitter(n0, 0.5, cache)
    j4 = random_jitter(n0, 0.5, cache)
    assert j3 == j1
    assert j4 != j3

    # Replaying with a fresh cache reproduces the sequence
    cache2 = {}
    seq = [random_jitter(n0, 0.5, cache2) for _ in range(2)]
    assert seq == [j3, j4]
