import random

from tnfr.operators.remesh import _knn_edges


def _setup():
    nodes = list(range(5))
    epi = {i: float(i) for i in nodes}
    return nodes, epi


def test_knn_edges_connects_nearest_neighbours():
    nodes, epi = _setup()
    rnd = random.Random(0)
    edges = _knn_edges(nodes, epi, k_val=2, p_rewire=0.0, rnd=rnd)
    expected = {
        (0, 1),
        (0, 2),
        (1, 2),
        (2, 3),
        (2, 4),
        (3, 4),
    }
    assert edges == expected


def test_knn_edges_rewire_preserves_edge_count():
    nodes, epi = _setup()
    edges_base = _knn_edges(nodes, epi, k_val=2, p_rewire=0.0, rnd=random.Random(0))
    edges_rewired = _knn_edges(nodes, epi, k_val=2, p_rewire=1.0, rnd=random.Random(1))
    assert len(edges_rewired) == len(edges_base)
    assert edges_rewired != edges_base
