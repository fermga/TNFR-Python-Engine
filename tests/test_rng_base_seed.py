import networkx as nx
from tnfr.rng import base_seed, set_cache_maxsize, clear_rng_cache


def test_base_seed_returns_value(graph_canon):
    G = graph_canon()
    G.graph["RANDOM_SEED"] = 123
    assert base_seed(G) == 123


def test_base_seed_defaults_to_zero(graph_canon):
    G = graph_canon()
    assert base_seed(G) == 0


def test_cache_clear_no_fail_when_cache_disabled():
    import tnfr.rng as rng_module

    old_size = rng_module._CACHE_MAXSIZE
    set_cache_maxsize(0)
    try:
        assert clear_rng_cache() is None
    finally:
        set_cache_maxsize(old_size)
