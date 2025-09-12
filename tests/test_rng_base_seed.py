from tnfr.rng import base_seed, clear_rng_cache


def test_base_seed_returns_value(graph_canon):
    G = graph_canon()
    G.graph["RANDOM_SEED"] = 123
    assert base_seed(G) == 123


def test_base_seed_defaults_to_zero(graph_canon):
    G = graph_canon()
    assert base_seed(G) == 0


def test_cache_clear_no_fail_when_cache_disabled():
    import tnfr.rng as rng_module
    old_cache = rng_module._seed_hash_cached
    old_size = rng_module._CACHE_MAXSIZE
    rng_module._CACHE_MAXSIZE = 0
    rng_module._seed_hash_cached = rng_module._make_cache(0)
    try:
        assert clear_rng_cache() is None
    finally:
        rng_module._CACHE_MAXSIZE = old_size
        rng_module._seed_hash_cached = old_cache


def test_set_cache_maxsize_resets_cache():
    import tnfr.rng as rng_module
    old_cache = rng_module._seed_hash_cached
    old_size = rng_module._CACHE_MAXSIZE
    try:
        rng_module._CACHE_MAXSIZE = 2
        rng_module._seed_hash_cached = rng_module._make_cache(2)
        # populate cache and keep a reference to the underlying cache object
        rng_module._seed_hash_cached(1, 1)
        cache = rng_module._seed_hash_cached.cache
        assert cache.currsize == 1

        # resetting cache size should clear old cache
        cache.clear()
        rng_module._CACHE_MAXSIZE = 3
        rng_module._seed_hash_cached = rng_module._make_cache(3)
        assert cache.currsize == 0
    finally:
        rng_module._CACHE_MAXSIZE = old_size
        rng_module._seed_hash_cached = old_cache


def test_set_cache_maxsize_allows_one_entry():
    import tnfr.rng as rng_module
    old_cache = rng_module._seed_hash_cached
    old_size = rng_module._CACHE_MAXSIZE
    try:
        rng_module._CACHE_MAXSIZE = 1
        rng_module._seed_hash_cached = rng_module._make_cache(1)
        rng_module._seed_hash_cached(1, 1)
        cache = rng_module._seed_hash_cached.cache
        assert cache.maxsize == 1
        assert cache.currsize == 1

        # adding a different key should evict the previous entry
        rng_module._seed_hash_cached(2, 2)
        assert cache.currsize == 1
    finally:
        rng_module._CACHE_MAXSIZE = old_size
        rng_module._seed_hash_cached = old_cache
