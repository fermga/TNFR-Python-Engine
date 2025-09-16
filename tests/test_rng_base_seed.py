from __future__ import annotations

from tnfr.rng import (
    base_seed,
    cache_enabled,
    clear_rng_cache,
    make_rng,
    seed_hash,
    set_cache_maxsize,
)


def test_base_seed_returns_value(graph_canon):
    G = graph_canon()
    G.graph["RANDOM_SEED"] = 123
    assert base_seed(G) == 123


def test_base_seed_defaults_to_zero(graph_canon):
    G = graph_canon()
    assert base_seed(G) == 0


def test_clear_rng_cache_no_fail_when_cache_disabled():
    import tnfr.rng as rng_module

    original_size = rng_module._CACHE_MAXSIZE
    original_locked = rng_module._CACHE_LOCKED
    try:
        set_cache_maxsize(0)
        assert clear_rng_cache() is None
    finally:
        set_cache_maxsize(original_size)
        rng_module._CACHE_LOCKED = original_locked
        clear_rng_cache()


def test_set_cache_maxsize_resets_cache():
    import tnfr.rng as rng_module

    original_size = rng_module._CACHE_MAXSIZE
    original_locked = rng_module._CACHE_LOCKED
    try:
        set_cache_maxsize(2)
        seed_hash.cache_clear()
        seed_hash(1, 1)
        assert len(seed_hash.cache) == 1

        set_cache_maxsize(3)
        assert len(seed_hash.cache) == 0
    finally:
        set_cache_maxsize(original_size)
        rng_module._CACHE_LOCKED = original_locked
        clear_rng_cache()


def test_set_cache_maxsize_limits_entries():
    import tnfr.rng as rng_module

    original_size = rng_module._CACHE_MAXSIZE
    original_locked = rng_module._CACHE_LOCKED
    try:
        set_cache_maxsize(1)
        seed_hash.cache_clear()
        seed_hash(1, 1)
        assert len(seed_hash.cache) == 1

        seed_hash(2, 2)
        assert len(seed_hash.cache) == 1
        assert (2, 2) in seed_hash.cache
        assert (1, 1) not in seed_hash.cache
    finally:
        set_cache_maxsize(original_size)
        rng_module._CACHE_LOCKED = original_locked
        clear_rng_cache()


def test_make_rng_deterministic_without_cache():
    import tnfr.rng as rng_module

    original_size = rng_module._CACHE_MAXSIZE
    original_locked = rng_module._CACHE_LOCKED
    try:
        set_cache_maxsize(0)
        seq1 = [make_rng(42, 7).random() for _ in range(3)]
        seq2 = [make_rng(42, 7).random() for _ in range(3)]
        assert seq1 == seq2
    finally:
        set_cache_maxsize(original_size)
        rng_module._CACHE_LOCKED = original_locked
        clear_rng_cache()


def test_graph_cache_size_applies_when_unlocked(graph_canon):
    import tnfr.rng as rng_module

    original_size = rng_module._CACHE_MAXSIZE
    original_locked = rng_module._CACHE_LOCKED
    try:
        set_cache_maxsize(rng_module._DEFAULT_CACHE_MAXSIZE)
        G = graph_canon()
        G.graph["JITTER_CACHE_SIZE"] = 5
        make_rng(1, 1, G)
        assert seed_hash.cache.maxsize == 5
    finally:
        set_cache_maxsize(original_size)
        rng_module._CACHE_LOCKED = original_locked
        clear_rng_cache()


def test_manual_disable_blocks_graph_override(graph_canon):
    import tnfr.rng as rng_module

    original_size = rng_module._CACHE_MAXSIZE
    original_locked = rng_module._CACHE_LOCKED
    try:
        set_cache_maxsize(0)
        G = graph_canon()
        # cache_enabled consults graph but keeps manual override disabled
        assert cache_enabled(G) is False
        make_rng(1, 2, G)
        assert cache_enabled(G) is False
    finally:
        set_cache_maxsize(original_size)
        rng_module._CACHE_LOCKED = original_locked
        clear_rng_cache()
