"""Deterministic RNG helpers."""

from __future__ import annotations

import hashlib
import random
import struct
from typing import Any, cast

from cachetools import cached  # type: ignore[import-untyped]

from .constants import DEFAULTS, get_param
from .locking import get_lock
from .types import GraphLike, TNFRGraph
from .utils import (
    ScopedCounterCache,
    _SeedHashCache,
    build_cache_manager,
    get_graph,
)

MASK64 = 0xFFFFFFFFFFFFFFFF

_RNG_LOCK = get_lock("rng")
_DEFAULT_CACHE_MAXSIZE = int(DEFAULTS.get("JITTER_CACHE_SIZE", 128))
_CACHE_MAXSIZE = _DEFAULT_CACHE_MAXSIZE
_CACHE_LOCKED = False

_RNG_CACHE_MANAGER = build_cache_manager(default_capacity=_DEFAULT_CACHE_MAXSIZE)

_seed_hash_cache = _SeedHashCache(
    manager=_RNG_CACHE_MANAGER,
    default_maxsize=_DEFAULT_CACHE_MAXSIZE,
)


def _compute_seed_hash(seed_int: int, key_int: int) -> int:
    seed_bytes = struct.pack(
        ">QQ",
        seed_int & MASK64,
        key_int & MASK64,
    )
    return int.from_bytes(hashlib.blake2b(seed_bytes, digest_size=8).digest(), "big")


@cached(cache=_seed_hash_cache, lock=_RNG_LOCK)
def _cached_seed_hash(seed_int: int, key_int: int) -> int:
    return _compute_seed_hash(seed_int, key_int)


def seed_hash(seed_int: int, key_int: int) -> int:
    """Return a 64-bit hash derived from ``seed_int`` and ``key_int``."""

    if _CACHE_MAXSIZE <= 0 or not _seed_hash_cache.enabled:
        return _compute_seed_hash(seed_int, key_int)
    return _cached_seed_hash(seed_int, key_int)


seed_hash.cache_clear = cast(Any, _cached_seed_hash).cache_clear  # type: ignore[attr-defined]
seed_hash.cache = _seed_hash_cache  # type: ignore[attr-defined]


def _sync_cache_size(G: TNFRGraph | GraphLike | None) -> None:
    """Synchronise cache size with ``G`` when needed."""

    global _CACHE_MAXSIZE
    if G is None or _CACHE_LOCKED:
        return
    size = get_cache_maxsize(G)
    with _RNG_LOCK:
        if size != _seed_hash_cache.maxsize:
            _seed_hash_cache.configure(size)
            _CACHE_MAXSIZE = _seed_hash_cache.maxsize


def make_rng(seed: int, key: int, G: TNFRGraph | GraphLike | None = None) -> random.Random:
    """Create a reproducible RNG instance from seed and key.

    This factory constructs a deterministic :class:`random.Random` generator
    by hashing the seed and key together. The hash result is cached for
    performance when the same (seed, key) pair is requested repeatedly.

    Parameters
    ----------
    seed : int
        Base random seed for the generator. Must be an integer.
    key : int
        Key used to derive a unique hash with the seed. Multiple keys
        allow independent RNG streams from the same base seed.
    G : TNFRGraph | GraphLike | None, optional
        Graph containing JITTER_CACHE_SIZE parameter. When provided, the
        internal cache size is synchronized with the graph configuration.

    Returns
    -------
    random.Random
        Deterministic random number generator seeded with hash(seed, key).

    Notes
    -----
    The same (seed, key) pair always produces the same generator state,
    ensuring reproducibility across TNFR simulations. Cache synchronization
    with ``G`` allows adaptive caching based on simulation requirements.
    """
    _sync_cache_size(G)
    seed_int = int(seed)
    key_int = int(key)
    return random.Random(seed_hash(seed_int, key_int))


def clear_rng_cache() -> None:
    """Clear cached seed hashes."""
    if _seed_hash_cache.maxsize <= 0 or not _seed_hash_cache.enabled:
        return
    seed_hash.cache_clear()  # type: ignore[attr-defined]


def get_cache_maxsize(G: TNFRGraph | GraphLike) -> int:
    """Return RNG cache maximum size for ``G``."""
    return int(get_param(G, "JITTER_CACHE_SIZE"))


def cache_enabled(G: TNFRGraph | GraphLike | None = None) -> bool:
    """Return ``True`` if RNG caching is enabled.

    When ``G`` is provided, the cache size is synchronised with
    ``JITTER_CACHE_SIZE`` stored in ``G``.
    """
    # Only synchronise the cache size with ``G`` when caching is enabled.  This
    # preserves explicit calls to :func:`set_cache_maxsize(0)` which are used in
    # tests to temporarily disable caching regardless of graph defaults.
    if _seed_hash_cache.maxsize > 0:
        _sync_cache_size(G)
    return _seed_hash_cache.maxsize > 0


def base_seed(G: TNFRGraph | GraphLike) -> int:
    """Return base RNG seed stored in ``G.graph``."""
    graph = get_graph(G)
    return int(graph.get("RANDOM_SEED", 0))


def _rng_for_step(seed: int, step: int) -> random.Random:
    """Return deterministic RNG for a simulation ``step``."""

    return make_rng(seed, step)


def set_cache_maxsize(size: int) -> None:
    """Update RNG cache maximum size.

    ``size`` must be a non-negative integer; ``0`` disables caching.
    Changing the cache size resets any cached seed hashes.
    If caching is disabled, ``clear_rng_cache`` has no effect.
    """

    global _CACHE_MAXSIZE, _CACHE_LOCKED
    new_size = int(size)
    if new_size < 0:
        raise ValueError("size must be non-negative")
    with _RNG_LOCK:
        _seed_hash_cache.configure(new_size)
        _CACHE_MAXSIZE = _seed_hash_cache.maxsize
    _CACHE_LOCKED = new_size != _DEFAULT_CACHE_MAXSIZE


__all__ = (
    "seed_hash",
    "make_rng",
    "get_cache_maxsize",
    "set_cache_maxsize",
    "base_seed",
    "cache_enabled",
    "clear_rng_cache",
    "ScopedCounterCache",
)
