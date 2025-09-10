"""Deterministic RNG helpers."""

from __future__ import annotations

import random
import hashlib
import struct
import threading
from typing import Any, Callable

from cachetools import LRUCache, cached
from .constants import DEFAULTS
from .helpers.cache import get_graph

MASK64 = 0xFFFFFFFFFFFFFFFF

_RNG_LOCK = threading.Lock()
_CACHE_MAXSIZE = int(DEFAULTS.get("JITTER_CACHE_SIZE", 128))


def seed_hash(seed_int: int, key_int: int) -> int:
    """Return a 64-bit hash derived from ``seed_int`` and ``key_int``."""
    seed_bytes = struct.pack(
        ">QQ",
        seed_int & MASK64,
        key_int & MASK64,
    )
    return int.from_bytes(
        hashlib.blake2b(seed_bytes, digest_size=8).digest(), "big"
    )


def _make_cache(size: int) -> Callable[[int, int], int]:
    if size > 0:
        cache = LRUCache(maxsize=max(1, size))
        return cached(cache=cache, lock=_RNG_LOCK)(seed_hash)
    return seed_hash


_seed_hash_cached = _make_cache(_CACHE_MAXSIZE)


def _seed_hash_for(seed_int: int, key_int: int) -> int:
    """Return a seed hash for ``seed_int`` and ``key_int``.

    Uses the cached hash when caching is enabled.
    """

    if _CACHE_MAXSIZE <= 0:
        return seed_hash(seed_int, key_int)
    return _seed_hash_cached(seed_int, key_int)


def make_rng(seed: int, key: int) -> random.Random:
    """Return a ``random.Random`` for ``seed`` and ``key``."""
    seed_int = int(seed)
    key_int = int(key)
    return random.Random(_seed_hash_for(seed_int, key_int))


def clear_rng_cache() -> None:
    """Clear cached seed hashes."""
    if _CACHE_MAXSIZE <= 0:
        return
    _seed_hash_cached.cache_clear()


def cache_enabled() -> bool:
    """Return ``True`` if RNG caching is enabled."""
    return _CACHE_MAXSIZE > 0


def base_seed(G: Any) -> int:
    """Return base RNG seed stored in ``G.graph``."""
    graph = get_graph(G)
    return int(graph.get("RANDOM_SEED", 0))


def _rng_for_step(seed: int, step: int) -> random.Random:
    """Return deterministic RNG for a simulation ``step``."""

    return make_rng(seed, step)


def set_cache_maxsize(size: int) -> None:
    """Update RNG cache maximum size.

    ``size`` must be a non-negative integer; ``0`` disables caching.
    If caching is disabled, ``clear_rng_cache`` has no effect.
    """

    global _CACHE_MAXSIZE, _seed_hash_cached
    new_size = int(size)
    if new_size < 0:
        raise ValueError("size must be non-negative")
    with _RNG_LOCK:
        _CACHE_MAXSIZE = new_size
        _seed_hash_cached = _make_cache(new_size)


__all__ = (
    "seed_hash",
    "make_rng",
    "set_cache_maxsize",
    "base_seed",
    "cache_enabled",
    "clear_rng_cache",
)
