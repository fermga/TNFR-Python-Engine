"""Deterministic RNG helpers."""

from __future__ import annotations

import random
import hashlib
import struct
import threading
from typing import MutableMapping, Any

from cachetools import LRUCache
from .constants import DEFAULTS
from .helpers.cache import get_graph

_RNG_LOCK = threading.Lock()
_CACHE_MAXSIZE = int(DEFAULTS.get("JITTER_CACHE_SIZE", 128))
_RNG_CACHE: MutableMapping[tuple[int, int], int] = LRUCache(
    maxsize=max(1, _CACHE_MAXSIZE)
)


def _seed_hash(seed_int: int, key_int: int) -> int:
    seed_bytes = struct.pack(
        ">QQ",
        seed_int & 0xFFFFFFFFFFFFFFFF,
        key_int & 0xFFFFFFFFFFFFFFFF,
    )
    return int.from_bytes(
        hashlib.blake2b(seed_bytes, digest_size=8).digest(), "big"
    )


def make_rng(seed_int: int, key_int: int) -> random.Random:
    return random.Random(_seed_hash(seed_int, key_int))


def get_rng(seed: int, key: int) -> random.Random:
    """Return a ``random.Random`` for ``(seed, key)`` using a cached seed."""
    seed_int = int(seed)
    key_int = int(key)
    k = (seed_int, key_int)
    with _RNG_LOCK:
        if _CACHE_MAXSIZE <= 0:
            seed_hash = _seed_hash(seed_int, key_int)
        else:
            seed_hash = _RNG_CACHE.get(k)
            if seed_hash is None:
                seed_hash = _seed_hash(seed_int, key_int)
                _RNG_CACHE[k] = seed_hash
        return random.Random(seed_hash)


def _cache_clear() -> None:
    with _RNG_LOCK:
        _RNG_CACHE.clear()


get_rng.cache_clear = _cache_clear  # type: ignore[attr-defined]


def cache_enabled() -> bool:
    """Return ``True`` if RNG caching is enabled."""
    return _CACHE_MAXSIZE > 0


def base_seed(G: Any) -> int:
    """Return base RNG seed stored in ``G.graph``."""
    graph = get_graph(G)
    return int(graph.get("RANDOM_SEED", 0))


def _rng_for_step(seed: int, step: int) -> random.Random:
    """Return deterministic RNG for a simulation ``step``."""

    return get_rng(seed, step)


def set_cache_maxsize(size: int) -> None:
    """Update RNG cache maximum size.

    ``size`` must be a non-negative integer; ``0`` disables caching.
    """

    global _CACHE_MAXSIZE, _RNG_CACHE
    new_size = int(size)
    if new_size < 0:
        raise ValueError("size must be non-negative")
    with _RNG_LOCK:
        _CACHE_MAXSIZE = new_size
        _RNG_CACHE.clear()
        if new_size > 0:
            _RNG_CACHE = LRUCache(maxsize=new_size)
        else:
            _RNG_CACHE = {}


__all__ = ["get_rng", "make_rng", "set_cache_maxsize", "base_seed", "cache_enabled"]
