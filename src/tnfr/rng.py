"""Deterministic RNG helpers."""
from __future__ import annotations

import hashlib
import random
import struct
import threading
from typing import MutableMapping, Tuple

from .constants import DEFAULTS
from cachetools import LRUCache

_RNG_LOCK = threading.Lock()
_CACHE_MAXSIZE = int(DEFAULTS.get("JITTER_CACHE_SIZE", 128))
_RNG_CACHE: MutableMapping[Tuple[int, int], random.Random] = LRUCache(
    maxsize=max(1, _CACHE_MAXSIZE)
)


def make_rng(seed: int, key: int) -> random.Random:
    seed_bytes = struct.pack(
        ">QQ",
        int(seed) & 0xFFFFFFFFFFFFFFFF,
        int(key) & 0xFFFFFFFFFFFFFFFF,
    )
    seed_int = int.from_bytes(
        hashlib.blake2b(seed_bytes, digest_size=8).digest(), "big"
    )
    return random.Random(seed_int)


def get_rng(seed: int, key: int) -> random.Random:
    """Return a cached ``random.Random`` for ``(seed, key)``."""
    k = (int(seed), int(key))
    with _RNG_LOCK:
        if _CACHE_MAXSIZE <= 0 and not isinstance(_RNG_CACHE, dict):
            return make_rng(seed, key)
        try:
            return _RNG_CACHE[k]
        except KeyError:
            rng = make_rng(seed, key)
            _RNG_CACHE[k] = rng
            return rng


def _cache_clear() -> None:
    with _RNG_LOCK:
        _RNG_CACHE.clear()


get_rng.cache_clear = _cache_clear  # type: ignore[attr-defined]


def set_cache_maxsize(size: int) -> None:
    """Update RNG cache maximum size."""

    global _CACHE_MAXSIZE, _RNG_CACHE
    new_size = int(size)
    with _RNG_LOCK:
        _CACHE_MAXSIZE = new_size
        if new_size > 0:
            _RNG_CACHE = LRUCache(maxsize=new_size)
        else:
            _RNG_CACHE = {}

__all__ = ["get_rng", "make_rng", "set_cache_maxsize"]
