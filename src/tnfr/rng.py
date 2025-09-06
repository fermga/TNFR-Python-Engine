"""Deterministic RNG helpers."""
from __future__ import annotations

import hashlib
import random
import struct
from collections import OrderedDict
from typing import Tuple
import threading

from .constants import DEFAULTS

_RNG_CACHE: OrderedDict[Tuple[int, int], random.Random] = OrderedDict()
_RNG_LOCK = threading.Lock()
_CACHE_MAXSIZE = int(DEFAULTS.get("JITTER_CACHE_SIZE", 128))


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
    """Return a cached ``random.Random`` for ``(seed, key)`` respecting current cache size."""
    k = (int(seed), int(key))
    cache = _RNG_CACHE
    with _RNG_LOCK:
        try:
            rng = cache.pop(k)
        except KeyError:
            rng = make_rng(seed, key)
        cache[k] = rng
        maxsize = _CACHE_MAXSIZE
        while maxsize > 0 and len(cache) > maxsize:
            cache.popitem(last=False)
    return rng


def _cache_clear() -> None:
    with _RNG_LOCK:
        _RNG_CACHE.clear()


get_rng.cache_clear = _cache_clear  # type: ignore[attr-defined]


def set_cache_maxsize(size: int) -> None:
    """Update RNG cache maximum size."""

    global _CACHE_MAXSIZE
    _CACHE_MAXSIZE = int(size)

__all__ = ["get_rng", "make_rng", "set_cache_maxsize"]
