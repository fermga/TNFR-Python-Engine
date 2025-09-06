"""Deterministic RNG helpers."""
from __future__ import annotations

import hashlib
import random
import struct
from collections import OrderedDict
from typing import Tuple

from .constants import DEFAULTS

_RNG_CACHE: OrderedDict[Tuple[int, int], random.Random] = OrderedDict()


def _make_rng(seed: int, key: int) -> random.Random:
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
    try:
        rng = cache.pop(k)
    except KeyError:
        rng = _make_rng(seed, key)
    cache[k] = rng
    maxsize = int(DEFAULTS.get("JITTER_CACHE_SIZE", 128))
    while maxsize > 0 and len(cache) > maxsize:
        cache.popitem(last=False)
    return rng


def _cache_clear() -> None:
    _RNG_CACHE.clear()


get_rng.cache_clear = _cache_clear  # type: ignore[attr-defined]
get_rng.__wrapped__ = _make_rng  # type: ignore[attr-defined]

__all__ = ["get_rng"]
