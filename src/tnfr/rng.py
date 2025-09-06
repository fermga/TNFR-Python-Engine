"""Deterministic RNG helpers."""
from __future__ import annotations

import hashlib
import random
import struct
from functools import lru_cache

from .constants import DEFAULTS


@lru_cache(maxsize=DEFAULTS["JITTER_CACHE_SIZE"])
def get_rng(seed: int, key: int) -> random.Random:
    """Return a cached ``random.Random`` for ``(seed, key)``."""
    seed_bytes = struct.pack(
        ">QQ",
        int(seed) & 0xFFFFFFFFFFFFFFFF,
        int(key) & 0xFFFFFFFFFFFFFFFF,
    )
    seed_int = int.from_bytes(
        hashlib.blake2b(seed_bytes, digest_size=8).digest(), "big"
    )
    return random.Random(seed_int)


__all__ = ["get_rng"]
