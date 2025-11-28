from __future__ import annotations

import random
from .types import GraphLike, TNFRGraph
from .utils import ScopedCounterCache as ScopedCounterCache

__all__ = [
    "seed_hash",
    "make_rng",
    "get_cache_maxsize",
    "set_cache_maxsize",
    "base_seed",
    "cache_enabled",
    "clear_rng_cache",
    "ScopedCounterCache",
]

def seed_hash(seed_int: int, key_int: int) -> int: ...
def make_rng(seed: int, key: int, G: TNFRGraph | GraphLike | None = None) -> random.Random: ...
def clear_rng_cache() -> None: ...
def get_cache_maxsize(G: TNFRGraph | GraphLike) -> int: ...
def cache_enabled(G: TNFRGraph | GraphLike | None = None) -> bool: ...
def base_seed(G: TNFRGraph | GraphLike) -> int: ...
def set_cache_maxsize(size: int) -> None: ...
