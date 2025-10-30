"""Compatibility re-export for cache infrastructure primitives.

The canonical implementations now live in :mod:`tnfr.utils.cache`. This module
is kept as a thin facade to preserve the historical import path
``tnfr.cache`` for downstream integrations.
"""

from __future__ import annotations

import warnings

warnings.warn(
    "Importing 'tnfr.cache' is deprecated; use 'tnfr.utils' for the supported API.",
    DeprecationWarning,
    stacklevel=2,
)

from .utils.cache import (
    CacheCapacityConfig,
    CacheLayer,
    CacheManager,
    CacheStatistics,
    InstrumentedLRUCache,
    ManagedLRUCache,
    MappingCacheLayer,
    RedisCacheLayer,
    ShelveCacheLayer,
    prune_lock_mapping,
)

__all__ = [
    "CacheLayer",
    "CacheManager",
    "CacheCapacityConfig",
    "CacheStatistics",
    "InstrumentedLRUCache",
    "MappingCacheLayer",
    "RedisCacheLayer",
    "ShelveCacheLayer",
    "ManagedLRUCache",
    "prune_lock_mapping",
]
