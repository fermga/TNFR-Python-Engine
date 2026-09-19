"""TNFR Unified Cache System - Consolidated Memory Management.

Redirects to tnfr.utils.unified_cache.
"""

from ..utils.unified_cache import (
    CacheLevel,
    CacheStats,
    TNFRUnifiedCacheSystem,
    UnifiedLRUCache,
    cache_tnfr_computation,
    clear_unified_caches,
    get_cache_region,
    get_unified_cache_system,
)

__all__ = [
    "CacheLevel",
    "cache_tnfr_computation",
    "TNFRUnifiedCacheSystem",
    "UnifiedLRUCache",
    "CacheStats",
    "get_unified_cache_system",
    "get_cache_region",
    "clear_unified_caches",
]
