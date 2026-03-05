"""TNFR Unified Cache System - Consolidated Memory Management.

Redirects to tnfr.utils.unified_cache.
"""

from ..utils.unified_cache import (
    CacheLevel,
    cache_tnfr_computation,
    TNFRUnifiedCacheSystem,
    UnifiedLRUCache,
    CacheStats,
    get_unified_cache_system,
    get_cache_region,
    clear_unified_caches
)

__all__ = [
    "CacheLevel",
    "cache_tnfr_computation",
    "TNFRUnifiedCacheSystem",
    "UnifiedLRUCache",
    "CacheStats",
    "get_unified_cache_system",
    "get_cache_region",
    "clear_unified_caches"
]
