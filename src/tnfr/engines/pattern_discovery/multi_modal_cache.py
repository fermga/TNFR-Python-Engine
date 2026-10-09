"""Compatibility shim re-exporting the unified cache helpers for pattern discovery."""

from ...dynamics.multi_modal_cache import (  # noqa: F401
    CacheEntryType,
    cache_unified_computation,
    get_unified_cache,
)

__all__ = [
    "CacheEntryType",
    "cache_unified_computation",
    "get_unified_cache",
]
