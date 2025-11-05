"""Hierarchical cache with dependency-aware invalidation for TNFR.

.. deprecated:: 
   This module is deprecated. Import from ``tnfr.cache`` or ``tnfr.utils.cache`` instead.

This module implements a multi-level cache system that respects TNFR structural
semantics and provides selective invalidation based on dependencies.

The hierarchical cache now uses ``CacheManager`` from ``tnfr.utils.cache`` as its
backend, providing unified cache management with consistent metrics and telemetry
across the entire TNFR codebase.
"""

from __future__ import annotations

import warnings

warnings.warn(
    "Importing from tnfr.caching.hierarchical_cache is deprecated. "
    "Use 'from tnfr.cache import TNFRHierarchicalCache, CacheLevel, CacheEntry' instead.",
    DeprecationWarning,
    stacklevel=2
)

# Re-export from consolidated location for backward compatibility
from ..utils.cache import (
    CacheLevel,
    CacheEntry,
    TNFRHierarchicalCache,
)

__all__ = ["CacheLevel", "CacheEntry", "TNFRHierarchicalCache"]
