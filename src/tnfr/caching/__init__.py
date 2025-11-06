"""Intelligent hierarchical caching system for TNFR operations.

.. deprecated::
   The ``tnfr.caching`` package is deprecated and will be removed in a future version.
   Please use ``tnfr.cache`` or ``tnfr.utils.cache`` instead for all caching functionality.

   Migration guide:

   - ``from tnfr.caching import TNFRHierarchicalCache`` → ``from tnfr.cache import TNFRHierarchicalCache``
   - ``from tnfr.caching import CacheLevel`` → ``from tnfr.cache import CacheLevel``
   - ``from tnfr.caching import cache_tnfr_computation`` → ``from tnfr.cache import cache_tnfr_computation``

   All functionality remains available through ``tnfr.cache`` with identical APIs.

This module provides a sophisticated caching infrastructure that respects TNFR
structural semantics, offering:

1. **Hierarchical cache levels** - Caches organized by computation cost and
   structural persistence (graph structure, node properties, derived metrics).

2. **Dependency-aware invalidation** - Selective cache invalidation based on
   which structural properties changed, preserving valid cached data.

3. **Memory management** - Intelligent LRU eviction weighted by computation cost
   and access patterns to optimize cache efficiency.

4. **Persistent caching** - Optional disk-backed cache for expensive computations
   that survive between sessions.

The system maintains TNFR canonical invariants:
- EPI changes tracked via structural operators
- νf expressed in Hz_str (structural hertz)
- ΔNFR semantics preserved during invalidation
- Phase synchrony respected in cache keys
- Controlled determinism through reproducible cache behavior

Examples
--------
Basic usage with the hierarchical cache:

>>> from tnfr.caching import TNFRHierarchicalCache, CacheLevel  # doctest: +SKIP
>>> cache = TNFRHierarchicalCache(max_memory_mb=256)  # doctest: +SKIP
>>> # Cache a computation result
>>> cache.set(  # doctest: +SKIP
...     "si_node_0",
...     0.85,
...     CacheLevel.DERIVED_METRICS,
...     dependencies={'node_epi', 'node_vf', 'graph_topology'},
...     computation_cost=10.0
... )
>>> # Retrieve from cache
>>> result = cache.get("si_node_0", CacheLevel.DERIVED_METRICS)  # doctest: +SKIP
>>> result  # doctest: +SKIP
0.85
>>> # Invalidate when EPI changes
>>> cache.invalidate_by_dependency('node_epi')  # doctest: +SKIP
>>> cache.get("si_node_0", CacheLevel.DERIVED_METRICS)  # doctest: +SKIP

Using decorators for transparent caching:

>>> from tnfr.caching import cache_tnfr_computation, CacheLevel  # doctest: +SKIP
>>> @cache_tnfr_computation(  # doctest: +SKIP
...     level=CacheLevel.DERIVED_METRICS,
...     dependencies={'node_vf', 'node_phase'},
... )
... def compute_expensive_metric(graph, node_id):  # doctest: +SKIP
...     # Expensive computation here
...     return 0.75
"""

from __future__ import annotations

import warnings

# Issue deprecation warning
warnings.warn(
    "The 'tnfr.caching' package is deprecated and will be removed in a future version. "
    "Please use 'tnfr.cache' instead. All functionality is available through tnfr.cache "
    "with identical APIs. See migration guide in documentation.",
    DeprecationWarning,
    stacklevel=2,
)

# Import from consolidated location (utils.cache) for compatibility
from ..utils.cache import (
    CacheLevel,
    CacheEntry,
    TNFRHierarchicalCache,
    cache_tnfr_computation,
    invalidate_function_cache,
    GraphChangeTracker,
    track_node_property_update,
    PersistentTNFRCache,
)

__all__ = [
    "CacheLevel",
    "CacheEntry",
    "TNFRHierarchicalCache",
    "cache_tnfr_computation",
    "invalidate_function_cache",
    "GraphChangeTracker",
    "track_node_property_update",
    "PersistentTNFRCache",
]
