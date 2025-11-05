"""Intelligent hierarchical caching system for TNFR operations.

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

>>> from tnfr.caching import TNFRHierarchicalCache, CacheLevel
>>> cache = TNFRHierarchicalCache(max_memory_mb=256)
>>> # Cache a computation result
>>> cache.set(
...     "si_node_0",
...     0.85,
...     CacheLevel.DERIVED_METRICS,
...     dependencies={'node_epi', 'node_vf', 'graph_topology'},
...     computation_cost=10.0
... )
>>> # Retrieve from cache
>>> result = cache.get("si_node_0", CacheLevel.DERIVED_METRICS)
>>> result
0.85
>>> # Invalidate when EPI changes
>>> cache.invalidate_by_dependency('node_epi')
>>> cache.get("si_node_0", CacheLevel.DERIVED_METRICS)  # Returns None

Using decorators for transparent caching:

>>> from tnfr.caching import cache_tnfr_computation, CacheLevel
>>> @cache_tnfr_computation(
...     level=CacheLevel.DERIVED_METRICS,
...     dependencies={'node_vf', 'node_phase'},
... )
... def compute_expensive_metric(graph, node_id):
...     # Expensive computation here
...     return 0.75
"""

from __future__ import annotations

from .hierarchical_cache import (
    CacheLevel,
    CacheEntry,
    TNFRHierarchicalCache,
)
from .decorators import cache_tnfr_computation, invalidate_function_cache
from .invalidation import GraphChangeTracker, track_node_property_update
from .persistence import PersistentTNFRCache

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
