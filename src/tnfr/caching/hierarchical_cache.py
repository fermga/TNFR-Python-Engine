"""Hierarchical cache with dependency-aware invalidation for TNFR.

This module implements a multi-level cache system that respects TNFR structural
semantics and provides selective invalidation based on dependencies.

The hierarchical cache now uses ``CacheManager`` from ``tnfr.utils.cache`` as its
backend, providing unified cache management with consistent metrics and telemetry
across the entire TNFR codebase.
"""

from __future__ import annotations

import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

from ..utils.cache import (
    CacheManager,
    CacheStatistics,
    InstrumentedLRUCache,
)

__all__ = ["CacheLevel", "CacheEntry", "TNFRHierarchicalCache"]


class CacheLevel(Enum):
    """Cache levels organized by persistence and computational cost.
    
    Levels are ordered from most persistent (rarely changes) to least
    persistent (frequently recomputed):
    
    - GRAPH_STRUCTURE: Topology, adjacency matrices (invalidated on add/remove node/edge)
    - NODE_PROPERTIES: EPI, νf, θ per node (invalidated on property updates)
    - DERIVED_METRICS: Si, coherence, ΔNFR (invalidated on dependency changes)
    - TEMPORARY: Intermediate computations (short-lived, frequently evicted)
    """
    
    GRAPH_STRUCTURE = "graph_structure"
    NODE_PROPERTIES = "node_properties"
    DERIVED_METRICS = "derived_metrics"
    TEMPORARY = "temporary"


@dataclass
class CacheEntry:
    """Cache entry with metadata for intelligent invalidation and eviction.
    
    Attributes
    ----------
    value : Any
        The cached computation result.
    dependencies : set[str]
        Set of structural properties this entry depends on. Used for
        selective invalidation. Examples: 'node_epi', 'node_vf', 'graph_topology'.
    timestamp : float
        Time when entry was created (from time.time()).
    access_count : int
        Number of times this entry has been accessed.
    computation_cost : float
        Estimated computational cost to regenerate this value. Higher cost
        entries are prioritized during eviction.
    size_bytes : int
        Estimated memory size in bytes.
    """
    
    value: Any
    dependencies: set[str]
    timestamp: float
    access_count: int = 0
    computation_cost: float = 1.0
    size_bytes: int = 0


class TNFRHierarchicalCache:
    """Hierarchical cache with dependency-aware selective invalidation.
    
    This cache system organizes entries by structural level and tracks
    dependencies to enable surgical invalidation. Only entries that depend
    on changed structural properties are evicted, preserving valid cached data.
    
    Internally uses ``CacheManager`` from ``tnfr.utils.cache`` for unified cache
    management, metrics, and telemetry integration with the rest of TNFR.
    
    Parameters
    ----------
    max_memory_mb : int, default: 512
        Maximum memory usage in megabytes before eviction starts.
    enable_metrics : bool, default: True
        Whether to track cache hit/miss metrics for telemetry.
    cache_manager : CacheManager, optional
        Existing CacheManager to use. If None, creates a new one.
    
    Attributes
    ----------
    hits : int
        Number of successful cache retrievals.
    misses : int
        Number of cache misses.
    evictions : int
        Number of entries evicted due to memory pressure.
    invalidations : int
        Number of entries invalidated due to dependency changes.
    
    Examples
    --------
    >>> cache = TNFRHierarchicalCache(max_memory_mb=128)
    >>> # Cache a derived metric with dependencies
    >>> cache.set(
    ...     "coherence_global",
    ...     0.95,
    ...     CacheLevel.DERIVED_METRICS,
    ...     dependencies={'graph_topology', 'all_node_vf'},
    ...     computation_cost=100.0
    ... )
    >>> cache.get("coherence_global", CacheLevel.DERIVED_METRICS)
    0.95
    >>> # Invalidate when topology changes
    >>> cache.invalidate_by_dependency('graph_topology')
    >>> cache.get("coherence_global", CacheLevel.DERIVED_METRICS)
    
    """
    
    def __init__(
        self,
        max_memory_mb: int = 512,
        enable_metrics: bool = True,
        cache_manager: Optional[CacheManager] = None,
    ):
        # Use provided CacheManager or create new one
        if cache_manager is None:
            # Estimate entries per MB (rough heuristic: ~100 entries per MB)
            default_capacity = max(32, int(max_memory_mb * 100 / len(CacheLevel)))
            cache_manager = CacheManager(
                storage={},
                default_capacity=default_capacity,
            )
        
        self._manager = cache_manager
        self._max_memory = max_memory_mb * 1024 * 1024
        self._current_memory = 0
        self._enable_metrics = enable_metrics
        
        # Dependency tracking (remains in hierarchical cache)
        self._dependencies: dict[str, set[tuple[CacheLevel, str]]] = defaultdict(set)
        
        # Register a cache for each level in the CacheManager
        self._level_cache_names: dict[CacheLevel, str] = {}
        for level in CacheLevel:
            cache_name = f"hierarchical_{level.value}"
            self._level_cache_names[level] = cache_name
            
            # Simple factory returning empty dict for each cache level
            self._manager.register(
                cache_name,
                factory=lambda: {},
                create=True,
            )
        
        # Metrics (tracked locally for backward compatibility)
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.invalidations = 0
    
    @property
    def _caches(self) -> dict[CacheLevel, dict[str, CacheEntry]]:
        """Provide backward compatibility for accessing internal caches.
        
        This property returns a view of the caches stored in the CacheManager,
        maintaining compatibility with code that directly accessed the old
        _caches attribute.
        """
        result: dict[CacheLevel, dict[str, CacheEntry]] = {}
        for level in CacheLevel:
            cache_name = self._level_cache_names[level]
            result[level] = self._manager.get(cache_name)
        return result
    
    def get(self, key: str, level: CacheLevel) -> Optional[Any]:
        """Retrieve value from cache if it exists and is valid.
        
        Parameters
        ----------
        key : str
            Cache key identifying the entry.
        level : CacheLevel
            Cache level to search in.
        
        Returns
        -------
        Any or None
            The cached value if found, None otherwise.
        
        Examples
        --------
        >>> cache = TNFRHierarchicalCache()
        >>> cache.set("key1", 42, CacheLevel.TEMPORARY, dependencies=set())
        >>> cache.get("key1", CacheLevel.TEMPORARY)
        42
        >>> cache.get("missing", CacheLevel.TEMPORARY)
        
        """
        cache_name = self._level_cache_names[level]
        level_cache: dict[str, CacheEntry] = self._manager.get(cache_name)
        
        if key in level_cache:
            entry = level_cache[key]
            entry.access_count += 1
            if self._enable_metrics:
                self.hits += 1
                self._manager.increment_hit(cache_name)
            return entry.value
        
        if self._enable_metrics:
            self.misses += 1
            self._manager.increment_miss(cache_name)
        return None
    
    def set(
        self,
        key: str,
        value: Any,
        level: CacheLevel,
        dependencies: set[str],
        computation_cost: float = 1.0,
    ) -> None:
        """Store value in cache with dependency metadata.
        
        Parameters
        ----------
        key : str
            Unique identifier for this cache entry.
        value : Any
            The value to cache.
        level : CacheLevel
            Which cache level to store in.
        dependencies : set[str]
            Set of structural properties this value depends on.
        computation_cost : float, default: 1.0
            Estimated cost to recompute this value. Used for eviction priority.
        
        Examples
        --------
        >>> cache = TNFRHierarchicalCache()
        >>> cache.set(
        ...     "si_node_5",
        ...     0.87,
        ...     CacheLevel.DERIVED_METRICS,
        ...     dependencies={'node_vf_5', 'node_phase_5'},
        ...     computation_cost=5.0
        ... )
        """
        # Estimate size
        estimated_size = self._estimate_size(value)
        
        # Check if we need to evict
        if self._current_memory + estimated_size > self._max_memory:
            self._evict_lru(estimated_size)
        
        # Create entry
        entry = CacheEntry(
            value=value,
            dependencies=dependencies.copy(),
            timestamp=time.time(),
            computation_cost=computation_cost,
            size_bytes=estimated_size,
        )
        
        cache_name = self._level_cache_names[level]
        level_cache: dict[str, CacheEntry] = self._manager.get(cache_name)
        
        # Remove old entry if exists
        if key in level_cache:
            old_entry = level_cache[key]
            self._current_memory -= old_entry.size_bytes
            # Clean up old dependencies
            for dep in old_entry.dependencies:
                if dep in self._dependencies:
                    self._dependencies[dep].discard((level, key))
        
        # Store entry
        level_cache[key] = entry
        self._manager.store(cache_name, level_cache)
        self._current_memory += estimated_size
        
        # Register dependencies
        for dep in dependencies:
            self._dependencies[dep].add((level, key))
    
    def invalidate_by_dependency(self, dependency: str) -> int:
        """Invalidate all cache entries that depend on a structural property.
        
        This implements selective invalidation: only entries that explicitly
        depend on the changed property are removed, preserving unaffected caches.
        
        Parameters
        ----------
        dependency : str
            The structural property that changed (e.g., 'graph_topology',
            'node_epi_5', 'all_node_vf').
        
        Returns
        -------
        int
            Number of entries invalidated.
        
        Examples
        --------
        >>> cache = TNFRHierarchicalCache()
        >>> cache.set("key1", 1, CacheLevel.TEMPORARY, {'dep1', 'dep2'})
        >>> cache.set("key2", 2, CacheLevel.TEMPORARY, {'dep2'})
        >>> cache.invalidate_by_dependency('dep1')  # Only invalidates key1
        1
        >>> cache.get("key1", CacheLevel.TEMPORARY)  # None
        
        >>> cache.get("key2", CacheLevel.TEMPORARY)  # Still cached
        2
        """
        count = 0
        if dependency in self._dependencies:
            entries_to_remove = list(self._dependencies[dependency])
            for level, key in entries_to_remove:
                cache_name = self._level_cache_names[level]
                level_cache: dict[str, CacheEntry] = self._manager.get(cache_name)
                
                if key in level_cache:
                    entry = level_cache[key]
                    self._current_memory -= entry.size_bytes
                    del level_cache[key]
                    self._manager.store(cache_name, level_cache)
                    count += 1
                    
                    # Clean up all dependency references for this entry
                    for dep in entry.dependencies:
                        if dep in self._dependencies:
                            self._dependencies[dep].discard((level, key))
            
            # Clean up the dependency key itself
            del self._dependencies[dependency]
        
        if self._enable_metrics:
            self.invalidations += count
        
        return count
    
    def invalidate_level(self, level: CacheLevel) -> int:
        """Invalidate all entries in a specific cache level.
        
        Parameters
        ----------
        level : CacheLevel
            The cache level to clear.
        
        Returns
        -------
        int
            Number of entries invalidated.
        """
        cache_name = self._level_cache_names[level]
        level_cache: dict[str, CacheEntry] = self._manager.get(cache_name)
        count = len(level_cache)
        
        # Clean up dependencies
        for key, entry in level_cache.items():
            self._current_memory -= entry.size_bytes
            for dep in entry.dependencies:
                if dep in self._dependencies:
                    self._dependencies[dep].discard((level, key))
        
        level_cache.clear()
        self._manager.store(cache_name, level_cache)
        
        if self._enable_metrics:
            self.invalidations += count
        
        return count
    
    def clear(self) -> None:
        """Clear all cache levels and reset metrics."""
        for level in CacheLevel:
            cache_name = self._level_cache_names[level]
            self._manager.clear(cache_name)
        
        self._dependencies.clear()
        self._current_memory = 0
        
        # Always reset metrics regardless of _enable_metrics
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.invalidations = 0
    
    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics for telemetry.
        
        Returns
        -------
        dict[str, Any]
            Dictionary containing:
            - hits: Number of cache hits
            - misses: Number of cache misses
            - hit_rate: Ratio of hits to total accesses
            - evictions: Number of evictions
            - invalidations: Number of invalidations
            - memory_used_mb: Current memory usage in MB
            - memory_limit_mb: Memory limit in MB
            - entry_counts: Number of entries per level
        """
        total_accesses = self.hits + self.misses
        hit_rate = self.hits / total_accesses if total_accesses > 0 else 0.0
        
        entry_counts = {}
        for level in CacheLevel:
            cache_name = self._level_cache_names[level]
            level_cache: dict[str, CacheEntry] = self._manager.get(cache_name)
            entry_counts[level.value] = len(level_cache)
        
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate,
            "evictions": self.evictions,
            "invalidations": self.invalidations,
            "memory_used_mb": self._current_memory / (1024 * 1024),
            "memory_limit_mb": self._max_memory / (1024 * 1024),
            "entry_counts": entry_counts,
        }
    
    def _estimate_size(self, value: Any) -> int:
        """Estimate memory size of a value in bytes.
        
        Uses sys.getsizeof for a rough estimate. For complex objects,
        this may underestimate the true memory usage.
        """
        try:
            return sys.getsizeof(value)
        except (TypeError, AttributeError):
            # Fallback for objects that don't support getsizeof
            return 64  # Default estimate
    
    def _evict_lru(self, needed_space: int) -> None:
        """Evict least valuable entries until enough space is freed.
        
        Value is determined by: access_count / computation_cost.
        Lower values are evicted first (low access, low cost to recompute).
        """
        # Collect all entries with priority scores
        all_entries: list[tuple[float, CacheLevel, str, CacheEntry]] = []
        for level in CacheLevel:
            cache_name = self._level_cache_names[level]
            level_cache: dict[str, CacheEntry] = self._manager.get(cache_name)
            for key, entry in level_cache.items():
                # Priority = (access_count + 1) * computation_cost
                # Higher priority = keep longer
                # Add 1 to access_count to avoid zero priority
                priority = (entry.access_count + 1) * entry.computation_cost
                all_entries.append((priority, level, key, entry))
        
        # Sort by priority (ascending - lowest priority first)
        all_entries.sort(key=lambda x: x[0])
        
        freed_space = 0
        for priority, level, key, entry in all_entries:
            if freed_space >= needed_space:
                break
            
            # Remove entry
            cache_name = self._level_cache_names[level]
            level_cache: dict[str, CacheEntry] = self._manager.get(cache_name)
            if key in level_cache:
                del level_cache[key]
                self._manager.store(cache_name, level_cache)
                freed_space += entry.size_bytes
                self._current_memory -= entry.size_bytes
                
                # Clean up dependencies
                for dep in entry.dependencies:
                    if dep in self._dependencies:
                        self._dependencies[dep].discard((level, key))
                
                if self._enable_metrics:
                    self.evictions += 1
                    self._manager.increment_eviction(cache_name)
