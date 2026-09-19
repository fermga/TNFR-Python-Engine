"""Unified cache interface for TNFR.

This module provides a consolidated entry point for all TNFR caching needs,
integrating both the core CacheManager infrastructure and the advanced
hierarchical cache with dependency tracking.

Quick Start
-----------
For basic caching with metrics and persistence layers::

    from tnfr.cache import CacheManager, build_cache_manager

    # Create a cache manager with default settings
    manager = build_cache_manager()

    # Register a named cache
    manager.register("my_cache", lambda: {})

    # Get and use the cache
    cache = manager.get("my_cache")

For advanced hierarchical caching with dependency tracking::

    from tnfr.cache import TNFRHierarchicalCache, CacheLevel

    # Create hierarchical cache
    cache = TNFRHierarchicalCache(max_memory_mb=256)

    # Store with dependencies
    cache.set(
        "metric_key",
        value=0.95,
        level=CacheLevel.DERIVED_METRICS,
        dependencies={'graph_topology', 'node_properties'}
    )

    # Selective invalidation
    cache.invalidate_by_dependency('graph_topology')

For graph-specific caching::

    from tnfr.cache import configure_graph_cache_limits
    import networkx as nx

    G = nx.Graph()
    configure_graph_cache_limits(
        G,
        default_capacity=256,
        overrides={"dnfr_prep": 512}
    )

See Also
--------
tnfr.utils.cache : Core cache infrastructure and CacheManager
tnfr.caching : Advanced hierarchical cache with dependency tracking
tnfr.metrics.cache_utils : Hot-path cache configuration helpers
"""

from __future__ import annotations

# Hot-path cache configuration helpers
from .metrics.cache_utils import (
    CacheStats,
    configure_hot_path_caches,
    get_cache_config,
    log_cache_metrics,
)

# Hierarchical cache with dependency tracking (now all in utils.cache)
# Core cache infrastructure from tnfr.utils.cache
from .utils.cache import (  # Main classes; Configuration and stats; Factory functions; Graph-specific helpers; ΔNFR caching; Security
    CacheCapacityConfig,
    CacheEntry,
    CacheLayer,
    CacheLevel,
    CacheManager,
    CacheStatistics,
    DnfrCache,
    DnfrPrepState,
    EdgeCacheManager,
    GraphChangeTracker,
    InstrumentedLRUCache,
    ManagedLRUCache,
    MappingCacheLayer,
    PersistentTNFRCache,
    RedisCacheLayer,
    SecurityError,
    SecurityWarning,
    ShelveCacheLayer,
    TNFRHierarchicalCache,
    build_cache_manager,
    cache_tnfr_computation,
    cached_node_list,
    cached_nodes_and_A,
    configure_global_cache_layers,
    configure_graph_cache_limits,
    create_hmac_signer,
    create_hmac_validator,
    create_secure_redis_layer,
    create_secure_shelve_layer,
    edge_version_cache,
    edge_version_update,
    get_global_cache,
    increment_edge_version,
    invalidate_function_cache,
    new_dnfr_cache,
    reset_global_cache,
    reset_global_cache_manager,
    set_global_cache,
    track_node_property_update,
)

__all__ = [
    # Core cache classes
    "CacheManager",
    "CacheLayer",
    "MappingCacheLayer",
    "ShelveCacheLayer",
    "RedisCacheLayer",
    "InstrumentedLRUCache",
    "ManagedLRUCache",
    "EdgeCacheManager",
    # Configuration and stats
    "CacheCapacityConfig",
    "CacheStatistics",
    "CacheStats",
    # Factory functions
    "build_cache_manager",
    "create_hmac_signer",
    "create_hmac_validator",
    "create_secure_shelve_layer",
    "create_secure_redis_layer",
    # Graph-specific helpers
    "configure_graph_cache_limits",
    "configure_global_cache_layers",
    "reset_global_cache_manager",
    "edge_version_cache",
    "cached_node_list",
    "cached_nodes_and_A",
    "increment_edge_version",
    "edge_version_update",
    "get_cache_config",
    "configure_hot_path_caches",
    "log_cache_metrics",
    # ΔNFR caching
    "DnfrCache",
    "DnfrPrepState",
    "new_dnfr_cache",
    # Hierarchical cache
    "TNFRHierarchicalCache",
    "CacheLevel",
    "CacheEntry",
    "cache_tnfr_computation",
    "invalidate_function_cache",
    "get_global_cache",
    "set_global_cache",
    "reset_global_cache",
    # Change tracking
    "GraphChangeTracker",
    "track_node_property_update",
    "PersistentTNFRCache",
    # Security
    "SecurityError",
    "SecurityWarning",
]
