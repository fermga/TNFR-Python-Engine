# TNFR Cache Architecture

## Overview

TNFR implements a unified, multi-layered caching system that respects structural semantics and provides consistent metrics, telemetry, and persistence across all cache types.

## Architecture Layers

### 1. Core Cache Infrastructure (`tnfr.utils.cache`)

The foundation of all TNFR caching, providing:

- **CacheManager**: Orchestrates named caches with per-entry locks and metrics
- **Cache Layers**: Abstract interface supporting memory (Mapping), disk (Shelve), and distributed (Redis) backends
- **Instrumented Caches**: LRU caches with built-in hit/miss/eviction telemetry
- **Edge Version Caching**: Graph-aware caching that invalidates on topology changes

**When to use**:
- Graph-level caching (ΔNFR prep, adjacency matrices, node lists)
- Hot-path caching with metrics
- Persistent caching across sessions
- Multi-layered cache hierarchies (memory → disk → redis)

**Example**:
```python
from tnfr.cache import build_cache_manager, edge_version_cache

# Create manager with persistent layer
manager = build_cache_manager()

# Cache tied to graph topology
def expensive_computation(G):
    return edge_version_cache(
        G,
        key="my_metric",
        builder=lambda: compute_expensive_metric(G),
        max_entries=128
    )
```

### 2. Hierarchical Cache (`tnfr.caching`)

Advanced caching with dependency-aware invalidation, built on top of CacheManager:

- **Dependency Tracking**: Selective invalidation based on what changed
- **Cache Levels**: Organize by persistence (graph structure → node properties → derived metrics)
- **Cost-aware Eviction**: Prioritize keeping expensive-to-recompute values
- **Persistent Storage**: Optional disk backing for expensive computations

**When to use**:
- Complex derived metrics with known dependencies
- Selective cache invalidation (only invalidate affected entries)
- Multi-level caching by computation cost
- Persistent caching of expensive operations

**Example**:
```python
from tnfr.cache import TNFRHierarchicalCache, CacheLevel

cache = TNFRHierarchicalCache(max_memory_mb=256)

# Cache with dependencies
cache.set(
    "coherence_global",
    value=0.95,
    level=CacheLevel.DERIVED_METRICS,
    dependencies={'graph_topology', 'all_node_vf'},
    computation_cost=100.0
)

# Selective invalidation: only topology-dependent entries cleared
cache.invalidate_by_dependency('graph_topology')
```

### 3. Specialized Caches

Domain-specific caches for hot paths:

- **TrigCache** (`tnfr.metrics.trig_cache`): Cosine/sine values for phase calculations
- **NodeNX Cache**: Cached node instances with weak/strong reference modes
- **SeedHashCache**: Deterministic seed generation for reproducibility

## Unified Interface

The `tnfr.cache` module provides a single import point:

```python
from tnfr.cache import (
    # Core infrastructure
    CacheManager,
    build_cache_manager,
    
    # Hierarchical cache
    TNFRHierarchicalCache,
    CacheLevel,
    
    # Graph-specific
    configure_graph_cache_limits,
    edge_version_cache,
    
    # Decorators
    cache_tnfr_computation,
    
    # Hot-path configuration
    configure_hot_path_caches,
)
```

## Design Principles

### 1. TNFR Canonical Invariants

All caches preserve TNFR semantics:

- **EPI coherence**: Cached structures change only via structural operators
- **Structural units**: Frequencies in Hz_str, not arbitrary time units
- **ΔNFR semantics**: Cache invalidation respects reorganization semantics
- **Phase synchrony**: Cache keys include phase information where relevant
- **Determinism**: Reproducible with seeds, traceable with structural logs

### 2. Unified Metrics

All caches report to the same telemetry infrastructure:

```python
from tnfr.cache import CacheManager

manager = build_cache_manager()
# ... use caches ...

# Get aggregated metrics
stats = manager.aggregate_metrics()
print(f"Hit rate: {stats.hits / (stats.hits + stats.misses):.2%}")

# Per-cache breakdown
for name, cache_stats in manager.iter_metrics():
    print(f"{name}: {cache_stats.hits} hits, {cache_stats.misses} misses")
```

### 3. Layered Storage

Caches can span multiple storage layers with automatic hydration:

```python
from tnfr.cache import build_cache_manager, create_secure_shelve_layer

# Create persistent layer
shelve_layer = create_secure_shelve_layer("cache.db")

# Manager with memory + disk layers
manager = build_cache_manager(layers=[shelve_layer])
manager.register("expensive_cache", lambda: {})

# First access: miss, computed, stored in both layers
result = manager.get("expensive_cache")

# Second access: hit from memory layer
result = manager.get("expensive_cache")

# After restart: hit from disk layer, hydrated to memory
```

### 4. Security by Default

Persistent caches use HMAC signatures to prevent tampering:

```python
from tnfr.cache import create_secure_shelve_layer
import os

# Secure cache with signature validation
os.environ["TNFR_CACHE_SECRET"] = "your-secure-random-key"
layer = create_secure_shelve_layer("coherence.db")

# Rejects tampered data automatically
```

## Cache Selection Guide

| Use Case | Recommended Cache | Why |
|----------|-------------------|-----|
| Graph topology (adjacency, degrees) | `edge_version_cache` | Auto-invalidates on graph changes |
| ΔNFR preparation buffers | `CacheManager` with ΔNFR | Integrated with dynamics engine |
| Trigonometric values | `TrigCache` | Specialized for cos/sin with theta tracking |
| Derived metrics (Si, coherence) | `TNFRHierarchicalCache` | Dependency tracking + smart eviction |
| Function memoization | `@cache_tnfr_computation` | Decorator with automatic key generation |
| Persistent expensive results | `PersistentTNFRCache` | Survives restarts, disk-backed |

## Configuration

### Graph-Level Configuration

```python
from tnfr.cache import configure_graph_cache_limits
import networkx as nx

G = nx.Graph()
configure_graph_cache_limits(
    G,
    default_capacity=256,  # Default for all caches
    overrides={
        "dnfr_prep": 512,  # Larger for ΔNFR
        "trig_cache": 1024,  # Very large for trig
    }
)
```

### Hot-Path Configuration

```python
from tnfr.cache import configure_hot_path_caches

configure_hot_path_caches(
    G,
    buffer_max_entries=256,
    trig_cache_size=512,
    coherence_cache_size=128,
)
```

### Global Cache Layers

```python
from tnfr.cache import configure_global_cache_layers

# Configure persistent layer for all graphs
configure_global_cache_layers(
    shelve={"path": "tnfr_cache.db"},
    redis={"enabled": True, "namespace": "tnfr:prod"},
)
```

## Performance Considerations

### Memory Management

1. **Set appropriate capacity limits**: Prevent unbounded memory growth
2. **Use weak references for ephemeral graphs**: Let GC reclaim temporary graphs
3. **Layer expensive operations**: Memory → disk → compute
4. **Monitor hit rates**: `manager.aggregate_metrics()`

### Invalidation Strategies

1. **Coarse-grained**: Clear entire cache levels (`cache.invalidate_level(level)`)
2. **Fine-grained**: Selective by dependency (`cache.invalidate_by_dependency('topology')`)
3. **Version-based**: Edge version cache auto-invalidates on graph changes

### Serialization Costs

1. **Shelve layer**: Pickle overhead, use for expensive computations only
2. **Redis layer**: Network latency, best for distributed systems
3. **Memory layer**: Zero overhead, always fastest

## Migration Guide

### From Old `tnfr.cache`

The old `tnfr.cache` raised ImportError. Now it's a unified interface:

```python
# OLD (raised ImportError)
from tnfr.cache import ...

# NEW (works!)
from tnfr.cache import CacheManager, TNFRHierarchicalCache, ...
```

### From Direct `tnfr.utils.cache` Imports

Still works, but unified interface is preferred:

```python
# Still works
from tnfr.utils.cache import CacheManager

# Preferred
from tnfr.cache import CacheManager
```

### From Standalone `tnfr.caching`

Integration is automatic - hierarchical cache now uses CacheManager:

```python
from tnfr.cache import TNFRHierarchicalCache

# Now backed by CacheManager, inherits all its features
cache = TNFRHierarchicalCache()
```

## Testing

All cache tests validate the unified architecture:

```bash
# Test core infrastructure
pytest tests/unit/structural/test_cache_*.py

# Test hierarchical cache
pytest tests/unit/caching/

# Test metrics integration
pytest tests/unit/metrics/test_*cache*.py
```

## Troubleshooting

### High Memory Usage

Check cache sizes and eviction policies:

```python
manager = build_cache_manager()
for name, stats in manager.iter_metrics():
    print(f"{name}: {stats.hits} hits, {stats.misses} misses, {stats.evictions} evictions")
```

### Poor Hit Rates

1. Check if cache size is too small
2. Verify cache keys are stable (deterministic)
3. Monitor invalidation frequency

### Security Warnings

Set `TNFR_ALLOW_UNSIGNED_PICKLE=1` to suppress (not recommended for production):

```python
import os
os.environ["TNFR_ALLOW_UNSIGNED_PICKLE"] = "1"
```

Better: Use secure layers with signatures:

```python
from tnfr.cache import create_secure_shelve_layer
layer = create_secure_shelve_layer("cache.db", secret="your-secret")
```

## Future Enhancements

- [ ] Automatic cache warming on graph load
- [ ] Cache compression for large structures
- [ ] Distributed cache coordination
- [ ] Cache versioning for schema changes
- [ ] ML-based cache replacement policies
