# TNFR Intelligent Caching System

A hierarchical, dependency-aware caching system designed for TNFR (Teoría de la Naturaleza Fractal Resonante) computations with selective invalidation.

## Features

### 🏗️ Hierarchical Cache Levels

Organize cached data by computational persistence and cost:

- **GRAPH_STRUCTURE**: Topology, adjacency matrices (invalidated on structural changes)
- **NODE_PROPERTIES**: EPI, νf, θ per node (invalidated on property updates)
- **DERIVED_METRICS**: Si, coherence, ΔNFR (invalidated on dependency changes)
- **TEMPORARY**: Short-lived intermediate computations

### 🎯 Dependency-Aware Invalidation

Selective cache invalidation based on structural dependencies:

```python
from tnfr.caching import TNFRHierarchicalCache, CacheLevel

cache = TNFRHierarchicalCache()

# Cache with dependencies
cache.set(
    "si_node_0",
    0.85,
    CacheLevel.DERIVED_METRICS,
    dependencies={'node_epi', 'node_vf', 'graph_topology'},
    computation_cost=10.0
)

# Invalidate only affected entries
cache.invalidate_by_dependency('node_epi')  # Surgical invalidation
```

### 🧠 Intelligent Memory Management

LRU eviction weighted by computation cost and access patterns:

- High-cost, frequently accessed entries prioritized for retention
- Low-cost, rarely accessed entries evicted first
- Configurable memory limits with automatic eviction

### 🎭 Decorator-Based Caching

Transparent caching with decorators:

```python
from tnfr.caching import cache_tnfr_computation, CacheLevel

@cache_tnfr_computation(
    level=CacheLevel.DERIVED_METRICS,
    dependencies={'graph_topology', 'all_node_epi'},
    cost_estimator=lambda graph: len(graph.nodes()) * 10,
)
def compute_expensive_metric(graph):
    # Expensive computation automatically cached
    return complex_calculation(graph)
```

### 📊 Graph Change Tracking

Automatic cache invalidation on graph modifications:

```python
from tnfr.caching import GraphChangeTracker

cache = TNFRHierarchicalCache()
tracker = GraphChangeTracker(cache)

# Track modifications
tracker.track_graph_changes(graph)

# Automatic invalidation on changes
graph.add_node("n1")  # Topology cache invalidated
tracker.on_node_property_change("n1", "epi", 0.5, 0.7)  # Property cache invalidated
```

### 💾 Persistent Cache

Optional disk-backed storage for expensive computations:

```python
from tnfr.caching import PersistentTNFRCache

cache = PersistentTNFRCache(cache_dir=".tnfr_cache")

# Persist to disk
cache.set_persistent(
    "large_graph_coherence",
    0.95,
    CacheLevel.DERIVED_METRICS,
    {'graph_topology'},
    persist_to_disk=True
)

# Survives between sessions
```

## Installation

The caching system is included with the TNFR package:

```bash
pip install tnfr
```

## Quick Start

```python
import networkx as nx
from tnfr.caching import (
    TNFRHierarchicalCache,
    CacheLevel,
    cache_tnfr_computation,
)

# Create cache
cache = TNFRHierarchicalCache(max_memory_mb=512)

# Use decorator for automatic caching
@cache_tnfr_computation(
    level=CacheLevel.DERIVED_METRICS,
    dependencies={'node_vf', 'node_phase'},
)
def compute_metric(graph, node_id):
    # Your expensive computation
    return calculate_something(graph, node_id)

# First call computes
G = nx.Graph()
G.add_nodes_from(['n1', 'n2', 'n3'])
result = compute_metric(G, 'n1')  # Computed

# Second call uses cache
result = compute_metric(G, 'n1')  # Cached (fast!)
```

## Examples

See `examples/intelligent_caching_demo.py` for comprehensive examples demonstrating:

1. Basic hierarchical cache operations
2. Decorator-based caching
3. Graph change tracking
4. Persistent caching
5. Intelligent cache eviction

Run the examples:

```bash
python examples/intelligent_caching_demo.py
```

## API Reference

### TNFRHierarchicalCache

Main cache class with hierarchical levels and dependency tracking.

**Methods:**
- `get(key, level)` - Retrieve cached value
- `set(key, value, level, dependencies, computation_cost)` - Store value with metadata
- `invalidate_by_dependency(dependency)` - Invalidate entries by dependency
- `invalidate_level(level)` - Clear entire cache level
- `clear()` - Clear all caches
- `get_stats()` - Get cache statistics

### Decorators

- `@cache_tnfr_computation(level, dependencies, cost_estimator)` - Cache function results

### GraphChangeTracker

Track graph modifications and invalidate affected caches.

**Methods:**
- `track_graph_changes(graph)` - Install modification hooks
- `on_node_property_change(node_id, property_name)` - Notify of property changes

### PersistentTNFRCache

Cache with optional disk persistence.

**Methods:**
- `get_persistent(key, level)` - Retrieve from memory or disk
- `set_persistent(key, value, level, dependencies, persist_to_disk)` - Store with persistence
- `clear_persistent_cache(level)` - Clear disk cache
- `cleanup_old_entries(max_age_days)` - Remove old cache files

## TNFR Canonical Invariants

The caching system preserves TNFR structural semantics:

- **Operator closure**: Cache invalidation respects structural operators
- **Structural units**: Cache keys preserve νf (Hz_str) semantics
- **ΔNFR semantics**: Invalidation based on structural reorganization
- **Phase synchrony**: Caching respects phase requirements
- **Controlled determinism**: Reproducible cache behavior

## Performance

Expected performance improvements:

- **Sense index (Si) computations**: 5-50x speedup on repeated calculations
- **Coherence metrics**: 10-100x speedup for large graphs
- **Topology queries**: Near-instant retrieval after first computation
- **Memory efficiency**: Intelligent eviction prevents memory bloat

## Testing

Run the test suite:

```bash
pytest tests/unit/caching/ -v
```

60 comprehensive tests covering:
- Cache correctness and invalidation
- Decorator functionality
- Graph change tracking
- Persistence
- Memory management
- Edge cases

## Contributing

Contributions welcome! Please ensure:

1. New features preserve TNFR canonical invariants
2. Tests achieve >90% coverage
3. Documentation includes examples
4. Performance improvements are benchmarked

## License

MIT License - See LICENSE file for details.
