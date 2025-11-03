# TNFR Caching Strategy

## Overview

The TNFR engine employs a multi-layered caching strategy to optimize hot computational paths while maintaining structural coherence and deterministic behavior. This document catalogs all cache patterns, key prefixes, and invalidation strategies used across the codebase.

## Core Caching Infrastructure

### Cache Layers

1. **`CacheManager`** (`tnfr.utils.cache.CacheManager`)
   - Centralized cache orchestration with per-entry locks
   - Supports multiple storage backends (in-memory, shelve, Redis)
   - Provides telemetry hooks for hit/miss tracking
   - Location: `src/tnfr/utils/cache.py`

2. **`EdgeCacheManager`** (`tnfr.utils.cache.EdgeCacheManager`)
   - Specialized cache tied to graph edge version
   - Automatic invalidation on edge structure changes
   - Used for caches dependent on graph topology
   - Location: `src/tnfr/utils/cache.py`

3. **`edge_version_cache`** (function)
   - High-level helper for edge-versioned caching
   - Returns cached value or rebuilds via provided builder function
   - Automatically increments on edge additions/removals
   - Location: `src/tnfr/utils/cache.py`

## Hot Path Caches

### 1. Sense Index (Si) Computation

**Module**: `src/tnfr/metrics/sense_index.py`

**Purpose**: Cache intermediate buffers and structural arrays for vectorized Si computation.

#### Cache Keys

| Key Prefix | Purpose | Buffer Count | Invalidation |
|-----------|---------|--------------|--------------|
| `_si_structural` | Cached νf and ΔNFR arrays per node set | N/A (cache object) | Edge version |
| `_Si_weights` | Normalized α, β, γ weights | N/A (tuple) | Edge version |
| `_si_edges` | Edge index arrays (src, dst) | N/A (arrays) | Edge version |
| `_si_buffers` | Reusable computation buffers | 3 arrays | Edge version |
| `_si_chunk_workspace` | Scratch space for chunked operations | 2 arrays | Edge version |
| `_si_neighbor_buffers` | Neighbor phase aggregation buffers | 5 arrays | Edge version |

#### Implementation Notes

- Uses `ensure_numpy_buffers` for consistent buffer allocation
- Structural arrays (`_si_structural`) track νf and ΔNFR snapshots with change detection
- Edge arrays are cached to avoid repeated neighbor iteration
- All buffers sized to node count for vectorization

### 2. Coherence Matrix (W_ij)

**Module**: `src/tnfr/metrics/coherence.py`

**Purpose**: Cache similarity components and coherence weights between nodes.

#### Cache Keys

| Key Prefix | Purpose | Content | Invalidation |
|-----------|---------|---------|--------------|
| (implicit via history) | Coherence matrix W | Sparse/dense matrix | Per-step computation |
| (implicit via history) | Row sums W_i | Per-node weights | Per-step computation |
| (implicit via history) | W_stats | Min/max/mean statistics | Per-step computation |

#### Implementation Notes

- Coherence matrices stored in history rather than edge-versioned cache
- Supports both sparse and dense storage modes
- Statistics computed once per matrix build
- Vectorized computation when NumPy available

### 3. ΔNFR Preparation Cache

**Module**: `src/tnfr/dynamics/dnfr.py`, `src/tnfr/utils/cache.py`

**Purpose**: Cache intermediate data structures for ΔNFR gradient computation.

#### Cache Structure: `DnfrCache`

| Field | Type | Purpose |
|-------|------|---------|
| `idx` | dict | Node-to-index mapping |
| `theta`, `epi`, `vf` | list | Structural attribute lists |
| `cos_theta`, `sin_theta` | list | Trigonometric values |
| `neighbor_*` | list | Neighbor accumulation buffers |
| `*_np` | array | NumPy array versions (optional) |
| `edge_src`, `edge_dst` | array | Edge index arrays (optional) |
| `checksum` | str | Cache validation token |

#### Implementation Notes

- Managed by `DnfrPrepState` with dedicated locks
- Uses `_graph_cache_manager` for lifecycle coordination
- Supports both Python fallback and NumPy acceleration
- Invalidated via `mark_dnfr_prep_dirty` on structural changes

### 4. Trigonometric Cache

**Module**: `src/tnfr/metrics/trig_cache.py`

**Purpose**: Cache cos(θ) and sin(θ) values per node to avoid redundant computations.

#### Cache Keys

| Key Prefix | Purpose | Content | Invalidation |
|-----------|---------|---------|--------------|
| `_trig` | Trigonometric values | `TrigCache` object | Theta change detection |
| `_trig_version` | Version counter | int | On theta attribute updates |

#### Implementation Notes

- Uses checksums to detect theta changes between cache hits
- Falls back to Python `math.cos`/`sin` when NumPy unavailable
- Stores both dict and array formats for flexible access
- Auto-increments version on cache invalidation

### 5. Neighbor Topology Cache

**Module**: `src/tnfr/metrics/common.py`

**Purpose**: Cache neighbor relationships to avoid repeated graph traversal.

#### Cache Keys

| Key Prefix | Purpose | Content | Invalidation |
|-----------|---------|---------|--------------|
| `_neighbors` | Adjacency mapping | dict[node, list[neighbors]] | Edge version |

#### Implementation Notes

- Simple edge-versioned cache via `edge_version_cache`
- Rebuilt on any topology change
- Used by Si, coherence, and ΔNFR computations

### 6. Node List Cache

**Module**: `src/tnfr/utils/cache.py`

**Purpose**: Cache node lists and mappings for deterministic iteration.

#### Cache Keys

| Key Prefix | Purpose | Content | Invalidation |
|-----------|---------|---------|--------------|
| `_node_list_cache` | Node tuple and metadata | `NodeCache` object | Node set changes |
| `_node_list_checksum` | Node set digest | BLAKE2b hex | Node additions/removals |
| `NODE_SET_CHECKSUM_KEY` | Cached checksum token | tuple | Node set changes |

#### Implementation Notes

- Maintains stable node ordering for reproducibility
- Supports optional sorted order via `SORT_NODES` graph attribute
- Tracks node count for fast invalidation
- Checksums derived from stable JSON representations

## Buffer Management

### Unified Buffer Cache

**Module**: `src/tnfr/metrics/buffer_cache.py`

**Function**: `ensure_numpy_buffers(G, key_prefix, count, buffer_count, np, dtype, max_cache_entries)`

**Purpose**: Centralized buffer allocation with edge-versioned invalidation.

#### Design Principles

1. **Consolidation**: Single function replaces ad-hoc buffer creation patterns
2. **Consistency**: Uniform cache key structure `(key_prefix, count, buffer_count)`
3. **Invalidation**: Automatic cleanup on edge structure changes
4. **Capacity**: Configurable `max_cache_entries` per graph

#### Usage Pattern

```python
# Allocate 3 buffers of 100 elements each
buffers = ensure_numpy_buffers(
    G, 
    key_prefix="_my_computation",
    count=100,
    buffer_count=3,
    np=np
)
buf1, buf2, buf3 = buffers
```

## Cache Key Naming Conventions

### Prefix Structure

| Pattern | Meaning | Example |
|---------|---------|---------|
| `_<metric>_buffers` | Computation buffers | `_si_buffers` |
| `_<metric>_structural` | Structural arrays | `_si_structural` |
| `_<metric>_edges` | Edge index arrays | `_si_edges` |
| `_<entity>_cache` | Entity cache object | `_node_list_cache` |
| `_<entity>_checksum` | Validation digest | `_node_list_checksum` |
| `_<metric>_version` | Version counter | `_trig_version` |

### Collision Avoidance

- **Namespacing**: Use specific prefixes per computation (`_si_`, `_dnfr_`, etc.)
- **Tuple Keys**: Combine prefix with parameters `(prefix, param1, param2, ...)`
- **Documentation**: All cache keys documented in this file

## Cache Invalidation Strategies

### Edge Version Tracking

**Mechanism**: Graph maintains `_edge_version` counter incremented on topology changes.

**Affected Caches**:
- All `edge_version_cache` calls
- Buffer caches via `ensure_numpy_buffers`
- Trigonometric cache
- Neighbor topology cache
- ΔNFR preparation cache

**Trigger Functions**:
- `increment_edge_version(G)` - Manual invalidation
- `edge_version_update(G)` - Context manager for batch updates
- Graph edge add/remove operations (via hooks)

### Attribute Change Detection

**Mechanism**: Caches store checksums of relevant attributes and compare on access.

**Implementation**:
- `TrigCache.theta_checksums`: BLAKE2b(theta) per node
- `_SiStructuralCache`: Snapshot comparison for νf and ΔNFR
- `NodeCache`: BLAKE2b of sorted node representations

### Manual Invalidation

**Functions**:
- `mark_dnfr_prep_dirty(G)` - Invalidate ΔNFR cache
- `clear_node_repr_cache()` - Clear node representation cache
- `CacheManager.clear(name)` - Reset specific cache entry

## Performance Considerations

### Cache Hit Rates

Optimal cache strategies prioritize:
1. **High reuse**: Buffers reused across multiple computation steps
2. **Low churn**: Stable cache entries between topology changes
3. **Small keys**: Lightweight key comparison for fast lookups

### Memory Trade-offs

| Cache Type | Memory Cost | Benefit | Recommendation |
|-----------|-------------|---------|----------------|
| Buffer cache | O(n) per buffer set | Avoids repeated allocation | Always enabled |
| Structural arrays | O(n) | Eliminates attribute iteration | Always enabled |
| Edge arrays | O(m) | Eliminates neighbor traversal | Enable for dense graphs |
| Trig cache | O(n) | Avoids transcendental functions | Always enabled |

### Chunking Strategy

For large graphs, Si and ΔNFR computations use chunked processing:
- Chunk size determined by `resolve_chunk_size()`
- Balances memory pressure with parallelism overhead
- Configurable via graph attributes (e.g., `SI_CHUNK_SIZE`)

## Testing and Validation

### Cache Coherence Tests

Required test coverage:
1. **Invalidation correctness**: Cache cleared on relevant mutations
2. **Key uniqueness**: No collisions between different computations
3. **Determinism**: Same graph state yields same cache contents
4. **Hit rate tracking**: Telemetry correctly records hits/misses

### Benchmark Targets

Hot path benchmarks in `benchmarks/`:
- `full_pipeline_profile.py`: End-to-end Si + ΔNFR
- `compute_si_profile.py`: Isolated Si computation
- `compute_dnfr_benchmark.py`: ΔNFR preparation overhead

## Future Enhancements

### Potential Optimizations

1. **Persistent caching**: Save expensive computations across sessions via shelve/Redis
2. **Hierarchical invalidation**: Fine-grained cache clearing (e.g., only phase-dependent caches)
3. **Lazy materialization**: Defer buffer allocation until actually needed
4. **Adaptive sizing**: Dynamic buffer capacity based on hit rate telemetry

### Monitoring Integration

Proposed telemetry:
- Per-cache hit/miss ratios
- Invalidation frequency per cache type
- Memory consumption per cache layer
- Cache operation latencies

## Reference

### Key Modules

- `src/tnfr/utils/cache.py` - Core infrastructure
- `src/tnfr/metrics/buffer_cache.py` - Buffer management
- `src/tnfr/metrics/trig_cache.py` - Trigonometric caching
- `src/tnfr/metrics/sense_index.py` - Si computation caches
- `src/tnfr/metrics/coherence.py` - Coherence matrix storage
- `src/tnfr/dynamics/dnfr.py` - ΔNFR preparation cache

### External Dependencies

- `cachetools.LRUCache` - LRU eviction policy
- `threading.RLock` - Cache access synchronization
- `hashlib.blake2b` - Fast checksumming
- `shelve` (optional) - Persistent storage
- `redis` (optional) - Distributed caching

---

**Last Updated**: 2025-11-03  
**Maintainer**: TNFR Core Team  
**Related**: ARCHITECTURE.md, CONTRIBUTING.md
