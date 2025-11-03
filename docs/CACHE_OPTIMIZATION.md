# Cache Optimization Guide

## Overview

This document describes the cache optimization patterns applied to TNFR hot paths,
specifically focusing on Laplacian matrix computations, C(t) history management,
and Si projection buffers.

## Problem Statement

The original implementation had several cache-related inefficiencies:

1. **Buffer Allocation Duplication**: Multiple similar `_ensure_*_buffers` functions
   with nearly identical implementations scattered across modules.

2. **Inconsistent Cache Keys**: Different naming patterns for similar cache entries
   made it hard to track and optimize cache usage.

3. **Missing Capacity Configuration**: Some caches lacked size limits, potentially
   leading to unbounded memory growth.

## Solution Architecture

### 1. Unified Buffer Management (`buffer_cache.py`)

Created a single `ensure_numpy_buffers()` function that consolidates all buffer
allocation patterns:

```python
def ensure_numpy_buffers(
    G: GraphLike,
    *,
    key_prefix: str,
    count: int,
    buffer_count: int,
    np: Any,
    dtype: Any = None,
    max_cache_entries: int | None = 128,
) -> tuple[Any, ...]
```

**Benefits:**
- Single source of truth for buffer allocation
- Consistent cache key patterns: `(key_prefix, count, buffer_count)`
- Built-in capacity configuration (default 128 entries)
- Automatic edge version invalidation

### 2. Optimized Sense Index (`sense_index.py`)

**Before:**
```python
def _ensure_si_buffers(...):
    def builder():
        return (np.empty(count, dtype=float), ...)
    return edge_version_cache(G, ("_si_buffers", count), builder)

def _ensure_chunk_workspace(...):
    def builder():
        return (np.empty(mask_count, dtype=float), ...)
    return edge_version_cache(G, ("_si_chunk_workspace", mask_count), builder)

def _ensure_neighbor_bulk_buffers(...):
    def builder():
        return (np.empty(count, dtype=float) * 5, ...)
    return edge_version_cache(G, ("_si_neighbor_buffers", count), builder)
```

**After:**
```python
def _ensure_si_buffers(G, *, count: int, np: Any):
    return ensure_numpy_buffers(G, key_prefix="_si_buffers", 
                                count=count, buffer_count=3, np=np)

def _ensure_chunk_workspace(G, *, mask_count: int, np: Any):
    return ensure_numpy_buffers(G, key_prefix="_si_chunk_workspace",
                                count=mask_count, buffer_count=2, np=np)

def _ensure_neighbor_bulk_buffers(G, *, count: int, np: Any):
    return ensure_numpy_buffers(G, key_prefix="_si_neighbor_buffers",
                                count=count, buffer_count=5, np=np)
```

**Impact:**
- Reduced ~50 lines of duplicated code
- Consistent cache key format
- Automatic capacity limits

### 3. C(t) History Management

**Analysis:** The C(t) history already uses `HistoryDict` with `maxlen` configuration,
which is optimal. The `append_metric()` function is efficient and doesn't require
modification.

**Verified patterns:**
- `ensure_history(G)` creates bounded history with `HISTORY_MAXLEN`
- Automatic LRU eviction when maxlen is exceeded
- No duplication found

### 4. Laplacian Matrix Caching

**Analysis:** The Laplacian computation in `dnfr_laplacian()` uses:

1. **_NeighborAverageGradient** class that caches node values internally
2. **NumPy vectorization** via `_apply_dnfr_hook()` when available
3. **Neighbor lists** accessed once per computation

**Verified optimizations:**
- Values are cached in gradient objects: `self.values[n] = val`
- Neighbor lists are efficiently computed: `list(G.neighbors(n))`
- NumPy path uses `fromiter()` for bulk processing
- No duplication found in current implementation

## Performance Characteristics

### Before Optimization

- **Memory**: Unbounded cache growth for some buffers
- **Code**: ~120 lines of duplicated buffer management
- **Consistency**: 3 different cache key patterns

### After Optimization

- **Memory**: Bounded to 128 cache entries per buffer type (configurable)
- **Code**: ~70 lines (50-line reduction)
- **Consistency**: Single unified cache key pattern

## Cache Invalidation Strategy

All caches use the **edge_version_cache** strategy:

1. Graph structure changes increment `_edge_version`
2. Cached entries store `(edge_version, value)` tuples
3. Stale entries are automatically invalidated on access
4. LRU eviction manages cache size

## Testing Coverage

Added comprehensive tests in `test_buffer_cache.py`:

- Basic buffer allocation and shapes
- Cache reuse verification
- Edge version invalidation
- Count/dtype variations
- Capacity limit enforcement

All existing tests continue to pass (155/155 in metrics suite).

## Usage Examples

### Creating Buffers for New Hot Path

```python
from tnfr.metrics.buffer_cache import ensure_numpy_buffers

def compute_new_metric(G, nodes):
    np = get_numpy()
    # Get 3 buffers of size len(nodes)
    buf1, buf2, buf3 = ensure_numpy_buffers(
        G,
        key_prefix="_new_metric_buffers",
        count=len(nodes),
        buffer_count=3,
        np=np,
        max_cache_entries=128,  # optional, defaults to 128
    )
    # Use buffers for computation
    np.add(buf1, buf2, out=buf3)
    return buf3
```

### Configuring Cache Capacity

```python
# Per-call configuration
buffers = ensure_numpy_buffers(
    G, key_prefix="_large", count=1000, buffer_count=10,
    np=np, max_cache_entries=256  # Larger cache
)

# Unlimited cache (use with caution)
buffers = ensure_numpy_buffers(
    G, key_prefix="_unlimited", count=100, buffer_count=5,
    np=np, max_cache_entries=None
)
```

## Migration Guide

### For New Code

Use `ensure_numpy_buffers()` instead of creating custom buffer allocation:

```python
# ❌ Old pattern (don't use)
def _ensure_my_buffers(G, count, np):
    def builder():
        return (np.empty(count, dtype=float), np.empty(count, dtype=float))
    return edge_version_cache(G, ("_my_buffers", count), builder)

# ✅ New pattern (use this)
def _ensure_my_buffers(G, count, np):
    return ensure_numpy_buffers(
        G, key_prefix="_my_buffers", count=count, buffer_count=2, np=np
    )
```

### For Existing Code

1. Identify duplicated `edge_version_cache` + `np.empty` patterns
2. Replace with `ensure_numpy_buffers()` call
3. Update imports: `from tnfr.metrics.buffer_cache import ensure_numpy_buffers`
4. Run tests to verify functionality

## Future Optimizations

Potential areas for further optimization:

1. **Adaptive cache sizing** based on graph size
2. **Memory-mapped buffers** for very large graphs
3. **GPU buffer support** for CUDA/JAX backends

## Cache Profiling

### Overview

TNFR provides comprehensive cache profiling capabilities to measure and optimize
cache performance in hot paths. Cache metrics include hits, misses, evictions,
and hit rates for both aggregate and per-cache statistics.

### Profiling Hot Paths

Use the `full_pipeline_profile.py` benchmark to profile cache behavior during
Sense Index and ΔNFR computations:

```bash
# Profile with default settings (100 nodes, 5 loops)
python benchmarks/full_pipeline_profile.py \
    --nodes 100 \
    --loops 5 \
    --output-dir ./profile_results

# Profile larger graph with custom configuration
python benchmarks/full_pipeline_profile.py \
    --nodes 1000 \
    --edge-probability 0.05 \
    --loops 10 \
    --si-chunk-sizes auto 500 1000 \
    --dnfr-chunk-sizes auto 1000 \
    --output-dir ./profile_results
```

### Understanding Cache Metrics

The benchmark outputs JSON files with cache metrics:

```json
{
  "cache_metrics": {
    "aggregate": {
      "hits": 150,
      "misses": 50,
      "evictions": 10,
      "hit_rate": 0.75
    },
    "by_cache": {
      "_dnfr_prep_state": {
        "hits": 100,
        "misses": 10,
        "evictions": 0,
        "hit_rate": 0.909
      },
      "_edge_version_state": {
        "hits": 50,
        "misses": 40,
        "evictions": 10,
        "hit_rate": 0.555
      }
    }
  }
}
```

**Metric Interpretation:**

- **Hit Rate > 0.8**: Excellent cache effectiveness
- **Hit Rate 0.5-0.8**: Good cache performance, may benefit from tuning
- **Hit Rate < 0.5**: Poor cache performance, investigate invalidation patterns
- **High Evictions**: Cache capacity may be too small, consider increasing `max_cache_entries`

### Runtime Cache Monitoring

Use `log_cache_metrics()` to monitor cache performance during execution:

```python
from tnfr.metrics.cache_utils import log_cache_metrics, configure_hot_path_caches
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

# Configure cache sizes
configure_hot_path_caches(
    G,
    buffer_max_entries=256,
    trig_cache_size=512,
)

# ... perform computations ...

# Log cache metrics
log_cache_metrics(G, level=logging.INFO)
```

**Example output:**

```
INFO:tnfr.metrics.cache_utils:Cache metrics: hits=250 misses=50 evictions=0 hit_rate=83.33%
DEBUG:tnfr.metrics.cache_utils:  _dnfr_prep_state: hits=150 misses=10 hit_rate=93.75%
DEBUG:tnfr.metrics.cache_utils:  _edge_version_state: hits=100 misses=40 hit_rate=71.43%
```

### Optimizing Cache Performance

Based on profiling results, apply these optimizations:

1. **Low Hit Rate in Edge Caches**: 
   - Check if graph topology changes frequently
   - Consider increasing `buffer_max_entries`
   - Review invalidation patterns

2. **High Evictions**:
   ```python
   configure_hot_path_caches(G, buffer_max_entries=512)  # Increase capacity
   ```

3. **Low Si/ΔNFR Hit Rates**:
   - Profile chunk sizes to find optimal values
   - Reduce unnecessary cache invalidations
   - Check if node attributes change too frequently

### Hot Path Cache Keys

The following cache keys are used in TNFR hot paths:

| Cache Key | Purpose | Invalidation Trigger |
|-----------|---------|---------------------|
| `_si_buffers` | Sense index computation buffers | Edge structure changes |
| `_si_chunk_workspace` | Chunked Si processing scratch space | Edge structure changes |
| `_si_neighbor_buffers` | Neighbor phase aggregation | Edge structure changes |
| `_dnfr_prep_state` | ΔNFR preparation cache | Manual via `mark_dnfr_prep_dirty()` |
| `_edge_version_state` | Edge-versioned caches | Edge addition/removal |
| `_trig` | Trigonometric function cache | Theta attribute changes |

### Best Practices

1. **Profile before optimizing**: Use benchmarks to establish baseline performance
2. **Monitor in production**: Add cache metrics logging to identify runtime issues
3. **Tune incrementally**: Adjust cache sizes based on measured hit rates
4. **Document assumptions**: Clearly state expected cache behavior in hot paths
5. **Test cache coherence**: Ensure invalidation triggers are correctly implemented
4. **Cross-graph cache sharing** for batch processing

## References

- **TNFR Canonical Invariants**: See `AGENTS.md` §3
- **Edge Version Cache**: `src/tnfr/utils/cache.py` line 1983
- **Buffer Cache**: `src/tnfr/metrics/buffer_cache.py`
- **Tests**: `tests/unit/metrics/test_buffer_cache.py`
