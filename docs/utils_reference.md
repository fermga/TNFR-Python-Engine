# TNFR Utilities Reference

This document provides a comprehensive reference for the centralized utility functions in `tnfr.utils`.

## Overview

The `tnfr.utils` package serves as the single point of access for generic helper functions used throughout the TNFR engine. All utilities follow TNFR structural semantics and maintain deterministic, traceable behavior.

## Module Organization

### Numeric Helpers (`tnfr.utils.numeric`)

**Purpose**: Compensated arithmetic, angle operations, and clamping utilities that preserve structural integrity.

- `clamp(x, a, b)` - Clamp value to interval [a, b]
- `clamp01(x)` - Clamp to unit interval [0, 1]
- `within_range(val, lower, upper, tol)` - Check if value lies within bounds (with tolerance)
- `similarity_abs(a, b, lo, hi)` - Absolute similarity metric over range
- `kahan_sum_nd(values, dims)` - Compensated summation for multi-dimensional data
- `angle_diff(a, b)` - Minimal angular difference in radians
- `angle_diff_array(a, b, np, out, where)` - Vectorized angle difference (NumPy compatible)

### Cache Infrastructure (`tnfr.utils.cache`)

**Purpose**: Structural cache layers, versioning, and graph-level caching orchestrated by locks.

#### Core Classes

- `CacheManager` - Coordinate named caches with per-entry locks and capacity policies
- `CacheLayer` (ABC) - Abstract storage backend interface
- `MappingCacheLayer` - In-memory cache backed by mutable mapping
- `ShelveCacheLayer` - Persistent cache using `shelve` module
- `RedisCacheLayer` - Distributed cache via Redis client
- `InstrumentedLRUCache` - LRU cache with telemetry and lock synchronization
- `ManagedLRUCache` - Lightweight LRU wrapper with callbacks

#### Configuration & Statistics

- `CacheCapacityConfig` - Immutable capacity policy snapshot
- `CacheStatistics` - Telemetry counters (hits, misses, evictions, timings)

#### Graph-Level Caching

- `cached_node_list(G)` - Return cached node tuple with checksum verification
- `cached_nodes_and_A(G, ...)` - Cache nodes + adjacency matrix
- `edge_version_cache(G, key, builder, ...)` - Version-aware edge cache
- `edge_version_update(G)` - Context manager for batch edge mutations
- `increment_edge_version(G)` - Increment edge version and invalidate caches
- `node_set_checksum(G, nodes, ...)` - BLAKE2b checksum of node set
- `clear_node_repr_cache()` - Clear node representation cache

#### ΔNFR Preparation

- `DnfrCache` - State container for ΔNFR orchestration arrays
- `new_dnfr_cache()` - Factory for empty ΔNFR cache
- `DnfrPrepState` - Coordination bundle (cache + locks)

#### Specialized Caches

- `EdgeCacheManager` - Per-graph edge version cache coordinator
- `ScopedCounterCache` - Thread-safe LRU cache for monotonic counters
- `_SeedHashCache` - Configurable LRU for seed hashing

#### Configuration Functions

- `configure_global_cache_layers(shelve, redis, replace)` - Process-wide layer config
- `configure_graph_cache_limits(G, default_capacity, overrides, ...)` - Per-graph capacity
- `build_cache_manager(graph, storage, ...)` - Construct manager with layers
- `reset_global_cache_manager()` - Dispose shared manager and close layers

#### Utilities

- `stable_json(obj)` - Deterministic JSON with sorted keys
- `prune_lock_mapping(cache, locks)` - Drop orphaned lock entries
- `ensure_node_index_map(G)` - Cached node-to-index mapping
- `ensure_node_offset_map(G)` - Cached node-to-offset mapping (sorted if configured)

### Data Normalization (`tnfr.utils.data`)

**Purpose**: Type conversion, weight normalization, and collection utilities.

- `convert_value(val, target_type, ...)` - Safe type coercion with fallback
- `normalize_optional_int(val, strict)` - Coerce to int or None
- `normalize_weights(weights, fields, ...)` - Normalize and validate weight mappings
- `normalize_counter(counter, ...)` - Normalize Counter to dict with weight checks
- `normalize_materialize_limit(max_materialize)` - Validate materialization limit
- `ensure_collection(obj, ...)` - Coerce to tuple/list, handling strings
- `flatten_structure(obj, ...)` - Recursively flatten nested sequences
- `is_non_string_sequence(obj)` - Check if iterable but not string/bytes
- `mix_groups(groups, weights, rng)` - Weighted sampling from groups
- `negative_weights_warn_once(...)` - Cached warning for negative weights

**Constants**:
- `MAX_MATERIALIZE_DEFAULT` - Default limit for materialization
- `STRING_TYPES` - Tuple of string/bytes types

### IO and Parsing (`tnfr.utils.io`)

**Purpose**: Structured file operations with JSON/YAML/TOML support and atomic writes.

#### JSON Serialization

- `json_dumps(obj, sort_keys, default, ensure_ascii, separators, cls, to_bytes, ...)` - Serialize to JSON using orjson when available
- `JsonDumpsParams` - Immutable parameter container
- `DEFAULT_PARAMS` - Default serialization parameters
- `clear_orjson_param_warnings()` - Reset orjson compatibility warnings

#### File Operations

- `read_structured_file(path)` - Parse JSON/YAML/TOML based on extension
- `safe_write(path, write, mode, encoding, atomic, sync, ...)` - Atomic file write with fsync
- `StructuredFileError` - Exception for file parsing errors

#### Lazy Imports

- `tomllib` / `TOMLDecodeError` - TOML parsing (tomllib or tomli)
- `yaml` / `YAMLError` - YAML parsing (PyYAML)
- `has_toml` - Boolean proxy for TOML availability

### Graph Utilities (`tnfr.utils.graph`)

**Purpose**: Graph metadata access and ΔNFR preparation management.

- `get_graph(G)` - Extract graph attribute dictionary
- `get_graph_mapping(G)` - Return graph as mutable mapping
- `supports_add_edge(G)` - Check if graph supports add_edge
- `mark_dnfr_prep_dirty(G)` - Invalidate ΔNFR preparation cache

### Chunking Utilities (`tnfr.utils.chunks`)

**Purpose**: Determine optimal chunk sizes for parallel operations.

- `auto_chunk_size(n, ...)` - Compute chunk size from collection size
- `resolve_chunk_size(n, requested, ...)` - Resolve chunk size with constraints

### Import and Logging (`tnfr.utils.init`)

**Purpose**: Lazy imports, logging configuration, and import registry.

#### Core Functions

- `cached_import(module, attr, emit, lazy, fallback)` - Cache module/attribute imports
- `warm_cached_import(module, attr, ...)` - Eager variant of cached_import
- `LazyImportProxy` - Proxy that defers import until first access
- `get_logger(name)` - Get configured logger for module
- `get_numpy()` - Lazy NumPy import
- `get_nodenx()` - Lazy node-extended NetworkX import
- `prune_failed_imports(limit)` - Remove failed imports from registry
- `warn_once(logger, message)` - Create one-time warning function
- `WarnOnce` - Callable wrapper for cached warnings

#### Internal State

- `IMPORT_LOG` - Import registry tracking metadata
- `_IMPORT_STATE` - Alias for IMPORT_LOG
- `_LOGGING_CONFIGURED` - Bootstrap flag
- `_configure_root()` - Configure root logger
- `_reset_logging_state()` - Reset logging for tests
- `_reset_import_state()` - Reset import registry
- `_warn_failure(name, ...)` - Log import failure
- `_FAILED_IMPORT_LIMIT` - Maximum failed imports tracked
- `_DEFAULT_CACHE_SIZE` - Default LRU cache size
- `EMIT_MAP` - Emission strategy mapping

## Locking Utilities (`tnfr.locking`)

**Purpose**: Process-wide named locks for coordination.

- `get_lock(name)` - Return or create RLock for name

## Structural Guarantees

All utilities in `tnfr.utils` respect these invariants:

1. **Determinism**: Given identical inputs and seeds, output is reproducible
2. **Traceability**: Operations that modify caches or state are logged at debug level
3. **Phase-awareness**: Caching respects structural frequency (νf) and version counters
4. **Thread-safety**: Where documented, utilities are safe for concurrent access
5. **Minimal mutation**: Prefer returning new objects over mutating inputs

## Import Patterns

```python
# Import individual utilities
from tnfr.utils import clamp, clamp01, json_dumps

# Import specialized submodules
from tnfr.utils import cache, numeric, data, io

# Access via namespace
import tnfr.utils as utils
result = utils.clamp(value, 0, 1)
```

## Testing

All utilities have corresponding tests in `tests/unit/structural/`. Cache-related tests demonstrate:
- Telemetry accuracy
- Lock coordination
- Layer orchestration
- Version invalidation

## Migration Notes

**Removed**: `tnfr.cache` module (legacy shim raises ImportError)

**Canonical location**: All cache, numeric, parsing, and data helpers now live in `tnfr.utils`.

**Stable API**: Functions documented here comprise the stable public interface. Internal helpers prefixed with `_` are subject to change.
