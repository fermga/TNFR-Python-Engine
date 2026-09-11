"""Exact-state cache shared by read-only TNFR computation engines.

Keys include graph identity, the complete logical graph snapshot, entry type,
and explicit parameters. Stored values and cache hits are detached copies.
Optional access tracking records hints only; it does not claim speculative
execution or mathematical dependency equivalence.
"""

import hashlib
import math
import time
import weakref
from functools import wraps
from collections import OrderedDict
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

from ..mathematics.unified_numerical import np

try:
    import networkx as nx

    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False
    nx = None

# Operational engine-tuning knobs (not TNFR physics) → tnfr.constants.operational
from ..constants.operational import (
    MULTIMODAL_CACHE_SPECTRAL_IMPORTANCE_CANONICAL,
    MULTIMODAL_CACHE_TARGET_CANONICAL,
    MULTIMODAL_CACHE_TETRAD_IMPORTANCE_CANONICAL,
)


_RUNTIME_GRAPH_KEYS = frozenset(
    {
        "_node_cache",
        "_node_cache_weak",
        "integrity_monitor",
        "runtime_lock",
    }
)


def _logical_graph_attributes(attributes: Mapping[Any, Any]) -> dict[Any, Any]:
    """Exclude only opaque runtime attachments from a logical cache snapshot."""

    return {
        key: value
        for key, value in attributes.items()
        if key not in _RUNTIME_GRAPH_KEYS
        and "cache" not in str(key).lower()
        and "lock" not in str(key).lower()
    }


def cache_signature_token(value: Any, seen: set[int] | None = None) -> Any:
    """Build a deterministic, shape-aware token for nested in-process values."""

    if seen is None:
        seen = set()
    if value is None or isinstance(value, (bool, int, str, bytes)):
        return (type(value).__name__, value)
    if isinstance(value, float):
        return ("float", value.hex())
    if isinstance(value, complex):
        return ("complex", value.real.hex(), value.imag.hex())
    if isinstance(value, np.generic):
        return cache_signature_token(value.item(), seen)
    if isinstance(value, np.ndarray):
        array = np.asarray(value)
        if array.dtype.kind == "O":
            return (
                "object-array",
                array.shape,
                tuple(cache_signature_token(item, seen) for item in array.flat),
            )
        contiguous = np.ascontiguousarray(array)
        return (
            "array",
            array.dtype.str,
            array.shape,
            hashlib.sha256(contiguous.tobytes()).hexdigest(),
        )

    object_id = id(value)
    if object_id in seen:
        return ("cycle", type(value).__module__, type(value).__qualname__)
    seen.add(object_id)
    try:
        if isinstance(value, Mapping):
            items = [
                (
                    cache_signature_token(key, seen),
                    cache_signature_token(item, seen),
                )
                for key, item in value.items()
            ]
            return ("mapping", tuple(sorted(items, key=repr)))
        if isinstance(value, (list, tuple)):
            return (
                type(value).__name__,
                tuple(cache_signature_token(item, seen) for item in value),
            )
        if isinstance(value, (set, frozenset)):
            items = [cache_signature_token(item, seen) for item in value]
            return (type(value).__name__, tuple(sorted(items, key=repr)))
        if hasattr(value, "nodes") and callable(value.nodes):
            return _graph_state_token(value, seen)
        return (
            "object",
            type(value).__module__,
            type(value).__qualname__,
            repr(value),
        )
    finally:
        seen.remove(object_id)


def cache_signature_digest(value: Any) -> str:
    """Hash a deterministic type- and shape-aware cache-key token."""

    token = cache_signature_token(value)
    return hashlib.sha256(repr(token).encode("utf-8")).hexdigest()


def _graph_state_token(graph: Any, seen: set[int] | None = None) -> Any:
    """Snapshot node order, full node data, edges, and logical graph metadata."""

    if seen is None:
        seen = set()
    directed = bool(graph.is_directed())
    multigraph = bool(graph.is_multigraph())
    nodes = tuple(
        (
            cache_signature_token(node, seen),
            cache_signature_token(dict(data), seen),
        )
        for node, data in graph.nodes(data=True)
    )
    if multigraph:
        edges = tuple(
            (
                cache_signature_token(left, seen),
                cache_signature_token(right, seen),
                cache_signature_token(key, seen),
                cache_signature_token(dict(data), seen),
            )
            for left, right, key, data in graph.edges(keys=True, data=True)
        )
    else:
        edges = tuple(
            (
                cache_signature_token(left, seen),
                cache_signature_token(right, seen),
                cache_signature_token(dict(data), seen),
            )
            for left, right, data in graph.edges(data=True)
        )
    return (
        "graph",
        type(graph).__module__,
        type(graph).__qualname__,
        directed,
        multigraph,
        cache_signature_token(
            _logical_graph_attributes(graph.graph),
            seen,
        ),
        nodes,
        edges,
    )

class CacheEntryType(Enum):
    """Types of cached computations."""

    SPECTRAL_DECOMPOSITION = "spectral_decomp"  # Eigenvalues/eigenvectors
    NODAL_STATE = "nodal_state"  # EPI, νf, phase, ΔNFR
    STRUCTURAL_FIELDS = "structural_fields"  # Φ_s, |∇φ|, K_φ, ξ_C
    TEMPORAL_TRAJECTORY = "temporal_traj"  # Multi-step evolution
    TOPOLOGY_ANALYSIS = "topology_analysis"  # Graph structure metrics
    OPERATOR_SEQUENCE = "operator_sequence"  # Applied operator results
    CROSS_CORRELATION = "cross_correlation"  # Inter-field correlations
    FFT_OPERATION = "fft_operation"  # FFT arithmetic artifacts


class CacheInvalidationTrigger(Enum):
    """Events that trigger cache invalidation."""

    TOPOLOGY_CHANGE = "topology_change"  # Graph structure modified
    NODE_PARAMETER_CHANGE = "node_param_change"  # EPI, νf, phase modified
    EDGE_WEIGHT_CHANGE = "edge_weight_change"  # Edge weights modified
    OPERATOR_APPLICATION = "operator_applied"  # Structural operator applied
    TIME_EVOLUTION = "time_evolution"  # Temporal step taken


@dataclass
class CacheEntry:
    """Entry in the unified cache."""

    entry_type: CacheEntryType
    data: Any
    graph_signature: str
    graph_identity: int
    timestamp: float
    access_count: int = 0
    last_access: float = field(default_factory=time.time)
    mathematical_importance: float = 1.0  # Higher = more important to keep
    dependencies: set[CacheEntryType] = field(default_factory=set)
    size_mb: float = 0.0
    computation_time: float = 0.0  # Time it took to compute


@dataclass
class CacheStatistics:
    """Statistics for cache performance."""

    total_entries: int = 0
    total_size_mb: float = 0.0
    hit_rate: float = 0.0
    miss_rate: float = 0.0
    eviction_count: int = 0
    invalidation_count: int = 0
    cache_hit_count: int = 0
    # Retained for serialized-statistics compatibility; engine identity is not tracked.
    cross_engine_reuse_count: int = 0
    memory_pressure_events: int = 0


class TNFRUnifiedMultiModalCache:
    """Cache declared read-only results by exact graph state and parameters.

    Reuse is valid only when the entry type, graph object, complete logical
    snapshot, and explicit parameters match. The cache does not infer that two
    different computations share an artifact merely because both concern TNFR.
    """

    def __init__(self, max_size_mb: float = 512.0, enable_prefetching: bool = True):
        if isinstance(max_size_mb, bool):
            raise ValueError("max_size_mb must be a positive finite scalar")
        self.max_size_mb = float(max_size_mb)
        if not math.isfinite(self.max_size_mb) or self.max_size_mb <= 0.0:
            raise ValueError("max_size_mb must be a positive finite scalar")
        if not isinstance(enable_prefetching, bool):
            raise ValueError("enable_prefetching must be boolean")
        self.enable_prefetching = enable_prefetching

        # Main cache storage (ordered for LRU)
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()

        # Stable per-object identities make graph-scoped invalidation exact.
        self._graph_identities: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
        self._next_graph_identity = 1

        # Statistics
        self.stats = CacheStatistics()
        self._total_requests = 0
        self._cache_hits = 0

        # Prefetching predictions
        self._access_patterns: dict[str, int] = {}

    def _graph_identity(self, G: Any) -> int:
        """Return a unique live-object token for graph-scoped invalidation."""

        if G is None:
            return 0
        try:
            identity = self._graph_identities.get(G)
        except TypeError:
            return id(G)
        if identity is None:
            identity = self._next_graph_identity
            self._next_graph_identity += 1
            self._graph_identities[G] = identity
        return int(identity)

    def compute_graph_signature(self, G: Any) -> str:
        """Hash the complete current logical state of one graph object."""

        if not HAS_NETWORKX or G is None:
            return "null_graph"
        identity = self._graph_identity(G)
        token = _graph_state_token(G)
        state_hash = cache_signature_digest(token)
        return f"graph-{identity}:{state_hash}"

    def _generate_cache_key(
        self,
        entry_type: CacheEntryType,
        graph_signature: str,
        parameters: dict[str, Any] | None = None,
    ) -> str:
        """Generate unique cache key."""
        parameter_hash = cache_signature_digest(parameters or {})
        return f"{entry_type.value}:{graph_signature}:{parameter_hash}"

    def get(
        self,
        entry_type: CacheEntryType,
        G: Any,
        parameters: dict[str, Any] | None = None,
        computation_func: Callable | None = None,
        mathematical_importance: float = 1.0,
    ) -> Any:
        """Return an isolated cached value or compute and cache one."""
        if not isinstance(entry_type, CacheEntryType):
            raise TypeError("entry_type must be a CacheEntryType")
        if isinstance(mathematical_importance, bool):
            raise ValueError("mathematical_importance must be finite and nonnegative")
        mathematical_importance = float(mathematical_importance)
        if not math.isfinite(mathematical_importance) or mathematical_importance < 0.0:
            raise ValueError("mathematical_importance must be finite and nonnegative")
        self._total_requests += 1

        # Generate cache key
        graph_signature = self.compute_graph_signature(G)
        cache_key = self._generate_cache_key(entry_type, graph_signature, parameters)

        # Check cache
        if cache_key in self._cache:
            entry = self._cache[cache_key]

            # Update access statistics
            entry.access_count += 1
            entry.last_access = time.time()

            # Move to end for LRU
            self._cache.move_to_end(cache_key)

            self._cache_hits += 1
            self.stats.cache_hit_count += 1

            return deepcopy(entry.data)

        # Cache miss - compute if function provided
        if computation_func is None:
            return None

        # Compute the result
        start_time = time.time()
        try:
            result = computation_func()
            computation_time = time.time() - start_time

            # Estimate size (rough approximation)
            size_estimate = self._estimate_size(result)

            # Cache only values that can be isolated from their callers.
            try:
                cached_snapshot = deepcopy(result)
            except Exception:
                return result

            entry = CacheEntry(
                entry_type=entry_type,
                data=cached_snapshot,
                graph_signature=graph_signature,
                graph_identity=self._graph_identity(G),
                timestamp=time.time(),
                mathematical_importance=mathematical_importance,
                size_mb=size_estimate,
                computation_time=computation_time,
            )

            # Store in cache
            self._store_entry(cache_key, entry)

            # Update access patterns for prefetching
            if self.enable_prefetching:
                self._update_access_patterns(entry_type)

            return result

        except Exception:
            # Failed computations never create cache entries.
            raise

    def _store_entry(self, cache_key: str, entry: CacheEntry) -> None:
        """Store entry in cache with size management."""
        # Add to cache
        self._cache[cache_key] = entry

        # Update statistics
        self.stats.total_entries = len(self._cache)
        self.stats.total_size_mb = sum(e.size_mb for e in self._cache.values())

        # Check memory pressure
        if self.stats.total_size_mb > self.max_size_mb:
            self._evict_entries()

    def _evict_entries(self) -> None:
        """Evict entries based on mathematical importance and access patterns."""
        self.stats.memory_pressure_events += 1

        # Calculate eviction scores (lower = evict first)
        eviction_candidates = []

        for key, entry in self._cache.items():
            # Score based on mathematical importance, access patterns, and age
            age_penalty = (time.time() - entry.last_access) / 3600  # Hours since access
            access_bonus = np.log(1 + entry.access_count)
            importance_bonus = entry.mathematical_importance

            eviction_score = importance_bonus + access_bonus - age_penalty
            eviction_candidates.append((eviction_score, key, entry))

        # Sort by score (lowest first)
        eviction_candidates.sort(key=lambda x: x[0])

        # Evict until we're under 80% of max size
        target_size = (
            self.max_size_mb * MULTIMODAL_CACHE_TARGET_CANONICAL
        )  # = 0.74 (operational)

        for score, key, entry in eviction_candidates:
            if self.stats.total_size_mb <= target_size:
                break

            del self._cache[key]
            self.stats.total_size_mb -= entry.size_mb
            self.stats.eviction_count += 1

        self.stats.total_entries = len(self._cache)

    def invalidate(
        self,
        trigger: CacheInvalidationTrigger,
        G: Any | None = None,
        affected_types: set[CacheEntryType] | None = None,
    ) -> int:
        """
        Invalidate cache entries based on mathematical dependencies.

        Returns number of entries invalidated.
        """
        invalidated_count = 0

        # Determine what to invalidate based on trigger
        if trigger in {
            CacheInvalidationTrigger.TOPOLOGY_CHANGE,
            CacheInvalidationTrigger.EDGE_WEIGHT_CHANGE,
        }:
            invalidated_count = self._invalidate_by_types(
                set(CacheEntryType), G
            )

        elif trigger == CacheInvalidationTrigger.NODE_PARAMETER_CHANGE:
            # Node parameter changes affect nodal states and dependent computations
            types_to_invalidate = {
                CacheEntryType.NODAL_STATE,
                CacheEntryType.STRUCTURAL_FIELDS,
                CacheEntryType.TEMPORAL_TRAJECTORY,
                CacheEntryType.OPERATOR_SEQUENCE,
                CacheEntryType.CROSS_CORRELATION,
            }
            invalidated_count = self._invalidate_by_types(types_to_invalidate, G)

        elif trigger == CacheInvalidationTrigger.OPERATOR_APPLICATION:
            # Operator application affects nodal states
            types_to_invalidate = {
                CacheEntryType.NODAL_STATE,
                CacheEntryType.STRUCTURAL_FIELDS,
                CacheEntryType.TEMPORAL_TRAJECTORY,
                CacheEntryType.OPERATOR_SEQUENCE,
                CacheEntryType.CROSS_CORRELATION,
            }
            invalidated_count = self._invalidate_by_types(types_to_invalidate, G)

        elif trigger == CacheInvalidationTrigger.TIME_EVOLUTION:
            types_to_invalidate = {
                CacheEntryType.NODAL_STATE,
                CacheEntryType.STRUCTURAL_FIELDS,
                CacheEntryType.TEMPORAL_TRAJECTORY,
                CacheEntryType.OPERATOR_SEQUENCE,
                CacheEntryType.CROSS_CORRELATION,
            }
            invalidated_count = self._invalidate_by_types(types_to_invalidate, G)

        elif affected_types:
            invalidated_count = self._invalidate_by_types(affected_types, G)

        self.stats.invalidation_count += invalidated_count
        return invalidated_count

    def _invalidate_by_types(
        self, types_to_invalidate: set[CacheEntryType], G: Any | None = None
    ) -> int:
        """Invalidate entries by type, optionally filtered by graph."""
        graph_identity = self._graph_identity(G) if G is not None else None

        keys_to_remove = []
        for key, entry in self._cache.items():
            if entry.entry_type in types_to_invalidate:
                if (
                    graph_identity is None
                    or entry.graph_identity == graph_identity
                ):
                    keys_to_remove.append(key)

        # Remove invalidated entries
        for key in keys_to_remove:
            entry = self._cache[key]
            self.stats.total_size_mb -= entry.size_mb
            del self._cache[key]

        self.stats.total_entries = len(self._cache)
        return len(keys_to_remove)

    def _update_access_patterns(self, entry_type: CacheEntryType) -> None:
        """Record an access hint without claiming or executing prefetch work."""
        pattern_key = entry_type.value
        self._access_patterns[pattern_key] = (
            self._access_patterns.get(pattern_key, 0) + 1
        )

    def _estimate_size(self, data: Any) -> float:
        """Rough estimation of data size in MB."""
        if isinstance(data, np.ndarray):
            return data.nbytes / (1024 * 1024)
        elif isinstance(data, dict):
            # Estimate based on number of items
            return len(data) * 0.001  # 1KB per dict item
        elif isinstance(data, (list, tuple)):
            return len(data) * 0.0001  # 100 bytes per list item
        else:
            return 0.001  # 1KB default

    def get_statistics(self) -> CacheStatistics:
        """Get current cache statistics."""
        if self._total_requests > 0:
            self.stats.hit_rate = self._cache_hits / self._total_requests
            self.stats.miss_rate = 1.0 - self.stats.hit_rate

        return deepcopy(self.stats)

    def clear(self) -> None:
        """Clear all cache entries."""
        self._cache.clear()
        self._graph_identities.clear()
        self._next_graph_identity = 1
        self._access_patterns.clear()

        # Reset statistics
        self.stats = CacheStatistics()
        self._total_requests = 0
        self._cache_hits = 0

    def get_cache_info(self) -> dict[str, Any]:
        """Get detailed cache information."""
        return {
            "total_entries": len(self._cache),
            "total_size_mb": sum(e.size_mb for e in self._cache.values()),
            "max_size_mb": self.max_size_mb,
            "utilization": sum(e.size_mb for e in self._cache.values())
            / self.max_size_mb,
            "entry_types": {
                entry_type.value: sum(
                    1 for e in self._cache.values() if e.entry_type == entry_type
                )
                for entry_type in CacheEntryType
            },
            "access_patterns": dict(self._access_patterns),
            "statistics": deepcopy(self.stats),
        }


# Global unified cache instance
_global_unified_cache: TNFRUnifiedMultiModalCache | None = None


def get_unified_cache() -> TNFRUnifiedMultiModalCache:
    """Get global unified cache instance."""
    global _global_unified_cache

    if _global_unified_cache is None:
        _global_unified_cache = TNFRUnifiedMultiModalCache()

    return _global_unified_cache


def cache_unified_computation(
    entry_type: CacheEntryType,
    mathematical_importance: float = 1.0,
    parameters_func: Callable | None = None,
):
    """
    Decorator for caching unified computations.

    Usage:
    @cache_unified_computation(
        CacheEntryType.SPECTRAL_DECOMPOSITION,
        mathematical_importance=MULTIMODAL_CACHE_SPECTRAL_IMPORTANCE_CANONICAL,
    )
    def compute_spectrum(G):
        # computation here
        return eigenvalues, eigenvectors
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(G: Any, *args, **kwargs) -> Any:
            # Extract parameters for cache key
            params = {}
            if parameters_func:
                params = parameters_func(*args, **kwargs)
            else:
                # Use function arguments as parameters
                params.update(kwargs)
                if args:
                    params["args"] = args

            cache = get_unified_cache()

            return cache.get(
                entry_type=entry_type,
                G=G,
                parameters=params,
                computation_func=lambda: func(G, *args, **kwargs),
                mathematical_importance=mathematical_importance,
            )

        return wrapper

    return decorator


# Convenience functions for common cache operations
def cache_spectral_decomposition(G: Any, computation_func: Callable) -> Any:
    """Cache spectral decomposition with high importance."""
    cache = get_unified_cache()
    return cache.get(
        CacheEntryType.SPECTRAL_DECOMPOSITION,
        G,
        computation_func=computation_func,
        mathematical_importance=MULTIMODAL_CACHE_TETRAD_IMPORTANCE_CANONICAL,
    )


def cache_structural_fields(
    G: Any, computation_func: Callable, field_params: dict[str, Any]
) -> Any:
    """Cache structural field computation."""
    cache = get_unified_cache()
    return cache.get(
        CacheEntryType.STRUCTURAL_FIELDS,
        G,
        parameters=field_params,
        computation_func=computation_func,
        mathematical_importance=MULTIMODAL_CACHE_SPECTRAL_IMPORTANCE_CANONICAL,
    )


def invalidate_after_operator(G: Any, operator_name: str) -> int:
    """Invalidate cache after operator application."""
    cache = get_unified_cache()
    return cache.invalidate(CacheInvalidationTrigger.OPERATOR_APPLICATION, G=G)


def invalidate_after_topology_change(G: Any) -> int:
    """Invalidate cache after topology change."""
    cache = get_unified_cache()
    return cache.invalidate(CacheInvalidationTrigger.TOPOLOGY_CHANGE, G=G)
