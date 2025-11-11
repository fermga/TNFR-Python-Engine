from __future__ import annotations

import logging
import threading
from collections import defaultdict
from collections.abc import (
    Callable,
    Hashable,
    Iterable,
    Iterator,
    Mapping,
    MutableMapping,
)
from dataclasses import dataclass
from typing import Any, ClassVar, ContextManager, Generic, TypeVar

import networkx as nx
from cachetools import LRUCache

from ..types import GraphLike, NodeId, TimingContext, TNFRGraph

K = TypeVar("K", bound=Hashable)
V = TypeVar("V")
T = TypeVar("T")

class SecurityError(RuntimeError):
    """Raised when a cache payload fails hardened validation."""

    ...

@dataclass(frozen=True)
class CacheCapacityConfig:
    default_capacity: int | None
    overrides: dict[str, int | None]

@dataclass(frozen=True)
class CacheStatistics:
    hits: int = ...
    misses: int = ...
    evictions: int = ...
    total_time: float = ...
    timings: int = ...

    def merge(self, other: CacheStatistics) -> CacheStatistics: ...

class CacheLayer:
    def load(self, name: str) -> Any: ...
    def store(self, name: str, value: Any) -> None: ...
    def delete(self, name: str) -> None: ...
    def clear(self) -> None: ...
    def close(self) -> None: ...

class MappingCacheLayer(CacheLayer):
    def __init__(self, storage: MutableMapping[str, Any] | None = ...) -> None: ...

class ShelveCacheLayer(CacheLayer):
    def __init__(
        self,
        path: str,
        *,
        flag: str = ...,
        protocol: int | None = ...,
        writeback: bool = ...,
    ) -> None: ...

class RedisCacheLayer(CacheLayer):
    def __init__(self, client: Any | None = ..., *, namespace: str = ...) -> None: ...

class CacheManager:
    _MISSING: ClassVar[object]

    def __init__(
        self,
        storage: MutableMapping[str, Any] | None = ...,
        *,
        default_capacity: int | None = ...,
        overrides: Mapping[str, int | None] | None = ...,
        layers: Iterable[CacheLayer] | None = ...,
    ) -> None: ...
    @staticmethod
    def _normalise_capacity(value: int | None) -> int | None: ...
    def register(
        self,
        name: str,
        factory: Callable[[], Any],
        *,
        lock_factory: Callable[[], threading.Lock | threading.RLock] | None = ...,
        reset: Callable[[Any], Any] | None = ...,
        create: bool = ...,
        encoder: Callable[[Any], Any] | None = ...,
        decoder: Callable[[Any], Any] | None = ...,
    ) -> None: ...
    def configure(
        self,
        *,
        default_capacity: int | None | object = ...,
        overrides: Mapping[str, int | None] | None = ...,
        replace_overrides: bool = ...,
    ) -> None: ...
    def configure_from_mapping(self, config: Mapping[str, Any]) -> None: ...
    def export_config(self) -> CacheCapacityConfig: ...
    def get_capacity(
        self,
        name: str,
        *,
        requested: int | None = ...,
        fallback: int | None = ...,
        use_default: bool = ...,
    ) -> int | None: ...
    def has_override(self, name: str) -> bool: ...
    def get_lock(self, name: str) -> threading.Lock | threading.RLock: ...
    def names(self) -> Iterator[str]: ...
    def get(self, name: str, *, create: bool = ...) -> Any: ...
    def peek(self, name: str) -> Any: ...
    def store(self, name: str, value: Any) -> None: ...
    def update(
        self,
        name: str,
        updater: Callable[[Any], Any],
        *,
        create: bool = ...,
    ) -> Any: ...
    def clear(self, name: str | None = ...) -> None: ...
    def increment_hit(
        self,
        name: str,
        *,
        amount: int = ...,
        duration: float | None = ...,
    ) -> None: ...
    def increment_miss(
        self,
        name: str,
        *,
        amount: int = ...,
        duration: float | None = ...,
    ) -> None: ...
    def increment_eviction(self, name: str, *, amount: int = ...) -> None: ...
    def record_timing(self, name: str, duration: float) -> None: ...
    def timer(self, name: str) -> TimingContext: ...
    def get_metrics(self, name: str) -> CacheStatistics: ...
    def iter_metrics(self) -> Iterator[tuple[str, CacheStatistics]]: ...
    def aggregate_metrics(self) -> CacheStatistics: ...
    def register_metrics_publisher(
        self, publisher: Callable[[str, CacheStatistics], None]
    ) -> None: ...
    def publish_metrics(
        self,
        *,
        publisher: Callable[[str, CacheStatistics], None] | None = ...,
    ) -> None: ...
    def log_metrics(self, logger: logging.Logger, *, level: int = ...) -> None: ...

class InstrumentedLRUCache(MutableMapping[K, V], Generic[K, V]):
    _MISSING: ClassVar[object]

    def __init__(
        self,
        maxsize: int,
        *,
        manager: CacheManager | None = ...,
        metrics_key: str | None = ...,
        telemetry_callbacks: Iterable[Callable[[K, V], None]] | Callable[[K, V], None] | None = ...,
        eviction_callbacks: Iterable[Callable[[K, V], None]] | Callable[[K, V], None] | None = ...,
        locks: MutableMapping[K, Any] | None = ...,
        getsizeof: Callable[[V], int] | None = ...,
        count_overwrite_hit: bool = ...,
    ) -> None: ...
    @property
    def telemetry_callbacks(self) -> tuple[Callable[[K, V], None], ...]: ...
    @property
    def eviction_callbacks(self) -> tuple[Callable[[K, V], None], ...]: ...
    def set_telemetry_callbacks(
        self,
        callbacks: Iterable[Callable[[K, V], None]] | Callable[[K, V], None] | None,
        *,
        append: bool = ...,
    ) -> None: ...
    def set_eviction_callbacks(
        self,
        callbacks: Iterable[Callable[[K, V], None]] | Callable[[K, V], None] | None,
        *,
        append: bool = ...,
    ) -> None: ...
    def pop(self, key: K, default: Any = ...) -> V: ...
    def popitem(self) -> tuple[K, V]: ...
    def clear(self) -> None: ...
    @property
    def maxsize(self) -> int: ...
    @property
    def currsize(self) -> int: ...
    def get(self, key: K, default: V | None = ...) -> V | None: ...

class ManagedLRUCache(LRUCache[K, V], Generic[K, V]):
    def __init__(
        self,
        maxsize: int,
        *,
        manager: CacheManager | None = ...,
        metrics_key: str | None = ...,
        eviction_callbacks: Iterable[Callable[[K, V], None]] | Callable[[K, V], None] | None = ...,
        telemetry_callbacks: Iterable[Callable[[K, V], None]] | Callable[[K, V], None] | None = ...,
        locks: MutableMapping[K, Any] | None = ...,
    ) -> None: ...
    def popitem(self) -> tuple[K, V]: ...

def prune_lock_mapping(
    cache: Mapping[K, Any] | MutableMapping[K, Any] | None,
    locks: MutableMapping[K, Any] | None,
) -> None: ...

__all__ = (
    "CacheLayer",
    "CacheManager",
    "CacheCapacityConfig",
    "CacheStatistics",
    "InstrumentedLRUCache",
    "ManagedLRUCache",
    "MappingCacheLayer",
    "RedisCacheLayer",
    "ShelveCacheLayer",
    "prune_lock_mapping",
    "EdgeCacheManager",
    "NODE_SET_CHECKSUM_KEY",
    "cached_node_list",
    "cached_nodes_and_A",
    "clear_node_repr_cache",
    "edge_version_cache",
    "edge_version_update",
    "ensure_node_index_map",
    "ensure_node_offset_map",
    "get_graph_version",
    "increment_edge_version",
    "increment_graph_version",
    "node_set_checksum",
    "stable_json",
    "configure_graph_cache_limits",
    "DNFR_PREP_STATE_KEY",
    "DnfrCache",
    "DnfrPrepState",
    "new_dnfr_cache",
    "build_cache_manager",
    "configure_global_cache_layers",
    "reset_global_cache_manager",
    "_GRAPH_CACHE_LAYERS_KEY",
    "_SeedHashCache",
    "ScopedCounterCache",
)

NODE_SET_CHECKSUM_KEY: str
_GRAPH_CACHE_LAYERS_KEY: str
DNFR_PREP_STATE_KEY: str

class DnfrCache:
    idx: dict[Any, int]
    theta: list[float]
    epi: list[float]
    vf: list[float]
    cos_theta: list[float]
    sin_theta: list[float]
    neighbor_x: list[float]
    neighbor_y: list[float]
    neighbor_epi_sum: list[float]
    neighbor_vf_sum: list[float]
    neighbor_count: list[float]
    neighbor_deg_sum: list[float] | None
    th_bar: list[float] | None
    epi_bar: list[float] | None
    vf_bar: list[float] | None
    deg_bar: list[float] | None
    degs: dict[Any, float] | None
    deg_list: list[float] | None
    theta_np: Any | None
    epi_np: Any | None
    vf_np: Any | None
    cos_theta_np: Any | None
    sin_theta_np: Any | None
    deg_array: Any | None
    edge_src: Any | None
    edge_dst: Any | None
    checksum: Any | None
    neighbor_x_np: Any | None
    neighbor_y_np: Any | None
    neighbor_epi_sum_np: Any | None
    neighbor_vf_sum_np: Any | None
    neighbor_count_np: Any | None
    neighbor_deg_sum_np: Any | None
    th_bar_np: Any | None
    epi_bar_np: Any | None
    vf_bar_np: Any | None
    deg_bar_np: Any | None
    grad_phase_np: Any | None
    grad_epi_np: Any | None
    grad_vf_np: Any | None
    grad_topo_np: Any | None
    grad_total_np: Any | None
    dense_components_np: Any | None
    dense_accum_np: Any | None
    dense_degree_np: Any | None
    neighbor_accum_np: Any | None
    neighbor_inv_count_np: Any | None
    neighbor_cos_avg_np: Any | None
    neighbor_sin_avg_np: Any | None
    neighbor_mean_tmp_np: Any | None
    neighbor_mean_length_np: Any | None
    edge_signature: Any | None
    neighbor_accum_signature: Any | None
    neighbor_edge_values_np: Any | None

class EdgeCacheState:
    cache: MutableMapping[Hashable, Any]
    locks: defaultdict[Hashable, threading.RLock]
    max_entries: int | None
    dirty: bool

def new_dnfr_cache() -> DnfrCache: ...

class DnfrPrepState:
    cache: DnfrCache
    cache_lock: threading.RLock
    vector_lock: threading.RLock

class EdgeCacheManager:
    _STATE_KEY: str

    def __init__(self, graph: MutableMapping[str, Any]) -> None: ...
    def record_hit(self) -> None: ...
    def record_miss(self, *, track_metrics: bool = ...) -> None: ...
    def record_eviction(self, *, track_metrics: bool = ...) -> None: ...
    def timer(self) -> TimingContext: ...
    def _default_state(self) -> EdgeCacheState: ...
    def resolve_max_entries(self, max_entries: int | None | object) -> int | None: ...
    def _build_state(self, max_entries: int | None) -> EdgeCacheState: ...
    def _ensure_state(
        self, state: EdgeCacheState | None, max_entries: int | None | object
    ) -> EdgeCacheState: ...
    def _reset_state(self, state: EdgeCacheState | None) -> EdgeCacheState: ...
    def get_cache(
        self,
        max_entries: int | None | object,
        *,
        create: bool = ...,
    ) -> EdgeCacheState | None: ...
    def flush_state(self, state: EdgeCacheState) -> None: ...
    def clear(self) -> None: ...

def get_graph_version(graph: Any, key: str, default: int = ...) -> int: ...
def increment_graph_version(graph: Any, key: str) -> int: ...
def stable_json(obj: Any) -> str: ...
def clear_node_repr_cache() -> None: ...
def configure_global_cache_layers(
    *,
    shelve: Mapping[str, Any] | None = ...,
    redis: Mapping[str, Any] | None = ...,
    replace: bool = ...,
) -> None: ...
def node_set_checksum(
    G: nx.Graph,
    nodes: Iterable[Any] | None = ...,
    *,
    presorted: bool = ...,
    store: bool = ...,
) -> str: ...
def reset_global_cache_manager() -> None: ...
def build_cache_manager(
    *,
    graph: MutableMapping[str, Any] | None = ...,
    storage: MutableMapping[str, Any] | None = ...,
    default_capacity: int | None = ...,
    overrides: Mapping[str, int | None] | None = ...,
) -> CacheManager: ...
def cached_node_list(G: nx.Graph) -> tuple[Any, ...]: ...
def ensure_node_index_map(G: TNFRGraph) -> dict[NodeId, int]: ...
def ensure_node_offset_map(G: TNFRGraph) -> dict[NodeId, int]: ...
def configure_graph_cache_limits(
    G: GraphLike | TNFRGraph | MutableMapping[str, Any],
    *,
    default_capacity: int | None | object = CacheManager._MISSING,
    overrides: Mapping[str, int | None] | None = ...,
    replace_overrides: bool = ...,
) -> CacheCapacityConfig: ...
def increment_edge_version(G: Any) -> None: ...
def edge_version_cache(
    G: Any,
    key: Hashable,
    builder: Callable[[], T],
    *,
    max_entries: int | None | object = CacheManager._MISSING,
) -> T: ...
def cached_nodes_and_A(
    G: nx.Graph,
    *,
    cache_size: int | None = ...,
    require_numpy: bool = ...,
    prefer_sparse: bool = ...,
    nodes: tuple[Any, ...] | None = ...,
) -> tuple[tuple[Any, ...], Any]: ...
def edge_version_update(G: TNFRGraph) -> ContextManager[None]: ...

class _SeedCacheState:
    cache: InstrumentedLRUCache[tuple[int, int], int] | None
    maxsize: int

class _CounterState(Generic[K]):
    cache: InstrumentedLRUCache[K, int]
    locks: dict[K, threading.RLock]
    max_entries: int

class _SeedHashCache(MutableMapping[tuple[int, int], int]):
    _state_key: str

    def __init__(
        self,
        *,
        manager: CacheManager | None = ...,
        state_key: str = ...,
        default_maxsize: int = ...,
    ) -> None: ...
    def configure(self, maxsize: int) -> None: ...
    def __getitem__(self, key: tuple[int, int]) -> int: ...
    def __setitem__(self, key: tuple[int, int], value: int) -> None: ...
    def __delitem__(self, key: tuple[int, int]) -> None: ...
    def __iter__(self) -> Iterator[tuple[int, int]]: ...
    def __len__(self) -> int: ...
    def clear(self) -> None: ...
    @property
    def maxsize(self) -> int: ...
    @property
    def enabled(self) -> bool: ...
    @property
    def data(self) -> InstrumentedLRUCache[tuple[int, int], int] | None: ...

class ScopedCounterCache(Generic[K]):
    _state_key: str

    def __init__(
        self,
        name: str,
        max_entries: int | None = ...,
        *,
        manager: CacheManager | None = ...,
        default_max_entries: int = ...,
    ) -> None: ...
    def configure(self, *, force: bool = ..., max_entries: int | None = ...) -> None: ...
    def clear(self) -> None: ...
    def bump(self, key: K) -> int: ...
    def __len__(self) -> int: ...
    @property
    def lock(self) -> threading.Lock | threading.RLock: ...
    @property
    def max_entries(self) -> int: ...
    @property
    def cache(self) -> InstrumentedLRUCache[K, int]: ...
    @property
    def locks(self) -> dict[K, threading.RLock]: ...

# Internal symbols used by utils.__init__.py
_GRAPH_CACHE_MANAGER_KEY: str

def _graph_cache_manager(graph: MutableMapping[str, Any]) -> CacheManager: ...
