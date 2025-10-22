from __future__ import annotations

import threading
from collections.abc import Callable, Hashable, Iterable, Iterator, Mapping, MutableMapping
from typing import Any, ContextManager, Generic, TypeVar

import networkx as nx

from ..cache import CacheCapacityConfig, CacheManager
from ..types import GraphLike, NodeId, TNFRGraph, TimingContext

K = TypeVar("K", bound=Hashable)
V = TypeVar("V")
T = TypeVar("T")

__all__ = (
    "EdgeCacheManager",
    "NODE_SET_CHECKSUM_KEY",
    "cached_node_list",
    "cached_nodes_and_A",
    "clear_node_repr_cache",
    "configure_graph_cache_limits",
    "edge_version_cache",
    "edge_version_update",
    "ensure_node_index_map",
    "ensure_node_offset_map",
    "get_graph_version",
    "increment_edge_version",
    "increment_graph_version",
    "node_set_checksum",
    "stable_json",
)

NODE_SET_CHECKSUM_KEY: str


class LRUCache(MutableMapping[K, V], Generic[K, V]):
    def __init__(self, maxsize: int = ...) -> None: ...

    def __getitem__(self, __key: K) -> V: ...

    def __setitem__(self, __key: K, __value: V) -> None: ...

    def __delitem__(self, __key: K) -> None: ...

    def __iter__(self) -> Iterator[K]: ...

    def __len__(self) -> int: ...


class EdgeCacheState:
    cache: MutableMapping[Hashable, Any]
    locks: MutableMapping[Hashable, threading.RLock]
    max_entries: int | None


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
    ) -> tuple[
        MutableMapping[Hashable, Any] | None,
        MutableMapping[Hashable, threading.RLock] | None,
    ]: ...

    def clear(self) -> None: ...


def get_graph_version(graph: Any, key: str, default: int = ...) -> int: ...


def increment_graph_version(graph: Any, key: str) -> int: ...


def stable_json(obj: Any) -> str: ...


def clear_node_repr_cache() -> None: ...


def node_set_checksum(
    G: nx.Graph,
    nodes: Iterable[Any] | None = ...,
    *,
    presorted: bool = ...,
    store: bool = ...,
) -> str: ...


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
