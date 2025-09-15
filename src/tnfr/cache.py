"""Core caching utilities shared across TNFR helpers.

This module consolidates structural cache helpers that previously lived in
``tnfr.helpers.cache_utils`` and ``tnfr.helpers.edge_cache``.  The functions
exposed here are responsible for maintaining deterministic node digests,
scoped graph caches guarded by locks, and version counters that keep edge
artifacts in sync with ΔNFR driven updates.
"""

from __future__ import annotations

import hashlib
import threading
from collections import defaultdict
from collections.abc import Callable, Hashable, Iterable
from functools import lru_cache
from typing import Any, TypeVar

from cachetools import LRUCache
import networkx as nx  # type: ignore[import-untyped]

from .graph_utils import get_graph
from .json_utils import json_dumps

T = TypeVar("T")

__all__ = (
    "CacheManager",
    "EdgeCacheManager",
    "LockAwareLRUCache",
    "NODE_SET_CHECKSUM_KEY",
    "clear_node_repr_cache",
    "get_graph_version",
    "increment_graph_version",
    "node_set_checksum",
    "stable_json",
    "_node_repr",
    "_node_repr_digest",
)

# Key used to store the node set checksum in a graph's ``graph`` attribute.
NODE_SET_CHECKSUM_KEY = "_node_set_checksum_cache"


class LockAwareLRUCache(LRUCache[Hashable, Any]):
    """``LRUCache`` that drops per-key locks when evicting items."""

    def __init__(self, maxsize: int, locks: dict[Hashable, threading.RLock]):
        super().__init__(maxsize)
        self._locks: dict[Hashable, threading.RLock] = locks

    def popitem(self) -> tuple[Hashable, Any]:  # type: ignore[override]
        key, value = super().popitem()
        self._locks.pop(key, None)
        return key, value


def _ensure_graph_entry(
    graph: Any,
    key: str,
    factory: Callable[[], T],
    validator: Callable[[Any], bool],
) -> T:
    """Return a validated entry from ``graph`` or create one when missing."""

    value = graph.get(key)
    if not validator(value):
        value = factory()
        graph[key] = value
    return value


def _ensure_lock_mapping(
    graph: Any,
    key: str,
    *,
    lock_factory: Callable[[], threading.RLock] = threading.RLock,
) -> defaultdict[Hashable, threading.RLock]:
    """Ensure ``graph`` holds a ``defaultdict`` of locks under ``key``."""

    return _ensure_graph_entry(
        graph,
        key,
        factory=lambda: defaultdict(lock_factory),
        validator=lambda value: isinstance(value, defaultdict)
        and value.default_factory is lock_factory,
    )


def _prune_locks(
    cache: dict[Hashable, Any] | LRUCache[Hashable, Any] | None,
    locks: dict[Hashable, threading.RLock]
    | defaultdict[Hashable, threading.RLock]
    | None,
) -> None:
    """Drop locks with no corresponding cache entry."""

    if not isinstance(locks, dict):
        return
    cache_keys = cache.keys() if isinstance(cache, dict) else ()
    for key in list(locks.keys()):
        if key not in cache_keys:
            locks.pop(key, None)


def get_graph_version(graph: Any, key: str, default: int = 0) -> int:
    """Return integer version stored in ``graph`` under ``key``."""

    return int(graph.get(key, default))


def increment_graph_version(graph: Any, key: str) -> int:
    """Increment and store a version counter in ``graph`` under ``key``."""

    version = get_graph_version(graph, key) + 1
    graph[key] = version
    return version


def stable_json(obj: Any) -> str:
    """Return a JSON string with deterministic ordering for ``obj``."""

    return json_dumps(
        obj,
        sort_keys=True,
        ensure_ascii=False,
        to_bytes=False,
    )


@lru_cache(maxsize=1024)
def _node_repr_digest(obj: Any) -> tuple[str, bytes]:
    """Return cached stable representation and digest for ``obj``."""

    repr_ = stable_json(obj)
    digest = hashlib.blake2b(repr_.encode("utf-8"), digest_size=16).digest()
    return repr_, digest


def clear_node_repr_cache() -> None:
    """Clear cached node representations used for checksums."""

    _node_repr_digest.cache_clear()


def _node_repr(n: Any) -> str:
    """Stable representation for node hashing and sorting."""

    return _node_repr_digest(n)[0]


def _iter_node_digests(
    nodes: Iterable[Any], *, presorted: bool
) -> Iterable[bytes]:
    """Yield node digests in a deterministic order."""

    if presorted:
        for node in nodes:
            yield _node_repr_digest(node)[1]
    else:
        for _, digest in sorted(
            (_node_repr_digest(n) for n in nodes), key=lambda x: x[0]
        ):
            yield digest


def _node_set_checksum_no_nodes(
    G: nx.Graph,
    graph: Any,
    *,
    presorted: bool,
    store: bool,
) -> str:
    """Checksum helper when no explicit node set is provided."""

    nodes_view = G.nodes()
    current_nodes = frozenset(nodes_view)
    cached = graph.get(NODE_SET_CHECKSUM_KEY)
    if cached and len(cached) == 3 and cached[2] == current_nodes:
        return cached[1]

    hasher = hashlib.blake2b(digest_size=16)
    for digest in _iter_node_digests(nodes_view, presorted=presorted):
        hasher.update(digest)

    checksum = hasher.hexdigest()
    if store:
        token = checksum[:16]
        if cached and cached[0] == token:
            return cached[1]
        graph[NODE_SET_CHECKSUM_KEY] = (token, checksum, current_nodes)
    return checksum


def node_set_checksum(
    G: nx.Graph,
    nodes: Iterable[Any] | None = None,
    *,
    presorted: bool = False,
    store: bool = True,
) -> str:
    """Return a BLAKE2b checksum of ``G``'s node set."""

    graph = get_graph(G)
    if nodes is None:
        return _node_set_checksum_no_nodes(
            G, graph, presorted=presorted, store=store
        )

    hasher = hashlib.blake2b(digest_size=16)
    for digest in _iter_node_digests(nodes, presorted=presorted):
        hasher.update(digest)

    checksum = hasher.hexdigest()
    if store:
        token = checksum[:16]
        cached = graph.get(NODE_SET_CHECKSUM_KEY)
        if cached and cached[0] == token:
            return cached[1]
        graph[NODE_SET_CHECKSUM_KEY] = (token, checksum)
    return checksum


class CacheManager:
    """Coordinate cache stores and per-key locks for graph-level caches."""

    _LOCK = threading.RLock()

    def __init__(self, graph: Any, cache_key: str, locks_key: str) -> None:
        self.graph = graph
        self.cache_key = cache_key
        self.locks_key = locks_key

    def _validator(self, max_entries: int | None) -> Callable[[Any], bool]:
        if max_entries is None:
            return lambda value: value is not None and not isinstance(value, LRUCache)
        return lambda value: isinstance(value, LRUCache) and value.maxsize == max_entries

    def _factory(
        self,
        max_entries: int | None,
        locks: dict[Hashable, threading.RLock]
        | defaultdict[Hashable, threading.RLock],
    ) -> dict[Hashable, Any] | LRUCache[Hashable, Any]:
        if max_entries:
            return LockAwareLRUCache(max_entries, locks)  # type: ignore[arg-type]
        return {}

    def get_cache(
        self,
        max_entries: int | None,
        *,
        create: bool = True,
    ) -> tuple[
        dict[Hashable, Any] | LRUCache[Hashable, Any] | None,
        dict[Hashable, threading.RLock]
        | defaultdict[Hashable, threading.RLock]
        | None,
    ]:
        """Return the cache and lock mapping for ``graph``."""

        with self._LOCK:
            if not create:
                cache = self.graph.get(self.cache_key)
                locks = self.graph.get(self.locks_key)
                return cache, locks

            locks = _ensure_lock_mapping(self.graph, self.locks_key)
            cache = _ensure_graph_entry(
                self.graph,
                self.cache_key,
                factory=lambda: self._factory(max_entries, locks),
                validator=self._validator(max_entries),
            )
            if max_entries is None:
                _prune_locks(cache, locks)
            return cache, locks


class EdgeCacheManager(CacheManager):
    """Cache manager specialised for edge version caches."""

    def __init__(self, graph: Any) -> None:
        super().__init__(graph, "_edge_version_cache", "_edge_version_cache_locks")
