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
from contextlib import contextmanager
from functools import lru_cache
from dataclasses import dataclass
from typing import Any, TypeVar

from cachetools import LRUCache
import networkx as nx  # type: ignore[import-untyped]

from .graph import get_graph, mark_dnfr_prep_dirty
from .init import get_logger, get_numpy
from .io import json_dumps

T = TypeVar("T")

__all__ = (
    "EdgeCacheManager",
    "LockAwareLRUCache",
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
)

# Key used to store the node set checksum in a graph's ``graph`` attribute.
NODE_SET_CHECKSUM_KEY = "_node_set_checksum_cache"

logger = get_logger(__name__)

# Keys of cache entries dependent on the edge version. Any change to the edge
# set requires these to be dropped to avoid stale data.
EDGE_VERSION_CACHE_KEYS = ("_trig_version",)


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

    try:
        repr_ = stable_json(obj)
    except TypeError:
        repr_ = repr(obj)
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
    else:
        graph.pop(NODE_SET_CHECKSUM_KEY, None)
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
    else:
        graph.pop(NODE_SET_CHECKSUM_KEY, None)
    return checksum


@dataclass(slots=True)
class NodeCache:
    """Container for cached node data."""

    checksum: str
    nodes: tuple[Any, ...]
    sorted_nodes: tuple[Any, ...] | None = None
    idx: dict[Any, int] | None = None
    offset: dict[Any, int] | None = None

    @property
    def n(self) -> int:
        return len(self.nodes)


def _update_node_cache(
    graph: Any,
    nodes: tuple[Any, ...],
    key: str,
    *,
    checksum: str,
    sorted_nodes: tuple[Any, ...] | None = None,
) -> None:
    """Store ``nodes`` and ``checksum`` in ``graph`` under ``key``."""

    graph[f"{key}_cache"] = NodeCache(
        checksum=checksum, nodes=nodes, sorted_nodes=sorted_nodes
    )
    graph[f"{key}_checksum"] = checksum


def _refresh_node_list_cache(
    G: nx.Graph,
    graph: Any,
    *,
    sort_nodes: bool,
    current_n: int,
) -> tuple[Any, ...]:
    """Refresh the cached node list and return the nodes."""

    nodes = tuple(G.nodes())
    checksum = node_set_checksum(G, nodes, store=True)
    sorted_nodes = tuple(sorted(nodes, key=_node_repr)) if sort_nodes else None
    _update_node_cache(
        graph,
        nodes,
        "_node_list",
        checksum=checksum,
        sorted_nodes=sorted_nodes,
    )
    graph["_node_list_len"] = current_n
    return nodes


def _reuse_node_list_cache(
    graph: Any,
    cache: NodeCache,
    nodes: tuple[Any, ...],
    sorted_nodes: tuple[Any, ...] | None,
    *,
    sort_nodes: bool,
    new_checksum: str | None,
) -> None:
    """Reuse existing node cache and record its checksum if missing."""

    checksum = cache.checksum if new_checksum is None else new_checksum
    if sort_nodes and sorted_nodes is None:
        sorted_nodes = tuple(sorted(nodes, key=_node_repr))
    _update_node_cache(
        graph,
        nodes,
        "_node_list",
        checksum=checksum,
        sorted_nodes=sorted_nodes,
    )


def _cache_node_list(G: nx.Graph) -> tuple[Any, ...]:
    """Cache and return the tuple of nodes for ``G``."""

    graph = get_graph(G)
    cache: NodeCache | None = graph.get("_node_list_cache")
    nodes = cache.nodes if cache else None
    sorted_nodes = cache.sorted_nodes if cache else None
    stored_len = graph.get("_node_list_len")
    current_n = G.number_of_nodes()
    dirty = bool(graph.pop("_node_list_dirty", False))

    invalid = nodes is None or stored_len != current_n or dirty
    new_checksum: str | None = None

    if not invalid and cache:
        new_checksum = node_set_checksum(G)
        invalid = cache.checksum != new_checksum

    sort_nodes = bool(graph.get("SORT_NODES", False))

    if invalid:
        nodes = _refresh_node_list_cache(
            G, graph, sort_nodes=sort_nodes, current_n=current_n
        )
    elif cache and "_node_list_checksum" not in graph:
        _reuse_node_list_cache(
            graph,
            cache,
            nodes,
            sorted_nodes,
            sort_nodes=sort_nodes,
            new_checksum=new_checksum,
        )
    else:
        if sort_nodes and sorted_nodes is None and cache is not None:
            cache.sorted_nodes = tuple(sorted(nodes, key=_node_repr))
    return nodes


def cached_node_list(G: nx.Graph) -> tuple[Any, ...]:
    """Public wrapper returning the cached node tuple for ``G``."""

    return _cache_node_list(G)


def _ensure_node_map(
    G,
    *,
    attrs: tuple[str, ...],
    sort: bool = False,
) -> dict[Any, int]:
    """Return cached node-to-index/offset mappings stored on ``NodeCache``."""

    graph = G.graph
    _cache_node_list(G)
    cache: NodeCache = graph["_node_list_cache"]

    missing = [attr for attr in attrs if getattr(cache, attr) is None]
    if missing:
        if sort:
            nodes = cache.sorted_nodes
            if nodes is None:
                nodes = cache.sorted_nodes = tuple(
                    sorted(cache.nodes, key=_node_repr)
                )
        else:
            nodes = cache.nodes
        mappings: dict[str, dict[Any, int]] = {attr: {} for attr in missing}
        for idx, node in enumerate(nodes):
            for attr in missing:
                mappings[attr][node] = idx
        for attr in missing:
            setattr(cache, attr, mappings[attr])
    return getattr(cache, attrs[0])


def ensure_node_index_map(G) -> dict[Any, int]:
    """Return cached node-to-index mapping for ``G``."""

    return _ensure_node_map(G, attrs=("idx",), sort=False)


def ensure_node_offset_map(G) -> dict[Any, int]:
    """Return cached node-to-offset mapping for ``G``."""

    sort = bool(G.graph.get("SORT_NODES", False))
    return _ensure_node_map(G, attrs=("offset",), sort=sort)


class EdgeCacheManager:
    """Coordinate cache storage and per-key locks for edge version caches."""

    _LOCK = threading.RLock()

    def __init__(self, graph: Any) -> None:
        self.graph = graph
        self.cache_key = "_edge_version_cache"
        self.locks_key = "_edge_version_cache_locks"

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
        """Return the cache and lock mapping for the manager's graph."""

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


def edge_version_cache(
    G: Any,
    key: Hashable,
    builder: Callable[[], T],
    *,
    max_entries: int | None = 128,
) -> T:
    """Return cached ``builder`` output tied to the edge version of ``G``."""

    if max_entries is not None:
        max_entries = int(max_entries)
        if max_entries < 0:
            raise ValueError("max_entries must be non-negative or None")
    if max_entries is not None and max_entries == 0:
        return builder()

    graph = get_graph(G)
    manager = graph.get("_edge_cache_manager")  # type: ignore[assignment]
    if not isinstance(manager, EdgeCacheManager) or manager.graph is not graph:
        manager = EdgeCacheManager(graph)
        graph["_edge_cache_manager"] = manager

    cache, locks = manager.get_cache(max_entries)
    edge_version = get_graph_version(graph, "_edge_version")
    lock = locks[key]

    with lock:
        entry = cache.get(key)
        if entry is not None and entry[0] == edge_version:
            return entry[1]

    try:
        value = builder()
    except (RuntimeError, ValueError) as exc:  # pragma: no cover - logging side effect
        logger.exception("edge_version_cache builder failed for %r: %s", key, exc)
        raise
    else:
        with lock:
            entry = cache.get(key)
            if entry is not None and entry[0] == edge_version:
                return entry[1]
            cache[key] = (edge_version, value)
            return value


def cached_nodes_and_A(
    G: nx.Graph, *, cache_size: int | None = 1, require_numpy: bool = False
) -> tuple[tuple[Any, ...], Any]:
    """Return cached nodes tuple and adjacency matrix for ``G``."""

    nodes = cached_node_list(G)
    graph = G.graph

    checksum = getattr(graph.get("_node_list_cache"), "checksum", None)
    if checksum is None:
        checksum = graph.get("_node_list_checksum")
    if checksum is None:
        node_set_cache = graph.get(NODE_SET_CHECKSUM_KEY)
        if isinstance(node_set_cache, tuple) and len(node_set_cache) >= 2:
            checksum = node_set_cache[1]
    if checksum is None:
        checksum = ""

    key = f"_dnfr_{len(nodes)}_{checksum}"
    graph["_dnfr_nodes_checksum"] = checksum

    def builder() -> tuple[tuple[Any, ...], Any]:
        np = get_numpy()
        if np is None:
            return nodes, None
        A = nx.to_numpy_array(G, nodelist=nodes, weight=None, dtype=float)
        return nodes, A

    nodes, A = edge_version_cache(G, key, builder, max_entries=cache_size)

    if require_numpy and A is None:
        raise RuntimeError("NumPy is required for adjacency caching")

    return nodes, A


def _reset_edge_caches(graph: Any, G: Any) -> None:
    """Clear caches affected by edge updates."""

    cache, locks = EdgeCacheManager(graph).get_cache(None, create=False)
    if isinstance(cache, (dict, LRUCache)):
        cache.clear()
    if isinstance(locks, dict):
        locks.clear()
    mark_dnfr_prep_dirty(G)
    clear_node_repr_cache()
    for key in EDGE_VERSION_CACHE_KEYS:
        graph.pop(key, None)


def increment_edge_version(G: Any) -> None:
    """Increment the edge version counter in ``G.graph``."""

    graph = get_graph(G)
    increment_graph_version(graph, "_edge_version")
    _reset_edge_caches(graph, G)


@contextmanager
def edge_version_update(G: Any):
    """Scope a batch of edge mutations."""

    increment_edge_version(G)
    try:
        yield
    finally:
        increment_edge_version(G)
