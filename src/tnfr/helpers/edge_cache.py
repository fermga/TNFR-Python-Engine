from __future__ import annotations

import threading
from collections import defaultdict
from collections.abc import Callable, Hashable
from contextlib import contextmanager
from typing import Any, TypeVar

from cachetools import LRUCache
import networkx as nx  # type: ignore[import-untyped]

from ..graph_utils import mark_dnfr_prep_dirty
from ..import_utils import optional_numpy, cached_import  # noqa: F401 - used in tests
from ..logging import get_module_logger
from .node_cache import get_graph, node_set_checksum, clear_node_repr_cache

T = TypeVar("T")
U = TypeVar("U")

logger = get_module_logger(__name__)

# Keys of cache entries dependent on the edge version. Any change to the edge
# set requires these to be dropped to avoid stale data.
EDGE_VERSION_CACHE_KEYS = (
    "_cos_th",
    "_sin_th",
    "_thetas",
    "_trig_cache",
    "_trig_version",
)

__all__ = (
    "EdgeCacheManager",
    "edge_version_cache",
    "cached_nodes_and_A",
    "invalidate_edge_version_cache",
    "increment_edge_version",
    "edge_version_update",
)


class _LockAwareLRUCache(LRUCache[Hashable, Any]):
    """``LRUCache`` that drops per-key locks when evicting items."""

    def __init__(self, maxsize: int, locks: dict[Hashable, threading.RLock]):
        super().__init__(maxsize)
        self._locks: dict[Hashable, threading.RLock] = locks

    def popitem(self) -> tuple[Hashable, Any]:  # type: ignore[override]
        key, value = super().popitem()
        self._locks.pop(key, None)
        return key, value


def _make_edge_cache(
    max_entries: int, locks: dict[Hashable, threading.RLock]
) -> LRUCache[Hashable, Any]:
    """Create an ``LRUCache`` for edge data."""

    return _LockAwareLRUCache(max_entries, locks)


class EdgeCacheManager:
    """Manage per-graph edge caches and their associated locks."""

    _LOCK = threading.RLock()

    def __init__(self, graph: Any) -> None:
        self.graph = graph

    def _ensure_graph_entry(
        self,
        key: str,
        factory: Callable[[], U],
        validator: Callable[[Any], bool],
    ) -> U:
        """Return a validated entry from ``graph`` or create one when missing."""

        value = self.graph.get(key)
        if not validator(value):
            value = factory()
            self.graph[key] = value
        return value

    def _ensure_locks(self) -> defaultdict[Hashable, threading.RLock]:
        return self._ensure_graph_entry(
            "_edge_version_cache_locks",
            factory=lambda: defaultdict(threading.RLock),
            validator=lambda v: (
                isinstance(v, defaultdict) and v.default_factory is threading.RLock
            ),
        )

    def _init_cache(
        self,
        locks: dict[Hashable, threading.RLock],
        max_entries: int | None,
    ) -> dict[Hashable, tuple[int, Any]] | LRUCache[Hashable, tuple[int, Any]]:
        use_lru = bool(max_entries)

        return self._ensure_graph_entry(
            "_edge_version_cache",
            factory=lambda: (
                _make_edge_cache(max_entries, locks) if use_lru else {}
            ),
            validator=lambda _: False,
        )

    def _maybe_init_cache(
        self,
        locks: dict[Hashable, threading.RLock],
        max_entries: int | None,
    ) -> dict[Hashable, tuple[int, Any]] | LRUCache[Hashable, tuple[int, Any]]:
        use_lru = bool(max_entries)

        def validator(value: Any) -> bool:
            if value is None:
                return False
            if use_lru:
                return isinstance(value, LRUCache) and value.maxsize == max_entries
            return not isinstance(value, LRUCache)

        return self._ensure_graph_entry(
            "_edge_version_cache",
            factory=lambda: (
                _make_edge_cache(max_entries, locks) if use_lru else {}
            ),
            validator=validator,
        )

    def _create_cache(
        self, max_entries: int | None
    ) -> tuple[
        dict[Hashable, tuple[int, Any]] | LRUCache[Hashable, tuple[int, Any]] | None,
        dict[Hashable, threading.RLock]
        | defaultdict[Hashable, threading.RLock]
        | None,
    ]:
        """Ensure locks and initialise the cache when requested."""

        locks = self._ensure_locks()
        cache = self._maybe_init_cache(locks, max_entries)
        return cache, locks

    def _synchronise_locks(
        self, create: bool, max_entries: int | None
    ) -> tuple[
        dict[Hashable, tuple[int, Any]] | LRUCache[Hashable, tuple[int, Any]] | None,
        dict[Hashable, threading.RLock]
        | defaultdict[Hashable, threading.RLock]
        | None,
    ]:
        """Return cache and locks, creating them when requested."""

        if create:
            return self._create_cache(max_entries)
        locks = self.graph.get("_edge_version_cache_locks")
        cache = self.graph.get("_edge_version_cache")
        return cache, locks

    def _prune_locks(
        self,
        cache: dict[Hashable, tuple[int, Any]] | LRUCache[Hashable, tuple[int, Any]] | None,
        locks: dict[Hashable, threading.RLock] | defaultdict[Hashable, threading.RLock] | None,
    ) -> None:
        """Drop locks with no corresponding cache entry."""

        if not isinstance(locks, dict):
            return
        cache_keys = cache.keys() if isinstance(cache, dict) else ()
        for key in list(locks.keys()):
            if key not in cache_keys:
                locks.pop(key, None)

    def get_cache(
        self, max_entries: int | None, *, create: bool = True
    ) -> tuple[
        dict[Hashable, tuple[int, Any]] | LRUCache[Hashable, tuple[int, Any]] | None,
        dict[Hashable, threading.RLock]
        | defaultdict[Hashable, threading.RLock]
        | None,
    ]:
        """Return the edge cache and lock mapping for ``graph``."""

        with self._LOCK:
            cache, locks = self._synchronise_locks(create, max_entries)
            if max_entries is None:
                self._prune_locks(cache, locks)
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
    cache, locks = EdgeCacheManager(graph).get_cache(max_entries)
    edge_version = int(graph.get("_edge_version", 0))
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


def invalidate_edge_version_cache(G: Any) -> None:
    """Clear cached entries associated with ``G``."""

    graph = get_graph(G)
    cache, locks = EdgeCacheManager(graph).get_cache(None, create=False)
    if isinstance(cache, (dict, LRUCache)):
        cache.clear()
    if isinstance(locks, dict):
        locks.clear()


def cached_nodes_and_A(
    G: nx.Graph, *, cache_size: int | None = 1, require_numpy: bool = False
) -> tuple[list[int], Any]:
    """Return list of nodes and adjacency matrix for ``G`` with caching."""

    nodes_list = list(G.nodes())
    checksum = node_set_checksum(G, nodes_list, store=False)
    key = f"_dnfr_{len(nodes_list)}_{checksum}"
    G.graph["_dnfr_nodes_checksum"] = checksum
    def builder() -> tuple[list[int], Any]:
        np = optional_numpy(logger)
        if np is None:
            return nodes_list, None
        A = nx.to_numpy_array(G, nodelist=nodes_list, weight=None, dtype=float)
        return nodes_list, A

    nodes, A = edge_version_cache(G, key, builder, max_entries=cache_size)

    if require_numpy and A is None:
        raise RuntimeError("NumPy is required for adjacency caching")

    return nodes, A


def _reset_edge_caches(graph: Any, G: Any) -> None:
    """Clear caches affected by edge updates."""

    invalidate_edge_version_cache(G)
    mark_dnfr_prep_dirty(G)
    clear_node_repr_cache()
    for key in EDGE_VERSION_CACHE_KEYS:
        graph.pop(key, None)


def increment_edge_version(G: Any) -> None:
    """Increment the edge version counter in ``G.graph``."""

    graph = get_graph(G)
    graph["_edge_version"] = int(graph.get("_edge_version", 0)) + 1
    _reset_edge_caches(graph, G)


@contextmanager
def edge_version_update(G: Any):
    """Scope a batch of edge mutations."""

    increment_edge_version(G)
    try:
        yield
    finally:
        increment_edge_version(G)

