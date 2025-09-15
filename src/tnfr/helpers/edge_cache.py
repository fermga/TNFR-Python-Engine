from __future__ import annotations

from collections.abc import Callable, Hashable
from contextlib import contextmanager
from typing import Any, TypeVar

from cachetools import LRUCache
import networkx as nx  # type: ignore[import-untyped]

from ..cache import (
    EdgeCacheManager,
    clear_node_repr_cache,
    get_graph_version,
    increment_graph_version,
    node_set_checksum,
)
from ..graph_utils import get_graph, mark_dnfr_prep_dirty
from ..import_utils import get_numpy
from ..logging_utils import get_logger

T = TypeVar("T")

logger = get_logger(__name__)

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
    "increment_edge_version",
    "edge_version_update",
)


def edge_version_cache(
    G: Any,
    key: Hashable,
    builder: Callable[[], T],
    *,
    max_entries: int | None = 128,
    manager: EdgeCacheManager | None = None,
) -> T:
    """Return cached ``builder`` output tied to the edge version of ``G``."""

    if max_entries is not None:
        max_entries = int(max_entries)
        if max_entries < 0:
            raise ValueError("max_entries must be non-negative or None")
    if max_entries is not None and max_entries == 0:
        return builder()

    graph = get_graph(G)
    if manager is None:
        manager = graph.get("_edge_cache_manager")  # type: ignore[assignment]
        if not isinstance(manager, EdgeCacheManager) or manager.graph is not graph:
            manager = EdgeCacheManager(graph)
            graph["_edge_cache_manager"] = manager
    else:
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
) -> tuple[list[int], Any]:
    """Return list of nodes and adjacency matrix for ``G`` with caching."""

    nodes_list = list(G.nodes())
    checksum = node_set_checksum(G, nodes_list, store=False)
    key = f"_dnfr_{len(nodes_list)}_{checksum}"
    G.graph["_dnfr_nodes_checksum"] = checksum

    def builder() -> tuple[list[int], Any]:
        np = get_numpy()
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
