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
from collections.abc import (
    Callable,
    Hashable,
    Iterable,
    Iterator,
    Mapping,
    MutableMapping,
)
from contextlib import contextmanager
from functools import lru_cache
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypeVar, cast

from cachetools import LRUCache
import networkx as nx

from ..cache import CacheCapacityConfig, CacheManager, InstrumentedLRUCache
from ..types import GraphLike, NodeId, TNFRGraph, TimingContext
from .graph import get_graph, mark_dnfr_prep_dirty
from .init import get_logger, get_numpy
from .io import json_dumps

T = TypeVar("T")

__all__ = (
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
    "DnfrPrepState",
)

if TYPE_CHECKING:  # pragma: no cover - typing aide
    from ..dynamics.dnfr import DnfrCache

# Key used to store the node set checksum in a graph's ``graph`` attribute.
NODE_SET_CHECKSUM_KEY = "_node_set_checksum_cache"

logger = get_logger(__name__)

# Keys of cache entries dependent on the edge version. Any change to the edge
# set requires these to be dropped to avoid stale data.
EDGE_VERSION_CACHE_KEYS = ("_trig_version",)


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
    G: TNFRGraph,
    *,
    attrs: tuple[str, ...],
    sort: bool = False,
) -> dict[NodeId, int]:
    """Return cached node-to-index/offset mappings stored on ``NodeCache``."""

    graph = G.graph
    _cache_node_list(G)
    cache: NodeCache = graph["_node_list_cache"]

    missing = [attr for attr in attrs if getattr(cache, attr) is None]
    if missing:
        if sort:
            nodes_opt = cache.sorted_nodes
            if nodes_opt is None:
                nodes_opt = tuple(sorted(cache.nodes, key=_node_repr))
                cache.sorted_nodes = nodes_opt
            nodes_seq = nodes_opt
        else:
            nodes_seq = cache.nodes
        node_ids = cast(tuple[NodeId, ...], nodes_seq)
        mappings: dict[str, dict[NodeId, int]] = {attr: {} for attr in missing}
        for idx, node in enumerate(node_ids):
            for attr in missing:
                mappings[attr][node] = idx
        for attr in missing:
            setattr(cache, attr, mappings[attr])
    return cast(dict[NodeId, int], getattr(cache, attrs[0]))


def ensure_node_index_map(G: TNFRGraph) -> dict[NodeId, int]:
    """Return cached node-to-index mapping for ``G``."""

    return _ensure_node_map(G, attrs=("idx",), sort=False)


def ensure_node_offset_map(G: TNFRGraph) -> dict[NodeId, int]:
    """Return cached node-to-offset mapping for ``G``."""

    sort = bool(G.graph.get("SORT_NODES", False))
    return _ensure_node_map(G, attrs=("offset",), sort=sort)


@dataclass
class EdgeCacheState:
    cache: MutableMapping[Hashable, Any]
    locks: defaultdict[Hashable, threading.RLock]
    max_entries: int | None


_GRAPH_CACHE_MANAGER_KEY = "_tnfr_cache_manager"
_GRAPH_CACHE_CONFIG_KEY = "_tnfr_cache_config"
DNFR_PREP_STATE_KEY = "_dnfr_prep_state"


@dataclass(slots=True)
class DnfrPrepState:
    """State container coordinating ΔNFR preparation caches."""

    cache: "DnfrCache"
    cache_lock: threading.RLock
    vector_lock: threading.RLock


def _new_dnfr_cache() -> "DnfrCache":
    """Return an empty :class:`~tnfr.dynamics.dnfr.DnfrCache` instance."""

    from ..dynamics.dnfr import DnfrCache

    return DnfrCache(
        idx={},
        theta=[],
        epi=[],
        vf=[],
        cos_theta=[],
        sin_theta=[],
        neighbor_x=[],
        neighbor_y=[],
        neighbor_epi_sum=[],
        neighbor_vf_sum=[],
        neighbor_count=[],
        neighbor_deg_sum=[],
    )


def _build_dnfr_prep_state(
    graph: MutableMapping[str, Any],
    previous: DnfrPrepState | None = None,
) -> DnfrPrepState:
    """Construct a :class:`DnfrPrepState` and mirror it on ``graph``."""

    cache_lock: threading.RLock
    vector_lock: threading.RLock
    if isinstance(previous, DnfrPrepState):
        cache_lock = previous.cache_lock
        vector_lock = previous.vector_lock
    else:
        cache_lock = threading.RLock()
        vector_lock = threading.RLock()
    state = DnfrPrepState(
        cache=_new_dnfr_cache(),
        cache_lock=cache_lock,
        vector_lock=vector_lock,
    )
    graph["_dnfr_prep_cache"] = state.cache
    return state


def _coerce_dnfr_state(
    graph: MutableMapping[str, Any],
    current: Any,
) -> DnfrPrepState:
    """Return ``current`` normalised into :class:`DnfrPrepState`."""

    if isinstance(current, DnfrPrepState):
        graph["_dnfr_prep_cache"] = current.cache
        return current
    try:
        from ..dynamics.dnfr import DnfrCache
    except Exception:  # pragma: no cover - defensive import
        DnfrCache = None  # type: ignore[assignment]
    if DnfrCache is not None and isinstance(current, DnfrCache):
        state = DnfrPrepState(
            cache=current,
            cache_lock=threading.RLock(),
            vector_lock=threading.RLock(),
        )
        graph["_dnfr_prep_cache"] = current
        return state
    return _build_dnfr_prep_state(graph)


def _graph_cache_manager(graph: MutableMapping[str, Any]) -> CacheManager:
    manager = graph.get(_GRAPH_CACHE_MANAGER_KEY)
    if not isinstance(manager, CacheManager):
        manager = CacheManager(default_capacity=128)
        graph[_GRAPH_CACHE_MANAGER_KEY] = manager
    config = graph.get(_GRAPH_CACHE_CONFIG_KEY)
    if isinstance(config, dict):
        manager.configure_from_mapping(config)
    def _dnfr_factory() -> DnfrPrepState:
        return _build_dnfr_prep_state(graph)

    def _dnfr_reset(current: Any) -> DnfrPrepState:
        if isinstance(current, DnfrPrepState):
            return _build_dnfr_prep_state(graph, current)
        return _build_dnfr_prep_state(graph)

    manager.register(
        DNFR_PREP_STATE_KEY,
        _dnfr_factory,
        reset=_dnfr_reset,
    )
    manager.update(
        DNFR_PREP_STATE_KEY,
        lambda current: _coerce_dnfr_state(graph, current),
    )
    return manager


def configure_graph_cache_limits(
    G: GraphLike | TNFRGraph | MutableMapping[str, Any],
    *,
    default_capacity: int | None | object = CacheManager._MISSING,
    overrides: Mapping[str, int | None] | None = None,
    replace_overrides: bool = False,
) -> CacheCapacityConfig:
    """Update cache capacity policy stored on ``G.graph``."""

    graph = get_graph(G)
    manager = _graph_cache_manager(graph)
    manager.configure(
        default_capacity=default_capacity,
        overrides=overrides,
        replace_overrides=replace_overrides,
    )
    snapshot = manager.export_config()
    graph[_GRAPH_CACHE_CONFIG_KEY] = {
        "default_capacity": snapshot.default_capacity,
        "overrides": dict(snapshot.overrides),
    }
    return snapshot


class EdgeCacheManager:
    """Coordinate cache storage and per-key locks for edge version caches."""

    _STATE_KEY = "_edge_version_state"

    def __init__(self, graph: MutableMapping[str, Any]) -> None:
        self.graph: MutableMapping[str, Any] = graph
        self._manager = _graph_cache_manager(graph)
        self._manager.register(
            self._STATE_KEY,
            self._default_state,
            reset=self._reset_state,
        )

    def record_hit(self) -> None:
        """Record a cache hit for telemetry."""

        self._manager.increment_hit(self._STATE_KEY)

    def record_miss(self, *, track_metrics: bool = True) -> None:
        """Record a cache miss for telemetry.

        When ``track_metrics`` is ``False`` the miss is acknowledged without
        mutating the aggregated metrics.
        """

        if track_metrics:
            self._manager.increment_miss(self._STATE_KEY)

    def record_eviction(self, *, track_metrics: bool = True) -> None:
        """Record cache eviction events for telemetry.

        When ``track_metrics`` is ``False`` the underlying metrics counter is
        left untouched while still signalling that an eviction occurred.
        """

        if track_metrics:
            self._manager.increment_eviction(self._STATE_KEY)

    def timer(self) -> TimingContext:
        """Return a timing context linked to this cache."""

        return self._manager.timer(self._STATE_KEY)

    def _default_state(self) -> EdgeCacheState:
        return self._build_state(None)

    def resolve_max_entries(self, max_entries: int | None | object) -> int | None:
        """Return effective capacity for the edge cache."""

        if max_entries is CacheManager._MISSING:
            return self._manager.get_capacity(self._STATE_KEY)
        return self._manager.get_capacity(
            self._STATE_KEY,
            requested=None if max_entries is None else int(max_entries),
            use_default=False,
        )

    def _build_state(self, max_entries: int | None) -> EdgeCacheState:
        locks: defaultdict[Hashable, threading.RLock] = defaultdict(threading.RLock)
        capacity = float("inf") if max_entries is None else int(max_entries)
        cache = InstrumentedLRUCache(
            capacity,
            manager=self._manager,
            metrics_key=self._STATE_KEY,
            locks=locks,
            count_overwrite_hit=False,
        )

        def _on_eviction(key: Hashable, _: Any) -> None:
            self.record_eviction(track_metrics=False)
            locks.pop(key, None)

        cache.set_eviction_callbacks(_on_eviction)
        return EdgeCacheState(cache=cache, locks=locks, max_entries=max_entries)

    def _ensure_state(
        self, state: EdgeCacheState | None, max_entries: int | None | object
    ) -> EdgeCacheState:
        target = self.resolve_max_entries(max_entries)
        if target is not None:
            target = int(target)
            if target < 0:
                raise ValueError("max_entries must be non-negative or None")
        if not isinstance(state, EdgeCacheState) or state.max_entries != target:
            return self._build_state(target)
        return state

    def _reset_state(self, state: EdgeCacheState | None) -> EdgeCacheState:
        if isinstance(state, EdgeCacheState):
            state.cache.clear()
            return state
        return self._build_state(None)

    def get_cache(
        self,
        max_entries: int | None | object,
        *,
        create: bool = True,
    ) -> tuple[
        MutableMapping[Hashable, Any] | None,
        dict[Hashable, threading.RLock]
        | defaultdict[Hashable, threading.RLock]
        | None,
    ]:
        """Return the cache and lock mapping for the manager's graph."""

        if not create:
            state = self._manager.peek(self._STATE_KEY)
            if isinstance(state, EdgeCacheState):
                return state.cache, state.locks
            return None, None

        state = self._manager.update(
            self._STATE_KEY,
            lambda current: self._ensure_state(current, max_entries),
        )
        return state.cache, state.locks

    def clear(self) -> None:
        """Reset cached data managed by this instance."""

        self._manager.clear(self._STATE_KEY)


def edge_version_cache(
    G: Any,
    key: Hashable,
    builder: Callable[[], T],
    *,
    max_entries: int | None | object = CacheManager._MISSING,
) -> T:
    """Return cached ``builder`` output tied to the edge version of ``G``."""

    graph = get_graph(G)
    manager = graph.get("_edge_cache_manager")  # type: ignore[assignment]
    if not isinstance(manager, EdgeCacheManager) or manager.graph is not graph:
        manager = EdgeCacheManager(graph)
        graph["_edge_cache_manager"] = manager

    resolved = manager.resolve_max_entries(max_entries)
    if resolved == 0:
        return builder()

    cache, locks = manager.get_cache(resolved)
    edge_version = get_graph_version(graph, "_edge_version")
    lock = locks[key]

    with lock:
        entry = cache.get(key)
        if entry is not None and entry[0] == edge_version:
            manager.record_hit()
            return entry[1]

    try:
        with manager.timer():
            value = builder()
    except (RuntimeError, ValueError) as exc:  # pragma: no cover - logging side effect
        logger.exception("edge_version_cache builder failed for %r: %s", key, exc)
        raise
    else:
        with lock:
            entry = cache.get(key)
            if entry is not None:
                cached_version, cached_value = entry
                manager.record_miss()
                if cached_version == edge_version:
                    manager.record_hit()
                    return cached_value
                manager.record_eviction()
            cache[key] = (edge_version, value)
            return value


def cached_nodes_and_A(
    G: nx.Graph,
    *,
    cache_size: int | None = 1,
    require_numpy: bool = False,
    prefer_sparse: bool = False,
    nodes: tuple[Any, ...] | None = None,
) -> tuple[tuple[Any, ...], Any]:
    """Return cached nodes tuple and adjacency matrix for ``G``.

    When ``prefer_sparse`` is true the adjacency matrix construction is skipped
    unless a caller later requests it explicitly.  This lets ΔNFR reuse the
    edge-index buffers stored on :class:`~tnfr.dynamics.dnfr.DnfrCache` without
    paying for ``nx.to_numpy_array`` on sparse graphs while keeping the
    canonical cache interface unchanged.
    """

    if nodes is None:
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
        if np is None or prefer_sparse:
            return nodes, None
        A = nx.to_numpy_array(G, nodelist=nodes, weight=None, dtype=float)
        return nodes, A

    nodes, A = edge_version_cache(G, key, builder, max_entries=cache_size)

    if require_numpy and A is None:
        raise RuntimeError("NumPy is required for adjacency caching")

    return nodes, A


def _reset_edge_caches(graph: Any, G: Any) -> None:
    """Clear caches affected by edge updates."""

    EdgeCacheManager(graph).clear()
    _graph_cache_manager(graph).clear(DNFR_PREP_STATE_KEY)
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
def edge_version_update(G: TNFRGraph) -> Iterator[None]:
    """Scope a batch of edge mutations."""

    increment_edge_version(G)
    try:
        yield
    finally:
        increment_edge_version(G)
