"""Helper utilities for caching graph-related data and computations."""

from __future__ import annotations

import hashlib
import threading
import warnings
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Hashable
from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
from types import MappingProxyType
from typing import Any, TypeVar

from cachetools import LRUCache
import networkx as nx  # type: ignore[import-untyped]

from ..graph_utils import mark_dnfr_prep_dirty
from ..import_utils import get_numpy
from ..json_utils import json_dumps
from ..logging_utils import get_logger

T = TypeVar("T")
U = TypeVar("U")

logger = get_logger(__name__)

# Key used to store the node set checksum in a graph's ``graph`` attribute.
NODE_SET_CHECKSUM_KEY = "_node_set_checksum_cache"

# Keys of cache entries dependent on the edge version.  Any change to the edge
# set requires these to be dropped to avoid stale data.
EDGE_VERSION_CACHE_KEYS = (
    "_cos_th",
    "_sin_th",
    "_thetas",
    "_trig_cache",
    "_trig_version",
)


__all__ = (
    "get_graph",
    "get_graph_mapping",
    "node_set_checksum",
    "_stable_json",
    "_node_repr_digest",
    "_node_repr",
    "_cache_node_list",
    "ensure_node_index_map",
    "ensure_node_offset_map",
    "edge_version_cache",
    "cached_nodes_and_A",
    "invalidate_edge_version_cache",
    "increment_edge_version",
    "edge_version_update",
)


def get_graph(obj: Any) -> Any:
    """Return ``obj.graph`` if available or ``obj`` otherwise."""
    return getattr(obj, "graph", obj)


def get_graph_mapping(
    G: Any, key: str, warn_msg: str
) -> Mapping[str, Any] | None:
    """Return an immutable view of ``G.graph[key]`` if it is a mapping.

    The mapping is wrapped in :class:`types.MappingProxyType` to prevent
    accidental modification. ``warn_msg`` is emitted via :func:`warnings.warn`
    when the stored value is not a mapping. ``None`` is returned when the key is
    absent or invalid.
    """
    data = G.graph.get(key)
    if data is None:
        return None
    if not isinstance(data, Mapping):
        warnings.warn(warn_msg, UserWarning, stacklevel=2)
        return None
    return MappingProxyType(data)


def _stable_json(obj: Any) -> str:
    """Return a JSON string with deterministic ordering."""
    return json_dumps(
        obj,
        sort_keys=True,
        ensure_ascii=False,
        to_bytes=False,
    )


@lru_cache(maxsize=1024)
def _node_repr_digest(obj: Any) -> tuple[str, bytes]:
    """Return cached stable representation and digest for ``obj``.

    This single helper centralises caching for node representations and their
    digests, ensuring both values stay in sync.
    """
    repr_ = _stable_json(obj)
    digest = hashlib.blake2b(repr_.encode("utf-8"), digest_size=16).digest()
    return repr_, digest


def _node_repr(n: Any) -> str:
    """Stable representation for node hashing and sorting."""
    return _node_repr_digest(n)[0]


def _hash_node(obj: Any) -> bytes:
    """Return a stable digest for ``obj`` used in node checksums."""
    return _node_repr_digest(obj)[1]


def _iter_node_digests(
    nodes: Iterable[Any], *, presorted: bool
) -> Iterable[bytes]:
    """Yield node digests in a deterministic order.

    When ``presorted`` is ``True`` the nodes are assumed to already be sorted
    in a stable manner and their digests are yielded directly. Otherwise,
    the tuple of representation and digest provided by
    :func:`_node_repr_digest` is used to avoid redundant computation.
    """
    if presorted:
        for node in nodes:
            yield _node_repr_digest(node)[1]
    else:
        pairs = [_node_repr_digest(n) for n in nodes]
        for _, digest in sorted(pairs, key=lambda x: x[0]):
            yield digest


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


def node_set_checksum(
    G: nx.Graph,
    nodes: Iterable[Any] | None = None,
    *,
    presorted: bool = False,
    store: bool = True,
) -> str:
    """Return a BLAKE2b checksum of ``G``'s node set.

    Nodes are serialised using :func:`_node_repr`. The helper
    :func:`_iter_node_digests` yields their digests in a deterministic order,
    handling the ``presorted`` and unsorted cases. When ``store`` is ``True``
    the final checksum is cached under ``NODE_SET_CHECKSUM_KEY`` to avoid
    recomputation for unchanged graphs.
    """
    graph = get_graph(G)
    node_iterable = G.nodes() if nodes is None else nodes

    hasher = hashlib.blake2b(digest_size=16)

    # Generate digests in stable order; `_iter_node_digests` sorts when needed
    # unless `presorted` indicates the nodes are already ordered.
    for digest in _iter_node_digests(node_iterable, presorted=presorted):
        hasher.update(digest)

    checksum = hasher.hexdigest()
    token = checksum[:16]
    if store:
        # Cache the result using a short token to detect unchanged node sets.
        cached = graph.get(NODE_SET_CHECKSUM_KEY)
        if cached and cached[0] == token:
            return cached[1]
        graph[NODE_SET_CHECKSUM_KEY] = (token, checksum)
    return checksum


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
    """Cache and return the tuple of nodes for ``G``.

    A :class:`NodeCache` instance is stored in ``G.graph`` under
    ``"_node_list_cache"``. The cache refreshes when the node set changes or when
    the optional ``"_node_list_dirty"`` flag is set to ``True``.
    """
    graph = get_graph(G)
    cache: NodeCache | None = graph.get("_node_list_cache")
    nodes = cache.nodes if cache else None
    sorted_nodes = cache.sorted_nodes if cache else None
    stored_len = graph.get("_node_list_len")
    current_n = G.number_of_nodes()
    dirty = bool(graph.pop("_node_list_dirty", False))

    # Determine if inexpensive checks already mark the cache as invalid
    invalid = nodes is None or stored_len != current_n or dirty
    new_checksum: str | None = None

    if not invalid and cache:
        # Only compute the checksum when the quick checks pass.
        # This avoids hashing the node set unless we suspect a change.
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


def _ensure_node_map(G, *, attrs: tuple[str, ...], sort: bool = False) -> dict[Any, int]:
    """Return cached node-to-index/offset mappings stored on ``NodeCache``.

    ``attrs`` selects the attributes on :class:`NodeCache` used to store the
    mapping(s). ``sort`` controls whether nodes are ordered by their string
    representation before assigning indices.
    """
    graph = G.graph
    _cache_node_list(G)
    cache: NodeCache = graph["_node_list_cache"]

    missing = [attr for attr in attrs if getattr(cache, attr) is None]
    if missing:
        if sort:
            nodes = cache.sorted_nodes
            if nodes is None:
                nodes = cache.sorted_nodes = tuple(sorted(cache.nodes, key=_node_repr))
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
        dict[Hashable, tuple[int, Any]]
        | LRUCache[Hashable, tuple[int, Any]]
        | None,
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
        dict[Hashable, tuple[int, Any]]
        | LRUCache[Hashable, tuple[int, Any]]
        | None,
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
        dict[Hashable, tuple[int, Any]]
        | LRUCache[Hashable, tuple[int, Any]]
        | None,
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
    """Return cached ``builder`` output tied to the edge version of ``G``.

    A per-key :class:`threading.RLock` is stored in a ``defaultdict`` to
    allow concurrent computations for different keys while still preventing
    duplicate work. The cache is checked while holding the key-specific lock;
    on a miss the lock is released for the potentially expensive ``builder``
    computation, then reacquired to verify and store the result.

    When ``max_entries`` is a positive integer-like value, only the most
    recent ``max_entries`` cache entries are kept (defaults to ``128``). If
    ``max_entries`` is ``None`` the cache may grow without bound. An explicit
    ``max_entries`` value of ``0`` disables caching entirely and ``builder``
    is executed on each invocation. ``max_entries`` is coerced to ``int`` on
    entry and validated to be non-negative. The ``builder`` runs outside the
    lock on a cache miss, so it **must** be pure and yield identical results
    across concurrent invocations.
    """
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
    # The lock is deliberately acquired twice. We first check for a cached
    # value under the lock, release it for the potentially expensive
    # ``builder`` computation and then re-acquire the lock to store the result.

    # First acquisition: inspect cache while holding the lock.
    with lock:
        entry = cache.get(key)
        if entry is not None and entry[0] == edge_version:
            return entry[1]

    # Execute builder without holding the lock to avoid blocking other threads.
    try:
        value = builder()
    except (
        RuntimeError,
        ValueError,
    ) as exc:  # pragma: no cover - logging side effect
        logger.exception(
            "edge_version_cache builder failed for %r: %s", key, exc
        )
        raise
    else:
        # Second acquisition: verify and store the result atomically after building.
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
    """Return list of nodes and adjacency matrix for ``G`` with caching.

    ``A`` is ``None`` when NumPy is unavailable. When ``require_numpy`` is
    ``True`` a :class:`RuntimeError` is raised if NumPy cannot be imported.
    """
    nodes_list = list(G.nodes())
    checksum = node_set_checksum(G, nodes_list, store=False)
    key = f"_dnfr_{len(nodes_list)}_{checksum}"
    G.graph["_dnfr_nodes_checksum"] = checksum
    np = get_numpy()
    if np is None:
        if require_numpy:
            raise RuntimeError("NumPy is required for adjacency caching")
        return edge_version_cache(
            G, key, lambda: (nodes_list, None), max_entries=cache_size
        )

    def builder() -> tuple[list[int], Any]:
        A = nx.to_numpy_array(G, nodelist=nodes_list, weight=None, dtype=float)
        return nodes_list, A

    return edge_version_cache(G, key, builder, max_entries=cache_size)


def _reset_edge_caches(graph: Any, G: Any) -> None:
    """Clear caches affected by edge updates."""
    invalidate_edge_version_cache(G)
    mark_dnfr_prep_dirty(G)
    _node_repr_digest.cache_clear()
    for key in EDGE_VERSION_CACHE_KEYS:
        graph.pop(key, None)


def increment_edge_version(G: Any) -> None:
    """Increment the edge version counter in ``G.graph``."""
    graph = get_graph(G)
    graph["_edge_version"] = int(graph.get("_edge_version", 0)) + 1
    _reset_edge_caches(graph, G)


@contextmanager
def edge_version_update(G: Any):
    """Scope a batch of edge mutations.

    Increments the edge version counter on entry **and** exit so caches and
    TNFR structural logs observe a single, coherent update cycle.  Use this to
    group related edge operations that should be treated as one semantic
    modification.
    """
    increment_edge_version(G)
    try:
        yield
    finally:
        increment_edge_version(G)
