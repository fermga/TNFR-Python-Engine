"""Helper utilities for caching graph-related data and computations."""

from __future__ import annotations

import hashlib
import threading
import warnings
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Hashable
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

logger = get_logger(__name__)

_EDGE_CACHE_LOCK = threading.RLock()

# Keys of cache entries dependent on the edge version.  Any change to the edge
# set requires these to be dropped to avoid stale data.
EDGE_VERSION_CACHE_KEYS = (
    "_cos_th",
    "_sin_th",
    "_thetas",
    "_trig_cache",
    "_trig_version",
)


__all__ = [
    "get_graph",
    "get_graph_mapping",
    "node_set_checksum",
    "_stable_json",
    "_node_repr",
    "_cache_node_list",
    "ensure_node_index_map",
    "ensure_node_offset_map",
    "edge_version_cache",
    "cached_nodes_and_A",
    "invalidate_edge_version_cache",
    "increment_edge_version",
]


def get_graph(obj: Any) -> Any:
    """Return ``obj.graph`` if available or ``obj`` otherwise."""
    return getattr(obj, "graph", obj)


def get_graph_mapping(
    G: Any, key: str, warn_msg: str
) -> dict[str, Any] | None:
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
def _node_repr(n: Any) -> str:
    """Stable representation for node hashing and sorting."""
    return _stable_json(n)


@lru_cache(maxsize=1024)
def _hash_node(obj: Any, repr_: str | None = None) -> bytes:
    """Return a stable digest for ``obj`` used in node checksums.

    ``repr_`` provides an optional precomputed representation of ``obj`` and
    avoids an additional call to :func:`_node_repr` when supplied.
    """
    if repr_ is None:
        repr_ = _node_repr(obj)
    return hashlib.blake2b(repr_.encode("utf-8"), digest_size=16).digest()


def _iter_node_digests(
    nodes: Iterable[Any], *, presorted: bool
) -> Iterable[bytes]:
    """Yield node digests in a deterministic order.

    When ``presorted`` is ``True`` the nodes are assumed to already be sorted
    in a stable manner and their digests are yielded directly. Otherwise,
    tuples of ``(repr, node)`` are generated and sorted by ``repr`` before
    passing both values to :func:`_hash_node`.
    """
    if presorted:
        yield from (_hash_node(n) for n in nodes)
    else:
        # Precompute representations to avoid duplicate work during hashing.
        for repr_, node in sorted((_node_repr(n), n) for n in nodes):
            yield _hash_node(node, repr_)


def _update_node_cache(
    graph: Any,
    nodes: Iterable[Any],
    key_prefix: str,
    value: Any | None = None,
    *,
    checksum: str | None = None,
    presorted: bool = False,
) -> str:
    """Store ``value`` and its node checksum in ``graph``.",

    ``nodes`` is the iterable used to compute the checksum. When ``value`` is
    ``None`` it defaults to ``nodes``. The computed checksum is returned.
    Set ``presorted`` to ``True`` when ``nodes`` is already sorted in a stable
    manner to avoid redundant sorting.
    """
    if checksum is None:
        checksum = node_set_checksum(
            graph, nodes, presorted=presorted, store=False
        )
    graph[key_prefix] = nodes if value is None else value
    graph[f"{key_prefix}_checksum"] = checksum
    return checksum


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
    the final checksum is cached under ``"_node_set_checksum_cache"`` to avoid
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
        cached = graph.get("_node_set_checksum_cache")
        if cached and cached[0] == token:
            return cached[1]
        graph["_node_set_checksum_cache"] = (token, checksum)
    return checksum


def _cache_node_list(G: nx.Graph) -> tuple[Any, ...]:
    """Cache and return the tuple of nodes for ``G``.

    The cached values are stored in ``G.graph`` under ``"_node_list"``,
    ``"_node_list_len"`` and ``"_node_list_checksum"``.  The cache is
    refreshed when the node set checksum or number of nodes changes, or when
    the optional ``"_node_list_dirty"`` flag is set to ``True``.
    """
    graph = get_graph(G)
    nodes = graph.get("_node_list")
    stored_len = graph.get("_node_list_len")
    current_n = G.number_of_nodes()
    dirty = bool(graph.pop("_node_list_dirty", False))
    if nodes is None or stored_len != current_n or dirty:
        nodes = tuple(G.nodes())
        checksum = node_set_checksum(G, nodes, store=True)
        _update_node_cache(graph, nodes, "_node_list", checksum=checksum)
        graph["_node_list_len"] = current_n
    else:
        new_checksum = node_set_checksum(G)
        if graph.get("_node_list_checksum") != new_checksum:
            nodes = tuple(G.nodes())
            _update_node_cache(graph, nodes, "_node_list", checksum=new_checksum)
            graph["_node_list_len"] = current_n
        elif "_node_list_checksum" not in graph:
            _update_node_cache(graph, nodes, "_node_list", checksum=new_checksum)
    return nodes


def _ensure_node_map(G, *, key: str, sort: bool = False) -> dict[Any, int]:
    """Return cached node→index mapping for ``G`` under ``key``.

    ``sort`` controls whether nodes are ordered by their string representation
    before assigning indices.
    """
    graph = G.graph
    mapping = graph.get(key)
    checksum_key = f"{key}_checksum"
    stored_checksum = graph.get(checksum_key)
    checksum = node_set_checksum(G, store=False)

    if mapping is None or stored_checksum != checksum:
        nodes = list(G.nodes())
        if sort:
            nodes.sort(key=_node_repr)
        mapping = {node: idx for idx, node in enumerate(nodes)}
        _update_node_cache(graph, nodes, key, mapping, checksum=checksum)
    return mapping


def ensure_node_index_map(G) -> dict[Any, int]:
    """Return cached node→index mapping for ``G``."""
    return _ensure_node_map(G, key="_node_index_map", sort=False)


def ensure_node_offset_map(G) -> dict[Any, int]:
    """Return cached node→offset mapping for ``G``."""
    sort = bool(G.graph.get("SORT_NODES", False))
    return _ensure_node_map(G, key="_node_offset_map", sort=sort)


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


def _ensure_edge_cache_locks(graph: Any) -> defaultdict[Hashable, threading.RLock]:
    """Ensure per-key lock mapping for edge cache."""
    locks = graph.get("_edge_version_cache_locks")
    if (
        not isinstance(locks, defaultdict)
        or locks.default_factory is not threading.RLock
    ):
        locks = defaultdict(threading.RLock)
        graph["_edge_version_cache_locks"] = locks
    return locks


def _init_edge_cache(
    graph: Any,
    locks: dict[Hashable, threading.RLock],
    max_entries: int | None,
) -> dict[Hashable, tuple[int, Any]] | LRUCache[Hashable, tuple[int, Any]]:
    """Initialize and store edge cache in ``graph``."""
    use_lru = bool(max_entries)
    cache: dict[Hashable, tuple[int, Any]] | LRUCache[Hashable, tuple[int, Any]]
    cache = _make_edge_cache(max_entries, locks) if use_lru else {}
    graph["_edge_version_cache"] = cache
    return cache


def _maybe_init_edge_cache(
    graph: Any,
    locks: dict[Hashable, threading.RLock],
    max_entries: int | None,
) -> dict[Hashable, tuple[int, Any]] | LRUCache[Hashable, tuple[int, Any]]:
    """Return existing cache or initialise a new one if needed."""
    use_lru = bool(max_entries)
    cache = graph.get("_edge_version_cache")
    if (
        cache is None
        or (
            use_lru
            and (
                not isinstance(cache, LRUCache)
                or cache.maxsize != max_entries
            )
        )
        or (not use_lru and isinstance(cache, LRUCache))
    ):
        cache = _init_edge_cache(graph, locks, max_entries)
    return cache


def _get_edge_cache(
    graph: Any, max_entries: int | None, *, create: bool = True
) -> tuple[
    dict[Hashable, tuple[int, Any]] | LRUCache[Hashable, tuple[int, Any]] | None,
    dict[Hashable, threading.RLock] | defaultdict[Hashable, threading.RLock] | None,
]:
    """Return edge cache and lock mapping for ``graph``.

    When ``create`` is ``True`` missing structures are initialized according
    to ``max_entries``. Returns a tuple ``(cache, locks)``. Actual cache
    construction is handled by :func:`_make_edge_cache`.
    """
    with _EDGE_CACHE_LOCK:
        if create:
            locks = _ensure_edge_cache_locks(graph)
            cache = _maybe_init_edge_cache(graph, locks, max_entries)
        else:
            locks = graph.get("_edge_version_cache_locks")
            cache = graph.get("_edge_version_cache")
        if isinstance(cache, dict) and isinstance(locks, dict):
            for key in list(locks.keys()):
                if key not in cache:
                    locks.pop(key, None)
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
    cache, locks = _get_edge_cache(graph, max_entries)
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
    except (RuntimeError, ValueError) as exc:  # pragma: no cover - logging side effect
        logger.exception("edge_version_cache builder failed for %r: %s", key, exc)
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
    cache, locks = _get_edge_cache(graph, None, create=False)
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
    _node_repr.cache_clear()
    _hash_node.cache_clear()
    for key in EDGE_VERSION_CACHE_KEYS:
        graph.pop(key, None)


def increment_edge_version(G: Any) -> None:
    """Increment the edge version counter in ``G.graph``."""
    graph = get_graph(G)
    graph["_edge_version"] = int(graph.get("_edge_version", 0)) + 1
    _reset_edge_caches(graph, G)
