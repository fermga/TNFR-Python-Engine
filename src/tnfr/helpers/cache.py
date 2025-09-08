from __future__ import annotations

import hashlib
import threading
import warnings
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping
from functools import lru_cache
from types import MappingProxyType
from typing import Any, TypeVar

from cachetools import LRUCache
import networkx as nx

from ..graph_utils import mark_dnfr_prep_dirty
from ..import_utils import get_numpy
from ..json_utils import json_dumps

T = TypeVar("T")

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
    return obj.graph if hasattr(obj, "graph") else obj


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
def _hash_node(obj: Any) -> bytes:
    """Return a stable digest for ``obj`` used in node checksums."""
    return hashlib.blake2b(
        _node_repr(obj).encode("utf-8"), digest_size=16
    ).digest()


def _update_node_cache(
    G: nx.Graph,
    nodes: Iterable[Any],
    key_prefix: str,
    value: Any | None = None,
    *,
    checksum: str | None = None,
    presorted: bool = False,
) -> str:
    """Store ``value`` and its node checksum in ``G.graph``.

    ``nodes`` is the iterable used to compute the checksum. When ``value`` is
    ``None`` it defaults to ``nodes``. The computed checksum is returned.
    Set ``presorted`` to ``True`` when ``nodes`` is already sorted in a stable
    manner to avoid redundant sorting.
    """
    if checksum is None:
        checksum = node_set_checksum(
            G, nodes, presorted=presorted, store=False
        )
    graph = get_graph(G)
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

    Nodes are serialised using :func:`_node_repr`. When ``presorted`` is
    ``False`` the resulting representations are sorted to make the checksum
    independent of node ordering. The digests of the ordered representations
    are fed into a new :class:`hashlib.blake2b` instance. When ``store`` is
    ``True`` the tuple of digests along with the final checksum is cached under
    ``"_node_set_checksum_cache"`` to avoid recalculating it for unchanged
    graphs.
    """
    graph = get_graph(G)
    node_iterable = G.nodes() if nodes is None else nodes
    if presorted:
        representations = map(_node_repr, node_iterable)
    else:
        representations = sorted(_node_repr(n) for n in node_iterable)
    hasher = hashlib.blake2b(digest_size=16)
    token_hasher = hashlib.blake2b(digest_size=8)
    for rep in representations:
        digest = hashlib.blake2b(rep.encode("utf-8"), digest_size=16).digest()
        hasher.update(digest)
        token_hasher.update(digest)
    checksum = hasher.hexdigest()
    token = token_hasher.hexdigest()
    if store:
        cached = graph.get("_node_set_checksum_cache")
        if cached and cached[0] == token:
            return cached[1]
        graph["_node_set_checksum_cache"] = (token, checksum)
    return checksum


def _cache_node_list(G: nx.Graph) -> tuple[Any, ...]:
    """Cache and return the tuple of nodes for ``G``.

    The cached values are stored in ``G.graph`` under ``"_node_list"``,
    ``"_node_list_len"`` and ``"_node_list_checksum"``.  The cache is
    refreshed when the number of nodes changes or when the optional
    ``"_node_list_dirty"`` flag is set to ``True``.
    """
    graph = get_graph(G)
    nodes = graph.get("_node_list")
    stored_len = graph.get("_node_list_len")
    current_n = G.number_of_nodes()
    dirty = bool(graph.pop("_node_list_dirty", False))
    if nodes is None or stored_len != current_n or dirty:
        nodes = tuple(G.nodes())
        _update_node_cache(G, nodes, "_node_list")
        graph["_node_list_len"] = current_n
    else:
        if "_node_list_checksum" not in graph:
            _update_node_cache(G, nodes, "_node_list")
    return nodes


def _ensure_node_map(G, *, key: str, sort: bool = False) -> dict[Any, int]:
    """Return cached node→index mapping for ``G`` under ``key``.

    ``sort`` controls whether nodes are ordered by their string representation
    before assigning indices.
    """
    nodes = list(G.nodes())
    if sort:
        nodes.sort(key=_node_repr)
    checksum = node_set_checksum(G, nodes, presorted=sort, store=False)
    mapping = G.graph.get(key)
    checksum_key = f"{key}_checksum"
    if mapping is None or G.graph.get(checksum_key) != checksum:
        mapping = {node: idx for idx, node in enumerate(nodes)}
        _update_node_cache(G, nodes, key, mapping, checksum=checksum)
    return mapping


def ensure_node_index_map(G) -> dict[Any, int]:
    """Return cached node→index mapping for ``G``."""
    return _ensure_node_map(G, key="_node_index_map", sort=False)


def ensure_node_offset_map(G) -> dict[Any, int]:
    """Return cached node→offset mapping for ``G``."""
    sort = bool(G.graph.get("SORT_NODES", False))
    return _ensure_node_map(G, key="_node_offset_map", sort=sort)


def _make_edge_cache(max_entries: int, locks: dict) -> Any:
    """Create an ``LRUCache`` for edge data with lock cleanup support.

    The cache removes any per-key locks when entries are evicted.  For older
    versions of :mod:`cachetools` lacking a ``callback`` parameter, a small
    subclass provides equivalent behaviour.
    """
    try:
        return LRUCache(max_entries, callback=lambda k, _: locks.pop(k, None))
    except TypeError:  # pragma: no cover - legacy cachetools

        class _LRUCache(LRUCache):
            def __init__(self, maxsize, *, callback=None):
                super().__init__(maxsize)
                self._callback = callback

            def popitem(self):  # type: ignore[override]
                key, value = super().popitem()
                cb = getattr(self, "_callback", None)
                if cb is not None:
                    cb(key, value)
                return key, value

        return _LRUCache(max_entries, callback=lambda k, v: locks.pop(k, None))


def _get_edge_cache(
    graph: Any, max_entries: int | None, *, create: bool = True
):
    """Return edge cache and lock mapping for ``graph``.

    When ``create`` is ``True`` missing structures are initialized according
    to ``max_entries``. Returns a tuple ``(cache, locks)``. Actual cache
    construction is handled by :func:`_make_edge_cache`.
    """
    use_lru = bool(max_entries)
    with _EDGE_CACHE_LOCK:
        locks = graph.get("_edge_version_cache_locks")
        if create:
            if (
                not isinstance(locks, defaultdict)
                or locks.default_factory is not threading.RLock
            ):
                locks = defaultdict(threading.RLock)
                graph["_edge_version_cache_locks"] = locks
        cache = graph.get("_edge_version_cache")
        if create:
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
                cache = _make_edge_cache(max_entries, locks) if use_lru else {}
                graph["_edge_version_cache"] = cache
        if isinstance(cache, dict) and isinstance(locks, dict):
            for key in list(locks.keys()):
                if key not in cache:
                    locks.pop(key, None)
    return cache, locks


def edge_version_cache(
    G: Any,
    key: str,
    builder: Callable[[], T],
    *,
    max_entries: int | None = 128,
) -> T:
    """Return cached ``builder`` output tied to the edge version of ``G``.

    A per-key :class:`threading.RLock` is stored in a ``defaultdict`` to
    allow concurrent computations for different keys while still preventing
    duplicate work. The function employs a simplified double-checked locking
    pattern: the cache is consulted before acquiring the key-specific lock,
    then checked again once the lock is held before computing and storing the
    value.

    When ``max_entries`` is a positive integer-like value, only the most
    recent ``max_entries`` cache entries are kept (defaults to ``128``). If
    ``max_entries`` is ``None`` the cache may grow without bound. An explicit
    ``max_entries`` value of ``0`` disables caching entirely and ``builder``
    is executed on each invocation. ``max_entries`` is coerced to ``int`` on
    entry and validated to be non-negative. The ``builder`` is executed outside
    any locks on a cache miss, so it **must** be pure and yield identical
    results across concurrent invocations.
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
    entry = cache.get(key)
    if entry is not None and entry[0] == edge_version:
        return entry[1]

    lock = locks[key]
    with lock:
        entry = cache.get(key)
        if entry is not None and entry[0] == edge_version:
            return entry[1]

        value = builder()
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
    G: nx.Graph, *, cache_size: int | None = 1
) -> tuple[list[int], Any]:
    """Return list of nodes and adjacency matrix for ``G`` with caching."""
    nodes_list = list(G.nodes())
    checksum = node_set_checksum(G, nodes_list, store=False)
    key = f"_dnfr_{len(nodes_list)}_{checksum}"
    G.graph["_dnfr_nodes_checksum"] = checksum
    np = get_numpy()

    def builder() -> tuple[list[int], Any]:
        if np is not None:
            A = nx.to_numpy_array(
                G, nodelist=nodes_list, weight=None, dtype=float
            )
        else:  # pragma: no cover - dependiente de numpy
            A = None
        return nodes_list, A

    return edge_version_cache(G, key, builder, max_entries=cache_size)


def increment_edge_version(G: Any) -> None:
    """Increment the edge version counter in ``G.graph``."""
    graph = get_graph(G)
    graph["_edge_version"] = int(graph.get("_edge_version", 0)) + 1
    invalidate_edge_version_cache(G)
    mark_dnfr_prep_dirty(G)
    _node_repr.cache_clear()
    _hash_node.cache_clear()
    for key in EDGE_VERSION_CACHE_KEYS:
        graph.pop(key, None)
