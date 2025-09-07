"""Helper functions."""

from __future__ import annotations
from typing import (
    Iterable,
    Dict,
    Any,
    Callable,
    TypeVar,
)
import math
import hashlib
from statistics import fmean, StatisticsError
import threading
import json
from functools import lru_cache
from cachetools import LRUCache
import networkx as nx

from .import_utils import get_numpy, import_nodonx
from .collections_utils import (
    MAX_MATERIALIZE_DEFAULT,
    ensure_collection,
    normalize_weights,
    normalize_counter,
    mix_groups,
)
from .alias import get_attr
from .graph_utils import mark_dnfr_prep_dirty

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
    "MAX_MATERIALIZE_DEFAULT",
    "ensure_collection",
    "clamp",
    "clamp01",
    "list_mean",
    "angle_diff",
    "normalize_weights",
    "neighbor_mean",
    "neighbor_phase_mean",
    "push_glyph",
    "recent_glyph",
    "ensure_history",
    "last_glyph",
    "count_glyphs",
    "normalize_counter",
    "mix_groups",
    "ensure_node_index_map",
    "ensure_node_offset_map",
    "edge_version_cache",
    "cached_nodes_and_A",
    "invalidate_edge_version_cache",
    "increment_edge_version",
    "node_set_checksum",
    "get_graph",
    "mark_dnfr_prep_dirty",
]


T = TypeVar("T")

# -------------------------
# Utilidades numéricas básicas
# -------------------------


def clamp(x: float, a: float, b: float) -> float:
    """Return ``x`` clamped to the ``[a, b]`` interval."""
    return max(a, min(b, x))


def clamp01(x: float) -> float:
    """Clamp ``x`` to the ``[0,1]`` interval."""
    return clamp(float(x), 0.0, 1.0)


def list_mean(xs: Iterable[float], default: float = 0.0) -> float:
    """Return the arithmetic mean of ``xs`` or ``default`` if empty."""
    # Pre-cast ``default`` to anticipate errors and avoid multiple casting
    result = float(default)
    try:
        result = float(fmean(xs))
    except (StatisticsError, ValueError, TypeError):
        result = result  # default already converted
    return result


def angle_diff(a: float, b: float) -> float:
    """Return the minimal difference between two angles in radians."""
    return (float(a) - float(b) + math.pi) % math.tau - math.pi


# -------------------------
# Estadísticos vecinales
# -------------------------


def neighbor_mean(G, n, aliases: tuple[str, ...], default: float = 0.0) -> float:
    """Mean of ``aliases`` attribute among neighbours of ``n``."""
    vals = (get_attr(G.nodes[v], aliases, default) for v in G.neighbors(n))
    return list_mean(vals, default)


def _neighbor_phase_mean(node, trig) -> float:
    cos_th, sin_th = trig.cos, trig.sin
    it = ((cos_th.get(v), sin_th.get(v)) for v in node.neighbors())
    fallback = trig.theta.get(node.n, 0.0)
    return _phase_mean_from_iter(it, fallback)


def _phase_mean_from_iter(
    it: Iterable[tuple[float, float] | None], fallback: float
) -> float:
    x = y = 0.0
    count = 0
    for cs in it:
        if cs is None:
            continue
        cos_val, sin_val = cs
        x += cos_val
        y += sin_val
        count += 1
    if count == 0:
        return fallback
    return math.atan2(y, x)


@lru_cache(maxsize=1)
def _get_nodonx():
    """Return :class:`NodoNX` caching the deferred import."""
    return import_nodonx()


def neighbor_phase_mean(obj, n=None) -> float:
    """Circular mean of neighbour phases.

    The :class:`NodoNX` import is cached after the first call.
    """
    NodoNX = _get_nodonx()

    node = NodoNX(obj, n) if n is not None else obj
    if getattr(node, "G", None) is None:
        raise TypeError("neighbor_phase_mean requires nodes bound to a graph")
    from .metrics_utils import get_trig_cache

    trig = get_trig_cache(node.G)
    return _neighbor_phase_mean(node, trig)


# -------------------------
# Historial de glyphs por nodo
# -------------------------

# Importaciones diferidas para evitar ciclos con ``alias``
from .glyph_history import (  # noqa: E402
    push_glyph,
    recent_glyph,
    ensure_history,
    last_glyph,
    count_glyphs,
)


# -------------------------
# Grafos
# -------------------------


def get_graph(obj: Any) -> Any:
    """Return ``obj.graph`` if available or ``obj`` otherwise."""

    return obj.graph if hasattr(obj, "graph") else obj


def _stable_json(obj: Any, max_depth: int = 10) -> str:
    """Return a JSON string with deterministic ordering."""

    class _Encoder(json.JSONEncoder):
        def __init__(self, *args, **kwargs):
            self._max_depth = max_depth
            super().__init__(*args, **kwargs)

        def default(self, o):
            if isinstance(o, set):
                return sorted(o, key=lambda x: repr(x))
            if hasattr(o, "__dict__"):
                return o.__dict__
            return repr(o)

        def encode(self, o):
            try:
                self._check_depth(o, 0)
            except RecursionError as exc:
                raise ValueError("circular reference detected") from exc
            return super().encode(o)

        def _check_depth(self, o, depth):
            if depth > self._max_depth:
                raise ValueError(f"max depth {self._max_depth} exceeded")
            if isinstance(o, dict):
                for v in o.values():
                    self._check_depth(v, depth + 1)
            elif isinstance(o, (list, tuple, set)):
                for item in o:
                    self._check_depth(item, depth + 1)
            elif hasattr(o, "__dict__"):
                self._check_depth(o.__dict__, depth + 1)

    return json.dumps(obj, sort_keys=True, ensure_ascii=False, cls=_Encoder)


@lru_cache(maxsize=1024)
def _node_repr(n: Any) -> str:
    """Stable representation for node hashing and sorting."""
    return _stable_json(n)


def node_set_checksum(
    G: nx.Graph,
    nodes: Iterable[Any] | None = None,
    *,
    presorted: bool = False,
    store: bool = True,
) -> str:
    """Return a BLAKE2b checksum of ``G``'s node set.

    Nodes are serialised using :func:`_node_repr`. Each node's digest is
    computed individually and the collection of digests is sorted to make the
    resulting checksum independent of node ordering. The sorted digests are then
    fed into a new :class:`hashlib.blake2b` instance. When ``store`` is ``True``
    the tuple of digests along with the final checksum is cached under
    ``"_node_set_checksum_cache"`` to avoid recalculating it for unchanged
    graphs.
    """

    graph = get_graph(G)
    node_iterable = G.nodes() if nodes is None else nodes

    if not presorted:
        node_iterable = sorted(node_iterable, key=_node_repr)

    hasher = hashlib.blake2b(digest_size=16)

    if store:
        digest_tuple = tuple(
            hashlib.blake2b(
                _node_repr(n).encode("utf-8"), digest_size=16
            ).digest()
            for n in node_iterable
        )

        cached = graph.get("_node_set_checksum_cache")
        if cached and cached[0] == digest_tuple:
            return cached[1]

        for d in digest_tuple:
            hasher.update(d)

        checksum = hasher.hexdigest()
        graph["_node_set_checksum_cache"] = (digest_tuple, checksum)
        return checksum

    for n in node_iterable:
        d = hashlib.blake2b(_node_repr(n).encode("utf-8"), digest_size=16).digest()
        hasher.update(d)

    return hasher.hexdigest()


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
        checksum = node_set_checksum(G, nodes, store=False)
        graph["_node_list"] = nodes
        graph["_node_list_len"] = current_n
        graph["_node_list_checksum"] = checksum
    else:
        if "_node_list_checksum" not in graph:
            graph["_node_list_checksum"] = node_set_checksum(
                G, nodes, store=False
            )
    return nodes


def _ensure_node_map(G, *, key: str, sort: bool = False) -> Dict[Any, int]:
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
        G.graph[key] = mapping
        G.graph[checksum_key] = checksum
    return mapping


def ensure_node_index_map(G) -> Dict[Any, int]:
    """Return cached node→index mapping for ``G``."""

    return _ensure_node_map(G, key="_node_index_map", sort=False)


def ensure_node_offset_map(G) -> Dict[Any, int]:
    """Return cached node→offset mapping for ``G``."""

    sort = bool(G.graph.get("SORT_NODES", False))
    return _ensure_node_map(G, key="_node_offset_map", sort=sort)


def edge_version_cache(
    G: Any, key: str, builder: Callable[[], T], *, max_entries: int | None = 128
) -> T:
    """Return cached ``builder`` output tied to the edge version of ``G``.

    Cache lookups and updates are serialized via ``_EDGE_CACHE_LOCK``.  A
    dedicated ``threading.Lock`` is maintained for each cache key so that
    different keys can be computed concurrently while still preventing
    duplicate work for the same key. Locks are stored in a simple ``dict``
    and cleared when the cache is invalidated.

    When ``max_entries`` is a positive integer, only the most recent
    ``max_entries`` cache entries are kept (defaults to ``128``).  The
    ``builder`` is executed outside the global lock when a cache miss occurs,
    so it **must** be pure and yield identical results across concurrent
    invocations.
    """
    if max_entries is not None and max_entries < 0:
        raise ValueError("max_entries must be non-negative or None")
    graph = get_graph(G)
    use_lru = bool(max_entries)

    with _EDGE_CACHE_LOCK:
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
            cache = LRUCache(max_entries) if use_lru else {}
            graph["_edge_version_cache"] = cache

        locks = graph.get("_edge_version_cache_locks")
        if locks is None or not isinstance(locks, dict):
            locks = {}
            graph["_edge_version_cache_locks"] = locks

        edge_version = int(graph.get("_edge_version", 0))
        entry = cache.get(key)
        if entry is not None and entry[0] == edge_version:
            return entry[1]

        lock = locks.setdefault(key, threading.Lock())
        # Double locking: _EDGE_CACHE_LOCK guards cache structures while each
        # per-key lock prevents duplicate work for a given key.

    with lock:
        with _EDGE_CACHE_LOCK:
            entry = cache.get(key)
            if entry is not None and entry[0] == edge_version:
                return entry[1]

        value = builder()

        with _EDGE_CACHE_LOCK:
            cache[key] = (edge_version, value)
            return value


def invalidate_edge_version_cache(G: Any) -> None:
    """Clear cached entries associated with ``G``."""
    graph = get_graph(G)
    cache = graph.get("_edge_version_cache")
    locks = graph.get("_edge_version_cache_locks")
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
            A = nx.to_numpy_array(G, nodelist=nodes_list, weight=None, dtype=float)
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
    for key in EDGE_VERSION_CACHE_KEYS:
        graph.pop(key, None)
