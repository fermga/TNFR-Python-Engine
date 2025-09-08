"""Helper functions."""

from __future__ import annotations
from typing import (
    Iterable,
    Sequence,
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
import warnings
from collections.abc import Mapping
from types import MappingProxyType
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
    "neighbor_phase_mean_list",
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
    "get_graph_mapping",
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
    try:
        return float(fmean(xs))
    except (StatisticsError, ValueError, TypeError):
        return float(default)


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
    """Internal helper delegating to :func:`neighbor_phase_mean_list`."""
    fallback = trig.theta.get(node.n, 0.0)
    neigh = node.G[node.n]
    np = get_numpy()
    return neighbor_phase_mean_list(
        neigh, trig.cos, trig.sin, np=np, fallback=fallback
    )


def _phase_mean_from_iter(
    it: Iterable[tuple[float, float] | None], fallback: float
) -> float:
    x = y = 0.0
    cx = cy = 0.0  # Kahan compensation terms
    count = 0
    for cs in it:
        if cs is None:
            continue
        cos_val, sin_val = cs
        # compensated summation for cosine
        tx = cos_val - cx
        vx = x + tx
        cx = (vx - x) - tx
        x = vx
        # compensated summation for sine
        ty = sin_val - cy
        vy = y + ty
        cy = (vy - y) - ty
        y = vy
        count += 1
    if count == 0:
        return fallback
    return math.atan2(y, x)


def neighbor_phase_mean_list(
    neigh: Sequence[Any],
    cos_th: Dict[Any, float],
    sin_th: Dict[Any, float],
    np=None,
    fallback: float = 0.0,
) -> float:
    """Return circular mean of neighbour phases from cosine/sine mappings.

    When ``np`` (NumPy) is provided, ``np.fromiter`` is used to compute the
    averages. Otherwise, the mean is computed using the pure-Python
    :func:`_phase_mean_from_iter` helper.
    """
    deg = len(neigh)
    if deg == 0:
        return fallback
    if np is not None:
        cos_vals = np.fromiter((cos_th[v] for v in neigh), dtype=float, count=deg)
        sin_vals = np.fromiter((sin_th[v] for v in neigh), dtype=float, count=deg)
        mean_cos = float(cos_vals.mean())
        mean_sin = float(sin_vals.mean())
        return float(np.arctan2(mean_sin, mean_cos))
    return _phase_mean_from_iter(((cos_th[v], sin_th[v]) for v in neigh), fallback)


def neighbor_phase_mean(obj, n=None) -> float:
    """Circular mean of neighbour phases.

    The :class:`NodoNX` import is cached after the first call.
    """
    NodoNX = import_nodonx()
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


def get_graph_mapping(G: Any, key: str, warn_msg: str) -> Dict[str, Any] | None:
    """Return a shallow copy of ``G.graph[key]`` if it is a mapping.

    ``warn_msg`` is emitted via :func:`warnings.warn` when the stored value is
    not a mapping. ``None`` is returned when the key is absent or invalid.
    """

    data = G.graph.get(key)
    if data is None:
        return None
    if not isinstance(data, Mapping):
        warnings.warn(warn_msg, UserWarning, stacklevel=2)
        return None
    return MappingProxyType(data)


class _Encoder(json.JSONEncoder):
    def __init__(self, *args, max_depth: int = 10, **kwargs):
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


def _stable_json(obj: Any, max_depth: int = 10) -> str:
    """Return a JSON string with deterministic ordering."""

    return json.dumps(
        obj, sort_keys=True, ensure_ascii=False, cls=_Encoder, max_depth=max_depth
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
) -> str:
    """Store ``value`` and its node checksum in ``G.graph``.

    ``nodes`` is the iterable used to compute the checksum. When ``value`` is
    ``None`` it defaults to ``nodes``. The computed checksum is returned.
    """

    if checksum is None:
        checksum = node_set_checksum(G, nodes, store=False)
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
    token_hasher = hashlib.blake2b(digest_size=8)
    for n in node_iterable:
        digest = _hash_node(n)
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
        _update_node_cache(
            G, nodes, key, mapping, checksum=checksum
        )
    return mapping


def ensure_node_index_map(G) -> Dict[Any, int]:
    """Return cached node→index mapping for ``G``."""

    return _ensure_node_map(G, key="_node_index_map", sort=False)


def ensure_node_offset_map(G) -> Dict[Any, int]:
    """Return cached node→offset mapping for ``G``."""

    sort = bool(G.graph.get("SORT_NODES", False))
    return _ensure_node_map(G, key="_node_offset_map", sort=sort)


def _get_edge_cache(graph: Any, max_entries: int | None, *, create: bool = True):
    """Return edge cache and lock mapping for ``graph``.

    When ``create`` is ``True`` missing structures are initialized according
    to ``max_entries``. Returns a tuple ``(cache, locks, use_lru)`` where
    ``use_lru`` indicates whether an ``LRUCache`` is employed.
    """

    use_lru = bool(max_entries)
    locks = graph.get("_edge_version_cache_locks")
    if create:
        if locks is None or not isinstance(locks, dict):
            locks = {}
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
            if use_lru:
                try:
                    cache = LRUCache(
                        max_entries,
                        callback=lambda k, _: locks.pop(k, None),
                    )
                except TypeError:  # pragma: no cover - legacy cachetools
                    class _LRUCache(LRUCache):
                        def __init__(self, maxsize, *, callback=None):
                            super().__init__(maxsize)
                            self._callback = callback

                        def popitem(self):
                            key, value = super().popitem()
                            cb = getattr(self, "_callback", None)
                            if cb is not None:
                                cb(key, value)
                            return key, value

                    cache = _LRUCache(
                        max_entries, callback=lambda k, v: locks.pop(k, None)
                    )
            else:
                cache = {}
            graph["_edge_version_cache"] = cache
    return cache, locks, use_lru


def edge_version_cache(
    G: Any, key: str, builder: Callable[[], T], *, max_entries: int | None = 128
) -> T:
    """Return cached ``builder`` output tied to the edge version of ``G``.

    Cache lookups and updates are serialized via ``_EDGE_CACHE_LOCK``.  A
    dedicated ``threading.Lock`` is maintained for each cache key so that
    different keys can be computed concurrently while still preventing
    duplicate work for the same key. Locks are stored in a simple ``dict``
    and cleared when the cache is invalidated.

    When ``max_entries`` is a positive integer-like value, only the most recent
    ``max_entries`` cache entries are kept (defaults to ``128``).  If
    ``max_entries`` is ``None`` the cache may grow without bound.  A
    ``max_entries`` value of ``0`` disables caching entirely and ``builder``
    is executed on each invocation.  ``max_entries`` is coerced to ``int`` on
    entry and validated to be non-negative. The ``builder`` is executed
    outside the global lock when a cache miss occurs, so it **must** be pure
    and yield identical results across concurrent invocations.
    """
    if max_entries is not None:
        max_entries = int(max_entries)
        if max_entries < 0:
            raise ValueError("max_entries must be non-negative or None")
    if max_entries == 0:
        return builder()

    graph = get_graph(G)
    with _EDGE_CACHE_LOCK:
        cache, locks, use_lru = _get_edge_cache(graph, max_entries)
        edge_version = int(graph.get("_edge_version", 0))
        entry = cache.get(key)
        if entry is not None and entry[0] == edge_version:
            return entry[1]

        lock = locks.setdefault(key, threading.RLock())

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
    cache, locks, _ = _get_edge_cache(graph, None, create=False)
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
    _hash_node.cache_clear()
    for key in EDGE_VERSION_CACHE_KEYS:
        graph.pop(key, None)
