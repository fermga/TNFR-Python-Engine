"""Helper functions."""

from __future__ import annotations
from typing import (
    Iterable,
    Dict,
    Any,
    TYPE_CHECKING,
    Callable,
    TypeVar,
)
import math
import hashlib
from statistics import fmean, StatisticsError
from collections import OrderedDict
import threading
import weakref

from .import_utils import get_numpy, import_nodonx
from .collections_utils import (
    MAX_MATERIALIZE_DEFAULT,
    ensure_collection,
    normalize_weights,
    normalize_counter,
    mix_groups,
)
from .alias import get_attr
from .constants import ALIAS_THETA

_EDGE_CACHE_LOCK = threading.Lock()

if TYPE_CHECKING:  # pragma: no cover - solo para type checkers
    import networkx as nx

PI = math.pi
TWO_PI = 2 * PI

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
    "get_cached_trig",
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
    "increment_edge_version",
    "node_set_checksum",
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
    except (StatisticsError, ValueError):
        return float(default)


def angle_diff(a: float, b: float) -> float:
    """Return the minimal difference between two angles in radians."""
    return (float(a) - float(b) + PI) % TWO_PI - PI


# -------------------------
# Estadísticos vecinales
# -------------------------


def neighbor_mean(G, n, aliases: tuple[str, ...], default: float = 0.0) -> float:
    """Mean of ``aliases`` attribute among neighbours of ``n``."""
    vals = (get_attr(G.nodes[v], aliases, default) for v in G.neighbors(n))
    return list_mean(vals, default)


def get_cached_trig(
    v: Any, cache: weakref.WeakKeyDictionary[Any, tuple[float, float] | None]
) -> tuple[float, float] | None:
    """Return ``(cos, sin)`` for ``v`` caching by weak reference."""
    val = cache.get(v)
    if val is not None:
        return val
    data = getattr(v, "__dict__", v if isinstance(v, dict) else {})
    th = get_attr(data, ALIAS_THETA, None)
    if th is None:
        cache[v] = None
        return None
    val = (math.cos(th), math.sin(th))
    cache[v] = val
    return val


def _neighbor_phase_mean(
    node,
    *,
    trig=None,
    cache: weakref.WeakKeyDictionary[Any, tuple[float, float] | None] | None = None,
) -> float:
    x = y = 0.0
    count = 0
    if trig is not None:
        cos_th, sin_th = trig.cos, trig.sin
        for v in node.neighbors():
            x += cos_th[v]
            y += sin_th[v]
            count += 1
        if count == 0:
            return get_attr(node.G.nodes[node.n], ALIAS_THETA, 0.0)
    else:
        if cache is None:
            cache = weakref.WeakKeyDictionary()
        for v in node.neighbors():
            cs = get_cached_trig(v, cache)
            if cs is None:
                continue
            cx, sy = cs
            x += cx
            y += sy
            count += 1
        if count == 0:
            return get_attr(getattr(node, "__dict__", {}), ALIAS_THETA, 0.0)
    return math.atan2(y, x)


def neighbor_phase_mean(obj, n=None) -> float:
    """Circular mean of neighbour phases."""
    NodoNX = import_nodonx()

    node = NodoNX(obj, n) if n is not None else obj

    trig = None
    if getattr(node, "G", None) is not None:
        from .metrics_utils import precompute_trigonometry

        trig = precompute_trigonometry(node.G)
        return _neighbor_phase_mean(node, trig=trig)

    cache = getattr(node, "_trig_cache", None)
    if cache is None:
        cache = weakref.WeakKeyDictionary()
        setattr(node, "_trig_cache", cache)
    return _neighbor_phase_mean(node, cache=cache)


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


def _stable_json(obj: Any, visited: set[int] | None = None) -> Any:
    """Return a structure with deterministic ordering suitable for ``repr``."""
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj

    if visited is None:
        visited = set()

    obj_id = id(obj)
    if obj_id in visited:
        return "<recursion>"
    visited.add(obj_id)

    if isinstance(obj, (list, tuple)):
        return [_stable_json(o, visited) for o in obj]
    if isinstance(obj, set):
        return [_stable_json(o, visited) for o in sorted(obj, key=lambda x: repr(x))]
    if isinstance(obj, dict):
        return {
            str(k): _stable_json(v, visited)
            for k, v in sorted(obj.items(), key=lambda kv: str(kv[0]))
        }
    if hasattr(obj, "__dict__"):
        return {
            k: _stable_json(v, visited)
            for k, v in sorted(vars(obj).items(), key=lambda kv: kv[0])
        }
    return f"{obj.__module__}.{obj.__class__.__qualname__}"


def _node_repr(n: Any) -> str:
    """Stable representation for node hashing and sorting."""
    return repr(_stable_json(n))


def node_set_checksum(
    G: "nx.Graph", nodes: Iterable[Any] | None = None, *, presorted: bool = False
) -> str:
    """Return a BLAKE2b checksum of ``G``'s node set using a stable ``repr``."""
    hasher = hashlib.blake2b(digest_size=16)

    node_iter = nodes if nodes is not None else G.nodes()
    serialised = (
        (_node_repr(n) for n in node_iter)
        if presorted
        else sorted(_node_repr(n) for n in node_iter)
    )
    for idx, node_repr in enumerate(serialised):
        if idx:
            hasher.update(b"|")
        hasher.update(node_repr.encode("utf-8"))
    return hasher.hexdigest()


def _ensure_node_map(G, *, key: str, sort: bool = False) -> Dict[Any, int]:
    """Return cached node→index mapping for ``G`` under ``key``.

    ``sort`` controls whether nodes are ordered by their string representation
    before assigning indices.
    """

    nodes = list(G.nodes())
    if sort:
        nodes.sort(key=_node_repr)
    checksum = node_set_checksum(G, nodes, presorted=sort)
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
    G: Any, key: str, builder: Callable[[], T], *, max_entries: int | None = None
) -> T:
    """Return cached ``builder`` output tied to the edge version of ``G``.

    All cache access is serialized via ``_EDGE_CACHE_LOCK`` to ensure
    thread-safety. When ``max_entries`` is a positive integer, only the most
    recent ``max_entries`` cache entries are kept. The potentially expensive
    ``builder`` is executed outside the lock when the cache is cold.
    """
    graph = G.graph if hasattr(G, "graph") else G
    edge_version = int(graph.get("_edge_version", 0))
    with _EDGE_CACHE_LOCK:
        cache_dict: OrderedDict = graph.setdefault("_edge_version_cache", OrderedDict())
        entry = cache_dict.get(key)
        if entry is not None and entry[0] == edge_version:
            if max_entries is not None and max_entries > 0:
                cache_dict.move_to_end(key)
            return entry[1]

    value = builder()
    with _EDGE_CACHE_LOCK:
        cache_dict = graph.setdefault("_edge_version_cache", OrderedDict())
        cache_dict[key] = (edge_version, value)
        if max_entries is not None and max_entries > 0:
            cache_dict.move_to_end(key)
            while len(cache_dict) > max_entries:
                cache_dict.popitem(last=False)
    return value


def cached_nodes_and_A(
    G: "nx.Graph", *, cache_size: int | None = 1
) -> tuple[list[int], Any]:
    """Return list of nodes and adjacency matrix for ``G`` with caching."""
    nodes_list = list(G.nodes())
    checksum = node_set_checksum(G, nodes_list)
    key = f"_dnfr_{len(nodes_list)}_{checksum}"
    G.graph["_dnfr_nodes_checksum"] = checksum

    def builder() -> tuple[list[int], Any]:
        nodes = nodes_list
        np = get_numpy()
        if np is not None:
            import networkx as nx  # importación tardía

            A = nx.to_numpy_array(G, nodelist=nodes, weight=None, dtype=float)
        else:  # pragma: no cover - dependiente de numpy
            A = None
        return nodes, A

    return edge_version_cache(G, key, builder, max_entries=cache_size)


def increment_edge_version(G: Any) -> None:
    """Increment the edge version counter in ``G.graph``."""
    graph = G.graph if hasattr(G, "graph") else G
    graph["_edge_version"] = int(graph.get("_edge_version", 0)) + 1
    for key in (
        "_neighbors",
        "_neighbors_version",
        "_cos_th",
        "_sin_th",
        "_thetas",
        "_trig_cache",
        "_trig_version",
    ):
        graph.pop(key, None)
