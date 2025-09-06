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
from functools import lru_cache
import threading

from .import_utils import get_numpy
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


@lru_cache(maxsize=1)
def _get_nodonx():
    from .node import NodoNX
    return NodoNX

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
    v, cache: Dict[int, tuple[float, float] | None]
) -> tuple[float, float] | None:
    """Return ``(cos, sin)`` for ``v`` caching by object id."""
    key = id(v)
    val = cache.get(key)
    if val is not None:
        return val
    data = getattr(v, "__dict__", v if isinstance(v, dict) else {})
    th = get_attr(data, ALIAS_THETA, None)
    if th is None:
        cache[key] = None
        return None
    val = (math.cos(th), math.sin(th))
    cache[key] = val
    return val


def _neighbor_phase_mean_graph(node) -> float:
    from .metrics_utils import precompute_trigonometry

    G = node.G
    trig = precompute_trigonometry(G)
    cos_th, sin_th = trig.cos, trig.sin
    x = y = 0.0
    count = 0
    for v in node.neighbors():
        x += cos_th[v]
        y += sin_th[v]
        count += 1
    if count == 0:
        return get_attr(G.nodes[node.n], ALIAS_THETA, 0.0)
    return math.atan2(y, x)


def _neighbor_phase_mean_generic(
    node, cache: Dict[int, tuple[float, float] | None]
) -> float:
    x = y = 0.0
    count = 0
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
    NodoNX = _get_nodonx()

    if n is not None:
        node = NodoNX(obj, n)
    else:
        node = obj  # se asume NodoProtocol

    if getattr(node, "G", None) is not None:
        return _neighbor_phase_mean_graph(node)
    cache = getattr(node, "_trig_cache", None)
    if cache is None:
        cache = {}
        setattr(node, "_trig_cache", cache)
    return _neighbor_phase_mean_generic(node, cache)


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


def node_set_checksum(
    G: "nx.Graph", nodes: Iterable[Any] | None = None
) -> str:
    """Return a BLAKE2b checksum of ``G``'s node set using a stable ``repr``."""
    hasher = hashlib.blake2b(digest_size=16)

    def serialise(n: Any) -> str:
        return repr(_stable_json(n))

    node_iter = nodes if nodes is not None else G.nodes()
    for idx, node_repr in enumerate(sorted(serialise(n) for n in node_iter)):
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
    checksum = node_set_checksum(G, nodes)
    mapping = G.graph.get(key)
    checksum_key = f"{key}_checksum"
    if mapping is None or G.graph.get(checksum_key) != checksum:
        if sort:
            nodes.sort(key=lambda x: str(x))
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

    When ``max_entries`` is set to a positive integer, only the most recent
    ``max_entries`` caches are kept; older entries are purged.
    """
    graph = G.graph if hasattr(G, "graph") else G
    edge_version = int(graph.get("_edge_version", 0))
    cache_dict: OrderedDict = graph.setdefault("_edge_version_cache", OrderedDict())
    entry = cache_dict.get(key)
    if entry is None or entry[0] != edge_version:
        with _EDGE_CACHE_LOCK:
            entry = cache_dict.get(key)
            if entry is None or entry[0] != edge_version:
                value = builder()
                cache_dict[key] = (edge_version, value)
            else:
                value = entry[1]
            if max_entries is not None and max_entries > 0:
                cache_dict.move_to_end(key)
                while len(cache_dict) > max_entries:
                    cache_dict.popitem(last=False)
    else:
        value = entry[1]
        if max_entries is not None and max_entries > 0:
            with _EDGE_CACHE_LOCK:
                cache_dict.move_to_end(key)
    return value


def cached_nodes_and_A(
    G: "nx.Graph", *, cache_size: int | None = 1
) -> tuple[list[int], Any]:
    """Return list of nodes and adjacency matrix for ``G`` with caching."""
    cache: OrderedDict = edge_version_cache(G, "_dnfr", OrderedDict)
    nodes_list = list(G.nodes())
    checksum = node_set_checksum(G, nodes_list)

    last_checksum = G.graph.get("_dnfr_nodes_checksum")
    if last_checksum != checksum:
        cache.clear()
        G.graph["_dnfr_nodes_checksum"] = checksum

    key = (len(nodes_list), checksum)
    nodes_and_A = cache.get(key)
    if nodes_and_A is None:
        nodes = nodes_list
        np = get_numpy()
        if np is not None:
            import networkx as nx  # importación tardía

            A = nx.to_numpy_array(G, nodelist=nodes, weight=None, dtype=float)
        else:  # pragma: no cover - dependiente de numpy
            A = None
        nodes_and_A = (nodes, A)
        cache[key] = nodes_and_A
        if cache_size is not None and cache_size > 0 and len(cache) > cache_size:
            cache.popitem(last=False)
    else:
        cache.move_to_end(key)

    return nodes_and_A


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

