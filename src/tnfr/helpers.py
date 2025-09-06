"""Helper functions."""

from __future__ import annotations
from typing import (
    Iterable,
    Sequence,
    Dict,
    Any,
    Optional,
    TYPE_CHECKING,
    Callable,
    TypeVar,
)
import math
import hashlib
from statistics import fmean, StatisticsError
from collections import OrderedDict

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


def neighbor_phase_mean(obj, n=None) -> float:
    """Circular mean of neighbour phases.

    Accepts a :class:`NodoProtocol` or a ``(G, n)`` pair from ``networkx``. The
    latter is wrapped in :class:`NodoNX` to reuse the same logic.

    When the underlying graph ``G`` is available, trigonometric values are
    obtained via :func:`precompute_trigonometry` to avoid recalculating
    :func:`math.cos` and :func:`math.sin` for each neighbour. Objects without
    access to ``G`` fall back to computing these values individually.
    """
    from .node import NodoNX  # importación local para evitar ciclo

    if n is not None:
        node = NodoNX(obj, n)
    else:
        node = obj  # se asume NodoProtocol

    x = y = 0.0
    count = 0
    G = getattr(node, "G", None)

    if G is not None:
        from .metrics_utils import precompute_trigonometry

        trig = precompute_trigonometry(G)
        cos_th, sin_th = trig.cos, trig.sin
        for v in node.neighbors():
            x += cos_th[v]
            y += sin_th[v]
            count += 1
    else:
        theta_cache: Dict[Any, float] = {}
        for v in node.neighbors():
            th = theta_cache.get(v)
            if th is None:
                if hasattr(v, "theta"):
                    th = getattr(v, "theta", 0.0)
                else:
                    th = NodoNX.from_graph(node.G, v).theta  # type: ignore[attr-defined]
                theta_cache[v] = th
            x += math.cos(th)
            y += math.sin(th)
            count += 1
    if count == 0:
        return getattr(node, "theta", 0.0)
    return math.atan2(y, x)


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
    """Return the SHA1 of ``G``'s node set using a stable ``repr``."""
    hasher = hashlib.blake2b(digest_size=16)

    def serialise(n: Any) -> str:
        return repr(_stable_json(n))

    node_iter = nodes if nodes is not None else G.nodes()
    serialised_nodes = [serialise(n) for n in node_iter]
    serialised_nodes.sort()
    for idx, node_repr in enumerate(serialised_nodes):
        if idx:
            hasher.update(b"|")
        hasher.update(node_repr.encode("utf-8"))
    return hasher.hexdigest()


def ensure_node_index_map(G) -> Dict[Any, int]:
    """Return cached node→index mapping for ``G``.

    The mapping is stored in ``G.graph['_node_index_map']`` and reused on
    subsequent calls. It is recalculated whenever the set of nodes in ``G``
    changes.
    """

    nodes = list(G.nodes())
    checksum = node_set_checksum(G, nodes)
    mapping = G.graph.get("_node_index_map")
    if mapping is None or G.graph.get("_node_index_checksum") != checksum:
        mapping = {node: idx for idx, node in enumerate(nodes)}
        G.graph["_node_index_map"] = mapping
        G.graph["_node_index_checksum"] = checksum
    return mapping


def ensure_node_offset_map(G) -> Dict[Any, int]:
    """Return cached node→index mapping for ``G``."""
    nodes = list(G.nodes())
    checksum = node_set_checksum(G, nodes)
    mapping = G.graph.get("_node_offset_map")
    if mapping is None or G.graph.get("_node_offset_checksum") != checksum:
        if bool(G.graph.get("SORT_NODES", False)):
            nodes.sort(key=lambda x: str(x))
        mapping = {node: idx for idx, node in enumerate(nodes)}
        G.graph["_node_offset_map"] = mapping
        G.graph["_node_offset_checksum"] = checksum
    return mapping


def edge_version_cache(G: Any, key: str, builder: Callable[[], T]) -> T:
    """Return cached ``builder`` output tied to the edge version of ``G``."""
    graph = G.graph if hasattr(G, "graph") else G
    edge_version = int(graph.get("_edge_version", 0))
    version_key = f"{key}_version"
    cache_key = f"{key}_cache"
    cached_version = graph.get(version_key)
    value = graph.get(cache_key)
    if cached_version != edge_version or value is None:
        value = builder()
        graph[cache_key] = value
        graph[version_key] = edge_version
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

