"""Helper functions."""

from __future__ import annotations
from typing import (
    Iterable,
    Sequence,
    Dict,
    Any,
    Callable,
    TypeVar,
    Optional,
    overload,
    Protocol,
    TYPE_CHECKING,
)
import logging
import math
import json
import hashlib
import random
import struct
from functools import lru_cache, partial
from statistics import fmean, StatisticsError
from json import JSONDecodeError
from pathlib import Path
from collections import OrderedDict

from .import_utils import optional_import, get_numpy

if TYPE_CHECKING:  # pragma: no cover - solo para type checkers
    import networkx as nx


tomllib = optional_import("tomllib") or optional_import("tomli")
if tomllib is not None:
    TOMLDecodeError = getattr(tomllib, "TOMLDecodeError", Exception)
    has_toml = True
else:  # pragma: no cover - depende de tomllib/tomli
    has_toml = False

    class TOMLDecodeError(Exception):
        pass

yaml = optional_import("yaml")
if yaml is not None:
    YAMLError = getattr(yaml, "YAMLError", Exception)
else:  # pragma: no cover - depende de pyyaml

    class YAMLError(Exception):
        pass


from .constants import (
    DEFAULTS,
    ALIAS_VF,
    ALIAS_THETA,
    ALIAS_DNFR,
    ALIAS_dEPI,
    ALIAS_SI,
    ALIAS_D2EPI,
)
from .collections_utils import (
    MAX_MATERIALIZE_DEFAULT,
    ensure_collection,
    normalize_weights,
    normalize_counter,
    mix_groups,
)
from .value_utils import _convert_value

PI = math.pi
TWO_PI = 2 * PI

T = TypeVar("T")

__all__ = [
    "MAX_MATERIALIZE_DEFAULT",
    "read_structured_file",
    "safe_write",
    "StructuredFileError",
    "ensure_collection",
    "clamp",
    "clamp01",
    "list_mean",
    "angle_diff",
    "normalize_weights",
    "alias_get",
    "alias_set",
    "get_attr",
    "set_attr",
    "get_attr_str",
    "set_attr_str",
    "set_attr_with_max",
    "set_vf",
    "set_dnfr",
    "neighbor_mean",
    "neighbor_phase_mean",
    "push_glyph",
    "recent_glyph",
    "ensure_history",
    "last_glyph",
    "count_glyphs",
    "normalize_counter",
    "mix_groups",
    "compute_dnfr_accel_max",
    "compute_coherence",
    "get_Si_weights",
    "precompute_trigonometry",
    "compute_Si_node",
    "compute_Si",
    "ensure_node_offset_map",
    "cached_nodes_and_A",
    "increment_edge_version",
    "node_set_checksum",
    "get_rng",
]

# -------------------------
# Generadores pseudoaleatorios
# -------------------------


@lru_cache(maxsize=DEFAULTS["JITTER_CACHE_SIZE"])
def get_rng(seed: int, key: int) -> random.Random:
    """Return a cached ``random.Random`` for ``(seed, key)``.

    A stable hash combines both values to guarantee reproducibility across
    runs and avoid ``PYTHONHASHSEED`` collisions.
    """

    seed_bytes = struct.pack(
        ">QQ",
        int(seed) & 0xFFFFFFFFFFFFFFFF,
        int(key) & 0xFFFFFFFFFFFFFFFF,
    )
    seed_int = int.from_bytes(
        hashlib.blake2b(seed_bytes, digest_size=8).digest(), "big"
    )
    return random.Random(seed_int)

# -------------------------
# Entrada/salida estructurada
# -------------------------


def _missing_dependency(name: str) -> Callable[[str], Any]:
    def _raise(_: str) -> Any:
        raise ImportError(f"{name} no está instalado")

    return _raise


PARSERS: Dict[str, Callable[[str], Any]] = {".json": json.loads}


def _get_parser(suffix: str) -> Callable[[str], Any]:
    parser = PARSERS.get(suffix)
    if parser is not None:
        return parser
    if suffix in (".yaml", ".yml"):
        parser = getattr(yaml, "safe_load", _missing_dependency("pyyaml"))
    elif suffix == ".toml":
        parser = getattr(tomllib, "loads", _missing_dependency("tomllib/tomli"))
    else:
        raise KeyError(suffix)
    PARSERS[suffix] = parser
    return parser


def _format_structured_file_error(path: Path, e: Exception) -> str:
    """Return a formatted error message for ``path``.

    Centralises message generation when handling different exceptions raised
    while reading or parsing structured files.
    """

    if isinstance(e, OSError):
        return f"No se pudo leer {path}: {e}"
    if isinstance(e, UnicodeDecodeError):
        return f"Error de codificación al leer {path}: {e}"
    if isinstance(e, JSONDecodeError):
        return f"Error al parsear archivo JSON en {path}: {e}"
    if isinstance(e, YAMLError):
        return f"Error al parsear archivo YAML en {path}: {e}"
    if has_toml and isinstance(e, TOMLDecodeError):
        return f"Error al parsear archivo TOML en {path}: {e}"
    if isinstance(e, ImportError):
        return f"Dependencia faltante al parsear {path}: {e}"
    return f"Error al parsear {path}: {e}"


class StructuredFileError(Exception):
    """Error while reading or parsing a structured file.

    The original exception is available via ``__cause__``.
    """

    def __init__(self, path: Path, original: Exception):
        super().__init__(_format_structured_file_error(path, original))
        self.path = path


def read_structured_file(path: Path) -> Any:
    """Read a JSON, YAML or TOML file and return parsed data.

    Raises
    ------
    StructuredFileError
        If a read or parse error occurs. The original exception is exposed as
        ``__cause__``.
    ValueError
        If the file extension is not supported.
    """

    suffix = path.suffix.lower()
    try:
        parser = _get_parser(suffix)
    except KeyError as e:
        raise ValueError(f"Extensión de archivo no soportada: {suffix}") from e
    try:
        text = path.read_text(encoding="utf-8")
        return parser(text)
    except Exception as e:
        raise StructuredFileError(path, e) from e


def safe_write(
    path: str | Path,
    write: Callable[[Any], Any],
    *,
    mode: str = "w",
    encoding: str | None = "utf-8",
    **open_kwargs: Any,
) -> None:
    """Write to ``path`` ensuring parent directory exists and handle errors.

    ``write`` receives a file object opened at ``path``. Any :class:`OSError`
    is re-raised with a descriptive message.
    """

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    open_params = dict(mode=mode, **open_kwargs)
    if encoding is not None:
        open_params["encoding"] = encoding
    try:
        with open(path, **open_params) as f:
            write(f)
    except OSError as e:
        raise OSError(f"Failed to write file {path}: {e}") from e


# -------------------------
# Grafos
# -------------------------


def _stable_json(obj: Any, visited: set[int] | None = None) -> Any:
    """Helper to obtain a JSON-serialisable structure for ``obj``.

    The default :func:`json.dumps` behaviour falls back to ``obj.__dict__``
    when available and otherwise uses ``repr(obj)`` which may include
    memory addresses. This function walks basic containers and objects to
    build a representation that avoids such non-deterministic data.
    """

    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj

    if visited is None:
        visited = set()

    obj_id = id(obj)
    if obj_id in visited:
        return "<recursion>"
    visited.add(obj_id)

    if isinstance(obj, (list, tuple, set)):
        return [_stable_json(o, visited) for o in obj]
    if isinstance(obj, dict):
        return {str(k): _stable_json(v, visited) for k, v in obj.items()}
    if hasattr(obj, "__dict__"):
        return {k: _stable_json(v, visited) for k, v in vars(obj).items()}
    return f"{obj.__module__}.{obj.__class__.__qualname__}"


def node_set_checksum(
    G: "nx.Graph", nodes: Iterable[Any] | None = None
) -> str:
    """Return the SHA1 of ``G``'s node set in sorted order.

    ``nodes`` allows reusing a previously materialised collection to avoid
    iterating over ``G`` twice.

    Each node is serialised to JSON with sorted keys and without memory
    addresses, ensuring a stable representation of the set.
    """

    hasher = hashlib.blake2b(digest_size=16)

    def serialise(n: Any) -> str:
        return json.dumps(
            _stable_json(n),
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        )

    node_iter = nodes if nodes is not None else G.nodes()
    sorted_nodes = sorted(node_iter, key=lambda n: serialise(n))
    for idx, n in enumerate(sorted_nodes):
        if idx:
            hasher.update(b"|")
        hasher.update(serialise(n).encode("utf-8"))
    return hasher.hexdigest()


def ensure_node_offset_map(G) -> Dict[Any, int]:
    """Return cached node→index mapping for ``G``.

    The mapping follows the natural insertion order of ``G.nodes`` for speed.
    When ``G.graph['SORT_NODES']`` is true a deterministic sort is applied.
    A checksum of the node set is stored so the mapping is recomputed only
    when the nodes change.
    """

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


def cached_nodes_and_A(
    G: "nx.Graph", *, cache_size: int | None = 1
) -> tuple[list[int], Any]:
    """Return list of nodes and adjacency matrix for ``G`` with caching.

    Information is stored in ``G.graph['_dnfr_cache']`` and reused while the
    graph structure remains unchanged. ``cache_size`` limits the number of
    entries per graph (``None`` or values <= 0 imply no limit). The node set
    is signed deterministically to ensure stable cache keys across runs.
    """

    cache: OrderedDict = G.graph.setdefault("_dnfr_cache", OrderedDict())
    nodes_list = list(G.nodes())
    checksum = node_set_checksum(G, nodes_list)

    last_checksum = G.graph.get("_dnfr_nodes_checksum")
    if last_checksum != checksum:
        cache.clear()
        G.graph["_dnfr_nodes_checksum"] = checksum

    key = (int(G.graph.get("_edge_version", 0)), len(nodes_list), checksum)
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


# -------------------------
# Iterables y colecciones
# -------------------------
# -------------------------
# Utilidades numéricas
# -------------------------


def clamp(x: float, a: float, b: float) -> float:
    """Clamp ``x`` to the closed interval [a, b]."""
    return min(max(x, a), b)


def clamp01(x: float) -> float:
    """Clamp ``x`` to the [0, 1] range."""
    return clamp(x, 0.0, 1.0)


def list_mean(xs: Iterable[float], default: float = 0.0) -> float:
    """Arithmetic mean or ``default`` if ``xs`` is empty."""
    try:
        return fmean(xs)
    except StatisticsError:
        return default


def _wrap_angle(a: float) -> float:
    """Wrap angle to (-π, π]."""
    return (a + PI) % TWO_PI - PI


def angle_diff(a: float, b: float) -> float:
    """Minimum difference between ``a`` and ``b`` in (-π, π]."""
    return _wrap_angle(a - b)


# -------------------------
# Acceso a atributos con alias
# -------------------------
def _validate_aliases(aliases: Sequence[str]) -> tuple[str, ...]:
    """Validate ``aliases`` and return them as a tuple of strings.

    This helper is intended to be used when building alias collections for
    :func:`alias_get`, :func:`alias_set` and demás utilidades públicas. A
    pre-existing tuple is returned unchanged; other sequences are converted to
    tuples. The result is guaranteed to be a non-empty tuple of strings.
    """

    if isinstance(aliases, str) or not isinstance(aliases, Sequence):
        raise TypeError("'aliases' must be a non-string sequence")

    seq = aliases if isinstance(aliases, tuple) else tuple(aliases)
    if not seq or any(not isinstance(a, str) for a in seq):
        if not seq:
            raise ValueError("'aliases' must contain at least one key")
        raise TypeError("'aliases' elements must be strings")
    return seq


def _alias_lookup(
    d: Dict[str, Any],
    aliases: tuple[str, ...],
    conv: Callable[[Any], T],
    *,
    default: Optional[Any] = None,
    strict: bool = False,
    log_level: int | None = None,
) -> Optional[T]:
    for key in aliases:
        if key in d:
            ok, val = _convert_value(
                d[key], conv, strict=strict, key=key, log_level=log_level
            )
            if ok:
                return val
    if default is None:
        return None
    ok, val = _convert_value(
        default,
        conv,
        strict=strict,
        key="default",
        log_level=logging.WARNING if not strict else log_level,
    )
    return val if ok else None


@overload
def alias_get(
    d: Dict[str, Any],
    aliases: Sequence[str],
    conv: Callable[[Any], T],
    *,
    default: T,
    strict: bool = False,
    log_level: int | None = None,
) -> T: ...


@overload
def alias_get(
    d: Dict[str, Any],
    aliases: Sequence[str],
    conv: Callable[[Any], T],
    *,
    default: None = ...,
    strict: bool = False,
    log_level: int | None = None,
) -> Optional[T]: ...


def alias_get(
    d: Dict[str, Any],
    aliases: Sequence[str],
    conv: Callable[[Any], T],
    *,
    default: Optional[Any] = None,
    strict: bool = False,
    log_level: int | None = None,
) -> Optional[T]:
    """Return the value for the first existing key in ``aliases``.

    ``aliases`` may be any sequence of strings. Non-tuple sequences are
    validated internally via :func:`_validate_aliases`, which guarantees a
    non-empty sequence of strings.

    If none of the keys are present or conversion fails, ``default`` is
    returned (converted or ``None`` if ``default`` is ``None``).

    ``log_level`` controls the logging level when conversion fails in lax
    mode.
    """
    return _alias_get(
        d,
        aliases,
        conv=conv,
        default=default,
        strict=strict,
        log_level=log_level,
    )


def alias_set(
    d: Dict[str, Any],
    aliases: Sequence[str],
    conv: Callable[[Any], T],
    value: Any,
) -> T:
    """Assign ``value`` converted to the first available key in ``aliases``.

    ``aliases`` may be any sequence of strings. Non-tuples are validated via
    :func:`_validate_aliases`.
    """
    if not isinstance(aliases, tuple):
        aliases = _validate_aliases(aliases)
    _, val = _convert_value(value, conv, strict=True)
    if val is None:
        raise ValueError("conversion yielded None")
    key = next((k for k in aliases if k in d), aliases[0])
    d[key] = val
    return val


class _Getter(Protocol[T]):
    @overload
    def __call__(
        self,
        d: Dict[str, Any],
        aliases: Sequence[str],
        default: T = ...,  # noqa: D401 - documented in alias_get
        *,
        strict: bool = False,
        log_level: int | None = None,
    ) -> T: ...

    @overload
    def __call__(
        self,
        d: Dict[str, Any],
        aliases: Sequence[str],
        default: None,
        *,
        strict: bool = False,
        log_level: int | None = None,
    ) -> Optional[T]: ...


@overload
def _alias_get(
    d: Dict[str, Any],
    aliases: Sequence[str],
    *,
    conv: Callable[[Any], T],
    default: T,
    strict: bool = False,
    log_level: int | None = None,
) -> T: ...


@overload
def _alias_get(
    d: Dict[str, Any],
    aliases: Sequence[str],
    *,
    conv: Callable[[Any], T],
    default: None = ...,  # noqa: D401 - documented in alias_get
    strict: bool = False,
    log_level: int | None = None,
) -> Optional[T]: ...


def _alias_get(
    d: Dict[str, Any],
    aliases: Sequence[str],
    *,
    conv: Callable[[Any], T],
    default: Optional[T] = None,
    strict: bool = False,
    log_level: int | None = None,
) -> Optional[T]:
    if not isinstance(aliases, tuple):
        aliases = _validate_aliases(aliases)
    return _alias_lookup(
        d,
        aliases,
        conv,
        default=default,
        strict=strict,
        log_level=log_level,
    )


def _alias_get_set(
    conv: Callable[[Any], T],
    *,
    default: T,
) -> tuple[_Getter[T], Callable[..., T]]:
    """Create alias ``get``/``set`` functions using ``conv``.

    Parameters
    ----------
    conv:
        Conversion function applied to retrieved values.
    default:
        Default value used when the key is missing.
    """

    _base_get = partial(_alias_get, conv=conv)

    def _get(
        d: Dict[str, Any],
        aliases: Sequence[str],
        default: Optional[T] = default,
        *,
        strict: bool = False,
        log_level: int | None = None,
    ) -> Optional[T]:
        """Obtain an attribute using :func:`alias_get`."""
        return _base_get(
            d,
            aliases,
            default=default,
            strict=strict,
            log_level=log_level,
        )

    def _set(d: Dict[str, Any], aliases: Sequence[str], value: T) -> T:
        """Set an attribute using :func:`alias_set`."""
        return alias_set(d, aliases, conv, value)

    return _get, _set


get_attr, set_attr = _alias_get_set(float, default=0.0)
get_attr_str, set_attr_str = _alias_get_set(str, default="")


# -------------------------
# Máximos globales con caché
# -------------------------


def _recompute_abs_max(G, aliases: tuple[str, ...]):
    """Recalcula y retorna ``(max_val, node)`` para ``aliases``."""
    node = max(
        G.nodes(),
        key=lambda m: abs(get_attr(G.nodes[m], aliases, 0.0)),
        default=None,
    )
    max_val = (
        abs(get_attr(G.nodes[node], aliases, 0.0)) if node is not None else 0.0
    )
    return max_val, node


def _update_cached_abs_max(
    G, aliases: tuple[str, ...], n, value, *, key: str
) -> None:
    """Update ``G.graph[key]`` and ``G.graph[f"{key}_node"]``."""
    node_key = f"{key}_node"
    val = abs(value)
    cur = float(G.graph.get(key, 0.0))
    cur_node = G.graph.get(node_key)
    if val >= cur:
        G.graph[key] = val
        G.graph[node_key] = n
    elif cur_node == n and val < cur:
        max_val, max_node = _recompute_abs_max(G, aliases)
        G.graph[key] = max_val
        G.graph[node_key] = max_node


def set_attr_with_max(
    G, n, aliases: tuple[str, ...], value: float, *, cache: str
) -> None:
    """Assign ``value`` to the given attribute and update the global maximum.

    ``aliases`` must be an immutable tuple of valid keys.
    """
    val = float(value)
    set_attr(G.nodes[n], aliases, val)
    _update_cached_abs_max(G, aliases, n, val, key=cache)


def set_vf(G, n, value: float) -> None:
    """Set ``νf`` and update the global maximum."""
    set_attr_with_max(G, n, ALIAS_VF, value, cache="_vfmax")


def set_dnfr(G, n, value: float) -> None:
    """Set ``ΔNFR`` and update the global maximum."""
    set_attr_with_max(G, n, ALIAS_DNFR, value, cache="_dnfrmax")


# -------------------------
# Normalizadores de ΔNFR y aceleración
# -------------------------


def compute_dnfr_accel_max(G) -> dict:
    """Compute absolute maxima of |ΔNFR| and |d²EPI/dt²|.

    Returns a dictionary with keys ``dnfr_max`` and ``accel_max``. If the graph
    has no nodes both values are ``0.0``.
    """

    dnfr_max = 0.0
    accel_max = 0.0
    for _, nd in G.nodes(data=True):
        dnfr_max = max(dnfr_max, abs(get_attr(nd, ALIAS_DNFR, 0.0)))
        accel_max = max(accel_max, abs(get_attr(nd, ALIAS_D2EPI, 0.0)))
    return {"dnfr_max": float(dnfr_max), "accel_max": float(accel_max)}


# -------------------------
# Coherencia global
# -------------------------


def compute_coherence(G) -> float:
    """Compute global coherence C(t) from ΔNFR and dEPI."""
    dnfr_sum = 0.0
    depi_sum = 0.0
    count = 0
    for _, nd in G.nodes(data=True):
        dnfr_sum += abs(get_attr(nd, ALIAS_DNFR, 0.0))
        depi_sum += abs(get_attr(nd, ALIAS_dEPI, 0.0))
        count += 1
    if count:
        dnfr_mean = dnfr_sum / count
        depi_mean = depi_sum / count
    else:
        dnfr_mean = depi_mean = 0.0
    return 1.0 / (1.0 + dnfr_mean + depi_mean)


# -------------------------
# Estadísticos vecinales
# -------------------------


def neighbor_mean(
    G, n, aliases: tuple[str, ...], default: float = 0.0
) -> float:
    """Mean of ``aliases`` attribute among neighbours of ``n``."""
    vals = (get_attr(G.nodes[v], aliases, default) for v in G.neighbors(n))
    return list_mean(vals, default)


def neighbor_phase_mean(obj, n=None) -> float:
    """Circular mean of neighbour phases.

    Accepts a :class:`NodoProtocol` or a ``(G, n)`` pair from ``networkx``. The
    latter is wrapped in :class:`NodoNX` to reuse the same logic.
    """

    from .node import NodoNX  # importación local para evitar ciclo

    if n is not None:
        node = NodoNX(obj, n)
    else:
        node = obj  # se asume NodoProtocol

    x = y = 0.0
    count = 0
    for v in node.neighbors():
        if hasattr(v, "theta"):
            th = getattr(v, "theta", 0.0)
        else:
            th = NodoNX.from_graph(
                node.G, v
            ).theta  # type: ignore[attr-defined]
        x += math.cos(th)
        y += math.sin(th)
        count += 1
    if count == 0:
        return getattr(node, "theta", 0.0)
    return math.atan2(y, x)


# -------------------------
# Historial de glyphs por nodo
# -------------------------

# Importaciones diferidas para evitar ciclos al definir ``get_attr_str`` arriba
from .glyph_history import (  # noqa: E402
    push_glyph,
    recent_glyph,
    ensure_history,
    last_glyph,
    count_glyphs,
)

# -------------------------
# Índice de sentido (Si)
# -------------------------


def get_Si_weights(G: Any) -> tuple[float, float, float]:
    """Obtain and normalise weights for the sense index."""
    w = {**DEFAULTS["SI_WEIGHTS"], **G.graph.get("SI_WEIGHTS", {})}
    weights = normalize_weights(w, ("alpha", "beta", "gamma"), default=0.0)
    alpha = weights["alpha"]
    beta = weights["beta"]
    gamma = weights["gamma"]
    G.graph["_Si_weights"] = weights
    G.graph["_Si_sensitivity"] = {
        "dSi_dvf_norm": alpha,
        "dSi_ddisp_fase": -beta,
        "dSi_ddnfr_norm": -gamma,
    }
    return alpha, beta, gamma


def precompute_trigonometry(
    G: Any,
) -> tuple[Dict[Any, float], Dict[Any, float], Dict[Any, float]]:
    """Precompute cosines and sines of ``θ`` per node.

    Values are stored in ``G.graph`` and reused while the graph structure
    (versioned via ``"_edge_version"``) remains unchanged.
    """

    graph = G.graph
    edge_version = int(graph.get("_edge_version", 0))
    cached_version = graph.get("_trig_version")

    cos_th = graph.get("_cos_th")
    sin_th = graph.get("_sin_th")
    thetas = graph.get("_thetas")

    if (
        cached_version == edge_version
        and cos_th is not None
        and sin_th is not None
        and thetas is not None
    ):
        return cos_th, sin_th, thetas

    cos_th = {}
    sin_th = {}
    thetas = {}
    for n, nd in G.nodes(data=True):
        th = get_attr(nd, ALIAS_THETA, 0.0)
        thetas[n] = th
        cos_th[n] = math.cos(th)
        sin_th[n] = math.sin(th)

    graph["_cos_th"] = cos_th
    graph["_sin_th"] = sin_th
    graph["_thetas"] = thetas
    graph["_trig_version"] = edge_version
    return cos_th, sin_th, thetas


def compute_Si_node(
    n: Any,
    nd: Dict[str, Any],
    *,
    alpha: float,
    beta: float,
    gamma: float,
    vfmax: float,
    dnfrmax: float,
    cos_th: Dict[Any, float],
    sin_th: Dict[Any, float],
    thetas: Dict[Any, float],
    neighbors: Dict[Any, Sequence[Any]],
    inplace: bool,
) -> float:
    """Compute ``Si`` for a single node."""
    vf = get_attr(nd, ALIAS_VF, 0.0)
    vf_norm = clamp01(abs(vf) / vfmax)

    th_i = thetas[n]
    neigh = neighbors[n]
    deg = len(neigh)
    if deg:
        sum_cos = sum(cos_th[v] for v in neigh)
        sum_sin = sum(sin_th[v] for v in neigh)
        mean_cos = sum_cos / deg
        mean_sin = sum_sin / deg
        th_bar = math.atan2(mean_sin, mean_cos)
    else:
        th_bar = th_i
    disp_fase = abs(angle_diff(th_i, th_bar)) / math.pi

    dnfr = get_attr(nd, ALIAS_DNFR, 0.0)
    dnfr_norm = clamp01(abs(dnfr) / dnfrmax)

    Si = alpha * vf_norm + beta * (1.0 - disp_fase) + gamma * (1.0 - dnfr_norm)
    Si = clamp01(Si)
    if inplace:
        set_attr(nd, ALIAS_SI, Si)
    return Si


def compute_Si(G, *, inplace: bool = True) -> Dict[Any, float]:
    """Compute ``Si`` per node and write it to ``G.nodes[n]["Si"]``.

    Formula:
        Si = α·νf_norm + β·(1 - disp_fase_local) + γ·(1 - |ΔNFR|/max|ΔNFR|)
    Also stores normalised weights and partial sensitivity (∂Si/∂component)
    in ``G.graph``.
    """
    graph = G.graph
    edge_version = int(graph.get("_edge_version", 0))

    neighbors = graph.get("_neighbors")
    if graph.get("_neighbors_version") != edge_version or neighbors is None:
        neighbors = {n: list(G.neighbors(n)) for n in G}
        graph["_neighbors"] = neighbors
        graph["_neighbors_version"] = edge_version

    alpha, beta, gamma = get_Si_weights(G)

    # Normalización de νf y ΔNFR en red usando máximos cacheados
    vfmax = G.graph.get("_vfmax")
    if vfmax is None:
        vfmax, vf_node = _recompute_abs_max(G, ALIAS_VF)
        G.graph.setdefault("_vfmax", vfmax)
        G.graph.setdefault("_vfmax_node", vf_node)
    dnfrmax = G.graph.get("_dnfrmax")
    if dnfrmax is None:
        dnfrmax, dnfr_node = _recompute_abs_max(G, ALIAS_DNFR)
        G.graph.setdefault("_dnfrmax", dnfrmax)
        G.graph.setdefault("_dnfrmax_node", dnfr_node)
    vfmax = 1.0 if vfmax == 0 else vfmax
    dnfrmax = 1.0 if dnfrmax == 0 else dnfrmax

    # precálculo de cosenos y senos de cada nodo
    cos_th, sin_th, thetas = precompute_trigonometry(G)

    out: Dict[Any, float] = {}
    for n, nd in G.nodes(data=True):
        out[n] = compute_Si_node(
            n,
            nd,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            vfmax=vfmax,
            dnfrmax=dnfrmax,
            cos_th=cos_th,
            sin_th=sin_th,
            thetas=thetas,
            neighbors=neighbors,
            inplace=inplace,
        )
    return out


def increment_edge_version(G: Any) -> None:
    """Increment the edge version counter in ``G.graph``.

    Accepts an ``nx.Graph`` or a dictionary acting as ``G.graph`` and updates
    ``"_edge_version"`` to invalidate edge-dependent caches.
    """
    graph = G.graph if hasattr(G, "graph") else G
    graph["_edge_version"] = int(graph.get("_edge_version", 0)) + 1
    # eliminar caches dependientes de la estructura
    for key in (
        "_neighbors",
        "_neighbors_version",
        "_cos_th",
        "_sin_th",
        "_thetas",
        "_trig_version",
    ):
        graph.pop(key, None)
