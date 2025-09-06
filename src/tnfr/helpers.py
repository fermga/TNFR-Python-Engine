"""Funciones auxiliares."""

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
)
import logging
import math
import json
import hashlib
import random
import struct
from functools import partial, lru_cache
from statistics import fmean, StatisticsError
from json import JSONDecodeError
from pathlib import Path
from collections import OrderedDict

import networkx as nx

from .import_utils import optional_import
np = optional_import("numpy")  # type: ignore

tomllib = optional_import("tomllib") or optional_import("tomli")  # type: ignore
if tomllib is not None:
    TOMLDecodeError = getattr(tomllib, "TOMLDecodeError", Exception)  # type: ignore[attr-defined]
    has_toml = True
else:
    class TOMLDecodeError(Exception):  # type: ignore
        pass
    has_toml = False

yaml = optional_import("yaml")  # type: ignore
if yaml is not None:
    YAMLError = getattr(yaml, "YAMLError", Exception)  # type: ignore[attr-defined]
else:
    class YAMLError(Exception):  # type: ignore
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
    "ensure_parent",
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


@lru_cache(maxsize=None)
def get_rng(seed: int, key: int) -> random.Random:
    """Devuelve un ``random.Random`` cacheado por ``(seed, key)``.

    Se utiliza un hash estable para combinar ambos valores, garantizando
    reproducibilidad entre ejecuciones y evitando colisiones con
    ``PYTHONHASHSEED``.
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


def _parse_json(text: str) -> Any:
    """Parsea ``text`` como JSON."""
    return json.loads(text)


def _parse_yaml(text: str) -> Any:
    """Parsea ``text`` como YAML."""
    if not yaml:  # pragma: no cover - dependencia opcional
        raise ImportError("pyyaml no está instalado")
    return yaml.safe_load(text)


def _parse_toml(text: str) -> Any:
    """Parsea ``text`` como TOML."""
    if not tomllib:  # pragma: no cover - dependencia opcional
        raise ImportError("tomllib/tomli no está instalado")
    return tomllib.loads(text)


PARSERS: Dict[str, Callable[[str], Any]] = {
    ".json": _parse_json,
    ".yaml": _parse_yaml,
    ".yml": _parse_yaml,
    ".toml": _parse_toml,
}


def _format_structured_file_error(path: Path, e: Exception) -> str:
    """Devuelve un mensaje de error formateado para ``path``.

    Esta función centraliza la lógica de generación de mensajes al manejar
    distintas excepciones ocurridas al leer o parsear archivos estructurados.
    """

    if isinstance(e, OSError):
        return f"No se pudo leer {path}: {e}"
    if isinstance(e, JSONDecodeError):
        return f"Error al parsear archivo JSON en {path}: {e}"
    if isinstance(e, YAMLError):
        return f"Error al parsear archivo YAML en {path}: {e}"
    if has_toml and isinstance(e, TOMLDecodeError):
        return f"Error al parsear archivo TOML en {path}: {e}"
    if isinstance(e, ImportError):
        return f"Dependencia faltante al parsear {path}: {e}"
    return f"Error al parsear {path}: {e}"


def read_structured_file(path: Path) -> Any:
    """Lee un archivo JSON, YAML o TOML y devuelve los datos parseados."""
    suffix = path.suffix.lower()
    if suffix not in PARSERS:
        raise ValueError(f"Extensión de archivo no soportada: {suffix}")
    parser = PARSERS[suffix]
    try:
        text = path.read_text(encoding="utf-8")
        return parser(text)
    except Exception as e:
        raise ValueError(_format_structured_file_error(path, e)) from e


def ensure_parent(path: str | Path) -> None:
    """Crea el directorio padre de ``path`` si hace falta."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)


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
    """Devuelve el SHA1 del conjunto de nodos de ``G`` ordenado.

    ``nodes`` permite reutilizar una colección ya obtenida para evitar recorrer
    los nodos de ``G`` dos veces cuando el llamador ya los materializó.

    Cada nodo se serializa a JSON con claves ordenadas y evitando incluir
    direcciones de memoria, asegurando así una representación estable del
    conjunto.
    """

    sha1 = hashlib.sha1()

    def serialise(n: Any) -> str:
        return json.dumps(
            _stable_json(n),
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        )

    node_iter = nodes if nodes is not None else G.nodes()
    serialised = sorted(serialise(n) for n in node_iter)
    sha1.update("|".join(serialised).encode("utf-8"))
    return sha1.hexdigest()


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
    G: nx.Graph, *, cache_size: int | None = 1
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
        if np is not None:
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
    """Constriñe ``x`` al intervalo cerrado [a, b]."""
    return a if x < a else b if x > b else x


def clamp01(x: float) -> float:
    """Ataja ``x`` a la banda [0, 1]."""
    return clamp(x, 0.0, 1.0)


def list_mean(xs: Iterable[float], default: float = 0.0) -> float:
    """Promedio aritmético o ``default`` si ``xs`` está vacío."""
    try:
        return fmean(xs)
    except StatisticsError:
        return default


def _wrap_angle(a: float) -> float:
    """Envuelve ángulo a (-π, π]."""
    return (a + PI) % TWO_PI - PI


def angle_diff(a: float, b: float) -> float:
    """Diferencia mínima entre ``a`` y ``b`` en (-π, π]."""
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
    default: None = ...,
    strict: bool = False,
    log_level: int | None = None,
) -> Optional[T]: ...


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


def alias_get(
    d: Dict[str, Any],
    aliases: Sequence[str],
    conv: Callable[[Any], T],
    *,
    default: Optional[Any] = None,
    strict: bool = False,
    log_level: int | None = None,
) -> Optional[T]:
    """Busca en ``d`` la primera clave de ``aliases`` y retorna el valor
    convertido.

    ``aliases`` puede ser cualquier secuencia de strings. Si no es una
    tupla inmutable, se validará internamente mediante
    :func:`_validate_aliases`. ``_validate_aliases`` garantiza que la
    secuencia no esté vacía y que todos sus elementos sean strings.

    Si ninguna de las claves está presente o la conversión falla,
    devuelve ``default`` convertido (o ``None`` si ``default`` es
    ``None``).

    ``log_level`` permite ajustar el nivel de logging cuando la conversión
    falla en modo laxo.
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
    """Asigna ``value`` convertido a la primera clave disponible de
    ``aliases``.

    ``aliases`` puede ser cualquier secuencia de strings. Si no es una
    tupla, se validará internamente mediante :func:`_validate_aliases`.
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
    """Crea funciones ``get``/``set`` para alias usando ``conv``.

    Parameters
    ----------
    conv:
        Función de conversión a aplicar al valor recuperado.
    default:
        Valor por defecto a utilizar cuando la clave no existe.
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
        """Obtiene un atributo usando :func:`alias_get`."""
        return _base_get(
            d,
            aliases,
            default=default,
            strict=strict,
            log_level=log_level,
        )

    def _set(d: Dict[str, Any], aliases: Sequence[str], value: T) -> T:
        """Establece un atributo usando :func:`alias_set`."""
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
    """Actualiza ``G.graph[key]`` y ``G.graph[f"{key}_node"]``."""
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
    """Asigna ``value`` al atributo indicado y actualiza el máximo global.

    ``aliases`` debe ser una tupla inmutable de claves válidas.
    """
    val = float(value)
    set_attr(G.nodes[n], aliases, val)
    _update_cached_abs_max(G, aliases, n, val, key=cache)


def set_vf(G, n, value: float) -> None:
    """Asigna ``νf`` y actualiza el máximo global."""
    set_attr_with_max(G, n, ALIAS_VF, value, cache="_vfmax")


def set_dnfr(G, n, value: float) -> None:
    """Asigna ``ΔNFR`` y actualiza el máximo global."""
    set_attr_with_max(G, n, ALIAS_DNFR, value, cache="_dnfrmax")


# -------------------------
# Normalizadores de ΔNFR y aceleración
# -------------------------


def compute_dnfr_accel_max(G) -> dict:
    """Calcula los máximos absolutos de |ΔNFR| y |d²EPI/dt²|.

    Devuelve un diccionario con las claves ``dnfr_max`` y ``accel_max``.
    Si el grafo no tiene nodos, ambos valores serán ``0.0``.
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
    """Calcula la coherencia global C(t) a partir de ΔNFR y dEPI."""
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
    """Promedio circular de las fases vecinales.

    Acepta un :class:`NodoProtocol` o un par ``(G, n)`` de ``networkx``. En el
    segundo caso se envuelve en :class:`NodoNX` para reutilizar la misma
    lógica.
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
    """Obtiene y normaliza los pesos del índice de sentido."""
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
    """Precálculo de cosenos y senos de ``θ`` por nodo.

    Los valores se almacenan en ``G.graph`` y se reutilizan mientras la
    estructura del grafo (versionada con ``"_edge_version"``) no cambie.
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
    """Calcula ``Si`` para un solo nodo."""
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
    """Calcula Si por nodo y lo escribe en G.nodes[n]["Si"].

    Fórmula:
        Si = α·νf_norm + β·(1 - disp_fase_local) + γ·(1 - |ΔNFR|/max|ΔNFR|)
    También guarda en ``G.graph`` los pesos normalizados y la
    sensibilidad parcial (∂Si/∂componente).
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
    """Incrementa el contador de versión de aristas en ``G.graph``.

    Acepta un ``nx.Graph`` o un diccionario que actúe como ``G.graph`` y
    actualiza ``"_edge_version"`` para invalidar caches dependientes de las
    aristas.
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
