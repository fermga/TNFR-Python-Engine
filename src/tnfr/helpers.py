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
from statistics import fmean, StatisticsError
import networkx as nx
from json import JSONDecodeError
from pathlib import Path

try:  # pragma: no cover - dependencia opcional
    import tomllib  # type: ignore[attr-defined]
    from tomllib import TOMLDecodeError  # type: ignore[attr-defined]
except ModuleNotFoundError:  # pragma: no cover
    try:
        import tomli as tomllib  # type: ignore
        from tomli import TOMLDecodeError  # type: ignore
    except ModuleNotFoundError:  # pragma: no cover
        tomllib = None  # type: ignore


try:  # pragma: no cover - dependencia opcional
    import yaml  # type: ignore
    from yaml import YAMLError  # type: ignore
except ImportError:  # pragma: no cover
    yaml = None

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

logger = logging.getLogger(__name__)

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
    "media_vecinal",
    "fase_media",
    "push_glyph",
    "recent_glyph",
    "ensure_history",
    "last_glyph",
    "count_glyphs",
    "normalize_counter",
    "mix_groups",
    "compute_dnfr_accel_max",
    "compute_coherence",
    "compute_Si",
    "increment_edge_version",
    "node_set_checksum",
]

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


def read_structured_file(path: Path) -> Any:
    """Lee un archivo JSON, YAML o TOML y devuelve los datos parseados."""
    suffix = path.suffix.lower()
    if suffix not in PARSERS:
        raise ValueError(f"Extensión de archivo no soportada: {suffix}")
    parser = PARSERS[suffix]
    try:
        text = path.read_text(encoding="utf-8")
        return parser(text)
    except OSError as e:
        raise ValueError(f"No se pudo leer {path}: {e}") from e
    except JSONDecodeError as e:
        raise ValueError(f"Error al parsear archivo JSON en {path}: {e}") from e
    except YAMLError as e:
        raise ValueError(f"Error al parsear archivo YAML en {path}: {e}") from e
    except TOMLDecodeError as e:
        raise ValueError(f"Error al parsear archivo TOML en {path}: {e}") from e
    except ImportError as e:
        raise ValueError(f"Dependencia faltante al parsear {path}: {e}") from e


def ensure_parent(path: str | Path) -> None:
    """Crea el directorio padre de ``path`` si hace falta."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)


# -------------------------
# Grafos
# -------------------------


def _stable_json(obj: Any) -> Any:
    """Helper to obtain a JSON-serialisable structure for ``obj``.

    The default :func:`json.dumps` behaviour falls back to ``obj.__dict__`` when
    available and otherwise uses ``repr(obj)`` which may include memory
    addresses.  This function walks basic containers and objects to build a
    representation that avoids such non-deterministic data.
    """

    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, (list, tuple, set)):
        return [_stable_json(o) for o in obj]
    if isinstance(obj, dict):
        return {str(k): _stable_json(v) for k, v in obj.items()}
    if hasattr(obj, "__dict__"):
        return {k: _stable_json(v) for k, v in vars(obj).items()}
    return obj.__class__.__qualname__


def node_set_checksum(G: nx.Graph) -> str:
    """Devuelve el SHA1 del conjunto de nodos de ``G`` ordenado.

    Cada nodo se serializa a JSON con claves ordenadas y evitando incluir
    direcciones de memoria, asegurando así una representación estable del
    conjunto.
    """

    sha1 = hashlib.sha1()

    def serialise(n: Any) -> str:
        return json.dumps(_stable_json(n), sort_keys=True, ensure_ascii=False)

    for i, node_repr in enumerate(sorted(serialise(n) for n in G.nodes())):
        if i:
            sha1.update(b"|")
        sha1.update(node_repr.encode("utf-8"))
    return sha1.hexdigest()


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
        raise TypeError("'aliases' must be a sequence of strings")
    seq = aliases if isinstance(aliases, tuple) else tuple(aliases)
    if not seq:
        raise ValueError("'aliases' must contain at least one key")
    if not all(isinstance(a, str) for a in seq):
        raise TypeError("'aliases' must be a sequence of strings")
    return seq


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
    """Busca en ``d`` la primera clave de ``aliases`` y retorna el valor convertido.

    ``aliases`` puede ser cualquier secuencia de strings. Si no es una tupla
    inmutable, se validará internamente mediante :func:`_validate_aliases`.
    ``_validate_aliases`` garantiza que la secuencia no esté vacía y que todos
    sus elementos sean strings.

    Si ninguna de las claves está presente o la conversión falla, devuelve
    ``default`` convertido (o ``None`` si ``default`` es ``None``).

    ``log_level`` permite ajustar el nivel de logging cuando la conversión
    falla en modo laxo.
    """
    if not isinstance(aliases, tuple):
        aliases = _validate_aliases(aliases)
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
        default, conv, strict=strict, key="default", log_level=log_level
    )
    return val if ok else None


def alias_set(
    d: Dict[str, Any],
    aliases: Sequence[str],
    conv: Callable[[Any], T],
    value: Any,
) -> T:
    """Asigna ``value`` convertido a la primera clave disponible de ``aliases``.

    ``aliases`` puede ser cualquier secuencia de strings. Si no es una tupla,
    se validará internamente mediante :func:`_validate_aliases`.
    """
    if not isinstance(aliases, tuple):
        aliases = _validate_aliases(aliases)
    _, val = _convert_value(value, conv, strict=True)
    if val is None:
        raise ValueError("conversion yielded None")
    for key in aliases:
        if key in d:
            d[key] = val
            return val
    key = aliases[0]
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
    ) -> T:
        ...

    @overload
    def __call__(
        self,
        d: Dict[str, Any],
        aliases: Sequence[str],
        default: None,
        *,
        strict: bool = False,
        log_level: int | None = None,
    ) -> Optional[T]:
        ...


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
    return alias_get(d, aliases, conv, default=default, strict=strict, log_level=log_level)


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

    def _get(
        d: Dict[str, Any],
        aliases: Sequence[str],
        default: Optional[T] = default,
        *,
        strict: bool = False,
        log_level: int | None = None,
    ) -> Optional[T]:
        """Obtiene un atributo usando :func:`alias_get`."""
        return _alias_get(
            d,
            aliases,
            conv=conv,
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
    max_val = abs(get_attr(G.nodes[node], aliases, 0.0)) if node is not None else 0.0
    return max_val, node


def _update_cached_abs_max(G, aliases: tuple[str, ...], n, value, *, key: str) -> None:
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


def media_vecinal(G, n, aliases: tuple[str, ...], default: float = 0.0) -> float:
    """Media del atributo indicado por ``aliases`` en los vecinos de ``n``."""
    vals = (get_attr(G.nodes[v], aliases, default) for v in G.neighbors(n))
    return list_mean(vals, default)


def fase_media(obj, n=None) -> float:
    """Promedio circular de las fases vecinales.

    Acepta un :class:`NodoProtocol` o un par ``(G, n)`` de ``networkx``. En el
    segundo caso se envuelve en :class:`NodoNX` para reutilizar la misma lógica.
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
            th = NodoNX.from_graph(node.G, v).theta  # type: ignore[attr-defined]
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


def compute_Si(G, *, inplace: bool = True) -> Dict[Any, float]:
    """Calcula Si por nodo y lo escribe en G.nodes[n]["Si"].

    Fórmula:
        Si = α·νf_norm + β·(1 - disp_fase_local) + γ·(1 - |ΔNFR|/max|ΔNFR|)
    También guarda en ``G.graph`` los pesos normalizados y la
    sensibilidad parcial (∂Si/∂componente).
    """
    neighbors = {n: list(G.neighbors(n)) for n in G}
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
    cos_th: Dict[Any, float] = {}
    sin_th: Dict[Any, float] = {}
    thetas: Dict[Any, float] = {}
    for n, nd in G.nodes(data=True):
        th = get_attr(nd, ALIAS_THETA, 0.0)
        thetas[n] = th
        cos_th[n] = math.cos(th)
        sin_th[n] = math.sin(th)

    out: Dict[Any, float] = {}
    for n, nd in G.nodes(data=True):
        vf = get_attr(nd, ALIAS_VF, 0.0)
        vf_norm = clamp01(abs(vf) / vfmax)

        # dispersión de fase local utilizando vecinos precomputados
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
        disp_fase = abs(angle_diff(th_i, th_bar)) / math.pi  # [0,1]

        dnfr = get_attr(nd, ALIAS_DNFR, 0.0)
        dnfr_norm = clamp01(abs(dnfr) / dnfrmax)

        Si = alpha * vf_norm + beta * (1.0 - disp_fase) + gamma * (1.0 - dnfr_norm)
        Si = clamp01(Si)
        out[n] = Si
        if inplace:
            set_attr(nd, ALIAS_SI, Si)
    return out


def increment_edge_version(G: Any) -> None:
    """Incrementa el contador de versión de aristas en ``G.graph``.

    Acepta un ``nx.Graph`` o un diccionario que actúe como ``G.graph`` y
    actualiza ``"_edge_version"`` para invalidar caches dependientes de las
    aristas.
    """
    graph = G.graph if hasattr(G, "graph") else G
    graph["_edge_version"] = int(graph.get("_edge_version", 0)) + 1
