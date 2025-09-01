"""
helpers.py — TNFR canónica

Utilidades transversales + cálculo de Índice de sentido (Si).
"""
from __future__ import annotations
from typing import Iterable, Dict, Any, Callable, TypeVar
import math
from collections import deque, Counter
from statistics import fmean, StatisticsError
import json
from pathlib import Path

try:  # pragma: no cover - dependencia opcional
    import yaml  # type: ignore
except ImportError:  # pragma: no cover
    yaml = None

try:
    import networkx as nx  # solo para tipos
except ImportError:  # pragma: no cover
    nx = None  # type: ignore

from .constants import DEFAULTS, ALIAS_VF, ALIAS_THETA, ALIAS_DNFR, ALIAS_SI, ALIAS_EPI_KIND

T = TypeVar("T")

__all__ = [
    "read_structured_file",
    "clamp",
    "clamp_abs",
    "clamp01",
    "list_mean",
    "angle_diff",
    "phase_distance",
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
    "push_glifo",
    "reciente_glifo",
    "ensure_history",
    "last_glifo",
    "count_glyphs",
    "register_callback",
    "invoke_callbacks",
    "compute_Si",
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
        raise RuntimeError("pyyaml no está instalado")
    return yaml.safe_load(text)


PARSERS: Dict[str, Callable[[str], Any]] = {
    ".json": _parse_json,
    ".yaml": _parse_yaml,
    ".yml": _parse_yaml,
}


def read_structured_file(path: Path) -> Any:
    """Lee un archivo JSON o YAML y devuelve los datos parseados."""
    suffix = path.suffix.lower()
    if suffix not in PARSERS:
        raise ValueError(f"Extensión de archivo no soportada: {path.suffix}")
    parser = PARSERS[suffix]
    try:
        text = path.read_text(encoding="utf-8")
        return parser(text)
    except (FileNotFoundError, PermissionError) as e:
        raise ValueError(f"No se pudo abrir {path}: {e}") from e
    except Exception as e:
        formato = "YAML" if parser is _parse_yaml else "JSON"
        raise ValueError(f"Error al parsear {formato} en {path}: {e}") from e

# -------------------------
# Utilidades numéricas
# -------------------------

def clamp(x: float, a: float, b: float) -> float:
    """Constriñe ``x`` al intervalo cerrado [a, b]."""
    return a if x < a else b if x > b else x


def clamp_abs(x: float, m: float) -> float:
    """Limita ``x`` al rango simétrico [-m, m] usando ``abs(m)``."""
    m = abs(m)
    return clamp(x, -m, m)


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
    pi = math.pi
    a = (a + pi) % (2 * pi) - pi
    return a


def angle_diff(a: float, b: float) -> float:
    """Diferencia mínima entre ``a`` y ``b`` en (-π, π]."""
    return _wrap_angle(a - b)


def phase_distance(a: float, b: float) -> float:
    """Distancia de fase normalizada en [0,1]. 0 = misma fase, 1 = opuesta."""
    return abs(angle_diff(a, b)) / math.pi


def normalize_weights(dict_like: Dict[str, Any], keys: Iterable[str], default: float = 0.0) -> Dict[str, float]:
    """Devuelve ``dict`` de ``keys`` normalizadas a sumatorio 1.

    Cada clave en ``keys`` se extrae de ``dict_like`` convirtiéndola a ``float``.
    Si la suma de los valores obtenidos es <= 0, se asignan proporciones
    uniformes entre todas las claves.
    """
    keys = list(keys)
    weights = {k: float(dict_like.get(k, default)) for k in keys}
    total = sum(weights.values())
    n = len(keys)
    if total <= 0:
        if n == 0:
            return {}
        uniform = 1.0 / n
        return {k: uniform for k in keys}
    return {k: v / total for k, v in weights.items()}


# -------------------------
# Acceso a atributos con alias
# -------------------------

_sentinel = object()


def _ensure_tuple(aliases: Iterable[str]) -> tuple[str, ...]:
    """Garantiza que ``aliases`` sea una tupla."""
    if isinstance(aliases, tuple):
        return aliases
    if isinstance(aliases, str):
        return (aliases,)
    return tuple(aliases)


def alias_get(
    d: Dict[str, Any],
    aliases: Iterable[str],
    conv: Callable[[Any], T],
    *,
    default: Any = _sentinel,
) -> T | None:
    """Busca en ``d`` la primera clave de ``aliases`` y retorna el valor convertido.

    Si ninguna de las claves está presente o la conversión falla, devuelve
    ``default`` convertido (o ``None`` si ``default`` es ``None``).
    """
    aliases = _ensure_tuple(aliases)
    if not aliases:
        raise ValueError("'aliases' must contain at least one key")
    for key in aliases:
        if key in d:
            try:
                return conv(d[key])
            except (ValueError, TypeError):
                continue
    if default is not _sentinel:
        if default is None:
            return None
        return conv(default)
    return None


def alias_set(
    d: Dict[str, Any],
    aliases: Iterable[str],
    conv: Callable[[Any], T],
    value: Any,
) -> T:
    """Asigna ``value`` convertido a la primera clave disponible de ``aliases``."""
    aliases = _ensure_tuple(aliases)
    if not aliases:
        raise ValueError("'aliases' must contain at least one key")
    for key in aliases:
        if key in d:
            d[key] = conv(value)
            return d[key]
    key = aliases[0]
    d[key] = conv(value)
    return d[key]


def get_attr(d: Dict[str, Any], aliases: Iterable[str], default: float = 0.0) -> float:
    return alias_get(d, aliases, float, default=default)


def set_attr(d, aliases, value: float) -> None:
    alias_set(d, aliases, float, value)


def get_attr_str(d: Dict[str, Any], aliases: Iterable[str], default: str = "") -> str:
    return alias_get(d, aliases, str, default=default)


def set_attr_str(d, aliases, value: str) -> None:
    alias_set(d, aliases, str, value)

# Retrocompatibilidad con nombres anteriores
_get_attr = get_attr
_set_attr = set_attr
_get_attr_str = get_attr_str
_set_attr_str = set_attr_str


# -------------------------
# Máximos globales con caché
# -------------------------

def _recompute_abs_max(G, aliases):
    """Recalcula y retorna ``(max_val, node)`` para ``aliases``."""
    node = max(
        G.nodes(),
        key=lambda m: abs(get_attr(G.nodes[m], aliases, 0.0)),
        default=None,
    )
    max_val = abs(get_attr(G.nodes[node], aliases, 0.0)) if node is not None else 0.0
    return max_val, node


def _update_cached_abs_max(G, aliases, n, value, *, key: str) -> None:
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


def set_attr_with_max(G, n, aliases, value: float, *, cache: str) -> None:
    """Asigna ``value`` al atributo indicado y actualiza el máximo global."""
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
# Estadísticos vecinales
# -------------------------

def media_vecinal(G, n, aliases: Iterable[str], default: float = 0.0) -> float:
    """Media del atributo indicado por ``aliases`` en los vecinos de ``n``."""
    vals = (get_attr(G.nodes[v], aliases, default) for v in G.neighbors(n))
    return list_mean(vals, default)


def fase_media(obj, n=None) -> float:
    """Promedio circular de las fases vecinales.

    Acepta un :class:`NodoProtocol` o un par ``(G, n)`` de ``networkx``. En el
    segundo caso se envuelve en :class:`NodoNX` para reutilizar la misma lógica.
    """

    if n is not None:
        from .node import NodoNX  # importación local para evitar ciclo
        node = NodoNX(obj, n)
    else:
        node = obj  # se asume NodoProtocol

    x = y = 0.0
    count = 0
    for v in node.neighbors():
        th = getattr(v, "theta", 0.0)
        x += math.cos(th)
        y += math.sin(th)
        count += 1
    if count == 0:
        return getattr(node, "theta", 0.0)
    return math.atan2(y / count, x / count)


# -------------------------
# Historial de glifos por nodo
# -------------------------

def push_glifo(nd: Dict[str, Any], glifo: str, window: int) -> None:
    """Añade ``glifo`` al historial del nodo con tamaño máximo ``window``."""
    hist = nd.get("hist_glifos")
    if hist is None or hist.maxlen != window:
        hist = deque(hist or [], maxlen=window)
        nd["hist_glifos"] = hist
    hist.append(str(glifo))


def reciente_glifo(nd: Dict[str, Any], glifo: str, ventana: int) -> bool:
    """Indica si ``glifo`` apareció en las últimas ``ventana`` emisiones"""
    hist = nd.get("hist_glifos")
    gl = str(glifo)
    if ventana < 0:
        raise ValueError("ventana debe ser >= 0")
    if hist:
        limite = min(len(hist), ventana)
        for i in range(1, limite + 1):
            if hist[-i] == gl:
                return True
    # fallback al glifo dominante actual
    return get_attr_str(nd, ALIAS_EPI_KIND, "") == gl

# -------------------------
# Utilidades de historial global
# -------------------------

def ensure_history(G) -> Dict[str, Any]:
    """Garantiza G.graph['history'] y la devuelve."""
    return G.graph.setdefault("history", {})


def last_glifo(nd: Dict[str, Any]) -> str | None:
    """Retorna el glifo más reciente del nodo o ``None``."""
    kind = get_attr_str(nd, ALIAS_EPI_KIND, "")
    if kind:
        return kind
    hist = nd.get("hist_glifos")
    if not hist:
        return None
    try:
        return hist[-1]
    except IndexError:
        return None


def _count_last_glifos(G) -> Counter:
    """Cuenta solo el último glifo de cada nodo."""
    counts: Counter[str] = Counter()
    for _, nd in G.nodes(data=True):
        g = last_glifo(nd)
        if g:
            counts[g] += 1
    return counts


def count_glyphs(G, window: int | None = None) -> Counter:
    """Cuenta glifos recientes en la red.

    Si ``window`` es ``1`` cuenta solo el último glifo de cada nodo. Con un
    valor mayor o ``None`` se usa el historial ``hist_glifos`` limitado a los
    últimos ``window`` elementos por nodo.
    """
    if window == 1:
        return _count_last_glifos(G)
    counts: Counter[str] = Counter()
    for _, nd in G.nodes(data=True):
        hist = nd.get("hist_glifos")
        if not hist:
            continue
        if window is not None and window > 0:
            limite = min(len(hist), int(window))
            seq = (hist[-i] for i in range(limite, 0, -1))
        else:
            seq = hist
        counts.update(seq)
    return counts

# -------------------------
# Callbacks Γ(R)
# -------------------------

def _ensure_callbacks(G):
    """Garantiza la estructura de callbacks en G.graph."""
    cbs = G.graph.setdefault("callbacks", {
        "before_step": [],
        "after_step": [],
        "on_remesh": [],
    })
    # normaliza claves por si vienen incompletas
    for k in ("before_step", "after_step", "on_remesh"):
        cbs.setdefault(k, [])
    return cbs

def register_callback(
    G,
    event: str,
    func=None,
    *,
    name: str | None = None,
):
    """Registra ``func`` como callback del ``event`` indicado.

    Permite tanto la forma posicional ``register_callback(G, "after_step", fn)``
    como la forma con palabras clave ``register_callback(G, event="after_step", func=fn)``.
    El parámetro ``name`` ahora se almacena junto con la función para facilitar
    su identificación.
    """
    if event not in ("before_step", "after_step", "on_remesh"):
        raise ValueError(f"Evento desconocido: {event}")
    if func is None:
        raise TypeError("func es obligatorio")
    cbs = _ensure_callbacks(G)
    cb_name = name or getattr(func, "__name__", None)
    cbs[event].append((cb_name, func))
    return func

def invoke_callbacks(G, event: str, ctx: dict | None = None):
    """Invoca todos los callbacks registrados para ``event`` con el contexto ``ctx``.

    Los callbacks se almacenan como tuplas ``(name, func)`` y se invocan en orden
    de registro. Se admite el formato antiguo de solo función para compatibilidad.
    """
    cbs = _ensure_callbacks(G).get(event, [])
    strict = bool(G.graph.get("CALLBACKS_STRICT", DEFAULTS["CALLBACKS_STRICT"]))
    ctx = ctx or {}
    for cb in list(cbs):
        if isinstance(cb, tuple):
            name, fn = (cb + (None, None))[:2]
        else:  # retrocompatibilidad
            name, fn = getattr(cb, "__name__", None), cb
        try:
            fn(G, ctx)
        except (KeyError, ValueError, TypeError) as e:
            if strict:
                raise
            G.graph.setdefault("_callback_errors", []).append({
                "event": event,
                "step": ctx.get("step"),
                "error": repr(e),
                "fn": repr(fn),
                "name": name,
            })

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
        deg = G.degree(n)
        if deg:
            neigh = G.neighbors(n)
            sum_cos = 0.0
            sum_sin = 0.0
            for v in neigh:
                sum_cos += cos_th[v]
                sum_sin += sin_th[v]
            mean_cos = sum_cos / deg
            mean_sin = sum_sin / deg
            th_bar = math.atan2(mean_sin, mean_cos)
        else:
            th_bar = th_i
        disp_fase = phase_distance(th_i, th_bar)  # [0,1]

        dnfr = get_attr(nd, ALIAS_DNFR, 0.0)
        dnfr_norm = clamp01(abs(dnfr) / dnfrmax)

        Si = alpha*vf_norm + beta*(1.0 - disp_fase) + gamma*(1.0 - dnfr_norm)
        Si = clamp01(Si)
        out[n] = Si
        if inplace:
            set_attr(nd, ALIAS_SI, Si)
    return out
