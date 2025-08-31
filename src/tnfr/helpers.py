"""
helpers.py — TNFR canónica

Utilidades transversales + cálculo de Índice de sentido (Si).
"""
from __future__ import annotations
from typing import Iterable, Dict, Any, TYPE_CHECKING
import math
from collections import deque, Counter
from itertools import islice
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

from .constants import DEFAULTS, ALIAS_VF, ALIAS_THETA, ALIAS_DNFR, ALIAS_EPI, ALIAS_SI, ALIAS_EPI_KIND

if TYPE_CHECKING:  # pragma: no cover - sólo para tipos
    from .node import NodoProtocol

# -------------------------
# Entrada/salida estructurada
# -------------------------


def read_structured_file(path: Path) -> Any:
    """Lee un archivo JSON o YAML y devuelve los datos parseados."""
    suffix = path.suffix.lower()
    if suffix not in {".json", ".yaml", ".yml"}:
        raise ValueError(f"Extensión de archivo no soportada: {path.suffix}")
    with path.open("r", encoding="utf-8") as f:
        if suffix in {".yaml", ".yml"}:
            if not yaml:  # pragma: no cover - dependencia opcional
                raise RuntimeError("pyyaml no está instalado")
            return yaml.safe_load(f)
        return json.load(f)

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
    return abs(_wrap_angle(a - b)) / math.pi


# -------------------------
# Acceso a atributos con alias
# -------------------------

_sentinel = object()


def alias_lookup(
    d: Dict[str, Any],
    aliases: Iterable[str],
    conv,
    *,
    default=_sentinel,
    value=_sentinel,
):
    """Busca en ``d`` la primera clave de ``aliases``.

    Si ``value`` se proporciona, actúa como asignación usando ``conv`` para
    convertir el valor. Si no, intenta obtener y convertir el valor existente,
    devolviendo ``default`` convertido si ninguna alias coincide o la conversión
    falla.
    """
    alist = tuple(aliases)

    for key in alist:
        if key in d:
            if value is not _sentinel:
                d[key] = conv(value)
                return d[key]
            try:
                return conv(d[key])
            except (ValueError, TypeError):
                continue

    if value is not _sentinel:
        key = alist[0]
        d[key] = conv(value)
        return d[key]

    if default is not _sentinel:
        if default is None:
            return None
        return conv(default)
    return None


def _get_attr(d: Dict[str, Any], aliases: Iterable[str], default: float = 0.0) -> float:
    return alias_lookup(d, aliases, float, default=default)


def _set_attr(d, aliases, value: float) -> None:
    alias_lookup(d, aliases, float, value=value)


def _get_attr_str(d: Dict[str, Any], aliases: Iterable[str], default: str = "") -> str:
    return alias_lookup(d, aliases, str, default=default)


def _set_attr_str(d, aliases, value: str) -> None:
    alias_lookup(d, aliases, str, value=value)


# -------------------------
# Máximos globales con caché
# -------------------------

def _recompute_abs_max(G, aliases):
    """Recalcula y retorna ``(max_val, node)`` para ``aliases``."""
    max_val = 0.0
    max_node = None
    for m in G.nodes():
        v = abs(_get_attr(G.nodes[m], aliases, 0.0))
        if v >= max_val:
            max_val = v
            max_node = m
    return max_val, max_node


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


def set_vf(G, n, value: float) -> None:
    """Asigna ``νf`` y actualiza el máximo global."""
    _set_attr(G.nodes[n], ALIAS_VF, float(value))
    _update_cached_abs_max(G, ALIAS_VF, n, float(value), key="_vfmax")


def set_dnfr(G, n, value: float) -> None:
    """Asigna ``ΔNFR`` y actualiza el máximo global."""
    _set_attr(G.nodes[n], ALIAS_DNFR, float(value))
    _update_cached_abs_max(G, ALIAS_DNFR, n, float(value), key="_dnfrmax")

# -------------------------
# Estadísticos vecinales
# -------------------------

def media_vecinal(G, n, aliases: Iterable[str], default: float = 0.0) -> float:
    """Media del atributo indicado por ``aliases`` en los vecinos de ``n``."""
    vals = (_get_attr(G.nodes[v], aliases, default) for v in G.neighbors(n))
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
    if hist and any(g == gl for g in islice(reversed(hist), ventana)):
        return True
    # fallback al glifo dominante actual
    return _get_attr_str(nd, ALIAS_EPI_KIND, "") == gl

# -------------------------
# Utilidades de historial global
# -------------------------

def ensure_history(G) -> Dict[str, Any]:
    """Garantiza G.graph['history'] y la devuelve."""
    return G.graph.setdefault("history", {})


def last_glifo(nd: Dict[str, Any]) -> str | None:
    """Retorna el glifo más reciente del nodo o ``None``."""
    kind = _get_attr_str(nd, ALIAS_EPI_KIND, "")
    if kind:
        return kind
    hist = nd.get("hist_glifos")
    if not hist:
        return None
    try:
        return hist[-1]
    except IndexError:
        return None


def count_glyphs(G, window: int | None = None) -> Counter:
    """Cuenta glifos recientes en la red.

    Si ``window`` es ``1`` cuenta solo el último glifo de cada nodo. Con un
    valor mayor o ``None`` se usa el historial ``hist_glifos`` limitado a los
    últimos ``window`` elementos por nodo.
    """
    counts: Counter[str] = Counter()
    for _, nd in G.nodes(data=True):
        if window == 1:
            g = last_glifo(nd)
            if g:
                counts[g] += 1
            continue
        hist = nd.get("hist_glifos")
        if not hist:
            continue
        if window is not None and window > 0:
            start = max(len(hist) - int(window), 0)
            seq = islice(hist, start, None)
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
    event: str | None = None,
    func=None,
    *,
    when: str | None = None,
    name: str | None = None,
):
    """Registra ``func`` como callback del ``event`` indicado.

    Permite tanto la forma posicional ``register_callback(G, "after_step", fn)``
    como la forma con palabras clave ``register_callback(G, when="after_step", func=fn)``.
    El parámetro ``name`` ahora se almacena junto con la función para facilitar
    su identificación.
    """
    event = event or when
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
    alpha = float(G.graph.get("SI_WEIGHTS", DEFAULTS["SI_WEIGHTS"]).get("alpha", 0.34))
    beta = float(G.graph.get("SI_WEIGHTS", DEFAULTS["SI_WEIGHTS"]).get("beta", 0.33))
    gamma = float(G.graph.get("SI_WEIGHTS", DEFAULTS["SI_WEIGHTS"]).get("gamma", 0.33))
    s = alpha + beta + gamma
    if s <= 0:
        alpha = beta = gamma = 1/3
    else:
        alpha, beta, gamma = alpha/s, beta/s, gamma/s
    G.graph["_Si_weights"] = {"alpha": alpha, "beta": beta, "gamma": gamma}
    G.graph["_Si_sensitivity"] = {"dSi_dvf_norm": alpha, "dSi_ddisp_fase": -beta, "dSi_ddnfr_norm": -gamma}

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

    out: Dict[Any, float] = {}
    for n, nd in G.nodes(data=True):
        vf = _get_attr(nd, ALIAS_VF, 0.0)
        vf_norm = clamp01(abs(vf) / vfmax)

        # dispersión de fase local
        th_i = _get_attr(nd, ALIAS_THETA, 0.0)
        th_bar = fase_media(G, n)
        disp_fase = phase_distance(th_i, th_bar)  # [0,1]

        dnfr = _get_attr(nd, ALIAS_DNFR, 0.0)
        dnfr_norm = clamp01(abs(dnfr) / dnfrmax)

        Si = alpha*vf_norm + beta*(1.0 - disp_fase) + gamma*(1.0 - dnfr_norm)
        Si = clamp01(Si)
        out[n] = Si
        if inplace:
            _set_attr(nd, ALIAS_SI, Si)
    return out
