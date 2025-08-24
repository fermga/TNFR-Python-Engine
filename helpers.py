"""
helpers.py — TNFR canónica

Utilidades transversales + cálculo de Índice de sentido (Si).
"""
from __future__ import annotations
from typing import Iterable, Dict, Any, Tuple, List
import math
from collections import deque

try:
    import networkx as nx  # solo para tipos
except Exception:  # pragma: no cover
    nx = None  # type: ignore

from constants import DEFAULTS, ALIAS_VF, ALIAS_THETA, ALIAS_DNFR, ALIAS_EPI, ALIAS_SI

# -------------------------
# Utilidades numéricas
# -------------------------

def clamp(x: float, a: float, b: float) -> float:
    return a if x < a else b if x > b else x


def clamp_abs(x: float, m: float) -> float:
    m = abs(m)
    return clamp(x, -m, m)


def clamp01(x: float) -> float:
    return clamp(x, 0.0, 1.0)


def list_mean(xs: Iterable[float], default: float = 0.0) -> float:
    xs = list(xs)
    return sum(xs) / len(xs) if xs else default


def _wrap_angle(a: float) -> float:
    """Envuelve ángulo a (-π, π]."""
    pi = math.pi
    a = (a + pi) % (2 * pi) - pi
    return a


def phase_distance(a: float, b: float) -> float:
    """Distancia de fase normalizada en [0,1]. 0 = misma fase, 1 = opuesta."""
    return abs(_wrap_angle(a - b)) / math.pi


# -------------------------
# Acceso a atributos con alias
# -------------------------

def _get_attr(d: Dict[str, Any], aliases: Iterable[str], default: float = 0.0) -> float:
    for k in aliases:
        if k in d:
            try:
                return float(d[k])
            except Exception:
                continue
    return float(default)

def _set_attr(d, aliases, value: float) -> None:
    for k in aliases:
        if k in d:
            d[k] = float(value)
            return
    d[next(iter(aliases))] = float(value)

# -------------------------
# Estadísticos vecinales
# -------------------------

def media_vecinal(G, n, aliases: Iterable[str], default: float = 0.0) -> float:
    vals: List[float] = []
    for v in G.neighbors(n):
        vals.append(_get_attr(G.nodes[v], aliases, default))
    return list_mean(vals, default)


def fase_media(G, n) -> float:
    """Promedio circular de las fases de los vecinos."""
    import math
    x = 0.0
    y = 0.0
    count = 0
    for v in G.neighbors(n):
        th = _get_attr(G.nodes[v], ALIAS_THETA, 0.0)
        x += math.cos(th)
        y += math.sin(th)
        count += 1
    if count == 0:
        return _get_attr(G.nodes[n], ALIAS_THETA, 0.0)
    return math.atan2(y / max(1, count), x / max(1, count))


# -------------------------
# Historial de glifos por nodo
# -------------------------

def push_glifo(nd: Dict[str, Any], glifo: str, window: int) -> None:
    hist = nd.setdefault("hist_glifos", deque(maxlen=window))
    hist.append(str(glifo))


def reciente_glifo(nd: Dict[str, Any], glifo: str, ventana: int) -> bool:
    hist = nd.get("hist_glifos")
    if not hist:
        return False
    last = list(hist)[-ventana:]
    return str(glifo) in last

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

def register_callback(G, event: str, func):
    """Registra un callback en G.graph['callbacks'][event]. Firma: func(G, ctx) -> None"""
    if event not in ("before_step", "after_step", "on_remesh"):
        raise ValueError(f"Evento desconocido: {event}")
    cbs = _ensure_callbacks(G)
    cbs[event].append(func)
    return func

def invoke_callbacks(G, event: str, ctx: dict | None = None):
    """Invoca todos los callbacks registrados para `event` con el contexto `ctx`."""
    cbs = _ensure_callbacks(G).get(event, [])
    strict = bool(G.graph.get("CALLBACKS_STRICT", DEFAULTS["CALLBACKS_STRICT"]))
    ctx = ctx or {}
    for fn in list(cbs):
        try:
            fn(G, ctx)
        except Exception as e:
            if strict:
                raise
            G.graph.setdefault("_callback_errors", []).append({
                "event": event, "step": ctx.get("step"), "error": repr(e), "fn": repr(fn)
            })

# -------------------------
# Índice de sentido (Si)
# -------------------------

def compute_Si(G, *, inplace: bool = True) -> Dict[Any, float]:
    """Calcula Si por nodo y lo escribe en G.nodes[n]["Si"].

    Si = α·νf_norm + β·(1 - disp_fase_local) + γ·(1 - |ΔNFR|/max|ΔNFR|)
    """
    alpha = float(G.graph.get("SI_WEIGHTS", DEFAULTS["SI_WEIGHTS"]).get("alpha", 0.34))
    beta = float(G.graph.get("SI_WEIGHTS", DEFAULTS["SI_WEIGHTS"]).get("beta", 0.33))
    gamma = float(G.graph.get("SI_WEIGHTS", DEFAULTS["SI_WEIGHTS"]).get("gamma", 0.33))
    s = alpha + beta + gamma
    if s <= 0:
        alpha = beta = gamma = 1/3
    else:
        alpha, beta, gamma = alpha/s, beta/s, gamma/s

    # Normalización de νf en red
    vfs = [abs(_get_attr(G.nodes[n], ALIAS_VF, 0.0)) for n in G.nodes()]
    vfmax = max(vfs) if vfs else 1.0
    # Normalización de ΔNFR
    dnfrs = [abs(_get_attr(G.nodes[n], ALIAS_DNFR, 0.0)) for n in G.nodes()]
    dnfrmax = max(dnfrs) if dnfrs else 1.0

    out: Dict[Any, float] = {}
    for n in G.nodes():
        nd = G.nodes[n]
        vf = _get_attr(nd, ALIAS_VF, 0.0)
        vf_norm = 0.0 if vfmax == 0 else clamp01(abs(vf)/vfmax)

        # dispersión de fase local
        th_i = _get_attr(nd, ALIAS_THETA, 0.0)
        th_bar = fase_media(G, n)
        disp_fase = phase_distance(th_i, th_bar)  # [0,1]

        dnfr = _get_attr(nd, ALIAS_DNFR, 0.0)
        dnfr_norm = 0.0 if dnfrmax == 0 else clamp01(abs(dnfr)/dnfrmax)

        Si = alpha*vf_norm + beta*(1.0 - disp_fase) + gamma*(1.0 - dnfr_norm)
        Si = clamp01(Si)
        out[n] = Si
        if inplace:
            _set_attr(nd, ALIAS_SI, Si)
    return out
