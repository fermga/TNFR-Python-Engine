from __future__ import annotations
from typing import Dict, List
import math
from collections import Counter

from .constants import ALIAS_SI, ALIAS_EPI, SIGMA
from .helpers import (
    _get_attr,
    clamp01,
    register_callback,
    ensure_history,
    last_glifo,
    count_glyphs,
)
from .types import Glyph
from .constants_glifos import ANGLE_MAP, ESTABILIZADORES, DISRUPTIVOS

# -------------------------
# Canon: orden circular de glifos y ángulos
# -------------------------
GLYPHS_CANONICAL: List[str] = [
    Glyph.AL.value,   # 0
    Glyph.EN.value,   # 1
    Glyph.IL.value,   # 2
    Glyph.UM.value,   # 3
    Glyph.RA.value,   # 4
    Glyph.VAL.value,  # 5
    Glyph.OZ.value,   # 6
    Glyph.ZHIR.value, # 7
    Glyph.NAV.value,  # 8
    Glyph.THOL.value, # 9
    Glyph.NUL.value,  #10
    Glyph.SHA.value,  #11
    Glyph.REMESH.value #12
]

# Glifos relevantes para el plano Σ de observadores de sentido
SIGMA_ANGLE_KEYS: tuple[str, ...] = tuple(ESTABILIZADORES + DISRUPTIVOS)

GLYPH_UNITS: Dict[str, complex] = {
    g: complex(math.cos(a), math.sin(a)) for g, a in ANGLE_MAP.items()
}

# -------------------------
# Utilidades básicas
# -------------------------

def glyph_angle(g: str) -> float:
    return float(ANGLE_MAP.get(g, 0.0))


def glyph_unit(g: str) -> complex:
    return GLYPH_UNITS.get(g, 1+0j)


def _weight(G, n, mode: str) -> float:
    nd = G.nodes[n]
    if mode == "Si":
        return clamp01(_get_attr(nd, ALIAS_SI, 0.5))
    if mode == "EPI":
        return max(0.0, float(_get_attr(nd, ALIAS_EPI, 0.0)))
    return 1.0


def _sigma_cfg(G):
    return G.graph.get("SIGMA", SIGMA)


    
# -------------------------
# σ por nodo y σ global
# -------------------------

def sigma_vector_node(G, n, weight_mode: str | None = None) -> Dict[str, float] | None:
    nd = G.nodes[n]
    g = last_glifo(nd)
    if g is None:
        return None
    cfg = _sigma_cfg(G)
    w = _weight(G, n, weight_mode or cfg.get("weight", "Si"))
    z = glyph_unit(g) * w
    x, y = z.real, z.imag
    mag = math.hypot(x, y)
    ang = math.atan2(y, x) if mag > 0 else glyph_angle(g)
    return {"x": float(x), "y": float(y), "mag": float(mag), "angle": float(ang), "glifo": g, "w": float(w)}


def sigma_vector_global(G, weight_mode: str | None = None) -> Dict[str, float]:
    """Vector global del plano del sentido σ.

    Acepta un grafo de NetworkX o una distribución precontada de glifos. En el
    primer caso, mapea el último glifo de cada nodo a un vector unitario en S¹,
    ponderado por `Si` (o `EPI`/1), y promedia para obtener componentes (x, y),
    magnitud |σ| y ángulo arg(σ). En el segundo, utiliza directamente la
    distribución proporcionada.

    Si no hay datos suficientes retorna el vector nulo.
    """
    if not hasattr(G, "nodes"):
        dist = G  # distribución ya normalizada o en conteos
        total = sum(float(dist.get(k, 0.0)) for k in SIGMA_ANGLE_KEYS)
        if total <= 0:
            return {"x": 0.0, "y": 0.0, "mag": 0.0, "angle": 0.0}
        x = y = 0.0
        for k in SIGMA_ANGLE_KEYS:
            p = float(dist.get(k, 0.0)) / total
            z = glyph_unit(k)
            x += p * z.real
            y += p * z.imag
        mag = math.hypot(x, y)
        ang = math.atan2(y, x)
        return {"x": float(x), "y": float(y), "mag": float(mag), "angle": float(ang)}

    cfg = _sigma_cfg(G)
    weight_mode = weight_mode or cfg.get("weight", "Si")
    acc = complex(0.0, 0.0)
    cnt = 0
    for n in G.nodes():
        nd = G.nodes[n]
        g = last_glifo(nd)
        if not g:
            continue
        w = _weight(G, n, weight_mode)
        acc += glyph_unit(g) * w
        cnt += 1
    if cnt == 0:
        return {"x": 0.0, "y": 0.0, "mag": 0.0, "angle": 0.0, "n": 0}
    x, y = acc.real / cnt, acc.imag / cnt
    mag = math.hypot(x, y)
    ang = math.atan2(y, x)
    return {"x": float(x), "y": float(y), "mag": float(mag), "angle": float(ang), "n": cnt}


# -------------------------
# Historia / series
# -------------------------

def push_sigma_snapshot(G, t: float | None = None) -> None:
    cfg = _sigma_cfg(G)
    if not cfg.get("enabled", True):
        return
    hist = ensure_history(G)
    key = cfg.get("history_key", "sigma_global")

    # Global
    sv = sigma_vector_global(G, cfg.get("weight", "Si"))

    # Suavizado exponencial (EMA) opcional
    alpha = float(cfg.get("smooth", 0.0))
    if alpha > 0 and hist.get(key):
        prev = hist[key][-1]
        x = (1-alpha)*prev["x"] + alpha*sv["x"]
        y = (1-alpha)*prev["y"] + alpha*sv["y"]
        mag = math.hypot(x, y)
        ang = math.atan2(y, x)
        sv = {"x": x, "y": y, "mag": mag, "angle": ang, "n": sv.get("n", 0)}

    sv["t"] = float(G.graph.get("_t", 0.0) if t is None else t)

    hist.setdefault(key, []).append(sv)

    # Conteo de glifos por paso (útil para rosa glífica)
    counts = count_glyphs(G, window=1)
    hist.setdefault("sigma_counts", []).append({"t": sv["t"], **counts})

    # Trayectoria por nodo (opcional)
    if cfg.get("per_node", False):
        per = hist.setdefault("sigma_per_node", {})
        for n, nd in G.nodes(data=True):
            g = last_glifo(nd)
            if not g:
                continue
            a = glyph_angle(g)
            d = per.setdefault(n, [])
            d.append({"t": sv["t"], "g": g, "angle": a})


# -------------------------
# Registro como callback automático (after_step)
# -------------------------

def register_sigma_callback(G) -> None:
    register_callback(G, when="after_step", func=push_sigma_snapshot, name="sigma_snapshot")


# -------------------------
# Series de utilidad
# -------------------------

def sigma_series(G, key: str | None = None) -> Dict[str, List[float]]:
    cfg = _sigma_cfg(G)
    key = key or cfg.get("history_key", "sigma_global")
    hist = G.graph.get("history", {})
    xs = hist.get(key, [])
    if not xs:
        return {"t": [], "angle": [], "mag": []}
    return {
        "t": [float(x.get("t", i)) for i, x in enumerate(xs)],
        "angle": [float(x["angle"]) for x in xs],
        "mag": [float(x["mag"]) for x in xs],
    }


def sigma_rose(G, steps: int | None = None) -> Dict[str, int]:
    """Histograma de glifos en los últimos `steps` pasos (o todos)."""
    hist = G.graph.get("history", {})
    counts = hist.get("sigma_counts", [])
    if not counts:
        return {g: 0 for g in GLYPHS_CANONICAL}
    if steps is None or steps >= len(counts):
        agg = Counter()
        for row in counts:
            agg.update({k: v for k, v in row.items() if k != "t"})
        out = {g: int(agg.get(g, 0)) for g in GLYPHS_CANONICAL}
        return out
    agg = Counter()
    for row in counts[-int(steps):]:
        agg.update({k: v for k, v in row.items() if k != "t"})
    return {g: int(agg.get(g, 0)) for g in GLYPHS_CANONICAL}
