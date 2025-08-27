from __future__ import annotations
from typing import Dict, Any, List, Tuple
import math
from collections import Counter

from .constants import DEFAULTS, ALIAS_SI, ALIAS_EPI
from .helpers import _get_attr, clamp01, register_callback

# -------------------------
# Canon: orden circular de glifos y ángulos
# -------------------------
GLYPHS_CANONICAL: List[str] = [
    "A’L",  # 0
    "E’N",  # 1
    "I’L",  # 2
    "U’M",  # 3
    "R’A",  # 4
    "VA’L", # 5
    "O’Z",  # 6
    "Z’HIR",# 7
    "NA’V", # 8
    "T’HOL",# 9
    "NU’L", #10
    "SH’A", #11
    "RE’MESH" #12
]

_SIGMA_ANGLES: Dict[str, float] = {g: (2.0*math.pi * i / len(GLYPHS_CANONICAL)) for i, g in enumerate(GLYPHS_CANONICAL)}

# -------------------------
# Config por defecto
# -------------------------
DEFAULTS.setdefault("SIGMA", {
    "enabled": True,
    "weight": "Si",      # "Si" | "EPI" | "1"
    "smooth": 0.0,        # EMA sobre el vector global (0=off)
    "history_key": "sigma_global",   # dónde guardar en G.graph['history']
    "per_node": False,    # si True, guarda trayectoria σ por nodo (más pesado)
})

# -------------------------
# Utilidades básicas
# -------------------------

def glyph_angle(g: str) -> float:
    return float(_SIGMA_ANGLES.get(g, 0.0))


def glyph_unit(g: str) -> complex:
    a = glyph_angle(g)
    return complex(math.cos(a), math.sin(a))


def _weight(G, n, mode: str) -> float:
    nd = G.nodes[n]
    if mode == "Si":
        return clamp01(_get_attr(nd, ALIAS_SI, 0.5))
    if mode == "EPI":
        return max(0.0, float(_get_attr(nd, ALIAS_EPI, 0.0)))
    return 1.0


def _last_glifo(nd: Dict[str, Any]) -> str | None:
    hist = nd.get("hist_glifos")
    if not hist:
        return None
    try:
        return list(hist)[-1]
    except Exception:
        return None


# -------------------------
# σ por nodo y σ global
# -------------------------

def sigma_vector_node(G, n, weight_mode: str | None = None) -> Dict[str, float] | None:
    nd = G.nodes[n]
    g = _last_glifo(nd)
    if g is None:
        return None
    w = _weight(G, n, weight_mode or G.graph.get("SIGMA", DEFAULTS["SIGMA"]).get("weight", "Si"))
    z = glyph_unit(g) * w
    x, y = z.real, z.imag
    mag = math.hypot(x, y)
    ang = math.atan2(y, x) if mag > 0 else glyph_angle(g)
    return {"x": float(x), "y": float(y), "mag": float(mag), "angle": float(ang), "glifo": g, "w": float(w)}


def sigma_vector_global(G, weight_mode: str | None = None) -> Dict[str, float]:
    """Vector global del plano del sentido σ.

    Mapea el último glifo de cada nodo a un vector unitario en S¹, ponderado
    por `Si` (o `EPI`/1), y promedia para obtener:
      - componentes (x, y), magnitud |σ| y ángulo arg(σ).

    Interpretación TNFR: |σ| mide cuán alineada está la red en su
    **recorrido glífico**; arg(σ) indica la **dirección funcional** dominante
    (p. ej., torno a I’L/RA para consolidación/distribución, O’Z/Z’HIR para cambio).
    """
    cfg = G.graph.get("SIGMA", DEFAULTS["SIGMA"])
    weight_mode = weight_mode or cfg.get("weight", "Si")
    acc = complex(0.0, 0.0)
    cnt = 0
    for n in G.nodes():
        v = sigma_vector_node(G, n, weight_mode)
        if v is None:
            continue
        acc += complex(v["x"], v["y"])
        cnt += 1
    if cnt == 0:
        return {"x": 1.0, "y": 0.0, "mag": 1.0, "angle": 0.0, "n": 0}
    x, y = acc.real / max(1, cnt), acc.imag / max(1, cnt)
    mag = math.hypot(x, y)
    ang = math.atan2(y, x)
    return {"x": float(x), "y": float(y), "mag": float(mag), "angle": float(ang), "n": cnt}


# -------------------------
# Historia / series
# -------------------------

def _ensure_history(G):
    if "history" not in G.graph:
        G.graph["history"] = {}
    return G.graph["history"]


def push_sigma_snapshot(G, t: float | None = None) -> None:
    cfg = G.graph.get("SIGMA", DEFAULTS["SIGMA"])
    if not cfg.get("enabled", True):
        return
    hist = _ensure_history(G)
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
    counts = Counter()
    for n in G.nodes():
        g = _last_glifo(G.nodes[n])
        if g:
            counts[g] += 1
    hist.setdefault("sigma_counts", []).append({"t": sv["t"], **counts})

    # Trayectoria por nodo (opcional)
    if cfg.get("per_node", False):
        per = hist.setdefault("sigma_per_node", {})
        for n in G.nodes():
            nd = G.nodes[n]
            g = _last_glifo(nd)
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
    cfg = G.graph.get("SIGMA", DEFAULTS["SIGMA"])
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
