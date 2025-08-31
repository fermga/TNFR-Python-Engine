"""
observers.py — TNFR canónica

Observadores y métricas auxiliares.
"""
from __future__ import annotations
from typing import Dict, Any
import math
import statistics as st

from .constants import ALIAS_DNFR, ALIAS_EPI, ALIAS_THETA, ALIAS_dEPI
from .helpers import _get_attr, list_mean, register_callback, angle_diff, ensure_history, count_glyphs
from .sense import glyph_unit, SIGMA_ANGLE_KEYS

# Clasificaciones funcionales de glifos
ESTABILIZADORES = ["IL", "RA", "UM", "SHA"]
DISRUPTIVOS = ["OZ", "ZHIR", "NAV", "THOL"]

# -------------------------
# Observador estándar Γ(R)
# -------------------------
def _std_log(G, kind: str, ctx: dict):
    """Guarda eventos compactos en history['events']."""
    h = ensure_history(G)
    h.setdefault("events", []).append((kind, dict(ctx)))

def std_before(G, ctx):
    _std_log(G, "before", ctx)

def std_after(G, ctx):
    _std_log(G, "after", ctx)

def std_on_remesh(G, ctx):
    _std_log(G, "remesh", ctx)

def attach_standard_observer(G):
    """Registra callbacks estándar: before_step, after_step, on_remesh."""
    register_callback(G, "before_step", std_before)
    register_callback(G, "after_step",  std_after)
    register_callback(G, "on_remesh",   std_on_remesh)
    G.graph.setdefault("_STD_OBSERVER", "attached")
    return G

def coherencia_global(G) -> float:
    """Proxy de C(t): alta cuando |ΔNFR| y |dEPI_dt| son pequeños."""
    dnfr = list_mean(abs(_get_attr(G.nodes[n], ALIAS_DNFR, 0.0)) for n in G.nodes())
    dEPI = list_mean(abs(_get_attr(G.nodes[n], ALIAS_dEPI, 0.0)) for n in G.nodes())
    return 1.0 / (1.0 + dnfr + dEPI)


def _phase_vectors(G) -> tuple[list, list]:
    """Devuelve listas de cosenos y senos de las fases nodales."""
    X = [math.cos(_get_attr(G.nodes[n], ALIAS_THETA, 0.0)) for n in G.nodes()]
    Y = [math.sin(_get_attr(G.nodes[n], ALIAS_THETA, 0.0)) for n in G.nodes()]
    return X, Y


def sincronía_fase(G) -> float:
    X, Y = _phase_vectors(G)
    if not X:
        return 1.0
    th = math.atan2(sum(Y) / len(Y), sum(X) / len(X))
    # varianza angular aproximada (0 = muy sincronizado)
    var = (
        st.pvariance(
            [
                angle_diff(_get_attr(G.nodes[n], ALIAS_THETA, 0.0), th)
                for n in G.nodes()
            ]
        )
        if len(X) > 1
        else 0.0
    )
    return 1.0 / (1.0 + var)

def orden_kuramoto(G) -> float:
    """R en [0,1], 1 = fases perfectamente alineadas."""
    X, Y = _phase_vectors(G)
    if not X:
        return 1.0
    R = ((sum(X)**2 + sum(Y)**2) ** 0.5) / max(1, len(X))
    return float(R)

def carga_glifica(G, window: int | None = None) -> dict:
    """Devuelve distribución de glifos aplicados en la red.
    - window: si se indica, cuenta solo los últimos `window` eventos por nodo; si no, usa el maxlen del deque.
    Retorna un dict con proporciones por glifo y agregados útiles.
    """
    total = count_glyphs(G, window=window)
    count = sum(total.values())
    if count == 0:
        return {"_count": 0}


    # Proporciones por glifo
    dist = {k: v / count for k, v in total.items()}

    # Agregados conceptuales (puedes ajustar categorías)
    # Glifos que consolidan la coherencia nodal: IL estabiliza el flujo (cap. 6),
    # RA propaga la resonancia (cap. 9), UM acopla nodos en fase (cap. 8)
    # y SHA ofrece silencio regenerativo (cap. 10). Véase manual TNFR,
    # sec. 18.19 "Análisis morfosintáctico" para la taxonomía funcional.
    # Glifos que perturban o reconfiguran la red: OZ introduce disonancia
    # evolutiva (cap. 7), ZHIR muta la estructura (cap. 14), NAV marca
    # el tránsito entre estados (cap. 15) y THOL autoorganiza un nuevo
    # orden (cap. 13). Véase manual TNFR, sec. 18.19 para esta clasificación.


    dist["_estabilizadores"] = sum(dist.get(k, 0.0) for k in ESTABILIZADORES)
    dist["_disruptivos"] = sum(dist.get(k, 0.0) for k in DISRUPTIVOS)
    dist["_count"] = count
    return dist

def sigma_vector(G, window: int | None = None) -> dict:
    """Vector de sentido Σ⃗ a partir de la distribución glífica reciente.
    Devuelve dict con x, y, mag (0..1) y angle (rad)."""
    # Distribución glífica (proporciones)
    dist = carga_glifica(G, window=window)
    if not dist or dist.get("_count", 0) == 0:
        return {"x": 0.0, "y": 0.0, "mag": 0.0, "angle": 0.0}

    # Usa el conjunto fijo de glifos en el plano de sentido
    total = sum(dist.get(k, 0.0) for k in SIGMA_ANGLE_KEYS)
    if total <= 0:
        return {"x": 0.0, "y": 0.0, "mag": 0.0, "angle": 0.0}

    x = 0.0
    y = 0.0
    for k in SIGMA_ANGLE_KEYS:
        p = dist.get(k, 0.0) / total
        z = glyph_unit(k)
        x += p * z.real
        y += p * z.imag

    mag = math.hypot(x, y)
    ang = math.atan2(y, x)
    return {"x": float(x), "y": float(y), "mag": float(mag), "angle": float(ang)}

def wbar(G, window: int | None = None) -> float:
    """Devuelve W̄ = media de C(t) en una ventana reciente."""
    hist = G.graph.get("history", {})
    cs = hist.get("C_steps", [])
    if not cs:
        # fallback: coherencia instantánea
        return coherencia_global(G)
    if window is None:
        window = int(G.graph.get("WBAR_WINDOW", 25))
    w = min(len(cs), max(1, int(window)))
    return float(sum(cs[-w:]) / w)
