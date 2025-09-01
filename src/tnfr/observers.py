"""
observers.py — TNFR canónica

Observadores y métricas auxiliares.
"""
from __future__ import annotations
import math
import statistics as st

from .constants import ALIAS_DNFR, ALIAS_THETA, ALIAS_dEPI, METRIC_DEFAULTS
from .helpers import (
    get_attr,
    list_mean,
    register_callback,
    angle_diff,
    ensure_history,
    count_glyphs,
)
from .constants_glifos import ESTABILIZADORES, DISRUPTIVOS
from .gamma import kuramoto_R_psi

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
    callbacks = [
        ("before_step", std_before),
        ("after_step", std_after),
        ("on_remesh", std_on_remesh),
    ]
    for event, fn in callbacks:
        register_callback(G, event, fn)
    G.graph.setdefault("_STD_OBSERVER", "attached")
    return G

def coherencia_global(G) -> float:
    """Proxy de C(t): alta cuando |ΔNFR| y |dEPI_dt| son pequeños."""
    dnfr = list_mean(abs(get_attr(G.nodes[n], ALIAS_DNFR, 0.0)) for n in G.nodes())
    dEPI = list_mean(abs(get_attr(G.nodes[n], ALIAS_dEPI, 0.0)) for n in G.nodes())
    return 1.0 / (1.0 + dnfr + dEPI)


def _phase_sums(G) -> tuple[float, float, int]:
    """Devuelve sumX, sumY y el número de nodos."""
    sumX = 0.0
    sumY = 0.0
    count = 0
    for n in G.nodes():
        th = get_attr(G.nodes[n], ALIAS_THETA, 0.0)
        sumX += math.cos(th)
        sumY += math.sin(th)
        count += 1
    return sumX, sumY, count


def sincronía_fase(G) -> float:
    sumX, sumY, count = _phase_sums(G)
    if count == 0:
        return 1.0
    th = math.atan2(sumY / count, sumX / count)
    # varianza angular aproximada (0 = muy sincronizado)
    var = (
        st.pvariance(
            [
                angle_diff(get_attr(G.nodes[n], ALIAS_THETA, 0.0), th)
                for n in G.nodes()
            ]
        )
        if count > 1
        else 0.0
    )
    return 1.0 / (1.0 + var)

def orden_kuramoto(G) -> float:
    """R en [0,1], 1 = fases perfectamente alineadas."""
    if G.number_of_nodes() == 0:
        return 1.0
    R, _ = kuramoto_R_psi(G)
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


def wbar(G, window: int | None = None) -> float:
    """Devuelve W̄ = media de C(t) en una ventana reciente."""
    hist = G.graph.get("history", {})
    cs = hist.get("C_steps", [])
    if not cs:
        # fallback: coherencia instantánea
        return coherencia_global(G)
    if window is None:
        window = int(G.graph.get("WBAR_WINDOW", METRIC_DEFAULTS.get("WBAR_WINDOW", 25)))
    w = min(len(cs), max(1, int(window)))
    return float(sum(cs[-w:]) / w)
