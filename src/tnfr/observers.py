"""Gestión de observadores."""
from __future__ import annotations
import math
import statistics as st

from .constants import ALIAS_THETA, METRIC_DEFAULTS
from .helpers import (
    get_attr,
    register_callback,
    angle_diff,
    ensure_history,
    count_glyphs,
    compute_coherence,
)
from .constants_glifos import GLYPH_GROUPS
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


def _phase_sums(G) -> tuple[float, float, list[float]]:
    """Devuelve sumX, sumY y la lista de fases nodales."""
    sumX = 0.0
    sumY = 0.0
    fases: list[float] = []
    for n in G.nodes():
        th = get_attr(G.nodes[n], ALIAS_THETA, 0.0)
        sumX += math.cos(th)
        sumY += math.sin(th)
        fases.append(th)
    return sumX, sumY, fases


def sincronía_fase(G) -> float:
    sumX, sumY, fases = _phase_sums(G)
    count = len(fases)
    if count == 0:
        return 1.0
    th = math.atan2(sumY / count, sumX / count)
    # varianza angular aproximada (0 = muy sincronizado)
    var = (
        st.pvariance([angle_diff(f, th) for f in fases])
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
    total = count_glyphs(G, window=window, last_only=(window == 1))
    count = sum(total.values())
    if count == 0:
        return {"_count": 0}


    # Proporciones por glifo
    dist = {k: v / count for k, v in total.items()}

    for label, glyphs in GLYPH_GROUPS.items():
        dist[f"_{label}"] = sum(dist.get(k, 0.0) for k in glyphs)

    dist["_count"] = count
    return dist


def wbar(G, window: int | None = None) -> float:
    """Devuelve W̄ = media de C(t) en una ventana reciente."""
    hist = G.graph.get("history", {})
    cs = hist.get("C_steps", [])
    if not cs:
        # fallback: coherencia instantánea
        return compute_coherence(G)
    if window is None:
        window = int(G.graph.get("WBAR_WINDOW", METRIC_DEFAULTS.get("WBAR_WINDOW", 25)))
    w = min(len(cs), max(1, int(window)))
    return float(sum(cs[-w:]) / w)
