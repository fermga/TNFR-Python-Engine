"""Observer management."""

from __future__ import annotations
import math
import statistics as st
from itertools import islice
from functools import partial

from .constants import ALIAS_THETA, METRIC_DEFAULTS
from .helpers import (
    get_attr,
    angle_diff,
    compute_coherence,
)
from .callback_utils import register_callback
from .glyph_history import ensure_history, count_glyphs
from .collections_utils import normalize_counter, mix_groups
from .constants_glyphs import GLYPH_GROUPS
from .gamma import kuramoto_R_psi


__all__ = [
    "attach_standard_observer",
    "std_before",
    "std_after",
    "std_on_remesh",
    "phase_sync",
    "kuramoto_order",
    "glyph_load",
    "wbar",
]


# -------------------------
# Observador estándar Γ(R)
# -------------------------
def _std_log(kind: str, G, ctx: dict):
    """Store compact events in ``history['events']``."""
    h = ensure_history(G)
    h.setdefault("events", []).append((kind, dict(ctx)))


_STD_CALLBACKS = {
    "before_step": partial(_std_log, "before"),
    "after_step": partial(_std_log, "after"),
    "on_remesh": partial(_std_log, "remesh"),
}

# alias conservados por compatibilidad
std_before = _STD_CALLBACKS["before_step"]
std_after = _STD_CALLBACKS["after_step"]
std_on_remesh = _STD_CALLBACKS["on_remesh"]


def attach_standard_observer(G):
    """Register standard callbacks: before_step, after_step, on_remesh."""
    for event, fn in _STD_CALLBACKS.items():
        register_callback(G, event, fn)
    G.graph.setdefault("_STD_OBSERVER", "attached")
    return G


def _phase_sums(G) -> tuple[float, float, int]:
    """Return ``sumX``, ``sumY`` and the number of nodes."""
    sumX = 0.0
    sumY = 0.0
    count = 0
    for _, data in G.nodes(data=True):
        th = get_attr(data, ALIAS_THETA, 0.0)
        sumX += math.cos(th)
        sumY += math.sin(th)
        count += 1
    return sumX, sumY, count


def phase_sync(G) -> float:
    sumX, sumY, count = _phase_sums(G)
    if count == 0:
        return 1.0
    th = math.atan2(sumY, sumX)
    # varianza angular aproximada (0 = muy sincronizado)
    mean = 0.0
    m2 = 0.0
    n = 0
    for _, data in G.nodes(data=True):
        diff = angle_diff(get_attr(data, ALIAS_THETA, 0.0), th)
        n += 1
        delta = diff - mean
        mean += delta / n
        m2 += delta * (diff - mean)
    var = m2 / n if n > 1 else 0.0
    return 1.0 / (1.0 + var)


def kuramoto_order(G) -> float:
    """R in [0,1], 1 means perfectly aligned phases."""
    if G.number_of_nodes() == 0:
        return 1.0
    R, _ = kuramoto_R_psi(G)
    return float(R)


def glyph_load(G, window: int | None = None) -> dict:
    """Return distribution of glyphs applied in the network.
    - ``window``: if provided, count only the last ``window`` events per node;
      otherwise use the deque's maxlen.
    Returns a dict with proportions per glyph and useful aggregates.
    """
    total = count_glyphs(G, window=window, last_only=(window == 1))
    dist, count = normalize_counter(total)
    if count == 0:
        return {"_count": 0}
    dist = mix_groups(dist, GLYPH_GROUPS)
    dist["_count"] = count
    return dist


def wbar(G, window: int | None = None) -> float:
    """Return W̄ = mean of C(t) over a recent window."""
    hist = G.graph.get("history", {})
    cs = hist.get("C_steps", [])
    if not cs:
        # fallback: coherencia instantánea
        return compute_coherence(G)
    if window is None:
        window = int(G.graph.get("WBAR_WINDOW", METRIC_DEFAULTS.get("WBAR_WINDOW", 25)))
    w = min(len(cs), max(1, int(window)))
    if isinstance(cs, list):
        tail = cs[-w:]
    else:
        start = len(cs) - w
        tail = islice(cs, start, None)
    return float(st.fmean(tail))
