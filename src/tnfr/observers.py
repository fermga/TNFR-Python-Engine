"""Observer management."""

from __future__ import annotations

import statistics
from collections.abc import Mapping
from functools import partial
from statistics import StatisticsError, pvariance

from .alias import get_theta_attr
from .utils import CallbackEvent, callback_manager
from .config.constants import GLYPH_GROUPS
from .gamma import kuramoto_R_psi
from .glyph_history import (
    append_metric,
    count_glyphs,
    ensure_history,
)
from .utils import angle_diff
from .metrics.common import compute_coherence
from .types import Glyph, GlyphLoadDistribution, TNFRGraph
from .utils import (
    get_logger,
    get_numpy,
    mix_groups,
    normalize_counter,
)
from .validation import validate_window
from .telemetry import ensure_nu_f_telemetry, record_nu_f_window

__all__ = (
    "attach_standard_observer",
    "kuramoto_metrics",
    "phase_sync",
    "kuramoto_order",
    "glyph_load",
    "wbar",
    "DEFAULT_GLYPH_LOAD_SPAN",
    "DEFAULT_WBAR_SPAN",
)

logger = get_logger(__name__)

DEFAULT_GLYPH_LOAD_SPAN = 50
DEFAULT_WBAR_SPAN = 25


# -------------------------
# Standard Γ(R) observer
# -------------------------
def _std_log(kind: str, G: TNFRGraph, ctx: Mapping[str, object]) -> None:
    """Store compact events in ``history['events']``."""
    h = ensure_history(G)
    append_metric(h, "events", (kind, dict(ctx)))


_STD_CALLBACKS = {
    CallbackEvent.BEFORE_STEP.value: partial(_std_log, "before"),
    CallbackEvent.AFTER_STEP.value: partial(_std_log, "after"),
    CallbackEvent.ON_REMESH.value: partial(_std_log, "remesh"),
    CallbackEvent.CACHE_METRICS.value: partial(_std_log, "cache"),
}

_REORG_STATE_KEY = "_std_observer_reorg"


def _resolve_reorg_state(G: TNFRGraph) -> dict[str, object]:
    state = G.graph.get(_REORG_STATE_KEY)
    if not isinstance(state, dict):
        state = {}
        G.graph[_REORG_STATE_KEY] = state
    return state


def _before_step_reorg(G: TNFRGraph, ctx: Mapping[str, object] | None) -> None:
    """Capture structural time metadata before the step starts."""

    ensure_nu_f_telemetry(G, confidence_level=None)
    state = _resolve_reorg_state(G)
    step_idx = ctx.get("step") if ctx else None
    try:
        state["step"] = int(step_idx) if step_idx is not None else None
    except (TypeError, ValueError):
        state["step"] = None
    start_t = float(G.graph.get("_t", 0.0))
    state["start_t"] = start_t
    dt_raw = ctx.get("dt") if ctx else None
    try:
        state["dt"] = float(dt_raw) if dt_raw is not None else None
    except (TypeError, ValueError):
        state["dt"] = None


def _after_step_reorg(G: TNFRGraph, ctx: Mapping[str, object] | None) -> None:
    """Record the reorganisation window for νf telemetry."""

    state = _resolve_reorg_state(G)
    pending_step = state.get("step")
    ctx_step = ctx.get("step") if ctx else None
    if pending_step is not None and ctx_step is not None and pending_step != ctx_step:
        # Ignore mismatched callbacks to avoid double counting.
        return

    try:
        start_t = float(state.get("start_t", float(G.graph.get("_t", 0.0))))
    except (TypeError, ValueError):
        start_t = float(G.graph.get("_t", 0.0))
    end_t = float(G.graph.get("_t", start_t))
    dt_raw = state.get("dt")
    try:
        duration = float(dt_raw) if dt_raw is not None else end_t - start_t
    except (TypeError, ValueError):
        duration = end_t - start_t
    if duration <= 0.0:
        duration = end_t - start_t
    if duration <= 0.0:
        return

    stable_frac = ctx.get("stable_frac") if ctx else None
    if stable_frac is None:
        hist = ensure_history(G)
        series = hist.get("stable_frac", [])
        stable_frac = series[-1] if series else None
    try:
        stable_frac_f = float(stable_frac) if stable_frac is not None else None
    except (TypeError, ValueError):
        stable_frac_f = None
    total_nodes = G.number_of_nodes()
    if stable_frac_f is None:
        reorganisations = total_nodes
    else:
        frac = min(max(stable_frac_f, 0.0), 1.0)
        stable_nodes = int(round(frac * total_nodes))
        reorganisations = max(total_nodes - stable_nodes, 0)

    record_nu_f_window(
        G,
        reorganisations,
        duration,
        start=start_t,
        end=end_t,
    )
    state["last_duration"] = duration
    state["last_reorganisations"] = reorganisations
    state["last_end_t"] = end_t
    state["step"] = None


def attach_standard_observer(G: TNFRGraph) -> TNFRGraph:
    """Register standard callbacks: before_step, after_step, on_remesh."""
    if G.graph.get("_STD_OBSERVER"):
        return G
    for event, fn in _STD_CALLBACKS.items():
        callback_manager.register_callback(G, event, fn)
    callback_manager.register_callback(
        G,
        CallbackEvent.BEFORE_STEP.value,
        _before_step_reorg,
        name="std_reorg_before",
    )
    callback_manager.register_callback(
        G,
        CallbackEvent.AFTER_STEP.value,
        _after_step_reorg,
        name="std_reorg_after",
    )
    ensure_nu_f_telemetry(G, confidence_level=None)
    G.graph["_STD_OBSERVER"] = "attached"
    return G


def _ensure_nodes(G: TNFRGraph) -> bool:
    """Return ``True`` when the graph has nodes."""
    return bool(G.number_of_nodes())


def kuramoto_metrics(G: TNFRGraph) -> tuple[float, float]:
    """Return Kuramoto order ``R`` and mean phase ``ψ``.

    Delegates to :func:`kuramoto_R_psi` and performs the computation exactly
    once per invocation.
    """
    return kuramoto_R_psi(G)


def phase_sync(
    G: TNFRGraph,
    R: float | None = None,
    psi: float | None = None,
) -> float:
    """Return a [0, 1] synchrony index derived from phase dispersion."""

    if not _ensure_nodes(G):
        return 1.0
    if psi is None:
        _, psi = kuramoto_metrics(G)

    def _theta(nd: Mapping[str, object]) -> float:
        value = get_theta_attr(nd, 0.0)
        return float(value) if value is not None else 0.0

    diffs = (angle_diff(_theta(data), psi) for _, data in G.nodes(data=True))
    # Try NumPy for a vectorised population variance
    np = get_numpy()
    if np is not None:
        arr = np.fromiter(diffs, dtype=float)
        var = float(np.var(arr)) if arr.size else 0.0
    else:
        try:
            var = pvariance(diffs)
        except StatisticsError:
            var = 0.0
    return 1.0 / (1.0 + var)


def kuramoto_order(G: TNFRGraph, R: float | None = None, psi: float | None = None) -> float:
    """R in [0,1], 1 means perfectly aligned phases."""
    if not _ensure_nodes(G):
        return 1.0
    if R is None or psi is None:
        R, psi = kuramoto_metrics(G)
    return float(R)


def glyph_load(G: TNFRGraph, window: int | None = None) -> GlyphLoadDistribution:
    """Return distribution of structural operators applied in the network.

    Analyzes which structural operator symbols (glyphs) have been applied to
    nodes in the network over a given time window.

    - ``window``: if provided, count only the last ``window`` events per node;
      otherwise use :data:`DEFAULT_GLYPH_LOAD_SPAN`.

    Returns a dict with proportions per structural operator symbol and useful aggregates.
    """
    if window == 0:
        return {"_count": 0.0}
    if window is None:
        window_int = DEFAULT_GLYPH_LOAD_SPAN
    else:
        window_int = validate_window(window, positive=True)
    total = count_glyphs(G, window=window_int, last_only=(window_int == 1))
    dist_raw, count = normalize_counter(total)
    if count == 0:
        return {"_count": 0.0}
    dist = mix_groups(dist_raw, GLYPH_GROUPS)
    glyph_dist: GlyphLoadDistribution = {}
    for key, value in dist.items():
        try:
            glyph_key: Glyph | str = Glyph(key)
        except ValueError:
            glyph_key = key
        glyph_dist[glyph_key] = value
    glyph_dist["_count"] = float(count)
    return glyph_dist


def wbar(G: TNFRGraph, window: int | None = None) -> float:
    """Return W̄ = mean of ``C(t)`` over a recent window.

    Uses :func:`ensure_history` to obtain ``G.graph['history']`` and falls back
    to the instantaneous coherence when ``"C_steps"`` is missing or empty.
    """
    hist = ensure_history(G)
    cs = list(hist.get("C_steps", []))
    if not cs:
        # fallback: instantaneous coherence
        return compute_coherence(G)
    w_param = DEFAULT_WBAR_SPAN if window is None else window
    w = validate_window(w_param, positive=True)
    w = min(len(cs), w)
    return float(statistics.fmean(cs[-w:]))
