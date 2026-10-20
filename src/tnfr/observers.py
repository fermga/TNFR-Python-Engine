"""Observer management."""

from __future__ import annotations

import statistics
from collections import deque
from collections.abc import Mapping
from functools import partial
from statistics import StatisticsError, pvariance

from ._exact_time import bounded_represented_real, finite_represented_real
from .alias import get_theta_attr
from .config.constants import GLYPH_GROUPS
from .gamma import kuramoto_R_psi
from .glyph_history import append_metric, count_glyphs, ensure_history
from .mathematics.unified_numerical import np
from .metrics.common import compute_coherence
from .telemetry import (
    NuFTelemetryAccumulator,
    ensure_nu_f_telemetry,
    record_nu_f_window,
)
from .telemetry.nu_f import _ACCUMULATOR_KEY
from .types import Glyph, GlyphLoadDistribution, TNFRGraph
from .utils import (
    CallbackEvent,
    angle_diff,
    callback_manager,
    get_logger,
    mix_groups,
    normalize_counter,
)
from .validation import validate_window

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
    """Capture one runtime interval and the current stability revision."""
    from .metrics.coherence import _stability_observation_revision

    state = _resolve_reorg_state(G)
    state.update(pending=False, status="unavailable", reason="missing_step")
    if ctx is None or "step" not in ctx:
        return
    step = validate_window(ctx["step"])
    # The runtime's initial clock defaults to zero. Later boundaries must be
    # actual clock observations; a requested dt never supplies elapsed time.
    start_t = finite_represented_real(G.graph.get("_t", 0.0), "start time")[0]
    revision = _stability_observation_revision(G)
    state.update(
        step=step,
        start_t=start_t,
        stability_revision=revision,
        pending=True,
        status="pending",
        reason=None,
    )


def _after_step_reorg(G: TNFRGraph, ctx: Mapping[str, object] | None) -> None:
    """Record the fresh unstable-node proxy once for a completed interval.

    This counts nodes outside the stored pressure/rate stability thresholds,
    not observed operator events or primitive capacity changes. The optional
    count-rate/Poisson interpretation belongs to the telemetry model. Missing,
    stale or invalid stability evidence contributes no window, not zero or a
    count of every node. Callback context can contain the previous history
    value, so freshness and the latest reading come from the metrics owner.
    """
    from .metrics.coherence import _stability_observation_revision
    from .metrics.core import _update_nu_f_snapshot

    state = _resolve_reorg_state(G)
    if state.get("pending") is not True or ctx is None or "step" not in ctx:
        return
    pending_step = state.get("step")
    ctx_step = validate_window(ctx["step"])
    if pending_step != ctx_step:
        return
    # Consume the matching callback even when its evidence is unavailable or
    # raises. A later duplicate must not reinterpret a different observation.
    state.update(
        pending=False,
        step=None,
        status="unavailable",
        reason="missing_clock",
        last_duration=None,
        last_reorganisations=None,
        last_end_t=None,
    )
    if "_t" not in G.graph:
        return
    start_t = finite_represented_real(state["start_t"], "start time")[0]
    end_t = finite_represented_real(G.graph["_t"], "end time")[0]
    duration = finite_represented_real(end_t - start_t, "elapsed time")[0]
    if duration <= 0.0:
        state["reason"] = "nonpositive_interval"
        return
    if _stability_observation_revision(G) <= state["stability_revision"]:
        state["reason"] = "missing_current_stability"
        return
    hist = G.graph.get("history")
    series = hist.get("stable_frac") if isinstance(hist, Mapping) else None
    if not isinstance(series, (list, tuple, deque)) or not series:
        state["reason"] = "missing_current_stability"
        return
    try:
        frac = bounded_represented_real(
            series[-1], "stable fraction", minimum=0.0, maximum=1.0
        )[0]
    except (TypeError, ValueError):
        state["reason"] = "invalid_current_stability"
        return
    total_nodes = G.number_of_nodes()
    stable_nodes = int(round(frac * total_nodes))
    reorganisations = total_nodes - stable_nodes
    snapshot = record_nu_f_window(
        G,
        reorganisations,
        duration,
        start=start_t,
        end=end_t,
    )
    _update_nu_f_snapshot(
        G,
        hist,
        record_history=False,
        snapshot=snapshot,
        replace_current_history=True,
    )
    state["last_duration"] = duration
    state["last_reorganisations"] = reorganisations
    state["last_end_t"] = end_t
    state.update(status="recorded", reason=None)


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
    existing = G.graph.get(_ACCUMULATOR_KEY)
    ensure_nu_f_telemetry(
        G,
        confidence_level=None,
        history_limit=None if isinstance(existing, NuFTelemetryAccumulator) else 128,
    )
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
    if np is not None:
        arr = np.fromiter(diffs, dtype=float)
        var = float(np.var(arr)) if arr.size else 0.0
    else:
        try:
            var = pvariance(diffs)
        except StatisticsError:
            var = 0.0
    return 1.0 / (1.0 + var)


def kuramoto_order(
    G: TNFRGraph, R: float | None = None, psi: float | None = None
) -> float:
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
