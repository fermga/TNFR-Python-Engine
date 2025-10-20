"""Basic metrics orchestrator."""

from __future__ import annotations

from typing import Any, cast

from ..types import (
    GlyphSelector,
    NodeId,
    SelectorPreselectionChoices,
    SelectorPreselectionMetrics,
    SelectorPreselectionPayload,
    TNFRGraph,
    TraceCallback,
    TraceFieldFn,
    TraceFieldMap,
    TraceFieldRegistry,
)

from ..callback_utils import CallbackEvent, callback_manager
from ..constants import get_param
from ..glyph_history import append_metric, ensure_history
from ..utils import get_logger
from .coherence import (
    _aggregate_si,
    _track_stability,
    _update_coherence,
    _update_phase_sync,
    _update_sigma,
    register_coherence_callbacks,
)
from .diagnosis import register_diagnosis_callbacks
from .glyph_timing import _compute_advanced_metrics, GlyphMetricsHistory
from .reporting import (
    Tg_by_node,
    Tg_global,
    glyphogram_series,
    glyph_top,
    latency_series,
)

logger = get_logger(__name__)

__all__ = [
    "TNFRGraph",
    "NodeId",
    "GlyphSelector",
    "SelectorPreselectionMetrics",
    "SelectorPreselectionChoices",
    "SelectorPreselectionPayload",
    "TraceCallback",
    "TraceFieldFn",
    "TraceFieldMap",
    "TraceFieldRegistry",
    "_metrics_step",
    "register_metrics_callbacks",
    "Tg_global",
    "Tg_by_node",
    "latency_series",
    "glyphogram_series",
    "glyph_top",
]


def _metrics_step(G: TNFRGraph, ctx: dict[str, Any] | None = None) -> None:
    """Update operational TNFR metrics per step."""

    del ctx

    cfg = get_param(G, "METRICS")
    if not cfg.get("enabled", True):
        return

    hist = ensure_history(G)
    metrics_sentinel_key = "_metrics_history_id"
    history_id = id(hist)
    if G.graph.get(metrics_sentinel_key) != history_id:
        for k in (
            "C_steps",
            "stable_frac",
            "phase_sync",
            "glyph_load_estab",
            "glyph_load_disr",
            "Si_mean",
            "Si_hi_frac",
            "Si_lo_frac",
            "delta_Si",
            "B",
        ):
            hist.setdefault(k, [])
        G.graph[metrics_sentinel_key] = history_id

    dt = float(get_param(G, "DT"))
    eps_dnfr = float(get_param(G, "EPS_DNFR_STABLE"))
    eps_depi = float(get_param(G, "EPS_DEPI_STABLE"))
    t = float(G.graph.get("_t", 0.0))

    _update_coherence(G, hist)

    raw_jobs = cfg.get("n_jobs")
    metrics_jobs: int | None
    try:
        metrics_jobs = None if raw_jobs is None else int(raw_jobs)
    except (TypeError, ValueError):
        metrics_jobs = None
    else:
        if metrics_jobs <= 0:
            metrics_jobs = None

    _track_stability(
        G,
        hist,
        dt,
        eps_dnfr,
        eps_depi,
        n_jobs=metrics_jobs,
    )
    try:
        _update_phase_sync(G, hist)
        _update_sigma(G, hist)
        if hist.get("C_steps") and hist.get("stable_frac"):
            append_metric(
                hist,
                "iota",
                hist["C_steps"][-1] * hist["stable_frac"][-1],
            )
    except (KeyError, AttributeError, TypeError) as exc:
        logger.debug("observer update failed: %s", exc)

    _aggregate_si(G, hist, n_jobs=metrics_jobs)

    _compute_advanced_metrics(
        G,
        cast(GlyphMetricsHistory, hist),
        t,
        dt,
        cfg,
        n_jobs=metrics_jobs,
    )


def register_metrics_callbacks(G: TNFRGraph) -> None:
    callback_manager.register_callback(
        G,
        event=CallbackEvent.AFTER_STEP.value,
        func=_metrics_step,
        name="metrics_step",
    )
    register_coherence_callbacks(G)
    register_diagnosis_callbacks(G)
