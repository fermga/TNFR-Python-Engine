"""Basic metrics orchestrator."""

from __future__ import annotations

from ..callback_utils import register_callback
from ..constants import get_param
from ..glyph_history import append_metric, ensure_history
from ..logging import get_module_logger
from .coherence import register_coherence_callbacks
from .diagnosis import register_diagnosis_callbacks
from .coherence_updates import (
    _aggregate_si,
    _record_metrics,
    _track_stability,
    _update_coherence,
    _update_phase_sync,
    _update_sigma,
)
from .glyph_timing import (
    GlyphTiming,
    _compute_advanced_metrics,
    LATENT_GLYPH,
    _tg_state,
    _update_tg,
    _update_tg_node,
    _update_glyphogram,
    _update_latency_index,
    _update_epi_support,
    _update_morph_metrics,
    for_each_glyph,
)
from .reporting import (
    Tg_by_node,
    Tg_global,
    glyph_dwell_stats,
    glyphogram_series,
    glyph_top,
    latency_series,
)

logger = get_module_logger(__name__)

__all__ = [
    "LATENT_GLYPH",
    "GlyphTiming",
    "_tg_state",
    "for_each_glyph",
    "_update_tg_node",
    "_update_tg",
    "_update_glyphogram",
    "_update_latency_index",
    "_update_epi_support",
    "_update_morph_metrics",
    "_update_coherence",
    "_record_metrics",
    "_update_phase_sync",
    "_update_sigma",
    "_track_stability",
    "_aggregate_si",
    "_compute_advanced_metrics",
    "_metrics_step",
    "register_metrics_callbacks",
    "Tg_global",
    "Tg_by_node",
    "latency_series",
    "glyphogram_series",
    "glyph_top",
    "glyph_dwell_stats",
]


def _metrics_step(G, *args, **kwargs):
    """Update operational TNFR metrics per step."""

    cfg = get_param(G, "METRICS")
    if not cfg.get("enabled", True):
        return

    hist = ensure_history(G)
    dt = float(get_param(G, "DT"))
    thr = float(get_param(G, "EPI_SUPPORT_THR"))
    eps_dnfr = float(get_param(G, "EPS_DNFR_STABLE"))
    eps_depi = float(get_param(G, "EPS_DEPI_STABLE"))
    t = float(G.graph.get("_t", 0.0))

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

    _update_coherence(G, hist)
    _track_stability(G, hist, dt, eps_dnfr, eps_depi)
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

    _aggregate_si(G, hist)
    _compute_advanced_metrics(G, hist, t, dt, cfg, thr)


def register_metrics_callbacks(G) -> None:
    register_callback(
        G, event="after_step", func=_metrics_step, name="metrics_step"
    )
    register_coherence_callbacks(G)
    register_diagnosis_callbacks(G)
