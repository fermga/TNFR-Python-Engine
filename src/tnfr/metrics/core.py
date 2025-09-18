"""Basic metrics orchestrator."""

from __future__ import annotations

from typing import Any

from ..callback_utils import CallbackEvent, callback_manager
from ..constants import get_param
from ..glyph_history import append_metric, ensure_history
from ..logging_utils import get_logger
from .coherence import (
    _aggregate_si,
    _track_stability,
    _update_coherence,
    _update_phase_sync,
    _update_sigma,
    register_coherence_callbacks,
)
from .diagnosis import register_diagnosis_callbacks
from .glyph_timing import _compute_advanced_metrics
from .reporting import (
    Tg_by_node,
    Tg_global,
    glyphogram_series,
    glyph_top,
    latency_series,
)

logger = get_logger(__name__)

__all__ = [
    "_metrics_step",
    "register_metrics_callbacks",
    "Tg_global",
    "Tg_by_node",
    "latency_series",
    "glyphogram_series",
    "glyph_top",
]


def _metrics_step(G, ctx: dict[str, Any] | None = None):
    """Update operational TNFR metrics per step."""

    del ctx

    cfg = get_param(G, "METRICS")
    if not cfg.get("enabled", True):
        return

    hist = ensure_history(G)
    dt = float(get_param(G, "DT"))
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
    _compute_advanced_metrics(G, hist, t, dt, cfg)


def register_metrics_callbacks(G) -> None:
    callback_manager.register_callback(
        G,
        event=CallbackEvent.AFTER_STEP.value,
        func=_metrics_step,
        name="metrics_step",
    )
    register_coherence_callbacks(G)
    register_diagnosis_callbacks(G)
