"""Registerable metrics."""

from __future__ import annotations

from .core import register_metrics_callbacks, _metrics_step
from .reporting import (
    Tg_global,
    Tg_by_node,
    latency_series,
    glyphogram_series,
    glyph_top,
    glyph_dwell_stats,
)
from .glyph_timing import (
    _tg_state,
    _update_tg,
    _update_latency_index,
    _update_epi_support,
    _compute_advanced_metrics,
)
from .coherence import (
    _aggregate_si,
    _track_stability,
    coherence_matrix,
    local_phase_sync,
    local_phase_sync_weighted,
    register_coherence_callbacks,
)
from .diagnosis import (
    register_diagnosis_callbacks,
    dissonance_events,
)
from .export import export_metrics

__all__ = (
    "register_metrics_callbacks",
    "Tg_global",
    "Tg_by_node",
    "latency_series",
    "glyphogram_series",
    "glyph_top",
    "glyph_dwell_stats",
    "_tg_state",
    "_update_tg",
    "_update_latency_index",
    "_update_epi_support",
    "_track_stability",
    "_aggregate_si",
    "_compute_advanced_metrics",
    "_metrics_step",
    "coherence_matrix",
    "local_phase_sync",
    "local_phase_sync_weighted",
    "register_coherence_callbacks",
    "register_diagnosis_callbacks",
    "dissonance_events",
    "export_metrics",
)
