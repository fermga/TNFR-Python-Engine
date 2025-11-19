"""Registerable metrics."""

from __future__ import annotations

from .cache_utils import (
    CacheStats,
    configure_hot_path_caches,
    get_cache_config,
    log_cache_metrics,
)
from .coherence import (
    coherence_matrix,
    local_phase_sync,
    local_phase_sync_weighted,
    register_coherence_callbacks,
)
from .core import register_metrics_callbacks
from .diagnosis import (
    dissonance_events,
    register_diagnosis_callbacks,
)
from .emergence import (
    compute_bifurcation_rate,
    compute_emergence_index,
    compute_metabolic_efficiency,
    compute_structural_complexity,
)
from .export import export_metrics
from .learning_metrics import (
    compute_consolidation_index,
    compute_learning_efficiency,
    compute_learning_plasticity,
    glyph_history_to_operator_names,
)
from .phase_compatibility import (
    compute_network_phase_alignment,
    compute_phase_coupling_strength,
    is_phase_compatible,
)
from .reporting import (
    Tg_by_node,
    Tg_global,
    build_metrics_summary,
    glyph_top,
    glyphogram_series,
    latency_series,
)
from .telemetry import (
    TelemetryEmitter,
    TelemetryEvent,
)
from .tetrad import (
    collect_tetrad_snapshot,
    get_tetrad_sample_interval,
)

__all__ = (
    "register_metrics_callbacks",
    "Tg_global",
    "Tg_by_node",
    "latency_series",
    "glyphogram_series",
    "glyph_top",
    "build_metrics_summary",
    "coherence_matrix",
    "local_phase_sync",
    "local_phase_sync_weighted",
    "register_coherence_callbacks",
    "register_diagnosis_callbacks",
    "dissonance_events",
    "export_metrics",
    "CacheStats",
    "configure_hot_path_caches",
    "get_cache_config",
    "log_cache_metrics",
    "compute_learning_plasticity",
    "compute_consolidation_index",
    "compute_learning_efficiency",
    "glyph_history_to_operator_names",
    "compute_structural_complexity",
    "compute_bifurcation_rate",
    "compute_metabolic_efficiency",
    "compute_emergence_index",
    "compute_phase_coupling_strength",
    "is_phase_compatible",
    "compute_network_phase_alignment",
    "TelemetryEmitter",
    "TelemetryEvent",
    "collect_tetrad_snapshot",
    "get_tetrad_sample_interval",
)
