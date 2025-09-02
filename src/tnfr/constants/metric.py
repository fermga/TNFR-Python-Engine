"""Metric constants."""
from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Dict, Any
from types import MappingProxyType


@dataclass(frozen=True)
class MetricDefaults:
    PHASE_HISTORY_MAXLEN: int = 50
    STOP_EARLY: Dict[str, Any] = field(
        default_factory=lambda: {"enabled": False, "window": 25, "fraction": 0.90}
    )
    SIGMA: Dict[str, Any] = field(
        default_factory=lambda: {
            "enabled": True,
            "weight": "Si",  # "Si" | "EPI" | "1"
            "smooth": 0.0,    # EMA sobre el vector global (0=off)
            "history_key": "sigma_global",
            "per_node": False,
        }
    )
    TRACE: Dict[str, Any] = field(
        default_factory=lambda: {
            "enabled": True,
            "capture": [
                "gamma",
                "grammar",
                "selector",
                "dnfr_weights",
                "si_weights",
                "callbacks",
                "thol_state",
                "sigma",
                "kuramoto",
                "glyph_counts",
            ],
            "history_key": "trace_meta",
        }
    )
    METRICS: Dict[str, Any] = field(
        default_factory=lambda: {
            "enabled": True,
            "save_by_node": True,
            "normalize_series": False,
        }
    )
    GRAMMAR_CANON: Dict[str, Any] = field(
        default_factory=lambda: {
            "enabled": True,
            "zhir_requires_oz_window": 3,
            "zhir_dnfr_min": 0.05,
            "thol_min_len": 2,
            "thol_max_len": 6,
            "thol_close_dnfr": 0.15,
            "si_high": 0.66,
        }
    )
    COHERENCE: Dict[str, Any] = field(
        default_factory=lambda: {
            "enabled": True,
            "scope": "neighbors",
            "weights": {"phase": 0.34, "epi": 0.33, "vf": 0.20, "si": 0.13},
            "self_on_diag": True,
            "store_mode": "sparse",
            "threshold": 0.0,
            "history_key": "W_sparse",
            "Wi_history_key": "W_i",
            "stats_history_key": "W_stats",
        }
    )
    DIAGNOSIS: Dict[str, Any] = field(
        default_factory=lambda: {
            "enabled": True,
            "window": 16,
            "history_key": "nodal_diag",
            "stable": {"Rloc_hi": 0.80, "dnfr_lo": 0.20, "persist": 3},
            "dissonance": {"Rloc_lo": 0.40, "dnfr_hi": 0.50, "persist": 3},
            "transition": {"persist": 2},
            "compute_symmetry": True,
            "include_typology": False,
            "advice": {
                "stable": ["Coherence", "Coupling", "Resonance"],
                "transition": ["Transition", "Resonance", "Self-organisation"],
                "dissonant": ["Silence", "Contraction", "Mutation"],
            },
        }
    )
    EPI_SUPPORT_THR: float = 0.05
    GLYPH_LOAD_WINDOW: int = 50
    WBAR_WINDOW: int = 25


METRIC_DEFAULTS = asdict(MetricDefaults())

SIGMA = MappingProxyType(METRIC_DEFAULTS["SIGMA"])
TRACE = MappingProxyType(METRIC_DEFAULTS["TRACE"])
METRICS = MappingProxyType(METRIC_DEFAULTS["METRICS"])
GRAMMAR_CANON = MappingProxyType(METRIC_DEFAULTS["GRAMMAR_CANON"])
COHERENCE = MappingProxyType(METRIC_DEFAULTS["COHERENCE"])
DIAGNOSIS = MappingProxyType(METRIC_DEFAULTS["DIAGNOSIS"])

DEFAULTS_PART: Dict[str, Any] = METRIC_DEFAULTS
