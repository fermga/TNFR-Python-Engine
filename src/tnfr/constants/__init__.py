"""Shared constants (backward compatibility layer).

This module re-exports configuration from tnfr.config for backward compatibility.
New code should import directly from tnfr.config.

Migration Path:
    Old: from tnfr.constants import DEFAULTS, inject_defaults
    New: from tnfr.config import DEFAULTS, inject_defaults
"""

from __future__ import annotations

# Re-export all constants and utilities from tnfr.config
from ..config import (
    ALIASES,
    CANONICAL_STATE_TOKENS,
    COHERENCE,
    CORE_DEFAULTS,
    D2EPI_PRIMARY,
    D2VF_PRIMARY,
    DEFAULT_SECTIONS,
    DEFAULTS,
    DIAGNOSIS,
    DNFR_PRIMARY,
    EPI_KIND_PRIMARY,
    EPI_PRIMARY,
    GRAMMAR_CANON,
    INIT_DEFAULTS,
    METRIC_DEFAULTS,
    METRICS,
    REMESH_DEFAULTS,
    SI_PRIMARY,
    SIGMA,
    STATE_DISSONANT,
    STATE_STABLE,
    STATE_TRANSITION,
    THETA_KEY,
    THETA_PRIMARY,
    TRACE,
    VF_KEY,
    VF_PRIMARY,
    dEPI_PRIMARY,
    dSI_PRIMARY,
    dVF_PRIMARY,
    ensure_node_offset_map,
    get_aliases,
    get_graph_param,
    get_param,
    inject_defaults,
    merge_overrides,
    normalise_state_token,
)

__all__ = (
    "CORE_DEFAULTS",
    "INIT_DEFAULTS",
    "REMESH_DEFAULTS",
    "METRIC_DEFAULTS",
    "SIGMA",
    "TRACE",
    "METRICS",
    "GRAMMAR_CANON",
    "COHERENCE",
    "DIAGNOSIS",
    "DEFAULTS",
    "DEFAULT_SECTIONS",
    "ALIASES",
    "inject_defaults",
    "merge_overrides",
    "get_param",
    "get_graph_param",
    "get_aliases",
    "VF_KEY",
    "THETA_KEY",
    "VF_PRIMARY",
    "THETA_PRIMARY",
    "DNFR_PRIMARY",
    "EPI_PRIMARY",
    "EPI_KIND_PRIMARY",
    "SI_PRIMARY",
    "dEPI_PRIMARY",
    "D2EPI_PRIMARY",
    "dVF_PRIMARY",
    "D2VF_PRIMARY",
    "dSI_PRIMARY",
    "STATE_STABLE",
    "STATE_TRANSITION",
    "STATE_DISSONANT",
    "CANONICAL_STATE_TOKENS",
    "normalise_state_token",
    "ensure_node_offset_map",
)
