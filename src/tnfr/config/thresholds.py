"""Canonical thresholds for structural operator preconditions.

This module defines configurable thresholds that enforce TNFR canonical
preconditions for structural operators. These thresholds ensure structural
integrity and operational fidelity according to TNFR.pdf specifications.

All thresholds are exported as module-level constants with sensible defaults
that can be overridden via graph metadata or configuration presets.
"""

from __future__ import annotations

__all__ = [
    "EPI_LATENT_MAX",
    "VF_BASAL_THRESHOLD",
    "EPSILON_MIN_EMISSION",
    "MIN_NETWORK_DEGREE_COUPLING",
]

# -------------------------
# AL (Emission) Thresholds
# -------------------------

# Maximum EPI for latent state - AL requires nodes in latent/low-activation state
# According to TNFR.pdf §2.2.1, emission activates nascent structures
EPI_LATENT_MAX: float = 0.8

# Minimum structural frequency (νf) for emission - ensures sufficient
# reorganization capacity. Below this threshold, the node cannot sustain
# the structural frequency activation that AL initiates.
VF_BASAL_THRESHOLD: float = 0.5

# Minimum coherence gradient (epsilon) for meaningful emission
# This represents the minimum structural pressure needed to justify activation
EPSILON_MIN_EMISSION: float = 0.1

# Minimum network degree for effective phase coupling
# Nodes with degree below this threshold will trigger a warning (not error)
# as AL can still activate isolated nodes, but coupling will be limited
MIN_NETWORK_DEGREE_COUPLING: int = 1
