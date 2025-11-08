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
    "EPI_SATURATION_MAX",
    "DNFR_RECEPTION_MAX",
    "EPI_IL_MIN",
    "EPI_IL_MAX",
    "VF_IL_MIN",
    "DNFR_IL_CRITICAL",
    "EPI_RA_MIN",
    "DNFR_RA_MAX",
    "VF_RA_MIN",
    "PHASE_RA_MAX_DIFF",
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

# -------------------------
# EN (Reception) Thresholds
# -------------------------

# Maximum EPI for reception - EN requires nodes with receptive capacity
# According to TNFR.pdf §2.2.1, reception integrates external coherence
# into local structure. If EPI is saturated, node cannot receive more coherence.
EPI_SATURATION_MAX: float = 0.9

# Maximum DNFR for stable reception - EN requires low dissonance
# Excessive reorganization pressure prevents effective integration
# of external coherence. Consider IL (Coherence) first to stabilize.
DNFR_RECEPTION_MAX: float = 0.15

# -------------------------
# IL (Coherence) Thresholds
# -------------------------

# Minimum EPI for coherence - IL requires active structural form
# According to TNFR.pdf §2.2.1, coherence stabilizes existing structure
# Zero or negative EPI indicates no structure to stabilize
EPI_IL_MIN: float = 0.0

# Maximum EPI for coherence - IL may increment EPI during stabilization
# Leave room for stabilization increment to avoid saturation
# Values near 1.0 indicate node is already highly stable
EPI_IL_MAX: float = 1.0

# Minimum structural frequency for coherence - IL requires active νf
# IL reduces ΔNFR proportional to νf via nodal equation ∂EPI/∂t = νf · ΔNFR
# Zero νf prevents effective stabilization
VF_IL_MIN: float = 0.0

# Critical ΔNFR threshold - excessive reorganization pressure
# Above this threshold, node is highly unstable and may benefit from
# OZ (Dissonance) → IL sequence for controlled stabilization
# This is a warning threshold, not a hard failure
DNFR_IL_CRITICAL: float = 0.8

# -------------------------
# RA (Resonance) Thresholds
# -------------------------

# Minimum EPI for resonance source - RA requires coherent structural form
# According to TNFR theory, resonance propagates existing coherence through
# network connections. Source node must have sufficient EPI to propagate.
EPI_RA_MIN: float = 0.1

# Maximum ΔNFR for resonance - RA requires controlled dissonance
# Excessive reorganization pressure prevents stable resonance propagation.
# Consider IL (Coherence) first to stabilize before applying RA.
DNFR_RA_MAX: float = 0.5

# Minimum structural frequency for resonance - RA requires active νf
# Resonance amplifies νf across the network. Zero νf prevents propagation
# dynamics from occurring. Consider AL (Emission) or VAL (Expansion) first.
VF_RA_MIN: float = 0.01

# Maximum phase difference for optimal resonance - RA prefers phase alignment
# This is a soft threshold (warning only). Larger phase differences reduce
# resonance effectiveness but don't prevent propagation entirely.
# Measured in radians: π/3 ≈ 1.0 rad ≈ 60 degrees
PHASE_RA_MAX_DIFF: float = 1.0
