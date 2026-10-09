"""Configured operator-admission defaults and legacy threshold exports.

Active consumers use these selected policies through their documented graph
configuration keys. Their values are not consequences of the nodal equation
and do not prove trajectory stability. Unused compatibility exports are marked
explicitly; exporting a number does not install a runtime admission condition.
"""

from __future__ import annotations

from ..constants.canonical import (
    CONFIG_EPI_LATENT_MAX_CANONICAL,
    CONFIG_EPSILON_MIN_CANONICAL,
    CONFIG_VF_BASAL_CANONICAL,
)

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

# Strict AL admission requires the signed scalar EPI to be below this selected
# ceiling. It is not an absolute-form bound or a universal latency criterion.
EPI_LATENT_MAX: float = CONFIG_EPI_LATENT_MAX_CANONICAL

# Strict AL admission requires capacity at least this selected value.
# AL reads this capacity as a precondition and does not write it.
VF_BASAL_THRESHOLD: float = CONFIG_VF_BASAL_CANONICAL

# Legacy compatibility export: current AL execution does not read this value
# or require a pressure threshold. It must not be presented as a live gate.
EPSILON_MIN_EMISSION: float = CONFIG_EPSILON_MIN_CANONICAL

# Minimum network degree for effective phase coupling
# Nodes with degree below this threshold will trigger a warning (not error)
# as AL can still activate isolated nodes, but coupling will be limited
MIN_NETWORK_DEGREE_COUPLING: int = 1

# -------------------------
# EN (Reception) Thresholds
# -------------------------

# Selected stored-EPI upper admission bound for Reception. This policy ceiling
# is not a theorem about receptive capacity; the runtime compares the signed
# stored coordinate and otherwise blends the current neighbour EPI field.
EPI_SATURATION_MAX: float = 0.9

# Selected signed-pressure upper admission bound for Reception. It is not an
# absolute-pressure stability threshold and does not certify post-refresh C(t).
# EN itself leaves this stored pressure unchanged.
DNFR_RECEPTION_MAX: float = 0.15

# -------------------------
# IL (Coherence) Thresholds
# -------------------------

# Strict IL admission requires signed EPI above this selected lower bound.
# Negative scalar EPI remains a valid structural coordinate; rejection by this
# configured policy does not establish absence of form or pressure.
EPI_IL_MIN: float = 0.0

# Deprecated compatibility value. IL does not write EPI, so this bound is not
# consumed by strict readiness and supplies no EPI-headroom condition.
EPI_IL_MAX: float = 1.0

# Strict IL admission requires capacity above this selected lower bound.
# Zero capacity suppresses unforced EPI evolution, not the mathematical ability
# to apply an instantaneous pressure-contraction map.
VF_IL_MIN: float = 0.0

# Critical |ΔNFR| warning threshold. Either pressure sign is an IL input;
# above this magnitude, repeated IL or THOL may be needed. This is advisory,
# not a hard failure.
DNFR_IL_CRITICAL: float = 0.8

# -------------------------
# RA (Resonance) Thresholds
# -------------------------

# Strict RA admission requires scalar EPI magnitude at least this value.
# Override: RA_MIN_SOURCE_EPI. This differs from IL's signed-coordinate gate.
EPI_RA_MIN: float = 0.1

# Strict RA admission allows stored absolute pressure at most this value.
# Override: RA_MAX_DISSONANCE. Admission is not a propagation-stability theorem.
DNFR_RA_MAX: float = 0.5

# Strict RA admission requires capacity at least this selected value.
# Override: RA_MIN_VF. Named-event writes remain distinct from continuous flow.
VF_RA_MIN: float = 0.01

# Deprecated compatibility export: one radian, approximately 57.3 degrees,
# not pi/3. Current RA readiness uses RA_MAX_PHASE_DIFF with DELTA_PHI_MAX as
# its default; the separate U3 hard phase-admission gate remains authoritative.
PHASE_RA_MAX_DIFF: float = 1.0
