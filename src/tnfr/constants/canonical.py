#!/usr/bin/env python3
"""
TNFR Canonical Constants
========================

Single source of truth for TNFR constants. Nodal equation:

∂EPI/∂t = νf · ΔNFR(t)

Only π is a genuine structural scale: the phase-wrap bound shared by the two
phase derivatives (|∇φ| ≤ π, |K_φ| < 0.9·π) — the whole phase sector — with
K_φ = L_rw·φ the central operator applied to phase. The coherence length is
set by the spectral gap (ξ_C ∝ 1/√λ₂). The Φ_s confinement bound is the
π-derived ``U6_STRUCTURAL_POTENTIAL_LIMIT`` = π/2.

Every other value here is one of:
- a structural quantity derived from the nodal equation, the spectral gap,
  or a π-fraction;
- a free simulation / operator parameter — a clean structural default (unit,
  π-fraction, or plain value) labeled tunable, NOT a derived constant.

Operational engine-tuning knobs (cache sizes, FFT / optimization tuning,
performance / memory estimates, engine scoring weights) carry no nodal-physics
meaning and live in the separate module ``tnfr.constants.operational``
(audit 2026) — they must never be cited as first-principles TNFR results.

The obsolete constants φ (golden ratio), γ (Euler–Mascheroni), and e (Napier)
are intentionally ABSENT: they are not structural scales, and nothing here is
derived from them (audit 2026). Everything emerges from the nodal dynamics;
only π is assumed as a genuine structural scale (ℝ is the assumed continuum
substrate).

Author: TNFR Research Team
Date: November 29, 2025 (φ/γ/e purge 2026-06)
"""

import math

import mpmath as mp

# set high precision for canonical derivations
mp.mp.dps = 35

# ============================================================================
# FUNDAMENTAL TNFR CONSTANTS (Canonical - Never Change)
# ============================================================================

# Genuine structural scale: π (the phase-wrap bound shared by |∇φ| and K_φ).
# φ, γ, e are NOT structural scales and are intentionally absent — only π is
# genuine; every other value emerges from the nodal dynamics (audit 2026).
PI = float(mp.pi)  # Pi π ≈ 3.141592653589793
LN_2 = float(
    mp.log(2)
)  # Natural log of 2 ≈ 0.693147180559945 (binary information unit)

# Inverse constants
INV_PI = 1.0 / PI  # 1/π ≈ 0.318309886183791

# ============================================================================
# CIRCULAR (π) HELPER
# ============================================================================

# Semi-inverse circular constant (π-only)
HALF_INV_PI = 1.0 / (2.0 * PI)  # 1/(2π) ≈ 0.159 (semi-inverse circular)

# ============================================================================
# TNFR STRUCTURAL CONSTANTS (only the π phase-wrap bounds and the spectral-gap
# ξ_C are genuine structural scales — see header)
# ============================================================================

# Heuristic regime-classification / spectral-shift scale (audit 2026: NOT a
# derived critical exponent; the measured phase-transition exponent is
# protocol-dependent). A small π-fraction, labeled tunable.
CRITICAL_EXPONENT = PI / 16  # π/16 ≈ 0.196 (heuristic regime/shift scale, tunable)

# Coherence band — the single structural quantity 1/(π+1) and its complement.
# π is the sole structural scale; the fragmentation / high-coherence band is
# [1/(π+1), π/(π+1)] ≈ [0.2415, 0.7585]. Constants representing "fragmentation
# risk" or a "high-coherence gate" reference these two (single source of truth).
FRAGMENTATION_THRESHOLD = 1.0 / (PI + 1.0)  # 1/(π+1) ≈ 0.2415 (fragmentation-risk level)
HIGH_COHERENCE_THRESHOLD = PI / (PI + 1.0)  # π/(π+1) ≈ 0.7585 (high-coherence gate)

# Channel-mixing weights — the coherence-band HIERARCHY (no magic numbers: the
# single π-derived quantity 1/(π+1) and its complement π/(π+1), applied
# recursively). Order the structurally-active channels by primacy and give each
# the high-coherence share π/(π+1) of what remains. The geometric series
# normalises EXACTLY:  π/(π+1) + π/(π+1)² + 1/(π+1)² = (π+1)²/(π+1)² = 1.
# Used by DNFR_WEIGHTS (phase ≻ EPI ≻ νf; topo inactive — graph fixed) and
# SI_WEIGHTS (νf-coherence ≻ phase-sync ≻ |ΔNFR|). Replaces the frozen φ/γ decimals.
CHANNEL_WEIGHT_PRIMARY = HIGH_COHERENCE_THRESHOLD  # π/(π+1) ≈ 0.7585 (dominant channel)
CHANNEL_WEIGHT_SECONDARY = (
    HIGH_COHERENCE_THRESHOLD * FRAGMENTATION_THRESHOLD
)  # π/(π+1)² ≈ 0.1832 (dominant-of-remainder)
CHANNEL_WEIGHT_TERTIARY = (
    FRAGMENTATION_THRESHOLD * FRAGMENTATION_THRESHOLD
)  # 1/(π+1)² ≈ 0.0583 (remainder)

# Pressure-lever (ΔNFR) operator gains — the coherence-band step and its
# reciprocal. IL (stabiliser) retains the high-coherence share π/(π+1); OZ
# (destabiliser) amplifies by (π+1)/π, so a balanced IL∘OZ is EXACTLY isometric
# (π/(π+1)·(π+1)/π = 1). π-derived; replaces the frozen φ/γ (IL=φ/(φ+γ),
# OZ=φ/γ) and the bare 0.75 / 2.0 operational values.
COHERENCE_RETENTION = HIGH_COHERENCE_THRESHOLD  # π/(π+1) ≈ 0.7585 (IL pressure retention)
DISSONANCE_AMPLIFICATION = (PI + 1.0) / PI  # (π+1)/π ≈ 1.3183 (OZ pressure amplification)

# Secondary operator couplings — gentle π-fraction gains at three structural
# scales. Each operator contract fixes its channel and sign; these secondary
# magnitudes are π-fractions (NO magic decimals), on the same π-fraction ladder
# as the adaptation rates UP/DOWN (1/(4π), 1/(2π)) and the coupling floor 1/(8π).
COUPLING_GENTLE = 1.0 / (4.0 * PI)  # 1/(4π) ≈ 0.0796 (gentle secondary gain)
COUPLING_MODERATE = 1.0 / (2.0 * PI)  # 1/(2π) ≈ 0.159 (moderate secondary gain)
COUPLING_FINE = 1.0 / (8.0 * PI)  # 1/(8π) ≈ 0.0398 (fine secondary gain)

# Rectified-mean coherence level 2/π — the natural π-derived threshold between
# the unit midpoint 0.5 and the high-coherence gate π/(π+1), used as the
# mid-high coherence/force trigger (Kuramoto R lower bound, grammar force level).
MID_COHERENCE_THRESHOLD = 2.0 / PI  # 2/π ≈ 0.6366 (mid-high coherence trigger)

# Operator gain parameters. The scale operators (Silence, Expansion,
# Contraction) act on the ν_f CAPACITY lever; the contracts fix the DIRECTION
# (ν_f↓ for SHA/NUL, ν_f↑ for VAL). The MAGNITUDE is the gentle π-derived
# adaptation step δ = 1/(4π) ≈ 0.0796 (the capacity lever evolves slowly, so the
# step is small — unlike the PRESSURE lever IL/OZ, which take the coherence-band
# ratio). δ = 1/(4π) is the same π-fraction as UP_CANONICAL below.
_CAPACITY_STEP = 1.0 / (4.0 * PI)  # 1/(4π) ≈ 0.0796 (gentle ν_f adaptation step)
SHA_VF_FACTOR = 1.0 - _CAPACITY_STEP  # 1 − 1/(4π) ≈ 0.9204 (gentle freeze step)
NUL_SCALE_FACTOR = (
    SHA_VF_FACTOR  # Contraction: same ν_f↓ step as Silence
)

# VAL expansion thresholds
VAL_MIN_EPI = 1.0 / (2.0 * PI)  # minimum structural base to expand (tunable)
VAL_BIFURCATION_THRESHOLD = FRAGMENTATION_THRESHOLD  # 1/(π+1) ≈ 0.2415 (bifurcation detection)
VAL_MIN_COHERENCE = math.sin(
    PI / 3
)  # sin(π/3) = √3/2 ≈ 0.8660 (60° harmonic coherence)

# THOL self-organization thresholds (operational)
THOL_MIN_COLLECTIVE_COHERENCE = (
    FRAGMENTATION_THRESHOLD  # 1/(π+1) ≈ 0.2415 (same physics as VAL bifurcation)
)

# Coupling and mixing thresholds (operational)
# Coupling forms when the composite compatibility (phase 50% + EPI 25% + Si 25%)
# exceeds the HIGH-COHERENCE level π/(π+1), the complement of the fragmentation
# threshold 1/(π+1). The coherence band [1/(π+1), π/(π+1)] is derived from the
# single structural quantity 1/(π+1) (π is the sole structural scale).
UM_COMPAT_THRESHOLD = HIGH_COHERENCE_THRESHOLD  # π/(π+1) ≈ 0.7585 (high-coherence gate)
EN_MIX_FACTOR = FRAGMENTATION_THRESHOLD  # 1/(π+1) ≈ 0.2415 (reception mixing fraction)
UM_THETA_PUSH = EN_MIX_FACTOR  # Same physics as EN mixing (coupling phase push)

# Expansion raises the ν_f capacity by the same gentle π-step δ = 1/(4π).
# Contraction (NUL) concentrates ΔNFR by the GEOMETRIC volume ratio 1/λ, where
# λ = the ν_f contraction factor: contracting volume by λ<1 densifies structural
# pressure by 1/λ>1 (the NUL contract "ν_f↓ and ΔNFR densifies"). Densification
# is thus DERIVED from the contraction factor, not a free magnitude.
VAL_SCALE_FACTOR = 1.0 + _CAPACITY_STEP  # 1 + 1/(4π) ≈ 1.0796 (gentle expansion step)
NUL_DENSIFICATION_FACTOR = (
    1.0 / NUL_SCALE_FACTOR  # 1/λ ≈ 1.0865 (geometric volume-ratio densification)
)

# Binary/structural escape threshold = 2.0. This is the EPI dynamic range
# (EPI_MAX − EPI_MIN = 1 − (−1) = 2, the maximum coherent-form span) — a plain
# structural value, NOT a transcendental (de-obfuscated from exp(ln 2), audit 2026).
STRUCTURAL_ESCAPE_THRESHOLD_THEORETICAL = 2.0  # EPI span (unit form range)


# ============================================================================
# CONFIGURATION CONSTANTS (Magic Number Replacements)
# ============================================================================

# Temporal constants. The integration timestep is a numerical PARAMETER (the
# nodal equation fixes the dynamics, not the discretisation); kept stable via
# νf·dt·λ_max < 2 for the EPI diffusion channel. Tunable.
DT_CANONICAL = 1.0 / 2.0  # stable explicit step (tunable parameter)
DT_MIN_CANONICAL = 1.0 / 16.0  # minimal adaptive-step floor (tunable)

# EPI bounds: the coherent form magnitude is bounded by the unit form scale.
EPI_MAX_CANONICAL = 1.0  # unit form-magnitude bound
EPI_MIN_CANONICAL = -1.0  # unit form-magnitude bound (symmetric)

# Frequency bounds. Maximum reorganisation rate = one full phase cycle per
# unit time (2π); νf = 0 is the death state (no reorganisation).
VF_MAX_CANONICAL = 2.0 * PI  # 2π: one phase cycle per unit time
VF_MIN_CANONICAL = 0.0  # Zero remains canonical (death state)

# Coupling bounds: local coupling strength as π-fractions (tunable).
KL_MIN_CANONICAL = 1.0 / (8.0 * PI)  # minimum local coupling (tunable)
KL_MAX_CANONICAL = 1.0 / (2.0 * PI)  # maximum local coupling (tunable)

# Adaptation rates (tunable parameters; π-fractions).
UP_CANONICAL = 1.0 / (4.0 * PI)  # increment rate (tunable)
DOWN_CANONICAL = 1.0 / (2.0 * PI)  # decrement rate (tunable)

# Operator gain parameters (free magnitudes; contracts fix channel+sign).
AL_BOOST_CANONICAL = 0.10  # emission EPI increment (tunable; energy-neutral)
VF_ADAPT_MU_CANONICAL = 0.10  # νf adaptation rate (tunable)

# Bifurcation thresholds (tunable parameters; midpoint of the unit range).
ZHIR_VF_THRESHOLD_CANONICAL = 0.5  # mutation viability νf threshold (tunable)
NUL_EPI_THRESHOLD_CANONICAL = 0.5  # contraction safety EPI threshold (tunable)

# Margin and selector constants (canonical selection boundaries)
GLYPH_SELECTOR_MARGIN_CANONICAL = KL_MIN_CANONICAL  # = 1/(8π) (selection boundary precision)

# ============================================================================
# TOPOLOGY AND SPECTRAL CONSTANTS (Phase 3 Canonicalization)
# ============================================================================

# Topological factor bounds for k_top = 1/λ₁ (Fiedler-inverse relaxation scale).
# k_top itself is spectral (emergent); these are operational clamp rails.
K_TOP_MIN_CANONICAL = 1.0 / (8.0 * PI)  # 1/(8π) ≈ 0.040 (fast-relaxation floor)
K_TOP_MAX_CANONICAL = 1.0  # unit ceiling (star-graph normalization reference)
K_TOP_FALLBACK_CANONICAL = K_TOP_MAX_CANONICAL  # = 1.0 (disconnected graph → max-relaxation ceiling, consistent with the clamp)

THERAPEUTIC_EXCELLENT_CANONICAL = math.sin(
    PI / 3
)  # sin(π/3) ≈ 0.8660 (excellent therapeutic result)

# ============================================================================
# PHASE 4: STRUCTURAL OPERATORS CONSTANTS (Canonical Operator Replacements)
# ============================================================================

# Cycle Detection Balance target (structural: 1/(π+1))
CYCLE_OPTIMAL_BALANCE_CANONICAL = FRAGMENTATION_THRESHOLD  # 1/(π+1) ≈ 0.2415 (balance)
# Cycle rails, pattern weights, and algebra tolerances (operational knobs) →
# moved to tnfr.constants.operational (audit 2026).

# ============================================================================
# PHASE 5: MATHEMATICS AND PHYSICS CONSTANTS (Advanced Systems Canonicalization)
# ============================================================================

# Prime detection: prime ⟺ ΔNFR = 0 EXACTLY (canonical unit coefficients
# ζ=η=θ=1), and every composite has ΔNFR > 1 (the η·(τ−2) ≥ 1 term plus a
# positive σ-pressure). Any cut in (0, 1) separates them; 0.5 is the unit-gap
# midpoint — a structural separator, not a fitted value.
MATH_DELTA_NFR_THRESHOLD_CANONICAL = 0.5  # prime ⟺ ΔNFR = 0 separator (gap midpoint)
MATH_DELTA_NFR_THRESHOLD_2X_CANONICAL = (
    2.0 * MATH_DELTA_NFR_THRESHOLD_CANONICAL
)  # 1.0 (looser emergent-pattern band; composites still excluded, all > 1)

# PHYSICS_* network-study calibration knobs (operational) →
# moved to tnfr.constants.operational (audit 2026).

# Dynamics/Adelic Constants (temporal evolution and resonance)
DYNAMICS_ADELIC_DRIFT_CANONICAL = 0.1  # adelic drift (tunable)
DYNAMICS_ADELIC_DT_STEP_CANONICAL = 1.0 / 16.0  # adelic timestep = DT_MIN (tunable)

# Dynamics/Adaptation Constants (adaptive structural evolution)
DYNAMICS_SI_HI_THRESHOLD_CANONICAL = HIGH_COHERENCE_THRESHOLD  # π/(π+1) high-coherence Si gate

# ============================================================================
# PHASE 6: DYNAMICS MODULE CONSTANTS (Comprehensive Canonicalization)
# ============================================================================

# Integrators Constants (RK4 and numerical computation)
INTEGRATORS_RK4_SIXTH_CANONICAL = 6.0  # 6.0 (RK4 divisor: dt/6)
INTEGRATORS_HALF_STEP_CANONICAL = 2.0  # 2.0 (half-step divisor: dt/2)
INTEGRATORS_EPI_MARGIN_CANONICAL = 0.1  # EPI clipping margin fraction (tunable)
INTEGRATORS_DNFR_BOUNDS_CANONICAL = 2.0  # 2.0 (ΔNFR bounds: max(-2,min(2,x)))
INTEGRATORS_CLIP_SOFT_K_CANONICAL = PI  # π ≈ 3.1416 (soft clipping K parameter)
INTEGRATORS_J_PHI_SCALE_CANONICAL = 0.1  # J_φ flux scaling factor (tunable)
INTEGRATORS_SYNTHETIC_DIV_CANONICAL = -0.15  # conservation coefficient (tunable)
INTEGRATORS_FLUX_FALLBACK_CANONICAL = 0.75  # neighbor flux fallback fraction (tunable)
INTEGRATORS_SIGMOID_OFFSET_CANONICAL = 0.5  # sigmoid coupling offset (unit midpoint)



# REMESH Operator Constants (structural memory)
REMESH_SIMILARITY_THRESHOLD_CANONICAL = (
    UM_COMPAT_THRESHOLD  # π/(π+1) ≈ 0.7585 (structural similarity = high-coherence gate)
)


# Optimization-orchestrator, multi-modal-cache, and FFT-arithmetic knobs
# (operational) → moved to tnfr.constants.operational (audit 2026).

# Structural Feedback Loop Constants (homeostatic regulation)
# These replace inline magic numbers in dynamics/feedback.py. The control
# tolerances and rates are π-fractions on the coupling ladder (1/(2π), 1/(4π),
# 1/(8π)); the target coherence is the high-coherence gate π/(π+1).
FEEDBACK_COHERENCE_TOL_LOW = COUPLING_MODERATE  # 1/(2π) ≈ 0.159 (low coherence tolerance)
FEEDBACK_COHERENCE_TOL_HIGH = COUPLING_GENTLE  # 1/(4π) ≈ 0.0796 (high coherence tolerance)
FEEDBACK_DNFR_THRESHOLD = math.sqrt(
    FEEDBACK_COHERENCE_TOL_LOW * FEEDBACK_COHERENCE_TOL_HIGH
)  # √(1/(2π)·1/(4π)) = 1/(2π√2) ≈ 0.1125
FEEDBACK_EPI_THRESHOLD = 1.0 / 3.0  # 1/3 EPI threshold (tunable)
FEEDBACK_TARGET_COHERENCE = UM_COMPAT_THRESHOLD  # π/(π+1) ≈ 0.7585 (target coherence)
FEEDBACK_TAU_ADAPTIVE = COUPLING_MODERATE  # 1/(2π) ≈ 0.159 (adaptive-tau π-fraction)
FEEDBACK_LEARNING_RATE = COUPLING_FINE  # 1/(8π) ≈ 0.0398 (feedback-loop gain, fine π-fraction)

# FFT, pattern-discovery, self-optimization, emergent-centralization, cache,
# unified-cache, integration, nodal-optimizer, and structural-cache knobs
# (operational, ~110 constants) → moved to tnfr.constants.operational (audit 2026).

# ============================================================================
# TELEMETRY CONSTANTS (Classical Mathematical Derivations)
# ============================================================================

# --- Canonical Structural Field Tetrad Thresholds ---
# Φ_s and K_φ are π-derived (phase-wrap fractions); |∇φ| is a HEURISTIC
# early-warning (the genuine |∇φ| bound is the π wrap, like K_φ); ξ_C is set by
# the spectral gap (ξ_C ∝ 1/√λ₂). Only π is a genuine structural scale.

# Φ_s: Structural Potential Field. The per-node confinement bound is π/4
# (quarter phase-wrap), consistent with the U6 drift bound π/2 (half phase-wrap):
# the phase sector (scaled by the sole structural constant π) confines Φ_s.
# The name retains VON_KOCH for code-compat.
PHI_S_VON_KOCH_THRESHOLD: float = PI / 4  # π/4 ≈ 0.7854 (quarter phase-wrap)

# |∇φ|: Phase Gradient Field — HEURISTIC early-warning level (audit 2026:
# NOT a derived bound; the kinematic bound is |∇φ| ≤ π (wrap), the SAME as K_φ;
# the sync-onset is σ-dependent ≈ 0.29, not a fixed constant).
GRAD_PHI_CANONICAL_THRESHOLD = PI / 16  # π/16 ≈ 0.196 (heuristic |∇φ| early-warning, tunable)

# |K_φ|: Phase Curvature Field — 0.9×π from wrap_angle bounds (90% of π maximum)
K_PHI_CANONICAL_THRESHOLD = 0.9 * PI  # 0.9×π ≈ 2.8274

# ξ_C: Coherence Length Field (critical phenomena + RG)
XI_C_CRITICAL_RATIO = 1.0  # 1.0 × diameter (finite-size scaling)
XI_C_WATCH_RATIO = PI  # π × mean_distance (RG scaling)

# ============================================================================
# PHASE AND RESONANCE CONSTANTS
# ============================================================================

# Phase coupling thresholds
DELTA_PHI_MAX = PI / 2  # π/2 ≈ 1.5708 rad (90° maximum phase mismatch for U3 coupling)
PHASE_SYNC_THRESHOLD = math.sin(PI / 6)  # sin(π/6) = 0.5 (30° tolerance)
PHASE_DESYNC_LIMIT = math.cos(PI / 3)  # cos(π/3) = 0.5 (60° limit)
ANTIPHASE_THRESHOLD = math.cos(2 * PI / 3)  # cos(2π/3) ≈ -0.5 (120° destructive)

# NOTE: the νf (structural-frequency) bounds are VF_MIN_CANONICAL = 0 and
# VF_MAX_CANONICAL = 2π (defined above). The former MIN/MAX_STRUCTURAL_FREQUENCY
# duplicated those (and were misused as a ΔNFR threshold) — removed (audit 2026).

# ============================================================================
# VALIDATION AND SAFETY CONSTANTS
# ============================================================================

# Grammar validation
U6_STRUCTURAL_POTENTIAL_LIMIT = PI / 2  # U6: ΔΦ_s < π/2 (half phase-wrap confinement bound)
GRAMMAR_TOLERANCE = 1e-10  # Numerical precision for grammar checks
PHASE_VERIFICATION_TOLERANCE = PI / 180  # 1° tolerance for phase coupling

# Convergence criteria
INTEGRAL_CONVERGENCE_TOLERANCE = 1e-8  # For ∫νf·ΔNFR convergence
BIFURCATION_DETECTION_SENSITIVITY = 1e-6  # ∂²EPI/∂t² threshold detection
COHERENCE_PRESERVATION_MINIMUM = 0.1  # Minimum C(t) for system stability

# Emergent-centralization, FFT-coordination, and cache-aware FFT knobs
# (operational) → moved to tnfr.constants.operational (audit 2026).

# ============================================================================
# PHASE 7: Physics Module Canonicalization (Fields & Interactions)
# ============================================================================

# Physics interactions thresholds
# Phase-gradient early-warning level (audit 2026: HEURISTIC, not derived —
# the kinematic |∇φ| bound is π (wrap); the sync-onset is σ-dependent).
PHASE_GRADIENT_THRESHOLD_CANONICAL = (
    GRAD_PHI_CANONICAL_THRESHOLD  # π/16 ≈ 0.196 heuristic |∇φ| early-warning (alias)
)
# Legacy alias for backward compatibility
PHYSICS_GRAD_THRESHOLD_CANONICAL = PHASE_GRADIENT_THRESHOLD_CANONICAL
PHYSICS_CURVATURE_HOTSPOT_CANONICAL = (
    K_PHI_CANONICAL_THRESHOLD  # 0.9×π ≈ 2.8274 (alias)
)
# Au-like permissive curvature threshold: the exact MIDPOINT between the strict
# K_φ gate (0.9·π) and the π phase-wrap maximum (|K_φ| ≤ π): (0.9π + π)/2 =
# 0.95π — a high-permissive confinement check, DERIVED from the two π-bounds
# (not a free 0.95 fraction).
AU_CURVATURE_PERMISSIVE_THRESHOLD = (K_PHI_CANONICAL_THRESHOLD + PI) / 2.0  # (0.9π+π)/2 = 0.95π ≈ 2.985 (permissive |K_φ|)
PHYSICS_HOTSPOT_FRACTION_CANONICAL = 0.1  # curvature-hotspot fraction warning (tunable)

# Operator pattern-scoring and domain-suitability weights (operational) →
# moved to tnfr.constants.operational (audit 2026).

# ============================================================================
# PHASE 7C: Mathematics Module Canonicalization
# ============================================================================

# Mathematics operators canonical constants
MATH_COHERENCE_MIN_CANONICAL = 0.1  # minimum coherence floor (tunable)
MATH_TOLERANCE_CANONICAL = 1.0e4  # numerical tolerance floor (operational)
MATH_PRECISION_ENHANCEMENT_CANONICAL = (
    MATH_TOLERANCE_CANONICAL / 100
)  # = 100.0 (machine-ε precision amplification factor)


# ============================================================================
# PHASE 7D: Configuration Module Canonicalization
# ============================================================================

# Config initialization canonical constants
CONFIG_INIT_VF_MEAN_CANONICAL = 0.5  # νf init mean (tunable)
CONFIG_INIT_VF_STD_CANONICAL = 0.15  # νf init stddev (tunable)

# Config thresholds canonical constants
CONFIG_EPI_LATENT_MAX_CANONICAL = 0.5  # AL emission max EPI for latent state (tunable)
CONFIG_VF_BASAL_CANONICAL = 0.5  # minimum νf for emission (tunable)
CONFIG_EPSILON_MIN_CANONICAL = 0.1  # minimum coherence gradient for AL (tunable)
