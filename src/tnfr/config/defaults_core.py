"""Core constants.

AUDIT 2026: only π (the phase-wrap bound) is a genuine structural scale. The
earlier φ/γ/e "tetrahedral correspondence" overlay was an anti-magic-number
naming convention, NOT a derivation from the nodal equation; it has been
removed and the thresholds below are plain calibrated parameters.
ξ_C's scale is the spectral gap (1/√λ₂). See AGENTS.md §3 and CHANGELOG (tetrad
correspondence audit).
"""

from __future__ import annotations

from dataclasses import asdict, field
from types import MappingProxyType
from typing import Any, Mapping

from ..compat.dataclass import dataclass
from ..constants.operational import MIN_BUSINESS_COHERENCE_CANONICAL
from ..constants.canonical import (
    CHANNEL_WEIGHT_PRIMARY,
    CHANNEL_WEIGHT_SECONDARY,
    CHANNEL_WEIGHT_TERTIARY,
    COHERENCE_RETENTION,
    COUPLING_FINE,
    COUPLING_GENTLE,
    COUPLING_MODERATE,
    DISSONANCE_AMPLIFICATION,
    DOWN_CANONICAL,
    DT_CANONICAL,
    DT_MIN_CANONICAL,
    EN_MIX_FACTOR,
    EPI_MAX_CANONICAL,
    EPI_MIN_CANONICAL,
    GLYPH_SELECTOR_MARGIN_CANONICAL,
    GRAD_PHI_CANONICAL_THRESHOLD,
    HIGH_COHERENCE_THRESHOLD,
    INV_PI,
    K_PHI_CANONICAL_THRESHOLD,
    KL_MAX_CANONICAL,
    KL_MIN_CANONICAL,
    MID_COHERENCE_THRESHOLD,
    NUL_DENSIFICATION_FACTOR,
    NUL_SCALE_FACTOR,
    PI,
    SHA_VF_FACTOR,
    THOL_MIN_COLLECTIVE_COHERENCE,
    U6_STRUCTURAL_POTENTIAL_LIMIT,
    UM_COMPAT_THRESHOLD,
    UM_THETA_PUSH,
    UP_CANONICAL,
    VAL_BIFURCATION_THRESHOLD,
    VAL_MIN_COHERENCE,
    VAL_MIN_EPI,
    VAL_SCALE_FACTOR,
    VF_ADAPT_MU_CANONICAL,
    VF_MAX_CANONICAL,
    VF_MIN_CANONICAL,
)

# U6 Structural Potential Confinement Constants
# Grammar U6: Monitor Δ Φ_s < π/2 (U6 confinement bound)
STRUCTURAL_ESCAPE_THRESHOLD = (
    U6_STRUCTURAL_POTENTIAL_LIMIT  # π/2 (canonical U6 confinement bound)
)

SELECTOR_THRESHOLD_DEFAULTS: Mapping[str, float] = MappingProxyType(
    {
        "si_hi": 0.5,  # unit midpoint (Si selector upper)
        "si_lo": 0.25,  # unit quarter (Si selector lower)
        "dnfr_hi": 0.133,  # operational |ΔNFR| selector upper (pressure-magnitude scale, not coherence)
        "dnfr_lo": 0.068,  # operational |ΔNFR| selector lower (pressure-magnitude scale, not coherence)
        "accel_hi": 0.276,  # operational acceleration selector upper (∂²EPI scale, not coherence)
        "accel_lo": 0.114,  # operational acceleration selector lower (∂²EPI scale, not coherence)
    }
)


@dataclass(frozen=True, slots=True)
class CoreDefaults:
    """Default parameters for the core engine.

    The fields are exported via :data:`CORE_DEFAULTS` and may therefore appear
    unused to static analysis tools such as Vulture.
    """

    DT: float = DT_CANONICAL  # = 0.5 (stable explicit step)
    INTEGRATOR_METHOD: str = "euler"
    DT_MIN: float = DT_MIN_CANONICAL  # = 1/16 ≈ 0.0625 (minimal temporal resolution)
    EPI_MIN: float = EPI_MIN_CANONICAL  # = -1.0 (unit form bound)
    EPI_MAX: float = EPI_MAX_CANONICAL  # = 1.0 (unit form bound)
    VF_MIN: float = VF_MIN_CANONICAL  # 0.0 (canonical death state)
    VF_MAX: float = VF_MAX_CANONICAL  # = 2π ≈ 6.283 (νf max, one phase cycle)
    THETA_WRAP: bool = True
    CLIP_MODE: str = "hard"
    CLIP_SOFT_K: float = PI  # π ≈ 3.14159 (geometric steepness for smooth transitions)
    DNFR_WEIGHTS: dict[str, float] = field(
        default_factory=lambda: {
            # Coherence-band hierarchy (π-derived, exact-normalising): phase ≻ EPI ≻ νf.
            "phase": CHANNEL_WEIGHT_PRIMARY,  # π/(π+1) ≈ 0.7585 (dominant desync channel)
            "epi": CHANNEL_WEIGHT_SECONDARY,  # π/(π+1)² ≈ 0.1832 (diffusion channel)
            "vf": CHANNEL_WEIGHT_TERTIARY,  # 1/(π+1)² ≈ 0.0583 (capacity-gradient channel)
            "topo": 0.0,  # Topological weight remains zero (graph fixed during evolution)
        }
    )
    SI_WEIGHTS: dict[str, float] = field(
        default_factory=lambda: {
            # Same coherence-band hierarchy: νf-coherence ≻ phase-sync ≻ |ΔNFR|.
            "alpha": CHANNEL_WEIGHT_PRIMARY,  # π/(π+1) ≈ 0.7585 (νf-coherence)
            "beta": CHANNEL_WEIGHT_SECONDARY,  # π/(π+1)² ≈ 0.1832 (phase-sync)
            "gamma": CHANNEL_WEIGHT_TERTIARY,  # 1/(π+1)² ≈ 0.0583 (|ΔNFR|)
        }
    )
    PHASE_K_GLOBAL: float = 1.0 / (2.0 * PI * PI)  # 1/(2π²) ≈ 0.0507 (global phase coupling = local/π)
    PHASE_K_LOCAL: float = COUPLING_MODERATE  # 1/(2π) ≈ 0.159 (local phase coupling)
    PHASE_ADAPT: dict[str, Any] = field(
        default_factory=lambda: {
            "enabled": True,
            "R_hi": HIGH_COHERENCE_THRESHOLD,  # π/(π+1) ≈ 0.7585 (high-coherence gate)
            "R_lo": MID_COHERENCE_THRESHOLD,  # 2/π ≈ 0.6366 (mid-high coherence trigger)
            "disr_hi": 0.5,  # unit midpoint (dissonance upper bound)
            "disr_lo": round(1.0 / (PI + 1.0), 3),  # 1/(π+1) ≈ 0.241 (fragmentation)
            "kG_min": 1.0 / (8.0 * PI * PI),  # 1/(8π²) ≈ 0.0127 (global-coupling floor = local floor/π)
            "kG_max": COUPLING_MODERATE,  # 1/(2π) ≈ 0.159 (global-coupling ceiling)
            "kL_min": KL_MIN_CANONICAL,  # 1/(8π) ≈ 0.0398 (operational)
            "kL_max": KL_MAX_CANONICAL,  # = 1/(2π) ≈ 0.159 (coupling limit)
            "up": UP_CANONICAL,  # 1/(4π) ≈ 0.0796 (constrained increment)
            "down": DOWN_CANONICAL,  # 1/(2π) ≈ 0.159 (decrement)
        }
    )
    UM_COMPAT_THRESHOLD: float = (
        UM_COMPAT_THRESHOLD  # = π/(π+1) ≈ 0.7585 (high-coherence gate)
    )
    UM_CANDIDATE_MODE: str = "sample"
    UM_CANDIDATE_COUNT: int = 0
    GLYPH_HYSTERESIS_WINDOW: int = 7
    AL_MAX_LAG: int = 5
    EN_MAX_LAG: int = 3
    GLYPH_SELECTOR_MARGIN: float = (
        GLYPH_SELECTOR_MARGIN_CANONICAL  # ≈ 0.0418 (boundary precision)
    )
    VF_ADAPT_TAU: int = 5
    VF_ADAPT_MU: float = (
        VF_ADAPT_MU_CANONICAL  # = 0.10 (adaptation)
    )
    HZ_STR_BRIDGE: float = 1.0
    GLYPH_FACTORS: dict[str, float] = field(
        default_factory=lambda: {
            "AL_boost": COUPLING_GENTLE,  # 1/(4π) ≈ 0.0796 (gentle emission EPI increment)
            "EN_mix": EN_MIX_FACTOR,  # 1/(π+1) ≈ 0.2415 (canonical reception mixing)
            "IL_dnfr_factor": COHERENCE_RETENTION,  # π/(π+1) ≈ 0.7585 (coherence-band retention, stabiliser)
            "OZ_dnfr_factor": DISSONANCE_AMPLIFICATION,  # (π+1)/π ≈ 1.3183 (band-reciprocal amplification, destabiliser)
            "UM_theta_push": UM_THETA_PUSH,  # 1/(π+1) ≈ 0.2415 (canonical coupling phase push)
            "UM_vf_sync": COUPLING_GENTLE,  # 1/(4π) ≈ 0.0796 (gentle νf sync)
            "UM_dnfr_reduction": COUPLING_MODERATE,  # 1/(2π) ≈ 0.159 (ΔNFR reduction by coupling)
            "RA_epi_diff": COUPLING_MODERATE,  # 1/(2π) ≈ 0.159 (resonant EPI diffusion)
            "RA_vf_amplification": COUPLING_FINE,  # 1/(8π) ≈ 0.0398 (fine νf amplification)
            "RA_phase_coupling": COUPLING_GENTLE,  # 1/(4π) ≈ 0.0796 (phase alignment strengthening)
            "SHA_vf_factor": SHA_VF_FACTOR,  # 1 − 1/(4π) ≈ 0.9204 (silence νf step)
            "VAL_scale": VAL_SCALE_FACTOR,  # 1 + 1/(4π) ≈ 1.0796 (gentle νf expansion step)
            "NUL_scale": NUL_SCALE_FACTOR,  # 1 − 1/(4π) ≈ 0.9204 (gentle νf contraction step)
            # NUL ΔNFR densification = 1/λ (geometric volume ratio): contracting
            # ν_f/volume by λ=NUL_scale concentrates structural pressure by 1/λ.
            "NUL_densification_factor": NUL_DENSIFICATION_FACTOR,  # 1/λ ≈ 1.0865
            "THOL_accel": COUPLING_GENTLE,  # 1/(4π) ≈ 0.0796 (self-organisation acceleration)
            # ZHIR uses canonical transformation by default (θ → θ' based on ΔNFR);
            # the shift magnitude is 1/π, a θ-shift on the π-scaled phase sector.
            "ZHIR_theta_shift_factor": INV_PI,  # 1/π ≈ 0.318 (mutation θ-shift magnitude)
            "NAV_jitter": COUPLING_FINE,  # 1/(8π) ≈ 0.0398 (transition jitter)
            "NAV_eta": 0.5,  # unit midpoint (transition mix ratio)
            "REMESH_alpha": 0.5,  # unit midpoint (REMESH echo mix)
        }
    )
    GLYPH_THRESHOLDS: dict[str, float] = field(
        default_factory=lambda: {
            "hi": 0.5,  # unit midpoint (coherence hysteresis upper)
            "lo": 0.25,  # unit quarter (coherence hysteresis lower)
            "dnfr": 1e-3,
        }  # π-derived hi/lo (unit-coherence fractions)
    )
    NAV_RANDOM: bool = True
    NAV_STRICT: bool = False
    RANDOM_SEED: int = 0
    JITTER_CACHE_SIZE: int = 256
    OZ_NOISE_MODE: bool = False
    OZ_SIGMA: float = COUPLING_GENTLE  # 1/(4π) ≈ 0.0796 (OZ stochastic-mode noise width)
    GRAMMAR: dict[str, Any] = field(
        default_factory=lambda: {
            "window": 3,
            "avoid_repeats": ["ZHIR", "OZ", "THOL"],
            "force_dnfr": MID_COHERENCE_THRESHOLD,  # 2/π ≈ 0.6366 (force threshold)
            "force_accel": MID_COHERENCE_THRESHOLD,  # 2/π ≈ 0.6366 (acceleration force threshold)
            "fallbacks": {"ZHIR": "NAV", "OZ": "ZHIR", "THOL": "NAV"},
        }
    )
    SELECTOR_WEIGHTS: dict[str, float] = field(
        default_factory=lambda: {
            # Coherence-band hierarchy (π-derived): Si ≻ ΔNFR ≻ acceleration.
            "w_si": CHANNEL_WEIGHT_PRIMARY,  # π/(π+1) ≈ 0.7585
            "w_dnfr": CHANNEL_WEIGHT_SECONDARY,  # π/(π+1)² ≈ 0.1832
            "w_accel": CHANNEL_WEIGHT_TERTIARY,  # 1/(π+1)² ≈ 0.0583
        }
    )
    SELECTOR_THRESHOLDS: dict[str, float] = field(
        default_factory=lambda: dict(SELECTOR_THRESHOLD_DEFAULTS)
    )
    GAMMA: dict[str, Any] = field(
        default_factory=lambda: {"type": "none", "beta": 0.0, "R0": 0.0}
    )
    CALLBACKS_STRICT: bool = False
    VALIDATORS_STRICT: bool = False
    PROGRAM_TRACE_MAXLEN: int = 50
    HISTORY_MAXLEN: int = 0
    NODAL_EQUATION_CLIP_AWARE: bool = True
    NODAL_EQUATION_TOLERANCE: float = 1e-9
    # THOL (Self-organization) vibrational metabolism parameters
    THOL_METABOLIC_ENABLED: bool = True
    THOL_METABOLIC_GRADIENT_WEIGHT: float = COUPLING_MODERATE  # 1/(2π) ≈ 0.159
    THOL_METABOLIC_COMPLEXITY_WEIGHT: float = COUPLING_GENTLE  # 1/(4π) ≈ 0.0796
    THOL_BIFURCATION_THRESHOLD: float = 0.1

    # THOL network propagation and cascade parameters
    THOL_PROPAGATION_ENABLED: bool = True
    THOL_MIN_COUPLING_FOR_PROPAGATION: float = 0.5
    THOL_PROPAGATION_ATTENUATION: float = 0.7
    THOL_CASCADE_MIN_NODES: int = 3

    # THOL precondition thresholds
    THOL_MIN_EPI: float = 0.2  # Minimum EPI for bifurcation
    THOL_MIN_VF: float = 0.1  # Minimum structural frequency for reorganization
    THOL_MIN_DEGREE: int = 1  # Minimum network connectivity
    THOL_MIN_HISTORY_LENGTH: int = 3  # Minimum EPI history for acceleration computation
    THOL_ALLOW_ISOLATED: bool = False  # Require network context by default
    THOL_MIN_COLLECTIVE_COHERENCE: float = (
        THOL_MIN_COLLECTIVE_COHERENCE  # 1/(π+1) ≈ 0.2415 (canonical collective coherence)
    )

    # VAL (Expansion) precondition thresholds
    VAL_MAX_VF: float = 10.0  # Maximum structural frequency threshold
    VAL_MIN_DNFR: float = (
        1e-6  # Minimum positive ΔNFR for coherent expansion (very low to minimize breaking changes)
    )
    VAL_MIN_EPI: float = (
        VAL_MIN_EPI  # = 1/(2π) ≈ 0.159 (minimum structural base)
    )
    VAL_CHECK_NETWORK_CAPACITY: bool = False  # Optional network capacity validation
    VAL_MAX_NETWORK_SIZE: int = (
        1000  # Maximum network size if capacity checking enabled
    )

    # VAL (Expansion) metric thresholds (Issue #2724)
    VAL_BIFURCATION_THRESHOLD: float = (
        VAL_BIFURCATION_THRESHOLD  # 1/(π+1) ≈ 0.2415 (canonical bifurcation detection)
    )
    VAL_MIN_COHERENCE: float = (
        VAL_MIN_COHERENCE  # sin(π/3) = √3/2 ≈ 0.8660 (canonical harmonic coherence)
    )
    VAL_FRACTAL_RATIO_MIN: float = (
        0.5  # Minimum vf_growth/epi_growth ratio for fractality
    )
    VAL_FRACTAL_RATIO_MAX: float = (
        2.0  # max vf_growth/epi_growth ratio for fractality (tunable, dyadic)
    )


@dataclass(frozen=True, slots=True)
class RemeshDefaults:
    """Default parameters for the remeshing subsystem.

    As with :class:`CoreDefaults`, the fields are exported via
    :data:`REMESH_DEFAULTS` and may look unused to static analysers.
    """

    EPS_DNFR_STABLE: float = 1e-3
    EPS_DEPI_STABLE: float = 1e-3
    FRACTION_STABLE_REMESH: float = 0.922  # operational stable fraction (free)
    REMESH_COOLDOWN_WINDOW: int = 20
    REMESH_COOLDOWN_TS: float = 0.0
    REMESH_REQUIRE_STABILITY: bool = True
    REMESH_STABILITY_WINDOW: int = 25
    REMESH_MIN_PHASE_SYNC: float = 0.751  # operational phase-sync threshold (free)
    REMESH_MAX_GLYPH_DISR: float = 0.269  # operational max glyph disruption (free)
    REMESH_MIN_SIGMA_MAG: float = 0.536  # operational sigma-magnitude threshold (free)
    REMESH_MIN_KURAMOTO_R: float = 0.922  # operational Kuramoto-R threshold (free)
    REMESH_MIN_SI_HI_FRAC: float = 0.536  # operational SI high fraction (free)
    REMESH_LOG_EVENTS: bool = True
    REMESH_MODE: str = "knn"
    REMESH_COMMUNITY_K: int = 2
    REMESH_TAU_GLOBAL: int = 8
    REMESH_TAU_LOCAL: int = 4
    REMESH_ALPHA: float = 0.5
    REMESH_ALPHA_HARD: bool = False


_core_defaults = asdict(CoreDefaults())
_remesh_defaults = asdict(RemeshDefaults())

CORE_DEFAULTS = MappingProxyType(_core_defaults)
REMESH_DEFAULTS = MappingProxyType(_remesh_defaults)

# ============================================================================
# STRUCTURAL FIELD CONSTANTS (operational thresholds; audit 2026: not derived — only π phase-wrap is genuine)
# ============================================================================

# Structural Field Thresholds (Research Constants)
K_PHI_ASYMPTOTIC_ALPHA = 2.76  # Power-law exponent for multiscale K_φ variance
# Tetrad thresholds: alias the single canonical source (constants.canonical)
K_PHI_CURVATURE_THRESHOLD = (
    K_PHI_CANONICAL_THRESHOLD  # 0.9×π ≈ 2.8274 (90% of theoretical maximum)
)
PHASE_GRADIENT_THRESHOLD = GRAD_PHI_CANONICAL_THRESHOLD  # heuristic early-warning ≈ 0.196 (π/16; audit 2026: not derived; |∇φ| bound is π)

# Business Domain Thresholds (operational constants, not derived)
MIN_BUSINESS_COHERENCE = (
    MIN_BUSINESS_COHERENCE_CANONICAL  # = 0.75 (operational)
)
MIN_BUSINESS_SENSE_INDEX = round(
    0.700034, 3
)  # ≈ 0.700 (CALIBRATED target; an empirical value, not derived)

# Statistical Analysis Constants
STATISTICAL_SIGNIFICANCE_THRESHOLD = 0.5  # R² threshold for regression validity
EXPONENT_TOLERANCE = 0.1  # Tolerance for critical exponent classification
ISING_2D_TOLERANCE = 0.15  # Larger tolerance for 2D Ising (nu ≈ 1.0)

# Critical Exponents (Universal Scaling)
MEAN_FIELD_EXPONENT = 0.5  # Mean-field critical exponent
ISING_3D_EXPONENT = 0.63  # 3D Ising universality class
ISING_2D_EXPONENT = 1.0  # 2D Ising universality class

# Coherence Length Constants
CRITICAL_INFORMATION_DENSITY = (
    1.400014
)  # ≈ 1.400 (operational critical density)
MIN_DISTANCE_THRESHOLD = 0.01  # Numerical stability minimum distance

# Field Optimization Constants
HIGH_CORRELATION_THRESHOLD = 0.8  # Strong field duality threshold
VERY_HIGH_CORRELATION_THRESHOLD = 0.95  # Very strong field duality
MODERATE_CORRELATION_THRESHOLD = 0.5  # Moderate correlation for speedup
CHIRALITY_THRESHOLD = 1.0  # Chirality magnitude threshold
HIGH_ENERGY_THRESHOLD = 5.083204  # ≈ 5.083 (operational high energy threshold)
LOW_ENERGY_THRESHOLD = 1.0  # Low energy density threshold
COMPLEX_FIELD_THRESHOLD = 1.5  # complex-field magnitude threshold (tunable)
SYMMETRY_BREAKING_THRESHOLD = 1.0  # Symmetry breaking threshold

# Performance Scaling Constants
CORRELATION_SPEEDUP_FACTOR = 2.0  # Speedup multiplier for high correlation
CHIRALITY_MEMORY_FACTOR = 0.3  # Memory reduction factor for chirality
MAX_MEMORY_REDUCTION = 0.8  # Maximum memory reduction
MIN_ENERGY_FACTOR = 1.2  # Minimum energy computation factor
ENERGY_SCALING_FACTOR = 0.1  # Energy scaling multiplier
MAX_ENERGY_FACTOR = PI  # π ≈ 3.14159 (geometric energy computation factor)
BASELINE_FACTOR = 1.0  # Baseline performance factor
