"""Core constants."""

from __future__ import annotations

from dataclasses import asdict, field
from types import MappingProxyType
from typing import Any, Mapping

from ..compat.dataclass import dataclass
from ..constants.canonical import (
    PHI_GAMMA_NORMALIZED, GAMMA_PI_RATIO, PI_MINUS_E_OVER_PI,
    GAMMA_OVER_PI_PLUS_E, PI, GAMMA, PHI, E, U6_STRUCTURAL_POTENTIAL_LIMIT,
    SHA_VF_FACTOR, NUL_SCALE_FACTOR, VAL_MIN_EPI, VAL_BIFURCATION_THRESHOLD, VAL_MIN_COHERENCE,
    THOL_MIN_COLLECTIVE_COHERENCE, UM_COMPAT_THRESHOLD, EN_MIX_FACTOR, UM_THETA_PUSH,
    VAL_SCALE_FACTOR, NUL_DENSIFICATION_FACTOR,
    # Configuration constants (magic number replacements)
    DT_CANONICAL, DT_MIN_CANONICAL, EPI_MAX_CANONICAL, EPI_MIN_CANONICAL,
    VF_MAX_CANONICAL, VF_MIN_CANONICAL, KL_MIN_CANONICAL, KL_MAX_CANONICAL,
    UP_CANONICAL, DOWN_CANONICAL, AL_BOOST_CANONICAL, VF_ADAPT_MU_CANONICAL,
    ZHIR_VF_THRESHOLD_CANONICAL, NUL_EPI_THRESHOLD_CANONICAL, GLYPH_SELECTOR_MARGIN_CANONICAL
)

# U6 Structural Potential Confinement Constants
# Grammar U6: Monitor Δ Φ_s < φ ≈ 1.618 (golden escape threshold)
STRUCTURAL_ESCAPE_THRESHOLD = U6_STRUCTURAL_POTENTIAL_LIMIT  # φ ≈ 1.618 (canonical golden escape threshold)

SELECTOR_THRESHOLD_DEFAULTS: Mapping[str, float] = MappingProxyType(
    {
        "si_hi": round(2.0 / (PI + GAMMA), 3),  # 2/(π+γ) ≈ 0.663 (tetrahedral correspondence: dual transcendental balance)
        "si_lo": round(1.0 / (PI + GAMMA), 3),  # 1/(π+γ) ≈ 0.331 (tetrahedral correspondence: transcendental unity)
        "dnfr_hi": round(GAMMA / (PHI + E), 3),  # γ/(φ+e) ≈ 0.500 (tetrahedral correspondence: dynamic balance between harmony and exponential growth)
        "dnfr_lo": round(GAMMA / (E * PI), 3),  # γ/(e×π) ≈ 0.102 (tetrahedral correspondence: constrained dynamics beneath natural exponential-geometric product)
        "accel_hi": round(PHI / (PI + E), 3),  # φ/(π+e) ≈ 0.503 (tetrahedral correspondence: golden ratio acceleration bounded by geometric-exponential sum)
        "accel_lo": round(GAMMA / (PHI * PI), 3),  # γ/(φ×π) ≈ 0.113 (tetrahedral correspondence: Euler constant constrained by golden-geometric product)
    }
)


@dataclass(frozen=True, slots=True)
class CoreDefaults:
    """Default parameters for the core engine.

    The fields are exported via :data:`CORE_DEFAULTS` and may therefore appear
    unused to static analysis tools such as Vulture.
    """

    DT: float = DT_CANONICAL  # 1/φ ≈ 0.6180 (golden period inverse)
    INTEGRATOR_METHOD: str = "euler"
    DT_MIN: float = DT_MIN_CANONICAL  # γ/(π×e) ≈ 0.0676 (minimal temporal resolution)
    EPI_MIN: float = EPI_MIN_CANONICAL  # -1/φ ≈ -0.6180 (negative golden bound)
    EPI_MAX: float = EPI_MAX_CANONICAL  # φ/e ≈ 0.5952 (golden-exponential limit)
    VF_MIN: float = VF_MIN_CANONICAL  # 0.0 (canonical death state)
    VF_MAX: float = VF_MAX_CANONICAL  # π×φ ≈ 5.0832 (geometric-harmonic frequency)
    THETA_WRAP: bool = True
    CLIP_MODE: str = "hard"
    CLIP_SOFT_K: float = PI  # π ≈ 3.14159 (geometric steepness for smooth transitions)
    DNFR_WEIGHTS: dict[str, float] = field(
        default_factory=lambda: {
            "phase": round(PHI_GAMMA_NORMALIZED, 3),      # φ/(φ+γ) ≈ 0.737 (áurea-euleriana dominante)
            "epi": round(GAMMA_PI_RATIO, 3),             # γ/(π+γ) ≈ 0.155 (euleriana-pi estabilizadora)
            "vf": round(PI_MINUS_E_OVER_PI * (2 / 3), 3),   # ((π-e)/π) * (2/3) ≈ 0.089 (transcendental moderada)
            "topo": 0.0,                                 # Topological weight remains zero
        }
    )
    SI_WEIGHTS: dict[str, float] = field(
        default_factory=lambda: {
            "alpha": round(PHI_GAMMA_NORMALIZED, 3),     # φ/(φ+γ) ≈ 0.737 (coherencia áurea)
            "beta": round(GAMMA_PI_RATIO, 3),            # γ/(π+γ) ≈ 0.155 (estabilidad euleriana)
            "gamma": round(GAMMA / (PHI * PI), 3)  # γ/(φ×π) ≈ 0.113 (tetrahedral reorganization via Euler constant constrained by golden-geometric product)
        }
    )
    PHASE_K_GLOBAL: float = round(GAMMA_OVER_PI_PLUS_E / 2, 4)  # γ/(π+e) / 2 ≈ 0.0495 (global phase coupling canónico)
    PHASE_K_LOCAL: float = round(GAMMA_PI_RATIO, 3)            # γ/(π+γ) ≈ 0.155 (local phase coupling canónico)
    PHASE_ADAPT: dict[str, Any] = field(
        default_factory=lambda: {
            "enabled": True,
            "R_hi": round(E * PHI / (PI + E), 3),  # (e×φ)/(π+e) ≈ 0.747 (tetrahedral high R threshold via exponential-harmonic ratio)
            "R_lo": round(PI / (E + PHI + GAMMA), 3),  # π/(e+φ+γ) ≈ 0.630 (tetrahedral low R threshold)
            "disr_hi": round(PI / (PI + E), 3),  # π/(π+e) ≈ 0.536 (tetrahedral high disruption threshold)
            "disr_lo": round(1.0 / (PI + 1.0), 3),  # 1/(π+1) ≈ 0.242 (tetrahedral low disruption threshold)
            "kG_min": round(1.0 / (10 * PI * E), 4),  # 1/(10πe) ≈ 0.0117 (tetrahedral minimum phase coupling)
            "kG_max": round(GAMMA / (PI + 1.0), 3),  # γ/(π+1) ≈ 0.139 (tetrahedral maximum phase coupling)
            "kL_min": KL_MIN_CANONICAL,  # γ/(π×e×φ) ≈ 0.0418 (tetrahedral minimum coupling)
            "kL_max": KL_MAX_CANONICAL,  # γ/(π+γ) ≈ 0.1552 (transcendental coupling limit)
            "up": UP_CANONICAL,  # γ/(e×π) ≈ 0.0676 (constrained increment)
            "down": DOWN_CANONICAL,  # γ/(φ×π) ≈ 0.1136 (harmonic-geometric decrement)
        }
    )
    UM_COMPAT_THRESHOLD: float = UM_COMPAT_THRESHOLD  # φ/(φ+γ) ≈ 0.7371 (canonical golden-Euler compatibility)
    UM_CANDIDATE_MODE: str = "sample"
    UM_CANDIDATE_COUNT: int = 0
    GLYPH_HYSTERESIS_WINDOW: int = 7
    AL_MAX_LAG: int = 5
    EN_MAX_LAG: int = 3
    GLYPH_SELECTOR_MARGIN: float = GLYPH_SELECTOR_MARGIN_CANONICAL  # γ/(π×e×φ) ≈ 0.0418 (boundary precision)
    VF_ADAPT_TAU: int = 5
    VF_ADAPT_MU: float = VF_ADAPT_MU_CANONICAL  # γ/(π+e) ≈ 0.0985 (transcendental adaptation)
    HZ_STR_BRIDGE: float = 1.0
    GLYPH_FACTORS: dict[str, float] = field(
        default_factory=lambda: {
            "AL_boost": AL_BOOST_CANONICAL,  # 1/(π×e) ≈ 0.1171 (transcendental emission)
            "EN_mix": EN_MIX_FACTOR,  # 1/(π+1) ≈ 0.2413 (canonical reception mixing)
            "IL_dnfr_factor": round(PHI_GAMMA_NORMALIZED, 3),  # φ/(φ+γ) ≈ 0.737 (coherencia áurea)
            "OZ_dnfr_factor": round(NUL_DENSIFICATION_FACTOR, 3),  # φ/γ ≈ 2.803 (disonancia áurea)
            "UM_theta_push": UM_THETA_PUSH,  # 1/(π+1) ≈ 0.2413 (canonical coupling phase push)
            "UM_vf_sync": 0.10,
            "UM_dnfr_reduction": 0.15,
            "RA_epi_diff": 0.15,
            "RA_vf_amplification": 0.05,
            "RA_phase_coupling": 0.10,  # Canonical phase alignment strengthening
            "SHA_vf_factor": SHA_VF_FACTOR,  # 1 - γ/(π+e) ≈ 0.8476 (canonical silence factor)
            # Conservative scaling (1.05) prevents EPI overflow near boundaries
            # while maintaining meaningful expansion capacity. Critical threshold:
            # EPI × 1.05 = 1.0 when EPI ≈ 0.952 (vs previous threshold ≈ 0.870).
            # This preserves structural identity at boundary (EPI_MAX as identity frontier).
            "VAL_scale": VAL_SCALE_FACTOR,  # 1 + γ/(π×e) ≈ 1.0673 (canonical natural expansion rate)
            "NUL_scale": NUL_SCALE_FACTOR,  # 1 - γ/(π+e) ≈ 0.8476 (canonical contraction factor)
            # NUL canonical ΔNFR densification factor: implements structural pressure
            # concentration due to volume reduction. When V' = V × 0.85, density increases
            # by ~1.176× geometrically. Canonical value 1.35 accounts for nonlinear
            # structural effects at smaller scales, per TNFR theory.
            "NUL_densification_factor": NUL_DENSIFICATION_FACTOR,  # φ/γ ≈ 2.8025 (canonical golden densification)
            "THOL_accel": 0.10,
            # ZHIR now uses canonical transformation by default (θ → θ' based on ΔNFR)
            # To use fixed shift, explicitly set ZHIR_theta_shift in graph
            "ZHIR_theta_shift_factor": 0.3,  # Canonical transformation magnitude
            "NAV_jitter": 0.05,
            "NAV_eta": 0.5,
            "REMESH_alpha": 0.5,
        }
    )
    GLYPH_THRESHOLDS: dict[str, float] = field(
        default_factory=lambda: {"hi": round(2.0 / (PI + GAMMA), 3), "lo": round(1.0 / (PI + GAMMA), 3), "dnfr": 1e-3}  # tetrahedral hi/lo thresholds
    )
    NAV_RANDOM: bool = True
    NAV_STRICT: bool = False
    RANDOM_SEED: int = 0
    JITTER_CACHE_SIZE: int = 256
    OZ_NOISE_MODE: bool = False
    OZ_SIGMA: float = round(GAMMA / (E * PI), 3)  # γ/(e×π) ≈ 0.102 (tetrahedral dissonance sigma)
    GRAMMAR: dict[str, Any] = field(
        default_factory=lambda: {
            "window": 3,
            "avoid_repeats": ["ZHIR", "OZ", "THOL"],
            "force_dnfr": round(PI / (E + PHI + GAMMA), 3),  # π/(e+φ+γ) ≈ 0.630 (tetrahedral force threshold)
            "force_accel": round(PI / (E + PHI + GAMMA), 3),  # π/(e+φ+γ) ≈ 0.630 (tetrahedral acceleration threshold)
            "fallbacks": {"ZHIR": "NAV", "OZ": "ZHIR", "THOL": "NAV"},
        }
    )
    SELECTOR_WEIGHTS: dict[str, float] = field(
        default_factory=lambda: {"w_si": round(PI / (PI + E), 3), "w_dnfr": round(1.0 / (PI + 1.0), 3), "w_accel": round(GAMMA / (PI + 1.0), 3)}  # tetrahedral selector weights: π/(π+e), 1/(π+1), γ/(π+1)
    )
    SELECTOR_THRESHOLDS: dict[str, float] = field(
        default_factory=lambda: dict(SELECTOR_THRESHOLD_DEFAULTS)
    )
    GAMMA: dict[str, Any] = field(default_factory=lambda: {"type": "none", "beta": 0.0, "R0": 0.0})
    CALLBACKS_STRICT: bool = False
    VALIDATORS_STRICT: bool = False
    PROGRAM_TRACE_MAXLEN: int = 50
    HISTORY_MAXLEN: int = 0
    NODAL_EQUATION_CLIP_AWARE: bool = True
    NODAL_EQUATION_TOLERANCE: float = 1e-9
    # THOL (Self-organization) vibrational metabolism parameters
    THOL_METABOLIC_ENABLED: bool = True
    THOL_METABOLIC_GRADIENT_WEIGHT: float = 0.15
    THOL_METABOLIC_COMPLEXITY_WEIGHT: float = 0.10
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
    THOL_MIN_COLLECTIVE_COHERENCE: float = THOL_MIN_COLLECTIVE_COHERENCE  # 1/(π+1) ≈ 0.2413 (canonical collective coherence)

    # VAL (Expansion) precondition thresholds
    VAL_MAX_VF: float = 10.0  # Maximum structural frequency threshold
    VAL_MIN_DNFR: float = (
        1e-6  # Minimum positive ΔNFR for coherent expansion (very low to minimize breaking changes)
    )
    VAL_MIN_EPI: float = VAL_MIN_EPI  # γ/(π+γ) ≈ 0.1550 (canonical minimum structural base)
    VAL_CHECK_NETWORK_CAPACITY: bool = False  # Optional network capacity validation
    VAL_MAX_NETWORK_SIZE: int = 1000  # Maximum network size if capacity checking enabled

    # VAL (Expansion) metric thresholds (Issue #2724)  
    VAL_BIFURCATION_THRESHOLD: float = VAL_BIFURCATION_THRESHOLD  # 1/(π+1) ≈ 0.2413 (canonical bifurcation detection)
    VAL_MIN_COHERENCE: float = VAL_MIN_COHERENCE  # sin(π/3) = √3/2 ≈ 0.8660 (canonical harmonic coherence)
    VAL_FRACTAL_RATIO_MIN: float = 0.5  # Minimum vf_growth/epi_growth ratio for fractality
    VAL_FRACTAL_RATIO_MAX: float = PHI  # φ ≈ 1.618 (golden ratio for fractal bounds)


@dataclass(frozen=True, slots=True)
class RemeshDefaults:
    """Default parameters for the remeshing subsystem.

    As with :class:`CoreDefaults`, the fields are exported via
    :data:`REMESH_DEFAULTS` and may look unused to static analysers.
    """

    EPS_DNFR_STABLE: float = 1e-3
    EPS_DEPI_STABLE: float = 1e-3
    FRACTION_STABLE_REMESH: float = round(4.0 / (E + PHI), 3)  # 4/(e+φ) ≈ 0.798 (tetrahedral stable fraction)
    REMESH_COOLDOWN_WINDOW: int = 20
    REMESH_COOLDOWN_TS: float = 0.0
    REMESH_REQUIRE_STABILITY: bool = True
    REMESH_STABILITY_WINDOW: int = 25
    REMESH_MIN_PHASE_SYNC: float = round((E * PHI) / (PI + E), 3)  # (e×φ)/(π+e) ≈ 0.747 (tetrahedral phase sync threshold via exponential-harmonic ratio)
    REMESH_MAX_GLYPH_DISR: float = round(1.0 / (PI + GAMMA), 3)  # 1/(π+γ) ≈ 0.269 → maximum glyph disruption via tetrahedral correspondence
    REMESH_MIN_SIGMA_MAG: float = round(PI / (PI + E), 3)  # π/(π+e) ≈ 0.536 (tetrahedral sigma magnitude threshold)
    REMESH_MIN_KURAMOTO_R: float = round(4.0 / (E + PHI), 3)  # 4/(e+φ) ≈ 0.798 (tetrahedral Kuramoto threshold)
    REMESH_MIN_SI_HI_FRAC: float = round(PI / (PI + E), 3)  # π/(π+e) ≈ 0.536 (tetrahedral SI high fraction)
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
# STRUCTURAL FIELD CONSTANTS (Universal Tetrahedral Correspondence)
# ============================================================================

# Structural Field Thresholds (Research Constants)
K_PHI_ASYMPTOTIC_ALPHA = 2.76  # Power-law exponent for multiscale K_φ variance
K_PHI_CURVATURE_THRESHOLD = PI * 0.9  # 0.9×π ≈ 2.827 (90% of theoretical maximum)
PHASE_GRADIENT_THRESHOLD = GAMMA / 2  # γ/2 ≈ 0.2886 (nodal dynamics phase stress threshold)

# Business Domain Thresholds (Tetrahedral Correspondence)
MIN_BUSINESS_COHERENCE = E * PHI / (PI + E)  # (e×φ)/(π+e) ≈ 0.7506 (exponential-golden vs geometric-exponential balance)
MIN_BUSINESS_SENSE_INDEX = round(1 / PHI + 0.082, 3)  # 1/φ + calibration ≈ 0.700 (golden ratio foundation + empirical adjustment)

# Statistical Analysis Constants
STATISTICAL_SIGNIFICANCE_THRESHOLD = 0.5  # R² threshold for regression validity
EXPONENT_TOLERANCE = 0.1  # Tolerance for critical exponent classification
ISING_2D_TOLERANCE = 0.15  # Larger tolerance for 2D Ising (nu ≈ 1.0)

# Critical Exponents (Universal Scaling)
MEAN_FIELD_EXPONENT = 0.5  # Mean-field critical exponent
ISING_3D_EXPONENT = 0.63  # 3D Ising universality class
ISING_2D_EXPONENT = 1.0  # 2D Ising universality class

# Coherence Length Constants
CRITICAL_INFORMATION_DENSITY = E * PHI / PI  # (e×φ)/π ≈ 2.015 (tetrahedral critical density)
MIN_DISTANCE_THRESHOLD = 0.01  # Numerical stability minimum distance

# Field Optimization Constants
HIGH_CORRELATION_THRESHOLD = 0.8  # Strong field duality threshold
VERY_HIGH_CORRELATION_THRESHOLD = 0.95  # Very strong field duality
MODERATE_CORRELATION_THRESHOLD = 0.5  # Moderate correlation for speedup
CHIRALITY_THRESHOLD = 1.0  # Chirality magnitude threshold
HIGH_ENERGY_THRESHOLD = PHI * PI  # φ×π ≈ 5.083 (tetrahedral high energy threshold)
LOW_ENERGY_THRESHOLD = 1.0  # Low energy density threshold
COMPLEX_FIELD_THRESHOLD = PHI  # φ ≈ 1.618 (golden ratio threshold for complex fields)
SYMMETRY_BREAKING_THRESHOLD = 1.0  # Symmetry breaking threshold

# Performance Scaling Constants
CORRELATION_SPEEDUP_FACTOR = 2.0  # Speedup multiplier for high correlation
CHIRALITY_MEMORY_FACTOR = 0.3  # Memory reduction factor for chirality
MAX_MEMORY_REDUCTION = 0.8  # Maximum memory reduction
MIN_ENERGY_FACTOR = 1.2  # Minimum energy computation factor
ENERGY_SCALING_FACTOR = 0.1  # Energy scaling multiplier
MAX_ENERGY_FACTOR = PI  # π ≈ 3.14159 (geometric energy computation factor)
BASELINE_FACTOR = 1.0  # Baseline performance factor
