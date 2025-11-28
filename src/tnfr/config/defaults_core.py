"""Core constants."""

from __future__ import annotations

from dataclasses import asdict, field
from types import MappingProxyType
from typing import Any, Mapping

from ..compat.dataclass import dataclass

SELECTOR_THRESHOLD_DEFAULTS: Mapping[str, float] = MappingProxyType(
    {
        "si_hi": 0.66,
        "si_lo": 0.33,
        "dnfr_hi": 0.50,
        "dnfr_lo": 0.10,
        "accel_hi": 0.50,
        "accel_lo": 0.10,
    }
)


@dataclass(frozen=True, slots=True)
class CoreDefaults:
    """Default parameters for the core engine.

    The fields are exported via :data:`CORE_DEFAULTS` and may therefore appear
    unused to static analysis tools such as Vulture.
    """

    DT: float = 1.0
    INTEGRATOR_METHOD: str = "euler"
    DT_MIN: float = 0.1
    EPI_MIN: float = -1.0
    EPI_MAX: float = 1.0
    VF_MIN: float = 0.0
    VF_MAX: float = 10.0
    THETA_WRAP: bool = True
    CLIP_MODE: str = "hard"
    CLIP_SOFT_K: float = 3.0
    DNFR_WEIGHTS: dict[str, float] = field(
        default_factory=lambda: {
            "phase": 0.34,
            "epi": 0.33,
            "vf": 0.33,
            "topo": 0.0,
        }
    )
    SI_WEIGHTS: dict[str, float] = field(
        default_factory=lambda: {"alpha": 0.34, "beta": 0.33, "gamma": 0.33}
    )
    PHASE_K_GLOBAL: float = 0.05
    PHASE_K_LOCAL: float = 0.15
    PHASE_ADAPT: dict[str, Any] = field(
        default_factory=lambda: {
            "enabled": True,
            "R_hi": 0.90,
            "R_lo": 0.60,
            "disr_hi": 0.50,
            "disr_lo": 0.25,
            "kG_min": 0.01,
            "kG_max": 0.20,
            "kL_min": 0.05,
            "kL_max": 0.25,
            "up": 0.10,
            "down": 0.07,
        }
    )
    UM_COMPAT_THRESHOLD: float = 0.75
    UM_CANDIDATE_MODE: str = "sample"
    UM_CANDIDATE_COUNT: int = 0
    GLYPH_HYSTERESIS_WINDOW: int = 7
    AL_MAX_LAG: int = 5
    EN_MAX_LAG: int = 3
    GLYPH_SELECTOR_MARGIN: float = 0.05
    VF_ADAPT_TAU: int = 5
    VF_ADAPT_MU: float = 0.1
    HZ_STR_BRIDGE: float = 1.0
    GLYPH_FACTORS: dict[str, float] = field(
        default_factory=lambda: {
            "AL_boost": 0.05,
            "EN_mix": 0.25,
            "IL_dnfr_factor": 0.7,
            "OZ_dnfr_factor": 1.3,
            "UM_theta_push": 0.25,
            "UM_vf_sync": 0.10,
            "UM_dnfr_reduction": 0.15,
            "RA_epi_diff": 0.15,
            "RA_vf_amplification": 0.05,
            "RA_phase_coupling": 0.10,  # Canonical phase alignment strengthening
            "SHA_vf_factor": 0.85,
            # Conservative scaling (1.05) prevents EPI overflow near boundaries
            # while maintaining meaningful expansion capacity. Critical threshold:
            # EPI × 1.05 = 1.0 when EPI ≈ 0.952 (vs previous threshold ≈ 0.870).
            # This preserves structural identity at boundary (EPI_MAX as identity frontier).
            "VAL_scale": 1.05,
            "NUL_scale": 0.85,
            # NUL canonical ΔNFR densification factor: implements structural pressure
            # concentration due to volume reduction. When V' = V × 0.85, density increases
            # by ~1.176× geometrically. Canonical value 1.35 accounts for nonlinear
            # structural effects at smaller scales, per TNFR theory.
            "NUL_densification_factor": 1.35,
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
        default_factory=lambda: {"hi": 0.66, "lo": 0.33, "dnfr": 1e-3}
    )
    NAV_RANDOM: bool = True
    NAV_STRICT: bool = False
    RANDOM_SEED: int = 0
    JITTER_CACHE_SIZE: int = 256
    OZ_NOISE_MODE: bool = False
    OZ_SIGMA: float = 0.1
    GRAMMAR: dict[str, Any] = field(
        default_factory=lambda: {
            "window": 3,
            "avoid_repeats": ["ZHIR", "OZ", "THOL"],
            "force_dnfr": 0.60,
            "force_accel": 0.60,
            "fallbacks": {"ZHIR": "NAV", "OZ": "ZHIR", "THOL": "NAV"},
        }
    )
    SELECTOR_WEIGHTS: dict[str, float] = field(
        default_factory=lambda: {"w_si": 0.5, "w_dnfr": 0.3, "w_accel": 0.2}
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
    THOL_MIN_COLLECTIVE_COHERENCE: float = 0.3  # Minimum collective coherence for sub-EPI ensemble

    # VAL (Expansion) precondition thresholds
    VAL_MAX_VF: float = 10.0  # Maximum structural frequency threshold
    VAL_MIN_DNFR: float = (
        1e-6  # Minimum positive ΔNFR for coherent expansion (very low to minimize breaking changes)
    )
    VAL_MIN_EPI: float = 0.2  # Minimum EPI for coherent expansion base
    VAL_CHECK_NETWORK_CAPACITY: bool = False  # Optional network capacity validation
    VAL_MAX_NETWORK_SIZE: int = 1000  # Maximum network size if capacity checking enabled

    # VAL (Expansion) metric thresholds (Issue #2724)
    VAL_BIFURCATION_THRESHOLD: float = 0.3  # Threshold for |∂²EPI/∂t²| bifurcation detection
    VAL_MIN_COHERENCE: float = 0.5  # Minimum local coherence for healthy expansion
    VAL_FRACTAL_RATIO_MIN: float = 0.5  # Minimum vf_growth/epi_growth ratio for fractality
    VAL_FRACTAL_RATIO_MAX: float = 2.0  # Maximum vf_growth/epi_growth ratio for fractality


@dataclass(frozen=True, slots=True)
class RemeshDefaults:
    """Default parameters for the remeshing subsystem.

    As with :class:`CoreDefaults`, the fields are exported via
    :data:`REMESH_DEFAULTS` and may look unused to static analysers.
    """

    EPS_DNFR_STABLE: float = 1e-3
    EPS_DEPI_STABLE: float = 1e-3
    FRACTION_STABLE_REMESH: float = 0.80
    REMESH_COOLDOWN_WINDOW: int = 20
    REMESH_COOLDOWN_TS: float = 0.0
    REMESH_REQUIRE_STABILITY: bool = True
    REMESH_STABILITY_WINDOW: int = 25
    REMESH_MIN_PHASE_SYNC: float = 0.85
    REMESH_MAX_GLYPH_DISR: float = 0.35
    REMESH_MIN_SIGMA_MAG: float = 0.50
    REMESH_MIN_KURAMOTO_R: float = 0.80
    REMESH_MIN_SI_HI_FRAC: float = 0.50
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
