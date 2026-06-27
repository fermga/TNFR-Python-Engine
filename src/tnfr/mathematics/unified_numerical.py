"""TNFR Unified Numerical Utilities - Consolidated NumPy and Constants Module.

CONSOLIDATION ACHIEVEMENT: This module unifies all numerical operations and
constants across TNFR codebase under a single coherent interface.

Theoretical Foundation:
Grounded in the nodal equation ∂EPI/∂t = νf · ΔNFR(t). The four structural
fields (Φ_s, |∇φ|, K_φ, ξ_C) are the orders of the derivative tower. Note
(audit 2026): only π (phase wrap) is a genuine structural scale — the other
threshold parameters are an operational convention, NOT derived scales.

Unified Architecture:
- Standardized NumPy imports with consistent aliasing (np)
- Centralized mathematical constants (π and TNFR-derived values)
- Unified numerical operations with fallback mechanisms
- Consistent random number generation with seed management
- Optimized array operations for TNFR structural computations

Consolidates:
- Scattered numpy imports across 50+ modules
- Mathematical constants from config/, engines/, mathematics/ modules
- Random number generators and seed management
- Array utility functions duplicated across codebase
- Numerical precision and error handling

Status: NUMERICAL CONSOLIDATION - All numerical operations centralized
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from ..errors import TNFRValueError

# UNIFIED NUMPY IMPORT - Single point of import for entire TNFR codebase
try:
    import numpy as np
    import numpy.typing as npt

    NUMPY_AVAILABLE = True

    # Compatibility for NumPy 2.0+ vs <2.0
    # Ensure trapezoid is available (renamed from trapz in NumPy 2.0)
    if hasattr(np, "trapezoid"):
        trapezoid = np.trapezoid
    else:
        trapezoid = np.trapz

    # Standard array types for TNFR operations
    ArrayLike = np.ndarray | list[float] | tuple[float, ...]
    ComplexArray = np.ndarray | list[complex]

except ImportError:
    # Fallback for environments without NumPy
    np = None
    npt = None
    NUMPY_AVAILABLE = False

    # Fallback types
    ArrayLike = list[float] | tuple[float, ...]
    ComplexArray = list[complex] | tuple[complex, ...]

logger = logging.getLogger(__name__)

# ============================================================================
# NUMERICAL PARAMETERS — only π is a genuine structural scale
# ============================================================================


@dataclass(frozen=True)
class TNFRConstants:
    """Numerical parameters for the unified numerical utilities.

    Of the constants that appear in TNFR, only π is a genuine structural
    scale — the phase-wrap bound shared by |∇φ| and K_φ (K_φ = L_rw·φ). The
    coherence-length scale is set by the spectral gap (ξ_C ∝ 1/√λ₂). Every
    other value below is a plain operational parameter or telemetry cut, not
    a derived structural scale.
    """

    PI: float = math.pi  # π — genuine phase-wrap scale

    # Coherence telemetry cuts (heuristic; not derived from the dynamics)
    MIN_BUSINESS_COHERENCE: float = 0.75  # strong-coherence cut (operational heuristic)
    THOL_MIN_COLLECTIVE_COHERENCE: float = float(
        1.0 / (math.pi + 1)
    )  # fragmentation-risk cut C < 0.2415
    HIGH_CORRELATION_THRESHOLD: float = 0.8  # Excellent stability threshold

    # Phase and frequency bounds
    MAX_PHASE: float = 2.0 * math.pi  # Phase normalization bound
    MIN_STRUCTURAL_FREQUENCY: float = 0.0  # Hz_str minimum
    MAX_STRUCTURAL_FREQUENCY: float = 1000.0  # Hz_str practical maximum

    # Structural field bounds (audit 2026: only the π phase-wrap bounds are
    # genuine; the |∇φ| early-warning level is a heuristic, not a derived bound).
    STRUCTURAL_POTENTIAL_ESCAPE_THRESHOLD: float = 2.0  # Δ Φ_s < 2.0 (empirical)
    PHASE_GRADIENT_STABILITY_THRESHOLD: float = float(
        math.pi / 16
    )  # heuristic early-warning ≈ 0.196 (π/16; kinematic bound is |∇φ| ≤ π)
    PHASE_CURVATURE_CONFINEMENT_THRESHOLD: float = float(
        0.9 * PI
    )  # |K_φ| < 0.9×π ≈ 2.8274 (phase wrap — genuine)
    COHERENCE_LENGTH_CRITICAL_RATIO: float = (
        PI  # ξ_C scale set by spectral gap (ξ_C ∝ 1/√λ₂)
    )

    # Numerical precision constants
    FLOAT_TOLERANCE: float = 1e-12  # Numerical precision for TNFR operations
    CONVERGENCE_TOLERANCE: float = 1e-8  # Iteration convergence threshold
    STABILITY_EPSILON: float = 1e-6  # Stability margin for bifurcation detection

    # Cache and performance constants
    DEFAULT_CACHE_SIZE: int = 1000  # Default cache size for unified systems
    MAX_ARRAY_SIZE: int = int(1e8)  # Maximum array size (100M elements)
    PERFORMANCE_THRESHOLD_MS: float = 1000.0  # Performance warning threshold

    # Random seed management
    DEFAULT_SEED: int = 42  # Default reproducible seed
    SEED_RANGE_MAX: int = 2**31 - 1  # Maximum valid seed value


# Global constants instance
CONSTANTS = TNFRConstants()

# ============================================================================
# UNIFIED NUMERICAL OPERATIONS
# ============================================================================


class TNFRNumericalUtilities:
    """Unified Numerical Utilities - Consolidated Mathematical Operations.

    ARCHITECTURE: Provides unified interface for all numerical operations
    across TNFR codebase with intelligent fallbacks and optimization.

    Features:
    - Standardized array operations
    - Consistent random number generation
    - Optimized mathematical functions for TNFR
    - Automatic fallback mechanisms
    - Performance monitoring and caching

    Usage:
        # Single entry point for all numerical operations
        num = TNFRNumericalUtilities()

        # Array operations
        result = num.normalize_phase(phase_array)

        # Random generation with seed management
        random_data = num.generate_random_array(100, seed=42)

        # Mathematical operations
        coherence = num.compute_coherence_metric(data)
    """

    def __init__(self, seed: int | None = None):
        """Initialize numerical utilities."""
        self.seed = seed or CONSTANTS.DEFAULT_SEED

        # Initialize random state
        if NUMPY_AVAILABLE:
            self._rng = np.random.RandomState(self.seed)
        else:
            import random

            random.seed(self.seed)
            self._rng = None

        # Performance tracking
        self._operation_count = 0
        self._total_time = 0.0

        logger.info(f"Initialized TNFR numerical utilities with seed {self.seed}")

    def normalize_phase(self, phase: ArrayLike) -> ArrayLike:
        """Normalize phase values to [0, 2π] range.

        TNFR PHYSICS: Phase normalization preserves resonance relationships
        while ensuring bounded evolution per nodal equation constraints.

        Parameters
        ----------
        phase : array-like
            Phase values in radians to normalize

        Returns
        -------
        array-like
            Normalized phase values in [0, 2π] range
        """
        if NUMPY_AVAILABLE and isinstance(phase, np.ndarray):
            return phase % CONSTANTS.MAX_PHASE
        else:
            # Fallback for non-NumPy environments
            if hasattr(phase, "__iter__"):
                return [p % CONSTANTS.MAX_PHASE for p in phase]
            else:
                return phase % CONSTANTS.MAX_PHASE

    def compute_phase_difference(
        self, phase1: ArrayLike, phase2: ArrayLike
    ) -> ArrayLike:
        """Compute phase difference with proper wraparound.

        TNFR PHYSICS: Phase differences determine coupling compatibility
        per grammar rule U3 (RESONANT COUPLING).
        """
        if NUMPY_AVAILABLE:
            phase1 = np.asarray(phase1)
            phase2 = np.asarray(phase2)
            diff = phase1 - phase2

            # Wrap to [-π, π] range
            return np.arctan2(np.sin(diff), np.cos(diff))
        else:
            # Fallback implementation
            if hasattr(phase1, "__iter__") and hasattr(phase2, "__iter__"):
                return [
                    math.atan2(math.sin(p1 - p2), math.cos(p1 - p2))
                    for p1, p2 in zip(phase1, phase2)
                ]
            else:
                diff = phase1 - phase2
                return math.atan2(math.sin(diff), math.cos(diff))

    def generate_random_array(
        self,
        size: int | tuple[int, ...],
        distribution: str = "uniform",
        seed: int | None = None,
    ) -> ArrayLike:
        """Generate random array with reproducible seeding.

        Parameters
        ----------
        size : int or tuple
            Array size specification
        distribution : str
            Distribution type ("uniform", "normal", "exponential")
        seed : int, optional
            Override seed for this operation

        Returns
        -------
        array-like
            Random array with specified distribution
        """
        if seed is not None:
            if NUMPY_AVAILABLE:
                local_rng = np.random.RandomState(seed)
            else:
                import random

                random.seed(seed)
        else:
            local_rng = self._rng

        if NUMPY_AVAILABLE:
            if distribution == "uniform":
                return local_rng.uniform(0, 1, size)
            elif distribution == "normal":
                return local_rng.normal(0, 1, size)
            elif distribution == "exponential":
                return local_rng.exponential(1.0, size)
            else:
                raise TNFRValueError(
                    f"Unknown distribution: {distribution}",
                    context={
                        "distribution": distribution,
                        "supported": ["uniform", "normal", "exponential"],
                    },
                    suggestion="Use one of the supported distributions.",
                )
        else:
            # Fallback for non-NumPy environments
            import random

            if isinstance(size, int):
                length = size
            else:
                length = int(np.prod(size)) if NUMPY_AVAILABLE else size[0]

            if distribution == "uniform":
                return [random.uniform(0, 1) for _ in range(length)]
            elif distribution == "normal":
                return [random.gauss(0, 1) for _ in range(length)]
            elif distribution == "exponential":
                return [random.expovariate(1.0) for _ in range(length)]
            else:
                raise TNFRValueError(
                    f"Unknown distribution: {distribution}",
                    context={
                        "distribution": distribution,
                        "supported": ["uniform", "normal", "exponential"],
                    },
                    suggestion="Use one of the supported distributions.",
                )

    def safe_divide(
        self, numerator: ArrayLike, denominator: ArrayLike, fallback: float = 0.0
    ) -> ArrayLike:
        """Safe division with zero-denominator handling.

        TNFR PHYSICS: Prevents division by zero in structural calculations
        while maintaining numerical stability.
        """
        if NUMPY_AVAILABLE:
            num = np.asarray(numerator)
            den = np.asarray(denominator)

            # Use numpy's divide with where clause for safety
            return np.divide(
                num, den, out=np.full_like(num, fallback), where=(den != 0)
            )
        else:
            # Fallback implementation
            if hasattr(numerator, "__iter__") and hasattr(denominator, "__iter__"):
                return [
                    n / d if d != 0 else fallback
                    for n, d in zip(numerator, denominator)
                ]
            else:
                return numerator / denominator if denominator != 0 else fallback

    def compute_circular_mean(self, angles: ArrayLike) -> float:
        """Compute circular mean of angles.

        TNFR PHYSICS: Circular statistics preserve phase relationships
        in network synchronization computations.
        """
        if NUMPY_AVAILABLE:
            angles = np.asarray(angles)
            return np.arctan2(np.mean(np.sin(angles)), np.mean(np.cos(angles)))
        else:
            # Fallback implementation
            if not hasattr(angles, "__iter__"):
                angles = [angles]

            sin_sum = sum(math.sin(a) for a in angles)
            cos_sum = sum(math.cos(a) for a in angles)
            n = len(angles)

            return math.atan2(sin_sum / n, cos_sum / n)

    def is_finite_array(self, arr: ArrayLike) -> bool:
        """Check if array contains only finite values.

        TNFR PHYSICS: Ensures numerical stability by detecting
        NaN and infinite values that violate nodal equation constraints.
        """
        if NUMPY_AVAILABLE:
            arr = np.asarray(arr)
            return np.all(np.isfinite(arr))
        else:
            # Fallback implementation
            if hasattr(arr, "__iter__"):
                return all(math.isfinite(x) for x in arr)
            else:
                return math.isfinite(arr)

    def clamp_value(
        self, value: ArrayLike, min_val: float, max_val: float
    ) -> ArrayLike:
        """Clamp values to specified range.

        TNFR PHYSICS: Enforces structural bounds (audit 2026: only π phase-wrap is genuine)
        to prevent parameter escape beyond coherence thresholds.
        """
        if NUMPY_AVAILABLE:
            return np.clip(value, min_val, max_val)
        else:
            # Fallback implementation
            if hasattr(value, "__iter__"):
                return [max(min_val, min(max_val, v)) for v in value]
            else:
                return max(min_val, min(max_val, value))

    def kahan_sum_nd(
        self, values: Iterable[Sequence[float]], dims: int
    ) -> tuple[float, ...]:
        """Return compensated sums of ``values`` with ``dims`` components.

        TNFR PHYSICS: Essential for high-precision accumulation of structural
        metrics (ΔNFR, EPI) over long integration periods to prevent
        floating point drift in coherence calculations.
        """
        if dims < 1:
            raise TNFRValueError(
                "dims must be >= 1",
                context={"dims": dims},
                suggestion="Provide a positive integer for dimensions.",
            )
        totals = [0.0] * dims
        comps = [0.0] * dims
        for vs in values:
            for i in range(dims):
                v = vs[i]
                t = totals[i] + v
                if abs(totals[i]) >= abs(v):
                    comps[i] += (totals[i] - t) + v
                else:
                    comps[i] += (v - t) + totals[i]
                totals[i] = t
        return tuple(float(totals[i] + comps[i]) for i in range(dims))

    def get_statistics(self) -> dict[str, Any]:
        """Get numerical utilities performance statistics."""
        return {
            "numpy_available": NUMPY_AVAILABLE,
            "operation_count": self._operation_count,
            "total_time": self._total_time,
            "seed": self.seed,
            "constants_version": "structural_tetrad_v1",
        }

    def reset_seed(self, new_seed: int) -> None:
        """Reset random seed for reproducibility."""
        self.seed = new_seed

        if NUMPY_AVAILABLE:
            self._rng = np.random.RandomState(new_seed)
        else:
            import random

            random.seed(new_seed)

        logger.info(f"Reset numerical utilities seed to {new_seed}")


# ============================================================================
# GLOBAL UNIFIED NUMERICAL INTERFACE
# ============================================================================

# Global numerical utilities instance
_unified_numerical_utils: TNFRNumericalUtilities | None = None


def get_unified_numerical_utils(seed: int | None = None) -> TNFRNumericalUtilities:
    """Get or create global unified numerical utilities.

    This provides a singleton interface for all TNFR numerical operations
    to ensure consistent seeding and performance across modules.

    Parameters
    ----------
    seed : int, optional
        Random seed (only used on first call)

    Returns
    -------
    TNFRNumericalUtilities
        Global unified numerical utilities instance
    """
    global _unified_numerical_utils

    if _unified_numerical_utils is None:
        _unified_numerical_utils = TNFRNumericalUtilities(seed)
        logger.info("Created global unified numerical utilities")

    return _unified_numerical_utils


# ============================================================================
# CONVENIENCE FUNCTIONS - Direct access to unified operations
# ============================================================================


def normalize_phase(phase: ArrayLike) -> ArrayLike:
    """Normalize phase - convenience function."""
    return get_unified_numerical_utils().normalize_phase(phase)


def compute_phase_difference(phase1: ArrayLike, phase2: ArrayLike) -> ArrayLike:
    """Compute phase difference - convenience function."""
    return get_unified_numerical_utils().compute_phase_difference(phase1, phase2)


def generate_random_array(size: int | tuple[int, ...], **kwargs) -> ArrayLike:
    """Generate random array - convenience function."""
    return get_unified_numerical_utils().generate_random_array(size, **kwargs)


def safe_divide(
    numerator: ArrayLike, denominator: ArrayLike, fallback: float = 0.0
) -> ArrayLike:
    """Safe division - convenience function."""
    return get_unified_numerical_utils().safe_divide(numerator, denominator, fallback)


def compute_circular_mean(angles: ArrayLike) -> float:
    """Compute circular mean - convenience function."""
    return get_unified_numerical_utils().compute_circular_mean(angles)


def is_finite_array(arr: ArrayLike) -> bool:
    """Check finite array - convenience function."""
    return get_unified_numerical_utils().is_finite_array(arr)


def clamp_value(value: ArrayLike, min_val: float, max_val: float) -> ArrayLike:
    """Clamp value - convenience function."""
    return get_unified_numerical_utils().clamp_value(value, min_val, max_val)


def kahan_sum_nd(values: Iterable[Sequence[float]], dims: int) -> tuple[float, ...]:
    """Kahan summation - convenience function."""
    return get_unified_numerical_utils().kahan_sum_nd(values, dims)


def reset_global_seed(seed: int) -> None:
    """Reset global numerical seed - convenience function."""
    utils = get_unified_numerical_utils()
    utils.reset_seed(seed)


# ============================================================================
# LEGACY COMPATIBILITY - Gradual migration support
# ============================================================================

# Export π for backward compatibility (only genuine structural scale)
PI = CONSTANTS.PI

# Export NumPy for backward compatibility
__all__ = [
    # Core constants
    "TNFRConstants",
    "CONSTANTS",
    # Numerical utilities
    "TNFRNumericalUtilities",
    "get_unified_numerical_utils",
    # Convenience functions
    "normalize_phase",
    "compute_phase_difference",
    "generate_random_array",
    "safe_divide",
    "compute_circular_mean",
    "is_finite_array",
    "clamp_value",
    "kahan_sum_nd",
    "reset_global_seed",
    # NumPy exports
    "np",
    "npt",
    "NUMPY_AVAILABLE",
    "ArrayLike",
    "ComplexArray",
    # Legacy constant (only genuine structural scale)
    "PI",
]
