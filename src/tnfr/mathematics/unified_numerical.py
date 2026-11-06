"""Shared numerical imports, constants and small deterministic utilities.

The module supplies the package-wide NumPy alias and a narrow collection of
phase, random-generation and accumulation helpers. Domain-specific TNFR
operators and metrics remain in their respective modules. Of the parameters
exposed here, only pi is the exact phase-wrap scale; other bounds and telemetry
cuts retain their documented operational scope.
"""

from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass
from numbers import Integral
from typing import Any, Iterable, Sequence

from .._exact_time import finite_represented_real
from ..constants.canonical import (
    FRAGMENTATION_THRESHOLD,
    GRAD_PHI_CANONICAL_THRESHOLD,
    K_PHI_CANONICAL_THRESHOLD,
)
from ..constants.canonical import PI as CANONICAL_PI
from ..constants.canonical import U6_STRUCTURAL_POTENTIAL_LIMIT, XI_C_CRITICAL_RATIO
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
    scale — the phase-wrap bound shared by |∇φ| and K_φ. In a smooth,
    consistent unwrapped branch with matching normalization, K_φ has the
    corresponding Laplacian linearization; the wrapped field is nonlinear. The
    coherence-length estimator uses a state-dependent fit with `1/√λ₂` as a
    connected-graph spectral comparison/fallback. Every
    other value below is a plain operational parameter or telemetry cut, not
    a derived structural scale.
    """

    PI: float = CANONICAL_PI  # π — exact phase-wrap scale

    # Coherence telemetry cuts (heuristic; not derived from the dynamics)
    MIN_BUSINESS_COHERENCE: float = 0.75  # strong-coherence cut (operational heuristic)
    # Deprecated, inert compatibility alias. This is the fragmentation-risk
    # cut, not THOL amplitude alignment or an implicit U5 threshold.
    THOL_MIN_COLLECTIVE_COHERENCE: float = float(FRAGMENTATION_THRESHOLD)
    HIGH_CORRELATION_THRESHOLD: float = 0.8  # Selected generic correlation cut

    # Phase and frequency bounds
    MAX_PHASE: float = 2.0 * PI  # Phase normalization bound
    MIN_STRUCTURAL_FREQUENCY: float = 0.0  # Hz_str minimum
    MAX_STRUCTURAL_FREQUENCY: float = 1000.0  # Hz_str practical maximum

    # Structural field bounds (audit 2026: only the π phase-wrap bounds are
    # genuine; the |∇φ| early-warning level is a heuristic, not a derived bound).
    STRUCTURAL_POTENTIAL_ESCAPE_THRESHOLD: float = U6_STRUCTURAL_POTENTIAL_LIMIT
    PHASE_GRADIENT_STABILITY_THRESHOLD: float = GRAD_PHI_CANONICAL_THRESHOLD
    PHASE_CURVATURE_CONFINEMENT_THRESHOLD: float = K_PHI_CANONICAL_THRESHOLD
    COHERENCE_LENGTH_CRITICAL_RATIO: float = XI_C_CRITICAL_RATIO

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


def _validated_seed(seed: Any, *, label: str = "seed") -> int:
    """Return an integer RandomState seed in the supported closed interval."""
    if (
        isinstance(seed, bool)
        or not isinstance(seed, Integral)
        or not 0 <= int(seed) <= CONSTANTS.SEED_RANGE_MAX
    ):
        raise TNFRValueError(
            f"{label} must be an integer in [0, {CONSTANTS.SEED_RANGE_MAX}]",
            context={label: seed},
        )
    return int(seed)


def _validated_random_size(
    size: int | tuple[int, ...],
) -> int | tuple[int, ...]:
    """Return a nonnegative integer size accepted by both RNG backends."""
    dimensions = (size,) if isinstance(size, Integral) else size
    if (
        isinstance(size, bool)
        or not isinstance(dimensions, tuple)
        or any(
            isinstance(dimension, bool)
            or not isinstance(dimension, Integral)
            or int(dimension) < 0
            for dimension in dimensions
        )
    ):
        raise TNFRValueError(
            "size must be a nonnegative integer or tuple of nonnegative integers"
        )
    normalized = tuple(int(dimension) for dimension in dimensions)
    return normalized[0] if isinstance(size, Integral) else normalized


def _finite_real(value: Any, *, label: str) -> float:
    """Admit a represented real, retaining signed zero for circular readers."""
    try:
        resolved = finite_represented_real(value, label)[0]
    except (TypeError, ValueError, OverflowError) as exc:
        raise TNFRValueError(
            f"{label} must be a finite real representable as binary64: {exc}"
        ) from exc
    return math.copysign(0.0, value) if resolved == 0.0 else resolved


def _real_array(value: Any, *, label: str) -> Any:
    """Admit raw array elements before materializing binary64 real values."""
    if isinstance(value, (str, bytes, bytearray)):
        raise TNFRValueError(f"{label} must contain only finite real values")
    # Typed real arrays already exclude Boolean, text and complex elements.
    # Keep this common numerical path vectorized; wider floats still need
    # the shared represented-real boundary's range and nonzero-loss checks.
    if isinstance(value, np.ndarray) and value.dtype.kind in "iuf":
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            represented = np.asarray(value, dtype=float)
        if not np.all(np.isfinite(represented)):
            raise TNFRValueError(f"{label} must contain only finite real values")
        if (
            value.dtype.kind == "f"
            and value.dtype.itemsize > np.dtype(float).itemsize
            and np.any((value != 0) & (represented == 0.0))
        ):
            raise TNFRValueError(f"{label} contains nonzero values that underflow")
        return represented

    # Object materialization preserves invalid elements in mixed Python
    # sequences instead of first coercing Boolean or text entries to numbers.
    try:
        raw = np.asarray(value, dtype=object)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TNFRValueError(f"{label} must contain only finite real values") from exc
    return np.fromiter(
        (_finite_real(item, label=label) for item in raw.flat),
        dtype=float,
        count=raw.size,
    ).reshape(raw.shape)


def _normalize_phase_scalar(value: Any) -> float:
    wrapped = _finite_real(value, label="phase") % CONSTANTS.MAX_PHASE
    # A tiny negative phase can round upward to the excluded endpoint.
    return 0.0 if wrapped == CONSTANTS.MAX_PHASE else wrapped


class TNFRNumericalUtilities:
    """Small numerical utilities with reproducible instance-local randomness.

    The methods implement generic numerical operations. They do not themselves
    compute DeltaNFR, C(t), Si or any other domain-specific TNFR observable.
    """

    def __init__(self, seed: int | None = None):
        """Initialize numerical utilities."""
        selected_seed = CONSTANTS.DEFAULT_SEED if seed is None else seed
        self.seed = _validated_seed(selected_seed)

        # Keep randomness local to this utility; never mutate module-global RNGs.
        if NUMPY_AVAILABLE:
            self._rng = np.random.RandomState(self.seed)
        else:
            self._rng = random.Random(self.seed)

        logger.info(f"Initialized TNFR numerical utilities with seed {self.seed}")

    def normalize_phase(self, phase: ArrayLike) -> ArrayLike:
        """Normalize finite phase values to the half-open [0, 2π) range.

        Inputs follow represented-real admission before modulo normalization.
        If binary64 modulo rounds to the excluded upper endpoint, return zero.

        Parameters
        ----------
        phase : array-like
            Phase values in radians to normalize

        Returns
        -------
        array-like
            Normalized phase values in the half-open [0, 2π) range
        """
        if NUMPY_AVAILABLE and isinstance(phase, np.ndarray):
            wrapped = _real_array(phase, label="phase") % CONSTANTS.MAX_PHASE
            return np.where(wrapped == CONSTANTS.MAX_PHASE, 0.0, wrapped)
        if hasattr(phase, "__iter__") and not isinstance(
            phase, (str, bytes, bytearray)
        ):
            return [_normalize_phase_scalar(item) for item in phase]
        return _normalize_phase_scalar(phase)

    def compute_phase_difference(
        self, phase1: ArrayLike, phase2: ArrayLike
    ) -> ArrayLike:
        """Compute a signed atan2 phase difference without shifting by pi.

        TNFR PHYSICS: Phase differences determine coupling compatibility
        per grammar rule U3 (RESONANT COUPLING).
        Raw elements follow represented-real admission in both numerical paths.
        Signed zero inputs retain the signed atan2 branch.
        Finite phases whose subtraction overflows binary64 are rejected;
        this reader does not invent a direction from a nonfinite difference.
        The signed endpoint convention differs from a half-open modulo chart.
        Binary64 trigonometry does not promise exact invariance after adding
        represented multiples of 2*pi, particularly for large coordinates.
        """
        if NUMPY_AVAILABLE:
            first = _real_array(phase1, label="phase1")
            second = _real_array(phase2, label="phase2")
            with np.errstate(over="ignore", invalid="ignore"):
                diff = first - second
            if not np.all(np.isfinite(diff)):
                raise TNFRValueError("phase differences must be finite binary64 values")
            return np.arctan2(np.sin(diff), np.cos(diff))

        if isinstance(phase1, (str, bytes, bytearray)) or isinstance(
            phase2, (str, bytes, bytearray)
        ):
            raise TNFRValueError("phases must contain only finite real values")
        first_iterable = hasattr(phase1, "__iter__")
        second_iterable = hasattr(phase2, "__iter__")
        if first_iterable != second_iterable:
            raise TNFRValueError(
                "phase inputs must both be scalars or equally sized iterables"
            )
        if first_iterable:
            first = [_finite_real(item, label="phase1") for item in phase1]
            second = [_finite_real(item, label="phase2") for item in phase2]
            if len(first) != len(second):
                raise TNFRValueError("phase iterables must have equal length")
            differences = [a - b for a, b in zip(first, second)]
            if not all(math.isfinite(value) for value in differences):
                raise TNFRValueError("phase differences must be finite binary64 values")
            return [
                math.atan2(math.sin(value), math.cos(value)) for value in differences
            ]
        diff = _finite_real(phase1, label="phase1") - _finite_real(
            phase2, label="phase2"
        )
        if not math.isfinite(diff):
            raise TNFRValueError("phase differences must be finite binary64 values")
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
        validated_size = _validated_random_size(size)
        if seed is not None:
            selected_seed = _validated_seed(seed)
            local_rng = (
                np.random.RandomState(selected_seed)
                if NUMPY_AVAILABLE
                else random.Random(selected_seed)
            )
        else:
            local_rng = self._rng

        if NUMPY_AVAILABLE:
            if distribution == "uniform":
                return local_rng.uniform(0, 1, validated_size)
            elif distribution == "normal":
                return local_rng.normal(0, 1, validated_size)
            elif distribution == "exponential":
                return local_rng.exponential(1.0, validated_size)
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
            if isinstance(validated_size, int):
                length = validated_size
            else:
                length = math.prod(validated_size)

            if distribution == "uniform":
                return [local_rng.uniform(0, 1) for _ in range(length)]
            elif distribution == "normal":
                return [local_rng.gauss(0, 1) for _ in range(length)]
            elif distribution == "exponential":
                return [local_rng.expovariate(1.0) for _ in range(length)]
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
        """Divide admitted reals, substituting fallback only at exact zeros.

        Raw values must survive binary64 materialization. Nonfinite results
        and nonzero quotients rounded to zero reject rather than using fallback.
        """
        fallback_value = _finite_real(fallback, label="fallback")
        if NUMPY_AVAILABLE:
            num, den = np.broadcast_arrays(
                _real_array(numerator, label="numerator"),
                _real_array(denominator, label="denominator"),
            )
            with np.errstate(over="ignore", under="ignore", invalid="ignore"):
                result = np.divide(
                    num,
                    den,
                    out=np.full(num.shape, fallback_value, dtype=float),
                    where=(den != 0),
                )
            if not np.all(np.isfinite(result)) or np.any(
                (num != 0) & (den != 0) & (result == 0)
            ):
                raise TNFRValueError("division result overflows or underflows binary64")
            return result

        def divide(num, den):
            if den == 0:
                return fallback_value
            result = num / den
            if not math.isfinite(result) or (num != 0 and result == 0):
                raise TNFRValueError("division result overflows or underflows binary64")
            return result

        numerator_is_iterable = hasattr(numerator, "__iter__") and not isinstance(
            numerator, (str, bytes, bytearray)
        )
        denominator_is_iterable = hasattr(denominator, "__iter__") and not isinstance(
            denominator, (str, bytes, bytearray)
        )
        if numerator_is_iterable:
            numerators = [_finite_real(item, label="numerator") for item in numerator]
        else:
            numerator = _finite_real(numerator, label="numerator")
            numerators = None
        if denominator_is_iterable:
            denominators = [
                _finite_real(item, label="denominator") for item in denominator
            ]
        else:
            denominator = _finite_real(denominator, label="denominator")
            denominators = None

        if numerators is not None and denominators is not None:
            if len(numerators) != len(denominators):
                raise TNFRValueError(
                    "numerator and denominator iterables must have equal length"
                )
            pairs = zip(numerators, denominators)
        elif numerators is not None:
            pairs = ((item, denominator) for item in numerators)
        elif denominators is not None:
            pairs = ((numerator, item) for item in denominators)
        else:
            return divide(numerator, denominator)
        return [divide(first, second) for first, second in pairs]

    def compute_circular_mean(self, angles: ArrayLike) -> float:
        """Compute the mean of admitted represented phases with a nonzero resultant."""
        if NUMPY_AVAILABLE:
            values = _real_array(angles, label="angle")
            if values.size == 0:
                raise TNFRValueError("circular mean requires at least one angle")
            mean_sin = float(np.mean(np.sin(values)))
            mean_cos = float(np.mean(np.cos(values)))
        else:
            source = (
                angles
                if hasattr(angles, "__iter__")
                and not isinstance(angles, (str, bytes, bytearray))
                else [angles]
            )
            values = [_finite_real(angle, label="angle") for angle in source]
            if not values:
                raise TNFRValueError("circular mean requires at least one angle")
            mean_sin = math.fsum(math.sin(angle) for angle in values) / len(values)
            mean_cos = math.fsum(math.cos(angle) for angle in values) / len(values)

        if math.hypot(mean_sin, mean_cos) <= CONSTANTS.FLOAT_TOLERANCE:
            raise TNFRValueError("circular mean is undefined for a vanishing resultant")
        return math.atan2(mean_sin, mean_cos)

    def is_finite_array(self, arr: ArrayLike) -> bool:
        """Return whether an array contains only finite numeric values."""
        if NUMPY_AVAILABLE:
            arr = np.asarray(arr)
            return bool(np.all(np.isfinite(arr)))
        else:
            # Fallback implementation
            if hasattr(arr, "__iter__"):
                return all(math.isfinite(x) for x in arr)
            else:
                return math.isfinite(arr)

    def clamp_value(
        self, value: ArrayLike, min_val: float, max_val: float
    ) -> ArrayLike:
        """Clamp admitted represented reals to a finite numeric interval."""
        lower = _finite_real(min_val, label="min_val")
        upper = _finite_real(max_val, label="max_val")
        if lower > upper:
            raise TNFRValueError("min_val must be less than or equal to max_val")
        if NUMPY_AVAILABLE:
            return np.clip(_real_array(value, label="value"), lower, upper)
        else:
            # Fallback implementation
            if hasattr(value, "__iter__") and not isinstance(
                value, (str, bytes, bytearray)
            ):
                return [
                    max(lower, min(upper, _finite_real(v, label="value")))
                    for v in value
                ]
            else:
                return max(lower, min(upper, _finite_real(value, label="value")))

    def kahan_sum_nd(
        self, values: Iterable[Sequence[float]], dims: int
    ) -> tuple[float, ...]:
        """Return compensated sums of values with dims components."""
        if isinstance(dims, bool) or not isinstance(dims, Integral) or dims < 1:
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
        """Return backend metadata with uninstrumented compatibility counters.

        The utility does not time calls. The two historical counter fields remain
        zero for schema compatibility and statistics_collected makes that scope
        machine-readable.
        """
        return {
            "numpy_available": NUMPY_AVAILABLE,
            "operation_count": 0,
            "total_time": 0.0,
            "statistics_collected": False,
            "seed": self.seed,
            "constants_version": "structural_tetrad_v1",
        }

    def reset_seed(self, new_seed: int) -> None:
        """Reset random seed for reproducibility."""
        self.seed = _validated_seed(new_seed)

        if NUMPY_AVAILABLE:
            self._rng = np.random.RandomState(self.seed)
        else:
            self._rng = random.Random(self.seed)

        logger.info(f"Reset numerical utilities seed to {self.seed}")


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
