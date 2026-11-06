"""Projective angles induced by positive spectral expectation operators.

These auxiliary Hilbert-space diagnostics are distinct from canonical
structural coherence ``C(t)``.
"""

from __future__ import annotations

import math
from typing import Sequence

from ..constants.canonical import MATH_PRECISION_ENHANCEMENT_CANONICAL
from ._complex_arrays import (
    nonnegative_tolerance,
    normalized_complex_vector,
    numpy_complex_array,
)
from .operators import CoherenceOperator, SpectralExpectationOperator
from .unified_numerical import TNFRValueError, np

__all__ = ["spectral_weighted_angle", "dcoh"]


def _as_spectral_vector(
    state: Sequence[complex] | np.ndarray,
    *,
    dimension: int,
) -> np.ndarray:
    """Return one finite complex vector of the operator dimension."""

    vector = numpy_complex_array(state, label="Spectral state")
    if vector.ndim != 1 or vector.shape[0] != dimension:
        raise TNFRValueError(
            "State vector dimension mismatch.",
            context={"expected_dimension": dimension, "received_shape": vector.shape},
            suggestion="Ensure state vector matches operator dimension.",
        )
    return vector


def spectral_weighted_angle(
    psi1: Sequence[complex] | np.ndarray,
    psi2: Sequence[complex] | np.ndarray,
    operator: SpectralExpectationOperator,
    *,
    normalise: bool = True,
    atol: float = 1e-9,
) -> float:
    r"""Return the projective angle induced by a PSD Hermitian operator.

    For non-null rays under ``A``, the result is

    ``arccos(|<psi1, A psi2>| / sqrt(<psi1,A psi1><psi2,A psi2>))``.

    It lies in ``[0, pi/2]`` and is invariant under independent global phases.
    Positive semidefiniteness is required for the weighted Cauchy inequality;
    a singular operator gives an angle only for states outside its null space.
    This is an auxiliary spectral geometry and is never structural ``C(t)``.
    """

    tolerance = nonnegative_tolerance(atol)
    if not operator.is_positive_semidefinite(atol=tolerance):
        raise TNFRValueError(
            "Spectral weighted angle requires a positive-semidefinite operator.",
            context={"spectral_minimum": float(np.min(operator.eigenvalues.real))},
            suggestion="Use a positive-semidefinite Hermitian operator.",
        )

    matrix = operator.matrix
    dimension = matrix.shape[0]
    vector1 = _as_spectral_vector(psi1, dimension=dimension)
    vector2 = _as_spectral_vector(psi2, dimension=dimension)
    if normalise:
        vector1 = normalized_complex_vector(
            vector1, atol=tolerance, label="spectral state psi1"
        )
        vector2 = normalized_complex_vector(
            vector2, atol=tolerance, label="spectral state psi2"
        )

    weighted_vector2 = matrix @ vector2
    cross = np.vdot(vector1, weighted_vector2)
    if not bool(np.isfinite(cross)):
        raise TNFRValueError("Weighted spectral overlap must be finite.")

    expect1 = float(operator.expectation(vector1, normalise=False, atol=tolerance))
    expect2 = float(operator.expectation(vector2, normalise=False, atol=tolerance))
    for index, value in enumerate((expect1, expect2), start=1):
        if not math.isfinite(value) or value <= tolerance:
            raise TNFRValueError(
                "Spectral expectation must be positive outside the operator "
                f"null space (state psi{index}).",
                context={"expectation_value": value, "atol": tolerance},
            )

    # Exact identity of the admitted represented vectors proves a common ray.
    # Avoid magnifying independent dot-product roundoff through acos near one.
    if np.array_equal(vector1, vector2):
        return 0.0

    # Combine binary exponents separately: the dimensionless ratio may be
    # representable even when either unscaled squared quantity is not.
    overlap_mantissa, overlap_exponent = math.frexp(
        math.hypot(float(cross.real), float(cross.imag))
    )
    first_mantissa, first_exponent = math.frexp(expect1)
    second_mantissa, second_exponent = math.frexp(expect2)
    try:
        ratio = math.ldexp(
            overlap_mantissa**2 / (first_mantissa * second_mantissa),
            2 * overlap_exponent - first_exponent - second_exponent,
        )
    except OverflowError as exc:
        raise TNFRValueError(
            "Weighted overlap violates the positive-semidefinite angle bound."
        ) from exc
    eps = max(
        np.finfo(float).eps * MATH_PRECISION_ENHANCEMENT_CANONICAL,
        tolerance,
    )
    if not math.isfinite(ratio) or ratio < -eps or ratio > 1.0 + eps:
        raise TNFRValueError(
            "Weighted overlap violates the positive-semidefinite angle bound.",
            context={"squared_ratio": ratio, "tolerance": eps},
        )
    bounded_ratio = min(1.0, max(0.0, ratio))
    return float(np.arccos(np.sqrt(bounded_ratio)))


def dcoh(
    psi1: Sequence[complex] | np.ndarray,
    psi2: Sequence[complex] | np.ndarray,
    operator: CoherenceOperator,
    *,
    normalise: bool = True,
    atol: float = 1e-9,
) -> float:
    """Compatibility alias for :func:`spectral_weighted_angle`.

    The historical name does not denote, compare or derive canonical ``C(t)``.
    """

    return spectral_weighted_angle(
        psi1,
        psi2,
        operator,
        normalise=normalise,
        atol=atol,
    )
