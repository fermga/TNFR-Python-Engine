"""Structural metrics preserving TNFR coherence invariants."""

from __future__ import annotations

from typing import Sequence

from .unified_numerical import np, TNFRValueError

from .operators import CoherenceOperator
from ..constants.canonical import MATH_PRECISION_ENHANCEMENT_CANONICAL

__all__ = ["dcoh"]


def _as_coherent_vector(
    state: Sequence[complex] | np.ndarray,
    *,
    dimension: int,
) -> np.ndarray:
    """Return a complex vector compatible with ``CoherenceOperator`` matrices."""

    vector = np.asarray(state, dtype=np.complex128)
    if vector.ndim != 1 or vector.shape[0] != dimension:
        raise TNFRValueError(
            "State vector dimension mismatch.",
            context={
                "expected_dimension": dimension,
                "received_shape": vector.shape
            },
            suggestion="Ensure state vector matches operator dimension."
        )
    return vector


def _normalise_vector(
    vector: np.ndarray,
    *,
    atol: float,
    label: str,
) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if np.isclose(norm, 0.0, atol=atol):
        raise TNFRValueError(
            f"Cannot normalise null coherence state {label}.",
            context={"norm": norm, "atol": atol},
            suggestion="Provide a non-zero state vector."
        )
    return vector / norm


def dcoh(
    psi1: Sequence[complex] | np.ndarray,
    psi2: Sequence[complex] | np.ndarray,
    operator: CoherenceOperator,
    *,
    normalise: bool = True,
    atol: float = 1e-9,
) -> float:
    """Return the TNFR dissimilarity of coherence between ``psi1`` and ``psi2``.

    The metric follows the canonical TNFR expectation contracts:

    * States are converted to Hilbert-compatible complex vectors respecting the
      ``CoherenceOperator`` dimension, preserving the spectral phase space.
    * Optional normalisation keeps overlap and expectations coherent with
      unit-phase contracts, preventing coherence inflation.
    * Expectation values ``⟨ψ|Ĉ|ψ⟩`` must remain strictly positive; null or
      negative projections signal a collapse and therefore raise ``TNFRValueError``.

    Parameters mirror the runtime helpers so callers can rely on the same
    tolerances.  Numerical overflow is contained by bounding intermediate ratios
    within ``[0, 1]`` up to ``atol`` before applying the Bures-style angle
    ``arccos(√ratio)``, ensuring the returned dissimilarity remains within the
    TNFR coherence interval.
    """

    dimension = operator.matrix.shape[0]
    vector1 = _as_coherent_vector(psi1, dimension=dimension)
    vector2 = _as_coherent_vector(psi2, dimension=dimension)

    if normalise:
        vector1_norm = _normalise_vector(vector1, atol=atol, label="ψ₁")
        vector2_norm = _normalise_vector(vector2, atol=atol, label="ψ₂")
    else:
        vector1_norm = vector1
        vector2_norm = vector2

    weighted_vector2 = operator.matrix @ vector2_norm
    if weighted_vector2.shape != vector2_norm.shape:
        raise TNFRValueError(
            "Operator application distorted coherence dimensionality.",
            context={
                "input_shape": vector2_norm.shape,
                "output_shape": weighted_vector2.shape
            },
            suggestion="Check operator matrix dimensions."
        )

    cross = np.vdot(vector1_norm, weighted_vector2)
    if not np.isfinite(cross):
        raise TNFRValueError(
            "State overlap produced a non-finite value.",
            context={"overlap": cross},
            suggestion="Check input states for NaN or Inf values."
        )

    expect1 = float(operator.expectation(vector1, normalise=normalise, atol=atol))
    expect2 = float(operator.expectation(vector2, normalise=normalise, atol=atol))

    for idx, value in enumerate((expect1, expect2), start=1):
        if not np.isfinite(value):
            raise TNFRValueError(
                f"Coherence expectation diverged for state ψ{idx}.",
                context={"expectation_value": value},
                suggestion="Check operator and state validity."
            )
        if value <= 0.0 or np.isclose(value, 0.0, atol=atol):
            raise TNFRValueError(
                f"Coherence expectation must remain strictly positive to preserve TNFR invariants (state ψ{idx}).",
                context={"expectation_value": value, "atol": atol},
                suggestion="Ensure states have non-zero projection on the operator."
            )

    denominator = expect1 * expect2
    if not np.isfinite(denominator):
        raise TNFRValueError(
            "Coherence expectations produced a non-finite product.",
            context={"denominator": denominator},
            suggestion="Check expectation values."
        )
    if denominator <= 0.0 or np.isclose(denominator, 0.0, atol=atol):
        raise TNFRValueError(
            "Product of coherence expectations must be strictly positive to evaluate dissimilarity.",
            context={"denominator": denominator, "atol": atol},
            suggestion="Ensure both states have positive expectations."
        )

    ratio = (np.abs(cross) ** 2) / denominator
    eps = max(np.finfo(float).eps * MATH_PRECISION_ENHANCEMENT_CANONICAL, atol)
    if ratio < -eps:
        raise TNFRValueError(
            "Overlap produced a negative coherence ratio.",
            context={"ratio": ratio, "eps": eps},
            suggestion="Check numerical stability or operator hermiticity."
        )
    if ratio < 0.0:
        ratio = 0.0
    if ratio > 1.0 + eps:
        raise TNFRValueError(
            "Coherence ratio exceeded unity beyond tolerance.",
            context={"ratio": ratio, "eps": eps},
            suggestion="Check normalization or operator properties."
        )
    if ratio > 1.0:
        ratio = 1.0

    return float(np.arccos(np.sqrt(ratio)))
