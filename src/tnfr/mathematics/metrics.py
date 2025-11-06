"""Structural metrics preserving TNFR coherence invariants."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .operators import CoherenceOperator

__all__ = ["dcoh"]


def _as_coherent_vector(
    state: Sequence[complex] | np.ndarray,
    *,
    dimension: int,
) -> np.ndarray:
    """Return a complex vector compatible with ``CoherenceOperator`` matrices."""

    vector = np.asarray(state, dtype=np.complex128)
    if vector.ndim != 1 or vector.shape[0] != dimension:
        raise ValueError(
            "State vector dimension mismatch: "
            f"expected ({dimension},), received {vector.shape!r}."
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
        raise ValueError(f"Cannot normalise null coherence state {label}.")
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
      negative projections signal a collapse and therefore raise ``ValueError``.

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
        raise ValueError("Operator application distorted coherence dimensionality.")

    cross = np.vdot(vector1_norm, weighted_vector2)
    if not np.isfinite(cross):
        raise ValueError("State overlap produced a non-finite value.")

    expect1 = float(operator.expectation(vector1, normalise=normalise, atol=atol))
    expect2 = float(operator.expectation(vector2, normalise=normalise, atol=atol))

    for idx, value in enumerate((expect1, expect2), start=1):
        if not np.isfinite(value):
            raise ValueError(f"Coherence expectation diverged for state ψ{idx}.")
        if value <= 0.0 or np.isclose(value, 0.0, atol=atol):
            raise ValueError(
                "Coherence expectation must remain strictly positive to"
                f" preserve TNFR invariants (state ψ{idx})."
            )

    denominator = expect1 * expect2
    if not np.isfinite(denominator):
        raise ValueError("Coherence expectations produced a non-finite product.")
    if denominator <= 0.0 or np.isclose(denominator, 0.0, atol=atol):
        raise ValueError(
            "Product of coherence expectations must be strictly positive to"
            " evaluate dissimilarity."
        )

    ratio = (np.abs(cross) ** 2) / denominator
    eps = max(np.finfo(float).eps * 10.0, atol)
    if ratio < -eps:
        raise ValueError("Overlap produced a negative coherence ratio.")
    if ratio < 0.0:
        ratio = 0.0
    if ratio > 1.0 + eps:
        raise ValueError("Coherence ratio exceeded unity beyond tolerance.")
    if ratio > 1.0:
        ratio = 1.0

    return float(np.arccos(np.sqrt(ratio)))
