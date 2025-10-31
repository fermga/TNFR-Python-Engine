"""Tests for TNFR coherence metrics ensuring canonical contracts."""

from __future__ import annotations

import numpy as np
import pytest

from tnfr.mathematics import CoherenceOperator
from tnfr.mathematics.metrics import dcoh


@pytest.fixture()
def hermitian_operator() -> CoherenceOperator:
    """Return a simple Hermitian coherence operator for two modes."""

    return CoherenceOperator([[1.0, 0.2], [0.2, 0.8]])


@pytest.fixture()
def orthonormal_basis() -> tuple[np.ndarray, np.ndarray]:
    """Provide the canonical orthonormal basis vectors in C^2."""

    identity = np.eye(2, dtype=np.complex128)
    return identity[0], identity[1]


def test_dcoh_identity_returns_zero(
    hermitian_operator: CoherenceOperator, orthonormal_basis: tuple[np.ndarray, np.ndarray]
) -> None:
    """The dissimilarity of a state with itself must vanish."""

    psi, _ = orthonormal_basis
    result = dcoh(psi, psi, hermitian_operator)
    assert result == pytest.approx(0.0, abs=1e-12)


def test_dcoh_identity_superposition(
    hermitian_operator: CoherenceOperator, orthonormal_basis: tuple[np.ndarray, np.ndarray]
) -> None:
    """Non-eigenstates remain coherent with themselves under any operator."""

    psi1, psi2 = orthonormal_basis
    superposition = (psi1 + psi2) / np.sqrt(2.0)

    result = dcoh(superposition, superposition, hermitian_operator)
    assert result == pytest.approx(0.0, abs=1e-12)


def test_dcoh_is_symmetric(
    hermitian_operator: CoherenceOperator, orthonormal_basis: tuple[np.ndarray, np.ndarray]
) -> None:
    """Swapping the states must not change the coherence dissimilarity."""

    psi1, psi2 = orthonormal_basis
    superposition = (psi1 + psi2) / np.sqrt(2.0)

    forward = dcoh(psi1, superposition, hermitian_operator)
    backward = dcoh(superposition, psi1, hermitian_operator)

    assert forward == pytest.approx(backward, rel=1e-12, abs=1e-12)


def test_dcoh_orthogonal_states_are_maximally_dissimilar(
    hermitian_operator: CoherenceOperator, orthonormal_basis: tuple[np.ndarray, np.ndarray]
) -> None:
    """Orthogonal basis vectors must yield maximal coherence dissimilarity."""

    psi1, psi2 = orthonormal_basis

    result = dcoh(psi1, psi2, hermitian_operator)

    assert result == pytest.approx(1.0, abs=1e-12)


def test_dcoh_satisfies_triangle_inequality(
    hermitian_operator: CoherenceOperator, orthonormal_basis: tuple[np.ndarray, np.ndarray]
) -> None:
    """The coherence dissimilarity behaves as a metric on the tested states."""

    psi1, psi3 = orthonormal_basis
    psi2 = (psi1 + psi3) / np.sqrt(2.0)

    direct = dcoh(psi1, psi3, hermitian_operator)
    via_intermediate = dcoh(psi1, psi2, hermitian_operator) + dcoh(psi2, psi3, hermitian_operator)

    assert direct <= via_intermediate + 1e-12


def test_dcoh_rejects_zero_expectation(
    orthonormal_basis: tuple[np.ndarray, np.ndarray]
) -> None:
    """Operators with null expectation for a state must raise an error."""

    psi, _ = orthonormal_basis
    singular_operator = CoherenceOperator([[0.0, 0.0], [0.0, 1.0]])

    with pytest.raises(ValueError, match="Coherence expectation must remain strictly positive"):
        dcoh(psi, psi, singular_operator)


def test_dcoh_respects_tolerance_thresholds(
    orthonormal_basis: tuple[np.ndarray, np.ndarray]
) -> None:
    """Adjusting the tolerance must allow near-null expectations when explicit."""

    psi1, psi2 = orthonormal_basis
    near_null_operator = CoherenceOperator([1e-12, 1.0])

    with pytest.raises(ValueError, match="Coherence expectation must remain strictly positive"):
        dcoh(psi1, psi2, near_null_operator)

    result = dcoh(psi1, psi2, near_null_operator, atol=1e-13)
    assert result == pytest.approx(1.0, abs=1e-12)
