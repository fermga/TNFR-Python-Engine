"""Tests for TNFR mathematics operator contracts."""

from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")

from tnfr.mathematics.operators import DEFAULT_C_MIN, CoherenceOperator, FrequencyOperator
from tnfr.mathematics.operators_factory import (
    make_coherence_operator,
    make_frequency_operator,
)

def test_coherence_operator_from_matrix(structural_tolerances: dict[str, float]) -> None:
    matrix = np.array([[2.0, 1.0 - 1.0j], [1.0 + 1.0j, 3.0]], dtype=np.complex128)
    operator = CoherenceOperator(matrix)

    assert operator.is_hermitian(atol=structural_tolerances["atol"])
    assert operator.is_positive_semidefinite(atol=structural_tolerances["atol"])
    assert operator.c_min == pytest.approx(min(operator.eigenvalues.real))

    spectral_radius = operator.spectral_radius()
    assert spectral_radius >= operator.c_min
    assert operator.spectral_bandwidth() == pytest.approx(
        operator.eigenvalues.real.max() - operator.eigenvalues.real.min()
    )

def test_coherence_operator_from_eigenvalues() -> None:
    operator = CoherenceOperator([1.0, 2.0, 3.0])

    np.testing.assert_array_equal(operator.matrix, np.diag([1.0, 2.0, 3.0]))
    np.testing.assert_array_equal(operator.spectrum(), np.array([1.0, 2.0, 3.0], dtype=np.complex128))

def test_coherence_operator_allows_custom_c_min() -> None:
    operator = CoherenceOperator([[1.0, 0.0], [0.0, 2.0]], c_min=0.25)

    assert operator.c_min == pytest.approx(0.25)
    assert min(operator.eigenvalues.real) != pytest.approx(operator.c_min)

def test_coherence_operator_respects_default_constant_when_explicit() -> None:
    operator = CoherenceOperator([[1.0, 0.0], [0.0, 2.0]], c_min=DEFAULT_C_MIN)

    assert operator.c_min == pytest.approx(DEFAULT_C_MIN)
    assert min(operator.eigenvalues.real) != pytest.approx(operator.c_min)

def test_coherence_operator_expectation(structural_rng: np.random.Generator) -> None:
    matrix = np.array([[1.0, 0.0], [0.0, 2.0]], dtype=np.complex128)
    operator = CoherenceOperator(matrix)

    state = structural_rng.normal(size=2) + 1j * structural_rng.normal(size=2)
    expectation = operator.expectation(state)
    manual = np.vdot(state / np.linalg.norm(state), matrix @ (state / np.linalg.norm(state)))
    assert isinstance(expectation, float)
    assert expectation == pytest.approx(float(manual.real))

    with pytest.raises(ValueError):
        operator.expectation([1.0, 0.0, 0.0])

def test_coherence_operator_non_hermitian_rejected() -> None:
    matrix = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=np.complex128)
    with pytest.raises(ValueError):
        CoherenceOperator(matrix)

def test_frequency_operator_properties(structural_rng: np.random.Generator) -> None:
    operator = FrequencyOperator([0.5, 1.5, 3.0])

    spectrum = operator.spectrum()
    assert spectrum.dtype == float
    assert np.all(spectrum >= 0)

    state = structural_rng.normal(size=3) + 1j * structural_rng.normal(size=3)
    projected = operator.project_frequency(state)
    assert isinstance(projected, float)
    assert projected == pytest.approx(operator.expectation(state))

def test_make_coherence_operator_defaults_to_c_min() -> None:
    operator = make_coherence_operator(2, c_min=0.6)

    np.testing.assert_allclose(operator.matrix, np.diag([0.6, 0.6]))
    assert operator.c_min == pytest.approx(0.6)

def test_make_frequency_operator_preserves_valid_matrix(
    structural_tolerances: dict[str, float]
) -> None:
    matrix = np.array([[1.0, 0.2j], [-0.2j, 2.0]], dtype=np.complex128)

    operator = make_frequency_operator(matrix)

    np.testing.assert_allclose(operator.matrix, matrix, atol=structural_tolerances["atol"])
