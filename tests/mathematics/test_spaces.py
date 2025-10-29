"""Tests for TNFR mathematics space abstractions."""

from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")

from tnfr.mathematics.spaces import BanachSpaceEPI, HilbertSpace


def test_hilbert_space_basis_is_orthonormal(structural_tolerances: dict[str, float]) -> None:
    space = HilbertSpace(dimension=3)

    basis = space.basis

    np.testing.assert_allclose(
        basis @ basis.conj().T,
        np.eye(3, dtype=space.dtype),
        atol=structural_tolerances["atol"],
    )


def test_hilbert_space_inner_product_and_norm(structural_rng: np.random.Generator) -> None:
    space = HilbertSpace(dimension=4)

    vector_a = structural_rng.normal(size=4) + 1j * structural_rng.normal(size=4)
    vector_b = structural_rng.normal(size=4) + 1j * structural_rng.normal(size=4)

    inner = space.inner_product(vector_a, vector_b)
    manual = np.vdot(np.asarray(vector_a, dtype=space.dtype), np.asarray(vector_b, dtype=space.dtype))
    assert inner == pytest.approx(manual)

    norm = space.norm(vector_a)
    assert norm == pytest.approx(np.linalg.norm(vector_a))


def test_hilbert_space_projection(structural_rng: np.random.Generator, structural_tolerances: dict[str, float]) -> None:
    space = HilbertSpace(dimension=3)
    vector = structural_rng.normal(size=3) + 1j * structural_rng.normal(size=3)

    projected_basis = space.project(vector, onto=1)
    expected_basis = np.zeros(3, dtype=space.dtype)
    basis_vector = space.basis[1]
    coefficient = space.inner_product(vector, basis_vector) / space.inner_product(basis_vector, basis_vector)
    expected_basis[1] = coefficient
    np.testing.assert_allclose(projected_basis, expected_basis, atol=structural_tolerances["atol"])

    onto_vector = np.array([1.0 + 0j, 1.0 + 0j, 0.0 + 0j], dtype=space.dtype)
    projected_vector = space.project(vector, onto=onto_vector)
    coefficient = np.vdot(vector, onto_vector) / np.vdot(onto_vector, onto_vector)
    np.testing.assert_allclose(
        projected_vector,
        coefficient * onto_vector,
        atol=structural_tolerances["atol"],
    )


def test_hilbert_space_validation_errors() -> None:
    with pytest.raises(ValueError):
        HilbertSpace(dimension=0)

    space = HilbertSpace(dimension=2)
    with pytest.raises(ValueError):
        space.inner_product([1.0, 2.0, 3.0], [1.0, 2.0])

    with pytest.raises(IndexError):
        space.project([1.0, 0.0], onto=2)


def test_banach_space_domain_validation() -> None:
    with pytest.raises(ValueError):
        BanachSpaceEPI([0.0])

    with pytest.raises(ValueError):
        BanachSpaceEPI([0.0, 0.0, 1.0])


def test_banach_space_coherence_functional(structural_tolerances: dict[str, float]) -> None:
    domain = (0.0, 0.5, 1.5)
    epi = np.array([1.0 + 1.0j, 2.0 + 0.0j, -1.0 + 0.5j], dtype=np.complex128)
    space = BanachSpaceEPI(domain, amplitude_weight=1.5, derivative_weight=0.75)

    functional = space.compute_coherence_functional(epi)
    amplitude_term = space.amplitude_weight * np.abs(epi) ** 2
    derivative = np.gradient(epi, space.domain, edge_order=2)
    derivative_term = space.derivative_weight * np.abs(derivative) ** 2

    np.testing.assert_allclose(functional, amplitude_term + derivative_term, atol=structural_tolerances["atol"])

    expected_norm = np.sqrt(np.trapz(amplitude_term + derivative_term, space.domain))
    assert space.coherence_norm(epi) == pytest.approx(expected_norm)
