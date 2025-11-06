"""Tests for TNFR mathematics space abstractions."""

from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")

from tnfr.mathematics.spaces import BanachSpaceEPI, HilbertSpace


def test_hilbert_space_basis_is_orthonormal(
    structural_tolerances: dict[str, float],
) -> None:
    space = HilbertSpace(dimension=3)

    basis = space.basis

    np.testing.assert_allclose(
        basis @ basis.conj().T,
        np.eye(3, dtype=space.dtype),
        atol=structural_tolerances["atol"],
    )


def test_hilbert_space_inner_product_and_norm(
    structural_rng: np.random.Generator,
) -> None:
    space = HilbertSpace(dimension=4)

    vector_a = structural_rng.normal(size=4) + 1j * structural_rng.normal(size=4)
    vector_b = structural_rng.normal(size=4) + 1j * structural_rng.normal(size=4)

    inner = space.inner_product(vector_a, vector_b)
    manual = np.vdot(
        np.asarray(vector_a, dtype=space.dtype), np.asarray(vector_b, dtype=space.dtype)
    )
    assert inner == pytest.approx(manual)

    norm = space.norm(vector_a)
    assert norm == pytest.approx(np.linalg.norm(vector_a))


def test_hilbert_space_projection(
    structural_rng: np.random.Generator, structural_tolerances: dict[str, float]
) -> None:
    space = HilbertSpace(dimension=3)
    vector = structural_rng.normal(size=3) + 1j * structural_rng.normal(size=3)

    coefficients = space.project(vector)
    np.testing.assert_allclose(coefficients, vector, atol=structural_tolerances["atol"])

    custom_basis = [
        np.array([1.0 + 0j, 0.0 + 0j, 0.0 + 0j], dtype=space.dtype),
        (space.basis[1] + space.basis[2]) / np.sqrt(2.0),
        (space.basis[1] - space.basis[2]) / np.sqrt(2.0),
    ]
    expected = np.array(
        [space.inner_product(b, vector) for b in custom_basis], dtype=space.dtype
    )
    projected = space.project(vector, basis=custom_basis)
    np.testing.assert_allclose(projected, expected, atol=structural_tolerances["atol"])


def test_hilbert_space_validation_errors() -> None:
    with pytest.raises(ValueError):
        HilbertSpace(dimension=0)

    space = HilbertSpace(dimension=2)
    with pytest.raises(ValueError):
        space.inner_product([1.0, 2.0, 3.0], [1.0, 2.0])

    with pytest.raises(ValueError):
        space.project([1.0, 0.0], basis=[])

    non_orthonormal_basis = [np.array([1.0 + 0j, 0.0 + 0j], dtype=space.dtype)] * 2
    with pytest.raises(ValueError):
        space.project([1.0, 0.0], basis=non_orthonormal_basis)


def test_banach_space_domain_validation(structural_rng: np.random.Generator) -> None:
    space = BanachSpaceEPI()
    f = structural_rng.normal(size=4) + 1j * structural_rng.normal(size=4)
    a = structural_rng.normal(size=2) + 1j * structural_rng.normal(size=2)

    space.validate_domain(f, a, x_grid=np.linspace(0.0, 1.0, 4))

    with pytest.raises(ValueError):
        space.validate_domain(f, a, x_grid=[0.0, 0.5, 0.75])

    with pytest.raises(ValueError):
        space.validate_domain(f, a, x_grid=[0.0, 0.4, 0.9, 1.1])

    with pytest.raises(ValueError):
        space.validate_domain(f, a, x_grid=[0.0, 0.3, 0.9, 0.9])


def test_banach_space_coherence_functional(
    structural_tolerances: dict[str, float],
) -> None:
    space = BanachSpaceEPI()
    x_grid = np.linspace(0.0, 1.0, 5)
    f = np.exp(1j * np.pi * x_grid)

    derivative = np.gradient(f, x_grid, edge_order=2)
    numerator = np.trapz(np.abs(derivative) ** 2, x_grid)
    denominator = 1.0 + np.trapz(np.abs(f) ** 2, x_grid)
    expected = numerator / denominator

    result = space.compute_coherence_functional(f, x_grid)
    assert result == pytest.approx(
        expected, rel=structural_tolerances["rtol"], abs=structural_tolerances["atol"]
    )


def test_banach_space_coherence_norm_combines_components(
    structural_tolerances: dict[str, float],
) -> None:
    space = BanachSpaceEPI()
    x_grid = np.linspace(0.0, 1.0, 4)
    f = np.array([0.0 + 0.0j, 1.0 + 0.0j, 0.5 + 0.5j, 0.0 + 0.0j], dtype=np.complex128)
    a = np.array([1.0 + 0.0j, -1.0j], dtype=np.complex128)

    cf_value = space.compute_coherence_functional(f, x_grid)
    expected = 2.0 * np.max(np.abs(f)) + 3.0 * np.linalg.norm(a) + 0.5 * cf_value

    result = space.coherence_norm(f, a, x_grid=x_grid, alpha=2.0, beta=3.0, gamma=0.5)
    assert result == pytest.approx(
        expected, rel=structural_tolerances["rtol"], abs=structural_tolerances["atol"]
    )

    with pytest.raises(ValueError):
        space.coherence_norm(f, a, x_grid=x_grid, alpha=-1.0)
