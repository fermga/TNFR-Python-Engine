"""Tests for EPI elements and Banach space delegation."""

from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")

from tnfr.mathematics import BEPIElement, BanachSpaceEPI, HilbertSpace


@pytest.fixture()
def sample_grid() -> np.ndarray:
    return np.linspace(0.0, 1.0, 4)


@pytest.fixture()
def sample_elements(sample_grid: np.ndarray) -> tuple[BEPIElement, BEPIElement]:
    first = BEPIElement(
        np.array([0.0 + 0.0j, 0.2 + 0.5j, -0.1 + 0.1j, 0.3 + 0.0j]),
        np.array([1.0 + 0.0j, 0.5 + 0.0j], dtype=np.complex128),
        sample_grid,
    )
    second = BEPIElement(
        np.array([0.1 + 0.0j, -0.1 + 0.1j, 0.2 - 0.2j, 0.0 + 0.0j]),
        np.array([-0.5 + 0.0j, 0.0 + 0.5j], dtype=np.complex128),
        sample_grid,
    )
    return first, second


def test_direct_sum_preserves_norm_structure(sample_elements: tuple[BEPIElement, BEPIElement]) -> None:
    space = BanachSpaceEPI()
    element_a, element_b = sample_elements

    combined = space.direct_sum(element_a, element_b)

    np.testing.assert_allclose(
        combined.f_continuous,
        element_a.f_continuous + element_b.f_continuous,
    )
    np.testing.assert_allclose(
        combined.a_discrete,
        element_a.a_discrete + element_b.a_discrete,
    )

    expected_norm = space.coherence_norm(
        element_a.f_continuous + element_b.f_continuous,
        element_a.a_discrete + element_b.a_discrete,
        x_grid=combined.x_grid,
    )
    combined_norm = space.coherence_norm(
        combined.f_continuous,
        combined.a_discrete,
        x_grid=combined.x_grid,
    )
    assert combined_norm == pytest.approx(expected_norm)


def test_adjoint_inverts_phase(sample_elements: tuple[BEPIElement, BEPIElement]) -> None:
    space = BanachSpaceEPI()
    element_a, _ = sample_elements

    adjoint = space.adjoint(element_a)

    np.testing.assert_allclose(adjoint.f_continuous, np.conjugate(element_a.f_continuous))
    np.testing.assert_allclose(adjoint.a_discrete, np.conjugate(element_a.a_discrete))

    original_norm = space.coherence_norm(
        element_a.f_continuous,
        element_a.a_discrete,
        x_grid=element_a.x_grid,
    )
    adjoint_norm = space.coherence_norm(
        adjoint.f_continuous,
        adjoint.a_discrete,
        x_grid=adjoint.x_grid,
    )
    assert original_norm == pytest.approx(adjoint_norm)


def test_tensor_with_hilbert_matches_outer_product(sample_elements: tuple[BEPIElement, BEPIElement]) -> None:
    element_a, _ = sample_elements
    hilbert = HilbertSpace(dimension=2)
    vector = np.array([1.0 + 0.0j, 1.0j], dtype=hilbert.dtype)

    tensor = element_a.tensor(vector)
    via_space = BanachSpaceEPI().tensor_with_hilbert(element_a, hilbert, vector)

    expected = np.outer(element_a.a_discrete, vector)
    np.testing.assert_allclose(tensor, expected)
    np.testing.assert_allclose(via_space, expected)


def test_compose_applies_componentwise(sample_elements: tuple[BEPIElement, BEPIElement]) -> None:
    space = BanachSpaceEPI()
    element_a, _ = sample_elements

    scaled = space.compose(
        element_a,
        lambda values: 2.0 * values,
        spectral_transform=lambda values: values + 1.0,
    )

    np.testing.assert_allclose(scaled.f_continuous, 2.0 * element_a.f_continuous)
    np.testing.assert_allclose(scaled.a_discrete, element_a.a_discrete + 1.0)

    scaled_norm = space.coherence_norm(
        scaled.f_continuous,
        scaled.a_discrete,
        x_grid=scaled.x_grid,
    )
    manual_norm = space.coherence_norm(
        2.0 * element_a.f_continuous,
        element_a.a_discrete + 1.0,
        x_grid=element_a.x_grid,
    )
    assert scaled_norm == pytest.approx(manual_norm)


def test_zero_and_basis_factories(sample_grid: np.ndarray) -> None:
    space = BanachSpaceEPI()

    zero = space.zero_element(continuous_size=4, discrete_size=3, x_grid=sample_grid)
    assert isinstance(zero, BEPIElement)
    assert np.allclose(zero.f_continuous, 0.0)
    assert np.allclose(zero.a_discrete, 0.0)

    basis = space.canonical_basis(
        continuous_size=4,
        discrete_size=3,
        continuous_index=2,
        discrete_index=1,
        x_grid=sample_grid,
    )
    assert basis.f_continuous[2] == pytest.approx(1.0)
    assert np.count_nonzero(basis.f_continuous) == 1
    assert basis.a_discrete[1] == pytest.approx(1.0)
    assert np.count_nonzero(basis.a_discrete) == 1

    combined = space.direct_sum(zero, basis)
    np.testing.assert_allclose(combined.f_continuous, basis.f_continuous)
    np.testing.assert_allclose(combined.a_discrete, basis.a_discrete)
