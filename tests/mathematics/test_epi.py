"""Tests for EPI elements and Banach space delegation."""

from __future__ import annotations

from fractions import Fraction

import pytest

np = pytest.importorskip("numpy")

from tnfr.mathematics import BanachSpaceEPI, BEPIElement, HilbertSpace
from tnfr.types import ensure_bepi, require_finite_real_scalar_epi, serialize_bepi_json


@pytest.mark.parametrize(
    "component",
    [True, np.bool_(False), "1", Fraction(1, 10**400), Fraction(10**400), np.inf],
)
@pytest.mark.parametrize("channel", ["continuous", "discrete"])
def test_bepi_admits_components_before_complex_materialization(component, channel):
    continuous = (component, 1) if channel == "continuous" else (1, 1)
    discrete = (component,) if channel == "discrete" else (1,)

    with pytest.raises(ValueError):
        BEPIElement(continuous, discrete, (0, 1))


@pytest.mark.parametrize("component", [True, Fraction(1, 10**400), Fraction(10**400)])
def test_scalar_bepi_adapter_cannot_discard_invalid_primitive(component):
    with pytest.raises(ValueError):
        ensure_bepi(component)


@pytest.mark.parametrize("channel", ["real", "imag"])
def test_serialized_bepi_admits_both_components_before_complex_conversion(channel):
    component = {"real": 1, "imag": 0}
    component[channel] = Fraction(1, 10**400)
    storage = {"continuous": (component, component), "discrete": (), "grid": (0, 1)}

    with pytest.raises(ValueError):
        ensure_bepi(storage)


def test_bepi_preserves_complex_nonuniform_storage_and_signed_subnormal():
    element = BEPIElement((1 + 2j, -3j), (Fraction(-3, 2),), (0, 1))
    restored = ensure_bepi(serialize_bepi_json(element))
    np.testing.assert_array_equal(restored.f_continuous, [1 + 2j, -3j])
    np.testing.assert_array_equal(restored.a_discrete, [-1.5])
    assert restored.real_scalar_embedding() is None

    smallest = float.fromhex("0x0.0000000000001p-1022")
    scalar = ensure_bepi(Fraction.from_float(-smallest))
    assert require_finite_real_scalar_epi(scalar) == -smallest


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


def test_direct_sum_preserves_regularity_evaluation(
    sample_elements: tuple[BEPIElement, BEPIElement],
) -> None:
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

    expected_regularity = space.composite_epi_regularity(
        element_a.f_continuous + element_b.f_continuous,
        element_a.a_discrete + element_b.a_discrete,
        x_grid=combined.x_grid,
    )
    combined_regularity = space.composite_epi_regularity(
        combined.f_continuous,
        combined.a_discrete,
        x_grid=combined.x_grid,
    )
    assert combined_regularity == pytest.approx(expected_regularity)


def test_adjoint_inverts_phase(
    sample_elements: tuple[BEPIElement, BEPIElement],
) -> None:
    space = BanachSpaceEPI()
    element_a, _ = sample_elements

    adjoint = space.adjoint(element_a)

    np.testing.assert_allclose(
        adjoint.f_continuous, np.conjugate(element_a.f_continuous)
    )
    np.testing.assert_allclose(adjoint.a_discrete, np.conjugate(element_a.a_discrete))

    original_regularity = space.composite_epi_regularity(
        element_a.f_continuous,
        element_a.a_discrete,
        x_grid=element_a.x_grid,
    )
    adjoint_regularity = space.composite_epi_regularity(
        adjoint.f_continuous,
        adjoint.a_discrete,
        x_grid=adjoint.x_grid,
    )
    assert original_regularity == pytest.approx(adjoint_regularity)


def test_tensor_with_hilbert_matches_outer_product(
    sample_elements: tuple[BEPIElement, BEPIElement],
) -> None:
    element_a, _ = sample_elements
    hilbert = HilbertSpace(dimension=2)
    vector = np.array([1.0 + 0.0j, 1.0j], dtype=hilbert.dtype)

    tensor = element_a.tensor(vector)
    via_space = BanachSpaceEPI().tensor_with_hilbert(element_a, hilbert, vector)

    expected = np.outer(element_a.a_discrete, vector)
    np.testing.assert_allclose(tensor, expected)
    np.testing.assert_allclose(via_space, expected)


def test_compose_applies_componentwise(
    sample_elements: tuple[BEPIElement, BEPIElement],
) -> None:
    space = BanachSpaceEPI()
    element_a, _ = sample_elements

    scaled = space.compose(
        element_a,
        lambda values: 2.0 * values,
        spectral_transform=lambda values: values + 1.0,
    )

    np.testing.assert_allclose(scaled.f_continuous, 2.0 * element_a.f_continuous)
    np.testing.assert_allclose(scaled.a_discrete, element_a.a_discrete + 1.0)

    scaled_regularity = space.composite_epi_regularity(
        scaled.f_continuous,
        scaled.a_discrete,
        x_grid=scaled.x_grid,
    )
    manual_regularity = space.composite_epi_regularity(
        2.0 * element_a.f_continuous,
        element_a.a_discrete + 1.0,
        x_grid=element_a.x_grid,
    )
    assert scaled_regularity == pytest.approx(manual_regularity)


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


def test_bepi_convergent_residuals_have_vanishing_regularity(
    sample_grid: np.ndarray,
) -> None:
    space = BanachSpaceEPI()

    def partial_weight(order: int) -> float:
        return sum(0.5**k for k in range(order + 1))

    def make_element(order: int) -> BEPIElement:
        weight = partial_weight(order)
        f_vector = np.zeros(4, dtype=np.complex128)
        a_vector = np.zeros(3, dtype=np.complex128)
        f_vector[1] = weight
        a_vector[0] = weight
        return BEPIElement(f_vector, a_vector, sample_grid)

    sequence = [make_element(order) for order in range(6)]
    limit_weight = 1.0 / (1.0 - 0.5)
    limit_element = BEPIElement(
        np.array([0.0, limit_weight, 0.0, 0.0], dtype=np.complex128),
        np.array([limit_weight, 0.0, 0.0], dtype=np.complex128),
        sample_grid,
    )

    residual_regularities = [
        space.composite_epi_regularity(
            limit_element.f_continuous - element.f_continuous,
            limit_element.a_discrete - element.a_discrete,
            x_grid=sample_grid,
        )
        for element in sequence
    ]

    assert residual_regularities[-1] < 0.08
    assert all(
        earlier >= later
        for earlier, later in zip(residual_regularities, residual_regularities[1:])
    )

    tail_regularity_bounds = []
    for start in range(4, len(sequence) - 1):
        diffs = [
            space.composite_epi_regularity(
                sequence[next_idx].f_continuous - sequence[start].f_continuous,
                sequence[next_idx].a_discrete - sequence[start].a_discrete,
                x_grid=sample_grid,
            )
            for next_idx in range(start + 1, len(sequence))
        ]
        tail_regularity_bounds.append(max(diffs))

    assert tail_regularity_bounds
    assert all(bound < 0.08 for bound in tail_regularity_bounds)
