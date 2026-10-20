"""Finite-output regressions for the auxiliary state projector."""

from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest

from tnfr.mathematics.projection import BasicStateProjector


@pytest.mark.parametrize("epi", [1e200, -1e200, 1e308])
def test_large_finite_envelope_preserves_unit_state_and_relative_phases(epi):
    vector = BasicStateProjector()(epi, nu_f=1.0, theta=0.25, dim=4)
    expected = np.exp(1j * (0.25 + np.arange(1, 5) / 2)) / 2

    assert np.all(np.isfinite(vector))
    assert np.linalg.norm(vector) == pytest.approx(1.0)
    np.testing.assert_allclose(vector, expected, atol=1e-15)


def test_large_stochastic_envelope_preserves_normalized_seeded_direction():
    seed = 71
    rng = np.random.default_rng(seed)
    expected = np.exp(1j * np.arange(1, 5) / 2) + 0.05 * (
        rng.standard_normal(4) + 1j * rng.standard_normal(4)
    )
    expected /= np.linalg.norm(expected)

    vector = BasicStateProjector()(
        1e200, nu_f=1.0, theta=0.0, dim=4, rng=np.random.default_rng(seed)
    )

    np.testing.assert_allclose(vector, expected, atol=1e-15)
    assert np.linalg.norm(vector) == pytest.approx(1.0)


def test_unrepresentable_projector_arithmetic_rejects_nonfinite_output():
    with np.errstate(over="ignore", invalid="ignore"):
        with pytest.raises(ValueError, match="finite"):
            BasicStateProjector()(1.0, nu_f=1e308, theta=1e308, dim=4)


@pytest.mark.parametrize("atol", [True, -1.0, np.nan, np.inf, Fraction(1, 10**400)])
def test_projector_rejects_invalid_normalization_tolerances(atol):
    with pytest.raises(ValueError, match="atol"):
        BasicStateProjector(atol=atol)(1.0, nu_f=1.0, theta=0.0, dim=4)


def test_projector_respects_the_declared_null_threshold():
    # With one mode and EPI=0 the amplitude envelope is exactly 1.5.
    assert abs(BasicStateProjector(atol=1.0)(0.0, 0.0, 0.0, 1)[0]) == pytest.approx(1)
    with pytest.raises(ValueError, match="null"):
        BasicStateProjector(atol=2.0)(0.0, 0.0, 0.0, 1)


@pytest.mark.parametrize("coordinate", ["epi", "nu_f", "theta"])
@pytest.mark.parametrize("invalid", [True, np.bool_(False), "1", Fraction(1, 10**400)])
def test_projector_rejects_primitive_coercion_before_random_excitation(
    coordinate, invalid
):
    rng = np.random.default_rng(71)
    original_state = rng.bit_generator.state
    values = {"epi": -0.25, "nu_f": 0.0, "theta": 0.5}
    values[coordinate] = invalid
    with pytest.raises(ValueError):
        BasicStateProjector()(**values, dim=2, rng=rng)
    assert rng.bit_generator.state == original_state


@pytest.mark.parametrize("dimension", [True, np.bool_(True), 0, -1, 1.5])
def test_projector_requires_positive_integer_dimension(dimension):
    with pytest.raises(ValueError, match="dimension"):
        BasicStateProjector()(0.0, 0.0, 0.0, dimension)


def test_projector_rejects_negative_capacity():
    with pytest.raises(ValueError, match="capacity"):
        BasicStateProjector()(0.0, -1.0, 0.0, 2)


@pytest.mark.parametrize(
    "options",
    [
        {"dtype": np.float64},
        {"dtype": np.int64},
        {"dtype": object},
        {"atol": True},
        {"atol": Fraction(1, 10**400)},
    ],
)
def test_projector_configuration_rejects_before_rng_changes(options):
    rng = np.random.default_rng(7)
    original_state = rng.bit_generator.state
    with pytest.raises(ValueError):
        BasicStateProjector(**options)(0.0, 0.0, 0.0, 2, rng=rng)
    assert rng.bit_generator.state == original_state


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
def test_projector_storage_keeps_complex_direction_and_normalization(dtype):
    state = BasicStateProjector(dtype=dtype)(0.0, 0.0, 0.0, 2)
    assert state.dtype == dtype
    assert np.all(state.imag > 0)
    assert np.linalg.norm(state) == pytest.approx(1.0, abs=2e-7)


def test_mutated_projector_dtype_is_readmitted_at_call():
    projector = BasicStateProjector()
    projector.dtype = np.float64
    with pytest.raises(ValueError, match="complex"):
        projector(0.0, 0.0, 0.0, 2)
