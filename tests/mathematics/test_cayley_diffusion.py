"""Stationary modes and represented-domain boundaries of circulant diffusion."""

import cmath
from fractions import Fraction

import numpy as np
import pytest

from tnfr.mathematics.cayley import cayley_diffusion_action, cayley_spectrum


@pytest.mark.parametrize("components", [1, 2])
def test_long_time_preserves_each_connected_component_mean(components):
    modulus = 50 * components
    support = set(range(components, modulus, components))
    vector = np.tile([2.0, -3.0][:components], 50)
    spectrum = cayley_spectrum(modulus, support)
    null_modes = {index for index, value in enumerate(spectrum) if value == 0}
    assert null_modes == {50 * index for index in range(components)}
    assert all(
        value.real > 0
        for index, value in enumerate(spectrum)
        if index not in null_modes
    )
    evolved = cayley_diffusion_action(modulus, support, vector, structural_time=1e20)
    np.testing.assert_allclose(evolved, vector, rtol=0, atol=2e-15)


def test_long_time_retains_component_means_after_nonstationary_modes_decay():
    vector = np.arange(100, dtype=float)
    expected = np.tile([np.mean(vector[::2]), np.mean(vector[1::2])], 50)
    evolved = cayley_diffusion_action(
        100, set(range(2, 100, 2)), vector, structural_time=1e20
    )
    np.testing.assert_allclose(evolved, expected, rtol=0, atol=2e-14)


def test_directed_complex_mode_has_the_expected_decay_and_rotation():
    # For the forward four-cycle, L(1,i,-1,-i)=(1-i)(1,i,-1,-i).
    vector = np.array([1, 1j, -1, -1j])
    elapsed = 0.35
    expected = cmath.exp(-elapsed * (1 - 1j)) * vector
    evolved = cayley_diffusion_action(4, {1}, vector, structural_time=elapsed)
    assert np.iscomplexobj(evolved)
    np.testing.assert_allclose(evolved, expected, rtol=2e-15, atol=2e-15)


@pytest.mark.parametrize("option", ["structural_time", "capacity"])
@pytest.mark.parametrize(
    "value",
    [
        True,
        np.bool_(True),
        -1,
        float("inf"),
        float("nan"),
        Fraction(1, 10**400),
        10**400,
    ],
)
def test_clock_admission_precedes_materialization(option, value):
    with pytest.raises((TypeError, ValueError), match=option):
        cayley_diffusion_action(3, {1, 2}, [1, 1, 1], **{option: value})


@pytest.mark.parametrize("time,capacity", [(1e308, 1e308), (1e-300, 1e-300)])
def test_unrepresentable_nonzero_clock_product_rejects(time, capacity):
    with pytest.raises(ValueError, match=r"capacity \* structural_time"):
        cayley_diffusion_action(
            3, {1, 2}, [1, 1, 1], structural_time=time, capacity=capacity
        )


@pytest.mark.parametrize("options", [{"capacity": 0}, {"structural_time": 0}])
def test_frozen_state_does_not_require_an_overflowing_fourier_transform(options):
    vector = np.full(3, 1e308)
    evolved = cayley_diffusion_action(3, {1, 2}, vector, **options)
    np.testing.assert_array_equal(evolved, vector)
    assert not np.shares_memory(evolved, vector)


def test_nonfinite_fourier_intermediates_raise_instead_of_returning_nan():
    with pytest.raises(ValueError, match="Fourier intermediates"):
        cayley_diffusion_action(3, {1, 2}, [1e308] * 3)
    with pytest.raises(ValueError, match="Fourier intermediates"):
        cayley_diffusion_action(2, {1}, [1, -1], structural_time=1e308)


@pytest.mark.parametrize(
    "value",
    [
        True,
        np.bool_(False),
        "1",
        float("inf"),
        float("nan"),
        Fraction(1, 10**400),
        complex(1, float("inf")),
    ],
)
def test_vector_values_are_admitted_even_for_frozen_flow(value):
    with pytest.raises((TypeError, ValueError), match="vector"):
        cayley_diffusion_action(3, {1, 2}, [0, value, 0], capacity=0)


@pytest.mark.parametrize(
    "value", [True, np.bool_(True), 5.0, Fraction(5), float("nan")]
)
def test_cayley_modulus_is_an_integer_without_truncation(value):
    with pytest.raises(TypeError, match="modulus"):
        cayley_spectrum(value, {1})


@pytest.mark.parametrize("value", [True, np.bool_(True), 1.5, Fraction(3, 2), "1"])
def test_cayley_support_shifts_are_integers_without_truncation(value):
    with pytest.raises(TypeError, match="connection shift"):
        cayley_spectrum(5, {value})


def test_numpy_integer_support_normalizes_before_modular_arithmetic():
    assert cayley_spectrum(np.int64(5), {np.int64(1), np.int64(2)}) == cayley_spectrum(
        5, {1, 2}
    )
