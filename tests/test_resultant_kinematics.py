"""Instantaneous resultant chain-rule bounds, without any dynamical execution."""

from fractions import Fraction as Q
from math import factorial, pi

import pytest

from tnfr.mathematics._phase_resultant_chamber import relative_resultant_rate_bounds


def _trig_reference(angle, *, sine):
    """Direct degree81 Taylor/Lagrange enclosure without argument reduction."""
    partial = sum(
        (
            Q((-1) ** k) * angle ** (2 * k + sine) / factorial(2 * k + sine)
            for k in range(41)
        ),
        Q(0),
    )
    remainder = abs(angle) ** 82 / factorial(82)
    return partial - remainder, partial + remainder


def _reference_rate(phases, neighbors, rates):
    """Differentiate the direct Taylor series with independent remainder bounds."""
    result = []
    for i, row in enumerate(neighbors):
        parts = []
        for sine in (True, False):
            lower = upper = Q(0)
            for j in row:
                coefficient = (rates[j] - rates[i]) * (-1 if sine else 1)
                left, right = _trig_reference(phases[j] - phases[i], sine=sine)
                lower += min(coefficient * left, coefficient * right)
                upper += max(coefficient * left, coefficient * right)
            parts.append((lower, upper))
        result.append(tuple(parts))
    return tuple(result)


@pytest.mark.parametrize("common", (Q(0), Q(7, 3), Q(-(10**100))))
def test_common_angular_rate_gives_exact_zero_even_with_unresolved_angles(common):
    phases = (Q(0), Q(1 << 5000), Q(-1, 3))
    neighbors = ((1, 2), (0,), ())
    result = relative_resultant_rate_bounds(phases, neighbors, (common,) * 3)
    assert result == (((Q(0), Q(0)), (Q(0), Q(0))),) * 3


@pytest.mark.parametrize("angle", (Q(1, 3), Q(-2), Q(pi)))
def test_signed_rate_bounds_contain_independent_direct_taylor_derivative(angle):
    phases = (Q(0), angle, Q(1, 8))
    neighbors = ((1, 2), (0, 2), (0,))
    rates = (Q(-3, 2), Q(23), Q(-7))
    actual = relative_resultant_rate_bounds(phases, neighbors, rates)
    reference = _reference_rate(phases, neighbors, rates)
    for row, expected_row in zip(actual, reference):
        for (lower, upper), (left, right) in zip(row, expected_row):
            assert lower <= left <= right <= upper


def test_large_signed_derivatives_are_not_clamped_like_trigonometric_values():
    result = relative_resultant_rate_bounds(
        (Q(0), Q(0), Q(0)), ((1, 2), (0,), (0,)), (Q(0), Q(10), Q(-30))
    )
    assert result == (
        ((Q(0), Q(0)), (Q(-20), Q(-20))),
        ((Q(0), Q(0)), (Q(-10), Q(-10))),
        ((Q(0), Q(0)), (Q(30), Q(30))),
    )


def test_simultaneous_phase_and_rate_reflection_conjugates_the_derivative():
    phases = (Q(-2), Q(1, 4), Q(1))
    neighbors = ((1, 2), (0, 2), (0, 1))
    rates = (Q(3), Q(-1, 2), Q(4, 5))
    original = relative_resultant_rate_bounds(phases, neighbors, rates)
    reflected = relative_resultant_rate_bounds(
        tuple(-value for value in phases), neighbors, tuple(-value for value in rates)
    )
    assert reflected == tuple((real, (-imag[1], -imag[0])) for real, imag in original)


def test_common_phase_and_rate_offsets_preserve_relative_derivatives():
    phases, rates = (Q(0), Q(3, 2)), (Q(-1), Q(5))
    neighbors = ((1,), (0,))
    expected = relative_resultant_rate_bounds(phases, neighbors, rates)
    assert (
        relative_resultant_rate_bounds(
            tuple(value + 10**40 for value in phases),
            neighbors,
            tuple(value - 10**40 for value in rates),
        )
        == expected
    )


def test_represented_pi_keeps_its_nonzero_sine_derivative():
    result = relative_resultant_rate_bounds((Q(0), Q(pi)), ((1,), (0,)), (Q(0), Q(1)))
    # The exact binary64 value is below mathematical pi. The real derivative
    # is strictly negative at both ends, unlike rounding this to an antipode.
    assert all(real[1] < 0 for real, _ in result)
    assert result[0][1][1] < Q(-99, 100)
    assert result[1][1][0] > Q(99, 100)


def test_empty_neighbor_row_has_zero_derivative_for_arbitrary_rate():
    assert relative_resultant_rate_bounds((Q(2),), ((),), (Q(37),)) == (
        ((Q(0), Q(0)), (Q(0), Q(0))),
    )


def test_unresolved_trigonometry_preserves_a_conservative_rate_enclosure():
    result = relative_resultant_rate_bounds(
        (Q(0), Q(1 << 5000)), ((1,), (0,)), (Q(0), Q(7))
    )
    assert result == (((Q(-7), Q(7)), (Q(-7), Q(7))),) * 2


@pytest.mark.parametrize("invalid", (True, 0, 1.5, "1", None))
def test_rates_require_exact_fractions_without_coercion(invalid):
    with pytest.raises(TypeError, match="Fractions"):
        relative_resultant_rate_bounds((Q(0), Q(1)), ((1,), (0,)), (Q(0), invalid))


@pytest.mark.parametrize("rates", ((), (Q(1),), (Q(1), Q(2), Q(3))))
def test_rate_count_must_match_phase_coordinates(rates):
    with pytest.raises(ValueError, match="match"):
        relative_resultant_rate_bounds((Q(0), Q(1)), ((1,), (0,)), rates)


@pytest.mark.parametrize("rates", ({Q(1), Q(2)}, {0: Q(1), 1: Q(2)}, "12", None))
def test_rates_require_ordered_input(rates):
    with pytest.raises(TypeError, match="ordered iterable"):
        relative_resultant_rate_bounds((Q(0), Q(1)), ((1,), (0,)), rates)


@pytest.mark.parametrize(
    "phases,neighbors,error",
    (
        ((Q(0), 1.0), ((1,), (0,)), TypeError),
        ((Q(0), Q(1)), ((1, 1), (0,)), ValueError),
        ((Q(0), Q(1)), ((0,), (0,)), ValueError),
        ((Q(0), Q(1)), ((True,), (0,)), ValueError),
        ((Q(0), Q(1)), ((2,), (0,)), ValueError),
        ((Q(0), Q(1)), ((1,),), ValueError),
    ),
)
def test_shared_phase_support_admission_is_preserved(phases, neighbors, error):
    with pytest.raises(error):
        relative_resultant_rate_bounds(phases, neighbors, (Q(0), Q(0)))
