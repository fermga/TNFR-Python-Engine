"""Independent arithmetic and scope controls for finite pressure alphabets."""

from fractions import Fraction as F
import math
import sys

import pytest

from tnfr.physics.nodal_remainder_pressure import derive_finite_level_nodal_return


@pytest.mark.parametrize("exponent", [-1074, -1022, -100, 0, 100, 1020])
def test_exact_common_scaling_preserves_the_return_restriction(exponent):
    scale = math.ldexp(1.0, exponent)
    levels = tuple(value * scale for value in (-2., 3., 8.))
    result = derive_finite_level_nodal_return(pressure_levels=levels)
    assert all(F(value) == integer * F(scale) for value, integer in zip(levels, (-2, 3, 8), strict=True))
    assert result.necessary_length_multiple == 5
    # This formal five-entry witness is independent of the chosen float scale.
    assert 3 * F(levels[0]) + 2 * F(levels[1]) == 0
    assert result.arithmetic_zero_area_possible
    assert not result.periodic_execution_certified


@pytest.mark.parametrize("extra,expected_denominator,expected_modulus", [
    (.5, 2, 5),
    (.25, 4, 1),
    (math.ldexp(1., -100), 2**100, 1),
    (math.ulp(0.), 2**1074, 1),
])
def test_finer_denominator_superset_can_retain_or_weaken_the_modulus(
    extra, expected_denominator, expected_modulus,
):
    original = derive_finite_level_nodal_return(pressure_levels=(-2., 3.))
    extended = derive_finite_level_nodal_return(pressure_levels=(-2., 3., extra))
    assert original.necessary_length_multiple == 5
    assert extended.common_denominator == expected_denominator
    assert extended.necessary_length_multiple == expected_modulus
    assert original.necessary_length_multiple % extended.necessary_length_multiple == 0
    assert tuple(F(z, extended.common_denominator) for z in extended.integer_levels) == tuple(
        F(value) for value in (-2., 3., extra)
    )
    # Refinement changes the integer encoding, not any old pressure or witness.
    assert 3 * F(-2.) + 2 * F(3.) == 0


def test_superset_can_remove_only_part_of_a_nonprime_length_restriction():
    pair = derive_finite_level_nodal_return(pressure_levels=(-4., 14.))
    triple = derive_finite_level_nodal_return(pressure_levels=(-4., 14., 26.))
    assert pair.necessary_length_multiple == 9
    assert triple.necessary_length_multiple == 3
    assert pair.necessary_length_multiple // triple.necessary_length_multiple == 3
    # A new congruence does not force a short feasible word or an engine return.
    assert 7 * F(-4.) + 2 * F(14.) == 0
    assert not triple.pressure_provenance_certified


@pytest.mark.parametrize("sign", [-1, 1])
def test_negative_anchor_residue_requires_the_full_positive_divisor(sign):
    levels = tuple(float(sign * value) for value in (-7, 5, 17))
    result = derive_finite_level_nodal_return(pressure_levels=levels)
    assert result.reference_integer == -7 * sign
    assert result.difference_gcd == 12
    assert result.residue_gcd == 1
    assert result.necessary_length_multiple == 12
    assert 5 * F(levels[0]) + 7 * F(levels[1]) == 0


def test_modulus_one_neither_promises_length_one_nor_joint_return():
    result = derive_finite_level_nodal_return(pressure_levels=(-2., 3., 4.))
    assert result.necessary_length_multiple == 1
    assert result.arithmetic_zero_area_possible
    assert all(value != 0 for value in result.integer_levels)
    assert F(-2.) + F(-2.) + F(4.) == 0
    # Those three scalar readings could accompany strictly positive other rows.
    assert sum((F(1), F(1), F(1))) != 0
    assert not result.periodic_execution_certified


def test_extreme_denominator_and_magnitude_do_not_use_float_period_arithmetic():
    tiny, largest = math.ulp(0.), sys.float_info.max
    result = derive_finite_level_nodal_return(pressure_levels=(-tiny, largest))
    positive_integer = F(largest) / F(tiny)
    assert positive_integer.denominator == 1
    assert positive_integer.numerator.bit_length() > 2000
    assert result.common_denominator == 2**1074
    assert result.integer_levels == (-1, positive_integer.numerator)
    assert result.necessary_length_multiple == positive_integer.numerator + 1
    assert positive_integer * F(-tiny) + F(largest) == 0
    assert result.arithmetic_zero_area_possible


def test_near_cancelling_binary64_levels_have_large_exact_restriction():
    above_one = math.nextafter(1., math.inf)
    result = derive_finite_level_nodal_return(pressure_levels=(-1., above_one))
    assert result.integer_levels == (-2**52, 2**52 + 1)
    assert result.necessary_length_multiple == 2**53 + 1
    assert F(-1.) + F(above_one) == F(1, 2**52)
    assert (2**52 + 1) * F(-1.) + 2**52 * F(above_one) == 0
    # No tolerance turns an almost-balanced pair into an exact return.
    assert result.necessary_length_multiple != 2
