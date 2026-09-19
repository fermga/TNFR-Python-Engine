"""Necessary return-length divisibility for exact finite pressure levels."""

import math
from dataclasses import FrozenInstanceError
from fractions import Fraction as F
from itertools import permutations, product

import pytest

from tnfr.physics.nodal_remainder_pressure import (
    derive_finite_level_nodal_return,
    derive_two_level_nodal_return,
)


def test_actual_third_level_retains_the_two_level_modulus():
    levels = (
        -float(F(128295757220873, 2**106)),
        float(F(2786216251278205, 2**108)),
        -float(F(4325765337928681, 2**109)),
    )
    result = derive_finite_level_nodal_return(pressure_levels=levels)
    assert result.pressure_levels == levels
    assert result.common_denominator == 2**109
    assert result.integer_levels == (
        -1026366057766984,
        5572432502556410,
        -4325765337928681,
    )
    assert result.reference_integer == -1026366057766984
    assert result.difference_gcd == 3299399280161697
    assert result.residue_gcd == 1
    assert result.necessary_length_multiple == 3299399280161697
    assert result.has_negative and result.has_positive and not result.has_zero
    assert result.arithmetic_zero_area_possible
    assert not result.pressure_provenance_certified
    assert not result.periodic_execution_certified


@pytest.mark.parametrize(
    "negative,positive",
    [
        (-2.0, 3.0),
        (-6.0, 9.0),
        (-0.375, 0.25),
        (-0.125, 1.5),
        (-2 * math.ulp(0.0), 3 * math.ulp(0.0)),
    ],
)
def test_two_level_payload_and_primitive_length_agree(negative, positive):
    old = derive_two_level_nodal_return(
        negative_pressure=negative, positive_pressure=positive
    )
    general = derive_finite_level_nodal_return(pressure_levels=(negative, positive))
    assert general.common_denominator == old.common_denominator
    assert general.integer_levels == (-old.negative_integer, old.positive_integer)
    assert general.necessary_length_multiple == old.minimum_total_steps
    assert general.arithmetic_zero_area_possible
    assert (
        old.minimum_negative_steps * F(negative)
        + old.minimum_positive_steps * F(positive)
        == 0
    )


def test_three_level_congruence_is_not_a_sufficient_short_return_condition():
    levels = (-4.0, 14.0, 26.0)
    result = derive_finite_level_nodal_return(pressure_levels=levels)
    assert result.difference_gcd == 6 and result.residue_gcd == 2
    assert result.necessary_length_multiple == 3
    assert result.arithmetic_zero_area_possible
    assert all(sum(word) != 0 for word in product((-4, 14, 26), repeat=3))
    assert (
        7 * F(-4) + 2 * F(14) == 0
    )  # A formal nine-entry multiset, not a runtime itinerary.


@pytest.mark.parametrize(
    "levels", [(-2.0, 3.0, 8.0), (-0.5, 0.75, 2.0), (-4.0, 14.0, 26.0)]
)
def test_permutation_and_duplicate_levels_do_not_change_the_modulus(levels):
    expected = derive_finite_level_nodal_return(pressure_levels=levels)
    for order in permutations(levels):
        result = derive_finite_level_nodal_return(pressure_levels=order + (order[0],))
        assert result.pressure_levels == order + (order[0],)
        assert result.common_denominator == expected.common_denominator
        assert result.difference_gcd == expected.difference_gcd
        assert result.necessary_length_multiple == expected.necessary_length_multiple
        assert (
            result.arithmetic_zero_area_possible
            == expected.arithmetic_zero_area_possible
        )


@pytest.mark.parametrize("levels", [(0.0,), (0.0, -0.0, 0.0)])
def test_equal_zero_levels_admit_every_length(levels):
    result = derive_finite_level_nodal_return(pressure_levels=levels)
    assert result.common_denominator == 1
    assert result.integer_levels == (0,) * len(levels)
    assert result.difference_gcd == result.residue_gcd == 0
    assert result.necessary_length_multiple == 1
    assert result.has_zero and not result.has_positive and not result.has_negative
    assert result.arithmetic_zero_area_possible


@pytest.mark.parametrize("level", [0.5, -0.75, math.ulp(0.0)])
def test_constant_nonzero_level_never_returns(level):
    result = derive_finite_level_nodal_return(pressure_levels=(level, level))
    assert result.difference_gcd == 0
    assert result.residue_gcd == abs(result.reference_integer)
    assert result.necessary_length_multiple is None
    assert not result.has_zero and not result.arithmetic_zero_area_possible


@pytest.mark.parametrize("sign", [-1, 1])
def test_unequal_one_sign_levels_are_impossible_even_with_nontrivial_congruence(sign):
    result = derive_finite_level_nodal_return(
        pressure_levels=tuple(float(sign * value) for value in (2, 5, 8))
    )
    assert result.difference_gcd == 3
    assert result.necessary_length_multiple == 3
    assert result.has_negative == (sign < 0)
    assert result.has_positive == (sign > 0)
    assert not result.has_zero and not result.arithmetic_zero_area_possible


def test_selectable_zero_allows_every_length_without_balancing_nonzero_levels():
    result = derive_finite_level_nodal_return(pressure_levels=(6.0, 0.0, 15.0))
    assert result.has_positive and result.has_zero and not result.has_negative
    assert result.arithmetic_zero_area_possible
    assert result.difference_gcd == 3 and result.residue_gcd == 3
    assert result.necessary_length_multiple == 1


def test_opposite_sign_possibility_is_not_the_average_of_listed_levels():
    result = derive_finite_level_nodal_return(pressure_levels=(-1.0, 100.0, 100.0))
    assert sum(result.integer_levels) > 0
    assert result.arithmetic_zero_area_possible
    assert result.necessary_length_multiple == 101


def test_subnormal_three_level_integerization_does_not_underflow():
    tiny = math.ulp(0.0)
    result = derive_finite_level_nodal_return(
        pressure_levels=(-2 * tiny, 3 * tiny, 8 * tiny)
    )
    assert result.common_denominator == 2**1074
    assert result.integer_levels == (-2, 3, 8)
    assert result.necessary_length_multiple == 5
    assert 3 * F(-2 * tiny) + 2 * F(3 * tiny) == 0


def test_every_short_formal_zero_word_obeys_the_necessary_congruence():
    # A bounded arithmetic control; it neither executes nor prescribes graph pressure.
    result = derive_finite_level_nodal_return(pressure_levels=(-2.0, 3.0, 8.0))
    observed_lengths = set()
    for length in range(1, 7):
        for word in product((-2, 3, 8), repeat=length):
            if sum(word) == 0:
                observed_lengths.add(length)
                assert length % result.necessary_length_multiple == 0
    assert observed_lengths == {5}


def test_public_result_is_frozen():
    result = derive_finite_level_nodal_return(pressure_levels=(-2.0, 3.0, 8.0))
    with pytest.raises(FrozenInstanceError):
        result.necessary_length_multiple = 1


class FloatSubclass(float):
    pass


@pytest.mark.parametrize(
    "levels,error",
    [
        ([], TypeError),
        ([-1.0, 1.0], TypeError),
        ({-1.0, 1.0}, TypeError),
        ((), ValueError),
        ((-1, 1.0), TypeError),
        ((False, 1.0), TypeError),
        ((F(-1), 1.0), TypeError),
        ((FloatSubclass(-1), 1.0), TypeError),
        ((-1.0, math.nan), ValueError),
        ((-math.inf, 1.0), ValueError),
        ((-1.0, math.inf), ValueError),
    ],
)
def test_invalid_pressure_inputs_are_rejected(levels, error):
    with pytest.raises(error):
        derive_finite_level_nodal_return(pressure_levels=levels)
