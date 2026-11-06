"""Exact finite affine budgets and necessary two-level nodal return counts."""

import math
from dataclasses import FrozenInstanceError
from fractions import Fraction as F

import pytest

from tnfr.physics.nodal_remainder_pressure import (
    derive_nodal_area_crossings,
    derive_two_level_nodal_return,
)


def _cross(**changes):
    args = dict(
        initial_area=(F(-3),),
        timestep=0.5,
        capacity=(1.0,),
        pressure=(2.0,),
        max_steps=4,
    )
    args.update(changes)
    return derive_nodal_area_crossings(**args)


def test_exact_integer_repayment_retains_before_and_after_areas():
    result = _cross()
    coordinate = result.coordinates[0]
    assert result.exact_increment == (1,) and result.endpoint_area == (1,)
    assert coordinate.continuous_zero_step == coordinate.integer_zero_step == 3
    assert coordinate.first_zero_or_opposite_step == 3
    assert coordinate.area_before_crossing == -1 and coordinate.area_at_crossing == 0
    assert coordinate.crossing_is_exact_zero and coordinate.crossing_in_prefix
    assert coordinate.exact_zero_in_prefix
    assert result.positive_joint_zero_step == 3 and result.joint_zero_in_prefix
    assert (
        not result.pressure_provenance_certified
        and not result.band_provenance_certified
    )


@pytest.mark.parametrize("sign", [-1, 1])
def test_fractional_root_crosses_without_exact_integer_zero_in_both_directions(sign):
    result = _cross(initial_area=(sign * F(3, 2),), pressure=(-sign * 2.0,))
    coordinate = result.coordinates[0]
    assert coordinate.continuous_zero_step == F(3, 2)
    assert coordinate.integer_zero_step is None
    assert coordinate.first_zero_or_opposite_step == 2
    assert coordinate.area_before_crossing == sign * F(1, 2)
    assert coordinate.area_at_crossing == -sign * F(1, 2)
    assert not coordinate.crossing_is_exact_zero and not coordinate.exact_zero_in_prefix
    assert coordinate.crossing_in_prefix
    assert result.positive_joint_zero_step is None


def test_actual_b29_positive_level_requires_four_steps_but_declared_prefix_has_two():
    debt = -F(2437619387196587, 2**110)
    pressure = float(F(2786216251278205, 2**108))
    result = _cross(
        initial_area=(debt,), timestep=0.0625, pressure=(pressure,), max_steps=2
    )
    row = result.coordinates[0]
    assert result.exact_increment == (F(2786216251278205, 2**112),)
    assert row.continuous_zero_step == F(9750477548786348, 2786216251278205)
    assert row.first_zero_or_opposite_step == 4 and not row.crossing_in_prefix
    assert row.area_before_crossing == -F(1391828794951733, 2**112)
    assert row.area_at_crossing == F(1394387456326472, 2**112)
    assert row.integer_zero_step is None and not row.exact_zero_in_prefix
    assert result.endpoint_area == (-F(2089022523114969, 2**111),)
    assert result.positive_joint_zero_step is None


def test_candidates_are_retained_outside_the_declared_prefix():
    result = _cross(max_steps=2)
    row = result.coordinates[0]
    assert row.integer_zero_step == row.first_zero_or_opposite_step == 3
    assert not row.exact_zero_in_prefix and not row.crossing_in_prefix
    assert result.positive_joint_zero_step == 3 and not result.joint_zero_in_prefix
    assert result.endpoint_area == (-1,)


def test_initial_zero_is_not_a_later_repayment():
    result = _cross(initial_area=(F(0),))
    row = result.coordinates[0]
    assert row.continuous_zero_step == row.integer_zero_step == 0
    assert row.exact_zero_in_prefix and not row.zero_for_all_steps
    assert row.first_zero_or_opposite_step is None
    assert row.area_before_crossing is row.area_at_crossing is None
    assert result.initially_joint_zero
    assert (
        result.positive_joint_zero_step is None and not result.joint_zero_for_all_steps
    )


@pytest.mark.parametrize("maximum", [0, 1, 100])
def test_stationary_zero_has_all_zero_steps_and_first_positive_candidate_one(maximum):
    result = _cross(
        initial_area=(F(0), F(0)),
        timestep=0.0,
        capacity=(1.0, 2.0),
        pressure=(-3.0, 5.0),
        max_steps=maximum,
    )
    assert result.initially_joint_zero and result.joint_zero_for_all_steps
    assert result.positive_joint_zero_step == 1
    assert result.joint_zero_in_prefix == (maximum >= 1)
    for row in result.coordinates:
        assert row.zero_for_all_steps and row.integer_zero_step == 0
        assert row.continuous_zero_step is None
        assert row.exact_zero_in_prefix and row.first_zero_or_opposite_step is None


def test_joint_vector_roots_must_agree_in_every_coordinate():
    mismatch = _cross(
        initial_area=(F(-2), F(-3)), capacity=(1.0, 1.0), pressure=(2.0, 2.0)
    )
    assert tuple(row.integer_zero_step for row in mismatch.coordinates) == (2, 3)
    assert all(row.exact_zero_in_prefix for row in mismatch.coordinates)
    assert mismatch.positive_joint_zero_step is None
    assert not mismatch.joint_zero_in_prefix


def test_constant_zero_coordinate_does_not_constrain_a_common_root():
    result = _cross(
        initial_area=(F(-1, 2), F(1, 4), F(0)),
        timestep=0.25,
        capacity=(2.0, 1.0, 0.0),
        pressure=(0.5, -0.5, 100.0),
        max_steps=2,
    )
    assert result.exact_increment == (F(1, 4), -F(1, 8), 0)
    assert result.positive_joint_zero_step == 2 and result.joint_zero_in_prefix
    assert result.endpoint_area == (0, 0, 0)
    assert result.coordinates[2].zero_for_all_steps


def test_constant_nonzero_coordinate_prevents_joint_zero():
    result = _cross(
        initial_area=(F(-2), F(1)), capacity=(1.0, 0.0), pressure=(2.0, 4.0)
    )
    row = result.coordinates[1]
    assert row.continuous_zero_step is row.integer_zero_step is None
    assert row.first_zero_or_opposite_step is None
    assert result.positive_joint_zero_step is None


def test_moving_away_from_zero_retains_negative_continuous_root_only():
    result = _cross(pressure=(-2.0,))
    row = result.coordinates[0]
    assert row.continuous_zero_step == -3
    assert row.integer_zero_step is row.first_zero_or_opposite_step is None
    assert result.endpoint_area == (-7,)


def test_zero_mean_endpoint_is_not_vector_return():
    result = _cross(
        initial_area=(F(-3), F(3)),
        capacity=(1.0, 1.0),
        pressure=(2.0, -2.0),
        max_steps=2,
    )
    assert sum(result.endpoint_area) == 0 and any(result.endpoint_area)
    assert not result.joint_zero_in_prefix


def test_maximum_zero_retains_the_supplied_baseline():
    result = _cross(max_steps=0)
    assert result.endpoint_area == result.initial_area
    assert result.coordinates[0].first_zero_or_opposite_step == 3
    assert not result.coordinates[0].crossing_in_prefix


def test_affine_result_and_coordinates_are_frozen():
    result = _cross()
    with pytest.raises(FrozenInstanceError):
        result.max_steps = 100
    with pytest.raises(FrozenInstanceError):
        result.coordinates[0].integer_zero_step = 2


@pytest.mark.parametrize(
    "changes,error",
    [
        ({"initial_area": [F(-1)]}, TypeError),
        ({"initial_area": ()}, ValueError),
        ({"initial_area": (-1,)}, TypeError),
        ({"initial_area": (-1.0,)}, TypeError),
        ({"timestep": -1.0}, ValueError),
        ({"timestep": True}, TypeError),
        ({"timestep": math.inf}, ValueError),
        ({"timestep": math.nan}, ValueError),
        ({"capacity": [1.0]}, TypeError),
        ({"capacity": (-1.0,)}, ValueError),
        ({"capacity": (1,)}, TypeError),
        ({"capacity": (math.inf,)}, ValueError),
        ({"capacity": (1.0, 2.0)}, ValueError),
        ({"pressure": (math.nan,)}, ValueError),
        ({"pressure": (F(1),)}, TypeError),
        ({"pressure": ()}, ValueError),
        ({"max_steps": -1}, ValueError),
        ({"max_steps": True}, TypeError),
        ({"max_steps": 2.0}, TypeError),
    ],
)
def test_malformed_affine_inputs_are_rejected(changes, error):
    with pytest.raises(error):
        _cross(**changes)


def test_actual_two_level_primitive_counts_exclude_short_exact_returns():
    result = derive_two_level_nodal_return(
        negative_pressure=-float(F(128295757220873, 2**106)),
        positive_pressure=float(F(2786216251278205, 2**108)),
    )
    assert result.common_denominator == 2**108
    assert result.negative_integer == 513183028883492
    assert result.positive_integer == 2786216251278205
    assert result.gcd == 1
    assert result.minimum_negative_steps == 2786216251278205
    assert result.minimum_positive_steps == 513183028883492
    assert result.minimum_total_steps == 3299399280161697
    assert (
        result.minimum_negative_steps * F(result.negative_pressure)
        + result.minimum_positive_steps * F(result.positive_pressure)
    ) == 0
    assert (
        not result.pressure_provenance_certified
        and not result.periodic_execution_certified
    )


def test_two_level_gcd_removes_a_common_numerator_factor():
    result = derive_two_level_nodal_return(
        negative_pressure=-6.0, positive_pressure=9.0
    )
    assert (
        result.common_denominator,
        result.negative_integer,
        result.positive_integer,
    ) == (1, 6, 9)
    assert result.gcd == 3
    assert (
        result.minimum_negative_steps,
        result.minimum_positive_steps,
        result.minimum_total_steps,
    ) == (3, 2, 5)


def test_two_level_different_denominators_and_subnormal_pressures():
    result = derive_two_level_nodal_return(
        negative_pressure=-0.375, positive_pressure=0.25
    )
    assert (
        result.common_denominator,
        result.negative_integer,
        result.positive_integer,
    ) == (8, 3, 2)
    assert (result.minimum_negative_steps, result.minimum_positive_steps) == (2, 3)
    tiny = math.ulp(0.0)
    result = derive_two_level_nodal_return(
        negative_pressure=-2 * tiny, positive_pressure=3 * tiny
    )
    assert result.common_denominator == 2**1074
    assert result.gcd == 1
    assert (result.minimum_negative_steps, result.minimum_positive_steps) == (3, 2)


def test_an_additional_pressure_level_invalidates_the_two_level_length_restriction():
    result = derive_two_level_nodal_return(
        negative_pressure=-0.25, positive_pressure=0.5
    )
    assert result.minimum_total_steps == 3
    assert (
        F(-0.25) + F(0.25) == 0
    )  # A new level permits two steps, outside this certificate.


@pytest.mark.parametrize(
    "negative,positive,error",
    [
        (0.0, 1.0, ValueError),
        (-0.0, 1.0, ValueError),
        (1.0, 1.0, ValueError),
        (-1.0, 0.0, ValueError),
        (-1.0, -1.0, ValueError),
        (-1, 1.0, TypeError),
        (-1.0, 1, TypeError),
        (False, 1.0, TypeError),
        (-math.inf, 1.0, ValueError),
        (-1.0, math.nan, ValueError),
    ],
)
def test_two_level_invalid_inputs_are_rejected(negative, positive, error):
    with pytest.raises(error):
        derive_two_level_nodal_return(
            negative_pressure=negative, positive_pressure=positive
        )


def test_two_level_result_is_frozen():
    result = derive_two_level_nodal_return(
        negative_pressure=-0.25, positive_pressure=0.5
    )
    with pytest.raises(FrozenInstanceError):
        result.minimum_total_steps = 2
