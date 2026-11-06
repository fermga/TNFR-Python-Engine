"""Exact affine-area crossings must survive float underflow and false zeroes."""

import math
import sys
from fractions import Fraction as F

import pytest

from tnfr.physics.nodal_remainder_pressure import derive_nodal_area_crossings

MIN_SUBNORMAL = math.ulp(0.0)
GRID = F(1, 2**3222)


def _derive(initial, *, timestep=1.0, capacity=1.0, pressure=1.0, max_steps=1):
    return derive_nodal_area_crossings(
        initial_area=(initial,),
        timestep=timestep,
        capacity=(capacity,),
        pressure=(pressure,),
        max_steps=max_steps,
    )


@pytest.mark.parametrize("direction", (-1, 1))
@pytest.mark.parametrize("offset", (-1, 1))
def test_binary64_cancellation_cannot_create_a_false_exact_return(direction, offset):
    initial = direction * (F(1) + offset * F(1, 2**55))
    assert float(initial) == direction
    assert float(initial) - direction == 0.0
    observed = _derive(initial, pressure=float(-direction))
    (crossing,) = observed.coordinates
    root = F(1) + offset * F(1, 2**55)
    assert observed.initial_area == (initial,)
    assert observed.endpoint_area == (direction * offset * F(1, 2**55),)
    assert crossing.continuous_zero_step == root
    assert crossing.integer_zero_step is None
    assert crossing.first_zero_or_opposite_step == (1 if offset < 0 else 2)
    assert crossing.crossing_in_prefix is (offset < 0)
    assert not crossing.crossing_is_exact_zero
    assert not crossing.exact_zero_in_prefix
    assert observed.positive_joint_zero_step is None


@pytest.mark.parametrize("direction", (-1, 1))
def test_underflowing_exact_baseline_is_nonzero_and_can_be_repaid(direction):
    initial = direction * GRID
    assert float(initial) == 0.0
    observed = _derive(
        initial,
        timestep=MIN_SUBNORMAL,
        capacity=MIN_SUBNORMAL,
        pressure=-direction * MIN_SUBNORMAL,
    )
    (coordinate,) = observed.coordinates
    assert observed.exact_increment == (-direction * GRID,)
    assert not observed.initially_joint_zero
    assert coordinate.continuous_zero_step == 1
    assert coordinate.integer_zero_step == coordinate.first_zero_or_opposite_step == 1
    assert (
        coordinate.area_before_crossing == initial and coordinate.area_at_crossing == 0
    )
    assert coordinate.crossing_is_exact_zero and coordinate.exact_zero_in_prefix
    assert observed.positive_joint_zero_step == 1 and observed.joint_zero_in_prefix


def test_non_dyadic_supplied_area_is_algebraic_input_not_provenance():
    observed = _derive(-F(1, 3), pressure=0.25, max_steps=2)
    (coordinate,) = observed.coordinates
    assert coordinate.continuous_zero_step == F(4, 3)
    assert coordinate.integer_zero_step is None
    assert coordinate.first_zero_or_opposite_step == 2
    assert coordinate.area_before_crossing == -F(1, 12)
    assert coordinate.area_at_crossing == F(1, 6)
    assert coordinate.crossing_in_prefix and not coordinate.crossing_is_exact_zero
    assert observed.positive_joint_zero_step is None
    assert observed.pressure_provenance_certified is False
    assert observed.band_provenance_certified is False


def test_astronomical_crossing_index_is_solved_without_iterating_the_prefix():
    root = 2**3222
    before = _derive(
        -F(1),
        timestep=MIN_SUBNORMAL,
        capacity=MIN_SUBNORMAL,
        pressure=MIN_SUBNORMAL,
        max_steps=root - 1,
    )
    (coordinate,) = before.coordinates
    assert coordinate.continuous_zero_step == root
    assert (
        coordinate.integer_zero_step == coordinate.first_zero_or_opposite_step == root
    )
    assert not coordinate.crossing_in_prefix and not coordinate.exact_zero_in_prefix
    assert coordinate.area_before_crossing == -GRID and coordinate.area_at_crossing == 0
    assert before.endpoint_area == (-GRID,)
    assert before.positive_joint_zero_step == root and not before.joint_zero_in_prefix
    at = _derive(
        -F(1),
        timestep=MIN_SUBNORMAL,
        capacity=MIN_SUBNORMAL,
        pressure=MIN_SUBNORMAL,
        max_steps=root,
    )
    assert at.endpoint_area == (F(0),)
    assert at.coordinates[0].crossing_in_prefix and at.joint_zero_in_prefix


def test_finite_factors_with_float_overflow_keep_their_exact_nodal_product():
    maximum = sys.float_info.max
    assert math.isinf(maximum * maximum)
    increment = F(maximum) ** 3
    observed = _derive(-increment, timestep=maximum, capacity=maximum, pressure=maximum)
    assert observed.exact_increment == (increment,)
    assert observed.endpoint_area == (F(0),)
    assert observed.coordinates[0].integer_zero_step == 1
    assert observed.joint_zero_in_prefix
    assert observed.band_provenance_certified is False


def test_roots_that_round_to_the_same_float_do_not_establish_a_joint_return():
    separation = F(1, 2**55)
    result = derive_nodal_area_crossings(
        initial_area=(-F(1), -F(1) - separation),
        timestep=1.0,
        capacity=(1.0, 1.0),
        pressure=(1.0, 1.0),
        max_steps=2,
    )
    roots = tuple(coordinate.continuous_zero_step for coordinate in result.coordinates)
    assert roots == (F(1), F(1) + separation)
    assert tuple(map(float, roots)) == (1.0, 1.0)
    assert result.coordinates[0].integer_zero_step == 1
    assert result.coordinates[1].integer_zero_step is None
    assert result.positive_joint_zero_step is None and not result.joint_zero_in_prefix


def test_initial_vector_zero_is_distinct_from_any_later_common_return():
    result = derive_nodal_area_crossings(
        initial_area=(F(0), F(0)),
        timestep=1.0,
        capacity=(1.0, 1.0),
        pressure=(MIN_SUBNORMAL, -MIN_SUBNORMAL),
        max_steps=2**1074,
    )
    assert result.initially_joint_zero and not result.joint_zero_for_all_steps
    assert all(item.exact_zero_in_prefix for item in result.coordinates)
    assert all(item.integer_zero_step == 0 for item in result.coordinates)
    assert all(item.first_zero_or_opposite_step is None for item in result.coordinates)
    assert result.positive_joint_zero_step is None and not result.joint_zero_in_prefix
    assert result.endpoint_area == (F(1), -F(1))


def test_a_tiny_nonzero_stationary_coordinate_blocks_every_joint_zero():
    result = derive_nodal_area_crossings(
        initial_area=(-F(1), GRID),
        timestep=1.0,
        capacity=(1.0, 0.0),
        pressure=(1.0, 1.0),
        max_steps=1,
    )
    assert result.coordinates[0].exact_zero_in_prefix
    assert result.coordinates[1].continuous_zero_step is None
    assert not result.coordinates[1].zero_for_all_steps
    assert result.endpoint_area == (F(0), GRID)
    assert result.positive_joint_zero_step is None and not result.joint_zero_in_prefix
