"""Sharp dyadic-grid and carry-composition controls for nodal itineraries."""

import math
from dataclasses import FrozenInstanceError, replace
from fractions import Fraction as F

import pytest

from tnfr.dynamics._euler_kernel import NodalRemainderState, advance_nodal_remainder
from tnfr.physics import nodal_remainder as owner

MIN_SUBNORMAL = math.ulp(0.0)
GRID = F(1, 2**3222)
ULP = 2.0**-53
ODD = 0.5 + ULP
EVEN = 0.5 + 2 * ULP
NEXT_ODD = 0.5 + 3 * ULP
BOUNDARY = F(0.5) + 3 * F(ULP) / 2


def _derive(states, timesteps, capacities, pressures, **kwargs):
    return owner.derive_nodal_remainder_itinerary(
        epi_states=tuple((value,) for value in states),
        timesteps=timesteps,
        capacities=tuple((value,) for value in capacities),
        pressures=tuple((value,) for value in pressures),
        **kwargs,
    )


def test_real_open_interval_can_have_no_legal_remainder_grid_point():
    result = _derive(
        (ODD, EVEN, NEXT_ODD),
        (1.0, MIN_SUBNORMAL),
        (1.0, MIN_SUBNORMAL),
        (ULP, MIN_SUBNORMAL),
    )
    (cell,) = result.coordinates
    assert result.cumulative_nodal_area == ((F(0),), (F(ULP),), (F(ULP) + GRID,))
    assert cell.lower == BOUNDARY - GRID < cell.upper == BOUNDARY
    assert not cell.lower_closed and not cell.upper_closed
    assert cell.first_grid_index == cell.last_grid_index + 1
    assert not result.feasible and not result.zero_initial_carry_feasible
    assert result.witness_initial is None and result.witness_sequence is None
    # There really is a real-valued witness, but its denominator exceeds
    # the existing solver representation, so it must not be certified.
    midpoint = (cell.lower + cell.upper) / 2
    assert midpoint.denominator == 2**3223
    assert float(midpoint) == ODD
    assert float(midpoint + F(ULP)) == EVEN
    assert float(midpoint + F(ULP) + GRID) == NEXT_ODD
    invalid = NodalRemainderState((ODD,), (midpoint - F(ODD),), 0.05, 1.0)
    with pytest.raises(ValueError, match="dyadic"):
        _ = invalid.exact_epi


def test_closed_singleton_at_the_full_denominator_limit_is_feasible():
    width = 3 * 2.0**-55
    result = _derive(
        (0.5, 0.5, 0.5),
        (MIN_SUBNORMAL, 1.0),
        (MIN_SUBNORMAL, 1.0),
        (-MIN_SUBNORMAL, width),
    )
    exact_initial = F(0.5) - F(1, 2**55) + GRID
    (cell,) = result.coordinates
    assert cell.lower == cell.upper == exact_initial
    assert cell.lower_closed and cell.upper_closed
    assert cell.first_grid_index == cell.last_grid_index
    assert result.feasible and not result.zero_initial_carry_feasible
    assert result.witness_initial.exact_epi == (exact_initial,)
    assert result.witness_initial.remainder[0].denominator == 2**3222
    assert result.witness_sequence.steps[0].after.exact_epi == (F(0.5) - F(1, 2**55),)
    assert result.witness_sequence.endpoint.exact_epi == (F(0.5) + F(1, 2**54),)
    assert result.visible_closed and not result.conditional_carried_cycle


def test_closed_tiny_area_cycle_can_require_nonzero_carry_without_source_provenance():
    result = _derive(
        (ODD, EVEN, ODD),
        (MIN_SUBNORMAL,) * 2,
        (MIN_SUBNORMAL,) * 2,
        (MIN_SUBNORMAL, -MIN_SUBNORMAL),
    )
    (cell,) = result.coordinates
    assert cell.lower == BOUNDARY - GRID and cell.upper == BOUNDARY
    assert cell.lower_closed and not cell.upper_closed
    assert cell.first_grid_index == cell.last_grid_index
    assert result.feasible and result.conditional_carried_cycle
    assert not result.zero_initial_carry_feasible
    assert result.total_nodal_area == (F(0),)
    assert result.witness_initial.exact_epi == (BOUNDARY - GRID,)
    assert result.witness_sequence.endpoint == result.witness_initial
    assert result.pressure_provenance_certified is False
    assert result.runtime_provenance_certified is False
    # Repeating the same supplied inputs verifies the conditional numerical
    # cycle; it adds no pressure-generation or graph-reachability evidence.
    second = owner.observe_nodal_remainder_sequence(
        initial=result.witness_sequence.endpoint,
        timesteps=result.timesteps,
        capacities=result.capacities,
        pressures=result.pressures,
    )
    assert second.endpoint == result.witness_initial


@pytest.mark.parametrize(
    "value,pressure,lower,upper,expected",
    (
        (1.0, MIN_SUBNORMAL, 0.05, 1.0, F(1) - GRID),
        (0.5, -MIN_SUBNORMAL, 0.5, 1.0, F(0.5) + GRID),
    ),
)
def test_exact_band_constraints_override_invisible_rounding_changes(
    value, pressure, lower, upper, expected
):
    result = _derive(
        (value, value),
        (MIN_SUBNORMAL,),
        (MIN_SUBNORMAL,),
        (pressure,),
        epi_lower=lower,
        epi_upper=upper,
    )
    assert result.feasible and not result.zero_initial_carry_feasible
    assert result.witness_initial.exact_epi == (expected,)
    initial = NodalRemainderState((value,), (F(0),), lower, upper)
    with pytest.raises(ValueError, match="exact nodal update"):
        advance_nodal_remainder(
            initial,
            timestep=MIN_SUBNORMAL,
            capacity=(MIN_SUBNORMAL,),
            pressure=(pressure,),
        )
    assert result.witness_sequence.endpoint.epi == (value,)


def test_intermediate_exact_band_failure_cannot_be_hidden_by_zero_final_area():
    result = _derive(
        (1.0, 1.0, 1.0),
        (MIN_SUBNORMAL,) * 2,
        (MIN_SUBNORMAL,) * 2,
        (MIN_SUBNORMAL, -MIN_SUBNORMAL),
        epi_lower=1.0,
        epi_upper=1.0,
    )
    assert result.visible_closed and result.total_nodal_area == (F(0),)
    assert not result.feasible and not result.conditional_carried_cycle
    assert result.witness_initial is None


def test_one_infeasible_coordinate_blocks_a_joint_witness():
    result = owner.derive_nodal_remainder_itinerary(
        epi_states=((0.5, ODD), (0.5, EVEN), (0.5, NEXT_ODD)),
        timesteps=(1.0, MIN_SUBNORMAL),
        capacities=((1.0, 1.0), (MIN_SUBNORMAL, MIN_SUBNORMAL)),
        pressures=((0.0, ULP), (0.0, MIN_SUBNORMAL)),
    )
    assert result.coordinates[0].feasible and not result.coordinates[1].feasible
    assert not result.feasible and not result.conditional_carried_cycle
    assert result.witness_initial is None and result.witness_sequence is None


def test_detached_result_flags_do_not_authenticate_forged_arithmetic_inputs():
    valid = _derive((0.5, 0.5), (1.0,), (1.0,), (0.0,))
    assert valid.feasible and valid.conditional_carried_cycle
    forged = replace(valid, pressures=((1.0,),))
    assert forged.pressure_provenance_certified is False
    assert forged.runtime_provenance_certified is False
    fresh = owner.derive_nodal_remainder_itinerary(
        epi_states=forged.epi_states,
        timesteps=forged.timesteps,
        capacities=forged.capacities,
        pressures=forged.pressures,
        epi_lower=forged.epi_lower,
        epi_upper=forged.epi_upper,
    )
    assert not fresh.feasible and not fresh.conditional_carried_cycle
    with pytest.raises(FrozenInstanceError):
        valid.feasible = False


class FloatSubclass(float):
    pass


class TupleSubclass(tuple):
    pass


@pytest.mark.parametrize(
    "changes",
    (
        {"epi_states": TupleSubclass(((0.5,), (0.5,)))},
        {"epi_states": (TupleSubclass((0.5,)), (0.5,))},
        {"epi_states": ((FloatSubclass(0.5),), (0.5,))},
        {"timesteps": (FloatSubclass(1.0),)},
        {"capacities": ((FloatSubclass(1.0),),)},
        {"pressures": ((FloatSubclass(0.0),),)},
        {"epi_lower": FloatSubclass(0.05)},
    ),
)
def test_user_numeric_or_container_subclasses_cannot_bypass_strict_primitive_validation(
    changes,
):
    inputs = dict(
        epi_states=((0.5,), (0.5,)),
        timesteps=(1.0,),
        capacities=((1.0,),),
        pressures=((0.0,),),
    )
    inputs.update(changes)
    with pytest.raises(TypeError):
        owner.derive_nodal_remainder_itinerary(**inputs)
