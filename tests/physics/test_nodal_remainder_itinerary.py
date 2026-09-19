"""Exact visible-word feasibility without resetting the numerical carry."""

import math
from dataclasses import FrozenInstanceError
from fractions import Fraction as F

import pytest

from tnfr.dynamics._euler_kernel import NODAL_REMAINDER_DENOMINATOR_BITS
from tnfr.physics.nodal_remainder import derive_nodal_remainder_itinerary


def _word(**changes):
    args = dict(
        epi_states=((0.5,), (0.5,)),
        timesteps=(0.0625,),
        capacities=((1.0,),),
        pressures=((0.0,),),
    )
    args.update(changes)
    return derive_nodal_remainder_itinerary(**args)


def test_stationary_word_has_full_source_cell_and_zero_carry_witness():
    observed = _word()
    cell = observed.coordinates[0]
    assert (cell.lower, cell.upper) == (F(1, 2) - F(1, 2**55), F(1, 2) + F(1, 2**54))
    assert cell.lower_closed and cell.upper_closed
    assert observed.cumulative_nodal_area == ((0,), (0,))
    assert observed.feasible and observed.zero_initial_carry_feasible
    assert observed.visible_closed and observed.conditional_carried_cycle
    assert observed.witness_initial.remainder == (0,)
    assert observed.witness_sequence.endpoint == observed.witness_initial
    assert not observed.pressure_provenance_certified
    assert not observed.runtime_provenance_certified


def test_upper_tie_is_feasible_for_even_half_but_carry_does_not_close():
    observed = _word(pressures=((2.0**-50,),))
    cell = observed.coordinates[0]
    assert observed.total_nodal_area == (F(1, 2**54),)
    assert cell.upper == F(1, 2) and cell.upper_closed
    assert observed.feasible and observed.zero_initial_carry_feasible
    assert observed.visible_closed and not observed.conditional_carried_cycle
    assert observed.witness_sequence.endpoint.epi == (0.5,)
    assert observed.witness_sequence.endpoint.remainder == (F(1, 2**54),)


def test_two_feasible_edges_cannot_be_joined_by_resetting_carry():
    first = _word(pressures=((2.0**-50,),))
    second = _word(pressures=((2.0**-50,),))
    assert first.feasible and second.feasible
    combined = _word(
        epi_states=((0.5,),) * 3,
        timesteps=(0.0625,) * 2,
        capacities=((1.0,),) * 2,
        pressures=((2.0**-50,),) * 2,
    )
    assert not combined.feasible and not combined.zero_initial_carry_feasible
    assert combined.coordinates[0].lower > combined.coordinates[0].upper
    assert combined.witness_initial is combined.witness_sequence is None
    assert not combined.conditional_carried_cycle


def test_nonzero_initial_carry_can_realize_a_word_that_zero_carry_cannot():
    successor = math.nextafter(0.5, math.inf)
    observed = _word(epi_states=((0.5,), (successor,)), pressures=((2.0**-50,),))
    cell = observed.coordinates[0]
    assert cell.lower == F(1, 2) and not cell.lower_closed
    assert observed.feasible and not observed.zero_initial_carry_feasible
    assert observed.witness_initial.remainder == (
        F(1, 2**NODAL_REMAINDER_DENOMINATOR_BITS),
    )
    assert observed.witness_sequence.endpoint.epi == (successor,)
    assert not observed.visible_closed and not observed.conditional_carried_cycle


def test_odd_source_cells_exclude_both_midpoint_ties():
    odd = math.nextafter(0.5, math.inf)
    observed = _word(epi_states=((odd,), (odd,)))
    cell = observed.coordinates[0]
    assert (cell.lower, cell.upper) == (F(1, 2) + F(1, 2**54), F(1, 2) + F(3, 2**54))
    assert not cell.lower_closed and not cell.upper_closed
    scale = 2**NODAL_REMAINDER_DENOMINATOR_BITS
    assert cell.first_grid_index == cell.lower * scale + 1
    assert cell.last_grid_index == cell.upper * scale - 1


def test_legitimate_nonstationary_visible_loop_closes_exact_carry():
    successor = math.nextafter(0.5, math.inf)
    observed = _word(
        epi_states=((0.5,), (successor,), (0.5,)),
        timesteps=(1.0, 1.0),
        capacities=((1.0,), (1.0,)),
        pressures=((2.0**-53,), (-(2.0**-53),)),
    )
    assert observed.cumulative_nodal_area == ((0,), (F(1, 2**53),), (0,))
    assert (
        observed.feasible
        and observed.visible_closed
        and observed.conditional_carried_cycle
    )
    assert observed.witness_sequence.endpoint == observed.witness_initial
    assert tuple(step.after.epi for step in observed.witness_sequence.steps) == (
        (successor,),
        (0.5,),
    )


def test_zero_mean_nonzero_coordinate_area_is_not_a_carried_cycle():
    observed = _word(
        epi_states=((0.75, 0.75),) * 2,
        timesteps=(1.0,),
        capacities=((1.0, 1.0),),
        pressures=((2.0**-55, -(2.0**-55)),),
    )
    assert sum(observed.total_nodal_area) == 0
    assert any(observed.total_nodal_area)
    assert observed.visible_closed and observed.feasible
    assert not observed.conditional_carried_cycle
    assert observed.witness_sequence.endpoint.remainder == (F(1, 2**55), -F(1, 2**55))


def test_heterogeneous_capacity_enters_each_exact_nodal_area():
    observed = _word(
        epi_states=((0.5, 0.5), (0.625, 0.4375)),
        timesteps=(0.25,),
        capacities=((2.0, 1.0),),
        pressures=((0.25, -0.25),),
    )
    assert observed.total_nodal_area == (F(1, 8), -F(1, 16))
    assert observed.zero_initial_carry_feasible
    assert observed.witness_sequence.endpoint.epi == (0.625, 0.4375)


@pytest.mark.parametrize("h,capacity", [(0.0, (1.0,)), (1.0, (0.0,))])
def test_zero_duration_or_capacity_preserves_exact_state(h, capacity):
    observed = _word(timesteps=(h,), capacities=(capacity,), pressures=((1e300,),))
    assert observed.total_nodal_area == (0,)
    assert observed.conditional_carried_cycle


def test_exact_band_intersection_prevents_hidden_boundary_excursion():
    free = _word(epi_states=((1.0,), (1.0,)), pressures=((2.0**-50,),))
    assert free.feasible and not free.zero_initial_carry_feasible
    assert free.coordinates[0].upper == F(1) - F(1, 2**54)
    fixed = _word(
        epi_states=((1.0,), (1.0,)),
        pressures=((2.0**-50,),),
        epi_lower=1.0,
        epi_upper=1.0,
    )
    assert not fixed.feasible
    assert fixed.witness_initial is None


def test_singleton_band_and_zero_area_give_one_exact_initial_state():
    observed = _word(epi_lower=0.5, epi_upper=0.5)
    cell = observed.coordinates[0]
    assert cell.lower == cell.upper == F(1, 2)
    assert cell.lower_closed and cell.upper_closed
    assert cell.first_grid_index == cell.last_grid_index
    assert observed.conditional_carried_cycle


def test_opposite_large_internal_areas_do_not_hide_an_infeasible_prefix():
    observed = _word(
        epi_states=((0.5,),) * 3,
        timesteps=(1.0, 1.0),
        capacities=((1.0,),) * 2,
        pressures=((1.0,), (-1.0,)),
    )
    assert observed.total_nodal_area == (0,)
    assert observed.visible_closed
    assert not observed.feasible and not observed.conditional_carried_cycle


def test_results_and_coordinate_cells_are_frozen():
    observed = _word()
    with pytest.raises(FrozenInstanceError):
        observed.feasible = False
    with pytest.raises(FrozenInstanceError):
        observed.coordinates[0].lower_closed = False


@pytest.mark.parametrize(
    "changes,error",
    [
        ({"epi_states": [[0.5], [0.5]]}, TypeError),
        ({"epi_states": ((0.5,), [0.5])}, TypeError),
        ({"epi_states": ((0.5,), (1,))}, TypeError),
        ({"epi_states": ((0.5,), (True,))}, TypeError),
        ({"epi_states": ((0.5,), (math.nan,))}, ValueError),
        ({"epi_states": ((0.5,), (math.inf,))}, ValueError),
        ({"epi_states": ()}, ValueError),
        ({"epi_states": ((0.5,),)}, ValueError),
        ({"epi_states": ((0.5,), (0.5, 0.5))}, ValueError),
        ({"epi_states": ((0.5,), ())}, ValueError),
        ({"epi_states": ((0.5,), (0.01,))}, ValueError),
        ({"timesteps": []}, TypeError),
        ({"timesteps": ()}, ValueError),
        ({"timesteps": (1,)}, TypeError),
        ({"timesteps": (False,)}, TypeError),
        ({"timesteps": (-0.25,)}, ValueError),
        ({"timesteps": (math.inf,)}, ValueError),
        ({"capacities": [[1.0]]}, TypeError),
        ({"capacities": ((-1.0,),)}, ValueError),
        ({"capacities": ((F(1),),)}, TypeError),
        ({"capacities": ((1.0, 1.0),)}, ValueError),
        ({"capacities": ()}, ValueError),
        ({"pressures": ((math.nan,),)}, ValueError),
        ({"pressures": ((0,),)}, TypeError),
        ({"pressures": ()}, ValueError),
        ({"epi_lower": 0.0}, ValueError),
        ({"epi_upper": 1.1}, ValueError),
        ({"epi_lower": 0.75, "epi_upper": 0.25}, ValueError),
        ({"epi_lower": F(1, 4)}, TypeError),
    ],
)
def test_invalid_primitive_inputs_are_rejected(changes, error):
    with pytest.raises(error):
        _word(**changes)
