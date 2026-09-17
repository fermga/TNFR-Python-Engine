"""Independent exact prefix and final-cell checks for carried nodal area."""

from dataclasses import FrozenInstanceError, replace
from fractions import Fraction as F
import math

import pytest

from tnfr.dynamics._euler_kernel import advance_nodal_remainder, initialize_nodal_remainder
from tnfr.physics import observe_nodal_remainder_sequence


def _observe(initial=None, *, timesteps=(.0625,), capacities=((1., 1.),),
             pressures=((2.**-50, -2.**-50),)):
    return observe_nodal_remainder_sequence(
        initial=initial or initialize_nodal_remainder((.5, .5)),
        timesteps=timesteps, capacities=capacities, pressures=pressures,
    )


def test_balanced_area_can_have_visible_mean_bias_but_exact_reconstructed_mean():
    result = _observe()
    prefix = result.prefixes[0]
    assert result.endpoint.epi == (.5, .5 - 2.**-54)
    assert prefix.mean_nodal_area == prefix.mean_reconstructed_change == 0
    assert prefix.mean_visible_change == prefix.mean_rounding_defect == -F(1, 2**55)
    assert prefix.mean_carry_transfer == prefix.mean_rounding_defect
    assert prefix.mean_rounding_lower_bound <= prefix.mean_rounding_defect <= prefix.mean_rounding_upper_bound
    assert result.uniform_mean_rounding_bound == F(1, 2**53)


def test_variable_supplied_inputs_have_independent_exact_prefix_balance():
    count = 67
    initial = initialize_nodal_remainder((.25, .5, .75))
    timesteps = tuple(.03125 if i % 2 else .0625 for i in range(count))
    capacities = tuple((1., .5, 2.) for _ in range(count))
    pressures = tuple(((-1.)**i * 2.**-48, 2.**-50, -2.**-52) for i in range(count))
    result = _observe(initial, timesteps=timesteps, capacities=capacities, pressures=pressures)
    area = [F(0)] * 3
    for k, prefix in enumerate(result.prefixes):
        for i in range(3):
            area[i] += F(timesteps[k]) * F(capacities[k][i]) * F(pressures[k][i])
        expected = tuple(F(initial.epi[i]) + area[i] for i in range(3))
        assert result.steps[k].after.exact_epi == expected
        assert prefix.cumulative_nodal_area == prefix.reconstructed_change == tuple(area)
        assert prefix.identity_residual == (F(0),) * 3
        assert prefix.mean_identity_residual == 0
        assert abs(prefix.mean_rounding_defect) <= result.uniform_mean_rounding_bound
        assert prefix.mean_rounding_lower_bound <= prefix.mean_rounding_defect <= prefix.mean_rounding_upper_bound
        for cell, coordinate in zip(prefix.output_cells, expected, strict=True):
            assert cell.contains_exact_input and cell.exact_input == coordinate
            assert cell.lower <= coordinate <= cell.upper


def test_nonzero_initial_carry_has_signed_endpoint_telescope():
    first = _observe().endpoint
    result = _observe(first, timesteps=(0.,))
    assert result.endpoint == first
    prefix = result.prefixes[0]
    assert prefix.mean_visible_change == prefix.mean_reconstructed_change == 0
    assert prefix.mean_carry_transfer == prefix.mean_rounding_defect == 0
    assert result.uniform_mean_rounding_bound == F(1, 2**55) + F(1, 2**53)
    assert prefix.mean_rounding_lower_bound <= 0 <= prefix.mean_rounding_upper_bound


def test_carry_reset_is_an_exact_state_jump_not_another_valid_continuation():
    first = _observe().endpoint
    reset = initialize_nodal_remainder(first.epi)
    assert tuple(y - x for x, y in zip(first.exact_epi, reset.exact_epi, strict=True)) == tuple(-r for r in first.remainder)
    carried = _observe(first).endpoint
    discarded = _observe(reset).endpoint
    assert tuple(x - y for x, y in zip(carried.exact_epi, discarded.exact_epi, strict=True)) == first.remainder
    assert carried.epi != discarded.epi


def test_heterogeneous_capacity_requires_balanced_nodal_area():
    result = _observe(capacities=((2., 1.),), pressures=((.125, -.125),))
    prefix = result.prefixes[0]
    assert prefix.mean_nodal_area == prefix.mean_reconstructed_change == F(1, 256)
    assert prefix.mean_visible_change == F(1, 256)
    assert prefix.mean_rounding_defect == 0


def test_subnormal_exact_area_is_retained_with_independent_cells():
    tiny = math.ulp(0.)
    result = _observe(initialize_nodal_remainder((tiny,), epi_lower=tiny),
                      timesteps=(tiny,), capacities=((tiny,),), pressures=((tiny,),))
    prefix = result.prefixes[0]
    assert prefix.mean_nodal_area == F(1, 2**3222)
    assert prefix.mean_visible_change == 0
    assert prefix.mean_rounding_defect == -F(1, 2**3222)
    assert prefix.mean_rounding_lower_bound == -F(1, 2**1075)
    assert prefix.mean_rounding_upper_bound == F(1, 2**1075)
    assert not prefix.output_cells[0].even_significand


@pytest.mark.parametrize("changes,error", (
    ({"timesteps": []}, TypeError),
    ({"timesteps": (), "capacities": (), "pressures": ()}, ValueError),
    ({"timesteps": (.0625, .0625)}, ValueError),
    ({"pressures": [[1., 1.]]}, TypeError),
    ({"capacities": ((1.,),)}, ValueError),
))
def test_schedule_rejects_missing_order_and_mismatched_dimensions(changes, error):
    with pytest.raises(error):
        _observe(**changes)


def test_failing_later_step_leaves_initial_encoding_unchanged():
    initial = initialize_nodal_remainder((.5, .5))
    with pytest.raises(ValueError, match="band"):
        _observe(initial, timesteps=(.0625, 1.), capacities=((1., 1.),) * 2,
                 pressures=((2.**-50, -2.**-50), (2., 2.)))
    assert initial.exact_epi == (F(1, 2),) * 2
    assert initial.remainder == (F(0),) * 2


def test_returned_sequence_is_frozen_and_public_forged_encoding_is_rejected():
    result = _observe()
    with pytest.raises(FrozenInstanceError):
        result.uniform_mean_rounding_bound = F(0)
    with pytest.raises(FrozenInstanceError):
        result.prefixes[0].mean_rounding_defect = F(0)
    forged = replace(result.endpoint, remainder=(F(1, 2), F(0)))
    with pytest.raises(ValueError):
        _observe(forged)


def test_odd_tie_rounds_to_even_cell_and_preserves_the_tie_remainder():
    source = math.nextafter(.5, 1.)
    initial = initialize_nodal_remainder((source,))
    result = _observe(initial, timesteps=(1.,), capacities=((1.,),), pressures=((2.**-54,),))
    cell = result.prefixes[0].output_cells[0]
    assert cell.even_significand and cell.lower_tie and cell.contains_exact_input
    assert result.endpoint.remainder == (-F(1, 2**54),)
    direct = advance_nodal_remainder(initial, timestep=1., capacity=(1.,), pressure=(2.**-54,))
    assert result.endpoint == direct.after
