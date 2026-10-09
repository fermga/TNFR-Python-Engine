"""Conditional local pressure persistence and selected-coordinate cell bounds."""

import math
from dataclasses import FrozenInstanceError, replace
from fractions import Fraction as F

import pytest

from tnfr.dynamics._euler_kernel import NodalRemainderState, initialize_nodal_remainder
from tnfr.physics.c6_pressure_lattice import (
    derive_c6_pressure_lattice,
    observe_c6_frozen_pressure_stencil,
)
from tnfr.physics.nodal_remainder import derive_nodal_remainder_cell_horizon


def _reference(**changes):
    args = dict(phase=(0.0,) * 6, epi_weight=0.5, phase_weight=0.5)
    args.update(changes)
    return derive_c6_pressure_lattice(**args)


def _state(epi=None, **changes):
    args = dict(epi_lower=0.375, epi_upper=0.625)
    args.update(changes)
    if epi is None:
        epi = (math.nextafter(0.5, -math.inf),) + (0.5,) * 5
    return initialize_nodal_remainder(epi, **args)


def _observe(reference=None, state=None, **changes):
    args = dict(state=_state() if state is None else state, node=1, timestep=0.0625)
    args.update(changes)
    return observe_c6_frozen_pressure_stencil(
        _reference() if reference is None else reference, **args
    )


def test_negative_center_bound_is_not_the_global_first_cell_exit():
    result = _observe()
    assert result.stencil == (0, 1, 2)
    assert result.stencil_epi == result.state.epi[:3]
    assert result.gradient_index == -1 and result.pressure == -(2.0**-56)
    assert result.exact_increment == -F(1, 2**60)
    assert result.center_cell_width == F(3, 2**55)
    assert (result.initial_max_frozen_steps, result.initial_first_exit_bound) == (
        32,
        33,
    )
    assert (result.uniform_max_frozen_steps, result.uniform_first_exit_bound) == (
        96,
        97,
    )
    whole = derive_nodal_remainder_cell_horizon(
        state=result.state,
        timestep=0.0625,
        capacity=(1.0,) * 6,
        pressure=result.initial_observation.pressure,
    )
    assert whole.coordinate_step_limits[1] == 32
    assert whole.first_exit_step == 16
    assert whole.first_exit_leaves_cell[0] and not whole.first_exit_leaves_cell[1]
    assert not result.graph_provenance_certified
    assert not result.sign_hit_certified and not result.positive_band_exit_certified


def test_positive_center_uses_the_wider_upper_half_cell_at_point_five():
    state = _state((math.nextafter(0.5, math.inf),) + (0.5,) * 5)
    result = _observe(state=state)
    assert result.gradient_index == 2 and result.pressure == 2.0**-55
    assert result.exact_increment == F(1, 2**59)
    assert (result.initial_max_frozen_steps, result.initial_first_exit_bound) == (
        32,
        33,
    )
    assert (result.uniform_max_frozen_steps, result.uniform_first_exit_bound) == (
        48,
        49,
    )


def test_remote_epi_and_carry_do_not_change_the_selected_row_or_center_bounds():
    initial = _observe()
    remote = _state(initial.state.epi[:3] + (0.4, 0.6, 0.45))
    carry = (F(0), F(0), F(0), F(1, 2**60), -F(1, 2**60), F(1, 2**60))
    remote = replace(remote, remainder=carry)
    changed = _observe(state=remote)
    assert changed.stencil_epi == initial.stencil_epi
    assert changed.pressure == initial.pressure
    assert changed.gradient_index == initial.gradient_index
    assert changed.initial_first_exit_bound == initial.initial_first_exit_bound
    assert changed.uniform_first_exit_bound == initial.uniform_first_exit_bound
    assert changed.initial_observation.pressure != initial.initial_observation.pressure


def test_center_carry_changes_its_conditional_limit_but_not_pressure_or_uniform_bound():
    state = _state()
    state = replace(state, remainder=(F(0), -F(1, 2**57), F(0), F(0), F(0), F(0)))
    result = _observe(state=state)
    assert result.pressure == -(2.0**-56)
    assert result.center_cell.exact_input == F(1, 2) - F(1, 2**57)
    assert (result.initial_max_frozen_steps, result.initial_first_exit_bound) == (
        24,
        25,
    )
    assert result.uniform_max_frozen_steps == 96


def test_odd_center_tie_is_open_while_the_uniform_enclosure_is_conservative():
    odd = math.nextafter(0.5, math.inf)
    result = _observe(state=_state((0.5, odd, 0.5, 0.5, 0.5, 0.5)))
    assert not result.center_cell.even_significand
    assert result.pressure == -(2.0**-54) and result.exact_increment == -F(1, 2**58)
    assert (result.initial_max_frozen_steps, result.initial_first_exit_bound) == (
        15,
        16,
    )
    assert (result.uniform_max_frozen_steps, result.uniform_first_exit_bound) == (
        32,
        33,
    )


@pytest.mark.parametrize("sign", [-1, 1])
def test_width_is_band_clipped_and_band_failure_need_not_change_display(sign):
    phases = (
        (0.0, 0.25, 0.0, 0.0, 0.0, 0.0)
        if sign < 0
        else (0.25, 0.0, 0.25, 0.0, 0.0, 0.0)
    )
    reference = _reference(phase=phases)
    lower, upper = (0.5, 0.625) if sign < 0 else (0.375, 0.5)
    result = _observe(reference, _state((0.5,) * 6, epi_lower=lower, epi_upper=upper))
    assert result.center_cell.upper - result.center_cell.lower == F(3, 2**55)
    assert result.center_cell_width == (F(1, 2**54) if sign < 0 else F(1, 2**55))
    assert result.pressure * sign > 0
    assert result.initial_max_frozen_steps == result.uniform_max_frozen_steps == 0
    assert result.initial_first_exit_bound == result.uniform_first_exit_bound == 1
    assert not result.sign_hit_certified and not result.positive_band_exit_certified


def test_zero_pressure_supplies_no_finite_bound_or_invariance_claim():
    result = _observe(state=_state((0.5,) * 6))
    assert result.pressure == 0 and result.exact_increment == 0
    assert result.initial_max_frozen_steps is result.initial_first_exit_bound is None
    assert result.uniform_max_frozen_steps is result.uniform_first_exit_bound is None
    assert result.center_cell_width == F(3, 2**55)
    assert not result.sign_hit_certified and not result.graph_provenance_certified


def test_one_point_band_has_zero_clipped_width_even_with_a_nonzero_source():
    reference = _reference(
        phase=(0.0, 0.25, 0.0, 0.0, 0.0, 0.0), epi_lower=0.5, epi_upper=0.5
    )
    state = _state((0.5,) * 6, epi_lower=0.5, epi_upper=0.5)
    result = _observe(reference, state)
    assert result.center_cell_width == 0
    assert result.center_cell.upper > result.center_cell.lower
    assert result.pressure < 0
    assert result.initial_first_exit_bound == result.uniform_first_exit_bound == 1


def test_a_changed_stencil_can_keep_the_identical_pressure():
    center = 0.4375
    delta = math.ulp(center)
    left = _observe(state=_state((center,) * 6))
    right = _observe(
        state=_state((center + delta, center, center - delta, center, center, center))
    )
    assert left.stencil_epi != right.stencil_epi
    assert left.gradient_index == right.gradient_index == 0
    assert left.pressure == right.pressure == 0
    assert not right.sign_hit_certified


def test_ordered_stencil_wraps_at_the_cycle_end():
    result = _observe(node=0)
    assert result.stencil == (5, 0, 1)
    assert result.stencil_epi == (
        result.state.epi[5],
        result.state.epi[0],
        result.state.epi[1],
    )


def test_inherited_node_four_static_stencil_has_distinct_initial_and_uniform_limits():
    phase = (float.fromhex("0x1.0b8fb3e3956cbp-55"),) + tuple(
        i * math.pi / 3 for i in range(1, 6)
    )
    reference = _reference(
        phase=phase,
        epi_weight=float(F(3299399280161697, 2**54)),
        phase_weight=float(F(854047988748571, 2**50)),
    )
    delta = F(1, 2**54)
    epi = tuple(float(F(1, 2) + index * delta) for index in (0, 0, 0, -2, 4, -3))
    carry = (F(0), F(0), F(0), F(0), F(4148688737535375, 2**110), F(0))
    state = NodalRemainderState(epi, carry, 0.375, 0.625)
    result = _observe(reference, state, node=4)
    assert result.gradient_index == -13
    assert F(result.pressure) == F(1668868774373469, 2**105)
    assert result.exact_increment == F(1668868774373469, 2**109)
    assert result.center_cell.upper - result.center_cell.exact_input == F(
        67908905300392561, 2**110
    )
    assert (result.initial_max_frozen_steps, result.initial_first_exit_bound) == (
        20,
        21,
    )
    assert (result.uniform_max_frozen_steps, result.uniform_first_exit_bound) == (
        43,
        44,
    )


def test_reference_caches_are_rebuilt_and_results_frozen():
    reference = _reference()
    forged = replace(reference, sources=(1.0,) * 6, gradient_quantum=F(1), rows=())
    result = _observe(forged)
    assert result.reference == reference
    assert result.initial_observation.reference == reference
    assert result.pressure == -(2.0**-56)
    with pytest.raises(FrozenInstanceError):
        result.initial_first_exit_bound = 100


@pytest.mark.parametrize(
    "node,error",
    [(True, TypeError), (1.0, TypeError), (-1, ValueError), (6, ValueError)],
)
def test_invalid_node_is_rejected(node, error):
    with pytest.raises(error):
        _observe(node=node)


@pytest.mark.parametrize(
    "timestep,error",
    [
        (0.0, ValueError),
        (-1.0, ValueError),
        (math.inf, ValueError),
        (math.nan, ValueError),
        (1, TypeError),
        (False, TypeError),
    ],
)
def test_invalid_timestep_is_rejected(timestep, error):
    with pytest.raises(error):
        _observe(timestep=timestep)


def test_state_validation_rejects_wrong_dimension_band_and_noncanonical_carry():
    with pytest.raises(ValueError, match="six ordered"):
        _observe(state=_state((0.5,)))
    with pytest.raises(ValueError, match="contained"):
        _observe(state=initialize_nodal_remainder((0.5,) * 6))
    state = _state()
    with pytest.raises(ValueError, match="dyadic"):
        _observe(state=replace(state, remainder=(F(1, 3),) + (F(0),) * 5))


def test_invalid_primitive_source_cannot_hide_behind_cached_pressure():
    reference = _reference()
    forged = replace(reference, source=replace(reference.source, epi_weight=F(1, 3)))
    with pytest.raises(ValueError, match="actual binary64"):
        _observe(forged)
