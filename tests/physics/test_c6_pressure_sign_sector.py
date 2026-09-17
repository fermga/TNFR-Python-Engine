"""Strict pressure-sector bounds from the exact C6 gradient lattice."""

from dataclasses import FrozenInstanceError, replace
from fractions import Fraction as F
import math

import pytest

from tnfr.dynamics._euler_kernel import NodalRemainderState, advance_nodal_remainder, initialize_nodal_remainder
from tnfr.physics.c6_pressure_lattice import (
    derive_c6_pressure_lattice, derive_c6_pressure_sign_sector, observe_c6_pressure_sector_exit,
)


def _reference(**changes):
    arguments = dict(phase=(0.,) * 6, epi_weight=.5, phase_weight=.5)
    arguments.update(changes)
    return derive_c6_pressure_lattice(**arguments)


def _state(epi=None, **changes):
    arguments = dict(epi_lower=.375, epi_upper=.625)
    arguments.update(changes)
    return initialize_nodal_remainder(epi or (math.nextafter(.5, -math.inf),) + (.5,) * 5, **arguments)


def _terminal_reference():
    phase = (float.fromhex('0x1.0b8fb3e3956cbp-55'),) + tuple(i * math.pi / 3 for i in range(1, 6))
    return _reference(phase=phase, epi_weight=float(F(3299399280161697, 2**54)),
                      phase_weight=float(F(854047988748571, 2**50)))


@pytest.mark.parametrize("sign", [-1, 1])
def test_synchronized_source_has_two_strict_sectors_and_a_zero_index(sign):
    reference = _reference()
    sector = derive_c6_pressure_sign_sector(reference, node=1, sign=sign)
    assert (sector.minimum_gradient_index, sector.maximum_gradient_index) == (-2**53, 2**53)
    assert sector.cut_index == sign
    assert sector.bounding_gradient_index == sign
    assert not sector.empty
    assert sector.pressure_bound == sign * 2.**-56
    assert sector.signed_pressure_margin == F(1, 2**56)
    assert (sector.sector_min_index, sector.sector_max_index) == (
        (-2**53, -1) if sign < 0 else (1, 2**53)
    )
    zero = observe_c6_pressure_sector_exit(
        reference, state=_state((.5,) * 6), node=1, sign=sign, timestep=.0625,
    )
    assert not zero.initial_in_sector
    assert zero.initial_observation.pressure[1] == 0.
    assert zero.per_step_margin is zero.max_sector_steps is zero.first_exit_bound is None
    assert not zero.opposite_sign_hit_certified


def test_terminal_node1_cut_separates_negative_and_positive_without_zero():
    reference = _terminal_reference()
    negative = derive_c6_pressure_sign_sector(reference, node=1, sign=-1)
    positive = derive_c6_pressure_sign_sector(reference, node=1, sign=1)
    assert negative.cut_index == negative.bounding_gradient_index == -1
    assert positive.cut_index == positive.bounding_gradient_index == 0
    assert F(negative.pressure_bound) == -F(128295757220873, 2**106)
    assert positive.pressure_bound == reference.sources[1] > 0
    for index in range(-4, 5):
        pressure = reference.sources[1] + float(reference.source.epi_weight * reference.gradient_quantum * index)
        assert (pressure < 0) == (index <= -1)
        assert (pressure > 0) == (index >= 0)


@pytest.mark.parametrize("sign", [-1, 1])
def test_exact_carried_distance_enters_the_signed_drift_budget(sign):
    reference = _reference()
    neighbor = math.nextafter(.5, -math.inf if sign < 0 else math.inf)
    initial = _state((neighbor,) + (.5,) * 5)
    carry = (F(0), F(1, 2**60), F(0), F(0), F(0), F(0))
    state = NodalRemainderState(initial.epi, carry, initial.epi_lower, initial.epi_upper)
    result = observe_c6_pressure_sector_exit(reference, state=state, node=1, sign=sign, timestep=.0625)
    expected_distance = F(1, 8) - sign * carry[1]
    assert result.initial_in_sector
    assert result.outward_distance == expected_distance
    assert result.per_step_margin == F(1, 2**60)
    assert result.max_sector_steps == 2**57 - sign
    assert result.first_exit_bound == result.max_sector_steps + 1
    assert result.max_sector_steps * result.per_step_margin <= expected_distance
    assert result.first_exit_bound * result.per_step_margin > expected_distance
    assert not result.opposite_sign_hit_certified
    assert not result.positive_band_exit_certified


@pytest.mark.parametrize("sign", [-1, 1])
def test_first_step_can_fail_the_band_instead_of_hitting_the_opposite_sector(sign):
    phase = ((.25,) + (0.,) * 5) if sign < 0 else ((0.,) + (.25,) * 5)
    reference = _reference(phase=phase)
    state = _state((.375 if sign < 0 else .625,) * 6)
    observed = observe_c6_pressure_sector_exit(reference, state=state, node=0, sign=sign, timestep=.0625)
    assert observed.initial_in_sector
    assert observed.outward_distance == 0
    assert observed.max_sector_steps == 0 and observed.first_exit_bound == 1
    with pytest.raises(ValueError, match="exact nodal update leaves"):
        advance_nodal_remainder(
            state, timestep=.0625, capacity=(1.,) * 6, pressure=observed.initial_observation.pressure,
        )
    assert not observed.opposite_sign_hit_certified


def test_subnormal_epi_weight_clips_unreachable_sign_cut_to_actual_gradient_range():
    tiny = math.ulp(0.)
    reference = _reference(phase=(0.,) + (.25,) * 5, epi_weight=tiny, phase_weight=1.)
    positive = derive_c6_pressure_sign_sector(reference, node=0, sign=1)
    negative = derive_c6_pressure_sign_sector(reference, node=0, sign=-1)
    assert not positive.empty
    assert positive.cut_index < positive.minimum_gradient_index
    assert positive.bounding_gradient_index == -2**53
    assert positive.pressure_bound == reference.sources[0]
    assert positive.signed_pressure_margin == F(reference.sources[0])
    assert negative.empty
    assert negative.sector_max_index < negative.sector_min_index
    assert negative.bounding_gradient_index is negative.pressure_bound is negative.signed_pressure_margin is None
    result = observe_c6_pressure_sector_exit(reference, state=_state(), node=0, sign=-1, timestep=.0625)
    assert not result.initial_in_sector
    assert result.max_sector_steps is None


@pytest.mark.parametrize("sign", [-1, 1])
def test_underflow_zero_plateau_makes_both_strict_sectors_empty(sign):
    reference = _reference(epi_weight=math.ulp(0.))
    sector = derive_c6_pressure_sign_sector(reference, node=2, sign=sign)
    assert sector.empty
    assert sector.pressure_bound is sector.signed_pressure_margin is None


@pytest.mark.parametrize("sign", [-1, 1])
def test_one_point_zero_pressure_band_has_no_strict_sector(sign):
    reference = _reference(epi_lower=.5, epi_upper=.5)
    sector = derive_c6_pressure_sign_sector(reference, node=0, sign=sign)
    assert sector.minimum_gradient_index == sector.maximum_gradient_index == 0
    assert sector.empty
    result = observe_c6_pressure_sector_exit(
        reference, state=_state((.5,) * 6, epi_lower=.5, epi_upper=.5), node=0, sign=sign, timestep=.0625,
    )
    assert result.outward_distance == 0
    assert not result.initial_in_sector and result.first_exit_bound is None


def test_one_point_nonzero_phase_band_has_a_nonempty_signed_sector():
    reference = _reference(phase=(0.,) + (.25,) * 5, epi_lower=.5, epi_upper=.5)
    sector = derive_c6_pressure_sign_sector(reference, node=0, sign=1)
    assert not sector.empty
    assert sector.sector_min_index == sector.sector_max_index == sector.bounding_gradient_index == 0
    state = _state((.5,) * 6, epi_lower=.5, epi_upper=.5)
    result = observe_c6_pressure_sector_exit(reference, state=state, node=0, sign=1, timestep=.0625)
    assert result.initial_in_sector and result.first_exit_bound == 1


def test_public_outer_and_source_caches_are_rebuilt():
    reference = _terminal_reference()
    forged_row = replace(reference.rows[1], nonnegative_min_index=1000, nonpositive_max_index=-1000)
    forged_source = replace(reference.source, no_zero_pressure=False, rows=())
    forged = replace(reference, source=forged_source, sources=(0.,) * 6,
                     rows=reference.rows[:1] + (forged_row,) + reference.rows[2:],
                     epi_quantum=F(1), gradient_quantum=F(1))
    sector = derive_c6_pressure_sign_sector(forged, node=1, sign=-1)
    assert sector.reference == reference
    assert sector.cut_index == -1
    result = observe_c6_pressure_sector_exit(forged, state=_state(), node=1, sign=-1, timestep=.0625)
    assert result.initial_observation.reference == reference
    assert result.initial_in_sector


def test_result_dataclasses_are_frozen():
    result = observe_c6_pressure_sector_exit(_reference(), state=_state(), node=1, sign=-1, timestep=.0625)
    with pytest.raises(FrozenInstanceError):
        result.max_sector_steps = 0
    with pytest.raises(FrozenInstanceError):
        result.sector.empty = True


@pytest.mark.parametrize("node,sign,error", [
    (True, 1, TypeError), (1., 1, TypeError), (-1, 1, ValueError), (6, 1, ValueError),
    (1, False, TypeError), (1, 1., TypeError), (1, 0, ValueError), (1, 2, ValueError),
])
def test_invalid_sector_selectors_are_rejected(node, sign, error):
    with pytest.raises(error):
        derive_c6_pressure_sign_sector(_reference(), node=node, sign=sign)


@pytest.mark.parametrize("h,error", [
    (0., ValueError), (-1., ValueError), (math.inf, ValueError),
    (math.nan, ValueError), (1, TypeError), (True, TypeError),
])
def test_invalid_timestep_is_rejected(h, error):
    with pytest.raises(error):
        observe_c6_pressure_sector_exit(_reference(), state=_state(), node=1, sign=-1, timestep=h)


def test_state_band_must_be_inside_the_certified_slab():
    state = initialize_nodal_remainder((.5,) * 6)
    with pytest.raises(ValueError, match="contained"):
        observe_c6_pressure_sector_exit(_reference(), state=state, node=1, sign=-1, timestep=.0625)


def test_state_dimension_and_forged_carry_are_rejected():
    with pytest.raises(ValueError, match="six ordered"):
        observe_c6_pressure_sector_exit(_reference(), state=_state((.5,)), node=1, sign=-1, timestep=.0625)
    state = _state()
    forged = replace(state, remainder=(F(1, 3),) + (F(0),) * 5)
    with pytest.raises(ValueError, match="dyadic"):
        observe_c6_pressure_sector_exit(_reference(), state=forged, node=1, sign=-1, timestep=.0625)


def test_forged_primitive_source_is_rejected_instead_of_using_cached_cut():
    reference = _reference()
    forged = replace(reference, source=replace(reference.source, epi_weight=F(1, 3)))
    with pytest.raises(ValueError, match="actual binary64"):
        derive_c6_pressure_sign_sector(forged, node=1, sign=-1)
