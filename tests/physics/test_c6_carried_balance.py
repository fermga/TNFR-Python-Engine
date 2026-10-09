"""Static exact pressure compensation is not a realizable temporal cycle."""

from dataclasses import FrozenInstanceError, replace
from fractions import Fraction as F

import pytest

from tnfr.constants.canonical import CHANNEL_WEIGHT_PRIMARY, CHANNEL_WEIGHT_SECONDARY
from tnfr.dynamics._euler_kernel import NodalRemainderState, initialize_nodal_remainder
from tnfr.physics.c6_carried_balance import (
    derive_c6_carried_pressure_balance,
    observe_c6_carried_pressure_point,
)
from tnfr.physics.c6_carried_closure import derive_c6_carried_closure
from tnfr.physics.c6_carried_profile import derive_c6_carried_profile
from tnfr.physics.c6_pressure_lattice import derive_c6_pressure_lattice

PHASE = tuple(
    map(
        float.fromhex,
        (
            "0x1.0b8fb3e3956cbp-55",
            "0x1.0c152382d7365p+0",
            "0x1.0c152382d7365p+1",
            "0x1.921fb54442d18p+1",
            "0x1.0c152382d7365p+2",
            "0x1.4f1a6c638d03fp+2",
        ),
    )
)
OFFSETS = (
    (-6, -1, 2, 2, 8, -7),
    (-5, 2, 4, -2, 6, -7),
    (-4, -2, 0, 2, 6, -4),
    (-4, -1, 2, -2, 8, -5),
    (-3, 0, 2, -2, 8, -7),
    (-2, -1, 4, -2, 6, -7),
    (-2, 2, 0, -2, 6, -6),
)
CARRY = F(96076792050570541, 2**113)


def _epi(offsets):
    values = tuple(F(1, 2) + F(value, 2**54) for value in offsets)
    result = tuple(map(float, values))
    assert tuple(map(F, result)) == values
    return result


def _state(offsets, carry=CARRY):
    return NodalRemainderState(_epi(offsets), (carry,) * 6, 0.375, 0.625)


@pytest.fixture(scope="module")
def closure():
    lattice = derive_c6_pressure_lattice(
        phase=PHASE,
        epi_weight=CHANNEL_WEIGHT_SECONDARY,
        phase_weight=CHANNEL_WEIGHT_PRIMARY,
    )
    profile = derive_c6_carried_profile(lattice)
    state = _state((-2, 0, 2, -2, 4, -4))
    return derive_c6_carried_closure(profile, state=state, timestep=0.0625)


def test_positive_exact_pressure_balance_refutes_only_the_class_wide_separator(closure):
    original = closure.base_tube.state
    result = derive_c6_carried_pressure_balance(
        closure, states=tuple(map(_state, OFFSETS))
    )
    assert all(type(value) is F and value > 0 for value in result.weights)
    assert result.weight_sum == sum(result.weights) == 1
    assert result.pressure_balance == (F(0),) * 6
    for i in range(6):
        assert (
            sum(
                weight * F(point.observation.pressure[i])
                for weight, point in zip(result.weights, result.points)
            )
            == 0
        )
    assert all(point.energy < 21 * F(1, 2**108) for point in result.points)
    assert all(point.energy <= result.closure.energy_bound for point in result.points)
    assert all(point.state.remainder == (CARRY,) * 6 for point in result.points)
    assert result.common_reconstructed_mean == closure.base_tube.initial_mean
    assert result.reconstructed_means == (closure.base_tube.initial_mean,) * 7
    assert result.on_origin_mean_slice
    assert closure.base_tube.state == original
    assert result.class_wide_strict_linear_drift_excluded
    assert not result.temporal_compensation_certified
    assert (
        not result.bounded_trajectory_certified and not result.future_runtime_certified
    )
    assert all(
        not point.reachable_from_initial_state_certified for point in result.points
    )
    with pytest.raises(FrozenInstanceError):
        result.weights = (F(0),) * 7


@pytest.mark.parametrize(
    "offsets,scaled_sum",
    (
        ((-4, 0, 2, -1, 8, -7), -48),
        ((-4, -1, 2, -1, 8, -6), 64),
        ((-3, -1, 2, -1, 6, -5), 0),
    ),
)
def test_all_three_mean_signs_occur_at_admitted_near_profile_points(
    closure, offsets, scaled_sum
):
    point = observe_c6_carried_pressure_point(closure, state=_state(offsets))
    assert sum(map(F, point.observation.pressure)) == F(scaled_sum, 2**112)
    assert sum(point.state.exact_epi) / 6 == closure.base_tube.initial_mean
    assert point.energy < 5 * F(1, 2**108)
    assert point.energy <= closure.energy_bound
    assert not point.reachable_from_initial_state_certified


def test_singular_pressure_family_cannot_receive_an_affine_witness(closure):
    with pytest.raises(ValueError, match="singular"):
        derive_c6_carried_pressure_balance(closure, states=(_state(OFFSETS[0]),) * 7)


def test_same_sign_pressure_family_rejects_nonpositive_affine_coefficients(closure):
    rows = list(OFFSETS)
    rows[3] = (-6, -2, 0, -2, 6, -7)
    with pytest.raises(ValueError, match="strictly positive"):
        derive_c6_carried_pressure_balance(closure, states=tuple(map(_state, rows)))


@pytest.mark.parametrize("rows", ([], (), ((0.5,) * 6,) * 6, ((0.5,) * 6,) * 8))
def test_exactly_seven_ordered_points_are_required(closure, rows):
    with pytest.raises(ValueError, match="exactly seven"):
        derive_c6_carried_pressure_balance(closure, states=rows)


@pytest.mark.parametrize(
    "epi",
    (
        [0.5] * 6,
        (0.5,) * 5,
        (0.5,) * 5 + (1,),
        (0.5,) * 5 + (float("nan"),),
        (0.5,) * 5 + (float("inf"),),
    ),
)
def test_malformed_static_coordinates_do_not_enter_the_certificate(closure, epi):
    with pytest.raises((TypeError, ValueError)):
        observe_c6_carried_pressure_point(
            closure, state=NodalRemainderState(epi, (F(0),) * 6, 0.375, 0.625)
        )


def test_energy_cache_is_rebuilt_before_admitting_a_point(closure):
    forged = replace(closure, energy_bound=F(1))
    with pytest.raises(ValueError, match="outside the rebuilt centered"):
        observe_c6_carried_pressure_point(
            forged,
            state=initialize_nodal_remainder(
                (0.375, 0.625) * 3,
                epi_lower=0.375,
                epi_upper=0.625,
            ),
        )
    reduced = replace(closure, energy_bound=F(0))
    admitted = observe_c6_carried_pressure_point(reduced, state=_state(OFFSETS[0]))
    assert 0 < admitted.energy <= closure.energy_bound


def test_declared_band_is_enforced_for_hypothetical_points(closure):
    with pytest.raises(ValueError, match="positive band"):
        observe_c6_carried_pressure_point(
            closure,
            state=NodalRemainderState((0.25,) * 6, (F(0),) * 6, 0.375, 0.625),
        )


def test_closure_type_cannot_be_replaced_by_an_unchecked_object():
    with pytest.raises(TypeError, match="C6CarriedClosure"):
        observe_c6_carried_pressure_point(object(), state=_state(OFFSETS[0]))


def test_different_declared_band_cannot_broaden_the_point_class(closure):
    state = replace(_state(OFFSETS[0]), epi_lower=0.25, epi_upper=0.75)
    with pytest.raises(ValueError, match="must equal"):
        observe_c6_carried_pressure_point(closure, state=state)


def test_invalid_carry_is_not_reset_during_observation(closure):
    state = replace(_state(OFFSETS[0]), remainder=(F(1, 3),) * 6)
    with pytest.raises(ValueError, match="dyadic"):
        observe_c6_carried_pressure_point(closure, state=state)


def test_zero_carry_same_mean_points_do_not_claim_the_origin_mean(closure):
    states = tuple(_state(row, F(0)) for row in OFFSETS)
    result = derive_c6_carried_pressure_balance(closure, states=states)
    assert result.common_reconstructed_mean == closure.base_tube.initial_mean - CARRY
    assert not result.on_origin_mean_slice


def test_differing_hypothetical_means_do_not_claim_a_common_slice(closure):
    states = list(map(_state, OFFSETS))
    states[0] = _state(OFFSETS[0], F(0))
    result = derive_c6_carried_pressure_balance(closure, states=tuple(states))
    assert result.common_reconstructed_mean is None
    assert not result.on_origin_mean_slice


def test_overridden_exact_property_cannot_forge_a_point(closure):
    class SpoofedState(NodalRemainderState):
        @property
        def exact_epi(self):
            return (F(1, 2),) * 6

    plain = _state(OFFSETS[0])
    spoofed = SpoofedState(plain.epi, plain.remainder, plain.epi_lower, plain.epi_upper)
    observed = observe_c6_carried_pressure_point(closure, state=spoofed)
    expected = observe_c6_carried_pressure_point(closure, state=plain)
    assert type(observed.state) is NodalRemainderState
    assert observed == expected
