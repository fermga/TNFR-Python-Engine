"""Self-consistent closure, exact integer bounds and independent scope controls."""

from dataclasses import replace
from fractions import Fraction as F
import math

import pytest

from tnfr.constants.canonical import CHANNEL_WEIGHT_PRIMARY, CHANNEL_WEIGHT_SECONDARY
from tnfr.dynamics._euler_kernel import advance_nodal_remainder, initialize_nodal_remainder
from tnfr.physics.c6_carried_closure import _sqrt_upper, derive_c6_carried_closure
from tnfr.physics.c6_carried_profile import derive_c6_carried_profile, observe_c6_carried_profile_step
from tnfr.physics.c6_pressure_lattice import derive_c6_pressure_lattice, observe_c6_pressure_lattice


PHASE = tuple(map(float.fromhex, (
    "0x1.0b8fb3e3956cbp-55", "0x1.0c152382d7365p+0", "0x1.0c152382d7365p+1",
    "0x1.921fb54442d18p+1", "0x1.0c152382d7365p+2", "0x1.4f1a6c638d03fp+2",
)))


def _profile(*, actual=False, weight=.5):
    return derive_c6_carried_profile(derive_c6_pressure_lattice(
        phase=PHASE if actual else (0.,) * 6,
        epi_weight=CHANNEL_WEIGHT_SECONDARY if actual else weight,
        phase_weight=CHANNEL_WEIGHT_PRIMARY if actual else .5,
    ))


def _state(epi=(.5,) * 6):
    return initialize_nodal_remainder(epi, epi_lower=.375, epi_upper=.625)


@pytest.mark.parametrize("value", (F(0), F(1), F(9, 16), F(2, 3), F(1, 2**2150)))
def test_rational_square_root_upper_bound_is_exact_when_possible(value):
    bound = _sqrt_upper(value)
    assert type(bound) is F and bound >= 0 and bound * bound >= value
    if value != F(2, 3):
        assert bound * bound == value


@pytest.mark.parametrize("value", (0., 1, F(-1)))
def test_square_root_helper_rejects_inexact_and_negative_arguments(value):
    with pytest.raises(ValueError, match="nonnegative Fraction"):
        _sqrt_upper(value)


def test_self_consistent_floor_closes_the_exact_affine_norm_recurrence():
    result = derive_c6_carried_closure(_profile(actual=True), state=_state(), timestep=1 / 16)
    base, h = result.base_tube, F(1, 16)
    weight = base.contraction.profile.forced_balance.epi_weight
    q, a, b = base.contraction.norm_factor, result.rounding_affine_constant, result.rounding_affine_slope
    assert result.effective_norm_factor == q + 3 * h * b < 1
    assert result.energy_floor * (1 - result.effective_norm_factor)**2 == 6 * h**2 * (2 * weight * base.carry_bound + a)**2
    assert result.energy_bound == max(result.energy_floor, base.initial_energy)
    assert result.rounding_bound == result.product_error_bound + result.assembly_error_bound
    assert result.rounding_bound == a + b * result.laplacian_error_upper_bound
    assert result.laplacian_error_upper_bound**2 >= F(3, 2) * result.energy_bound
    assert result.energy_bound < base.energy_bound
    assert result.rounding_bound < base.rounding_bound / 10**13


@pytest.mark.parametrize("epi,actual", (
    ((.5,) * 6, False),
    ((.375, .625) * 3, False),
    ((.375, .5, .625) * 2, False),
    (tuple(.5 + n * 2.**-54 for n in (-2, 0, 2, -2, 4, -4)), True),
))
def test_shared_pressure_steps_satisfy_closed_spatial_rounding_and_mean_bounds(epi, actual):
    profile, state = _profile(actual=actual), _state(epi)
    result = derive_c6_carried_closure(profile, state=state, timestep=1 / 16)
    for _ in range(3):
        reading = observe_c6_pressure_lattice(profile.lattice, epi=state.epi)
        step = advance_nodal_remainder(state, timestep=1 / 16, capacity=(1.,) * 6, pressure=reading.pressure)
        observed = observe_c6_carried_profile_step(profile, step=step)
        assert sum(value * value for value in observed.error_after) <= result.energy_bound
        assert all(abs(value) <= result.product_error_bound for value in reading.epi_reduction_error)
        assert all(abs(value) <= result.assembly_error_bound for value in reading.assembly_error)
        assert all(abs(value) <= result.rounding_bound for value in observed.rounding_defect)
        assert result.mean_increment_lower <= observed.mean_change <= result.mean_increment_upper
        assert all(lo <= value <= hi for value, lo, hi in zip(
            reading.gradient_indices, result.gradient_index_lower, result.gradient_index_upper, strict=True))
        state = step.after


def test_gradient_integer_hulls_are_sharp_for_the_derived_scalar_inequality():
    result = derive_c6_carried_closure(_profile(actual=True), state=_state(), timestep=1 / 16)
    base = result.base_tube
    quantum, carry = base.contraction.profile.lattice.gradient_quantum, base.carry_bound
    for center, lower, upper in zip(
        result.profile_gradient_indices, result.gradient_index_lower, result.gradient_index_upper, strict=True,
    ):
        def admitted(index):
            remaining = abs(quantum * (index - center)) - 2 * carry
            return remaining <= 0 or remaining**2 <= result.laplacian_error_squared_bound
        assert admitted(lower) and admitted(upper)
        assert not admitted(lower - 1) and not admitted(upper + 1)
    assert not result.gradient_reachability_certified


def test_actual_endpoint_spatial_floor_leaves_the_signed_mean_unresolved():
    state = _state(tuple(.5 + n * 2.**-54 for n in (-2, 0, 2, -2, 4, -4)))
    result = derive_c6_carried_closure(_profile(actual=True), state=state, timestep=1 / 16)
    assert result.gradient_index_lower == (-26, -28, -33, -17, -49, -13)
    assert result.gradient_index_upper == (29, 27, 22, 38, 6, 42)
    assert result.mean_increment_lower < 0 < result.mean_increment_upper
    assert (result.mean_increment_lower + result.mean_increment_upper) / 2 == -F(1, 6 * 2**113)
    assert not result.infinite_mean_control_certified and not result.future_runtime_certified


def test_near_noncontracting_boundary_cannot_hide_rounding_feedback():
    with pytest.raises(ValueError, match=r"q \+ 3\*h\*b < 1"):
        derive_c6_carried_closure(_profile(weight=1.), state=_state(), timestep=math.nextafter(1., 0.))


def test_profile_caches_are_rebuilt_from_primitive_pressure_source():
    profile, state = _profile(actual=True), _state()
    forged = replace(profile, forced_balance=replace(profile.forced_balance,
                     relative_profile=(F(0),) * 6, mean_drift=F(100)))
    assert derive_c6_carried_closure(forged, state=state, timestep=1 / 16) == derive_c6_carried_closure(
        profile, state=state, timestep=1 / 16)


def test_large_initial_disagreement_is_retained_instead_of_replaced_by_small_floor():
    result = derive_c6_carried_closure(_profile(), state=_state((.375, .5, .625) * 2), timestep=1 / 16)
    assert result.energy_bound == result.base_tube.initial_energy == F(1, 16)
    assert result.energy_bound > result.energy_floor
    # This case has a simple non-square rational radius; integer bounds must
    # use logarithmic exact searches instead of enumerating ~10^14 lattice points.
    assert result.gradient_index_upper[0] > 10**12


def test_invalid_carried_encoding_is_rejected_before_any_closed_bound():
    state = replace(_state(), remainder=(F(1),) * 6)
    with pytest.raises(ValueError):
        derive_c6_carried_closure(_profile(), state=state, timestep=1 / 16)
