"""Independent centered-profile and replayed carried-pressure identities."""

from dataclasses import FrozenInstanceError, replace
from fractions import Fraction as F
import math

import pytest

from tnfr.constants.canonical import CHANNEL_WEIGHT_PRIMARY, CHANNEL_WEIGHT_SECONDARY
from tnfr.dynamics._euler_kernel import (
    NodalRemainderState, advance_nodal_remainder, initialize_nodal_remainder,
)
from tnfr.physics.c6_carried_profile import (
    derive_c6_carried_profile, observe_c6_carried_profile_step,
)
from tnfr.physics.c6_pressure_lattice import (
    derive_c6_pressure_lattice, observe_c6_pressure_lattice,
)


PHASE = tuple(map(float.fromhex, (
    "0x1.0b8fb3e3956cbp-55", "0x1.0c152382d7365p+0", "0x1.0c152382d7365p+1",
    "0x1.921fb54442d18p+1", "0x1.0c152382d7365p+2", "0x1.4f1a6c638d03fp+2",
)))
ZERO = (F(0),) * 6


def _profile(*, actual=False):
    lattice = derive_c6_pressure_lattice(
        phase=PHASE if actual else (0.,) * 6,
        epi_weight=CHANNEL_WEIGHT_SECONDARY if actual else .5,
        phase_weight=CHANNEL_WEIGHT_PRIMARY if actual else .5,
    )
    return derive_c6_carried_profile(lattice)


def _state(epi=(.5,) * 6, remainder=ZERO, **band):
    state = initialize_nodal_remainder(epi, epi_lower=band.get("epi_lower", .375),
                                       epi_upper=band.get("epi_upper", .625))
    return replace(state, remainder=remainder)


def _step(profile, state=None, **changes):
    state = _state() if state is None else state
    pressure = observe_c6_pressure_lattice(profile.lattice, epi=state.epi).pressure
    args = dict(timestep=.0625, capacity=(1.,) * 6, pressure=pressure)
    args.update(changes)
    return advance_nodal_remainder(state, **args)


def _lap(values):
    return tuple(value - (values[i - 1] + values[(i + 1) % 6]) / 2
                 for i, value in enumerate(values))


def test_zero_phase_source_has_zero_profile_and_exact_unit_cycle_metric():
    profile = _profile()
    balance = profile.forced_balance
    assert balance.relative_profile == ZERO and balance.forcing == ZERO
    assert balance.metric_weights == (F(2),) * 6
    assert balance.source.capacity == (F(1),) * 6
    assert balance.mean_drift == 0 and balance.has_zero_pressure_equilibrium
    assert profile.graph_provenance_certified is False
    with pytest.raises(FrozenInstanceError):
        profile.forced_balance = None


def test_actual_closed_phase_profile_matches_independent_exact_poisson_witness():
    profile = _profile(actual=True)
    balance = profile.forced_balance
    numerators = (-920914671527074529, -109391571287326151, 538473136940002879,
                  -66355793410064327, 1834202553394660903, -1276013654110198775)
    expected = tuple(F(value, 4279441930180617987241864442413056) for value in numerators)
    assert balance.relative_profile == expected
    assert sum(expected) == 0
    assert balance.mean_drift == -F(1, 6 * 2**109)
    assert not balance.has_zero_pressure_equilibrium
    assert tuple(balance.epi_weight * value for value in _lap(expected)) == tuple(
        value - balance.mean_drift for value in balance.forcing)
    assert balance.profile_residual == ZERO and balance.profile_center_residual == 0


def test_exact_alternating_diffusion_has_hand_derived_fifteen_sixteenths_factor():
    profile = _profile()
    state = _state(tuple(.5 + (-1)**i / 16 for i in range(6)))
    result = observe_c6_carried_profile_step(profile, step=_step(profile, state))
    expected = tuple(F((-1)**i, 16) for i in range(6))
    assert result.error_before == expected
    assert result.error_after == tuple(F(15, 16) * value for value in expected)
    assert result.centered_before == expected
    assert result.carry_feedback == result.rounding_defect == ZERO
    assert result.recurrence_residual == ZERO
    assert result.mean_change == result.mean_identity_residual == 0


def test_hidden_alternating_carry_is_exactly_cancelled_by_readout_feedback():
    profile = _profile()
    carry = tuple(F((-1)**i, 2**60) for i in range(6))
    state = _state(remainder=carry)
    result = observe_c6_carried_profile_step(profile, step=_step(profile, state))
    assert result.step.pressure == (0.,) * 6
    assert result.step.after == state
    assert result.error_before == result.error_after == carry
    assert result.carry_feedback == carry  # w=1/2 and alternating L eigenvalue=2.
    assert result.rounding_defect == ZERO
    assert result.forcing_defect == result.centered_forcing_defect == carry
    assert result.recurrence_residual == ZERO
    assert result.mean_carry_contribution == 0


def test_uniform_carry_is_a_mean_coordinate_and_not_a_shape_error():
    profile = _profile()
    result = observe_c6_carried_profile_step(
        profile, step=_step(profile, _state(remainder=(F(1, 2**60),) * 6)),
    )
    assert result.centered_before == result.error_before == ZERO
    assert result.carry_feedback == ZERO
    assert result.step.before.exact_epi == (F(1, 2) + F(1, 2**60),) * 6
    assert result.mean_change == 0


def test_actual_uniform_visible_state_has_nonzero_mean_drift_with_zero_realization_error():
    profile = _profile(actual=True)
    result = observe_c6_carried_profile_step(profile, step=_step(profile))
    assert result.rounding_defect == result.carry_feedback == ZERO
    assert result.mean_change == result.mean_source_contribution == -F(1, 96 * 2**109)
    assert result.mean_identity_residual == 0 and result.recurrence_residual == ZERO
    assert result.graph_provenance_certified is False


def test_actual_local_assembly_can_cancel_the_nonzero_source_mean():
    profile = _profile(actual=True)
    state = _state((math.nextafter(.5, -math.inf),) + (.5,) * 5)
    result = observe_c6_carried_profile_step(profile, step=_step(profile, state))
    assert result.pressure_observation.mean_pressure == 0
    assert result.mean_change == 0
    assert result.mean_rounding_contribution == -result.mean_source_contribution
    assert result.mean_rounding_contribution > 0
    assert result.mean_carry_contribution == 0
    assert result.recurrence_residual == ZERO


def test_unequal_epi_signed_decomposition_uses_full_vector_and_retained_carry():
    profile = _profile(actual=True)
    state = _state((.4, .45, .5, .55, .6, .475), tuple(F((-1)**i, 2**61) for i in range(6)))
    result = observe_c6_carried_profile_step(profile, step=_step(profile, state))
    assert any(result.rounding_defect) and any(result.carry_feedback)
    p, weight, h = tuple(map(F, result.step.pressure)), profile.forced_balance.epi_weight, F(1, 16)
    expected_rounding = tuple(force - source + weight * lap for force, source, lap in zip(
        p, profile.forced_balance.forcing, _lap(tuple(map(F, state.epi))), strict=True))
    assert result.rounding_defect == expected_rounding
    assert result.carry_feedback == tuple(weight * value for value in _lap(state.remainder))
    assert result.mean_change == h * sum(p) / 6
    assert result.mean_carry_contribution == 0
    assert result.recurrence_residual == ZERO and result.mean_identity_residual == 0


def test_profile_and_nested_source_caches_are_rebuilt():
    profile = _profile(actual=True)
    forged_lattice = replace(profile.lattice, sources=(0.,) * 6, nonnegative_min_sum=-100)
    forged_source = replace(forged_lattice.source, rows=())
    forged_lattice = replace(forged_lattice, source=forged_source)
    forged = replace(profile, lattice=forged_lattice,
                     forced_balance=replace(profile.forced_balance, relative_profile=(F(99),) * 6))
    result = observe_c6_carried_profile_step(forged, step=_step(profile))
    assert result.profile == profile


@pytest.mark.parametrize("field,value", [
    ("exact_increment", (F(1),) * 6),
    ("visible_increment", (F(1),) * 6),
    ("carry_transfer", (F(1),) * 6),
    ("nodal_balance_residual", (F(1),) * 6),
])
def test_forged_step_caches_are_rejected(field, value):
    profile = _profile()
    with pytest.raises(ValueError, match="replay"):
        observe_c6_carried_profile_step(profile, step=replace(_step(profile), **{field: value}))


def test_forged_endpoint_cannot_pass_by_using_a_zero_laplacian_shift():
    profile = _profile()
    step = _step(profile)
    after = replace(step.after, remainder=(F(1, 2**60),) * 6)
    with pytest.raises(ValueError, match="replay"):
        observe_c6_carried_profile_step(profile, step=replace(step, after=after))


@pytest.mark.parametrize("alias", [0, False, 0.])
def test_equal_valued_wrong_cache_types_are_rejected(alias):
    profile = _profile()
    with pytest.raises(TypeError, match="exact Fraction"):
        observe_c6_carried_profile_step(
            profile, step=replace(_step(profile), exact_increment=(alias,) * 6),
        )


def test_subclass_exact_epi_property_cannot_forge_the_initial_shape():
    class AlteredReadout(NodalRemainderState):
        @property
        def exact_epi(self):
            return tuple(F(1, 2) + F((-1)**i, 128) for i in range(6))

    profile = derive_c6_carried_profile(derive_c6_pressure_lattice(
        phase=(0.,) * 6, epi_weight=1., phase_weight=.5,
    ))
    state = AlteredReadout((.5,) * 6, ZERO, .375, .625)
    step = _step(profile, state, timestep=.5)
    result = observe_c6_carried_profile_step(profile, step=step)
    assert result.centered_before == result.error_before == ZERO
    assert result.error_after == ZERO and result.recurrence_residual == ZERO


def test_valid_shared_step_with_stale_or_external_pressure_is_rejected():
    profile = _profile(actual=True)
    supplied = _step(profile, pressure=(0.,) * 6)
    with pytest.raises(ValueError, match="freshly"):
        observe_c6_carried_profile_step(profile, step=supplied)


@pytest.mark.parametrize("capacity", [(2.,) * 6, (0.,) * 6, (1., 1., 1., 1., 1., 2.)])
def test_nonunit_capacity_is_outside_profile_scope(capacity):
    profile = _profile()
    with pytest.raises(ValueError, match="unit capacities"):
        observe_c6_carried_profile_step(profile, step=_step(profile, capacity=capacity))


def test_zero_duration_is_not_a_positive_flow_claim():
    profile = _profile()
    with pytest.raises(ValueError, match="positive timestep"):
        observe_c6_carried_profile_step(profile, step=_step(profile, timestep=0.))


def test_wider_declared_state_band_is_rejected_even_if_point_is_inside_slab():
    profile = _profile()
    with pytest.raises(ValueError, match="complete carried-state band"):
        observe_c6_carried_profile_step(profile, step=_step(profile, _state(epi_lower=.05, epi_upper=1.)))


def test_narrower_state_band_is_accepted():
    profile = _profile()
    result = observe_c6_carried_profile_step(
        profile, step=_step(profile, _state(epi_lower=.45, epi_upper=.55)),
    )
    assert result.recurrence_residual == ZERO


@pytest.mark.parametrize("profile,step", [(None, None), ({}, None)])
def test_profile_type_is_explicit(profile, step):
    with pytest.raises(TypeError, match="profile"):
        observe_c6_carried_profile_step(profile, step=step)


def test_invalid_step_type_is_rejected():
    with pytest.raises(TypeError, match="step"):
        observe_c6_carried_profile_step(_profile(), step=None)


def test_invalid_primitive_profile_weight_is_not_repaired_from_caches():
    profile = _profile()
    bad_source = replace(profile.lattice.source, epi_weight=F(1, 3))
    with pytest.raises(ValueError, match="actual binary64"):
        derive_c6_carried_profile(replace(profile.lattice, source=bad_source))


def test_result_is_frozen():
    profile = _profile()
    result = observe_c6_carried_profile_step(profile, step=_step(profile))
    with pytest.raises(FrozenInstanceError):
        result.mean_change = F(1)
