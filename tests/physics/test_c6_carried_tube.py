"""Uniform numerical envelopes bind generated rows, not sampled error maxima."""

from fractions import Fraction as F

import pytest

from tnfr.dynamics._euler_kernel import initialize_nodal_remainder, advance_nodal_remainder
from tnfr.physics.c6_pressure_lattice import derive_c6_pressure_lattice, observe_c6_pressure_lattice
from tnfr.physics.c6_carried_profile import derive_c6_carried_profile, observe_c6_carried_profile_step
from tnfr.physics.c6_carried_tube import (
    derive_c6_carried_contraction, derive_c6_carried_tube, derive_c6_carried_band_horizon,
)


def _profile(weight=.3):
    return derive_c6_carried_profile(derive_c6_pressure_lattice(
        phase=(0.,) * 6, epi_weight=weight, phase_weight=1.,
    ))


@pytest.mark.parametrize("h", (0., -1., 1., 2.))
def test_noncontracting_step_cannot_receive_a_strict_certificate(h):
    with pytest.raises(ValueError, match="strict C6 spatial contraction"):
        derive_c6_carried_contraction(_profile(1.), timestep=h)


@pytest.mark.parametrize("h", (True, 1, float('inf'), float('nan')))
def test_timestep_is_an_actual_finite_binary64_value(h):
    with pytest.raises((TypeError, ValueError)):
        derive_c6_carried_contraction(_profile(), timestep=h)


@pytest.mark.parametrize("epi", (
    (.5,) * 6,
    (.375, .625, .375, .625, .375, .625),
    (.375, .5, .625, .375, .5, .625),
    (.5 - 2.**-54, .5, .5 + 2.**-53, .5, .5 - 2.**-53, .5 + 2.**-52),
))
def test_generated_steps_obey_every_uniform_defect_and_energy_bound(epi):
    profile = _profile()
    state = initialize_nodal_remainder(epi, epi_lower=.375, epi_upper=.625)
    tube = derive_c6_carried_tube(profile, state=state, timestep=1 / 16)
    # Two admitted steps also exercise a nonzero retained rounding remainder.
    for _ in range(2):
        reading = observe_c6_pressure_lattice(profile.lattice, epi=state.epi)
        step = advance_nodal_remainder(state, timestep=1 / 16, capacity=(1.,) * 6, pressure=reading.pressure)
        observed = observe_c6_carried_profile_step(profile, step=step)
        assert all(abs(value) <= tube.product_error_bound for value in reading.epi_reduction_error)
        assert all(abs(value) <= tube.assembly_error_bound for value in reading.assembly_error)
        assert all(abs(value) <= tube.rounding_bound for value in observed.rounding_defect)
        assert all(abs(value) <= tube.forcing_component_bound for value in observed.forcing_defect)
        assert sum(value**2 for value in observed.centered_forcing_defect) <= tube.centered_forcing_norm_squared_bound
        before = sum(value**2 for value in observed.error_before)
        after = sum(value**2 for value in observed.error_after)
        q = tube.contraction.norm_factor
        affine_bound = q * before + F(1, 16)**2 * tube.centered_forcing_norm_squared_bound / (1 - q)
        assert after <= affine_bound and after <= tube.energy_bound
        assert abs(observed.mean_change) <= tube.mean_increment_bound
        assert observed.mean_carry_contribution == 0
        state = step.after


def test_valid_initial_state_does_not_imply_the_whole_tube_fits_its_band():
    profile = _profile()
    state = initialize_nodal_remainder((.375, .625) * 3, epi_lower=.375, epi_upper=.625)
    tube = derive_c6_carried_tube(profile, state=state, timestep=1 / 16)
    horizon = derive_c6_carried_band_horizon(tube)
    assert not horizon.tube_initially_admitted
    assert horizon.maximum_steps is None and not horizon.unbounded_conditional_prefix
    assert not horizon.actual_band_exit_certified and not horizon.future_runtime_certified


def test_mean_budget_does_not_charge_the_zero_mean_carry_feedback():
    profile = _profile()
    state = initialize_nodal_remainder((.5,) * 6, epi_lower=.375, epi_upper=.625)
    tube = derive_c6_carried_tube(profile, state=state, timestep=1 / 16)
    assert profile.forced_balance.mean_drift == 0
    assert tube.mean_increment_bound == F(1, 16) * tube.rounding_bound
    assert tube.forcing_component_bound > tube.rounding_bound
    assert not tube.infinite_mean_control_certified


def test_reference_slab_cannot_silently_admit_a_larger_state_band():
    state = initialize_nodal_remainder((.5,) * 6, epi_lower=.25, epi_upper=.75)
    with pytest.raises(ValueError, match="inside the reference slab"):
        derive_c6_carried_tube(_profile(), state=state, timestep=1 / 16)
