"""Independent centered-mode, rounding and finite-band bootstrap controls."""

import math
from dataclasses import replace
from fractions import Fraction as F

import pytest

from tnfr.dynamics._euler_kernel import (
    NodalRemainderState,
    advance_nodal_remainder,
    initialize_nodal_remainder,
)
from tnfr.physics.c6_carried_profile import (
    derive_c6_carried_profile,
    observe_c6_carried_profile_step,
)
from tnfr.physics.c6_carried_tube import (
    derive_c6_carried_band_horizon,
    derive_c6_carried_contraction,
    derive_c6_carried_tube,
    observe_c6_carried_cut_exclusion,
)
from tnfr.physics.c6_pressure_lattice import (
    derive_c6_pressure_lattice,
    observe_c6_pressure_lattice,
)


def _profile(weight=0.5):
    return derive_c6_carried_profile(
        derive_c6_pressure_lattice(
            phase=(0.0,) * 6,
            epi_weight=weight,
            phase_weight=0.5,
            epi_lower=0.375,
            epi_upper=0.625,
        )
    )


def _state(**changes):
    args = dict(epi_lower=0.375, epi_upper=0.625)
    args.update(changes)
    return initialize_nodal_remainder((0.5,) * 6, **args)


@pytest.mark.parametrize(
    "weight,mode,eigenvalue",
    [
        (math.nextafter(0.8, -math.inf), (2, 1, -1, -2, -1, 1), F(1, 2)),
        (0.8, (1, -1, 1, -1, 1, -1), F(2)),
    ],
)
def test_represented_values_on_opposite_sides_of_the_sharp_mode_switch(
    weight, mode, eigenvalue
):
    ref = derive_c6_carried_contraction(_profile(weight), timestep=1.0)
    source = tuple(map(F, mode))
    image = tuple(
        value - F(weight) * (value - (source[i - 1] + source[(i + 1) % 6]) / 2)
        for i, value in enumerate(source)
    )
    factor = 1 - F(weight) * eigenvalue
    assert image == tuple(factor * value for value in source)
    assert ref.norm_factor == abs(factor)
    assert sum(value**2 for value in image) == ref.energy_factor * sum(
        value**2 for value in source
    )
    assert ref.energy_factor == ref.norm_factor**2 < ref.norm_factor
    assert all(
        sum(row) == 1 for row in ref.transition
    )  # The uniform mode never contracts.
    assert F(math.nextafter(0.8, -math.inf)) < F(4, 5) < F(0.8)


def test_carry_feedback_can_sustain_nonzero_centered_error_under_zero_visible_pressure():
    profile = _profile()
    carry = F(1, 2**54)
    residuals = (carry, -carry, F(0), F(0), F(0), -carry)
    state = NodalRemainderState((0.5625,) * 6, residuals, 0.375, 0.625)
    tube = derive_c6_carried_tube(profile, state=state, timestep=0.25)
    step = advance_nodal_remainder(
        state, timestep=0.25, capacity=(1.0,) * 6, pressure=(0.0,) * 6
    )
    observed = observe_c6_carried_profile_step(profile, step=step)
    assert tube.carry_bound == carry
    assert observed.carry_feedback[0] == 2 * profile.forced_balance.epi_weight * carry
    assert sum(observed.carry_feedback) == 0
    assert observed.error_before == observed.error_after
    assert any(observed.error_before) and tube.initial_energy > 0
    assert tube.contraction.energy_factor < 1
    assert tube.energy_floor >= tube.initial_energy
    assert not tube.infinite_mean_control_certified


@pytest.mark.parametrize("weight", [0.3, math.nextafter(1.0, 0.0), math.ulp(0.0)])
def test_uniform_rounding_envelopes_cover_actual_extreme_slab_reducers(weight):
    profile = _profile(weight)
    state = initialize_nodal_remainder(
        (0.375, 0.625, 0.375, 0.625, 0.375, 0.625), epi_lower=0.375, epi_upper=0.625
    )
    tube = derive_c6_carried_tube(profile, state=state, timestep=0.5)
    reading = observe_c6_pressure_lattice(profile.lattice, epi=state.epi)
    assert all(
        abs(value) <= tube.product_error_bound for value in reading.epi_reduction_error
    )
    assert all(
        abs(value) <= tube.assembly_error_bound for value in reading.assembly_error
    )
    assert tube.product_error_bound >= F(1, 2**1075)
    assert tube.rounding_bound == tube.product_error_bound + tube.assembly_error_bound
    assert (
        tube.centered_forcing_norm_squared_bound == 6 * tube.forcing_component_bound**2
    )
    if weight == math.ulp(0.0):
        assert not any(reading.epi_contributions)
        assert set(map(abs, reading.epi_reduction_error)) == {F(1, 2**1076)}


def test_narrower_state_band_controls_admission_even_with_the_same_wider_pressure_slab():
    profile = _profile()
    wide = derive_c6_carried_band_horizon(
        derive_c6_carried_tube(profile, state=_state(), timestep=0.25)
    )
    narrow_state = _state(epi_upper=0.5)
    narrow = derive_c6_carried_band_horizon(
        derive_c6_carried_tube(profile, state=narrow_state, timestep=0.25)
    )
    assert wide.tube_initially_admitted and wide.maximum_steps > 0
    assert narrow.minimum_initial_margin == 0
    assert not narrow.tube_initially_admitted
    assert narrow.maximum_steps is None and not narrow.unbounded_conditional_prefix
    # Rejection of this sufficient tube is not rejection of the actual zero-pressure step.
    step = advance_nodal_remainder(
        narrow_state, timestep=0.25, capacity=(1.0,) * 6, pressure=(0.0,) * 6
    )
    assert step.before.exact_epi == step.after.exact_epi
    assert not narrow.actual_band_exit_certified


def test_band_horizon_is_the_exact_maximal_integer_for_its_sufficient_inequality():
    tube = derive_c6_carried_tube(_profile(), state=_state(), timestep=0.25)
    horizon = derive_c6_carried_band_horizon(tube)
    maximum = horizon.maximum_steps
    assert type(maximum) is int and maximum > 10**12
    margin, rate = horizon.minimum_initial_margin, tube.mean_increment_bound
    squared = F(5, 6) * tube.energy_bound
    reserve = margin - maximum * rate
    next_reserve = margin - (maximum + 1) * rate
    assert reserve >= 0 and reserve**2 >= squared
    assert next_reserve < 0 or next_reserve**2 < squared
    assert (
        horizon.next_step_margin == next_reserve and horizon.next_step_passes is False
    )
    assert horizon.centered_coordinate_squared_bound == squared
    assert (
        not horizon.actual_band_exit_certified and not horizon.future_runtime_certified
    )


def test_forged_zero_mean_rate_or_energy_does_not_create_an_infinite_certificate():
    tube = derive_c6_carried_tube(_profile(), state=_state(), timestep=0.25)
    expected = derive_c6_carried_band_horizon(tube)
    forged = replace(
        tube, mean_increment_bound=F(0), energy_bound=F(0), initial_mean=F(1, 2)
    )
    result = derive_c6_carried_band_horizon(forged)
    assert result == expected
    assert result.tube.mean_increment_bound > 0
    assert not result.unbounded_conditional_prefix


def test_forged_spectral_and_profile_caches_are_not_authority_for_the_horizon():
    tube = derive_c6_carried_tube(_profile(), state=_state(), timestep=0.25)
    bad_balance = replace(
        tube.contraction.profile.forced_balance, relative_profile=(F(100),) * 6
    )
    bad_profile = replace(tube.contraction.profile, forced_balance=bad_balance)
    bad_contraction = replace(
        tube.contraction, profile=bad_profile, norm_factor=F(0), energy_factor=F(0)
    )
    forged = replace(tube, contraction=bad_contraction)
    assert derive_c6_carried_band_horizon(forged) == derive_c6_carried_band_horizon(
        tube
    )


def test_cut_not_excluded_does_not_certify_a_future_sign_transition():
    tube = derive_c6_carried_tube(_profile(), state=_state(), timestep=0.25)
    result = observe_c6_carried_cut_exclusion(tube, node=4)
    assert result.profile_gradient_index == result.nonpositive_cut == 0
    assert result.remaining_laplacian_distance == -2 * tube.carry_bound
    assert not result.cut_excluded
    assert not result.cut_reachability_certified
    forged = replace(tube, carry_bound=F(0), energy_bound=F(0))
    assert observe_c6_carried_cut_exclusion(forged, node=4) == result
