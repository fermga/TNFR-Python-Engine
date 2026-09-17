"""Local pressure stencils are distinct from global cells and sign events."""

from fractions import Fraction as F

from tnfr.dynamics._euler_kernel import (
    NodalRemainderState, advance_nodal_remainder, initialize_nodal_remainder,
)
from tnfr.physics.c6_pressure_lattice import (
    derive_c6_pressure_lattice, observe_c6_frozen_pressure_stencil,
)
from tnfr.physics.nodal_remainder import derive_nodal_remainder_cell_horizon


def _reference():
    return derive_c6_pressure_lattice(
        phase=(0.,) * 6, epi_weight=1., phase_weight=1.,
        epi_lower=.375, epi_upper=.625,
    )


def test_remote_cell_exit_does_not_end_the_selected_stencil_or_its_pressure():
    epi = (.5, .5, .5, .5 + 2**-52, .5, .5 + 2**-52)
    # Node zero is just below its upper tie, while the selected center has no carry.
    initial = NodalRemainderState(epi, (F(1, 2**54) - F(1, 2**60),) + (F(0),) * 5, .375, .625)
    reference = _reference()
    observed = observe_c6_frozen_pressure_stencil(reference, state=initial, node=4, timestep=1/16)
    pressure = observed.initial_observation.pressure
    horizon = derive_nodal_remainder_cell_horizon(
        state=initial, timestep=1/16, capacity=(1.,) * 6, pressure=pressure,
    )
    assert observed.stencil == (3, 4, 5)
    assert F(observed.pressure) == F(1, 2**52)
    assert horizon.first_exit_step == 1
    assert horizon.coordinate_step_limits[4] == observed.initial_max_frozen_steps == 4
    assert observed.initial_first_exit_bound == 5
    assert observed.uniform_max_frozen_steps == 6

    step = advance_nodal_remainder(initial, timestep=1/16, capacity=(1.,) * 6, pressure=pressure)
    assert step.after.epi[0] != initial.epi[0]
    assert tuple(step.after.epi[i] for i in observed.stencil) == observed.stencil_epi
    refreshed = observe_c6_frozen_pressure_stencil(reference, state=step.after, node=4, timestep=1/16)
    assert refreshed.pressure == observed.pressure
    assert refreshed.gradient_index == observed.gradient_index
    assert refreshed.initial_max_frozen_steps == observed.initial_max_frozen_steps - 1
    assert refreshed.uniform_max_frozen_steps == observed.uniform_max_frozen_steps
    assert not refreshed.sign_hit_certified


def test_a_changed_stencil_can_preserve_its_gradient_and_pressure_exactly():
    delta = F(1, 2**54)
    before = (.5, .5, .5, float(F(1, 2) - 2*delta), .5, float(F(1, 2) - 3*delta))
    after = (.5, .5, .5, float(F(1, 2) - delta), .5, float(F(1, 2) - 4*delta))
    reference = _reference()
    observations = tuple(observe_c6_frozen_pressure_stencil(
        reference, state=initialize_nodal_remainder(epi, epi_lower=.375, epi_upper=.625),
        node=4, timestep=1/16,
    ) for epi in (before, after))
    left, right = observations
    assert left.stencil_epi != right.stencil_epi
    assert left.gradient_index == right.gradient_index == -5
    assert F(left.pressure) == F(right.pressure) == -5 * delta / 2
    assert left.center_cell == right.center_cell
    assert left.initial_max_frozen_steps == right.initial_max_frozen_steps
    assert not any(item.sign_hit_certified or item.graph_provenance_certified for item in observations)
    # The two supplied states do not claim an admitted transition between them.


def test_stationary_center_has_no_local_deadline_despite_a_finite_global_deadline():
    epi = (.5, .5, .5, .5 - 2**-53, .5, .5 + 2**-53)
    state = initialize_nodal_remainder(epi, epi_lower=.375, epi_upper=.625)
    observed = observe_c6_frozen_pressure_stencil(_reference(), state=state, node=4, timestep=1/16)
    horizon = derive_nodal_remainder_cell_horizon(
        state=state, timestep=1/16, capacity=(1.,) * 6,
        pressure=observed.initial_observation.pressure,
    )
    assert observed.pressure == 0. and observed.gradient_index == 0
    assert any(observed.initial_observation.pressure)
    assert horizon.first_exit_step is not None
    assert horizon.coordinate_step_limits[4] is None
    assert observed.initial_max_frozen_steps is observed.initial_first_exit_bound is None
    assert observed.uniform_max_frozen_steps is observed.uniform_first_exit_bound is None
    assert not observed.sign_hit_certified and not observed.positive_band_exit_certified


def test_exact_band_exit_can_end_the_bound_before_any_center_display_change():
    reference = derive_c6_pressure_lattice(
        phase=(.25, 0., .25, 0., 0., 0.), epi_weight=.5, phase_weight=.5,
        epi_lower=.375, epi_upper=.625,
    )
    state = initialize_nodal_remainder((.5,) * 6, epi_lower=.375, epi_upper=.5)
    timestep = 2.**-60
    observed = observe_c6_frozen_pressure_stencil(reference, state=state, node=1, timestep=timestep)
    horizon = derive_nodal_remainder_cell_horizon(
        state=state, timestep=timestep, capacity=(1.,) * 6,
        pressure=observed.initial_observation.pressure,
    )
    assert observed.pressure > 0
    assert observed.initial_first_exit_bound == horizon.first_exit_step == 1
    assert observed.uniform_max_frozen_steps > 0  # Other admissible carries begin below the band edge.
    assert horizon.first_exit_exact[1] > F(state.epi_upper)
    assert horizon.first_exit_leaves_band[1]
    assert not horizon.first_exit_leaves_cell[1]
    assert float(horizon.first_exit_exact[1]) == state.epi[1]
    assert not observed.sign_hit_certified and not observed.positive_band_exit_certified
