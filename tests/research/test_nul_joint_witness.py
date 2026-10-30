"""A fixed analytical positive-width NUL successor, without ODE integration.

The law supplies the nominal whole-cell boxes in the canonical chapter. These
tests check the assembled arithmetic against independently derived rational
budgets; a comparison report cannot authenticate its nominal solution premise.
"""

from fractions import Fraction as Q

import pytest

from tnfr.research.nul_successor import bound_nul_successor_cells

EPSILON = Q(1, 65536)
HORIZON = Q(1, 64)


@pytest.fixture(scope="module")
def witness():
    epsilon = float(EPSILON)
    return bound_nul_successor_cells(
        factor=0.5,
        parent=0,
        child=1,
        form_bounds=((2.0 - epsilon, 2.0, 2.0 + epsilon), (0.0,) * 3),
        capacity_bounds=((2.0 - epsilon, 2.0, 2.0 + epsilon), (1.0,) * 3),
        phase_bounds=((-epsilon, 0.0, epsilon), (0.0,) * 3),
        pressure_bounds=(-epsilon, 0.0, epsilon),
        edges=((0, 1),),
        conductances=(1,),
        degrees=(1, 1),
        reference_weight=1,
        nominal_form=(1, 0),
        nominal_phase=(0, 0),
        nominal_capacities=(1, 1),
        form_cells=(((Q(127, 128), Q(129, 128)), (Q(-1, 128), Q(1, 128))),),
        phase_cells=(((0, Q(1, 128)), (Q(-1, 128), 0)),),
        edge_allowances=(Q(1, 16),),
        durations=(HORIZON,),
        epi_min=-4.0,
        epi_max=4.0,
    )


def test_fixed_joint_image_and_affine_comparison_fit_analytic_budget(witness):
    result, family = witness.run, witness.family
    assert result.status == "completed"
    assert result.attempted_cells == result.completed_cells == 1
    assert witness.nominal_form == (1, 0)
    assert witness.nominal_capacities == (1, 1)
    assert family.initial_norm_squared == 5 * EPSILON**2 / 4
    assert family.initial_radius_upper < 3 * EPSILON / 2
    assert family.observation_factor == 1
    step = result.steps[0]
    assert result.relative_capacity_error == EPSILON / 2
    assert step.gamma_upper < 1
    # Nominal |sin(phi)| <= 1/64, |z| <= 65/64. Only node 0 has
    # mobility mismatch, of magnitude epsilon/2. Both edge energies count.
    independent_forcing_square = (EPSILON / 128) ** 2 + (65 * EPSILON / 128) ** 2
    assert step.forcing_norm_squared <= independent_forcing_square
    assert step.forcing_upper < 11 * EPSILON / 64 < EPSILON
    # exp(H) < 1/(1-H) implies the positive affine bound below.
    independent_radius = (3 * EPSILON / 2 + HORIZON * EPSILON) / (1 - HORIZON)
    assert independent_radius == 97 * EPSILON / 63
    assert result.final_radius_upper < independent_radius < 2 * EPSILON
    assert result.final_portrait_error_upper == result.final_radius_upper
    assert all(margin > 0 for margin in step.edge_margins)


def test_full_law_rational_margins_give_uniform_directional_observation(witness):
    # Direct nominal law: |z'| <= 2/pi < 2/3, phi'=2z/pi.
    z_lower, z_upper = 1 - Q(1, 64), 1 + Q(1, 64)
    assert 2 * HORIZON / 3 < 1 - z_lower
    assert 2 * z_upper * HORIZON / 3 < Q(1, 64)
    # pi < 4 gives the lower endpoint phase bound independently of the
    # supplied cells. Mean conservation gives the node-wise boxes in the fixture.
    phase_endpoint_lower = 2 * z_lower * HORIZON / 4
    assert phase_endpoint_lower == Q(63, 8192)
    error = witness.run.final_portrait_error_upper
    assert z_lower - error > Q(31, 32)
    assert phase_endpoint_lower - error > Q(1, 256)
    assert z_lower - error > 0  # The same radius floor holds throughout.

    tangent_lower = phase_endpoint_lower / z_upper
    assert tangent_lower == Q(63, 8320)
    # Integral inequality atan(r) >= r/(1+r^2); no floating atan evaluation.
    angle_lower = tangent_lower / (1 + tangent_lower**2)
    assert angle_lower > Q(1, 256)
    assert error / z_lower < 3 * EPSILON < Q(1, 2)
    # asin(a)<2a in this domain bounds BOTH endpoint angular errors.
    endpoint_error_upper = 6 * EPSILON
    assert 2 * error / z_lower < endpoint_error_upper
    assert Q(1, 256) - 2 * endpoint_error_upper > Q(1, 512)
