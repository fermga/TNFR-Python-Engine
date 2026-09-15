"""Independent scalar controls for the joint cycle model's limit boundaries."""

from fractions import Fraction

from tnfr.constants.canonical import COUPLING_GENTLE, SHA_VF_FACTOR
from tnfr.physics.cycle_support_dynamics import (
    observe_cycle_support_balance, observe_cycle_support_euler,
    observe_cycle_support_reset,
)
from tnfr.physics.reversible_eigenmode_reference import _negative_exp_bounds


F = Fraction
V = (1, -1) * 4


def test_default_checkerboard_reduces_to_two_scalar_nodal_equations():
    base = observe_cycle_support_balance((1,) * 8, (1,) * 8, (0,) * 8)
    e, f = base.epi_weight, base.vf_weight
    mean_capacity, contrast_capacity = F(1), F(-1, 4)
    contrast_epi = -f * contrast_capacity / e
    before = observe_cycle_support_balance(
        tuple(1 + contrast_epi * v for v in V),
        tuple(mean_capacity + contrast_capacity * v for v in V), (0,) * 8,
    )
    assert before.pressure == (0,) * 8
    reset = observe_cycle_support_reset(before)
    q, s = F(SHA_VF_FACTOR), F(COUPLING_GENTLE)
    assert 0 < s < F(1, 2) and 0 < q < 1
    next_mean = q * mean_capacity
    next_contrast = q * (1 - 2 * s) * contrast_capacity
    assert reset.after.capacity == tuple(next_mean + next_contrast * v for v in V)
    pressure_amplitude = -2 * (e * contrast_epi + f * next_contrast)
    assert reset.after.pressure == tuple(pressure_amplitude * v for v in V)
    mean_rate = next_contrast * pressure_amplitude
    contrast_rate = next_mean * pressure_amplitude
    assert reset.after.epi_rate == tuple(mean_rate + contrast_rate * v for v in V)
    assert mean_rate > 0 and contrast_rate < 0
    # The continuous-flow comparison, not the Euler product, has this positive
    # lower bound. Its proof uses b' >= -2e*A_k*b on every held segment.
    h = F(1, 2)
    exposure = mean_capacity * h * q / (1 - q)
    lower, upper = _negative_exp_bounds(2 * e * exposure)
    assert 0 < contrast_epi * lower <= contrast_epi * upper < contrast_epi


def test_nondamping_phase_boundary_has_an_exact_euler_two_cycle():
    # This is an omitted-SHA algebraic control: q=1 is not admitted by SHA.
    # Euler rho differs from the exponential factor of exact continuous flow.
    base = observe_cycle_support_balance((1,) * 8, (1,) * 8, (0,) * 8)
    e, w, dt, amplitude = base.epi_weight, base.phase_weight, F(1, 4), F(1, 64)
    rho = 1 - 2 * e * dt
    assert 0 < rho < 1
    response = (w * amplitude / e) * (1 - rho) / (1 + rho)
    initial = observe_cycle_support_balance(
        tuple(1 - response * v for v in V), (1,) * 8,
        tuple(amplitude * v for v in V),
    )
    first_reset = observe_cycle_support_reset(initial, eta=1, silence_factor=1)
    first = observe_cycle_support_euler(first_reset.after, dt=dt)
    assert first.convex_step_condition
    assert first.after.epi == tuple(1 + response * v for v in V)
    second_reset = observe_cycle_support_reset(first.after, eta=1, silence_factor=1)
    second = observe_cycle_support_euler(second_reset.after, dt=dt)
    assert second.after == initial
    assert first_reset.energy_change + second_reset.energy_change > 0
    assert (
        first_reset.energy_change + first.energy_change
        + second_reset.energy_change + second.energy_change
    ) == 0
