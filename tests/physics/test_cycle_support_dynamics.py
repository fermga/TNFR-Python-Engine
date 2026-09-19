"""Independent exact controls for joint cycle support and nodal budgets."""

from dataclasses import FrozenInstanceError, replace
from fractions import Fraction

import numpy as np
import pytest

from tnfr.constants import DEFAULTS
from tnfr.constants.canonical import COUPLING_GENTLE, SHA_VF_FACTOR, UM_THETA_PUSH
from tnfr.physics.cycle_support_dynamics import (
    observe_cycle_support_balance,
    observe_cycle_support_euler,
    observe_cycle_support_reset,
)
from tnfr.utils import normalize_weights

F = Fraction


def _laplace(values):
    return tuple(
        value - (values[i - 1] + values[(i + 1) % len(values)]) / 2
        for i, value in enumerate(values)
    )


def _energy(values):
    return sum((value - values[i - 1]) ** 2 for i, value in enumerate(values)) / 2


def _sample():
    return observe_cycle_support_balance(
        (F(3, 4), F(1, 2), F(1, 4), F(1, 2)),
        (F(1, 2), 1, 2, 1),
        (F(1, 8), 0, F(-1, 8), 0),
        epi_weight=F(1, 2),
        vf_weight=F(1, 4),
        phase_weight=F(1, 4),
    )


def test_three_channel_pressure_and_fixed_dirichlet_balance():
    result = _sample()
    lx, ln, la = map(
        _laplace, (result.epi, result.capacity, result.phase_offset_over_pi)
    )
    expected = tuple(-x / 2 - nu / 4 - a / 4 for x, nu, a in zip(lx, ln, la))
    assert result.pressure == expected
    assert result.epi_rate == tuple(v * p for v, p in zip(result.capacity, expected))
    ly = _laplace(result.shifted_epi)
    assert result.dirichlet_energy == _energy(result.shifted_epi)
    assert result.dirichlet_derivative == -sum(
        v * value**2 for v, value in zip(result.capacity, ly)
    )
    assert result.dirichlet_derivative < 0
    assert result.pressure_identity_residual == (0,) * 4
    assert result.dirichlet_balance_residual == 0


def test_defaults_keep_full_channel_normalization_and_canonical_factors():
    result = observe_cycle_support_balance((1, 0, 0), (1,) * 3, (0,) * 3)
    defaults = normalize_weights(
        DEFAULTS["DNFR_WEIGHTS"], ("phase", "epi", "vf", "topo")
    )
    assert result.epi_weight == F.from_float(defaults["epi"])
    assert result.vf_weight == F.from_float(defaults["vf"])
    assert result.phase_weight == F.from_float(defaults["phase"])
    assert result.epi_weight + result.vf_weight + result.phase_weight < 1
    reset = observe_cycle_support_reset(result)
    assert reset.eta == F.from_float(UM_THETA_PUSH)
    assert reset.vf_sync == F.from_float(COUPLING_GENTLE)
    assert reset.silence_factor == F.from_float(SHA_VF_FACTOR)


def test_joint_zero_pressure_balance_can_have_nonuniform_epi():
    nu = tuple(F(value, 4) for value in (3, 4, 5, 4))
    phase = (F(1, 8), 0, F(-1, 8), 0)
    x = tuple(F(2) - v / 2 - a / 2 for v, a in zip(nu, phase))
    before = observe_cycle_support_balance(
        x,
        nu,
        phase,
        epi_weight=F(1, 2),
        vf_weight=F(1, 4),
        phase_weight=F(1, 4),
    )
    assert before.shifted_epi == (2,) * 4
    assert before.pressure == before.epi_rate == (0,) * 4
    assert _energy(before.epi) > 0
    after = observe_cycle_support_reset(before)
    assert after.before.epi == after.after.epi
    assert after.cross_term == 0
    assert after.energy_change == after.quadratic_term > 0


def test_reset_pressure_formula_and_energy_are_exact_independent_expansions():
    original = _sample()
    x = tuple(
        F(2) - nu / 2 - phase / 2
        for nu, phase in zip(original.capacity, original.phase_offset_over_pi)
    )
    before = observe_cycle_support_balance(
        x,
        original.capacity,
        original.phase_offset_over_pi,
        epi_weight=F(1, 2),
        vf_weight=F(1, 4),
        phase_weight=F(1, 4),
    )
    eta, s, q = F(1, 3), F(1, 4), F(3, 4)
    result = observe_cycle_support_reset(before, eta=eta, vf_sync=s, silence_factor=q)
    ln = _laplace(before.capacity)
    lln, lla = _laplace(ln), _laplace(_laplace(before.phase_offset_over_pi))
    expected = tuple(
        ((1 - q) * first + q * s * second + eta * third) / 4
        for first, second, third in zip(ln, lln, lla)
    )
    assert result.after.pressure == expected
    assert result.quadratic_term == _energy(result.support_jump)
    assert result.identity_residual == 0


def test_correlated_phase_and_capacity_can_cancel_the_reset_pressure():
    mode = (F(1), F(0), F(-1), F(0))  # L mode = mode.
    nu = tuple(1 + v / 8 for v in mode)
    phase = tuple(-3 * v / 16 for v in mode)
    x = tuple(2 - v / 2 - a / 2 for v, a in zip(nu, phase))
    before = observe_cycle_support_balance(
        x,
        nu,
        phase,
        epi_weight=F(1, 2),
        vf_weight=F(1, 4),
        phase_weight=F(1, 4),
    )
    reset = observe_cycle_support_reset(
        before, eta=F(1, 2), vf_sync=F(1, 2), silence_factor=F(1, 2)
    )
    assert reset.before.pressure == reset.after.pressure == (0,) * 4
    assert reset.before.capacity != reset.after.capacity
    assert reset.before.phase_offset_over_pi != reset.after.phase_offset_over_pi
    assert reset.energy_change == 0


def test_reset_energy_change_is_signed_and_not_a_dissipation_claim():
    before = observe_cycle_support_balance(
        (0,) * 4,
        (1,) * 4,
        (1, -1, 1, -1),
        epi_weight=1,
        vf_weight=0,
        phase_weight=1,
    )
    reset = observe_cycle_support_reset(
        before, eta=F(1, 2), vf_sync=0, silence_factor=1
    )
    assert reset.after.phase_offset_over_pi == (0,) * 4
    assert reset.energy_change == -8
    assert reset.cross_term == -16 and reset.quadratic_term == 8
    assert reset.identity_residual == 0


def test_zero_capacity_keeps_the_budget_but_not_pressure_rate_equivalence():
    before = observe_cycle_support_balance(
        (1, 0, -1, 0),
        (0,) * 4,
        (0,) * 4,
        epi_weight=1,
        vf_weight=0,
        phase_weight=0,
    )
    assert any(before.pressure)
    assert before.epi_rate == (0,) * 4
    assert before.dirichlet_derivative == 0
    step = observe_cycle_support_euler(before, dt=100)
    assert step.after == before
    assert step.convex_step_condition
    assert step.energy_change == step.quadratic_energy_change == 0
    reset = observe_cycle_support_reset(_sample(), silence_factor=0)
    assert reset.after.capacity == reset.after.epi_rate == (0,) * 4
    assert reset.identity_residual == 0


def test_euler_budget_records_the_positive_remainder_and_stability_boundary():
    before = observe_cycle_support_balance(
        (1, -1, 1, -1),
        (1,) * 4,
        (0,) * 4,
        epi_weight=1,
        vf_weight=0,
        phase_weight=0,
    )
    midpoint = observe_cycle_support_euler(before, dt=F(1, 2))
    assert midpoint.after.epi == (0,) * 4
    assert midpoint.linear_energy_change == -16
    assert midpoint.quadratic_energy_change == 8
    assert midpoint.energy_change == -8
    boundary = observe_cycle_support_euler(before, dt=1)
    assert boundary.convex_step_condition and boundary.energy_change == 0
    unstable = observe_cycle_support_euler(before, dt=F(3, 2))
    assert not unstable.convex_step_condition
    assert unstable.energy_change == 24
    assert unstable.identity_residual == 0


def test_adjacent_exact_reset_and_euler_steps_have_a_finite_budget():
    initial = state = _sample()
    resets = linear = quadratic = F(0)
    for _ in range(5):
        reset = observe_cycle_support_reset(state)
        resets += reset.energy_change
        step = observe_cycle_support_euler(reset.after, dt=F(1, 8))
        linear += step.linear_energy_change
        quadratic += step.quadratic_energy_change
        assert step.convex_step_condition
        state = step.after
    assert state.dirichlet_energy - initial.dirichlet_energy == (
        resets + linear + quadratic
    )
    assert linear < 0 and quadratic > 0


def test_public_cached_fields_are_recomputed_before_a_transition():
    original = _sample()
    tampered = replace(
        original,
        pressure=(F(0),) * 4,
        epi_rate=(F(0),) * 4,
        shifted_epi=(F(100),) * 4,
        dirichlet_energy=F(-100),
        dirichlet_derivative=F(99),
    )
    assert observe_cycle_support_reset(tampered) == (
        observe_cycle_support_reset(original)
    )
    assert observe_cycle_support_euler(tampered, dt=F(1, 4)) == (
        observe_cycle_support_euler(original, dt=F(1, 4))
    )


def test_exact_reader_detachment_and_immutability():
    x = np.array([1, 0, -1], dtype=np.int64)
    nu = [F(np.int64(1))] * 3
    tiny = F(1, 10**500)
    before = observe_cycle_support_balance(x, nu, (tiny, 0, -tiny))
    x[0], nu[0] = 99, 99
    assert before.epi == (1, 0, -1) and before.capacity == (1,) * 3
    assert before.phase_offset_over_pi == (tiny, 0, -tiny)
    assert all(type(value.numerator) is int for value in before.capacity)
    with pytest.raises(FrozenInstanceError):
        before.epi_weight = F(0)


@pytest.mark.parametrize(
    "epi,nu,phase",
    [
        ((), (), ()),
        ((0, 0), (1, 1), (0, 0)),
        ((0,) * 3, (1,) * 2, (0,) * 3),
        ((0,) * 3, (1,) * 3, (0,) * 2),
        ((0,) * 3, (-1, 1, 1), (0,) * 3),
        ((True, 0, 0), (1,) * 3, (0,) * 3),
        ((0,) * 3, (1,) * 3, (float("nan"), 0, 0)),
        ((0,) * 3, (1,) * 3, {0, 1, 2}),
    ],
)
def test_invalid_coordinates_are_rejected(epi, nu, phase):
    with pytest.raises((TypeError, ValueError)):
        observe_cycle_support_balance(epi, nu, phase)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"epi_weight": 0},
        {"phase_weight": -1},
        {"vf_weight": -1},
        {"phase_weight": True},
        {"epi_weight": float("inf")},
    ],
)
def test_invalid_coefficients_are_rejected(kwargs):
    with pytest.raises((TypeError, ValueError)):
        observe_cycle_support_balance((0,) * 3, (1,) * 3, (0,) * 3, **kwargs)


@pytest.mark.parametrize("name", ["eta", "vf_sync", "silence_factor"])
@pytest.mark.parametrize("value", [F(-1, 10), F(11, 10), True])
def test_invalid_reset_factors_are_rejected(name, value):
    with pytest.raises((TypeError, ValueError)):
        observe_cycle_support_reset(_sample(), **{name: value})


@pytest.mark.parametrize("dt", [0, -1, True, float("inf")])
def test_invalid_euler_duration_is_rejected(dt):
    with pytest.raises((TypeError, ValueError)):
        observe_cycle_support_euler(_sample(), dt=dt)


def test_transition_requires_a_balance_record():
    with pytest.raises(TypeError):
        observe_cycle_support_reset({})
    with pytest.raises(TypeError):
        observe_cycle_support_euler(None, dt=1)
