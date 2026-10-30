"""Independent static controls for conditional near-child-swap error bounds.

The incidence, symmetry and trigonometric checks use supplied states only.
No trajectory, event search, operator execution or attraction test is run.
"""

from fractions import Fraction as Q

import mpmath
import pytest

from tests.sine_evidence_helpers import forbid_organized_response_evaluation
from tnfr.dynamics import integrators, runtime
from tnfr.operators import network_stage, word_execution
from tnfr.operators.definitions_base import Operator
from tnfr.physics import _organized_source_child_error as owner
from tnfr.physics._organized_source_child_quotient import (
    _restrict_organized_source_child_quotient,
)

EDGES = tuple(
    (9 * child + slot, 9 * child + (slot + 1) % 9)
    for child in range(5)
    for slot in range(9)
) + tuple(
    (9 * a + slot, 9 * b + slot)
    for a, b in ((0, 1), (2, 3), (0, 2), (1, 3), (0, 4))
    for slot in range(9)
)
SWAP = tuple(9 * (0, 2, 1, 3, 4)[i // 9] + i % 9 for i in range(45))
ZERO = (Q(0),) * 45
UNIT = (Q(1),) * 45
MAIN = (Q(1, 64), -Q(1, 64)) + ZERO[2:]
EPSILON = Q(1, 2**20)
HORIZON = Q(1, 100)
LEVEL = Q(59, 40960)


def _exchange(values):
    return tuple(values[j] for j in SWAP)


def _project(values):
    return tuple((a + b) / 2 for a, b in zip(values, _exchange(values)))


def _perturb(values, epsilon):
    result = list(values)
    result[9] += epsilon
    result[18] -= epsilon
    return tuple(result)


def _gradient(values):
    result = [Q(0)] * 45
    for left, right in EDGES:
        result[left] += values[left] - values[right]
        result[right] += values[right] - values[left]
    return tuple(result)


def _bound(**overrides):
    return owner._bound_organized_source_child_error(
        **{
            "form": _perturb(MAIN, EPSILON),
            "phase_offsets": _perturb(ZERO, EPSILON),
            "capacities": UNIT,
            "horizon": HORIZON,
            "guard_level": LEVEL,
            **overrides,
        }
    )


@pytest.fixture(scope="module", autouse=True)
def no_event_flow_or_source_execution():
    def forbidden(*args, **kwargs):
        pytest.fail("Child error bounds cannot execute events or trajectories")

    with forbid_organized_response_evaluation(), pytest.MonkeyPatch.context() as patch:
        for object_, name in (
            (runtime, "step"),
            (runtime, "run"),
            (runtime, "_update_nodes"),
            (integrators.DefaultIntegrator, "integrate"),
            (Operator, "__call__"),
            (network_stage, "execute_neighbor_stage"),
            (network_stage, "execute_pointwise_stage"),
            (network_stage, "execute_operator_major_stage"),
            (word_execution, "execute_network_operator_stage"),
        ):
            patch.setattr(object_, name, forbidden)
        yield


@pytest.fixture(scope="module")
def mp():
    context = mpmath.mp.clone()
    context.dps = 100
    return context


@pytest.fixture(scope="module")
def report():
    return _bound()


def _mp(context, value):
    return context.mpf(value.numerator) / int(value.denominator)


def _contains(context, interval, value):
    assert _mp(context, interval.lo) <= value <= _mp(context, interval.hi)


def _fields(context, form, offsets):
    """Full two-row law and excess storage from independently listed edges."""
    gamma = 1 / (1023 * context.pi)
    phases = tuple(
        2 * context.pi * (i % 9 - 4) / 9 + _mp(context, value)
        for i, value in enumerate(offsets)
    )
    q = _gradient(form)
    current = [context.mpf(0)] * 45
    energy = context.mpf(0)
    for left, right in EDGES:
        gap = phases[right] - phases[left]
        sine = context.sin(gap)
        current[left] += sine
        current[right] -= sine
        energy += _mp(context, form[left] - form[right]) ** 2 / 2 + 1 - context.cos(gap)
    energy -= 45 * (1 - context.cos(2 * context.pi / 9))
    dx = tuple(
        (-_mp(context, value) + gamma * sine) / 4 for value, sine in zip(q, current)
    )
    dtheta = tuple(gamma * _mp(context, value) / 4 for value in q)
    return dx, dtheta, energy


def test_projection_is_exact_separate_from_strict_restriction(report):
    assert report.form == _perturb(MAIN, EPSILON)
    assert report.phase_offsets == _perturb(ZERO, EPSILON)
    assert report.projected_state.form == _project(report.form) == MAIN
    assert (
        report.projected_state.phase_offsets == _project(report.phase_offsets) == ZERO
    )
    assert report.hidden_form_coordinates == (EPSILON,) + ZERO[:8]
    assert report.hidden_phase_offset_coordinates == (EPSILON,) + ZERO[:8]
    for slot in range(9):
        for values, hidden, projected in (
            (report.form, report.hidden_form_coordinates, MAIN),
            (report.phase_offsets, report.hidden_phase_offset_coordinates, ZERO),
        ):
            assert values[9 + slot] == projected[9 + slot] + hidden[slot]
            assert values[18 + slot] == projected[18 + slot] - hidden[slot]
    with pytest.raises(ValueError, match="exactly equal"):
        _restrict_organized_source_child_quotient(
            form=report.form, phase_offsets=report.phase_offsets, capacities=UNIT
        )


def test_coefficients_follow_independent_incidence_and_hidden_support(report, mp):
    degrees = tuple(sum(i in edge for edge in EDGES) for i in range(45))
    hidden_mask = tuple(Q(9 <= i < 27) for i in range(45))
    local_phase_remainder = [Q(0)] * 45
    hidden_energy = Q(0)
    for left, right in EDGES:
        gap_bound = hidden_mask[left] + hidden_mask[right]
        cost = gap_bound**2 / 2
        hidden_energy += cost
        local_phase_remainder[left] += cost
        local_phase_remainder[right] += cost
    assert len(EDGES) == 90
    assert max(degrees) == 5
    assert max(local_phase_remainder) == 5
    assert hidden_energy == 54
    g = report.gamma_upper
    assert 1 / (1023 * mp.pi) <= _mp(mp, g)
    assert report.lipschitz_bound == max(degrees) * (1 + g) / 2
    assert report.quadratic_defect_coefficient == max(local_phase_remainder) * g / 4
    assert report.energy_error_bound == (
        2 * hidden_energy * report.hidden_radius_bound**2
        + (4 * len(EDGES) * report.form_radius_bound + 2 * len(EDGES))
        * report.projected_state_error_bound
    )


def test_growth_and_frozen_guard_control_are_independently_enclosed(report, mp):
    rate = _mp(mp, report.lipschitz_bound)
    duration = _mp(mp, HORIZON)
    exact_growth = mp.exp(rate * duration)
    exact_convolution = mp.expm1(rate * duration) / rate
    assert exact_growth <= _mp(mp, report.growth_upper) < exact_growth + mp.mpf("1e-30")
    assert exact_convolution <= _mp(mp, report.convolution_upper)
    assert report.initial_hidden_radius == EPSILON
    assert report.hidden_radius_bound == EPSILON * report.growth_upper
    assert report.projected_state_error_bound == (
        report.quadratic_defect_coefficient
        * report.hidden_radius_bound**2
        * report.convolution_upper
    )
    assert report.crossing_certified
    assert report.unavailable_reasons == ()
    assert 0 < report.crossing_time_error_bound < HORIZON
    assert report.crossing_time_error_bound == report.energy_error_bound / min(
        report.fine_loss_floor, report.core_loss_floor
    )


def test_initial_energy_and_whole_window_loss_use_fine_incidence(report, mp):
    _, _, fine_energy = _fields(mp, report.form, report.phase_offsets)
    _, _, core_energy = _fields(mp, MAIN, ZERO)
    _contains(mp, report.fine_initial_excess_bounds, fine_energy)
    _contains(mp, report.core_initial_excess_bounds, core_energy)
    assert abs(fine_energy - core_energy) <= _mp(mp, report.energy_error_bound)
    assert abs(core_energy - _mp(mp, Q(3, 2048))) < mp.mpf("1e-95")
    x0 = max(map(abs, report.form))
    g, t = report.gamma_upper, HORIZON
    drift = (25 * x0 + Q(25, 2) * g) * t + Q(125, 8) * g * t**2
    assert report.initial_form_radius == x0
    assert report.form_radius_bound == x0 + Q(5, 4) * g * t
    assert report.gradient_drift_bound == drift
    for values, floor in (
        (report.form, report.fine_loss_floor),
        (MAIN, report.core_loss_floor),
    ):
        gradient = _gradient(values)
        expected = sum((max(abs(q) - drift, Q(0)) ** 2 for q in gradient), Q(0)) / 4
        assert floor == expected > 0
        assert floor < sum((q**2 for q in gradient), Q(0)) / 4
    assert (
        report.fine_initial_guard_margin == report.fine_initial_excess_bounds.lo - LEVEL
    )
    assert (
        report.core_initial_guard_margin == report.core_initial_excess_bounds.lo - LEVEL
    )
    assert report.terminal_guard_margin == LEVEL - (
        report.core_initial_excess_bounds.hi
        - report.core_loss_floor * HORIZON
        + report.energy_error_bound
    )
    assert (
        min(
            report.fine_initial_guard_margin,
            report.core_initial_guard_margin,
            report.terminal_guard_margin,
        )
        > 0
    )


def test_zero_hidden_state_has_exact_zero_error_but_conditional_crossing():
    exact = _bound(form=MAIN, phase_offsets=ZERO)
    assert exact.initial_hidden_radius == 0
    assert exact.hidden_radius_bound == 0
    assert exact.projected_state_error_bound == 0
    assert exact.energy_error_bound == 0
    assert exact.crossing_certified
    assert exact.crossing_time_error_bound == 0
    unavailable = _bound(form=MAIN, phase_offsets=ZERO, guard_level=Q(0))
    assert unavailable.energy_error_bound == 0
    assert not unavailable.crossing_certified
    assert unavailable.crossing_time_error_bound is None
    assert unavailable.unavailable_reasons


def test_subgrid_hidden_radius_retains_exact_quadratic_scaling():
    epsilon = Q(1, 2**600)
    small = _bound(form=_perturb(MAIN, epsilon), phase_offsets=_perturb(ZERO, epsilon))
    doubled = _bound(
        form=_perturb(MAIN, 2 * epsilon), phase_offsets=_perturb(ZERO, 2 * epsilon)
    )
    assert small.initial_hidden_radius == epsilon
    assert 0 < small.projected_state_error_bound < Q(1, 2**1000)
    assert 0 < small.energy_error_bound < Q(1, 2**1000)
    assert doubled.hidden_radius_bound == 2 * small.hidden_radius_bound
    assert doubled.projected_state_error_bound == 4 * small.projected_state_error_bound
    assert doubled.energy_error_bound == 4 * small.energy_error_bound
    assert small.crossing_certified and doubled.crossing_certified


def test_exchange_reverses_hidden_coordinates_and_preserves_certificates(report):
    exchanged = _bound(
        form=_exchange(report.form), phase_offsets=_exchange(report.phase_offsets)
    )
    assert exchanged.projected_state == report.projected_state
    assert exchanged.hidden_form_coordinates == tuple(
        -v for v in report.hidden_form_coordinates
    )
    assert exchanged.hidden_phase_offset_coordinates == tuple(
        -v for v in report.hidden_phase_offset_coordinates
    )
    for name in (
        "initial_hidden_radius",
        "hidden_radius_bound",
        "projected_state_error_bound",
        "energy_error_bound",
        "gradient_drift_bound",
        "fine_loss_floor",
        "core_loss_floor",
        "crossing_time_error_bound",
    ):
        assert getattr(exchanged, name) == getattr(report, name)


def test_hidden_form_alone_requires_a_finite_error_bound(mp):
    result = _bound(phase_offsets=ZERO)
    assert result.hidden_phase_offset_coordinates == ZERO[:9]
    assert result.initial_hidden_radius == EPSILON
    assert result.projected_state_error_bound > 0
    dx, dtheta, _ = _fields(mp, result.form, ZERO)
    core_dx, core_dtheta, _ = _fields(mp, MAIN, ZERO)
    assert max(abs(a - b) for a, b in zip(_project(dx), core_dx)) < mp.mpf("1e-95")
    assert max(abs(a - b) for a, b in zip(_project(dtheta), core_dtheta)) < mp.mpf(
        "1e-95"
    )
    hidden_phase_rates = tuple((dtheta[9 + j] - dtheta[18 + j]) / 2 for j in range(9))
    assert max(map(abs, hidden_phase_rates)) > 0


def test_projected_sine_defect_is_even_quadratic_and_phase_row_is_linear(mp):
    core_form = _project(tuple(Q((5 * i) % 17 - 8, 4096) for i in range(45)))
    core_phase = _project(tuple(Q((7 * i) % 11 - 5, 4096) for i in range(45)))
    epsilon = Q(1, 1024)
    result = _bound(
        form=_perturb(core_form, epsilon),
        phase_offsets=_perturb(core_phase, epsilon),
    )
    dx, dtheta, _ = _fields(mp, result.form, result.phase_offsets)
    core_dx, core_dtheta, _ = _fields(mp, core_form, core_phase)
    projected_defect = tuple(a - b for a, b in zip(_project(dx), core_dx))
    gamma = 1 / (1023 * mp.pi)
    expected = [mp.mpf(0)] * 45
    hidden = tuple(
        (a - b) / 2
        for a, b in zip(result.phase_offsets, _exchange(result.phase_offsets))
    )
    for left, right in EDGES:
        gap = 2 * mp.pi * ((right % 9) - (left % 9)) / 9 + _mp(
            mp, core_phase[right] - core_phase[left]
        )
        hidden_gap = _mp(mp, hidden[right] - hidden[left])
        residual = gamma * mp.sin(gap) * (-2 * mp.sin(hidden_gap / 2) ** 2) / 4
        expected[left] += residual
        expected[right] -= residual
    assert max(abs(a - b) for a, b in zip(projected_defect, expected)) < mp.mpf("1e-95")
    assert (
        0
        < max(map(abs, projected_defect))
        <= _mp(mp, result.quadratic_defect_coefficient * epsilon**2)
    )
    assert max(abs(a - b) for a, b in zip(_project(dtheta), core_dtheta)) < mp.mpf(
        "1e-95"
    )
    for interval, value in zip(result.fine_initial.form_tau_rate_bounds, dx):
        _contains(mp, interval, value)
    for interval, value in zip(result.core_initial.phase_tau_rate_bounds, core_dtheta):
        _contains(mp, interval, value)


def test_same_guard_requires_both_initial_signs_and_an_endpoint_bracket():
    for level in (Q(1), Q(0)):
        result = _bound(guard_level=level)
        assert result.fine_loss_floor > 0 and result.core_loss_floor > 0
        assert result.energy_error_bound > 0
        assert not result.crossing_certified
        assert result.crossing_time_error_bound is None
        assert result.unavailable_reasons
        if level == 1:
            assert result.fine_initial_guard_margin < 0
            assert result.core_initial_guard_margin < 0
        else:
            assert result.terminal_guard_margin < 0


def test_pessimistic_loss_floor_is_unavailable_despite_positive_initial_loss():
    result = _bound(horizon=Q(1, 4))
    assert sum(q**2 for q in _gradient(result.form)) > 0
    assert result.fine_loss_floor == result.core_loss_floor == 0
    assert not result.crossing_certified
    assert result.crossing_time_error_bound is None
    assert result.projected_state_error_bound > 0
    assert result.unavailable_reasons


def test_zero_gradient_does_not_certify_a_crossing():
    result = _bound(form=ZERO, phase_offsets=ZERO, guard_level=-Q(1, 100))
    assert result.fine_loss_floor == result.core_loss_floor == 0
    assert not result.crossing_certified
    assert result.crossing_time_error_bound is None
    assert result.energy_error_bound == 0


@pytest.mark.parametrize(
    "overrides",
    (
        {"form": (True,) + ZERO[1:]},
        {"phase_offsets": (float("nan"),) + ZERO[1:]},
        {"form": ZERO[:-1]},
        {"phase_offsets": ZERO + (Q(0),)},
        {"capacities": UNIT[:-1]},
        {"capacities": (True,) + UNIT[1:]},
        {"capacities": (float("inf"),) + UNIT[1:]},
        {"capacities": (Q(0),) + UNIT[1:]},
        {"capacities": (Q(1) + Q(1, 2**600),) + UNIT[1:]},
        {"horizon": True},
        {"horizon": Q(0)},
        {"horizon": -Q(1, 100)},
        {"horizon": float("nan")},
        {"horizon": Q(1)},
        {"guard_level": False},
        {"guard_level": float("inf")},
    ),
)
def test_invalid_primitives_and_unsupported_laws_reject(overrides):
    with pytest.raises((TypeError, ValueError)):
        _bound(**overrides)
