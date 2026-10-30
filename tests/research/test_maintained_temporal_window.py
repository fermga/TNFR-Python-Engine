"""Independent static temporal-window controls; no event or trajectory runs."""

from copy import deepcopy
from dataclasses import FrozenInstanceError
from fractions import Fraction as Q

import mpmath
import pytest

from tnfr.dynamics import integrators, relational
from tnfr.mathematics import _validated_taylor
from tnfr.mathematics._rational_interval import I, pi_interval, sqrt
from tnfr.operators import network_stage, word_execution
from tnfr.research import maintained_temporal_window as owner


@pytest.fixture(scope="module", autouse=True)
def no_execution():
    def forbidden(*args, **kwargs):
        pytest.fail("static window bounds cannot execute an event or trajectory")

    with pytest.MonkeyPatch.context() as patch:
        for module, name in (
            (integrators, "rk4_coupled_step"),
            (relational, "step_relational_exchange"),
            (_validated_taylor, "validated_box_taylor_step"),
            (network_stage, "execute_self_organization_stage"),
            (word_execution, "execute_network_operator_stage"),
        ):
            patch.setattr(module, name, forbidden)
        yield


def _small(**changes):
    values = dict(
        epi=(0, 1),
        phases=(I(0), pi_interval() / 6),
        capacities=(2, 3),
        edges=((0, 1),),
        conductances=(2,),
        shifts=(0,),
        horizon=0,
        target_index=0,
    )
    values.update(changes)
    return owner.bound_node_history_window(**values)


def _same_value(actual, expected):
    defect = actual - expected
    assert defect.contains(0)
    assert defect.abs_max < Q(1, 10**30)


def test_two_node_instantaneous_chain_rule_keeps_target_orientation_and_weights():
    left, right = _small(), _small(target_index=1)
    pi, root_three = pi_interval(), sqrt(I(3))
    # W=2, x=(0,1), nu=(2,3), gap=pi/6 imply theta_dot=(-4,6)/pi.
    _same_value(left.pressure, 1 / (2 * pi))
    _same_value(right.pressure, -1 / (2 * pi))
    _same_value(left.form_rate, 1 / pi)
    _same_value(right.form_rate, -3 / (2 * pi))
    _same_value(left.form_acceleration, 10 * root_three / pi**2)
    _same_value(right.form_acceleration, -15 * root_three / pi**2)
    assert left.form == I(0) and right.form == I(1)
    assert left.edge_phase_differences == right.edge_phase_differences
    _same_value(left.edge_phase_velocities[0], 10 / pi)
    assert left.retained_edge_indices == right.retained_edge_indices == (0,)
    assert left.horizon == Q(0)


def test_global_jerk_bound_uses_storage_speed_and_phase_acceleration():
    report = _small()
    pi = pi_interval()
    # H=2-sqrt(3)/2, V^2=100H/pi^2, B=50/pi^2 and M_0=2/pi.
    expected = (500 - 100 * sqrt(I(3))) / pi**3
    assert type(report.third_form_derivative_bound) is Q
    assert report.third_form_derivative_bound >= expected.lo
    assert report.third_form_derivative_bound - expected.lo < Q(1, 10**30)
    assert report.third_form_derivative_bound > 0


@pytest.mark.parametrize("mode", ("window", "box"))
def test_three_node_nonuniform_weighted_law_matches_independent_chain_rule(mode):
    forms = (Q(-1, 2), Q(3, 4), Q(1, 8))
    phases = (Q(-1, 5), Q(1, 4), Q(3, 8))
    capacities = (Q(1), Q(2), Q(4))
    edges, weights = ((0, 1), (0, 2), (1, 2)), (Q(1, 2), Q(3, 4), Q(5, 4))
    calculate = (
        owner.bound_node_history_window
        if mode == "window"
        else owner.bound_node_history_box
    )
    reports = tuple(
        calculate(
            epi=forms,
            phases=phases,
            capacities=capacities,
            edges=edges,
            conductances=weights,
            shifts=(0, 0, 0),
            target_index=target,
            **({"horizon": 0} if mode == "window" else {}),
        )
        for target in range(3)
    )
    with mpmath.workdps(100):

        def mp(value):
            return mpmath.mpf(value.numerator) / value.denominator

        def contains(interval, value):
            assert mp(interval.lo) <= value <= mp(interval.hi)

        x, theta, nu = (
            tuple(map(mp, values)) for values in (forms, phases, capacities)
        )
        adjacent = tuple(
            tuple(
                (b if a == i else a, mp(w))
                for (a, b), w in zip(edges, weights)
                if i in (a, b)
            )
            for i in range(3)
        )
        omega = tuple(
            nu[i] * sum(w * (x[i] - x[j]) for j, w in row) / (len(row) * mpmath.pi)
            for i, row in enumerate(adjacent)
        )
        for i, (report, row) in enumerate(zip(reports, adjacent)):
            pressure = sum(mpmath.sin(theta[j] - theta[i]) for j, _ in row) / (
                len(row) * mpmath.pi
            )
            acceleration = (
                nu[i]
                * sum(
                    mpmath.cos(theta[j] - theta[i]) * (omega[j] - omega[i])
                    for j, _ in row
                )
                / (len(row) * mpmath.pi)
            )
            contains(report.pressure, pressure)
            contains(report.form_rate, nu[i] * pressure)
            contains(report.form_acceleration, acceleration)
            contains(report.form, x[i])


def test_original_branch_lift_is_preserved_without_changing_local_pressure():
    plain = _small()
    lifted = _small(phases=(0, pi_interval() / 6 + 2 * pi_interval()), shifts=(1,))
    for name in ("pressure", "form_rate", "form_acceleration", "form"):
        _same_value(getattr(lifted, name), getattr(plain, name))
    _same_value(lifted.edge_phase_differences[0], pi_interval() / 6)
    assert lifted.motion.phases[1].lo > 2 * pi_interval().hi
    assert lifted.principal_margins[0] > 0


def test_short_window_contains_initial_values_and_expands_with_horizon():
    instant = _small()
    small, large = _small(horizon=Q(1, 1000)), _small(horizon=Q(1, 100))
    for name in ("pressure", "form_rate", "form_acceleration", "form"):
        assert getattr(instant, name).subset_of(getattr(small, name))
        assert getattr(small, name).subset_of(getattr(large, name))
    assert small.form_acceleration.lo > 0
    assert small.pressure.lo > 0
    assert small.form.lo == 0 and small.form.hi > 0
    assert large.principal_margins[0] <= small.principal_margins[0]
    assert small.retained_edge_indices == large.retained_edge_indices == (0,)
    assert small.third_form_derivative_bound == large.third_form_derivative_bound


def test_exact_stationary_solution_remains_enclosed_and_energy_retains_identity():
    report = _small(epi=(Q(-3, 4), Q(-3, 4)), phases=(0, 0), horizon=10)
    # Equal form and phase give the exact constant solution at every time.
    for name in ("pressure", "form_rate", "form_acceleration"):
        assert getattr(report, name).contains(0)
    assert report.form.contains(Q(-3, 4))
    assert report.edge_phase_differences[0].contains(0)
    assert report.edge_phase_velocities[0].contains(0)
    # The quadratic tube alone is too wide, but source H=0 supplies retention.
    assert report.edge_phase_differences[0].abs_max > pi_interval().hi
    assert report.principal_margins[0] > 3
    assert report.retained_edge_indices == (0,)


def test_unresolved_long_window_abstains_without_claiming_a_crossing_or_history():
    report = _small(horizon=100)
    assert report.principal_margins[0] < 0
    assert report.retained_edge_indices == ()
    assert not hasattr(report, "crossing_certified")
    assert not hasattr(report, "observed_acceleration")
    with pytest.raises(FrozenInstanceError):
        report.horizon = Q(0)


@pytest.mark.parametrize("target", (True, False, Q(0), 0.0, -1, 2))
def test_target_index_is_not_coerced(target):
    with pytest.raises((TypeError, ValueError)):
        _small(target_index=target)


@pytest.mark.parametrize("horizon", (True, False, -1, float("nan"), float("inf")))
def test_invalid_clock_is_rejected_before_primitive_consumption(horizon):
    def unavailable():
        raise AssertionError("invalid horizon must reject before consuming state")
        yield

    with pytest.raises((TypeError, ValueError)):
        _small(horizon=horizon, epi=unavailable())


@pytest.mark.parametrize(
    "changes",
    (
        {"epi": (False, 1)},
        {"epi": (float("nan"), 1)},
        {"phases": (0, float("inf"))},
        {"capacities": (True, 3)},
        {"capacities": (0, 3)},
        {"capacities": (I(-1, 1), 3)},
        {"conductances": (0,)},
        {"conductances": (-1,)},
        {"conductances": (False,)},
        {"phases": (0,)},
        {"capacities": (2,)},
        {"conductances": ()},
        {"shifts": (False,)},
        {"shifts": (Q(0),)},
        {"shifts": (1,)},
        {"edges": ((1, 0),)},
        {"edges": ((0, 0),)},
        {"edges": ((False, 1),)},
        {"edges": ((0, 1), (0, 1)), "conductances": (2, 2), "shifts": (0, 0)},
        {"epi": (0, 1, 2), "phases": (0, 0, 0), "capacities": (1, 1, 1)},
    ),
)
def test_invalid_primitives_or_branches_are_not_repaired(changes):
    with pytest.raises((TypeError, ValueError)):
        _small(**changes)


def test_single_pass_inputs_are_normalized_without_mutating_the_source():
    arguments = dict(
        epi=[0, 1],
        phases=[0, Q(1, 2)],
        capacities=[2, 3],
        edges=[(0, 1)],
        conductances=[2],
        shifts=[0],
    )
    before = deepcopy(arguments)
    direct = owner.bound_node_history_window(
        **arguments, horizon=Q(1, 100), target_index=1
    )
    streamed = owner.bound_node_history_window(
        **{key: iter(values) for key, values in arguments.items()},
        horizon=Q(1, 100),
        target_index=1,
    )
    assert streamed == direct and arguments == before


def _box(**changes):
    values = dict(
        epi=(I(Q(-1, 8), Q(1, 8)), I(Q(7, 8), Q(9, 8))),
        phases=(0, pi_interval() / 6),
        capacities=(2, 3),
        edges=((0, 1),),
        conductances=(2,),
        shifts=(0,),
        target_index=0,
    )
    values.update(changes)
    return owner.bound_node_history_box(**values)


def test_box_ranges_use_actual_whole_form_box_and_signed_target_rows():
    left, right = _box(), _box(target_index=1)
    pi, root_three = pi_interval(), sqrt(I(3))
    # x_1-x_0 ranges through [3/4,5/4]. With W=2 and nu=(2,3),
    # theta_dot_1-theta_dot_0 = 10*(x_1-x_0)/pi.
    _same_value(left.pressure, 1 / (2 * pi))
    _same_value(right.pressure, -1 / (2 * pi))
    expected_left = I(Q(3, 4), Q(5, 4)) * 10 * root_three / pi**2
    expected_right = I(Q(3, 4), Q(5, 4)) * -15 * root_three / pi**2
    for actual, expected in (
        (left.form_acceleration, expected_left),
        (right.form_acceleration, expected_right),
    ):
        assert abs(actual.lo - expected.lo) < Q(1, 10**30)
        assert abs(actual.hi - expected.hi) < Q(1, 10**30)
    assert left.form == I(Q(-1, 8), Q(1, 8))
    assert right.form == I(Q(7, 8), Q(9, 8))
    assert left.form_acceleration.lo > 0 and right.form_acceleration.hi < 0
    assert left.edge_phase_differences == right.edge_phase_differences
    assert left.edge_phase_velocities == right.edge_phase_velocities


def test_box_does_not_expand_time_or_construct_motion_evidence(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("a conditional box range must not construct source motion bounds")

    monkeypatch.setattr(owner, "_bound_motion", forbidden)
    monkeypatch.setattr(owner, "bound_node_history_window", forbidden)
    report = _box()
    assert report.form == I(Q(-1, 8), Q(1, 8))
    assert not hasattr(report, "horizon")
    assert not hasattr(report, "motion")
    assert not hasattr(report, "trajectory_enclosed")
    assert not hasattr(report, "observed_acceleration")
    with pytest.raises(FrozenInstanceError):
        report.target_index = 1


def test_rectangular_box_retains_uncertainty_from_discarded_correlations():
    report = _box(epi=(I(0, 1), I(0, 1)), phases=(I(-1, 1) / 8,) * 2)
    # Every correlated state x_0=x_1, theta_0=theta_1 is stationary.
    # Its rectangular hull also contains unequal coordinates and cannot
    # report that correlated family's exactly zero pressure/acceleration.
    assert report.form == I(0, 1)
    assert report.pressure.lo < 0 < report.pressure.hi
    assert report.form_acceleration.lo < 0 < report.form_acceleration.hi
    assert report.edge_phase_velocities[0].lo < 0 < report.edge_phase_velocities[0].hi
    assert all(value > 0 for value in report.principal_margins)


def test_wide_phase_box_reports_unresolved_branch_without_rejecting_the_law():
    report = _box(phases=(0, I(-4, 4)))
    assert report.principal_margins[0] < 0
    assert report.pressure.contains(0)
    assert report.form_acceleration.contains(0)
    assert not hasattr(report, "crossing_certified")
    assert not hasattr(report, "retention_certified")


def test_box_preserves_requested_lift_and_reports_wrong_branch_as_unresolved():
    phases = (0, pi_interval() / 6 + 2 * pi_interval())
    admitted = _box(phases=phases, shifts=(1,))
    unresolved = _box(phases=phases, shifts=(0,))
    _same_value(admitted.pressure, unresolved.pressure)
    for actual, expected in zip(
        (admitted.form_acceleration.lo, admitted.form_acceleration.hi),
        (unresolved.form_acceleration.lo, unresolved.form_acceleration.hi),
    ):
        assert abs(actual - expected) < Q(1, 10**30)
    assert admitted.principal_margins[0] > 0
    assert unresolved.principal_margins[0] < 0
    assert unresolved.edge_phase_differences[0].lo > 2 * pi_interval().hi


@pytest.mark.parametrize("target", (True, False, Q(0), 0.0, -1, 2))
def test_box_target_requires_an_existing_exact_index(target):
    with pytest.raises((TypeError, ValueError)):
        _box(target_index=target)


@pytest.mark.parametrize(
    "changes",
    (
        {"epi": (False, 1)},
        {"epi": (float("nan"), 1)},
        {"phases": (0, float("inf"))},
        {"capacities": (True, 3)},
        {"capacities": (0, 3)},
        {"capacities": (I(-1, 1), 3)},
        {"conductances": (0,)},
        {"conductances": (I(-1, 1),)},
        {"conductances": (False,)},
        {"phases": (0,)},
        {"capacities": (2,)},
        {"conductances": ()},
        {"shifts": ()},
        {"shifts": (False,)},
        {"shifts": (Q(0),)},
        {"edges": ((1, 0),)},
        {"edges": ((0, 0),)},
        {"edges": ((False, 1),)},
        {"edges": ((0, 1), (0, 1)), "conductances": (2, 2), "shifts": (0, 0)},
        {"epi": (0, 1, 2), "phases": (0, 0, 0), "capacities": (1, 1, 1)},
    ),
)
def test_box_rejects_invalid_primitives_and_malformed_branch_labels(changes):
    with pytest.raises((TypeError, ValueError)):
        _box(**changes)


def test_box_admits_each_single_pass_input_once_and_preserves_the_source():
    arguments = dict(
        epi=[I(0, 1), I(1, 2)],
        phases=[0, I(Q(1, 4), Q(1, 2))],
        capacities=[I(2, 3), 3],
        edges=[(0, 1)],
        conductances=[I(1, 2)],
        shifts=[0],
    )
    before = deepcopy(arguments)
    direct = owner.bound_node_history_box(**arguments, target_index=1)
    streamed = owner.bound_node_history_box(
        **{key: iter(values) for key, values in arguments.items()}, target_index=1
    )
    assert streamed == direct and arguments == before
