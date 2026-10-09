"""Independent geometry, proof constants and finite full-law transit controls."""

from fractions import Fraction as Q
from inspect import signature
from typing import get_type_hints

import mpmath
import networkx as nx
import numpy as np
import pytest
from scipy.integrate import solve_ivp

from tnfr.physics import relational_sine_two_port_transit as owner
from tnfr.sdk import export_to_json, relational_report_to_dict
from tnfr.utils.io import json_loads


@pytest.fixture(scope="module")
def report():
    return owner.assess_sine_two_port_transit(
        form_error_radius=Q(1, 65536), phase_error_radius=Q(1, 65536)
    )


@pytest.fixture(scope="module")
def mp():
    context = mpmath.mp.clone()
    context.dps = 85
    return context


def _mp(mp, value):
    value = Q(value)
    return mp.mpf(value.numerator) / value.denominator


def _graph():
    graph = nx.disjoint_union(nx.cycle_graph(9), nx.cycle_graph(9))
    graph.add_edges_from(((0, 9), (1, 10)))
    return graph


def test_full_support_source_mean_and_cycle_branches_are_independently_rebuilt(report):
    graph = _graph()
    assert report.geometry.edges == tuple(sorted(tuple(sorted(e)) for e in graph.edges))
    assert report.degrees == tuple(graph.degree[i] for i in graph)
    assert report.invariant_weights == report.degrees
    assert report.weighted_coordinate_mass == 40
    raw = tuple(Q(2 * j, 9) for j in range(9)) + tuple(
        Q(1, 18) + Q(j, 9) for j in range(9)
    )
    mean = sum(graph.degree[i] * raw[i] for i in graph) / sum(
        dict(graph.degree).values()
    )
    assert report.nominal_uncentered_phase_mean_turns == mean == Q(229, 360)
    assert report.nominal_phase_turns == tuple(value - mean for value in raw)
    assert sum(graph.degree[i] * report.nominal_phase_turns[i] for i in graph) == 0
    for (i, j), offset, turn in zip(
        report.geometry.edges, report.edge_integer_offsets, report.nominal_edge_turns
    ):
        assert (
            report.nominal_phase_turns[j] - report.nominal_phase_turns[i] - offset
            == turn
        )
        assert abs(turn) < Q(1, 4)
    turns = dict(zip(report.geometry.edges, report.nominal_edge_turns))
    periods = tuple(
        sum(
            turns[tuple(sorted((i, j)))] * (1 if i < j else -1)
            for i, j in zip(cycle, cycle[1:] + cycle[:1])
        )
        for cycle in report.named_cycles
    )
    assert report.named_cycle_periods == periods == (2, 1, 0)
    assert report.initial_form_mean_bounds == (-Q(1, 65536), Q(1, 65536))
    assert report.initial_phase_mean_bounds == (-Q(1, 65536), Q(1, 65536))


def test_normalized_diffusion_bounds_use_true_degrees_and_independent_spectrum(
    report, mp
):
    graph = _graph()
    laplacian = tuple(
        tuple(
            Q(graph.degree[i] if i == j else -int(graph.has_edge(i, j))) for j in graph
        )
        for i in graph
    )
    assert report.laplacian == laplacian
    for i in graph:
        for j in graph:
            centered_metric = Q(graph.degree[i] * int(i == j)) - Q(
                graph.degree[i] * graph.degree[j], 40
            )
            assert (
                report.normalized_gap_slack_matrix[i][j]
                == laplacian[i][j] - report.normalized_gap_lower_bound * centered_metric
            )
            assert (
                report.normalized_upper_slack_matrix[i][j]
                == 2 * graph.degree[i] * int(i == j) - laplacian[i][j]
            )
    normalized = mp.matrix(
        [
            [
                _mp(mp, laplacian[i][j]) / mp.sqrt(graph.degree[i] * graph.degree[j])
                for j in graph
            ]
            for i in graph
        ]
    )
    spectrum = mp.eigsy(normalized, eigvals_only=True)
    assert abs(spectrum[0]) < mp.mpf("1e-78")
    assert spectrum[1] > _mp(mp, report.normalized_gap_lower_bound)
    assert spectrum[len(graph) - 1] < _mp(mp, report.normalized_rate_upper_bound)


def test_independent_initial_full_rows_preserve_fast_transient_and_reference_directions(
    report, mp
):
    graph = _graph()
    phases = tuple(2 * mp.pi * _mp(mp, value) for value in report.nominal_phase_turns)
    sine = tuple(sum(mp.sin(phases[j] - phases[i]) for j in graph[i]) for i in graph)
    gamma = 1 / (1023 * mp.pi)
    reference_rate = tuple(value / graph.degree[i] for i, value in enumerate(sine))
    forms = tuple(_mp(mp, value) for value in report.nominal_epi)
    gradient = tuple(
        sum(forms[i] - forms[j] for j in graph[i]) / graph.degree[i] for i in graph
    )
    form_rate = tuple(-value + gamma * f for value, f in zip(gradient, reference_rate))
    phase_rate = tuple(gamma * value for value in gradient)
    assert all(value == 0 for value in phase_rate)
    assert max(abs(value) for value in form_rate) > 0
    delta = mp.pi / 9
    assert abs(reference_rate[1] - reference_rate[0] + 2 * mp.sin(delta) / 3) < mp.mpf(
        "1e-78"
    )
    assert abs(reference_rate[10] - reference_rate[9] - 2 * mp.sin(delta) / 3) < mp.mpf(
        "1e-78"
    )
    # Full-law initial phase velocity is zero; its second derivative follows
    # gamma*K*L*x', not the nonzero reference gradient velocity.
    phase_acceleration = tuple(
        gamma * sum(form_rate[i] - form_rate[j] for j in graph[i]) / graph.degree[i]
        for i in graph
    )
    assert abs(
        phase_acceleration[1]
        - phase_acceleration[0]
        + gamma**2 * 10 * mp.sin(delta) / 9
    ) < mp.mpf("1e-78")
    assert abs(
        phase_acceleration[10]
        - phase_acceleration[9]
        - gamma**2 * 10 * mp.sin(delta) / 9
    ) < mp.mpf("1e-78")
    forcing_norm = mp.sqrt(
        sum(graph.degree[i] * value**2 for i, value in enumerate(reference_rate))
    )
    assert forcing_norm < _mp(mp, report.reference_forcing_norm_upper_bound)
    assert mp.sin(delta) > _mp(mp, Q(53, 162))
    assert _mp(mp, report.gamma_bounds[0]) < gamma < _mp(mp, report.gamma_bounds[1])
    assert gamma**2 < _mp(mp, report.feedback_strength_upper_bound)
    assert mp.pi / 18 - _mp(
        mp, report.reference_forcing_norm_upper_bound * report.slow_time
    ) > _mp(mp, report.reference_acute_margin_lower_bound)
    assert mp.sqrt(mp.mpf(2) / 3) < _mp(mp, report.port_dual_norm_upper_bound)
    assert report.scaled_time_pi_squared_coefficient == Q(1023**2, 4)
    assert (
        report.original_time_pi_squared_coefficient * Q(1023, 1024)
        == report.scaled_time_pi_squared_coefficient
    )


def test_energy_comparison_and_integral_constants_retain_each_budget(report, mp):
    rx, rt = report.form_error_radius, report.phase_error_radius
    graph = _graph()
    nominal_contact_energy = 2 * (1 - mp.cos(mp.pi / 9))
    assert nominal_contact_energy < _mp(mp, report.nominal_contact_storage_upper_bound)
    assert (
        report.initial_excess_storage_upper_bound
        == report.nominal_contact_storage_upper_bound
        + sum(2 * rx**2 + 2 * rt for _ in graph.edges)
    )
    assert report.energy_budget_margin == Q(40566379, 43486543872)
    eta, gap = report.feedback_strength_upper_bound, report.normalized_gap_lower_bound
    integral_term = 2 * eta * Q(3, 16) / gap
    assert report.integral_comparison_error_upper_bound == integral_term
    assert report.joint_error_candidate == 7 * rt + 7 * rx / 3069 + integral_term
    assert report.scaled_form_norm_candidate == (
        report.initial_scaled_form_norm_upper_bound
        + (eta / gap)
        * (report.reference_forcing_norm_upper_bound + 2 * report.joint_error_candidate)
    ) / (1 - 2 * eta / gap)
    assert report.phase_error_upper_bound == Q(72153997, 628518629376) < Q(1, 8192)
    assert (
        report.short_arc_change_lower_bound
        == Q(4439240884187, 135760023945216)
        > Q(1, 32)
    )
    assert report.direction_margin == Q(196740135899, 135760023945216)
    assert (
        report.error_certified
        and report.whole_window_acute_certified
        and report.direction_certified
    )
    assert report.status == "certified_directional_transit"
    assert report.unavailable_reasons == report.direction_limitations == ()


@pytest.mark.parametrize("phase_radius", (Q(1, 25920), Q(1, 100)))
def test_failed_or_equal_energy_budget_cannot_promote_conditional_bounds(phase_radius):
    result = owner.assess_sine_two_port_transit(
        form_error_radius=0, phase_error_radius=phase_radius
    )
    assert result.energy_budget_margin <= 0
    assert not result.energy_budget_admitted
    assert (
        not result.error_certified
        and not result.whole_window_acute_certified
        and not result.direction_certified
    )
    assert result.status == "unavailable"
    assert (
        "strict_initial_excess_storage_budget_not_certified"
        in result.unavailable_reasons
    )
    for name in (
        "joint_error_upper_bound",
        "scaled_form_norm_upper_bound",
        "phase_error_upper_bound",
        "relative_form_norm_upper_bound",
        "whole_window_acute_margin_lower_bound",
        "short_arc_change_lower_bound",
        "direction_margin",
    ):
        assert getattr(result, name) is None
    assert result.phase_error_candidate > 0


def test_large_form_errors_preserve_nonnegative_energy_and_abstain():
    result = owner.assess_sine_two_port_transit(
        form_error_radius=Q(10**400), phase_error_radius=0
    )
    assert result.initial_excess_storage_upper_bound > 0
    assert result.form_error_radius == Q(10**400)
    assert result.status == "unavailable" and result.phase_error_upper_bound is None


@pytest.mark.parametrize("radius", (Q(0), Q(1, 10**400), -0.0, np.int64(0), 1e-6))
def test_scalar_admission_preserves_exact_and_represented_values(radius):
    result = owner.assess_sine_two_port_transit(
        form_error_radius=radius, phase_error_radius=radius
    )
    expected = Q.from_float(radius) if type(radius) is float else Q(radius)
    assert result.form_error_radius == result.phase_error_radius == expected
    assert type(result.form_error_radius) is Q
    assert result.direction_certified


class _UnderflowingReal(float):
    def __float__(self):
        return 0.0


@pytest.mark.parametrize("field", ("form_error_radius", "phase_error_radius"))
@pytest.mark.parametrize(
    "value",
    (
        True,
        np.bool_(False),
        float("nan"),
        float("inf"),
        -1,
        -Q(1, 10**400),
        "0",
        None,
        1j,
        _UnderflowingReal(1),
    ),
)
def test_invalid_domains_reject_before_constructing_geometry(field, value, monkeypatch):
    monkeypatch.setattr(
        owner, "_derive", lambda *args: pytest.fail("invalid source reached geometry")
    )
    values = dict(form_error_radius=0, phase_error_radius=0)
    values[field] = value
    with pytest.raises((TypeError, ValueError), match=field):
        owner.assess_sine_two_port_transit(**values)


def test_detached_assessor_and_sdk_need_no_target_root_or_trajectory(
    monkeypatch, tmp_path
):
    from tnfr.physics import relational_sine_two_port_compatibility as compatibility

    def forbidden(*args, **kwargs):
        pytest.fail("an analytic transit certificate cannot run a target producer")

    for name in (
        "assess_sine_two_port_compatibility",
        "_root_enclosures",
        "sin",
        "cos",
        "pi_interval",
    ):
        monkeypatch.setattr(compatibility, name, forbidden)
    result = owner.assess_sine_two_port_transit(
        form_error_radius=Q(1, 65536), phase_error_radius=Q(1, 65536)
    )
    direct = result.to_dict()
    generic = relational_report_to_dict(result)
    assert direct["schema"] == "tnfr.sine-two-port-transit.v1"
    assert generic["report"] == direct["report"]
    assert generic["report_type"] == "SineTwoPortTransit"
    path = tmp_path / "transit.json"
    export_to_json(result, path)
    assert json_loads(path.read_text(encoding="utf-8")) == direct
    assert get_type_hints(owner.SineTwoPortTransit)
    parameters = signature(owner.assess_sine_two_port_transit).parameters
    assert tuple(parameters) == ("form_error_radius", "phase_error_radius")
    assert all(value.default is value.empty for value in parameters.values())


@pytest.fixture(scope="module")
def finite_reference_crosscheck(report):
    # Fixed independent binary64 cross-check of an already proved analytic
    # reference bound, not validated trajectory evidence or a parameter scan.
    graph = _graph()
    phases = 2 * np.pi * np.array(tuple(map(float, report.nominal_phase_turns)))

    def rate(_time, theta):
        return np.array(
            [
                sum(np.sin(theta[j] - theta[i]) for j in graph[i]) / graph.degree[i]
                for i in graph
            ]
        )

    result = solve_ivp(rate, (0, 0.25), phases, method="DOP853", rtol=1e-10, atol=1e-12)
    assert result.success and result.t[-1] == 0.25
    return result.y[:, -1]


@pytest.mark.parametrize("perturbed", (False, True))
def test_fixed_stiff_full_law_crosscheck_of_actual_finite_changes(
    report, finite_reference_crosscheck, perturbed
):
    # Full 36-coordinate original form/phase rows, in scaled tau. The two
    # preparations, horizon, method and tolerances are fixed before execution.
    graph = _graph()
    degrees = np.array(report.degrees, dtype=float)
    laplacian = np.array(report.laplacian, dtype=float)
    a = laplacian / degrees[:, None]
    gamma = 1 / (1023 * np.pi)
    tau_end = 1023**2 * np.pi**2 / 4
    phases = 2 * np.pi * np.array(tuple(map(float, report.nominal_phase_turns)))
    forms = np.zeros(18)
    if perturbed:
        forms += float(report.form_error_radius) * np.array(
            [1 if i % 3 else -1 for i in graph]
        )
        phases += float(report.phase_error_radius) * np.array(
            [1 if i % 4 else -1 for i in graph]
        )
    initial = np.concatenate((forms, phases))

    def field(_time, state):
        x, theta = state[:18], state[18:]
        f = np.array(
            [
                sum(np.sin(theta[j] - theta[i]) for j in graph[i]) / graph.degree[i]
                for i in graph
            ]
        )
        return np.concatenate((-a @ x + gamma * f, gamma * (a @ x)))

    def jacobian(_time, state):
        hessian = np.zeros((18, 18))
        for i, j in graph.edges:
            value = np.cos(state[18 + j] - state[18 + i])
            hessian[i, i] += value
            hessian[j, j] += value
            hessian[i, j] -= value
            hessian[j, i] -= value
        return np.block(
            [[-a, -gamma * hessian / degrees[:, None]], [gamma * a, np.zeros((18, 18))]]
        )

    times = np.linspace(0, tau_end, 9)
    solved = solve_ivp(
        field,
        (0, tau_end),
        initial,
        jac=jacobian,
        method="Radau",
        rtol=1e-10,
        atol=1e-12,
        t_eval=times,
    )
    assert solved.success and solved.t[-1] == tau_end
    final = solved.y[:, -1]
    initial_arcs = (phases[1] - phases[0], phases[10] - phases[9])
    final_arcs = (final[19] - final[18], final[28] - final[27])
    assert initial_arcs[0] - final_arcs[0] >= float(report.short_arc_change_lower_bound)
    assert final_arcs[1] - initial_arcs[1] >= float(report.short_arc_change_lower_bound)
    means = np.array([np.dot(degrees, forms), np.dot(degrees, phases)]) / 40
    nominal_mean = (
        np.dot(
            degrees, 2 * np.pi * np.array(tuple(map(float, report.nominal_phase_turns)))
        )
        / 40
    )
    member_reference = finite_reference_crosscheck + means[1] - nominal_mean
    phase_error = final[18:] - member_reference
    assert np.sqrt(np.dot(degrees, phase_error**2)) <= float(
        report.phase_error_upper_bound
    )
    energies = []
    for state in solved.y.T:
        assert abs(np.dot(degrees, state[:18]) / 40 - means[0]) < 1e-11
        assert abs(np.dot(degrees, state[18:]) / 40 - means[1]) < 1e-11
        energy = sum(
            (state[j] - state[i]) ** 2 / 2 + 1 - np.cos(state[18 + j] - state[18 + i])
            for i, j in graph.edges
        )
        energies.append(energy)
    assert np.all(np.diff(energies) <= 1e-10)
