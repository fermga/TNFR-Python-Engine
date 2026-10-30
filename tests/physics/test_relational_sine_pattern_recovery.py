"""Full-support sine recovery and causal intermediary controls.

All scientific controls are static: independent graph matrices, edge equations,
initial derivatives and uncertainty corners. No frozen producer or trajectory
is evaluated, and the eleven-node proof does not extend solver size limits.
"""

import json
import pickle
from dataclasses import replace
from fractions import Fraction as Q

import mpmath as mp
import networkx as nx
import pytest

from tnfr.dynamics.relational import RelationalExchangeModel
from tnfr.mathematics._rational_interval import I, pi_interval
from tnfr.mathematics._validated_taylor import flow_jets
from tnfr.physics.relational_sine_forecast import SineForecast, _sine_flow
from tnfr.physics.relational_sine_pattern import (
    SineRelativeForecast,
    bound_relational_sine_pattern,
)
from tnfr.physics.relational_sine_recovery import certify_sine_pattern_recovery
from tnfr.sdk import export_to_json

MODEL = RelationalExchangeModel(1, phase_domain="regular")
RADIUS = Q(1, 16)
ERROR = Q(1, 65536)


def _graph(*, receiver_sign=1, hidden_capacity=1):
    graph = nx.Graph()
    graph.add_nodes_from(range(11))
    for offset in (0, 5):
        graph.add_edges_from((offset + j, offset + (j + 1) % 5) for j in range(5))
    graph.add_edges_from(((0, 10), (5, 10)))
    for node in graph:
        position = node % 5 if node != 10 else 0
        sign = receiver_sign if 5 <= node < 10 else 1
        graph.nodes[node].update(
            EPI=1 / 4096 if node == 0 else 0,
            theta=float(sign * Q(1287 * position, 1024)),
            nu_f=hidden_capacity if node == 10 else 1,
        )
    graph.graph["GAMMA"] = {"type": "none"}
    return graph


def _target(receiver_sign=1):
    return (
        tuple(Q(j, 5) for j in range(5))
        + tuple(receiver_sign * Q(j, 5) for j in range(5))
        + (Q(0),)
    )


def _pattern(graph=None, *, model=MODEL, error=ERROR):
    graph = _graph() if graph is None else graph
    return bound_relational_sine_pattern(
        graph,
        reference_node=0,
        reference_model=model,
        form_error_bounds=(error,) * len(graph),
        phase_error_bounds=(error,) * len(graph),
    )


def _certificate(pattern=None, *, target=None, **changes):
    arguments = dict(
        target_phase_turns=_target() if target is None else target, radius=RADIUS
    )
    arguments.update(changes)
    return certify_sine_pattern_recovery(
        _pattern() if pattern is None else pattern, **arguments
    )


def _mp(value):
    value = Q(value)
    return mp.mpf(value.numerator) / value.denominator


def _contains(bound, value):
    assert _mp(bound.lo) <= value <= _mp(bound.hi)


def _laplacian(graph, weights=None):
    matrix = mp.matrix(len(graph))
    for i, j in graph.edges:
        weight = mp.mpf(1) if weights is None else weights(i, j)
        matrix[i, i] += weight
        matrix[j, j] += weight
        matrix[i, j] -= weight
        matrix[j, i] -= weight
    return matrix


@pytest.mark.parametrize("receiver_sign", (-1, 1))
def test_uncertain_two_ring_intermediary_pattern_has_full_recovery_admission(
    receiver_sign,
):
    graph = _graph(receiver_sign=receiver_sign)
    target = _target(receiver_sign)
    report = _certificate(_pattern(graph), target=target)
    assert report.admitted
    assert len(report.nodes) == 11 and len(report.edges) == 12
    assert report.nodes[-1] == 10
    assert report.norm_margin > 0 and report.energy_margin > 0
    assert report.hypothesis_failures == report.unresolved_conditions == ()
    assert report.norm_squared_upper_bound <= Q(41805, 536870912) < RADIUS**2
    assert report.excess_storage_upper_bound <= Q(3563, 536870912) < Q(1, 28160)
    assert all(not row for row in report.target_geometry.symbolic_sine_coefficients)
    with mp.workdps(90):
        spectrum = mp.eigsy(_laplacian(graph), eigvals_only=True)
        assert 0 < _mp(report.spectral_gap_lower_bound) <= spectrum[1]
        _contains(report.target_phase_storage_bounds, 10 * (1 - mp.cos(2 * mp.pi / 5)))
        _contains(report.maximum_target_angle_bounds, 2 * mp.pi / 5)
        cosine = mp.cos(2 * mp.pi / 5 + mp.sqrt(2) * _mp(RADIUS))
        assert 0 < _mp(report.cosine_lower_bound) <= cosine


def test_independent_uncertainty_corners_obey_centered_norm_and_actual_energy_bounds():
    graph, pattern, target = _graph(), _pattern(), _target()
    report = _certificate(pattern)
    signs = (
        (1,) * 11,
        (-1,) * 11,
        tuple((-1) ** i for i in range(11)),
        tuple(1 if i < 5 else -1 for i in range(11)),
        tuple(1 if i in (0, 5, 10) else -1 for i in range(11)),
        tuple(1 if i == 10 else -1 for i in range(11)),
    )
    with mp.workdps(90):
        theta_star = tuple(2 * mp.pi * _mp(value) for value in target)
        target_energy = sum(
            1 - mp.cos(theta_star[j] - theta_star[i]) for i, j in graph.edges
        )
        for row in signs:
            form = tuple(
                _mp(value) + 7 + sign * _mp(ERROR)
                for value, sign in zip(pattern.nominal_form, row)
            )
            phase = tuple(
                _mp(value) - 3 - sign * _mp(ERROR)
                for value, sign in zip(pattern.nominal_phase, row)
            )
            mean_form = sum(form) / 11
            deviation = tuple(
                value - target_value for value, target_value in zip(phase, theta_star)
            )
            mean_phase = sum(deviation) / 11
            norm = sum((value - mean_form) ** 2 for value in form) + sum(
                (value - mean_phase) ** 2 for value in deviation
            )
            excess = (
                sum(
                    (form[j] - form[i]) ** 2 / 2 + 1 - mp.cos(phase[j] - phase[i])
                    for i, j in graph.edges
                )
                - target_energy
            )
            assert norm <= _mp(report.norm_squared_upper_bound) < _mp(RADIUS) ** 2
            assert (
                excess
                <= _mp(report.excess_storage_upper_bound)
                < _mp(report.barrier_lower_bound)
            )


def test_heterogeneous_positive_capacities_do_not_import_equal_capacity_phase_lock():
    graph = _graph()
    for node in graph:
        graph.nodes[node]["nu_f"] = 2.0 ** (node % 4 - 2)
    report = _certificate(_pattern(graph))
    assert report.admitted
    assert len(set(report.exact_held_capacity)) > 1


def test_frozen_intermediary_cannot_use_positive_capacity_recovery_theorem():
    pattern = _pattern(_graph(hidden_capacity=0))
    report = _certificate(pattern)
    assert not report.admitted
    assert "strictly_positive_held_capacity_required" in report.hypothesis_failures
    assert pattern.form_rate_bounds[10] == pattern.phase_rate_bounds[10] == I(0)
    assert len(report.nodes) == 11


def test_small_positive_intermediary_remains_held_exact_not_an_arbitrary_cutoff():
    report = _certificate(_pattern(_graph(hidden_capacity=2.0**-200)))
    assert report.admitted
    assert report.exact_held_capacity[10] == Q(1, 2**200)
    assert report.capacity_bounds[10].lo == 0


def test_extra_acute_edge_can_break_exact_criticality():
    graph = _graph()
    graph.add_edge(1, 5)
    with pytest.raises(ValueError):
        _certificate(_pattern(graph))


def test_extra_zero_phase_edge_is_retained_when_criticality_is_preserved():
    graph = _graph()
    graph.add_edge(1, 6)
    report = _certificate(_pattern(graph))
    assert report.admitted
    assert len(report.edges) == 13
    assert (1, 6) in report.edges
    assert all(not row for row in report.target_geometry.symbolic_sine_coefficients)


def test_small_residual_is_not_an_exact_critical_target():
    target = list(_target())
    target[1] += Q(1, 10**20)
    with pytest.raises(ValueError):
        _certificate(target=target)


def test_nonacute_target_is_not_admitted_by_recovery_geometry():
    target = list(_target())
    target[1] = Q(1, 4)
    with pytest.raises(ValueError):
        _certificate(target=target)


def test_disconnected_support_is_not_a_collective_recovery_domain():
    graph = _graph()
    graph.remove_edge(5, 10)
    with pytest.raises(ValueError):
        _pattern(graph)


@pytest.mark.parametrize("graph", (nx.path_graph(33), nx.complete_graph(11)))
def test_exact_target_owner_retains_its_declared_size_budget(graph):
    for node in graph:
        graph.nodes[node].update(EPI=0, theta=0, nu_f=1)
    with pytest.raises(ValueError):
        _certificate(_pattern(graph, error=0), target=(0,) * len(graph))


def test_generic_recovery_still_requires_actual_form_dissipation():
    model = RelationalExchangeModel(
        1, epi_weight=0, phase_weight=1, phase_domain="regular"
    )
    report = _certificate(_pattern(model=model))
    assert not report.admitted
    assert "positive_epi_weight_required" in report.hypothesis_failures


@pytest.mark.parametrize(
    "invalid",
    (
        _target()[:-1],
        _target() + (Q(0),),
        (False,) + _target()[1:],
        (0.0,) + _target()[1:],
        (float("nan"),) + _target()[1:],
    ),
)
def test_target_requires_one_exact_rational_turn_per_retained_node(invalid):
    with pytest.raises((ValueError, TypeError)):
        _certificate(target=invalid)


def test_common_target_origin_and_declared_full_turns_preserve_the_same_pattern():
    pattern = _pattern()
    ordinary = _certificate(pattern)
    shifted = _certificate(
        pattern, target=tuple(value + Q(2, 7) for value in _target())
    )
    target = list(_target())
    target[3] += 1
    explicit = _certificate(pattern, target=target, phase_turns=(0, 0, 0, 1) + (0,) * 7)
    missing = _certificate(pattern, target=target)
    assert shifted.admitted and explicit.admitted
    assert (
        ordinary.norm_squared_upper_bound
        == shifted.norm_squared_upper_bound
        == explicit.norm_squared_upper_bound
    )
    assert not missing.admitted


@pytest.mark.parametrize("hidden_capacity", (0, 1, 2))
def test_causal_receiver_onset_matches_complete_edge_tangent_and_nonlinear_jets(
    hidden_capacity,
):
    graph, target = _graph(hidden_capacity=hidden_capacity), _target()
    n, epsilon = len(graph), Q(1, 4096)
    neighbors = tuple(tuple(graph[i]) for i in graph)
    pi = pi_interval()
    state = (
        tuple(I(epsilon if i == 0 else 0) for i in graph)
        + tuple(2 * value * pi for value in target)
        + (I(hidden_capacity),)
    )

    def flow(values):
        return _sine_flow(
            values, neighbors=neighbors, visible_capacity=(Q(1),) * 10, model=MODEL
        )

    series = flow_jets(state, 2, flow)
    assert series[5][1].contains(0)
    assert series[n + 5][1].contains(0)
    with mp.workdps(90):
        e, w = map(_mp, MODEL.effective_weights)
        a, b = w / mp.pi, w / (_mp(MODEL.storage_scale) * mp.pi)
        phase = tuple(2 * mp.pi * _mp(value) for value in target)
        laplacian = _laplacian(graph)
        hessian = _laplacian(graph, lambda i, j: mp.cos(phase[j] - phase[i]))
        matrix = mp.matrix(2 * n)
        for i in graph:
            mobility = mp.mpf(hidden_capacity if i == 10 else 1) / graph.degree[i]
            for j in graph:
                matrix[i, j] = -e * mobility * laplacian[i, j]
                matrix[i, n + j] = -a * mobility * hessian[i, j]
                matrix[n + i, j] = b * mobility * laplacian[i, j]
        perturbation = mp.matrix(2 * n, 1)
        perturbation[0] = _mp(epsilon)
        first, second = matrix * perturbation, matrix**2 * perturbation
        expected_form = hidden_capacity * _mp(epsilon) * (e**2 - a * b) / 6
        expected_phase = -hidden_capacity * _mp(epsilon) * e * b / 6
        assert first[5] == first[n + 5] == 0
        assert mp.almosteq(second[5], expected_form)
        assert mp.almosteq(second[n + 5], expected_phase)
        _contains(2 * series[5][2], expected_form)
        _contains(2 * series[n + 5][2], expected_phase)
        if hidden_capacity:
            assert series[n + 5][2].hi < 0
        else:
            assert series[10][1] == series[n + 10][1] == I(0)


def test_zero_capacity_intermediary_keeps_receiver_rows_independent_of_donor_state():
    graph = _graph(hidden_capacity=0)
    first = _pattern(graph)
    for node in range(5):
        graph.nodes[node]["EPI"] += node + 1
        graph.nodes[node]["theta"] -= (node + 1) / 8
    second = _pattern(graph)
    assert first.form_rate_bounds[5:10] == second.form_rate_bounds[5:10]
    assert first.phase_rate_bounds[5:10] == second.phase_rate_bounds[5:10]
    assert first.form_rate_bounds[10] == second.form_rate_bounds[10] == I(0)
    assert first.phase_rate_bounds[10] == second.phase_rate_bounds[10] == I(0)


@pytest.mark.parametrize("hidden_capacity", (Q(1, 2), Q(2)))
def test_conserved_weighted_means_determine_the_recovered_common_origins(
    hidden_capacity,
):
    graph = _graph(hidden_capacity=float(hidden_capacity))
    target, n, epsilon = _target(), len(graph), Q(1, 4096)
    capacities = (Q(1),) * 10 + (hidden_capacity,)
    weights = tuple(Q(graph.degree[i]) / capacities[i] for i in graph)
    neighbors = tuple(tuple(graph[i]) for i in graph)
    # Probe the actual full nonlinear rows away from equilibrium. Positive
    # heterogeneous capacities cancel only with d_i/nu_i, not plain means.
    form = tuple(Q((i % 3) - 1, 128) for i in graph)
    phases = tuple(
        2 * turn * pi_interval() + Q(i % 4, 256) for i, turn in enumerate(target)
    )
    state = tuple(map(I, form)) + phases + (I(hidden_capacity),)
    rates = _sine_flow(
        state, neighbors=neighbors, visible_capacity=capacities[:-1], model=MODEL
    )
    assert sum((weight * rates[i] for i, weight in enumerate(weights)), I(0)).contains(
        0
    )
    assert sum(
        (weight * rates[n + i] for i, weight in enumerate(weights)), I(0)
    ).contains(0)
    with mp.workdps(90):
        e, w = map(_mp, MODEL.effective_weights)
        theta = tuple(
            2 * mp.pi * _mp(turn) + _mp(Q(i % 4, 256)) for i, turn in enumerate(target)
        )
        independent_form, independent_phase = [], []
        for i in graph:
            gradient = sum(_mp(form[i] - form[j]) for j in graph[i])
            current = sum(mp.sin(theta[j] - theta[i]) for j in graph[i])
            mobility = _mp(capacities[i]) / graph.degree[i]
            independent_form.append(mobility * (-e * gradient + w * current / mp.pi))
            independent_phase.append(mobility * w * gradient / mp.pi)
        assert abs(
            sum(_mp(weight) * value for weight, value in zip(weights, independent_form))
        ) < mp.mpf("1e-80")
        assert abs(
            sum(
                _mp(weight) * value for weight, value in zip(weights, independent_phase)
            )
        ) < mp.mpf("1e-80")

    # In the distinct exact-target-phase donor preparation, the recovery
    # theorem gives uniform form and the same twist up to common origins.
    # Independent degree/capacity accounting then fixes those origins.
    initial_form = (epsilon,) + (Q(0),) * 10
    conserved = sum(weight * value for weight, value in zip(weights, initial_form))
    limiting_form = conserved / sum(weights)
    assert limiting_form == 3 * epsilon / (22 + 2 / hidden_capacity)
    assert sum(weight * limiting_form for weight in weights) == conserved
    assert Q(10, 11) * epsilon**2 < RADIUS**2
    assert Q(3, 2) * epsilon**2 < Q(1, 28160)
    # Initial and target lifted phases coincide. Their conserved weighted
    # difference is zero, forcing the recovered common phase displacement 0.
    assert sum(weights) > 0
    recovered = (
        (I(limiting_form),) * n
        + tuple(2 * turn * pi_interval() for turn in target)
        + (I(hidden_capacity),)
    )
    equilibrium_rates = _sine_flow(
        recovered, neighbors=neighbors, visible_capacity=capacities[:-1], model=MODEL
    )
    assert all(value.contains(0) for value in equilibrium_rates)


def test_forecast_admission_keeps_actual_endpoint_time_and_box_provenance():
    # Three nodes remain within the existing solver's supported layout;
    # this intentionally unavailable initial report runs no numerical steps.
    graph = nx.path_graph(3)
    for i in graph:
        graph.nodes[i].update(EPI=0, theta=0, nu_f=1)
    pattern = _pattern(graph, error=0)
    initial = pattern.relative_form_bounds + pattern.relative_phase_bounds + (I(1),)
    full = SineForecast(
        model=MODEL,
        neighbors=pattern.neighbors,
        visible_capacity=(Q(1), Q(1)),
        initial_box=initial,
        observation_time=Q(2),
        end_time=Q(3),
        time_step=Q(1, 8),
        order=6,
        steps=(),
        validated_end_time=Q(2),
        endpoint=initial,
        failed_tube=None,
        status="unavailable",
        reasons=("synthetic_initial_endpoint_only",),
    )
    forecast = SineRelativeForecast(
        pattern=pattern,
        full_forecast=full,
        relative_form_bounds=pattern.relative_form_bounds,
        relative_phase_bounds=pattern.relative_phase_bounds,
        reference_form_displacement_bounds=I(0),
        reference_phase_displacement_bounds=I(0),
    )
    report = _certificate(forecast, target=(0, 0, 0))
    assert report.admitted
    assert report.input_forecast_admitted is False
    assert report.observation_time == 2
    assert report.input_forecast_requested_end_time == 3
    assert "endpoint" in report.uncertainty_scope
    changed = replace(full, endpoint=(I(-1, 1),) + full.endpoint[1:])
    uncertain = _certificate(replace(forecast, full_forecast=changed), target=(0, 0, 0))
    assert not uncertain.admitted
    assert (
        uncertain.form_norm_squared_upper_bound > report.form_norm_squared_upper_bound
    )


def test_exact_export_and_detached_source_preserve_full_support(tmp_path):
    graph = _graph()
    pattern = _pattern(graph)
    before = pickle.dumps(
        (graph.graph, tuple(graph.nodes(data=True)), tuple(graph.edges(data=True))),
        protocol=5,
    )
    report = pattern.certify_pattern_recovery(
        target_phase_turns=_target(), radius=RADIUS
    )
    payload = report.to_dict()
    assert payload["schema"] == "tnfr.relational-sine-pattern-recovery.v1"
    assert report.source is pattern
    path = tmp_path / "full-pattern-recovery.json"
    export_to_json(payload, path)
    assert json.loads(path.read_text(encoding="utf-8")) == payload
    assert (
        pickle.dumps(
            (graph.graph, tuple(graph.nodes(data=True)), tuple(graph.edges(data=True))),
            protocol=5,
        )
        == before
    )
