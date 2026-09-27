"""Static local composition checks, without a nonlinear trajectory campaign.

Rational coefficient probes exercise exact algebra only for the supplied
family. The ideal trigonometric rank proof is analytic; finite differences
below independently compare its Jacobian formula with the native field.
"""

import math
from dataclasses import FrozenInstanceError
from fractions import Fraction as Q

import networkx as nx
import numpy as np
import pytest

from benchmarks.relational_local_composition import (
    analyze_local_composition,
    analyze_state_rate_obstruction,
)
from tnfr.dynamics.relational import (
    RelationalExchangeModel,
    evaluate_relational_exchange,
)
from tnfr.sdk import Network


def _mv(matrix, vector):
    return tuple(
        sum((a * b for a, b in zip(row, vector, strict=True)), Q(0)) for row in matrix
    )


def _mm(left, right):
    columns = tuple(zip(*right, strict=True))
    return tuple(_mv(columns, row) for row in left)


def _transpose(matrix):
    return tuple(tuple(column) for column in zip(*matrix, strict=True))


def _identity(size):
    return tuple(tuple(Q(i == j) for j in range(size)) for i in range(size))


def _zero(rows, columns):
    return tuple((Q(0),) * columns for _ in range(rows))


@pytest.fixture(scope="module")
def probe():
    return analyze_local_composition(ring_cosine=Q(1, 3), inverse_pi=Q(2, 7))


@pytest.mark.parametrize(
    "parameters",
    (
        dict(ring_cosine=Q(1, 3), inverse_pi=Q(2, 7)),
        dict(
            ring_cosine=Q(2, 5),
            inverse_pi=Q(1, 4),
            epi_weight=Q(3, 5),
            phase_weight=Q(2, 5),
            storage_scale=Q(7, 4),
        ),
        dict(
            ring_cosine=Q(3, 8),
            inverse_pi=Q(4, 11),
            epi_weight=0,
            phase_weight=Q(3, 2),
            storage_scale=2,
        ),
    ),
)
def test_rational_family_observable_rank_stabilizes_at_ten(parameters):
    report = analyze_local_composition(**parameters)
    result = report.realization
    assert result.output_count == result.output_rank == 6
    assert result.dimension == 10
    assert result.full_state_dimension == 20
    assert result.extra_coordinates == 4
    assert result.rank_progression == (6, 10, 10)
    assert not report.ideal_trigonometric_coefficients
    assert all(isinstance(value, Q) for row in report.generator for value in row)
    assert _mm(result.observation, report.generator) == _mm(
        result.reduced_generator, result.observation
    )
    assert _mm(result.output_map, result.observation) == report.output_rows


def test_natural_coordinates_close_the_full_linear_observation(probe):
    assert _mm(probe.natural_rows, probe.natural_lift) == _identity(10)
    assert _mm(probe.natural_rows, probe.generator) == _mm(
        probe.natural_generator, probe.natural_rows
    )
    assert _mm(probe.output_map, probe.natural_rows) == probe.output_rows
    assert _mm(probe.output_rows, probe.natural_lift) == probe.output_map
    assert probe.exact_identity_checks
    with pytest.raises(FrozenInstanceError):
        probe.generator = ()


def test_independent_hidden_form_witness_changes_requested_rates(probe):
    eta = tuple(map(Q, (0, 1, -1, -1, 1) + (0,) * 15))
    assert _mv(probe.output_rows, eta) == (Q(0),) * 6
    rates = _mv(probe.output_rows, _mv(probe.generator, eta))
    assert rates[:3] == (Q(-1, 30), Q(11, 30), Q(0))
    c, inverse_pi = Q(1, 3), Q(2, 7)
    port_metric_inverse = inverse_pi / (1 + 2 * c)
    interior_metric_inverse = inverse_pi / (2 * c)
    mean = 2 * (interior_metric_inverse - port_metric_inverse) / 5
    port = -(8 * port_metric_inverse + 2 * interior_metric_inverse) / 5
    assert port < 0
    assert rates[3:] == (mean / 2, port / 2, Q(0))
    assert probe.hidden_output_rates == _mm(
        _mm(probe.output_rows, probe.generator), _transpose(probe.hidden_directions)
    )


def test_offsets_and_reflection_odd_modes_remain_linearly_unobserved(probe):
    assert len(probe.offset_directions) == 2
    assert len(probe.odd_directions) == 8
    offsets = _transpose(probe.offset_directions)
    assert _mm(probe.generator, offsets) == _zero(20, 2)
    assert _mm(probe.natural_rows, offsets) == _zero(10, 2)
    odd = _transpose(probe.odd_directions)
    assert _mm(probe.natural_rows, odd) == _zero(10, 8)
    assert _mm(_mm(probe.natural_rows, probe.generator), odd) == _zero(10, 8)
    # Independent odd form perturbation: left neighbors of the port are opposite.
    eta = tuple(map(Q, (0, 1, 0, 0, -1) + (0,) * 15))
    assert _mv(probe.natural_rows, eta) == (Q(0),) * 10
    assert _mv(probe.natural_rows, _mv(probe.generator, eta)) == (Q(0),) * 10


@pytest.mark.parametrize(
    "name,value",
    (
        ("ring_cosine", 0),
        ("ring_cosine", Q(-1, 3)),
        ("ring_cosine", 1 / 3),
        ("ring_cosine", True),
        ("inverse_pi", 0),
        ("inverse_pi", float("nan")),
        ("epi_weight", Q(-1, 3)),
        ("epi_weight", False),
        ("phase_weight", 0),
        ("phase_weight", 0.5),
        ("storage_scale", 0),
        ("storage_scale", "1"),
    ),
)
def test_coefficient_admission_rejects_unstated_rationalization(name, value):
    arguments = dict(ring_cosine=Q(1, 3), inverse_pi=Q(2, 7))
    arguments[name] = value
    with pytest.raises((TypeError, ValueError)):
        analyze_local_composition(**arguments)


def _equilibrium_graph():
    graph = nx.disjoint_union(nx.cycle_graph(5), nx.cycle_graph(5))
    graph.add_edge(0, 5)
    graph.graph.update(GAMMA={"type": "none"}, vectorized_dnfr=True)
    kappa = math.tau / 5
    for node in graph:
        graph.nodes[node].update(EPI=0.0, theta=(node % 5) * kappa, nu_f=1.0)
    return graph


def _rates(graph):
    field = evaluate_relational_exchange(graph, model=RelationalExchangeModel(1.0))
    return np.array(field.form_rate + field.phase_rate)


def _analytic_float_jacobian(graph):
    """Independent equilibrium derivative; no rationalized theorem assertion."""
    adjacency = nx.to_numpy_array(graph, nodelist=range(10))
    degrees = adjacency.sum(axis=1)
    laplacian = np.diag(degrees) - adjacency
    c = math.cos(math.tau / 5)
    edge_cosines = c * adjacency
    edge_cosines[0, 5] = edge_cosines[5, 0] = 1.0
    hessian = np.diag(edge_cosines.sum(axis=1)) - edge_cosines
    inverse_metric = np.diag(1 / (math.pi * edge_cosines.sum(axis=1)))
    return np.block(
        [
            [-0.5 * np.diag(1 / degrees) @ laplacian, -0.5 * inverse_metric @ hessian],
            [0.5 * inverse_metric @ laplacian, np.zeros((10, 10))],
        ]
    )


def test_materialized_coefficient_probe_matches_the_independent_equilibrium_matrix():
    # Fractions preserve these binary64 approximations, not mathematical pi/cosine.
    report = analyze_local_composition(
        ring_cosine=Q(math.cos(math.tau / 5)), inverse_pi=Q(1 / math.pi)
    )
    assert not report.ideal_trigonometric_coefficients
    expected = _analytic_float_jacobian(_equilibrium_graph())
    actual = np.array(report.generator, dtype=float)
    np.testing.assert_allclose(actual, expected, rtol=3e-15, atol=1e-16)


@pytest.mark.parametrize("sector", ("form", "phase", "mixed"))
def test_joint_jacobian_matches_native_static_central_differences(sector):
    graph = _equilibrium_graph()
    direction = np.array(
        [1, -2, 1, 0, -1, 2, 0, -1, 1, -1, 0, 1, -1, 2, -1, 0, 2, -2, 1, -1],
        dtype=float,
    )
    if sector == "form":
        direction[10:] = 0
    elif sector == "phase":
        direction[:10] = 0
    expected = _analytic_float_jacobian(graph) @ direction
    step = 2.0**-15
    plus, minus = graph.copy(), graph.copy()
    for node in graph:
        plus.nodes[node]["EPI"] += step * direction[node]
        minus.nodes[node]["EPI"] -= step * direction[node]
        plus.nodes[node]["theta"] += step * direction[10 + node]
        minus.nodes[node]["theta"] -= step * direction[10 + node]
    observed = (_rates(plus) - _rates(minus)) / (2 * step)
    np.testing.assert_allclose(observed, expected, rtol=2e-7, atol=3e-9)
    assert all(graph.nodes[node]["EPI"] == 0 for node in graph)


def test_hidden_odd_phase_changes_nonlinear_port_rate_under_even_form(probe):
    graph = _equilibrium_graph()
    epsilon, phase_shift = 1 / 32, 1 / 64
    for node, value in enumerate((0, epsilon, -epsilon, -epsilon, epsilon)):
        graph.nodes[node]["EPI"] = value
    changed = graph.copy()
    changed.nodes[1]["theta"] += phase_shift
    changed.nodes[4]["theta"] -= phase_shift
    state = tuple(
        Q(graph.nodes[node][key]) for key in ("EPI", "theta") for node in graph
    )
    other = tuple(
        Q(changed.nodes[node][key]) for key in ("EPI", "theta") for node in changed
    )
    assert _mv(probe.natural_rows, state) == _mv(probe.natural_rows, other)
    rates, changed_rates = _rates(graph), _rates(changed)
    # O_phase_mean_difference + O_phase_left_port = theta_0 - mean(theta_right).
    difference = (changed_rates[10] - np.mean(changed_rates[15:])) - (
        rates[10] - np.mean(rates[15:])
    )
    kappa = math.tau / 5
    expected = (
        -epsilon
        / math.pi
        * (1 / (1 + 2 * math.cos(kappa + phase_shift)) - 1 / (1 + 2 * math.cos(kappa)))
    )
    assert expected < 0
    assert difference == pytest.approx(expected, rel=1e-11, abs=2e-15)
    assert difference != 0


@pytest.fixture(scope="module")
def state_rate_obstruction():
    return analyze_state_rate_obstruction(
        ring_cosine=Q(3, 5), ring_sine=Q(4, 5), inverse_pi=Q(2, 7)
    )


def test_rational_state_rate_fiber_has_distinct_coarse_acceleration(
    state_rate_obstruction,
):
    report = state_rate_obstruction
    assert report.projected_state_minus == report.projected_state_plus
    assert report.projected_rate_minus == report.projected_rate_plus
    assert report.rate_minus != report.rate_plus
    assert report.projected_acceleration_minus != report.projected_acceleration_plus
    assert not report.composition.ideal_trigonometric_coefficients
    assert report.obstruction_established
    c, s, rho = Q(3, 5), Q(4, 5), Q(2, 7)
    epsilon, delta, k = Q(1, 32), Q(1, 64), Q(1, 2)
    expected = -8 * k**2 * epsilon * delta * s * rho**2 / (c * (1 + 2 * c) ** 2)
    assert report.scalar_gap_actual == report.scalar_gap_expected == expected
    assert expected < 0
    rows = report.composition.natural_rows
    assert report.projected_state_plus == _mv(rows, report.state_plus)
    assert report.projected_rate_plus == _mv(rows, report.rate_plus)
    assert report.projected_acceleration_plus == _mv(rows, report.acceleration_plus)
    assert report.exact_identity_checks
    with pytest.raises(FrozenInstanceError):
        report.scalar_gap_actual = Q(0)


def test_even_odd_preparations_preserve_exact_form_storage_and_loss(
    state_rate_obstruction,
):
    report = state_rate_obstruction
    epsilon, delta = Q(1, 32), Q(1, 64)
    expected_storage = 5 * epsilon**2 + 2 * delta**2
    expected_loss = (Q(43, 3) * epsilon**2 + 5 * delta**2) / 2
    assert report.form_storage_minus == report.form_storage_plus == expected_storage
    assert report.loss_minus == report.loss_plus == expected_loss


@pytest.mark.parametrize("zero_amplitude", ("even_amplitude", "odd_amplitude"))
def test_missing_cross_amplitude_does_not_claim_a_state_rate_obstruction(
    zero_amplitude,
):
    report = analyze_state_rate_obstruction(
        ring_cosine=Q(3, 5),
        ring_sine=Q(4, 5),
        inverse_pi=Q(2, 7),
        **{zero_amplitude: 0},
    )
    assert report.projected_state_minus == report.projected_state_plus
    assert report.projected_rate_minus == report.projected_rate_plus
    assert report.projected_acceleration_minus == report.projected_acceleration_plus
    assert report.scalar_gap_expected == report.scalar_gap_actual == 0
    assert not report.obstruction_established


@pytest.mark.parametrize(
    "name,value",
    (
        ("ring_sine", 0),
        ("ring_sine", 0.5),
        ("even_amplitude", -1),
        ("even_amplitude", True),
        ("odd_amplitude", 1 / 64),
    ),
)
def test_state_rate_witness_rejects_implicit_parameter_coercion(name, value):
    arguments = dict(ring_cosine=Q(3, 5), ring_sine=Q(4, 5), inverse_pi=Q(2, 7))
    arguments[name] = value
    with pytest.raises((TypeError, ValueError)):
        analyze_state_rate_obstruction(**arguments)


def _state_rate_pair(*, even_amplitude=1 / 32):
    base = _equilibrium_graph()
    eta, zeta = (0, 1, -1, -1, 1), (0, 1, 0, 0, -1)
    result = []
    for sign in (-1, 1):
        graph = base.copy()
        for node in range(5):
            graph.nodes[node]["EPI"] = (
                even_amplitude * eta[node] + sign * zeta[node] / 64
            )
        report = Network(graph).relational_pattern(
            RelationalExchangeModel(1),
            reference_phase={node: graph.nodes[node]["theta"] for node in graph},
            regions=(tuple(range(5)), tuple(range(5, 10))),
        )
        result.append((graph, report))
    return tuple(result)


def _static_directional_acceleration(graph, field):
    """Central field probes along its tangent, without advancing an integrator."""
    step = 2.0**-10
    plus, minus = graph.copy(), graph.copy()
    for node in graph:
        plus.nodes[node]["EPI"] += step * field.form_rate[node]
        minus.nodes[node]["EPI"] -= step * field.form_rate[node]
        plus.nodes[node]["theta"] += step * field.phase_rate[node]
        minus.nodes[node]["theta"] -= step * field.phase_rate[node]
    return (_rates(plus) - _rates(minus)) / (2 * step)


def test_native_preparations_match_coarse_state_rate_storage_and_regional_budgets(
    probe,
):
    (_, minus_report), (_, plus_report) = _state_rate_pair()
    minus, plus = minus_report.field, plus_report.field
    # The complete existing pattern report retains what the proposed C10 discards.
    assert minus_report.regions[0].centered_form != plus_report.regions[0].centered_form
    assert (
        plus_report.regions[0].centered_form[1]
        - minus_report.regions[0].centered_form[1]
    ) == Q(1, 32)
    state_minus, state_plus = (
        tuple(map(Q, field.epi + field.phase)) for field in (minus, plus)
    )
    assert _mv(probe.natural_rows, state_minus) == _mv(probe.natural_rows, state_plus)
    rates_minus, rates_plus = (
        field.form_rate + field.phase_rate for field in (minus, plus)
    )
    assert rates_minus != rates_plus
    np.testing.assert_allclose(
        _mv(probe.natural_rows, rates_minus),
        _mv(probe.natural_rows, rates_plus),
        rtol=0,
        atol=2e-15,
    )
    epsilon, delta = Q(1, 32), Q(1, 64)
    assert minus.form_storage == plus.form_storage == 5 * epsilon**2 + 2 * delta**2
    assert minus.storage == plus.storage
    assert (
        minus.continuous_loss
        == plus.continuous_loss
        == (Q(43, 3) * epsilon**2 + 5 * delta**2) / 2
    )
    for region_index, indices in enumerate((range(5), range(5, 10))):
        assert sum(plus.phase_rate[i] for i in indices) == pytest.approx(
            sum(minus.phase_rate[i] for i in indices), abs=2e-15
        )
        assert float(
            plus_report.regions[region_index].phase_response.total_rate
        ) == pytest.approx(
            float(minus_report.regions[region_index].phase_response.total_rate),
            abs=2e-15,
        )


@pytest.mark.parametrize("even_amplitude", (0, 1 / 32))
def test_native_static_acceleration_matches_mixed_even_odd_prediction(even_amplitude):
    pair = _state_rate_pair(even_amplitude=even_amplitude)
    observed = []
    for graph, report in pair:
        field = report.field
        before = tuple(
            (graph.nodes[node]["EPI"], graph.nodes[node]["theta"]) for node in graph
        )
        acceleration = _static_directional_acceleration(graph, field)
        observed.append(acceleration[10] - np.mean(acceleration[15:]))
        assert (
            tuple(
                (graph.nodes[node]["EPI"], graph.nodes[node]["theta"]) for node in graph
            )
            == before
        )
    c, s = math.cos(math.tau / 5), math.sin(math.tau / 5)
    expected = (
        -8
        * 0.5**2
        * even_amplitude
        * (1 / 64)
        * s
        / (math.pi**2 * c * (1 + 2 * c) ** 2)
    )
    difference = observed[1] - observed[0]
    # This is a finite-difference check, not a certified acceleration enclosure.
    assert difference == pytest.approx(expected, rel=2e-7, abs=2e-12)
    if even_amplitude:
        assert difference < 0
    else:
        assert difference == pytest.approx(0, abs=2e-12)
