"""Independent static mechanism checks; no evolved or fitted final offset."""

import cmath
import math
from fractions import Fraction as Q

import networkx as nx
import pytest

from tnfr.dynamics.relational import (
    RelationalExchangeModel,
    evaluate_relational_exchange,
    evaluate_relational_uniform_tangent,
)
from tnfr.physics.relational_observations import (
    observe_relational_attachment,
    observe_relational_pattern,
)


def _ring(*, start=0, u=(0,) * 5, v=(0,) * 5, amplitude=0, offset=0, capacity=1):
    graph = nx.cycle_graph(range(start, start + 5))
    for i, node in enumerate(graph):
        graph.nodes[node].update(
            EPI=amplitude * u[i],
            theta=2 * math.pi * i / 5 + amplitude * v[i] + offset,
            nu_f=capacity,
        )
    return graph


@pytest.mark.parametrize("capacity", (1, 2))
def test_native_covariance_supplies_the_derived_quadratic_mean_rate(capacity):
    u, v = (1, -1, 0, 0, 0), (0, 1, -1, 0, 0)
    model = RelationalExchangeModel(1)
    epsilon = 2**-12
    means = []
    for sign in (1, -1):
        graph = _ring(
            u=u,
            v=tuple(sign * value for value in v),
            amplitude=epsilon,
            capacity=capacity,
        )
        report = observe_relational_pattern(
            graph,
            model=model,
            reference_phase={i: 2 * math.pi * i / 5 for i in graph},
            regions=(tuple(graph),),
        )
        response = report.regions[0].phase_response
        assert response.identity_residual == 0
        assert response.mean_mobility_boundary_rate == 0
        assert response.model_total_rate == response.covariance_rate
        assert abs(sum(report.field.form_rate)) < 1e-15
        means.append(float(response.mean_rate))
    laplacian_u = tuple(2 * u[i] - u[(i + 1) % 5] - u[(i - 1) % 5] for i in range(5))
    skew_v = tuple(v[(i + 1) % 5] - v[(i - 1) % 5] for i in range(5))
    kappa = math.tau / 5
    expected = (
        capacity
        * model.phase_weight
        * math.sin(kappa)
        / (20 * model.storage_scale * math.pi * math.cos(kappa) ** 2)
        * sum(a * b for a, b in zip(laplacian_u, skew_v))
    )
    # Centered phase variation cancels odd-in-epsilon errors in this local
    # coefficient check; it is not a long-time or finite-response enclosure.
    observed = (means[0] - means[1]) / (2 * epsilon**2)
    assert observed == pytest.approx(expected, rel=3e-6)


def test_reference_attachment_reads_signed_offset_while_cost_is_even():
    model = RelationalExchangeModel(1)
    delta = 2**-10
    reports = []
    for sign in (1, -1):
        left = _ring(offset=sign * delta)
        right = _ring(start=5)
        report = observe_relational_attachment(left, right, model=model, bridge=(0, 5))
        reports.append(report)
        assert left.number_of_edges() == right.number_of_edges() == 5
        assert report.storage_change > 0
        assert not report.represented_zero_supply_passive
        assert all(value == 0 for value in report.joined.phase_rate)
        assert sign * report.joined.form_rate[0] < 0
        assert sign * report.joined.form_rate[5] > 0
    plus, minus = reports
    assert float(plus.storage_change) == pytest.approx(
        float(minus.storage_change), abs=2e-15
    )
    cosine = math.cos(math.tau / 5)
    expected = -model.phase_weight / (math.pi * (1 + 2 * cosine))
    derivative = (plus.joined.form_rate[0] - minus.joined.form_rate[0]) / (2 * delta)
    assert derivative == pytest.approx(expected, rel=1e-7)
    expected_cost = model.storage_scale * (1 - math.cos(delta))
    assert float(plus.storage_change) == pytest.approx(expected_cost, abs=2e-15)
    # A supplied work budget may admit this hypothetical addition; equality
    # of endpoint storage across the two signs does not erase their response.
    work = max(plus.storage_change, minus.storage_change)
    assert plus.assess_supply(work).represented_balance_satisfied
    assert minus.assess_supply(work).represented_balance_satisfied
    assert work > Q(0)


def test_finite_residual_contact_keeps_background_and_port_degree_changes():
    model = RelationalExchangeModel(1)
    epsilon, offset = 2**-12, 2**-9
    u, v = (1, -1, 0, 0, 0), (0, 1, -1, 0, 0)
    left = _ring(u=u, v=v, amplitude=epsilon, offset=offset)
    right = _ring(start=5)
    report = observe_relational_attachment(left, right, model=model, bridge=(0, 5))
    form = [epsilon * value for value in u]
    phase = [epsilon * value for value in v]
    kappa = math.tau / 5
    gap = offset + phase[0]
    left_resultant = (
        cmath.exp(1j * (kappa + phase[1] - phase[0]))
        + cmath.exp(1j * (-kappa + phase[4] - phase[0]))
        + cmath.exp(-1j * gap)
    )
    right_resultant = 2 * math.cos(kappa) + cmath.exp(1j * gap)
    expected_left = (
        -(3 * form[0] - form[1] - form[4]) / 6 + cmath.phase(left_resultant) / math.tau
    )
    expected_right = form[0] / 6 + cmath.phase(right_resultant) / math.tau
    expected_background = -(2 * form[0] - form[1] - form[4]) / 4 - (
        2 * phase[0] - phase[1] - phase[4]
    ) / (4 * math.pi)
    assert report.joined.form_rate[0] == pytest.approx(expected_left, abs=1e-15)
    assert report.joined.form_rate[5] == pytest.approx(expected_right, abs=1e-15)
    assert report.components[0].form_rate[0] == pytest.approx(
        expected_background, abs=1e-15
    )
    assert report.components[1].form_rate[0] == pytest.approx(0, abs=1e-15)
    # At insertion the untouched receiver's other four nodes still have
    # zero form rate. Its regional mean rate is the port rate divided by
    # five, although neither regional sum is constrained to cancel the other.
    assert report.joined.form_rate[6:] == pytest.approx([0] * 4, abs=1e-15)
    assert sum(report.joined.form_rate[5:]) / 5 == pytest.approx(
        expected_right / 5, abs=1e-15
    )
    assert abs(expected_left + expected_right) > epsilon / 4
    assert report.joined.phase_rate[0] != 0 and report.joined.phase_rate[5] != 0
    expected_work = form[0] ** 2 / 2 + 2 * math.sin(gap / 2) ** 2
    assert float(report.storage_change) == pytest.approx(expected_work, abs=1e-15)
    assert left.number_of_edges() == right.number_of_edges() == 5


def test_postcut_mean_conservation_requires_common_capacity_and_keeps_offsets():
    model = RelationalExchangeModel(1)
    graph = _ring(u=(2, -1, 3, 0, 1), v=(0, 1, -1, 2, 0), amplitude=2**-8)
    for data in graph.nodes.values():
        data["EPI"] += 3
    for capacity in (1, 2):
        nx.set_node_attributes(graph, capacity, "nu_f")
        field = evaluate_relational_exchange(graph, model=model)
        assert sum(field.form_rate) == pytest.approx(0, abs=1e-14)
        assert abs(field.form_rate[0]) > 1e-4
    # Isolated recovery with nonuniform capacity is a different hypothesis:
    # the arithmetic mean need not be invariant even on this acute C5.
    graph.nodes[0]["nu_f"] = 3
    heterogeneous = evaluate_relational_exchange(graph, model=model)
    assert sum(heterogeneous.form_rate) == pytest.approx(
        field.form_rate[0] / 2, abs=1e-14
    )


@pytest.mark.parametrize("joined", (False, True))
def test_native_growth_and_form_derivative_match_the_contact_proof_domain(joined):
    left = _ring(u=(1, -1, 0, 0, 0), v=(0, 1, -1, 0, 0), amplitude=2**-12, offset=2**-9)
    graph = nx.compose(left, _ring(start=5)) if joined else left
    if joined:
        graph.add_edge(0, 5)
    model = RelationalExchangeModel(1)
    field = evaluate_relational_exchange(graph, model=model)
    radius = 2**-9 + 2**-12
    assert max(map(abs, (*field.form_rate, *field.phase_rate))) < 3 * radius
    for node, (real, _) in zip(field.nodes, field.relative_resultant):
        assert real > graph.degree[node] * 7 / 25
    # The form row depends linearly on form, so this shared tangent's upper
    # rows give its same Jacobian at nonuniform form and the retained phases.
    uniform = graph.copy()
    nx.set_node_attributes(uniform, 0, "EPI")
    tangent = evaluate_relational_uniform_tangent(uniform, model=model)
    assert all(sum(map(abs, row)) < 3 for row in tangent.generator[: len(graph)])
