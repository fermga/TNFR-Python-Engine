"""Unweighted phase response retains mobility geometry and arithmetic defects.

These static observations add neither a trajectory nor a regional evolution
law. Exact identities concern captured represented inputs and actual rates.
"""

import math
import pickle
from dataclasses import FrozenInstanceError
from fractions import Fraction as Q

import networkx as nx
import pytest

from benchmarks.relational_local_composition import analyze_local_composition
from tnfr.dynamics.relational import RelationalExchangeModel
from tnfr.physics import relational_observations as owner

MODEL = RelationalExchangeModel(1.0)


def _path(*, capacity=(1, 1, 1), form=(1, 0, 0), phase=(0, 0, 0)):
    graph = nx.path_graph(3)
    graph.graph.update(GAMMA={"type": "none"}, preserved={"values": [1, 2]})
    for node in graph:
        graph.nodes[node].update(
            EPI=form[node], theta=phase[node], nu_f=capacity[node], delta_nfr=999.0
        )
    return graph


def _observe(graph, regions, *, model=MODEL):
    return owner.observe_relational_pattern(
        graph,
        model=model,
        reference_phase=dict.fromkeys(graph, 0.0),
        regions=regions,
    )


def test_geometry_alone_gives_nonzero_mean_phase_rate_at_zero_total_cut():
    report = _observe(_path(), ((0, 1, 2),))
    field, region = report.field, report.regions[0]
    response = region.phase_response
    inverse_pi = 1 / Q(math.pi)
    assert field.phase_mobility == (inverse_pi, inverse_pi / 2, inverse_pi)
    assert region.boundary.cut.outward_cut_current == 0
    assert response.mean_mobility == 5 * inverse_pi / 6
    assert response.mean_form_gradient == 0
    assert response.mobility_variance == inverse_pi**2 / 18
    assert response.form_gradient_variance == Q(2, 3)
    assert response.mobility_gradient_covariance == inverse_pi / 6
    assert response.mean_mobility_boundary_rate == 0
    assert response.covariance_rate == response.model_total_rate == inverse_pi / 4
    assert response.covariance_rate_squared_bound == inverse_pi**2 / 12
    assert response.total_rate > 0
    assert response.mean_rate == response.total_rate / 3
    assert response.identity_residual == 0
    assert response.total_rate == response.model_total_rate + response.rounding_residual
    assert response.rounding_residual == sum(field.phase_rate_rounding_defect)


def test_response_uses_exact_form_gradient_instead_of_its_rounded_display():
    epsilon = 2.0**-54
    report = _observe(_path(form=(1, epsilon, 0)), ((0, 1, 2),))
    field, response = report.field, report.regions[0].phase_response
    assert Q(field.form_gradient[0]) != field.work.form_gradient[0]
    inverse_pi = 1 / Q(math.pi)
    assert field.work.form_gradient[0] == 1 - Q(epsilon)
    assert response.mean_form_gradient == 0
    assert response.model_total_rate == inverse_pi / 4 - inverse_pi * Q(epsilon) / 2
    assert response.identity_residual == 0


def test_capacity_heterogeneity_can_produce_the_same_kind_of_covariance():
    graph = nx.cycle_graph(3)
    for node in graph:
        graph.nodes[node].update(EPI=(1, 0, 0)[node], theta=0, nu_f=(1, 2, 4)[node])
    report = _observe(graph, ((0, 1, 2),))
    field, region = report.field, report.regions[0]
    inverse_pi = 1 / Q(math.pi)
    assert field.phase_metric == (2 * math.pi,) * 3
    assert field.phase_mobility == (inverse_pi / 2, inverse_pi, 2 * inverse_pi)
    assert region.phase_response.mobility_gradient_covariance == -2 * inverse_pi / 3
    assert region.phase_response.model_total_rate == -inverse_pi
    assert region.phase_response.total_rate < 0
    assert region.phase_response.identity_residual == 0


def test_compensating_capacity_and_geometry_remove_the_covariance_correction():
    report = _observe(_path(capacity=(1, 2, 1)), ((0, 1, 2),))
    response = report.regions[0].phase_response
    assert len(set(report.field.phase_mobility)) == 1
    assert response.mobility_variance == response.mobility_gradient_covariance == 0
    assert response.covariance_rate == response.covariance_rate_squared_bound == 0
    assert response.model_total_rate == 0
    assert response.total_rate == response.rounding_residual


def test_singleton_has_no_covariance_and_retains_its_actual_rounding():
    report = _observe(_path(), ((0,),))
    response, field = report.regions[0].phase_response, report.field
    assert response.mobility_variance == response.form_gradient_variance == 0
    assert response.mobility_gradient_covariance == 0
    assert response.covariance_rate == response.covariance_rate_squared_bound == 0
    assert response.mean_mobility_boundary_rate == response.model_total_rate
    assert response.total_rate == response.mean_rate == Q(field.phase_rate[0])
    assert response.rounding_residual == field.phase_rate_rounding_defect[0]
    assert response.identity_residual == 0


def test_lossless_positive_resultant_response_admits_a_nonacute_support_edge():
    graph = nx.disjoint_union(nx.complete_graph(3), nx.complete_graph(3))
    graph.add_edge(0, 3)
    for node in graph:
        graph.nodes[node].update(
            EPI=1.0 if node == 0 else 0.0,
            theta=0.0 if node < 3 else 2 * math.pi / 3,
            nu_f=1.0,
        )
    model = RelationalExchangeModel(
        2.0, epi_weight=0.0, phase_weight=0.5, phase_domain="positive_resultant"
    )
    assert model.phase_weight == 1.0
    report = _observe(graph, ((0, 1, 2), tuple(graph)), model=model)
    field = report.field
    assert abs(field.phase[3] - field.phase[0]) > math.pi / 2
    assert min(field.resultant_real_lower_bounds) > 0
    gradient = (3, -1, -1, -1, 0, 0)
    for region, indices in zip(
        report.regions, ((0, 1, 2), tuple(range(6))), strict=True
    ):
        response = region.phase_response
        assert region.transport is None
        assert region.transport_unavailable_reason == "zero_epi_weight"
        assert response.identity_residual == 0
        assert response.total_rate == sum(
            (Q(field.phase_rate[i]) for i in indices), Q(0)
        )
        assert response.model_total_rate == sum(
            (Q(gradient[i]) / (2 * Q(field.phase_metric[i])) for i in indices), Q(0)
        )
        assert (
            response.total_rate
            == response.model_total_rate + response.rounding_residual
        )


@pytest.mark.parametrize("capacity", ((0, 0, 0), (1, 0, 2)))
def test_zero_capacity_retains_unweighted_response_when_divided_balance_is_unavailable(
    capacity,
):
    report = _observe(_path(capacity=capacity), ((1,), (0, 1, 2)))
    for region in report.regions:
        assert (
            region.boundary.weighted_rate_unavailable_reason
            == "zero_capacity_in_region"
        )
        assert region.boundary.phase_weighted_rate is None
        assert region.phase_response is not None
        assert region.phase_response.identity_residual == 0
        assert (
            region.phase_response.covariance_rate**2
            <= region.phase_response.covariance_rate_squared_bound
        )
    inactive = report.regions[0].phase_response
    assert (
        inactive.mean_mobility == inactive.model_total_rate == inactive.total_rate == 0
    )
    assert inactive.rounding_residual == 0


def test_internal_edges_and_cut_give_an_independent_phase_work_identity():
    graph = nx.path_graph(4)
    for node in graph:
        graph.nodes[node].update(
            EPI=(0.75, 0.125, -0.25, 0.5)[node],
            theta=(0, 0.2, -0.1, 0.15)[node],
            nu_f=(1, 0, 2, 0.5)[node],
        )
    report = _observe(graph, ((0, 1, 2),), model=RelationalExchangeModel(2))
    field, region = report.field, report.regions[0]
    mobility, form = field.phase_mobility, tuple(map(Q, field.epi))
    # Region-internal edges cancel q into differences of mobility; the cut does not.
    edge_sum = (
        (mobility[0] - mobility[1]) * (form[0] - form[1])
        + (mobility[1] - mobility[2]) * (form[1] - form[2])
        + mobility[2] * (form[2] - form[3])
    )
    response = region.phase_response
    assert response.model_total_rate == edge_sum / 4
    assert response.total_rate == edge_sum / 4 + response.rounding_residual
    assert response.identity_residual == 0
    assert response.covariance_rate**2 <= response.covariance_rate_squared_bound
    assert (
        response.total_rate
        - response.mean_mobility_boundary_rate
        - response.rounding_residual
    ) ** 2 <= response.covariance_rate_squared_bound
    assert response.mobility_gradient_covariance**2 <= (
        response.mobility_variance * response.form_gradient_variance
    )
    for i, gradient in enumerate(field.work.form_gradient):
        assert field.phase_rate_rounding_defect[i] == (
            Q(field.phase_rate[i]) - field.phase_mobility[i] * gradient / 4
        )


def test_complement_rates_add_even_when_mean_mobility_cut_terms_do_not_cancel():
    report = _observe(_path(), ((0,), (1, 2), (0, 1, 2)))
    left, right, whole = report.regions
    assert (
        left.boundary.cut.outward_cut_current == -right.boundary.cut.outward_cut_current
    )
    a, b, all_nodes = (region.phase_response for region in report.regions)
    assert a.mean_mobility_boundary_rate + b.mean_mobility_boundary_rate != 0
    for name in ("total_rate", "model_total_rate", "rounding_residual"):
        assert getattr(a, name) + getattr(b, name) == getattr(all_nodes, name)
    assert a.total_rate + b.total_rate == sum(map(Q, report.field.phase_rate))


def test_form_reversal_reverses_phase_response_but_preserves_its_squared_bound():
    graph = _path(phase=(-0.25, 0, 0.125))
    first = _observe(graph, ((0, 1, 2),)).regions[0].phase_response
    for node in graph:
        graph.nodes[node]["EPI"] *= -1
    second = _observe(graph, ((0, 1, 2),)).regions[0].phase_response
    for name in (
        "mean_mobility",
        "mobility_variance",
        "form_gradient_variance",
        "covariance_rate_squared_bound",
    ):
        assert getattr(second, name) == getattr(first, name)
    for name in (
        "mean_form_gradient",
        "mobility_gradient_covariance",
        "mean_mobility_boundary_rate",
        "covariance_rate",
        "model_total_rate",
        "rounding_residual",
        "total_rate",
        "mean_rate",
    ):
        assert getattr(second, name) == -getattr(first, name)
    assert first.identity_residual == second.identity_residual == 0


def test_response_reuses_one_fresh_read_only_field_and_is_detached(monkeypatch):
    graph = _path()
    before = pickle.dumps(
        (graph.graph, tuple(graph.nodes(data=True)), tuple(graph.edges(data=True)))
    )
    original, calls = owner.evaluate_relational_exchange, []

    def observe(*args, **kwargs):
        calls.append(args[0])
        return original(*args, **kwargs)

    monkeypatch.setattr(owner, "evaluate_relational_exchange", observe)
    report = _observe(graph, ((2, 0), (1,), (0, 1)))
    assert calls == [graph]
    assert (
        pickle.dumps(
            (graph.graph, tuple(graph.nodes(data=True)), tuple(graph.edges(data=True)))
        )
        == before
    )
    retained = report.regions[0].phase_response
    graph.nodes[0]["nu_f"] = 0
    assert retained.mean_mobility == 1 / Q(math.pi)
    with pytest.raises(FrozenInstanceError):
        retained.total_rate = Q(0)


def test_relabeling_retains_the_same_selected_regional_response():
    graph = _path(phase=(-0.25, 0, 0.125), capacity=(1, 0, 2))
    mapping = {0: ("left", 0), 1: "middle", 2: 19}
    first = _observe(graph, ((2, 0), (1,)))
    changed = nx.relabel_nodes(graph, mapping)
    second = _observe(changed, ((19, ("left", 0)), ("middle",)))
    assert first.field.phase_mobility == second.field.phase_mobility
    assert tuple(region.phase_response for region in first.regions) == tuple(
        region.phase_response for region in second.regions
    )


def test_same_even_coordinates_and_zero_cut_can_have_different_covariance_response():
    graph = nx.disjoint_union(nx.cycle_graph(5), nx.cycle_graph(5))
    graph.add_edge(0, 5)
    form = (0, 1 / 32, -1 / 32, -1 / 32, 1 / 32) + (0,) * 5
    for node in graph:
        graph.nodes[node].update(EPI=form[node], theta=node % 5 * math.tau / 5, nu_f=1)
    changed = graph.copy()
    changed.nodes[1]["theta"] += 1 / 64
    changed.nodes[4]["theta"] -= 1 / 64
    rows = analyze_local_composition(
        ring_cosine=Q(1, 3), inverse_pi=Q(1, 3)
    ).natural_rows

    def natural_state(value):
        state = tuple(
            Q(value.nodes[node][key]) for key in ("EPI", "theta") for node in value
        )
        return tuple(
            sum(a * b for a, b in zip(row, state, strict=True)) for row in rows
        )

    assert natural_state(graph) == natural_state(changed)
    first, second = (_observe(value, (tuple(range(5)),)) for value in (graph, changed))
    for report in (first, second):
        region = report.regions[0]
        assert region.boundary.cut.outward_cut_current == 0
        assert region.phase_response.mean_mobility_boundary_rate == 0
        assert (
            region.phase_response.model_total_rate
            == region.phase_response.covariance_rate
        )
        assert region.phase_response.identity_residual == 0
    a, b = (report.regions[0].phase_response for report in (first, second))
    assert a.model_total_rate != b.model_total_rate
    assert a.total_rate != b.total_rate
    assert b.total_rate - a.total_rate == (
        b.covariance_rate
        - a.covariance_rate
        + b.rounding_residual
        - a.rounding_residual
    )
