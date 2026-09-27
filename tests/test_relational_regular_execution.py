"""Opt-in regular-chamber execution without a formation trajectory sweep."""

import math
import pickle
from copy import deepcopy
from fractions import Fraction as Q

import networkx as nx
import pytest

from tnfr.dynamics.relational import (
    RelationalExchangeModel,
    evaluate_relational_exchange,
    step_relational_exchange,
)
from tnfr.sdk import Network, relational_report_to_dict


def _paired_crossing():
    graph = nx.disjoint_union(nx.cycle_graph(5), nx.cycle_graph(5))
    graph.add_edges_from(((0, 5), (1, 6)))
    scale = 2 * math.sqrt(2) / 5
    forms = tuple(
        scale * a
        for a in (math.pi + 8, -math.pi - 8, -3 * math.pi - 4, 0, 3 * math.pi + 4)
    )
    phases = (0, -math.pi, -3 * math.pi / 4, -math.pi / 2, -math.pi / 4)
    for node in graph:
        graph.nodes[node].update(
            EPI=forms[node % 5], theta=phases[node % 5], nu_f=1.0, delta_nfr=999.0
        )
    graph.graph.update(GAMMA={"type": "none"}, _t=0.0)
    return graph


def _pair():
    graph = nx.path_graph(2)
    for node, form in enumerate((1.0, -1.0)):
        graph.nodes[node].update(EPI=form, theta=0.0, nu_f=1.0)
    return graph


def _state(graph):
    return pickle.dumps(
        (graph.graph, tuple(graph.nodes(data=True)), tuple(graph.edges(data=True)))
    )


def _model(**kwargs):
    return RelationalExchangeModel(1.0, phase_domain="positive_resultant", **kwargs)


def test_regular_field_realizes_analytic_transverse_tangent_without_mutating_graph():
    graph = _paired_crossing()
    before = _state(graph)
    with pytest.raises(ValueError, match="acute"):
        evaluate_relational_exchange(graph, model=RelationalExchangeModel(1.0))
    report = evaluate_relational_exchange(graph, model=_model())
    assert _state(graph) == before
    assert report.phase_rate == pytest.approx((2, -2, -1, 0, 1) * 2, abs=1e-14)
    assert report.phase_metric == pytest.approx(
        (2 * math.sqrt(2),) * 2
        + (math.pi * math.sqrt(2),) * 3
        + (2 * math.sqrt(2),) * 2
        + (math.pi * math.sqrt(2),) * 3
    )
    assert report.phase_source == pytest.approx((-0.25, 0.25, 0, 0, 0) * 2, abs=1e-15)
    lower = report.resultant_real_lower_bounds
    assert all(isinstance(a, Q) and a > 0 for a in lower)
    assert 0 < lower[0] < 1 and 1 < lower[2] < 2
    assert "positive" in report.scope and "acute represented" not in report.scope
    assert abs(report.balance_residual) < Q(1, 10**11)


def test_sdk_step_retains_whole_segment_evidence_and_exact_export():
    network = Network(_paired_crossing())
    report = network.step_relational(_model(), dt=1 / 4096)
    assert report.after.phase[1] - report.after.phase[0] < -math.pi
    assert report.after.capacity == report.before.capacity
    assert report.after.edges == report.before.edges
    assert report.energy_change == report.after.storage - report.before.storage
    increments = tuple(
        Q(b) - Q(a) for a, b in zip(report.before.phase, report.after.phase)
    )
    index = {node: i for i, node in enumerate(report.before.nodes)}
    expected = tuple(
        bound - sum(abs(increments[index[j]] - increments[i]) for j in network.G[node])
        for i, (node, bound) in enumerate(
            zip(report.before.nodes, report.before.resultant_real_lower_bounds)
        )
    )
    assert report.segment_resultant_real_lower_bounds == expected
    assert all(a > 0 for a in expected)
    encoded = relational_report_to_dict(report)["report"]
    assert encoded["before"]["model"]["phase_domain"] == "positive_resultant"
    assert encoded["segment_resultant_real_lower_bounds"][0] == {
        "numerator": expected[0].numerator,
        "denominator": expected[0].denominator,
    }


def test_regular_endpoints_cannot_hide_a_nonregular_segment_and_commit_is_atomic():
    graph, model = _pair(), _model(epi_weight=0, phase_weight=1)
    initial = _state(graph)
    field = evaluate_relational_exchange(graph, model=model)
    dt = math.pi**2 / 2
    proposal = deepcopy(graph)
    for i in graph:
        proposal.nodes[i]["theta"] += dt * field.phase_rate[i]
    # Both endpoints admit positive cosine resultants; the intervening full
    # relative turn crosses negative-real resultants and cannot be skipped.
    endpoint = evaluate_relational_exchange(proposal, model=model)
    assert all(a > 0 for a in endpoint.resultant_real_lower_bounds)
    with pytest.raises(ValueError):
        step_relational_exchange(graph, model=model, dt=dt)
    assert _state(graph) == initial


def test_certified_chamber_does_not_trust_positive_libm_cosine_at_negative_row(
    monkeypatch,
):
    from tnfr.dynamics import relational

    graph = _pair()
    graph.nodes[1]["theta"] = math.pi
    before = _state(graph)
    monkeypatch.setattr(relational.math, "cos", lambda value: 1.0)
    with pytest.raises(ValueError):
        evaluate_relational_exchange(graph, model=_model())
    assert _state(graph) == before


@pytest.mark.parametrize("domain", (True, None, "regular", ""))
def test_undeclared_domains_reject(domain):
    with pytest.raises((TypeError, ValueError)):
        RelationalExchangeModel(1.0, phase_domain=domain)


def test_acute_default_keeps_its_original_admission_and_has_no_wider_certificate():
    graph = _pair()
    acute = RelationalExchangeModel(1.0)
    field = evaluate_relational_exchange(graph, model=acute)
    assert acute.phase_domain == "acute"
    assert field.resultant_real_lower_bounds is None
    report = step_relational_exchange(graph, model=acute, dt=1 / 128)
    assert report.segment_resultant_real_lower_bounds is None
    regular = evaluate_relational_exchange(_pair(), model=_model())
    assert regular.pressure == field.pressure
    assert regular.form_rate == field.form_rate
    assert regular.phase_rate == field.phase_rate


def test_zero_capacity_stays_frozen_under_explicit_regular_chamber():
    graph = _paired_crossing()
    for node in graph:
        graph.nodes[node]["nu_f"] = 0.0
    report = step_relational_exchange(graph, model=_model(), dt=1.0)
    assert report.before.epi == report.after.epi
    assert report.before.phase == report.after.phase
    assert (
        report.segment_resultant_real_lower_bounds
        == report.before.resultant_real_lower_bounds
    )
    assert any(value != 0 for value in report.before.pressure)
