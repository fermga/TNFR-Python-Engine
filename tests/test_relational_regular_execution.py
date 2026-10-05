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
    evaluate_relational_uniform_tangent,
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


@pytest.mark.parametrize("domain", (True, None, "unknown", ""))
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


def _regular_source_receiver():
    graph = nx.disjoint_union(nx.cycle_graph(5), nx.cycle_graph(5))
    graph.add_edges_from(((0, 5), (1, 6)))
    for node in graph:
        graph.nodes[node].update(
            EPI=0.0,
            nu_f=1.0,
            theta=math.tau * node / 5 if node < 5 else 6 * math.pi / 5,
        )
    return graph


def test_full_regular_source_receiver_uses_native_slit_metric_and_pressure():
    graph = _regular_source_receiver()
    before = _state(graph)
    model = RelationalExchangeModel(1.0, phase_domain="regular")
    with pytest.raises(ValueError, match="positive real part"):
        evaluate_relational_exchange(graph, model=_model())
    field = evaluate_relational_exchange(graph, model=model)
    assert _state(graph) == before
    assert all(margin > 0 for margin in field.resultant_regular_margin_lower_bounds)
    assert field.resultant_bounds[0][0][1] < 0
    assert field.resultant_bounds[0][1][1] < 0
    assert field.resultant_bounds[1][0][1] < 0
    assert field.resultant_bounds[1][1][0] > 0
    metric = 5 * math.sin(math.pi / 5) / 3
    assert field.phase_metric[:2] == pytest.approx((metric, metric), abs=1e-14)
    assert field.phase_source[:2] == pytest.approx((-0.6, 0.6), abs=1e-14)
    assert field.pressure[:2] == pytest.approx((-0.3, 0.3), abs=1e-14)
    assert field.phase_rate == (0.0,) * 10
    assert field.pressure_path == "relative_resultant_canonical"
    assert max(map(abs, field.pressure_split_residual)) < Q(1, 10**14)
    # This static new admission is not a formation run or a capture claim.
    assert float(field.storage) > 10 * (1 - math.cos(math.tau / 5))
    tangent = evaluate_relational_uniform_tangent(graph, model=model)
    assert tangent.field.phase_source == field.phase_source
    assert all(math.isfinite(value) for row in tangent.generator for value in row)


@pytest.mark.parametrize("gap", (-2.1, 2.1))
def test_regular_uniform_tangent_matches_independent_static_field_differences(gap):
    graph = _pair()
    nx.set_node_attributes(graph, 0.125, "EPI")
    graph.nodes[1].update(theta=gap, nu_f=0.75)
    before = _state(graph)
    model = RelationalExchangeModel(1.0, phase_domain="regular")
    tangent = evaluate_relational_uniform_tangent(graph, model=model)
    assert all(real[1] < 0 for real, _ in tangent.field.resultant_bounds)
    # Independently perturb each physical coordinate and evaluate the full
    # native field. No Jacobian formula or trajectory supplies the reference.
    increment = 2**-19
    for column in range(4):
        node = column % 2
        attribute = "EPI" if column < 2 else "theta"
        plus, minus = deepcopy(graph), deepcopy(graph)
        plus.nodes[node][attribute] += increment
        minus.nodes[node][attribute] -= increment
        positive = evaluate_relational_exchange(plus, model=model)
        negative = evaluate_relational_exchange(minus, model=model)
        positive_rates = positive.form_rate + positive.phase_rate
        negative_rates = negative.form_rate + negative.phase_rate
        differences = tuple(
            (left - right) / (2 * increment)
            for left, right in zip(positive_rates, negative_rates)
        )
        predicted = tuple(row[column] for row in tangent.generator)
        assert predicted == pytest.approx(differences, rel=0, abs=1e-9)
    assert _state(graph) == before


@pytest.mark.parametrize(
    "failure", ("reversed_imaginary_sign", "lost_imaginary_part", "rounded_branch")
)
def test_regular_materialization_cannot_override_exact_admission(monkeypatch, failure):
    graph = _pair()
    graph.nodes[1]["theta"] = gap = 2.1
    model = RelationalExchangeModel(1.0, phase_domain="regular")
    original = evaluate_relational_exchange(graph, model=model)
    assert original.resultant_bounds[0][0][1] < 0
    assert original.resultant_bounds[0][1][0] > 0
    before = _state(graph)
    sine, atan2 = math.sin, math.atan2
    if failure == "rounded_branch":
        monkeypatch.setattr(
            "tnfr.dynamics.relational.math.atan2",
            lambda imag, real: (
                math.copysign(math.pi, imag) if real < 0 else atan2(imag, real)
            ),
        )
        message = "nonpositive-real branch"
    else:
        monkeypatch.setattr(
            "tnfr.dynamics.relational.math.sin",
            lambda angle: (
                (-sine(angle) if failure == "reversed_imaginary_sign" else 0.0)
                if abs(angle) == gap
                else sine(angle)
            ),
        )
        message = "certified component sign"
    # Rational admission remains valid for the mathematical point, but a
    # conflicting or branch-rounded materialization must fail explicitly.
    with pytest.raises(ValueError, match=message):
        evaluate_relational_exchange(graph, model=model)
    assert _state(graph) == before


def test_regular_sdk_step_exports_chord_evidence_and_actual_work_defects():
    graph = _regular_source_receiver()
    graph.nodes[0]["EPI"] = 0.125
    model = RelationalExchangeModel(1.0, phase_domain="regular")
    step = Network(graph).step_relational(model, dt=1 / 4096)
    assert step.segment_resultant_real_lower_bounds is None
    assert all(
        value > 0 for value in step.segment_resultant_regular_margin_lower_bounds
    )
    assert abs(step.before.balance_residual) < Q(1, 10**13)
    assert (
        step.energy_step_defect
        == step.energy_change - Q(step.dt) * step.before.storage_rate
    )
    data = relational_report_to_dict(step)["report"]
    assert data["before"]["model"]["phase_domain"] == "regular"
    margin = step.segment_resultant_regular_margin_lower_bounds[0]
    assert data["segment_resultant_regular_margin_lower_bounds"][0] == {
        "numerator": margin.numerator,
        "denominator": margin.denominator,
    }


def test_regular_step_can_cross_imaginary_axis_without_crossing_excluded_ray():
    graph = _pair()
    graph.nodes[1]["theta"] = 2.1
    report = step_relational_exchange(
        graph, model=RelationalExchangeModel(1.0, phase_domain="regular"), dt=0.4
    )
    assert report.before.relative_resultant[0][0] < 0
    assert report.after.relative_resultant[0][0] > 0
    assert all(
        value > 0 for value in report.segment_resultant_regular_margin_lower_bounds
    )


def test_regular_branch_crossing_rejects_atomically_despite_regular_endpoints():
    graph = _pair()
    model = RelationalExchangeModel(
        1.0, epi_weight=0, phase_weight=1, phase_domain="regular"
    )
    field = evaluate_relational_exchange(graph, model=model)
    duration = math.pi**2 / 2
    proposal = deepcopy(graph)
    for i in graph:
        proposal.nodes[i]["theta"] += duration * field.phase_rate[i]
    endpoint = evaluate_relational_exchange(proposal, model=model)
    assert all(value > 0 for value in endpoint.resultant_regular_margin_lower_bounds)
    before = _state(graph)
    with pytest.raises(ValueError, match="whole Euler phase segment"):
        step_relational_exchange(graph, model=model, dt=duration)
    assert _state(graph) == before


def test_regular_point_rejects_a_materialized_unresolved_negative_axis():
    graph = _pair()
    graph.nodes[1]["theta"] = math.pi
    before = _state(graph)
    with pytest.raises(ValueError, match="ray|branch"):
        evaluate_relational_exchange(
            graph, model=RelationalExchangeModel(1.0, phase_domain="regular")
        )
    assert _state(graph) == before


def test_regular_positive_axis_limit_and_zero_capacity_keep_nonzero_pressure():
    graph = _pair()
    nx.set_node_attributes(graph, 0, "nu_f")
    model = RelationalExchangeModel(1.0, phase_domain="regular")
    step = step_relational_exchange(graph, model=model, dt=1)
    assert step.before.phase_metric == (math.pi, math.pi)
    assert step.before.epi == step.after.epi and step.before.phase == step.after.phase
    assert any(value != 0 for value in step.before.pressure)
    assert (
        step.segment_resultant_regular_margin_lower_bounds
        == step.before.resultant_regular_margin_lower_bounds
    )
