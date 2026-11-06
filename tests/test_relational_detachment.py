"""Component capture and supplied two-bridge work, without a trajectory."""

import math
from decimal import Decimal, localcontext
from fractions import Fraction as Q

import networkx as nx
import pytest

from tests.test_relational_capture import CYCLES, _graph, _local_graph, _model, _state
from tnfr.dynamics.relational import evaluate_relational_exchange
from tnfr.physics.relational_capture import (
    certify_relational_cycle_capture,
    certify_relational_sector_capture,
    observe_relational_detachment,
)


def _cycle_certificate(graph, *, cycle=CYCLES[0], sector=1, model=None):
    return certify_relational_cycle_capture(
        graph,
        model=_model(storage_scale=1) if model is None else model,
        cycle=cycle,
        target_sector=sector,
    )


def _detach(graph, *, cycles=CYCLES, sector=1, model=None):
    return observe_relational_detachment(
        graph,
        model=_model(storage_scale=1) if model is None else model,
        cycles=cycles,
        target_sector=sector,
    )


def test_c5_barrier_uses_independent_radical_and_oriented_integer_period():
    graph = _local_graph().subgraph(CYCLES[0]).copy()
    for node in graph:
        graph.nodes[node]["nu_f"] = (node + 1) / 4
    before = _state(graph)
    certificate = _cycle_certificate(graph, model=_model(storage_scale=2))
    reverse = _cycle_certificate(
        graph,
        cycle=tuple(reversed(CYCLES[0])),
        sector=-1,
        model=_model(storage_scale=2),
    )
    assert _state(graph) == before
    assert certificate.admitted and reverse.admitted
    assert certificate.cycle_winding == 1 and reverse.cycle_winding == -1
    assert certificate.field.capacity == tuple((i + 1) / 4 for i in range(5))
    assert certificate.storage_bounds == reverse.storage_bounds
    assert certificate.field.form_storage == 0
    with localcontext() as context:
        context.prec = 80
        # cos(3*pi/8)=sqrt(2-sqrt(2))/2, independently of series admission.
        face = Q(Decimal(5) - 2 * (Decimal(2) - Decimal(2).sqrt()).sqrt())
        minimum = Q(5 * (Decimal(5) - Decimal(5).sqrt()) / 4)
    lower, upper = certificate.geometric_barrier_bounds
    assert lower < face < upper
    assert certificate.capture_barrier_bounds == (2 * lower, 2 * upper)
    assert minimum < face
    assert certificate.storage_bounds[1] < 2 * face
    assert certificate.normalized_energy_margin_lower_bound > 0


def test_cycle_certificate_separates_energy_winding_and_loss_premises():
    graph = _local_graph().subgraph(CYCLES[0]).copy()
    graph.nodes[0]["nu_f"] = 0
    frozen = _cycle_certificate(graph)
    assert frozen.energy_admitted and frozen.sector_admitted
    assert frozen.field.form_rate[0] == frozen.field.phase_rate[0] == 0
    assert frozen.unavailable_reasons == ("strictly_positive_held_capacity_required",)
    graph.nodes[0]["nu_f"] = 1
    lossless = _cycle_certificate(
        graph, model=_model(storage_scale=1, epi_weight=0, phase_weight=1)
    )
    assert lossless.unavailable_reasons == ("positive_epi_weight_required",)
    graph.nodes[0]["EPI"] = 1
    excessive = _cycle_certificate(graph)
    assert excessive.acute_admitted and excessive.sector_admitted
    assert excessive.unavailable_reasons == (
        "strict_cycle_energy_barrier_not_certified",
    )
    consensus = _graph(a=0, b=0).subgraph(CYCLES[0]).copy()
    wrong_period = _cycle_certificate(consensus)
    assert wrong_period.energy_admitted and wrong_period.acute_admitted
    assert wrong_period.cycle_winding == 0
    assert wrong_period.unavailable_reasons == ("declared_cycle_period_not_certified",)
    assert all(
        not row.admitted and row.target_sector is None
        for row in (frozen, lossless, excessive, wrong_period)
    )


@pytest.mark.parametrize("cycle", ((0, 1, 2, 3), (0, 1, 2, 3, 3), (0, 2, 1, 3, 4)))
def test_cycle_certificate_rejects_a_different_support_without_mutation(cycle):
    graph = _local_graph().subgraph(CYCLES[0]).copy()
    before = _state(graph)
    with pytest.raises(ValueError, match="cycle|support"):
        _cycle_certificate(graph, cycle=cycle)
    assert _state(graph) == before


def test_joined_sector_inherits_component_capture_but_is_not_required():
    graph = _local_graph()
    for node in graph:
        graph.nodes[node]["nu_f"] = (node + 1) / 8
    graph.nodes[2]["EPI"] = 1 / 64
    graph.nodes[7]["theta"] += 1 / 128
    before = _state(graph)
    pair = certify_relational_sector_capture(
        graph, model=_model(storage_scale=1), cycles=CYCLES
    )
    report = _detach(graph)
    assert pair.admitted and report.capture_admitted
    assert all(row.target_sector == 1 for row in report.components)
    assert report.components[0].field.epi != report.components[1].field.epi
    assert _state(graph) == before

    # A common form offset inside the other ring adds only bridge storage.
    # Direct component admission remains useful outside the sufficient pair basin.
    graph = _local_graph()
    for node in CYCLES[1]:
        graph.nodes[node]["EPI"] = 1 / 8
    pair = certify_relational_sector_capture(
        graph, model=_model(storage_scale=1), cycles=CYCLES
    )
    report = _detach(graph)
    assert pair.acute_admitted and pair.sector_admitted and not pair.energy_admitted
    assert report.capture_admitted
    assert report.reset.form_support_change == -Q(1, 64)
    assert report.reset.phase_support_change == 0
    assert report.reset.storage_change == -Q(1, 64)
    assert report.reset.identity_residual == 0


def test_detachment_evaluates_only_connected_fields_and_orders_changes_by_before(
    monkeypatch,
):
    from tnfr.physics import relational_capture as owner

    graph = _local_graph()
    graph.nodes[0]["EPI"] = graph.nodes[5]["EPI"] = 1 / 64
    labels = {node: ("node", 20 - node) for node in graph}
    relabeled = nx.relabel_nodes(graph, labels)
    graph = nx.Graph()
    graph.graph.update(relabeled.graph)
    graph.add_nodes_from(reversed(tuple(relabeled.nodes(data=True))))
    graph.add_edges_from(reversed(tuple(relabeled.edges(data=True))))
    cycles = tuple(tuple(labels[node] for node in cycle) for cycle in CYCLES)
    calls = []
    evaluate = owner.evaluate_relational_exchange

    def connected_only(value, **kwargs):
        assert nx.is_connected(value)
        calls.append(tuple(value))
        return evaluate(value, **kwargs)

    monkeypatch.setattr(owner, "evaluate_relational_exchange", connected_only)
    before = _state(graph)
    report = _detach(graph, cycles=cycles)
    assert _state(graph) == before
    assert len(calls) == 3 and tuple(map(len, calls)) == (10, 5, 5)
    assert calls[0] == report.before.nodes == tuple(graph)
    assert report.removed_bridges == ((labels[0], labels[5]), (labels[1], labels[6]))
    assert len(report.reset.edges_before) == 12 and len(report.reset.edges_after) == 10
    assert all(graph.nodes[node]["delta_nfr"] == 999 for node in graph)
    for name in ("pressure", "form_rate", "phase_rate", "phase_metric"):
        after = {
            node: value
            for component in report.components
            for node, value in zip(
                component.field.nodes, getattr(component.field, name)
            )
        }
        expected = tuple(
            Q(after[node]) - Q(value)
            for node, value in zip(report.before.nodes, getattr(report.before, name))
        )
        assert getattr(report, name + "_change") == expected
    for name in ("form_storage", "phase_storage", "storage"):
        expected = sum(getattr(row.field, name) for row in report.components) - getattr(
            report.before, name
        )
        assert getattr(report, "field_" + name + "_change") == expected


def test_zero_event_cost_can_change_rates_and_metrics_at_all_four_ports():
    graph = _local_graph()
    graph.nodes[0]["EPI"] = graph.nodes[5]["EPI"] = 1 / 64
    report = _detach(graph)
    assert report.capture_admitted
    assert report.reset.form_state_change == report.reset.phase_state_change == 0
    assert report.reset.form_support_change == report.reset.phase_support_change == 0
    assert report.reset.storage_change == report.reset.identity_residual == 0
    assert report.field_storage_change == 0
    assert report.storage_reconciliation_residual == 0
    ports = {0, 1, 5, 6}
    for name in ("form_rate_change", "phase_rate_change", "phase_metric_change"):
        nonzero = {
            node
            for node, value in zip(report.before.nodes, getattr(report, name))
            if value
        }
        assert nonzero == ports
    # The two ring-internal differences are unchanged, but degrees are reduced.
    # At port zero the EPI contribution changes from -(2/3)*(1/64) to -1/64.
    assert float(report.form_rate_change[0]) == pytest.approx(-1 / 384, abs=1e-16)


def test_early_zero_cost_cut_retains_valid_fields_without_capture_claim():
    # This copied formation seed has positive post-cut resultants, but its
    # large cycle gaps do not admit the later acute winding-one certificate.
    a = math.pi / 2 - 1 / 64
    graph = _graph(a=a, b=a, amplitude=0.4, contrast=0.2)
    before = _state(graph)
    report = _detach(graph)
    assert _state(graph) == before
    assert report.reset.storage_change == 0
    assert not report.capture_admitted
    for component in report.components:
        assert all(value > 0 for value in component.field.phase_metric)
        assert not component.acute_admitted and component.cycle_winding is None
        assert component.target_sector is None
        assert "strict_acute_edge_lifts_not_certified" in component.unavailable_reasons
    assert any(report.form_rate_change)


def test_joined_field_admission_does_not_replace_postcut_admission():
    graph = _graph(a=2.1, b=1.0)
    before = _state(graph)
    joined = evaluate_relational_exchange(graph, model=_model(storage_scale=1))
    assert all(value > 0 for value in joined.phase_metric)
    with pytest.raises(ValueError, match="positive real part"):
        _detach(graph)
    assert _state(graph) == before


def test_event_cost_and_native_raw_phase_storage_keep_representation_residual():
    graph = _local_graph()
    for node in CYCLES[1]:
        # A supplied raw offset, not an asserted exact multiple of mathematical pi.
        graph.nodes[node]["theta"] += 6.25
    before = _state(graph)
    report = _detach(graph)
    assert _state(graph) == before
    assert report.capture_admitted
    assert report.reset.form_state_change == report.reset.phase_state_change == 0
    assert report.reset.form_storage_change == 0
    assert report.reset.phase_storage_change < 0
    assert report.reset.identity_residual == 0
    assert (
        report.storage_reconciliation_residual
        == (report.field_storage_change - report.reset.storage_change)
        != 0
    )
    # Both ledgers retain their arithmetic; neither is rounded into equality.
    assert report.field_phase_storage_change != report.reset.phase_storage_change
