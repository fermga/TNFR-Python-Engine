"""Signed work and regional ports reuse one detached relational field.

Small analytic states test the implemented accounting without evolving a
trajectory, selecting a basin or rebuilding a retained research campaign.
"""

import pickle
from dataclasses import FrozenInstanceError
from fractions import Fraction as Q

import networkx as nx
import pytest

from tnfr.dynamics.relational import (
    RelationalExchangeModel,
    evaluate_relational_exchange,
)
from tnfr.physics.relational_observations import observe_relational_pattern

MODEL = RelationalExchangeModel(2.0)


def _path(*, form=(0.75, 0.25, 0.5), phase=(0.0, 0.0, 0.0), capacity=(1, 0.5, 2)):
    graph = nx.path_graph(3)
    graph.graph.update(GAMMA={"type": "none"}, preserved={"items": [1, 2]})
    for node in graph:
        graph.nodes[node].update(
            EPI=form[node], theta=phase[node], nu_f=capacity[node], delta_nfr=999.0
        )
    return graph


def _snapshot(graph):
    return pickle.dumps(
        (graph.graph, tuple(graph.nodes(data=True)), tuple(graph.edges(data=True))),
        protocol=5,
    )


def _pattern(graph, regions, *, model=MODEL):
    return observe_relational_pattern(
        graph,
        model=model,
        reference_phase=dict.fromkeys(graph, 0.0),
        regions=regions,
    )


def test_analytic_nodal_work_matches_heterogeneous_diffusion_and_total_balance():
    field = evaluate_relational_exchange(_path(), model=MODEL)
    work = field.work
    assert work.form_gradient == (Q(1, 2), Q(-3, 4), Q(1, 4))
    assert work.dissipation == (Q(1, 8), Q(9, 128), Q(1, 16))
    assert work.form_work == (Q(-1, 8), Q(-9, 128), Q(-1, 16))
    zero = (Q(0),) * 3
    assert work.exchange == work.phase_work == zero
    assert work.form_residual == work.phase_residual == work.balance_residual == zero
    assert sum(work.dissipation) == field.continuous_loss
    assert sum(work.form_work) + sum(work.phase_work) == field.storage_rate
    assert sum(work.balance_residual) == field.balance_residual


def test_work_preserves_exact_gradient_lost_by_binary64_materialization():
    epsilon = 2.0**-54
    graph = _path(form=(1.0, epsilon, 0.0), capacity=(1, 1, 1))
    field = evaluate_relational_exchange(graph, model=MODEL)
    exact = 1 - Q(epsilon)
    assert Q(field.form_gradient[0]) != exact
    assert field.work.form_gradient[0] == exact
    assert field.work.dissipation[0] == exact**2 / 2
    assert field.work.form_work[0] == exact * Q(field.form_rate[0])
    assert field.work.form_work[0] != Q(field.form_gradient[0]) * Q(field.form_rate[0])
    assert any(value != 0 for value in field.work.form_residual)
    for i, gradient in enumerate(field.work.form_gradient):
        expected = gradient * (
            Q(field.capacity[i]) * field.pressure_split_residual[i]
            + field.nodal_rate_rounding_defect[i]
        )
        assert field.work.form_residual[i] == expected
    assert sum(field.work.balance_residual) == field.balance_residual


def test_form_reversal_preserves_loss_and_reverses_exchange_and_phase_work():
    graph = _path(phase=(-0.25, 0.0, 0.125))
    before = evaluate_relational_exchange(graph, model=MODEL)
    for node in graph:
        graph.nodes[node]["EPI"] *= -1
    after = evaluate_relational_exchange(graph, model=MODEL)
    assert before.storage == after.storage
    assert before.continuous_loss == after.continuous_loss
    assert before.work.dissipation == after.work.dissipation
    assert any(value != 0 for value in before.work.exchange)
    assert after.work.exchange == tuple(-value for value in before.work.exchange)
    assert after.work.phase_work == tuple(-value for value in before.work.phase_work)
    assert after.phase_rate == tuple(-value for value in before.phase_rate)
    for field in (before, after):
        assert (
            sum(field.work.form_work) + sum(field.work.phase_work) == field.storage_rate
        )
        assert sum(field.work.balance_residual) == field.balance_residual


def test_uniform_form_has_zero_initial_work_while_phase_creates_form_contrast():
    graph = _path(form=(0.5,) * 3, phase=(-0.25, 0.0, 0.125))
    field = evaluate_relational_exchange(graph, model=MODEL)
    assert field.form_rate[0] > 0 > field.form_rate[2]
    assert field.phase_rate == (0.0,) * 3
    zero = (Q(0),) * 3
    for name in (
        "form_gradient",
        "dissipation",
        "exchange",
        "form_work",
        "phase_work",
        "form_residual",
        "phase_residual",
        "balance_residual",
    ):
        assert getattr(field.work, name) == zero


@pytest.mark.parametrize("capacity", ((1, 0.5, 2), (0, 0, 0)))
def test_balanced_consensus_and_zero_capacity_have_zero_work(capacity):
    field = evaluate_relational_exchange(
        _path(form=(0.5,) * 3, capacity=capacity), model=MODEL
    )
    assert field.form_rate == field.phase_rate == (0.0,) * 3
    assert field.work.exchange == field.work.dissipation == (Q(0),) * 3
    assert field.storage_rate == field.balance_residual == 0


def test_inactive_node_work_is_available_under_nonzero_form_and_phase_pressure():
    field = evaluate_relational_exchange(
        _path(phase=(-0.25, 0.0, 0.125), capacity=(1, 0, 2)), model=MODEL
    )
    assert field.work.form_gradient[1] != 0
    assert field.pressure[1] != 0
    for name in (
        "dissipation",
        "exchange",
        "form_work",
        "phase_work",
        "form_residual",
        "phase_residual",
        "balance_residual",
    ):
        assert getattr(field.work, name)[1] == 0
    assert sum(field.work.balance_residual) == field.balance_residual


def test_signed_work_is_read_only_and_detached_from_live_state():
    graph = _path(phase=(-0.25, 0.0, 0.125))
    before = _snapshot(graph)
    field = evaluate_relational_exchange(graph, model=MODEL)
    assert _snapshot(graph) == before
    retained = field.work.exchange
    graph.nodes[0]["EPI"] = 50.0
    assert field.work.exchange == retained
    assert field.work.form_gradient == (Q(1, 2), Q(-3, 4), Q(1, 4))
    with pytest.raises(FrozenInstanceError):
        field.work.exchange = (Q(0),) * 3


def test_node_relabeling_preserves_signed_work_with_captured_order():
    graph = _path(phase=(-0.25, 0.0, 0.125))
    mapping = {0: ("left", 1), 1: "middle", 2: 17}
    changed = nx.relabel_nodes(graph, mapping)
    before = evaluate_relational_exchange(graph, model=MODEL)
    after = evaluate_relational_exchange(changed, model=MODEL)
    assert after.nodes == tuple(mapping[node] for node in before.nodes)
    assert after.work == before.work


def test_complementary_regions_have_opposite_cut_and_paired_boundary_rates():
    report = _pattern(_path(), ((2, 0), (1,), (0, 1, 2)))
    outer, middle, whole = report.regions
    assert outer.boundary.cut.region == (2, 0)
    assert outer.boundary.cut.environment == (1,)
    assert outer.boundary.cut.outward_cut_current == Q(3, 4)
    assert middle.boundary.cut.outward_cut_current == Q(-3, 4)
    assert whole.boundary.cut.outward_cut_current == 0
    assert outer.boundary.form_weighted_rate == Q(-3, 8)
    assert outer.boundary.form_boundary_rate == Q(-3, 8)
    assert outer.boundary.form_source_rate == 0
    assert outer.boundary.phase_boundary_rate == Q(3, 16)
    assert float(outer.boundary.phase_weighted_rate) == pytest.approx(3 / 16)
    assert outer.boundary.phase_rate_residual == (
        outer.boundary.phase_weighted_rate - Q(3, 16)
    )
    assert outer.work.dissipation == Q(3, 16)
    assert outer.work.form_work == Q(-3, 16)
    for name in ("form_boundary_rate", "phase_boundary_rate"):
        assert getattr(outer.boundary, name) == -getattr(middle.boundary, name)
        assert getattr(whole.boundary, name) == 0
    for region in report.regions:
        boundary = region.boundary
        assert boundary.weighted_rate_unavailable_reason is None
        assert boundary.form_identity_residual == 0
        assert region.work.balance_residual == (
            region.work.form_residual + region.work.phase_residual
        )
    assert whole.work.dissipation == report.field.continuous_loss
    assert whole.work.form_work + whole.work.phase_work == report.field.storage_rate
    assert whole.work.balance_residual == report.field.balance_residual


def test_actual_regional_rates_retain_independent_sources_and_rounding_defects():
    graph = _path(form=(1.0, 2.0**-54, 0.0), phase=(-0.25, 0.0, 0.125))
    report = _pattern(graph, ((0, 2),))
    field, region = report.field, report.regions[0]
    boundary = region.boundary
    indices = (0, 2)
    assert boundary.form_source_rate == sum(
        (Q(field.phase_source[i]) / 2 for i in indices), Q(0)
    )
    assert boundary.form_pressure_defect_rate == sum(
        (field.pressure_split_residual[i] for i in indices), Q(0)
    )
    assert boundary.form_rounding_defect_rate == sum(
        (field.nodal_rate_rounding_defect[i] / Q(field.capacity[i]) for i in indices),
        Q(0),
    )
    expected_rate = Q(field.form_rate[0]) + Q(field.form_rate[2]) / 2
    assert boundary.form_weighted_rate == expected_rate
    assert boundary.form_identity_residual == 0
    assert expected_rate == (
        boundary.form_boundary_rate
        + boundary.form_source_rate
        + boundary.form_pressure_defect_rate
        + boundary.form_rounding_defect_rate
    )
    expected_phase = sum(
        (
            Q(field.phase_metric[i]) * Q(field.phase_rate[i]) / Q(field.capacity[i])
            for i in indices
        ),
        Q(0),
    )
    assert boundary.phase_weighted_rate == expected_phase
    assert boundary.phase_rate_residual == expected_phase - boundary.phase_boundary_rate
    assert region.work.exchange == field.work.exchange[0] + field.work.exchange[2]


def test_regional_zero_capacity_retains_work_cut_and_undivided_model_terms():
    report = _pattern(_path(capacity=(1, 0, 2)), ((0,), (1,)))
    active, inactive = report.regions
    assert active.transport is None
    assert active.transport_unavailable_reason == "zero_capacity_in_full_support"
    assert active.boundary.weighted_rate_unavailable_reason is None
    assert active.boundary.form_identity_residual == 0
    assert (
        inactive.boundary.weighted_rate_unavailable_reason == "zero_capacity_in_region"
    )
    for name in (
        "form_weighted_rate",
        "form_rounding_defect_rate",
        "form_identity_residual",
        "phase_weighted_rate",
        "phase_rate_residual",
    ):
        assert getattr(inactive.boundary, name) is None
    assert inactive.boundary.cut.outward_cut_current == Q(-3, 4)
    assert inactive.boundary.form_boundary_rate == Q(3, 8)
    assert inactive.boundary.phase_boundary_rate == Q(-3, 16)
    assert inactive.work.dissipation == inactive.work.exchange == 0
    assert inactive.work.form_work == inactive.work.phase_work == 0


def test_lossless_and_full_support_ports_do_not_inherit_transport_restrictions():
    model = RelationalExchangeModel(2.0, epi_weight=0.0, phase_weight=1.0)
    report = _pattern(_path(phase=(-0.25, 0.0, 0.125)), ((0,), (0, 1, 2)), model=model)
    for region in report.regions:
        assert region.transport is None
        assert region.transport_unavailable_reason == "zero_epi_weight"
        assert region.boundary.weighted_rate_unavailable_reason is None
        assert region.boundary.form_boundary_rate == 0
        assert region.boundary.form_identity_residual == 0
        assert region.work.dissipation == 0
    assert report.regions[0].boundary.phase_boundary_rate == Q(1, 4)
    assert report.regions[1].boundary.cut.outward_cut_current == 0
    assert report.regions[1].boundary.phase_boundary_rate == 0


def test_overlapping_regional_work_is_not_misreported_as_a_partition():
    graph = _path()
    before = _snapshot(graph)
    report = _pattern(graph, ((0, 1), (1, 2)))
    assert _snapshot(graph) == before
    first, second = report.regions
    assert first.work.dissipation + second.work.dissipation == (
        report.field.continuous_loss + report.field.work.dissipation[1]
    )
    assert first.work.form_work + second.work.form_work != report.field.storage_rate
    with pytest.raises(FrozenInstanceError):
        first.work.exchange = Q(0)
    with pytest.raises(FrozenInstanceError):
        first.boundary.form_source_rate = Q(0)
