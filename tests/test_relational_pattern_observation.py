"""Routine contracts for detached relational pattern geometry and accounting.

Analytic small states and selected retained checkpoints exercise the shared
observer without executing a trajectory or regenerating scientific records.
"""

import json
import math
import pickle
from dataclasses import FrozenInstanceError
from fractions import Fraction as Q
from pathlib import Path

import networkx as nx
import pytest

from tnfr.dynamics.relational import RelationalExchangeModel
from tnfr.physics import relational_observations as observations
from tnfr.physics.relational_observations import observe_relational_pattern

ROOT = Path(__file__).resolve().parents[1]
MODEL = RelationalExchangeModel(2.0)


def _path():
    graph = nx.path_graph(3)
    graph.graph.update(
        GAMMA={"type": "none"},
        DNFR_WEIGHTS={"epi": 0.1, "phase": 0.2, "vf": 0.3, "topo": 0.4},
        _t=2.0,
        preserved={"payload": [1, 2]},
    )
    for node, form, phase, capacity in zip(
        graph,
        (0.75, 0.25, 0.5),
        (-math.pi / 6, 0.0, math.pi / 6),
        (1.0, 1.0, 2.0),
        strict=True,
    ):
        graph.nodes[node].update(EPI=form, theta=phase, nu_f=capacity, delta_nfr=999.0)
    return graph


def _snapshot(graph):
    return pickle.dumps(
        (graph.graph, tuple(graph.nodes(data=True)), tuple(graph.edges(data=True))),
        protocol=5,
    )


def test_fresh_field_and_detached_evidence_leave_live_pressure_and_state_untouched():
    graph = _path()
    before = _snapshot(graph)
    reference = dict.fromkeys(graph, 0.0)
    regions = [[2, 0], [1]]
    report = observe_relational_pattern(
        graph, model=MODEL, reference_phase=reference, regions=regions
    )
    assert _snapshot(graph) == before
    assert report.field.pressure == pytest.approx((-1 / 6, 3 / 16, -5 / 24), abs=1e-15)
    assert all(graph.nodes[node]["delta_nfr"] == 999.0 for node in graph)
    assert report.regions[0].transport.source.stored_pressure == tuple(
        map(Q, report.field.pressure)
    )
    reference[0] = 10.0
    regions[0].append(1)
    graph.nodes[0]["EPI"] = 12.0
    assert report.reference_phase == (0.0, 0.0, 0.0)
    assert report.regions[0].nodes == (2, 0)
    assert report.field.epi == (0.75, 0.25, 0.5)
    with pytest.raises(FrozenInstanceError):
        report.regions[0].form_mean = Q(0)


def test_exact_unweighted_geometry_is_distinct_from_transport_weighted_mean():
    graph = _path()
    report = observe_relational_pattern(
        graph,
        model=MODEL,
        reference_phase=dict.fromkeys(graph, 0.0),
        regions=((2, 0), (1,)),
    )
    outer, middle = report.regions
    phase = Q(math.pi / 6)
    assert outer.form_mean == Q(5, 8)
    assert outer.centered_form == (Q(-1, 8), Q(1, 8))
    assert outer.form_norm_squared == Q(1, 32)
    assert outer.phase_error_mean == 0
    assert outer.centered_phase_error == (phase, -phase)
    assert outer.phase_norm_squared == 2 * phase**2
    assert middle.form_mean == Q(1, 4)
    assert middle.centered_form == middle.centered_phase_error == (Q(0),)
    assert middle.form_norm_squared == middle.phase_norm_squared == 0
    assert outer.transport.mean == Q(2, 3)
    assert outer.transport.mean != outer.form_mean


def test_noncomparable_labels_mapping_order_and_region_order_are_preserved():
    graph = nx.Graph()
    nodes = ("middle", ("tuple", 0), 7)
    graph.add_nodes_from(nodes)
    graph.add_edges_from(((nodes[0], nodes[1]), (nodes[0], nodes[2])))
    for i, node in enumerate(nodes):
        graph.nodes[node].update(EPI=i / 4, theta=i / 8, nu_f=1.0)
    reference = {7: 0.25, ("tuple", 0): 0.125, "middle": 0.0}
    regions = ((7, ("tuple", 0)), ("middle",))
    report = observe_relational_pattern(
        graph, model=MODEL, reference_phase=reference, regions=regions
    )
    assert report.field.nodes == nodes
    assert report.reference_phase == (0.0, 0.125, 0.25)
    assert tuple(region.nodes for region in report.regions) == regions
    assert report.regions[0].form_mean == Q(3, 8)
    assert report.regions[0].centered_form == (Q(1, 8), Q(-1, 8))
    assert report.regions[0].phase_norm_squared == 0


def test_overlapping_regions_reuse_one_fresh_field_and_one_transport_capture(
    monkeypatch,
):
    field_owner = observations.evaluate_relational_exchange
    transport_owner = observations.observe_support_transport
    calls = {"field": 0, "transport": 0}

    def field(*args, **kwargs):
        calls["field"] += 1
        return field_owner(*args, **kwargs)

    def transport(*args, **kwargs):
        calls["transport"] += 1
        return transport_owner(*args, **kwargs)

    monkeypatch.setattr(observations, "evaluate_relational_exchange", field)
    monkeypatch.setattr(observations, "observe_support_transport", transport)
    graph = _path()
    regions = ((0, 1), (1, 2), (1,))
    report = observe_relational_pattern(
        graph, model=MODEL, reference_phase=dict.fromkeys(graph, 0.0), regions=regions
    )
    assert calls == {"field": 1, "transport": 1}
    assert tuple(region.nodes for region in report.regions) == regions
    assert all(region.transport is not None for region in report.regions)


def test_absent_cycle_is_unavailable_without_discarding_region_geometry():
    graph = _path()
    report = observe_relational_pattern(
        graph,
        model=MODEL,
        reference_phase=dict.fromkeys(graph, 0.0),
        regions=((0, 1),),
        cycles=((0, 1, 2),),
    )
    assert not report.winding[0].is_defined
    assert report.winding[0].winding is None
    assert report.regions[0].form_norm_squared > 0


def test_winding_reference_uses_real_lifts_without_a_semicircle_restriction():
    graph = nx.cycle_graph(5)
    reference = {node: math.tau * node / 5 for node in graph}
    for node in graph:
        graph.nodes[node].update(EPI=0.0, theta=reference[node], nu_f=1.0)
    report = observe_relational_pattern(
        graph,
        model=MODEL,
        reference_phase=reference,
        regions=(tuple(graph),),
        cycles=(tuple(graph),),
    )
    assert report.regions[0].phase_norm_squared == 0
    assert report.winding[0].is_defined and report.winding[0].winding == 1
    # A supplied lift is evidence, not a silently wrapped phase difference.
    changed_reference = dict(reference)
    changed_reference[0] += math.tau
    changed = observe_relational_pattern(
        graph,
        model=MODEL,
        reference_phase=changed_reference,
        regions=(tuple(graph),),
        cycles=(tuple(graph),),
    )
    assert changed.winding[0].winding == 1
    assert changed.regions[0].phase_error_mean == -Q(math.tau) / 5
    assert changed.regions[0].phase_norm_squared == Q(4, 5) * Q(math.tau) ** 2


@pytest.mark.parametrize(
    "boundary,reason",
    (
        ("zero_capacity", "zero_capacity_in_full_support"),
        ("lossless", "zero_epi_weight"),
        ("whole", "full_support_region"),
    ),
)
def test_transport_unavailability_does_not_discard_admitted_geometry(boundary, reason):
    graph = _path()
    model = MODEL
    regions = ((0, 1),)
    if boundary == "zero_capacity":
        graph.nodes[2][
            "nu_f"
        ] = 0.0  # Outside the observed region still affects H admission.
    elif boundary == "lossless":
        model = RelationalExchangeModel(2.0, epi_weight=0.0, phase_weight=1.0)
    else:
        regions = ((0, 1, 2),)
    report = observe_relational_pattern(
        graph, model=model, reference_phase=dict.fromkeys(graph, 0.0), regions=regions
    )
    region = report.regions[0]
    assert region.transport is None
    assert region.transport_unavailable_reason == reason
    assert region.form_norm_squared > 0
    assert region.phase_norm_squared > 0


@pytest.mark.parametrize(
    "reference",
    (
        {0: 0.0, 1: 0.0},
        {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0},
        {0: True, 1: 0.0, 2: 0.0},
        {0: Q(1, 2**2000), 1: 0.0, 2: 0.0},
    ),
)
def test_reference_support_and_raw_scalar_admission_precede_live_mutation(reference):
    graph = _path()
    before = _snapshot(graph)
    with pytest.raises((TypeError, ValueError)):
        observe_relational_pattern(
            graph, model=MODEL, reference_phase=reference, regions=((0, 1),)
        )
    assert _snapshot(graph) == before


@pytest.mark.parametrize("regions", (((0, 0),), ((0, 9),), ({0, 1},), {(0, 1)}, ()))
def test_malformed_region_orders_reject_without_mutation(regions):
    graph = _path()
    before = _snapshot(graph)
    with pytest.raises((TypeError, ValueError)):
        observe_relational_pattern(
            graph,
            model=MODEL,
            reference_phase=dict.fromkeys(graph, 0.0),
            regions=regions,
        )
    assert _snapshot(graph) == before


def test_transport_forcing_uses_independent_phase_source_and_retains_pressure_defect():
    graph = _path()
    report = observe_relational_pattern(
        graph, model=MODEL, reference_phase=dict.fromkeys(graph, 0.0), regions=((2, 0),)
    )
    field, balance = report.field, report.regions[0].transport
    expected_forcing = tuple(Q(1, 2) * Q(value) for value in field.phase_source)
    assert balance.forcing == expected_forcing
    q = (Q(1, 2), Q(-3, 4), Q(1, 4))
    degrees = (1, 2, 1)
    expected_pressure = tuple(
        -Q(1, 2) * value / degree + source
        for value, degree, source in zip(q, degrees, expected_forcing, strict=True)
    )
    expected_defect = tuple(
        Q(value) - expected
        for value, expected in zip(field.pressure, expected_pressure, strict=True)
    )
    assert balance.stored_pressure_defect == expected_defect
    assert balance.mass_defect_rate == expected_defect[2] + expected_defect[0]
    assert balance.mass_boundary_rate == -Q(3, 8)
    assert balance.mass_identity_residual == balance.variance_identity_residual == 0


def _exact(item):
    if set(item) == {"numerator", "denominator"}:
        return Q(item["numerator"], item["denominator"])
    return item


def _retained(name):
    path = ROOT / "docs/assets" / name / "result.json"
    return json.loads(path.read_text(encoding="utf-8"), object_hook=_exact)


def _checkpoint_graph(prediction, saved, nodes):
    graph = nx.Graph()
    graph.add_nodes_from(nodes)
    graph.add_edges_from((i, j, {"weight": 1.0}) for i, j in prediction["edges"])
    graph.graph.update(GAMMA={"type": "none"}, vectorized_dnfr=True)
    for i, node in enumerate(nodes):
        graph.nodes[node].update(
            EPI=saved["epi"][i],
            theta=saved["phase"][i],
            nu_f=saved["capacity"][i],
            delta_nfr=999.0,
        )
    return graph


def test_selected_recovery_checkpoint_preserves_separate_norms_and_means():
    retained = _retained("relational_local_recovery")
    prediction = retained["prediction"]
    saved = retained["traces"]["4096"]["checkpoints"]["4096"]
    nodes = tuple(prediction["nodes"])
    graph = _checkpoint_graph(prediction, saved, nodes)
    reference = dict(zip(nodes, prediction["reference_phase"], strict=True))
    observed = observe_relational_pattern(
        graph,
        model=RelationalExchangeModel(1.0),
        reference_phase=reference,
        regions=(nodes,),
        cycles=(nodes,),
    )
    region = observed.regions[0]
    expected_form_mean = sum(map(Q, saved["epi"])) / 5
    expected_phase_mean = (
        sum(
            Q(a) - Q(b)
            for a, b in zip(saved["phase"], prediction["reference_phase"], strict=True)
        )
        / 5
    )
    assert region.form_mean == expected_form_mean
    assert region.phase_error_mean == expected_phase_mean
    assert float(region.form_mean) == pytest.approx(saved["form_mean"], abs=1e-15)
    assert float(region.phase_error_mean) == pytest.approx(
        saved["phase_error_mean"], abs=1e-15
    )
    assert float(region.form_norm_squared + region.phase_norm_squared) == pytest.approx(
        saved["quotient_distance"] ** 2, abs=1e-16
    )
    assert observed.winding[0].winding == 1
    assert observed.field.pressure == pytest.approx(saved["pressure"], abs=1e-12)


def test_selected_interaction_checkpoint_reuses_geometry_and_cut_balance():
    retained = _retained("relational_region_interaction")
    prediction = retained["prediction"]
    saved = retained["traces"]["joined"]["256"]["checkpoints"]["256"]
    nodes = tuple(saved["nodes"])
    graph = _checkpoint_graph(prediction, saved, nodes)
    reference = dict(zip(nodes, prediction["initial_phase"], strict=True))
    regions = (tuple(range(5)), tuple(range(5, 10)))
    observed = observe_relational_pattern(
        graph,
        model=RelationalExchangeModel(1.0),
        reference_phase=reference,
        regions=regions,
        cycles=regions,
    )
    for name, region in zip(("left", "right"), observed.regions, strict=True):
        previous = saved["regions"][name]
        assert region.form_mean == previous["form_mean"]
        assert region.phase_error_mean == previous["phase_error_mean"]
        assert region.centered_form + region.centered_phase_error == tuple(
            previous["centered_state"]
        )
        assert (
            region.transport.mass_identity_residual
            == region.transport.variance_identity_residual
            == 0
        )
        for key, value in saved["regional_transport"][name].items():
            assert float(getattr(region.transport, key)) == pytest.approx(
                float(value), abs=1e-12
            )
        boundary, field = region.boundary, observed.field
        assert boundary.cut == region.transport.cut
        assert boundary.form_identity_residual == 0
        assert boundary.form_weighted_rate == (
            region.transport.stored_mass_rate + boundary.form_rounding_defect_rate
        )
        assert boundary.form_boundary_rate == region.transport.mass_boundary_rate
        indices = tuple(field.nodes.index(node) for node in region.nodes)
        actual_phase_rate = sum(
            (
                Q(field.phase_metric[i]) * Q(field.phase_rate[i]) / Q(field.capacity[i])
                for i in indices
            ),
            Q(0),
        )
        assert boundary.phase_weighted_rate == actual_phase_rate
        assert boundary.phase_rate_residual == (
            actual_phase_rate - boundary.phase_boundary_rate
        )
        assert region.work.exchange == sum(
            (field.work.exchange[i] for i in indices), Q(0)
        )
    left, right = (region.boundary for region in observed.regions)
    assert left.cut.outward_cut_current == -right.cut.outward_cut_current
    assert left.phase_boundary_rate == -right.phase_boundary_rate
    assert left.form_boundary_rate == -right.form_boundary_rate
    assert sum(region.work.dissipation for region in observed.regions) == (
        observed.field.continuous_loss
    )
    assert (
        sum(
            region.work.form_work + region.work.phase_work
            for region in observed.regions
        )
        == observed.field.storage_rate
    )
    assert sum(region.work.balance_residual for region in observed.regions) == (
        observed.field.balance_residual
    )
    assert tuple(item.winding for item in observed.winding) == (1, 1)
    assert observed.field.pressure == pytest.approx(saved["pressure"], abs=1e-12)
