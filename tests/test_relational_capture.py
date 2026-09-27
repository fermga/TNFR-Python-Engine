"""Ideal-law basin admission from actual detached fields, without trajectories."""

import json
import math
import pickle
from dataclasses import replace
from decimal import Decimal, localcontext
from fractions import Fraction as Q
from pathlib import Path

import networkx as nx
import pytest

from tnfr.dynamics.relational import RelationalExchangeModel
from tnfr.mathematics._phase_midpoint import _affine_interval, _pi_bounds
from tnfr.physics.relational_capture import (
    certify_relational_capture,
    certify_relational_local_capture,
    certify_relational_sector_capture,
)
from tnfr.sdk import Network, relational_report_to_dict

CYCLES = (tuple(range(5)), tuple(range(5, 10)))


def _graph(
    *, a=2.25, b=1.125, amplitude=0.0, contrast=0.0, form_offset=0.0, center=0.0
):
    graph = nx.disjoint_union(nx.cycle_graph(5), nx.cycle_graph(5))
    graph.add_edges_from(((0, 5), (1, 6)))
    form = (amplitude, -amplitude, -contrast, 0.0, contrast)
    phase = (a, -a, -b, 0.0, b)
    for node in graph:
        graph.nodes[node].update(
            EPI=form_offset + form[node % 5],
            theta=center + phase[node % 5],
            nu_f=1.0,
            delta_nfr=999.0,
        )
    graph.graph.update(GAMMA={"type": "none"}, _t=0.0)
    return graph


def _model(**kwargs):
    return RelationalExchangeModel(phase_domain="positive_resultant", **kwargs)


def _capture(graph, **kwargs):
    return certify_relational_capture(
        graph, model=_model(storage_scale=1.0, **kwargs), cycles=CYCLES
    )


def _state(graph):
    return pickle.dumps(
        (graph.graph, tuple(graph.nodes(data=True)), tuple(graph.edges(data=True)))
    )


@pytest.mark.parametrize("sign", (1, -1))
def test_nonacute_zero_form_geometry_has_certified_twist_capture(sign):
    graph = _graph(a=sign * 2.25, b=sign * 1.125)
    before = _state(graph)
    certificate = _capture(graph)
    assert _state(graph) == before
    assert certificate.admitted and certificate.target_sector == sign
    assert certificate.coordinates == (0, 0, Q(sign * 2.25), Q(sign * 1.125))
    assert all(row.winding == sign for row in certificate.winding)
    assert all(lower > 0 for lower, _ in certificate.rectangle_margin_bounds)
    assert 0 < certificate.storage_bounds[0] <= certificate.storage_bounds[1] < 7
    assert certificate.field.form_storage == 0
    # It lies outside the old acute chart and is not already an equilibrium.
    phase = certificate.field.phase
    assert abs(math.remainder(phase[1] - phase[0], math.tau)) > math.pi / 2
    assert any(value != 0 for value in certificate.field.form_rate)
    assert certificate.field.phase_rate == (0.0,) * 10


def test_current_winding_does_not_replace_the_predicted_consensus_basin():
    certificate = _capture(_graph(a=2.0, b=1.0))
    assert certificate.admitted
    assert certificate.rectangle_kind == "consensus"
    assert certificate.target_sector == 0
    assert tuple(row.winding for row in certificate.winding) == (1, 1)
    assert certificate.storage_bounds[1] < 7


def test_shared_rectangle_ledger_preserves_exact_point_margin_arithmetic():
    certificate = _capture(_graph(a=2.25, b=1.125))
    a, b = certificate.coordinates[2:]
    pi = _pi_bounds()

    def affine(rational, coefficient):
        return _affine_interval(rational, coefficient, pi)

    assert certificate.candidate_rectangle_margin_bounds == (
        (1, (affine(a, -Q(2, 3)), affine(-a, 1), (b, b), affine(-b, Q(1, 2)))),
        (
            0,
            (
                affine(-a, Q(2, 3)),
                affine(a, Q(2, 3)),
                affine(-b, Q(1, 2)),
                affine(b, Q(1, 2)),
            ),
        ),
        (-1, (affine(-a, -Q(2, 3)), affine(a, 1), (-b, -b), affine(b, Q(1, 2)))),
    )


def test_consensus_storage_is_exact_and_strict_energy_boundary_is_not_tolerated():
    model = _model(storage_scale=0.5)
    graph = _graph(a=0.0, b=0.0, amplitude=0.5, contrast=1.0)
    certificate = certify_relational_capture(graph, model=model, cycles=CYCLES)
    assert certificate.exact_symmetry and certificate.rectangle_admitted
    assert certificate.phase_storage_bounds == (0, 0)
    assert certificate.storage_bounds == (Q(7, 2), Q(7, 2))
    assert certificate.capture_threshold == Q(7, 2)
    assert not certificate.admitted and certificate.target_sector is None
    assert certificate.unavailable_reasons == ("strict_storage_sublevel_not_certified",)
    graph = _graph(a=0.0, b=0.0, amplitude=0.5, contrast=0.5)
    inside = certify_relational_capture(graph, model=model, cycles=CYCLES)
    assert inside.storage_bounds == (Q(3, 2), Q(3, 2))
    assert inside.admitted and inside.target_sector == 0


def test_common_offsets_are_retained_and_do_not_change_the_certificate():
    parameters = dict(a=2.5, b=1.25, amplitude=1 / 32, contrast=1 / 16)
    original = _capture(_graph(**parameters))
    shifted = _capture(_graph(**parameters, form_offset=8.0, center=0.125))
    assert original.admitted and shifted.admitted
    assert shifted.form_offset == 8 and shifted.phase_center == Q(1, 8)
    assert shifted.coordinates == original.coordinates
    assert shifted.storage_bounds == original.storage_bounds
    assert shifted.rectangle_margin_bounds == original.rectangle_margin_bounds


@pytest.mark.parametrize("attribute", ("EPI", "theta"))
def test_nonzero_symmetry_defect_is_retained_without_projection(attribute):
    graph = _graph()
    graph.nodes[0][attribute] += 2**-40
    before = _state(graph)
    certificate = _capture(graph)
    assert _state(graph) == before
    assert not certificate.exact_symmetry and not certificate.admitted
    assert certificate.target_sector is None
    assert "exact_copy_reflection_not_satisfied" in certificate.unavailable_reasons
    assert any(
        (
            *certificate.copy_form_defects,
            *certificate.copy_phase_defects,
            *certificate.reflection_form_defects,
            *certificate.reflection_phase_defects,
        )
    )


@pytest.mark.parametrize("capacity", (0.0, 0.5))
def test_other_capacity_laws_are_unavailable_even_when_snapshot_is_admitted(capacity):
    graph = _graph()
    for node in graph:
        graph.nodes[node]["nu_f"] = capacity
    certificate = _capture(graph)
    assert not certificate.admitted
    assert "unit_capacity_required" in certificate.unavailable_reasons


def test_lossless_model_cannot_use_the_dissipative_capture_theorem():
    certificate = _capture(_graph(), epi_weight=0, phase_weight=1)
    assert not certificate.admitted
    assert "positive_epi_weight_required" in certificate.unavailable_reasons


def test_high_form_storage_does_not_get_a_capture_target_from_geometry_alone():
    certificate = _capture(_graph(amplitude=1.0))
    assert certificate.exact_symmetry and certificate.rectangle_admitted
    assert certificate.storage_bounds[0] > certificate.capture_threshold
    assert not certificate.admitted and certificate.target_sector is None


def test_chosen_phase_lifts_are_not_silently_replaced_to_enter_a_rectangle():
    certificate = _capture(_graph(a=2.25 + 2 * math.tau))
    assert certificate.exact_symmetry and certificate.energy_admitted
    assert not certificate.rectangle_admitted and not certificate.admitted
    assert certificate.target_sector is None
    assert "strict_capture_rectangle_not_certified" in certificate.unavailable_reasons


@pytest.mark.parametrize("extra_edge", ((2, 7), (0, 2)))
def test_additional_support_rejects_instead_of_reusing_the_wrong_theorem(extra_edge):
    graph = _graph(a=0.0, b=0.0)
    graph.add_edge(*extra_edge)
    before = _state(graph)
    with pytest.raises(ValueError, match="support"):
        _capture(graph)
    assert _state(graph) == before


@pytest.mark.parametrize("cycles", ((), (tuple(range(5)),), (tuple(range(5)),) * 2))
def test_malformed_cycle_covers_reject(cycles):
    with pytest.raises(ValueError):
        certify_relational_capture(
            _graph(), model=_model(storage_scale=1), cycles=cycles
        )


def test_node_relabeling_and_iteration_order_do_not_select_a_different_basin():
    graph = _graph()
    labels = {node: ("n", 10 - node) for node in graph}
    relabeled = nx.relabel_nodes(graph, labels)
    reordered = nx.Graph()
    reordered.graph.update(relabeled.graph)
    reordered.add_nodes_from(reversed(tuple(relabeled.nodes(data=True))))
    reordered.add_edges_from(reversed(tuple(relabeled.edges(data=True))))
    cycles = tuple(tuple(labels[node] for node in row) for row in CYCLES)
    certificate = certify_relational_capture(
        reordered, model=_model(storage_scale=1), cycles=cycles
    )
    assert certificate.admitted and certificate.target_sector == 1
    assert certificate.storage_bounds == _capture(graph).storage_bounds


def test_sdk_uses_one_fresh_field_and_export_preserves_exact_bounds(monkeypatch):
    from tnfr.physics import relational_capture as owner

    calls = []
    evaluate = owner.evaluate_relational_exchange

    def observed(*args, **kwargs):
        calls.append(1)
        return evaluate(*args, **kwargs)

    monkeypatch.setattr(owner, "evaluate_relational_exchange", observed)
    network = Network(_graph())
    before = _state(network.G)
    certificate = network.relational_capture(_model(storage_scale=1), cycles=CYCLES)
    assert calls == [1] and _state(network.G) == before
    output = relational_report_to_dict(certificate)
    assert output["report_type"] == "RelationalCaptureCertificate"
    upper = certificate.storage_bounds[1]
    assert output["report"]["storage_bounds"][1] == {
        "numerator": upper.numerator,
        "denominator": upper.denominator,
    }
    output["report"]["field"]["epi"][0] = 123.0
    assert certificate.field.epi[0] == 0 and _state(network.G) == before


def _local_graph(sector=1):
    return _graph(a=sector * 4 * math.pi / 5, b=sector * 2 * math.pi / 5)


def _local_capture(graph, *, sector=1, model=None):
    return certify_relational_local_capture(
        graph,
        model=_model(storage_scale=1) if model is None else model,
        cycles=CYCLES,
        target_sector=sector,
    )


@pytest.mark.parametrize("sector", (-1, 0, 1))
def test_full_state_local_capture_keeps_asymmetric_coordinates_and_fresh_field(sector):
    graph = _local_graph(sector)
    graph.nodes[0]["EPI"] = 1 / 1024
    graph.nodes[7]["theta"] += 1 / 8192
    before = _state(graph)
    certificate = _local_capture(graph, sector=sector)
    assert certificate.admitted and certificate.target_sector == sector
    assert _state(graph) == before
    assert all(graph.nodes[node]["delta_nfr"] == 999 for node in graph)
    assert max(map(abs, certificate.field.pressure)) < 999
    assert certificate.centered_form == (Q(9, 10240),) + (Q(-1, 10240),) * 9
    assert certificate.form_norm_squared == Q(9, 10 * 1024**2)
    assert certificate.field.form_storage == Q(3, 2 * 1024**2)
    phase_mean = sum(Q(graph.nodes[node]["theta"]) for node in graph) / 10
    assert certificate.phase_mean == phase_mean
    expected_reference = tuple(
        sector * coefficient
        for coefficient in (Q(4, 5), Q(-4, 5), Q(-2, 5), Q(0), Q(2, 5)) * 2
    )
    assert certificate.reference_phase_pi_coefficients == expected_reference
    assert certificate.phase_error_affine == tuple(
        (Q(graph.nodes[node]["theta"]) - phase_mean, -expected_reference[node])
        for node in graph
    )
    assert certificate.quotient_squared_bounds[1] < Q(9, 800)
    assert certificate.excess_storage_bounds[1] < Q(1, 100000)
    assert tuple(item.winding for item in certificate.winding) == (sector, sector)
    assert not _capture(graph).exact_symmetry


def test_exact_ideal_reference_energy_and_zero_straddling_enclosure_are_retained():
    certificate = _local_capture(_local_graph())
    with localcontext() as context:
        context.prec = 80
        # cos(2*pi/5)=(sqrt(5)-1)/4 is independent of the cosine-series owner.
        ideal_energy = Q((Decimal(25) - 5 * Decimal(5).sqrt()) / 2)
    lower, upper = certificate.reference_phase_storage_bounds
    assert lower < ideal_energy < upper
    assert (
        certificate.excess_storage_bounds[0] < 0 < certificate.excess_storage_bounds[1]
    )
    assert certificate.admitted
    assert certificate.phase_norm_squared_bounds[1] < Q(1, 10**27)
    assert certificate.centered_form == (0,) * 10


def test_full_state_common_offsets_preserve_exact_quotient_and_storage_bounds():
    graph = _local_graph(0)
    graph.nodes[2]["EPI"] = 1 / 1024
    graph.nodes[7]["theta"] = 1 / 2048
    original = _local_capture(graph, sector=0)
    for node in graph:
        graph.nodes[node]["EPI"] += 4
        graph.nodes[node]["theta"] += 2
    shifted = _local_capture(graph, sector=0)
    assert original.admitted and shifted.admitted
    assert shifted.form_mean - original.form_mean == 4
    assert shifted.phase_mean - original.phase_mean == 2
    assert shifted.centered_form == original.centered_form
    assert shifted.phase_error_affine == original.phase_error_affine
    assert shifted.quotient_squared_bounds == original.quotient_squared_bounds
    assert shifted.storage_bounds == original.storage_bounds


def test_local_energy_boundary_uses_exact_form_storage_without_a_tolerance():
    threshold = Q(1, 100000)
    root = math.sqrt(float(threshold))
    below, above = math.nextafter(root, 0.0), math.nextafter(root, math.inf)
    assert Q(below) ** 2 < threshold < Q(above) ** 2
    for value, admitted in ((below, True), (above, False)):
        graph = _local_graph(0)
        graph.nodes[2]["EPI"] = value  # Degree two gives E_D=value**2 exactly.
        certificate = _local_capture(graph, sector=0)
        assert certificate.radius_admitted
        assert certificate.storage_bounds == (Q(value) ** 2,) * 2
        assert certificate.excess_storage_bounds == certificate.storage_bounds
        assert certificate.admitted is admitted
        assert certificate.energy_admitted is admitted
    assert certificate.unavailable_reasons == ("strict_local_energy_not_certified",)
    assert certificate.target_sector is None


def test_low_total_energy_does_not_admit_the_wrong_local_target():
    certificate = _local_capture(_local_graph(0), sector=1)
    assert certificate.energy_admitted
    assert certificate.excess_storage_bounds[1] < 0
    assert not certificate.radius_admitted and not certificate.admitted
    assert certificate.target_sector is None
    assert certificate.unavailable_reasons == ("strict_local_radius_not_certified",)


def test_local_certificate_does_not_rewrap_individual_phase_lifts():
    graph = _local_graph()
    graph.nodes[0]["theta"] += math.tau
    before = _state(graph)
    certificate = _local_capture(graph)
    assert certificate.energy_admitted
    assert not certificate.radius_admitted and not certificate.admitted
    assert _state(graph) == before
    assert certificate.field.phase[0] == graph.nodes[0]["theta"]


@pytest.mark.parametrize("premise", ("capacity", "storage_scale", "dissipation"))
def test_local_certificate_retains_unsupported_model_premises(premise):
    graph, model = _local_graph(0), _model(storage_scale=1)
    if premise == "capacity":
        graph.nodes[0]["nu_f"] = 0
        reason = "unit_capacity_required"
    elif premise == "storage_scale":
        model = _model(storage_scale=2)
        reason = "unit_storage_scale_required"
    else:
        model = _model(storage_scale=1, epi_weight=0, phase_weight=1)
        reason = "positive_epi_weight_required"
    certificate = _local_capture(graph, sector=0, model=model)
    assert certificate.radius_admitted and certificate.energy_admitted
    assert not certificate.admitted and certificate.target_sector is None
    assert certificate.unavailable_reasons == (reason,)


@pytest.mark.parametrize(
    "sector, error", ((True, TypeError), (1.0, TypeError), (2, ValueError))
)
def test_local_target_is_a_declared_integer_not_a_coerced_scalar(sector, error):
    with pytest.raises(error, match="target_sector"):
        _local_capture(_local_graph(), sector=sector)


def test_local_node_order_and_labels_only_reorder_the_exact_reference():
    graph = _local_graph()
    graph.nodes[7]["theta"] += 1 / 8192
    original = _local_capture(graph)
    labels = {node: ("node", 10 - node) for node in graph}
    relabeled = nx.relabel_nodes(graph, labels)
    reordered = nx.Graph()
    reordered.graph.update(relabeled.graph)
    reordered.add_nodes_from(reversed(tuple(relabeled.nodes(data=True))))
    reordered.add_edges_from(reversed(tuple(relabeled.edges(data=True))))
    cycles = tuple(tuple(labels[node] for node in row) for row in CYCLES)
    certificate = certify_relational_local_capture(
        reordered, model=_model(storage_scale=1), cycles=cycles
    )
    assert certificate.admitted
    assert certificate.reference_phase_pi_coefficients == (
        original.reference_phase_pi_coefficients[::-1]
    )
    assert certificate.phase_error_affine == original.phase_error_affine[::-1]
    assert certificate.quotient_squared_bounds == original.quotient_squared_bounds
    assert certificate.storage_bounds == original.storage_bounds
    reordered.add_edge(labels[2], labels[7])
    with pytest.raises(ValueError, match="support"):
        certify_relational_local_capture(
            reordered, model=_model(storage_scale=1), cycles=cycles
        )


def _sector_capture(graph, *, sector=1, model=None):
    return certify_relational_sector_capture(
        graph,
        model=_model(storage_scale=1) if model is None else model,
        cycles=CYCLES,
        target_sector=sector,
    )


@pytest.mark.parametrize("sector", (-1, 1))
def test_acute_sector_admits_asymmetric_states_outside_the_old_local_energy_gate(
    sector,
):
    graph = _local_graph(sector)
    graph.nodes[0]["EPI"] = 1 / 16
    graph.nodes[7]["theta"] += 1 / 64
    before = _state(graph)
    certificate = _sector_capture(graph, sector=sector)
    assert certificate.admitted and certificate.target_sector == sector
    assert certificate.ring_windings == (sector, sector)
    assert certificate.bridge_winding == 0
    assert certificate.bridge_cycle == (0, 1, 6, 5)
    assert certificate.field.form_storage == Q(3, 512)
    assert certificate.storage_margin_lower_bound > 0
    assert _state(graph) == before
    assert not _capture(graph).exact_symmetry
    assert not _local_capture(graph, sector=sector).energy_admitted
    oriented = {}
    for edge, (raw, coefficient), turn, margins in zip(
        certificate.field.edges,
        certificate.edge_gap_affine,
        certificate.edge_turn_candidates,
        certificate.edge_acute_margin_bounds,
    ):
        left, right = edge
        assert raw == Q(graph.nodes[right]["theta"]) - Q(graph.nodes[left]["theta"])
        assert coefficient == 2 * turn
        assert all(lower > 0 for lower, _ in margins)
        oriented[left, right], oriented[right, left] = turn, -turn
    for cycle in CYCLES:
        assert (
            sum(
                oriented[left, right]
                for left, right in zip(cycle, cycle[1:] + cycle[:1])
            )
            == sector
        )


def test_sector_barrier_matches_an_independent_algebraic_value():
    certificate = _sector_capture(_local_graph())
    with localcontext() as context:
        context.prec = 80
        # Independent closed values for cos(2*pi/5) and cos(3*pi/8).
        barrier = Q(
            10
            - 5 * (Decimal(5).sqrt() - 1) / 4
            - 2 * (Decimal(2) - Decimal(2).sqrt()).sqrt()
        )
    lower, upper = certificate.capture_barrier_bounds
    assert Q(6924179, 1000000) < lower < barrier < upper
    assert certificate.storage_bounds[1] < lower
    assert certificate.energy_admitted


def test_exact_sector_admission_does_not_consume_floating_winding_verdicts(monkeypatch):
    from tnfr.physics import relational_capture as owner

    original = owner.certify_phase_winding

    def unavailable(*args, **kwargs):
        return replace(
            original(*args, **kwargs),
            status="undefined",
            winding=None,
            reason="independently unavailable telemetry control",
        )

    monkeypatch.setattr(owner, "certify_phase_winding", unavailable)
    certificate = _sector_capture(_local_graph())
    assert certificate.admitted and certificate.ring_windings == (1, 1)
    assert all(not row.is_defined for row in certificate.winding)


def test_sector_accepts_certified_large_full_turn_representatives_without_rewriting():
    graph = _local_graph()
    graph.nodes[0]["theta"] += 100 * math.tau
    before = _state(graph)
    certificate = _sector_capture(graph)
    assert certificate.admitted
    assert max(map(abs, certificate.edge_turn_candidates)) >= 99
    assert certificate.ring_windings == (1, 1)
    assert _state(graph) == before
    assert not _local_capture(graph).radius_admitted


def test_true_pi_acute_boundary_is_distinct_from_the_represented_half_pi():
    for gap, admitted in (
        (math.pi / 2, True),
        (math.nextafter(math.pi / 2, math.inf), False),
    ):
        graph = _local_graph()
        graph.nodes[2]["theta"] = -gap
        certificate = _sector_capture(graph)
        assert certificate.acute_admitted is admitted
        if admitted:
            assert certificate.ring_windings == (1, 1)
        else:
            assert certificate.ring_windings[0] is None
            assert (
                "strict_acute_edge_lifts_not_certified"
                in certificate.unavailable_reasons
            )
    # An individually antipodal edge can still have a regular full resultant.
    # Low energy there does not place the state in the acute sector theorem.
    nonacute = _sector_capture(_graph(a=math.pi / 2, b=0.875))
    assert nonacute.energy_admitted and not nonacute.acute_admitted
    assert not nonacute.admitted


def test_sector_energy_and_period_admission_are_independent():
    wrong_sector = _sector_capture(_local_graph(0))
    assert wrong_sector.acute_admitted and wrong_sector.energy_admitted
    assert wrong_sector.ring_windings == (0, 0)
    assert not wrong_sector.sector_admitted and wrong_sector.target_sector is None
    assert wrong_sector.unavailable_reasons == ("declared_cycle_periods_not_certified",)
    graph = _local_graph()
    graph.nodes[2]["EPI"] = 1 / 8
    high_energy = _sector_capture(graph)
    assert high_energy.field.form_storage == Q(1, 64)
    assert high_energy.acute_admitted and high_energy.sector_admitted
    assert high_energy.storage_bounds[0] > high_energy.capture_barrier_bounds[1]
    assert not high_energy.energy_admitted and high_energy.target_sector is None
    assert high_energy.unavailable_reasons == (
        "strict_sector_energy_barrier_not_certified",
    )


@pytest.mark.parametrize("premise", ("capacity", "dissipation"))
def test_sector_certificate_retains_unsupported_model_premises(premise):
    graph, model = _local_graph(), _model(storage_scale=1)
    if premise == "capacity":
        graph.nodes[0]["nu_f"] = 0
        reason = "strictly_positive_held_capacity_required"
    else:
        model = _model(storage_scale=1, epi_weight=0, phase_weight=1)
        reason = "positive_epi_weight_required"
    certificate = _sector_capture(graph, model=model)
    assert certificate.acute_admitted and certificate.energy_admitted
    assert certificate.unavailable_reasons == (reason,)
    assert not certificate.admitted and certificate.target_sector is None
    assert certificate.future_acute_margin_lower_bound is None
    assert certificate.future_resultant_real_lower_bounds is None
    assert certificate.future_phase_metric_lower_bounds is None


@pytest.mark.parametrize(
    "beta, admitted", ((Q(1, 4), False), (Q(1), True), (Q(4), True))
)
def test_sector_scales_the_energy_barrier_without_scaling_form_storage(beta, admitted):
    graph = _local_graph()
    graph.nodes[0]["EPI"] = 1 / 16
    certificate = _sector_capture(graph, model=_model(storage_scale=beta))
    assert certificate.field.form_storage == Q(3, 512)
    assert certificate.positive_capacity
    assert certificate.unit_storage_scale == (beta == 1)
    assert certificate.capture_barrier_bounds == tuple(
        beta * value for value in certificate.geometric_barrier_bounds
    )
    assert certificate.storage_bounds == tuple(
        Q(3, 512) + beta * value for value in certificate.phase_storage_bounds
    )
    assert certificate.normalized_energy_margin_lower_bound == (
        certificate.geometric_barrier_bounds[0] - certificate.storage_bounds[1] / beta
    )
    assert certificate.admitted is admitted
    assert (certificate.future_acute_margin_lower_bound is not None) is admitted


def test_heterogeneous_capacity_changes_rates_but_not_the_sector_energy_basin():
    graph = _local_graph()
    graph.nodes[0]["EPI"] = 1 / 32
    graph.nodes[7]["theta"] += 1 / 128
    model = _model(storage_scale=2)
    original = _sector_capture(graph, model=model)
    scales = (0.5, 1.0, 2.0, 4.0, 0.5, 2.0, 4.0, 1.0, 0.5, 2.0)
    for node, scale in enumerate(scales):
        graph.nodes[node]["nu_f"] = scale
    before = _state(graph)
    changed = _sector_capture(graph, model=model)
    assert _state(graph) == before
    assert original.admitted and changed.admitted and changed.positive_capacity
    assert not changed.unit_capacity and not changed.unit_storage_scale
    assert changed.storage_bounds == original.storage_bounds
    assert changed.capture_barrier_bounds == original.capture_barrier_bounds
    assert (
        changed.future_phase_metric_lower_bounds
        == original.future_phase_metric_lower_bounds
    )
    assert changed.field.pressure == original.field.pressure
    for name in ("form_rate", "phase_rate"):
        assert getattr(changed.field, name) == pytest.approx(
            tuple(
                scale * rate
                for scale, rate in zip(scales, getattr(original.field, name))
            ),
            rel=2e-15,
            abs=0,
        )


def test_joint_form_and_beta_rescaling_preserves_normalized_basin_and_margin():
    graph = _local_graph()
    graph.nodes[0]["EPI"] = 1 / 16
    original = _sector_capture(graph)
    for node in graph:
        graph.nodes[node]["EPI"] *= 2
    rescaled = _sector_capture(graph, model=_model(storage_scale=4))
    assert original.admitted and rescaled.admitted
    assert rescaled.storage_bounds == tuple(
        4 * value for value in original.storage_bounds
    )
    assert (
        rescaled.normalized_energy_margin_lower_bound
        == original.normalized_energy_margin_lower_bound
    )
    assert (
        rescaled.future_acute_margin_lower_bound
        == original.future_acute_margin_lower_bound
    )
    # Beta/form rescaling is not a common clock rescaling: the two rates differ.
    assert rescaled.field.form_rate[0] == pytest.approx(2 * original.field.form_rate[0])
    assert rescaled.field.phase_rate[0] == pytest.approx(
        original.field.phase_rate[0] / 2
    )


def test_future_regularity_bounds_are_quantitative_and_available_only_on_full_admission():
    graph = _local_graph()
    graph.nodes[0]["EPI"] = 1 / 16
    graph.nodes[7]["theta"] += 1 / 64
    certificate = _sector_capture(graph)
    eta = certificate.normalized_energy_margin_lower_bound
    assert certificate.admitted and eta > 0
    assert certificate.future_acute_margin_lower_bound == 13 * eta
    assert certificate.future_acute_margin_lower_bound < min(
        lower for row in certificate.edge_acute_margin_bounds for lower, _ in row
    )
    expected = tuple(26 * graph.degree(node) * eta for node in certificate.field.nodes)
    assert certificate.future_phase_metric_lower_bounds == expected
    assert certificate.future_resultant_real_lower_bounds == tuple(
        value / certificate.pi_bounds[1] for value in expected
    )
    assert all(
        lower < Q(value)
        for lower, value in zip(expected, certificate.field.phase_metric)
    )
    # Low energy in the wrong topological sector cannot supply future bounds.
    wrong = _sector_capture(_local_graph(0))
    assert wrong.energy_admitted and not wrong.sector_admitted
    assert wrong.future_acute_margin_lower_bound is None
    assert wrong.future_resultant_real_lower_bounds is None
    assert wrong.future_phase_metric_lower_bounds is None


@pytest.mark.parametrize(
    "sector, error", ((True, TypeError), (1.0, TypeError), (0, ValueError))
)
def test_sector_target_requires_a_declared_nonzero_integer(sector, error):
    with pytest.raises(error, match="target_sector"):
        _sector_capture(_local_graph(), sector=sector)


def test_sector_ordered_topology_and_exact_periods_survive_relabeling():
    graph = _local_graph()
    labels = {node: ("site", -node) for node in graph}
    relabeled = nx.relabel_nodes(graph, labels)
    reordered = nx.Graph()
    reordered.graph.update(relabeled.graph)
    reordered.add_nodes_from(reversed(tuple(relabeled.nodes(data=True))))
    reordered.add_edges_from(reversed(tuple(relabeled.edges(data=True))))
    cycles = tuple(tuple(labels[node] for node in row) for row in CYCLES)
    certificate = certify_relational_sector_capture(
        reordered, model=_model(storage_scale=1), cycles=cycles
    )
    assert certificate.admitted and certificate.ring_windings == (1, 1)
    assert certificate.bridge_winding == 0
    reordered.add_edge(labels[2], labels[7])
    with pytest.raises(ValueError, match="support"):
        certify_relational_sector_capture(
            reordered, model=_model(storage_scale=1), cycles=cycles
        )


def test_new_sector_reading_does_not_rewrite_the_failed_frozen_endpoint_gate():
    path = (
        Path(__file__).resolve().parents[1]
        / "docs/assets/relational_capture_response/result.json"
    )
    original = path.read_bytes()
    saved = json.loads(original)
    finest = saved["traces"][str(max(map(int, saved["traces"])))]
    endpoint = finest["checkpoints"][finest["last_admitted_checkpoint"]]
    field = endpoint["reflected"]["report"]["field"]
    graph = nx.Graph()
    graph.add_nodes_from(field["nodes"])
    graph.add_edges_from(field["edges"])
    for node, form, phase, capacity in zip(
        field["nodes"], field["epi"], field["phase"], field["capacity"], strict=True
    ):
        graph.nodes[node].update(EPI=form, theta=phase, nu_f=capacity)
    certificate = _sector_capture(graph)
    assert certificate.admitted and certificate.target_sector == 1
    assert not saved["finite_prediction_passed"]
    assert not endpoint["capture_admitted"]
    assert endpoint["local"]["report"]["status"] == "unavailable"
    assert endpoint["reflected"]["report"]["status"] == "unavailable"
    assert path.read_bytes() == original
