"""Adversarial boundaries for observed represented EPI schedule composition."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from fractions import Fraction
from typing import Any

import networkx as nx
import pytest

from tnfr.operators.event_runtime import (
    ObservedRepresentedEPIScheduleComposition,
    RepresentedEPIScheduleOperation,
    execute_operator_event_schedule,
)
from tnfr.operators.event_timing import build_operator_event_schedule
from tnfr.operators.network_stage import NetworkStageResult, TWO_PHASE_JACOBI


def _graph() -> nx.Graph:
    graph = nx.path_graph(2)
    graph.graph.update(
        _t=0.0,
        DT_MIN=0.0,
        GAMMA={"type": "none"},
        EPI_MIN=-10.0,
        EPI_MAX=10.0,
        NAV_RANDOM=False,
        RANDOM_SEED=7,
        GLYPH_FACTORS={"NAV_eta": 0.25, "NAV_jitter": 0.0},
    )
    for node, epi, pressure in zip(
        graph,
        (0.25, -0.25),
        (-0.5, 0.5),
        strict=True,
    ):
        graph.nodes[node].update(
            EPI=epi,
            epi_kind="test",
            nu_f=1.0,
            theta=0.0,
            delta_nfr=pressure,
            latent=False,
            glyph_history=[],
        )
    return graph


def _refresh_pure_epi_pressure(graph: nx.Graph) -> None:
    left = float(graph.nodes[0]["EPI"])
    right = float(graph.nodes[1]["EPI"])
    graph.nodes[0]["delta_nfr"] = right - left
    graph.nodes[1]["delta_nfr"] = left - right


def _transition_composition() -> ObservedRepresentedEPIScheduleComposition:
    graph = _graph()
    graph.graph["compute_delta_nfr"] = _refresh_pure_epi_pressure
    schedule = build_operator_event_schedule(
        ("transition",),
        start_time=0.0,
        flow_durations=(0.125, 0.125),
    )
    result = execute_operator_event_schedule(
        graph,
        schedule,
        include_stage_certificates=True,
    )
    composition = result.represented_epi_schedule_composition
    assert composition is not None
    return composition


def test_complete_trace_records_every_flow_and_glyph_in_execution_order() -> None:
    composition = _transition_composition()

    assert isinstance(composition, ObservedRepresentedEPIScheduleComposition)
    assert all(
        isinstance(operation, RepresentedEPIScheduleOperation)
        for operation in composition.operations
    )
    assert tuple(
        (operation.position, operation.operation_kind, operation.operation_index)
        for operation in composition.operations
    ) == ((0, "flow", 0), (1, "glyph", 0), (2, "flow", 1))
    assert all(
        operation.represented_affine_gain_certified
        for operation in composition.operations
    )
    assert composition.represented_affine_composition_gain_certified
    assert composition.represented_map_global_disagreement_contraction_certified
    assert composition.exact_operation_energy_gain_factors == (
        Fraction(9, 16),
        Fraction(1),
        Fraction(9, 16),
    )
    assert composition.exact_energy_gain_upper_bound == Fraction(81, 256)
    assert not composition.runtime_schedule_global_gain_certified
    assert not composition.future_or_repeated_schedule_stability_certified
    assert composition._proof_fields_are_intact()
    assert dict(composition.conditions)["exact_operation_support_continuity"]


def test_unsupported_gaps_keep_indexed_abstentions_and_no_partial_product() -> None:
    graph = _graph()
    schedule = build_operator_event_schedule(
        ("emission", "coupling", "coherence", "silence"),
        start_time=0.0,
        flow_durations=(0.0, 0.0, 0.0, 0.0, 0.0),
    )

    result = execute_operator_event_schedule(
        graph,
        schedule,
        include_stage_certificates=True,
        suppress_birth_warnings=True,
    )
    composition = result.represented_epi_schedule_composition
    assert composition is not None
    assert tuple(
        (item.position, item.operation_kind, item.operation_index, item.operator_name)
        for item in composition.operations
    ) == (
        (0, "glyph", 0, "emission"),
        (1, "glyph", 1, "coupling"),
        (2, "glyph", 2, "coherence"),
        (3, "glyph", 3, "silence"),
    )
    coupling, coherence = composition.operations[1:3]
    assert any("UM" in reason for reason in coupling.ineligibility_reasons)
    assert any("IL" in reason for reason in coherence.ineligibility_reasons)
    assert composition.exact_operation_energy_gain_factors == ()
    assert composition.exact_energy_gain_upper_bound is None
    assert composition.exact_normalized_metric is None
    assert not composition.represented_affine_composition_gain_certified
    conditions = dict(composition.conditions)
    assert conditions["complete_operation_cardinality"]
    assert conditions["all_operation_nodes_match"]
    assert conditions["exact_operation_support_continuity"]
    assert conditions["exact_operation_endpoint_continuity"]
    assert not conditions["all_glyph_stages_represented_affine"]
    assert not conditions["one_exact_normalized_metric"]


def test_composition_and_operation_proof_stamps_reject_replaced_claims() -> None:
    composition = _transition_composition()
    first = composition.operations[0]

    with pytest.raises(ValueError, match="proof fields"):
        replace(
            first,
            exact_epi_after=tuple(value + 1 for value in first.exact_epi_after or ()),
        )
    with pytest.raises(ValueError, match="composition gain"):
        replace(composition, exact_energy_gain_upper_bound=Fraction(0))
    changed_conditions = tuple(
        (name, False if name == "one_exact_normalized_metric" else passed)
        for name, passed in composition.conditions
    )
    with pytest.raises(ValueError, match="conditions"):
        replace(composition, conditions=changed_conditions)


def test_object_setattr_tamper_cannot_promote_operation_or_composition() -> None:
    composition = _transition_composition()
    first = composition.operations[0]

    object.__setattr__(first, "exact_energy_gain_upper_bound", Fraction(0))

    assert not first._proof_fields_are_intact()
    assert not first.represented_affine_gain_certified
    assert not composition._proof_fields_are_intact()
    assert not composition.represented_affine_composition_gain_certified
    assert not composition.represented_map_global_disagreement_contraction_certified
    assert composition.failed_conditions == ("composition_proof_fields_intact",)


def test_object_setattr_tamper_cannot_promote_composition_gain() -> None:
    composition = _transition_composition()

    object.__setattr__(composition, "exact_energy_gain_upper_bound", Fraction(0))

    assert not composition._proof_fields_are_intact()
    assert not composition.represented_affine_composition_gain_certified
    assert not composition.represented_map_global_disagreement_contraction_certified
    assert composition.failed_conditions == ("composition_proof_fields_intact",)
    for field_name in (
        "runtime_schedule_global_gain_certified",
        "solver_accuracy_certified",
        "full_multichannel_stability_certified",
        "future_or_repeated_schedule_stability_certified",
    ):
        with pytest.raises(AttributeError):
            object.__setattr__(composition, field_name, True)
        assert getattr(composition, field_name) is False


def test_replaced_flow_certificate_cannot_enter_composition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tnfr.physics import runtime_flow_stability

    certify = runtime_flow_stability.certify_observed_nodal_flow_interval

    def replaced_certificate(*args: Any, **kwargs: Any) -> Any:
        return replace(
            certify(*args, **kwargs),
            exact_quotient_energy_gain_upper_bound=Fraction(0),
        )

    monkeypatch.setattr(
        runtime_flow_stability,
        "certify_observed_nodal_flow_interval",
        replaced_certificate,
    )
    graph = _graph()
    graph.graph["compute_delta_nfr"] = _refresh_pure_epi_pressure
    schedule = build_operator_event_schedule(
        ("transition",),
        start_time=0.0,
        flow_durations=(0.125, 0.125),
    )

    result = execute_operator_event_schedule(
        graph,
        schedule,
        include_stage_certificates=True,
    )

    assert all(
        evidence.certificate is not None
        and not evidence.certificate._proof_fields_are_intact()
        and not evidence.runtime_bound_binary64_interval_identified
        and not evidence.runtime_bound_exact_affine_map_identified
        and not evidence.runtime_bound_global_disagreement_contraction_certified
        for evidence in result.flow_interval_evidence
    )
    composition = result.represented_epi_schedule_composition
    assert composition is not None
    assert composition.exact_operation_energy_gain_factors == ()
    assert composition.exact_energy_gain_upper_bound is None
    assert not composition.represented_affine_composition_gain_certified
    assert "all_positive_flows_exact_affine" in composition.failed_conditions


def test_cross_operator_nested_certificate_aborts_and_rolls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tnfr.operators import word_execution

    source_graph = _graph()
    source_schedule = build_operator_event_schedule(
        ("transition",),
        start_time=0.0,
        flow_durations=(0.0, 0.0),
    )
    source_result = execute_operator_event_schedule(
        source_graph,
        source_schedule,
        include_stage_certificates=True,
    )
    wrong_certificate = source_result.glyph_stage_evidence[0].certificate
    assert wrong_certificate is not None

    def forged_stage(
        _graph_value: nx.Graph,
        operator: Any,
        targets: tuple[Any, ...],
        **_kwargs: Any,
    ) -> NetworkStageResult:
        return NetworkStageResult(
            operator=operator.name,
            glyph=operator.glyph.value,
            schedule=TWO_PHASE_JACOBI,
            nodes_processed=len(targets),
            pointwise_epi_jump_certificate=wrong_certificate,
        )

    monkeypatch.setattr(word_execution, "execute_network_operator_stage", forged_stage)
    graph = _graph()
    before_nodes = deepcopy(dict(graph.nodes(data=True)))
    before_graph = deepcopy(dict(graph.graph))
    schedule = build_operator_event_schedule(
        ("emission", "coupling", "coherence", "silence"),
        start_time=0.0,
        flow_durations=(0.0, 0.0, 0.0, 0.0, 0.0),
    )

    with pytest.raises(RuntimeError, match="certificate identity mismatch"):
        execute_operator_event_schedule(
            graph,
            schedule,
            include_stage_certificates=True,
            suppress_birth_warnings=True,
        )

    assert dict(graph.nodes(data=True)) == before_nodes
    assert dict(graph.graph) == before_graph
