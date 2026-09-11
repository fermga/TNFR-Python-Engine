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
    OperatorEventExecutionResult,
    RepresentedEPIScheduleOperation,
    execute_operator_event_schedule,
)
from tnfr.operators.event_timing import (
    OperatorEventSchedule,
    build_operator_event_schedule,
)
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
    left_node, right_node = tuple(graph)
    left = float(graph.nodes[left_node]["EPI"])
    right = float(graph.nodes[right_node]["EPI"])
    graph.nodes[left_node]["delta_nfr"] = right - left
    graph.nodes[right_node]["delta_nfr"] = left - right


def _transition_result(
    *,
    epi: tuple[float, float] = (0.25, -0.25),
    schedule: OperatorEventSchedule | None = None,
) -> OperatorEventExecutionResult:
    graph = _graph()
    for node, value in zip(graph, epi, strict=True):
        graph.nodes[node]["EPI"] = value
    _refresh_pure_epi_pressure(graph)
    graph.graph["compute_delta_nfr"] = _refresh_pure_epi_pressure
    if schedule is None:
        schedule = build_operator_event_schedule(
            ("transition",),
            start_time=0.0,
            flow_durations=(0.125, 0.125),
        )
    return execute_operator_event_schedule(
        graph,
        schedule,
        include_stage_certificates=True,
    )


def _transition_composition() -> ObservedRepresentedEPIScheduleComposition:
    result = _transition_result()
    composition = result.represented_epi_schedule_composition
    assert composition is not None
    return composition


def test_result_rejects_foreign_flow_and_composition_evidence() -> None:
    schedule = build_operator_event_schedule(
        ("transition",),
        start_time=0.0,
        flow_durations=(0.125, 0.125),
    )
    result = _transition_result(schedule=schedule)
    foreign = _transition_result(epi=(0.3, -0.3), schedule=schedule)

    with pytest.raises(
        ValueError,
        match="proof fields",
    ):
        replace(
            result,
            flow_interval_evidence=foreign.flow_interval_evidence,
        )
    with pytest.raises(
        ValueError,
        match="proof fields",
    ):
        replace(
            result,
            represented_epi_schedule_composition=(
                foreign.represented_epi_schedule_composition
            ),
        )
    with pytest.raises(ValueError, match="proof fields"):
        replace(
            result,
            flow_interval_evidence=tuple(
                reversed(result.flow_interval_evidence)
            ),
        )


def test_mutable_node_state_is_part_of_operation_and_composition_seals() -> None:
    class MutableNode:
        def __init__(self, label: str) -> None:
            self.label = label

    left = MutableNode("left")
    right = MutableNode("right")
    graph = nx.relabel_nodes(_graph(), {0: left, 1: right}, copy=True)
    _refresh_pure_epi_pressure(graph)
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
    assert composition._proof_fields_are_intact()

    left.label = "changed"

    assert not composition.operations[0]._proof_fields_are_intact()
    assert not composition._proof_fields_are_intact()
    assert not result._proof_fields_are_intact()


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
    with pytest.raises(ValueError, match="proof fields"):
        replace(composition, exact_energy_gain_upper_bound=Fraction(0))
    changed_conditions = tuple(
        (name, False if name == "one_exact_normalized_metric" else passed)
        for name, passed in composition.conditions
    )
    with pytest.raises(ValueError, match="proof fields"):
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
    # ``scope`` is a published ``init=False`` dataclass field.  Raw slot
    # mutation must therefore invalidate the seal while the public read-out
    # remains pinned to the canonical represented-only claim.
    object.__setattr__(composition, "scope", "global executable schedule")
    assert not composition._proof_fields_are_intact()
    assert "represented" in composition.scope


def test_runtime_proof_records_reject_hostile_stamp_tokens_without_dispatch() -> None:
    class HostileProofToken:
        calls = 0

        def __bool__(self) -> bool:
            type(self).calls += 1
            raise SystemExit("proof validation invoked hostile truth conversion")

        def __eq__(self, _other: object) -> bool:
            type(self).calls += 1
            raise SystemExit("proof validation invoked hostile equality")

    cases = (
        (lambda result: result.events[0], "zero_duration"),
        (
            lambda result: result.flow_interval_evidence[0],
            "integrator_provenance_certified",
        ),
        (
            lambda result: result.glyph_stage_evidence[0],
            "represented_affine_gain_bound_at_observed_endpoint_certified",
        ),
        (
            lambda result: result.represented_epi_schedule_composition.operations[0],
            "represented_affine_gain_certified",
        ),
        (
            lambda result: result.represented_epi_schedule_composition,
            "represented_affine_composition_gain_certified",
        ),
        (lambda result: result, "runtime_clock_checked"),
    )

    for select, claim_name in cases:
        result = _transition_result()
        record = select(result)
        assert record is not None
        original = object.__getattribute__(record, "_proof_stamp")
        assert type(original) is tuple and original
        probe = HostileProofToken()
        object.__setattr__(record, "_proof_stamp", (*original[:-1], probe))

        assert getattr(record, claim_name) is False
        assert HostileProofToken.calls == 0


@pytest.mark.parametrize(
    ("select", "deleted_slot", "claim_name", "has_event_record"),
    (
        (lambda result: result.events[0], "_proof_stamp", "zero_duration", True),
        (lambda result: result.events[0], "zero_duration", "zero_duration", True),
        (
            lambda result: result.flow_interval_evidence[0],
            "_proof_stamp",
            "integrator_provenance_certified",
            False,
        ),
        (
            lambda result: result.flow_interval_evidence[0],
            "integrator_provenance_certified",
            "integrator_provenance_certified",
            False,
        ),
        (
            lambda result: result.glyph_stage_evidence[0],
            "_proof_stamp",
            "represented_affine_gain_bound_at_observed_endpoint_certified",
            False,
        ),
        (
            lambda result: result.represented_epi_schedule_composition.operations[0],
            "_proof_stamp",
            "represented_affine_gain_certified",
            False,
        ),
        (
            lambda result: result.represented_epi_schedule_composition,
            "_proof_stamp",
            "represented_affine_composition_gain_certified",
            False,
        ),
        (lambda result: result, "_proof_stamp", "runtime_clock_checked", False),
        (
            lambda result: result,
            "runtime_clock_checked",
            "runtime_clock_checked",
            False,
        ),
    ),
)
def test_runtime_proof_records_fail_closed_when_a_slot_is_deleted(
    select: Any,
    deleted_slot: str,
    claim_name: str,
    has_event_record: bool,
) -> None:
    result = _transition_result()
    record = select(result)
    assert record is not None
    assert record._proof_fields_are_intact()

    object.__delattr__(record, deleted_slot)

    assert record._proof_fields_are_intact() is False
    assert getattr(record, claim_name) is False
    if has_event_record:
        with pytest.raises(ValueError, match="proof fields are not intact"):
            record.as_record()


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
