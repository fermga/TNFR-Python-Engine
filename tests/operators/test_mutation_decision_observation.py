"""Immutable executor-owned observations for accepted ZHIR decisions."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
from types import SimpleNamespace
from typing import get_type_hints

import networkx as nx
import pytest

import tnfr.operators.network_stage as network_stage
from tnfr.operators.definitions import Emission, Mutation
from tnfr.operators.event_runtime import (
    OperatorEventExecutionResult,
    execute_operator_event_schedule,
)
from tnfr.operators.event_timing import build_operator_event_schedule
from tnfr.operators.network_stage import (
    MutationStageDecisionObservation,
    NetworkStageResult,
    TWO_PHASE_JACOBI,
    execute_pointwise_stage,
)
from tnfr.types import Glyph


def _mutation_graph() -> nx.Graph:
    graph = nx.path_graph(3)
    graph.graph.update(
        GLYPH_FACTORS={"ZHIR_theta_shift_factor": 0.5},
        ZHIR_THRESHOLD_XI=0.1,
        EPI_MIN=-1.0,
        EPI_MAX=1.0,
    )
    for node in graph:
        epi = 0.25 + 0.1 * node
        graph.nodes[node].update(
            EPI=epi,
            nu_f=1.0 + 0.25 * node,
            DeltaNFR=0.2 + 0.05 * node,
            theta=0.1 * node,
            EPI_kind="wave",
            glyph_history=["AL", "IL", "OZ"],
            epi_history=[0.0, 0.1, epi],
        )
    return graph


def test_two_phase_mutation_exposes_ordered_decisions_without_epi_certificate(
) -> None:
    graph = _mutation_graph()
    targets = (2, 0, 1)

    result = execute_pointwise_stage(
        graph,
        Mutation(),
        targets,
        tau=0.01,
    )

    observations = result.mutation_decision_observations
    assert result.pointwise_epi_jump_certificate is None
    assert tuple(item.target_index for item in observations) == (0, 1, 2)
    assert tuple(item.node for item in observations) == targets
    assert all(type(item) is MutationStageDecisionObservation for item in observations)
    assert all(item.glyph is Glyph.ZHIR for item in observations)
    assert all(item._proof_fields_are_intact() for item in observations)
    assert all(
        item.trigger_certificate.threshold_gate_satisfied
        for item in observations
    )
    assert all(item.trigger_certificate.evidence_valid for item in observations)
    assert all(item.trigger_certificate.evidence is not None for item in observations)
    assert tuple(item.operator_step for item in observations) == (4, 4, 4)
    assert tuple(item.destabilizer_operator for item in observations) == (
        "dissonance",
        "dissonance",
        "dissonance",
    )


def test_mutation_decisions_are_detached_from_later_live_graph_writes() -> None:
    graph = _mutation_graph()
    result = execute_pointwise_stage(graph, Mutation(), (0, 1, 2), tau=0.01)
    observations = result.mutation_decision_observations
    retained = deepcopy(observations)

    for node in graph:
        graph.nodes[node].update(
            EPI=-0.75,
            theta=5.0,
            epi_history=[9.0, 9.0, 9.0],
            _zhir_gate_depi_dt=-99.0,
            _zhir_tau=99.0,
            _mutation_context={"destabilizer_operator": "changed"},
        )
    graph.graph["zhir_bifurcation_events"] = []

    assert observations == retained
    assert all(item._proof_fields_are_intact() for item in observations)
    assert observations[0].trigger_certificate.current_epi == pytest.approx(0.25)
    assert observations[0].tau == pytest.approx(0.01)
    assert observations[0].destabilizer_operator == "dissonance"
    with pytest.raises(FrozenInstanceError):
        observations[0].tau = 9.0  # type: ignore[misc]


def test_event_stage_retains_executor_owned_mutation_decisions() -> None:
    graph = _mutation_graph()
    graph.graph.update(
        _t=0.0,
        RANDOM_SEED=7,
        GAMMA={"type": "none"},
        EPI_MIN=-4.0,
        EPI_MAX=4.0,
        DT_MIN=0.25,
    )
    for node in graph:
        graph.nodes[node]["epi_time_history"] = [
            (-1.0, float(graph.nodes[node]["EPI"])),
            (0.0, float(graph.nodes[node]["EPI"])),
        ]
    schedule = build_operator_event_schedule(
        (
            "emission",
            "coherence",
            "dissonance",
            "mutation",
            "coherence",
            "silence",
        ),
        start_time=0.0,
        flow_durations=(0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0),
    )

    result = execute_operator_event_schedule(
        graph,
        schedule,
        include_stage_certificates=True,
    )
    stage = result.glyph_stage_evidence[3]

    assert stage.event.glyph is Glyph.ZHIR
    assert len(stage.mutation_decision_observations) == len(graph)
    assert tuple(
        item.node for item in stage.mutation_decision_observations
    ) == result.target_nodes
    assert all(
        item._proof_fields_are_intact()
        for item in stage.mutation_decision_observations
    )
    assert all(
        item.trigger_certificate.physical_time_resolved
        for item in stage.mutation_decision_observations
    )


def test_network_stage_result_rejects_mutation_observation_corruption() -> None:
    result = execute_pointwise_stage(
        _mutation_graph(), Mutation(), (0, 1, 2), tau=0.01
    )
    observations = result.mutation_decision_observations

    with pytest.raises(ValueError, match="one decision observation"):
        replace(result, mutation_decision_observations=observations[:-1])
    with pytest.raises(ValueError, match="target order"):
        replace(result, mutation_decision_observations=tuple(reversed(observations)))
    with pytest.raises(ValueError, match="Only two-phase Mutation"):
        replace(result, glyph=Glyph.AL.value)

    object.__setattr__(observations[0], "tau", 0.02)
    assert not observations[0]._proof_fields_are_intact()
    with pytest.raises(ValueError, match="proof fields"):
        result.__post_init__()


def test_mutation_observation_rejects_forged_nested_decisions() -> None:
    result = execute_pointwise_stage(
        _mutation_graph(), Mutation(), (0, 1, 2), tau=0.01
    )
    observation = result.mutation_decision_observations[0]
    certificate = observation.trigger_certificate
    assert certificate.evidence is not None

    with pytest.raises(ValueError, match="internally inconsistent"):
        replace(
            observation,
            trigger_certificate=replace(certificate, observed_crossed=False),
        )
    with pytest.raises(ValueError, match="internally inconsistent"):
        replace(
            observation,
            trigger_certificate=replace(
                certificate,
                evidence=replace(
                    certificate.evidence,
                    observed_depi_dt=certificate.observed_depi_dt + 1.0,
                ),
            ),
        )
    with pytest.raises(ValueError, match="regime decision"):
        replace(observation, regime_changed=not observation.regime_changed)
    with pytest.raises(ValueError, match="phase decision"):
        replace(observation, theta_after=observation.theta_before)


class _MutableNode:
    def __init__(self, label: str) -> None:
        self.label = label

    __hash__ = object.__hash__


class _HostileEqualityNode:
    __slots__ = ("label",)

    def __init__(self, label: str) -> None:
        self.label = label

    __hash__ = object.__hash__

    def __eq__(self, _other: object) -> bool:
        raise RuntimeError("hostile equality must not be called")


def test_mutable_and_hostile_node_identifiers_are_structurally_sealed() -> None:
    mutable = _MutableNode("before")
    peer = _MutableNode("peer")
    graph = nx.Graph()
    graph.add_edge(mutable, peer)
    graph.graph.update(
        GLYPH_FACTORS={"ZHIR_theta_shift_factor": 0.5},
        ZHIR_THRESHOLD_XI=0.1,
    )
    for index, node in enumerate((mutable, peer)):
        epi = 0.25 + 0.1 * index
        graph.nodes[node].update(
            EPI=epi,
            nu_f=1.0,
            DeltaNFR=0.2,
            theta=0.1 * index,
            EPI_kind="wave",
            glyph_history=["AL", "IL", "OZ"],
            epi_history=[0.0, 0.1, epi],
        )

    result = execute_pointwise_stage(
        graph, Mutation(), (mutable, peer), tau=0.01
    )
    mutable_observation = result.mutation_decision_observations[0]
    hostile_observation = replace(
        mutable_observation,
        node=_HostileEqualityNode("stable"),
    )
    coherent_replacement = replace(
        mutable_observation,
        operator_step=mutable_observation.operator_step + 1,
    )

    assert mutable_observation._proof_fields_are_intact()
    assert not hostile_observation._proof_fields_are_intact()
    assert not coherent_replacement._proof_fields_are_intact()
    mutable.label = "after"
    assert not mutable_observation._proof_fields_are_intact()


def test_network_stage_result_preserves_legacy_positional_field_order() -> None:
    legacy_certificate = object()

    result = NetworkStageResult(
        "emission",
        Glyph.AL.value,
        TWO_PHASE_JACOBI,
        1,
        legacy_certificate,
        None,
        None,
    )

    assert result.pointwise_epi_jump_certificate is legacy_certificate
    assert result.mutation_decision_observations == ()


def test_nonmutation_stage_has_no_mutation_decisions() -> None:
    graph = _mutation_graph()

    result = execute_pointwise_stage(graph, Emission(), (0, 1, 2))

    assert result.mutation_decision_observations == ()


def test_mutation_decision_observation_has_public_inline_type_surface() -> None:
    public_names = set(network_stage.__all__)
    result_hints = get_type_hints(NetworkStageResult)

    assert "MutationStageDecisionObservation" in public_names
    assert "NetworkStageResult" in public_names
    assert result_hints["mutation_decision_observations"] == tuple[
        MutationStageDecisionObservation, ...
    ]


class _VariableReprNode:
    __slots__ = ()

    calls = 0
    __hash__ = object.__hash__

    def __repr__(self) -> str:
        type(self).calls += 1
        return f"volatile-{type(self).calls}"


def _identity_node_graph(left: object, right: object) -> nx.Graph:
    graph = nx.Graph()
    graph.add_edge(left, right)
    graph.graph.update(
        GLYPH_FACTORS={"ZHIR_theta_shift_factor": 0.5},
        ZHIR_THRESHOLD_XI=0.1,
    )
    for index, node in enumerate((left, right)):
        epi = 0.25 + 0.1 * index
        graph.nodes[node].update(
            EPI=epi,
            nu_f=1.0,
            DeltaNFR=0.2,
            theta=0.1 * index,
            EPI_kind="wave",
            glyph_history=["AL", "IL", "OZ"],
            epi_history=[0.0, 0.1, epi],
        )
    return graph


def test_distinct_identity_semantic_node_cannot_reuse_observation_seal() -> None:
    left = _MutableNode("same")
    right = _MutableNode("peer")
    result = execute_pointwise_stage(
        _identity_node_graph(left, right),
        Mutation(),
        (left, right),
        tau=0.01,
    )
    observations = result.mutation_decision_observations
    substituted = replace(observations[0], node=_MutableNode("same"))

    assert not substituted._proof_fields_are_intact()
    with pytest.raises(ValueError, match="proof fields"):
        replace(
            result,
            mutation_decision_observations=(substituted, observations[1]),
        )


def test_hostile_equality_node_uses_identity_short_circuit() -> None:
    left = _HostileEqualityNode("left")
    right = _HostileEqualityNode("right")

    result = execute_pointwise_stage(
        _identity_node_graph(left, right),
        Mutation(),
        (left, right),
        tau=0.01,
    )

    assert result.mutation_decision_observations[0].node is left
    assert all(
        observation._proof_fields_are_intact()
        for observation in result.mutation_decision_observations
    )


def test_variable_repr_does_not_destabilize_opaque_node_seal() -> None:
    left = _VariableReprNode()
    right = _VariableReprNode()

    result = execute_pointwise_stage(
        _identity_node_graph(left, right),
        Mutation(),
        (left, right),
        tau=0.01,
    )

    assert tuple(
        observation.node for observation in result.mutation_decision_observations
    ) == (left, right)
    assert all(
        observation._proof_fields_are_intact()
        for observation in result.mutation_decision_observations
    )


def test_distinct_hostile_equality_payload_fails_closed() -> None:
    target = _HostileEqualityNode("target")
    payload_node = _HostileEqualityNode("payload")
    proposal = network_stage.PointwiseStageProposal(
        node=target,
        glyph=Glyph.ZHIR,
        payload=SimpleNamespace(node=payload_node, glyph=Glyph.ZHIR),
    )

    with pytest.raises(RuntimeError, match="payload target changed"):
        network_stage._validate_pointwise_proposals(
            (proposal,),
            (target,),
            Glyph.ZHIR,
        )


def _event_mutation_execution() -> OperatorEventExecutionResult:
    graph = _mutation_graph()
    graph.graph.update(
        _t=0.0,
        RANDOM_SEED=7,
        GAMMA={"type": "none"},
        EPI_MIN=-4.0,
        EPI_MAX=4.0,
        DT_MIN=0.25,
    )
    for node in graph:
        graph.nodes[node]["epi_time_history"] = [
            (-1.0, float(graph.nodes[node]["EPI"])),
            (0.0, float(graph.nodes[node]["EPI"])),
        ]
    schedule = build_operator_event_schedule(
        (
            "emission",
            "coherence",
            "dissonance",
            "mutation",
            "coherence",
            "silence",
        ),
        start_time=0.0,
        flow_durations=(0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0),
    )
    return execute_operator_event_schedule(
        graph,
        schedule,
        include_stage_certificates=True,
    )


def test_executed_glyph_stage_seal_fails_closed_under_decision_tampering(
) -> None:
    import tnfr.operators.event_runtime as event_runtime

    result = _event_mutation_execution()
    stages = result.glyph_stage_evidence
    mutation_stage = stages[3]
    observations = mutation_stage.mutation_decision_observations

    assert mutation_stage._proof_fields_are_intact()

    removed = replace(
        mutation_stage,
        mutation_decision_observations=(),
    )
    attached_to_emission = replace(
        stages[0],
        mutation_decision_observations=observations,
    )
    promoted_coherence = replace(
        stages[1],
        _represented_affine_gain_bound_at_observed_endpoint_certified=True,
    )
    reversed_observations = replace(
        mutation_stage,
        mutation_decision_observations=tuple(reversed(observations)),
    )
    assert observations[0].minimum_nu_f.hex() == 0.0.hex()
    signed_zero_observation = replace(
        observations[0],
        minimum_nu_f=-0.0,
    )
    assert not signed_zero_observation._proof_fields_are_intact()

    for forged in (
        removed,
        attached_to_emission,
        promoted_coherence,
        reversed_observations,
    ):
        assert not forged._proof_fields_are_intact()
        assert not (
            forged
            .represented_affine_gain_bound_at_observed_endpoint_certified
        )

    resealed_attachment = replace(
        attached_to_emission,
        _proof_stamp=event_runtime._executed_glyph_stage_stamp(
            attached_to_emission
        ),
    )
    resealed_reversal = replace(
        reversed_observations,
        _proof_stamp=event_runtime._executed_glyph_stage_stamp(
            reversed_observations
        ),
    )
    assert not resealed_attachment._proof_fields_are_intact()
    assert not resealed_reversal._proof_fields_are_intact()


def test_event_result_rejects_removed_reordered_or_forged_stage_evidence(
) -> None:
    result = _event_mutation_execution()
    stages = result.glyph_stage_evidence
    mutation_stage = stages[3]
    removed = replace(
        mutation_stage,
        mutation_decision_observations=(),
    )
    forged_stages = list(stages)
    forged_stages[3] = removed

    with pytest.raises(ValueError, match="one record per committed event"):
        replace(result, glyph_stage_evidence=stages[:-1])
    with pytest.raises(ValueError, match="proof fields are not intact"):
        replace(result, glyph_stage_evidence=tuple(forged_stages))

    reordered = list(stages)
    reordered[0], reordered[3] = reordered[3], reordered[0]
    with pytest.raises(ValueError, match="committed event order"):
        replace(result, glyph_stage_evidence=tuple(reordered))

    with pytest.raises(ValueError, match="disabled stage certification"):
        replace(result, stage_certification_requested=False)


def test_stage_seal_accepts_ordered_target_subset_against_full_endpoints(
) -> None:
    import tnfr.operators.event_runtime as event_runtime
    from tnfr.physics.runtime_flow_stability import capture_nodal_flow_state

    graph = _mutation_graph()
    targets = (2, 0)
    schedule = build_operator_event_schedule(
        ("mutation",),
        start_time=0.0,
        flow_durations=(0.0, 0.0),
    )
    left = capture_nodal_flow_state(graph)
    stage_result = execute_pointwise_stage(
        graph,
        Mutation(),
        targets,
        tau=0.01,
    )
    right = capture_nodal_flow_state(graph)
    event = event_runtime.ExecutedOperatorEvent.from_stage(
        schedule.events[0],
        stage_result,
    )
    stage = event_runtime._finalize_glyph_stage(
        event_runtime._PendingGlyphStage(
            event=event,
            result=stage_result,
            left=left,
            right=right,
            left_captured=True,
            right_captured=True,
        ),
        schedule,
        {},
    )

    assert left.nodes == (0, 1, 2)
    assert tuple(
        observation.node
        for observation in stage.mutation_decision_observations
    ) == targets
    assert stage._proof_fields_are_intact()

    composition = event_runtime._compose_observed_represented_epi_schedule(
        schedule,
        targets,
        (),
        (stage,),
    )
    execution = OperatorEventExecutionResult(
        schedule=schedule,
        target_nodes=targets,
        flow_interval_indices=(0, 1),
        positive_flow_interval_indices=(),
        events=(event,),
        final_time=0.0,
        integrator_name=None,
        pressure_refresh_callback_invocations=0,
        flow_certification_requested=True,
        stage_certification_requested=True,
        glyph_stage_evidence=(stage,),
        represented_epi_schedule_composition=composition,
    )

    assert execution.target_nodes == targets
    with pytest.raises(
        ValueError,
        match="observations do not match execution targets",
    ):
        replace(execution, target_nodes=tuple(reversed(targets)))
