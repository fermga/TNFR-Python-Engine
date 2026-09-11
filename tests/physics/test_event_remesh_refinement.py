"""Finite three-mesh event/REMESH refinement observations."""

from __future__ import annotations

from collections import deque
from fractions import Fraction
from types import SimpleNamespace

import networkx as nx
import pytest
import tnfr.physics.event_remesh_refinement as refinement_module

from tnfr.operators.event_remesh_runtime import execute_event_remesh_cycle
from tnfr.operators.event_timing import (
    build_operator_event_schedule,
    build_physical_flow_partition,
)
from tnfr.physics.event_remesh_refinement import (
    EventRemeshThreeMeshRefinementObservation,
    observe_event_remesh_three_mesh_refinement,
)
from tnfr.types import scalarize_epi


_ZHIR_WORD = (
    "emission",
    "coherence",
    "dissonance",
    "mutation",
    "coherence",
    "silence",
)

_TWO_ZHIR_WORD = (
    "emission",
    "coherence",
    "dissonance",
    "mutation",
    "coherence",
    "dissonance",
    "coherence",
    "mutation",
    "coherence",
    "silence",
)


def _refresh_pure_epi_pressure(graph: nx.Graph) -> None:
    for node in graph:
        neighbours = tuple(graph.neighbors(node))
        if not neighbours:
            pressure = 0.0
        else:
            total_weight = sum(
                float(graph.edges[node, neighbour].get("weight", 1.0))
                for neighbour in neighbours
            )
            mean = sum(
                float(graph.edges[node, neighbour].get("weight", 1.0))
                * scalarize_epi(graph.nodes[neighbour]["EPI"])
                for neighbour in neighbours
            ) / total_weight
            pressure = mean - scalarize_epi(graph.nodes[node]["EPI"])
        graph.nodes[node]["delta_nfr"] = pressure


def _refresh_positive_time_pressure(graph: nx.Graph) -> None:
    pressure = 1.0 + float(graph.graph["_t"])
    for node in graph:
        graph.nodes[node]["delta_nfr"] = pressure


def _refresh_scaled_pure_epi_pressure(graph: nx.Graph) -> None:
    scale = float(graph.graph["PRESSURE_SCALE"])
    _refresh_pure_epi_pressure(graph)
    for node in graph:
        graph.nodes[node]["delta_nfr"] *= scale


def _graph(
    *,
    current: tuple[float, ...] = (1.0, -1.0),
    edge_weight: float = 1.0,
) -> nx.Graph:
    graph = nx.path_graph(len(current))
    for left, right in graph.edges:
        graph.edges[left, right]["weight"] = edge_weight
    graph.graph.update(
        _t=0.0,
        DT_MIN=0.0,
        RANDOM_SEED=23,
        GAMMA={"type": "none"},
        _gamma_spec={"type": "none"},
        DNFR_WEIGHTS={
            "phase": 0.0,
            "epi": 1.0,
            "vf": 0.0,
            "topo": 0.0,
        },
        REMESH_TAU_GLOBAL=1,
        REMESH_TAU_LOCAL=1,
        REMESH_ALPHA=0.5,
        REMESH_ALPHA_HARD=True,
        REMESH_LOG_EVENTS=False,
        EPI_MIN=-1.0e300,
        EPI_MAX=1.0e300,
        CLIP_MODE="hard",
        ZHIR_THRESHOLD_XI=0.0,
        compute_delta_nfr=_refresh_pure_epi_pressure,
    )
    for node, epi in zip(graph, current, strict=True):
        graph.nodes[node].update(
            EPI=epi,
            epi_kind="test",
            nu_f=1.0,
            theta=0.0,
            delta_nfr=0.0,
            glyph_history=[],
        )
    graph.graph["_epi_hist"] = deque(
        [{node: value for node, value in enumerate(current)}],
        maxlen=64,
    )
    _refresh_pure_epi_pressure(graph)
    return graph


def _cycle(
    segment_durations: tuple[float, ...],
    *,
    current: tuple[float, ...] = (1.0, -1.0),
    edge_weight: float = 1.0,
    operators: tuple[str, ...] = (),
    positive_interval_indices: tuple[int, ...] | None = None,
    include_stage_certificates: bool = True,
    initial_pressure_offset: float = 0.0,
    metric_weights: tuple[float, ...] | None = None,
    integrator_method: str = "euler",
    dt_min: float = 0.0,
    pressure_scale: float | None = None,
):
    graph = _graph(current=current, edge_weight=edge_weight)
    graph.graph["INTEGRATOR_METHOD"] = integrator_method
    graph.graph["DT_MIN"] = dt_min
    if operators:
        graph.graph["compute_delta_nfr"] = _refresh_positive_time_pressure
        _refresh_positive_time_pressure(graph)
        partition_indices = positive_interval_indices or (3,)
        flow_durations = tuple(
            0.5 if index in partition_indices else 0.0
            for index in range(len(operators) + 1)
        )
    else:
        flow_durations = (0.5,)
        partition_indices = (0,)
        if pressure_scale is not None:
            graph.graph["PRESSURE_SCALE"] = pressure_scale
            graph.graph["compute_delta_nfr"] = _refresh_scaled_pure_epi_pressure
    if initial_pressure_offset:
        for node in graph:
            graph.nodes[node]["delta_nfr"] += initial_pressure_offset
    schedule = build_operator_event_schedule(
        operators,
        start_time=0.0,
        flow_durations=flow_durations,
    )
    partitions = tuple(
        build_physical_flow_partition(
            schedule.intervals[partition_index],
            segment_durations,
        )
        for partition_index in partition_indices
    )
    return execute_event_remesh_cycle(
        graph,
        schedule,
        metric_weights=metric_weights,
        physical_flow_partitions=partitions,
        include_stage_certificates=include_stage_certificates,
        suppress_birth_warnings=True,
    )


def _three_cycles():
    return (
        _cycle((0.25, 0.25)),
        _cycle((0.125, 0.125, 0.125, 0.125)),
        _cycle((0.0625,) * 8),
    )


@pytest.fixture(scope="module")
def base_cycles():
    return _three_cycles()


@pytest.fixture(scope="module")
def zhir_cycles():
    return (
        _cycle((0.25, 0.25), operators=_ZHIR_WORD),
        _cycle((0.125,) * 4, operators=_ZHIR_WORD),
        _cycle((0.0625,) * 8, operators=_ZHIR_WORD),
    )


def test_three_mesh_observation_keeps_finite_claim_boundaries(base_cycles) -> None:
    result = observe_event_remesh_three_mesh_refinement(*base_cycles)

    assert type(result) is EventRemeshThreeMeshRefinementObservation
    assert result.three_mesh_observation_certified
    assert result.persistent_nodes == (0, 1)
    assert result.supports_equal_across_meshes
    assert result.coarse_to_intermediate_strict_refinement
    assert result.intermediate_to_fine_strict_refinement
    assert result.intermediate_fine_error_decreases_at_coarse_checkpoints
    assert len(result.coarse.checkpoints) == 6
    assert len(result.intermediate.checkpoints) == 8
    assert len(result.fine.checkpoints) == 12
    assert len(result.schedule_compositions) == 3
    assert all(item is not None for item in result.schedule_compositions)
    assert len(result.remesh_results) == 3

    modal = result.modal_observations[0]
    assert modal.common_generator_modal_factors_observed
    assert modal.common_decay_rates == pytest.approx((2.0,))
    assert modal.coarse_composed_modal_factors == pytest.approx((0.25,))
    assert modal.intermediate_composed_modal_factors == pytest.approx(
        (0.75**4,)
    )
    assert modal.fine_composed_modal_factors == pytest.approx((0.875**8,))
    assert modal.coarse_segment_stability_decisions == (True, True)
    assert modal.intermediate_segment_stability_decisions == (True,) * 4
    assert modal.fine_segment_stability_decisions == (True,) * 8
    assert modal.composed_stability_decisions_agree
    assert result.zhir_observations == ()
    assert result.zhir_abstention_reason == "schedule_has_no_zhir_event"

    assert not result.solver_accuracy_certified
    assert not result.solver_order_certified
    assert not result.mesh_convergence_certified
    assert not result.lyapunov_decrease_certified
    assert not result.runtime_global_gain_certified
    assert not result.future_or_repeated_behavior_certified
    assert not result.combined_schedule_remesh_gain_certified
    assert not result.whole_three_mesh_atomicity_certified
    assert not result.complete_reference_problem_certified
    assert not result.epi_differences_attributable_only_to_mesh_certified


def test_physical_zhir_rates_and_gates_are_compared_at_both_windows(
    zhir_cycles,
) -> None:
    result = observe_event_remesh_three_mesh_refinement(*zhir_cycles)

    assert result.three_mesh_observation_certified
    assert result.zhir_abstention_reason is None
    assert len(result.zhir_observations) == 1
    zhir = result.zhir_observations[0]
    assert zhir.physical_zhir_comparison_certified
    assert zhir.coarse_observation.common_execution_provenance_certified
    assert zhir.intermediate_observation.common_execution_provenance_certified
    assert zhir.fine_observation.common_execution_provenance_certified
    assert zhir.coarse_observation.execution_result is zhir_cycles[0].event_execution
    assert zhir.coarse_observation.glyph_stage.event.event_index == 3
    assert zhir.nodes == (0, 1)
    assert zhir.coarse_terminal_exact_binary64_rates
    assert zhir.intermediate_terminal_exact_binary64_rates
    assert zhir.fine_terminal_exact_binary64_rates
    assert zhir.coarse_terminal_gate_decisions == (True, True)
    assert zhir.intermediate_terminal_gate_decisions == (True, True)
    assert zhir.fine_terminal_gate_decisions == (True, True)
    assert zhir.terminal_gate_decisions_agree
    assert zhir.whole_parent_gate_decisions_agree
    assert (
        zhir.exact_intermediate_fine_terminal_rate_error_linf
        < zhir.exact_coarse_intermediate_terminal_rate_error_linf
    )


def test_each_zhir_is_selected_by_event_index_with_multiple_events(
    monkeypatch,
) -> None:
    flow_durations = tuple(
        0.5 if index in (3, 7) else 0.0
        for index in range(len(_TWO_ZHIR_WORD) + 1)
    )
    schedule = build_operator_event_schedule(
        _TWO_ZHIR_WORD,
        start_time=0.0,
        flow_durations=flow_durations,
    )
    executions = tuple(
        SimpleNamespace(
            label=label,
            schedule=schedule,
            stage_certification_requested=True,
        )
        for label in ("coarse", "intermediate", "fine")
    )
    meshes = tuple(
        SimpleNamespace(cycle_result=SimpleNamespace(event_execution=execution))
        for execution in executions
    )
    selected: list[tuple[str, int]] = []

    def fake_executed_observer(execution, *, event_index):
        selected.append((execution.label, event_index))
        return (execution.label, event_index)

    def fake_three_mesh_observation(coarse, intermediate, fine, *_):
        assert coarse[1] == intermediate[1] == fine[1]
        return coarse[1]

    monkeypatch.setattr(
        refinement_module,
        "observe_executed_event_local_zhir_physical_prejump",
        fake_executed_observer,
    )
    monkeypatch.setattr(
        refinement_module,
        "_zhir_observation",
        fake_three_mesh_observation,
    )
    observations, abstention = refinement_module._physical_zhir_observations(
        meshes[0],
        meshes[1],
        meshes[2],
        (0,),
        (("int", 0),),
        None,
    )

    assert observations == (3, 7)
    assert abstention is None
    assert selected == [
        ("coarse", 3),
        ("intermediate", 3),
        ("fine", 3),
        ("coarse", 7),
        ("intermediate", 7),
        ("fine", 7),
    ]


def test_zhir_abstains_when_any_cycle_lacks_stage_certificates(
    zhir_cycles,
) -> None:
    coarse_without_stages = _cycle(
        (0.25, 0.25),
        operators=_ZHIR_WORD,
        include_stage_certificates=False,
    )
    result = observe_event_remesh_three_mesh_refinement(
        coarse_without_stages,
        zhir_cycles[1],
        zhir_cycles[2],
    )

    assert result.three_mesh_observation_certified
    assert result.zhir_observations == ()
    assert result.zhir_abstention_reason == (
        "zhir_stage_certificates_not_available_for_all_meshes"
    )


def test_declared_zhir_threshold_must_match_the_executed_stages(
    zhir_cycles,
) -> None:
    with pytest.raises(ValueError, match="executed Mutation threshold"):
        observe_event_remesh_three_mesh_refinement(
            *zhir_cycles,
            zhir_xi=0.5,
        )


def test_non_nested_physical_meshes_are_rejected(base_cycles) -> None:
    coarse, _, fine = base_cycles
    intermediate = _cycle((0.125, 0.375))

    with pytest.raises(ValueError, match="strictly refine coarse"):
        observe_event_remesh_three_mesh_refinement(
            coarse,
            intermediate,
            fine,
        )


def test_tampered_cycle_evidence_is_rejected(base_cycles) -> None:
    coarse, intermediate, fine = base_cycles
    old_final_time = coarse.event_execution.final_time
    object.__setattr__(coarse.event_execution, "final_time", 9.0)
    try:
        with pytest.raises(ValueError, match="unsealed, tampered, or stale"):
            observe_event_remesh_three_mesh_refinement(
                coarse,
                intermediate,
                fine,
            )
    finally:
        object.__setattr__(coarse.event_execution, "final_time", old_final_time)


def test_incompatible_initial_epi_endpoints_are_rejected(base_cycles) -> None:
    coarse, _, fine = base_cycles
    intermediate = _cycle(
        (0.125, 0.125, 0.25),
        current=(2.0, -2.0),
    )
    with pytest.raises(ValueError, match="EPI endpoints are incompatible"):
        observe_event_remesh_three_mesh_refinement(
            coarse,
            intermediate,
            fine,
        )


def test_incompatible_initial_pressure_is_rejected(base_cycles) -> None:
    coarse, _, fine = base_cycles
    intermediate = _cycle(
        (0.125, 0.125, 0.25),
        initial_pressure_offset=0.25,
    )
    with pytest.raises(ValueError, match="initial pressure differs"):
        observe_event_remesh_three_mesh_refinement(
            coarse,
            intermediate,
            fine,
        )


def test_hidden_pressure_callback_parameter_changes_are_rejected() -> None:
    coarse = _cycle((0.25, 0.25), pressure_scale=1.0)
    intermediate = _cycle((0.125, 0.125, 0.25), pressure_scale=2.0)
    fine = _cycle(
        (0.0625, 0.0625, 0.125, 0.25),
        pressure_scale=1.0,
    )

    assert coarse.pressure_before_schedule == intermediate.pressure_before_schedule
    assert (
        coarse.event_execution.physical_flow_partition_evidence[0]
        .boundary_observations[0]
        .callback_name
        == intermediate.event_execution.physical_flow_partition_evidence[0]
        .boundary_observations[0]
        .callback_name
    )
    with pytest.raises(ValueError, match="post-refresh nodal inputs differ"):
        observe_event_remesh_three_mesh_refinement(
            coarse,
            intermediate,
            fine,
        )


def test_incompatible_effective_conductance_is_rejected(base_cycles) -> None:
    coarse, _, fine = base_cycles
    intermediate = _cycle(
        (0.125, 0.125, 0.25),
        edge_weight=2.0,
    )
    with pytest.raises(ValueError, match="topology or effective conductance"):
        observe_event_remesh_three_mesh_refinement(
            coarse,
            intermediate,
            fine,
        )


def test_metrics_are_compared_by_their_exact_normalized_ray(base_cycles) -> None:
    coarse, _, fine = base_cycles
    scaled_intermediate = _cycle(
        (0.125, 0.125, 0.25),
        metric_weights=(2.0, 2.0),
    )
    accepted = observe_event_remesh_three_mesh_refinement(
        coarse,
        scaled_intermediate,
        fine,
    )
    assert accepted.three_mesh_observation_certified

    incompatible_intermediate = _cycle(
        (0.125, 0.125, 0.25),
        metric_weights=(1.0, 2.0),
    )
    with pytest.raises(ValueError, match="normalized cycle metric differs"):
        observe_event_remesh_three_mesh_refinement(
            coarse,
            incompatible_intermediate,
            fine,
        )


def test_incompatible_integrator_metadata_is_rejected(base_cycles) -> None:
    coarse, _, fine = base_cycles
    intermediate = _cycle(
        (0.125, 0.125, 0.25),
        integrator_method="rk4",
    )
    with pytest.raises(ValueError, match="executor or integrator metadata differs"):
        observe_event_remesh_three_mesh_refinement(
            coarse,
            intermediate,
            fine,
        )


def test_common_integrator_policy_allows_mesh_dependent_resolved_substeps() -> None:
    coarse = _cycle((0.25, 0.25), dt_min=0.1)
    intermediate = _cycle((0.125,) * 4, dt_min=0.1)
    fine = _cycle((0.0625,) * 8, dt_min=0.1)

    resolved = tuple(
        tuple(
            flow.resolved_substeps
            for evidence in (
                cycle.event_execution.physical_flow_partition_evidence
            )
            for flow in evidence.segment_flow_evidence
        )
        for cycle in (coarse, intermediate, fine)
    )
    assert resolved == ((2, 2), (1, 1, 1, 1), (1,) * 8)

    observation = observe_event_remesh_three_mesh_refinement(
        coarse,
        intermediate,
        fine,
    )
    assert observation.three_mesh_observation_certified
    assert (
        "executor_integrator_metadata_compatible",
        True,
    ) in observation.conditions


def test_different_full_support_is_rejected_even_with_persistent_nodes(
    base_cycles,
) -> None:
    coarse, intermediate, _ = base_cycles
    fine = _cycle(
        (0.0625, 0.0625, 0.125, 0.125, 0.125),
        current=(1.0, -1.0, 0.0),
    )
    with pytest.raises(ValueError, match="identical ordered full support"):
        observe_event_remesh_three_mesh_refinement(
            coarse,
            intermediate,
            fine,
        )


def test_nested_tampering_invalidates_the_three_mesh_seal(base_cycles) -> None:
    result = observe_event_remesh_three_mesh_refinement(*base_cycles)
    checkpoint = result.coarse.checkpoints[0]
    object.__setattr__(checkpoint, "exact_time", Fraction(9))

    assert not result.three_mesh_observation_certified
    assert result.failed_conditions == ("three_mesh_proof_fields_intact",)
    assert result.maximum_coarse_fine_epi_error_linf is None


def test_identity_hash_nodes_do_not_collide_in_support_tokens() -> None:
    class Node:
        pass

    left = Node()
    right = Node()

    tokens = refinement_module._node_tokens((left, right), "support")

    assert len(tokens) == 2
    assert tokens[0] != tokens[1]


def test_tampered_proof_stamp_comparison_does_not_dispatch_equality(
    base_cycles,
) -> None:
    result = observe_event_remesh_three_mesh_refinement(*base_cycles)
    calls = 0

    class Probe:
        def __eq__(self, other: object) -> bool:
            nonlocal calls
            calls += 1
            return False

    object.__setattr__(result, "_proof_stamp", (Probe(),))

    assert not result.three_mesh_observation_certified
    assert calls == 0
