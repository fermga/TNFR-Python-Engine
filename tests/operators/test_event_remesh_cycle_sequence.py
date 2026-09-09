"""Pure composition of consecutive event/REMESH cycle observations."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
from fractions import Fraction

import networkx as nx
import pytest

from tnfr.operators.event_remesh_runtime import (
    EventRemeshCycleResult,
    execute_event_remesh_cycle,
)
from tnfr.operators.event_remesh_sequence import (
    EventRemeshCycleBoundaryObservation,
    ObservedEventRemeshCycleSequence,
    compose_event_remesh_cycle_observations,
)
from tnfr.operators.event_timing import build_operator_event_schedule


def _set_pure_epi_pressure(graph: nx.Graph) -> None:
    values = {node: graph.nodes[node]["EPI"] for node in graph}
    for node in graph:
        neighbors = tuple(graph.neighbors(node))
        neighbor_mean = sum(values[item] for item in neighbors) / len(neighbors)
        graph.nodes[node]["delta_nfr"] = neighbor_mean - values[node]


def _graph(
    *,
    current: tuple[float, float] = (2.0, 0.0),
    past: tuple[float, float] = (0.0, 2.0),
    vf: tuple[float, float] = (1.0, 1.0),
    alpha: float = 0.5,
) -> nx.Graph:
    graph = nx.path_graph(2)
    graph.graph.update(
        _t=0.0,
        RANDOM_SEED=23,
        _gamma_spec={"type": "none"},
        REMESH_TAU_GLOBAL=1,
        REMESH_TAU_LOCAL=1,
        REMESH_ALPHA=alpha,
        REMESH_ALPHA_HARD=True,
        REMESH_LOG_EVENTS=False,
        EPI_MIN=-10.0,
        EPI_MAX=10.0,
        CLIP_MODE="hard",
        DT_MIN=0.0,
        compute_delta_nfr=_set_pure_epi_pressure,
    )
    for node, epi in enumerate(current):
        graph.nodes[node].update(
            EPI=epi,
            nu_f=vf[node],
            theta=0.0,
            delta_nfr=0.0,
            glyph_history=[],
        )
    graph.graph["_epi_hist"] = deque(
        [{node: value for node, value in enumerate(past)}],
        maxlen=64,
    )
    _set_pure_epi_pressure(graph)
    return graph


def _schedule(
    graph: nx.Graph,
    *,
    operators: tuple[str, ...] = (),
    durations: tuple[float, ...] = (0.0,),
):
    return build_operator_event_schedule(
        operators,
        start_time=graph.graph["_t"],
        flow_durations=durations,
    )


def _certified_cycle(
    graph: nx.Graph,
    *,
    metric_weights: tuple[float, float] = (1.0, 1.0),
) -> EventRemeshCycleResult:
    return execute_event_remesh_cycle(
        graph,
        _schedule(graph, durations=(0.125,)),
        metric_weights=metric_weights,
        refresh_pressure_after_remesh=True,
        include_stage_certificates=True,
    )


def _two_certified_cycles(
    *,
    metric_left: tuple[float, float] = (1.0, 1.0),
    metric_right: tuple[float, float] = (1.0, 1.0),
    vf: tuple[float, float] = (1.0, 1.0),
) -> tuple[EventRemeshCycleResult, EventRemeshCycleResult]:
    graph = _graph(vf=vf)
    left = _certified_cycle(graph, metric_weights=metric_left)
    right = _certified_cycle(graph, metric_weights=metric_right)
    return left, right


def test_two_committed_cycles_bind_every_recorded_exact_boundary() -> None:
    left, right = _two_certified_cycles()

    sequence = compose_event_remesh_cycle_observations(
        cycle for cycle in (left, right)
    )

    assert isinstance(sequence, ObservedEventRemeshCycleSequence)
    assert sequence.cycles == (left, right)
    assert sequence.cycle_indices == (0, 1)
    assert sequence.schedule_compositions == (
        left.event_execution.represented_epi_schedule_composition,
        right.event_execution.represented_epi_schedule_composition,
    )
    assert sequence.remesh_results == (left.remesh, right.remesh)
    assert len(sequence.boundaries) == 1
    boundary = sequence.boundaries[0]
    assert isinstance(boundary, EventRemeshCycleBoundaryObservation)
    assert boundary._proof_fields_are_intact()
    assert boundary.exact_recorded_state_continuity_certified
    assert boundary.failed_conditions == ()
    assert sequence._proof_fields_are_intact()
    assert sequence.exact_recorded_boundary_continuity_certified
    assert sequence.exact_common_metric_cycle_sequence_certified
    assert sequence.failed_conditions == ()
    assert sequence.per_cycle_proof_fields_intact == (True, True)
    assert sequence.per_cycle_graph_state_atomic == (True, True)
    assert sequence.per_cycle_metric_bound == (True, True)
    assert sequence.raw_metric_weights_equal
    assert sequence.nested_schedule_metric_alignment == (True, True)
    assert sequence.all_nested_schedule_metrics_aligned
    assert sequence.exact_common_normalized_metric_ray == (
        Fraction(1, 2),
        Fraction(1, 2),
    )

    assert not sequence.mixed_schedule_remesh_gain_certified
    assert not sequence.evolving_history_remesh_gain_certified
    assert not sequence.evolving_history_remesh_repetition_certified
    assert not sequence.runtime_global_gain_certified
    assert not sequence.future_cycle_stability_certified
    assert not sequence.whole_sequence_atomicity_certified
    assert not sequence.full_graph_state_continuity_certified
    assert not sequence.grammar_history_continuity_certified
    assert not sequence.shared_graph_execution_provenance_certified
    assert "hybrid_event_log" in sequence.scope
    assert "epi_time_history" in sequence.scope


def test_proportional_metrics_share_a_ray_but_not_raw_energy_scale() -> None:
    left, right = _two_certified_cycles(
        metric_left=(1.0, 2.0),
        metric_right=(2.0, 4.0),
        vf=(1.0, 0.5),
    )

    sequence = compose_event_remesh_cycle_observations((left, right))

    assert sequence.exact_recorded_boundary_continuity_certified
    assert sequence.exact_common_metric_cycle_sequence_certified
    assert sequence.cycle_exact_normalized_metric_rays == (
        (Fraction(1, 3), Fraction(2, 3)),
        (Fraction(1, 3), Fraction(2, 3)),
    )
    assert sequence.exact_common_normalized_metric_ray == (
        Fraction(1, 3),
        Fraction(2, 3),
    )
    assert not sequence.raw_metric_weights_equal
    assert sequence.nested_schedule_metric_alignment == (True, True)
    assert sequence.all_nested_schedule_metrics_aligned


def test_nonproportional_metrics_abstain_from_common_geometry() -> None:
    left, right = _two_certified_cycles(
        metric_left=(1.0, 1.0),
        metric_right=(1.0, 3.0),
    )

    sequence = compose_event_remesh_cycle_observations((left, right))

    assert sequence.boundaries[0].exact_recorded_state_continuity_certified
    assert sequence.exact_recorded_boundary_continuity_certified
    assert sequence.exact_common_normalized_metric_ray is None
    assert not sequence.exact_common_metric_cycle_sequence_certified
    assert sequence.failed_conditions == (
        "one_exact_normalized_metric_ray",
        "every_nested_schedule_exposes_the_common_metric",
    )
    assert sequence.nested_schedule_metric_alignment == (True, False)
    assert not sequence.all_nested_schedule_metrics_aligned


def test_changed_remesh_requires_completed_refresh_before_next_cycle() -> None:
    graph = _graph()
    left = execute_event_remesh_cycle(graph, _schedule(graph))
    right = execute_event_remesh_cycle(graph, _schedule(graph))

    sequence = compose_event_remesh_cycle_observations((left, right))
    boundary = sequence.boundaries[0]

    assert boundary.exact_left_pre_remesh_epi != (
        boundary.exact_left_post_remesh_epi
    )
    assert not boundary.left_post_remesh_pressure_refresh_requested
    assert (
        boundary.left_post_remesh_pressure_refresh_callback_invocations == 0
    )
    assert not boundary.exact_recorded_state_continuity_certified
    assert boundary.failed_conditions == (
        "changed_applied_remesh_has_completed_pressure_refresh",
    )
    assert not sequence.exact_recorded_boundary_continuity_certified


def test_configuration_equality_is_diagnostic_only() -> None:
    graph = _graph()
    left = _certified_cycle(graph)
    graph.graph["REMESH_ALPHA"] = 0.25
    right = _certified_cycle(graph)

    sequence = compose_event_remesh_cycle_observations((left, right))

    assert sequence.exact_common_metric_cycle_sequence_certified
    assert not sequence.remesh_configurations_equal
    assert not sequence.evolving_history_remesh_repetition_certified
    assert "remesh_configurations_equal" not in dict(sequence.conditions)


def test_full_history_mismatch_is_reported_at_the_exact_boundary() -> None:
    left_graph = _graph()
    left = execute_event_remesh_cycle(
        left_graph,
        _schedule(left_graph),
        refresh_pressure_after_remesh=True,
    )
    right_graph = _graph(
        current=left.post_remesh_epi.epi_values,
        past=(9.0, 9.0),
    )
    right = execute_event_remesh_cycle(
        right_graph,
        _schedule(right_graph),
        refresh_pressure_after_remesh=True,
    )

    sequence = compose_event_remesh_cycle_observations((left, right))
    boundary = sequence.boundaries[0]

    assert (
        boundary.exact_left_post_remesh_epi
        == boundary.exact_right_pre_schedule_epi
    )
    assert (
        boundary.left_outgoing_exact_history
        != boundary.right_incoming_exact_history
    )
    assert boundary.failed_conditions == (
        "full_exact_remesh_history_continuous",
    )
    assert not sequence.exact_recorded_boundary_continuity_certified


def test_proof_seals_fail_closed_after_boundary_or_cycle_tampering() -> None:
    left, right = _two_certified_cycles()
    sequence = compose_event_remesh_cycle_observations((left, right))
    boundary = sequence.boundaries[0]

    object.__setattr__(
        boundary,
        "exact_left_post_remesh_epi",
        (Fraction(99), Fraction(99)),
    )
    assert not boundary._proof_fields_are_intact()
    assert not boundary.exact_recorded_state_continuity_certified
    assert boundary.failed_conditions == ("boundary_proof_fields_intact",)
    assert not sequence._proof_fields_are_intact()
    assert not sequence.exact_common_metric_cycle_sequence_certified
    assert sequence.failed_conditions == ("sequence_proof_fields_intact",)

    clean_left, clean_right = _two_certified_cycles()
    clean = compose_event_remesh_cycle_observations(
        (clean_left, clean_right)
    )
    with pytest.raises(ValueError, match="diagnostics"):
        replace(clean, raw_metric_weights_equal=False)
    with pytest.raises(AttributeError):
        object.__setattr__(clean, "runtime_global_gain_certified", True)
    assert not clean.runtime_global_gain_certified

    object.__setattr__(
        clean_left,
        "pressure_after_optional_refresh",
        (99.0, 99.0),
    )
    assert not clean_left._proof_fields_are_intact()
    assert not clean._proof_fields_are_intact()
    assert not clean.exact_common_metric_cycle_sequence_certified


@pytest.mark.parametrize("cycles", [(), (object(),)])
def test_composer_requires_two_exact_cycle_results(
    cycles: tuple[object, ...],
) -> None:
    with pytest.raises(ValueError, match="at least two"):
        compose_event_remesh_cycle_observations(cycles)


def test_composer_rejects_noncycle_when_cardinality_is_sufficient() -> None:
    left, _ = _two_certified_cycles()
    with pytest.raises(TypeError, match=r"cycles\[1\]"):
        compose_event_remesh_cycle_observations((left, object()))


def test_alpha_one_history_evolution_refutes_gain_multiplication() -> None:
    graph = _graph(alpha=1.0)
    initial = tuple(graph.nodes[node]["EPI"] for node in graph)

    first = execute_event_remesh_cycle(
        graph,
        _schedule(graph),
        refresh_pressure_after_remesh=True,
    )
    second = execute_event_remesh_cycle(
        graph,
        _schedule(graph),
        refresh_pressure_after_remesh=True,
    )
    sequence = compose_event_remesh_cycle_observations((first, second))

    assert first.post_remesh_epi.epi_values == (0.0, 2.0)
    assert second.post_remesh_epi.epi_values == initial
    assert first.remesh.evidence is not None
    assert second.remesh.evidence is not None
    assert first.remesh.evidence.beta == 0.0
    assert second.remesh.evidence.beta == 0.0
    assert first.pre_schedule_epi.exact_disagreement_energy == 1
    assert second.post_remesh_epi.exact_disagreement_energy == 1
    assert sequence.exact_recorded_boundary_continuity_certified
    assert not sequence.exact_common_metric_cycle_sequence_certified
    assert sequence.failed_conditions == (
        "every_nested_schedule_exposes_the_common_metric",
    )
    assert not hasattr(sequence, "exact_energy_gain_upper_bound")
    assert not sequence.mixed_schedule_remesh_gain_certified
    assert not sequence.evolving_history_remesh_gain_certified
    assert not sequence.evolving_history_remesh_repetition_certified


class _EqIterable:
    def __init__(
        self,
        cycles: tuple[EventRemeshCycleResult, EventRemeshCycleResult],
    ) -> None:
        self.cycles = cycles

    def __iter__(self):
        return iter(self.cycles)

    def __eq__(self, other: object) -> bool:
        raise RuntimeError("iterable equality must not be invoked")


class _ExplodingTuple(tuple):
    def __eq__(self, other: object) -> bool:
        raise RuntimeError("tuple equality must not be invoked")


class _ExplodingEqualityNode:
    explode = False

    def __init__(self, label: int) -> None:
        self.label = label

    def __hash__(self) -> int:
        return hash(self.label)

    def __eq__(self, other: object) -> bool:
        if type(self).explode:
            raise RuntimeError("node equality must not be invoked")
        return (
            type(other) is _ExplodingEqualityNode
            and self.label == other.label
        )


@dataclass(eq=False, frozen=True, slots=True)
class _IdentityOnlyEmptyNode:
    def __deepcopy__(self, memo: dict[int, object]):
        return self


def _graph_with_nodes(
    nodes: tuple[_ExplodingEqualityNode, _ExplodingEqualityNode],
    *,
    current: tuple[float, float],
    history: tuple[tuple[float, float], ...],
) -> nx.Graph:
    graph = _graph(current=current)
    nx.relabel_nodes(graph, {0: nodes[0], 1: nodes[1]}, copy=False)
    graph.graph["_epi_hist"] = deque(
        [
            {nodes[index]: row[index] for index in range(2)}
            for row in history
        ],
        maxlen=64,
    )
    _set_pure_epi_pressure(graph)
    return graph


def test_input_iterable_equality_is_never_consulted() -> None:
    left, right = _two_certified_cycles()

    sequence = compose_event_remesh_cycle_observations(
        _EqIterable((left, right))
    )

    assert sequence.exact_common_metric_cycle_sequence_certified


def test_hostile_node_equality_fails_closed_without_leaking_runtime_error() -> None:
    left_nodes = (_ExplodingEqualityNode(0), _ExplodingEqualityNode(1))
    left_graph = _graph_with_nodes(
        left_nodes,
        current=(2.0, 0.0),
        history=((0.0, 2.0),),
    )
    left = execute_event_remesh_cycle(
        left_graph,
        _schedule(left_graph),
        refresh_pressure_after_remesh=True,
    )
    incoming = tuple(
        tuple(float(value) for value in row)
        for row in left.history_transition.outgoing_exact_history
    )
    right_nodes = (_ExplodingEqualityNode(0), _ExplodingEqualityNode(1))
    right_graph = _graph_with_nodes(
        right_nodes,
        current=left.post_remesh_epi.epi_values,
        history=incoming,
    )
    right = execute_event_remesh_cycle(
        right_graph,
        _schedule(right_graph),
        refresh_pressure_after_remesh=True,
    )

    _ExplodingEqualityNode.explode = True
    try:
        with pytest.raises(ValueError, match="proof fields are not intact"):
            compose_event_remesh_cycle_observations((left, right))
    finally:
        _ExplodingEqualityNode.explode = False


def test_conditions_require_exact_pairs_and_canonical_order() -> None:
    left, right = _two_certified_cycles()
    sequence = compose_event_remesh_cycle_observations((left, right))
    boundary = sequence.boundaries[0]

    with pytest.raises(ValueError, match="canonical order"):
        replace(boundary, conditions=tuple(reversed(boundary.conditions)))
    with pytest.raises(TypeError, match="exact tuple"):
        replace(
            sequence,
            conditions=_ExplodingTuple(sequence.conditions),
        )


def test_nested_records_require_cycle_owned_identity() -> None:
    left, right = _two_certified_cycles()
    sequence = compose_event_remesh_cycle_observations((left, right))
    first_schedule = sequence.schedule_compositions[0]
    assert first_schedule is not None
    forged_schedule = replace(first_schedule)
    assert forged_schedule is not first_schedule

    with pytest.raises(ValueError, match="cycle-owned identity"):
        replace(
            sequence,
            schedule_compositions=(
                forged_schedule,
                sequence.schedule_compositions[1],
            ),
        )

    forged_remesh = replace(sequence.remesh_results[0])
    assert forged_remesh is not sequence.remesh_results[0]
    with pytest.raises(ValueError, match="cycle-owned identity"):
        replace(
            sequence,
            remesh_results=(forged_remesh, sequence.remesh_results[1]),
        )


def test_scope_is_sealed_for_boundaries_and_sequences() -> None:
    left, right = _two_certified_cycles()
    sequence = compose_event_remesh_cycle_observations((left, right))
    boundary = sequence.boundaries[0]

    object.__setattr__(boundary, "scope", "forged stronger boundary theorem")
    assert not boundary._proof_fields_are_intact()
    assert not boundary.exact_recorded_state_continuity_certified

    clean_left, clean_right = _two_certified_cycles()
    clean = compose_event_remesh_cycle_observations(
        (clean_left, clean_right)
    )
    object.__setattr__(clean, "scope", "forged causal sequence theorem")
    assert not clean._proof_fields_are_intact()
    assert not clean.exact_common_metric_cycle_sequence_certified


def test_same_cycle_identity_cannot_masquerade_as_two_observations() -> None:
    left, _ = _two_certified_cycles()

    with pytest.raises(ValueError, match="distinct identities"):
        compose_event_remesh_cycle_observations((left, left))


def test_structurally_empty_identity_nodes_do_not_match_across_supports() -> None:
    left_nodes = (_IdentityOnlyEmptyNode(), _IdentityOnlyEmptyNode())
    left_graph = _graph_with_nodes(
        left_nodes,  # type: ignore[arg-type]
        current=(2.0, 0.0),
        history=((0.0, 2.0),),
    )
    left = execute_event_remesh_cycle(
        left_graph,
        _schedule(left_graph),
        refresh_pressure_after_remesh=True,
    )
    incoming = tuple(
        tuple(float(value) for value in row)
        for row in left.history_transition.outgoing_exact_history
    )
    right_nodes = (_IdentityOnlyEmptyNode(), _IdentityOnlyEmptyNode())
    right_graph = _graph_with_nodes(
        right_nodes,  # type: ignore[arg-type]
        current=left.post_remesh_epi.epi_values,
        history=incoming,
    )
    right = execute_event_remesh_cycle(
        right_graph,
        _schedule(right_graph),
        refresh_pressure_after_remesh=True,
    )

    sequence = compose_event_remesh_cycle_observations((left, right))

    assert sequence.boundaries[0].failed_conditions == (
        "ordered_node_support_continuous",
    )
    assert not sequence.exact_recorded_boundary_continuity_certified


def test_nested_metric_false_takes_precedence_over_missing_evidence() -> None:
    graph = _graph()
    left = _certified_cycle(graph, metric_weights=(1.0, 3.0))
    right = execute_event_remesh_cycle(
        graph,
        _schedule(graph),
        metric_weights=(1.0, 3.0),
        refresh_pressure_after_remesh=True,
    )

    sequence = compose_event_remesh_cycle_observations((left, right))

    assert sequence.nested_schedule_metric_alignment == (False, None)
    assert sequence.all_nested_schedule_metrics_aligned is False
