"""Causal finite execution of operator-event/REMESH cycle sequences."""

from __future__ import annotations

from collections import deque
from fractions import Fraction

import networkx as nx
import pytest

import tnfr.operators.event_remesh_causal_runtime as causal_module
import tnfr.physics.runtime_remesh_schedule_stability as telescope_module
from tnfr.errors import TNFRValueError
from tnfr.operators.event_remesh_causal_runtime import (
    CausalEventRemeshCycleReceipt,
    EventRemeshCycleExecutionSpec,
    ExecutedEventRemeshCycleSequence,
    execute_event_remesh_cycle_sequence,
)
from tnfr.operators.event_remesh_sequence import (
    compose_event_remesh_cycle_observations,
)
from tnfr.operators.event_timing import (
    build_operator_event_schedule,
    build_physical_flow_partition,
)
from tnfr.utils._structural_signature import structural_proof_signature


def _set_pure_epi_pressure(graph: nx.Graph) -> None:
    values = {node: float(graph.nodes[node]["EPI"]) for node in graph}
    for node in graph:
        neighbours = tuple(graph.neighbors(node))
        graph.nodes[node]["delta_nfr"] = (
            sum(values[item] for item in neighbours) / len(neighbours)
            - values[node]
        )


def _graph(*, alpha: float = 0.5) -> nx.Graph:
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
    for node, epi in enumerate((2.0, 0.0)):
        graph.nodes[node].update(
            EPI=epi,
            nu_f=1.0,
            theta=0.0,
            delta_nfr=0.0,
            glyph_history=[],
        )
    graph.graph["_epi_hist"] = deque([{0: 0.0, 1: 2.0}], maxlen=64)
    _set_pure_epi_pressure(graph)
    return graph


def _specs(
    *,
    count: int = 2,
    duration: float = 0.125,
    partitioned: bool = False,
) -> tuple[EventRemeshCycleExecutionSpec, ...]:
    specs: list[EventRemeshCycleExecutionSpec] = []
    start = 0.0
    for _index in range(count):
        schedule = build_operator_event_schedule(
            (),
            start_time=start,
            flow_durations=(duration,),
        )
        if partitioned:
            partition = build_physical_flow_partition(
                schedule.intervals[0],
                (duration / 2.0, duration / 2.0),
            )
            physical = (partition,)
        else:
            physical = ()
        specs.append(
            EventRemeshCycleExecutionSpec(
                schedule,
                physical_flow_partitions=physical,
            )
        )
        start = schedule.end_time
    return tuple(specs)


def _graph_signature(graph: nx.Graph) -> tuple[object, ...]:
    return (
        structural_proof_signature(graph),
        id(graph.graph),
        id(graph._node),
        id(graph._adj),
        tuple(id(graph.nodes[node]) for node in graph),
        id(graph.graph["_epi_hist"]),
    )


def test_graph_validation_does_not_read_a_virtual_class_attribute() -> None:
    referenced_graph = _graph()
    before = _graph_signature(referenced_graph)
    reads = 0

    class HostileGraphProxy:
        @property
        def __class__(self):
            nonlocal reads
            reads += 1
            referenced_graph.graph["pre_snapshot_leak"] = True
            return nx.Graph

    with pytest.raises(TypeError, match="NetworkX graph"):
        execute_event_remesh_cycle_sequence(
            HostileGraphProxy(),  # type: ignore[arg-type]
            (),
        )

    assert reads == 0
    assert _graph_signature(referenced_graph) == before
    assert "pre_snapshot_leak" not in referenced_graph.graph


def test_two_cycles_receive_same_invocation_provenance_and_telescope() -> None:
    graph = _graph()
    specs = _specs()

    result = execute_event_remesh_cycle_sequence(
        graph,
        (spec for spec in specs),
        metric_weights=(1.0, 1.0),
    )

    assert type(result) is ExecutedEventRemeshCycleSequence
    assert result.cycle_indices == (0, 1)
    assert all(
        type(receipt) is CausalEventRemeshCycleReceipt
        and receipt.receipt_binding_certified
        for receipt in result.receipts
    )
    assert all(
        receipt.spec is specs[index]
        and receipt.schedule is specs[index].schedule
        and receipt.cycle_result is result.cycles[index]
        and receipt.cycle_result.event_execution.schedule
        is specs[index].schedule
        for index, receipt in enumerate(result.receipts)
    )
    assert result.observed_sequence.cycles[0] is result.cycles[0]
    assert result.observed_sequence.cycles[1] is result.cycles[1]
    assert result.runtime_telescope.source_sequence is result.observed_sequence
    assert result.exact_start_time == Fraction(0)
    assert result.exact_end_time == Fraction(1, 4)
    assert graph.graph["_t"] == 0.25

    assert result.causal_cycle_order_certified
    assert result.same_graph_execution_provenance_certified
    assert result.whole_sequence_graph_state_atomic
    assert result.exact_recorded_boundary_continuity_certified
    assert result.exact_finite_energy_telescope_certified
    assert result.failed_conditions == ()

    # The pure observers retain their deliberately weaker offline contract.
    assert not result.observed_sequence.shared_graph_execution_provenance_certified
    assert not result.observed_sequence.whole_sequence_atomicity_certified
    assert not result.runtime_telescope.shared_graph_execution_provenance_certified
    assert not result.runtime_telescope.whole_sequence_atomicity_certified

    assert not result.runtime_global_gain_certified
    assert not result.repeated_runtime_stability_certified
    assert not result.future_stability_certified
    assert not result.solver_accuracy_certified
    assert not result.solver_order_certified
    assert not result.mesh_convergence_certified
    assert not result.adaptive_u2_u4_policy_certified
    assert not result.full_tnfr_stability_certified
    assert not result.full_graph_state_continuity_certified
    assert not result.grammar_history_continuity_certified
    assert not result.concurrent_writer_atomicity_certified
    assert not result.cryptographic_or_durable_provenance_certified
    assert not result.external_side_effects_rolled_back


def test_three_partitioned_cycles_bind_every_physical_partition_identity() -> None:
    graph = _graph()
    specs = _specs(count=3, partitioned=True)

    result = execute_event_remesh_cycle_sequence(graph, specs)

    assert result.cycle_indices == (0, 1, 2)
    assert len(result.runtime_telescope.boundaries) == 2
    assert result.runtime_telescope.exact_total_energy_drop == (
        result.runtime_telescope.exact_augmented_energy_initial
        - result.runtime_telescope.exact_augmented_energy_final
    )
    for index, receipt in enumerate(result.receipts):
        partition = tuple(specs[index].physical_flow_partitions)[0]
        assert receipt.physical_flow_partitions[0] is partition
        evidence = receipt.cycle_result.event_execution
        assert evidence.physical_flow_partition_evidence[0].partition is partition


def test_outer_transaction_precedes_specs_iterable_materialization() -> None:
    graph = _graph()
    before = _graph_signature(graph)
    first, second = _specs()

    def mutating_specs():
        graph.graph["materialization_leak"] = ["must roll back"]
        graph.nodes[0]["EPI"] = 99.0
        yield first
        yield second

    with pytest.raises(
        TNFRValueError,
        match="observational materialization changed graph state",
    ):
        execute_event_remesh_cycle_sequence(graph, mutating_specs())

    assert _graph_signature(graph) == before
    assert "materialization_leak" not in graph.graph


def test_outer_transaction_precedes_partition_iterable_materialization() -> None:
    graph = _graph()
    before = _graph_signature(graph)
    schedules = tuple(spec.schedule for spec in _specs())

    def mutating_partitions():
        graph.add_edge(0, 0, injected=True)
        return
        yield  # pragma: no cover

    specs = (
        EventRemeshCycleExecutionSpec(
            schedules[0],
            physical_flow_partitions=mutating_partitions(),
        ),
        EventRemeshCycleExecutionSpec(schedules[1]),
    )
    with pytest.raises(
        TNFRValueError,
        match="observational materialization changed graph state",
    ):
        execute_event_remesh_cycle_sequence(graph, specs)

    assert _graph_signature(graph) == before
    assert not graph.has_edge(0, 0)


def test_second_cycle_failure_rolls_back_first_committed_cycle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = _graph()
    before = _graph_signature(graph)
    original = causal_module.execute_event_remesh_cycle
    calls = 0

    def fail_after_first(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            graph.add_node("transient")
            graph.graph["_t"] = 99.0
            raise RuntimeError("second cycle failed")
        return original(*args, **kwargs)

    monkeypatch.setattr(causal_module, "execute_event_remesh_cycle", fail_after_first)
    with pytest.raises(RuntimeError, match="second cycle failed"):
        execute_event_remesh_cycle_sequence(graph, _specs())

    assert calls == 2
    assert _graph_signature(graph) == before
    assert "transient" not in graph


def test_observer_mutation_after_all_cycles_triggers_total_rollback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = _graph()
    before = _graph_signature(graph)

    def mutate_during_composition(cycles):
        graph.graph["observer_leak"] = object()
        return compose_event_remesh_cycle_observations(cycles)

    monkeypatch.setattr(
        causal_module,
        "compose_event_remesh_cycle_observations",
        mutate_during_composition,
    )
    with pytest.raises(
        TNFRValueError,
        match="observational materialization changed graph state",
    ):
        execute_event_remesh_cycle_sequence(graph, _specs())

    assert _graph_signature(graph) == before
    assert "observer_leak" not in graph.graph


def test_telescope_failure_after_all_cycles_triggers_total_rollback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = _graph()
    before = _graph_signature(graph)

    def fail_telescope(_sequence):
        graph.nodes[0]["EPI"] = -999.0
        raise RuntimeError("telescope failed")

    monkeypatch.setattr(
        telescope_module,
        "observe_runtime_remesh_schedule_sequence",
        fail_telescope,
    )
    with pytest.raises(RuntimeError, match="telescope failed"):
        execute_event_remesh_cycle_sequence(graph, _specs())

    assert _graph_signature(graph) == before


def test_discontinuous_or_reused_schedules_fail_before_cycle_execution() -> None:
    graph = _graph()
    first = _specs()[0]
    discontinuous = build_operator_event_schedule(
        (),
        start_time=0.5,
        flow_durations=(0.125,),
    )
    before = _graph_signature(graph)

    with pytest.raises(TNFRValueError, match="clock chain"):
        execute_event_remesh_cycle_sequence(
            graph,
            (first, EventRemeshCycleExecutionSpec(discontinuous)),
        )
    assert _graph_signature(graph) == before

    reused = EventRemeshCycleExecutionSpec(first.schedule)
    with pytest.raises(TNFRValueError, match="schedule.*distinct"):
        execute_event_remesh_cycle_sequence(graph, (first, reused))
    assert _graph_signature(graph) == before


def test_nested_tampering_invalidates_receipt_and_outer_result() -> None:
    result = execute_event_remesh_cycle_sequence(_graph(), _specs())
    receipt = result.receipts[0]
    object.__setattr__(
        receipt.cycle_result,
        "exact_total_weighted_mean_drift",
        Fraction(99),
    )

    assert not receipt.receipt_binding_certified
    assert not result.causal_cycle_order_certified
    assert result.failed_conditions == ("executed_sequence_proof_fields_intact",)


def test_spec_partition_source_replacement_invalidates_nested_and_outer_seals() -> None:
    specs = list(_specs())
    original_source: list[object] = []
    specs[0] = EventRemeshCycleExecutionSpec(
        specs[0].schedule,
        physical_flow_partitions=original_source,
    )
    result = execute_event_remesh_cycle_sequence(_graph(), specs)
    object.__setattr__(specs[0], "physical_flow_partitions", [])

    assert not result.receipts[0].receipt_binding_certified
    assert not result.same_graph_execution_provenance_certified


@pytest.mark.parametrize("specs", [(), (object(), object())])
def test_invalid_spec_collections_fail_closed(specs: tuple[object, ...]) -> None:
    graph = _graph()
    before = _graph_signature(graph)
    error = TNFRValueError if not specs else TypeError
    with pytest.raises(error):
        execute_event_remesh_cycle_sequence(graph, specs)  # type: ignore[arg-type]
    assert _graph_signature(graph) == before
