"""Executor-bound evidence for explicit physical pressure-refresh partitions."""

from __future__ import annotations

import threading
from collections import defaultdict
from copy import deepcopy
from dataclasses import replace
from fractions import Fraction
from functools import partial
from random import Random

import networkx as nx
import pytest

from tnfr.dynamics.integrators import AbstractIntegrator
from tnfr.errors import TNFRValueError
from tnfr.mathematics.unified_numerical import np
from tnfr.operators.event_runtime import (
    ExecutedPressureRefreshedFlowPartition,
    execute_operator_event_schedule,
)
from tnfr.operators.event_timing import (
    build_operator_event_schedule,
    build_physical_flow_partition,
)
from tnfr.operators.network_stage import GraphTransactionSnapshot


def _pure_epi_graph(*, uniform: bool = False) -> nx.Graph:
    graph = nx.path_graph(2)
    graph.graph.update(
        _t=0.0,
        DT_MIN=0.0,
        GAMMA={"type": "none"},
        EPI_MIN=-1.0e300,
        EPI_MAX=1.0e300,
        DNFR_WEIGHTS={
            "phase": 0.0,
            "epi": 1.0,
            "vf": 0.0,
            "topo": 0.0,
        },
    )
    epi = (1.0, 1.0) if uniform else (1.0, -1.0)
    for node, value in zip(graph, epi, strict=True):
        graph.nodes[node].update(
            EPI=value,
            epi_kind="test",
            nu_f=1.0,
            theta=0.0,
            delta_nfr=0.0,
            glyph_history=[],
        )
    return graph


def _partitioned_empty_schedule():
    schedule = build_operator_event_schedule(
        (),
        start_time=0.0,
        flow_durations=(0.5,),
    )
    partition = build_physical_flow_partition(
        schedule.intervals[0],
        (0.25, 0.25),
    )
    return schedule, partition


def test_physical_partition_refreshes_pressure_and_changes_the_k2_endpoint() -> None:
    graph = _pure_epi_graph()
    schedule, partition = _partitioned_empty_schedule()

    result = execute_operator_event_schedule(
        graph,
        schedule,
        physical_flow_partitions=(partition,),
        suppress_birth_warnings=True,
    )

    assert tuple(float(graph.nodes[node]["EPI"]) for node in graph) == (
        0.25,
        -0.25,
    )
    assert graph.graph["_t"] == 0.5
    assert result.flow_certification_requested
    assert result.flow_interval_evidence == ()
    assert result.physical_flow_partition_indices == (0,)
    assert result.pressure_refresh_callback_invocations == 3
    assert result.physical_pressure_refresh_callback_invocations == 3
    assert result.stage_pressure_refresh_callback_invocations == 0
    assert result.physical_pressure_reevaluated_partitions_established is True
    evidence = result.physical_flow_partition_evidence[0]
    assert type(evidence) is ExecutedPressureRefreshedFlowPartition
    assert evidence.physical_pressure_reevaluated_partition_established
    assert len(evidence.boundary_observations) == 3
    assert all(
        boundary.edge_state_preserved
        and boundary.graph_configuration_preserved
        and boundary.dnfr_weights_preserved_or_canonically_initialized
        for boundary in evidence.boundary_observations
    )
    assert len(evidence.segment_flow_evidence) == 2
    assert evidence.all_segment_binary64_intervals_identified
    assert evidence.all_segment_binary64_replays_identified
    assert evidence.all_segment_exact_affine_maps_identified
    assert evidence.all_segment_disagreement_contractions_certified
    assert evidence.all_boundaries_binary64_pure_epi_pressure_realized
    assert evidence.all_segment_modal_diagnostics_applicable
    assert evidence.all_segment_modal_decisions_stable
    assert evidence.exact_common_metric == (Fraction(1, 2), Fraction(1, 2))
    assert evidence.exact_segment_gain_bounds == (
        Fraction(1, 4),
        Fraction(1, 4),
    )
    assert evidence.exact_composed_gain_bound == Fraction(1, 16)
    assert evidence.exact_common_metric_gain_product_certified
    assert result.all_positive_flow_intervals_binary64_identified is True
    assert result.all_positive_flow_intervals_exact_affine is True
    assert result.all_positive_flow_intervals_contracting is True
    assert tuple(graph.nodes[1]["epi_time_history"]) == (
        (0.0, -1.0),
        (0.25, -0.5),
        (0.5, -0.25),
    )


@pytest.mark.parametrize(
    "deleted_slot",
    ("_proof_stamp", "callback_binding_preserved"),
)
def test_pressure_boundary_fails_closed_when_a_slot_is_deleted(
    deleted_slot: str,
) -> None:
    graph = _pure_epi_graph()
    schedule, partition = _partitioned_empty_schedule()
    result = execute_operator_event_schedule(
        graph,
        schedule,
        physical_flow_partitions=(partition,),
        suppress_birth_warnings=True,
    )
    boundary = result.physical_flow_partition_evidence[0].boundary_observations[0]
    assert boundary._proof_fields_are_intact()
    assert boundary.callback_completed

    object.__delattr__(boundary, deleted_slot)

    assert boundary._proof_fields_are_intact() is False
    assert boundary.callback_completed is False
    assert boundary.nonpressure_state_preserved is False
    assert result._proof_fields_are_intact() is False
    assert result.runtime_clock_checked is False


def test_executed_zhir_reads_the_terminal_physical_segment_secant() -> None:
    graph = _pure_epi_graph()
    graph.graph["ZHIR_THRESHOLD_XI"] = 0.0

    def refresh(live_graph: nx.Graph) -> None:
        pressure = 1.0 if float(live_graph.graph["_t"]) < 0.25 else 2.0
        for node in live_graph:
            live_graph.nodes[node]["delta_nfr"] = pressure

    graph.graph["compute_delta_nfr"] = refresh
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
    partition = build_physical_flow_partition(
        schedule.intervals[3],
        (0.25, 0.25),
    )

    result = execute_operator_event_schedule(
        graph,
        schedule,
        include_stage_certificates=True,
        physical_flow_partitions=(partition,),
        suppress_birth_warnings=True,
    )

    physical = result.physical_flow_partition_evidence[0]
    initial = physical.boundary_observations[0].after
    terminal = physical.boundary_observations[-1].after
    terminal_certificate = physical.segment_flow_evidence[-1].certificate
    assert terminal_certificate is not None
    terminal_left = terminal_certificate.left
    terminal_rates = tuple(
        (right - left) / 0.25
        for left, right in zip(
            terminal_left.epi,
            terminal.epi,
            strict=True,
        )
    )
    whole_parent_rates = tuple(
        (right - left) / 0.5
        for left, right in zip(initial.epi, terminal.epi, strict=True)
    )
    mutation = result.glyph_stage_evidence[3]
    trigger_evidence = tuple(
        item.trigger_certificate.evidence
        for item in mutation.mutation_decision_observations
    )
    assert all(item is not None for item in trigger_evidence)
    observed = tuple(
        item.observed_depi_dt for item in trigger_evidence if item is not None
    )

    assert mutation.event.operator_name == "mutation"
    assert all(
        item.sample_interval == 0.25
        for item in trigger_evidence
        if item is not None
    )
    assert observed == terminal_rates
    assert observed != whole_parent_rates


def test_physical_refresh_rejects_signed_zero_as_pure_epi_pressure() -> None:
    graph = _pure_epi_graph(uniform=True)

    def signed_zero_pressure(live_graph: nx.Graph) -> None:
        for node in live_graph:
            live_graph.nodes[node]["delta_nfr"] = -0.0

    graph.graph["compute_delta_nfr"] = signed_zero_pressure
    schedule, partition = _partitioned_empty_schedule()

    result = execute_operator_event_schedule(
        graph,
        schedule,
        physical_flow_partitions=(partition,),
    )
    evidence = result.physical_flow_partition_evidence[0]

    assert evidence.physical_pressure_reevaluated_partition_established
    assert not evidence.all_boundaries_binary64_pure_epi_pressure_realized
    assert not evidence.all_segment_modal_diagnostics_applicable
    assert all(not item.available for item in evidence.modal_observations)
    assert all(
        item.abstention_reason
        == "refreshed_pressure_is_not_binary64_pure_epi_diffusion"
        for item in evidence.modal_observations
    )


def test_terminal_nonpure_pressure_does_not_erase_segment_modal_evidence() -> None:
    graph = _pure_epi_graph()

    def refresh(live_graph: nx.Graph) -> None:
        if float(live_graph.graph["_t"]) == 0.5:
            live_graph.nodes[0]["delta_nfr"] = 1.0
            live_graph.nodes[1]["delta_nfr"] = 1.0
            return
        left = float(live_graph.nodes[0]["EPI"])
        right = float(live_graph.nodes[1]["EPI"])
        live_graph.nodes[0]["delta_nfr"] = right - left
        live_graph.nodes[1]["delta_nfr"] = left - right

    graph.graph["compute_delta_nfr"] = refresh
    schedule, partition = _partitioned_empty_schedule()

    result = execute_operator_event_schedule(
        graph,
        schedule,
        physical_flow_partitions=(partition,),
    )

    evidence = result.physical_flow_partition_evidence[0]
    assert not evidence.all_boundaries_binary64_pure_epi_pressure_realized
    assert evidence.all_segment_modal_diagnostics_applicable
    assert evidence.all_segment_binary64_replays_identified
    assert evidence.all_segment_modal_decisions_stable


def test_callback_nonpressure_mutation_fails_and_rolls_back_graph_state() -> None:
    graph = _pure_epi_graph()
    schedule, partition = _partitioned_empty_schedule()
    calls = []

    def invalid_pressure_callback(live_graph: nx.Graph) -> None:
        calls.append(float(live_graph.graph["_t"]))
        live_graph.nodes[0]["EPI"] = 99.0
        live_graph.nodes[0]["delta_nfr"] = -1.0

    graph.graph["compute_delta_nfr"] = invalid_pressure_callback
    graph_before = deepcopy(dict(graph.graph))
    nodes_before = {
        node: deepcopy(dict(data)) for node, data in graph.nodes(data=True)
    }

    with pytest.raises(TNFRValueError, match="non-pressure graph state"):
        execute_operator_event_schedule(
            graph,
            schedule,
            physical_flow_partitions=(partition,),
        )

    assert calls == []
    assert dict(graph.graph) == graph_before
    assert {
        node: dict(data) for node, data in graph.nodes(data=True)
    } == nodes_before


def test_callback_owned_state_mutation_fails_and_rolls_back() -> None:
    class StatefulPressureCallback:
        def __init__(self) -> None:
            self.calls: list[float] = []

        def __call__(self, live_graph: nx.Graph) -> None:
            self.calls.append(float(live_graph.graph["_t"]))
            left = float(live_graph.nodes[0]["EPI"])
            right = float(live_graph.nodes[1]["EPI"])
            live_graph.nodes[0]["delta_nfr"] = right - left
            live_graph.nodes[1]["delta_nfr"] = left - right

    graph = _pure_epi_graph()
    callback = StatefulPressureCallback()
    graph.graph["compute_delta_nfr"] = callback
    schedule, partition = _partitioned_empty_schedule()

    with pytest.raises(TNFRValueError, match="non-pressure graph state"):
        execute_operator_event_schedule(
            graph,
            schedule,
            physical_flow_partitions=(partition,),
        )

    assert callback.calls == []
    assert graph.graph["compute_delta_nfr"] is callback


@pytest.mark.skipif(np is None, reason="NumPy is unavailable")
def test_callback_numpy_state_mutation_fails_and_rolls_back() -> None:
    class ArrayPressureCallback:
        def __init__(self) -> None:
            self.values = np.array([1.0, 2.0])

        def __call__(self, live_graph: nx.Graph) -> None:
            self.values[0] = 9.0
            left = float(live_graph.nodes[0]["EPI"])
            right = float(live_graph.nodes[1]["EPI"])
            live_graph.nodes[0]["delta_nfr"] = right - left
            live_graph.nodes[1]["delta_nfr"] = left - right

    graph = _pure_epi_graph()
    callback = ArrayPressureCallback()
    values = callback.values
    graph.graph["compute_delta_nfr"] = callback
    schedule, partition = _partitioned_empty_schedule()

    with pytest.raises(TNFRValueError, match="non-pressure graph state"):
        execute_operator_event_schedule(
            graph,
            schedule,
            physical_flow_partitions=(partition,),
        )

    assert callback.values is values
    assert np.array_equal(values, np.array([1.0, 2.0]))


def test_callback_random_state_mutation_fails_and_rolls_back() -> None:
    class RandomPressureCallback:
        def __init__(self) -> None:
            self.generator = Random(1234)

        def __call__(self, live_graph: nx.Graph) -> None:
            self.generator.random()
            left = float(live_graph.nodes[0]["EPI"])
            right = float(live_graph.nodes[1]["EPI"])
            live_graph.nodes[0]["delta_nfr"] = right - left
            live_graph.nodes[1]["delta_nfr"] = left - right

    graph = _pure_epi_graph()
    callback = RandomPressureCallback()
    state = callback.generator.getstate()
    graph.graph["compute_delta_nfr"] = callback
    schedule, partition = _partitioned_empty_schedule()

    with pytest.raises(TNFRValueError, match="non-pressure graph state"):
        execute_operator_event_schedule(
            graph,
            schedule,
            physical_flow_partitions=(partition,),
        )

    assert callback.generator.getstate() == state


def test_callback_defaultdict_factory_mutation_fails_and_rolls_back() -> None:
    class MappingPressureCallback:
        def __init__(self) -> None:
            self.storage = defaultdict(list, before=[1])

        def __call__(self, live_graph: nx.Graph) -> None:
            self.storage.default_factory = dict
            left = float(live_graph.nodes[0]["EPI"])
            right = float(live_graph.nodes[1]["EPI"])
            live_graph.nodes[0]["delta_nfr"] = right - left
            live_graph.nodes[1]["delta_nfr"] = left - right

    graph = _pure_epi_graph()
    callback = MappingPressureCallback()
    graph.graph["compute_delta_nfr"] = callback
    schedule, partition = _partitioned_empty_schedule()

    with pytest.raises(TNFRValueError, match="non-pressure graph state"):
        execute_operator_event_schedule(
            graph,
            schedule,
            physical_flow_partitions=(partition,),
        )

    assert callback.storage.default_factory is list
    assert callback.storage == {"before": [1]}


def test_opaque_callback_state_is_rejected_before_it_can_be_consumed() -> None:
    graph = _pure_epi_graph()
    cursor = iter((1, 2))

    def refresh(live_graph: nx.Graph) -> None:
        next(cursor)
        left = float(live_graph.nodes[0]["EPI"])
        right = float(live_graph.nodes[1]["EPI"])
        live_graph.nodes[0]["delta_nfr"] = right - left
        live_graph.nodes[1]["delta_nfr"] = left - right

    graph.graph["compute_delta_nfr"] = refresh
    schedule, partition = _partitioned_empty_schedule()

    with pytest.raises(TNFRValueError, match="unsupported mutable state"):
        execute_operator_event_schedule(
            graph,
            schedule,
            physical_flow_partitions=(partition,),
        )

    assert next(cursor) == 1


@pytest.mark.parametrize("use_partial", (False, True))
def test_bound_callback_receiver_mutation_fails_and_rolls_back(
    use_partial: bool,
) -> None:
    class PressureWorker:
        def __init__(self) -> None:
            self.calls: list[float] = []

        def refresh(self, live_graph: nx.Graph) -> None:
            self.calls.append(float(live_graph.graph["_t"]))
            left = float(live_graph.nodes[0]["EPI"])
            right = float(live_graph.nodes[1]["EPI"])
            live_graph.nodes[0]["delta_nfr"] = right - left
            live_graph.nodes[1]["delta_nfr"] = left - right

    graph = _pure_epi_graph()
    worker = PressureWorker()
    callback = partial(worker.refresh) if use_partial else worker.refresh
    graph.graph["compute_delta_nfr"] = callback
    schedule, partition = _partitioned_empty_schedule()

    with pytest.raises(TNFRValueError, match="non-pressure graph state"):
        execute_operator_event_schedule(
            graph,
            schedule,
            physical_flow_partitions=(partition,),
        )

    assert worker.calls == []
    assert graph.graph["compute_delta_nfr"] is callback


def test_function_default_state_mutation_fails_and_rolls_back() -> None:
    graph = _pure_epi_graph()
    schedule, partition = _partitioned_empty_schedule()

    def refresh(live_graph: nx.Graph, calls: list[float] = []) -> None:
        calls.append(float(live_graph.graph["_t"]))
        left = float(live_graph.nodes[0]["EPI"])
        right = float(live_graph.nodes[1]["EPI"])
        live_graph.nodes[0]["delta_nfr"] = right - left
        live_graph.nodes[1]["delta_nfr"] = left - right

    default_calls = refresh.__defaults__[0]
    graph.graph["compute_delta_nfr"] = refresh

    with pytest.raises(TNFRValueError, match="non-pressure graph state"):
        execute_operator_event_schedule(
            graph,
            schedule,
            physical_flow_partitions=(partition,),
        )

    assert refresh.__defaults__[0] is default_calls
    assert default_calls == []


def test_function_namespace_rebinding_fails_and_restores_identity() -> None:
    graph = _pure_epi_graph()
    schedule, partition = _partitioned_empty_schedule()

    def refresh(live_graph: nx.Graph) -> None:
        refresh.__dict__ = {"marker": "before"}
        left = float(live_graph.nodes[0]["EPI"])
        right = float(live_graph.nodes[1]["EPI"])
        live_graph.nodes[0]["delta_nfr"] = right - left
        live_graph.nodes[1]["delta_nfr"] = left - right

    refresh.marker = "before"
    original_namespace = refresh.__dict__
    graph.graph["compute_delta_nfr"] = refresh

    with pytest.raises(TNFRValueError, match="non-pressure graph state"):
        execute_operator_event_schedule(
            graph,
            schedule,
            physical_flow_partitions=(partition,),
        )

    assert refresh.__dict__ is original_namespace
    assert refresh.__dict__ == {"marker": "before"}


@pytest.mark.parametrize("use_partial", (False, True))
def test_callback_graph_alias_is_opaque_and_preserves_identity(
    use_partial: bool,
) -> None:
    graph = _pure_epi_graph()
    schedule, partition = _partitioned_empty_schedule()

    class PressureWorker:
        def __init__(self, owned_graph: nx.Graph) -> None:
            self.graph = owned_graph

        def refresh(self, live_graph: nx.Graph) -> None:
            assert live_graph is self.graph
            left = float(live_graph.nodes[0]["EPI"])
            right = float(live_graph.nodes[1]["EPI"])
            live_graph.nodes[0]["delta_nfr"] = right - left
            live_graph.nodes[1]["delta_nfr"] = left - right

    worker = PressureWorker(graph)
    callback = partial(worker.refresh) if use_partial else worker.refresh
    graph.graph["compute_delta_nfr"] = callback

    result = execute_operator_event_schedule(
        graph,
        schedule,
        physical_flow_partitions=(partition,),
    )

    assert worker.graph is graph
    assert graph.graph["compute_delta_nfr"] is callback
    assert result.physical_pressure_refresh_callback_invocations == 3
    assert all(
        item.callback_state_preserved
        for item in result.physical_flow_partition_evidence[
            0
        ].boundary_observations
    )


def test_callback_label_does_not_invoke_instance_attribute_access() -> None:
    graph = _pure_epi_graph()
    schedule, partition = _partitioned_empty_schedule()

    class LabelTrap:
        __slots__ = ("label_reads",)

        def __init__(self) -> None:
            self.label_reads: list[str] = []

        def __getattribute__(self, name: str):
            if name in {"__module__", "__qualname__"}:
                reads = object.__getattribute__(self, "label_reads")
                reads.append(name)
            return object.__getattribute__(self, name)

        def __call__(self, live_graph: nx.Graph) -> None:
            left = float(live_graph.nodes[0]["EPI"])
            right = float(live_graph.nodes[1]["EPI"])
            live_graph.nodes[0]["delta_nfr"] = right - left
            live_graph.nodes[1]["delta_nfr"] = left - right

    callback = LabelTrap()
    graph.graph["compute_delta_nfr"] = callback

    result = execute_operator_event_schedule(
        graph,
        schedule,
        physical_flow_partitions=(partition,),
    )

    assert callback.label_reads == []
    assert all(
        item.callback_name.endswith(".LabelTrap")
        for item in result.physical_flow_partition_evidence[
            0
        ].boundary_observations
    )


@pytest.mark.parametrize("mutation", ["graph_configuration", "edge_state"])
def test_callback_graph_or_edge_mutation_fails_and_rolls_back(
    mutation: str,
) -> None:
    graph = _pure_epi_graph()
    schedule, partition = _partitioned_empty_schedule()
    calls = []

    def invalid_pressure_callback(live_graph: nx.Graph) -> None:
        calls.append(float(live_graph.graph["_t"]))
        left = float(live_graph.nodes[0]["EPI"])
        right = float(live_graph.nodes[1]["EPI"])
        live_graph.nodes[0]["delta_nfr"] = right - left
        live_graph.nodes[1]["delta_nfr"] = left - right
        if mutation == "graph_configuration":
            live_graph.graph["GAMMA"] = {"type": "linear", "value": 0.5}
        else:
            live_graph.edges[0, 1]["length"] = 2.0

    graph.graph["compute_delta_nfr"] = invalid_pressure_callback
    graph_before = deepcopy(dict(graph.graph))
    nodes_before = {
        node: deepcopy(dict(data)) for node, data in graph.nodes(data=True)
    }
    edges_before = {
        (source, target): deepcopy(dict(data))
        for source, target, data in graph.edges(data=True)
    }

    with pytest.raises(TNFRValueError, match="non-pressure graph state"):
        execute_operator_event_schedule(
            graph,
            schedule,
            physical_flow_partitions=(partition,),
        )

    assert calls == []
    assert dict(graph.graph) == graph_before
    assert {
        node: dict(data) for node, data in graph.nodes(data=True)
    } == nodes_before
    assert {
        (source, target): dict(data)
        for source, target, data in graph.edges(data=True)
    } == edges_before


def test_callback_graph_instance_mutation_fails_and_rolls_back() -> None:
    graph = _pure_epi_graph()
    graph._last_operator_applied = "coherence"
    schedule, partition = _partitioned_empty_schedule()

    def invalid_pressure_callback(live_graph: nx.Graph) -> None:
        left = float(live_graph.nodes[0]["EPI"])
        right = float(live_graph.nodes[1]["EPI"])
        live_graph.nodes[0]["delta_nfr"] = right - left
        live_graph.nodes[1]["delta_nfr"] = left - right
        live_graph._last_operator_applied = "dissonance"
        live_graph.callback_owned_state = {"changed": True}

    graph.graph["compute_delta_nfr"] = invalid_pressure_callback

    with pytest.raises(TNFRValueError, match="non-pressure graph state"):
        execute_operator_event_schedule(
            graph,
            schedule,
            physical_flow_partitions=(partition,),
        )

    assert graph._last_operator_applied == "coherence"
    assert not hasattr(graph, "callback_owned_state")


def test_callback_graph_slot_mutation_fails_and_rolls_back() -> None:
    class SlotGraph(nx.Graph):
        __slots__ = ("custom_state",)

    source = _pure_epi_graph()
    graph = SlotGraph(source)
    graph.graph.update(source.graph)
    graph.custom_state = {"value": 1}
    schedule, partition = _partitioned_empty_schedule()

    def invalid_pressure_callback(live_graph: nx.Graph) -> None:
        left = float(live_graph.nodes[0]["EPI"])
        right = float(live_graph.nodes[1]["EPI"])
        live_graph.nodes[0]["delta_nfr"] = right - left
        live_graph.nodes[1]["delta_nfr"] = left - right
        live_graph.custom_state["value"] = 2

    graph.graph["compute_delta_nfr"] = invalid_pressure_callback

    with pytest.raises(TNFRValueError, match="non-pressure graph state"):
        execute_operator_event_schedule(
            graph,
            schedule,
            physical_flow_partitions=(partition,),
        )

    assert graph.custom_state == {"value": 1}


def test_callback_cannot_hide_redeclared_base_slot_mutation() -> None:
    class BaseSlotGraph(nx.Graph):
        __slots__ = ("custom_state",)

    class DerivedSlotGraph(BaseSlotGraph):
        __slots__ = ("custom_state",)

    source = _pure_epi_graph()
    graph = DerivedSlotGraph(source)
    graph.graph.update(source.graph)
    base_descriptor = vars(BaseSlotGraph)["custom_state"]
    derived_descriptor = vars(DerivedSlotGraph)["custom_state"]
    base_state = {"owner": "base", "value": 1}
    derived_state = {"owner": "derived", "value": 1}
    base_descriptor.__set__(graph, base_state)
    derived_descriptor.__set__(graph, derived_state)
    schedule, partition = _partitioned_empty_schedule()

    def invalid_pressure_callback(live_graph: nx.Graph) -> None:
        left = float(live_graph.nodes[0]["EPI"])
        right = float(live_graph.nodes[1]["EPI"])
        live_graph.nodes[0]["delta_nfr"] = right - left
        live_graph.nodes[1]["delta_nfr"] = left - right
        base_descriptor.__get__(live_graph, type(live_graph))["value"] = 2
        derived_descriptor.__get__(live_graph, type(live_graph))["value"] = 2

    graph.graph["compute_delta_nfr"] = invalid_pressure_callback

    with pytest.raises(TNFRValueError, match="non-pressure graph state"):
        execute_operator_event_schedule(
            graph,
            schedule,
            physical_flow_partitions=(partition,),
        )

    assert base_descriptor.__get__(graph, type(graph)) is base_state
    assert derived_descriptor.__get__(graph, type(graph)) is derived_state
    assert base_state == {"owner": "base", "value": 1}
    assert derived_state == {"owner": "derived", "value": 1}


def test_callback_cannot_rebind_one_side_of_a_protected_alias() -> None:
    graph = _pure_epi_graph()
    shared_payload = {"marker": "shared"}
    graph.nodes[0]["payload"] = shared_payload
    graph.edges[0, 1]["payload"] = shared_payload
    schedule, partition = _partitioned_empty_schedule()

    def invalid_pressure_callback(live_graph: nx.Graph) -> None:
        left = float(live_graph.nodes[0]["EPI"])
        right = float(live_graph.nodes[1]["EPI"])
        live_graph.nodes[0]["delta_nfr"] = right - left
        live_graph.nodes[1]["delta_nfr"] = left - right
        live_graph.nodes[0]["payload"] = {"marker": "shared"}

    graph.graph["compute_delta_nfr"] = invalid_pressure_callback

    with pytest.raises(TNFRValueError, match="non-pressure graph state") as failure:
        execute_operator_event_schedule(
            graph,
            schedule,
            physical_flow_partitions=(partition,),
        )

    assert "protected_identity_preserved" in failure.value.context[
        "failed_preservation_checks"
    ]
    assert graph.nodes[0]["payload"] is graph.edges[0, 1]["payload"]
    assert graph.nodes[0]["payload"] == shared_payload


def test_callback_cannot_rebind_nested_graph_instance_state() -> None:
    graph = _pure_epi_graph()
    nested = {"marker": "shared"}
    graph.custom_payload = {"nested": nested}
    schedule, partition = _partitioned_empty_schedule()

    def invalid_pressure_callback(live_graph: nx.Graph) -> None:
        left = float(live_graph.nodes[0]["EPI"])
        right = float(live_graph.nodes[1]["EPI"])
        live_graph.nodes[0]["delta_nfr"] = right - left
        live_graph.nodes[1]["delta_nfr"] = left - right
        live_graph.custom_payload["nested"] = {"marker": "shared"}

    graph.graph["compute_delta_nfr"] = invalid_pressure_callback

    with pytest.raises(TNFRValueError, match="non-pressure graph state") as failure:
        execute_operator_event_schedule(
            graph,
            schedule,
            physical_flow_partitions=(partition,),
        )

    assert "protected_identity_preserved" in failure.value.context[
        "failed_preservation_checks"
    ]
    assert graph.custom_payload["nested"] == nested


def test_callback_cannot_rebind_nested_graph_factory_state() -> None:
    class StatefulFactory:
        def __init__(self) -> None:
            self.configuration = {"nested": {"marker": "shared"}}

        def __call__(self) -> dict:
            return {}

    factory = StatefulFactory()

    class FactoryGraph(nx.Graph):
        node_attr_dict_factory = factory

    source = _pure_epi_graph()
    graph = FactoryGraph(source)
    graph.graph.update(source.graph)
    nested = factory.configuration["nested"]
    schedule, partition = _partitioned_empty_schedule()

    def invalid_pressure_callback(live_graph: nx.Graph) -> None:
        left = float(live_graph.nodes[0]["EPI"])
        right = float(live_graph.nodes[1]["EPI"])
        live_graph.nodes[0]["delta_nfr"] = right - left
        live_graph.nodes[1]["delta_nfr"] = left - right
        factory.configuration["nested"] = {"marker": "shared"}

    graph.graph["compute_delta_nfr"] = invalid_pressure_callback

    with pytest.raises(TNFRValueError, match="non-pressure graph state") as failure:
        execute_operator_event_schedule(
            graph,
            schedule,
            physical_flow_partitions=(partition,),
        )

    assert "protected_identity_preserved" in failure.value.context[
        "failed_preservation_checks"
    ]
    assert factory.configuration["nested"] == nested


@pytest.mark.parametrize("directed", (False, True))
def test_callback_cannot_break_networkx_edge_storage_alias(
    directed: bool,
) -> None:
    source = _pure_epi_graph()
    graph = nx.DiGraph(source) if directed else source
    graph.graph.update(source.graph)
    schedule, partition = _partitioned_empty_schedule()
    original_edge_storage = graph._adj[0][1]

    def invalid_pressure_callback(live_graph: nx.Graph) -> None:
        left = float(live_graph.nodes[0]["EPI"])
        right = float(live_graph.nodes[1]["EPI"])
        live_graph.nodes[0]["delta_nfr"] = right - left
        live_graph.nodes[1]["delta_nfr"] = left - right
        if directed:
            live_graph._pred[1][0] = dict(live_graph._pred[1][0])
        else:
            live_graph._adj[1][0] = dict(live_graph._adj[1][0])

    graph.graph["compute_delta_nfr"] = invalid_pressure_callback

    with pytest.raises(TNFRValueError, match="non-pressure graph state") as failure:
        execute_operator_event_schedule(
            graph,
            schedule,
            physical_flow_partitions=(partition,),
        )

    assert "protected_identity_preserved" in failure.value.context[
        "failed_preservation_checks"
    ]
    if directed:
        assert graph._adj[0][1] is graph._pred[1][0]
    else:
        assert graph._adj[0][1] is graph._adj[1][0]
    assert graph._adj[0][1] is original_edge_storage


@pytest.mark.parametrize("multigraph", (False, True))
def test_callback_cannot_break_directed_root_adjacency_alias(
    multigraph: bool,
) -> None:
    graph_type = nx.MultiDiGraph if multigraph else nx.DiGraph
    source = _pure_epi_graph()
    graph = graph_type(source)
    graph.graph.update(source.graph)
    schedule, partition = _partitioned_empty_schedule()
    original_adjacency = graph._adj

    def invalid_pressure_callback(live_graph: nx.Graph) -> None:
        left = float(live_graph.nodes[0]["EPI"])
        right = float(live_graph.nodes[1]["EPI"])
        live_graph.nodes[0]["delta_nfr"] = right - left
        live_graph.nodes[1]["delta_nfr"] = left - right
        live_graph.__dict__["_succ"] = dict(live_graph._succ)

    graph.graph["compute_delta_nfr"] = invalid_pressure_callback

    with pytest.raises(TNFRValueError, match="non-pressure graph state") as failure:
        execute_operator_event_schedule(
            graph,
            schedule,
            physical_flow_partitions=(partition,),
        )

    assert "protected_identity_preserved" in failure.value.context[
        "failed_preservation_checks"
    ]
    assert graph._adj is graph._succ
    assert graph._adj is original_adjacency


@pytest.mark.parametrize(
    "attribute_name",
    ("nodes", "adj", "edges", "__networkx_cache__"),
)
def test_callback_cannot_corrupt_networkx_cached_surfaces(
    attribute_name: str,
) -> None:
    graph = _pure_epi_graph()
    if attribute_name == "__networkx_cache__":
        original = graph.__dict__[attribute_name]
    else:
        original = getattr(graph, attribute_name)
    schedule, partition = _partitioned_empty_schedule()

    def invalid_pressure_callback(live_graph: nx.Graph) -> None:
        left = float(live_graph.nodes[0]["EPI"])
        right = float(live_graph.nodes[1]["EPI"])
        live_graph.nodes[0]["delta_nfr"] = right - left
        live_graph.nodes[1]["delta_nfr"] = left - right
        if float(live_graph.graph["_t"]) == 0.5:
            live_graph.__dict__[attribute_name] = "CORRUPT"

    graph.graph["compute_delta_nfr"] = invalid_pressure_callback

    with pytest.raises(TNFRValueError, match="non-pressure graph state") as failure:
        execute_operator_event_schedule(
            graph,
            schedule,
            physical_flow_partitions=(partition,),
        )

    assert "protected_identity_preserved" in failure.value.context[
        "failed_preservation_checks"
    ]
    assert graph.__dict__[attribute_name] is original


def test_callback_cannot_install_an_invalid_previously_absent_cached_view() -> None:
    graph = _pure_epi_graph()
    graph.__dict__.pop("edges", None)
    schedule, partition = _partitioned_empty_schedule()

    def invalid_pressure_callback(live_graph: nx.Graph) -> None:
        left = float(live_graph.nodes[0]["EPI"])
        right = float(live_graph.nodes[1]["EPI"])
        live_graph.nodes[0]["delta_nfr"] = right - left
        live_graph.nodes[1]["delta_nfr"] = left - right
        if float(live_graph.graph["_t"]) == 0.5:
            live_graph.__dict__["edges"] = "CORRUPT"

    graph.graph["compute_delta_nfr"] = invalid_pressure_callback

    with pytest.raises(TNFRValueError, match="non-pressure graph state"):
        execute_operator_event_schedule(
            graph,
            schedule,
            physical_flow_partitions=(partition,),
        )

    assert "edges" not in graph.__dict__


@pytest.mark.parametrize(
    "mutation",
    (
        "graph_mapping",
        "node_attribute_mapping",
        "edge_attribute_mapping",
        "factory",
        "node_outer_mapping",
        "adjacency_outer_mapping",
        "adjacency_inner_mapping",
        "edge_key_mapping",
    ),
)
def test_callback_factory_mapping_state_mutation_fails_and_rolls_back(
    mutation: str,
) -> None:
    class NamespacedDict(dict):
        pass

    class StatefulMapping(NamespacedDict):
        __slots__ = ("slot_state",)

        def __init__(self, label: str) -> None:
            super().__init__()
            self.namespace_state = {"label": label, "value": 1}
            self.slot_state = {"label": label, "value": 1}

    class StatefulFactory:
        __slots__ = ("slot_state", "__dict__")

        def __init__(self, label: str) -> None:
            self.label = label
            self.namespace_state = {"label": label, "value": 1}
            self.slot_state = {"label": label, "value": 1}

        def __call__(self) -> StatefulMapping:
            return StatefulMapping(self.label)

    factories = {
        name: StatefulFactory(name)
        for name in (
            "graph",
            "node_outer",
            "node_attribute",
            "adjacency_outer",
            "adjacency_inner",
            "edge_key",
            "edge_attribute",
        )
    }

    class StatefulMultiGraph(nx.MultiGraph):
        graph_attr_dict_factory = factories["graph"]
        node_dict_factory = factories["node_outer"]
        node_attr_dict_factory = factories["node_attribute"]
        adjlist_outer_dict_factory = factories["adjacency_outer"]
        adjlist_inner_dict_factory = factories["adjacency_inner"]
        edge_key_dict_factory = factories["edge_key"]
        edge_attr_dict_factory = factories["edge_attribute"]

    graph = StatefulMultiGraph()
    graph.graph.update(
        _t=0.0,
        DT_MIN=0.0,
        GAMMA={"type": "none"},
        EPI_MIN=-1.0e300,
        EPI_MAX=1.0e300,
        DNFR_WEIGHTS={
            "phase": 0.0,
            "epi": 1.0,
            "vf": 0.0,
            "topo": 0.0,
        },
    )
    for node, value in ((0, 1.0), (1, -1.0)):
        graph.add_node(
            node,
            EPI=value,
            epi_kind="test",
            nu_f=1.0,
            theta=0.0,
            delta_nfr=0.0,
            glyph_history=[],
        )
    graph.add_edge(0, 1, key="edge", weight=1.0)
    schedule, partition = _partitioned_empty_schedule()
    targets = {
        "graph_mapping": graph.graph,
        "node_attribute_mapping": graph._node[0],
        "edge_attribute_mapping": graph._adj[0][1]["edge"],
        "factory": factories["node_attribute"],
        "node_outer_mapping": graph._node,
        "adjacency_outer_mapping": graph._adj,
        "adjacency_inner_mapping": graph._adj[0],
        "edge_key_mapping": graph._adj[0][1],
    }
    target = targets[mutation]

    class StatefulPressureCallback:
        __slots__ = ("calls", "__dict__")

        def __init__(self) -> None:
            self.calls = []
            self.namespace_state = {"value": 1}

        def __call__(self, live_graph: nx.Graph) -> None:
            self.calls.append(float(live_graph.graph["_t"]))
            left = float(live_graph.nodes[0]["EPI"])
            right = float(live_graph.nodes[1]["EPI"])
            live_graph.nodes[0]["delta_nfr"] = right - left
            live_graph.nodes[1]["delta_nfr"] = left - right
            target.namespace_state["value"] = 2
            target.slot_state["value"] = 2

    callback = StatefulPressureCallback()
    graph.graph["compute_delta_nfr"] = callback

    with pytest.raises(TNFRValueError, match="non-pressure graph state"):
        execute_operator_event_schedule(
            graph,
            schedule,
            physical_flow_partitions=(partition,),
        )

    restored_targets = {
        "graph_mapping": graph.graph,
        "node_attribute_mapping": graph._node[0],
        "edge_attribute_mapping": graph._adj[0][1]["edge"],
        "factory": factories["node_attribute"],
        "node_outer_mapping": graph._node,
        "adjacency_outer_mapping": graph._adj,
        "adjacency_inner_mapping": graph._adj[0],
        "edge_key_mapping": graph._adj[0][1],
    }
    assert restored_targets[mutation] is target
    assert target.namespace_state["value"] == 1
    assert target.slot_state["value"] == 1
    assert graph.graph["compute_delta_nfr"] is callback
    assert callback.calls == []


def test_internal_mapping_snapshot_does_not_copy_stateful_node_keys() -> None:
    class LockedNode:
        def __init__(self, label: str) -> None:
            self.label = label
            self.lock = threading.Lock()

    left = LockedNode("left")
    right = LockedNode("right")
    graph = nx.Graph()
    graph.add_edge(left, right, weight=1.0)
    graph.graph.update(
        _t=0.0,
        DT_MIN=0.0,
        GAMMA={"type": "none"},
        EPI_MIN=-1.0e300,
        EPI_MAX=1.0e300,
        DNFR_WEIGHTS={
            "phase": 0.0,
            "epi": 1.0,
            "vf": 0.0,
            "topo": 0.0,
        },
    )
    for node, value in ((left, 1.0), (right, -1.0)):
        graph.nodes[node].update(
            EPI=value,
            epi_kind="test",
            nu_f=1.0,
            theta=0.0,
            delta_nfr=0.0,
            glyph_history=[],
        )
    schedule, partition = _partitioned_empty_schedule()

    def invalid_pressure_callback(live_graph: nx.Graph) -> None:
        live_graph.graph["GAMMA"] = {"type": "linear", "value": 0.5}

    graph.graph["compute_delta_nfr"] = invalid_pressure_callback

    with pytest.raises(TNFRValueError, match="non-pressure graph state"):
        execute_operator_event_schedule(
            graph,
            schedule,
            physical_flow_partitions=(partition,),
        )

    assert tuple(graph.nodes) == (left, right)
    assert graph.has_edge(left, right)
    assert graph.graph["GAMMA"] == {"type": "none"}


@pytest.mark.parametrize(
    "graph_type",
    (nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph),
)
def test_snapshot_restores_factory_mapping_identity_for_all_graph_kinds(
    graph_type,
) -> None:
    graph = graph_type()
    graph.add_node(0, marker="left")
    graph.add_node(1, marker="right")
    if graph.is_multigraph():
        graph.add_edge(0, 1, key="shared", marker="forward")
        if graph.is_directed():
            graph.add_edge(1, 0, key="shared", marker="reverse")
    else:
        graph.add_edge(0, 1, marker="forward")
        if graph.is_directed():
            graph.add_edge(1, 0, marker="reverse")
    graph.graph["marker"] = "before"

    node_outer = graph._node
    adjacency_outer = graph._adj
    predecessor_outer = graph._pred if graph.is_directed() else None
    node_mappings = {node: graph._node[node] for node in graph}
    adjacency_inner = {node: graph._adj[node] for node in graph}
    predecessor_inner = (
        {node: graph._pred[node] for node in graph}
        if graph.is_directed()
        else {}
    )
    edge_mappings = {
        (source, target, key): data
        for source, target, key, data in (
            graph.edges(keys=True, data=True)
            if graph.is_multigraph()
            else (
                (source, target, None, data)
                for source, target, data in graph.edges(data=True)
            )
        )
    }
    edge_key_mappings = (
        {
            (node, neighbor): graph._adj[node][neighbor]
            for node in graph
            for neighbor in graph._adj[node]
        }
        if graph.is_multigraph()
        else {}
    )
    snapshot = GraphTransactionSnapshot(graph)

    graph.clear()
    graph.add_edge("new", "topology", marker="after")
    graph.graph["marker"] = "after"
    snapshot.restore(graph)

    assert graph._node is node_outer
    assert graph._adj is adjacency_outer
    if graph.is_directed():
        assert graph._succ is adjacency_outer
        assert graph._pred is predecessor_outer
    assert tuple(graph.nodes) == (0, 1)
    assert graph.graph["marker"] == "before"
    assert all(graph._node[node] is node_mappings[node] for node in graph)
    assert all(graph._adj[node] is adjacency_inner[node] for node in graph)
    assert all(
        graph._pred[node] is predecessor_inner[node] for node in predecessor_inner
    )
    for (source, target, key), data in edge_mappings.items():
        restored = (
            graph._adj[source][target][key]
            if graph.is_multigraph()
            else graph._adj[source][target]
        )
        assert restored is data
    assert all(
        graph._adj[node][neighbor] is edge_keys
        for (node, neighbor), edge_keys in edge_key_mappings.items()
    )


def test_callback_cannot_change_existing_cached_dnfr_weights() -> None:
    graph = _pure_epi_graph()
    graph.graph["_dnfr_weights"] = {
        "phase": 0.0,
        "epi": 1.0,
        "vf": 0.0,
        "topo": 0.0,
    }
    schedule, partition = _partitioned_empty_schedule()

    def invalid_pressure_callback(live_graph: nx.Graph) -> None:
        live_graph.graph["_dnfr_weights"]["epi"] = 0.5

    graph.graph["compute_delta_nfr"] = invalid_pressure_callback
    graph_before = deepcopy(dict(graph.graph))

    with pytest.raises(TNFRValueError, match="non-pressure graph state"):
        execute_operator_event_schedule(
            graph,
            schedule,
            physical_flow_partitions=(partition,),
        )

    assert dict(graph.graph) == graph_before


def test_callback_failure_at_later_boundary_rolls_back_prior_segments() -> None:
    graph = _pure_epi_graph()
    schedule, partition = _partitioned_empty_schedule()

    def failing_pressure_callback(live_graph: nx.Graph) -> None:
        if float(live_graph.graph["_t"]) == 0.25:
            raise RuntimeError("boundary refresh failed")
        left = float(live_graph.nodes[0]["EPI"])
        right = float(live_graph.nodes[1]["EPI"])
        live_graph.nodes[0]["delta_nfr"] = right - left
        live_graph.nodes[1]["delta_nfr"] = left - right

    graph.graph["compute_delta_nfr"] = failing_pressure_callback
    nodes_before = {
        node: deepcopy(dict(data)) for node, data in graph.nodes(data=True)
    }

    with pytest.raises(RuntimeError, match="boundary refresh failed"):
        execute_operator_event_schedule(
            graph,
            schedule,
            physical_flow_partitions=(partition,),
        )

    assert graph.graph["_t"] == 0.0
    assert {
        node: dict(data) for node, data in graph.nodes(data=True)
    } == nodes_before


def test_partition_evidence_and_ordered_result_fail_closed_after_tampering() -> None:
    graph = _pure_epi_graph()
    schedule, partition = _partitioned_empty_schedule()
    result = execute_operator_event_schedule(
        graph,
        schedule,
        physical_flow_partitions=(partition,),
    )
    evidence = result.physical_flow_partition_evidence[0]

    segment_identities = tuple(id(item) for item in evidence.partition.segments)
    evidence.partition.__post_init__()
    assert tuple(id(item) for item in evidence.partition.segments) == (
        segment_identities
    )
    assert evidence.physical_pressure_reevaluated_partition_established
    altered = replace(evidence, exact_composed_gain_bound=Fraction(1))
    assert not altered.physical_pressure_reevaluated_partition_established
    reordered = replace(
        evidence,
        segment_flow_evidence=tuple(reversed(evidence.segment_flow_evidence)),
    )
    assert not reordered.physical_pressure_reevaluated_partition_established
    with pytest.raises(ValueError, match="proof fields"):
        replace(result, physical_flow_partition_indices=())
    with pytest.raises(ValueError, match="proof fields"):
        replace(result, physical_flow_partition_evidence=(altered,))
    with pytest.raises(ValueError, match="proof fields"):
        replace(result, pressure_refresh_callback_invocations=4)


def test_execution_result_rejects_physical_evidence_from_another_schedule() -> None:
    graph = _pure_epi_graph()
    schedule, partition = _partitioned_empty_schedule()
    result = execute_operator_event_schedule(
        graph,
        schedule,
        physical_flow_partitions=(partition,),
    )
    other_schedule = build_operator_event_schedule(
        (),
        start_time=0.0,
        flow_durations=(1.0,),
    )

    with pytest.raises(
        ValueError,
        match="proof fields",
    ):
        replace(
            result,
            schedule=other_schedule,
            final_time=other_schedule.end_time,
        )


def test_tiny_capacity_commits_with_explicit_modal_abstention() -> None:
    graph = _pure_epi_graph()
    for node in graph:
        graph.nodes[node]["nu_f"] = 5e-324
    schedule, partition = _partitioned_empty_schedule()

    result = execute_operator_event_schedule(
        graph,
        schedule,
        physical_flow_partitions=(partition,),
    )

    evidence = result.physical_flow_partition_evidence[0]
    assert evidence.physical_pressure_reevaluated_partition_established
    assert not evidence.all_segment_modal_diagnostics_applicable
    assert all(not item.available for item in evidence.modal_observations)
    assert all(
        item.abstention_reason
        == "frozen_euler_modal_diagnostic_not_representable"
        for item in evidence.modal_observations
    )
    assert graph.graph["_t"] == 0.5


def test_integrator_interior_history_samples_fail_and_roll_back() -> None:
    class InteriorHistoryIntegrator(AbstractIntegrator):
        def integrate(
            self,
            live_graph,
            *,
            dt,
            t,
            method,
            n_jobs,
        ) -> None:
            del method, n_jobs
            midpoint = float(t) + float(dt) / 2.0
            endpoint = float(t) + float(dt)
            for node in live_graph:
                data = live_graph.nodes[node]
                initial = float(data["EPI"])
                rate = float(data["nu_f"]) * float(data["delta_nfr"])
                middle_epi = initial + (float(dt) / 2.0) * rate
                final_epi = initial + float(dt) * rate
                data["epi_time_history"].append((midpoint, middle_epi))
                data["epi_time_history"].append((endpoint, final_epi))
                data["EPI"] = final_epi
            live_graph.graph["_t"] = endpoint

    graph = _pure_epi_graph()
    graph.graph["integrator"] = InteriorHistoryIntegrator()
    schedule, partition = _partitioned_empty_schedule()

    with pytest.raises(TNFRValueError, match="must not write epi_time_history"):
        execute_operator_event_schedule(
            graph,
            schedule,
            physical_flow_partitions=(partition,),
        )

    assert graph.graph["_t"] == 0.0
    assert tuple(graph.nodes[node]["EPI"] for node in graph) == (1.0, -1.0)
    assert all("epi_time_history" not in graph.nodes[node] for node in graph)


def test_custom_expansive_flow_is_rejected_before_modal_promotion() -> None:
    class ExpansiveIntegrator(AbstractIntegrator):
        def integrate(
            self,
            live_graph,
            *,
            dt,
            t,
            method,
            n_jobs,
        ) -> None:
            del method, n_jobs
            for node in live_graph:
                live_graph.nodes[node]["EPI"] *= 10.0
            live_graph.graph["_t"] = float(t) + float(dt)

    graph = _pure_epi_graph()
    graph.graph["integrator"] = ExpansiveIntegrator()
    schedule, partition = _partitioned_empty_schedule()

    with pytest.raises(TNFRValueError, match="exact held-input nodal equation"):
        execute_operator_event_schedule(
            graph,
            schedule,
            physical_flow_partitions=(partition,),
        )

    assert graph.graph["_t"] == 0.0
    assert tuple(graph.nodes[node]["EPI"] for node in graph) == (1.0, -1.0)
    assert all("epi_time_history" not in graph.nodes[node] for node in graph)


def test_partition_declarations_are_validated_before_execution() -> None:
    graph = _pure_epi_graph()
    schedule, partition = _partitioned_empty_schedule()

    with pytest.raises(ValueError, match="duplicate parent intervals"):
        execute_operator_event_schedule(
            graph,
            schedule,
            physical_flow_partitions=(partition, partition),
        )
    with pytest.raises(TypeError, match="iterable of partitions"):
        execute_operator_event_schedule(
            graph,
            schedule,
            physical_flow_partitions=None,  # type: ignore[arg-type]
        )
    assert graph.graph["_t"] == 0.0
    assert all("epi_time_history" not in graph.nodes[node] for node in graph)


def test_stage_composition_uses_last_and_first_physical_segment_evidence() -> None:
    graph = _pure_epi_graph()
    graph.nodes[0]["EPI"] = 0.25
    graph.nodes[1]["EPI"] = -0.25
    graph.graph.update(
        EPI_MIN=-10.0,
        EPI_MAX=10.0,
        NAV_RANDOM=False,
        NAV_MAX_DNFR=10.0,
        RANDOM_SEED=7,
        GLYPH_FACTORS={"NAV_eta": 0.25, "NAV_jitter": 0.0},
    )

    def refresh(live_graph: nx.Graph) -> None:
        left = float(live_graph.nodes[0]["EPI"])
        right = float(live_graph.nodes[1]["EPI"])
        live_graph.nodes[0]["delta_nfr"] = right - left
        live_graph.nodes[1]["delta_nfr"] = left - right

    graph.graph["compute_delta_nfr"] = refresh
    schedule = build_operator_event_schedule(
        ("transition",),
        start_time=0.0,
        flow_durations=(0.125, 0.125),
    )
    partitions = tuple(
        build_physical_flow_partition(interval, (0.0625, 0.0625))
        for interval in schedule.intervals
    )

    result = execute_operator_event_schedule(
        graph,
        schedule,
        include_stage_certificates=True,
        physical_flow_partitions=partitions,
    )

    stage = result.glyph_stage_evidence[0]
    assert stage.pre_flow_evidence is (
        result.physical_flow_partition_evidence[0].segment_flow_evidence[-1]
    )
    assert stage.post_flow_evidence is (
        result.physical_flow_partition_evidence[1].segment_flow_evidence[0]
    )
    assert stage.pre_flow_endpoint_continuous is True
    assert stage.post_flow_endpoint_continuous is True
    composition = result.represented_epi_schedule_composition
    assert composition is not None
    assert composition.represented_affine_composition_gain_certified
    assert composition.exact_operation_energy_gain_factors == (
        Fraction(2401, 4096),
        Fraction(1),
        Fraction(2401, 4096),
    )
    assert result.pressure_refresh_callback_invocations == 7
    assert result.physical_pressure_refresh_callback_invocations == 6
    assert result.stage_pressure_refresh_callback_invocations == 1


def test_constant_runtime_evidence_claims_remain_fail_closed() -> None:
    graph = _pure_epi_graph()
    graph.graph.update(
        NAV_RANDOM=False,
        NAV_MAX_DNFR=10.0,
        RANDOM_SEED=7,
        GLYPH_FACTORS={"NAV_eta": 0.25, "NAV_jitter": 0.0},
    )
    schedule = build_operator_event_schedule(
        ("transition",),
        start_time=0.0,
        flow_durations=(0.25, 0.25),
    )
    partitions = tuple(
        build_physical_flow_partition(interval, (0.125, 0.125))
        for interval in schedule.intervals
    )
    result = execute_operator_event_schedule(
        graph,
        schedule,
        include_stage_certificates=True,
        physical_flow_partitions=partitions,
    )
    partition = result.physical_flow_partition_evidence[0]
    flow = partition.segment_flow_evidence[0]
    boundary = partition.boundary_observations[0]
    stage = result.glyph_stage_evidence[0]

    historical_dataclass_claims = (
        (flow, "solver_accuracy_certified", True),
        (flow, "future_or_repeated_schedule_stability_certified", True),
        (stage, "solver_accuracy_certified", True),
        (stage, "future_or_repeated_schedule_stability_certified", True),
        (stage, "scope", "forged"),
    )
    for value, name, forged in historical_dataclass_claims:
        original = object.__getattribute__(value, name)
        public_value = getattr(value, name)
        assert value._proof_fields_are_intact()
        object.__setattr__(value, name, forged)
        assert not value._proof_fields_are_intact()
        assert getattr(value, name) == public_value
        object.__setattr__(value, name, original)
        assert value._proof_fields_are_intact()

    property_claims = (
        (boundary, "callback_completed", False),
        (boundary, "external_side_effects_rolled_back", True),
        (boundary, "scope", "forged"),
        (partition, "solver_accuracy_certified", True),
        (partition, "solver_order_certified", True),
        (partition, "mesh_convergence_certified", True),
        (partition, "future_or_repeated_behavior_certified", True),
        (partition, "adaptive_u2_u4_policy_certified", True),
        (partition, "external_side_effects_rolled_back", True),
        (partition, "scope", "forged"),
    )
    prior_values = tuple(
        getattr(value, name) for value, name, _forged in property_claims
    )

    for value, name, forged in property_claims:
        with pytest.raises((AttributeError, TypeError)):
            object.__setattr__(value, name, forged)

    assert tuple(
        getattr(value, name) for value, name, _forged in property_claims
    ) == prior_values
