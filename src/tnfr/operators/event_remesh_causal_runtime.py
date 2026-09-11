"""Causal execution of a finite operator-event/REMESH cycle sequence.

The existing cycle-sequence and schedule/history observers intentionally accept
already executed records and therefore cannot establish common graph
provenance, causal order, or cross-cycle atomicity.  This module supplies that
stronger finite boundary: it executes every declared cycle on one graph under
one controlling graph transaction and only then constructs those observers.

The exact schedule/history energy telescope is an optional strengthening of
that causal trace.  Some valid operator words have no common affine schedule
metric, so provenance and graph-owned atomicity must not depend on that
additional observation.  The promoted facts remain local to the returned
finite trace.  They do not establish a global mixed schedule/REMESH gain,
repeated or future stability, solver accuracy or order, mesh convergence, or
rollback of external effects.
"""

from __future__ import annotations

from collections.abc import Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any

import networkx as nx

from ..errors import TNFRValueError
from ..utils._structural_signature import (
    proof_stamps_are_identical,
    structural_proof_signature,
)
from ._delayed_remesh_kernel import _materialize_remesh_metric
from .event_remesh_runtime import (
    EventRemeshCycleResult,
    _ordered_identity_is,
    _read_only_graph_state,
    _require_canonical_nodes_surface,
    _require_graph,
    _require_read_only_graph_state,
    execute_event_remesh_cycle,
)
from .event_remesh_sequence import (
    ObservedEventRemeshCycleSequence,
    compose_event_remesh_cycle_observations,
)
from .event_runtime import _require_runtime_clock
from .event_timing import OperatorEventSchedule, PhysicalFlowPartition
from .network_stage import GraphTransactionSnapshot, _networkx_runtime_layout

__all__ = (
    "CausalEventRemeshCycleReceipt",
    "EventRemeshCycleExecutionSpec",
    "ExecutedEventRemeshCycleSequence",
    "execute_event_remesh_cycle_sequence",
)


_RECEIPT_PROOF_VERSION = "causal_event_remesh_cycle_receipt_v1"
_SEQUENCE_PROOF_VERSION = "executed_event_remesh_cycle_sequence_v2"
_SCOPE = (
    "One finite same-invocation execution of at least two declared "
    "operator-event/REMESH cycles on one graph. A controlling outer graph "
    "transaction encloses input materialization, all nested cycle savepoints, "
    "offline sequence composition, an optional exact schedule/history telescope "
    "when requested, and final sealing. Receipts bind each ordinal, exact spec "
    "identity, schedule identity, materialized physical partitions and cycle "
    "result to one opaque "
    "process-local token. This certifies causal order and graph-owned atomicity "
    "only for the returned finite invocation. It does not certify a global "
    "mixed schedule/REMESH gain, a uniform repeated margin, repeated or future "
    "stability, solver accuracy or order, mesh or binary64 convergence, "
    "adaptive grammar, concurrent-writer isolation, full graph/grammar-history "
    "boundary reconstruction, cryptographic provenance, or rollback of "
    "emitted I/O, warnings, external resources or external-only aliases."
)
_CONDITION_NAMES = (
    "complete_zero_based_cycle_range",
    "distinct_spec_identities",
    "distinct_schedule_identities",
    "distinct_cycle_result_identities",
    "one_execution_token_and_graph_owner",
    "every_receipt_intact",
    "exact_schedule_clock_chain",
    "common_ordered_target_support",
    "offline_sequence_bound_by_identity",
    "runtime_telescope_requirement_satisfied",
)


@dataclass(frozen=True, slots=True)
class EventRemeshCycleExecutionSpec:
    """Declare one cycle without consuming its physical-partition iterable.

    The executor deliberately materializes ``physical_flow_partitions`` only
    after its outer graph transaction has begun.  Keeping this constructor
    non-consuming ensures that a hostile or stateful iterable cannot mutate the
    graph before the controlling rollback boundary exists.
    """

    schedule: OperatorEventSchedule
    physical_flow_partitions: Iterable[PhysicalFlowPartition] = ()

    def __post_init__(self) -> None:
        if type(self.schedule) is not OperatorEventSchedule:
            raise TypeError("schedule must be an exact OperatorEventSchedule")


class _CausalExecutionToken:
    """Opaque process-local owner token retained strongly by every receipt."""

    __slots__ = ("graph_owner",)

    def __init__(self, graph: nx.Graph) -> None:
        self.graph_owner = graph


def _raw_proof_stamp(value: Any) -> tuple[Any, ...] | None:
    try:
        stamp = object.__getattribute__(value, "_proof_stamp")
    except BaseException:
        return None
    return stamp if type(stamp) is tuple else None


def _cycle_is_intact(value: Any) -> bool:
    if type(value) is not EventRemeshCycleResult:
        return False
    try:
        return bool(EventRemeshCycleResult._proof_fields_are_intact(value))
    except BaseException:
        return False


def _partition_identities_are_bound(
    cycle: EventRemeshCycleResult,
    partitions: tuple[PhysicalFlowPartition, ...],
) -> bool:
    try:
        evidence = cycle.event_execution.physical_flow_partition_evidence
        return bool(
            type(evidence) is tuple
            and len(evidence) == len(partitions)
            and all(
                item.partition is partition
                for item, partition in zip(evidence, partitions, strict=True)
            )
        )
    except BaseException:
        return False


def _receipt_stamp(
    *,
    cycle_index: int,
    spec: EventRemeshCycleExecutionSpec,
    schedule: OperatorEventSchedule,
    physical_flow_partition_source: Iterable[PhysicalFlowPartition],
    physical_flow_partitions: tuple[PhysicalFlowPartition, ...],
    cycle_result: EventRemeshCycleResult,
    target_nodes: tuple[Hashable, ...],
    exact_start_time: Fraction,
    exact_end_time: Fraction,
    graph_owner: nx.Graph,
    execution_token: _CausalExecutionToken,
) -> tuple[Any, ...]:
    return (
        _RECEIPT_PROOF_VERSION,
        cycle_index,
        ("graph-owner", id(graph_owner)),
        ("execution-token", id(execution_token)),
        ("spec", id(spec)),
        ("schedule", id(schedule)),
        ("physical-partition-source", id(physical_flow_partition_source)),
        tuple(("physical-partition", id(item)) for item in physical_flow_partitions),
        ("cycle-result", id(cycle_result), _raw_proof_stamp(cycle_result)),
        structural_proof_signature(target_nodes, identity_sensitive=True),
        exact_start_time,
        exact_end_time,
    )


@dataclass(frozen=True, slots=True)
class CausalEventRemeshCycleReceipt:
    """Sealed process-local binding for one executor-owned cycle result."""

    cycle_index: int
    spec: EventRemeshCycleExecutionSpec = field(repr=False, compare=False)
    schedule: OperatorEventSchedule = field(repr=False, compare=False)
    _physical_flow_partition_source: Iterable[PhysicalFlowPartition] = field(
        repr=False,
        compare=False,
    )
    physical_flow_partitions: tuple[PhysicalFlowPartition, ...] = field(
        repr=False,
        compare=False,
    )
    cycle_result: EventRemeshCycleResult = field(repr=False, compare=False)
    target_nodes: tuple[Hashable, ...]
    exact_start_time: Fraction
    exact_end_time: Fraction
    _graph_owner: nx.Graph = field(repr=False, compare=False)
    _execution_token: _CausalExecutionToken = field(repr=False, compare=False)
    _proof_stamp: tuple[Any, ...] = field(repr=False, compare=False)

    def _validate(self, *, cycle_already_validated: bool = False) -> None:
        if type(self.cycle_index) is not int or self.cycle_index < 0:
            raise ValueError("cycle_index must be a nonnegative integer")
        if type(self.spec) is not EventRemeshCycleExecutionSpec:
            raise TypeError("spec must be an exact EventRemeshCycleExecutionSpec")
        if type(self.schedule) is not OperatorEventSchedule:
            raise TypeError("schedule must be an exact OperatorEventSchedule")
        if self.spec.schedule is not self.schedule:
            raise ValueError("receipt schedule must be the spec schedule by identity")
        if (
            self.spec.physical_flow_partitions
            is not self._physical_flow_partition_source
        ):
            raise ValueError(
                "receipt partition source must remain the submitted spec field"
            )
        if type(self.physical_flow_partitions) is not tuple or any(
            type(item) is not PhysicalFlowPartition
            for item in self.physical_flow_partitions
        ):
            raise TypeError(
                "physical_flow_partitions must be an exact partition tuple"
            )
        if not cycle_already_validated and not _cycle_is_intact(self.cycle_result):
            raise ValueError("cycle_result proof fields are not intact")
        if self.cycle_result.event_execution.schedule is not self.schedule:
            raise ValueError(
                "executed cycle schedule must be the spec schedule by identity"
            )
        if not _partition_identities_are_bound(
            self.cycle_result,
            self.physical_flow_partitions,
        ):
            raise ValueError(
                "executed physical partitions must retain submitted identities"
            )
        if type(self.target_nodes) is not tuple or not _ordered_identity_is(
            self.cycle_result.target_nodes,
            self.target_nodes,
        ):
            raise ValueError("receipt target order must match the cycle result")
        if (
            type(self.exact_start_time) is not Fraction
            or type(self.exact_end_time) is not Fraction
            or self.exact_start_time != self.schedule.exact_start_time
            or self.exact_end_time != self.schedule.exact_end_time
        ):
            raise ValueError("receipt times must match the exact schedule boundary")
        if type(self._execution_token) is not _CausalExecutionToken:
            raise TypeError("receipt execution token is not canonical")
        if (
            self._execution_token.graph_owner is not self._graph_owner
            or not isinstance(
                self._graph_owner,
                (nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph),
            )
        ):
            raise ValueError("receipt token is not bound to its graph owner")
        expected = _receipt_stamp(
            cycle_index=self.cycle_index,
            spec=self.spec,
            schedule=self.schedule,
            physical_flow_partition_source=self._physical_flow_partition_source,
            physical_flow_partitions=self.physical_flow_partitions,
            cycle_result=self.cycle_result,
            target_nodes=self.target_nodes,
            exact_start_time=self.exact_start_time,
            exact_end_time=self.exact_end_time,
            graph_owner=self._graph_owner,
            execution_token=self._execution_token,
        )
        if not proof_stamps_are_identical(self._proof_stamp, expected):
            raise ValueError("causal cycle receipt proof fields are inconsistent")

    def __post_init__(self) -> None:
        self._validate()

    def _proof_fields_are_intact(self) -> bool:
        try:
            self._validate()
        except BaseException:
            return False
        return True

    def _proof_fields_are_intact_after_cycle_validation(self) -> bool:
        """Validate this binding without rewalking an already checked cycle."""

        try:
            self._validate(cycle_already_validated=True)
        except BaseException:
            return False
        return True

    @property
    def receipt_binding_certified(self) -> bool:
        """Whether this ordinal/spec/schedule/result binding remains intact."""

        return self._proof_fields_are_intact()


def _sealed_receipt(
    *,
    cycle_index: int,
    spec: EventRemeshCycleExecutionSpec,
    schedule: OperatorEventSchedule,
    physical_flow_partition_source: Iterable[PhysicalFlowPartition],
    physical_flow_partitions: tuple[PhysicalFlowPartition, ...],
    cycle_result: EventRemeshCycleResult,
    target_nodes: tuple[Hashable, ...],
    graph_owner: nx.Graph,
    execution_token: _CausalExecutionToken,
) -> CausalEventRemeshCycleReceipt:
    values = {
        "cycle_index": cycle_index,
        "spec": spec,
        "schedule": schedule,
        "physical_flow_partition_source": physical_flow_partition_source,
        "physical_flow_partitions": physical_flow_partitions,
        "cycle_result": cycle_result,
        "target_nodes": target_nodes,
        "exact_start_time": schedule.exact_start_time,
        "exact_end_time": schedule.exact_end_time,
        "graph_owner": graph_owner,
        "execution_token": execution_token,
    }
    return CausalEventRemeshCycleReceipt(
        cycle_index=cycle_index,
        spec=spec,
        schedule=schedule,
        _physical_flow_partition_source=physical_flow_partition_source,
        physical_flow_partitions=physical_flow_partitions,
        cycle_result=cycle_result,
        target_nodes=target_nodes,
        exact_start_time=schedule.exact_start_time,
        exact_end_time=schedule.exact_end_time,
        _graph_owner=graph_owner,
        _execution_token=execution_token,
        _proof_stamp=_receipt_stamp(**values),
    )


def _conditions_for(
    *,
    cycle_indices: tuple[int, ...],
    specs: tuple[EventRemeshCycleExecutionSpec, ...],
    schedules: tuple[OperatorEventSchedule, ...],
    receipts: tuple[CausalEventRemeshCycleReceipt, ...],
    cycles: tuple[EventRemeshCycleResult, ...],
    target_nodes: tuple[Hashable, ...],
    graph_owner: nx.Graph,
    execution_token: _CausalExecutionToken,
    observed_sequence: ObservedEventRemeshCycleSequence,
    runtime_telescope_required: bool,
    runtime_telescope: Any,
) -> tuple[tuple[str, bool], ...]:
    exact_clock_chain = bool(
        schedules
        and all(
            left.exact_end_time == right.exact_start_time
            and left.end_time == right.start_time
            for left, right in zip(schedules, schedules[1:])
        )
    )
    common_targets = all(
        _ordered_identity_is(cycle.target_nodes, target_nodes) for cycle in cycles
    )
    telescope_bound = False
    if runtime_telescope is not None:
        try:
            from ..physics.runtime_remesh_schedule_stability import (
                RuntimeRemeshScheduleSequenceObservation,
            )

            telescope_bound = bool(
                type(runtime_telescope)
                is RuntimeRemeshScheduleSequenceObservation
                and runtime_telescope.sequence_observation_certified
                and runtime_telescope.source_sequence is observed_sequence
                and len(runtime_telescope.boundaries) == len(cycles) - 1
            )
        except BaseException:
            telescope_bound = False
    try:
        offline_bound = bool(
            type(observed_sequence) is ObservedEventRemeshCycleSequence
            and ObservedEventRemeshCycleSequence._proof_fields_are_intact(
                observed_sequence
            )
            is True
            and len(observed_sequence.cycles) == len(cycles)
            and all(
                observed is expected
                for observed, expected in zip(
                    observed_sequence.cycles,
                    cycles,
                    strict=True,
                )
            )
        )
    except BaseException:
        offline_bound = False
    telescope_requirement_satisfied = bool(
        type(runtime_telescope_required) is bool
        and (
            (runtime_telescope_required and telescope_bound)
            or (not runtime_telescope_required and runtime_telescope is None)
        )
    )
    return (
        (
            "complete_zero_based_cycle_range",
            cycle_indices == tuple(range(len(cycles))) and len(cycles) >= 2,
        ),
        ("distinct_spec_identities", len({id(item) for item in specs}) == len(specs)),
        (
            "distinct_schedule_identities",
            len({id(item) for item in schedules}) == len(schedules),
        ),
        (
            "distinct_cycle_result_identities",
            len({id(item) for item in cycles}) == len(cycles),
        ),
        (
            "one_execution_token_and_graph_owner",
            type(execution_token) is _CausalExecutionToken
            and execution_token.graph_owner is graph_owner
            and all(
                receipt._execution_token is execution_token
                and receipt._graph_owner is graph_owner
                for receipt in receipts
            ),
        ),
        (
            "every_receipt_intact",
            len(receipts) == len(cycles)
            and all(
                receipt._proof_fields_are_intact_after_cycle_validation()
                for receipt in receipts
            ),
        ),
        ("exact_schedule_clock_chain", exact_clock_chain),
        ("common_ordered_target_support", common_targets),
        ("offline_sequence_bound_by_identity", offline_bound),
        (
            "runtime_telescope_requirement_satisfied",
            telescope_requirement_satisfied,
        ),
    )


def _sequence_stamp(
    *,
    cycle_indices: tuple[int, ...],
    specs: tuple[EventRemeshCycleExecutionSpec, ...],
    schedules: tuple[OperatorEventSchedule, ...],
    physical_flow_partitions_by_cycle: tuple[
        tuple[PhysicalFlowPartition, ...], ...
    ],
    receipts: tuple[CausalEventRemeshCycleReceipt, ...],
    cycles: tuple[EventRemeshCycleResult, ...],
    target_nodes: tuple[Hashable, ...],
    exact_start_time: Fraction,
    exact_end_time: Fraction,
    observed_sequence: ObservedEventRemeshCycleSequence,
    runtime_telescope_required: bool,
    runtime_telescope: Any,
    conditions: tuple[tuple[str, bool], ...],
    graph_owner: nx.Graph,
    execution_token: _CausalExecutionToken,
) -> tuple[Any, ...]:
    return (
        _SEQUENCE_PROOF_VERSION,
        cycle_indices,
        tuple(("spec", id(item)) for item in specs),
        tuple(("schedule", id(item)) for item in schedules),
        tuple(
            tuple(("physical-partition", id(item)) for item in row)
            for row in physical_flow_partitions_by_cycle
        ),
        tuple(
            ("receipt", id(item), _raw_proof_stamp(item)) for item in receipts
        ),
        tuple(("cycle", id(item), _raw_proof_stamp(item)) for item in cycles),
        structural_proof_signature(target_nodes, identity_sensitive=True),
        exact_start_time,
        exact_end_time,
        (
            "observed-sequence",
            id(observed_sequence),
            _raw_proof_stamp(observed_sequence),
        ),
        ("runtime-telescope-required", runtime_telescope_required),
        (
            "runtime-telescope",
            id(runtime_telescope),
            _raw_proof_stamp(runtime_telescope),
        ),
        conditions,
        ("graph-owner", id(graph_owner)),
        ("execution-token", id(execution_token)),
    )


@dataclass(frozen=True, slots=True)
class ExecutedEventRemeshCycleSequence:
    """Sealed evidence for one committed causal finite cycle sequence."""

    cycle_indices: tuple[int, ...]
    specs: tuple[EventRemeshCycleExecutionSpec, ...] = field(
        repr=False,
        compare=False,
    )
    schedules: tuple[OperatorEventSchedule, ...] = field(
        repr=False,
        compare=False,
    )
    physical_flow_partitions_by_cycle: tuple[
        tuple[PhysicalFlowPartition, ...], ...
    ] = field(repr=False, compare=False)
    receipts: tuple[CausalEventRemeshCycleReceipt, ...]
    cycles: tuple[EventRemeshCycleResult, ...] = field(repr=False, compare=False)
    target_nodes: tuple[Hashable, ...]
    exact_start_time: Fraction
    exact_end_time: Fraction
    observed_sequence: ObservedEventRemeshCycleSequence = field(
        repr=False,
        compare=False,
    )
    runtime_telescope_required: bool
    runtime_telescope: Any = field(repr=False, compare=False)
    conditions: tuple[tuple[str, bool], ...]
    _graph_owner: nx.Graph = field(repr=False, compare=False)
    _execution_token: _CausalExecutionToken = field(repr=False, compare=False)
    _proof_stamp: tuple[Any, ...] = field(repr=False, compare=False)
    scope: str = field(default=_SCOPE, init=False)

    def __post_init__(self) -> None:
        tuple_fields = (
            self.cycle_indices,
            self.specs,
            self.schedules,
            self.physical_flow_partitions_by_cycle,
            self.receipts,
            self.cycles,
            self.target_nodes,
            self.conditions,
        )
        if any(type(value) is not tuple for value in tuple_fields):
            raise TypeError("executed sequence collection fields must be tuples")
        cardinality = len(self.cycles)
        if cardinality < 2 or any(
            len(value) != cardinality
            for value in (
                self.cycle_indices,
                self.specs,
                self.schedules,
                self.physical_flow_partitions_by_cycle,
                self.receipts,
            )
        ):
            raise ValueError(
                "executed sequence fields must describe at least two cycles"
            )
        if any(
            type(item) is not EventRemeshCycleExecutionSpec for item in self.specs
        ):
            raise TypeError("specs must contain exact execution specs")
        if any(type(item) is not OperatorEventSchedule for item in self.schedules):
            raise TypeError("schedules must contain exact operator-event schedules")
        if any(
            type(item) is not CausalEventRemeshCycleReceipt
            for item in self.receipts
        ):
            raise TypeError("receipts must contain exact causal receipts")
        if any(type(item) is not EventRemeshCycleResult for item in self.cycles):
            raise TypeError("cycles must contain exact cycle results")
        if type(self.runtime_telescope_required) is not bool:
            raise TypeError("runtime_telescope_required must be a bool")
        if not self.runtime_telescope_required and self.runtime_telescope is not None:
            raise ValueError(
                "runtime_telescope must be absent when it was not requested"
            )
        if any(
            type(row) is not tuple
            or any(type(item) is not PhysicalFlowPartition for item in row)
            for row in self.physical_flow_partitions_by_cycle
        ):
            raise TypeError("physical partitions must be exact per-cycle tuples")
        if any(
            receipt.cycle_index != index
            or receipt.spec is not self.specs[index]
            or receipt.schedule is not self.schedules[index]
            or receipt.physical_flow_partitions
            is not self.physical_flow_partitions_by_cycle[index]
            or receipt.cycle_result is not self.cycles[index]
            for index, receipt in enumerate(self.receipts)
        ):
            raise ValueError("receipts do not retain exact sequence identities")
        if (
            type(self.exact_start_time) is not Fraction
            or type(self.exact_end_time) is not Fraction
            or self.exact_start_time != self.schedules[0].exact_start_time
            or self.exact_end_time != self.schedules[-1].exact_end_time
        ):
            raise ValueError("sequence exact times do not match its schedules")
        expected_conditions = _conditions_for(
            cycle_indices=self.cycle_indices,
            specs=self.specs,
            schedules=self.schedules,
            receipts=self.receipts,
            cycles=self.cycles,
            target_nodes=self.target_nodes,
            graph_owner=self._graph_owner,
            execution_token=self._execution_token,
            observed_sequence=self.observed_sequence,
            runtime_telescope_required=self.runtime_telescope_required,
            runtime_telescope=self.runtime_telescope,
        )
        if (
            type(self.conditions) is not tuple
            or tuple(name for name, _passed in self.conditions) != _CONDITION_NAMES
            or self.conditions != expected_conditions
            or not all(passed for _name, passed in expected_conditions)
        ):
            raise ValueError("executed sequence causal conditions are inconsistent")
        if self.scope != _SCOPE:
            raise ValueError("executed sequence scope is inconsistent")
        expected_stamp = _sequence_stamp(
            cycle_indices=self.cycle_indices,
            specs=self.specs,
            schedules=self.schedules,
            physical_flow_partitions_by_cycle=(
                self.physical_flow_partitions_by_cycle
            ),
            receipts=self.receipts,
            cycles=self.cycles,
            target_nodes=self.target_nodes,
            exact_start_time=self.exact_start_time,
            exact_end_time=self.exact_end_time,
            observed_sequence=self.observed_sequence,
            runtime_telescope_required=self.runtime_telescope_required,
            runtime_telescope=self.runtime_telescope,
            conditions=self.conditions,
            graph_owner=self._graph_owner,
            execution_token=self._execution_token,
        )
        if not proof_stamps_are_identical(self._proof_stamp, expected_stamp):
            raise ValueError("executed sequence proof fields are inconsistent")

    def _proof_fields_are_intact(self) -> bool:
        try:
            self.__post_init__()
        except BaseException:
            return False
        return True

    @property
    def causal_cycle_order_certified(self) -> bool:
        return self._proof_fields_are_intact()

    @property
    def same_graph_execution_provenance_certified(self) -> bool:
        return self._proof_fields_are_intact()

    @property
    def whole_sequence_graph_state_atomic(self) -> bool:
        return self._proof_fields_are_intact()

    @property
    def exact_recorded_boundary_continuity_certified(self) -> bool:
        return bool(
            self._proof_fields_are_intact()
            and self.observed_sequence.exact_recorded_boundary_continuity_certified
        )

    @property
    def exact_finite_energy_telescope_certified(self) -> bool:
        if not self._proof_fields_are_intact() or self.runtime_telescope is None:
            return False
        try:
            return bool(
                self.runtime_telescope.source_sequence is self.observed_sequence
                and self.runtime_telescope.exact_finite_energy_telescope_certified
            )
        except BaseException:
            return False

    @property
    def failed_conditions(self) -> tuple[str, ...]:
        if not self._proof_fields_are_intact():
            return ("executed_sequence_proof_fields_intact",)
        return ()

    @property
    def runtime_global_gain_certified(self) -> bool:
        return False

    @property
    def repeated_runtime_stability_certified(self) -> bool:
        return False

    @property
    def future_stability_certified(self) -> bool:
        return False

    @property
    def solver_accuracy_certified(self) -> bool:
        return False

    @property
    def solver_order_certified(self) -> bool:
        return False

    @property
    def mesh_convergence_certified(self) -> bool:
        return False

    @property
    def adaptive_u2_u4_policy_certified(self) -> bool:
        return False

    @property
    def full_tnfr_stability_certified(self) -> bool:
        return False

    @property
    def full_graph_state_continuity_certified(self) -> bool:
        return False

    @property
    def grammar_history_continuity_certified(self) -> bool:
        return False

    @property
    def concurrent_writer_atomicity_certified(self) -> bool:
        return False

    @property
    def cryptographic_or_durable_provenance_certified(self) -> bool:
        return False

    @property
    def external_side_effects_rolled_back(self) -> bool:
        return False


def _sealed_sequence(
    *,
    specs: tuple[EventRemeshCycleExecutionSpec, ...],
    schedules: tuple[OperatorEventSchedule, ...],
    physical_flow_partitions_by_cycle: tuple[
        tuple[PhysicalFlowPartition, ...], ...
    ],
    receipts: tuple[CausalEventRemeshCycleReceipt, ...],
    cycles: tuple[EventRemeshCycleResult, ...],
    target_nodes: tuple[Hashable, ...],
    observed_sequence: ObservedEventRemeshCycleSequence,
    runtime_telescope_required: bool,
    runtime_telescope: Any,
    graph_owner: nx.Graph,
    execution_token: _CausalExecutionToken,
) -> ExecutedEventRemeshCycleSequence:
    cycle_indices = tuple(range(len(cycles)))
    conditions = _conditions_for(
        cycle_indices=cycle_indices,
        specs=specs,
        schedules=schedules,
        receipts=receipts,
        cycles=cycles,
        target_nodes=target_nodes,
        graph_owner=graph_owner,
        execution_token=execution_token,
        observed_sequence=observed_sequence,
        runtime_telescope_required=runtime_telescope_required,
        runtime_telescope=runtime_telescope,
    )
    if not all(passed for _name, passed in conditions):
        failed = ", ".join(name for name, passed in conditions if not passed)
        raise TNFRValueError(f"causal cycle sequence failed: {failed}")
    exact_start = schedules[0].exact_start_time
    exact_end = schedules[-1].exact_end_time
    stamp = _sequence_stamp(
        cycle_indices=cycle_indices,
        specs=specs,
        schedules=schedules,
        physical_flow_partitions_by_cycle=physical_flow_partitions_by_cycle,
        receipts=receipts,
        cycles=cycles,
        target_nodes=target_nodes,
        exact_start_time=exact_start,
        exact_end_time=exact_end,
        observed_sequence=observed_sequence,
        runtime_telescope_required=runtime_telescope_required,
        runtime_telescope=runtime_telescope,
        conditions=conditions,
        graph_owner=graph_owner,
        execution_token=execution_token,
    )
    return ExecutedEventRemeshCycleSequence(
        cycle_indices=cycle_indices,
        specs=specs,
        schedules=schedules,
        physical_flow_partitions_by_cycle=physical_flow_partitions_by_cycle,
        receipts=receipts,
        cycles=cycles,
        target_nodes=target_nodes,
        exact_start_time=exact_start,
        exact_end_time=exact_end,
        observed_sequence=observed_sequence,
        runtime_telescope_required=runtime_telescope_required,
        runtime_telescope=runtime_telescope,
        conditions=conditions,
        _graph_owner=graph_owner,
        _execution_token=execution_token,
        _proof_stamp=stamp,
    )


def _materialize_specs_and_partitions(
    specs: Iterable[EventRemeshCycleExecutionSpec],
) -> tuple[
    tuple[EventRemeshCycleExecutionSpec, ...],
    tuple[tuple[PhysicalFlowPartition, ...], ...],
]:
    try:
        materialized_specs = tuple(specs)
    except TypeError as exc:
        raise TypeError("specs must be an iterable of execution specs") from exc
    if len(materialized_specs) < 2:
        raise TNFRValueError("at least two event/REMESH cycle specs are required")
    if len({id(item) for item in materialized_specs}) != len(materialized_specs):
        raise TNFRValueError("cycle specs must have distinct identities")

    rows: list[tuple[PhysicalFlowPartition, ...]] = []
    for index, spec in enumerate(materialized_specs):
        if type(spec) is not EventRemeshCycleExecutionSpec:
            raise TypeError(
                f"specs[{index}] must be an exact EventRemeshCycleExecutionSpec"
            )
        spec.schedule.__post_init__()
        try:
            partitions = tuple(spec.physical_flow_partitions)
        except TypeError as exc:
            raise TypeError(
                f"specs[{index}].physical_flow_partitions must be iterable"
            ) from exc
        for position, partition in enumerate(partitions):
            if type(partition) is not PhysicalFlowPartition:
                raise TypeError(
                    f"specs[{index}].physical_flow_partitions[{position}] "
                    "must be an exact PhysicalFlowPartition"
                )
        rows.append(partitions)
    return materialized_specs, tuple(rows)


def _require_schedule_chain(
    graph: nx.Graph,
    schedules: tuple[OperatorEventSchedule, ...],
) -> None:
    if len({id(item) for item in schedules}) != len(schedules):
        raise TNFRValueError("cycle schedules must have distinct identities")
    _require_runtime_clock(
        graph,
        schedules[0].start_time,
        boundary="event_remesh_sequence.start",
    )
    for index, (left, right) in enumerate(zip(schedules, schedules[1:])):
        if (
            left.exact_end_time != right.exact_start_time
            or left.end_time != right.start_time
        ):
            raise TNFRValueError(
                "cycle schedules must form one exact and represented clock chain",
                context={"boundary_index": index},
            )


def execute_event_remesh_cycle_sequence(
    graph: nx.Graph,
    specs: Iterable[EventRemeshCycleExecutionSpec],
    *,
    metric_weights: Mapping[Hashable, Any] | Sequence[Any] | None = None,
    context: Mapping[str, Any] | None = None,
    method: str | None = None,
    n_jobs: int | None = None,
    suppress_birth_warnings: bool = False,
    require_runtime_telescope: bool = True,
) -> ExecutedEventRemeshCycleSequence:
    """Execute and seal one causal finite sequence of cycle specifications.

    The outer snapshot is created before ``specs``, every per-spec physical
    partition iterable, or ``metric_weights`` is consumed.  Each nested cycle
    keeps its ordinary transaction as an internal savepoint.  Any exception
    from input materialization, cycle execution, observation, requested
    telescope construction or sealing restores the graph to the state captured
    before the whole sequence began while preserving the primary exception.
    """

    graph = _require_graph(graph)
    if type(suppress_birth_warnings) is not bool:
        raise TypeError("suppress_birth_warnings must be a bool")
    if type(require_runtime_telescope) is not bool:
        raise TypeError("require_runtime_telescope must be a bool")
    transaction = GraphTransactionSnapshot(graph)
    try:
        preparation_state = _read_only_graph_state(graph)
        _require_canonical_nodes_surface(graph)
        layout = _networkx_runtime_layout(graph)
        target_nodes = tuple(node for node, _data in layout.node_data)
        if not target_nodes:
            raise TNFRValueError("causal cycle execution requires nonempty support")
        materialized_specs, partition_rows = _materialize_specs_and_partitions(specs)
        schedules = tuple(spec.schedule for spec in materialized_specs)
        frozen_metric = _materialize_remesh_metric(metric_weights, target_nodes)
        _require_schedule_chain(graph, schedules)
        _require_read_only_graph_state(
            graph,
            preparation_state,
            boundary="causal_sequence_input_materialization",
        )

        token = _CausalExecutionToken(graph)
        receipt_items: list[CausalEventRemeshCycleReceipt] = []
        cycle_items: list[EventRemeshCycleResult] = []
        for index, (spec, schedule, partitions) in enumerate(
            zip(materialized_specs, schedules, partition_rows, strict=True)
        ):
            _require_runtime_clock(
                graph,
                schedule.start_time,
                boundary=f"event_remesh_sequence.cycle[{index}].start",
            )
            cycle = execute_event_remesh_cycle(
                graph,
                schedule,
                metric_weights=frozen_metric,
                refresh_pressure_after_remesh=True,
                context=context,
                method=method,
                n_jobs=n_jobs,
                suppress_birth_warnings=suppress_birth_warnings,
                include_stage_certificates=True,
                physical_flow_partitions=partitions,
            )
            if not _ordered_identity_is(cycle.target_nodes, target_nodes):
                raise TNFRValueError("cycle execution changed ordered node support")
            receipt = _sealed_receipt(
                cycle_index=index,
                spec=spec,
                schedule=schedule,
                physical_flow_partition_source=(
                    spec.physical_flow_partitions
                ),
                physical_flow_partitions=partitions,
                cycle_result=cycle,
                target_nodes=target_nodes,
                graph_owner=graph,
                execution_token=token,
            )
            cycle_items.append(cycle)
            receipt_items.append(receipt)

        cycles = tuple(cycle_items)
        receipts = tuple(receipt_items)
        observation_state = _read_only_graph_state(graph)
        observed_sequence = compose_event_remesh_cycle_observations(cycles)

        runtime_telescope = None
        if require_runtime_telescope:
            # Local import avoids a package-initialization cycle: tnfr.physics
            # imports the public operator facade during ordinary package startup.
            from ..physics.runtime_remesh_schedule_stability import (
                observe_runtime_remesh_schedule_sequence,
            )

            runtime_telescope = observe_runtime_remesh_schedule_sequence(
                observed_sequence
            )
        result = _sealed_sequence(
            specs=materialized_specs,
            schedules=schedules,
            physical_flow_partitions_by_cycle=partition_rows,
            receipts=receipts,
            cycles=cycles,
            target_nodes=target_nodes,
            observed_sequence=observed_sequence,
            runtime_telescope_required=require_runtime_telescope,
            runtime_telescope=runtime_telescope,
            graph_owner=graph,
            execution_token=token,
        )
        _require_read_only_graph_state(
            graph,
            observation_state,
            boundary="causal_sequence_observation_and_sealing",
        )
        return result
    except BaseException as failure:
        transaction.restore_after_failure(graph, failure)
        raise
