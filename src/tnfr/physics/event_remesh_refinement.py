"""Finite three-mesh observations for event/REMESH cycle executions.

The records in this module compare three already committed
``execute_event_remesh_cycle`` results.  They require strictly nested explicit
physical pressure-refresh partitions and retain the represented schedule and
delayed REMESH evidence as separate objects.  Pairwise EPI errors, physical
ZHIR secants and compatible-generator modal factors are observations of those
three finite executions; they are not a solver-order or convergence theorem.
"""

from __future__ import annotations

import math
from collections.abc import Hashable
from dataclasses import dataclass, field, fields, replace
from fractions import Fraction
from numbers import Real
from typing import Any

from ..operators.event_remesh_runtime import EventRemeshCycleResult, _proof_value
from ..operators.event_runtime import (
    ExecutedPressureRefreshedFlowPartition,
    ObservedRepresentedEPIScheduleComposition,
)
from ..operators.remesh import DelayedRemeshResult
from ..types import Glyph
from ..utils._structural_signature import (
    binary64_vectors_are_identical,
    proof_stamps_are_identical,
)
from ._exact_metric import normalized_positive_binary64_metric
from .event_refinement import (
    ExecutedEventLocalZHIRPhysicalPrejumpObservation,
    observe_executed_event_local_zhir_physical_prejump,
)

__all__ = (
    "EventRemeshEPICheckpointObservation",
    "EventRemeshMeshObservation",
    "EventRemeshPersistentEPIError",
    "EventRemeshThreeMeshModalObservation",
    "EventRemeshThreeMeshRefinementObservation",
    "EventRemeshThreeMeshZHIRObservation",
    "observe_event_remesh_three_mesh_refinement",
)


_CHECKPOINT_PROOF_VERSION = "event_remesh_epi_checkpoint_v1"
_MESH_PROOF_VERSION = "event_remesh_mesh_observation_v1"
_ERROR_PROOF_VERSION = "event_remesh_persistent_epi_error_v1"
_ZHIR_PROOF_VERSION = "event_remesh_three_mesh_zhir_v1"
_MODAL_PROOF_VERSION = "event_remesh_three_mesh_modal_v1"
_THREE_MESH_PROOF_VERSION = "event_remesh_three_mesh_refinement_v1"
_MESH_NAMES = ("coarse", "intermediate", "fine")
_SCOPE = (
    "Three finite executor-sealed event/REMESH cycle observations with strictly "
    "nested explicit physical pressure-refresh checkpoints. Pairwise EPI "
    "errors use one identical ordered full node support. ZHIR rates and gates "
    "are retained only when executor-linked physical pre-jump and glyph-stage "
    "evidence exists. "
    "Modal factors are published only when all three executions share one "
    "ordered support, capacity/conductance generator and modal spectrum. "
    "Represented schedule compositions and delayed REMESH results remain "
    "separate. Solver accuracy/order, mesh convergence, Lyapunov decrease, "
    "global or future behavior, a combined schedule-times-REMESH gain and "
    "whole-three-mesh atomicity are not certified. The detached cycle artifacts "
    "do not expose the complete pre-schedule graph namespace, callback closure "
    "state, RNG state, node metadata or sub-EPI state, so exclusive attribution "
    "of observed differences to mesh refinement is not certified."
)


def _proof_stamp(value: Any, expected_type: type[Any], version: str) -> tuple[Any, ...]:
    if type(value) is not expected_type:
        raise TypeError("proof value must have its canonical result type")
    return (
        version,
        _compact_proof_value(
            tuple(
                (item.name, object.__getattribute__(value, item.name))
                for item in fields(expected_type)
                if item.name != "_proof_stamp"
            )
        ),
    )


def _compact_proof_value(value: Any) -> Any:
    """Serialize nested seals through their immutable proof stamps."""

    known_names = (
        "EventRemeshEPICheckpointObservation",
        "EventRemeshMeshObservation",
        "EventRemeshPersistentEPIError",
        "EventRemeshThreeMeshZHIRObservation",
        "EventRemeshThreeMeshModalObservation",
        "EventRemeshThreeMeshRefinementObservation",
    )
    known_types = (
        EventRemeshCycleResult,
        ObservedRepresentedEPIScheduleComposition,
        ExecutedEventLocalZHIRPhysicalPrejumpObservation,
        *(globals()[name] for name in known_names if name in globals()),
    )
    if type(value) in known_types:
        stamp = object.__getattribute__(value, "_proof_stamp")
        if type(stamp) is not tuple:
            return (
                "invalid-proof-stamp",
                type(value).__module__,
                type(value).__qualname__,
            )
        return (
            "nested-proof-stamp",
            type(value).__module__,
            type(value).__qualname__,
            stamp,
        )
    if type(value) is tuple:
        return ("tuple", tuple(_compact_proof_value(item) for item in value))
    return _proof_value(value)


def _seal(value: Any, expected_type: type[Any], version: str) -> Any:
    return replace(
        value,
        _proof_stamp=_proof_stamp(value, expected_type, version),
    )


def _sealed(value: Any, expected_type: type[Any], version: str) -> bool:
    try:
        expected = _proof_stamp(value, expected_type, version)
        observed = object.__getattribute__(value, "_proof_stamp")
        return proof_stamps_are_identical(observed, expected)
    except BaseException:
        return False


def _exact_vector(values: tuple[float, ...]) -> tuple[Fraction, ...]:
    return tuple(Fraction.from_float(value) for value in values)


def _optional_float(value: Fraction) -> float | None:
    try:
        result = float(value)
    except (OverflowError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _node_tokens(
    nodes: tuple[Hashable, ...],
    label: str,
) -> tuple[tuple[Any, ...], ...]:
    tokens = tuple(_proof_value(node) for node in nodes)
    if len(set(tokens)) != len(tokens):
        raise ValueError(f"{label} contains duplicate structural node identifiers")
    return tokens


def _node_identifiers_match(left: Hashable, right: Hashable) -> bool:
    if left is right:
        return True
    try:
        equality_holds = bool(left == right)
        hashes_match = hash(left) == hash(right)
    except Exception:
        return False
    return bool(
        equality_holds
        and hashes_match
        and _proof_value(left) == _proof_value(right)
    )


def _ordered_supports_match(
    left: tuple[Hashable, ...],
    right: tuple[Hashable, ...],
) -> bool:
    return bool(
        len(left) == len(right)
        and all(
            _node_identifiers_match(left_node, right_node)
            for left_node, right_node in zip(left, right, strict=True)
        )
    )


def _persistent_support(
    coarse: tuple[Hashable, ...],
    intermediate: tuple[Hashable, ...],
    fine: tuple[Hashable, ...],
) -> tuple[tuple[Hashable, ...], tuple[tuple[Any, ...], ...]]:
    coarse_tokens = _node_tokens(coarse, "coarse support")
    intermediate_tokens = _node_tokens(intermediate, "intermediate support")
    fine_tokens = _node_tokens(fine, "fine support")
    intermediate_by_token = dict(zip(intermediate_tokens, intermediate, strict=True))
    fine_by_token = dict(zip(fine_tokens, fine, strict=True))
    pairs = tuple(
        (node, token)
        for node, token in zip(coarse, coarse_tokens, strict=True)
        if token in intermediate_by_token
        and token in fine_by_token
        and _node_identifiers_match(node, intermediate_by_token[token])
        and _node_identifiers_match(node, fine_by_token[token])
    )
    if not pairs:
        raise ValueError(
            "coarse, intermediate and fine cycles require persistent node support"
        )
    return (
        tuple(node for node, _ in pairs),
        tuple(token for _, token in pairs),
    )


def _project_values(
    nodes: tuple[Hashable, ...],
    values: tuple[Any, ...],
    persistent_tokens: tuple[tuple[Any, ...], ...],
    *,
    label: str,
) -> tuple[Any, ...]:
    if len(nodes) != len(values):
        raise ValueError(f"{label} does not align with its node support")
    tokens = _node_tokens(nodes, f"{label} support")
    by_token = {token: index for index, token in enumerate(tokens)}
    try:
        return tuple(values[by_token[token]] for token in persistent_tokens)
    except KeyError as exc:
        raise ValueError(f"{label} omits a persistent node") from exc


@dataclass(frozen=True, slots=True)
class EventRemeshEPICheckpointObservation:
    """One exact represented EPI vector at a cycle or physical boundary."""

    mesh_name: str
    checkpoint_kind: str
    parent_interval_index: int | None
    boundary_index: int | None
    time: float
    exact_time: Fraction
    nodes: tuple[Hashable, ...]
    epi_values: tuple[float, ...]
    exact_epi_values: tuple[Fraction, ...]
    _proof_stamp: tuple[Any, ...] = field(default=(), repr=False, compare=False)

    def _proof_fields_are_intact(self) -> bool:
        return _sealed(
            self,
            EventRemeshEPICheckpointObservation,
            _CHECKPOINT_PROOF_VERSION,
        )

    @property
    def checkpoint_observation_certified(self) -> bool:
        return self._proof_fields_are_intact()

    @property
    def key(self) -> tuple[str, int | None, Fraction]:
        return (
            self.checkpoint_kind,
            self.parent_interval_index,
            self.exact_time,
        )


@dataclass(frozen=True, slots=True)
class EventRemeshMeshObservation:
    """One sealed cycle projected onto its explicit physical checkpoints."""

    mesh_name: str
    cycle_result: EventRemeshCycleResult = field(repr=False)
    nodes: tuple[Hashable, ...]
    physical_partition_interval_indices: tuple[int, ...]
    exact_partition_boundary_times: tuple[
        tuple[int, tuple[Fraction, ...]], ...
    ]
    checkpoints: tuple[EventRemeshEPICheckpointObservation, ...]
    schedule_composition: ObservedRepresentedEPIScheduleComposition | None = field(
        repr=False
    )
    remesh_result: DelayedRemeshResult = field(repr=False)
    _proof_stamp: tuple[Any, ...] = field(default=(), repr=False, compare=False)

    def _proof_fields_are_intact(self) -> bool:
        return bool(
            _sealed(self, EventRemeshMeshObservation, _MESH_PROOF_VERSION)
            and type(self.cycle_result) is EventRemeshCycleResult
            and self.cycle_result._proof_fields_are_intact()
            and self.remesh_result is self.cycle_result.remesh
            and self.schedule_composition
            is self.cycle_result.event_execution.represented_epi_schedule_composition
            and all(
                type(item) is EventRemeshEPICheckpointObservation
                and item._proof_fields_are_intact()
                for item in self.checkpoints
            )
            and (
                self.schedule_composition is None
                or (
                    type(self.schedule_composition)
                    is ObservedRepresentedEPIScheduleComposition
                    and self.schedule_composition._proof_fields_are_intact()
                )
            )
        )

    @property
    def mesh_observation_certified(self) -> bool:
        return self._proof_fields_are_intact()


@dataclass(frozen=True, slots=True)
class EventRemeshPersistentEPIError:
    """Exact pairwise EPI L-infinity error on persistent node identifiers."""

    left_mesh: str
    right_mesh: str
    checkpoint_kind: str
    parent_interval_index: int | None
    exact_time: Fraction
    nodes: tuple[Hashable, ...]
    exact_left_epi: tuple[Fraction, ...]
    exact_right_epi: tuple[Fraction, ...]
    exact_absolute_epi_errors: tuple[Fraction, ...]
    exact_epi_error_linf: Fraction
    epi_error_linf: float | None
    _proof_stamp: tuple[Any, ...] = field(default=(), repr=False, compare=False)

    def _proof_fields_are_intact(self) -> bool:
        return _sealed(self, EventRemeshPersistentEPIError, _ERROR_PROOF_VERSION)

    @property
    def error_observation_certified(self) -> bool:
        return self._proof_fields_are_intact()

    @property
    def checkpoint_key(self) -> tuple[str, int | None, Fraction]:
        return (
            self.checkpoint_kind,
            self.parent_interval_index,
            self.exact_time,
        )


@dataclass(frozen=True, slots=True)
class EventRemeshThreeMeshZHIRObservation:
    """Executed physical pre-jump rate and gate comparison across three meshes."""

    event_identity: tuple[int, int, int, str, str]
    nodes: tuple[Hashable, ...]
    coarse_observation: ExecutedEventLocalZHIRPhysicalPrejumpObservation = field(
        repr=False
    )
    intermediate_observation: ExecutedEventLocalZHIRPhysicalPrejumpObservation = field(
        repr=False
    )
    fine_observation: ExecutedEventLocalZHIRPhysicalPrejumpObservation = field(
        repr=False
    )
    coarse_terminal_exact_binary64_rates: tuple[Fraction, ...]
    intermediate_terminal_exact_binary64_rates: tuple[Fraction, ...]
    fine_terminal_exact_binary64_rates: tuple[Fraction, ...]
    coarse_terminal_gate_decisions: tuple[bool, ...]
    intermediate_terminal_gate_decisions: tuple[bool, ...]
    fine_terminal_gate_decisions: tuple[bool, ...]
    terminal_gate_decisions_agree_by_node: tuple[bool, ...]
    terminal_gate_decisions_agree: bool
    exact_coarse_intermediate_terminal_rate_error_linf: Fraction
    exact_intermediate_fine_terminal_rate_error_linf: Fraction
    exact_coarse_fine_terminal_rate_error_linf: Fraction
    coarse_whole_parent_exact_binary64_rates: tuple[Fraction, ...]
    intermediate_whole_parent_exact_binary64_rates: tuple[Fraction, ...]
    fine_whole_parent_exact_binary64_rates: tuple[Fraction, ...]
    coarse_whole_parent_gate_decisions: tuple[bool, ...]
    intermediate_whole_parent_gate_decisions: tuple[bool, ...]
    fine_whole_parent_gate_decisions: tuple[bool, ...]
    whole_parent_gate_decisions_agree_by_node: tuple[bool, ...]
    whole_parent_gate_decisions_agree: bool
    exact_coarse_intermediate_whole_parent_rate_error_linf: Fraction
    exact_intermediate_fine_whole_parent_rate_error_linf: Fraction
    exact_coarse_fine_whole_parent_rate_error_linf: Fraction
    _proof_stamp: tuple[Any, ...] = field(default=(), repr=False, compare=False)

    def _proof_fields_are_intact(self) -> bool:
        observations = (
            self.coarse_observation,
            self.intermediate_observation,
            self.fine_observation,
        )
        return bool(
            _sealed(
                self,
                EventRemeshThreeMeshZHIRObservation,
                _ZHIR_PROOF_VERSION,
            )
            and all(
                type(item) is ExecutedEventLocalZHIRPhysicalPrejumpObservation
                and item.common_execution_provenance_certified
                for item in observations
            )
        )

    @property
    def physical_zhir_comparison_certified(self) -> bool:
        return self._proof_fields_are_intact()


@dataclass(frozen=True, slots=True)
class EventRemeshThreeMeshModalObservation:
    """Modal factors for one parent interval, or an explicit abstention."""

    parent_interval_index: int
    applicable: bool
    abstention_reason: str | None
    coarse_exact_segment_durations: tuple[Fraction, ...]
    intermediate_exact_segment_durations: tuple[Fraction, ...]
    fine_exact_segment_durations: tuple[Fraction, ...]
    nodes: tuple[Hashable, ...] | None
    common_decay_rates: tuple[float, ...] | None
    coarse_composed_modal_factors: tuple[float, ...] | None
    intermediate_composed_modal_factors: tuple[float, ...] | None
    fine_composed_modal_factors: tuple[float, ...] | None
    coarse_segment_stability_decisions: tuple[bool, ...] | None
    intermediate_segment_stability_decisions: tuple[bool, ...] | None
    fine_segment_stability_decisions: tuple[bool, ...] | None
    coarse_composed_stable: bool | None
    intermediate_composed_stable: bool | None
    fine_composed_stable: bool | None
    composed_stability_decisions_agree: bool | None
    _proof_stamp: tuple[Any, ...] = field(default=(), repr=False, compare=False)

    def _proof_fields_are_intact(self) -> bool:
        return _sealed(self, EventRemeshThreeMeshModalObservation, _MODAL_PROOF_VERSION)

    @property
    def common_generator_modal_factors_observed(self) -> bool:
        return bool(self._proof_fields_are_intact() and self.applicable)


@dataclass(frozen=True, slots=True)
class EventRemeshThreeMeshRefinementObservation:
    """Sealed finite coarse/intermediate/fine event/REMESH comparison."""

    coarse: EventRemeshMeshObservation
    intermediate: EventRemeshMeshObservation
    fine: EventRemeshMeshObservation
    persistent_nodes: tuple[Hashable, ...]
    supports_equal_across_meshes: bool
    coarse_to_intermediate_strict_refinement: bool
    intermediate_to_fine_strict_refinement: bool
    coarse_intermediate_epi_errors: tuple[EventRemeshPersistentEPIError, ...]
    intermediate_fine_epi_errors: tuple[EventRemeshPersistentEPIError, ...]
    coarse_fine_epi_errors: tuple[EventRemeshPersistentEPIError, ...]
    zhir_xi: float | None
    zhir_observations: tuple[EventRemeshThreeMeshZHIRObservation, ...]
    zhir_abstention_reason: str | None
    modal_observations: tuple[EventRemeshThreeMeshModalObservation, ...]
    conditions: tuple[tuple[str, bool], ...]
    _proof_stamp: tuple[Any, ...] = field(default=(), repr=False, compare=False)

    def _proof_fields_are_intact(self) -> bool:
        return bool(
            _sealed(
                self,
                EventRemeshThreeMeshRefinementObservation,
                _THREE_MESH_PROOF_VERSION,
            )
            and all(
                type(item) is EventRemeshMeshObservation
                and item._proof_fields_are_intact()
                for item in (self.coarse, self.intermediate, self.fine)
            )
            and all(
                type(item) is EventRemeshPersistentEPIError
                and item._proof_fields_are_intact()
                for group in (
                    self.coarse_intermediate_epi_errors,
                    self.intermediate_fine_epi_errors,
                    self.coarse_fine_epi_errors,
                )
                for item in group
            )
            and all(
                type(item) is EventRemeshThreeMeshZHIRObservation
                and item._proof_fields_are_intact()
                for item in self.zhir_observations
            )
            and all(
                type(item) is EventRemeshThreeMeshModalObservation
                and item._proof_fields_are_intact()
                for item in self.modal_observations
            )
        )

    @property
    def scope(self) -> str:
        return _SCOPE

    @property
    def three_mesh_observation_certified(self) -> bool:
        return bool(
            self._proof_fields_are_intact()
            and all(passed for _, passed in self.conditions)
        )

    @property
    def failed_conditions(self) -> tuple[str, ...]:
        if not self._proof_fields_are_intact():
            return ("three_mesh_proof_fields_intact",)
        return tuple(name for name, passed in self.conditions if not passed)

    @property
    def schedule_compositions(
        self,
    ) -> tuple[ObservedRepresentedEPIScheduleComposition | None, ...]:
        return (
            self.coarse.schedule_composition,
            self.intermediate.schedule_composition,
            self.fine.schedule_composition,
        )

    @property
    def remesh_results(self) -> tuple[DelayedRemeshResult, ...]:
        return (
            self.coarse.remesh_result,
            self.intermediate.remesh_result,
            self.fine.remesh_result,
        )

    @property
    def maximum_coarse_intermediate_epi_error_linf(self) -> Fraction | None:
        if not self._proof_fields_are_intact():
            return None
        return max(
            (
                item.exact_epi_error_linf
                for item in self.coarse_intermediate_epi_errors
            ),
            default=None,
        )

    @property
    def maximum_intermediate_fine_epi_error_linf(self) -> Fraction | None:
        if not self._proof_fields_are_intact():
            return None
        return max(
            (
                item.exact_epi_error_linf
                for item in self.intermediate_fine_epi_errors
            ),
            default=None,
        )

    @property
    def maximum_coarse_fine_epi_error_linf(self) -> Fraction | None:
        if not self._proof_fields_are_intact():
            return None
        return max(
            (item.exact_epi_error_linf for item in self.coarse_fine_epi_errors),
            default=None,
        )

    @property
    def intermediate_fine_error_decreases_at_coarse_checkpoints(self) -> bool:
        if not self._proof_fields_are_intact():
            return False
        intermediate_fine = {
            item.checkpoint_key: item.exact_epi_error_linf
            for item in self.intermediate_fine_epi_errors
        }
        comparisons = tuple(
            (
                item.exact_epi_error_linf,
                intermediate_fine[item.checkpoint_key],
            )
            for item in self.coarse_intermediate_epi_errors
        )
        return bool(
            comparisons
            and all(finer <= coarser for coarser, finer in comparisons)
            and any(finer < coarser for coarser, finer in comparisons)
        )

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
    def lyapunov_decrease_certified(self) -> bool:
        return False

    @property
    def runtime_global_gain_certified(self) -> bool:
        return False

    @property
    def future_or_repeated_behavior_certified(self) -> bool:
        return False

    @property
    def combined_schedule_remesh_gain_certified(self) -> bool:
        return False

    @property
    def whole_three_mesh_atomicity_certified(self) -> bool:
        return False

    @property
    def complete_reference_problem_certified(self) -> bool:
        """Detached cycles do not retain every pre-schedule configuration input."""

        return False

    @property
    def epi_differences_attributable_only_to_mesh_certified(self) -> bool:
        """Keep causal mesh attribution outside the available runtime artifacts."""

        return False


def _validate_cycle(value: Any, mesh_name: str) -> EventRemeshCycleResult:
    if type(value) is not EventRemeshCycleResult:
        raise TypeError(f"{mesh_name} must be an EventRemeshCycleResult")
    if not value._proof_fields_are_intact():
        raise ValueError(f"{mesh_name} cycle result is unsealed, tampered, or stale")
    execution = value.event_execution
    if not execution.physical_pressure_reevaluated_partitions_established:
        raise ValueError(
            f"{mesh_name} cycle lacks intact physical partition evidence"
        )
    if (
        execution.positive_flow_interval_indices
        != execution.physical_flow_partition_indices
        or not execution.physical_flow_partition_indices
    ):
        raise ValueError(
            f"{mesh_name} cycle must physically partition every positive interval"
        )
    return value


def _checkpoint(
    *,
    mesh_name: str,
    checkpoint_kind: str,
    parent_interval_index: int | None,
    boundary_index: int | None,
    time: float,
    exact_time: Fraction,
    nodes: tuple[Hashable, ...],
    epi_values: tuple[float, ...],
    exact_epi_values: tuple[Fraction, ...],
) -> EventRemeshEPICheckpointObservation:
    value = EventRemeshEPICheckpointObservation(
        mesh_name=mesh_name,
        checkpoint_kind=checkpoint_kind,
        parent_interval_index=parent_interval_index,
        boundary_index=boundary_index,
        time=time,
        exact_time=exact_time,
        nodes=nodes,
        epi_values=epi_values,
        exact_epi_values=exact_epi_values,
    )
    return _seal(
        value,
        EventRemeshEPICheckpointObservation,
        _CHECKPOINT_PROOF_VERSION,
    )


def _mesh_observation(
    mesh_name: str,
    cycle: EventRemeshCycleResult,
) -> EventRemeshMeshObservation:
    execution = cycle.event_execution
    schedule = execution.schedule
    checkpoints: list[EventRemeshEPICheckpointObservation] = [
        _checkpoint(
            mesh_name=mesh_name,
            checkpoint_kind="pre_schedule",
            parent_interval_index=None,
            boundary_index=None,
            time=schedule.start_time,
            exact_time=schedule.exact_start_time,
            nodes=cycle.pre_schedule_epi.nodes,
            epi_values=cycle.pre_schedule_epi.epi_values,
            exact_epi_values=_exact_vector(cycle.pre_schedule_epi.epi_values),
        )
    ]
    boundary_rows: list[tuple[int, tuple[Fraction, ...]]] = []
    for evidence in execution.physical_flow_partition_evidence:
        if (
            type(evidence) is not ExecutedPressureRefreshedFlowPartition
            or not evidence._proof_fields_are_intact()
        ):
            raise ValueError(f"{mesh_name} contains invalid physical evidence")
        parent_index = evidence.partition.parent_interval.index
        boundary_rows.append(
            (parent_index, evidence.partition.exact_boundary_times)
        )
        for boundary in evidence.boundary_observations:
            snapshot = boundary.after
            checkpoints.append(
                _checkpoint(
                    mesh_name=mesh_name,
                    checkpoint_kind="physical_flow_boundary",
                    parent_interval_index=parent_index,
                    boundary_index=boundary.boundary_index,
                    time=boundary.time,
                    exact_time=boundary.exact_time,
                    nodes=snapshot.nodes,
                    epi_values=snapshot.epi,
                    exact_epi_values=snapshot.exact_epi,
                )
            )
    checkpoints.extend(
        (
            _checkpoint(
                mesh_name=mesh_name,
                checkpoint_kind="pre_remesh",
                parent_interval_index=None,
                boundary_index=None,
                time=schedule.end_time,
                exact_time=schedule.exact_end_time,
                nodes=cycle.pre_remesh_epi.nodes,
                epi_values=cycle.pre_remesh_epi.epi_values,
                exact_epi_values=_exact_vector(cycle.pre_remesh_epi.epi_values),
            ),
            _checkpoint(
                mesh_name=mesh_name,
                checkpoint_kind="post_remesh",
                parent_interval_index=None,
                boundary_index=None,
                time=schedule.end_time,
                exact_time=schedule.exact_end_time,
                nodes=cycle.post_remesh_epi.nodes,
                epi_values=cycle.post_remesh_epi.epi_values,
                exact_epi_values=_exact_vector(cycle.post_remesh_epi.epi_values),
            ),
        )
    )
    node_tokens = _node_tokens(cycle.target_nodes, f"{mesh_name} cycle support")
    for item in checkpoints:
        if _node_tokens(item.nodes, f"{mesh_name} checkpoint support") != node_tokens:
            raise ValueError(f"{mesh_name} checkpoint changed ordered node support")
    keys = tuple(item.key for item in checkpoints)
    if len(set(keys)) != len(keys):
        raise ValueError(f"{mesh_name} contains duplicate checkpoint coordinates")
    value = EventRemeshMeshObservation(
        mesh_name=mesh_name,
        cycle_result=cycle,
        nodes=cycle.target_nodes,
        physical_partition_interval_indices=(
            execution.physical_flow_partition_indices
        ),
        exact_partition_boundary_times=tuple(boundary_rows),
        checkpoints=tuple(checkpoints),
        schedule_composition=execution.represented_epi_schedule_composition,
        remesh_result=cycle.remesh,
    )
    return _seal(value, EventRemeshMeshObservation, _MESH_PROOF_VERSION)


def _strictly_nested_boundaries(
    coarser: EventRemeshMeshObservation,
    finer: EventRemeshMeshObservation,
) -> bool:
    if (
        coarser.physical_partition_interval_indices
        != finer.physical_partition_interval_indices
    ):
        return False
    finer_rows = dict(finer.exact_partition_boundary_times)
    for index, coarse_boundaries in coarser.exact_partition_boundary_times:
        fine_boundaries = finer_rows.get(index)
        if (
            fine_boundaries is None
            or len(fine_boundaries) <= len(coarse_boundaries)
            or not set(coarse_boundaries).issubset(fine_boundaries)
            or fine_boundaries[0] != coarse_boundaries[0]
            or fine_boundaries[-1] != coarse_boundaries[-1]
        ):
            return False
    return True


def _projected_checkpoint(
    checkpoint: EventRemeshEPICheckpointObservation,
    persistent_tokens: tuple[tuple[Any, ...], ...],
) -> tuple[Fraction, ...]:
    return _project_values(
        checkpoint.nodes,
        checkpoint.exact_epi_values,
        persistent_tokens,
        label=f"{checkpoint.mesh_name} checkpoint",
    )


def _pairwise_errors(
    left: EventRemeshMeshObservation,
    right: EventRemeshMeshObservation,
    persistent_nodes: tuple[Hashable, ...],
    persistent_tokens: tuple[tuple[Any, ...], ...],
) -> tuple[EventRemeshPersistentEPIError, ...]:
    right_by_key = {item.key: item for item in right.checkpoints}
    results: list[EventRemeshPersistentEPIError] = []
    for left_checkpoint in left.checkpoints:
        right_checkpoint = right_by_key.get(left_checkpoint.key)
        if right_checkpoint is None:
            raise ValueError(
                f"{right.mesh_name} omits a {left.mesh_name} checkpoint"
            )
        left_values = _projected_checkpoint(left_checkpoint, persistent_tokens)
        right_values = _projected_checkpoint(right_checkpoint, persistent_tokens)
        errors = tuple(
            abs(left_value - right_value)
            for left_value, right_value in zip(
                left_values,
                right_values,
                strict=True,
            )
        )
        linf = max(errors, default=Fraction(0))
        value = EventRemeshPersistentEPIError(
            left_mesh=left.mesh_name,
            right_mesh=right.mesh_name,
            checkpoint_kind=left_checkpoint.checkpoint_kind,
            parent_interval_index=left_checkpoint.parent_interval_index,
            exact_time=left_checkpoint.exact_time,
            nodes=persistent_nodes,
            exact_left_epi=left_values,
            exact_right_epi=right_values,
            exact_absolute_epi_errors=errors,
            exact_epi_error_linf=linf,
            epi_error_linf=_optional_float(linf),
        )
        results.append(
            _seal(value, EventRemeshPersistentEPIError, _ERROR_PROOF_VERSION)
        )
    return tuple(results)


def _max_abs_difference(
    left: tuple[Fraction, ...],
    right: tuple[Fraction, ...],
) -> Fraction:
    return max(
        (
            abs(left_value - right_value)
            for left_value, right_value in zip(left, right, strict=True)
        ),
        default=Fraction(0),
    )


def _project_zhir_field(
    observation: ExecutedEventLocalZHIRPhysicalPrejumpObservation,
    field_name: str,
    persistent_tokens: tuple[tuple[Any, ...], ...],
) -> tuple[Any, ...]:
    physical = observation.physical_observation
    return _project_values(
        physical.nodes,
        object.__getattribute__(physical, field_name),
        persistent_tokens,
        label=f"physical ZHIR {field_name}",
    )


def _zhir_observation(
    coarse: ExecutedEventLocalZHIRPhysicalPrejumpObservation,
    intermediate: ExecutedEventLocalZHIRPhysicalPrejumpObservation,
    fine: ExecutedEventLocalZHIRPhysicalPrejumpObservation,
    persistent_nodes: tuple[Hashable, ...],
    persistent_tokens: tuple[tuple[Any, ...], ...],
    expected_xi: float | None,
) -> EventRemeshThreeMeshZHIRObservation:
    physical = tuple(
        item.physical_observation for item in (coarse, intermediate, fine)
    )
    if not (
        physical[0].event_identity
        == physical[1].event_identity
        == physical[2].event_identity
    ):
        raise ValueError("physical ZHIR observations identify different events")
    exact_xi = tuple(item.exact_xi for item in physical)
    if not exact_xi[0] == exact_xi[1] == exact_xi[2]:
        raise ValueError("executed ZHIR thresholds differ across physical meshes")
    if (
        expected_xi is not None
        and exact_xi[0] != Fraction.from_float(expected_xi)
    ):
        raise ValueError("zhir_xi does not match the executed Mutation threshold")

    terminal_rates = tuple(
        _project_zhir_field(
            item,
            "terminal_exact_binary64_observed_gate_rates",
            persistent_tokens,
        )
        for item in (coarse, intermediate, fine)
    )
    terminal_gates = tuple(
        _project_zhir_field(
            item,
            "terminal_observed_strict_gate_decisions",
            persistent_tokens,
        )
        for item in (coarse, intermediate, fine)
    )
    whole_rates = tuple(
        _project_zhir_field(
            item,
            "whole_parent_exact_binary64_gate_rates",
            persistent_tokens,
        )
        for item in (coarse, intermediate, fine)
    )
    whole_gates = tuple(
        _project_zhir_field(
            item,
            "whole_parent_strict_gate_decisions",
            persistent_tokens,
        )
        for item in (coarse, intermediate, fine)
    )
    terminal_agreement = tuple(
        left == middle == right
        for left, middle, right in zip(*terminal_gates, strict=True)
    )
    whole_agreement = tuple(
        left == middle == right
        for left, middle, right in zip(*whole_gates, strict=True)
    )
    value = EventRemeshThreeMeshZHIRObservation(
        event_identity=physical[0].event_identity,
        nodes=persistent_nodes,
        coarse_observation=coarse,
        intermediate_observation=intermediate,
        fine_observation=fine,
        coarse_terminal_exact_binary64_rates=terminal_rates[0],
        intermediate_terminal_exact_binary64_rates=terminal_rates[1],
        fine_terminal_exact_binary64_rates=terminal_rates[2],
        coarse_terminal_gate_decisions=terminal_gates[0],
        intermediate_terminal_gate_decisions=terminal_gates[1],
        fine_terminal_gate_decisions=terminal_gates[2],
        terminal_gate_decisions_agree_by_node=terminal_agreement,
        terminal_gate_decisions_agree=all(terminal_agreement),
        exact_coarse_intermediate_terminal_rate_error_linf=(
            _max_abs_difference(terminal_rates[0], terminal_rates[1])
        ),
        exact_intermediate_fine_terminal_rate_error_linf=(
            _max_abs_difference(terminal_rates[1], terminal_rates[2])
        ),
        exact_coarse_fine_terminal_rate_error_linf=(
            _max_abs_difference(terminal_rates[0], terminal_rates[2])
        ),
        coarse_whole_parent_exact_binary64_rates=whole_rates[0],
        intermediate_whole_parent_exact_binary64_rates=whole_rates[1],
        fine_whole_parent_exact_binary64_rates=whole_rates[2],
        coarse_whole_parent_gate_decisions=whole_gates[0],
        intermediate_whole_parent_gate_decisions=whole_gates[1],
        fine_whole_parent_gate_decisions=whole_gates[2],
        whole_parent_gate_decisions_agree_by_node=whole_agreement,
        whole_parent_gate_decisions_agree=all(whole_agreement),
        exact_coarse_intermediate_whole_parent_rate_error_linf=(
            _max_abs_difference(whole_rates[0], whole_rates[1])
        ),
        exact_intermediate_fine_whole_parent_rate_error_linf=(
            _max_abs_difference(whole_rates[1], whole_rates[2])
        ),
        exact_coarse_fine_whole_parent_rate_error_linf=(
            _max_abs_difference(whole_rates[0], whole_rates[2])
        ),
    )
    return _seal(value, EventRemeshThreeMeshZHIRObservation, _ZHIR_PROOF_VERSION)


def _partition_by_index(
    mesh: EventRemeshMeshObservation,
) -> dict[int, ExecutedPressureRefreshedFlowPartition]:
    evidence = mesh.cycle_result.event_execution.physical_flow_partition_evidence
    return {
        item.partition.parent_interval.index: item
        for item in evidence
    }


def _physical_zhir_observations(
    coarse: EventRemeshMeshObservation,
    intermediate: EventRemeshMeshObservation,
    fine: EventRemeshMeshObservation,
    persistent_nodes: tuple[Hashable, ...],
    persistent_tokens: tuple[tuple[Any, ...], ...],
    xi: float | None,
) -> tuple[tuple[EventRemeshThreeMeshZHIRObservation, ...], str | None]:
    events = tuple(
        event
        for event in coarse.cycle_result.event_execution.schedule.events
        if event.glyph is Glyph.ZHIR and event.operator_name == "mutation"
    )
    if not events:
        return (), "schedule_has_no_zhir_event"
    executions = tuple(
        mesh.cycle_result.event_execution
        for mesh in (coarse, intermediate, fine)
    )
    if not all(execution.stage_certification_requested for execution in executions):
        return (), "zhir_stage_certificates_not_available_for_all_meshes"
    observations: list[EventRemeshThreeMeshZHIRObservation] = []
    for event in events:
        executed = tuple(
            observe_executed_event_local_zhir_physical_prejump(
                execution,
                event_index=event.event_index,
            )
            for execution in executions
        )
        observations.append(
            _zhir_observation(
                executed[0],
                executed[1],
                executed[2],
                persistent_nodes,
                persistent_tokens,
                xi,
            )
        )
    return tuple(observations), None


def _generator_signature(
    evidence: ExecutedPressureRefreshedFlowPartition,
) -> tuple[Any, ...] | None:
    if (
        not evidence.all_segment_modal_diagnostics_applicable
        or not evidence.all_segment_binary64_replays_identified
    ):
        return None
    starts = evidence.boundary_observations[:-1]
    if not starts:
        return None
    signatures = tuple(
        (
            _proof_value(boundary.after.nodes),
            boundary.after.exact_nu_f,
            boundary.after.conductance,
        )
        for boundary in starts
    )
    if any(item != signatures[0] for item in signatures[1:]):
        return None
    for flow in evidence.segment_flow_evidence:
        certificate = flow.certificate
        if certificate is None:
            return None
        for snapshot in (certificate.left, certificate.right):
            signature = (
                _proof_value(snapshot.nodes),
                snapshot.exact_nu_f,
                snapshot.conductance,
            )
            if signature != signatures[0]:
                return None
    return signatures[0]


def _modal_rows(
    evidence: ExecutedPressureRefreshedFlowPartition,
) -> tuple[
    tuple[float, ...],
    tuple[tuple[float, ...], ...],
] | None:
    observations = evidence.modal_observations
    if not observations:
        return None
    first_rates = observations[0].decay_rates
    first_target = observations[0].target_fraction
    first_tolerance = observations[0].spectral_relative_tolerance
    if first_rates is None:
        return None
    rows: list[tuple[float, ...]] = []
    for item in observations:
        multipliers = item.modal_multipliers
        if (
            item.decay_rates is None
            or not binary64_vectors_are_identical(first_rates, item.decay_rates)
            or _proof_value(first_target) != _proof_value(item.target_fraction)
            or _proof_value(first_tolerance)
            != _proof_value(item.spectral_relative_tolerance)
            or multipliers is None
            or len(multipliers) != len(first_rates)
        ):
            return None
        rows.append(multipliers)
    return first_rates, tuple(rows)


def _composed_modal_factors(
    rows: tuple[tuple[float, ...], ...],
) -> tuple[float, ...] | None:
    if not rows:
        return None
    factors = tuple(
        math.prod(row[index] for row in rows)
        for index in range(len(rows[0]))
    )
    if any(not math.isfinite(value) for value in factors):
        return None
    return factors


def _modal_observation(
    index: int,
    evidences: tuple[
        ExecutedPressureRefreshedFlowPartition,
        ExecutedPressureRefreshedFlowPartition,
        ExecutedPressureRefreshedFlowPartition,
    ],
) -> EventRemeshThreeMeshModalObservation:
    durations = tuple(
        tuple(segment.exact_duration for segment in item.partition.segments)
        for item in evidences
    )
    supports = tuple(item.boundary_observations[0].after.nodes for item in evidences)
    signatures = tuple(_generator_signature(item) for item in evidences)
    rows = tuple(_modal_rows(item) for item in evidences)
    reason: str | None = None
    if any(item is None for item in signatures):
        reason = "modal_diagnostics_or_trusted_segment_replays_unavailable"
    elif not (
        _ordered_supports_match(supports[0], supports[1])
        and _ordered_supports_match(supports[1], supports[2])
    ):
        reason = "physical_meshes_do_not_share_one_generator"
    elif not signatures[0] == signatures[1] == signatures[2]:
        reason = "physical_meshes_do_not_share_one_generator"
    elif any(item is None for item in rows):
        reason = "modal_rows_are_incomplete_or_change_within_a_mesh"
    else:
        typed_rows = tuple(item for item in rows if item is not None)
        rates = tuple(item[0] for item in typed_rows)
        if not (
            binary64_vectors_are_identical(rates[0], rates[1])
            and binary64_vectors_are_identical(rates[1], rates[2])
        ):
            reason = "physical_meshes_have_different_modal_decay_rates"

    factors: tuple[tuple[float, ...], ...] = ()
    if reason is None:
        typed_rows = tuple(item for item in rows if item is not None)
        computed = tuple(_composed_modal_factors(item[1]) for item in typed_rows)
        if any(item is None for item in computed):
            reason = "composed_modal_factors_are_unavailable_or_nonfinite"
        else:
            factors = tuple(item for item in computed if item is not None)

    if reason is not None:
        value = EventRemeshThreeMeshModalObservation(
            parent_interval_index=index,
            applicable=False,
            abstention_reason=reason,
            coarse_exact_segment_durations=durations[0],
            intermediate_exact_segment_durations=durations[1],
            fine_exact_segment_durations=durations[2],
            nodes=None,
            common_decay_rates=None,
            coarse_composed_modal_factors=None,
            intermediate_composed_modal_factors=None,
            fine_composed_modal_factors=None,
            coarse_segment_stability_decisions=None,
            intermediate_segment_stability_decisions=None,
            fine_segment_stability_decisions=None,
            coarse_composed_stable=None,
            intermediate_composed_stable=None,
            fine_composed_stable=None,
            composed_stability_decisions_agree=None,
        )
        return _seal(
            value,
            EventRemeshThreeMeshModalObservation,
            _MODAL_PROOF_VERSION,
        )

    modal_rates = rows[0][0] if rows[0] is not None else ()
    nodes = evidences[0].boundary_observations[0].after.nodes
    segment_stability = tuple(
        tuple(bool(item.is_euler_stable) for item in evidence.modal_observations)
        for evidence in evidences
    )
    stable = tuple(max(abs(value) for value in item) < 1.0 for item in factors)
    value = EventRemeshThreeMeshModalObservation(
        parent_interval_index=index,
        applicable=True,
        abstention_reason=None,
        coarse_exact_segment_durations=durations[0],
        intermediate_exact_segment_durations=durations[1],
        fine_exact_segment_durations=durations[2],
        nodes=nodes,
        common_decay_rates=modal_rates,
        coarse_composed_modal_factors=factors[0],
        intermediate_composed_modal_factors=factors[1],
        fine_composed_modal_factors=factors[2],
        coarse_segment_stability_decisions=segment_stability[0],
        intermediate_segment_stability_decisions=segment_stability[1],
        fine_segment_stability_decisions=segment_stability[2],
        coarse_composed_stable=stable[0],
        intermediate_composed_stable=stable[1],
        fine_composed_stable=stable[2],
        composed_stability_decisions_agree=stable[0] == stable[1] == stable[2],
    )
    return _seal(
        value,
        EventRemeshThreeMeshModalObservation,
        _MODAL_PROOF_VERSION,
    )


def _modal_observations(
    coarse: EventRemeshMeshObservation,
    intermediate: EventRemeshMeshObservation,
    fine: EventRemeshMeshObservation,
) -> tuple[EventRemeshThreeMeshModalObservation, ...]:
    maps = tuple(_partition_by_index(mesh) for mesh in (coarse, intermediate, fine))
    return tuple(
        _modal_observation(
            index,
            (maps[0][index], maps[1][index], maps[2][index]),
        )
        for index in coarse.physical_partition_interval_indices
    )


def _project_history(
    mesh: EventRemeshMeshObservation,
    persistent_tokens: tuple[tuple[Any, ...], ...],
) -> tuple[tuple[Fraction, ...], ...]:
    transition = mesh.cycle_result.history_transition
    return tuple(
        _project_values(
            transition.nodes,
            snapshot,
            persistent_tokens,
            label=f"{mesh.mesh_name} incoming REMESH history",
        )
        for snapshot in transition.incoming_exact_history
    )


def _remesh_configuration(mesh: EventRemeshMeshObservation) -> tuple[Any, ...]:
    plan = mesh.remesh_result.plan
    return (
        plan.status,
        plan.history_length,
        plan.required_history_length,
        Fraction.from_float(plan.alpha),
        plan.alpha_source,
        plan.tau_local,
        plan.tau_global,
        Fraction.from_float(plan.epi_min),
        Fraction.from_float(plan.epi_max),
        plan.clip_mode,
        mesh.cycle_result.history_transition.history_maxlen,
        mesh.cycle_result.post_remesh_pressure_refresh_requested,
    )


def _schedule_signature(mesh: EventRemeshMeshObservation) -> tuple[Any, ...]:
    schedule = mesh.cycle_result.event_execution.schedule
    return (
        schedule.operator_names,
        schedule.cycles,
        schedule.start_time.hex(),
        tuple(value.hex() for value in schedule.flow_durations),
        tuple(
            (
                interval.index,
                interval.start_time.hex(),
                interval.end_time.hex(),
                interval.duration.hex(),
                interval.start_offset,
                interval.end_offset,
                interval.exact_start_time,
                interval.exact_end_time,
                interval.exact_duration,
            )
            for interval in schedule.intervals
        ),
        tuple(
            (
                event.event_index,
                event.cycle_index,
                event.word_position,
                event.operator_name,
                event.glyph.value,
                event.event_time.hex(),
                event.event_offset,
                event.exact_event_time,
            )
            for event in schedule.events
        ),
        schedule.end_time.hex(),
        schedule.total_flow_duration.hex(),
        schedule.exact_start_time,
        schedule.exact_end_time,
        schedule.exact_total_flow_duration,
    )


def _captured_conductance_signature(
    mesh: EventRemeshMeshObservation,
) -> tuple[tuple[int, tuple[tuple[Fraction, ...], ...]], ...]:
    """Return effective conductance at every captured parent-interval start."""

    rows: list[tuple[int, tuple[tuple[Fraction, ...], ...]]] = []
    for evidence in mesh.cycle_result.event_execution.physical_flow_partition_evidence:
        initial = evidence.boundary_observations[0].before
        if not _ordered_supports_match(mesh.nodes, initial.nodes):
            raise ValueError(
                f"{mesh.mesh_name} captured conductance changed ordered support"
            )
        rows.append((evidence.partition.parent_interval.index, initial.conductance))
    return tuple(rows)


def _first_captured_flow_input_signature(
    mesh: EventRemeshMeshObservation,
) -> tuple[Any, ...]:
    """Seal the earliest post-refresh nodal input exposed by the cycle trace."""

    evidence = mesh.cycle_result.event_execution.physical_flow_partition_evidence[0]
    boundary = evidence.boundary_observations[0]
    snapshot = boundary.after
    if not _ordered_supports_match(mesh.nodes, snapshot.nodes):
        raise ValueError(
            f"{mesh.mesh_name} first captured flow input changed ordered support"
        )
    return (
        _proof_value(snapshot.nodes),
        snapshot.exact_epi,
        snapshot.exact_nu_f,
        snapshot.exact_delta_nfr,
    )


def _integrator_execution_signature(
    mesh: EventRemeshMeshObservation,
) -> tuple[Any, ...]:
    """Compress common integrator policy independent of mesh resolution.

    ``resolved_substeps`` is validated inside each sealed flow certificate but
    is intentionally absent here: a fixed ``DT_MIN`` can resolve different
    substep counts when physical segment durations are refined.
    """

    execution = mesh.cycle_result.event_execution
    rows = tuple(
        (
            flow.integrator_name,
            flow.integrator_provenance_certified,
            flow.resolved_method,
            flow.gamma_is_none,
            flow.extended_dynamics_requested,
        )
        for evidence in execution.physical_flow_partition_evidence
        for flow in evidence.segment_flow_evidence
    )
    if not rows or any(row != rows[0] for row in rows[1:]):
        raise ValueError(
            f"{mesh.mesh_name} integrator metadata changes within its physical mesh"
        )
    return (execution.integrator_name, rows[0])


def _pressure_callback_signature(mesh: EventRemeshMeshObservation) -> tuple[str, bool]:
    """Compress the callback metadata recorded at physical boundaries."""

    rows = tuple(
        (boundary.callback_name, boundary.configured_callback)
        for evidence in (
            mesh.cycle_result.event_execution.physical_flow_partition_evidence
        )
        for boundary in evidence.boundary_observations
    )
    if not rows or any(row != rows[0] for row in rows[1:]):
        raise ValueError(
            f"{mesh.mesh_name} pressure callback metadata changes within its mesh"
        )
    return rows[0]


def _same_binary64_vectors(vectors: tuple[tuple[float, ...], ...]) -> bool:
    return bool(
        binary64_vectors_are_identical(vectors[0], vectors[1])
        and binary64_vectors_are_identical(vectors[1], vectors[2])
    )


def _require_common_reference_problem(
    meshes: tuple[
        EventRemeshMeshObservation,
        EventRemeshMeshObservation,
        EventRemeshMeshObservation,
    ],
    persistent_tokens: tuple[tuple[Any, ...], ...],
) -> None:
    if not (
        _ordered_supports_match(meshes[0].nodes, meshes[1].nodes)
        and _ordered_supports_match(meshes[1].nodes, meshes[2].nodes)
    ):
        raise ValueError(
            "coarse, intermediate and fine require identical ordered full support"
        )
    schedules = tuple(_schedule_signature(mesh) for mesh in meshes)
    if not schedules[0] == schedules[1] == schedules[2]:
        raise ValueError("coarse, intermediate and fine schedules are incompatible")
    initial_epi = tuple(
        mesh.cycle_result.pre_schedule_epi.epi_values for mesh in meshes
    )
    if not _same_binary64_vectors(initial_epi):
        raise ValueError(
            "coarse, intermediate and fine pre-schedule EPI endpoints are incompatible"
        )
    initial_capacity = tuple(
        mesh.cycle_result.capacity_before_schedule for mesh in meshes
    )
    if not _same_binary64_vectors(initial_capacity):
        raise ValueError("initial capacity differs across physical meshes")
    initial_phase = tuple(
        mesh.cycle_result.phase_before_schedule for mesh in meshes
    )
    if not _same_binary64_vectors(initial_phase):
        raise ValueError("initial phase differs across physical meshes")
    initial_pressure = tuple(
        mesh.cycle_result.pressure_before_schedule for mesh in meshes
    )
    if not _same_binary64_vectors(initial_pressure):
        raise ValueError("initial pressure differs across physical meshes")

    captured_inputs = tuple(
        _first_captured_flow_input_signature(mesh) for mesh in meshes
    )
    if not captured_inputs[0] == captured_inputs[1] == captured_inputs[2]:
        raise ValueError(
            "first captured post-refresh nodal inputs differ across physical meshes"
        )

    conductance = tuple(_captured_conductance_signature(mesh) for mesh in meshes)
    if not conductance[0] == conductance[1] == conductance[2]:
        raise ValueError(
            "captured topology or effective conductance differs across physical meshes"
        )

    metrics = tuple(
        normalized_positive_binary64_metric(mesh.cycle_result.metric_weights)
        for mesh in meshes
    )
    if any(metric is None for metric in metrics):
        raise ValueError("cycle metric cannot be normalized exactly")
    if not metrics[0] == metrics[1] == metrics[2]:
        raise ValueError("normalized cycle metric differs across physical meshes")

    integrators = tuple(_integrator_execution_signature(mesh) for mesh in meshes)
    if not integrators[0] == integrators[1] == integrators[2]:
        raise ValueError("executor or integrator metadata differs across meshes")
    callbacks = tuple(_pressure_callback_signature(mesh) for mesh in meshes)
    if not callbacks[0] == callbacks[1] == callbacks[2]:
        raise ValueError("pressure callback metadata differs across physical meshes")

    histories = tuple(_project_history(mesh, persistent_tokens) for mesh in meshes)
    if not histories[0] == histories[1] == histories[2]:
        raise ValueError("incoming REMESH histories are incompatible")
    remesh = tuple(_remesh_configuration(mesh) for mesh in meshes)
    if not remesh[0] == remesh[1] == remesh[2]:
        raise ValueError("REMESH configurations are incompatible across meshes")


def _materialize_xi(value: Real | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError("zhir_xi must be a finite nonnegative real scalar or None")
    try:
        result = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError("zhir_xi must be representable as binary64") from exc
    if not math.isfinite(result) or result < 0.0:
        raise ValueError("zhir_xi must be finite and nonnegative")
    return result


def observe_event_remesh_three_mesh_refinement(
    coarse: EventRemeshCycleResult,
    intermediate: EventRemeshCycleResult,
    fine: EventRemeshCycleResult,
    *,
    zhir_xi: Real | None = None,
) -> EventRemeshThreeMeshRefinementObservation:
    """Compare three sealed cycles on strictly nested physical partitions.

    The function is pure: it does not execute or mutate a graph.  The finest
    execution is retained as an observation, not promoted to an exact solution.
    Pairwise errors therefore remain empirical finite-mesh differences. When
    supplied, ``zhir_xi`` checks the threshold sealed into each executed Mutation
    stage; it is never substituted for that executor-owned evidence.
    """

    cycles = tuple(
        _validate_cycle(value, name)
        for value, name in zip(
            (coarse, intermediate, fine),
            _MESH_NAMES,
            strict=True,
        )
    )
    meshes = tuple(
        _mesh_observation(name, cycle)
        for name, cycle in zip(_MESH_NAMES, cycles, strict=True)
    )
    typed_meshes = (meshes[0], meshes[1], meshes[2])
    coarse_to_intermediate = _strictly_nested_boundaries(meshes[0], meshes[1])
    intermediate_to_fine = _strictly_nested_boundaries(meshes[1], meshes[2])
    if not coarse_to_intermediate:
        raise ValueError(
            "intermediate physical checkpoints must strictly refine coarse checkpoints"
        )
    if not intermediate_to_fine:
        raise ValueError(
            "fine physical checkpoints must strictly refine intermediate checkpoints"
        )

    persistent_nodes, persistent_tokens = _persistent_support(
        meshes[0].nodes,
        meshes[1].nodes,
        meshes[2].nodes,
    )
    _require_common_reference_problem(typed_meshes, persistent_tokens)
    coarse_intermediate_errors = _pairwise_errors(
        meshes[0], meshes[1], persistent_nodes, persistent_tokens
    )
    intermediate_fine_errors = _pairwise_errors(
        meshes[1], meshes[2], persistent_nodes, persistent_tokens
    )
    coarse_fine_errors = _pairwise_errors(
        meshes[0], meshes[2], persistent_nodes, persistent_tokens
    )
    xi = _materialize_xi(zhir_xi)
    zhir, zhir_abstention = _physical_zhir_observations(
        meshes[0],
        meshes[1],
        meshes[2],
        persistent_nodes,
        persistent_tokens,
        xi,
    )
    modal = _modal_observations(meshes[0], meshes[1], meshes[2])
    supports_equal = bool(
        _ordered_supports_match(meshes[0].nodes, meshes[1].nodes)
        and _ordered_supports_match(meshes[1].nodes, meshes[2].nodes)
    )
    conditions = (
        ("coarse_cycle_proof_intact", True),
        ("intermediate_cycle_proof_intact", True),
        ("fine_cycle_proof_intact", True),
        ("same_schedule", True),
        ("every_positive_interval_physically_partitioned", True),
        ("identical_ordered_full_support", supports_equal),
        ("coarse_to_intermediate_strict_refinement", coarse_to_intermediate),
        ("intermediate_to_fine_strict_refinement", intermediate_to_fine),
        ("persistent_node_support_nonempty", bool(persistent_nodes)),
        ("initial_epi_capacity_phase_pressure_compatible", True),
        ("first_captured_post_refresh_nodal_input_compatible", True),
        ("captured_effective_conductance_compatible", True),
        ("normalized_metric_compatible", True),
        ("executor_integrator_metadata_compatible", True),
        ("pressure_callback_metadata_compatible", True),
        ("incoming_remesh_history_compatible", True),
        ("remesh_configuration_compatible", True),
    )
    value = EventRemeshThreeMeshRefinementObservation(
        coarse=meshes[0],
        intermediate=meshes[1],
        fine=meshes[2],
        persistent_nodes=persistent_nodes,
        supports_equal_across_meshes=supports_equal,
        coarse_to_intermediate_strict_refinement=coarse_to_intermediate,
        intermediate_to_fine_strict_refinement=intermediate_to_fine,
        coarse_intermediate_epi_errors=coarse_intermediate_errors,
        intermediate_fine_epi_errors=intermediate_fine_errors,
        coarse_fine_epi_errors=coarse_fine_errors,
        zhir_xi=xi,
        zhir_observations=zhir,
        zhir_abstention_reason=zhir_abstention,
        modal_observations=modal,
        conditions=conditions,
    )
    return _seal(
        value,
        EventRemeshThreeMeshRefinementObservation,
        _THREE_MESH_PROOF_VERSION,
    )
