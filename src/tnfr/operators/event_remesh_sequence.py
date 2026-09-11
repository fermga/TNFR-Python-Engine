"""Exact comparisons among supplied event/REMESH cycle observations.

The caller supplies independently sealed results in a local order. This module
checks recorded nodal channels, exact schedule clocks and REMESH history at
adjacent positions in that order. It does not establish shared graph identity,
causal execution order or provenance. Represented schedule compositions and
REMESH results stay separate because evolving history invalidates products of
their one-step gain bounds.
"""

from __future__ import annotations

from collections.abc import Hashable, Iterable
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any

from ..physics._exact_metric import (
    finite_binary64_fraction_vector_or_none,
    normalized_positive_fraction_metric,
)
from .event_remesh_runtime import EventRemeshCycleResult, _proof_value
from .event_runtime import ObservedRepresentedEPIScheduleComposition
from .remesh import DelayedRemeshResult

__all__ = [
    "EventRemeshCycleBoundaryObservation",
    "ObservedEventRemeshCycleSequence",
    "compose_event_remesh_cycle_observations",
]

ExactVector = tuple[Fraction, ...]
ExactHistory = tuple[ExactVector, ...]

_BOUNDARY_PROOF_VERSION = "event_remesh_cycle_boundary_v1"
_BOUNDARY_SCOPE = (
    "One exact comparison between adjacent positions in a caller-supplied "
    "local ordering of sealed cycle observations. Equality of recorded "
    "channels does not prove shared graph identity, causal succession, full "
    "graph-state continuity or cross-call atomicity."
)
_SEQUENCE_PROOF_VERSION = "event_remesh_cycle_sequence_v1"
_SCOPE = (
    "Exact comparisons among at least two independently sealed cycle "
    "observations in the caller-supplied local order. Recorded nodal channels, "
    "schedule clocks and REMESH history are compared at adjacent positions. "
    "Positive diagonal metrics are compared as exact normalized rays and raw "
    "materialized tuples. Nested represented schedule compositions and one-step "
    "REMESH results remain separate. Configuration equality is diagnostic only. "
    "Edges, hybrid_event_log and epi_time_history are not captured at both "
    "boundaries; shared graph identity and causal execution provenance are not "
    "recorded. A completed pressure callback does not identify a pure-EPI "
    "pressure law. No field composes gains or certifies full graph-state or "
    "grammar-history continuity, runtime-global gain, evolving-history REMESH "
    "gain, repetition, future stability or atomic rollback across calls."
)


def _exact_vector(values: Any, label: str) -> ExactVector:
    exact = finite_binary64_fraction_vector_or_none(values)
    if exact is None:
        raise ValueError(f"{label} must contain finite binary64 values")
    return exact


def _exact_history_is_well_typed(value: Any) -> bool:
    return bool(
        type(value) is tuple
        and all(
            type(row) is tuple
            and all(type(item) is Fraction for item in row)
            for row in value
        )
    )


def _cycle_proof_is_intact(cycle: EventRemeshCycleResult) -> bool:
    checker = getattr(cycle, "_proof_fields_are_intact", None)
    if not callable(checker):
        return False
    try:
        return checker() is True
    except Exception:
        return False


def _structural_equal(left: Any, right: Any) -> bool:
    """Compare proof serializations without invoking evidence equality."""

    try:
        return _proof_value(left) == _proof_value(right)
    except Exception:
        return False


def _hashable_items_match(left: Hashable, right: Hashable) -> bool:
    """Match identifiers by identity or a safe Hashable equality contract."""

    if left is right:
        return True
    try:
        equal = left == right
        hashes_match = hash(left) == hash(right)
        equality_holds = bool(equal)
    except Exception:
        return False
    return bool(
        equality_holds
        and hashes_match
        and _structural_equal(left, right)
    )


def _ordered_support_is_continuous(
    left: tuple[Hashable, ...],
    right: tuple[Hashable, ...],
) -> bool:
    """Compare ordered support without trusting structural shape alone."""

    return bool(
        len(left) == len(right)
        and all(
            _hashable_items_match(left_node, right_node)
            for left_node, right_node in zip(left, right, strict=True)
        )
        and _structural_equal(left, right)
    )


def _validate_conditions(
    value: Any,
    expected_names: tuple[str, ...],
    *,
    label: str,
) -> None:
    """Require exact, ordered pairs of condition names and booleans."""

    if type(value) is not tuple or len(value) != len(expected_names):
        raise TypeError(f"{label} conditions must be an exact tuple")
    for index, (item, expected_name) in enumerate(
        zip(value, expected_names, strict=True)
    ):
        if (
            type(item) is not tuple
            or len(item) != 2
            or type(item[0]) is not str
            or type(item[1]) is not bool
        ):
            raise TypeError(
                f"{label} conditions[{index}] must be tuple[str, bool]"
            )
        if item[0] != expected_name:
            raise ValueError(
                f"{label} condition names must retain canonical order"
            )


def _applicable_proof_is_intact(value: Any) -> bool:
    """Validate a nested seal when its exact type publishes one."""

    checker = getattr(value, "_proof_fields_are_intact", None)
    if not callable(checker):
        return True
    try:
        return checker() is True
    except Exception:
        return False


_BOUNDARY_FIELD_NAMES = (
    "boundary_index",
    "left_cycle_index",
    "right_cycle_index",
    "left_nodes",
    "right_nodes",
    "exact_left_schedule_end_time",
    "exact_right_schedule_start_time",
    "exact_left_pre_remesh_epi",
    "exact_left_post_remesh_epi",
    "exact_right_pre_schedule_epi",
    "left_outgoing_exact_history",
    "right_incoming_exact_history",
    "exact_left_capacity_after_remesh",
    "exact_right_capacity_before_schedule",
    "exact_left_pressure_after_optional_refresh",
    "exact_right_pressure_before_schedule",
    "exact_left_phase_after_optional_refresh",
    "exact_right_phase_before_schedule",
    "left_cycle_proof_intact",
    "right_cycle_proof_intact",
    "left_cycle_graph_state_atomic",
    "right_cycle_graph_state_atomic",
    "left_remesh_applied",
    "left_post_remesh_pressure_refresh_requested",
    "left_post_remesh_pressure_refresh_callback_invocations",
)


def _boundary_conditions(**fields: Any) -> tuple[tuple[str, bool], ...]:
    return (
        (
            "consecutive_cycle_indices",
            fields["boundary_index"] == fields["left_cycle_index"]
            and fields["right_cycle_index"] == fields["left_cycle_index"] + 1,
        ),
        ("left_cycle_proof_fields_intact", fields["left_cycle_proof_intact"]),
        ("right_cycle_proof_fields_intact", fields["right_cycle_proof_intact"]),
        (
            "left_cycle_graph_state_atomic",
            fields["left_cycle_graph_state_atomic"],
        ),
        (
            "right_cycle_graph_state_atomic",
            fields["right_cycle_graph_state_atomic"],
        ),
        (
            "ordered_node_support_continuous",
            _ordered_support_is_continuous(
                fields["left_nodes"],
                fields["right_nodes"],
            ),
        ),
        (
            "exact_schedule_time_continuous",
            _structural_equal(
                fields["exact_left_schedule_end_time"],
                fields["exact_right_schedule_start_time"],
            ),
        ),
        (
            "exact_post_remesh_to_pre_schedule_epi_continuous",
            _structural_equal(
                fields["exact_left_post_remesh_epi"],
                fields["exact_right_pre_schedule_epi"],
            ),
        ),
        (
            "full_exact_remesh_history_continuous",
            _structural_equal(
                fields["left_outgoing_exact_history"],
                fields["right_incoming_exact_history"],
            ),
        ),
        (
            "exact_capacity_continuous",
            _structural_equal(
                fields["exact_left_capacity_after_remesh"],
                fields["exact_right_capacity_before_schedule"],
            ),
        ),
        (
            "changed_applied_remesh_has_completed_pressure_refresh",
            not (
                fields["left_remesh_applied"]
                and not _structural_equal(
                    fields["exact_left_pre_remesh_epi"],
                    fields["exact_left_post_remesh_epi"],
                )
            )
            or (
                fields["left_post_remesh_pressure_refresh_requested"]
                and fields[
                    "left_post_remesh_pressure_refresh_callback_invocations"
                ]
                == 1
            ),
        ),
        (
            "exact_post_refresh_pressure_continuous",
            _structural_equal(
                fields["exact_left_pressure_after_optional_refresh"],
                fields["exact_right_pressure_before_schedule"],
            ),
        ),
        (
            "exact_phase_continuous",
            _structural_equal(
                fields["exact_left_phase_after_optional_refresh"],
                fields["exact_right_phase_before_schedule"],
            ),
        ),
    )


def _boundary_stamp(
    fields: dict[str, Any],
    conditions: tuple[tuple[str, bool], ...],
    scope: str,
) -> tuple[Any, ...]:
    return (
        _BOUNDARY_PROOF_VERSION,
        tuple(
            (name, _proof_value(fields[name]))
            for name in _BOUNDARY_FIELD_NAMES
        ),
        _proof_value(conditions),
        _proof_value(scope),
    )


@dataclass(frozen=True, slots=True)
class EventRemeshCycleBoundaryObservation:
    """Sealed comparison of recorded state at one adjacent call boundary."""

    boundary_index: int
    left_cycle_index: int
    right_cycle_index: int
    left_nodes: tuple[Hashable, ...]
    right_nodes: tuple[Hashable, ...]
    exact_left_schedule_end_time: Fraction
    exact_right_schedule_start_time: Fraction
    exact_left_pre_remesh_epi: ExactVector
    exact_left_post_remesh_epi: ExactVector
    exact_right_pre_schedule_epi: ExactVector
    left_outgoing_exact_history: ExactHistory
    right_incoming_exact_history: ExactHistory
    exact_left_capacity_after_remesh: ExactVector
    exact_right_capacity_before_schedule: ExactVector
    exact_left_pressure_after_optional_refresh: ExactVector
    exact_right_pressure_before_schedule: ExactVector
    exact_left_phase_after_optional_refresh: ExactVector
    exact_right_phase_before_schedule: ExactVector
    left_cycle_proof_intact: bool
    right_cycle_proof_intact: bool
    left_cycle_graph_state_atomic: bool
    right_cycle_graph_state_atomic: bool
    left_remesh_applied: bool
    left_post_remesh_pressure_refresh_requested: bool
    left_post_remesh_pressure_refresh_callback_invocations: int
    conditions: tuple[tuple[str, bool], ...]
    _proof_stamp: tuple[Any, ...] = field(repr=False, compare=False)
    scope: str = field(default=_BOUNDARY_SCOPE, init=False)

    def __post_init__(self) -> None:
        indices = (
            self.boundary_index,
            self.left_cycle_index,
            self.right_cycle_index,
        )
        if any(type(index) is not int or index < 0 for index in indices):
            raise ValueError("cycle and boundary indices must be nonnegative integers")
        if type(self.left_nodes) is not tuple or type(self.right_nodes) is not tuple:
            raise TypeError("boundary node supports must be tuples")
        if (
            type(self.exact_left_schedule_end_time) is not Fraction
            or type(self.exact_right_schedule_start_time) is not Fraction
        ):
            raise TypeError("schedule boundary times must be exact Fractions")

        vectors = (
            (self.exact_left_pre_remesh_epi, self.left_nodes),
            (self.exact_left_post_remesh_epi, self.left_nodes),
            (self.exact_right_pre_schedule_epi, self.right_nodes),
            (self.exact_left_capacity_after_remesh, self.left_nodes),
            (self.exact_right_capacity_before_schedule, self.right_nodes),
            (self.exact_left_pressure_after_optional_refresh, self.left_nodes),
            (self.exact_right_pressure_before_schedule, self.right_nodes),
            (self.exact_left_phase_after_optional_refresh, self.left_nodes),
            (self.exact_right_phase_before_schedule, self.right_nodes),
        )
        if any(
            type(vector) is not tuple
            or len(vector) != len(nodes)
            or any(type(item) is not Fraction for item in vector)
            for vector, nodes in vectors
        ):
            raise ValueError("exact boundary vectors must align with node support")
        histories = (
            (self.left_outgoing_exact_history, self.left_nodes),
            (self.right_incoming_exact_history, self.right_nodes),
        )
        if any(
            not _exact_history_is_well_typed(history)
            or any(len(row) != len(nodes) for row in history)
            for history, nodes in histories
        ):
            raise ValueError("exact history rows must align with node support")
        flags = (
            self.left_cycle_proof_intact,
            self.right_cycle_proof_intact,
            self.left_cycle_graph_state_atomic,
            self.right_cycle_graph_state_atomic,
            self.left_remesh_applied,
            self.left_post_remesh_pressure_refresh_requested,
        )
        if any(type(value) is not bool for value in flags):
            raise TypeError("boundary proof and atomicity flags must be bools")
        if (
            type(self.left_post_remesh_pressure_refresh_callback_invocations)
            is not int
            or self.left_post_remesh_pressure_refresh_callback_invocations < 0
        ):
            raise ValueError("pressure refresh invocation count must be nonnegative")

        fields = {name: getattr(self, name) for name in _BOUNDARY_FIELD_NAMES}
        expected_conditions = _boundary_conditions(**fields)
        expected_names = tuple(name for name, _ in expected_conditions)
        _validate_conditions(self.conditions, expected_names, label="boundary")
        if not _structural_equal(self.conditions, expected_conditions):
            raise ValueError("boundary conditions do not match endpoint evidence")
        if type(self.scope) is not str or self.scope != _BOUNDARY_SCOPE:
            raise ValueError("boundary scope must retain its canonical value")
        if (
            type(self._proof_stamp) is not tuple
            or not _structural_equal(
                self._proof_stamp,
                _boundary_stamp(fields, self.conditions, self.scope),
            )
        ):
            raise ValueError("boundary proof fields are inconsistent")

    def _proof_fields_are_intact(self) -> bool:
        try:
            self.__post_init__()
        except Exception:
            return False
        return True

    @property
    def exact_recorded_state_continuity_certified(self) -> bool:
        """Whether all recorded adjacent boundary conditions are sealed and true."""

        return bool(
            self._proof_fields_are_intact()
            and all(passed for _, passed in self.conditions)
        )

    @property
    def failed_conditions(self) -> tuple[str, ...]:
        if not self._proof_fields_are_intact():
            return ("boundary_proof_fields_intact",)
        return tuple(name for name, passed in self.conditions if not passed)


def _build_boundary(
    index: int,
    left: EventRemeshCycleResult,
    right: EventRemeshCycleResult,
) -> EventRemeshCycleBoundaryObservation:
    left_transition = left.history_transition
    right_transition = right.history_transition
    fields: dict[str, Any] = {
        "boundary_index": index,
        "left_cycle_index": index,
        "right_cycle_index": index + 1,
        "left_nodes": left.target_nodes,
        "right_nodes": right.target_nodes,
        "exact_left_schedule_end_time": (
            left.event_execution.schedule.exact_end_time
        ),
        "exact_right_schedule_start_time": (
            right.event_execution.schedule.exact_start_time
        ),
        "exact_left_pre_remesh_epi": _exact_vector(
            left.pre_remesh_epi.epi_values,
            f"cycles[{index}].pre_remesh_epi",
        ),
        "exact_left_post_remesh_epi": _exact_vector(
            left.post_remesh_epi.epi_values,
            f"cycles[{index}].post_remesh_epi",
        ),
        "exact_right_pre_schedule_epi": _exact_vector(
            right.pre_schedule_epi.epi_values,
            f"cycles[{index + 1}].pre_schedule_epi",
        ),
        "left_outgoing_exact_history": left_transition.outgoing_exact_history,
        "right_incoming_exact_history": right_transition.incoming_exact_history,
        "exact_left_capacity_after_remesh": _exact_vector(
            left.capacity_after_remesh,
            f"cycles[{index}].capacity_after_remesh",
        ),
        "exact_right_capacity_before_schedule": _exact_vector(
            right.capacity_before_schedule,
            f"cycles[{index + 1}].capacity_before_schedule",
        ),
        "exact_left_pressure_after_optional_refresh": _exact_vector(
            left.pressure_after_optional_refresh,
            f"cycles[{index}].pressure_after_optional_refresh",
        ),
        "exact_right_pressure_before_schedule": _exact_vector(
            right.pressure_before_schedule,
            f"cycles[{index + 1}].pressure_before_schedule",
        ),
        "exact_left_phase_after_optional_refresh": _exact_vector(
            left.phase_after_optional_refresh,
            f"cycles[{index}].phase_after_optional_refresh",
        ),
        "exact_right_phase_before_schedule": _exact_vector(
            right.phase_before_schedule,
            f"cycles[{index + 1}].phase_before_schedule",
        ),
        "left_cycle_proof_intact": _cycle_proof_is_intact(left),
        "right_cycle_proof_intact": _cycle_proof_is_intact(right),
        "left_cycle_graph_state_atomic": (
            type(left.whole_cycle_graph_state_atomic) is bool
            and left.whole_cycle_graph_state_atomic
        ),
        "right_cycle_graph_state_atomic": (
            type(right.whole_cycle_graph_state_atomic) is bool
            and right.whole_cycle_graph_state_atomic
        ),
        "left_remesh_applied": left.remesh.applied,
        "left_post_remesh_pressure_refresh_requested": (
            left.post_remesh_pressure_refresh_requested
        ),
        "left_post_remesh_pressure_refresh_callback_invocations": (
            left.post_remesh_pressure_refresh_callback_invocations
        ),
    }
    conditions = _boundary_conditions(**fields)
    return EventRemeshCycleBoundaryObservation(
        **fields,
        conditions=conditions,
        _proof_stamp=_boundary_stamp(fields, conditions, _BOUNDARY_SCOPE),
    )


def _cycle_metric_binding(cycle: EventRemeshCycleResult) -> bool:
    raw = _exact_vector(cycle.metric_weights, "cycle metric")
    observations = (
        cycle.pre_schedule_epi,
        cycle.pre_remesh_epi,
        cycle.post_remesh_epi,
    )
    if any(
        not _structural_equal(observation.nodes, cycle.target_nodes)
        or not _structural_equal(
            _exact_vector(observation.metric_weights, "observation metric"),
            raw,
        )
        for observation in observations
    ):
        return False
    evidence = cycle.remesh.evidence
    if cycle.remesh.applied and evidence is None:
        return False
    return bool(
        evidence is None
        or _structural_equal(
            _exact_vector(evidence.metric_weights, "REMESH evidence metric"),
            raw,
        )
    )


def _nested_schedule_metric_alignment(
    cycle: EventRemeshCycleResult,
    ray: ExactVector,
) -> bool | None:
    composition = cycle.event_execution.represented_epi_schedule_composition
    if composition is None:
        return None
    if (
        type(composition) is not ObservedRepresentedEPIScheduleComposition
        or not composition._proof_fields_are_intact()
    ):
        return False
    if not _structural_equal(composition.nodes, cycle.target_nodes):
        return False
    metric = composition.exact_normalized_metric
    if metric is None:
        return None
    return _structural_equal(metric, ray)


def _configuration_signature(cycle: EventRemeshCycleResult) -> tuple[Any, ...]:
    transition = cycle.history_transition
    plan = cycle.remesh.plan
    return (
        transition.tau_local,
        transition.tau_global,
        transition.history_maxlen,
        Fraction.from_float(plan.alpha),
        plan.alpha_source,
        Fraction.from_float(plan.epi_min),
        Fraction.from_float(plan.epi_max),
        plan.clip_mode,
    )


def _sequence_conditions(
    *,
    cycle_indices: tuple[int, ...],
    cycles: tuple[EventRemeshCycleResult, ...],
    boundaries: tuple[EventRemeshCycleBoundaryObservation, ...],
    per_cycle_metric_bound: tuple[bool, ...],
    cycle_exact_normalized_metric_rays: tuple[ExactVector, ...],
    nested_schedule_metric_alignment: tuple[bool | None, ...],
) -> tuple[tuple[str, bool], ...]:
    return (
        (
            "complete_indexed_cycle_range",
            _structural_equal(cycle_indices, tuple(range(len(cycles)))),
        ),
        (
            "all_cycle_proof_fields_intact",
            all(_cycle_proof_is_intact(cycle) for cycle in cycles),
        ),
        (
            "all_cycles_individually_graph_state_atomic",
            all(
                type(cycle.whole_cycle_graph_state_atomic) is bool
                and cycle.whole_cycle_graph_state_atomic
                for cycle in cycles
            ),
        ),
        (
            "complete_boundary_cardinality",
            len(boundaries) == len(cycles) - 1
            and _structural_equal(
                tuple(boundary.boundary_index for boundary in boundaries),
                tuple(range(len(boundaries))),
            ),
        ),
        (
            "all_exact_recorded_adjacent_states_continuous",
            bool(boundaries)
            and all(
                boundary.exact_recorded_state_continuity_certified
                for boundary in boundaries
            ),
        ),
        (
            "every_cycle_metric_bound",
            len(per_cycle_metric_bound) == len(cycles)
            and all(per_cycle_metric_bound),
        ),
        (
            "one_exact_normalized_metric_ray",
            len(cycle_exact_normalized_metric_rays) == len(cycles)
            and bool(cycle_exact_normalized_metric_rays)
            and all(
                _structural_equal(
                    ray,
                    cycle_exact_normalized_metric_rays[0],
                )
                for ray in cycle_exact_normalized_metric_rays[1:]
            ),
        ),
        (
            "every_nested_schedule_exposes_the_common_metric",
            len(nested_schedule_metric_alignment) == len(cycles)
            and all(
                alignment is True
                for alignment in nested_schedule_metric_alignment
            ),
        ),
    )


_SEQUENCE_FIELD_NAMES = (
    "cycle_indices",
    "cycles",
    "boundaries",
    "schedule_compositions",
    "remesh_results",
    "cycle_exact_metric_weights",
    "cycle_exact_normalized_metric_rays",
    "exact_common_normalized_metric_ray",
    "per_cycle_proof_fields_intact",
    "per_cycle_graph_state_atomic",
    "per_cycle_metric_bound",
    "raw_metric_weights_equal",
    "nested_schedule_metric_alignment",
    "remesh_configurations_equal",
    "conditions",
    "scope",
)


def _sequence_stamp(fields: dict[str, Any]) -> tuple[Any, ...]:
    return (
        _SEQUENCE_PROOF_VERSION,
        tuple(
            (name, _proof_value(fields[name]))
            for name in _SEQUENCE_FIELD_NAMES
        ),
    )


@dataclass(frozen=True, slots=True)
class ObservedEventRemeshCycleSequence:
    """Sealed comparisons for supplied cycle observations in local order."""

    cycle_indices: tuple[int, ...]
    cycles: tuple[EventRemeshCycleResult, ...]
    boundaries: tuple[EventRemeshCycleBoundaryObservation, ...]
    schedule_compositions: tuple[
        ObservedRepresentedEPIScheduleComposition | None, ...
    ]
    remesh_results: tuple[DelayedRemeshResult, ...]
    cycle_exact_metric_weights: tuple[ExactVector, ...]
    cycle_exact_normalized_metric_rays: tuple[ExactVector, ...]
    exact_common_normalized_metric_ray: ExactVector | None
    per_cycle_proof_fields_intact: tuple[bool, ...]
    per_cycle_graph_state_atomic: tuple[bool, ...]
    per_cycle_metric_bound: tuple[bool, ...]
    raw_metric_weights_equal: bool
    nested_schedule_metric_alignment: tuple[bool | None, ...]
    remesh_configurations_equal: bool
    conditions: tuple[tuple[str, bool], ...]
    _proof_stamp: tuple[Any, ...] = field(repr=False, compare=False)
    scope: str = field(default=_SCOPE, init=False)

    def __post_init__(self) -> None:
        if type(self.cycles) is not tuple or len(self.cycles) < 2:
            raise ValueError("a cycle sequence requires at least two cycles")
        if any(type(cycle) is not EventRemeshCycleResult for cycle in self.cycles):
            raise TypeError(
                "cycles must contain exact EventRemeshCycleResult objects"
            )
        if len({id(cycle) for cycle in self.cycles}) != len(self.cycles):
            raise ValueError("cycle observations must have distinct identities")
        if any(not _cycle_proof_is_intact(cycle) for cycle in self.cycles):
            raise ValueError("cycle proof fields are not intact")
        if (
            type(self.cycle_indices) is not tuple
            or any(type(index) is not int for index in self.cycle_indices)
            or not _structural_equal(
                self.cycle_indices,
                tuple(range(len(self.cycles))),
            )
        ):
            raise ValueError("cycle_indices must be the complete zero-based range")

        expected_boundaries = tuple(
            _build_boundary(index, left, right)
            for index, (left, right) in enumerate(
                zip(self.cycles, self.cycles[1:])
            )
        )
        if (
            type(self.boundaries) is not tuple
            or len(self.boundaries) != len(expected_boundaries)
            or any(
                type(boundary) is not EventRemeshCycleBoundaryObservation
                or not boundary._proof_fields_are_intact()
                or not _structural_equal(
                    boundary._proof_stamp,
                    expected._proof_stamp,
                )
                for boundary, expected in zip(
                    self.boundaries, expected_boundaries, strict=True
                )
            )
        ):
            raise ValueError("sequence boundaries do not match adjacent cycles")
        expected_schedules = tuple(
            cycle.event_execution.represented_epi_schedule_composition
            for cycle in self.cycles
        )
        expected_remesh = tuple(cycle.remesh for cycle in self.cycles)
        if (
            type(self.schedule_compositions) is not tuple
            or len(self.schedule_compositions) != len(expected_schedules)
            or any(
                observed is not expected
                for observed, expected in zip(
                    self.schedule_compositions,
                    expected_schedules,
                    strict=True,
                )
            )
        ):
            raise ValueError(
                "nested schedule compositions must retain cycle-owned identity"
            )
        if any(
            composition is not None
            and (
                type(composition)
                is not ObservedRepresentedEPIScheduleComposition
                or not _applicable_proof_is_intact(composition)
            )
            for composition in self.schedule_compositions
        ):
            raise ValueError("nested schedule composition proofs are not intact")
        if (
            type(self.remesh_results) is not tuple
            or len(self.remesh_results) != len(expected_remesh)
            or any(
                observed is not expected
                for observed, expected in zip(
                    self.remesh_results,
                    expected_remesh,
                    strict=True,
                )
            )
        ):
            raise ValueError("nested REMESH results must retain cycle-owned identity")
        if any(
            type(result) is not DelayedRemeshResult
            or not _applicable_proof_is_intact(result)
            for result in self.remesh_results
        ):
            raise ValueError("nested REMESH result proofs are not intact")

        raw_metrics = tuple(
            _exact_vector(cycle.metric_weights, "cycle metric")
            for cycle in self.cycles
        )
        rays_optional = tuple(
            normalized_positive_fraction_metric(metric)
            for metric in raw_metrics
        )
        if any(ray is None for ray in rays_optional):
            raise ValueError("cycle metrics must define positive exact rays")
        rays = tuple(ray for ray in rays_optional if ray is not None)
        proof_states = tuple(
            _cycle_proof_is_intact(cycle) for cycle in self.cycles
        )
        atomic_states = tuple(
            type(cycle.whole_cycle_graph_state_atomic) is bool
            and cycle.whole_cycle_graph_state_atomic
            for cycle in self.cycles
        )
        bindings = tuple(_cycle_metric_binding(cycle) for cycle in self.cycles)
        raw_equal = all(
            _structural_equal(metric, raw_metrics[0])
            for metric in raw_metrics[1:]
        )
        ray_equal = all(_structural_equal(ray, rays[0]) for ray in rays[1:])
        common_ray = rays[0] if ray_equal else None
        schedule_alignment = tuple(
            _nested_schedule_metric_alignment(cycle, ray)
            for cycle, ray in zip(self.cycles, rays, strict=True)
        )
        configurations = tuple(
            _configuration_signature(cycle) for cycle in self.cycles
        )
        config_equal = all(
            _structural_equal(signature, configurations[0])
            for signature in configurations[1:]
        )
        expected_conditions = _sequence_conditions(
            cycle_indices=self.cycle_indices,
            cycles=self.cycles,
            boundaries=self.boundaries,
            per_cycle_metric_bound=bindings,
            cycle_exact_normalized_metric_rays=rays,
            nested_schedule_metric_alignment=schedule_alignment,
        )
        expected = {
            "cycle_exact_metric_weights": raw_metrics,
            "cycle_exact_normalized_metric_rays": rays,
            "exact_common_normalized_metric_ray": common_ray,
            "per_cycle_proof_fields_intact": proof_states,
            "per_cycle_graph_state_atomic": atomic_states,
            "per_cycle_metric_bound": bindings,
            "raw_metric_weights_equal": raw_equal,
            "nested_schedule_metric_alignment": schedule_alignment,
            "remesh_configurations_equal": config_equal,
            "conditions": expected_conditions,
        }
        expected_names = tuple(name for name, _ in expected_conditions)
        _validate_conditions(self.conditions, expected_names, label="sequence")
        if any(
            not _structural_equal(getattr(self, name), value)
            for name, value in expected.items()
        ):
            raise ValueError("sequence diagnostics do not match cycle evidence")
        if type(self.scope) is not str or self.scope != _SCOPE:
            raise ValueError("sequence scope must retain its canonical value")

        fields = {name: getattr(self, name) for name in _SEQUENCE_FIELD_NAMES}
        if (
            type(self._proof_stamp) is not tuple
            or not _structural_equal(
                self._proof_stamp,
                _sequence_stamp(fields),
            )
        ):
            raise ValueError("sequence proof fields are inconsistent")

    def _proof_fields_are_intact(self) -> bool:
        try:
            self.__post_init__()
        except Exception:
            return False
        return True

    @property
    def exact_recorded_boundary_continuity_certified(self) -> bool:
        """Whether the sealed recorded nodal/history boundaries are continuous."""

        return bool(
            self._proof_fields_are_intact()
            and all(self.per_cycle_proof_fields_intact)
            and all(self.per_cycle_graph_state_atomic)
            and all(
                boundary.exact_recorded_state_continuity_certified
                for boundary in self.boundaries
            )
        )

    @property
    def exact_common_metric_cycle_sequence_certified(self) -> bool:
        """Whether boundaries and all retained schedule metrics share one ray."""

        return bool(
            self._proof_fields_are_intact()
            and all(passed for _, passed in self.conditions)
        )

    @property
    def all_nested_schedule_metrics_aligned(self) -> bool | None:
        """Aggregate alignment only when every nested schedule exposes a metric."""

        if not self._proof_fields_are_intact():
            return False
        if any(
            value is False for value in self.nested_schedule_metric_alignment
        ):
            return False
        if any(
            value is None for value in self.nested_schedule_metric_alignment
        ):
            return None
        return True

    @property
    def failed_conditions(self) -> tuple[str, ...]:
        if not self._proof_fields_are_intact():
            return ("sequence_proof_fields_intact",)
        return tuple(name for name, passed in self.conditions if not passed)

    @property
    def mixed_schedule_remesh_gain_certified(self) -> bool:
        return False

    @property
    def evolving_history_remesh_gain_certified(self) -> bool:
        return False

    @property
    def evolving_history_remesh_repetition_certified(self) -> bool:
        return False

    @property
    def runtime_global_gain_certified(self) -> bool:
        return False

    @property
    def future_cycle_stability_certified(self) -> bool:
        return False

    @property
    def whole_sequence_atomicity_certified(self) -> bool:
        return False

    @property
    def full_graph_state_continuity_certified(self) -> bool:
        return False

    @property
    def grammar_history_continuity_certified(self) -> bool:
        return False

    @property
    def shared_graph_execution_provenance_certified(self) -> bool:
        return False


def compose_event_remesh_cycle_observations(
    cycles: Iterable[EventRemeshCycleResult],
) -> ObservedEventRemeshCycleSequence:
    """Bind at least two already executed cycle results in observed order."""

    try:
        materialized = tuple(cycles)
    except TypeError as exc:
        raise TypeError("cycles must be an iterable of cycle results") from exc
    if len(materialized) < 2:
        raise ValueError("at least two cycle results are required")
    if len({id(cycle) for cycle in materialized}) != len(materialized):
        raise ValueError("cycle observations must have distinct identities")
    for index, cycle in enumerate(materialized):
        if type(cycle) is not EventRemeshCycleResult:
            raise TypeError(
                f"cycles[{index}] must be an exact EventRemeshCycleResult"
            )
        if not _cycle_proof_is_intact(cycle):
            raise ValueError(f"cycles[{index}] proof fields are not intact")

    boundaries = tuple(
        _build_boundary(index, left, right)
        for index, (left, right) in enumerate(
            zip(materialized, materialized[1:])
        )
    )
    raw_metrics = tuple(
        _exact_vector(
            cycle.metric_weights,
            f"cycles[{index}].metric_weights",
        )
        for index, cycle in enumerate(materialized)
    )
    rays_optional = tuple(
        normalized_positive_fraction_metric(metric) for metric in raw_metrics
    )
    if any(ray is None for ray in rays_optional):
        raise ValueError("every cycle metric must define a positive exact ray")
    rays = tuple(ray for ray in rays_optional if ray is not None)
    ray_equal = all(_structural_equal(ray, rays[0]) for ray in rays[1:])
    proof_states = tuple(
        _cycle_proof_is_intact(cycle) for cycle in materialized
    )
    atomic_states = tuple(
        type(cycle.whole_cycle_graph_state_atomic) is bool
        and cycle.whole_cycle_graph_state_atomic
        for cycle in materialized
    )
    bindings = tuple(
        _cycle_metric_binding(cycle) for cycle in materialized
    )
    schedule_compositions = tuple(
        cycle.event_execution.represented_epi_schedule_composition
        for cycle in materialized
    )
    remesh_results = tuple(cycle.remesh for cycle in materialized)
    schedule_alignment = tuple(
        _nested_schedule_metric_alignment(cycle, ray)
        for cycle, ray in zip(materialized, rays, strict=True)
    )
    configurations = tuple(
        _configuration_signature(cycle) for cycle in materialized
    )
    fields: dict[str, Any] = {
        "cycle_indices": tuple(range(len(materialized))),
        "cycles": materialized,
        "boundaries": boundaries,
        "schedule_compositions": schedule_compositions,
        "remesh_results": remesh_results,
        "cycle_exact_metric_weights": raw_metrics,
        "cycle_exact_normalized_metric_rays": rays,
        "exact_common_normalized_metric_ray": (
            rays[0] if ray_equal else None
        ),
        "per_cycle_proof_fields_intact": proof_states,
        "per_cycle_graph_state_atomic": atomic_states,
        "per_cycle_metric_bound": bindings,
        "raw_metric_weights_equal": all(
            _structural_equal(metric, raw_metrics[0])
            for metric in raw_metrics[1:]
        ),
        "nested_schedule_metric_alignment": schedule_alignment,
        "remesh_configurations_equal": all(
            _structural_equal(signature, configurations[0])
            for signature in configurations[1:]
        ),
    }
    fields["conditions"] = _sequence_conditions(
        cycle_indices=fields["cycle_indices"],
        cycles=materialized,
        boundaries=boundaries,
        per_cycle_metric_bound=bindings,
        cycle_exact_normalized_metric_rays=rays,
        nested_schedule_metric_alignment=schedule_alignment,
    )
    stamp_fields = {**fields, "scope": _SCOPE}
    return ObservedEventRemeshCycleSequence(
        **fields,
        _proof_stamp=_sequence_stamp(stamp_fields),
    )
