r"""Finite runtime binding for adjacent schedule/REMESH cycle observations.

One cycle appends its pre-REMESH head and then applies REMESH.  The next cycle
starts from that bounded REMESH result, applies its represented event schedule,
and appends the scheduled endpoint.  This module binds those two records to the
exact companion-history and REMESH-head/schedule-head balances.

The input sequence is still an offline ordering of independently sealed cycle
results.  Exact recorded boundaries and individually atomic cycles do not prove
shared graph identity, causal execution order or one atomic multi-cycle run.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from fractions import Fraction
from typing import Any

from ..errors import TNFRValueError
from ..operators.event_remesh_runtime import _proof_value
from ..operators.event_remesh_sequence import ObservedEventRemeshCycleSequence
from ..operators.event_runtime import ObservedRepresentedEPIScheduleComposition
from ..utils._structural_signature import proof_stamps_are_identical
from ._exact_metric import finite_binary64_fraction_vector_or_none
from .remesh_history_stability import (
    ExactHistory,
    ExactVector,
    UniformRemeshHistoryTransitionObservation,
    observe_uniform_remesh_history_transition,
)
from .remesh_schedule_stability import (
    RemeshScheduleHistoryStabilityObservation,
    observe_remesh_schedule_history_transition,
)
from .runtime_remesh_history_stability import (
    RuntimeRemeshHistoryBridgeObservation,
    observe_runtime_remesh_history_bridge,
)

__all__ = (
    "RuntimeRemeshScheduleBoundaryObservation",
    "RuntimeRemeshScheduleSequenceObservation",
    "observe_runtime_remesh_schedule_sequence",
)

_BOUNDARY_PROOF_VERSION = "runtime_remesh_schedule_boundary_v1"
_SEQUENCE_PROOF_VERSION = "runtime_remesh_schedule_sequence_v1"
_BOUNDARY_SCOPE = (
    "One exact finite binding from an applied REMESH result at supplied cycle "
    "i through the represented EPI schedule and history append at supplied "
    "cycle i+1. The common normalized metric, fixed REMESH configuration, "
    "binary64 runtime bridge, schedule endpoints, gain, exact history shift "
    "and augmented-energy balance are identified. Shared graph provenance, "
    "causal succession, a global executable gain, repetition, future behavior "
    "and full TNFR stability are not certified."
)
_SEQUENCE_SCOPE = (
    "A finite telescope of adjacent runtime REMESH/schedule boundary balances "
    "from one intact caller-ordered cycle sequence. Every row uses one common "
    "normalized positive metric and one fixed REMESH configuration. The result "
    "binds recorded binary64 heads and represented schedule gains, but it does "
    "not establish shared graph provenance, causal execution order, cross-call "
    "atomicity, a global executable map, repeated or future stability, solver "
    "accuracy, changing support or full multichannel TNFR stability."
)

_BOUNDARY_CONDITION_NAMES = (
    "source_sequence_intact",
    "source_common_metric_sequence_certified",
    "remesh_configuration_fixed",
    "source_boundary_exactly_continuous",
    "left_remesh_applied",
    "left_runtime_bridge_intact",
    "normalized_companion_matches_runtime_bridge",
    "right_schedule_composition_intact",
    "right_schedule_represented_gain_certified",
    "right_schedule_uses_common_metric",
    "schedule_input_matches_bounded_remesh_head",
    "schedule_endpoint_matches_right_pre_remesh_head",
    "scheduled_history_matches_next_cycle_history",
    "exact_schedule_balance_intact",
)

_SEQUENCE_CONDITION_NAMES = (
    "source_sequence_intact",
    "complete_adjacent_boundary_cardinality",
    "every_adjacent_boundary_intact",
    "exact_intermediate_augmented_energies_continuous",
    "exact_finite_energy_drop_telescope",
    "summed_gain_lower_bound_and_slack_identity",
)


def _strict_exact_vector(value: Any, width: int) -> bool:
    return bool(
        type(value) is tuple
        and len(value) == width
        and all(type(item) is Fraction for item in value)
    )


def _strict_exact_history(value: Any, width: int) -> bool:
    return bool(
        type(value) is tuple
        and value
        and all(_strict_exact_vector(row, width) for row in value)
    )


def _strict_conditions(value: Any, names: tuple[str, ...]) -> bool:
    return bool(
        type(value) is tuple
        and len(value) == len(names)
        and all(
            type(item) is tuple
            and len(item) == 2
            and type(item[0]) is str
            and type(item[1]) is bool
            and item[0] == names[index]
            for index, item in enumerate(value)
        )
    )


def _exact_binary64_vector(value: Any, label: str) -> ExactVector:
    exact = finite_binary64_fraction_vector_or_none(value)
    if exact is None:
        raise TNFRValueError(f"{label} must contain finite binary64 values")
    return exact


def _same(left: Any, right: Any) -> bool:
    """Compare proof-domain values without caller-owned equality."""

    try:
        return proof_stamps_are_identical(_proof_value(left), _proof_value(right))
    except BaseException:
        return False


def _raw_stamp_or_none(value: Any) -> tuple[Any, ...] | None:
    try:
        stamp = object.__getattribute__(value, "_proof_stamp")
    except BaseException:
        return None
    return stamp if type(stamp) is tuple else None


def _compact_proof_value(value: Any) -> Any:
    nested_types = (
        ObservedEventRemeshCycleSequence,
        RuntimeRemeshHistoryBridgeObservation,
        UniformRemeshHistoryTransitionObservation,
        ObservedRepresentedEPIScheduleComposition,
        RemeshScheduleHistoryStabilityObservation,
    )
    boundary_type = globals().get("RuntimeRemeshScheduleBoundaryObservation")
    if boundary_type is not None:
        nested_types = (*nested_types, boundary_type)
    if type(value) in nested_types:
        stamp = _raw_stamp_or_none(value)
        version = stamp[0] if stamp and type(stamp[0]) is str else None
        return (
            "tnfr-validated-proof-identity-v1",
            type(value).__module__,
            type(value).__qualname__,
            id(value),
            version,
        )
    if type(value) is tuple:
        return ("tuple", tuple(_compact_proof_value(item) for item in value))
    return _proof_value(value)


def _stamp_from_values(
    version: str,
    field_names: tuple[str, ...],
    values: dict[str, Any],
) -> tuple[Any, ...]:
    return (
        version,
        tuple(
            (name, _compact_proof_value(values[name])) for name in field_names
        ),
    )


def _object_values(value: Any, expected_type: type[Any]) -> dict[str, Any]:
    if type(value) is not expected_type:
        raise TypeError("proof value must have its canonical result type")
    return {
        item.name: object.__getattribute__(value, item.name)
        for item in fields(expected_type)
        if item.name != "_proof_stamp"
    }


def _source_sequence_is_intact(value: Any) -> bool:
    if type(value) is not ObservedEventRemeshCycleSequence:
        return False
    try:
        return (
            ObservedEventRemeshCycleSequence._proof_fields_are_intact(value)
            is True
        )
    except BaseException:
        return False


def _composition_is_intact(value: Any) -> bool:
    if type(value) is not ObservedRepresentedEPIScheduleComposition:
        return False
    try:
        return (
            ObservedRepresentedEPIScheduleComposition._proof_fields_are_intact(
                value
            )
            is True
        )
    except BaseException:
        return False


@dataclass(frozen=True, slots=True)
class RuntimeRemeshScheduleBoundaryObservation:
    """One sealed adjacent-cycle runtime/history energy balance."""

    source_sequence: ObservedEventRemeshCycleSequence = field(repr=False)
    boundary_index: int
    runtime_bridge: RuntimeRemeshHistoryBridgeObservation = field(repr=False)
    exact_transition: UniformRemeshHistoryTransitionObservation = field(
        repr=False
    )
    schedule_composition: ObservedRepresentedEPIScheduleComposition = field(
        repr=False
    )
    schedule_balance: RemeshScheduleHistoryStabilityObservation = field(
        repr=False
    )
    exact_common_normalized_metric: ExactVector
    exact_schedule_input_head: ExactVector
    exact_scheduled_head: ExactVector
    exact_scheduled_post_history: ExactHistory
    exact_next_cycle_history: ExactHistory
    exact_augmented_energy_before: Fraction
    exact_augmented_energy_after: Fraction
    exact_energy_drop: Fraction
    exact_gain_based_energy_drop_lower_bound: Fraction
    exact_schedule_augmented_energy_gain_slack: Fraction
    conditions: tuple[tuple[str, bool], ...]
    _proof_stamp: tuple[Any, ...] = field(default=(), repr=False, compare=False)

    def _proof_fields_are_intact(self) -> bool:
        return _boundary_proof_fields_are_intact(self)

    @property
    def scope(self) -> str:
        return _BOUNDARY_SCOPE

    @property
    def boundary_observation_certified(self) -> bool:
        return self._proof_fields_are_intact()

    @property
    def failed_conditions(self) -> tuple[str, ...]:
        if not self._proof_fields_are_intact():
            return ("runtime_remesh_schedule_boundary_proof_fields_intact",)
        return tuple(name for name, passed in self.conditions if not passed)

    @property
    def exact_recorded_history_advance_certified(self) -> bool:
        return self._proof_fields_are_intact()

    @property
    def exact_finite_schedule_remesh_balance_certified(self) -> bool:
        return self._proof_fields_are_intact()

    @property
    def energy_nonincrease_observed(self) -> bool:
        return bool(self._proof_fields_are_intact() and self.exact_energy_drop >= 0)

    @property
    def energy_nonincrease_sufficiently_certified(self) -> bool:
        return bool(
            self._proof_fields_are_intact()
            and self.exact_gain_based_energy_drop_lower_bound >= 0
        )

    @property
    def shared_graph_execution_provenance_certified(self) -> bool:
        return False

    @property
    def runtime_global_gain_certified(self) -> bool:
        return False

    @property
    def repeated_runtime_stability_certified(self) -> bool:
        return False

    @property
    def future_stability_certified(self) -> bool:
        return False


_BOUNDARY_FIELD_NAMES = tuple(
    item.name
    for item in fields(RuntimeRemeshScheduleBoundaryObservation)
    if item.name != "_proof_stamp"
)


def _boundary_proof_fields_are_intact(
    value: Any,
    *,
    source_already_validated: bool = False,
) -> bool:
    try:
        current = _object_values(
            value,
            RuntimeRemeshScheduleBoundaryObservation,
        )
        current_stamp = _stamp_from_values(
            _BOUNDARY_PROOF_VERSION,
            _BOUNDARY_FIELD_NAMES,
            current,
        )
        if not proof_stamps_are_identical(
            object.__getattribute__(value, "_proof_stamp"),
            current_stamp,
        ):
            return False
        source = object.__getattribute__(value, "source_sequence")
        index = object.__getattribute__(value, "boundary_index")
        if (
            type(source) is not ObservedEventRemeshCycleSequence
            or type(index) is not int
            or (
                not source_already_validated
                and not _source_sequence_is_intact(source)
            )
        ):
            return False
        expected = _derive_boundary_values(
            source,
            index,
            runtime_bridge_override=object.__getattribute__(
                value,
                "runtime_bridge",
            ),
            exact_transition_override=object.__getattribute__(
                value,
                "exact_transition",
            ),
            schedule_balance_override=object.__getattribute__(
                value,
                "schedule_balance",
            ),
            source_already_validated=True,
        )
        expected_stamp = _stamp_from_values(
            _BOUNDARY_PROOF_VERSION,
            _BOUNDARY_FIELD_NAMES,
            expected,
        )
        return proof_stamps_are_identical(
            object.__getattribute__(value, "_proof_stamp"),
            expected_stamp,
        )
    except BaseException:
        return False


def _derive_boundary_values(
    source: ObservedEventRemeshCycleSequence,
    index: int,
    *,
    runtime_bridge_override: RuntimeRemeshHistoryBridgeObservation | None = None,
    exact_transition_override: (
        UniformRemeshHistoryTransitionObservation | None
    ) = None,
    schedule_balance_override: (
        RemeshScheduleHistoryStabilityObservation | None
    ) = None,
    source_already_validated: bool = False,
) -> dict[str, Any]:
    if not source_already_validated and not _source_sequence_is_intact(source):
        raise TNFRValueError("cycle sequence is unsealed, tampered, or inconsistent")
    if not source.exact_common_metric_cycle_sequence_certified:
        raise TNFRValueError("cycle sequence lacks one exact common schedule metric")
    if source.remesh_configurations_equal is not True:
        raise TNFRValueError("adjacent cycles must use one REMESH configuration")
    if type(index) is not int or index < 0 or index >= len(source.boundaries):
        raise TNFRValueError("boundary_index is outside the cycle sequence")

    source_boundary = source.boundaries[index]
    if not source_boundary.exact_recorded_state_continuity_certified:
        raise TNFRValueError("source cycle boundary is not exactly continuous")
    common_metric = source.exact_common_normalized_metric_ray
    if common_metric is None:
        raise TNFRValueError("cycle sequence omits its exact common metric")
    width = len(common_metric)
    if (
        not _strict_exact_vector(common_metric, width)
        or not common_metric
        or any(weight <= 0 for weight in common_metric)
        or sum(common_metric, Fraction(0)) != 1
    ):
        raise TNFRValueError("cycle sequence common metric is invalid")

    left = source.cycles[index]
    right = source.cycles[index + 1]
    if not left.remesh.applied:
        raise TNFRValueError("left cycle must contain an applied REMESH map")
    if runtime_bridge_override is None:
        runtime_bridge = observe_runtime_remesh_history_bridge(left)
        runtime_bridge_intact = True
    else:
        runtime_bridge = runtime_bridge_override
        runtime_bridge_intact = bool(
            type(runtime_bridge) is RuntimeRemeshHistoryBridgeObservation
            and runtime_bridge.bridge_observation_certified
            and runtime_bridge.cycle_result is left
        )
        if not runtime_bridge_intact:
            raise TNFRValueError(
                "stored runtime bridge does not belong to the left cycle"
            )

    if exact_transition_override is None:
        exact_transition = observe_uniform_remesh_history_transition(
            runtime_bridge.exact_transition.certificate,
            runtime_bridge.exact_history,
            common_metric,
            nodes=runtime_bridge.nodes,
        )
    else:
        exact_transition = exact_transition_override
        if (
            type(exact_transition)
            is not UniformRemeshHistoryTransitionObservation
        ):
            raise TNFRValueError(
                "stored normalized companion has the wrong result type"
            )
        certificate = exact_transition.certificate
        runtime_certificate = runtime_bridge.exact_transition.certificate
        if (
            not exact_transition.transition_observation_certified
            or not _same(exact_transition.nodes, runtime_bridge.nodes)
            or not _same(exact_transition.exact_metric_weights, common_metric)
            or not _same(exact_transition.exact_history, runtime_bridge.exact_history)
            or type(certificate.alpha) is not Fraction
            or certificate.alpha != runtime_certificate.alpha
            or certificate.tau_local != runtime_certificate.tau_local
            or certificate.tau_global != runtime_certificate.tau_global
        ):
            raise TNFRValueError(
                "stored normalized companion does not match the left bridge"
            )
    normalized_companion_matches = bool(
        _same(
            exact_transition.exact_next_field,
            runtime_bridge.exact_ideal_next_field,
        )
        and _same(exact_transition.nodes, runtime_bridge.nodes)
    )
    if not normalized_companion_matches:
        raise TNFRValueError(
            "normalized companion does not match the runtime REMESH bridge"
        )

    composition = right.event_execution.represented_epi_schedule_composition
    composition_intact = _composition_is_intact(composition)
    if not composition_intact:
        raise TNFRValueError(
            "right cycle lacks an intact represented schedule composition"
        )
    assert composition is not None
    composition_gain_certified = bool(
        composition.represented_affine_composition_gain_certified
    )
    if not composition_gain_certified:
        raise TNFRValueError("right schedule lacks a represented affine gain")
    if not _same(composition.exact_normalized_metric, common_metric):
        raise TNFRValueError("right schedule does not use the common metric")
    gain = composition.exact_energy_gain_upper_bound
    if type(gain) is not Fraction or gain < 0:
        raise TNFRValueError("right schedule has no exact nonnegative gain")
    if not composition.operations:
        raise TNFRValueError("right schedule composition must be nonempty")

    schedule_input = composition.operations[0].exact_epi_before
    scheduled = composition.operations[-1].exact_epi_after
    if not _strict_exact_vector(schedule_input, width):
        raise TNFRValueError("right schedule input head is not exact and complete")
    if not _strict_exact_vector(scheduled, width):
        raise TNFRValueError("right schedule endpoint is not exact and complete")
    exact_right_pre_remesh = _exact_binary64_vector(
        right.pre_remesh_epi.epi_values,
        "right pre-REMESH EPI",
    )
    schedule_input_matches = _same(
        schedule_input,
        runtime_bridge.exact_runtime_bounded_next_field,
    )
    schedule_endpoint_matches = _same(scheduled, exact_right_pre_remesh)
    if not schedule_input_matches:
        raise TNFRValueError(
            "right schedule input does not match the left bounded REMESH head"
        )
    if not schedule_endpoint_matches:
        raise TNFRValueError(
            "right schedule endpoint does not match right pre-REMESH EPI"
        )

    if schedule_balance_override is None:
        balance = observe_remesh_schedule_history_transition(
            exact_transition,
            runtime_bridge.exact_runtime_raw_next_field,
            runtime_bridge.exact_runtime_bounded_next_field,
            scheduled,
            gain,
        )
        balance_intact = True
    else:
        balance = schedule_balance_override
        balance_intact = bool(
            type(balance) is RemeshScheduleHistoryStabilityObservation
            and balance.transition_observation_certified
            and balance.transition is exact_transition
            and _same(
                balance.exact_runtime_raw_head,
                runtime_bridge.exact_runtime_raw_next_field,
            )
            and _same(
                balance.exact_runtime_bounded_head,
                runtime_bridge.exact_runtime_bounded_next_field,
            )
            and _same(balance.exact_scheduled_head, scheduled)
            and balance.exact_schedule_energy_gain_upper_bound == gain
        )
        if not balance_intact:
            raise TNFRValueError(
                "stored schedule balance does not match the adjacent boundary"
            )
    scheduled_history = (scheduled,) + exact_transition.exact_history[:-1]
    required = exact_transition.certificate.active_max_delay + 1
    outgoing = right.history_transition.outgoing_exact_history
    if len(outgoing) < required:
        raise TNFRValueError(
            "right cycle omits the required post-schedule history window"
        )
    next_history = tuple(reversed(outgoing[-required:]))
    history_matches = _same(scheduled_history, next_history)
    if not history_matches:
        raise TNFRValueError(
            "scheduled head does not produce the next recorded companion history"
        )

    conditions = (
        ("source_sequence_intact", True),
        ("source_common_metric_sequence_certified", True),
        ("remesh_configuration_fixed", True),
        ("source_boundary_exactly_continuous", True),
        ("left_remesh_applied", True),
        ("left_runtime_bridge_intact", runtime_bridge_intact),
        (
            "normalized_companion_matches_runtime_bridge",
            normalized_companion_matches,
        ),
        ("right_schedule_composition_intact", composition_intact),
        (
            "right_schedule_represented_gain_certified",
            composition_gain_certified,
        ),
        (
            "right_schedule_uses_common_metric",
            _same(composition.exact_normalized_metric, common_metric),
        ),
        ("schedule_input_matches_bounded_remesh_head", schedule_input_matches),
        (
            "schedule_endpoint_matches_right_pre_remesh_head",
            schedule_endpoint_matches,
        ),
        ("scheduled_history_matches_next_cycle_history", history_matches),
        ("exact_schedule_balance_intact", balance_intact),
    )
    if not _strict_conditions(conditions, _BOUNDARY_CONDITION_NAMES) or not all(
        passed for _, passed in conditions
    ):
        failed = ", ".join(name for name, passed in conditions if not passed)
        raise RuntimeError(f"runtime REMESH/schedule binding failed: {failed}")

    return {
        "source_sequence": source,
        "boundary_index": index,
        "runtime_bridge": runtime_bridge,
        "exact_transition": exact_transition,
        "schedule_composition": composition,
        "schedule_balance": balance,
        "exact_common_normalized_metric": common_metric,
        "exact_schedule_input_head": schedule_input,
        "exact_scheduled_head": scheduled,
        "exact_scheduled_post_history": scheduled_history,
        "exact_next_cycle_history": next_history,
        "exact_augmented_energy_before": balance.exact_augmented_energy_before,
        "exact_augmented_energy_after": balance.exact_augmented_energy_after,
        "exact_energy_drop": balance.exact_energy_drop,
        "exact_gain_based_energy_drop_lower_bound": (
            balance.exact_gain_based_energy_drop_lower_bound
        ),
        "exact_schedule_augmented_energy_gain_slack": (
            balance.exact_schedule_augmented_energy_gain_slack
        ),
        "conditions": conditions,
    }


def _build_boundary(
    source: ObservedEventRemeshCycleSequence,
    index: int,
    *,
    source_already_validated: bool = False,
) -> RuntimeRemeshScheduleBoundaryObservation:
    values = _derive_boundary_values(
        source,
        index,
        source_already_validated=source_already_validated,
    )
    return RuntimeRemeshScheduleBoundaryObservation(
        **values,
        _proof_stamp=_stamp_from_values(
            _BOUNDARY_PROOF_VERSION,
            _BOUNDARY_FIELD_NAMES,
            values,
        ),
    )


@dataclass(frozen=True, slots=True)
class RuntimeRemeshScheduleSequenceObservation:
    """One sealed finite telescope of adjacent runtime/history balances."""

    source_sequence: ObservedEventRemeshCycleSequence = field(repr=False)
    boundaries: tuple[RuntimeRemeshScheduleBoundaryObservation, ...]
    exact_common_normalized_metric: ExactVector
    exact_augmented_energy_initial: Fraction
    exact_augmented_energy_final: Fraction
    exact_total_energy_drop: Fraction
    exact_total_gain_based_energy_drop_lower_bound: Fraction
    exact_total_schedule_augmented_energy_gain_slack: Fraction
    conditions: tuple[tuple[str, bool], ...]
    _proof_stamp: tuple[Any, ...] = field(default=(), repr=False, compare=False)

    def _proof_fields_are_intact(self) -> bool:
        try:
            current = _object_values(
                self,
                RuntimeRemeshScheduleSequenceObservation,
            )
            current_stamp = _stamp_from_values(
                _SEQUENCE_PROOF_VERSION,
                _SEQUENCE_FIELD_NAMES,
                current,
            )
            if not proof_stamps_are_identical(
                object.__getattribute__(self, "_proof_stamp"),
                current_stamp,
            ):
                return False
            source = object.__getattribute__(self, "source_sequence")
            if not _source_sequence_is_intact(source):
                return False
            expected = _derive_sequence_values(
                source,
                boundary_overrides=object.__getattribute__(self, "boundaries"),
                source_already_validated=True,
            )
            expected_stamp = _stamp_from_values(
                _SEQUENCE_PROOF_VERSION,
                _SEQUENCE_FIELD_NAMES,
                expected,
            )
            return proof_stamps_are_identical(
                object.__getattribute__(self, "_proof_stamp"),
                expected_stamp,
            )
        except BaseException:
            return False

    @property
    def scope(self) -> str:
        return _SEQUENCE_SCOPE

    @property
    def sequence_observation_certified(self) -> bool:
        return self._proof_fields_are_intact()

    @property
    def failed_conditions(self) -> tuple[str, ...]:
        if not self._proof_fields_are_intact():
            return ("runtime_remesh_schedule_sequence_proof_fields_intact",)
        return tuple(name for name, passed in self.conditions if not passed)

    @property
    def exact_recorded_history_advance_certified(self) -> bool:
        return self._proof_fields_are_intact()

    @property
    def exact_finite_energy_telescope_certified(self) -> bool:
        return self._proof_fields_are_intact()

    @property
    def energy_nonincrease_observed(self) -> bool:
        return bool(
            self._proof_fields_are_intact() and self.exact_total_energy_drop >= 0
        )

    @property
    def energy_nonincrease_sufficiently_certified(self) -> bool:
        return bool(
            self._proof_fields_are_intact()
            and self.exact_total_gain_based_energy_drop_lower_bound >= 0
        )

    @property
    def every_boundary_energy_nonincrease_observed(self) -> bool:
        return bool(
            self._proof_fields_are_intact()
            and all(
                boundary.energy_nonincrease_observed
                for boundary in self.boundaries
            )
        )

    @property
    def shared_graph_execution_provenance_certified(self) -> bool:
        return False

    @property
    def whole_sequence_atomicity_certified(self) -> bool:
        return False

    @property
    def runtime_global_gain_certified(self) -> bool:
        return False

    @property
    def repeated_runtime_stability_certified(self) -> bool:
        return False

    @property
    def future_stability_certified(self) -> bool:
        return False


_SEQUENCE_FIELD_NAMES = tuple(
    item.name
    for item in fields(RuntimeRemeshScheduleSequenceObservation)
    if item.name != "_proof_stamp"
)


def _derive_sequence_values(
    source: ObservedEventRemeshCycleSequence,
    *,
    boundary_overrides: (
        tuple[RuntimeRemeshScheduleBoundaryObservation, ...] | None
    ) = None,
    source_already_validated: bool = False,
) -> dict[str, Any]:
    if not source_already_validated and not _source_sequence_is_intact(source):
        raise TNFRValueError("cycle sequence is unsealed, tampered, or inconsistent")
    boundaries_were_built = boundary_overrides is None
    if boundaries_were_built:
        boundaries = tuple(
            _build_boundary(
                source,
                index,
                source_already_validated=True,
            )
            for index in range(len(source.boundaries))
        )
    else:
        boundaries = boundary_overrides
        if (
            type(boundaries) is not tuple
            or len(boundaries) != len(source.boundaries)
            or any(
                type(boundary)
                is not RuntimeRemeshScheduleBoundaryObservation
                or boundary.source_sequence is not source
                or boundary.boundary_index != index
                or not _boundary_proof_fields_are_intact(
                    boundary,
                    source_already_validated=True,
                )
                for index, boundary in enumerate(boundaries)
            )
        ):
            raise TNFRValueError(
                "stored runtime REMESH/schedule boundaries are inconsistent"
            )
    if not boundaries:
        raise TNFRValueError("cycle sequence must contain an adjacent boundary")

    initial = boundaries[0].exact_augmented_energy_before
    final = boundaries[-1].exact_augmented_energy_after
    total_drop = sum(
        (boundary.exact_energy_drop for boundary in boundaries),
        Fraction(0),
    )
    total_lower_bound = sum(
        (
            boundary.exact_gain_based_energy_drop_lower_bound
            for boundary in boundaries
        ),
        Fraction(0),
    )
    total_slack = sum(
        (
            boundary.exact_schedule_augmented_energy_gain_slack
            for boundary in boundaries
        ),
        Fraction(0),
    )
    intermediate_continuity = all(
        left.exact_augmented_energy_after == right.exact_augmented_energy_before
        for left, right in zip(boundaries, boundaries[1:])
    )
    conditions = (
        ("source_sequence_intact", True),
        (
            "complete_adjacent_boundary_cardinality",
            len(boundaries) == len(source.cycles) - 1,
        ),
        (
            "every_adjacent_boundary_intact",
            boundaries_were_built
            or all(
                boundary.boundary_observation_certified
                for boundary in boundaries
            ),
        ),
        (
            "exact_intermediate_augmented_energies_continuous",
            intermediate_continuity,
        ),
        (
            "exact_finite_energy_drop_telescope",
            total_drop == initial - final,
        ),
        (
            "summed_gain_lower_bound_and_slack_identity",
            total_drop == total_lower_bound + total_slack,
        ),
    )
    if not _strict_conditions(conditions, _SEQUENCE_CONDITION_NAMES) or not all(
        passed for _, passed in conditions
    ):
        failed = ", ".join(name for name, passed in conditions if not passed)
        raise TNFRValueError(f"runtime REMESH/schedule telescope failed: {failed}")

    common_metric = source.exact_common_normalized_metric_ray
    if common_metric is None:
        raise TNFRValueError("cycle sequence omits its exact common metric")
    return {
        "source_sequence": source,
        "boundaries": boundaries,
        "exact_common_normalized_metric": common_metric,
        "exact_augmented_energy_initial": initial,
        "exact_augmented_energy_final": final,
        "exact_total_energy_drop": total_drop,
        "exact_total_gain_based_energy_drop_lower_bound": total_lower_bound,
        "exact_total_schedule_augmented_energy_gain_slack": total_slack,
        "conditions": conditions,
    }


def observe_runtime_remesh_schedule_sequence(
    sequence: ObservedEventRemeshCycleSequence,
) -> RuntimeRemeshScheduleSequenceObservation:
    """Bind adjacent recorded cycles to one exact finite energy telescope."""

    if type(sequence) is not ObservedEventRemeshCycleSequence:
        raise TypeError("sequence must be an ObservedEventRemeshCycleSequence")
    values = _derive_sequence_values(sequence)
    result = RuntimeRemeshScheduleSequenceObservation(
        **values,
        _proof_stamp=_stamp_from_values(
            _SEQUENCE_PROOF_VERSION,
            _SEQUENCE_FIELD_NAMES,
            values,
        ),
    )
    current = _object_values(result, RuntimeRemeshScheduleSequenceObservation)
    if not proof_stamps_are_identical(
        object.__getattribute__(result, "_proof_stamp"),
        _stamp_from_values(
            _SEQUENCE_PROOF_VERSION,
            _SEQUENCE_FIELD_NAMES,
            current,
        ),
    ):
        raise RuntimeError(
            "constructed runtime REMESH/schedule sequence is inconsistent"
        )
    return result
