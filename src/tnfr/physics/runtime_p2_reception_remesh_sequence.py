r"""Finite causal extinction for executed P2 Reception/REMESH cycles.

The global P2 theorem proves that binary64 half-Reception maps every finite
pair to numeric consensus and that alpha-one hard-clipped REMESH copies the
selected global-delay field in real numeric value.  This module binds those
two facts to every cycle of one executor-owned causal sequence.

The certificate is deliberately finite.  It proves exact spatial
disagreement extinction inside the completed sequence once at least
``tau_global + 1`` cycles have been observed.  It does not promote the
observation to future runtime executions, auxiliary channels, solver
accuracy, or full TNFR stability.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, fields, replace
from fractions import Fraction
from typing import Any

from ..errors import TNFRValueError
from ..operators.event_remesh_causal_runtime import (
    ExecutedEventRemeshCycleSequence,
)
from ..operators.event_remesh_runtime import EventRemeshCycleResult
from ..utils._structural_signature import (
    binary64_vectors_are_identical,
    proof_stamps_are_identical,
    structural_proof_signature,
)
from .binary64_p2_reception_stability import (
    P2HalfReceptionRemeshStabilityCertificate,
)
from .runtime_p2_reception_stage import (
    ExecutedP2HalfReceptionStageCertificate,
    certify_executed_p2_half_reception_stage,
)
from .runtime_remesh_history_stability import (
    RuntimeRemeshHistoryBridgeObservation,
    observe_runtime_remesh_history_bridge,
)

__all__ = (
    "ExecutedP2HalfReceptionRemeshSequenceCertificate",
    "certify_executed_p2_half_reception_remesh_sequence",
)

ExactPair = tuple[Fraction, Fraction]

_PROOF_VERSION = "executed_p2_half_reception_remesh_sequence_v1"
_SCOPE = (
    "One executor-sealed finite causal P2 sequence on fixed ordered support "
    "and metric. Every selected Reception stage realizes the binary64 "
    "half-mix consensus kernel, every applied alpha-one hard-clipped REMESH "
    "step is bound to its exact active outgoing-history suffix, and all "
    "observed post-REMESH endpoints from cycle index tau_global onward have "
    "exactly zero spatial disagreement energy. "
    "The result retains same-invocation causal order, boundary continuity, "
    "and graph-owned sequence atomicity. Future or unobserved repetition, "
    "auxiliary-state stability, solver accuracy, and full TNFR stability "
    "are not certified."
)

_CONDITION_NAMES = (
    "global_p2_kernel_certificate_is_intact",
    "executed_causal_sequence_is_intact",
    "cycle_count_reaches_the_active_history_extinction_horizon",
    "ordered_support_matches_the_p2_kernel",
    "fixed_exact_metric_matches_the_p2_kernel",
    "same_invocation_causal_order_is_certified",
    "whole_sequence_graph_state_atomicity_is_certified",
    "exact_recorded_cycle_boundaries_are_continuous",
    "every_selected_reception_stage_is_executor_bound",
    "every_reception_stage_spans_its_cycle_schedule_epi_endpoints",
    "every_reception_output_has_exact_zero_spatial_energy",
    "every_remesh_bridge_is_bound_to_its_same_cycle",
    "every_remesh_configuration_matches_the_source_class",
    "every_active_history_suffix_is_sufficient_and_inside_the_source_interval",
    "every_active_history_suffix_is_bound_to_its_remesh_bridge",
    "every_remesh_is_an_exact_numeric_global_delay_copy",
    "every_post_horizon_active_history_suffix_has_zero_spatial_energy",
    "final_observed_post_remesh_spatial_energy_is_zero",
    "global_kernel_cycle_bound_at_the_observed_horizon_is_zero",
)


def _same(left: Any, right: Any) -> bool:
    try:
        return proof_stamps_are_identical(
            structural_proof_signature(left),
            structural_proof_signature(right),
        )
    except BaseException:
        return False


def _strict_conditions(value: Any) -> bool:
    return bool(
        type(value) is tuple
        and len(value) == len(_CONDITION_NAMES)
        and all(
            type(item) is tuple
            and len(item) == 2
            and type(item[0]) is str
            and type(item[1]) is bool
            and item[0] == _CONDITION_NAMES[index]
            for index, item in enumerate(value)
        )
    )


def _proof_stamp(
    value: "ExecutedP2HalfReceptionRemeshSequenceCertificate",
) -> tuple[Any, ...]:
    if type(value) is not ExecutedP2HalfReceptionRemeshSequenceCertificate:
        raise TypeError("proof value must have its canonical result type")
    stages = object.__getattribute__(value, "reception_stage_certificates")
    bridges = object.__getattribute__(value, "remesh_history_bridges")
    stage_references = stages if type(stages) is tuple else ()
    bridge_references = bridges if type(bridges) is tuple else ()
    opaque = (
        object.__getattribute__(value, "kernel_certificate"),
        object.__getattribute__(value, "execution"),
        *stage_references,
        *bridge_references,
    )
    payload = tuple(
        (item.name, object.__getattribute__(value, item.name))
        for item in fields(ExecutedP2HalfReceptionRemeshSequenceCertificate)
        if item.name != "_proof_stamp"
    )
    return (
        _PROOF_VERSION,
        structural_proof_signature(payload, opaque_references=opaque),
    )


def _seal(
    value: "ExecutedP2HalfReceptionRemeshSequenceCertificate",
) -> "ExecutedP2HalfReceptionRemeshSequenceCertificate":
    return replace(value, _proof_stamp=_proof_stamp(value))


def _sealed(value: Any) -> bool:
    try:
        return proof_stamps_are_identical(
            object.__getattribute__(value, "_proof_stamp"),
            _proof_stamp(value),
        )
    except BaseException:
        return False


def _exact_pair(values: Any, *, label: str) -> ExactPair:
    if (
        type(values) is not tuple
        or len(values) != 2
        or any(type(item) is not float or not math.isfinite(item) for item in values)
    ):
        raise TNFRValueError(f"{label} must be a finite binary64 pair")
    return Fraction.from_float(values[0]), Fraction.from_float(values[1])


def _active_history_suffix_is_admissible(
    cycle: EventRemeshCycleResult,
    *,
    active_length: int,
    history_maxlen: int,
    lower: Fraction,
    upper: Fraction,
) -> bool:
    history = cycle.history_transition.outgoing_exact_history
    return bool(
        type(history) is tuple
        and active_length <= len(history) <= history_maxlen
        and all(
            type(row) is tuple
            and len(row) == 2
            and all(
                type(value) is Fraction and lower <= value <= upper
                for value in row
            )
            for row in history[-active_length:]
        )
    )


def _normalize_metric(value: Any) -> ExactPair:
    if (
        type(value) is not tuple
        or len(value) != 2
        or any(
            type(item) is not float
            or not math.isfinite(item)
            or item <= 0.0
            for item in value
        )
    ):
        raise TNFRValueError("cycle metric must be a positive binary64 pair")
    exact = tuple(Fraction.from_float(item) for item in value)
    total = exact[0] + exact[1]
    return exact[0] / total, exact[1] / total


def _normalize_exact_metric(value: Any) -> ExactPair:
    if (
        type(value) is not tuple
        or len(value) != 2
        or any(type(item) is not Fraction or item <= 0 for item in value)
    ):
        raise TNFRValueError("bridge metric must be a positive exact pair")
    total = value[0] + value[1]
    return value[0] / total, value[1] / total


def _centered_energy(value: ExactPair, metric: ExactPair) -> Fraction:
    center = metric[0] * value[0] + metric[1] * value[1]
    return (
        metric[0] * (value[0] - center) ** 2
        + metric[1] * (value[1] - center) ** 2
    ) / 2


def _remesh_configuration_matches_source(
    cycle: EventRemeshCycleResult,
    bridge: RuntimeRemeshHistoryBridgeObservation,
    *,
    source: Any,
    nodes: tuple[Any, Any],
    metric: ExactPair,
    alpha_source: str,
) -> bool:
    plan = cycle.remesh.plan
    transition = cycle.history_transition
    try:
        return bool(
            cycle.remesh.applied
            and type(plan.alpha) is float
            and binary64_vectors_are_identical((plan.alpha,), (1.0,))
            and type(plan.alpha_source) is str
            and plan.alpha_source
            and plan.alpha_source == alpha_source
            and plan.tau_local == source.tau_local
            and plan.tau_global == source.tau_global
            and plan.required_history_length == source.required_history_length
            and plan.clip_mode == "hard"
            and binary64_vectors_are_identical(
                (plan.epi_min, plan.epi_max),
                (source.configuration.epi_min, source.configuration.epi_max),
            )
            and transition.history_maxlen == source.history_maxlen
            and cycle.history_maxlen == source.history_maxlen
            and _same(cycle.target_nodes, nodes)
            and _normalize_metric(cycle.metric_weights) == metric
            and _normalize_exact_metric(bridge.exact_metric_weights) == metric
        )
    except BaseException:
        return False


@dataclass(frozen=True, slots=True)
class _SequenceModel:
    cycle_indices: tuple[int, ...]
    reception_event_indices: tuple[int, ...]
    reception_stage_certificates: tuple[
        ExecutedP2HalfReceptionStageCertificate, ...
    ]
    remesh_history_bridges: tuple[
        RuntimeRemeshHistoryBridgeObservation, ...
    ]
    node_order: tuple[Any, Any]
    exact_normalized_metric: ExactPair
    cycle_count: int
    active_history_extinction_horizon: int
    guaranteed_extinction_cycle_index: int
    exact_schedule_input_energies: tuple[Fraction, ...]
    exact_post_reception_energies: tuple[Fraction, ...]
    exact_post_remesh_energies: tuple[Fraction, ...]
    exact_active_history_suffixes: tuple[tuple[ExactPair, ...], ...]
    exact_final_post_remesh_energy: Fraction
    conditions: tuple[tuple[str, bool], ...]


def _derive_model(
    kernel: P2HalfReceptionRemeshStabilityCertificate,
    execution: ExecutedEventRemeshCycleSequence,
    reception_event_indices: tuple[int, ...],
    *,
    stage_evidence: tuple[
        ExecutedP2HalfReceptionStageCertificate, ...
    ] | None = None,
    bridge_evidence: tuple[
        RuntimeRemeshHistoryBridgeObservation, ...
    ] | None = None,
) -> _SequenceModel:
    cycles = object.__getattribute__(execution, "cycles")
    cycle_count = len(cycles)
    if len(reception_event_indices) != cycle_count:
        raise TNFRValueError(
            "reception_event_indices must contain one index per executed cycle"
        )
    if stage_evidence is not None and (
        type(stage_evidence) is not tuple or len(stage_evidence) != cycle_count
    ):
        raise TNFRValueError("stored Reception evidence has invalid cardinality")
    if bridge_evidence is not None and (
        type(bridge_evidence) is not tuple or len(bridge_evidence) != cycle_count
    ):
        raise TNFRValueError("stored REMESH evidence has invalid cardinality")

    horizon = object.__getattribute__(kernel, "active_history_extinction_horizon")
    if cycle_count < horizon:
        raise TNFRValueError(
            "executed cycle count does not reach the active-history "
            "extinction horizon"
        )

    source = object.__getattribute__(kernel, "remesh_class_certificate")
    nodes = object.__getattribute__(kernel, "node_order")
    metric = object.__getattribute__(kernel, "exact_normalized_metric")
    if type(nodes) is not tuple or len(nodes) != 2:
        raise TNFRValueError("kernel must retain exactly two ordered nodes")
    typed_nodes = (nodes[0], nodes[1])
    typed_metric = (metric[0], metric[1])
    if not _same(execution.target_nodes, typed_nodes):
        raise TNFRValueError("executed support does not match the P2 kernel")

    stages: list[ExecutedP2HalfReceptionStageCertificate] = []
    bridges: list[RuntimeRemeshHistoryBridgeObservation] = []
    schedule_input_energies: list[Fraction] = []
    post_reception_energies: list[Fraction] = []
    post_remesh_energies: list[Fraction] = []
    active_history_suffixes: list[tuple[ExactPair, ...]] = []
    first_plan = cycles[0].remesh.plan
    if type(first_plan.alpha_source) is not str or not first_plan.alpha_source:
        raise TNFRValueError("REMESH alpha_source must be a nonempty string")
    common_alpha_source = first_plan.alpha_source

    for cycle_index, (cycle, event_index) in enumerate(
        zip(cycles, reception_event_indices, strict=True)
    ):
        if type(cycle) is not EventRemeshCycleResult:
            raise TNFRValueError("execution contains a noncanonical cycle result")
        if stage_evidence is None:
            stage = certify_executed_p2_half_reception_stage(
                kernel,
                cycle.event_execution,
                event_index=event_index,
            )
        else:
            stage = stage_evidence[cycle_index]
        if bridge_evidence is None:
            bridge = observe_runtime_remesh_history_bridge(cycle)
        else:
            bridge = bridge_evidence[cycle_index]
        if (
            type(stage) is not ExecutedP2HalfReceptionStageCertificate
            or stage.kernel_certificate is not kernel
            or stage.execution_result is not cycle.event_execution
            or stage.event_index != event_index
            or not stage._proof_fields_are_intact_after_dependencies_validation()
        ):
            raise TNFRValueError("Reception stage lost same-cycle identity")
        if (
            type(bridge) is not RuntimeRemeshHistoryBridgeObservation
            or bridge.cycle_result is not cycle
            or not bridge._proof_fields_are_intact()
        ):
            raise TNFRValueError("REMESH bridge lost same-cycle identity")

        pre_schedule_exact = _exact_pair(
            cycle.pre_schedule_epi.epi_values,
            label=f"cycle[{cycle_index}] pre-schedule EPI",
        )
        pre_remesh_exact = _exact_pair(
            cycle.pre_remesh_epi.epi_values,
            label=f"cycle[{cycle_index}] pre-REMESH EPI",
        )
        if stage.exact_epi_before != pre_schedule_exact:
            raise TNFRValueError(
                "selected Reception does not start at the cycle schedule input"
            )
        if stage.exact_epi_after != pre_remesh_exact:
            raise TNFRValueError(
                "selected Reception does not end at the cycle schedule output"
            )
        if not binary64_vectors_are_identical(
            stage.binary64_epi_before,
            cycle.pre_schedule_epi.epi_values,
        ) or not binary64_vectors_are_identical(
            stage.binary64_epi_after,
            cycle.pre_remesh_epi.epi_values,
        ):
            raise TNFRValueError(
                "selected Reception binary64 endpoints do not span the schedule"
            )
        if stage.exact_centered_energy_after != 0:
            raise TNFRValueError("Reception output must have exact zero energy")
        schedule_input_energy = _centered_energy(
            pre_schedule_exact,
            typed_metric,
        )
        if stage.exact_centered_energy_before != schedule_input_energy:
            raise TNFRValueError(
                "Reception input energy does not use the normalized source metric"
            )

        transition = cycle.history_transition
        if not _remesh_configuration_matches_source(
            cycle,
            bridge,
            source=source,
            nodes=typed_nodes,
            metric=typed_metric,
            alpha_source=common_alpha_source,
        ):
            raise TNFRValueError(
                "executed REMESH configuration does not match the source class"
            )
        if not _active_history_suffix_is_admissible(
            cycle,
            active_length=horizon,
            history_maxlen=source.history_maxlen,
            lower=source.epi_min,
            upper=source.epi_max,
        ):
            raise TNFRValueError(
                "active REMESH history suffix is insufficient, exceeds history "
                "capacity, or leaves the source interval"
            )
        active_suffix = tuple(transition.outgoing_exact_history[-horizon:])
        if tuple(reversed(active_suffix)) != bridge.exact_history:
            raise TNFRValueError(
                "active outgoing-history suffix does not match the REMESH bridge"
            )
        selected_global = transition.selected_global_delayed_epi
        if (
            type(selected_global) is not tuple
            or len(selected_global) != 2
            or any(type(item) is not Fraction for item in selected_global)
            or bridge.exact_runtime_raw_next_field != selected_global
            or bridge.exact_runtime_bounded_next_field != selected_global
            or any(value != 0 for value in bridge.exact_rounding_residual)
            or any(value != 0 for value in bridge.exact_clipping_residual)
            or any(value != 0 for value in bridge.exact_total_residual)
        ):
            raise TNFRValueError(
                "alpha-one REMESH is not an exact numeric global-delay copy"
            )
        post_exact = _exact_pair(
            cycle.post_remesh_epi.epi_values,
            label=f"cycle[{cycle_index}] post-REMESH EPI",
        )
        post_remesh_energy = _centered_energy(post_exact, typed_metric)
        if (
            post_exact != selected_global
            or bridge.exact_runtime_bounded_next_field != post_exact
            or bridge.exact_runtime_bounded_next_energy
            != cycle.post_remesh_epi.exact_disagreement_energy
        ):
            raise TNFRValueError(
                "REMESH bridge does not match the committed cycle endpoint"
            )
        if cycle_index > 0:
            previous = cycles[cycle_index - 1]
            if not binary64_vectors_are_identical(
                previous.post_remesh_epi.epi_values,
                cycle.pre_schedule_epi.epi_values,
            ):
                raise TNFRValueError("adjacent cycle EPI boundaries are discontinuous")

        stages.append(stage)
        bridges.append(bridge)
        schedule_input_energies.append(schedule_input_energy)
        post_reception_energies.append(stage.exact_centered_energy_after)
        post_remesh_energies.append(post_remesh_energy)
        active_history_suffixes.append(active_suffix)

    extinction_index = horizon - 1
    if any(value != 0 for value in post_remesh_energies[extinction_index:]):
        raise TNFRValueError(
            "observed post-horizon REMESH disagreement did not extinguish"
        )
    if not (
        kernel.exact_effective_head_energy_gain_upper_bound == 0
        and cycle_count >= horizon
    ):
        raise TNFRValueError("global kernel bound is not zero at this cycle count")

    stage_tuple = tuple(stages)
    bridge_tuple = tuple(bridges)
    schedule_tuple = tuple(schedule_input_energies)
    reception_tuple = tuple(post_reception_energies)
    remesh_tuple = tuple(post_remesh_energies)
    active_suffix_tuple = tuple(active_history_suffixes)
    post_horizon_suffixes_are_consensus = all(
        all(_centered_energy(row, typed_metric) == 0 for row in suffix)
        for suffix in active_suffix_tuple[extinction_index:]
    )
    if not post_horizon_suffixes_are_consensus:
        raise TNFRValueError(
            "post-horizon active history retains nonzero spatial disagreement"
        )
    boundary_continuity = (
        execution.observed_sequence.exact_recorded_boundary_continuity_certified
    )
    if not boundary_continuity:
        raise TNFRValueError("recorded cycle boundaries are not continuous")
    conditions = (
        (
            "global_p2_kernel_certificate_is_intact",
            True,
        ),
        ("executed_causal_sequence_is_intact", True),
        (
            "cycle_count_reaches_the_active_history_extinction_horizon",
            cycle_count >= horizon,
        ),
        ("ordered_support_matches_the_p2_kernel", _same(execution.target_nodes, nodes)),
        (
            "fixed_exact_metric_matches_the_p2_kernel",
            all(
                _normalize_metric(cycle.metric_weights) == typed_metric
                for cycle in cycles
            ),
        ),
        (
            "same_invocation_causal_order_is_certified",
            True,
        ),
        (
            "whole_sequence_graph_state_atomicity_is_certified",
            True,
        ),
        (
            "exact_recorded_cycle_boundaries_are_continuous",
            boundary_continuity,
        ),
        (
            "every_selected_reception_stage_is_executor_bound",
            all(
                stage.execution_result is cycle.event_execution
                for stage, cycle in zip(stage_tuple, cycles, strict=True)
            ),
        ),
        (
            "every_reception_stage_spans_its_cycle_schedule_epi_endpoints",
            all(
                stage.exact_epi_before
                == _exact_pair(cycle.pre_schedule_epi.epi_values, label="pre")
                and stage.exact_epi_after
                == _exact_pair(cycle.pre_remesh_epi.epi_values, label="post")
                for stage, cycle in zip(stage_tuple, cycles, strict=True)
            ),
        ),
        (
            "every_reception_output_has_exact_zero_spatial_energy",
            all(value == 0 for value in reception_tuple),
        ),
        (
            "every_remesh_bridge_is_bound_to_its_same_cycle",
            all(
                bridge.cycle_result is cycle
                for bridge, cycle in zip(bridge_tuple, cycles, strict=True)
            ),
        ),
        (
            "every_remesh_configuration_matches_the_source_class",
            all(
                _remesh_configuration_matches_source(
                    cycle,
                    bridge,
                    source=source,
                    nodes=typed_nodes,
                    metric=typed_metric,
                    alpha_source=common_alpha_source,
                )
                for cycle, bridge in zip(cycles, bridge_tuple, strict=True)
            ),
        ),
        (
            "every_active_history_suffix_is_sufficient_and_inside_the_source_interval",
            all(
                _active_history_suffix_is_admissible(
                    cycle,
                    active_length=horizon,
                    history_maxlen=source.history_maxlen,
                    lower=source.epi_min,
                    upper=source.epi_max,
                )
                for cycle in cycles
            ),
        ),
        (
            "every_active_history_suffix_is_bound_to_its_remesh_bridge",
            all(
                tuple(reversed(suffix)) == bridge.exact_history
                for suffix, bridge in zip(
                    active_suffix_tuple,
                    bridge_tuple,
                    strict=True,
                )
            ),
        ),
        (
            "every_remesh_is_an_exact_numeric_global_delay_copy",
            all(
                bridge.exact_total_residual == (Fraction(0), Fraction(0))
                and bridge.exact_runtime_bounded_next_field
                == cycle.history_transition.selected_global_delayed_epi
                for bridge, cycle in zip(bridge_tuple, cycles, strict=True)
            ),
        ),
        (
            "every_post_horizon_active_history_suffix_has_zero_spatial_energy",
            post_horizon_suffixes_are_consensus,
        ),
        (
            "final_observed_post_remesh_spatial_energy_is_zero",
            remesh_tuple[-1] == 0,
        ),
        (
            "global_kernel_cycle_bound_at_the_observed_horizon_is_zero",
            kernel.exact_effective_head_energy_gain_upper_bound == 0
            and cycle_count >= horizon,
        ),
    )
    if not all(passed for _name, passed in conditions):
        failed = ", ".join(name for name, passed in conditions if not passed)
        raise TNFRValueError(f"executed P2 sequence conditions failed: {failed}")
    return _SequenceModel(
        cycle_indices=execution.cycle_indices,
        reception_event_indices=reception_event_indices,
        reception_stage_certificates=stage_tuple,
        remesh_history_bridges=bridge_tuple,
        node_order=typed_nodes,
        exact_normalized_metric=typed_metric,
        cycle_count=cycle_count,
        active_history_extinction_horizon=horizon,
        guaranteed_extinction_cycle_index=extinction_index,
        exact_schedule_input_energies=schedule_tuple,
        exact_post_reception_energies=reception_tuple,
        exact_post_remesh_energies=remesh_tuple,
        exact_active_history_suffixes=active_suffix_tuple,
        exact_final_post_remesh_energy=remesh_tuple[-1],
        conditions=conditions,
    )


def _matches_model(
    value: "ExecutedP2HalfReceptionRemeshSequenceCertificate",
    model: _SequenceModel,
) -> bool:
    for item in fields(_SequenceModel):
        observed = object.__getattribute__(value, item.name)
        expected = object.__getattribute__(model, item.name)
        if item.name == "reception_stage_certificates":
            if type(observed) is not tuple or len(observed) != len(expected):
                return False
            if any(
                left is not right
                for left, right in zip(observed, expected, strict=True)
            ):
                return False
        elif item.name == "remesh_history_bridges":
            if type(observed) is not tuple or len(observed) != len(expected):
                return False
            if any(
                left is not right
                for left, right in zip(observed, expected, strict=True)
            ):
                return False
        elif not _same(observed, expected):
            return False
    return True


@dataclass(frozen=True, slots=True)
class ExecutedP2HalfReceptionRemeshSequenceCertificate:
    """Sealed finite causal certificate for executed P2 EN/REMESH cycles."""

    kernel_certificate: P2HalfReceptionRemeshStabilityCertificate = field(
        repr=False
    )
    execution: ExecutedEventRemeshCycleSequence = field(repr=False)
    cycle_indices: tuple[int, ...]
    reception_event_indices: tuple[int, ...]
    reception_stage_certificates: tuple[
        ExecutedP2HalfReceptionStageCertificate, ...
    ] = field(repr=False)
    remesh_history_bridges: tuple[
        RuntimeRemeshHistoryBridgeObservation, ...
    ] = field(repr=False)
    node_order: tuple[Any, Any]
    exact_normalized_metric: ExactPair
    cycle_count: int
    active_history_extinction_horizon: int
    guaranteed_extinction_cycle_index: int
    exact_schedule_input_energies: tuple[Fraction, ...]
    exact_post_reception_energies: tuple[Fraction, ...]
    exact_post_remesh_energies: tuple[Fraction, ...]
    exact_active_history_suffixes: tuple[tuple[ExactPair, ...], ...]
    exact_final_post_remesh_energy: Fraction
    conditions: tuple[tuple[str, bool], ...]
    _proof_stamp: tuple[Any, ...] = field(default=(), repr=False, compare=False)

    def _proof_fields_are_intact(self) -> bool:
        try:
            if type(self) is not ExecutedP2HalfReceptionRemeshSequenceCertificate:
                return False
            if not _sealed(self) or not _strict_conditions(self.conditions):
                return False
            kernel = object.__getattribute__(self, "kernel_certificate")
            execution = object.__getattribute__(self, "execution")
            stages = object.__getattribute__(self, "reception_stage_certificates")
            bridges = object.__getattribute__(self, "remesh_history_bridges")
            if (
                type(kernel) is not P2HalfReceptionRemeshStabilityCertificate
                or not kernel._proof_fields_are_intact()
                or type(execution) is not ExecutedEventRemeshCycleSequence
                or not execution._proof_fields_are_intact()
                or type(stages) is not tuple
                or type(bridges) is not tuple
            ):
                return False
            model = _derive_model(
                kernel,
                execution,
                object.__getattribute__(self, "reception_event_indices"),
                stage_evidence=stages,
                bridge_evidence=bridges,
            )
            return _matches_model(self, model)
        except BaseException:
            return False

    @property
    def scope(self) -> str:
        return _SCOPE

    @property
    def executed_p2_half_reception_remesh_sequence_certificate_certified(
        self,
    ) -> bool:
        return bool(
            self._proof_fields_are_intact()
            and all(passed for _name, passed in self.conditions)
        )

    @property
    def failed_conditions(self) -> tuple[str, ...]:
        if not self._proof_fields_are_intact():
            return ("executed_p2_reception_remesh_sequence_proof_fields_intact",)
        return tuple(name for name, passed in self.conditions if not passed)

    @property
    def finite_causal_extinction_certified(self) -> bool:
        return (
            self.executed_p2_half_reception_remesh_sequence_certificate_certified
        )

    @property
    def same_invocation_causal_provenance_certified(self) -> bool:
        return self.executed_p2_half_reception_remesh_sequence_certificate_certified

    @property
    def every_reception_stage_q_zero_certified(self) -> bool:
        return self.executed_p2_half_reception_remesh_sequence_certificate_certified

    @property
    def every_remesh_global_delay_copy_eta_zero_certified(self) -> bool:
        return self.executed_p2_half_reception_remesh_sequence_certificate_certified

    @property
    def observed_active_history_extinction_certified(self) -> bool:
        return self.executed_p2_half_reception_remesh_sequence_certificate_certified

    @property
    def whole_sequence_graph_state_atomicity_certified(self) -> bool:
        return self.executed_p2_half_reception_remesh_sequence_certificate_certified

    @property
    def future_runtime_stability_certified(self) -> bool:
        return False

    @property
    def unobserved_repetition_stability_certified(self) -> bool:
        return False

    @property
    def auxiliary_state_stability_certified(self) -> bool:
        return False

    @property
    def current_live_graph_state_bound(self) -> bool:
        return False

    @property
    def solver_accuracy_certified(self) -> bool:
        return False

    @property
    def full_tnfr_stability_certified(self) -> bool:
        return False


def _materialize_event_indices(
    execution: ExecutedEventRemeshCycleSequence,
    value: tuple[int, ...] | None,
) -> tuple[int, ...]:
    count = len(execution.cycles)
    if value is None:
        result: list[int] = []
        for cycle in execution.cycles:
            indices = tuple(
                index
                for index, stage in enumerate(
                    cycle.event_execution.glyph_stage_evidence
                )
                if type(stage.event.operator_name) is str
                and stage.event.operator_name == "reception"
            )
            if len(indices) != 1:
                raise TNFRValueError(
                    "reception_event_indices is required unless every cycle "
                    "has exactly one Reception event"
                )
            result.append(indices[0])
        return tuple(result)
    if type(value) is not tuple or len(value) != count or any(
        type(item) is not int or item < 0 for item in value
    ):
        raise TNFRValueError(
            "reception_event_indices must be a nonnegative integer tuple with "
            "one item per cycle"
        )
    return value


def certify_executed_p2_half_reception_remesh_sequence(
    kernel_certificate: P2HalfReceptionRemeshStabilityCertificate,
    execution: ExecutedEventRemeshCycleSequence,
    *,
    reception_event_indices: tuple[int, ...] | None = None,
) -> ExecutedP2HalfReceptionRemeshSequenceCertificate:
    """Bind one completed causal sequence to finite P2 spatial extinction."""

    if type(kernel_certificate) is not P2HalfReceptionRemeshStabilityCertificate:
        raise TNFRValueError(
            "kernel_certificate must be an exact P2 half-Reception certificate"
        )
    if not kernel_certificate._proof_fields_are_intact() or not all(
        passed for _name, passed in kernel_certificate.conditions
    ):
        raise TNFRValueError(
            "kernel_certificate is unsealed, tampered, or inconsistent"
        )
    if type(execution) is not ExecutedEventRemeshCycleSequence:
        raise TNFRValueError(
            "execution must be an exact executed event/REMESH cycle sequence"
        )
    if not execution._proof_fields_are_intact():
        raise TNFRValueError("execution is unsealed, tampered, or inconsistent")
    indices = _materialize_event_indices(execution, reception_event_indices)
    model = _derive_model(kernel_certificate, execution, indices)
    value = ExecutedP2HalfReceptionRemeshSequenceCertificate(
        kernel_certificate=kernel_certificate,
        execution=execution,
        cycle_indices=model.cycle_indices,
        reception_event_indices=model.reception_event_indices,
        reception_stage_certificates=model.reception_stage_certificates,
        remesh_history_bridges=model.remesh_history_bridges,
        node_order=model.node_order,
        exact_normalized_metric=model.exact_normalized_metric,
        cycle_count=model.cycle_count,
        active_history_extinction_horizon=(
            model.active_history_extinction_horizon
        ),
        guaranteed_extinction_cycle_index=(
            model.guaranteed_extinction_cycle_index
        ),
        exact_schedule_input_energies=model.exact_schedule_input_energies,
        exact_post_reception_energies=model.exact_post_reception_energies,
        exact_post_remesh_energies=model.exact_post_remesh_energies,
        exact_active_history_suffixes=model.exact_active_history_suffixes,
        exact_final_post_remesh_energy=model.exact_final_post_remesh_energy,
        conditions=model.conditions,
    )
    result = _seal(value)
    if not result.executed_p2_half_reception_remesh_sequence_certificate_certified:
        raise RuntimeError("constructed executed P2 sequence is inconsistent")
    return result
