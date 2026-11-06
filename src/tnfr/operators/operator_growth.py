"""Typed signed EPI accounting for a committed operator schedule on (s, t].

Exact fractions describe the captured binary64 states and declared schedule
clock. Flow increments are numerical endpoint differences, not certified ODE
integrals. This passive observation neither supplies a Mutation permission nor
writes a continuous history or an event-consumption receipt.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any

from ..utils._structural_signature import (
    proof_stamps_are_identical,
    structural_proof_signature,
)
from .event_runtime import OperatorEventExecutionResult

__all__ = (
    "OperatorEPIContribution",
    "OperatorEPIGrowthObservation",
    "observe_operator_epi_growth",
)

ExactVector = tuple[Fraction, ...]


@dataclass(frozen=True, slots=True)
class OperatorEPIContribution:
    """One included flow or zero-duration glyph, in actual execution order."""

    position: int
    operation_kind: str
    operation_index: int
    operator_name: str | None
    exact_start_time: Fraction
    exact_end_time: Fraction
    exact_epi_before: ExactVector
    exact_epi_after: ExactVector
    exact_increment: ExactVector


@dataclass(frozen=True, slots=True)
class OperatorEPIGrowthObservation:
    """Detached accounting report, not an authenticated activating certificate.

    None totals mean unavailable coverage, never zero growth. Contributions
    may retain a partial captured trace when the complete window is unavailable.
    Every fraction concerns represented endpoints; the declared duration is
    not inferred by subtracting displayed binary64 timestamps. A later consumer
    must rebuild the source and separately admit live identity/context and reuse.
    """

    nodes: tuple[Any, ...]
    exact_start_time: Fraction
    exact_end_time: Fraction
    exact_duration: Fraction
    available: bool
    unavailability_reasons: tuple[str, ...]
    included_event_indices: tuple[int, ...]
    excluded_initial_event_indices: tuple[int, ...]
    contributions: tuple[OperatorEPIContribution, ...]
    exact_epi_before: ExactVector | None = None
    exact_epi_after: ExactVector | None = None
    exact_flow_increment: ExactVector | None = None
    exact_jump_increment: ExactVector | None = None
    exact_jump_variation: ExactVector | None = None
    exact_net_increment: ExactVector | None = None
    exact_net_growth_rate: ExactVector | None = None
    source: str = field(default="operator_schedule_net_growth", init=False)


def _same_nodes(left: Any, right: Any) -> bool:
    return proof_stamps_are_identical(
        structural_proof_signature(left), structural_proof_signature(right)
    )


def _difference(after: ExactVector, before: ExactVector) -> ExactVector:
    return tuple(right - left for left, right in zip(before, after, strict=True))


def observe_operator_epi_growth(
    result: OperatorEventExecutionResult,
) -> OperatorEPIGrowthObservation:
    """Observe signed flow/jump/net growth on the whole declared (s, t] window.

    All initial-time glyphs are excluded and all terminal glyphs included.
    Positive-duration flow intervals and included glyphs must have complete,
    contiguous EPI endpoints on the executor's fixed ordered target population.
    Edge invariance and affine-energy/metric certification are not required.

    A malformed, substituted or unsealed execution record raises. An intact
    record without requested/available endpoint coverage returns an unavailable
    observation. A zero-duration window never yields a rate. The graph is not
    read or modified, and no action is replayed.
    """
    if type(result) is not OperatorEventExecutionResult:
        raise TypeError("operator growth requires an OperatorEventExecutionResult")
    if not result._proof_fields_are_intact():
        raise ValueError("operator growth requires intact committed execution evidence")

    schedule = result.schedule
    start, end = schedule.exact_start_time, schedule.exact_end_time
    duration = schedule.exact_total_flow_duration
    excluded = tuple(
        event.event_index
        for event in schedule.events
        if event.exact_event_time == start
    )
    included = tuple(
        event.event_index
        for event in schedule.events
        if start < event.exact_event_time <= end
    )
    reasons: list[str] = []
    contributions: list[OperatorEPIContribution] = []
    if duration <= 0:
        reasons.append("nonpositive_window_duration")
    composition = result.represented_epi_schedule_composition
    if composition is None:
        reasons.append("operation_endpoint_trace_unavailable")
    else:
        # Rebuild coverage from the admitted schedule and records. Its cached
        # energy-gain conditions deliberately do not decide this observable.
        expected = []
        for interval in schedule.intervals:
            if interval.exact_duration > 0:
                expected.append(("flow", interval.index))
            if interval.index < schedule.event_count:
                expected.append(("glyph", interval.index))
        actual = tuple(
            (operation.operation_kind, operation.operation_index)
            for operation in composition.operations
        )
        if actual != tuple(expected):
            reasons.append("operation_coverage_incomplete")

        previous_after = None
        initial_after = None
        included_count = 0
        for operation in composition.operations:
            index = operation.operation_index
            if operation.operation_kind == "glyph":
                event = schedule.events[index]
                left_time = right_time = event.exact_event_time
                if left_time == start:
                    if index == excluded[-1]:
                        if _same_nodes(operation.nodes, result.target_nodes):
                            initial_after = operation.exact_epi_after
                    continue
            else:
                interval = schedule.intervals[index]
                left_time = interval.exact_start_time
                right_time = interval.exact_end_time
            included_count += 1
            before, after = operation.exact_epi_before, operation.exact_epi_after
            prefix = f"operation_{operation.position}"
            if not _same_nodes(operation.nodes, result.target_nodes):
                reasons.append(f"{prefix}_node_population_or_order_changed")
                if before is None or after is None:
                    reasons.append(f"{prefix}_endpoints_unavailable")
                previous_after = None
                continue
            if before is None or after is None:
                reasons.append(f"{prefix}_endpoints_unavailable")
                previous_after = None
                continue
            if len(before) != len(result.target_nodes) or len(after) != len(before):
                reasons.append(f"{prefix}_endpoint_dimension_mismatch")
                previous_after = None
                continue
            if included_count == 1 and initial_after is not None:
                if before != initial_after:
                    reasons.append("initial_right_boundary_disconnected")
            if previous_after is not None and before != previous_after:
                reasons.append(f"{prefix}_endpoint_disconnected")
            previous_after = after
            contributions.append(
                OperatorEPIContribution(
                    position=operation.position,
                    operation_kind=operation.operation_kind,
                    operation_index=index,
                    operator_name=operation.operator_name,
                    exact_start_time=left_time,
                    exact_end_time=right_time,
                    exact_epi_before=before,
                    exact_epi_after=after,
                    exact_increment=_difference(after, before),
                )
            )
        if not included_count:
            reasons.append("included_operation_trace_empty")

    common = dict(
        nodes=result.target_nodes,
        exact_start_time=start,
        exact_end_time=end,
        exact_duration=duration,
        included_event_indices=included,
        excluded_initial_event_indices=excluded,
        contributions=tuple(contributions),
    )
    if reasons:
        return OperatorEPIGrowthObservation(
            **common,
            available=False,
            unavailability_reasons=tuple(dict.fromkeys(reasons)),
        )

    initial, final = (
        contributions[0].exact_epi_before,
        contributions[-1].exact_epi_after,
    )
    zero = Fraction(0)
    flow = tuple(
        sum(
            (
                item.exact_increment[i]
                for item in contributions
                if item.operation_kind == "flow"
            ),
            zero,
        )
        for i in range(len(result.target_nodes))
    )
    jump = tuple(
        sum(
            (
                item.exact_increment[i]
                for item in contributions
                if item.operation_kind == "glyph"
            ),
            zero,
        )
        for i in range(len(result.target_nodes))
    )
    variation = tuple(
        sum(
            (
                abs(item.exact_increment[i])
                for item in contributions
                if item.operation_kind == "glyph"
            ),
            zero,
        )
        for i in range(len(result.target_nodes))
    )
    net = _difference(final, initial)
    if net != tuple(a + b for a, b in zip(flow, jump, strict=True)):
        # A complete contiguous trace telescopes exactly. Keep this explicit
        # rather than publish a favorable net if that invariant ever breaks.
        return OperatorEPIGrowthObservation(
            **common,
            available=False,
            unavailability_reasons=("exact_accounting_identity_failed",),
        )
    return OperatorEPIGrowthObservation(
        **common,
        available=True,
        unavailability_reasons=(),
        exact_epi_before=initial,
        exact_epi_after=final,
        exact_flow_increment=flow,
        exact_jump_increment=jump,
        exact_jump_variation=variation,
        exact_net_increment=net,
        exact_net_growth_rate=tuple(value / duration for value in net),
    )
