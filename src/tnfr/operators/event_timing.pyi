from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from numbers import Real
from typing import Iterable

from ..types import Glyph


@dataclass(frozen=True, slots=True)
class StructuralFlowInterval:
    index: int
    start_time: float
    end_time: float
    duration: float
    start_offset: Fraction
    end_offset: Fraction
    exact_start_time: Fraction
    exact_end_time: Fraction
    exact_duration: Fraction
    time_basis: str = field(default=..., init=False)
    timestamp_role: str = field(default=..., init=False)
    duration_is_authoritative: bool = field(default=..., init=False)
    feeds_epi_time_history: bool = field(default=..., init=False)

    def __post_init__(self) -> None: ...


@dataclass(frozen=True, slots=True)
class PhysicalFlowPartition:
    parent_interval: StructuralFlowInterval
    segment_durations: tuple[float, ...]
    segments: tuple[StructuralFlowInterval, ...]

    @property
    def time_basis(self) -> str: ...
    @property
    def boundary_role(self) -> str: ...
    @property
    def numerical_substeps_are_physical_boundaries(self) -> bool: ...

    @property
    def segment_count(self) -> int: ...

    @property
    def boundary_times(self) -> tuple[float, ...]: ...

    @property
    def exact_boundary_times(self) -> tuple[Fraction, ...]: ...


@dataclass(frozen=True, slots=True)
class ScheduledOperatorEvent:
    event_index: int
    cycle_index: int
    word_position: int
    operator_name: str
    glyph: Glyph
    event_time: float
    event_offset: Fraction
    exact_event_time: Fraction
    time_basis: str = field(default=..., init=False)
    timestamp_role: str = field(default=..., init=False)
    history_channel: str = field(default=..., init=False)
    feeds_epi_time_history: bool = field(default=..., init=False)
    coincident_event_order: str = field(default=..., init=False)

    def __post_init__(self) -> None: ...


@dataclass(frozen=True, slots=True)
class OperatorEventSchedule:
    operator_names: tuple[str, ...]
    cycles: int
    start_time: float
    flow_durations: tuple[float, ...]
    intervals: tuple[StructuralFlowInterval, ...]
    events: tuple[ScheduledOperatorEvent, ...]
    end_time: float
    total_flow_duration: float
    exact_start_time: Fraction
    exact_end_time: Fraction
    exact_total_flow_duration: Fraction
    time_basis: str = field(default=..., init=False)
    timestamp_role: str = field(default=..., init=False)
    duration_and_offsets_are_authoritative: bool = field(
        default=..., init=False
    )
    event_timestamps_feed_epi_time_history: bool = field(
        default=..., init=False
    )
    event_history_channel: str = field(default=..., init=False)
    coincident_event_order: str = field(default=..., init=False)

    def __post_init__(self) -> None: ...

    @property
    def event_count(self) -> int: ...


@dataclass(frozen=True, slots=True)
class OperatorEventRuntimeClockDiagnostic:
    schedule: OperatorEventSchedule
    scope: str = field(default=..., init=False)

    def __post_init__(self) -> None: ...

    @property
    def collapsed_positive_interval_indices(self) -> tuple[int, ...]: ...

    @property
    def nonadditive_positive_interval_indices(self) -> tuple[int, ...]: ...

    @property
    def zhir_event_indices_without_positive_preflow(self) -> tuple[int, ...]: ...

    @property
    def zhir_event_indices_with_collapsed_preflow(self) -> tuple[int, ...]: ...

    @property
    def zhir_event_indices_with_duration_mismatch(self) -> tuple[int, ...]: ...

    @property
    def binary64_runtime_clock_compatible(self) -> bool: ...

    @property
    def timestamped_zhir_preflow_clock_compatible(self) -> bool: ...

    @property
    def clock_binding_ready(self) -> bool: ...


def build_physical_flow_partition(
    parent_interval: StructuralFlowInterval,
    segment_durations: Iterable[Real],
) -> PhysicalFlowPartition: ...


def build_operator_event_schedule(
    operator_names: Iterable[str],
    *,
    cycles: int = ...,
    start_time: Real,
    flow_durations: Iterable[Real],
) -> OperatorEventSchedule: ...


def diagnose_operator_event_runtime_clock(
    schedule: OperatorEventSchedule,
) -> OperatorEventRuntimeClockDiagnostic: ...


__all__: tuple[str, ...]
