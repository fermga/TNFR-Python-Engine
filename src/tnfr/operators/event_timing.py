"""Immutable physical timeline for canonical TNFR operator events.

Canonical operators are represented here as instantaneous hybrid jumps.  The
TNFR nodal equation governs the explicitly declared flow intervals before,
between and after those jumps.  Consequently, a finite schedule with ``m``
operator events contains exactly ``m + 1`` flow intervals.

The timeline is independent of a numerical solver partition.  Refining an
interval may add solver substeps later, but it must not move an event boundary
or duplicate an operator event.  This module only constructs and validates the
timeline; it never executes an operator or advances a graph.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from numbers import Real
from operator import index as integer_index
from typing import Any, Iterable

from .._exact_time import (
    finite_represented_real,
    materialize_nonnegative_time_sequence,
    represented_fraction,
    represented_fraction_as_float,
)
from ..config.operator_names import CANONICAL_OPERATOR_NAMES
from ..types import Glyph
from .registry import get_operator_class

__all__ = (
    "OperatorEventRuntimeClockDiagnostic",
    "OperatorEventSchedule",
    "PhysicalFlowPartition",
    "ScheduledOperatorEvent",
    "StructuralFlowInterval",
    "build_physical_flow_partition",
    "build_operator_event_schedule",
    "diagnose_operator_event_runtime_clock",
)


_PHYSICAL_TIME_BASIS = "physical_time"
_TIMESTAMP_ROLE = "binary64_representation_only"
_EVENT_HISTORY_CHANNEL = "hybrid_event_log"
_CLOCK_DIAGNOSTIC_SCOPE = (
    "schedule-only necessary conditions for binding exact operator-event "
    "coordinates to the current binary64 runtime clock, including direct "
    "interval addition, and timestamped ZHIR evidence; graph state, solver "
    "accuracy and trigger magnitude remain untested"
)


def _positive_integer(value: Any, label: str) -> int:
    """Return a positive integer without accepting Boolean coercion."""

    if isinstance(value, bool):
        raise TypeError(f"{label} must be a positive integer, not a boolean")
    try:
        result = integer_index(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise TypeError(f"{label} must be a positive integer") from exc
    if result <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return result


def _nonnegative_index(value: Any, label: str) -> int:
    """Validate a stored zero-based integer index."""

    if isinstance(value, bool):
        raise TypeError(f"{label} must be a nonnegative integer, not a boolean")
    try:
        result = integer_index(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise TypeError(f"{label} must be a nonnegative integer") from exc
    if result < 0:
        raise ValueError(f"{label} must be a nonnegative integer")
    return result


def _canonical_operator_names(values: Iterable[str]) -> tuple[str, ...]:
    """Resolve lowercase public executable identifiers without aliases."""

    if isinstance(values, (str, bytes, bytearray)):
        raise TypeError("operator_names must be an iterable of public operator names")
    try:
        names = tuple(values)
    except TypeError as exc:
        raise TypeError(
            "operator_names must be an iterable of public operator names"
        ) from exc
    for position, name in enumerate(names):
        if type(name) is not str:
            raise TypeError(
                f"operator_names[{position}] must be a public operator name string"
            )
        if name not in CANONICAL_OPERATOR_NAMES:
            raise ValueError(
                f"unknown public operator name at operator_names[{position}]: {name!r}"
            )
        operator_type = get_operator_class(name)
        if operator_type.name != name or not isinstance(operator_type.glyph, Glyph):
            raise RuntimeError(
                f"canonical operator registry is inconsistent for public name {name!r}"
            )
    return names


def _operator_glyph(name: str) -> Glyph:
    """Return the glyph for one validated public executable identifier."""

    glyph = get_operator_class(name).glyph
    if not isinstance(glyph, Glyph):  # Defensive against registry corruption.
        raise RuntimeError(f"canonical operator {name!r} has no Glyph binding")
    return glyph


@dataclass(frozen=True, slots=True)
class StructuralFlowInterval:
    """One nodal-flow interval with authoritative rational time coordinates.

    ``start_time`` and ``end_time`` are display representations.  Physical
    duration and ordering use the exact rational fields, which are derived from
    the materialized binary64 duration rather than by subtracting timestamps.
    Interval records do not populate ``epi_time_history``; runtime integration
    will require an explicit sampling policy.
    """

    index: int
    start_time: float
    end_time: float
    duration: float
    start_offset: Fraction
    end_offset: Fraction
    exact_start_time: Fraction
    exact_end_time: Fraction
    exact_duration: Fraction
    time_basis: str = field(default=_PHYSICAL_TIME_BASIS, init=False)
    timestamp_role: str = field(default=_TIMESTAMP_ROLE, init=False)
    duration_is_authoritative: bool = field(default=True, init=False)
    feeds_epi_time_history: bool = field(default=False, init=False)

    def __getattribute__(self, name: str) -> Any:
        """Expose the historical fixed metadata without allowing promotion."""

        fixed = {
            "time_basis": _PHYSICAL_TIME_BASIS,
            "timestamp_role": _TIMESTAMP_ROLE,
            "duration_is_authoritative": True,
            "feeds_epi_time_history": False,
        }
        if name in fixed:
            return fixed[name]
        return object.__getattribute__(self, name)

    def __post_init__(self) -> None:
        fixed = {
            "time_basis": _PHYSICAL_TIME_BASIS,
            "timestamp_role": _TIMESTAMP_ROLE,
            "duration_is_authoritative": True,
            "feeds_epi_time_history": False,
        }
        if any(
            object.__getattribute__(self, name) != expected
            for name, expected in fixed.items()
        ):
            raise ValueError("flow interval contract fields are inconsistent")
        index = _nonnegative_index(self.index, "interval index")
        start_exact_from_float = represented_fraction(
            self.start_time, "interval start_time"
        )
        end_exact_from_float = represented_fraction(
            self.end_time, "interval end_time"
        )
        duration_exact_from_float = represented_fraction(
            self.duration, "interval duration"
        )
        if duration_exact_from_float < 0:
            raise ValueError("interval duration must be nonnegative")
        exact_fields = (
            (self.start_offset, "interval start_offset"),
            (self.end_offset, "interval end_offset"),
            (self.exact_start_time, "interval exact_start_time"),
            (self.exact_end_time, "interval exact_end_time"),
            (self.exact_duration, "interval exact_duration"),
        )
        for value, label in exact_fields:
            if type(value) is not Fraction:
                raise TypeError(f"{label} must be an exact Fraction")
        if self.start_offset < 0 or self.end_offset < self.start_offset:
            raise ValueError("interval offsets must be nonnegative and ordered")
        if self.exact_duration != duration_exact_from_float:
            raise ValueError(
                "interval exact_duration must equal represented duration"
            )
        if self.end_offset - self.start_offset != self.exact_duration:
            raise ValueError("interval offsets must differ by exact_duration")
        if self.exact_end_time - self.exact_start_time != self.exact_duration:
            raise ValueError("interval exact times must differ by exact_duration")
        expected_start = represented_fraction_as_float(
            self.exact_start_time, "interval exact_start_time"
        )
        expected_end = represented_fraction_as_float(
            self.exact_end_time, "interval exact_end_time"
        )
        if Fraction.from_float(expected_start) != start_exact_from_float:
            raise ValueError("interval start_time must represent exact_start_time")
        if Fraction.from_float(expected_end) != end_exact_from_float:
            raise ValueError("interval end_time must represent exact_end_time")
        object.__setattr__(self, "index", index)

def _derive_physical_flow_segments(
    parent_interval: StructuralFlowInterval,
    segment_durations: tuple[float, ...],
    exact_segment_durations: tuple[Fraction, ...],
) -> tuple[StructuralFlowInterval, ...]:
    """Derive an executable binary64 partition of one physical interval."""

    if type(parent_interval) is not StructuralFlowInterval:
        raise TypeError("parent_interval must be a StructuralFlowInterval")
    parent_interval.__post_init__()
    if parent_interval.exact_duration <= 0:
        raise ValueError("parent_interval must have positive exact duration")
    if not parent_interval.end_time > parent_interval.start_time:
        raise ValueError("parent_interval collapses on the binary64 clock")
    if (
        parent_interval.start_time + parent_interval.duration
        != parent_interval.end_time
    ):
        raise ValueError("parent_interval is nonadditive on the binary64 clock")
    if (
        parent_interval.end_time - parent_interval.start_time
        != parent_interval.duration
    ):
        raise ValueError(
            "parent_interval binary64 subtraction does not recover duration"
        )
    if len(segment_durations) < 2:
        raise ValueError(
            "segment_durations must contain at least two positive segments"
        )
    if len(exact_segment_durations) != len(segment_durations):
        raise ValueError("exact segment durations must align with segment_durations")
    if any(duration <= 0 for duration in exact_segment_durations):
        raise ValueError("segment_durations must be strictly positive")
    if (
        sum(exact_segment_durations, Fraction(0))
        != parent_interval.exact_duration
    ):
        raise ValueError(
            "exact segment duration sum must equal parent exact_duration"
        )

    segments: list[StructuralFlowInterval] = []
    current_offset = parent_interval.start_offset
    exact_current_time = parent_interval.exact_start_time
    for segment_index, (duration, exact_duration) in enumerate(
        zip(segment_durations, exact_segment_durations)
    ):
        end_offset = current_offset + exact_duration
        exact_end_time = exact_current_time + exact_duration
        start_time = represented_fraction_as_float(
            exact_current_time,
            f"physical segments[{segment_index}] exact_start_time",
        )
        end_time = represented_fraction_as_float(
            exact_end_time,
            f"physical segments[{segment_index}] exact_end_time",
        )
        if not end_time > start_time:
            raise ValueError(
                f"physical segment {segment_index} collapses on the binary64 clock"
            )
        if start_time + duration != end_time:
            raise ValueError(
                f"physical segment {segment_index} is nonadditive on the "
                "binary64 clock"
            )
        if end_time - start_time != duration:
            raise ValueError(
                f"physical segment {segment_index} binary64 subtraction does "
                "not recover duration"
            )
        segment = StructuralFlowInterval(
            index=segment_index,
            start_time=start_time,
            end_time=end_time,
            duration=duration,
            start_offset=current_offset,
            end_offset=end_offset,
            exact_start_time=exact_current_time,
            exact_end_time=exact_end_time,
            exact_duration=exact_duration,
        )
        if segments:
            previous = segments[-1]
            if (
                previous.end_offset != segment.start_offset
                or previous.exact_end_time != segment.exact_start_time
                or previous.end_time != segment.start_time
            ):
                raise RuntimeError(
                    "derived physical segment boundaries are not continuous"
                )
        segments.append(segment)
        current_offset = end_offset
        exact_current_time = exact_end_time

    first = segments[0]
    last = segments[-1]
    if (
        first.start_offset != parent_interval.start_offset
        or first.exact_start_time != parent_interval.exact_start_time
        or first.start_time != parent_interval.start_time
        or last.end_offset != parent_interval.end_offset
        or last.exact_end_time != parent_interval.exact_end_time
        or last.end_time != parent_interval.end_time
    ):
        raise RuntimeError(
            "derived physical segment boundaries do not cover parent_interval"
        )
    return tuple(segments)


@dataclass(frozen=True, slots=True)
class PhysicalFlowPartition:
    """Immutable physical subdivision of one scheduled flow interval.

    Every segment is an independently executable physical interval.  Its
    boundaries are distinct from any numerical substeps an integrator may use
    internally.  Construction rejects a partition unless exact rational time
    and both directions of binary64 endpoint arithmetic agree for every
    segment.
    """

    parent_interval: StructuralFlowInterval
    segment_durations: tuple[float, ...]
    segments: tuple[StructuralFlowInterval, ...]

    def __post_init__(self) -> None:
        if type(self.parent_interval) is not StructuralFlowInterval:
            raise TypeError("parent_interval must be a StructuralFlowInterval")
        if type(self.segment_durations) is not tuple:
            raise TypeError("segment_durations must be an immutable tuple")
        if any(type(value) is not float for value in self.segment_durations):
            raise TypeError("segment_durations must contain binary64 floats")
        if type(self.segments) is not tuple:
            raise TypeError("segments must be an immutable tuple")
        if any(type(item) is not StructuralFlowInterval for item in self.segments):
            raise TypeError(
                "segments must contain StructuralFlowInterval records"
            )
        durations, exact_durations = materialize_nonnegative_time_sequence(
            self.segment_durations, "segment_durations"
        )
        expected_segments = _derive_physical_flow_segments(
            self.parent_interval,
            durations,
            exact_durations,
        )
        if self.segments != expected_segments:
            raise ValueError(
                "physical partition segments do not match their canonical "
                "timeline"
            )

    @property
    def time_basis(self) -> str:
        """Name the physical-time coordinate shared by all segments."""

        return _PHYSICAL_TIME_BASIS

    @property
    def boundary_role(self) -> str:
        """Identify each segment boundary as an explicit pressure refresh."""

        return "explicit_physical_pressure_refresh"

    @property
    def numerical_substeps_are_physical_boundaries(self) -> bool:
        """Keep integrator substeps distinct from declared physical segments."""

        return False

    @property
    def segment_count(self) -> int:
        """Number of explicit physical flow segments."""

        return len(self.segments)

    @property
    def boundary_times(self) -> tuple[float, ...]:
        """Ordered binary64 boundary representations, including both ends."""

        return (self.segments[0].start_time,) + tuple(
            segment.end_time for segment in self.segments
        )

    @property
    def exact_boundary_times(self) -> tuple[Fraction, ...]:
        """Ordered authoritative boundary coordinates, including both ends."""

        return (self.segments[0].exact_start_time,) + tuple(
            segment.exact_end_time for segment in self.segments
        )


def build_physical_flow_partition(
    parent_interval: StructuralFlowInterval,
    segment_durations: Iterable[Real],
) -> PhysicalFlowPartition:
    """Build an explicit pressure-refresh partition of one flow interval.

    At least two strictly positive represented durations are required.  Their
    exact binary64 rationals must sum to the parent's authoritative duration,
    and every materialized segment must advance, land on, and subtract back
    from its binary64 endpoint exactly.
    """

    if type(parent_interval) is not StructuralFlowInterval:
        raise TypeError("parent_interval must be a StructuralFlowInterval")
    durations, exact_durations = materialize_nonnegative_time_sequence(
        segment_durations, "segment_durations"
    )
    segments = _derive_physical_flow_segments(
        parent_interval,
        durations,
        exact_durations,
    )
    return PhysicalFlowPartition(
        parent_interval=parent_interval,
        segment_durations=durations,
        segments=segments,
    )


@dataclass(frozen=True, slots=True)
class ScheduledOperatorEvent:
    """One zero-duration jump routed to the ordered hybrid event log.

    The represented ``event_time`` is display-only.  ``event_index`` orders
    coincident jumps, while ``event_offset`` and ``exact_event_time`` carry the
    physical coordinate.  An event timestamp is never a secant sample for
    ``epi_time_history`` because an instantaneous jump has no finite derivative.
    """

    event_index: int
    cycle_index: int
    word_position: int
    operator_name: str
    glyph: Glyph
    event_time: float
    event_offset: Fraction
    exact_event_time: Fraction
    time_basis: str = field(default=_PHYSICAL_TIME_BASIS, init=False)
    timestamp_role: str = field(default=_TIMESTAMP_ROLE, init=False)
    history_channel: str = field(default=_EVENT_HISTORY_CHANNEL, init=False)
    feeds_epi_time_history: bool = field(default=False, init=False)
    coincident_event_order: str = field(default="event_index", init=False)

    def __getattribute__(self, name: str) -> Any:
        """Expose the historical fixed metadata without allowing promotion."""

        fixed = {
            "time_basis": _PHYSICAL_TIME_BASIS,
            "timestamp_role": _TIMESTAMP_ROLE,
            "history_channel": _EVENT_HISTORY_CHANNEL,
            "feeds_epi_time_history": False,
            "coincident_event_order": "event_index",
        }
        if name in fixed:
            return fixed[name]
        return object.__getattribute__(self, name)

    def __post_init__(self) -> None:
        fixed = {
            "time_basis": _PHYSICAL_TIME_BASIS,
            "timestamp_role": _TIMESTAMP_ROLE,
            "history_channel": _EVENT_HISTORY_CHANNEL,
            "feeds_epi_time_history": False,
            "coincident_event_order": "event_index",
        }
        if any(
            object.__getattribute__(self, name) != expected
            for name, expected in fixed.items()
        ):
            raise ValueError("scheduled event contract fields are inconsistent")
        event_index = _nonnegative_index(self.event_index, "event index")
        cycle_index = _nonnegative_index(self.cycle_index, "cycle index")
        word_position = _nonnegative_index(self.word_position, "word position")
        names = _canonical_operator_names((self.operator_name,))
        expected_glyph = _operator_glyph(names[0])
        if not isinstance(self.glyph, Glyph) or self.glyph is not expected_glyph:
            raise ValueError(
                f"event glyph must be {expected_glyph.value!r} for operator "
                f"{names[0]!r}"
            )
        represented_event = represented_fraction(self.event_time, "event_time")
        if type(self.event_offset) is not Fraction:
            raise TypeError("event_offset must be an exact Fraction")
        if type(self.exact_event_time) is not Fraction:
            raise TypeError("exact_event_time must be an exact Fraction")
        if self.event_offset < 0:
            raise ValueError("event_offset must be nonnegative")
        expected_event = represented_fraction_as_float(
            self.exact_event_time, "exact_event_time"
        )
        if Fraction.from_float(expected_event) != represented_event:
            raise ValueError("event_time must represent exact_event_time")
        object.__setattr__(self, "event_index", event_index)
        object.__setattr__(self, "cycle_index", cycle_index)
        object.__setattr__(self, "word_position", word_position)
        object.__setattr__(self, "operator_name", names[0])

def _derive_timeline(
    operator_names: tuple[str, ...],
    cycles: int,
    start_time: float,
    flow_durations: tuple[float, ...],
    exact_flow_durations: tuple[Fraction, ...],
) -> tuple[
    tuple[StructuralFlowInterval, ...],
    tuple[ScheduledOperatorEvent, ...],
    float,
    float,
    Fraction,
    Fraction,
    Fraction,
]:
    """Derive the canonical alternating flow/jump timeline."""

    event_count = len(operator_names) * cycles
    expected_duration_count = event_count + 1
    if len(flow_durations) != expected_duration_count:
        raise ValueError(
            "flow_durations must contain exactly one more entry than the "
            f"{event_count} scheduled operator events"
        )
    if len(exact_flow_durations) != len(flow_durations):
        raise ValueError("exact flow durations must align with flow_durations")
    exact_start = represented_fraction(start_time, "start_time")
    exact_total = sum(exact_flow_durations, Fraction(0))
    total_duration = represented_fraction_as_float(
        exact_total, "total flow duration"
    )

    intervals: list[StructuralFlowInterval] = []
    events: list[ScheduledOperatorEvent] = []
    current_offset = Fraction(0)
    exact_current_time = exact_start
    word_length = len(operator_names)
    for interval_index, (duration, exact_duration) in enumerate(
        zip(flow_durations, exact_flow_durations)
    ):
        end_offset = current_offset + exact_duration
        exact_end_time = exact_start + end_offset
        represented_start = represented_fraction_as_float(
            exact_current_time,
            f"intervals[{interval_index}] exact_start_time",
        )
        represented_end = represented_fraction_as_float(
            exact_end_time,
            f"intervals[{interval_index}] exact_end_time",
        )
        interval = StructuralFlowInterval(
            index=interval_index,
            start_time=represented_start,
            end_time=represented_end,
            duration=duration,
            start_offset=current_offset,
            end_offset=end_offset,
            exact_start_time=exact_current_time,
            exact_end_time=exact_end_time,
            exact_duration=exact_duration,
        )
        intervals.append(interval)
        if interval_index < event_count:
            cycle_index, word_position = divmod(interval_index, word_length)
            name = operator_names[word_position]
            events.append(
                ScheduledOperatorEvent(
                    event_index=interval_index,
                    cycle_index=cycle_index,
                    word_position=word_position,
                    operator_name=name,
                    glyph=_operator_glyph(name),
                    event_time=represented_end,
                    event_offset=end_offset,
                    exact_event_time=exact_end_time,
                )
            )
        current_offset = end_offset
        exact_current_time = exact_end_time
    end_time = represented_fraction_as_float(exact_current_time, "end_time")
    return (
        tuple(intervals),
        tuple(events),
        end_time,
        total_duration,
        exact_start,
        exact_current_time,
        exact_total,
    )


@dataclass(frozen=True, slots=True)
class OperatorEventSchedule:
    """Finite immutable schedule on an exact represented physical clock.

    ``operator_names`` describes one word. ``events`` expands that word over
    ``cycles``. Interval ``k`` ends at event ``k`` for every event, and the
    final interval follows the last event. Rational offsets and durations are
    authoritative. Float timestamps are display representations and must not
    be subtracted to infer duration or written to ``epi_time_history``. Public
    construction should normally use :func:`build_operator_event_schedule`;
    ``__post_init__`` independently rebuilds all derived fields.
    """

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
    time_basis: str = field(default=_PHYSICAL_TIME_BASIS, init=False)
    timestamp_role: str = field(default=_TIMESTAMP_ROLE, init=False)
    duration_and_offsets_are_authoritative: bool = field(
        default=True, init=False
    )
    event_timestamps_feed_epi_time_history: bool = field(
        default=False, init=False
    )
    event_history_channel: str = field(
        default=_EVENT_HISTORY_CHANNEL, init=False
    )
    coincident_event_order: str = field(default="event_index", init=False)

    def __getattribute__(self, name: str) -> Any:
        """Expose the historical fixed metadata without allowing promotion."""

        fixed = {
            "time_basis": _PHYSICAL_TIME_BASIS,
            "timestamp_role": _TIMESTAMP_ROLE,
            "duration_and_offsets_are_authoritative": True,
            "event_timestamps_feed_epi_time_history": False,
            "event_history_channel": _EVENT_HISTORY_CHANNEL,
            "coincident_event_order": "event_index",
        }
        if name in fixed:
            return fixed[name]
        return object.__getattribute__(self, name)

    def __post_init__(self) -> None:
        fixed = {
            "time_basis": _PHYSICAL_TIME_BASIS,
            "timestamp_role": _TIMESTAMP_ROLE,
            "duration_and_offsets_are_authoritative": True,
            "event_timestamps_feed_epi_time_history": False,
            "event_history_channel": _EVENT_HISTORY_CHANNEL,
            "coincident_event_order": "event_index",
        }
        if any(
            object.__getattribute__(self, name) != expected
            for name, expected in fixed.items()
        ):
            raise ValueError("operator schedule contract fields are inconsistent")
        if type(self.operator_names) is not tuple:
            raise TypeError("operator_names must be an immutable tuple")
        if type(self.flow_durations) is not tuple:
            raise TypeError("flow_durations must be an immutable tuple")
        if any(type(value) is not float for value in self.flow_durations):
            raise TypeError("flow_durations must contain binary64 floats")
        if type(self.intervals) is not tuple or type(self.events) is not tuple:
            raise TypeError("intervals and events must be immutable tuples")
        names = _canonical_operator_names(self.operator_names)
        cycles = _positive_integer(self.cycles, "cycles")
        if type(self.start_time) is not float:
            raise TypeError("start_time must be a binary64 float")
        start, start_exact = finite_represented_real(
            self.start_time, "start_time"
        )
        durations, exact_durations = materialize_nonnegative_time_sequence(
            self.flow_durations
        )
        intervals = self.intervals
        events = self.events
        if any(type(item) is not StructuralFlowInterval for item in intervals):
            raise TypeError("intervals must contain StructuralFlowInterval records")
        if any(type(item) is not ScheduledOperatorEvent for item in events):
            raise TypeError("events must contain ScheduledOperatorEvent records")
        if type(self.end_time) is not float:
            raise TypeError("end_time must be a binary64 float")
        if type(self.total_flow_duration) is not float:
            raise TypeError("total_flow_duration must be a binary64 float")
        end_exact_from_float = represented_fraction(self.end_time, "end_time")
        total_exact_from_float = represented_fraction(
            self.total_flow_duration, "total_flow_duration"
        )
        if total_exact_from_float < 0:
            raise ValueError("total_flow_duration must be nonnegative")
        exact_fields = (
            (self.exact_start_time, "exact_start_time"),
            (self.exact_end_time, "exact_end_time"),
            (self.exact_total_flow_duration, "exact_total_flow_duration"),
        )
        for value, label in exact_fields:
            if type(value) is not Fraction:
                raise TypeError(f"{label} must be an exact Fraction")
        (
            expected_intervals,
            expected_events,
            expected_end,
            expected_total,
            expected_exact_start,
            expected_exact_end,
            expected_exact_total,
        ) = _derive_timeline(
            names,
            cycles,
            start,
            durations,
            exact_durations,
        )
        if intervals != expected_intervals:
            raise ValueError("schedule intervals do not match its canonical timeline")
        if events != expected_events:
            raise ValueError("schedule events do not match its canonical timeline")
        if Fraction.from_float(expected_end) != end_exact_from_float:
            raise ValueError("schedule end_time does not match its canonical timeline")
        if Fraction.from_float(expected_total) != total_exact_from_float:
            raise ValueError(
                "schedule total_flow_duration does not match its declared durations"
            )
        if self.exact_start_time != expected_exact_start:
            raise ValueError("schedule exact_start_time does not match start_time")
        if self.exact_end_time != expected_exact_end:
            raise ValueError("schedule exact_end_time does not match its timeline")
        if self.exact_total_flow_duration != expected_exact_total:
            raise ValueError(
                "schedule exact_total_flow_duration does not match durations"
            )
        if start_exact != expected_exact_start:
            raise RuntimeError("represented schedule start lost its exact value")
        object.__setattr__(self, "operator_names", names)
        object.__setattr__(self, "cycles", cycles)
        object.__setattr__(self, "start_time", start)
        object.__setattr__(self, "flow_durations", durations)
        object.__setattr__(self, "intervals", intervals)
        object.__setattr__(self, "events", events)

    @property
    def event_count(self) -> int:
        """Number of instantaneous operator events in the finite schedule."""

        return len(self.events)


def build_operator_event_schedule(
    operator_names: Iterable[str],
    *,
    cycles: int = 1,
    start_time: Real,
    flow_durations: Iterable[Real],
) -> OperatorEventSchedule:
    """Build a canonical finite operator-event schedule.

    The public operator word is repeated ``cycles`` times without assigning a
    duration to any operator itself. The supplied ``flow_durations`` declare
    the physical nodal-flow intervals around those instantaneous jumps and
    must therefore contain exactly ``cycles * len(operator_names) + 1`` items.
    Zero-duration intervals are valid for a finite schedule.
    """

    names = _canonical_operator_names(operator_names)
    cycle_count = _positive_integer(cycles, "cycles")
    represented_start, _ = finite_represented_real(start_time, "start_time")
    durations, exact_durations = materialize_nonnegative_time_sequence(
        flow_durations
    )
    (
        intervals,
        events,
        end_time,
        total_duration,
        exact_start,
        exact_end,
        exact_total,
    ) = _derive_timeline(
        names,
        cycle_count,
        represented_start,
        durations,
        exact_durations,
    )
    return OperatorEventSchedule(
        operator_names=names,
        cycles=cycle_count,
        start_time=represented_start,
        flow_durations=durations,
        intervals=intervals,
        events=events,
        end_time=end_time,
        total_flow_duration=total_duration,
        exact_start_time=exact_start,
        exact_end_time=exact_end,
        exact_total_flow_duration=exact_total,
    )

@dataclass(frozen=True, slots=True)
class OperatorEventRuntimeClockDiagnostic:
    """Necessary binary64-clock conditions for future schedule execution.

    The exact schedule remains valid when a positive rational duration is too
    small to change its rounded display timestamp. Exact-prefix rounding can
    also make start_time + duration differ from the scheduled end even when
    both endpoints are ordered. The current runtime and
    ``epi_time_history`` use binary64 timestamps, so such an interval cannot
    yet be replayed as two strictly ordered physical samples. A ZHIR event
    additionally needs a positive representable flow interval immediately
    before its jump whose endpoint subtraction exactly recovers the declared
    duration. These are clock conditions only; they do not establish a Mutation
    threshold crossing or certify any numerical flow solver.
    """

    schedule: OperatorEventSchedule
    scope: str = field(default=_CLOCK_DIAGNOSTIC_SCOPE, init=False)

    def __getattribute__(self, name: str) -> Any:
        """Expose the historical fixed scope without allowing promotion."""

        if name == "scope":
            return _CLOCK_DIAGNOSTIC_SCOPE
        return object.__getattribute__(self, name)

    def __post_init__(self) -> None:
        if object.__getattribute__(self, "scope") != _CLOCK_DIAGNOSTIC_SCOPE:
            raise ValueError("runtime clock diagnostic scope is inconsistent")
        if type(self.schedule) is not OperatorEventSchedule:
            raise TypeError("schedule must be an OperatorEventSchedule")
        self.schedule.__post_init__()

    @property
    def collapsed_positive_interval_indices(self) -> tuple[int, ...]:
        """Positive exact intervals whose binary64 endpoints coincide."""

        return tuple(
            interval.index
            for interval in self.schedule.intervals
            if interval.exact_duration > 0
            and not interval.end_time > interval.start_time
        )

    @property
    def nonadditive_positive_interval_indices(self) -> tuple[int, ...]:
        """Positive intervals whose binary64 addition misses the endpoint."""

        return tuple(
            interval.index
            for interval in self.schedule.intervals
            if interval.exact_duration > 0
            and interval.start_time + interval.duration != interval.end_time
        )

    @property
    def zhir_event_indices_without_positive_preflow(self) -> tuple[int, ...]:
        """ZHIR jumps preceded by an exactly zero flow interval."""

        intervals = self.schedule.intervals
        return tuple(
            event.event_index
            for event in self.schedule.events
            if event.glyph is Glyph.ZHIR
            and intervals[event.event_index].exact_duration == 0
        )

    @property
    def zhir_event_indices_with_collapsed_preflow(self) -> tuple[int, ...]:
        """ZHIR jumps whose positive pre-flow has one rounded timestamp."""

        intervals = self.schedule.intervals
        return tuple(
            event.event_index
            for event in self.schedule.events
            if event.glyph is Glyph.ZHIR
            and intervals[event.event_index].exact_duration > 0
            and not (
                intervals[event.event_index].end_time
                > intervals[event.event_index].start_time
            )
        )

    @property
    def zhir_event_indices_with_duration_mismatch(self) -> tuple[int, ...]:
        """ZHIR jumps whose timestamp difference is not the declared duration.

        Flow duration is authoritative in an operator-event schedule. Mutation's
        timestamped secant currently reads two binary64 absolute timestamps, so
        execution must abstain when subtracting those representations would use
        a different physical interval.
        """

        intervals = self.schedule.intervals
        return tuple(
            event.event_index
            for event in self.schedule.events
            if event.glyph is Glyph.ZHIR
            and intervals[event.event_index].exact_duration > 0
            and (
                intervals[event.event_index].end_time
                - intervals[event.event_index].start_time
                != intervals[event.event_index].duration
            )
        )

    @property
    def binary64_runtime_clock_compatible(self) -> bool:
        """Whether each positive flow advances and lands on its endpoint."""

        return not (
            self.collapsed_positive_interval_indices
            or self.nonadditive_positive_interval_indices
        )

    @property
    def timestamped_zhir_preflow_clock_compatible(self) -> bool:
        """Whether every ZHIR can read its declared pre-flow duration exactly."""

        return not (
            self.zhir_event_indices_without_positive_preflow
            or self.zhir_event_indices_with_collapsed_preflow
            or self.zhir_event_indices_with_duration_mismatch
        )

    @property
    def clock_binding_ready(self) -> bool:
        """Whether the schedule passes both necessary clock conditions."""

        return bool(
            self.binary64_runtime_clock_compatible
            and self.timestamped_zhir_preflow_clock_compatible
        )


def diagnose_operator_event_runtime_clock(
    schedule: OperatorEventSchedule,
) -> OperatorEventRuntimeClockDiagnostic:
    """Inspect exact-to-binary64 clock compatibility without evolving a graph."""

    return OperatorEventRuntimeClockDiagnostic(schedule=schedule)
