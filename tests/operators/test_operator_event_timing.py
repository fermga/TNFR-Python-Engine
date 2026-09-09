"""Canonical timing contract for finite operator-event schedules."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from fractions import Fraction

import pytest

from tnfr.config.operator_names import CANONICAL_OPERATOR_NAMES
from tnfr.operators.event_timing import (
    OperatorEventSchedule,
    build_operator_event_schedule,
    diagnose_operator_event_runtime_clock,
)
from tnfr.operators.registry import get_operator_class
from tnfr.types import Glyph


def test_schedule_alternates_m_events_with_m_plus_one_flow_intervals() -> None:
    schedule = build_operator_event_schedule(
        ["emission", "silence"],
        cycles=2,
        start_time=1.0,
        flow_durations=[0.5, 0.0, 0.25, 0.75, 1.0],
    )

    assert schedule.event_count == 4
    assert len(schedule.intervals) == 5
    assert schedule.operator_names == ("emission", "silence")
    assert schedule.flow_durations == (0.5, 0.0, 0.25, 0.75, 1.0)
    assert schedule.total_flow_duration == pytest.approx(2.5)
    assert schedule.end_time == pytest.approx(3.5)
    assert schedule.exact_start_time == Fraction(1)
    assert schedule.exact_total_flow_duration == Fraction(5, 2)
    assert schedule.exact_end_time == Fraction(7, 2)
    assert [event.operator_name for event in schedule.events] == [
        "emission",
        "silence",
        "emission",
        "silence",
    ]
    assert [event.glyph for event in schedule.events] == [
        Glyph.AL,
        Glyph.SHA,
        Glyph.AL,
        Glyph.SHA,
    ]
    assert [event.event_time for event in schedule.events] == pytest.approx(
        [1.5, 1.5, 1.75, 2.5]
    )
    assert [
        (event.event_index, event.cycle_index, event.word_position)
        for event in schedule.events
    ] == [(0, 0, 0), (1, 0, 1), (2, 1, 0), (3, 1, 1)]
    assert all(item.time_basis == "physical_time" for item in schedule.intervals)
    assert all(item.time_basis == "physical_time" for item in schedule.events)
    assert schedule.time_basis == "physical_time"
    assert schedule.duration_and_offsets_are_authoritative
    assert not schedule.event_timestamps_feed_epi_time_history
    assert schedule.event_history_channel == "hybrid_event_log"
    assert schedule.coincident_event_order == "event_index"
    assert [event.event_offset for event in schedule.events] == [
        Fraction(1, 2),
        Fraction(1, 2),
        Fraction(3, 4),
        Fraction(3, 2),
    ]


def test_every_public_operator_name_resolves_its_registry_glyph() -> None:
    names = tuple(sorted(CANONICAL_OPERATOR_NAMES))
    schedule = build_operator_event_schedule(
        names,
        start_time=0.0,
        flow_durations=[0.0] * (len(names) + 1),
    )

    assert tuple(event.operator_name for event in schedule.events) == names
    assert tuple(event.glyph for event in schedule.events) == tuple(
        get_operator_class(name).glyph for name in names
    )


@pytest.mark.parametrize("alias", ["AL", "Emission", Glyph.AL])
def test_internal_glyphs_and_display_aliases_are_not_public_names(alias) -> None:
    with pytest.raises((TypeError, ValueError), match="public operator name"):
        build_operator_event_schedule(
            [alias], start_time=0.0, flow_durations=[0.0, 0.0]
        )


@pytest.mark.parametrize("cycles", [True, False, 0, -1, 1.0, float("inf"), "1"])
def test_cycles_must_be_a_positive_non_boolean_integer(cycles) -> None:
    with pytest.raises((TypeError, ValueError), match="positive integer"):
        build_operator_event_schedule(
            ["emission"],
            cycles=cycles,
            start_time=0.0,
            flow_durations=[0.0, 0.0],
        )


@pytest.mark.parametrize(
    "start_time", [True, False, "0", float("nan"), float("inf"), -float("inf")]
)
def test_start_time_must_be_a_finite_non_boolean_real(start_time) -> None:
    with pytest.raises((TypeError, ValueError), match="start_time"):
        build_operator_event_schedule(
            ["emission"], start_time=start_time, flow_durations=[0.0, 0.0]
        )


@pytest.mark.parametrize("durations", ["0,0", 0.0, None])
def test_duration_collection_must_be_an_iterable_not_a_scalar(durations) -> None:
    with pytest.raises(TypeError, match="flow_durations"):
        build_operator_event_schedule(
            ["emission"], start_time=0.0, flow_durations=durations
        )


@pytest.mark.parametrize("durations", [[], [0.0], [0.0, 0.0, 0.0]])
def test_schedule_rejects_the_wrong_number_of_flow_intervals(durations) -> None:
    with pytest.raises(ValueError, match="exactly one more"):
        build_operator_event_schedule(
            ["emission"], start_time=0.0, flow_durations=durations
        )


@pytest.mark.parametrize(
    "bad_duration",
    [True, False, "0", -0.1, float("nan"), float("inf"), -float("inf")],
)
def test_flow_durations_reject_boolean_nonreal_negative_and_nonfinite_values(
    bad_duration,
) -> None:
    with pytest.raises((TypeError, ValueError), match=r"flow_durations\[0\]"):
        build_operator_event_schedule(
            ["emission"],
            start_time=0.0,
            flow_durations=[bad_duration, 0.0],
        )


def test_schedule_rejects_unknown_operator_before_building_timeline() -> None:
    with pytest.raises(ValueError, match="unknown public operator name"):
        build_operator_event_schedule(
            ["not_an_operator"], start_time=0.0, flow_durations=[0.0, 0.0]
        )


def test_total_duration_overflow_is_rejected() -> None:
    with pytest.raises(ValueError, match="total flow duration"):
        build_operator_event_schedule(
            ["emission"],
            start_time=0.0,
            flow_durations=[1e308, 1e308],
        )


def test_exact_duration_survives_a_stationary_display_timestamp() -> None:
    schedule = build_operator_event_schedule(
        ["emission"],
        start_time=float(2**53),
        flow_durations=[0.5, 0.0],
    )

    interval = schedule.intervals[0]
    assert interval.start_time == interval.end_time
    assert interval.exact_duration == Fraction(1, 2)
    assert interval.exact_end_time - interval.exact_start_time == Fraction(1, 2)
    assert interval.end_offset - interval.start_offset == Fraction(1, 2)
    assert interval.duration_is_authoritative
    assert not interval.feeds_epi_time_history
    assert interval.timestamp_role == "binary64_representation_only"


def test_absolute_timestamp_subtraction_does_not_define_duration() -> None:
    schedule = build_operator_event_schedule(
        ["emission"],
        start_time=float(2**52),
        flow_durations=[0.6, 0.0],
    )

    interval = schedule.intervals[0]
    assert interval.end_time - interval.start_time == 1.0
    assert interval.duration == 0.6
    assert interval.exact_duration == Fraction.from_float(0.6)
    assert schedule.events[0].event_offset == interval.end_offset
    assert schedule.events[0].history_channel == "hybrid_event_log"
    assert not schedule.events[0].feeds_epi_time_history


def test_positive_source_duration_cannot_underflow_silently_to_zero() -> None:
    with pytest.raises(ValueError, match="underflows.*zero"):
        build_operator_event_schedule(
            ["emission"],
            start_time=0.0,
            flow_durations=[Fraction(1, 10**1000), 0.0],
        )


def test_zero_duration_intervals_and_an_empty_finite_word_are_supported() -> None:
    coincident = build_operator_event_schedule(
        ["emission", "silence"],
        start_time=-2.0,
        flow_durations=(0.0, 0.0, 0.0),
    )
    pure_flow = build_operator_event_schedule(
        (), start_time=1.0, flow_durations=(0.25,)
    )

    assert [event.event_time for event in coincident.events] == [-2.0, -2.0]
    assert [event.event_index for event in coincident.events] == [0, 1]
    assert all(
        event.history_channel == "hybrid_event_log"
        for event in coincident.events
    )
    assert all(not event.feeds_epi_time_history for event in coincident.events)
    assert coincident.end_time == -2.0
    assert pure_flow.events == ()
    assert pure_flow.event_count == 0
    assert len(pure_flow.intervals) == 1
    assert pure_flow.end_time == pytest.approx(1.25)


def test_schedule_is_deeply_immutable_and_deterministic_for_generators() -> None:
    def build() -> OperatorEventSchedule:
        return build_operator_event_schedule(
            (name for name in ("emission", "silence")),
            cycles=2,
            start_time=Fraction(1, 2),
            flow_durations=(value for value in (0.25, 0.0, 0.5, 0.25, 0.0)),
        )

    first = build()
    second = build()

    assert first == second
    assert hash(first) == hash(second)
    assert isinstance(first.operator_names, tuple)
    assert isinstance(first.flow_durations, tuple)
    assert isinstance(first.intervals, tuple)
    assert isinstance(first.events, tuple)
    with pytest.raises(FrozenInstanceError):
        first.start_time = 3.0  # type: ignore[misc]


def test_structural_interval_rejects_inconsistent_derived_endpoint() -> None:
    interval = build_operator_event_schedule(
        ["emission"], start_time=1.0, flow_durations=[0.5, 0.0]
    ).intervals[0]

    with pytest.raises(ValueError, match="represent exact_end_time"):
        replace(interval, end_time=1.25)


def test_event_rejects_a_glyph_that_does_not_match_its_public_name() -> None:
    event = build_operator_event_schedule(
        ["emission"], start_time=0.0, flow_durations=[0.0, 0.0]
    ).events[0]

    with pytest.raises(ValueError, match="event glyph"):
        replace(event, glyph=Glyph.EN)


def test_schedule_rebuild_rejects_derived_field_tampering() -> None:
    schedule = build_operator_event_schedule(
        ["emission"], start_time=0.0, flow_durations=[0.25, 0.5]
    )
    shifted_index = replace(schedule.events[0], event_index=1)

    with pytest.raises(ValueError, match="events do not match"):
        replace(schedule, events=(shifted_index,))
    with pytest.raises(ValueError, match="end_time does not match"):
        replace(schedule, end_time=1.0)
    with pytest.raises(ValueError, match="intervals do not match"):
        replace(schedule, flow_durations=(0.5, 0.25))


def test_exact_prefix_sum_unifies_total_duration_and_final_time() -> None:
    schedule = build_operator_event_schedule(
        ["emission"] * 10,
        start_time=0.0,
        flow_durations=[0.1] * 11,
    )
    exact_tenth = Fraction.from_float(0.1)

    assert schedule.exact_total_flow_duration == 11 * exact_tenth
    assert schedule.exact_end_time == schedule.exact_total_flow_duration
    assert schedule.end_time == float(schedule.exact_end_time)
    assert schedule.total_flow_duration == float(
        schedule.exact_total_flow_duration
    )
    assert schedule.intervals[-1].end_offset == 11 * exact_tenth


def test_signed_zero_is_normalized_across_the_schedule() -> None:
    schedule = build_operator_event_schedule(
        ["emission"],
        start_time=-0.0,
        flow_durations=[-0.0, 0.0],
    )

    floats = (
        schedule.start_time,
        schedule.end_time,
        schedule.total_flow_duration,
        *schedule.flow_durations,
        *(interval.start_time for interval in schedule.intervals),
        *(interval.end_time for interval in schedule.intervals),
        *(event.event_time for event in schedule.events),
    )
    assert all(value.hex() == "0x0.0p+0" for value in floats)


@pytest.mark.parametrize(
    "field, replacement, message",
    [
        ("operator_names", ["emission"], "immutable tuple"),
        ("flow_durations", [0.0, 0.0], "immutable tuple"),
        ("flow_durations", (0, 0), "binary64 floats"),
        ("intervals", [], "immutable tuples"),
        ("events", [], "immutable tuples"),
        ("start_time", 0, "binary64 float"),
        ("end_time", 0, "binary64 float"),
        ("total_flow_duration", 0, "binary64 float"),
    ],
)
def test_schedule_rejects_type_coercing_replacements(
    field, replacement, message
) -> None:
    schedule = build_operator_event_schedule(
        ["emission"], start_time=0.0, flow_durations=[0.0, 0.0]
    )

    with pytest.raises(TypeError, match=message):
        replace(schedule, **{field: replacement})

def test_runtime_clock_diagnostic_accepts_representable_zhir_preflow() -> None:
    schedule = build_operator_event_schedule(
        ["emission", "mutation", "silence"],
        start_time=1.0,
        flow_durations=[0.25, 0.5, 0.125, 0.25],
    )

    diagnostic = diagnose_operator_event_runtime_clock(schedule)

    assert diagnostic.collapsed_positive_interval_indices == ()
    assert diagnostic.nonadditive_positive_interval_indices == ()
    assert diagnostic.zhir_event_indices_without_positive_preflow == ()
    assert diagnostic.zhir_event_indices_with_collapsed_preflow == ()
    assert diagnostic.zhir_event_indices_with_duration_mismatch == ()
    assert diagnostic.binary64_runtime_clock_compatible
    assert diagnostic.timestamped_zhir_preflow_clock_compatible
    assert diagnostic.clock_binding_ready
    assert "solver accuracy" in diagnostic.scope


def test_runtime_clock_diagnostic_separates_zero_zhir_preflow() -> None:
    schedule = build_operator_event_schedule(
        ["emission", "mutation", "silence"],
        start_time=0.0,
        flow_durations=[0.25, 0.0, 0.125, 0.25],
    )

    diagnostic = diagnose_operator_event_runtime_clock(schedule)

    assert diagnostic.binary64_runtime_clock_compatible
    assert diagnostic.zhir_event_indices_without_positive_preflow == (1,)
    assert diagnostic.zhir_event_indices_with_collapsed_preflow == ()
    assert not diagnostic.timestamped_zhir_preflow_clock_compatible
    assert not diagnostic.clock_binding_ready


def test_runtime_clock_diagnostic_detects_nonadditive_prefix_rounding() -> None:
    schedule = build_operator_event_schedule(
        ("emission",) * 6,
        start_time=0.0,
        flow_durations=(0.1,) * 7,
    )

    diagnostic = diagnose_operator_event_runtime_clock(schedule)

    assert diagnostic.collapsed_positive_interval_indices == ()
    assert diagnostic.nonadditive_positive_interval_indices == (5,)
    assert not diagnostic.binary64_runtime_clock_compatible
    assert not diagnostic.clock_binding_ready


def test_runtime_clock_diagnostic_detects_rounded_positive_preflow() -> None:
    schedule = build_operator_event_schedule(
        ["mutation"],
        start_time=float(2**53),
        flow_durations=[0.5, 0.0],
    )

    diagnostic = diagnose_operator_event_runtime_clock(schedule)

    assert diagnostic.collapsed_positive_interval_indices == (0,)
    assert diagnostic.zhir_event_indices_without_positive_preflow == ()
    assert diagnostic.zhir_event_indices_with_collapsed_preflow == (0,)
    assert not diagnostic.binary64_runtime_clock_compatible
    assert not diagnostic.timestamped_zhir_preflow_clock_compatible
    assert not diagnostic.clock_binding_ready


def test_runtime_clock_diagnostic_rejects_zhir_timestamp_duration_mismatch() -> None:
    schedule = build_operator_event_schedule(
        ["mutation"],
        start_time=float(2**52),
        flow_durations=[0.6, 0.0],
    )

    interval = schedule.intervals[0]
    diagnostic = diagnose_operator_event_runtime_clock(schedule)

    assert interval.start_time + interval.duration == interval.end_time
    assert interval.end_time - interval.start_time == 1.0
    assert interval.duration == 0.6
    assert diagnostic.binary64_runtime_clock_compatible
    assert diagnostic.zhir_event_indices_with_duration_mismatch == (0,)
    assert not diagnostic.timestamped_zhir_preflow_clock_compatible
    assert not diagnostic.clock_binding_ready


def test_runtime_clock_diagnostic_rejects_non_schedule() -> None:
    with pytest.raises(TypeError, match="OperatorEventSchedule"):
        diagnose_operator_event_runtime_clock(object())  # type: ignore[arg-type]
