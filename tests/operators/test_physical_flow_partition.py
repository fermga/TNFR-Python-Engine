"""Exact timing contract for explicit physical flow partitions."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from fractions import Fraction
import math

import pytest

from tnfr.operators.event_timing import (
    PhysicalFlowPartition,
    StructuralFlowInterval,
    build_operator_event_schedule,
    build_physical_flow_partition,
)


def _parent_interval(
    *, start_time: float = 0.0, duration: float = 1.0
) -> StructuralFlowInterval:
    return build_operator_event_schedule(
        (),
        start_time=start_time,
        flow_durations=(duration,),
    ).intervals[0]


def test_partition_builds_local_canonical_segments_with_exact_boundaries() -> None:
    parent = _parent_interval(start_time=1.0, duration=1.0)

    partition = build_physical_flow_partition(
        parent,
        (value for value in (0.25, 0.25, 0.5)),
    )

    assert type(partition) is PhysicalFlowPartition
    assert partition.parent_interval is parent
    assert partition.segment_count == 3
    assert partition.segment_durations == (0.25, 0.25, 0.5)
    assert all(type(item) is StructuralFlowInterval for item in partition.segments)
    assert tuple(item.index for item in partition.segments) == (0, 1, 2)
    assert partition.boundary_times == (1.0, 1.25, 1.5, 2.0)
    assert partition.exact_boundary_times == (
        Fraction(1),
        Fraction(5, 4),
        Fraction(3, 2),
        Fraction(2),
    )
    assert partition.time_basis == "physical_time"
    assert partition.boundary_role == "explicit_physical_pressure_refresh"
    assert not partition.numerical_substeps_are_physical_boundaries
    assert partition.segments[0].start_offset == parent.start_offset
    assert partition.segments[-1].end_offset == parent.end_offset
    assert partition.segments[0].exact_start_time == parent.exact_start_time
    assert partition.segments[-1].exact_end_time == parent.exact_end_time
    assert all(
        segment.start_time + segment.duration == segment.end_time
        and segment.end_time - segment.start_time == segment.duration
        for segment in partition.segments
    )


def test_partition_preserves_parent_global_offsets_and_float_continuity() -> None:
    schedule = build_operator_event_schedule(
        ["emission"],
        start_time=10.0,
        flow_durations=(0.5, 1.0),
    )
    parent = schedule.intervals[1]

    partition = build_physical_flow_partition(parent, (0.25, 0.75))

    first, second = partition.segments
    assert first.start_offset == Fraction(1, 2)
    assert first.end_offset == second.start_offset == Fraction(3, 4)
    assert second.end_offset == Fraction(3, 2)
    assert first.end_time == second.start_time == 10.75
    assert first.exact_end_time == second.exact_start_time
    assert first.start_time == parent.start_time
    assert second.end_time == parent.end_time


def test_partition_is_frozen_hashable_and_deterministic() -> None:
    parent = _parent_interval()
    first = build_physical_flow_partition(parent, (0.25, 0.75))
    second = build_physical_flow_partition(parent, (0.25, 0.75))

    assert first == second
    assert hash(first) == hash(second)
    with pytest.raises(FrozenInstanceError):
        first.segment_durations = (0.5, 0.5)  # type: ignore[misc]


@pytest.mark.parametrize("durations", [(), (1.0,)])
def test_partition_requires_at_least_two_segments(durations) -> None:
    with pytest.raises(ValueError, match="at least two positive segments"):
        build_physical_flow_partition(_parent_interval(), durations)


@pytest.mark.parametrize(
    "durations",
    [
        (0.0, 1.0),
        (-0.0, 1.0),
        (0.5, 0.0, 0.5),
        (-0.25, 1.25),
    ],
)
def test_partition_requires_strictly_positive_segments(durations) -> None:
    with pytest.raises((TypeError, ValueError), match="segment_durations"):
        build_physical_flow_partition(_parent_interval(), durations)


@pytest.mark.parametrize(
    "durations",
    ["0.5,0.5", 1.0, None, (True, 0.5), (float("nan"), 0.5)],
)
def test_partition_rejects_non_replayable_duration_inputs(durations) -> None:
    with pytest.raises((TypeError, ValueError), match="segment_durations"):
        build_physical_flow_partition(_parent_interval(), durations)


def test_partition_requires_the_exact_represented_duration_sum() -> None:
    below_half = math.nextafter(0.5, 0.0)

    with pytest.raises(ValueError, match="exact segment duration sum"):
        build_physical_flow_partition(
            _parent_interval(),
            (below_half, 0.5),
        )


def test_partition_rejects_a_parent_with_collapsed_binary64_time() -> None:
    parent = _parent_interval(start_time=float(2**53), duration=0.5)

    with pytest.raises(ValueError, match="parent_interval collapses"):
        build_physical_flow_partition(parent, (0.25, 0.25))


def test_partition_rejects_parent_timestamp_subtraction_mismatch() -> None:
    parent = _parent_interval(start_time=float(2**52), duration=0.6)

    with pytest.raises(ValueError, match="subtraction does not recover"):
        build_physical_flow_partition(parent, (0.3, 0.3))


def test_partition_rejects_a_collapsed_child_boundary() -> None:
    parent = _parent_interval(start_time=float(2**52), duration=2.0)

    with pytest.raises(ValueError, match="physical segment 0 collapses"):
        build_physical_flow_partition(parent, (0.5, 1.5))


def test_partition_rejects_nonadditive_cumulative_binary64_boundary() -> None:
    durations = (0.1, 0.4, 0.1, 0.2)
    assert sum((Fraction.from_float(item) for item in durations), Fraction()) == (
        Fraction.from_float(0.8)
    )

    with pytest.raises(ValueError, match="physical segment 2 is nonadditive"):
        build_physical_flow_partition(
            _parent_interval(duration=0.8),
            durations,
        )


def test_partition_rejects_child_timestamp_subtraction_mismatch() -> None:
    parent = _parent_interval(duration=0.4)

    with pytest.raises(ValueError, match="subtraction does not recover"):
        build_physical_flow_partition(parent, (0.1, 0.2, 0.1))


def test_partition_post_init_rebuilds_and_rejects_derived_tampering() -> None:
    partition = build_physical_flow_partition(
        _parent_interval(),
        (0.25, 0.75),
    )
    shifted = replace(partition.segments[0], index=1)

    with pytest.raises(ValueError, match="canonical timeline"):
        replace(partition, segments=(shifted, partition.segments[1]))
    with pytest.raises(TypeError, match="immutable tuple"):
        replace(partition, segment_durations=[0.25, 0.75])
    with pytest.raises(TypeError, match="binary64 floats"):
        replace(partition, segment_durations=(Fraction(1, 4), 0.75))
    with pytest.raises(TypeError, match="immutable tuple"):
        replace(partition, segments=list(partition.segments))


def test_partition_rejects_noncanonical_parent_type() -> None:
    with pytest.raises(TypeError, match="StructuralFlowInterval"):
        build_physical_flow_partition(object(), (0.5, 0.5))  # type: ignore[arg-type]
