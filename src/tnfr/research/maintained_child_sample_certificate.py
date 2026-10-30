"""Associate acquired child samples with retained complete exact-flow bounds.

Preparation evaluates the retained cell's interior Taylor enclosure and must
therefore occur only after the acquisition protocol is frozen. It executes no
trajectory or event and supplies no runtime sample. Public consumers reload
primitive evidence; internal orchestration may share one freshly loaded source.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q

from ..mathematics._rational_interval import I
from ..mathematics._validated_taylor import reconstruct_box_taylor_partial
from ..operators._thol_config import resolve_thol_bifurcation_threshold
from ..physics._sample_reader_budget import (
    Binary64AccelerationReaderBudget,
    bound_binary64_acceleration_reader,
)
from ..physics.relational_observations import _ordered
from .artifact_io import exact_record
from .maintained_child_history import CHILD_INDEX
from .maintained_child_joint_source import (
    CELL_END,
    CELL_INDEX,
    CELL_START,
    ChildJointAdmission,
    load_child_joint_window,
)
from .maintained_child_oz_timing import (
    PRESSURE_ERROR_ALLOWANCE,
    SAMPLE_ERROR_ALLOWANCE,
    SAMPLE_STEP,
)
from .maintained_parent_thol import _represented
from .maintained_sha_retention_evidence import _box, _interval, _require
from .maintained_temporal_window import bound_node_history_box


@dataclass(frozen=True)
class ChildSampleCertificate:
    """Fresh source and three exact-flow enclosures, never acquired samples.

    All boxes retain the 21 forms followed by their 21 declared phase lifts.
    The source's recorded jets keep their frozen-producer provenance. A public
    assessment reloads evidence; internal assessments rebuild these fields
    from the freshly admitted source rather than trusting replaceable outputs.
    """

    source: ChildJointAdmission
    boxes: tuple[tuple[I, ...], ...]
    sample_times: tuple[Q, Q, Q]


@dataclass(frozen=True)
class ChildSampleBoundary:
    """Errors at one associated instant, without claiming an acquired path.

    A represented point need not lie inside the exact-flow enclosure. Its
    maximum distance to either endpoint bounds its error relative to the true
    solution. Pressure error against that solution differs from the numerical
    pressure evaluation defect at the represented point; both are reported.
    """

    index: int
    sample_time: Q
    state: tuple[Q, ...]
    pressures: tuple[Q, ...]
    state_error_bounds: tuple[Q, ...]
    max_state_error_bound: Q
    child_error_bound: Q
    true_pressure_bounds: tuple[I, ...]
    point_pressure_bounds: tuple[I, ...]
    pressure_error_bounds: tuple[Q, ...]
    max_pressure_error_bound: Q
    point_pressure_error_bounds: tuple[Q, ...]
    exact_principal_margins: tuple[Q, ...]
    represented_principal_margins: tuple[Q, ...]
    checks: dict[str, bool]
    passed: bool


@dataclass(frozen=True)
class ChildSampleAssessment:
    """Rebuilt value-error and reader bounds, separate from runtime provenance.

    Passing requires independently acquired records from one full-state path,
    the pressure-only event/refresh identity and live clock/history association.
    This arithmetic consumer establishes neither those execution facts nor
    U4b, a completed word or full THOL admission.
    """

    sample_times: tuple[Q, Q, Q]
    states: tuple[tuple[Q, ...], ...]
    pressures: tuple[tuple[Q, ...], ...]
    boundaries: tuple[ChildSampleBoundary, ...]
    state_error_bounds: tuple[tuple[Q, ...], ...]
    max_state_error_bounds: tuple[Q, ...]
    child_error_bounds: tuple[Q, ...]
    true_pressure_bounds: tuple[tuple[I, ...], ...]
    point_pressure_bounds: tuple[tuple[I, ...], ...]
    pressure_error_bounds: tuple[tuple[Q, ...], ...]
    max_pressure_error_bounds: tuple[Q, ...]
    exact_principal_margins: tuple[tuple[Q, ...], ...]
    represented_principal_margins: tuple[tuple[Q, ...], ...]
    reader_budget: Binary64AccelerationReaderBudget
    threshold: Q
    checks: dict[str, bool]
    passed: bool


def _prepare_child_sample_certificate(source):
    """Evaluate only an already admitted source's fixed cell, after freezing.

    This private composition accepts the freshly loaded reader result within
    one orchestration. It is not an admission endpoint for caller-built reports.
    Stored source-box jets/remainders retain the upstream provenance boundary;
    they are not regenerated, nor used to synthesize a numerical state.
    """
    step = source.step
    _require(
        type(step.index) is int
        and step.index == CELL_INDEX
        and step.time == CELL_START
        and step.duration == 2 * SAMPLE_STEP
        and step.time + step.duration == CELL_END,
        "child sample cell chronology differs",
    )
    record = source.retention.records[CELL_INDEX]
    raw = record["step"]
    _require(
        type(record["index"]) is int
        and record["index"] == CELL_INDEX
        and exact_record(record["time"]) == CELL_START
        and exact_record(raw["time"]) == CELL_START
        and exact_record(raw["duration"]) == step.duration,
        "child sample Taylor record chronology differs",
    )
    initial, tube, endpoint = (
        _box(raw[name]) for name in ("initial_box", "tube", "endpoint")
    )
    _require(
        initial == step.initial and tube == step.tube and endpoint == step.endpoint,
        "child sample Taylor record differs from the admitted cell",
    )
    series = tuple(tuple(_interval(value) for value in row) for row in raw["series"])
    remainder = _box(raw["local_remainder_bounds"])
    _, middle = reconstruct_box_taylor_partial(
        initial,
        tube,
        series,
        remainder,
        step.duration,
        SAMPLE_STEP,
        order=raw["order"],
    )
    source_time = _represented(
        source.retention.admitted.graph.graph["_t"], "source structural time"
    )
    times = tuple(
        source_time + elapsed
        for elapsed in (CELL_START, CELL_START + SAMPLE_STEP, CELL_END)
    )
    for index, time in enumerate(times):
        _represented(time, f"sample time[{index}]")
    return ChildSampleCertificate(source, (initial, middle, endpoint), times)


def prepare_child_sample_certificate(evidence_dir, *, repository_root):
    """Reload once and evaluate the fixed midpoint after protocol freezing.

    Before freezing, use ``load_child_joint_window`` alone for source admission.
    This function does not itself freeze, authorize or authenticate a protocol.
    """
    source = load_child_joint_window(evidence_dir, repository_root=repository_root)
    return _prepare_child_sample_certificate(source)


def _represented_row(values, size, label):
    row = _ordered(values, label, limit=size + 1)
    if len(row) != size:
        raise ValueError(f"{label} requires exactly {size} ordered binary64 values")
    return tuple(_represented(value, f"{label}[{i}]") for i, value in enumerate(row))


def _field(source, state):
    prediction = source.retention.prediction
    return bound_node_history_box(
        epi=state[:21],
        phases=state[21:],
        capacities=prediction.capacities,
        edges=prediction.edges,
        conductances=prediction.conductances,
        shifts=prediction.shifts,
        target_index=CHILD_INDEX,
    )


def _farther_error(value, bound):
    return max(abs(value - bound.lo), abs(value - bound.hi))


def _assess_boundary(prepared, *, index, state, pressures, sample_time):
    if type(index) is not int or not 0 <= index < 3:
        raise ValueError("child boundary index must be an ordinary integer in 0..2")
    time = _represented(sample_time, "acquired sample time")
    if time != prepared.sample_times[index]:
        raise ValueError("acquired sample time differs from the fixed exact clock")
    values = _represented_row(state, 42, "acquired state")
    pressure_values = _represented_row(pressures, 21, "acquired pressures")
    exact_box = prepared.boxes[index]
    exact_field = _field(prepared.source, exact_box)
    point_field = _field(prepared.source, tuple(I(value) for value in values))
    state_errors = tuple(
        _farther_error(value, bound)
        for value, bound in zip(values, exact_box, strict=True)
    )
    pressure_errors = tuple(
        _farther_error(value, bound)
        for value, bound in zip(
            pressure_values, exact_field.state.pressure, strict=True
        )
    )
    point_pressure_errors = tuple(
        _farther_error(value, bound)
        for value, bound in zip(
            pressure_values, point_field.state.pressure, strict=True
        )
    )
    checks = {
        "full_state_error_within_allowance": max(state_errors)
        <= SAMPLE_ERROR_ALLOWANCE,
        "exact_flow_pressure_error_within_allowance": max(pressure_errors)
        <= PRESSURE_ERROR_ALLOWANCE,
        "exact_flow_original_branches_retained": min(exact_field.principal_margins) > 0,
        "represented_original_branches_retained": min(point_field.principal_margins)
        > 0,
    }
    return ChildSampleBoundary(
        index=index,
        sample_time=time,
        state=values,
        pressures=pressure_values,
        state_error_bounds=state_errors,
        max_state_error_bound=max(state_errors),
        child_error_bound=state_errors[CHILD_INDEX],
        true_pressure_bounds=exact_field.state.pressure,
        point_pressure_bounds=point_field.state.pressure,
        pressure_error_bounds=pressure_errors,
        max_pressure_error_bound=max(pressure_errors),
        point_pressure_error_bounds=point_pressure_errors,
        exact_principal_margins=exact_field.principal_margins,
        represented_principal_margins=point_field.principal_margins,
        checks=checks,
        passed=all(checks.values()),
    )


def _assess_child_boundary(certificate, *, index, state, pressures, sample_time):
    """Pre-event guard using one fresh orchestration, not a cached verdict.

    Rebuild the certificate's replaceable box/time fields from its freshly
    admitted source. A point-pressure rounding budget and runtime law/state
    association remain the constitutive writer's separate obligations.
    """
    prepared = _prepare_child_sample_certificate(certificate.source)
    return _assess_boundary(
        prepared, index=index, state=state, pressures=pressures, sample_time=sample_time
    )


def _assess_child_samples(certificate, *, states, pressures, sample_times):
    """Assess all boundaries and the reader within one fresh orchestration."""
    prepared = _prepare_child_sample_certificate(certificate.source)
    rows = _ordered(states, "acquired states", limit=4)
    pressure_rows = _ordered(pressures, "acquired pressures", limit=4)
    times = _ordered(sample_times, "sample times", limit=4)
    if len(rows) != 3 or len(pressure_rows) != 3 or len(times) != 3:
        raise ValueError("child history requires exactly three full-state boundaries")
    boundaries = tuple(
        _assess_boundary(
            prepared, index=index, state=state, pressures=pressure, sample_time=time
        )
        for index, (state, pressure, time) in enumerate(
            zip(rows, pressure_rows, times, strict=True)
        )
    )
    whole = _field(prepared.source, prepared.source.step.tube)
    child_errors = tuple(item.child_error_bound for item in boundaries)
    reader = bound_binary64_acceleration_reader(
        form_bounds=whole.form,
        rate_bounds=whole.form_rate,
        acceleration_bounds=whole.form_acceleration,
        sample_step=SAMPLE_STEP,
        observation_time=prepared.sample_times[0],
        sample_error_bounds=child_errors,
    )
    threshold = _represented(
        resolve_thol_bifurcation_threshold(
            prepared.source.retention.admitted.graph.graph
        ),
        "THOL threshold",
    )
    checks = {
        "all_full_state_boundaries_admitted": all(item.passed for item in boundaries),
        "child_sample_errors_within_allowance": max(child_errors)
        <= SAMPLE_ERROR_ALLOWANCE,
        "reader_arithmetic_exact": reader.exact,
        "reader_acceleration_above_threshold": reader.exact
        and reader.represented_acceleration_bounds.lo > threshold,
    }
    return ChildSampleAssessment(
        sample_times=prepared.sample_times,
        states=tuple(item.state for item in boundaries),
        pressures=tuple(item.pressures for item in boundaries),
        boundaries=boundaries,
        state_error_bounds=tuple(item.state_error_bounds for item in boundaries),
        max_state_error_bounds=tuple(item.max_state_error_bound for item in boundaries),
        child_error_bounds=child_errors,
        true_pressure_bounds=tuple(item.true_pressure_bounds for item in boundaries),
        point_pressure_bounds=tuple(item.point_pressure_bounds for item in boundaries),
        pressure_error_bounds=tuple(item.pressure_error_bounds for item in boundaries),
        max_pressure_error_bounds=tuple(
            item.max_pressure_error_bound for item in boundaries
        ),
        exact_principal_margins=tuple(
            item.exact_principal_margins for item in boundaries
        ),
        represented_principal_margins=tuple(
            item.represented_principal_margins for item in boundaries
        ),
        reader_budget=reader,
        threshold=threshold,
        checks=checks,
        passed=all(checks.values()),
    )


def assess_child_samples(
    evidence_dir, *, repository_root, states, pressures, sample_times
):
    """Reload primitive source evidence before assessing supplied observations."""
    certificate = prepare_child_sample_certificate(
        evidence_dir, repository_root=repository_root
    )
    return _assess_child_samples(
        certificate, states=states, pressures=pressures, sample_times=sample_times
    )
