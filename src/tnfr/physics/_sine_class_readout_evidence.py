"""Rebuild live full54 readout arithmetic before composing observations.

This narrow reader admits the declared complete law and branch evidence. Stored
derivative enclosures and Picard certificates remain execution premises: their
generation is not replayed or authenticated by arithmetic reconstruction.
"""

from dataclasses import dataclass
from fractions import Fraction as Q

from .._exact_time import exact_or_represented_real
from ..mathematics._comparison_flow import _exact
from ..mathematics._interval_taylor import MAX_ORDER
from ..mathematics._rational_interval import INTERVAL_METHOD, I
from ..mathematics._validated_taylor import (
    MAX_BOX_TAYLOR_DIMENSION,
    reconstruct_box_taylor_arithmetic,
)
from .relational_observations import _ordered
from .relational_sine_class_mediation import _EDGES
from .relational_sine_class_readout import _admit_class_readout_selectors


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _integer(value):
    _require(
        type(value) is int and value >= 0, "evidence count/index must be an integer"
    )
    return value


def _interval(value):
    _require(isinstance(value, I), "live evidence requires interval endpoints")
    lower, upper = _exact(value.lo), _exact(value.hi)
    _require(lower <= upper, "evidence interval is reversed")
    admitted = I(lower, upper)
    _require(admitted == value, "evidence endpoints must already lie on the grid")
    return admitted


def _evidence_rows(values, size, label):
    rows = _ordered(values, label, limit=size + 1)
    _require(len(rows) == size, f"{label} must contain exactly {size} entries")
    return rows


def _interval_vector(values, size, label):
    return tuple(map(_interval, _evidence_rows(values, size, label)))


def _box(values, size=54):
    return _interval_vector(values, size, "evidence state")


def _reconstruct_readout_step(step, initial_box, time, duration, *, order):
    """Admit and rebuild one retained smooth-domain source-box step.

    The caller owns law, source, event, schedule and attempt-budget association.
    Derivative, remainder and Picard generation remain execution premises;
    this helper checks their ordered representation and consumed arithmetic.
    """
    time, duration = _exact(time), _exact(duration)
    _require(time >= 0 and duration > 0, "step clock is outside its domain")
    _require(
        type(order) is int and 1 <= order <= MAX_ORDER,
        "step order is outside its domain",
    )
    initial = _ordered(
        initial_box, "expected initial box", limit=MAX_BOX_TAYLOR_DIMENSION + 1
    )
    size = len(initial)
    _require(
        1 <= size <= MAX_BOX_TAYLOR_DIMENSION,
        "step dimension is outside its domain",
    )
    initial = _box(initial, size)
    _require(
        _exact(step.time) == time
        and _exact(step.duration) == duration
        and _integer(step.order) == order
        and _box(step.initial_box, size) == initial,
        "step schedule or complete source differs",
    )
    _require(
        step.method == "direct_source_box_Picard_Taylor_dyadic128_v1"
        and _exact(step.picard_interior_margin) > 0
        and tuple(
            map(_exact, _evidence_rows(step.domain_lower_bounds, 1, "domain bounds"))
        )
        == (Q(1),),
        "strict smooth-domain step evidence differs",
    )
    tube = _box(step.tube, size)
    remainder = _box(step.local_remainder_bounds, size)
    series = tuple(
        _interval_vector(row, order + 1, "Taylor coefficient row")
        for row in _evidence_rows(step.series, size, "Taylor series")
    )
    increment, endpoint = reconstruct_box_taylor_arithmetic(
        initial, tube, series, remainder, duration, order=order
    )
    _require(
        _box(step.increment, size) == increment
        and _box(step.endpoint, size) == endpoint,
        "retained Taylor arithmetic differs",
    )
    return increment, endpoint


@dataclass(frozen=True)
class _ClassReadoutEvidence:
    complete: bool
    planned_step_count: int
    attempted_step_count: int
    completed_step_count: int
    endpoint_bounds: tuple[I, ...] | None
    suffix_increment_bounds: tuple[I, ...] | None
    mixed_bounds: I | None
    raw_mixed_bounds: I | None


def _reconstruct_class_readout(
    report,
    admitted_inputs,
    *,
    first_probe_node=4,
    second_probe_node=4,
    readout_node=22,
):
    """Check one fresh child against normalized inputs and rebuild its bands."""
    form, phase, a, b, delay, total, width, order, cap = admitted_inputs
    selectors = _admit_class_readout_selectors(
        first_probe_node, second_probe_node, readout_node
    )
    first_node, second_node, observer = selectors
    # Historical v1 records predate the optional selectors. Only the complete
    # absence of all three fields is compatible with the original defaults;
    # a present invalid value never falls through to an alias or default.
    missing = object()
    recorded_selectors = []
    for key in ("first_probe_node", "second_probe_node", "readout_node"):
        try:
            value = getattr(report, key)
        except AttributeError:
            value = missing
        except KeyError as error:
            # Older lazy record views use a mapping's missing-key convention.
            if error.args != (key,):
                raise
            value = missing
        recorded_selectors.append(value)
    if all(value is missing for value in recorded_selectors):
        _require(selectors == (4, 4, 22), "legacy evidence lacks selected nodes")
    else:
        _require(
            all(value is not missing for value in recorded_selectors),
            "partial selector evidence is unavailable",
        )
        _require(
            _admit_class_readout_selectors(*recorded_selectors) == selectors,
            "event or observation selectors differ",
        )
    nodes = tuple(range(27))
    edges = tuple(sorted(tuple(sorted(edge)) for edge in _EDGES))
    _require(
        tuple(map(_integer, report.geometry.nodes)) == nodes
        and tuple(tuple(map(_integer, edge)) for edge in report.geometry.edges)
        == edges,
        "readout support differs",
    )
    degrees = tuple(sum(node in edge for edge in edges) for node in nodes)
    _require(tuple(map(_integer, report.degrees)) == degrees, "readout degrees differ")
    model = report.reference_model
    for key, expected in (
        ("storage_scale", Q(1)),
        ("epi_weight", Q(1023, 1024)),
        ("phase_weight", Q(1, 1024)),
    ):
        _require(
            exact_or_represented_real(getattr(model, key), key) == expected,
            "complete held law differs",
        )
    _require(model.phase_domain == "regular", "phase law differs")
    _require(tuple(map(_exact, report.capacity)) == (Q(1),) * 27, "capacities differ")
    _require(report.clock == "tau=e*t; e=1023/1024", "clock differs")
    _require(
        report.state_order == tuple(f"{c}_{i}" for c in ("x", "theta") for i in nodes),
        "state order differs",
    )
    _require(
        _integer(report.donor_node) == first_node
        and _integer(report.receiver_node) == observer,
        "event or observation node differs",
    )
    source = form + phase
    _require(_box(report.source_box) == source, "source cover differs")
    _require(
        _interval_vector(report.initial_form_bounds, 27, "initial form") == form
        and _interval_vector(report.initial_phase_bounds, 27, "initial phase") == phase
        and _interval(report.initial_receiver_form_bounds) == form[observer],
        "source channel association differs",
    )
    for key, expected in (
        ("first_probe_amplitude", a),
        ("second_probe_amplitude", b),
        ("delay", delay),
        ("total_duration", total),
        ("time_step", width),
    ):
        _require(
            _exact(getattr(report, key)) == expected, "event or clock input differs"
        )
    _require(
        _integer(report.order) == order and _integer(report.max_steps) == cap,
        "work policy differs",
    )
    _require(report.arithmetic_method == INTERVAL_METHOD, "arithmetic method differs")
    _require(
        report.method == "shared_prefix_full54_source_box_Picard_Taylor_v1",
        "producer method differs",
    )
    _require(
        report.history_order == ("neither", "first_only", "second_only", "both")
        and tuple(tuple(map(_integer, row)) for row in report.history_segment_indices)
        == ((0, 2), (1, 3), (0, 4), (1, 5)),
        "history association differs",
    )
    coefficients = tuple(report.mixed_readout_coefficients)
    _require(
        all(type(value) is int for value in coefficients)
        and coefficients == (1, -1, -1, 1),
        "mixed observation differs",
    )
    _require(
        tuple(tuple(map(_exact, pair)) for pair in report.history_event_amplitudes)
        == ((Q(0), Q(0)), (a, Q(0)), (Q(0), b), (a, b)),
        "history events differ",
    )
    plan = (
        ("prefix_unprobed", None, Q(0), delay, Q(0), first_node),
        ("prefix_first", None, Q(0), delay, a, first_node),
        ("neither", 0, delay, total, Q(0), second_node),
        ("first_only", 1, delay, total, Q(0), second_node),
        ("second_only", 0, delay, total, b, second_node),
        ("both", 1, delay, total, b, second_node),
    )
    _require(len(report.segments) == 6, "complete branch inventory is required")
    final, increments = {}, {}
    success = failed_attempts = 0
    failure = None
    reasons = []
    for index, (segment, expected) in enumerate(zip(report.segments, plan)):
        label, parent, start, end, jump, jump_node = expected
        if segment.parent_segment_index is not None:
            _integer(segment.parent_segment_index)
        actual = (
            segment.label,
            segment.parent_segment_index,
            _exact(segment.start_time),
            _exact(segment.end_time),
            _exact(segment.form_jump),
        )
        _require(actual == expected[:5], "branch ancestry or events differ")
        if failure is not None:
            _require(
                segment.status == "not_attempted" and not segment.steps,
                "execution continued after first failure",
            )
            _require(
                all(
                    getattr(segment, key) is None
                    for key in (
                        "pre_event_box",
                        "initial_box",
                        "completed_time",
                        "completed_endpoint_box",
                        "completed_receiver_increment_bounds",
                        "final_state_bounds",
                        "failed_initial_box",
                        "failed_time",
                        "failed_tube",
                        "reason",
                    )
                ),
                "unattempted branch invented evidence",
            )
            continue
        if segment.initial_box is None:
            _require(
                start < end
                and success == cap
                and segment.status == "budget_exhausted"
                and segment.reason == "unique_step_budget_exhausted_before_event"
                and not segment.steps,
                "unsupported before-event budget failure",
            )
            _require(
                all(
                    getattr(segment, key) is None
                    for key in (
                        "pre_event_box",
                        "completed_time",
                        "completed_endpoint_box",
                        "completed_receiver_increment_bounds",
                        "final_state_bounds",
                        "failed_initial_box",
                        "failed_time",
                        "failed_tube",
                    )
                ),
                "budget stop applied an event",
            )
            failure = index
            reasons.append(f"{label}: {segment.reason}")
            continue
        before = source if parent is None else final[parent]
        _require(_box(segment.pre_event_box) == before, "full pre-event carry differs")
        state = tuple(
            value + jump if i == jump_node and jump else value
            for i, value in enumerate(before)
        )
        _require(_box(segment.initial_box) == state, "event changed another coordinate")
        time, change = start, I(0)
        for step in segment.steps:
            h = min(width, end - time)
            increment, endpoint = _reconstruct_readout_step(
                step, state, time, h, order=order
            )
            change += increment[observer]
            state, time = endpoint, time + h
            success += 1
        _require(
            _exact(segment.completed_time) == time
            and _box(segment.completed_endpoint_box) == state
            and _interval(segment.completed_receiver_increment_bounds) == change,
            "retained branch completion differs",
        )
        if segment.status == "admitted":
            _require(
                time == end and _box(segment.final_state_bounds) == state,
                "branch is incomplete",
            )
            _require(
                all(
                    getattr(segment, key) is None
                    for key in (
                        "failed_initial_box",
                        "failed_time",
                        "failed_tube",
                        "reason",
                    )
                ),
                "complete branch retains failure evidence",
            )
            final[index], increments[index] = state, change
        else:
            _require(
                segment.status in ("unavailable", "budget_exhausted")
                and time < end
                and segment.final_state_bounds is None
                and _exact(segment.failed_time) == time
                and _box(segment.failed_initial_box) == state
                and isinstance(segment.reason, str)
                and bool(segment.reason),
                "failed attempt association differs",
            )
            if segment.status == "unavailable":
                if segment.failed_tube is not None:
                    _box(segment.failed_tube)
                failed_attempts = 1
            else:
                _require(
                    success == cap
                    and segment.reason == "unique_step_budget_exhausted"
                    and segment.failed_tube is None,
                    "budget failure provenance differs",
                )
            failure = index
            reasons.append(f"{label}: {segment.reason}")
    attempted = success + failed_attempts
    _require(attempted <= cap, "attempt cap exceeded")
    _require(
        _integer(report.completed_step_count) == success
        and _integer(report.attempted_step_count) == attempted
        and _integer(report.completed_segment_count) == len(final),
        "evidence counts differ",
    )
    if report.failed_segment_index is not None:
        _integer(report.failed_segment_index)
    _require(
        report.failed_segment_index == failure, "failed branch association differs"
    )
    _require(
        tuple(map(_integer, report.unattempted_segment_indices))
        == (() if failure is None else tuple(range(failure + 1, 6))),
        "unattempted ancestry differs",
    )
    planned = 2 * -(-delay // width) + 4 * -(-(total - delay) // width)
    _require(_integer(report.planned_unique_step_count) == planned, "step plan differs")
    readings = tuple(
        (plan[i][0], final[i][observer]) for i in range(2, 6) if i in final
    )
    _require(
        tuple(
            (label, _interval(value))
            for label, value in report.completed_history_readout_bounds
        )
        == readings,
        "individual completed readings differ",
    )
    complete = len(final) == 6
    endpoints = tuple(value for _, value in readings) if complete else None
    suffix = tuple(increments[i] for i in range(2, 6)) if complete else None

    def mixed(values):
        return values[0] - values[1] - values[2] + values[3]

    primary, raw = (mixed(suffix), mixed(endpoints)) if complete else (None, None)
    for key, expected in (
        ("mixed_readout_bounds", primary),
        ("raw_endpoint_mixed_bounds", raw),
    ):
        actual = getattr(report, key)
        _require(
            (None if actual is None else _interval(actual)) == expected,
            "cached mixed bounds differ",
        )
    for key, expected in (
        ("endpoint_readout_bounds", endpoints),
        ("suffix_receiver_increment_bounds", suffix),
    ):
        actual = getattr(report, key)
        _require(
            (None if actual is None else _interval_vector(actual, 4, key)) == expected,
            "cached observation tuple differs",
        )
    _require(
        report.status == ("admitted" if complete else "unavailable")
        and _evidence_rows(report.unavailable_reasons, len(reasons), "failure reasons")
        == tuple(reasons),
        "cached availability differs",
    )
    return _ClassReadoutEvidence(
        complete, planned, attempted, success, endpoints, suffix, primary, raw
    )
