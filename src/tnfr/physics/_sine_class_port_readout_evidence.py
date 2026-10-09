"""Re-admit paired port-readout evidence and rebuild its consumed endpoints.

Retained derivative coefficients, remainders and strict Picard certificates
remain execution premises. This arithmetic reader neither regenerates them
nor authenticates execution, preparation or model selection.
"""

from dataclasses import dataclass
from fractions import Fraction as Q

from .._exact_time import exact_or_represented_real
from ..mathematics._comparison_flow import _exact
from ..mathematics._rational_interval import INTERVAL_METHOD, I
from ..mathematics._validated_taylor import reconstruct_box_taylor_arithmetic
from ._sine_class_readout_evidence import _box, _integer, _interval, _require
from .relational_sine_class_mediation import _EDGES


@dataclass(frozen=True)
class _PortReadoutEvidence:
    complete: bool
    planned_step_count: int
    attempted_step_count: int
    completed_step_count: int
    completed_source_count: int
    completed_source_readout_bounds: tuple[tuple[int, I], ...]
    endpoint_bounds: tuple[I, I] | None
    failed_source_index: int | None
    unattempted_source_indices: tuple[int, ...]


def _reconstruct_port_readout(report, admitted_inputs) -> _PortReadoutEvidence:
    """Rebuild observations after matching independently admitted primitive inputs."""
    forms, phases, impulse, total, width, order, cap = admitted_inputs
    nodes = tuple(range(27))
    edges = tuple(sorted(tuple(sorted(edge)) for edge in _EDGES))
    _require(
        tuple(map(_integer, report.geometry.nodes)) == nodes
        and tuple(tuple(map(_integer, edge)) for edge in report.geometry.edges)
        == edges,
        "port readout support differs",
    )
    degrees = tuple(sum(node in edge for edge in edges) for node in nodes)
    _require(tuple(map(_integer, report.degrees)) == degrees, "port degrees differ")
    for key, expected in (
        ("storage_scale", Q(1)),
        ("epi_weight", Q(1023, 1024)),
        ("phase_weight", Q(1, 1024)),
    ):
        _require(
            exact_or_represented_real(getattr(report.reference_model, key), key)
            == expected,
            "complete held law differs",
        )
    _require(report.reference_model.phase_domain == "regular", "phase law differs")
    _require(tuple(map(_exact, report.capacity)) == (Q(1),) * 27, "capacities differ")
    _require(report.clock == "tau=e*t; e=1023/1024", "clock differs")
    _require(
        report.state_order == tuple(f"{c}_{i}" for c in ("x", "theta") for i in nodes),
        "state order differs",
    )
    _require(
        report.source_coordinates
        == "pre-input form x and absolute continuous phase theta at 0-",
        "phase coordinate interpretation differs",
    )
    _require(
        tuple(map(_integer, report.source_order)) == (0, 1), "source order differs"
    )
    _require(
        tuple(map(_integer, report.port_nodes)) == (4, 13, 22), "event ports differ"
    )
    _require(_integer(report.readout_node) == 13, "observation node differs")
    _require(tuple(map(_exact, report.port_impulse)) == impulse, "port event differs")
    for key, expected in (("horizon", total), ("time_step", width)):
        _require(_exact(getattr(report, key)) == expected, "clock input differs")
    _require(
        _integer(report.order) == order and _integer(report.max_steps) == cap,
        "work policy differs",
    )
    _require(report.arithmetic_method == INTERVAL_METHOD, "arithmetic method differs")
    _require(
        report.method == "two_source_single_event_full54_Picard_Taylor_v1",
        "producer method differs",
    )
    for actual, expected in (
        (report.initial_form_bounds, forms),
        (report.initial_phase_bounds, phases),
    ):
        _require(
            tuple(tuple(map(_interval, source)) for source in actual) == expected,
            "primitive source association differs",
        )
    _require(len(report.histories) == 2, "complete history inventory is required")
    failure = None
    successes = attempts = 0
    readings, reasons, unattempted = [], [], []
    absent_fields = (
        "pre_event_box",
        "initial_box",
        "completed_time",
        "completed_endpoint_box",
        "final_state_bounds",
        "readout_bounds",
        "failed_initial_box",
        "failed_time",
        "failed_tube",
    )
    jumps = dict(zip((4, 13, 22), impulse))
    for index, history in enumerate(report.histories):
        _require(
            _integer(history.source_index) == index,
            "history source association differs",
        )
        count = _integer(history.attempted_step_count)
        if failure is not None:
            _require(
                history.status == "not_attempted"
                and not history.steps
                and count == 0
                and history.reason is None,
                "execution continued after first failure",
            )
            _require(
                all(getattr(history, key) is None for key in absent_fields),
                "unattempted source invented evidence",
            )
            unattempted.append(index)
            continue
        if history.initial_box is None:
            _require(
                total > 0
                and successes == cap
                and history.status == "budget_exhausted"
                and history.reason == "total_step_budget_exhausted_before_event"
                and not history.steps
                and count == 0,
                "unsupported before-event budget failure",
            )
            _require(
                all(getattr(history, key) is None for key in absent_fields),
                "budget stop applied an event",
            )
            failure = index
            reasons.append(f"source_{index}: {history.reason}")
            unattempted.append(index)
            continue
        before = forms[index] + phases[index]
        _require(_box(history.pre_event_box) == before, "pre-event source differs")
        state = tuple(
            value + jumps[i] if jumps.get(i) else value
            for i, value in enumerate(before)
        )
        _require(_box(history.initial_box) == state, "event changed another coordinate")
        time = Q(0)
        history_successes = 0
        for step in history.steps:
            h = min(width, total - time)
            _require(
                successes < cap
                and h > 0
                and _exact(step.time) == time
                and _exact(step.duration) == h
                and _integer(step.order) == order
                and _box(step.initial_box) == state,
                "step schedule or full-state source differs",
            )
            _require(
                step.method == "direct_source_box_Picard_Taylor_dyadic128_v1",
                "shared step method differs",
            )
            _require(
                _exact(step.picard_interior_margin) > 0
                and tuple(map(_exact, step.domain_lower_bounds)) == (Q(1),),
                "strict smooth-domain evidence is unavailable",
            )
            tube = _box(step.tube)
            remainder = _box(step.local_remainder_bounds)
            series = tuple(tuple(map(_interval, row)) for row in step.series)
            increment, endpoint = reconstruct_box_taylor_arithmetic(
                state, tube, series, remainder, h, order=order
            )
            _require(
                _box(step.increment) == increment and _box(step.endpoint) == endpoint,
                "retained Taylor arithmetic differs",
            )
            state, time = endpoint, time + h
            successes += 1
            history_successes += 1
        _require(
            _exact(history.completed_time) == time
            and _box(history.completed_endpoint_box) == state,
            "retained completion differs",
        )
        if history.status == "admitted":
            _require(
                time == total
                and _box(history.final_state_bounds) == state
                and _interval(history.readout_bounds) == state[13],
                "history endpoint is incomplete or differs",
            )
            _require(
                all(
                    getattr(history, key) is None
                    for key in (
                        "failed_initial_box",
                        "failed_time",
                        "failed_tube",
                        "reason",
                    )
                ),
                "completed history retains failure evidence",
            )
            _require(count == history_successes, "complete attempt count differs")
            readings.append((index, state[13]))
        else:
            _require(
                history.status in ("unavailable", "budget_exhausted")
                and time < total
                and history.final_state_bounds is None
                and history.readout_bounds is None
                and _exact(history.failed_time) == time
                and _box(history.failed_initial_box) == state
                and isinstance(history.reason, str)
                and bool(history.reason),
                "failed attempt association differs",
            )
            if history.status == "unavailable":
                _require(
                    successes < cap and count == history_successes + 1,
                    "failed kernel attempt count differs",
                )
                if history.failed_tube is not None:
                    _box(history.failed_tube)
            else:
                _require(
                    successes == cap
                    and count == history_successes
                    and history.failed_tube is None
                    and history.reason == "total_step_budget_exhausted",
                    "budget failure provenance differs",
                )
            failure = index
            reasons.append(f"source_{index}: {history.reason}")
        attempts += count
    _require(attempts <= cap, "attempt cap exceeded")
    ratio = total / width
    planned = 2 * ((ratio.numerator + ratio.denominator - 1) // ratio.denominator)
    _require(
        _integer(report.planned_step_count) == planned
        and _integer(report.attempted_step_count) == attempts
        and _integer(report.completed_step_count) == successes
        and _integer(report.completed_source_count) == len(readings),
        "evidence counts differ",
    )
    if report.failed_source_index is not None:
        _integer(report.failed_source_index)
    _require(report.failed_source_index == failure, "failed source differs")
    _require(
        tuple(map(_integer, report.unattempted_source_indices)) == tuple(unattempted),
        "unattempted source association differs",
    )
    _require(
        tuple(
            (_integer(index), _interval(value))
            for index, value in report.completed_source_readout_bounds
        )
        == tuple(readings),
        "cached completed readings differ",
    )
    complete = len(readings) == 2
    endpoints = tuple(value for _, value in readings) if complete else None
    actual = report.endpoint_readout_bounds
    _require(
        (None if actual is None else tuple(map(_interval, actual))) == endpoints,
        "cached endpoint pair differs",
    )
    _require(
        report.status == ("admitted" if complete else "unavailable")
        and tuple(report.unavailable_reasons) == tuple(reasons),
        "cached availability differs",
    )
    return _PortReadoutEvidence(
        complete,
        planned,
        attempts,
        successes,
        len(readings),
        tuple(readings),
        endpoints,
        failure,
        tuple(unattempted),
    )
