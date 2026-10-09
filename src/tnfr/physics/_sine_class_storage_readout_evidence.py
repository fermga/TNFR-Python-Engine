"""Re-admit complete storage-readout evidence without regenerating a field.

The retained derivative, remainder and Picard generation remain execution
premises. This reader reconstructs arithmetic, source/event association and
completion, rather than authenticating execution or preparation.
"""

from dataclasses import dataclass
from fractions import Fraction as Q

from .._exact_time import exact_or_represented_real
from ..mathematics._comparison_flow import _exact
from ..mathematics._rational_interval import INTERVAL_METHOD, I
from ..mathematics._validated_taylor import reconstruct_box_taylor_arithmetic
from ._sine_class_readout_evidence import _integer, _interval, _require
from .relational_observations import _ordered
from .relational_sine_class_cubic_response import _cubic_parameters
from .relational_sine_class_mediation import _EDGES, _NODES
from .relational_sine_class_storage_readout import _HISTORIES, _MODELS, _storage_target


def _box55(values):
    rows = _ordered(values, "storage evidence state", limit=56)
    _require(len(rows) == 55, "all 55 storage state coordinates are required")
    return tuple(map(_interval, rows))


@dataclass(frozen=True)
class _StorageReadoutEvidence:
    complete: bool
    planned_step_count: int
    attempted_step_count: int
    completed_step_count: int
    completed_history_count: int
    completed_history_loss_bounds: tuple[tuple[int, I], ...]
    loss_integral_bounds: tuple[I, ...] | None
    raw_endpoint_loss_bounds: tuple[I, ...] | None
    integrated_excess_loss_bounds: I | None
    excess_storage_bounds: I | None
    failed_history_index: int | None
    unattempted_history_indices: tuple[int, ...]


def _reconstruct_storage_readout(report, admitted_inputs) -> _StorageReadoutEvidence:
    """Match already admitted primitive inputs and rebuild every consumed step."""
    k, forms, phases, a, b, total, width, order, cap = admitted_inputs
    _require(_integer(report.mediator_class) == k, "mediator class differs")
    edges = tuple(sorted(tuple(sorted(edge)) for edge in _EDGES))
    _require(
        tuple(map(_integer, report.geometry.nodes)) == _NODES
        and tuple(tuple(map(_integer, edge)) for edge in report.geometry.edges)
        == edges,
        "storage support differs",
    )
    for key, value in (
        ("storage_scale", Q(1)),
        ("epi_weight", Q(1023, 1024)),
        ("phase_weight", Q(1, 1024)),
    ):
        _require(
            exact_or_represented_real(getattr(report.reference_model, key), key)
            == value,
            "complete held law differs",
        )
    _require(report.reference_model.phase_domain == "regular", "phase law differs")
    expected = _cubic_parameters(k)
    _require(
        tuple(map(_integer, report.parameters.classes)) == expected.classes
        and tuple(map(_integer, report.parameters.degrees)) == expected.degrees,
        "tangent target or degrees differ",
    )
    for key in ("gamma", "eta"):
        _require(
            _interval(getattr(report.parameters, key)) == getattr(expected, key),
            "tangent coupling differs",
        )
    for key in ("edge_sines", "edge_cosines"):
        _require(
            tuple(map(_interval, getattr(report.parameters, key)))
            == getattr(expected, key),
            "target edge coefficients differ",
        )
    for key, value in (
        ("initial_form_bounds", forms),
        ("initial_phase_deviation_bounds", phases),
    ):
        _require(
            tuple(map(_interval, getattr(report, key))) == value,
            "primitive source differs",
        )
    for key, value in (
        ("donor_amplitude", a),
        ("receiver_amplitude", b),
        ("horizon", total),
        ("time_step", width),
    ):
        _require(_exact(getattr(report, key)) == value, "event or clock input differs")
    _require(
        _integer(report.order) == order and _integer(report.max_steps) == cap,
        "numerical budget differs",
    )
    _require(tuple(map(_exact, report.capacity)) == (Q(1),) * 27, "capacity differs")
    _require(report.clock == "tau=e*t; e=1023/1024", "clock differs")
    _require(
        report.source_coordinates == "pre-input x and y=theta-Theta_k in radians at 0-",
        "source coordinates differ",
    )
    _require(
        report.loss_law == "sum_i (L*x)_i**2/d_i, using each model's own form",
        "loss law differs",
    )
    _require(
        tuple(report.history_order) == _HISTORIES
        and tuple(report.model_order) == _MODELS,
        "execution order differs",
    )
    _require(
        tuple(map(_integer, report.intervention_nodes)) == (4, 22), "event nodes differ"
    )
    for key, labels in (
        ("full_state_order", ("x", "theta")),
        ("tangent_state_order", ("v", "w")),
    ):
        _require(
            tuple(getattr(report, key))
            == tuple(f"{c}_{i}" for c in labels for i in _NODES) + ("cumulative_loss",),
            "state order differs",
        )
    _require(
        report.arithmetic_method == INTERVAL_METHOD
        and report.method
        == "matched_full_tangent_eight_history_loss55_Picard_Taylor_v1",
        "producer method differs",
    )
    target = _storage_target(expected)
    full_source = forms + tuple(t + y for t, y in zip(target, phases)) + (I(0),)
    tangent_source = forms + phases + (I(0),)
    _require(
        tuple(map(_interval, report.target_phase_bounds)) == target,
        "target construction differs",
    )
    _require(
        _box55(report.full_source_box) == full_source
        and _box55(report.tangent_source_box) == tangent_source,
        "matched source construction differs",
    )
    histories = _ordered(report.histories, "histories", limit=9)
    _require(len(histories) == 8, "all eight history associations are required")
    successes = attempts = 0
    failure = None
    readings, raw_endpoints, reasons, unattempted = [], [], [], []
    absent = (
        "pre_event_box",
        "initial_box",
        "completed_time",
        "completed_endpoint_box",
        "completed_loss_increment_bounds",
        "final_state_bounds",
        "loss_integral_bounds",
        "failed_initial_box",
        "failed_time",
        "failed_tube",
    )
    for index, history in enumerate(histories):
        word, model_index = divmod(index, 2)
        impulses = ((Q(0), Q(0)), (a, Q(0)), (Q(0), b), (a, b))[word]
        _require(
            _integer(history.history_index) == index
            and history.history_name == _HISTORIES[word]
            and history.model_name == _MODELS[model_index],
            "history association differs",
        )
        _require(
            tuple(map(_exact, history.event_amplitudes)) == impulses,
            "history event differs",
        )
        count = _integer(history.attempted_step_count)
        if failure is not None:
            _require(
                history.status == "not_attempted"
                and not history.steps
                and count == 0
                and history.reason is None
                and all(getattr(history, key) is None for key in absent),
                "execution continued after first failure",
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
                and count == 0
                and all(getattr(history, key) is None for key in absent),
                "unsupported before-event budget failure",
            )
            failure = index
            reasons.append(f"history_{index}: {history.reason}")
            unattempted.append(index)
            continue
        before = (full_source, tangent_source)[model_index]
        _require(_box55(history.pre_event_box) == before, "pre-event source differs")
        jumps = dict(zip((4, 22), impulses))
        state = tuple(
            value + jumps[i] if jumps.get(i) else value
            for i, value in enumerate(before)
        )
        _require(
            _box55(history.initial_box) == state,
            "form event changed another coordinate",
        )
        time, accumulated, local_successes = Q(0), I(0), 0
        for step in history.steps:
            h = min(width, total - time)
            _require(
                successes < cap
                and h > 0
                and _exact(step.time) == time
                and _exact(step.duration) == h
                and _integer(step.order) == order
                and _box55(step.initial_box) == state,
                "step schedule or complete source differs",
            )
            _require(
                step.method == "direct_source_box_Picard_Taylor_dyadic128_v1"
                and _exact(step.picard_interior_margin) > 0
                and tuple(map(_exact, step.domain_lower_bounds)) == (Q(1),),
                "strict smooth-domain evidence differs",
            )
            tube = _box55(step.tube)
            remainder = _box55(step.local_remainder_bounds)
            series = tuple(tuple(map(_interval, row)) for row in step.series)
            increment, endpoint = reconstruct_box_taylor_arithmetic(
                state, tube, series, remainder, h, order=order
            )
            _require(
                _box55(step.increment) == increment
                and _box55(step.endpoint) == endpoint,
                "retained Taylor arithmetic differs",
            )
            accumulated += increment[54]
            state, time = endpoint, time + h
            successes += 1
            local_successes += 1
        _require(
            _exact(history.completed_time) == time
            and _box55(history.completed_endpoint_box) == state
            and _interval(history.completed_loss_increment_bounds) == accumulated,
            "retained completion differs",
        )
        if history.status == "admitted":
            _require(
                time == total
                and _box55(history.final_state_bounds) == state
                and _interval(history.loss_integral_bounds) == accumulated,
                "history completion or loss differs",
            )
            _require(
                count == local_successes
                and all(
                    getattr(history, key) is None
                    for key in (
                        "failed_initial_box",
                        "failed_time",
                        "failed_tube",
                        "reason",
                    )
                ),
                "complete history has failure evidence",
            )
            readings.append((index, accumulated))
            raw_endpoints.append(state[54])
        else:
            _require(
                history.status in ("unavailable", "budget_exhausted")
                and time < total
                and history.final_state_bounds is None
                and history.loss_integral_bounds is None
                and _exact(history.failed_time) == time
                and _box55(history.failed_initial_box) == state
                and isinstance(history.reason, str)
                and bool(history.reason),
                "failed attempt association differs",
            )
            if history.status == "unavailable":
                _require(
                    successes < cap and count == local_successes + 1,
                    "failed attempt count differs",
                )
                if history.failed_tube is not None:
                    _box55(history.failed_tube)
            else:
                _require(
                    successes == cap
                    and count == local_successes
                    and history.failed_tube is None
                    and history.reason == "total_step_budget_exhausted",
                    "budget failure differs",
                )
            failure = index
            reasons.append(f"history_{index}: {history.reason}")
        attempts += count
    _require(attempts <= cap, "attempt cap exceeded")
    ratio = total / width
    planned = 8 * ((ratio.numerator + ratio.denominator - 1) // ratio.denominator)
    for key, value in (
        ("planned_step_count", planned),
        ("attempted_step_count", attempts),
        ("completed_step_count", successes),
        ("completed_history_count", len(readings)),
    ):
        _require(_integer(getattr(report, key)) == value, "evidence counts differ")
    if report.failed_history_index is not None:
        _integer(report.failed_history_index)
    _require(
        report.failed_history_index == failure
        and tuple(map(_integer, report.unattempted_history_indices))
        == tuple(unattempted),
        "failure inventory differs",
    )
    _require(
        tuple(
            (_integer(index), _interval(value))
            for index, value in report.completed_history_loss_bounds
        )
        == tuple(readings),
        "cached individual losses differ",
    )
    complete = len(readings) == 8
    losses = tuple(value for _, value in readings) if complete else None
    raw = tuple(raw_endpoints) if complete else None
    # Independent signed combination, ordered as full/tangent for each word.
    mixed = None
    if complete:
        full = losses[6] - losses[2] - losses[4] + losses[0]
        tangent = losses[7] - losses[3] - losses[5] + losses[1]
        mixed = full - tangent
    storage = -mixed if mixed is not None else None
    for key, expected_bounds in (
        ("loss_integral_bounds", losses),
        ("raw_endpoint_loss_bounds", raw),
    ):
        actual = getattr(report, key)
        _require(
            (None if actual is None else tuple(map(_interval, actual)))
            == expected_bounds,
            "cached loss vector differs",
        )
    for key, expected_bounds in (
        ("integrated_excess_loss_bounds", mixed),
        ("excess_storage_bounds", storage),
    ):
        actual = getattr(report, key)
        _require(
            (None if actual is None else _interval(actual)) == expected_bounds,
            "cached mixed bounds differ",
        )
    _require(
        report.status == ("admitted" if complete else "unavailable")
        and tuple(report.unavailable_reasons) == tuple(reasons),
        "cached availability differs",
    )
    return _StorageReadoutEvidence(
        complete,
        planned,
        attempts,
        successes,
        len(readings),
        tuple(readings),
        losses,
        raw,
        mixed,
        storage,
        failure,
        tuple(unattempted),
    )
