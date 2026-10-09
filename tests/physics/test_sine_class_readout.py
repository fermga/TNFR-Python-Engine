"""Primitive admission, carried branch ancestry and bounded partial evidence.

All executed flows use independent constant polynomial controls. Neither the
reserved class-two design nor a formation/analytic response is evaluated.
"""

import json
import subprocess
from dataclasses import replace
from decimal import Decimal
from fractions import Fraction as Q
from inspect import Parameter, signature
from itertools import product
from types import SimpleNamespace

import numpy as np
import pytest

from tnfr.mathematics._rational_interval import I
from tnfr.mathematics._validated_taylor import ValidatedBoxTaylorStep
from tnfr.physics import _sine_class_readout_evidence as evidence_owner
from tnfr.physics import relational_sine_class_readout as owner
from tnfr.physics._sine_class_readout_evidence import _reconstruct_class_readout
from tnfr.sdk.relational_reports import relational_report_to_dict
from tnfr.utils.io import json_loads


def _arguments():
    return dict(
        initial_form_bounds=tuple((Q(i - 13, 101), Q(i - 13, 101)) for i in range(27)),
        initial_phase_bounds=tuple(
            (Q(2 * i - 25, 103), Q(2 * i - 25, 103)) for i in range(27)
        ),
        first_probe_amplitude=Q(2, 7),
        second_probe_amplitude=Q(-3, 11),
        delay=Q(2, 7),
        total_duration=Q(5, 7),
        time_step=Q(1, 5),
        order=3,
        max_steps=16,
    )


def _constant_field(*_):
    def flow(state):
        return tuple(state[0] * 0 + Q(i + 1, 100) for i in range(54))

    return flow, lambda _: (Q(1),)


@pytest.fixture(scope="module", autouse=True)
def no_source_or_prediction():
    from tnfr.physics import (
        _sine_formed_contact,
        relational_sine_class_mediation,
        relational_sine_class_nonlinear_protocol,
        relational_sine_class_superposition,
    )

    def forbidden(*args, **kwargs):
        pytest.fail("observation must not execute acquisition, prediction or workers")

    with pytest.MonkeyPatch.context() as patch:
        for module, names in (
            (_sine_formed_contact, ("_unprobed_handoff",)),
            (relational_sine_class_mediation, ("assess_sine_class_mediation",)),
            (
                relational_sine_class_nonlinear_protocol,
                ("bound_sine_class_nonlinear_protocol", "_heat_cubic_channels"),
            ),
            (relational_sine_class_superposition, ("bound_sine_class_superposition",)),
            (subprocess, ("run", "Popen")),
        ):
            for name in names:
                patch.setattr(module, name, forbidden)
        yield


@pytest.fixture(scope="module")
def polynomial_report():
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(owner, "_full_sine_field", _constant_field)
        return owner.bound_sine_class_four_history_readout(**_arguments())


def _arithmetic_step(size):
    """Independent polynomial corners, with endpoint-only tube intersection."""
    initial = tuple(I(i - 7, i + 9) for i in range(size))
    changes = tuple(
        c1 / 2 + Q(3, 4) + c3 / 8 + remainder
        for c1, c3, remainder in product(
            (Q(-2), Q(1)), (Q(-1), Q(2)), (Q(-1, 16), Q(1, 8))
        )
    )
    increment = I(min(changes), max(changes))
    return ValidatedBoxTaylorStep(
        time=Q(3, 4),
        duration=Q(1, 2),
        order=3,
        initial_box=initial,
        tube=tuple(I(i - 8, i + 10) for i in range(size)),
        series=tuple((x, I(-2, 1), I(3), I(-1, 2)) for x in initial),
        local_remainder_bounds=(I(Q(-1, 16), Q(1, 8)),) * size,
        increment=(increment,) * size,
        endpoint=tuple(I(i - 7 + min(changes), i + 10) for i in range(size)),
        picard_interior_margin=Q(1, 16),
        domain_lower_bounds=(Q(1),),
    )


def _rebuild_step(step, initial):
    return evidence_owner._reconstruct_readout_step(
        step, initial, Q(3, 4), Q(1, 2), order=3
    )


@pytest.mark.parametrize("size", (54, 55))
@pytest.mark.parametrize("container", (tuple, list, iter))
def test_shared_retained_step_preserves_independent_corner_arithmetic(size, container):
    step = _arithmetic_step(size)
    converted = replace(
        step,
        **{
            key: container(getattr(step, key))
            for key in (
                "initial_box",
                "tube",
                "local_remainder_bounds",
                "increment",
                "endpoint",
                "domain_lower_bounds",
            )
        },
        series=container(container(row) for row in step.series),
    )
    increment, endpoint = _rebuild_step(converted, container(step.initial_box))
    assert increment == step.increment == (I(Q(-7, 16), Q(13, 8)),) * size
    assert endpoint == step.endpoint
    assert endpoint[0] == I(Q(-119, 16), 10)
    assert increment[0] != endpoint[0] - step.initial_box[0]


@pytest.mark.parametrize("size", (54, 55))
@pytest.mark.parametrize("container", (set, frozenset, dict.fromkeys))
@pytest.mark.parametrize(
    "field",
    (
        "initial_box",
        "tube",
        "local_remainder_bounds",
        "increment",
        "endpoint",
        "series",
        "coefficient_row",
        "domain_lower_bounds",
    ),
)
def test_retained_step_rejects_unordered_evidence_before_materialization(
    size, container, field
):
    step = _arithmetic_step(size)
    if field == "coefficient_row":
        changes = {"series": (container(step.series[0]),) + step.series[1:]}
    else:
        changes = {field: container(getattr(step, field))}
    with pytest.raises(TypeError, match="ordered iterable"):
        _rebuild_step(replace(step, **changes), step.initial_box)


@pytest.mark.parametrize("size", (54, 55))
@pytest.mark.parametrize("field", ("initial_box", "series", "coefficient_row"))
def test_retained_step_bounds_iterator_consumption(size, field):
    step = _arithmetic_step(size)
    seen = []

    def endless(value):
        while True:
            seen.append(None)
            yield value

    if field == "coefficient_row":
        changes = {"series": (endless(I(0)),) + step.series[1:]}
        expected_count = step.order + 2
    else:
        value = step.series[0] if field == "series" else I(0)
        changes = {field: endless(value)}
        expected_count = size + 1
    with pytest.raises(ValueError, match="exactly"):
        _rebuild_step(replace(step, **changes), step.initial_box)
    assert len(seen) == expected_count


@pytest.mark.parametrize("size", (54, 55))
@pytest.mark.parametrize(
    "changes",
    (
        {"time": True},
        {"duration": Q(1, 4)},
        {"order": True},
        {"method": "another_step_method"},
        {"picard_interior_margin": Q(0)},
        {"domain_lower_bounds": (True,)},
    ),
)
def test_retained_step_rejects_clock_method_and_domain_tampering(size, changes):
    step = _arithmetic_step(size)
    with pytest.raises((TypeError, ValueError)):
        _rebuild_step(replace(step, **changes), step.initial_box)


@pytest.mark.parametrize(
    "changes",
    (
        {"initial_box": ()},
        {"initial_box": (I(0),) * 65},
        {"initial_box": (True,) * 54},
        {"initial_box": dict.fromkeys(I(i) for i in range(54))},
        {"time": True},
        {"time": -1},
        {"duration": True},
        {"duration": 0},
        {"duration": -1},
        {"order": True},
        {"order": Q(3)},
        {"order": 17},
    ),
)
def test_retained_step_readmits_its_expected_state_and_policy(changes):
    step = _arithmetic_step(54)
    arguments = dict(
        initial_box=step.initial_box, time=step.time, duration=step.duration, order=3
    )
    arguments.update(changes)
    with pytest.raises((TypeError, ValueError)):
        evidence_owner._reconstruct_readout_step(step, **arguments)


@pytest.mark.parametrize("field", ("initial_box", "series", "method"))
def test_four_history_reader_uses_shared_ordered_step_admission(
    polynomial_report, field
):
    segment = polynomial_report.segments[0]
    step = segment.steps[0]
    value = (
        "another_step_method"
        if field == "method"
        else dict.fromkeys(getattr(step, field))
    )
    changed = replace(
        segment, steps=(replace(step, **{field: value}),) + segment.steps[1:]
    )
    report = replace(
        polynomial_report, segments=(changed,) + polynomial_report.segments[1:]
    )
    with pytest.raises((TypeError, ValueError)):
        _reconstruct_class_readout(
            report, owner._admit_class_readout_inputs(**_arguments())
        )


def test_nine_mandatory_primitives_exclude_target_prediction_and_sensor():
    parameters = signature(owner.bound_sine_class_four_history_readout).parameters
    selectors = dict(first_probe_node=4, second_probe_node=4, readout_node=22)
    mandatory = {
        key: p for key, p in parameters.items() if p.default == Parameter.empty
    }
    assert set(mandatory) == set(_arguments()) and len(mandatory) == 9
    assert all(p.kind == Parameter.KEYWORD_ONLY for p in parameters.values())
    assert {
        key: p.default for key, p in parameters.items() if key not in mandatory
    } == (selectors)
    for key in (
        "mediator_class",
        "source_handoff",
        "readout_error_bound",
        "prediction",
        "gain",
    ):
        with pytest.raises(TypeError):
            owner.bound_sine_class_four_history_readout(**_arguments(), **{key: 0})


def test_explicit_legacy_selectors_preserve_default_report(
    polynomial_report, monkeypatch
):
    monkeypatch.setattr(owner, "_full_sine_field", _constant_field)
    explicit = owner.bound_sine_class_four_history_readout(
        **_arguments(), first_probe_node=4, second_probe_node=4, readout_node=22
    )
    assert explicit == polynomial_report


@pytest.mark.parametrize(
    "key", ("first_probe_node", "second_probe_node", "readout_node")
)
@pytest.mark.parametrize(
    "bad", (True, np.bool_(False), np.int64(4), Q(4), 4.0, -1, 27, None)
)
def test_node_selectors_reject_before_field_or_step(key, bad, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("selector admission must precede execution")

    monkeypatch.setattr(owner, "_full_sine_field", forbidden)
    monkeypatch.setattr(owner, "validated_box_taylor_step", forbidden)
    with pytest.raises(ValueError, match=key):
        owner.bound_sine_class_four_history_readout(**_arguments(), **{key: bad})


@pytest.mark.parametrize("name", tuple(_arguments())[2:])
@pytest.mark.parametrize("invalid", (True, np.bool_(False), float("inf"), float("nan")))
def test_invalid_scalars_fail_before_shared_field(name, invalid, monkeypatch):
    monkeypatch.setattr(
        owner, "_full_sine_field", lambda *_: pytest.fail("admission bypassed")
    )
    values = _arguments()
    values[name] = invalid
    with pytest.raises((TypeError, ValueError)):
        owner.bound_sine_class_four_history_readout(**values)


@pytest.mark.parametrize(
    "key,value",
    (
        ("delay", -1),
        ("delay", Q(6, 7)),
        ("total_duration", Q(2) + Q(1, 2**400)),
        ("time_step", 0),
        ("time_step", Q(2) + Q(1, 2**400)),
        ("order", Q(3)),
        ("order", np.int64(3)),
        ("order", 0),
        ("order", 17),
        ("max_steps", Q(16)),
        ("max_steps", 0),
        ("max_steps", 4097),
        ("first_probe_amplitude", Decimal("1e-400")),
    ),
)
def test_domains_and_representation_precede_execution(key, value, monkeypatch):
    monkeypatch.setattr(
        owner,
        "validated_box_taylor_step",
        lambda *_a, **_k: pytest.fail("admission bypassed"),
    )
    values = _arguments()
    values[key] = value
    with pytest.raises((TypeError, ValueError)):
        owner.bound_sine_class_four_history_readout(**values)


@pytest.mark.parametrize("channel", ("initial_form_bounds", "initial_phase_bounds"))
@pytest.mark.parametrize(
    "invalid", (True, np.bool_(False), float("nan"), float("inf"), Decimal("1e-400"))
)
def test_every_source_channel_readmits_authoritative_endpoints(
    channel, invalid, monkeypatch
):
    monkeypatch.setattr(
        owner, "_full_sine_field", lambda *_: pytest.fail("admission bypassed")
    )
    values = _arguments()
    source = list(values[channel])
    source[26] = (Q(0), invalid)
    values[channel] = source
    with pytest.raises((TypeError, ValueError)):
        owner.bound_sine_class_four_history_readout(**values)


@pytest.mark.parametrize(
    "source",
    (((0, 0),) * 26, ((0, 0),) * 28, ((1, 0),) * 27, (I(0),) * 27, {(0, 0)}, "source"),
)
def test_source_shape_order_and_interval_objects_cannot_bypass_admission(
    source, monkeypatch
):
    monkeypatch.setattr(
        owner, "_full_sine_field", lambda *_: pytest.fail("admission bypassed")
    )
    values = _arguments()
    values["initial_form_bounds"] = source
    with pytest.raises((TypeError, ValueError)):
        owner.bound_sine_class_four_history_readout(**values)


def test_six_segments_share_two_prefixes_and_retain_exact_fixed_grid(polynomial_report):
    report = polynomial_report
    assert report.admitted
    assert (
        report.planned_unique_step_count
        == report.attempted_step_count
        == report.completed_step_count
        == 16
    )
    assert report.completed_segment_count == 6
    assert report.history_segment_indices == ((0, 2), (1, 3), (0, 4), (1, 5))
    assert (
        report.failed_segment_index is None and report.unattempted_segment_indices == ()
    )
    assert report.unavailable_reasons == ()
    for segment in report.segments:
        time = segment.start_time
        for step in segment.steps:
            assert step.time == time
            assert step.duration == min(report.time_step, segment.end_time - time)
            assert step.picard_interior_margin > 0
            assert step.domain_lower_bounds == (Q(1),)
            assert len(step.series) == 54 and all(
                len(row) == report.order + 1 for row in step.series
            )
            time += step.duration
        assert time == segment.completed_time == segment.end_time
        assert segment.final_state_bounds == segment.completed_endpoint_box
        assert segment.completed_receiver_increment_bounds == sum(
            (step.increment[22] for step in segment.steps), I(0)
        )


def test_every_event_carries_all54_coordinates_and_completed_states(polynomial_report):
    report = polynomial_report
    for segment in report.segments:
        previous = (
            report.source_box
            if segment.parent_segment_index is None
            else report.segments[segment.parent_segment_index].final_state_bounds
        )
        assert segment.pre_event_box == previous
        expected = tuple(
            value + segment.form_jump if i == 4 and segment.form_jump else value
            for i, value in enumerate(previous)
        )
        assert segment.initial_box == expected
        assert segment.initial_box[27:] == previous[27:]
        for index, step in enumerate(segment.steps):
            assert step.initial_box == (
                segment.initial_box if index == 0 else segment.steps[index - 1].endpoint
            )
        duration = segment.end_time - segment.start_time
        for i in range(54):
            propagated = I(
                expected[i].lo + Q(i + 1, 100) * duration,
                expected[i].hi + Q(i + 1, 100) * duration,
            )
            assert segment.final_state_bounds[i].contains(propagated)
    expected_receiver = (
        _arguments()["initial_form_bounds"][22][0] + Q(23, 100) * report.total_duration
    )
    assert all(
        value.contains(expected_receiver) for value in report.endpoint_readout_bounds
    )
    assert report.mixed_readout_bounds.contains(0)
    assert report.suffix_receiver_increment_bounds == tuple(
        segment.completed_receiver_increment_bounds for segment in report.segments[2:]
    )


def _small_arguments(**updates):
    values = _arguments()
    values.update(delay=Q(1, 8), total_duration=Q(1, 4), time_step=Q(1, 8), max_steps=6)
    values.update(updates)
    return values


@pytest.fixture(scope="module", params=("before_event", "inside_segment"))
def budget_stopped_report(request):
    arguments = _small_arguments(max_steps=1)
    if request.param == "inside_segment":
        arguments.update(delay=Q(3, 16), time_step=Q(1, 16))
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(owner, "_full_sine_field", _constant_field)
        report = owner.bound_sine_class_four_history_readout(**arguments)
    return report, owner._admit_class_readout_inputs(**arguments)


@pytest.mark.parametrize(
    "mutation", ("reason", "failed_tube", "missing_summary", "different_summary")
)
def test_reader_rejects_contradictory_budget_failure_provenance(
    budget_stopped_report, mutation
):
    report, admitted = budget_stopped_report
    retained = _reconstruct_class_readout(report, admitted)
    assert not retained.complete and retained.attempted_step_count == 1
    index = report.failed_segment_index
    segment = report.segments[index]
    if mutation == "reason":
        # Keep the summary consistent with the forged segment: the exact
        # budget-stop contract itself must reject a solver-failure label.
        segment = replace(segment, reason="solver_domain_failure")
        report = replace(
            report, unavailable_reasons=(f"{segment.label}: {segment.reason}",)
        )
    elif mutation == "failed_tube":
        segment = replace(segment, failed_tube=(I(0),) * 54)
    elif mutation == "missing_summary":
        report = replace(report, unavailable_reasons=())
    else:
        report = replace(report, unavailable_reasons=("another_branch: failure",))
    report = replace(
        report,
        segments=report.segments[:index] + (segment,) + report.segments[index + 1 :],
    )
    with pytest.raises(ValueError):
        _reconstruct_class_readout(report, admitted)


@pytest.mark.parametrize(
    "changes",
    (
        {"method": "another_producer"},
        {"unavailable_reasons": ("invented_failure",)},
    ),
)
def test_reader_checks_complete_report_method_and_failure_summary(
    polynomial_report, changes
):
    with pytest.raises(ValueError):
        _reconstruct_class_readout(
            replace(polynomial_report, **changes),
            owner._admit_class_readout_inputs(**_arguments()),
        )


@pytest.mark.parametrize("failure_call", (1, 2, 4, 6))
@pytest.mark.parametrize("selected", (False, True))
def test_first_failed_step_stops_later_events_and_retains_partial_evidence(
    failure_call, selected, monkeypatch
):
    monkeypatch.setattr(owner, "_full_sine_field", _constant_field)
    actual_step = owner.validated_box_taylor_step
    calls = []

    def selected_failure(box, duration, flow, domain, **kwargs):
        calls.append((box, kwargs["time"]))
        if len(calls) == failure_call:
            tube = tuple(I(value.lo - 1, value.hi + 1) for value in box)
            return None, tube, "independent_control_failure"
        return actual_step(box, duration, flow, domain, **kwargs)

    monkeypatch.setattr(owner, "validated_box_taylor_step", selected_failure)
    selectors = (
        dict(first_probe_node=4, second_probe_node=22, readout_node=13)
        if selected
        else {}
    )
    arguments = _small_arguments(**selectors)
    report = owner.bound_sine_class_four_history_readout(**arguments)
    index = failure_call - 1
    assert len(calls) == report.attempted_step_count == failure_call
    assert report.completed_step_count == failure_call - 1
    assert report.failed_segment_index == index
    segment = report.segments[index]
    assert segment.failed_initial_box == calls[-1][0] == segment.initial_box
    assert segment.failed_time == calls[-1][1]
    assert segment.failed_tube == tuple(
        I(value.lo - 1, value.hi + 1) for value in segment.failed_initial_box
    )
    assert (
        segment.final_state_bounds is None
        and segment.reason == "independent_control_failure"
    )
    assert report.unattempted_segment_indices == tuple(range(index + 1, 6))
    assert all(
        s.initial_box is s.pre_event_box is s.final_state_bounds is None
        for s in report.segments[index + 1 :]
    )
    assert report.endpoint_readout_bounds is report.raw_endpoint_mixed_bounds is None
    assert (
        report.mixed_readout_bounds is report.suffix_receiver_increment_bounds is None
    )
    assert len(report.completed_history_readout_bounds) == max(0, index - 2)
    if index == 5:
        assert segment.pre_event_box == report.segments[1].final_state_bounds
        assert (
            segment.initial_box[report.second_probe_node]
            == segment.pre_event_box[report.second_probe_node]
            + report.second_probe_amplitude
        )
        assert segment.initial_box[27:] == segment.pre_event_box[27:]
    evidence = _reconstruct_class_readout(
        report,
        owner._admit_class_readout_inputs(
            **{k: v for k, v in arguments.items() if k not in selectors}
        ),
        **selectors,
    )
    assert not evidence.complete and evidence.mixed_bounds is None
    assert evidence.attempted_step_count == failure_call


@pytest.mark.parametrize("budget", (1, 2, 3, 5))
@pytest.mark.parametrize("selected", (False, True))
def test_global_work_cap_stops_before_the_next_event(budget, selected, monkeypatch):
    monkeypatch.setattr(owner, "_full_sine_field", _constant_field)
    selectors = (
        dict(first_probe_node=4, second_probe_node=22, readout_node=13)
        if selected
        else {}
    )
    report = owner.bound_sine_class_four_history_readout(
        **_small_arguments(max_steps=budget, **selectors)
    )
    assert report.attempted_step_count == report.completed_step_count == budget
    assert report.failed_segment_index == budget
    failed = report.segments[budget]
    assert failed.status == "budget_exhausted"
    assert failed.initial_box is failed.pre_event_box is None
    assert failed.failed_tube is None and failed.completed_time is None
    assert failed.reason == "unique_step_budget_exhausted_before_event"
    assert not report.admitted and report.mixed_readout_bounds is None
    evidence = _reconstruct_class_readout(
        report,
        owner._admit_class_readout_inputs(**_small_arguments(max_steps=budget)),
        **selectors,
    )
    assert not evidence.complete and evidence.attempted_step_count == budget


def test_budget_failure_inside_segment_retains_last_successful_full_endpoint(
    monkeypatch,
):
    monkeypatch.setattr(owner, "_full_sine_field", _constant_field)
    report = owner.bound_sine_class_four_history_readout(
        **_small_arguments(delay=Q(3, 16), time_step=Q(1, 16), max_steps=1)
    )
    failed = report.segments[0]
    assert failed.status == "budget_exhausted" and len(failed.steps) == 1
    assert failed.completed_time == failed.failed_time == Q(1, 16)
    assert (
        failed.failed_initial_box
        == failed.completed_endpoint_box
        == failed.steps[0].endpoint
    )
    assert failed.final_state_bounds is None and failed.failed_tube is None
    assert report.unattempted_segment_indices == (1, 2, 3, 4, 5)


def test_zero_duration_composes_signed_events_without_any_flow(monkeypatch):
    monkeypatch.setattr(owner, "_full_sine_field", _constant_field)
    monkeypatch.setattr(
        owner,
        "validated_box_taylor_step",
        lambda *_a, **_k: pytest.fail("zero-duration flow"),
    )
    report = owner.bound_sine_class_four_history_readout(
        **_small_arguments(delay=0, total_duration=0, max_steps=1)
    )
    assert (
        report.admitted
        and report.attempted_step_count == report.planned_unique_step_count == 0
    )
    assert report.completed_segment_count == 6
    assert report.mixed_readout_bounds == I(0)
    for (first, second), (_, suffix) in zip(
        report.history_event_amplitudes, report.history_segment_indices
    ):
        assert (
            report.segments[suffix]
            .final_state_bounds[4]
            .contains(_arguments()["initial_form_bounds"][4][0] + first + second)
        )


def test_exact_tiny_time_and_events_are_not_coerced_to_zero(monkeypatch):
    monkeypatch.setattr(owner, "_full_sine_field", _constant_field)
    tiny = Q(1, 2**500)
    report = owner.bound_sine_class_four_history_readout(
        **_small_arguments(
            first_probe_amplitude=tiny,
            second_probe_amplitude=-tiny,
            delay=tiny,
            total_duration=2 * tiny,
            time_step=tiny,
        )
    )
    assert report.admitted and report.attempted_step_count == 6
    assert report.first_probe_amplitude == report.delay == tiny
    assert report.segments[5].steps[0].duration == tiny


def test_sdk_projects_full_step_provenance_and_explicit_availability(
    polynomial_report, monkeypatch
):
    direct = polynomial_report.to_dict()
    assert direct["schema"] == "tnfr.sine-class-four-history-readout.v1"
    projected = relational_report_to_dict(polynomial_report)
    assert projected["report"] == direct["report"]
    assert json_loads(json.dumps(projected, allow_nan=False)) == projected
    fraction = projected["report"]["segments"][0]["steps"][0]["duration"]
    assert Q(fraction["numerator"], fraction["denominator"]) == Q(1, 5)
    monkeypatch.setattr(owner, "_full_sine_field", _constant_field)
    partial = owner.bound_sine_class_four_history_readout(
        **_small_arguments(max_steps=1)
    )
    assert partial.to_dict()["report"]["mixed_readout_bounds"] is None
    assert partial.to_dict()["report"]["segments"][2]["initial_box"] is None


_DISTINCT_SELECTORS = dict(first_probe_node=4, second_probe_node=22, readout_node=13)


def _distinct_polynomial_field(*_):
    def flow(state):
        result = [state[0] * 0] * 54
        result[13] = state[4] * state[22]
        return tuple(result)

    return flow, lambda _: (Q(1),)


def _distinct_arguments(delay, *, offset=Q(0)):
    forms = [(Q(0), Q(0))] * 27
    forms[4], forms[22], forms[13] = (
        (Q(1, 8),) * 2,
        (Q(-1, 16),) * 2,
        (10 + offset, 11 + offset),
    )
    return dict(
        initial_form_bounds=tuple(forms),
        initial_phase_bounds=tuple((Q(i, 64), Q(i, 64)) for i in range(27)),
        first_probe_amplitude=Q(1, 16),
        second_probe_amplitude=Q(-1, 32),
        delay=delay,
        total_duration=Q(3, 8),
        time_step=Q(1, 8),
        order=2,
        max_steps=12,
    )


@pytest.fixture(scope="module", params=(Q(0), Q(1, 8)))
def distinct_polynomial(request):
    arguments = _distinct_arguments(request.param)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(owner, "_full_sine_field", _distinct_polynomial_field)
        report = owner.bound_sine_class_four_history_readout(
            **arguments, **_DISTINCT_SELECTORS
        )
    return report, arguments


def test_distinct_simultaneous_and_delayed_events_have_independent_exact_mixed_signal(
    distinct_polynomial,
):
    report, arguments = distinct_polynomial
    a, b = arguments["first_probe_amplitude"], arguments["second_probe_amplitude"]
    delay, total = arguments["delay"], arguments["total_duration"]
    assert report.admitted
    assert (report.first_probe_node, report.second_probe_node, report.readout_node) == (
        4,
        22,
        13,
    )
    assert report.donor_node == 4 and report.receiver_node == 13
    assert report.initial_receiver_form_bounds == I(10, 11)
    for history, (first, second) in enumerate(report.history_event_amplitudes):
        suffix = report.segments[history + 2]
        parent = report.segments[suffix.parent_segment_index]
        assert suffix.pre_event_box == parent.final_state_bounds
        initial = list(parent.final_state_bounds)
        initial[22] += second
        assert suffix.initial_box == tuple(initial)
        assert suffix.final_state_bounds[27:] == report.source_box[27:]
        assert suffix.final_state_bounds[4] == I(Q(1, 8) + first)
        assert suffix.final_state_bounds[22] == I(Q(-1, 16) + second)
        increment = (Q(1, 8) + first) * (Q(-1, 16) + second) * (total - delay)
        assert report.suffix_receiver_increment_bounds[history] == I(increment)
        prefix_change = (Q(1, 8) + first) * Q(-1, 16) * delay
        assert report.endpoint_readout_bounds[history] == I(
            10 + prefix_change + increment, 11 + prefix_change + increment
        )
        for i in range(54):
            if i not in (4, 13, 22):
                assert suffix.final_state_bounds[i] == report.source_box[i]
    assert report.mixed_readout_bounds == I(a * b * (total - delay))
    assert report.raw_endpoint_mixed_bounds.width == 4
    evidence = _reconstruct_class_readout(
        report, owner._admit_class_readout_inputs(**arguments), **_DISTINCT_SELECTORS
    )
    assert evidence.complete and evidence.mixed_bounds == report.mixed_readout_bounds
    assert evidence.endpoint_bounds == report.endpoint_readout_bounds


def test_distinct_observation_cancels_shared_offset_and_projects_selectors(
    distinct_polynomial, monkeypatch
):
    report, arguments = distinct_polynomial
    monkeypatch.setattr(owner, "_full_sine_field", _distinct_polynomial_field)
    shifted = owner.bound_sine_class_four_history_readout(
        **_distinct_arguments(arguments["delay"], offset=Q(2**100)),
        **_DISTINCT_SELECTORS,
    )
    assert shifted.mixed_readout_bounds == report.mixed_readout_bounds
    assert (
        shifted.suffix_receiver_increment_bounds
        == report.suffix_receiver_increment_bounds
    )
    assert shifted.initial_receiver_form_bounds == I(10 + 2**100, 11 + 2**100)
    encoded = relational_report_to_dict(report)
    assert json_loads(json.dumps(encoded, allow_nan=False)) == encoded
    assert {
        key: encoded["report"][key] for key in _DISTINCT_SELECTORS
    } == _DISTINCT_SELECTORS
    assert encoded["report"]["donor_node"] == 4
    assert encoded["report"]["receiver_node"] == 13


@pytest.mark.parametrize("observer", (4, 22))
@pytest.mark.parametrize("horizon", (Q(0), Q(1, 8)))
def test_observed_impulse_at_final_time_cancels_without_losing_raw_event(
    observer, horizon, monkeypatch
):
    monkeypatch.setattr(owner, "_full_sine_field", _constant_field)
    arguments = _small_arguments(delay=horizon, total_duration=horizon, max_steps=2)
    selectors = dict(first_probe_node=4, second_probe_node=22, readout_node=observer)
    result = owner.bound_sine_class_four_history_readout(**arguments, **selectors)
    assert result.admitted and result.suffix_receiver_increment_bounds == (I(0),) * 4
    assert result.mixed_readout_bounds == I(0)
    baseline = (
        arguments["initial_form_bounds"][observer][0] + Q(observer + 1, 100) * horizon
    )
    for endpoint, (a, b) in zip(
        result.endpoint_readout_bounds, result.history_event_amplitudes
    ):
        assert endpoint.contains(baseline + (a if observer == 4 else b))
    rebuilt = _reconstruct_class_readout(
        result, owner._admit_class_readout_inputs(**arguments), **selectors
    )
    assert rebuilt.complete and rebuilt.mixed_bounds == I(0)


@pytest.mark.parametrize("key", (*_DISTINCT_SELECTORS, "donor_node", "receiver_node"))
@pytest.mark.parametrize("bad", (True, Q(4), None, 26))
def test_evidence_rejects_invalid_or_conflicting_selector_metadata(
    distinct_polynomial, key, bad
):
    report, arguments = distinct_polynomial
    with pytest.raises(ValueError):
        _reconstruct_class_readout(
            replace(report, **{key: bad}),
            owner._admit_class_readout_inputs(**arguments),
            **_DISTINCT_SELECTORS,
        )


def test_evidence_does_not_reinterpret_distinct_events_under_default_policy(
    distinct_polynomial,
):
    report, arguments = distinct_polynomial
    with pytest.raises(ValueError, match="selectors differ"):
        _reconstruct_class_readout(
            report, owner._admit_class_readout_inputs(**arguments)
        )


def test_evidence_rejects_wrong_event_coordinate_and_cached_observer_increment(
    distinct_polynomial,
):
    report, arguments = distinct_polynomial
    admitted = owner._admit_class_readout_inputs(**arguments)
    segment = report.segments[4]
    wrong = list(segment.pre_event_box)
    wrong[4] += report.second_probe_amplitude
    altered = replace(segment, initial_box=tuple(wrong))
    with pytest.raises(ValueError, match="event changed another coordinate"):
        _reconstruct_class_readout(
            replace(
                report, segments=report.segments[:4] + (altered,) + report.segments[5:]
            ),
            admitted,
            **_DISTINCT_SELECTORS,
        )
    altered = replace(segment, completed_receiver_increment_bounds=I(42))
    with pytest.raises(ValueError, match="branch completion differs"):
        _reconstruct_class_readout(
            replace(
                report, segments=report.segments[:4] + (altered,) + report.segments[5:]
            ),
            admitted,
            **_DISTINCT_SELECTORS,
        )


class _HistoricalRecord:
    """Model the missing-key convention in retained lazy evidence views."""

    def __init__(self, values):
        self.values = values

    def __getattr__(self, key):
        return self.values[key]


@pytest.mark.parametrize("record_type", (SimpleNamespace, _HistoricalRecord))
def test_historical_evidence_without_selectors_is_legacy_only(
    polynomial_report, record_type
):
    values = {
        key: value
        for key, value in vars(polynomial_report).items()
        if key not in _DISTINCT_SELECTORS
    }
    old = (
        record_type(**values) if record_type is SimpleNamespace else record_type(values)
    )
    admitted = owner._admit_class_readout_inputs(**_arguments())
    assert _reconstruct_class_readout(old, admitted).complete
    with pytest.raises(ValueError, match="legacy evidence lacks selected nodes"):
        _reconstruct_class_readout(old, admitted, **_DISTINCT_SELECTORS)
    values["first_probe_node"] = 4
    partial = (
        record_type(**values) if record_type is SimpleNamespace else record_type(values)
    )
    with pytest.raises(ValueError, match="partial selector evidence"):
        _reconstruct_class_readout(partial, admitted)


def test_consumer_admits_policy_selectors_before_reading_report():
    with pytest.raises(ValueError, match="first_probe_node"):
        _reconstruct_class_readout(
            object(),
            owner._admit_class_readout_inputs(**_arguments()),
            first_probe_node=True,
        )
