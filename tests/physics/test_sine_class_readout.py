"""Primitive admission, carried branch ancestry and bounded partial evidence.

All executed flows use independent constant polynomial controls. Neither the
reserved class-two design nor a formation/analytic response is evaluated.
"""

import json
import subprocess
from decimal import Decimal
from fractions import Fraction as Q
from inspect import Parameter, signature

import numpy as np
import pytest

from tnfr.mathematics._rational_interval import I
from tnfr.physics import relational_sine_class_readout as owner
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


def test_nine_mandatory_primitives_exclude_target_prediction_and_sensor():
    parameters = signature(owner.bound_sine_class_four_history_readout).parameters
    assert set(parameters) == set(_arguments()) and len(parameters) == 9
    assert all(
        p.kind == Parameter.KEYWORD_ONLY and p.default == Parameter.empty
        for p in parameters.values()
    )
    for key in (
        "mediator_class",
        "source_handoff",
        "readout_error_bound",
        "prediction",
        "gain",
    ):
        with pytest.raises(TypeError):
            owner.bound_sine_class_four_history_readout(**_arguments(), **{key: 0})


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


@pytest.mark.parametrize("failure_call", (1, 2, 4, 6))
def test_first_failed_step_stops_later_events_and_retains_partial_evidence(
    failure_call, monkeypatch
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
    report = owner.bound_sine_class_four_history_readout(**_small_arguments())
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
            segment.initial_box[4]
            == segment.pre_event_box[4] + report.second_probe_amplitude
        )
        assert segment.initial_box[27:] == segment.pre_event_box[27:]


@pytest.mark.parametrize("budget", (1, 2, 3, 5))
def test_global_work_cap_stops_before_the_next_event(budget, monkeypatch):
    monkeypatch.setattr(owner, "_full_sine_field", _constant_field)
    report = owner.bound_sine_class_four_history_readout(
        **_small_arguments(max_steps=budget)
    )
    assert report.attempted_step_count == report.completed_step_count == budget
    assert report.failed_segment_index == budget
    failed = report.segments[budget]
    assert failed.status == "budget_exhausted"
    assert failed.initial_box is failed.pre_event_box is None
    assert failed.failed_tube is None and failed.completed_time is None
    assert failed.reason == "unique_step_budget_exhausted_before_event"
    assert not report.admitted and report.mixed_readout_bounds is None


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
