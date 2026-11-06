"""Composition and evidence admission with independent polynomial fields only.

No selected class source, analytic coefficient, formation or reserved nonlinear
response is evaluated. The toy field tests complete carry and observation
arithmetic; it does not establish the producer's sine derivative premise.
"""

import json
import subprocess
from dataclasses import replace
from decimal import Decimal
from fractions import Fraction as Q
from inspect import Parameter, signature

import numpy as np
import pytest

from tnfr.mathematics._rational_interval import I
from tnfr.physics import relational_sine_class_comparison_readout as owner
from tnfr.physics import relational_sine_class_readout as single
from tnfr.physics._sine_class_readout_evidence import _reconstruct_class_readout
from tnfr.sdk.relational_reports import relational_report_to_dict
from tnfr.utils.io import json_loads


def _arguments():
    sources = []
    for donor in (Q(1, 16), Q(-3, 32)):
        form = [(Q(i - 13, 128),) * 2 for i in range(27)]
        form[4], form[22] = (donor, donor), (Q(-2), Q(3))
        sources.append(tuple(form))
    return dict(
        initial_form_bounds=tuple(sources),
        initial_phase_bounds=tuple(
            tuple((Q(i + shift, 64),) * 2 for i in range(27)) for shift in (1, 2)
        ),
        first_probe_amplitude=Q(1, 8),
        second_probe_amplitude=Q(-1, 16),
        delay=Q(1, 8),
        total_duration=Q(3, 8),
        time_step=Q(1, 8),
        order=3,
        max_steps=20,
    )


def _polynomial_field(*_):
    def flow(state):
        zero = state[0] * 0
        return tuple(state[4] ** 3 if i == 22 else zero for i in range(54))

    return flow, lambda _: (Q(1),)


@pytest.fixture(scope="module", autouse=True)
def no_source_or_coefficient_execution():
    from tnfr.physics import (
        _sine_flow,
        _sine_formed_contact,
        relational_sine_class_cubic_response,
        relational_sine_class_nonlinear_protocol,
    )

    def forbidden(*args, **kwargs):
        pytest.fail("comparison admission must not evaluate a research response")

    with pytest.MonkeyPatch.context() as patch:
        for module, names in (
            (_sine_flow, ("_full_sine_field",)),
            (_sine_formed_contact, ("_unprobed_handoff",)),
            (
                relational_sine_class_cubic_response,
                ("bound_sine_class_cubic_response", "_class_cubic_coefficients"),
            ),
            (
                relational_sine_class_nonlinear_protocol,
                ("bound_sine_class_nonlinear_protocol", "_heat_cubic_channels"),
            ),
            (subprocess, ("run", "Popen", "check_output")),
        ):
            for name in names:
                patch.setattr(module, name, forbidden)
        patch.setattr(single, "_full_sine_field", _polynomial_field)
        yield


@pytest.fixture(scope="module")
def report():
    return owner.bound_sine_class_comparison_readout(**_arguments())


def test_nine_primitives_exclude_verdicts_predictions_and_sensors():
    parameters = signature(owner.bound_sine_class_comparison_readout).parameters
    assert set(parameters) == set(_arguments()) and len(parameters) == 9
    assert all(
        p.kind == Parameter.KEYWORD_ONLY and p.default == Parameter.empty
        for p in parameters.values()
    )
    for key in (
        "report",
        "mediator_classes",
        "prediction",
        "endpoint_radius",
        "readout_error_bound",
    ):
        with pytest.raises(TypeError):
            owner.bound_sine_class_comparison_readout(**_arguments(), **{key: 0})


@pytest.mark.parametrize("channel", ("initial_form_bounds", "initial_phase_bounds"))
@pytest.mark.parametrize(
    "bad", (True, np.bool_(False), float("nan"), float("inf"), Decimal("1e-400"))
)
def test_second_source_is_admitted_before_first_execution(channel, bad, monkeypatch):
    monkeypatch.setattr(
        owner,
        "bound_sine_class_four_history_readout",
        lambda **_: pytest.fail("early flow"),
    )
    values = _arguments()
    covers = list(values[channel])
    second = list(covers[1])
    second[26] = (0, bad)
    covers[1] = second
    values[channel] = covers
    with pytest.raises((TypeError, ValueError)):
        owner.bound_sine_class_comparison_readout(**values)


@pytest.mark.parametrize("key", tuple(_arguments())[2:])
@pytest.mark.parametrize("bad", (True, np.bool_(True), float("nan")))
def test_shared_scalar_admission_precedes_all_execution(key, bad, monkeypatch):
    monkeypatch.setattr(
        owner,
        "bound_sine_class_four_history_readout",
        lambda **_: pytest.fail("early flow"),
    )
    with pytest.raises((TypeError, ValueError)):
        owner.bound_sine_class_comparison_readout(**{**_arguments(), key: bad})


@pytest.mark.parametrize(
    "change",
    (
        {"initial_form_bounds": ()},
        {"initial_phase_bounds": ((),)},
        {"initial_form_bounds": ((), (), ())},
        {"max_steps": 0},
        {"max_steps": 8193},
        {"max_steps": Q(20)},
        {"order": Q(3)},
        {"order": 17},
        {"time_step": 0},
        {"delay": Q(1, 2)},
        {"total_duration": 3},
    ),
)
def test_order_shape_and_resource_domains(change, monkeypatch):
    monkeypatch.setattr(
        owner,
        "bound_sine_class_four_history_readout",
        lambda **_: pytest.fail("early flow"),
    )
    with pytest.raises((TypeError, ValueError)):
        owner.bound_sine_class_comparison_readout(**{**_arguments(), **change})


def test_independent_polynomial_contrast_and_common_prefix_cancellation(report):
    assert report.admitted and report.completed_source_count == 2
    assert (
        report.planned_step_count
        == report.attempted_step_count
        == report.completed_step_count
        == 20
    )
    a, b, duration = Q(1, 8), Q(-1, 16), Q(1, 4)
    expected = []
    for donor, child, rebuilt in zip(
        (Q(1, 16), Q(-3, 32)), report.readout_reports, report.reconstructed_observations
    ):
        value = 3 * a * b * (2 * donor + a + b) * duration
        expected.append(value)
        assert rebuilt.mixed_bounds.lo <= value <= rebuilt.mixed_bounds.hi
        assert rebuilt.mixed_bounds.hi - rebuilt.mixed_bounds.lo < Q(1, 10**30)
        assert (
            child.raw_endpoint_mixed_bounds.hi - child.raw_endpoint_mixed_bounds.lo > 10
        )
        assert rebuilt.complete and rebuilt.attempted_step_count == 10
        assert len(child.segments) == 6
        for segment in child.segments:
            before = (
                child.source_box
                if segment.parent_segment_index is None
                else child.segments[segment.parent_segment_index].final_state_bounds
            )
            assert segment.pre_event_box == before
            assert segment.initial_box == tuple(
                value + segment.form_jump if i == 4 and segment.form_jump else value
                for i, value in enumerate(before)
            )
    exact = expected[0] - expected[1]
    assert exact == 6 * a * b * duration * (Q(1, 16) + Q(3, 32))
    assert (
        report.contrast_readout_bounds.lo <= exact <= report.contrast_readout_bounds.hi
    )
    assert report.raw_endpoint_contrast_bounds.lo < report.contrast_readout_bounds.lo
    assert report.raw_endpoint_contrast_bounds.hi > report.contrast_readout_bounds.hi


@pytest.mark.parametrize(
    "cap,counts,completed,unattempted",
    (
        (1, (1, 0), 0, (1,)),
        (10, (10, 0), 1, (1,)),
        (11, (10, 1), 1, ()),
    ),
)
def test_global_budget_retains_available_child_and_stops_later_events(
    cap, counts, completed, unattempted
):
    result = owner.bound_sine_class_comparison_readout(
        **{**_arguments(), "max_steps": cap}
    )
    assert result.status == "unavailable" and result.completed_source_count == completed
    assert result.attempted_step_count == cap
    assert result.class_mixed_readout_bounds is None
    assert (
        result.contrast_readout_bounds is None
        and result.raw_endpoint_contrast_bounds is None
    )
    assert result.unattempted_source_indices == unattempted
    assert (
        tuple(
            child.attempted_step_count if child is not None else 0
            for child in result.readout_reports
        )
        == counts
    )
    if cap == 10:
        assert result.readout_reports[0].endpoint_readout_bounds is not None
        assert result.readout_reports[1] is None
        assert result.failed_source_index == 1
    else:
        child = result.readout_reports[0 if cap == 1 else 1]
        assert child.segments[1].initial_box is None
        assert child.segments[1].status == "budget_exhausted"


@pytest.mark.parametrize("failure_call", (1, 12))
def test_shared_kernel_failure_preserves_failed_source_and_no_later_call(
    failure_call, monkeypatch
):
    original = single.validated_box_taylor_step
    calls = []

    def step(box, *args, **kwargs):
        calls.append(box)
        if len(calls) == failure_call:
            return None, tuple(box), "independent_mock_failure"
        return original(box, *args, **kwargs)

    monkeypatch.setattr(single, "validated_box_taylor_step", step)
    result = owner.bound_sine_class_comparison_readout(**_arguments())
    assert len(calls) == result.attempted_step_count == failure_call
    assert result.completed_step_count == failure_call - 1
    assert result.completed_source_count == int(failure_call > 10)
    child = result.readout_reports[result.failed_source_index]
    failed = child.segments[child.failed_segment_index]
    assert failed.failed_initial_box == calls[-1] == failed.failed_tube
    assert result.contrast_readout_bounds is None
    assert all(
        child is None
        for child in result.readout_reports[result.failed_source_index + 1 :]
    )


def test_zero_suffix_is_exactly_zero_without_dropping_source_history(monkeypatch):
    values = {**_arguments(), "delay": Q(3, 8), "max_steps": 12}
    result = owner.bound_sine_class_comparison_readout(**values)
    assert result.admitted and result.attempted_step_count == 12
    assert result.contrast_readout_bounds == I(0)
    assert result.class_mixed_readout_bounds == (I(0), I(0))
    assert all(
        child.endpoint_readout_bounds is not None for child in result.readout_reports
    )


def test_zero_total_applies_both_form_events_without_kernel(monkeypatch):
    monkeypatch.setattr(
        single,
        "validated_box_taylor_step",
        lambda *_a, **_k: pytest.fail("zero window"),
    )
    result = owner.bound_sine_class_comparison_readout(
        **{**_arguments(), "delay": 0, "total_duration": 0, "max_steps": 1}
    )
    assert result.admitted and result.attempted_step_count == 0
    assert result.contrast_readout_bounds == I(0)
    assert all(
        child.segments[-1].form_jump == Q(-1, 16) for child in result.readout_reports
    )


@pytest.mark.parametrize(
    "tamper",
    (
        "mixed",
        "increment",
        "endpoint",
        "parent",
        "clock",
        "capacity",
        "count",
        "history",
        "event",
        "margin",
        "support",
        "source",
    ),
)
def test_consumer_rebuilds_primitives_and_rejects_cached_forgery(report, tamper):
    child = report.readout_reports[0]
    if tamper == "mixed":
        child = replace(child, mixed_readout_bounds=I(42))
    elif tamper == "clock":
        child = replace(child, clock="seconds")
    elif tamper == "capacity":
        child = replace(child, capacity=(True,) + child.capacity[1:])
    elif tamper == "count":
        child = replace(child, completed_step_count=True)
    elif tamper == "history":
        child = replace(
            child,
            history_segment_indices=((False, 2),) + child.history_segment_indices[1:],
        )
    elif tamper == "event":
        child = replace(
            child,
            history_event_amplitudes=((True, 0),) + child.history_event_amplitudes[1:],
        )
    elif tamper == "support":
        child = replace(child, degrees=(True,) + child.degrees[1:])
    elif tamper == "source":
        child = replace(child, source_box=(I(42),) + child.source_box[1:])
    else:
        segment = child.segments[2]
        if tamper == "parent":
            segment = replace(segment, parent_segment_index=False)
        else:
            step = segment.steps[0]
            if tamper == "margin":
                step = replace(step, picard_interior_margin=True)
            else:
                key = "increment" if tamper == "increment" else "endpoint"
                row = list(getattr(step, key))
                row[22] = I(42)
                step = replace(step, **{key: tuple(row)})
            segment = replace(segment, steps=(step,) + segment.steps[1:])
        child = replace(
            child, segments=child.segments[:2] + (segment,) + child.segments[3:]
        )
    values = _arguments()
    admitted = single._admit_class_readout_inputs(
        **{
            **values,
            "initial_form_bounds": values["initial_form_bounds"][0],
            "initial_phase_bounds": values["initial_phase_bounds"][0],
        }
    )
    with pytest.raises((TypeError, ValueError)):
        _reconstruct_class_readout(child, admitted)


def test_exact_sdk_json_and_partial_availability(report):
    encoded = relational_report_to_dict(report)
    assert encoded["report_type"] == "SineClassComparisonReadout"
    assert encoded["report"] == report.to_dict()["report"]
    assert report.to_dict()["schema"] == "tnfr.sine-class-comparison-readout.v1"
    decoded = json_loads(json.dumps(encoded, allow_nan=False))
    assert decoded["report"]["first_probe_amplitude"] == {
        "numerator": 1,
        "denominator": 8,
    }
    result = owner.bound_sine_class_comparison_readout(
        **{**_arguments(), "max_steps": 10}
    )
    data = result.to_dict()["report"]
    assert data["readout_reports"][0] is not None and data["readout_reports"][1] is None
    assert data["contrast_readout_bounds"] is None
