"""Port-event carry, independent retained arithmetic and unreserved short flows."""

import json
import subprocess
from dataclasses import replace
from decimal import Decimal
from fractions import Fraction as Q
from inspect import Parameter, signature

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from tnfr.mathematics._rational_interval import I
from tnfr.physics import _sine_class_port_readout_evidence as evidence
from tnfr.physics import relational_sine_class_port_readout as owner
from tnfr.utils.io import json_loads


def _arguments(**updates):
    values = dict(
        initial_form_bounds=tuple(
            tuple((Q(i - 13 + c, 101), Q(i - 13 + c, 101)) for i in range(27))
            for c in (0, 2)
        ),
        initial_phase_bounds=tuple(
            tuple((Q(2 * i - 25 + c, 103), Q(2 * i - 25 + c, 103)) for i in range(27))
            for c in (0, 3)
        ),
        port_impulse=(Q(2, 7), Q(-3, 11), Q(1, 13)),
        horizon=Q(2, 7),
        time_step=Q(1, 5),
        order=3,
        max_steps=4,
    )
    values.update(updates)
    return values


def _constant_field(*_):
    def flow(state):
        return tuple(state[0] * 0 + Q(i + 1, 100) for i in range(54))

    return flow, lambda _: (Q(1),)


@pytest.fixture(scope="module", autouse=True)
def no_prediction_source_or_worker_execution():
    from tnfr.physics import (
        _sine_class_port_prediction,
        _sine_formed_contact,
        relational_sine_class_comparison_readout,
        relational_sine_class_cubic_response,
        relational_sine_class_readout,
    )

    def forbidden(*args, **kwargs):
        pytest.fail(
            "port readout controls cannot call predictions, old producers or workers"
        )

    with pytest.MonkeyPatch.context() as patch:
        for module, names in (
            (
                _sine_class_port_prediction,
                ("_predict_collective_port_response", "_causal_coefficients"),
            ),
            (_sine_formed_contact, ("_unprobed_handoff",)),
            (
                relational_sine_class_cubic_response,
                ("_class_cubic_coefficients", "_time_coefficients"),
            ),
            (relational_sine_class_readout, ("bound_sine_class_four_history_readout",)),
            (
                relational_sine_class_comparison_readout,
                ("bound_sine_class_comparison_readout",),
            ),
            (subprocess, ("run", "Popen", "check_output")),
        ):
            for name in names:
                patch.setattr(module, name, forbidden)
        yield


@pytest.fixture(scope="module")
def polynomial_report():
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(owner, "_full_sine_field", _constant_field)
        return owner.bound_sine_class_port_readout(**_arguments())


def _rebuild(report, **updates):
    return evidence._reconstruct_port_readout(
        report, owner._admit_port_readout_inputs(**_arguments(**updates))
    )


def test_seven_primitives_exclude_prediction_class_and_sensor_inputs():
    parameters = signature(owner.bound_sine_class_port_readout).parameters
    assert len(parameters) == 7 and set(parameters) == set(_arguments())
    assert all(
        p.kind == Parameter.KEYWORD_ONLY and p.default == Parameter.empty
        for p in parameters.values()
    )
    for key in (
        "mediator_class",
        "prediction",
        "readout_error_bound",
        "target",
        "source_handoff",
    ):
        with pytest.raises(TypeError):
            owner.bound_sine_class_port_readout(**_arguments(), **{key: 0})


@pytest.mark.parametrize("key", ("horizon", "time_step", "order", "max_steps"))
@pytest.mark.parametrize("invalid", (True, np.bool_(False), float("nan"), float("inf")))
def test_invalid_policy_primitives_precede_field(key, invalid, monkeypatch):
    monkeypatch.setattr(
        owner,
        "_full_sine_field",
        lambda *_: pytest.fail("invalid admission reached field"),
    )
    with pytest.raises((TypeError, ValueError)):
        owner.bound_sine_class_port_readout(**_arguments(**{key: invalid}))


@pytest.mark.parametrize(
    "key,value",
    (
        ("horizon", Q(-1)),
        ("horizon", Q(2) + Q(1, 2**400)),
        ("time_step", Q(0)),
        ("time_step", Q(2) + Q(1, 2**400)),
        ("order", Q(3)),
        ("order", np.int64(3)),
        ("order", 0),
        ("order", 17),
        ("max_steps", Q(4)),
        ("max_steps", 0),
        ("max_steps", 8193),
        ("port_impulse", (0, 0)),
        ("port_impulse", (0, 0, 0, 0)),
        ("port_impulse", (0, True, 0)),
        ("port_impulse", (0, float("nan"), 0)),
        ("port_impulse", (0, Decimal("1e-400"), 0)),
    ),
)
def test_domains_and_representation_reject_before_execution(key, value, monkeypatch):
    monkeypatch.setattr(
        owner,
        "_full_sine_field",
        lambda *_: pytest.fail("invalid admission reached field"),
    )
    with pytest.raises((TypeError, ValueError)):
        owner.bound_sine_class_port_readout(**_arguments(**{key: value}))


@pytest.mark.parametrize("channel", ("initial_form_bounds", "initial_phase_bounds"))
@pytest.mark.parametrize(
    "bad", (True, np.bool_(False), float("nan"), float("inf"), Decimal("1e-400"))
)
def test_both_sources_are_admitted_before_first_execution(channel, bad, monkeypatch):
    monkeypatch.setattr(
        owner,
        "_full_sine_field",
        lambda *_: pytest.fail("second source was not admitted first"),
    )
    arguments = _arguments()
    second = list(arguments[channel][1])
    second[-1] = (0, bad)
    arguments[channel] = (arguments[channel][0], second)
    with pytest.raises((TypeError, ValueError)):
        owner.bound_sine_class_port_readout(**arguments)


@pytest.mark.parametrize(
    "level",
    ("source_count", "coordinate_count", "pair_count", "interval_object", "reversed"),
)
def test_source_shapes_and_encoded_objects_cannot_bypass_admission(level, monkeypatch):
    monkeypatch.setattr(
        owner,
        "_full_sine_field",
        lambda *_: pytest.fail("invalid source reached field"),
    )
    args = _arguments()
    source = args["initial_form_bounds"]
    if level == "source_count":
        args["initial_form_bounds"] = source + (source[0],)
    elif level == "coordinate_count":
        args["initial_form_bounds"] = (source[0], source[1] + ((0, 0),))
    else:
        row = {"pair_count": (0, 0, 0), "interval_object": I(0), "reversed": (1, 0)}[
            level
        ]
        args["initial_form_bounds"] = (source[0], (row,) + source[1][1:])
    with pytest.raises((TypeError, ValueError)):
        owner.bound_sine_class_port_readout(**args)


def test_unbounded_source_iterator_is_consumed_only_to_shape_boundary(monkeypatch):
    monkeypatch.setattr(
        owner,
        "_full_sine_field",
        lambda *_: pytest.fail("oversized source reached field"),
    )
    seen = []

    def endless():
        while True:
            seen.append(1)
            yield (0, 0)

    forms = (_arguments()["initial_form_bounds"][0], endless())
    with pytest.raises(ValueError):
        owner.bound_sine_class_port_readout(**_arguments(initial_form_bounds=forms))
    assert len(seen) == 28


def test_fixed_partition_carries_every_coordinate_after_one_event(polynomial_report):
    report = polynomial_report
    assert report.admitted and report.source_order == (0, 1)
    assert (
        report.planned_step_count
        == report.attempted_step_count
        == report.completed_step_count
        == 4
    )
    assert (
        report.completed_source_count == 2 and report.unattempted_source_indices == ()
    )
    assert report.failed_source_index is None and report.unavailable_reasons == ()
    for index, history in enumerate(report.histories):
        before = report.initial_form_bounds[index] + report.initial_phase_bounds[index]
        assert history.pre_event_box == before
        for i, value in enumerate(before):
            impulse = (
                report.port_impulse[report.port_nodes.index(i)]
                if i in report.port_nodes
                else 0
            )
            assert history.initial_box[i] == (value + impulse if impulse else value)
        assert history.initial_box[27:] == before[27:]
        assert history.attempted_step_count == 2
        assert tuple(step.duration for step in history.steps) == (Q(1, 5), Q(3, 35))
        current, state = Q(0), history.initial_box
        for step in history.steps:
            assert step.time == current and step.initial_box == state
            assert len(step.endpoint) == 54
            assert step.picard_interior_margin > 0 and step.domain_lower_bounds == (
                Q(1),
            )
            state, current = step.endpoint, current + step.duration
        assert current == history.completed_time == report.horizon
        assert state == history.completed_endpoint_box == history.final_state_bounds
        for i, initial in enumerate(history.initial_box):
            drift = Q(i + 1, 100) * report.horizon
            assert state[i].contains(I(initial.lo + drift, initial.hi + drift))
        assert history.readout_bounds == state[13]
    assert report.endpoint_readout_bounds[0] != report.endpoint_readout_bounds[1]
    rebuilt = _rebuild(report)
    assert (
        rebuilt.complete and rebuilt.endpoint_bounds == report.endpoint_readout_bounds
    )


def test_kernel_failure_keeps_completed_first_source_and_stops_without_retry(
    monkeypatch,
):
    monkeypatch.setattr(owner, "_full_sine_field", _constant_field)
    kernel = owner.validated_box_taylor_step
    calls = []

    def fail_third(state, duration, flow, domain, **kwargs):
        calls.append((state, duration))
        if len(calls) == 3:
            return None, tuple(value + I(-1, 1) for value in state), "synthetic_failure"
        return kernel(state, duration, flow, domain, **kwargs)

    monkeypatch.setattr(owner, "validated_box_taylor_step", fail_third)
    report = owner.bound_sine_class_port_readout(**_arguments())
    assert len(calls) == report.attempted_step_count == 3
    assert report.completed_step_count == 2 and report.completed_source_count == 1
    assert report.failed_source_index == 1 and report.endpoint_readout_bounds is None
    failed = report.histories[1]
    assert failed.steps == () and failed.completed_time == failed.failed_time == 0
    assert failed.failed_initial_box == failed.initial_box == calls[-1][0]
    assert failed.completed_endpoint_box == failed.initial_box
    assert failed.final_state_bounds is failed.readout_bounds is None
    assert report.completed_source_readout_bounds == (
        (0, report.histories[0].readout_bounds),
    )
    rebuilt = _rebuild(report)
    assert not rebuilt.complete and rebuilt.attempted_step_count == 3


def test_first_failure_prevents_second_source_event(monkeypatch):
    monkeypatch.setattr(owner, "_full_sine_field", _constant_field)
    calls = []

    def fail(state, *args, **kwargs):
        calls.append(state)
        return None, state, None

    monkeypatch.setattr(owner, "validated_box_taylor_step", fail)
    report = owner.bound_sine_class_port_readout(**_arguments())
    assert len(calls) == 1 and report.failed_source_index == 0
    assert report.histories[0].reason == "shared_step_unavailable"
    assert report.histories[1].status == "not_attempted"
    assert report.histories[1].initial_box is report.histories[1].pre_event_box is None
    assert report.unattempted_source_indices == (1,)
    assert not _rebuild(report).complete


@pytest.mark.parametrize("cap", (1, 2))
def test_budget_exhaustion_distinguishes_carried_prefix_and_unapplied_event(
    monkeypatch, cap
):
    monkeypatch.setattr(owner, "_full_sine_field", _constant_field)
    report = owner.bound_sine_class_port_readout(**_arguments(max_steps=cap))
    assert report.attempted_step_count == report.completed_step_count == cap
    assert report.endpoint_readout_bounds is None
    failed = report.histories[report.failed_source_index]
    assert failed.status == "budget_exhausted" and failed.failed_tube is None
    if cap == 1:
        assert failed.completed_time == Q(1, 5)
        assert (
            failed.failed_initial_box
            == failed.completed_endpoint_box
            == failed.steps[-1].endpoint
        )
        assert failed.failed_time == failed.completed_time
    else:
        assert (
            failed.pre_event_box is failed.initial_box is failed.completed_time is None
        )
        assert failed.failed_initial_box is None and failed.steps == ()
        assert report.completed_source_count == 1
    assert report.unattempted_source_indices == (1,)
    assert not _rebuild(report, max_steps=cap).complete


def test_zero_horizon_retains_absolute_phase_and_exact_event_without_kernel(
    monkeypatch,
):
    monkeypatch.setattr(owner, "_full_sine_field", _constant_field)
    monkeypatch.setattr(
        owner,
        "validated_box_taylor_step",
        lambda *_a, **_k: pytest.fail("zero horizon called flow"),
    )
    phases = tuple(
        tuple((Q(12 + i + c), Q(12 + i + c)) for i in range(27)) for c in (0, 1)
    )
    report = owner.bound_sine_class_port_readout(
        **_arguments(horizon=0, max_steps=1, initial_phase_bounds=phases)
    )
    assert (
        report.admitted
        and report.planned_step_count == report.attempted_step_count == 0
    )
    for index, history in enumerate(report.histories):
        assert history.final_state_bounds[27:] == tuple(
            I(*pair) for pair in phases[index]
        )
        assert history.readout_bounds == history.initial_box[13]
    assert _rebuild(
        report, horizon=0, max_steps=1, initial_phase_bounds=phases
    ).complete


def test_exact_subgrid_event_and_time_are_not_zeroed(monkeypatch):
    monkeypatch.setattr(owner, "_full_sine_field", _constant_field)
    tiny = Q(1, 2**500)
    updates = dict(port_impulse=(0, tiny, -tiny), horizon=tiny, time_step=tiny)
    report = owner.bound_sine_class_port_readout(**_arguments(**updates))
    assert report.port_impulse == (0, tiny, -tiny) and report.horizon == tiny
    assert report.attempted_step_count == 2
    assert report.histories[0].steps[0].duration == tiny
    assert _rebuild(report, **updates).complete


@pytest.mark.parametrize(
    "field,value",
    (
        ("clock", "t"),
        ("source_coordinates", "phase residual y"),
        ("state_order", tuple(f"wrong_{i}" for i in range(54))),
        ("port_nodes", (4, 12, 22)),
        ("readout_node", 22),
        ("source_order", (False, 1)),
        ("horizon", True),
        ("order", True),
        ("max_steps", True),
        ("attempted_step_count", 5),
        ("completed_source_count", 1),
        ("failed_source_index", False),
        ("unattempted_source_indices", (1,)),
        ("endpoint_readout_bounds", (I(0), I(0))),
        ("status", "unavailable"),
        ("capacity", (True,) + (Q(1),) * 26),
    ),
)
def test_reader_rejects_tampered_policy_and_cached_observations(
    polynomial_report, field, value
):
    with pytest.raises((TypeError, ValueError)):
        _rebuild(replace(polynomial_report, **{field: value}))


@pytest.mark.parametrize(
    "field,value",
    (
        ("source_index", False),
        ("attempted_step_count", 3),
        ("completed_time", 0),
        ("readout_bounds", I(0)),
        ("failed_time", Q(0)),
        ("reason", "fake_failure"),
    ),
)
def test_reader_rejects_tampered_history_association(polynomial_report, field, value):
    history = replace(polynomial_report.histories[0], **{field: value})
    report = replace(
        polynomial_report, histories=(history, polynomial_report.histories[1])
    )
    with pytest.raises((TypeError, ValueError)):
        _rebuild(report)


@pytest.mark.parametrize(
    "change",
    (
        "increment",
        "endpoint",
        "series",
        "source",
        "tube",
        "remainder",
        "time",
        "duration",
        "order",
        "margin",
        "domain",
    ),
)
def test_reader_rebuilds_step_arithmetic_and_rejects_forged_cached_step(
    polynomial_report, change
):
    history = polynomial_report.histories[0]
    step = history.steps[0]
    if change in ("increment", "endpoint", "source", "tube", "remainder"):
        field = {"source": "initial_box", "remainder": "local_remainder_bounds"}.get(
            change, change
        )
        values = list(getattr(step, field))
        values[13] = values[13] + 1
        bad = replace(step, **{field: tuple(values)})
    elif change == "series":
        rows = list(step.series)
        row = list(rows[13])
        row[1] += 1
        rows[13] = tuple(row)
        bad = replace(step, series=tuple(rows))
    else:
        field, value = {
            "time": ("time", True),
            "duration": ("duration", Q(1, 9)),
            "order": ("order", True),
            "margin": ("picard_interior_margin", True),
            "domain": ("domain_lower_bounds", (True,)),
        }[change]
        bad = replace(step, **{field: value})
    replaced_history = replace(history, steps=(bad,) + history.steps[1:])
    report = replace(
        polynomial_report, histories=(replaced_history, polynomial_report.histories[1])
    )
    with pytest.raises((TypeError, ValueError, ArithmeticError)):
        _rebuild(report)


def test_reader_rejects_changed_support_law_and_source(polynomial_report):
    bad_geometry = replace(
        polynomial_report.geometry, edges=polynomial_report.geometry.edges[:-1]
    )
    bad_model = replace(polynomial_report.reference_model, phase_weight=Q(1, 512))
    forms = list(polynomial_report.initial_form_bounds)
    source = list(forms[1])
    source[13] += 1
    forms[1] = tuple(source)
    for report in (
        replace(polynomial_report, geometry=bad_geometry),
        replace(polynomial_report, reference_model=bad_model),
        replace(polynomial_report, initial_form_bounds=tuple(forms)),
    ):
        with pytest.raises(ValueError):
            _rebuild(report)


def test_reader_checks_partial_count_and_no_continuation(monkeypatch):
    monkeypatch.setattr(owner, "_full_sine_field", _constant_field)
    monkeypatch.setattr(
        owner,
        "validated_box_taylor_step",
        lambda state, *_a, **_k: (None, state, "failure"),
    )
    report = owner.bound_sine_class_port_readout(**_arguments())
    for changed in (
        replace(report, attempted_step_count=2),
        replace(
            report,
            histories=(
                replace(report.histories[0], attempted_step_count=0),
                report.histories[1],
            ),
        ),
        replace(
            report,
            histories=(
                report.histories[0],
                replace(
                    report.histories[1], pre_event_box=report.histories[0].pre_event_box
                ),
            ),
        ),
        replace(report, endpoint_readout_bounds=(I(0), I(0))),
    ):
        with pytest.raises(ValueError):
            _rebuild(changed)


def test_serialization_retains_exact_step_evidence_and_partial_unavailability(
    polynomial_report, monkeypatch
):
    payload = polynomial_report.to_dict()
    assert payload["schema"] == "tnfr.sine-class-port-readout.v1"
    assert json_loads(json.dumps(payload, allow_nan=False)) == payload
    duration = payload["report"]["histories"][0]["steps"][0]["duration"]
    assert Q(duration["numerator"], duration["denominator"]) == Q(1, 5)
    monkeypatch.setattr(owner, "_full_sine_field", _constant_field)
    partial = owner.bound_sine_class_port_readout(**_arguments(max_steps=1)).to_dict()
    assert partial["report"]["endpoint_readout_bounds"] is None
    assert partial["report"]["histories"][1]["initial_box"] is None


def test_unreserved_short_sine_flows_cover_independent_both_row_solution():
    radius = Q(1, 2**35)
    forms = tuple(
        tuple(
            (
                Q((i * 5 + c) % 13 - 6, 1000) - radius,
                Q((i * 5 + c) % 13 - 6, 1000) + radius,
            )
            for i in range(27)
        )
        for c in (1, 3)
    )
    phases = tuple(
        tuple(
            (
                Q((i * 7 + c) % 17 - 8, 100) - radius,
                Q((i * 7 + c) % 17 - 8, 100) + radius,
            )
            for i in range(27)
        )
        for c in (2, 5)
    )
    impulse = (Q(1, 1300), Q(-1, 1700), Q(1, 1900))
    args = _arguments(
        initial_form_bounds=forms,
        initial_phase_bounds=phases,
        port_impulse=impulse,
        horizon=Q(1, 512),
        time_step=Q(1, 1024),
        order=4,
    )
    report = owner.bound_sine_class_port_readout(**args)
    assert report.admitted
    edges = tuple(
        (9 * c + j, 9 * c + (j + 1) % 9) for c in range(3) for j in range(9)
    ) + ((4, 13), (13, 22))
    degree = np.array([sum(i in edge for edge in edges) for i in range(27)])
    gamma = 1 / (1023 * np.pi)

    def rates(_, state):
        form_rate, phase_rate = np.zeros(27), np.zeros(27)
        for i, j in edges:
            dx = state[j] - state[i]
            current = dx + gamma * np.sin(state[27 + j] - state[27 + i])
            form_rate[i] += current / degree[i]
            form_rate[j] -= current / degree[j]
            phase_rate[i] -= gamma * dx / degree[i]
            phase_rate[j] += gamma * dx / degree[j]
        return np.concatenate((form_rate, phase_rate))

    for index, history in enumerate(report.histories):
        initial = np.array(
            [float((lo + hi) / 2) for lo, hi in forms[index] + phases[index]]
        )
        for node, jump in zip((4, 13, 22), impulse):
            initial[node] += float(jump)
        solved = solve_ivp(
            rates,
            (0, float(args["horizon"])),
            initial,
            method="DOP853",
            rtol=1e-12,
            atol=1e-14,
        )
        assert solved.success
        assert all(
            bound.contains(Q.from_float(float(value)))
            for bound, value in zip(history.final_state_bounds, solved.y[:, -1])
        )
        derivative = rates(0, initial)
        assert abs(degree @ derivative[:27]) < 1e-16
        assert abs(degree @ derivative[27:]) < 1e-18
    admitted = owner._admit_port_readout_inputs(**args)
    rebuilt = evidence._reconstruct_port_readout(report, admitted)
    assert (
        rebuilt.complete and rebuilt.endpoint_bounds == report.endpoint_readout_bounds
    )
