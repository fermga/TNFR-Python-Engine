"""Shared-kernel storage wiring, full-field algebra and retained evidence checks."""

import subprocess
from dataclasses import replace
from fractions import Fraction as Q
from inspect import Parameter, signature

import mpmath
import numpy as np
import pytest
from scipy.integrate import solve_ivp

from tests.physics.test_sine_class_neighbor_nonadditivity import _geometry
from tnfr.mathematics._interval_taylor import Jet
from tnfr.mathematics._rational_interval import I
from tnfr.physics import _sine_class_storage_readout_evidence as evidence
from tnfr.physics import relational_sine_class_storage_readout as owner
from tnfr.utils.io import json_loads


def _arguments(**updates):
    values = dict(
        mediator_class=1,
        initial_form_bounds=tuple((Q(i - 13, 128),) * 2 for i in range(27)),
        initial_phase_deviation_bounds=tuple(
            (Q((i * 7) % 11 - 5, 256),) * 2 for i in range(27)
        ),
        donor_amplitude=Q(1, 16),
        receiver_amplitude=Q(-1, 32),
        horizon=Q(1, 8),
        time_step=Q(1, 16),
        order=3,
        max_steps=16,
    )
    values.update(updates)
    return values


def _polynomial_fields(*_):
    def make(full):
        def field(state):
            zero = state[0] * 0
            rates = tuple(
                zero if i in (4, 22) else zero + Q(i + 1, 1024) for i in range(54)
            )
            loss = (state[4] + (2 if full else -1) * state[22]) ** 2
            return rates + (loss,)

        return field

    return (make(True), make(False)), lambda _: (Q(1),)


@pytest.fixture(scope="module", autouse=True)
def no_old_producers_or_selected_response():
    from tnfr.physics import (
        _sine_class_port_prediction,
        _sine_formed_contact,
        relational_sine_class_cubic_response,
        relational_sine_class_port_readout,
        relational_sine_class_readout,
    )

    def forbidden(*args, **kwargs):
        pytest.fail(
            "storage admission must not execute old producers or selected science"
        )

    original = owner.bound_sine_class_storage_readout

    def nonreserved_only(**kwargs):
        if (
            kwargs.get("donor_amplitude") == Q(7, 10000)
            and kwargs.get("receiver_amplitude") == Q(7, 10000)
            and kwargs.get("horizon") == Q(1, 8)
        ):
            forbidden()
        return original(**kwargs)

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
            (relational_sine_class_port_readout, ("bound_sine_class_port_readout",)),
            (relational_sine_class_readout, ("bound_sine_class_four_history_readout",)),
            (subprocess, ("run", "Popen", "check_output")),
        ):
            for name in names:
                patch.setattr(module, name, forbidden)
        patch.setattr(owner, "bound_sine_class_storage_readout", nonreserved_only)
        yield original


@pytest.fixture(scope="module")
def polynomial_report():
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(owner, "_storage_loss_fields", _polynomial_fields)
        return owner.bound_sine_class_storage_readout(**_arguments())


def _rebuild(report, **updates):
    return evidence._reconstruct_storage_readout(
        report, owner._admit_storage_readout_inputs(**_arguments(**updates))
    )


def test_nine_primitive_signature_has_no_prediction_sensor_or_cached_report(
    no_old_producers_or_selected_response,
):
    parameters = signature(no_old_producers_or_selected_response).parameters
    assert set(parameters) == set(_arguments()) and len(parameters) == 9
    assert all(
        p.kind == Parameter.KEYWORD_ONLY and p.default is Parameter.empty
        for p in parameters.values()
    )
    for key in ("prediction", "readout_error_bound", "source_certificate", "target"):
        with pytest.raises(TypeError):
            no_old_producers_or_selected_response(**_arguments(), **{key: 0})


def test_shared_polynomial_steps_carry_complete_states_and_passive_baselines(
    polynomial_report,
):
    result = polynomial_report
    assert result.admitted
    assert (
        result.attempted_step_count
        == result.completed_step_count
        == result.planned_step_count
        == 16
    )
    assert result.completed_history_count == 8
    a, b, h = result.donor_amplitude, result.receiver_amplitude, result.horizon
    for index, history in enumerate(result.histories):
        assert history.history_index == index
        assert history.initial_box[54] == I(0)
        assert history.initial_box[27:54] == history.pre_event_box[27:54]
        first, second = history.steps
        assert second.initial_box == first.endpoint
        assert second.initial_box[54] != I(0)
        accumulated = first.increment[54] + second.increment[54]
        assert history.loss_integral_bounds == accumulated
        assert history.completed_loss_increment_bounds == accumulated
        assert history.final_state_bounds == second.endpoint
        assert len(first.initial_box) == len(second.endpoint) == 55
        x4, x22 = (
            result.initial_form_bounds[i].lo + history.event_amplitudes[j]
            for j, i in enumerate((4, 22))
        )
        expected = h * (x4 + (2 if index % 2 == 0 else -1) * x22) ** 2
        assert accumulated.contains(expected)
        # Repeated endpoint sums would count the first-step baseline twice.
        assert (first.endpoint[54] + second.endpoint[54]).lo > accumulated.hi
    assert result.integrated_excess_loss_bounds.contains(6 * a * b * h)
    assert result.excess_storage_bounds.contains(-6 * a * b * h)
    assert _rebuild(result).excess_storage_bounds == result.excess_storage_bounds


def test_wound_target_is_added_only_to_full_model_and_source_once(polynomial_report):
    result = polynomial_report
    assert (
        result.target_phase_bounds[4]
        == result.target_phase_bounds[13]
        == result.target_phase_bounds[22]
        == I(0)
    )
    assert result.target_phase_bounds[0].hi < -2
    assert result.tangent_source_box[27:54] == result.initial_phase_deviation_bounds
    assert result.full_source_box[27:54] == tuple(
        t + y
        for t, y in zip(
            result.target_phase_bounds, result.initial_phase_deviation_bounds
        )
    )
    for index, history in enumerate(result.histories):
        assert history.pre_event_box == (
            result.full_source_box if index % 2 == 0 else result.tangent_source_box
        )
        a, b = history.event_amplitudes
        for i, (before, after) in enumerate(
            zip(history.pre_event_box, history.initial_box)
        ):
            assert after == (
                before + a if i == 4 and a else before + b if i == 22 and b else before
            )


@pytest.mark.parametrize(
    "key,value",
    (
        ("mediator_class", True),
        ("mediator_class", 1.0),
        ("mediator_class", 0),
        ("donor_amplitude", True),
        ("receiver_amplitude", float("nan")),
        ("horizon", True),
        ("horizon", -1),
        ("horizon", Q(2) + Q(1, 2**300)),
        ("time_step", 0),
        ("time_step", 3),
        ("order", True),
        ("order", Q(3)),
        ("order", 17),
        ("max_steps", 0),
        ("max_steps", 8193),
        ("max_steps", np.int64(16)),
        ("initial_form_bounds", ((0, 0),) * 26),
        ("initial_phase_deviation_bounds", ((0, 0),) * 28),
        ("initial_form_bounds", ((0, 0, 0),) + ((0, 0),) * 26),
        ("initial_phase_deviation_bounds", ((0, True),) + ((0, 0),) * 26),
        ("initial_phase_deviation_bounds", ((1, 0),) + ((0, 0),) * 26),
    ),
)
def test_invalid_primitives_precede_target_field_and_solver(key, value, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("invalid primitive reached scientific construction")

    monkeypatch.setattr(owner, "_cubic_parameters", forbidden)
    monkeypatch.setattr(owner, "validated_box_taylor_step", forbidden)
    with pytest.raises((TypeError, ValueError)):
        owner.bound_sine_class_storage_readout(**_arguments(**{key: value}))


def test_oversized_source_iterator_is_bounded_before_field(monkeypatch):
    yielded = []

    def endless():
        while True:
            yielded.append(1)
            if len(yielded) > 28:
                pytest.fail("unbounded source iterator consumption")
            yield (0, 0)

    monkeypatch.setattr(
        owner,
        "_cubic_parameters",
        lambda *_: pytest.fail("oversized source reached field"),
    )
    with pytest.raises(ValueError):
        owner.bound_sine_class_storage_readout(
            **_arguments(initial_form_bounds=endless())
        )
    assert len(yielded) == 28


@pytest.mark.parametrize("cap", (1, 2, 3))
def test_global_budget_retains_prefixes_without_later_events(cap, monkeypatch):
    monkeypatch.setattr(owner, "_storage_loss_fields", _polynomial_fields)
    result = owner.bound_sine_class_storage_readout(**_arguments(max_steps=cap))
    assert result.attempted_step_count == result.completed_step_count == cap
    assert result.completed_history_count == cap // 2
    failure = result.histories[result.failed_history_index]
    if cap % 2:
        assert failure.completed_time == Q(1, 16)
        assert failure.completed_loss_increment_bounds is not None
        assert failure.failed_initial_box == failure.completed_endpoint_box
    else:
        assert failure.initial_box is failure.pre_event_box is None
        assert failure.reason == "total_step_budget_exhausted_before_event"
    assert all(
        row.initial_box is None
        for row in result.histories[result.failed_history_index + 1 :]
    )
    assert (
        result.loss_integral_bounds
        is result.raw_endpoint_loss_bounds
        is result.integrated_excess_loss_bounds
        is result.excess_storage_bounds
        is None
    )
    rebuilt = _rebuild(result, max_steps=cap)
    assert not rebuilt.complete
    assert rebuilt.completed_history_count == cap // 2


def test_first_failed_kernel_stops_all_work_and_retains_completed_individual_losses(
    monkeypatch,
):
    monkeypatch.setattr(owner, "_storage_loss_fields", _polynomial_fields)
    shared = owner.validated_box_taylor_step
    calls = []

    def fail_fourth(state, duration, flow, domain, **kwargs):
        calls.append((state, kwargs["time"]))
        if len(calls) == 4:
            return None, tuple(state), "synthetic_failure"
        return shared(state, duration, flow, domain, **kwargs)

    monkeypatch.setattr(owner, "validated_box_taylor_step", fail_fourth)
    result = owner.bound_sine_class_storage_readout(**_arguments())
    assert result.attempted_step_count == 4 and result.completed_step_count == 3
    assert result.completed_history_count == 1
    assert result.completed_history_loss_bounds == (
        (0, result.histories[0].loss_integral_bounds),
    )
    assert result.failed_history_index == 1
    failed = result.histories[1]
    assert failed.failed_time == Q(1, 16)
    assert failed.failed_tube == failed.failed_initial_box == calls[-1][0]
    assert failed.completed_loss_increment_bounds == failed.steps[0].increment[54]
    assert all(row.status == "not_attempted" for row in result.histories[2:])
    assert _rebuild(result).attempted_step_count == 4


def test_zero_horizon_applies_matched_events_without_flow(monkeypatch):
    monkeypatch.setattr(
        owner,
        "validated_box_taylor_step",
        lambda *a, **k: pytest.fail("zero horizon called kernel"),
    )
    result = owner.bound_sine_class_storage_readout(
        **_arguments(horizon=0, max_steps=1)
    )
    assert result.admitted and result.completed_history_count == 8
    assert (
        result.attempted_step_count
        == result.completed_step_count
        == result.planned_step_count
        == 0
    )
    assert result.integrated_excess_loss_bounds == result.excess_storage_bounds == I(0)
    assert all(
        row.initial_box == row.final_state_bounds and row.loss_integral_bounds == I(0)
        for row in result.histories
    )
    assert _rebuild(result, horizon=0, max_steps=1).complete


@pytest.mark.parametrize(
    "mutation",
    (
        lambda r: replace(r, mediator_class=True),
        lambda r: replace(r, donor_amplitude=True),
        lambda r: replace(r, capacity=(True,) + r.capacity[1:]),
        lambda r: replace(r, clock="t"),
        lambda r: replace(r, source_coordinates="absolute phases"),
        lambda r: replace(r, parameters=replace(r.parameters, classes=(True, 1, 1))),
        lambda r: replace(r, parameters=replace(r.parameters, gamma=I(0))),
        lambda r: replace(r, target_phase_bounds=(I(0),) + r.target_phase_bounds[1:]),
        lambda r: replace(r, full_source_box=r.tangent_source_box),
        lambda r: replace(r, planned_step_count=15),
        lambda r: replace(r, completed_history_count=7),
        lambda r: replace(r, integrated_excess_loss_bounds=I(0)),
        lambda r: replace(r, excess_storage_bounds=-r.excess_storage_bounds),
        lambda r: replace(
            r, raw_endpoint_loss_bounds=(I(0),) + r.raw_endpoint_loss_bounds[1:]
        ),
        lambda r: replace(r, failed_history_index=True),
    ),
)
def test_reader_rejects_model_source_counts_and_cached_verdict_tampering(
    polynomial_report, mutation
):
    with pytest.raises((TypeError, ValueError)):
        _rebuild(mutation(polynomial_report))


@pytest.mark.parametrize(
    "mutation",
    (
        lambda h: replace(h, history_index=True),
        lambda h: replace(h, event_amplitudes=(True, 0)),
        lambda h: replace(h, model_name="full"),
        lambda h: replace(h, pre_event_box=h.initial_box[:54] + (I(1),)),
        lambda h: replace(
            h, initial_box=h.initial_box[:27] + (I(0),) + h.initial_box[28:]
        ),
        lambda h: replace(h, completed_loss_increment_bounds=I(0)),
        lambda h: replace(h, reason="invented_failure"),
        lambda h: replace(h, attempted_step_count=3),
        lambda h: replace(
            h, steps=(replace(h.steps[0], picard_interior_margin=True),) + h.steps[1:]
        ),
        lambda h: replace(
            h, steps=(replace(h.steps[0], domain_lower_bounds=(True,)),) + h.steps[1:]
        ),
        lambda h: replace(
            h,
            steps=(replace(h.steps[0], increment=h.steps[0].increment[:54] + (I(0),)),)
            + h.steps[1:],
        ),
        lambda h: replace(
            h, steps=(h.steps[0], replace(h.steps[1], initial_box=h.initial_box))
        ),
        lambda h: replace(
            h,
            steps=(
                replace(
                    h.steps[0],
                    local_remainder_bounds=h.steps[0].local_remainder_bounds[:54],
                ),
            )
            + h.steps[1:],
        ),
    ),
)
def test_reader_rejects_event_carry_integral_and_step_tampering(
    polynomial_report, mutation
):
    histories = polynomial_report.histories
    forged = replace(
        polynomial_report,
        histories=(histories[0], mutation(histories[1])) + histories[2:],
    )
    with pytest.raises((TypeError, ValueError)):
        _rebuild(forged)


def test_report_projection_preserves_exact_bounds_and_null_availability(
    polynomial_report,
):
    import json

    projected = polynomial_report.to_dict()
    decoded = json_loads(json.dumps(projected))
    assert decoded["schema"] == "tnfr.sine-class-storage-readout.v1"
    assert decoded["report"]["horizon"]["numerator"] == 1
    assert decoded["report"]["horizon"]["denominator"] == 8
    assert decoded["report"]["failed_history_index"] is None
    assert len(decoded["report"]["histories"][0]["steps"][0]["series"]) == 55


@pytest.mark.parametrize("mediator_class", (1, 2))
def test_full_and_tangent_augmented_fields_match_independent_original_rows(
    mediator_class,
):
    mp = mpmath.mp.clone()
    mp.dps = 80
    parameters = owner._cubic_parameters(mediator_class)
    _, edges, degrees, _, _ = _geometry()
    geometry = owner._derive(
        tuple(range(27)), tuple(sorted(tuple(sorted(e)) for e in edges))
    )
    model = owner.RelationalExchangeModel(
        1, epi_weight=Q(1023, 1024), phase_weight=Q(1, 1024), phase_domain="regular"
    )
    fields, domain = owner._storage_loss_fields(model, geometry, parameters)
    x = tuple(Q((i * 5) % 17 - 8, 97) for i in range(27))
    y = tuple(Q((i * 7) % 19 - 9, 101) for i in range(27))
    target = owner._storage_target(parameters)

    def number(value):
        return mp.mpf(value.numerator) / value.denominator

    theta = tuple(
        2 * mp.pi * parameters.classes[i // 9] * (i % 9 - 4) / 9 for i in range(27)
    )
    gamma = 1 / (1023 * mp.pi)
    for model_index, field in enumerate(fields):
        phase_box = (
            tuple(t + value for t, value in zip(target, y))
            if model_index == 0
            else tuple(map(I, y))
        )
        state = tuple(map(I, x)) + phase_box + (I(Q(5, 7)),)
        rates = field(state)
        gradient, currents = [mp.mpf(0)] * 27, [mp.mpf(0)] * 27
        for i, j in edges:
            difference = number(x[j] - x[i])
            gradient[i] -= difference
            gradient[j] += difference
            gap = theta[j] - theta[i]
            current = (
                mp.sin(gap + number(y[j] - y[i]))
                if model_index == 0
                else mp.cos(gap) * number(y[j] - y[i])
            )
            currents[i] += current
            currents[j] -= current
        expected = (
            tuple(
                (-q + gamma * current) / d
                for q, current, d in zip(gradient, currents, degrees)
            )
            + tuple(gamma * q / d for q, d in zip(gradient, degrees))
            + (sum(q**2 / d for q, d in zip(gradient, degrees)),)
        )
        for interval, value in zip(rates, expected):
            assert number(interval.lo) <= value <= number(interval.hi)
        assert domain(state) == (Q(1),)
        # Passive integral magnitude does not feed either nodal row or itself.
        assert field(state[:-1] + (I(-100, 100),)) == rates
        directions = tuple(Q((3 * i) % 13 - 6, 37) for i in range(55))
        jets = field(
            tuple(Jet((value, I(slope))) for value, slope in zip(state, directions))
        )
        derivative = 2 * sum(
            gradient[i]
            * sum(
                number(directions[i] - directions[j])
                for edge in edges
                if i in edge
                for j in edge
                if j != i
            )
            / degrees[i]
            for i in range(27)
        )
        assert (
            number(jets[54].coeffs[1].lo) <= derivative <= number(jets[54].coeffs[1].hi)
        )


def test_unrelated_tiny_complete_histories_cover_independent_full_and_tangent_flows():
    radius = Q(1, 2**36)
    forms = tuple(Q((i * 5) % 11 - 5, 2000) for i in range(27))
    phases = tuple(Q((i * 7) % 13 - 6, 3000) for i in range(27))
    arguments = _arguments(
        mediator_class=1,
        initial_form_bounds=tuple((v - radius, v + radius) for v in forms),
        initial_phase_deviation_bounds=tuple((v - radius, v + radius) for v in phases),
        donor_amplitude=Q(1, 2000),
        receiver_amplitude=Q(-1, 3000),
        horizon=Q(1, 4096),
        time_step=Q(1, 8192),
        order=4,
        max_steps=16,
    )
    result = owner.bound_sine_class_storage_readout(**arguments)
    assert result.admitted
    _, edges, degrees, _, _ = _geometry()
    theta = np.array([2 * np.pi * (i % 9 - 4) / 9 for i in range(27)])
    gamma = 1 / (1023 * np.pi)
    losses = []
    for index, history in enumerate(result.histories):
        full = index % 2 == 0

        def field(_, state):
            x, y = state[:27], state[27:54]
            gradient, currents = np.zeros(27), np.zeros(27)
            for i, j in edges:
                gradient[i] += x[i] - x[j]
                gradient[j] += x[j] - x[i]
                current = (
                    np.sin(y[j] - y[i])
                    if full
                    else np.cos(theta[j] - theta[i]) * (y[j] - y[i])
                )
                currents[i] += current
                currents[j] -= current
            return np.r_[
                (-gradient + gamma * currents) / degrees,
                gamma * gradient / degrees,
                np.sum(gradient**2 / degrees),
            ]

        initial = np.r_[
            list(map(float, forms)),
            np.array(list(map(float, phases))) + (theta if full else 0),
            0.0,
        ]
        initial[4] += float(history.event_amplitudes[0])
        initial[22] += float(history.event_amplitudes[1])
        solved = solve_ivp(
            field,
            (0, float(arguments["horizon"])),
            initial,
            method="DOP853",
            rtol=1e-12,
            atol=1e-15,
        )
        assert solved.success
        for interval, value in zip(history.final_state_bounds, solved.y[:, -1]):
            assert float(interval.lo) <= value <= float(interval.hi)
        losses.append(Q.from_float(float(solved.y[54, -1])))
    mixed = (
        losses[6]
        - losses[2]
        - losses[4]
        + losses[0]
        - losses[7]
        + losses[3]
        + losses[5]
        - losses[1]
    )
    assert result.integrated_excess_loss_bounds.contains(mixed)
    assert evidence._reconstruct_storage_readout(
        result, owner._admit_storage_readout_inputs(**arguments)
    ).complete
