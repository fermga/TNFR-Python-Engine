"""Shared finite-sample enclosures and their observation-scope boundaries.

Polynomial extremizers and static nodal symmetries provide independent
controls. No producer, trajectory, physical sensor or reserved response runs.
"""

import json
from dataclasses import FrozenInstanceError
from fractions import Fraction as Q
from itertools import product

import mpmath as mp
import networkx as nx
import pytest

from tnfr.dynamics.relational import RelationalExchangeModel
from tnfr.mathematics._rational_interval import I
from tnfr.mathematics._validated_taylor import flow_jets
from tnfr.physics.relational_observations import (
    bound_relational_coefficient_from_samples,
    bound_relational_jet_from_samples,
    bound_relational_rate_from_samples,
    bound_relational_sample_jet_budget,
)
from tnfr.physics.relational_sine_forecast import _sine_flow, admit_sine_prior
from tnfr.physics.relational_sine_observation import infer_relational_sine_hidden_state
from tnfr.sdk import export_to_json, relational_report_to_dict


def _arguments(**changes):
    arguments = dict(
        sample_step=Q(1, 8),
        sample_error_bounds=(Q(1, 1024),) * 3,
        third_derivative_bound=6,
    )
    arguments.update(changes)
    return arguments


def _inside(bounds, value):
    assert bounds[0] <= value <= bounds[1]


def test_cubic_and_unequal_noise_attain_both_separate_error_bounds():
    h, errors = Q(1, 8), (Q(0), Q(1, 512), Q(1, 1024))
    # The same positive cubic attains the rate and acceleration remainders;
    # opposite stencil signs make its rate error negative and acceleration
    # error positive. Noise has the corresponding extremizing signs.
    samples = tuple(
        2 - 3 * t + Q(7, 2) * t**2 + t**3 + sign * error
        for t, sign, error in zip((0, h, 2 * h), (1, -1, 1), errors)
    )
    report = bound_relational_jet_from_samples(
        iter(samples), **_arguments(sample_error_bounds=errors)
    )
    assert report.value_bounds == (2, 2)
    assert report.rate_bounds[1] == -3
    assert report.acceleration_bounds[0] == 7
    assert report.budget.rate_sample_error_bound == Q(9, 256)
    assert report.budget.rate_error_bound == Q(17, 256)
    assert report.budget.acceleration_sample_error_bound == Q(5, 16)
    assert report.budget.acceleration_error_bound == Q(17, 16)
    assert report.samples == samples


def test_quadratic_is_exact_without_noise_and_anchor_time_is_first_sample():
    h, at = Q(1, 10), Q(7, 3)
    samples = tuple(5 + 2 * t - 3 * t**2 for t in (0, h, 2 * h))
    report = bound_relational_jet_from_samples(
        samples,
        **_arguments(
            sample_step=h,
            sample_error_bounds=(0, 0, 0),
            third_derivative_bound=0,
            observation_time=at,
        ),
    )
    assert report.value_bounds == (5, 5)
    assert report.rate_bounds == (2, 2)
    assert report.acceleration_bounds == (-6, -6)
    assert report.budget.sample_times == (at, at + h, at + 2 * h)
    assert report.budget.evidence_window == (at, at + 2 * h)


def test_timestamp_jitter_uses_independent_speed_and_enlarges_evidence_window():
    h, at = Q(1, 8), Q(1)
    jitter = (Q(1, 64), Q(1, 128), Q(1, 32))
    signs = (-1, 1, -1)
    # A linear signal reaches the Lipschitz timing-error allowance exactly.
    samples = tuple(
        4 + 3 * (j * h + sign * error)
        for j, sign, error in zip(range(3), signs, jitter)
    )
    report = bound_relational_jet_from_samples(
        samples,
        **_arguments(
            sample_error_bounds=(0, 0, 0),
            third_derivative_bound=0,
            timestamp_error_bounds=jitter,
            first_derivative_bound=3,
            observation_time=at,
        ),
    )
    budget = report.budget
    assert budget.timing_value_error_bounds == tuple(3 * value for value in jitter)
    assert budget.evidence_window == (at - jitter[0], at + 2 * h + jitter[2])
    assert report.value_bounds[1] == 4
    assert report.rate_bounds[0] == 3
    _inside(report.acceleration_bounds, 0)
    assert budget.rate_sample_error_bound == 0
    assert budget.acceleration_sample_error_bound == 0
    assert budget.rate_truncation_error_bound == 0
    assert budget.rate_error_bound == budget.rate_timing_error_bound


def test_clock_and_gain_covariance_retains_separate_uncertainty_terms():
    raw = (Q(2), Q(19, 10), Q(7, 4))
    original = bound_relational_jet_from_samples(raw, **_arguments())
    gain, dilation, offset = Q(-3), Q(5), Q(7)
    transformed = bound_relational_jet_from_samples(
        tuple(gain * value + offset for value in raw),
        **_arguments(
            sample_step=original.budget.sample_step * dilation,
            sample_error_bounds=tuple(
                abs(gain) * value for value in original.budget.sample_error_bounds
            ),
            third_derivative_bound=abs(gain) * 6 / dilation**3,
        ),
    )
    assert transformed.value_bounds == tuple(
        reversed(tuple(gain * value + offset for value in original.value_bounds))
    )
    assert transformed.rate_bounds == tuple(
        reversed(tuple(gain * value / dilation for value in original.rate_bounds))
    )
    assert transformed.acceleration_bounds == tuple(
        reversed(
            tuple(gain * value / dilation**2 for value in original.acceleration_bounds)
        )
    )


def test_shared_samples_do_not_make_cartesian_derivative_bounds_independent():
    report = bound_relational_jet_from_samples(
        (0, 0, 0),
        **_arguments(
            sample_step=1, sample_error_bounds=(1, 1, 1), third_derivative_bound=0
        ),
    )
    assert report.rate_bounds == (-4, 4)
    assert report.acceleration_bounds == (-4, 4)
    # These intervals safely enclose both outputs, but their upper corner
    # cannot be reached by one shared sample-error vector. The extrema of
    # linear stencils occur at vertices of the sample-error cube.
    vertices = {
        ((-3 * a + 4 * b - c) / 2, a - 2 * b + c)
        for a, b, c in product((-1, 1), repeat=3)
    }
    assert (4, 4) not in vertices
    assert max(rate for rate, _ in vertices) == 4
    assert max(acceleration for _, acceleration in vertices) == 4
    assert report.budget.sample_error_bounds == (1, 1, 1)


def test_legacy_rate_and_coefficient_adapters_retain_values_and_schema():
    raw, h, error, third = (Q(1), Q(9, 10), Q(4, 5)), Q(1, 16), Q(1, 1024), Q(3)
    shared = bound_relational_jet_from_samples(
        raw,
        sample_step=h,
        sample_error_bounds=(error,) * 3,
        third_derivative_bound=third,
    )
    arguments = dict(
        sample_step=h, sample_error_bound=error, third_derivative_bound=third
    )
    rate = bound_relational_rate_from_samples(raw, **arguments)
    coefficient = bound_relational_coefficient_from_samples(raw, **arguments)
    assert rate.rate_bounds == shared.rate_bounds
    # The coefficient observer separately projects onto its established
    # outward dyadic arithmetic; preserve that adapter's rounding contract.
    for exact, outward in (
        (shared.value_bounds, coefficient.jet.form_bounds),
        (shared.rate_bounds, coefficient.jet.rate_bounds),
        (shared.acceleration_bounds, coefficient.jet.acceleration_bounds),
    ):
        assert outward[0] <= exact[0] <= exact[1] <= outward[1]
        assert exact[0] - outward[0] < Q(1, 2**120)
        assert outward[1] - exact[1] < Q(1, 2**120)
    assert rate.rate_estimate == coefficient.rate_estimate == shared.rate_estimate
    assert (
        coefficient.acceleration_error_bound == shared.budget.acceleration_error_bound
    )
    assert (
        relational_report_to_dict(rate)["report_type"] == "RelationalRateSampleBounds"
    )
    assert (
        relational_report_to_dict(coefficient)["report_type"]
        == "RelationalCoefficientSampleBounds"
    )


def test_exact_export_and_immutable_retained_evidence(tmp_path):
    samples = [Q(1, 3), Q(1, 4), Q(1, 5)]
    report = bound_relational_jet_from_samples(samples, **_arguments())
    samples[0] = 999
    assert report.samples[0] == Q(1, 3)
    with pytest.raises(FrozenInstanceError):
        report.rate_bounds = (0, 0)
    for value in (report, report.budget):
        payload = relational_report_to_dict(value)
        assert payload["report_type"] == type(value).__name__
        path = tmp_path / (type(value).__name__ + ".json")
        export_to_json(payload, path)
        assert json.loads(path.read_text(encoding="utf-8")) == payload
    body = relational_report_to_dict(report)["report"]
    assert (
        tuple(Q(row["numerator"], row["denominator"]) for row in body["rate_bounds"])
        == report.rate_bounds
    )
    body["samples"][0]["numerator"] = 999
    assert report.samples[0] == Q(1, 3)


@pytest.mark.parametrize(
    "keyword,value",
    (
        ("sample_step", 0),
        ("sample_step", True),
        ("sample_error_bounds", (0, -1, 0)),
        ("sample_error_bounds", (0, 0)),
        ("sample_error_bounds", {0, 1, 2}),
        ("sample_error_bounds", (0, False, 0)),
        ("third_derivative_bound", -1),
        ("third_derivative_bound", float("inf")),
        ("timestamp_error_bounds", (0, -1, 0)),
        ("timestamp_error_bounds", (0, 0, 0, 0)),
        ("timestamp_error_bounds", (0, Q(1, 128), 0)),
        ("first_derivative_bound", -1),
        ("first_derivative_bound", float("nan")),
        ("observation_time", -1),
        ("observation_time", True),
    ),
)
def test_budget_rejects_invalid_inputs_or_unsupported_jitter(keyword, value):
    with pytest.raises((TypeError, ValueError)):
        bound_relational_sample_jet_budget(**_arguments(**{keyword: value}))


def test_jitter_window_must_not_cross_negative_time():
    with pytest.raises(ValueError):
        bound_relational_sample_jet_budget(
            **_arguments(
                timestamp_error_bounds=(Q(1, 64), 0, 0), first_derivative_bound=1
            )
        )


@pytest.mark.parametrize(
    "samples",
    ((0, 1), (0, 1, 2, 3), {0, 1, 2}, (0, True, 2), (0, float("nan"), 2), (0, 1j, 2)),
)
def test_sample_admission_rejects_invalid_representations(samples):
    with pytest.raises((TypeError, ValueError)):
        bound_relational_jet_from_samples(samples, **_arguments())


def test_exact_rational_information_is_not_lost_to_float_underflow():
    tiny = Q(1, 10**400)
    report = bound_relational_jet_from_samples(
        (0, tiny, 2 * tiny),
        **_arguments(
            sample_step=1, sample_error_bounds=(0, 0, 0), third_derivative_bound=0
        ),
    )
    assert report.rate_bounds == (tiny, tiny)
    assert report.acceleration_bounds == (0, 0)


def test_three_samples_alone_cannot_bound_initial_derivatives():
    h, amplitude = Q(1, 8), Q(100)
    # This cubic vanishes at every sample while its derivatives at zero do
    # not. Its independently supplied C3 bound is therefore indispensable.
    samples = tuple(amplitude * t * (t - h) * (t - 2 * h) for t in (0, h, 2 * h))
    report = bound_relational_jet_from_samples(
        samples,
        **_arguments(
            sample_error_bounds=(0, 0, 0), third_derivative_bound=6 * amplitude
        ),
    )
    assert samples == (0, 0, 0)
    _inside(report.rate_bounds, 2 * amplitude * h**2)
    _inside(report.acceleration_bounds, -6 * amplitude * h)
    assert report.rate_estimate == report.acceleration_estimate == 0


def test_circular_samples_alias_even_when_third_derivative_is_zero():
    # Exact angles in turns avoid replacing mathematical 2*pi by a float.
    # Lift N*t/h has C3=0; all its circular samples coincide modulo one turn.
    h = Q(1, 8)
    for winding in (0, 1, -3, 20):
        lift = tuple(winding * t / h for t in (0, h, 2 * h))
        assert tuple(value % 1 for value in lift) == (0, 0, 0)
        report = bound_relational_jet_from_samples(
            lift, **_arguments(sample_error_bounds=(0, 0, 0), third_derivative_bound=0)
        )
        assert report.rate_bounds == (winding / h, winding / h)
        assert report.acceleration_bounds == (0, 0)


def _mp(value):
    value = Q(value)
    return mp.mpf(value.numerator) / value.denominator


def _fine_oracle(form, phase, capacities, neighbors, model):
    form, phase, capacities = [
        tuple(map(_mp, row)) for row in (form, phase, capacities)
    ]
    e, w = map(_mp, model.effective_weights)
    beta = _mp(model.storage_scale)
    gradients = [sum(form[i] - form[j] for j in row) for i, row in enumerate(neighbors)]
    currents = [
        sum(mp.sin(phase[j] - phase[i]) for j in row) for i, row in enumerate(neighbors)
    ]
    rates = [
        capacities[i] * (-e * gradients[i] + w * currents[i] / mp.pi) / len(row)
        for i, row in enumerate(neighbors)
    ]
    phases = [
        capacities[i] * w * gradients[i] / (beta * mp.pi * len(row))
        for i, row in enumerate(neighbors)
    ]
    gradient_rates = [
        sum(rates[i] - rates[j] for j in row) for i, row in enumerate(neighbors)
    ]
    current_rates = [
        sum(mp.cos(phase[j] - phase[i]) * (phases[j] - phases[i]) for j in row)
        for i, row in enumerate(neighbors)
    ]
    accelerations = [
        capacities[i]
        * (-e * gradient_rates[i] + w * current_rates[i] / mp.pi)
        / len(row)
        for i, row in enumerate(neighbors)
    ]
    phase_accelerations = [
        capacities[i] * w * gradient_rates[i] / (beta * mp.pi * len(row))
        for i, row in enumerate(neighbors)
    ]
    return rates + phases + [mp.mpf(0)], accelerations + phase_accelerations + [
        mp.mpf(0)
    ]


def test_common_sensor_bias_evades_derivative_witness_but_not_initial_box():
    model = RelationalExchangeModel(1, phase_domain="regular")
    neighbors, form, phase, capacities = (
        ((2,), (2,), (0, 1)),
        (Q(0), Q(1, 2), Q(1)),
        (Q(1, 4), Q(-1, 8), Q(1, 2)),
        (Q(1), Q(2), Q(3)),
    )
    original = form + phase + (capacities[-1],)
    translated = (
        tuple(value + Q(1, 8) for value in form)
        + tuple(value + Q(1, 16) for value in phase)
        + (capacities[-1],)
    )

    def flow(values):
        return _sine_flow(
            values, neighbors=neighbors, visible_capacity=capacities[:-1], model=model
        )

    first = flow(tuple(map(I, original)))
    assert first == flow(tuple(map(I, translated)))
    series = flow_jets(tuple(map(I, original)), 2, flow)
    shifted_series = flow_jets(tuple(map(I, translated)), 2, flow)
    assert tuple(row[1:] for row in series) == tuple(row[1:] for row in shifted_series)
    with mp.workdps(90):
        oracle = _fine_oracle(form, phase, capacities, neighbors, model)
        for i, row in enumerate(series):
            for degree, derivative in ((1, oracle[0][i]), (2, oracle[1][i])):
                bound = degree * row[degree]
                assert _mp(bound.lo) <= derivative <= _mp(bound.hi)

        def evidence(value):
            center, radius = Q(mp.nstr(value, 80)), Q(1, 10**9)
            return center - radius, center + radius

        visible = nx.Graph()
        for i in (0, 1):
            visible.add_node(
                i, EPI=float(form[i]), theta=float(phase[i]), nu_f=float(capacities[i])
            )
        state = infer_relational_sine_hidden_state(
            visible,
            ports=(0, 1),
            reference_model=model,
            form_rate_bounds={i: evidence(oracle[0][i]) for i in (0, 1)},
            phase_rate_bounds={i: evidence(oracle[0][3 + i]) for i in (0, 1)},
            source_id="static-prior",
            clock_id="fixed-clock",
            observation_time=0,
            evidence_window=(0, 0),
            forecast_start=1,
        )
        capacity = state.infer_capacity(
            form_acceleration_bounds={i: evidence(oracle[1][i]) for i in (0, 1)},
            phase_acceleration_bounds={i: evidence(oracle[1][3 + i]) for i in (0, 1)},
            source_id="static-prior-second",
            clock_id="fixed-clock",
            observation_time=0,
            evidence_window=(0, 0),
        )
    prior = admit_sine_prior(capacity)
    assert prior.admitted
    assert all(box.contains(value) for box, value in zip(prior.initial_box, original))
    assert not prior.initial_box[0].contains(translated[0])
    assert not prior.initial_box[3].contains(translated[3])
    # Constant sensor bias cancels the common translation at all times. A
    # valid derivative witness therefore does not authenticate its anchor.
    assert translated[0] - Q(1, 8) == original[0]
    assert translated[3] - Q(1, 16) == original[3]
