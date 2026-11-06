"""Conditional interval admission and exact export of declared response jets."""

import json
from dataclasses import FrozenInstanceError
from fractions import Fraction as Q

import mpmath as mp
import pytest

from tnfr.physics.relational_observations import (
    bound_relational_coefficient_from_jet,
    bound_relational_coefficient_from_samples,
)
from tnfr.sdk import export_to_json, relational_report_to_dict


def _bounds(form=(1, 1), rate=(-3, -3), acceleration=(7, 7)):
    return bound_relational_coefficient_from_jet(
        form_bounds=form, rate_bounds=rate, acceleration_bounds=acceleration
    )


def _mp(value):
    return mp.mpf(value.numerator) / value.denominator


def test_point_jets_enclose_the_ratio_with_mathematical_pi():
    report = _bounds()
    assert report.unavailable_reasons == ()
    assert report.restoring_gap_bounds == (2, 2)
    assert report.squared_rate_bounds == (9, 9)
    lower, upper = report.coefficient_bounds
    assert lower < upper
    with mp.workdps(80):
        assert _mp(lower) <= 9 / (2 * mp.pi**2) <= _mp(upper)
    assert (
        "no_graph_read_preparation_authentication_derivative_estimation_or_evolution"
        in report.scope
    )
    with pytest.raises(FrozenInstanceError):
        report.coefficient_bounds = (0, 0)


@pytest.mark.parametrize(
    "arguments,reason",
    (
        ({"form": (-1, 1)}, "initial_form_not_separated_from_zero"),
        ({"rate": (1, 1)}, "incompatible_initial_decay"),
        ({"rate": (-1, -1), "acceleration": (1, 1)}, "nonpositive_restoring_gap"),
        ({"rate": (-1, -1), "acceleration": (0, 2)}, "unresolved_restoring_gap"),
    ),
)
def test_unresolved_or_incompatible_jets_never_supply_a_coefficient(arguments, reason):
    report = _bounds(**arguments)
    assert report.coefficient_bounds is None
    assert reason in report.unavailable_reasons


def test_zero_damping_and_critical_damping_are_not_singular_identifications():
    undamped = _bounds(rate=(0, 0), acceleration=(-1, -1))
    assert undamped.coefficient_bounds == (0, 0)
    assert undamped.unavailable_reasons == ()
    critical = _bounds(rate=(-2, -2), acceleration=(3, 3))
    lower, upper = critical.coefficient_bounds
    with mp.workdps(80):
        assert _mp(lower) <= 4 / mp.pi**2 <= _mp(upper)


def test_interval_uncertainty_contains_independent_corner_ratios():
    form = (Q(99, 100), Q(101, 100))
    rate = (Q(-301, 100), Q(-299, 100))
    acceleration = (Q(699, 100), Q(701, 100))
    report = _bounds(form, rate, acceleration)
    assert report.restoring_gap_bounds[0] > 0
    lower, upper = report.coefficient_bounds
    with mp.workdps(80):
        for m0 in form:
            for m1 in rate:
                for m2 in acceleration:
                    gap = m1**2 - m0 * m2
                    reference = _mp(m1**2 / gap) / mp.pi**2
                    assert _mp(lower) <= reference <= _mp(upper)


def test_unknown_nonzero_gain_and_constant_clock_scale_cancel():
    baseline = _bounds()
    gain, clock = -2, 4
    transformed = _bounds(
        (gain, gain),
        (-3 * Q(gain, clock),) * 2,
        (7 * Q(gain, clock**2),) * 2,
    )
    with mp.workdps(80):
        reference = 9 / (2 * mp.pi**2)
        for report in (baseline, transformed):
            lower, upper = report.coefficient_bounds
            assert _mp(lower) <= reference <= _mp(upper)


@pytest.mark.parametrize(
    "invalid",
    (
        (True, 1),
        (0, "1"),
        (0, float("inf")),
        (0, float("nan")),
        (2, 1),
        (0,),
        (0, 1, 2),
        {0, 1},
    ),
)
def test_invalid_intervals_reject_before_estimation(invalid):
    with pytest.raises((ValueError, TypeError)):
        _bounds(form=invalid)


def test_tiny_nonzero_input_is_enclosed_not_certified_as_resolved():
    tiny = Q(1, 10**200)
    report = _bounds((tiny, tiny), (-tiny, -tiny), (0, 0))
    assert report.form_bounds[0] <= tiny <= report.form_bounds[1]
    assert report.coefficient_bounds is None
    assert "initial_form_not_separated_from_zero" in report.unavailable_reasons


@pytest.mark.parametrize("sign", (-1, 1))
def test_resolved_incompatible_sign_is_not_hidden_by_product_rounding(sign):
    tiny = Q(sign, 2**80)
    report = _bounds((tiny, tiny), (tiny, tiny), (-sign, -sign))
    assert report.form_bounds == (tiny, tiny)
    assert report.rate_bounds == (tiny, tiny)
    assert report.restoring_gap_bounds[0] > 0
    assert report.coefficient_bounds is None
    assert report.unavailable_reasons == ("incompatible_initial_decay",)


def test_one_shot_input_and_json_export_preserve_bounds_and_abstention(tmp_path):
    report = _bounds(iter((1, 1)), iter((-3, -3)), iter((7, 7)))
    for value in (report, _bounds(rate=(-1, -1), acceleration=(0, 2))):
        payload = relational_report_to_dict(value)
        path = tmp_path / (
            "bounded.json" if value.coefficient_bounds else "unavailable.json"
        )
        export_to_json(payload, path)
        assert json.loads(path.read_text(encoding="utf-8")) == payload
        assert payload["report_type"] == "RelationalCoefficientJetBounds"
        body = payload["report"]
        if value.coefficient_bounds is None:
            assert body["coefficient_bounds"] is None
            assert body["unavailable_reasons"] == ["unresolved_restoring_gap"]
        else:
            assert (
                tuple(
                    Q(item["numerator"], item["denominator"])
                    for item in body["coefficient_bounds"]
                )
                == value.coefficient_bounds
            )
            body["coefficient_bounds"][0]["numerator"] = -1
            assert value.coefficient_bounds[0] > 0


def test_sample_wrapper_retains_stencil_and_separate_error_terms(tmp_path):
    step, error, third = Q(1, 128), Q(1, 2**30), Q(6)
    samples = tuple(
        1 - 3 * t + Q(7, 2) * t**2 + t**3 + sign * error
        for t, sign in ((0, 1), (step, -1), (2 * step, 1))
    )
    report = bound_relational_coefficient_from_samples(
        iter(samples),
        sample_step=step,
        sample_error_bound=error,
        third_derivative_bound=third,
    )
    assert report.samples == samples
    assert abs(report.rate_estimate + 3) == report.rate_error_bound
    assert abs(report.acceleration_estimate - 7) == report.acceleration_error_bound
    assert report.rate_error_bound == 4 * error / step + third * step**2 / 3
    assert report.acceleration_error_bound == 4 * error / step**2 + third * step
    assert report.jet.coefficient_bounds is not None
    payload = relational_report_to_dict(report)
    path = tmp_path / "sample-bounds.json"
    export_to_json(payload, path)
    assert json.loads(path.read_text(encoding="utf-8")) == payload
    assert payload["report"]["jet"] == relational_report_to_dict(report.jet)["report"]


@pytest.mark.parametrize(
    "keyword,value",
    (
        ("sample_step", 0),
        ("sample_step", True),
        ("sample_error_bound", -1),
        ("third_derivative_bound", -1),
        ("third_derivative_bound", float("inf")),
    ),
)
def test_sample_wrapper_rejects_invalid_time_or_error_budget(keyword, value):
    arguments = dict(
        sample_step=Q(1, 16), sample_error_bound=0, third_derivative_bound=1
    )
    arguments[keyword] = value
    with pytest.raises((ValueError, TypeError)):
        bound_relational_coefficient_from_samples((1, Q(3, 4), Q(1, 2)), **arguments)


@pytest.mark.parametrize("samples", ((1, 2), (1, 2, 3, 4), (True, 1, 2), {1, 2, 3}))
def test_sample_wrapper_requires_three_ordered_physical_scalars(samples):
    with pytest.raises((ValueError, TypeError)):
        bound_relational_coefficient_from_samples(
            samples,
            sample_step=1,
            sample_error_bound=0,
            third_derivative_bound=0,
        )
