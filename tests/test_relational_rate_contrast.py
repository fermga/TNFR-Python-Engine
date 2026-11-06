"""Declared rate contrasts: outward bounds, normalization limits and export."""

import json
from dataclasses import FrozenInstanceError
from fractions import Fraction as Q

import pytest

from tnfr.physics.relational_observations import (
    bound_relational_rate_contrast,
    bound_relational_rate_from_samples,
)
from tnfr.sdk import export_to_json, relational_report_to_dict


def _observe(before=(2, 2), after=(1, 1)):
    return bound_relational_rate_contrast(before_bounds=before, after_bounds=after)


@pytest.mark.parametrize("sign", (-1, 1))
def test_exact_point_change_allows_either_baseline_sign(sign):
    report = _observe((2 * sign,) * 2, (sign,) * 2)
    assert report.change_bounds == (-sign,) * 2
    assert report.normalized_change_bounds == (-Q(1, 2),) * 2
    assert report.unavailable_reasons == ()
    same = _observe((2 * sign,) * 2, (2 * sign,) * 2)
    assert same.normalized_change_bounds == (0, 0)
    with pytest.raises(FrozenInstanceError):
        report.normalized_change_bounds = (0, 0)


@pytest.mark.parametrize(
    "before,after",
    (
        ((Q(7, 3), Q(9, 2)), (Q(5, 7), Q(4, 3))),
        ((Q(-9, 2), Q(-7, 3)), (Q(-4, 3), Q(-5, 7))),
        ((-4, -2), (-1, 3)),
        ((2, 4), (-3, 1)),
    ),
)
def test_outward_intervals_enclose_independent_exact_corner_ratios(before, after):
    report = _observe(before, after)
    lower, upper = report.normalized_change_bounds
    for baseline in before:
        for intervention in after:
            reference = Q(intervention) / Q(baseline) - 1
            assert lower <= reference <= upper
            assert report.change_bounds[0] <= intervention - baseline
            assert intervention - baseline <= report.change_bounds[1]


def test_normalization_preserves_repeated_baseline_dependency():
    report = _observe((2, 4), (3, 5))
    assert report.change_bounds == (-1, 3)
    # Dividing the independently widened change by the baseline would admit
    # -1/2, although the actual minimum 3/4-1 is only -1/4.
    assert report.normalized_change_bounds == (-Q(1, 4), Q(3, 2))
    assert report.normalized_change_bounds[0] > -Q(1, 2)


def test_equal_uncertain_intervals_do_not_certify_equal_unknown_rates():
    report = _observe((2, 4), (2, 4))
    assert report.normalized_change_bounds == (-Q(1, 2), 1)
    point = _observe((Q(1, 3),) * 2, (Q(1, 3),) * 2)
    assert point.normalized_change_bounds[0] <= 0 <= point.normalized_change_bounds[1]


@pytest.mark.parametrize("factor", (Q(8), Q(-8), Q(1, 2), Q(-5, 7)))
def test_common_nonzero_gain_and_clock_rate_factor_cancel(factor):
    before = (Q(2), Q(3))
    after = (Q(1), Q(3, 2))
    transformed = _observe(
        tuple(sorted(factor * value for value in before)),
        tuple(sorted(factor * value for value in after)),
    )
    lower, upper = transformed.normalized_change_bounds
    for baseline in before:
        for intervention in after:
            assert lower <= intervention / baseline - 1 <= upper


def test_different_gains_or_additive_rate_offsets_do_not_cancel():
    baseline = _observe()
    changed_gain = _observe(after=(2, 2))
    changed_offset = _observe(before=(3, 3), after=(2, 2))
    assert baseline.normalized_change_bounds == (-Q(1, 2),) * 2
    assert changed_gain.normalized_change_bounds == (0, 0)
    assert changed_offset.normalized_change_bounds[0] > -Q(1, 2)


@pytest.mark.parametrize("before", ((0, 0), (-1, 1), (0, 1), (-1, 0)))
def test_baseline_zero_is_unavailable_but_absolute_change_is_retained(before):
    report = _observe(before, (2, 3))
    assert report.normalized_change_bounds is None
    assert report.unavailable_reasons == ("baseline_rate_not_separated_from_zero",)
    assert report.change_bounds == (2 - before[1], 3 - before[0])


@pytest.mark.parametrize("sign", (-1, 1))
def test_tiny_exact_baseline_can_lose_resolution_without_becoming_a_zero_claim(sign):
    tiny = Q(sign, 2**200)
    report = _observe((tiny, tiny), (2 * tiny, 2 * tiny))
    assert report.before_bounds[0] <= tiny <= report.before_bounds[1]
    assert report.normalized_change_bounds is None
    assert report.unavailable_reasons == ("baseline_rate_not_separated_from_zero",)


@pytest.mark.parametrize("argument", ("before_bounds", "after_bounds"))
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
        {"lower": 0, "upper": 1},
        "01",
        None,
    ),
)
def test_invalid_raw_intervals_reject_in_either_position(argument, invalid):
    arguments = {"before_bounds": (2, 2), "after_bounds": (1, 1)}
    arguments[argument] = invalid
    with pytest.raises((ValueError, TypeError)):
        bound_relational_rate_contrast(**arguments)


def test_lost_nonzero_real_input_rejects_before_interval_construction():
    class UnderflowingReal(float):
        def __float__(self):
            return 0.0

    with pytest.raises(ValueError, match="underflows"):
        _observe((UnderflowingReal(1),) * 2)


def test_one_shot_inputs_and_export_retain_exact_bounds_and_abstention(tmp_path):
    before, after = [2, 4], [3, 5]
    report = _observe(iter(before), iter(after))
    before[0] = 999
    after.reverse()
    assert report.before_bounds == (2, 4)
    assert report.after_bounds == (3, 5)
    for name, value in (("available", report), ("unavailable", _observe((-1, 1)))):
        payload = relational_report_to_dict(value)
        assert payload["report_type"] == "RelationalRateContrastBounds"
        path = tmp_path / f"{name}.json"
        export_to_json(payload, path)
        assert json.loads(path.read_text(encoding="utf-8")) == payload
        body = payload["report"]
        assert body["unavailable_reasons"] == list(value.unavailable_reasons)
        encoded = body["normalized_change_bounds"]
        if encoded is None:
            assert value.normalized_change_bounds is None
        else:
            assert tuple(Q(v["numerator"], v["denominator"]) for v in encoded) == (
                value.normalized_change_bounds
            )
        body["before_bounds"][0]["numerator"] = 999
        assert value.before_bounds[0] != 999


@pytest.mark.parametrize("cubic_sign", (-1, 1))
def test_cubic_response_and_aligned_noise_attain_the_rate_error_bound(cubic_sign):
    step, noise = Q(1, 8), Q(1, 1024)

    def response(time):
        # Its independently known derivative at zero is 3 and |y'''| is 6.
        return 7 + 3 * time - 2 * time**2 + cubic_sign * time**3

    samples = tuple(
        response(index * step) + cubic_sign * noise * noise_sign
        for index, noise_sign in enumerate((1, -1, 1))
    )
    report = bound_relational_rate_from_samples(
        samples,
        sample_step=step,
        sample_error_bound=noise,
        third_derivative_bound=6,
    )
    assert abs(report.rate_estimate - 3) == report.rate_error_bound
    assert cubic_sign * (report.rate_estimate - 3) < 0
    assert 3 in report.rate_bounds
    assert report.rate_bounds[0] <= 3 <= report.rate_bounds[1]


@pytest.mark.parametrize("gain", (Q(3, 2), Q(-3, 2)))
def test_sample_rate_is_covariant_under_gain_clock_and_constant_offset(gain):
    step, noise, third = Q(1, 7), Q(1, 128), Q(6)
    samples = tuple(2 - time + time**3 for time in (0, step, 2 * step))
    original = bound_relational_rate_from_samples(
        samples,
        sample_step=step,
        sample_error_bound=noise,
        third_derivative_bound=third,
    )
    clock_scale, offset = Q(5, 3), Q(17, 13)
    transformed = bound_relational_rate_from_samples(
        tuple(gain * sample + offset for sample in samples),
        sample_step=clock_scale * step,
        sample_error_bound=abs(gain) * noise,
        third_derivative_bound=abs(gain) * third / clock_scale**3,
    )
    factor = gain / clock_scale
    assert transformed.rate_estimate == factor * original.rate_estimate
    assert transformed.rate_error_bound == abs(factor) * original.rate_error_bound
    assert transformed.rate_bounds == tuple(
        sorted(factor * endpoint for endpoint in original.rate_bounds)
    )
    assert transformed.rate_bounds[0] <= -factor <= transformed.rate_bounds[1]


@pytest.mark.parametrize(
    "argument,invalid",
    (
        ("samples", (0, 1)),
        ("samples", (0, 1, 2, 3)),
        ("samples", {0, 1, 2}),
        ("samples", {"initial": 0, "middle": 1, "final": 2}),
        ("samples", "012"),
        ("samples", None),
        ("samples", (0, True, 2)),
        ("samples", (0, float("nan"), 2)),
        ("sample_step", 0),
        ("sample_step", -1),
        ("sample_step", True),
        ("sample_step", float("inf")),
        ("sample_error_bound", -1),
        ("sample_error_bound", True),
        ("sample_error_bound", float("nan")),
        ("third_derivative_bound", -1),
        ("third_derivative_bound", True),
        ("third_derivative_bound", float("inf")),
    ),
)
def test_sample_rate_rejects_invalid_preparation_or_error_arguments(argument, invalid):
    arguments = dict(
        samples=(0, 1, 2),
        sample_step=1,
        sample_error_bound=0,
        third_derivative_bound=0,
    )
    arguments[argument] = invalid
    with pytest.raises((TypeError, ValueError)):
        bound_relational_rate_from_samples(**arguments)


def test_independent_sample_rates_compose_with_contrast_and_exact_export(tmp_path):
    step, noise = Q(1, 8), Q(1, 1024)
    reports = []
    for rate, quadratic in ((2, 1), (1, -3)):
        samples = [
            rate * time + quadratic * time**2 + error
            for time, error in zip(
                (0, step, 2 * step), (noise, -noise, noise), strict=True
            )
        ]
        report = bound_relational_rate_from_samples(
            iter(samples),
            sample_step=step,
            sample_error_bound=noise,
            third_derivative_bound=0,
        )
        assert report.rate_bounds[0] <= rate <= report.rate_bounds[1]
        samples[0] = 999
        assert report.samples[0] == noise
        reports.append(report)

    contrast = bound_relational_rate_contrast(
        before_bounds=reports[0].rate_bounds, after_bounds=reports[1].rate_bounds
    )
    assert contrast.unavailable_reasons == ()
    assert (
        contrast.normalized_change_bounds[0]
        <= -Q(1, 2)
        <= contrast.normalized_change_bounds[1]
    )
    with pytest.raises(FrozenInstanceError):
        reports[0].rate_bounds = (0, 0)

    payload = relational_report_to_dict(reports[0])
    assert payload["report_type"] == "RelationalRateSampleBounds"
    output = tmp_path / "sample-rate.json"
    export_to_json(payload, output)
    assert json.loads(output.read_text(encoding="utf-8")) == payload
    body = payload["report"]
    assert (
        tuple(
            Q(value["numerator"], value["denominator"]) for value in body["rate_bounds"]
        )
        == reports[0].rate_bounds
    )
    body["samples"][0]["numerator"] = 999
    assert reports[0].samples[0] == noise
