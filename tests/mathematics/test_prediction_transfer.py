"""Independent affine histories and counterexamples for error transfer."""

from fractions import Fraction as Q

import pytest

from tnfr.mathematics._prediction_transfer import candidate_forecast_error
from tnfr.mathematics._rational_interval import I


def test_affine_future_saturates_reading_residual_and_endpoint_allowances():
    # For every affine history q(t)=a+b*t, q(2)=2*q(1)-q(0), so G=3.
    # Candidate is identically zero; truth has q(0)=2 eta, q(1)=-2 eta.
    eta, numerical_radius = Q(1, 16), Q(1, 256)
    truth = lambda time: 2 * eta - 4 * eta * time  # noqa: E731
    record = (eta, -eta)
    assert all(abs(truth(t) - record[t]) == eta for t in (0, 1))
    report = candidate_forecast_error(
        record,
        (I(0), I(0)),
        (I(0, 2 * numerical_radius),),
        gain=3,
        source_error=0,
        reading_error=eta,
    )
    assert report.residual_upper == eta
    assert report.forecast_center == (numerical_radius,)
    assert abs(truth(2) - report.forecast_center[0]) == report.true_output_radii[0]


def test_source_error_is_needed_at_both_history_and_future():
    # Reference 2-4t; fine source adds 1-2(t-1)^2, bounded by one on [0,2].
    # Fine samples are +1,-1 and read as zero with error one; fine future -7.
    reference = lambda t: 2 - 4 * t  # noqa: E731
    fine = lambda t: reference(t) + 1 - 2 * (t - 1) ** 2  # noqa: E731
    assert (fine(0), fine(1), fine(2)) == (1, -1, -7)
    report = candidate_forecast_error(
        (0, 0),
        (I(0), I(0)),
        (I(0),),
        gain=3,
        source_error=1,
        reading_error=1,
    )
    assert report.residual_upper == 0
    assert abs(fine(2)) == report.maximum_true_output_radius


def test_overlapping_reading_intervals_do_not_make_candidate_residual_zero():
    # One constant-zero candidate is enclosed by both [-1,1] boxes.
    # Noiseless true affine history goes through +1,-1 and reaches -3.
    report = candidate_forecast_error(
        (1, -1),
        (I(-1, 1), I(-1, 1)),
        (I(0),),
        gain=3,
        source_error=0,
        reading_error=0,
    )
    assert all(
        box.contains(y) for box, y in zip(report.candidate_samples, report.observations)
    )
    assert report.residual_upper == 2
    assert report.maximum_true_output_radius >= 3


def test_unknown_common_form_is_retained_and_numerical_error_counted_once():
    offset, r = Q(17, 8), Q(1, 128)
    report = candidate_forecast_error(
        (offset, offset),
        (I(offset - r, offset + r),) * 2,
        (I(offset - r, offset + r),),
        gain=1,
        source_error=0,
        reading_error=0,
    )
    assert report.forecast_center == (offset,)
    assert report.residual_upper == r
    assert report.maximum_true_output_radius == 2 * r
    assert not hasattr(report, "admitted")


@pytest.mark.parametrize("field", ("gain", "source_error", "reading_error"))
@pytest.mark.parametrize("invalid", (True, float("nan"), float("inf"), -1))
def test_invalid_bound_primitives_reject(field, invalid):
    bounds = dict(gain=1, source_error=0, reading_error=0)
    bounds[field] = invalid
    with pytest.raises((ValueError, TypeError)):
        candidate_forecast_error((0,), (I(0),), (I(0),), **bounds)


@pytest.mark.parametrize(
    "observations", ((True,), (float("nan"),), (float("inf"),), (), (0, 1), {0})
)
def test_invalid_or_unmatched_records_reject(observations):
    with pytest.raises((ValueError, TypeError)):
        candidate_forecast_error(
            observations, (I(0),), (I(0),), gain=1, source_error=0, reading_error=0
        )


@pytest.mark.parametrize(
    "samples,future", (((), (I(0),)), ((I(0),), ()), ((0,), (I(0),)), ((I(0),), (0,)))
)
def test_explicit_nonempty_interval_evidence_required(samples, future):
    with pytest.raises((ValueError, TypeError)):
        candidate_forecast_error(
            (0,), samples, future, gain=1, source_error=0, reading_error=0
        )


def test_exact_tiny_residual_is_not_lost_in_float_conversion():
    tiny = Q(1, 10**1000)
    report = candidate_forecast_error(
        (tiny,), (I(0),), (I(0),), gain=1, source_error=0, reading_error=0
    )
    assert report.residual_upper == report.maximum_true_output_radius == tiny
