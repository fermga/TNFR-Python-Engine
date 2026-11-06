"""Synthetic shared-phase certificates; no record, design or trajectory is read."""

from fractions import Fraction as Q

import pytest

from tnfr.mathematics._rational_interval import I, cos, pi_interval, sqrt
from tnfr.physics._organized_computable_prediction import _forecast_stages
from tnfr.physics._organized_law_observation import _certify_organized_phase_circle
from tnfr.research._organized_forecast_inputs import _sample_times


def _certificate(**changes):
    arguments = dict(
        odd_coefficients=(1, 8), even_coefficients=(1, 16), odd_error=0, even_error=0
    )
    arguments.update(changes)
    return _certify_organized_phase_circle(**arguments)


def test_long_ramp_contrast_uses_existing_endpoints_and_excludes_the_last_return():
    stages, readings = _forecast_stages(), set(_sample_times())
    for child in range(5):
        ramps = tuple(
            stage for stage in stages if stage.name == "ramp" and stage.child == child
        )
        first, _, third = ramps
        assert first.base == 0 and first.increment == Q(1, 10)
        assert third.base == -Q(1, 10) and third.increment == Q(1, 10)
        assert first.duration == third.duration == 1
        assert first.start in readings and first.start + 1 in readings
        assert third.start in readings
        assert (third.start + 1 in readings) == (child < 4)
    # The declared edge 1->2 is probed by children 1 and 2, both covered above.


@pytest.mark.parametrize("mode", (1, 2))
def test_spectral_edge_extractor_norm_follows_from_projected_incidence_columns(mode):
    angle = 2 * mode * pi_interval() / 5
    projector = tuple(
        tuple(Q(2, 5) * cos((i - j) * angle) for j in range(5)) for i in range(5)
    )
    columns = tuple(
        tuple(projector[i][edge + 1] - projector[i][edge] for i in range(5))
        for edge in (0, 1)
    )
    gram = tuple(
        tuple(sum((a * b for a, b in zip(left, right)), I(0)) for right in columns)
        for left in columns
    )
    determinant = gram[0][0] * gram[1][1] - gram[0][1] * gram[1][0]
    assert determinant.lo > 0
    for diagonal in (gram[1][1] / determinant, gram[0][0] / determinant):
        # diag((Q'Q)^-1) gives the squared norm of each pseudoinverse row.
        assert (diagonal - 2).contains(0)
        assert diagonal.width < Q(1, 10**24)
    mu = (5 + (1 if mode == 2 else -1) * sqrt(5)) / 8
    assert 0 < mu.lo <= mu.hi < 1
    # cosh(mu/2)<sum_k(1/8)^k=8/7 and sqrt(10)<13/4 suffice.
    assert Q(13, 4) ** 2 > 10 and Q(13, 4) * Q(8, 7) < 4


@pytest.mark.parametrize(
    "cosine,sine",
    ((1, 0), (-1, 0), (0, 1), (Q(3, 5), Q(4, 5)), (Q(-3, 5), Q(4, 5))),
)
def test_actual_double_angle_constraint_retains_the_synthetic_uniform_margin(
    cosine, sine
):
    report = _certificate()
    assert cosine**2 + sine**2 == 1
    # Independent phasor algebra, without any constitutive or trajectory call.
    doubled = (cosine**2 - sine**2, 2 * cosine * sine)
    odd = doubled[0] - cosine / Q(4)
    even = doubled[1] - sine / Q(8)
    assert odd**2 + even**2 >= report.margin_lower**2
    assert report.ratio_upper == Q(1, 4)
    assert report.margin_lower == Q(3, 4)
    assert report.available and report.exclusion_certified
    assert (
        report.scope
        == "conditional_shared_phase_contrast_exclusion_not_a_record_verdict"
    )
    expected_radius = Q(1, 1023 * 4 * 2187) / pi_interval()
    difference = report.ninth_radius - expected_radius
    assert difference.contains(0) and difference.width < Q(1, 10**30)


def test_error_that_cancels_a_real_synthetic_contrast_cannot_certify_exclusion():
    report = _certificate()
    # At theta=0 the two raw contrasts are exactly 3*r9 and zero.
    cancelled = _certificate(odd_error=3 * report.ninth_radius.hi)
    assert cancelled.available and not cancelled.exclusion_certified
    assert cancelled.reason == "error_budget_not_strict"


def test_strict_boundary_is_not_rounded_into_a_certificate():
    report = _certificate()
    boundary_error = report.ninth_radius.lo * 8 * report.margin_lower / 2
    equality = _certificate(odd_error=boundary_error)
    assert equality.error_square_upper == equality.margin_lower**2
    assert not equality.exclusion_certified
    assert _certificate(odd_error=boundary_error / 2).exclusion_certified


@pytest.mark.parametrize(
    "odd,even",
    (
        ((-1, -8), (1, -16)),
        ((1.0, 8.0), (1, 16)),
        ((I(1, Q(11, 10)), I(8, 9)), (I(-1), I(-16, -15))),
    ),
)
def test_signed_interval_and_represented_coefficients_use_outward_ratio_bounds(
    odd, even
):
    report = _certificate(odd_coefficients=odd, even_coefficients=even)
    assert report.available and report.exclusion_certified
    assert report.ratio_upper == max(
        ratio.abs_max for ratio in report.normalized_ratios
    )
    assert report.margin_lower == 1 - report.ratio_upper
    assert report.normalized_error_bounds == (0, 0)


@pytest.mark.parametrize("field", ("odd_coefficients", "even_coefficients"))
@pytest.mark.parametrize("denominator", (0, I(-1, 1), I(0, 2)))
def test_uncertain_or_zero_high_harmonic_scale_is_unavailable(field, denominator):
    report = _certificate(**{field: (1, denominator)})
    assert not report.available and not report.exclusion_certified
    assert report.reason == "high_harmonic_denominator_contains_zero"
    assert report.normalized_ratios is report.margin_lower is None


@pytest.mark.parametrize("coefficient", (1, 2, I(Q(1, 2), 2)))
def test_no_positive_uniform_margin_is_unavailable_even_with_zero_error(coefficient):
    report = _certificate(odd_coefficients=(coefficient, 2))
    assert not report.available and not report.exclusion_certified
    assert report.ratio_upper >= 1
    assert report.reason == "no_positive_uniform_circle_margin"
    # For coefficient=1, theta=0 actually cancels both contrasts.
    if coefficient == 1:
        cancellation = report.ninth_radius * (
            -report.odd_coefficients[0] + report.odd_coefficients[1] / 2
        )
        assert cancellation.contains(0)


@pytest.mark.parametrize("field", ("odd_coefficients", "even_coefficients"))
@pytest.mark.parametrize("bad", ((), (1,), (1, 2, 3), "12", {1: 2}, {1, 2}, None))
def test_coefficients_require_an_ordered_pair(field, bad):
    with pytest.raises((TypeError, ValueError)):
        _certificate(**{field: bad})


@pytest.mark.parametrize("bad", (True, False, float("inf"), float("nan"), "1", None))
@pytest.mark.parametrize(
    "field", ("odd_coefficients", "even_coefficients", "odd_error", "even_error")
)
def test_invalid_primitive_is_rejected_before_arithmetic(field, bad):
    value = (bad, 8) if field.endswith("coefficients") else bad
    with pytest.raises((TypeError, ValueError)):
        _certificate(**{field: value})


@pytest.mark.parametrize("field", ("odd_error", "even_error"))
def test_negative_error_is_rejected_even_if_denominator_is_unavailable(field):
    with pytest.raises(ValueError, match="nonnegative"):
        _certificate(odd_coefficients=(1, 0), **{field: -Q(1, 100)})
