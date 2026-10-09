"""Independent exact reading-error controls for shared class decisions."""

from fractions import Fraction as Q

import pytest

from tnfr.physics._sine_class_contrast import _contrast_decision


@pytest.mark.parametrize("orientation", (-1, 1))
def test_noise_boundaries_keep_sign_and_separate_null_distinct(orientation):
    def at(value):
        return _contrast_decision((orientation * value,) * 2, 0, 1)

    assert at(8).true_sign and not at(8).recorded_sign
    assert at(8).scalar_cancellation and at(8).cancellation_margin == 0
    assert at(9).recorded_sign and not at(9).null_excluded
    assert not at(9).scalar_cancellation
    assert at(16).recorded_sign and not at(16).null_excluded
    assert at(17).null_excluded


def test_error_expansion_and_unavailable_orientation_do_not_invent_sign():
    result = _contrast_decision((Q(-2), Q(3)), Q(5), Q(1))
    assert result.true_bounds == (-7, 8)
    assert result.recorded_bounds == (-15, 16)
    assert result.orientation == 0 and result.oriented_lower is None
    assert result.noise_ceiling is None and result.status == "bounds_only"
    assert result.scalar_cancellation
    assert not result.true_sign


@pytest.mark.parametrize("contrast", (Q(-8), Q(-3, 7), Q(0), Q(8)))
def test_allowed_errors_construct_zero_mixed_contrast_without_equal_raw_records(
    contrast,
):
    signs = (1, -1, -1, 1, -1, 1, 1, -1)
    error = tuple(-contrast * sign / 8 for sign in signs)
    assert max(map(abs, error)) <= 1
    assert contrast + sum(c * e for c, e in zip(signs, error)) == 0
    # Equal mixed statistics do not require equal four-reading vectors.
    class_one, class_two = (Q(100), Q(100), Q(0), Q(0)), (Q(0),) * 4
    assert sum(c * (a - b) for c, a, b in zip(signs, class_one, class_two)) == 0
    assert abs(class_one[0] - class_two[0]) > 2


def test_exact_pairing_identity_overrides_reference_error_but_retains_readout():
    result = _contrast_decision((Q(-1), Q(1)), 10, Q(1, 7), exact_zero=True)
    assert result.true_bounds == (0, 0)
    assert result.recorded_bounds == (Q(-8, 7), Q(8, 7))
    assert result.status == "exact_contrast_zero" and result.scalar_cancellation
    assert not result.true_sign


@pytest.mark.parametrize(
    "bounds,error,delta,zero",
    (
        ((2, 1), 0, 0, False),
        ((0, 1), -1, 0, False),
        ((0, 1), 0, -1, False),
        ((False, 1), 0, 0, False),
        ((0, 1), True, 0, False),
        ((0, 1), 0, 0.0, False),
        ((0, 1), 0, 0, 1),
    ),
)
def test_invalid_arithmetic_cannot_certify_an_observation(bounds, error, delta, zero):
    with pytest.raises((TypeError, ValueError)):
        _contrast_decision(bounds, error, delta, exact_zero=zero)


@pytest.mark.parametrize("reading_count", (1, 8, 16))
@pytest.mark.parametrize("orientation", (-1, 1))
def test_declared_reading_count_controls_strict_sign_and_separate_null(
    reading_count, orientation
):
    def at(value):
        return _contrast_decision(
            (orientation * value,) * 2, 0, 1, reading_count=reading_count
        )

    equal = at(reading_count)
    assert equal.true_sign and not equal.recorded_sign
    assert equal.scalar_cancellation and equal.noise_ceiling == 1
    assert at(reading_count + Q(1, 2)).recorded_sign
    assert not at(2 * reading_count).null_excluded
    assert at(2 * reading_count + Q(1, 2)).null_excluded


def test_default_eight_readings_preserve_the_original_decision_record():
    for reference, error, noise in (((-3, 2), 1, 7), ((20, 21), 2, 1)):
        assert _contrast_decision(reference, error, noise) == _contrast_decision(
            reference, error, noise, reading_count=8
        )


@pytest.mark.parametrize("count", (False, True, 0, -1, Q(16), 16.0, "16"))
def test_reading_count_requires_a_positive_ordinary_integer(count):
    with pytest.raises((TypeError, ValueError)):
        _contrast_decision((1, 1), 0, 0, reading_count=count)


@pytest.mark.parametrize("contrast", (Q(-16), Q(-1, 7), Q(0), Q(16)))
def test_sixteen_nodal_errors_construct_scalar_cancellation(contrast):
    # Class, four-history and two-node signs each contribute to the actual
    # observation map. The support radius is 16, not an eight-reading sensor.
    signs = tuple(c * h * n for c in (1, -1) for h in (1, -1, -1, 1) for n in (1, -1))
    errors = tuple(-contrast * sign / 16 for sign in signs)
    assert len(errors) == 16 and max(map(abs, errors)) <= 1
    assert contrast + sum(c * e for c, e in zip(signs, errors)) == 0
    assert _contrast_decision(
        (contrast, contrast), 0, 1, reading_count=16
    ).scalar_cancellation
