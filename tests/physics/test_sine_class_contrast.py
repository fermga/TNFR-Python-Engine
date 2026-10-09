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
