"""Independent exact-arithmetic checks of relative-resultant admission bounds.

The reference below evaluates the original rational angle directly. It uses
a high-order Taylor polynomial with a symmetric Lagrange remainder, without
the production kernel's pi reduction, alternating-tail rule or dyadic grid.
No engine state advances and no retained research campaign is repeated.
"""

from fractions import Fraction as Q
from math import factorial, pi

import pytest

from tnfr.mathematics._phase_resultant_chamber import (
    certified_cosine_bounds,
    certified_cosine_lower_bound,
    certified_sine_bounds,
    regular_resultant_margin,
    relative_resultant_bounds,
    relative_resultant_lower_bounds,
    relative_resultant_regular_margins,
)


def _direct_cosine_reference(angle):
    """Enclose cosine directly by Taylor degree121 and its Lagrange bound."""
    # Odd coefficients vanish. |cos^(122)|<=1 supplies a symmetric remainder
    # independent of the alternating-tail and true-pi reduction under test.
    partial = sum(
        (Q((-1) ** k) * angle ** (2 * k) / factorial(2 * k) for k in range(61)),
        Q(0),
    )
    remainder = abs(angle) ** 122 / factorial(122)
    return partial - remainder, partial + remainder


def _direct_sine_reference(angle):
    """Direct degree121 Taylor/Lagrange enclosure without phase reduction."""
    partial = sum(
        (Q((-1) ** k) * angle ** (2 * k + 1) / factorial(2 * k + 1) for k in range(61)),
        Q(0),
    )
    remainder = abs(angle) ** 122 / factorial(122)
    return partial - remainder, partial + remainder


@pytest.mark.parametrize(
    "angle", (Q(0), Q(1, 64), Q(1, 2), Q(1), Q(2), Q(3), Q(7), Q(100, 7))
)
def test_sine_contains_independent_direct_angle_reference_and_preserves_oddness(angle):
    lower, upper = certified_sine_bounds(angle)
    expected_lower, expected_upper = _direct_sine_reference(angle)
    assert -1 <= lower <= expected_lower <= expected_upper <= upper <= 1
    assert certified_sine_bounds(-angle) == (-upper, -lower)


def test_sine_preserves_exact_zero_but_does_not_invent_tiny_sign_evidence():
    assert certified_sine_bounds(Q(0)) == (Q(0), Q(0))
    tiny = Q(1, 1 << 1074)
    lower, upper = certified_sine_bounds(tiny)
    assert lower <= 0 < tiny <= upper
    assert lower < upper


def test_sine_uses_true_pi_when_a_represented_pi_value_is_not_on_the_cut():
    # A binary64 value of pi is not mathematical pi. Its nonzero sine must
    # not be forced to zero by a floating principal-angle branch convention.
    angle = Q(pi)
    lower, upper = certified_sine_bounds(angle)
    expected_lower, expected_upper = _direct_sine_reference(angle)
    assert 0 < lower <= expected_lower <= expected_upper <= upper


@pytest.mark.parametrize("angle", (Q(10) ** 100, Q(1 << 5000)))
def test_sine_work_limits_retain_the_full_range(angle):
    assert certified_sine_bounds(angle) == (Q(-1), Q(1))
    assert certified_sine_bounds(-angle) == (Q(-1), Q(1))


def test_regular_work_uncertainty_cannot_create_false_admission():
    phases = (Q(0), Q(1 << 5000))
    assert relative_resultant_regular_margins(phases, ((1,), (0,))) == (Q(0), Q(0))


def test_custom_sine_precision_keeps_independent_enclosure():
    angle = Q(2)
    lower, upper = certified_sine_bounds(angle, terms=32, bits=128)
    reference_lower, reference_upper = _direct_sine_reference(angle)
    assert lower <= reference_lower <= reference_upper <= upper
    assert upper - lower < Q(1, 1 << 125)


@pytest.mark.parametrize("value", (True, 0, 0.0, "0"))
def test_sine_rejects_nonrational_input(value):
    with pytest.raises(TypeError, match="Fraction"):
        certified_sine_bounds(value)


@pytest.mark.parametrize(
    "angle",
    (Q(0), Q(1, 64), Q(1, 16), Q(65, 1024), Q(1, 2), Q(1), Q(3), Q(4), Q(7), Q(100, 7)),
)
def test_enclosure_contains_independent_direct_angle_reference(angle):
    lower, upper = certified_cosine_bounds(angle)
    reference_lower, reference_upper = _direct_cosine_reference(angle)
    assert Q(-1) <= lower <= reference_lower <= reference_upper <= upper <= Q(1)
    assert certified_cosine_bounds(-angle) == (lower, upper)
    assert certified_cosine_lower_bound(angle) == lower


def test_zero_is_exact_and_subnormal_angle_is_not_certified_as_exact_zero():
    assert certified_cosine_bounds(Q(0)) == (Q(1), Q(1))
    tiny = Q(1, 1 << 1074)
    lower, upper = certified_cosine_bounds(tiny)
    # The elementary inequality 1-x²/2 <= cos(x) <= 1 needs no libm oracle.
    assert lower <= 1 - tiny * tiny / 2 <= upper
    assert lower < 1 and upper == 1


@pytest.mark.parametrize("angle", (Q(10) ** 100, Q(1 << 5000)))
def test_precision_or_work_limit_returns_the_conservative_cosine_range(angle):
    assert certified_cosine_bounds(angle) == (Q(-1), Q(1))
    assert certified_cosine_bounds(-angle) == (Q(-1), Q(1))


@pytest.mark.parametrize("value", (True, 0, 0.0, "0"))
def test_cosine_kernel_requires_an_exact_rational_input(value):
    with pytest.raises(TypeError, match="Fraction"):
        certified_cosine_bounds(value)


@pytest.mark.parametrize("angle", (Q(0), Q(1, 100), Q(1, 16), Q(1, 2), Q(3), Q(7)))
def test_custom_precision_retains_rigorous_bounds_without_small_angle_floor(angle):
    lower, upper = certified_cosine_bounds(angle, terms=32, bits=128)
    reference_lower, reference_upper = _direct_cosine_reference(angle)
    assert lower <= reference_lower <= reference_upper <= upper
    assert upper - lower <= Q(1, 1 << 127)
    default_lower, default_upper = certified_cosine_bounds(angle)
    assert default_lower <= lower <= upper <= default_upper


@pytest.mark.parametrize("terms,bits", ((2, 64), (3, 64), (31, 128), (32, 128)))
def test_custom_even_or_odd_series_lengths_enclose_the_alternating_tail(terms, bits):
    lower, upper = certified_cosine_bounds(Q(3), terms=terms, bits=bits)
    reference_lower, reference_upper = _direct_cosine_reference(Q(3))
    assert lower <= reference_lower <= reference_upper <= upper


@pytest.mark.parametrize(
    "options,error",
    (
        ({"terms": True}, TypeError),
        ({"bits": 128.0}, TypeError),
        ({"terms": 1}, ValueError),
        ({"terms": 65}, ValueError),
        ({"bits": 0}, ValueError),
        ({"bits": 257}, ValueError),
    ),
)
@pytest.mark.parametrize("owner", (certified_cosine_bounds, certified_sine_bounds))
def test_trigonometric_custom_work_policy_is_explicit_and_bounded(
    options, error, owner
):
    with pytest.raises(error):
        owner(Q(1), **options)


def test_raw_difference_is_not_rounded_before_cosine_admission():
    # Both node phases are exactly representable binary64 values; subtracting
    # them in binary64 would erase the small second node's contribution.
    phases = (Q(1), Q(-1, 1 << 54))
    gap = phases[0] - phases[1]
    assert Q(float(gap)) == Q(1) and gap > 1
    lower, upper = certified_cosine_bounds(gap)
    reference_lower, reference_upper = _direct_cosine_reference(gap)
    rounded_lower, _ = _direct_cosine_reference(Q(1))
    assert lower <= reference_lower <= reference_upper <= upper < rounded_lower
    assert relative_resultant_lower_bounds(phases, ((1,), (0,))) == (lower, lower)


def test_common_rotation_preserves_exact_relative_margins():
    phases = (Q(0), Q(1, 2), Q(1))
    neighbors = ((1, 2), (0, 2), (0, 1))
    original = relative_resultant_lower_bounds(phases, neighbors)
    for offset in (Q(-7), Q(1 << 50)):
        shifted = tuple(value + offset for value in phases)
        assert relative_resultant_lower_bounds(shifted, neighbors) == original
    assert min(original) > Q(11, 8)


def test_positive_resultants_can_include_nearly_antipodal_ordinary_edges():
    represented_pi = Q(pi)
    ring = tuple(-represented_pi * k / 4 for k in (0, 4, 3, 2, 1))
    phases = ring + ring
    neighbors = (
        (1, 4, 5),
        (0, 2, 6),
        (1, 3),
        (2, 4),
        (3, 0),
        (6, 9, 0),
        (5, 7, 1),
        (6, 8),
        (7, 9),
        (8, 5),
    )
    margins = relative_resultant_lower_bounds(phases, neighbors)
    assert abs(phases[1] - phases[0]) == represented_pi
    assert min(margins) > Q(7, 10)
    for i, row in enumerate(neighbors):
        true_lower = sum(
            (_direct_cosine_reference(phases[j] - phases[i])[0] for j in row),
            Q(0),
        )
        assert 0 < margins[i] <= true_lower


def test_unresolved_lower_margin_does_not_claim_a_true_zero_resultant():
    represented_pi = Q(pi)
    phases = (Q(0), Q(0), represented_pi)
    margins = relative_resultant_lower_bounds(phases, ((1, 2), (0, 2), (0, 1)))
    actual_lower, _ = _direct_cosine_reference(represented_pi)
    assert 1 + actual_lower > 0
    assert margins[0] == 0


@pytest.mark.parametrize(
    ("phases", "neighbors", "error"),
    (
        ((), (), TypeError),
        ((Q(0), 0.0), ((1,), (0,)), TypeError),
        ({Q(0), Q(1)}, ((1,), (0,)), TypeError),
        ((Q(0), Q(1)), ((1,),), ValueError),
        ((Q(0), Q(1)), ((1, 1), (0,)), ValueError),
        ((Q(0), Q(1)), ((0,), (0,)), ValueError),
        ((Q(0), Q(1)), ((2,), (0,)), ValueError),
        ((Q(0), Q(1)), ((True,), (0,)), ValueError),
        ((Q(0), Q(1)), ({1}, (0,)), TypeError),
        ((Q(0), Q(1)), {0: (1,), 1: (0,)}, TypeError),
    ),
)
@pytest.mark.parametrize(
    "owner",
    (
        relative_resultant_lower_bounds,
        relative_resultant_bounds,
        relative_resultant_regular_margins,
    ),
)
def test_malformed_phase_or_support_input_cannot_supply_an_admission_margin(
    phases, neighbors, error, owner
):
    with pytest.raises(error):
        owner(phases, neighbors)


def test_full_regular_bounds_admit_negative_real_resultants_with_separated_imaginary_part():
    phases, neighbors = (Q(0), Q(2)), ((1,), (0,))
    bounds = relative_resultant_bounds(phases, neighbors)
    margins = relative_resultant_regular_margins(phases, neighbors)
    expected_real = _direct_cosine_reference(Q(2))
    expected_imag = _direct_sine_reference(Q(2))
    for (real, imag), sign, margin in zip(bounds, (1, -1), margins):
        assert real[0] <= expected_real[0] <= expected_real[1] <= real[1] < 0
        reference = (
            expected_imag if sign > 0 else (-expected_imag[1], -expected_imag[0])
        )
        assert imag[0] <= reference[0] <= reference[1] <= imag[1]
        assert Q(9, 10) < margin <= min(abs(value) for value in reference)
        assert margin == regular_resultant_margin(real, imag)
    assert bounds[0][0] == bounds[1][0]
    assert bounds[0][1] == (-bounds[1][1][1], -bounds[1][1][0])
    assert max(relative_resultant_lower_bounds(phases, neighbors)) < 0


def test_positive_axis_is_regular_while_zero_and_negative_axis_are_not_admitted():
    assert relative_resultant_regular_margins((Q(0), Q(0)), ((1,), (0,))) == (
        Q(1),
        Q(1),
    )
    assert relative_resultant_regular_margins((Q(0),), ((),)) == (Q(0),)
    # At node zero, conjugate phasors cancel imaginary parts exactly and
    # have negative cosine. Rounding the rectangle cannot make a false sign.
    phases, neighbors = (Q(0), Q(2), Q(-2)), ((1, 2), (0,), (0,))
    real, imag = relative_resultant_bounds(phases, neighbors)[0]
    assert real[1] < 0 and imag[0] <= 0 <= imag[1]
    assert relative_resultant_regular_margins(phases, neighbors)[0] == 0
    assert regular_resultant_margin((Q(-2), Q(-1)), (Q(0), Q(0))) == 0


def test_regular_full_chord_margin_rejects_a_cut_crossing_despite_regular_endpoints():
    neighbors = ((1,), (0,))
    start = (Q(0), Q(2))
    end = (Q(0), Q(4))
    assert min(relative_resultant_regular_margins(start, neighbors)) > 0
    assert min(relative_resultant_regular_margins(end, neighbors)) > 0
    increments = tuple(right - left for left, right in zip(start, end))
    for i, row in enumerate(neighbors):
        drift = sum((abs(increments[j] - increments[i]) for j in row), Q(0))
        margin = relative_resultant_regular_margins(start, neighbors)[i]
        assert margin - drift < 0
    # The continuous relative gap traverses mathematical pi between 2 and
    # 4, so endpoint-only admission would jump across the negative-real cut.
    start_imag = relative_resultant_bounds(start, neighbors)[0][1]
    end_imag = relative_resultant_bounds(end, neighbors)[0][1]
    assert start_imag[0] > 0 > end_imag[1]


def test_regular_full_chord_bound_admits_a_small_move_and_is_rotation_invariant():
    phases, neighbors = (Q(0), Q(2)), ((1,), (0,))
    increments = (Q(1000), Q(10001, 10))
    initial = relative_resultant_regular_margins(phases, neighbors)
    for i, row in enumerate(neighbors):
        drift = sum((abs(increments[j] - increments[i]) for j in row), Q(0))
        assert initial[i] - drift > Q(4, 5)
    rotated = tuple(value + Q(1 << 50) for value in phases)
    assert relative_resultant_bounds(rotated, neighbors) == relative_resultant_bounds(
        phases, neighbors
    )
    assert relative_resultant_regular_margins(rotated, neighbors) == initial


@pytest.mark.parametrize(
    "real,imag,error",
    (
        ((Q(0), Q(1)), (0.0, Q(1)), TypeError),
        ((Q(0), True), (Q(0), Q(1)), TypeError),
        ((Q(1), Q(0)), (Q(0), Q(1)), ValueError),
        ((Q(0), Q(1)), (Q(1), Q(0)), ValueError),
        ((Q(0), Q(1), Q(2)), (Q(0), Q(1)), TypeError),
        ({Q(0), Q(1)}, (Q(0), Q(1)), TypeError),
    ),
)
def test_regular_margin_requires_ordered_exact_rectangle_bounds(real, imag, error):
    with pytest.raises(error):
        regular_resultant_margin(real, imag)
