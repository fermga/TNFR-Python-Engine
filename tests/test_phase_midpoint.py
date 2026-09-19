"""Independent exact branch, remainder and rounding tests for phase midpoints."""

import math
from dataclasses import FrozenInstanceError
from fractions import Fraction
from functools import lru_cache

import pytest

import tnfr.mathematics._phase_midpoint as owner
from tnfr.mathematics._phase_midpoint import certified_two_neighbor_phase


def _alternating_bounds(denominator, count):
    terms = tuple(
        Fraction(1, (2 * k + 1) * denominator ** (2 * k + 1)) for k in range(count + 1)
    )
    partial = sum(
        (term if k % 2 == 0 else -term for k, term in enumerate(terms[:-1])),
        Fraction(0),
    )
    next_partial = partial + (-1 if count % 2 else 1) * terms[-1]
    return min(partial, next_partial), max(partial, next_partial)


@lru_cache(maxsize=None)
def _independent_pi_bounds(count=128):
    low5, high5 = _alternating_bounds(5, count)
    low239, high239 = _alternating_bounds(239, count)
    return 16 * low5 - 4 * high239, 16 * high5 - 4 * low239


def _affine_round(rational, coefficient):
    if coefficient == 0:
        return float(rational)
    low, high = _independent_pi_bounds()
    endpoints = sorted((rational + coefficient * low, rational + coefficient * high))
    rounded = tuple(float(value) for value in endpoints)
    assert rounded[0].hex() == rounded[1].hex()
    return rounded[0]


@pytest.mark.parametrize("denominator", [5, 239])
@pytest.mark.parametrize("count", [1, 2, 3, 4, 64])
def test_reciprocal_arctangent_enclosure_uses_the_signed_first_omitted_term(
    denominator, count
):
    actual = owner._atan_reciprocal_bounds(denominator, count)
    expected = _alternating_bounds(denominator, count)
    refined = _alternating_bounds(denominator, count + 2)
    assert actual == expected
    assert actual[0] < refined[0] < refined[1] < actual[1]
    assert actual[1] - actual[0] == Fraction(
        1, (2 * count + 1) * denominator ** (2 * count + 1)
    )


def test_machin_identity_is_verified_by_rational_tangent_addition():
    tangent = Fraction(1, 5)
    twice = 2 * tangent / (1 - tangent * tangent)
    four_times = 2 * twice / (1 - twice * twice)
    difference = (four_times - Fraction(1, 239)) / (1 + four_times / 239)
    assert twice == Fraction(5, 12)
    assert four_times == Fraction(120, 119)
    assert difference == 1
    low, high = _independent_pi_bounds()
    assert Fraction(3, 4) < low / 4 < high / 4 < 1


def test_cached_pi_bounds_enclose_raw_machin_series_with_outward_rounding():
    raw_low, raw_high = _independent_pi_bounds(owner._PI_TERMS)
    refined_low, refined_high = _independent_pi_bounds(128)
    low, high = owner._pi_bounds()
    assert low <= raw_low < refined_low < refined_high < raw_high <= high
    assert 3 < low < high < Fraction(22, 7)
    assert high - low < Fraction(1, 2**250)
    assert Fraction(math.pi) < low
    assert float(low) == float(high) == math.pi


def test_retained_k3_midpoint_tie_rounds_to_even_and_delta_is_independent():
    center = float.fromhex("0x1.9219a3c32cd66p+1")
    first = float(Fraction(4716575516971799, 2**51))
    second = float(Fraction(2358183504581023, 2**49))
    exact_mean = Fraction(14149309535295891, 2**52)
    previous_production_mean = float(Fraction(7074654767647945, 2**51))
    expected = float.fromhex("0x1.9225c6c558ccap+1")
    result = certified_two_neighbor_phase(center, first, second)
    assert result is not None
    assert Fraction(center) == Fraction(3536910368204467, 2**50)
    assert exact_mean == (Fraction(first) + Fraction(second)) / 2
    assert result.mean_rational == exact_mean
    assert result.mean_pi_coefficient == result.delta_pi_coefficient == 0
    assert exact_mean == (Fraction(previous_production_mean) + Fraction(expected)) / 2
    assert Fraction(previous_production_mean) / Fraction(math.ulp(expected)) % 2 == 1
    assert Fraction(expected) / Fraction(math.ulp(expected)) % 2 == 0
    assert result.mean == expected == math.nextafter(previous_production_mean, math.inf)
    assert result.delta == float(exact_mean - Fraction(center))
    assert result.delta != result.mean - center


def test_exact_cancellation_does_not_round_a_large_common_phase_offset_first():
    result = certified_two_neighbor_phase(4.0, 4.0 - 2.0**-49, 4.0 + 2.0**-49)
    assert result is not None
    assert result.mean == 4.0
    assert result.delta == 0.0
    assert result.delta.hex() == (0.0).hex()
    assert result.delta_enclosure == (0, 0)
    assert result.mean_enclosure == (Fraction(4), Fraction(4))


@pytest.mark.parametrize(
    "first,second,expected_mean",
    [
        (1.0, 1.0 + 2.0**-52, 1.0),
        (1.0 + 2.0**-52, 1.0 + 2.0**-51, 1.0 + 2.0**-51),
    ],
)
def test_rational_midpoint_ties_preserve_nearest_even_and_unrounded_displacement(
    first,
    second,
    expected_mean,
):
    result = certified_two_neighbor_phase(1.0, first, second)
    assert result is not None
    midpoint = (Fraction(first) + Fraction(second)) / 2
    assert result.mean == expected_mean
    assert result.delta == float(midpoint - 1)
    assert result.delta != result.mean - 1.0
    assert result.mean_enclosure == (midpoint, midpoint)


@pytest.mark.parametrize(
    "center_n,first_n,second_n,mean_n,delta_n,negative_zero",
    [
        (0, 0, 1, 0, 0, False),
        (0, 1, 2, 2, 2, False),
        (2, 1, 2, 2, 0, True),
        (4, 0, 3, 2, -2, False),
    ],
)
def test_subnormal_midpoints_round_once_including_negative_underflow(
    center_n,
    first_n,
    second_n,
    mean_n,
    delta_n,
    negative_zero,
):
    tiny = math.ulp(0.0)
    result = certified_two_neighbor_phase(
        center_n * tiny, first_n * tiny, second_n * tiny
    )
    assert result is not None
    assert result.mean == mean_n * tiny
    assert result.delta == delta_n * tiny
    if result.delta == 0:
        assert math.copysign(1.0, result.delta) == (-1 if negative_zero else 1)


@pytest.mark.parametrize(
    "center,first,second,mean_coefficient,delta_coefficient",
    [
        (0.0, 0.25, math.tau - 0.25, 1, -1),
        (math.tau - 0.125, math.tau - 0.25, 0.125, 1, 1),
        (0.125, math.tau - 0.25, math.tau - 0.125, 0, -2),
        (math.tau - 0.125, 0.125, 0.25, 0, 2),
    ],
)
def test_symbolic_pi_handles_both_wrap_directions_and_two_crossing_neighbors(
    center,
    first,
    second,
    mean_coefficient,
    delta_coefficient,
):
    result = certified_two_neighbor_phase(center, first, second)
    assert result is not None
    midpoint = (Fraction(first) + Fraction(second)) / 2
    assert result.mean_pi_coefficient == mean_coefficient
    assert result.delta_pi_coefficient == delta_coefficient
    assert result.mean == _affine_round(midpoint, mean_coefficient)
    assert result.delta == _affine_round(midpoint - Fraction(center), delta_coefficient)
    assert result == certified_two_neighbor_phase(center, second, first)
    independent_low, independent_high = _independent_pi_bounds()
    for rational, coefficient, enclosure in (
        (midpoint, mean_coefficient, result.mean_enclosure),
        (midpoint - Fraction(center), delta_coefficient, result.delta_enclosure),
    ):
        true_bounds = sorted(
            (
                rational + coefficient * independent_low,
                rational + coefficient * independent_high,
            )
        )
        assert enclosure[0] <= true_bounds[0] <= true_bounds[1] <= enclosure[1]


def test_true_circle_mean_can_round_to_math_tau_without_erasing_small_displacement():
    result = certified_two_neighbor_phase(0.0, 0.25, math.tau - 0.25)
    assert result is not None
    assert result.mean == math.tau
    assert result.delta < 0
    assert result.delta == _affine_round(Fraction(math.pi), -1)
    assert result.delta != 0.0


@pytest.mark.parametrize(
    "center,first,second",
    [
        (1.0, 0.75, 1.25),
        (math.pi, math.pi - 0.25, math.pi + 0.25),
        (0.0, math.pi / 3, 5 * math.pi / 3),
        (math.tau - 0.125, 0.125, 0.25),
    ],
)
def test_neighbor_order_is_irrelevant(center, first, second):
    result = certified_two_neighbor_phase(center, first, second)
    assert result is not None
    assert result == certified_two_neighbor_phase(center, second, first)


def test_strict_eligibility_uses_true_half_pi_not_the_represented_threshold():
    low, high = _independent_pi_bounds()
    inside = math.pi / 2
    outside = math.nextafter(inside, math.inf)
    assert 2 * Fraction(inside) < low < high < 2 * Fraction(outside)
    assert certified_two_neighbor_phase(0.0, inside, 0.0) is not None
    assert certified_two_neighbor_phase(0.0, outside, 0.0) is None


def test_wrapped_negative_half_pi_boundary_uses_true_pi_too():
    low, high = _independent_pi_bounds()
    outside = 3 * math.pi / 2
    inside = math.nextafter(outside, math.inf)
    assert 2 * Fraction(outside) / 3 < low < high < 2 * Fraction(inside) / 3
    assert certified_two_neighbor_phase(0.0, inside, 0.0) is not None
    assert certified_two_neighbor_phase(0.0, outside, 0.0) is None


@pytest.mark.parametrize(
    "center,first,second",
    [
        (0.0, math.pi, 0.0),
        (0.0, 1.6, math.tau - 1.6),
        (math.pi, 0.0, math.pi),
    ],
)
def test_uncertified_or_outside_half_circle_uses_fallback(center, first, second):
    assert certified_two_neighbor_phase(center, first, second) is None


def test_undecided_branch_uses_fallback_instead_of_selecting_a_lift(monkeypatch):
    monkeypatch.setattr(owner, "_pi_bounds", lambda: (Fraction(3), Fraction(4)))
    assert certified_two_neighbor_phase(0.0, 1.6, 0.0) is None


def test_undecided_irrational_rounding_uses_fallback(monkeypatch):
    monkeypatch.setattr(
        owner, "_pi_bounds", lambda: (Fraction(314, 100), Fraction(315, 100))
    )
    assert certified_two_neighbor_phase(0.0, 0.25, math.tau - 0.25) is None


@pytest.mark.parametrize("position", [0, 1, 2])
@pytest.mark.parametrize(
    "value",
    [
        True,
        1,
        Fraction(1, 2),
        "0.5",
        None,
        math.nan,
        math.inf,
        -math.inf,
        -0.1,
        math.tau,
        math.nextafter(math.tau, math.inf),
    ],
)
def test_malformed_nonfinite_and_unnormalized_inputs_request_fallback(position, value):
    values = [0.5, 0.25, 0.75]
    values[position] = value
    assert certified_two_neighbor_phase(*values) is None


def test_signed_zero_input_has_canonical_exact_zero_outputs():
    result = certified_two_neighbor_phase(-0.0, -0.0, 0.0)
    assert result is not None
    assert result.delta.hex() == result.mean.hex() == (0.0).hex()


def test_missing_ieee_precondition_requests_fallback(monkeypatch):
    monkeypatch.setattr(owner, "uses_ieee_binary64_rounding", lambda: False)
    assert certified_two_neighbor_phase(1.0, 0.75, 1.25) is None


def test_result_is_frozen():
    result = certified_two_neighbor_phase(1.0, 0.75, 1.25)
    assert result is not None
    assert result.method == "exact_two_neighbor_midpoint"
    with pytest.raises(FrozenInstanceError):
        result.delta = 123.0
