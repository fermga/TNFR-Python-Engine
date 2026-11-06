"""Independent rational controls for a finite Metzler exponential enclosure."""

from dataclasses import FrozenInstanceError
from fractions import Fraction as Q
from itertools import repeat
from math import factorial

import pytest

from tnfr.mathematics import _comparison_flow as owner


def _decay(value):
    value = Q(value)
    assert 0 <= value <= 1
    upper = sum(((-value) ** k / factorial(k) for k in range(101)), Q(0))
    return upper - value**101 / factorial(101), upper


def _contains(report, expected):
    for i, row in enumerate(expected):
        for j, (lower, upper) in enumerate(row):
            assert report.lower[i][j] <= lower <= upper <= report.upper[i][j]
            assert report.upper[i][j] - report.lower[i][j] < Q(1, 10**30)


@pytest.mark.parametrize(
    "rate,time", [(Q(-1), Q(1, 4)), (Q(-2), Q(1, 4)), (Q(-1), Q(1))]
)
def test_scalar_decay_encloses_independent_alternating_series(rate, time):
    report = owner.metzler_exponential_bounds(((rate,),), time)
    _contains(report, ((_decay(-rate * time),),))
    assert report.diagonal_shift == -rate
    assert report.shifted_matrix == ((Q(0),),)


def test_scalar_growth_and_zero_matrix():
    lower, upper = _decay(Q(1, 4))
    report = owner.metzler_exponential_bounds(((Q(1),),), Q(1, 4))
    _contains(report, (((1 / upper, 1 / lower),),))
    assert report.diagonal_shift == 0
    zero = owner.metzler_exponential_bounds(((0, 0), (0, 0)), Q(1, 4))
    assert zero.lower == zero.upper == ((Q(1), Q(0)), (Q(0), Q(1)))


@pytest.mark.parametrize("second_rate", [Q(1), Q(2)])
def test_two_node_conservative_flow_uses_actual_unequal_rows(second_rate):
    b = second_rate
    lower, upper = _decay((1 + b) / 4)
    expected = (
        (
            ((b + lower) / (1 + b), (b + upper) / (1 + b)),
            ((1 - upper) / (1 + b), (1 - lower) / (1 + b)),
        ),
        (
            ((b - b * upper) / (1 + b), (b - b * lower) / (1 + b)),
            ((1 + b * lower) / (1 + b), (1 + b * upper) / (1 + b)),
        ),
    )
    report = owner.metzler_exponential_bounds(((-1, 1), (b, -b)), Q(1, 4))
    _contains(report, expected)
    for lower_row, upper_row in zip(report.lower, report.upper):
        assert sum(lower_row) <= 1 <= sum(upper_row)


def test_repeated_eigenvalue_retains_nilpotent_offdiagonal_factor():
    lower, upper = _decay(Q(1, 4))
    report = owner.metzler_exponential_bounds(((-1, 1), (0, -1)), Q(1, 4))
    _contains(
        report, (((lower, upper), (lower / 4, upper / 4)), ((0, 0), (lower, upper)))
    )
    assert report.lower[1][0] == report.upper[1][0] == 0


def test_zero_duration_and_growth_domain_boundaries_are_exact():
    report = owner.metzler_exponential_bounds(((-1, 3), (2, -2)), 0)
    assert report.lower == report.upper == ((Q(1), Q(0)), (Q(0), Q(1)))
    boundary = owner.metzler_exponential_bounds(((0, 128), (0, 0)), Q(1, 4))
    assert boundary.lower == boundary.upper == ((Q(1), Q(32)), (Q(0), Q(1)))
    largest_zero = tuple((Q(0),) * 24 for _ in range(24))
    admitted = owner.metzler_exponential_bounds(largest_zero, 0)
    assert len(admitted.lower) == 24


def test_common_clock_and_node_permutation_preserve_exact_enclosures():
    matrix = ((Q(-1), Q(1)), (Q(2), Q(-2)))
    baseline = owner.metzler_exponential_bounds(matrix, Q(1, 4))
    scaled = owner.metzler_exponential_bounds(
        tuple(tuple(3 * x for x in row) for row in matrix), Q(1, 12)
    )
    assert scaled.lower == baseline.lower
    assert scaled.upper == baseline.upper
    permuted = owner.metzler_exponential_bounds(((-2, 2), (1, -1)), Q(1, 4))
    for attribute in ("lower", "upper"):
        original = getattr(baseline, attribute)
        assert getattr(permuted, attribute) == tuple(
            tuple(original[1 - i][1 - j] for j in range(2)) for i in range(2)
        )


@pytest.mark.parametrize(
    "matrix,duration",
    [
        ((), 0),
        (((0, 1),), 0),
        (((0, -1), (1, 0)), 0),
        (((True,),), 0),
        (((float("nan"),),), 0),
        (((float("inf"),),), 0),
        ((("1",),), 0),
        (((0,),), True),
        (((0,),), -1),
        (((0,),), float("inf")),
        (((-2,),), Q(1, 2) + Q(1, 10**100)),
        (((0, 129), (0, 0)), Q(1, 4)),
        (tuple((0,) * 25 for _ in range(25)), 0),
        (repeat((0,)), 0),
        ({(0,)}, 0),
    ],
)
def test_invalid_admission_precedes_all_exponential_work(monkeypatch, matrix, duration):
    def forbidden(*args, **kwargs):
        pytest.fail("invalid primitive reached exponential or matrix work")

    for name in (
        "nonnegative_exponential_upper",
        "exp_unit_bounds",
        "exact_matrix_inverse",
        "exact_square_matrix_product",
    ):
        monkeypatch.setattr(owner, name, forbidden)
    with pytest.raises((ValueError, TypeError, OverflowError)):
        owner.metzler_exponential_bounds(matrix, duration)


def test_normalization_and_report_immutability():
    report = owner.metzler_exponential_bounds([[0, 0], [0, 0]], Q(1, 4))
    assert report.matrix == ((Q(0), Q(0)), (Q(0), Q(0)))
    with pytest.raises(FrozenInstanceError):
        report.duration = Q(9)
