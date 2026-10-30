"""Independent exact controls for the positive matrix exponential majorant."""

from dataclasses import FrozenInstanceError
from fractions import Fraction as Q
from itertools import repeat
from math import factorial

import pytest

from tnfr.mathematics import _comparison_flow as owner


def _kernel(matrix, duration=Q(1)):
    return owner.nonnegative_exponential_upper(matrix, duration)


def _identity(size):
    return tuple(tuple(Q(i == j) for j in range(size)) for i in range(size))


def _exponential_bracket(value):
    """A separate degree-96 positive series with a scalar geometric tail."""
    assert 0 <= value <= 32
    lower = sum((value**k / factorial(k) for k in range(97)), Q(0))
    first = value**97 / factorial(97)
    return lower, lower + first / (1 - value / 98)


@pytest.mark.parametrize("size", (1, 4, 24))
def test_zero_matrix_and_zero_clock_give_exact_identity(size):
    zero = tuple((Q(0),) * size for _ in range(size))
    assert _kernel(zero).upper == _identity(size)
    diagonal = tuple(
        tuple(Q(1000) if i == j else Q(0) for j in range(size)) for i in range(size)
    )
    result = _kernel(diagonal, Q(0))
    assert result.upper == result.partial_sum == _identity(size)
    assert result.tail_upper_bound == zero
    assert result.scaled_row_sum_upper_bound == 0


@pytest.mark.parametrize("value", (Q(1, 8), Q(1), Q(32)))
def test_scalar_upper_bound_encloses_independent_exponential_bracket(value):
    report = _kernel(((value,),))
    lower, upper = _exponential_bracket(value)
    assert lower <= upper <= report.upper[0][0]
    assert report.upper[0][0] == (
        report.partial_sum[0][0] + report.tail_upper_bound[0][0]
    )
    assert report.scaled_row_sum_upper_bound == value
    if value <= 1:
        assert report.upper[0][0] - lower < Q(1, 10**35)


def test_diagonal_matrix_keeps_all_off_diagonal_entries_exactly_zero():
    values = (Q(0), Q(1, 8), Q(1), Q(2))
    matrix = tuple(
        tuple(value if i == j else Q(0) for j in range(4))
        for i, value in enumerate(values)
    )
    report = _kernel(matrix)
    for i, value in enumerate(values):
        lower, upper = _exponential_bracket(value)
        assert lower <= upper <= report.upper[i][i]
        assert all(report.upper[i][j] == 0 for j in range(4) if j != i)


def test_nilpotent_chain_is_its_exact_finite_polynomial_including_tiny_path():
    epsilon, h = Q(1, 1 << 200), Q(3, 2)
    matrix = ((0, 2, 0, 0), (0, 0, epsilon, 0), (0, 0, 0, 3), (0, 0, 0, 0))
    expected = (
        (1, 2 * h, epsilon * h**2, epsilon * h**3),
        (0, 1, epsilon * h, 3 * epsilon * h**2 / 2),
        (0, 0, 1, 3 * h),
        (0, 0, 0, 1),
    )
    report = _kernel(matrix, h)
    assert report.upper == report.partial_sum == expected
    assert report.tail_upper_bound == ((0,) * 4,) * 4
    assert 0 < report.upper[0][3] < Q(1, 1 << 128)


def test_rank_one_projector_reduces_to_one_scalar_exponential():
    # P**2=P, so exp(tP)=Id+(exp(t)-1)P independently of matrix code.
    projector = ((Q(1, 3), Q(2, 3)), (Q(1, 3), Q(2, 3)))
    report = _kernel(projector, Q(1, 2))
    lower, upper = _exponential_bracket(Q(1, 2))
    for i in range(2):
        for j in range(2):
            lo = Q(i == j) + (lower - 1) * projector[i][j]
            hi = Q(i == j) + (upper - 1) * projector[i][j]
            assert lo <= hi <= report.upper[i][j]
            assert report.upper[i][j] - lo < Q(1, 10**35)


def test_reciprocal_two_node_kernel_matches_even_and_odd_scalar_series():
    # A**2=Id/4: the diagonal is cosh(h/2), off-diagonal is 2*A*sinh(h/2).
    matrix, h = ((0, Q(1, 4)), (1, 0)), Q(1, 2)
    value = h / 2
    even = sum((value ** (2 * k) / factorial(2 * k) for k in range(49)), Q(0))
    odd = sum((value ** (2 * k + 1) / factorial(2 * k + 1) for k in range(48)), Q(0))
    report = _kernel(matrix, h)
    expected_lower = ((even, odd / 2), (2 * odd, even))
    for row, lower in zip(report.upper, expected_lower):
        for bound, exact_lower in zip(row, lower):
            assert exact_lower <= bound < exact_lower + Q(1, 10**35)


def test_block_zeros_and_continuous_tiny_link_survive_the_resolvent_tail():
    epsilon = Q(1, 1 << 200)
    separate = ((1, 0), (0, 1))
    linked = ((1, epsilon), (0, 1))
    left, right = _kernel(separate), _kernel(linked)
    assert left.upper[0][1] == 0
    assert right.upper[1][0] == 0
    assert right.upper[0][0] == left.upper[0][0] == right.upper[1][1]
    # exp(Id+epsilon*N)=e*(Id+epsilon*N), N**2=0. A scalar tail
    # distributed over reachable entries would lose this linear smallness.
    lower, upper = _exponential_bracket(Q(1))
    assert epsilon * lower <= epsilon * upper <= right.upper[0][1]
    assert right.upper[0][1] < 3 * epsilon
    assert right.tail_upper_bound[0][1] > 0


def test_exact_relabeling_and_clock_rescaling_commute_with_kernel():
    matrix = ((Q(1, 4), Q(1, 8), 0), (0, Q(1, 2), Q(1, 3)), (0, 0, 0))
    order, h, c = (2, 0, 1), Q(1, 4), Q(3, 2)
    report = _kernel(matrix, c * h)
    rescaled = _kernel(tuple(tuple(c * x for x in row) for row in matrix), h)
    permuted = _kernel(tuple(tuple(matrix[i][j] for j in order) for i in order), c * h)
    assert report.upper == rescaled.upper
    assert permuted.upper == tuple(
        tuple(report.upper[i][j] for j in order) for i in order
    )


def test_fixed_width_fraction_payloads_normalize_before_exact_matrix_work():
    np = pytest.importorskip("numpy")

    def payload(value):
        return Q(np.int64(value.numerator), np.int64(value.denominator))

    matrix, time = ((Q(0), Q(1, 3)), (Q(1, 4), Q(0))), Q(1, 2)
    report = _kernel(
        tuple(tuple(payload(x) for x in row) for row in matrix), payload(time)
    )
    assert report.upper == _kernel(matrix, time).upper
    assert all(
        type(x.numerator) is type(x.denominator) is int
        for row in report.upper
        for x in row
    )


def test_kernel_report_is_frozen_and_retains_normalized_primitives():
    report = _kernel(((0, 1), (1, 0)), Q(1, 4))
    assert report.matrix == ((Q(0), Q(1)), (Q(1), Q(0)))
    assert report.duration == Q(1, 4)
    with pytest.raises(FrozenInstanceError):
        report.duration = Q(2)


@pytest.mark.parametrize(
    "matrix,duration",
    (
        ((), Q(1)),
        (((0, 1),), Q(1)),
        (((0,), (1,)), Q(1)),
        (((-1,),), Q(1)),
        (((True,),), Q(1)),
        (((1.0,),), Q(1)),
        (((float("nan"),),), Q(1)),
        (((float("inf"),),), Q(1)),
        (((0,),), True),
        (((0,),), Q(-1)),
        (((Q(32) + Q(1, 1 << 200),),), Q(1)),
        (((0,) * 25,) * 25, Q(0)),
        ({0: (1,)}, Q(1)),
        (({1},), Q(1)),
    ),
)
def test_invalid_kernel_primitives_are_rejected_before_inverse_or_dyadic_work(
    matrix, duration, monkeypatch
):
    def forbidden(*args, **kwargs):
        pytest.fail("invalid kernel input reached numerical work")

    monkeypatch.setattr(owner, "I", forbidden)
    monkeypatch.setattr(owner, "exact_matrix_inverse", forbidden)
    monkeypatch.setattr(owner, "exact_square_matrix_product", forbidden)
    with pytest.raises((TypeError, ValueError, OverflowError)):
        _kernel(matrix, duration)


def test_unbounded_outer_matrix_and_row_iterables_are_admitted_with_a_limit():
    with pytest.raises((TypeError, ValueError)):
        _kernel(repeat((0,)), Q(0))
    with pytest.raises((TypeError, ValueError)):
        _kernel((repeat(0),), Q(0))
