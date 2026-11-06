"""Exact sparse products retain dense rational results and output geometry."""

from fractions import Fraction as Q

import pytest

from tnfr.mathematics._exact_linear_algebra import exact_matrix_product


def _dense_product(left, right):
    return tuple(
        tuple(
            sum((left[i][k] * right[k][j] for k in range(len(right))), Q(0))
            for j in range(len(right[0]))
        )
        for i in range(len(left))
    )


@pytest.mark.parametrize("shape", [(1, 1, 1), (2, 3, 4), (5, 2, 1), (4, 4, 4)])
@pytest.mark.parametrize("sparse", [False, True])
def test_product_matches_dense_rational_oracle(shape, sparse):
    rows, inner, columns = shape
    left = tuple(
        tuple(
            Q(0) if sparse and (i + k) % 3 else Q(3 * i - 2 * k + 1, k + 2)
            for k in range(inner)
        )
        for i in range(rows)
    )
    right = tuple(
        tuple(
            Q(0) if sparse and (k + j) % 2 else Q(2 * k + 3 * j - 2, j + 3)
            for j in range(columns)
        )
        for k in range(inner)
    )
    actual = exact_matrix_product(left, right)
    assert actual == _dense_product(left, right)
    assert type(actual) is tuple and len(actual) == rows
    assert all(type(row) is tuple and len(row) == columns for row in actual)
    assert all(type(value) is Q for row in actual for value in row)


def test_product_retains_cancellation_tiny_fractions_and_zero_rows_and_columns():
    tiny = Q(1, 10**400)
    left = ((Q(1, 3), Q(1, 3), Q(0)), (Q(0),) * 3, (tiny, -tiny, Q(2, 7)))
    right = (
        (Q(3), Q(-5), Q(0), Q(7)),
        (Q(-3), Q(5), Q(0), Q(7)),
        (Q(0),) * 3 + (Q(7, 2),),
    )
    actual = exact_matrix_product(left, right)
    assert actual == (
        (Q(0), Q(0), Q(0), Q(14, 3)),
        (Q(0),) * 4,
        (6 * tiny, -10 * tiny, Q(0), Q(1)),
    )
    assert all(type(value) is Q for row in actual for value in row)


@pytest.mark.parametrize(
    "left, right, expected",
    [
        ((), ((Q(1),), ()), ()),
        (((), ()), ((), (Q(9),)), ((), ())),
        (((Q(1), Q(2), Q(999)),), ((Q(3),), (Q(4), Q(999))), ((Q(11),),)),
    ],
)
def test_unchecked_product_preserves_existing_indexed_extents(left, right, expected):
    # Model owners admit compatible shapes; these retain the private helper's
    # existing empty-output paths and bounded indexing without widening its API.
    assert exact_matrix_product(left, right) == expected


@pytest.mark.parametrize(
    "left, right",
    [
        ((), ()),
        (((Q(0),),), ((Q(0),), (Q(0),))),
        (((Q(0), Q(0)),), ((Q(0),), ())),
    ],
)
def test_zero_skipping_does_not_hide_existing_missing_index_errors(left, right):
    with pytest.raises(IndexError):
        exact_matrix_product(left, right)
