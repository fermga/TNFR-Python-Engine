"""The p-adic adapter preserves exact values and independent mutable rows."""

from fractions import Fraction as Q

import pytest

from tnfr.mathematics import padic_tower as pt


def test_rectangular_padic_product_preserves_exact_cancellation_and_tiny_values():
    tiny = Q(1, 10**400)
    left = [[Q(1, 3), Q(1, 3), Q(0)], [tiny, -tiny, Q(2, 7)]]
    right = [[Q(3), Q(-5), Q(0)], [Q(-3), Q(5), Q(0)], [Q(0), Q(0), Q(7, 2)]]

    result = pt._matmul(left, right)

    assert result == [[Q(0), Q(0), Q(0)], [6 * tiny, -10 * tiny, Q(1)]]
    assert type(result) is list and all(type(row) is list for row in result)
    assert all(type(value) is Q for row in result for value in row)


def test_product_rebuilds_after_input_mutation_and_returns_independent_rows():
    left = [[Q(1), Q(0)], [Q(0), Q(1)]]
    right = [[Q(0), Q(2)], [Q(3), Q(0)]]
    first = pt._matmul(left, right)
    right[0][0] = Q(1, 7)
    second = pt._matmul(left, right)

    assert first == [[Q(0), Q(2)], [Q(3), Q(0)]]
    assert second == [[Q(1, 7), Q(2)], [Q(3), Q(0)]]
    second[0][0] = Q(99)
    assert right[0][0] == Q(1, 7)
    assert second[1][0] == Q(3)


@pytest.mark.parametrize(
    "left,right,expected",
    [([], [[Q(1)]], []), ([[], []], [[], []], [[], []])],
)
def test_product_retains_empty_list_output_geometry(left, right, expected):
    assert pt._matmul(left, right) == expected


@pytest.mark.parametrize(
    "left,right",
    [([], []), ([[Q(0)]], [[Q(0)], [Q(0)]]), ([[Q(0), Q(0)]], [[Q(0)], []])],
)
def test_zero_skipping_does_not_hide_missing_matrix_indices(left, right):
    with pytest.raises(IndexError):
        pt._matmul(left, right)
