"""Exact rank-one lifting preserves realizability and primitive admission."""

from fractions import Fraction as Q

import numpy as np
import pytest

from tnfr.mathematics._exact_linear_algebra import exact_rank_one_rational_lift


def _gram(vector):
    return tuple(tuple(left * right for right in vector) for left in vector)


@pytest.mark.parametrize(
    "vector",
    [
        (Q(0),),
        (Q(0),) * 18,
        (Q(-3, 7), Q(2, 5), Q(0)),
        (Q(0), Q(-1, 3), Q(8)),
        (Q(1, 7), Q(-11), Q(0)),
        (Q(0), Q(0), Q(-5, 9)),
        (Q(1, 2**1200), Q(-1, 2**1300)),
        (Q(2**1200, 3), Q(-(2**1300), 7)),
    ],
)
def test_canonical_lift_identifies_one_global_sign_without_rounding(vector):
    gram = _gram(vector)
    lifted = exact_rank_one_rational_lift(gram)
    assert _gram(lifted) == gram
    assert lifted in (vector, tuple(-value for value in vector))
    first_nonzero = next((value for value in lifted if value), Q(0))
    assert first_nonzero >= 0
    assert all(type(value) is Q for value in lifted)


def test_lift_detaches_ordered_iterators_and_admits_represented_scalars():
    matrix = [[1.0, -0.5, 0], [-0.5, 0.25, 0], [0, 0, -0.0]]
    lifted = exact_rank_one_rational_lift(iter(iter(row) for row in matrix))
    assert lifted == (Q(1), Q(-1, 2), Q(0))
    matrix[0][0] = 4
    assert lifted == (Q(1), Q(-1, 2), Q(0))


def test_lift_normalizes_fixed_width_fraction_payloads_before_products():
    vector = (Q(2**30), Q(-(2**29)))
    gram = _gram(vector)
    matrix = tuple(
        tuple(
            Q(np.int64(value.numerator), np.int64(value.denominator)) for value in row
        )
        for row in gram
    )
    # Rank-one minor products exceed int64 even though every input fits.
    with np.errstate(over="raise"):
        lifted = exact_rank_one_rational_lift(matrix)
    assert lifted == vector
    assert all(
        type(value.numerator) is type(value.denominator) is int for value in lifted
    )


@pytest.mark.parametrize(
    "matrix",
    [
        ((-1,),),
        ((1, 0), (0, 1)),
        ((0, 1), (1, 0)),
        ((1, 2), (2, 1)),
        ((1, 0), (0, Q(1, 2**2400))),
        ((1, 1, -1), (1, 1, 1), (-1, 1, 1)),
    ],
)
def test_incompatible_joint_data_are_rejected_exactly(matrix):
    with pytest.raises(ValueError, match="positive semidefinite.*rank"):
        exact_rank_one_rational_lift(matrix)


@pytest.mark.parametrize(
    "matrix", [((2,),), ((Q(1, 2),),), ((0, 0), (0, 2)), ((2, -2), (-2, 2))]
)
def test_real_rank_one_factors_without_rational_lifts_are_unsupported(matrix):
    with pytest.raises(ValueError, match="rational"):
        exact_rank_one_rational_lift(matrix)


@pytest.mark.parametrize(
    "matrix, error, message",
    [
        ((), ValueError, "nonempty"),
        (((),), ValueError, "square"),
        (((1, 0),), ValueError, "square"),
        (((1, 0), (0,)), ValueError, "square"),
        (((1, 0), (1, 1)), ValueError, "symmetric"),
        ("1", TypeError, "ordered"),
        ({0: (1,)}, TypeError, "ordered"),
        ({(1,)}, TypeError, "ordered"),
        (({0: 1},), TypeError, "ordered"),
        (({1},), TypeError, "ordered"),
        ((1,), TypeError, "ordered"),
        (((True,),), TypeError, "boolean"),
        (((np.bool_(False),),), TypeError, "boolean"),
        ((("1",),), TypeError, "real"),
        (((1 + 0j,),), TypeError, "real"),
        (((float("nan"),),), ValueError, "finite"),
        (((float("inf"),),), ValueError, "finite"),
    ],
)
def test_primitive_and_shape_admission_precedes_factorization(matrix, error, message):
    with pytest.raises(error, match=message):
        exact_rank_one_rational_lift(matrix)
