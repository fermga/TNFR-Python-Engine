"""Exact rank and recurrence boundaries use independently checkable witnesses."""

from fractions import Fraction as Q

import numpy as np
import pytest

from tnfr.mathematics.krylov import (
    exact_rank,
    hankel_matrix,
    hankel_rank,
    krylov_dimension,
    matvec,
    moment_sequence,
)


@pytest.mark.parametrize("size", (2**54, 10**400))
def test_large_integer_rank_retains_nonzero_determinant(size):
    matrix = ((size, size + 1), (size + 1, size + 2))
    assert matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0] == -1
    assert exact_rank(matrix) == 2
    assert exact_rank((*matrix, tuple(a + b for a, b in zip(*matrix)))) == 2
    assert exact_rank(((size, size + 1), (2 * size, 2 * size + 2))) == 1


def test_large_integer_seed_retains_two_nearly_collinear_krylov_directions():
    size = 2**54
    generator, seed = ((0, 1), (-1, 2)), (size, size + 1)
    image = matvec(generator, seed)
    assert image == [size + 1, size + 2]
    assert seed[0] * image[1] - seed[1] * image[0] == -1
    assert krylov_dimension(generator, seed) == 2
    assert krylov_dimension(((1, 0), (0, 1)), seed) == 1


@pytest.mark.parametrize("entry", (np.int64(2**62), Q(np.int64(2**62), np.int64(3))))
def test_numpy_rational_payloads_are_normalized_before_products(entry):
    exact = Q(int(entry.numerator), int(entry.denominator))
    result = matvec(((entry,),), (entry,))
    moments = moment_sequence(((entry,),), (entry,), 3)
    assert result == [exact**2]
    assert moments == [exact**2, exact**3, exact**4]
    assert all(
        type(value.numerator) is type(value.denominator) is int for value in moments
    )


def test_represented_floats_and_small_fractions_retain_their_declared_values():
    represented = Q.from_float(0.1)
    assert moment_sequence(((0.1,),), (0.1,), 3) == [
        represented**2,
        represented**3,
        represented**4,
    ]
    tiny = Q(1, 10**400)
    assert exact_rank(((tiny, 0), (0, tiny))) == 2
    assert matvec(((tiny,),), (tiny,)) == [tiny**2]
    assert hankel_matrix((0.1, tiny, 0)) == [[represented, tiny], [tiny, Q(0)]]


def test_hankel_visibility_remains_distinct_from_krylov_reachability():
    generator, seed = ((0, 0), (1, 0)), (1, 0)
    assert krylov_dimension(generator, seed) == 2
    assert moment_sequence(generator, seed, 5) == [Q(1), Q(0), Q(0), Q(0), Q(0)]
    assert hankel_rank(generator, seed) == 1
    assert hankel_rank(((2, 0), (0, 5)), (1, 1)) == 2
    assert hankel_rank(((2, 0), (0, 5)), (1, 1), count=1) == 1


def test_empty_shapes_and_zero_counts_keep_their_exact_meaning():
    assert exact_rank(()) == exact_rank(((), ())) == 0
    assert matvec((), ()) == []
    assert krylov_dimension((), ()) == hankel_rank((), ()) == 0
    assert moment_sequence((), (), 3) == [Q(0)] * 3
    assert hankel_matrix(()) == []
    assert moment_sequence(((7,),), (1,), 0) == []
    assert hankel_rank(((7,),), (1,), count=0) == 0
    assert moment_sequence(((2,),), (3,), np.int64(2)) == [Q(9), Q(18)]


@pytest.mark.parametrize(
    "bad", (True, np.bool_(False), "1", 1j, float("nan"), float("inf"))
)
def test_invalid_scalars_cannot_hide_in_unused_or_zero_inputs(bad):
    operations = (
        lambda: exact_rank(((0, bad),)),
        lambda: matvec(((bad,),), (0,)),
        lambda: krylov_dimension(((bad,),), (0,)),
        lambda: moment_sequence(((bad,),), (0,), 0),
        lambda: moment_sequence(((1,),), (bad,), 0),
        lambda: hankel_rank(((bad,),), (0,), count=0),
        # For an even moment count the final entry is not used in the matrix.
        lambda: hankel_matrix((1, bad)),
    )
    for operation in operations:
        with pytest.raises((TypeError, ValueError)):
            operation()


@pytest.mark.parametrize("matrix", (((1,), (2, 3)), ((), (1,))))
def test_rank_rejects_ragged_rows_instead_of_truncating(matrix):
    with pytest.raises(ValueError, match="rectangular"):
        exact_rank(matrix)


@pytest.mark.parametrize("matrix", ((), ((1, 99),), ((1,), (99,))))
def test_system_shape_must_match_even_when_no_moment_is_requested(matrix):
    for operation in (
        lambda: matvec(matrix, (1,)),
        lambda: krylov_dimension(matrix, (1,)),
        lambda: moment_sequence(matrix, (1,), 0),
        lambda: hankel_rank(matrix, (1,), count=0),
    ):
        with pytest.raises(ValueError, match="square.*vector dimension"):
            operation()


@pytest.mark.parametrize("count", (True, np.bool_(True), 1.0, Q(1), -1))
def test_counts_do_not_truncate_or_accept_booleans_or_negative_values(count):
    error = ValueError if type(count) is int else TypeError
    with pytest.raises(error, match="count"):
        moment_sequence(((1,),), (1,), count)
    with pytest.raises(error, match="count"):
        hankel_rank(((1,),), (1,), count=count)


@pytest.mark.parametrize("values", ({1, 2}, {0: 1}, "12"))
def test_unordered_or_text_collections_do_not_supply_matrix_coordinates(values):
    with pytest.raises(TypeError, match="ordered collection"):
        exact_rank(values)
    with pytest.raises(TypeError, match="ordered collection"):
        matvec(((1, 0), (0, 1)), values)
    with pytest.raises(TypeError, match="ordered collection"):
        hankel_matrix(values)


def test_one_shot_inputs_are_detached_before_recurrence():
    rows = (iter(row) for row in ((2, 0), (0, 5)))
    seed = iter((1, 1))
    assert moment_sequence(rows, seed, 4) == [Q(2), Q(7), Q(29), Q(133)]
    assert exact_rank(iter((iter((1, 2)), iter((2, 4))))) == 1
