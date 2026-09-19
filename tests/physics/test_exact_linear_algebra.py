"""Tests for closed exact rational linear-algebra primitives."""

from fractions import Fraction
from itertools import product

import pytest

from tnfr.physics._exact_linear_algebra import (
    exact_matrix_inverse,
    exact_square_matrix_power,
    exact_square_matrix_product,
    exact_symmetric_semidefinite,
)


def _matrix(*rows: tuple[int, ...]) -> tuple[tuple[Fraction, ...], ...]:
    return tuple(tuple(Fraction(value) for value in row) for row in rows)


def test_exact_square_matrix_product_uses_rational_arithmetic() -> None:
    left = (
        (Fraction(1, 2), Fraction(2, 3)),
        (Fraction(-3, 5), Fraction(7, 11)),
    )
    right = (
        (Fraction(5, 7), Fraction(-1, 3)),
        (Fraction(4, 9), Fraction(6, 13)),
    )

    assert exact_square_matrix_product(left, right) == (
        (Fraction(247, 378), Fraction(11, 78)),
        (Fraction(-101, 693), Fraction(353, 715)),
    )


def test_exact_square_matrix_power_handles_zero_and_binary_exponentiation() -> None:
    fibonacci = _matrix((1, 1), (1, 0))

    assert exact_square_matrix_power(fibonacci, 0) == _matrix((1, 0), (0, 1))
    assert exact_square_matrix_power(fibonacci, 1) == fibonacci
    assert exact_square_matrix_power(fibonacci, 8) == _matrix((34, 21), (21, 13))


@pytest.mark.parametrize(
    ("matrix", "error", "message"),
    [
        ([], TypeError, "exact tuple"),
        ((), ValueError, "positive dimension"),
        (
            ((Fraction(1), Fraction(0)), [Fraction(0), Fraction(1)]),
            TypeError,
            "rows",
        ),
        (((Fraction(1), Fraction(0)),), ValueError, "square"),
        (((Fraction(1), 0), (Fraction(0), Fraction(1))), TypeError, "Fraction"),
        (((Fraction(1), True), (Fraction(0), Fraction(1))), TypeError, "Fraction"),
    ],
)
def test_exact_square_matrix_product_rejects_open_or_malformed_inputs(
    matrix: object,
    error: type[Exception],
    message: str,
) -> None:
    identity = _matrix((1, 0), (0, 1))

    with pytest.raises(error, match=message):
        exact_square_matrix_product(matrix, identity)  # type: ignore[arg-type]


def test_exact_square_matrix_product_rejects_different_dimensions() -> None:
    with pytest.raises(ValueError, match="same dimension"):
        exact_square_matrix_product(_matrix((1,)), _matrix((1, 0), (0, 1)))


@pytest.mark.parametrize("exponent", [True, False, 1.0, Fraction(1)])
def test_exact_square_matrix_power_rejects_non_builtin_integer_exponents(
    exponent: object,
) -> None:
    with pytest.raises(TypeError, match="exact integer"):
        exact_square_matrix_power(_matrix((1,)), exponent)  # type: ignore[arg-type]


def test_exact_square_matrix_power_rejects_negative_exponents() -> None:
    with pytest.raises(ValueError, match="nonnegative"):
        exact_square_matrix_power(_matrix((1,)), -1)


def test_type_rejection_does_not_invoke_foreign_protocols() -> None:
    class ProtocolTrap:
        def __iter__(self):
            raise AssertionError("iteration protocol must not be called")

        def __len__(self):
            raise AssertionError("length protocol must not be called")

        def __index__(self):
            raise AssertionError("index protocol must not be called")

    trap = ProtocolTrap()
    identity = _matrix((1,))

    with pytest.raises(TypeError, match="exact tuple"):
        exact_square_matrix_product(trap, identity)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="exact integer"):
        exact_square_matrix_power(identity, trap)  # type: ignore[arg-type]


def test_closed_types_reject_container_scalar_and_integer_subclasses() -> None:
    class TupleSubclass(tuple):
        pass

    class FractionSubclass(Fraction):
        pass

    class IntSubclass(int):
        pass

    identity = _matrix((1,))

    with pytest.raises(TypeError, match="exact tuple"):
        exact_square_matrix_product(TupleSubclass(identity), identity)
    with pytest.raises(TypeError, match="rows"):
        exact_square_matrix_product((TupleSubclass(identity[0]),), identity)
    with pytest.raises(TypeError, match="Fraction"):
        exact_square_matrix_product(((FractionSubclass(1),),), identity)
    with pytest.raises(TypeError, match="exact integer"):
        exact_square_matrix_power(identity, IntSubclass(1))


def test_inverse_preserves_exact_arithmetic_with_a_row_exchange() -> None:
    matrix = _matrix((0, 2, 1), (1, 1, 0), (3, 0, 2))

    inverse = exact_matrix_inverse(matrix)

    assert inverse == (
        (Fraction(-2, 7), Fraction(4, 7), Fraction(1, 7)),
        (Fraction(2, 7), Fraction(3, 7), Fraction(-1, 7)),
        (Fraction(3, 7), Fraction(-6, 7), Fraction(2, 7)),
    )
    assert all(type(value) is Fraction for row in inverse for value in row)
    assert matrix == _matrix((0, 2, 1), (1, 1, 0), (3, 0, 2))
    for left, right in ((matrix, inverse), (inverse, matrix)):
        assert all(
            sum(left[i][k] * right[k][j] for k in range(3)) == int(i == j)
            for i in range(3)
            for j in range(3)
        )


def test_inverse_accepts_nonsymmetric_exact_matrices() -> None:
    assert exact_matrix_inverse(_matrix((1, 10), (0, 1))) == _matrix((1, -10), (0, 1))
    assert exact_matrix_inverse(((Fraction(3),),)) == ((Fraction(1, 3),),)


@pytest.mark.parametrize("matrix", [_matrix((0,)), _matrix((1, 2), (2, 4))])
def test_inverse_rejects_singular_exact_matrices(matrix) -> None:
    with pytest.raises(ValueError, match="singular"):
        exact_matrix_inverse(matrix)


@pytest.mark.parametrize(
    "operation", [exact_matrix_inverse, exact_symmetric_semidefinite]
)
@pytest.mark.parametrize(
    ("matrix", "error", "message"),
    [
        (None, TypeError, "exact tuple"),
        ([], TypeError, "exact tuple"),
        ((), ValueError, "positive dimension"),
        (((Fraction(1), Fraction(99)),), ValueError, "square"),
        (((Fraction(1),), (Fraction(0), Fraction(1))), ValueError, "square"),
        (([Fraction(1)],), TypeError, "rows"),
        (((1,),), TypeError, "Fraction"),
        (((True,),), TypeError, "Fraction"),
        (((3.0,),), TypeError, "Fraction"),
        (((float("nan"),),), TypeError, "Fraction"),
        (((float("inf"),),), TypeError, "Fraction"),
        ((("1",),), TypeError, "Fraction"),
    ],
)
def test_inverse_and_semidefinite_reject_malformed_exact_inputs(
    operation,
    matrix,
    error,
    message,
) -> None:
    with pytest.raises(error, match=message):
        operation(matrix)


@pytest.mark.parametrize(
    "operation", [exact_matrix_inverse, exact_symmetric_semidefinite]
)
def test_inverse_and_semidefinite_reject_foreign_types_without_coercion(
    operation,
) -> None:
    class ProtocolTrap:
        def __iter__(self):
            raise AssertionError("foreign iteration must not be called")

        def __len__(self):
            raise AssertionError("foreign length must not be called")

    class TupleSubclass(tuple):
        pass

    class FractionSubclass(Fraction):
        pass

    with pytest.raises(TypeError, match="exact tuple"):
        operation(ProtocolTrap())
    with pytest.raises(TypeError, match="exact tuple"):
        operation(TupleSubclass(_matrix((1,))))
    with pytest.raises(TypeError, match="rows"):
        operation((TupleSubclass((Fraction(1),)),))
    with pytest.raises(TypeError, match="Fraction"):
        operation(((FractionSubclass(1),),))


@pytest.mark.parametrize("strict", [False, True])
@pytest.mark.parametrize(
    "matrix",
    [_matrix((1, 10), (0, 1)), _matrix((1, 0), (10, 1)), _matrix((0, 0), (10, 1))],
)
def test_semidefinite_rejects_asymmetry_instead_of_accepting_false_psd(
    matrix, strict
) -> None:
    with pytest.raises(ValueError, match="exactly symmetric"):
        exact_symmetric_semidefinite(matrix, strict=strict)


@pytest.mark.parametrize("strict", [0, 1, None, "false", [], Fraction(1)])
def test_semidefinite_requires_a_builtin_boolean_strict_flag(strict) -> None:
    with pytest.raises(ValueError, match="exact boolean"):
        exact_symmetric_semidefinite(_matrix((1,)), strict=strict)


def test_semidefinite_does_not_coerce_foreign_strict_truth_values() -> None:
    class TruthTrap:
        def __bool__(self):
            raise AssertionError("foreign truth coercion must not be called")

    with pytest.raises(ValueError, match="exact boolean"):
        exact_symmetric_semidefinite(_matrix((1,)), strict=TruthTrap())  # type: ignore[arg-type]


def test_semidefinite_matches_two_dimensional_principal_minor_oracle() -> None:
    for a, b, d in product((Fraction(value, 3) for value in range(-2, 3)), repeat=3):
        matrix = ((a, b), (b, d))
        determinant = a * d - b * b
        assert exact_symmetric_semidefinite(matrix) is (
            a >= 0 and d >= 0 and determinant >= 0
        )
        assert exact_symmetric_semidefinite(matrix, strict=True) is (
            a > 0 and determinant > 0
        )
        assert matrix == ((a, b), (b, d))


@pytest.mark.parametrize(
    ("matrix", "semidefinite", "definite"),
    [
        (_matrix((0, 0, 0), (0, 0, 0), (0, 0, 0)), True, False),
        (_matrix((0, 0, 0), (0, 2, 0), (0, 0, 3)), True, False),
        (_matrix((1, 1, 1), (1, 1, 1), (1, 1, 1)), True, False),
        (_matrix((2, -1, 0), (-1, 2, -1), (0, -1, 2)), True, True),
        (_matrix((1, 2, 0), (2, 1, 0), (0, 0, 1)), False, False),
    ],
)
def test_semidefinite_preserves_singular_and_definite_symmetric_cases(
    matrix,
    semidefinite,
    definite,
) -> None:
    assert exact_symmetric_semidefinite(matrix) is semidefinite
    assert exact_symmetric_semidefinite(matrix, strict=True) is definite
