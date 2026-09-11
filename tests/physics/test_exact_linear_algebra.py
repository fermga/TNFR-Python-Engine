"""Tests for closed exact rational linear-algebra primitives."""

from fractions import Fraction

import pytest

from tnfr.physics._exact_linear_algebra import (
    exact_square_matrix_power,
    exact_square_matrix_product,
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
