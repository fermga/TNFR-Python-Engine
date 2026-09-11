"""Private exact rational linear-algebra primitives for physics proofs."""

from __future__ import annotations

from fractions import Fraction

__all__ = [
    "ExactSquareMatrix",
    "exact_matrix_inverse",
    "exact_square_matrix_power",
    "exact_square_matrix_product",
    "exact_symmetric_semidefinite",
]

ExactSquareMatrix = tuple[tuple[Fraction, ...], ...]


def _require_exact_square_matrix(
    matrix: object,
    *,
    name: str,
) -> ExactSquareMatrix:
    """Return a closed, nonempty square matrix over exact ``Fraction`` values."""

    if type(matrix) is not tuple:
        raise TypeError(f"{name} must be an exact tuple of tuple rows")
    if not matrix:
        raise ValueError(f"{name} must have positive dimension")

    dimension = len(matrix)
    for row in matrix:
        if type(row) is not tuple:
            raise TypeError(f"{name} rows must be exact tuples")
        if len(row) != dimension:
            raise ValueError(f"{name} must be square")
        if any(type(value) is not Fraction for value in row):
            raise TypeError(f"{name} entries must be exact Fraction values")
    return matrix


def _exact_square_matrix_product_unchecked(
    left: ExactSquareMatrix,
    right: ExactSquareMatrix,
) -> ExactSquareMatrix:
    """Multiply equally sized validated square rational matrices."""

    dimension = len(left)
    return tuple(
        tuple(
            sum(
                (
                    left[row][inner] * right[inner][column]
                    for inner in range(dimension)
                ),
                Fraction(0),
            )
            for column in range(dimension)
        )
        for row in range(dimension)
    )


def exact_square_matrix_product(
    left: ExactSquareMatrix,
    right: ExactSquareMatrix,
) -> ExactSquareMatrix:
    """Multiply two equally sized, nonempty square matrices over ``Fraction``."""

    left_matrix = _require_exact_square_matrix(left, name="left matrix")
    right_matrix = _require_exact_square_matrix(right, name="right matrix")
    if len(left_matrix) != len(right_matrix):
        raise ValueError("square matrices must have the same dimension")
    return _exact_square_matrix_product_unchecked(left_matrix, right_matrix)


def exact_square_matrix_power(
    matrix: ExactSquareMatrix,
    exponent: int,
) -> ExactSquareMatrix:
    """Raise a nonempty square rational matrix to a nonnegative integer power."""

    if type(exponent) is not int:
        raise TypeError("matrix exponent must be an exact integer, not a boolean")
    if exponent < 0:
        raise ValueError("matrix exponent must be nonnegative")

    factor = _require_exact_square_matrix(matrix, name="matrix")
    dimension = len(factor)
    result = tuple(
        tuple(
            Fraction(1) if row == column else Fraction(0)
            for column in range(dimension)
        )
        for row in range(dimension)
    )
    power = exponent
    while power:
        if power & 1:
            result = _exact_square_matrix_product_unchecked(result, factor)
        power >>= 1
        if power:
            factor = _exact_square_matrix_product_unchecked(factor, factor)
    return result


def exact_matrix_inverse(
    matrix: tuple[tuple[Fraction, ...], ...],
) -> tuple[tuple[Fraction, ...], ...]:
    """Invert a nonsingular rational matrix by exact Gauss--Jordan steps."""

    dimension = len(matrix)
    augmented = [
        list(row)
        + [Fraction(1) if i == j else Fraction(0) for j in range(dimension)]
        for i, row in enumerate(matrix)
    ]
    for column in range(dimension):
        pivot_row = next(
            (
                row
                for row in range(column, dimension)
                if augmented[row][column] != 0
            ),
            None,
        )
        if pivot_row is None:
            raise ValueError("exact matrix is singular")
        if pivot_row != column:
            augmented[column], augmented[pivot_row] = (
                augmented[pivot_row],
                augmented[column],
            )
        pivot = augmented[column][column]
        augmented[column] = [value / pivot for value in augmented[column]]
        for row in range(dimension):
            if row == column:
                continue
            factor = augmented[row][column]
            if factor == 0:
                continue
            augmented[row] = [
                value - factor * pivot_value
                for value, pivot_value in zip(
                    augmented[row], augmented[column]
                )
            ]
    return tuple(tuple(row[dimension:]) for row in augmented)


def exact_symmetric_semidefinite(
    matrix: tuple[tuple[Fraction, ...], ...],
    *,
    strict: bool = False,
) -> bool:
    """Test rational positive (semi)definiteness by exact LDL elimination."""

    work = [list(row) for row in matrix]
    dimension = len(work)
    for pivot_index in range(dimension):
        pivot = work[pivot_index][pivot_index]
        if pivot < 0 or (strict and pivot == 0):
            return False
        if pivot == 0:
            if any(
                work[pivot_index][column] != 0
                for column in range(pivot_index + 1, dimension)
            ):
                return False
            continue
        for row in range(pivot_index + 1, dimension):
            for column in range(row, dimension):
                updated = (
                    work[row][column]
                    - work[row][pivot_index]
                    * work[pivot_index][column]
                    / pivot
                )
                work[row][column] = updated
                work[column][row] = updated
    return True
