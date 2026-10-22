"""Private exact rational linear-algebra primitives shared by model proofs."""

from __future__ import annotations

from collections.abc import Mapping, Set
from fractions import Fraction
from math import isqrt

from .._exact_time import exact_or_represented_real

__all__ = [
    "ExactSquareMatrix",
    "exact_matrix_inverse",
    "exact_matrix_product",
    "exact_rank_one_rational_lift",
    "exact_square_matrix_power",
    "exact_square_matrix_product",
    "exact_symmetric_semidefinite",
]

ExactSquareMatrix = tuple[tuple[Fraction, ...], ...]


def _ordered(values, label):
    """Detach an ordered input without accepting mappings or unordered sets."""
    if isinstance(values, (str, bytes, bytearray, Mapping, Set)):
        raise TypeError(f"{label} must be an ordered collection")
    try:
        return tuple(values)
    except TypeError as exc:
        raise TypeError(f"{label} must be an ordered collection") from exc


def _rational_vector(values, label):
    """Admit rational or represented-real entries before exact arithmetic."""
    return tuple(
        exact_or_represented_real(value, f"{label}[{i}]")
        for i, value in enumerate(_ordered(values, label))
    )


def _rational_matrix(values, label, *, allow_empty=False):
    """Detach and normalize rows; the consumer admits its required shape."""
    rows = _ordered(values, label)
    if not rows and not allow_empty:
        raise ValueError(f"{label} must be nonempty")
    return tuple(_rational_vector(row, f"{label}[{i}]") for i, row in enumerate(rows))


def exact_rank_one_rational_lift(matrix: object) -> tuple[Fraction, ...]:
    """Factor a nonempty positive-semidefinite rank-at-most-one matrix exactly.

    Ordered rows admit exact rational or finite represented-real entries and
    normalize them before algebra. The first positive diagonal chooses a
    positive coordinate of the canonical factor; the zero matrix gives the
    zero vector. A valid real factor without a rational square root is outside
    this decoder's domain and raises ``ValueError`` instead of being rounded.
    The factor's sign convention is not a globally smooth coordinate chart.
    """
    values = _rational_matrix(matrix, "matrix")
    dimension = len(values)
    if any(len(row) != dimension for row in values):
        raise ValueError("matrix must be square")
    if any(
        values[i][j] != values[j][i]
        for i in range(dimension)
        for j in range(i + 1, dimension)
    ):
        raise ValueError("matrix must be exactly symmetric")
    cone_error = "matrix must be positive semidefinite with rank at most one"
    if any(values[i][i] < 0 for i in range(dimension)):
        raise ValueError(cone_error)
    pivot = next((i for i in range(dimension) if values[i][i] > 0), None)
    if pivot is None:
        if any(value for row in values for value in row):
            raise ValueError(cone_error)
        return (Fraction(0),) * dimension

    diagonal = values[pivot][pivot]
    column = tuple(row[pivot] for row in values)
    if any(
        diagonal * values[i][j] != column[i] * column[j]
        for i in range(dimension)
        for j in range(dimension)
    ):
        raise ValueError(cone_error)
    numerator = isqrt(diagonal.numerator)
    denominator = isqrt(diagonal.denominator)
    if (
        numerator * numerator != diagonal.numerator
        or denominator * denominator != diagonal.denominator
    ):
        raise ValueError(
            "matrix admits a real lift but this decoder requires an exact rational "
            "square root"
        )
    root = Fraction(numerator, denominator)
    return tuple(value / root for value in column)


def _require_exact_square_matrix(
    matrix: object,
    *,
    name: str,
) -> ExactSquareMatrix:
    """Admit closed Fraction matrices and normalize their integer payloads.

    A Fraction can retain NumPy numerator/denominator integers, whose fixed-
    width arithmetic can overflow. Preserve the strict outer scalar contract
    while rebuilding the admitted rationals with Python integer components.
    """

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
    return tuple(
        tuple(
            exact_or_represented_real(value, f"{name}[{i}][{j}]")
            for j, value in enumerate(row)
        )
        for i, row in enumerate(matrix)
    )


def _exact_square_matrix_product_unchecked(
    left: ExactSquareMatrix,
    right: ExactSquareMatrix,
) -> ExactSquareMatrix:
    """Multiply equally sized validated square rational matrices."""

    return exact_matrix_product(left, right)


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
            Fraction(1) if row == column else Fraction(0) for column in range(dimension)
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
    """Invert a nonempty nonsingular square Fraction matrix by Gauss--Jordan."""

    matrix = _require_exact_square_matrix(matrix, name="matrix")
    dimension = len(matrix)
    augmented = [
        list(row) + [Fraction(1) if i == j else Fraction(0) for j in range(dimension)]
        for i, row in enumerate(matrix)
    ]
    for column in range(dimension):
        pivot_row = next(
            (row for row in range(column, dimension) if augmented[row][column] != 0),
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
                for value, pivot_value in zip(augmented[row], augmented[column])
            ]
    return tuple(tuple(row[dimension:]) for row in augmented)


def exact_symmetric_semidefinite(
    matrix: tuple[tuple[Fraction, ...], ...],
    *,
    strict: bool = False,
) -> bool:
    """Test a nonempty symmetric Fraction matrix by exact LDL elimination.

    Malformed or asymmetric inputs are rejected, not interpreted as an
    indefinite matrix. ``strict`` must be an exact boolean.
    """

    if type(strict) is not bool:
        raise ValueError("strict must be an exact boolean")
    matrix = _require_exact_square_matrix(matrix, name="matrix")
    if any(
        matrix[row][column] != matrix[column][row]
        for row in range(len(matrix))
        for column in range(row + 1, len(matrix))
    ):
        raise ValueError("matrix must be exactly symmetric")
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
                    - work[row][pivot_index] * work[pivot_index][column] / pivot
                )
                work[row][column] = updated
                work[column][row] = updated
    return True


def exact_matrix_product(
    left: tuple[tuple[Fraction, ...], ...],
    right: tuple[tuple[Fraction, ...], ...],
) -> tuple[tuple[Fraction, ...], ...]:
    """Multiply compatible nonempty rational matrices; callers admit shapes."""
    rows = len(left)
    inner = len(right)
    columns = len(right[0])
    if rows == 0 or columns == 0:
        return tuple(() for _ in range(rows))

    # Keep full output rows while avoiding Fraction arithmetic for exact zeros.
    # Indexed columns retain the caller-admitted width without zip truncation.
    right_nonzero = tuple(
        tuple((j, row[j]) for j in range(columns) if row[j]) for row in right
    )
    result = []
    for row in left:
        output = [Fraction(0)] * columns
        for k in range(inner):
            factor = row[k]
            if factor:
                for column, value in right_nonzero[k]:
                    output[column] += factor * value
        result.append(tuple(output))
    return tuple(result)
