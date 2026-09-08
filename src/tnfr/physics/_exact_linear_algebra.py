"""Private exact rational linear-algebra primitives for physics proofs."""

from __future__ import annotations

from fractions import Fraction

__all__ = ["exact_matrix_inverse", "exact_symmetric_semidefinite"]


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
