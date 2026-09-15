"""Shared exact coordinates and energy for a simple unit-conductance cycle.

These private kernels assume an ordered cycle, without identifying a live
graph, phase chart or operator execution. Public observers validate their
coefficients and dimensions before using the kernels.
"""

from collections.abc import Mapping, Set
from fractions import Fraction

from .._exact_time import exact_or_represented_real

Vector = tuple[Fraction, ...]
Matrix = tuple[Vector, ...]


def ordered_vector(values, label: str) -> Vector:
    """Retain exact rationals and materialize other supported real values."""
    if isinstance(values, (str, bytes, bytearray, Mapping, Set)):
        raise TypeError(f"{label} must be an ordered sequence")
    try:
        items = tuple(values)
    except TypeError as exc:
        raise TypeError(f"{label} must be an ordered sequence") from exc
    return tuple(exact_or_represented_real(value, label) for value in items)


def dot(left: Vector, right: Vector) -> Fraction:
    return sum((a * b for a, b in zip(left, right, strict=True)), Fraction(0))


def laplacian_matrix(count: int) -> Matrix:
    """Return L_rw=I-W/2 on the declared simple cycle."""
    if count < 3:
        raise ValueError("a simple cycle requires at least three coordinates")
    return tuple(
        tuple(
            Fraction(1) if i == j else
            Fraction(-1, 2) if (j - i) % count in (1, count - 1) else
            Fraction(0)
            for j in range(count)
        )
        for i in range(count)
    )


def laplacian_action(values: Vector) -> Vector:
    """Apply the same cycle Laplacian in linear time."""
    count = len(values)
    if count < 3:
        raise ValueError("a simple cycle requires at least three coordinates")
    return tuple(
        value - (values[i - 1] + values[(i + 1) % count]) / 2
        for i, value in enumerate(values)
    )


def dirichlet_energy(values: Vector) -> Fraction:
    """Return x^T B x/2, with conductance Laplacian B=2L_rw."""
    count = len(values)
    if count < 3:
        raise ValueError("a simple cycle requires at least three coordinates")
    return sum(
        ((values[(i + 1) % count] - value) ** 2
         for i, value in enumerate(values)),
        Fraction(0),
    ) / 2
