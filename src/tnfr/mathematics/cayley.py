"""Exact Cayley random-walk operators shared by arithmetic studies."""

from __future__ import annotations

from fractions import Fraction
import cmath
import math

import numpy as np

Matrix = list[list[Fraction]]

__all__ = [
    "cayley_laplacian",
    "cayley_first_row",
    "cayley_action",
    "cayley_spectrum",
    "cayley_diffusion_action",
]


def _normalized_connection(
    modulus: int, connection: set[int]
) -> tuple[set[int], Fraction]:
    if modulus < 1:
        raise ValueError("modulus must be positive")
    normalized = {int(value) % modulus for value in connection} - {0}
    if not normalized:
        raise ValueError("empty connection set")
    return normalized, Fraction(1, len(normalized))


def cayley_laplacian(modulus: int, connection: set[int]) -> Matrix:
    r"""Return exact ``L_rw = I - (1/d)W`` on ``Z/modulus Z``."""
    normalized, inverse_degree = _normalized_connection(modulus, connection)
    matrix: Matrix = [
        [Fraction(0) for _ in range(modulus)] for _ in range(modulus)
    ]
    for source in range(modulus):
        matrix[source][source] = Fraction(1)
        for target in range(modulus):
            if source != target and (target - source) % modulus in normalized:
                matrix[source][target] = -inverse_degree
    return matrix


def cayley_first_row(modulus: int, connection: set[int]) -> list[Fraction]:
    """Return the exact first row of the Cayley Laplacian circulant."""
    normalized, inverse_degree = _normalized_connection(modulus, connection)
    row = [Fraction(0) for _ in range(modulus)]
    row[0] = Fraction(1)
    for shift in normalized - {0}:
        row[shift] = -inverse_degree
    return row


def cayley_action(
    modulus: int, connection: set[int], vector: list[Fraction]
) -> list[Fraction]:
    """Apply the exact circulant Laplacian without materializing its matrix."""
    if len(vector) != modulus:
        raise ValueError("vector length must equal modulus")
    normalized, inverse_degree = _normalized_connection(modulus, connection)
    shifts = normalized - {0}
    return [
        vector[source] - inverse_degree * sum(
            (vector[(source + shift) % modulus] for shift in shifts),
            Fraction(0),
        )
        for source in range(modulus)
    ]


def cayley_spectrum(modulus: int, connection: set[int]) -> list[complex]:
    """Fourier spectrum of the declared circulant, in canonical node order."""
    normalized, inverse_degree = _normalized_connection(modulus, connection)
    shifts = normalized - {0}
    omega = cmath.exp(2j * math.pi / modulus)
    return [
        1.0 - float(inverse_degree) * sum(
            omega ** (mode * shift) for shift in shifts
        )
        for mode in range(modulus)
    ]


def cayley_diffusion_action(
    modulus: int,
    connection: set[int],
    vector,
    *,
    structural_time: float = 1.0,
    capacity: float = 1.0,
) -> np.ndarray:
    r"""Evolve a circulant EPI field under common-capacity diffusion.

    For the Cayley random-walk Laplacian ``L_rw`` this evaluates the isolated
    EPI channel of the nodal equation,

    ``dEPI/dt = -nu_f L_rw EPI``,

    as ``IFFT(exp(-nu_f * structural_time * lambda_k) * FFT(EPI_0))``.
    The Fourier path is exact up to floating-point arithmetic and avoids
    materializing the dense exponential. It applies only to a fixed circulant
    topology with one common nonnegative capacity; heterogeneous ``nu_f``
    requires the noncommuting-capacity transport path instead.
    """
    if (
        not math.isfinite(structural_time)
        or structural_time < 0.0
    ):
        raise ValueError("structural_time must be finite and nonnegative")
    if not math.isfinite(capacity) or capacity < 0.0:
        raise ValueError("capacity must be finite and nonnegative")

    # Validate the cyclic connection through the shared exact constructor.
    spectrum = np.asarray(cayley_spectrum(modulus, connection), dtype=complex)
    is_complex = np.iscomplexobj(vector)
    dtype = complex if is_complex else float
    initial = np.asarray(vector, dtype=dtype)
    if initial.ndim != 1 or initial.shape[0] != modulus:
        raise ValueError("vector length must equal modulus")
    if not np.all(np.isfinite(initial)):
        raise ValueError("vector must contain finite values")

    multiplier = np.exp(-capacity * structural_time * spectrum)
    evolved = np.fft.ifft(multiplier * np.fft.fft(initial))
    return evolved if is_complex else evolved.real
