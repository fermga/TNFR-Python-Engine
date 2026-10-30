"""Exact Cayley random-walk operators shared by arithmetic studies."""

from __future__ import annotations

import math
from fractions import Fraction
from numbers import Complex, Real

import numpy as np

from .._exact_time import finite_represented_real, nonnegative_represented_time
from ._complex_admission import finite_represented_complex
from ._integer_admission import _integer_argument

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
) -> tuple[int, set[int], Fraction]:
    modulus = _integer_argument(modulus, "modulus")
    if modulus < 1:
        raise ValueError("modulus must be positive")
    normalized = {
        _integer_argument(value, "connection shift") % modulus for value in connection
    } - {0}
    if not normalized:
        raise ValueError("empty connection set")
    return modulus, normalized, Fraction(1, len(normalized))


def cayley_laplacian(modulus: int, connection: set[int]) -> Matrix:
    r"""Return exact ``L_rw = I - (1/d)W`` on ``Z/modulus Z``."""
    modulus, normalized, inverse_degree = _normalized_connection(modulus, connection)
    matrix: Matrix = [[Fraction(0) for _ in range(modulus)] for _ in range(modulus)]
    for source in range(modulus):
        matrix[source][source] = Fraction(1)
        for target in range(modulus):
            if source != target and (target - source) % modulus in normalized:
                matrix[source][target] = -inverse_degree
    return matrix


def cayley_first_row(modulus: int, connection: set[int]) -> list[Fraction]:
    """Return the exact first row of the Cayley Laplacian circulant."""
    modulus, normalized, inverse_degree = _normalized_connection(modulus, connection)
    row = [Fraction(0) for _ in range(modulus)]
    row[0] = Fraction(1)
    for shift in normalized - {0}:
        row[shift] = -inverse_degree
    return row


def cayley_action(
    modulus: int, connection: set[int], vector: list[Fraction]
) -> list[Fraction]:
    """Apply the exact circulant Laplacian without materializing its matrix."""
    modulus, normalized, inverse_degree = _normalized_connection(modulus, connection)
    if len(vector) != modulus:
        raise ValueError("vector length must equal modulus")
    shifts = normalized - {0}
    return [
        vector[source]
        - inverse_degree
        * sum(
            (vector[(source + shift) % modulus] for shift in shifts),
            Fraction(0),
        )
        for source in range(modulus)
    ]


def cayley_spectrum(modulus: int, connection: set[int]) -> list[complex]:
    """Fourier spectrum of the declared circulant, in canonical node order."""
    modulus, shifts, _ = _normalized_connection(modulus, connection)
    spectrum = []
    for mode in range(modulus):
        residues = [(mode * shift) % modulus for shift in shifts]
        if not any(residues):
            # Every support character is one, including disconnected null modes.
            spectrum.append(0j)
            continue
        centered = [r if 2 * r <= modulus else r - modulus for r in residues]
        real = math.fsum(
            2.0 * math.sin(math.pi * r / modulus) ** 2 for r in centered
        ) / len(shifts)
        imaginary = -math.fsum(
            0.0 if 2 * r == modulus else math.sin(math.tau * r / modulus)
            for r in centered
        ) / len(shifts)
        spectrum.append(complex(real, imaginary))
    return spectrum


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

    Real/complex components and nonnegative clocks must be representable as
    binary64 without losing nonzero inputs. Unrepresentable clock products or
    Fourier intermediates raise ``ValueError`` instead of returning nonfinite
    state. Exact support-derived null eigenvalues are retained at every time.
    """
    _, exact_time = nonnegative_represented_time(structural_time, "structural_time")
    _, exact_capacity = nonnegative_represented_time(capacity, "capacity")

    # Validate the cyclic connection through the shared exact constructor.
    spectrum = np.asarray(cayley_spectrum(modulus, connection), dtype=complex)
    raw = tuple(vector)
    if len(raw) != len(spectrum):
        raise ValueError("vector length must equal modulus")
    is_complex = any(
        isinstance(value, Complex) and not isinstance(value, Real) for value in raw
    )
    admitted = [
        finite_represented_complex(value, f"vector[{index}]")
        for index, value in enumerate(raw)
    ]
    initial = np.asarray(admitted, dtype=complex)
    if not is_complex:
        initial = initial.real
    if not exact_time or not exact_capacity:
        return initial.copy()
    elapsed, _ = finite_represented_real(
        exact_capacity * exact_time, "capacity * structural_time"
    )
    with np.errstate(over="ignore", invalid="ignore"):
        exponents = -elapsed * spectrum
        transformed = np.fft.fft(initial)
    if not np.all(np.isfinite(exponents)) or not np.all(np.isfinite(transformed)):
        raise ValueError("diffusion Fourier intermediates must be finite")
    with np.errstate(over="ignore", invalid="ignore"):
        evolved = np.fft.ifft(np.exp(exponents) * transformed)
    if not np.all(np.isfinite(evolved)):
        raise ValueError("diffusion result must be finite")
    return evolved if is_complex else evolved.real
