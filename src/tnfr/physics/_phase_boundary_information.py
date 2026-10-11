"""Exact static boundary information for a supplied finite Fourier current.

The current convention is ``J(phi) = sum_k j(delta_k - phi)`` with
``j(a) = sum_n b_n sin(n*a)``. The supplied coefficients define a law;
these calculations neither select that law nor evolve a boundary state.
The regular-summary theorem concerns all finite unit-weight gap lists.
Its symbolic regular-grid witness is not a sampled or evaluated response.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from numbers import Integral
from typing import Any

from .._exact_time import exact_or_represented_real
from .phase_response import _admit_relative_phasors
from .relational_observations import _ordered

_MAX_COEFFICIENT_ROWS = 128
_MAX_HARMONIC = 128
_MAX_GAPS = 128

_Phasor = tuple[Fraction, Fraction]
_Coefficients = tuple[tuple[int, Fraction], ...]
_Moments = tuple[tuple[int, _Phasor], ...]


def _admit_fourier_current(coefficients: Any) -> _Coefficients:
    """Admit ordered distinct positive harmonics and exact current scales.

    At most 128 supplied rows and harmonic orders 1..128 are computational
    budgets, not physical restrictions. Duplicate orders reject even when
    their coefficients are zero. Exact rational scales remain exact; other
    reals use shared represented-real admission. Only exact zeros disappear.
    The caller's harmonic order is retained after removing those zeros.
    """
    rows = _ordered(coefficients, "coefficients", limit=_MAX_COEFFICIENT_ROWS + 1)
    if len(rows) > _MAX_COEFFICIENT_ROWS:
        raise ValueError("coefficients exceed the 128-row computational budget")
    admitted: list[tuple[int, Fraction]] = []
    seen: set[int] = set()
    for index, row in enumerate(rows):
        pair = _ordered(row, f"coefficients[{index}]", limit=3)
        if len(pair) != 2:
            raise ValueError("each coefficient row must contain harmonic and scale")
        harmonic, raw_scale = pair
        if isinstance(harmonic, bool) or not isinstance(harmonic, Integral):
            raise TypeError("harmonic must be an integer, not a boolean")
        harmonic = int(harmonic)
        if not 1 <= harmonic <= _MAX_HARMONIC:
            raise ValueError("harmonic must lie in the computational range 1..128")
        if harmonic in seen:
            raise ValueError("coefficient harmonics must be distinct")
        seen.add(harmonic)
        scale = exact_or_represented_real(raw_scale, f"coefficients[{index}].scale")
        if scale != 0:
            admitted.append((harmonic, scale))
    return tuple(admitted)


@dataclass(frozen=True)
class _BoundaryInformation:
    """Symbolic sharp dimension for regular query-independent summaries.

    The dimension is uniform over every finite degree with C1 encoding and
    C1 decoding of every query current. It is not a claim of pointwise rank,
    noise robustness, dynamic closure, physical law selection or the minimum
    for a fixed restricted family. At the q-point circular grid, the Jacobian
    has the displayed signed Fourier-mode eigenvalues. No irrational grid
    phasors or trajectories are materialized. q can exceed the separate
    128-gap budget of the rational static response calculator.
    """

    coefficients: _Coefficients
    minimum_regular_dimension: int
    witness_degree: int
    signed_frequency_eigenvalues: tuple[tuple[int, Fraction], ...]


def _derive_boundary_information(coefficients: Any) -> _BoundaryInformation:
    """Rebuild the Fourier support and exact regular-grid rank witness.

    For q=2*max(n)+1 the derivative matrix j'(delta_k-phi_l) has eigenvalue
    q*n*b_n/2 at both signed modes +/-n, with no aliasing. Each active
    harmonic contributes two nonzero eigenvalues, including negative b_n.
    All remaining modes are zero. For the identically zero law the dimension
    and witness degree are zero, and the eigenvalue list is empty.
    """
    admitted = _admit_fourier_current(coefficients)
    degree = 2 * max((harmonic for harmonic, _ in admitted), default=0) + 1
    if not admitted:
        degree = 0
    eigenvalues = tuple(
        (signed_harmonic, Fraction(degree * harmonic, 2) * scale)
        for harmonic, scale in admitted
        for signed_harmonic in (harmonic, -harmonic)
    )
    return _BoundaryInformation(admitted, 2 * len(admitted), degree, eigenvalues)


def _multiply(left: _Phasor, right: _Phasor) -> _Phasor:
    return (
        left[0] * right[0] - left[1] * right[1],
        left[0] * right[1] + left[1] * right[0],
    )


def _harmonic_power(phasor: _Phasor, harmonic: int) -> _Phasor:
    """Exact complex power of already admitted inputs by repeated squaring."""
    result = (Fraction(1), Fraction(0))
    factor = phasor
    while harmonic:
        if harmonic % 2:
            result = _multiply(result, factor)
        harmonic //= 2
        if harmonic:
            factor = _multiply(factor, factor)
    return result


@dataclass(frozen=True)
class _BoundaryResponse:
    """Exact unnormalized additive current rebuilt from primitive inputs.

    Degree remains available independently of the active harmonic moments.
    A zero moment is retained; a zero law still admits every supplied input.
    This detached readout establishes no autonomous evolution or event law.
    """

    coefficients: _Coefficients
    gap_phasors: tuple[_Phasor, ...]
    query_phasor: _Phasor
    degree: int
    moments: _Moments
    harmonic_contributions: tuple[tuple[int, Fraction], ...]
    current: Fraction


def _evaluate_boundary_response(
    coefficients: Any, *, gap_phasors: Any, query_phasor: Any
) -> _BoundaryResponse:
    """Evaluate sum_n b_n Im(Z_n * exp(-i*n*phi)) using exact moments.

    Coefficients, every gap and the query are freshly admitted. No cached
    report or supplied moment is consumed as evidence. Nonempty gap lists
    reuse the shared 128-neighbor exact unit-circle contract; an empty list
    has degree zero and zero moments. Repeated gaps are distinct incidences.
    Query phasors also obey exact unit admission, with no renormalization.
    The coefficients carry the supplied current scale; no degree division,
    pi factor, pressure conversion or clock interpretation is inferred.
    """
    admitted = _admit_fourier_current(coefficients)
    raw_gaps = _ordered(gap_phasors, "gap_phasors", limit=_MAX_GAPS + 1)
    gaps: tuple[_Phasor, ...] = (
        _admit_relative_phasors(raw_gaps, "gap_phasors") if raw_gaps else ()
    )
    query: _Phasor = _admit_relative_phasors((query_phasor,), "query_phasor")[0]
    moments: list[tuple[int, _Phasor]] = []
    contributions: list[tuple[int, Fraction]] = []
    for harmonic, scale in admitted:
        real, imaginary = Fraction(0), Fraction(0)
        for gap in gaps:
            c, s = _harmonic_power(gap, harmonic)
            real += c
            imaginary += s
        moments.append((harmonic, (real, imaginary)))
        c_query, s_query = _harmonic_power(query, harmonic)
        contributions.append((harmonic, scale * (imaginary * c_query - real * s_query)))
    return _BoundaryResponse(
        admitted,
        gaps,
        query,
        len(gaps),
        tuple(moments),
        tuple(contributions),
        sum((contribution for _, contribution in contributions), Fraction(0)),
    )
