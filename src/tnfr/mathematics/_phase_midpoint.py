"""Certified two-neighbor circular means without trigonometric evaluation.

Both neighbors must have a certified oriented displacement strictly inside
the center's open half-pi interval. Their phasor sum then has nonzero
resultant and its direction is the midpoint of those lifted displacements.
The midpoint identity uses mathematical pi, not a represented substitute
for the circle's period. Rational Machin enclosures certify branch choices
and final float rounding; undecided cases retain the caller's legacy path.

This is a local numeric kernel. It neither changes a graph or operator
coefficient nor projects pressures onto a prescribed conservation law.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache

from .._binary64 import uses_ieee_binary64_rounding

__all__ = ("CertifiedTwoNeighborPhase", "certified_two_neighbor_phase")

# This is a numerical work limit, not a phase or structural parameter.
_PI_TERMS = 64
_PI_BITS = 256


def _atan_reciprocal_bounds(denominator: int, terms: int) -> tuple[Fraction, Fraction]:
    """Enclose atan(1/q) by its alternating series and first omitted term.

    For q>1 the positive term magnitudes strictly decrease to zero. The
    remainder has the next term's sign and magnitude smaller than that
    term. No floating arctangent or empirical precision estimate is used.
    """
    if type(denominator) is not int or type(terms) is not int:
        raise TypeError("the reciprocal denominator and term count must be integers")
    if denominator <= 1 or terms < 1:
        raise ValueError(
            "the reciprocal denominator must exceed one and terms must be positive"
        )
    partial = sum(
        (
            Fraction((-1) ** index, (2 * index + 1) * denominator ** (2 * index + 1))
            for index in range(terms)
        ),
        Fraction(0),
    )
    next_term = Fraction(
        (-1) ** terms, (2 * terms + 1) * denominator ** (2 * terms + 1)
    )
    return min(partial, partial + next_term), max(partial, partial + next_term)


@lru_cache(maxsize=1)
def _pi_bounds() -> tuple[Fraction, Fraction]:
    """Cache a rational enclosure from pi=16*atan(1/5)-4*atan(1/239).

    The Machin identity follows by rational tangent addition: the tangent
    of 4*atan(1/5)-atan(1/239) equals one, and the angle lies in (0,pi/2).
    Alternating-series remainders give both bounds explicitly. Round the
    lower endpoint down and the upper endpoint up to a 2**-256 dyadic
    grid, keeping subsequent exact arithmetic small. Both the series and
    enclosure precision are numerical work limits; they do not promise
    that every input rounding decision is resolved.
    """
    first_lower, first_upper = _atan_reciprocal_bounds(5, _PI_TERMS)
    second_lower, second_upper = _atan_reciprocal_bounds(239, _PI_TERMS)
    lower = 16 * first_lower - 4 * second_upper
    upper = 16 * first_upper - 4 * second_lower
    scale = 2**_PI_BITS
    lower_index = (lower.numerator * scale) // lower.denominator
    upper_index = -((-upper.numerator * scale) // upper.denominator)
    return Fraction(lower_index, scale), Fraction(upper_index, scale)


def _affine_interval(
    rational: Fraction, coefficient, pi_bounds
) -> tuple[Fraction, Fraction]:
    """Enclose rational+coefficient*pi, retaining an exact rational branch."""
    lower, upper = pi_bounds
    if coefficient >= 0:
        return rational + coefficient * lower, rational + coefficient * upper
    return rational + coefficient * upper, rational + coefficient * lower


def _positive(rational: Fraction, coefficient, pi_bounds) -> bool:
    return _affine_interval(rational, coefficient, pi_bounds)[0] > 0


def _nonnegative(rational: Fraction, coefficient, pi_bounds) -> bool:
    return (rational == 0 and coefficient == 0) or _positive(
        rational, coefficient, pi_bounds
    )


def _oriented_turn(difference: Fraction, pi_bounds) -> int | None:
    """Certify a unique 2pi lift in the center's strict half-pi interval."""
    if 2 * abs(difference) < pi_bounds[0]:
        return 0
    # A positive raw difference can only need a negative full turn, and
    # conversely. The other sign moves farther from the half-pi interval.
    turn = -1 if difference > 0 else 1
    if _positive(difference, 2 * turn + Fraction(1, 2), pi_bounds) and _positive(
        -difference, Fraction(1, 2) - 2 * turn, pi_bounds
    ):
        return turn
    return None


def _canonical_mean_coefficient(
    rational: Fraction, coefficient: int, pi_bounds
) -> int | None:
    """Select the mathematical [0,2pi) branch before any float conversion."""
    # The translated half-open intervals are disjoint, so one certified
    # branch is sufficient. Most normalized inputs need no mean turn.
    for turn in (0, -1, 1):
        if _nonnegative(rational, coefficient - 2 * turn, pi_bounds) and _positive(
            -rational, 2 - coefficient + 2 * turn, pi_bounds
        ):
            return coefficient - 2 * turn
    return None


def _rounded_affine(rational: Fraction, coefficient: int, pi_bounds):
    """Return a uniquely certified RN result and its rational enclosure.

    Fraction-to-float conversion supplies nearest-even rounding of exact
    rational endpoints under the declared Python binary64 premises. By
    monotonicity, identical rounded endpoints certify the enclosed real.
    Exact rational branches, including ties and zero, use one conversion.
    Hex comparison distinguishes the two zero signs for tiny intervals.
    """
    enclosure = _affine_interval(rational, coefficient, pi_bounds)
    if coefficient == 0:
        return float(rational), enclosure
    first, second = (float(value) for value in enclosure)
    if first.hex() != second.hex():
        return None
    return first, enclosure


@dataclass(frozen=True, slots=True)
class CertifiedTwoNeighborPhase:
    """Independent correctly rounded displacement and canonical mean.

    The exact source expressions are rational+pi_coefficient*pi. Their
    rational enclosures remain separate from the rounded outputs. The
    canonical real mean is in [0,2pi), but its nearest float may equal
    math.tau. The displacement must not be reconstructed from that mean.
    """

    delta: float
    mean: float
    method: str
    delta_rational: Fraction
    delta_pi_coefficient: int
    mean_rational: Fraction
    mean_pi_coefficient: int
    delta_enclosure: tuple[Fraction, Fraction]
    mean_enclosure: tuple[Fraction, Fraction]


def certified_two_neighbor_phase(
    center: float,
    first: float,
    second: float,
) -> CertifiedTwoNeighborPhase | None:
    """Certify the true-circle two-neighbor midpoint, or request fallback.

    Inputs must already be finite Python floats in [0,math.tau); no
    normalization, scalar coercion or graph mutation occurs here. Signed
    zero is accepted, with exact zero outputs canonically positive. The
    shared IEEE format/rounding precondition must hold. It does not assert
    accuracy for external trigonometric or custom numerical kernels.

    Treat the represented inputs as exact real angles. Certify separate
    lifts d_j=neighbor_j-center+2*k_j*pi in (-pi/2,pi/2). Their separation
    is less than pi, so exp(i*d_1)+exp(i*d_2) has direction
    delta=(d_1+d_2)/2. Its exact affine expression is rounded once, without
    first rounding a mean and subtracting the center. The canonical mean
    is independently normalized with true 2pi and rounded once as well.

    Strict branch conditions and all irrational rounding decisions must
    follow from the cached analytic pi enclosure. A finite work limit may
    leave either undecided; returning None preserves the caller's existing
    general phasor calculation. No universal fast-path coverage or global
    conservation, equivariance or complete-runtime invariant is inferred.
    """
    if any(
        type(value) is not float
        or not math.isfinite(value)
        or not 0.0 <= value < math.tau
        for value in (center, first, second)
    ):
        return None
    if not uses_ieee_binary64_rounding():
        return None
    exact_center, exact_first, exact_second = (
        Fraction.from_float(value) for value in (center, first, second)
    )
    pi_bounds = _pi_bounds()
    first_turn = _oriented_turn(exact_first - exact_center, pi_bounds)
    second_turn = _oriented_turn(exact_second - exact_center, pi_bounds)
    if first_turn is None or second_turn is None:
        return None
    coefficient = first_turn + second_turn
    mean_rational = (exact_first + exact_second) / 2
    delta_rational = mean_rational - exact_center
    canonical_coefficient = _canonical_mean_coefficient(
        mean_rational, coefficient, pi_bounds
    )
    if canonical_coefficient is None:
        return None
    rounded_delta = _rounded_affine(delta_rational, coefficient, pi_bounds)
    rounded_mean = _rounded_affine(mean_rational, canonical_coefficient, pi_bounds)
    if rounded_delta is None or rounded_mean is None:
        return None
    delta, delta_enclosure = rounded_delta
    mean, mean_enclosure = rounded_mean
    return CertifiedTwoNeighborPhase(
        delta,
        mean,
        "exact_two_neighbor_midpoint",
        delta_rational,
        coefficient,
        mean_rational,
        canonical_coefficient,
        delta_enclosure,
        mean_enclosure,
    )
