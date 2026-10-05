"""Rational admission margins for relative-phasor phase domains.

Angles are exact Fractions of materialized real phase coordinates. Admission
uses mathematical trigonometric functions, not libm estimates or a represented
substitute for pi. Bounds may be inconclusive; a nonpositive lower margin is
not proof that the actual resultant is nonpositive or singular. Full-regular
admission also encloses the imaginary part and separates the resultant from
the nonpositive real axis; the older right-half-plane contract is unchanged.

The Machin pi enclosure has one existing owner. Only a short cosine Taylor
enclosure and its exact Lipschitz range allowance are added here. Work limits
and outward dyadic rounding are numerical policies, not dynamical thresholds.
"""

from __future__ import annotations

from collections.abc import Mapping, Set
from fractions import Fraction as Q
from functools import lru_cache

from ._phase_midpoint import _pi_bounds

__all__ = (
    "COSINE_ENCLOSURE_METHOD",
    "REGULAR_RESULTANT_ENCLOSURE_METHOD",
    "certified_cosine_bounds",
    "certified_cosine_lower_bound",
    "certified_sine_bounds",
    "relative_resultant_bounds",
    "relative_resultant_rate_bounds",
    "relative_resultant_lower_bounds",
    "relative_resultant_regular_margins",
    "regular_resultant_margin",
)

COSINE_ENCLOSURE_METHOD = "rational_machin_pi_reduction_alternating_cosine_lipschitz_v1"
REGULAR_RESULTANT_ENCLOSURE_METHOD = (
    "rational_machin_sine_cosine_rectangle_cut_distance_lower_bound_v1"
)
_COSINE_TERMS = 12
_OUTPUT_BITS = 64
_MAX_INPUT_BITS = 4096


def _outward(lower: Q, upper: Q, bits: int = _OUTPUT_BITS) -> tuple[Q, Q]:
    """Widen to a fixed dyadic grid after all exact enclosure arithmetic."""
    scale = 1 << bits
    lower = max(Q(-1), lower)
    upper = min(Q(1), upper)
    left = (lower.numerator * scale) // lower.denominator
    right = -((-upper.numerator * scale) // upper.denominator)
    return Q(left, scale), Q(right, scale)


@lru_cache(maxsize=4096)
def _nonnegative_cosine_bounds(
    angle: Q, terms: int = _COSINE_TERMS, bits: int = _OUTPUT_BITS
) -> tuple[Q, Q]:
    if not angle:
        return Q(1), Q(1)
    if (
        max(angle.numerator.bit_length(), angle.denominator.bit_length())
        > _MAX_INPUT_BITS
    ):
        return Q(-1), Q(1)

    # Near zero this global lower bound and the alternating quartic upper
    # bound avoid high powers of subnormal binary64 denominators.
    if angle <= Q(1, 16) and terms == _COSINE_TERMS and bits == _OUTPUT_BITS:
        square = angle * angle
        lower = 1 - square / 2
        return _outward(lower, lower + square * square / 24, bits)

    pi_lower, pi_upper = _pi_bounds()
    pi_midpoint = (pi_lower + pi_upper) / 2
    # Any integer turn preserves cosine. No principal-angle branch choice,
    # floating quotient or antipodal tolerance is required.
    turn = (angle + pi_midpoint) // (2 * pi_midpoint)
    midpoint = angle - 2 * turn * pi_midpoint
    radius = turn * (pi_upper - pi_lower)
    if radius >= 2:
        return Q(-1), Q(1)
    if abs(midpoint) > 4:
        # The shared pi enclosure puts pi_midpoint below four. Retain an
        # explicit fail-closed work/domain boundary if that owner changes.
        return Q(-1), Q(1)

    # With |midpoint|<=4 and terms>=2, the omitted tail has strictly
    # decreasing magnitudes. Its sign is that of the first omitted term,
    # whose magnitude bounds it. Defaults retain the degree22 enclosure.
    square = midpoint * midpoint
    term = partial = Q(1)
    for index in range(1, terms):
        term *= -square / ((2 * index - 1) * (2 * index))
        partial += term
    next_term = -term * square / ((2 * terms - 1) * (2 * terms))
    # True reduced angle lies within radius of the rational midpoint, and
    # |cos(a)-cos(b)|<=|a-b|. This remains valid across an antipodal edge.
    return _outward(
        min(partial, partial + next_term) - radius,
        max(partial, partial + next_term) + radius,
        bits,
    )


def _admit_trigonometric_policy(angle, terms, bits):
    """Preserve shared exact-domain and bounded-work admission."""
    if not isinstance(angle, Q):
        raise TypeError("certified cosine requires an exact Fraction angle")
    if type(terms) is not int or type(bits) is not int:
        raise TypeError("cosine term count and precision must be integers")
    if not 2 <= terms <= 64 or not 1 <= bits <= 256:
        raise ValueError("cosine bounds require 2..64 terms and 1..256 bits")


def certified_cosine_bounds(
    angle: Q, *, terms: int = _COSINE_TERMS, bits: int = _OUTPUT_BITS
) -> tuple[Q, Q]:
    """Enclose cos(angle), with conservative bounded numerical work.

    The input must already be an exact rational angle; coercing an arbitrary
    object or silently rounding an input is outside this kernel. Bounds use
    mathematical pi. Returning [-1,1] is an unresolved work/precision outcome,
    not an assertion about the angle's actual cosine. Optional ``terms``
    (2..64) and dyadic ``bits`` (1..256) select bounded numerical work.
    Defaults retain the historical small-angle shortcut and output; custom
    precision evaluates the full alternating series even near zero.
    """
    _admit_trigonometric_policy(angle, terms, bits)
    return _nonnegative_cosine_bounds(abs(angle), terms, bits)


def certified_cosine_lower_bound(angle: Q) -> Q:
    """Return the exact rational lower endpoint of the shared enclosure."""
    return certified_cosine_bounds(angle)[0]


def certified_sine_bounds(
    angle: Q, *, terms: int = _COSINE_TERMS, bits: int = _OUTPUT_BITS
) -> tuple[Q, Q]:
    """Enclose sin(angle) using the shared cosine and mathematical pi bounds.

    The cosine of a rational approximation to ``pi/2-angle`` is enlarged by
    the exact pi enclosure radius, using its unit Lipschitz constant. Exact
    oddness and zero are preserved; no represented substitute for pi or
    floating-point sine determines admission. Work limits match cosine.
    """
    # Share domain and numerical-policy admission even for the exact zero.
    _admit_trigonometric_policy(angle, terms, bits)
    if not angle:
        return Q(0), Q(0)
    if (
        max(angle.numerator.bit_length(), angle.denominator.bit_length())
        > _MAX_INPUT_BITS
    ):
        return Q(-1), Q(1)
    pi_lower, pi_upper = _pi_bounds()
    half_pi_midpoint = (pi_lower + pi_upper) / 4
    radius = (pi_upper - pi_lower) / 4
    lower, upper = certified_cosine_bounds(
        half_pi_midpoint - abs(angle), terms=terms, bits=bits
    )
    result = _outward(lower - radius, upper + radius, bits)
    return result if angle > 0 else (-result[1], -result[0])


def _ordered(value, label):
    if isinstance(value, (str, bytes, bytearray, Mapping, Set)):
        raise TypeError(f"{label} must be an ordered iterable")
    try:
        return tuple(value)
    except TypeError as exc:
        raise TypeError(f"{label} must be an ordered iterable") from exc


def _phase_support(phases, neighbors):
    """Share exact coordinate and ordered unique-support admission."""
    phases = _ordered(phases, "phases")
    if not phases or any(not isinstance(value, Q) for value in phases):
        raise TypeError("phases must be a nonempty ordered sequence of Fractions")
    rows = _ordered(neighbors, "neighbors")
    if len(rows) != len(phases):
        raise ValueError("neighbor rows must match the phase coordinates")
    result = []
    for i, supplied in enumerate(rows):
        row = _ordered(supplied, "neighbor row")
        if any(type(j) is not int or not 0 <= j < len(phases) or j == i for j in row):
            raise ValueError("neighbors must be distinct other-node integer indices")
        if len(set(row)) != len(row):
            raise ValueError("neighbor rows must not repeat an index")
        result.append(row)
    return phases, tuple(result)


def relative_resultant_lower_bounds(phases, neighbors) -> tuple[Q, ...]:
    """Enclose each Re(sum_j exp(i*(phase_j-phase_i))) from below.

    Phases are exact represented Fractions and neighbors are ordered unique
    integer indices, with no self loops. Topological symmetry and engine
    graph admission belong to the caller. Strictly positive returned margins
    certify the right-half-plane chamber. A caller can certify the full
    linear represented proposal segment by subtracting, in each row,
    ``sum_j abs(increment_j-increment_i)`` and requiring strict positivity.
    That follows from the unit Lipschitz constant of cosine; it neither
    certifies an exact ODE trajectory nor selects a time step.
    """
    phases, rows = _phase_support(phases, neighbors)
    return tuple(
        sum(
            (certified_cosine_lower_bound(phases[j] - phases[i]) for j in row),
            Q(0),
        )
        for i, row in enumerate(rows)
    )


def relative_resultant_bounds(phases, neighbors):
    """Enclose each relative resultant by exact rational real/imaginary bounds.

    Return one ``((real_lower, real_upper), (imag_lower, imag_upper))`` per
    node. Unresolved trig ranges are retained rather than coerced into a
    branch. Empty neighbor rows give a zero resultant; graph admission and
    any required nonzero resultant remain the caller's responsibility.
    """
    phases, rows = _phase_support(phases, neighbors)
    result = []
    for i, row in enumerate(rows):
        real_lower = real_upper = imag_lower = imag_upper = Q(0)
        for j in row:
            angle = phases[j] - phases[i]
            lower, upper = certified_cosine_bounds(angle)
            real_lower += lower
            real_upper += upper
            lower, upper = certified_sine_bounds(angle)
            imag_lower += lower
            imag_upper += upper
        result.append(((real_lower, real_upper), (imag_lower, imag_upper)))
    return tuple(result)


def relative_resultant_rate_bounds(phases, neighbors, phase_rates):
    """Enclose the instantaneous derivative of every relative resultant.

    Phases and supplied angular rates are exact Fractions in matching node
    order. For ``z_i=sum_j exp(i*(phase_j-phase_i))``, this evaluates the
    chain rule ``z_i'=sum_j i*exp(i*delta_ij)*(rate_j-rate_i)`` using the
    shared mathematical sine/cosine enclosures. Each returned row is
    ``((real_rate_lower, real_rate_upper), (imag_rate_lower, imag_rate_upper))``.

    This kinematic statement neither admits an evolution law nor certifies
    supplied rates as ideal ODE rates. Materialized engine rates retain that
    provenance. It is instantaneous, not a future enclosure or an existence
    certificate. No resultant division occurs: even a zero resultant admits
    this derivative when phases and their instantaneous rates are supplied.
    Empty neighbor rows and common angular rates give exact zero. Arithmetic
    on rate-weighted bounds remains rational without trigonometric clipping.
    """
    phases, rows = _phase_support(phases, neighbors)
    rates = _ordered(phase_rates, "phase rates")
    if len(rates) != len(phases):
        raise ValueError("phase rates must match the phase coordinates")
    if any(not isinstance(value, Q) for value in rates):
        raise TypeError("phase rates must be an ordered sequence of Fractions")
    result = []
    for i, row in enumerate(rows):
        real_lower = real_upper = imag_lower = imag_upper = Q(0)
        for j in row:
            rate = rates[j] - rates[i]
            if not rate:
                continue
            angle = phases[j] - phases[i]
            lower, upper = certified_sine_bounds(angle)
            real_lower += min(-rate * lower, -rate * upper)
            real_upper += max(-rate * lower, -rate * upper)
            lower, upper = certified_cosine_bounds(angle)
            imag_lower += min(rate * lower, rate * upper)
            imag_upper += max(rate * lower, rate * upper)
        result.append(((real_lower, real_upper), (imag_lower, imag_upper)))
    return tuple(result)


def relative_resultant_regular_margins(phases, neighbors) -> tuple[Q, ...]:
    """Lower-bound distance from each resultant to the nonpositive real ray.

    A rectangle gives ``max(0, real_lower, imag_lower, -imag_upper)`` as a
    conservative Euclidean distance lower bound. Strict positivity admits
    the full regular principal-Arg domain: positive real resultants are
    allowed, while zero and the negative real branch cut are excluded.
    A zero margin is unresolved/excluded evidence, not proof of a singularity.

    For a straight represented phase proposal, subtract in each row
    ``sum_j abs(increment_j-increment_i)``. Strictly positive remainders
    certify every point of that phase chord by the complex exponential's
    unit Lipschitz bound and the distance function's unit Lipschitz bound.
    This sufficient test may reject other regular chords and does not
    enclose an exact ODE trajectory or choose a step size.
    """
    return tuple(
        regular_resultant_margin(real, imag)
        for real, imag in relative_resultant_bounds(phases, neighbors)
    )


def regular_resultant_margin(real_bounds, imag_bounds) -> Q:
    """Return a conservative cut-distance margin from an exact rectangle.

    Each component requires two ordered finite rational endpoints. This
    kernel consumes bounds supplied by the caller, not an unverified claim
    that any measured or computed resultant is actually in the rectangle.
    """
    real = _ordered(real_bounds, "real bounds")
    imag = _ordered(imag_bounds, "imaginary bounds")
    for bounds in (real, imag):
        if len(bounds) != 2 or any(not isinstance(value, Q) for value in bounds):
            raise TypeError("resultant bounds require two exact Fraction endpoints")
        if bounds[0] > bounds[1]:
            raise ValueError("resultant lower endpoint exceeds upper endpoint")
    return max(Q(0), real[0], imag[0], -imag[1])
