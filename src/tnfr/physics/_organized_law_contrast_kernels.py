"""Fixed analytical moment enclosures for the declared long-ramp contrasts.

The no-argument assessment evaluates only twenty scalar moment integrals.
It observes no record, selects no law and supplies no nonlinear-history error.
The exclusive benchmark archives its committed declaration before execution.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q
from functools import lru_cache
from math import comb, factorial

from ..mathematics._rational_interval import INTERVAL_BITS, I, cos, sin, sqrt

_TERMS = 20
_HARMONICS = (1, 2, 9, 18, 36)
_RADIUS_BUDGET = Q(1, 10**28)
_SELECTED_MODE = 2


@dataclass(frozen=True)
class _MomentKernel:
    mode: int
    harmonic: int
    mu: I
    sine_hyperbolic: I
    cosine_hyperbolic: I
    sine_truncation: Q
    cosine_truncation: Q


@dataclass(frozen=True)
class _RampContrast:
    construction: str
    mode: int | None
    alpha_even: I
    alpha_odd: I
    even_coefficients: tuple[I, I]
    odd_coefficients: tuple[I, I]
    even_noise_multiplier: I
    odd_noise_multiplier: I


@dataclass(frozen=True)
class _OrganizedContrastKernels:
    terms: int
    harmonics: tuple[int, ...]
    interval_bits: int
    kernel_radius_budget: Q
    selected_mode: int
    kernels: tuple[_MomentKernel, ...]
    across_mode_contrast: _RampContrast | None
    three_ramp_contrasts: tuple[_RampContrast, ...]
    maximum_kernel_radius: Q
    admitted: bool
    reason: str
    scope: str = "moment_kernel_arithmetic_only_not_a_record_verdict"


def _ramp_monomial_moment(power: int, moment: int) -> Q:
    """Exactly integrate r**power * xi(r)**moment on [0, 1/2]."""
    if type(power) is not int or type(moment) is not int:
        raise TypeError("moment powers require ordinary integers")
    if not 0 <= power <= 39 or not 0 <= moment <= 39:
        raise ValueError("moment powers must lie between zero and 39")
    return _exact_ramp_monomial_moment(power, moment)


@lru_cache(maxsize=1600)
def _exact_ramp_monomial_moment(power: int, moment: int) -> Q:
    return sum(
        (
            Q(comb(moment, index))
            * Q(3, 20) ** (moment - index)
            * Q(-1, 5) ** index
            * Q(1, 2) ** (power + moment + 2 * index + 1)
            / (power + moment + 2 * index + 1)
            for index in range(moment + 1)
        ),
        Q(0),
    )


def _series_tails() -> tuple[Q, Q, Q, Q]:
    """Global fixed truncations for |mu*r|<=1/2 and |n*xi|<=9/5."""
    sinh_tail = Q(1, 2) ** 41 / factorial(41) / (1 - Q(1, 4 * 42 * 43))
    cosh_tail = Q(1, 2) ** 40 / factorial(40) / (1 - Q(1, 4 * 41 * 42))
    sine_tail = Q(9, 5) ** 41 / factorial(41)
    cosine_tail = Q(9, 5) ** 40 / factorial(40)
    return sinh_tail, cosh_tail, sine_tail, cosine_tail


def _enclose_moment(mode: int, mu: I, harmonic: int) -> _MomentKernel:
    """Integrate fixed finite products exactly, then add uniform product tails."""
    if mode not in (1, 2) or type(mode) is not int:
        raise ValueError("moment mode must be one or two")
    if type(harmonic) is not int or harmonic not in _HARMONICS:
        raise ValueError("harmonic is outside the fixed declaration")
    mu = I(mu.lo, mu.hi)
    if not 0 <= mu.lo <= mu.hi <= 1:
        raise ValueError("spectral parameter must lie in [0, 1]")
    sinh_tail, cosh_tail, sine_tail, cosine_tail = _series_tails()
    truncations = ((cosh_tail + 2 * cosine_tail) / 2, (sinh_tail + 2 * sine_tail) / 2)
    moments = []
    for parity in (0, 1):
        value = I(0)
        for left in range(_TERMS):
            power = 2 * left + parity
            # The coefficient is rational before interval multiplication;
            # large n**k and cancelling polynomial terms are never rounded.
            coefficient = sum(
                (
                    Q((-1) ** right * harmonic ** (2 * right + parity))
                    * _ramp_monomial_moment(power, 2 * right + parity)
                    / (factorial(power) * factorial(2 * right + parity))
                    for right in range(_TERMS)
                ),
                Q(0),
            )
            value = value + coefficient * mu**power
        tail = truncations[parity]
        moments.append(value + I(-tail, tail))
    return _MomentKernel(
        mode=mode,
        harmonic=harmonic,
        mu=mu,
        sine_hyperbolic=moments[1],
        cosine_hyperbolic=moments[0],
        sine_truncation=truncations[1],
        cosine_truncation=truncations[0],
    )


def _cosh_half(mu: I) -> I:
    """Enclose cosh(mu/2) using the declared fixed positive series."""
    value = sum(
        ((mu / 2) ** (2 * index) / factorial(2 * index) for index in range(_TERMS)),
        I(0),
    )
    return value + I(0, _series_tails()[1])


def _three_ramp_contrast(
    mode: int, mu: I, moments: dict[int, _MomentKernel]
) -> _RampContrast | None:
    first = moments[1]
    fundamental_even = (
        cos(Q(1, 20)) * first.cosine_hyperbolic - moments[2].cosine_hyperbolic
    )
    fundamental_odd = cos(Q(1, 20)) * first.sine_hyperbolic + moments[2].sine_hyperbolic
    if fundamental_even.contains(0) or fundamental_odd.contains(0):
        return None
    alpha_even = sin(Q(1, 20)) * first.sine_hyperbolic / fundamental_even
    alpha_odd = sin(Q(1, 20)) * first.cosine_hyperbolic / fundamental_odd
    even, odd = [], []
    for harmonic in (9, 18):
        item, doubled = moments[harmonic], moments[2 * harmonic]
        sine, cosine = sin(Q(harmonic, 20)), cos(Q(harmonic, 20))
        even.append(
            sine * item.sine_hyperbolic
            - alpha_even * (cosine * item.cosine_hyperbolic - doubled.cosine_hyperbolic)
        )
        odd.append(
            sine * item.cosine_hyperbolic
            - alpha_odd * (cosine * item.sine_hyperbolic + doubled.sine_hyperbolic)
        )
    noise = sqrt(10) * _cosh_half(mu)
    return _RampContrast(
        construction="same_mode_three_ramp_fundamental_elimination",
        mode=mode,
        alpha_even=alpha_even,
        alpha_odd=alpha_odd,
        even_coefficients=(even[0], even[1]),
        odd_coefficients=(odd[0], odd[1]),
        even_noise_multiplier=noise * (1 + 2 * abs(alpha_even)),
        odd_noise_multiplier=noise * (1 + 2 * abs(alpha_odd)),
    )


def _across_mode_contrast(
    first: dict[int, _MomentKernel], second: dict[int, _MomentKernel]
) -> _RampContrast | None:
    if second[1].sine_hyperbolic.contains(0) or second[1].cosine_hyperbolic.contains(0):
        return None
    alpha_even = first[1].sine_hyperbolic / second[1].sine_hyperbolic
    alpha_odd = first[1].cosine_hyperbolic / second[1].cosine_hyperbolic
    even, odd = [], []
    for harmonic in (9, 18):
        factor = sin(Q(harmonic, 20))
        even.append(
            factor
            * (
                first[harmonic].sine_hyperbolic
                - alpha_even * second[harmonic].sine_hyperbolic
            )
        )
        odd.append(
            factor
            * (
                first[harmonic].cosine_hyperbolic
                - alpha_odd * second[harmonic].cosine_hyperbolic
            )
        )
    noise_first = sqrt(10) * _cosh_half(first[1].mu)
    noise_second = sqrt(10) * _cosh_half(second[1].mu)
    return _RampContrast(
        construction="two_mode_first_minus_third_fundamental_elimination",
        mode=None,
        alpha_even=alpha_even,
        alpha_odd=alpha_odd,
        even_coefficients=(even[0], even[1]),
        odd_coefficients=(odd[0], odd[1]),
        even_noise_multiplier=noise_first + abs(alpha_even) * noise_second,
        odd_noise_multiplier=noise_first + abs(alpha_odd) * noise_second,
    )


def _assess_organized_law_contrast_kernels() -> _OrganizedContrastKernels:
    """Evaluate the single declared kernel assessment, with no response data."""
    modes = ((5 - sqrt(5)) / 8, (5 + sqrt(5)) / 8)
    kernels = tuple(
        _enclose_moment(mode, mu, harmonic)
        for mode, mu in enumerate(modes, 1)
        for harmonic in _HARMONICS
    )
    by_mode = tuple(
        {item.harmonic: item for item in kernels if item.mode == mode}
        for mode in (1, 2)
    )
    maximum = max(
        value.radius
        for item in kernels
        for value in (item.sine_hyperbolic, item.cosine_hyperbolic)
    )
    across = _across_mode_contrast(*by_mode)
    three = tuple(
        result
        for mode, (mu, moments) in enumerate(zip(modes, by_mode), 1)
        if (result := _three_ramp_contrast(mode, mu, moments)) is not None
    )
    admitted = maximum <= _RADIUS_BUDGET and across is not None and len(three) == 2
    reason = (
        "fixed_kernel_arithmetic_admitted"
        if admitted
        else "fixed_kernel_radius_or_denominator_unresolved"
    )
    return _OrganizedContrastKernels(
        terms=_TERMS,
        harmonics=_HARMONICS,
        interval_bits=INTERVAL_BITS,
        kernel_radius_budget=_RADIUS_BUDGET,
        selected_mode=_SELECTED_MODE,
        kernels=kernels,
        across_mode_contrast=across,
        three_ramp_contrasts=three,
        maximum_kernel_radius=maximum,
        admitted=admitted,
        reason=reason,
    )
