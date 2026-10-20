"""Conditional binary64 arithmetic bounds for the timestamped EPI reader.

This pure calculation reads no samples. Whole-window form, rate and acceleration
bounds and independently justified errors of the already materialized samples
are supplied premises. Exact reader arithmetic does not certify acquisition,
the producer, the live endpoint or the association with a structural clock.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q
from math import isfinite
from sys import float_info

from ..mathematics._rational_interval import I
from .relational_observations import _sample_jet_clock, _sample_triple

_MIN_NORMAL = Q(1, 2**1022)
_MAX_FINITE = Q.from_float(float_info.max)


@dataclass(frozen=True)
class Binary64AccelerationReaderBudget:
    """Prospective value error and a separate sufficient arithmetic decision.

    ``arithmetic_error_bound == 0`` is conditional on binary64 sample values
    obeying the supplied errors relative to one C2 curve with all three bounds
    valid over the same sample window. ``None`` explicitly abstains outside
    this narrow positive-form/positive-secant exact-arithmetic domain.
    ``represented_acceleration_bounds`` encloses the exact stencil of the
    represented values; it encloses the runtime result only when ``exact``.
    """

    sample_times: tuple[Q, Q, Q]
    sample_step: Q
    sample_error_bounds: tuple[Q, Q, Q]
    acceleration_sample_error_bound: Q
    represented_form_bounds: tuple[I, I, I]
    represented_secant_bounds: tuple[I, I]
    represented_acceleration_bounds: I
    arithmetic_error_bound: Q | None
    exact: bool
    abstention_reason: str | None


def _binary64_exact(value):
    try:
        materialized = float(value)
    except (OverflowError, ValueError):
        return False
    return isfinite(materialized) and Q.from_float(materialized) == value


def _normal_or_zero_scalar(value):
    return value == 0 or _MIN_NORMAL <= abs(value) <= _MAX_FINITE


def _normal_or_zero_interval(value):
    if value.lo == value.hi == 0:
        return True
    return (
        not value.contains(0)
        and min(abs(value.lo), abs(value.hi)) >= _MIN_NORMAL
        and value.abs_max <= _MAX_FINITE
    )


def _positive_sterbenz_pair(left, right):
    lower = min(left.lo, right.lo)
    return lower > 0 and max(left.hi, right.hi) <= 2 * lower


def bound_binary64_acceleration_reader(
    *,
    form_bounds,
    rate_bounds,
    acceleration_bounds,
    sample_step,
    observation_time,
    sample_error_bounds,
):
    """Bound the actual three-sample reader without generating observations.

    The reader computes two form differences, divides each by its time gap,
    subtracts the secants, multiplies by two and divides by the total span.
    Sterbenz subtraction and finite normal power-of-two scaling make every
    operation exact under the sufficient conditions checked here. Exact zero
    intermediates remain zero; an interval containing arbitrarily small
    nonzero values never substitutes a zero or a normal-range certificate.
    The clock owner constructs ``t0, t0+h, t0+2*h`` in exact arithmetic. Both
    timestamps and ``h, 2*h`` must be binary64-exact, so timestamp differences
    and their sum also have exactly representable results.

    The true three-point stencil is a positive average of the supplied
    whole-window second derivative. Its sample-value error is therefore
    ``(e0+2*e1+e2)/h**2``; no point-derivative truncation is involved. Timestamp
    conversion must be exact. Unknown jitter or an unknown clock association
    is not covered. Valid unsupported regimes return explicit abstention;
    invalid real domains and malformed sample-error triples reject.
    """
    form, rate, acceleration = map(
        I.coerce, (form_bounds, rate_bounds, acceleration_bounds)
    )
    step, times, _, _ = _sample_jet_clock(
        sample_step=sample_step, observation_time=observation_time
    )
    errors = _sample_triple(
        sample_error_bounds, "sample_error_bounds", nonnegative=True
    )
    forms = tuple(I(form.lo - error, form.hi + error) for error in errors)
    secant_errors = tuple((errors[i] + errors[i + 1]) / step for i in range(2))
    secants = tuple(I(rate.lo - error, rate.hi + error) for error in secant_errors)
    sample_error = (errors[0] + 2 * errors[1] + errors[2]) / step**2
    stencil = I(acceleration.lo - sample_error, acceleration.hi + sample_error)
    # Multiply exact endpoints before interval materialization. Very small
    # valid clocks must abstain, not fail through a rounded-zero denominator.
    differences = tuple(I(value.lo * step, value.hi * step) for value in secants)
    slope_difference = I(stencil.lo * step, stencil.hi * step)
    numerator = I(2 * slope_difference.lo, 2 * slope_difference.hi)
    span = 2 * step
    reasons = []
    if (
        float_info.radix,
        float_info.mant_dig,
        float_info.min_exp,
        float_info.max_exp,
    ) != (2, 53, -1021, 1024):
        reasons.append("runtime_is_not_binary64")
    if not all(_binary64_exact(value) for value in (*times, step, span)):
        reasons.append("clock_not_exactly_representable_in_binary64")
    if not all(_normal_or_zero_scalar(value) for value in (*times, step, span)):
        reasons.append("clock_outside_normal_finite_range")
    if (step.numerator & (step.numerator - 1)) or (
        step.denominator & (step.denominator - 1)
    ):
        reasons.append("sample_step_is_not_a_power_of_two")
    if not all(_normal_or_zero_interval(value) for value in forms):
        reasons.append("represented_forms_outside_normal_finite_range")
    if not all(_positive_sterbenz_pair(forms[i], forms[i + 1]) for i in range(2)):
        reasons.append("form_subtractions_lack_positive_sterbenz_bounds")
    if not all(_normal_or_zero_interval(value) for value in differences + secants):
        reasons.append("secant_scaling_outside_normal_finite_range")
    if not _positive_sterbenz_pair(*secants):
        reasons.append("secant_subtraction_lacks_positive_sterbenz_bounds")
    if not all(
        _normal_or_zero_interval(value)
        for value in (slope_difference, numerator, stencil)
    ):
        reasons.append("acceleration_scaling_not_separated_from_nonnormal_values")
    exact = not reasons
    return Binary64AccelerationReaderBudget(
        sample_times=times,
        sample_step=step,
        sample_error_bounds=errors,
        acceleration_sample_error_bound=sample_error,
        represented_form_bounds=forms,
        represented_secant_bounds=secants,
        represented_acceleration_bounds=stencil,
        arithmetic_error_bound=Q(0) if exact else None,
        exact=exact,
        abstention_reason="; ".join(reasons) if reasons else None,
    )
