"""Conditional exact enclosures for two pure-EPI transport models on P2.

The nodal pressure is ``p=(x2-x1, x1-x2)``. With common capacity ``nu``, the
mean is fixed and the contrast obeys ``d'=-2*nu*d``. These enclosures do not admit a
physical measurement, certify a live graph or identify an unknown clock.
Calibration uses the continuous contrast ratio, not an Euler increment fit.
The separately named fixed-reference model instead assigns capacities
``(nu, 0)``: ``x'=nu*(r-x)``, ``r'=0`` and ``d'=-nu*d``. Its mean is not
conserved, and a physical reservoir is not automatically an inactive node.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from itertools import islice
from typing import Any, Iterable

from .._exact_time import exact_log_bounds, exact_or_represented_real
from .reversible_eigenmode_reference import _MAX_RATIONAL_EXPONENT, _negative_exp_bounds

__all__ = [
    "P2CapacityEnclosure",
    "P2TransportSample",
    "P2TransportTube",
    "bound_p2_capacity",
    "bound_p2_transport",
    "FixedReferenceCapacityEnclosure",
    "FixedReferenceTransportSample",
    "FixedReferenceTransportTube",
    "bound_fixed_reference_capacity",
    "bound_fixed_reference_transport",
]

Interval = tuple[Fraction, Fraction]
EPIIntervals = tuple[Interval, Interval]
_MAX_SAMPLES = 10_000


def _pair(value: Any, name: str) -> tuple[Any, Any]:
    if isinstance(value, (str, bytes)):
        raise TypeError(f"{name} must contain two entries")
    try:
        entries = tuple(islice(iter(value), 3))
    except TypeError as exc:
        raise TypeError(f"{name} must contain two entries") from exc
    if len(entries) != 2:
        raise ValueError(f"{name} must contain two entries")
    return entries


def _interval(value: Any, name: str) -> Interval:
    low, high = _pair(value, name)
    low = exact_or_represented_real(low, name)
    high = exact_or_represented_real(high, name)
    if low > high:
        raise ValueError(f"{name} is inverted")
    return low, high


def _epi(value: Any, name: str) -> EPIIntervals:
    first, second = _pair(value, name)
    return _interval(first, name), _interval(second, name)


def _mean(epi: EPIIntervals) -> Interval:
    first, second = epi
    return (first[0] + second[0]) / 2, (first[1] + second[1]) / 2


def _contrast(epi: EPIIntervals) -> Interval:
    first, second = epi
    return first[0] - second[1], first[1] - second[0]


def _multiply(left: Interval, right: Interval) -> Interval:
    corners = tuple(a * b for a in left for b in right)
    return min(corners), max(corners)


def _contrast_capacity(
    start: Interval, end: Interval, elapsed: Interval, decay_multiplier: int
) -> Interval:
    """Outer rate for either declared nodal contrast law, not a free model fit."""
    if elapsed[0] <= 0:
        raise ValueError("elapsed_time must be strictly positive")
    if start[0] > 0 and end[0] > 0:
        oriented_start, oriented_end = start, end
    elif start[1] < 0 and end[1] < 0:
        oriented_start = -start[1], -start[0]
        oriented_end = -end[1], -end[0]
    else:
        raise ValueError("contrast_sign_unresolved_or_changed")
    ratio = (oriented_end[0] / oriented_start[1], oriented_end[1] / oriented_start[0])
    if ratio[1] >= 1:
        raise ValueError("strict_decay_unresolved")
    low_log, _ = exact_log_bounds(ratio[0])
    _, high_log = exact_log_bounds(ratio[1])
    capacity = (
        -high_log / (decay_multiplier * elapsed[1]),
        -low_log / (decay_multiplier * elapsed[0]),
    )
    if capacity[0] <= 0 or capacity[1] < capacity[0]:
        raise ValueError("positive_capacity_enclosure_unresolved")
    return capacity


@dataclass(frozen=True)
class P2CapacityEnclosure:
    """Necessary outer capacity interval under the declared continuous model.

    A nonempty interval is not a proof that one common latent mean, capacity
    and sensor/clock realization fit all the input boxes or multiple runs.
    This is a value record, not an authenticated scientific certificate.
    """

    capacity: Interval
    initial_contrast: Interval
    final_contrast: Interval
    mean_overlap: Interval


def bound_p2_capacity(
    initial_epi: Any,
    final_epi: Any,
    *,
    elapsed_time: Any,
) -> P2CapacityEnclosure:
    """Enclose positive capacity from two sign-resolved contrast boxes.

    Inputs are two coordinate intervals per observation and one positive
    structural-time interval. Rational inputs stay exact; other accepted real
    scalars use their finite binary64 representation via the shared boundary.
    No measurement uncertainty is silently inferred from representation.

    For positive oriented contrasts ``A`` then ``B``, ``r=B/A`` and
    ``nu=-log(r)/(2*T)``. Resolved decay requires ``0<r_low<=r_high<1``.
    Opposite/ambiguous signs, unresolved decay and incompatible means raise
    explicit errors rather than clipping a fitted rate into its domain.
    """
    initial = _epi(initial_epi, "initial_epi")
    final = _epi(final_epi, "final_epi")
    elapsed = _interval(elapsed_time, "elapsed_time")
    if elapsed[0] <= 0:
        raise ValueError("elapsed_time must be strictly positive")
    initial_mean, final_mean = _mean(initial), _mean(final)
    overlap = max(initial_mean[0], final_mean[0]), min(initial_mean[1], final_mean[1])
    if overlap[0] > overlap[1]:
        raise ValueError("mean_intervals_disjoint")
    start, end = _contrast(initial), _contrast(final)
    capacity = _contrast_capacity(start, end, elapsed, 2)
    return P2CapacityEnclosure(capacity, start, end, overlap)


@dataclass(frozen=True)
class P2TransportSample:
    """One rational outer observation box; shared parameters remain shared."""

    elapsed_time: Interval
    decay: Interval
    epi: EPIIntervals
    mean: Interval
    contrast: Interval


@dataclass(frozen=True)
class P2TransportTube:
    """Finite conditional tube, not a statistical confidence statement.

    Every model trajectory with inputs in the declared boxes lies in these
    per-time boxes. Membership in all boxes does not prove that one common
    parameter realization generates all observations. Endpoints are rational;
    ordinary float conversion need not preserve outward rounding.
    """

    samples: tuple[P2TransportSample, ...]


def _convex_image(first: Interval, second: Interval, weight: Interval) -> Interval:
    # The two coefficients are nonnegative and sum to one. First take the
    # monotone extrema in the initial coordinates, then both decay endpoints.
    lower = tuple(w * first[0] + (1 - w) * second[0] for w in weight)
    upper = tuple(w * first[1] + (1 - w) * second[1] for w in weight)
    return min(lower), max(upper)


def _decay_samples(
    capacity: Any, elapsed_times: Iterable[Any], decay_multiplier: int
) -> tuple[tuple[Interval, Interval], ...]:
    """Validate one finite schedule and reuse rational decay endpoints."""
    rate = _interval(capacity, "capacity")
    if rate[0] <= 0:
        raise ValueError("capacity must be strictly positive")
    if isinstance(elapsed_times, (str, bytes)):
        raise TypeError("elapsed_times must be a finite sequence of intervals")
    times = tuple(islice(iter(elapsed_times), _MAX_SAMPLES + 1))
    if not times or len(times) > _MAX_SAMPLES:
        raise ValueError("elapsed_times requires between 1 and 10000 samples")
    times = tuple(_interval(value, "elapsed_time") for value in times)
    if any(value[0] < 0 for value in times):
        raise ValueError("elapsed_time must be nonnegative")
    if any(
        left[0] > right[0] or left[1] > right[1]
        for left, right in zip(times, times[1:])
    ):
        raise ValueError("elapsed_time intervals must be ordered by both endpoints")
    if any(
        decay_multiplier * rate[1] * value[1] > _MAX_RATIONAL_EXPONENT
        for value in times
    ):
        raise ValueError("rational exponential enclosure requires exponent <= 4096")
    cache: dict[Fraction, Interval] = {}

    def exponential(exponent: Fraction) -> Interval:
        if exponent not in cache:
            cache[exponent] = _negative_exp_bounds(exponent)
        return cache[exponent]

    samples = []
    for elapsed in times:
        low = exponential(decay_multiplier * rate[1] * elapsed[1])[0]
        high = exponential(decay_multiplier * rate[0] * elapsed[0])[1]
        samples.append((elapsed, (low, high)))
    return tuple(samples)


def bound_p2_transport(
    initial_epi: Any,
    *,
    capacity: Any,
    elapsed_times: Iterable[Any],
) -> P2TransportTube:
    """Enclose the exact continuous P2 semigroup at declared elapsed times.

    ``exp(-2*nu*t)`` is bounded by the existing rational exponential owner.
    Its exponent cap is 4096; at most 10,000 samples are materialized. Time
    intervals must be nonnegative and ordered by both endpoints (overlap is
    allowed). These are computation/domain guards, not physical constants.

    Coordinate extrema use the positive semigroup weights ``(1+e)/2`` and
    ``(1-e)/2`` directly, preserving the mean/contrast dependency that would
    be lost by independently hulling those two variables. All uncertainty
    boxes are caller-supplied hypotheses, not instrument admission evidence.
    """
    initial = _epi(initial_epi, "initial_epi")
    decays = _decay_samples(capacity, elapsed_times, 2)
    mean, contrast = _mean(initial), _contrast(initial)
    samples = []
    for elapsed, decay in decays:
        weight = (1 + decay[0]) / 2, (1 + decay[1]) / 2
        samples.append(
            P2TransportSample(
                elapsed,
                decay,
                (
                    _convex_image(initial[0], initial[1], weight),
                    _convex_image(initial[1], initial[0], weight),
                ),
                mean,
                _multiply(contrast, decay),
            )
        )
    return P2TransportTube(tuple(samples))


@dataclass(frozen=True)
class FixedReferenceCapacityEnclosure:
    """Necessary outer capacity interval for one constant reference value.

    The reference is the same latent value at both observations. Forming
    contrast intervals loses some of this correlation, so a nonempty capacity
    interval does not prove joint feasibility of the input boxes.
    """

    capacity: Interval
    initial_contrast: Interval
    final_contrast: Interval
    reference: Interval


def bound_fixed_reference_capacity(
    initial_epi: Any,
    final_epi: Any,
    *,
    reference: Any,
    elapsed_time: Any,
) -> FixedReferenceCapacityEnclosure:
    """Enclose ``nu=-log((x1-r)/(x0-r))/T`` for fixed-reference nodal P2.

    Capacities are ``(nu,0)`` and only the pure-EPI pressure channel is active.
    There is no conserved arithmetic mean to test. The same-sign and strictly
    resolved decay conditions match the common-capacity contrast estimator,
    but its factor two is absent. No reference level is inferred from outcomes.
    """
    initial = _interval(initial_epi, "initial_epi")
    final = _interval(final_epi, "final_epi")
    fixed = _interval(reference, "reference")
    elapsed = _interval(elapsed_time, "elapsed_time")
    start, end = _contrast((initial, fixed)), _contrast((final, fixed))
    capacity = _contrast_capacity(start, end, elapsed, 1)
    return FixedReferenceCapacityEnclosure(capacity, start, end, fixed)


@dataclass(frozen=True)
class FixedReferenceTransportSample:
    """One outer box for active EPI, fixed reference and their contrast."""

    elapsed_time: Interval
    decay: Interval
    epi: Interval
    reference: Interval
    contrast: Interval


@dataclass(frozen=True)
class FixedReferenceTransportTube:
    """Conditional finite tube with no measured-reservoir or confidence claim.

    One constant reference and one capacity generate each admitted model
    trajectory. Independent membership in all outer boxes is weaker than
    existence of such a common realization. This is an ordinary value record,
    not a sealed certificate of graph execution or physical admission.
    """

    samples: tuple[FixedReferenceTransportSample, ...]


def bound_fixed_reference_transport(
    initial_epi: Any,
    *,
    reference: Any,
    capacity: Any,
    elapsed_times: Iterable[Any],
) -> FixedReferenceTransportTube:
    """Enclose ``x(t)=e*x0+(1-e)*r``, ``e=exp(-nu*t)``, for fixed ``r``.

    The reference capacity is zero, even when its stored pressure ``x-r`` is
    nonzero. It is not replaced by a time-varying measured input or a fitted
    asymptote. Positive semigroup weights preserve coordinate dependencies.
    Rationalization, time ordering, sample and exponential resource bounds
    are shared with :func:`bound_p2_transport`; ordinary float output casts
    do not necessarily round outwards.
    """
    initial = _interval(initial_epi, "initial_epi")
    fixed = _interval(reference, "reference")
    contrast = _contrast((initial, fixed))
    samples = tuple(
        FixedReferenceTransportSample(
            elapsed,
            decay,
            _convex_image(initial, fixed, decay),
            fixed,
            _multiply(contrast, decay),
        )
        for elapsed, decay in _decay_samples(capacity, elapsed_times, 1)
    )
    return FixedReferenceTransportTube(samples)
