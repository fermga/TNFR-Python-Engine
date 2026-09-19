r"""Exact finite-history reference for the derived P5 EPI observation.

The fixed unit-weight path and partition ``((0, 4), (1, 2, 3))`` are the
analytical fixture in ``theory/DERIVED_EPI_MEMORY.md``. Its common capacity
rescales time. Truncating the derived memory convolution retains the initial
hidden-state source; the window is an approximation budget, not a new law.

Finite method-of-steps formulas and rational exponential enclosures separate
model truncation error from evaluation uncertainty. No timestep solver, graph
mutation, runtime binding or REMESH identification is performed.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import comb, factorial

from .p5_reduction import _rational, _vector, reduce_p5_state
from .reversible_eigenmode_reference import _negative_exp_bounds

__all__ = [
    "P5MemoryTruncationSample",
    "P5MemoryTruncationReference",
    "bound_p5_memory_truncation",
]

Interval = tuple[Fraction, Fraction]
MacroIntervals = tuple[Interval, Interval]
_ZERO = (Fraction(0), Fraction(0))
_MAX_DELAYED_CORRECTIONS = 32
# Every required exponential has exponent at most twice scaled time. The
# existing rational exponential helper admits exponents up to 4096.
_MAX_SCALED_TIME = Fraction(2048)


@dataclass(frozen=True)
class P5MemoryTruncationSample:
    """Exact rational enclosures, with a distinct model-error upper bound.

    ``contrast_error`` encloses truncated minus full contrast, whereas
    ``dropped_reference_forcing`` encloses the omitted convolution evaluated
    on the full reference. Their units and meanings differ. ``evaluation_width``
    is the largest width of the truncated macro intervals; it excludes model
    error and any later binary64 presentation rounding.
    """

    time: Fraction
    reference_contrast: Interval
    truncated_contrast: Interval
    contrast_error: Interval
    reference_macro: MacroIntervals
    truncated_macro: MacroIntervals
    initial_source_contrast: Interval
    dropped_reference_forcing: Interval
    dropped_reference_forcing_bound: Fraction
    macro_error_bound: Fraction
    macro_error_enclosure: Interval
    evaluation_width: Fraction
    delayed_corrections: int
    coincides_before_cutoff: bool


@dataclass(frozen=True)
class P5MemoryTruncationReference:
    """Reference for an explicitly fixed P5 model, not a live-graph certificate."""

    initial_epi: tuple[Fraction, ...]
    capacity: Fraction
    memory_window: Fraction
    initial_macro: tuple[Fraction, Fraction]
    initial_hidden_contrast: Fraction
    conserved_mean: Fraction
    contrast_coefficients: tuple[Fraction, Fraction]
    max_reference_contrast: Fraction
    samples: tuple[P5MemoryTruncationSample, ...]
    scope: str


def _add(left: Interval, right: Interval) -> Interval:
    return left[0] + right[0], left[1] + right[1]


def _scale(value: Fraction, interval: Interval) -> Interval:
    ends = value * interval[0], value * interval[1]
    return min(ends), max(ends)


def _absolute(interval: Interval) -> Interval:
    low, high = interval
    return (
        Fraction(0) if low <= 0 <= high else min(abs(low), abs(high)),
        max(abs(low), abs(high)),
    )


def _macro(mean: Fraction, contrast: Interval) -> MacroIntervals:
    center = (mean, mean)
    return (
        _add(center, _scale(Fraction(3, 4), contrast)),
        _add(center, _scale(Fraction(-1, 4), contrast)),
    )


def _partial_fraction_polynomials(
    p: int, q: int, time: Fraction
) -> tuple[Fraction, Fraction]:
    """Inverse Laplace of (s+1)^(-p)(s+2)^(-q), sans exponentials."""
    first = sum(
        (
            (-1) ** (p - k)
            * comb(p + q - k - 1, p - k)
            * time ** (k - 1)
            / factorial(k - 1)
            for k in range(1, p + 1)
        ),
        Fraction(0),
    )
    second = sum(
        (
            (-1) ** p * comb(p + q - k - 1, q - k) * time ** (k - 1) / factorial(k - 1)
            for k in range(1, q + 1)
        ),
        Fraction(0),
    )
    return first, second


def _correction_count(time: Fraction, window: Fraction) -> int:
    if window == 0 or time <= window:
        return 0
    count, remainder = divmod(time, window)
    # The newly starting delayed term vanishes at an exact window boundary.
    return int(count) - (remainder == 0)


def bound_p5_memory_truncation(
    initial_epi, *, memory_window, times, capacity=1
) -> P5MemoryTruncationReference:
    """Bound history truncation for the abstract unit-weight P5 fixture.

    Supply five real EPI coordinates in path order 0..4, a fixed positive
    common capacity, a nonnegative retained history window and a nonempty
    ordered sequence of nonnegative sample times. Integers/Fractions remain
    exact; other real inputs use the shared finite binary64 materialization.
    The initial hidden source is retained even for a zero-length window.

    Outputs are immutable rational intervals. The full and truncated model
    trajectories are evaluated by analytical formulas, without integration
    steps. The theorem's ``macro_error_bound`` is separate from enclosure width.
    Bare float casts of interval endpoints do not preserve outward rounding.

    Resource admission requires ``capacity*time <= 2048`` and at most 32
    nonzero delayed corrections per sample. These are representation/work
    limits, not TNFR constants; rational bit size itself is not bounded here.
    """
    state = reduce_p5_state(initial_epi)
    initial = state.epi
    nu = _rational(capacity, "capacity")
    window = _rational(memory_window, "memory_window")
    sample_times = _vector(times, "times")
    if nu <= 0:
        raise ValueError("capacity must be positive")
    if window < 0 or any(time < 0 for time in sample_times):
        raise ValueError("memory_window and times must be nonnegative")
    for time in sample_times:
        if nu * time > _MAX_SCALED_TIME:
            raise ValueError("rational enclosure requires capacity*time <= 2048")
        if _correction_count(time, window) > _MAX_DELAYED_CORRECTIONS:
            raise ValueError("at most 32 nonzero delayed corrections are supported")

    a0, b0, u0 = state.memory_epi
    d0 = a0 - b0
    mean = state.conserved_mean
    c1, c2 = state.contrast_coefficients
    amplitude = state.max_reference_contrast

    cache: dict[Fraction, Interval] = {}

    def exp_bound(exponent: Fraction) -> Interval:
        if exponent not in cache:
            cache[exponent] = _negative_exp_bounds(exponent)
        return cache[exponent]

    def evaluate(terms) -> Interval:
        # Combine equal exponents exactly, retaining cancellations at t=0.
        coefficients: dict[Fraction, Fraction] = {}
        for coefficient, exponent in terms:
            coefficients[exponent] = (
                coefficients.get(exponent, Fraction(0)) + coefficient
            )
        interval = _ZERO
        for exponent, coefficient in coefficients.items():
            if coefficient:
                interval = _add(interval, _scale(coefficient, exp_bound(exponent)))
        return interval

    samples = []
    lag = nu * window
    for time in sample_times:
        scaled = nu * time
        full_terms = [(c1, scaled), (c2, 2 * scaled)]
        full = evaluate(full_terms)
        count = _correction_count(time, window)
        error_terms = []
        if time <= window:
            error = _ZERO
            truncated = full
        elif window == 0:
            truncated_terms = [
                (d0 - 4 * u0 / 3, 4 * scaled / 3),
                (4 * u0 / 3, 5 * scaled / 3),
            ]
            truncated = evaluate(truncated_terms)
            error = evaluate(
                truncated_terms
                + [(-coefficient, exponent) for coefficient, exponent in full_terms]
            )
        else:
            for index in range(1, count + 1):
                shifted = scaled - index * lag
                p1, q1 = _partial_fraction_polynomials(index + 1, index, shifted)
                p2, q2 = _partial_fraction_polynomials(index, index + 1, shifted)
                factor = Fraction(-2, 9) ** index
                error_terms.extend(
                    [
                        (factor * (c1 * p1 + c2 * p2), scaled + 2 * index * lag / 3),
                        (factor * (c1 * q1 + c2 * q2), 2 * scaled - index * lag / 3),
                    ]
                )
            error = evaluate(error_terms)
            truncated = evaluate(full_terms + error_terms)

        source = evaluate([(-4 * nu * u0 / 9, 5 * scaled / 3)])
        if time <= window:
            tail = _ZERO
            forcing_bound = macro_bound = Fraction(0)
        else:
            tail = evaluate(
                [
                    (nu * c1 / 3, scaled + 2 * lag / 3),
                    (nu * (-c1 + 2 * c2) / 3, 5 * scaled / 3),
                    (-2 * nu * c2 / 3, 2 * scaled - lag / 3),
                ]
            )
            delta_upper = exp_bound(5 * lag / 3)[1]
            elapsed = scaled - lag
            forcing_bound = (
                2
                * nu
                * amplitude
                * delta_upper
                / 15
                * (1 - exp_bound(5 * elapsed / 3)[0])
            )
            macro_bound = (
                3
                * amplitude
                / 4
                * min(
                    delta_upper / (9 + delta_upper),
                    delta_upper / 9 * (1 - exp_bound(elapsed)[0]) ** 2,
                )
            )
        full_macro = _macro(mean, full)
        truncated_macro = _macro(mean, truncated)
        samples.append(
            P5MemoryTruncationSample(
                time=time,
                reference_contrast=full,
                truncated_contrast=truncated,
                contrast_error=error,
                reference_macro=full_macro,
                truncated_macro=truncated_macro,
                initial_source_contrast=source,
                dropped_reference_forcing=tail,
                dropped_reference_forcing_bound=forcing_bound,
                macro_error_bound=macro_bound,
                macro_error_enclosure=_scale(Fraction(3, 4), _absolute(error)),
                evaluation_width=max(high - low for low, high in truncated_macro),
                delayed_corrections=count,
                coincides_before_cutoff=time <= window,
            )
        )
    return P5MemoryTruncationReference(
        initial_epi=initial,
        capacity=nu,
        memory_window=window,
        initial_macro=(a0, b0),
        initial_hidden_contrast=u0,
        conserved_mean=mean,
        contrast_coefficients=(c1, c2),
        max_reference_contrast=amplitude,
        samples=tuple(samples),
        scope=(
            "Exact fixed unit-weight P5 reference, partition ((0,4),(1,2,3)), "
            "positive common fixed capacity, real scalar EPI. The finite-window "
            "approximation retains the derived initial hidden source. Rational "
            "enclosures and model-error bounds do not bind a live graph, "
            "runtime solver, REMESH, changing support or physical experiment."
        ),
    )
