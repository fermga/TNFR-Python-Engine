"""Conditional transfer from a complete-record residual to forecast error.

This arithmetic consumes a separately proved same-law sample-to-future gain
and enclosures of one candidate history. It neither establishes those
premises nor certifies joint feasibility, source membership or physical
identification. In particular, arbitrary intersecting sample boxes are not
evidence of one realizable history.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q
from typing import Any, Iterable

from .._exact_time import exact_or_represented_real
from ._exact_linear_algebra import _ordered
from ._rational_interval import I


@dataclass(frozen=True)
class CandidateForecastError:
    """Conditional bounds, with no trajectory-admission or feasibility flag."""

    observations: tuple[Q, ...]
    candidate_samples: tuple[I, ...]
    candidate_future: tuple[I, ...]
    gain: Q
    source_error: Q
    reading_error: Q
    residual_upper: Q
    carrier_sample_difference_upper: Q
    candidate_exact_future_error_upper: Q
    forecast_center: tuple[Q, ...]
    true_output_radii: tuple[Q, ...]
    maximum_true_output_radius: Q


def candidate_forecast_error(
    observations: Iterable[Any],
    candidate_samples: Iterable[I],
    candidate_future: Iterable[I],
    *,
    gain: Any,
    source_error: Any,
    reading_error: Any,
) -> CandidateForecastError:
    """Bound forecast error conditional on one retained candidate trajectory.

    The caller establishes that the true source differs from its reference
    by at most ``source_error`` at every sample and at the future, and that
    reference pairs in the actual admitted model obey a sample-to-future
    infinity-norm bound with factor ``gain``. Every candidate sample and
    future interval must enclose the same continuously carried reference
    solution under that law, input, clock and domain. A record-consuming
    model owner must re-admit/rebuild these premises; this scalar calculation
    cannot substitute for their proof or authenticate interval reports.

    Raw observations have error at most ``reading_error``. We use the upper
    distance to each candidate interval, not its nearest point or overlap
    with a reading band. Historical numerical errors are thus already in rho.
    The exact candidate future differs from the true output by at most
    gain*(reading_error+source_error+rho)+source_error. Each endpoint interval
    radius is then added once to obtain a bound about its exact midpoint.

    All bounds are deterministic; arbitrary correlation is allowed. The
    result concerns the true future output. A future reading requires its
    separately declared sensor allowance. No zero common-form prior, phase
    reconstruction, existence of a compatible source or acceptance policy
    is inferred. Failure of a candidate's bound proves no record obstruction.
    """
    raw = tuple(
        exact_or_represented_real(value, "observation")
        for value in _ordered(observations, "observations")
    )

    def intervals(values: Iterable[I], name: str) -> tuple[I, ...]:
        rows = tuple(_ordered(values, name))
        if not rows:
            raise ValueError(f"{name} must be nonempty")
        if any(not isinstance(value, I) for value in rows):
            raise TypeError(f"{name} requires explicit rational intervals")
        return rows

    samples = intervals(candidate_samples, "candidate samples")
    future = intervals(candidate_future, "candidate future")
    if len(raw) != len(samples):
        raise ValueError(
            "observations and candidate samples must have matching lengths"
        )
    gain, source_error, reading_error = (
        exact_or_represented_real(value, name)
        for value, name in (
            (gain, "gain"),
            (source_error, "source error"),
            (reading_error, "reading error"),
        )
    )
    if min(gain, source_error, reading_error) < 0:
        raise ValueError("gain and error bounds must be nonnegative")
    residual = max(
        max(abs(value.lo - observed), abs(value.hi - observed))
        for observed, value in zip(raw, samples)
    )
    separation = reading_error + source_error + residual
    exact_error = gain * separation + source_error
    radii = tuple(exact_error + value.radius for value in future)
    return CandidateForecastError(
        raw,
        samples,
        future,
        gain,
        source_error,
        reading_error,
        residual,
        separation,
        exact_error,
        tuple(value.midpoint for value in future),
        radii,
        max(radii),
    )
