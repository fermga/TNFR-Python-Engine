"""Shared exact decisions for an eight-reading class contrast.

Callers supply a justified reference enclosure and its complete error budget.
These arithmetic decisions supply neither a response nor source admission.
"""

from dataclasses import dataclass
from fractions import Fraction as Q

from ..mathematics._rational_interval import _exact


@dataclass(frozen=True)
class _ContrastDecision:
    true_bounds: tuple[Q, Q]
    recorded_bounds: tuple[Q, Q]
    orientation: int
    oriented_lower: Q | None
    recorded_sign_margin: Q | None
    null_separation_margin: Q | None
    noise_ceiling: Q | None
    true_sign: bool
    recorded_sign: bool
    null_excluded: bool
    cancellation_margin: Q
    scalar_cancellation: bool
    status: str


def _contrast_decision(reference_bounds, total_error, delta, *, exact_zero=False):
    """Expand exact bounds, keeping sign, null exclusion and cancellation apart.

    A scalar cancellation flag proves that each allowed true contrast admits
    some eight reading errors making its mixed contrast zero. It does not
    certify overlap of raw reading vectors or one error choice for all states.
    """
    lower, upper = map(_exact, reference_bounds)
    error, noise = _exact(total_error), _exact(delta)
    if lower > upper or min(error, noise) < 0:
        raise ValueError("ordered reference bounds and nonnegative errors required")
    if type(exact_zero) is not bool:
        raise TypeError("exact_zero must be an admitted Boolean identity")
    true = (Q(0), Q(0)) if exact_zero else (lower - error, upper + error)
    recorded = true[0] - 8 * noise, true[1] + 8 * noise
    orientation = 0 if exact_zero else 1 if lower > 0 else -1 if upper < 0 else 0
    oriented = (true[0] if orientation > 0 else -true[1]) if orientation else None
    sign_margin = None if oriented is None else oriented - 8 * noise
    null_margin = None if oriented is None else oriented - 16 * noise
    true_sign = oriented is not None and oriented > 0
    recorded_sign = sign_margin is not None and sign_margin > 0
    null_excluded = null_margin is not None and null_margin > 0
    cancellation_margin = 8 * noise - max(map(abs, true))
    status = (
        "zero_contrast_record_sets_disjoint"
        if null_excluded
        else (
            "recorded_sign_certified"
            if recorded_sign
            else (
                "true_sign_certified"
                if true_sign
                else "exact_contrast_zero" if exact_zero else "bounds_only"
            )
        )
    )
    return _ContrastDecision(
        true,
        recorded,
        orientation,
        oriented,
        sign_margin,
        null_margin,
        oriented / 8 if true_sign else None,
        true_sign,
        recorded_sign,
        null_excluded,
        cancellation_margin,
        cancellation_margin >= 0,
        status,
    )
