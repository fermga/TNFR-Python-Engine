"""Exact modified-energy kernels for conditional nonlinear sine recovery.

Public owners admit the full law, mean leaf, acute trapping neighborhood and
primitive values. These private kernels take normalized rational bounds only;
they neither admit a report nor establish those geometric premises. They run
no exponential evaluator, choose no clock or horizon, and cache no evidence.
"""

from __future__ import annotations

from fractions import Fraction as Q


def _sine_lyapunov_coefficients(
    *,
    eta_lower: Q,
    eta_upper: Q,
    cosine_lower: Q | None,
    gap_lower: Q,
    rate_upper: Q,
) -> tuple[Q | None, Q, Q | None, Q, Q | None]:
    """Return ``(mu, M, a_minus, a_plus, decay_rate)`` for admitted bounds.

    Callers supply ``0 <= eta_lower <= eta_upper``, ``0 < gap_lower <=
    rate_upper`` and an admitted nonnegative acute cosine lower bound. None
    denotes a missing cosine premise, retaining only the upper coefficients.
    A zero lower stiffness supplies no positive decay rate. The spectral and
    nonlinear Hessian hypotheses belong to the caller's complete model.
    """
    epsilon = gap_lower / 4
    mu = eta_lower * cosine_lower * gap_lower**2 if cosine_lower is not None else None
    upper_stiffness = eta_upper * rate_upper**2
    lower_position = mu / 2 + gap_lower**2 / 16 if mu is not None else None
    upper_position = upper_stiffness / 2 + epsilon * rate_upper / 2 + epsilon**2
    decay_rate = (
        min(4 * (gap_lower - epsilon) / 3, epsilon * mu / upper_position)
        if mu is not None and mu > 0
        else None
    )
    return mu, upper_stiffness, lower_position, upper_position, decay_rate


def _sine_lyapunov_initial_upper(
    *,
    eta_upper: Q,
    rate_upper: Q,
    gap_lower: Q,
    position_upper: Q,
    form_norm_upper: Q,
    phase_norm_upper: Q,
) -> Q:
    """Bound modified energy from admitted Euclidean form/target-phase radii.

    Radii and coefficients are nonnegative; ``gap_lower`` is positive. The
    phase radius is about the critical target, not a nominal source proxy.
    No intervention is presumed: the same expression applies to an unprobed
    formed family or a separately admitted post-intervention family.
    """
    return (
        Q(3, 4) * eta_upper * rate_upper * form_norm_upper**2
        + position_upper * phase_norm_upper**2 / gap_lower
    )


def _sine_lyapunov_return_squared(
    *,
    initial_upper: Q,
    decay_upper: Q,
    eta_lower: Q,
    gap_lower: Q,
    rate_upper: Q,
    position_lower: Q,
) -> tuple[Q, Q, Q]:
    """Return energy, Euclidean form norm squared and phase norm squared.

    The caller proves uninterrupted trapping and supplies a nonnegative
    exponential upper bound; positive denominator bounds are already admitted.
    All arithmetic stays rational, including decay bounds below an interval
    grid. Exponential precision, work caps and final verdicts belong to the
    consuming owner. Neither an event nor a half-radius threshold is assumed.
    """
    returned = initial_upper * decay_upper
    form_squared = 4 * returned / (eta_lower * gap_lower)
    phase_squared = rate_upper * returned / position_lower
    return returned, form_squared, phase_squared
