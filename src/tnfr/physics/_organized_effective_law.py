"""Detached finite bounds for the supplied collective matrix-response law.

``theory/nodal/ORGANIZED_EFFECTIVE_LAW.md`` owns the complete C5 law,
prepared-reference declaration and analytical proof. This calculation admits
primitive bounds and evaluates that proof; it integrates no trajectory and
does not certify that an observed preparation satisfies the supplied bounds.
No fine-source acquisition, physical measurement or event law is installed.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q

from .._exact_time import exact_or_represented_real
from ._organized_c5_preparation import (
    IDENTITY_RADIUS,
    STORAGE_BARRIER,
    prepared_c5_initial_energy_upper,
)


@dataclass(frozen=True)
class PreparedMatrixResponseBound:
    """Exact conditional bounds retaining every consumed primitive."""

    response_lower: Q
    response_upper: Q
    amplitude: Q
    horizon: Q
    preparation_radius: Q
    observation_error: Q
    identity_radius: Q
    storage_barrier: Q
    initial_energy_upper: Q
    energy_upper: Q
    state_norm_upper: Q
    response_remainder: Q
    response_gain_lower: Q
    response_gain_upper: Q
    positive_observation_bounds: tuple[Q, Q]
    control_observation_bounds: tuple[Q, Q]
    response_gap: Q
    pulse_absolute_work_upper: Q
    pulse_loss_upper: Q
    identity_certified: bool
    response_separated: bool


def bound_prepared_matrix_response(
    *,
    response_lower=Q(1, 100),
    response_upper=Q(1, 10),
    amplitude=Q(1, 16),
    horizon=Q(1, 1024),
    preparation_radius=Q(1, 2**26),
    observation_error=Q(1, 2**26),
) -> PreparedMatrixResponseBound:
    """Bound the declared C5 pulse for a positive-r family and its r=0 control.

    The supplied law holds unit capacities, ``gamma=1/(1023*pi)``, the fixed
    support and the structural clock. Each preparation has zero form mean
    and a zero-mean continuous phase perturbation about winding +1 or -1,
    with ``||U(0)||_2 + ||theta(0)||_2 <= preparation_radius``. Its input is
    ``amplitude*(1,-1,0,0,0)`` through the complete row ``W_r U'=...+f``
    during the pulse, then zero. The observation is ``U_0(T)-U_1(T)`` with
    bounded additive error. Preparation and observation bounds are premises,
    not inferred from the predicted response.

    The identity flag uses the own-law energy barrier for the fixed gap
    collar ``rho=1/10``. When true, it protects that identity during the
    pulse and all subsequent unforced evolution. The finite response and
    pulse work/loss bounds hold even when this sufficient identity check
    fails. ``response_separated`` means only a strictly positive interval
    gap; a protocol-specific larger separation allowance belongs to its
    declaration. A false flag is an insufficient certificate, not a proved
    failure of identity or a realized response collision.

    Rational inputs retain their exact values; other admitted real values
    denote their binary64 representations. Both bounds and control are
    rebuilt from normalized primitives, without cached reports or evidence.
    """
    names = (
        "response_lower",
        "response_upper",
        "amplitude",
        "horizon",
        "preparation_radius",
        "observation_error",
    )
    rlo, rhi, a, duration, delta, eta = (
        exact_or_represented_real(value, name)
        for value, name in zip(
            (
                response_lower,
                response_upper,
                amplitude,
                horizon,
                preparation_radius,
                observation_error,
            ),
            names,
        )
    )
    if not 0 < rlo <= rhi:
        raise ValueError("response interval requires 0 < lower <= upper")
    if a <= 0:
        raise ValueError("amplitude must be strictly positive")
    if not 0 < duration < Q(1, 2):
        raise ValueError("horizon must lie strictly between zero and one half")
    if min(delta, eta) < 0:
        raise ValueError("preparation radius and observation error must be nonnegative")

    rho, barrier = IDENTITY_RADIUS, STORAGE_BARRIER
    initial_energy = prepared_c5_initial_energy_upper(
        response_upper=rhi, preparation_radius=delta
    )
    energy = initial_energy + 2 * a**2 * duration
    state_norm = (delta + 2 * a * duration) / (1 - 2 * duration)
    remainder = 2 * delta + 2 * duration * state_norm

    def gain(response):
        # v^T M_r v decreases for r>=0; this rational C5 identity retains
        # the complete matrix input rather than treating M_r as a capacity.
        return 2 * (1 + 5 * response) / (1 + 15 * response + 25 * response**2)

    lower_gain, upper_gain = gain(rhi), gain(rlo)
    uncertainty = remainder + eta
    positive = (
        a * duration * lower_gain - uncertainty,
        a * duration * upper_gain + uncertainty,
    )
    control = (2 * a * duration - uncertainty, 2 * a * duration + uncertainty)
    gap = control[0] - positive[1]
    return PreparedMatrixResponseBound(
        response_lower=rlo,
        response_upper=rhi,
        amplitude=a,
        horizon=duration,
        preparation_radius=delta,
        observation_error=eta,
        identity_radius=rho,
        storage_barrier=barrier,
        initial_energy_upper=initial_energy,
        energy_upper=energy,
        state_norm_upper=state_norm,
        response_remainder=remainder,
        response_gain_lower=lower_gain,
        response_gain_upper=upper_gain,
        positive_observation_bounds=positive,
        control_observation_bounds=control,
        response_gap=gap,
        pulse_absolute_work_upper=8 * a * duration * state_norm,
        pulse_loss_upper=4 * duration * state_norm**2,
        identity_certified=2 * delta < rho and energy < barrier,
        response_separated=gap > 0,
    )
