"""Detached area bounds for the fixed loaded C5 pulse and its recovery.

``theory/research/ORGANIZED_RECOVERY_OBSERVATION.md`` owns the complete models,
fixed declaration and own-law decay proof. This consumer performs no trajectory
evaluation and supplies no physical preparation, sensor or clock admission.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q

from .._exact_time import exact_or_represented_real
from ._organized_window_observation import bound_loaded_window_observation


@dataclass(frozen=True)
class RecoveryAreaObservationBound:
    """Conditional bounds retaining the fixed pulse and variable window."""

    response_lower: Q
    response_upper: Q
    amplitude: Q
    pulse_duration: Q
    sensor_coupling: Q
    sensor_time_constant: Q
    preparation_radius: Q
    sensor_preparation_radius: Q
    recovery_duration: Q
    area_error: Q
    total_horizon: Q
    pulse_energy_upper: Q
    decay_exponent: int
    decay_factor_upper: Q
    full_state_squared_norm_upper: Q
    linear_state_squared_norm_upper: Q
    endpoint_norm_upper: Q
    sensor_endpoint_upper: Q
    full_terminal_remainder_upper: Q
    linear_terminal_remainder_upper: Q
    full_asymptotic_area_bounds: tuple[Q, Q]
    linear_asymptotic_area_bounds: tuple[Q, Q]
    full_area_bounds: tuple[Q, Q]
    linear_area_bounds: tuple[Q, Q]
    response_gap: Q
    identity_certified: bool
    response_separated: bool


def bound_recovery_area_observation(
    *,
    recovery_duration: object = Q(2**40),
    area_error: object = Q(1, 2**18),
) -> RecoveryAreaObservationBound:
    """Bound the raw sensor area through a declared unforced recovery.

    All model, input, preparation and sensor primitives are fixed by the
    owner's declaration. In particular, the pulse duration and sensor time
    constant remain 1/1024 when recovery duration changes. The positive-r
    family is compared with a passive phase-blind model having the same r,
    linear components, initial form/sensor state, input and clock.

    The statistic is ``integral_0^(pulse_duration+recovery_duration) z(t) dt``;
    ``area_error`` bounds error on that integral, not on a mean voltage or
    each sample. Terminal sensor terms are bounded, not measured or silently
    dropped. Both complete-model identities retain initial state terms.

    The own-law decay proof applies on the retained identity collar. Its
    structural rate exceeds 2^-32, so ``n=floor(recovery_duration/2^32)``
    gives the rational decay majorant 2^-n. The closed arithmetic domain
    permits 0<=n<=4096. A zero duration is admitted and can give inconclusive
    bounds. No integration, exponential fit or source replay is performed.

    ``response_separated`` means a strictly positive gap between the entire
    observed intervals. A protocol-specific larger margin is a separate
    requirement. A false flag supplies no realized collision. The identity
    and response flags do not admit an apparatus, preparation or physical
    law. Rational inputs retain normalized exact values; other admitted
    reals denote their binary64 representations.
    """
    duration = exact_or_represented_real(recovery_duration, "recovery_duration")
    eta = exact_or_represented_real(area_error, "area_error")
    if duration < 0:
        raise ValueError("recovery duration must be nonnegative")
    if eta < 0:
        raise ValueError("area error must be nonnegative")
    exponent = duration // 2**32
    if exponent > 4096:
        raise ValueError("recovery decay exponent exceeds the 4096 arithmetic budget")

    # Rebuild the fixed pulse's own loaded storage bound from its primitive
    # declaration. The subsequent raw-area error is a different observation.
    pulse = bound_loaded_window_observation(
        response_lower=Q(1, 100),
        response_upper=Q(1, 10),
        amplitude=Q(1, 16),
        horizon=Q(1, 1024),
        preparation_radius=Q(1, 2**26),
        sensor_preparation_radius=Q(1, 2**26),
        sensor_coupling=Q(1, 64),
        observation_error=Q(0),
    )
    if not pulse.identity_certified or 320 * pulse.energy_upper >= Q(1, 2**7):
        raise ValueError("the fixed pulse does not meet the recovery proof premises")

    decay = Q(1, 2**exponent)
    full_state_squared = 320 * pulse.energy_upper * decay
    linear_state_squared = 2 * pulse.energy_upper * decay
    # N_full <= 320*E_pulse*2^-n < 2^(-n-7); N_linear has the same
    # weaker envelope. This rational norm majorant avoids rounded roots.
    endpoint_norm = Q(1, 2 ** ((exponent + 7) // 2))
    sensor_endpoint = 256 * endpoint_norm  # sqrt(k*tau_s) = 1/256.
    tau_s = pulse.horizon
    k = pulse.sensor_coupling
    delta = pulse.preparation_radius
    sensor_delta = pulse.sensor_preparation_radius

    # ||v|| < 3/2, gamma^-1 < 4096, and ||W_r|| <= 13/5 hold throughout
    # the fixed family. The initial phase and form terms remain distinct.
    full_prior = 6 * 4096 * delta + tau_s * sensor_delta
    linear_prior = 6 * Q(13, 5) * delta + (1 + 8 * k) * tau_s * sensor_delta
    linear_center = 8 * pulse.amplitude * pulse.horizon
    full_terminal = 6 * 4096 * endpoint_norm + tau_s * sensor_endpoint
    linear_terminal = (
        6 * Q(13, 5) * endpoint_norm + (1 + 8 * k) * tau_s * sensor_endpoint
    )
    full_asymptotic = (-full_prior, full_prior)
    linear_asymptotic = (
        linear_center - linear_prior,
        linear_center + linear_prior,
    )
    full = (
        full_asymptotic[0] - full_terminal - eta,
        full_asymptotic[1] + full_terminal + eta,
    )
    linear = (
        linear_asymptotic[0] - linear_terminal - eta,
        linear_asymptotic[1] + linear_terminal + eta,
    )
    gap = linear[0] - full[1]
    return RecoveryAreaObservationBound(
        response_lower=pulse.response_lower,
        response_upper=pulse.response_upper,
        amplitude=pulse.amplitude,
        pulse_duration=pulse.horizon,
        sensor_coupling=k,
        sensor_time_constant=tau_s,
        preparation_radius=delta,
        sensor_preparation_radius=sensor_delta,
        recovery_duration=duration,
        area_error=eta,
        total_horizon=pulse.horizon + duration,
        pulse_energy_upper=pulse.energy_upper,
        decay_exponent=exponent,
        decay_factor_upper=decay,
        full_state_squared_norm_upper=full_state_squared,
        linear_state_squared_norm_upper=linear_state_squared,
        endpoint_norm_upper=endpoint_norm,
        sensor_endpoint_upper=sensor_endpoint,
        full_terminal_remainder_upper=full_terminal,
        linear_terminal_remainder_upper=linear_terminal,
        full_asymptotic_area_bounds=full_asymptotic,
        linear_asymptotic_area_bounds=linear_asymptotic,
        full_area_bounds=full,
        linear_area_bounds=linear,
        response_gap=gap,
        identity_certified=pulse.identity_certified,
        response_separated=gap > 0,
    )
