"""Detached bounds for the supplied loaded C5 observation model.

``theory/research/ORGANIZED_FINITE_WINDOW_DESIGN.md`` owns the complete law,
declaration and proof. This calculation evaluates conditional inequalities;
it integrates no trajectory, admits no apparatus and infers no preparation
or physical law from its own predicted response.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q

from .._exact_time import exact_or_represented_real, exp_unit_bounds
from ._organized_c5_preparation import (
    IDENTITY_RADIUS,
    STORAGE_BARRIER,
    prepared_c5_initial_energy_upper,
)


@dataclass(frozen=True)
class LoadedWindowObservationBound:
    """Exact bounds rebuilt from the retained primitive assumptions."""

    response_lower: Q
    response_upper: Q
    amplitude: Q
    horizon: Q
    preparation_radius: Q
    sensor_preparation_radius: Q
    sensor_coupling: Q
    observation_error: Q
    identity_radius: Q
    storage_barrier: Q
    initial_energy_upper: Q
    energy_upper: Q
    state_norm_upper: Q
    form_rate_remainder_upper: Q
    response_remainder: Q
    average_ramp_factor_bounds: tuple[Q, Q]
    response_gain_lower: Q
    response_gain_upper: Q
    positive_observation_bounds: tuple[Q, Q]
    control_observation_bounds: tuple[Q, Q]
    response_gap: Q
    pulse_absolute_work_upper: Q
    pulse_loss_upper: Q
    sensor_loss_upper: Q
    identity_certified: bool
    response_separated: bool
    phase_blind_difference_upper: Q
    phase_blind_records_overlap: bool


def bound_loaded_window_observation(
    *,
    response_lower: object = Q(1, 100),
    response_upper: object = Q(1, 10),
    amplitude: object = Q(1, 16),
    horizon: object = Q(1, 1024),
    preparation_radius: object = Q(1, 2**26),
    sensor_preparation_radius: object = Q(1, 2**26),
    sensor_coupling: object = Q(1, 64),
    observation_error: object = Q(1, 2**26),
) -> LoadedWindowObservationBound:
    """Bound a loaded finite-window response and its phase-blind control.

    The complete C5 model holds unit capacities, the structural clock and
    ``gamma=1/(1023*pi)``. The centered initial form and continuous phase
    perturbation about winding +1 or -1 satisfy the supplied joint radius.
    The independently bounded sensor state obeys ``T*z'=v^T*L*U-z`` and
    loads the form row by ``-k*v*(v^T*L*U-z)`` before multiplication by
    ``M_r``. Thus its time constant equals the positive pulse horizon; it
    is not another fitted parameter. Input is ``amplitude*v`` during that
    horizon and zero afterwards, with no jumps. The observed scalar is the
    sensor state's average over the whole pulse, with additive error.

    ``identity_certified`` protects the fixed 1/10 phase-gap collar during
    the pulse and subsequent unforced evolution using the loaded storage.
    The finite response and work bounds do not require that sufficient
    identity check to pass. ``response_separated`` checks only a positive
    family/control interval gap; larger protocol margins remain separate.

    The phase-blind comparison retains the same r, form/sensor initial
    state, input, clock and sensor while omitting the sine-current branch.
    A true overlap flag certifies intersection of each pair's permitted
    additive-error records. A false flag is inconclusive, not a proof of
    phase-sensitive separation. No flag establishes physical preparation,
    measurement capability, source acquisition or constitutive selection.

    Rational primitives retain exact values with normalized integer
    components; other admitted reals denote their binary64 representations.
    Cached reports, fitted responses and evaluated evidence are not used.
    """
    names = (
        "response_lower",
        "response_upper",
        "amplitude",
        "horizon",
        "preparation_radius",
        "sensor_preparation_radius",
        "sensor_coupling",
        "observation_error",
    )
    rlo, rhi, a, duration, delta, sensor_delta, k, eta = (
        exact_or_represented_real(value, name)
        for value, name in zip(
            (
                response_lower,
                response_upper,
                amplitude,
                horizon,
                preparation_radius,
                sensor_preparation_radius,
                sensor_coupling,
                observation_error,
            ),
            names,
        )
    )
    if not 0 < rlo <= rhi:
        raise ValueError("response interval requires 0 < lower <= upper")
    if a <= 0:
        raise ValueError("amplitude must be strictly positive")
    if duration <= 0:
        raise ValueError("horizon must be strictly positive")
    if k <= 0:
        raise ValueError("sensor coupling must be strictly positive")
    if min(delta, sensor_delta, eta) < 0:
        raise ValueError("preparation radii and observation error must be nonnegative")
    if (2 + 20 * k) * duration >= 1:
        raise ValueError("horizon and sensor coupling require (2 + 20*k)*T < 1")

    rho, barrier = IDENTITY_RADIUS, STORAGE_BARRIER
    initial_energy = (
        prepared_c5_initial_energy_upper(response_upper=rhi, preparation_radius=delta)
        + k * duration * sensor_delta**2 / 2
    )
    energy = initial_energy + 2 * a**2 * duration
    state_norm = (delta + 2 * a * duration + 2 * k * duration * sensor_delta) / (
        1 - (2 + 20 * k) * duration
    )
    rate_remainder = (1 + 20 * k) * state_norm + 2 * k * sensor_delta
    remainder = sensor_delta + 5 * delta + 5 * duration * rate_remainder / 2

    # The average of the sensor's response to a unit ramp is T*alpha.
    # Reciprocal endpoints reverse order when bounding exp(-1).
    e_lower, e_upper = exp_unit_bounds(Q(1))
    alpha_lower = Q(1, 2) - 1 / e_lower
    alpha_upper = Q(1, 2) - 1 / e_upper

    def gain(response: Q) -> Q:
        # v^T L M_r v decreases for nonnegative r. This voltage port is
        # different from the form contrast v^T U in the unloaded theorem.
        return (6 + 20 * response) / (1 + 15 * response + 25 * response**2)

    lower_gain, upper_gain = gain(rhi), gain(rlo)
    uncertainty = remainder + eta
    positive = (
        a * duration * alpha_lower * lower_gain - uncertainty,
        a * duration * alpha_upper * upper_gain + uncertainty,
    )
    control = (
        6 * a * duration * alpha_lower - uncertainty,
        6 * a * duration * alpha_upper + uncertainty,
    )
    gap = control[0] - positive[1]

    # gamma < 1/3000 follows from pi > 3. The comparison keeps the actual
    # phase row and bounds its effect against a matched phase-blind model.
    gamma_upper = Q(1, 3000)
    phase_blind_difference = (
        5
        * gamma_upper
        * (delta * duration + gamma_upper * state_norm * duration**2 / 2)
        / (1 - (1 + 20 * k) * duration)
    )
    return LoadedWindowObservationBound(
        response_lower=rlo,
        response_upper=rhi,
        amplitude=a,
        horizon=duration,
        preparation_radius=delta,
        sensor_preparation_radius=sensor_delta,
        sensor_coupling=k,
        observation_error=eta,
        identity_radius=rho,
        storage_barrier=barrier,
        initial_energy_upper=initial_energy,
        energy_upper=energy,
        state_norm_upper=state_norm,
        form_rate_remainder_upper=rate_remainder,
        response_remainder=remainder,
        average_ramp_factor_bounds=(alpha_lower, alpha_upper),
        response_gain_lower=lower_gain,
        response_gain_upper=upper_gain,
        positive_observation_bounds=positive,
        control_observation_bounds=control,
        response_gap=gap,
        pulse_absolute_work_upper=8 * a * duration * state_norm,
        pulse_loss_upper=4 * duration * state_norm**2,
        sensor_loss_upper=k * duration * (sensor_delta + 10 * state_norm) ** 2,
        identity_certified=2 * delta < rho and energy < barrier,
        response_separated=gap > 0,
        phase_blind_difference_upper=phase_blind_difference,
        phase_blind_records_overlap=phase_blind_difference <= 2 * eta,
    )
