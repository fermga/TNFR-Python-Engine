"""Shared energy admission for the supplied loaded collective C5 pulse.

``theory/research/ORGANIZED_FINITE_WINDOW_DESIGN.md`` owns the complete
matrix-response and passive-sensor law. The sensor time constant equals the
pulse horizon. This owner retains that model's primitive preparation and
energy balance; it computes no observation, transfer gain or recovery rate.
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
class LoadedC5PulseBound:
    """Normalized pulse primitives and their conditional energy bounds."""

    response_lower: Q
    response_upper: Q
    amplitude: Q
    horizon: Q
    preparation_radius: Q
    sensor_preparation_radius: Q
    sensor_coupling: Q
    identity_radius: Q
    storage_barrier: Q
    initial_energy_upper: Q
    energy_upper: Q
    identity_certified: bool


def bound_loaded_c5_pulse(
    *,
    response_lower: object,
    response_upper: object,
    amplitude: object,
    horizon: object,
    preparation_radius: object,
    sensor_preparation_radius: object,
    sensor_coupling: object,
) -> LoadedC5PulseBound:
    """Bound loaded storage during a fixed-support C5 pulse and release.

    The declared family has ``0 < response_lower <= r <= response_upper``;
    the energy bound also covers r=0. The joint centered form/phase radius
    and independent sensor radius are supplied premises. Input is the
    neutral vector ``amplitude*(1,-1,0,0,0)`` during the positive horizon,
    then zero, with held sensor time constant equal to that horizon.

    Completing the square bounds energy growth by ``2*amplitude**2*horizon``.
    This is neither supplied work nor an observation error. A true identity
    flag excludes first contact with the fixed phase collar during the pulse
    and later unforced evolution. It does not establish preparation, source
    acquisition, a sensor readout or a physical realization. A false flag is
    an insufficient certificate, not a demonstrated identity failure.

    This energy bound has no short-window response-denominator restriction.
    Consumers deriving state or observation bounds must admit their own
    additional domains. Every primitive is normalized before arithmetic;
    no cached report or observation calculation is consumed.
    """
    names = (
        "response_lower",
        "response_upper",
        "amplitude",
        "horizon",
        "preparation_radius",
        "sensor_preparation_radius",
        "sensor_coupling",
    )
    rlo, rhi, a, duration, delta, sensor_delta, k = (
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
    if min(delta, sensor_delta) < 0:
        raise ValueError("preparation radii must be nonnegative")

    initial_energy = (
        prepared_c5_initial_energy_upper(response_upper=rhi, preparation_radius=delta)
        + k * duration * sensor_delta**2 / 2
    )
    energy = initial_energy + 2 * a**2 * duration
    return LoadedC5PulseBound(
        response_lower=rlo,
        response_upper=rhi,
        amplitude=a,
        horizon=duration,
        preparation_radius=delta,
        sensor_preparation_radius=sensor_delta,
        sensor_coupling=k,
        identity_radius=IDENTITY_RADIUS,
        storage_barrier=STORAGE_BARRIER,
        initial_energy_upper=initial_energy,
        energy_upper=energy,
        identity_certified=2 * delta < IDENTITY_RADIUS and energy < STORAGE_BARRIER,
    )
