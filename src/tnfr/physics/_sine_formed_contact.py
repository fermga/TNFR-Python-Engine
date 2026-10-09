"""Shared actual-family handoff and joined C9 identity proof bounds.

Private callers admit every primitive before entry. The handoff reconstructs
formation from those primitives; no incoming report or phase probe is used.
The joined bounds apply to either class in {1, 2}, independently of a finite
response approximation. Neither helper installs an event or evolution law.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q

from ..mathematics._rational_interval import I, cos, pi_interval
from ._sine_lyapunov import (
    _sine_lyapunov_coefficients,
    _sine_lyapunov_initial_upper,
    _sine_lyapunov_return_squared,
)
from .relational_sine_formed_classes import (
    SineFormedClassPair,
    assess_sine_formed_class_pair,
)
from .reversible_eigenmode_reference import _MAX_RATIONAL_EXPONENT


@dataclass(frozen=True)
class _UnprobedHandoff:
    formation_certificate: SineFormedClassPair
    lyapunov_decay_rate: Q | None
    decay_exponent: Q | None
    exact_decay_upper_bound: Q | None
    initial_lyapunov_upper_bounds: tuple[Q, Q]
    endpoint_form_norm_squared_upper_bounds: tuple[Q, Q] | None
    endpoint_phase_norm_squared_upper_bounds: tuple[Q, Q] | None
    handoff_certified_by_class: tuple[bool, bool]


def _unprobed_handoff(
    *,
    formation_time,
    relaxation_duration,
    form_error_bound,
    phase_error_bound,
    radius,
    endpoint_radius,
    decay_power,
) -> _UnprobedHandoff:
    fresh = assess_sine_formed_class_pair(
        scaled_time=formation_time,
        form_error_bound=form_error_bound,
        phase_error_bound=phase_error_bound,
        radius=radius,
    )
    eta = fresh.eta_bounds
    _, _, lower, upper, rate = _sine_lyapunov_coefficients(
        eta_lower=eta.lo,
        eta_upper=eta.hi,
        cosine_lower=(
            (fresh.coercivity_lower_bound / Q(1, 5))
            if fresh.coercivity_lower_bound is not None
            else None
        ),
        gap_lower=Q(1, 5),
        rate_upper=Q(2),
    )
    initial = tuple(
        _sine_lyapunov_initial_upper(
            eta_upper=eta.hi,
            rate_upper=Q(2),
            gap_lower=Q(1, 5),
            position_upper=upper,
            form_norm_upper=x,
            phase_norm_upper=distance.hi + phase,
        )
        for x, distance, phase in zip(
            fresh.endpoint_form_norm_upper_bounds,
            fresh.proxy_target_distance_bounds,
            fresh.endpoint_phase_error_norm_upper_bounds,
        )
    )
    exponent = rate * relaxation_duration if rate is not None else None
    if exponent is not None and exponent > _MAX_RATIONAL_EXPONENT:
        raise ValueError("lyapunov_decay_rate*relaxation_duration must not exceed4096")
    decay = xs = ys = None
    flags = (False, False)
    eps = endpoint_radius
    if (
        fresh.status == "certified_two_formed_classes"
        and all(fresh.formation_certified_by_class)
        and exponent is not None
        and exponent >= decay_power
    ):
        # exp(-z) <= 2**(-N) for z>=N>=0; retain exact tiny squared bounds.
        decay = Q(1, 2**decay_power)
        returned = tuple(
            _sine_lyapunov_return_squared(
                initial_upper=value,
                decay_upper=decay,
                eta_lower=eta.lo,
                gap_lower=Q(1, 5),
                rate_upper=Q(2),
                position_lower=lower,
            )
            for value in initial
        )
        xs, ys = tuple(row[1] for row in returned), tuple(row[2] for row in returned)
        flags = tuple(x <= eps**2 and y <= eps**2 for x, y in zip(xs, ys))

    return _UnprobedHandoff(
        formation_certificate=fresh,
        lyapunov_decay_rate=rate,
        decay_exponent=exponent,
        exact_decay_upper_bound=decay,
        initial_lyapunov_upper_bounds=initial,
        endpoint_form_norm_squared_upper_bounds=xs,
        endpoint_phase_norm_squared_upper_bounds=ys,
        handoff_certified_by_class=flags,
    )


@dataclass(frozen=True)
class _JoinedContactBounds:
    joined_radius_squared_upper_bound: Q
    joined_excess_storage_upper_bound: Q
    joined_barrier_lower_bound: Q
    joined_radius_margin_bounds: I
    joined_storage_margin_bounds: I
    bridge_work_bounds: I
    work_margin_bounds: I
    joined_form_mean_bounds: I
    joined_phase_mean_bounds: I
    identity_certified: bool
    work_within_allowance: bool


def _joined_contact_bounds(
    *,
    endpoint_radius,
    phase_origin_difference,
    radius,
    work_allowance,
) -> _JoinedContactBounds:
    eps, phi = endpoint_radius, phase_origin_difference
    pi = pi_interval()
    barrier = Q(1, 1620) * radius**2
    z2 = 4 * eps**2 + Q(9, 2) * phi**2
    energy = 10 * eps**2 + (phi + 2 * eps) ** 2 / 2
    radius_margin = I(radius**2 - z2)
    storage_margin = I(barrier - energy)
    identity = radius_margin.lo > 0 and storage_margin.lo > 0
    work_upper = 2 * eps**2 + (phi + 2 * eps) ** 2 / 2
    work_lower = (
        max(Q(0), (1 - cos(I(phi - 2 * eps))).lo)
        if 0 < phi - 2 * eps and phi + 2 * eps < pi.lo
        else Q(0)
    )
    work = I(work_lower, work_upper)
    work_margin = I(work_allowance - work_upper)
    allowed = work_upper <= work_allowance
    mean_x = I(-eps / 19, eps / 19)
    mean_phase = phi / 2 + mean_x
    return _JoinedContactBounds(
        joined_radius_squared_upper_bound=z2,
        joined_excess_storage_upper_bound=energy,
        joined_barrier_lower_bound=barrier,
        joined_radius_margin_bounds=radius_margin,
        joined_storage_margin_bounds=storage_margin,
        bridge_work_bounds=work,
        work_margin_bounds=work_margin,
        joined_form_mean_bounds=mean_x,
        joined_phase_mean_bounds=mean_phase,
        identity_certified=identity,
        work_within_allowance=allowed,
    )
