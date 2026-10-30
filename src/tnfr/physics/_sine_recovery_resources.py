"""Own-class-one recovery resources for freshly admitted isolated C9 sources.

This analytical consumer retains the actual acquired state and its full
relative preparation. The quarter-law dwell rescales both half-law rows;
no reached state, supplied source report or cached verdict is accepted.
Proof: theory/nodal/ORGANIZED_JOINT_RESOURCES.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q

from .._exact_time import exact_or_represented_real as _exact
from ..mathematics._rational_interval import INTERVAL_METHOD, I, cos, pi_interval, sqrt
from ._sine_class_constitutive_acquisition import (
    _assess_constitutive_class_pair,
    _ConstitutiveFormedClassPair,
)
from ._sine_lyapunov import (
    _sine_lyapunov_coefficients,
    _sine_lyapunov_initial_upper,
    _sine_lyapunov_return_squared,
)

_FORMATION_TIME = Q(100)
_RADIUS = Q(1, 12)
_GAP, _RATE = Q(1, 5), Q(2)
_MAX_DECAY_POWER = 4096


@dataclass(frozen=True)
class _SineRecoveryResources:
    """Conditional source, contraction and actual endpoint radius evidence.

    Failed sufficient margins retain unavailability, not a necessary time
    obstruction. Neither support attachment nor observation is certified.
    The power cap bounds integer arithmetic, not physically permitted time.
    """

    source: _ConstitutiveFormedClassPair
    form_error_bound: Q
    phase_error_bound: Q
    quarter_dwell: Q
    half_law_dwell: Q
    quarter_handoff_time: Q
    original_handoff_time: Q
    endpoint_radius: Q
    decay_power: int
    class_one_radius_angle_bounds: I
    class_one_phase_hessian_lower_bound: Q
    transformed_stiffness_lower_bound: Q
    transformed_stiffness_upper_bound: Q
    lyapunov_position_lower_coefficient: Q
    lyapunov_position_upper_coefficient: Q
    lyapunov_initial_upper_bound: Q
    lyapunov_decay_rate: Q
    decay_exponent: Q
    decay_exponent_margin: Q
    exact_decay_upper_bound: Q | None
    returned_lyapunov_upper_bound: Q | None
    endpoint_form_norm_squared_upper_bound: Q | None
    endpoint_phase_norm_squared_upper_bound: Q | None
    endpoint_form_margin: Q | None
    endpoint_phase_margin: Q | None
    source_certified: bool
    decay_certified: bool
    recovery_certified: bool
    status: str
    unresolved_conditions: tuple[str, ...]
    quarter_formation_time: Q = 2 * _FORMATION_TIME
    radius: Q = _RADIUS
    class_one_phase_hessian_upper_bound: Q = Q(1)
    target_winding: int = 1
    clock: str = "tau=e*t; e=1023/1024; isolated quarter-law half-time=tau/2"
    arithmetic_method: str = INTERVAL_METHOD
    scope: tuple[str, ...] = (
        "fresh_original_class_one_ramp_and_all_zero_sum_preparation_errors",
        "sine_law_acquisition_precedes_own_class_one_curvature_bound",
        "uninterrupted_quarter_law_dwell_transforms_both_half_law_rows",
        "exact_rational_power_certificate_without_exponential_grid_loss",
        "endpoint_radius_bounds_reached_state_not_a_new_source_or_reset",
        "unavailable_sufficient_bound_is_not_a_necessary_resource_obstruction",
        "no_contact_observation_trajectory_event_occurrence_or_physical_bridge",
    )


def _assess_sine_recovery_resources(
    *, form_error_bound, phase_error_bound, quarter_dwell, endpoint_radius, decay_power
) -> _SineRecoveryResources:
    """Certify a declared recovery budget after fresh primitive acquisition.

    All scalars are admitted before constructing a source or interval.
    Preparation allowances and quarter-law dwell are nonnegative; the
    endpoint radius is positive. ``decay_power`` is an exact integer in
    [1, 4096]. Its bound is an arithmetic policy, not a limit on the dwell
    or on the decay exponent. Only class-one capture is consumed.
    """
    rx = _exact(form_error_bound, "form_error_bound")
    rt = _exact(phase_error_bound, "phase_error_bound")
    dwell = _exact(quarter_dwell, "quarter_dwell")
    q = _exact(endpoint_radius, "endpoint_radius")
    power = _exact(decay_power, "decay_power")
    if min(rx, rt, dwell) < 0:
        raise ValueError("preparation allowances and quarter dwell must be nonnegative")
    if q <= 0:
        raise ValueError("endpoint_radius must be strictly positive")
    if power.denominator != 1 or not 1 <= power <= _MAX_DECAY_POWER:
        raise ValueError("decay_power must be an exact integer in [1, 4096]")
    power = int(power)

    source = _assess_constitutive_class_pair(
        deformation=Q(0),
        scaled_time=_FORMATION_TIME,
        form_error_bound=rx,
        phase_error_bound=rt,
        radius=_RADIUS,
    )
    angle = 2 * pi_interval() / 9 + sqrt(I(2)) * _RADIUS
    curvature = cos(angle).lo
    eta = source.eta_bounds
    mu, maximum, lower, upper, rate = _sine_lyapunov_coefficients(
        eta_lower=eta.lo,
        eta_upper=eta.hi,
        cosine_lower=curvature,
        gap_lower=_GAP,
        rate_upper=_RATE,
        phase_hessian_upper=Q(1),
    )
    initial = _sine_lyapunov_initial_upper(
        eta_upper=eta.hi,
        rate_upper=_RATE,
        gap_lower=_GAP,
        position_upper=upper,
        form_norm_upper=source.endpoint_form_norm_upper_bounds[0],
        phase_norm_upper=source.endpoint_target_phase_norm_upper_bounds[0],
    )
    source_ok = (
        source.initial_zero_winding_certified
        and source.formation_certified_by_class[0]
        and source.source_budget_certified_by_class[0]
    )
    exponent = rate * dwell / 2
    exponent_margin = exponent - power
    decay_ok = exponent_margin >= 0
    decay = Q(1, 2**power) if decay_ok else None
    returned = x2 = y2 = form_margin = phase_margin = None
    if decay is not None:
        returned, x2, y2 = _sine_lyapunov_return_squared(
            initial_upper=initial,
            decay_upper=decay,
            eta_lower=eta.lo,
            gap_lower=_GAP,
            rate_upper=_RATE,
            position_lower=lower,
        )
        form_margin, phase_margin = q**2 - x2, q**2 - y2
    unresolved = tuple(
        name
        for condition, name in (
            (source_ok, "original_class_one_source_not_certified"),
            (decay_ok, "declared_dwell_decay_not_certified"),
            (
                form_margin is not None and form_margin > 0,
                "actual_form_handoff_radius_not_certified",
            ),
            (
                phase_margin is not None and phase_margin > 0,
                "actual_phase_handoff_radius_not_certified",
            ),
        )
        if not condition
    )
    return _SineRecoveryResources(
        source=source,
        form_error_bound=rx,
        phase_error_bound=rt,
        quarter_dwell=dwell,
        half_law_dwell=dwell / 2,
        quarter_handoff_time=2 * _FORMATION_TIME + dwell,
        original_handoff_time=(2 * _FORMATION_TIME + dwell) * Q(1024, 1023),
        endpoint_radius=q,
        decay_power=power,
        class_one_radius_angle_bounds=angle,
        class_one_phase_hessian_lower_bound=curvature,
        transformed_stiffness_lower_bound=mu,
        transformed_stiffness_upper_bound=maximum,
        lyapunov_position_lower_coefficient=lower,
        lyapunov_position_upper_coefficient=upper,
        lyapunov_initial_upper_bound=initial,
        lyapunov_decay_rate=rate,
        decay_exponent=exponent,
        decay_exponent_margin=exponent_margin,
        exact_decay_upper_bound=decay,
        returned_lyapunov_upper_bound=returned,
        endpoint_form_norm_squared_upper_bound=x2,
        endpoint_phase_norm_squared_upper_bound=y2,
        endpoint_form_margin=form_margin,
        endpoint_phase_margin=phase_margin,
        source_certified=source_ok,
        decay_certified=decay_ok,
        recovery_certified=not unresolved,
        status="certified_sine_recovery_resources" if not unresolved else "unavailable",
        unresolved_conditions=unresolved,
    )
