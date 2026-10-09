"""Shared own-law recovery of the original isolated class-one C9 source.

Every call re-admits primitive preparation and derives its capture and
uninterrupted fixed dwell. This owner contains no later support, contact,
input or response premise. Bounds concern the actual state, without a reset.
Proof: theory/nodal/ORGANIZED_PATTERN_CONSTITUTIVE_FREEDOM.md
#organized-pattern-carried-handoff.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q

from .._exact_time import exact_or_represented_real as _exact
from ..mathematics._rational_interval import INTERVAL_METHOD
from ._sine_class_constitutive_acquisition import (
    _assess_constitutive_class_pair,
    _ConstitutiveFormedClassPair,
)
from ._sine_class_constitutive_comparison import _deformation
from ._sine_lyapunov import (
    _sine_lyapunov_coefficients,
    _sine_lyapunov_initial_upper,
    _sine_lyapunov_return_squared,
)
from .reversible_eigenmode_reference import _MAX_RATIONAL_EXPONENT

_FORMATION_TIME = Q(100)
_RADIUS = Q(1, 12)
_DWELL = Q(10**13)
_ENDPOINT_RADIUS = Q(1, 10**70)
_DECAY_POWER = 512
_GAP, _RATE = Q(1, 5), Q(2)


@dataclass(frozen=True)
class _ConstitutiveRecovery:
    """Actual isolated class-one source and its own-law recovery certificate.

    Only the first source class is consumed. Later consumers must admit
    their own support changes and events; none is certified by this report.
    """

    source: _ConstitutiveFormedClassPair
    deformation: Q
    form_error_bound: Q
    phase_error_bound: Q
    transformed_stiffness_lower_bound: Q | None
    transformed_stiffness_upper_bound: Q
    lyapunov_position_lower_coefficient: Q | None
    lyapunov_position_upper_coefficient: Q
    lyapunov_initial_upper_bound: Q
    lyapunov_decay_rate: Q | None
    decay_exponent: Q | None
    exact_decay_upper_bound: Q | None
    returned_lyapunov_upper_bound: Q | None
    endpoint_form_norm_squared_upper_bound: Q | None
    endpoint_phase_norm_squared_upper_bound: Q | None
    endpoint_form_margin: Q | None
    endpoint_phase_margin: Q | None
    source_certified: bool
    recovery_certified: bool
    status: str
    unresolved_conditions: tuple[str, ...]
    formation_time: Q = _FORMATION_TIME
    relaxation_duration: Q = _DWELL
    scaled_handoff_time: Q = _FORMATION_TIME + _DWELL
    original_handoff_time: Q = (_FORMATION_TIME + _DWELL) * Q(1024, 1023)
    radius: Q = _RADIUS
    endpoint_radius: Q = _ENDPOINT_RADIUS
    decay_power: int = _DECAY_POWER
    target_winding: int = 1
    clock: str = "tau=e*t; e=1023/1024"
    arithmetic_method: str = INTERVAL_METHOD
    scope: tuple[str, ...] = (
        "fresh_own_law_class_one_source_and_original_zero_sum_errors",
        "own_potential_Hessian_bounds_in_complete_nonlinear_modified_energy",
        "fixed_unforced_dwell_and_exact_decay_below_the_interval_grid",
        "endpoint_radius_bounds_reached_state_not_a_new_source_or_reset",
        "no_contact_support_input_or_response_premise",
        "unavailable_is_not_failed_recovery_or_an_obstruction",
        "no_trajectory_event_occurrence_or_physical_identity_claim",
    )


def _assess_constitutive_recovery(
    *, deformation, form_error_bound, phase_error_bound
) -> _ConstitutiveRecovery:
    """Rebuild isolated capture and recovery before any application event.

    All raw scalars are admitted before source or interval construction.
    The proved fine-law family is [0,1]; other nonnegative coefficients
    retain explicit unavailability. No incoming report or altered dwell
    can replace the primitive preparation and fixed recovery premises.
    """
    coefficient = _deformation(deformation)
    rx = _exact(form_error_bound, "form_error_bound")
    rt = _exact(phase_error_bound, "phase_error_bound")
    if rx < 0 or rt < 0:
        raise ValueError("preparation error bounds must be nonnegative")
    source = _assess_constitutive_class_pair(
        deformation=coefficient,
        scaled_time=_FORMATION_TIME,
        form_error_bound=rx,
        phase_error_bound=rt,
        radius=_RADIUS,
    )
    eta = source.eta_bounds
    mu, maximum, lower, upper, rate = _sine_lyapunov_coefficients(
        eta_lower=eta.lo,
        eta_upper=eta.hi,
        cosine_lower=source.phase_hessian_lower_bound,
        gap_lower=_GAP,
        rate_upper=_RATE,
        phase_hessian_upper=source.phase_hessian_upper_bound,
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
        coefficient <= 1
        and source.initial_zero_winding_certified
        and source.formation_certified_by_class[0]
        and source.source_budget_certified_by_class[0]
    )
    exponent = rate * _DWELL if rate is not None else None
    decay_ok = (
        exponent is not None and _DECAY_POWER <= exponent <= _MAX_RATIONAL_EXPONENT
    )
    decay = Q(1, 2**_DECAY_POWER) if decay_ok else None
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
        form_margin = _ENDPOINT_RADIUS**2 - x2
        phase_margin = _ENDPOINT_RADIUS**2 - y2
    recovery = (
        source_ok and form_margin is not None and form_margin > 0 and phase_margin > 0
    )
    unresolved = []
    if coefficient > 1:
        unresolved.append("deformation_outside_proved_family")
    if not source_ok:
        unresolved.append("original_class_one_source_not_certified")
    if not decay_ok:
        unresolved.append("fixed_dwell_decay_not_certified")
    if form_margin is None or form_margin <= 0:
        unresolved.append("actual_form_handoff_radius_not_certified")
    if phase_margin is None or phase_margin <= 0:
        unresolved.append("actual_phase_handoff_radius_not_certified")
    return _ConstitutiveRecovery(
        source=source,
        deformation=coefficient,
        form_error_bound=rx,
        phase_error_bound=rt,
        transformed_stiffness_lower_bound=mu,
        transformed_stiffness_upper_bound=maximum,
        lyapunov_position_lower_coefficient=lower,
        lyapunov_position_upper_coefficient=upper,
        lyapunov_initial_upper_bound=initial,
        lyapunov_decay_rate=rate,
        decay_exponent=exponent,
        exact_decay_upper_bound=decay,
        returned_lyapunov_upper_bound=returned,
        endpoint_form_norm_squared_upper_bound=x2,
        endpoint_phase_norm_squared_upper_bound=y2,
        endpoint_form_margin=form_margin,
        endpoint_phase_margin=phase_margin,
        source_certified=source_ok,
        recovery_certified=recovery,
        status="certified_constitutive_recovery" if not unresolved else "unavailable",
        unresolved_conditions=tuple(unresolved),
    )
