"""Own-law recovery and supplied three-C9 contact of actual acquired sources.

Every call rebuilds primitive class-one acquisition under the supplied member
of the matched-potential family. Fixed analytic recovery bounds then retain
the actual state and its original mean constraints through contact and four
form-input branches. No trajectory, reset, report admission or event execution
occurs. Conditional branch arithmetic is not certified when handoff fails.
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
_AMPLITUDE = Q(7, 10000)
_JOINED_BARRIER = Q(1, 486000)


@dataclass(frozen=True)
class _ConstitutiveHandoffBranch:
    """Conditional carried-state work and identity for one prescribed input."""

    name: str
    donor_amplitude: Q
    receiver_amplitude: Q
    input_work_bounds: tuple[Q, Q]
    relative_norm_squared_upper_bound: Q
    excess_storage_upper_bound: Q
    radius_margin: Q
    storage_margin: Q
    form_mean_bounds: tuple[Q, Q]
    phase_mean_bounds: tuple[Q, Q]
    identity_certified: bool


@dataclass(frozen=True)
class _ConstitutiveHandoff:
    """Fixed class-one preparation, own-law dwell and supplied contact bounds.

    The source's first class is consumed; its second class is not a premise.
    Norm and event bounds cover each actually reached preparation member.
    Prospective contact/branch fields retain their formulas on unavailable
    reports, but their premise is the separately reported certified handoff.
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
    handoff_certified: bool
    attachment_work_bounds: tuple[Q, Q]
    joined_excess_storage_upper_bound: Q
    joined_form_mean_bounds: tuple[Q, Q]
    joined_phase_mean_bounds: tuple[Q, Q]
    branches: tuple[_ConstitutiveHandoffBranch, ...]
    status: str
    unresolved_conditions: tuple[str, ...]
    formation_time: Q = _FORMATION_TIME
    relaxation_duration: Q = _DWELL
    scaled_handoff_time: Q = _FORMATION_TIME + _DWELL
    original_handoff_time: Q = (_FORMATION_TIME + _DWELL) * Q(1024, 1023)
    radius: Q = _RADIUS
    endpoint_radius: Q = _ENDPOINT_RADIUS
    decay_power: int = _DECAY_POWER
    joined_storage_barrier: Q = _JOINED_BARRIER
    target_windings: tuple[int, int, int] = (1, 1, 1)
    bridges: tuple[tuple[int, int], ...] = ((4, 13), (13, 22))
    central_degrees: tuple[int, int, int] = (3, 4, 3)
    degree_sum: int = 58
    clock: str = "tau=e*t; e=1023/1024"
    arithmetic_method: str = INTERVAL_METHOD
    scope: tuple[str, ...] = (
        "fresh_own_law_class_one_source_and_original_zero_sum_errors",
        "all_three_actual_components_keep_their_full_form_and_phase_state",
        "own_potential_Hessian_bounds_in_complete_nonlinear_modified_energy",
        "fixed_unforced_dwell_and_exact_decay_below_the_interval_grid",
        "endpoint_radius_bounds_reached_state_not_a_new_source_or_reset",
        "supplied_attachment_changes_degrees_without_changing_state",
        "own_law_attachment_work_and_actual_pre_input_form_work",
        "all_four_inputs_share_one_actual_pre_input_state_within_each_law",
        "retained_weighted_means_and_whole_future_joined_identity",
        "unavailable_is_not_failed_recovery_or_an_obstruction",
        "no_trajectory_event_occurrence_finite_response_or_physical_identity_claim",
    )


def _assess_constitutive_handoff(
    *, deformation, form_error_bound, phase_error_bound
) -> _ConstitutiveHandoff:
    """Rebuild fixed own-law acquisition, uninterrupted dwell and contact.

    All raw scalars are admitted before source or interval construction.
    Negative/nonfinite/Boolean inputs reject. Valid coefficients outside the
    proved family [0,1], absent capture, insufficient recovery and nonpositive
    identity margins yield explicit unavailability rather than a reset.
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
    handoff = (
        source_ok and form_margin is not None and form_margin > 0 and phase_margin > 0
    )
    eps, u = _ENDPOINT_RADIUS, source.phase_hessian_upper_bound
    contact_work = Q(0), 4 * (1 + u) * eps**2
    joined_excess = 10 * (1 + u) * eps**2
    mean = -4 * eps / 58, 4 * eps / 58
    branches = []
    for name, a, b in (
        ("neither", Q(0), Q(0)),
        ("donor", _AMPLITUDE, Q(0)),
        ("receiver", Q(0), _AMPLITUDE),
        ("both", _AMPLITUDE, _AMPLITUDE),
    ):
        nominal_work, work_error = Q(3, 2) * (a**2 + b**2), 6 * (a + b) * eps
        work = nominal_work - work_error, nominal_work + work_error
        norm = 6 * eps**2 + 2 * (a + b) * eps + a**2 + b**2 - (a + b) ** 2 / 27
        excess = joined_excess + work[1]
        radius_margin, storage_margin = _RADIUS**2 - norm, _JOINED_BARRIER - excess
        branches.append(
            _ConstitutiveHandoffBranch(
                name=name,
                donor_amplitude=a,
                receiver_amplitude=b,
                input_work_bounds=work,
                relative_norm_squared_upper_bound=norm,
                excess_storage_upper_bound=excess,
                radius_margin=radius_margin,
                storage_margin=storage_margin,
                form_mean_bounds=tuple(value + 3 * (a + b) / 58 for value in mean),
                phase_mean_bounds=mean,
                identity_certified=handoff and radius_margin > 0 and storage_margin > 0,
            )
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
    if not all(branch.identity_certified for branch in branches):
        unresolved.append("joined_four_branch_identity_not_certified")
    return _ConstitutiveHandoff(
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
        handoff_certified=handoff,
        attachment_work_bounds=contact_work,
        joined_excess_storage_upper_bound=joined_excess,
        joined_form_mean_bounds=mean,
        joined_phase_mean_bounds=mean,
        branches=tuple(branches),
        status="certified_constitutive_handoff" if not unresolved else "unavailable",
        unresolved_conditions=tuple(unresolved),
    )
