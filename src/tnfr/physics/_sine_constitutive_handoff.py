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

from ..mathematics._rational_interval import INTERVAL_METHOD
from ._sine_class_constitutive_acquisition import _ConstitutiveFormedClassPair
from ._sine_constitutive_recovery import (
    _DECAY_POWER,
    _DWELL,
    _ENDPOINT_RADIUS,
    _FORMATION_TIME,
    _RADIUS,
    _assess_constitutive_recovery,
)

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
    recovery = _assess_constitutive_recovery(
        deformation=deformation,
        form_error_bound=form_error_bound,
        phase_error_bound=phase_error_bound,
    )
    source, handoff = recovery.source, recovery.recovery_certified
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
    unresolved = list(recovery.unresolved_conditions)
    if not all(branch.identity_certified for branch in branches):
        unresolved.append("joined_four_branch_identity_not_certified")
    return _ConstitutiveHandoff(
        source=source,
        deformation=recovery.deformation,
        form_error_bound=recovery.form_error_bound,
        phase_error_bound=recovery.phase_error_bound,
        transformed_stiffness_lower_bound=recovery.transformed_stiffness_lower_bound,
        transformed_stiffness_upper_bound=recovery.transformed_stiffness_upper_bound,
        lyapunov_position_lower_coefficient=recovery.lyapunov_position_lower_coefficient,
        lyapunov_position_upper_coefficient=recovery.lyapunov_position_upper_coefficient,
        lyapunov_initial_upper_bound=recovery.lyapunov_initial_upper_bound,
        lyapunov_decay_rate=recovery.lyapunov_decay_rate,
        decay_exponent=recovery.decay_exponent,
        exact_decay_upper_bound=recovery.exact_decay_upper_bound,
        returned_lyapunov_upper_bound=recovery.returned_lyapunov_upper_bound,
        endpoint_form_norm_squared_upper_bound=recovery.endpoint_form_norm_squared_upper_bound,
        endpoint_phase_norm_squared_upper_bound=recovery.endpoint_phase_norm_squared_upper_bound,
        endpoint_form_margin=recovery.endpoint_form_margin,
        endpoint_phase_margin=recovery.endpoint_phase_margin,
        source_certified=recovery.source_certified,
        handoff_certified=handoff,
        attachment_work_bounds=contact_work,
        joined_excess_storage_upper_bound=joined_excess,
        joined_form_mean_bounds=mean,
        joined_phase_mean_bounds=mean,
        branches=tuple(branches),
        status="certified_constitutive_handoff" if not unresolved else "unavailable",
        unresolved_conditions=tuple(unresolved),
    )
