"""One supplied internal intervention and an acquired parent's restoring response.

ORGANIZED_PARENT_RESTORING_RESPONSE.md owns the frozen protocol and proof.
Every assessment rebuilds the original acquisition. The event and eight-state
reference remain detached; no graph, trajectory or event selector is installed.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q

from ..mathematics._rational_interval import I, cos, pi_interval
from ._organized_parent_acquisition import (
    _assess_organized_parent_acquisition,
    _OrganizedParentAcquisition,
)
from ._repeated_organized_composition import _vector
from ._sine_preparation import _prepared_heat_error_radii
from .reversible_eigenmode_reference import _negative_exp_bounds

_KICK = Q(1, 4)
_READOUT_DURATION, _RECOVERY_DURATION = Q(1), Q(4)
_WORK_BUDGET = Q(3)
_PREDICTION_BUDGET = Q(1, 10**7)
_MINIMUM_CONTRAST = Q(1, 10**6)
_INNER_CAP, _OUTER_CAP = Q(4, 15), Q(1, 500)
_INNER_RATE, _MIXED_RATE = Q(19, 20), Q(19, 10)
_GRADIENT_DELAY = Q(3, 2 * 10**6)
_EVENT = tuple(
    (2 * (node // 9) - 1) * _KICK if node < 18 else Q(0) for node in range(36)
)


def _apply_parent_registration_intervention(*, form, phase_offsets):
    """Apply the fixed simultaneous lifted-phase map to detached primitives.

    Both complete vectors are admitted before constructing the result. Forms,
    each child's internal gaps and both parent means are preserved exactly.
    This map neither executes a registered operator nor certifies event work,
    source history, occurrence or subsequent recovery.
    """
    values = _vector(form, "form", 36)
    offsets = _vector(phase_offsets, "phase_offsets", 36)
    return values, tuple(value + change for value, change in zip(offsets, _EVENT))


@dataclass(frozen=True)
class _OrganizedParentRestoringResponse:
    """Conditional arithmetic and separately admitted whole-source decisions.

    Parent phase residuals subtract both the winding carrier ``psi_j`` and
    the actual parent's mean phase offset; form residuals use its actual mean.
    """

    form_error_bounds: tuple[Q, ...]
    phase_error_bounds: tuple[Q, ...]
    acquisition: _OrganizedParentAcquisition
    full_phase_error_upper_bound: Q
    full_form_error_upper_bound: Q
    event_nominal_work_bounds: I
    event_work_error_upper_bound: Q
    event_work_residual_bounds: tuple[Q, Q]
    event_work_bounds: I
    post_event_disruption_margin: Q
    post_event_inner_mode_initial_bounds: I
    post_event_outer_mode_initial_bounds: I
    inner_recovery_decay_bounds: I
    mixed_recovery_decay_bounds: I
    recovered_inner_mode_upper_bound: Q
    recovered_mixed_mode_upper_bound: Q
    parent_gap_at_recovery_upper_bound: Q
    parent_phase_residual_at_recovery_upper_bound: Q
    parent_form_residual_at_recovery_upper_bound: Q
    child_positive_gap_margin: Q
    child_acute_gap_margin: Q
    interaction_delay_integral_lower_bound: Q
    gradient_phase_delay_lower_bound: Q
    per_history_gradient_phase_error_upper_bound: Q
    paired_outer_gap_contrast_lower_bound: Q
    eight_coordinate_phase_prediction_error_upper_bound: Q
    elapsed_scaled_readout_duration_bounds: I
    elapsed_scaled_recovery_duration_bounds: I
    elapsed_original_readout_duration_bounds: I
    elapsed_original_recovery_duration_bounds: I
    sufficient_margins: tuple[tuple[str, Q], ...]
    source_certified: bool
    conditional_response_certified: bool
    disruption_certified: bool
    work_certified: bool
    child_identities_retained: bool
    recovery_certified: bool
    prediction_certified: bool
    contrast_certified: bool
    admitted: bool
    status: str
    unresolved_conditions: tuple[str, ...]
    event_phase_increments: tuple[Q, ...] = _EVENT
    parent_observation_jump: tuple[Q, ...] = (Q(0),) * 4
    event_slow_time: Q = Q(4)
    readout_slow_time: Q = Q(5)
    recovery_slow_time: Q = Q(8)
    post_event_readout_duration: Q = _READOUT_DURATION
    post_event_recovery_duration: Q = _RECOVERY_DURATION
    post_event_mixed_mode_initial: Q = _KICK
    event_work_budget: Q = _WORK_BUDGET
    eight_coordinate_prediction_error_budget: Q = _PREDICTION_BUDGET
    minimum_paired_outer_gap_contrast: Q = _MINIMUM_CONTRAST
    protocol_revision: str = "37add507"
    clock: str = "tau=(1023/1024)*t; s=gamma^2*tau/2; post-event h=s-4"
    scope: tuple[str, ...] = (
        "fresh_original_acquisition_from_all_eight_primitive_allowances",
        "same_actual_slow_time_four_state_for_intervened_and_unintervened_histories",
        "simultaneous_parent_zero_child_phase_shifts_without_form_or_support_change",
        "event_preserves_child_windings_and_all_four_parent_observations_exactly",
        "signed_complete_event_work_supplied_externally_not_recycled_loss",
        "same_jump_on_actual_and_reference_states_preserves_their_full_difference",
        "global_heat_comparison_keeps_all_original_sixty_four_fine_source_directions",
        "post_event_reference_uses_three_modes_not_a_reset_two_pendulum_family",
        "eight_child_mean_predictor_initialized_from_actual_post_event_means",
        "four_parent_means_alone_omit_hidden_registration_and_intervention_history",
        "finite_interaction_contrast_not_only_an_instantaneous_current_difference",
        "child_retention_through_slow_time_eight_and_parent_return_at_that_deadline",
        "no_all_future_return_or_generic_residual_ball_certificate",
        "no_trajectory_graph_mutation_registered_operator_or_autonomous_occurrence",
        "no_physical_binding_scale_selection_or_laboratory_clock_identification",
    )


def _assess_organized_parent_restoring_response(
    *, form_error_bounds, phase_error_bounds
) -> _OrganizedParentRestoringResponse:
    """Rebuild the acquired source and assess the single frozen intervention.

    No incoming acquisition report, event, horizon or readout can replace the
    fixed protocol. Every primitive is admitted before the first source call;
    failed sufficient margins remain unavailable rather than proving failure.
    """
    rx = _vector(form_error_bounds, "form_error_bounds", 4)
    rt = _vector(phase_error_bounds, "phase_error_bounds", 4)
    if min(*rx, *rt) < 0:
        raise ValueError("original preparation errors must be nonnegative")
    acquisition = _assess_organized_parent_acquisition(
        form_error_bounds=rx, phase_error_bounds=rt
    )
    source_ok = (
        acquisition.sources_certified
        and acquisition.entry_certified
        and acquisition.child_identities_retained
    )
    geometry = acquisition.geometry
    gamma, eta = geometry.gamma_bounds, geometry.gamma_bounds**2
    radius = acquisition.recovered_child_norm_bound
    form_error, phase_error = _prepared_heat_error_radii(
        initial_form_error=2 * radius,
        initial_phase_error=2 * radius,
        gamma_upper=gamma.hi,
        phase_growth_upper=Q(3**32),
        form_convolution_factor=Q(24),
    )
    pre_error = acquisition.full_phase_error_upper_bound
    inner = acquisition.nominal_inner_gap_at_entry_bounds
    outer = acquisition.nominal_outer_gap_at_entry_bounds
    post_inner = inner + _KICK
    a = 1 + 2 * eta.hi
    inner_decay = I(*_negative_exp_bounds(_INNER_RATE * (_RECOVERY_DURATION - eta.hi)))
    mixed_decay = I(
        *_negative_exp_bounds(_MIXED_RATE * (_RECOVERY_DURATION - eta.hi / 2))
    )
    inner_return = post_inner.hi * inner_decay.hi
    mixed_return = _KICK * mixed_decay.hi
    returned_sum = inner_return + mixed_return
    parent_gap = returned_sum + 2 * phase_error
    parent_phase = returned_sum / 2 + 2 * phase_error
    parent_form = gamma.hi * a * returned_sum / 2 + 2 * form_error
    disruption = inner.lo + 2 * _KICK - 2 * pre_error - acquisition.parent_identity_gap
    child_positive = geometry.alpha_bounds.lo - 2 * phase_error
    child_acute = pi_interval().lo / 2 - geometry.alpha_bounds.hi - 2 * phase_error

    # The acquired fine errors need not have zero child means. This event
    # therefore retains a signed linear error, not the attachment's q^2 cost.
    work_nominal = 9 * (cos(inner) - cos(inner + 2 * _KICK)) + 18 * cos(outer) * (
        1 - cos(I(_KICK))
    )
    work_error = 18 * pre_error
    work_residual = (-work_error, work_error)
    work = work_nominal + I(*work_residual)

    decay_one = _negative_exp_bounds(Q(1))[0]
    decay_four = _negative_exp_bounds(Q(4))[1]
    delay_integral = (1 - decay_four) / (192 * a)
    phase_slope = Q(99, 100) * outer.lo * decay_one
    gradient_delay = delay_integral * phase_slope
    gradient_error = eta.hi / 10
    contrast = _GRADIENT_DELAY - 2 * gradient_error - 4 * phase_error
    predictor_error = 2 * phase_error + 18 * (phase_error + gamma.hi * form_error)

    mode_margins = (
        ("post_event_inner_mode_domain", _INNER_CAP - post_inner.hi),
        ("post_event_outer_mode_domain", _OUTER_CAP - outer.hi),
        ("small_reciprocal_parameter", Q(1, 10**7) - eta.hi),
        ("invariant_velocity_cones", a - 1 - eta.hi * a**2),
        (
            "inner_mode_restoring_force",
            (1 - _INNER_CAP**2 / 6) * (1 - _KICK**2 / 2) - _INNER_RATE,
        ),
        (
            "mixed_mode_restoring_force",
            (2 - (_INNER_CAP**2 + _OUTER_CAP**2) / 2) * (1 - _KICK**2 / 6)
            - _MIXED_RATE,
        ),
        ("child_positive_winding_gap", child_positive),
        ("child_acute_winding_gap", child_acute),
    )
    event_margins = (
        ("actual_parent_registration_disrupted", disruption),
        ("positive_supplied_event_work", work.lo),
        ("supplied_event_work_budget", _WORK_BUDGET - work.hi),
    )
    recovery_margin = acquisition.parent_identity_gap - parent_gap
    contrast_margins = (
        ("outer_initial_velocity_lag", Q(1, 400) - a * _OUTER_CAP),
        (
            "outer_current_time_derivative",
            Q(1, 400) - a * _OUTER_CAP * (1 + 2 * _KICK**2),
        ),
        ("strict_hidden_mode_delay_integral", delay_integral - Q(1, 200)),
        ("strict_gradient_phase_slope", phase_slope - Q(3, 10000)),
        ("strict_gradient_phase_delay", gradient_delay - _GRADIENT_DELAY),
        ("resolved_paired_outer_gap_contrast", contrast - _MINIMUM_CONTRAST),
    )
    prediction_margin = _PREDICTION_BUDGET - predictor_error
    margins = (
        mode_margins
        + event_margins
        + (("parent_identity_at_recovery", recovery_margin),)
        + contrast_margins
        + (("actual_eight_coordinate_prediction", prediction_margin),)
    )
    mode_ok = all(value > 0 for _, value in mode_margins)
    conditional = all(value > 0 for _, value in margins)
    admitted = source_ok and conditional
    source_conditions = (
        ("original_children_not_certified", acquisition.sources_certified),
        ("original_parent_entry_not_certified", acquisition.entry_certified),
        (
            "original_child_identities_not_certified",
            acquisition.child_identities_retained,
        ),
    )
    unresolved = tuple(
        name for name, passed in source_conditions if not passed
    ) + tuple(name for name, value in margins if value <= 0)
    tau_readout = 2 * _READOUT_DURATION / eta
    tau_recovery = 2 * _RECOVERY_DURATION / eta
    return _OrganizedParentRestoringResponse(
        form_error_bounds=rx,
        phase_error_bounds=rt,
        acquisition=acquisition,
        full_phase_error_upper_bound=phase_error,
        full_form_error_upper_bound=form_error,
        event_nominal_work_bounds=work_nominal,
        event_work_error_upper_bound=work_error,
        event_work_residual_bounds=work_residual,
        event_work_bounds=work,
        post_event_disruption_margin=disruption,
        post_event_inner_mode_initial_bounds=post_inner,
        post_event_outer_mode_initial_bounds=outer,
        inner_recovery_decay_bounds=inner_decay,
        mixed_recovery_decay_bounds=mixed_decay,
        recovered_inner_mode_upper_bound=inner_return,
        recovered_mixed_mode_upper_bound=mixed_return,
        parent_gap_at_recovery_upper_bound=parent_gap,
        parent_phase_residual_at_recovery_upper_bound=parent_phase,
        parent_form_residual_at_recovery_upper_bound=parent_form,
        child_positive_gap_margin=child_positive,
        child_acute_gap_margin=child_acute,
        interaction_delay_integral_lower_bound=delay_integral,
        gradient_phase_delay_lower_bound=gradient_delay,
        per_history_gradient_phase_error_upper_bound=gradient_error,
        paired_outer_gap_contrast_lower_bound=contrast,
        eight_coordinate_phase_prediction_error_upper_bound=predictor_error,
        elapsed_scaled_readout_duration_bounds=tau_readout,
        elapsed_scaled_recovery_duration_bounds=tau_recovery,
        elapsed_original_readout_duration_bounds=tau_readout / Q(1023, 1024),
        elapsed_original_recovery_duration_bounds=tau_recovery / Q(1023, 1024),
        sufficient_margins=margins,
        source_certified=source_ok,
        conditional_response_certified=conditional,
        disruption_certified=source_ok and disruption > 0,
        work_certified=source_ok and work.lo > 0 and work.hi < _WORK_BUDGET,
        child_identities_retained=source_ok and child_positive > 0 and child_acute > 0,
        recovery_certified=source_ok and mode_ok and recovery_margin > 0,
        prediction_certified=source_ok and mode_ok and prediction_margin > 0,
        contrast_certified=source_ok
        and mode_ok
        and all(value > 0 for _, value in contrast_margins),
        admitted=admitted,
        status=(
            "certified_organized_parent_restoring_response"
            if admitted
            else "unavailable"
        ),
        unresolved_conditions=unresolved,
    )
