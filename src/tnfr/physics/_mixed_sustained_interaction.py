"""Finite sustained validity using actual centered form dissipation.

The complete and effective laws are unchanged. A gap-filtered comparison
retains phase displacement, collective error and hidden storage over one
fixed window. Its positive comparison mode also records why this estimate
does not establish all-time attraction of the actual organized source.
"""

from dataclasses import dataclass
from fractions import Fraction as Q

from .._exact_time import exact_or_represented_real as _exact
from ..mathematics._rational_interval import I, pi_interval
from ._mixed_effective_interaction import (
    _assess_mixed_effective_interaction,
    _MixedEffectiveInteraction,
)
from ._mixed_reciprocal_partner import _nonnegative, _nonnegative_pair
from ._mixed_registered_phase_probe import _derive_registered_phase_probe_geometry
from ._mixed_resource_transfer_obstruction import _original_source_matches

_DELTA = Q(1, 1024)
_WEIGHT, _ETA, _MATURE = Q(2, 10**60), Q(2, 10**69), Q(1, 2**28)
_SECTION = 3 * _DELTA / 16
_SECTION_ERROR = Q(3, 2) * (14 * _ETA + 2 * 10**10 * _WEIGHT)
_GL, _G = Q(1, 3300), Q(1, 3000)
_ACTIVATION, _SWITCH = Q(2010000), Q(3375000)
_SEGMENT, _PREVIOUS, _DURATION = Q(8), Q(16), Q(4096)
_DECAY = Q(1, 2**16)
_NOISE = Q(1, 10**9)
_PAIRS = tuple((j, 18 + j) for j in range(18))


@dataclass(frozen=True)
class _DampedCenteredEnvelope:
    """Candidate finite bounds for the declared two-inequality comparison."""

    initial_form_norm: Q
    initial_phase_norm: Q
    duration: Q
    feedback_denominator: Q
    uniform_form_norm_upper_bound: Q
    uniform_phase_norm_upper_bound: Q
    form_norm_integral_upper_bound: Q
    phase_displacement_upper_bound: Q
    endpoint_form_norm_upper_bound: Q
    scalar_decay_upper_bound: Q


def _damped_centered_envelope(*, initial_form_norm, initial_phase_norm, duration):
    """Filter X' <= -X/3+4gY and Y' <= 4gX on an admitted finite window.

    This scalar calculation does not establish the actual source, matrix gap,
    Dini inequalities or phase displacement law. Those are caller premises.
    Every raw input is normalized before domain checks or arithmetic.
    """
    form = _exact(initial_form_norm, "initial_form_norm")
    phase = _exact(initial_phase_norm, "initial_phase_norm")
    time = _exact(duration, "duration")
    if min(form, phase, time) < 0:
        raise ValueError("initial norms and duration must be nonnegative")
    denominator = 1 - 48 * _G**2 * time
    if denominator <= 0:
        raise ValueError("the finite comparison feedback denominator must be positive")
    phase_bound = (phase + 12 * _G * form) / denominator
    form_bound = max(form, 12 * _G * phase_bound)
    integral = 3 * form + 12 * _G * time * phase_bound
    displacement = 4 * _G * integral
    decay = 3 / (3 + time)
    endpoint = form * decay + 12 * _G * phase_bound
    return _DampedCenteredEnvelope(
        initial_form_norm=form,
        initial_phase_norm=phase,
        duration=time,
        feedback_denominator=denominator,
        uniform_form_norm_upper_bound=form_bound,
        uniform_phase_norm_upper_bound=phase_bound,
        form_norm_integral_upper_bound=integral,
        phase_displacement_upper_bound=displacement,
        endpoint_form_norm_upper_bound=endpoint,
        scalar_decay_upper_bound=decay,
    )


@dataclass(frozen=True)
class _MixedSustainedInteraction:
    """A finite controlled prediction and an explicit comparison limitation."""

    form_error_bounds: tuple[Q, Q]
    phase_error_bounds: tuple[Q, Q]
    registered_observation_error_bound: Q
    aligned_observation_error_bound: Q
    source: _MixedEffectiveInteraction
    source_association_certified: bool
    registration_covariance_certified: bool
    weighted_gap_certified: bool
    damped_centered_bound_certified: bool
    centered_identity_certified: bool
    phase_branch_certified: bool
    phase_moment_remainder_certified: bool
    switching_work_certified: bool
    effective_balance_certified: bool
    form_prediction_certified: bool
    phase_prediction_certified: bool
    comparison_limitation_certified: bool
    observation_allowances_admitted: bool
    registered_record_prediction_certified: bool
    aligned_record_prediction_certified: bool
    gamma_bounds: I
    coupling_bounds: I
    nominal_q0_bounds: tuple[Q, Q]
    slow_amplitude_bounds: tuple[Q, Q]
    inherited_output_error_upper_bound: Q
    initial_relative_phase_error_upper_bound: Q
    centered_entry_norm_upper_bound: Q
    centered_envelope: _DampedCenteredEnvelope
    old_joint_growth_denominator: Q
    marked_gap_upper_bound: Q
    registration_margin: Q
    aggregate_absolute_upper_bound: Q
    relative_phase_deviation_upper_bound: Q
    boundary_branch_margin: Q
    phase_moment_remainder_upper_bound: Q
    prediction_feedback_denominator: Q
    prediction_decay_upper_bound: Q
    uniform_form_error_upper_bound: Q
    phase_error_upper_bound: Q
    endpoint_form_error_upper_bound: Q
    form_prediction_margin: Q
    phase_prediction_margin: Q
    phase_switching_work_allowance: Q
    aligned_storage_jump_bounds: tuple[Q, Q]
    registered_storage_jump_bounds: tuple[Q, Q]
    storage_remainder_upper_bound: Q
    unresolved_exchange_integral_upper_bound: Q
    storage_prediction_error_upper_bound: Q
    registered_record_error_upper_bound: Q
    aligned_record_error_upper_bound: Q
    comparison_matrix: tuple[tuple[Q, Q], tuple[Q, Q]]
    comparison_determinant: Q
    comparison_positive_witness: tuple[Q, Q]
    comparison_positive_quadratic_value: Q
    certified: bool
    status: str
    unresolved_conditions: tuple[str, ...]
    initial_registrations_turns: tuple[Q, Q] = (Q(0), Q(1, 4))
    initial_reduced_form: str = "same source-correlated Q_s(3375000)"
    reduced_state_coordinates: tuple[str, str] = ("Q", "psi")
    reduced_state_dimension: int = 2
    full_state_dimension: int = 72
    physical_capacities: tuple[Q, ...] = (Q(1),) * 36
    full_contact_pairs: tuple[tuple[int, int], ...] = _PAIRS
    internal_divisor: Q = Q(9)
    bare_gap_lower_bound: Q = Q(4)
    minimum_internal_conductance: Q = Q(3, 4)
    weighted_gap_lower_bound: Q = Q(3)
    transport_norm_upper_bound: Q = Q(36)
    phase_force_lipschitz_upper_bound: Q = Q(36)
    form_damping_lower_bound: Q = Q(1, 3)
    centered_coupling_upper_bound: Q = 4 * _G
    activation_time: Q = _ACTIVATION
    switch_age: Q = _SWITCH
    switch_time: Q = _ACTIVATION + _SWITCH
    previous_certified_duration: Q = _PREVIOUS
    post_duration: Q = _DURATION
    total_duration: Q = _ACTIVATION + _SWITCH + _DURATION
    original_clock_horizon: Q = (_ACTIVATION + _SWITCH + _DURATION) * Q(1024, 1023)
    horizon_extension_factor: Q = _DURATION / _PREVIOUS
    original_donor_form_total: Q = Q(0)
    whole_form_mean: Q = Q(0)
    registered_observation_error_cap: Q = _NOISE
    aligned_observation_error_cap: Q = _NOISE
    prediction_tolerance: Q = _NOISE
    phase_prediction_tolerance: Q = _NOISE
    effective_form_equation: str = "Qhat_prime=-2*Qhat-18*gamma*sin(psihat)"
    effective_phase_equation: str = "psihat_prime=2*gamma*Qhat/9"
    actual_form_equation: str = "Q_prime=-2*Q-18*gamma*sin(psi)-gamma*R"
    actual_phase_equation: str = "psi_prime=2*gamma*Q/9"
    moment_decomposition: str = "M=18*sin(psi)+R; abs(R)<=norm(y_centered)**2"
    effective_storage_rate: str = "-4*Qhat**2"
    projected_storage_rate: str = "-4*Q**2-2*gamma*Q*R"
    unresolved_storage_rate: str = "-norm((H+9G)*x_centered)**2/9+2*gamma*Q*R"
    full_storage_rate: str = "-norm((H+9G)*x)**2/9"
    common_storage_reference: str = "same unshifted Theta in Uplus(Theta)"
    same_complete_and_effective_laws: bool = True
    whole_history_covariance_retained: bool = True
    source_correlated_initialization_retained: bool = True
    actual_weighted_gap_used: bool = True
    unequal_internal_blocks_assumed_commuting: bool = False
    boundary_convexity_assumed: bool = False
    centered_phase_displacement_priced: bool = True
    phase_only_moment_bound_used: bool = True
    intrinsic_phase_accuracy_is_phase_reading: bool = False
    independent_record_errors: bool = True
    observation_error_cancellation_assumed: bool = False
    original_attachment_work_retained_once: bool = True
    reference_shift_used_to_erase_switch_work: bool = False
    state_reset_performed: bool = False
    phase_clamped: bool = False
    independent_marginal_state_reselection: bool = False
    original_source_reacquired: bool = False
    original_source_replayed: bool = False
    original_source_trajectory_evaluated: bool = False
    reduced_trajectory_evaluated: bool = False
    exact_actual_two_coordinate_closure: bool = False
    all_future_identity_claimed: bool = False
    actual_asymptotic_attraction_certified: bool = False
    actual_instability_claimed: bool = False
    optimal_validity_horizon_identified: bool = False
    comparison_cutoff_is_actual_lifetime: bool = False
    endpoint_envelope_is_actual_norm_reduction: bool = False
    unresolved_storage_assumed_nonnegative: bool = False
    earlier_short_window_results_invalidated: bool = False
    minimal_intrinsic_macrostate_claimed: bool = False
    autonomous_pair_formation_claimed: bool = False
    runtime_event_executed: bool = False
    operator_occurrence_derived: bool = False
    exponentials_evaluated: bool = False
    spectral_roots_evaluated: bool = False
    newly_observed_aggregates: int = 0
    newly_observed_phases: int = 0
    required_endpoint_records: int = 2
    numerical_response_allowance: Q = Q(0)
    physical_property_identified: bool = False
    empirical_validation: bool = False
    clock: str = "tau=(1023/1024)*t; sustained unchanged actual and effective rows"


def _assess_mixed_sustained_interaction(
    *,
    form_error_bounds,
    phase_error_bounds,
    registered_observation_error_bound,
    aligned_observation_error_bound,
):
    """Re-admit the original source before the fixed sustained comparison.

    Previous identity, short-window prediction and global verdicts do not
    certify this longer window. Initial source bounds and registration are
    rebuilt; no previous endpoint replaces the original attachment state.
    """
    forms = _nonnegative_pair(form_error_bounds, "form_error_bounds")
    phases = _nonnegative_pair(phase_error_bounds, "phase_error_bounds")
    registered_noise = _nonnegative(
        registered_observation_error_bound, "registered_observation_error_bound"
    )
    aligned_noise = _nonnegative(
        aligned_observation_error_bound, "aligned_observation_error_bound"
    )
    source = _assess_mixed_effective_interaction(
        form_error_bounds=forms,
        phase_error_bounds=phases,
        registered_observation_error_bound=Q(0),
        aligned_observation_error_bound=Q(0),
    )
    registration = source.source
    phase = registration.source
    renewed = phase.source
    slow = renewed.source
    collective = slow.source
    persistent = collective.source
    storage = persistent.source
    positive = storage.source
    handoff = positive.handoff
    prefix = handoff.prefix
    reference = prefix.reference
    registered = _derive_registered_phase_probe_geometry()
    geometry = registered.successor
    pi = pi_interval()
    gamma, coupling = 1 / (1023 * pi), Q(13, 27) - _DELTA / (18 * pi)
    qlo, qhi = _SECTION - _SECTION_ERROR / 2, _SECTION + _SECTION_ERROR / 2
    budget = 200 * _MATURE
    cut_error = 8 * _MATURE / (1 + Q(10000, 3)) + 48 * _G * _MATURE + 6 * _WEIGHT
    output_force = 2 * _G * budget / 9 + 5 * _G * budget**2 / 81 + 14 * _WEIGHT / 2700
    original_output_error = 3 * cut_error + Q(81, 8) * output_force
    inherited_error = original_output_error + 3 * _G * qhi**3 + 27 * _G * qhi * _DECAY
    amplitude = (
        Q(4, 3) * _GL * Q(233, 300) * qlo * (1 - _G**2 / Q(4, 9)),
        16 * _G * qhi,
    )
    original_comparison = (
        3 * (24 * _MATURE + 6 * 14 * _WEIGHT * 10 * qhi / _GL) / (1 - 135 * qhi)
    )
    inherited_source_ok = (
        phase.source_association_certified
        and phase.relative_phase_registration_certified
        and renewed.source_association_certified
        and slow.uniform_approximation_certified
        and slow.tangent_roots_certified
        and slow.signed_kernel_certified
        and slow.nonlinear_error_certified
        and slow.fast_tail_certified
        and slow.amplitude_certified
        and slow.original_output_error_upper_bound == original_output_error
        and slow.nonlinear_output_error_upper_bound == 3 * _G * qhi**3
        and slow.fast_output_tail_upper_bound == 27 * _G * qhi * _DECAY
        and collective.cut_form_error_upper_bound == cut_error
        and phase.form_error_bounds
        == renewed.form_error_bounds
        == slow.form_error_bounds
        == forms
        and phase.phase_error_bounds
        == renewed.phase_error_bounds
        == slow.phase_error_bounds
        == phases
        and phase.phase_sensitive_observation_error_bound
        == phase.form_only_observation_error_bound
        == 0
        and phase.nominal_q0_bounds == renewed.nominal_q0_bounds == (qlo, qhi)
        and phase.gamma_bounds == renewed.gamma_bounds == gamma
        and phase.coupling_bounds == renewed.coupling_bounds == coupling
        and _GL < gamma.lo <= gamma.hi < _G
        and Q(4, 9) < coupling.lo <= coupling.hi < Q(1, 2)
        and phase.inherited_output_error_upper_bound
        == renewed.inherited_output_error_upper_bound
        == slow.total_output_error_upper_bound
        == inherited_error
        and phase.slow_amplitude_bounds
        == renewed.slow_amplitude_bounds
        == slow.slow_amplitude_bounds
        == amplitude
        and 0 < inherited_error < _NOISE
        and amplitude[0] / 2 - inherited_error > 0
        and geometry == reference.source_geometry == prefix.source_geometry
        and registered.cross_symmetry_certified
        and geometry.nodes == tuple(range(18))
        and geometry.degrees == (9,) * 18
        and geometry.held_capacities == (Q(1),) * 18
        and geometry.support_certified
        and geometry.reference_stationary_certified
        and geometry.spectral_gap_certified
        and geometry.bare_gap_lower_bound == 4
        and geometry.weighted_gap_lower_bound >= 3
        and sum(geometry.target_phase_turns, Q(0)) == 0
        and reference.sources
        == (reference.donor.source, reference.recipient_readiness.sources[0])
        and all(
            _original_source_matches(row, geometry, rx, rt)
            for row, rx, rt in zip(reference.sources, forms, phases, strict=True)
        )
        and reference.original_relative_phase_gap == _DELTA
        and reference.original_supplied_phase_origins == (-_DELTA / 2, _DELTA / 2)
        and reference.actual_weight_deviation_upper_bound == _WEIGHT
        and phase.phase_mean_error_upper_bound == 3 * _ETA
        and phase.relative_phase_entry_error_upper_bound == 6 * _ETA
        and reference.donor.section_mean_gap == reference.section_mean_gap == _SECTION
        and reference.donor.comparison_error_upper_bound
        == reference.donor_section_comparison_error
        == _SECTION_ERROR
        and reference.recipient_mature_radius == _MATURE
        and reference.recipient_isolated_deadline
        == reference.recipient_readiness.waiting_duration
        == Q(10**10)
        and reference.recipient_readiness.derived_mature_norm_upper_bound == _MATURE
        and reference.initial_form_means == handoff.retained_form_means == (Q(0), Q(0))
        and phase.original_donor_form_total == renewed.original_donor_form_total == 0
        and phase.whole_form_mean == persistent.whole_form_mean == 0
        and renewed.whole_form_mean
        == collective.whole_form_mean
        == storage.whole_form_mean
        == positive.whole_form_mean
        == 0
        and positive.exact_zero_entry_total_from_prior_laws
        and phase.physical_capacities == persistent.physical_capacities == (Q(1),) * 36
        and phase.full_state_dimension == persistent.full_state_dimension == 72
        and phase.full_contact_pairs == renewed.full_contact_pairs == _PAIRS
        and phase.internal_divisor == 9
        and phase.aggregate_equation == "Q_prime=-2*Q-gamma*M"
        and phase.relative_phase_equation == "psi_prime=2*gamma*Q/9"
        and phase.original_phase_lifts_retained
        and phase.original_baseline_retained
        and phase.inherited_amplitude_rate_correlation_retained
        and phase.same_actual_member_comparator
        and slow.same_actual_history_required
        and slow.correlated_initialization_retained
        and slow.initialization == "u0=0; b0 and z0 carried from the same original cut"
        and positive.prefix_full_duration
        == handoff.prefix_duration
        == prefix.contact_duration
        == Q(2000000)
        and positive.prefix_cut_duration == handoff.continuation_duration == Q(10000)
        and positive.handoff_time
        == handoff.total_duration
        == persistent.activation_time
        == phase.activation_time
        == _ACTIVATION
        and positive.contact_pairs
        == persistent.contact_pairs
        == tuple((j, 18 + j) for j in range(9))
        and persistent.fixed_port_strength == storage.fixed_port_strength == 1
        and handoff.switch_word == (1, 0)
        and phase.switch_age == renewed.switch_age == slow.early_age == _SWITCH
        and phase.switch_time == _ACTIVATION + _SWITCH
        and phase.post_duration == _SEGMENT
        and phase.total_duration == _ACTIVATION + _SWITCH + _SEGMENT
        and phase.original_clock_horizon
        == (_ACTIVATION + _SWITCH + _SEGMENT) * Q(1024, 1023)
        and phase.clock
        == "tau=(1023/1024)*t; complete internal and phase-sensitive matching rows"
        and renewed.clock
        == "tau=(1023/1024)*t; all internal rows and renewed form port"
        and slow.clock == "tau=(1023/1024)*t; age from original unit A-port activation"
        and persistent.clock
        == "tau=(1023/1024)*t; same complete law and unit A form port"
        and positive.clock
        == "tau=(1023/1024)*t; complete internal rows and reciprocal A form port"
        and handoff.clock == "tau=(1023/1024)*t; all complete internal rows retained"
        and prefix.clock == "tau=(1023/1024)*t; complete full and cut internal laws"
        and reference.clock
        == "tau=(1023/1024)*t; complete internal rows and reciprocal form inputs"
        and persistent.uniform_tube_certified
        and persistent.nominal_cone_certified
        and persistent.bregman_comparison_certified
        and persistent.phase_convexity_certified
        and persistent.bregman_growth_integral_upper_bound == 135 * qhi < 1
        and persistent.bregman_denominator == 1 - 135 * qhi
        and persistent.nominal_norm_upper_bound == 4 * qhi
        and persistent.uniform_comparison_budget == budget
        and persistent.uniform_comparison_upper_bound == original_comparison < budget
        and persistent.marked_gap_upper_bound == qhi + 300 * _MATURE
        and persistent.registration_gap_limit == _DELTA / 4
        and persistent.original_phase_gauges_and_branches_retained
        and storage.selected_mu == persistent.selected_mu == 1
        and storage.storage_domain_certified
        and storage.actual_dissipation_certified
        and storage.mean_projector_rank == 1
        and storage.origin_reference
        == "conserved_whole_form_mean; P_rel=P_copymean-P_wholemean"
        and storage.cross_strength_bounds == 9 * coupling
        and not any(
            (
                phase.state_reset_performed,
                phase.original_source_reacquired,
                phase.observation_rebased_at_switch,
                renewed.state_reset_performed,
                slow.state_reset_performed,
                positive.state_reset_performed,
                handoff.state_reset_performed,
            )
        )
    )
    initial_phase_error = 6 * _ETA
    entry = 4 * qhi + budget
    registration_ok = (
        inherited_source_ok
        and registration.source_association_certified
        and registration.registration_covariance_certified
        and registration.form_error_bounds == forms
        and registration.phase_error_bounds == phases
        and registration.registered_observation_error_bound
        == registration.aligned_observation_error_bound
        == 0
        and registration.gamma_bounds == gamma
        and registration.coupling_bounds == coupling
        and registration.nominal_q0_bounds == (qlo, qhi)
        and registration.slow_amplitude_bounds == amplitude
        and registration.inherited_output_error_upper_bound == inherited_error
        and registration.initial_relative_phase_error_upper_bound == initial_phase_error
        and registration.centered_entry_norm_upper_bound == entry
        and registration.registration_turns == Q(1, 4)
        and registration.registration_angle_bounds == pi / 2
        and registration.translated_copy == "recipient"
        and registration.preparation
        == "recipient whole precontact phase history translated by +pi/2"
        and registration.complete_precontact_history_translated
        and registration.source_covariance_is_conditional
        and registration.internal_identity_and_winding_unchanged_by_translation
        and registration.prior_form_observations_unchanged_by_translation
        and registration.aligned_counterpart_retains_same_underlying_source
        and registration.original_baseline_retained
        and registration.original_lifts_and_error_association_retained
        and phase.initial_relative_phase_mean_assumed_zero is False
        and registration.same_complete_contact_law
        and registration.existing_field_reused_unchanged
        and registration.original_donor_form_total == registration.whole_form_mean == 0
        and registration.full_state_dimension == 72
        and registration.physical_capacities == (Q(1),) * 36
        and registration.full_contact_pairs == _PAIRS
        and registration.internal_divisor == 9
        and registration.aggregate_equation == "Q_prime=-2*Q-gamma*M"
        and registration.relative_phase_equation == "psi_prime=2*gamma*Q/9"
        and registration.phase_moment_decomposition == "M=18*sin(psi)+R; abs(R)<=Nc**2"
        and registration.common_storage_reference
        == "same unshifted Theta in Uplus(Theta)"
        and registration.comparison_storage_rate == "-norm((H+9G)*x)**2/9"
        and registration.activation_time == _ACTIVATION
        and registration.switch_age == _SWITCH
        and registration.switch_time == _ACTIVATION + _SWITCH
        and registration.post_duration == _SEGMENT
        and registration.total_duration == _ACTIVATION + _SWITCH + _SEGMENT
        and registration.original_clock_horizon
        == (_ACTIVATION + _SWITCH + _SEGMENT) * Q(1024, 1023)
        and registration.clock
        == "tau=(1023/1024)*t; translated history and all reciprocal contact rows"
        and not any(
            (
                registration.source_errors_translated,
                registration.original_source_report_mutated,
                registration.boundary_phase_clamped,
                registration.later_phase_intervention_performed,
                registration.state_reset_performed,
                registration.original_source_reacquired,
                registration.reference_shift_used_to_erase_switch_work,
            )
        )
    )
    source_ok = (
        registration_ok
        and source.source_association_certified
        and source.registration_covariance_certified
        and source.form_error_bounds == forms
        and source.phase_error_bounds == phases
        and source.registered_observation_error_bound
        == source.aligned_observation_error_bound
        == 0
        and source.gamma_bounds == gamma
        and source.coupling_bounds == coupling
        and source.nominal_q0_bounds == (qlo, qhi)
        and source.slow_amplitude_bounds == amplitude
        and source.inherited_output_error_upper_bound == inherited_error
        and source.initial_relative_phase_error_upper_bound == initial_phase_error
        and source.centered_entry_norm_upper_bound == entry
        and source.initial_registrations_turns == (Q(0), Q(1, 4))
        and source.initial_reduced_form == "same source-correlated Q_s(3375000)"
        and source.reduced_state_coordinates == ("Q", "psi")
        and source.reduced_state_dimension == 2
        and source.full_state_dimension == 72
        and source.physical_capacities == (Q(1),) * 36
        and source.full_contact_pairs == _PAIRS
        and source.internal_divisor == 9
        and source.original_donor_form_total == source.whole_form_mean == 0
        and source.effective_form_equation == "Qhat_prime=-2*Qhat-18*gamma*sin(psihat)"
        and source.effective_phase_equation == "psihat_prime=2*gamma*Qhat/9"
        and source.actual_form_equation == "Q_prime=-2*Q-18*gamma*sin(psi)-gamma*R"
        and source.actual_phase_equation == "psi_prime=2*gamma*Q/9"
        and source.moment_decomposition == "M=18*sin(psi)+R; abs(R)<=Nc**2"
        and source.common_storage_reference == "same unshifted Theta in Uplus(Theta)"
        and source.full_storage_rate == "-norm((H+9G)*x)**2/9"
        and source.same_law_for_both_preparations
        and source.original_complete_field_unchanged
        and source.whole_history_covariance_retained
        and source.source_correlated_initialization_retained
        and source.activation_time == _ACTIVATION
        and source.switch_age == _SWITCH
        and source.switch_time == _ACTIVATION + _SWITCH
        and source.segment_duration == _SEGMENT
        and source.post_duration == _PREVIOUS
        and source.total_duration == _ACTIVATION + _SWITCH + _PREVIOUS
        and source.original_clock_horizon
        == (_ACTIVATION + _SWITCH + _PREVIOUS) * Q(1024, 1023)
        and source.clock
        == "tau=(1023/1024)*t; unchanged actual and effective interaction rows"
        and not any(
            (
                source.pressure_selected_from_preparation_label,
                source.state_reset_performed,
                source.phase_clamped,
                source.independent_marginal_state_reselection,
                source.original_source_reacquired,
                source.reference_shift_used_to_erase_switch_work,
            )
        )
    )
    covariance = source_ok
    # The actual weighted internal blocks retain the bare graph gap times
    # the minimum held conductance. Adding positive matching cannot lower
    # that centered gap; no commutation of the unequal blocks is assumed.
    gap = (
        source_ok
        and geometry.bare_gap_lower_bound == 4
        and geometry.weighted_gap_lower_bound >= 3
        and geometry.spectral_gap_certified
        and geometry.degrees == (9,) * 18
        and len(geometry.internal_edges) == 18
        and all(weight == 1 for _, _, weight in geometry.internal_edges)
        and len(geometry.new_edge_pairs) == 63
        and len(geometry.edge_pairs) == 81
        and all(row.new_weight_range == (Q(3, 4), Q(1)) for row in reference.sources)
    )
    # Each coordinate norm is at most the original joint norm. This does
    # not assert that the two separate conservative bounds saturate jointly.
    envelope = _damped_centered_envelope(
        initial_form_norm=entry, initial_phase_norm=entry, duration=_DURATION
    )
    form_norm = envelope.uniform_form_norm_upper_bound
    phase_norm = envelope.uniform_phase_norm_upper_bound
    filtered = gap and envelope.feedback_denominator > 0
    marked = qhi + 300 * _MATURE + Q(3, 2) * envelope.phase_displacement_upper_bound
    registration_margin = _DELTA / 4 - marked
    identity = (
        filtered
        and form_norm + phase_norm < Q(1, 100) < Q(1, 32)
        and registration_margin > 0
    )
    aggregate = amplitude[1] + inherited_error + 9 * _G
    phase_deviation = initial_phase_error + 2 * _G * _DURATION * aggregate / 9
    branch_margin = Q(1, 4) - phase_deviation - 2 * phase_norm
    branch = identity and branch_margin > 0
    # Only centered phases enter the discarded sine moment.
    remainder = phase_norm**2
    moment_ok = branch and remainder > 0
    phase_work = 972 * _ETA + Q(9, 2) * entry**3
    aligned_jump = (Q(0), 9 * (entry + 18 * _ETA) ** 2)
    registered_jump = (162 - phase_work, 162 + phase_work + 9 * entry**2)
    work_ok = (
        source_ok
        and aligned_jump[1] >= 0
        and 0 < registered_jump[0] <= registered_jump[1]
    )
    storage_remainder = 18 * (form_norm**2 + phase_norm**2)
    exchange = 2 * _G * _DURATION * aggregate * remainder
    balance = moment_ok and work_ok

    # This fixed-H transfer is distinct from the unchanged old 8/16 helper.
    # H>=16 permits the existing dyadic majorant without a huge power or an
    # evaluated exponential. Phase accumulation is explicitly paid.
    prediction_denominator = 1 - 2 * _G**2 * _DURATION
    decay = Q(1, 2**32)
    uniform_error = (
        inherited_error + 9 * _G * initial_phase_error + _G * remainder / 2
    ) / prediction_denominator
    phase_error = initial_phase_error + 2 * _G * _DURATION * uniform_error / 9
    endpoint_error = (
        inherited_error * decay
        + 9 * _G * initial_phase_error
        + 2 * _G**2 * _DURATION * uniform_error
        + _G * remainder / 2
    )
    form_margin = _NOISE - max(uniform_error, endpoint_error)
    phase_margin = _NOISE - phase_error
    form_ok = (
        balance and prediction_denominator > 0 and _DURATION >= 16 and form_margin > 0
    )
    phase_ok = balance and prediction_denominator > 0 and phase_margin > 0
    storage_error = (
        storage_remainder + 2 * aggregate * uniform_error + 162 * phase_error
    )
    # A positive comparison mode prevents an attraction conclusion from
    # this estimate. It is not an instability witness for the actual law.
    matrix = ((-Q(1, 3), 4 * _G), (4 * _G, Q(0)))
    determinant = matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
    witness = (12 * _G, Q(1))
    quadratic = sum(
        witness[i] * matrix[i][j] * witness[j] for i in (0, 1) for j in (0, 1)
    )
    limitation = gap and determinant == -16 * _G**2 < 0 and quadratic == 48 * _G**2 > 0
    admitted = max(registered_noise, aligned_noise) <= _NOISE
    registered_record_error = endpoint_error + registered_noise
    aligned_record_error = endpoint_error + aligned_noise
    registered_record_ok = form_ok and phase_ok and admitted
    aligned_record_ok = form_ok and phase_ok and admitted
    certified = (
        covariance
        and identity
        and balance
        and limitation
        and registered_record_ok
        and aligned_record_ok
    )
    unresolved = tuple(
        label
        for good, label in (
            (source_ok, "original_source_or_inherited_bounds_unavailable"),
            (covariance, "whole_history_registration_covariance_unavailable"),
            (gap, "actual_centered_weighted_gap_unavailable"),
            (filtered, "finite_damped_centered_comparison_unavailable"),
            (identity, "sustained_internal_identity_unavailable"),
            (branch, "sustained_matching_branch_unavailable"),
            (balance, "complete_and_effective_storage_balance_unavailable"),
            (form_ok, "sustained_form_prediction_tolerance_not_certified"),
            (phase_ok, "accumulated_phase_prediction_tolerance_not_certified"),
            (limitation, "comparison_limitation_not_established"),
            (admitted, "independent_observation_errors_outside_protocol"),
        )
        if not good
    )
    return _MixedSustainedInteraction(
        form_error_bounds=forms,
        phase_error_bounds=phases,
        registered_observation_error_bound=registered_noise,
        aligned_observation_error_bound=aligned_noise,
        source=source,
        source_association_certified=source_ok,
        registration_covariance_certified=covariance,
        weighted_gap_certified=gap,
        damped_centered_bound_certified=filtered,
        centered_identity_certified=identity,
        phase_branch_certified=branch,
        phase_moment_remainder_certified=moment_ok,
        switching_work_certified=work_ok,
        effective_balance_certified=balance,
        form_prediction_certified=form_ok,
        phase_prediction_certified=phase_ok,
        comparison_limitation_certified=limitation,
        observation_allowances_admitted=admitted,
        registered_record_prediction_certified=registered_record_ok,
        aligned_record_prediction_certified=aligned_record_ok,
        gamma_bounds=gamma,
        coupling_bounds=coupling,
        nominal_q0_bounds=(qlo, qhi),
        slow_amplitude_bounds=amplitude,
        inherited_output_error_upper_bound=inherited_error,
        initial_relative_phase_error_upper_bound=initial_phase_error,
        centered_entry_norm_upper_bound=entry,
        centered_envelope=envelope,
        old_joint_growth_denominator=1 - 4 * _G * _DURATION,
        marked_gap_upper_bound=marked,
        registration_margin=registration_margin,
        aggregate_absolute_upper_bound=aggregate,
        relative_phase_deviation_upper_bound=phase_deviation,
        boundary_branch_margin=branch_margin,
        phase_moment_remainder_upper_bound=remainder,
        prediction_feedback_denominator=prediction_denominator,
        prediction_decay_upper_bound=decay,
        uniform_form_error_upper_bound=uniform_error,
        phase_error_upper_bound=phase_error,
        endpoint_form_error_upper_bound=endpoint_error,
        form_prediction_margin=form_margin,
        phase_prediction_margin=phase_margin,
        phase_switching_work_allowance=phase_work,
        aligned_storage_jump_bounds=aligned_jump,
        registered_storage_jump_bounds=registered_jump,
        storage_remainder_upper_bound=storage_remainder,
        unresolved_exchange_integral_upper_bound=exchange,
        storage_prediction_error_upper_bound=storage_error,
        registered_record_error_upper_bound=registered_record_error,
        aligned_record_error_upper_bound=aligned_record_error,
        comparison_matrix=matrix,
        comparison_determinant=determinant,
        comparison_positive_witness=witness,
        comparison_positive_quadratic_value=quadratic,
        certified=certified,
        status=(
            "conditional_sustained_interaction_validity" if certified else "unavailable"
        ),
        unresolved_conditions=unresolved,
    )
