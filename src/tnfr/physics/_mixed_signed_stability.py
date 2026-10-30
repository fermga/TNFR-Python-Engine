"""Signed transverse attraction with separate identity and prediction limits.

The unchanged common-quadratic owner acts on the actual moving centered
state after a jointly admitted phase-sector and source inclusion proof.
The isotropic bound does not establish the stricter marked collar, and an
integrable discarded moment is not itself an all-time prediction theorem.
"""

from dataclasses import dataclass
from fractions import Fraction as Q

from ..mathematics._rational_interval import I, cos, pi_interval
from ._mixed_reciprocal_partner import _nonnegative, _nonnegative_pair
from ._mixed_registered_phase_probe import _derive_registered_phase_probe_geometry
from ._mixed_resource_transfer_obstruction import _original_source_matches
from ._mixed_sustained_interaction import (
    _assess_mixed_sustained_interaction,
    _MixedSustainedInteraction,
)
from ._sine_lyapunov import _sine_common_quadratic_coefficients

_DELTA = Q(1, 1024)
_WEIGHT, _ETA, _MATURE = Q(2, 10**60), Q(2, 10**69), Q(1, 2**28)
_SECTION = 3 * _DELTA / 16
_SECTION_ERROR = Q(3, 2) * (14 * _ETA + 2 * 10**10 * _WEIGHT)
_GL, _G = Q(1, 3300), Q(1, 3000)
_ACTIVATION, _SWITCH = Q(2010000), Q(3375000)
_SEGMENT, _PREVIOUS, _FINITE = Q(8), Q(16), Q(4096)
_DECAY, _NOISE = Q(1, 2**16), Q(1, 10**9)
_RADIUS, _PHASE_EXCESS = Q(1, 500), Q(1, 1000)
_PAIRS = tuple((j, 18 + j) for j in range(18))


@dataclass(frozen=True)
class _MixedSignedStability:
    """Conditional transverse decay without extending distinct old contracts."""

    form_error_bounds: tuple[Q, Q]
    phase_error_bounds: tuple[Q, Q]
    registered_observation_error_bound: Q
    aligned_observation_error_bound: Q
    source: _MixedSustainedInteraction
    source_association_certified: bool
    registration_covariance_certified: bool
    sharp_source_bound_certified: bool
    weighted_gap_certified: bool
    phase_stiffness_certified: bool
    common_metric_certified: bool
    joint_bootstrap_certified: bool
    transverse_decay_certified: bool
    cycle_windings_certified: bool
    integrable_moment_certified: bool
    switching_work_certified: bool
    strict_identity_nonimplication_certified: bool
    collective_comparison_limitation_certified: bool
    observation_allowances_admitted: bool
    gamma_bounds: I
    coupling_bounds: I
    nominal_q0_bounds: tuple[Q, Q]
    slow_amplitude_bounds: tuple[Q, Q]
    inherited_output_error_upper_bound: Q
    initial_relative_phase_error_upper_bound: Q
    centered_entry_norm_upper_bound: Q
    source_comparison_upper_bound: Q
    nominal_centered_form_upper_bound: Q
    nominal_centered_phase_upper_bound: Q
    initial_centered_form_upper_bound: Q
    initial_centered_phase_upper_bound: Q
    initial_scaled_norm_squared_upper_bound: Q
    normalized_metric_lower_bound: Q
    normalized_metric_upper_bound: Q
    metric_lower_bound: Q
    metric_upper_bound: Q
    shared_metric_decay_rate: Q
    decay_rate: Q
    scaled_norm_squared_upper_bound: Q
    scaled_radius_margin: Q
    phase_sector_bound: I
    internal_edge_cosine_lower_bound: Q
    phase_stiffness_lower_bound: Q
    phase_stiffness_margin: Q
    metric_stiffness_margin: Q
    entry_aggregate_upper_bound: Q
    aggregate_absolute_upper_bound: Q
    initial_full_storage_upper_bound: Q
    phase_boundary_storage_lower_bound: Q
    phase_barrier_margin: Q
    phase_moment_amplitude_upper_bound: Q
    phase_moment_integral_upper_bound: Q
    unresolved_exchange_integral_upper_bound: Q
    storage_remainder_upper_bound: Q
    phase_switching_work_allowance: Q
    aligned_storage_jump_bounds: tuple[Q, Q]
    registered_storage_jump_bounds: tuple[Q, Q]
    marked_witness_phase: tuple[Q, ...]
    marked_witness_pair: tuple[int, int]
    marked_witness_gap: Q
    marked_witness_norm_squared: Q
    marked_witness_allowance_margin: Q
    comparison_matrix: tuple[tuple[Q, Q], tuple[Q, Q]]
    comparison_determinant: Q
    comparison_positive_witness: tuple[Q, Q]
    comparison_positive_quadratic_value: Q
    certified: bool
    status: str
    unresolved_conditions: tuple[str, ...]
    initial_registrations_turns: tuple[Q, Q] = (Q(0), Q(1, 4))
    activation_time: Q = _ACTIVATION
    switch_age: Q = _SWITCH
    switch_time: Q = _ACTIVATION + _SWITCH
    original_clock_start_time: Q = (_ACTIVATION + _SWITCH) * Q(1024, 1023)
    previous_certified_duration: Q = _FINITE
    scaled_radius_limit: Q = _RADIUS
    relative_phase_excess_limit: Q = _PHASE_EXCESS
    shared_phase_scale: Q = Q(2)
    auxiliary_capacity: Q = Q(4, 9)
    metric_cosine_lower_bound: Q = Q(1, 500)
    weighted_gap_lower_bound: Q = Q(3)
    bare_gap_lower_bound: Q = Q(4)
    transport_norm_upper_bound: Q = Q(36)
    matching_cosine_lower_bound: Q = -Q(1, 400)
    selected_phase_stiffness_lower_bound: Q = Q(1, 3)
    internal_divisor: Q = Q(9)
    full_state_dimension: int = 72
    physical_capacities: tuple[Q, ...] = (Q(1),) * 36
    full_contact_pairs: tuple[tuple[int, int], ...] = _PAIRS
    original_donor_form_total: Q = Q(0)
    whole_form_mean: Q = Q(0)
    registered_observation_error_cap: Q = _NOISE
    aligned_observation_error_cap: Q = _NOISE
    common_storage_reference: str = "same unshifted Theta in Uplus(Theta)"
    metric_formula: str = "W=(9/4)*(X**2+4*Y**2+8*gamma*dot(xc,yc))"
    scaled_norm_formula: str = "Z**2=X**2+4*Y**2"
    complete_storage_rate: str = "-norm((H+9G)*x)**2/9"
    centered_decay_formula: str = "Z(t)**2<=Zbar_squared*exp(-lambda*t)"
    phase_moment_decay_formula: str = "abs(R(t))<=Zbar_squared*exp(-lambda*t)/4"
    collective_error_coordinates: tuple[str, str] = ("abs(Q-Qhat)/9", "abs(psi-psihat)")
    all_postattachment_times_covered: bool = False
    actual_asymptotic_attraction_certified: bool = False
    strict_marked_identity_certified: bool = False
    uniform_collective_prediction_certified: bool = False
    original_finite_identity_contract_extended: bool = False
    original_finite_prediction_contract_extended: bool = False
    new_endpoint_or_phase_record_claimed: bool = False
    record_allowances_used_for_stability: bool = False
    same_complete_and_effective_laws: bool = True
    whole_history_covariance_retained: bool = True
    source_correlated_initialization_retained: bool = True
    original_attachment_state_retained: bool = True
    prior_finite_endpoint_used_as_initial_state: bool = False
    original_attachment_work_retained_once: bool = True
    direct_entry_energy_adds_attachment_twice: bool = False
    unequal_internal_blocks_assumed_commuting: bool = False
    stiffness_derivative_consumed: bool = False
    independently_fixed_copy_gauges_assumed: bool = False
    boundary_convexity_assumed: bool = False
    phase_clamped: bool = False
    unresolved_storage_assumed_nonnegative: bool = False
    marked_witness_is_acquired_state: bool = False
    actual_strict_identity_failure_claimed: bool = False
    positive_comparison_direction_is_eigenvector: bool = False
    actual_collective_instability_claimed: bool = False
    all_time_prediction_failure_proved: bool = False
    state_reset_performed: bool = False
    original_source_reacquired: bool = False
    original_source_replayed: bool = False
    original_source_trajectory_evaluated: bool = False
    reduced_trajectory_evaluated: bool = False
    exact_actual_two_coordinate_closure: bool = False
    autonomous_pair_formation_claimed: bool = False
    minimal_intrinsic_macrostate_claimed: bool = False
    runtime_event_executed: bool = False
    operator_occurrence_derived: bool = False
    exponentials_evaluated: bool = False
    spectral_roots_evaluated: bool = False
    newly_observed_aggregates: int = 0
    newly_observed_phases: int = 0
    numerical_response_allowance: Q = Q(0)
    physical_property_identified: bool = False
    empirical_validation: bool = False
    clock: str = "tau=(1023/1024)*t; all postattachment actual and effective rows"


def _assess_mixed_signed_stability(
    *,
    form_error_bounds,
    phase_error_bounds,
    registered_observation_error_bound,
    aligned_observation_error_bound,
):
    """Rebuild source inclusion, joint trapping, signed decay and its limits."""
    forms = _nonnegative_pair(form_error_bounds, "form_error_bounds")
    phases = _nonnegative_pair(phase_error_bounds, "phase_error_bounds")
    registered_noise = _nonnegative(
        registered_observation_error_bound, "registered_observation_error_bound"
    )
    aligned_noise = _nonnegative(
        aligned_observation_error_bound, "aligned_observation_error_bound"
    )
    source = _assess_mixed_sustained_interaction(
        form_error_bounds=forms,
        phase_error_bounds=phases,
        registered_observation_error_bound=Q(0),
        aligned_observation_error_bound=Q(0),
    )
    effective = source.source
    registration = effective.source
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
    effective_source_ok = (
        registration_ok
        and effective.source_association_certified
        and effective.registration_covariance_certified
        and effective.form_error_bounds == forms
        and effective.phase_error_bounds == phases
        and effective.registered_observation_error_bound
        == effective.aligned_observation_error_bound
        == 0
        and effective.gamma_bounds == gamma
        and effective.coupling_bounds == coupling
        and effective.nominal_q0_bounds == (qlo, qhi)
        and effective.slow_amplitude_bounds == amplitude
        and effective.inherited_output_error_upper_bound == inherited_error
        and effective.initial_relative_phase_error_upper_bound == initial_phase_error
        and effective.centered_entry_norm_upper_bound == entry
        and effective.initial_registrations_turns == (Q(0), Q(1, 4))
        and effective.initial_reduced_form == "same source-correlated Q_s(3375000)"
        and effective.reduced_state_coordinates == ("Q", "psi")
        and effective.reduced_state_dimension == 2
        and effective.full_state_dimension == 72
        and effective.physical_capacities == (Q(1),) * 36
        and effective.full_contact_pairs == _PAIRS
        and effective.internal_divisor == 9
        and effective.original_donor_form_total == effective.whole_form_mean == 0
        and effective.effective_form_equation
        == "Qhat_prime=-2*Qhat-18*gamma*sin(psihat)"
        and effective.effective_phase_equation == "psihat_prime=2*gamma*Qhat/9"
        and effective.actual_form_equation == "Q_prime=-2*Q-18*gamma*sin(psi)-gamma*R"
        and effective.actual_phase_equation == "psi_prime=2*gamma*Q/9"
        and effective.moment_decomposition == "M=18*sin(psi)+R; abs(R)<=Nc**2"
        and effective.common_storage_reference == "same unshifted Theta in Uplus(Theta)"
        and effective.full_storage_rate == "-norm((H+9G)*x)**2/9"
        and effective.same_law_for_both_preparations
        and effective.original_complete_field_unchanged
        and effective.whole_history_covariance_retained
        and effective.source_correlated_initialization_retained
        and effective.activation_time == _ACTIVATION
        and effective.switch_age == _SWITCH
        and effective.switch_time == _ACTIVATION + _SWITCH
        and effective.segment_duration == _SEGMENT
        and effective.post_duration == _PREVIOUS
        and effective.total_duration == _ACTIVATION + _SWITCH + _PREVIOUS
        and effective.original_clock_horizon
        == (_ACTIVATION + _SWITCH + _PREVIOUS) * Q(1024, 1023)
        and effective.clock
        == "tau=(1023/1024)*t; unchanged actual and effective interaction rows"
        and not any(
            (
                effective.pressure_selected_from_preparation_label,
                effective.state_reset_performed,
                effective.phase_clamped,
                effective.independent_marginal_state_reselection,
                effective.original_source_reacquired,
                effective.reference_shift_used_to_erase_switch_work,
            )
        )
    )
    sine_coefficient = 1 + 2 * cos(pi / 9)
    cone_margins = (
        2 * coupling.lo / 3 - _G**2,
        2 * coupling.lo - Q(1, 5) - sine_coefficient.hi / 9 - 4 * _G**2 * coupling.hi,
        Q(1, 5) - Q(24, 5) * _G**2 * coupling.hi,
    )
    source_ok = (
        effective_source_ok
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
        and source.same_complete_and_effective_laws
        and source.whole_history_covariance_retained
        and source.source_correlated_initialization_retained
        and source.effective_form_equation == "Qhat_prime=-2*Qhat-18*gamma*sin(psihat)"
        and source.effective_phase_equation == "psihat_prime=2*gamma*Qhat/9"
        and source.actual_form_equation == "Q_prime=-2*Q-18*gamma*sin(psi)-gamma*R"
        and source.actual_phase_equation == "psi_prime=2*gamma*Q/9"
        and source.common_storage_reference == "same unshifted Theta in Uplus(Theta)"
        and source.full_storage_rate == "-norm((H+9G)*x)**2/9"
        and source.activation_time == _ACTIVATION
        and source.switch_age == _SWITCH
        and source.switch_time == _ACTIVATION + _SWITCH
        and source.post_duration == _FINITE
        and source.total_duration == _ACTIVATION + _SWITCH + _FINITE
        and source.original_clock_horizon
        == (_ACTIVATION + _SWITCH + _FINITE) * Q(1024, 1023)
        and source.clock
        == "tau=(1023/1024)*t; sustained unchanged actual and effective rows"
        and persistent.gamma_bounds == (_GL, _G)
        and persistent.sine_coefficient_bounds
        == reference.donor.regional_sine_coefficient_bounds
        == sine_coefficient
        and persistent.coupling_bounds
        == reference.donor.nominal_coupling_bounds
        == coupling
        and reference.donor.gamma_bounds == gamma
        and persistent.nominal_q0_bounds == (qlo, qhi)
        and persistent.nominal_cone_margins == cone_margins
        and persistent.nominal_common_form_cone_factor == Q(6, 5)
        and persistent.nominal_contrast_form_cone_factor == 1
        and persistent.activation_relative_common_form == 0
        and persistent.inherited_form_velocity_retained
        and reference.inherited_donor_velocity_retained
        and reference.inherited_nominal_velocity_relation
        == "-gamma**2*q0 <= w0 <= 0 at the same isolated age"
        and reference.nominal_wedges_certified
        and handoff.continuation_wedges_certified
        and reference.donor.nominal_reference_certified
        and persistent.finite_variation_certified
        and persistent.nominal_phase_variation_upper_bound == 3 * qhi
        and persistent.nominal_form_integral_upper_bound == 10 * qhi / _GL
        and persistent.initial_bregman_root_upper_bound == 24 * _MATURE
        and persistent.weight_matrix_deviation_upper_bound == 14 * _WEIGHT
        and persistent.bregman_phase_growth_coefficient == 45
        and persistent.bregman_weight_forcing_coefficient == 6
        and persistent.bregman_coercivity_lower_bound == Q(1, 5)
        and not any(
            (
                source.state_reset_performed,
                source.phase_clamped,
                source.original_source_reacquired,
                source.independent_marginal_state_reselection,
                source.reference_shift_used_to_erase_switch_work,
            )
        )
    )
    covariance = source_ok
    gap = (
        source_ok
        and geometry.bare_gap_lower_bound == 4
        and geometry.weighted_gap_lower_bound >= 3
        and geometry.spectral_gap_certified
        and len(geometry.internal_edges) == 18
        and all(weight == 1 for _, _, weight in geometry.internal_edges)
        and len(geometry.new_edge_pairs) == 63
        and len(geometry.edge_pairs) == 81
        and all(row.new_weight_range == (Q(3, 4), Q(1)) for row in reference.sources)
    )
    # Projecting out the generated common form removes u from the nominal
    # source norm. The old signed cone bounds b and V, while the same actual
    # Bregman error remains paid in both centered components.
    nominal_form = 4 * _G * qhi
    nominal_phase = Q(9, 4) * qhi
    form_initial = nominal_form + original_comparison
    phase_initial = nominal_phase + original_comparison
    initial_square = form_initial**2 + 4 * phase_initial**2
    sharp = (
        source_ok
        and all(value > 0 for value in cone_margins)
        and 2 < sine_coefficient.lo <= sine_coefficient.hi < 3
        and 4 * coupling.lo > 1
        and 9 * (1 + Q(9, 16) ** 2) < 16
        and Q(9, 2) < Q(9, 4) ** 2
        and 0 < original_comparison < budget
    )
    internal_cosine = cos(4 * pi / 9 + 3 * _RADIUS / 4).lo
    phase_stiffness = Q(2, 5) - Q(18, 400)
    phase_stiffness_margin = phase_stiffness - Q(1, 3)
    metric_stiffness_margin = Q(1, 12) - Q(36, 500)
    stiffness = (
        gap
        and internal_cosine >= Q(1, 10)
        and _PHASE_EXCESS + 3 * _RADIUS / 4 == Q(1, 400)
        and phase_stiffness_margin > 0
        and metric_stiffness_margin > 0
    )
    metric_lower, metric_upper, shared_rate = _sine_common_quadratic_coefficients(
        gamma_lower=2 * _GL,
        gamma_upper=2 * _G,
        cosine_lower=Q(1, 500),
        gap_lower=Q(3),
        capacity_lower=Q(4, 9),
        capacity_upper=Q(4, 9),
    )
    normalized_lower, normalized_upper = 1 - 2 * _G, 1 + 2 * _G
    rate = _GL**2 / (375 * normalized_upper)
    metric_ok = (
        stiffness
        and metric_lower == Q(9, 4) * normalized_lower > 0
        and metric_upper == Q(9, 4) * normalized_upper
        and shared_rate == rate > 0
    )
    norm_square = normalized_upper * initial_square / normalized_lower
    norm_margin = _RADIUS**2 - norm_square
    entry_aggregate = amplitude[1] + inherited_error
    aggregate = entry_aggregate + 9 * _G
    entry_storage = 162 + entry_aggregate**2 + 972 * _ETA + 18 * entry**2
    phase_barrier = 162 + Q(81, 1000) - 18 * _RADIUS**2
    phase_margin = phase_barrier - entry_storage
    # This first-exit argument closes both the centered metric and actual
    # phase sector together. The full storage uses its original reference;
    # the already charged attachment work is not added to Eentry again.
    bootstrap = (
        covariance
        and sharp
        and metric_ok
        and norm_margin > 0
        and initial_phase_error < _PHASE_EXCESS
        and phase_margin > 0
    )
    decay = bootstrap
    windings = bootstrap and internal_cosine >= Q(1, 10)
    moment_amplitude = norm_square / 4
    moment_integral = moment_amplitude / rate
    exchange = 2 * _G * aggregate * moment_integral
    integrable = decay and moment_integral > 0
    storage_remainder = 18 * norm_square
    phase_work = 972 * _ETA + Q(9, 2) * entry**3
    aligned_jump = (Q(0), 9 * (entry + 18 * _ETA) ** 2)
    registered_jump = (162 - phase_work, 162 + phase_work + 9 * entry**2)
    work = (
        source_ok and 0 <= aligned_jump[0] < aligned_jump[1] and registered_jump[0] > 0
    )

    # This belongs to the larger inference class admitted by the isotropic
    # norm alone, not to a newly acquired source or an attained trajectory.
    pair_shape = len(geometry.marked_pairs) == 9 and all(
        0 <= a < 9 <= b < 18 for a, b in geometry.marked_pairs
    )
    witness_pair = geometry.marked_pairs[0] if pair_shape else (0, 9)
    witness_phase = [Q(0)] * 36
    witness_phase[witness_pair[0]] = -_DELTA / 6
    witness_phase[witness_pair[1]] = _DELTA / 6
    witness_phase = tuple(witness_phase)
    witness_gap = witness_phase[witness_pair[1]] - witness_phase[witness_pair[0]]
    witness_norm = sum(value**2 for value in witness_phase)
    witness_margin = phase_initial**2 - witness_norm
    identity_boundary = (
        sharp
        and pair_shape
        and sum(witness_phase[:18]) == sum(witness_phase[18:]) == 0
        and witness_gap == _DELTA / 3 > _DELTA / 4
        and witness_norm == _DELTA**2 / 18
        and witness_margin > 0
        and 4 * witness_norm < initial_square < _RADIUS**2
    )
    matrix = ((-Q(2), 2 * _G), (2 * _G, Q(0)))
    determinant = matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
    comparison_witness = (_G, Q(1))
    quadratic = sum(
        comparison_witness[i] * matrix[i][j] * comparison_witness[j]
        for i in (0, 1)
        for j in (0, 1)
    )
    prediction_boundary = (
        source_ok and determinant == -4 * _G**2 < 0 and quadratic == 2 * _G**2 > 0
    )
    admitted = max(registered_noise, aligned_noise) <= _NOISE
    certified = (
        bootstrap
        and windings
        and integrable
        and work
        and identity_boundary
        and prediction_boundary
        and admitted
    )
    unresolved = tuple(
        label
        for good, label in (
            (source_ok, "original_source_or_consumed_signed_cone_unavailable"),
            (sharp, "source_sensitive_centered_inclusion_unavailable"),
            (gap, "actual_centered_weighted_gap_unavailable"),
            (stiffness, "joint_sector_phase_coercivity_unavailable"),
            (metric_ok, "shared_common_metric_coefficients_unavailable"),
            (bootstrap, "joint_source_and_phase_barrier_not_closed"),
            (integrable, "integrable_internal_moment_not_established"),
            (work, "original_same_reference_attachment_unavailable"),
            (identity_boundary, "strict_marked_identity_nonimplication_unavailable"),
            (prediction_boundary, "collective_error_comparison_limit_unavailable"),
            (admitted, "original_observation_allowances_outside_contract"),
        )
        if not good
    )
    return _MixedSignedStability(
        form_error_bounds=forms,
        phase_error_bounds=phases,
        registered_observation_error_bound=registered_noise,
        aligned_observation_error_bound=aligned_noise,
        source=source,
        source_association_certified=source_ok,
        registration_covariance_certified=covariance,
        sharp_source_bound_certified=sharp,
        weighted_gap_certified=gap,
        phase_stiffness_certified=stiffness,
        common_metric_certified=metric_ok,
        joint_bootstrap_certified=bootstrap,
        transverse_decay_certified=decay,
        cycle_windings_certified=windings,
        integrable_moment_certified=integrable,
        switching_work_certified=work,
        strict_identity_nonimplication_certified=identity_boundary,
        collective_comparison_limitation_certified=prediction_boundary,
        observation_allowances_admitted=admitted,
        gamma_bounds=gamma,
        coupling_bounds=coupling,
        nominal_q0_bounds=(qlo, qhi),
        slow_amplitude_bounds=amplitude,
        inherited_output_error_upper_bound=inherited_error,
        initial_relative_phase_error_upper_bound=initial_phase_error,
        centered_entry_norm_upper_bound=entry,
        source_comparison_upper_bound=original_comparison,
        nominal_centered_form_upper_bound=nominal_form,
        nominal_centered_phase_upper_bound=nominal_phase,
        initial_centered_form_upper_bound=form_initial,
        initial_centered_phase_upper_bound=phase_initial,
        initial_scaled_norm_squared_upper_bound=initial_square,
        normalized_metric_lower_bound=normalized_lower,
        normalized_metric_upper_bound=normalized_upper,
        metric_lower_bound=metric_lower,
        metric_upper_bound=metric_upper,
        shared_metric_decay_rate=shared_rate,
        decay_rate=rate,
        scaled_norm_squared_upper_bound=norm_square,
        scaled_radius_margin=norm_margin,
        phase_sector_bound=pi / 2 + _PHASE_EXCESS,
        internal_edge_cosine_lower_bound=internal_cosine,
        phase_stiffness_lower_bound=phase_stiffness,
        phase_stiffness_margin=phase_stiffness_margin,
        metric_stiffness_margin=metric_stiffness_margin,
        entry_aggregate_upper_bound=entry_aggregate,
        aggregate_absolute_upper_bound=aggregate,
        initial_full_storage_upper_bound=entry_storage,
        phase_boundary_storage_lower_bound=phase_barrier,
        phase_barrier_margin=phase_margin,
        phase_moment_amplitude_upper_bound=moment_amplitude,
        phase_moment_integral_upper_bound=moment_integral,
        unresolved_exchange_integral_upper_bound=exchange,
        storage_remainder_upper_bound=storage_remainder,
        phase_switching_work_allowance=phase_work,
        aligned_storage_jump_bounds=aligned_jump,
        registered_storage_jump_bounds=registered_jump,
        marked_witness_phase=witness_phase,
        marked_witness_pair=witness_pair,
        marked_witness_gap=witness_gap,
        marked_witness_norm_squared=witness_norm,
        marked_witness_allowance_margin=witness_margin,
        comparison_matrix=matrix,
        comparison_determinant=determinant,
        comparison_positive_witness=comparison_witness,
        comparison_positive_quadratic_value=quadratic,
        certified=certified,
        status=(
            "conditional_transverse_attraction_with_identity_boundary"
            if certified
            else "unavailable"
        ),
        unresolved_conditions=unresolved,
        all_postattachment_times_covered=bootstrap,
        actual_asymptotic_attraction_certified=decay,
    )
