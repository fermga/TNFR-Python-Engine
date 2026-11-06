"""Source-correlated marked identity under unchanged reciprocal interaction.

The centered reference follows the same mathematical relative phase as the
actual source. It is an analytical comparison, not an autonomous prediction
or a phase observation. The existing common metric prices its correlated
transport forcing without discarding the original phase/form correlations.
"""

from dataclasses import dataclass
from fractions import Fraction as Q

from ..mathematics._rational_interval import I, cos, pi_interval
from ._mixed_reciprocal_partner import _nonnegative, _nonnegative_pair
from ._mixed_registered_phase_probe import _derive_registered_phase_probe_geometry
from ._mixed_resource_transfer_obstruction import _original_source_matches
from ._mixed_signed_stability import (
    _assess_mixed_signed_stability,
    _MixedSignedStability,
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
class _MixedMarkedIdentity:
    """Conditional strict identity, with collective prediction kept separate."""

    form_error_bounds: tuple[Q, Q]
    phase_error_bounds: tuple[Q, Q]
    registered_observation_error_bound: Q
    aligned_observation_error_bound: Q
    source: _MixedSignedStability
    source_association_certified: bool
    registration_covariance_certified: bool
    prior_domain_certified: bool
    source_cone_certified: bool
    reference_cone_certified: bool
    reference_integrals_certified: bool
    incremental_stiffness_certified: bool
    common_metric_certified: bool
    forced_comparison_certified: bool
    strict_marked_identity_certified: bool
    cycle_windings_certified: bool
    switching_work_certified: bool
    observation_allowances_admitted: bool
    gamma_bounds: I
    coupling_bounds: I
    sine_coefficient_bounds: I
    nominal_q0_bounds: tuple[Q, Q]
    source_comparison_upper_bound: Q
    centered_entry_norm_upper_bound: Q
    inherited_output_error_upper_bound: Q
    slow_amplitude_bounds: tuple[Q, Q]
    initial_scaled_norm_squared_upper_bound: Q
    scaled_norm_squared_upper_bound: Q
    scaled_radius_margin: Q
    phase_sector_bound: I
    initial_full_storage_upper_bound: Q
    phase_boundary_storage_lower_bound: Q
    phase_barrier_margin: Q
    normalized_metric_lower_bound: Q
    normalized_metric_upper_bound: Q
    metric_lower_bound: Q
    metric_upper_bound: Q
    shared_metric_decay_rate: Q
    decay_rate: Q
    reference_restoring_lower_margin: Q
    reference_velocity_cone_margins: tuple[Q, Q]
    reference_form_upper_bound: Q
    reference_phase_upper_bound: Q
    reference_form_integral_upper_bound: Q
    reference_phase_variation_upper_bound: Q
    reference_integral_coefficient_margin: Q
    incremental_stiffness_lower_bound: Q
    weight_matrix_deviation_upper_bound: Q
    initial_scaled_error_upper_bound: Q
    initial_metric_root_upper_bound: Q
    forcing_integral_upper_bound: Q
    metric_root_upper_bound: Q
    scaled_error_upper_bound: Q
    phase_error_upper_bound: Q
    marked_edge_error_upper_bound: Q
    nominal_marked_gap_upper_bound: Q
    marked_gap_upper_bound: Q
    registration_margin: Q
    marked_pairs: tuple[tuple[int, int], ...]
    phase_switching_work_allowance: Q
    aligned_storage_jump_bounds: tuple[Q, Q]
    registered_storage_jump_bounds: tuple[Q, Q]
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
    registration_gap_limit: Q = _DELTA / 4
    shared_phase_scale: Q = Q(2)
    auxiliary_capacity: Q = Q(4, 9)
    metric_cosine_lower_bound: Q = Q(1, 500)
    weighted_gap_lower_bound: Q = Q(3)
    bare_gap_lower_bound: Q = Q(4)
    transport_norm_upper_bound: Q = Q(36)
    reference_relative_cosine_lower_bound: Q = -_PHASE_EXCESS
    reference_position_upper_bound_factor: Q = Q(1)
    reference_velocity_upper_bound_factor: Q = Q(1)
    marked_pair_count: int = 18
    original_cycle_count: int = 4
    internal_divisor: Q = Q(9)
    full_state_dimension: int = 72
    physical_capacities: tuple[Q, ...] = (Q(1),) * 36
    full_contact_pairs: tuple[tuple[int, int], ...] = _PAIRS
    original_donor_form_total: Q = Q(0)
    whole_form_mean: Q = Q(0)
    registered_observation_error_cap: Q = _NOISE
    aligned_observation_error_cap: Q = _NOISE
    common_storage_reference: str = "same unshifted Theta in Uplus(Theta)"
    complete_storage_rate: str = "-norm((H+9G)*x)**2/9"
    reference_coordinates: tuple[str, ...] = ("s", "z", "V", "r")
    reference_relative_phase: str = "same actual continuous psi(t), analytically"
    reference_initialization: str = "original nominal centered source at A age3375000"
    reference_sum_phase_equation: str = "s_prime=4*gamma*k*V"
    reference_sum_form_equation: str = "V_prime=-2*k*V-(2*gamma*C/9)*sin(s/2)*cos(z/2)"
    reference_difference_phase_equation: str = "z_prime=4*gamma*(k+1)*r"
    reference_difference_form_equation: str = (
        "r_prime=-2*(k+1)*r-2*gamma*((C/9)*cos(s/2)+cos(psi))*sin(z/2)"
    )
    error_coordinates: tuple[str, str] = ("xc-xbar", "2*(yc-ybar)")
    metric_formula: str = "J=norm(a)**2+norm(b)**2+4*gamma*dot(a,b)"
    forcing_metric_rate: str = "-2*(1-4*gamma**2)*dot(a,deltaH*xbar)/9"
    all_postattachment_times_covered: bool = False
    original_finite_identity_contract_extended: bool = False
    original_finite_prediction_contract_extended: bool = False
    uniform_collective_prediction_certified: bool = False
    source_error_paid_once: bool = True
    source_correlated_initialization_retained: bool = True
    whole_history_covariance_retained: bool = True
    original_attachment_state_retained: bool = True
    original_attachment_work_retained_once: bool = True
    same_complete_and_effective_laws: bool = True
    original_marked_branches_retained: bool = True
    reference_is_autonomous_predictor: bool = False
    reference_is_complete_source_realization: bool = False
    actual_phase_used_as_analytical_coefficient: bool = True
    phase_readout_input_required: bool = False
    new_field_installed: bool = False
    independently_fixed_copy_gauges_assumed: bool = False
    copy_means_assumed_conserved: bool = False
    incremental_stiffness_replaced_by_radial_stiffness: bool = False
    stiffness_derivative_consumed: bool = False
    unequal_internal_blocks_assumed_commuting: bool = False
    mean_error_forcing_discarded: bool = False
    reference_velocity_reset: bool = False
    prior_finite_endpoint_used_as_initial_state: bool = False
    direct_entry_energy_adds_attachment_twice: bool = False
    prior_isotropic_witness_invalidated: bool = False
    phase_clamped: bool = False
    state_reset_performed: bool = False
    original_source_reacquired: bool = False
    original_source_replayed: bool = False
    original_source_trajectory_evaluated: bool = False
    reference_trajectory_evaluated: bool = False
    reduced_trajectory_evaluated: bool = False
    new_endpoint_or_phase_record_claimed: bool = False
    record_allowances_used_for_identity: bool = False
    exact_actual_two_coordinate_closure: bool = False
    intrinsic_pair_identity_claimed: bool = False
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
    clock: str = "tau=(1023/1024)*t; actual rows and same-psi centered reference"


def _assess_mixed_marked_identity(
    *,
    form_error_bounds,
    phase_error_bounds,
    registered_observation_error_bound,
    aligned_observation_error_bound,
):
    """Rebuild the actual sector, correlated reference and strict marked tube."""
    forms = _nonnegative_pair(form_error_bounds, "form_error_bounds")
    phases = _nonnegative_pair(phase_error_bounds, "phase_error_bounds")
    registered_noise = _nonnegative(
        registered_observation_error_bound, "registered_observation_error_bound"
    )
    aligned_noise = _nonnegative(
        aligned_observation_error_bound, "aligned_observation_error_bound"
    )
    signed_source = _assess_mixed_signed_stability(
        form_error_bounds=forms,
        phase_error_bounds=phases,
        registered_observation_error_bound=Q(0),
        aligned_observation_error_bound=Q(0),
    )
    source = signed_source.source
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

    # Rebuild the previously proved joint domain from its own consumed
    # source and metric premises. Its global or asymptotic verdict is unused.
    base = signed_source
    normalized_lower, normalized_upper = 1 - 2 * _G, 1 + 2 * _G
    metric_lower, metric_upper, shared_rate = _sine_common_quadratic_coefficients(
        gamma_lower=2 * _GL,
        gamma_upper=2 * _G,
        cosine_lower=Q(1, 500),
        gap_lower=Q(3),
        capacity_lower=Q(4, 9),
        capacity_upper=Q(4, 9),
    )
    rate = _GL**2 / (375 * normalized_upper)
    form_initial = 4 * _G * qhi + original_comparison
    phase_initial = Q(9, 4) * qhi + original_comparison
    initial_square = form_initial**2 + 4 * phase_initial**2
    norm_square = normalized_upper * initial_square / normalized_lower
    norm_margin = _RADIUS**2 - norm_square
    entry_aggregate = amplitude[1] + inherited_error
    entry_storage = 162 + entry_aggregate**2 + 972 * _ETA + 18 * entry**2
    phase_barrier = 162 + Q(81, 1000) - 18 * _RADIUS**2
    phase_margin = phase_barrier - entry_storage
    internal_cosine = cos(4 * pi / 9 + 3 * _RADIUS / 4).lo
    phase_stiffness = Q(2, 5) - Q(18, 400)
    source_ok = (
        source_ok
        and base.source_association_certified
        and base.registration_covariance_certified
        and base.form_error_bounds == forms
        and base.phase_error_bounds == phases
        and base.registered_observation_error_bound
        == base.aligned_observation_error_bound
        == 0
        and base.gamma_bounds == gamma
        and base.coupling_bounds == coupling
        and base.nominal_q0_bounds == (qlo, qhi)
        and base.source_comparison_upper_bound == original_comparison
        and base.centered_entry_norm_upper_bound == entry
        and base.inherited_output_error_upper_bound == inherited_error
        and base.slow_amplitude_bounds == amplitude
        and base.initial_relative_phase_error_upper_bound == 6 * _ETA
        and base.activation_time == _ACTIVATION
        and base.switch_age == _SWITCH
        and base.switch_time == _ACTIVATION + _SWITCH
        and base.original_clock_start_time == (_ACTIVATION + _SWITCH) * Q(1024, 1023)
        and base.previous_certified_duration == _FINITE
        and base.clock
        == "tau=(1023/1024)*t; all postattachment actual and effective rows"
        and base.initial_registrations_turns == (Q(0), Q(1, 4))
        and base.whole_history_covariance_retained
        and base.source_correlated_initialization_retained
        and base.original_attachment_state_retained
        and base.same_complete_and_effective_laws
        and base.full_state_dimension == 72
        and base.physical_capacities == (Q(1),) * 36
        and base.full_contact_pairs == _PAIRS
        and base.internal_divisor == 9
        and base.original_donor_form_total == base.whole_form_mean == 0
        and base.common_storage_reference == "same unshifted Theta in Uplus(Theta)"
        and base.complete_storage_rate == "-norm((H+9G)*x)**2/9"
        and base.original_attachment_work_retained_once
        and not any(
            (
                base.state_reset_performed,
                base.original_source_reacquired,
                base.phase_clamped,
                base.prior_finite_endpoint_used_as_initial_state,
                base.independently_fixed_copy_gauges_assumed,
                base.direct_entry_energy_adds_attachment_twice,
            )
        )
    )
    source_cone = (
        source_ok
        and all(margin > 0 for margin in cone_margins)
        and 2 < sine_coefficient.lo <= sine_coefficient.hi < 3
        and Q(4, 9) < coupling.lo <= coupling.hi < Q(1, 2)
        and 4 * coupling.lo > 1
        and 0 < original_comparison < budget
    )
    metric_ok = (
        source_cone
        and base.common_metric_certified
        and base.normalized_metric_lower_bound == normalized_lower > 0
        and base.normalized_metric_upper_bound == normalized_upper
        and base.metric_lower_bound == metric_lower == Q(9, 4) * normalized_lower
        and base.metric_upper_bound == metric_upper == Q(9, 4) * normalized_upper
        and base.shared_metric_decay_rate == base.decay_rate == shared_rate == rate > 0
        and base.shared_phase_scale == 2
        and base.auxiliary_capacity == Q(4, 9)
        and base.metric_cosine_lower_bound == Q(1, 500)
        and base.weighted_gap_lower_bound == 3
        and base.bare_gap_lower_bound == 4
        and base.transport_norm_upper_bound == 36
        and base.metric_formula == "W=(9/4)*(X**2+4*Y**2+8*gamma*dot(xc,yc))"
    )
    prior_domain = (
        metric_ok
        and base.joint_bootstrap_certified
        and base.sharp_source_bound_certified
        and base.weighted_gap_certified
        and base.phase_stiffness_certified
        and base.nominal_centered_form_upper_bound == 4 * _G * qhi
        and base.nominal_centered_phase_upper_bound == Q(9, 4) * qhi
        and base.initial_centered_form_upper_bound == form_initial
        and base.initial_centered_phase_upper_bound == phase_initial
        and base.initial_scaled_norm_squared_upper_bound == initial_square
        and base.scaled_norm_squared_upper_bound == norm_square
        and base.scaled_radius_margin == norm_margin > 0
        and base.scaled_radius_limit == _RADIUS
        and base.relative_phase_excess_limit == _PHASE_EXCESS
        and base.phase_sector_bound == pi / 2 + _PHASE_EXCESS
        and base.internal_edge_cosine_lower_bound == internal_cosine >= Q(1, 10)
        and base.matching_cosine_lower_bound == -Q(1, 400)
        and base.phase_stiffness_lower_bound == phase_stiffness
        and base.phase_stiffness_margin == phase_stiffness - Q(1, 3) > 0
        and base.selected_phase_stiffness_lower_bound == Q(1, 3)
        and base.metric_stiffness_margin == Q(1, 12) - Q(36, 500) > 0
        and base.entry_aggregate_upper_bound == entry_aggregate
        and base.initial_full_storage_upper_bound == entry_storage
        and base.phase_boundary_storage_lower_bound == phase_barrier
        and base.phase_barrier_margin == phase_margin > 0
        and 6 * _ETA < _PHASE_EXCESS
        and not base.boundary_convexity_assumed
        and not base.stiffness_derivative_consumed
        and not base.unequal_internal_blocks_assumed_commuting
    )

    # The four-coordinate reference receives the SAME actual psi(t).
    # It keeps the inherited positions and velocities, rather than replacing
    # them by independently selected source-sized coordinates.
    restoring_margin = Q(1, 9) - _PHASE_EXCESS
    velocity_margins = (
        2 * coupling.lo - sine_coefficient.hi / 9 - 4 * _G**2 * coupling.hi,
        2 * (coupling.lo + 1)
        - sine_coefficient.hi / 9
        - 1
        - 4 * _G**2 * (coupling.hi + 1),
    )
    reference_form = 5 * _G * qhi
    reference_phase = Q(9, 4) * qhi
    reference_cone = (
        prior_domain
        and source_cone
        and restoring_margin > 0
        and all(margin > 0 for margin in velocity_margins)
        and cos(qhi / 2).lo > Q(1, 2)
        and 18 < 25
        and Q(9, 2) < Q(81, 16)
        and reference_phase < Q(1, 1000)
    )
    integral_margin = 1 - (1 / coupling.lo + 1 / (coupling.lo + 1)) / 4
    form_integral = 3 * qhi / _GL
    phase_variation = 3 * qhi
    integrals = reference_cone and integral_margin > 0

    # Every connecting phase segment lies in the admitted sector: both
    # endpoints have phase norm below1/1000, and the actual relative mean
    # is shared. The incremental Hessian is not the old radial Hessian.
    incremental = (
        reference_cone
        and norm_square < _RADIUS**2
        and reference_phase < _RADIUS / 2
        and phase_stiffness > Q(1, 3)
        and Q(1, 12) > Q(36, 500)
        and internal_cosine >= Q(1, 10)
    )
    matrix_error = 14 * _WEIGHT
    initial_scaled_error = 2 * original_comparison
    initial_root = 2 * (1 + _G) * original_comparison
    forcing_integral = matrix_error * qhi / (3 * _GL)
    root_bound = initial_root + forcing_integral
    difference_bound = root_bound / normalized_lower
    phase_error = difference_bound / 2
    edge_error = Q(3, 4) * difference_bound
    forced = (
        incremental
        and metric_ok
        and integrals
        and matrix_error == persistent.weight_matrix_deviation_upper_bound
        and (1 + _G) ** 2 >= normalized_upper
        and 0 < normalized_lower <= 1
        and matrix_error * form_integral / 9 == forcing_integral
        and 2 < Q(9, 4)
        and 0 < root_bound < difference_bound
    )
    marked_shape = len(geometry.marked_pairs) == 9 and all(
        0 <= a < 9 <= b < 18 for a, b in geometry.marked_pairs
    )
    marked_pairs = tuple(
        (a + offset, b + offset) for offset in (0, 18) for a, b in geometry.marked_pairs
    )
    marked_gap = qhi + edge_error
    registration_margin = _DELTA / 4 - marked_gap
    windings = prior_domain and internal_cosine >= Q(1, 10)
    identity = (
        forced
        and windings
        and marked_shape
        and len(set(marked_pairs)) == 18
        and registration_margin > 0
    )
    phase_work = 972 * _ETA + Q(9, 2) * entry**3
    aligned_jump = (Q(0), 9 * (entry + 18 * _ETA) ** 2)
    registered_jump = (162 - phase_work, 162 + phase_work + 9 * entry**2)
    work = (
        source_ok
        and base.phase_switching_work_allowance == phase_work
        and base.aligned_storage_jump_bounds == aligned_jump
        and base.registered_storage_jump_bounds == registered_jump
        and 0 <= aligned_jump[0] < aligned_jump[1]
        and registered_jump[0] > 0
    )
    admitted = max(registered_noise, aligned_noise) <= _NOISE
    certified = identity and work and admitted
    unresolved = tuple(
        label
        for good, label in (
            (source_ok, "original_source_or_complete_history_unavailable"),
            (source_cone, "original_correlated_phase_and_velocity_cone_unavailable"),
            (metric_ok, "shared_incremental_metric_coefficients_unavailable"),
            (
                prior_domain,
                "actual_joint_centered_and_relative_phase_domain_unavailable",
            ),
            (reference_cone, "same_psi_correlated_reference_cone_unavailable"),
            (integrals, "finite_reference_form_integral_unavailable"),
            (incremental, "actual_reference_segment_stiffness_unavailable"),
            (forced, "source_paid_correlated_transport_comparison_unavailable"),
            (identity, "uninterrupted_strict_marked_identity_not_established"),
            (work, "original_attachment_work_association_unavailable"),
            (admitted, "original_observation_allowances_outside_contract"),
        )
        if not good
    )
    return _MixedMarkedIdentity(
        form_error_bounds=forms,
        phase_error_bounds=phases,
        registered_observation_error_bound=registered_noise,
        aligned_observation_error_bound=aligned_noise,
        source=signed_source,
        source_association_certified=source_ok,
        registration_covariance_certified=source_ok,
        prior_domain_certified=prior_domain,
        source_cone_certified=source_cone,
        reference_cone_certified=reference_cone,
        reference_integrals_certified=integrals,
        incremental_stiffness_certified=incremental,
        common_metric_certified=metric_ok,
        forced_comparison_certified=forced,
        strict_marked_identity_certified=identity,
        cycle_windings_certified=windings,
        switching_work_certified=work,
        observation_allowances_admitted=admitted,
        gamma_bounds=gamma,
        coupling_bounds=coupling,
        sine_coefficient_bounds=sine_coefficient,
        nominal_q0_bounds=(qlo, qhi),
        source_comparison_upper_bound=original_comparison,
        centered_entry_norm_upper_bound=entry,
        inherited_output_error_upper_bound=inherited_error,
        slow_amplitude_bounds=amplitude,
        initial_scaled_norm_squared_upper_bound=initial_square,
        scaled_norm_squared_upper_bound=norm_square,
        scaled_radius_margin=norm_margin,
        phase_sector_bound=pi / 2 + _PHASE_EXCESS,
        initial_full_storage_upper_bound=entry_storage,
        phase_boundary_storage_lower_bound=phase_barrier,
        phase_barrier_margin=phase_margin,
        normalized_metric_lower_bound=normalized_lower,
        normalized_metric_upper_bound=normalized_upper,
        metric_lower_bound=metric_lower,
        metric_upper_bound=metric_upper,
        shared_metric_decay_rate=shared_rate,
        decay_rate=rate,
        reference_restoring_lower_margin=restoring_margin,
        reference_velocity_cone_margins=velocity_margins,
        reference_form_upper_bound=reference_form,
        reference_phase_upper_bound=reference_phase,
        reference_form_integral_upper_bound=form_integral,
        reference_phase_variation_upper_bound=phase_variation,
        reference_integral_coefficient_margin=integral_margin,
        incremental_stiffness_lower_bound=phase_stiffness,
        weight_matrix_deviation_upper_bound=matrix_error,
        initial_scaled_error_upper_bound=initial_scaled_error,
        initial_metric_root_upper_bound=initial_root,
        forcing_integral_upper_bound=forcing_integral,
        metric_root_upper_bound=root_bound,
        scaled_error_upper_bound=difference_bound,
        phase_error_upper_bound=phase_error,
        marked_edge_error_upper_bound=edge_error,
        nominal_marked_gap_upper_bound=qhi,
        marked_gap_upper_bound=marked_gap,
        registration_margin=registration_margin,
        marked_pairs=marked_pairs,
        phase_switching_work_allowance=phase_work,
        aligned_storage_jump_bounds=aligned_jump,
        registered_storage_jump_bounds=registered_jump,
        certified=certified,
        status="conditional_all_time_marked_identity" if certified else "unavailable",
        unresolved_conditions=unresolved,
        all_postattachment_times_covered=identity,
        original_finite_identity_contract_extended=identity,
    )
