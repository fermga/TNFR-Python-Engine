"""Relative registration as supplied state for one reciprocal interaction.

A quarter-turn translates the recipient's complete preceding history. The
existing phase-sensitive field is unchanged. Fresh centered-state, switching
work and output bounds price the new relative mean without resetting it or
placing it inside the previous small global phase chart.
"""

from dataclasses import dataclass
from fractions import Fraction as Q

from ..mathematics._rational_interval import I, pi_interval
from ._mixed_phase_sensitive_interface import (
    _assess_mixed_phase_sensitive_interface,
    _MixedPhaseSensitiveInterface,
)
from ._mixed_reciprocal_partner import _nonnegative, _nonnegative_pair
from ._mixed_registered_phase_probe import _derive_registered_phase_probe_geometry
from ._mixed_resource_transfer_obstruction import _original_source_matches

_DELTA = Q(1, 1024)
_WEIGHT, _ETA, _MATURE = Q(2, 10**60), Q(2, 10**69), Q(1, 2**28)
_SECTION = 3 * _DELTA / 16
_SECTION_ERROR = Q(3, 2) * (14 * _ETA + 2 * 10**10 * _WEIGHT)
_GL, _G = Q(1, 3300), Q(1, 3000)
_ACTIVATION, _SWITCH, _DURATION = Q(2010000), Q(3375000), Q(8)
_DECAY = Q(1, 2**16)
_NOISE, _THRESHOLD = Q(1, 10**9), Q(1, 10**8)
_PAIRS = tuple((j, 18 + j) for j in range(18))


@dataclass(frozen=True)
class _MixedRelativeRegistration:
    """A conditional registration-aware prediction and a specific blind failure."""

    form_error_bounds: tuple[Q, Q]
    phase_error_bounds: tuple[Q, Q]
    registered_observation_error_bound: Q
    aligned_observation_error_bound: Q
    source: _MixedPhaseSensitiveInterface
    source_association_certified: bool
    registration_covariance_certified: bool
    centered_identity_certified: bool
    relative_phase_drift_certified: bool
    phase_moment_remainder_certified: bool
    switching_work_certified: bool
    affine_prediction_certified: bool
    blind_prediction_obstruction_certified: bool
    common_prediction_obstruction_certified: bool
    observation_allowances_admitted: bool
    registered_record_certified: bool
    aligned_record_certified: bool
    independent_record_sets_separated: bool
    fixed_threshold_certified: bool
    gamma_bounds: I
    coupling_bounds: I
    nominal_q0_bounds: tuple[Q, Q]
    slow_amplitude_bounds: tuple[Q, Q]
    registration_angle_bounds: I
    inherited_output_error_upper_bound: Q
    initial_relative_phase_error_upper_bound: Q
    centered_entry_norm_upper_bound: Q
    centered_norm_growth_denominator: Q
    centered_norm_upper_bound: Q
    phase_drift_upper_bound: Q
    marked_gap_upper_bound: Q
    registration_margin: Q
    entry_aggregate_upper_bound: Q
    aggregate_absolute_upper_bound: Q
    relative_phase_deviation_upper_bound: Q
    boundary_branch_margin: Q
    centered_moment_remainder_upper_bound: Q
    phase_moment_deficit_upper_bound: Q
    phase_switching_work_allowance: Q
    storage_jump_bounds: tuple[Q, Q]
    affine_initializer_error_upper_bound: Q
    affine_prediction_error_upper_bound: Q
    affine_prediction_margin: Q
    aligned_scalar_effect_upper_bound: Q
    reconstructed_aligned_return_upper_bound: Q
    registered_true_bounds: tuple[Q, Q]
    aligned_true_bounds: tuple[Q, Q]
    registered_record_bounds: tuple[Q, Q]
    aligned_record_bounds: tuple[Q, Q]
    record_separation_margin: Q
    registered_threshold_margin: Q
    aligned_threshold_margin: Q
    blind_prediction_error_lower_bound: Q
    blind_prediction_margin: Q
    true_response_separation_lower_bound: Q
    common_prediction_obstruction_margin: Q
    certified: bool
    status: str
    unresolved_conditions: tuple[str, ...]
    registration_turns: Q = Q(1, 4)
    translated_copy: str = "recipient"
    preparation: str = "recipient whole precontact phase history translated by +pi/2"
    switch_age: Q = _SWITCH
    post_duration: Q = _DURATION
    activation_time: Q = _ACTIVATION
    switch_time: Q = _ACTIVATION + _SWITCH
    total_duration: Q = _ACTIVATION + _SWITCH + _DURATION
    original_clock_horizon: Q = (_ACTIVATION + _SWITCH + _DURATION) * Q(1024, 1023)
    full_contact_pairs: tuple[tuple[int, int], ...] = _PAIRS
    physical_capacities: tuple[Q, ...] = (Q(1),) * 36
    full_state_dimension: int = 72
    internal_divisor: Q = Q(9)
    scalar_decay_upper_bound: Q = _DECAY
    kernel_mass_upper_bound: Q = Q(1, 2)
    original_donor_form_total: Q = Q(0)
    whole_form_mean: Q = Q(0)
    registered_observation_error_cap: Q = _NOISE
    aligned_observation_error_cap: Q = _NOISE
    prediction_tolerance: Q = _NOISE
    threshold: Q = _THRESHOLD
    aggregate_equation: str = "Q_prime=-2*Q-gamma*M"
    relative_phase_equation: str = "psi_prime=2*gamma*Q/9"
    phase_moment_decomposition: str = "M=18*sin(psi)+R; abs(R)<=Nc**2"
    affine_equation: str = "qhat_prime=-2*qhat-18*gamma"
    affine_predictor: str = "Q_s(3375000)*exp(-2*s)-9*gamma*(1-exp(-2*s))"
    blind_predictor: str = "Q_s(3375000)*exp(-2*s)"
    common_prediction_contract: str = (
        "same complete prior form history and postlaw, relative registration discarded"
    )
    common_storage_reference: str = "same unshifted Theta in Uplus(Theta)"
    comparison_storage_rate: str = "-norm((H+9G)*x)**2/9"
    complete_precontact_history_translated: bool = True
    source_errors_translated: bool = False
    original_source_report_mutated: bool = False
    source_covariance_is_conditional: bool = True
    internal_identity_and_winding_unchanged_by_translation: bool = True
    prior_form_observations_unchanged_by_translation: bool = True
    same_complete_contact_law: bool = True
    existing_field_reused_unchanged: bool = True
    moving_copy_centering_not_fixed_gauges: bool = True
    old_global_acute_ball_reused: bool = False
    boundary_convexity_assumed: bool = False
    boundary_phase_clamped: bool = False
    initial_relative_phase_exactly_quarter_turn: bool = False
    aligned_counterpart_retains_same_underlying_source: bool = True
    original_baseline_retained: bool = True
    original_lifts_and_error_association_retained: bool = True
    supplied_registration_retained_by_affine_predictor: bool = True
    all_scalar_predictions_excluded: bool = False
    dynamically_resolved_phase_coordinate_required: bool = False
    exact_two_coordinate_closure: bool = False
    independent_record_errors: bool = True
    observation_error_cancellation_assumed: bool = False
    all_future_identity_claimed: bool = False
    reference_shift_used_to_erase_switch_work: bool = False
    zero_physical_switching_cost_claimed: bool = False
    later_phase_intervention_performed: bool = False
    state_reset_performed: bool = False
    original_source_reacquired: bool = False
    original_source_replayed: bool = False
    original_source_trajectory_evaluated: bool = False
    registration_selected_after_response: bool = False
    runtime_event_executed: bool = False
    operator_occurrence_derived: bool = False
    exponentials_evaluated: bool = False
    roots_numerically_evaluated: bool = False
    newly_observed_aggregates: int = 0
    required_endpoint_records: int = 2
    numerical_response_allowance: Q = Q(0)
    physical_property_identified: bool = False
    empirical_validation: bool = False
    clock: str = "tau=(1023/1024)*t; translated history and all reciprocal contact rows"


def _assess_mixed_relative_registration(
    *,
    form_error_bounds,
    phase_error_bounds,
    registered_observation_error_bound,
    aligned_observation_error_bound,
):
    """Admit a fixed whole-history translation and its finite task bounds.

    The original source is rebuilt unchanged. Translation is an exact
    precontact covariance construction, not a replacement report claiming
    that the translated origin was the original measured one. Every bound
    used below is reconstructed; old global and endpoint verdicts are unused.
    """
    forms = _nonnegative_pair(form_error_bounds, "form_error_bounds")
    phases = _nonnegative_pair(phase_error_bounds, "phase_error_bounds")
    registered_noise = _nonnegative(
        registered_observation_error_bound, "registered_observation_error_bound"
    )
    aligned_noise = _nonnegative(
        aligned_observation_error_bound, "aligned_observation_error_bound"
    )
    source = _assess_mixed_phase_sensitive_interface(
        form_error_bounds=forms,
        phase_error_bounds=phases,
        phase_sensitive_observation_error_bound=Q(0),
        form_only_observation_error_bound=Q(0),
    )
    renewed = source.source
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
    source_ok = (
        source.source_association_certified
        and source.relative_phase_registration_certified
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
        and source.form_error_bounds
        == renewed.form_error_bounds
        == slow.form_error_bounds
        == forms
        and source.phase_error_bounds
        == renewed.phase_error_bounds
        == slow.phase_error_bounds
        == phases
        and source.phase_sensitive_observation_error_bound
        == source.form_only_observation_error_bound
        == 0
        and source.nominal_q0_bounds == renewed.nominal_q0_bounds == (qlo, qhi)
        and source.gamma_bounds == renewed.gamma_bounds == gamma
        and source.coupling_bounds == renewed.coupling_bounds == coupling
        and _GL < gamma.lo <= gamma.hi < _G
        and Q(4, 9) < coupling.lo <= coupling.hi < Q(1, 2)
        and source.inherited_output_error_upper_bound
        == renewed.inherited_output_error_upper_bound
        == slow.total_output_error_upper_bound
        == inherited_error
        and source.slow_amplitude_bounds
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
        and source.phase_mean_error_upper_bound == 3 * _ETA
        and source.relative_phase_entry_error_upper_bound == 6 * _ETA
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
        and source.original_donor_form_total == renewed.original_donor_form_total == 0
        and source.whole_form_mean == persistent.whole_form_mean == 0
        and renewed.whole_form_mean
        == collective.whole_form_mean
        == storage.whole_form_mean
        == positive.whole_form_mean
        == 0
        and positive.exact_zero_entry_total_from_prior_laws
        and source.physical_capacities == persistent.physical_capacities == (Q(1),) * 36
        and source.full_state_dimension == persistent.full_state_dimension == 72
        and source.full_contact_pairs == renewed.full_contact_pairs == _PAIRS
        and source.internal_divisor == 9
        and source.aggregate_equation == "Q_prime=-2*Q-gamma*M"
        and source.relative_phase_equation == "psi_prime=2*gamma*Q/9"
        and source.original_phase_lifts_retained
        and source.original_baseline_retained
        and source.inherited_amplitude_rate_correlation_retained
        and source.same_actual_member_comparator
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
        == source.activation_time
        == _ACTIVATION
        and positive.contact_pairs
        == persistent.contact_pairs
        == tuple((j, 18 + j) for j in range(9))
        and persistent.fixed_port_strength == storage.fixed_port_strength == 1
        and handoff.switch_word == (1, 0)
        and source.switch_age == renewed.switch_age == slow.early_age == _SWITCH
        and source.switch_time == _ACTIVATION + _SWITCH
        and source.post_duration == _DURATION
        and source.total_duration == _ACTIVATION + _SWITCH + _DURATION
        and source.original_clock_horizon
        == (_ACTIVATION + _SWITCH + _DURATION) * Q(1024, 1023)
        and source.clock
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
                source.state_reset_performed,
                source.original_source_reacquired,
                source.observation_rebased_at_switch,
                renewed.state_reset_performed,
                slow.state_reset_performed,
                positive.state_reset_performed,
                handoff.state_reset_performed,
            )
        )
    )
    # Both old internal rows use within-copy differences; all old external
    # inputs consume forms only. Translate the entire recipient history and
    # its lifts, retaining the same original source and uncertainty predicate.
    covariance = source_ok and source.initial_relative_phase_mean_assumed_zero is False
    initial_phase_error = 6 * _ETA
    entry = 4 * qhi + budget
    denominator = 1 - 4 * _G * _DURATION
    norm = entry / denominator
    phase_drift = 6 * _G * _DURATION * norm
    marked = qhi + 300 * _MATURE + phase_drift
    registration_margin = _DELTA / 4 - marked
    identity = (
        covariance
        and denominator > 0
        and norm < Q(1, 100) < Q(1, 32)
        and registration_margin > 0
    )
    entry_aggregate = amplitude[1] + inherited_error
    aggregate = entry_aggregate + 9 * _G
    phase_deviation = initial_phase_error + 2 * _G * _DURATION * aggregate / 9
    branch_margin = Q(1, 4) - phase_deviation - 2 * norm
    phase_ok = identity and branch_margin > 0
    moment_remainder = norm**2
    deficit = 9 * phase_deviation**2 + moment_remainder
    moment_ok = phase_ok and 0 < deficit < 18
    # The common comparison reference is unchanged. Quarter-turn matching
    # contributes 162 at the reference carrier; the original errors are paid.
    phase_work = 972 * _ETA + Q(9, 2) * entry**3
    jump = (162 - phase_work, 162 + phase_work + 9 * entry**2)
    work_ok = covariance and identity and 0 < jump[0] <= jump[1]
    initializer_error = inherited_error * _DECAY
    affine_error = initializer_error + _G * deficit / 2
    affine_margin = _NOISE - affine_error
    affine_ok = moment_ok and work_ok and affine_margin > 0
    registered_upper = (
        -9 * _GL + (entry_aggregate + 9 * _GL) * _DECAY + _G * deficit / 2
    )
    registered_true = (-9 * _G, registered_upper)

    # Reconstruct the completed aligned task from the original source. Its
    # near-zero relative gauge has its own small joint chart and cubic defect;
    # those bounds are never applied to the quarter-turn state.
    aligned_norm = (entry + 18 * _ETA) / denominator
    aligned_remainder = aligned_norm**3 / 2
    aligned_forcing = _G * _DURATION * aligned_remainder
    aligned_aggregate = amplitude[1] + inherited_error + 54 * _ETA + aligned_forcing
    aligned_effect = (
        54 * _G * _ETA
        + 2 * _G**2 * _DURATION * aligned_aggregate
        + _G * aligned_remainder / 2
    )
    form_upper = entry_aggregate * _DECAY
    aligned_upper = form_upper + aligned_effect
    aligned_true = (-aligned_effect, aligned_upper)
    aligned_identity = (
        source_ok
        and aligned_norm < Q(1, 100)
        and qhi + 300 * _MATURE + 6 * _G * _DURATION * aligned_norm < _DELTA / 4
    )
    aligned_ok = aligned_identity and aligned_upper < _NOISE
    blind_lower = -registered_upper
    blind_margin = blind_lower - _NOISE
    blind = affine_ok and amplitude[0] > 0 and blind_margin > 0
    true_separation = -aligned_effect - registered_upper
    common_margin = true_separation - 2 * _NOISE
    # Identical retained form histories yield one common prediction when
    # registration is discarded. The triangle inequality prices both errors.
    common_obstruction = affine_ok and aligned_ok and common_margin > 0
    registered_record = (
        registered_true[0] - registered_noise,
        registered_upper + registered_noise,
    )
    aligned_record = (aligned_true[0] - aligned_noise, aligned_upper + aligned_noise)
    admitted = max(registered_noise, aligned_noise) <= _NOISE
    registered_threshold = -_THRESHOLD - registered_record[1]
    aligned_threshold = _THRESHOLD - max(abs(value) for value in aligned_record)
    separation = aligned_record[0] - registered_record[1]
    registered_record_ok = affine_ok and admitted and registered_threshold > 0
    aligned_record_ok = aligned_ok and admitted and aligned_threshold > 0
    separated = registered_record_ok and aligned_record_ok and separation > 0
    threshold = separated and min(registered_threshold, aligned_threshold) > 0
    certified = threshold and blind and work_ok and identity
    unresolved = tuple(
        label
        for good, label in (
            (source_ok, "original_source_or_inherited_bounds_unavailable"),
            (covariance, "whole_history_registration_covariance_unavailable"),
            (identity, "fresh_centered_identity_unavailable"),
            (phase_ok, "relative_phase_drift_or_branch_unavailable"),
            (work_ok, "switching_work_not_certified"),
            (affine_ok, "affine_prediction_tolerance_not_certified"),
            (blind, "specified_blind_predictor_obstruction_not_certified"),
            (admitted, "independent_observation_errors_outside_protocol"),
            (threshold, "registered_and_aligned_records_not_separated"),
        )
        if not good
    )
    return _MixedRelativeRegistration(
        form_error_bounds=forms,
        phase_error_bounds=phases,
        registered_observation_error_bound=registered_noise,
        aligned_observation_error_bound=aligned_noise,
        source=source,
        source_association_certified=source_ok,
        registration_covariance_certified=covariance,
        centered_identity_certified=identity,
        relative_phase_drift_certified=phase_ok,
        phase_moment_remainder_certified=moment_ok,
        switching_work_certified=work_ok,
        affine_prediction_certified=affine_ok,
        blind_prediction_obstruction_certified=blind,
        common_prediction_obstruction_certified=common_obstruction,
        observation_allowances_admitted=admitted,
        registered_record_certified=registered_record_ok,
        aligned_record_certified=aligned_record_ok,
        independent_record_sets_separated=separated,
        fixed_threshold_certified=threshold,
        gamma_bounds=gamma,
        coupling_bounds=coupling,
        nominal_q0_bounds=(qlo, qhi),
        slow_amplitude_bounds=amplitude,
        registration_angle_bounds=pi / 2,
        inherited_output_error_upper_bound=inherited_error,
        initial_relative_phase_error_upper_bound=initial_phase_error,
        centered_entry_norm_upper_bound=entry,
        centered_norm_growth_denominator=denominator,
        centered_norm_upper_bound=norm,
        phase_drift_upper_bound=phase_drift,
        marked_gap_upper_bound=marked,
        registration_margin=registration_margin,
        entry_aggregate_upper_bound=entry_aggregate,
        aggregate_absolute_upper_bound=aggregate,
        relative_phase_deviation_upper_bound=phase_deviation,
        boundary_branch_margin=branch_margin,
        centered_moment_remainder_upper_bound=moment_remainder,
        phase_moment_deficit_upper_bound=deficit,
        phase_switching_work_allowance=phase_work,
        storage_jump_bounds=jump,
        affine_initializer_error_upper_bound=initializer_error,
        affine_prediction_error_upper_bound=affine_error,
        affine_prediction_margin=affine_margin,
        aligned_scalar_effect_upper_bound=aligned_effect,
        reconstructed_aligned_return_upper_bound=aligned_upper,
        registered_true_bounds=registered_true,
        aligned_true_bounds=aligned_true,
        registered_record_bounds=registered_record,
        aligned_record_bounds=aligned_record,
        record_separation_margin=separation,
        registered_threshold_margin=registered_threshold,
        aligned_threshold_margin=aligned_threshold,
        blind_prediction_error_lower_bound=blind_lower,
        blind_prediction_margin=blind_margin,
        true_response_separation_lower_bound=true_separation,
        common_prediction_obstruction_margin=common_margin,
        certified=certified,
        status=(
            "conditional_relative_registration_resolved" if certified else "unavailable"
        ),
        unresolved_conditions=unresolved,
    )
