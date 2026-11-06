"""Source-inclusive collective envelopes from signed lifted sensitivity.

The effective phase evolves independently. A different, same-actual-phase
reference is used only to bound the discarded internal moment. Conditional
uniform bounds and the two original accuracy targets have separate verdicts.
"""

from dataclasses import dataclass
from fractions import Fraction as Q

from ..mathematics._rational_interval import I, cos, pi_interval
from ._mixed_marked_identity import _assess_mixed_marked_identity, _MixedMarkedIdentity
from ._mixed_reciprocal_partner import _nonnegative, _nonnegative_pair
from ._mixed_registered_phase_probe import _derive_registered_phase_probe_geometry
from ._mixed_resource_transfer_obstruction import _original_source_matches
from ._sine_lyapunov import _sine_common_quadratic_coefficients

_DELTA, _ETA, _WEIGHT = Q(1, 1024), Q(2, 10**69), Q(2, 10**60)
_MATURE, _GL, _G = Q(1, 2**28), Q(1, 3300), Q(1, 3000)
_NOISE, _EPSILON = Q(1, 10**9), Q(1, 100)
_ACTIVATION, _SWITCH = Q(2010000), Q(3375000)
_PAIRS = tuple((j, 18 + j) for j in range(18))


@dataclass(frozen=True)
class _MixedCollectiveSensitivity:
    """A uniform envelope can be valid without establishing both targets."""

    form_error_bounds: tuple[Q, Q]
    phase_error_bounds: tuple[Q, Q]
    registered_observation_error_bound: Q
    aligned_observation_error_bound: Q
    source: _MixedMarkedIdentity
    source_association_certified: bool
    complete_laws_certified: bool
    prior_domain_certified: bool
    reference_cone_certified: bool
    incremental_comparison_certified: bool
    individual_rectangles_certified: bool
    central_entry_certified: bool
    signed_sensitivity_certified: bool
    residual_decomposition_certified: bool
    reference_l2_certified: bool
    error_l2_certified: bool
    residual_integral_certified: bool
    uniform_envelope_certified: bool
    form_accuracy_certified: bool
    phase_accuracy_certified: bool
    switching_work_certified: bool
    observation_allowances_admitted: bool
    gamma_bounds: I
    coupling_bounds: I
    sine_coefficient_bounds: I
    nominal_q0_bounds: tuple[Q, Q]
    source_comparison_upper_bound: Q
    inherited_output_error_upper_bound: Q
    slow_amplitude_bounds: tuple[Q, Q]
    centered_entry_norm_upper_bound: Q
    initial_scaled_norm_squared_upper_bound: Q
    scaled_norm_squared_upper_bound: Q
    decay_rate: Q
    normalized_metric_lower_bound: Q
    normalized_metric_upper_bound: Q
    metric_root_upper_bound: Q
    initial_aggregate_upper_bound: Q
    aggregate_absolute_upper_bound: Q
    phase_moment_upper_bound: Q
    old_moment_integral_upper_bound: Q
    initial_relative_phase_error_upper_bound: Q
    initial_lift_excess_upper_bound: Q
    lift_monotonicity_margin: Q
    rectangle_initial_margin: Q
    first_unit_phase_displacement_upper_bound: Q
    central_entry_margin: Q
    pre_entry_row_sum_upper_bound: Q
    sensitivity_gain_denominator: Q
    sensitivity_gain_upper_bound: Q
    reference_restoring_lower_bound: Q
    reference_sinc_lower_bound: Q
    reference_phase_squared_integral_upper_bound: Q
    comparison_phase_squared_integral_upper_bound: Q
    moment_integral_upper_bound: Q
    moment_integral_improvement_margin: Q
    initial_lift_error_upper_bound: Q
    uniform_phase_error_upper_bound: Q
    uniform_form_error_upper_bound: Q
    form_accuracy_margin: Q
    phase_accuracy_margin: Q
    registered_form_record_error_upper_bound: Q
    aligned_form_record_error_upper_bound: Q
    phase_switching_work_allowance: Q
    aligned_storage_jump_bounds: tuple[Q, Q]
    registered_storage_jump_bounds: tuple[Q, Q]
    certified: bool
    status: str
    unresolved_conditions: tuple[str, ...]
    fixed_young_epsilon: Q = _EPSILON
    analytical_entry_time: Q = Q(1)
    intrinsic_form_tolerance: Q = _NOISE
    intrinsic_phase_tolerance: Q = _NOISE
    initial_registrations_turns: tuple[Q, Q] = (Q(0), Q(1, 4))
    activation_time: Q = _ACTIVATION
    switch_age: Q = _SWITCH
    switch_time: Q = _ACTIVATION + _SWITCH
    original_clock_start_time: Q = (_ACTIVATION + _SWITCH) * Q(1024, 1023)
    previous_certified_duration: Q = Q(4096)
    full_state_dimension: int = 72
    reduced_state_dimension: int = 2
    physical_capacities: tuple[Q, ...] = (Q(1),) * 36
    full_contact_pairs: tuple[tuple[int, int], ...] = _PAIRS
    internal_divisor: Q = Q(9)
    whole_form_mean: Q = Q(0)
    original_donor_form_total: Q = Q(0)
    initial_reduced_form: str = "same source-correlated Q_s(3375000)"
    reduced_state_coordinates: tuple[str, str] = ("Q", "psi")
    actual_form_equation: str = "Q_prime=-2*Q-18*gamma*sin(psi)-gamma*R"
    actual_phase_equation: str = "psi_prime=2*gamma*Q/9"
    effective_form_equation: str = "Qhat_prime=-2*Qhat-18*gamma*sin(psihat)"
    effective_phase_equation: str = "psihat_prime=2*gamma*Qhat/9"
    collective_error_coordinates: tuple[str, str] = ("(Q-Qhat)/9", "psi-psihat")
    sensitivity_coordinates: tuple[str, str] = ("b", "b+2*gamma*a")
    sensitivity_matrix: str = "[[-1,1],[1-4*gamma**2*kappa,-1]]"
    sensitivity_forcing: str = "(0,-2*gamma**2*R/9)"
    residual_decomposition: str = "R=-A*sin(psi)+Bsin*cos(psi)"
    reference_moment: str = "Rbar=-18*sin(psi)*(1-cos(z/2))"
    common_storage_reference: str = "same unshifted Theta in Uplus(Theta)"
    complete_storage_rate: str = "-norm((H+9G)*x)**2/9"
    all_postattachment_times_covered: bool = False
    original_uniform_prediction_tolerance_certified: bool = False
    same_complete_and_effective_laws: bool = True
    same_law_for_both_preparations: bool = True
    original_source_error_paid_once: bool = True
    source_correlated_initialization_retained: bool = True
    whole_history_covariance_retained: bool = True
    original_attachment_work_retained_once: bool = True
    actual_phase_supplied_to_predictor: bool = False
    analytical_reference_is_autonomous_predictor: bool = False
    previous_finite_guarantees_replaced: bool = False
    centered_stiffness_used_as_collective_stiffness: bool = False
    uniform_positive_collective_cosine_assumed: bool = False
    stiffness_derivative_consumed: bool = False
    original_target_relaxed: bool = False
    actual_prediction_failure_claimed: bool = False
    actual_collective_instability_claimed: bool = False
    exact_actual_two_coordinate_closure: bool = False
    new_field_installed: bool = False
    new_memory_state_installed: bool = False
    state_reset_performed: bool = False
    phase_clamped: bool = False
    original_source_reacquired: bool = False
    original_source_replayed: bool = False
    source_trajectory_evaluated: bool = False
    reduced_trajectory_evaluated: bool = False
    new_endpoint_or_phase_record_claimed: bool = False
    runtime_event_executed: bool = False
    exponentials_evaluated: bool = False
    newly_observed_aggregates: int = 0
    newly_observed_phases: int = 0
    physical_property_identified: bool = False
    empirical_validation: bool = False
    clock: str = "tau=(1023/1024)*t; unchanged actual and independent effective rows"


def _assess_mixed_collective_sensitivity(
    *,
    form_error_bounds,
    phase_error_bounds,
    registered_observation_error_bound,
    aligned_observation_error_bound,
):
    """Rebuild source bounds before evaluating the two independent targets."""
    forms = _nonnegative_pair(form_error_bounds, "form_error_bounds")
    phases = _nonnegative_pair(phase_error_bounds, "phase_error_bounds")
    registered_noise = _nonnegative(
        registered_observation_error_bound, "registered_observation_error_bound"
    )
    aligned_noise = _nonnegative(
        aligned_observation_error_bound, "aligned_observation_error_bound"
    )
    source = _assess_mixed_marked_identity(
        form_error_bounds=forms,
        phase_error_bounds=phases,
        registered_observation_error_bound=Q(0),
        aligned_observation_error_bound=Q(0),
    )
    signed = source.source
    sustained = signed.source
    effective = sustained.source
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
    registered_geometry = _derive_registered_phase_probe_geometry()
    geometry = registered_geometry.successor
    pi = pi_interval()
    gamma = 1 / (1023 * pi)
    coupling = Q(13, 27) - _DELTA / (18 * pi)
    sine_coefficient = 1 + 2 * cos(pi / 9)
    section = 3 * _DELTA / 16
    section_error = Q(3, 2) * (14 * _ETA + 2 * 10**10 * _WEIGHT)
    qlo, qhi = section - section_error / 2, section + section_error / 2
    budget = 200 * _MATURE
    cut_error = 8 * _MATURE / (1 + Q(10000, 3)) + 48 * _G * _MATURE + 6 * _WEIGHT
    force = 2 * _G * budget / 9 + 5 * _G * budget**2 / 81 + 14 * _WEIGHT / 2700
    output_error = 3 * cut_error + Q(81, 8) * force
    inherited_error = output_error + 3 * _G * qhi**3 + 27 * _G * qhi / 2**16
    amplitude = (
        Q(4, 3) * _GL * Q(233, 300) * qlo * (1 - _G**2 / Q(4, 9)),
        16 * _G * qhi,
    )
    comparison = (
        3 * (24 * _MATURE + 6 * 14 * _WEIGHT * 10 * qhi / _GL) / (1 - 135 * qhi)
    )
    entry = 4 * qhi + budget

    # The fresh owner establishes the inherited theorem. Its primitive source
    # association, complete law and the exact bounds consumed here are checked
    # independently; its global identity/forecast verdicts are not consulted.
    source_ok = (
        source.source_association_certified
        and source.registration_covariance_certified
        and source.form_error_bounds == forms
        and source.phase_error_bounds == phases
        and source.registered_observation_error_bound
        == source.aligned_observation_error_bound
        == 0
        and source.gamma_bounds == gamma
        and source.coupling_bounds == coupling
        and source.sine_coefficient_bounds == sine_coefficient
        and source.nominal_q0_bounds == (qlo, qhi)
        and source.source_comparison_upper_bound == comparison
        and source.inherited_output_error_upper_bound == inherited_error
        and source.slow_amplitude_bounds == amplitude
        and source.centered_entry_norm_upper_bound == entry
        and _GL < gamma.lo <= gamma.hi < _G
        and Q(4, 9) < coupling.lo <= coupling.hi < Q(1, 2)
        and 2 < sine_coefficient.lo <= sine_coefficient.hi < 3
        and 0 < comparison < budget
        and 0 < inherited_error < _NOISE
        and amplitude[0] / 2 > inherited_error
        and geometry == reference.source_geometry == prefix.source_geometry
        and registered_geometry.cross_symmetry_certified
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
        and reference.donor.section_mean_gap == reference.section_mean_gap == section
        and reference.donor.comparison_error_upper_bound
        == reference.donor_section_comparison_error
        == section_error
        and reference.recipient_mature_radius == _MATURE
        and reference.recipient_isolated_deadline
        == reference.recipient_readiness.waiting_duration
        == 10**10
        and reference.recipient_readiness.derived_mature_norm_upper_bound == _MATURE
        and reference.initial_form_means == handoff.retained_form_means == (Q(0), Q(0))
        and phase.phase_mean_error_upper_bound == 3 * _ETA
        and phase.relative_phase_entry_error_upper_bound == 6 * _ETA
        and phase.original_phase_lifts_retained
        and phase.original_baseline_retained
        and registration.registration_turns == Q(1, 4)
        and registration.registration_angle_bounds == pi / 2
        and registration.complete_precontact_history_translated
        and registration.prior_form_observations_unchanged_by_translation
        and registration.aligned_counterpart_retains_same_underlying_source
        and registration.original_lifts_and_error_association_retained
        and positive.exact_zero_entry_total_from_prior_laws
        and positive.prefix_full_duration == handoff.prefix_duration == 2000000
        and prefix.contact_duration == 2000000
        and positive.prefix_cut_duration == handoff.continuation_duration == 10000
        and positive.handoff_time == handoff.total_duration == _ACTIVATION
        and handoff.switch_word == (1, 0)
        and slow.early_age == phase.switch_age == _SWITCH
        and slow.same_actual_history_required
        and slow.correlated_initialization_retained
        and slow.tangent_roots_certified
        and slow.signed_kernel_certified
        and slow.nonlinear_error_certified
        and slow.fast_tail_certified
        and slow.amplitude_certified
        and slow.original_output_error_upper_bound == output_error
        and slow.nonlinear_output_error_upper_bound == 3 * _G * qhi**3
        and slow.fast_output_tail_upper_bound == 27 * _G * qhi / 2**16
        and slow.total_output_error_upper_bound == inherited_error
        and slow.slow_amplitude_bounds == amplitude
        and collective.cut_form_error_upper_bound == cut_error
        and persistent.bregman_comparison_certified
        and persistent.phase_convexity_certified
        and persistent.bregman_growth_integral_upper_bound == 135 * qhi < 1
        and persistent.bregman_denominator == 1 - 135 * qhi
        and persistent.uniform_comparison_upper_bound == comparison
        and persistent.original_phase_gauges_and_branches_retained
        and storage.storage_domain_certified
        and storage.actual_dissipation_certified
        and storage.selected_mu == persistent.selected_mu == 1
        and storage.mean_projector_rank == 1
    )
    common_contract = {
        "initial_registrations_turns": (Q(0), Q(1, 4)),
        "activation_time": _ACTIVATION,
        "switch_age": _SWITCH,
        "switch_time": _ACTIVATION + _SWITCH,
        "full_state_dimension": 72,
        "physical_capacities": (Q(1),) * 36,
        "full_contact_pairs": _PAIRS,
        "internal_divisor": Q(9),
        "whole_form_mean": Q(0),
        "original_donor_form_total": Q(0),
        "source_correlated_initialization_retained": True,
        "whole_history_covariance_retained": True,
        "state_reset_performed": False,
        "phase_clamped": False,
        "original_source_reacquired": False,
        "common_storage_reference": "same unshifted Theta in Uplus(Theta)",
    }
    laws = source_ok and all(
        getattr(row, key) == value
        for row in (source, signed, sustained, effective)
        for key, value in common_contract.items()
    )
    laws = (
        laws
        and source.same_complete_and_effective_laws
        and source.original_attachment_state_retained
        and source.original_attachment_work_retained_once
        and source.original_clock_start_time == (_ACTIVATION + _SWITCH) * Q(1024, 1023)
        and effective.initial_reduced_form == "same source-correlated Q_s(3375000)"
        and effective.effective_form_equation
        == "Qhat_prime=-2*Qhat-18*gamma*sin(psihat)"
        and effective.effective_phase_equation == "psihat_prime=2*gamma*Qhat/9"
        and effective.actual_form_equation == "Q_prime=-2*Q-18*gamma*sin(psi)-gamma*R"
        and effective.actual_phase_equation == "psi_prime=2*gamma*Q/9"
        and effective.same_law_for_both_preparations
        and effective.original_complete_field_unchanged
        and not effective.pressure_selected_from_preparation_label
        and not effective.independent_marginal_state_reselection
        and not source.reference_is_autonomous_predictor
        and not source.reference_is_complete_source_realization
        and source.actual_phase_used_as_analytical_coefficient
        and not source.phase_readout_input_required
        and not source.reference_velocity_reset
        and not source.prior_finite_endpoint_used_as_initial_state
        and source.clock
        == "tau=(1023/1024)*t; actual rows and same-psi centered reference"
        and signed.clock
        == "tau=(1023/1024)*t; all postattachment actual and effective rows"
        and sustained.clock
        == "tau=(1023/1024)*t; sustained unchanged actual and effective rows"
        and effective.clock
        == "tau=(1023/1024)*t; unchanged actual and effective interaction rows"
        and registration.clock
        == "tau=(1023/1024)*t; translated history and all reciprocal contact rows"
        and phase.clock
        == "tau=(1023/1024)*t; complete internal and phase-sensitive matching rows"
        and renewed.clock
        == "tau=(1023/1024)*t; all internal rows and renewed form port"
        and slow.clock == "tau=(1023/1024)*t; age from original unit A-port activation"
        and collective.clock
        == "tau=(1023/1024)*t; age from original unit A-port activation"
        and persistent.clock
        == "tau=(1023/1024)*t; same complete law and unit A form port"
        and positive.clock
        == "tau=(1023/1024)*t; complete internal rows and reciprocal A form port"
        and handoff.clock == "tau=(1023/1024)*t; all complete internal rows retained"
        and prefix.clock == "tau=(1023/1024)*t; complete full and cut internal laws"
        and reference.clock
        == "tau=(1023/1024)*t; complete internal rows and reciprocal form inputs"
    )

    lower, upper = 1 - 2 * _G, 1 + 2 * _G
    metric_lower, metric_upper, shared_rate = _sine_common_quadratic_coefficients(
        gamma_lower=2 * _GL,
        gamma_upper=2 * _G,
        cosine_lower=Q(1, 500),
        gap_lower=Q(3),
        capacity_lower=Q(4, 9),
        capacity_upper=Q(4, 9),
    )
    rate = _GL**2 / (375 * upper)
    initial_square = (4 * _G * qhi + comparison) ** 2 + 4 * (
        Q(9, 4) * qhi + comparison
    ) ** 2
    square = upper * initial_square / lower
    initial_aggregate = amplitude[1] + inherited_error
    storage_upper = 162 + initial_aggregate**2 + 972 * _ETA + 18 * entry**2
    storage_barrier = 162 + Q(81, 1000) - 18 * Q(1, 500) ** 2
    domain = (
        laws
        and source.prior_domain_certified
        and signed.joint_bootstrap_certified
        and signed.sharp_source_bound_certified
        and signed.weighted_gap_certified
        and signed.phase_stiffness_certified
        and source.common_metric_certified
        and source.normalized_metric_lower_bound == lower > 0
        and source.normalized_metric_upper_bound == upper
        and source.metric_lower_bound == metric_lower == Q(9, 4) * lower
        and source.metric_upper_bound == metric_upper == Q(9, 4) * upper
        and source.shared_metric_decay_rate == source.decay_rate == shared_rate == rate
        and source.shared_phase_scale == 2
        and source.auxiliary_capacity == Q(4, 9)
        and source.metric_cosine_lower_bound == Q(1, 500)
        and source.weighted_gap_lower_bound == 3
        and source.bare_gap_lower_bound == 4
        and source.transport_norm_upper_bound == 36
        and source.initial_scaled_norm_squared_upper_bound == initial_square
        and source.scaled_norm_squared_upper_bound
        == signed.scaled_norm_squared_upper_bound
        == square
        and source.scaled_radius_margin == Q(1, 500) ** 2 - square > 0
        and source.phase_sector_bound == pi / 2 + Q(1, 1000)
        and source.relative_phase_excess_limit == Q(1, 1000)
        and source.initial_full_storage_upper_bound == storage_upper
        and source.phase_boundary_storage_lower_bound == storage_barrier
        and source.phase_barrier_margin == storage_barrier - storage_upper > 0
        and signed.internal_edge_cosine_lower_bound == cos(4 * pi / 9 + Q(3, 2000)).lo
        and signed.matching_cosine_lower_bound == -Q(1, 400)
        and signed.phase_stiffness_lower_bound == Q(71, 200)
        and not source.stiffness_derivative_consumed
        and not source.unequal_internal_blocks_assumed_commuting
    )
    velocity_margins = (
        2 * coupling.lo - sine_coefficient.hi / 9 - 4 * _G**2 * coupling.hi,
        2 * (coupling.lo + 1)
        - sine_coefficient.hi / 9
        - 1
        - 4 * _G**2 * (coupling.hi + 1),
    )
    cone = (
        domain
        and source.source_cone_certified
        and source.reference_cone_certified
        and source.reference_integrals_certified
        and source.reference_restoring_lower_margin == Q(1, 9) - Q(1, 1000) > 0
        and source.reference_velocity_cone_margins == velocity_margins
        and min(velocity_margins) > 0
        and source.reference_position_upper_bound_factor == 1
        and source.reference_velocity_upper_bound_factor == 1
        and source.reference_relative_cosine_lower_bound == -Q(1, 1000)
        and source.reference_initialization
        == "original nominal centered source at A age3375000"
        and source.reference_relative_phase
        == "same actual continuous psi(t), analytically"
        and source.reference_sum_phase_equation == "s_prime=4*gamma*k*V"
        and source.reference_sum_form_equation
        == "V_prime=-2*k*V-(2*gamma*C/9)*sin(s/2)*cos(z/2)"
        and source.reference_difference_phase_equation == "z_prime=4*gamma*(k+1)*r"
        and source.reference_difference_form_equation
        == "r_prime=-2*(k+1)*r-2*gamma*((C/9)*cos(s/2)+cos(psi))*sin(z/2)"
        and source.reference_form_integral_upper_bound == 3 * qhi / _GL
        and source.reference_phase_upper_bound == Q(9, 4) * qhi
        and source.reference_integral_coefficient_margin
        == 1 - (1 / coupling.lo + 1 / (coupling.lo + 1)) / 4
        > 0
        and persistent.nominal_cone_certified
        and persistent.inherited_form_velocity_retained
        and reference.inherited_donor_velocity_retained
        and reference.nominal_wedges_certified
        and handoff.continuation_wedges_certified
    )
    initial_root = 2 * (1 + _G) * comparison
    forcing_integral = 14 * _WEIGHT * qhi / (3 * _GL)
    root_bound = initial_root + forcing_integral
    incremental = (
        cone
        and source.incremental_stiffness_certified
        and source.forced_comparison_certified
        and source.incremental_stiffness_lower_bound == Q(71, 200)
        and source.weight_matrix_deviation_upper_bound
        == persistent.weight_matrix_deviation_upper_bound
        == 14 * _WEIGHT
        and source.initial_scaled_error_upper_bound == 2 * comparison
        and source.initial_metric_root_upper_bound == initial_root
        and source.forcing_integral_upper_bound == forcing_integral
        and source.metric_root_upper_bound == root_bound
        and source.metric_formula == "J=norm(a)**2+norm(b)**2+4*gamma*dot(a,b)"
        and source.forcing_metric_rate == "-2*(1-4*gamma**2)*dot(a,deltaH*xbar)/9"
        and not source.incremental_stiffness_replaced_by_radial_stiffness
        and not source.mean_error_forcing_discarded
        and source.source_error_paid_once
    )

    moment = square / 4
    aggregate = initial_aggregate + 9 * _G
    initial_phase = 6 * _ETA
    delta = initial_phase + 2 * _G * initial_aggregate / 9
    monotonicity = 1 - 4 * _G**2
    first_displacement = initial_phase + 2 * _G * aggregate / 9
    entry_margin = _GL**2 / 4 - delta
    rectangles = (
        domain and 0 <= delta < Q(1, 4) and 0 <= moment <= Q(9, 2) and monotonicity > 0
    )
    central = rectangles and first_displacement < Q(1, 4) and entry_margin > 0
    row_sum = 4 * _G**2 * delta
    gain_denominator = 1 - row_sum
    gain = 1 / gain_denominator
    sensitivity = central and 0 <= row_sum < 1 and gain >= 1

    # Paired offsets cancel their common linear remainder. The comparison
    # reference is used here, never as an input to the independent predictor.
    decomposition = cone and incremental and len(_PAIRS) == 18
    restoring = Q(2, 9) * (1 - qhi**2 / 8) - Q(1, 1000)
    sinc_lower = 1 - qhi**2 / 24
    reference_l2_bound = (
        qhi**2 / (restoring * sinc_lower) * (1 / (4 * _GL**2) + Q(1, 2))
    )
    reference_l2 = cone and restoring > 0 and sinc_lower > 0 and _G**2 < Q(1, 2)
    error_l2_bound = root_bound**2 / (4 * lower * rate)
    error_l2 = incremental and rate > 0 and lower > 0
    moment_integral = (Q(9, 4) + 3 * _EPSILON / 2) * reference_l2_bound + (
        1 + 3 / (2 * _EPSILON)
    ) * error_l2_bound
    residual_integral = decomposition and reference_l2 and error_l2
    initial_lift_error = initial_phase + 2 * _G * inherited_error / 9
    phase_error = gain * (initial_lift_error + 2 * _G**2 * moment_integral / 9)
    form_error = max(inherited_error, 9 * _G * phase_error + _G * moment / 2)
    envelope = sensitivity and residual_integral
    form_accuracy = envelope and form_error < _NOISE
    phase_accuracy = envelope and phase_error < _NOISE

    phase_work = 972 * _ETA + Q(9, 2) * entry**3
    aligned_jump = (Q(0), 9 * (entry + 18 * _ETA) ** 2)
    registered_jump = (162 - phase_work, 162 + phase_work + 9 * entry**2)
    work = (
        laws
        and source.switching_work_certified
        and source.phase_switching_work_allowance == phase_work
        and source.aligned_storage_jump_bounds == aligned_jump
        and source.registered_storage_jump_bounds == registered_jump
        and not source.direct_entry_energy_adds_attachment_twice
    )
    admitted = max(registered_noise, aligned_noise) <= _NOISE
    certified = envelope and work and admitted
    unresolved = tuple(
        label
        for good, label in (
            (source_ok, "original_source_association_unavailable"),
            (laws, "complete_law_history_or_independent_initializer_unavailable"),
            (domain, "actual_joint_domain_unavailable"),
            (cone, "correlated_reference_cone_unavailable"),
            (incremental, "source_inclusive_incremental_comparison_unavailable"),
            (rectangles, "individual_lift_rectangles_unavailable"),
            (central, "analytical_central_entry_unavailable"),
            (sensitivity, "signed_collective_gain_unavailable"),
            (residual_integral, "source_correlated_moment_integral_unavailable"),
            (work, "original_attachment_work_association_unavailable"),
            (admitted, "observation_allowances_outside_original_contract"),
        )
        if not good
    )
    return _MixedCollectiveSensitivity(
        form_error_bounds=forms,
        phase_error_bounds=phases,
        registered_observation_error_bound=registered_noise,
        aligned_observation_error_bound=aligned_noise,
        source=source,
        source_association_certified=source_ok,
        complete_laws_certified=laws,
        prior_domain_certified=domain,
        reference_cone_certified=cone,
        incremental_comparison_certified=incremental,
        individual_rectangles_certified=rectangles,
        central_entry_certified=central,
        signed_sensitivity_certified=sensitivity,
        residual_decomposition_certified=decomposition,
        reference_l2_certified=reference_l2,
        error_l2_certified=error_l2,
        residual_integral_certified=residual_integral,
        uniform_envelope_certified=envelope,
        form_accuracy_certified=form_accuracy,
        phase_accuracy_certified=phase_accuracy,
        switching_work_certified=work,
        observation_allowances_admitted=admitted,
        gamma_bounds=gamma,
        coupling_bounds=coupling,
        sine_coefficient_bounds=sine_coefficient,
        nominal_q0_bounds=(qlo, qhi),
        source_comparison_upper_bound=comparison,
        inherited_output_error_upper_bound=inherited_error,
        slow_amplitude_bounds=amplitude,
        centered_entry_norm_upper_bound=entry,
        initial_scaled_norm_squared_upper_bound=initial_square,
        scaled_norm_squared_upper_bound=square,
        decay_rate=rate,
        normalized_metric_lower_bound=lower,
        normalized_metric_upper_bound=upper,
        metric_root_upper_bound=root_bound,
        initial_aggregate_upper_bound=initial_aggregate,
        aggregate_absolute_upper_bound=aggregate,
        phase_moment_upper_bound=moment,
        old_moment_integral_upper_bound=moment / rate,
        initial_relative_phase_error_upper_bound=initial_phase,
        initial_lift_excess_upper_bound=delta,
        lift_monotonicity_margin=monotonicity,
        rectangle_initial_margin=Q(1, 4) - delta,
        first_unit_phase_displacement_upper_bound=first_displacement,
        central_entry_margin=entry_margin,
        pre_entry_row_sum_upper_bound=row_sum,
        sensitivity_gain_denominator=gain_denominator,
        sensitivity_gain_upper_bound=gain,
        reference_restoring_lower_bound=restoring,
        reference_sinc_lower_bound=sinc_lower,
        reference_phase_squared_integral_upper_bound=reference_l2_bound,
        comparison_phase_squared_integral_upper_bound=error_l2_bound,
        moment_integral_upper_bound=moment_integral,
        moment_integral_improvement_margin=moment / rate - moment_integral,
        initial_lift_error_upper_bound=initial_lift_error,
        uniform_phase_error_upper_bound=phase_error,
        uniform_form_error_upper_bound=form_error,
        form_accuracy_margin=_NOISE - form_error,
        phase_accuracy_margin=_NOISE - phase_error,
        registered_form_record_error_upper_bound=form_error + registered_noise,
        aligned_form_record_error_upper_bound=form_error + aligned_noise,
        phase_switching_work_allowance=phase_work,
        aligned_storage_jump_bounds=aligned_jump,
        registered_storage_jump_bounds=registered_jump,
        certified=certified,
        status=(
            "conditional_uniform_collective_envelope" if certified else "unavailable"
        ),
        unresolved_conditions=unresolved,
        all_postattachment_times_covered=envelope,
        original_uniform_prediction_tolerance_certified=form_accuracy
        and phase_accuracy,
    )
