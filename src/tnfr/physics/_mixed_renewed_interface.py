"""A protocol-aware scalar response at the original pair's renewed port.

Full matching exchange closes the original block total exactly. The unchanged
A branch is a boundary-effect counterfactual, whereas the reduced scalar sees
the same full-port input. Identity and conjugate work have their own budgets.
"""

from dataclasses import dataclass
from fractions import Fraction as Q

from ..mathematics._rational_interval import I, cos, pi_interval
from ._mixed_reciprocal_partner import _nonnegative, _nonnegative_pair
from ._mixed_registered_phase_probe import _derive_registered_phase_probe_geometry
from ._mixed_resource_transfer_obstruction import _original_source_matches
from ._mixed_slow_response import _assess_mixed_slow_response, _MixedSlowResponse

_DELTA = Q(1, 1024)
_WEIGHT = Q(2, 10**60)
_MATURE = Q(1, 2**28)
_ETA = Q(2, 10**69)
_SECTION = 3 * _DELTA / 16
_SECTION_ERROR = Q(3, 2) * (14 * _ETA + 2 * 10**10 * _WEIGHT)
_GL, _G = Q(1, 3300), Q(1, 3000)
_ACTIVATION = Q(2010000)
_LAYER = Q(64)
_SWITCH = Q(3375000)
_DURATION = Q(8)
_NOISE = Q(1, 10**9)
_THRESHOLD = Q(1, 10**8)


@dataclass(frozen=True)
class _MixedRenewedInterface:
    """Exact aggregate closure with a carried initializer and finite identity."""

    form_error_bounds: tuple[Q, Q]
    phase_error_bounds: tuple[Q, Q]
    full_observation_error_bound: Q
    continued_observation_error_bound: Q
    source: _MixedSlowResponse
    source_association_certified: bool
    sum_closure_certified: bool
    scalar_response_certified: bool
    changed_port_storage_certified: bool
    finite_identity_certified: bool
    work_allowance_certified: bool
    observation_allowances_admitted: bool
    full_return_certified: bool
    continued_response_certified: bool
    full_record_certified: bool
    continued_record_certified: bool
    independent_record_sets_separated: bool
    fixed_threshold_certified: bool
    nominal_q0_bounds: tuple[Q, Q]
    coupling_bounds: I
    sine_coefficient_bounds: I
    gamma_bounds: I
    slow_amplitude_bounds: tuple[Q, Q]
    slow_scale_bounds: tuple[Q, Q]
    inherited_output_error_upper_bound: Q
    scalar_initialization_error_upper_bound: Q
    scalar_output_error_upper_bound: Q
    entry_aggregate_bounds: tuple[Q, Q]
    entry_norm_upper_bound: Q
    norm_growth_denominator: Q
    post_norm_upper_bound: Q
    phase_drift_upper_bound: Q
    marked_gap_upper_bound: Q
    registration_margin: Q
    actual_dissipation_lower_bound: Q
    port_power_upper_bound: Q
    positive_work_upper_bound: Q
    full_true_bounds: tuple[Q, Q]
    continued_true_bounds: tuple[Q, Q]
    full_record_bounds: tuple[Q, Q]
    continued_record_bounds: tuple[Q, Q]
    full_return_margin: Q
    full_threshold_margin: Q
    continued_threshold_margin: Q
    record_separation_margin: Q
    certified: bool
    status: str
    unresolved_conditions: tuple[str, ...]
    switch_age: Q = _SWITCH
    post_duration: Q = _DURATION
    activation_time: Q = _ACTIVATION
    switch_time: Q = _ACTIVATION + _SWITCH
    total_duration: Q = _ACTIVATION + _SWITCH + _DURATION
    original_clock_horizon: Q = (_ACTIVATION + _SWITCH + _DURATION) * Q(1024, 1023)
    scalar_decay_rate: Q = Q(2)
    scalar_decay_upper_bound: Q = Q(1, 2**16)
    storage_jump: Q = Q(0)
    nominal_dissipation_lower_bound: Q = Q(1)
    actual_dissipation_perturbation_upper_bound: Q = 84 * _WEIGHT
    port_power_weight_coefficient: Q = 28 * _WEIGHT
    block_total_generator: tuple[tuple[Q, Q], tuple[Q, Q]] = (
        (Q(-1), Q(1)),
        (Q(1), Q(-1)),
    )
    full_contact_pairs: tuple[tuple[int, int], ...] = tuple(
        (j, 18 + j) for j in range(18)
    )
    added_contact_pairs: tuple[tuple[int, int], ...] = tuple(
        (j, 18 + j) for j in range(9, 18)
    )
    full_port_row_sums: tuple[Q, ...] = (Q(1),) * 18
    full_port_column_sums: tuple[Q, ...] = (Q(1),) * 18
    observation_covector: tuple[Q, ...] = (-Q(1),) * 18 + (Q(0),) * 54
    original_donor_form_total: Q = Q(0)
    whole_form_mean: Q = Q(0)
    full_observation_error_cap: Q = _NOISE
    continued_observation_error_cap: Q = _NOISE
    true_return_target: Q = _NOISE
    threshold: Q = _THRESHOLD
    scalar_equation: str = "Q_prime=-2*Q"
    reduced_response: str = "Q_hat(s)=Q_s(3375000)*exp(-2*s)"
    rebased_equation: str = "Z_prime=-2*(Z+Q_entry)"
    same_input_scalar_model: bool = True
    continued_A_is_boundary_effect_control: bool = True
    counterfactuals_share_reached_actual_member: bool = True
    inherited_amplitude_rate_correlation_retained: bool = True
    original_baseline_retained: bool = True
    internal_rows_and_clock_unchanged: bool = True
    independent_record_errors: bool = True
    observation_error_cancellation_assumed: bool = False
    full_nonlinear_state_closure: bool = False
    inherited_slow_rate_used_after_switch: bool = False
    all_future_changed_port_identity_claimed: bool = False
    port_only_passivity_claimed: bool = False
    zero_physical_switching_cost_claimed: bool = False
    original_source_reacquired: bool = False
    state_reset_performed: bool = False
    observation_rebased_at_switch: bool = False
    switch_observations: int = 0
    supplied_external_protocol: bool = True
    runtime_event_executed: bool = False
    operator_occurrence_derived: bool = False
    source_trajectory_evaluated: bool = False
    exponentials_evaluated: bool = False
    roots_numerically_evaluated: bool = False
    physical_timescale_identified: bool = False
    empirical_validation: bool = False
    newly_observed_aggregates: int = 0
    required_endpoint_records: int = 2
    numerical_response_allowance: Q = Q(0)
    clock: str = "tau=(1023/1024)*t; all internal rows and renewed form port"


def _assess_mixed_renewed_interface(
    *,
    form_error_bounds,
    phase_error_bounds,
    full_observation_error_bound,
    continued_observation_error_bound,
):
    """Admit one fixed A-to-full protocol without changing its reached state."""
    forms = _nonnegative_pair(form_error_bounds, "form_error_bounds")
    phases = _nonnegative_pair(phase_error_bounds, "phase_error_bounds")
    full_noise = _nonnegative(
        full_observation_error_bound, "full_observation_error_bound"
    )
    continued_noise = _nonnegative(
        continued_observation_error_bound, "continued_observation_error_bound"
    )
    source = _assess_mixed_slow_response(
        form_error_bounds=forms,
        phase_error_bounds=phases,
        early_observation_error_bound=Q(0),
        late_observation_error_bound=Q(0),
    )
    collective = source.source
    persistent = collective.source
    storage = persistent.source
    positive = storage.source
    handoff = positive.handoff
    prefix = handoff.prefix
    reference = prefix.reference
    registered = _derive_registered_phase_probe_geometry()
    geometry = registered.successor
    pi = pi_interval()
    gamma = 1 / (1023 * pi)
    k = Q(13, 27) - _DELTA / (18 * pi)
    coefficient = 1 + 2 * cos(pi / 9)
    qlo, qhi = _SECTION - _SECTION_ERROR / 2, _SECTION + _SECTION_ERROR / 2
    q0 = (qlo, qhi)
    source_ok = (
        collective.form_error_bounds
        == persistent.form_error_bounds
        == storage.form_error_bounds
        == positive.form_error_bounds
        == forms
        and collective.phase_error_bounds
        == persistent.phase_error_bounds
        == storage.phase_error_bounds
        == positive.phase_error_bounds
        == phases
        and collective.source_association_certified
        and persistent.source_association_certified
        and storage.source_association_certified
        and positive.source_association_certified
        and handoff.source_association_certified
        and reference.source_association_certified
        and reference.donor_source_certified
        and reference.recipient_source_certified
        and collective.projected_rows_certified
        and positive.projected_rows_certified
        and registered.cross_symmetry_certified
        and geometry == prefix.source_geometry == reference.source_geometry
        and geometry.nodes == tuple(range(18))
        and geometry.degrees == (9,) * 18
        and geometry.held_capacities == (Q(1),) * 18
        and geometry.support_certified
        and geometry.reference_stationary_certified
        and geometry.spectral_gap_certified
        and geometry.bare_gap_lower_bound == 4
        and geometry.weighted_gap_lower_bound >= 3
        and reference.donor.cross_incidence == registered.cross_incidence
        and reference.sources
        == (reference.donor.source, reference.recipient_readiness.sources[0])
        and all(
            _original_source_matches(row, geometry, rx, rt)
            for row, rx, rt in zip(reference.sources, forms, phases, strict=True)
        )
        and reference.original_relative_phase_gap == _DELTA
        and reference.original_supplied_phase_origins == (-_DELTA / 2, _DELTA / 2)
        and reference.actual_weight_deviation_upper_bound == _WEIGHT
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
        and positive.exact_zero_entry_total_from_prior_laws
        and collective.initial_donor_form_total
        == positive.initial_donor_form_total
        == 0
        and collective.initial_common_form_difference
        == persistent.activation_relative_common_form
        == 0
        and collective.whole_form_mean
        == persistent.whole_form_mean
        == storage.whole_form_mean
        == positive.whole_form_mean
        == 0
        and positive.physical_capacities
        == reference.physical_capacities
        == persistent.physical_capacities
        == (Q(1),) * 36
        and positive.full_state_dimension
        == reference.full_state_dimension
        == persistent.full_state_dimension
        == 72
        and positive.contact_pairs
        == persistent.contact_pairs
        == tuple((j, 18 + j) for j in range(9))
        and collective.fixed_port_strength
        == persistent.fixed_port_strength
        == storage.fixed_port_strength
        == 1
        and positive.prefix_full_duration == handoff.prefix_duration == Q(2000000)
        and positive.prefix_cut_duration == handoff.continuation_duration == Q(10000)
        and positive.handoff_time
        == handoff.total_duration
        == persistent.activation_time
        == collective.activation_time
        == _ACTIVATION
        and positive.positive_duration
        == persistent.previously_reached_duration
        == storage.horizon
        == Q(1, 2)
        and positive.total_duration
        == persistent.previously_reached_time
        == _ACTIVATION + Q(1, 2)
        and positive.original_clock_horizon == (_ACTIVATION + Q(1, 2)) * Q(1024, 1023)
        and handoff.original_clock_horizon == _ACTIVATION * Q(1024, 1023)
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
        and handoff.switch_word == (1, 0)
        and collective.full_observation_error_bound
        == collective.form_only_observation_error_bound
        == 0
        and positive.observation_error_bound == handoff.observation_error_bound == 0
        and positive.handoff_observations == handoff.handoff_observations == 0
        and positive.inherited_form_velocity_retained
        and positive.relative_common_form_mode_retained
        and positive.original_phase_gauges_and_branches_retained
        and persistent.original_phase_gauges_and_branches_retained
        and handoff.original_phase_gauges_and_branches_retained
        and handoff.reference_family_carried
        and handoff.reached_velocity_retained
        and handoff.source_weight_association_retained
        and reference.inherited_donor_velocity_retained
        and not any(
            (
                collective.state_reset_performed,
                persistent.state_reset_performed,
                storage.state_reset_performed,
                positive.state_reset_performed,
                handoff.state_reset_performed,
                persistent.port_switch_performed,
            )
        )
    )
    coefficients_ok = (
        source_ok
        and collective.coupling_bounds
        == persistent.coupling_bounds
        == storage.coupling_bounds
        == reference.donor.nominal_coupling_bounds
        == k
        and storage.cross_strength_bounds == 9 * k
        and Q(4, 9) < k.lo <= k.hi < Q(1, 2)
        and collective.sine_coefficient_bounds
        == persistent.sine_coefficient_bounds
        == reference.donor.regional_sine_coefficient_bounds
        == coefficient
        and 2 < coefficient.lo <= coefficient.hi < 3
        and collective.gamma_bounds == reference.donor.gamma_bounds == gamma
        and persistent.gamma_bounds == (_GL, _G)
        and _GL <= gamma.lo <= gamma.hi <= _G
        and collective.nominal_q0_bounds
        == persistent.nominal_q0_bounds
        == reference.nominal_q0_bounds
        == positive.nominal_q0_bounds
        == q0
        and 0 < qlo <= qhi < _DELTA / 4
    )
    error, budget, forcing, matrix_error = (
        8 * _MATURE,
        200 * _MATURE,
        2 * _WEIGHT,
        14 * _WEIGHT,
    )
    cut_error = error / (1 + Q(10000, 3)) + 6 * _G * error + 3 * forcing
    rate = reference.conservative_metric_decay_rate
    carried_error = (
        4 * (_MATURE + _SECTION_ERROR + 2 * forcing / rate) if rate > 0 else None
    )
    prefix_ok = (
        coefficients_ok
        and handoff.handoff_certified
        and handoff.switched_metric_certified
        and reference.metric_certified
        and reference.comparison_certified
        and rate
        == handoff.metric_decay_rate
        == reference.base_metric_decay_rate / 8
        > 0
        and handoff.carried_error_upper_bound
        == reference.uniform_comparison_bound
        == carried_error
        and carried_error < error
        and reference.comparison_forcing_upper_bound
        == handoff.metric_forcing_upper_bound
        == forcing
        and reference.comparison_matrix_deviation_upper_bound == matrix_error
        and handoff.actual_observation_allowance
        == persistent.original_comparison_upper_bound
        == error
        and positive.cut_form_error_certified
        and positive.cut_decay_upper_bound == 1 / (1 + Q(10000, 3))
        and positive.cut_form_error_upper_bound
        == collective.cut_form_error_upper_bound
        == cut_error
        and positive.handoff_phase_error_upper_bound == error
    )
    cone_margins = (
        2 * k.lo / 3 - _G**2,
        2 * k.lo - Q(1, 5) - coefficient.hi / 9 - 4 * _G**2 * k.hi,
        Q(1, 5) - Q(24, 5) * _G**2 * k.hi,
    )
    cone = (
        prefix_ok
        and collective.nominal_cone_certified
        and persistent.nominal_cone_certified
        and handoff.continuation_wedges_certified
        and reference.nominal_wedges_certified
        and reference.donor.nominal_reference_certified
        and reference.nominal_velocity_factor_upper_bound == _G**2
        and reference.nominal_wedge_margin == cone_margins[0] > 0
        and persistent.nominal_cone_margins == cone_margins
        and all(value > 0 for value in cone_margins)
        and persistent.nominal_common_form_cone_factor == Q(6, 5)
        and persistent.nominal_contrast_form_cone_factor == 1
        and persistent.nominal_norm_upper_bound == 4 * qhi < Q(1, 100)
        and reference.inherited_nominal_velocity_relation
        == "-gamma**2*q0 <= w0 <= 0 at the same isolated age"
        and positive.phase_difference_entry_lower_bound == Q(233, 300) * qlo
        and positive.inherited_form_contrast_lower_bound == qlo / 20000
        and positive.inherited_orientation_certified
        and collective.tangent_initialization
        == "u0=0; b0 and z0 carried from the same original cut"
        and collective.nonlinear_sum_phase_retained
        and collective.tangent_coefficients_correlated
    )
    energy = Q(27, 2) * _DELTA**2 + 486 * _ETA + 18 * _ETA**2 + 9 * _MATURE**2
    ly, lx = 3 * qhi, 10 * qhi / _GL
    bregman = 3 * (3 * error + 6 * matrix_error * lx) / (1 - 45 * ly)
    tube = (
        cone
        and persistent.broad_trap_certified
        and persistent.phase_convexity_certified
        and storage.storage_domain_certified
        and storage.actual_dissipation_certified
        and persistent.selected_mu == storage.selected_mu == 1
        and storage.mean_projector_rank == 1
        and storage.port_norm_upper_bound == 2
        and storage.loss_divisor == 9
        and storage.origin_reference
        == "conserved_whole_form_mean; P_rel=P_copymean-P_wholemean"
        and storage.actual_weight_perturbation_upper_bound == 84 * _WEIGHT
        and persistent.bregman_comparison_certified
        and persistent.uniform_tube_certified
        and persistent.finite_variation_certified
        and persistent.initial_excess_storage_upper_bound
        == reference.initial_excess_storage_upper_bound
        == energy
        and persistent.full_norm_squared_upper_bound == 5 * energy < Q(1, 10000)
        and persistent.actual_dissipation_lower_bound
        == storage.actual_dissipation_lower_bound
        == Q(1, 4) - 84 * _WEIGHT
        > 0
        and persistent.storage_coercivity_lower_bound == Q(1, 5)
        and persistent.storage_norm_upper_bound == 9
        and persistent.storage_barrier == Q(1, 5) * Q(1, 32) ** 2
        and persistent.storage_margin == persistent.storage_barrier - energy > 0
        and persistent.phase_radius == reference.phase_radius == Q(1, 32)
        and persistent.nominal_phase_variation_upper_bound == ly
        and persistent.nominal_form_integral_upper_bound == lx
        and persistent.bregman_growth_integral_upper_bound == 45 * ly < 1
        and persistent.bregman_denominator == 1 - 45 * ly
        and persistent.initial_bregman_root_upper_bound == 3 * error
        and persistent.weight_matrix_deviation_upper_bound == matrix_error
        and persistent.bregman_coercivity_lower_bound == Q(1, 5)
        and persistent.bregman_norm_upper_bound == 9
        and persistent.bregman_phase_growth_coefficient == 45
        and persistent.bregman_weight_forcing_coefficient == 6
        and persistent.uniform_comparison_upper_bound == bregman < budget
        and persistent.uniform_comparison_budget
        == collective.full_state_comparison_upper_bound
        == budget
        and persistent.uniform_comparison_margin == budget - bregman > 0
    )
    fq = 2 * _G * budget / 9 + 5 * _G * budget**2 / 81 + matrix_error / 2700
    eq = 3 * cut_error + Q(81, 8) * fq
    original = (
        tube
        and collective.kernel_certified
        and collective.uniform_output_certified
        and collective.kernel_mass_upper_bound == Q(81, 8)
        and collective.kernel_impulse_upper_bound == 1
        and 9 / (2 * k.lo) < Q(81, 8)
        and collective.forcing_error_upper_bound == fq
        and collective.uniform_output_error_upper_bound == eq
        and collective.output_error_margin == _NOISE - eq > 0
        and collective.observation_covector
        == positive.observation_covector
        == (-Q(1),) * 18 + (Q(0),) * 54
        and collective.centered_observation_covector_norm_squared == 9
    )
    beta, chi = gamma * coefficient / 9, 4 * gamma * k
    rho = 2 * gamma**2 * coefficient / 9
    stiffness = 2 * k * rho
    polynomial = (I(1), 2 * k + 2, 2 * k + stiffness, stiffness)
    signs = (
        rho**2 * (rho - 2),
        rho * (2 * k - (4 * k + 8) * rho + 8 * rho**2),
        3 * k / 8 - Q(7, 64) - 3 * stiffness / 4,
        k / 2 - Q(3, 8) - stiffness / 2,
        stiffness - 4 * k,
        9 - 12 * k + 2 * stiffness,
    )
    brackets = ((rho, 2 * rho), (I(Q(1, 4)), I(Q(1, 2))), (I(2), I(3)))
    roots = (
        cone
        and collective.beta_bounds == beta
        and collective.chi_bounds == chi
        and collective.stiffness_bounds == stiffness
        and collective.slow_rate_scale_bounds == rho
        and collective.characteristic_coefficients == polynomial
        and collective.decay_rate_brackets == brackets
        and collective.tangent_root_sign_bounds == signs
        and collective.tangent_laplace_numerator == "p*b0+beta*z0"
        and collective.tangent_laplace_denominator == "p^3+(2*k+2)*p^2+(2*k+d)*p+d"
        and rho.lo > 0
        and 2 * rho.hi < Q(1, 36)
        and signs[0].hi < 0 < signs[1].lo
        and signs[2].lo > 0 > signs[3].hi
        and signs[4].hi < 0 < signs[5].lo
    )
    # The 3D forcing kernel has transfer p/P(p), not the old positive
    # two-stage kernel. Its signed integral is zero; its ABSOLUTE gain is paid.
    signed_kernel = roots and Q(1, 4) - Q(1, 36) == Q(2, 9) and 2 - Q(1, 2) == Q(3, 2)
    nonlinear_force = _G * qhi**3 / 18
    nonlinear_error = 3 * _G * qhi**3
    nonlinear = signed_kernel and cone and 9 * 6 * nonlinear_force == nonlinear_error
    fast_residues = 3 * _G * qhi
    fast_error = 9 * fast_residues / 2**16
    fast = roots and cone and Q(137, 48) < 3 and _LAYER / 4 == 16
    amplitude = (
        Q(4, 3) * _GL * Q(233, 300) * qlo * (1 - _G**2 / Q(4, 9)),
        16 * _G * qhi,
    )
    amplitude_ok = (
        roots and cone and _G**2 < Q(4, 9) and 0 < amplitude[0] < amplitude[1]
    )
    total = eq + nonlinear_error + fast_error
    approximation_margin = _NOISE - total
    approximation = (
        original and nonlinear and fast and amplitude_ok and approximation_margin > 0
    )
    slow_scale = (4 * _GL**2 / 9, 2 * _G**2 / 3)
    # Re-admit the slow family's consumed premises, not its previous endpoint
    # bands or overall verdict. The switch adds only the B matching form input.
    slow_source = (
        source_ok
        and source.source_association_certified
        and source.form_error_bounds == forms
        and source.phase_error_bounds == phases
        and source.early_observation_error_bound
        == source.late_observation_error_bound
        == 0
        and source.clock
        == "tau=(1023/1024)*t; age from original unit A-port activation"
        and source.activation_time == _ACTIVATION
        and source.early_age == _SWITCH == 1 / (4 * slow_scale[1])
        and source.layer_age == _LAYER < _SWITCH
        and source.same_actual_history_required
        and source.correlated_initialization_retained
        and source.initialization
        == "u0=0; b0 and z0 carried from the same original cut"
        and source.nonlinear_phase_sum_retained
        and source.all_three_residues_retained_before_layer
        and not any(
            (
                source.state_reset_performed,
                source.fitted_amplitude_or_rate,
                source.amplitude_rate_marginals_independently_attainable,
            )
        )
    )
    inherited = (
        slow_source
        and approximation
        and source.uniform_approximation_certified
        and source.tangent_roots_certified
        and source.signed_kernel_certified
        and source.nonlinear_error_certified
        and source.fast_tail_certified
        and source.amplitude_certified
        and source.nominal_q0_bounds == q0
        and source.coupling_bounds == k
        and source.sine_coefficient_bounds == coefficient
        and source.gamma_bounds == gamma
        and source.beta_bounds == beta
        and source.chi_bounds == chi
        and source.stiffness_bounds == stiffness
        and source.slow_rate_scale_bounds == rho
        and source.characteristic_coefficients == polynomial
        and source.decay_rate_brackets == brackets
        and source.tangent_root_sign_bounds == signs
        and source.kernel_abs_mass_upper_bound == 6
        and source.kernel_signed_integral == 0
        and not source.kernel_is_nonnegative
        and source.kernel_transfer_numerator == "p"
        and source.root_gap_lower_bounds == (Q(2, 9), Q(3, 2))
        and source.original_output_error_upper_bound == eq
        and source.nonlinear_forcing_upper_bound == nonlinear_force
        and source.nonlinear_output_error_upper_bound == nonlinear_error
        and source.fast_residue_factor == Q(137, 48)
        and source.fast_residue_upper_bound == fast_residues
        and source.fast_output_tail_upper_bound == fast_error
        and source.total_output_error_upper_bound == total
        and source.approximation_margin == approximation_margin > 0
        and source.slow_amplitude_bounds == amplitude
        and source.slow_scale_bounds == slow_scale
        and source.slow_amplitude_expression == "9*(beta*z0-r_s*b0)/P_prime(-r_s)"
        and source.slow_family_expression == "A_Q*exp(-r_s*age)"
    )
    # Internal symmetric-transport and antisymmetric-sine edge sums vanish in
    # each copy. The identity matching has constant row and column sums one.
    # Thus (S_D,S_R)'=(-S_D+S_R,S_D-S_R), including actual unequal weights.
    closure = slow_source and coefficients_ok and collective.whole_form_mean == 0
    scalar = closure and inherited
    entry = (amplitude[0] / 2 - total, amplitude[1] + total)
    scalar_error = total / 2**16
    entry_norm = 4 * qhi + budget
    denominator = 1 - 2 * _G * _DURATION
    post_norm = entry_norm / denominator
    phase_drift = 3 * _G * _DURATION * post_norm
    marked = qhi + 300 * _MATURE + phase_drift
    margin = _DELTA / 4 - marked
    identity = (
        inherited
        and tube
        and persistent.nominal_norm_upper_bound == 4 * qhi
        and persistent.marked_gap_upper_bound == qhi + 300 * _MATURE
        and persistent.registration_gap_limit
        == reference.registration_gap_limit
        == _DELTA / 4
        and persistent.registration_margin == _DELTA / 4 - (qhi + 300 * _MATURE) > 0
        and denominator > 0
        and post_norm < Q(1, 100) < Q(1, 32)
        and margin > 0
    )
    # Full matching commutes with nominal H0. On relative common forms its
    # Eplus loss is stronger; on the centered leaf H0^2/9 >= I. The actual
    # noncommuting perturbation is retained, not assumed to vanish.
    loss = 1 - 84 * _WEIGHT
    storage_ok = (
        closure
        and identity
        and tube
        and geometry.weighted_gap_lower_bound >= 3
        and storage.cross_strength_bounds == 9 * k
        and storage.mean_projector_rank == 1
        and storage.selected_mu == 1
        and loss > 0
    )
    power = 28 * _WEIGHT * post_norm**2
    work = _DURATION * power
    work_ok = storage_ok and power > 0 and work > 0
    full_upper = (amplitude[1] + total) / 2**16
    continuation_factor = Q(1, 2) - 2 * slow_scale[1] * _DURATION
    continued_lower = amplitude[0] * continuation_factor - total
    full_true = (Q(0), full_upper)
    continued_true = (continued_lower, amplitude[1] + total)
    full_record = (-full_noise, full_upper + full_noise)
    continued_record = (
        continued_lower - continued_noise,
        amplitude[1] + total + continued_noise,
    )
    admitted = full_noise <= _NOISE and continued_noise <= _NOISE
    full_return_margin = _NOISE - full_upper
    full_margin = _THRESHOLD - max(abs(value) for value in full_record)
    continued_margin = continued_record[0] - _THRESHOLD
    separation = continued_record[0] - full_record[1]
    full_return = scalar and identity and entry[0] > 0 and full_return_margin > 0
    continued = inherited and continuation_factor > 0 and continued_lower > _THRESHOLD
    full_record_ok = full_return and admitted and full_margin > 0
    continued_record_ok = continued and admitted and continued_margin > 0
    separated = full_record_ok and continued_record_ok and separation > 0
    threshold = separated and full_margin > 0 and continued_margin > 0
    unresolved = tuple(
        label
        for good, label in (
            (slow_source, "original_reached_source_association_unavailable"),
            (inherited, "carried_slow_initializer_and_error_unavailable"),
            (closure, "full_matching_actual_sum_closure_unavailable"),
            (scalar, "same_input_scalar_prediction_unavailable"),
            (identity, "renewed_interval_identity_unavailable"),
            (storage_ok, "changed_port_total_storage_loss_unavailable"),
            (work_ok, "actual_positive_port_work_allowance_unavailable"),
            (admitted, "independent_record_error_exceeds_protocol"),
            (full_record_ok, "full_matching_record_return_unavailable"),
            (continued_record_ok, "unchanged_A_record_persistence_unavailable"),
            (separated, "same_source_boundary_effect_not_resolved"),
            (threshold, "original_threshold_not_certified"),
        )
        if not good
    )
    return _MixedRenewedInterface(
        form_error_bounds=forms,
        phase_error_bounds=phases,
        full_observation_error_bound=full_noise,
        continued_observation_error_bound=continued_noise,
        source=source,
        source_association_certified=slow_source,
        sum_closure_certified=closure,
        scalar_response_certified=scalar,
        changed_port_storage_certified=storage_ok,
        finite_identity_certified=identity,
        work_allowance_certified=work_ok,
        observation_allowances_admitted=admitted,
        full_return_certified=full_return,
        continued_response_certified=continued,
        full_record_certified=full_record_ok,
        continued_record_certified=continued_record_ok,
        independent_record_sets_separated=separated,
        fixed_threshold_certified=threshold,
        nominal_q0_bounds=q0,
        coupling_bounds=k,
        sine_coefficient_bounds=coefficient,
        gamma_bounds=gamma,
        slow_amplitude_bounds=amplitude,
        slow_scale_bounds=slow_scale,
        inherited_output_error_upper_bound=total,
        scalar_initialization_error_upper_bound=total,
        scalar_output_error_upper_bound=scalar_error,
        entry_aggregate_bounds=entry,
        entry_norm_upper_bound=entry_norm,
        norm_growth_denominator=denominator,
        post_norm_upper_bound=post_norm,
        phase_drift_upper_bound=phase_drift,
        marked_gap_upper_bound=marked,
        registration_margin=margin,
        actual_dissipation_lower_bound=loss,
        port_power_upper_bound=power,
        positive_work_upper_bound=work,
        full_true_bounds=full_true,
        continued_true_bounds=continued_true,
        full_record_bounds=full_record,
        continued_record_bounds=continued_record,
        full_return_margin=full_return_margin,
        full_threshold_margin=full_margin,
        continued_threshold_margin=continued_margin,
        record_separation_margin=separation,
        certified=not unresolved,
        status="conditional_renewed_interface" if not unresolved else "unavailable",
        unresolved_conditions=unresolved,
    )
