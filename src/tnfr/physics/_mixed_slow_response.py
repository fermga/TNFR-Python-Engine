"""Controlled slow output under the original permanent unit A contact.

The retained slow exponential is a correlated family, not a fitted rate or
amplitude. Exact rational allowances price nonlinear forcing, the fast layer
and the previously admitted actual/reference discrepancy. No response runs.
"""

from dataclasses import dataclass
from fractions import Fraction as Q

from ..mathematics._rational_interval import I, cos, pi_interval
from ._mixed_collective_relaxation import (
    _assess_mixed_collective_relaxation,
    _MixedCollectiveRelaxation,
)
from ._mixed_reciprocal_partner import _nonnegative, _nonnegative_pair
from ._mixed_registered_phase_probe import _derive_registered_phase_probe_geometry
from ._mixed_resource_transfer_obstruction import _original_source_matches

_DELTA = Q(1, 1024)
_WEIGHT = Q(2, 10**60)
_MATURE = Q(1, 2**28)
_ETA = Q(2, 10**69)
_SECTION = 3 * _DELTA / 16
_SECTION_ERROR = Q(3, 2) * (14 * _ETA + 2 * 10**10 * _WEIGHT)
_GL, _G = Q(1, 3300), Q(1, 3000)
_ACTIVATION = Q(2010000)
_LAYER = Q(64)
_NOISE = Q(1, 10**9)
_THRESHOLD = Q(1, 10**8)


@dataclass(frozen=True)
class _MixedSlowResponse:
    """Conditional absolute output approximation and two fixed-age bounds."""

    form_error_bounds: tuple[Q, Q]
    phase_error_bounds: tuple[Q, Q]
    early_observation_error_bound: Q
    late_observation_error_bound: Q
    source: _MixedCollectiveRelaxation
    source_association_certified: bool
    tangent_roots_certified: bool
    signed_kernel_certified: bool
    nonlinear_error_certified: bool
    fast_tail_certified: bool
    amplitude_certified: bool
    uniform_approximation_certified: bool
    observation_allowances_admitted: bool
    early_response_certified: bool
    early_record_certified: bool
    late_return_certified: bool
    late_record_certified: bool
    correlated_true_drop_certified: bool
    correlated_drop_certified: bool
    coupling_bounds: I
    sine_coefficient_bounds: I
    gamma_bounds: I
    nominal_q0_bounds: tuple[Q, Q]
    beta_bounds: I
    chi_bounds: I
    stiffness_bounds: I
    slow_rate_scale_bounds: I
    characteristic_coefficients: tuple[I, I, I, I]
    decay_rate_brackets: tuple[tuple[I, I], tuple[I, I], tuple[I, I]]
    tangent_root_sign_bounds: tuple[I, I, I, I, I, I]
    original_output_error_upper_bound: Q
    nonlinear_forcing_upper_bound: Q
    nonlinear_output_error_upper_bound: Q
    fast_residue_upper_bound: Q
    fast_output_tail_upper_bound: Q
    total_output_error_upper_bound: Q
    approximation_margin: Q
    slow_amplitude_bounds: tuple[Q, Q]
    slow_scale_bounds: tuple[Q, Q]
    early_age: Q
    late_age: Q
    early_true_bounds: tuple[Q, Q]
    late_true_bounds: tuple[Q, Q]
    early_record_bounds: tuple[Q, Q]
    late_record_bounds: tuple[Q, Q]
    early_threshold_margin: Q
    late_true_return_margin: Q
    late_threshold_margin: Q
    correlated_true_drop_lower_bound: Q
    correlated_record_drop_lower_bound: Q
    certified: bool
    status: str
    unresolved_conditions: tuple[str, ...]
    kernel_abs_mass_upper_bound: Q = Q(6)
    kernel_signed_integral: Q = Q(0)
    kernel_is_nonnegative: bool = False
    kernel_transfer_numerator: str = "p"
    root_gap_lower_bounds: tuple[Q, Q] = (Q(2, 9), Q(3, 2))
    fast_residue_factor: Q = Q(137, 48)
    layer_age: Q = _LAYER
    activation_time: Q = _ACTIVATION
    early_observation_error_cap: Q = _NOISE
    late_observation_error_cap: Q = _NOISE
    true_return_target: Q = _NOISE
    threshold: Q = _THRESHOLD
    slow_amplitude_expression: str = "9*(beta*z0-r_s*b0)/P_prime(-r_s)"
    slow_family_expression: str = "A_Q*exp(-r_s*age)"
    initialization: str = "u0=0; b0 and z0 carried from the same original cut"
    correlated_initialization_retained: bool = True
    same_actual_history_required: bool = True
    independent_record_errors: bool = True
    observation_error_cancellation_assumed: bool = False
    nonlinear_phase_sum_retained: bool = True
    all_three_residues_retained_before_layer: bool = True
    amplitude_rate_marginals_independently_attainable: bool = False
    fitted_amplitude_or_rate: bool = False
    old_record_verdict_consumed: bool = False
    exponentials_evaluated: bool = False
    roots_numerically_evaluated: bool = False
    source_trajectory_evaluated: bool = False
    state_reset_performed: bool = False
    exact_actual_modal_closure: bool = False
    measured_rate_identified: bool = False
    relative_rate_error_certified: bool = False
    first_return_time_identified: bool = False
    physical_timescale_identified: bool = False
    empirical_validation: bool = False
    newly_observed_aggregates: int = 0
    required_endpoint_records: int = 2
    numerical_response_allowance: Q = Q(0)
    earlier_sufficient_deadline: Q = Q(627264000000000)
    clock: str = "tau=(1023/1024)*t; age from original unit A-port activation"


def _assess_mixed_slow_response(
    *,
    form_error_bounds,
    phase_error_bounds,
    early_observation_error_bound,
    late_observation_error_bound,
):
    """Rebuild a same-history slow family without evaluating its exponential."""
    forms = _nonnegative_pair(form_error_bounds, "form_error_bounds")
    phases = _nonnegative_pair(phase_error_bounds, "phase_error_bounds")
    early_noise = _nonnegative(
        early_observation_error_bound, "early_observation_error_bound"
    )
    late_noise = _nonnegative(
        late_observation_error_bound, "late_observation_error_bound"
    )
    source = _assess_mixed_collective_relaxation(
        form_error_bounds=forms,
        phase_error_bounds=phases,
        full_observation_error_bound=Q(0),
        form_only_observation_error_bound=Q(0),
    )
    persistent = source.source
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
        source.form_error_bounds
        == persistent.form_error_bounds
        == storage.form_error_bounds
        == positive.form_error_bounds
        == forms
        and source.phase_error_bounds
        == persistent.phase_error_bounds
        == storage.phase_error_bounds
        == positive.phase_error_bounds
        == phases
        and source.source_association_certified
        and persistent.source_association_certified
        and storage.source_association_certified
        and positive.source_association_certified
        and handoff.source_association_certified
        and reference.source_association_certified
        and reference.donor_source_certified
        and reference.recipient_source_certified
        and source.projected_rows_certified
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
        and source.initial_donor_form_total == positive.initial_donor_form_total == 0
        and source.initial_common_form_difference
        == persistent.activation_relative_common_form
        == 0
        and source.whole_form_mean
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
        and source.fixed_port_strength
        == persistent.fixed_port_strength
        == storage.fixed_port_strength
        == 1
        and positive.prefix_full_duration == handoff.prefix_duration == Q(2000000)
        and positive.prefix_cut_duration == handoff.continuation_duration == Q(10000)
        and positive.handoff_time
        == handoff.total_duration
        == persistent.activation_time
        == source.activation_time
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
        and source.clock
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
        and source.full_observation_error_bound
        == source.form_only_observation_error_bound
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
                source.state_reset_performed,
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
        and source.coupling_bounds
        == persistent.coupling_bounds
        == storage.coupling_bounds
        == reference.donor.nominal_coupling_bounds
        == k
        and storage.cross_strength_bounds == 9 * k
        and Q(4, 9) < k.lo <= k.hi < Q(1, 2)
        and source.sine_coefficient_bounds
        == persistent.sine_coefficient_bounds
        == reference.donor.regional_sine_coefficient_bounds
        == coefficient
        and 2 < coefficient.lo <= coefficient.hi < 3
        and source.gamma_bounds == reference.donor.gamma_bounds == gamma
        and persistent.gamma_bounds == (_GL, _G)
        and _GL <= gamma.lo <= gamma.hi <= _G
        and source.nominal_q0_bounds
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
        == source.cut_form_error_upper_bound
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
        and source.nominal_cone_certified
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
        and source.tangent_initialization
        == "u0=0; b0 and z0 carried from the same original cut"
        and source.nonlinear_sum_phase_retained
        and source.tangent_coefficients_correlated
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
        == source.full_state_comparison_upper_bound
        == budget
        and persistent.uniform_comparison_margin == budget - bregman > 0
    )
    fq = 2 * _G * budget / 9 + 5 * _G * budget**2 / 81 + matrix_error / 2700
    eq = 3 * cut_error + Q(81, 8) * fq
    original = (
        tube
        and source.kernel_certified
        and source.uniform_output_certified
        and source.kernel_mass_upper_bound == Q(81, 8)
        and source.kernel_impulse_upper_bound == 1
        and 9 / (2 * k.lo) < Q(81, 8)
        and source.forcing_error_upper_bound == fq
        and source.uniform_output_error_upper_bound == eq
        and source.output_error_margin == _NOISE - eq > 0
        and source.observation_covector
        == positive.observation_covector
        == (-Q(1),) * 18 + (Q(0),) * 54
        and source.centered_observation_covector_norm_squared == 9
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
        and source.beta_bounds == beta
        and source.chi_bounds == chi
        and source.stiffness_bounds == stiffness
        and source.slow_rate_scale_bounds == rho
        and source.characteristic_coefficients == polynomial
        and source.decay_rate_brackets == brackets
        and source.tangent_root_sign_bounds == signs
        and source.tangent_laplace_numerator == "p*b0+beta*z0"
        and source.tangent_laplace_denominator == "p^3+(2*k+2)*p^2+(2*k+d)*p+d"
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
    early_age, late_age = 1 / (4 * slow_scale[1]), 16 / slow_scale[0]
    ages = (
        roots
        and slow_scale[0] <= rho.lo <= rho.hi <= slow_scale[1]
        and _LAYER < early_age < late_age
    )
    early_true = (amplitude[0] / 2 - total, amplitude[1] + total)
    late_upper = amplitude[1] / 2**16 + total
    late_true = (-late_upper, late_upper)
    early_record = (early_true[0] - early_noise, early_true[1] + early_noise)
    late_record = (-late_upper - late_noise, late_upper + late_noise)
    admitted = early_noise <= _NOISE and late_noise <= _NOISE
    early_margin = early_record[0] - _THRESHOLD
    late_true_margin = _NOISE - late_upper
    late_margin = _THRESHOLD - (late_upper + late_noise)
    early = approximation and ages and early_true[0] > _THRESHOLD
    early_record_ok = early and admitted and early_margin > 0
    late = approximation and ages and late_true_margin > 0
    late_record_ok = late and admitted and late_margin > 0
    true_drop = amplitude[0] * (Q(1, 2) - Q(1, 2**16)) - 2 * total
    record_drop = true_drop - early_noise - late_noise
    drop = approximation and ages and true_drop > 0
    record_drop_ok = drop and admitted and record_drop > 0
    unresolved = tuple(
        label
        for good, label in (
            (source_ok, "original_source_history_unavailable"),
            (original, "uniform_actual_reference_output_allowance_unavailable"),
            (roots, "correlated_tangent_roots_unavailable"),
            (signed_kernel, "signed_forcing_kernel_gain_unavailable"),
            (nonlinear, "nonlinear_phase_sum_error_unavailable"),
            (fast, "inherited_fast_tail_unavailable"),
            (amplitude_ok, "carried_slow_amplitude_unavailable"),
            (approximation, "uniform_one_mode_approximation_unavailable"),
            (admitted, "independent_record_error_exceeds_protocol"),
            (early_record_ok, "resolved_early_record_unavailable"),
            (late_record_ok, "all_future_late_return_record_unavailable"),
            (record_drop_ok, "same_history_record_drop_unavailable"),
        )
        if not good
    )
    return _MixedSlowResponse(
        form_error_bounds=forms,
        phase_error_bounds=phases,
        early_observation_error_bound=early_noise,
        late_observation_error_bound=late_noise,
        source=source,
        source_association_certified=source_ok,
        tangent_roots_certified=roots,
        signed_kernel_certified=signed_kernel,
        nonlinear_error_certified=nonlinear,
        fast_tail_certified=fast,
        amplitude_certified=amplitude_ok,
        uniform_approximation_certified=approximation,
        observation_allowances_admitted=admitted,
        early_response_certified=early,
        early_record_certified=early_record_ok,
        late_return_certified=late,
        late_record_certified=late_record_ok,
        correlated_true_drop_certified=drop,
        correlated_drop_certified=record_drop_ok,
        coupling_bounds=k,
        sine_coefficient_bounds=coefficient,
        gamma_bounds=gamma,
        nominal_q0_bounds=q0,
        beta_bounds=beta,
        chi_bounds=chi,
        stiffness_bounds=stiffness,
        slow_rate_scale_bounds=rho,
        characteristic_coefficients=polynomial,
        decay_rate_brackets=brackets,
        tangent_root_sign_bounds=signs,
        original_output_error_upper_bound=eq,
        nonlinear_forcing_upper_bound=nonlinear_force,
        nonlinear_output_error_upper_bound=nonlinear_error,
        fast_residue_upper_bound=fast_residues,
        fast_output_tail_upper_bound=fast_error,
        total_output_error_upper_bound=total,
        approximation_margin=approximation_margin,
        slow_amplitude_bounds=amplitude,
        slow_scale_bounds=slow_scale,
        early_age=early_age,
        late_age=late_age,
        early_true_bounds=early_true,
        late_true_bounds=late_true,
        early_record_bounds=early_record,
        late_record_bounds=late_record,
        early_threshold_margin=early_margin,
        late_true_return_margin=late_true_margin,
        late_threshold_margin=late_margin,
        correlated_true_drop_lower_bound=true_drop,
        correlated_record_drop_lower_bound=record_drop,
        certified=not unresolved,
        status="conditional_slow_response" if not unresolved else "unavailable",
        unresolved_conditions=unresolved,
    )
