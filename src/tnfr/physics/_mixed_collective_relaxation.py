"""Collective output filtering and phase-restoring persistence at a fixed age.

The original actual state is shared with a form-only diffusion comparator.
A positive response kernel prices retained source and fine-phase uncertainty
without multiplying it by time. Nominal tangent modes remain distinct from
nonlinear actual response. No trajectory, eigenvalue solver or record runs.
"""

from dataclasses import dataclass
from fractions import Fraction as Q

from ..mathematics._rational_interval import I, cos, pi_interval
from ._mixed_persistent_contact import (
    _assess_mixed_persistent_contact,
    _MixedPersistentContact,
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
_GAMMA_LOWER, _GAMMA_UPPER = Q(1, 3300), Q(1, 3000)
_ACTIVATION = Q(2010000)
_EARLY = Q(1, 2)
_HORIZON = Q(64)
_NOISE = Q(1, 10**9)
_THRESHOLD = Q(1, 10**8)


@dataclass(frozen=True)
class _MixedCollectiveRelaxation:
    """Conditional actual output, matched comparator and nominal tangent bounds."""

    form_error_bounds: tuple[Q, Q]
    phase_error_bounds: tuple[Q, Q]
    full_observation_error_bound: Q
    form_only_observation_error_bound: Q
    source: _MixedPersistentContact
    source_association_certified: bool
    projected_rows_certified: bool
    kernel_certified: bool
    uniform_output_certified: bool
    nominal_cone_certified: bool
    form_only_gap_certified: bool
    full_persistence_certified: bool
    observation_allowances_admitted: bool
    independent_record_sets_separated: bool
    fixed_threshold_certified: bool
    tangent_certified: bool
    nominal_slow_mode_excited_certified: bool
    coupling_bounds: I
    sine_coefficient_bounds: I
    gamma_bounds: I
    nominal_q0_bounds: tuple[Q, Q]
    cut_form_error_upper_bound: Q
    full_state_comparison_upper_bound: Q
    forcing_error_upper_bound: Q
    uniform_output_error_upper_bound: Q
    output_error_margin: Q
    phase_difference_lower_bound: Q
    nominal_forcing_lower_bound: Q
    nominal_output_bounds: tuple[Q, Q]
    actual_form_damping_lower_bound: Q
    initial_form_norm_upper_bound: Q
    form_only_output_upper_bound: Q
    full_true_bounds: tuple[Q, Q]
    form_only_true_bounds: tuple[Q, Q]
    full_record_bounds: tuple[Q, Q]
    form_only_record_bounds: tuple[Q, Q]
    full_threshold_margin: Q
    form_only_threshold_margin: Q
    record_separation_margin: Q
    beta_bounds: I
    chi_bounds: I
    stiffness_bounds: I
    slow_rate_scale_bounds: I
    characteristic_coefficients: tuple[I, I, I, I]
    decay_rate_brackets: tuple[tuple[I, I], tuple[I, I], tuple[I, I]]
    tangent_root_sign_bounds: tuple[I, I, I, I, I, I]
    tangent_state_matrix: tuple[tuple[I, I, I], tuple[I, I, I], tuple[I, I, I]]
    certified: bool
    status: str
    unresolved_conditions: tuple[str, ...]
    kernel_mass_upper_bound: Q = Q(81, 8)
    kernel_impulse_upper_bound: Q = Q(1)
    nominal_form_damping_lower_bound: Q = Q(1, 3)
    form_only_decay_power: int = 16
    form_only_decay_upper_bound: Q = Q(1, 2**16)
    activation_time: Q = _ACTIVATION
    horizon: Q = _HORIZON
    total_duration: Q = _ACTIVATION + _HORIZON
    original_clock_horizon: Q = (_ACTIVATION + _HORIZON) * Q(1024, 1023)
    full_observation_error_cap: Q = _NOISE
    form_only_observation_error_cap: Q = _NOISE
    threshold: Q = _THRESHOLD
    observation_covector: tuple[Q, ...] = (-Q(1),) * 18 + (Q(0),) * 54
    centered_observation_covector_norm_squared: Q = Q(9)
    initial_donor_form_total: Q = Q(0)
    initial_common_form_difference: Q = Q(0)
    fixed_port_strength: Q = Q(1)
    whole_form_mean: Q = Q(0)
    nominal_state_dimension: int = 5
    tangent_output_subsystem_dimension: int = 3
    nonlinear_sum_phase_retained: bool = True
    tangent_laplace_numerator: str = "p*b0+beta*z0"
    tangent_laplace_denominator: str = "p^3+(2*k+2)*p^2+(2*k+d)*p+d"
    tangent_initialization: str = "u0=0; b0 and z0 carried from the same original cut"
    tangent_coefficients_correlated: bool = True
    all_tangent_modes_excited_claimed: bool = False
    numerical_eigenvalues_evaluated: bool = False
    observed_slow_rate_identified: bool = False
    actual_nonlinear_slow_timescale_identified: bool = False
    same_actual_initial_state_required: bool = True
    same_actual_transport_required: bool = True
    independent_record_errors: bool = True
    observation_error_cancellation_assumed: bool = False
    exact_actual_five_coordinate_closure: bool = False
    source_error_reinitialized: bool = False
    source_association_authenticated: bool = False
    form_only_is_actual_complete_law: bool = False
    state_reset_performed: bool = False
    source_trajectory_evaluated: bool = False
    numerical_response_allowance: Q = Q(0)
    newly_observed_aggregates: int = 0
    required_endpoint_records: int = 2
    runtime_event_executed: bool = False
    physical_timescale_identified: bool = False
    empirical_validation: bool = False
    clock: str = "tau=(1023/1024)*t; age from original unit A-port activation"


def _assess_mixed_collective_relaxation(
    *,
    form_error_bounds,
    phase_error_bounds,
    full_observation_error_bound,
    form_only_observation_error_bound,
):
    """Rebuild filtered actual output and one fixed matched simpler model."""
    forms = _nonnegative_pair(form_error_bounds, "form_error_bounds")
    phases = _nonnegative_pair(phase_error_bounds, "phase_error_bounds")
    full_noise = _nonnegative(
        full_observation_error_bound, "full_observation_error_bound"
    )
    form_noise = _nonnegative(
        form_only_observation_error_bound, "form_only_observation_error_bound"
    )
    source = _assess_mixed_persistent_contact(
        form_error_bounds=forms, phase_error_bounds=phases
    )
    storage = source.source
    positive = storage.source
    handoff = positive.handoff
    prefix = handoff.prefix
    reference = prefix.reference
    registered = _derive_registered_phase_probe_geometry()
    geometry = registered.successor
    pi = pi_interval()
    gamma = 1 / (1023 * pi)
    coupling = Q(13, 27) - _DELTA / (18 * pi)
    sine_coefficient = 1 + 2 * cos(pi / 9)
    q0 = (_SECTION - _SECTION_ERROR / 2, _SECTION + _SECTION_ERROR / 2)
    energy = Q(27, 2) * _DELTA**2 + 486 * _ETA + 18 * _ETA**2 + 9 * _MATURE**2
    actual_loss = Q(1, 4) - 84 * _WEIGHT
    source_ok = (
        source.form_error_bounds
        == storage.form_error_bounds
        == positive.form_error_bounds
        == forms
        and source.phase_error_bounds
        == storage.phase_error_bounds
        == positive.phase_error_bounds
        == phases
        and source.source_association_certified
        and storage.source_association_certified
        and positive.source_association_certified
        and handoff.source_association_certified
        and reference.source_association_certified
        and reference.donor_source_certified
        and reference.recipient_source_certified
        and registered.cross_symmetry_certified
        and geometry == prefix.source_geometry == reference.source_geometry
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
        and positive.initial_donor_form_total
        == positive.whole_form_mean
        == source.whole_form_mean
        == storage.whole_form_mean
        == 0
        and positive.exact_zero_entry_total_from_prior_laws
        and positive.physical_capacities
        == reference.physical_capacities
        == source.physical_capacities
        == (Q(1),) * 36
        and positive.full_state_dimension
        == reference.full_state_dimension
        == source.full_state_dimension
        == 72
        and positive.contact_pairs
        == source.contact_pairs
        == tuple((j, 18 + j) for j in range(9))
        and positive.prefix_full_duration == handoff.prefix_duration == Q(2000000)
        and positive.prefix_cut_duration == handoff.continuation_duration == Q(10000)
        and positive.handoff_time
        == handoff.total_duration
        == source.activation_time
        == _ACTIVATION
        and source.activation_relative_common_form == 0
        and positive.positive_duration
        == source.previously_reached_duration
        == storage.horizon
        == _EARLY
        and positive.total_duration
        == source.previously_reached_time
        == _ACTIVATION + _EARLY
        and positive.original_clock_horizon == (_ACTIVATION + _EARLY) * Q(1024, 1023)
        and handoff.original_clock_horizon == _ACTIVATION * Q(1024, 1023)
        and source.clock == "tau=(1023/1024)*t; same complete law and unit A form port"
        and positive.clock
        == "tau=(1023/1024)*t; complete internal rows and reciprocal A form port"
        and handoff.clock == "tau=(1023/1024)*t; all complete internal rows retained"
        and prefix.clock == "tau=(1023/1024)*t; complete full and cut internal laws"
        and reference.clock
        == "tau=(1023/1024)*t; complete internal rows and reciprocal form inputs"
        and handoff.switch_word == (1, 0)
        and positive.observation_error_bound == handoff.observation_error_bound == 0
        and positive.handoff_observations == handoff.handoff_observations == 0
        and positive.inherited_form_velocity_retained
        and positive.relative_common_form_mode_retained
        and positive.original_phase_gauges_and_branches_retained
        and source.original_phase_gauges_and_branches_retained
        and handoff.reference_family_carried
        and handoff.reached_velocity_retained
        and handoff.source_weight_association_retained
        and handoff.original_phase_gauges_and_branches_retained
        and reference.inherited_donor_velocity_retained
        and not any(
            (
                source.state_reset_performed,
                storage.state_reset_performed,
                positive.state_reset_performed,
                handoff.state_reset_performed,
                source.port_switch_performed,
            )
        )
    )
    geometry_ok = (
        source_ok
        and positive.projected_rows_certified
        and geometry.nodes == tuple(range(18))
        and geometry.degrees == (9,) * 18
        and geometry.held_capacities == (Q(1),) * 18
        and geometry.support_certified
        and geometry.reference_stationary_certified
        and geometry.spectral_gap_certified
        and geometry.bare_gap_lower_bound == 4
        and geometry.weighted_gap_lower_bound >= 3
        and source.coupling_bounds
        == storage.coupling_bounds
        == reference.donor.nominal_coupling_bounds
        == coupling
        and storage.cross_strength_bounds == 9 * coupling
        and Q(4, 9) < coupling.lo <= coupling.hi < Q(1, 2)
        and source.sine_coefficient_bounds
        == reference.donor.regional_sine_coefficient_bounds
        == sine_coefficient
        and 2 < sine_coefficient.lo <= sine_coefficient.hi < 3
        and reference.donor.gamma_bounds == gamma
        and _GAMMA_LOWER <= gamma.lo <= gamma.hi <= _GAMMA_UPPER
        and source.gamma_bounds == (_GAMMA_LOWER, _GAMMA_UPPER)
        and source.nominal_q0_bounds
        == reference.nominal_q0_bounds
        == positive.nominal_q0_bounds
        == q0
        and 0 < q0[0] <= q0[1] < _DELTA / 4
    )
    # The initial donor total is exact; only b=-v_D+v_R has inherited form
    # uncertainty. No measured or freshly centered handoff state is inserted.
    error = 8 * _MATURE
    comparison_budget = 200 * _MATURE
    forcing = 2 * _WEIGHT
    matrix_error = 14 * _WEIGHT
    cut_error = error / (1 + Q(10000, 3)) + 6 * _GAMMA_UPPER * error + 3 * forcing
    rate = reference.conservative_metric_decay_rate
    old_comparison = (
        4 * (_MATURE + _SECTION_ERROR + 2 * forcing / rate) if rate > 0 else None
    )
    prefix_ok = (
        geometry_ok
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
        == old_comparison
        and old_comparison < error
        and reference.comparison_forcing_upper_bound
        == handoff.metric_forcing_upper_bound
        == forcing
        and reference.comparison_matrix_deviation_upper_bound == matrix_error
        and handoff.actual_observation_allowance
        == source.original_comparison_upper_bound
        == error
        and positive.cut_form_error_certified
        and positive.cut_decay_upper_bound == 1 / (1 + Q(10000, 3))
        and positive.cut_form_error_upper_bound == cut_error
        and positive.handoff_phase_error_upper_bound == error
    )
    cone_margins = (
        2 * coupling.lo / 3 - _GAMMA_UPPER**2,
        2 * coupling.lo
        - Q(1, 5)
        - sine_coefficient.hi / 9
        - 4 * _GAMMA_UPPER**2 * coupling.hi,
        Q(1, 5) - Q(24, 5) * _GAMMA_UPPER**2 * coupling.hi,
    )
    cone = (
        prefix_ok
        and source.nominal_cone_certified
        and handoff.continuation_wedges_certified
        and reference.nominal_wedges_certified
        and reference.donor.nominal_reference_certified
        and reference.nominal_velocity_factor_upper_bound == _GAMMA_UPPER**2
        and reference.nominal_wedge_margin == cone_margins[0] > 0
        and source.nominal_cone_margins == cone_margins
        and all(value > 0 for value in cone_margins)
        and source.nominal_common_form_cone_factor == Q(6, 5)
        and source.nominal_contrast_form_cone_factor == 1
        and source.nominal_norm_upper_bound == 4 * q0[1] < Q(1, 100)
        and reference.inherited_nominal_velocity_relation
        == "-gamma**2*q0 <= w0 <= 0 at the same isolated age"
        and positive.phase_difference_entry_lower_bound == Q(233, 300) * q0[0]
        and positive.inherited_form_contrast_lower_bound == q0[0] / 20000
        and positive.inherited_orientation_certified
    )
    ly, lx = 3 * q0[1], 10 * q0[1] / _GAMMA_LOWER
    bregman = 3 * (3 * error + 6 * matrix_error * lx) / (1 - 45 * ly)
    tube = (
        cone
        and source.broad_trap_certified
        and source.phase_convexity_certified
        and storage.storage_domain_certified
        and storage.actual_dissipation_certified
        and storage.actual_weight_perturbation_upper_bound == 84 * _WEIGHT
        and source.selected_mu == storage.selected_mu == 1
        and storage.mean_projector_rank == 1
        and storage.port_norm_upper_bound == 2
        and storage.loss_divisor == 9
        and source.bregman_comparison_certified
        and source.uniform_tube_certified
        and source.finite_variation_certified
        and source.initial_excess_storage_upper_bound
        == reference.initial_excess_storage_upper_bound
        == energy
        and source.full_norm_squared_upper_bound == 5 * energy < Q(1, 10000)
        and source.actual_dissipation_lower_bound
        == storage.actual_dissipation_lower_bound
        == actual_loss
        > 0
        and source.storage_coercivity_lower_bound == Q(1, 5)
        and source.storage_norm_upper_bound == 9
        and source.storage_barrier == Q(1, 5) * Q(1, 32) ** 2
        and source.storage_margin == source.storage_barrier - energy > 0
        and source.phase_radius == reference.phase_radius == Q(1, 32)
        and source.nominal_phase_variation_upper_bound == ly
        and source.nominal_form_integral_upper_bound == lx
        and source.bregman_growth_integral_upper_bound == 45 * ly < 1
        and source.bregman_denominator == 1 - 45 * ly
        and source.initial_bregman_root_upper_bound == 3 * error
        and source.weight_matrix_deviation_upper_bound == matrix_error
        and source.bregman_coercivity_lower_bound == Q(1, 5)
        and source.bregman_norm_upper_bound == 9
        and source.bregman_phase_growth_coefficient == 45
        and source.bregman_weight_forcing_coefficient == 6
        and source.uniform_comparison_upper_bound == bregman < comparison_budget
        and source.uniform_comparison_budget == comparison_budget
        and source.uniform_comparison_margin == comparison_budget - bregman > 0
    )
    kernel = (
        geometry_ok
        and prefix_ok
        and 9 / (2 * coupling.lo) < Q(81, 8)
        and positive.observation_covector == (-Q(1),) * 18 + (Q(0),) * 54
        and source.fixed_port_strength == storage.fixed_port_strength == 1
        and source.activation_relative_common_form == 0
    )
    forcing_error = 2 * _GAMMA_UPPER * comparison_budget / 9
    forcing_error += 5 * _GAMMA_UPPER * comparison_budget**2 / 81
    forcing_error += matrix_error / 2700
    output_error = 3 * cut_error + Q(81, 8) * forcing_error
    error_margin = _NOISE - output_error
    uniform = kernel and tube and error_margin > 0
    # Both nonlinear phase combinations stay live. The slow drift is bounded
    # on this one window; no phase is replaced by its tangent or frozen value.
    z_min = Q(233, 300) * q0[0] - 2 * _GAMMA_UPPER**2 * _HORIZON * q0[1]
    minimum_force = _GAMMA_LOWER * Q(2, 9) * Q(99, 100) * z_min
    nominal_lower = 9 * minimum_force * (_HORIZON / (1 + _HORIZON)) ** 2 / 2
    nominal_upper = 9 * Q(6, 5) * _GAMMA_UPPER * q0[1]
    nominal_output = (nominal_lower, nominal_upper)
    full_true = (nominal_lower - output_error, nominal_upper + output_error)
    full = (
        uniform
        and cone
        and z_min > 0
        and 1 - q0[1] ** 2 / 6 > Q(99, 100)
        and 2 * coupling.hi + 1 < 2
        and 0 < nominal_lower <= nominal_upper
        and full_true[0] > 0
    )
    actual_gap = Q(1, 3) - matrix_error / 9
    entry_form = 9 * _GAMMA_UPPER * q0[1] + cut_error
    form_upper = 3 * entry_form / 2**16
    form_gap = (
        geometry_ok
        and prefix_ok
        and cone
        and 4 * coupling.lo / 3 - Q(5, 9) > 0
        and 2 * coupling.lo > Q(1, 3)
        and geometry.weighted_gap_lower_bound / 9 >= Q(1, 3)
        and actual_gap > Q(1, 4)
        and _HORIZON / 4 == 16
        and entry_form > 0
        and form_upper < _NOISE
    )
    form_true = (-form_upper, form_upper)
    full_record = (full_true[0] - full_noise, full_true[1] + full_noise)
    form_record = (form_true[0] - form_noise, form_true[1] + form_noise)
    admitted = full_noise <= _NOISE and form_noise <= _NOISE
    full_margin = full_record[0] - _THRESHOLD
    form_margin = _THRESHOLD - max(abs(value) for value in form_record)
    separation = full_record[0] - form_record[1]
    separated = full and form_gap and admitted and separation > 0
    threshold = separated and full_margin > 0 and form_margin > 0
    # These intervals enclose correlated coefficients of ONE nominal tangent.
    # Sign changes, not an eigenvalue solver, isolate its three simple roots.
    beta, chi = gamma * sine_coefficient / 9, 4 * gamma * coupling
    rho = 2 * gamma**2 * sine_coefficient / 9
    stiffness = 2 * coupling * rho
    coefficients = (I(1), 2 * coupling + 2, 2 * coupling + stiffness, stiffness)
    signs = (
        rho**2 * (rho - 2),
        rho * (2 * coupling - (4 * coupling + 8) * rho + 8 * rho**2),
        3 * coupling / 8 - Q(7, 64) - 3 * stiffness / 4,
        coupling / 2 - Q(3, 8) - stiffness / 2,
        stiffness - 4 * coupling,
        9 - 12 * coupling + 2 * stiffness,
    )
    brackets = ((rho, 2 * rho), (I(Q(1, 4)), I(Q(1, 2))), (I(2), I(3)))
    tangent = (
        geometry_ok
        and cone
        and rho.lo > 0
        and 2 * rho.hi < Q(1, 4)
        and signs[0].hi < 0 < signs[1].lo
        and signs[2].lo > 0 > signs[3].hi
        and signs[4].hi < 0 < signs[5].lo
    )
    slow_excited = tangent and z_min > 0 and _GAMMA_UPPER**2 < coupling.lo
    tangent_matrix = (
        (I(-1), I(1), I(0)),
        (I(1), -2 * coupling - 1, beta),
        (I(0), -chi, I(0)),
    )
    unresolved = tuple(
        label
        for good, label in (
            (source_ok, "original_source_and_same_actual_initialization_unavailable"),
            (
                geometry_ok,
                "complete_projected_rows_and_original_coefficients_unavailable",
            ),
            (kernel, "positive_output_kernel_and_inherited_form_error_unavailable"),
            (cone, "correlated_nonlinear_nominal_cone_unavailable"),
            (uniform, "uniform_actual_nominal_output_error_unavailable"),
            (form_gap, "same_state_actual_form_only_decay_unavailable"),
            (full, "full_phase_restoring_output_at_fixed_age_unavailable"),
            (admitted, "independent_record_error_exceeds_protocol"),
            (separated, "full_and_form_only_records_not_certified_separate"),
            (threshold, "fixed_threshold_not_certified_between_records"),
            (tangent, "nominal_tangent_root_brackets_unavailable"),
            (slow_excited, "original_nominal_slow_mode_amplitude_unavailable"),
        )
        if not good
    )
    return _MixedCollectiveRelaxation(
        form_error_bounds=forms,
        phase_error_bounds=phases,
        full_observation_error_bound=full_noise,
        form_only_observation_error_bound=form_noise,
        source=source,
        source_association_certified=source_ok,
        projected_rows_certified=geometry_ok,
        kernel_certified=kernel,
        uniform_output_certified=uniform,
        nominal_cone_certified=cone,
        form_only_gap_certified=form_gap,
        full_persistence_certified=full,
        observation_allowances_admitted=admitted,
        independent_record_sets_separated=separated,
        fixed_threshold_certified=threshold,
        tangent_certified=tangent,
        nominal_slow_mode_excited_certified=slow_excited,
        coupling_bounds=coupling,
        sine_coefficient_bounds=sine_coefficient,
        gamma_bounds=gamma,
        nominal_q0_bounds=q0,
        cut_form_error_upper_bound=cut_error,
        full_state_comparison_upper_bound=comparison_budget,
        forcing_error_upper_bound=forcing_error,
        uniform_output_error_upper_bound=output_error,
        output_error_margin=error_margin,
        phase_difference_lower_bound=z_min,
        nominal_forcing_lower_bound=minimum_force,
        nominal_output_bounds=nominal_output,
        actual_form_damping_lower_bound=actual_gap,
        initial_form_norm_upper_bound=entry_form,
        form_only_output_upper_bound=form_upper,
        full_true_bounds=full_true,
        form_only_true_bounds=form_true,
        full_record_bounds=full_record,
        form_only_record_bounds=form_record,
        full_threshold_margin=full_margin,
        form_only_threshold_margin=form_margin,
        record_separation_margin=separation,
        beta_bounds=beta,
        chi_bounds=chi,
        stiffness_bounds=stiffness,
        slow_rate_scale_bounds=rho,
        characteristic_coefficients=coefficients,
        decay_rate_brackets=brackets,
        tangent_root_sign_bounds=signs,
        tangent_state_matrix=tangent_matrix,
        certified=not unresolved,
        status="conditional_collective_relaxation" if not unresolved else "unavailable",
        unresolved_conditions=unresolved,
    )
