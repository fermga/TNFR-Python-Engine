"""A supplied reciprocal phase port and its conditional aggregate accuracy.

The original reached member is retained at the same A-contact age. A fresh
complete field moves the relative phase mean; neither the old fixed gauges
nor the exact form-only scalar law is imposed on this new response.
"""

from dataclasses import dataclass
from fractions import Fraction as Q

from ..mathematics._interval_taylor import Jet
from ..mathematics._interval_taylor import sin as jet_sin
from ..mathematics._rational_interval import I, cos, pi_interval, sin
from ._mixed_reciprocal_partner import _nonnegative, _nonnegative_pair
from ._mixed_registered_phase_probe import _derive_registered_phase_probe_geometry
from ._mixed_renewed_interface import (
    _assess_mixed_renewed_interface,
    _MixedRenewedInterface,
)
from ._mixed_resource_transfer_obstruction import _original_source_matches
from ._mixed_two_channel_circulation import _mixed_two_channel_field
from .relational_observations import _ordered

_DELTA = Q(1, 1024)
_WEIGHT = Q(2, 10**60)
_MATURE = Q(1, 2**28)
_ETA = Q(2, 10**69)
_SECTION = 3 * _DELTA / 16
_SECTION_ERROR = Q(3, 2) * (14 * _ETA + 2 * 10**10 * _WEIGHT)
_GL, _G = Q(1, 3300), Q(1, 3000)
_ACTIVATION, _SWITCH, _DURATION = Q(2010000), Q(3375000), Q(8)
_NOISE, _THRESHOLD = Q(1, 10**9), Q(1, 10**8)
_PAIRS = tuple((j, 18 + j) for j in range(18))


def _mixed_phase_sensitive_field(
    *, first_new_edge_conductances, second_new_edge_conductances
):
    """Add both reciprocal phase-port rows to the existing full form field.

    The same 72 coordinates, internal degree-nine laws and unit capacities
    remain. The external matching contributes gamma*sin(theta_R-theta_D)
    to donor form and gamma*(x_D-x_R) to donor phase, with opposite recipient
    contributions. These supplied inputs do not change neighbor divisors.
    The complete raw state is admitted before any trigonometric arithmetic.
    """
    full = _mixed_two_channel_field(
        first_new_edge_conductances=first_new_edge_conductances,
        second_new_edge_conductances=second_new_edge_conductances,
        law="full",
    )
    gamma = 1 / (1023 * pi_interval())

    def flow(values):
        state = _ordered(values, "state", limit=73)
        if len(state) != 72:
            raise ValueError("state requires all 36 forms followed by all 36 phases")
        jets = tuple(isinstance(value, Jet) for value in state)
        if any(jets):
            if not all(jets) or any(value.order != state[0].order for value in state):
                raise ValueError("state Taylor jets must all have the same order")
            sine = jet_sin
        else:
            state = tuple(I.coerce(value) for value in state)
            sine = sin
        rates = list(full(state))
        for donor, recipient in _PAIRS:
            current = sine(state[36 + recipient] - state[36 + donor]) * gamma
            transport = (state[donor] - state[recipient]) * gamma
            rates[donor] += current
            rates[recipient] -= current
            rates[36 + donor] += transport
            rates[36 + recipient] -= transport
        return tuple(rates)

    return flow


@dataclass(frozen=True)
class _MixedPhaseSensitiveInterface:
    """Conditional finite accuracy, with true and noisy equivalence separated."""

    form_error_bounds: tuple[Q, Q]
    phase_error_bounds: tuple[Q, Q]
    phase_sensitive_observation_error_bound: Q
    form_only_observation_error_bound: Q
    source: _MixedRenewedInterface
    source_association_certified: bool
    relative_phase_registration_certified: bool
    aggregate_phase_law_certified: bool
    phase_moment_remainder_certified: bool
    fresh_storage_certified: bool
    finite_identity_certified: bool
    collective_prediction_certified: bool
    scalar_effect_bound_certified: bool
    scalar_tolerance_certified: bool
    observation_allowances_admitted: bool
    phase_sensitive_return_certified: bool
    form_only_return_certified: bool
    same_threshold_classification_certified: bool
    same_member_record_overlap_certified: bool
    nominal_q0_bounds: tuple[Q, Q]
    coupling_bounds: I
    gamma_bounds: I
    slow_amplitude_bounds: tuple[Q, Q]
    inherited_output_error_upper_bound: Q
    phase_mean_error_upper_bound: Q
    relative_phase_entry_error_upper_bound: Q
    entry_norm_upper_bound: Q
    norm_growth_denominator: Q
    post_norm_upper_bound: Q
    phase_drift_upper_bound: Q
    marked_gap_upper_bound: Q
    registration_margin: Q
    storage_jump_bounds: tuple[Q, Q]
    nonlinear_moment_remainder_upper_bound: Q
    nonlinear_output_error_upper_bound: Q
    collective_initialization_error_upper_bound: Q
    collective_prediction_error_upper_bound: Q
    aggregate_uniform_upper_bound: Q
    relative_phase_upper_bound: Q
    scalar_effect_upper_bound: Q
    scalar_initialization_error_upper_bound: Q
    scalar_prediction_error_upper_bound: Q
    scalar_tolerance_margin: Q
    phase_sensitive_true_bounds: tuple[Q, Q]
    form_only_true_bounds: tuple[Q, Q]
    phase_sensitive_record_bounds: tuple[Q, Q]
    form_only_record_bounds: tuple[Q, Q]
    record_difference_absolute_upper_bound: Q
    same_member_record_overlap_margin: Q
    phase_sensitive_return_margin: Q
    form_only_return_margin: Q
    phase_sensitive_threshold_margin: Q
    form_only_threshold_margin: Q
    certified: bool
    status: str
    unresolved_conditions: tuple[str, ...]
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
    new_transport_norm_upper_bound: Q = Q(36)
    new_phase_hessian_norm_upper_bound: Q = Q(36)
    scalar_damping_kernel_mass_upper_bound: Q = Q(1, 2)
    scalar_decay_upper_bound: Q = Q(1, 2**16)
    comparison_loss_divisor: Q = Q(9)
    original_donor_form_total: Q = Q(0)
    whole_form_mean: Q = Q(0)
    observation_covector: tuple[Q, ...] = (-Q(1),) * 18 + (Q(0),) * 54
    phase_sensitive_observation_error_cap: Q = _NOISE
    form_only_observation_error_cap: Q = _NOISE
    intrinsic_output_tolerance: Q = _NOISE
    true_return_target: Q = _NOISE
    threshold: Q = _THRESHOLD
    aggregate_equation: str = "Q_prime=-2*Q-gamma*M"
    relative_phase_equation: str = "psi_prime=2*gamma*Q/9"
    phase_moment_equation: str = (
        "M_prime=sum(cos(delta_j)*(theta_Rj_prime-theta_Dj_prime))"
    )
    collective_coordinates: tuple[str, str] = ("Q", "9*psi")
    collective_generator: str = "[[-2,-2*gamma],[2*gamma,0]]"
    collective_initializer: str = (
        "(Q_s(3375000),0), with paid original phase-mean error"
    )
    scalar_predictor: str = "Q_s(3375000)*exp(-2*s)"
    comparison_storage: str = ".5*x.T*(H+9G)*x+Uint+9V-Uplus(Theta)"
    comparison_storage_rate: str = "-norm((H+9G)*x)**2/9"
    switching_storage_difference: str = "(9/2)*x.T*G*x+9V-9k*x.T*P_rel*x"
    whole_phase_gauge_only: bool = True
    original_phase_means_preserved_after_switch: bool = False
    initial_relative_phase_mean_assumed_zero: bool = False
    original_phase_lifts_retained: bool = True
    same_actual_member_comparator: bool = True
    same_input_collective_prediction: bool = True
    form_only_is_boundary_effect_control: bool = True
    original_baseline_retained: bool = True
    inherited_amplitude_rate_correlation_retained: bool = True
    full_nonlinear_state_closure: bool = False
    exact_nonlinear_two_coordinate_closure: bool = False
    scalar_exact_under_phase_port: bool = False
    phase_moment_claimed_zero: bool = False
    independent_record_errors: bool = True
    all_record_values_claimed_equal: bool = False
    observation_error_cancellation_assumed: bool = False
    all_future_changed_port_identity_claimed: bool = False
    zero_switching_storage_jump_claimed: bool = False
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
    clock: str = (
        "tau=(1023/1024)*t; complete internal and phase-sensitive matching rows"
    )


def _assess_mixed_phase_sensitive_interface(
    *,
    form_error_bounds,
    phase_error_bounds,
    phase_sensitive_observation_error_bound,
    form_only_observation_error_bound,
):
    """Re-admit the reached source and bound one fixed reciprocal phase task.

    The fresh previous reader owns the complete inherited source proof. This
    consumer checks its required primitives and reconstructs the particular
    initialization and tube allowances used here. Its old endpoint, storage,
    independent-gauge continuation and global verdict are not new premises.
    """
    forms = _nonnegative_pair(form_error_bounds, "form_error_bounds")
    phases = _nonnegative_pair(phase_error_bounds, "phase_error_bounds")
    phase_noise = _nonnegative(
        phase_sensitive_observation_error_bound,
        "phase_sensitive_observation_error_bound",
    )
    form_noise = _nonnegative(
        form_only_observation_error_bound, "form_only_observation_error_bound"
    )
    source = _assess_mixed_renewed_interface(
        form_error_bounds=forms,
        phase_error_bounds=phases,
        full_observation_error_bound=Q(0),
        continued_observation_error_bound=Q(0),
    )
    slow = source.source
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
    gamma = 1 / (1023 * pi)
    coupling = Q(13, 27) - _DELTA / (18 * pi)
    coefficient = 1 + 2 * cos(pi / 9)
    qlo, qhi = _SECTION - _SECTION_ERROR / 2, _SECTION + _SECTION_ERROR / 2
    budget = 200 * _MATURE
    cut_error = 8 * _MATURE / (1 + Q(10000, 3)) + 6 * _G * 8 * _MATURE + 6 * _WEIGHT
    output_force = 2 * _G * budget / 9 + 5 * _G * budget**2 / 81 + 14 * _WEIGHT / 2700
    output_error = 3 * cut_error + Q(81, 8) * output_force
    inherited_error = output_error + 3 * _G * qhi**3 + 27 * _G * qhi / 2**16
    amplitude = (
        Q(4, 3) * _GL * Q(233, 300) * qlo * (1 - _G**2 / Q(4, 9)),
        16 * _G * qhi,
    )
    growth = 135 * qhi
    comparison = 3 * (24 * _MATURE + 6 * (14 * _WEIGHT) * 10 * qhi / _GL) / (1 - growth)
    source_ok = (
        source.source_association_certified
        and slow.source_association_certified
        and source.form_error_bounds == slow.form_error_bounds == forms
        and source.phase_error_bounds == slow.phase_error_bounds == phases
        and source.full_observation_error_bound
        == source.continued_observation_error_bound
        == slow.early_observation_error_bound
        == slow.late_observation_error_bound
        == 0
        and geometry == prefix.source_geometry == reference.source_geometry
        and registered.cross_symmetry_certified
        and geometry.support_certified
        and geometry.reference_stationary_certified
        and geometry.spectral_gap_certified
        and geometry.nodes == tuple(range(18))
        and geometry.degrees == (9,) * 18
        and geometry.held_capacities == (Q(1),) * 18
        and geometry.bare_gap_lower_bound == 4
        and geometry.weighted_gap_lower_bound >= 3
        and reference.sources
        == (reference.donor.source, reference.recipient_readiness.sources[0])
        and all(
            _original_source_matches(row, geometry, rx, rt)
            for row, rx, rt in zip(reference.sources, forms, phases, strict=True)
        )
        and reference.actual_weight_deviation_upper_bound == _WEIGHT
        and reference.original_relative_phase_gap == _DELTA
        and reference.original_supplied_phase_origins == (-_DELTA / 2, _DELTA / 2)
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
        and source.original_donor_form_total == collective.initial_donor_form_total == 0
        and source.whole_form_mean == persistent.whole_form_mean == 0
        and collective.whole_form_mean
        == storage.whole_form_mean
        == positive.whole_form_mean
        == 0
        and positive.prefix_full_duration
        == handoff.prefix_duration
        == prefix.contact_duration
        == Q(2000000)
        and positive.prefix_cut_duration == handoff.continuation_duration == Q(10000)
        and handoff.switch_word == (1, 0)
        and source.activation_time
        == slow.activation_time
        == positive.handoff_time
        == handoff.total_duration
        == persistent.activation_time
        == _ACTIVATION
        and source.switch_age == slow.early_age == _SWITCH
        and source.switch_time == _ACTIVATION + _SWITCH
        and source.post_duration == _DURATION
        and source.total_duration == _ACTIVATION + _SWITCH + _DURATION
        and source.original_clock_horizon
        == (_ACTIVATION + _SWITCH + _DURATION) * Q(1024, 1023)
        and source.full_contact_pairs == _PAIRS
        and persistent.physical_capacities
        == positive.physical_capacities
        == reference.physical_capacities
        == (Q(1),) * 36
        and persistent.full_state_dimension
        == positive.full_state_dimension
        == reference.full_state_dimension
        == 72
        and positive.contact_pairs
        == persistent.contact_pairs
        == tuple((j, 18 + j) for j in range(9))
        and collective.fixed_port_strength
        == persistent.fixed_port_strength
        == storage.fixed_port_strength
        == 1
        and source.original_baseline_retained
        and source.counterfactuals_share_reached_actual_member
        and slow.same_actual_history_required
        and slow.correlated_initialization_retained
        and slow.initialization == "u0=0; b0 and z0 carried from the same original cut"
        and source.inherited_amplitude_rate_correlation_retained
        and not any(
            (
                source.state_reset_performed,
                source.original_source_reacquired,
                source.observation_rebased_at_switch,
                slow.state_reset_performed,
                slow.fitted_amplitude_or_rate,
                slow.amplitude_rate_marginals_independently_attainable,
                handoff.state_reset_performed,
                positive.state_reset_performed,
                persistent.state_reset_performed,
            )
        )
        and source.clock == "tau=(1023/1024)*t; all internal rows and renewed form port"
        and slow.clock
        == collective.clock
        == "tau=(1023/1024)*t; age from original unit A-port activation"
        and persistent.clock
        == "tau=(1023/1024)*t; same complete law and unit A form port"
        and positive.clock
        == "tau=(1023/1024)*t; complete internal rows and reciprocal A form port"
        and handoff.clock == "tau=(1023/1024)*t; all complete internal rows retained"
        and prefix.clock == "tau=(1023/1024)*t; complete full and cut internal laws"
        and reference.clock
        == "tau=(1023/1024)*t; complete internal rows and reciprocal form inputs"
        and source.nominal_q0_bounds == slow.nominal_q0_bounds == (qlo, qhi)
        and source.gamma_bounds == slow.gamma_bounds == gamma
        and source.coupling_bounds == slow.coupling_bounds == coupling
        and _GL < gamma.lo <= gamma.hi < _G
        and Q(4, 9) < coupling.lo <= coupling.hi < Q(1, 2)
    )
    inherited = (
        source_ok
        and slow.uniform_approximation_certified
        and slow.tangent_roots_certified
        and slow.signed_kernel_certified
        and slow.nonlinear_error_certified
        and slow.fast_tail_certified
        and slow.amplitude_certified
        and collective.cut_form_error_upper_bound == cut_error
        and slow.original_output_error_upper_bound == output_error
        and slow.nonlinear_output_error_upper_bound == 3 * _G * qhi**3
        and slow.fast_output_tail_upper_bound == 27 * _G * qhi / 2**16
        and source.inherited_output_error_upper_bound
        == source.scalar_initialization_error_upper_bound
        == slow.total_output_error_upper_bound
        == inherited_error
        and source.slow_amplitude_bounds == slow.slow_amplitude_bounds == amplitude
        and 0 < inherited_error < _NOISE
        and source.entry_aggregate_bounds
        == (amplitude[0] / 2 - inherited_error, amplitude[1] + inherited_error)
        and source.entry_norm_upper_bound == 4 * qhi + budget
        and persistent.uniform_tube_certified
        and persistent.bregman_comparison_certified
        and persistent.phase_convexity_certified
        and persistent.nominal_cone_certified
        and persistent.nominal_norm_upper_bound == 4 * qhi
        and persistent.uniform_comparison_budget == budget
        and persistent.uniform_comparison_upper_bound == comparison < budget
        and persistent.bregman_growth_integral_upper_bound == growth < 1
        and persistent.bregman_denominator == 1 - growth
        and persistent.marked_gap_upper_bound == qhi + 300 * _MATURE
        and persistent.registration_gap_limit == _DELTA / 4
        and storage.selected_mu == persistent.selected_mu == 1
        and storage.storage_domain_certified
        and storage.actual_dissipation_certified
        and storage.mean_projector_rank == 1
        and storage.cross_strength_bounds == 9 * coupling
        and storage.origin_reference
        == "conserved_whole_form_mean; P_rel=P_copymean-P_wholemean"
        and persistent.coupling_bounds == coupling
        and persistent.sine_coefficient_bounds == coefficient
    )
    # Theta and the two supplied within-copy origins each sum to zero.
    # Hence the raw reset bound controls the actual copy mean directly.
    # The separately centered 6*eta deviation is not used as a mean bound.
    gauge = (
        inherited
        and sum(geometry.target_phase_turns, Q(0)) == 0
        and sum(reference.original_supplied_phase_origins, Q(0)) == 0
        and all(row.phase_reset_error_bound == 3 * _ETA for row in reference.sources)
        and persistent.original_phase_gauges_and_branches_retained
        and positive.original_phase_gauges_and_branches_retained
        and handoff.original_phase_gauges_and_branches_retained
    )
    phase_mean_error = 3 * _ETA
    relative_phase_entry = 2 * phase_mean_error
    entry_norm = 4 * qhi + budget + 18 * _ETA
    denominator = 1 - 4 * _G * _DURATION
    norm = entry_norm / denominator
    phase_drift = 6 * _G * _DURATION * norm
    marked = qhi + 300 * _MATURE + phase_drift
    registration = _DELTA / 4 - marked
    identity = (
        gauge and denominator > 0 and norm < Q(1, 100) < Q(1, 32) and registration > 0
    )
    # Fresh matching potential and both reciprocal rows give the exact loss.
    # On common relative forms G=2P_rel; 9k<9/2 makes the switch cost positive.
    storage_ok = identity and 0 < coupling.lo <= coupling.hi < Q(1, 2)
    jump = (Q(0), 9 * entry_norm**2)
    law = source_ok and gauge
    remainder = norm**3 / 2
    moment = law and identity
    forcing_error = _G * _DURATION * remainder
    initial_error = inherited_error + 9 * relative_phase_entry
    collective_error = initial_error + forcing_error
    collective_ok = moment and storage_ok and collective_error < _NOISE
    aggregate_bound = amplitude[1] + collective_error
    relative_phase = relative_phase_entry + 2 * _G * _DURATION * aggregate_bound / 9
    effect = (
        54 * _G * _ETA + 2 * _G**2 * _DURATION * aggregate_bound + _G * remainder / 2
    )
    effect_ok = collective_ok and effect > 0
    scalar_initial_error = inherited_error / 2**16
    scalar_error = effect + scalar_initial_error
    tolerance_margin = _NOISE - scalar_error
    tolerance = effect_ok and effect < _NOISE and tolerance_margin > 0
    form_upper = (amplitude[1] + inherited_error) / 2**16
    form_bounds = (Q(0), form_upper)
    phase_bounds = (-effect, form_upper + effect)
    phase_return_margin = _NOISE - (form_upper + effect)
    form_return_margin = _NOISE - form_upper
    phase_return = tolerance and phase_return_margin > 0
    form_return = (
        inherited and amplitude[0] / 2 - inherited_error > 0 and form_return_margin > 0
    )
    phase_record = (phase_bounds[0] - phase_noise, phase_bounds[1] + phase_noise)
    form_record = (-form_noise, form_upper + form_noise)
    admitted = phase_noise <= _NOISE and form_noise <= _NOISE
    phase_threshold = _THRESHOLD - max(abs(value) for value in phase_record)
    form_threshold = _THRESHOLD - max(abs(value) for value in form_record)
    threshold = (
        phase_return
        and form_return
        and admitted
        and min(phase_threshold, form_threshold) > 0
    )
    overlap_margin = phase_noise + form_noise - effect
    overlap = effect_ok and admitted and overlap_margin >= 0
    # Overlap is an optional consequence of the supplied noise budgets. It is
    # not required for the true tolerance or common-threshold conclusions.
    certified = threshold and collective_ok and storage_ok and tolerance
    unresolved = tuple(
        label
        for good, label in (
            (inherited, "inherited_source_or_initializer_unavailable"),
            (gauge, "original_phase_mean_registration_unavailable"),
            (identity, "fresh_finite_identity_unavailable"),
            (storage_ok, "fresh_storage_and_switching_work_unavailable"),
            (collective_ok, "two_coordinate_prediction_error_not_certified"),
            (tolerance, "scalar_task_tolerance_not_certified"),
            (admitted, "independent_observation_errors_outside_protocol"),
            (threshold, "common_threshold_classification_not_certified"),
        )
        if not good
    )
    return _MixedPhaseSensitiveInterface(
        form_error_bounds=forms,
        phase_error_bounds=phases,
        phase_sensitive_observation_error_bound=phase_noise,
        form_only_observation_error_bound=form_noise,
        source=source,
        source_association_certified=inherited,
        relative_phase_registration_certified=gauge,
        aggregate_phase_law_certified=law,
        phase_moment_remainder_certified=moment,
        fresh_storage_certified=storage_ok,
        finite_identity_certified=identity,
        collective_prediction_certified=collective_ok,
        scalar_effect_bound_certified=effect_ok,
        scalar_tolerance_certified=tolerance,
        observation_allowances_admitted=admitted,
        phase_sensitive_return_certified=phase_return,
        form_only_return_certified=form_return,
        same_threshold_classification_certified=threshold,
        same_member_record_overlap_certified=overlap,
        nominal_q0_bounds=(qlo, qhi),
        coupling_bounds=coupling,
        gamma_bounds=gamma,
        slow_amplitude_bounds=amplitude,
        inherited_output_error_upper_bound=inherited_error,
        phase_mean_error_upper_bound=phase_mean_error,
        relative_phase_entry_error_upper_bound=relative_phase_entry,
        entry_norm_upper_bound=entry_norm,
        norm_growth_denominator=denominator,
        post_norm_upper_bound=norm,
        phase_drift_upper_bound=phase_drift,
        marked_gap_upper_bound=marked,
        registration_margin=registration,
        storage_jump_bounds=jump,
        nonlinear_moment_remainder_upper_bound=remainder,
        nonlinear_output_error_upper_bound=forcing_error,
        collective_initialization_error_upper_bound=initial_error,
        collective_prediction_error_upper_bound=collective_error,
        aggregate_uniform_upper_bound=aggregate_bound,
        relative_phase_upper_bound=relative_phase,
        scalar_effect_upper_bound=effect,
        scalar_initialization_error_upper_bound=scalar_initial_error,
        scalar_prediction_error_upper_bound=scalar_error,
        scalar_tolerance_margin=tolerance_margin,
        phase_sensitive_true_bounds=phase_bounds,
        form_only_true_bounds=form_bounds,
        phase_sensitive_record_bounds=phase_record,
        form_only_record_bounds=form_record,
        record_difference_absolute_upper_bound=effect + phase_noise + form_noise,
        same_member_record_overlap_margin=overlap_margin,
        phase_sensitive_return_margin=phase_return_margin,
        form_only_return_margin=form_return_margin,
        phase_sensitive_threshold_margin=phase_threshold,
        form_only_threshold_margin=form_threshold,
        certified=certified,
        status=(
            "conditional_phase_port_scalar_tolerance" if certified else "unavailable"
        ),
        unresolved_conditions=unresolved,
    )
