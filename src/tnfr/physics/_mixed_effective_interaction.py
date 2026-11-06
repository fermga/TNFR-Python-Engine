"""One conditional nonlinear interaction model with carried prediction errors.

The complete phase-sensitive field remains unchanged. This private two-state
field retains signed aggregate form and its relative phase lift, with one law
for both admitted preparations. Analytical comparison bounds carry the same
predicted member through a bookkeeping seam; no state or trajectory is read.
"""

from dataclasses import dataclass
from fractions import Fraction as Q

from .._exact_time import exact_or_represented_real as _exact
from ..mathematics._interval_taylor import Jet
from ..mathematics._interval_taylor import sin as jet_sin
from ..mathematics._rational_interval import I, pi_interval, sin
from ._mixed_reciprocal_partner import _nonnegative, _nonnegative_pair
from ._mixed_registered_phase_probe import _derive_registered_phase_probe_geometry
from ._mixed_relative_registration import (
    _assess_mixed_relative_registration,
    _MixedRelativeRegistration,
)
from ._mixed_resource_transfer_obstruction import _original_source_matches
from .relational_observations import _ordered

_DELTA = Q(1, 1024)
_WEIGHT, _ETA, _MATURE = Q(2, 10**60), Q(2, 10**69), Q(1, 2**28)
_SECTION = 3 * _DELTA / 16
_SECTION_ERROR = Q(3, 2) * (14 * _ETA + 2 * 10**10 * _WEIGHT)
_GL, _G = Q(1, 3300), Q(1, 3000)
_ACTIVATION, _SWITCH = Q(2010000), Q(3375000)
_SEGMENT, _DURATION = Q(8), Q(16)
_DECAY = Q(1, 2**16)
_NOISE = Q(1, 10**9)
_PAIRS = tuple((j, 18 + j) for j in range(18))


def _mixed_effective_interaction_field():
    """Compile the same two rows for every admitted phase registration.

    State is exactly ``(Q, psi)``. The circular sine is evaluated on the
    retained real lift; neither wrapping nor a preparation-dependent force
    replaces that state. All raw coordinates precede trigonometric work.
    """
    gamma = 1 / (1023 * pi_interval())

    def flow(values):
        state = _ordered(values, "state", limit=3)
        if len(state) != 2:
            raise ValueError("state requires aggregate form and relative phase")
        jets = tuple(isinstance(value, Jet) for value in state)
        if any(jets):
            if not all(jets) or any(value.order != state[0].order for value in state):
                raise ValueError("state Taylor jets must all have the same order")
            sine = jet_sin
        else:
            state = tuple(I.coerce(value) for value in state)
            sine = sin
        aggregate, phase = state
        return (
            -2 * aggregate - sine(phase) * (18 * gamma),
            aggregate * (2 * gamma / 9),
        )

    return flow


@dataclass(frozen=True)
class _EffectiveErrorTransfer:
    """Coordinatewise comparison bounds for one admitted fixed segment."""

    initial_form_error: Q
    initial_phase_error: Q
    moment_error: Q
    duration: Q
    feedback_denominator: Q
    decay_upper_bound: Q
    uniform_form_error_upper_bound: Q
    phase_error_upper_bound: Q
    endpoint_form_error_upper_bound: Q


def _effective_error_transfer(
    *, initial_form_error, initial_phase_error, moment_error, duration
):
    """Price damping, phase feedback and the unresolved matching moment.

    All four inputs are normalized before any comparison or arithmetic.
    Durations 8 and 16 use the analytical inequality exp(-2h) < 2**(-2h).
    This helper supplies bounds only, without an exponential or flow solver.
    """
    form = _exact(initial_form_error, "initial_form_error")
    phase = _exact(initial_phase_error, "initial_phase_error")
    moment = _exact(moment_error, "moment_error")
    time = _exact(duration, "duration")
    if min(form, phase, moment) < 0:
        raise ValueError("initial errors and moment allowance must be nonnegative")
    if time not in (_SEGMENT, _DURATION):
        raise ValueError("duration must be one of the declared segments 8 or 16")
    denominator = 1 - 2 * _G**2 * time
    decay = Q(1, 2 ** int(2 * time))
    uniform = (form + 9 * _G * phase + _G * moment / 2) / denominator
    phase_bound = phase + 2 * _G * time * uniform / 9
    endpoint = (
        form * decay + 9 * _G * phase + 2 * _G**2 * time * uniform + _G * moment / 2
    )
    return _EffectiveErrorTransfer(
        initial_form_error=form,
        initial_phase_error=phase,
        moment_error=moment,
        duration=time,
        feedback_denominator=denominator,
        decay_upper_bound=decay,
        uniform_form_error_upper_bound=uniform,
        phase_error_upper_bound=phase_bound,
        endpoint_form_error_upper_bound=endpoint,
    )


@dataclass(frozen=True)
class _MixedEffectiveInteraction:
    """Conditional prediction and continuation for the same two-state law."""

    form_error_bounds: tuple[Q, Q]
    phase_error_bounds: tuple[Q, Q]
    registered_observation_error_bound: Q
    aligned_observation_error_bound: Q
    source: _MixedRelativeRegistration
    source_association_certified: bool
    registration_covariance_certified: bool
    centered_identity_certified: bool
    phase_branch_certified: bool
    phase_moment_remainder_certified: bool
    switching_work_certified: bool
    effective_balance_certified: bool
    direct_prediction_certified: bool
    first_segment_certified: bool
    continuation_certified: bool
    memberwise_semigroup_certified: bool
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
    centered_norm_growth_denominator: Q
    centered_norm_upper_bound: Q
    phase_drift_upper_bound: Q
    marked_gap_upper_bound: Q
    registration_margin: Q
    aggregate_absolute_upper_bound: Q
    relative_phase_deviation_upper_bound: Q
    boundary_branch_margin: Q
    phase_moment_remainder_upper_bound: Q
    phase_switching_work_allowance: Q
    aligned_storage_jump_bounds: tuple[Q, Q]
    registered_storage_jump_bounds: tuple[Q, Q]
    storage_remainder_upper_bound: Q
    unresolved_exchange_integral_upper_bound: Q
    direct_transfer: _EffectiveErrorTransfer
    first_transfer: _EffectiveErrorTransfer
    continued_transfer: _EffectiveErrorTransfer
    direct_storage_prediction_error_upper_bound: Q
    continued_storage_prediction_error_upper_bound: Q
    registered_record_error_upper_bound: Q
    aligned_record_error_upper_bound: Q
    prediction_margin: Q
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
    activation_time: Q = _ACTIVATION
    switch_age: Q = _SWITCH
    switch_time: Q = _ACTIVATION + _SWITCH
    segment_duration: Q = _SEGMENT
    post_duration: Q = _DURATION
    seam_time: Q = _ACTIVATION + _SWITCH + _SEGMENT
    total_duration: Q = _ACTIVATION + _SWITCH + _DURATION
    original_clock_horizon: Q = (_ACTIVATION + _SWITCH + _DURATION) * Q(1024, 1023)
    original_donor_form_total: Q = Q(0)
    whole_form_mean: Q = Q(0)
    registered_observation_error_cap: Q = _NOISE
    aligned_observation_error_cap: Q = _NOISE
    prediction_tolerance: Q = _NOISE
    seam_storage_jump: Q = Q(0)
    effective_form_equation: str = "Qhat_prime=-2*Qhat-18*gamma*sin(psihat)"
    effective_phase_equation: str = "psihat_prime=2*gamma*Qhat/9"
    actual_form_equation: str = "Q_prime=-2*Q-18*gamma*sin(psi)-gamma*R"
    actual_phase_equation: str = "psi_prime=2*gamma*Q/9"
    moment_decomposition: str = "M=18*sin(psi)+R; abs(R)<=Nc**2"
    effective_storage: str = "Qhat**2+162*(1-cos(psihat))"
    effective_storage_rate: str = "-4*Qhat**2"
    projected_storage_rate: str = "-4*Q**2-2*gamma*Q*R"
    unresolved_storage_rate: str = "-norm((H+9G)*x_centered)**2/9+2*gamma*Q*R"
    full_storage_rate: str = "-norm((H+9G)*x)**2/9"
    common_storage_reference: str = "same unshifted Theta in Uplus(Theta)"
    same_law_for_both_preparations: bool = True
    pressure_selected_from_preparation_label: bool = False
    original_complete_field_unchanged: bool = True
    whole_history_covariance_retained: bool = True
    source_correlated_initialization_retained: bool = True
    predicted_state_carried_at_seam: bool = True
    source_member_correlation_carried_at_seam: bool = True
    phase_lift_and_uncertainty_carried_at_seam: bool = True
    independent_marginal_state_reselection: bool = False
    seam_is_bookkeeping_only: bool = True
    state_reset_performed: bool = False
    phase_clamped: bool = False
    microscopic_residual_queried_at_seam: bool = False
    actual_state_queried_at_seam: bool = False
    original_source_reacquired: bool = False
    original_source_replayed: bool = False
    original_source_trajectory_evaluated: bool = False
    reduced_trajectory_evaluated: bool = False
    exact_actual_two_coordinate_closure: bool = False
    exact_memberwise_reduced_semigroup: bool = False
    propagated_bound_equality_claimed: bool = False
    unresolved_storage_assumed_nonnegative: bool = False
    physical_metric_invented_for_error_coordinates: bool = False
    earlier_affine_predictions_invalidated: bool = False
    dynamically_resolved_phase_coordinate_necessary: bool = False
    minimal_state_dimension_claimed: bool = False
    independent_record_errors: bool = True
    observation_error_cancellation_assumed: bool = False
    all_future_identity_claimed: bool = False
    reference_shift_used_to_erase_switch_work: bool = False
    runtime_event_executed: bool = False
    operator_occurrence_derived: bool = False
    exponentials_evaluated: bool = False
    roots_numerically_evaluated: bool = False
    newly_observed_aggregates: int = 0
    required_endpoint_records: int = 2
    numerical_response_allowance: Q = Q(0)
    physical_property_identified: bool = False
    empirical_validation: bool = False
    clock: str = "tau=(1023/1024)*t; unchanged actual and effective interaction rows"


def _assess_mixed_effective_interaction(
    *,
    form_error_bounds,
    phase_error_bounds,
    registered_observation_error_bound,
    aligned_observation_error_bound,
):
    """Re-admit both preparations and carry one finite nonlinear prediction.

    Previous endpoint, affine, obstruction and global verdicts are unused.
    The original source, covariance and consumed inherited error premises are
    rebuilt before extending the centered chart and the two coordinate bounds.
    """
    forms = _nonnegative_pair(form_error_bounds, "form_error_bounds")
    phases = _nonnegative_pair(phase_error_bounds, "phase_error_bounds")
    registered_noise = _nonnegative(
        registered_observation_error_bound, "registered_observation_error_bound"
    )
    aligned_noise = _nonnegative(
        aligned_observation_error_bound, "aligned_observation_error_bound"
    )
    source = _assess_mixed_relative_registration(
        form_error_bounds=forms,
        phase_error_bounds=phases,
        registered_observation_error_bound=Q(0),
        aligned_observation_error_bound=Q(0),
    )
    phase = source.source
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
    source_ok = (
        inherited_source_ok
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
        and source.registration_turns == Q(1, 4)
        and source.registration_angle_bounds == pi / 2
        and source.translated_copy == "recipient"
        and source.preparation
        == "recipient whole precontact phase history translated by +pi/2"
        and source.complete_precontact_history_translated
        and source.source_covariance_is_conditional
        and source.internal_identity_and_winding_unchanged_by_translation
        and source.prior_form_observations_unchanged_by_translation
        and source.aligned_counterpart_retains_same_underlying_source
        and source.original_baseline_retained
        and source.original_lifts_and_error_association_retained
        and phase.initial_relative_phase_mean_assumed_zero is False
        and source.same_complete_contact_law
        and source.existing_field_reused_unchanged
        and source.original_donor_form_total == source.whole_form_mean == 0
        and source.full_state_dimension == 72
        and source.physical_capacities == (Q(1),) * 36
        and source.full_contact_pairs == _PAIRS
        and source.internal_divisor == 9
        and source.aggregate_equation == "Q_prime=-2*Q-gamma*M"
        and source.relative_phase_equation == "psi_prime=2*gamma*Q/9"
        and source.phase_moment_decomposition == "M=18*sin(psi)+R; abs(R)<=Nc**2"
        and source.common_storage_reference == "same unshifted Theta in Uplus(Theta)"
        and source.comparison_storage_rate == "-norm((H+9G)*x)**2/9"
        and source.activation_time == _ACTIVATION
        and source.switch_age == _SWITCH
        and source.switch_time == _ACTIVATION + _SWITCH
        and source.post_duration == _SEGMENT
        and source.total_duration == _ACTIVATION + _SWITCH + _SEGMENT
        and source.original_clock_horizon
        == (_ACTIVATION + _SWITCH + _SEGMENT) * Q(1024, 1023)
        and source.clock
        == "tau=(1023/1024)*t; translated history and all reciprocal contact rows"
        and not any(
            (
                source.source_errors_translated,
                source.original_source_report_mutated,
                source.boundary_phase_clamped,
                source.later_phase_intervention_performed,
                source.state_reset_performed,
                source.original_source_reacquired,
                source.reference_shift_used_to_erase_switch_work,
            )
        )
    )
    covariance = source_ok
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
    aggregate = amplitude[1] + inherited_error + 9 * _G
    phase_deviation = initial_phase_error + 2 * _G * _DURATION * aggregate / 9
    branch_margin = Q(1, 4) - phase_deviation - 2 * norm
    branch = identity and branch_margin > 0
    remainder = norm**2
    moment_ok = branch and remainder > 0

    # Attachment has already occurred at the original common switch. The
    # split at age 8 creates no new field, reference, attachment or work jump.
    phase_work = 972 * _ETA + Q(9, 2) * entry**3
    aligned_jump = (Q(0), 9 * (entry + 18 * _ETA) ** 2)
    registered_jump = (162 - phase_work, 162 + phase_work + 9 * entry**2)
    work_ok = (
        source_ok
        and aligned_jump[1] >= 0
        and 0 < registered_jump[0] <= registered_jump[1]
    )
    storage_remainder = 18 * remainder
    exchange = 2 * _G * _DURATION * aggregate * remainder
    balance = moment_ok and work_ok
    direct = _effective_error_transfer(
        initial_form_error=inherited_error,
        initial_phase_error=initial_phase_error,
        moment_error=remainder,
        duration=_DURATION,
    )
    first = _effective_error_transfer(
        initial_form_error=inherited_error,
        initial_phase_error=initial_phase_error,
        moment_error=remainder,
        duration=_SEGMENT,
    )
    continued = _effective_error_transfer(
        initial_form_error=first.endpoint_form_error_upper_bound,
        initial_phase_error=first.phase_error_upper_bound,
        moment_error=remainder,
        duration=_SEGMENT,
    )
    direct_ok = (
        balance
        and max(
            direct.uniform_form_error_upper_bound,
            direct.endpoint_form_error_upper_bound,
        )
        < _NOISE
    )
    first_ok = (
        balance
        and max(
            first.uniform_form_error_upper_bound, first.endpoint_form_error_upper_bound
        )
        < _NOISE
    )
    continued_ok = (
        first_ok
        and max(
            continued.uniform_form_error_upper_bound,
            continued.endpoint_form_error_upper_bound,
        )
        < _NOISE
    )
    # Global smoothness of this one fixed field gives uniqueness. Continuing
    # each predicted member therefore equals its direct forecast; propagated
    # enclosures need not be equal and do not reselect marginal states.
    semigroup = direct_ok and continued_ok
    direct_storage_error = (
        storage_remainder
        + 2 * aggregate * direct.uniform_form_error_upper_bound
        + 162 * direct.phase_error_upper_bound
    )
    continued_storage_error = (
        storage_remainder
        + 2 * aggregate * continued.uniform_form_error_upper_bound
        + 162 * continued.phase_error_upper_bound
    )
    admitted = max(registered_noise, aligned_noise) <= _NOISE
    registered_record_error = (
        continued.endpoint_form_error_upper_bound + registered_noise
    )
    aligned_record_error = continued.endpoint_form_error_upper_bound + aligned_noise
    registered_record_ok = semigroup and admitted
    aligned_record_ok = semigroup and admitted
    prediction_margin = _NOISE - max(
        direct.uniform_form_error_upper_bound,
        direct.endpoint_form_error_upper_bound,
        first.uniform_form_error_upper_bound,
        first.endpoint_form_error_upper_bound,
        continued.uniform_form_error_upper_bound,
        continued.endpoint_form_error_upper_bound,
    )
    certified = balance and semigroup and registered_record_ok and aligned_record_ok
    unresolved = tuple(
        label
        for good, label in (
            (source_ok, "original_source_or_inherited_bounds_unavailable"),
            (covariance, "whole_history_registration_covariance_unavailable"),
            (identity, "fresh_centered_identity_unavailable"),
            (branch, "relative_phase_branch_unavailable"),
            (balance, "complete_and_effective_balance_unavailable"),
            (direct_ok, "direct_prediction_tolerance_not_certified"),
            (first_ok, "first_segment_tolerance_not_certified"),
            (continued_ok, "carried_continuation_tolerance_not_certified"),
            (admitted, "independent_observation_errors_outside_protocol"),
        )
        if not good
    )
    return _MixedEffectiveInteraction(
        form_error_bounds=forms,
        phase_error_bounds=phases,
        registered_observation_error_bound=registered_noise,
        aligned_observation_error_bound=aligned_noise,
        source=source,
        source_association_certified=source_ok,
        registration_covariance_certified=covariance,
        centered_identity_certified=identity,
        phase_branch_certified=branch,
        phase_moment_remainder_certified=moment_ok,
        switching_work_certified=work_ok,
        effective_balance_certified=balance,
        direct_prediction_certified=direct_ok,
        first_segment_certified=first_ok,
        continuation_certified=continued_ok,
        memberwise_semigroup_certified=semigroup,
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
        centered_norm_growth_denominator=denominator,
        centered_norm_upper_bound=norm,
        phase_drift_upper_bound=phase_drift,
        marked_gap_upper_bound=marked,
        registration_margin=registration_margin,
        aggregate_absolute_upper_bound=aggregate,
        relative_phase_deviation_upper_bound=phase_deviation,
        boundary_branch_margin=branch_margin,
        phase_moment_remainder_upper_bound=remainder,
        phase_switching_work_allowance=phase_work,
        aligned_storage_jump_bounds=aligned_jump,
        registered_storage_jump_bounds=registered_jump,
        storage_remainder_upper_bound=storage_remainder,
        unresolved_exchange_integral_upper_bound=exchange,
        direct_transfer=direct,
        first_transfer=first,
        continued_transfer=continued,
        direct_storage_prediction_error_upper_bound=direct_storage_error,
        continued_storage_prediction_error_upper_bound=continued_storage_error,
        registered_record_error_upper_bound=registered_record_error,
        aligned_record_error_upper_bound=aligned_record_error,
        prediction_margin=prediction_margin,
        certified=certified,
        status=(
            "conditional_effective_interaction_continuation"
            if certified
            else "unavailable"
        ),
        unresolved_conditions=unresolved,
        exact_memberwise_reduced_semigroup=semigroup,
    )
