"""Carry the reached nominal family and its error through one declared cut.

The complete actual state is continuous at the cut. The regional reference
remains an associated family, not four measured numbers or an exact actual
closure. Existing source and metric owners price the discarded microscopic
information; no new field, endpoint reading or trajectory is constructed.
"""

from dataclasses import dataclass
from fractions import Fraction as Q

from ._mixed_recipient_phase_response import (
    _assess_mixed_recipient_phase_response,
    _MixedRecipientPhaseResponse,
)
from ._mixed_reciprocal_partner import _nonnegative, _nonnegative_pair
from ._mixed_registered_phase_probe import _derive_registered_phase_probe_geometry
from ._mixed_resource_transfer_obstruction import _original_source_matches

_PREFIX = Q(2000000)
_CONTINUATION = Q(10000)
_TOTAL = _PREFIX + _CONTINUATION
_NOISE_CAP = Q(1, 10**6)
_THRESHOLD = Q(1, 800000)
_DELTA = Q(1, 1024)
_ETA = Q(20, 10**70)
_WEIGHT = Q(2, 10**60)
_SECTION = 3 * _DELTA / 16
_SECTION_ERROR = Q(3, 2) * (14 * _ETA + 2 * 10**10 * _WEIGHT)
_MATURE = Q(1, 2**28)
_ALLOWANCE = 8 * _MATURE
_GAMMA_LOWER = Q(1, 3300)
_GAMMA_UPPER = Q(1, 3000)
_SCALE = Q(6, 5)
_D = (-Q(1, 9),) * 9 + (Q(1, 9),) * 9


@dataclass(frozen=True)
class _MixedCarriedFormPhase:
    """Conditional late record sets after full contact followed by removal.

    The prefix report supplies freshly rebuilt source and comparison premises.
    Its previous record-set and global verdicts are not inputs to this result.
    Source synchronization and observation association remain supplied; no
    handoff reading, actual state reset or independently calibrated sensor is
    granted. The control has never had external contact.
    """

    form_error_bounds: tuple[Q, Q]
    phase_error_bounds: tuple[Q, Q]
    observation_error_bound: Q
    prefix: _MixedRecipientPhaseResponse
    source_association_certified: bool
    projected_rows_certified: bool
    handoff_certified: bool
    switched_metric_certified: bool
    continuation_wedges_certified: bool
    identity_certified: bool
    nominal_q0_bounds: tuple[Q, Q]
    nominal_prefix_response_lower_bound: Q
    continuation_drift_upper_bound: Q
    nominal_late_response_lower_bound: Q
    carried_error_upper_bound: Q
    metric_decay_rate: Q
    metric_forcing_upper_bound: Q
    carried_metric_upper_bound: Q | None
    metric_lower_bound: Q
    metric_upper_bound: Q
    actual_observation_allowance: Q
    finite_envelope_certified: bool
    switched_true_bounds: tuple[Q, Q]
    never_contact_true_bounds: tuple[Q, Q]
    observation_allowance_admitted: bool
    switched_record_bounds: tuple[Q, Q]
    never_contact_record_bounds: tuple[Q, Q]
    record_separation_margin: Q
    switched_threshold_margin: Q
    never_contact_threshold_margin: Q
    independent_record_sets_separated: bool
    fixed_threshold_certified: bool
    certified: bool
    status: str
    unresolved_conditions: tuple[str, ...]
    prefix_duration: Q = _PREFIX
    continuation_duration: Q = _CONTINUATION
    total_duration: Q = _TOTAL
    original_clock_horizon: Q = _TOTAL * Q(1024, 1023)
    threshold: Q = _THRESHOLD
    observation_error_cap: Q = _NOISE_CAP
    numerical_response_allowance: Q = Q(0)
    handoff_observations: int = 0
    independently_observed_aggregates: int = 1
    reference_coordinates: tuple[str, ...] = ("q_D", "q_R", "v_D", "v_R")
    projected_form_covector: tuple[Q, ...] = tuple(value / 2 for value in _D)
    projected_phase_covector: tuple[Q, ...] = _D
    retained_form_means: tuple[Q, Q] = (Q(0), Q(0))
    physical_capacities: tuple[Q, ...] = (Q(1),) * 36
    full_state_dimension: int = 72
    reference_family_carried: bool = True
    reached_velocity_retained: bool = True
    source_weight_association_retained: bool = True
    original_phase_gauges_and_branches_retained: bool = True
    exact_actual_four_coordinate_closure: bool = False
    minimal_collective_state_established: bool = False
    state_reset_performed: bool = False
    source_trajectory_evaluated: bool = False
    runtime_event_executed: bool = False
    observed_record_supplied: bool = False
    empirical_validation: bool = False
    switch_word: tuple[int, int] = (1, 0)
    control_word: tuple[int, int] = (0, 0)
    clock: str = "tau=(1023/1024)*t; all complete internal rows retained"
    scope: tuple[str, ...] = (
        "same_actual_first_section_donor_and_independently_mature_recipient",
        "full_matching_form_contact_then_one_prescribed_zero_port_continuation",
        "same_reached_reference_family_and_actual_state_at_the_cut",
        "actual_projected_rows_keep_sine_and_both_weight_remainders",
        "metric_decreases_at_switch_and_carries_one_accumulated_error_budget",
        "late_phase_endpoint_vs_never_contact_with_independent_readout_errors",
        "no_fresh_handoff_reading_or_reference_reset",
        "no_exact_minimal_state_second_positive_port_or_outer_identity_claim",
    )


def _assess_mixed_carried_form_phase(
    *, form_error_bounds, phase_error_bounds, observation_error_bound
):
    """Rebuild the original premises and assess the fixed two-piece protocol."""
    forms = _nonnegative_pair(form_error_bounds, "form_error_bounds")
    phases = _nonnegative_pair(phase_error_bounds, "phase_error_bounds")
    noise = _nonnegative(observation_error_bound, "observation_error_bound")
    prefix = _assess_mixed_recipient_phase_response(
        form_error_bounds=forms,
        phase_error_bounds=phases,
        observation_error_bound=noise,
    )
    reference = prefix.reference
    registered = _derive_registered_phase_probe_geometry()
    geometry = registered.successor
    q0 = (_SECTION - _SECTION_ERROR / 2, _SECTION + _SECTION_ERROR / 2)
    source = (
        prefix.form_error_bounds == reference.form_error_bounds == forms
        and prefix.phase_error_bounds == reference.phase_error_bounds == phases
        and prefix.source_association_certified
        and reference.source_association_certified
        and reference.donor_source_certified
        and reference.recipient_source_certified
        and prefix.observation_geometry_certified
        and registered.cross_symmetry_certified
        and geometry == prefix.source_geometry == reference.source_geometry
        and reference.donor.cross_incidence == registered.cross_incidence
        and reference.sources
        == (reference.donor.source, reference.recipient_readiness.sources[0])
        and all(
            _original_source_matches(source, geometry, rx, rt)
            for source, rx, rt in zip(reference.sources, forms, phases, strict=True)
        )
        and reference.original_relative_phase_gap == _DELTA
        and reference.original_supplied_phase_origins == (-_DELTA / 2, _DELTA / 2)
        and prefix.contact_duration == _PREFIX
        and prefix.original_clock_duration == _PREFIX * Q(1024, 1023)
        and prefix.section_mean_gap == reference.section_mean_gap == _SECTION
        and reference.donor.section_mean_gap == _SECTION
        and prefix.donor_section_comparison_error == _SECTION_ERROR
        and reference.donor_section_comparison_error == _SECTION_ERROR
        and reference.donor.comparison_error_upper_bound == _SECTION_ERROR
        and reference.actual_weight_deviation_upper_bound == _WEIGHT
        and prefix.recipient_mature_radius
        == reference.recipient_mature_radius
        == _MATURE
        and reference.recipient_isolated_deadline == Q(10**10)
        and reference.recipient_readiness.waiting_duration == Q(10**10)
        and reference.recipient_readiness.derived_mature_norm_upper_bound == _MATURE
        and reference.initial_form_means == (Q(0), Q(0))
        and reference.physical_capacities == (Q(1),) * 36
        and reference.contact_pairs == tuple((j, 18 + j) for j in range(18))
        and reference.full_state_dimension == 72
        and reference.clock
        == "tau=(1023/1024)*t; complete internal rows and reciprocal form inputs"
        and not reference.donor_dwell_applied
        and not reference.charge_or_leak_prefix_applied
        and not reference.phase_jump_performed
        and prefix.inherited_donor_velocity_retained
        and reference.inherited_donor_velocity_retained
    )
    projection = (
        source
        and geometry.nodes == tuple(range(18))
        and geometry.degrees == (9,) * 18
        and geometry.held_capacities == (Q(1),) * 18
        and geometry.reference_stationary_certified
        and geometry.support_certified
        and reference.donor.nominal_reference_certified
        and reference.nominal_q0_bounds == prefix.nominal_q0_bounds == q0
        and prefix.observation_covector
        == (Q(0),) * 54 + (Q(1, 9),) * 9 + (-Q(1, 9),) * 9
        and prefix.observation_reference_turns == 0
    )
    handoff = (
        projection
        and reference.inherited_nominal_velocity_relation
        == "-gamma**2*q0 <= w0 <= 0 at the same isolated age"
        and reference.initial_comparison_norm_upper_bound == _MATURE + _SECTION_ERROR
        and prefix.identity_certified
        and not prefix.donor_dwell_applied
        and not prefix.charge_or_leak_prefix_applied
        and not prefix.phase_jump_performed
        and not prefix.source_trajectory_evaluated
    )
    rate = reference.conservative_metric_decay_rate
    forcing = reference.comparison_forcing_upper_bound
    metric_envelope = (
        reference.metric_upper_bound
        * _SCALE**2
        * (_MATURE + _SECTION_ERROR + 2 * forcing / rate) ** 2
        if rate > 0
        else None
    )
    # W_0 <= W_+ at the cut, on the same reached error. The square-root
    # inequality therefore continues without a new initial error or factor.
    metric = (
        handoff
        and prefix.actual_trap_certified
        and reference.actual_trap_certified
        and prefix.comparison_certified
        and reference.metric_certified
        and reference.comparison_certified
        and reference.phase_scale == _SCALE
        and reference.metric_lower_bound
        == reference.base_metric_lower_bound
        == Q(9, 4) * (1 - Q(1, 2500))
        and reference.base_metric_upper_bound == Q(9, 4) * (1 + Q(1, 2500))
        and reference.metric_upper_bound == reference.base_metric_upper_bound + Q(27, 2)
        and reference.full_norm_upper_bound == Q(1, 100) < Q(1, 32)
        and reference.dissipation_margin == 1 - 14 * _WEIGHT > 0
        and rate == reference.base_metric_decay_rate / 8 > 0
        and reference.metric_decay_rate >= rate
        and reference.metric_growth_squared_margin > 0
        and reference.comparison_matrix_deviation_upper_bound == 14 * _WEIGHT
        and forcing == 2 * _WEIGHT
        and reference.comparison_forcing_margin > 0
        and reference.uniform_comparison_bound
        == 4 * (_MATURE + _SECTION_ERROR + 2 * forcing / rate)
        < _ALLOWANCE
        and reference.uniform_comparison_upper_bound == _ALLOWANCE
        and prefix.actual_observation_allowance == _ALLOWANCE
    )
    wedges = (
        handoff
        and reference.nominal_wedges_certified
        and prefix.finite_envelope_certified
        and reference.nominal_velocity_factor_upper_bound == _GAMMA_UPPER**2
        and prefix.gamma_bounds == (_GAMMA_LOWER, _GAMMA_UPPER)
        and _GAMMA_LOWER
        <= reference.donor.gamma_bounds.lo
        <= reference.donor.gamma_bounds.hi
        <= _GAMMA_UPPER
        and Q(4, 9)
        < reference.donor.nominal_coupling_bounds.lo
        <= reference.donor.nominal_coupling_bounds.hi
        < Q(1, 2)
        and 2
        < reference.donor.regional_sine_coefficient_bounds.lo
        <= reference.donor.regional_sine_coefficient_bounds.hi
        < 3
        and reference.nominal_wedge_margin
        == 2 * reference.donor.nominal_coupling_bounds.lo / 3 - _GAMMA_UPPER**2
        > 0
        and 0 < q0[0] <= q0[1] < _DELTA / 4
    )
    identity = (
        metric
        and wedges
        and reference.all_time_identity_certified
        and reference.all_contact_durations_covered
        and reference.phase_radius == Q(1, 32)
        and reference.registration_gap_limit == _DELTA / 4
        and reference.marked_gap_upper_bound == q0[1] + 12 * _MATURE
        and reference.registration_margin == _DELTA / 4 - q0[1] - 12 * _MATURE > 0
    )
    stiffness = Q(32, 81) * _GAMMA_LOWER**2
    prefix_lower = (
        q0[0]
        * (stiffness * (Q(21, 52) * _PREFIX - Q(243, 256)) - Q(9, 26) * _GAMMA_UPPER**2)
        / 2
    )
    drift = _CONTINUATION * _GAMMA_UPPER**2 * q0[1] / 2
    late_lower = prefix_lower - drift
    finite = (
        identity
        and prefix.nominal_response_lower_bound == prefix_lower > Q(13, 5000000)
        and prefix.stiffness_lower_bound == stiffness
        and prefix.inherited_velocity_penalty == Q(9, 26) * _GAMMA_UPPER**2
        and drift > 0
        and late_lower > 0
    )
    true_switched = (late_lower - _ALLOWANCE, q0[1] / 2 + _ALLOWANCE)
    true_control = (-_ALLOWANCE, _ALLOWANCE)
    admitted = noise <= _NOISE_CAP
    switched = (true_switched[0] - noise, true_switched[1] + noise)
    control = (true_control[0] - noise, true_control[1] + noise)
    separation = switched[0] - control[1]
    switched_threshold = switched[0] - _THRESHOLD
    control_threshold = _THRESHOLD - control[1]
    separated = finite and admitted and switched[0] <= switched[1] and separation > 0
    threshold = separated and switched_threshold > 0 and control_threshold > 0
    unresolved = tuple(
        label
        for good, label in (
            (source, "original_source_and_interface_association_unavailable"),
            (projection, "regional_projection_premises_unavailable"),
            (handoff, "reached_reference_family_and_handoff_association_unavailable"),
            (metric, "continuous_state_metric_drop_and_carried_error_unavailable"),
            (wedges, "post_cut_nonlinear_velocity_wedges_unavailable"),
            (identity, "switched_identity_and_original_branches_unavailable"),
            (finite, "late_nonlinear_phase_envelope_unavailable"),
            (admitted, "aggregate_observation_error_exceeds_protocol"),
            (separated, "independent_late_record_sets_not_certified_separate"),
            (threshold, "fixed_threshold_not_certified_between_late_records"),
        )
        if not good
    )
    return _MixedCarriedFormPhase(
        form_error_bounds=forms,
        phase_error_bounds=phases,
        observation_error_bound=noise,
        prefix=prefix,
        source_association_certified=source,
        projected_rows_certified=projection,
        handoff_certified=handoff,
        switched_metric_certified=metric,
        continuation_wedges_certified=wedges,
        identity_certified=identity,
        nominal_q0_bounds=q0,
        nominal_prefix_response_lower_bound=prefix_lower,
        continuation_drift_upper_bound=drift,
        nominal_late_response_lower_bound=late_lower,
        carried_error_upper_bound=reference.uniform_comparison_bound,
        metric_decay_rate=rate,
        metric_forcing_upper_bound=forcing,
        carried_metric_upper_bound=metric_envelope,
        metric_lower_bound=reference.metric_lower_bound,
        metric_upper_bound=reference.metric_upper_bound,
        actual_observation_allowance=_ALLOWANCE,
        finite_envelope_certified=finite,
        switched_true_bounds=true_switched,
        never_contact_true_bounds=true_control,
        observation_allowance_admitted=admitted,
        switched_record_bounds=switched,
        never_contact_record_bounds=control,
        record_separation_margin=separation,
        switched_threshold_margin=switched_threshold,
        never_contact_threshold_margin=control_threshold,
        independent_record_sets_separated=separated,
        fixed_threshold_certified=threshold,
        certified=not unresolved,
        status="conditional_carried_form_phase" if not unresolved else "unavailable",
        unresolved_conditions=unresolved,
    )
