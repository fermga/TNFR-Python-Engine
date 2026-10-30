"""A finite recipient phase reading in the already admitted interaction domain.

The existing full-slot form port is compared with its cut alternative. Both
hypotheses keep the original actual donor/recipient source family. A nonlinear
finite envelope and a uniform fine-state error produce independent record
sets; no trajectory, measured reading or old target preparation is supplied.
"""

from dataclasses import dataclass
from fractions import Fraction as Q

from ._mixed_reciprocal_partner import _nonnegative, _nonnegative_pair
from ._mixed_registered_phase_probe import _derive_registered_phase_probe_geometry
from ._mixed_resource_transfer_obstruction import (
    _assess_mixed_resource_transfer_obstruction,
    _MixedResourceTransferObstruction,
    _original_source_matches,
)
from ._sine_coupling_successor import _CouplingSuccessorGeometry

_TIME = Q(2000000)
_NOISE_CAP = Q(1, 10**6)
_THRESHOLD = Q(1, 800000)
_SIGNAL_FLOOR = Q(13, 5000000)
_SOURCE_CAP = Q(1, 10**10)
_DELTA = Q(1, 1024)
_ETA = Q(20, 10**70)
_SECTION = 3 * _DELTA / 16
_MATURE = Q(1, 2**28)
_WAIT = Q(10**10)
_WEIGHT_DEVIATION = Q(2, 10**60)
_SECTION_ERROR = Q(3, 2) * (14 * _ETA + 2 * _WAIT * _WEIGHT_DEVIATION)
_GAMMA_LOWER = Q(1, 3300)
_GAMMA_UPPER = Q(1, 3000)
_STATE_ALLOWANCE = 8 * _MATURE
_RADIUS = Q(1, 32)


@dataclass(frozen=True)
class _MixedRecipientPhaseResponse:
    """Conditional endpoint record sets, not an observed or executed response.

    Source synchronization, the original marked branches and association of
    a possible aggregate reading with its actual recipient are premises.
    The noise allowance concerns that one aggregate, not each node. Either
    hypothesis may independently select any admitted source and readout
    error. Candidate bounds require the accompanying certification flags.
    Earlier target, resource and global verdicts are not consumed.
    """

    form_error_bounds: tuple[Q, Q]
    phase_error_bounds: tuple[Q, Q]
    observation_error_bound: Q
    reference: _MixedResourceTransferObstruction
    source_geometry: _CouplingSuccessorGeometry
    observation_covector: tuple[Q, ...]
    observation_reference_turns: Q
    observation_geometry_certified: bool
    source_association_certified: bool
    actual_trap_certified: bool
    comparison_certified: bool
    identity_certified: bool
    nominal_q0_bounds: tuple[Q, Q]
    wedge_fraction_lower_bound: Q
    sine_factor_lower_bound: Q
    sine_factor_margin: Q
    restoring_force_fraction_margin: Q
    stiffness_lower_bound: Q
    duhamel_time_coefficient: Q
    inherited_velocity_penalty: Q
    nominal_response_bracket: Q
    nominal_response_lower_bound: Q
    nominal_signal_margin: Q
    finite_envelope_certified: bool
    actual_observation_allowance: Q
    connected_true_bounds: tuple[Q, Q]
    cut_true_bounds: tuple[Q, Q]
    observation_allowance_admitted: bool
    connected_record_bounds: tuple[Q, Q]
    cut_record_bounds: tuple[Q, Q]
    record_separation_margin: Q
    connected_threshold_margin: Q
    cut_threshold_margin: Q
    independent_record_sets_separated: bool
    fixed_threshold_certified: bool
    certified: bool
    status: str
    unresolved_conditions: tuple[str, ...]
    source_error_cap: Q = _SOURCE_CAP
    observation_error_cap: Q = _NOISE_CAP
    contact_duration: Q = _TIME
    original_clock_duration: Q = _TIME * Q(1024, 1023)
    threshold: Q = _THRESHOLD
    nominal_signal_floor: Q = _SIGNAL_FLOOR
    section_mean_gap: Q = _SECTION
    donor_section_comparison_error: Q = _SECTION_ERROR
    recipient_mature_radius: Q = _MATURE
    recipient_isolated_deadline: Q = _WAIT
    gamma_bounds: tuple[Q, Q] = (_GAMMA_LOWER, _GAMMA_UPPER)
    phase_radius: Q = _RADIUS
    registration_gap_limit: Q = _DELTA / 4
    observation_covector_norm_squared: Q = Q(2, 9)
    numerical_response_allowance: Q = Q(0)
    independently_observed_aggregates: int = 1
    full_state_dimension: int = 72
    physical_capacities: tuple[Q, ...] = (Q(1),) * 36
    contact_pairs: tuple[tuple[int, int], ...] = tuple((j, 18 + j) for j in range(18))
    independently_selected_hypothesis_sources: bool = True
    inherited_donor_velocity_retained: bool = True
    source_synchronization_authenticated: bool = False
    observation_is_endpoint_not_integral: bool = True
    observed_record_supplied: bool = False
    sensor_model_derived: bool = False
    old_opposite_gap_target_claimed_reached: bool = False
    donor_dwell_applied: bool = False
    charge_or_leak_prefix_applied: bool = False
    phase_jump_performed: bool = False
    source_trajectory_evaluated: bool = False
    runtime_event_executed: bool = False
    output_orientation: str = "negative_recipient_mean_marked_registered_phase_gap"
    clock: str = "tau=(1023/1024)*t; complete full and cut internal laws"
    scope: tuple[str, ...] = (
        "same_actual_first_section_donor_and_independently_mature_recipient_family",
        "one_recipient_registered_phase_endpoint_observation_on_original_branches",
        "recipient_common_phase_gauge_cancels_without_an_inter_copy_gauge",
        "full_matching_slot_form_port_vs_cut_with_both_internal_rows_retained",
        "nonlinear_finite_envelope_retains_inherited_phase_velocity_correlation",
        "uniform_fine_state_error_does_not_accumulate_as_an_integrated_record",
        "independent_preparation_and_aggregate_readout_sets_are_separated",
        "earlier_opposite_gap_target_obstruction_remains_intact",
        "no_observed_record_sensor_authentication_event_or_physical_claim",
    )


def _assess_mixed_recipient_phase_response(
    *, form_error_bounds, phase_error_bounds, observation_error_bound
):
    """Rebuild the admitted pair and its finite registered-phase envelopes."""
    forms = _nonnegative_pair(form_error_bounds, "form_error_bounds")
    phases = _nonnegative_pair(phase_error_bounds, "phase_error_bounds")
    noise = _nonnegative(observation_error_bound, "observation_error_bound")
    reference = _assess_mixed_resource_transfer_obstruction(
        form_error_bounds=forms, phase_error_bounds=phases
    )
    registered = _derive_registered_phase_probe_geometry()
    geometry = registered.successor
    covector = [Q(0)] * 72
    for a, b in geometry.marked_pairs:
        covector[54 + a] += Q(1, 9)
        covector[54 + b] -= Q(1, 9)
    covector = tuple(covector)
    reference_turns = sum(geometry.marked_branch_turns, Q(0)) / 9
    geometry_ok = (
        registered.cross_symmetry_certified
        and reference.source_geometry == geometry
        and reference.donor.cross_incidence == registered.cross_incidence
        and geometry.completion == "common_sense"
        and geometry.nodes == tuple(range(18))
        and geometry.degrees == (9,) * 18
        and geometry.components == (tuple(range(18)),)
        and geometry.held_capacities == (Q(1),) * 18
        and geometry.support_certified
        and geometry.reference_stationary_certified
        and geometry.spectral_gap_certified
        and len(geometry.edge_pairs) == 81
        and len(geometry.marked_pairs) == len(geometry.marked_branch_turns) == 9
        and tuple(sorted(a for a, _ in geometry.marked_pairs)) == tuple(range(9))
        and tuple(sorted(b for _, b in geometry.marked_pairs)) == tuple(range(9, 18))
        and all(
            turn.denominator == 1
            and geometry.target_phase_turns[b] - geometry.target_phase_turns[a] == turn
            for (a, b), turn in zip(
                geometry.marked_pairs, geometry.marked_branch_turns, strict=True
            )
        )
        and reference_turns == 0
        and covector == (Q(0),) * 54 + (Q(1, 9),) * 9 + (-Q(1, 9),) * 9
        and sum(value**2 for value in covector) == Q(2, 9)
    )
    source_ok = (
        geometry_ok
        and reference.form_error_bounds == forms
        and reference.phase_error_bounds == phases
        and reference.donor_source_certified
        and reference.recipient_source_certified
        and reference.source_association_certified
        and reference.sources
        == (reference.donor.source, reference.recipient_readiness.sources[0])
        and all(
            _original_source_matches(source, geometry, rx, rt)
            for source, rx, rt in zip(reference.sources, forms, phases, strict=True)
        )
        and reference.original_relative_phase_gap == _DELTA
        and reference.original_supplied_phase_origins == (-_DELTA / 2, _DELTA / 2)
        and reference.section_mean_gap == _SECTION
        and reference.donor.section_mean_gap == _SECTION
        and reference.donor_section_comparison_error == _SECTION_ERROR
        and reference.donor.comparison_error_upper_bound == _SECTION_ERROR
        and reference.recipient_mature_radius == _MATURE
        and reference.recipient_isolated_deadline == _WAIT
        and reference.actual_weight_deviation_upper_bound == _WEIGHT_DEVIATION
        and reference.initial_form_means == (Q(0), Q(0))
        and reference.physical_capacities == (Q(1),) * 36
        and reference.full_state_dimension == 72
        and reference.contact_pairs == tuple((j, 18 + j) for j in range(18))
        and reference.clock
        == "tau=(1023/1024)*t; complete internal rows and reciprocal form inputs"
        and not reference.donor_dwell_applied
        and not reference.charge_or_leak_prefix_applied
        and not reference.phase_jump_performed
        and reference.inherited_donor_velocity_retained
        and reference.inherited_nominal_velocity_relation
        == "-gamma**2*q0 <= w0 <= 0 at the same isolated age"
    )
    trap = (
        source_ok
        and reference.actual_trap_certified
        and reference.full_norm_upper_bound == Q(1, 100) < _RADIUS
        and 0
        <= reference.full_norm_squared_upper_bound
        < reference.full_norm_upper_bound**2
        and reference.trap_norm_margin > 0
        and reference.acute_chart_margin > 0
        and reference.dissipation_margin == 1 - 14 * _WEIGHT_DEVIATION > 0
    )
    comparison = (
        trap
        and reference.metric_certified
        and reference.comparison_certified
        and reference.base_metric_decay_rate > 0
        and reference.conservative_metric_decay_rate
        == reference.base_metric_decay_rate / 8
        and reference.metric_decay_rate >= reference.conservative_metric_decay_rate
        and reference.metric_growth_squared_margin > 0
        and reference.initial_comparison_norm_upper_bound == _MATURE + _SECTION_ERROR
        and reference.comparison_matrix_deviation_upper_bound == 14 * _WEIGHT_DEVIATION
        and reference.comparison_forcing_upper_bound == 2 * _WEIGHT_DEVIATION
        and reference.comparison_forcing_margin > 0
        and reference.uniform_comparison_bound
        == 4
        * (
            _MATURE
            + _SECTION_ERROR
            + 4 * _WEIGHT_DEVIATION / reference.conservative_metric_decay_rate
        )
        < _STATE_ALLOWANCE
        and reference.uniform_comparison_upper_bound == _STATE_ALLOWANCE
        and reference.uniform_comparison_margin > 0
    )
    identity = (
        comparison
        and reference.all_time_identity_certified
        and reference.all_contact_durations_covered
        and reference.phase_radius == _RADIUS
        and reference.registration_gap_limit == _DELTA / 4
        and reference.marked_gap_upper_bound
        == _SECTION + _SECTION_ERROR / 2 + 12 * _MATURE
        and reference.registration_margin
        == _DELTA / 4 - reference.marked_gap_upper_bound
        > 0
    )
    q0 = (_SECTION - _SECTION_ERROR / 2, _SECTION + _SECTION_ERROR / 2)
    wedge_fraction = 1 - _GAMMA_UPPER**2 * _TIME
    sine_factor = 1 - q0[1] ** 2 / 6
    sine_margin = sine_factor - Q(27, 28)
    force_margin = wedge_fraction * sine_factor - Q(3, 4)
    stiffness = Q(32, 81) * _GAMMA_LOWER**2
    time_coefficient = Q(21, 52) * _TIME - Q(243, 256)
    velocity_penalty = Q(9, 26) * _GAMMA_UPPER**2
    bracket = stiffness * time_coefficient - velocity_penalty
    lower = q0[0] * bracket / 2
    signal_margin = lower - _SIGNAL_FLOOR
    donor = reference.donor
    finite = (
        identity
        and reference.nominal_wedges_certified
        and reference.nominal_q0_bounds == q0
        and reference.nominal_velocity_factor_upper_bound == _GAMMA_UPPER**2
        and reference.nominal_wedge_margin
        == 2 * donor.nominal_coupling_bounds.lo / 3 - _GAMMA_UPPER**2
        > 0
        and Q(4, 9)
        < donor.nominal_coupling_bounds.lo
        <= donor.nominal_coupling_bounds.hi
        < Q(1, 2)
        and 2
        < donor.regional_sine_coefficient_bounds.lo
        <= donor.regional_sine_coefficient_bounds.hi
        < 3
        and _GAMMA_LOWER
        <= donor.gamma_bounds.lo
        <= donor.gamma_bounds.hi
        <= _GAMMA_UPPER
        and 0 < q0[0] <= q0[1] < _DELTA / 4
        and wedge_fraction == Q(7, 9)
        and sine_margin > 0
        and force_margin > 0
        and time_coefficient > 0
        and bracket > 0
        and signal_margin > 0
    )
    true_full = (lower - _STATE_ALLOWANCE, q0[1] / 2 + _STATE_ALLOWANCE)
    true_cut = (-_STATE_ALLOWANCE, _STATE_ALLOWANCE)
    admitted = noise <= _NOISE_CAP
    full_record = (true_full[0] - noise, true_full[1] + noise)
    cut_record = (true_cut[0] - noise, true_cut[1] + noise)
    separation = full_record[0] - cut_record[1]
    full_threshold = full_record[0] - _THRESHOLD
    cut_threshold = _THRESHOLD - cut_record[1]
    separated = finite and admitted and true_full[0] <= true_full[1] and separation > 0
    threshold = separated and full_threshold > 0 and cut_threshold > 0
    unresolved = tuple(
        label
        for good, label in (
            (geometry_ok, "original_marked_branch_observation_geometry_unavailable"),
            (source_ok, "actual_donor_recipient_source_association_unavailable"),
            (trap, "both_actual_all_time_traps_unavailable"),
            (comparison, "uniform_fine_state_comparison_unavailable"),
            (identity, "retained_identity_and_continuous_phase_branches_unavailable"),
            (finite, "finite_nonlinear_recipient_phase_envelope_unavailable"),
            (admitted, "aggregate_observation_error_exceeds_protocol"),
            (separated, "independent_record_sets_not_certified_separate"),
            (threshold, "fixed_decision_threshold_not_certified_between_records"),
        )
        if not good
    )
    return _MixedRecipientPhaseResponse(
        form_error_bounds=forms,
        phase_error_bounds=phases,
        observation_error_bound=noise,
        reference=reference,
        source_geometry=geometry,
        observation_covector=covector,
        observation_reference_turns=reference_turns,
        observation_geometry_certified=geometry_ok,
        source_association_certified=source_ok,
        actual_trap_certified=trap,
        comparison_certified=comparison,
        identity_certified=identity,
        nominal_q0_bounds=q0,
        wedge_fraction_lower_bound=wedge_fraction,
        sine_factor_lower_bound=sine_factor,
        sine_factor_margin=sine_margin,
        restoring_force_fraction_margin=force_margin,
        stiffness_lower_bound=stiffness,
        duhamel_time_coefficient=time_coefficient,
        inherited_velocity_penalty=velocity_penalty,
        nominal_response_bracket=bracket,
        nominal_response_lower_bound=lower,
        nominal_signal_margin=signal_margin,
        finite_envelope_certified=finite,
        actual_observation_allowance=_STATE_ALLOWANCE,
        connected_true_bounds=true_full,
        cut_true_bounds=true_cut,
        observation_allowance_admitted=admitted,
        connected_record_bounds=full_record,
        cut_record_bounds=cut_record,
        record_separation_margin=separation,
        connected_threshold_margin=full_threshold,
        cut_threshold_margin=cut_threshold,
        independent_record_sets_separated=separated,
        fixed_threshold_certified=threshold,
        certified=not unresolved,
        status=(
            "conditional_recipient_phase_response" if not unresolved else "unavailable"
        ),
        unresolved_conditions=unresolved,
    )
