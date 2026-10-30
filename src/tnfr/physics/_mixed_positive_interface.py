"""A nonzero regional form port after the already paid contact-and-cut history.

The new port admits a live relative common-form mode. Its short response uses
component error bounds and the actual reached history, without extending the
old commuting metric to a different leaf or changing an existing field.
"""

from dataclasses import dataclass
from fractions import Fraction as Q

from ._mixed_carried_form_phase import (
    _assess_mixed_carried_form_phase,
    _MixedCarriedFormPhase,
)
from ._mixed_reciprocal_partner import _nonnegative, _nonnegative_pair
from ._mixed_registered_phase_probe import _derive_registered_phase_probe_geometry
from ._mixed_resource_transfer_obstruction import _original_source_matches

_FULL = Q(2000000)
_CUT = Q(10000)
_HANDOFF = _FULL + _CUT
_DURATION = Q(1, 2)
_NOISE = Q(1, 10**9)
_THRESHOLD = Q(1, 10**8)
_DELTA = Q(1, 1024)
_ETA = Q(2, 10**69)
_WEIGHT = Q(2, 10**60)
_SECTION = 3 * _DELTA / 16
_SECTION_ERROR = Q(3, 2) * (14 * _ETA + 2 * 10**10 * _WEIGHT)
_MATURE = Q(1, 2**28)
_ERROR = 8 * _MATURE
_FORCING = 2 * _WEIGHT
_GAMMA_LOWER = Q(1, 3300)
_GAMMA_UPPER = Q(1, 3000)


@dataclass(frozen=True)
class _MixedPositiveInterface:
    """Candidate form-record intervals with independently checked premises.

    The old phase reading is not consumed. The zero entry total is an exact
    consequence of the preceding laws, not an extra measured baseline. The
    proposed source associations and readout error are not authenticated by
    this detached calculation.
    """

    form_error_bounds: tuple[Q, Q]
    phase_error_bounds: tuple[Q, Q]
    observation_error_bound: Q
    handoff: _MixedCarriedFormPhase
    source_association_certified: bool
    projected_rows_certified: bool
    cut_form_error_certified: bool
    new_trap_certified: bool
    inherited_orientation_certified: bool
    nominal_transfer_certified: bool
    component_comparison_certified: bool
    identity_certified: bool
    nominal_q0_bounds: tuple[Q, Q]
    cut_decay_upper_bound: Q
    cut_form_error_upper_bound: Q
    handoff_phase_error_upper_bound: Q
    component_coupling: Q
    component_denominator: Q
    positive_form_error_upper_bound: Q
    positive_phase_error_upper_bound: Q
    comparison_forcing_upper_bound: Q
    comparison_forcing_margin: Q
    new_norm_upper_bound: Q
    nominal_entry_norm_upper_bound: Q
    phase_difference_entry_lower_bound: Q
    phase_difference_drift_upper_bound: Q
    phase_difference_lower_bound: Q
    common_phase_sum_absolute_upper_bound: Q
    marked_gap_drift_upper_bound: Q
    marked_gap_upper_bound: Q
    registration_margin: Q
    inherited_sine_factor_lower_bound: Q
    inherited_restoring_fraction_margin: Q
    inherited_velocity_decay_upper_bound: Q
    inherited_form_contrast_lower_bound: Q
    nominal_transfer_lower_bound: Q
    transfer_error_upper_bound: Q
    positive_true_bounds: tuple[Q, Q]
    cut_true_bounds: tuple[Q, Q]
    observation_allowance_admitted: bool
    positive_record_bounds: tuple[Q, Q]
    cut_record_bounds: tuple[Q, Q]
    record_separation_margin: Q
    positive_threshold_margin: Q
    cut_threshold_margin: Q
    independent_record_sets_separated: bool
    fixed_threshold_certified: bool
    certified: bool
    status: str
    unresolved_conditions: tuple[str, ...]
    prefix_full_duration: Q = _FULL
    prefix_cut_duration: Q = _CUT
    handoff_time: Q = _HANDOFF
    positive_duration: Q = _DURATION
    total_duration: Q = _HANDOFF + _DURATION
    original_clock_horizon: Q = (_HANDOFF + _DURATION) * Q(1024, 1023)
    observation_error_cap: Q = _NOISE
    threshold: Q = _THRESHOLD
    numerical_response_allowance: Q = Q(0)
    handoff_observations: int = 0
    independently_observed_aggregates: int = 1
    initial_donor_form_total: Q = Q(0)
    whole_form_mean: Q = Q(0)
    nominal_coordinates: tuple[str, ...] = ("a", "v_D", "v_R", "q_D", "q_R")
    contact_pairs: tuple[tuple[int, int], ...] = tuple((j, 18 + j) for j in range(9))
    physical_capacities: tuple[Q, ...] = (Q(1),) * 36
    observation_covector: tuple[Q, ...] = (-Q(1),) * 18 + (Q(0),) * 54
    outgoing_current_covector: tuple[Q, ...] = (
        (Q(1),) * 9 + (Q(0),) * 9 + (-Q(1),) * 9 + (Q(0),) * 45
    )
    outgoing_current_covector_norm_squared: Q = Q(18)
    full_state_dimension: int = 72
    old_phase_reading_consumed: bool = False
    exact_zero_entry_total_from_prior_laws: bool = True
    relative_common_form_mode_retained: bool = True
    original_phase_gauges_and_branches_retained: bool = True
    inherited_form_velocity_retained: bool = True
    state_reset_performed: bool = False
    old_commuting_metric_transferred: bool = False
    per_copy_zero_form_leaf_assumed_after_switch: bool = False
    storage_monotonicity_claimed_after_switch: bool = False
    exact_actual_five_coordinate_closure: bool = False
    minimal_collective_state_established: bool = False
    observed_record_supplied: bool = False
    source_trajectory_evaluated: bool = False
    runtime_event_executed: bool = False
    empirical_validation: bool = False
    clock: str = "tau=(1023/1024)*t; complete internal rows and reciprocal A form port"


def _assess_mixed_positive_interface(
    *, form_error_bounds, phase_error_bounds, observation_error_bound
):
    """Assess the fixed regional port from the actually reached old history."""
    forms = _nonnegative_pair(form_error_bounds, "form_error_bounds")
    phases = _nonnegative_pair(phase_error_bounds, "phase_error_bounds")
    noise = _nonnegative(observation_error_bound, "observation_error_bound")
    handoff = _assess_mixed_carried_form_phase(
        form_error_bounds=forms,
        phase_error_bounds=phases,
        observation_error_bound=Q(0),
    )
    prefix = handoff.prefix
    reference = prefix.reference
    registered = _derive_registered_phase_probe_geometry()
    geometry = registered.successor
    q0 = (_SECTION - _SECTION_ERROR / 2, _SECTION + _SECTION_ERROR / 2)
    source = (
        handoff.form_error_bounds == prefix.form_error_bounds == forms
        and handoff.phase_error_bounds == prefix.phase_error_bounds == phases
        and handoff.source_association_certified
        and prefix.source_association_certified
        and reference.source_association_certified
        and reference.donor_source_certified
        and reference.recipient_source_certified
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
        and reference.donor.section_mean_gap == reference.section_mean_gap == _SECTION
        and reference.donor.comparison_error_upper_bound
        == reference.donor_section_comparison_error
        == _SECTION_ERROR
        and reference.actual_weight_deviation_upper_bound == _WEIGHT
        and reference.recipient_isolated_deadline == Q(10**10)
        and reference.recipient_readiness.waiting_duration == Q(10**10)
        and reference.recipient_mature_radius
        == reference.recipient_readiness.derived_mature_norm_upper_bound
        == _MATURE
        and reference.initial_form_means == handoff.retained_form_means == (Q(0), Q(0))
        and reference.physical_capacities == handoff.physical_capacities == (Q(1),) * 36
        and reference.full_state_dimension == handoff.full_state_dimension == 72
        and reference.contact_pairs == tuple((j, 18 + j) for j in range(18))
        and reference.clock
        == "tau=(1023/1024)*t; complete internal rows and reciprocal form inputs"
        and handoff.prefix_duration == prefix.contact_duration == _FULL
        and prefix.original_clock_duration == _FULL * Q(1024, 1023)
        and handoff.continuation_duration == _CUT
        and handoff.total_duration == _HANDOFF
        and handoff.original_clock_horizon == _HANDOFF * Q(1024, 1023)
        and handoff.switch_word == (1, 0)
        and handoff.control_word == (0, 0)
        and handoff.clock == "tau=(1023/1024)*t; all complete internal rows retained"
        and handoff.observation_error_bound == 0
        and handoff.handoff_observations == 0
        and handoff.reference_family_carried
        and handoff.reached_velocity_retained
        and handoff.source_weight_association_retained
        and handoff.original_phase_gauges_and_branches_retained
        and not handoff.state_reset_performed
        and not reference.donor_dwell_applied
        and not reference.charge_or_leak_prefix_applied
        and not reference.phase_jump_performed
    )
    projected = (
        source
        and handoff.projected_rows_certified
        and geometry.nodes == tuple(range(18))
        and geometry.degrees == (9,) * 18
        and geometry.held_capacities == (Q(1),) * 18
        and geometry.support_certified
        and geometry.reference_stationary_certified
        and reference.donor.nominal_reference_certified
        and handoff.nominal_q0_bounds == reference.nominal_q0_bounds == q0
        and reference.inherited_donor_velocity_retained
        and reference.inherited_nominal_velocity_relation
        == "-gamma**2*q0 <= w0 <= 0 at the same isolated age"
    )
    rate = reference.conservative_metric_decay_rate
    comparison = (
        projected
        and handoff.handoff_certified
        and handoff.switched_metric_certified
        and handoff.identity_certified
        and reference.comparison_certified
        and reference.metric_certified
        and reference.actual_trap_certified
        and reference.initial_comparison_norm_upper_bound == _MATURE + _SECTION_ERROR
        and reference.phase_scale == Q(6, 5)
        and reference.metric_lower_bound
        == handoff.metric_lower_bound
        == Q(9, 4) * (1 - Q(1, 2500))
        and reference.metric_upper_bound
        == handoff.metric_upper_bound
        == Q(9, 4) * (1 + Q(1, 2500)) + Q(27, 2)
        and rate
        == handoff.metric_decay_rate
        == reference.base_metric_decay_rate / 8
        > 0
        and reference.comparison_forcing_upper_bound
        == handoff.metric_forcing_upper_bound
        == _FORCING
        and reference.comparison_matrix_deviation_upper_bound == 14 * _WEIGHT
        and handoff.carried_error_upper_bound
        == reference.uniform_comparison_bound
        == 4 * (_MATURE + _SECTION_ERROR + 2 * _FORCING / rate)
        < _ERROR
        and handoff.carried_metric_upper_bound
        == reference.metric_upper_bound
        * Q(6, 5) ** 2
        * (_MATURE + _SECTION_ERROR + 2 * _FORCING / rate) ** 2
        and handoff.actual_observation_allowance
        == reference.uniform_comparison_upper_bound
        == _ERROR
    )
    cut_decay = 1 / (1 + _CUT / 3)
    cut_error = _ERROR * cut_decay + 6 * _GAMMA_UPPER * _ERROR + 3 * _FORCING
    sharpened = (
        comparison
        and geometry.weighted_gap_lower_bound >= 3
        and 0 < cut_decay < 1
        and 0 < cut_error < _ERROR
    )
    s = 2 * _GAMMA_UPPER * _DURATION
    denominator = 1 - s**2
    norm = Q(1, 100) / (1 - s)
    nominal_entry_norm = 3 * (1 + _GAMMA_UPPER) * q0[1]
    trap = (
        projected
        and reference.actual_trap_certified
        and handoff.continuation_wedges_certified
        and reference.nominal_wedges_certified
        and reference.nominal_velocity_factor_upper_bound == _GAMMA_UPPER**2
        and reference.full_norm_upper_bound == Q(1, 100)
        and reference.phase_radius == Q(1, 32)
        and 0 < s < 1
        and nominal_entry_norm < Q(1, 100)
        and norm < Q(1, 32)
    )
    forcing_margin = _FORCING - Q(28, 9) * _WEIGHT * norm
    ux = cut_error + _FORCING * _DURATION + s * (_ERROR + _FORCING * _DURATION)
    ux /= denominator
    uy = _ERROR + _FORCING * _DURATION + s * (cut_error + _FORCING * _DURATION)
    uy /= denominator
    component = sharpened and trap and denominator > 0 and forcing_margin > 0
    entry_z = Q(233, 300) * q0[0]
    z_drift = Q(4, 3) * _GAMMA_UPPER * _DURATION * norm
    z_lower = entry_z - z_drift
    sum_upper = 2 * norm / 3
    gap_drift = 4 * _GAMMA_UPPER * _DURATION * norm
    gap_upper = q0[1] + 12 * _MATURE + gap_drift
    registration = _DELTA / 4 - gap_upper
    identity = (
        trap
        and handoff.identity_certified
        and reference.registration_gap_limit == _DELTA / 4
        and reference.marked_gap_upper_bound == q0[1] + 12 * _MATURE
        and registration > 0
    )
    sine_factor = 1 - q0[1] ** 2 / 6
    restoring_fraction_margin = Q(233, 300) * Q(99, 100) - Q(3, 4)
    velocity_decay = 1 / (1 + Q(8, 9) * _CUT)
    form_contrast = q0[0] / 20000
    donor = reference.donor
    orientation = (
        projected
        and handoff.continuation_wedges_certified
        and reference.nominal_wedges_certified
        and reference.nominal_velocity_factor_upper_bound == _GAMMA_UPPER**2
        and reference.nominal_wedge_margin
        == 2 * donor.nominal_coupling_bounds.lo / 3 - _GAMMA_UPPER**2
        > 0
        and _GAMMA_LOWER
        <= donor.gamma_bounds.lo
        <= donor.gamma_bounds.hi
        <= _GAMMA_UPPER
        and Q(4, 9)
        < donor.nominal_coupling_bounds.lo
        <= donor.nominal_coupling_bounds.hi
        < Q(1, 2)
        and 2
        < donor.regional_sine_coefficient_bounds.lo
        <= donor.regional_sine_coefficient_bounds.hi
        < 3
        and 1 - _GAMMA_UPPER**2 * _HANDOFF == Q(233, 300)
        and sine_factor > Q(99, 100)
        and restoring_fraction_margin > 0
        and velocity_decay < Q(1, 100)
        and Q(99, 100) * _GAMMA_LOWER * 2 * Q(3, 4) / 9 == Q(1, 20000)
    )
    nominal_lower = 9 * form_contrast * (1 - _DURATION) * (_DURATION - _DURATION**2 / 2)
    nominal = (
        orientation
        and trap
        and identity
        and z_lower > 0
        and sum_upper < 1
        and nominal_lower == Q(27, 16) * form_contrast > 0
    )
    transfer_error = Q(9, 2) * _DURATION * ux
    positive_true = (nominal_lower - transfer_error, 54 * _DURATION * norm)
    cut_true = (Q(0), Q(0))
    admitted = noise <= _NOISE
    positive_record = (positive_true[0] - noise, positive_true[1] + noise)
    cut_record = (-noise, noise)
    separation = positive_record[0] - cut_record[1]
    positive_threshold = positive_record[0] - _THRESHOLD
    cut_threshold = _THRESHOLD - cut_record[1]
    separated = (
        nominal
        and component
        and admitted
        and positive_true[0] <= positive_true[1]
        and separation > 0
    )
    threshold = separated and positive_threshold > 0 and cut_threshold > 0
    unresolved = tuple(
        label
        for good, label in (
            (source, "actually_reached_original_source_and_history_unavailable"),
            (projected, "regional_interface_and_live_common_form_mode_unavailable"),
            (sharpened, "paid_cut_form_error_sharpening_unavailable"),
            (trap, "new_interface_norm_trap_unavailable"),
            (
                identity,
                "original_registered_identity_during_positive_contact_unavailable",
            ),
            (orientation, "inherited_nonzero_form_orientation_unavailable"),
            (nominal, "cooperative_nominal_outgoing_transfer_unavailable"),
            (component, "both_row_component_error_budget_unavailable"),
            (admitted, "aggregate_form_observation_error_exceeds_protocol"),
            (separated, "independent_form_record_sets_not_certified_separate"),
            (threshold, "fixed_threshold_not_certified_between_form_records"),
        )
        if not good
    )
    return _MixedPositiveInterface(
        form_error_bounds=forms,
        phase_error_bounds=phases,
        observation_error_bound=noise,
        handoff=handoff,
        source_association_certified=source,
        projected_rows_certified=projected,
        cut_form_error_certified=sharpened,
        new_trap_certified=trap,
        inherited_orientation_certified=orientation,
        nominal_transfer_certified=nominal,
        component_comparison_certified=component,
        identity_certified=identity,
        nominal_q0_bounds=q0,
        cut_decay_upper_bound=cut_decay,
        cut_form_error_upper_bound=cut_error,
        handoff_phase_error_upper_bound=_ERROR,
        component_coupling=s,
        component_denominator=denominator,
        positive_form_error_upper_bound=ux,
        positive_phase_error_upper_bound=uy,
        comparison_forcing_upper_bound=_FORCING,
        comparison_forcing_margin=forcing_margin,
        new_norm_upper_bound=norm,
        nominal_entry_norm_upper_bound=nominal_entry_norm,
        phase_difference_entry_lower_bound=entry_z,
        phase_difference_drift_upper_bound=z_drift,
        phase_difference_lower_bound=z_lower,
        common_phase_sum_absolute_upper_bound=sum_upper,
        marked_gap_drift_upper_bound=gap_drift,
        marked_gap_upper_bound=gap_upper,
        registration_margin=registration,
        inherited_sine_factor_lower_bound=sine_factor,
        inherited_restoring_fraction_margin=restoring_fraction_margin,
        inherited_velocity_decay_upper_bound=velocity_decay,
        inherited_form_contrast_lower_bound=form_contrast,
        nominal_transfer_lower_bound=nominal_lower,
        transfer_error_upper_bound=transfer_error,
        positive_true_bounds=positive_true,
        cut_true_bounds=cut_true,
        observation_allowance_admitted=admitted,
        positive_record_bounds=positive_record,
        cut_record_bounds=cut_record,
        record_separation_margin=separation,
        positive_threshold_margin=positive_threshold,
        cut_threshold_margin=cut_threshold,
        independent_record_sets_separated=separated,
        fixed_threshold_certified=threshold,
        certified=not unresolved,
        status=(
            "conditional_positive_regional_interface"
            if not unresolved
            else "unavailable"
        ),
        unresolved_conditions=unresolved,
    )
