"""A finite observable return deadline for the unchanged actual A contact.

Strict storage controls the complete state, including relative common form.
The deadline is measured from original A activation, not from a reset of the
previous half-unit endpoint. Two independently noisy aggregate records refer
to the same actual history. No trajectory or measured return time is supplied.
"""

from dataclasses import dataclass
from fractions import Fraction as Q

from ..mathematics._rational_interval import cos, pi_interval
from ._mixed_persistent_contact import (
    _assess_mixed_persistent_contact,
    _MixedPersistentContact,
)
from ._mixed_reciprocal_partner import _nonnegative, _nonnegative_pair
from ._mixed_registered_phase_probe import _derive_registered_phase_probe_geometry
from ._mixed_resource_transfer_obstruction import _original_source_matches
from ._sine_lyapunov import _sine_strict_energy_coefficients

_DELTA = Q(1, 1024)
_WEIGHT = Q(2, 10**60)
_MATURE = Q(1, 2**28)
_ETA = Q(2, 10**69)
_SECTION = 3 * _DELTA / 16
_SECTION_ERROR = Q(3, 2) * (14 * _ETA + 2 * 10**10 * _WEIGHT)
_GAMMA_LOWER, _GAMMA_UPPER = Q(1, 3300), Q(1, 3000)
_EPSILON = Q(1, 2000)
_ACTIVATION = Q(2010000)
_EARLY = Q(1, 2)
_RETURN = Q(627264000000000)
_TOLERANCE = Q(1, 10**9)
_THRESHOLD = Q(1, 10**8)


@dataclass(frozen=True)
class _MixedFiniteReturn:
    """A sufficient all-future deadline and conditional two-record contract."""

    form_error_bounds: tuple[Q, Q]
    phase_error_bounds: tuple[Q, Q]
    early_observation_error_bound: Q
    late_observation_error_bound: Q
    source: _MixedPersistentContact
    source_association_certified: bool
    strict_decay_certified: bool
    finite_return_certified: bool
    early_transfer_certified: bool
    observation_allowances_admitted: bool
    independent_record_sets_separated: bool
    fixed_threshold_certified: bool
    initial_excess_storage_upper_bound: Q
    strict_storage_lower_bound: Q
    strict_storage_upper_bound: Q
    strict_form_loss_lower_bound: Q
    strict_phase_loss_lower_bound: Q
    strict_decay_rate: Q | None
    conservative_decay_rate: Q
    strict_initial_upper_bound: Q
    returned_norm_squared_upper_bound: Q
    aggregate_squared_upper_bound: Q
    aggregate_return_margin: Q
    early_true_bounds: tuple[Q, Q]
    late_true_bounds: tuple[Q, Q]
    early_record_bounds: tuple[Q, Q]
    late_record_bounds: tuple[Q, Q]
    returned_form_true_bounds: tuple[Q, Q]
    returned_form_record_bounds: tuple[Q, Q]
    early_threshold_margin: Q
    late_threshold_margin: Q
    record_separation_margin: Q
    certified: bool
    status: str
    unresolved_conditions: tuple[str, ...]
    cross_coefficient: Q = _EPSILON
    storage_lower_bound: Q = Q(1, 5)
    storage_upper_bound: Q = Q(9)
    form_loss_lower_bound: Q = Q(1, 5)
    transport_norm_upper_bound: Q = Q(18)
    damping_norm_upper_bound: Q = Q(4)
    phase_stiffness_lower_bound: Q = Q(2, 5)
    bare_phase_gap_lower_bound: Q = Q(4)
    phase_cosine_lower_bound: Q = Q(1, 10)
    gamma_bounds: tuple[Q, Q] = (_GAMMA_LOWER, _GAMMA_UPPER)
    loss_divisor: Q = Q(9)
    activation_time: Q = _ACTIVATION
    early_age: Q = _EARLY
    return_age: Q = _RETURN
    total_return_time: Q = _ACTIVATION + _RETURN
    original_clock_return_time: Q = (_ACTIVATION + _RETURN) * Q(1024, 1023)
    decay_power: int = 64
    decay_upper_bound: Q = Q(1, 2**64)
    true_return_tolerance: Q = _TOLERANCE
    early_observation_error_cap: Q = _TOLERANCE
    late_observation_error_cap: Q = _TOLERANCE
    threshold: Q = _THRESHOLD
    aggregate_covector: tuple[Q, ...] = (-Q(1),) * 18 + (Q(0),) * 54
    centered_aggregate_covector: tuple[Q, ...] = (
        (-Q(1, 2),) * 18 + (Q(1, 2),) * 18 + (Q(0),) * 36
    )
    centered_aggregate_covector_norm_squared: Q = Q(9)
    fixed_port_strength: Q = Q(1)
    whole_form_mean: Q = Q(0)
    all_future_ages_covered: bool = False
    same_actual_history_required: bool = True
    independent_endpoint_errors: bool = True
    endpoint_error_cancellation_assumed: bool = False
    state_error_reinitialized: bool = False
    source_association_authenticated: bool = False
    observed_records_supplied: bool = False
    current_or_phase_precision_granted: bool = False
    first_return_time_identified: bool = False
    exact_finite_zero_arrival_claimed: bool = False
    observed_relaxation_rate: Q | None = None
    aggregate_monotonicity_claimed: bool = False
    state_reset_performed: bool = False
    port_switch_performed: bool = False
    source_trajectory_evaluated: bool = False
    runtime_event_executed: bool = False
    numerical_response_allowance: Q = Q(0)
    newly_observed_aggregates: int = 0
    required_endpoint_records: int = 2
    physical_clock_identified: bool = False
    empirical_validation: bool = False
    clock: str = "tau=(1023/1024)*t; age from original unit A-port activation"


def _assess_mixed_finite_return(
    *,
    form_error_bounds,
    phase_error_bounds,
    early_observation_error_bound,
    late_observation_error_bound,
):
    """Rebuild strict decay and independently priced early/late record bands."""
    forms = _nonnegative_pair(form_error_bounds, "form_error_bounds")
    phases = _nonnegative_pair(phase_error_bounds, "phase_error_bounds")
    early_noise = _nonnegative(
        early_observation_error_bound, "early_observation_error_bound"
    )
    late_noise = _nonnegative(
        late_observation_error_bound, "late_observation_error_bound"
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
    lower, upper, form_loss, phase_loss, rate = _sine_strict_energy_coefficients(
        storage_lower=Q(1, 5),
        storage_upper=Q(9),
        form_loss_lower=Q(1, 5),
        transport_norm_upper=Q(18),
        damping_norm_upper=Q(4),
        phase_stiffness_lower=Q(2, 5),
        gamma_lower=_GAMMA_LOWER,
        gamma_upper=_GAMMA_UPPER,
        loss_divisor=Q(9),
        cross_coefficient=_EPSILON,
    )
    conservative_rate = _EPSILON * _GAMMA_LOWER**2 / 450
    strict = (
        source_ok
        and source.broad_trap_certified
        and source.phase_convexity_certified
        and storage.actual_dissipation_certified
        and positive.projected_rows_certified
        and geometry.nodes == tuple(range(18))
        and geometry.degrees == (9,) * 18
        and geometry.held_capacities == (Q(1),) * 18
        and geometry.support_certified
        and geometry.reference_stationary_certified
        and geometry.spectral_gap_certified
        and geometry.bare_gap_lower_bound == 4
        and geometry.weighted_gap_lower_bound >= 3
        and geometry.quotient_coercivity_lower_bound >= Q(1, 5)
        and all(row.phase_cosine_lower_bound >= Q(1, 10) for row in reference.sources)
        and source.storage_coercivity_lower_bound
        == storage.storage_coercivity_lower_bound
        == Q(1, 5)
        and source.storage_norm_upper_bound == storage.storage_norm_upper_bound == 9
        and source.selected_mu
        == source.fixed_port_strength
        == storage.selected_mu
        == storage.fixed_port_strength
        == 1
        and storage.mean_projector_rank == 1
        and storage.port_norm_upper_bound == 2
        and storage.loss_divisor == 9
        and source.actual_dissipation_lower_bound
        == storage.actual_dissipation_lower_bound
        == actual_loss
        > Q(1, 5)
        and storage.actual_weight_perturbation_upper_bound == 84 * _WEIGHT
        and source.initial_excess_storage_upper_bound
        == reference.initial_excess_storage_upper_bound
        == energy
        and source.full_norm_squared_upper_bound == 5 * energy < Q(1, 10000)
        and source.storage_barrier == Q(1, 5) * Q(1, 32) ** 2
        and source.storage_margin == source.storage_barrier - energy > 0
        and source.phase_radius == reference.phase_radius == Q(1, 32)
        and reference.donor.gamma_bounds == gamma
        and _GAMMA_LOWER <= gamma.lo <= gamma.hi <= _GAMMA_UPPER
        and source.gamma_bounds == (_GAMMA_LOWER, _GAMMA_UPPER)
        and source.coupling_bounds
        == storage.coupling_bounds
        == reference.donor.nominal_coupling_bounds
        == coupling
        and storage.cross_strength_bounds == 9 * coupling
        and Q(4, 9) < coupling.lo <= coupling.hi < Q(1, 2)
        and lower > Q(1, 6)
        and upper < 10
        and form_loss > Q(1, 12)
        and phase_loss > _EPSILON * _GAMMA_LOWER**2 / 45
        and _EPSILON * _GAMMA_LOWER**2 / 45 < Q(1, 12)
        and rate is not None
        and rate >= conservative_rate > 0
        and 1 + Q(5, 2) * _EPSILON * _GAMMA_UPPER < 2
    )
    strict_initial = 2 * energy
    returned_norm_squared = 6 * strict_initial / 2**64
    aggregate_squared = 9 * returned_norm_squared
    return_margin = _TOLERANCE**2 - aggregate_squared
    returned = (
        strict
        and conservative_rate == Q(1, 9801000000000)
        and conservative_rate * _RETURN == 64
        and _RETURN > _EARLY
        and return_margin > 0
        and positive.observation_covector == (-Q(1),) * 18 + (Q(0),) * 54
    )
    # The earlier true response is rebuilt. Its cached record bands, noise
    # verdict and global report are irrelevant to these two new readout errors.
    error, forcing = 8 * _MATURE, 2 * _WEIGHT
    component = 2 * _GAMMA_UPPER * _EARLY
    norm = Q(1, 100) / (1 - component)
    cut_error = error / (1 + Q(10000, 3)) + 6 * _GAMMA_UPPER * error + 3 * forcing
    ux = (cut_error + forcing * _EARLY + component * (error + forcing * _EARLY)) / (
        1 - component**2
    )
    uy = (error + forcing * _EARLY + component * (cut_error + forcing * _EARLY)) / (
        1 - component**2
    )
    nominal_lower = Q(27, 320000) * q0[0]
    transfer_error = Q(9, 2) * _EARLY * ux
    early_true = (nominal_lower - transfer_error, 54 * _EARLY * norm)
    early = (
        source_ok
        and positive.inherited_orientation_certified
        and positive.nominal_transfer_certified
        and positive.component_comparison_certified
        and positive.cut_form_error_certified
        and positive.identity_certified
        and positive.new_trap_certified
        and source.nominal_q0_bounds
        == reference.nominal_q0_bounds
        == positive.nominal_q0_bounds
        == q0
        and source.coupling_bounds
        == reference.donor.nominal_coupling_bounds
        == coupling
        and Q(4, 9) < coupling.lo <= coupling.hi < Q(1, 2)
        and reference.donor.regional_sine_coefficient_bounds == sine_coefficient
        and 2 < sine_coefficient.lo <= sine_coefficient.hi < 3
        and handoff.continuation_wedges_certified
        and positive.inherited_form_contrast_lower_bound == q0[0] / 20000
        and positive.inherited_sine_factor_lower_bound
        == 1 - q0[1] ** 2 / 6
        > Q(99, 100)
        and positive.inherited_velocity_decay_upper_bound
        == 1 / (1 + Q(80000, 9))
        < Q(1, 100)
        and positive.inherited_restoring_fraction_margin
        == Q(233, 300) * Q(99, 100) - Q(3, 4)
        > 0
        and positive.phase_difference_entry_lower_bound == Q(233, 300) * q0[0]
        and positive.phase_difference_drift_upper_bound
        == Q(4, 3) * _GAMMA_UPPER * _EARLY * norm
        and positive.phase_difference_lower_bound
        == Q(233, 300) * q0[0] - Q(4, 3) * _GAMMA_UPPER * _EARLY * norm
        > 0
        and positive.common_phase_sum_absolute_upper_bound == 2 * norm / 3 < 1
        and positive.cut_form_error_upper_bound == cut_error
        and positive.handoff_phase_error_upper_bound == error
        and positive.component_coupling == component
        and positive.component_denominator == 1 - component**2
        and positive.positive_form_error_upper_bound == ux
        and positive.positive_phase_error_upper_bound == uy
        and positive.comparison_forcing_upper_bound == forcing
        and positive.comparison_forcing_margin
        == forcing - Q(28, 9) * _WEIGHT * norm
        > 0
        and positive.new_norm_upper_bound == norm < Q(1, 32)
        and positive.nominal_transfer_lower_bound == nominal_lower
        and positive.transfer_error_upper_bound == transfer_error
        and early_true[1] >= early_true[0] > 0
    )
    late_true = (-_TOLERANCE, _TOLERANCE)
    early_record = (early_true[0] - early_noise, early_true[1] + early_noise)
    late_record = (late_true[0] - late_noise, late_true[1] + late_noise)
    true_difference = (early_true[0] - late_true[1], early_true[1] - late_true[0])
    record_difference = (
        early_record[0] - late_record[1],
        early_record[1] - late_record[0],
    )
    admitted = early_noise <= _TOLERANCE and late_noise <= _TOLERANCE
    early_margin = early_record[0] - _THRESHOLD
    late_margin = _THRESHOLD - max(abs(value) for value in late_record)
    separation = record_difference[0]
    separated = early and returned and admitted and separation > 0
    threshold = separated and early_margin > 0 and late_margin > 0
    unresolved = tuple(
        label
        for good, label in (
            (source_ok, "same_actual_source_and_uninterrupted_A_history_unavailable"),
            (strict, "complete_state_strict_decay_premises_unavailable"),
            (returned, "finite_all_future_aggregate_return_bound_unavailable"),
            (early, "original_positive_half_unit_transfer_unavailable"),
            (admitted, "independent_endpoint_error_exceeds_protocol"),
            (separated, "early_late_record_sets_not_certified_separate"),
            (threshold, "fixed_threshold_not_certified_between_records"),
        )
        if not good
    )
    return _MixedFiniteReturn(
        form_error_bounds=forms,
        phase_error_bounds=phases,
        early_observation_error_bound=early_noise,
        late_observation_error_bound=late_noise,
        source=source,
        source_association_certified=source_ok,
        strict_decay_certified=strict,
        finite_return_certified=returned,
        early_transfer_certified=early,
        observation_allowances_admitted=admitted,
        independent_record_sets_separated=separated,
        fixed_threshold_certified=threshold,
        initial_excess_storage_upper_bound=energy,
        strict_storage_lower_bound=lower,
        strict_storage_upper_bound=upper,
        strict_form_loss_lower_bound=form_loss,
        strict_phase_loss_lower_bound=phase_loss,
        strict_decay_rate=rate,
        conservative_decay_rate=conservative_rate,
        strict_initial_upper_bound=strict_initial,
        returned_norm_squared_upper_bound=returned_norm_squared,
        aggregate_squared_upper_bound=aggregate_squared,
        aggregate_return_margin=return_margin,
        early_true_bounds=early_true,
        late_true_bounds=late_true,
        early_record_bounds=early_record,
        late_record_bounds=late_record,
        returned_form_true_bounds=true_difference,
        returned_form_record_bounds=record_difference,
        early_threshold_margin=early_margin,
        late_threshold_margin=late_margin,
        record_separation_margin=separation,
        certified=not unresolved,
        status="conditional_finite_return" if not unresolved else "unavailable",
        unresolved_conditions=unresolved,
        all_future_ages_covered=returned,
    )
