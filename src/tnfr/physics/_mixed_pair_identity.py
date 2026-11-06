"""A maintained marked pair and its oriented, source-specific prediction.

Membership and an invariant representation are different contracts. The
existing effective law predicts only the admitted original source family;
the larger membership class retains an explicit exact-closure obstruction.
"""

from dataclasses import dataclass
from fractions import Fraction as Q

from ..mathematics._rational_interval import I, cos, pi_interval, sin
from ._mixed_collective_sensitivity import (
    _assess_mixed_collective_sensitivity,
    _MixedCollectiveSensitivity,
)
from ._mixed_reciprocal_partner import _nonnegative, _nonnegative_pair
from ._mixed_registered_phase_probe import _derive_registered_phase_probe_geometry
from ._mixed_resource_transfer_obstruction import _original_source_matches

_DELTA, _ETA, _WEIGHT = Q(1, 1024), Q(2, 10**69), Q(2, 10**60)
_GL, _G, _MATURE = Q(1, 3300), Q(1, 3000), Q(1, 2**28)
_TOLERANCE, _RADIUS = Q(1, 10**9), Q(1, 500)
_ACTIVATION, _SWITCH, _H = Q(2010000), Q(3375000), Q(8)
_PAIRS = tuple((j, 18 + j) for j in range(18))


@dataclass(frozen=True)
class _MixedPairIdentity:
    """Conditional marked membership with retained physical port association."""

    form_error_bounds: tuple[Q, Q]
    phase_error_bounds: tuple[Q, Q]
    registered_observation_error_bound: Q
    aligned_observation_error_bound: Q
    source: _MixedCollectiveSensitivity
    source_association_certified: bool
    source_maintained_membership_certified: bool
    complete_law_transport_certified: bool
    common_origin_invariance_certified: bool
    transported_relabeling_covariance_certified: bool
    constituent_exchange_covariance_certified: bool
    invariant_chart_certified: bool
    oriented_port_descent_certified: bool
    source_prediction_certified: bool
    inherited_negative_band_certified: bool
    port_association_obstruction_certified: bool
    larger_class_exact_nonclosure_certified: bool
    phase_resultant_boundary_certified: bool
    switching_work_certified: bool
    observation_allowances_admitted: bool
    gamma_bounds: I
    nominal_q0_bounds: tuple[Q, Q]
    source_comparison_upper_bound: Q
    centered_entry_norm_upper_bound: Q
    scaled_norm_squared_upper_bound: Q
    scaled_radius_margin: Q
    marked_gap_upper_bound: Q
    registration_margin: Q
    phase_sector_bound: I
    phase_lift_margin: Q
    marked_pairs: tuple[tuple[int, int], ...]
    transported_windings: tuple[int, ...]
    aggregate_absolute_upper_bound: Q
    uniform_form_error_upper_bound: Q
    uniform_phase_error_upper_bound: Q
    invariant_prediction_error_bounds: tuple[Q, Q, Q]
    registered_form_record_error_upper_bound: Q
    aligned_form_record_error_upper_bound: Q
    inherited_registered_true_bounds: tuple[Q, Q]
    port_blind_error_lower_bound: Q
    port_association_obstruction_margin: Q
    phase_switching_work_allowance: Q
    aligned_storage_jump_bounds: tuple[Q, Q]
    registered_storage_jump_bounds: tuple[Q, Q]
    witness_qstar: Q
    witness_offset_amplitude: Q
    witness_matching_offsets: tuple[Q, ...]
    witness_centered_phase: tuple[Q, ...]
    witness_scaled_norm_squared: Q
    witness_marked_gap_upper_bound: Q
    witness_moment_difference_bounds: I
    witness_form_rate_difference_bounds: I
    witness_invariant_u_rate_difference_bounds: I
    certified: bool
    status: str
    unresolved_conditions: tuple[str, ...]
    invariant_coordinates: tuple[str, ...] = ("Q**2", "Q*sin(psi)", "cos(psi)")
    invariant_constraint: str = "V**2=U*(1-W**2)"
    invariant_coordinate_count: int = 3
    invariant_continuous_dimension: int = 2
    invariant_effective_rows: tuple[str, ...] = (
        "U_prime=-4*U-36*gamma*V",
        "V_prime=-2*V-18*gamma*(1-W**2)+(2*gamma/9)*U*W",
        "W_prime=-(2*gamma/9)*V",
    )
    invariant_actual_defects: tuple[str, ...] = (
        "-2*gamma*Q*R",
        "-gamma*R*sin(psi)",
        "0",
    )
    physical_port_observation: str = "o*Q; o transforms with the local copy chart"
    copy_chart_reversal: tuple[str, ...] = ("-Q", "-psi", "-R", "-o")
    invariant_singular_point: tuple[Q, Q, Q] = (Q(0), Q(0), Q(1))
    full_state_dimension: int = 72
    reduced_state_dimension: int = 2
    physical_capacities: tuple[Q, ...] = (Q(1),) * 36
    full_contact_pairs: tuple[tuple[int, int], ...] = _PAIRS
    internal_divisor: Q = Q(9)
    whole_form_mean: Q = Q(0)
    original_donor_form_total: Q = Q(0)
    initial_registrations_turns: tuple[Q, Q] = (Q(0), Q(1, 4))
    activation_time: Q = _ACTIVATION
    switch_age: Q = _SWITCH
    switch_time: Q = _ACTIVATION + _SWITCH
    original_clock_start_time: Q = (_ACTIVATION + _SWITCH) * Q(1024, 1023)
    inherited_discrimination_age: Q = _H
    intrinsic_form_tolerance: Q = _TOLERANCE
    registration_gap_limit: Q = _DELTA / 4
    scaled_radius_limit: Q = _RADIUS
    original_correlated_initializer: str = "(Q_s(3375000),alpha); alpha in{0,pi/2}"
    common_storage_reference: str = "same unshifted Theta in Uplus(Theta)"
    source_membership_retained_for_all_times: bool = False
    original_source_error_paid_once: bool = True
    source_correlated_initialization_retained: bool = True
    original_attachment_work_retained_once: bool = True
    weights_support_cycles_and_history_transported: bool = True
    physical_port_association_retained: bool = True
    membership_implies_arbitrary_source_admission: bool = False
    arbitrary_member_maintenance_certified: bool = False
    arbitrary_member_prediction_certified: bool = False
    arbitrary_age_predictor_restart_admitted: bool = False
    exact_actual_invariant_closure: bool = False
    globally_regular_invariant_chart: bool = False
    invariant_coordinates_independently_free: bool = False
    fixed_weight_swap_symmetry_claimed: bool = False
    primitive_whole_pair_phase_defined: bool = False
    independent_phase_representatives_quotiented: bool = False
    circular_resultant_nonzero_lower_bound: Q | None = None
    time_dependent_phase_frame_admitted: bool = False
    orientation_metadata_is_new_state_coordinate: bool = False
    relabeling_is_physical_intervention: bool = False
    original_noise_arguments_reassigned_to_port_readings: bool = False
    invariant_observables_share_one_accuracy_target: bool = False
    witness_is_acquired_source: bool = False
    witness_refutes_source_approximation: bool = False
    pair_acquisition_claimed: bool = False
    autonomous_pair_formation_claimed: bool = False
    intrinsic_pair_epi_or_phase_identified: bool = False
    new_field_installed: bool = False
    new_observation_acquired: bool = False
    state_reset_performed: bool = False
    source_trajectory_evaluated: bool = False
    reduced_trajectory_evaluated: bool = False
    original_source_replayed: bool = False
    runtime_event_executed: bool = False
    physical_comparator_installed: bool = False
    physical_property_identified: bool = False
    empirical_validation: bool = False
    clock: str = "tau=(1023/1024)*t; transported actual and effective descriptions"


def _assess_mixed_pair_identity(
    *,
    form_error_bounds,
    phase_error_bounds,
    registered_observation_error_bound,
    aligned_observation_error_bound,
):
    """Readmit the source, then price membership and the retained response."""
    forms = _nonnegative_pair(form_error_bounds, "form_error_bounds")
    phases = _nonnegative_pair(phase_error_bounds, "phase_error_bounds")
    registered_noise = _nonnegative(
        registered_observation_error_bound, "registered_observation_error_bound"
    )
    aligned_noise = _nonnegative(
        aligned_observation_error_bound, "aligned_observation_error_bound"
    )
    source = _assess_mixed_collective_sensitivity(
        form_error_bounds=forms,
        phase_error_bounds=phases,
        registered_observation_error_bound=Q(0),
        aligned_observation_error_bound=Q(0),
    )
    marked = source.source
    signed = marked.source
    sustained = signed.source
    effective = sustained.source
    registration = effective.source
    phase = registration.source
    slow = phase.source.source
    persistent = slow.source.source
    positive = persistent.source.source
    handoff = positive.handoff
    reference = handoff.prefix.reference
    geometry = _derive_registered_phase_probe_geometry().successor
    pi = pi_interval()
    gamma = 1 / (1023 * pi)
    section_error = Q(3, 2) * (14 * _ETA + 2 * 10**10 * _WEIGHT)
    qlo, qhi = 3 * _DELTA / 16 - section_error / 2, 3 * _DELTA / 16 + section_error / 2
    comparison = 3 * (24 * _MATURE + 84 * _WEIGHT * 10 * qhi / _GL) / (1 - 135 * qhi)
    budget = 200 * _MATURE
    cut_error = 8 * _MATURE / (1 + Q(10000, 3)) + 48 * _G * _MATURE + 6 * _WEIGHT
    force = 2 * _G * budget / 9 + 5 * _G * budget**2 / 81 + 14 * _WEIGHT / 2700
    inherited_error = (
        3 * cut_error + Q(81, 8) * force + 3 * _G * qhi**3 + 27 * _G * qhi / 2**16
    )
    amplitude_upper = 16 * _G * qhi
    entry, entry_aggregate = 4 * qhi + budget, amplitude_upper + inherited_error
    aggregate = entry_aggregate + 9 * _G
    lower, upper = 1 - 2 * _G, 1 + 2 * _G
    square = (
        upper
        * ((4 * _G * qhi + comparison) ** 2 + 4 * (Q(9, 4) * qhi + comparison) ** 2)
        / lower
    )
    root_bound = 2 * (1 + _G) * comparison + 14 * _WEIGHT * qhi / (3 * _GL)
    marked_gap = qhi + 3 * root_bound / (4 * lower)
    sector = pi / 2 + Q(1, 1000)
    branch_margin = pi.lo - sector.hi
    marked_pairs = tuple(
        (a + offset, b + offset) for offset in (0, 18) for a, b in geometry.marked_pairs
    )
    windings = tuple(
        sum(
            (
                (
                    geometry.target_phase_turns[9 * ring + (j + 1) % 9]
                    - geometry.target_phase_turns[9 * ring + j]
                    + Q(1, 2)
                )
                % 1
                - Q(1, 2)
            )
            for j in range(9)
        )
        for ring in (0, 1)
    )
    source_ok = (
        source.source_association_certified
        and source.complete_laws_certified
        and source.form_error_bounds == forms
        and source.phase_error_bounds == phases
        and source.registered_observation_error_bound
        == source.aligned_observation_error_bound
        == 0
        and source.gamma_bounds == gamma
        and _GL < gamma.lo <= gamma.hi < _G
        and source.nominal_q0_bounds == (qlo, qhi)
        and source.source_comparison_upper_bound == comparison
        and source.inherited_output_error_upper_bound == inherited_error
        and source.slow_amplitude_bounds[1] == amplitude_upper
        and source.centered_entry_norm_upper_bound == entry
        and source.initial_aggregate_upper_bound == entry_aggregate
        and source.aggregate_absolute_upper_bound == aggregate
        and geometry == reference.source_geometry == handoff.prefix.source_geometry
        and geometry.nodes == tuple(range(18))
        and geometry.degrees == (9,) * 18
        and geometry.held_capacities == (Q(1),) * 18
        and geometry.support_certified
        and geometry.reference_stationary_certified
        and geometry.spectral_gap_certified
        and geometry.bare_gap_lower_bound == 4
        and geometry.weighted_gap_lower_bound >= 3
        and reference.sources
        == (reference.donor.source, reference.recipient_readiness.sources[0])
        and all(
            _original_source_matches(row, geometry, rx, rt)
            for row, rx, rt in zip(reference.sources, forms, phases, strict=True)
        )
        and reference.original_supplied_phase_origins == (-_DELTA / 2, _DELTA / 2)
        and reference.original_relative_phase_gap == _DELTA
        and reference.actual_weight_deviation_upper_bound == _WEIGHT
        and reference.initial_form_means == handoff.retained_form_means == (Q(0), Q(0))
        and reference.recipient_isolated_deadline
        == reference.recipient_readiness.waiting_duration
        == 10**10
        and reference.inherited_donor_velocity_retained
        and persistent.uniform_comparison_upper_bound == comparison
        and persistent.original_phase_gauges_and_branches_retained
        and positive.exact_zero_entry_total_from_prior_laws
        and positive.prefix_full_duration == handoff.prefix_duration == 2000000
        and positive.prefix_cut_duration == handoff.continuation_duration == 10000
        and handoff.total_duration == _ACTIVATION
        and handoff.switch_word == (1, 0)
        and slow.same_actual_history_required
        and slow.correlated_initialization_retained
        and slow.early_age == _SWITCH
        and phase.phase_mean_error_upper_bound == 3 * _ETA
        and phase.relative_phase_entry_error_upper_bound == 6 * _ETA
        and phase.original_phase_lifts_retained
        and registration.complete_precontact_history_translated
        and registration.original_lifts_and_error_association_retained
        and registration.aligned_counterpart_retains_same_underlying_source
        and registration.registration_turns == Q(1, 4)
    )
    common = {
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
        "original_source_reacquired": False,
        "common_storage_reference": "same unshifted Theta in Uplus(Theta)",
    }
    laws = source_ok and all(
        getattr(row, name) == value
        for row in (source, marked, effective)
        for name, value in common.items()
    )
    laws = (
        laws
        and source.same_complete_and_effective_laws
        and source.original_clock_start_time == (_ACTIVATION + _SWITCH) * Q(1024, 1023)
        and source.actual_form_equation
        == effective.actual_form_equation
        == "Q_prime=-2*Q-18*gamma*sin(psi)-gamma*R"
        and source.actual_phase_equation
        == effective.actual_phase_equation
        == "psi_prime=2*gamma*Q/9"
        and source.effective_form_equation
        == effective.effective_form_equation
        == "Qhat_prime=-2*Qhat-18*gamma*sin(psihat)"
        and source.effective_phase_equation
        == effective.effective_phase_equation
        == "psihat_prime=2*gamma*Qhat/9"
        and source.initial_reduced_form
        == effective.initial_reduced_form
        == "same source-correlated Q_s(3375000)"
        and not source.actual_phase_supplied_to_predictor
        and not source.new_field_installed
        and not source.state_reset_performed
        and not effective.pressure_selected_from_preparation_label
        and source.clock
        == "tau=(1023/1024)*t; unchanged actual and independent effective rows"
        and marked.clock
        == "tau=(1023/1024)*t; actual rows and same-psi centered reference"
        and effective.clock
        == "tau=(1023/1024)*t; unchanged actual and effective interaction rows"
        and registration.clock
        == "tau=(1023/1024)*t; translated history and all reciprocal contact rows"
        and phase.clock
        == "tau=(1023/1024)*t; complete internal and phase-sensitive matching rows"
        and reference.clock
        == "tau=(1023/1024)*t; complete internal rows and reciprocal form inputs"
    )
    membership = (
        laws
        and source.prior_domain_certified
        and source.reference_cone_certified
        and source.incremental_comparison_certified
        and marked.prior_domain_certified
        and marked.forced_comparison_certified
        and marked.cycle_windings_certified
        and marked.normalized_metric_lower_bound == lower
        and marked.metric_root_upper_bound == root_bound
        and source.scaled_norm_squared_upper_bound
        == marked.scaled_norm_squared_upper_bound
        == square
        and marked.scaled_radius_margin == _RADIUS**2 - square > 0
        and marked.marked_gap_upper_bound == marked_gap
        and marked.registration_margin == _DELTA / 4 - marked_gap > 0
        and marked.marked_pairs == marked_pairs
        and len(set(marked_pairs)) == 18
        and marked.phase_sector_bound == signed.phase_sector_bound == sector
        and branch_margin > 0
        and windings == (Q(1), Q(2))
    )

    # Reuse the owned source estimates, rebuilding just the bound arithmetic
    # consumed by this observation transport. No new prediction is fitted.
    rate = _GL**2 / (375 * upper)
    iz = (
        qhi**2
        / ((Q(2, 9) * (1 - qhi**2 / 8) - Q(1, 1000)) * (1 - qhi**2 / 24))
        * (1 / (4 * _GL**2) + Q(1, 2))
    )
    ie = root_bound**2 / (4 * lower * rate)
    ir = Q(453, 200) * iz + 151 * ie
    delta = 6 * _ETA + 2 * _G * entry_aggregate / 9
    gain = 1 / (1 - 4 * _G**2 * delta)
    phase_error = gain * (6 * _ETA + 2 * _G * inherited_error / 9 + 2 * _G**2 * ir / 9)
    form_error = max(inherited_error, 9 * _G * phase_error + _G * square / 8)
    prediction = (
        membership
        and source.signed_sensitivity_certified
        and source.central_entry_certified
        and source.residual_integral_certified
        and source.decay_rate == rate
        and source.metric_root_upper_bound == root_bound
        and source.reference_phase_squared_integral_upper_bound == iz
        and source.comparison_phase_squared_integral_upper_bound == ie
        and source.moment_integral_upper_bound == ir
        and source.sensitivity_gain_upper_bound == gain
        and source.uniform_form_error_upper_bound == form_error < _TOLERANCE
        and source.uniform_phase_error_upper_bound == phase_error
        and source.intrinsic_form_tolerance == _TOLERANCE
        and source.original_source_error_paid_once
    )
    invariant_errors = (
        2 * aggregate * form_error,
        form_error + aggregate * phase_error,
        phase_error,
    )

    # Under a carried copy chart the output oQ is invariant. Dropping o
    # instead asks the same retained description to predict opposite signs.
    norm8 = entry / (1 - 4 * _G * _H)
    drift8 = 6 * _ETA + 2 * _G * _H * aggregate / 9
    deficit8 = 9 * drift8**2 + norm8**2
    band8 = (
        -9 * _G,
        -9 * _GL + (entry_aggregate + 9 * _GL) / 2**16 + _G * deficit8 / 2,
    )
    negative_band = (
        laws
        and registration.source_association_certified
        and registration.registration_covariance_certified
        and registration.centered_identity_certified
        and registration.relative_phase_drift_certified
        and registration.phase_moment_remainder_certified
        and registration.gamma_bounds == gamma
        and registration.registration_angle_bounds == pi / 2
        and registration.initial_relative_phase_error_upper_bound == 6 * _ETA
        and registration.full_contact_pairs == _PAIRS
        and registration.internal_divisor == 9
        and registration.post_duration == _H
        and registration.original_clock_horizon
        == (_ACTIVATION + _SWITCH + _H) * Q(1024, 1023)
        and registration.centered_entry_norm_upper_bound == entry
        and registration.entry_aggregate_upper_bound == entry_aggregate
        and registration.centered_norm_upper_bound == norm8
        and registration.relative_phase_deviation_upper_bound == drift8
        and registration.phase_moment_deficit_upper_bound == deficit8
        and registration.registered_true_bounds == band8
        and band8[1] < 0
    )
    obstruction_margin = -band8[1] - _TOLERANCE
    obstruction = negative_band and obstruction_margin > 0

    # The larger-class witness has the same Q and psi, but a nonzero hidden
    # moment. It is constructed algebraically and is not an acquired source.
    a = qstar = _DELTA / 10000
    offsets = (a, a, -2 * a) + (Q(0),) * 15
    witness_phase = (Q(0),) * 18 + offsets
    witness_square = 4 * sum(value**2 for value in witness_phase)
    witness_moment = 2 * sin(a) * (1 - cos(a))
    witness_q_rate = -gamma * witness_moment
    witness_u_rate = -2 * gamma * qstar * witness_moment
    witness = (
        laws
        and a > 0
        and sum(offsets, Q(0)) == 0
        and witness_square == 24 * a**2 < _RADIUS**2
        and 3 * a < _DELTA / 4
        and (4 * pi / 9 + 3 * a).hi < pi.lo
        and witness_moment.lo > 0
        and witness_q_rate.hi < 0
        and witness_u_rate.hi < 0
    )
    # Complete source/history transport makes these algebraic equivariances
    # applicable. It does not assert an automorphism of fixed unequal weights.
    invariant = laws and branch_margin > 0
    port = invariant and source.whole_form_mean == 0
    resultant = laws and all(
        tuple(
            sorted(
                int((9 * turn) % 9)
                for turn in geometry.target_phase_turns[9 * r : 9 * r + 9]
            )
        )
        == tuple(range(9))
        for r in (0, 1)
    )
    phase_work = 972 * _ETA + Q(9, 2) * entry**3
    aligned_jump = (Q(0), 9 * (entry + 18 * _ETA) ** 2)
    registered_jump = (162 - phase_work, 162 + phase_work + 9 * entry**2)
    work = (
        laws
        and source.switching_work_certified
        and marked.switching_work_certified
        and source.original_attachment_work_retained_once
        and source.phase_switching_work_allowance == phase_work
        and source.aligned_storage_jump_bounds == aligned_jump
        and source.registered_storage_jump_bounds == registered_jump
        and not marked.direct_entry_energy_adds_attachment_twice
    )
    admitted = max(registered_noise, aligned_noise) <= _TOLERANCE
    certified = (
        membership
        and prediction
        and invariant
        and port
        and obstruction
        and witness
        and resultant
        and work
        and admitted
    )
    unresolved = tuple(
        label
        for good, label in (
            (source_ok, "original_source_association_unavailable"),
            (laws, "complete_law_and_history_transport_unavailable"),
            (membership, "original_source_maintained_membership_unavailable"),
            (prediction, "original_source_prediction_unavailable"),
            (invariant, "declared_invariant_representation_domain_unavailable"),
            (port, "oriented_port_descent_unavailable"),
            (obstruction, "inherited_port_association_discriminator_unavailable"),
            (witness, "larger_class_exact_closure_witness_unavailable"),
            (resultant, "primitive_phase_resultant_boundary_unavailable"),
            (work, "original_attachment_work_association_unavailable"),
            (admitted, "original_reading_allowances_outside_contract"),
        )
        if not good
    )
    return _MixedPairIdentity(
        form_error_bounds=forms,
        phase_error_bounds=phases,
        registered_observation_error_bound=registered_noise,
        aligned_observation_error_bound=aligned_noise,
        source=source,
        source_association_certified=source_ok,
        source_maintained_membership_certified=membership,
        complete_law_transport_certified=laws,
        common_origin_invariance_certified=laws,
        transported_relabeling_covariance_certified=laws,
        constituent_exchange_covariance_certified=laws,
        invariant_chart_certified=invariant,
        oriented_port_descent_certified=port,
        source_prediction_certified=prediction,
        inherited_negative_band_certified=negative_band,
        port_association_obstruction_certified=obstruction,
        larger_class_exact_nonclosure_certified=witness,
        phase_resultant_boundary_certified=resultant,
        switching_work_certified=work,
        observation_allowances_admitted=admitted,
        gamma_bounds=gamma,
        nominal_q0_bounds=(qlo, qhi),
        source_comparison_upper_bound=comparison,
        centered_entry_norm_upper_bound=entry,
        scaled_norm_squared_upper_bound=square,
        scaled_radius_margin=_RADIUS**2 - square,
        marked_gap_upper_bound=marked_gap,
        registration_margin=_DELTA / 4 - marked_gap,
        phase_sector_bound=sector,
        phase_lift_margin=branch_margin,
        marked_pairs=marked_pairs,
        transported_windings=tuple(int(v) for v in windings * 2),
        aggregate_absolute_upper_bound=aggregate,
        uniform_form_error_upper_bound=form_error,
        uniform_phase_error_upper_bound=phase_error,
        invariant_prediction_error_bounds=invariant_errors,
        registered_form_record_error_upper_bound=form_error + registered_noise,
        aligned_form_record_error_upper_bound=form_error + aligned_noise,
        inherited_registered_true_bounds=band8,
        port_blind_error_lower_bound=-band8[1],
        port_association_obstruction_margin=obstruction_margin,
        phase_switching_work_allowance=phase_work,
        aligned_storage_jump_bounds=aligned_jump,
        registered_storage_jump_bounds=registered_jump,
        witness_qstar=qstar,
        witness_offset_amplitude=a,
        witness_matching_offsets=offsets,
        witness_centered_phase=witness_phase,
        witness_scaled_norm_squared=witness_square,
        witness_marked_gap_upper_bound=3 * a,
        witness_moment_difference_bounds=witness_moment,
        witness_form_rate_difference_bounds=witness_q_rate,
        witness_invariant_u_rate_difference_bounds=witness_u_rate,
        certified=certified,
        status=(
            "conditional_marked_pair_and_oriented_prediction"
            if certified
            else "unavailable"
        ),
        unresolved_conditions=unresolved,
        source_membership_retained_for_all_times=membership,
    )
