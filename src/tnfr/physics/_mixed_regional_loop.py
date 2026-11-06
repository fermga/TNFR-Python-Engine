"""Regional contact coefficients of the retained complete interaction law.

The detached observation keeps coefficient magnitudes and phase information.
Its source theorem bounds unknown actual profiles without acquiring a state,
evolving a predictor, or identifying an independent link field.
"""

from dataclasses import dataclass
from fractions import Fraction as Q

from ..mathematics._rational_interval import I, arg, cos, pi_interval, sin, sqrt
from ._mixed_marked_identity import _assess_mixed_marked_identity, _MixedMarkedIdentity
from ._mixed_reciprocal_partner import _nonnegative_pair
from ._mixed_registered_phase_probe import _derive_registered_phase_probe_geometry
from ._mixed_resource_transfer_obstruction import _original_source_matches
from .relational_observations import _ordered

_DELTA, _ETA, _WEIGHT = Q(1, 1024), Q(2, 10**69), Q(2, 10**60)
_GL, _G, _MATURE = Q(1, 3300), Q(1, 3000), Q(1, 2**28)
_ACTIVATION, _SWITCH = Q(2010000), Q(3375000)
_REGIONS = tuple(tuple(range(start, start + 9)) for start in (0, 9, 18, 27))
_CYCLE = ((0, 1), (1, 3), (3, 2), (2, 0))
_PAIRS = tuple((j, 18 + j) for j in range(18))


def _contact_geometry(registered):
    """Select actual phase-potential edges, independently of form weights."""
    geometry = registered.successor
    cross = tuple((a, b, Q(1)) for a, b in geometry.edge_pairs if a < 9 <= b)
    edges = (
        cross,
        tuple((j, j + 18, Q(9)) for j in range(9, 18)),
        tuple((b + 18, a + 18, weight) for a, b, weight in cross),
        tuple((j + 18, j, Q(9)) for j in range(9)),
    )
    expected = tuple(
        tuple(int((j - i) % 9 in (0, 1, 2, 7, 8)) for j in range(9)) for i in range(9)
    )
    admitted = (
        registered.cross_symmetry_certified
        and registered.cross_incidence == expected
        and tuple(a for a, _ in geometry.marked_pairs) == tuple(range(9))
        and tuple(sorted(registered.marked_partner_indices)) == tuple(range(9, 18))
        and registered.marked_partner_indices
        == tuple(b for _, b in geometry.marked_pairs)
        and registered.cross_incidence
        == tuple(
            tuple(
                int((a, b) in geometry.edge_pairs)
                for b in registered.marked_partner_indices
            )
            for a in range(9)
        )
        and geometry.nodes == tuple(range(18))
        and geometry.degrees == (9,) * 18
        and geometry.held_capacities == (Q(1),) * 18
        and geometry.support_certified
        and geometry.reference_stationary_certified
        and geometry.spectral_gap_certified
        and geometry.bare_gap_lower_bound == 4
        and geometry.weighted_gap_lower_bound >= 3
        and len(geometry.edge_pairs) == len(set(geometry.edge_pairs)) == 81
        and len(cross) == 45
        and len(geometry.target_phase_turns) == 18
        and geometry.target_phase_turns
        == tuple(Q((1 + j // 9) * (j % 9 - 4), 9) for j in range(18))
        and geometry.phase_indices
        == tuple(int((9 * turn) % 9) for turn in geometry.target_phase_turns)
        and registered.marked_branch_turns == geometry.marked_branch_turns
        and all(
            geometry.target_phase_turns[b] - geometry.target_phase_turns[a] == branch
            and branch.denominator == 1
            for (a, b), branch in zip(
                geometry.marked_pairs, geometry.marked_branch_turns, strict=True
            )
        )
        and all(
            sum(geometry.target_phase_turns[j] for j in region) == 0
            for region in _REGIONS[:2]
        )
    )
    return edges, admitted


def _complex_product(left, right):
    return (
        left[0] * right[0] - left[1] * right[1],
        left[0] * right[1] + left[1] * right[0],
    )


@dataclass(frozen=True)
class _RegionalContactCoefficients:
    """An interval observation of supplied profiles, without state provenance.

    Failure to certify nonzero means that normalization is unavailable; it
    does not prove a zero coefficient. A principal argument has an additional
    branch condition, distinct from the existence of the unit loop.
    """

    phase_profiles: tuple[I, ...]
    coefficients: tuple[tuple[I, I], ...]
    squared_moduli: tuple[I, ...]
    magnitudes: tuple[I, ...]
    nonzero_certified: tuple[bool, ...]
    unit_coefficients: tuple[tuple[I, I], ...] | None
    raw_loop_product: tuple[I, I]
    loop_product: tuple[I, I] | None
    loop_argument: I | None
    argument_available: bool
    region_nodes: tuple[tuple[int, ...], ...] = _REGIONS
    cycle: tuple[tuple[int, int], ...] = _CYCLE
    coefficient_units: str = "phase-potential coefficient; internal1, matching9"
    supplied_profiles_authenticated_as_source: bool = False
    zero_coefficient_inferred_from_unavailable_chart: bool = False
    independent_link_field_installed: bool = False


def _mixed_regional_contact_coefficients(phase_profiles):
    """Observe four contacts from 36 regional-frame lifts in physical node order.

    A regional profile contains its carried winding reference plus its local
    offsets. Mean angles are separate: the force is the imaginary part after
    multiplying this coefficient by exp(i*(phi_target-phi_source)). No zero
    mean or source provenance is imposed on arbitrary supplied profiles.
    """
    raw = _ordered(phase_profiles, "phase_profiles", limit=37)
    if len(raw) != 36:
        raise ValueError("phase_profiles requires all 36 regional-frame lifts")
    profiles = tuple(I.coerce(value) for value in raw)
    edges, admitted = _contact_geometry(_derive_registered_phase_probe_geometry())
    if not admitted:
        raise ValueError("the carried regional phase support is unavailable")
    coefficients = tuple(
        (
            sum(
                (weight * cos(profiles[b] - profiles[a]) for a, b, weight in row), I(0)
            ),
            sum(
                (weight * sin(profiles[b] - profiles[a]) for a, b, weight in row), I(0)
            ),
        )
        for row in edges
    )
    squares = tuple(real**2 + imaginary**2 for real, imaginary in coefficients)
    magnitudes = tuple(sqrt(value) for value in squares)
    nonzero = tuple(value.lo > 0 for value in magnitudes)
    raw_product = (I(1), I(0))
    for value in coefficients:
        raw_product = _complex_product(raw_product, value)
    units = loop = angle = None
    if all(nonzero):
        units = tuple(
            (real / magnitude, imaginary / magnitude)
            for (real, imaginary), magnitude in zip(
                coefficients, magnitudes, strict=True
            )
        )
        loop = (I(1), I(0))
        for value in units:
            loop = _complex_product(loop, value)
        try:
            angle = arg(*loop)
        except ValueError:
            pass
    return _RegionalContactCoefficients(
        phase_profiles=profiles,
        coefficients=coefficients,
        squared_moduli=squares,
        magnitudes=magnitudes,
        nonzero_certified=nonzero,
        unit_coefficients=units,
        raw_loop_product=raw_product,
        loop_product=loop,
        loop_argument=angle,
        argument_available=angle is not None,
    )


@dataclass(frozen=True)
class _MixedRegionalLoop:
    """Source-conditioned bounds and a distinct constructed information loss."""

    form_error_bounds: tuple[Q, Q]
    phase_error_bounds: tuple[Q, Q]
    source: _MixedMarkedIdentity
    source_association_certified: bool
    complete_law_certified: bool
    regional_geometry_certified: bool
    source_comparison_certified: bool
    nominal_factorization_certified: bool
    source_nonzero_certified: bool
    source_loop_bound_certified: bool
    same_state_response_bound_certified: bool
    witness_membership_certified: bool
    matched_magnitude_obstruction_certified: bool
    switching_work_certified: bool
    gamma_bounds: I
    nominal_internal_sum: I
    nominal_internal_coefficient: I
    nominal_matching_coefficient: Q
    source_comparison_upper_bound: Q
    metric_root_upper_bound: Q
    scaled_error_upper_bound: Q
    comparison_phase_error_upper_bound: Q
    hidden_profile_squared_upper_bound: Q
    internal_coefficient_defect_upper_bound: Q
    matching_coefficient_defect_upper_bound: Q
    internal_coefficient_modulus_lower_bound: Q
    matching_coefficient_modulus_lower_bound: Q
    loop_deviation_upper_bound: Q
    regional_mean_rate_error_upper_bound: Q
    marked_gap_upper_bound: Q
    registration_margin: Q
    phase_switching_work_allowance: Q
    aligned_storage_jump_bounds: tuple[Q, Q]
    registered_storage_jump_bounds: tuple[Q, Q]
    witness_amplitude: Q
    witness_scaled_norm_squared: Q
    witness_marked_perturbation_upper_bound: Q
    witness_imaginary_coefficient_bounds: I
    witness_matching_coefficient_bounds: I
    witness_positive_regional_mean_rate_bounds: I
    witness_regional_mean_rate_separation_bounds: I
    certified: bool
    status: str
    unresolved_conditions: tuple[str, ...]
    region_nodes: tuple[tuple[int, ...], ...] = _REGIONS
    cycle: tuple[tuple[int, int], ...] = _CYCLE
    full_state_dimension: int = 72
    physical_capacities: tuple[Q, ...] = (Q(1),) * 36
    full_contact_pairs: tuple[tuple[int, int], ...] = _PAIRS
    internal_divisor: Q = Q(9)
    activation_time: Q = _ACTIVATION
    switch_age: Q = _SWITCH
    switch_time: Q = _ACTIVATION + _SWITCH
    initial_registrations_turns: tuple[Q, Q] = (Q(0), Q(1, 4))
    response_units: str = "regional mean form per structural tau"
    loop_units: str = "dimensionless"
    response_comparator: str = "same state and magnitudes; zero contact phases"
    coefficient_frame: str = "carried winding reference plus region-zero-mean offsets"
    source_bound_for_all_postattachment_times: bool = False
    source_actual_loop_nontrivial_certified: bool = False
    source_exact_factorization_certified: bool = False
    regional_frame_change_is_physical_intervention: bool = False
    coefficient_magnitudes_discarded: bool = False
    held_form_weights_used_as_phase_coefficients: bool = False
    original_attachment_work_retained_once: bool = True
    original_source_error_paid_once: bool = True
    same_actual_phase_reference_is_forecast: bool = False
    comparison_is_autonomous_predictor: bool = False
    rate_bound_integrated_as_trajectory_error: bool = False
    original_form_tolerance_reassigned: bool = False
    witness_is_acquired_source: bool = False
    witness_discriminates_whole_copy_transfer: bool = False
    witness_refutes_existing_aggregate_prediction: bool = False
    witness_regional_mean_forms: tuple[Q, ...] = (Q(0),) * 4
    witness_regional_mean_phases: tuple[Q, ...] = (Q(0),) * 4
    witness_whole_copy_transfer_rate: Q = Q(0)
    new_field_installed: bool = False
    new_observation_acquired: bool = False
    source_trajectory_evaluated: bool = False
    original_source_replayed: bool = False
    state_reset_performed: bool = False
    independent_link_field_identified: bool = False
    electromagnetic_sector_identified: bool = False
    empirical_validation: bool = False
    clock: str = "tau=(1023/1024)*t; unchanged complete regional interaction rows"


def _assess_mixed_regional_loop(*, form_error_bounds, phase_error_bounds):
    """Re-admit the owned source comparison before bounding contact information."""
    forms = _nonnegative_pair(form_error_bounds, "form_error_bounds")
    phases = _nonnegative_pair(phase_error_bounds, "phase_error_bounds")
    source = _assess_mixed_marked_identity(
        form_error_bounds=forms,
        phase_error_bounds=phases,
        registered_observation_error_bound=Q(0),
        aligned_observation_error_bound=Q(0),
    )
    signed = source.source
    effective = signed.source.source
    registration = effective.source
    phase = registration.source
    slow = phase.source.source
    persistent = slow.source.source
    positive = persistent.source.source
    handoff = positive.handoff
    reference = handoff.prefix.reference
    registered = _derive_registered_phase_probe_geometry()
    geometry = registered.successor
    _, geometry_ok = _contact_geometry(registered)
    pi = pi_interval()
    gamma = 1 / (1023 * pi)
    coupling = Q(13, 27) - _DELTA / (18 * pi)
    section_error = Q(3, 2) * (14 * _ETA + 2 * 10**10 * _WEIGHT)
    qlo, qhi = 3 * _DELTA / 16 - section_error / 2, 3 * _DELTA / 16 + section_error / 2
    comparison = 3 * (24 * _MATURE + 84 * _WEIGHT * 10 * qhi / _GL) / (1 - 135 * qhi)
    lower, upper = 1 - 2 * _G, 1 + 2 * _G
    root = 2 * (1 + _G) * comparison + 14 * _WEIGHT * qhi / (3 * _GL)
    difference = root / lower
    hidden_square = difference**2 / 4
    marked_gap = qhi + 3 * difference / 4
    source_ok = (
        source.source_association_certified
        and source.form_error_bounds == forms
        and source.phase_error_bounds == phases
        and source.registered_observation_error_bound
        == source.aligned_observation_error_bound
        == 0
        and max(forms + phases) <= Q(1, 10**10)
        and source.gamma_bounds == gamma
        and _GL < gamma.lo <= gamma.hi < _G
        and source.coupling_bounds == coupling
        and source.nominal_q0_bounds == (qlo, qhi)
        and source.source_comparison_upper_bound
        == persistent.uniform_comparison_upper_bound
        == comparison
        and geometry == reference.source_geometry == handoff.prefix.source_geometry
        and reference.sources
        == (reference.donor.source, reference.recipient_readiness.sources[0])
        and all(
            _original_source_matches(row, geometry, rx, rt)
            for row, rx, rt in zip(reference.sources, forms, phases, strict=True)
        )
        and reference.original_supplied_phase_origins == (-_DELTA / 2, _DELTA / 2)
        and reference.original_relative_phase_gap == _DELTA
        and reference.actual_weight_deviation_upper_bound == _WEIGHT
        and reference.recipient_isolated_deadline
        == reference.recipient_readiness.waiting_duration
        == 10**10
        and reference.initial_form_means == handoff.retained_form_means == (Q(0), Q(0))
        and reference.inherited_donor_velocity_retained
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
    )
    common = {
        "activation_time": _ACTIVATION,
        "switch_age": _SWITCH,
        "switch_time": _ACTIVATION + _SWITCH,
        "initial_registrations_turns": (Q(0), Q(1, 4)),
        "full_state_dimension": 72,
        "physical_capacities": (Q(1),) * 36,
        "full_contact_pairs": _PAIRS,
        "internal_divisor": Q(9),
        "whole_form_mean": Q(0),
        "original_donor_form_total": Q(0),
        "source_correlated_initialization_retained": True,
        "whole_history_covariance_retained": True,
        "original_attachment_state_retained": True,
        "original_attachment_work_retained_once": True,
        "common_storage_reference": "same unshifted Theta in Uplus(Theta)",
        "state_reset_performed": False,
        "original_source_reacquired": False,
    }
    laws = (
        source_ok
        and geometry_ok
        and all(
            getattr(row, name) == value
            for row in (source, signed)
            for name, value in common.items()
        )
        and source.original_clock_start_time == (_ACTIVATION + _SWITCH) * Q(1024, 1023)
        and source.clock
        == "tau=(1023/1024)*t; actual rows and same-psi centered reference"
        and phase.clock
        == "tau=(1023/1024)*t; complete internal and phase-sensitive matching rows"
        and reference.clock
        == "tau=(1023/1024)*t; complete internal rows and reciprocal form inputs"
        and source.same_complete_and_effective_laws
        and effective.original_complete_field_unchanged
        and effective.full_contact_pairs == _PAIRS
        and effective.internal_divisor == 9
        and effective.physical_capacities == (Q(1),) * 36
        and effective.whole_history_covariance_retained
        and effective.source_correlated_initialization_retained
        and source.complete_storage_rate == "-norm((H+9G)*x)**2/9"
        and source.original_marked_branches_retained
        and not source.phase_clamped
        and not source.prior_finite_endpoint_used_as_initial_state
        and not source.independently_fixed_copy_gauges_assumed
    )
    comparison_ok = (
        laws
        and source.prior_domain_certified
        and source.source_cone_certified
        and source.reference_cone_certified
        and source.reference_integrals_certified
        and source.incremental_stiffness_certified
        and source.common_metric_certified
        and source.forced_comparison_certified
        and source.cycle_windings_certified
        and source.normalized_metric_lower_bound == lower > 0
        and source.normalized_metric_upper_bound == upper
        and source.initial_scaled_error_upper_bound == 2 * comparison
        and source.initial_metric_root_upper_bound == 2 * (1 + _G) * comparison
        and source.weight_matrix_deviation_upper_bound == 14 * _WEIGHT
        and source.forcing_integral_upper_bound == 14 * _WEIGHT * qhi / (3 * _GL)
        and source.metric_root_upper_bound == root
        and source.scaled_error_upper_bound == difference
        and source.phase_error_upper_bound == difference / 2
        and source.reference_form_integral_upper_bound == 3 * qhi / _GL
        and source.reference_relative_phase
        == "same actual continuous psi(t), analytically"
        and source.reference_initialization
        == "original nominal centered source at A age3375000"
        and source.reference_position_upper_bound_factor == 1
        and source.reference_velocity_upper_bound_factor == 1
        and source.marked_gap_upper_bound == marked_gap
        and source.registration_margin == _DELTA / 4 - marked_gap > 0
        and source.marked_pairs
        == tuple(
            (a + offset, b + offset)
            for offset in (0, 18)
            for a, b in geometry.marked_pairs
        )
        and source.source_error_paid_once
        and not source.mean_error_forcing_discarded
        and not source.reference_velocity_reset
        and not source.reference_is_autonomous_predictor
        and not source.phase_readout_input_required
    )
    c0 = 1 + 2 * cos(2 * pi / 9) + 2 * cos(4 * pi / 9)
    nominal = geometry_ok and c0.lo > 2
    internal_lower, matching_lower = 18 - 5 * hidden_square, 81 - 9 * hidden_square
    nonzero = comparison_ok and nominal and min(internal_lower, matching_lower) > 0
    loop_bound = 7 * hidden_square / 9
    rate_bound = 28 * _G * hidden_square / 81
    entry = 4 * qhi + 200 * _MATURE
    work_allowance = 972 * _ETA + Q(9, 2) * entry**3
    aligned_work = (Q(0), 9 * (entry + 18 * _ETA) ** 2)
    registered_work = (162 - work_allowance, 162 + work_allowance + 9 * entry**2)
    work = (
        laws
        and source.switching_work_certified
        and source.phase_switching_work_allowance == work_allowance
        and source.aligned_storage_jump_bounds == aligned_work
        and source.registered_storage_jump_bounds == registered_work
        and not source.direct_entry_energy_adds_attachment_twice
    )
    # These are fixed constructed profiles, not a newly observed source member.
    amplitude = _DELTA / 10000
    omega = 2 * pi / 9
    d = 2 * (1 - cos(amplitude))
    imaginary = 2 * d * sin(omega) * (1 - cos(omega - amplitude))
    witness_rate = gamma * imaginary / 81
    witness_member = (
        geometry_ok
        and 16 * amplitude**2 < Q(1, 500) ** 2
        and 2 * amplitude < _DELTA / 4
        and 4 * pi.hi / 9 + 2 * amplitude < pi.lo / 2
    )
    witness = nominal and witness_member and imaginary.lo > 0 and witness_rate.lo > 0
    certified = comparison_ok and nonzero and work and witness
    unresolved = tuple(
        label
        for valid, label in (
            (source_ok, "original_source_association_unavailable"),
            (laws, "complete_regional_law_unavailable"),
            (comparison_ok, "source_hidden_profile_comparison_unavailable"),
            (nonzero, "source_contact_normalization_unavailable"),
            (work, "retained_attachment_work_unavailable"),
            (witness, "constructed_information_loss_unavailable"),
        )
        if not valid
    )
    return _MixedRegionalLoop(
        form_error_bounds=forms,
        phase_error_bounds=phases,
        source=source,
        source_association_certified=source_ok,
        complete_law_certified=laws,
        regional_geometry_certified=geometry_ok,
        source_comparison_certified=comparison_ok,
        nominal_factorization_certified=nominal,
        source_nonzero_certified=nonzero,
        source_loop_bound_certified=nonzero,
        same_state_response_bound_certified=nonzero,
        witness_membership_certified=witness_member,
        matched_magnitude_obstruction_certified=witness,
        switching_work_certified=work,
        gamma_bounds=gamma,
        nominal_internal_sum=c0,
        nominal_internal_coefficient=9 * c0,
        nominal_matching_coefficient=Q(81),
        source_comparison_upper_bound=comparison,
        metric_root_upper_bound=root,
        scaled_error_upper_bound=difference,
        comparison_phase_error_upper_bound=difference / 2,
        hidden_profile_squared_upper_bound=hidden_square,
        internal_coefficient_defect_upper_bound=5 * hidden_square,
        matching_coefficient_defect_upper_bound=9 * hidden_square,
        internal_coefficient_modulus_lower_bound=internal_lower,
        matching_coefficient_modulus_lower_bound=matching_lower,
        loop_deviation_upper_bound=loop_bound,
        regional_mean_rate_error_upper_bound=rate_bound,
        marked_gap_upper_bound=marked_gap,
        registration_margin=_DELTA / 4 - marked_gap,
        phase_switching_work_allowance=work_allowance,
        aligned_storage_jump_bounds=aligned_work,
        registered_storage_jump_bounds=registered_work,
        witness_amplitude=amplitude,
        witness_scaled_norm_squared=16 * amplitude**2,
        witness_marked_perturbation_upper_bound=2 * amplitude,
        witness_imaginary_coefficient_bounds=imaginary,
        witness_matching_coefficient_bounds=9 * (7 + 2 * cos(amplitude)),
        witness_positive_regional_mean_rate_bounds=witness_rate,
        witness_regional_mean_rate_separation_bounds=2 * witness_rate,
        source_bound_for_all_postattachment_times=nonzero,
        certified=certified,
        status=(
            "conditional_regional_contact_information" if certified else "unavailable"
        ),
        unresolved_conditions=unresolved,
    )
