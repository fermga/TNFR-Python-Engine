"""A conditional resolution limit for phase-only probes retaining registration.

The bound holds on any finite window where the full law's original marked
gaps remain inside the stated collar. It neither certifies that future
condition nor admits a phase intervention. No new field or solver is needed.
"""

from dataclasses import dataclass
from fractions import Fraction as Q

from ._mixed_reciprocal_partner import (
    _assess_mixed_reciprocal_partner,
    _MixedReciprocalPartner,
    _nonnegative,
    _nonnegative_pair,
)
from ._sine_coupling_successor import (
    _CouplingSuccessorGeometry,
    _derive_coupling_successor_geometry,
)
from .support_transport import _support_components

_DELTA = Q(1, 1024)
_REGISTRATION = _DELTA / 4
_GAMMA_UPPER = Q(1, 3000)
_WEIGHT_DEVIATION = Q(20 * 10**9, 10**70)
_SOURCE_CAP = Q(1, 10**10)
_READOUT_CAP = Q(1, 10**6)
_GAP = Q(1, 7350)
_DIFFERENCE_ENVELOPE = Q(15)
_COUPLING_LOWER = Q(4, 9)
_CONTACT_PAIRS = tuple((j, 18 + j) for j in range(9))


@dataclass(frozen=True)
class _RegisteredPhaseProbeGeometry:
    """Fresh bare cross geometry and lower conductances of the form generator.

    The combined edges describe a lower bounding Laplacian for the paired
    form difference. The nine exchange pairs are algebraic input terms,
    not additions to either copy's internal neighbor or phase law.
    """

    successor: _CouplingSuccessorGeometry
    marked_partner_indices: tuple[int, ...]
    marked_branch_turns: tuple[Q, ...]
    cross_incidence: tuple[tuple[int, ...], ...]
    combined_weighted_edges: tuple[tuple[int, int, Q], ...]
    combined_components: tuple[tuple[int, ...], ...]
    minimum_weight_lower_bound: Q
    spanning_tree_gap_lower_bound: Q
    cross_symmetry_certified: bool
    connected_support_certified: bool
    certified: bool


def _derive_registered_phase_probe_geometry():
    """Rebuild the marked cross cancellation and the elementary connected gap."""
    source = _derive_coupling_successor_geometry(completion="common_sense")
    pairs = tuple(source.edge_pairs)
    shape = (
        source.support_certified
        and source.nodes == tuple(range(18))
        and source.degree == 9
        and source.degrees == (9,) * 18
        and source.held_capacities == (Q(1),) * 18
        and len(source.internal_edges) == 18
        and all(w == 1 for _, _, w in source.internal_edges)
        and len(source.new_edge_pairs) == 63
        and len(pairs) == len(set(pairs)) == 81
        and all(0 <= a < b < 18 for a, b in pairs)
        and set(pairs)
        == {(a, b) for a, b, _ in source.internal_edges} | set(source.new_edge_pairs)
        and len(source.target_phase_turns) == 18
        and len(source.marked_pairs) == len(source.marked_branch_turns) == 9
        and tuple(a for a, _ in source.marked_pairs) == tuple(range(9))
        and tuple(sorted(b for _, b in source.marked_pairs)) == tuple(range(9, 18))
    )
    partners = tuple(b for _, b in source.marked_pairs) if shape else ()
    branches = (
        tuple(
            source.target_phase_turns[b] - source.target_phase_turns[a]
            for a, b in source.marked_pairs
        )
        if shape
        else ()
    )
    cross = (
        tuple(tuple(int((i, partners[j]) in pairs) for j in range(9)) for i in range(9))
        if shape
        else ()
    )
    expected = tuple(
        tuple(int((j - i) % 9 in (0, 1, 2, 7, 8)) for j in range(9)) for i in range(9)
    )
    symmetric = (
        shape
        and branches == source.marked_branch_turns
        and all(turn.denominator == 1 for turn in branches)
        and cross == expected
        and all(sum(row) == 5 for row in cross)
        and all(cross[i][j] == cross[j][i] for i in range(9) for j in range(9))
    )
    internal_lower = tuple((a, b, w / 9) for a, b, w in source.internal_edges)
    internal_lower += tuple((a, b, Q(1, 12)) for a, b in source.new_edge_pairs)
    combined = (
        tuple(
            (18 * copy + a, 18 * copy + b, weight)
            for copy in range(2)
            for a, b, weight in internal_lower
        )
        + tuple((a, b, Q(1)) for a, b in _CONTACT_PAIRS)
        if shape
        else ()
    )
    rows = tuple(
        tuple(b if i == a else a for a, b, _ in combined if i in (a, b))
        for i in range(36)
    )
    components = _support_components(rows)
    minimum = min((w for _, _, w in combined), default=Q(0))
    # For n=36, zero-mean variance and paths of at most n-1 tree edges
    # give ||v||^2 <= (n-1)^2/(2*w_min) * v^T D v.
    gap = 2 * minimum / 35**2
    connected = (
        shape
        and len(combined) == 171
        and components == (tuple(range(36)),)
        and minimum >= Q(1, 12)
        and gap >= _GAP
    )
    return _RegisteredPhaseProbeGeometry(
        successor=source,
        marked_partner_indices=partners,
        marked_branch_turns=branches,
        cross_incidence=cross,
        combined_weighted_edges=combined,
        combined_components=components,
        minimum_weight_lower_bound=minimum,
        spanning_tree_gap_lower_bound=gap,
        cross_symmetry_certified=symmetric,
        connected_support_certified=connected,
        certified=symmetric and connected,
    )


@dataclass(frozen=True)
class _MixedRegisteredPhaseProbe:
    """A theorem conditional on future registration, not a control admission.

    Every finite window satisfying the stated full-law marked-gap condition
    has the true paired transfer bound below. No horizon, endpoint, phase
    command or cached source report is accepted. A certified common possible
    record is a worst-case observation obstruction, not equality of responses.
    Candidate bounds require their source and geometry flags. Failure of the
    sufficient overlap condition never certifies distinguishability.
    """

    form_error_bounds: tuple[Q, Q]
    phase_error_bounds: tuple[Q, Q]
    starting_gap_upper_bounds: tuple[Q, Q]
    readout_error_bound: Q
    reference: _MixedReciprocalPartner
    geometry: _RegisteredPhaseProbeGeometry
    source_certified: tuple[bool, bool]
    finite_protocol_ready_ages_exist: tuple[bool, bool]
    conditional_ready_sources: tuple[bool, bool]
    source_association_certified: bool
    cross_current_upper_bound: Q
    regional_sine_force_upper_bound: Q
    full_sine_force_norm_upper_bound: Q
    form_difference_upper_bound: Q
    form_difference_envelope: Q
    weight_residual_upper_bound: Q
    regional_residual_upper_bound: Q
    nominal_coupling_lower_bound: Q
    nominal_coupling_lower_margin: Q
    output_kernel_mass_upper_bound: Q
    transfer_difference_upper_bound: Q
    difference_bound_certified: bool
    readout_allowance_admitted: bool
    overlap_margin: Q
    shared_record_overlap_certified: bool
    certified: bool
    status: str
    unresolved_conditions: tuple[str, ...]
    registration_gap_bound: Q = _REGISTRATION
    gamma_upper_bound: Q = _GAMMA_UPPER
    edge_weight_deviation_upper_bound: Q = _WEIGHT_DEVIATION
    source_error_cap: Q = _SOURCE_CAP
    readout_error_cap: Q = _READOUT_CAP
    existence_gap_threshold: Q = _DELTA / 8
    full_state_dimension: int = 72
    contact_pairs: tuple[tuple[int, int], ...] = _CONTACT_PAIRS
    all_window_registration_is_external_condition: bool = True
    all_window_registration_certified: bool = False
    identity_retention_certified: bool = False
    intervention_admitted: bool = False
    matched_initial_state_assumed: bool = True
    comparator_own_law_source_acquisition_claimed: bool = False
    duration_scope: str = (
        "every finite observation window satisfying the registration condition"
    )
    clock: str = "tau=(1023/1024)*t; original-time input rates divide by1023/1024"
    output_orientation: str = "outgoing_from_first_copy_to_second"
    continuous_phase_work: str = "-S(theta) dot c; regional work=-(c_A-c_B)*sum_A S"
    phase_jump_work: str = (
        "U(theta+z)-U(theta), with bare sine potential and held support"
    )
    scope: tuple[str, ...] = (
        "same_actual_prepared_entry_state_and_fixed_reciprocal_form_contact",
        "phase_only_inputs_and_jumps_leave_form_support_weights_and_capacities_unchanged",
        "condition_on_full_law_marked_registration_not_on_an_equitable_fine_carrier",
        "actual_weight_defect_acts_on_paired_form_difference_not_old_short_time_tube",
        "uniform_true_output_bound_without_a_selected_horizon_or_actuation",
        "no_future_identity_certificate_control_admission_or_recovery_claim",
        "no_equal_outputs_statistical_equivalence_physical_law_or_outer_NFR_claim",
    )


def _assess_mixed_registered_phase_probe(
    *,
    form_error_bounds,
    phase_error_bounds,
    starting_gap_upper_bounds,
    readout_error_bound,
):
    """Re-admit the actual source before the conditional phase-probe obstruction."""
    forms = _nonnegative_pair(form_error_bounds, "form_error_bounds")
    phases = _nonnegative_pair(phase_error_bounds, "phase_error_bounds")
    gaps = _nonnegative_pair(starting_gap_upper_bounds, "starting_gap_upper_bounds")
    readout = _nonnegative(readout_error_bound, "readout_error_bound")
    reference = _assess_mixed_reciprocal_partner(
        form_error_bounds=forms,
        phase_error_bounds=phases,
        starting_gap_upper_bounds=gaps,
        readout_error_bound=Q(0),
    )
    geometry = _derive_registered_phase_probe_geometry()
    association = (
        reference.form_error_bounds == forms
        and reference.phase_error_bounds == phases
        and reference.starting_gap_upper_bounds == gaps
        and reference.source_count == 2
        and reference.full_state_dimension == 72
        and reference.contact_pairs == _CONTACT_PAIRS
        and reference.contact_rate == 1
        and reference.charge_duration == reference.prefix_leak_duration == Q(1, 32)
        and reference.existence_gap_threshold == _DELTA / 8
        and len(reference.prefix_reports) == 2
    )
    source_ok = tuple(
        association
        and max(rx, rt) <= _SOURCE_CAP
        and present
        and prefix.form_error_bound == rx
        and prefix.phase_error_bound == rt
        and prefix.starting_gap_upper_bound == gap
        and prefix.source_certified
        and prefix.full_state_tube_certified
        and 0 < prefix.gamma_upper_bound <= _GAMMA_UPPER
        and 0 <= prefix.edge_weight_deviation_upper_bound <= _WEIGHT_DEVIATION
        and prefix.successor.conditional_image_certified
        and 0
        < prefix.successor.gamma_range[0]
        <= prefix.successor.gamma_range[1]
        <= _GAMMA_UPPER
        and prefix.successor.relative_phase_gap == _DELTA
        and prefix.successor.new_weight_range == (Q(3, 4), Q(1))
        and 0 <= prefix.successor.edge_weight_deviation_upper_bound <= _WEIGHT_DEVIATION
        and prefix.successor.completion_reports[0].geometry == geometry.successor
        for rx, rt, gap, present, prefix in zip(
            forms,
            phases,
            gaps,
            reference.source_certified,
            reference.prefix_reports,
            strict=True,
        )
    )
    exists = tuple(
        source and present and prefix.finite_protocol_ready_age_exists
        for source, present, prefix in zip(
            source_ok,
            reference.finite_protocol_ready_ages_exist,
            reference.prefix_reports,
            strict=True,
        )
    )
    ready = tuple(
        source
        and present
        and gap <= _DELTA / 8
        and prefix.conditional_ready_source
        and prefix.joint_registration_retained_certified
        for source, present, gap, prefix in zip(
            source_ok,
            reference.conditional_ready_sources,
            gaps,
            reference.prefix_reports,
            strict=True,
        )
    )
    source_association = all(source_ok) and all(exists) and all(ready)
    current = 45 * _REGISTRATION
    regional_force = _GAMMA_UPPER * current / 81
    full_force = 6 * _GAMMA_UPPER
    difference = full_force / _GAP
    weight_residual = Q(10, 9) * _WEIGHT_DEVIATION * _DIFFERENCE_ENVELOPE
    residual = regional_force + weight_residual
    # pi>3 and the source's original Delta imply k>this rational lower bound.
    coupling_margin = Q(13, 27) - _DELTA / 54 - _COUPLING_LOWER
    kernel_mass = 9 / (2 * _COUPLING_LOWER)
    transfer = 2 * residual * kernel_mass
    difference_ok = (
        source_association
        and geometry.certified
        and geometry.spanning_tree_gap_lower_bound >= _GAP
        and difference < _DIFFERENCE_ENVELOPE
        and coupling_margin > 0
        and transfer < _READOUT_CAP
    )
    observation_ok = readout <= _READOUT_CAP
    overlap_margin = 2 * readout - transfer
    overlap = difference_ok and observation_ok and overlap_margin >= 0
    unresolved = tuple(
        label
        for passed, label in (
            (
                all(source_ok),
                "actual_sources_or_frozen_weight_associations_unavailable",
            ),
            (all(exists), "finite_protocol_ready_age_existence_unavailable"),
            (all(ready), "conditional_source_or_shared_prefix_readiness_unavailable"),
            (geometry.certified, "marked_cross_geometry_or_connected_gap_unavailable"),
            (difference_ok, "conditional_uniform_transfer_bound_unavailable"),
            (observation_ok, "readout_allowance_outside_protocol"),
            (overlap_margin >= 0, "reading_allowance_does_not_certify_overlap"),
        )
        if not passed
    )
    return _MixedRegisteredPhaseProbe(
        form_error_bounds=forms,
        phase_error_bounds=phases,
        starting_gap_upper_bounds=gaps,
        readout_error_bound=readout,
        reference=reference,
        geometry=geometry,
        source_certified=source_ok,
        finite_protocol_ready_ages_exist=exists,
        conditional_ready_sources=ready,
        source_association_certified=source_association,
        cross_current_upper_bound=current,
        regional_sine_force_upper_bound=regional_force,
        full_sine_force_norm_upper_bound=full_force,
        form_difference_upper_bound=difference,
        form_difference_envelope=_DIFFERENCE_ENVELOPE,
        weight_residual_upper_bound=weight_residual,
        regional_residual_upper_bound=residual,
        nominal_coupling_lower_bound=_COUPLING_LOWER,
        nominal_coupling_lower_margin=coupling_margin,
        output_kernel_mass_upper_bound=kernel_mass,
        transfer_difference_upper_bound=transfer,
        difference_bound_certified=difference_ok,
        readout_allowance_admitted=observation_ok,
        overlap_margin=overlap_margin,
        shared_record_overlap_certified=overlap,
        certified=not unresolved,
        status="conditional_shared_record_overlap" if not unresolved else "unavailable",
        unresolved_conditions=unresolved,
    )
