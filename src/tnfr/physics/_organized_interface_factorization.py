"""Finite scalar-factorization test for two fixed normalized contact interfaces.

The prospective owner is ORGANIZED_INTERFACE_FACTORIZATION.md. Fresh source
reconstruction retains the entire preparation and observation history. True,
observed and error-rectification matrices have distinct meanings; no sampled
record, trajectory or independently tunable interface is consumed here.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q

from ..mathematics._rational_interval import I
from ._organized_external_interaction import (
    _OrganizedExternalInteractionGeometry,
    _derive_organized_external_interaction,
)
from ._organized_joint_resources import (
    _OrganizedJointResources,
    _assess_organized_joint_resources,
)
from ._organized_slot_registration import (
    _bounded_slot_registration_moments,
    _derive_organized_slot_registration,
)

_REGISTRATIONS = ((0, 0, 0, 0, 0), (5, 7, 0, 2, 4))
_OLD_RESPONSE_MARGINS = frozenset(
    (
        "outer_one_all_origin_response",
        "outer_zero_witness_response",
        "fixed_origin_response_contrast",
    )
)
_REQUIRED_RESOURCE_MARGINS = frozenset(
    (
        "outer_one_form",
        "outer_zero_form",
        "reprojected_fine_phase",
        "outer_one_origin_gap",
        "outer_zero_origin_gap",
        "internal_positive_gap",
        "internal_acute_gap",
        "outer_acute_gap",
        "absolute_event_work",
        "cumulative_absolute_event_work",
        "finite_nonlinear_comparison_domain",
        "positive_cosine_chart",
        "complete_cycle_winding_resolution",
        "host_internal_positive",
        "host_internal_acute",
        "donor_internal_positive",
        "donor_internal_acute",
        "outer_acute",
        "outer_capture",
        "outer_control_capture",
        "outer_attachment_work",
        "outer_initial_storage",
        "contact_attachment_work",
        "outer_one_joined_storage",
        "outer_zero_joined_storage",
    )
)


def _determinant(matrix):
    """Outward determinant of a caller-owned two-by-two interval rectangle."""
    return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]


def _zero_exclusion_margin(bounds):
    return max(bounds.lo, -bounds.hi)


def _support_admitted(geometry, original, slots):
    """Check the actual permutation support needed by the degree-based proof."""
    contact = tuple(
        (9 * c + j, 45 + (j + slots[c]) % 9, Q(1)) for c in range(5) for j in range(9)
    )
    expected = original.internal_edges + original.outer_edges + contact
    if geometry.weighted_edges != expected:
        return False
    degrees = [0] * 54
    for a, b, _ in expected:
        degrees[a] += 1
        degrees[b] += 1
    return (
        geometry.slot_offsets == slots
        and geometry.contact_edges == contact
        and geometry.internal_edges == original.internal_edges
        and geometry.outer_edges == original.outer_edges
        and geometry.internal_cycles == original.internal_cycles
        and geometry.outer_cycles == original.outer_cycles
        and geometry.node_order == tuple(range(54))
        and tuple(degrees) == geometry.support_degrees == (5,) * 45 + (7,) * 9
        and geometry.weighted_degrees == (Q(5),) * 45 + (Q(7),) * 9
        and geometry.held_capacities == (Q(1),) * 54
        and geometry.clock_factor == Q(1, 4)
        and geometry.phase_domain == "regular"
        and geometry.alpha_bounds == original.alpha_bounds
        and geometry.gamma_bounds == original.gamma_bounds
        and geometry.external_phase_origin_bounds
        == original.external_phase_origin_bounds
    )


@dataclass(frozen=True)
class _OrganizedInterfaceFactorization:
    source: _OrganizedJointResources
    geometries: tuple[_OrganizedExternalInteractionGeometry, ...]
    slot_registrations: tuple[tuple[int, ...], ...]
    source_phase_offset_bounds: tuple[tuple[I, ...], ...]
    cosine_moment_matrix: tuple[tuple[I, ...], ...]
    sine_moment_matrix: tuple[tuple[I, ...], ...]
    true_response_matrix: tuple[tuple[I, ...], ...]
    observed_response_matrix: tuple[tuple[I, ...], ...]
    rectification_matrix: tuple[tuple[I, ...], ...]
    true_determinant_bounds: I
    observed_determinant_bounds: I
    determinant_bounds: I
    finite_response_remainder_upper_bound: Q
    raw_readout_error_upper_bound: Q
    increment_readout_error_upper_bound: Q
    rectification_allowance_upper_bound: Q
    prerequisite_flags: tuple[tuple[str, bool], ...]
    prerequisite_margins: tuple[tuple[str, Q], ...]
    support_admission_flags: tuple[bool, ...]
    true_rank_margin: Q
    rectification_rank_margin: Q
    source_history_certified: bool
    finite_contact_certified: bool
    scalar_factorization_excluded: bool
    noise_robust_exclusion_certified: bool
    admitted: bool
    unresolved_conditions: tuple[str, ...]
    family_order: tuple[str, ...] = ("outer_zero", "outer_one")
    interface_order: tuple[str, ...] = ("same_slot", "cyclic_slot_registration")
    row_source_quantifier: str = (
        "one actual precontact source per row in two counterfactual contacts"
    )
    donor_quantifier: str = (
        "one shared actual donor member in all four counterfactual contacts"
    )
    factorization_hypothesis: str = (
        "true_increment(source, interface)=a(source)*b(interface), real a and b"
    )
    contact_horizon: Q = Q(1, 1000)
    raw_readings: int = 8
    scope: tuple[str, ...] = (
        "same_acquired_sources_after_the_entire_sequential_observation_word",
        "same_unit_contact_count_degrees_capacity_clock_donor_and_readout",
        "changed_slot_incidence_is_consumed_boundary_state_not_a_source_reset",
        "all_fine_coordinates_both_reactions_and_fifteen_finite_identities_retained",
        "true_finite_rank_distinct_from_onset_or_noisy_observed_rank",
        "rectification_encloses_every_observed_record_and_allowed_error_correction",
        "rank_obstruction_only_to_one_purely_multiplicative_real_factor",
        "no_operational_copying_intrinsic_charge_or_physical_identification",
        "no_trajectory_registration_scan_or_frozen_producer_replay",
    )


def _assess_organized_interface_factorization() -> _OrganizedInterfaceFactorization:
    """Assess the one declared pair; no report or retuned source is accepted."""
    source = _assess_organized_joint_resources(
        form_error_bounds=(Q(1, 10**6),) * 6,
        phase_error_bounds=(Q(1, 10**6),) * 6,
        quarter_dwell=Q(8 * 10**10),
        recovered_radius=Q(1, 10**8),
        decay_power=80,
        sequential_readout_error=Q(1, 10**14),
        contact_readout_error=Q(1, 10**9),
    )
    original = _derive_organized_external_interaction()
    geometries = tuple(
        _derive_organized_slot_registration(slot_offsets=slots)
        for slots in _REGISTRATIONS
    )
    support_flags = tuple(
        _support_admitted(geometry, original, slots)
        for geometry, slots in zip(geometries, _REGISTRATIONS)
    )
    outer = source.outer_resources
    # The old fixed-interface response verdict is not a source premise.
    flags = (
        ("six_actual_recoveries", all(s.recovery_certified for s in source.recoveries)),
        ("outer_one_capture", outer.conditional_capture_certified),
        ("outer_zero_capture", outer.control_conditional_capture_certified),
        ("outer_attachment_budget", outer.attachment_budget_certified),
        ("sequential_observation", source.observation.admitted),
    )
    margins = tuple(
        (name, value)
        for name, value in source.sufficient_margins
        if name not in _OLD_RESPONSE_MARGINS
    )
    margin_names = tuple(name for name, _ in margins)
    flags += (
        (
            "complete_resource_margin_inventory",
            len(set(margin_names)) == len(margin_names)
            and _REQUIRED_RESOURCE_MARGINS.issubset(margin_names),
        ),
    )
    history_ok = all(ok for _, ok in flags) and all(value > 0 for _, value in margins)
    contact_ok = history_ok and all(support_flags)
    z = outer.full_phase_error_upper_bound + source.observation_phase_drift_upper_bound
    host_error = I(-z, z)
    donor_error = I(-source.recovered_radius, source.recovered_radius)
    origins = ((I(0),) * 5, outer.primitive_entry.endpoint_phase_bounds)
    phase_families = tuple(
        tuple(value + host_error for value in phase for _ in range(9))
        + (donor_error,) * 9
        for phase in origins
    )
    moments = tuple(
        tuple(
            _bounded_slot_registration_moments(slot_offsets=slots, phase_offsets=phase)
            for slots in _REGISTRATIONS
        )
        for phase in phase_families
    )
    cosines = tuple(tuple(pair[0] for pair in row) for row in moments)
    sines = tuple(tuple(pair[1] for pair in row) for row in moments)
    factor = original.gamma_bounds * source.contact_horizon / 4
    remainder = source.response_remainder_upper_bound
    eta = source.contact_readout_error
    true = tuple(
        tuple(-factor * cosine + I(-remainder, remainder) for cosine in row)
        for row in cosines
    )
    observed = tuple(
        tuple(value + I(-2 * eta, 2 * eta) for value in row) for row in true
    )
    rectified = tuple(
        tuple(value + I(-4 * eta, 4 * eta) for value in row) for row in true
    )
    true_det, observed_det, rectified_det = map(
        _determinant, (true, observed, rectified)
    )
    true_margin = _zero_exclusion_margin(true_det)
    rectified_margin = _zero_exclusion_margin(rectified_det)
    excluded = contact_ok and true_margin > 0
    noise_ok = excluded and rectified_margin > 0
    unresolved = (
        tuple(name for name, ok in flags if not ok)
        + tuple(name for name, value in margins if value <= 0)
        + tuple(
            f"interface_{i}_support" for i, ok in enumerate(support_flags) if not ok
        )
        + tuple(
            name
            for name, value in (
                ("true_finite_rank", true_margin),
                ("noise_rectified_finite_rank", rectified_margin),
            )
            if value <= 0
        )
    )
    return _OrganizedInterfaceFactorization(
        source=source,
        geometries=geometries,
        slot_registrations=_REGISTRATIONS,
        source_phase_offset_bounds=phase_families,
        cosine_moment_matrix=cosines,
        sine_moment_matrix=sines,
        true_response_matrix=true,
        observed_response_matrix=observed,
        rectification_matrix=rectified,
        true_determinant_bounds=true_det,
        observed_determinant_bounds=observed_det,
        determinant_bounds=rectified_det,
        finite_response_remainder_upper_bound=remainder,
        raw_readout_error_upper_bound=eta,
        increment_readout_error_upper_bound=2 * eta,
        rectification_allowance_upper_bound=4 * eta,
        prerequisite_flags=flags,
        prerequisite_margins=margins,
        support_admission_flags=support_flags,
        true_rank_margin=true_margin,
        rectification_rank_margin=rectified_margin,
        source_history_certified=history_ok,
        finite_contact_certified=contact_ok,
        scalar_factorization_excluded=excluded,
        noise_robust_exclusion_certified=noise_ok,
        admitted=noise_ok,
        unresolved_conditions=unresolved,
    )
