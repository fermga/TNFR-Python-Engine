"""Detached finite contact capability across supplied relative phase origins.

ORGANIZED_EXTERNAL_REGISTRATION.md owns the frozen prepared family. Both
quadratures retain actual matching boundary phases; their norm is not a
complete evolving state or an operational alignment selector.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from fractions import Fraction as Q

from .._exact_time import exact_or_represented_real as _exact
from ..mathematics._rational_interval import I, sin, sqrt
from ._organized_external_interaction import _Q as _RECOVERED_PHASE_BOUND
from ._organized_external_interaction import (
    _SOURCE_FAILURE_NAMES,
    _assess_organized_external_interaction,
    _derive_organized_external_interaction,
    _observe_admitted_external_interaction,
    _OrganizedExternalInteraction,
    _OrganizedExternalInteractionObservation,
)
from ._repeated_organized_composition import _bounded_sine_edge_fields
from ._sine_preparation import _prepared_coordinate_vector as _vector

_OUTER_ONE_UPPER_THRESHOLD = Q(1, 10**8)
_OUTER_ZERO_LOWER_THRESHOLD = Q(3, 10**7)


def _quadrature_norm(cosine, sine):
    """Enclose the norm of every correlated point inside a rectangle."""
    return sqrt(cosine**2 + sine**2)


@dataclass(frozen=True)
class _OrganizedExternalRegistrationObservation:
    relative_phase_origin: Q
    interaction: _OrganizedExternalInteractionObservation
    boundary_cosine_bounds: I
    boundary_sine_bounds: I
    boundary_amplitude_bounds: I
    scope: tuple[str, ...] = (
        "full_detached_state_with_supplied_exact_or_represented_relative_origin",
        "boundary_quadratures_exclude_the_supplied_common_external_origin",
        "all_forty_five_fine_matching_phase_correlations_retained",
        "instantaneous_amplitude_is_not_a_future_response_or_acquisition_certificate",
    )


def _observe_organized_external_registration(
    *, relative_phase_origin, form, phase_offsets
):
    """Observe an explicitly prepared origin without a contact or source reset."""
    beta = _exact(relative_phase_origin, "relative_phase_origin")
    values = _vector(form, "form", 54)
    offsets = _vector(phase_offsets, "phase_offsets", 54)
    original_geometry = _derive_organized_external_interaction()
    geometry = replace(
        original_geometry,
        external_phase_origin_bounds=I(beta),
        scope=tuple(
            (
                "external_origin_supplied_from_primitive_preparation"
                if entry
                == "external_pi_over_two_origin_supplied_from_primitive_preparation"
                else entry
            )
            for entry in original_geometry.scope
        ),
    )
    interaction = _observe_admitted_external_interaction(values, offsets, geometry)
    interaction = replace(
        interaction,
        scope=tuple(
            (
                "raw_rational_offsets_relative_to_carrier_and_declared_external_origin"
                if entry
                == "raw_rational_offsets_relative_to_carrier_and_symbolic_external_pi_over_two"
                else entry
            )
            for entry in interaction.scope
        ),
    )
    # Same-slot contacts cancel the winding carrier. Excluding beta here
    # retains the primitive relative state rather than rotating rounded fields.
    _, potential, edge_currents = _bounded_sine_edge_fields(
        phase_offsets=offsets,
        weighted_edges=geometry.contact_edges,
        alpha_bounds=geometry.alpha_bounds,
    )
    cosine = (45 - sum(potential, I(0))) / 9
    sine = sum(edge_currents, I(0)) / 9
    return _OrganizedExternalRegistrationObservation(
        relative_phase_origin=beta,
        interaction=interaction,
        boundary_cosine_bounds=cosine,
        boundary_sine_bounds=sine,
        boundary_amplitude_bounds=_quadrature_norm(cosine, sine),
    )


@dataclass(frozen=True)
class _OrganizedExternalRegistration:
    form_error_bounds: tuple[Q, ...]
    phase_error_bounds: tuple[Q, ...]
    external_interaction: _OrganizedExternalInteraction
    boundary_cosine_bounds: tuple[I, I]
    boundary_sine_bounds: tuple[I, I]
    boundary_amplitude_bounds: tuple[I, I]
    observed_capability_bounds: tuple[I, I]
    observation_remainder_upper_bound: Q
    outer_one_upper_margin: Q
    outer_zero_lower_margin: Q
    paired_absolute_response_gap_lower_bound: Q
    sufficient_margins: tuple[tuple[str, Q], ...]
    source_certified: bool
    work_certified: bool
    identity_certified: bool
    comparison_certified: bool
    admitted: bool
    status: str
    unresolved_conditions: tuple[str, ...]
    outer_one_upper_threshold: Q = _OUTER_ONE_UPPER_THRESHOLD
    outer_zero_lower_threshold: Q = _OUTER_ZERO_LOWER_THRESHOLD
    family_order: tuple[str, str] = ("outer_one", "outer_zero")
    protocol_revision: str = "d2f6f3df"
    scope: tuple[str, ...] = (
        "separately_prepared_relative_origins_not_a_sweep_of_one_contacted_object",
        "actual_acquired_sources_after_the_complete_single_history_observation",
        "both_fine_boundary_quadratures_before_uniform_whole_law_remainder",
        "per_source_supremum_over_origins_with_arbitrary_bounded_reading_errors",
        "outer_one_upper_bound_for_every_origin_outer_zero_witness_for_each_source",
        "witness_may_consume_hidden_source_information_no_alignment_selector",
        "work_and_fifteen_marked_identities_uniform_only_through_contact_horizon",
        "legacy_pi_over_two_comparison_verdict_is_not_a_registration_premise",
        "no_autonomous_moment_closure_intrinsic_charge_or_physical_identification",
        "no_trajectory_frozen_producer_replay_or_origin_scan",
    )


def _assess_organized_external_registration(*, form_error_bounds, phase_error_bounds):
    """Rebuild actual sources once, then bound their entire prepared-origin family.

    The shared primitive-only assessor re-admits all twelve allowances and
    rebuilds acquisition, recovery, observation transport and complete-law
    bounds. Its pi/2 response is retained as prior evidence, but neither its
    comparison verdict nor its combined admission selects the present result.
    No incoming report or cached source verdict is accepted.
    """
    source = _assess_organized_external_interaction(
        form_error_bounds=form_error_bounds, phase_error_bounds=phase_error_bounds
    )
    donor_error = I(-_RECOVERED_PHASE_BOUND, _RECOVERED_PHASE_BOUND)
    phase_families = (
        source.outer_one_host_phase_offset_bounds,
        source.outer_zero_host_phase_offset_bounds,
    )
    # The freshly rebuilt pi/2 boundary moments are these same beta-free
    # cosine quadratures. Reuse their arithmetic, not the old response verdict.
    cosines = (
        source.outer_one_boundary_moment_bounds,
        source.outer_zero_boundary_moment_bounds,
    )
    sines = tuple(
        -sum((sin(box + donor_error) for box in phases), I(0))
        for phases in phase_families
    )
    amplitudes = tuple(
        _quadrature_norm(cosine, sine) for cosine, sine in zip(cosines, sines)
    )
    factor = source.geometry.gamma_bounds * source.elapsed_scaled_time / 4
    error = (
        source.finite_response_remainder_upper_bound
        + 2 * source.readout_error_upper_bound
    )
    capability = tuple(
        I(max(Q(0), (factor * amplitude).lo - error), (factor * amplitude).hi + error)
        for amplitude in amplitudes
    )
    one_margin = _OUTER_ONE_UPPER_THRESHOLD - capability[0].hi
    zero_margin = capability[1].lo - _OUTER_ZERO_LOWER_THRESHOLD
    work_margins = (
        (
            "attachment_work",
            source.attachment_work_budget - source.contact_work_upper_bound,
        ),
    ) + tuple(
        (f"{family}_joined_storage", source.joined_storage_budget - storage)
        for family, storage in zip(
            source.family_order, source.joined_initial_storage_upper_bounds
        )
    )
    margins = (
        source.source_inclusion_margins
        + source.identity_margins
        + work_margins
        + (
            ("outer_one_all_origin_upper_bound", one_margin),
            ("outer_zero_per_source_witness_lower_bound", zero_margin),
        )
    )
    source_ok = source.source_certified
    work_ok = source_ok and all(value > 0 for _, value in work_margins)
    identity_ok = source_ok and all(value > 0 for _, value in source.identity_margins)
    comparison_ok = source_ok and one_margin > 0 and zero_margin > 0
    admitted = source_ok and work_ok and identity_ok and comparison_ok
    unresolved = tuple(
        name
        for name, ok in zip(_SOURCE_FAILURE_NAMES, source.source_admission_flags)
        if not ok
    ) + tuple(name for name, value in margins if value <= 0)
    return _OrganizedExternalRegistration(
        form_error_bounds=source.form_error_bounds,
        phase_error_bounds=source.phase_error_bounds,
        external_interaction=source,
        boundary_cosine_bounds=cosines,
        boundary_sine_bounds=sines,
        boundary_amplitude_bounds=amplitudes,
        observed_capability_bounds=capability,
        observation_remainder_upper_bound=error,
        outer_one_upper_margin=one_margin,
        outer_zero_lower_margin=zero_margin,
        paired_absolute_response_gap_lower_bound=capability[1].lo - capability[0].hi,
        sufficient_margins=margins,
        source_certified=source_ok,
        work_certified=work_ok,
        identity_certified=identity_ok,
        comparison_certified=comparison_ok,
        admitted=admitted,
        status=(
            "certified_organized_external_registration" if admitted else "unavailable"
        ),
        unresolved_conditions=unresolved,
    )
