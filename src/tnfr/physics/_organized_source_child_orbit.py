"""Exact organized-source state modulo the single global child exchange.

The 36 projected coordinates in each primary channel and the joint Gram of
18 hidden half-differences retain the full state up to exchanging children
one and two everywhere. Independent slot exchanges are not this symmetry.
The redundant rank-one Gram is not a dimension reduction: the generic
primary dimension remains 90, with a singular zero-hidden stratum.

This detached rational chart reuses the complete fine observer. It installs
no evolution law, event selector or operator executor, and admits no word
context or energy guard merely by encoding a state.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q

from ..mathematics._exact_linear_algebra import (
    ExactSquareMatrix,
    exact_rank_one_rational_lift,
)
from ..mathematics._rational_interval import I
from ._organized_source_child_quotient import (
    _derive_organized_source_child_quotient,
    _lift_organized_source_child_quotient,
    _OrganizedSourceChildQuotientGeometry,
)
from ._organized_source_port import (
    _observe_organized_source_capacity,
    _OrganizedSourceCapacityObservation,
)
from ._repeated_organized_composition import _vector


def _project(row, blocks):
    """Average every orbit, including off-sector fine rows."""
    return tuple(
        (
            row[block[0]]
            if len(block) == 1
            else sum((row[node] for node in block), Q(0)) / len(block)
        )
        for block in blocks
    )


def _gram(vector):
    return tuple(tuple(left * right for right in vector) for left in vector)


@dataclass(frozen=True)
class _OrganizedSourceChildOrbitState:
    """Normalized invariant coordinates and one canonical labeled fine lift.

    The hidden vector orders the nine form half-differences before the nine
    real phase-offset half-differences. Its first nonzero coordinate is
    positive. Thus the fine lift is the encoded input or its one global
    child exchange; the discarded orientation is not silently recovered.
    This sign convention is not a globally smooth coordinate chart.
    """

    geometry: _OrganizedSourceChildQuotientGeometry
    form_coordinates: tuple[Q, ...]
    phase_offset_coordinates: tuple[Q, ...]
    capacity_coordinates: tuple[Q, ...]
    hidden_gram: ExactSquareMatrix
    hidden_coordinates: tuple[Q, ...]
    form: tuple[Q, ...]
    phase_offsets: tuple[Q, ...]
    capacities: tuple[Q, ...]
    scope: tuple[str, ...] = (
        "one_global_child_exchange_not_independent_slot_exchanges",
        "joint_form_phase_gram_retains_cross_slot_and_cross_channel_information",
        "positive_held_capacities_equal_on_exchanged_children",
        "raw_carrier_phase_offsets_are_retained_without_wrapping_or_centering",
        "canonical_rational_lift_forgets_only_the_global_child_orientation",
        "real_gram_points_without_rational_lifts_are_outside_this_decoder",
        "redundant_rank_one_gram_is_not_a_dimension_reduction",
        "no_word_context_guard_occurrence_or_retention_admission",
    )


def _decode_organized_source_child_orbit(
    *,
    form_coordinates,
    phase_offset_coordinates,
    capacity_coordinates,
    hidden_gram,
):
    """Admit a realized joint Gram and recover its canonical rational lift.

    All supplied entries pass shared scalar admission before geometry or
    fine observation. A positive-semidefinite rank-at-most-one real Gram
    with an irrational canonical square root is outside the exact rational
    decoder, rather than physically impossible or rounded into this chart.
    """
    values = _vector(form_coordinates, "form_coordinates", 36)
    offsets = _vector(phase_offset_coordinates, "phase_offset_coordinates", 36)
    mobilities = _vector(capacity_coordinates, "capacity_coordinates", 36)
    hidden = exact_rank_one_rational_lift(hidden_gram)
    if len(hidden) != 18:
        raise ValueError("hidden_gram requires 18 by 18 joint coordinates")
    even = _lift_organized_source_child_quotient(
        form_coordinates=values,
        phase_offset_coordinates=offsets,
        capacity_coordinates=mobilities,
    )
    pairs = tuple(block for block in even.geometry.blocks if len(block) == 2)
    form, phase_offsets = list(even.form), list(even.phase_offsets)
    for slot, (left, right) in enumerate(pairs):
        form[left] += hidden[slot]
        form[right] -= hidden[slot]
        phase_offsets[left] += hidden[9 + slot]
        phase_offsets[right] -= hidden[9 + slot]
    return _OrganizedSourceChildOrbitState(
        geometry=even.geometry,
        form_coordinates=values,
        phase_offset_coordinates=offsets,
        capacity_coordinates=mobilities,
        hidden_gram=_gram(hidden),
        hidden_coordinates=hidden,
        form=tuple(form),
        phase_offsets=tuple(phase_offsets),
        capacities=even.capacities,
    )


def _encode_organized_source_child_orbit(*, form, phase_offsets, capacities):
    """Encode fine primitives, retaining unequal children through their Gram.

    Only corresponding capacities must agree exactly. Forms and raw phase
    offsets are averaged and their joint half-difference Gram is retained;
    no tolerance, circular wrapping or independent local relabeling applies.
    """
    values = _vector(form, "form", 45)
    offsets = _vector(phase_offsets, "phase_offsets", 45)
    mobilities = _vector(capacities, "capacities", 45)
    if any(value <= 0 for value in mobilities):
        raise ValueError("child orbit requires strictly positive held capacities")
    if any(mobilities[9 + slot] != mobilities[18 + slot] for slot in range(9)):
        raise ValueError("capacities must be exactly equal on exchanged children")
    geometry = _derive_organized_source_child_quotient()
    pairs = tuple(block for block in geometry.blocks if len(block) == 2)
    hidden = tuple(
        (row[left] - row[right]) / 2
        for row in (values, offsets)
        for left, right in pairs
    )
    return _decode_organized_source_child_orbit(
        form_coordinates=_project(values, geometry.blocks),
        phase_offset_coordinates=_project(offsets, geometry.blocks),
        capacity_coordinates=_project(mobilities, geometry.blocks),
        hidden_gram=_gram(hidden),
    )


@dataclass(frozen=True)
class _OrganizedSourceChildOrbitObservation:
    """Complete averaged rows and invariant Gram rate of the canonical lift.

    The associated fine report retains labeled rows in the chosen lift.
    Coarse fields average every orbit; representative selection would give
    the wrong projected off-sector state derivative. Hidden vector rates
    refer to the chosen lift, whereas their joint Gram derivative is
    independent of its global sign. All rate bounds are outward enclosures.
    """

    state: _OrganizedSourceChildOrbitState
    fine: _OrganizedSourceCapacityObservation
    form_gradient: tuple[Q, ...]
    phase_current_bounds: tuple[I, ...]
    pressure_tau_bounds: tuple[I, ...]
    form_tau_rate_bounds: tuple[I, ...]
    phase_tau_rate_bounds: tuple[I, ...]
    form_slow_rate_bounds: tuple[I, ...]
    phase_slow_rate_bounds: tuple[I, ...]
    form_original_rate_bounds: tuple[I, ...]
    phase_original_rate_bounds: tuple[I, ...]
    hidden_tau_rate_bounds: tuple[I, ...]
    hidden_gram_tau_rate_bounds: tuple[tuple[I, ...], ...]
    joint_storage_bounds: I
    dissipation_tau: Q
    storage_tau_rate_bounds: I
    inverse_capacity_weight_total: Q
    inverse_capacity_weighted_form_total: Q
    inverse_capacity_weighted_phase_offset_total: Q
    inverse_capacity_weighted_form_mean: Q
    inverse_capacity_weighted_phase_offset_mean: Q
    scope: tuple[str, ...] = (
        "shared_fine_observer_owns_both_nonlinear_rows_and_all_three_clocks",
        "all_projected_rows_average_fine_orbits_including_unequal_children",
        "gram_tau_derivative_is_the_full_joint_outer_product_chain_rule",
        "fine_report_and_hidden_vector_rates_use_the_canonical_labeled_lift",
        "positive_held_shared_capacities_may_be_nonuniform_across_slots",
        "full_storage_loss_and_inverse_capacity_totals_retain_hidden_state",
        "orbit_mass_and_fine_capacity_are_separate_quantities",
        "generic_primary_dimension_remains_ninety_and_zero_hidden_is_singular",
        "no_events_trajectories_integrator_or_operator_executor",
        "word_domain_context_and_guard_admission_remain_separate_premises",
    )


def _observe_organized_source_child_orbit(
    *,
    form_coordinates,
    phase_offset_coordinates,
    capacity_coordinates,
    hidden_gram,
):
    """Decode primitives and project the shared complete-law fine observation.

    With H=L_C9+2I and V the exchanged children's held capacity diagonal,
    the hidden rows are ``a'=-(V H a+gamma V K b)/4`` and
    ``b'=gamma V H a/4``. Their Gram derivative is obtained here directly
    from the fine rows, without rebuilding the phase-dependent K or a sine
    kernel. In particular the hidden loss is ``tr(H V H G_aa)/2``; uniform
    capacity must not be silently substituted for the supplied V.
    """
    state = _decode_organized_source_child_orbit(
        form_coordinates=form_coordinates,
        phase_offset_coordinates=phase_offset_coordinates,
        capacity_coordinates=capacity_coordinates,
        hidden_gram=hidden_gram,
    )
    fine = _observe_organized_source_capacity(
        form=state.form,
        phase_offsets=state.phase_offsets,
        capacities=state.capacities,
        phase_mobility="shared_capacity",
    )
    reference = fine.unit_capacity_reference
    blocks = state.geometry.blocks
    pairs = tuple(block for block in blocks if len(block) == 2)
    hidden_rate = tuple(
        (row[left] - row[right]) / 2
        for row in (fine.form_tau_rate_bounds, fine.phase_tau_rate_bounds)
        for left, right in pairs
    )
    hidden = state.hidden_coordinates
    gram_rate = tuple(
        tuple(
            hidden_rate[i] * hidden[j] + hidden[i] * hidden_rate[j] for j in range(18)
        )
        for i in range(18)
    )
    weight_total = sum(
        (
            Q(mass) / capacity
            for mass, capacity in zip(
                state.geometry.orbit_masses, state.capacity_coordinates
            )
        ),
        Q(0),
    )
    form_total = fine.inverse_capacity_weighted_form_total
    phase_total = fine.inverse_capacity_weighted_phase_offset_total
    return _OrganizedSourceChildOrbitObservation(
        state=state,
        fine=fine,
        form_gradient=_project(reference.form_gradient, blocks),
        phase_current_bounds=_project(reference.phase_current_bounds, blocks),
        pressure_tau_bounds=_project(fine.pressure_tau_bounds, blocks),
        form_tau_rate_bounds=_project(fine.form_tau_rate_bounds, blocks),
        phase_tau_rate_bounds=_project(fine.phase_tau_rate_bounds, blocks),
        form_slow_rate_bounds=_project(fine.form_slow_rate_bounds, blocks),
        phase_slow_rate_bounds=_project(fine.phase_slow_rate_bounds, blocks),
        form_original_rate_bounds=_project(fine.form_original_rate_bounds, blocks),
        phase_original_rate_bounds=_project(fine.phase_original_rate_bounds, blocks),
        hidden_tau_rate_bounds=hidden_rate,
        hidden_gram_tau_rate_bounds=gram_rate,
        joint_storage_bounds=reference.joint_storage_bounds,
        dissipation_tau=fine.dissipation_tau,
        storage_tau_rate_bounds=fine.storage_tau_rate_bounds,
        inverse_capacity_weight_total=weight_total,
        inverse_capacity_weighted_form_total=form_total,
        inverse_capacity_weighted_phase_offset_total=phase_total,
        inverse_capacity_weighted_form_mean=form_total / weight_total,
        inverse_capacity_weighted_phase_offset_mean=phase_total / weight_total,
    )
