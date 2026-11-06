"""Exact fixed-child-swap chart for the complete organized-source law.

The child exchange 1 <-> 2 fixes the marked targets in child zero. Its fixed
sector retains 36 forms, 36 continuous phase offsets and 36 held capacities.
This is a restriction to equal primitives, not an averaging of arbitrary
states. Shared support metadata supplies counts and conductance only; the
complete fine observer remains the pressure, phase and storage owner.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q

from ..mathematics._rational_interval import I
from ._organized_source_port import (
    _derive_organized_source_port,
    _observe_organized_source_capacity,
    _OrganizedSourceCapacityObservation,
    _OrganizedSourceGeometry,
)
from ._repeated_organized_composition import _vector
from .quotient_structure import observe_quotient_structure
from .support_transport import _from_data

_CHILD_SWAP = (0, 2, 1, 3, 4)


@dataclass(frozen=True)
class _OrganizedSourceChildQuotientGeometry:
    """Inherited geometry; orbit mass is distinct from retained fine capacity.

    Coordinates follow child orbits ``{0}, {1,2}, {3}, {4}``, then slots 0..8.
    ``multiplicity[a][b]`` counts neighbors of each fine member of block a in
    block b. ``weighted_edges`` aggregates conductance across whole blocks.
    Thus the per-member Laplacian is ``M^-1 L_w``, not ``L_w`` or a
    degree-normalized graph Laplacian. No effective-capacity law is imported.
    """

    fine_geometry: _OrganizedSourceGeometry
    blocks: tuple[tuple[int, ...], ...]
    representatives: tuple[int, ...]
    node_blocks: tuple[int, ...]
    orbit_masses: tuple[int, ...]
    multiplicity: tuple[tuple[int, ...], ...]
    support_degrees: tuple[int, ...]
    weighted_edges: tuple[tuple[int, int, Q], ...]
    weighted_degrees: tuple[Q, ...]
    scope: tuple[str, ...] = (
        "fixed_child_swap_orbits_on_the_shared_forty_five_node_support",
        "support_counts_and_aggregate_conductance_from_shared_metadata_only",
        "orbit_mass_and_retained_fine_capacity_are_separate_quantities",
        "no_degree_normalized_flow_or_effective_capacity_law_is_imported",
    )


def _derive_organized_source_child_quotient():
    """Derive the 36-node chart from the shared fine support and fixed swap."""
    fine = _derive_organized_source_port()
    nodes = fine.node_order
    swap = tuple(9 * _CHILD_SWAP[node // 9] + node % 9 for node in nodes)
    seen, blocks = set(), []
    for node in nodes:
        if node not in seen:
            block = tuple(sorted({node, swap[node]}))
            blocks.append(block)
            seen.update(block)
    blocks = tuple(blocks)
    edges = fine.weighted_edges
    canonical = {(min(i, j), max(i, j), w) for i, j, w in edges}
    exchanged = {(min(swap[i], swap[j]), max(swap[i], swap[j]), w) for i, j, w in edges}
    if canonical != exchanged:
        raise ValueError("child exchange must preserve the fine weighted support")
    neighbors = [set() for _ in nodes]
    directed = []
    for left, right, weight in edges:
        neighbors[left].add(right)
        neighbors[right].add(left)
        directed.extend(((left, right, weight), (right, left, weight)))
    # Zero form/pressure and unit capacities isolate support metadata. The
    # snapshot's normalized fields and effective capacity are not this law.
    snapshot = _from_data(
        nodes,
        tuple(directed),
        tuple(tuple(sorted(row)) for row in neighbors),
        (Q(0),) * len(nodes),
        (Q(1),) * len(nodes),
        (Q(0),) * len(nodes),
    )
    structure = observe_quotient_structure(snapshot, blocks)
    conductance = structure.macro_conductance
    weighted_edges = tuple(
        (a, b, weight)
        for a, row in enumerate(conductance)
        for b, weight in enumerate(row)
        if a < b and weight
    )
    return _OrganizedSourceChildQuotientGeometry(
        fine_geometry=fine,
        blocks=structure.blocks,
        representatives=tuple(block[0] for block in structure.blocks),
        node_blocks=structure.node_blocks,
        orbit_masses=tuple(len(block) for block in structure.blocks),
        multiplicity=structure.multiplicity,
        support_degrees=structure.block_degree,
        weighted_edges=weighted_edges,
        weighted_degrees=tuple(sum(row, Q(0)) for row in conductance),
    )


@dataclass(frozen=True)
class _OrganizedSourceChildQuotientState:
    """Normalized chart coordinates and their exact associated fine lift."""

    geometry: _OrganizedSourceChildQuotientGeometry
    form_coordinates: tuple[Q, ...]
    phase_offset_coordinates: tuple[Q, ...]
    capacity_coordinates: tuple[Q, ...]
    form: tuple[Q, ...]
    phase_offsets: tuple[Q, ...]
    capacities: tuple[Q, ...]


def _coordinates(form, phase_offsets, capacities, size):
    values = _vector(form, "form", size)
    offsets = _vector(phase_offsets, "phase_offsets", size)
    mobilities = _vector(capacities, "capacities", size)
    if any(value <= 0 for value in mobilities):
        raise ValueError("child quotient requires strictly positive held capacities")
    return values, offsets, mobilities


def _lift_coordinates(geometry, values, offsets, mobilities):
    def lift(row):
        return tuple(row[block] for block in geometry.node_blocks)

    return _OrganizedSourceChildQuotientState(
        geometry=geometry,
        form_coordinates=values,
        phase_offset_coordinates=offsets,
        capacity_coordinates=mobilities,
        form=lift(values),
        phase_offsets=lift(offsets),
        capacities=lift(mobilities),
    )


def _lift_organized_source_child_quotient(
    *, form_coordinates, phase_offset_coordinates, capacity_coordinates
):
    """Admit all primitives before copying coordinates into the fixed sector.

    Offsets refer to the shared carrier lift ``alpha*(slot-4)+offset``; they
    are neither wrapped nor centered. This chart has no energy, event-context
    or future-retention admission. Positive capacities are held parameters.
    """
    values, offsets, mobilities = _coordinates(
        form_coordinates, phase_offset_coordinates, capacity_coordinates, 36
    )
    geometry = _derive_organized_source_child_quotient()
    return _lift_coordinates(geometry, values, offsets, mobilities)


def _restrict_organized_source_child_quotient(*, form, phase_offsets, capacities):
    """Restrict only exactly equal normalized primitives in each swap orbit.

    Equality is checked after shared scalar admission, including the raw
    continuous phase offsets and positive capacities. No orbit averaging,
    circular equality, tolerance or projection repairs an off-sector state.
    """
    rows = _coordinates(form, phase_offsets, capacities, 45)
    geometry = _derive_organized_source_child_quotient()
    for label, row in zip(("form", "phase_offsets", "capacities"), rows):
        for block in geometry.blocks:
            if any(row[node] != row[block[0]] for node in block):
                raise ValueError(
                    f"{label} must be exactly equal within child-swap orbits"
                )
    reduced = tuple(tuple(row[i] for i in geometry.representatives) for row in rows)
    return _lift_coordinates(geometry, *reduced)


@dataclass(frozen=True)
class _OrganizedSourceChildQuotientObservation:
    """Representative fine rows and complete inherited energy/clock evidence.

    Pressure and gradients are per fine member, not aggregate covectors. The
    complete fine report retains storage and port evidence. Inverse-capacity
    totals and means include orbit multiplicity; phase means use the supplied
    real offsets. Context and ideal word admission remain separate premises.
    """

    state: _OrganizedSourceChildQuotientState
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
    joint_storage_bounds: I
    dissipation_tau: Q
    storage_tau_rate_bounds: I
    inverse_capacity_weight_total: Q
    inverse_capacity_weighted_form_total: Q
    inverse_capacity_weighted_phase_offset_total: Q
    inverse_capacity_weighted_form_mean: Q
    inverse_capacity_weighted_phase_offset_mean: Q
    scope: tuple[str, ...] = (
        "exact_fixed_sector_restriction_not_averaging_of_arbitrary_fine_states",
        "shared_fine_observer_owns_pressure_phase_storage_and_all_three_clocks",
        "both_rows_have_positive_held_shared_capacity_and_uniform_quarter_factor",
        "representative_pressure_is_per_member_not_unscaled_aggregate_pressure",
        "orbit_mass_is_retained_in_storage_loss_and_inverse_capacity_totals",
        "raw_continuous_phase_offsets_and_common_offsets_are_retained",
        "source_chart_does_not_admit_word_context_guard_level_or_retention",
        "no_events_trajectories_integrator_or_operator_executor",
    )


def _observe_organized_source_child_quotient(
    *, form_coordinates, phase_offset_coordinates, capacity_coordinates
):
    """Lift admitted state and select the shared complete-law fine rows.

    With orbit mass M and aggregate conductance Laplacian L_w, the retained
    form gradient is ``q=M^-1 L_w X``. Both phase and form rows use the actual
    held capacity D, so ``X'=D*(-q+gamma*s)/4`` and ``Theta'=gamma*D*q/4``.
    Sine currents, pressure, storage and clock conversion are never rebuilt
    by a competing quotient kernel. The chart alone certifies no event.
    """
    state = _lift_organized_source_child_quotient(
        form_coordinates=form_coordinates,
        phase_offset_coordinates=phase_offset_coordinates,
        capacity_coordinates=capacity_coordinates,
    )
    fine = _observe_organized_source_capacity(
        form=state.form,
        phase_offsets=state.phase_offsets,
        capacities=state.capacities,
        phase_mobility="shared_capacity",
    )
    reference = fine.unit_capacity_reference

    def select(row):
        return tuple(row[i] for i in state.geometry.representatives)

    weights = tuple(
        Q(mass) / capacity
        for mass, capacity in zip(
            state.geometry.orbit_masses, state.capacity_coordinates
        )
    )
    weight_total = sum(weights, Q(0))
    form_total = fine.inverse_capacity_weighted_form_total
    phase_total = fine.inverse_capacity_weighted_phase_offset_total
    return _OrganizedSourceChildQuotientObservation(
        state=state,
        fine=fine,
        form_gradient=select(reference.form_gradient),
        phase_current_bounds=select(reference.phase_current_bounds),
        pressure_tau_bounds=select(fine.pressure_tau_bounds),
        form_tau_rate_bounds=select(fine.form_tau_rate_bounds),
        phase_tau_rate_bounds=select(fine.phase_tau_rate_bounds),
        form_slow_rate_bounds=select(fine.form_slow_rate_bounds),
        phase_slow_rate_bounds=select(fine.phase_slow_rate_bounds),
        form_original_rate_bounds=select(fine.form_original_rate_bounds),
        phase_original_rate_bounds=select(fine.phase_original_rate_bounds),
        joint_storage_bounds=reference.joint_storage_bounds,
        dissipation_tau=fine.dissipation_tau,
        storage_tau_rate_bounds=fine.storage_tau_rate_bounds,
        inverse_capacity_weight_total=weight_total,
        inverse_capacity_weighted_form_total=form_total,
        inverse_capacity_weighted_phase_offset_total=phase_total,
        inverse_capacity_weighted_form_mean=form_total / weight_total,
        inverse_capacity_weighted_phase_offset_mean=phase_total / weight_total,
    )
