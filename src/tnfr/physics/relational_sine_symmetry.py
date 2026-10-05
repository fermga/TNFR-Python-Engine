"""Full-state permutation symmetry and a conditional cycle-winding obstruction.

The supplied permutation is checked against the entire unit support, held
capacity and exact initial form/phase lifts. A graph automorphism alone is not
a symmetry of a prepared trajectory. No group or trajectory is enumerated.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q
from typing import Any

from ._sine_admission import _admit_sine_source
from .phase_cycle_geometry import _cycle_row
from .relational_observations import _ordered
from .relational_sine_comparison import (
    SineExchangeComparison,
    _validate_comparison_labels,
)
from .symmetry_sectors import _validated_bijection

__all__ = ("SineCycleSymmetryAssessment", "assess_sine_cycle_symmetry")


@dataclass(frozen=True)
class SineCycleSymmetryAssessment:
    """One supplied full-state symmetry and its action on an oriented cycle.

    ``trajectory_symmetry_certified`` is independent of cycle reversal. The
    zero-winding conclusion additionally requires the mapped integer cycle
    chain to be its negative, and applies only at times when the selected
    cycle has no antipodal edge. It supplies no phase-strip or consensus bound.
    Unavailable means these sufficient premises failed, not that winding forms.
    """

    source: SineExchangeComparison
    permutation_indices: tuple[int, ...]
    cycle: tuple[Any, ...]
    cycle_indices: tuple[int, ...]
    edge_indices: tuple[tuple[int, int], ...]
    cycle_chain: tuple[int, ...]
    mapped_cycle_chain: tuple[int, ...] | None
    support_automorphism: bool
    capacity_preserved: bool
    form_preserved: bool
    phase_lift_preserved: bool
    cycle_orientation_reversed: bool
    trajectory_symmetry_certified: bool
    zero_winding_when_nonantipodal: bool
    nonzero_winding_limit_excluded: bool
    form_residuals: tuple[Q, ...]
    phase_lift_residuals: tuple[Q, ...]
    capacity_residuals: tuple[Q, ...]
    initial_form_storage: Q
    status: str
    unresolved: tuple[str, ...]
    scope: tuple[str, ...] = (
        "complete_unforced_sine_law_fixed_connected_simple_unit_support",
        "shared_primitive_source_admission_nonnegative_loss_and_held_capacity",
        "one_explicit_complete_permutation_no_automorphism_group_enumeration",
        "exact_source_fixed_point_not_invariance_of_independent_uncertainty_boxes",
        "exact_real_phase_lift_equality_no_tolerance_or_rounded_two_pi_identification",
        "full_support_and_capacity_equivariance_plus_smooth_uniqueness_preserve_state_symmetry",
        "mapped_integer_cycle_chain_must_equal_negative_original_chain",
        "zero_selected_cycle_winding_only_when_its_principal_increments_avoid_antipodes",
        "excludes_nonzero_winding_circular_limits_away_from_antipodal_edges",
        "no_antipodal_avoidance_acute_confinement_stationarity_or_consensus_conclusion",
        "no_discrete_runtime_guarantee_selector_event_or_physical_identification",
        "cached_source_fields_unused_export_does_not_authenticate_preparation",
    )

    def to_dict(self):
        from ..sdk.relational_reports import _project

        _validate_comparison_labels(self.source)
        return {
            "schema": "tnfr.relational-sine-cycle-symmetry.v1",
            "report": _project(self),
        }


def assess_sine_cycle_symmetry(
    source, *, permutation_indices, cycle
) -> SineCycleSymmetryAssessment:
    """Check a sufficient full-law obstruction to oriented cycle formation.

    ``permutation_indices[i]`` is the destination position of node i in the
    source's retained node order. ``cycle`` contains at least three distinct
    node labels in traversal order, without repeating the first node. It may
    cover only part of the support; every external edge and node still enters
    full-source symmetry admission. Bijections that fail to preserve support,
    capacity, form, phase lifts or cycle orientation return unavailable.

    Malformed inputs raise. Only exact SineExchangeComparison sources are
    supported: a symmetric nominal state or a set-invariant residual box does
    not prove every member fixed. Nonnegative loss/capacity is sufficient;
    storage scale and exchange remain positive under shared law admission.
    """
    if not isinstance(source, SineExchangeComparison):
        raise TypeError("an exact SineExchangeComparison is required")
    admitted, edges = _admit_sine_source(source)
    size = len(admitted.nodes)
    raw = _ordered(permutation_indices, "permutation_indices", limit=size + 1)
    if any(type(value) is not int for value in raw):
        raise ValueError("permutation_indices must contain nonboolean integer indices")
    permutation = _validated_bijection(
        dict(enumerate(raw)), {i: i for i in range(size)}
    )
    cycle = _ordered(cycle, "cycle", limit=size + 1)
    positions = {node: i for i, node in enumerate(admitted.nodes)}
    try:
        indices = tuple(positions[node] for node in cycle)
    except (KeyError, TypeError) as exc:
        raise ValueError("cycle nodes must belong to the complete source") from exc
    if len(indices) < 3 or len(set(indices)) != len(indices):
        raise ValueError("cycle must contain at least three distinct ordered nodes")
    edge_positions = {edge: i for i, edge in enumerate(edges)}
    try:
        chain = _cycle_row(indices, edge_positions)
    except KeyError as exc:
        raise ValueError("every cycle edge must belong to the complete source") from exc
    mapped_edges = {
        (min(permutation[i], permutation[j]), max(permutation[i], permutation[j]))
        for i, j in edges
    }
    support_preserved = mapped_edges == set(edges)
    try:
        mapped_chain = _cycle_row(
            tuple(permutation[i] for i in indices), edge_positions
        )
    except KeyError:
        mapped_chain = None
    reversed_cycle = mapped_chain == tuple(-value for value in chain)

    def residual(values):
        return tuple(values[j] - values[i] for i, j in enumerate(permutation))

    form_residuals = residual(admitted.epi)
    phase_residuals = residual(admitted.phase)
    capacity_residuals = residual(admitted.capacity)
    form_preserved = not any(form_residuals)
    phase_preserved = not any(phase_residuals)
    capacity_preserved = not any(capacity_residuals)
    symmetry = all(
        (support_preserved, capacity_preserved, form_preserved, phase_preserved)
    )
    obstruction = symmetry and reversed_cycle
    unresolved = tuple(
        label
        for passed, label in (
            (support_preserved, "full_support_not_preserved"),
            (capacity_preserved, "held_capacity_not_preserved"),
            (form_preserved, "initial_form_not_fixed"),
            (phase_preserved, "initial_real_phase_lift_not_fixed"),
            (reversed_cycle, "selected_cycle_orientation_not_reversed"),
        )
        if not passed
    )
    storage = sum(
        ((admitted.epi[j] - admitted.epi[i]) ** 2 / 2 for i, j in edges), Q(0)
    )
    return SineCycleSymmetryAssessment(
        source=admitted,
        permutation_indices=permutation,
        cycle=cycle,
        cycle_indices=indices,
        edge_indices=edges,
        cycle_chain=chain,
        mapped_cycle_chain=mapped_chain,
        support_automorphism=support_preserved,
        capacity_preserved=capacity_preserved,
        form_preserved=form_preserved,
        phase_lift_preserved=phase_preserved,
        cycle_orientation_reversed=reversed_cycle,
        trajectory_symmetry_certified=symmetry,
        zero_winding_when_nonantipodal=obstruction,
        nonzero_winding_limit_excluded=obstruction,
        form_residuals=form_residuals,
        phase_lift_residuals=phase_residuals,
        capacity_residuals=capacity_residuals,
        initial_form_storage=storage,
        status="certified" if obstruction else "unavailable",
        unresolved=unresolved,
    )
