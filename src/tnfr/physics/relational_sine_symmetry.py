"""Full-state sine symmetries, exact reductions and winding obstructions.

The supplied permutation is checked against the entire unit support, held
capacity and exact initial form/phase lifts. A graph automorphism alone is not
a symmetry of a prepared trajectory. No group or trajectory is enumerated.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q
from typing import Any

from ..mathematics._rational_interval import I
from ._cycle_algebra import ordered_vector
from ._sine_admission import _admit_sine_source
from .phase_cycle_geometry import _cycle_row
from .relational_observations import _ordered
from .relational_sine_comparison import (
    SineExchangeComparison,
    _comparison_from_state,
    _sine_state_from_rows,
    _validate_comparison_labels,
)
from .symmetry_sectors import _validated_bijection

__all__ = (
    "SineCycleSymmetryAssessment",
    "assess_sine_cycle_symmetry",
    "SineInvolutionReduction",
    "SineInvolutionState",
    "assess_sine_involution_reduction",
)


def _admit_permutation_source(source, permutation_indices):
    """Share complete-source and permutation admission across symmetry readers."""
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
    mapped_edges = {
        (min(permutation[i], permutation[j]), max(permutation[i], permutation[j]))
        for i, j in edges
    }
    capacity_residuals = tuple(
        admitted.capacity[j] - admitted.capacity[i] for i, j in enumerate(permutation)
    )
    return admitted, edges, permutation, mapped_edges == set(edges), capacity_residuals


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
    admitted, edges, permutation, support_preserved, capacity_residuals = (
        _admit_permutation_source(source, permutation_indices)
    )
    size = len(admitted.nodes)
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


@dataclass(frozen=True)
class SineInvolutionReduction:
    """An invariant signed fixed-point family and separate source membership.

    For sign -1, paired nodes have opposite form and real phase coordinates;
    fixed nodes have both coordinates zero. Sign +1 identifies paired values
    and retains fixed-node coordinates. Each reconstruction column is positive
    at its smallest-index representative. No affine origin or circular lift
    adjustment is implicit. Source membership is not required to declare the
    invariant family, but is required to apply it to the supplied source.
    """

    source: SineExchangeComparison
    permutation_indices: tuple[int, ...]
    sign: int
    orbits: tuple[tuple[int, ...], ...]
    representative_indices: tuple[int, ...]
    reconstruction_matrix: tuple[tuple[Q, ...], ...]
    source_form_coordinates: tuple[Q, ...]
    source_phase_coordinates: tuple[Q, ...]
    form_reconstruction_residuals: tuple[Q, ...]
    phase_lift_reconstruction_residuals: tuple[Q, ...]
    capacity_residuals: tuple[Q, ...]
    support_automorphism: bool
    capacity_preserved: bool
    family_invariance_certified: bool
    source_membership_certified: bool
    source_trajectory_reduction_certified: bool
    status: str
    reasons: tuple[str, ...]
    scope: tuple[str, ...] = (
        "same_complete_unforced_sine_law_fixed_support_and_held_capacity",
        "exact_involutive_permutation_and_one_common_state_sign",
        "full_support_automorphism_and_preserved_capacity_prove_field_equivariance",
        "oddness_applies_to_both_form_and_real_phase_coordinates_together",
        "family_invariance_and_captured_source_membership_are_separate",
        "negative_sign_fixes_singleton_form_and_real_phase_to_zero",
        "positive_sign_retains_every_permutation_orbit_coordinate",
        "no_implicit_affine_origin_phase_wrapping_or_rounded_turn_identification",
        "arbitrary_independent_full_state_uncertainty_is_not_in_the_fixed_point_family",
        "no_winding_obstruction_transverse_stability_attraction_or_formation_claim",
        "no_solver_forcing_event_new_constitutive_law_or_physical_identification",
    )

    def evaluate(self, form_coordinates, phase_coordinates):
        """Reconstruct and evaluate one detached state in the admitted family."""
        return _evaluate_involution_state(self, form_coordinates, phase_coordinates)

    def to_dict(self):
        from ..sdk.relational_reports import _project

        _validate_comparison_labels(self.source)
        return {"schema": "tnfr.sine-involution-reduction.v1", "report": _project(self)}


@dataclass(frozen=True)
class SineInvolutionState:
    """A detached full state and its exactly inherited reduced field.

    ``comparison`` retains every full form, radian phase, held capacity, edge,
    rate and storage term. Reduced rates select representative rows of that
    complete field. Reconstruction residual intervals check arithmetic;
    full-row equality follows from the re-admitted family theorem, not zero
    overlap of those intervals. Rates use the original structural t clock.
    """

    reduction: SineInvolutionReduction
    form_coordinates: tuple[Q, ...]
    phase_coordinates: tuple[Q, ...]
    comparison: SineExchangeComparison
    form_rates: tuple[I, ...]
    phase_rates: tuple[I, ...]
    form_rate_reconstruction_residual_bounds: tuple[I, ...]
    phase_rate_reconstruction_residual_bounds: tuple[I, ...]
    full_row_equality_certified: bool
    clock: str = "original_structural_t"
    scope: tuple[str, ...] = (
        "source_law_support_capacity_permutation_and_sign_readmitted",
        "cached_reconstruction_membership_and_invariance_fields_not_used_as_evidence",
        "all_full_state_rows_and_storage_rebuilt_by_the_shared_sine_comparison",
        "reduced_rates_are_actual_representative_rows_not_a_different_quotient_law",
        "phase_coordinates_are_real_radian_lifts_and_rates_use_original_t",
        "detached_evaluation_does_not_replace_or_evolve_the_captured_source",
        "no_full_dimensional_uncertainty_stability_or_trajectory_certificate",
    )

    def to_dict(self):
        from ..sdk.relational_reports import _project

        self.reduction.to_dict()
        _validate_comparison_labels(self.comparison)
        return {"schema": "tnfr.sine-involution-state.v1", "report": _project(self)}


def _reconstruct_involution(matrix, coordinates, zero):
    return tuple(
        sum((entry * value for entry, value in zip(row, coordinates)), zero)
        for row in matrix
    )


def assess_sine_involution_reduction(
    source, *, permutation_indices, sign=-1
) -> SineInvolutionReduction:
    """Admit a signed fixed-point reduction of the complete supplied sine law.

    The full permutation must be an involution. Both signs +/-1 are admitted,
    with exact integer type; capacities themselves are not sign reversed.
    Nonautomorphisms or changed capacities leave family invariance unavailable.
    A source outside a valid family is reported separately and is not projected
    into it. No theorem about a surrounding independent uncertainty box follows.
    """
    if type(sign) is not int or sign not in (-1, 1):
        raise ValueError("sign must be the exact integer -1 or +1")
    admitted, edges, permutation, support_preserved, capacity_residuals = (
        _admit_permutation_source(source, permutation_indices)
    )
    size = len(admitted.nodes)
    if any(permutation[permutation[i]] != i for i in range(size)):
        raise ValueError("permutation_indices must define an involution")
    orbits = tuple(
        (i,) if i == j else (i, j) for i, j in enumerate(permutation) if i <= j
    )
    representatives = tuple(
        orbit[0] for orbit in orbits if sign == 1 or len(orbit) == 2
    )
    matrix = tuple(
        tuple(
            (
                Q(1)
                if i == representative
                else Q(sign if permutation[representative] == i else 0)
            )
            for representative in representatives
        )
        for i in range(size)
    )
    form_coordinates = tuple(admitted.epi[i] for i in representatives)
    phase_coordinates = tuple(admitted.phase[i] for i in representatives)
    form_residuals = tuple(
        full - value
        for full, value in zip(
            _reconstruct_involution(matrix, form_coordinates, Q(0)), admitted.epi
        )
    )
    phase_residuals = tuple(
        full - value
        for full, value in zip(
            _reconstruct_involution(matrix, phase_coordinates, Q(0)), admitted.phase
        )
    )
    capacity_preserved = not any(capacity_residuals)
    invariant = support_preserved and capacity_preserved
    member = not any(form_residuals) and not any(phase_residuals)
    reasons = tuple(
        reason
        for passed, reason in (
            (support_preserved, "full_support_not_preserved"),
            (capacity_preserved, "held_capacity_not_preserved"),
        )
        if not passed
    )
    return SineInvolutionReduction(
        source=admitted,
        permutation_indices=permutation,
        sign=sign,
        orbits=orbits,
        representative_indices=representatives,
        reconstruction_matrix=matrix,
        source_form_coordinates=form_coordinates,
        source_phase_coordinates=phase_coordinates,
        form_reconstruction_residuals=form_residuals,
        phase_lift_reconstruction_residuals=phase_residuals,
        capacity_residuals=capacity_residuals,
        support_automorphism=support_preserved,
        capacity_preserved=capacity_preserved,
        family_invariance_certified=invariant,
        source_membership_certified=member,
        source_trajectory_reduction_certified=invariant and member,
        status="certified" if invariant else "unavailable",
        reasons=reasons,
    )


def _evaluate_involution_state(report, form_coordinates, phase_coordinates):
    if not isinstance(report, SineInvolutionReduction):
        raise TypeError("a SineInvolutionReduction is required")
    rebuilt = assess_sine_involution_reduction(
        report.source, permutation_indices=report.permutation_indices, sign=report.sign
    )
    if not rebuilt.family_invariance_certified:
        raise ValueError("a certified invariant family is required for evaluation")
    coordinates = tuple(
        ordered_vector(
            _ordered(values, label, limit=len(rebuilt.representative_indices) + 1),
            label,
        )
        for values, label in (
            (form_coordinates, "form_coordinates"),
            (phase_coordinates, "phase_coordinates"),
        )
    )
    if any(len(row) != len(rebuilt.representative_indices) for row in coordinates):
        raise ValueError("coordinates must match every retained representative")
    matrix = rebuilt.reconstruction_matrix
    forms, phases = (
        _reconstruct_involution(matrix, values, Q(0)) for values in coordinates
    )
    source = rebuilt.source
    positions = {node: i for i, node in enumerate(source.nodes)}
    neighbors = [[] for _ in source.nodes]
    for left, right in source.edges:
        i, j = positions[left], positions[right]
        neighbors[i].append(j)
        neighbors[j].append(i)
    comparison = _comparison_from_state(
        _sine_state_from_rows(
            source.nodes,
            source.edges,
            forms,
            phases,
            source.capacity,
            tuple(tuple(row) for row in neighbors),
        ),
        source.reference_model,
    )
    form_rates = tuple(comparison.form_rates[i] for i in rebuilt.representative_indices)
    phase_rates = tuple(
        comparison.phase_rates[i] for i in rebuilt.representative_indices
    )
    residuals = tuple(
        tuple(
            full - recovered
            for full, recovered in zip(
                actual, _reconstruct_involution(matrix, reduced, I(0))
            )
        )
        for actual, reduced in (
            (comparison.form_rates, form_rates),
            (comparison.phase_rates, phase_rates),
        )
    )
    return SineInvolutionState(
        reduction=rebuilt,
        form_coordinates=coordinates[0],
        phase_coordinates=coordinates[1],
        comparison=comparison,
        form_rates=form_rates,
        phase_rates=phase_rates,
        form_rate_reconstruction_residual_bounds=residuals[0],
        phase_rate_reconstruction_residual_bounds=residuals[1],
        full_row_equality_certified=True,
    )
