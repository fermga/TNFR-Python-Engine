"""Finite critical geometry and asymptotic admission for the complete sine law.

The exact geometry is independent of the captured represented initial phases.
Positive loss and every held capacity are separate premises for convergence;
the catalog neither selects a terminal branch nor executes a trajectory.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from fractions import Fraction as Q

from ._cycle_algebra import ordered_vector
from ._sine_admission import _sine_model_coefficients
from .phase_cycle_geometry import (
    C5PhaseHessianInertia,
    C5SineCriticalSet,
    _derive,
    _validate_critical_set_labels,
    _validate_phase_hessian_labels,
    classify_c5_sine_critical_set,
)
from .relational_sine_comparison import (
    SineExchangeComparison,
    _sine_form_weights,
    _validate_comparison_labels,
)

__all__ = (
    "SineAsymptoticEquilibria",
    "SineEquilibriumStability",
    "assess_sine_asymptotic_equilibria",
)


@dataclass(frozen=True)
class SineAsymptoticEquilibria:
    """Conditional convergence to one unspecified exact critical phase pattern.

    This holds for every finite initial state under the captured positive-loss
    sine law on the admitted eleven-node support, with all capacities held
    strictly positive. It does not require acute phases or an initial energy
    bound. The common form limit is its conserved weighted mean. Lifted phase
    convergence follows separately from that row's conserved weighted mean
    and eventual consistent local lifts near one relative limit; the terminal
    integer-turn vector and the limiting critical branch are not selected.
    """

    source: SineExchangeComparison
    critical_set: C5SineCriticalSet
    conserved_form_mean: Q | None
    conserved_lifted_phase_mean: Q | None
    single_relative_equilibrium_convergence_certified: bool
    full_lifted_state_convergence_certified: bool
    hypothesis_failures: tuple[str, ...]
    status: str
    selected_equilibrium_status: str = "unavailable_no_basin_selection"
    proof_id: str = "eleven_node_positive_loss_sine_single_equilibrium_convergence"
    scope: tuple[str, ...] = (
        "complete_normalized_sine_law_not_native_Arg_or_variable_mobility",
        "exact_two_C5_intermediary_support_with_all_finite_initial_states",
        "positive_effective_loss_exchange_and_storage_scale_all_held_capacities_positive",
        "fixed_connected_unit_support_no_input_events_clipping_or_extra_phase_velocity",
        "compact_full_state_and_equilibrium_set_approach_reused_with_actual_law_premises",
        "finite_relative_critical_set_and_connected_limit_set_imply_one_relative_limit",
        "conserved_weighted_means_reconstruct_form_and_full_lifted_phase_convergence",
        "symbolic_catalog_is_not_a_rounded_initial_or_terminal_graph_state",
        "no_branch_selection_stability_classification_recovery_basin_or_transfer_verdict",
        "no_convergence_rate_deadline_or_finite_step_certificate",
        "zero_loss_or_capacity_leaves_this_theorem_unavailable_not_divergence_proved",
        "no_source_recapture_trajectory_or_new_constitutive_law",
        "public_dataclass_construction_is_not_provenance_authentication",
    )

    @property
    def admitted(self):
        return self.status == "admitted"

    def classify_equilibrium(
        self, *, cycle_choices, bridge_turns
    ) -> SineEquilibriumStability:
        """Apply the actual sine-law stability theorem to one exact catalog member.

        Both consumed source premises and supplied catalog fields are
        revalidated. Existing report flags cannot replace that admission.
        """
        if not isinstance(self.critical_set, C5SineCriticalSet):
            raise TypeError("a C5SineCriticalSet is required")
        validated = assess_sine_asymptotic_equilibria(
            self.source, cycles=self.critical_set.cycles
        )
        if validated.critical_set != self.critical_set:
            raise ValueError(
                "critical-set fields do not match the captured source support"
            )
        inertia = validated.critical_set.phase_hessian_inertia(
            cycle_choices=cycle_choices, bridge_turns=bridge_turns
        )
        stable = unstable = center = None
        attraction = instability = False
        status = "unavailable"
        if validated.admitted:
            unstable = inertia.relative_inertia[1]
            stable = 2 * (len(validated.source.nodes) - 1) - unstable
            center = 0
            attraction, instability = unstable == 0, unstable > 0
            status = (
                "locally_exponentially_attracting"
                if attraction
                else "nonlinearly_unstable"
            )
        return SineEquilibriumStability(
            source=validated,
            phase_hessian=inertia,
            relative_stable_modes=stable,
            relative_unstable_modes=unstable,
            relative_center_modes=center,
            local_exponential_attraction_certified=attraction,
            nonlinear_instability_certified=instability,
            hypothesis_failures=validated.hypothesis_failures,
            status=status,
        )

    def to_dict(self):
        from ..sdk.relational_reports import _project

        _validate_comparison_labels(self.source)
        _validate_critical_set_labels(self.critical_set)
        return {
            "schema": "tnfr.relational-sine-asymptotic-equilibria.v1",
            "report": _project(self),
        }


@dataclass(frozen=True)
class SineEquilibriumStability:
    """Full-law local stability on the relative twenty-dimensional state.

    Positive loss and positive held capacities turn a phase-Hessian negative
    index n into n positive real linear modes and 20-n stable modes. There
    are no relative center modes. The unrestricted state retains two common
    origin zero modes, so attraction is asserted only modulo those origins
    or on the corresponding conserved weighted-mean leaf. Nonlinear
    instability concerns this selected critical state, not every trajectory.
    No numeric exponent, neighborhood radius or initial-to-terminal selection
    is provided. Failed law premises leave all dynamic mode counts unavailable.
    """

    source: SineAsymptoticEquilibria
    phase_hessian: C5PhaseHessianInertia
    relative_stable_modes: int | None
    relative_unstable_modes: int | None
    relative_center_modes: int | None
    local_exponential_attraction_certified: bool
    nonlinear_instability_certified: bool
    hypothesis_failures: tuple[str, ...]
    status: str
    proof_id: str = "eleven_node_positive_loss_sine_relative_stability"
    scope: tuple[str, ...] = (
        "revalidated_source_law_and_exact_critical_catalog_not_inherited_status_flags",
        "positive_effective_loss_exchange_and_storage_scale_all_capacities_held_positive",
        "complete_joint_form_phase_Jacobian_not_phase_Hessian_alone",
        "relative_state_removes_two_common_origins_or_fixes_both_weighted_means",
        "positive_real_unstable_dimension_equals_phase_Hessian_negative_index",
        "all_other_relative_modes_stable_no_relative_center_modes",
        "local_exponential_attraction_or_nonlinear_instability_of_the_selected_member",
        "no_uniform_exponent_numeric_radius_or_specific_recovery_basin",
        "no_initial_state_terminal_branch_selection_or_receiver_transfer_verdict",
        "no_finite_step_solver_or_changed_runtime_stability_claim",
    )

    def to_dict(self):
        from ..sdk.relational_reports import _project

        _validate_comparison_labels(self.source.source)
        _validate_critical_set_labels(self.source.critical_set)
        _validate_phase_hessian_labels(self.phase_hessian)
        return {
            "schema": "tnfr.relational-sine-equilibrium-stability.v1",
            "report": _project(self),
        }


def assess_sine_asymptotic_equilibria(source, *, cycles) -> SineAsymptoticEquilibria:
    """Separate exact critical geometry from full-law asymptotic admission.

    ``cycles`` uses the captured node labels in each declared cycle order.
    Unsupported topology raises; an otherwise valid comparison with zero
    dissipation or an inactive node retains its exact geometric classification
    but cannot use the positive-loss, positive-capacity convergence theorem.
    """
    if not isinstance(source, SineExchangeComparison):
        raise TypeError("a captured SineExchangeComparison is required")
    model = source.reference_model
    e, w, beta = _sine_model_coefficients(model)
    positions = {node: i for i, node in enumerate(source.nodes)}
    edges = tuple(
        sorted(
            (min(positions[i], positions[j]), max(positions[i], positions[j]))
            for i, j in source.edges
        )
    )
    critical = classify_c5_sine_critical_set(
        _derive(source.nodes, edges), cycles=cycles
    )
    forms = ordered_vector(source.epi, "epi")
    phases = ordered_vector(source.phase, "phase")
    capacities = ordered_vector(source.capacity, "capacity")
    size = len(source.nodes)
    if any(len(values) != size for values in (forms, phases, capacities)):
        raise ValueError(
            "form, phase and capacity vectors must match the captured node order"
        )
    if any(nu < 0 for nu in capacities):
        raise ValueError("capacity must be nonnegative")
    degrees = tuple(sum(i in edge for edge in edges) for i in range(size))
    if (
        type(source.degrees) is not tuple
        or len(source.degrees) != size
        or any(type(degree) is not int for degree in source.degrees)
        or source.degrees != degrees
    ):
        raise ValueError("captured integer degrees must match the complete support")
    positive_capacity = all(nu > 0 for nu in capacities)
    failures = tuple(
        reason
        for condition, reason in (
            (
                source.law == "normalized_sine_reciprocal_exchange",
                "original_sine_law_required",
            ),
            (e > 0, "positive_epi_weight_required"),
            (w > 0 and beta > 0, "positive_exchange_and_storage_scale_required"),
            (positive_capacity, "strictly_positive_held_capacity_required"),
        )
        if not condition
    )
    form_mean = phase_mean = None
    if positive_capacity:
        weights = _sine_form_weights(replace(source, capacity=capacities))
        mass = sum(weights, Q(0))
        form_mean = sum((mu * x for mu, x in zip(weights, forms)), Q(0)) / mass
        phase_mean = (
            sum((mu * theta for mu, theta in zip(weights, phases)), Q(0)) / mass
        )
    return SineAsymptoticEquilibria(
        source=source,
        critical_set=critical,
        conserved_form_mean=form_mean,
        conserved_lifted_phase_mean=phase_mean,
        single_relative_equilibrium_convergence_certified=not failures,
        full_lifted_state_convergence_certified=not failures,
        hypothesis_failures=failures,
        status="unavailable" if failures else "admitted",
    )
