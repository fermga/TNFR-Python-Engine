"""All-time consensus for a phase-flat family with bounded original form storage.

The mixed Lyapunov function is derived from the complete positive-loss sine
law. It is an analytic certificate, not a pressure law or a new storage model.
No representative state, finite trajectory or reference endpoint is supplied.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q
from typing import Any

from .._exact_time import exact_or_represented_real
from ..dynamics.relational import RelationalExchangeModel
from ..mathematics._rational_interval import INTERVAL_METHOD, I, sqrt
from ._sine_preparation import _sine_domain
from .phase_cycle_geometry import PhaseCycleGeometry
from .relational_sine_comparison import _validate_comparison_labels

__all__ = (
    "SinePhaseFlatBudgetFamily",
    "SineBudgetConsensus",
    "certify_sine_budget_consensus",
)


@dataclass(frozen=True)
class SinePhaseFlatBudgetFamily:
    """All signed forms with half x.T L x <= B and initially identical phases.

    Every member has its own arbitrary, independent common form and phase
    origins. Capacity is exact and held in geometry node order. Flatness means
    an identical real phase lift; independent nonzero phase errors are not part
    of this family. The unit support and complete unforced sine law are supplied.
    """

    geometry: PhaseCycleGeometry
    reference_model: RelationalExchangeModel
    capacity: tuple[Q, ...]
    form_storage_budget: Q

    @property
    def nodes(self) -> tuple[Any, ...]:
        return self.geometry.nodes

    @property
    def edges(self) -> tuple[tuple[Any, Any], ...]:
        return tuple((self.nodes[i], self.nodes[j]) for i, j in self.geometry.edges)


@dataclass(frozen=True)
class SineBudgetConsensus:
    """Conditional all-time bounds and a necessary winding-acquisition budget.

    Candidate edge bounds describe the proposed invariant Lyapunov sublevel;
    they are not trajectory bounds unless admission succeeds. Certified bounds
    are None when unavailable. A failed comparison establishes no instability,
    nonzero winding or absence of consensus. No convergence rate is claimed.

    The necessary nonzero-winding budget applies only under the proved feedback
    premise and exact phase-flat preparation. It is conservative, not sufficient
    for formation. The auxiliary function W is not the model's original energy.
    """

    family: SinePhaseFlatBudgetFamily
    coefficient_ratio: Q
    mobility: tuple[Q, ...]
    metric_weights: tuple[Q, ...]
    weighted_gap_lower_bound: Q
    form_to_phase_scale_bounds: I
    feedback_strength_bounds: I
    initial_form_norm_upper_bound: Q
    initial_lyapunov_upper_bound: Q
    candidate_phase_edge_upper_bounds: tuple[Q, ...]
    feedback_margin: I
    branch_margins: tuple[I, ...]
    acute_margins: tuple[I, ...]
    nonzero_winding_budget_lower_bound: Q | None
    stationary: bool
    consensus_certified: bool
    zero_winding_certified: bool
    acute_trapping_certified: bool
    lyapunov_upper_bound: Q | None
    phase_norm_upper_bound: Q | None
    scaled_form_norm_upper_bound: Q | None
    form_norm_upper_bound: Q | None
    phase_edge_upper_bounds: tuple[Q, ...] | None
    status: str
    unresolved: tuple[str, ...]
    arithmetic_method: str = INTERVAL_METHOD
    scope: tuple[str, ...] = (
        "whole_phase_flat_family_with_original_form_storage_at_most_declared_budget",
        "complete_unforced_positive_loss_sine_law_on_supplied_connected_unit_support",
        "strictly_positive_exact_held_capacities_in_geometry_node_order",
        "each_member_has_its_own_conserved_weighted_form_and_phase_means",
        "scaled_clock_tau_equals_e_times_structural_time_covers_all_nonnegative_time",
        "mixed_lyapunov_W_is_auxiliary_not_the_original_storage_or_a_new_law",
        "feedback_at_most_one_and_strict_raw_pi_strip_are_sufficient_proof_premises",
        "zero_budget_family_is_stationary_for_every_admitted_positive_ratio",
        "strict_invariant_sublevel_and_LaSalle_prove_full_consensus_not_just_winding",
        "winding_budget_lower_bound_is_necessary_not_sufficient_or_a_sharp_threshold",
        "candidate_bounds_do_not_certify_a_trajectory_when_status_is_unavailable",
        "fixed_dyadic_enclosures_can_leave_a_valid_mathematical_condition_unresolved",
        "no_finite_solver_event_support_birth_law_selection_or_physical_identification",
        "public_dataclass_construction_and_export_do_not_authenticate_provenance",
    )

    def to_dict(self):
        from ..sdk.relational_reports import _project

        _validate_comparison_labels(self.family)
        return {
            "schema": "tnfr.relational-sine-budget-consensus.v1",
            "report": _project(self),
        }


def certify_sine_budget_consensus(
    geometry, *, reference_model, capacity, form_storage_budget
) -> SineBudgetConsensus:
    """Certify full consensus for every member of a bounded phase-flat family.

    Rebuild the supplied geometry (32 nodes/50 edges maximum), admit the regular
    positive-loss/exchange model and positive held capacities, then bound the
    original form preparation using a proved weighted Laplacian gap. No state
    coordinates are consumed or invented. Invalid domains raise. Sufficient
    inequalities that cannot be proved return unavailable, with no certified
    trajectory bounds. The zero-budget family is exactly stationary.

    W = ||vartheta+z||_M^2/2 + ||z||_M^2/2, z=alpha*P_M*x,
    alpha=w/(beta*pi*e), eta=beta*alpha^2. For eta<=1 its derivative
    is strictly negative away from consensus while every raw edge gap is in
    (-pi, pi). An invariant sublevel inside that strip proves the conclusion.
    """
    budget = exact_or_represented_real(form_storage_budget, "form_storage_budget")
    if budget < 0:
        raise ValueError("form_storage_budget must be nonnegative")
    d = _sine_domain(geometry, reference_model, capacity)
    family = SinePhaseFlatBudgetFamily(d.geometry, d.model, d.capacity, budget)
    ratio = d.w / d.e
    form_norm = sqrt(I(2 * budget / d.gap)).hi
    # W(0)=alpha^2 ||P_M x(0)||_M^2. Never invert a rounded eta or alpha
    # interval: their lower endpoint can be zero for a positive exact law.
    initial_lyapunov = (d.eta * (2 * budget / (d.beta * d.gap))).hi
    root_w = sqrt(I(initial_lyapunov)).hi
    edge_constants = tuple(d.mobility[i] + d.mobility[j] for i, j in d.geometry.edges)
    candidates = tuple(2 * sqrt(I(value)).hi * root_w for value in edge_constants)
    feedback_margin = 1 - d.eta
    branch_margins = tuple(d.pi - value for value in candidates)
    acute_margins = tuple(d.pi / 2 - value for value in candidates)
    feedback_admitted = feedback_margin.lo >= 0
    stationary = budget == 0
    unresolved = []
    if not stationary:
        if not feedback_admitted:
            unresolved.append("feedback_strength_at_most_one_not_certified")
        if any(margin.lo <= 0 for margin in branch_margins):
            unresolved.append("strict_raw_pi_strip_not_certified")
    certified = not unresolved
    # Contrapositive of the invariant-strip theorem. Exact coefficients are
    # inverted before pi enclosure; this bound does not assume capture passes.
    winding_budget = (
        ((d.beta**2 * d.gap / (8 * max(edge_constants) * ratio**2)) * d.pi**4).lo
        if feedback_admitted
        else None
    )
    return SineBudgetConsensus(
        family=family,
        coefficient_ratio=ratio,
        mobility=d.mobility,
        metric_weights=d.weights,
        weighted_gap_lower_bound=d.gap,
        form_to_phase_scale_bounds=d.alpha,
        feedback_strength_bounds=d.eta,
        initial_form_norm_upper_bound=form_norm,
        initial_lyapunov_upper_bound=initial_lyapunov,
        candidate_phase_edge_upper_bounds=candidates,
        feedback_margin=feedback_margin,
        branch_margins=branch_margins,
        acute_margins=acute_margins,
        nonzero_winding_budget_lower_bound=winding_budget,
        stationary=stationary,
        consensus_certified=certified,
        zero_winding_certified=certified,
        acute_trapping_certified=certified and all(m.lo > 0 for m in acute_margins),
        lyapunov_upper_bound=initial_lyapunov if certified else None,
        phase_norm_upper_bound=2 * root_w if certified else None,
        scaled_form_norm_upper_bound=(
            sqrt(I(2 * initial_lyapunov)).hi if certified else None
        ),
        # Cancel alpha in ||P_M x|| <= sqrt(2 W(0))/alpha analytically.
        # This avoids artificial blowup from the fixed dyadic enclosure floor.
        form_norm_upper_bound=sqrt(I(4 * budget / d.gap)).hi if certified else None,
        phase_edge_upper_bounds=candidates if certified else None,
        status="certified" if certified else "unavailable",
        unresolved=tuple(unresolved),
    )
