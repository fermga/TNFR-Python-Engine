"""Finite directional transit of the full two-port C9 sine preparation.

An acute reference comparison and the complete-law dissipation integral bound
the actual phase and form coordinates. Only rational geometry and inequalities
are evaluated; no reference trajectory, implicit target, or root is supplied.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q

from .._exact_time import exact_or_represented_real
from ..dynamics.relational import RelationalExchangeModel
from ..mathematics._exact_linear_algebra import (
    exact_matrix_product,
    exact_symmetric_semidefinite,
)
from ._sine_admission import _sine_model_coefficients
from .phase_cycle_geometry import PhaseCycleGeometry, _cycle_row, _derive
from .relational_sine_two_port_compatibility import _EDGES, _NODES, _affine_geometry

__all__ = ("SineTwoPortTransit", "assess_sine_two_port_transit")


@dataclass(frozen=True)
class SineTwoPortTransit:
    """Whole-window acute comparison and finite short-arc deformation.

    Actual initial forms differ from zero by at most ``form_error_radius``;
    real phase lifts differ from ``2*pi*nominal_phase_turns`` by at most
    ``phase_error_radius`` radians, independently at every node. The nominal
    common phase gauge is zero, but actual source means are only bounded.

    Candidate comparison values require the displayed energy budget and
    bootstrap premises. Actual error and directional fields remain unavailable
    when those premises fail. Phase errors use the degree-weighted norm against
    the nominal gradient flow shifted to each member's conserved phase mean.
    This implicit reference is not the compatible equilibrium. All-time capture,
    convergence, support events and formation are not certified.
    """

    form_error_radius: Q
    phase_error_radius: Q
    reference_model: RelationalExchangeModel
    geometry: PhaseCycleGeometry
    degrees: tuple[int, ...]
    invariant_weights: tuple[Q, ...]
    weighted_coordinate_mass: Q
    nominal_phase_turns: tuple[Q, ...]
    nominal_edge_turns: tuple[Q, ...]
    edge_integer_offsets: tuple[int, ...]
    nominal_uncentered_phase_mean_turns: Q
    named_cycles: tuple[tuple[int, ...], ...]
    named_cycle_periods: tuple[int, ...]
    laplacian: tuple[tuple[Q, ...], ...]
    normalized_gap_slack_matrix: tuple[tuple[Q, ...], ...]
    normalized_upper_slack_matrix: tuple[tuple[Q, ...], ...]
    initial_form_mean_bounds: tuple[Q, Q]
    initial_phase_mean_bounds: tuple[Q, Q]
    initial_excess_storage_upper_bound: Q
    energy_budget_margin: Q
    energy_budget_admitted: bool
    initial_scaled_form_norm_upper_bound: Q
    initial_joint_error_upper_bound: Q
    joint_error_candidate: Q
    scaled_form_norm_candidate: Q
    phase_error_candidate: Q
    acute_margin_candidate: Q
    joint_error_upper_bound: Q | None
    scaled_form_norm_upper_bound: Q | None
    phase_error_upper_bound: Q | None
    relative_form_norm_upper_bound: Q | None
    whole_window_acute_margin_lower_bound: Q | None
    short_arc_change_lower_bound: Q | None
    direction_margin: Q | None
    error_certified: bool
    whole_window_acute_certified: bool
    direction_certified: bool
    status: str
    unavailable_reasons: tuple[str, ...]
    direction_limitations: tuple[str, ...]
    classes: tuple[int, int] = (2, 1)
    capacity: tuple[Q, ...] = (Q(1),) * 18
    nominal_epi: tuple[Q, ...] = (Q(0),) * 18
    slow_time: Q = Q(1, 4)
    scaled_time_pi_squared_coefficient: Q = Q(1023**2, 4)
    original_time_pi_squared_coefficient: Q = Q(261888)
    gamma_bounds: tuple[Q, Q] = (Q(1, 3216), Q(1, 3069))
    feedback_strength_upper_bound: Q = Q(1, 9000000)
    normalized_gap_lower_bound: Q = Q(1, 90)
    normalized_rate_upper_bound: Q = Q(2)
    source_error_norm_multiplier: Q = Q(7)
    nominal_contact_storage_upper_bound: Q = Q(10, 81)
    energy_budget: Q = Q(1, 8)
    reference_forcing_norm_upper_bound: Q = Q(5, 12)
    reference_acute_margin_lower_bound: Q = Q(1, 16)
    integral_comparison_error_upper_bound: Q = Q(3, 800000)
    scaled_form_loop_gain_upper_bound: Q = Q(1, 50000)
    reference_short_arc_change_lower_bound: Q = Q(53, 972) - Q(25, 1152)
    port_dual_norm_upper_bound: Q = Q(5, 6)
    short_arc_change_threshold: Q = Q(1, 32)
    law: str = "normalized_sine_reciprocal_exchange"
    clock: str = "sigma=gamma^2*tau; tau=e*t; gamma=1/(1023*pi)"
    arithmetic_method: str = (
        "exact_rational_geometry_and_acute_energy_integral_comparison"
    )
    scope: tuple[str, ...] = (
        "fixed_unit_two_C9_support_with_contacts_D0_R0_and_D1_R1",
        "midpoint_aligned_undeformed_two_one_twists_and_zero_nominal_form",
        "independent_nodewise_source_errors_retained_without_mean_zero_constraints",
        "same_complete_positive_loss_sine_law_without_forcing_events_or_resets",
        "normalized_diffusion_gap_and_upper_bound_verified_by_exact_PSD",
        "reference_is_implicit_gradient_flow_from_nominal_source_not_target_equilibrium",
        "actual_phase_errors_use_reference_shifted_to_each_member_conserved_phase_mean",
        "actual_dissipation_integral_and_acute_comparison_preserve_original_form",
        "candidate_bounds_require_strict_energy_and_positive_acute_bootstrap_margins",
        "whole_window_covers_slow_time_zero_through_one_quarter_only",
        "actual_donor_short_arc_contracts_and_receiver_short_arc_expands",
        "directional_change_is_relative_to_each_actual_member_initial_short_arcs",
        "no_trajectory_root_trigonometric_evaluation_or_archived_producer_call",
        "no_endpoint_capture_all_time_retention_convergence_or_physical_identification",
    )

    def to_dict(self):
        from ..sdk.relational_reports import _project

        return {"schema": "tnfr.sine-two-port-transit.v1", "report": _project(self)}


def assess_sine_two_port_transit(*, form_error_radius, phase_error_radius):
    """Bound a fixed finite transit from independent preparation budgets.

    Both mandatory radii are nonnegative and admitted before constructing
    geometry. Exact rational inputs remain exact; other real values use the
    shared represented-real contract. Phase radii use radians. No maximum
    input radius is imposed, but the theorem requires strict excess-storage
    budget below 1/8 and a positive acute bootstrap margin.

    The proof in ``SINE_TWO_PORT_TRANSIT.md`` uses slow horizon 1/4 and
    elementary analytic constants. A passing error certificate controls the
    full interval, including the initial fast transient. Phase errors compare
    against each member's mean-matched nominal gradient reference in the degree
    metric. The separate direction flag requires both actual endpoint short-arc
    changes to exceed 1/32 radian.
    Failed sufficient margins are unavailable, not failed dynamics.
    """
    radii = tuple(
        exact_or_represented_real(value, label)
        for value, label in (
            (form_error_radius, "form_error_radius"),
            (phase_error_radius, "phase_error_radius"),
        )
    )
    if any(value < 0 for value in radii):
        raise ValueError("form_error_radius and phase_error_radius must be nonnegative")
    rx, rt = radii
    model = RelationalExchangeModel(
        1, epi_weight=Q(1023, 1024), phase_weight=Q(1, 1024), phase_domain="regular"
    )
    _sine_model_coefficients(model, positive_loss=True)
    geometry = _derive(_NODES, _EDGES)
    degrees, raw_mean, nodal, edges, offsets = _affine_geometry((2, 1), geometry)

    def affine(row):
        return row[0] + row[1] * Q(2, 9) + row[2] * Q(1, 9)

    phases, edge_turns = tuple(map(affine, nodal)), tuple(map(affine, edges))
    mass = Q(sum(degrees))
    incidence = tuple(tuple(map(Q, row)) for row in geometry.incidence)
    laplacian = exact_matrix_product(incidence, tuple(zip(*incidence)))
    lower_slack = tuple(
        tuple(
            laplacian[i][j]
            - Q(1, 90)
            * (Q(degrees[i] * int(i == j)) - Q(degrees[i] * degrees[j]) / mass)
            for j in _NODES
        )
        for i in _NODES
    )
    upper_slack = tuple(
        tuple(2 * degrees[i] * int(i == j) - laplacian[i][j] for j in _NODES)
        for i in _NODES
    )
    if not all(map(exact_symmetric_semidefinite, (lower_slack, upper_slack))):
        raise ArithmeticError("the fixed normalized diffusion bounds failed")
    cycles = (tuple(range(9)), tuple(range(9, 18)), (0, 9, 10, 1))
    indices = {edge: i for i, edge in enumerate(geometry.edges)}
    periods = tuple(
        sum(
            (sign * turn for sign, turn in zip(_cycle_row(cycle, indices), edge_turns)),
            Q(0),
        )
        for cycle in cycles
    )
    if periods != (2, 1, 0) or any(abs(value) >= Q(1, 4) for value in edge_turns):
        raise ArithmeticError("the fixed nominal acute preparation is inconsistent")
    energy = Q(10, 81) + 40 * rt + 40 * rx**2
    energy_margin = Q(1, 8) - energy
    energy_admitted = energy_margin > 0
    z0 = 7 * rx / 3069
    v0 = 7 * rt + z0
    joint = v0 + Q(3, 800000)
    scaled_form = (z0 + Q(1, 100000) * (Q(5, 12) + 2 * joint)) / (1 - Q(1, 50000))
    phase_error = joint + scaled_form
    acute_margin = Q(1, 16) - phase_error
    error_certified = energy_admitted and acute_margin > 0
    change = (
        Q(53, 972) - Q(25, 1152) - Q(5, 6) * phase_error - 2 * rt
        if error_certified
        else None
    )
    direction_margin = change - Q(1, 32) if change is not None else None
    directional = direction_margin is not None and direction_margin > 0
    reasons = tuple(
        reason
        for passed, reason in (
            (energy_admitted, "strict_initial_excess_storage_budget_not_certified"),
            (acute_margin > 0, "positive_acute_bootstrap_margin_not_certified"),
        )
        if not passed
    )
    limitations = (
        ("actual_error_bound_unavailable",)
        if not error_certified
        else () if directional else ("strict_short_arc_change_threshold_not_certified",)
    )
    status = (
        "certified_directional_transit"
        if directional
        else "transit_bound_only" if error_certified else "unavailable"
    )
    return SineTwoPortTransit(
        form_error_radius=rx,
        phase_error_radius=rt,
        reference_model=model,
        geometry=geometry,
        degrees=degrees,
        invariant_weights=tuple(map(Q, degrees)),
        weighted_coordinate_mass=mass,
        nominal_phase_turns=phases,
        nominal_edge_turns=edge_turns,
        edge_integer_offsets=offsets,
        nominal_uncentered_phase_mean_turns=affine(raw_mean),
        named_cycles=cycles,
        named_cycle_periods=tuple(map(int, periods)),
        laplacian=laplacian,
        normalized_gap_slack_matrix=lower_slack,
        normalized_upper_slack_matrix=upper_slack,
        initial_form_mean_bounds=(-rx, rx),
        initial_phase_mean_bounds=(-rt, rt),
        initial_excess_storage_upper_bound=energy,
        energy_budget_margin=energy_margin,
        energy_budget_admitted=energy_admitted,
        initial_scaled_form_norm_upper_bound=z0,
        initial_joint_error_upper_bound=v0,
        joint_error_candidate=joint,
        scaled_form_norm_candidate=scaled_form,
        phase_error_candidate=phase_error,
        acute_margin_candidate=acute_margin,
        joint_error_upper_bound=joint if error_certified else None,
        scaled_form_norm_upper_bound=scaled_form if error_certified else None,
        phase_error_upper_bound=phase_error if error_certified else None,
        relative_form_norm_upper_bound=3216 * scaled_form if error_certified else None,
        whole_window_acute_margin_lower_bound=acute_margin if error_certified else None,
        short_arc_change_lower_bound=change,
        direction_margin=direction_margin,
        error_certified=error_certified,
        whole_window_acute_certified=error_certified,
        direction_certified=directional,
        status=status,
        unavailable_reasons=reasons,
        direction_limitations=limitations,
    )
