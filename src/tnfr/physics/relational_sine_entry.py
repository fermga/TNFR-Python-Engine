"""Analytic preparation-to-entry bounds for declared complete sine laws.

The globally smooth law supplies a weighted Duhamel estimate through nonacute
passage. The resulting full-state endpoint enclosure is passed to the shared
sector capture owner without a trajectory run, fabricated observation or
equilibrium target. Supplied form information, support and coefficients remain
preparation premises.

A separate conservative reader uses a global phase-advance bound to certify
finite regional winding without damping. It does not import positive-loss
capture or acute-target conclusions. Exact initial phase-transport derivatives
and globally bounded contrast acceleration retain their separate scopes.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q

from .._exact_time import exact_or_represented_real
from ..dynamics.relational import RelationalExchangeModel
from ..mathematics._rational_interval import INTERVAL_METHOD, I, cos, pi_interval, sqrt
from ..mathematics.krylov import exact_rank
from ._sine_preparation import _sine_preparation
from .phase_cycle_geometry import PhaseCycleGeometry
from .relational_sine_comparison import (
    SineExchangeComparison,
    _validate_comparison_labels,
)
from .relational_sine_pattern import SineRelativePattern
from .relational_sine_recovery import (
    SineSectorCapture,
    _certify_sine_sector_set,
    _validate_sine_sector_capture_labels,
)
from .reversible_eigenmode_reference import _negative_exp_bounds

__all__ = (
    "SinePreparedEntry",
    "certify_sine_prepared_entry",
    "SineConservativeWindingEntry",
    "certify_sine_conservative_winding_entry",
    "SineConservativePhaseTransport",
    "analyze_sine_conservative_phase_transport",
    "SineConservativeSourceGeometry",
    "analyze_sine_conservative_source_geometry",
    "SineConservativeHandoff",
    "assess_sine_conservative_handoff",
)


def _admit_common_phase_conservative_source(source):
    """Admit the shared conservative law and exact common-phase preparation."""
    from ._sine_admission import _admit_sine_source, _sine_model_coefficients

    if not isinstance(source, SineExchangeComparison):
        raise TypeError("an exact SineExchangeComparison is required")
    admitted, edges = _admit_sine_source(source)
    if _sine_model_coefficients(admitted.reference_model) != (0, 1, 1) or any(
        value != 1 for value in admitted.capacity
    ):
        raise ValueError(
            "conservative analysis requires zero loss and unit capacity, exchange and beta"
        )
    if any(value != admitted.phase[0] for value in admitted.phase):
        raise ValueError("complete initial phase lifts must be equal")
    return admitted, edges


def _admit_conservative_cycle_source(source, cycle, *, induced_five=False):
    """Share full-source and uniform receiver admission before cycle bounds."""
    from .phase_cycle_geometry import _cycle_row
    from .relational_observations import _ordered

    admitted, edges = _admit_common_phase_conservative_source(source)
    cycle = _ordered(cycle, "cycle", limit=len(admitted.nodes) + 1)
    positions = {node: i for i, node in enumerate(admitted.nodes)}
    try:
        indices = tuple(positions[node] for node in cycle)
    except (KeyError, TypeError) as exc:
        raise ValueError("cycle nodes must belong to the complete source") from exc
    if len(indices) < 3 or len(set(indices)) != len(indices):
        raise ValueError("cycle must contain at least three distinct ordered nodes")
    try:
        _cycle_row(indices, {edge: i for i, edge in enumerate(edges)})
    except KeyError as exc:
        raise ValueError("every cycle edge must belong to the complete source") from exc
    if any(admitted.epi[i] != admitted.epi[indices[0]] for i in indices):
        raise ValueError("receiver cycle must have constant initial form")
    if induced_five:
        members = set(indices)
        if (
            len(indices) != 5
            or sum(i in members and j in members for i, j in edges) != 5
        ):
            raise ValueError("handoff requires an induced five-node cycle")
    return admitted, edges, cycle, indices


def _unit_laplacian_action(values, edges):
    """Apply the supplied unit-support Laplacian to admitted exact rows."""
    gradient = [Q(0)] * len(values)
    for i, j in edges:
        difference = values[i] - values[j]
        gradient[i] += difference
        gradient[j] -= difference
    return tuple(gradient)


def _conservative_kl_action(values, edges, degrees):
    """Apply A=K L with the complete graph degrees, K=diag(1/d_i)."""
    return tuple(
        value / degree
        for value, degree in zip(_unit_laplacian_action(values, edges), degrees)
    )


@dataclass(frozen=True)
class SineConservativeSourceGeometry:
    """Exact environmental map to an initially uniform receiver's phase rates.

    In tau=t/pi the map T=-D_R**-1 A_RQ acts on x_Q-a, where a is
    the receiver's uniform initial form. Relative rows subtract the first
    receiver row and omit its zero row. Matrix ranks describe available
    initial directions, not nonlinear reachability or a preparation selector.
    Identical row groups use full-source indices and have at least two nodes.
    """

    source: SineExchangeComparison
    receiver: tuple
    receiver_indices: tuple[int, ...]
    environment: tuple
    environment_indices: tuple[int, ...]
    receiver_form_origin: Q
    environment_form_contrasts: tuple[Q, ...]
    source_velocity_matrix: tuple[tuple[Q, ...], ...]
    relative_source_velocity_matrix: tuple[tuple[Q, ...], ...]
    source_rank: int
    relative_source_rank: int
    actual_receiver_velocity: tuple[Q, ...]
    relative_velocity: tuple[Q, ...]
    reconstruction_residuals: tuple[Q, ...]
    identical_velocity_row_groups: tuple[tuple[int, ...], ...]
    clock: str = "tau=t/pi"
    node_form_rate_bound: Q = Q(1)
    node_phase_acceleration_bound: Q = Q(2)
    scope: tuple[str, ...] = (
        "complete_normalized_sine_law_unit_capacity_zero_loss_w_beta_one",
        "complete_initial_phase_lifts_equal_and_receiver_form_uniform",
        "supplied_full_support_and_environmental_form_remain_preparation_premises",
        "full_support_degrees_not_receiver_induced_degrees",
        "relative_rank_is_initial_direction_freedom_not_nonlinear_reachability",
        "full_relative_rank_suffices_for_arbitrary_initial_relative_velocity",
        "rank_deficiency_alone_does_not_exclude_an_acute_target",
        "identical_rows_are_equal_initial_velocities_not_an_invariant_partition",
        "global_form_rate_bound_one_and_phase_acceleration_bound_two_in_tau",
        "no_trajectory_event_forcing_preparation_selection_or_physical_identification",
        "cached_source_fields_unused_export_does_not_authenticate_preparation",
    )

    def to_dict(self):
        from ..sdk.relational_reports import _project, _validate_label_groups

        _validate_comparison_labels(self.source)
        _validate_label_groups(self.receiver, self.environment)
        return {
            "schema": "tnfr.sine-conservative-source-geometry.v1",
            "report": _project(self),
        }


def analyze_sine_conservative_source_geometry(
    source, *, receiver
) -> SineConservativeSourceGeometry:
    """Rebuild the exact source map without prescribing future nodal motion.

    Every full-support environmental coordinate is retained, including zero
    columns for nodes without receiver contacts. The supplied receiver order
    sets the relative anchor. A whole-support receiver and a single receiver
    node are admitted, with explicit empty environmental or relative rows.
    """
    from .relational_observations import _ordered, _regions

    admitted, edges = _admit_common_phase_conservative_source(source)
    size = len(admitted.nodes)
    receiver = _ordered(receiver, "receiver", limit=size + 1)
    indices = _regions(admitted.nodes, (receiver,))[0]
    origin = admitted.epi[indices[0]]
    if any(admitted.epi[i] != origin for i in indices):
        raise ValueError("receiver must have constant initial form")
    members = set(indices)
    environment = tuple(i for i in range(size) if i not in members)
    support = {frozenset(edge) for edge in edges}
    matrix = tuple(
        tuple(
            -Q(int(frozenset((i, j)) in support), admitted.degrees[i])
            for j in environment
        )
        for i in indices
    )
    relative_matrix = tuple(
        tuple(value - anchor for value, anchor in zip(row, matrix[0]))
        for row in matrix[1:]
    )
    contrasts = tuple(admitted.epi[i] - origin for i in environment)
    full_velocity = _conservative_kl_action(admitted.epi, edges, admitted.degrees)
    velocity = tuple(full_velocity[i] for i in indices)
    reconstruction = tuple(
        speed - sum((value * form for value, form in zip(row, contrasts)), Q(0))
        for speed, row in zip(velocity, matrix)
    )
    groups = {}
    for index, row in zip(indices, matrix):
        groups.setdefault(row, []).append(index)
    return SineConservativeSourceGeometry(
        source=admitted,
        receiver=receiver,
        receiver_indices=indices,
        environment=tuple(admitted.nodes[i] for i in environment),
        environment_indices=environment,
        receiver_form_origin=origin,
        environment_form_contrasts=contrasts,
        source_velocity_matrix=matrix,
        relative_source_velocity_matrix=relative_matrix,
        source_rank=exact_rank(matrix),
        relative_source_rank=exact_rank(relative_matrix),
        actual_receiver_velocity=velocity,
        relative_velocity=tuple(value - velocity[0] for value in velocity[1:]),
        reconstruction_residuals=reconstruction,
        identical_velocity_row_groups=tuple(
            tuple(group) for group in groups.values() if len(group) > 1
        ),
    )


@dataclass(frozen=True)
class SineConservativePhaseTransport:
    """Exact initial derivatives and global bounds for declared phase contrasts.

    With A=K L and common initial phase, theta'=A x, theta''=0,
    x''=-A**2 x and theta'''=-A**3 x at tau=0. These derivatives do
    not provide a finite-time cubic approximation. Independently, for each
    admitted row c, theta contrast acceleration obeys the global bound
    |c theta''|<=B=sum_edges |(c A K)_i-(c A K)_j|. Hence
    |c theta(tau)-tau*c A x(0)|<=B*tau**2/2 for tau>=0. Its zero
    initial contrast follows from common phase and sum(c)=0.
    """

    source: SineExchangeComparison
    edge_indices: tuple[tuple[int, int], ...]
    contrasts: tuple[tuple[Q, ...], ...]
    initial_phase_velocity: tuple[Q, ...]
    initial_phase_acceleration: tuple[Q, ...]
    initial_phase_jerk: tuple[Q, ...]
    initial_form_acceleration: tuple[Q, ...]
    contrast_initial_velocity: tuple[Q, ...]
    contrast_initial_jerk: tuple[Q, ...]
    contrast_current_weights: tuple[tuple[Q, ...], ...]
    contrast_edge_current_coefficients: tuple[tuple[Q, ...], ...]
    contrast_acceleration_bounds: tuple[Q, ...]
    contrast_quadratic_remainder_coefficients: tuple[Q, ...]
    clock: str = "tau=t/pi"
    arithmetic_method: str = "exact_rational_unit_graph_algebra"
    scope: tuple[str, ...] = (
        "complete_normalized_sine_law_unit_capacity_zero_loss_w_beta_one",
        "scaled_clock_tau_equals_original_structural_time_divided_by_pi",
        "exact_common_initial_phase_and_declared_signed_form_preparation",
        "contrasts_are_nonzero_exact_zero_sum_rows_in_complete_source_node_order",
        "initial_derivatives_use_full_support_normalized_laplacian_powers",
        "global_acceleration_bound_retains_edge_current_correlations",
        "local_jerk_is_not_a_finite_time_cubic_enclosure_or_formation_certificate",
        "quadratic_remainder_bound_is_global_and_does_not_use_the_local_jerk",
        "no_spectral_truncation_native_law_transfer_or_uncertainty_midpoint",
        "no_trajectory_forcing_event_constitutive_selection_or_physical_identification",
        "cached_source_fields_unused_export_does_not_authenticate_preparation",
    )

    def to_dict(self):
        from ..sdk.relational_reports import _project

        _validate_comparison_labels(self.source)
        return {
            "schema": "tnfr.sine-conservative-phase-transport.v1",
            "report": _project(self),
        }


def analyze_sine_conservative_phase_transport(
    source, *, contrasts
) -> SineConservativePhaseTransport:
    """Retain exact initial transport and all-time contrast acceleration bounds.

    Every contrast is a nonzero signed row with exact zero sum and one entry
    per full-source node. Primitive state/law admission precedes graph algebra;
    cached pressure, rates, gradients and storage are not consumed. No time,
    target winding, trajectory or automatic preparation is selected here.
    """
    from .relational_observations import _ordered

    admitted, edges = _admit_common_phase_conservative_source(source)
    size = len(admitted.nodes)
    rows = _ordered(contrasts, "contrasts")
    if not rows:
        raise ValueError("contrasts must contain at least one nonzero row")
    normalized = []
    for index, row in enumerate(rows):
        row = _ordered(row, f"contrasts[{index}]", limit=size + 1)
        if len(row) != size:
            raise ValueError("each contrast requires one scalar per source node")
        row = tuple(
            exact_or_represented_real(value, f"contrasts[{index}][{j}]")
            for j, value in enumerate(row)
        )
        if not any(row) or sum(row, Q(0)) != 0:
            raise ValueError("each contrast must be nonzero with exact zero sum")
        normalized.append(row)
    contrasts = tuple(normalized)
    degrees = admitted.degrees
    velocity = _conservative_kl_action(admitted.epi, edges, degrees)
    acceleration = tuple(
        -value for value in _conservative_kl_action(velocity, edges, degrees)
    )
    jerk = _conservative_kl_action(acceleration, edges, degrees)
    weights, coefficients, bounds = [], [], []
    for row in contrasts:
        # c A = (L K c^T)^T since the unit Laplacian is symmetric. Retain
        # both degree factors: c A K, not the unnormalized c L row.
        current_weights = tuple(
            value / degree
            for value, degree in zip(
                _unit_laplacian_action(
                    tuple(value / degree for value, degree in zip(row, degrees)),
                    edges,
                ),
                degrees,
            )
        )
        edge_coefficients = tuple(
            current_weights[i] - current_weights[j] for i, j in edges
        )
        weights.append(current_weights)
        coefficients.append(edge_coefficients)
        bounds.append(sum(map(abs, edge_coefficients), Q(0)))

    def projected(values):
        return tuple(
            sum((coefficient * value for coefficient, value in zip(row, values)), Q(0))
            for row in contrasts
        )

    return SineConservativePhaseTransport(
        source=admitted,
        edge_indices=edges,
        contrasts=contrasts,
        initial_phase_velocity=velocity,
        initial_phase_acceleration=(Q(0),) * size,
        initial_phase_jerk=jerk,
        initial_form_acceleration=acceleration,
        contrast_initial_velocity=projected(velocity),
        contrast_initial_jerk=projected(jerk),
        contrast_current_weights=tuple(weights),
        contrast_edge_current_coefficients=tuple(coefficients),
        contrast_acceleration_bounds=tuple(bounds),
        contrast_quadratic_remainder_coefficients=tuple(value / 2 for value in bounds),
    )


@dataclass(frozen=True)
class SineConservativeWindingEntry:
    """Whole-window phase enclosure from a declared environmental preparation.

    The nominal receiver has constant initial form and the nominal full network
    has a common phase. An optional uniform error covers every initial form and
    phase independently. Fixed branch offsets determine whole-window winding.
    The independent entry box also retains that winding for its stated future
    duration. An unavailable test proves no failure of the actual dynamics.
    ``initial_total_storage`` is the nominal center's storage; the uncertain
    preparation uses ``initial_storage_bounds``. Cached rates are not evidence.
    """

    source: SineExchangeComparison
    cycle: tuple
    cycle_indices: tuple[int, ...]
    scaled_window: tuple[Q, Q]
    clock: str
    original_time_window_bounds: tuple[I, I]
    initial_phase_velocity: tuple[Q, ...]
    node_form_remainder_bound: Q
    node_phase_remainder_bound: Q
    cycle_gap_remainder_bound: Q
    whole_window_form_bounds: tuple[I, ...]
    whole_window_phase_bounds: tuple[I, ...]
    cycle_raw_gap_bounds: tuple[I, ...]
    cycle_principal_gap_bounds: tuple[I, ...]
    edge_turn_offsets: tuple[int, ...]
    branch_margin_lower_bound: Q
    initial_winding: int | None
    declared_winding: int
    certified_winding: int | None
    acquisition_certified: bool
    initial_total_storage: Q
    status: str
    unresolved: tuple[str, ...]
    arithmetic_method: str = INTERVAL_METHOD
    scope: tuple[str, ...] = (
        "complete_normalized_sine_law_unit_capacity_zero_loss_w_beta_one",
        "scaled_clock_tau_equals_original_structural_time_divided_by_pi",
        "globally_smooth_complete_rows_not_a_native_Arg_continuation",
        "nominal_common_phase_and_uniform_receiver_form_with_optional_full_state_source_error",
        "supplied_environmental_form_and_support_are_not_autonomously_generated",
        "global_form_rate_bound_one_and_phase_acceleration_bound_two",
        "correlated_raw_cycle_gaps_have_exact_zero_sum_despite_outer_interval_boxes",
        "one_fixed_integer_branch_per_edge_for_every_time_in_the_declared_window",
        "winding_identity_need_not_be_acute_or_an_equilibrium",
        "optional_strict_acute_margin_uses_the_same_whole_window_principal_bounds",
        "finite_retention_not_attractive_capture_or_indefinite_maintenance",
        "entry_box_contains_every_source_member_at_the_common_declared_entry_time",
        "effective_entry_radii_include_outward_materialization_of_the_reported_box",
        "source_flow_bounds_and_independent_materialized_entry_box_bounds_are_distinct",
        "future_acute_window_applies_to_every_entry_box_member_independently",
        "initial_total_storage_is_nominal_center_only_initial_storage_bounds_cover_source_box",
        "initial_storage_bounds_do_not_constrain_every_independent_entry_box_member",
        "source_box_members_retain_their_own_conserved_means_and_environmental_state",
        "entry_box_is_an_acute_passage_set_not_an_invariant_phase_offset_family",
        "no_trajectory_samples_forcing_events_or_physical_identification",
        "cached_source_fields_unused_export_does_not_authenticate_preparation",
    )
    acute_margin_lower_bound: Q | None = None
    acute_acquisition_certified: bool = False
    source_error_bound: Q = Q(0)
    initial_storage_bounds: I | None = None
    source_initial_zero_winding_certified: bool = True
    source_initial_branch_margin_lower_bound: Q | None = None
    entry_form_bounds: tuple[I, ...] | None = None
    entry_phase_bounds: tuple[I, ...] | None = None
    entry_form_radius: Q | None = None
    entry_phase_radius: Q | None = None
    scaled_retention_duration: Q | None = None
    entry_box_contains_source_flow: bool = False
    entry_box_acute_retention_certified: bool = False
    analytic_entry_form_radius: Q | None = None
    analytic_entry_phase_radius: Q | None = None
    entry_box_form_remainder_bound: Q | None = None
    entry_box_phase_remainder_bound: Q | None = None
    entry_box_cycle_principal_gap_bounds: tuple[I, ...] | None = None
    entry_box_branch_margin_lower_bound: Q | None = None
    entry_box_acute_margin_lower_bound: Q | None = None

    def to_dict(self):
        """Project exact preparation and whole-time evidence with label admission."""
        from ..sdk.relational_reports import _project, _validate_label_groups

        _validate_comparison_labels(self.source)
        _validate_label_groups(self.cycle)
        return {
            "schema": "tnfr.sine-conservative-winding-entry.v1",
            "report": _project(self),
        }


def certify_sine_conservative_winding_entry(
    source, *, cycle, scaled_window, edge_turn_offsets, source_error_bound=0
) -> SineConservativeWindingEntry:
    """Bound finite regional winding acquisition under conservative exchange.

    In tau=t/pi, x'=K S(theta), theta'=K L x with unit capacities, w=beta=1
    and zero loss. Since |x_i'|<=1 and ||K L||_infinity=2, every node obeys
    |theta_i(tau)-theta_i(0)-tau*(K L x(0))_i|<=tau^2 globally. Two node
    remainders enclose a correlated oriented cycle gap. The caller supplies
    integer wrap offsets; strict whole-window branch inclusion, not rounded
    winding at a sample, certifies their period.

    Nominal initial phase lifts must be equal and receiver forms constant.
    ``source_error_bound=eps>=0`` admits independent form and phase errors at
    every full-support node, with phase errors in radians. Global form speed
    and phase acceleration give form error ``t+eps`` and phase error
    ``t**2+eps*(1+2*t)`` relative to the nominal affine centers. No exponential
    approximation or short-time evaluation domain is needed.

    Outward materialization can slightly enlarge the entry box. Its reported
    effective radii include that enlargement, and the separate ``entry_box_*``
    continuation bounds propagate those actual radii for ``end-start``. The
    original whole-window bounds continue to enclose the source flow. The
    source box has zero winding only
    if ``2*eps<pi``;
    otherwise acquisition remains unavailable even though an entry enclosure
    still exists. Positive window endpoints use tau, not original time t.
    A zero period does not establish nonzero acquisition. This is a finite
    passage certificate, not entry into an invariant moving-pattern family.
    """
    from .relational_observations import _ordered

    admitted, edges, cycle, indices = _admit_conservative_cycle_source(source, cycle)
    window = _ordered(scaled_window, "scaled_window", limit=3)
    if len(window) != 2:
        raise ValueError("scaled_window requires two ordered endpoints")
    start, end = (exact_or_represented_real(value, "scaled_window") for value in window)
    if not 0 < start < end:
        raise ValueError("scaled_window requires 0 < start < end in tau=t/pi")
    error = exact_or_represented_real(source_error_bound, "source_error_bound")
    if error < 0:
        raise ValueError("source_error_bound must be nonnegative")
    offsets = _ordered(edge_turn_offsets, "edge_turn_offsets", limit=len(indices) + 1)
    if len(offsets) != len(indices) or any(type(value) is not int for value in offsets):
        raise ValueError(
            "edge_turn_offsets requires one nonboolean integer per cycle edge"
        )
    velocity = _conservative_kl_action(admitted.epi, edges, admitted.degrees)
    analytic_entry_form_radius = start + error
    analytic_entry_phase_radius = start**2 + error * (1 + 2 * start)
    entry_phase_centers = tuple(
        value + start * speed for value, speed in zip(admitted.phase, velocity)
    )
    entry_forms = tuple(
        I(value - analytic_entry_form_radius, value + analytic_entry_form_radius)
        for value in admitted.epi
    )
    entry_phases = tuple(
        I(value - analytic_entry_phase_radius, value + analytic_entry_phase_radius)
        for value in entry_phase_centers
    )
    # A rational center/radius can have non-dyadic endpoints. The interval
    # kernel rounds those outwards; every member of that stored box, including
    # the rounded fringe, must satisfy the independent continuation guarantee.
    entry_form_radius = max(
        max(value - interval.lo, interval.hi - value)
        for value, interval in zip(admitted.epi, entry_forms)
    )
    entry_phase_radius = max(
        max(value - interval.lo, interval.hi - value)
        for value, interval in zip(entry_phase_centers, entry_phases)
    )
    duration = end - start
    entry_box_form_remainder = entry_form_radius + duration
    entry_box_phase_remainder = (
        entry_phase_radius + 2 * duration * entry_form_radius + duration**2
    )
    remainder = end**2 + error * (1 + 2 * end)

    def affine_enclosure(origin, slope, radius):
        first, last = origin + start * slope, origin + end * slope
        return I(min(first, last) - radius, max(first, last) + radius)

    raw_gaps = tuple(
        affine_enclosure(Q(0), velocity[j] - velocity[i], 2 * remainder)
        for i, j in zip(indices, indices[1:] + indices[:1])
    )
    pi = pi_interval()
    initial_margin = pi.lo - 2 * error
    initial_zero = initial_margin > 0
    principal = tuple(gap - 2 * offset * pi for gap, offset in zip(raw_gaps, offsets))
    margin = min(pi.lo - gap.abs_max for gap in principal)
    acute_margin = min(pi.lo / 2 - gap.abs_max for gap in principal)
    entry_box_principal = tuple(
        affine_enclosure(Q(0), velocity[j] - velocity[i], 2 * entry_box_phase_remainder)
        - 2 * offset * pi
        for i, j, offset in zip(indices, indices[1:] + indices[:1], offsets)
    )
    entry_box_margin = min(pi.lo - gap.abs_max for gap in entry_box_principal)
    entry_box_acute_margin = min(pi.lo / 2 - gap.abs_max for gap in entry_box_principal)
    declared = -sum(offsets)
    certified = declared if margin > 0 else None
    acquired = initial_zero and certified is not None and certified != 0
    retained_acute = declared != 0 and entry_box_acute_margin > 0
    initial_phase_cost = 1 - cos(I(-2 * error, 2 * error))
    initial_storage = sum(
        (
            I(
                admitted.epi[j] - admitted.epi[i] - 2 * error,
                admitted.epi[j] - admitted.epi[i] + 2 * error,
            )
            ** 2
            / 2
            + initial_phase_cost
            for i, j in edges
        ),
        I(0),
    )
    unresolved = tuple(
        reason
        for failed, reason in (
            (not initial_zero, "whole_source_box_zero_winding_not_certified"),
            (margin <= 0, "whole_window_strict_branch_inclusion_unresolved"),
            (declared == 0, "declared_sector_does_not_establish_nonzero_acquisition"),
        )
        if failed
    )
    return SineConservativeWindingEntry(
        source=admitted,
        cycle=cycle,
        cycle_indices=indices,
        scaled_window=(start, end),
        clock="tau=t/pi",
        original_time_window_bounds=(pi * start, pi * end),
        initial_phase_velocity=velocity,
        node_form_remainder_bound=end + error,
        node_phase_remainder_bound=remainder,
        cycle_gap_remainder_bound=2 * remainder,
        whole_window_form_bounds=tuple(
            I(value - end - error, value + end + error) for value in admitted.epi
        ),
        whole_window_phase_bounds=tuple(
            affine_enclosure(value, speed, remainder)
            for value, speed in zip(admitted.phase, velocity)
        ),
        cycle_raw_gap_bounds=raw_gaps,
        cycle_principal_gap_bounds=principal,
        edge_turn_offsets=offsets,
        branch_margin_lower_bound=margin,
        initial_winding=0 if initial_zero else None,
        declared_winding=declared,
        certified_winding=certified,
        acquisition_certified=acquired,
        initial_total_storage=sum(
            ((admitted.epi[j] - admitted.epi[i]) ** 2 / 2 for i, j in edges), Q(0)
        ),
        status="certified_finite_acquisition" if acquired else "unavailable",
        unresolved=unresolved,
        acute_margin_lower_bound=acute_margin,
        acute_acquisition_certified=acquired and acute_margin > 0,
        source_error_bound=error,
        initial_storage_bounds=initial_storage,
        source_initial_zero_winding_certified=initial_zero,
        source_initial_branch_margin_lower_bound=initial_margin,
        entry_form_bounds=entry_forms,
        entry_phase_bounds=entry_phases,
        entry_form_radius=entry_form_radius,
        entry_phase_radius=entry_phase_radius,
        scaled_retention_duration=duration,
        entry_box_contains_source_flow=True,
        entry_box_acute_retention_certified=retained_acute,
        analytic_entry_form_radius=analytic_entry_form_radius,
        analytic_entry_phase_radius=analytic_entry_phase_radius,
        entry_box_form_remainder_bound=entry_box_form_remainder,
        entry_box_phase_remainder_bound=entry_box_phase_remainder,
        entry_box_cycle_principal_gap_bounds=entry_box_principal,
        entry_box_branch_margin_lower_bound=entry_box_margin,
        entry_box_acute_margin_lower_bound=entry_box_acute_margin,
    )


@dataclass(frozen=True)
class SineConservativeHandoff:
    """A low-storage acute entry followed by a certified finite acute exit.

    The receiver has an exact affine initial velocity profile along its first
    four oriented edges. Analytic full-law bounds establish separate entry,
    storage and exit claims. Their conjunction prevents an acute retention
    certificate through the exit deadline, despite subbarrier entry storage.
    The positive work lower bound is attained by a first exit; it is not a
    claim about net work at the deadline or permanent loss of winding.
    """

    source: SineExchangeComparison
    cycle: tuple
    cycle_indices: tuple[int, ...]
    orientation: int
    omega: Q
    initial_phase_velocity: tuple[Q, ...]
    scaled_entry_time_bounds: I
    scaled_exit_time: Q
    original_entry_time_bounds: I
    original_exit_time_bounds: I
    entry_cycle_gap_error_bound: Q
    entry_relative_phase_rate_bounds: tuple[I, ...]
    entry_form_storage_upper_bound: Q
    target_phase_storage_bounds: I
    entry_phase_storage_upper_bound: Q
    entry_storage_upper_bound: Q
    boundary_storage_bounds: I
    entry_acute_margin_lower_bound: Q
    entry_subbarrier_margin_lower_bound: Q
    exit_oriented_gap_bounds: I
    entry_acute_certified: bool
    entry_subbarrier_certified: bool
    forced_exit_certified: bool
    handoff_obstruction_certified: bool
    unavoidable_boundary_work_lower_bound: Q | None
    status: str
    unresolved: tuple[str, ...]
    clock: str = "tau=t/pi"
    arithmetic_method: str = INTERVAL_METHOD
    scope: tuple[str, ...] = (
        "same_complete_conservative_unit_capacity_sine_law_and_unchanged_full_support",
        "globally_equal_initial_phase_uniform_receiver_form_and_induced_C5",
        "exact_nonzero_affine_receiver_velocity_profile_is_a_preparation_premise",
        "global_form_speed_one_and_phase_acceleration_two_bound_full_nonlinear_flow",
        "entry_relative_phase_rates_follow_oriented_cycle_edges_in_scaled_tau_clock",
        "entry_at_exact_tau_2pi_over_5omega_not_a_floating_time_sample",
        "correlated_zero_sum_phase_errors_cancel_linear_target_storage_term",
        "entry_storage_bound_includes_receiver_form_and_phase",
        "strict_forward_gap_between_pi_over_two_and_pi_certifies_later_nonacuteness",
        "combined_result_excludes_continuous_acute_retention_through_declared_deadline",
        "positive_running_net_work_is_attained_by_first_exit_not_necessarily_at_deadline",
        "no_damping_event_environment_freeze_work_schedule_or_new_trajectory",
        "failed_sufficient_predicates_are_unavailable_not_dynamics_failure",
        "no_general_formation_obstruction_indefinite_maintenance_or_physical_identification",
        "cached_source_fields_unused_export_does_not_authenticate_preparation",
    )

    def to_dict(self):
        from ..sdk.relational_reports import _project, _validate_label_groups

        _validate_comparison_labels(self.source)
        _validate_label_groups(self.cycle)
        return {
            "schema": "tnfr.sine-conservative-handoff.v1",
            "report": _project(self),
        }


def assess_sine_conservative_handoff(source, *, cycle) -> SineConservativeHandoff:
    """Assess a prospective low-storage entry and finite acute-exit obstruction.

    Let omega be the absolute common initial receiver velocity difference on
    the first four cycle edges. At tau*=2pi/(5omega), all five principal gaps
    differ from the oriented 2pi/5 twist by at most 2tau*^2 with zero-sum
    errors. Thus F_R<=10tau*^2 and V_R<=V5+10tau*^4. At 5/(3omega) a forward
    oriented raw gap lies in 5/3+[-2tau^2,2tau^2]. Strict inclusion in
    (pi/2,pi) proves exit independently of the initial storage test.

    Supplied source geometry and all environmental coordinates remain live.
    The rational bounds, rather than a hard-coded omega cutoff, decide each
    claim. The common proof admits every omega>=64 and may admit smaller ones.
    """
    admitted, edges, cycle, indices = _admit_conservative_cycle_source(
        source, cycle, induced_five=True
    )
    velocity = _conservative_kl_action(admitted.epi, edges, admitted.degrees)
    differences = tuple(velocity[j] - velocity[i] for i, j in zip(indices, indices[1:]))
    step = differences[0]
    if not step or any(value != step for value in differences[1:]):
        raise ValueError(
            "handoff requires an exact nonzero affine receiver velocity profile"
        )
    omega = abs(step)
    orientation = 1 if step > 0 else -1
    pi = pi_interval()
    entry_time = 2 * pi / (5 * omega)
    exit_time = Q(5, 3) / omega
    entry_square = entry_time**2
    error = 2 * entry_square.hi
    rate_error = 4 * entry_time.hi
    entry_relative_rates = tuple(
        I(
            velocity[j] - velocity[i] - rate_error,
            velocity[j] - velocity[i] + rate_error,
        )
        for i, j in zip(indices, indices[1:] + indices[:1])
    )
    form_upper = 10 * entry_square.hi
    phase_minimum = 5 * (1 - cos(2 * pi / 5))
    phase_upper = phase_minimum.hi + 10 * (entry_square**2).hi
    storage_upper = form_upper + phase_upper
    barrier = 5 - 4 * cos(3 * pi / 8)
    acute_margin = pi.lo / 10 - error
    subbarrier_margin = barrier.lo - storage_upper
    exit_error = 2 * exit_time**2
    exit_gap = I(Q(5, 3) - exit_error, Q(5, 3) + exit_error)
    entry_acute = acute_margin > 0
    subbarrier = subbarrier_margin > 0
    forced_exit = exit_gap.lo > pi.hi / 2 and exit_gap.hi < pi.lo
    ordered = entry_time.hi < exit_time
    combined = entry_acute and subbarrier and forced_exit and ordered
    unresolved = tuple(
        reason
        for passed, reason in (
            (entry_acute, "strict_acute_entry_margin_unresolved"),
            (subbarrier, "strict_subbarrier_entry_storage_unresolved"),
            (forced_exit, "strict_nonacute_forward_gap_unresolved"),
            (ordered, "entry_before_exit_time_unresolved"),
        )
        if not passed
    )
    return SineConservativeHandoff(
        source=admitted,
        cycle=cycle,
        cycle_indices=indices,
        orientation=orientation,
        omega=omega,
        initial_phase_velocity=velocity,
        scaled_entry_time_bounds=entry_time,
        scaled_exit_time=exit_time,
        original_entry_time_bounds=pi * entry_time,
        original_exit_time_bounds=pi * exit_time,
        entry_cycle_gap_error_bound=error,
        entry_relative_phase_rate_bounds=entry_relative_rates,
        entry_form_storage_upper_bound=form_upper,
        target_phase_storage_bounds=phase_minimum,
        entry_phase_storage_upper_bound=phase_upper,
        entry_storage_upper_bound=storage_upper,
        boundary_storage_bounds=barrier,
        entry_acute_margin_lower_bound=acute_margin,
        entry_subbarrier_margin_lower_bound=subbarrier_margin,
        exit_oriented_gap_bounds=exit_gap,
        entry_acute_certified=entry_acute,
        entry_subbarrier_certified=subbarrier,
        forced_exit_certified=forced_exit,
        handoff_obstruction_certified=combined,
        unavoidable_boundary_work_lower_bound=subbarrier_margin if combined else None,
        status="certified_handoff_obstruction" if combined else "unavailable",
        unresolved=unresolved,
    )


@dataclass(frozen=True)
class SinePreparedEntry:
    """Conditional same-law acquisition from a complete preparation set.

    The node intervals are rectangular outer bounds. Tighter edge intervals
    retain the same weighted-norm correlation and each member's conserved
    means; they
    need not contain every point of the larger Cartesian product of node
    intervals. Capture is proved for the enclosed actual endpoint set.
    Pattern endpoints are centered separately by each member's conserved
    means; their absolute endpoints and common origins remain unavailable.
    ``phase_radius_upper_bound`` bounds the dynamic Duhamel correction only.
    Node and edge phase intervals additionally include the initial form and
    phase residual errors.
    """

    source: SineExchangeComparison | SineRelativePattern
    reference_model: RelationalExchangeModel
    geometry: PhaseCycleGeometry
    scaled_time: Q
    horizon: Q
    coefficient_ratio: Q
    weighted_form_mean: Q | None
    common_initial_phase: Q | None
    initial_form_storage: Q | None
    initial_cycle_periods: tuple[int, ...] | None
    mobility: tuple[Q, ...]
    metric_weights: tuple[Q, ...]
    weighted_gap_lower_bound: Q
    exponential_decay_bounds: I
    form_to_phase_scale_bounds: I
    feedback_strength_bounds: I
    scaled_initial_form_bounds: tuple[I, ...]
    scaled_initial_norm_bounds: I
    forcing_norm_bounds: I
    scaled_form_radius_upper_bound: Q
    phase_radius_upper_bound: Q
    endpoint_form_bounds: tuple[I, ...] | None
    endpoint_phase_bounds: tuple[I, ...] | None
    endpoint_form_edge_gap_bounds: tuple[I, ...]
    endpoint_phase_edge_gap_bounds: tuple[I, ...]
    capture: SineSectorCapture
    hypothesis_failures: tuple[str, ...]
    unresolved_conditions: tuple[str, ...]
    status: str
    arithmetic_method: str = INTERVAL_METHOD
    scope: tuple[str, ...] = (
        "exact_phase_flat_source_or_original_complete_correlated_residual_preparation_set",
        "same_complete_normalized_sine_law_positive_held_capacity_and_loss_throughout",
        "scaled_clock_tau_equals_epi_weight_times_structural_time_all_rows_transformed",
        "full_support_weighted_mean_and_exact_rational_reversible_quotient_gap",
        "global_sine_current_bound_and_weighted_Duhamel_estimate_allow_nonacute_transit",
        "shared_rational_negative_exponential_with_exponent_at_most_4096",
        "full_node_endpoint_boxes_and_tighter_correlated_weighted_norm_edge_bounds",
        "each_actual_trajectory_retains_its_own_initial_weighted_form_and_phase_means",
        "unknown_common_origins_make_absolute_pattern_endpoints_unavailable",
        "weighted_centering_contracts_residual_norm_and_retains_node_and_edge_error_bounds",
        "node_box_is_an_outer_projection_not_an_independent_initial_observation_set",
        "shared_all_face_total_storage_capture_without_supplied_equilibrium_coordinates",
        "acquisition_requires_a_nonzero_period_after_initial_zero_phase_periods",
        "supplied_coefficients_time_and_form_profile_are_not_retuned_or_optimized",
        "unavailable_is_not_impossibility_and_capture_alone_is_not_nonzero_acquisition",
        "source_dataclasses_and_projection_do_not_authenticate_capture_provenance",
        "no_solver_trajectory_forcing_event_autonomous_preparation_or_physical_identity",
    )
    centered_endpoint_form_bounds: tuple[I, ...] = ()
    centered_endpoint_phase_bounds: tuple[I, ...] = ()
    endpoint_coordinate_scope: str = "exact_source_absolute_and_centered_endpoints"
    mean_scope: str = "exact_captured_conserved_weighted_means"
    nominal_weighted_form_mean: Q | None = None
    nominal_weighted_phase_mean: Q | None = None
    form_error_bounds: tuple[Q, ...] = ()
    phase_error_bounds: tuple[Q, ...] = ()
    centered_form_error_bounds: tuple[Q, ...] = ()
    centered_phase_error_bounds: tuple[Q, ...] = ()
    nominal_scaled_initial_norm_bounds: I | None = None
    scaled_initial_error_norm_upper_bound: Q | None = None
    initial_form_storage_bounds: I | None = None
    initial_phase_storage_bounds: I | None = None
    initial_storage_bounds: I | None = None
    initial_phase_edge_gap_bounds: tuple[I, ...] = ()
    initial_acute_margin_bounds: tuple[I, ...] = ()
    initial_zero_winding_certified: bool = False

    @property
    def admitted(self):
        """Return whether the reported acquisition conditions passed."""
        return self.status == "admitted"

    def to_dict(self):
        """Validate nested labels and project the retained analytic evidence."""
        from ..sdk.relational_reports import _project

        _validate_sine_prepared_entry_labels(self)
        return {
            "schema": "tnfr.relational-sine-prepared-entry.v1",
            "report": _project(self),
        }

    def compose_with(
        self,
        other,
        *,
        bridge,
        left_initial_time,
        right_initial_time,
        edge_turn_offsets,
        form_origin_difference=None,
        phase_origin_difference=None,
        work_allowance=None,
    ):
        """Assess a declared join of two synchronous analytic endpoints."""
        from .relational_sine_composition import assess_sine_prepared_composition

        return assess_sine_prepared_composition(
            self,
            other,
            bridge=bridge,
            left_initial_time=left_initial_time,
            right_initial_time=right_initial_time,
            edge_turn_offsets=edge_turn_offsets,
            form_origin_difference=form_origin_difference,
            phase_origin_difference=phase_origin_difference,
            work_allowance=work_allowance,
        )


def _validate_sine_prepared_entry_labels(entry):
    """Check label projection only, without trusting or replaying evidence."""
    from ..sdk.relational_reports import _validate_label, _validate_label_groups

    _validate_comparison_labels(entry.source)
    if isinstance(entry.source, SineRelativePattern):
        _validate_label(entry.source.reference_node)
    _validate_label_groups(entry.geometry.nodes)
    _validate_sine_sector_capture_labels(entry.capture)


def certify_sine_prepared_entry(
    source: SineExchangeComparison | SineRelativePattern,
    *,
    scaled_time,
    edge_turn_offsets,
) -> SinePreparedEntry:
    """Bound an analytic full-state transit, then certify nonzero-sector capture.

    An exact comparison retains the original identical-phase contract. A
    relative pattern supplies its original per-node form and phase residual
    radii, exact held capacities and arbitrary common origins. Its initial
    zero winding is certified from all raw edge intervals, without choosing
    wrap branches. All complete states in the declared set must pass.
    The supplied nonnegative time is
    ``tau=e*t`` using the reference model's effective positive form weight e.
    The shared exact exponential work limit is ``lambda*tau<=4096``; graph
    admission retains the shared 32-node/50-edge budget. Unsupported input
    domains raise. Failure of a sufficient endpoint/capture comparison returns
    unavailable without asserting nonexistence or changing the supplied time.

    For ``K_i=nu_i/d_i``, ``H=K^-1``, ``alpha=w/(beta*pi*e)`` and
    ``z=alpha*(x-weighted_mean(x))``, the scaled law is
    ``z'=-K*L*z+eta*K*S``, ``theta'=K*L*z``, with ``eta=beta*alpha**2``.
    Thus ``theta+z-v`` has derivative ``eta*K*S``, where v=z(0), and
    ``||K*S||_H<=sqrt(sum(nu_i*d_i))`` globally. Weighted contraction of
    the diffusion semigroup supplies the documented endpoint radii. This is
    a comparison estimate for the actual full sine law, not pure diffusion
    substituted for that law.

    For a pattern, weighted centering contracts the form residual norm.
    The phase endpoint also retains its initial phase and form residuals;
    common origins cancel, but the family does not have a single known
    conserved mean or a stationary trajectory merely because its nominal
    form is uniform. The source's projected relative boxes are not consumed.
    """
    preparation = _sine_preparation(source)
    admitted, geometry = preparation.admitted, preparation.geometry
    uncertain = preparation.uncertain
    form, phase = preparation.form, preparation.phase
    form_errors, phase_errors = preparation.form_errors, preparation.phase_errors
    model = preparation.model
    e, w = preparation.e, preparation.w
    weights, mobility = preparation.weights, preparation.mobility
    mean, phase_mean = preparation.mean, preparation.phase_mean
    centered_form_errors = preparation.centered_form_errors
    centered_phase_errors = preparation.centered_phase_errors
    gap, pi = preparation.gap, preparation.pi
    alpha, inverse_alpha, eta = (
        preparation.alpha,
        preparation.inverse_alpha,
        preparation.eta,
    )
    nominal_initial_norm = preparation.nominal_initial_norm
    initial_error_norm = preparation.initial_error_norm
    initial_norm, forcing = preparation.initial_norm, preparation.forcing
    scaled_nominal, scaled_initial = (
        preparation.scaled_nominal,
        preparation.scaled_initial,
    )
    if not uncertain and any(value != phase[0] for value in phase[1:]):
        raise ValueError("prepared entry requires identical exact initial phase lifts")
    tau = exact_or_represented_real(scaled_time, "scaled_time")
    if tau < 0:
        raise ValueError("scaled_time must be nonnegative")
    horizon = tau / e
    decay = I(*_negative_exp_bounds(gap * tau))
    scaled_form_radius = decay.hi * initial_norm.hi + eta.hi * forcing.hi * min(
        tau, Q(1) / gap
    )
    phase_radius = scaled_form_radius + eta.hi * tau * forcing.hi
    centered_form_bounds, centered_phase_bounds = [], []
    absolute_form_bounds, absolute_phase_bounds = [], []
    for index, value in enumerate(scaled_nominal):
        node_gain = sqrt(I(mobility[index])).hi
        form_radius = inverse_alpha.hi * node_gain * scaled_form_radius
        angle_radius = (
            node_gain * phase_radius
            + centered_phase_errors[index]
            + alpha.hi * centered_form_errors[index]
        )
        centered_form_bounds.append(I(-form_radius, form_radius))
        centered_phase_bounds.append(
            phase[index] - phase_mean + value + I(-angle_radius, angle_radius)
        )
        if not uncertain:
            absolute_form_bounds.append(I(mean - form_radius, mean + form_radius))
            absolute_phase_bounds.append(
                phase[0] + value + I(-angle_radius, angle_radius)
            )
    form_bounds = None if uncertain else tuple(absolute_form_bounds)
    phase_bounds = None if uncertain else tuple(absolute_phase_bounds)
    form_edges, phase_edges = [], []
    for left, right in geometry.edges:
        edge_gain = sqrt(I(mobility[left] + mobility[right])).hi
        form_radius = inverse_alpha.hi * edge_gain * scaled_form_radius
        angle_radius = (
            edge_gain * phase_radius
            + phase_errors[left]
            + phase_errors[right]
            + alpha.hi * (form_errors[left] + form_errors[right])
        )
        form_edges.append(I(-form_radius, form_radius))
        # Cancel the common origin and combine the exact form difference
        # before enclosing its shared mathematical-pi coefficient.
        phase_edges.append(
            phase[right]
            - phase[left]
            + alpha * (form[right] - form[left])
            + I(-angle_radius, angle_radius)
        )
    initial_form_storage = preparation.initial_form_storage_bounds
    initial_phase_storage = preparation.initial_phase_storage_bounds
    initial_phase_gaps = preparation.initial_phase_gaps
    initial_acute_margins = tuple(pi / 2 - abs(value) for value in initial_phase_gaps)
    initial_zero = all(value.lo > 0 for value in initial_acute_margins)
    mean_scope = (
        "unobserved_absolute_common_origins_each_family_member_has_its_own_conserved_means"
        if uncertain
        else "exact_initial_means_of_enclosed_trajectory_not_every_point_of_outer_node_box"
    )
    capture = _certify_sine_sector_set(
        source=source,
        geometry=geometry,
        model=model,
        capacity_bounds=tuple(I(value) for value in admitted.capacity),
        exact_held_capacity=admitted.capacity,
        form_edge_gap_bounds=tuple(form_edges),
        phase_edge_gap_bounds=tuple(phase_edges),
        edge_turn_offsets=edge_turn_offsets,
        uncertainty_scope=(
            "analytic_weighted_Duhamel_endpoint_of_original_residual_family_with_memberwise_conserved_means"
            if uncertain
            else "analytic_weighted_Duhamel_endpoint_with_correlated_norm_bounds_and_conserved_mean_leaf"
        ),
        observation_time=horizon,
        weighted_form_mean=None if uncertain else mean,
        weighted_phase_mean=None if uncertain else phase_mean,
        weighted_mean_scope=mean_scope,
    )
    unresolved = tuple(
        reason
        for condition, reason in (
            (initial_zero, "whole_initial_set_zero_winding_not_certified"),
            (capture.admitted, "analytic_endpoint_sector_capture_not_certified"),
            (any(capture.cycle_periods), "nonzero_captured_sector_required"),
        )
        if not condition
    )
    return SinePreparedEntry(
        source=source,
        reference_model=model,
        geometry=geometry,
        scaled_time=tau,
        horizon=horizon,
        coefficient_ratio=w / e,
        weighted_form_mean=None if uncertain else mean,
        common_initial_phase=None if uncertain else phase[0],
        initial_form_storage=preparation.initial_form_storage,
        initial_cycle_periods=(0,) * geometry.cycle_rank if initial_zero else None,
        mobility=mobility,
        metric_weights=weights,
        weighted_gap_lower_bound=gap,
        exponential_decay_bounds=decay,
        form_to_phase_scale_bounds=alpha,
        feedback_strength_bounds=eta,
        scaled_initial_form_bounds=scaled_initial,
        scaled_initial_norm_bounds=initial_norm,
        forcing_norm_bounds=forcing,
        scaled_form_radius_upper_bound=scaled_form_radius,
        phase_radius_upper_bound=phase_radius,
        endpoint_form_bounds=form_bounds,
        endpoint_phase_bounds=phase_bounds,
        endpoint_form_edge_gap_bounds=tuple(form_edges),
        endpoint_phase_edge_gap_bounds=tuple(phase_edges),
        capture=capture,
        hypothesis_failures=capture.hypothesis_failures,
        unresolved_conditions=unresolved,
        status=(
            "unavailable" if capture.hypothesis_failures or unresolved else "admitted"
        ),
        centered_endpoint_form_bounds=tuple(centered_form_bounds),
        centered_endpoint_phase_bounds=tuple(centered_phase_bounds),
        endpoint_coordinate_scope=(
            "centered_by_each_member_conserved_weighted_means_absolute_origins_unavailable"
            if uncertain
            else "exact_source_absolute_and_centered_endpoints"
        ),
        mean_scope=mean_scope,
        nominal_weighted_form_mean=mean,
        nominal_weighted_phase_mean=phase_mean,
        form_error_bounds=form_errors,
        phase_error_bounds=phase_errors,
        centered_form_error_bounds=centered_form_errors,
        centered_phase_error_bounds=centered_phase_errors,
        nominal_scaled_initial_norm_bounds=nominal_initial_norm,
        scaled_initial_error_norm_upper_bound=initial_error_norm,
        initial_form_storage_bounds=initial_form_storage,
        initial_phase_storage_bounds=initial_phase_storage,
        initial_storage_bounds=preparation.initial_storage_bounds,
        initial_phase_edge_gap_bounds=tuple(initial_phase_gaps),
        initial_acute_margin_bounds=initial_acute_margins,
        initial_zero_winding_certified=initial_zero,
    )
