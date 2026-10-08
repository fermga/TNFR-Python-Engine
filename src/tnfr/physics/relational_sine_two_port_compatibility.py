"""Two-contact C9 equilibria and a direct storage-handoff obstruction.

Strict scalar root brackets and an exact incidence factorization establish
criticality. Interval nodal coordinates enclose that one correlated target;
neither their midpoint nor every point in their product is an equilibrium.
The separate exact-rational handoff report excludes an initial-storage
certificate for a specified family near undeformed twists. Neither report
evaluates a contact event or trajectory, or establishes source formation.
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
from ..mathematics._rational_interval import INTERVAL_METHOD, I, cos, pi_interval, sin
from ._sine_admission import _sine_model_coefficients
from .phase_cycle_geometry import (
    PhaseCycleGeometry,
    PhaseRootBracket,
    _cycle_row,
    _derive,
    _enclose_decreasing_phase_root,
)
from .relational_observations import _ordered

__all__ = (
    "SineTwoPortCompatibility",
    "assess_sine_two_port_compatibility",
    "SineTwoPortHandoffObstruction",
    "assess_sine_two_port_handoff_obstruction",
)

_NODES = tuple(range(18))
_EDGES = tuple(
    sorted(
        tuple(sorted((offset + i, offset + (i + 1) % 9)))
        for offset in (0, 9)
        for i in range(9)
    )
)
_EDGES = tuple(sorted((*_EDGES, (0, 9), (1, 10))))
_ZERO = (Q(0), Q(0), Q(0))


def _add(left, right):
    return tuple(a + b for a, b in zip(left, right))


def _scale(factor, row):
    return tuple(factor * value for value in row)


def _affine_geometry(classes, geometry):
    """Return exact turn correlations in the actual ordered short arcs A,C.

    Each coefficient row is (constant, coefficient of A, coefficient of C).
    The receiver port lift is delta=(A-C)/2. Integer edge branches retain both
    cycle classes and zero interface winding throughout this affine family.
    The phase gauge is the full degree-weighted mean, not two separate means.
    """
    short = ((Q(0), Q(1), Q(0)), (Q(0), Q(0), Q(1)))
    delta = (Q(0), Q(1, 2), Q(-1, 2))
    rows = []
    for component, k in enumerate(classes):
        start = _ZERO if component == 0 else delta
        bulk = _scale(Q(1, 8), _add((Q(k), Q(0), Q(0)), _scale(-1, short[component])))
        rows.append(start)
        rows.extend(
            _add(start, _add(short[component], _scale(Q(j - 1), bulk)))
            for j in range(1, 9)
        )
    degrees = tuple(sum(i in edge for edge in geometry.edges) for i in _NODES)
    mass = sum(degrees)
    mean = tuple(
        sum((Q(degrees[i]) * row[column] for i, row in enumerate(rows)), Q(0)) / mass
        for column in range(3)
    )
    centered = tuple(_add(row, _scale(-1, mean)) for row in rows)
    offsets, edges = [], []
    for left, right in geometry.edges:
        offset = classes[0] if (left, right) == (0, 8) else 0
        if (left, right) == (9, 17):
            offset = classes[1]
        offsets.append(offset)
        edges.append(
            _add(_add(rows[right], _scale(-1, rows[left])), (-Q(offset), Q(0), Q(0)))
        )
    return degrees, mean, centered, tuple(edges), tuple(offsets)


def _current_factorization(geometry):
    """Factor all fine currents through two balance equations exactly.

    The five sine symbols are short_D, bulk_D, short_R, bulk_R, delta.
    F_D=bulk_D-short_D-delta and F_R=bulk_R-short_R+delta.
    """
    edge_rows = []
    for edge in geometry.edges:
        row = [Q(0)] * 5
        if edge == (0, 9):
            row[4] = 1
        elif edge == (1, 10):
            row[4] = -1
        else:
            component = int(edge[0] >= 9)
            local = tuple(i - 9 * component for i in edge)
            if local == (0, 1):
                row[2 * component] = 1
            else:
                row[2 * component + 1] = -1 if local == (0, 8) else 1
        edge_rows.append(tuple(Q(value) for value in row))
    negative_incidence = tuple(tuple(-Q(v) for v in row) for row in geometry.incidence)
    nodal = exact_matrix_product(negative_incidence, tuple(edge_rows))
    equations = (
        (Q(-1), Q(1), Q(0), Q(0), Q(-1)),
        (Q(0), Q(0), Q(-1), Q(1), Q(1)),
    )
    multipliers = tuple(
        (Q(-int(i == 0) + int(i == 1)), Q(-int(i == 9) + int(i == 10))) for i in _NODES
    )
    if exact_matrix_product(multipliers, equations) != nodal:
        raise ArithmeticError("full sine-current incidence factorization failed")
    return tuple(edge_rows), nodal, equations, multipliers


def _h(k, turns):
    angle = 2 * pi_interval()
    return sin(angle * ((k - I.coerce(turns)) / 8)) - sin(angle * turns)


def _inner_root(a, refinements):
    if a == Q(2, 9):
        return I(Q(1, 9)), None
    current = _h(2, a)
    bracket = _enclose_decreasing_phase_root(
        lambda c: _h(1, c) + current,
        lower=Q(1, 9),
        upper=Q(2, 9),
        refinements=refinements,
    )
    return I(bracket.lower, bracket.upper), bracket


def _root_enclosures(outer_refinements, inner_refinements):
    """Enclose the canonical (2,1) root with declared bounded work only."""

    def residual(a):
        c, _ = _inner_root(a, inner_refinements)
        current = I(0) if a == Q(2, 9) else _h(2, a)
        return current - sin(pi_interval() * (a - c))

    outer = _enclose_decreasing_phase_root(
        residual,
        lower=Q(1, 9),
        upper=Q(2, 9),
        refinements=outer_refinements,
    )
    low_c, upper_inner = _inner_root(outer.upper, inner_refinements)
    high_c, lower_inner = _inner_root(outer.lower, inner_refinements)
    return (
        outer,
        (lower_inner, upper_inner),
        (I(outer.lower, outer.upper), I(low_c.lo, high_c.hi)),
    )


def _laplacian_gap(geometry, degrees):
    laplacian = tuple(
        tuple(
            Q(degrees[i] if i == j else -int(tuple(sorted((i, j))) in geometry.edges))
            for j in _NODES
        )
        for i in _NODES
    )
    adjacency = tuple(
        tuple(j for j in _NODES if i != j and laplacian[i][j]) for i in _NODES
    )
    diameter = 0
    for start in _NODES:
        distances, pending = {start: 0}, [start]
        for node in pending:
            for other in adjacency[node]:
                if other not in distances:
                    distances[other] = distances[node] + 1
                    pending.append(other)
        if len(distances) != len(_NODES):
            raise ArithmeticError("the fixed two-port support must be connected")
        diameter = max(diameter, max(distances.values()))
    gap = Q(4, len(_NODES) * diameter)
    shifted = tuple(
        tuple(laplacian[i][j] - gap * (Q(int(i == j)) - Q(1, 18)) for j in _NODES)
        for i in _NODES
    )
    if not exact_symmetric_semidefinite(shifted):
        raise ArithmeticError("the fixed full-support Laplacian gap failed")
    return laplacian, diameter, gap, shifted


@dataclass(frozen=True)
class SineTwoPortCompatibility:
    """One implicit full-network equilibrium, with conditional local attraction.

    Affine turn coefficients are (constant, donor-short coefficient,
    receiver-short coefficient). Their shared variables are correlated by the
    implicit root equations; independent interval points are not equilibria.
    The gauge removes the full degree-weighted phase mean. Neither source
    formation nor access from a previously prepared class pair is certified.
    ``uniform_pair_nodal_current_bounds`` uses the particular undeformed
    alignment D0=R0. Exclusion for every relative origin instead follows
    from the two exact bridge-balance equations for unequal classes.
    """

    classes: tuple[int, int]
    outer_refinements: int
    inner_refinements: int
    reference_model: RelationalExchangeModel
    geometry: PhaseCycleGeometry
    degrees: tuple[int, ...]
    invariant_weights: tuple[Q, ...]
    weighted_coordinate_mass: Q
    named_cycles: tuple[tuple[int, ...], ...]
    named_cycle_periods: tuple[int, int, int]
    uncentered_phase_mean_turn_coefficients: tuple[Q, Q, Q]
    nodal_turn_affine_coefficients: tuple[tuple[Q, Q, Q], ...]
    edge_turn_affine_coefficients: tuple[tuple[Q, Q, Q], ...]
    edge_integer_offsets: tuple[int, ...]
    edge_sine_symbol_coefficients: tuple[tuple[Q, ...], ...]
    nodal_sine_symbol_coefficients: tuple[tuple[Q, ...], ...]
    balance_equation_sine_coefficients: tuple[tuple[Q, ...], ...]
    nodal_balance_multipliers: tuple[tuple[Q, Q], ...]
    laplacian: tuple[tuple[Q, ...], ...]
    support_diameter: int
    laplacian_gap_lower_bound: Q
    laplacian_shifted_matrix: tuple[tuple[Q, ...], ...]
    canonical_root_turn_bracket: PhaseRootBracket | None
    inner_root_brackets_at_outer_endpoints: (
        tuple[PhaseRootBracket | None, PhaseRootBracket | None] | None
    )
    short_arc_turn_bounds: tuple[I, I] | None
    bulk_arc_turn_bounds: tuple[I, I] | None
    bridge_turn_bounds: tuple[I, I] | None
    nodal_turn_bounds: tuple[I, ...] | None
    target_phase_bounds: tuple[I, ...] | None
    edge_turn_bounds: tuple[I, ...] | None
    edge_current_bounds: tuple[I, ...] | None
    balance_equation_bounds: tuple[I, I] | None
    nodal_current_residual_bounds: tuple[I, ...] | None
    target_form_rate_bounds: tuple[I, ...] | None
    target_phase_rate_bounds: tuple[I, ...] | None
    edge_cosine_bounds: tuple[I, ...] | None
    phase_hessian_bounds: tuple[tuple[I, ...], ...] | None
    acute_margin_turns_bounds: I | None
    minimum_cosine_lower_bound: Q | None
    phase_hessian_gap_lower_bound: Q | None
    target_phase_storage_bounds: I | None
    uniform_pair_nodal_current_bounds: tuple[I, ...]
    uniform_pair_compatible: bool
    uniform_pair_excluded: bool
    implicit_equilibrium_certified: bool
    full_nodal_residuals_consistent: bool
    acute_geometry_certified: bool
    local_attraction_certified: bool
    status: str
    unavailable_reasons: tuple[str, ...]
    arithmetic_method: str = INTERVAL_METHOD
    law: str = "normalized_sine_reciprocal_exchange"
    clock: str = "tau=e*t"
    capacity: tuple[Q, ...] = (Q(1),) * 18
    target_epi: tuple[Q, ...] = (Q(0),) * 18
    weighted_form_mean: Q = Q(0)
    weighted_phase_mean: Q = Q(0)
    sine_symbol_order: tuple[str, ...] = (
        "donor_short",
        "donor_bulk",
        "receiver_short",
        "receiver_bulk",
        "bridge_delta",
    )
    scope: tuple[str, ...] = (
        "fixed_two_unit_C9_cycles_with_contacts_D0_R0_and_D1_R1",
        "same_positive_loss_and_exchange_beta_and_held_capacities_as_C9_formation",
        "canonical_root_is_ordered_classes_two_one_other_order_uses_exact_swap",
        "matched_classes_use_exact_uniform_twists_and_zero_bridge_current",
        "uniform_pair_nodal_current_bounds_use_the_specific_D0_R0_alignment",
        "unequal_uniform_pairs_are_excluded_by_both_bridge_equations_for_every_origin",
        "strict_root_signs_and_full_incidence_factorization_prove_criticality",
        "interval_residuals_are_consistency_checks_not_a_small_residual_certificate",
        "one_correlated_implicit_target_not_every_point_of_its_interval_box",
        "unique_equilibrium_in_the_declared_acute_period_cell_modulo_common_phase",
        "local_attraction_on_each_conserved_full_network_mean_leaf",
        "no_source_handoff_formation_trajectory_event_selection_or_physical_identification",
    )

    def to_dict(self):
        from ..sdk.relational_reports import _project

        return {
            "schema": "tnfr.sine-two-port-compatibility.v1",
            "report": _project(self),
        }


def assess_sine_two_port_compatibility(
    *, classes, outer_refinements, inner_refinements
):
    """Certify acute compatibility of two supplied C9 classes on fixed support.

    ``classes`` contains exactly two ordinary integers 1 or 2. Both mandatory
    refinement budgets are ordinary integers from 1 through 64. Nested scalar
    bisection uses strict outward interval signs; an unresolved comparison
    returns unavailable without expanding the bracket or work budget.

    Forms and the full degree-weighted phase lift mean are fixed to zero.
    Class orders (2,1) and (1,2) share the same implicit construction by
    exact component exchange; matched orders use exact uniform twists.
    This is a target-compatibility and local-attraction certificate, not a
    claim that any previously supplied preparation reaches that target.
    """
    classes = _ordered(classes, "classes", limit=3)
    if len(classes) != 2 or any(type(k) is not int or k not in (1, 2) for k in classes):
        raise ValueError("classes must contain exactly two ordinary integers 1 or 2")
    for value, name in (
        (outer_refinements, "outer_refinements"),
        (inner_refinements, "inner_refinements"),
    ):
        if type(value) is not int or not 1 <= value <= 64:
            raise ValueError(f"{name} must be an ordinary integer from 1 through 64")
    model = RelationalExchangeModel(
        1, epi_weight=Q(1023, 1024), phase_weight=Q(1, 1024), phase_domain="regular"
    )
    e, w, _ = _sine_model_coefficients(model, positive_loss=True)
    geometry = _derive(_NODES, _EDGES)
    degrees, mean, nodal, edges, offsets = _affine_geometry(classes, geometry)
    edge_symbols, nodal_symbols, equations, multipliers = _current_factorization(
        geometry
    )
    laplacian, diameter, gap, shifted = _laplacian_gap(geometry, degrees)
    pi = pi_interval()
    gamma = (w / e) / pi
    matched = classes[0] == classes[1]
    uniform_bridge = I(0) if matched else sin(2 * pi * Q(classes[1] - classes[0], 9))
    uniform_residuals = tuple(
        uniform_bridge * (int(i == 1) - int(i == 10)) for i in _NODES
    )
    outer = inner = short = None
    reasons = []
    try:
        if matched:
            short = (I(Q(classes[0], 9)),) * 2
        else:
            outer, inner, canonical = _root_enclosures(
                outer_refinements, inner_refinements
            )
            short = canonical if classes == (2, 1) else canonical[::-1]
    except ArithmeticError:
        reasons.append("strict_root_sign_unresolved_within_declared_budget")
    bulk = bridge = node_bounds = phase_bounds = edge_bounds = currents = None
    balances = residuals = form_rates = phase_rates = cosines = hessian = None
    acute_margin = cosine_lower = hessian_gap = storage = None
    consistent = acute = attraction = False
    if short is not None:

        def evaluate(row):
            if matched:
                return I(row[0] + (row[1] + row[2]) * Q(classes[0], 9))
            return I(row[0]) + row[1] * short[0] + row[2] * short[1]

        bulk = (
            (I(Q(classes[0], 9)),) * 2
            if matched
            else tuple((k - angle) / 8 for k, angle in zip(classes, short))
        )
        delta = I(0) if matched else (short[0] - short[1]) / 2
        bridge = (delta, -delta)
        node_bounds = tuple(map(evaluate, nodal))
        phase_bounds = tuple(2 * pi * value for value in node_bounds)
        edge_bounds = tuple(map(evaluate, edges))
        currents = tuple(
            I(0) if matched and value.lo == value.hi == 0 else sin(2 * pi * value)
            for value in edge_bounds
        )
        symbols = (
            sin(2 * pi * short[0]),
            sin(2 * pi * bulk[0]),
            sin(2 * pi * short[1]),
            sin(2 * pi * bulk[1]),
            sin(2 * pi * delta),
        )
        balances = tuple(
            sum((value * current for value, current in zip(row, symbols)), I(0))
            for row in equations
        )
        residuals = tuple(
            sum((-value * current for value, current in zip(row, currents)), I(0))
            for row in geometry.incidence
        )
        form_rates = tuple(
            gamma * value / degree for value, degree in zip(residuals, degrees)
        )
        phase_rates = (I(0),) * 18
        cosines = tuple(cos(2 * pi * value) for value in edge_bounds)
        hessian = tuple(
            tuple(
                sum(
                    (
                        Q(geometry.incidence[i][edge] * geometry.incidence[j][edge])
                        * value
                        for edge, value in enumerate(cosines)
                    ),
                    I(0),
                )
                for j in _NODES
            )
            for i in _NODES
        )
        absolute_edges = tuple(map(abs, edge_bounds))
        acute_margin = I(Q(1, 4)) - I(
            max(value.lo for value in absolute_edges),
            max(value.hi for value in absolute_edges),
        )
        cosine_lower = min(value.lo for value in cosines)
        hessian_gap = cosine_lower * gap if cosine_lower > 0 else None
        storage = sum((1 - value for value in cosines), I(0))
        consistent = all(value.contains(0) for value in (*residuals, *balances))
        acute = acute_margin.lo > 0 and cosine_lower > 0
        attraction = (
            consistent and acute and hessian_gap is not None and hessian_gap > 0
        )
        if not consistent:
            reasons.append("full_nodal_residual_enclosures_inconsistent")
        if not acute:
            reasons.append("strict_acute_geometry_not_certified")
    return SineTwoPortCompatibility(
        classes=classes,
        outer_refinements=outer_refinements,
        inner_refinements=inner_refinements,
        reference_model=model,
        geometry=geometry,
        degrees=degrees,
        invariant_weights=tuple(map(Q, degrees)),
        weighted_coordinate_mass=Q(sum(degrees)),
        named_cycles=(tuple(range(9)), tuple(range(9, 18)), (0, 9, 10, 1)),
        named_cycle_periods=(*classes, 0),
        uncentered_phase_mean_turn_coefficients=mean,
        nodal_turn_affine_coefficients=nodal,
        edge_turn_affine_coefficients=edges,
        edge_integer_offsets=offsets,
        edge_sine_symbol_coefficients=edge_symbols,
        nodal_sine_symbol_coefficients=nodal_symbols,
        balance_equation_sine_coefficients=equations,
        nodal_balance_multipliers=multipliers,
        laplacian=laplacian,
        support_diameter=diameter,
        laplacian_gap_lower_bound=gap,
        laplacian_shifted_matrix=shifted,
        canonical_root_turn_bracket=outer,
        inner_root_brackets_at_outer_endpoints=inner,
        short_arc_turn_bounds=short,
        bulk_arc_turn_bounds=bulk,
        bridge_turn_bounds=bridge,
        nodal_turn_bounds=node_bounds,
        target_phase_bounds=phase_bounds,
        edge_turn_bounds=edge_bounds,
        edge_current_bounds=currents,
        balance_equation_bounds=balances,
        nodal_current_residual_bounds=residuals,
        target_form_rate_bounds=form_rates,
        target_phase_rate_bounds=phase_rates,
        edge_cosine_bounds=cosines,
        phase_hessian_bounds=hessian,
        acute_margin_turns_bounds=acute_margin,
        minimum_cosine_lower_bound=cosine_lower,
        phase_hessian_gap_lower_bound=hessian_gap,
        target_phase_storage_bounds=storage,
        uniform_pair_nodal_current_bounds=uniform_residuals,
        uniform_pair_compatible=matched,
        uniform_pair_excluded=not matched,
        implicit_equilibrium_certified=short is not None,
        full_nodal_residuals_consistent=consistent,
        acute_geometry_certified=acute,
        local_attraction_certified=attraction,
        status="certified_compatible" if attraction else "unavailable",
        unavailable_reasons=tuple(reasons),
    )


@dataclass(frozen=True)
class SineTwoPortHandoffObstruction:
    """One exact obstruction to a direct all-face storage capture certificate.

    The source family consists of undeformed winding-(2,1) C9 twists with
    arbitrary relative phase origins, nodewise phase errors in radians at
    most ``phase_error_radius``, and arbitrary finite signed forms. Its
    members need not be acute. For acute members covered by the positive
    margin, one boundary witness lies in the same named period cell and
    has less storage. Consequently no common lower bound on all sector
    faces can exceed those members' initial storage.

    The zero-mean, zero-form witness represents every conserved-mean leaf
    after common phase and uniform form shifts. It is neither a critical
    target nor an evaluated trajectory. Failure of this sufficient bound
    makes the obstruction unavailable; neither outcome decides capture.
    """

    phase_error_radius: Q
    reference_model: RelationalExchangeModel
    geometry: PhaseCycleGeometry
    degrees: tuple[int, ...]
    invariant_weights: tuple[Q, ...]
    weighted_coordinate_mass: Q
    witness_uncentered_nodal_turns: tuple[Q, ...]
    witness_uncentered_weighted_phase_mean_turns: Q
    witness_nodal_turns: tuple[Q, ...]
    witness_edge_integer_offsets: tuple[int, ...]
    witness_edge_turns: tuple[Q, ...]
    named_cycles: tuple[tuple[int, ...], ...]
    named_cycle_periods: tuple[int, ...]
    fundamental_cycle_periods: tuple[int, ...]
    witness_boundary_edge_indices: tuple[int, ...]
    ideal_storage_gap_lower_bound: Q
    phase_storage_lipschitz: Q
    phase_storage_error_allowance: Q
    storage_gap_lower_bound: Q
    max_certifying_phase_error_radius: Q
    handoff_obstruction_certified: bool
    status: str
    unavailable_reasons: tuple[str, ...]
    classes: tuple[int, int] = (2, 1)
    capacity: tuple[Q, ...] = (Q(1),) * 18
    witness_epi: tuple[Q, ...] = (Q(0),) * 18
    witness_weighted_form_mean: Q = Q(0)
    witness_weighted_phase_mean_turns: Q = Q(0)
    law: str = "normalized_sine_reciprocal_exchange"
    clock: str = "tau=e*t"
    arithmetic_method: str = "exact_rational_witness_and_analytic_storage_gap"
    scope: tuple[str, ...] = (
        "fixed_two_unit_C9_cycles_with_contacts_D0_R0_and_D1_R1",
        "same_positive_loss_complete_sine_law_and_held_unit_capacities",
        "undeformed_winding_two_one_twists_with_arbitrary_relative_phase_origins",
        "source_errors_are_nodewise_real_phase_lift_errors_in_radians",
        "arbitrary_finite_signed_source_forms_have_nonnegative_form_storage",
        "source_family_includes_nonacute_members_no_live_state_admission",
        "positive_margin_preserves_ring_classes_and_covers_every_acute_family_member",
        "one_feasible_boundary_witness_bounds_any_common_all_face_lower_bound",
        "common_phase_and_uniform_form_shifts_match_each_source_conserved_mean_leaf",
        "zero_mean_witness_is_a_representative_not_a_phase_or_form_reset",
        "strict_positive_declared_lower_bound_required_otherwise_unavailable",
        "no_root_trigonometric_evaluation_trajectory_or_frozen_producer_call",
        "no_failed_convergence_instability_formation_or_physical_identification_claim",
    )

    def to_dict(self):
        from ..sdk.relational_reports import _project

        return {
            "schema": "tnfr.sine-two-port-handoff-obstruction.v1",
            "report": _project(self),
        }


def assess_sine_two_port_handoff_obstruction(*, phase_error_radius):
    """Assess an analytic obstruction to direct initial-storage handoff.

    The complete winding-(2,1) source family and boundary witness are fixed
    by the theorem in ``SINE_TWO_PORT_COMPATIBILITY.md``. Only the nonnegative
    nodewise phase error radius in radians is supplied. Shared admission
    preserves exact rationals and otherwise uses the represented-real
    contract. No source observation, cached report, root solve, or trajectory
    enters the calculation.

    Every ideal family member has storage greater than the explicit witness
    by at least the strict rational bound 17/13824. The sine potential is
    40-Lipschitz in the nodewise maximum phase norm on this 20-edge graph.
    A positive 17/13824-40*radius therefore excludes a strict initial-storage
    all-face certificate for every acute member. Nonpositive declared margins
    return unavailable, without asserting that a source is acute or captured.
    """
    radius = exact_or_represented_real(phase_error_radius, "phase_error_radius")
    if radius < 0:
        raise ValueError("phase_error_radius must be nonnegative")
    model = RelationalExchangeModel(
        1, epi_weight=Q(1023, 1024), phase_weight=Q(1, 1024), phase_domain="regular"
    )
    _sine_model_coefficients(model, positive_loss=True)
    geometry = _derive(_NODES, _EDGES)
    degrees = tuple(sum(i in edge for edge in geometry.edges) for i in _NODES)
    mass = Q(sum(degrees))
    # A single donor edge is exactly a quarter turn. All other donor
    # principal edges are 7/32; receiver edges remain 1/9. The two contacts
    # are +/-31/576. These rational lifts retain all three cycle periods.
    raw = (
        Q(0),
        Q(7, 32),
        *(Q(63 * j + 9, 288) for j in range(2, 9)),
        *(Q(31, 576) + Q(j, 9) for j in range(9)),
    )
    mean = sum((degree * value for degree, value in zip(degrees, raw)), Q(0)) / mass
    nodes = tuple(value - mean for value in raw)
    offsets = tuple(
        2 if edge == (0, 8) else int(edge == (9, 17)) for edge in geometry.edges
    )
    turns = tuple(
        nodes[j] - nodes[i] - offset for (i, j), offset in zip(geometry.edges, offsets)
    )
    cycles = (tuple(range(9)), tuple(range(9, 18)), (0, 9, 10, 1))
    indices = {edge: i for i, edge in enumerate(geometry.edges)}

    def period(row):
        value = sum((sign * turn for sign, turn in zip(row, turns)), Q(0))
        if value.denominator != 1:
            raise ArithmeticError("boundary witness must have integral cycle periods")
        return value.numerator

    named_periods = tuple(period(_cycle_row(cycle, indices)) for cycle in cycles)
    fundamental_periods = tuple(map(period, geometry.cycle_rows))
    boundary = tuple(i for i, value in enumerate(turns) if abs(value) == Q(1, 4))
    if (
        named_periods != (2, 1, 0)
        or boundary != (indices[(1, 2)],)
        or any(abs(value) > Q(1, 4) for value in turns)
        or sum((degree * value for degree, value in zip(degrees, nodes)), Q(0)) != 0
    ):
        raise ArithmeticError("the fixed boundary witness geometry is inconsistent")
    ideal_gap = Q(17, 13824)
    lipschitz = Q(2 * len(geometry.edges))
    allowance = lipschitz * radius
    gap = ideal_gap - allowance
    certified = gap > 0
    return SineTwoPortHandoffObstruction(
        phase_error_radius=radius,
        reference_model=model,
        geometry=geometry,
        degrees=degrees,
        invariant_weights=tuple(map(Q, degrees)),
        weighted_coordinate_mass=mass,
        witness_uncentered_nodal_turns=raw,
        witness_uncentered_weighted_phase_mean_turns=mean,
        witness_nodal_turns=nodes,
        witness_edge_integer_offsets=offsets,
        witness_edge_turns=turns,
        named_cycles=cycles,
        named_cycle_periods=named_periods,
        fundamental_cycle_periods=fundamental_periods,
        witness_boundary_edge_indices=boundary,
        ideal_storage_gap_lower_bound=ideal_gap,
        phase_storage_lipschitz=lipschitz,
        phase_storage_error_allowance=allowance,
        storage_gap_lower_bound=gap,
        max_certifying_phase_error_radius=ideal_gap / lipschitz,
        handoff_obstruction_certified=certified,
        status="certified_handoff_obstruction" if certified else "unavailable",
        unavailable_reasons=() if certified else ("strict_storage_gap_not_certified",),
    )
