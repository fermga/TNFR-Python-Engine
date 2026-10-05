"""Exact phase-support topology and rational-turn reconstruction.

Fundamental cycles are coordinates for the integer cycle lattice, not physical
selectors. Edge turns mean an exact angle divided by mathematical ``2*pi``;
they are not inferred from rounded radians or from inverse trigonometry.
The original reconstruction certifies a strictly acute circular configuration.
A separate all-phase reader also admits nonacute and antipodal edges, folding
exact sine symmetries without changing that acute contract. These sufficient
sine checks leave other identities unresolved. Neither reconstruction executes
or derives a phase evolution law; the finite C5 classification is geometric.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from fractions import Fraction
from numbers import Integral
from typing import Any

from ..mathematics.krylov import exact_rank
from ._cycle_algebra import Vector, ordered_vector

__all__ = [
    "AcuteCyclePeriodAssessment",
    "BridgeTreeHessianInertia",
    "CircularPhaseState",
    "C5SineCriticalSet",
    "C5PhaseHessianInertia",
    "PhaseChordExtension",
    "PhaseChordReset",
    "PhaseCycleGeometry",
    "PhaseCycleState",
    "assess_acute_cycle_periods",
    "derive_phase_chord_extension",
    "derive_phase_cycle_geometry",
    "classify_c5_sine_critical_set",
    "compose_bridge_tree_hessian_inertia",
    "observe_phase_chord_reset",
    "reconstruct_phase_cycle_state",
    "reconstruct_circular_phase_state",
]

_MAX_NODES = 32
_MAX_EDGES = 50


@dataclass(frozen=True)
class PhaseCycleGeometry:
    """Ordered simple support and an integer fundamental cycle basis.

    Edges use node indices with tail < head. Incidence columns have -1 at
    the tail and +1 at the head. Each fundamental cycle traverses its chord
    tail-to-head first and returns through the tree; chord columns therefore
    form the identity. Its rows are a basis over the integers, not merely a
    basis over GF(2). Node insertion order supplies coordinates only.
    """

    nodes: tuple[Any, ...]
    edges: tuple[tuple[int, int], ...]
    incidence: tuple[tuple[int, ...], ...]
    tree_edge_indices: tuple[int, ...]
    fundamental_cycles: tuple[tuple[int, ...], ...]
    cycle_rows: tuple[tuple[int, ...], ...]
    bridge_edge_indices: tuple[int, ...]
    cycle_rank: int
    short_cycle_span_rank: int
    scope: tuple[str, ...] = (
        "connected_simple_undirected_phase_support_without_self_loops",
        "node_order_and_cycle_basis_are_coordinates_not_physical_selectors",
        "all_support_edges_retained_independently_of_transport_weights",
        "integer_fundamental_cycles_and_exact_rational_short_cycle_rank",
        "32_node_50_edge_evaluation_budget_is_not_a_physical_restriction",
        "no_graph_write_live_phase_admission_or_trajectory_claim",
    )

    @property
    def short_cycle_consensus_only(self) -> bool:
        """Sufficient topology condition for the stated equal-capacity lock.

        If all support edges are admitted, all gaps are strictly acute, and
        equal capacities follow the configured positive-coupling sine law,
        full triangle/square span excludes nonuniform locks. False means
        this obstruction is unavailable; it does not establish a lock.
        """
        return self.short_cycle_span_rank == self.cycle_rank


@dataclass(frozen=True)
class PhaseCycleState:
    """Exact circular reconstruction and a sufficient symbolic sine check.

    ``nodal_turns`` lie in [0,1), fixing node zero's common-rotation gauge.
    For edge (u,v), ``nodal_turns[v]-nodal_turns[u]-edge_turns[e]`` equals
    the displayed integer offset. Each symbolic row contains the nonzero
    integer coefficients of ``sin(2*pi*abs_turn)`` in that node's incoming
    sine sum. Empty rows prove balance by oddness, without evaluating sine.
    Other identities between sine values are deliberately not inferred.
    """

    geometry: PhaseCycleGeometry
    edge_turns: Vector
    cycle_periods: tuple[int, ...]
    nodal_turns: Vector
    edge_integer_offsets: tuple[int, ...]
    symbolic_sine_coefficients: tuple[tuple[tuple[Fraction, int], ...], ...]
    sine_balance_status: str
    scope: tuple[str, ...] = (
        "declared_exact_turns_are_angles_divided_by_mathematical_two_pi",
        "integral_fundamental_periods_certify_circular_reconstruction",
        "strict_acute_support_does_not_infer_a_tighter_configured_U3_gate",
        "odd_cancellation_is_sufficient_but_not_a_complete_sine_decision",
        "equal_capacity_sine_lock_interpretation_requires_its_supplied_law",
        "no_binary64_phase_lock_trajectory_stability_or_sector_generation",
    )


@dataclass(frozen=True)
class AcuteCyclePeriodAssessment:
    """One exact necessary period bound for a strictly acute phase sector.

    With fundamental cycle rows C, supplied integral periods k and a nonzero
    combination r, every acute edge-turn vector t with C*t=k must satisfy
    ``abs(r*k) < sum(abs(r*C))/4``. Equality also obstructs strict acuteness.
    Passing this one inequality establishes neither a realizable sector nor
    zero nodal sine divergence. The combination is a supplied witness, not a
    search over all witnesses or a selected circulation of a dynamical law.
    """

    geometry: PhaseCycleGeometry
    cycle_periods: tuple[int, ...]
    cycle_combination: Vector
    combined_edge_chain: Vector
    combined_period: Fraction
    strict_period_bound: Fraction
    strict_bound_margin: Fraction
    obstruction_certified: bool
    status: str
    proof_id: str = "strict_acute_joint_cycle_period_bound"
    scope: tuple[str, ...] = (
        "rederived_integer_fundamental_cycle_basis_on_complete_supplied_support",
        "integral_periods_and_nonzero_cycle_combination_in_the_same_basis",
        "edge_turns_would_require_absolute_value_strictly_less_than_one_quarter",
        "joint_edge_chain_retains_cancellation_between_overlapping_cycles",
        "nonpositive_exact_margin_obstructs_every_acute_state_in_the_sector",
        "positive_margin_passes_one_necessary_bound_not_sector_or_equilibrium_existence",
        "no_sine_current_decision_Hessian_law_stability_or_trajectory_admission",
        "no_search_for_witness_support_event_or_component_state_closure",
        "public_dataclass_construction_and_projection_do_not_authenticate_provenance",
    )

    def to_dict(self):
        from ..sdk.relational_reports import _project, _validate_label

        for node in self.geometry.nodes:
            _validate_label(node)
        return {
            "schema": "tnfr.acute-cycle-period-assessment.v1",
            "report": _project(self),
        }


def _adjacency(node_count, edges, selected=None):
    adjacency = [[] for _ in range(node_count)]
    indices = range(len(edges)) if selected is None else selected
    for index in indices:
        left, right = edges[index]
        adjacency[left].append((right, index))
        adjacency[right].append((left, index))
    return tuple(tuple(sorted(row)) for row in adjacency)


def _tree_edges(adjacency):
    reached = {0}
    queue = deque([0])
    result = []
    while queue:
        node = queue.popleft()
        for neighbor, edge_index in adjacency[node]:
            if neighbor not in reached:
                reached.add(neighbor)
                queue.append(neighbor)
                result.append(edge_index)
    if len(reached) != len(adjacency):
        raise ValueError("phase support must be connected")
    return tuple(result)


def _tree_path(adjacency, source, target):
    parents = {source: None}
    queue = deque([source])
    while queue:
        node = queue.popleft()
        if node == target:
            break
        for neighbor, _ in adjacency[node]:
            if neighbor not in parents:
                parents[neighbor] = node
                queue.append(neighbor)
    path = [target]
    while path[-1] != source:
        path.append(parents[path[-1]])
    return tuple(reversed(path))


def _cycle_row(cycle, edge_indices):
    row = [0] * len(edge_indices)
    for left, right in zip(cycle, cycle[1:] + cycle[:1]):
        edge = (min(left, right), max(left, right))
        row[edge_indices[edge]] += 1 if left < right else -1
    return tuple(row)


def _short_cycle_rank(adjacency, edge_indices):
    rows = []
    neighbor_sets = tuple({node for node, _ in row} for row in adjacency)
    for start in range(len(adjacency)):
        # Minimum vertex first, smaller of the two orientations second.
        # This enumerates all simple triangles/squares, including chords.
        paths = [(start,)]
        while paths:
            path = paths.pop()
            if len(path) >= 3 and start in neighbor_sets[path[-1]]:
                if path[1] < path[-1]:
                    rows.append(
                        tuple(
                            Fraction(value) for value in _cycle_row(path, edge_indices)
                        )
                    )
            if len(path) == 4:
                continue
            for neighbor, _ in adjacency[path[-1]]:
                if neighbor > start and neighbor not in path:
                    paths.append(path + (neighbor,))
    return exact_rank(rows)


def _derive(nodes, edges):
    if type(nodes) is not tuple or not 2 <= len(nodes) <= _MAX_NODES:
        raise ValueError("phase support requires 2 to 32 ordered nodes")
    if len(set(nodes)) != len(nodes):
        raise ValueError("phase support nodes must be unique")
    if type(edges) is not tuple or len(edges) > _MAX_EDGES:
        raise ValueError("phase support permits at most 50 ordered edges")
    for edge in edges:
        if (
            type(edge) is not tuple
            or len(edge) != 2
            or any(type(index) is not int for index in edge)
            or not 0 <= edge[0] < edge[1] < len(nodes)
        ):
            raise ValueError(
                "edges must be oriented index pairs with 0 <= tail < head < n"
            )
    if edges != tuple(sorted(set(edges))):
        raise ValueError(
            "phase support edges must be unique and lexicographically ordered"
        )
    adjacency = _adjacency(len(nodes), edges)
    tree_edges = _tree_edges(adjacency)
    tree = _adjacency(len(nodes), edges, tree_edges)
    tree_set = set(tree_edges)
    edge_indices = {edge: index for index, edge in enumerate(edges)}
    cycles = tuple(
        (left,) + _tree_path(tree, right, left)[:-1]
        for index, (left, right) in enumerate(edges)
        if index not in tree_set
    )
    cycle_rows = tuple(_cycle_row(cycle, edge_indices) for cycle in cycles)
    incidence = tuple(
        tuple(int(node == right) - int(node == left) for left, right in edges)
        for node in range(len(nodes))
    )
    bridges = tuple(
        index
        for index in range(len(edges))
        if all(row[index] == 0 for row in cycle_rows)
    )
    return PhaseCycleGeometry(
        nodes=nodes,
        edges=edges,
        incidence=incidence,
        tree_edge_indices=tree_edges,
        fundamental_cycles=cycles,
        cycle_rows=cycle_rows,
        bridge_edge_indices=bridges,
        cycle_rank=len(edges) - len(nodes) + 1,
        short_cycle_span_rank=_short_cycle_rank(adjacency, edge_indices),
    )


def derive_phase_cycle_geometry(graph) -> PhaseCycleGeometry:
    """Read simple connected phase support without inspecting attributes.

    Zero/negative/missing transport weights have no role in this topology
    calculation. Direction, multiedges and loops are unsupported. The 32-node,
    50-edge budget bounds triangle/square rank evaluation; it is an execution
    limit, not a mathematical or physical premise. No live field is required.
    """
    if graph.is_directed() or graph.is_multigraph():
        raise ValueError("phase support must be simple and undirected")
    nodes = tuple(graph.nodes())
    indices = {node: index for index, node in enumerate(nodes)}
    edges = tuple(
        sorted(
            (min(indices[left], indices[right]), max(indices[left], indices[right]))
            for left, right in graph.edges()
        )
    )
    return _derive(nodes, edges)


def _rebuild(geometry):
    if type(geometry) is not PhaseCycleGeometry:
        raise TypeError("geometry must be a PhaseCycleGeometry")
    rebuilt = _derive(geometry.nodes, geometry.edges)
    if geometry != rebuilt:
        raise ValueError("geometry derived fields do not match its primitive support")
    return rebuilt


def assess_acute_cycle_periods(
    geometry, *, cycle_periods, cycle_combination
) -> AcuteCyclePeriodAssessment:
    """Test one exact joint necessary bound on declared integral cycle periods.

    Period entries are non-Boolean integers in the geometry's fundamental
    cycle order. The combination has the same length and must be nonzero;
    exact rational coefficients retain their values, while other supported
    real coefficients use the shared represented-real boundary. All derived
    geometry fields are rebuilt before use. A tree has no nonzero cycle
    combination and therefore cannot supply a witness to this reader.

    The returned bound uses the combined edge chain, including cancellation
    on shared edges. A nonpositive strict margin excludes all edge turns in
    (-1/4,1/4) with the supplied periods. A positive margin passes only this
    necessary inequality; it does not solve phase reconstruction or certify
    sine balance, local recovery or the existence of an equilibrium.
    """
    from .relational_observations import _ordered

    reference = _rebuild(geometry)
    rank = reference.cycle_rank
    periods = _ordered(cycle_periods, "cycle_periods", limit=rank + 1)
    if len(periods) != rank:
        raise ValueError("cycle_periods must match the fundamental cycle order")
    if any(
        isinstance(value, bool) or not isinstance(value, Integral) for value in periods
    ):
        raise TypeError("cycle_periods must contain non-Boolean integers")
    periods = tuple(int(value) for value in periods)
    combination = ordered_vector(
        _ordered(cycle_combination, "cycle_combination", limit=rank + 1),
        "cycle_combination",
    )
    if len(combination) != rank:
        raise ValueError("cycle_combination must match the fundamental cycle order")
    if not any(combination):
        raise ValueError("cycle_combination must be nonzero")
    chain = tuple(
        sum(
            (
                coefficient * row[edge]
                for coefficient, row in zip(combination, reference.cycle_rows)
            ),
            Fraction(0),
        )
        for edge in range(len(reference.edges))
    )
    period = sum(
        (coefficient * value for coefficient, value in zip(combination, periods)),
        Fraction(0),
    )
    bound = sum(map(abs, chain), Fraction(0)) / 4
    margin = bound - abs(period)
    obstructed = margin <= 0
    return AcuteCyclePeriodAssessment(
        geometry=reference,
        cycle_periods=periods,
        cycle_combination=combination,
        combined_edge_chain=chain,
        combined_period=period,
        strict_period_bound=bound,
        strict_bound_margin=margin,
        obstruction_certified=obstructed,
        status="obstructed" if obstructed else "necessary_bound_passed",
    )


def _integral_turn_reconstruction(reference, values):
    """Share exact traversal without choosing an acute or circular sine domain."""
    raw_periods = tuple(
        sum((sign * value for sign, value in zip(row, values)), Fraction(0))
        for row in reference.cycle_rows
    )
    if any(period.denominator != 1 for period in raw_periods):
        raise ValueError("fundamental cycle periods must be integral turns")
    tree = _adjacency(
        len(reference.nodes), reference.edges, reference.tree_edge_indices
    )
    lifts = {0: Fraction(0)}
    queue = deque([0])
    while queue:
        node = queue.popleft()
        for neighbor, edge_index in tree[node]:
            if neighbor not in lifts:
                sign = 1 if node < neighbor else -1
                lifts[neighbor] = lifts[node] + sign * values[edge_index]
                queue.append(neighbor)
    nodal_turns = tuple(lifts[index] % 1 for index in range(len(reference.nodes)))
    raw_offsets = tuple(
        nodal_turns[right] - nodal_turns[left] - value
        for (left, right), value in zip(reference.edges, values)
    )
    if any(offset.denominator != 1 for offset in raw_offsets):
        raise RuntimeError("integral cycle periods lost circular reconstruction")
    return dict(
        geometry=reference,
        edge_turns=values,
        cycle_periods=tuple(int(period) for period in raw_periods),
        nodal_turns=nodal_turns,
        edge_integer_offsets=tuple(int(offset) for offset in raw_offsets),
    )


def _sine_coefficients(reference, values, *, fold_circle=False):
    """Collect sufficient exact sine identities without numerical zero tests."""
    coefficients = [{} for _ in reference.nodes]
    for (left, right), value in zip(reference.edges, values):
        if fold_circle:
            value = (value + Fraction(1, 2)) % 1 - Fraction(1, 2)
            if abs(value) == Fraction(1, 2):
                continue
        if not value:
            continue
        magnitude = abs(value)
        if fold_circle:
            magnitude = min(magnitude, Fraction(1, 2) - magnitude)
        sign = 1 if value > 0 else -1
        for node, coefficient in ((left, sign), (right, -sign)):
            coefficients[node][magnitude] = (
                coefficients[node].get(magnitude, 0) + coefficient
            )
    return tuple(
        tuple(
            (value, coefficient)
            for value, coefficient in sorted(row.items())
            if coefficient
        )
        for row in coefficients
    )


def reconstruct_phase_cycle_state(geometry, *, edge_turns) -> PhaseCycleState:
    """Reconstruct a strictly acute rational-turn field, rejecting bad periods.

    Exact rational inputs retain their values; other supported real inputs
    mean their materialized binary64 rational values in *turns*. A rounded
    radian divided by ``math.tau`` is not an exact-angle certificate.
    Integer fundamental periods are necessary and sufficient for circular
    reconstruction. All supplied geometry fields are rederived before use.

    The sufficient sine check collects terms of equal absolute turn using
    only ``sin(-a)=-sin(a)`` and ``sin(0)=0``. If every coefficient vanishes,
    the ideal incoming sine sum is zero at every node. Otherwise the result
    is unresolved, not a claim that the state fails to lock. Capacities, K,
    tighter U3 gates and the actual producer remain separate admissions.
    """
    reference = _rebuild(geometry)
    values = ordered_vector(edge_turns, "edge_turns")
    if len(values) != len(reference.edges):
        raise ValueError("edge_turns must match the ordered support edges")
    if any(abs(value) >= Fraction(1, 4) for value in values):
        raise ValueError("edge_turns must be strictly acute: abs(turn) < 1/4")
    reconstruction = _integral_turn_reconstruction(reference, values)
    symbolic = _sine_coefficients(reference, values)
    return PhaseCycleState(
        **reconstruction,
        symbolic_sine_coefficients=symbolic,
        sine_balance_status=(
            "proved_by_odd_cancellation" if not any(symbolic) else "unresolved"
        ),
    )


@dataclass(frozen=True)
class CircularPhaseState:
    """Exact circular reconstruction without an acute-domain assertion.

    Edge inputs retain their declared real lifts in turns. ``cycle_periods``
    are their integral circulations, not a principal winding assignment at an
    antipodal edge. ``nodal_turns`` fix node zero at zero modulo one common
    phase origin. Sine balance uses periodicity, oddness, half-turn zeros and
    supplementary-angle reflection; unresolved rows are not disproved balance.
    """

    geometry: PhaseCycleGeometry
    edge_turns: Vector
    cycle_periods: tuple[int, ...]
    nodal_turns: Vector
    edge_integer_offsets: tuple[int, ...]
    symbolic_sine_coefficients: tuple[tuple[tuple[Fraction, int], ...], ...]
    sine_balance_status: str
    scope: tuple[str, ...] = (
        "exact_rational_turns_and_integral_cycle_periods_on_complete_support",
        "one_common_phase_gauge_no_quotient_of_labels_reflections_or_orientations",
        "edge_lift_periods_are_not_principal_windings_at_antipodal_edges",
        "periodicity_oddness_and_supplementary_sine_reflection_are_sufficient_identities",
        "symbolic_exact_angles_are_not_binary64_radian_equilibria",
        "no_acute_U3_native_resultant_stability_or_evolution_admission",
    )

    def to_dict(self):
        from ..sdk.relational_reports import _project, _validate_label

        for node in self.geometry.nodes:
            _validate_label(node)
        return {"schema": "tnfr.circular-phase-state.v1", "report": _project(self)}


def reconstruct_circular_phase_state(geometry, *, edge_turns) -> CircularPhaseState:
    """Reconstruct declared exact circular turns with a sufficient sine check.

    This separate reader admits nonacute and antipodal edges. The original
    ``reconstruct_phase_cycle_state`` retains its strict acute domain and
    oddness-only certificate. No phase law or future motion follows from this
    reconstruction. Represented real inputs retain the shared turns boundary;
    dividing a rounded radian by a rounded tau does not certify that angle.
    """
    reference = _rebuild(geometry)
    values = ordered_vector(edge_turns, "edge_turns")
    if len(values) != len(reference.edges):
        raise ValueError("edge_turns must match the ordered support edges")
    reconstruction = _integral_turn_reconstruction(reference, values)
    symbolic = _sine_coefficients(reference, values, fold_circle=True)
    return CircularPhaseState(
        **reconstruction,
        symbolic_sine_coefficients=symbolic,
        sine_balance_status=(
            "proved_by_period_reflection_cancellation"
            if not any(symbolic)
            else "unresolved"
        ),
    )


@dataclass(frozen=True)
class BridgeTreeHessianInertia:
    """Conditional composition of declared relative phase-Hessian inertias.

    Each component contributes its supplied (positive, negative, zero) counts
    after removing one constant phase direction. Nonzero signed bridge terms
    connect the components as a tree, so their differences can be independent
    coordinates. The report checks this combinatorial contract; it does not
    authenticate the component Hessians, actual bridge weights or a state.
    """

    component_inertias: tuple[tuple[int, int, int], ...]
    bridges: tuple[tuple[int, int, int], ...]
    total_nodes: int
    relative_inertia: tuple[int, int, int]
    common_phase_nullity: int = 1
    proof_id: str = "bridge_tree_phase_Hessian_exact_congruence"
    scope: tuple[str, ...] = (
        "component_relative_inertias_are_supplied_geometric_premises_not_verified_Hessians",
        "each_component_has_one_removed_common_phase_direction_and_retains_relative_zeros",
        "nonzero_bridge_quadratic_terms_connect_disjoint_components_as_a_tree",
        "bridge_signs_are_supplied_not_inferred_from_phase_or_support_data",
        "arbitrary_bridge_attachment_vertices_do_not_change_congruence_inertia",
        "relative_inertia_orders_positive_negative_zero_phase_dimensions",
        "one_global_phase_direction_removed_without_discarding_component_relative_zeros",
        "no_equilibrium_stability_basin_law_or_live_graph_admission",
    )

    def to_dict(self):
        from ..sdk.relational_reports import _project

        return {
            "schema": "tnfr.bridge-tree-hessian-inertia.v1",
            "report": _project(self),
        }


def compose_bridge_tree_hessian_inertia(
    component_inertias, bridges
) -> BridgeTreeHessianInertia:
    """Add relative component inertias and nonzero bridge signs on a tree.

    Ordered component triples contain nonnegative non-Boolean integer counts.
    A zero-dimensional triple represents a singleton component. Each ordered
    bridge triple is (component_index, component_index, sign), with sign +/-1.
    The supplied component quotient forms must annihilate their respective
    common phase shifts; they may retain relative zero directions. Under those
    premises an invertible tree coordinate change separates their forms from
    one scalar term per bridge. No numerical eigenvalue calculation, actual
    Hessian verification, equilibrium check or dynamical conclusion is made.
    """
    from .relational_observations import _ordered

    raw_components = _ordered(component_inertias, "component_inertias")
    if not raw_components:
        raise ValueError("component_inertias must be nonempty")
    components = []
    for raw in raw_components:
        counts = _ordered(raw, "component inertia", limit=4)
        if len(counts) != 3:
            raise ValueError("each component inertia requires three counts")
        if any(
            isinstance(value, bool) or not isinstance(value, Integral)
            for value in counts
        ):
            raise TypeError("component inertia counts must be non-Boolean integers")
        if any(value < 0 for value in counts):
            raise ValueError("component inertia counts must be nonnegative")
        components.append(tuple(int(value) for value in counts))
    components = tuple(components)
    raw_bridges = _ordered(bridges, "bridges")
    if len(raw_bridges) != len(components) - 1:
        raise ValueError("a bridge tree requires one fewer bridge than components")
    admitted_bridges, edges, seen = [], [], set()
    for raw in raw_bridges:
        values = _ordered(raw, "bridge", limit=4)
        if len(values) != 3:
            raise ValueError("each bridge requires two component indices and one sign")
        if any(
            isinstance(value, bool) or not isinstance(value, Integral)
            for value in values
        ):
            raise TypeError("bridge indices and signs must be non-Boolean integers")
        left, right, sign = (int(value) for value in values)
        if not 0 <= left < len(components) or not 0 <= right < len(components):
            raise ValueError("bridge indices must belong to the component order")
        if left == right:
            raise ValueError("bridge trees do not admit self-connections")
        if sign not in (-1, 1):
            raise ValueError("bridge signs must be -1 or +1")
        edge = (min(left, right), max(left, right))
        if edge in seen:
            raise ValueError("bridge component pairs must be unique")
        seen.add(edge)
        edges.append(edge)
        admitted_bridges.append((left, right, sign))
    # Connectedness together with m-1 distinct edges excludes every cycle.
    _tree_edges(_adjacency(len(components), tuple(edges)))
    relative = tuple(sum(row[i] for row in components) for i in range(3))
    positive = sum(sign == 1 for _, _, sign in admitted_bridges)
    negative = len(admitted_bridges) - positive
    return BridgeTreeHessianInertia(
        component_inertias=components,
        bridges=tuple(admitted_bridges),
        total_nodes=sum(relative) + len(components),
        relative_inertia=(relative[0] + positive, relative[1] + negative, relative[2]),
    )


@dataclass(frozen=True)
class C5SineCriticalSet:
    """Factorized exact critical phases on two C5 rings and one intermediary.

    Each local option gives oriented increments around either supplied cycle,
    without principal reduction: a or 1/2-a, where |a|<1/4. Two independently
    selected zero/half-turn bridge gaps complete the state. All labels and
    orientations remain distinct; only one common phase rotation is removed.
    The finite classification concerns unit pairwise sine currents, not a
    stability, basin-selection or dynamics certificate.
    """

    geometry: PhaseCycleGeometry
    cycles: tuple[tuple[Any, ...], tuple[Any, ...]]
    cycle_indices: tuple[tuple[int, ...], tuple[int, ...]]
    mediator: Any
    bridge_edge_indices: tuple[int, int]
    cycle_edge_turn_options: tuple[Vector, ...]
    cycle_principal_sine_turns: Vector
    cycle_supplementary_masks: tuple[int, ...]
    bridge_turn_options: tuple[Fraction, Fraction]
    relative_state_count: int
    scope: tuple[str, ...] = (
        "exact_two_disjoint_C5_rings_one_intermediary_and_two_unit_sine_bridges",
        "all_cycle_sine_branches_including_nonacute_states_and_zero_current_patterns",
        "both_zero_and_half_turn_bridge_branches_retained",
        "finite_rational_branch_closure_not_numerical_roots_or_trajectory_sampling",
        "thirty_cycle_options_and_four_bridge_choices_not_eager_cartesian_materialization",
        "one_common_phase_origin_removed_labels_and_reflections_retained",
        "no_principal_winding_assignment_at_antipodal_edges",
        "no_capacity_loss_law_stability_basin_or_convergence_admission",
    )

    def reconstruct(self, *, cycle_choices, bridge_turns) -> CircularPhaseState:
        """Select one exact member without enumerating the product.

        ``bridge_turns`` follows the stored ``bridge_edge_indices`` order in
        the geometry, not the supplied ring order. Each bridge value is the
        increment along that edge's lower-to-higher node-index orientation.
        """
        return _critical_member(self, cycle_choices, bridge_turns)[2]

    def phase_hessian_inertia(
        self, *, cycle_choices, bridge_turns
    ) -> C5PhaseHessianInertia:
        """Count exact Hessian signs without applying a dynamical stability law."""
        choices, bridges, state = _critical_member(self, cycle_choices, bridge_turns)
        counts = tuple(
            self.cycle_supplementary_masks[index].bit_count() for index in choices
        )
        bridge_signs = tuple(1 if turn == 0 else -1 for turn in bridges)
        negative_modes = tuple(_c5_negative_phase_modes(k) for k in counts)
        component_inertias = tuple((4 - k, k, 0) for k in negative_modes) + ((0, 0, 0),)
        component_of = {
            node: component
            for component, cycle in enumerate(self.cycle_indices)
            for node in cycle
        }
        component_of[self.geometry.nodes.index(self.mediator)] = 2
        component_bridges = tuple(
            (component_of[left], component_of[right], sign)
            for edge, sign in zip(self.bridge_edge_indices, bridge_signs)
            for left, right in (self.geometry.edges[edge],)
        )
        composed = compose_bridge_tree_hessian_inertia(
            component_inertias, component_bridges
        )
        return C5PhaseHessianInertia(
            state=state,
            cycles=self.cycles,
            cycle_choices=choices,
            bridge_turns=bridges,
            cycle_negative_edge_counts=counts,
            bridge_signs=bridge_signs,
            relative_inertia=composed.relative_inertia,
        )

    @property
    def phase_hessian_index_counts(self) -> tuple[int, ...]:
        """Count catalog members by relative negative index, without its Cartesian product."""
        _rebuild_critical_set(self)
        local = [0] * 5
        for mask in self.cycle_supplementary_masks:
            local[_c5_negative_phase_modes(mask.bit_count())] += 1
        counts = [1]
        for factor in (local, local, (1, 1), (1, 1)):
            result = [0] * (len(counts) + len(factor) - 1)
            for left, coefficient in enumerate(counts):
                for right, value in enumerate(factor):
                    result[left + right] += coefficient * value
            counts = result
        return tuple(counts)

    def to_dict(self):
        from ..sdk.relational_reports import _project

        _validate_critical_set_labels(self)
        return {"schema": "tnfr.c5-sine-critical-set.v1", "report": _project(self)}


@dataclass(frozen=True)
class C5PhaseHessianInertia:
    """Exact signs of the phase-storage Hessian at one circular critical state.

    ``relative_inertia`` is (positive, negative, zero) after removing one
    common phase origin. Each cycle is a diagonal signed edge form restricted
    to the zero-sum increment subspace; its five common cosine magnitudes are
    strictly positive. Bridge signs then add independently. The result is
    geometric and does not by itself classify a capacity/phase evolution law.
    """

    state: CircularPhaseState
    cycles: tuple[tuple[Any, ...], tuple[Any, ...]]
    cycle_choices: tuple[int, int]
    bridge_turns: tuple[Fraction, Fraction]
    cycle_negative_edge_counts: tuple[int, int]
    bridge_signs: tuple[int, int]
    relative_inertia: tuple[int, int, int]
    common_phase_nullity: int = 1
    proof_id: str = "two_C5_sine_phase_Hessian_exact_congruence"
    scope: tuple[str, ...] = (
        "same_exact_critical_member_and_full_unit_support_as_factorized_catalog",
        "relative_inertia_orders_positive_negative_zero_phase_dimensions",
        "odd_cycle_constraint_removes_one_sign_without_a_relative_zero_mode",
        "independent_bridge_cosine_signs_add_to_cycle_congruence_inertia",
        "one_common_phase_zero_mode_is_not_a_relative_degeneracy",
        "no_numerical_eigenscan_tolerance_or_stability_law_admission",
        "no_basin_selection_convergence_rate_or_finite_radius_claim",
    )

    def to_dict(self):
        from ..sdk.relational_reports import _project

        _validate_phase_hessian_labels(self)
        return {"schema": "tnfr.c5-phase-hessian-inertia.v1", "report": _project(self)}


def _validate_phase_hessian_labels(report):
    from ..sdk.relational_reports import _validate_label

    for node in report.state.geometry.nodes:
        _validate_label(node)
    for cycle in report.cycles:
        for node in cycle:
            _validate_label(node)


def _c5_negative_phase_modes(negative_edges):
    return negative_edges - int(negative_edges >= 3)


def _rebuild_critical_set(critical):
    rebuilt = classify_c5_sine_critical_set(critical.geometry, cycles=critical.cycles)
    if rebuilt != critical:
        raise ValueError("critical-set fields do not match their declared support")
    return rebuilt


def _critical_member(critical, cycle_choices, bridge_turns):
    """Share exact option admission and one-member reconstruction with all readers."""
    from .relational_observations import _ordered

    _rebuild_critical_set(critical)
    choices = _ordered(cycle_choices, "cycle_choices", limit=3)
    if len(choices) != 2 or any(
        type(choice) is not int
        or not 0 <= choice < len(critical.cycle_edge_turn_options)
        for choice in choices
    ):
        raise ValueError("cycle_choices requires two nonboolean option indices")
    bridges = ordered_vector(bridge_turns, "bridge_turns")
    if len(bridges) != 2 or any(
        turn not in critical.bridge_turn_options for turn in bridges
    ):
        raise ValueError("bridge_turns requires two exact zero or half turns")
    edge_indices = {edge: index for index, edge in enumerate(critical.geometry.edges)}
    values = [Fraction(0)] * len(critical.geometry.edges)
    for cycle, choice in zip(critical.cycle_indices, choices):
        pattern = critical.cycle_edge_turn_options[choice]
        for i, j, turn in zip(cycle, cycle[1:] + cycle[:1], pattern):
            values[edge_indices[min(i, j), max(i, j)]] = turn if i < j else -turn
    for edge, turn in zip(critical.bridge_edge_indices, bridges):
        values[edge] = turn
    state = reconstruct_circular_phase_state(
        critical.geometry, edge_turns=tuple(values)
    )
    if any(state.symbolic_sine_coefficients):
        raise ArithmeticError("critical branch reconstruction lost exact sine balance")
    return choices, bridges, state


def _validate_critical_set_labels(report):
    from ..sdk.relational_reports import _validate_label

    for node in report.geometry.nodes:
        _validate_label(node)
    for cycle in report.cycles:
        for node in cycle:
            _validate_label(node)
    _validate_label(report.mediator)


def _c5_sine_templates():
    """Solve all labeled branch closures using exact integer arithmetic."""
    options, currents, masks = [], [], []
    for mask in range(32):
        k = mask.bit_count()
        # m=(5-2k)*a+k/2 with |a|<1/4 implies -1 <= m <= 3.
        # Since five is odd, 5-2k never vanishes; the quarter-turn endpoints
        # cannot close a five-cycle and are not silently identified.
        for period in range(-1, 4):
            a = Fraction(2 * period - k, 2 * (5 - 2 * k))
            if abs(a) >= Fraction(1, 4):
                continue
            options.append(
                tuple(Fraction(1, 2) - a if mask & (1 << j) else a for j in range(5))
            )
            currents.append(a)
            masks.append(mask)
    return tuple(options), tuple(currents), tuple(masks)


def classify_c5_sine_critical_set(geometry, *, cycles) -> C5SineCriticalSet:
    """Classify exact unit-sine equilibria on a declared two-C5 support.

    ``cycles`` contains two ordered five-node cycles using the labels in
    ``geometry.nodes``. They must cover ten distinct nodes; the remaining
    intermediary has exactly one bridge to each ring. No extra edge, missing
    node or alternate topology inherits this classification.
    """
    from .relational_observations import _ordered

    reference = _rebuild(geometry)
    rows = tuple(
        _ordered(cycle, "cycle", limit=6)
        for cycle in _ordered(cycles, "cycles", limit=3)
    )
    if len(rows) != 2 or any(len(cycle) != 5 or len(set(cycle)) != 5 for cycle in rows):
        raise ValueError("cycles requires two ordered five-node simple cycles")
    positions = {node: index for index, node in enumerate(reference.nodes)}
    covered = set(rows[0]) | set(rows[1])
    if (
        len(reference.nodes) != 11
        or len(covered) != 10
        or not covered <= positions.keys()
    ):
        raise ValueError(
            "two disjoint cycles must cover ten of the eleven support nodes"
        )
    indices = tuple(tuple(positions[node] for node in row) for row in rows)
    cycle_edges = {
        (min(i, j), max(i, j))
        for row in indices
        for i, j in zip(row, row[1:] + row[:1])
    }
    remaining = set(reference.edges) - cycle_edges
    mediator = next(node for node in reference.nodes if node not in covered)
    hidden = positions[mediator]
    if (
        len(reference.edges) != 12
        or not cycle_edges <= set(reference.edges)
        or len(remaining) != 2
        or any(hidden not in edge for edge in remaining)
        or any(
            sum(bool(set(edge) & set(row)) for edge in remaining) != 1
            for row in indices
        )
    ):
        raise ValueError("support must be exactly two C5 rings and one bridge per ring")
    options, currents, masks = _c5_sine_templates()
    return C5SineCriticalSet(
        geometry=reference,
        cycles=rows,
        cycle_indices=indices,
        mediator=mediator,
        bridge_edge_indices=reference.bridge_edge_indices,
        cycle_edge_turn_options=options,
        cycle_principal_sine_turns=currents,
        cycle_supplementary_masks=masks,
        bridge_turn_options=(Fraction(0), Fraction(1, 2)),
        relative_state_count=len(options) ** 2
        * 2 ** len(reference.bridge_edge_indices),
    )


@dataclass(frozen=True)
class PhaseChordExtension:
    """Integer basis change for exactly one added undirected support edge.

    ``inherited_edge_indices`` maps the old edge order into the new one.
    ``inherited_cycle_coordinates`` and ``created_cycle_coordinates`` express
    the embedded old cycles and the added edge's old-tree cycle in the new
    fundamental basis. ``after_cycle_coordinates`` is the inverse matrix,
    with columns ordered as the inherited cycles followed by the created one.
    All these identities hold over the integers, independently of phases.
    """

    before: PhaseCycleGeometry
    after: PhaseCycleGeometry
    added_edge_index: int
    inherited_edge_indices: tuple[int, ...]
    inherited_cycle_coordinates: tuple[tuple[int, ...], ...]
    created_cycle: tuple[int, ...]
    created_cycle_coordinates: tuple[int, ...]
    after_cycle_coordinates: tuple[tuple[int, ...], ...]
    scope: tuple[str, ...] = (
        "same_ordered_nodes_and_exactly_one_added_simple_support_edge",
        "integer_cycle_lattice_extension_with_verified_inverse_coordinates",
        "created_cycle_uses_added_edge_and_the_before_geometry_tree",
        "basis_coordinates_do_not_select_physical_phase_sectors",
        "no_graph_mutation_operator_execution_or_phase_admission",
    )


@dataclass(frozen=True)
class PhaseChordReset:
    """Detached exact endpoint comparison across a one-edge extension.

    The before/after inputs are acute rational-turn states. Inherited periods
    use the same old cycles on both supports. The created period additionally
    uses the newly available old-tree cycle. Its before-phase value is a
    counterfactual closure using the shortest new-edge gap in the old field;
    it need not be acute or U3-admissible. At an antipodal old pair, that gap
    and its period/change are unavailable rather than assigned a branch.

    Endpoint period changes do not identify a continuous phase path, a branch
    crossing, an executed operator, or an autonomous sector-generation law.
    """

    extension: PhaseChordExtension
    before: PhaseCycleState
    after: PhaseCycleState
    inherited_periods_before: tuple[int, ...]
    inherited_periods_after: tuple[int, ...]
    inherited_period_changes: tuple[int, ...]
    created_period_after: int
    created_edge_turn_before_phase: Fraction | None
    created_period_before_phase: int | None
    created_period_phase_change: int | None
    phase_unchanged_modulo_rotation: bool
    scope: tuple[str, ...] = (
        "validated_exact_turn_endpoint_states_not_live_binary64_radians",
        "inherited_and_created_periods_use_one_adapted_integer_cycle_basis",
        "before_phase_closure_is_counterfactual_not_runtime_U3_admission",
        "antipodal_counterfactual_chord_has_no_selected_shortest_branch",
        "endpoint_period_difference_does_not_certify_continuous_winding_slip",
        "no_graph_mutation_event_provenance_or_autonomous_generation_claim",
    )


def _row_combination(coefficients, rows, width):
    return tuple(
        sum(coefficient * row[column] for coefficient, row in zip(coefficients, rows))
        for column in range(width)
    )


def derive_phase_chord_extension(before, after) -> PhaseChordExtension:
    """Rederive and compare supports, retaining their exact cycle lattice.

    Both supports must be connected and simple, with identical ordered nodes
    and exactly one added edge. The existing geometry evaluation budget is
    retained. The added edge together with the old spanning tree supplies a
    new integral fundamental basis. Coordinate rows are read at chord columns
    and checked by exact reconstruction and both inverse identities; no
    floating rank, inversion or phase data enter this result.
    """
    before = _rebuild(before)
    after = _rebuild(after)
    if before.nodes != after.nodes:
        raise ValueError("chord extension requires the same ordered nodes")
    before_edges = set(before.edges)
    after_edges = set(after.edges)
    if not before_edges < after_edges or len(after_edges - before_edges) != 1:
        raise ValueError(
            "chord extension requires exactly one added edge and none removed"
        )
    after_indices = {edge: index for index, edge in enumerate(after.edges)}
    inherited_indices = tuple(after_indices[edge] for edge in before.edges)
    added_index = after_indices[(after_edges - before_edges).pop()]
    added_left, added_right = after.edges[added_index]
    old_tree = _adjacency(len(before.nodes), before.edges, before.tree_edge_indices)
    created_cycle = (added_left,) + _tree_path(old_tree, added_right, added_left)[:-1]
    created_row = _cycle_row(created_cycle, after_indices)
    embedded_rows = []
    for row in before.cycle_rows:
        embedded = [0] * len(after.edges)
        for old_index, new_index in enumerate(inherited_indices):
            embedded[new_index] = row[old_index]
        embedded_rows.append(tuple(embedded))
    adapted_rows = tuple(embedded_rows) + (created_row,)
    old_tree_indices = set(before.tree_edge_indices)
    adapted_chords = tuple(
        inherited_indices[index]
        for index in range(len(before.edges))
        if index not in old_tree_indices
    ) + (added_index,)
    new_tree_indices = set(after.tree_edge_indices)
    after_chords = tuple(
        index for index in range(len(after.edges)) if index not in new_tree_indices
    )
    forward = tuple(tuple(row[index] for index in after_chords) for row in adapted_rows)
    inverse = tuple(
        tuple(row[index] for index in adapted_chords) for row in after.cycle_rows
    )
    rank = after.cycle_rank
    identity = tuple(tuple(int(i == j) for j in range(rank)) for i in range(rank))
    if (
        tuple(
            _row_combination(row, after.cycle_rows, len(after.edges)) for row in forward
        )
        != adapted_rows
        or tuple(
            _row_combination(row, adapted_rows, len(after.edges)) for row in inverse
        )
        != after.cycle_rows
        or tuple(_row_combination(row, inverse, rank) for row in forward) != identity
        or tuple(_row_combination(row, forward, rank) for row in inverse) != identity
    ):
        raise RuntimeError("chord extension lost its integral cycle basis identities")
    return PhaseChordExtension(
        before=before,
        after=after,
        added_edge_index=added_index,
        inherited_edge_indices=inherited_indices,
        inherited_cycle_coordinates=forward[:-1],
        created_cycle=created_cycle,
        created_cycle_coordinates=forward[-1],
        after_cycle_coordinates=inverse,
    )


def _rebuild_state(state):
    if type(state) is not PhaseCycleState:
        raise TypeError("state must be a PhaseCycleState")
    rebuilt = reconstruct_phase_cycle_state(state.geometry, edge_turns=state.edge_turns)
    if state != rebuilt:
        raise ValueError(
            "state derived fields do not match its support and exact turns"
        )
    return rebuilt


def observe_phase_chord_reset(before, after) -> PhaseChordReset:
    """Compare exact states without inventing a path between their endpoints.

    Supplied states, geometry and derived fields are reconstructed before use.
    The old phase's hypothetical new-edge gap is the unique shortest rational
    turn except at an exactly antipodal pair. No continuous trajectory, live
    admission, event provenance or binary64-radian conversion is inferred.
    """
    before = _rebuild_state(before)
    after = _rebuild_state(after)
    extension = derive_phase_chord_extension(before.geometry, after.geometry)
    inherited_after = tuple(
        sum(
            coefficient * period
            for coefficient, period in zip(row, after.cycle_periods)
        )
        for row in extension.inherited_cycle_coordinates
    )
    created_after = sum(
        coefficient * period
        for coefficient, period in zip(
            extension.created_cycle_coordinates, after.cycle_periods
        )
    )
    adapted_periods = inherited_after + (created_after,)
    if (
        tuple(
            sum(
                coefficient * period
                for coefficient, period in zip(row, adapted_periods)
            )
            for row in extension.after_cycle_coordinates
        )
        != after.cycle_periods
    ):
        raise RuntimeError("chord reset lost its integral period coordinates")
    left, right = after.geometry.edges[extension.added_edge_index]
    positive_gap = (before.nodal_turns[right] - before.nodal_turns[left]) % 1
    if positive_gap == Fraction(1, 2):
        old_phase_gap = old_phase_period = period_phase_change = None
    else:
        old_phase_gap = (
            positive_gap if positive_gap < Fraction(1, 2) else positive_gap - 1
        )
        extended_turns = [old_phase_gap] * len(after.geometry.edges)
        for old_index, new_index in enumerate(extension.inherited_edge_indices):
            extended_turns[new_index] = before.edge_turns[old_index]
        after_indices = {edge: index for index, edge in enumerate(after.geometry.edges)}
        created_row = _cycle_row(extension.created_cycle, after_indices)
        raw_period = sum(
            (
                coefficient * turn
                for coefficient, turn in zip(created_row, extended_turns)
            ),
            Fraction(0),
        )
        if raw_period.denominator != 1:
            raise RuntimeError("before-phase chord closure lost integral winding")
        old_phase_period = int(raw_period)
        period_phase_change = created_after - old_phase_period
    return PhaseChordReset(
        extension=extension,
        before=before,
        after=after,
        inherited_periods_before=before.cycle_periods,
        inherited_periods_after=inherited_after,
        inherited_period_changes=tuple(
            new - old for old, new in zip(before.cycle_periods, inherited_after)
        ),
        created_period_after=created_after,
        created_edge_turn_before_phase=old_phase_gap,
        created_period_before_phase=old_phase_period,
        created_period_phase_change=period_phase_change,
        phase_unchanged_modulo_rotation=before.nodal_turns == after.nodal_turns,
    )
