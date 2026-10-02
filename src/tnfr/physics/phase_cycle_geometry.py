"""Exact phase-support topology and rational-turn reconstruction.

Fundamental cycles are coordinates for the integer cycle lattice, not physical
selectors. Edge turns mean an exact angle divided by mathematical ``2*pi``;
they are not inferred from rounded radians or from inverse trigonometry.
The reconstruction certifies an acute circular configuration. A separate
symbolic odd-cancellation check can establish sine balance, but its failure is
only unresolved. Neither operation executes or derives a phase evolution law.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from fractions import Fraction
from typing import Any

from ..mathematics.krylov import exact_rank
from ._cycle_algebra import Vector, ordered_vector

__all__ = [
    "PhaseChordExtension",
    "PhaseChordReset",
    "PhaseCycleGeometry",
    "PhaseCycleState",
    "derive_phase_chord_extension",
    "derive_phase_cycle_geometry",
    "observe_phase_chord_reset",
    "reconstruct_phase_cycle_state",
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
    coefficients = [{} for _ in reference.nodes]
    for (left, right), value in zip(reference.edges, values):
        if not value:
            continue
        magnitude = abs(value)
        sign = 1 if value > 0 else -1
        for node, coefficient in ((left, sign), (right, -sign)):
            coefficients[node][magnitude] = (
                coefficients[node].get(magnitude, 0) + coefficient
            )
    symbolic = tuple(
        tuple(
            (value, coefficient)
            for value, coefficient in sorted(row.items())
            if coefficient
        )
        for row in coefficients
    )
    return PhaseCycleState(
        geometry=reference,
        edge_turns=values,
        cycle_periods=tuple(int(period) for period in raw_periods),
        nodal_turns=nodal_turns,
        edge_integer_offsets=tuple(int(offset) for offset in raw_offsets),
        symbolic_sine_coefficients=symbolic,
        sine_balance_status=(
            "proved_by_odd_cancellation" if not any(symbolic) else "unresolved"
        ),
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
