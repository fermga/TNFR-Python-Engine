"""Static invariant separators for the held SHA law after its pendant cut.

The public boundary re-admits the primitive source. The detached cut uses the
shared complete frozen-isolate law, its own component means and edge storage.
Two fixed grounded solves supply the form and bare-phase coefficients. No
connected response, earlier bound, trajectory or continuation cell is consumed.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
from fractions import Fraction as Q

from .._exact_time import finite_represented_real
from ..mathematics._exact_linear_algebra import (
    exact_matrix_inverse,
    exact_matrix_product,
)
from ..mathematics._rational_interval import INTERVAL_METHOD, I, pi_interval, sqrt
from ..physics._sine_admission import _sine_model_coefficients
from ..physics.relational_sine_support import (
    SineFrozenIsolateComparison,
    _capture_support_rows,
    _support_topology,
    bound_relational_sine_frozen_isolates,
)
from ..sdk._state import copy_graph_state
from .local_capacity_history import MODEL, TIMES
from .maintained_parent_thol import _signature
from .maintained_sha_motion_source import validate_live_sha_successor


@dataclass(frozen=True)
class CutSeparatorSource:
    """Exact represented source and the cut host's own conserved mean weights.

    ``edges`` and ``cut_edges`` use original node positions. ``host_edges``
    uses positions within ``host_indices``. The removed conductance remains
    only the marked portrait scale. The isolated child has no host mean weight.
    """

    nodes: tuple
    edges: tuple[tuple[int, int], ...]
    conductances: tuple[Q, ...]
    epi: tuple[Q, ...]
    phase: tuple[Q, ...]
    capacity: tuple[Q, ...]
    connected_degrees: tuple[int, ...]
    cut_degrees: tuple[int, ...]
    parent_index: int
    child_index: int
    host_indices: tuple[int, ...]
    cut_edges: tuple[tuple[int, int], ...]
    cut_conductances: tuple[Q, ...]
    host_edges: tuple[tuple[int, int], ...]
    host_conductances: tuple[Q, ...]
    reference_conductance: Q
    structural_time: Q
    host_mean_weights: tuple[Q, ...]
    host_alpha: tuple[Q, ...]
    form_mean: Q
    phase_lift_mean: Q
    form_offset: Q
    phase_offset: Q


@dataclass(frozen=True)
class GroundedResponse:
    """Exact grounded solution, full residual and verified inverse provenance."""

    laplacian: tuple[tuple[Q, ...], ...]
    ell: tuple[Q, ...]
    ground_index: int
    grounded_inverse: tuple[tuple[Q, ...], ...]
    solution: tuple[Q, ...]
    residual: tuple[Q, ...]
    resistance: Q
    inverse_verified: bool


@dataclass(frozen=True)
class CutSeparatorGate:
    """A strict sufficient inequality, not an observation of cut rotation.

    Only a positive lower margin passes. Nonpositive or unresolved margins,
    and missing joint premises, are unavailable certificates. They never
    establish that a cut turn occurs or that the two laws are equivalent.
    """

    margin: I | None
    status: str
    reason: str | None = None


@dataclass(frozen=True)
class ShaCutSeparator:
    """Point-preparation certificates under the complete held cut law.

    Either passing separator keeps the marked portrait in an open half-plane
    and implies abs(continuous net cut angle)<pi on every finite window.
    A mathematically passing gate can retain ``None`` for its numerical radius
    enclosure if a required positive denominator is unresolved on the grid.
    The radius intervals enclose the analytic lower-bound formulas; only their
    lower endpoints are certified numerical floors for the actual radius.
    ``joint_margin`` remains a diagnostic when its independent premises fail.
    """

    source: CutSeparatorSource
    cut: SineFrozenIsolateComparison
    weighted_response: GroundedResponse
    unweighted_response: GroundedResponse
    storage: I
    raw_edge_chart_margins: tuple[I, ...]
    energy_chart_margin: I
    chart_gate: CutSeparatorGate
    energy_gate: CutSeparatorGate
    form_gate: CutSeparatorGate
    joint_quadratic: I
    joint_margin: I
    joint_gate: CutSeparatorGate
    form_radius_lower: I | None
    joint_radius_lower: I | None
    radius_unavailable_reasons: tuple[str, ...]
    persistent_source_unchanged: bool
    source_receipts: tuple[dict, ...] = ()
    arithmetic_method: str = INTERVAL_METHOD
    scope: tuple[str, ...] = (
        "fresh_primitive_source_and_shared_complete_frozen_isolate_cut_law",
        "remove_only_pendant_bridge_keep_all_coordinates_capacities_and_clock",
        "cut_host_own_degree_over_capacity_means_and_nonnegative_edge_storage",
        "fixed_last_host_node_ground_exact_weighted_and_bare_laplacian_solves",
        "exact_full_residual_and_two_sided_inverse_verification",
        "form_separator_independent_of_lifted_edge_chart",
        "joint_separator_requires_all_actual_lifted_edges_and_storage_below_two",
        "strict_separator_implies_nonzero_radius_and_abs_net_cut_angle_below_pi",
        "unavailable_gate_neither_observes_cut_rotation_nor_proves_equivalence",
        "point_preparation_not_a_uniform_uncertainty_family",
        "no_connected_response_retained_cells_new_flow_solver_event_or_trajectory",
        "no_formation_autonomous_identity_or_physical_identification_claim",
    )


def _capture_cut_source(graph, *, parent, child, reference_model):
    """Admit connected primitives once and derive exact cut component weights."""
    if _sine_model_coefficients(reference_model) != (Q(0), Q(1), Q(1)):
        raise ValueError(
            "cut separator requires the beta=1 unit-exchange zero-loss law"
        )
    _, clock = finite_represented_real(graph.graph.get("_t"), "structural clock")
    if any("weight" not in data for _, _, data in graph.edges(data=True)):
        raise ValueError("cut separator requires stored conductance on every edge")
    captured = _capture_support_rows(graph, reference_model)
    nodes = captured["nodes"]
    edges, neighbors = _support_topology(nodes, captured["edges"])
    weights, capacity = captured["conductances"], captured["capacity"]
    if any(value <= 0 for value in weights + capacity):
        raise ValueError("cut separator requires positive capacities and conductances")
    positions = {node: index for index, node in enumerate(nodes)}
    if parent == child or parent not in positions or child not in positions:
        raise ValueError("distinct parent and child must belong to the source")
    a, c = positions[parent], positions[child]
    degrees = tuple(map(len, neighbors))
    if degrees[c] != 1 or a not in neighbors[c]:
        raise ValueError("the marked child must be pendant at the marked parent")
    if degrees[a] < 2:
        raise ValueError("the marked parent requires a nontrivial cut host")
    bridge = next(index for index, edge in enumerate(edges) if set(edge) == {a, c})
    cut_edges = tuple(edge for index, edge in enumerate(edges) if index != bridge)
    cut_weights = tuple(value for index, value in enumerate(weights) if index != bridge)
    cut_degrees = tuple(
        degree - (index in (a, c)) for index, degree in enumerate(degrees)
    )
    host = tuple(index for index in range(len(nodes)) if index != c)
    local = {index: position for position, index in enumerate(host)}
    host_edges = tuple((local[i], local[j]) for i, j in cut_edges)
    # A pendant deletion from an admitted connected graph leaves this host
    # connected. Re-admit that explicit support rather than assume a partition.
    _support_topology(
        tuple(nodes[i] for i in host), tuple((nodes[i], nodes[j]) for i, j in cut_edges)
    )
    rho = tuple(Q(cut_degrees[i]) / capacity[i] for i in host)
    total = sum(rho, Q(0))
    alpha = tuple(value / total for value in rho)

    def mean(values):
        return sum(
            (weight * values[i] for weight, i in zip(alpha, host, strict=True)), Q(0)
        )

    form_mean, phase_mean = mean(captured["epi"]), mean(captured["phase"])
    return CutSeparatorSource(
        nodes,
        edges,
        weights,
        captured["epi"],
        captured["phase"],
        capacity,
        degrees,
        cut_degrees,
        a,
        c,
        host,
        cut_edges,
        cut_weights,
        host_edges,
        cut_weights,
        weights[bridge],
        clock,
        rho,
        alpha,
        form_mean,
        phase_mean,
        captured["epi"][c] - form_mean,
        captured["phase"][c] - phase_mean,
    )


def _laplacian(dimension, edges, weights):
    rows = [[Q(0) for _ in range(dimension)] for _ in range(dimension)]
    for (i, j), weight in zip(edges, weights, strict=True):
        rows[i][i] += weight
        rows[j][j] += weight
        rows[i][j] -= weight
        rows[j][i] -= weight
    return tuple(tuple(row) for row in rows)


def _grounded_response(laplacian, ell):
    """Solve at the fixed final-node ground and verify every exact identity.

    This private helper consumes the admitted connected positive host's
    Laplacian and its zero-sum vector. The inverse is the shared exact owner;
    no floating pseudoinverse, alternate ground or tolerance is used.
    """
    dimension = len(laplacian)
    if dimension < 2 or len(ell) != dimension or sum(ell, Q(0)) != 0:
        raise ValueError("grounded response requires a nontrivial zero-sum host vector")
    if any(len(row) != dimension for row in laplacian):
        raise ValueError("host Laplacian must be square")
    if any(sum(row, Q(0)) != 0 for row in laplacian) or any(
        laplacian[i][j] != laplacian[j][i]
        for i in range(dimension)
        for j in range(dimension)
    ):
        raise ValueError("host Laplacian must be symmetric with zero row sums")
    ground = dimension - 1
    reduced = tuple(tuple(row[:ground]) for row in laplacian[:ground])
    inverse = exact_matrix_inverse(reduced)
    identity = tuple(tuple(Q(i == j) for j in range(ground)) for i in range(ground))
    if (
        exact_matrix_product(reduced, inverse) != identity
        or exact_matrix_product(inverse, reduced) != identity
    ):
        raise ValueError("exact grounded inverse verification failed")
    solution = tuple(
        sum((value * rhs for value, rhs in zip(row, ell[:ground], strict=True)), Q(0))
        for row in inverse
    ) + (Q(0),)
    residual = tuple(
        sum((value * answer for value, answer in zip(row, solution, strict=True)), Q(0))
        - rhs
        for row, rhs in zip(laplacian, ell, strict=True)
    )
    if any(residual) or solution[ground] != 0:
        raise ValueError("exact full Laplacian residual verification failed")
    resistance = sum(
        (rhs * answer for rhs, answer in zip(ell, solution, strict=True)), Q(0)
    )
    if resistance <= 0:
        raise ValueError("cut separator requires strictly positive quadratic response")
    return GroundedResponse(
        laplacian, ell, ground, inverse, solution, residual, resistance, True
    )


def _strict_gate(margin, *, reason):
    if margin is None:
        return CutSeparatorGate(None, "unavailable", reason)
    margin = I.coerce(margin)
    if margin.lo > 0:
        return CutSeparatorGate(margin, "pass")
    outcome = "nonpositive_recipe" if margin.hi <= 0 else "unresolved_interval"
    return CutSeparatorGate(margin, "unavailable", f"{reason}:{outcome}")


def _fresh_cut(graph, source, reference_model):
    cut_graph = copy_graph_state(graph)
    cut_graph.remove_edge(
        source.nodes[source.parent_index], source.nodes[source.child_index]
    )
    cut = bound_relational_sine_frozen_isolates(
        cut_graph, reference_model=reference_model
    )
    labels = tuple((source.nodes[i], source.nodes[j]) for i, j in source.cut_edges)
    if (
        any(
            getattr(cut, name) != getattr(source, name)
            for name in ("nodes", "epi", "phase", "capacity")
        )
        or cut.edges != labels
        or cut.conductances != source.cut_conductances
        or cut.degrees != source.cut_degrees
    ):
        raise ValueError(
            "fresh cut changed a retained primitive or support association"
        )
    components = {frozenset(component) for component in cut.component_indices}
    c = source.child_index
    if components != {
        frozenset(source.host_indices),
        frozenset((c,)),
    } or cut.isolate_indices != (c,):
        raise ValueError(
            "the cut must leave exactly the connected host and isolated child"
        )
    if cut.inverse_phase_metric[c] is not None or any(
        getattr(cut, name)[c] != I(0)
        for name in ("pressure", "form_rates", "phase_rates")
    ):
        raise ValueError("the detached child must use the complete frozen-isolate law")
    if cut.continuous_loss != 0:
        raise ValueError("the cut separator requires conservative component storage")
    return cut


def _bound_cut_separator(graph, *, parent, child, reference_model):
    """Evaluate only the fixed static recipes on a freshly admitted graph."""
    before = _signature(graph)
    source = _capture_cut_source(
        graph, parent=parent, child=child, reference_model=reference_model
    )
    cut = _fresh_cut(graph, source, reference_model)
    host_parent = source.host_indices.index(source.parent_index)
    ell = tuple(
        Q(i == host_parent) - weight for i, weight in enumerate(source.host_alpha)
    )
    dimension = len(source.host_indices)
    weighted = _grounded_response(
        _laplacian(dimension, source.host_edges, source.host_conductances), ell
    )
    bare = _grounded_response(
        _laplacian(dimension, source.host_edges, (Q(1),) * len(source.host_edges)), ell
    )
    # Every term of this supplied edge storage is nonnegative. Intersect its
    # enclosure with that proved range, without changing the captured state.
    if cut.storage.hi < 0:
        raise ValueError("nonnegative cut storage has an inconsistent enclosure")
    storage = I(max(Q(0), cut.storage.lo), cut.storage.hi)
    pi = pi_interval()
    chart_margins = tuple(
        pi - abs(source.phase[j] - source.phase[i]) for i, j in source.cut_edges
    )
    chart_margin = I(
        min(value.lo for value in chart_margins),
        min(value.hi for value in chart_margins),
    )
    chart_gate = _strict_gate(chart_margin, reason="actual_lifted_edge_chart")
    energy_margin = 2 - storage
    energy_gate = _strict_gate(energy_margin, reason="host_energy_chart_barrier")
    rx, rt = weighted.resistance, bare.resistance
    rstar, delta = source.form_offset, source.phase_offset
    w = source.reference_conductance
    form_margin = I(rstar**2) - 2 * storage * rx
    form_gate = _strict_gate(form_margin, reason="form_separator")
    # Divide exact rational coefficients before interval materialization.
    # This retains positive subgrid resistances without a spurious zero divisor.
    quadratic = I(rstar**2 / rx) + I(4 * delta**2 / rt) / (pi**2)
    joint_margin = quadratic - 2 * storage
    if chart_gate.status == energy_gate.status == "pass":
        joint_gate = _strict_gate(joint_margin, reason="joint_separator")
    else:
        missing = tuple(
            name
            for name, gate in (
                ("actual_lifted_edge_chart", chart_gate),
                ("host_energy_chart_barrier", energy_gate),
            )
            if gate.status != "pass"
        )
        joint_gate = CutSeparatorGate(
            joint_margin, "unavailable", "joint_premises:" + ",".join(missing)
        )

    form_radius = joint_radius = None
    radius_reasons = []
    if form_gate.status == "pass":
        denominator = abs(I(rstar)) + sqrt(2 * storage * rx)
        if denominator.lo > 0:
            candidate = sqrt(I(w)) * form_margin / denominator
            if candidate.lo > 0:
                form_radius = candidate
        if form_radius is None:
            radius_reasons.append("form_positive_radius_enclosure_unresolved")
    if joint_gate.status == "pass":
        root_q, root_energy = sqrt(quadratic), sqrt(2 * storage)
        support_norm = sqrt(
            I(16 * delta**2 / rt**2) / (pi**4) + I(rstar**2 / (w * rx**2))
        )
        denominator = root_q + root_energy
        if denominator.lo > 0 and support_norm.lo > 0:
            candidate = root_q * joint_margin / denominator / support_norm
            if candidate.lo > 0:
                joint_radius = candidate
        if joint_radius is None:
            radius_reasons.append("joint_positive_radius_enclosure_unresolved")
    unchanged = _signature(graph) == before
    if not unchanged:
        raise ValueError("read-only cut separator changed persistent source state")
    return ShaCutSeparator(
        source,
        cut,
        weighted,
        bare,
        storage,
        chart_margins,
        energy_margin,
        chart_gate,
        energy_gate,
        form_gate,
        quadratic,
        joint_margin,
        joint_gate,
        form_radius,
        joint_radius,
        tuple(radius_reasons),
        unchanged,
    )


def prepare_sha_cut_separator(admitted):
    """Re-admit the fixed post-SHA preparation and its two static cut gates.

    This is a source-specific static evaluation requiring its own prospective
    freeze. Cached predictions or caller reports are not mathematical premises.
    Historical source certificates retain the shared reader's separate scope;
    no post-SHA flow, retained continuation or new trajectory is evaluated.
    """
    admitted = validate_live_sha_successor(admitted)
    graph = admitted.graph
    before = _signature(graph)
    if len(graph) != 21 or graph.number_of_edges() != 53:
        raise ValueError(
            "SHA cut separator requires the retained 21-node 53-edge source"
        )
    _, clock = finite_represented_real(graph.graph.get("_t"), "structural clock")
    if clock != TIMES[-1]:
        raise ValueError("SHA cut separator requires the retained structural clock")
    result = _bound_cut_separator(
        graph, parent=2, child="2_sub_0", reference_model=MODEL
    )
    if result.source.structural_time != TIMES[-1]:
        raise ValueError("SHA cut separator requires the retained structural clock")
    if _signature(graph) != before:
        raise ValueError("read-only SHA cut separator changed persistent source state")
    return replace(result, source_receipts=deepcopy(admitted.receipts))
