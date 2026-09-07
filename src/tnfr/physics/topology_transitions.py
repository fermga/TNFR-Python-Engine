r"""Read-only certificates for observed TNFR nodal-topology transitions.

The canonical topology classifier labels a graph ``radial``, ``annular`` or
``multinodal`` from its unit-source structural-potential geometry.  The label
is therefore a property of the sampled graph geometry, not of the instantaneous
EPI, phase or pressure values.  This module records label changes across a
declared sequence and places them beside the canonical structural telemetry.

Every reported delta is an endpoint difference.  A label change between two
samples brackets an observed transition; it does not locate a continuous-time
crossing or establish a precursor.  The classifier's existing calibrated cuts
are reused verbatim and no predictive threshold is introduced here.  Equal
node ids define persistence when the two supports use the same ids.  Otherwise,
exact state-preserving graph isomorphisms are aligned before pointwise
differences, so a pure node relabeling is not reported as structural evolution.
Every step reports which correspondence was available.  When graph symmetry
leaves several correspondences and they imply different pointwise changes,
the ambiguous components are suppressed instead of selecting the matcher's
first result.  Detecting that ambiguity enumerates isomorphisms and can have
factorial worst-case cost on highly symmetric graphs.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any

import networkx as nx
from networkx.algorithms import isomorphism as iso

from ..alias import get_attr
from ..constants.aliases import (
    ALIAS_DEPI,
    ALIAS_DNFR,
    ALIAS_EPI,
    ALIAS_THETA,
    ALIAS_VF,
)
from ..metrics.common import compute_coherence
from ..metrics.sense_index import compute_Si
from ._edge_semantics import effective_edge_length
from ._helpers import wrap_angle
from .fields import classify_nodal_topology, _validate_nodal_topology_alpha
from .telemetry import compute_structural_telemetry

__all__ = [
    "NodalTopologySnapshot",
    "NodalTopologyStep",
    "NodalTopologyTransitionCertificate",
    "capture_nodal_topology_snapshot",
    "detect_nodal_topology_transitions",
]


@dataclass(frozen=True)
class NodalTopologySnapshot:
    """One topology label and its simultaneous structural observations.

    Per-node tuples follow ``node_order``.  They detach the returned certificate
    from later graph mutation.  ``classification_alpha`` is the distance-kernel
    exponent passed to :func:`classify_nodal_topology`; it is not a transition
    threshold.  ``epi``, ``frequency``, canonical wrapped ``phase``, ``dnfr``
    and ``depi`` retain the five effective scalar state channels used by exact
    state-isomorphism alignment. Edge state retains transport conductance and
    effective structural length separately.
    """

    index: int
    time: float | None
    topology: str
    centers: tuple[Any, ...]
    node_order: tuple[Any, ...]
    n_nodes: int
    n_edges: int
    is_directed: bool
    edge_order: tuple[tuple[Any, Any], ...]
    edge_conductance: tuple[float, ...]
    edge_length: tuple[float, ...]
    epi: tuple[float, ...]
    frequency: tuple[float, ...]
    phase: tuple[float, ...]
    dnfr: tuple[float, ...]
    depi: tuple[float, ...]
    classification_alpha: float
    concentration: float
    dispersion: float
    coherence: float
    mean_abs_dnfr: float
    mean_abs_depi: float
    sense_index_mean: float
    coherence_length: float
    centrality: tuple[float, ...]
    structural_potential: tuple[float, ...]
    phase_gradient: tuple[float, ...]
    phase_curvature: tuple[float, ...]
    phase_current: tuple[float, ...]
    dnfr_flux: tuple[float, ...]
    sense_index: tuple[float, ...]


@dataclass(frozen=True)
class NodalTopologyStep:
    """Exact endpoint differences between two consecutive snapshots.

    ``node_mapping`` contains the before-to-after correspondence.  Pointwise
    field deltas follow its before-node order in ``common_nodes`` and equal
    ``after(mapped node)-before(node)``.  Equal ids define persistence on an
    unchanged named support; otherwise exact state isomorphisms remove pure
    relabelings, followed by partial equal ids or a bare topology isomorphism.
    The chosen rule is explicitly reported.  ``node_alignment_candidate_count``
    records the number of full isomorphisms when that route is used.  An empty
    ``node_mapping`` with ``node_alignment_ambiguous`` means no arbitrary
    correspondence was published.  If candidates imply different pointwise
    results, ``mapping_dependent_deltas_available`` is false and those tuples
    are empty; aggregate deltas remain available.  Values without a counterpart
    stay in the snapshots instead of receiving an artificial zero.  A
    non-finite coherence length makes ``coherence_length_delta`` undefined as
    ``None``.
    """

    before_index: int
    after_index: int
    before_time: float | None
    after_time: float | None
    before_topology: str
    after_topology: str
    before_centers: tuple[Any, ...]
    after_centers: tuple[Any, ...]
    topology_changed: bool
    centers_changed: bool | None
    node_mapping: tuple[tuple[Any, Any], ...]
    node_alignment_status: str
    node_alignment_candidate_count: int
    node_alignment_ambiguous: bool
    mapping_dependent_deltas_available: bool
    common_nodes: tuple[Any, ...]
    added_nodes: tuple[Any, ...]
    removed_nodes: tuple[Any, ...]
    node_support_changed: bool
    node_count_delta: int
    edge_count_delta: int
    edge_direction_changed: bool
    common_edges: tuple[tuple[tuple[Any, Any], tuple[Any, Any]], ...]
    added_edges: tuple[tuple[Any, Any], ...]
    removed_edges: tuple[tuple[Any, Any], ...]
    edge_support_changed: bool
    edge_conductance_delta: tuple[float, ...]
    edge_weight_changed: bool | None
    edge_length_delta: tuple[float, ...]
    edge_length_changed: bool | None
    concentration_delta: float
    dispersion_delta: float
    coherence_delta: float
    mean_abs_dnfr_delta: float
    mean_abs_depi_delta: float
    sense_index_mean_delta: float
    coherence_length_delta: float | None
    centrality_delta: tuple[float, ...]
    structural_potential_delta: tuple[float, ...]
    phase_gradient_delta: tuple[float, ...]
    phase_curvature_delta: tuple[float, ...]
    phase_current_delta: tuple[float, ...]
    dnfr_flux_delta: tuple[float, ...]
    sense_index_delta: tuple[float, ...]


@dataclass(frozen=True)
class NodalTopologyTransitionCertificate:
    """Descriptive certificate for a supplied sequence of graph states."""

    snapshots: tuple[NodalTopologySnapshot, ...]
    steps: tuple[NodalTopologyStep, ...]
    transitions: tuple[NodalTopologyStep, ...]
    label_sequence: tuple[str, ...]
    transition_indices: tuple[int, ...]
    classification_alpha: float
    is_read_only_diagnostic: bool
    predictive_status: str
    scope: str


def _validate_alpha(alpha: float) -> float:
    return _validate_nodal_topology_alpha(alpha)


def _validate_index(index: int) -> int:
    if isinstance(index, bool) or not isinstance(index, Integral) or index < 0:
        raise ValueError("index must be a non-negative integer")
    return int(index)


def _validate_time(time: float | None) -> float | None:
    if time is None:
        return None
    if isinstance(time, bool) or not isinstance(time, Real):
        raise ValueError("time must be finite when provided")
    try:
        value = float(time)
    except (TypeError, ValueError) as exc:
        raise ValueError("time must be finite when provided") from exc
    if not math.isfinite(value):
        raise ValueError("time must be finite when provided")
    return value


def _validate_graph(graph: Any) -> tuple[Any, ...]:
    if not all(
        hasattr(graph, name)
        for name in ("nodes", "edges", "number_of_nodes", "number_of_edges")
    ):
        raise TypeError("each state must provide the NetworkX graph interface")
    nodes = tuple(graph.nodes())
    if not nodes:
        raise ValueError("topology-transition states must contain at least one node")
    if not hasattr(graph, "is_directed") or not hasattr(graph, "is_multigraph"):
        raise TypeError("each state must provide the NetworkX graph interface")
    if graph.is_multigraph():
        raise ValueError("topology-transition snapshots require simple graphs")
    return nodes


def _edge_snapshot(
    graph: Any,
) -> tuple[
    tuple[tuple[Any, Any], ...],
    tuple[float, ...],
    tuple[float, ...],
]:
    edges: list[tuple[Any, Any]] = []
    conductance: list[float] = []
    length: list[float] = []
    for source, target, data in graph.edges(data=True):
        value = data.get("weight", 1.0)
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError("edge weights must be finite nonnegative real scalars")
        weight = float(value)
        if not math.isfinite(weight) or weight < 0.0:
            raise ValueError("edge weights must be finite nonnegative real scalars")
        try:
            edge_length = effective_edge_length(data)
        except ValueError as exc:
            raise ValueError(
                "edge lengths must be finite nonnegative real scalars"
            ) from exc
        edges.append((source, target))
        conductance.append(weight)
        length.append(edge_length)
    return tuple(edges), tuple(conductance), tuple(length)


def _ordered_field(
    values: Mapping[Any, Any], node_order: tuple[Any, ...], name: str
) -> tuple[float, ...]:
    try:
        result = tuple(float(values[node]) for node in node_order)
    except KeyError as exc:
        raise RuntimeError(f"{name} omitted a graph node") from exc
    if not all(math.isfinite(value) for value in result):
        raise ValueError(f"{name} must contain finite values")
    return result


def _sense_index_values(graph: Any, node_order: tuple[Any, ...]) -> tuple[float, ...]:
    values = compute_Si(graph, inplace=False, n_jobs=1)
    if not isinstance(values, Mapping):
        raise RuntimeError("read-only sense-index computation did not return a mapping")
    return _ordered_field(values, node_order, "sense_index")


def _state_channel(
    graph: Any,
    node_order: tuple[Any, ...],
    aliases: Sequence[str],
    name: str,
    *,
    circular: bool = False,
    nonnegative: bool = False,
) -> tuple[float, ...]:
    """Detach one effective scalar state channel using engine alias defaults."""
    raw_values = tuple(
        get_attr(
            graph.nodes[node],
            aliases,
            0.0,
            strict=True,
            conv=lambda value: value,
        )
        for node in node_order
    )
    if any(
        isinstance(value, bool) or not isinstance(value, Real)
        for value in raw_values
    ):
        raise ValueError(f"{name} must contain finite real values")
    values = tuple(float(value) for value in raw_values)
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"{name} must contain finite values")
    if nonnegative and any(value < 0.0 for value in values):
        raise ValueError(f"{name} must contain nonnegative values")
    if circular:
        return tuple(wrap_angle(value) for value in values)
    return values


def capture_nodal_topology_snapshot(
    graph: Any,
    *,
    index: int = 0,
    time: float | None = None,
    alpha: float = 2.0,
) -> NodalTopologySnapshot:
    """Capture a detached topology and telemetry snapshot without evolving a graph.

    The caller must hold the graph fixed for the duration of the call.  Field
    implementations may warm internal caches, but this function does not write
    node attributes, change edges, or apply a TNFR operator.
    """

    sample_index = _validate_index(index)
    sample_time = _validate_time(time)
    exponent = _validate_alpha(alpha)
    node_order = _validate_graph(graph)
    edge_order, edge_conductance, edge_length = _edge_snapshot(graph)
    epi = _state_channel(graph, node_order, ALIAS_EPI, "EPI")
    frequency = _state_channel(
        graph, node_order, ALIAS_VF, "frequency", nonnegative=True
    )
    phase = _state_channel(
        graph, node_order, ALIAS_THETA, "phase", circular=True
    )
    dnfr = _state_channel(graph, node_order, ALIAS_DNFR, "DeltaNFR")
    depi = _state_channel(graph, node_order, ALIAS_DEPI, "dEPI")

    topology = classify_nodal_topology(graph, alpha=exponent)
    telemetry = compute_structural_telemetry(graph)
    coherence, mean_abs_dnfr, mean_abs_depi = compute_coherence(
        graph, return_means=True
    )
    sense_index = _sense_index_values(graph, node_order)
    coherence_length = float(telemetry["xi_c"])
    return NodalTopologySnapshot(
        index=sample_index,
        time=sample_time,
        topology=str(topology["topology"]),
        centers=tuple(topology["centers"]),
        node_order=node_order,
        n_nodes=int(topology["n_nodes"]),
        n_edges=int(graph.number_of_edges()),
        is_directed=bool(graph.is_directed()),
        edge_order=edge_order,
        edge_conductance=edge_conductance,
        edge_length=edge_length,
        epi=epi,
        frequency=frequency,
        phase=phase,
        dnfr=dnfr,
        depi=depi,
        classification_alpha=exponent,
        concentration=float(topology["concentration"]),
        dispersion=float(topology["dispersion"]),
        coherence=float(coherence),
        mean_abs_dnfr=float(mean_abs_dnfr),
        mean_abs_depi=float(mean_abs_depi),
        sense_index_mean=sum(sense_index) / len(sense_index),
        coherence_length=coherence_length,
        centrality=_ordered_field(topology["centrality"], node_order, "centrality"),
        structural_potential=_ordered_field(
            telemetry["phi_s"], node_order, "structural_potential"
        ),
        phase_gradient=_ordered_field(
            telemetry["grad_phi"], node_order, "phase_gradient"
        ),
        phase_curvature=_ordered_field(
            telemetry["curv_phi"], node_order, "phase_curvature"
        ),
        phase_current=_ordered_field(
            telemetry["j_phi"], node_order, "phase_current"
        ),
        dnfr_flux=_ordered_field(telemetry["j_dnfr"], node_order, "dnfr_flux"),
        sense_index=sense_index,
    )


def _pointwise_delta(
    before: NodalTopologySnapshot,
    after: NodalTopologySnapshot,
    before_values: tuple[float, ...],
    after_values: tuple[float, ...],
    node_mapping: tuple[tuple[Any, Any], ...],
    *,
    circular: bool = False,
) -> tuple[float, ...]:
    before_map = dict(zip(before.node_order, before_values))
    after_map = dict(zip(after.node_order, after_values))
    if circular:
        return tuple(
            _wrapped_phase_delta(before_map[source], after_map[target])
            for source, target in node_mapping
        )
    return tuple(
        _finite_difference(
            after_map[target], before_map[source], "pointwise field delta"
        )
        for source, target in node_mapping
    )


def _wrapped_phase_delta(before: float, after: float) -> float:
    """Return the signed circular endpoint change in ``[-pi, pi)``."""
    return wrap_angle(wrap_angle(after) - wrap_angle(before))


def _finite_difference(after: float, before: float, name: str) -> float:
    """Subtract two finite observations or reject unrepresentable output."""
    if after == before:
        return 0.0
    result = after - before
    if math.isfinite(result) and result != 0.0:
        return result

    # Recover the exact binary-rational subtraction only on exceptional
    # overflow/underflow paths.  A mathematically nonzero delta must never be
    # published as zero, infinity or NaN.
    from fractions import Fraction

    exact = Fraction.from_float(after) - Fraction.from_float(before)
    try:
        result = float(exact)
    except OverflowError as exc:
        raise ValueError(f"{name} exceeds finite floating-point range") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} exceeds finite floating-point range")
    if result == 0.0 and exact:
        raise ValueError(f"{name} is below nonzero floating-point range")
    return result


def _edge_key(
    directed: bool, source: Any, target: Any
) -> tuple[Any, Any] | frozenset[Any]:
    return (source, target) if directed else frozenset((source, target))


def _node_signature(snapshot: NodalTopologySnapshot, node: Any) -> tuple[float, ...]:
    index = snapshot.node_order.index(node)
    return tuple(
        values[index]
        for values in (
            snapshot.epi,
            snapshot.frequency,
            snapshot.phase,
            snapshot.dnfr,
            snapshot.depi,
            snapshot.centrality,
            snapshot.structural_potential,
            snapshot.phase_gradient,
            snapshot.phase_curvature,
            snapshot.phase_current,
            snapshot.dnfr_flux,
            snapshot.sense_index,
        )
    )


def _snapshot_graph(
    snapshot: NodalTopologySnapshot, *, include_state: bool
) -> nx.Graph:
    graph: nx.Graph = nx.DiGraph() if snapshot.is_directed else nx.Graph()
    for node in snapshot.node_order:
        attributes = (
            {"signature": _node_signature(snapshot, node)} if include_state else {}
        )
        graph.add_node(node, **attributes)
    for edge, weight, length in zip(
        snapshot.edge_order,
        snapshot.edge_conductance,
        snapshot.edge_length,
        strict=True,
    ):
        attributes = (
            {"conductance": weight, "length": length}
            if include_state
            else {}
        )
        graph.add_edge(*edge, **attributes)
    return graph


def _isomorphism_mappings(
    before: NodalTopologySnapshot,
    after: NodalTopologySnapshot,
    *,
    include_state: bool,
) -> tuple[tuple[tuple[Any, Any], ...], ...]:
    if before.is_directed != after.is_directed:
        return ()
    before_graph = _snapshot_graph(before, include_state=include_state)
    after_graph = _snapshot_graph(after, include_state=include_state)
    matcher_class = iso.DiGraphMatcher if before.is_directed else iso.GraphMatcher
    kwargs: dict[str, Any] = {}
    if include_state:
        kwargs["node_match"] = lambda left, right: (
            left["signature"] == right["signature"]
        )
        kwargs["edge_match"] = lambda left, right: (
            left["conductance"] == right["conductance"]
            and left["length"] == right["length"]
        )
    matcher = matcher_class(before_graph, after_graph, **kwargs)
    return tuple(
        tuple((node, mapping[node]) for node in before.node_order)
        for mapping in matcher.isomorphisms_iter()
    )


def _align_nodes(
    before: NodalTopologySnapshot, after: NodalTopologySnapshot
) -> tuple[tuple[tuple[tuple[Any, Any], ...], ...], str]:
    after_nodes = set(after.node_order)
    if set(before.node_order) == after_nodes:
        return (
            (tuple((node, node) for node in before.node_order),),
            "persistent_equal_node_ids",
        )

    exact = _isomorphism_mappings(before, after, include_state=True)
    if exact:
        status = (
            "exact_state_isomorphism"
            if len(exact) == 1
            else "ambiguous_exact_state_isomorphism"
        )
        return exact, status

    persistent = tuple(
        (node, node) for node in before.node_order if node in after_nodes
    )
    if persistent:
        return (persistent,), "persistent_equal_node_ids"

    topology = _isomorphism_mappings(before, after, include_state=False)
    if topology:
        status = (
            "bare_topology_isomorphism"
            if len(topology) == 1
            else "ambiguous_bare_topology_isomorphism"
        )
        return topology, status
    return (), "no_node_correspondence"


def _compare_edges(
    before: NodalTopologySnapshot,
    after: NodalTopologySnapshot,
    mapping: Mapping[Any, Any],
) -> tuple[
    tuple[tuple[tuple[Any, Any], tuple[Any, Any]], ...],
    tuple[tuple[Any, Any], ...],
    tuple[tuple[Any, Any], ...],
    tuple[float, ...],
    tuple[float, ...],
]:
    if before.is_directed != after.is_directed:
        return (), after.edge_order, before.edge_order, (), ()

    after_lookup = {
        _edge_key(after.is_directed, *edge): (edge, weight, length)
        for edge, weight, length in zip(
            after.edge_order,
            after.edge_conductance,
            after.edge_length,
            strict=True,
        )
    }
    used_after: set[tuple[Any, Any] | frozenset[Any]] = set()
    common: list[tuple[tuple[Any, Any], tuple[Any, Any]]] = []
    removed: list[tuple[Any, Any]] = []
    conductance_deltas: list[float] = []
    length_deltas: list[float] = []

    for edge, before_weight, before_length in zip(
        before.edge_order,
        before.edge_conductance,
        before.edge_length,
        strict=True,
    ):
        source, target = edge
        if source not in mapping or target not in mapping:
            removed.append(edge)
            continue
        key = _edge_key(
            after.is_directed, mapping[source], mapping[target]
        )
        match = after_lookup.get(key)
        if match is None:
            removed.append(edge)
            continue
        after_edge, after_weight, after_length = match
        used_after.add(key)
        common.append((edge, after_edge))
        conductance_deltas.append(
            _finite_difference(
                after_weight, before_weight, "edge-conductance delta"
            )
        )
        length_deltas.append(
            _finite_difference(
                after_length, before_length, "edge-length delta"
            )
        )

    added = tuple(
        edge
        for edge in after.edge_order
        if _edge_key(after.is_directed, *edge) not in used_after
    )
    return (
        tuple(common),
        added,
        tuple(removed),
        tuple(conductance_deltas),
        tuple(length_deltas),
    )


def _finite_scalar_delta(before: float, after: float) -> float | None:
    if math.isfinite(before) and math.isfinite(after):
        return _finite_difference(after, before, "scalar telemetry delta")
    return None


@dataclass(frozen=True)
class _MappingObservations:
    mapping: tuple[tuple[Any, Any], ...]
    centers_changed: bool
    common_edges: tuple[tuple[tuple[Any, Any], tuple[Any, Any]], ...]
    added_edges: tuple[tuple[Any, Any], ...]
    removed_edges: tuple[tuple[Any, Any], ...]
    edge_conductance_delta: tuple[float, ...]
    edge_length_delta: tuple[float, ...]
    centrality_delta: tuple[float, ...]
    structural_potential_delta: tuple[float, ...]
    phase_gradient_delta: tuple[float, ...]
    phase_curvature_delta: tuple[float, ...]
    phase_current_delta: tuple[float, ...]
    dnfr_flux_delta: tuple[float, ...]
    sense_index_delta: tuple[float, ...]

    @property
    def invariant_fingerprint(self) -> tuple[Any, ...]:
        """Return every mapping-dependent value other than node identifiers."""
        return (
            self.centers_changed,
            self.edge_conductance_delta,
            self.edge_length_delta,
            self.centrality_delta,
            self.structural_potential_delta,
            self.phase_gradient_delta,
            self.phase_curvature_delta,
            self.phase_current_delta,
            self.dnfr_flux_delta,
            self.sense_index_delta,
        )


def _observe_mapping(
    before: NodalTopologySnapshot,
    after: NodalTopologySnapshot,
    node_mapping: tuple[tuple[Any, Any], ...],
) -> _MappingObservations:
    mapping = dict(node_mapping)
    (
        common_edges,
        added_edges,
        removed_edges,
        conductance_delta,
        length_delta,
    ) = _compare_edges(before, after, mapping)

    def delta(
        before_values: tuple[float, ...], after_values: tuple[float, ...]
    ) -> tuple[float, ...]:
        return _pointwise_delta(
            before, after, before_values, after_values, node_mapping
        )

    mapped_before_centers = {
        mapping[node] for node in before.centers if node in mapping
    }
    unmapped_before_center = any(node not in mapping for node in before.centers)
    return _MappingObservations(
        mapping=node_mapping,
        centers_changed=(
            unmapped_before_center
            or mapped_before_centers != set(after.centers)
        ),
        common_edges=common_edges,
        added_edges=added_edges,
        removed_edges=removed_edges,
        edge_conductance_delta=conductance_delta,
        edge_length_delta=length_delta,
        centrality_delta=delta(before.centrality, after.centrality),
        structural_potential_delta=delta(
            before.structural_potential, after.structural_potential
        ),
        phase_gradient_delta=delta(before.phase_gradient, after.phase_gradient),
        phase_curvature_delta=_pointwise_delta(
            before,
            after,
            before.phase_curvature,
            after.phase_curvature,
            node_mapping,
            circular=True,
        ),
        phase_current_delta=delta(before.phase_current, after.phase_current),
        dnfr_flux_delta=delta(before.dnfr_flux, after.dnfr_flux),
        sense_index_delta=delta(before.sense_index, after.sense_index),
    )


def _compare_snapshots(
    before: NodalTopologySnapshot, after: NodalTopologySnapshot
) -> NodalTopologyStep:
    candidate_mappings, alignment_status = _align_nodes(before, after)
    candidate_count = len(candidate_mappings)
    ambiguous = candidate_count > 1
    observations = tuple(
        _observe_mapping(before, after, mapping)
        for mapping in candidate_mappings
    )
    identifiable = bool(observations) and all(
        observation.invariant_fingerprint
        == observations[0].invariant_fingerprint
        for observation in observations[1:]
    )

    if observations:
        selected = observations[0]
        internal_mapping = dict(selected.mapping)
    else:
        selected = _observe_mapping(before, after, ())
        internal_mapping = {}

    # Full isomorphism ambiguity changes labels, not quotient support.  Node
    # and edge correspondences themselves are withheld because NetworkX's
    # enumeration order is not a mathematical selector.
    public_mapping = () if ambiguous else selected.mapping
    if ambiguous:
        added_nodes: tuple[Any, ...] = ()
        removed_nodes: tuple[Any, ...] = ()
        node_support_changed = False
        added_edges: tuple[tuple[Any, Any], ...] = ()
        removed_edges: tuple[tuple[Any, Any], ...] = ()
        edge_support_changed = False
        common_edges: tuple[
            tuple[tuple[Any, Any], tuple[Any, Any]], ...
        ] = ()
    else:
        mapped_after_nodes = set(internal_mapping.values())
        added_nodes = tuple(
            node for node in after.node_order if node not in mapped_after_nodes
        )
        removed_nodes = tuple(
            node for node in before.node_order if node not in internal_mapping
        )
        node_support_changed = bool(added_nodes or removed_nodes)
        added_edges = selected.added_edges
        removed_edges = selected.removed_edges
        edge_support_changed = bool(added_edges or removed_edges) or (
            before.is_directed != after.is_directed
        )
        common_edges = selected.common_edges

    deltas_available = identifiable
    common_nodes = (
        tuple(source for source, _ in selected.mapping)
        if deltas_available
        else ()
    )
    if deltas_available:
        centers_changed: bool | None = selected.centers_changed
        conductance_delta = selected.edge_conductance_delta
        length_delta = selected.edge_length_delta
        edge_weight_changed: bool | None = any(
            value != 0.0 for value in conductance_delta
        )
        edge_length_changed: bool | None = any(
            value != 0.0 for value in length_delta
        )
        field_deltas = selected
    else:
        centers_changed = None
        conductance_delta = ()
        length_delta = ()
        edge_weight_changed = None
        edge_length_changed = None
        field_deltas = None

    return NodalTopologyStep(
        before_index=before.index,
        after_index=after.index,
        before_time=before.time,
        after_time=after.time,
        before_topology=before.topology,
        after_topology=after.topology,
        before_centers=before.centers,
        after_centers=after.centers,
        topology_changed=before.topology != after.topology,
        centers_changed=centers_changed,
        node_mapping=public_mapping,
        node_alignment_status=alignment_status,
        node_alignment_candidate_count=candidate_count,
        node_alignment_ambiguous=ambiguous,
        mapping_dependent_deltas_available=deltas_available,
        common_nodes=common_nodes,
        added_nodes=added_nodes,
        removed_nodes=removed_nodes,
        node_support_changed=node_support_changed,
        node_count_delta=after.n_nodes - before.n_nodes,
        edge_count_delta=after.n_edges - before.n_edges,
        edge_direction_changed=before.is_directed != after.is_directed,
        common_edges=common_edges,
        added_edges=added_edges,
        removed_edges=removed_edges,
        edge_support_changed=edge_support_changed,
        edge_conductance_delta=conductance_delta,
        edge_weight_changed=edge_weight_changed,
        edge_length_delta=length_delta,
        edge_length_changed=edge_length_changed,
        concentration_delta=_finite_difference(
            after.concentration, before.concentration, "concentration delta"
        ),
        dispersion_delta=_finite_difference(
            after.dispersion, before.dispersion, "dispersion delta"
        ),
        coherence_delta=_finite_difference(
            after.coherence, before.coherence, "coherence delta"
        ),
        mean_abs_dnfr_delta=_finite_difference(
            after.mean_abs_dnfr, before.mean_abs_dnfr, "mean pressure delta"
        ),
        mean_abs_depi_delta=_finite_difference(
            after.mean_abs_depi, before.mean_abs_depi, "mean EPI-rate delta"
        ),
        sense_index_mean_delta=_finite_difference(
            after.sense_index_mean,
            before.sense_index_mean,
            "mean sense-index delta",
        ),
        coherence_length_delta=_finite_scalar_delta(
            before.coherence_length, after.coherence_length
        ),
        centrality_delta=(field_deltas.centrality_delta if field_deltas else ()),
        structural_potential_delta=(
            field_deltas.structural_potential_delta if field_deltas else ()
        ),
        phase_gradient_delta=(
            field_deltas.phase_gradient_delta if field_deltas else ()
        ),
        phase_curvature_delta=(
            field_deltas.phase_curvature_delta if field_deltas else ()
        ),
        phase_current_delta=(
            field_deltas.phase_current_delta if field_deltas else ()
        ),
        dnfr_flux_delta=(field_deltas.dnfr_flux_delta if field_deltas else ()),
        sense_index_delta=(
            field_deltas.sense_index_delta if field_deltas else ()
        ),
    )


def detect_nodal_topology_transitions(
    graph_sequence: Sequence[Any],
    *,
    times: Sequence[float] | None = None,
    alpha: float = 2.0,
) -> NodalTopologyTransitionCertificate:
    """Record observed topology-label changes in a time-ordered graph sequence.

    ``transition_indices`` contains the index of the first sample carrying each
    new label.  When times are supplied, every transition is bracketed by its
    step's ``before_time`` and ``after_time``.  No interpolation is attempted:
    the classifier is discrete and the states between samples are unknown.
    """

    states = tuple(graph_sequence)
    if not states:
        raise ValueError("graph_sequence must contain at least one state")
    exponent = _validate_alpha(alpha)

    if times is None:
        sample_times: tuple[float | None, ...] = (None,) * len(states)
    else:
        if len(times) != len(states):
            raise ValueError("times must have one value per graph state")
        if any(value is None for value in times):
            raise ValueError("times must contain only finite values")
        parsed_times = tuple(_validate_time(value) for value in times)
        if any(
            right <= left
            for left, right in zip(parsed_times, parsed_times[1:])
            if left is not None and right is not None
        ):
            raise ValueError("times must be strictly increasing")
        sample_times = parsed_times

    snapshots = tuple(
        capture_nodal_topology_snapshot(
            graph, index=index, time=sample_times[index], alpha=exponent
        )
        for index, graph in enumerate(states)
    )
    steps = tuple(
        _compare_snapshots(before, after)
        for before, after in zip(snapshots, snapshots[1:])
    )
    transitions = tuple(step for step in steps if step.topology_changed)

    return NodalTopologyTransitionCertificate(
        snapshots=snapshots,
        steps=steps,
        transitions=transitions,
        label_sequence=tuple(snapshot.topology for snapshot in snapshots),
        transition_indices=tuple(step.after_index for step in transitions),
        classification_alpha=exponent,
        is_read_only_diagnostic=True,
        predictive_status="descriptive_finite_sequence_only",
        scope=(
            "relabeling-aligned endpoint labels, node/edge support, conductance, "
            "structural length and canonical telemetry on the supplied "
            "simple-graph sequence; "
            "ambiguous isomorphisms are enumerated (factorial worst case) and "
            "non-identifiable pointwise deltas are suppressed; no continuous "
            "crossing time, precursor, causality, or graph-family universality "
            "is certified"
        ),
    )
