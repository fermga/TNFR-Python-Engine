"""Measured THOL network inputs and sub-EPI diagnostics.

THOL uses this module to read neighboring EPI and circular phase signals before
its atomic SelfOrganization transaction. The sub-EPI amplitude is determined by
the parent form plus declared network weights after structural acceleration has
admitted the bifurcation. Acceleration is evidence for admission; it is not a
second amplitude multiplier.

The variance-based ensemble readout in this module measures alignment of
sub-EPI magnitudes only. It is not canonical C(t), a fragmentation detector, or
a U5 parent/child coherence certificate. Use
tnfr.physics.assess_u5_parent_child_coherence for an explicit U5 target.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..types import NodeId, TNFRGraph

from ..alias import get_attr
from ..constants.aliases import ALIAS_EPI, ALIAS_THETA
from ..constants.canonical import COUPLING_GENTLE, COUPLING_MODERATE
from ..mathematics.unified_numerical import np
from ._argument_validation import (
    finite_real,
    nonnegative_integer,
    reject_operator_argument,
)
from ._epi_domain import require_real_scalar_epi
from ._thol_constants import THOL_SUB_EPI_SCALING

_OPERATOR = "Self-organization metabolism"

__all__ = [
    "capture_network_signals",
    "compose_subepi_amplitude",
    "metabolize_signals_into_subepi",
    "propagate_subepi_to_network",
    "compute_cascade_depth",
    "compute_hierarchical_depth",
    "compute_propagation_radius",
    "compute_subepi_amplitude_alignment",
    "compute_subepi_collective_coherence",
    "compute_metabolic_activity_index",
]


def capture_network_signals(G: TNFRGraph, node: NodeId) -> dict[str, Any] | None:
    """Capture external vibrational patterns from coupled neighbors.

    This function implements the "perception" phase of THOL's vibrational metabolism.
    It samples the network environment around the target node, computing structural
    gradients, phase variance, and coupling strength.

    Parameters
    ----------
    G : TNFRGraph
        Graph containing the node and its network context
    node : NodeId
        Node performing metabolic capture

    Returns
    -------
    dict | None
        Network signal structure containing:
        - epi_gradient: Difference between mean neighbor EPI and node EPI
        - phase_variance: Circular variance of neighbor phases
        - neighbor_count: Number of coupled neighbors
        - coupling_strength_mean: Average phase alignment with neighbors
        - mean_neighbor_epi: Mean EPI value of neighbors
        Returns None if node has no neighbors (isolated metabolism).

    Notes
    -----
    TNFR Principle: THOL doesn't operate in vacuum—it metabolizes the network's
    vibrational field. EPI gradient represents "structural pressure" from environment.
    Circular phase variance indicates "complexity" of external patterns to digest.

    Examples
    --------
    >>> import networkx as nx
    >>> graph = nx.Graph()
    >>> graph.add_node("parent", EPI=1.0, theta=0.0)
    >>> graph.add_node("neighbor", EPI=1.2, theta=0.0)
    >>> graph.add_edge("parent", "neighbor")
    >>> signals = capture_network_signals(graph, "parent")
    >>> signals["neighbor_count"]
    1
    >>> round(signals["epi_gradient"], 10)
    0.2
    >>> signals["phase_variance"]
    0.0
    """
    from ..metrics.phase_compatibility import compute_phase_coupling_strength

    neighbors = list(G.neighbors(node))
    if not neighbors:
        return None

    node_epi = _read_node_scalar_epi(G.nodes[node], label=f"node {node!r} EPI")
    node_theta = finite_real(
        get_attr(G.nodes[node], ALIAS_THETA, 0.0),
        operator=_OPERATOR,
        label=f"node {node!r} phase",
    )

    # Aggregate neighbor states
    neighbor_epis = []
    neighbor_thetas = []
    coupling_strengths = []

    for n in neighbors:
        n_epi = _read_node_scalar_epi(
            G.nodes[n], label=f"neighbor {n!r} EPI"
        )
        n_theta = finite_real(
            get_attr(G.nodes[n], ALIAS_THETA, 0.0),
            operator=_OPERATOR,
            label=f"neighbor {n!r} phase",
        )

        neighbor_epis.append(n_epi)
        neighbor_thetas.append(n_theta)

        # Coupling strength using canonical phase compatibility formula
        # (unified across UM, RA, THOL operators - see phase_compatibility module)
        coupling_strength = compute_phase_coupling_strength(node_theta, n_theta)
        coupling_strengths.append(coupling_strength)

    # Compute structural gradients
    mean_neighbor_epi = float(np.mean(neighbor_epis))
    epi_gradient = mean_neighbor_epi - node_epi

    # Circular variance is invariant under the 0/2pi chart boundary.  A linear
    # variance would label two nearly aligned phases on opposite sides of that
    # boundary as maximally dissonant.
    mean_cos = float(np.mean(np.cos(neighbor_thetas)))
    mean_sin = float(np.mean(np.sin(neighbor_thetas)))
    phase_variance = max(0.0, min(1.0, 1.0 - (mean_cos**2 + mean_sin**2) ** 0.5))

    # Mean coupling strength
    coupling_strength_mean = float(np.mean(coupling_strengths))

    return {
        "epi_gradient": epi_gradient,
        "phase_variance": phase_variance,
        "neighbor_count": len(neighbors),
        "coupling_strength_mean": coupling_strength_mean,
        "mean_neighbor_epi": mean_neighbor_epi,
    }


def compose_subepi_amplitude(
    parent_epi: float,
    signals: dict[str, Any] | None,
    *,
    scaling_factor: float = THOL_SUB_EPI_SCALING,
    gradient_weight: float = COUPLING_MODERATE,
    complexity_weight: float = COUPLING_GENTLE,
) -> float:
    """Compose an already-admitted THOL sub-EPI amplitude.

    Admission belongs to the SelfOrganization acceleration gate. This pure
    amplitude map therefore accepts only the form and network channels that
    actually determine its result.
    """

    parent_epi = finite_real(
        parent_epi, operator=_OPERATOR, label="parent EPI"
    )
    scaling_factor = finite_real(
        scaling_factor,
        operator=_OPERATOR,
        label="sub-EPI scaling factor",
        lower=0.0,
        upper=1.0,
    )
    gradient_weight = finite_real(
        gradient_weight,
        operator=_OPERATOR,
        label="gradient weight",
        lower=0.0,
        upper=1.0,
    )
    complexity_weight = finite_real(
        complexity_weight,
        operator=_OPERATOR,
        label="complexity weight",
        lower=0.0,
        upper=1.0,
    )

    base_sub_epi = parent_epi * scaling_factor
    if signals is None:
        return float(np.clip(base_sub_epi, 0.0, 1.0))

    if not isinstance(signals, Mapping):
        reject_operator_argument(_OPERATOR, "signals must be a mapping or None")
    if "epi_gradient" not in signals or "phase_variance" not in signals:
        reject_operator_argument(
            _OPERATOR,
            "signals must contain epi_gradient and phase_variance",
        )
    epi_gradient = finite_real(
        signals["epi_gradient"], operator=_OPERATOR, label="EPI gradient"
    )
    phase_variance = finite_real(
        signals["phase_variance"],
        operator=_OPERATOR,
        label="circular phase variance",
        lower=0.0,
        upper=1.0,
    )

    metabolized_epi = (
        base_sub_epi
        + epi_gradient * gradient_weight
        + phase_variance * complexity_weight
    )
    return float(np.clip(metabolized_epi, 0.0, 1.0))


def metabolize_signals_into_subepi(
    parent_epi: float,
    signals: dict[str, Any] | None,
    d2_epi: float,
    scaling_factor: float = THOL_SUB_EPI_SCALING,
    gradient_weight: float = COUPLING_MODERATE,
    complexity_weight: float = COUPLING_GENTLE,
) -> float:
    """Compatibility wrapper for the historical THOL amplitude helper.

    d2_epi is validated because callers historically supplied it, but it is
    diagnostic metadata and never an amplitude channel. Canonical
    SelfOrganization performs the acceleration admission first and then calls
    compose_subepi_amplitude without this inert argument.
    """

    finite_real(d2_epi, operator=_OPERATOR, label="structural acceleration")
    return compose_subepi_amplitude(
        parent_epi,
        signals,
        scaling_factor=scaling_factor,
        gradient_weight=gradient_weight,
        complexity_weight=complexity_weight,
    )

def propagate_subepi_to_network(
    G: TNFRGraph,
    parent_node: NodeId,
    sub_epi_record: dict[str, Any],
) -> list[tuple[NodeId, float]]:
    """Read propagation from a historical pre-channel-partition THOL event.

    This compatibility helper is intentionally read-only. Canonical THOL writes
    the DeltaNFR channel and no longer propagates EPI. New form propagation must
    use Resonance in an explicit grammar-valid word. Calling this helper with an
    unattached record is rejected instead of reviving the former direct write.

    Parameters
    ----------
    G : TNFRGraph
        Graph containing the network
    parent_node : NodeId
        Node where sub-EPI originated (bifurcation source)
    sub_epi_record : dict
        Sub-EPI record from bifurcation, containing:
        - "epi": sub-EPI magnitude
        - "vf": inherited structural frequency
        - "timestamp": creation time

    Returns
    -------
    list of (NodeId, float)
        Historical ``(neighbor_id, injected_epi)`` telemetry for this exact
        sub-EPI record. New canonical THOL executions return an empty list.

    Notes
    -----
    The function never changes EPI, histories, caches, or telemetry. It only
    retrieves already recorded compatibility evidence.
    """
    if parent_node not in G:
        reject_operator_argument(_OPERATOR, f"parent node {parent_node!r} is missing")
    if not isinstance(sub_epi_record, Mapping):
        reject_operator_argument(_OPERATOR, "sub_epi_record must be a mapping")
    if "epi" not in sub_epi_record or "timestamp" not in sub_epi_record:
        reject_operator_argument(
            _OPERATOR, "sub_epi_record must contain epi and timestamp"
        )
    sub_epi = finite_real(
        sub_epi_record["epi"],
        operator=_OPERATOR,
        label="sub-EPI magnitude",
        lower=0.0,
        upper=1.0,
    )
    timestamp = finite_real(
        sub_epi_record["timestamp"],
        operator=_OPERATOR,
        label="sub-EPI timestamp",
        lower=0.0,
    )

    attached = G.nodes[parent_node].get("sub_epis", ())
    if not isinstance(attached, (list, tuple)) or not any(
        record is sub_epi_record for record in attached
    ):
        reject_operator_argument(
            _OPERATOR,
            "sub_epi_record is not attached to the parent by canonical THOL",
        )

    events = G.graph.get("thol_propagations", ())
    if not isinstance(events, (list, tuple)):
        reject_operator_argument(_OPERATOR, "thol_propagations must be a sequence")
    for event in reversed(events):
        if not isinstance(event, Mapping):
            reject_operator_argument(
                _OPERATOR, "thol_propagations entries must be mappings"
            )
        if (
            event.get("source_node") != parent_node
            or event.get("sub_epi") != sub_epi
            or event.get("timestamp") != timestamp
        ):
            continue
        payload = event.get("propagations")
        if not isinstance(payload, list):
            reject_operator_argument(
                _OPERATOR, "committed propagation payload must be a list"
            )
        result: list[tuple[NodeId, float]] = []
        for index, item in enumerate(payload):
            if not isinstance(item, (list, tuple)) or len(item) != 2:
                reject_operator_argument(
                    _OPERATOR,
                    f"committed propagation item {index} must be a node/magnitude pair",
                )
            result.append(
                (
                    item[0],
                    finite_real(
                        item[1],
                        operator=_OPERATOR,
                        label=f"propagation magnitude {index}",
                        lower=0.0,
                    ),
                )
            )
        return result
    return []


def _read_node_scalar_epi(node_data: Mapping[str, Any], *, label: str) -> float:
    """Read an authoritative EPI alias through the shared scalar boundary."""

    raw = get_attr(
        node_data,  # type: ignore[arg-type]
        ALIAS_EPI,
        0.0,
        strict=True,
        conv=lambda value: value,
    )
    try:
        return require_real_scalar_epi(raw, operator=_OPERATOR, label=label)
    except Exception:
        reject_operator_argument(
            _OPERATOR,
            f"{label} must be a valid uniform-real signed scalar embedding",
        )


def _hierarchy_cycle(node: Any) -> None:
    """Reject a hierarchy that cannot represent nested identity."""

    reject_operator_argument(
        _OPERATOR,
        f"sub-EPI hierarchy contains a cycle at node {node!r}",
    )


def _sequence_metadata(value: Any, *, label: str) -> list[Any]:
    """Validate one hierarchy sequence without accepting strings as containers."""

    if not isinstance(value, (list, tuple)):
        reject_operator_argument(_OPERATOR, f"{label} must be a list or tuple")
    return list(value)


def compute_cascade_depth(G: TNFRGraph, node: NodeId) -> int:
    """Return the maximum number of nested THOL parent/child links.

    Both independent sub_nodes and legacy sub_epis references are read. Shared
    descendants are memoized, while a back-edge is rejected because a cyclic
    ownership graph violates operational nesting and previously caused
    unbounded recursion.
    """

    if node not in G:
        reject_operator_argument(_OPERATOR, f"node {node!r} is missing")
    memo: dict[Any, int] = {}
    active: set[Any] = set()

    def visit(current: Any) -> int:
        if current in active:
            _hierarchy_cycle(current)
        if current in memo:
            return memo[current]
        if current not in G:
            return 0

        active.add(current)
        data = G.nodes[current]
        max_depth = 0
        seen_children: set[Any] = set()

        for child in _sequence_metadata(
            data.get("sub_nodes", []), label=f"node {current!r} sub_nodes"
        ):
            if child in seen_children:
                continue
            seen_children.add(child)
            max_depth = max(
                max_depth,
                1 + visit(child) if child in G else 1,
            )

        records = _sequence_metadata(
            data.get("sub_epis", []), label=f"node {current!r} sub_epis"
        )
        for index, record in enumerate(records):
            if not isinstance(record, Mapping):
                reject_operator_argument(
                    _OPERATOR,
                    f"node {current!r} sub_epis[{index}] must be a mapping",
                )
            child = record.get("node_id")
            if child is not None and child not in seen_children:
                seen_children.add(child)
                if child in G:
                    max_depth = max(max_depth, 1 + visit(child))
                    continue
            legacy_depth = nonnegative_integer(
                record.get("cascade_depth", 0),
                operator=_OPERATOR,
                label=f"node {current!r} sub_epis[{index}].cascade_depth",
            )
            max_depth = max(max_depth, 1 + legacy_depth)

        active.remove(current)
        memo[current] = max_depth
        return max_depth

    return visit(node)


def compute_hierarchical_depth(G: TNFRGraph, node: NodeId) -> int:
    """Return the deepest validated THOL hierarchy level.

    Explicit bifurcation_level metadata and actual parent/child distance are
    both respected. Falsy node identifiers such as 0 remain valid. Cycles are
    rejected as invalid U5 ownership rather than ending in RecursionError.
    """

    if node not in G:
        reject_operator_argument(_OPERATOR, f"node {node!r} is missing")
    memo: dict[Any, int] = {}
    active: set[Any] = set()

    def visit(current: Any) -> int:
        if current in active:
            _hierarchy_cycle(current)
        if current in memo:
            return memo[current]

        active.add(current)
        data = G.nodes[current]
        max_depth = 0
        seen_children: set[Any] = set()

        for child in _sequence_metadata(
            data.get("sub_nodes", []), label=f"node {current!r} sub_nodes"
        ):
            if child in seen_children:
                continue
            seen_children.add(child)
            max_depth = max(
                max_depth,
                1 + visit(child) if child in G else 1,
            )

        records = _sequence_metadata(
            data.get("sub_epis", []), label=f"node {current!r} sub_epis"
        )
        for index, record in enumerate(records):
            if not isinstance(record, Mapping):
                reject_operator_argument(
                    _OPERATOR,
                    f"node {current!r} sub_epis[{index}] must be a mapping",
                )
            level = nonnegative_integer(
                record.get("bifurcation_level", 1),
                operator=_OPERATOR,
                label=f"node {current!r} sub_epis[{index}].bifurcation_level",
            )
            max_depth = max(max_depth, level)
            child = record.get("node_id")
            if child is not None and child not in seen_children and child in G:
                seen_children.add(child)
                max_depth = max(max_depth, 1 + visit(child))

        active.remove(current)
        memo[current] = max_depth
        return max_depth

    return visit(node)


def compute_propagation_radius(G: TNFRGraph) -> int:
    """Count total unique nodes affected by THOL cascades.

    Parameters
    ----------
    G : TNFRGraph
        Graph with THOL propagation history

    Returns
    -------
    int
        Number of nodes reached by at least one propagation event

    Notes
    -----
    TNFR Principle: Propagation radius measures the spatial extent
    of cascade effects across the network. High radius indicates
    network-wide self-organization.

    Examples
    --------
    >>> import networkx as nx
    >>> graph = nx.Graph()
    >>> graph.graph["thol_propagations"] = [
    ...     {"source_node": "root", "propagations": [("a", 0.1), ("b", 0.2)]},
    ...     {"source_node": "a", "propagations": [("b", 0.3)]},
    ... ]
    >>> compute_propagation_radius(graph)
    3
    """
    propagations = G.graph.get("thol_propagations", [])
    affected_nodes = set()

    for prop in propagations:
        affected_nodes.add(prop["source_node"])
        for target, _ in prop["propagations"]:
            affected_nodes.add(target)

    return len(affected_nodes)


def compute_subepi_amplitude_alignment(G: TNFRGraph, node: NodeId) -> float:
    """Measure variance-based alignment of stored sub-EPI magnitudes.

    The readout is 1 / (1 + var(amplitudes)) and lies in (0, 1] when
    at least two sub-EPIs exist. A value of 0 means the ensemble is too
    small to assess. This is an amplitude-dispersion diagnostic. It does not
    use DeltaNFR or dEPI, so it must not be interpreted as canonical C(t), a
    fragmentation threshold, or the U5 parent/child inequality.
    """

    if node not in G:
        reject_operator_argument(_OPERATOR, f"node {node!r} is missing")
    sub_epis = G.nodes[node].get("sub_epis", [])
    return _subepi_amplitude_alignment_from_records(sub_epis)


def _subepi_amplitude_alignment_from_records(records: Any) -> float:
    """Return the shared population-variance readout for sub-EPI records.

    This value kernel lets THOL assess a fully validated proposal before its
    records exist on the live graph. The graph-facing public function and the
    proposal path therefore use the same arithmetic and domain checks.
    """

    if not isinstance(records, (list, tuple)):
        reject_operator_argument(_OPERATOR, "sub_epis must be a sequence")
    if len(records) < 2:
        return 0.0

    epi_values: list[float] = []
    for index, record in enumerate(records):
        if not isinstance(record, Mapping) or "epi" not in record:
            reject_operator_argument(
                _OPERATOR, f"sub_epis[{index}] must be a mapping with epi"
            )
        epi_values.append(
            finite_real(
                record["epi"],
                operator=_OPERATOR,
                label=f"sub_epis[{index}].epi",
            )
        )
    variance = float(np.var(epi_values))
    return finite_real(
        1.0 / (1.0 + variance),
        operator=_OPERATOR,
        label="sub-EPI amplitude alignment",
        lower=0.0,
        upper=1.0,
    )


def compute_subepi_collective_coherence(G: TNFRGraph, node: NodeId) -> float:
    """Compatibility alias for compute_subepi_amplitude_alignment.

    The historical name is retained for callers, but the returned quantity is
    only amplitude alignment and never a U5 or canonical-coherence result.
    """

    return compute_subepi_amplitude_alignment(G, node)


def compute_metabolic_activity_index(G: TNFRGraph, node: NodeId) -> float:
    """Measure proportion of sub-EPIs generated through network metabolism.

    Parameters
    ----------
    G : TNFRGraph
        Graph containing the node
    node : NodeId
        Node to analyze

    Returns
    -------
    float
        Ratio [0, 1] of records tagged as having included network context

    Notes
    -----
    This is provenance coverage, not a causal attribution or an effect-size
    estimate. A tagged record used network inputs; the ratio cannot determine
    how strongly those inputs affected its amplitude.

    Examples
    --------
    >>> import networkx as nx
    >>> graph = nx.Graph()
    >>> graph.add_node(
    ...     "parent",
    ...     sub_epis=[{"metabolized": True}, {"metabolized": False}],
    ... )
    >>> compute_metabolic_activity_index(graph, "parent")
    0.5
    """
    sub_epis = G.nodes[node].get("sub_epis", [])
    if not sub_epis:
        return 0.0

    metabolized_count = sum(1 for sub in sub_epis if sub.get("metabolized", False))
    return metabolized_count / len(sub_epis)
