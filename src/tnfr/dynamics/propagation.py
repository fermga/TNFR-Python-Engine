"""Network propagation dynamics for OZ-induced dissonance.

This module implements propagation of dissonance across network neighbors
following TNFR resonance principles. When OZ (Dissonance) is applied to a node,
structural dissonance propagates through the network based on phase compatibility,
frequency matching, and coupling strength.

According to TNFR canonical theory:
    "Nodal interference: Dissonance between nodes that disrupts coherence.
    Can induce reorganization or collapse."

OZ introduces topological asymmetry that propagates beyond the local node,
potentially triggering bifurcation cascades in phase-compatible neighbors.

References
----------
- TNFR.pdf §2.3.3: OZ introduces topological dissonance
- Issue: [OZ] Implement dissonance propagation and neighborhood network effects
"""

from __future__ import annotations

import math
from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass
from numbers import Real
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..types import NodeId, TNFRGraph

from ..alias import get_attr
from ..constants.aliases import ALIAS_DNFR, ALIAS_THETA, ALIAS_VF
from ..constants.canonical import DELTA_PHI_MAX
from ..constants.operational import EMERGENT_FREQ_BALANCE_CANONICAL
from ..errors import TNFRValueError
from ..utils import angle_diff

__all__ = [
    "propagate_dissonance",
    "compute_network_dissonance_field",
    "detect_bifurcation_cascade",
]


_PROPAGATION_MODES = frozenset(
    {"phase_weighted", "uniform", "frequency_weighted"}
)
_PROPAGATION_EVENTS_KEY = "_oz_propagation"
_MISSING = object()


@dataclass(frozen=True, slots=True)
class _DissonanceProposal:
    """One fully validated neighbour update and its rollback snapshot."""

    neighbor: Any
    data: MutableMapping[str, Any]
    new_dnfr: float
    event: dict[str, Any]
    had_primary_dnfr: bool
    primary_dnfr_before: Any
    events_before: list[Any] | None
    events_length_before: int


@dataclass(frozen=True, slots=True)
class _DissonancePropagationPlan:
    """Fully validated OZ neighbor transaction awaiting its commit."""

    proposals: tuple[_DissonanceProposal, ...]
    affected: frozenset[Any]


def _finite_scalar(value: Any, label: str) -> float:
    """Materialize one finite real scalar for the propagation transaction."""

    if isinstance(value, bool) or not isinstance(value, Real):
        raise TNFRValueError(f"{label} must be a finite real scalar")
    try:
        result = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise TNFRValueError(f"{label} must be a finite real scalar") from exc
    if not math.isfinite(result):
        raise TNFRValueError(f"{label} must be finite")
    return result


def _finite_nonnegative(value: Any, label: str) -> float:
    result = _finite_scalar(value, label)
    if result < 0.0:
        raise TNFRValueError(f"{label} must be nonnegative")
    return result


def _read_node_scalar(
    data: MutableMapping[str, Any],
    aliases: tuple[str, ...],
    default: float,
    label: str,
) -> float:
    """Read the first declared alias strictly, without non-finite fallback."""

    try:
        raw = get_attr(
            data,
            aliases,
            default,
            strict=True,
            conv=lambda item: item,
        )
    except (OverflowError, TypeError, ValueError) as exc:
        raise TNFRValueError(f"{label} must be a finite real scalar") from exc
    return _finite_scalar(raw, label)


def _coupling_weight(G: TNFRGraph, source: NodeId, target: NodeId) -> float:
    """Read outgoing conductance, summing parallel edges when present."""

    edge_data = G.get_edge_data(source, target)
    if edge_data is None:
        raise TNFRValueError(
            f"OZ neighbor {target!r} has no outgoing edge from {source!r}"
        )
    if not isinstance(edge_data, Mapping):
        raise TNFRValueError("OZ edge data must be a mapping")

    if bool(G.is_multigraph()):
        weights = []
        for key, attributes in edge_data.items():
            if not isinstance(attributes, Mapping):
                raise TNFRValueError(
                    f"OZ parallel edge {key!r} attributes must be a mapping"
                )
            weights.append(
                _finite_nonnegative(
                    attributes.get("weight", 1.0),
                    f"OZ edge weight {source!r}->{target!r}[{key!r}]",
                )
            )
        try:
            combined = math.fsum(weights)
        except OverflowError as exc:
            raise TNFRValueError(
                f"OZ aggregate edge weight {source!r}->{target!r} must be finite"
            ) from exc
        return _finite_nonnegative(
            combined, f"OZ aggregate edge weight {source!r}->{target!r}"
        )

    return _finite_nonnegative(
        edge_data.get("weight", 1.0),
        f"OZ edge weight {source!r}->{target!r}",
    )


def _prepare_dissonance_propagation(
    G: TNFRGraph,
    source_node: NodeId,
    dissonance_magnitude: float,
    propagation_mode: str = "phase_weighted",
    *,
    dnfr_overrides: Mapping[Any, float] | None = None,
) -> _DissonancePropagationPlan:
    """Validate and materialize a complete OZ propagation transaction."""
    if dnfr_overrides is None:
        dnfr_overrides = {}
    elif not isinstance(dnfr_overrides, Mapping):
        raise TNFRValueError("OZ pressure overrides must be a mapping")

    magnitude = _finite_nonnegative(dissonance_magnitude, "OZ dissonance magnitude")
    if (
        not isinstance(propagation_mode, str)
        or propagation_mode not in _PROPAGATION_MODES
    ):
        supported = ", ".join(sorted(_PROPAGATION_MODES))
        raise TNFRValueError(f"OZ propagation_mode must be one of: {supported}")

    neighbors = list(G.neighbors(source_node))
    if not neighbors:
        return _DissonancePropagationPlan((), frozenset())

    source_data = G.nodes[source_node]
    if not isinstance(source_data, MutableMapping):
        raise TNFRValueError("OZ source node data must be mutable mapping")
    source_theta = _read_node_scalar(
        source_data,
        ALIAS_THETA,
        0.0,
        f"OZ source phase {source_node!r}",
    )
    source_vf = _finite_nonnegative(
        _read_node_scalar(
            source_data,
            ALIAS_VF,
            1.0,
            f"OZ source structural frequency {source_node!r}",
        ),
        f"OZ source structural frequency {source_node!r}",
    )

    # These graph-level values are propagation policy thresholds, not glyph
    # factors. A zero phase threshold makes the compatibility weight undefined.
    phase_threshold = _finite_scalar(
        G.graph.get("OZ_PHASE_THRESHOLD", DELTA_PHI_MAX),
        "OZ phase threshold",
    )
    if phase_threshold <= 0.0:
        raise TNFRValueError("OZ phase threshold must be positive")
    min_propagation = _finite_nonnegative(
        G.graph.get("OZ_MIN_PROPAGATION", 0.05),
        "OZ minimum propagation",
    )

    # Phase one validates the complete outgoing neighborhood and materializes
    # every proposal without mutating nodal state or propagation telemetry.
    proposals: list[_DissonanceProposal] = []
    seen: set[NodeId] = set()
    for neighbor in neighbors:
        if neighbor in seen:
            raise TNFRValueError(
                f"OZ outgoing neighborhood repeats node {neighbor!r}"
            )
        seen.add(neighbor)

        neighbor_data = G.nodes[neighbor]
        if not isinstance(neighbor_data, MutableMapping):
            raise TNFRValueError(
                f"OZ neighbor data {neighbor!r} must be mutable mapping"
            )
        neighbor_theta = _read_node_scalar(
            neighbor_data,
            ALIAS_THETA,
            0.0,
            f"OZ neighbor phase {neighbor!r}",
        )
        neighbor_vf = _finite_nonnegative(
            _read_node_scalar(
                neighbor_data,
                ALIAS_VF,
                1.0,
                f"OZ neighbor structural frequency {neighbor!r}",
            ),
            f"OZ neighbor structural frequency {neighbor!r}",
        )
        has_dnfr_override = neighbor in dnfr_overrides
        if has_dnfr_override:
            neighbor_dnfr = _finite_scalar(
                dnfr_overrides[neighbor],
                f"OZ planned neighbor pressure {neighbor!r}",
            )
        else:
            neighbor_dnfr = _read_node_scalar(
                neighbor_data,
                ALIAS_DNFR,
                0.0,
                f"OZ neighbor pressure {neighbor!r}",
            )
        coupling_weight = _coupling_weight(G, source_node, neighbor)

        delta_theta = _finite_nonnegative(
            abs(angle_diff(source_theta, neighbor_theta)),
            f"OZ phase separation {source_node!r}->{neighbor!r}",
        )
        if delta_theta > phase_threshold:
            continue

        phase_weight = _finite_nonnegative(
            1.0 - (delta_theta / phase_threshold),
            f"OZ phase weight {source_node!r}->{neighbor!r}",
        )

        if propagation_mode == "frequency_weighted":
            frequency_denominator = _finite_scalar(
                max(neighbor_vf, source_vf, 1e-10),
                f"OZ frequency denominator {source_node!r}->{neighbor!r}",
            )
            freq_weight = _finite_nonnegative(
                min(neighbor_vf, source_vf) / frequency_denominator,
                f"OZ frequency weight {source_node!r}->{neighbor!r}",
            )
        else:
            freq_weight = 1.0

        weighted_magnitude = _finite_nonnegative(
            magnitude * coupling_weight,
            f"OZ weighted magnitude {source_node!r}->{neighbor!r}",
        )
        phase_magnitude = _finite_nonnegative(
            weighted_magnitude * phase_weight,
            f"OZ phase-weighted magnitude {source_node!r}->{neighbor!r}",
        )
        propagated_dnfr = _finite_nonnegative(
            phase_magnitude * freq_weight,
            f"OZ propagated pressure {source_node!r}->{neighbor!r}",
        )

        if propagated_dnfr == 0.0 or propagated_dnfr < min_propagation:
            continue

        new_dnfr = _finite_scalar(
            neighbor_dnfr + propagated_dnfr,
            f"OZ proposed pressure {neighbor!r}",
        )
        events = neighbor_data.get(_PROPAGATION_EVENTS_KEY, _MISSING)
        if events is _MISSING:
            events_before = None
            events_length_before = 0
        elif isinstance(events, list):
            events_before = events
            events_length_before = len(events)
        else:
            raise TNFRValueError(
                f"{_PROPAGATION_EVENTS_KEY} for neighbor {neighbor!r} "
                "must be a list"
            )

        had_primary_dnfr = ALIAS_DNFR[0] in neighbor_data
        if has_dnfr_override and had_primary_dnfr:
            primary_dnfr_before = neighbor_dnfr
        else:
            primary_dnfr_before = neighbor_data.get(ALIAS_DNFR[0], _MISSING)

        proposals.append(
            _DissonanceProposal(
                neighbor=neighbor,
                data=neighbor_data,
                new_dnfr=new_dnfr,
                event={
                    "from_node": source_node,
                    "magnitude": propagated_dnfr,
                    "phase_weight": phase_weight,
                    "coupling_weight": coupling_weight,
                },
                had_primary_dnfr=had_primary_dnfr,
                primary_dnfr_before=primary_dnfr_before,
                events_before=events_before,
                events_length_before=events_length_before,
            )
        )

    return _DissonancePropagationPlan(
        tuple(proposals), frozenset(proposal.neighbor for proposal in proposals)
    )


def _restore_dissonance_proposals(
    proposals: tuple[_DissonanceProposal, ...] | list[_DissonanceProposal],
) -> None:
    """Restore proposal targets and telemetry containers in reverse order."""

    for proposal in reversed(proposals):
        if proposal.had_primary_dnfr:
            proposal.data[ALIAS_DNFR[0]] = proposal.primary_dnfr_before
        else:
            proposal.data.pop(ALIAS_DNFR[0], None)

        if proposal.events_before is None:
            proposal.data.pop(_PROPAGATION_EVENTS_KEY, None)
        else:
            del proposal.events_before[proposal.events_length_before :]
            proposal.data[_PROPAGATION_EVENTS_KEY] = proposal.events_before


def _rollback_dissonance_propagation(plan: _DissonancePropagationPlan) -> None:
    """Undo a successfully committed internal propagation plan."""

    _restore_dissonance_proposals(plan.proposals)


def _commit_dissonance_propagation(
    plan: _DissonancePropagationPlan,
) -> set[Any]:
    """Commit one validated plan, rolling back any unexpected write failure."""

    # The public Dissonance path prepares before applying local OZ. Reject a
    # stale plan before the first neighbor write if lifecycle hooks changed one
    # of the fields that the propagation transaction owns.
    for proposal in plan.proposals:
        if proposal.had_primary_dnfr:
            if (
                ALIAS_DNFR[0] not in proposal.data
                or proposal.data[ALIAS_DNFR[0]] != proposal.primary_dnfr_before
            ):
                raise TNFRValueError(
                    f"OZ propagation plan for {proposal.neighbor!r} became stale"
                )
        elif ALIAS_DNFR[0] in proposal.data:
            raise TNFRValueError(
                f"OZ propagation plan for {proposal.neighbor!r} became stale"
            )

        current_events = proposal.data.get(_PROPAGATION_EVENTS_KEY, _MISSING)
        if proposal.events_before is None:
            events_unchanged = current_events is _MISSING
        else:
            events_unchanged = (
                current_events is proposal.events_before
                and len(proposal.events_before) == proposal.events_length_before
            )
        if not events_unchanged:
            raise TNFRValueError(
                f"OZ propagation telemetry for {proposal.neighbor!r} became stale"
            )

    attempted: list[_DissonanceProposal] = []
    try:
        for proposal in plan.proposals:
            attempted.append(proposal)
            proposal.data[ALIAS_DNFR[0]] = proposal.new_dnfr
            if proposal.events_before is None:
                proposal.data[_PROPAGATION_EVENTS_KEY] = [proposal.event]
            else:
                proposal.events_before.append(proposal.event)
    except BaseException:
        _restore_dissonance_proposals(attempted)
        raise

    return set(plan.affected)


def propagate_dissonance(
    G: TNFRGraph,
    source_node: NodeId,
    dissonance_magnitude: float,
    propagation_mode: str = "phase_weighted",
) -> set[NodeId]:
    """Atomically propagate OZ pressure to phase-compatible outgoing neighbors.

    The finite proposal is
    ``magnitude * conductance * phase_weight * frequency_weight``. Directed
    graphs use outgoing arcs and parallel conductances are summed. Validation
    and proposal materialization precede every nodal or telemetry write.
    """

    plan = _prepare_dissonance_propagation(
        G, source_node, dissonance_magnitude, propagation_mode
    )
    return _commit_dissonance_propagation(plan)


def compute_network_dissonance_field(
    G: TNFRGraph,
    source_node: NodeId,
    radius: int = 2,
) -> dict[NodeId, float]:
    """Compute dissonance field propagation up to radius hops.

    Returns dict mapping node -> dissonance_level for all nodes
    within radius hops of source.

    Parameters
    ----------
    G : TNFRGraph
        Network
    source_node : NodeId
        OZ application point
    radius : int
        Maximum propagation distance (default 2)

    Returns
    -------
    dict[NodeId, float]
        Mapping of affected nodes to dissonance level

    Notes
    -----
    Uses exponential decay: dissonance_level = source_dnfr * (0.5 ** distance)

    Only nodes reachable via paths (connected) are included in the field.

    Examples
    --------
    >>> from tnfr.structural import create_nfr
    >>> from tnfr.operators.definitions import Dissonance
    >>> from tnfr.dynamics.propagation import compute_network_dissonance_field
    >>>
    >>> G, node0 = create_nfr("source")
    >>> # Create path topology: 0-1-2-3
    >>> for i in range(1, 4):
    ...     G.add_node(i)
    ...     G.add_edge(i-1, i)
    >>>
    >>> Dissonance()(G, node0)
    >>> field = compute_network_dissonance_field(G, node0, radius=2)
    >>> # Returns: {1: high, 2: medium} (node 3 beyond radius)
    """
    import networkx as nx

    field = {}
    source_dnfr = abs(float(get_attr(G.nodes[source_node], ALIAS_DNFR, 0.0)))

    # Get decay factor from graph config
    decay_factor = float(
        G.graph.get("OZ_DECAY_FACTOR", EMERGENT_FREQ_BALANCE_CANONICAL)
    )

    # BFS to propagate with distance decay
    # Optimized implementation using single_source_shortest_path_length
    # This is O(N+E) instead of O(R*N*(N+E))
    lengths = nx.single_source_shortest_path_length(G, source_node, cutoff=radius)

    for node, distance in lengths.items():
        if distance == 0:
            continue
        decay = decay_factor**distance
        field[node] = source_dnfr * decay

    return field


def detect_bifurcation_cascade(
    G: TNFRGraph,
    source_node: NodeId,
    threshold: float = EMERGENT_FREQ_BALANCE_CANONICAL,
) -> list[NodeId]:
    """Detect if OZ triggers bifurcation cascade in network.

    When source node undergoes bifurcation (∂²EPI/∂t² > τ), check if
    propagated dissonance pushes neighbors over their own thresholds.

    Parameters
    ----------
    G : TNFRGraph
        Network containing nodes
    source_node : NodeId
        Node where OZ was applied
    threshold : float
        Bifurcation threshold τ (default 0.5)

    Returns
    -------
    list[NodeId]
        Nodes that entered bifurcation state due to cascade

    Notes
    -----
    A node is considered in bifurcation cascade if:
    - It received propagated dissonance from source
    - Its ∂²EPI/∂t² now exceeds threshold τ

    The function marks cascade nodes with `_bifurcation_cascade` metadata
    for telemetry and further analysis.

    Examples
    --------
    >>> from tnfr.structural import create_nfr
    >>> from tnfr.operators.definitions import Emission, Dissonance
    >>> from tnfr.dynamics.propagation import detect_bifurcation_cascade
    >>>
    >>> G, node0 = create_nfr("source", epi=0.5, vf=1.2)
    >>> # Add neighbors with EPI history
    >>> for i in range(3):
    ...     G.add_node(f"n{i}")
    ...     G.add_edge(node0, f"n{i}")
    ...     G.nodes[f"n{i}"]["_epi_history"] = [0.3, 0.45, 0.55]
    >>>
    >>> Dissonance()(G, node0, propagate_to_network=True)
    >>> cascade = detect_bifurcation_cascade(G, node0)
    >>> print(f"Cascade size: {len(cascade)}")

    See Also
    --------
    tnfr.operators.nodal_equation.compute_d2epi_dt2 : Compute structural acceleration
    tnfr.dynamics.bifurcation.get_bifurcation_paths : Identify viable paths
    """
    from ..operators.nodal_equation import compute_d2epi_dt2

    cascade_nodes = []

    # Get neighbors affected by propagation
    neighbors = list(G.neighbors(source_node))

    for neighbor in neighbors:
        # Check if neighbor has propagation record (was affected)
        if "_oz_propagation" not in G.nodes[neighbor]:
            continue

        # Check if neighbor now in bifurcation state
        d2epi_neighbor = compute_d2epi_dt2(G, neighbor)

        if abs(d2epi_neighbor) > threshold:
            cascade_nodes.append(neighbor)

            # Mark for telemetry
            G.nodes[neighbor]["_bifurcation_cascade"] = {
                "triggered_by": source_node,
                "d2epi": d2epi_neighbor,
                "threshold": threshold,
            }

            # set bifurcation_ready flag for path detection
            G.nodes[neighbor]["_bifurcation_ready"] = True

    return cascade_nodes
