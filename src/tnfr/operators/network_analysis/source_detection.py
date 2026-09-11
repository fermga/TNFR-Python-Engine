"""Source-activity detection for the Reception (EN) operator.

The returned diagnostics are phase compatibility and capacity-weighted EPI
activity. Neither quantity is canonical structural coherence C(t) because the
calculation reads neither DeltaNFR nor dEPI.

Operational Source-Discovery Policy
-----------------------------------
The engine's EN source-discovery helper applies:

1. **Source Detection**: Identify nodes above the selected EPI activity cut
2. **Phase Compatibility**: Measure and rank the circular phase score
3. **Emission Activity**: Measure capacity-weighted form (EPI × nu_f)
4. **Network Distance**: Respect structural proximity in network

For directed support, an arc ``source -> receiver`` defines incoming EN
causality. Source paths and the direct-neighbour numeric Reception input use
the same orientation. Source discovery is optional telemetry: it neither
selects nor gates the neighbours used by the numeric EN blend, and its bounded
search can report more distant ancestors that are absent from that blend.
"""

from __future__ import annotations

import math
from numbers import Integral
from typing import TYPE_CHECKING, Any

from ...constants.operational import ACTIVE_EMISSION_THRESHOLD
from ...utils import angle_diff
from .._diagnostic_scores import (
    finite_real,
    nonnegative_magnitude,
    unit_score,
)
from .._epi_domain import require_real_scalar_epi

if TYPE_CHECKING:
    from ...types import TNFRGraph

__all__ = [
    "detect_emission_sources",
]


def _same_node_key(left: Any, right: Any) -> bool:
    """Compare node keys without requiring hostile equality to succeed."""

    if left is right:
        return True
    try:
        return bool(left == right)
    except BaseException:
        return False


def _canonical_node_buckets(
    node_order: tuple[Any, ...],
) -> dict[int, tuple[Any, ...]]:
    """Index canonical graph keys by hash without comparing caller objects."""

    pending: dict[int, list[Any]] = {}
    for node in node_order:
        try:
            key_hash = hash(node)
        except BaseException:
            continue
        pending.setdefault(key_hash, []).append(node)
    return {key_hash: tuple(nodes) for key_hash, nodes in pending.items()}


def _canonical_node_key(
    node: Any,
    buckets: dict[int, tuple[Any, ...]],
) -> Any:
    """Resolve an equal fresh adjacency key to its node-view representative."""

    try:
        candidates = buckets.get(hash(node), ())
    except BaseException:
        return node
    return next(
        (candidate for candidate in candidates if _same_node_key(candidate, node)),
        node,
    )


def _bounded_source_identity_distances(
    graph: Any,
    receiver_node: Any,
    max_distance: int,
    canonical_buckets: dict[int, tuple[Any, ...]],
) -> dict[int, int]:
    """Return one reverse-BFS distance per canonical source object identity."""

    from .._reception_kernel import reception_input_neighbors

    seen_identities = {id(receiver_node)}
    frontier = (receiver_node,)
    distances: dict[int, int] = {}
    for distance in range(1, max_distance + 1):
        next_frontier: list[Any] = []
        for target in frontier:
            for observed_source in reception_input_neighbors(graph, target):
                source = _canonical_node_key(
                    observed_source,
                    canonical_buckets,
                )
                source_identity = id(source)
                if source_identity in seen_identities:
                    continue
                seen_identities.add(source_identity)
                distances[source_identity] = distance
                next_frontier.append(source)
        if not next_frontier:
            break
        frontier = tuple(next_frontier)
    return distances


def detect_emission_sources(
    G: TNFRGraph,
    receiver_node: Any,
    max_distance: int = 2,
) -> list[tuple[Any, float, float]]:
    """Detect potential emission sources for EN receiver node.

    Classifies potential sources in the network for telemetry at the receiving
    node, ranked by phase compatibility. The result does not validate, select
    or gate the direct neighbours that EN integrates numerically.

    Parameters
    ----------
    G : TNFRGraph
        Network graph containing TNFR nodes
    receiver_node : Any
        Node whose optional EN source telemetry is being evaluated
    max_distance : int, optional
        Selected finite graph-distance window for source telemetry (default: 2).
        This cutoff does not weight or gate the numeric EN blend.

    Returns
    -------
    list[tuple[Any, float, float]]
        List of (source_node, phase_compatibility_score,
        emission_activity) tuples,
        sorted by phase compatibility (most compatible first).

        - source_node: Node identifier
        - phase_compatibility_score: bounded score in [0, 1]
        - emission_activity: unbounded nonnegative EPI * nu_f product

    TNFR Structural Logic
    ---------------------
    **Phase Compatibility Calculation:**

    Given receiver phase θ_r and source phase θ_s:

    .. code-block:: text

        phase_diff = |wrap(θ_r - θ_s)|
        normalized_diff = phase_diff / π  # phase_diff lies in [0, π]
        phase_compatibility_score = 1.0 - normalized_diff

    ``phase_diff`` is the shortest-arc distance on the phase circle. Arbitrary
    representatives that differ by complete turns therefore remain equivalent.

    **Emission Activity:**

    The unbounded activity readout weights source form by reorganization rate:

    .. code-block:: text

        emission_activity = EPI * nu_f

    This product is not C(t) and has no [0, 1] bound.

    **Active Emission Threshold:**

    Only nodes with EPI ≥ 0.5 are considered active emission sources. The
    selected operational threshold is centralized in
    :mod:`tnfr.constants.operational`; it is not a derived activation law.
    Values below this selected cut are excluded by the EN source-discovery
    policy; no universal activation claim follows from that classification.

    Examples
    --------
    >>> import networkx as nx
    >>> # Create network with emitter and receiver
    >>> G = nx.Graph()
    >>> emitter, receiver = "teacher", "student"
    >>> G.add_node(emitter, EPI=0.5, nu_f=1.0, theta=0.3)
    >>> G.add_node(receiver, EPI=0.25, nu_f=0.9, theta=0.35)
    >>> G.add_edge(emitter, receiver)
    >>> # Detect sources
    >>> sources = detect_emission_sources(G, receiver)
    >>> len(sources)
    1
    >>> source_node, compatibility_score, activity = sources[0]
    >>> source_node == emitter
    True
    >>> 0.9 <= compatibility_score <= 1.0
    True
    >>> activity > 0.4
    True

    See Also
    --------
    Reception : Operator that can record this optional telemetry
    """
    from ...alias import get_attr
    from ...constants.aliases import ALIAS_EPI, ALIAS_THETA, ALIAS_VF

    if (
        isinstance(max_distance, bool)
        or not isinstance(max_distance, Integral)
        or max_distance < 0
    ):
        raise ValueError("Reception max_distance must be nonnegative integer")
    source_max_distance = int(max_distance)

    node_order = tuple(G.nodes())
    canonical_buckets = _canonical_node_buckets(node_order)
    canonical_receiver = next(
        (
            candidate
            for candidate in node_order
            if _same_node_key(candidate, receiver_node)
        ),
        receiver_node,
    )
    source_distances = _bounded_source_identity_distances(
        G,
        canonical_receiver,
        source_max_distance,
        canonical_buckets,
    )

    # Get receiver phase
    receiver_theta = finite_real(
        get_attr(
            G.nodes[receiver_node],
            ALIAS_THETA,
            0.0,
            strict=True,
            conv=lambda value: value,
        ),
        label=f"EN receiver phase {receiver_node!r}",
    )
    sources: list[tuple[Any, float, float]] = []

    # Scan network for potential sources
    for source in node_order:
        if _same_node_key(source, canonical_receiver):
            continue
        if id(source) not in source_distances:
            continue

        # Apply the selected EPI activity threshold.
        source_epi = require_real_scalar_epi(
            get_attr(
                G.nodes[source],
                ALIAS_EPI,
                0.0,
                strict=True,
                conv=lambda value: value,
            ),
            operator="Reception",
            label=f"source EPI {source!r}",
        )
        if source_epi < ACTIVE_EMISSION_THRESHOLD:
            continue

        # Calculate phase compatibility
        source_theta = finite_real(
            get_attr(
                G.nodes[source],
                ALIAS_THETA,
                0.0,
                strict=True,
                conv=lambda value: value,
            ),
            label=f"EN source phase {source!r}",
        )
        # Phase difference normalized to [0, 1] scale
        phase_diff = abs(angle_diff(receiver_theta, source_theta))
        normalized_diff = phase_diff / math.pi
        phase_compatibility_score = unit_score(
            1.0 - normalized_diff,
            label=f"EN phase compatibility {source!r}->{receiver_node!r}",
        )

        source_vf = nonnegative_magnitude(
            get_attr(
                G.nodes[source],
                ALIAS_VF,
                0.0,
                strict=True,
                conv=lambda value: value,
            ),
            label=f"EN source structural frequency {source!r}",
        )
        emission_activity = nonnegative_magnitude(
            source_epi * source_vf,
            label=f"EN source emission activity {source!r}",
        )

        sources.append((source, phase_compatibility_score, emission_activity))

    # Sort by phase compatibility (most compatible first)
    sources.sort(key=lambda x: x[1], reverse=True)

    return sources
