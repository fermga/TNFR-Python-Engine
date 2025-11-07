"""Source detection for Reception (EN) operator.

This module implements emission source detection for the Reception operator,
enabling active reorganization through identification of compatible coherence
sources in the network.

TNFR Context
------------
According to TNFR.pdf §2.2.1, EN (Reception) requires:

1. **Source Detection**: Identify nodes emitting coherence (active EPI)
2. **Phase Compatibility**: Validate θᵢ ≈ θⱼ for effective coupling
3. **Coherence Strength**: Measure available coherence (EPI × νf)
4. **Network Distance**: Respect structural proximity in network

These functions enable Reception to operate as "active reorganization from
the exterior" rather than passive data absorption.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ...types import TNFRGraph

import math

try:
    import networkx as nx
except ImportError:
    nx = None  # Fallback to neighbor-only detection if networkx unavailable

__all__ = [
    "detect_emission_sources",
]

# Active emission threshold: minimum EPI for node to be considered emission source
# Below this threshold, structural form is too weak to contribute coherence
ACTIVE_EMISSION_THRESHOLD = 0.2


def detect_emission_sources(
    G: TNFRGraph,
    receiver_node: Any,
    max_distance: int = 2,
) -> list[tuple[Any, float, float]]:
    """Detect potential emission sources for EN receiver node.

    Identifies nodes in the network that can serve as coherence sources for
    the receiving node, ranked by phase compatibility. This implements the
    "active reception" principle from TNFR.pdf §2.2.1 where EN must detect
    and validate compatible emission sources before integrating external
    coherence.

    Parameters
    ----------
    G : TNFRGraph
        Network graph containing TNFR nodes
    receiver_node : Any
        Node applying EN (Reception) that needs to detect sources
    max_distance : int, optional
        Maximum network distance to search for sources (default: 2)
        Respects structural locality principle - distant nodes have
        negligible coupling

    Returns
    -------
    list[tuple[Any, float, float]]
        List of (source_node, phase_compatibility, coherence_strength) tuples,
        sorted by phase compatibility (most compatible first).

        - source_node: Node identifier
        - phase_compatibility: 0.0 (incompatible) to 1.0 (perfect sync)
        - coherence_strength: Available coherence (EPI × νf)

    TNFR Structural Logic
    ---------------------
    **Phase Compatibility Calculation:**

    Given receiver phase θ_r and source phase θ_s:

    .. code-block:: text

        phase_diff = |θ_r - θ_s|
        normalized_diff = min(phase_diff / π, 1.0)  # Normalize to [0, 1]
        compatibility = 1.0 - normalized_diff

    Phase values are normalized to [0, π] range before comparison to respect
    phase periodicity in TNFR.

    **Coherence Strength:**

    Coherence strength represents the emission capacity of the source:

    .. code-block:: text

        coherence_strength = EPI × νf

    Higher values indicate stronger emission that can be more effectively
    integrated by the receiver.

    **Active Emission Threshold:**

    Only nodes with EPI ≥ 0.2 are considered active emission sources.
    Below this threshold, the node's structural form is too weak to
    effectively contribute coherence.

    Examples
    --------
    >>> from tnfr.structural import create_nfr
    >>> import networkx as nx
    >>> # Create network with emitter and receiver
    >>> G = nx.Graph()
    >>> G, emitter = create_nfr("teacher", epi=0.5, vf=1.0, theta=0.3, G=G)
    >>> _, receiver = create_nfr("student", epi=0.25, vf=0.9, theta=0.35, G=G)
    >>> G.add_edge(emitter, receiver)
    >>> # Detect sources
    >>> sources = detect_emission_sources(G, receiver)
    >>> len(sources)
    1
    >>> source_node, compatibility, strength = sources[0]
    >>> source_node == emitter
    True
    >>> 0.9 <= compatibility <= 1.0  # High phase compatibility
    True
    >>> strength > 0.4  # Strong coherence (0.5 * 1.0)
    True

    See Also
    --------
    Reception : Operator that uses source detection
    """
    from ...alias import get_attr
    from ...constants.aliases import ALIAS_THETA, ALIAS_EPI, ALIAS_VF

    # Get receiver phase
    receiver_theta = float(get_attr(G.nodes[receiver_node], ALIAS_THETA, 0.0))
    # Normalize to [0, π] range for phase comparison
    receiver_theta = abs(receiver_theta) % math.pi

    sources = []

    # Scan network for potential sources
    for source in G.nodes():
        if source == receiver_node:
            continue

        # Check network distance
        if nx is not None:
            try:
                distance = nx.shortest_path_length(G, source, receiver_node)
                if distance > max_distance:
                    continue
            except nx.NetworkXNoPath:
                continue
        else:
            # Fallback: only check immediate neighbors
            if source not in G.neighbors(receiver_node):
                continue

        # Check if source is active (has coherent EPI)
        source_epi = float(get_attr(G.nodes[source], ALIAS_EPI, 0.0))
        if source_epi < ACTIVE_EMISSION_THRESHOLD:
            continue

        # Calculate phase compatibility
        source_theta = float(get_attr(G.nodes[source], ALIAS_THETA, 0.0))
        # Normalize to [0, π] range
        source_theta = abs(source_theta) % math.pi

        # Phase difference normalized to [0, 1] scale
        phase_diff = abs(receiver_theta - source_theta)
        normalized_diff = min(phase_diff / math.pi, 1.0)
        phase_compatibility = 1.0 - normalized_diff

        # Coherence strength (EPI × νf)
        source_vf = float(get_attr(G.nodes[source], ALIAS_VF, 0.0))
        coherence_strength = source_epi * source_vf

        sources.append((source, phase_compatibility, coherence_strength))

    # Sort by phase compatibility (most compatible first)
    sources.sort(key=lambda x: x[1], reverse=True)

    return sources
