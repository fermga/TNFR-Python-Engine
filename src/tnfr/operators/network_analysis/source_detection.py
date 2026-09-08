"""Source-activity detection for the Reception (EN) operator.

The returned diagnostics are phase compatibility and capacity-weighted EPI
activity. Neither quantity is canonical structural coherence C(t) because the
calculation reads neither DeltaNFR nor dEPI.

Operational Source-Discovery Policy
-----------------------------------
The engine's EN source-discovery helper applies:

1. **Source Detection**: Identify nodes above the selected EPI activity cut
2. **Phase Compatibility**: Validate θᵢ ≈ θⱼ for effective coupling
3. **Emission Activity**: Measure capacity-weighted form (EPI × nu_f)
4. **Network Distance**: Respect structural proximity in network

These functions enable Reception to operate as "active reorganization from
the exterior" rather than passive data absorption.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

from ...constants.operational import ACTIVE_EMISSION_THRESHOLD
from ...utils import angle_diff
from .._diagnostic_scores import (
    finite_real,
    nonnegative_magnitude,
    unit_score,
)
from .._epi_domain import require_real_scalar_epi

try:
    import networkx as nx
except ImportError:
    nx = None  # Fallback to neighbor-only detection if networkx unavailable

if TYPE_CHECKING:
    from ...types import TNFRGraph

__all__ = [
    "detect_emission_sources",
]


def detect_emission_sources(
    G: TNFRGraph,
    receiver_node: Any,
    max_distance: int = 2,
) -> list[tuple[Any, float, float]]:
    """Detect potential emission sources for EN receiver node.

    Identifies nodes in the network that can serve as active sources for
    the receiving node, ranked by phase compatibility. This operational
    prefilter validates candidate sources before EN integrates resonance
    intake.

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
    >>> source_node, compatibility_score, activity = sources[0]
    >>> source_node == emitter
    True
    >>> 0.9 <= compatibility_score <= 1.0
    True
    >>> activity > 0.4
    True

    See Also
    --------
    Reception : Operator that uses source detection
    """
    from ...alias import get_attr
    from ...constants.aliases import ALIAS_EPI, ALIAS_THETA, ALIAS_VF

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
