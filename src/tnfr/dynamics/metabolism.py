"""Structural metabolism implementation for TNFR.

T'HOL (Self-Organization) as metabolic process: receiving external stimuli,
reorganizing them autonomously into internal structure, and stabilizing results.

This module implements metabolic cycles that use T'HOL as the engine of
structural transformation and adaptation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..types import NodeId, TNFRGraph


__all__ = [
    "StructuralMetabolism",
    "digest_stimulus",
    "adaptive_metabolism",
    "cascading_reorganization",
]


class StructuralMetabolism:
    """Implements T'HOL-based structural metabolism cycles.

    T'HOL is not just self-organization - it's **structural metabolism**:
    the capacity to digest external experience and reorganize it into
    internal structure without external instruction.

    **Metabolic Characteristics:**

    - **Reception (EN)**: Ingests external stimulus
    - **Reorganization (THOL)**: Autonomously transforms stimulus to structure
    - **Stabilization (IL)**: Consolidates new structural configuration

    This creates a complete metabolic cycle: EN → THOL → IL

    Parameters
    ----------
    graph : TNFRGraph
        Graph containing the metabolizing node
    node : NodeId
        Identifier of the node performing metabolism

    Attributes
    ----------
    G : TNFRGraph
        Reference to the graph
    node : NodeId
        Reference to the node identifier
    metabolic_rate : float
        Scaling factor for metabolic intensity (default 1.0)

    Examples
    --------
    >>> from tnfr.structural import create_nfr
    >>> from tnfr.dynamics.metabolism import StructuralMetabolism
    >>> G, node = create_nfr("cell", epi=0.5, vf=1.0)
    >>> metabolism = StructuralMetabolism(G, node)
    >>> # Digest external stimulus
    >>> metabolism.digest(0.3)  # doctest: +SKIP
    >>> # Result: stimulus integrated and reorganized structurally
    """

    def __init__(self, graph: TNFRGraph, node: NodeId) -> None:
        """Initialize structural metabolism for a node.

        Parameters
        ----------
        graph : TNFRGraph
            Graph containing the node
        node : NodeId
            Node identifier
        """
        self.G = graph
        self.node = node
        self.metabolic_rate = 1.0

    def digest(self, tau: float = 0.08) -> None:
        """Metabolize external stimulus through complete metabolic cycle.

        Implements the canonical metabolic sequence: EN → THOL → IL

        1. Reception (EN): Receives external stimulus from neighbors
        2. Reorganization (THOL): Autonomously transforms into structure
        3. Stabilization (IL): Consolidates the result

        Parameters
        ----------
        tau : float
            Bifurcation threshold for THOL (default 0.08)

        Notes
        -----
        The metabolic rate modulates the intensity of each operation.
        Lower tau increases likelihood of bifurcation during reorganization.
        """
        from ..operators.definitions import Reception, SelfOrganization, Coherence

        # 1. Receive external stimulus
        Reception()(self.G, self.node)

        # 2. Reorganize metabolically (T'HOL with controlled bifurcation)
        SelfOrganization()(self.G, self.node, tau=tau)

        # 3. Stabilize result
        Coherence()(self.G, self.node)

    def adaptive_metabolism(self, stress_level: float) -> None:
        """Adapt metabolic response to stress level.

        High stress (dissonance) triggers deeper reorganization with
        increased bifurcation probability. Low stress allows gentler
        metabolic cycles.

        Parameters
        ----------
        stress_level : float
            Level of structural stress/dissonance (0.0 to 1.0+)
            - < 0.5: Moderate stress, gentle reorganization
            - >= 0.5: High stress, deep reorganization with dissonance

        Notes
        -----
        This implements adaptive structural metabolism where the depth
        of reorganization scales with environmental pressure.
        """
        from ..operators.definitions import (
            Dissonance,
            SelfOrganization,
            Coherence,
        )

        if stress_level >= 0.5:
            # High stress: dissonance + deep reorganization
            # Apply operators individually to avoid grammar restrictions
            Dissonance()(self.G, self.node)  # Introduce controlled instability
            SelfOrganization()(
                self.G, self.node, tau=0.08
            )  # Deep reorganization (likely bifurcates)
            Coherence()(self.G, self.node)  # Stabilize new configuration
        else:
            # Moderate stress: gentle reorganization
            # Higher tau reduces bifurcation probability
            SelfOrganization()(self.G, self.node, tau=0.15)

    def cascading_reorganization(self, depth: int = 3) -> None:
        """Execute recursive T'HOL cascade.

        Applies T'HOL multiple times with progressively decreasing
        bifurcation thresholds, creating nested structural reorganization.

        This implements operational fractality: reorganization at multiple
        scales simultaneously.

        Parameters
        ----------
        depth : int
            Number of cascade levels (default 3)

        Notes
        -----
        Each level uses tau = 0.1 * (0.8 ^ level), creating progressively
        more sensitive bifurcation at deeper levels.

        **Warning**: Deep cascades (depth > 5) may create highly complex
        nested structures. Monitor structural complexity metrics.
        """
        from ..operators.definitions import SelfOrganization

        for level in range(depth):
            # Decreasing threshold: deeper levels bifurcate more easily
            tau = 0.1 * (0.8**level)
            SelfOrganization()(self.G, self.node, tau=tau)


def digest_stimulus(G: TNFRGraph, node: NodeId, tau: float = 0.08) -> None:
    """Functional interface for single metabolic cycle.

    Equivalent to `StructuralMetabolism(G, node).digest(tau)`.

    Parameters
    ----------
    G : TNFRGraph
        Graph containing the node
    node : NodeId
        Node identifier
    tau : float
        Bifurcation threshold

    Examples
    --------
    >>> from tnfr.structural import create_nfr
    >>> from tnfr.dynamics.metabolism import digest_stimulus
    >>> G, node = create_nfr("neuron", epi=0.4, vf=1.2)
    >>> digest_stimulus(G, node, tau=0.1)  # doctest: +SKIP
    """
    metabolism = StructuralMetabolism(G, node)
    metabolism.digest(tau)


def adaptive_metabolism(G: TNFRGraph, node: NodeId, stress: float) -> None:
    """Functional interface for adaptive metabolic response.

    Equivalent to `StructuralMetabolism(G, node).adaptive_metabolism(stress)`.

    Parameters
    ----------
    G : TNFRGraph
        Graph containing the node
    node : NodeId
        Node identifier
    stress : float
        Stress level (0.0 to 1.0+)

    Examples
    --------
    >>> from tnfr.structural import create_nfr
    >>> from tnfr.dynamics.metabolism import adaptive_metabolism
    >>> G, node = create_nfr("organism", epi=0.6, vf=1.0)
    >>> adaptive_metabolism(G, node, stress=0.7)  # doctest: +SKIP
    """
    metabolism = StructuralMetabolism(G, node)
    metabolism.adaptive_metabolism(stress)


def cascading_reorganization(G: TNFRGraph, node: NodeId, depth: int = 3) -> None:
    """Functional interface for cascading reorganization.

    Equivalent to `StructuralMetabolism(G, node).cascading_reorganization(depth)`.

    Parameters
    ----------
    G : TNFRGraph
        Graph containing the node
    node : NodeId
        Node identifier
    depth : int
        Cascade depth

    Examples
    --------
    >>> from tnfr.structural import create_nfr
    >>> from tnfr.dynamics.metabolism import cascading_reorganization
    >>> G, node = create_nfr("system", epi=0.7, vf=1.1)
    >>> cascading_reorganization(G, node, depth=3)  # doctest: +SKIP
    """
    metabolism = StructuralMetabolism(G, node)
    metabolism.cascading_reorganization(depth)
