"""TNFR Operator: SelfOrganization

Self-Organization structural operator (THOL) - Autonomous emergent reorganization.

**Physics**: See AGENTS.md § SelfOrganization
**Grammar**: UNIFIED_GRAMMAR_RULES.md
"""

from __future__ import annotations

import warnings
from typing import Any, ClassVar

from ..alias import get_attr
from ..config.operator_names import SELF_ORGANIZATION
from ..constants.aliases import ALIAS_DNFR, ALIAS_EPI
from ..types import Glyph, TNFRGraph
from .definitions_base import Operator

# Private module constants for THOL bifurcation
_THOL_SUB_EPI_SCALING = 0.25  # Sub-EPI is 25% of parent (first-order bifurcation)
_THOL_EMERGENCE_CONTRIBUTION = 0.1  # Parent EPI increases by 10% of sub-EPI


class SelfOrganization(Operator):
    """Self-Organization structural operator (THOL) - Autonomous emergent reorganization.

    Activates glyph ``THOL`` to spawn nested EPIs and trigger self-organizing cascades
    within the local structure, enabling autonomous coherent reorganization.

    TNFR Context: Self-organization (THOL) embodies emergence - when ∂²EPI/∂t² > τ, the
    system bifurcates and generates new sub-EPIs that organize coherently without external
    direction. THOL is the engine of complexity and novelty in TNFR. This is not just
    autoorganization but **structural metabolism**: T'HOL reorganizes experience into
    structure without external instruction.

    **Canonical Characteristics:**

    - **Bifurcation nodal**: When ∂²EPI/∂t² > τ, spawns new sub-EPIs
    - **Autonomous reorganization**: No external control, self-directed
    - **Vibrational metabolism**: Digests external experience into internal structure
    - **Complexity emergence**: Engine of novelty and evolution in TNFR

    **Vibrational Metabolism (Canonical THOL):**

    THOL implements the metabolic principle: capturing network vibrational signals
    (EPI gradients, phase variance) and transforming them into internal structure
    (sub-EPIs). This ensures that bifurcation reflects not only internal acceleration
    but also the network's coherence field.

    Metabolic formula: ``sub-EPI = base + gradient*w₁ + variance*w₂``

    - If node has neighbors: Captures and metabolizes network signals
    - If node is isolated: Falls back to pure internal bifurcation
    - Configurable via ``THOL_METABOLIC_ENABLED`` and weight parameters

    Use Cases: Emergence processes, bifurcation events, creative reorganization, complex
    system evolution, spontaneous order generation.

    Typical Sequences: OZ → THOL (dissonance catalyzes emergence), THOL → RA (emergent
    forms propagate), THOL → IL (organize then stabilize), EN → THOL (reception triggers
    reorganization).

    Critical: THOL requires sufficient ΔNFR and network connectivity for bifurcation.

    Examples
    --------
    >>> from tnfr.constants import EPI_PRIMARY, VF_PRIMARY
    >>> from tnfr.dynamics import set_delta_nfr_hook
    >>> from tnfr.structural import create_nfr, run_sequence
    >>> from tnfr.operators.definitions import SelfOrganization
    >>> G, node = create_nfr("kappa", epi=0.66, vf=1.10)
    >>> cascades = iter([(0.04, 0.05)])
    >>> def spawn(graph):
    ...     d_epi, d_vf = next(cascades)
    ...     graph.nodes[node][EPI_PRIMARY] += d_epi
    ...     graph.nodes[node][VF_PRIMARY] += d_vf
    ...     graph.graph.setdefault("sub_epi", []).append(round(graph.nodes[node][EPI_PRIMARY], 2))
    >>> set_delta_nfr_hook(G, spawn)
    >>> run_sequence(G, node, [SelfOrganization()])
    >>> G.graph["sub_epi"]
    [0.7]

    **Biomedical**: Embryogenesis, immune response, neural plasticity, wound healing
    **Cognitive**: Insight generation, creative breakthroughs, paradigm shifts
    **Social**: Innovation emergence, cultural evolution, spontaneous movements
    """

    __slots__ = ()
    name: ClassVar[str] = SELF_ORGANIZATION
    glyph: ClassVar[Glyph] = Glyph.THOL

    def __call__(self, G: TNFRGraph, node: Any, **kw: Any) -> None:
        """Apply T'HOL with bifurcation logic.

        If ∂²EPI/∂t² > τ, generates sub-EPIs through bifurcation.

        Parameters
        ----------
        G : TNFRGraph
            Graph storing TNFR nodes
        node : Any
            Target node identifier
        **kw : Any
            Additional parameters including:
            - tau: Bifurcation threshold (default from graph config or 0.1)
            - validate_preconditions: Enable precondition checks (default True)
            - collect_metrics: Enable metrics collection (default False)
        """
        # Compute structural acceleration before base operator
        d2_epi = self._compute_epi_acceleration(G, node)

        # Get bifurcation threshold (tau) from kwargs or graph config
        tau = kw.get("tau")
        if tau is None:
            tau = float(G.graph.get("THOL_BIFURCATION_THRESHOLD", 0.1))

        # Apply base operator (includes glyph application and metrics)
        super().__call__(G, node, **kw)

        # Bifurcate if acceleration exceeds threshold
        if d2_epi > tau:
            # Validate depth before bifurcation
            self._validate_bifurcation_depth(G, node)
            self._spawn_sub_epi(G, node, d2_epi=d2_epi, tau=tau)

        # CANONICAL VALIDATION: Verify collective coherence of sub-EPIs
        # When THOL creates multiple sub-EPIs, they must form a coherent ensemble
        # that preserves the structural identity of the parent node (TNFR Manual §2.2.10)
        # Always validate if node has sub-EPIs (whether created now or previously)
        if G.nodes[node].get("sub_epis"):
            self._validate_collective_coherence(G, node)

    def _compute_epi_acceleration(self, G: TNFRGraph, node: Any) -> float:
        """Calculate ∂²EPI/∂t² from node's EPI history.

        Uses finite difference approximation:
        d²EPI/dt² ≈ (EPI_t - 2*EPI_{t-1} + EPI_{t-2}) / (Δt)²
        For unit time steps: d²EPI/dt² ≈ EPI_t - 2*EPI_{t-1} + EPI_{t-2}

        Parameters
        ----------
        G : TNFRGraph
            Graph containing the node
        node : Any
            Node identifier

        Returns
        -------
        float
            Magnitude of EPI acceleration (always non-negative)
        """

        # Get EPI history (maintained by node for temporal analysis)
        history = G.nodes[node].get("epi_history", [])

        # Need at least 3 points for second derivative
        if len(history) < 3:
            return 0.0

        # Finite difference: d²EPI/dt² ≈ (EPI_t - 2*EPI_{t-1} + EPI_{t-2})
        epi_t = float(history[-1])
        epi_t1 = float(history[-2])
        epi_t2 = float(history[-3])

        d2_epi = epi_t - 2.0 * epi_t1 + epi_t2

        return abs(d2_epi)

    def _spawn_sub_epi(self, G: TNFRGraph, node: Any, d2_epi: float, tau: float) -> None:
        """Generate sub-EPI through bifurcation with vibrational metabolism.

        When acceleration exceeds threshold, creates nested sub-structure that:
        1. Captures network vibrational signals (metabolic perception)
        2. Metabolizes signals into sub-EPI magnitude (digestion)
        3. Inherits properties from parent while integrating field context

        This implements canonical THOL: "reorganizes external experience into
        internal structure without external instruction".

        ARCHITECTURAL: Sub-EPIs are created as independent NFR nodes to enable
        operational fractality - recursive operator application, hierarchical metrics,
        and multi-level bifurcation.

        Parameters
        ----------
        G : TNFRGraph
            Graph containing the node
        node : Any
            Node identifier
        d2_epi : float
            Current EPI acceleration
        tau : float
            Bifurcation threshold that was exceeded
        """
        from ..alias import get_attr, set_attr
        from ..constants.aliases import ALIAS_EPI, ALIAS_VF, ALIAS_THETA
        from .metabolism import capture_network_signals, metabolize_signals_into_subepi

        # Get current node state
        parent_epi = float(get_attr(G.nodes[node], ALIAS_EPI, 0.0))
        parent_vf = float(get_attr(G.nodes[node], ALIAS_VF, 1.0))
        parent_theta = float(get_attr(G.nodes[node], ALIAS_THETA, 0.0))

        # Check if vibrational metabolism is enabled
        metabolic_enabled = G.graph.get("THOL_METABOLIC_ENABLED", True)

        # CANONICAL METABOLISM: Capture network context
        network_signals = None
        if metabolic_enabled:
            network_signals = capture_network_signals(G, node)

        # Get metabolic weights from graph config
        gradient_weight = float(G.graph.get("THOL_METABOLIC_GRADIENT_WEIGHT", 0.15))
        complexity_weight = float(G.graph.get("THOL_METABOLIC_COMPLEXITY_WEIGHT", 0.10))

        # CANONICAL METABOLISM: Digest signals into sub-EPI
        sub_epi_value = metabolize_signals_into_subepi(
            parent_epi=parent_epi,
            signals=network_signals if metabolic_enabled else None,
            d2_epi=d2_epi,
            scaling_factor=_THOL_SUB_EPI_SCALING,
            gradient_weight=gradient_weight,
            complexity_weight=complexity_weight,
        )

        # Get current timestamp from glyph history length
        timestamp = len(G.nodes[node].get("glyph_history", []))

        # Determine parent bifurcation level for hierarchical telemetry
        parent_level = G.nodes[node].get("_bifurcation_level", 0)
        child_level = parent_level + 1

        # Construct hierarchy path for full traceability
        parent_path = G.nodes[node].get("_hierarchy_path", [])
        child_path = parent_path + [node]

        # ARCHITECTURAL: Create sub-EPI as independent NFR node
        # This enables operational fractality - recursive operators, hierarchical metrics
        sub_node_id = self._create_sub_node(
            G,
            parent_node=node,
            sub_epi=sub_epi_value,
            parent_vf=parent_vf,
            parent_theta=parent_theta,
            child_level=child_level,
            child_path=child_path,
        )

        # Store sub-EPI metadata for telemetry and backward compatibility
        sub_epi_record = {
            "epi": sub_epi_value,
            "vf": parent_vf,
            "timestamp": timestamp,
            "d2_epi": d2_epi,
            "tau": tau,
            "node_id": sub_node_id,  # Reference to independent node
            "metabolized": network_signals is not None and metabolic_enabled,
            "network_signals": network_signals,
            "bifurcation_level": child_level,  # Hierarchical depth tracking
            "hierarchy_path": child_path,  # Full parent chain for traceability
        }

        # Keep metadata list for telemetry/metrics backward compatibility
        sub_epis = G.nodes[node].get("sub_epis", [])
        sub_epis.append(sub_epi_record)
        G.nodes[node]["sub_epis"] = sub_epis

        # Increment parent EPI using canonical emergence contribution
        # This reflects that bifurcation increases total structural complexity
        new_epi = parent_epi + sub_epi_value * _THOL_EMERGENCE_CONTRIBUTION
        set_attr(G.nodes[node], ALIAS_EPI, new_epi)

        # CANONICAL PROPAGATION: Enable network cascade dynamics
        if G.graph.get("THOL_PROPAGATION_ENABLED", True):
            from .metabolism import propagate_subepi_to_network

            propagations = propagate_subepi_to_network(G, node, sub_epi_record)

            # Record propagation telemetry for cascade analysis
            if propagations:
                G.graph.setdefault("thol_propagations", []).append(
                    {
                        "source_node": node,
                        "sub_epi": sub_epi_value,
                        "propagations": propagations,
                        "timestamp": timestamp,
                    }
                )

    def _create_sub_node(
        self,
        G: TNFRGraph,
        parent_node: Any,
        sub_epi: float,
        parent_vf: float,
        parent_theta: float,
        child_level: int,
        child_path: list,
    ) -> str:
        """Create sub-EPI as independent NFR node for operational fractality.

        Sub-nodes are full TNFR nodes that can have operators applied, bifurcate
        recursively, and contribute to hierarchical metrics.

        Parameters
        ----------
        G : TNFRGraph
            Graph containing the parent node
        parent_node : Any
            Parent node identifier
        sub_epi : float
            EPI value for the sub-node
        parent_vf : float
            Parent's structural frequency (inherited with damping)
        parent_theta : float
            Parent's phase (inherited)
        child_level : int
            Bifurcation level for hierarchical tracking
        child_path : list
            Full hierarchy path (ancestor chain)

        Returns
        -------
        str
            Identifier of the newly created sub-node
        """
        from ..constants import EPI_PRIMARY, VF_PRIMARY, THETA_PRIMARY, DNFR_PRIMARY

        # Generate unique sub-node ID
        sub_nodes_list = G.nodes[parent_node].get("sub_nodes", [])
        sub_index = len(sub_nodes_list)
        sub_node_id = f"{parent_node}_sub_{sub_index}"

        # Get parent hierarchy level
        parent_hierarchy_level = G.nodes[parent_node].get("hierarchy_level", 0)

        # Inherit parent's vf with slight damping (canonical: 95%)
        sub_vf = parent_vf * 0.95

        # Create the sub-node with full TNFR state
        G.add_node(
            sub_node_id,
            **{
                EPI_PRIMARY: float(sub_epi),
                VF_PRIMARY: float(sub_vf),
                THETA_PRIMARY: float(parent_theta),
                DNFR_PRIMARY: 0.0,
                "parent_node": parent_node,
                "hierarchy_level": parent_hierarchy_level + 1,
                "_bifurcation_level": child_level,  # Hierarchical depth tracking
                "_hierarchy_path": child_path,  # Full ancestor chain
                "epi_history": [float(sub_epi)],  # Initialize history for future bifurcation
                "glyph_history": [],
            },
        )

        # Ensure ΔNFR hook is set for the sub-node
        # (inherits from graph-level hook, but ensure it's activated)
        if hasattr(G, "graph") and "_delta_nfr_hook" in G.graph:
            # Hook already set at graph level, will apply to sub-node automatically
            pass

        # Track sub-node in parent
        sub_nodes_list.append(sub_node_id)
        G.nodes[parent_node]["sub_nodes"] = sub_nodes_list

        # Track hierarchy in graph metadata
        hierarchy = G.graph.setdefault("hierarchy", {})
        hierarchy.setdefault(parent_node, []).append(sub_node_id)

        return sub_node_id

    def _validate_bifurcation_depth(self, G: TNFRGraph, node: Any) -> None:
        """Validate bifurcation depth before creating new sub-EPI.

        Checks if the current bifurcation level is at or exceeds the configured
        maximum depth. Issues a warning if depth limit is reached but still
        allows the bifurcation (for flexibility in research contexts).

        Parameters
        ----------
        G : TNFRGraph
            Graph containing the node
        node : Any
            Node about to undergo bifurcation

        Notes
        -----
        TNFR Principle: Deep nesting reflects operational fractality (Invariant #7),
        but excessive depth may impact performance and interpretability. This
        validation provides observability without hard constraints.

        The warning allows tracking when hierarchies become complex, enabling
        researchers to study bifurcation patterns while maintaining system
        performance awareness.
        """
        import logging

        # Get current bifurcation level
        current_level = G.nodes[node].get("_bifurcation_level", 0)

        # Get max depth from graph config (default: 5 levels)
        max_depth = int(G.graph.get("THOL_MAX_BIFURCATION_DEPTH", 5))

        # Warn if at or exceeding maximum
        if current_level >= max_depth:
            logger = logging.getLogger(__name__)
            logger.warning(
                f"Node {node}: Bifurcation depth ({current_level}) at/exceeds "
                f"maximum ({max_depth}). Deep nesting may impact performance. "
                f"Consider adjusting THOL_MAX_BIFURCATION_DEPTH if intended."
            )

            # Record warning in node for telemetry
            G.nodes[node]["_thol_max_depth_warning"] = True

            # Record event for analysis
            events = G.graph.setdefault("thol_depth_warnings", [])
            events.append(
                {
                    "node": node,
                    "depth": current_level,
                    "max_depth": max_depth,
                }
            )

    def _validate_collective_coherence(self, G: TNFRGraph, node: Any) -> None:
        """Validate collective coherence of sub-EPI ensemble after bifurcation.

        When THOL creates multiple sub-EPIs, they must form a coherent ensemble
        that preserves the structural identity of the parent node. This validation
        ensures the emergent sub-structures maintain structural alignment.

        Parameters
        ----------
        G : TNFRGraph
            Graph containing the node
        node : Any
            Node that underwent bifurcation

        Notes
        -----
        TNFR Canonical Principle (TNFR Manual §2.2.10):
        "THOL reorganiza la forma desde dentro, en respuesta a la coherencia
        vibracional del campo. La autoorganización es resonancia estructurada
        desde el interior del nodo."

        Implication: Sub-EPIs are not random fragments but coherent structures
        that emerge from internal resonance.

        This method:
        1. Computes collective coherence of sub-EPI ensemble
        2. Stores coherence value for telemetry
        3. Logs warning if coherence < threshold
        4. Records event for analysis

        Does NOT fail the operation - allows monitoring and analysis of
        low-coherence bifurcations for research purposes.
        """
        import logging
        from .metabolism import compute_subepi_collective_coherence

        # Compute collective coherence
        coherence = compute_subepi_collective_coherence(G, node)

        # Store for telemetry (always store, even if 0.0 for single/no sub-EPIs)
        G.nodes[node]["_thol_collective_coherence"] = coherence

        # Get threshold from graph config
        min_coherence = float(G.graph.get("THOL_MIN_COLLECTIVE_COHERENCE", 0.3))

        # Validate against threshold (only warn if we have multiple sub-EPIs)
        sub_epis = G.nodes[node].get("sub_epis", [])
        if len(sub_epis) >= 2 and coherence < min_coherence:
            # Log warning (but don't fail - allow monitoring)
            logger = logging.getLogger(__name__)
            logger.warning(
                f"Node {node}: THOL collective coherence ({coherence:.3f}) < "
                f"threshold ({min_coherence}). Sub-EPIs may be fragmenting. "
                f"Sub-EPI count: {len(sub_epis)}."
            )

            # Record event for analysis
            events = G.graph.setdefault("thol_coherence_warnings", [])
            events.append(
                {
                    "node": node,
                    "coherence": coherence,
                    "threshold": min_coherence,
                    "sub_epi_count": len(sub_epis),
                }
            )

    def _validate_preconditions(self, G: TNFRGraph, node: Any) -> None:
        """Validate THOL-specific preconditions."""
        from .preconditions import validate_self_organization

        validate_self_organization(G, node)

    def _collect_metrics(
        self, G: TNFRGraph, node: Any, state_before: dict[str, Any]
    ) -> dict[str, Any]:
        """Collect THOL-specific metrics."""
        from .metrics import self_organization_metrics

        return self_organization_metrics(G, node, state_before["epi"], state_before["vf"])