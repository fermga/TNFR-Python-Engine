"""SelfOrganization (THOL) operator.

Purpose: autonomous emergence; spawn sub-EPIs when d2_epi>tau.
Physics: bifurcation + metabolic capture of network signals.
Grammar: transformer (U4b) + handler (U4a) during bifurcation.
Effects: adds sub-structure; parent epi increments; preserves identity.
Preconditions: sufficient epi history; vf>0; elevated d2_epi; capacity.
Typical: OZ->THOL; THOL->IL; EN->THOL; THOL->RA; THOL->IL->RA.
Avoid: THOL without ΔNFR elevation; deep nesting beyond max depth.
"""

from __future__ import annotations

from typing import Any, ClassVar

from ..config.operator_names import SELF_ORGANIZATION
from ..types import Glyph, TNFRGraph
from .definitions_base import Operator
_THOL_SUB_EPI_SCALING = 0.25  # sub-EPI scale factor
_THOL_EMERGENCE_CONTRIBUTION = 0.1  # parent epi increment fraction


class SelfOrganization(Operator):
    """Spawn sub-EPIs on bifurcation; metabolic capture; update parent epi.

    Invariants: parent identity preserved; sub-EPIs coherent ensemble.
    Typical: OZ->THOL; THOL->IL; THOL->RA; EN->THOL; THOL->IL->RA.
    Metrics: d2_epi, sub_epi_value, bifurcation_level, collective_coherence.
    """

    __slots__ = ()
    name: ClassVar[str] = SELF_ORGANIZATION
    glyph: ClassVar[Glyph] = Glyph.THOL

    def __call__(self, G: TNFRGraph, node: Any, **kw: Any) -> None:
        """Apply THOL; if d2_epi>tau spawn sub-EPI; validate ensemble."""
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
        # Ensemble must stay coherent and preserve parent identity.
        # Always validate if node has sub-EPIs (new or existing).
        if G.nodes[node].get("sub_epis"):
            self._validate_collective_coherence(G, node)

    def _compute_epi_acceleration(self, G: TNFRGraph, node: Any) -> float:
        """Finite diff second derivative abs value from epi_history."""

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

    def _spawn_sub_epi(
        self, G: TNFRGraph, node: Any, d2_epi: float, tau: float
    ) -> None:
        """Create sub-EPI node; apply metabolic weights; update parent epi."""
        from ..alias import get_attr, set_attr
        from ..constants.aliases import ALIAS_EPI, ALIAS_VF, ALIAS_THETA
        from .metabolism import (
            capture_network_signals,
            metabolize_signals_into_subepi,
        )

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
        gradient_weight = float(
            G.graph.get("THOL_METABOLIC_GRADIENT_WEIGHT", 0.15)
        )
        complexity_weight = float(
            G.graph.get("THOL_METABOLIC_COMPLEXITY_WEIGHT", 0.10)
        )

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
        # Enables fractality: recursive operators + hierarchical metrics.
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
        """Add sub-node with inherited state; record hierarchy metadata."""
        from ..constants import (
            EPI_PRIMARY,
            VF_PRIMARY,
            THETA_PRIMARY,
            DNFR_PRIMARY,
        )

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
                "_bifurcation_level": child_level,
                "_hierarchy_path": child_path,  # Full ancestor chain
                "epi_history": [float(sub_epi)],
                "glyph_history": [],
            },
        )

        # Ensure ΔNFR hook is set for the sub-node
        # (inherits from graph-level hook, but ensure it's activated)
        if hasattr(G, "graph") and "_delta_nfr_hook" in G.graph:
            # Graph-level hook applies to sub-node automatically.
            pass

        # Track sub-node in parent
        sub_nodes_list.append(sub_node_id)
        G.nodes[parent_node]["sub_nodes"] = sub_nodes_list

        # Track hierarchy in graph metadata
        hierarchy = G.graph.setdefault("hierarchy", {})
        hierarchy.setdefault(parent_node, []).append(sub_node_id)

        return sub_node_id

    def _validate_bifurcation_depth(self, G: TNFRGraph, node: Any) -> None:
        """Warn if bifurcation depth exceeds configured max."""
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
        """Compute ensemble coherence; warn if below threshold."""
        import logging
        from .metabolism import compute_subepi_collective_coherence

        # Compute collective coherence
        coherence = compute_subepi_collective_coherence(G, node)

        # Always store telemetry value (even if 0.0).
        G.nodes[node]["_thol_collective_coherence"] = coherence

        # Get threshold from graph config
        min_coherence = float(
            G.graph.get("THOL_MIN_COLLECTIVE_COHERENCE", 0.3)
        )

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

        return self_organization_metrics(
            G, node, state_before["epi"], state_before["vf"]
        )
