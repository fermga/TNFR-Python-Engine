"""Precondition validators for TNFR structural operators.

Each operator has specific requirements that must be met before execution
to maintain TNFR structural invariants. This package provides validators
for each of the 13 canonical operators.

The preconditions package has been restructured to support both legacy
imports (from ..preconditions import validate_*) and new modular imports
(from ..preconditions.emission import validate_emission_strict).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ...types import NodeId, TNFRGraph

from ...alias import get_attr
from ...constants.aliases import ALIAS_DNFR, ALIAS_EPI, ALIAS_THETA, ALIAS_VF

__all__ = [
    "OperatorPreconditionError",
    "validate_emission",
    "validate_reception",
    "validate_coherence",
    "validate_dissonance",
    "validate_coupling",
    "validate_resonance",
    "validate_silence",
    "validate_expansion",
    "validate_contraction",
    "validate_self_organization",
    "validate_mutation",
    "validate_transition",
    "validate_recursivity",
]


class OperatorPreconditionError(Exception):
    """Raised when an operator's preconditions are not met."""

    def __init__(self, operator: str, reason: str) -> None:
        """Initialize precondition error.

        Parameters
        ----------
        operator : str
            Name of the operator that failed validation
        reason : str
            Description of why the precondition failed
        """
        self.operator = operator
        self.reason = reason
        super().__init__(f"{operator}: {reason}")


def _get_node_attr(
    G: "TNFRGraph", node: "NodeId", aliases: tuple[str, ...], default: float = 0.0
) -> float:
    """Get node attribute using alias fallback."""
    return float(get_attr(G.nodes[node], aliases, default))


def validate_emission(G: "TNFRGraph", node: "NodeId") -> None:
    """AL - Emission requires node in latent or low activation state.

    Parameters
    ----------
    G : TNFRGraph
        Graph containing the node
    node : NodeId
        Node to validate

    Raises
    ------
    OperatorPreconditionError
        If EPI is already too high for emission to be meaningful
    """
    epi = _get_node_attr(G, node, ALIAS_EPI)
    # Emission is meant to activate latent nodes, not boost already active ones
    # This is a soft threshold - configurable via graph metadata
    max_epi = float(G.graph.get("AL_MAX_EPI_FOR_EMISSION", 0.8))
    if epi >= max_epi:
        raise OperatorPreconditionError(
            "Emission", f"Node already active (EPI={epi:.3f} >= {max_epi:.3f})"
        )


def validate_reception(G: "TNFRGraph", node: "NodeId") -> None:
    """EN - Reception requires node to have neighbors to receive from.

    Parameters
    ----------
    G : TNFRGraph
        Graph containing the node
    node : NodeId
        Node to validate

    Raises
    ------
    OperatorPreconditionError
        If node has no neighbors to receive energy from
    """
    neighbors = list(G.neighbors(node))
    if not neighbors:
        raise OperatorPreconditionError(
            "Reception", "Node has no neighbors to receive energy from"
        )


def validate_coherence(G: "TNFRGraph", node: "NodeId") -> None:
    """IL - Coherence requires significant ΔNFR to stabilize.

    Parameters
    ----------
    G : TNFRGraph
        Graph containing the node
    node : NodeId
        Node to validate

    Raises
    ------
    OperatorPreconditionError
        If |ΔNFR| is already near zero (nothing meaningful to stabilize)

    Notes
    -----
    Coherence acts on the absolute magnitude of ΔNFR, reducing structural
    instability regardless of sign. We validate that |ΔNFR| > threshold
    to ensure there is sufficient reorganization to compress.
    """
    dnfr = _get_node_attr(G, node, ALIAS_DNFR)
    min_dnfr = float(G.graph.get("IL_MIN_DNFR", 1e-6))
    if abs(dnfr) < min_dnfr:
        raise OperatorPreconditionError(
            "Coherence",
            f"ΔNFR already minimal (|ΔNFR|={abs(dnfr):.3e} < {min_dnfr:.3e})",
        )


def validate_dissonance(G: "TNFRGraph", node: "NodeId") -> None:
    """OZ - Dissonance requires vf > 0 to generate meaningful dissonance.

    Parameters
    ----------
    G : TNFRGraph
        Graph containing the node
    node : NodeId
        Node to validate

    Raises
    ------
    OperatorPreconditionError
        If structural frequency is too low for dissonance to be effective
    """
    vf = _get_node_attr(G, node, ALIAS_VF)
    min_vf = float(G.graph.get("OZ_MIN_VF", 0.01))
    if vf < min_vf:
        raise OperatorPreconditionError(
            "Dissonance", f"Structural frequency too low (νf={vf:.3f} < {min_vf:.3f})"
        )


def validate_coupling(G: "TNFRGraph", node: "NodeId") -> None:
    """UM - Coupling requires node to have potential coupling targets.

    Parameters
    ----------
    G : TNFRGraph
        Graph containing the node
    node : NodeId
        Node to validate

    Raises
    ------
    OperatorPreconditionError
        If node is isolated with no potential coupling targets
    """
    # Coupling can work with existing neighbors or create new links
    # Only fail if graph has no other nodes at all
    if G.number_of_nodes() <= 1:
        raise OperatorPreconditionError(
            "Coupling", "Graph has no other nodes to couple with"
        )


def validate_resonance(G: "TNFRGraph", node: "NodeId") -> None:
    """RA - Resonance requires neighbors to propagate energy.

    Parameters
    ----------
    G : TNFRGraph
        Graph containing the node
    node : NodeId
        Node to validate

    Raises
    ------
    OperatorPreconditionError
        If node has no neighbors for resonance propagation
    """
    neighbors = list(G.neighbors(node))
    if not neighbors:
        raise OperatorPreconditionError(
            "Resonance", "Node has no neighbors for resonance propagation"
        )


def validate_silence(G: "TNFRGraph", node: "NodeId") -> None:
    """SHA - Silence requires vf > 0 to reduce.

    Parameters
    ----------
    G : TNFRGraph
        Graph containing the node
    node : NodeId
        Node to validate

    Raises
    ------
    OperatorPreconditionError
        If structural frequency already near zero
    """
    vf = _get_node_attr(G, node, ALIAS_VF)
    min_vf = float(G.graph.get("SHA_MIN_VF", 0.01))
    if vf < min_vf:
        raise OperatorPreconditionError(
            "Silence",
            f"Structural frequency already minimal (νf={vf:.3f} < {min_vf:.3f})",
        )


def validate_expansion(G: "TNFRGraph", node: "NodeId") -> None:
    """VAL - Expansion requires vf below maximum threshold.

    Parameters
    ----------
    G : TNFRGraph
        Graph containing the node
    node : NodeId
        Node to validate

    Raises
    ------
    OperatorPreconditionError
        If structural frequency already at maximum
    """
    vf = _get_node_attr(G, node, ALIAS_VF)
    max_vf = float(G.graph.get("VAL_MAX_VF", 10.0))
    if vf >= max_vf:
        raise OperatorPreconditionError(
            "Expansion",
            f"Structural frequency at maximum (νf={vf:.3f} >= {max_vf:.3f})",
        )


def validate_contraction(G: "TNFRGraph", node: "NodeId") -> None:
    """NUL - Contraction requires vf > minimum to reduce.

    Parameters
    ----------
    G : TNFRGraph
        Graph containing the node
    node : NodeId
        Node to validate

    Raises
    ------
    OperatorPreconditionError
        If structural frequency already at minimum
    """
    vf = _get_node_attr(G, node, ALIAS_VF)
    min_vf = float(G.graph.get("NUL_MIN_VF", 0.1))
    if vf <= min_vf:
        raise OperatorPreconditionError(
            "Contraction",
            f"Structural frequency at minimum (νf={vf:.3f} <= {min_vf:.3f})",
        )


def validate_self_organization(G: "TNFRGraph", node: "NodeId") -> None:
    """THOL - Self-organization requires minimum EPI, positive ΔNFR, and connectivity.

    T'HOL implements structural metabolism and bifurcation. Preconditions ensure
    sufficient structure and reorganization pressure for self-organization.

    Parameters
    ----------
    G : TNFRGraph
        Graph containing the node
    node : NodeId
        Node to validate

    Raises
    ------
    OperatorPreconditionError
        If EPI is too low for bifurcation, or if ΔNFR is non-positive

    Warnings
    --------
    Warns if node is isolated - bifurcation may not propagate through network
    """
    import warnings

    epi = _get_node_attr(G, node, ALIAS_EPI)
    dnfr = _get_node_attr(G, node, ALIAS_DNFR)

    # EPI must be sufficient for bifurcation
    min_epi = float(G.graph.get("THOL_MIN_EPI", 0.2))
    if epi < min_epi:
        raise OperatorPreconditionError(
            "Self-organization",
            f"EPI too low for bifurcation (EPI={epi:.3f} < {min_epi:.3f})",
        )

    # ΔNFR must be positive (reorganization pressure required)
    if dnfr <= 0:
        raise OperatorPreconditionError(
            "Self-organization",
            f"ΔNFR non-positive, no reorganization pressure (ΔNFR={dnfr:.3f})",
        )

    # Warn if node is isolated (bifurcation won't propagate)
    if G.degree(node) == 0:
        warnings.warn(
            f"Node {node} is isolated - bifurcation may not propagate through network",
            stacklevel=3,
        )


def validate_mutation(G: "TNFRGraph", node: "NodeId") -> None:
    """ZHIR - Mutation requires node to be in valid structural state.

    Parameters
    ----------
    G : TNFRGraph
        Graph containing the node
    node : NodeId
        Node to validate

    Raises
    ------
    OperatorPreconditionError
        If node state is unsuitable for mutation
    """
    # Mutation is a phase change, require minimum vf for meaningful transition
    vf = _get_node_attr(G, node, ALIAS_VF)
    min_vf = float(G.graph.get("ZHIR_MIN_VF", 0.05))
    if vf < min_vf:
        raise OperatorPreconditionError(
            "Mutation",
            f"Structural frequency too low for mutation (νf={vf:.3f} < {min_vf:.3f})",
        )


def validate_transition(G: "TNFRGraph", node: "NodeId") -> None:
    """NAV - Transition requires ΔNFR and vf for controlled handoff.

    Parameters
    ----------
    G : TNFRGraph
        Graph containing the node
    node : NodeId
        Node to validate

    Raises
    ------
    OperatorPreconditionError
        If node lacks necessary dynamics for transition
    """
    vf = _get_node_attr(G, node, ALIAS_VF)
    min_vf = float(G.graph.get("NAV_MIN_VF", 0.01))
    if vf < min_vf:
        raise OperatorPreconditionError(
            "Transition",
            f"Structural frequency too low for transition (νf={vf:.3f} < {min_vf:.3f})",
        )


def validate_recursivity(G: "TNFRGraph", node: "NodeId") -> None:
    """REMESH - Recursivity requires global network coherence threshold.

    Parameters
    ----------
    G : TNFRGraph
        Graph containing the node
    node : NodeId
        Node to validate

    Raises
    ------
    OperatorPreconditionError
        If network is not ready for remesh operation
    """
    # REMESH is a network-scale operation, check graph state
    min_nodes = int(G.graph.get("REMESH_MIN_NODES", 2))
    if G.number_of_nodes() < min_nodes:
        raise OperatorPreconditionError(
            "Recursivity",
            f"Network too small for remesh (n={G.number_of_nodes()} < {min_nodes})",
        )
