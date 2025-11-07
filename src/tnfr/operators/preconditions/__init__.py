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
    import logging

from ...alias import get_attr
from ...constants.aliases import ALIAS_DNFR, ALIAS_EPI, ALIAS_THETA, ALIAS_VF
from ...config.operator_names import (
    DESTABILIZERS_STRONG,
    DESTABILIZERS_MODERATE,
    DESTABILIZERS_WEAK,
    BIFURCATION_WINDOWS,
)

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
    "diagnose_coherence_readiness",
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
    """IL - Coherence requires active EPI, νf, and manageable ΔNFR.

    This function delegates to the strict validation implementation
    in coherence.py module, which provides comprehensive canonical
    precondition checks according to TNFR.pdf §2.2.1.

    Parameters
    ----------
    G : TNFRGraph
        Graph containing the node
    node : NodeId
        Node to validate

    Raises
    ------
    ValueError
        If critical preconditions are not met (active EPI, νf, non-saturated state)

    Warnings
    --------
    UserWarning
        For suboptimal conditions (zero ΔNFR, critical ΔNFR, isolated node)

    Notes
    -----
    For backward compatibility, this function maintains the same signature
    as the legacy validate_coherence but now provides enhanced validation.
    
    See Also
    --------
    tnfr.operators.preconditions.coherence.validate_coherence_strict : Full implementation
    """
    from .coherence import validate_coherence_strict

    validate_coherence_strict(G, node)


def diagnose_coherence_readiness(G: "TNFRGraph", node: "NodeId") -> dict:
    """Diagnose node readiness for IL (Coherence) operator.

    Provides comprehensive diagnostic report with readiness status and
    actionable recommendations for IL operator application.

    Parameters
    ----------
    G : TNFRGraph
        Graph containing the node
    node : NodeId
        Node to diagnose

    Returns
    -------
    dict
        Diagnostic report with readiness status, check results, values, and recommendations

    See Also
    --------
    tnfr.operators.preconditions.coherence.diagnose_coherence_readiness : Full implementation
    """
    from .coherence import diagnose_coherence_readiness as _diagnose

    return _diagnose(G, node)


def validate_dissonance(G: "TNFRGraph", node: "NodeId") -> None:
    """OZ - Dissonance requires vf > 0 to generate meaningful dissonance.
    
    Also detects bifurcation readiness when ∂²EPI/∂t² > τ, enabling
    alternative structural paths (ZHIR, NUL, IL, THOL).

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
        
    Notes
    -----
    When bifurcation threshold is exceeded, sets node['_bifurcation_ready'] = True
    and logs the event for telemetry. This enables downstream operators to
    respond to OZ-induced structural acceleration.
    """
    import logging
    import warnings
    
    logger = logging.getLogger(__name__)
    
    vf = _get_node_attr(G, node, ALIAS_VF)
    min_vf = float(G.graph.get("OZ_MIN_VF", 0.01))
    if vf < min_vf:
        raise OperatorPreconditionError(
            "Dissonance", f"Structural frequency too low (νf={vf:.3f} < {min_vf:.3f})"
        )
    
    # Check bifurcation readiness using existing THOL infrastructure
    # Reuse _compute_epi_acceleration from SelfOrganization
    from ..definitions import SelfOrganization
    
    thol_instance = SelfOrganization()
    d2_epi = thol_instance._compute_epi_acceleration(G, node)
    
    # Get bifurcation threshold
    tau = float(G.graph.get("BIFURCATION_THRESHOLD_TAU", 0.5))
    
    # Store d²EPI for telemetry (using existing ALIAS_D2EPI)
    from ...constants.aliases import ALIAS_D2EPI
    from ...alias import set_attr
    set_attr(G.nodes[node], ALIAS_D2EPI, d2_epi)
    
    # Check if bifurcation threshold exceeded
    if d2_epi > tau:
        # Mark node as bifurcation-ready
        G.nodes[node]["_bifurcation_ready"] = True
        logger.info(
            f"Node {node}: bifurcation threshold exceeded "
            f"(∂²EPI/∂t²={d2_epi:.3f} > τ={tau}). "
            f"Alternative structural paths enabled."
        )
    else:
        # Clear flag if previously set
        G.nodes[node]["_bifurcation_ready"] = False
    
    # Additional checks for OZ preconditions
    epi = _get_node_attr(G, node, ALIAS_EPI)
    dnfr = _get_node_attr(G, node, ALIAS_DNFR)
    
    # Warn if EPI too low to withstand dissonance
    min_epi = float(G.graph.get("OZ_MIN_EPI", 0.2))
    if epi < min_epi:
        raise OperatorPreconditionError(
            "Dissonance",
            f"EPI too low to withstand dissonance (EPI={epi:.3f} < {min_epi:.3f})"
        )
    
    # Warn if ΔNFR already critically high
    max_dnfr = float(G.graph.get("OZ_MAX_DNFR", 0.8))
    if abs(dnfr) > max_dnfr:
        warnings.warn(
            f"Applying OZ with high ΔNFR (|ΔNFR|={abs(dnfr):.3f}) may cause collapse. "
            f"Consider IL (Coherence) before OZ.",
            stacklevel=3
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
    
    Also detects and records the destabilizer type that enabled this self-organization
    for telemetry and structural tracing purposes.

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
    
    Notes
    -----
    This function implements R4 Extended telemetry by analyzing the glyph_history
    to determine which destabilizer (strong/moderate/weak) enabled the self-organization.
    """
    import logging
    import warnings

    logger = logging.getLogger(__name__)

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
    
    # R4 Extended: Detect and record destabilizer type for telemetry
    _record_destabilizer_context(G, node, logger)


def _record_destabilizer_context(
    G: "TNFRGraph", node: "NodeId", logger: "logging.Logger"
) -> None:
    """Detect and record which destabilizer enabled the current mutation.
    
    This implements R4 Extended telemetry by analyzing the glyph_history
    to determine which destabilizer type (strong/moderate/weak) is within
    its appropriate bifurcation window.
    
    Parameters
    ----------
    G : TNFRGraph
        Graph containing the node
    node : NodeId
        Node being mutated
    logger : logging.Logger
        Logger for telemetry output
        
    Notes
    -----
    The destabilizer context is stored in node['_mutation_context'] for
    structural tracing and post-hoc analysis. This enables understanding
    of bifurcation pathways without breaking TNFR structural invariants.
    """
    # Get glyph history from node
    history = G.nodes[node].get('glyph_history', [])
    if not history:
        # No history available, mutation enabled by external factors
        G.nodes[node]['_mutation_context'] = {
            'destabilizer_type': None,
            'destabilizer_operator': None,
            'destabilizer_distance': None,
            'recent_history': [],
        }
        return
    
    # Import glyph_function_name to convert glyphs to operator names
    from ..grammar import glyph_function_name
    
    # Get recent history (up to max window size)
    max_window = BIFURCATION_WINDOWS['strong']
    recent = list(history)[-max_window:] if len(history) > max_window else list(history)
    recent_names = [glyph_function_name(g) for g in recent]
    
    # Search backwards for destabilizers, checking window constraints
    destabilizer_found = None
    destabilizer_type = None
    destabilizer_distance = None
    
    for i, op_name in enumerate(reversed(recent_names)):
        distance = i + 1  # Distance from mutation (1 = immediate predecessor)
        
        # Check strong destabilizers (window = 4)
        if op_name in DESTABILIZERS_STRONG and distance <= BIFURCATION_WINDOWS['strong']:
            destabilizer_found = op_name
            destabilizer_type = 'strong'
            destabilizer_distance = distance
            break
        
        # Check moderate destabilizers (window = 2)
        if op_name in DESTABILIZERS_MODERATE and distance <= BIFURCATION_WINDOWS['moderate']:
            destabilizer_found = op_name
            destabilizer_type = 'moderate'
            destabilizer_distance = distance
            break
        
        # Check weak destabilizers (window = 1, immediate only)
        if op_name in DESTABILIZERS_WEAK and distance == 1:
            destabilizer_found = op_name
            destabilizer_type = 'weak'
            destabilizer_distance = distance
            break
    
    # Store context in node metadata for telemetry
    context = {
        'destabilizer_type': destabilizer_type,
        'destabilizer_operator': destabilizer_found,
        'destabilizer_distance': destabilizer_distance,
        'recent_history': recent_names,
    }
    G.nodes[node]['_mutation_context'] = context
    
    # Log telemetry for structural tracing
    if destabilizer_found:
        logger.info(
            f"Node {node}: ZHIR enabled by {destabilizer_type} destabilizer "
            f"({destabilizer_found}) at distance {destabilizer_distance}"
        )
    else:
        logger.warning(
            f"Node {node}: ZHIR without detectable destabilizer in history. "
            f"Recent operators: {recent_names}"
        )


def validate_mutation(G: "TNFRGraph", node: "NodeId") -> None:
    """ZHIR - Mutation requires node to be in valid structural state.
    
    Also detects and records the destabilizer type that enabled this mutation
    for telemetry and structural tracing purposes.

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
        
    Notes
    -----
    This function implements R4 Extended telemetry by analyzing the glyph_history
    to determine which destabilizer (strong/moderate/weak) enabled the mutation.
    The destabilizer context is stored in node metadata for structural tracing.
    """
    import logging
    
    logger = logging.getLogger(__name__)
    
    # Mutation is a phase change, require minimum vf for meaningful transition
    vf = _get_node_attr(G, node, ALIAS_VF)
    min_vf = float(G.graph.get("ZHIR_MIN_VF", 0.05))
    if vf < min_vf:
        raise OperatorPreconditionError(
            "Mutation",
            f"Structural frequency too low for mutation (νf={vf:.3f} < {min_vf:.3f})",
        )
    
    # R4 Extended: Detect and record destabilizer type for telemetry
    # This provides structural traceability for bifurcation events
    _record_destabilizer_context(G, node, logger)


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
