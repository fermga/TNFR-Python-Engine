"""Node lifecycle management for TNFR canonical theory.

According to TNFR theory (El pulso que nos atraviesa, p.44), nodes follow
a canonical lifecycle:

1. Activation - Node emerges through sufficient reorganization
2. Stabilization - Finds coherent phase and form
3. Propagation - Reorganizes its network environment
4. Mutation - Transforms through dissonance
5. Collapse - Loses phase/frequency and dissolves

This module provides lifecycle state tracking and transition validation.
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..types import NodeId, TNFRGraph

from ..alias import get_attr
from ..constants.aliases import ALIAS_EPI, ALIAS_VF, ALIAS_DNFR, ALIAS_THETA

__all__ = [
    "LifecycleState",
    "CollapseReason",
    "get_lifecycle_state",
    "check_collapse_conditions",
    "should_collapse",
]

# Default thresholds for lifecycle state determination
DEFAULT_MIN_PHASE_COUPLING = 0.1  # Minimum phase coupling before decoupling collapse


class LifecycleState(Enum):
    """Canonical TNFR node lifecycle states.

    These states correspond to the fundamental phases of node existence
    in the Resonant Fractal Nature paradigm.
    """

    DORMANT = "dormant"
    """Node exists but has minimal structural frequency (νf < activation_threshold)."""

    ACTIVATION = "activation"
    """Node is emerging with increasing νf and ΔNFR."""

    STABILIZATION = "stabilization"
    """Node is finding coherent form (high C(t), decreasing |ΔNFR|)."""

    PROPAGATION = "propagation"
    """Node is reorganizing its environment (high phase coupling)."""

    MUTATION = "mutation"
    """Node is undergoing phase transformation (high |ΔNFR|, phase shifts)."""

    COLLAPSING = "collapsing"
    """Node is losing coherence and approaching dissolution."""

    COLLAPSED = "collapsed"
    """Node has dissolved (νf → 0 or extreme dissonance)."""


class CollapseReason(Enum):
    """Canonical reasons for node collapse in TNFR.

    These correspond to the fundamental ways structural coherence can fail.
    """

    FREQUENCY_FAILURE = "frequency_failure"
    """Structural frequency dropped below collapse threshold (νf → 0)."""

    EXTREME_DISSONANCE = "extreme_dissonance"
    """ΔNFR magnitude exceeded bifurcation threshold."""

    NETWORK_DECOUPLING = "network_decoupling"
    """Phase coherence with network dropped below coupling threshold."""

    EPI_DISSOLUTION = "epi_dissolution"
    """Primary Information Structure lost coherence (EPI → 0)."""


def _get_node_attr(
    G: TNFRGraph, node: NodeId, aliases: tuple[str, ...], default: float = 0.0
) -> float:
    """Get node attribute using alias fallback."""
    return float(get_attr(G.nodes[node], aliases, default))


def get_lifecycle_state(
    G: TNFRGraph,
    node: NodeId,
    *,
    config: dict[str, Any] | None = None,
) -> LifecycleState:
    """Determine current lifecycle state of a node.

    Analyzes node's structural parameters (νf, ΔNFR, EPI, θ) to determine
    its position in the canonical TNFR lifecycle.

    Parameters
    ----------
    G : TNFRGraph
        Graph containing the node
    node : NodeId
        Node to analyze
    config : dict, optional
        Configuration overrides for thresholds:
        - activation_threshold: Min νf for activation (default: 0.1)
        - collapse_threshold: Min νf to avoid collapse (default: 0.01)
        - bifurcation_threshold: Max |ΔNFR| before bifurcation (default: 10.0)
        - stabilization_dnfr: Max |ΔNFR| for stabilization (default: 1.0)
        - stabilization_coherence: Min coherence for stabilization (default: 0.8)
        - propagation_coupling: Min phase coupling for propagation (default: 0.7)
        - mutation_dnfr: Min |ΔNFR| for mutation state (default: 5.0)

    Returns
    -------
    LifecycleState
        Current lifecycle state

    Notes
    -----
    Collapse conditions are checked first. Among active states, the state
    with the strongest indicators is returned (e.g., high |ΔNFR| → mutation
    takes precedence over stabilization).

    Examples
    --------
    >>> from tnfr.structural import create_nfr
    >>> G, node = create_nfr("test", epi=0.5, vf=1.0)
    >>> G.nodes[node]["ΔNFR"] = 0.5
    >>> state = get_lifecycle_state(G, node)
    >>> state.value
    'activation'
    """
    if config is None:
        config = {}

    # Get thresholds from config or graph or defaults
    def _get_threshold(key: str, default: float) -> float:
        return float(config.get(key, G.graph.get(key.upper(), default)))

    activation_threshold = _get_threshold("activation_threshold", 0.1)
    collapse_threshold = _get_threshold("collapse_threshold", 0.01)
    bifurcation_threshold = _get_threshold("bifurcation_threshold", 10.0)
    stabilization_dnfr = _get_threshold("stabilization_dnfr", 1.0)
    stabilization_coherence = _get_threshold("stabilization_coherence", 0.8)
    propagation_coupling = _get_threshold("propagation_coupling", 0.7)
    mutation_dnfr = _get_threshold("mutation_dnfr", 5.0)

    # Get node structural parameters
    vf = _get_node_attr(G, node, ALIAS_VF)
    dnfr = _get_node_attr(G, node, ALIAS_DNFR)
    epi = _get_node_attr(G, node, ALIAS_EPI)
    theta = _get_node_attr(G, node, ALIAS_THETA)

    # Check for collapse conditions first
    if vf < collapse_threshold:
        return LifecycleState.COLLAPSING

    if abs(dnfr) > bifurcation_threshold:
        return LifecycleState.COLLAPSING

    # Compute phase coupling (simplified - could use full network coupling)
    neighbors = list(G.neighbors(node))
    if neighbors:
        import math

        neighbor_phases = [_get_node_attr(G, n, ALIAS_THETA) for n in neighbors]
        mean_neighbor_phase = sum(neighbor_phases) / len(neighbor_phases)
        phase_diff = abs(theta - mean_neighbor_phase)
        # Normalize to [0, 1] where 1 is perfect alignment
        phase_coupling = 1.0 - min(phase_diff, math.pi) / math.pi
    else:
        phase_coupling = 0.0

    # Check for decoupling collapse
    if neighbors and phase_coupling < DEFAULT_MIN_PHASE_COUPLING:
        return LifecycleState.COLLAPSING

    # Check active states (priority: mutation > propagation > stabilization > activation)

    # Mutation: High dissonance with sufficient frequency
    if abs(dnfr) > mutation_dnfr and vf > activation_threshold:
        return LifecycleState.MUTATION

    # Propagation: Strong network coupling
    if phase_coupling > propagation_coupling and vf > activation_threshold:
        return LifecycleState.PROPAGATION

    # Stabilization: High coherence, low dissonance
    # Note: C(t) computation would require full graph state, using EPI as proxy
    if abs(dnfr) < stabilization_dnfr and epi > stabilization_coherence:
        return LifecycleState.STABILIZATION

    # Activation: Above activation threshold but not yet stabilized
    if vf >= activation_threshold:
        return LifecycleState.ACTIVATION

    # Dormant: Below activation threshold but above collapse
    return LifecycleState.DORMANT


def check_collapse_conditions(
    G: TNFRGraph,
    node: NodeId,
    *,
    config: dict[str, Any] | None = None,
) -> tuple[bool, CollapseReason | None]:
    """Check if node meets any collapse conditions.

    Parameters
    ----------
    G : TNFRGraph
        Graph containing the node
    node : NodeId
        Node to check
    config : dict, optional
        Configuration overrides for collapse thresholds

    Returns
    -------
    should_collapse : bool
        True if node should collapse
    reason : CollapseReason | None
        Reason for collapse, or None if not collapsing

    Notes
    -----
    Multiple collapse conditions may be met simultaneously. This function
    returns the first detected condition in priority order:
    1. Frequency failure (most fundamental)
    2. Extreme dissonance (structural instability)
    3. Network decoupling (loss of resonance)
    4. EPI dissolution (form loss)
    """
    if config is None:
        config = {}

    def _get_threshold(key: str, default: float) -> float:
        return float(config.get(key, G.graph.get(key.upper(), default)))

    collapse_threshold = _get_threshold("collapse_threshold", 0.01)
    bifurcation_threshold = _get_threshold("bifurcation_threshold", 10.0)
    min_coupling = _get_threshold("min_phase_coupling", 0.1)
    min_epi = _get_threshold("min_epi", 0.01)

    # Get node parameters
    vf = _get_node_attr(G, node, ALIAS_VF)
    dnfr = _get_node_attr(G, node, ALIAS_DNFR)
    epi = _get_node_attr(G, node, ALIAS_EPI)
    theta = _get_node_attr(G, node, ALIAS_THETA)

    # Check frequency failure (most fundamental)
    if vf < collapse_threshold:
        return (True, CollapseReason.FREQUENCY_FAILURE)

    # Check extreme dissonance
    if abs(dnfr) > bifurcation_threshold:
        return (True, CollapseReason.EXTREME_DISSONANCE)

    # Check network decoupling
    neighbors = list(G.neighbors(node))
    if neighbors:
        import math

        neighbor_phases = [_get_node_attr(G, n, ALIAS_THETA) for n in neighbors]
        mean_neighbor_phase = sum(neighbor_phases) / len(neighbor_phases)
        phase_diff = abs(theta - mean_neighbor_phase)
        phase_coupling = 1.0 - min(phase_diff, math.pi) / math.pi

        if phase_coupling < min_coupling:
            return (True, CollapseReason.NETWORK_DECOUPLING)

    # Check EPI dissolution
    if epi < min_epi:
        return (True, CollapseReason.EPI_DISSOLUTION)

    return (False, None)


def should_collapse(
    G: TNFRGraph,
    node: NodeId,
    *,
    config: dict[str, Any] | None = None,
) -> bool:
    """Check if node should collapse (simplified interface).

    Parameters
    ----------
    G : TNFRGraph
        Graph containing the node
    node : NodeId
        Node to check
    config : dict, optional
        Configuration overrides

    Returns
    -------
    bool
        True if node meets collapse conditions

    See Also
    --------
    check_collapse_conditions : Full collapse check with reason
    get_lifecycle_state : Complete lifecycle state determination
    """
    should_collapse_flag, _ = check_collapse_conditions(G, node, config=config)
    return should_collapse_flag
