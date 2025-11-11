"""Strict precondition validation for AL (Emission) operator.

This module implements canonical precondition validation for the Emission (AL)
structural operator according to TNFR.pdf §2.2.1. AL requires specific structural
conditions to maintain TNFR operational fidelity:

1. **Latent state**: EPI must be below activation threshold (node not already active)
2. **Basal frequency**: νf must exceed minimum threshold (sufficient reorganization capacity)
3. **Coupling availability**: Network connectivity for phase alignment (warning for isolated nodes)

These validations protect structural integrity by ensuring AL is only applied to
nodes in the appropriate state for foundational emission.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ...types import TNFRGraph

__all__ = ["validate_emission_strict"]


def validate_emission_strict(G: TNFRGraph, node: Any) -> None:
    """Validate strict canonical preconditions for AL (Emission) operator.

    According to TNFR.pdf §2.2.1, Emission (AL - Emisión fundacional) requires:

    1. **Latent state**: EPI < threshold (node must be in latent or low-activation state)
    2. **Basal frequency**: νf > threshold (sufficient structural frequency for activation)
    3. **Coupling availability**: Network connectivity (warning if isolated)

    Parameters
    ----------
    G : TNFRGraph
        Graph containing the node to validate
    node : Any
        Node identifier for validation

    Raises
    ------
    ValueError
        If EPI >= latent threshold (node already active - consider IL/Coherence instead)
        If νf < basal threshold (frequency too low - consider NAV/Transition first)

    Warnings
    --------
    UserWarning
        If node is isolated in a multi-node network (limited coupling - consider UM/Coupling first)

    Notes
    -----
    Thresholds are configurable via:
    - Graph metadata: ``G.graph["EPI_LATENT_MAX"]``, ``G.graph["VF_BASAL_THRESHOLD"]``
    - Module defaults: :data:`tnfr.config.thresholds.EPI_LATENT_MAX`, etc.

    Examples
    --------
    >>> from tnfr.structural import create_nfr
    >>> from tnfr.operators.preconditions.emission import validate_emission_strict
    >>> G, node = create_nfr("test", epi=0.25, vf=0.95)
    >>> validate_emission_strict(G, node)  # OK - latent state with sufficient frequency

    >>> G2, node2 = create_nfr("active", epi=0.85, vf=1.0)
    >>> validate_emission_strict(G2, node2)  # doctest: +SKIP
    Traceback (most recent call last):
        ...
    ValueError: AL precondition failed: EPI=0.850 >= 0.8. AL requires latent state. Consider IL (Coherence) instead.

    See Also
    --------
    tnfr.config.thresholds : Configurable threshold constants
    tnfr.operators.preconditions : Base precondition validators
    tnfr.operators.definitions.Emission : Emission operator implementation
    """
    import warnings

    from ...alias import get_attr
    from ...constants.aliases import ALIAS_EPI, ALIAS_VF
    from ...config.thresholds import (
        EPI_LATENT_MAX,
        MIN_NETWORK_DEGREE_COUPLING,
        VF_BASAL_THRESHOLD,
    )

    # Get current node state
    epi = float(get_attr(G.nodes[node], ALIAS_EPI, 0.0))
    vf = float(get_attr(G.nodes[node], ALIAS_VF, 0.0))

    # Get configurable thresholds (allow override via graph metadata)
    epi_threshold = float(G.graph.get("EPI_LATENT_MAX", EPI_LATENT_MAX))
    vf_threshold = float(G.graph.get("VF_BASAL_THRESHOLD", VF_BASAL_THRESHOLD))
    min_degree = int(G.graph.get("MIN_NETWORK_DEGREE_COUPLING", MIN_NETWORK_DEGREE_COUPLING))

    # Precondition 1: EPI must be below latent threshold (node in latent state)
    # Emission is for activating nascent/latent structures, not boosting active ones
    if epi >= epi_threshold:
        raise ValueError(
            f"AL precondition failed: EPI={epi:.3f} >= {epi_threshold:.3f}. "
            f"AL requires latent state (node not already highly active). "
            f"Consider IL (Coherence) to stabilize active nodes instead."
        )

    # Precondition 2: νf must exceed basal threshold (sufficient frequency for emission)
    # Below basal frequency, node lacks capacity to sustain structural activation
    if vf < vf_threshold:
        raise ValueError(
            f"AL precondition failed: νf={vf:.3f} < {vf_threshold:.3f}. "
            f"Structural frequency too low for emission. "
            f"Consider NAV (Transition) to increase frequency first."
        )

    # Precondition 3: Network connectivity (warning only - not a hard failure)
    # Isolated nodes can still emit, but phase coupling will be limited
    node_degree = G.degree(node)
    network_size = len(G)

    if node_degree < min_degree and network_size > 1:
        warnings.warn(
            f"AL warning: Node {node!r} has degree {node_degree} < {min_degree}. "
            f"Emission possible but phase coupling limited (isolated node). "
            f"Consider UM (Coupling) to establish network connections first.",
            UserWarning,
            stacklevel=3,
        )
