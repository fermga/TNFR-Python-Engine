"""Strict precondition validation for EN (Reception) operator.

This module implements canonical precondition validation for the Reception (EN)
structural operator according to TNFR.pdf §2.2.1. EN requires specific structural
conditions to maintain TNFR operational fidelity:

1. **Receptive capacity**: EPI must be below saturation threshold (node not saturated)
2. **Minimal dissonance**: DNFR must be below threshold (low reorganization pressure)
3. **Emission sources**: Network should have active emission sources (warning for isolated nodes)

These validations protect structural integrity by ensuring EN is only applied to
nodes in the appropriate state for coherence integration.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ...types import TNFRGraph

__all__ = ["validate_reception_strict"]


def validate_reception_strict(G: TNFRGraph, node: Any) -> None:
    """Validate strict canonical preconditions for EN (Reception) operator.

    According to TNFR.pdf §2.2.1, Reception (EN - Recepción estructural) requires:

    1. **Receptive capacity**: EPI < saturation threshold (node has capacity to receive)
    2. **Minimal dissonance**: DNFR < threshold (low reorganization pressure for stable integration)
    3. **Emission sources**: Network connectivity with active sources (warning if isolated)

    Parameters
    ----------
    G : TNFRGraph
        Graph containing the node to validate
    node : Any
        Node identifier for validation

    Raises
    ------
    ValueError
        If EPI >= saturation threshold (node saturated - cannot receive more coherence)
        If DNFR >= threshold (excessive dissonance - consider IL/Coherence first)

    Warnings
    --------
    UserWarning
        If node is isolated in a multi-node network (no emission sources available)

    Notes
    -----
    Thresholds are configurable via:
    - Graph metadata: ``G.graph["EPI_SATURATION_MAX"]``, ``G.graph["DNFR_RECEPTION_MAX"]``
    - Module defaults: :data:`tnfr.config.thresholds.EPI_SATURATION_MAX`, etc.

    Examples
    --------
    >>> from tnfr.structural import create_nfr
    >>> from tnfr.operators.preconditions.reception import validate_reception_strict
    >>> G, node = create_nfr("test", epi=0.5, vf=0.9)
    >>> G.nodes[node]["dnfr"] = 0.08
    >>> validate_reception_strict(G, node)  # OK - receptive capacity available

    >>> G2, node2 = create_nfr("saturated", epi=0.95, vf=1.0)
    >>> validate_reception_strict(G2, node2)  # doctest: +SKIP
    Traceback (most recent call last):
        ...
    ValueError: EN precondition failed: EPI=0.950 >= 0.9. Node saturated, cannot receive more coherence.

    See Also
    --------
    tnfr.config.thresholds : Configurable threshold constants
    tnfr.operators.preconditions : Base precondition validators
    tnfr.operators.definitions.Reception : Reception operator implementation
    """
    import warnings

    from ...alias import get_attr
    from ...constants.aliases import ALIAS_DNFR, ALIAS_EPI
    from ...config.thresholds import DNFR_RECEPTION_MAX, EPI_SATURATION_MAX

    # Get current node state
    epi = float(get_attr(G.nodes[node], ALIAS_EPI, 0.0))
    dnfr = float(get_attr(G.nodes[node], ALIAS_DNFR, 0.0))

    # Get configurable thresholds (allow override via graph metadata)
    epi_threshold = float(G.graph.get("EPI_SATURATION_MAX", EPI_SATURATION_MAX))
    dnfr_threshold = float(G.graph.get("DNFR_RECEPTION_MAX", DNFR_RECEPTION_MAX))

    # Precondition 1: EPI must be below saturation threshold (receptive capacity available)
    # Reception integrates external coherence into local structure.
    # If EPI is saturated, node cannot accommodate more coherence.
    if epi >= epi_threshold:
        raise ValueError(
            f"EN precondition failed: EPI={epi:.3f} >= {epi_threshold:.3f}. "
            f"Node saturated, cannot receive more coherence. "
            f"Apply IL (Coherence) first to stabilize and compress structure, "
            f"or apply NUL (Contraction) to reduce complexity if appropriate."
        )

    # Precondition 2: DNFR must be below threshold (minimal dissonance for stable integration)
    # Excessive reorganization pressure prevents effective integration of external coherence.
    # Node must first stabilize before receiving more information.
    if dnfr >= dnfr_threshold:
        raise ValueError(
            f"EN precondition failed: DNFR={dnfr:.3f} >= {dnfr_threshold:.3f}. "
            f"Excessive dissonance prevents reception. "
            f"Consider IL (Coherence) first to reduce reorganization pressure."
        )

    # Precondition 3: Emission sources check (warning only - not a hard failure)
    # Isolated nodes can still apply EN, but there are no external sources to receive from
    node_degree = G.degree(node)
    network_size = len(G)

    if node_degree == 0 and network_size > 1:
        warnings.warn(
            f"EN warning: Node {node!r} isolated. No emission sources available. "
            f"Reception possible but no external coherence to integrate. "
            f"Consider UM (Coupling) to establish network connections first.",
            UserWarning,
            stacklevel=3,
        )
