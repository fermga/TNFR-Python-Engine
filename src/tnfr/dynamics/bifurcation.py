"""Bifurcation dynamics and structural path selection for TNFR operators.

This module provides utilities for detecting bifurcation readiness and
determining viable structural reorganization paths after OZ-induced dissonance.

According to TNFR canonical theory (§2.3.3, R4), when ∂²EPI/∂t² > τ,
the system enters a bifurcation state enabling multiple reorganization
trajectories. This module implements path selection based on nodal state.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..types import Glyph, NodeId, TNFRGraph

from ..alias import get_attr
from ..constants.aliases import ALIAS_DNFR, ALIAS_EPI, ALIAS_VF
from ..types import Glyph

__all__ = [
    "get_bifurcation_paths",
]


def get_bifurcation_paths(G: "TNFRGraph", node: "NodeId") -> list["Glyph"]:
    """Return viable structural paths after OZ-induced bifurcation.
    
    When OZ (Dissonance) creates bifurcation readiness (∂²EPI/∂t² > τ),
    this function determines which operators can resolve the dissonance
    based on current nodal state.
    
    Parameters
    ----------
    G : TNFRGraph
        Graph containing the node
    node : NodeId
        Node identifier
        
    Returns
    -------
    list[Glyph]
        List of viable operator glyphs for structural reorganization.
        Empty list if node is not in bifurcation state.
        
    Notes
    -----
    **Canonical bifurcation paths:**
    
    - **ZHIR (Mutation)**: Viable if νf > 0.8 (sufficient for controlled transformation)
    - **NUL (Contraction)**: Viable if EPI < 0.5 (safe collapse window)
    - **IL (Coherence)**: Always viable (universal resolution path)
    - **THOL (Self-organization)**: Viable if degree >= 2 (network support)
    
    The node must have `_bifurcation_ready = True` flag, typically set by
    OZ precondition validation when ∂²EPI/∂t² exceeds threshold τ.
    
    Examples
    --------
    >>> from tnfr.structural import create_nfr
    >>> from tnfr.operators.definitions import Dissonance
    >>> from tnfr.dynamics.bifurcation import get_bifurcation_paths
    >>> G, node = create_nfr("test", epi=0.4, vf=1.0)
    >>> # Set up bifurcation conditions
    >>> G.nodes[node]["epi_history"] = [0.2, 0.35, 0.55]
    >>> Dissonance()(G, node, validate_preconditions=True)
    >>> paths = get_bifurcation_paths(G, node)
    >>> # Returns viable operators: [ZHIR, NUL, IL, THOL] or subset
    
    See Also
    --------
    tnfr.operators.preconditions.validate_dissonance : Sets bifurcation_ready flag
    tnfr.operators.definitions.SelfOrganization : Spawns sub-EPIs on bifurcation
    """
    # Check if bifurcation active
    if not G.nodes[node].get("_bifurcation_ready", False):
        return []  # No bifurcation active
    
    # Get node state for path evaluation
    dnfr = abs(float(get_attr(G.nodes[node], ALIAS_DNFR, 0.0)))
    epi = float(get_attr(G.nodes[node], ALIAS_EPI, 0.0))
    vf = float(get_attr(G.nodes[node], ALIAS_VF, 0.0))
    degree = G.degree(node)
    
    paths = []
    
    # ZHIR (Mutation) viable if sufficient νf for controlled transformation
    zhir_threshold = float(G.graph.get("ZHIR_BIFURCATION_VF_THRESHOLD", 0.8))
    if vf > zhir_threshold:
        paths.append(Glyph.ZHIR)
    
    # NUL (Contraction) viable if EPI low enough for safe collapse
    nul_threshold = float(G.graph.get("NUL_BIFURCATION_EPI_THRESHOLD", 0.5))
    if epi < nul_threshold:
        paths.append(Glyph.NUL)
    
    # IL (Coherence) always viable as universal resolution path
    paths.append(Glyph.IL)
    
    # THOL (Self-organization) viable if network connectivity supports it
    thol_min_degree = int(G.graph.get("THOL_BIFURCATION_MIN_DEGREE", 2))
    if degree >= thol_min_degree:
        paths.append(Glyph.THOL)
    
    return paths
