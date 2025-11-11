"""Unified phase compatibility calculations for TNFR operators.

This module provides canonical implementations of phase-based coupling strength
calculations used by multiple TNFR operators (UM, RA, THOL). All operators that
perform phase-based coupling or propagation MUST use these functions to ensure
consistency with TNFR physics and Invariant #5.

Physical Foundation
-------------------

**Phase Compatibility in TNFR:**

Coupling between nodes requires phase synchronization. Destructive interference
occurs when phases are misaligned (antiphase), while constructive interference
occurs when phases align. The coupling strength formula reflects this physics:

.. math::
    \\text{coupling_strength} = 1.0 - \\frac{|\\Delta\\phi|}{\\pi}

where Δφ is the phase difference in radians.

**Physical Interpretation:**

- Δφ = 0 (perfect alignment) → coupling = 1.0 (maximum constructive interference)
- Δφ = π/2 (orthogonal) → coupling = 0.5 (partial coupling)
- Δφ = π (antiphase) → coupling = 0.0 (destructive interference)

**TNFR Invariant #5:** "No coupling without explicit phase verification"
(see AGENTS.md). All coupling operations must verify phase compatibility
before propagating structural information.

Canonical Usage
---------------

**Operators Using This Module:**

1. **UM (Coupling)**: Phase synchronization and network formation
2. **RA (Resonance)**: Coherence propagation through phase-aligned paths
3. **THOL (Self-organization)**: Sub-EPI propagation to coupled neighbors

**Before Refactoring:**

Each operator implemented its own phase compatibility calculation, leading
to potential inconsistencies and maintenance burden.

**After Refactoring:**

All operators use the canonical functions defined here, ensuring theoretical
consistency and simplifying validation against TNFR physics.

Examples
--------

**Basic coupling strength calculation:**

>>> import math
>>> # Perfect alignment
>>> compute_phase_coupling_strength(0.0, 0.0)
1.0
>>> # Orthogonal phases
>>> compute_phase_coupling_strength(0.0, math.pi/2)
0.5
>>> # Antiphase (destructive)
>>> round(compute_phase_coupling_strength(0.0, math.pi), 10)
0.0

**Phase compatibility check:**

>>> # Check if phases are compatible for coupling
>>> is_phase_compatible(0.0, 0.1, threshold=0.5)
True
>>> is_phase_compatible(0.0, math.pi, threshold=0.5)
False

**Network phase alignment:**

>>> import networkx as nx
>>> from tnfr.constants.aliases import ALIAS_THETA
>>> G = nx.Graph()
>>> G.add_edges_from([(0, 1), (1, 2)])
>>> for i, theta in enumerate([0.0, 0.1, 0.2]):
...     G.nodes[i][ALIAS_THETA] = theta
>>> alignment = compute_network_phase_alignment(G, node=1, radius=1)
>>> 0.0 <= alignment <= 1.0
True

See Also
--------

operators.definitions : Operator implementations (UM, RA, THOL)
metrics.phase_coherence : Kuramoto order parameter and phase metrics
AGENTS.md : Invariant #5 - Phase Verification requirement
UNIFIED_GRAMMAR_RULES.md : U3 - RESONANT COUPLING grammar rule

References
----------

.. [1] TNFR.pdf § 2.3: Phase synchronization and coupling
.. [2] AGENTS.md: Invariant #5 - No coupling without phase verification
.. [3] UNIFIED_GRAMMAR_RULES.md: U3 - Resonant Coupling requires |φᵢ - φⱼ| ≤ Δφ_max
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..types import TNFRGraph, NodeId

from ..utils.numeric import angle_diff

__all__ = [
    "compute_phase_coupling_strength",
    "is_phase_compatible",
    "compute_network_phase_alignment",
]


def compute_phase_coupling_strength(
    theta_a: float,
    theta_b: float,
) -> float:
    """Compute canonical coupling strength from phase difference.

    This is the canonical TNFR formula for phase-based coupling strength,
    representing the degree of constructive vs. destructive interference
    between two oscillating nodes.

    Parameters
    ----------
    theta_a : float
        Phase of first node in radians [0, 2π)
    theta_b : float
        Phase of second node in radians [0, 2π)

    Returns
    -------
    float
        Coupling strength in [0, 1]:
        - 1.0: Perfect phase alignment (Δφ = 0)
        - 0.5: Orthogonal phases (Δφ = π/2)
        - 0.0: Antiphase (Δφ = π, destructive interference)

    Notes
    -----
    **Formula:**

    .. math::
        \\text{coupling_strength} = 1.0 - \\frac{|\\text{angle_diff}(\\theta_b, \\theta_a)|}{\\pi}

    The formula uses :func:`~tnfr.utils.numeric.angle_diff` to compute the
    shortest angular distance between phases, properly handling wrap-around
    at 2π boundaries.

    **Physics:**

    - Based on wave interference physics: aligned phases → constructive interference
    - Antiphase (Δφ = π) → destructive interference → zero coupling
    - Linear interpolation between extremes reflects gradual transition

    **Used By:**

    - UM (Coupling): For determining link formation and synchronization strength
    - RA (Resonance): For gating coherence propagation to neighbors
    - THOL (Self-organization): For sub-EPI propagation through coupled nodes

    **Invariant #5:** This function implements the explicit phase verification
    required by TNFR Invariant #5 (AGENTS.md). All coupling operations must
    verify phase compatibility before propagating structural information.

    Examples
    --------
    >>> import math
    >>> # Perfect alignment
    >>> compute_phase_coupling_strength(0.0, 0.0)
    1.0
    >>> # Small misalignment
    >>> compute_phase_coupling_strength(0.0, 0.1)  # doctest: +ELLIPSIS
    0.96...
    >>> # Orthogonal phases
    >>> compute_phase_coupling_strength(0.0, math.pi/2)
    0.5
    >>> # Antiphase (destructive)
    >>> round(compute_phase_coupling_strength(0.0, math.pi), 10)
    0.0
    >>> # Wrap-around handling
    >>> compute_phase_coupling_strength(0.1, 2*math.pi - 0.1)  # doctest: +ELLIPSIS
    0.93...

    See Also
    --------
    is_phase_compatible : Boolean compatibility check with threshold
    angle_diff : Shortest angular distance between phases
    """
    phase_diff = abs(angle_diff(theta_b, theta_a))
    return 1.0 - (phase_diff / math.pi)


def is_phase_compatible(
    theta_a: float,
    theta_b: float,
    threshold: float = 0.5,
) -> bool:
    """Check if two phases are compatible for coupling/propagation.

    Determines whether two nodes are sufficiently phase-aligned to support
    resonant coupling, based on a configurable coupling strength threshold.

    Parameters
    ----------
    theta_a : float
        Phase of first node in radians [0, 2π)
    theta_b : float
        Phase of second node in radians [0, 2π)
    threshold : float, default=0.5
        Minimum coupling strength required for compatibility [0, 1].
        Default 0.5 corresponds to maximum phase difference of π/2 (orthogonal).

    Returns
    -------
    bool
        True if coupling_strength >= threshold (nodes are compatible)
        False if coupling_strength < threshold (nodes are incompatible)

    Notes
    -----
    **Common Thresholds:**

    - 0.5 (default): Allows coupling up to π/2 phase difference
    - 0.7: More restrictive, requires Δφ < π/2.1 (~95°)
    - 0.9: Very restrictive, requires Δφ < π/10 (~18°)

    **Usage:**

    - **UM (Coupling)**: Gate link formation based on phase compatibility
    - **RA (Resonance)**: Filter neighbors for coherence propagation
    - **THOL propagation**: Minimum coupling for sub-EPI propagation

    **Invariant #5:** This function provides a boolean interface to the
    phase verification requirement (AGENTS.md Invariant #5).

    Examples
    --------
    >>> import math
    >>> # In-phase: compatible
    >>> is_phase_compatible(0.0, 0.1, threshold=0.5)
    True
    >>> # Orthogonal: at threshold boundary
    >>> is_phase_compatible(0.0, math.pi/2, threshold=0.5)
    True
    >>> # Slightly beyond orthogonal: incompatible
    >>> is_phase_compatible(0.0, math.pi/2 + 0.1, threshold=0.5)
    False
    >>> # Antiphase: incompatible
    >>> is_phase_compatible(0.0, math.pi, threshold=0.5)
    False
    >>> # Higher threshold: more restrictive
    >>> is_phase_compatible(0.0, math.pi/4, threshold=0.9)
    False
    >>> is_phase_compatible(0.0, 0.1, threshold=0.9)
    True

    See Also
    --------
    compute_phase_coupling_strength : Continuous coupling strength [0, 1]
    """
    coupling = compute_phase_coupling_strength(theta_a, theta_b)
    return coupling >= threshold


def compute_network_phase_alignment(
    G: TNFRGraph,
    node: NodeId,
    radius: int = 1,
) -> float:
    """Compute phase alignment in local neighborhood using Kuramoto order parameter.

    This is a convenience wrapper around the existing
    :func:`~tnfr.metrics.phase_coherence.compute_phase_alignment` function,
    provided for API consistency within this module.

    Parameters
    ----------
    G : TNFRGraph
        TNFR network graph containing nodes with phase (theta) attributes
    node : NodeId
        Central node for neighborhood analysis
    radius : int, default=1
        Neighborhood radius in hops from central node

    Returns
    -------
    float
        Phase alignment quality in [0, 1]:
        - 1.0: Perfect phase synchronization (all nodes aligned)
        - 0.0: Complete phase disorder (random phases)

    Notes
    -----
    **Kuramoto Order Parameter:**

    Measures collective phase synchrony using:

    .. math::
        r = |\\frac{1}{N} \\sum_{j=1}^{N} e^{i\\theta_j}|

    **Used By:**

    - **RA (Resonance)**: Assess network coherence for propagation gating
    - **IL (Coherence)**: Validate phase locking effectiveness

    **Implementation:**

    This function delegates to the existing implementation in
    :mod:`tnfr.metrics.phase_coherence` to avoid code duplication
    while providing a unified API for phase compatibility calculations.

    Examples
    --------
    >>> import networkx as nx
    >>> from tnfr.constants.aliases import ALIAS_THETA
    >>> G = nx.Graph()
    >>> G.add_edges_from([(0, 1), (1, 2), (2, 3)])
    >>> # Highly aligned phases
    >>> for i in range(4):
    ...     G.nodes[i][ALIAS_THETA] = i * 0.1
    >>> alignment = compute_network_phase_alignment(G, node=1, radius=1)
    >>> alignment > 0.9  # High alignment
    True
    >>> # Random phases
    >>> import math
    >>> G.nodes[0][ALIAS_THETA] = 0.0
    >>> G.nodes[1][ALIAS_THETA] = math.pi/3
    >>> G.nodes[2][ALIAS_THETA] = 2*math.pi/3
    >>> G.nodes[3][ALIAS_THETA] = math.pi
    >>> alignment = compute_network_phase_alignment(G, node=1, radius=1)
    >>> 0.0 <= alignment <= 1.0
    True

    See Also
    --------
    metrics.phase_coherence.compute_phase_alignment : Underlying implementation
    metrics.phase_coherence.compute_global_phase_coherence : Global network metric
    """
    # Import existing function to avoid duplication
    from .phase_coherence import compute_phase_alignment

    return compute_phase_alignment(G, node, radius=radius)
