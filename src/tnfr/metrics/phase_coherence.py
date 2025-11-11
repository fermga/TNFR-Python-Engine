"""Phase coherence metrics for TNFR networks.

This module provides phase alignment and synchronization metrics based on
circular statistics and the Kuramoto order parameter. These metrics are
essential for measuring the effectiveness of the IL (Coherence) operator's
phase locking mechanism.

Mathematical Foundation
-----------------------

**Kuramoto Order Parameter:**

The phase alignment quality is measured using the Kuramoto order parameter r:

.. math::
    r = |\\frac{1}{N} \\sum_{j=1}^{N} e^{i\\theta_j}|

where:
- r ∈ [0, 1]
- r = 1: Perfect phase synchrony (all nodes aligned)
- r = 0: Complete phase disorder (uniformly distributed phases)
- θ_j: Phase of node j in radians

**Circular Mean:**

The mean phase of a set of angles is computed using the circular mean to
properly handle phase wrap-around at 2π:

.. math::
    \\theta_{mean} = \\text{arg}\\left(\\frac{1}{N} \\sum_{j=1}^{N} e^{i\\theta_j}\\right)

This ensures that phases near 0 and 2π are correctly averaged (e.g., 0.1 and
6.2 radians average to near 0, not π).

TNFR Context
------------

Phase alignment is a key component of the IL (Coherence) operator:

- **IL Phase Locking**: θ_node → θ_node + α * (θ_network - θ_node)
- **Network Synchrony**: High r indicates effective IL application
- **Local vs. Global**: Phase alignment can be measured at node or network level
- **Structural Traceability**: Phase metrics enable telemetry of synchronization

Examples
--------

**Compute phase alignment for a node:**

>>> import networkx as nx
>>> from tnfr.metrics.phase_coherence import compute_phase_alignment
>>> from tnfr.constants import THETA_PRIMARY
>>> G = nx.Graph()
>>> G.add_edges_from([(1, 2), (2, 3)])
>>> G.nodes[1][THETA_PRIMARY] = 0.0
>>> G.nodes[2][THETA_PRIMARY] = 0.1
>>> G.nodes[3][THETA_PRIMARY] = 0.2
>>> alignment = compute_phase_alignment(G, 2, radius=1)
>>> 0.0 <= alignment <= 1.0
True

**Compute global phase coherence:**

>>> from tnfr.metrics.phase_coherence import compute_global_phase_coherence
>>> coherence = compute_global_phase_coherence(G)
>>> 0.0 <= coherence <= 1.0
True

See Also
--------

operators.definitions.Coherence : IL operator that applies phase locking
metrics.coherence.compute_global_coherence : Global structural coherence C(t)
observers.kuramoto_order : Alternative Kuramoto order parameter implementation
"""

from __future__ import annotations

import cmath
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..types import TNFRGraph

from ..alias import get_attr
from ..constants.aliases import ALIAS_THETA
from ..utils import get_numpy

__all__ = [
    "compute_phase_alignment",
    "compute_global_phase_coherence",
]


def compute_phase_alignment(G: TNFRGraph, node: Any, radius: int = 1) -> float:
    """Compute phase alignment quality for node and neighborhood.

    Uses Kuramoto order parameter r = |⟨e^(iθ)⟩| to measure phase synchrony
    within a node's neighborhood. Higher values indicate better phase alignment,
    which is the goal of IL (Coherence) phase locking.

    Parameters
    ----------
    G : TNFRGraph
        Network graph with node phase attributes (θ)
    node : Any
        Central node for local phase alignment computation
    radius : int, default=1
        Neighborhood radius:
        - 1 = node + immediate neighbors (default)
        - 2 = node + neighbors + neighbors-of-neighbors
        - etc.

    Returns
    -------
    float
        Phase alignment in [0, 1] where:
        - 1.0 = Perfect phase synchrony (all phases aligned)
        - 0.0 = Complete phase disorder (uniformly distributed)

    Notes
    -----
    **Mathematical Foundation:**

    Kuramoto order parameter for local neighborhood:

    .. math::
        r = |\\frac{1}{N} \\sum_{j \\in \\mathcal{N}(i)} e^{i\\theta_j}|

    where 𝒩(i) is the set of neighbors within `radius` of node i (including i).

    **Use Cases:**

    - **IL Effectiveness**: Measure phase locking success after IL application
    - **Synchrony Monitoring**: Track local phase coherence over time
    - **Hotspot Detection**: Identify regions with poor phase alignment
    - **Coupling Validation**: Verify phase prerequisites before UM (Coupling)

    **Special Cases:**

    - Isolated node (no neighbors): Returns 1.0 (trivially synchronized)
    - Single neighbor: Returns 1.0 (two nodes always "aligned")
    - Empty neighborhood: Returns 1.0 (no disorder by definition)

    **TNFR Context:**

    Phase alignment is a precondition for effective coupling (UM operator) and
    resonance (RA operator). The IL operator increases phase alignment through
    its phase locking mechanism: θ_node → θ_node + α * (θ_network - θ_node).

    See Also
    --------
    compute_global_phase_coherence : Network-wide phase coherence
    operators.definitions.Coherence : IL operator with phase locking

    Examples
    --------
    >>> import networkx as nx
    >>> from tnfr.metrics.phase_coherence import compute_phase_alignment
    >>> from tnfr.constants import THETA_PRIMARY
    >>> G = nx.Graph()
    >>> G.add_edges_from([(1, 2), (2, 3), (3, 4)])
    >>> # Highly aligned phases
    >>> for n in [1, 2, 3, 4]:
    ...     G.nodes[n][THETA_PRIMARY] = 0.1 * n  # Small differences
    >>> r = compute_phase_alignment(G, node=2, radius=1)
    >>> r > 0.9  # Should be highly aligned
    True
    >>> # Disordered phases
    >>> import numpy as np
    >>> for n in [1, 2, 3, 4]:
    ...     G.nodes[n][THETA_PRIMARY] = np.random.uniform(0, 2*np.pi)
    >>> r = compute_phase_alignment(G, node=2, radius=1)
    >>> 0.0 <= r <= 1.0  # Could be anywhere in range
    True
    """
    import networkx as nx

    # Get neighborhood
    if radius == 1:
        neighbors = set(G.neighbors(node)) | {node}
    else:
        try:
            neighbors = set(nx.single_source_shortest_path_length(G, node, cutoff=radius).keys())
        except (nx.NetworkXError, KeyError):
            # Node not in graph or graph is empty
            neighbors = {node} if node in G.nodes else set()

    # Collect phases from neighborhood
    phases = []
    for n in neighbors:
        try:
            theta = float(get_attr(G.nodes[n], ALIAS_THETA, 0.0))
            phases.append(theta)
        except (KeyError, ValueError, TypeError):
            # Skip nodes with invalid phase data
            continue

    # Handle edge cases
    if not phases:
        return 1.0  # Empty neighborhood: trivially synchronized

    if len(phases) == 1:
        return 1.0  # Single node: perfect synchrony

    # Compute Kuramoto order parameter using circular statistics
    np = get_numpy()

    if np is not None:
        # NumPy vectorized computation
        phases_array = np.array(phases)
        complex_phases = np.exp(1j * phases_array)
        mean_complex = np.mean(complex_phases)
        r = np.abs(mean_complex)
        return float(r)
    else:
        # Pure Python fallback

        # Convert phases to complex exponentials
        complex_phases = [cmath.exp(1j * theta) for theta in phases]

        # Compute mean complex phasor
        mean_real = sum(z.real for z in complex_phases) / len(complex_phases)
        mean_imag = sum(z.imag for z in complex_phases) / len(complex_phases)
        mean_complex = complex(mean_real, mean_imag)

        # Kuramoto order parameter is magnitude of mean phasor
        r = abs(mean_complex)
        return float(r)


def compute_global_phase_coherence(G: TNFRGraph) -> float:
    """Compute global phase coherence across entire network.

    Measures network-wide phase synchronization using the Kuramoto order
    parameter applied to all nodes. This is the global analog of
    compute_phase_alignment and indicates overall phase alignment quality.

    Parameters
    ----------
    G : TNFRGraph
        Network graph with node phase attributes (θ)

    Returns
    -------
    float
        Global phase coherence in [0, 1] where:
        - 1.0 = Perfect network-wide phase synchrony
        - 0.0 = Complete phase disorder across network

    Notes
    -----
    **Mathematical Foundation:**

    Global Kuramoto order parameter:

    .. math::
        r_{global} = |\\frac{1}{N} \\sum_{j=1}^{N} e^{i\\theta_j}|

    where N is the total number of nodes in the network.

    **Use Cases:**

    - **IL Effectiveness**: Measure global impact of IL phase locking
    - **Network Health**: Monitor overall synchronization state
    - **Convergence Tracking**: Verify phase alignment over time
    - **Bifurcation Detection**: Low r_global may indicate impending split

    **Special Cases:**

    - Empty network: Returns 1.0 (no disorder by definition)
    - Single node: Returns 1.0 (trivially synchronized)
    - All phases = 0: Returns 1.0 (perfect alignment)

    **TNFR Context:**

    Global phase coherence is a key metric for network structural health.
    Repeated IL application should increase r_global as nodes synchronize
    their phases. Combined with C(t) (structural coherence), r_global provides
    a complete picture of network stability.

    See Also
    --------
    compute_phase_alignment : Local phase alignment for node neighborhoods
    metrics.coherence.compute_global_coherence : Global structural coherence C(t)
    observers.kuramoto_order : Alternative Kuramoto implementation

    Examples
    --------
    >>> import networkx as nx
    >>> from tnfr.metrics.phase_coherence import compute_global_phase_coherence
    >>> from tnfr.constants import THETA_PRIMARY
    >>> G = nx.Graph()
    >>> G.add_nodes_from([1, 2, 3, 4])
    >>> # Aligned network
    >>> for n in [1, 2, 3, 4]:
    ...     G.nodes[n][THETA_PRIMARY] = 0.5  # All same phase
    >>> r = compute_global_phase_coherence(G)
    >>> r == 1.0  # Perfect alignment
    True
    >>> # Disordered network
    >>> import numpy as np
    >>> for n in [1, 2, 3, 4]:
    ...     G.nodes[n][THETA_PRIMARY] = np.random.uniform(0, 2*np.pi)
    >>> r = compute_global_phase_coherence(G)
    >>> 0.0 <= r <= 1.0
    True
    """
    # Collect all node phases
    phases = []
    for n in G.nodes():
        try:
            theta = float(get_attr(G.nodes[n], ALIAS_THETA, 0.0))
            phases.append(theta)
        except (KeyError, ValueError, TypeError):
            # Skip nodes with invalid phase data
            continue

    # Handle edge cases
    if not phases:
        return 1.0  # Empty network: trivially synchronized

    if len(phases) == 1:
        return 1.0  # Single node: perfect synchrony

    # Compute Kuramoto order parameter using circular statistics
    np = get_numpy()

    if np is not None:
        # NumPy vectorized computation
        phases_array = np.array(phases)
        complex_phases = np.exp(1j * phases_array)
        mean_complex = np.mean(complex_phases)
        r = np.abs(mean_complex)
        return float(r)
    else:
        # Pure Python fallback

        # Convert phases to complex exponentials
        complex_phases = [cmath.exp(1j * theta) for theta in phases]

        # Compute mean complex phasor
        mean_real = sum(z.real for z in complex_phases) / len(complex_phases)
        mean_imag = sum(z.imag for z in complex_phases) / len(complex_phases)
        mean_complex = complex(mean_real, mean_imag)

        # Kuramoto order parameter is magnitude of mean phasor
        r = abs(mean_complex)
        return float(r)
