"""Topological asymmetry analysis for TNFR networks.

This module provides functions to measure local topological asymmetry around
nodes in a TNFR network. According to TNFR canonical theory, the OZ (Dissonance)
operator introduces topological disruption that breaks structural symmetry.

The asymmetry measure quantifies this symmetry breaking by analyzing the
heterogeneity of the node's ego-network (the node and its immediate neighbors).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..types import NodeId, TNFRGraph

__all__ = [
    "compute_topological_asymmetry",
]


def compute_topological_asymmetry(G: "TNFRGraph", node: "NodeId") -> float:
    """Measure local topological asymmetry around node.

    Uses ego-graph analysis to detect structural symmetry breaking introduced
    by dissonance operators. According to TNFR canonical theory (§2.3.3, R4),
    OZ (Dissonance) introduces **topological disruption**, not just numerical
    instability.

    The asymmetry is computed by analyzing the heterogeneity of degree and
    clustering distributions in the node's 1-hop neighborhood (ego-graph).
    Higher asymmetry indicates successful structural disruption.

    Parameters
    ----------
    G : TNFRGraph
        Graph containing the node
    node : NodeId
        Node identifier to analyze

    Returns
    -------
    float
        Topological asymmetry measure, range [0.0, 1.0]:
        - 0.0 = perfect local symmetry (homogeneous neighborhood)
        - 1.0 = maximal asymmetry (heterogeneous structure)

    Notes
    -----
    The asymmetry measure combines two components:

    1. **Degree heterogeneity**: Coefficient of variation (CV) of node degrees
       in the ego-graph. Measures structural connectivity variation.

    2. **Clustering heterogeneity**: CV of local clustering coefficients.
       Measures variation in local cohesion patterns.

    The final asymmetry score is a weighted combination:
        asymmetry = 0.6 * degree_cv + 0.4 * clustering_cv

    For isolated nodes or very small neighborhoods (≤2 nodes), returns 0.0
    as there is insufficient structure for meaningful asymmetry measurement.

    Examples
    --------
    >>> from tnfr.structural import create_nfr
    >>> from tnfr.operators.definitions import Dissonance
    >>> from tnfr.topology import compute_topological_asymmetry
    >>>
    >>> G, node = create_nfr("test", epi=0.5, vf=1.0)
    >>> # Add neighbors to create network structure
    >>> for i in range(4):
    ...     neighbor = f"n{i}"
    ...     G.add_node(neighbor)
    ...     G.add_edge(node, neighbor)
    >>>
    >>> # Measure asymmetry before dissonance
    >>> asym_before = compute_topological_asymmetry(G, node)
    >>>
    >>> # Apply dissonance operator
    >>> Dissonance()(G, node)
    >>>
    >>> # Measure asymmetry after dissonance
    >>> asym_after = compute_topological_asymmetry(G, node)
    >>>
    >>> # Dissonance should increase asymmetry (topological disruption)
    >>> assert asym_after >= asym_before

    See Also
    --------
    tnfr.operators.definitions.Dissonance : OZ operator that introduces dissonance
    tnfr.operators.metrics.dissonance_metrics : Collects asymmetry in metrics
    """
    import networkx as nx

    from ..utils import get_numpy

    np = get_numpy()

    # Extract 1-hop ego graph (node + immediate neighbors)
    try:
        ego = nx.ego_graph(G, node, radius=1)
    except nx.NetworkXError:
        return 0.0

    n_nodes = ego.number_of_nodes()

    if n_nodes <= 2:
        # Too small for meaningful asymmetry
        # Isolated node (n=1) or single connection (n=2) are trivially symmetric
        return 0.0

    # Compute degree heterogeneity in ego-graph
    degrees = [ego.degree(n) for n in ego.nodes()]

    if not degrees or all(d == 0 for d in degrees):
        # No edges in ego graph - symmetric by definition
        return 0.0

    degrees_arr = np.array(degrees, dtype=float)
    mean_degree = np.mean(degrees_arr)

    if mean_degree < 1e-10:
        degree_cv = 0.0
    else:
        std_degree = np.std(degrees_arr)
        degree_cv = std_degree / mean_degree

    # Compute clustering heterogeneity in ego-graph
    try:
        clustering = [nx.clustering(ego, n) for n in ego.nodes()]
    except (ZeroDivisionError, nx.NetworkXError):
        # If clustering computation fails, use only degree asymmetry
        clustering = [0.0] * n_nodes

    clustering_arr = np.array(clustering, dtype=float)
    mean_clustering = np.mean(clustering_arr)

    if mean_clustering < 1e-10:
        clustering_cv = 0.0
    else:
        std_clustering = np.std(clustering_arr)
        clustering_cv = std_clustering / mean_clustering

    # Combined asymmetry score (weighted)
    # Degree asymmetry is primary (60%), clustering is secondary (40%)
    asymmetry = 0.6 * degree_cv + 0.4 * clustering_cv

    # Clip to [0, 1] range
    return float(np.clip(asymmetry, 0.0, 1.0))
