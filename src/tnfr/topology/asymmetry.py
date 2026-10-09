"""Static ego-network degree and clustering heterogeneity.

This configured score describes supplied graph topology. A single snapshot
cannot attribute that topology to an operator or establish symmetry breaking.
The native OZ pressure transformation does not itself modify support.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..types import NodeId, TNFRGraph

__all__ = [
    "compute_topological_asymmetry",
]


def compute_topological_asymmetry(G: "TNFRGraph", node: "NodeId") -> float:
    """Return a configured static heterogeneity score for the 1-hop ego graph.

    The score is ``clip(0.6 * degree_cv + 0.4 * clustering_cv, 0, 1)``, using
    population standard deviations divided by their corresponding means.
    Near-zero means use the implementation cutoff 1e-10. Neighborhoods with
    at most two nodes return zero by policy, not by a proof of graph symmetry.
    A missing center raises the graph's support error rather than reporting zero.

    Degree and clustering heterogeneity need not characterize the automorphism
    group. Zero score means these measured distributions are homogeneous under
    the selected policy; a positive score is neither an observed change nor a
    certificate of successful structural disruption. Temporal differences require
    an independently captured preceding score on the intended support.

    This reader preserves the graph. ``dissonance_metrics`` reuses it for the
    explicit before/after observations, without inferring a preceding zero.
    """
    import networkx as nx

    from ..mathematics.unified_numerical import np

    # Extract 1-hop ego graph (node + immediate neighbors)
    ego = nx.ego_graph(G, node, radius=1)

    n_nodes = ego.number_of_nodes()

    if n_nodes <= 2:
        # Too small for meaningful asymmetry
        # The selected diagnostic assigns zero to these small neighborhoods.
        return 0.0

    # Compute degree heterogeneity in ego-graph
    degrees = [ego.degree(n) for n in ego.nodes()]

    if not degrees or all(d == 0 for d in degrees):
        # No degree variation is available for this diagnostic.
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
