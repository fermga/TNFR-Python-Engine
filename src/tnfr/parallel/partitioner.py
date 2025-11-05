"""TNFR-aware network partitioning for parallel computation.

Partitions networks respecting structural coherence rather than classical graph
metrics. Communities are grown based on phase synchrony and frequency alignment
to preserve the fractal organization inherent in TNFR.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, List, Set, Tuple

if TYPE_CHECKING:  # pragma: no cover
    from ..types import TNFRGraph

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    np = None  # type: ignore
    HAS_NUMPY = False

from ..alias import get_attr
from ..constants.aliases import ALIAS_THETA, ALIAS_VF


class FractalPartitioner:
    """Partitions TNFR networks respecting structural coherence.

    This partitioner detects communities based on TNFR metrics (frequency and
    phase) rather than classical graph metrics. It ensures that nodes with
    similar structural frequencies and synchronized phases are grouped together,
    preserving operational fractality during parallel processing.

    Parameters
    ----------
    max_partition_size : int, default=100
        Maximum number of nodes per partition. Larger partitions reduce
        communication overhead but may limit parallelism.
    coherence_threshold : float, default=0.3
        Minimum coherence score for adding a node to a community. Higher values
        create tighter communities but may result in more partitions.

    Examples
    --------
    >>> import networkx as nx
    >>> from tnfr.parallel import FractalPartitioner
    >>> G = nx.Graph()
    >>> G.add_edges_from([("a", "b"), ("b", "c")])
    >>> for node in G.nodes():
    ...     G.nodes[node]["vf"] = 1.0
    ...     G.nodes[node]["phase"] = 0.0
    >>> partitioner = FractalPartitioner(max_partition_size=50)
    >>> partitions = partitioner.partition_network(G)
    >>> len(partitions) >= 1
    True
    """

    def __init__(
        self,
        max_partition_size: int = 100,
        coherence_threshold: float = 0.3,
    ):
        self.max_partition_size = max_partition_size
        self.coherence_threshold = coherence_threshold

    def partition_network(
        self, graph: TNFRGraph
    ) -> List[Tuple[Set[Any], TNFRGraph]]:
        """Partition network into coherent subgraphs.

        Parameters
        ----------
        graph : TNFRGraph
            TNFR network to partition. Nodes must have 'vf' and 'phase' attrs.

        Returns
        -------
        List[Tuple[Set[Any], TNFRGraph]]
            List of (node_set, subgraph) tuples for parallel processing.

        Notes
        -----
        Maintains TNFR structural invariants:
        - Communities formed by resonance (not just topology)
        - Phase coherence preserved within partitions
        - Frequency alignment respected
        """
        import networkx as nx

        if len(graph) == 0:
            return []

        # Detect TNFR communities
        communities = self._detect_tnfr_communities(graph)

        # Create balanced partitions
        partitions = []
        current_partition = set()

        for community in communities:
            if len(current_partition) + len(community) <= self.max_partition_size:
                current_partition.update(community)
            else:
                if current_partition:
                    subgraph = graph.subgraph(current_partition).copy()
                    partitions.append((current_partition.copy(), subgraph))
                current_partition = community.copy()

        # Add final partition
        if current_partition:
            subgraph = graph.subgraph(current_partition).copy()
            partitions.append((current_partition, subgraph))

        return partitions

    def _detect_tnfr_communities(self, graph: TNFRGraph) -> List[Set[Any]]:
        """Detect communities using TNFR coherence metrics.

        Uses structural frequency and phase to grow coherent communities rather
        than classical modularity or betweenness metrics.
        """
        communities = []
        unprocessed = set(graph.nodes())

        while unprocessed:
            # Select seed node
            seed = next(iter(unprocessed))
            community = self._grow_coherent_community(graph, seed, unprocessed)
            communities.append(community)
            unprocessed -= community

        return communities

    def _grow_coherent_community(
        self, graph: TNFRGraph, seed: Any, available: Set[Any]
    ) -> Set[Any]:
        """Grow community from seed based on structural coherence.

        Parameters
        ----------
        graph : TNFRGraph
            Full network graph
        seed : Any
            Starting node for community growth
        available : Set[Any]
            Nodes that haven't been assigned to communities yet

        Returns
        -------
        Set[Any]
            Set of nodes forming a coherent community
        """
        community = {seed}
        neighbors = graph.neighbors(seed)
        candidates = set(neighbors) & available

        while candidates:
            # Find most coherent candidate
            best_candidate = None
            best_coherence = -1.0

            for candidate in candidates:
                coherence = self._compute_community_coherence(
                    graph, community, candidate
                )
                if coherence > best_coherence:
                    best_coherence = coherence
                    best_candidate = candidate

            # Add if above threshold
            if best_coherence > self.coherence_threshold:
                community.add(best_candidate)
                candidates.remove(best_candidate)
                # Add new neighbors as candidates
                new_neighbors = set(graph.neighbors(best_candidate)) & available
                candidates.update(new_neighbors - community)
            else:
                break  # No more coherent candidates

        return community

    def _compute_community_coherence(
        self, graph: TNFRGraph, community: Set[Any], candidate: Any
    ) -> float:
        """Compute coherence between candidate and existing community.

        Uses TNFR metrics: frequency alignment (νf) and phase synchrony.

        Parameters
        ----------
        graph : TNFRGraph
            Network graph
        community : Set[Any]
            Existing community nodes
        candidate : Any
            Candidate node to evaluate

        Returns
        -------
        float
            Coherence score in [0, 1], where higher means better alignment
        """
        if not community:
            return 0.0

        # Try to get values using TNFR aliases or direct access
        candidate_vf = float(get_attr(graph.nodes[candidate], ALIAS_VF, None) or 
                           graph.nodes[candidate].get("vf", 1.0))
        candidate_phase = float(get_attr(graph.nodes[candidate], ALIAS_THETA, None) or 
                              graph.nodes[candidate].get("phase", 0.0))

        coherences = []
        for member in community:
            member_vf = float(get_attr(graph.nodes[member], ALIAS_VF, None) or 
                            graph.nodes[member].get("vf", 1.0))
            member_phase = float(get_attr(graph.nodes[member], ALIAS_THETA, None) or 
                               graph.nodes[member].get("phase", 0.0))

            # Frequency coherence: inversely proportional to difference
            vf_diff = abs(candidate_vf - member_vf)
            vf_coherence = 1.0 / (1.0 + vf_diff)

            # Phase coherence: cosine of phase difference
            phase_diff = candidate_phase - member_phase
            if HAS_NUMPY:
                phase_coherence = float(np.cos(phase_diff))
            else:
                phase_coherence = math.cos(phase_diff)

            # Weighted combination: prioritize frequency alignment
            coherences.append(0.6 * vf_coherence + 0.4 * phase_coherence)

        return sum(coherences) / len(coherences) if coherences else 0.0
