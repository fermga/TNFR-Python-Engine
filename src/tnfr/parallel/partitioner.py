"""TNFR-aware network partitioning for parallel computation.

Partitions networks respecting structural coherence rather than classical graph
metrics. Communities are grown based on phase synchrony and frequency alignment
to preserve the fractal organization inherent in TNFR.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, List, Optional, Set, Tuple

if TYPE_CHECKING:  # pragma: no cover
    from ..types import TNFRGraph

try:
    import numpy as np

    HAS_NUMPY = True
except ImportError:
    np = None  # type: ignore
    HAS_NUMPY = False

try:
    from scipy.spatial import KDTree

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    KDTree = None  # type: ignore

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
        communication overhead but may limit parallelism. If None, uses
        adaptive partitioning based on network density.
    coherence_threshold : float, default=0.3
        Minimum coherence score for adding a node to a community. Higher values
        create tighter communities but may result in more partitions.
    use_spatial_index : bool, default=True
        Whether to use spatial indexing (KDTree) for O(n log n) neighbor
        finding. Requires scipy. Falls back to O(n²) if unavailable.
    adaptive : bool, default=True
        Whether to use adaptive partitioning that adjusts partition size
        based on network density and clustering coefficient.

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

    Notes
    -----
    Spatial indexing provides O(n log n) complexity for large networks
    compared to O(n²) without it. Adaptive partitioning automatically
    adjusts partition size based on network characteristics.
    """

    def __init__(
        self,
        max_partition_size: Optional[int] = 100,
        coherence_threshold: float = 0.3,
        use_spatial_index: bool = True,
        adaptive: bool = True,
    ):
        self.max_partition_size = max_partition_size
        self.coherence_threshold = coherence_threshold
        self.use_spatial_index = use_spatial_index and HAS_SCIPY and HAS_NUMPY
        self.adaptive = adaptive
        self._kdtree = None
        self._node_index_map = None

    def partition_network(self, graph: TNFRGraph) -> List[Tuple[Set[Any], TNFRGraph]]:
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

        Uses spatial indexing for O(n log n) complexity when available.
        Adapts partition size based on network density when adaptive=True.
        """

        if len(graph) == 0:
            return []

        # Determine optimal partition size adaptively
        if self.adaptive:
            partition_size = self._compute_adaptive_partition_size(graph)
        else:
            partition_size = self.max_partition_size or 100

        # Build spatial index if requested and available
        if self.use_spatial_index:
            self._build_spatial_index(graph)

        # Detect TNFR communities
        communities = self._detect_tnfr_communities(graph)

        # Create balanced partitions
        partitions = []
        current_partition = set()

        for community in communities:
            if len(current_partition) + len(community) <= partition_size:
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

        # Clean up spatial index
        self._kdtree = None
        self._node_index_map = None

        return partitions

    def _compute_adaptive_partition_size(self, graph: TNFRGraph) -> int:
        """Compute optimal partition size based on network characteristics.

        Adapts partition size based on:
        - Network density (sparse vs dense)
        - Clustering coefficient (community structure)
        - Total network size

        Returns
        -------
        int
            Recommended partition size for this network
        """
        import networkx as nx

        n_nodes = len(graph)

        # Base size from configuration or defaults
        if self.max_partition_size:
            base_size = self.max_partition_size
        else:
            # Default adaptive sizing
            if n_nodes < 100:
                base_size = n_nodes  # Don't partition small networks
            elif n_nodes < 1000:
                base_size = 100
            else:
                base_size = 200

        # Adjust based on density
        density = nx.density(graph)

        if density > 0.5:
            # Dense networks: smaller partitions reduce communication overhead
            size_multiplier = 0.5
        elif density > 0.1:
            # Medium density: balanced partitioning
            size_multiplier = 1.0
        else:
            # Sparse networks: larger partitions okay
            size_multiplier = 1.5

        # Adjust based on clustering
        try:
            avg_clustering = nx.average_clustering(graph)
            if avg_clustering > 0.6:
                # High clustering: communities are well-defined, can use smaller partitions
                size_multiplier *= 0.8
            elif avg_clustering < 0.2:
                # Low clustering: use larger partitions
                size_multiplier *= 1.2
        except (AttributeError, ZeroDivisionError, ValueError, TypeError):
            # If clustering calculation fails, skip adjustment
            pass

        adapted_size = int(base_size * size_multiplier)
        # Ensure reasonable bounds
        return max(10, min(adapted_size, 500))

    def _build_spatial_index(self, graph: TNFRGraph) -> None:
        """Build KDTree spatial index for O(n log n) neighbor finding.

        Constructs a 2D spatial index using (νf, phase) coordinates
        to enable fast nearest-neighbor queries.
        """
        if not HAS_SCIPY or not HAS_NUMPY:
            return

        nodes = list(graph.nodes())
        if len(nodes) == 0:
            return

        # Extract νf and phase coordinates
        def _get_node_attr(node_id: Any, alias: tuple, fallback_key: str, default: float) -> float:
            """Get node attribute via TNFR alias or direct access."""
            return float(
                get_attr(graph.nodes[node_id], alias, None)
                or graph.nodes[node_id].get(fallback_key, default)
            )

        coords = np.array(
            [
                [
                    _get_node_attr(node, ALIAS_VF, "vf", 1.0),
                    _get_node_attr(node, ALIAS_THETA, "phase", 0.0),
                ]
                for node in nodes
            ]
        )

        # Normalize coordinates for better distance metrics
        # νf: normalize by mean
        if coords[:, 0].std() > 0:
            coords[:, 0] = (coords[:, 0] - coords[:, 0].mean()) / coords[:, 0].std()

        # phase: wrap to [-π, π] for periodicity
        coords[:, 1] = np.arctan2(np.sin(coords[:, 1]), np.cos(coords[:, 1]))

        # Build KDTree
        self._kdtree = KDTree(coords)
        self._node_index_map = {i: node for i, node in enumerate(nodes)}

    def _find_coherent_neighbors_spatial(
        self, graph: TNFRGraph, seed: Any, available: Set[Any], k: int = 20
    ) -> List[Any]:
        """Find k nearest coherent neighbors using spatial index.

        Uses KDTree for O(log n) nearest neighbor finding instead of O(n).

        Parameters
        ----------
        graph : TNFRGraph
            Network graph
        seed : Any
            Seed node
        available : Set[Any]
            Available nodes to consider
        k : int
            Number of nearest neighbors to find

        Returns
        -------
        List[Any]
            List of up to k nearest coherent neighbors
        """
        if self._kdtree is None or self._node_index_map is None:
            # Fallback to graph neighbors
            return list(set(graph.neighbors(seed)) & available)

        # Find seed index
        seed_idx = None
        for idx, node in self._node_index_map.items():
            if node == seed:
                seed_idx = idx
                break

        if seed_idx is None:
            return []

        # Query k nearest neighbors (k+1 to exclude seed itself)
        distances, indices = self._kdtree.query(
            self._kdtree.data[seed_idx], k=min(k + 1, len(self._node_index_map))
        )

        # Filter to available nodes and exclude seed
        neighbors = []
        for idx in indices:
            if idx == seed_idx:
                continue
            node = self._node_index_map[idx]
            if node in available:
                neighbors.append(node)

        return neighbors

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

        Notes
        -----
        Uses spatial indexing for O(log n) neighbor finding when available,
        falling back to O(n) graph neighbors otherwise.
        """
        community = {seed}

        # Use spatial index if available for faster neighbor finding
        if self.use_spatial_index and self._kdtree is not None:
            candidates = set(self._find_coherent_neighbors_spatial(graph, seed, available, k=50))
        else:
            neighbors = graph.neighbors(seed)
            candidates = set(neighbors) & available

        while candidates:
            # Find most coherent candidate
            best_candidate = None
            best_coherence = -1.0

            for candidate in candidates:
                coherence = self._compute_community_coherence(graph, community, candidate)
                if coherence > best_coherence:
                    best_coherence = coherence
                    best_candidate = candidate

            # Add if above threshold
            if best_coherence > self.coherence_threshold:
                community.add(best_candidate)
                candidates.remove(best_candidate)

                # Add new neighbors as candidates
                if self.use_spatial_index and self._kdtree is not None:
                    new_neighbors = set(
                        self._find_coherent_neighbors_spatial(
                            graph, best_candidate, available, k=50
                        )
                    )
                else:
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

        def _get_node_attr(node_id: Any, alias: tuple, fallback_key: str, default: float) -> float:
            """Get node attribute via TNFR alias or direct access."""
            return float(
                get_attr(graph.nodes[node_id], alias, None)
                or graph.nodes[node_id].get(fallback_key, default)
            )

        candidate_vf = _get_node_attr(candidate, ALIAS_VF, "vf", 1.0)
        candidate_phase = _get_node_attr(candidate, ALIAS_THETA, "phase", 0.0)

        coherences = []
        for member in community:
            member_vf = _get_node_attr(member, ALIAS_VF, "vf", 1.0)
            member_phase = _get_node_attr(member, ALIAS_THETA, "phase", 0.0)

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
