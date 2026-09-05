"""TNFR-aware network partitioning for parallel computation.

Partitions networks respecting structural coherence rather than classical graph
metrics. Communities are grown based on phase synchrony and frequency alignment
to preserve the fractal organization inherent in TNFR.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    from ..types import TNFRGraph

from ..mathematics.unified_numerical import NUMPY_AVAILABLE as HAS_NUMPY
from ..mathematics.unified_numerical import np

try:
    from scipy.spatial import KDTree

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    KDTree = None  # type: ignore

from ..alias import get_attr
from ..constants.aliases import ALIAS_THETA, ALIAS_VF

# ---------------------------------------------------------------------------
# Network density / clustering thresholds for partition sizing
# ---------------------------------------------------------------------------
_DENSITY_DENSE_THRESHOLD = 0.5
_DENSITY_MEDIUM_THRESHOLD = 0.1
_CLUSTERING_HIGH_THRESHOLD = 0.6
_CLUSTERING_LOW_THRESHOLD = 0.2


def _node_attribute(graph: Any, node: Any, aliases: tuple, default: float) -> float:
    """Resolve structural aliases without treating a physical zero as absent."""
    value = get_attr(graph.nodes[node], aliases, None)
    return float(default if value is None else value)


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
        max_partition_size: int | None = 100,
        coherence_threshold: float = 0.3,
        use_spatial_index: bool = True,
        adaptive: bool = True,
    ):
        if max_partition_size is not None and (
            not isinstance(max_partition_size, int) or isinstance(max_partition_size, bool)
            or max_partition_size <= 0
        ):
            raise ValueError("max_partition_size must be a positive integer or None")
        self.max_partition_size = max_partition_size
        self.coherence_threshold = coherence_threshold
        self.use_spatial_index = use_spatial_index and HAS_SCIPY and HAS_NUMPY
        self.adaptive = adaptive
        self._kdtree = None
        self._node_index_map = None

    def partition_network(self, graph: TNFRGraph) -> list[tuple[set[Any], TNFRGraph]]:
        """Partition network into coherent subgraphs.

        Parameters
        ----------
        graph : TNFRGraph
            TNFR network to partition. Nodes must have 'vf' and 'phase' attrs.

        Returns
        -------
        list[tuple[set[Any], TNFRGraph]]
            list of (node_set, subgraph) tuples for parallel processing.

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
        if self.max_partition_size is not None:
            partition_size = min(partition_size, self.max_partition_size)

        # Build spatial index if requested and available
        if self.use_spatial_index:
            self._build_spatial_index(graph)

        # Detect TNFR communities
        communities = self._detect_tnfr_communities(graph)

        # Create balanced partitions
        partitions = []
        current_partition = set()

        for community in communities:
            # A coherent community may itself exceed the capacity. Split it
            # in stable graph order while preserving each induced subgraph.
            ordered = [node for node in graph if node in community]
            for start in range(0, len(ordered), partition_size):
                chunk = set(ordered[start:start + partition_size])
                if len(current_partition) + len(chunk) <= partition_size:
                    current_partition.update(chunk)
                else:
                    if current_partition:
                        subgraph = graph.subgraph(current_partition).copy()
                        partitions.append((current_partition.copy(), subgraph))
                    current_partition = chunk

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

        if density > _DENSITY_DENSE_THRESHOLD:
            # Dense networks: smaller partitions reduce communication overhead
            size_multiplier = 0.5
        elif density > _DENSITY_MEDIUM_THRESHOLD:
            # Medium density: balanced partitioning
            size_multiplier = 1.0
        else:
            # Sparse networks: larger partitions okay
            size_multiplier = 1.5

        # Adjust based on clustering
        try:
            avg_clustering = nx.average_clustering(graph)
            if avg_clustering > _CLUSTERING_HIGH_THRESHOLD:
                # High clustering: communities are well-defined, can use smaller partitions
                size_multiplier *= 0.8
            elif avg_clustering < _CLUSTERING_LOW_THRESHOLD:
                # Low clustering: use larger partitions
                size_multiplier *= 1.2
        except (AttributeError, ZeroDivisionError, ValueError, TypeError, nx.NetworkXNotImplemented):
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
        coords = np.array(
            [
                [
                    _node_attribute(graph, node, ALIAS_VF, 1.0),
                    _node_attribute(graph, node, ALIAS_THETA, 0.0),
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
        self, graph: TNFRGraph, seed: Any, available: set[Any], k: int = 20
    ) -> list[Any]:
        """Find k nearest coherent neighbors using spatial index.

        Uses KDTree for O(log n) nearest neighbor finding instead of O(n).

        Parameters
        ----------
        graph : TNFRGraph
            Network graph
        seed : Any
            Seed node
        available : set[Any]
            Available nodes to consider
        k : int
            Number of nearest neighbors to find

        Returns
        -------
        list[Any]
            list of up to k nearest coherent neighbors
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
        for idx in np.atleast_1d(indices):
            if idx == seed_idx:
                continue
            node = self._node_index_map[idx]
            if node in available:
                neighbors.append(node)

        return neighbors

    def _detect_tnfr_communities(self, graph: TNFRGraph) -> list[set[Any]]:
        """Detect communities using TNFR coherence metrics.

        Uses structural frequency and phase to grow coherent communities rather
        than classical modularity or betweenness metrics.
        """
        communities = []
        unprocessed = set(graph.nodes())

        while unprocessed:
            # Select seed node
            seed = next(node for node in graph if node in unprocessed)
            community = self._grow_coherent_community(graph, seed, unprocessed)
            communities.append(community)
            unprocessed -= community

        return communities

    def _grow_coherent_community(
        self, graph: TNFRGraph, seed: Any, available: set[Any]
    ) -> set[Any]:
        """Grow community from seed based on structural coherence.

        Parameters
        ----------
        graph : TNFRGraph
            Full network graph
        seed : Any
            Starting node for community growth
        available : set[Any]
            Nodes that haven't been assigned to communities yet

        Returns
        -------
        set[Any]
            set of nodes forming a coherent community

        Notes
        -----
        Uses spatial indexing for O(log n) neighbor finding when available,
        falling back to O(n) graph neighbors otherwise.
        """
        community = {seed}
        node_order = {node: index for index, node in enumerate(graph)}

        # Use spatial index if available for faster neighbor finding
        if self.use_spatial_index and self._kdtree is not None:
            candidates = set(
                self._find_coherent_neighbors_spatial(graph, seed, available, k=50)
            )
        else:
            neighbors = graph.neighbors(seed)
            candidates = set(neighbors) & available

        while candidates:
            # Find most coherent candidate
            best_candidate = None
            best_coherence = -1.0

            for candidate in sorted(candidates, key=node_order.__getitem__):
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
        self, graph: TNFRGraph, community: set[Any], candidate: Any
    ) -> float:
        """Compute coherence between candidate and existing community.

        Uses TNFR metrics: frequency alignment (νf) and phase synchrony.

        Parameters
        ----------
        graph : TNFRGraph
            Network graph
        community : set[Any]
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

        candidate_vf = _node_attribute(graph, candidate, ALIAS_VF, 1.0)
        candidate_phase = _node_attribute(graph, candidate, ALIAS_THETA, 0.0)

        coherences = []
        for member in community:
            member_vf = _node_attribute(graph, member, ALIAS_VF, 1.0)
            member_phase = _node_attribute(graph, member, ALIAS_THETA, 0.0)

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

        return math.fsum(coherences) / len(coherences) if coherences else 0.0

    def partition_with_manifest(
        self,
        graph: TNFRGraph,
        output_dir: Path,
        partition_id: str,
    ) -> dict[str, Any]:
        """Partition network and export manifest for self-optimization.

        Parameters
        ----------
        graph : TNFRGraph
            TNFR network to partition.
        output_dir : Path
            Directory where manifests will be written.
        partition_id : str
            Unique identifier for this fractal partition operation.

        Returns
        -------
        dict[str, Any]
            Dictionary with keys:
            - 'partitions': list[(node_set, subgraph)] from partition_network
            - 'manifest_absolute': Path to partition manifest
            - 'summary_absolute': Path to partition summary

        Notes
        -----
        Manifest format compatible with self_opt_support pipeline:
        - operation_type: 'fractal_partition'
        - partition_id: unique identifier
        - communities: list of community metadata with coherence scores
        - telemetry: global coherence, sense_index, phase metrics
        - network_metadata: node count, edge count, partition count

        The entries index references one graph payload per returned partition.
        Node IDs retain their scalar JSON types. Graph attributes, triad, and
        JSON history are preserved; unsupported runtime state raises ValueError.
        """
        from datetime import datetime, timezone
        from pathlib import Path
        from ..engines.manifest import collect_manifest_telemetry, write_manifest_bundle

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Perform partitioning
        partitions = self.partition_network(graph)

        telemetry = collect_manifest_telemetry(graph)

        # Extract network metadata
        node_count = len(graph.nodes()) if hasattr(graph, "nodes") else 0
        edge_count = len(graph.edges()) if hasattr(graph, "edges") else 0

        # Serialize partition communities
        communities_serialized = []
        graph_payloads = []
        for partition_idx, (node_set, subgraph) in enumerate(partitions):
            community_telemetry = collect_manifest_telemetry(subgraph)
            graph_payloads.append((f"{partition_id}:p{partition_idx}", subgraph, community_telemetry))

            community_data = {
                "partition_index": partition_idx,
                "node_count": len(node_set),
                "edge_count": (
                    len(subgraph.edges()) if hasattr(subgraph, "edges") else 0
                ),
                "node_ids": [n for n in graph if n in node_set],
                "community_coherence": community_telemetry["coherence"],
            }
            communities_serialized.append(community_data)

        # Build manifest
        manifest = {
            "operation_type": "fractal_partition",
            "partition_id": partition_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "network_metadata": {
                "node_count": node_count,
                "edge_count": edge_count,
                "partition_count": len(partitions),
            },
            "telemetry": telemetry,
            "communities": communities_serialized,
            "partitioner_config": {
                "max_partition_size": self.max_partition_size,
                "coherence_threshold": self.coherence_threshold,
                "use_spatial_index": self.use_spatial_index,
                "adaptive": self.adaptive,
            },
        }

        # Write summary
        summary = {
            "operation_type": "fractal_partition",
            "partition_id": partition_id,
            "partition_count": len(partitions),
            "coherence": telemetry.get("coherence"),
            "sense_index": telemetry.get("sense_index"),
            "average_community_size": node_count / len(partitions) if partitions else 0,
        }
        return {
            "partitions": partitions,
            **write_manifest_bundle(
                output_dir, "fractal_partition_manifest.json", "fractal_partition_summary.json",
                manifest, summary, graph_payloads,
            ),
        }
