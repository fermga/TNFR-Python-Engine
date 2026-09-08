"""TNFR-aware affinity partitioning for parallel computation.

Communities are grown with a bounded heuristic that combines structural-frequency
proximity and circular phase alignment. The affinity is a partitioning score; it
does not estimate canonical coherence C(t), which depends on DeltaNFR and dEPI.
"""

from __future__ import annotations

import math
from numbers import Real
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

from ..constants.aliases import ALIAS_THETA, ALIAS_VF

# ---------------------------------------------------------------------------
# Network density / clustering thresholds for partition sizing
# ---------------------------------------------------------------------------
_DENSITY_DENSE_THRESHOLD = 0.5
_DENSITY_MEDIUM_THRESHOLD = 0.1
_CLUSTERING_HIGH_THRESHOLD = 0.6
_CLUSTERING_LOW_THRESHOLD = 0.2
_DEFAULT_AFFINITY_THRESHOLD = 0.3


def _validated_affinity_threshold(value: Any, *, label: str) -> float:
    """Return a finite positive-alignment threshold in [0, 1]."""
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(float(value))
        or not 0.0 <= float(value) <= 1.0
    ):
        raise ValueError(f"{label} must be a finite real in [0, 1]")
    return float(value)


def _node_attribute(
    graph: Any,
    node: Any,
    aliases: tuple[str, ...],
    default: float,
    *,
    label: str,
    nonnegative: bool = False,
) -> float:
    """Read one finite structural scalar while preserving an explicit zero."""
    attributes = graph.nodes[node]
    value = default
    for key in aliases:
        if key in attributes:
            value = attributes[key]
            break
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{label} for node {node!r} must be a finite real")
    resolved = float(value)
    if not math.isfinite(resolved):
        raise ValueError(f"{label} for node {node!r} must be a finite real")
    if nonnegative and resolved < 0.0:
        raise ValueError(f"{label} for node {node!r} must be nonnegative")
    return resolved


class FractalPartitioner:
    """Partition TNFR networks by frequency-and-phase affinity.

    The partitioner groups nodes with similar structural frequencies and aligned
    circular phases. This operational affinity is independent of canonical C(t);
    manifests compute C(t) separately from the resulting subgraphs.

    Parameters
    ----------
    max_partition_size : int, default=100
        Maximum number of nodes per partition. Larger partitions reduce
        communication overhead but may limit parallelism. If None, uses
        adaptive partitioning based on network density.
    coherence_threshold : float, optional
        Compatibility name for affinity_threshold. The default is 0.3 when
        neither name is supplied.
    affinity_threshold : float, optional
        Minimum frequency-and-phase affinity for adding a node. Higher values
        create tighter communities but may result in more partitions.
    use_spatial_index : bool, default=True
        Whether to build a KDTree for structural-coordinate candidate queries.
        Index construction is O(n log n); total partitioning complexity also
        depends on repeated community scoring. Requires SciPy.
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
    The optional KDTree has O(n log n) construction and O(log n + k) query
    cost under its usual assumptions. Community growth still evaluates repeated
    candidates, so these bounds are not an end-to-end partitioning bound.
    """

    def __init__(
        self,
        max_partition_size: int | None = 100,
        coherence_threshold: float | None = None,
        use_spatial_index: bool = True,
        adaptive: bool = True,
        *,
        affinity_threshold: float | None = None,
    ):
        if max_partition_size is not None and (
            not isinstance(max_partition_size, int)
            or isinstance(max_partition_size, bool)
            or max_partition_size <= 0
        ):
            raise ValueError("max_partition_size must be a positive integer or None")

        legacy_threshold = (
            None
            if coherence_threshold is None
            else _validated_affinity_threshold(
                coherence_threshold, label="coherence_threshold"
            )
        )
        canonical_threshold = (
            None
            if affinity_threshold is None
            else _validated_affinity_threshold(
                affinity_threshold, label="affinity_threshold"
            )
        )
        if (
            legacy_threshold is not None
            and canonical_threshold is not None
            and legacy_threshold != canonical_threshold
        ):
            raise ValueError(
                "coherence_threshold and affinity_threshold must agree when both "
                "are provided"
            )
        self.max_partition_size = max_partition_size
        self.affinity_threshold = (
            canonical_threshold
            if canonical_threshold is not None
            else (
                legacy_threshold
                if legacy_threshold is not None
                else _DEFAULT_AFFINITY_THRESHOLD
            )
        )
        self.use_spatial_index = use_spatial_index and HAS_SCIPY and HAS_NUMPY
        self.adaptive = adaptive
        self._kdtree = None
        self._node_index_map = None

    @property
    def affinity_threshold(self) -> float:
        """Return the operational partition-affinity threshold."""
        return self._affinity_threshold

    @affinity_threshold.setter
    def affinity_threshold(self, value: float) -> None:
        self._affinity_threshold = _validated_affinity_threshold(
            value, label="affinity_threshold"
        )

    @property
    def coherence_threshold(self) -> float:
        """Compatibility alias for affinity_threshold; this is not C(t)."""
        return self.affinity_threshold

    @coherence_threshold.setter
    def coherence_threshold(self, value: float) -> None:
        self.affinity_threshold = _validated_affinity_threshold(
            value, label="coherence_threshold"
        )

    @staticmethod
    def _validate_structural_coordinates(graph: TNFRGraph) -> None:
        """Validate the finite canonical coordinates used by partition scoring."""
        for node in graph:
            _node_attribute(
                graph, node, ALIAS_VF, 1.0, label="nu_f", nonnegative=True
            )
            _node_attribute(graph, node, ALIAS_THETA, 0.0, label="phase")

    def partition_network(self, graph: TNFRGraph) -> list[tuple[set[Any], TNFRGraph]]:
        """Partition a network into affinity-grouped subgraphs.

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
        - Communities grouped by the declared frequency-and-phase affinity
        - Circular phase alignment contributes to the grouping score
        - Frequency alignment respected

        Optionally builds a KDTree for candidate queries; this does not establish
        an end-to-end O(n log n) partitioning bound. Adaptive sizing uses
        declared density and clustering heuristics.
        """

        if len(graph) == 0:
            return []
        self._validate_structural_coordinates(graph)

        # Determine the configured heuristic partition size
        if self.adaptive:
            partition_size = self._compute_adaptive_partition_size(graph)
        else:
            partition_size = self.max_partition_size or 100
        if self.max_partition_size is not None:
            partition_size = min(partition_size, self.max_partition_size)

        # Build spatial index if requested and available
        if self.use_spatial_index:
            self._build_spatial_index(graph)

        # Detect operational affinity groups
        communities = self._detect_tnfr_communities(graph)

        # Create balanced partitions
        partitions = []
        current_partition = set()

        for community in communities:
            # An affinity group may exceed the capacity. Split it
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
        """Compute a heuristic partition size from declared graph characteristics.

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
                # High clustering permits smaller partitions with fewer cut edges
                size_multiplier *= 0.8
            elif avg_clustering < _CLUSTERING_LOW_THRESHOLD:
                # Low clustering: use larger partitions
                size_multiplier *= 1.2
        except (
            AttributeError,
            ZeroDivisionError,
            ValueError,
            TypeError,
            nx.NetworkXNotImplemented,
        ):
            # If clustering calculation fails, skip adjustment
            pass

        adapted_size = int(base_size * size_multiplier)
        # Ensure reasonable bounds
        return max(10, min(adapted_size, 500))

    def _build_spatial_index(self, graph: TNFRGraph) -> None:
        """Build a KDTree over structural coordinates in O(n log n).

        Constructs a 3D index using normalized νf and (cos(phase), sin(phase))
        to enable fast nearest-neighbor queries.
        """
        if not HAS_SCIPY or not HAS_NUMPY:
            return

        nodes = list(graph.nodes())
        if len(nodes) == 0:
            return

        # Embed phase on the unit circle so angles adjacent across the wrap
        # boundary remain adjacent in the KDTree.
        frequencies = np.array(
            [
                _node_attribute(
                    graph, node, ALIAS_VF, 1.0, label="nu_f", nonnegative=True
                )
                for node in nodes
            ],
            dtype=float,
        )
        phases = np.array(
            [
                _node_attribute(graph, node, ALIAS_THETA, 0.0, label="phase")
                for node in nodes
            ],
            dtype=float,
        )
        if frequencies.std() > 0:
            frequencies = (
                frequencies - frequencies.mean()
            ) / frequencies.std()
        coords = np.column_stack(
            (frequencies, np.cos(phases), np.sin(phases))
        )

        # Build KDTree
        self._kdtree = KDTree(coords)
        self._node_index_map = {i: node for i, node in enumerate(nodes)}

    def _find_affinity_candidates_spatial(
        self, graph: TNFRGraph, seed: Any, available: set[Any], k: int = 20
    ) -> list[Any]:
        """Find up to k nearby affinity candidates using the spatial index.

        The KDTree query has expected O(log n + k) cost under standard assumptions;
        filtering and subsequent community scoring are separate costs.

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
            list of up to k nearby affinity candidates
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
        """Detect communities with the frequency-and-phase affinity.

        This operational grouping score is distinct from canonical C(t) and
        from graph modularity or betweenness.
        """
        communities = []
        unprocessed = set(graph.nodes())

        while unprocessed:
            # Select seed node
            seed = next(node for node in graph if node in unprocessed)
            community = self._grow_affinity_community(graph, seed, unprocessed)
            communities.append(community)
            unprocessed -= community

        return communities

    def _grow_affinity_community(
        self, graph: TNFRGraph, seed: Any, available: set[Any]
    ) -> set[Any]:
        """Grow a community from a seed using structural affinity.

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
            Nodes selected by the affinity threshold

        Notes
        -----
        Uses an optional KDTree to propose structural-coordinate candidates.
        Repeated scoring against the growing community remains an additional cost.
        """
        community = {seed}
        node_order = {node: index for index, node in enumerate(graph)}

        # Use spatial index if available for faster neighbor finding
        if self.use_spatial_index and self._kdtree is not None:
            candidates = set(
                self._find_affinity_candidates_spatial(graph, seed, available, k=50)
            )
        else:
            neighbors = graph.neighbors(seed)
            candidates = set(neighbors) & available

        while candidates:
            # Find the candidate with greatest affinity
            best_candidate = None
            best_affinity = -1.0

            for candidate in sorted(candidates, key=node_order.__getitem__):
                affinity = self._compute_community_affinity(
                    graph, community, candidate
                )
                if affinity > best_affinity:
                    best_affinity = affinity
                    best_candidate = candidate

            # Add if above threshold
            if best_affinity > self.affinity_threshold:
                community.add(best_candidate)
                candidates.remove(best_candidate)

                # Add new neighbors as candidates
                if self.use_spatial_index and self._kdtree is not None:
                    new_neighbors = set(
                        self._find_affinity_candidates_spatial(
                            graph, best_candidate, available, k=50
                        )
                    )
                else:
                    new_neighbors = set(graph.neighbors(best_candidate)) & available

                candidates.update(new_neighbors - community)
            else:
                break  # No remaining candidate passes the affinity threshold

        return community

    def _grow_coherent_community(
        self, graph: TNFRGraph, seed: Any, available: set[Any]
    ) -> set[Any]:
        """Compatibility alias for affinity-based community growth."""
        return self._grow_affinity_community(graph, seed, available)
    def _compute_community_affinity(
        self, graph: TNFRGraph, community: set[Any], candidate: Any
    ) -> float:
        """Compute frequency-and-phase affinity to an existing community.

        This selected 60/40 blend is an operational partitioning heuristic. It
        does not use DeltaNFR or dEPI and therefore is not canonical C(t).

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
            Weighted affinity in [-0.4, 1], where higher means better alignment
        """
        if not community:
            return 0.0

        candidate_vf = _node_attribute(
            graph, candidate, ALIAS_VF, 1.0, label="nu_f", nonnegative=True
        )
        candidate_phase = _node_attribute(
            graph, candidate, ALIAS_THETA, 0.0, label="phase"
        )

        affinities = []
        for member in community:
            member_vf = _node_attribute(
                graph, member, ALIAS_VF, 1.0, label="nu_f", nonnegative=True
            )
            member_phase = _node_attribute(
                graph, member, ALIAS_THETA, 0.0, label="phase"
            )

            # Frequency proximity: inversely proportional to difference
            vf_diff = abs(candidate_vf - member_vf)
            frequency_proximity = 1.0 / (1.0 + vf_diff)

            # Circular phase alignment
            phase_diff = candidate_phase - member_phase
            if HAS_NUMPY:
                phase_alignment = float(np.cos(phase_diff))
            else:
                phase_alignment = math.cos(phase_diff)

            # Weighted combination: prioritize frequency alignment
            affinities.append(0.6 * frequency_proximity + 0.4 * phase_alignment)

        return (
            math.fsum(affinities) / len(affinities)
            if affinities
            else 0.0
        )

    def _compute_community_coherence(
        self, graph: TNFRGraph, community: set[Any], candidate: Any
    ) -> float:
        """Compatibility alias for community affinity; this is not C(t)."""
        return self._compute_community_affinity(graph, community, candidate)

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
        - communities: metadata with separately computed canonical C(t)
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
            graph_payloads.append(
                (
                    f"{partition_id}:p{partition_idx}",
                    subgraph,
                    community_telemetry,
                )
            )

            community_data = {
                "partition_index": partition_idx,
                "node_count": len(node_set),
                "edge_count": (
                    len(subgraph.edges()) if hasattr(subgraph, "edges") else 0
                ),
                "node_ids": [n for n in graph if n in node_set],
                "community_C_t": community_telemetry["coherence"],
                "community_coherence": community_telemetry["coherence"],
                "coherence_metric_kind": "canonical_C_t",
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
                "affinity_threshold": self.affinity_threshold,
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
            "C_t": telemetry.get("coherence"),
            "coherence": telemetry.get("coherence"),
            "coherence_metric_kind": "canonical_C_t",
            "sense_index": telemetry.get("sense_index"),
            "average_community_size": node_count / len(partitions) if partitions else 0,
        }
        return {
            "partitions": partitions,
            **write_manifest_bundle(
                output_dir,
                "fractal_partition_manifest.json",
                "fractal_partition_summary.json",
                manifest, summary, graph_payloads,
            ),
        }
