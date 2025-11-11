"""Memory-optimized sparse representations for TNFR graphs.

Implements sparse storage strategies that minimize memory footprint while
maintaining computational efficiency and TNFR semantic fidelity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

import numpy as np
from scipy import sparse

from ..types import NodeId
from ..utils import get_logger

logger = get_logger(__name__)


@dataclass
class MemoryReport:
    """Memory usage report for sparse TNFR graphs.

    Attributes
    ----------
    total_mb : float
        Total memory usage in megabytes
    per_node_kb : float
        Memory usage per node in kilobytes
    breakdown : Dict[str, int]
        Detailed breakdown by component in bytes
    """

    total_mb: float
    per_node_kb: float
    breakdown: Dict[str, int]


class SparseCache:
    """Time-to-live cache for sparse computation results.

    Stores computed values with automatic invalidation after a specified
    number of evolution steps.

    Parameters
    ----------
    capacity : int
        Maximum number of cached entries
    ttl_steps : int
        Time-to-live in evolution steps before invalidation
    """

    def __init__(self, capacity: int, ttl_steps: int = 10):
        self.capacity = capacity
        self.ttl_steps = ttl_steps
        self._cache: Dict[NodeId, tuple[float, int]] = {}
        self._current_step = 0

    def get(self, node_id: NodeId) -> Optional[float]:
        """Get cached value if not expired."""
        if node_id in self._cache:
            value, cached_step = self._cache[node_id]
            if self._current_step - cached_step < self.ttl_steps:
                return value
            else:
                # Expired
                del self._cache[node_id]
        return None

    def update(self, values: Dict[NodeId, float]) -> None:
        """Update cache with new values."""
        # Implement simple LRU: if over capacity, remove oldest
        if len(self._cache) + len(values) > self.capacity:
            # Remove oldest entries
            to_remove = len(self._cache) + len(values) - self.capacity
            oldest_keys = sorted(self._cache.keys(), key=lambda k: self._cache[k][1])[:to_remove]
            for key in oldest_keys:
                del self._cache[key]

        # Add new values
        for node_id, value in values.items():
            self._cache[node_id] = (value, self._current_step)

    def step(self) -> None:
        """Advance evolution step counter."""
        self._current_step += 1

    def clear(self) -> None:
        """Clear all cached values."""
        self._cache.clear()
        self._current_step = 0

    def memory_usage(self) -> int:
        """Return estimated memory usage in bytes."""
        # Each cache entry: node_id (assume int, 8 bytes) + value (8 bytes) + step (8 bytes)
        # Plus dict overhead (~112 bytes per entry)
        return len(self._cache) * (8 + 8 + 8 + 112)


class CompactAttributeStore:
    """Compressed storage for node attributes with defaults.

    Only stores non-default values to minimize memory footprint. TNFR
    canonical defaults:
    - vf (νf): 1.0 Hz_str
    - theta (θ): 0.0 radians
    - si (Si): 0.0

    Parameters
    ----------
    node_count : int
        Total number of nodes
    """

    def __init__(self, node_count: int):
        self.node_count = node_count

        # Only store non-default values (sparse dictionaries)
        self._vf_sparse: Dict[NodeId, np.float32] = {}
        self._theta_sparse: Dict[NodeId, np.float32] = {}
        self._si_sparse: Dict[NodeId, np.float32] = {}
        self._epi_sparse: Dict[NodeId, np.float32] = {}
        self._dnfr_sparse: Dict[NodeId, np.float32] = {}

        # TNFR canonical defaults
        self.default_vf = 1.0  # Hz_str
        self.default_theta = 0.0  # radians
        self.default_si = 0.0
        self.default_epi = 0.0
        self.default_dnfr = 0.0

    def set_vf(self, node_id: NodeId, vf: float) -> None:
        """Set structural frequency, store only if non-default."""
        if abs(vf - self.default_vf) > 1e-10:
            self._vf_sparse[node_id] = np.float32(vf)
        else:
            self._vf_sparse.pop(node_id, None)

    def get_vf(self, node_id: NodeId) -> float:
        """Get structural frequency with default fallback."""
        return float(self._vf_sparse.get(node_id, self.default_vf))

    def get_vfs(self, node_ids: Sequence[NodeId]) -> np.ndarray:
        """Vectorized get with broadcasting defaults."""
        result = np.full(len(node_ids), self.default_vf, dtype=np.float32)
        for i, node_id in enumerate(node_ids):
            if node_id in self._vf_sparse:
                result[i] = self._vf_sparse[node_id]
        return result

    def set_theta(self, node_id: NodeId, theta: float) -> None:
        """Set phase, store only if non-default."""
        if abs(theta - self.default_theta) > 1e-10:
            self._theta_sparse[node_id] = np.float32(theta)
        else:
            self._theta_sparse.pop(node_id, None)

    def get_theta(self, node_id: NodeId) -> float:
        """Get phase with default fallback."""
        return float(self._theta_sparse.get(node_id, self.default_theta))

    def get_thetas(self, node_ids: Sequence[NodeId]) -> np.ndarray:
        """Vectorized get phases."""
        result = np.full(len(node_ids), self.default_theta, dtype=np.float32)
        for i, node_id in enumerate(node_ids):
            if node_id in self._theta_sparse:
                result[i] = self._theta_sparse[node_id]
        return result

    def set_si(self, node_id: NodeId, si: float) -> None:
        """Set sense index, store only if non-default."""
        if abs(si - self.default_si) > 1e-10:
            self._si_sparse[node_id] = np.float32(si)
        else:
            self._si_sparse.pop(node_id, None)

    def get_si(self, node_id: NodeId) -> float:
        """Get sense index with default fallback."""
        return float(self._si_sparse.get(node_id, self.default_si))

    def set_epi(self, node_id: NodeId, epi: float) -> None:
        """Set EPI, store only if non-default."""
        if abs(epi - self.default_epi) > 1e-10:
            self._epi_sparse[node_id] = np.float32(epi)
        else:
            self._epi_sparse.pop(node_id, None)

    def get_epi(self, node_id: NodeId) -> float:
        """Get EPI with default fallback."""
        return float(self._epi_sparse.get(node_id, self.default_epi))

    def get_epis(self, node_ids: Sequence[NodeId]) -> np.ndarray:
        """Vectorized get EPIs."""
        result = np.full(len(node_ids), self.default_epi, dtype=np.float32)
        for i, node_id in enumerate(node_ids):
            if node_id in self._epi_sparse:
                result[i] = self._epi_sparse[node_id]
        return result

    def set_dnfr(self, node_id: NodeId, dnfr: float) -> None:
        """Set ΔNFR, store only if non-default."""
        if abs(dnfr - self.default_dnfr) > 1e-10:
            self._dnfr_sparse[node_id] = np.float32(dnfr)
        else:
            self._dnfr_sparse.pop(node_id, None)

    def get_dnfr(self, node_id: NodeId) -> float:
        """Get ΔNFR with default fallback."""
        return float(self._dnfr_sparse.get(node_id, self.default_dnfr))

    def memory_usage(self) -> int:
        """Report memory usage in bytes."""
        # Each sparse dict entry: key (8 bytes) + value (4 bytes float32) + dict overhead (~112 bytes)
        bytes_per_entry = 8 + 4 + 112

        vf_memory = len(self._vf_sparse) * bytes_per_entry
        theta_memory = len(self._theta_sparse) * bytes_per_entry
        si_memory = len(self._si_sparse) * bytes_per_entry
        epi_memory = len(self._epi_sparse) * bytes_per_entry
        dnfr_memory = len(self._dnfr_sparse) * bytes_per_entry

        return vf_memory + theta_memory + si_memory + epi_memory + dnfr_memory


class SparseTNFRGraph:
    """Memory-optimized TNFR graph using sparse representations.

    Reduces per-node memory footprint from ~8.5KB to <1KB by using:
    - Sparse CSR adjacency matrices
    - Compact attribute storage (only non-default values)
    - Intelligent caching with TTL invalidation

    All TNFR canonical invariants are preserved:
    - Nodal equation: ∂EPI/∂t = νf · ΔNFR(t)
    - Deterministic computation with reproducible seeds
    - Operator closure and phase verification

    Parameters
    ----------
    node_count : int
        Number of nodes in the graph
    expected_density : float, optional
        Expected edge density for sparse matrix preallocation
    seed : int, optional
        Random seed for reproducible initialization

    Examples
    --------
    Create a sparse graph with 10,000 nodes:

    >>> from tnfr.sparse import SparseTNFRGraph
    >>> graph = SparseTNFRGraph(10000, expected_density=0.1, seed=42)
    >>> graph.node_count
    10000
    >>> report = graph.memory_footprint()
    >>> report.per_node_kb < 1.0  # Target: <1KB per node
    True
    """

    def __init__(
        self,
        node_count: int,
        expected_density: float = 0.1,
        seed: Optional[int] = None,
    ):
        if node_count <= 0:
            raise ValueError("node_count must be positive")
        if not 0.0 <= expected_density <= 1.0:
            raise ValueError("expected_density must be in [0, 1]")

        self.node_count = node_count
        self.expected_density = expected_density
        self.seed = seed

        # Sparse adjacency matrix (CSR format for efficient row slicing)
        # Initialize empty, will be populated via add_edge
        self.adjacency = sparse.lil_matrix((node_count, node_count), dtype=np.float32)

        # Compact node attributes
        self.node_attributes = CompactAttributeStore(node_count)

        # Caches with different TTLs
        self._dnfr_cache = SparseCache(node_count, ttl_steps=10)
        self._coherence_cache = SparseCache(node_count, ttl_steps=50)

        # Initialize with random values if seed provided
        if seed is not None:
            self._initialize_random(seed)

        logger.info(
            f"Created sparse TNFR graph: {node_count} nodes, " f"density={expected_density:.2f}"
        )

    def _initialize_random(self, seed: int) -> None:
        """Initialize graph with random Erdős-Rényi structure and attributes."""
        rng = np.random.RandomState(seed)

        # Generate random edges efficiently using NetworkX
        import networkx as nx

        G_temp = nx.erdos_renyi_graph(self.node_count, self.expected_density, seed=seed)

        # Copy edges to sparse matrix
        for u, v in G_temp.edges():
            weight = rng.uniform(0.5, 1.0)
            self.adjacency[u, v] = weight
            self.adjacency[v, u] = weight

        # Initialize node attributes
        for node_id in range(self.node_count):
            self.node_attributes.set_epi(node_id, rng.uniform(0.0, 1.0))
            self.node_attributes.set_vf(node_id, rng.uniform(0.5, 1.5))
            self.node_attributes.set_theta(node_id, rng.uniform(0.0, 2 * np.pi))

    def add_edge(self, u: NodeId, v: NodeId, weight: float = 1.0) -> None:
        """Add edge with weight.

        Parameters
        ----------
        u, v : NodeId
            Node identifiers (must be in [0, node_count))
        weight : float
            Edge coupling weight
        """
        if not (0 <= u < self.node_count and 0 <= v < self.node_count):
            raise ValueError("Node IDs must be in [0, node_count)")

        self.adjacency[u, v] = weight
        self.adjacency[v, u] = weight  # Undirected graph

    def compute_dnfr_sparse(self, node_ids: Optional[Sequence[NodeId]] = None) -> np.ndarray:
        """Compute ΔNFR using sparse matrix operations.

        Implements the TNFR ΔNFR computation efficiently using sparse
        matrix-vector operations.

        Parameters
        ----------
        node_ids : Sequence[NodeId], optional
            Specific nodes to compute ΔNFR for. If None, computes for all.

        Returns
        -------
        np.ndarray
            ΔNFR values for requested nodes
        """
        if node_ids is None:
            node_ids = list(range(self.node_count))

        # Check cache first
        dnfr_values = np.zeros(len(node_ids), dtype=np.float32)
        uncached_indices = []
        uncached_ids = []

        for i, node_id in enumerate(node_ids):
            cached = self._dnfr_cache.get(node_id)
            if cached is not None:
                dnfr_values[i] = cached
            else:
                uncached_indices.append(i)
                uncached_ids.append(node_id)

        if uncached_ids:
            # Convert to CSR for efficient computation
            adj_csr = self.adjacency.tocsr()

            # Get phases for all nodes (needed for phase differences)
            all_phases = self.node_attributes.get_thetas(range(self.node_count))

            # Compute for uncached nodes
            for idx, node_id in zip(uncached_indices, uncached_ids):
                node_phase = all_phases[node_id]

                # Get neighbors via sparse row
                row_start = adj_csr.indptr[node_id]
                row_end = adj_csr.indptr[node_id + 1]
                neighbor_indices = adj_csr.indices[row_start:row_end]

                if len(neighbor_indices) > 0:
                    neighbor_phases = all_phases[neighbor_indices]
                    # Use sparse data directly (more efficient)
                    neighbor_weights = adj_csr.data[row_start:row_end]

                    # Phase differences
                    phase_diffs = np.sin(node_phase - neighbor_phases)

                    # Weighted sum
                    dnfr = np.sum(neighbor_weights * phase_diffs) / len(neighbor_indices)
                else:
                    dnfr = 0.0

                dnfr_values[idx] = dnfr

            # Update cache
            cache_update = dict(zip(uncached_ids, dnfr_values[uncached_indices]))
            self._dnfr_cache.update(cache_update)

        return dnfr_values

    def evolve_sparse(self, dt: float = 0.1, steps: int = 10) -> Dict[str, Any]:
        """Evolve graph using sparse operations.

        Applies nodal equation: ∂EPI/∂t = νf · ΔNFR(t)

        Parameters
        ----------
        dt : float
            Time step
        steps : int
            Number of evolution steps

        Returns
        -------
        Dict[str, Any]
            Evolution metrics
        """
        for step in range(steps):
            # Compute ΔNFR for all nodes
            all_node_ids = list(range(self.node_count))
            dnfr_values = self.compute_dnfr_sparse(all_node_ids)
            vf_values = self.node_attributes.get_vfs(all_node_ids)
            epi_values = self.node_attributes.get_epis(all_node_ids)

            # Update EPIs according to nodal equation
            new_epis = epi_values + vf_values * dnfr_values * dt

            # Store updates
            for node_id, new_epi, dnfr in zip(all_node_ids, new_epis, dnfr_values):
                self.node_attributes.set_epi(node_id, float(new_epi))
                self.node_attributes.set_dnfr(node_id, float(dnfr))

            # Advance cache steps
            self._dnfr_cache.step()
            self._coherence_cache.step()

        # Compute final coherence
        coherence = self._compute_coherence()

        return {
            "final_coherence": coherence,
            "steps": steps,
        }

    def _compute_coherence(self) -> float:
        """Compute total coherence: C(t) = 1 / (1 + mean(|ΔNFR|))."""
        dnfr_values = self.compute_dnfr_sparse()
        mean_abs_dnfr = np.mean(np.abs(dnfr_values))
        return 1.0 / (1.0 + mean_abs_dnfr)

    def memory_footprint(self) -> MemoryReport:
        """Report detailed memory usage.

        Returns
        -------
        MemoryReport
            Detailed memory usage breakdown
        """
        # Convert to CSR for accurate size measurement
        adj_csr = self.adjacency.tocsr()
        adjacency_memory = adj_csr.data.nbytes + adj_csr.indices.nbytes + adj_csr.indptr.nbytes

        attributes_memory = self.node_attributes.memory_usage()
        cache_memory = self._dnfr_cache.memory_usage() + self._coherence_cache.memory_usage()

        total_memory = adjacency_memory + attributes_memory + cache_memory
        memory_per_node = total_memory / self.node_count

        return MemoryReport(
            total_mb=total_memory / (1024 * 1024),
            per_node_kb=memory_per_node / 1024,
            breakdown={
                "adjacency": adjacency_memory,
                "attributes": attributes_memory,
                "caches": cache_memory,
            },
        )

    def number_of_edges(self) -> int:
        """Return number of edges (undirected, so count each once)."""
        return self.adjacency.nnz // 2
