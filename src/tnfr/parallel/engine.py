"""Parallel execution engine for TNFR computations.

Provides thread and process-based parallelization for ΔNFR and Si computations
while preserving all TNFR structural invariants.
"""

from __future__ import annotations

from multiprocessing import cpu_count
from typing import Any, Dict, Optional

from .partitioner import FractalPartitioner


class TNFRParallelEngine:
    """Parallel computation engine for TNFR networks.

    Leverages multiprocessing or threading to accelerate ΔNFR and Si
    computations on medium to large networks. Respects TNFR invariants by
    partitioning networks along coherence boundaries.

    Parameters
    ----------
    max_workers : int or None, optional
        Maximum number of parallel workers. None auto-detects CPU count.
    execution_mode : {"threads", "processes"}, default="threads"
        Execution backend. "threads" for I/O-bound tasks, "processes" for
        CPU-bound tasks. Note: "processes" has serialization overhead.
    partition_size : int, default=100
        Maximum nodes per partition for parallel processing.

    Examples
    --------
    >>> import networkx as nx
    >>> from tnfr.parallel import TNFRParallelEngine
    >>> G = nx.Graph()
    >>> G.add_edges_from([("a", "b"), ("b", "c")])
    >>> for node in G.nodes():
    ...     G.nodes[node]["vf"] = 1.0
    ...     G.nodes[node]["phase"] = 0.0
    ...     G.nodes[node]["epi"] = 0.5
    ...     G.nodes[node]["delta_nfr"] = 0.0
    >>> engine = TNFRParallelEngine(max_workers=2)
    >>> # Engine ready for parallel computation
    >>> engine.max_workers <= cpu_count()
    True

    Notes
    -----
    This engine integrates with existing TNFR dynamics by:
    1. Using the same n_jobs parameter conventions
    2. Preserving all node attributes and graph structure
    3. Maintaining canonical ΔNFR semantics
    4. Respecting phase coherence in partitioning
    """

    def __init__(
        self,
        max_workers: Optional[int] = None,
        execution_mode: str = "threads",
        partition_size: int = 100,
        cache_aware: bool = True,
    ):
        if max_workers is None:
            max_workers = cpu_count()

        self.max_workers = max_workers
        self.execution_mode = execution_mode
        self.cache_aware = cache_aware
        self.partitioner = FractalPartitioner(
            max_partition_size=partition_size
        )

    def _distribute_work_cache_aware(
        self, partitions: list, num_workers: int
    ) -> list:
        """Distribute work across workers in a cache-aware manner.

        Groups related partitions together to improve cache locality
        and reduce cache misses during parallel execution.

        Parameters
        ----------
        partitions : list
            List of work partitions to distribute
        num_workers : int
            Number of worker processes/threads

        Returns
        -------
        list
            List of work chunks, one per worker, organized for cache efficiency
        """
        if not self.cache_aware or len(partitions) <= num_workers:
            # Simple round-robin distribution
            chunks: list = [[] for _ in range(num_workers)]
            for i, partition in enumerate(partitions):
                chunks[i % num_workers].append(partition)
            return chunks

        # Cache-aware distribution: group spatially nearby partitions
        # This reduces cache misses when processing related nodes

        # Sort partitions by their "center" (average νf of nodes)
        def partition_center(partition_info: Any) -> float:
            node_set, subgraph = partition_info
            if not node_set:
                return 0.0
            try:
                from ..alias import get_attr
                from ..constants.aliases import ALIAS_VF

                vf_sum = sum(
                    float(
                        get_attr(subgraph.nodes[n], ALIAS_VF, None)
                        or subgraph.nodes[n].get("vf", 1.0)
                    )
                    for n in node_set
                )
                return vf_sum / len(node_set)
            except (ZeroDivisionError, AttributeError, KeyError, TypeError):
                return 0.0

        sorted_partitions = sorted(partitions, key=partition_center)

        # Distribute sorted partitions in contiguous blocks
        # This ensures workers process spatially nearby partitions
        chunks = [[] for _ in range(num_workers)]
        chunk_size = len(sorted_partitions) // num_workers
        remainder = len(sorted_partitions) % num_workers

        start_idx = 0
        for worker_id in range(num_workers):
            # Give some workers an extra partition to handle remainder
            extra = 1 if worker_id < remainder else 0
            end_idx = start_idx + chunk_size + extra
            chunks[worker_id] = sorted_partitions[start_idx:end_idx]
            start_idx = end_idx

        return chunks

    def compute_delta_nfr_parallel(
        self, graph: Any, **kwargs: Any
    ) -> Dict[Any, float]:
        """Compute ΔNFR in parallel using fractal partitioning.

        Delegates to existing default_compute_delta_nfr with n_jobs parameter.
        This method exists primarily for API consistency with the proposal.

        Parameters
        ----------
        graph : TNFRGraph
            Network graph with structural attributes
        **kwargs
            Additional arguments passed to compute function

        Returns
        -------
        Dict[Any, float]
            Mapping from node IDs to ΔNFR values

        Notes
        -----
        Currently delegates to the existing implementation in
        tnfr.dynamics.dnfr.default_compute_delta_nfr which already supports
        n_jobs for parallelization. Future enhancements could use explicit
        partitioning strategies.
        """
        from ..dynamics.dnfr import default_compute_delta_nfr
        from ..constants.aliases import ALIAS_DNFR
        from ..alias import get_attr

        # Use existing parallel infrastructure
        kwargs.setdefault("n_jobs", self.max_workers)
        default_compute_delta_nfr(graph, **kwargs)

        # Extract results
        return {
            node_id: float(
                get_attr(graph.nodes[node_id], ALIAS_DNFR, 0.0) or 0.0
            )
            for node_id in graph.nodes()
        }

    def compute_si_parallel(
        self, graph: Any, **kwargs: Any
    ) -> Dict[Any, float]:
        """Compute sense index in parallel.

        Delegates to existing compute_Si with n_jobs parameter.
        This method exists primarily for API consistency with the proposal.

        Parameters
        ----------
        graph : TNFRGraph
            Network graph with structural attributes
        **kwargs
            Additional arguments passed to compute function

        Returns
        -------
        Dict[Any, float]
            Mapping from node IDs to Si values

        Notes
        -----
        Currently delegates to the existing implementation in
        tnfr.metrics.sense_index.compute_Si which already supports n_jobs for
        parallelization. Future enhancements could use explicit partitioning.
        """
        from ..metrics.sense_index import compute_Si

        # Use existing parallel infrastructure
        kwargs.setdefault("n_jobs", self.max_workers)
        kwargs.setdefault("inplace", False)
        return compute_Si(graph, **kwargs)

    def recommend_workers(self, graph_size: int) -> int:
        """Recommend optimal worker count for given graph size.

        Parameters
        ----------
        graph_size : int
            Number of nodes in the network

        Returns
        -------
        int
            Recommended number of workers

        Notes
        -----
        Uses heuristics:
        - Small graphs (<50 nodes): serial execution
        - Medium graphs: min(cpu_count, graph_size // 25)
        - Large graphs: full parallelism
        """
        if graph_size < 50:
            return 1  # Serial is faster for small graphs
        elif graph_size < 500:
            return min(self.max_workers, graph_size // 25)
        else:
            return self.max_workers
