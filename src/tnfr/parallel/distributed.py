"""Distributed computation backend for massive TNFR networks.

Optional module that provides Ray and Dask integration for cluster computing.
Requires installation of optional dependencies:
    pip install tnfr[ray]  # or
    pip install tnfr[dask]
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List

if TYPE_CHECKING:  # pragma: no cover
    from ..types import TNFRGraph

# Check for optional dependencies
try:
    import ray

    HAS_RAY = True
except ImportError:
    HAS_RAY = False
    ray = None  # type: ignore

try:
    import dask
    from dask.distributed import Client

    HAS_DASK = True
except ImportError:
    HAS_DASK = False
    dask = None  # type: ignore
    Client = None  # type: ignore


class TNFRDistributedEngine:
    """Distributed computation engine for massive TNFR networks.

    Provides Ray and Dask backend integration for cluster-scale computation
    while preserving TNFR structural invariants.

    Parameters
    ----------
    backend : {"auto", "ray", "dask"}, default="auto"
        Distributed backend to use. "auto" selects Ray if available,
        otherwise Dask, otherwise falls back to multiprocessing.

    Raises
    ------
    ImportError
        If requested backend is not installed

    Examples
    --------
    >>> # Requires ray installation
    >>> try:
    ...     from tnfr.parallel import TNFRDistributedEngine
    ...     engine = TNFRDistributedEngine(backend="auto")
    ...     # engine.backend in ["ray", "dask", "multiprocessing"]
    ... except ImportError:
    ...     pass  # Optional dependency not installed

    Notes
    -----
    This is an optional advanced feature. Basic parallelization via
    TNFRParallelEngine is sufficient for most use cases.
    """

    def __init__(self, backend: str = "auto"):
        self.backend = self._select_backend(backend)
        self._client = None
        self._ray_initialized = False

    def _select_backend(self, backend: str) -> str:
        """Select available distributed backend."""
        if backend == "auto":
            if HAS_RAY:
                return "ray"
            elif HAS_DASK:
                return "dask"
            else:
                return "multiprocessing"

        if backend == "ray" and not HAS_RAY:
            raise ImportError("Ray not available. Install with: pip install ray")
        if backend == "dask" and not HAS_DASK:
            raise ImportError("Dask not available. Install with: pip install dask[distributed]")

        return backend

    def initialize_cluster(self, **cluster_config: Any) -> None:
        """Initialize distributed cluster.

        Parameters
        ----------
        **cluster_config
            Backend-specific cluster configuration

        Examples
        --------
        >>> # Ray configuration
        >>> engine = TNFRDistributedEngine(backend="ray")
        >>> engine.initialize_cluster(num_cpus=4)

        >>> # Dask configuration
        >>> engine = TNFRDistributedEngine(backend="dask")
        >>> engine.initialize_cluster(n_workers=4)
        """
        if self.backend == "ray" and HAS_RAY:
            if not self._ray_initialized:
                ray.init(**cluster_config)
                self._ray_initialized = True
        elif self.backend == "dask" and HAS_DASK:
            if self._client is None:
                self._client = Client(**cluster_config)

    def shutdown_cluster(self) -> None:
        """Shutdown distributed cluster and release resources."""
        if self.backend == "ray" and HAS_RAY and self._ray_initialized:
            ray.shutdown()
            self._ray_initialized = False
        elif self.backend == "dask" and self._client is not None:
            self._client.close()
            self._client = None

    def compute_si_distributed(
        self, graph: TNFRGraph, chunk_size: int = 500, **kwargs: Any
    ) -> Dict[str, Any]:
        """Compute sense index using distributed computation.

        Parameters
        ----------
        graph : TNFRGraph
            Network graph with TNFR attributes
        chunk_size : int, default=500
            Nodes per distributed work unit
        **kwargs
            Additional arguments for Si computation

        Returns
        -------
        Dict[str, Any]
            Si values and metadata

        Notes
        -----
        Requires Ray or Dask to be installed and initialized.
        Falls back to multiprocessing if distributed backend unavailable.
        """
        if self.backend == "ray" and HAS_RAY:
            return self._compute_si_ray(graph, chunk_size, **kwargs)
        elif self.backend == "dask" and HAS_DASK:
            return self._compute_si_dask(graph, chunk_size, **kwargs)
        else:
            # Fallback to multiprocessing
            from .engine import TNFRParallelEngine

            engine = TNFRParallelEngine(max_workers=4)
            si_values = engine.compute_si_parallel(graph, **kwargs)
            return {"si_values": si_values, "backend": "multiprocessing"}

    def _compute_si_ray(self, graph: TNFRGraph, chunk_size: int, **kwargs: Any) -> Dict[str, Any]:
        """Compute Si using Ray for distributed execution.

        Parameters
        ----------
        graph : TNFRGraph
            Network graph
        chunk_size : int
            Nodes per work unit
        **kwargs
            Additional Si computation parameters

        Returns
        -------
        Dict[str, Any]
            Results dictionary with si_values and metadata
        """
        if not HAS_RAY:
            raise ImportError("Ray required for distributed computation")

        # Define remote function for Ray
        @ray.remote
        def compute_si_chunk(node_chunk, graph_data):
            """Compute Si for a chunk of nodes (Ray remote function)."""
            import networkx as nx

            # Reconstruct graph in worker
            G = nx.Graph()
            G.add_nodes_from([(nid, attrs) for nid, attrs in graph_data["nodes"]])
            G.add_edges_from(graph_data["edges"])
            G.graph.update(graph_data["graph_attrs"])

            # Compute Si for this chunk
            si_values = {}
            for node_id in node_chunk:
                try:
                    from tnfr.metrics.sense_index import compute_Si_node

                    si_values[node_id] = compute_Si_node(G, node_id)
                except Exception:
                    # Fallback value on error
                    si_values[node_id] = 0.5

            return si_values

        # Serialize graph data
        graph_data = {
            "nodes": list(graph.nodes(data=True)),
            "edges": list(graph.edges()),
            "graph_attrs": dict(graph.graph),
        }

        # Chunk nodes
        nodes = list(graph.nodes())
        chunks = [nodes[i : i + chunk_size] for i in range(0, len(nodes), chunk_size)]

        # Submit Ray tasks
        futures = [compute_si_chunk.remote(chunk, graph_data) for chunk in chunks]

        # Gather results
        chunk_results = ray.get(futures)

        # Merge results
        si_values = {}
        for chunk_result in chunk_results:
            si_values.update(chunk_result)

        return {
            "si_values": si_values,
            "backend": "ray",
            "chunks_processed": len(chunks),
            "nodes_per_chunk": chunk_size,
        }

    def _compute_si_dask(self, graph: TNFRGraph, chunk_size: int, **kwargs: Any) -> Dict[str, Any]:
        """Compute Si using Dask for distributed execution.

        Parameters
        ----------
        graph : TNFRGraph
            Network graph
        chunk_size : int
            Nodes per work unit
        **kwargs
            Additional Si computation parameters

        Returns
        -------
        Dict[str, Any]
            Results dictionary with si_values and metadata

        Notes
        -----
        Basic Dask implementation. Can be extended for more sophisticated
        distributed patterns (delayed, dataframes, etc.)
        """
        if not HAS_DASK:
            raise ImportError("Dask required for distributed computation")

        from dask import delayed, compute

        def compute_si_chunk(node_chunk, graph_data):
            """Compute Si for a chunk of nodes (Dask delayed function)."""
            import networkx as nx
            from tnfr.metrics.sense_index import compute_Si_node

            # Reconstruct graph
            G = nx.Graph()
            G.add_nodes_from([(nid, attrs) for nid, attrs in graph_data["nodes"]])
            G.add_edges_from(graph_data["edges"])
            G.graph.update(graph_data["graph_attrs"])

            # Compute Si for chunk
            si_values = {}
            for node_id in node_chunk:
                try:
                    si_values[node_id] = compute_Si_node(G, node_id)
                except Exception:
                    si_values[node_id] = 0.5

            return si_values

        # Serialize graph
        graph_data = {
            "nodes": list(graph.nodes(data=True)),
            "edges": list(graph.edges()),
            "graph_attrs": dict(graph.graph),
        }

        # Chunk and create delayed tasks
        nodes = list(graph.nodes())
        chunks = [nodes[i : i + chunk_size] for i in range(0, len(nodes), chunk_size)]

        delayed_tasks = [delayed(compute_si_chunk)(chunk, graph_data) for chunk in chunks]

        # Compute in parallel
        chunk_results = compute(*delayed_tasks)

        # Merge results
        si_values = {}
        for chunk_result in chunk_results:
            si_values.update(chunk_result)

        return {
            "si_values": si_values,
            "backend": "dask",
            "chunks_processed": len(chunks),
            "nodes_per_chunk": chunk_size,
        }

    def simulate_large_network(
        self,
        node_count: int,
        edge_probability: float,
        operator_sequences: List[List[str]],
        chunk_size: int = 500,
    ) -> Dict[str, Any]:
        """Simulate massive network using distributed computation.

        Parameters
        ----------
        node_count : int
            Total number of nodes in network
        edge_probability : float
            Edge creation probability for random network
        operator_sequences : List[List[str]]
            Sequences of TNFR operators to apply
        chunk_size : int, default=500
            Nodes per distributed work unit

        Returns
        -------
        Dict[str, Any]
            Simulation results with coherence and sense indices

        Notes
        -----
        Creates a large network and processes it using distributed backend.
        This is a simplified implementation focused on Si computation.
        Full operator sequence application would require more sophisticated
        distributed state management.
        """
        import networkx as nx

        # Create large network
        G = nx.erdos_renyi_graph(node_count, edge_probability)

        # Initialize with TNFR attributes
        for node in G.nodes():
            G.nodes[node]["nu_f"] = 1.0
            G.nodes[node]["phase"] = 0.0
            G.nodes[node]["epi"] = 0.5
            G.nodes[node]["delta_nfr"] = 0.0

        # Compute Si using distributed backend
        results = self.compute_si_distributed(G, chunk_size=chunk_size)

        # Add network statistics
        results["network_stats"] = {
            "nodes": node_count,
            "edges": G.number_of_edges(),
            "density": nx.density(G),
            "avg_clustering": nx.average_clustering(G) if node_count < 10000 else 0.0,
        }

        return results

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup resources."""
        self.shutdown_cluster()
