"""Optional distributed scheduling for canonical TNFR Si computation.

Ray and Dask are optional. The local fallback delegates to :func:`compute_Si`
and reports that implementation explicitly because a worker request alone does
not prove that a multiprocessing path executed.
"""

from __future__ import annotations

import math
from multiprocessing import cpu_count
from numbers import Integral, Real
from typing import TYPE_CHECKING, Any, Mapping

if TYPE_CHECKING:  # pragma: no cover
    from ..types import TNFRGraph

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

_VALID_BACKENDS = frozenset({"auto", "ray", "dask", "multiprocessing"})


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be a positive integer")
    resolved = int(value)
    if resolved <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return resolved


def _distributed_compute_kwargs(kwargs: Mapping[str, Any]) -> dict[str, Any]:
    """Return kwargs that avoid nested pools and remote mutation claims."""
    normalized = dict(kwargs)
    if normalized.get("inplace") not in (None, False):
        raise ValueError("distributed Si computation requires inplace=False")
    if normalized.get("n_jobs") not in (None, 1):
        raise ValueError("distributed Si workers require n_jobs=1")
    if normalized.get("profile") is not None:
        raise ValueError("distributed Si computation cannot return a shared profile")
    normalized["inplace"] = False
    normalized["n_jobs"] = 1
    normalized.pop("profile", None)
    return normalized


def _compute_si_chunk(
    node_chunk: list[Any], graph: Any, compute_kwargs: Mapping[str, Any]
) -> dict[Any, float]:
    """Compute globally normalized Si, then select one disjoint node chunk."""
    from ..metrics.sense_index import compute_Si

    values = compute_Si(graph, **dict(compute_kwargs))
    if not isinstance(values, Mapping):
        raise RuntimeError("compute_Si must return a node-to-value mapping")
    result: dict[Any, float] = {}
    for node in node_chunk:
        if node not in values:
            raise RuntimeError(f"compute_Si omitted requested node {node!r}")
        raw = values[node]
        if isinstance(raw, bool) or not isinstance(raw, Real):
            raise RuntimeError(f"compute_Si produced a non-real value for node {node!r}")
        value = float(raw)
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise RuntimeError(
                f"compute_Si produced an invalid value for node {node!r}"
            )
        result[node] = value
    return result


def _merge_si_chunks(
    expected_nodes: list[Any], chunk_results: list[Mapping[Any, Any]] | tuple[Mapping[Any, Any], ...]
) -> dict[Any, float]:
    """Merge disjoint chunks and reject duplicate, missing, or foreign nodes."""
    expected = set(expected_nodes)
    merged: dict[Any, float] = {}
    for chunk in chunk_results:
        if not isinstance(chunk, Mapping):
            raise RuntimeError("distributed Si worker returned a non-mapping result")
        for node, raw in chunk.items():
            if node not in expected:
                raise RuntimeError(f"distributed Si worker returned foreign node {node!r}")
            if node in merged:
                raise RuntimeError(f"distributed Si worker duplicated node {node!r}")
            if isinstance(raw, bool) or not isinstance(raw, Real):
                raise RuntimeError(f"distributed Si worker returned a non-real value for {node!r}")
            value = float(raw)
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise RuntimeError(f"distributed Si worker returned an invalid value for {node!r}")
            merged[node] = value
    missing = [node for node in expected_nodes if node not in merged]
    if missing:
        raise RuntimeError(f"distributed Si workers omitted nodes: {missing!r}")
    return {node: merged[node] for node in expected_nodes}


class TNFRDistributedEngine:
    """Run canonical Si computation through an optional distributed scheduler."""

    def __init__(self, backend: str = "auto"):
        self.backend = self._select_backend(backend)
        self._client = None
        self._ray_initialized = False

    def _select_backend(self, backend: str) -> str:
        if not isinstance(backend, str) or backend not in _VALID_BACKENDS:
            choices = ", ".join(sorted(_VALID_BACKENDS))
            raise ValueError(f"backend must be one of: {choices}")
        if backend == "auto":
            if HAS_RAY:
                return "ray"
            if HAS_DASK:
                return "dask"
            return "multiprocessing"
        if backend == "ray" and not HAS_RAY:
            raise ImportError("Ray not available. Install with: pip install ray")
        if backend == "dask" and not HAS_DASK:
            raise ImportError("Dask not available. Install with: pip install dask[distributed]")
        return backend

    def initialize_cluster(self, **cluster_config: Any) -> None:
        """Initialize the selected optional scheduler."""
        if self.backend == "ray" and HAS_RAY:
            if not ray.is_initialized():
                ray.init(**cluster_config)
                self._ray_initialized = True
        elif self.backend == "dask" and HAS_DASK and self._client is None:
            self._client = Client(**cluster_config)
        elif cluster_config:
            raise ValueError("the local compute_Si fallback accepts no cluster configuration")

    def shutdown_cluster(self) -> None:
        """Release scheduler resources owned by this engine."""
        if self.backend == "ray" and HAS_RAY and self._ray_initialized:
            ray.shutdown()
            self._ray_initialized = False
        elif self.backend == "dask" and self._client is not None:
            self._client.close()
            self._client = None

    def compute_si_distributed(
        self, graph: TNFRGraph, chunk_size: int = 500, **kwargs: Any
    ) -> dict[str, Any]:
        """Compute Si with complete, disjoint chunk validation."""
        size = _positive_int(chunk_size, "chunk_size")
        if self.backend == "ray":
            return self._compute_si_ray(graph, size, **kwargs)
        if self.backend == "dask":
            return self._compute_si_dask(graph, size, **kwargs)

        from .engine import TNFRParallelEngine

        engine = TNFRParallelEngine()
        values = engine.compute_si_parallel(graph, **kwargs)
        return {
            "si_values": values,
            "backend": "compute_Si",
            "selected_backend": "multiprocessing",
            "configured_workers": engine.max_workers,
            "parallel_execution_observed": None,
        }

    def _compute_si_ray(
        self, graph: TNFRGraph, chunk_size: int, **kwargs: Any
    ) -> dict[str, Any]:
        if not HAS_RAY:
            raise ImportError("Ray required for distributed computation")
        compute_kwargs = _distributed_compute_kwargs(kwargs)
        nodes = list(graph)
        chunks = [nodes[i : i + chunk_size] for i in range(0, len(nodes), chunk_size)]
        remote_worker = ray.remote(_compute_si_chunk)
        futures = [remote_worker.remote(chunk, graph, compute_kwargs) for chunk in chunks]
        values = _merge_si_chunks(nodes, ray.get(futures))
        return {
            "si_values": values,
            "backend": "ray",
            "chunks_processed": len(chunks),
            "nodes_per_chunk": chunk_size,
        }

    def _compute_si_dask(
        self, graph: TNFRGraph, chunk_size: int, **kwargs: Any
    ) -> dict[str, Any]:
        if not HAS_DASK:
            raise ImportError("Dask required for distributed computation")
        from dask import compute, delayed

        compute_kwargs = _distributed_compute_kwargs(kwargs)
        nodes = list(graph)
        chunks = [nodes[i : i + chunk_size] for i in range(0, len(nodes), chunk_size)]
        tasks = [delayed(_compute_si_chunk)(chunk, graph, compute_kwargs) for chunk in chunks]
        values = _merge_si_chunks(nodes, compute(*tasks))
        return {
            "si_values": values,
            "backend": "dask",
            "chunks_processed": len(chunks),
            "nodes_per_chunk": chunk_size,
        }

    def simulate_large_network(
        self,
        node_count: int,
        edge_probability: float,
        operator_sequences: list[list[str]],
        chunk_size: int = 500,
        *,
        seed: int | None = 0,
    ) -> dict[str, Any]:
        """Create a reproducible graph and compute Si without applying operators.

        Non-empty operator sequences are rejected because this wrapper has no
        distributed atomic operator protocol.
        """
        import networkx as nx

        if isinstance(node_count, bool) or not isinstance(node_count, Integral):
            raise TypeError("node_count must be a nonnegative integer")
        count = int(node_count)
        if count < 0:
            raise ValueError("node_count must be a nonnegative integer")
        if isinstance(edge_probability, bool) or not isinstance(edge_probability, Real):
            raise TypeError("edge_probability must be a finite real in [0, 1]")
        probability = float(edge_probability)
        if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError("edge_probability must be a finite real in [0, 1]")
        if seed is not None and (isinstance(seed, bool) or not isinstance(seed, Integral)):
            raise TypeError("seed must be an integer or None")
        if not isinstance(operator_sequences, list):
            raise TypeError("operator_sequences must be a list")
        if operator_sequences:
            raise NotImplementedError(
                "distributed operator sequences require an atomic cross-partition protocol"
            )

        graph = nx.erdos_renyi_graph(count, probability, seed=seed)
        for node in graph:
            graph.nodes[node].update(nu_f=1.0, phase=0.0, EPI=0.5, delta_nfr=0.0)
        results = self.compute_si_distributed(graph, chunk_size=chunk_size)
        clustering = nx.average_clustering(graph) if count < 10_000 else None
        results["network_stats"] = {
            "nodes": count,
            "edges": graph.number_of_edges(),
            "density": nx.density(graph),
            "avg_clustering": clustering,
            "avg_clustering_observed": clustering is not None,
            "seed": seed,
        }
        return results

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown_cluster()
