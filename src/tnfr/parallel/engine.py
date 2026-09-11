"""Parallel execution facade for canonical TNFR computations.

The facade delegates to the existing ΔNFR and Si implementations. Worker counts
are requests to those implementations; this module does not infer or claim a
measured acceleration from the request alone.
"""

from __future__ import annotations

import math
from multiprocessing import cpu_count
from numbers import Integral, Real
from typing import Any, Mapping

from .partitioner import FractalPartitioner


def _positive_worker_count(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be a positive integer")
    resolved = int(value)
    if resolved <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return resolved


def _finite_result_mapping(values: Any, *, label: str) -> dict[Any, float]:
    if not isinstance(values, Mapping):
        raise RuntimeError(f"{label} must return a node-to-value mapping")
    result: dict[Any, float] = {}
    for node, value in values.items():
        if isinstance(value, bool) or not isinstance(value, Real):
            raise RuntimeError(f"{label} produced a non-real value for node {node!r}")
        scalar = float(value)
        if not math.isfinite(scalar):
            raise RuntimeError(f"{label} produced a non-finite value for node {node!r}")
        result[node] = scalar
    return result


class TNFRParallelEngine:
    """Validated facade over the canonical parallel-capable computations.

    ``execution_mode`` is retained as compatibility metadata. The delegated
    routines select their actual vectorized or process execution path from the
    graph and ``n_jobs``; this facade therefore makes no thread/process claim.
    """

    def __init__(
        self,
        max_workers: int | None = None,
        execution_mode: str = "threads",
        partition_size: int = 100,
        cache_aware: bool = True,
    ):
        resolved_workers = cpu_count() if max_workers is None else max_workers
        self.max_workers = _positive_worker_count(resolved_workers, "max_workers")
        if execution_mode not in {"threads", "processes"}:
            raise ValueError("execution_mode must be 'threads' or 'processes'")
        if not isinstance(cache_aware, bool):
            raise TypeError("cache_aware must be boolean")
        self.execution_mode = execution_mode
        self.cache_aware = cache_aware
        self.partitioner = FractalPartitioner(max_partition_size=partition_size)

    def _distribute_work_cache_aware(self, partitions: list, num_workers: int) -> list:
        """Group partitions deterministically by structural-frequency affinity.

        The historical method name is retained for compatibility. The grouping
        is a scheduling heuristic; no cache-hit or runtime improvement is
        asserted without measurements.
        """
        workers = _positive_worker_count(num_workers, "num_workers")
        if not self.cache_aware or len(partitions) <= workers:
            chunks: list = [[] for _ in range(workers)]
            for index, partition in enumerate(partitions):
                chunks[index % workers].append(partition)
            return chunks

        from ..alias import get_attr
        from ..constants.aliases import ALIAS_VF

        def partition_center(partition_info: Any) -> float:
            node_set, subgraph = partition_info
            if not node_set:
                return 0.0
            values: list[float] = []
            for node in node_set:
                raw = get_attr(subgraph.nodes[node], ALIAS_VF, 1.0)
                if isinstance(raw, bool) or not isinstance(raw, Real):
                    raise ValueError("partition nu_f values must be finite nonnegative reals")
                value = float(raw)
                if not math.isfinite(value) or value < 0.0:
                    raise ValueError("partition nu_f values must be finite nonnegative reals")
                values.append(value)
            return math.fsum(values) / len(values)

        sorted_partitions = sorted(partitions, key=partition_center)
        chunks = [[] for _ in range(workers)]
        chunk_size, remainder = divmod(len(sorted_partitions), workers)
        start = 0
        for worker_id in range(workers):
            end = start + chunk_size + (1 if worker_id < remainder else 0)
            chunks[worker_id] = sorted_partitions[start:end]
            start = end
        return chunks

    def compute_delta_nfr_parallel(self, graph: Any, **kwargs: Any) -> dict[Any, float]:
        """Compute ΔNFR through the canonical implementation with a worker hint."""
        from ..alias import get_attr
        from ..constants.aliases import ALIAS_DNFR
        from ..dynamics.dnfr import default_compute_delta_nfr

        if "n_jobs" in kwargs and kwargs["n_jobs"] is not None:
            kwargs["n_jobs"] = _positive_worker_count(kwargs["n_jobs"], "n_jobs")
        else:
            kwargs["n_jobs"] = self.max_workers
        default_compute_delta_nfr(graph, **kwargs)
        raw = {node: get_attr(graph.nodes[node], ALIAS_DNFR, 0.0) for node in graph}
        return _finite_result_mapping(raw, label="default_compute_delta_nfr")

    def compute_si_parallel(self, graph: Any, **kwargs: Any) -> dict[Any, float]:
        """Compute Si through the canonical implementation with a worker hint."""
        from ..metrics.sense_index import compute_Si

        if kwargs.get("inplace") not in (None, False):
            raise ValueError("compute_si_parallel requires inplace=False")
        kwargs["inplace"] = False
        if "n_jobs" in kwargs and kwargs["n_jobs"] is not None:
            kwargs["n_jobs"] = _positive_worker_count(kwargs["n_jobs"], "n_jobs")
        else:
            kwargs["n_jobs"] = self.max_workers
        values = compute_Si(graph, **kwargs)
        return _finite_result_mapping(values, label="compute_Si")

    def recommend_workers(self, graph_size: int) -> int:
        """Return the documented size-based worker heuristic."""
        if isinstance(graph_size, bool) or not isinstance(graph_size, Integral):
            raise TypeError("graph_size must be a nonnegative integer")
        size = int(graph_size)
        if size < 0:
            raise ValueError("graph_size must be a nonnegative integer")
        if size < 50:
            return 1
        if size < 500:
            return min(self.max_workers, max(1, size // 25))
        return self.max_workers
