"""Evidence-neutral execution-strategy recommendations for TNFR computations.

The size thresholds in this module are configuration policy. They do not
constitute benchmark evidence and therefore do not produce time, memory, or
speedup estimates.
"""

from __future__ import annotations

import math
from multiprocessing import cpu_count
from numbers import Real
from operator import index as integer_index
from typing import Any

# ---------------------------------------------------------------------------
# Efficiency alert thresholds
# ---------------------------------------------------------------------------
_PARALLELIZATION_EFFICIENCY_ALERT = 0.5
_MEMORY_EFFICIENCY_CRITICAL = 0.1


def _finite_nonnegative_metric(metrics: dict[str, Any], name: str) -> float | None:
    """Read one finite nonnegative observation without inventing evidence."""

    value = metrics.get(name)
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(float(value))
        or float(value) < 0.0
    ):
        return None
    return float(value)


class TNFRAutoScaler:
    """Apply a deterministic size policy to choose an execution strategy.

    The recommendation preserves the legacy result keys but reports performance
    estimates as None until a measured calibration interface is supplied.

    Examples
    --------
    >>> from tnfr.parallel import TNFRAutoScaler
    >>> scaler = TNFRAutoScaler()
    >>> strategy = scaler.recommend_execution_strategy(
    ...     graph_size=500,
    ...     available_memory_gb=8.0,
    ...     has_gpu=False
    ... )
    >>> strategy["backend"] in ["sequential", "multiprocessing"]
    True
    >>> strategy["estimated_time_minutes"] is None
    True
    """

    def __init__(self):
        self.performance_history: dict[str, Any] = {}
        self.optimal_configs: dict[str, Any] = {}

    def recommend_execution_strategy(
        self,
        graph_size: int,
        available_memory_gb: float = 8.0,
        has_gpu: bool = False,
    ) -> dict[str, Any]:
        """Recommend an execution route under the configured size policy.

        Parameters
        ----------
        graph_size : int
            Number of nodes in the network.
        available_memory_gb : float, default=8.0
            Retained for API compatibility. It cannot be compared with an
            estimate until measured memory evidence is available.
        has_gpu : bool, default=False
            Caller-declared GPU availability. It does not certify an
            accelerated TNFR kernel or a speedup.

        Returns
        -------
        dict[str, Any]
            Strategy recommendation with keys for the selected backend, worker
            count, policy explanation, explicit performance-evidence status,
            and nullable time and memory estimates.

        Notes
        -----
        Strategy selection follows configured size thresholds. These thresholds
        are routing policy rather than measured performance claims.
        """
        if isinstance(graph_size, bool):
            raise TypeError("graph_size must be a nonnegative integer")
        try:
            graph_size = integer_index(graph_size)
        except (TypeError, ValueError, OverflowError) as exc:
            raise TypeError("graph_size must be a nonnegative integer") from exc
        if graph_size < 0:
            raise ValueError("graph_size must be a nonnegative integer")
        if not isinstance(has_gpu, bool):
            raise TypeError("has_gpu must be a boolean availability observation")
        if (
            isinstance(available_memory_gb, bool)
            or not isinstance(available_memory_gb, Real)
            or not math.isfinite(float(available_memory_gb))
            or float(available_memory_gb) < 0.0
        ):
            raise ValueError("available_memory_gb must be finite and nonnegative")

        strategy: dict[str, Any] = {}

        if graph_size < 100:
            strategy["backend"] = "sequential"
            strategy["workers"] = 1
            strategy["explanation"] = (
                "Configured small-network policy selects sequential execution"
            )
        elif graph_size < 1000:
            strategy["backend"] = "multiprocessing"
            strategy["workers"] = max(1, min(cpu_count(), graph_size // 50))
            strategy["explanation"] = (
                "Configured medium-network policy selects multiprocessing"
            )
        else:
            strategy["backend"] = "distributed"
            strategy["workers"] = max(1, cpu_count() * 2)
            strategy["chunk_size"] = max(1, min(500, graph_size // 20))
            strategy["explanation"] = (
                "Configured large-network policy selects the distributed wrapper"
            )

        # Hardware presence does not establish a semantically compatible graph
        # kernel. Keep the legacy routing hints explicit and inactive until one
        # is supplied and benchmarked for the requested workload.
        strategy["declared_gpu_available"] = has_gpu
        strategy["use_gpu_strategies"] = False
        strategy["preferred_operators"] = []
        strategy["gpu_route_reason"] = (
            "no_verified_generic_tnfr_graph_kernel"
            if has_gpu
            else "gpu_not_declared_available"
        )

        # Graph size and a backend label cannot establish memory or runtime.
        # Keep the historical keys so callers can handle explicit abstention.
        strategy["estimated_memory_gb"] = self._estimate_memory_usage(
            graph_size, strategy["backend"]
        )
        strategy["estimated_time_minutes"] = self._estimate_execution_time(
            graph_size, strategy["backend"]
        )
        strategy["performance_evidence"] = "not_measured"
        return strategy

    def _estimate_memory_usage(self, graph_size: int, backend: str) -> None:
        """Return no estimate when no measured memory model is available.

        The arguments remain in this private signature for compatibility with
        existing subclasses.
        """
        del graph_size, backend
        return None

    def _estimate_execution_time(self, graph_size: int, backend: str) -> None:
        """Return no estimate when no measured timing model is available.

        The arguments remain in this private signature for compatibility with
        existing subclasses.
        """
        del graph_size, backend
        return None

    def get_optimization_suggestions(
        self, performance_metrics: dict[str, Any]
    ) -> list[str]:
        """Generate suggestions from supplied, observed performance metrics."""
        suggestions = []

        eff = _finite_nonnegative_metric(
            performance_metrics, "parallelization_efficiency"
        )
        if eff is not None and eff < _PARALLELIZATION_EFFICIENCY_ALERT:
            suggestions.append(
                "⚡ Low parallelization efficiency - consider reducing "
                "worker count or increasing partition size"
            )

        mem_eff = _finite_nonnegative_metric(performance_metrics, "memory_efficiency")
        if mem_eff is not None and mem_eff < _MEMORY_EFFICIENCY_CRITICAL:
            suggestions.append(
                "💾 High memory usage - consider distributed execution "
                "or memory optimization"
            )

        ops = _finite_nonnegative_metric(performance_metrics, "operations_per_second")
        if ops is not None and ops < 100:
            suggestions.append(
                "📈 Low throughput - profile supported backends and "
                "algorithm choices"
            )

        if not suggestions:
            suggestions.append(
                "No threshold-based suggestion was triggered by the supplied metrics"
            )

        return suggestions
