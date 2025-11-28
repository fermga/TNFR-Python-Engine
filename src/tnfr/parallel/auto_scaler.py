"""Auto-scaling and execution strategy recommendation for TNFR computations.

Recommends optimal execution strategies based on network size, available
resources, and hardware capabilities.
"""

from __future__ import annotations

from multiprocessing import cpu_count
from typing import Any, Dict, List


class TNFRAutoScaler:
    """Auto-scaler for TNFR parallel execution strategies.

    Analyzes network characteristics and system resources to recommend optimal
    execution strategies (sequential, multiprocessing, GPU, or distributed).

    Examples
    --------
    >>> from tnfr.parallel import TNFRAutoScaler
    >>> scaler = TNFRAutoScaler()
    >>> strategy = scaler.recommend_execution_strategy(
    ...     graph_size=500,
    ...     available_memory_gb=8.0,
    ...     has_gpu=False
    ... )
    >>> strategy['backend'] in ['sequential', 'multiprocessing']
    True
    """

    def __init__(self):
        self.performance_history: Dict[str, Any] = {}
        self.optimal_configs: Dict[str, Any] = {}

    def recommend_execution_strategy(
        self,
        graph_size: int,
        available_memory_gb: float = 8.0,
        has_gpu: bool = False,
    ) -> Dict[str, Any]:
        """Recommend optimal execution strategy for given configuration.

        Parameters
        ----------
        graph_size : int
            Number of nodes in the network
        available_memory_gb : float, default=8.0
            Available system memory in gigabytes
        has_gpu : bool, default=False
            Whether GPU acceleration is available

        Returns
        -------
        Dict[str, Any]
            Strategy recommendation with keys:
            - backend: str (sequential/multiprocessing/gpu/distributed)
            - workers: int (recommended worker count)
            - explanation: str (reasoning)
            - estimated_time_minutes: float (expected duration)
            - estimated_memory_gb: float (expected memory usage)

        Notes
        -----
        Strategy selection follows TNFR-aware heuristics:
        - Small networks (<100): Sequential is fastest (overhead dominates)
        - Medium networks (100-1000): Multiprocessing optimal
        - Large networks (1000-10000) with GPU: Vectorized GPU
        - Massive networks (>10000): Distributed computation required
        """
        strategy: Dict[str, Any] = {}

        # Select backend based on size
        if graph_size < 100:
            strategy["backend"] = "sequential"
            strategy["workers"] = 1
            strategy["explanation"] = (
                "Small network - sequential processing fastest due to overhead"
            )

        elif graph_size < 1000:
            strategy["backend"] = "multiprocessing"
            strategy["workers"] = min(cpu_count(), graph_size // 50)
            strategy["explanation"] = "Medium network - multiprocessing provides optimal speedup"

        elif graph_size < 10000 and has_gpu:
            strategy["backend"] = "gpu"
            strategy["workers"] = 1
            strategy["gpu_engine"] = "jax"
            strategy["explanation"] = "Large network with GPU - vectorized acceleration available"

        else:
            strategy["backend"] = "distributed"
            strategy["workers"] = cpu_count() * 2
            strategy["chunk_size"] = min(500, graph_size // 20)
            strategy["explanation"] = "Massive network - distributed computation recommended"

        # Estimate memory requirements
        estimated_memory = self._estimate_memory_usage(graph_size, strategy["backend"])
        strategy["estimated_memory_gb"] = estimated_memory

        # Check memory constraints
        if estimated_memory > available_memory_gb * 0.8:
            strategy["warning"] = (
                f"Estimated memory ({estimated_memory:.1f}GB) may exceed "
                f"available memory ({available_memory_gb:.1f}GB)"
            )
            strategy["recommendation"] = "Consider distributed backend or smaller partition sizes"

        # Estimate execution time
        estimated_time = self._estimate_execution_time(graph_size, strategy["backend"])
        strategy["estimated_time_minutes"] = estimated_time

        return strategy

    def _estimate_memory_usage(self, graph_size: int, backend: str) -> float:
        """Estimate memory usage in gigabytes.

        Parameters
        ----------
        graph_size : int
            Number of nodes
        backend : str
            Execution backend

        Returns
        -------
        float
            Estimated memory in GB
        """
        # Base memory: ~1KB per node for attributes
        base_memory_gb = graph_size * 0.001 / 1024

        # Backend multipliers account for overhead
        backend_multipliers = {
            "sequential": 1.0,
            "multiprocessing": 1.5,  # Serialization overhead
            "gpu": 2.0,  # GPU + CPU copies
            "distributed": 1.2,  # Network overhead minimal
        }

        multiplier = backend_multipliers.get(backend, 1.0)
        return base_memory_gb * multiplier

    def _estimate_execution_time(self, graph_size: int, backend: str) -> float:
        """Estimate execution time in minutes.

        Parameters
        ----------
        graph_size : int
            Number of nodes
        backend : str
            Execution backend

        Returns
        -------
        float
            Estimated time in minutes

        Notes
        -----
        Based on empirical observations. Actual times depend on:
        - Network density (edges per node)
        - Operator complexity
        - Hardware specifications
        - Cache efficiency
        """
        # Base time per 1000 nodes (calibrated with benchmarks)
        base_time_per_1k = {
            "sequential": 2.0,  # 2 min per 1000 nodes
            "multiprocessing": 0.5,  # 4x speedup typical
            "gpu": 0.1,  # 20x speedup on modern GPUs
            "distributed": 0.2,  # 10x speedup with cluster
        }

        time_factor = base_time_per_1k.get(backend, 2.0)
        return (graph_size / 1000.0) * time_factor

    def get_optimization_suggestions(self, performance_metrics: Dict[str, Any]) -> List[str]:
        """Generate optimization suggestions based on observed performance.

        Parameters
        ----------
        performance_metrics : Dict[str, Any]
            Performance data from execution monitoring

        Returns
        -------
        list[str]
            List of actionable optimization suggestions
        """
        suggestions = []

        # Check parallelization efficiency
        if "parallelization_efficiency" in performance_metrics:
            eff = performance_metrics["parallelization_efficiency"]
            if eff < 0.5:
                suggestions.append(
                    "⚡ Low parallelization efficiency - consider reducing "
                    "worker count or increasing partition size"
                )

        # Check memory usage
        if "memory_efficiency" in performance_metrics:
            mem_eff = performance_metrics["memory_efficiency"]
            if mem_eff < 0.1:
                suggestions.append(
                    "💾 High memory usage - consider distributed execution "
                    "or memory optimization"
                )

        # Check throughput
        if "operations_per_second" in performance_metrics:
            ops = performance_metrics["operations_per_second"]
            if ops < 100:
                suggestions.append(
                    "📈 Low throughput - consider GPU backend or algorithm " "optimization"
                )

        if not suggestions:
            suggestions.append("✨ Performance looks optimal!")

        return suggestions
