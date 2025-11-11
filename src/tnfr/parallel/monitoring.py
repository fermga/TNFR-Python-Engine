"""Performance monitoring for parallel TNFR computations.

Tracks execution metrics to enable optimization and auto-scaling decisions.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

try:
    pass

    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


@dataclass
class PerformanceMetrics:
    """Performance metrics for parallel TNFR execution.

    Attributes
    ----------
    start_time : float
        Unix timestamp when execution started
    end_time : float
        Unix timestamp when execution completed
    duration_seconds : float
        Total execution time in seconds
    peak_memory_mb : float
        Peak memory usage in megabytes
    avg_cpu_percent : float
        Average CPU utilization percentage
    workers_used : int
        Number of parallel workers employed
    nodes_processed : int
        Total number of nodes processed
    operations_per_second : float
        Throughput metric (nodes/second)
    coherence_improvement : float
        Change in global coherence C(t)
    parallelization_efficiency : float
        Actual speedup / theoretical speedup ratio
    memory_efficiency : float
        Useful work / total memory ratio
    """

    start_time: float
    end_time: float
    duration_seconds: float
    peak_memory_mb: float
    avg_cpu_percent: float
    workers_used: int
    nodes_processed: int
    operations_per_second: float
    coherence_improvement: float
    parallelization_efficiency: float
    memory_efficiency: float


class ParallelExecutionMonitor:
    """Real-time monitoring for parallel TNFR execution.

    Tracks resource usage, throughput, and efficiency metrics during parallel
    computation to enable dynamic optimization and post-execution analysis.

    Examples
    --------
    >>> from tnfr.parallel import ParallelExecutionMonitor
    >>> monitor = ParallelExecutionMonitor()
    >>> monitor.start_monitoring(expected_nodes=100, workers=2)
    >>> # ... perform computation ...
    >>> metrics = monitor.stop_monitoring(
    ...     final_coherence=0.85,
    ...     initial_coherence=0.75
    ... )
    >>> metrics.nodes_processed
    100
    >>> metrics.workers_used
    2
    """

    def __init__(self):
        self._metrics_history: List[PerformanceMetrics] = []
        self._current_metrics: Optional[Dict[str, Any]] = None
        self._process = None
        if HAS_PSUTIL:
            try:
                import psutil

                self._process = psutil.Process()
            except Exception:
                self._process = None

    def start_monitoring(self, expected_nodes: int, workers: int) -> None:
        """Start monitoring execution.

        Parameters
        ----------
        expected_nodes : int
            Expected number of nodes to process
        workers : int
            Number of parallel workers
        """
        self._current_metrics = {
            "start_time": time.time(),
            "expected_nodes": expected_nodes,
            "workers": workers,
            "memory_samples": [],
            "cpu_samples": [],
        }

        # Take initial resource snapshot
        if self._process:
            try:
                mem_info = self._process.memory_info()
                self._current_metrics["memory_samples"].append(mem_info.rss / 1024 / 1024)
                self._current_metrics["cpu_samples"].append(self._process.cpu_percent())
            except Exception:
                pass

    def stop_monitoring(
        self, final_coherence: float, initial_coherence: float
    ) -> PerformanceMetrics:
        """Stop monitoring and compute final metrics.

        Parameters
        ----------
        final_coherence : float
            Final network coherence C(t)
        initial_coherence : float
            Initial network coherence C(t)

        Returns
        -------
        PerformanceMetrics
            Complete performance metrics for the execution
        """
        if self._current_metrics is None:
            raise RuntimeError("Monitoring not started")

        end_time = time.time()
        duration = end_time - self._current_metrics["start_time"]

        # Take final resource snapshot
        if self._process:
            try:
                mem_info = self._process.memory_info()
                self._current_metrics["memory_samples"].append(mem_info.rss / 1024 / 1024)
                self._current_metrics["cpu_samples"].append(self._process.cpu_percent())
            except Exception:
                pass

        # Calculate aggregated metrics
        memory_samples = self._current_metrics.get("memory_samples", [])
        cpu_samples = self._current_metrics.get("cpu_samples", [])

        peak_memory = max(memory_samples) if memory_samples else 0.0
        avg_cpu = sum(cpu_samples) / len(cpu_samples) if cpu_samples else 0.0

        nodes = self._current_metrics["expected_nodes"]
        workers = self._current_metrics["workers"]

        # Calculate parallelization efficiency
        # NOTE: This is a heuristic approximation. True efficiency requires:
        # - Baseline sequential measurement
        # - Accounting for Amdahl's law (sequential portions)
        # - Consideration of communication overhead
        # Current approach: estimate from CPU utilization as proxy
        theoretical_speedup = workers
        # Estimate actual speedup from CPU utilization
        # If we're using N workers, we expect ~N * 100% CPU in ideal case
        expected_cpu = workers * 100.0
        actual_speedup = (avg_cpu / 100.0) if expected_cpu > 0 else 1.0
        parallelization_eff = (
            min(1.0, actual_speedup / theoretical_speedup) if theoretical_speedup > 0 else 0.0
        )

        # Memory efficiency: nodes per MB
        memory_eff = nodes / peak_memory if peak_memory > 0 else 0.0

        metrics = PerformanceMetrics(
            start_time=self._current_metrics["start_time"],
            end_time=end_time,
            duration_seconds=duration,
            peak_memory_mb=peak_memory,
            avg_cpu_percent=avg_cpu,
            workers_used=workers,
            nodes_processed=nodes,
            operations_per_second=nodes / duration if duration > 0 else 0.0,
            coherence_improvement=final_coherence - initial_coherence,
            parallelization_efficiency=parallelization_eff,
            memory_efficiency=memory_eff,
        )

        self._metrics_history.append(metrics)
        self._current_metrics = None

        return metrics

    def get_optimization_suggestions(self) -> List[str]:
        """Generate optimization suggestions based on execution history.

        Returns
        -------
        List[str]
            List of actionable suggestions for improving performance
        """
        if not self._metrics_history:
            return ["No execution history available"]

        latest = self._metrics_history[-1]
        suggestions = []

        if latest.parallelization_efficiency < 0.5:
            suggestions.append(
                "⚡ Low parallelization efficiency - consider reducing "
                "worker count or increasing chunk size"
            )

        if latest.memory_efficiency < 0.1:
            suggestions.append(
                "💾 High memory usage - consider distributed execution " "or memory optimization"
            )

        if latest.operations_per_second < 100:
            suggestions.append(
                "📈 Low throughput - consider GPU backend or algorithm " "optimization"
            )

        if not suggestions:
            suggestions.append("✨ Performance looks optimal!")

        return suggestions

    @property
    def history(self) -> List[PerformanceMetrics]:
        """Get execution history."""
        return self._metrics_history.copy()
