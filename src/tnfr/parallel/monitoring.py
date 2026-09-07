"""Measured performance telemetry for parallel TNFR computations.

Resource samples and throughput describe one execution. Speedup and parallel
efficiency require a caller-supplied sequential baseline for the same workload;
CPU utilization alone is not such a baseline.
"""

from __future__ import annotations

import math
from numbers import Real
from operator import index as integer_index
import time
from dataclasses import dataclass
from typing import Any

try:
    import psutil
except ImportError:  # pragma: no cover - optional dependency
    psutil = None  # type: ignore[assignment]

HAS_PSUTIL = psutil is not None

# ---------------------------------------------------------------------------
# Efficiency alert thresholds
# ---------------------------------------------------------------------------
_PARALLELIZATION_EFFICIENCY_ALERT = 0.5
_MEMORY_EFFICIENCY_CRITICAL = 0.1


def _integer_count(value: Any, *, name: str, positive: bool) -> int:
    """Validate a non-boolean integer count at the monitoring boundary."""

    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    try:
        result = integer_index(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError(f"{name} must be an integer") from exc
    if result < (1 if positive else 0):
        qualifier = "positive" if positive else "nonnegative"
        raise ValueError(f"{name} must be {qualifier}")
    return result


def _finite_coherence(value: Any, *, name: str) -> float:
    """Validate one canonical C(t) endpoint observation."""

    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(float(value))
        or not 0.0 <= float(value) <= 1.0
    ):
        raise ValueError(f"{name} must be a finite coherence in [0, 1]")
    return float(value)


def _finite_nonnegative_samples(values: Any) -> list[float]:
    """Discard unavailable resource samples without creating zero evidence."""

    samples: list[float] = []
    for value in values or ():
        if (
            not isinstance(value, bool)
            and isinstance(value, Real)
            and math.isfinite(float(value))
            and float(value) >= 0.0
        ):
            samples.append(float(value))
    return samples


@dataclass
class PerformanceMetrics:
    """Performance observations for one parallel TNFR execution.

    parallelization_efficiency is speedup divided by workers and remains None
    without a measured sequential baseline for the same workload. Resource
    fields remain None when the optional process sampler supplies no
    observations. memory_efficiency has units nodes/MB.
    """

    start_time: float
    end_time: float
    duration_seconds: float
    peak_memory_mb: float | None
    avg_cpu_percent: float | None
    workers_used: int
    nodes_processed: int
    operations_per_second: float
    coherence_improvement: float
    parallelization_efficiency: float | None
    memory_efficiency: float | None
    speedup: float | None = None
    parallelization_efficiency_basis: str = (
        "not_measured_no_sequential_baseline"
    )
    resource_metrics_available: bool = False


class ParallelExecutionMonitor:
    """Collect execution telemetry without inferring unobserved speedups.

    Examples
    --------
    >>> from tnfr.parallel import ParallelExecutionMonitor
    >>> monitor = ParallelExecutionMonitor()
    >>> monitor.start_monitoring(expected_nodes=100, workers=2)
    >>> metrics = monitor.stop_monitoring(
    ...     final_coherence=0.85,
    ...     initial_coherence=0.75
    ... )
    >>> metrics.nodes_processed
    100
    >>> metrics.parallelization_efficiency is None
    True
    """

    def __init__(self):
        self._metrics_history: list[PerformanceMetrics] = []
        self._current_metrics: dict[str, Any] | None = None
        self._process = None
        if HAS_PSUTIL:
            try:
                self._process = psutil.Process()
            except Exception:
                self._process = None

    def start_monitoring(self, expected_nodes: int, workers: int) -> None:
        """Start collecting observations for one execution."""
        expected_nodes = _integer_count(
            expected_nodes, name="expected_nodes", positive=False
        )
        workers = _integer_count(workers, name="workers", positive=True)
        self._current_metrics = {
            "start_time": time.time(),
            "expected_nodes": expected_nodes,
            "workers": workers,
            "memory_samples": [],
            "cpu_samples": [],
        }

        if self._process:
            try:
                mem_info = self._process.memory_info()
                self._current_metrics["memory_samples"].append(
                    mem_info.rss / 1024 / 1024
                )
                self._current_metrics["cpu_samples"].append(
                    self._process.cpu_percent()
                )
            except Exception:
                pass

    def stop_monitoring(
        self,
        final_coherence: float,
        initial_coherence: float,
        *,
        sequential_baseline_seconds: float | None = None,
    ) -> PerformanceMetrics:
        """Stop monitoring and return measured metrics.

        Parameters
        ----------
        final_coherence, initial_coherence
            Endpoint C(t) observations for this execution.
        sequential_baseline_seconds
            Measured duration of the same workload under sequential execution.
            When omitted, speedup and parallelization efficiency are unavailable.

        Raises
        ------
        RuntimeError
            If monitoring has not started.
        ValueError
            If a supplied sequential baseline is not finite and positive.
        """
        if self._current_metrics is None:
            raise RuntimeError("Monitoring not started")
        final_coherence = _finite_coherence(
            final_coherence, name="final_coherence"
        )
        initial_coherence = _finite_coherence(
            initial_coherence, name="initial_coherence"
        )
        if sequential_baseline_seconds is not None:
            if (
                isinstance(sequential_baseline_seconds, bool)
                or not isinstance(sequential_baseline_seconds, Real)
                or not math.isfinite(float(sequential_baseline_seconds))
                or float(sequential_baseline_seconds) <= 0.0
            ):
                raise ValueError(
                    "sequential_baseline_seconds must be finite and positive"
                )
            sequential_baseline_seconds = float(sequential_baseline_seconds)

        end_time = time.time()
        duration = end_time - self._current_metrics["start_time"]

        if self._process:
            try:
                mem_info = self._process.memory_info()
                self._current_metrics["memory_samples"].append(
                    mem_info.rss / 1024 / 1024
                )
                self._current_metrics["cpu_samples"].append(
                    self._process.cpu_percent()
                )
            except Exception:
                pass

        memory_samples = _finite_nonnegative_samples(
            self._current_metrics.get("memory_samples", [])
        )
        cpu_samples = _finite_nonnegative_samples(
            self._current_metrics.get("cpu_samples", [])
        )
        peak_memory = max(memory_samples) if memory_samples else None
        avg_cpu = (
            sum(cpu_samples) / len(cpu_samples) if cpu_samples else None
        )

        nodes = self._current_metrics["expected_nodes"]
        workers = self._current_metrics["workers"]

        speedup: float | None = None
        parallelization_eff: float | None = None
        efficiency_basis = "not_measured_no_sequential_baseline"
        if sequential_baseline_seconds is not None:
            if duration > 0.0:
                candidate_speedup = sequential_baseline_seconds / duration
                candidate_efficiency = candidate_speedup / workers
                if math.isfinite(candidate_speedup) and math.isfinite(
                    candidate_efficiency
                ):
                    speedup = candidate_speedup
                    parallelization_eff = candidate_efficiency
                    efficiency_basis = "measured_sequential_baseline"
                else:
                    efficiency_basis = "unavailable_nonfinite_ratio"
            else:
                efficiency_basis = "unavailable_nonpositive_duration"

        memory_eff = (
            nodes / peak_memory
            if peak_memory is not None and peak_memory > 0.0
            else None
        )

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
            speedup=speedup,
            parallelization_efficiency_basis=efficiency_basis,
            resource_metrics_available=bool(memory_samples or cpu_samples),
        )

        self._metrics_history.append(metrics)
        self._current_metrics = None
        return metrics

    def get_optimization_suggestions(self) -> list[str]:
        """Generate threshold-based suggestions from observed execution history."""
        if not self._metrics_history:
            return ["No execution history available"]

        latest = self._metrics_history[-1]
        suggestions = []

        if (
            latest.parallelization_efficiency is not None
            and latest.parallelization_efficiency
            < _PARALLELIZATION_EFFICIENCY_ALERT
        ):
            suggestions.append(
                "⚡ Low measured parallelization efficiency - consider reducing "
                "worker count or increasing chunk size"
            )

        if (
            latest.memory_efficiency is not None
            and latest.memory_efficiency < _MEMORY_EFFICIENCY_CRITICAL
        ):
            suggestions.append(
                "💾 High observed memory use per node - consider distributed "
                "execution or memory optimization"
            )

        if latest.operations_per_second < 100:
            suggestions.append(
                "📈 Low measured throughput - profile supported backends and "
                "algorithm choices"
            )

        if not suggestions:
            suggestions.append(
                "No threshold-based suggestion was triggered by the observed metrics"
            )

        return suggestions

    @property
    def history(self) -> list[PerformanceMetrics]:
        """Return a detached list of recorded execution metrics."""
        return self._metrics_history.copy()
