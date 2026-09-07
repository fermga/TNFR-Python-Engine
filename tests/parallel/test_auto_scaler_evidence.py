"""Evidence contracts for the public parallel strategy facade."""

from __future__ import annotations

import math

import pytest

import tnfr.parallel as parallel
from tnfr.parallel import (
    ParallelExecutionMonitor,
    TNFRAutoScaler,
    TNFRDistributedEngine,
)
from tnfr.parallel import monitoring as monitoring_module


def test_parallel_exports_resolve_every_declared_name() -> None:
    assert parallel.TNFRDistributedEngine is TNFRDistributedEngine
    assert all(hasattr(parallel, name) for name in parallel.__all__)


def test_auto_scaler_abstains_from_unmeasured_performance_estimates() -> None:
    strategy = TNFRAutoScaler().recommend_execution_strategy(
        graph_size=2_000,
        available_memory_gb=0.001,
        has_gpu=True,
    )

    assert strategy["backend"] == "distributed"
    assert strategy["declared_gpu_available"] is True
    assert strategy["use_gpu_strategies"] is False
    assert strategy["preferred_operators"] == []
    assert strategy["gpu_route_reason"] == "no_verified_generic_tnfr_graph_kernel"
    assert strategy["estimated_time_minutes"] is None
    assert strategy["estimated_memory_gb"] is None
    assert strategy["performance_evidence"] == "not_measured"
    assert "warning" not in strategy


def test_auto_scaler_default_suggestion_does_not_claim_optimality() -> None:
    suggestions = TNFRAutoScaler().get_optimization_suggestions({})

    assert suggestions == [
        "No threshold-based suggestion was triggered by the supplied metrics"
    ]


def _monitor_with_clock(
    monkeypatch: pytest.MonkeyPatch, *, start: float, end: float
) -> ParallelExecutionMonitor:
    clock = iter((start, end))
    monkeypatch.setattr(monitoring_module.time, "time", lambda: next(clock))
    monitor = ParallelExecutionMonitor()
    monitor._process = None
    return monitor


def test_monitor_abstains_from_speedup_without_a_sequential_baseline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monitor = _monitor_with_clock(monkeypatch, start=10.0, end=12.0)
    monitor.start_monitoring(expected_nodes=200, workers=4)
    metrics = monitor.stop_monitoring(
        final_coherence=0.8,
        initial_coherence=0.7,
    )

    assert metrics.duration_seconds == 2.0
    assert metrics.operations_per_second == 100.0
    assert metrics.speedup is None
    assert metrics.parallelization_efficiency is None
    assert (
        metrics.parallelization_efficiency_basis
        == "not_measured_no_sequential_baseline"
    )
    assert metrics.peak_memory_mb is None
    assert metrics.avg_cpu_percent is None
    assert metrics.memory_efficiency is None
    assert metrics.resource_metrics_available is False
    assert monitor.get_optimization_suggestions() == [
        "No threshold-based suggestion was triggered by the observed metrics"
    ]


def test_monitor_uses_an_explicit_sequential_baseline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monitor = _monitor_with_clock(monkeypatch, start=20.0, end=22.0)
    monitor.start_monitoring(expected_nodes=50, workers=4)
    metrics = monitor.stop_monitoring(
        final_coherence=0.9,
        initial_coherence=0.8,
        sequential_baseline_seconds=8.0,
    )

    assert metrics.speedup == 4.0
    assert metrics.parallelization_efficiency == 1.0
    assert (
        metrics.parallelization_efficiency_basis
        == "measured_sequential_baseline"
    )


@pytest.mark.parametrize("baseline", [0.0, -1.0, math.inf, math.nan, True])
def test_monitor_rejects_invalid_sequential_baselines(
    monkeypatch: pytest.MonkeyPatch, baseline: object
) -> None:
    monitor = _monitor_with_clock(monkeypatch, start=30.0, end=31.0)
    monitor.start_monitoring(expected_nodes=10, workers=2)

    with pytest.raises(ValueError, match="finite and positive"):
        monitor.stop_monitoring(
            final_coherence=0.5,
            initial_coherence=0.4,
            sequential_baseline_seconds=baseline,
        )


@pytest.mark.parametrize("graph_size", [-1, 1.5, True])
def test_auto_scaler_rejects_invalid_graph_sizes(graph_size: object) -> None:
    with pytest.raises((TypeError, ValueError), match="nonnegative integer"):
        TNFRAutoScaler().recommend_execution_strategy(graph_size=graph_size)


@pytest.mark.parametrize("available_memory_gb", [-1.0, math.inf, math.nan, True])
def test_auto_scaler_rejects_invalid_memory_observations(
    available_memory_gb: object,
) -> None:
    with pytest.raises(ValueError, match="finite and nonnegative"):
        TNFRAutoScaler().recommend_execution_strategy(
            graph_size=10,
            available_memory_gb=available_memory_gb,
        )


def test_auto_scaler_rejects_nonboolean_gpu_availability() -> None:
    with pytest.raises(TypeError, match="boolean availability"):
        TNFRAutoScaler().recommend_execution_strategy(graph_size=2_000, has_gpu=1)
@pytest.mark.parametrize("value", [None, math.inf, math.nan, True, "unknown"])
def test_auto_scaler_ignores_unobserved_or_invalid_performance_metrics(
    value: object,
) -> None:
    suggestions = TNFRAutoScaler().get_optimization_suggestions(
        {
            "parallelization_efficiency": value,
            "memory_efficiency": value,
            "operations_per_second": value,
        }
    )

    assert suggestions == [
        "No threshold-based suggestion was triggered by the supplied metrics"
    ]
@pytest.mark.parametrize(
    ("expected_nodes", "workers", "message"),
    [(-1, 1, "expected_nodes"), (1.5, 1, "expected_nodes"), (1, 0, "workers"),
     (1, True, "workers")],
)
def test_monitor_rejects_invalid_counts(
    expected_nodes: object,
    workers: object,
    message: str,
) -> None:
    monitor = ParallelExecutionMonitor()
    monitor._process = None

    with pytest.raises((TypeError, ValueError), match=message):
        monitor.start_monitoring(expected_nodes=expected_nodes, workers=workers)


@pytest.mark.parametrize("coherence", [-0.1, 1.1, math.inf, math.nan, True])
def test_monitor_rejects_invalid_coherence_observations(
    monkeypatch: pytest.MonkeyPatch,
    coherence: object,
) -> None:
    monitor = _monitor_with_clock(monkeypatch, start=40.0, end=41.0)
    monitor.start_monitoring(expected_nodes=10, workers=2)

    with pytest.raises(ValueError, match="finite coherence"):
        monitor.stop_monitoring(
            final_coherence=coherence,
            initial_coherence=0.5,
        )


def test_monitor_discards_nonfinite_resource_samples(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monitor = _monitor_with_clock(monkeypatch, start=50.0, end=51.0)
    monitor.start_monitoring(expected_nodes=10, workers=2)
    assert monitor._current_metrics is not None
    monitor._current_metrics["memory_samples"] = [math.nan, math.inf]
    monitor._current_metrics["cpu_samples"] = [math.nan, -1.0]

    metrics = monitor.stop_monitoring(final_coherence=0.6, initial_coherence=0.5)

    assert metrics.peak_memory_mb is None
    assert metrics.avg_cpu_percent is None
    assert metrics.memory_efficiency is None
    assert metrics.resource_metrics_available is False
