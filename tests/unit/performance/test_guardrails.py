"""Tests for performance guardrails instrumentation.

Ensures perf_guard adds minimal overhead (<10% ratio baseline) for a trivial
function. Threshold is conservative to reduce flakiness across CI schedulers.
"""
from __future__ import annotations

from time import perf_counter

from tnfr.performance.guardrails import (
    PerformanceRegistry,
    perf_guard,
    compare_overhead,
)


def _baseline_op() -> int:
    # Moderate workload to reduce relative overhead impact of instrumentation
    x = 0
    for _ in range(2000):
        x += 1
    return x


def test_perf_guard_overhead_ratio():
    registry = PerformanceRegistry()

    @perf_guard("test", registry)
    def _instrumented() -> int:
        return _baseline_op()

    stats = compare_overhead(_baseline_op, _instrumented, runs=500)
    # Overhead ratio should remain below 8% for moderate workload
    assert stats["ratio"] < 0.08, stats
    # Registry should have at least one record (warmup + runs)
    assert registry.summary()["count"] >= 1


def test_perf_registry_summary_fields():
    registry = PerformanceRegistry()
    # Manually record
    start = perf_counter()
    _baseline_op()
    registry.record(
        "manual",
        perf_counter() - start,
        meta={"kind": "baseline"},
    )
    summary = registry.summary()
    assert summary["count"] == 1
    assert "mean" in summary and summary["mean"] > 0
    assert "labels" in summary and summary["labels"] == ["manual"]
