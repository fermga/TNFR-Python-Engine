"""Performance guardrails instrumentation for TNFR Phase 3.

Lightweight timing utilities ensuring added structural validation / telemetry
instrumentation remains below configured overhead thresholds.

Design Goals
------------
1. Zero external dependencies (stdlib only).
2. Minimal footprint: single perf_counter measurement plus registry append.
3. Opt-in: instrumentation only active when explicitly passed a registry.
4. Composable: decorator or manual timing blocks.

Physics Alignment
-----------------
Performance measurement is purely operational and never alters TNFR physics;
it wraps functions that perform read-only structural computations. The
guardrails act as a containment layer ensuring added monitoring does not
fragment coherence through excessive latency.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Callable, Dict, List

__all__ = [
    "PerformanceRegistry",
    "perf_guard",
    "compare_overhead",
]


@dataclass(slots=True)
class PerformanceRecord:
    label: str
    elapsed: float
    meta: Dict[str, Any] | None = None


@dataclass(slots=True)
class PerformanceRegistry:
    """Collects performance timing records.

    Methods
    -------
    record(label, elapsed, meta=None): Add a timing entry.
    summary(): Aggregate statistics (count, mean, max, min, labels).
    filter(label): Return list of records matching label.
    """

    records: List[PerformanceRecord] = field(default_factory=list)

    def record(
        self, label: str, elapsed: float, meta: Dict[str, Any] | None = None
    ) -> None:
        self.records.append(PerformanceRecord(label, float(elapsed), meta))

    def filter(self, label: str) -> List[PerformanceRecord]:
        return [r for r in self.records if r.label == label]

    def summary(self) -> Dict[str, Any]:
        if not self.records:
            return {"count": 0}
        total = sum(r.elapsed for r in self.records)
        return {
            "count": len(self.records),
            "total": total,
            "mean": total / len(self.records),
            "max": max(r.elapsed for r in self.records),
            "min": min(r.elapsed for r in self.records),
            "labels": sorted({r.label for r in self.records}),
        }


def perf_guard(label: str, registry: PerformanceRegistry | None) -> Callable:
    """Decorator adding a single perf_counter measurement if registry provided.

    Parameters
    ----------
    label : str
        Logical name for the operation (e.g. "validation" or "telemetry").
    registry : PerformanceRegistry | None
        Active registry; if None instrumentation is skipped.
    """

    def decorator(fn: Callable) -> Callable:
        def wrapped(*args, **kwargs):  # type: ignore[override]
            if registry is None:
                return fn(*args, **kwargs)
            start = perf_counter()
            result = fn(*args, **kwargs)
            registry.record(label, perf_counter() - start, meta={
                "fn": fn.__name__,
                "arg_count": len(args),
                "kw_count": len(kwargs),
            })
            return result

        wrapped.__name__ = fn.__name__  # preserve for introspection
        wrapped.__doc__ = fn.__doc__
        return wrapped

    return decorator


def compare_overhead(
    baseline_fn: Callable[[], Any],
    instrumented_fn: Callable[[], Any],
    *,
    runs: int = 5000,
) -> Dict[str, float]:
    """Compare overhead ratio between baseline and instrumented call sets.

    Returns timing dict with baseline, instrumented and ratio
    (instrumented - baseline) / baseline.
    """
    # Warmup
    for _ in range(10):
        baseline_fn()
        instrumented_fn()
    b_start = perf_counter()
    for _ in range(runs):
        baseline_fn()
    b_elapsed = perf_counter() - b_start
    i_start = perf_counter()
    for _ in range(runs):
        instrumented_fn()
    i_elapsed = perf_counter() - i_start
    ratio = (i_elapsed - b_elapsed) / b_elapsed if b_elapsed > 0 else 0.0
    return {
        "baseline": b_elapsed,
        "instrumented": i_elapsed,
        "ratio": ratio,
        "runs": float(runs),
    }
