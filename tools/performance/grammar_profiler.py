"""Performance profiling tool for Grammar 2.0 capabilities.

This module provides tools to profile and analyze the performance of
Grammar 2.0 features including validation, health analysis, pattern detection,
and cycle validation. It reuses existing timing utilities and provides
comprehensive reports for optimization and regression testing.
"""

from __future__ import annotations

import statistics
import sys
from time import perf_counter
from typing import Any, Callable, Dict, List, Optional, Tuple

# Import Grammar 2.0 components
from tnfr.operators.grammar import validate_sequence, validate_sequence_with_health
from tnfr.operators.health_analyzer import SequenceHealthAnalyzer
from tnfr.operators.patterns import AdvancedPatternDetector
from tnfr.operators.cycle_detection import CycleDetector


class PerformanceReport:
    """Container for performance profiling results."""

    def __init__(self) -> None:
        """Initialize empty performance report."""
        self.metrics: Dict[str, Dict[str, Any]] = {}

    def add_metric(
        self,
        name: str,
        times: List[float],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add a performance metric to the report.

        Parameters
        ----------
        name : str
            Name of the metric/component being measured
        times : List[float]
            List of timing measurements in seconds
        metadata : Optional[Dict[str, Any]]
            Additional context about the measurement
        """
        if not times:
            return

        self.metrics[name] = {
            "count": len(times),
            "min_us": min(times) * 1e6,
            "max_us": max(times) * 1e6,
            "mean_us": statistics.mean(times) * 1e6,
            "median_us": statistics.median(times) * 1e6,
            "stdev_us": statistics.stdev(times) * 1e6 if len(times) > 1 else 0.0,
            "metadata": metadata or {},
        }

    def print_report(self, file=None) -> None:
        """Print formatted performance report.

        Parameters
        ----------
        file : file-like object, optional
            Output stream (defaults to sys.stdout)
        """
        if file is None:
            file = sys.stdout

        print("=" * 80, file=file)
        print("Grammar 2.0 Performance Profile", file=file)
        print("=" * 80, file=file)

        for name, data in self.metrics.items():
            print(f"\n{name}:", file=file)
            print(f"  Runs:     {data['count']}", file=file)
            print(f"  Min:      {data['min_us']:.2f} μs", file=file)
            print(f"  Max:      {data['max_us']:.2f} μs", file=file)
            print(f"  Mean:     {data['mean_us']:.2f} μs", file=file)
            print(f"  Median:   {data['median_us']:.2f} μs", file=file)
            print(f"  StdDev:   {data['stdev_us']:.2f} μs", file=file)

            if data["metadata"]:
                print("  Metadata:", file=file)
                for key, value in data["metadata"].items():
                    print(f"    {key}: {value}", file=file)

        print("\n" + "=" * 80, file=file)


class Grammar20Profiler:
    """Performance profiler for all Grammar 2.0 capabilities."""

    def __init__(self) -> None:
        """Initialize the profiler."""
        self.report = PerformanceReport()

    def profile_validation_pipeline(
        self, sequences: List[List[str]], iterations: int = 100
    ) -> PerformanceReport:
        """Profile complete validation pipeline with all components.

        Parameters
        ----------
        sequences : List[List[str]]
            Test sequences to profile
        iterations : int, default=100
            Number of iterations per sequence for averaging

        Returns
        -------
        PerformanceReport
            Comprehensive performance report
        """
        print(f"Profiling validation pipeline with {len(sequences)} sequences...")

        # Profile basic validation
        print("  - Basic validation...")
        basic_times = []
        for seq in sequences:
            for _ in range(iterations):
                start = perf_counter()
                validate_sequence(seq)
                basic_times.append(perf_counter() - start)

        self.report.add_metric(
            "Basic Validation",
            basic_times,
            {
                "sequences": len(sequences),
                "iterations": iterations,
                "avg_length": sum(len(s) for s in sequences) / len(sequences),
            },
        )

        # Profile health analysis only (for sequences that pass validation)
        print("  - Health analysis...")
        health_analyzer = SequenceHealthAnalyzer()
        health_times = []
        for seq in sequences:
            # Pre-validate to avoid measuring failed sequences
            result = validate_sequence(seq)
            if result.passed:
                for _ in range(iterations):
                    start = perf_counter()
                    health_analyzer.analyze_health(seq)
                    health_times.append(perf_counter() - start)

        if health_times:
            self.report.add_metric(
                "Health Analysis",
                health_times,
                {"valid_sequences": len([s for s in sequences if validate_sequence(s).passed])},
            )

        # Profile pattern detection
        print("  - Pattern detection...")
        pattern_detector = AdvancedPatternDetector()
        pattern_times = []
        for seq in sequences:
            for _ in range(iterations):
                start = perf_counter()
                pattern_detector.detect_pattern(seq)
                pattern_times.append(perf_counter() - start)

        self.report.add_metric(
            "Pattern Detection",
            pattern_times,
            {"sequences": len(sequences), "iterations": iterations},
        )

        # Profile cycle detection (only for sequences with regenerators)
        print("  - Cycle detection...")
        cycle_detector = CycleDetector()
        cycle_times = []
        regenerators = {"transition", "recursivity"}  # Silence is not a regenerator
        for seq in sequences:
            # Find regenerator positions
            regenerator_indices = [
                i for i, op in enumerate(seq) if op in regenerators
            ]
            if regenerator_indices:
                for idx in regenerator_indices[:1]:  # Test first regenerator only
                    for _ in range(iterations):
                        start = perf_counter()
                        cycle_detector.analyze_potential_cycle(seq, idx)
                        cycle_times.append(perf_counter() - start)

        if cycle_times:
            self.report.add_metric(
                "Cycle Detection",
                cycle_times,
                {
                    "sequences_with_regenerators": sum(
                        1 for s in sequences if any(op in regenerators for op in s)
                    )
                },
            )

        # Profile full validation with health
        print("  - Full validation with health...")
        full_times = []
        for seq in sequences:
            for _ in range(iterations):
                start = perf_counter()
                validate_sequence_with_health(seq)
                full_times.append(perf_counter() - start)

        self.report.add_metric(
            "Full Validation with Health",
            full_times,
            {"sequences": len(sequences), "iterations": iterations},
        )

        print("Profiling complete!\n")
        return self.report

    def identify_bottlenecks(self) -> List[Tuple[str, float]]:
        """Identify slowest components for optimization.

        Returns
        -------
        List[Tuple[str, float]]
            List of (component_name, mean_time_us) sorted by time
        """
        bottlenecks = [
            (name, data["mean_us"]) for name, data in self.report.metrics.items()
        ]
        return sorted(bottlenecks, key=lambda x: x[1], reverse=True)


def main():
    """Run profiler with default test sequences."""
    # Generate diverse test sequences
    test_sequences = [
        # Minimal sequences
        ["emission", "reception", "coherence", "silence"],
        # Therapeutic pattern
        [
            "emission",
            "reception",
            "coherence",
            "dissonance",
            "self_organization",
            "coherence",
            "silence",
        ],
        # Educational pattern
        [
            "emission",
            "reception",
            "coherence",
            "expansion",
            "dissonance",
            "mutation",
            "coherence",
            "silence",
        ],
        # Regenerative pattern
        [
            "emission",
            "reception",
            "coherence",
            "resonance",
            "transition",
            "emission",
            "reception",
            "coherence",
            "recursivity",
        ],
        # Complex sequence
        [
            "emission",
            "reception",
            "coherence",
            "dissonance",
            "coupling",
            "resonance",
            "expansion",
            "contraction",
            "self_organization",
            "mutation",
            "transition",
            "recursivity",
        ],
    ]

    profiler = Grammar20Profiler()

    # Profile validation pipeline
    report = profiler.profile_validation_pipeline(test_sequences, iterations=100)
    report.print_report()

    # Identify bottlenecks
    print("\nPerformance Bottlenecks (slowest components):")
    print("-" * 80)
    bottlenecks = profiler.identify_bottlenecks()
    for i, (name, time_us) in enumerate(bottlenecks, 1):
        print(f"{i}. {name}: {time_us:.2f} μs")

    # Check against targets (from issue)
    print("\nPerformance Targets Check:")
    print("-" * 80)
    targets = {
        "Basic Validation": 2000.0,  # < 2ms
        "Health Analysis": 10000.0,  # < 10ms
        "Pattern Detection": 5000.0,  # < 5ms
        "Full Validation with Health": 10000.0,  # < 10ms
    }

    for component, target_us in targets.items():
        if component in report.metrics:
            actual_us = report.metrics[component]["mean_us"]
            status = "✓ PASS" if actual_us < target_us else "✗ FAIL"
            pct = (actual_us / target_us) * 100
            print(f"{component}:")
            print(f"  Target: {target_us:.0f} μs")
            print(f"  Actual: {actual_us:.2f} μs ({pct:.1f}% of target)")
            print(f"  Status: {status}")


if __name__ == "__main__":
    main()
