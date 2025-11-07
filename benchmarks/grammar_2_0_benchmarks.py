"""Comprehensive benchmark suite for Grammar 2.0.

This module provides detailed benchmarks for all Grammar 2.0 capabilities,
following the patterns established in the repository's benchmark infrastructure.
It measures validation, health analysis, pattern detection, and cycle validation
across various sequence types and lengths.
"""

from __future__ import annotations

import time
from typing import List

# Import Grammar 2.0 components
from tnfr.operators.grammar import (
    validate_sequence,
    validate_sequence_with_health,
    StructuralPattern,
)
from tnfr.operators.health_analyzer import SequenceHealthAnalyzer
from tnfr.operators.patterns import AdvancedPatternDetector
from tnfr.operators.cycle_detection import CycleDetector


def generate_test_sequences(length: int, count: int) -> List[List[str]]:
    """Generate test sequences of specified length.
    
    This is a simple generator for benchmark purposes. It creates
    valid sequences by starting with a base pattern and extending it.
    """
    sequences = []
    base_patterns = [
        ["emission", "reception", "coherence"],
        ["emission", "reception", "coherence", "resonance"],
        ["emission", "reception", "coherence", "dissonance", "coherence"],
    ]
    
    extensions = [
        "resonance", "silence", "transition", "recursivity",
        "coupling", "expansion", "self_organization",
    ]
    
    for base in base_patterns:
        if len(base) == length:
            sequences.append(base)
        elif len(base) < length:
            # Extend to desired length
            extended = base.copy()
            idx = 0
            while len(extended) < length - 1:
                extended.append(extensions[idx % len(extensions)])
                idx += 1
            # End with valid terminator
            extended.append("silence")
            sequences.append(extended)
    
    # Pad with simple valid sequences if needed
    while len(sequences) < count:
        seq = ["emission", "reception", "coherence"]
        while len(seq) < length - 1:
            seq.append("resonance")
        seq.append("silence")
        sequences.append(seq)
    
    return sequences[:count]


# Test sequence library
MINIMAL_SEQUENCES = [
    ["emission", "reception", "coherence", "silence"],
    ["emission", "reception", "coherence", "resonance", "silence"],
]

MEDIUM_SEQUENCES = [
    [
        "emission",
        "reception",
        "coherence",
        "dissonance",
        "self_organization",
        "coherence",
        "silence",
    ],
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
]

COMPLEX_SEQUENCES = [
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

REGENERATIVE_SEQUENCES = [
    [
        "emission",
        "reception",
        "coherence",
        "resonance",
        "expansion",
        "silence",
        "transition",
        "emission",
        "reception",
        "coupling",
        "coherence",
    ],
]


class Grammar20Benchmarks:
    """Benchmark suite for Grammar 2.0 capabilities."""

    def __init__(self) -> None:
        """Initialize benchmark suite."""
        self.results: dict[str, dict[str, float]] = {}

    def benchmark_validation_performance(self) -> None:
        """Benchmark validation performance across sequence lengths."""
        print("Benchmarking validation performance across sequence lengths...")
        print("=" * 80)

        for length in [3, 5, 8, 10, 13, 15]:
            # Generate test sequences
            sequences = generate_test_sequences(length, count=100)
            if not sequences:
                print(f"  Length {length:2d}: No valid sequences generated")
                continue

            # Benchmark basic validation
            start = time.perf_counter()
            for seq in sequences:
                validate_sequence(seq)
            basic_time = (time.perf_counter() - start) / len(sequences) * 1e6

            # Benchmark health validation
            start = time.perf_counter()
            for seq in sequences:
                validate_sequence_with_health(seq)
            health_time = (time.perf_counter() - start) / len(sequences) * 1e6

            print(
                f"  Length {length:2d}: Basic {basic_time:7.2f} μs, "
                f"With Health {health_time:7.2f} μs"
            )

            self.results[f"validation_length_{length}"] = {
                "basic": basic_time,
                "with_health": health_time,
            }

        print()

    def benchmark_pattern_detection(self) -> None:
        """Benchmark pattern detection for all pattern types."""
        print("Benchmarking pattern detection for pattern types...")
        print("=" * 80)

        detector = AdvancedPatternDetector()

        pattern_examples = {
            "MINIMAL": MINIMAL_SEQUENCES,
            "MEDIUM": MEDIUM_SEQUENCES,
            "COMPLEX": COMPLEX_SEQUENCES,
            "REGENERATIVE": REGENERATIVE_SEQUENCES,
        }

        for pattern_name, sequences in pattern_examples.items():
            detection_times = []

            for seq in sequences:
                # Warmup
                detector.detect_pattern(seq)

                # Measure
                start = time.perf_counter()
                for _ in range(1000):
                    detected = detector.detect_pattern(seq)
                elapsed = (time.perf_counter() - start) / 1000 * 1e6
                detection_times.append(elapsed)

            avg_time = sum(detection_times) / len(detection_times)
            print(f"  {pattern_name:15s}: {avg_time:7.2f} μs avg")

            self.results[f"pattern_{pattern_name.lower()}"] = {"avg_us": avg_time}

        print()

    def benchmark_health_analysis(self) -> None:
        """Benchmark health analysis performance."""
        print("Benchmarking health analysis performance...")
        print("=" * 80)

        analyzer = SequenceHealthAnalyzer()

        test_cases = {
            "minimal": MINIMAL_SEQUENCES,
            "medium": MEDIUM_SEQUENCES,
            "complex": COMPLEX_SEQUENCES,
            "regenerative": REGENERATIVE_SEQUENCES,
        }

        for case_name, sequences in test_cases.items():
            times = []

            for seq in sequences:
                # Warmup
                analyzer.analyze_health(seq)

                # Measure
                start = time.perf_counter()
                for _ in range(1000):
                    metrics = analyzer.analyze_health(seq)
                elapsed = (time.perf_counter() - start) / 1000 * 1e6
                times.append(elapsed)

                # Validate
                assert 0.0 <= metrics.overall_health <= 1.0

            avg = sum(times) / len(times)
            std = (
                (sum((t - avg) ** 2 for t in times) / len(times)) ** 0.5
                if len(times) > 1
                else 0.0
            )
            print(f"  {case_name:15s}: {avg:7.2f} μs ± {std:6.2f} μs")

            self.results[f"health_{case_name}"] = {"avg_us": avg, "std_us": std}

        print()

    def benchmark_cycle_detection(self) -> None:
        """Benchmark cycle detection for regenerative sequences."""
        print("Benchmarking cycle detection performance...")
        print("=" * 80)

        detector = CycleDetector()

        for seq in REGENERATIVE_SEQUENCES:
            # Find regenerator positions
            regenerators = {"transition", "recursivity", "silence"}
            regenerator_indices = [i for i, op in enumerate(seq) if op in regenerators]

            if not regenerator_indices:
                continue

            times = []
            for idx in regenerator_indices:
                # Warmup
                detector.analyze_potential_cycle(seq, idx)

                # Measure
                start = time.perf_counter()
                for _ in range(1000):
                    analysis = detector.analyze_potential_cycle(seq, idx)
                elapsed = (time.perf_counter() - start) / 1000 * 1e6
                times.append(elapsed)

            if times:
                avg = sum(times) / len(times)
                print(f"  Regenerative cycle: {avg:7.2f} μs avg")
                self.results["cycle_regenerative"] = {"avg_us": avg}

        print()

    def benchmark_caching_efficiency(self) -> None:
        """Benchmark caching efficiency for repeated analyses."""
        print("Benchmarking caching efficiency...")
        print("=" * 80)

        # Test sequence that will be repeated
        test_seq = MEDIUM_SEQUENCES[0]

        # Health analysis caching
        analyzer = SequenceHealthAnalyzer()

        # First run (cache miss)
        start = time.perf_counter()
        for _ in range(1000):
            analyzer.analyze_health(test_seq)
        first_run = (time.perf_counter() - start) / 1000 * 1e6

        # Second run (should hit cache)
        start = time.perf_counter()
        for _ in range(1000):
            analyzer.analyze_health(test_seq)
        cached_run = (time.perf_counter() - start) / 1000 * 1e6

        speedup = first_run / cached_run if cached_run > 0 else 1.0
        print(f"  Health Analysis:")
        print(f"    First run:  {first_run:7.2f} μs")
        print(f"    Cached run: {cached_run:7.2f} μs")
        print(f"    Speedup:    {speedup:.2f}x")

        # Pattern detection caching
        detector = AdvancedPatternDetector()

        # First run (cache miss)
        start = time.perf_counter()
        for _ in range(1000):
            detector.detect_pattern(test_seq)
        first_run = (time.perf_counter() - start) / 1000 * 1e6

        # Second run (should hit cache)
        start = time.perf_counter()
        for _ in range(1000):
            detector.detect_pattern(test_seq)
        cached_run = (time.perf_counter() - start) / 1000 * 1e6

        speedup = first_run / cached_run if cached_run > 0 else 1.0
        print(f"  Pattern Detection:")
        print(f"    First run:  {first_run:7.2f} μs")
        print(f"    Cached run: {cached_run:7.2f} μs")
        print(f"    Speedup:    {speedup:.2f}x")

        print()

    def benchmark_worst_case_scenarios(self) -> None:
        """Benchmark performance in worst-case scenarios."""
        print("Benchmarking worst-case scenarios...")
        print("=" * 80)

        # Maximum length sequence
        max_seq = [
            "emission",
            "reception",
            "coherence",
            "dissonance",
            "coupling",
            "resonance",
            "silence",
            "expansion",
            "contraction",
            "self_organization",
            "mutation",
            "transition",
            "recursivity",
        ]

        # Validate maximum length
        start = time.perf_counter()
        for _ in range(100):
            validate_sequence_with_health(max_seq)
        max_length_time = (time.perf_counter() - start) / 100 * 1e6

        print(f"  Maximum length (13 ops): {max_length_time:7.2f} μs")

        # Many destabilizers (challenging for health analysis)
        destabilizer_heavy = [
            "emission",
            "reception",
            "coherence",
            "dissonance",
            "dissonance",
            "coherence",
            "dissonance",
            "coherence",
            "silence",
        ]

        analyzer = SequenceHealthAnalyzer()
        start = time.perf_counter()
        for _ in range(100):
            analyzer.analyze_health(destabilizer_heavy)
        destabilizer_time = (time.perf_counter() - start) / 100 * 1e6

        print(f"  Destabilizer-heavy:      {destabilizer_time:7.2f} μs")

        print()

    def print_summary(self) -> None:
        """Print benchmark summary and check against targets."""
        print("=" * 80)
        print("BENCHMARK SUMMARY")
        print("=" * 80)

        # Performance targets from issue
        targets = {
            "basic_validation": 2000.0,  # < 2ms
            "health_analysis": 10000.0,  # < 10ms
            "pattern_detection": 5000.0,  # < 5ms
            "cycle_validation": 3000.0,  # < 3ms
        }

        print("\nTarget Compliance:")
        print("-" * 80)

        # Check basic validation (use length 10 as representative)
        if "validation_length_10" in self.results:
            basic = self.results["validation_length_10"]["basic"]
            status = "✓" if basic < targets["basic_validation"] else "✗"
            print(
                f"{status} Basic Validation:    {basic:7.2f} μs "
                f"(target: {targets['basic_validation']:.0f} μs)"
            )

        # Check health analysis (use medium as representative)
        if "health_medium" in self.results:
            health = self.results["health_medium"]["avg_us"]
            status = "✓" if health < targets["health_analysis"] else "✗"
            print(
                f"{status} Health Analysis:     {health:7.2f} μs "
                f"(target: {targets['health_analysis']:.0f} μs)"
            )

        # Check pattern detection (use medium as representative)
        if "pattern_medium" in self.results:
            pattern = self.results["pattern_medium"]["avg_us"]
            status = "✓" if pattern < targets["pattern_detection"] else "✗"
            print(
                f"{status} Pattern Detection:   {pattern:7.2f} μs "
                f"(target: {targets['pattern_detection']:.0f} μs)"
            )

        # Check cycle validation
        if "cycle_regenerative" in self.results:
            cycle = self.results["cycle_regenerative"]["avg_us"]
            status = "✓" if cycle < targets["cycle_validation"] else "✗"
            print(
                f"{status} Cycle Validation:    {cycle:7.2f} μs "
                f"(target: {targets['cycle_validation']:.0f} μs)"
            )

        print("=" * 80)

    def run_all(self) -> None:
        """Run all benchmarks."""
        print("\n")
        print("*" * 80)
        print("Grammar 2.0 Comprehensive Benchmark Suite")
        print("*" * 80)
        print()

        self.benchmark_validation_performance()
        self.benchmark_pattern_detection()
        self.benchmark_health_analysis()
        self.benchmark_cycle_detection()
        self.benchmark_caching_efficiency()
        self.benchmark_worst_case_scenarios()
        self.print_summary()


if __name__ == "__main__":
    benchmarks = Grammar20Benchmarks()
    benchmarks.run_all()
