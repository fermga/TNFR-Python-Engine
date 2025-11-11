"""Performance tests for Grammar 2.0 features.

Tests performance benchmarks for health analysis, pattern detection,
and cycle validation to ensure they meet efficiency requirements.
"""

import pytest

from tnfr.operators.grammar import validate_sequence_with_health
from tnfr.operators.health_analyzer import SequenceHealthAnalyzer
from tnfr.operators.patterns import AdvancedPatternDetector
from tnfr.operators.cycle_detection import CycleDetector


class TestGrammar20Performance:
    """Performance tests for Grammar 2.0 capabilities."""

    def test_health_analysis_performance(self, benchmark):
        """Health analysis should be fast for normal sequences (<15 ops)."""
        # Normal-length therapeutic sequence
        sequence = [
            "emission",
            "reception",
            "coherence",
            "dissonance",
            "self_organization",
            "coherence",
            "silence",
        ]

        analyzer = SequenceHealthAnalyzer()

        # Benchmark health analysis
        result = benchmark(analyzer.analyze_health, sequence)

        # Should complete successfully
        assert result.overall_health > 0.0
        assert result.sequence_length == len(sequence)

    def test_advanced_pattern_detection_speed(self, benchmark):
        """Pattern detection should scale well with sequence length."""
        # Medium-length sequence
        sequence = [
            "emission",
            "reception",
            "coherence",
            "dissonance",
            "mutation",
            "coherence",
            "resonance",
            "coupling",
            "silence",
        ]

        detector = AdvancedPatternDetector()

        # Benchmark pattern detection
        result = benchmark(detector.detect_pattern, sequence)

        # Should complete and detect a pattern
        assert result is not None

    def test_pattern_composition_analysis_speed(self, benchmark):
        """Comprehensive composition analysis should be efficient."""
        sequence = [
            "emission",
            "reception",
            "coherence",
            "dissonance",
            "self_organization",
            "coherence",
            "silence",
        ]

        detector = AdvancedPatternDetector()

        # Benchmark composition analysis
        result = benchmark(detector.analyze_sequence_composition, sequence)

        # Should complete with results
        assert result["primary_pattern"] is not None
        assert "pattern_scores" in result

    def test_regenerative_cycle_validation_efficiency(self, benchmark):
        """Cycle validation should be efficient for sequences with regenerators."""
        sequence = [
            "emission",
            "reception",
            "coherence",
            "resonance",
            "recursivity",
        ]

        cycle_detector = CycleDetector()
        regenerator_idx = sequence.index("recursivity")

        # Benchmark cycle analysis
        result = benchmark(cycle_detector.analyze_potential_cycle, sequence, regenerator_idx)

        # Should complete
        assert result is not None
        assert result.cycle_type is not None

    def test_validation_with_health_performance(self, benchmark):
        """Full validation with health should be reasonably fast."""
        sequence = [
            "emission",
            "reception",
            "coherence",
            "dissonance",
            "self_organization",
            "coherence",
            "silence",
        ]

        # Benchmark complete validation + health analysis
        result = benchmark(validate_sequence_with_health, sequence)

        # Should pass and have health metrics
        assert result.passed
        assert result.health_metrics is not None

    def test_health_analysis_scales_linearly(self):
        """Health analysis should scale approximately linearly with sequence length."""
        import time

        analyzer = SequenceHealthAnalyzer()

        # Short sequence (5 ops)
        short_seq = ["emission", "reception", "coherence", "resonance", "silence"]

        # Long sequence (10 ops) - 2x length
        long_seq = [
            "emission",
            "reception",
            "coherence",
            "dissonance",
            "mutation",
            "coherence",
            "resonance",
            "coupling",
            "recursivity",
            "silence",
        ]

        # Measure short sequence (with warmup)
        analyzer.analyze_health(short_seq)  # Warmup
        start = time.perf_counter()
        for _ in range(100):
            analyzer.analyze_health(short_seq)
        short_time = time.perf_counter() - start

        # Measure long sequence
        analyzer.analyze_health(long_seq)  # Warmup
        start = time.perf_counter()
        for _ in range(100):
            analyzer.analyze_health(long_seq)
        long_time = time.perf_counter() - start

        # Long sequence should not be more than 3x slower (allowing overhead)
        # Linear scaling would be 2x, we allow 3x for overhead
        assert long_time / short_time < 3.0, (
            f"Health analysis doesn't scale well: "
            f"{long_time/short_time:.2f}x slower for 2x length"
        )

    def test_pattern_detection_on_maximum_sequence(self):
        """Pattern detection should handle maximum-length sequences (13 ops)."""
        import time

        # Sequence with all 13 canonical operators
        max_sequence = [
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

        detector = AdvancedPatternDetector()

        # Should complete in reasonable time (< 100ms for single call)
        start = time.perf_counter()
        pattern = detector.detect_pattern(max_sequence)
        elapsed = time.perf_counter() - start

        assert pattern is not None
        assert elapsed < 0.1, f"Pattern detection too slow: {elapsed*1000:.2f}ms"

    def test_cycle_detection_multiple_regenerators(self):
        """Cycle detection should efficiently handle multiple regenerator positions."""
        import time

        sequence = [
            "emission",
            "reception",
            "coherence",
            "transition",  # Regenerator 1
            "resonance",
            "recursivity",  # Regenerator 2 (and valid end)
        ]

        cycle_detector = CycleDetector()

        # Find all regenerators
        regenerator_indices = [
            i for i, op in enumerate(sequence) if op in ["transition", "recursivity", "silence"]
        ]

        # Should analyze all positions efficiently
        start = time.perf_counter()
        for idx in regenerator_indices:
            cycle_detector.analyze_potential_cycle(sequence, idx)
        elapsed = time.perf_counter() - start

        assert elapsed < 0.05, f"Cycle detection too slow: {elapsed*1000:.2f}ms"

    def test_memory_efficiency_repeated_analysis(self):
        """Repeated analyses should not accumulate memory."""
        import gc
        import sys

        sequence = [
            "emission",
            "reception",
            "coherence",
            "dissonance",
            "self_organization",
            "coherence",
            "silence",
        ]

        analyzer = SequenceHealthAnalyzer()
        detector = AdvancedPatternDetector()

        # Force garbage collection and get baseline
        gc.collect()
        baseline = sys.getsizeof(analyzer) + sys.getsizeof(detector)

        # Perform many analyses
        for _ in range(1000):
            analyzer.analyze_health(sequence)
            detector.detect_pattern(sequence)

        # Force garbage collection
        gc.collect()
        after = sys.getsizeof(analyzer) + sys.getsizeof(detector)

        # Objects should not have grown significantly
        # Allow some growth for internal caching but not excessive
        growth = after - baseline
        assert growth < 10000, f"Excessive memory growth: {growth} bytes"
