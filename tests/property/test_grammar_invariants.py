"""Property-based tests for Grammar 2.0 invariants.

Uses Hypothesis to test that Grammar 2.0 features maintain their
structural invariants across a wide range of valid sequences.
"""

import pytest
from hypothesis import given, settings, strategies as st
from hypothesis import HealthCheck

from tnfr.operators.grammar import validate_sequence, validate_sequence_with_health
from tnfr.operators.health_analyzer import SequenceHealthAnalyzer
from tnfr.operators.patterns import AdvancedPatternDetector
from tnfr.operators.cycle_detection import CycleDetector

# All canonical operators
ALL_OPERATORS = [
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

# Valid ending operators
VALID_ENDINGS = ["dissonance", "recursivity", "silence", "transition"]

# Stabilizers
STABILIZERS = ["coherence", "self_organization", "silence", "resonance", "coupling"]


def build_valid_sequence_strategy():
    """Build a strategy for generating likely-valid sequences."""
    # Start with emission or recursivity
    starts = st.sampled_from(["emission", "recursivity"])
    
    # Middle can be various operators
    middles = st.lists(
        st.sampled_from(ALL_OPERATORS),
        min_size=0,
        max_size=10
    )
    
    # Must end with valid ending
    ends = st.sampled_from(VALID_ENDINGS)
    
    @st.composite
    def sequence_strategy(draw):
        start = draw(starts)
        middle = draw(middles)
        end = draw(ends)
        
        # Build sequence: start + middle + end
        seq = [start] + middle + [end]
        
        # Try to ensure reception → coherence if we have reception
        # This is a soft requirement to increase valid sequence rate
        if "reception" in seq and "coherence" not in seq:
            # Insert coherence after reception
            idx = seq.index("reception")
            seq.insert(idx + 1, "coherence")
        
        return seq
    
    return sequence_strategy()


class TestGrammarInvariants:
    """Property-based tests for Grammar 2.0 invariants."""

    @given(build_valid_sequence_strategy())
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_health_metrics_always_in_range(self, sequence):
        """Health metrics should always be in range [0.0, 1.0]."""
        # Only test sequences that pass validation
        result = validate_sequence(sequence)
        if not result.passed:
            return  # Skip invalid sequences
        
        analyzer = SequenceHealthAnalyzer()
        health = analyzer.analyze_health(sequence)

        # All metrics must be in valid range
        assert 0.0 <= health.coherence_index <= 1.0
        assert 0.0 <= health.balance_score <= 1.0
        assert 0.0 <= health.sustainability_index <= 1.0
        assert 0.0 <= health.complexity_efficiency <= 1.0
        assert 0.0 <= health.frequency_harmony <= 1.0
        assert 0.0 <= health.pattern_completeness <= 1.0
        assert 0.0 <= health.transition_smoothness <= 1.0
        assert 0.0 <= health.overall_health <= 1.0

    @given(build_valid_sequence_strategy())
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_pattern_detection_deterministic(self, sequence):
        """Pattern detection should be deterministic for the same sequence."""
        # Only test sequences that pass validation
        result = validate_sequence(sequence)
        if not result.passed:
            return
        
        detector = AdvancedPatternDetector()

        # Detect pattern multiple times
        pattern1 = detector.detect_pattern(sequence)
        pattern2 = detector.detect_pattern(sequence)
        pattern3 = detector.detect_pattern(sequence)

        # Should always return the same pattern
        assert pattern1 == pattern2 == pattern3

    @given(build_valid_sequence_strategy())
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_health_analysis_deterministic(self, sequence):
        """Health analysis should be deterministic for the same sequence."""
        # Only test sequences that pass validation
        result = validate_sequence(sequence)
        if not result.passed:
            return
        
        analyzer = SequenceHealthAnalyzer()

        # Analyze multiple times
        health1 = analyzer.analyze_health(sequence)
        health2 = analyzer.analyze_health(sequence)

        # Should return identical results
        assert health1.overall_health == health2.overall_health
        assert health1.coherence_index == health2.coherence_index
        assert health1.balance_score == health2.balance_score
        assert health1.dominant_pattern == health2.dominant_pattern

    @given(build_valid_sequence_strategy())
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_sequence_length_preserved(self, sequence):
        """Health metrics should correctly report sequence length."""
        # Only test sequences that pass validation
        result = validate_sequence(sequence)
        if not result.passed:
            return
        
        analyzer = SequenceHealthAnalyzer()
        health = analyzer.analyze_health(sequence)

        # Reported length should match actual length
        assert health.sequence_length == len(sequence)

    @given(build_valid_sequence_strategy())
    @settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow])
    def test_cycle_analysis_valid_health_score(self, sequence):
        """Cycle analysis should always produce valid health scores."""
        # Only test sequences that pass validation
        result = validate_sequence(sequence)
        if not result.passed:
            return
        
        # Find regenerators in sequence
        regenerators = ["transition", "recursivity", "silence"]
        regenerator_indices = [
            i for i, op in enumerate(sequence) if op in regenerators
        ]
        
        if not regenerator_indices:
            return  # Skip if no regenerators
        
        cycle_detector = CycleDetector()
        
        for idx in regenerator_indices:
            analysis = cycle_detector.analyze_potential_cycle(sequence, idx)
            
            # Health score should be in valid range
            assert 0.0 <= analysis.health_score <= 1.0
            
            # Balance score should be in valid range
            assert 0.0 <= analysis.balance_score <= 1.0
            
            # Diversity score should be in valid range
            assert 0.0 <= analysis.diversity_score <= 1.0
            
            # Coherence score should be in valid range
            assert 0.0 <= analysis.coherence_score <= 1.0

    @given(build_valid_sequence_strategy())
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_validation_with_health_consistency(self, sequence):
        """validate_sequence_with_health should be consistent with validate_sequence."""
        # Get both validation results
        basic_result = validate_sequence(sequence)
        health_result = validate_sequence_with_health(sequence)

        # Validation outcome should match
        assert basic_result.passed == health_result.passed

        # If passed, health result should have metrics
        if health_result.passed:
            assert health_result.health_metrics is not None
        else:
            assert health_result.health_metrics is None

    @given(build_valid_sequence_strategy())
    @settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow])
    def test_pattern_composition_structure(self, sequence):
        """Pattern composition should always return expected structure."""
        # Only test sequences that pass validation
        result = validate_sequence(sequence)
        if not result.passed:
            return
        
        detector = AdvancedPatternDetector()
        composition = detector.analyze_sequence_composition(sequence)

        # Should have all expected keys
        assert "primary_pattern" in composition
        assert "pattern_scores" in composition
        assert "weighted_scores" in composition
        assert "coherence_weights" in composition
        assert "components" in composition
        assert "complexity_score" in composition
        assert "domain_suitability" in composition
        assert "structural_health" in composition

        # Primary pattern should be a string
        assert isinstance(composition["primary_pattern"], str)

        # Scores should be dictionaries
        assert isinstance(composition["pattern_scores"], dict)
        assert isinstance(composition["weighted_scores"], dict)

        # All pattern scores should be non-negative
        for score in composition["pattern_scores"].values():
            assert score >= 0.0

        # Complexity score should be in range
        assert 0.0 <= composition["complexity_score"] <= 1.0

    @given(st.lists(st.sampled_from(ALL_OPERATORS), min_size=1, max_size=13))
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_health_analyzer_never_crashes(self, sequence):
        """Health analyzer should handle any sequence without crashing."""
        analyzer = SequenceHealthAnalyzer()
        
        # Should complete without exception (even for invalid sequences)
        try:
            health = analyzer.analyze_health(sequence)
            # If it completes, check basic invariants
            assert 0.0 <= health.overall_health <= 1.0
            assert health.sequence_length == len(sequence)
        except Exception as e:
            pytest.fail(f"Health analyzer crashed on sequence {sequence}: {e}")

    @given(st.lists(st.sampled_from(ALL_OPERATORS), min_size=1, max_size=13))
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_pattern_detector_never_crashes(self, sequence):
        """Pattern detector should handle any sequence without crashing."""
        detector = AdvancedPatternDetector()
        
        # Should complete without exception
        try:
            pattern = detector.detect_pattern(sequence)
            # If it completes, should return a valid StructuralPattern
            assert pattern is not None
            assert hasattr(pattern, 'value')
        except Exception as e:
            pytest.fail(f"Pattern detector crashed on sequence {sequence}: {e}")

    @given(build_valid_sequence_strategy())
    @settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow])
    def test_recommendations_are_strings(self, sequence):
        """Health analyzer recommendations should always be strings."""
        # Only test sequences that pass validation
        result = validate_sequence(sequence)
        if not result.passed:
            return
        
        analyzer = SequenceHealthAnalyzer()
        health = analyzer.analyze_health(sequence)

        # Recommendations should be a list of strings
        assert isinstance(health.recommendations, list)
        for rec in health.recommendations:
            assert isinstance(rec, str)
            assert len(rec) > 0  # Non-empty recommendations
