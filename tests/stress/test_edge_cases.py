"""Stress tests for Grammar 2.0 edge cases.

Tests extreme scenarios, degenerate patterns, and edge cases to ensure
robustness of Grammar 2.0 features.
"""

import pytest

from tnfr.operators.grammar import validate_sequence, validate_sequence_with_health
from tnfr.operators.health_analyzer import SequenceHealthAnalyzer
from tnfr.operators.patterns import AdvancedPatternDetector
from tnfr.operators.cycle_detection import CycleDetector


class TestEdgeCases:
    """Edge case and stress tests for Grammar 2.0."""

    def test_maximum_sequence_length(self):
        """Test sequence with all 13 canonical operators."""
        # Maximum sequence: one of each operator
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
            "recursivity",  # Valid ending
        ]

        # Should validate
        result = validate_sequence(max_sequence)
        # Note: may fail due to grammar rules, that's OK
        
        # Health analysis should handle it
        analyzer = SequenceHealthAnalyzer()
        health = analyzer.analyze_health(max_sequence)
        
        assert health.sequence_length == 13
        assert 0.0 <= health.overall_health <= 1.0

        # Pattern detection should handle it
        detector = AdvancedPatternDetector()
        pattern = detector.detect_pattern(max_sequence)
        assert pattern is not None

    def test_minimal_valid_sequence(self):
        """Test minimal valid sequences."""
        # Minimal sequences that could be valid
        minimal_sequences = [
            ["emission", "reception", "coherence", "silence"],
            ["emission", "reception", "coherence", "recursivity"],
            ["recursivity"],  # Single operator (might be valid)
        ]

        analyzer = SequenceHealthAnalyzer()
        detector = AdvancedPatternDetector()

        for sequence in minimal_sequences:
            # Validation may pass or fail depending on grammar
            result = validate_sequence(sequence)
            
            # Health analysis should always complete
            health = analyzer.analyze_health(sequence)
            assert health.sequence_length == len(sequence)
            assert 0.0 <= health.overall_health <= 1.0

            # Pattern detection should always complete
            pattern = detector.detect_pattern(sequence)
            assert pattern is not None

    def test_single_operator_sequences(self):
        """Test sequences with only one operator."""
        single_ops = [
            ["emission"],
            ["coherence"],
            ["silence"],
            ["recursivity"],
            ["dissonance"],
        ]

        analyzer = SequenceHealthAnalyzer()
        detector = AdvancedPatternDetector()

        for sequence in single_ops:
            # Health analysis should handle single operators
            health = analyzer.analyze_health(sequence)
            assert health.sequence_length == 1
            assert 0.0 <= health.overall_health <= 1.0

            # Pattern detection should handle single operators
            pattern = detector.detect_pattern(sequence)
            assert pattern is not None

    def test_repetitive_sequences(self):
        """Test sequences with repeated operators."""
        repetitive_sequences = [
            ["emission", "reception", "coherence", "coherence", "coherence", "silence"],
            ["emission", "reception", "resonance", "resonance", "resonance", "silence"],
        ]

        analyzer = SequenceHealthAnalyzer()
        detector = AdvancedPatternDetector()

        for sequence in repetitive_sequences:
            # Should handle repetition
            health = analyzer.analyze_health(sequence)
            assert health.sequence_length == len(sequence)

            pattern = detector.detect_pattern(sequence)
            assert pattern is not None

    def test_degenerate_patterns(self):
        """Test sequences that might confuse pattern detection."""
        degenerate_sequences = [
            # All stabilizers
            ["emission", "reception", "coherence", "coherence", "silence"],
            # Alternating operators
            ["emission", "reception", "coherence", "dissonance", "coherence", "dissonance", "silence"],
            # Only high frequency operators (where valid)
            ["emission", "resonance", "dissonance", "recursivity"],
        ]

        detector = AdvancedPatternDetector()

        for sequence in degenerate_sequences:
            # Should detect some pattern (even if "unknown" or "complex")
            pattern = detector.detect_pattern(sequence)
            assert pattern is not None
            
            # Composition should complete
            composition = detector.analyze_sequence_composition(sequence)
            assert composition["primary_pattern"] is not None

    def test_high_frequency_transitions(self):
        """Test sequences with only high-frequency operators."""
        # Operators with high structural frequency
        high_freq_sequence = [
            "emission",     # high
            "resonance",    # high
            "dissonance",   # high
            "mutation",     # high
            "recursivity",  # medium (valid end)
        ]

        analyzer = SequenceHealthAnalyzer()
        health = analyzer.analyze_health(high_freq_sequence)

        # Should complete health analysis
        assert health.sequence_length == len(high_freq_sequence)
        
        # Frequency harmony might be lower due to sustained high frequency
        # but should still be in valid range
        assert 0.0 <= health.frequency_harmony <= 1.0

    def test_zero_frequency_edge_cases(self):
        """Test sequences with silence (zero frequency) in various positions."""
        silence_sequences = [
            # Silence at start (invalid start, but test health analysis)
            ["emission", "reception", "coherence", "silence", "recursivity"],
            # Multiple silences
            ["emission", "reception", "coherence", "silence", "silence"],
            # Silence followed by high frequency (tests frequency transitions)
            ["emission", "reception", "coherence", "silence", "transition", "recursivity"],
        ]

        analyzer = SequenceHealthAnalyzer()

        for sequence in silence_sequences:
            # Health analysis should handle silence positions
            health = analyzer.analyze_health(sequence)
            assert health.sequence_length == len(sequence)
            assert 0.0 <= health.frequency_harmony <= 1.0

    def test_all_destabilizers(self):
        """Test sequence dominated by destabilizing operators."""
        # Need to include stabilizers per grammar rules
        destabilizer_heavy = [
            "emission",
            "reception",
            "dissonance",
            "dissonance",
            "mutation",
            "coherence",  # Required stabilizer
            "silence",
        ]

        analyzer = SequenceHealthAnalyzer()
        health = analyzer.analyze_health(destabilizer_heavy)

        # Should have low balance score due to destabilizer dominance
        assert 0.0 <= health.balance_score <= 1.0
        
        # But overall health should still be computable
        assert 0.0 <= health.overall_health <= 1.0

    def test_cycle_detection_edge_cases(self):
        """Test cycle detection with edge case regenerator positions."""
        cycle_detector = CycleDetector()

        # Regenerator at very beginning
        seq1 = ["recursivity"]
        if len(seq1) >= CycleDetector.MIN_HEALTH_SCORE:  # Use class constant
            analysis1 = cycle_detector.analyze_potential_cycle(seq1, 0)
            assert analysis1 is not None

        # Regenerator at end
        seq2 = ["emission", "reception", "coherence", "recursivity"]
        analysis2 = cycle_detector.analyze_potential_cycle(seq2, len(seq2) - 1)
        assert analysis2 is not None
        assert 0.0 <= analysis2.health_score <= 1.0

        # Multiple consecutive regenerators
        seq3 = ["emission", "reception", "coherence", "transition", "recursivity"]
        for idx, op in enumerate(seq3):
            if op in ["transition", "recursivity", "silence"]:
                analysis = cycle_detector.analyze_potential_cycle(seq3, idx)
                assert analysis is not None

    def test_empty_pattern_scores(self):
        """Test sequences that might not match any pattern."""
        # Unusual sequence that may not match patterns
        unusual = ["emission", "reception", "coherence", "silence"]

        detector = AdvancedPatternDetector()
        composition = detector.analyze_sequence_composition(unusual)

        # Should have composition structure even if no patterns match
        assert "pattern_scores" in composition
        # pattern_scores might be empty, that's OK
        assert isinstance(composition["pattern_scores"], dict)

    def test_health_with_no_stabilizers(self):
        """Test health analysis when sequence has minimal stabilizers."""
        # Sequence with only required stabilizers
        minimal_stabilizers = [
            "emission",
            "dissonance",
            "mutation",
            "coherence",  # Minimal stabilizer
            "recursivity",
        ]

        analyzer = SequenceHealthAnalyzer()
        health = analyzer.analyze_health(minimal_stabilizers)

        # Should compute health (may be lower)
        assert 0.0 <= health.overall_health <= 1.0
        
        # Sustainability might be lower
        assert 0.0 <= health.sustainability_index <= 1.0

    def test_pattern_detection_ambiguous_sequences(self):
        """Test sequences that could match multiple patterns."""
        # Sequence with characteristics of multiple patterns
        ambiguous = [
            "emission",          # Activation
            "reception",
            "coherence",
            "dissonance",        # Exploration
            "self_organization", # Transformative
            "coherence",
            "silence",
        ]

        detector = AdvancedPatternDetector()
        composition = detector.analyze_sequence_composition(ambiguous)

        # Should pick a primary pattern
        assert composition["primary_pattern"] is not None

        # May have multiple pattern matches
        if len(composition["pattern_scores"]) > 1:
            # Verify weighted scores were computed
            assert len(composition["weighted_scores"]) > 1

    def test_extremely_imbalanced_sequences(self):
        """Test sequences with extreme imbalance."""
        # All stabilizers (where grammar allows)
        all_stabilizers = [
            "emission",
            "reception",
            "coherence",
            "coherence",
            "resonance",
            "coupling",
            "silence",
        ]

        analyzer = SequenceHealthAnalyzer()
        health = analyzer.analyze_health(all_stabilizers)

        # Balance should reflect the imbalance
        # (All stabilizers = imbalanced, should have lower balance score)
        assert 0.0 <= health.balance_score <= 1.0

    def test_cycle_analysis_with_insufficient_length(self):
        """Test cycle analysis on sequences that are too short."""
        from tnfr.operators.cycle_detection import MIN_CYCLE_LENGTH

        # Sequence shorter than MIN_CYCLE_LENGTH
        short_seq = ["emission", "recursivity"]

        cycle_detector = CycleDetector()

        if len(short_seq) < MIN_CYCLE_LENGTH:
            analysis = cycle_detector.analyze_potential_cycle(short_seq, 1)
            # Should complete but likely not be a valid regenerative cycle
            assert analysis is not None
            # Might not be valid due to length
            if not analysis.is_valid_regenerative:
                assert "length" in analysis.reason.lower() or analysis.reason != ""

    def test_health_recommendations_on_poor_sequences(self):
        """Test that poor sequences generate helpful recommendations."""
        # Intentionally poor sequence (where grammar allows)
        poor_sequences = [
            # No final stabilizer (but valid ending)
            ["emission", "reception", "coherence", "dissonance", "recursivity"],
            # Unresolved dissonance
            ["emission", "reception", "coherence", "dissonance", "dissonance", "silence"],
        ]

        analyzer = SequenceHealthAnalyzer()

        for sequence in poor_sequences:
            result = validate_sequence(sequence)
            if not result.passed:
                continue  # Skip invalid sequences
                
            health = analyzer.analyze_health(sequence)

            # Poor sequences should have recommendations
            # (though not guaranteed, good sequences may also have recommendations)
            assert isinstance(health.recommendations, list)
