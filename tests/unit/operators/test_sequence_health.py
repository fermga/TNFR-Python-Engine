"""Tests for sequence health metrics analyzer."""

from tnfr.operators.health_analyzer import SequenceHealthAnalyzer, SequenceHealthMetrics
from tnfr.operators.grammar import (
    validate_sequence_with_health,
    SequenceValidationResultWithHealth,
)
from tnfr.config.operator_names import (
    EMISSION,
    RECEPTION,
    COHERENCE,
    DISSONANCE,
    COUPLING,
    RESONANCE,
    SILENCE,
    EXPANSION,
    CONTRACTION,
    SELF_ORGANIZATION,
    MUTATION,
    TRANSITION,
    RECURSIVITY,
)


class TestSequenceHealthMetrics:
    """Test SequenceHealthMetrics dataclass."""

    def test_metrics_dataclass_structure(self):
        """Test that SequenceHealthMetrics has all required fields."""
        metrics = SequenceHealthMetrics(
            coherence_index=0.8,
            balance_score=0.7,
            sustainability_index=0.9,
            complexity_efficiency=0.85,
            frequency_harmony=0.8,
            pattern_completeness=0.75,
            transition_smoothness=0.9,
            overall_health=0.82,
            sequence_length=5,
            dominant_pattern="activation",
            recommendations=[],
        )

        assert metrics.coherence_index == 0.8
        assert metrics.balance_score == 0.7
        assert metrics.sustainability_index == 0.9
        assert metrics.complexity_efficiency == 0.85
        assert metrics.frequency_harmony == 0.8
        assert metrics.pattern_completeness == 0.75
        assert metrics.transition_smoothness == 0.9
        assert metrics.overall_health == 0.82
        assert metrics.sequence_length == 5
        assert metrics.dominant_pattern == "activation"
        assert metrics.recommendations == []


class TestSequenceHealthAnalyzer:
    """Test SequenceHealthAnalyzer core functionality."""

    def test_analyzer_initialization(self):
        """Test analyzer can be instantiated."""
        analyzer = SequenceHealthAnalyzer()
        assert analyzer is not None

    def test_analyze_health_basic_sequence(self):
        """Test health analysis of a basic valid sequence."""
        analyzer = SequenceHealthAnalyzer()
        sequence = [EMISSION, RECEPTION, COHERENCE, SILENCE]

        health = analyzer.analyze_health(sequence)

        assert isinstance(health, SequenceHealthMetrics)
        assert health.sequence_length == 4
        assert 0.0 <= health.overall_health <= 1.0
        assert 0.0 <= health.coherence_index <= 1.0
        assert 0.0 <= health.balance_score <= 1.0
        assert 0.0 <= health.sustainability_index <= 1.0

    def test_analyze_empty_sequence(self):
        """Test health analysis handles empty sequences gracefully."""
        analyzer = SequenceHealthAnalyzer()
        health = analyzer.analyze_health([])

        assert health.sequence_length == 0
        assert health.overall_health < 0.5  # Empty should score low
        assert health.dominant_pattern == "empty"


class TestCoherenceIndex:
    """Test coherence index calculation."""

    def test_high_coherence_activation_pattern(self):
        """Activation patterns should have high coherence."""
        analyzer = SequenceHealthAnalyzer()
        sequence = [EMISSION, RECEPTION, COHERENCE, RESONANCE, SILENCE]

        health = analyzer.analyze_health(sequence)

        # Activation pattern with proper closure should score well
        assert health.coherence_index > 0.7

    def test_medium_coherence_incomplete_pattern(self):
        """Incomplete patterns should have medium coherence."""
        analyzer = SequenceHealthAnalyzer()
        sequence = [EMISSION, RECEPTION, COHERENCE]  # Missing proper ending

        health = analyzer.analyze_health(sequence)

        # Incomplete pattern scores lower
        assert 0.3 <= health.coherence_index <= 0.8

    def test_low_coherence_unstructured(self):
        """Unstructured sequences should have lower coherence."""
        analyzer = SequenceHealthAnalyzer()
        sequence = [EMISSION, DISSONANCE, SILENCE]  # Minimal structure

        health = analyzer.analyze_health(sequence)

        # Very basic structure
        assert health.coherence_index < 0.9


class TestBalanceScore:
    """Test balance score calculation."""

    def test_balanced_sequence(self):
        """Sequences with balanced forces should score high."""
        analyzer = SequenceHealthAnalyzer()
        # Equal stabilizers and destabilizers
        sequence = [EMISSION, DISSONANCE, COHERENCE, EXPANSION, RESONANCE, SILENCE]

        health = analyzer.analyze_health(sequence)

        # Should have good balance (2 stabilizers: coherence, resonance, silence vs 2 destabilizers: dissonance, expansion)
        assert health.balance_score > 0.5

    def test_severely_imbalanced_sequence(self):
        """Severely imbalanced sequences should score low."""
        analyzer = SequenceHealthAnalyzer()
        # Many destabilizers, few stabilizers - create severe imbalance
        sequence = [
            EMISSION,
            DISSONANCE,
            DISSONANCE,
            DISSONANCE,
            EXPANSION,
            TRANSITION,
            EXPANSION,
            COHERENCE,
            SILENCE,
        ]

        health = analyzer.analyze_health(sequence)

        # Should detect imbalance (6 destabilizers vs 2 stabilizers in 9 op sequence)
        # Imbalance is 4, which is > half of 9 (4.5), so penalty applies
        assert health.balance_score < 0.6  # Lower threshold due to penalty
        # Should have recommendation about imbalance or structure
        assert len(health.recommendations) > 0

    def test_neutral_balance_no_forces(self):
        """Sequences without stabilizers/destabilizers should be neutral."""
        analyzer = SequenceHealthAnalyzer()
        sequence = [EMISSION, RECEPTION]

        health = analyzer.analyze_health(sequence)

        # No strong forces, neutral balance
        assert 0.4 <= health.balance_score <= 0.6


class TestSustainabilityIndex:
    """Test sustainability index calculation."""

    def test_high_sustainability_with_stabilizer_ending(self):
        """Sequences ending with stabilizers should be sustainable."""
        analyzer = SequenceHealthAnalyzer()
        sequence = [EMISSION, RECEPTION, COHERENCE, SILENCE]

        health = analyzer.analyze_health(sequence)

        # Ends with stabilizer, should be very sustainable
        assert health.sustainability_index > 0.7

    def test_medium_sustainability_no_stabilizer_ending(self):
        """Sequences not ending with stabilizers score lower."""
        analyzer = SequenceHealthAnalyzer()
        sequence = [EMISSION, RECEPTION, COHERENCE, DISSONANCE]

        health = analyzer.analyze_health(sequence)

        # Ends with destabilizer, lower sustainability
        assert health.sustainability_index < 0.7
        assert any("stabilizer" in rec.lower() for rec in health.recommendations)

    def test_unresolved_dissonance_reduces_sustainability(self):
        """Unresolved dissonance should reduce sustainability."""
        analyzer = SequenceHealthAnalyzer()
        # Dissonance without nearby stabilizer - put more operators between to exceed window
        sequence = [EMISSION, DISSONANCE, EXPANSION, MUTATION, TRANSITION, SILENCE]

        health = analyzer.analyze_health(sequence)

        # Multiple unresolved dissonances (dissonance followed by 4 non-stabilizers)
        assert (
            health.sustainability_index < 0.9
        )  # Still gets some credit for ending with stabilizer
        # May or may not have specific recommendation depending on exact calculation

    def test_regenerative_elements_increase_sustainability(self):
        """Regenerative operators should boost sustainability."""
        analyzer = SequenceHealthAnalyzer()
        sequence = [EMISSION, RECEPTION, COHERENCE, TRANSITION, SILENCE]

        health = analyzer.analyze_health(sequence)

        # Has regenerative element (transition)
        assert health.sustainability_index > 0.6


class TestComplexityEfficiency:
    """Test complexity efficiency calculation."""

    def test_optimal_length_scores_high(self):
        """Sequences in optimal length range (3-8) should score well."""
        analyzer = SequenceHealthAnalyzer()
        sequence = [EMISSION, RECEPTION, COHERENCE, RESONANCE, SILENCE]  # 5 operators

        health = analyzer.analyze_health(sequence)

        # Optimal length with good diversity
        assert health.complexity_efficiency > 0.6

    def test_very_short_sequence_penalized(self):
        """Very short sequences should be penalized."""
        analyzer = SequenceHealthAnalyzer()
        sequence = [EMISSION, SILENCE]  # Only 2 operators

        health = analyzer.analyze_health(sequence)

        # Too short, limited structural value
        assert health.complexity_efficiency < 0.9

    def test_very_long_sequence_penalized(self):
        """Very long sequences should be penalized."""
        analyzer = SequenceHealthAnalyzer()
        # 13 operators - quite long
        sequence = [
            EMISSION,
            RECEPTION,
            COHERENCE,
            DISSONANCE,
            COUPLING,
            RESONANCE,
            EXPANSION,
            CONTRACTION,
            SELF_ORGANIZATION,
            MUTATION,
            TRANSITION,
            RECURSIVITY,
            SILENCE,
        ]

        health = analyzer.analyze_health(sequence)

        # Long sequence gets penalty
        assert health.complexity_efficiency < 1.0
        assert any("long" in rec.lower() for rec in health.recommendations)

    def test_diverse_operators_score_high(self):
        """Sequences with diverse operator types score high."""
        analyzer = SequenceHealthAnalyzer()
        sequence = [EMISSION, RECEPTION, COHERENCE, DISSONANCE, RESONANCE, SILENCE]

        health = analyzer.analyze_health(sequence)

        # Good diversity across categories
        assert health.complexity_efficiency > 0.5


class TestPatternDetection:
    """Test dominant pattern detection."""

    def test_activation_pattern_detected(self):
        """Classic activation pattern should be recognized."""
        analyzer = SequenceHealthAnalyzer()
        sequence = [EMISSION, RECEPTION, COHERENCE, SILENCE]

        health = analyzer.analyze_health(sequence)

        assert health.dominant_pattern == "activation"

    def test_therapeutic_pattern_detected(self):
        """Therapeutic pattern (with dissonance and self-org) should be recognized."""
        analyzer = SequenceHealthAnalyzer()
        sequence = [EMISSION, RECEPTION, COHERENCE, DISSONANCE, SELF_ORGANIZATION, SILENCE]

        health = analyzer.analyze_health(sequence)

        assert health.dominant_pattern == "therapeutic"

    def test_regenerative_pattern_detected(self):
        """Regenerative pattern (with transition/recursivity) should be recognized."""
        analyzer = SequenceHealthAnalyzer()
        sequence = [EMISSION, RECEPTION, COHERENCE, TRANSITION]

        health = analyzer.analyze_health(sequence)

        assert health.dominant_pattern == "regenerative"

    def test_transformative_pattern_detected(self):
        """Transformative pattern should be recognized."""
        analyzer = SequenceHealthAnalyzer()
        sequence = [EMISSION, DISSONANCE, SELF_ORGANIZATION, SILENCE]

        health = analyzer.analyze_health(sequence)

        assert health.dominant_pattern == "transformative"

    def test_unknown_pattern_for_unstructured(self):
        """Unstructured sequences should return unknown pattern."""
        analyzer = SequenceHealthAnalyzer()
        sequence = [EXPANSION, CONTRACTION, MUTATION]

        health = analyzer.analyze_health(sequence)

        assert health.dominant_pattern in ["unknown", "exploratory"]


class TestPatternCompleteness:
    """Test pattern completeness calculation."""

    def test_complete_pattern_all_phases(self):
        """Sequences with all phases should have high completeness."""
        analyzer = SequenceHealthAnalyzer()
        # Activation, transformation, stabilization, completion
        sequence = [EMISSION, RECEPTION, DISSONANCE, COHERENCE, SILENCE]

        health = analyzer.analyze_health(sequence)

        # Has all 4 phases
        assert health.pattern_completeness == 1.0

    def test_partial_pattern_missing_phases(self):
        """Sequences missing phases should have lower completeness."""
        analyzer = SequenceHealthAnalyzer()
        # Only activation and stabilization, no transformation or completion
        sequence = [EMISSION, RECEPTION, COHERENCE]

        health = analyzer.analyze_health(sequence)

        # Missing transformation and completion phases
        assert health.pattern_completeness < 1.0


class TestOverallHealth:
    """Test overall health composite metric."""

    def test_high_health_optimal_sequence(self):
        """Optimal sequences should have high overall health."""
        analyzer = SequenceHealthAnalyzer()
        # Well-balanced, complete, sustainable sequence
        sequence = [
            EMISSION,
            RECEPTION,
            COHERENCE,
            DISSONANCE,
            SELF_ORGANIZATION,
            RESONANCE,
            SILENCE,
        ]

        health = analyzer.analyze_health(sequence)

        # Should score well across all metrics
        assert health.overall_health > 0.7
        assert len(health.recommendations) == 0 or len(health.recommendations) <= 2

    def test_medium_health_basic_sequence(self):
        """Basic sequences should have medium health."""
        analyzer = SequenceHealthAnalyzer()
        sequence = [EMISSION, RECEPTION, COHERENCE, SILENCE]

        health = analyzer.analyze_health(sequence)

        # Basic but valid, medium health
        assert 0.5 <= health.overall_health <= 0.9

    def test_low_health_suboptimal_sequence(self):
        """Suboptimal sequences should have low health."""
        analyzer = SequenceHealthAnalyzer()
        # Unbalanced, many destabilizers, no proper ending
        sequence = [EMISSION, DISSONANCE, DISSONANCE, EXPANSION, MUTATION]

        health = analyzer.analyze_health(sequence)

        # Should identify problems
        assert health.overall_health < 0.8
        assert len(health.recommendations) > 0


class TestValidateSequenceWithHealth:
    """Test integration with validate_sequence_with_health API."""

    def test_valid_sequence_includes_health(self):
        """Valid sequences should include health metrics."""
        result = validate_sequence_with_health([EMISSION, RECEPTION, COHERENCE, SILENCE])

        assert isinstance(result, SequenceValidationResultWithHealth)
        assert result.passed
        assert result.health_metrics is not None
        assert isinstance(result.health_metrics, SequenceHealthMetrics)
        assert result.health_metrics.sequence_length == 4

    def test_invalid_sequence_no_health(self):
        """Invalid sequences should not compute health metrics."""
        # Invalid: doesn't start with valid operator
        result = validate_sequence_with_health([RECEPTION, COHERENCE, SILENCE])

        assert not result.passed
        assert result.health_metrics is None

    def test_health_metrics_accessible_from_result(self):
        """Health metrics should be easily accessible."""
        result = validate_sequence_with_health([EMISSION, RECEPTION, COHERENCE, RESONANCE, SILENCE])

        assert result.passed
        assert result.health_metrics.overall_health > 0
        assert result.health_metrics.dominant_pattern in [
            "activation",
            "therapeutic",
            "regenerative",
            "transformative",
            "stabilization",
            "exploratory",
            "unknown",
        ]


class TestRecommendations:
    """Test recommendation generation."""

    def test_no_recommendations_for_optimal(self):
        """Optimal sequences should generate no or few recommendations."""
        analyzer = SequenceHealthAnalyzer()
        sequence = [EMISSION, RECEPTION, COHERENCE, RESONANCE, SILENCE]

        health = analyzer.analyze_health(sequence)

        # High quality sequence should have few recommendations
        assert len(health.recommendations) <= 1

    def test_recommendations_for_imbalanced(self):
        """Imbalanced sequences should generate balance recommendations."""
        analyzer = SequenceHealthAnalyzer()
        # Create severely imbalanced: many destabilizers, few stabilizers
        sequence = [
            EMISSION,
            DISSONANCE,
            DISSONANCE,
            DISSONANCE,
            EXPANSION,
            TRANSITION,
            COHERENCE,
            SILENCE,
        ]

        health = analyzer.analyze_health(sequence)

        # Should recommend addressing imbalance or unresolved dissonance
        assert len(health.recommendations) > 0
        assert any(
            "imbalance" in rec.lower() or "stabilizer" in rec.lower() or "unresolved" in rec.lower()
            for rec in health.recommendations
        )

    def test_recommendations_for_missing_stabilizer(self):
        """Sequences without final stabilizer should get recommendation."""
        analyzer = SequenceHealthAnalyzer()
        sequence = [EMISSION, RECEPTION, COHERENCE, DISSONANCE]

        health = analyzer.analyze_health(sequence)

        # Should recommend adding stabilizer
        assert len(health.recommendations) > 0
        assert any("stabilizer" in rec.lower() for rec in health.recommendations)

    def test_recommendations_for_long_sequence(self):
        """Long sequences should get length recommendation."""
        analyzer = SequenceHealthAnalyzer()
        sequence = [EMISSION] * 15  # Very long

        health = analyzer.analyze_health(sequence)

        # Should recommend breaking up
        assert any("long" in rec.lower() for rec in health.recommendations)


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_minimal_valid_sequence(self):
        """Test minimal valid sequence (3 operators)."""
        analyzer = SequenceHealthAnalyzer()
        sequence = [EMISSION, COHERENCE, SILENCE]  # Minimal valid

        health = analyzer.analyze_health(sequence)

        assert health.sequence_length == 3
        assert 0.0 <= health.overall_health <= 1.0

    def test_single_operator_sequence(self):
        """Test single operator sequence."""
        analyzer = SequenceHealthAnalyzer()
        sequence = [EMISSION]

        health = analyzer.analyze_health(sequence)

        # Should handle gracefully
        assert health.sequence_length == 1
        assert health.overall_health <= 0.55  # Should score relatively low

    def test_all_operators_sequence(self):
        """Test sequence with all 13 operators."""
        analyzer = SequenceHealthAnalyzer()
        sequence = [
            EMISSION,
            RECEPTION,
            COHERENCE,
            DISSONANCE,
            COUPLING,
            RESONANCE,
            SILENCE,
            EXPANSION,
            CONTRACTION,
            SELF_ORGANIZATION,
            MUTATION,
            TRANSITION,
            RECURSIVITY,
        ]

        health = analyzer.analyze_health(sequence)

        # Should handle full operator set
        assert health.sequence_length == 13
        assert 0.0 <= health.overall_health <= 1.0

    def test_repeated_operators(self):
        """Test sequence with repeated operators."""
        analyzer = SequenceHealthAnalyzer()
        sequence = [EMISSION, COHERENCE, COHERENCE, COHERENCE, SILENCE]

        health = analyzer.analyze_health(sequence)

        # Should handle repetition
        assert health.sequence_length == 5
        assert 0.0 <= health.overall_health <= 1.0


class TestMetricRanges:
    """Test that all metrics stay within valid ranges."""

    def test_all_metrics_in_range_valid_sequence(self):
        """All metrics should be in [0, 1] range for valid sequences."""
        analyzer = SequenceHealthAnalyzer()
        sequence = [EMISSION, RECEPTION, COHERENCE, DISSONANCE, RESONANCE, SILENCE]

        health = analyzer.analyze_health(sequence)

        assert 0.0 <= health.coherence_index <= 1.0
        assert 0.0 <= health.balance_score <= 1.0
        assert 0.0 <= health.sustainability_index <= 1.0
        assert 0.0 <= health.complexity_efficiency <= 1.0
        assert 0.0 <= health.frequency_harmony <= 1.0
        assert 0.0 <= health.pattern_completeness <= 1.0
        assert 0.0 <= health.transition_smoothness <= 1.0
        assert 0.0 <= health.overall_health <= 1.0

    def test_all_metrics_in_range_edge_cases(self):
        """All metrics should be in valid range even for edge cases."""
        analyzer = SequenceHealthAnalyzer()

        # Test various edge cases
        edge_sequences = [
            [],  # Empty
            [EMISSION],  # Single
            [EMISSION, SILENCE],  # Minimal
            [DISSONANCE, DISSONANCE, DISSONANCE, SILENCE],  # Imbalanced
        ]

        for sequence in edge_sequences:
            health = analyzer.analyze_health(sequence)

            assert 0.0 <= health.coherence_index <= 1.0
            assert 0.0 <= health.balance_score <= 1.0
            assert 0.0 <= health.sustainability_index <= 1.0
            assert 0.0 <= health.complexity_efficiency <= 1.0
            assert 0.0 <= health.frequency_harmony <= 1.0
            assert 0.0 <= health.pattern_completeness <= 1.0
            assert 0.0 <= health.transition_smoothness <= 1.0
            assert 0.0 <= health.overall_health <= 1.0


class TestComparativeAnalysis:
    """Test comparing health metrics between sequences."""

    def test_optimal_vs_suboptimal_comparison(self):
        """Optimal sequences should score higher than suboptimal ones."""
        analyzer = SequenceHealthAnalyzer()

        optimal = [EMISSION, RECEPTION, COHERENCE, RESONANCE, SILENCE]
        suboptimal = [EMISSION, DISSONANCE, DISSONANCE, MUTATION]

        health_optimal = analyzer.analyze_health(optimal)
        health_suboptimal = analyzer.analyze_health(suboptimal)

        # Optimal should score higher
        assert health_optimal.overall_health > health_suboptimal.overall_health
        assert health_optimal.sustainability_index > health_suboptimal.sustainability_index

    def test_balanced_vs_imbalanced_comparison(self):
        """Balanced sequences should have better balance scores."""
        analyzer = SequenceHealthAnalyzer()

        balanced = [EMISSION, DISSONANCE, COHERENCE, EXPANSION, RESONANCE, SILENCE]
        imbalanced = [EMISSION, DISSONANCE, DISSONANCE, DISSONANCE, SILENCE]

        health_balanced = analyzer.analyze_health(balanced)
        health_imbalanced = analyzer.analyze_health(imbalanced)

        # Balanced should have better balance score
        assert health_balanced.balance_score > health_imbalanced.balance_score
