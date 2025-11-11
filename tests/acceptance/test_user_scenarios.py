"""Acceptance tests for Grammar 2.0 user scenarios.

Tests complete end-to-end workflows that users would perform with Grammar 2.0
features to ensure usability and correctness.
"""

import pytest

from tnfr.operators.grammar import validate_sequence
from tnfr.operators.health_analyzer import SequenceHealthAnalyzer
from tnfr.operators.patterns import AdvancedPatternDetector
from tnfr.operators.cycle_detection import CycleDetector


def validate_sequence_with_health(sequence):
    """Helper function to validate sequence and compute health metrics."""
    result = validate_sequence(sequence)
    if result.passed:
        analyzer = SequenceHealthAnalyzer()
        health = analyzer.analyze_health(sequence)
        # Attach health metrics to result
        result.health_metrics = health
    else:
        result.health_metrics = None
    return result


class TestUserScenarios:
    """End-to-end user workflow tests for Grammar 2.0."""

    def test_user_validates_therapeutic_intervention(self):
        """User workflow: Validate a therapeutic intervention sequence."""
        # User creates a therapeutic sequence
        sequence = [
            "emission",
            "reception",
            "coherence",
            "dissonance",
            "contraction",
            "coherence",
            "coupling",
            "silence",
        ]

        # Step 1: User validates the sequence
        result = validate_sequence_with_health(sequence)

        # User expects validation to pass
        assert result.passed, f"Sequence failed validation: {result.message}"

        # Step 2: User checks health metrics
        health = result.health_metrics
        assert health is not None

        # User expects good overall health
        assert health.overall_health > 0.7, "Therapeutic sequence should have high health"

        # Step 3: User reviews recommendations
        # Recommendations should be actionable strings
        for rec in health.recommendations:
            assert isinstance(rec, str)
            assert len(rec) > 0

    def test_user_analyzes_educational_sequence(self):
        """User workflow: Analyze an educational learning sequence."""
        # User creates an educational sequence
        # Must follow compatibility rules
        sequence = [
            "emission",
            "reception",
            "coherence",
            "dissonance",
            "mutation",
            "coherence",
            "recursivity",  # Valid ending after coherence
        ]

        # Step 1: Validate and get health
        result = validate_sequence_with_health(sequence)
        assert result.passed, f"Validation failed: {result.message}"

        # Step 2: Analyze pattern composition
        detector = AdvancedPatternDetector()
        composition = detector.analyze_sequence_composition(sequence)

        # User expects to see pattern analysis
        assert composition["primary_pattern"] is not None

        # User reviews structural health
        assert "structural_health" in composition
        health_data = composition["structural_health"]
        assert isinstance(health_data, dict)

        # User checks domain suitability
        assert "domain_suitability" in composition

    def test_user_optimizes_sequence_based_on_recommendations(self):
        """User workflow: Improve sequence based on health recommendations."""
        # User starts with a suboptimal sequence
        initial_sequence = [
            "emission",
            "reception",
            "coherence",
            "dissonance",
            "dissonance",  # Multiple dissonances
            "recursivity",  # Valid ending
        ]

        # Step 1: Check initial health
        analyzer = SequenceHealthAnalyzer()
        initial_health = analyzer.analyze_health(initial_sequence)
        initial_score = initial_health.overall_health

        # Step 2: User reads recommendations
        recommendations = initial_health.recommendations
        # Recommendations are optional - test works either way
        assert isinstance(recommendations, list)

        # Step 3: User improves sequence (adds stabilizer after dissonance)
        improved_sequence = [
            "emission",
            "reception",
            "coherence",
            "dissonance",
            "coherence",  # Added stabilizer
            "dissonance",
            "coherence",  # Added stabilizer
            "silence",
        ]

        # Step 4: Check improved health
        improved_health = analyzer.analyze_health(improved_sequence)
        improved_score = improved_health.overall_health

        # Health should be reasonable for both
        assert 0.0 <= initial_score <= 1.0
        assert 0.0 <= improved_score <= 1.0

    def test_user_compares_multiple_sequence_variants(self):
        """User workflow: Compare different sequence variants."""
        # User has multiple sequence options
        variants = {
            "variant_a": ["emission", "reception", "coherence", "silence"],
            "variant_b": ["emission", "reception", "coherence", "resonance", "silence"],
            "variant_c": [
                "emission",
                "reception",
                "coherence",
                "dissonance",
                "self_organization",
                "coherence",
                "silence",
            ],
        }

        analyzer = SequenceHealthAnalyzer()
        results = {}

        # Step 1: Analyze all variants
        for name, sequence in variants.items():
            result = validate_sequence_with_health(sequence)
            assert result.passed, f"{name} failed validation"
            results[name] = result.health_metrics

        # Step 2: User compares health scores
        for name, health in results.items():
            assert health.overall_health > 0.0

        # Step 3: User identifies best variant
        best_variant = max(results.items(), key=lambda x: x[1].overall_health)
        assert best_variant[0] in variants.keys()

    def test_user_detects_regenerative_cycles(self):
        """User workflow: Identify regenerative cycles in sequences."""
        # User creates a sequence with regenerators
        sequence = [
            "emission",
            "reception",
            "coherence",
            "resonance",
            "recursivity",  # Regenerator
        ]

        # Step 1: Validate sequence
        result = validate_sequence(sequence)
        assert result.passed

        # Step 2: User searches for regenerators
        regenerators = ["transition", "recursivity", "silence"]
        regenerator_positions = [i for i, op in enumerate(sequence) if op in regenerators]
        assert len(regenerator_positions) > 0, "Sequence should have regenerators"

        # Step 3: Analyze each regenerator position
        cycle_detector = CycleDetector()
        cycle_results = []

        for idx in regenerator_positions:
            analysis = cycle_detector.analyze_potential_cycle(sequence, idx)
            cycle_results.append((idx, analysis))

        # Step 4: User reviews cycle analysis
        for idx, analysis in cycle_results:
            assert analysis is not None
            assert analysis.cycle_type is not None

            # User checks if it's a valid regenerative cycle
            if analysis.is_valid_regenerative:
                assert analysis.health_score >= 0.6

    def test_user_explores_pattern_taxonomy(self):
        """User workflow: Understand different pattern types."""
        # User tests examples of each major pattern type
        pattern_examples = {
            "linear": ["emission", "reception", "coherence", "silence"],
            "hierarchical": [
                "emission",
                "reception",
                "coherence",
                "self_organization",
                "silence",
            ],
            "bifurcated": [
                "emission",
                "reception",
                "coherence",
                "dissonance",
                "mutation",
                "silence",
            ],
        }

        detector = AdvancedPatternDetector()

        # User detects patterns
        detected_patterns = {}
        for pattern_type, sequence in pattern_examples.items():
            # Validate first
            result = validate_sequence(sequence)
            if not result.passed:
                continue

            pattern = detector.detect_pattern(sequence)
            detected_patterns[pattern_type] = pattern.value

        # User verifies patterns were detected
        assert len(detected_patterns) > 0

    def test_user_workflow_complete_analysis_pipeline(self):
        """User workflow: Complete analysis from validation to recommendations."""
        # User has a complex organizational sequence
        # Must follow compatibility rules
        sequence = [
            "emission",
            "reception",
            "coupling",
            "coherence",
            "dissonance",  # Moved before resonance to avoid incompatibility
            "self_organization",
            "coherence",
            "resonance",
            "transition",
            "silence",
        ]

        # Step 1: Validate with health
        result = validate_sequence_with_health(sequence)
        assert result.passed, f"Sequence must be valid: {result.message}"

        # Step 2: Review overall health
        health = result.health_metrics
        overall = health.overall_health
        assert 0.0 <= overall <= 1.0

        # Step 3: Check specific metrics
        metrics_to_review = {
            "coherence_index": health.coherence_index,
            "balance_score": health.balance_score,
            "sustainability_index": health.sustainability_index,
            "complexity_efficiency": health.complexity_efficiency,
        }

        for metric_name, value in metrics_to_review.items():
            assert 0.0 <= value <= 1.0, f"{metric_name} out of range"

        # Step 4: Get pattern insights
        detector = AdvancedPatternDetector()
        composition = detector.analyze_sequence_composition(sequence)

        primary_pattern = composition["primary_pattern"]
        assert primary_pattern is not None

        # Step 5: Review recommendations
        recommendations = health.recommendations
        assert isinstance(recommendations, list)

        # Step 6: Check for regenerative cycles
        cycle_detector = CycleDetector()
        transition_idx = sequence.index("transition")
        cycle_analysis = cycle_detector.analyze_potential_cycle(sequence, transition_idx)
        assert cycle_analysis is not None

    def test_user_builds_custom_sequence_incrementally(self):
        """User workflow: Build and test sequence step by step."""
        # User starts with minimal sequence
        sequence = ["emission", "reception", "coherence", "silence"]

        analyzer = SequenceHealthAnalyzer()

        # Step 1: Test base sequence
        base_health = analyzer.analyze_health(sequence)
        base_score = base_health.overall_health

        # Step 2: User adds resonance for amplification
        sequence_v2 = sequence[:-1] + ["resonance"] + [sequence[-1]]
        health_v2 = analyzer.analyze_health(sequence_v2)

        # Step 3: User adds exploration phase
        sequence_v3 = sequence_v2[:-1] + ["dissonance", "coherence"] + [sequence_v2[-1]]
        health_v3 = analyzer.analyze_health(sequence_v3)

        # All versions should have valid health
        assert 0.0 <= base_score <= 1.0
        assert 0.0 <= health_v2.overall_health <= 1.0
        assert 0.0 <= health_v3.overall_health <= 1.0

    def test_user_diagnoses_failed_validation(self):
        """User workflow: Understand why a sequence fails validation."""
        # User tries an invalid sequence
        invalid_sequence = [
            "dissonance",  # Invalid start
            "mutation",
            "coherence",
        ]

        # Step 1: User validates
        result = validate_sequence(invalid_sequence)

        # Step 2: User sees it failed
        assert not result.passed

        # Step 3: User reviews error message
        assert result.message is not None
        assert len(result.message) > 0

        # Step 4: User checks metadata for hints
        assert result.metadata is not None

        # User learns what went wrong and can fix it
        # (In this case, needs to start with emission or recursivity)
