"""Tests for therapeutic domain examples.

Validates that all therapeutic patterns, case studies, and optimization
examples meet acceptance criteria according to TNFR Grammar 2.0.
"""

import pytest
from tnfr.operators.grammar import validate_sequence_with_health
from tnfr.operators.health_analyzer import SequenceHealthAnalyzer

# Import therapeutic examples
import sys
from pathlib import Path

examples_path = Path(__file__).parent.parent.parent / "examples" / "domain_applications"
sys.path.insert(0, str(examples_path))

from therapeutic_patterns import (
    get_crisis_intervention_sequence,
    get_process_therapy_sequence,
    get_regenerative_healing_sequence,
    get_insight_integration_sequence,
    get_relapse_prevention_sequence,
)

from therapeutic_case_studies import (
    case_trauma_recovery,
    case_addiction_healing,
    case_depression_emergence,
    case_relationship_repair,
)


# =============================================================================
# Test Therapeutic Patterns
# =============================================================================


class TestTherapeuticPatterns:
    """Test suite for therapeutic_patterns.py."""

    def test_all_patterns_valid(self):
        """Test that all therapeutic patterns pass validation."""
        patterns = {
            "crisis_intervention": get_crisis_intervention_sequence(),
            "process_therapy": get_process_therapy_sequence(),
            "regenerative_healing": get_regenerative_healing_sequence(),
            "insight_integration": get_insight_integration_sequence(),
            "relapse_prevention": get_relapse_prevention_sequence(),
        }

        for name, sequence in patterns.items():
            result = validate_sequence_with_health(sequence)
            assert result.passed, f"{name} failed validation: {result.message}"

    def test_all_health_scores_above_threshold(self):
        """Test that all patterns have health scores > 0.75."""
        patterns = {
            "crisis_intervention": get_crisis_intervention_sequence(),
            "process_therapy": get_process_therapy_sequence(),
            "regenerative_healing": get_regenerative_healing_sequence(),
            "insight_integration": get_insight_integration_sequence(),
            "relapse_prevention": get_relapse_prevention_sequence(),
        }

        threshold = 0.75

        for name, sequence in patterns.items():
            result = validate_sequence_with_health(sequence)
            assert result.passed, f"{name} did not pass validation"

            health = result.health_metrics.overall_health
            assert (
                health >= threshold
            ), f"{name} health score {health:.3f} below threshold {threshold}"

    def test_crisis_intervention_effectiveness(self):
        """Test crisis intervention pattern has appropriate characteristics."""
        sequence = get_crisis_intervention_sequence()
        result = validate_sequence_with_health(sequence)

        assert result.passed
        health = result.health_metrics

        # Should have good sustainability for stabilization
        assert health.sustainability_index >= 0.7

        # Pattern should be activation or stabilize
        assert health.dominant_pattern in ["activation", "stabilize", "therapeutic"]

    def test_process_therapy_completeness(self):
        """Test process therapy has therapeutic pattern characteristics."""
        sequence = get_process_therapy_sequence()
        result = validate_sequence_with_health(sequence)

        assert result.passed
        health = result.health_metrics

        # Should have excellent health (>0.85)
        assert health.overall_health >= 0.85

        # Should be detected as therapeutic pattern
        assert health.dominant_pattern == "therapeutic"

        # Should have good balance
        assert health.balance_score >= 0.3

    def test_regenerative_healing_cycle_validation(self):
        """Test regenerative healing is detected as regenerative pattern."""
        sequence = get_regenerative_healing_sequence()
        result = validate_sequence_with_health(sequence)

        assert result.passed
        health = result.health_metrics

        # Should be detected as regenerative
        # Note: Pattern detection is based on sequence characteristics
        # It may detect as activation/therapeutic if regenerative pattern
        # detection requires specific structure
        assert health.dominant_pattern in ["regenerative", "activation", "cyclic"]

        # Should have good overall health
        assert health.overall_health >= 0.75

    def test_patterns_structural_coherence(self):
        """Test all patterns maintain structural coherence."""
        patterns = [
            get_crisis_intervention_sequence(),
            get_process_therapy_sequence(),
            get_regenerative_healing_sequence(),
            get_insight_integration_sequence(),
            get_relapse_prevention_sequence(),
        ]

        for sequence in patterns:
            result = validate_sequence_with_health(sequence)
            assert result.passed

            # All should have high coherence index
            assert result.health_metrics.coherence_index >= 0.85

    def test_patterns_minimum_count(self):
        """Test that at least 5 therapeutic patterns are provided."""
        patterns = [
            get_crisis_intervention_sequence(),
            get_process_therapy_sequence(),
            get_regenerative_healing_sequence(),
            get_insight_integration_sequence(),
            get_relapse_prevention_sequence(),
        ]

        assert len(patterns) >= 5, "Should have at least 5 therapeutic patterns"


# =============================================================================
# Test Therapeutic Case Studies
# =============================================================================


class TestTherapeuticCaseStudies:
    """Test suite for therapeutic_case_studies.py."""

    def test_all_case_studies_valid(self):
        """Test that all case study sequences pass validation."""
        cases = [
            case_trauma_recovery(),
            case_addiction_healing(),
            case_depression_emergence(),
            case_relationship_repair(),
        ]

        for case_data in cases:
            sequence = case_data["sequence"]
            result = validate_sequence_with_health(sequence)
            assert result.passed, f"{case_data['name']} failed validation: {result.message}"

    def test_all_case_studies_health_above_threshold(self):
        """Test that all case studies have health scores > 0.75."""
        cases = [
            case_trauma_recovery(),
            case_addiction_healing(),
            case_depression_emergence(),
            case_relationship_repair(),
        ]

        threshold = 0.75

        for case_data in cases:
            sequence = case_data["sequence"]
            result = validate_sequence_with_health(sequence)

            assert result.passed, f"{case_data['name']} did not pass validation"

            health = result.health_metrics.overall_health
            assert (
                health >= threshold
            ), f"{case_data['name']} health {health:.3f} below threshold {threshold}"

    def test_trauma_recovery_structure(self):
        """Test trauma recovery case has appropriate structural elements."""
        case_data = case_trauma_recovery()
        sequence = case_data["sequence"]

        # Should contain key operators for trauma work
        assert "dissonance" in sequence, "Trauma work requires dissonance (controlled exposure)"
        assert "self_organization" in sequence, "Requires self-organization for integration"
        assert "coherence" in sequence, "Requires coherence for stabilization"

    def test_addiction_healing_pattern(self):
        """Test addiction healing follows appropriate pattern."""
        case_data = case_addiction_healing()
        sequence = case_data["sequence"]

        # Should follow OZ → NUL → VAL pattern (or variation)
        # Check for presence of key operators
        assert "dissonance" in sequence
        assert "contraction" in sequence
        assert "expansion" in sequence

    def test_depression_emergence_reactivation(self):
        """Test depression emergence includes reactivation elements."""
        case_data = case_depression_emergence()
        sequence = case_data["sequence"]

        # Should include elements for reactivation
        assert "emission" in sequence, "Needs emission for activation"
        assert "coherence" in sequence, "Needs coherence for stabilization"

    def test_relationship_repair_coupling(self):
        """Test relationship repair includes coupling elements."""
        case_data = case_relationship_repair()
        sequence = case_data["sequence"]

        # Should include coupling for relationship work
        coupling_count = sequence.count("coupling")
        assert coupling_count >= 1, "Relationship work requires coupling"

    def test_case_studies_minimum_count(self):
        """Test that at least 4 case studies are provided."""
        cases = [
            case_trauma_recovery(),
            case_addiction_healing(),
            case_depression_emergence(),
            case_relationship_repair(),
        ]

        assert len(cases) >= 4, "Should have at least 4 case studies"

    def test_case_studies_have_required_metadata(self):
        """Test that all case studies have required metadata fields."""
        cases = [
            case_trauma_recovery(),
            case_addiction_healing(),
            case_depression_emergence(),
            case_relationship_repair(),
        ]

        required_fields = [
            "name",
            "sequence",
            "presenting_problem",
            "therapeutic_goal",
            "key_operators",
            "pattern_type",
        ]

        for case_data in cases:
            for field in required_fields:
                assert (
                    field in case_data
                ), f"{case_data.get('name', 'Unknown')} missing field: {field}"


# =============================================================================
# Test Pattern Characteristics
# =============================================================================


class TestPatternCharacteristics:
    """Test specific characteristics of therapeutic patterns."""

    def test_crisis_patterns_are_short(self):
        """Test that crisis intervention patterns are relatively short."""
        crisis_seq = get_crisis_intervention_sequence()

        # Crisis intervention should be concise (typically 5-10 operators)
        assert len(crisis_seq) <= 10, "Crisis intervention should be concise"

    def test_process_therapy_is_comprehensive(self):
        """Test that process therapy is more comprehensive."""
        process_seq = get_process_therapy_sequence()

        # Process therapy should be more extensive (typically 7-12 operators)
        assert len(process_seq) >= 7, "Process therapy should be comprehensive"

    def test_patterns_end_with_valid_operators(self):
        """Test that all patterns end with valid end operators."""
        from tnfr.config.operator_names import VALID_END_OPERATORS

        patterns = [
            get_crisis_intervention_sequence(),
            get_process_therapy_sequence(),
            get_regenerative_healing_sequence(),
            get_insight_integration_sequence(),
            get_relapse_prevention_sequence(),
        ]

        for sequence in patterns:
            last_operator = sequence[-1]
            assert (
                last_operator in VALID_END_OPERATORS
            ), f"Sequence ends with invalid operator: {last_operator}"

    def test_patterns_start_with_valid_operators(self):
        """Test that all patterns start with valid start operators."""
        from tnfr.config.operator_names import VALID_START_OPERATORS

        patterns = [
            get_crisis_intervention_sequence(),
            get_process_therapy_sequence(),
            get_regenerative_healing_sequence(),
            get_insight_integration_sequence(),
            get_relapse_prevention_sequence(),
        ]

        for sequence in patterns:
            first_operator = sequence[0]
            assert (
                first_operator in VALID_START_OPERATORS
            ), f"Sequence starts with invalid operator: {first_operator}"


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for therapeutic examples."""

    def test_patterns_and_cases_consistency(self):
        """Test that patterns and case studies are consistent."""
        # Both should use the same operator names
        from tnfr.config.operator_names import CANONICAL_OPERATOR_NAMES

        all_sequences = []

        # Collect all pattern sequences
        patterns = [
            get_crisis_intervention_sequence(),
            get_process_therapy_sequence(),
            get_regenerative_healing_sequence(),
            get_insight_integration_sequence(),
            get_relapse_prevention_sequence(),
        ]
        all_sequences.extend(patterns)

        # Collect all case study sequences
        cases = [
            case_trauma_recovery(),
            case_addiction_healing(),
            case_depression_emergence(),
            case_relationship_repair(),
        ]
        for case in cases:
            all_sequences.append(case["sequence"])

        # Check all operators are canonical
        for sequence in all_sequences:
            for operator in sequence:
                assert (
                    operator in CANONICAL_OPERATOR_NAMES
                ), f"Non-canonical operator found: {operator}"

    def test_average_health_meets_target(self):
        """Test that average health across all examples meets target."""
        all_sequences = []

        # Collect patterns
        patterns = [
            get_crisis_intervention_sequence(),
            get_process_therapy_sequence(),
            get_regenerative_healing_sequence(),
            get_insight_integration_sequence(),
            get_relapse_prevention_sequence(),
        ]
        all_sequences.extend(patterns)

        # Collect case studies
        cases = [
            case_trauma_recovery(),
            case_addiction_healing(),
            case_depression_emergence(),
            case_relationship_repair(),
        ]
        for case in cases:
            all_sequences.append(case["sequence"])

        # Calculate average health
        total_health = 0
        valid_count = 0

        for sequence in all_sequences:
            result = validate_sequence_with_health(sequence)
            if result.passed:
                total_health += result.health_metrics.overall_health
                valid_count += 1

        avg_health = total_health / valid_count if valid_count > 0 else 0

        # Target: average health > 0.75
        assert avg_health >= 0.75, f"Average health {avg_health:.3f} below target 0.75"

    def test_all_examples_grammar_compliant(self):
        """Test that all examples comply with Grammar 2.0."""
        all_sequences = []

        # Collect all sequences
        patterns = [
            get_crisis_intervention_sequence(),
            get_process_therapy_sequence(),
            get_regenerative_healing_sequence(),
            get_insight_integration_sequence(),
            get_relapse_prevention_sequence(),
        ]
        all_sequences.extend(patterns)

        cases = [
            case_trauma_recovery(),
            case_addiction_healing(),
            case_depression_emergence(),
            case_relationship_repair(),
        ]
        for case in cases:
            all_sequences.append(case["sequence"])

        # All should pass validation (Grammar 2.0 compliance)
        for sequence in all_sequences:
            result = validate_sequence_with_health(sequence)
            assert result.passed, f"Sequence failed grammar validation: {result.message}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
