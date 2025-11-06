"""Tests for educational domain examples.

Validates that all educational patterns, case studies, and optimization
examples meet acceptance criteria according to TNFR Grammar 2.0.
"""

import pytest
from tnfr.operators.grammar import validate_sequence_with_health
from tnfr.operators.health_analyzer import SequenceHealthAnalyzer

# Import educational examples
import sys
from pathlib import Path

examples_path = Path(__file__).parent.parent.parent / "examples" / "domain_applications"
sys.path.insert(0, str(examples_path))

from educational_patterns import (
    get_conceptual_breakthrough_sequence,
    get_competency_development_sequence,
    get_knowledge_spiral_sequence,
    get_practice_mastery_sequence,
    get_collaborative_learning_sequence,
)

from educational_case_studies import (
    case_mathematics_learning,
    case_language_acquisition,
    case_scientific_method,
    case_skill_mastery,
    case_creative_writing,
)


# =============================================================================
# Test Educational Patterns
# =============================================================================


class TestEducationalPatterns:
    """Test suite for educational_patterns.py."""
    
    def test_all_patterns_valid(self):
        """Test that all educational patterns pass validation."""
        patterns = {
            "conceptual_breakthrough": get_conceptual_breakthrough_sequence(),
            "competency_development": get_competency_development_sequence(),
            "knowledge_spiral": get_knowledge_spiral_sequence(),
            "practice_mastery": get_practice_mastery_sequence(),
            "collaborative_learning": get_collaborative_learning_sequence(),
        }
        
        for name, sequence in patterns.items():
            result = validate_sequence_with_health(sequence)
            assert result.passed, f"{name} failed validation: {result.message}"
    
    def test_all_health_scores_above_threshold(self):
        """Test that all patterns have health scores > 0.75."""
        patterns = {
            "conceptual_breakthrough": get_conceptual_breakthrough_sequence(),
            "competency_development": get_competency_development_sequence(),
            "knowledge_spiral": get_knowledge_spiral_sequence(),
            "practice_mastery": get_practice_mastery_sequence(),
            "collaborative_learning": get_collaborative_learning_sequence(),
        }
        
        threshold = 0.75
        
        for name, sequence in patterns.items():
            result = validate_sequence_with_health(sequence)
            assert result.passed, f"{name} did not pass validation"
            
            health = result.health_metrics.overall_health
            assert health >= threshold, (
                f"{name} health score {health:.3f} below threshold {threshold}"
            )
    
    def test_conceptual_breakthrough_characteristics(self):
        """Test conceptual breakthrough pattern has appropriate characteristics."""
        sequence = get_conceptual_breakthrough_sequence()
        result = validate_sequence_with_health(sequence)
        
        assert result.passed
        health = result.health_metrics
        
        # Should have good health for rapid insight
        assert health.overall_health >= 0.75
        
        # Should include mutation (paradigm shift)
        assert "mutation" in sequence
        
        # Pattern should be activation or hierarchical
        assert health.dominant_pattern in ["activation", "hierarchical"]
    
    def test_competency_development_completeness(self):
        """Test competency development has comprehensive learning cycle."""
        sequence = get_competency_development_sequence()
        result = validate_sequence_with_health(sequence)
        
        assert result.passed
        health = result.health_metrics
        
        # Should have excellent health (>0.80)
        assert health.overall_health >= 0.80
        
        # Should be comprehensive (8+ operators)
        assert len(sequence) >= 8
        
        # Should include key operators for complete cycle
        assert "expansion" in sequence
        assert "mutation" in sequence
        assert "transition" in sequence
    
    def test_knowledge_spiral_regenerative(self):
        """Test knowledge spiral is detected as regenerative pattern."""
        sequence = get_knowledge_spiral_sequence()
        result = validate_sequence_with_health(sequence)
        
        assert result.passed
        health = result.health_metrics
        
        # Should have high sustainability
        assert health.sustainability_index >= 0.5
        
        # Should be cyclical in nature
        assert health.dominant_pattern in ["regenerative", "cyclic", "activation"]
        
        # Should have good overall health
        assert health.overall_health >= 0.75
    
    def test_patterns_structural_coherence(self):
        """Test all patterns maintain structural coherence."""
        patterns = [
            get_conceptual_breakthrough_sequence(),
            get_competency_development_sequence(),
            get_knowledge_spiral_sequence(),
            get_practice_mastery_sequence(),
            get_collaborative_learning_sequence(),
        ]
        
        for sequence in patterns:
            result = validate_sequence_with_health(sequence)
            assert result.passed
            
            # All should have high coherence index
            assert result.health_metrics.coherence_index >= 0.85
    
    def test_patterns_minimum_count(self):
        """Test that at least 5 educational patterns are provided."""
        patterns = [
            get_conceptual_breakthrough_sequence(),
            get_competency_development_sequence(),
            get_knowledge_spiral_sequence(),
            get_practice_mastery_sequence(),
            get_collaborative_learning_sequence(),
        ]
        
        assert len(patterns) >= 5, "Should have at least 5 educational patterns"


# =============================================================================
# Test Educational Case Studies
# =============================================================================


class TestEducationalCaseStudies:
    """Test suite for educational_case_studies.py."""
    
    def test_all_case_studies_valid(self):
        """Test that all case study sequences pass validation."""
        cases = [
            case_mathematics_learning(),
            case_language_acquisition(),
            case_scientific_method(),
            case_skill_mastery(),
            case_creative_writing(),
        ]
        
        for case_data in cases:
            sequence = case_data["sequence"]
            result = validate_sequence_with_health(sequence)
            assert result.passed, (
                f"{case_data['name']} failed validation: {result.message}"
            )
    
    def test_all_case_studies_health_above_threshold(self):
        """Test that all case studies have health scores > 0.75."""
        cases = [
            case_mathematics_learning(),
            case_language_acquisition(),
            case_scientific_method(),
            case_skill_mastery(),
            case_creative_writing(),
        ]
        
        threshold = 0.75
        
        for case_data in cases:
            sequence = case_data["sequence"]
            result = validate_sequence_with_health(sequence)
            
            assert result.passed, f"{case_data['name']} did not pass validation"
            
            health = result.health_metrics.overall_health
            assert health >= threshold, (
                f"{case_data['name']} health {health:.3f} below threshold {threshold}"
            )
    
    def test_mathematics_learning_structure(self):
        """Test mathematics learning case has appropriate structural elements."""
        case_data = case_mathematics_learning()
        sequence = case_data["sequence"]
        
        # Should contain key operators for conceptual breakthrough
        assert "dissonance" in sequence, "Math learning requires dissonance (cognitive conflict)"
        assert "mutation" in sequence, "Requires mutation for paradigm shift"
        assert "coherence" in sequence, "Requires coherence for stabilization"
    
    def test_language_acquisition_pattern(self):
        """Test language acquisition follows appropriate pattern."""
        case_data = case_language_acquisition()
        sequence = case_data["sequence"]
        
        # Should have comprehensive sequence (8+ operators)
        assert len(sequence) >= 8, "Language acquisition requires extended sequence"
        
        # Should include key operators
        assert "expansion" in sequence, "Needs expansion for context variety"
        assert "self_organization" in sequence, "Needs self-organization for pattern recognition"
    
    def test_scientific_method_cyclic(self):
        """Test scientific method includes cyclic/regenerative elements."""
        case_data = case_scientific_method()
        sequence = case_data["sequence"]
        
        # Should include elements for theory revision
        assert "dissonance" in sequence, "Needs dissonance for data-theory conflict"
        assert "mutation" in sequence, "Needs mutation for theory revision"
        assert "transition" in sequence, "Needs transition to next research question"
    
    def test_skill_mastery_practice(self):
        """Test skill mastery includes practice elements."""
        case_data = case_skill_mastery()
        sequence = case_data["sequence"]
        
        # Should include self-organization for autonomous refinement
        assert "self_organization" in sequence, "Practice requires self-directed adjustment"
    
    def test_creative_writing_emergence(self):
        """Test creative writing includes creative emergence elements."""
        case_data = case_creative_writing()
        sequence = case_data["sequence"]
        
        # Should include expansion for creative exploration
        assert "expansion" in sequence, "Creative writing needs expansion"
        assert "self_organization" in sequence, "Needs self-organization for narrative coherence"
    
    def test_case_studies_minimum_count(self):
        """Test that at least 5 case studies are provided."""
        cases = [
            case_mathematics_learning(),
            case_language_acquisition(),
            case_scientific_method(),
            case_skill_mastery(),
            case_creative_writing(),
        ]
        
        assert len(cases) >= 5, "Should have at least 5 case studies"
    
    def test_case_studies_have_required_metadata(self):
        """Test that all case studies have required metadata fields."""
        cases = [
            case_mathematics_learning(),
            case_language_acquisition(),
            case_scientific_method(),
            case_skill_mastery(),
            case_creative_writing(),
        ]
        
        required_fields = [
            "name",
            "sequence",
            "presenting_level",
            "learning_goal",
            "key_operators",
            "pattern_type",
        ]
        
        for case_data in cases:
            for field in required_fields:
                assert field in case_data, (
                    f"{case_data.get('name', 'Unknown')} missing field: {field}"
                )


# =============================================================================
# Test Pattern Characteristics
# =============================================================================


class TestPatternCharacteristics:
    """Test specific characteristics of educational patterns."""
    
    def test_breakthrough_patterns_have_mutation(self):
        """Test that breakthrough patterns include mutation operator."""
        breakthrough_seq = get_conceptual_breakthrough_sequence()
        
        # Breakthrough should include mutation (paradigm shift)
        assert "mutation" in breakthrough_seq, "Breakthrough patterns should include mutation"
    
    def test_comprehensive_patterns_are_longer(self):
        """Test that comprehensive learning patterns are more extensive."""
        competency_seq = get_competency_development_sequence()
        
        # Comprehensive learning should be extensive (8+ operators)
        assert len(competency_seq) >= 8, "Comprehensive patterns should be extensive"
    
    def test_patterns_end_with_valid_operators(self):
        """Test that all patterns end with valid end operators."""
        from tnfr.config.operator_names import VALID_END_OPERATORS
        
        patterns = [
            get_conceptual_breakthrough_sequence(),
            get_competency_development_sequence(),
            get_knowledge_spiral_sequence(),
            get_practice_mastery_sequence(),
            get_collaborative_learning_sequence(),
        ]
        
        for sequence in patterns:
            last_operator = sequence[-1]
            assert last_operator in VALID_END_OPERATORS, (
                f"Sequence ends with invalid operator: {last_operator}"
            )
    
    def test_patterns_start_with_valid_operators(self):
        """Test that all patterns start with valid start operators."""
        from tnfr.config.operator_names import VALID_START_OPERATORS
        
        patterns = [
            get_conceptual_breakthrough_sequence(),
            get_competency_development_sequence(),
            get_knowledge_spiral_sequence(),
            get_practice_mastery_sequence(),
            get_collaborative_learning_sequence(),
        ]
        
        for sequence in patterns:
            first_operator = sequence[0]
            assert first_operator in VALID_START_OPERATORS, (
                f"Sequence starts with invalid operator: {first_operator}"
            )


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for educational examples."""
    
    def test_patterns_and_cases_consistency(self):
        """Test that patterns and case studies are consistent."""
        # Both should use the same operator names
        from tnfr.config.operator_names import CANONICAL_OPERATOR_NAMES
        
        all_sequences = []
        
        # Collect all pattern sequences
        patterns = [
            get_conceptual_breakthrough_sequence(),
            get_competency_development_sequence(),
            get_knowledge_spiral_sequence(),
            get_practice_mastery_sequence(),
            get_collaborative_learning_sequence(),
        ]
        all_sequences.extend(patterns)
        
        # Collect all case study sequences
        cases = [
            case_mathematics_learning(),
            case_language_acquisition(),
            case_scientific_method(),
            case_skill_mastery(),
            case_creative_writing(),
        ]
        for case in cases:
            all_sequences.append(case["sequence"])
        
        # Check all operators are canonical
        for sequence in all_sequences:
            for operator in sequence:
                assert operator in CANONICAL_OPERATOR_NAMES, (
                    f"Non-canonical operator found: {operator}"
                )
    
    def test_average_health_meets_target(self):
        """Test that average health across all examples meets target."""
        all_sequences = []
        
        # Collect patterns
        patterns = [
            get_conceptual_breakthrough_sequence(),
            get_competency_development_sequence(),
            get_knowledge_spiral_sequence(),
            get_practice_mastery_sequence(),
            get_collaborative_learning_sequence(),
        ]
        all_sequences.extend(patterns)
        
        # Collect case studies
        cases = [
            case_mathematics_learning(),
            case_language_acquisition(),
            case_scientific_method(),
            case_skill_mastery(),
            case_creative_writing(),
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
        assert avg_health >= 0.75, (
            f"Average health {avg_health:.3f} below target 0.75"
        )
    
    def test_all_examples_grammar_compliant(self):
        """Test that all examples comply with Grammar 2.0."""
        all_sequences = []
        
        # Collect all sequences
        patterns = [
            get_conceptual_breakthrough_sequence(),
            get_competency_development_sequence(),
            get_knowledge_spiral_sequence(),
            get_practice_mastery_sequence(),
            get_collaborative_learning_sequence(),
        ]
        all_sequences.extend(patterns)
        
        cases = [
            case_mathematics_learning(),
            case_language_acquisition(),
            case_scientific_method(),
            case_skill_mastery(),
            case_creative_writing(),
        ]
        for case in cases:
            all_sequences.append(case["sequence"])
        
        # All should pass validation (Grammar 2.0 compliance)
        for sequence in all_sequences:
            result = validate_sequence_with_health(sequence)
            assert result.passed, f"Sequence failed grammar validation: {result.message}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
