"""Tests for creative domain examples.

Validates that all creative patterns, case studies, and optimization
examples meet acceptance criteria according to TNFR Grammar 2.0.
"""

import pytest
from tnfr.operators.grammar import validate_sequence_with_health
from tnfr.operators.health_analyzer import SequenceHealthAnalyzer

# Import creative examples
import sys
from pathlib import Path

examples_path = Path(__file__).parent.parent.parent / "examples" / "domain_applications"
sys.path.insert(0, str(examples_path))

from creative_patterns import (
    get_artistic_creation_sequence,
    get_design_thinking_sequence,
    get_innovation_cycle_sequence,
    get_creative_flow_sequence,
    get_creative_block_resolution_sequence,
)

from creative_case_studies import (
    case_music_composition,
    case_visual_art,
    case_writing_process,
    case_product_design,
    case_software_development,
    case_choreography_creation,
)


# =============================================================================
# Test Creative Patterns
# =============================================================================


class TestCreativePatterns:
    """Test suite for creative_patterns.py."""
    
    def test_all_patterns_valid(self):
        """Test that all creative patterns pass validation."""
        patterns = {
            "artistic_creation": get_artistic_creation_sequence(),
            "design_thinking": get_design_thinking_sequence(),
            "innovation_cycle": get_innovation_cycle_sequence(),
            "creative_flow": get_creative_flow_sequence(),
            "creative_block_resolution": get_creative_block_resolution_sequence(),
        }
        
        for name, sequence in patterns.items():
            result = validate_sequence_with_health(sequence)
            assert result.passed, f"{name} failed validation: {result.message}"
    
    def test_all_health_scores_above_threshold(self):
        """Test that all patterns have health scores > 0.75."""
        patterns = {
            "artistic_creation": get_artistic_creation_sequence(),
            "design_thinking": get_design_thinking_sequence(),
            "innovation_cycle": get_innovation_cycle_sequence(),
            "creative_flow": get_creative_flow_sequence(),
            "creative_block_resolution": get_creative_block_resolution_sequence(),
        }
        
        threshold = 0.75
        
        for name, sequence in patterns.items():
            result = validate_sequence_with_health(sequence)
            assert result.passed, f"{name} did not pass validation"
            
            health = result.health_metrics.overall_health
            assert health >= threshold, (
                f"{name} health score {health:.3f} below threshold {threshold}"
            )
    
    def test_artistic_creation_characteristics(self):
        """Test artistic creation pattern has appropriate characteristics."""
        sequence = get_artistic_creation_sequence()
        result = validate_sequence_with_health(sequence)
        
        assert result.passed
        health = result.health_metrics
        
        # Should have good health for complete creative process
        assert health.overall_health >= 0.75
        
        # Should include mutation (creative breakthrough)
        assert "mutation" in sequence
        
        # Should include silence (contemplative space)
        assert "silence" in sequence
    
    def test_design_thinking_user_centered(self):
        """Test design thinking has user-centered elements."""
        sequence = get_design_thinking_sequence()
        result = validate_sequence_with_health(sequence)
        
        assert result.passed
        health = result.health_metrics
        
        # Should have excellent health (>0.80)
        assert health.overall_health >= 0.80
        
        # Should include key design thinking operators
        assert "reception" in sequence  # Empathize
        assert "coupling" in sequence   # Test with users
        assert "transition" in sequence # Implement
        assert "silence" in sequence    # Reflect
    
    def test_innovation_cycle_regenerative(self):
        """Test innovation cycle is detected as regenerative pattern."""
        sequence = get_innovation_cycle_sequence()
        result = validate_sequence_with_health(sequence)
        
        assert result.passed
        health = result.health_metrics
        
        # Should have high sustainability
        assert health.sustainability_index >= 0.5
        
        # Should be regenerative in nature
        assert health.dominant_pattern in ["regenerative", "cyclic", "activation"]
        
        # Should have transition for cycling
        assert "transition" in sequence
        
        # Should have good overall health
        assert health.overall_health >= 0.75
    
    def test_patterns_structural_coherence(self):
        """Test all patterns maintain structural coherence."""
        patterns = [
            get_artistic_creation_sequence(),
            get_design_thinking_sequence(),
            get_innovation_cycle_sequence(),
            get_creative_flow_sequence(),
            get_creative_block_resolution_sequence(),
        ]
        
        for sequence in patterns:
            result = validate_sequence_with_health(sequence)
            assert result.passed
            
            # All should have high coherence index
            assert result.health_metrics.coherence_index >= 0.85
    
    def test_patterns_minimum_count(self):
        """Test that at least 3 creative patterns are provided."""
        patterns = [
            get_artistic_creation_sequence(),
            get_design_thinking_sequence(),
            get_innovation_cycle_sequence(),
            get_creative_flow_sequence(),
            get_creative_block_resolution_sequence(),
        ]
        
        assert len(patterns) >= 3, "Should have at least 3 creative patterns"
    
    def test_silence_usage(self):
        """Test that SHA (silence) is used for contemplative space."""
        patterns_with_silence = [
            get_artistic_creation_sequence(),
            get_design_thinking_sequence(),
            get_creative_block_resolution_sequence(),
        ]
        
        # At least 3 patterns should emphasize silence
        assert len(patterns_with_silence) >= 3
        
        for sequence in patterns_with_silence:
            assert "silence" in sequence, "Pattern should include silence for contemplation"
    
    def test_mutation_usage(self):
        """Test that ZHIR (mutation) is used for breakthrough moments."""
        patterns = [
            get_artistic_creation_sequence(),
            get_design_thinking_sequence(),
            get_innovation_cycle_sequence(),
            get_creative_flow_sequence(),
            get_creative_block_resolution_sequence(),
        ]
        
        # All patterns should model breakthrough moments
        mutation_count = sum(1 for seq in patterns if "mutation" in seq)
        assert mutation_count >= 4, "Most patterns should include mutation for breakthroughs"


# =============================================================================
# Test Creative Case Studies
# =============================================================================


class TestCreativeCaseStudies:
    """Test suite for creative_case_studies.py."""
    
    def test_all_case_studies_valid(self):
        """Test that all case study sequences pass validation."""
        cases = [
            case_music_composition(),
            case_visual_art(),
            case_writing_process(),
            case_product_design(),
            case_software_development(),
            case_choreography_creation(),
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
            case_music_composition(),
            case_visual_art(),
            case_writing_process(),
            case_product_design(),
            case_software_development(),
            case_choreography_creation(),
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
    
    def test_music_composition_structure(self):
        """Test music composition case has appropriate structural elements."""
        case_data = case_music_composition()
        sequence = case_data["sequence"]
        
        # Should contain key operators for musical creativity
        assert "dissonance" in sequence, "Music requires dissonance (harmonic tension)"
        assert "mutation" in sequence, "Requires mutation for thematic development"
        assert "resonance" in sequence, "Requires resonance for motif repetition"
    
    def test_product_design_user_centered(self):
        """Test product design follows design thinking pattern."""
        case_data = case_product_design()
        sequence = case_data["sequence"]
        
        # Should have comprehensive sequence (10+ operators)
        assert len(sequence) >= 10, "Product design requires extended sequence"
        
        # Should include key operators
        assert "reception" in sequence, "Needs reception for user research"
        assert "coupling" in sequence, "Needs coupling for user testing"
    
    def test_software_development_innovation(self):
        """Test software development includes innovation elements."""
        case_data = case_software_development()
        sequence = case_data["sequence"]
        
        # Should include elements for technical creativity
        assert "expansion" in sequence, "Needs expansion for exploring approaches"
        assert "mutation" in sequence, "Needs mutation for innovative solutions"
        assert "transition" in sequence, "Needs transition for deployment"
    
    def test_case_studies_minimum_count(self):
        """Test that at least 5 case studies are provided."""
        cases = [
            case_music_composition(),
            case_visual_art(),
            case_writing_process(),
            case_product_design(),
            case_software_development(),
            case_choreography_creation(),
        ]
        
        assert len(cases) >= 5, "Should have at least 5 case studies"
    
    def test_case_studies_have_required_metadata(self):
        """Test that all case studies have required metadata fields."""
        cases = [
            case_music_composition(),
            case_visual_art(),
            case_writing_process(),
            case_product_design(),
            case_software_development(),
            case_choreography_creation(),
        ]
        
        required_fields = [
            "name",
            "sequence",
            "domain",
            "creative_challenge",
            "key_operators",
            "pattern_type",
        ]
        
        for case_data in cases:
            for field in required_fields:
                assert field in case_data, (
                    f"{case_data.get('name', 'Unknown')} missing field: {field}"
                )
    
    def test_diverse_domains(self):
        """Test that case studies cover diverse creative domains."""
        cases = [
            case_music_composition(),
            case_visual_art(),
            case_writing_process(),
            case_product_design(),
            case_software_development(),
            case_choreography_creation(),
        ]
        
        domains = {case["domain"] for case in cases}
        
        # Should cover at least 5 different domains
        assert len(domains) >= 5, f"Should cover diverse domains, got: {domains}"


# =============================================================================
# Test Pattern Characteristics
# =============================================================================


class TestPatternCharacteristics:
    """Test specific characteristics of creative patterns."""
    
    def test_breakthrough_patterns_have_mutation(self):
        """Test that breakthrough patterns include mutation operator."""
        artistic_seq = get_artistic_creation_sequence()
        innovation_seq = get_innovation_cycle_sequence()
        
        # Breakthrough patterns should include mutation
        assert "mutation" in artistic_seq, "Artistic creation should include mutation"
        assert "mutation" in innovation_seq, "Innovation should include mutation"
    
    def test_comprehensive_patterns_are_longer(self):
        """Test that comprehensive creative patterns are more extensive."""
        artistic_seq = get_artistic_creation_sequence()
        design_seq = get_design_thinking_sequence()
        
        # Comprehensive patterns should be extensive (10+ operators)
        assert len(artistic_seq) >= 10, "Artistic creation should be extensive"
        assert len(design_seq) >= 10, "Design thinking should be extensive"
    
    def test_patterns_end_with_valid_operators(self):
        """Test that all patterns end with valid end operators."""
        from tnfr.config.operator_names import VALID_END_OPERATORS
        
        patterns = [
            get_artistic_creation_sequence(),
            get_design_thinking_sequence(),
            get_innovation_cycle_sequence(),
            get_creative_flow_sequence(),
            get_creative_block_resolution_sequence(),
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
            get_artistic_creation_sequence(),
            get_design_thinking_sequence(),
            get_innovation_cycle_sequence(),
            get_creative_flow_sequence(),
            get_creative_block_resolution_sequence(),
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
    """Integration tests for creative examples."""
    
    def test_patterns_and_cases_consistency(self):
        """Test that patterns and case studies are consistent."""
        # Both should use the same operator names
        from tnfr.config.operator_names import CANONICAL_OPERATOR_NAMES
        
        all_sequences = []
        
        # Collect all pattern sequences
        patterns = [
            get_artistic_creation_sequence(),
            get_design_thinking_sequence(),
            get_innovation_cycle_sequence(),
            get_creative_flow_sequence(),
            get_creative_block_resolution_sequence(),
        ]
        all_sequences.extend(patterns)
        
        # Collect all case study sequences
        cases = [
            case_music_composition(),
            case_visual_art(),
            case_writing_process(),
            case_product_design(),
            case_software_development(),
            case_choreography_creation(),
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
            get_artistic_creation_sequence(),
            get_design_thinking_sequence(),
            get_innovation_cycle_sequence(),
            get_creative_flow_sequence(),
            get_creative_block_resolution_sequence(),
        ]
        all_sequences.extend(patterns)
        
        # Collect case studies
        cases = [
            case_music_composition(),
            case_visual_art(),
            case_writing_process(),
            case_product_design(),
            case_software_development(),
            case_choreography_creation(),
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
            get_artistic_creation_sequence(),
            get_design_thinking_sequence(),
            get_innovation_cycle_sequence(),
            get_creative_flow_sequence(),
            get_creative_block_resolution_sequence(),
        ]
        all_sequences.extend(patterns)
        
        cases = [
            case_music_composition(),
            case_visual_art(),
            case_writing_process(),
            case_product_design(),
            case_software_development(),
            case_choreography_creation(),
        ]
        for case in cases:
            all_sequences.append(case["sequence"])
        
        # All should pass validation (Grammar 2.0 compliance)
        for sequence in all_sequences:
            result = validate_sequence_with_health(sequence)
            assert result.passed, f"Sequence failed grammar validation: {result.message}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
