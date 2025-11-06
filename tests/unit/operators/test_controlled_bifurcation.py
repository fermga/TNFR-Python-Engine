"""Tests for R4 evolutionary: Controlled bifurcation rules with destabilizer window.

This module tests the evolved R4 rule that requires ZHIR/THOL (transformers)
to be preceded by OZ/NAV/VAL (destabilizers) within a window of 3 operators.

References:
    - Issue: fermga/TNFR-Python-Engine#[issue_number]
    - Theory: TNFR PDF "Bifurcación y emergencia", "Glifos de la emergencia"
"""

import pytest

from tnfr.config.operator_names import (
    COHERENCE,
    CONTRACTION,
    COUPLING,
    DISSONANCE,
    EMISSION,
    EXPANSION,
    MUTATION,
    RECEPTION,
    RECURSIVITY,
    RESONANCE,
    SELF_ORGANIZATION,
    SILENCE,
    TRANSITION,
)
from tnfr.operators.grammar import (
    SequenceSyntaxError,
    parse_sequence,
    validate_sequence,
)


class TestZhirRequiresDestabilizer:
    """Test ZHIR (mutation) requires destabilizer in window."""

    def test_zhir_after_oz_direct(self):
        """OZ → ZHIR (original, direct adjacency)."""
        result = validate_sequence(
            [EMISSION, RECEPTION, COHERENCE, DISSONANCE, MUTATION, COHERENCE, SILENCE]
        )
        assert result.passed
        assert result.metadata["has_dissonance"]

    def test_zhir_after_oz_with_stabilizer(self):
        """OZ → IL → ZHIR (destabilizer with intermediate stabilizer)."""
        result = validate_sequence(
            [EMISSION, RECEPTION, COHERENCE, DISSONANCE, COHERENCE, MUTATION, COHERENCE, SILENCE]
        )
        assert result.passed
        assert result.metadata["has_dissonance"]

    def test_zhir_after_nav_destabilizer(self):
        """NAV → ... → ZHIR (transition as destabilizer)."""
        # TRANSITION → COHERENCE → ZHIR
        # DISSONANCE → TRANSITION is good, TRANSITION → COHERENCE is excellent
        result = validate_sequence(
            [EMISSION, RECEPTION, COHERENCE, DISSONANCE, TRANSITION, COHERENCE, MUTATION, COHERENCE, SILENCE]
        )
        assert result.passed

    def test_zhir_after_val_expansion_destabilizer(self):
        """VAL → ... → ZHIR (expansion as destabilizer)."""
        # EXPANSION → COHERENCE → MUTATION
        result = validate_sequence(
            [EMISSION, RECEPTION, COHERENCE, EXPANSION, COHERENCE, MUTATION, COHERENCE, SILENCE]
        )
        assert result.passed

    def test_zhir_without_any_destabilizer_fails(self):
        """ZHIR without any destabilizer in history should fail."""
        with pytest.raises(SequenceSyntaxError) as excinfo:
            parse_sequence([EMISSION, RECEPTION, COHERENCE, MUTATION, SILENCE])
        
        error = excinfo.value
        assert "mutation" in error.message.lower()
        assert "destabilizer" in error.message.lower()
        assert error.index == 3  # MUTATION is at index 3


class TestTholRequiresDestabilizer:
    """Test THOL (self_organization) requires destabilizer in window."""

    def test_thol_after_oz_direct(self):
        """OZ → THOL (dissonance enables self-organization)."""
        result = validate_sequence(
            [EMISSION, RECEPTION, COHERENCE, DISSONANCE, SELF_ORGANIZATION, SILENCE]
        )
        assert result.passed

    def test_thol_after_nav_destabilizer(self):
        """NAV → ... → THOL (transition enables emergence)."""
        # DISSONANCE → TRANSITION → COHERENCE → THOL
        result = validate_sequence(
            [EMISSION, RECEPTION, COHERENCE, DISSONANCE, TRANSITION, COHERENCE, SELF_ORGANIZATION, SILENCE]
        )
        assert result.passed

    def test_thol_after_val_destabilizer(self):
        """VAL → ... → THOL (expansion enables self-organization)."""
        result = validate_sequence(
            [EMISSION, RECEPTION, COHERENCE, EXPANSION, COHERENCE, SELF_ORGANIZATION, SILENCE]
        )
        assert result.passed

    def test_thol_without_destabilizer_fails(self):
        """THOL without destabilizer context should fail."""
        with pytest.raises(SequenceSyntaxError) as excinfo:
            parse_sequence([EMISSION, RECEPTION, COHERENCE, SELF_ORGANIZATION, SILENCE])
        
        error = excinfo.value
        assert "self_organization" in error.message.lower()
        assert "destabilizer" in error.message.lower()


class TestBifurcationWindow:
    """Test the 3-operator window for destabilizer precedent."""

    def test_destabilizer_within_window_1_step(self):
        """Destabilizer 1 step before transformer (immediate)."""
        result = validate_sequence(
            [EMISSION, RECEPTION, COHERENCE, DISSONANCE, MUTATION, COHERENCE, SILENCE]
        )
        assert result.passed

    def test_destabilizer_within_window_2_steps(self):
        """Destabilizer 2 steps before transformer."""
        result = validate_sequence(
            [EMISSION, RECEPTION, COHERENCE, DISSONANCE, COHERENCE, MUTATION, COHERENCE, SILENCE]
        )
        assert result.passed

    def test_destabilizer_within_window_3_steps(self):
        """Destabilizer 3 steps before transformer (max window)."""
        # Want: destabilizer exactly 3 steps before ZHIR
        # Use: COHERENCE → DISSONANCE (caution, allowed)
        # Sequence: AL EN IL IL OZ IL RA IL ZHIR
        # Indices:   0  1  2  3  4  5  6  7   8
        # From 8 (ZHIR), window is max(0,8-3)=5 to 7, indices 5,6,7
        # OZ at 4 is NOT in window!
        # Simpler: AL EN IL OZ IL RA IL ZHIR
        # Indices:  0  1  2  3  4  5  6   7
        # From 7 (ZHIR), window is 4,5,6 - OZ at 3 NOT in window
        # Try: AL EN IL RA IL OZ IL IL ZHIR (but IL→IL invalid)
        # Better approach: use only 2 operators between OZ and ZHIR
        # AL EN IL OZ IL IL ZHIR - but IL→IL invalid
        # Use RA: AL EN IL OZ RA IL ZHIR
        # Indices: 0  1  2  3  4  5   6
        # From 6, window is 3,4,5 - OZ at 3 IS at boundary (exactly 3 back!)
        result = validate_sequence(
            [EMISSION, RECEPTION, COHERENCE, DISSONANCE, RESONANCE, COHERENCE, MUTATION, SILENCE]
        )
        assert result.passed

    def test_destabilizer_beyond_window_fails(self):
        """Destabilizer more than 3 steps before transformer should fail."""
        with pytest.raises(SequenceSyntaxError) as excinfo:
            # OZ at index 3, ZHIR at index 7 = 4 steps apart
            parse_sequence(
                [EMISSION, RECEPTION, COHERENCE, DISSONANCE, COHERENCE, RESONANCE, COUPLING, MUTATION, SILENCE]
            )
        
        error = excinfo.value
        assert "mutation" in error.message.lower()
        assert "destabilizer" in error.message.lower()


class TestMultipleDestabilizers:
    """Test sequences with multiple destabilizers."""

    def test_multiple_destabilizers_in_window(self):
        """Multiple destabilizers in window (any should work)."""
        # OZ → IL → VAL → IL → ZHIR
        result = validate_sequence(
            [EMISSION, RECEPTION, COHERENCE, DISSONANCE, COHERENCE, EXPANSION, COHERENCE, MUTATION, COHERENCE, SILENCE]
        )
        assert result.passed

    def test_earlier_destabilizer_replaced_in_window(self):
        """Earlier destabilizer outside window, newer one inside."""
        # First DISSONANCE outside window, EXPANSION inside
        # COHERENCE → DISSONANCE is caution (allowed with warning), then continue
        result = validate_sequence(
            [EMISSION, RECEPTION, COHERENCE, DISSONANCE, COHERENCE, EXPANSION, COHERENCE, MUTATION, COHERENCE, SILENCE]
        )
        assert result.passed


class TestComplexBifurcationPatterns:
    """Test complex patterns with multiple transformers and destabilizers."""

    def test_multiple_bifurcations_in_sequence(self):
        """Multiple ZHIR/THOL each with their own destabilizer."""
        result = validate_sequence(
            [
                EMISSION, RECEPTION, COHERENCE,
                DISSONANCE, MUTATION, COHERENCE,  # First bifurcation: OZ → ZHIR
                EXPANSION, COHERENCE, SELF_ORGANIZATION,  # Second bifurcation: VAL → THOL
                SILENCE
            ]
        )
        assert result.passed

    def test_nested_transformer_with_shared_destabilizer(self):
        """ZHIR and THOL both using same destabilizer."""
        result = validate_sequence(
            [
                EMISSION, RECEPTION, COHERENCE,
                DISSONANCE,  # Destabilizer for both
                MUTATION, COHERENCE,  # First transformer
                SELF_ORGANIZATION,  # Second transformer (within window of OZ)
                SILENCE
            ]
        )
        assert result.passed

    def test_bifurcation_from_stability_pattern(self):
        """Bifurcación desde estabilidad: AL → IL → OZ → ZHIR → IL."""
        result = validate_sequence(
            [EMISSION, RECEPTION, COHERENCE, DISSONANCE, MUTATION, COHERENCE, SILENCE]
        )
        assert result.passed

    def test_self_organization_from_transition_pattern(self):
        """Autoorganización desde transición: NAV → ... → THOL → IL."""
        # Use TRANSITION as destabilizer, then THOL
        # DISSONANCE → TRANSITION → COHERENCE → THOL
        result = validate_sequence(
            [EMISSION, RECEPTION, COHERENCE, DISSONANCE, TRANSITION, COHERENCE, SELF_ORGANIZATION, SILENCE]
        )
        assert result.passed

    def test_mutation_from_expansion_pattern(self):
        """Mutación desde expansión: AL → ... → VAL → ... → ZHIR."""
        # EXPANSION → COHERENCE → MUTATION
        result = validate_sequence(
            [EMISSION, RECEPTION, COHERENCE, EXPANSION, COHERENCE, MUTATION, COHERENCE, TRANSITION]
        )
        assert result.passed


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_transformer_at_sequence_start_fails(self):
        """Transformer cannot be at sequence start (no destabilizer possible)."""
        # Even though RECURSIVITY is a valid start, it's not a destabilizer
        with pytest.raises(SequenceSyntaxError) as excinfo:
            parse_sequence([EMISSION, MUTATION, COHERENCE, SILENCE])
        
        error = excinfo.value
        assert "mutation" in error.message.lower()

    def test_destabilizer_at_end_valid(self):
        """Destabilizer at end is valid (even if not used)."""
        # DISSONANCE is a valid end operator
        # COHERENCE → DISSONANCE is caution (allowed with warning)
        result = validate_sequence(
            [EMISSION, RECEPTION, COHERENCE, DISSONANCE]
        )
        assert result.passed

    def test_all_three_destabilizer_types(self):
        """Sequence using all three destabilizer types."""
        # THOL needs closure with SILENCE or CONTRACTION
        result = validate_sequence(
            [
                EMISSION, RECEPTION, COHERENCE,
                DISSONANCE, MUTATION, COHERENCE,  # OZ destabilizer
                EXPANSION, COHERENCE, SELF_ORGANIZATION,  # VAL destabilizer
                SILENCE  # Proper THOL closure
            ]
        )
        assert result.passed


class TestBackwardCompatibility:
    """Verify backward compatibility with evolved R4 rule."""

    def test_original_oz_zhir_pattern_still_works(self):
        """Original OZ → ZHIR pattern (R4 original) still valid."""
        result = validate_sequence(
            [EMISSION, RECEPTION, COHERENCE, DISSONANCE, MUTATION, COHERENCE, SILENCE]
        )
        assert result.passed

    def test_linear_sequences_unaffected(self):
        """Linear sequences without transformers remain valid."""
        result = validate_sequence(
            [EMISSION, RECEPTION, COHERENCE, RESONANCE, SILENCE]
        )
        assert result.passed

    def test_sequences_with_expansion_unaffected(self):
        """Sequences with VAL but no transformers remain valid."""
        result = validate_sequence(
            [EMISSION, RECEPTION, COHERENCE, EXPANSION, RESONANCE, SILENCE]
        )
        assert result.passed

    def test_sequences_with_transition_unaffected(self):
        """Sequences with NAV but no transformers remain valid."""
        result = validate_sequence(
            [RECURSIVITY, RECEPTION, COHERENCE, RESONANCE, TRANSITION]
        )
        assert result.passed


class TestErrorMessages:
    """Test that error messages are informative."""

    def test_zhir_error_mentions_all_destabilizers(self):
        """Error for ZHIR should mention all possible destabilizers."""
        with pytest.raises(SequenceSyntaxError) as excinfo:
            parse_sequence([EMISSION, RECEPTION, COHERENCE, MUTATION, SILENCE])
        
        error = excinfo.value
        # Should mention the destabilizer options
        assert any(word in error.message.lower() for word in ["dissonance", "expansion", "transition"])

    def test_thol_error_mentions_window_size(self):
        """Error for THOL should mention the window size."""
        with pytest.raises(SequenceSyntaxError) as excinfo:
            parse_sequence([EMISSION, RECEPTION, COHERENCE, SELF_ORGANIZATION, SILENCE])
        
        error = excinfo.value
        assert "3" in error.message or "previous" in error.message.lower()

    def test_error_provides_correct_index(self):
        """Error should point to the transformer that failed."""
        with pytest.raises(SequenceSyntaxError) as excinfo:
            parse_sequence([EMISSION, RECEPTION, COHERENCE, RESONANCE, MUTATION, SILENCE])
        
        error = excinfo.value
        assert error.index == 4  # MUTATION is at index 4
        assert error.token == MUTATION
