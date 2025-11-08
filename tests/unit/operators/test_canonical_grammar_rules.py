"""Tests for canonical TNFR grammar rules (R1-R5) and structural patterns."""

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
    StructuralPattern,
    parse_sequence,
    validate_sequence,
)


class TestR1StartOperators:
    """Test R1: Toda secuencia debe comenzar con AL o NAV (glifo generador)."""

    def test_valid_start_with_emission(self):
        """AL (emission) is a valid start operator."""
        result = validate_sequence([EMISSION, RECEPTION, COHERENCE, RESONANCE, SILENCE])
        assert result.passed

    def test_valid_start_with_recursivity(self):
        """REMESH (recursivity) is a valid start operator."""
        result = validate_sequence(
            [RECURSIVITY, RECEPTION, COHERENCE, RESONANCE, SILENCE]
        )
        assert result.passed

    def test_valid_start_with_transition(self):
        """NAV (transition) is a valid start operator (physics-derived)."""
        result = validate_sequence(
            [TRANSITION, RECEPTION, COHERENCE, RESONANCE, SILENCE]
        )
        assert result.passed

    def test_invalid_start_with_reception(self):
        """Starting with EN (reception) is invalid."""
        result = validate_sequence([RECEPTION, COHERENCE, RESONANCE, SILENCE])
        assert not result.passed
        assert "must start" in result.message.lower()

    def test_invalid_start_with_coherence(self):
        """Starting with IL (coherence) is invalid."""
        result = validate_sequence([COHERENCE, RESONANCE, SILENCE])
        assert not result.passed
        assert "must start" in result.message.lower()


class TestR2RequiredStabilizer:
    """Test R2: Debe contener al menos un estabilizador (IL o THOL)."""

    def test_valid_with_coherence_stabilizer(self):
        """Sequence with IL (coherence) as stabilizer is valid."""
        result = validate_sequence([EMISSION, RECEPTION, COHERENCE, RESONANCE, SILENCE])
        assert result.passed
        assert result.metadata["has_stabilizer"]

    def test_valid_with_self_organization_stabilizer(self):
        """Sequence with THOL (self_organization) as stabilizer is valid."""
        # Updated: THOL now requires destabilizer (R4 evolved rule)
        result = validate_sequence(
            [EMISSION, RECEPTION, COHERENCE, DISSONANCE, SELF_ORGANIZATION, SILENCE]
        )
        assert result.passed
        assert result.metadata["has_stabilizer"]

    def test_invalid_without_stabilizer(self):
        """Sequence without IL or THOL should fail."""
        # This would need a sequence that somehow passes other checks but lacks stabilizer
        # In practice, the RECEPTION→COHERENCE requirement ensures stabilizer exists
        # But we test the explicit check
        result = validate_sequence([EMISSION, RECEPTION, RESONANCE, SILENCE])
        assert not result.passed
        # Either missing stabilizer or missing coherence segment
        assert "missing" in result.message.lower()


class TestR3FinalizationOperators:
    """Test R3: Debe finalizar con NUL o SHA para evitar colapso."""

    def test_valid_end_with_silence(self):
        """Ending with SHA (silence) is valid."""
        result = validate_sequence([EMISSION, RECEPTION, COHERENCE, RESONANCE, SILENCE])
        assert result.passed

    def test_valid_end_with_transition(self):
        """Ending with NAV (transition) is valid."""
        result = validate_sequence(
            [EMISSION, RECEPTION, COHERENCE, RESONANCE, TRANSITION]
        )
        assert result.passed

    def test_valid_end_with_recursivity(self):
        """Ending with RECURSIVITY is valid (REMESH glyph, NAV/transition operator)."""
        result = validate_sequence(
            [EMISSION, RECEPTION, COHERENCE, RESONANCE, RECURSIVITY]
        )
        assert result.passed

    def test_invalid_end_with_emission(self):
        """Ending with AL (emission) is invalid."""
        result = validate_sequence(
            [EMISSION, RECEPTION, COHERENCE, RESONANCE, EMISSION]
        )
        assert not result.passed
        assert "must end" in result.message.lower()

    def test_invalid_end_with_coherence(self):
        """Ending with IL (coherence) is invalid."""
        result = validate_sequence(
            [EMISSION, RECEPTION, COHERENCE, RESONANCE, COHERENCE]
        )
        assert not result.passed
        assert "must end" in result.message.lower()


class TestR4MutationRequiresDissonance:
    """Test R4: ZHIR debe ir precedido por OZ (no muta sin disonancia)."""

    def test_valid_mutation_after_dissonance(self):
        """ZHIR (mutation) after OZ (dissonance) is valid."""
        result = validate_sequence(
            [EMISSION, RECEPTION, COHERENCE, DISSONANCE, MUTATION, COHERENCE, SILENCE]
        )
        assert result.passed
        assert result.metadata["has_dissonance"]

    def test_invalid_mutation_without_dissonance(self):
        """ZHIR (mutation) without OZ precedent should fail."""
        with pytest.raises(SequenceSyntaxError) as excinfo:
            parse_sequence([EMISSION, RECEPTION, COHERENCE, MUTATION, SILENCE])

        error = excinfo.value
        assert "mutation" in error.message.lower()
        assert "dissonance" in error.message.lower()
        assert error.index == 3  # MUTATION is at index 3


class TestR5CompatibilityRules:
    """Test R5 and sequential compatibility rules."""

    def test_compatible_emission_to_reception(self):
        """AL → EN is a compatible transition."""
        result = validate_sequence([EMISSION, RECEPTION, COHERENCE, RESONANCE, SILENCE])
        assert result.passed

    def test_compatible_coherence_to_resonance(self):
        """IL → RA is a compatible transition."""
        result = validate_sequence([EMISSION, RECEPTION, COHERENCE, RESONANCE, SILENCE])
        assert result.passed

    def test_incompatible_silence_to_dissonance(self):
        """SHA → OZ is incompatible (silence followed by dissonance)."""
        with pytest.raises(SequenceSyntaxError) as excinfo:
            parse_sequence(
                [EMISSION, RECEPTION, COHERENCE, SILENCE, DISSONANCE, MUTATION, SILENCE]
            )

        error = excinfo.value
        # Check that error message explains the physical incompatibility
        assert "invalid after silence" in error.message.lower() or "contradicts" in error.message.lower()

    def test_expansion_to_contraction_valid_with_medium_to_high(self):
        """VAL → NUL is valid (medium → high frequency transition)."""
        # EXPANSION has medium freq, CONTRACTION has high freq
        # medium → high is a valid transition in TNFR physics
        result = validate_sequence(
            [EMISSION, RECEPTION, COHERENCE, EXPANSION, CONTRACTION, SILENCE]
        )
        assert result.passed, f"Expected valid but got: {result.message}"


class TestValidCanonicalSequences:
    """Test valid canonical TNFR sequences from the specification."""

    def test_linear_basic_sequence(self):
        """Test: AL → IL → RA → SHA (linear básica)."""
        result = validate_sequence([EMISSION, RECEPTION, COHERENCE, RESONANCE, SILENCE])
        assert result.passed
        assert result.metadata["detected_pattern"] == StructuralPattern.LINEAR.value

    def test_mutation_canonical_sequence(self):
        """Test: AL → OZ → ZHIR → IL → SHA (mutación canónica)."""
        result = validate_sequence(
            [EMISSION, RECEPTION, COHERENCE, DISSONANCE, MUTATION, COHERENCE, SILENCE]
        )
        assert result.passed
        assert result.metadata["has_dissonance"]

    def test_self_organization_sequence(self):
        """Test: THOL[AL → OZ → ZHIR → IL] → RA → SHA."""
        # Simplified: THOL followed by operators and closure
        # Updated: Added OZ before THOL (R4 evolved rule requires destabilizer)
        result = validate_sequence(
            [
                EMISSION,
                RECEPTION,
                COHERENCE,
                DISSONANCE,
                SELF_ORGANIZATION,
                MUTATION,
                COHERENCE,
                SILENCE,
            ]
        )
        assert result.passed
        assert (
            result.metadata["detected_pattern"] == StructuralPattern.HIERARCHICAL.value
        )

    def test_cyclic_regenerative_sequence(self):
        """Test: NAV → AL → IL → RA → NAV → THOL (ciclo regenerativo)."""
        # This would have transition appearing multiple times
        result = validate_sequence(
            [
                RECURSIVITY,
                RECEPTION,
                COHERENCE,
                RESONANCE,
                TRANSITION,
                COHERENCE,
                TRANSITION,
            ]
        )
        assert result.passed
        # Pattern detection looks for multiple TRANSITION occurrences
        assert result.metadata["detected_pattern"] in {
            StructuralPattern.CYCLIC.value,
            StructuralPattern.FRACTAL.value,
        }


class TestStructuralPatternDetection:
    """Test automatic structural pattern detection."""

    def test_detect_linear_pattern(self):
        """Simple sequences should be detected as linear."""
        result = validate_sequence([EMISSION, RECEPTION, COHERENCE, RESONANCE, SILENCE])
        assert result.passed
        assert result.metadata["detected_pattern"] == StructuralPattern.LINEAR.value

    def test_detect_hierarchical_pattern(self):
        """Sequences with THOL should be detected as hierarchical."""
        # Updated: Added DISSONANCE before THOL (R4 evolved rule requires destabilizer)
        result = validate_sequence(
            [EMISSION, RECEPTION, COHERENCE, DISSONANCE, SELF_ORGANIZATION, SILENCE]
        )
        assert result.passed
        assert (
            result.metadata["detected_pattern"] == StructuralPattern.HIERARCHICAL.value
        )

    def test_detect_bifurcated_pattern(self):
        """OZ → ZHIR sequences should be detected as bifurcated."""
        result = validate_sequence(
            [EMISSION, RECEPTION, COHERENCE, DISSONANCE, MUTATION, COHERENCE, SILENCE]
        )
        assert result.passed
        assert result.metadata["detected_pattern"] == StructuralPattern.BIFURCATED.value

    def test_detect_cyclic_pattern(self):
        """Multiple NAV occurrences suggest cyclic or fractal pattern.
        
        Note: With coherence weighting, FRACTAL (NAV + RECURSIVITY) may be
        detected instead of CYCLIC when both are present, as FRACTAL represents
        deeper structural complexity (recursive structure across scales).
        """
        result = validate_sequence(
            [
                RECURSIVITY,
                RECEPTION,
                COHERENCE,
                TRANSITION,
                RESONANCE,
                TRANSITION,
            ]
        )
        assert result.passed
        # FRACTAL wins due to NAV + RECURSIVITY having higher structural depth
        assert result.metadata["detected_pattern"] in {
            StructuralPattern.CYCLIC.value,
            StructuralPattern.FRACTAL.value,
        }

    def test_detect_fractal_pattern(self):
        """NAV with coupling suggests fractal recursion."""
        result = validate_sequence(
            [RECURSIVITY, RECEPTION, COHERENCE, COUPLING, RESONANCE, TRANSITION]
        )
        assert result.passed
        assert result.metadata["detected_pattern"] == StructuralPattern.FRACTAL.value


class TestInvalidSequences:
    """Test that invalid sequences are properly rejected."""

    def test_empty_sequence(self):
        """Empty sequence should fail."""
        with pytest.raises(SequenceSyntaxError) as excinfo:
            parse_sequence([])

        assert "empty" in excinfo.value.message.lower()

    def test_mutation_without_dissonance_at_start(self):
        """Mutation at the start without prior dissonance should fail."""
        with pytest.raises(SequenceSyntaxError) as excinfo:
            parse_sequence([EMISSION, MUTATION, COHERENCE, SILENCE])

        error = excinfo.value
        assert "mutation" in error.message.lower()
        assert "dissonance" in error.message.lower()

    def test_unknown_operators(self):
        """Unknown operators should be rejected."""
        result = validate_sequence([EMISSION, "UNKNOWN_OP", COHERENCE, SILENCE])
        assert not result.passed
        assert "unknown" in result.message.lower()


class TestMetadata:
    """Test that metadata is correctly populated."""

    def test_metadata_has_dissonance(self):
        """Metadata should track dissonance presence."""
        result = validate_sequence(
            [EMISSION, RECEPTION, COHERENCE, DISSONANCE, MUTATION, COHERENCE, SILENCE]
        )
        assert result.passed
        assert result.metadata["has_dissonance"] is True

    def test_metadata_no_dissonance(self):
        """Metadata should track absence of dissonance."""
        result = validate_sequence([EMISSION, RECEPTION, COHERENCE, RESONANCE, SILENCE])
        assert result.passed
        assert result.metadata["has_dissonance"] is False

    def test_metadata_has_stabilizer(self):
        """Metadata should track stabilizer presence."""
        result = validate_sequence([EMISSION, RECEPTION, COHERENCE, RESONANCE, SILENCE])
        assert result.passed
        assert result.metadata["has_stabilizer"] is True

    def test_metadata_pattern_detection(self):
        """Metadata should include detected pattern."""
        result = validate_sequence([EMISSION, RECEPTION, COHERENCE, RESONANCE, SILENCE])
        assert result.passed
        assert "detected_pattern" in result.metadata
        assert result.metadata["detected_pattern"] in {
            pattern.value for pattern in StructuralPattern
        }
