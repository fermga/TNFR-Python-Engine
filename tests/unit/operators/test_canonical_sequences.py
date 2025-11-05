"""Tests for canonical glyph sequences as specified in TNFR.pdf pp. 133-137.

This test suite validates that the grammar implementation correctly enforces
the canonical rules from the foundational TNFR document:

R1: Sequences must start with AL (EMISSION) or NAV (TRANSITION)
R2: Must contain at least one stabilizer IL (COHERENCE) or THOL (SELF_ORGANIZATION)
R3: Must end with SHA (SILENCE) or NUL (CONTRACTION)
R4: ZHIR (MUTATION) must be preceded by OZ (DISSONANCE)
R5: Loops must include transformation or become entropic
"""

import pytest

from tnfr.config.operator_names import (
    COHERENCE,
    CONTRACTION,
    COUPLING,
    DISSONANCE,
    EMISSION,
    MUTATION,
    RESONANCE,
    SELF_ORGANIZATION,
    SILENCE,
    TRANSITION,
)
from tnfr.operators.grammar import SequenceSyntaxError, validate_sequence


class TestCanonicalBasicSequences:
    """Test basic canonical sequences from TNFR.pdf."""

    def test_al_to_il_to_sha(self) -> None:
        """AL → IL → SHA: Basic activation with coherence and silence."""
        result = validate_sequence([EMISSION, COHERENCE, SILENCE])
        assert result.passed, f"Expected valid sequence, got: {result.message}"
        assert result.metadata["has_stabilizer"]

    def test_al_to_il_to_nul(self) -> None:
        """AL → IL → NUL: Basic activation with coherence and contraction."""
        result = validate_sequence([EMISSION, COHERENCE, CONTRACTION])
        assert result.passed, f"Expected valid sequence, got: {result.message}"
        assert result.metadata["has_stabilizer"]

    def test_nav_to_il_to_sha(self) -> None:
        """NAV → IL → SHA: Transition with coherence stabilization."""
        result = validate_sequence([TRANSITION, COHERENCE, SILENCE])
        assert result.passed, f"Expected valid sequence, got: {result.message}"

    def test_al_to_thol_to_sha(self) -> None:
        """AL → THOL → SHA: Emission with self-organization and silence closure."""
        result = validate_sequence([EMISSION, SELF_ORGANIZATION, SILENCE])
        assert result.passed, f"Expected valid sequence, got: {result.message}"
        assert result.metadata["has_stabilizer"]

    def test_al_to_thol_to_nul(self) -> None:
        """AL → THOL → NUL: Emission with self-organization and contraction closure."""
        result = validate_sequence([EMISSION, SELF_ORGANIZATION, CONTRACTION])
        assert result.passed, f"Expected valid sequence, got: {result.message}"


class TestCanonicalComplexSequences:
    """Test complex canonical sequences from TNFR.pdf."""

    def test_complete_reorganization_cycle(self) -> None:
        """AL → NAV → IL → OZ → THOL → RA → UM → SHA: Complete transformation cycle."""
        sequence = [
            EMISSION,
            TRANSITION,
            COHERENCE,
            DISSONANCE,
            SELF_ORGANIZATION,
            RESONANCE,
            COUPLING,
            SILENCE,
        ]
        result = validate_sequence(sequence)
        assert result.passed, f"Expected valid sequence, got: {result.message}"
        assert result.metadata["has_stabilizer"]
        assert not result.metadata["open_thol"]


class TestCanonicalAntiPatterns:
    """Test anti-patterns that should fail according to TNFR.pdf."""

    def test_sha_to_oz_contradictory(self) -> None:
        """SHA → OZ: Silence followed by dissonance is contradictory.
        
        Problem: SHA reduces νf to preserve EPI, but OZ immediately increases
        ΔNFR, creating reorganization pressure that violates SHA intention.
        """
        result = validate_sequence([SILENCE, DISSONANCE, SILENCE])
        assert not result.passed
        # Should fail because SILENCE is not a valid start operator
        assert "must start" in result.message

    def test_oz_to_oz_excessive_instability(self) -> None:
        """OZ → OZ: Consecutive dissonance causes excessive instability.
        
        Problem: Cumulative ΔNFR without resolution leads to structural collapse.
        """
        result = validate_sequence([DISSONANCE, DISSONANCE])
        assert not result.passed
        # Should fail because DISSONANCE is not a valid start operator
        assert "must start" in result.message

    def test_al_to_sha_no_stabilizer(self) -> None:
        """AL → SHA: Activation immediately silenced without stabilization.
        
        Problem: Contradicts activation purpose and lacks required stabilizer.
        """
        result = validate_sequence([EMISSION, SILENCE])
        assert not result.passed
        assert "stabilizer" in result.message

    def test_sequence_without_proper_end(self) -> None:
        """AL → IL → RA: Missing proper closure operator.
        
        R3 requires sequences to end with SHA or NUL to avoid collapse.
        """
        result = validate_sequence([EMISSION, COHERENCE, RESONANCE])
        assert not result.passed
        assert "must end with" in result.message


class TestCanonicalRules:
    """Test explicit canonical rules from TNFR.pdf."""

    def test_rule_r1_valid_start_operators(self) -> None:
        """R1: Sequences must start with AL (EMISSION) or NAV (TRANSITION)."""
        # Valid starts
        for start_op in [EMISSION, TRANSITION]:
            result = validate_sequence([start_op, COHERENCE, SILENCE])
            assert result.passed, f"{start_op} should be valid start operator"
        
        # Invalid starts
        for invalid_start in [COHERENCE, DISSONANCE, SILENCE, CONTRACTION]:
            result = validate_sequence([invalid_start, COHERENCE, SILENCE])
            assert not result.passed, f"{invalid_start} should not be valid start operator"
            assert "must start" in result.message

    def test_rule_r2_requires_stabilizer(self) -> None:
        """R2: Must contain at least one stabilizer IL or THOL."""
        # Valid: contains COHERENCE
        result = validate_sequence([EMISSION, COHERENCE, SILENCE])
        assert result.passed
        assert result.metadata["has_stabilizer"]
        
        # Valid: contains SELF_ORGANIZATION
        result = validate_sequence([EMISSION, SELF_ORGANIZATION, SILENCE])
        assert result.passed
        assert result.metadata["has_stabilizer"]
        
        # Invalid: no stabilizer
        result = validate_sequence([EMISSION, RESONANCE, SILENCE])
        assert not result.passed
        assert "stabilizer" in result.message

    def test_rule_r3_valid_end_operators(self) -> None:
        """R3: Must end with SHA (SILENCE) or NUL (CONTRACTION)."""
        # Valid ends
        for end_op in [SILENCE, CONTRACTION]:
            result = validate_sequence([EMISSION, COHERENCE, end_op])
            assert result.passed, f"{end_op} should be valid end operator"
        
        # Invalid ends - TRANSITION is no longer a valid end operator
        result = validate_sequence([EMISSION, COHERENCE, TRANSITION])
        assert not result.passed
        assert "must end with" in result.message

    def test_rule_r5_thol_requires_closure(self) -> None:
        """R5: Self-organization blocks must have proper closure.
        
        THOL blocks that remain open indicate incomplete transformation.
        """
        # Valid: THOL with SILENCE closure
        result = validate_sequence([EMISSION, SELF_ORGANIZATION, SILENCE])
        assert result.passed
        assert not result.metadata["open_thol"]
        
        # Valid: THOL with CONTRACTION closure
        result = validate_sequence([EMISSION, SELF_ORGANIZATION, CONTRACTION])
        assert result.passed
        assert not result.metadata["open_thol"]


class TestSequenceMetadata:
    """Test that sequence validation provides correct metadata."""

    def test_metadata_tracks_stabilizer(self) -> None:
        """Metadata should correctly track presence of stabilizers."""
        # With COHERENCE stabilizer
        result = validate_sequence([EMISSION, COHERENCE, SILENCE])
        assert result.metadata["has_stabilizer"] is True
        
        # With SELF_ORGANIZATION stabilizer
        result = validate_sequence([EMISSION, SELF_ORGANIZATION, SILENCE])
        assert result.metadata["has_stabilizer"] is True

    def test_metadata_tracks_thol_closure(self) -> None:
        """Metadata should correctly track THOL block closure."""
        # Closed THOL block
        result = validate_sequence([EMISSION, SELF_ORGANIZATION, SILENCE])
        assert result.metadata["open_thol"] is False
        
        # No THOL block
        result = validate_sequence([EMISSION, COHERENCE, SILENCE])
        assert result.metadata["open_thol"] is False

    def test_metadata_tracks_unknown_tokens(self) -> None:
        """Metadata should track any unknown operator tokens."""
        result = validate_sequence([EMISSION, COHERENCE, SILENCE])
        assert result.metadata["unknown_tokens"] == frozenset()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
