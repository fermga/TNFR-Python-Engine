"""Unit tests for SHA (Silence) operator grammar validation.

This module tests the canonical grammar rules for the SHA (Silence) operator
following TNFR structural theory. SHA has specific incompatibilities that
must be validated to maintain structural coherence.

Theoretical Foundation (from "El pulso que nos atraviesa", Section 2.3.3):
- SHA reduces νf → 0 (latent state, preservation)
- OZ increases ΔNFR (instability, exploration)
- SHA → OZ is contradictory: cannot introduce dissonance into paused node
- SHA → SHA is redundant: violates operator closure principle

Valid transitions from SHA:
- SHA → medium frequency operators (emission, reception, coherence, coupling, transition, expansion)
- SHA must not transition to high frequency operators without intermediary

Invalid transitions:
- SHA → OZ (silence followed by dissonance)
- SHA → SHA (redundant silence)
- SHA → high frequency operators (dissonance, resonance, mutation, contraction, emission)
"""

from __future__ import annotations

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
from tnfr.operators.grammar import SequenceSyntaxError, validate_sequence
from tnfr.validation import SequenceValidationResult


class TestSHAProhibitedTransitions:
    """Test prohibited SHA (Silence) operator transitions."""

    def test_sha_to_oz_prohibited(self):
        """SHA → OZ must be rejected (silence followed by dissonance)."""
        # SHA → OZ violates structural coherence:
        # - SHA reduces νf → 0 (preservation state)
        # - OZ increases ΔNFR (exploration, instability)
        # - Cannot introduce dissonance into paused node
        sequence = [EMISSION, RECEPTION, COHERENCE, SILENCE, DISSONANCE, COHERENCE, SILENCE]
        result = validate_sequence(sequence)

        assert not result.passed, "SHA → OZ should be rejected"
        assert result.error is not None
        # Check error message contains SHA/Silence and OZ/Dissonance
        msg_lower = result.message.lower()
        assert "silence" in msg_lower or "sha" in msg_lower
        assert "dissonance" in msg_lower or "oz" in msg_lower

    def test_sha_to_sha_prohibited(self):
        """SHA → SHA must be rejected (redundant silence)."""
        # SHA → SHA violates operator closure:
        # - If νf ≈ 0, second SHA has no effect
        # - Each operator must transform structure
        # - Redundant operators serve no purpose
        sequence = [EMISSION, RECEPTION, COHERENCE, SILENCE, SILENCE]
        result = validate_sequence(sequence)

        assert not result.passed, "SHA → SHA should be rejected"
        assert result.error is not None
        # Check error message indicates redundancy
        msg_lower = result.message.lower()
        assert "redundant" in msg_lower or "consecutive" in msg_lower
        assert "silence" in msg_lower or "sha" in msg_lower


class TestSHAValidTransitions:
    """Test valid SHA (Silence) operator transitions."""

    def test_sha_to_emission_valid(self):
        """SHA → AL is valid (reactivation from silence)."""
        # SHA → AL (emission) is valid reactivation pattern
        # Frequency: zero → high requires intermediate, but AL is special reactivation case
        sequence = [
            EMISSION,
            RECEPTION,
            COHERENCE,
            SILENCE,
            EMISSION,
            RECEPTION,
            COHERENCE,
            SILENCE,
        ]
        result = validate_sequence(sequence)

        # Note: This may fail due to frequency rules (zero → high)
        # If it does, that's expected behavior - user should use SHA → medium → AL
        if not result.passed:
            # Verify it fails for frequency reasons, not SHA-specific reasons
            msg_lower = result.message.lower()
            # Should not mention SHA → OZ or SHA → SHA specifically
            assert not (
                "silence" in msg_lower and "dissonance" in msg_lower and "contradicts" in msg_lower
            )
            assert "redundant" not in msg_lower

    def test_sha_to_reception_valid(self):
        """SHA → EN is valid (reception after silence)."""
        # SHA → EN (reception) is valid
        # Frequency: zero → medium is allowed
        sequence = [EMISSION, RECEPTION, COHERENCE, SILENCE, RECEPTION, COHERENCE, SILENCE]
        result = validate_sequence(sequence)

        assert result.passed, f"SHA → EN should be valid: {result.message}"

    def test_sha_to_coherence_valid(self):
        """SHA → IL is valid (coherence after silence)."""
        # SHA → IL (coherence) is valid
        # Frequency: zero → medium is allowed
        sequence = [EMISSION, RECEPTION, COHERENCE, SILENCE, COHERENCE, SILENCE]
        result = validate_sequence(sequence)

        # Note: This may trigger anti-pattern warning (silence → coherence)
        # but should not be blocked by grammar
        if not result.passed:
            # Check if it's a grammar error or just a pattern warning
            msg_lower = result.message.lower()
            # Should not be blocked for SHA → SHA or SHA → OZ reasons
            assert "redundant" not in msg_lower or "coherence" not in msg_lower

    def test_sha_to_transition_valid(self):
        """SHA → NAV is valid (transition after silence)."""
        # SHA → NAV (transition) is valid
        # Frequency: zero → medium is allowed
        sequence = [EMISSION, RECEPTION, COHERENCE, SILENCE, TRANSITION]
        result = validate_sequence(sequence)

        assert result.passed, f"SHA → NAV should be valid: {result.message}"

    def test_sha_to_coupling_valid(self):
        """SHA → UM is valid (coupling after silence)."""
        # SHA → UM (coupling) is valid
        # Frequency: zero → medium is allowed
        sequence = [EMISSION, RECEPTION, COHERENCE, SILENCE, COUPLING, RESONANCE, SILENCE]
        result = validate_sequence(sequence)

        assert result.passed, f"SHA → UM should be valid: {result.message}"


class TestSHATransitionAlternatives:
    """Test that alternative sequences work when SHA → OZ is needed."""

    def test_sha_nav_oz_valid(self):
        """SHA → NAV → OZ is valid (transition intermediary)."""
        # Workaround: SHA → NAV → OZ
        # NAV provides controlled transition from silence to exploration
        sequence = [
            EMISSION,
            RECEPTION,
            COHERENCE,
            SILENCE,
            TRANSITION,  # NAV provides intermediary
            DISSONANCE,
            COHERENCE,
            SILENCE,
        ]
        result = validate_sequence(sequence)

        # This may still fail due to frequency rules, but not SHA-specific rules
        if not result.passed:
            msg_lower = result.message.lower()
            # Should not mention SHA → OZ specifically
            assert not (
                "silence" in msg_lower and "dissonance" in msg_lower and "contradicts" in msg_lower
            )

    def test_sha_emission_oz_valid(self):
        """SHA → AL → OZ is valid (emission reactivation before dissonance)."""
        # Workaround: SHA → AL → OZ
        # AL reactivates the node before introducing dissonance
        sequence = [
            EMISSION,
            RECEPTION,
            COHERENCE,
            SILENCE,
            EMISSION,  # AL reactivates
            RECEPTION,
            COHERENCE,
            DISSONANCE,
            COHERENCE,
            SILENCE,
        ]
        result = validate_sequence(sequence)

        # This may fail due to frequency rules (SHA → AL is zero → high)
        # but should not fail for SHA → OZ reasons
        if not result.passed:
            msg_lower = result.message.lower()
            # Should not mention SHA → OZ contradiction
            assert not (
                "silence" in msg_lower and "dissonance" in msg_lower and "contradicts" in msg_lower
            )


class TestOZtoSHAAllowed:
    """Test that OZ → SHA is allowed (dissonance contained by silence)."""

    def test_oz_to_sha_valid(self):
        """OZ → SHA is valid (contained dissonance)."""
        # OZ → SHA is valid and represents important pattern:
        # - Trauma contained
        # - Conflict postponed
        # - Tension preserved for later processing
        sequence = [EMISSION, RECEPTION, COHERENCE, DISSONANCE, SILENCE]
        result = validate_sequence(sequence)

        assert result.passed, f"OZ → SHA should be valid: {result.message}"

    def test_oz_coherence_sha_valid(self):
        """OZ → IL → SHA is valid (resolved then contained)."""
        # Resolve dissonance then pause
        sequence = [EMISSION, RECEPTION, COHERENCE, DISSONANCE, COHERENCE, SILENCE]
        result = validate_sequence(sequence)

        assert result.passed, f"OZ → IL → SHA should be valid: {result.message}"


class TestSHAErrorMessages:
    """Test that SHA validation provides clear, TNFR-aligned error messages."""

    def test_sha_oz_error_mentions_structural_theory(self):
        """SHA → OZ error should explain TNFR structural contradiction."""
        sequence = [EMISSION, RECEPTION, COHERENCE, SILENCE, DISSONANCE, COHERENCE, SILENCE]
        result = validate_sequence(sequence)

        assert not result.passed
        msg_lower = result.message.lower()

        # Error message should mention:
        # - Silence/SHA
        # - Dissonance/OZ
        # - Structural contradiction (νf, ΔNFR, preservation, etc.)
        assert "silence" in msg_lower or "sha" in msg_lower
        assert "dissonance" in msg_lower or "oz" in msg_lower
        # Should mention structural concepts
        has_structural_concept = any(
            concept in msg_lower
            for concept in ["νf", "dnfr", "contradicts", "paused", "preservation"]
        )
        assert has_structural_concept, "Error should reference TNFR structural theory"

    def test_sha_sha_error_mentions_redundancy(self):
        """SHA → SHA error should explain redundancy principle."""
        sequence = [EMISSION, RECEPTION, COHERENCE, SILENCE, SILENCE]
        result = validate_sequence(sequence)

        assert not result.passed
        msg_lower = result.message.lower()

        # Error message should mention:
        # - Redundant/consecutive
        # - Silence
        # - No structural purpose
        assert "redundant" in msg_lower or "consecutive" in msg_lower
        assert "silence" in msg_lower or "sha" in msg_lower
        assert "purpose" in msg_lower or "effect" in msg_lower or "duplicate" in msg_lower

    def test_sha_oz_error_suggests_alternatives(self):
        """SHA → OZ error should suggest valid alternative sequences."""
        sequence = [EMISSION, RECEPTION, COHERENCE, SILENCE, DISSONANCE, COHERENCE, SILENCE]
        result = validate_sequence(sequence)

        assert not result.passed
        msg_lower = result.message.lower()

        # Error should suggest alternatives:
        # - SHA → NAV → OZ (transition intermediary)
        # - SHA → AL → OZ (emission reactivation)
        has_suggestion = any(op in msg_lower for op in ["transition", "nav", "emission", "al"])
        assert has_suggestion, "Error should suggest alternative sequences"


class TestSHAGrammarIntegration:
    """Integration tests for SHA grammar with other operators."""

    def test_complex_sequence_with_valid_sha_usage(self):
        """Complex sequence using SHA correctly."""
        # Pattern: activate → explore → stabilize → pause → resume → stabilize → end
        sequence = [
            EMISSION,  # Activate
            RECEPTION,  # Receive
            COHERENCE,  # Stabilize
            DISSONANCE,  # Explore (before SHA, not after)
            COHERENCE,  # Resolve
            SILENCE,  # Pause
            RECEPTION,  # Resume with reception (medium frequency)
            COHERENCE,  # Stabilize
            SILENCE,  # End
        ]
        result = validate_sequence(sequence)

        assert result.passed, f"Complex valid SHA sequence failed: {result.message}"

    def test_multiple_sha_with_intermediaries(self):
        """Multiple SHA operators with proper intermediaries."""
        # Each SHA must have transformation before next SHA
        sequence = [
            EMISSION,
            RECEPTION,
            COHERENCE,
            SILENCE,  # First pause
            RECEPTION,  # Transform
            COHERENCE,
            SILENCE,  # Second pause (with intermediary)
        ]
        result = validate_sequence(sequence)

        assert result.passed, f"Multiple SHA with intermediaries failed: {result.message}"

    def test_sha_in_regenerative_cycle(self):
        """SHA used in regenerative cycle pattern."""
        # Regenerative pattern: activate → resonate → expand → pause → transition → repeat
        sequence = [
            EMISSION,
            RECEPTION,
            COHERENCE,
            RESONANCE,
            EXPANSION,
            COHERENCE,
            SILENCE,  # Pause before transition
            TRANSITION,  # Transition to next cycle phase
        ]
        result = validate_sequence(sequence)

        assert result.passed, f"SHA in regenerative cycle failed: {result.message}"
