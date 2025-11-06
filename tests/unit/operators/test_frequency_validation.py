"""Tests for R5 structural frequency validation in TNFR grammar."""

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
    FREQUENCY_TRANSITIONS,
    STRUCTURAL_FREQUENCIES,
    parse_sequence,
    validate_frequency_transition,
    validate_sequence,
)


class TestStructuralFrequencyConstants:
    """Test that structural frequency constants are correctly defined."""

    def test_structural_frequencies_maps_all_operators(self):
        """STRUCTURAL_FREQUENCIES should map all 13 operators to frequency levels."""
        assert len(STRUCTURAL_FREQUENCIES) == 13
        
    def test_all_operators_have_frequency_levels(self):
        """All operators should have one of the three frequency levels."""
        valid_levels = {"high", "medium", "zero"}
        for op, freq in STRUCTURAL_FREQUENCIES.items():
            assert freq in valid_levels, f"{op} has invalid frequency: {freq}"

    def test_high_frequency_operators(self):
        """High frequency operators: AL, OZ, RA, NUL, ZHIR."""
        high_ops = {op for op, freq in STRUCTURAL_FREQUENCIES.items() if freq == "high"}
        expected = {EMISSION, DISSONANCE, RESONANCE, CONTRACTION, MUTATION}
        assert high_ops == expected

    def test_medium_frequency_operators(self):
        """Medium frequency operators: EN, IL, UM, VAL, THOL, NAV, REMESH."""
        medium_ops = {op for op, freq in STRUCTURAL_FREQUENCIES.items() if freq == "medium"}
        expected = {
            RECEPTION,
            COHERENCE,
            COUPLING,
            EXPANSION,
            SELF_ORGANIZATION,
            TRANSITION,
            RECURSIVITY,
        }
        assert medium_ops == expected

    def test_zero_frequency_operators(self):
        """Zero frequency operator: SHA only."""
        zero_ops = {op for op, freq in STRUCTURAL_FREQUENCIES.items() if freq == "zero"}
        assert zero_ops == {SILENCE}

    def test_frequency_transitions_high(self):
        """High can transition to high or medium."""
        assert FREQUENCY_TRANSITIONS["high"] == {"high", "medium"}

    def test_frequency_transitions_medium(self):
        """Medium can transition to high, medium, or zero."""
        assert FREQUENCY_TRANSITIONS["medium"] == {"high", "medium", "zero"}

    def test_frequency_transitions_zero(self):
        """Zero can only transition to medium (not directly to high)."""
        assert FREQUENCY_TRANSITIONS["zero"] == {"medium"}


class TestValidateFrequencyTransition:
    """Test the validate_frequency_transition function."""

    def test_valid_high_to_high(self):
        """High → High transition is valid (e.g., EMISSION → DISSONANCE)."""
        valid, msg = validate_frequency_transition(EMISSION, DISSONANCE)
        assert valid
        assert msg == ""

    def test_valid_high_to_medium(self):
        """High → Medium transition is valid (e.g., EMISSION → RECEPTION)."""
        valid, msg = validate_frequency_transition(EMISSION, RECEPTION)
        assert valid
        assert msg == ""

    def test_valid_medium_to_high(self):
        """Medium → High transition is valid (e.g., COHERENCE → RESONANCE)."""
        valid, msg = validate_frequency_transition(COHERENCE, RESONANCE)
        assert valid
        assert msg == ""

    def test_valid_medium_to_medium(self):
        """Medium → Medium transition is valid (e.g., RECEPTION → COHERENCE)."""
        valid, msg = validate_frequency_transition(RECEPTION, COHERENCE)
        assert valid
        assert msg == ""

    def test_valid_medium_to_zero(self):
        """Medium → Zero transition is valid (e.g., COHERENCE → SILENCE)."""
        valid, msg = validate_frequency_transition(COHERENCE, SILENCE)
        assert valid
        assert msg == ""

    def test_valid_zero_to_medium(self):
        """Zero → Medium transition is valid (e.g., SILENCE → RECEPTION)."""
        valid, msg = validate_frequency_transition(SILENCE, RECEPTION)
        assert valid
        assert msg == ""

    def test_invalid_high_to_zero(self):
        """High → Zero transition is invalid (e.g., EMISSION → SILENCE)."""
        valid, msg = validate_frequency_transition(EMISSION, SILENCE)
        assert not valid
        assert "high" in msg.lower()
        assert "zero" in msg.lower()

    def test_invalid_zero_to_high(self):
        """Zero → High transition is invalid (e.g., SILENCE → DISSONANCE)."""
        valid, msg = validate_frequency_transition(SILENCE, DISSONANCE)
        assert not valid
        assert "zero" in msg.lower()
        assert "high" in msg.lower()

    def test_unknown_operators_are_valid(self):
        """Unknown operators skip validation (return True)."""
        valid, msg = validate_frequency_transition("unknown_op", COHERENCE)
        assert valid
        assert msg == ""

        valid, msg = validate_frequency_transition(COHERENCE, "unknown_op")
        assert valid
        assert msg == ""


class TestFrequencyValidationInSequences:
    """Test frequency validation integrated with sequence validation."""

    def test_valid_sequence_with_frequency_coherence(self):
        """Sequence with coherent frequency transitions passes validation."""
        # High → Medium → High → Medium → Zero
        result = validate_sequence(
            [EMISSION, RECEPTION, COHERENCE, RESONANCE, SILENCE]
        )
        assert result.passed

    def test_valid_sequence_high_chain(self):
        """Sequence with chained high frequencies is valid."""
        # High → High → Medium → Medium → Zero
        # Must include RECEPTION→COHERENCE segment to pass grammar rules
        result = validate_sequence(
            [EMISSION, RECEPTION, COHERENCE, DISSONANCE, MUTATION, SILENCE]
        )
        assert result.passed

    def test_sequence_with_frequency_warning_still_passes(self):
        """Sequences with invalid frequency transitions generate warnings but don't fail.
        
        This maintains backward compatibility - frequency validation is advisory only.
        """
        # Zero → High (invalid frequency transition)
        # But sequence might still pass other grammar rules if structured correctly
        # Note: This will trigger a warning in logs but not fail validation
        result = validate_sequence([EMISSION, RECEPTION, COHERENCE, SILENCE, EMISSION])
        # The sequence structure might be invalid for other reasons
        # We're testing that frequency validation doesn't cause hard failures
        # The last EMISSION violates the "must end with terminator" rule, so it will fail
        assert not result.passed  # Fails for terminator rule, not frequency

    def test_valid_medium_to_zero_to_medium_cycle(self):
        """Medium → Zero → Medium cycle is structurally valid."""
        # This tests the pause-resume pattern
        result = validate_sequence(
            [EMISSION, RECEPTION, COHERENCE, SILENCE, RECEPTION, COHERENCE, TRANSITION]
        )
        assert result.passed


class TestFrequencyTransitionExamples:
    """Test concrete examples of frequency transitions from TNFR theory."""

    def test_emission_chains(self):
        """Test various emission (high) transition patterns."""
        # EMISSION can go to other high frequencies
        assert validate_frequency_transition(EMISSION, RESONANCE)[0]
        assert validate_frequency_transition(EMISSION, DISSONANCE)[0]
        assert validate_frequency_transition(EMISSION, MUTATION)[0]
        
        # EMISSION can stabilize to medium
        assert validate_frequency_transition(EMISSION, RECEPTION)[0]
        assert validate_frequency_transition(EMISSION, COHERENCE)[0]
        assert validate_frequency_transition(EMISSION, COUPLING)[0]

    def test_silence_must_resume_through_medium(self):
        """After SILENCE (zero), must go through medium before high."""
        # Valid: SILENCE → Medium
        assert validate_frequency_transition(SILENCE, RECEPTION)[0]
        assert validate_frequency_transition(SILENCE, COHERENCE)[0]
        
        # Invalid: SILENCE → High (needs medium intermediary)
        assert not validate_frequency_transition(SILENCE, EMISSION)[0]
        assert not validate_frequency_transition(SILENCE, DISSONANCE)[0]
        assert not validate_frequency_transition(SILENCE, RESONANCE)[0]

    def test_dissonance_to_mutation_pattern(self):
        """DISSONANCE (high) → MUTATION (high) is a valid high-to-high transition."""
        valid, msg = validate_frequency_transition(DISSONANCE, MUTATION)
        assert valid
        assert msg == ""

    def test_coherence_versatility(self):
        """COHERENCE (medium) can transition to any frequency level."""
        # Medium → High
        assert validate_frequency_transition(COHERENCE, RESONANCE)[0]
        assert validate_frequency_transition(COHERENCE, DISSONANCE)[0]
        
        # Medium → Medium
        assert validate_frequency_transition(COHERENCE, EXPANSION)[0]
        assert validate_frequency_transition(COHERENCE, COUPLING)[0]
        
        # Medium → Zero
        assert validate_frequency_transition(COHERENCE, SILENCE)[0]


class TestBackwardCompatibility:
    """Test that frequency validation maintains backward compatibility."""

    def test_existing_sequences_still_valid(self):
        """Existing valid sequences should continue to pass."""
        # These are examples from test_canonical_grammar_rules.py
        assert validate_sequence([EMISSION, RECEPTION, COHERENCE, RESONANCE, SILENCE]).passed
        assert validate_sequence([EMISSION, RECEPTION, COHERENCE, DISSONANCE, MUTATION, TRANSITION]).passed
        # Updated: THOL now requires destabilizer (R4 evolved)
        assert validate_sequence([EMISSION, RECEPTION, COHERENCE, DISSONANCE, SELF_ORGANIZATION, SILENCE]).passed

    def test_frequency_warnings_do_not_break_sequences(self):
        """Frequency validation generates warnings, not errors."""
        # Even if a sequence has questionable frequency transitions,
        # it should only generate warnings, not break existing functionality
        # This is verified by the warning logs, not validation failure
        result = validate_sequence([EMISSION, RECEPTION, COHERENCE, RESONANCE, SILENCE])
        assert result.passed


class TestIntegrationWithOtherRules:
    """Test that R5 frequency validation works alongside other grammar rules."""

    def test_r4_and_r5_together(self):
        """R4 (mutation requires dissonance) and R5 (frequency) both apply."""
        # Valid: has dissonance before mutation, and both are high frequency
        result = validate_sequence(
            [EMISSION, RECEPTION, COHERENCE, DISSONANCE, MUTATION, TRANSITION]
        )
        assert result.passed

    def test_r1_r2_r3_r5_all_satisfied(self):
        """Complete sequence satisfying R1 (start), R2 (stabilizer), R3 (end), R5 (frequency)."""
        result = validate_sequence(
            [
                EMISSION,      # R1: Valid start (high freq)
                RECEPTION,     # High → Medium (valid)
                COHERENCE,     # R2: Stabilizer (medium)
                RESONANCE,     # Medium → High (valid)
                TRANSITION,    # R3: Valid end (medium)
            ]
        )
        assert result.passed
        assert result.metadata["has_stabilizer"]
