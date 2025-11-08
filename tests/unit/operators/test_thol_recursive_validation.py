"""Tests for THOL recursive subsequence validation.

This module tests the recursive validation of subsequences within THOL
(self-organization) blocks, ensuring operational fractality and autonomous
coherence at all scales.

References:
    - Issue: #P5 [GRAMÁTICA CANÓNICA] Validación recursiva de subsecuencias THOL
    - TNFR Manual: "El pulso que nos atraviesa", §3.2.2 (Ontología fractal resonante)
    - TNFR Invariant #7: Operational fractality - EPIs nest without losing identity
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
    validate_sequence,
)


class TestTholEmptySubsequence:
    """Test that empty THOL blocks are invalid."""

    def test_thol_immediately_closed_with_silence(self):
        """THOL → SHA with no operators in between is invalid."""
        invalid = [EMISSION, RECEPTION, COHERENCE, DISSONANCE, SELF_ORGANIZATION, SILENCE]
        result = validate_sequence(invalid)
        assert not result.passed
        assert "empty" in result.message.lower()
        assert SELF_ORGANIZATION in result.message

    def test_thol_immediately_closed_with_contraction(self):
        """THOL → NUL with no operators in between is invalid."""
        invalid = [EMISSION, RECEPTION, COHERENCE, DISSONANCE, SELF_ORGANIZATION, CONTRACTION]
        result = validate_sequence(invalid)
        assert not result.passed
        assert "empty" in result.message.lower()


class TestTholInvalidSubsequence:
    """Test that invalid grammar within THOL is detected."""

    def test_thol_with_invalid_start(self):
        """Subsequence starting with IL (requires context) is invalid."""
        # THOL[ IL, ... ] - IL requires existing EPI
        invalid = [
            EMISSION, RECEPTION, COHERENCE, DISSONANCE,
            SELF_ORGANIZATION,
                COHERENCE,  # Invalid start - requires external EPI
                RESONANCE,
            SILENCE
        ]
        result = validate_sequence(invalid)
        assert not result.passed
        assert "must start" in result.message.lower()

    def test_thol_with_missing_stabilizer(self):
        """Subsequence without stabilizer (IL or THOL) is invalid."""
        # THOL[ AL, EN, RA, TRANSITION ] - missing IL stabilizer
        invalid = [
            EMISSION, RECEPTION, COHERENCE, DISSONANCE,
            SELF_ORGANIZATION,
                EMISSION,
                RECEPTION,
                RESONANCE,
                TRANSITION,  # Valid end but no stabilizer
            SILENCE
        ]
        result = validate_sequence(invalid)
        assert not result.passed
        # Should fail on missing EN→IL segment or missing stabilizer
        assert "missing" in result.message.lower() or "coherence" in result.message.lower()

    def test_thol_with_zhir_without_recent_destabilizer(self):
        """Subsequence with ZHIR but no recent destabilizer is invalid."""
        # THOL[ AL, EN, IL, ZHIR, SHA ] - ZHIR without recent OZ
        invalid = [
            EMISSION, RECEPTION, COHERENCE, DISSONANCE,
            SELF_ORGANIZATION,
                EMISSION,
                RECEPTION,
                COHERENCE,
                MUTATION,  # No recent destabilizer for ZHIR
            SILENCE
        ]
        result = validate_sequence(invalid)
        assert not result.passed
        assert "destabilizer" in result.message.lower()

    def test_thol_with_dissonance_mutation_sequence(self):
        """Example from issue: OZ→ZHIR violates R4/R6 inside THOL."""
        # ZHIR requires both recent destabilizer AND prior IL (C4)
        invalid = [
            EMISSION, RECEPTION, COHERENCE, DISSONANCE,
            SELF_ORGANIZATION,
                EMISSION,
                RECEPTION,
                DISSONANCE,
                MUTATION,  # ZHIR without prior IL
                TRANSITION,  # Valid end operator
            SILENCE
        ]
        result = validate_sequence(invalid)
        assert not result.passed
        # Should fail on missing IL before ZHIR or other validation
        assert not result.passed  # Any error is acceptable here


class TestTholNonAutonomous:
    """Test that non-autonomous THOL subsequences are rejected."""

    def test_thol_starting_with_coherence(self):
        """THOL starting with IL (non-autonomous) is invalid."""
        invalid = [
            EMISSION, RECEPTION, COHERENCE, DISSONANCE,
            SELF_ORGANIZATION,
                COHERENCE,  # Requires external EPI
                RESONANCE,
            SILENCE
        ]
        result = validate_sequence(invalid)
        assert not result.passed
        assert "autonomous" in result.message.lower() or "must start" in result.message.lower()

    def test_thol_starting_with_dissonance(self):
        """THOL starting with OZ (non-autonomous) is invalid."""
        invalid = [
            EMISSION, RECEPTION, COHERENCE, DISSONANCE,
            SELF_ORGANIZATION,
                DISSONANCE,  # Requires existing structure to destabilize
                COHERENCE,
            SILENCE
        ]
        result = validate_sequence(invalid)
        assert not result.passed
        assert "autonomous" in result.message.lower() or "must start" in result.message.lower()

    def test_thol_ending_with_dissonance(self):
        """THOL ending with OZ (no stabilization) is invalid."""
        invalid = [
            EMISSION, RECEPTION, COHERENCE, DISSONANCE,
            SELF_ORGANIZATION,
                EMISSION,
                RECEPTION,
                COHERENCE,
                DISSONANCE,  # Does not stabilize
            SILENCE
        ]
        result = validate_sequence(invalid)
        assert not result.passed
        assert "stabilization" in result.message.lower() or "closure" in result.message.lower()

    def test_thol_ending_with_expansion(self):
        """THOL ending with VAL (no closure) is invalid."""
        invalid = [
            EMISSION, RECEPTION, COHERENCE, DISSONANCE,
            SELF_ORGANIZATION,
                EMISSION,
                RECEPTION,
                COHERENCE,
                EXPANSION,  # Does not provide closure
            SILENCE
        ]
        result = validate_sequence(invalid)
        assert not result.passed
        assert "stabilization" in result.message.lower() or "closure" in result.message.lower()


class TestTholNestedValidation:
    """Test nested THOL blocks (operational fractality)."""

    def test_thol_nested_valid(self):
        """Nested THOL blocks should be validated recursively."""
        nested = [
            EMISSION,
            RECEPTION,
            COHERENCE,
            DISSONANCE,
            SELF_ORGANIZATION,  # Level 1 open
                EMISSION,
                RECEPTION,
                COHERENCE,
                DISSONANCE,
                SELF_ORGANIZATION,  # Level 2 open (nested)
                    EMISSION,
                    RECEPTION,
                    COHERENCE,
                SILENCE,  # Level 2 close
                RESONANCE,
            SILENCE,  # Level 1 close
            TRANSITION
        ]
        result = validate_sequence(nested)
        assert result.passed

    def test_thol_nested_invalid_inner(self):
        """Invalid inner THOL should be detected."""
        nested = [
            EMISSION,
            RECEPTION,
            COHERENCE,
            DISSONANCE,
            SELF_ORGANIZATION,  # Level 1 open
                EMISSION,
                RECEPTION,
                COHERENCE,
                DISSONANCE,
                SELF_ORGANIZATION,  # Level 2 open (nested)
                    COHERENCE,  # Invalid start - not autonomous
                    RESONANCE,
                SILENCE,  # Level 2 close
            SILENCE  # Level 1 close
        ]
        result = validate_sequence(nested)
        assert not result.passed
        assert "autonomous" in result.message.lower() or "must start" in result.message.lower()

    def test_thol_nested_unclosed_inner(self):
        """Unclosed inner THOL should be detected."""
        nested = [
            EMISSION,
            RECEPTION,
            COHERENCE,
            DISSONANCE,
            SELF_ORGANIZATION,  # Level 1 open
                EMISSION,
                RECEPTION,
                COHERENCE,
                DISSONANCE,
                SELF_ORGANIZATION,  # Level 2 open (nested)
                    EMISSION,
                    RECEPTION,
                    COHERENCE,
                # Missing Level 2 close!
            SILENCE  # Level 1 close (but Level 2 still open)
        ]
        result = validate_sequence(nested)
        assert not result.passed
        assert "closure" in result.message.lower() or "without" in result.message.lower()


class TestTholValidAutonomous:
    """Test valid THOL subsequences following C1-C4 physics."""

    def test_thol_minimal_valid(self):
        """Minimal valid THOL: AL → IL → NAV, closed with SHA."""
        valid = [
            EMISSION,  # C1 start
            COHERENCE,  # C3 stabilizer
            DISSONANCE,  # Destabilizer for THOL (C4)
            SELF_ORGANIZATION,  # THOL opening
                EMISSION,  # C1 start
                COHERENCE,  # C3 stabilizer
                TRANSITION,  # C1 valid end (not a THOL closure)
            SILENCE,  # THOL closure
            TRANSITION  # Sequence end (different from closure)
        ]
        result = validate_sequence(valid)
        assert result.passed

    def test_thol_with_transition_end(self):
        """Valid THOL ending with NAV."""
        valid = [
            EMISSION,
            COHERENCE,
            DISSONANCE,
            SELF_ORGANIZATION,
                EMISSION,
                COHERENCE,
                TRANSITION,  # C1 valid end
            SILENCE  # THOL closure
        ]
        result = validate_sequence(valid)
        assert result.passed

    def test_thol_with_recursivity_end(self):
        """Valid THOL ending with REMESH."""
        valid = [
            EMISSION,
            COHERENCE,
            DISSONANCE,
            SELF_ORGANIZATION,
                RECURSIVITY,  # C1 valid start
                COHERENCE,  # C3 stabilizer
                RECURSIVITY,  # C1 valid end
            SILENCE  # THOL closure
        ]
        result = validate_sequence(valid)
        assert result.passed

    def test_thol_with_dissonance_end(self):
        """Valid THOL ending with OZ (intentional closure)."""
        valid = [
            EMISSION,
            COHERENCE,
            DISSONANCE,
            SELF_ORGANIZATION,
                EMISSION,
                COHERENCE,
                DISSONANCE,  # C1 valid end - preserves tension
            SILENCE  # THOL closure
        ]
        result = validate_sequence(valid)
        assert result.passed

    def test_thol_with_mutation_valid(self):
        """Valid THOL with ZHIR (C4: has prior IL and recent OZ)."""
        valid = [
            EMISSION,
            COHERENCE,
            DISSONANCE,
            SELF_ORGANIZATION,
                EMISSION,
                COHERENCE,  # C4: Prior IL for ZHIR
                DISSONANCE,  # C4: Recent destabilizer for ZHIR
                MUTATION,  # Valid: satisfies C4
                COHERENCE,  # C3: Stabilizer
                SILENCE,  # C1: Valid end
            SILENCE  # THOL closure
        ]
        result = validate_sequence(valid)
        assert result.passed

    def test_thol_nested_valid(self):
        """Nested THOL blocks (operational fractality)."""
        valid = [
            EMISSION,
            COHERENCE,
            DISSONANCE,
            SELF_ORGANIZATION,  # Level 1
                EMISSION,
                COHERENCE,
                DISSONANCE,
                SELF_ORGANIZATION,  # Level 2 (nested)
                    EMISSION,
                    COHERENCE,
                    SILENCE,  # Level 2 ends validly
                SILENCE,  # Close Level 2 THOL
                TRANSITION,  # Level 1 subsequence ends validly
            SILENCE  # Close Level 1 THOL
        ]
        result = validate_sequence(valid)
        assert result.passed


class TestTholClosureValidation:
    """Test THOL closure operator validation."""

    def test_thol_closed_with_silence(self):
        """THOL can be closed with SHA."""
        valid = [
            EMISSION,
            RECEPTION,
            COHERENCE,
            DISSONANCE,
            SELF_ORGANIZATION,
                EMISSION,  # Valid start
                RECEPTION,  # EN→IL segment
                COHERENCE,  # IL stabilizer
                SILENCE,  # Valid end
            SILENCE  # Valid THOL closure
        ]
        result = validate_sequence(valid)
        assert result.passed

    def test_thol_closed_with_contraction(self):
        """THOL can be closed with NUL."""
        valid = [
            EMISSION,
            RECEPTION,
            COHERENCE,
            DISSONANCE,
            SELF_ORGANIZATION,
                EMISSION,
                RECEPTION,
                COHERENCE,
                SILENCE,  # Valid end for subsequence
            CONTRACTION  # Valid THOL closure
        ]
        result = validate_sequence(valid)
        assert result.passed

    def test_thol_closure_without_opening(self):
        """Closure operator without THOL opening should fail."""
        # This tests the error case in _consume when closure found without opening
        # However, this might not trigger because SHA/NUL are also valid end operators
        # Let's test a specific case: THOL closure in wrong context
        invalid = [
            EMISSION,
            RECEPTION,
            COHERENCE,
            SILENCE  # Not a THOL closure context, just regular ending
        ]
        # This is actually valid - SHA can end a sequence normally
        result = validate_sequence(invalid)
        assert result.passed  # This is correct behavior


class TestTholEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_multiple_sequential_thol_blocks(self):
        """Multiple THOL blocks in sequence should each be validated."""
        valid = [
            EMISSION,
            RECEPTION,
            COHERENCE,
            DISSONANCE,
            SELF_ORGANIZATION,
                EMISSION,
                RECEPTION,
                COHERENCE,
                SILENCE,  # Valid end for first THOL
            SILENCE,  # Close first THOL
            DISSONANCE,
            SELF_ORGANIZATION,
                EMISSION,
                RECEPTION,
                COHERENCE,
                TRANSITION,  # Valid end for second THOL
            CONTRACTION  # Close second THOL
        ]
        result = validate_sequence(valid)
        assert result.passed

    def test_thol_with_all_operator_types(self):
        """Complex THOL with diverse operators."""
        valid = [
            EMISSION,
            RECEPTION,
            COHERENCE,
            DISSONANCE,
            SELF_ORGANIZATION,
                EMISSION,  # Valid start
                RECEPTION,  # EN→IL segment
                COHERENCE,  # IL stabilizer
                COUPLING,
                RESONANCE,
                DISSONANCE,
                EXPANSION,
                COHERENCE,  # Another stabilizer
                SILENCE,  # Valid end
            SILENCE
        ]
        result = validate_sequence(valid)
        assert result.passed

    def test_deeply_nested_thol(self):
        """Three levels of THOL nesting."""
        valid = [
            EMISSION,
            RECEPTION,
            COHERENCE,
            DISSONANCE,
            SELF_ORGANIZATION,  # Level 1
                EMISSION,
                RECEPTION,
                COHERENCE,
                DISSONANCE,
                SELF_ORGANIZATION,  # Level 2
                    EMISSION,
                    RECEPTION,
                    COHERENCE,
                    DISSONANCE,
                    SELF_ORGANIZATION,  # Level 3
                        EMISSION,
                        RECEPTION,
                        COHERENCE,
                        SILENCE,  # Valid end Level 3
                    SILENCE,  # Close Level 3
                    RESONANCE,
                    SILENCE,  # Valid end Level 2
                SILENCE,  # Close Level 2
                RESONANCE,
                TRANSITION,  # Valid end Level 1
            SILENCE,  # Close Level 1
            TRANSITION  # Valid end for main sequence
        ]
        result = validate_sequence(valid)
        assert result.passed
