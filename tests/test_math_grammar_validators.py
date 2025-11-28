"""
Tests for TNFR Mathematical Grammar Validators.

Validates:
- U2 Convergence analysis for glyph sequences.
- U4 Bifurcation risk analysis for glyph sequences.
"""

import pytest
from tnfr.types import Glyph
from tnfr.math import grammar_validators


class TestU2ConvergenceValidator:
    """Test U2 CONVERGENCE & BOUNDEDNESS validator."""

    def test_divergent_sequence(self):
        """Sequence with unhandled destabilizers should be divergent."""
        # OZ is a destabilizer
        sequence = [Glyph.AL, Glyph.OZ, Glyph.RA]
        converges, growth_rate, explanation = \
            grammar_validators.verify_convergence_for_sequence(sequence)
        
        assert not converges
        assert growth_rate > 0
        assert "VIOLATION" in explanation

    def test_convergent_sequence(self):
        """Sequence with stabilizers balancing destabilizers should converge."""
        # OZ (destabilizer) is balanced by IL (stabilizer)
        sequence = [Glyph.AL, Glyph.OZ, Glyph.IL, Glyph.RA]
        converges, growth_rate, explanation = \
            grammar_validators.verify_convergence_for_sequence(sequence)
            
        assert converges
        assert growth_rate <= 0
        assert "SATISFIED" in explanation

    def test_neutral_sequence(self):
        """Sequence with no stabilizers or destabilizers should be neutral."""
        sequence = [Glyph.AL, Glyph.EN, Glyph.RA]
        converges, growth_rate, _ = \
            grammar_validators.verify_convergence_for_sequence(sequence)
        
        assert converges
        assert growth_rate == 0.0

    def test_multiple_destabilizers(self):
        """Multiple destabilizers should require multiple stabilizers."""
        # Two destabilizers (OZ, VAL) and one stabilizer (IL)
        sequence = [Glyph.OZ, Glyph.VAL, Glyph.IL]
        converges, _, _ = \
            grammar_validators.verify_convergence_for_sequence(sequence)
        
        assert not converges # Net destabilizing

        # Add another stabilizer
        sequence.append(Glyph.THOL)
        converges, _, _ = \
            grammar_validators.verify_convergence_for_sequence(sequence)
        assert converges


class TestU4BifurcationValidator:
    """Test U4 BIFURCATION DYNAMICS validator."""

    def test_unsafe_sequence_no_handler(self):
        """A trigger without a handler should be unsafe."""
        # OZ is a trigger, but UM is not a handler
        sequence = [Glyph.EN, Glyph.OZ, Glyph.UM]
        is_safe, _, explanation = \
            grammar_validators.verify_bifurcation_risk_for_sequence(sequence)
            
        assert not is_safe
        assert "VIOLATION" in explanation

    def test_safe_sequence_with_handler(self):
        """A trigger followed by a handler should be safe."""
        # OZ (trigger) is followed by THOL (handler)
        sequence = [Glyph.EN, Glyph.OZ, Glyph.THOL, Glyph.UM]
        is_safe, _, explanation = \
            grammar_validators.verify_bifurcation_risk_for_sequence(sequence)
            
        assert is_safe
        assert "SATISFIED" in explanation

    def test_handler_outside_window(self):
        """A handler too far from the trigger should be unsafe."""
        # IL (handler) is outside the 3-operator window of OZ (trigger)
        sequence = [Glyph.OZ, Glyph.RA, Glyph.UM, Glyph.EN, Glyph.IL]
        is_safe, _, _ = \
            grammar_validators.verify_bifurcation_risk_for_sequence(sequence)
            
        assert not is_safe

    def test_multiple_triggers_and_handlers(self):
        """Complex sequences should be evaluated correctly."""
        # Unsafe: two triggers, one handler
        sequence1 = [Glyph.OZ, Glyph.ZHIR, Glyph.IL]
        is_safe1, _, _ = \
            grammar_validators.verify_bifurcation_risk_for_sequence(sequence1)
        assert not is_safe1

        # Safe: each trigger has a handler
        sequence2 = [Glyph.OZ, Glyph.IL, Glyph.ZHIR, Glyph.THOL]
        is_safe2, _, _ = \
            grammar_validators.verify_bifurcation_risk_for_sequence(sequence2)
        assert is_safe2
        
    def test_empty_and_no_trigger_sequences(self):
        """Sequences with no triggers should always be safe."""
    assert grammar_validators.verify_bifurcation_risk_for_sequence([])[0]
    sequence = [Glyph.AL, Glyph.EN, Glyph.UM, Glyph.RA]
    assert grammar_validators.verify_bifurcation_risk_for_sequence(sequence)[0]
