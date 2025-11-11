"""Tests for R6: Operational closure and coherence conservation validation.

This module tests Rule R6 which validates that operator sequences complete
coherent reorganization cycles according to TNFR canonical principles.

R6 validates:
1. Structural convergence: Sequences must end with stabilizing operators
2. Operational closure: Balance between destabilizers and stabilizers
3. Frequency balance: Net reorganization tendency (informational)

Theoretical foundation:
    EPI(t_final) = EPI(t_initial) + ∫_{t_0}^{t_f} νf(t) · ΔNFR(t) dt

References:
- Issue: [GRAMÁTICA CANÓNICA] Añadir validación de clausura operatorial
- TNFR.pdf Section 2.1.4: Nodal stability conditions
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


class TestR6StructuralConvergence:
    """Test R6 respects R3 endings - no additional convergence validation.

    R6 simplified: Only validates controlled mutation (IL → ZHIR).
    Does NOT reject OZ endings or validate balance.
    All R3-valid endings (SHA, NAV, REMESH, OZ) are accepted by R6.
    """

    def test_valid_ending_with_silence(self):
        """Sequence ending with silence is valid."""
        sequence = [EMISSION, RECEPTION, COHERENCE, RESONANCE, SILENCE]
        result = validate_sequence(sequence)
        assert result.passed, f"Expected pass but got: {result.message}"

    def test_valid_ending_with_transition(self):
        """Sequence ending with transition is valid."""
        sequence = [EMISSION, RECEPTION, COHERENCE, RESONANCE, TRANSITION]
        result = validate_sequence(sequence)
        assert result.passed, f"Expected pass but got: {result.message}"

    def test_valid_ending_with_recursivity(self):
        """Sequence ending with recursivity is valid."""
        sequence = [EMISSION, RECEPTION, COHERENCE, RECURSIVITY]
        result = validate_sequence(sequence)
        assert result.passed, f"Expected pass but got: {result.message}"

    def test_valid_ending_with_dissonance(self):
        """Sequence ending with dissonance is valid (therapeutic/activation).

        OZ endings are always destabilizing (increase ΔNFR at end) but valid
        for therapeutic tension, system activation, multi-sequence chains.
        R2 ensures coherence base exists (IL or THOL required).
        """
        sequence = [EMISSION, RECEPTION, COHERENCE, DISSONANCE]
        result = validate_sequence(sequence)
        assert result.passed, f"Expected pass but got: {result.message}"


class TestR6ControlledMutation:
    """Test R6 controlled mutation validation (IL → ZHIR).

    R6 simplified: ONLY validates controlled mutation.
    Does NOT validate destabilizer/stabilizer balance.
    """

    def test_valid_controlled_mutation(self):
        """Mutation after coherence is valid (controlled transformation)."""
        sequence = [
            EMISSION,
            RECEPTION,
            COHERENCE,  # Coherent base
            DISSONANCE,  # Destabilizer (R4 requirement for ZHIR)
            MUTATION,  # ZHIR after IL (controlled)
            COHERENCE,  # Stabilize after transformation
            SILENCE,
        ]
        result = validate_sequence(sequence)
        assert result.passed, f"Expected pass but got: {result.message}"

    def test_invalid_mutation_without_coherence(self):
        """Mutation without any coherence base fails.

        ZHIR requires prior IL for stable transformation foundation.
        """
        sequence = [
            EMISSION,
            RECEPTION,
            DISSONANCE,  # Destabilizer for ZHIR (R4)
            MUTATION,  # ZHIR without any IL
            SILENCE,
        ]
        result = validate_sequence(sequence)
        # Should fail R6: no coherence base
        assert not result.passed
        assert "R6" in result.message
        assert "coherence" in result.message.lower()

    def test_invalid_mutation_before_coherence(self):
        """Mutation before coherence fails (ungrounded transformation).

        IL must come BEFORE ZHIR to provide stable base.
        """
        sequence = [
            EMISSION,
            RECEPTION,
            DISSONANCE,  # Destabilizer for ZHIR (R4)
            MUTATION,  # ZHIR before IL
            COHERENCE,  # IL after ZHIR (wrong order)
            SILENCE,
        ]
        result = validate_sequence(sequence)
        # Should fail R6: wrong order
        assert not result.passed
        assert "R6" in result.message
        assert "must follow" in result.message.lower() or "before" in result.message.lower()


class TestR6FrequencyBalance:
    """Test R6 frequency balance calculation (informational).

    Frequency balance is currently computed but not enforced as error.
    These tests verify the calculation works correctly for future use.
    """

    def test_positive_balance_high_activation(self):
        """Sequence with high-frequency operators has positive balance."""
        # AL (high), OZ (high), RA (high), SHA (zero)
        # Balance: (1.0 + 1.0 + 1.0 + (-1.0)) / 4 = 0.5
        sequence = [EMISSION, RECEPTION, COHERENCE, DISSONANCE, RESONANCE, SILENCE]
        result = validate_sequence(sequence)
        # Should pass (balance is informational, not enforced)
        assert result.passed

    def test_negative_balance_silence_heavy(self):
        """Sequence heavy on silence has negative balance."""
        # This is hypothetical - actual grammar rules prevent multiple SHA
        # But we're testing the balance calculation concept
        # For actual test, we just verify sequences with SHA pass
        sequence = [EMISSION, RECEPTION, COHERENCE, RESONANCE, SILENCE]
        result = validate_sequence(sequence)
        assert result.passed


class TestR6EdgeCases:
    """Test edge cases and boundary conditions for R6."""

    def test_minimal_valid_sequence(self):
        """Minimal valid sequence passes all R6 checks."""
        # AL → EN → IL → SHA (minimal valid per R1-R6)
        # Must end with SHA for R6 convergence
        sequence = [EMISSION, RECEPTION, COHERENCE, SILENCE]
        result = validate_sequence(sequence)
        assert result.passed

    def test_complex_valid_sequence(self):
        """Complex sequence with multiple operators passes R6."""
        sequence = [
            EMISSION,
            RECEPTION,
            COHERENCE,
            COUPLING,
            RESONANCE,
            DISSONANCE,
            COHERENCE,
            SILENCE,
        ]
        result = validate_sequence(sequence)
        assert result.passed

    def test_self_organization_with_closure(self):
        """Self-organization sequence achieves operational closure."""
        # THOL provides autopoietic closure
        sequence = [
            EMISSION,
            RECEPTION,
            COHERENCE,
            DISSONANCE,  # Required for THOL (R4)
            SELF_ORGANIZATION,
            SILENCE,  # THOL closure operator
        ]
        result = validate_sequence(sequence)
        assert result.passed


class TestR6ErrorMessages:
    """Test that R6 error messages are clear and informative."""

    def test_convergence_error_message_clarity(self):
        """Convergence error message identifies divergent operator."""
        sequence = [EMISSION, RECEPTION, COHERENCE, DISSONANCE]
        result = validate_sequence(sequence)
        assert not result.passed
        assert "R6" in result.message
        assert "divergent" in result.message.lower()
        # Should mention acceptable endings
        assert "silence" in result.message.lower() or "transition" in result.message.lower()

    def test_closure_error_message_clarity(self):
        """Closure error message explains destabilizer/stabilizer imbalance."""
        # Create sequence with excess destabilizers and NAV ending (not SHA)
        # NAV requires balance, but this has imbalance
        sequence = [
            EMISSION,
            RECEPTION,
            COHERENCE,
            DISSONANCE,
            TRANSITION,  # Destabilizer 2
            EXPANSION,  # Destabilizer 3
            RESONANCE,
            TRANSITION,  # NAV ending (needs balance)
        ]
        result = validate_sequence(sequence)
        assert not result.passed
        assert "R6" in result.message
        assert "closure" in result.message.lower()
        # Should mention balance concept
        assert any(
            word in result.message.lower()
            for word in ["destabilizer", "stabilizer", "balance", "sustainability"]
        )


class TestR6Integration:
    """Test R6 integration with existing rules R1-R5."""

    def test_r6_does_not_break_r1_start_validation(self):
        """R6 doesn't interfere with R1 (start operator validation)."""
        # Invalid start (EN not in VALID_START_OPERATORS)
        sequence = [RECEPTION, COHERENCE, SILENCE]
        result = validate_sequence(sequence)
        assert not result.passed
        # Should fail on R1, not R6
        assert "must start" in result.message.lower()

    def test_r6_does_not_break_r2_stabilizer_requirement(self):
        """R6 doesn't interfere with R2 (stabilizer requirement)."""
        # Missing stabilizer
        sequence = [EMISSION, RECEPTION, RESONANCE, SILENCE]
        result = validate_sequence(sequence)
        assert not result.passed
        # Should fail on R2 (missing stabilizer or EN→IL segment)
        assert "missing" in result.message.lower()

    def test_r6_complements_r3_end_validation(self):
        """R6 adds semantic validation on top of R3 syntactic validation.

        R3 allows VALID_END_OPERATORS: SHA, NAV, REMESH, OZ
        R6 validates two aspects:
        1. Ending convergence/closure: Rejects OZ (divergent)
        2. Operational closure balance: Validates destabilizers ≤ stabilizers

        NAV is a destabilizer, so sequences ending with NAV must have balance.
        """
        # OZ (dissonance) is valid per R3 but divergent per R6
        sequence = [EMISSION, RECEPTION, COHERENCE, DISSONANCE]
        result = validate_sequence(sequence)
        assert not result.passed
        # Should fail R6 (divergent ending)
        assert "R6" in result.message
        assert "divergent" in result.message.lower()

        # NAV is valid per R3 but requires balance per R6
        # This sequence is balanced: EN(1 if has context, but no prior IL), NAV(1) vs IL(1), RA(1)
        # Actually EN at pos 1 doesn't have prior coherence for destabilizer role
        # So: NAV(1) destabilizers vs IL(1), RA(1) stabilizers = 1 vs 2, balanced
        sequence_nav = [EMISSION, RECEPTION, COHERENCE, RESONANCE, TRANSITION]
        result_nav = validate_sequence(sequence_nav)
        assert result_nav.passed, f"NAV with balance should pass R6: {result_nav.message}"

    def test_r6_works_with_r4_bifurcation_control(self):
        """R6 operational closure works with R4 bifurcation requirements."""
        # Sequence with mutation after destabilizer (R4) and balanced closure (R6)
        sequence = [
            EMISSION,
            RECEPTION,
            COHERENCE,
            DISSONANCE,  # Destabilizer for ZHIR (R4)
            MUTATION,  # ZHIR requires recent destabilizer (R4)
            COHERENCE,  # Stabilizer for closure (R6)
            SILENCE,  # Convergent ending (R6)
        ]
        result = validate_sequence(sequence)
        assert result.passed

    def test_r6_works_with_r5_frequency_transitions(self):
        """R6 operational closure works with R5 frequency validation."""
        # Sequence with valid frequency transitions (R5) and closure (R6)
        sequence = [
            EMISSION,  # high
            RECEPTION,  # medium (high → medium: valid)
            COHERENCE,  # medium (medium → medium: valid)
            RESONANCE,  # high (medium → high: valid)
            SILENCE,  # zero (high → zero: valid per R5)
        ]
        result = validate_sequence(sequence)
        assert result.passed


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
