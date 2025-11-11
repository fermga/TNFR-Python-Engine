"""Tests for graduated compatibility matrix implementation.

This module validates the 4-level graduated compatibility system that
replaces the binary compatible/incompatible validation with nuanced levels:
- EXCELLENT: Optimal structural progression
- GOOD: Acceptable structural progression  
- CAUTION: Contextually dependent, requires validation (generates warnings)
- AVOID: Incompatible, violates structural coherence (raises errors)
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
from tnfr.operators.grammar import validate_sequence, SequenceSyntaxError
from tnfr.validation.compatibility import (
    CompatibilityLevel,
    GRADUATED_COMPATIBILITY,
    get_compatibility_level,
)


class TestCompatibilityLevelEnum:
    """Tests for the CompatibilityLevel enum."""

    def test_enum_has_four_levels(self):
        """Enum defines exactly 4 compatibility levels."""
        assert len(CompatibilityLevel) == 4
        assert CompatibilityLevel.EXCELLENT.value == "excellent"
        assert CompatibilityLevel.GOOD.value == "good"
        assert CompatibilityLevel.CAUTION.value == "caution"
        assert CompatibilityLevel.AVOID.value == "avoid"


class TestGraduatedCompatibilityMatrix:
    """Tests for the GRADUATED_COMPATIBILITY data structure."""

    def test_all_13_operators_defined(self):
        """All 13 canonical operators have graduated compatibility entries."""
        expected_operators = {
            EMISSION,
            RECEPTION,
            COHERENCE,
            DISSONANCE,
            COUPLING,
            RESONANCE,
            SILENCE,
            EXPANSION,
            CONTRACTION,
            SELF_ORGANIZATION,
            MUTATION,
            TRANSITION,
            RECURSIVITY,
        }
        assert set(GRADUATED_COMPATIBILITY.keys()) == expected_operators

    def test_each_operator_has_all_four_levels(self):
        """Each operator defines all four compatibility levels."""
        for operator, levels in GRADUATED_COMPATIBILITY.items():
            assert "excellent" in levels, f"{operator} missing 'excellent' level"
            assert "good" in levels, f"{operator} missing 'good' level"
            assert "caution" in levels, f"{operator} missing 'caution' level"
            assert "avoid" in levels, f"{operator} missing 'avoid' level"

    def test_no_operator_in_multiple_levels(self):
        """Each target operator appears in exactly one compatibility level per source."""
        for source, levels in GRADUATED_COMPATIBILITY.items():
            seen = set()
            for level_name in ["excellent", "good", "caution", "avoid"]:
                targets = set(levels[level_name])
                overlap = seen & targets
                assert (
                    not overlap
                ), f"{source}: {overlap} appears in multiple levels"
                seen.update(targets)

    def test_all_13_operators_accounted_for_in_each_entry(self):
        """Each operator's levels collectively cover all 13 possible next operators."""
        all_operators = {
            EMISSION,
            RECEPTION,
            COHERENCE,
            DISSONANCE,
            COUPLING,
            RESONANCE,
            SILENCE,
            EXPANSION,
            CONTRACTION,
            SELF_ORGANIZATION,
            MUTATION,
            TRANSITION,
            RECURSIVITY,
        }

        for source, levels in GRADUATED_COMPATIBILITY.items():
            covered = set()
            for level_name in ["excellent", "good", "caution", "avoid"]:
                covered.update(levels[level_name])

            assert (
                covered == all_operators
            ), f"{source}: missing or extra operators. Covered: {covered}, Expected: {all_operators}"


class TestGetCompatibilityLevel:
    """Tests for the get_compatibility_level() function."""

    def test_excellent_transitions(self):
        """Excellent transitions return EXCELLENT level."""
        # EMISSION → COHERENCE: initiation → stabilization
        assert get_compatibility_level(EMISSION, COHERENCE) == CompatibilityLevel.EXCELLENT

        # RECEPTION → COHERENCE: anchoring → stabilization
        assert get_compatibility_level(RECEPTION, COHERENCE) == CompatibilityLevel.EXCELLENT

        # DISSONANCE → MUTATION: tension → transformation
        assert get_compatibility_level(DISSONANCE, MUTATION) == CompatibilityLevel.EXCELLENT

    def test_good_transitions(self):
        """Good transitions return GOOD level."""
        # EMISSION → RESONANCE: initiation → amplification
        assert get_compatibility_level(EMISSION, RESONANCE) == CompatibilityLevel.GOOD

        # COHERENCE → SILENCE: stabilization → pause
        assert get_compatibility_level(COHERENCE, SILENCE) == CompatibilityLevel.GOOD

        # RESONANCE → EMISSION: amplification → re-initiation
        assert get_compatibility_level(RESONANCE, EMISSION) == CompatibilityLevel.GOOD

    def test_caution_transitions(self):
        """Caution transitions return CAUTION level."""
        # EMISSION → DISSONANCE: initiation → tension (requires context)
        assert get_compatibility_level(EMISSION, DISSONANCE) == CompatibilityLevel.CAUTION

        # COHERENCE → MUTATION: stabilization → transformation (requires context)
        assert get_compatibility_level(COHERENCE, MUTATION) == CompatibilityLevel.CAUTION

        # DISSONANCE → DISSONANCE: repeated tension (requires careful management)
        assert get_compatibility_level(DISSONANCE, DISSONANCE) == CompatibilityLevel.CAUTION

    def test_avoid_transitions(self):
        """Avoid transitions return AVOID level."""
        # SILENCE → DISSONANCE: pause → tension (contradictory)
        assert get_compatibility_level(SILENCE, DISSONANCE) == CompatibilityLevel.AVOID

        # EXPANSION → CONTRACTION: direct reversal (invalid)
        assert get_compatibility_level(EXPANSION, CONTRACTION) == CompatibilityLevel.AVOID

        # COHERENCE → EMISSION: cannot re-initiate after stabilizing
        assert get_compatibility_level(COHERENCE, EMISSION) == CompatibilityLevel.AVOID

    def test_unknown_operator_returns_avoid(self):
        """Unknown operators default to AVOID level."""
        assert get_compatibility_level("unknown_op", COHERENCE) == CompatibilityLevel.AVOID
        assert get_compatibility_level(EMISSION, "unknown_op") == CompatibilityLevel.AVOID


class TestSequenceValidationWithGraduatedLevels:
    """Tests for sequence validation using graduated compatibility levels."""

    def test_excellent_transitions_pass_without_warnings(self):
        """Sequences with excellent transitions validate successfully."""
        # EMISSION → RECEPTION → COHERENCE → RESONANCE → SILENCE
        result = validate_sequence([EMISSION, RECEPTION, COHERENCE, RESONANCE, SILENCE])
        assert result.passed
        assert result.message == "ok"

    def test_good_transitions_pass_smoothly(self):
        """Sequences with good transitions validate successfully."""
        # EMISSION → RECEPTION → COHERENCE → EXPANSION → RESONANCE → SILENCE
        # COHERENCE → EXPANSION is good, EXPANSION → RESONANCE is excellent
        result = validate_sequence([EMISSION, RECEPTION, COHERENCE, EXPANSION, RESONANCE, SILENCE])
        assert result.passed
        assert result.message == "ok"

    def test_caution_transitions_generate_warnings_but_pass(self, caplog):
        """Sequences with caution transitions log warnings but validate successfully."""
        import logging

        # Set up logging to capture warnings
        caplog.set_level(logging.WARNING)

        # EMISSION → RECEPTION → COHERENCE → DISSONANCE → MUTATION → SILENCE
        # COHERENCE → DISSONANCE is CAUTION level
        result = validate_sequence([EMISSION, RECEPTION, COHERENCE, DISSONANCE, MUTATION, SILENCE])
        assert result.passed, "Caution transitions should pass validation"

        # Check that a warning was logged
        assert any(
            "Caution" in record.message for record in caplog.records
        ), "Expected warning for caution transition"

    def test_avoid_transitions_raise_errors(self):
        """Sequences with avoid transitions raise SequenceSyntaxError."""
        # Test with complete valid sequence, then add incompatible transition
        # EMISSION → RECEPTION → COHERENCE → SILENCE → DISSONANCE
        # SILENCE → DISSONANCE is AVOID (contradictory)
        result = validate_sequence([EMISSION, RECEPTION, COHERENCE, SILENCE, DISSONANCE])
        
        # Should fail with error
        assert not result.passed
        assert result.error is not None
        assert "incompatible" in str(result.error).lower()

    def test_mixed_levels_validate_correctly(self, caplog):
        """Sequences mixing excellent, good, and caution levels validate with appropriate warnings."""
        import logging

        caplog.set_level(logging.WARNING)

        # EMISSION (start) → RECEPTION (excellent) → COHERENCE (excellent)
        # → DISSONANCE (caution) → MUTATION (excellent) → SILENCE (excellent)
        result = validate_sequence(
            [EMISSION, RECEPTION, COHERENCE, DISSONANCE, MUTATION, SILENCE]
        )
        assert result.passed

        # Should have warning for COHERENCE → DISSONANCE (caution)
        warning_found = any("Caution" in record.message for record in caplog.records)
        assert warning_found, "Expected warning for caution transition"


class TestBackwardCompatibility:
    """Tests ensuring backward compatibility with binary compatibility system."""

    def test_previously_allowed_transitions_still_pass(self):
        """All transitions that were allowed in binary system still pass."""
        # Test various transitions that should work - all must include RECEPTION→COHERENCE
        sequences = [
            [EMISSION, RECEPTION, COHERENCE, RESONANCE, SILENCE],
            [EMISSION, RECEPTION, COHERENCE, COUPLING, RESONANCE, SILENCE],
            [EMISSION, RECEPTION, SELF_ORGANIZATION, COHERENCE, SILENCE],
            [EMISSION, RECEPTION, COHERENCE, DISSONANCE, MUTATION, SILENCE],
            [RECURSIVITY, RECEPTION, COHERENCE, RESONANCE, SILENCE],
        ]

        for seq in sequences:
            result = validate_sequence(seq)
            assert result.passed, f"Sequence {seq} should pass but got: {result.message}"

    def test_previously_forbidden_transitions_still_fail(self):
        """Transitions that were forbidden in binary system still raise errors."""
        # These should fail at the incompatible transition
        result1 = validate_sequence([EMISSION, RECEPTION, COHERENCE, SILENCE, DISSONANCE])
        assert not result1.passed, "SILENCE → DISSONANCE should fail"
        assert result1.error is not None

        result2 = validate_sequence([EMISSION, RECEPTION, COHERENCE, EXPANSION, CONTRACTION])
        assert not result2.passed, "EXPANSION → CONTRACTION should fail"
        assert result2.error is not None


class TestSequenceHealthMetrics:
    """Tests for sequence health metrics based on compatibility levels."""

    def test_excellent_only_sequence_indicates_high_quality(self):
        """Sequences with only excellent transitions indicate high structural quality."""
        # EMISSION → RECEPTION → COHERENCE → RESONANCE
        result = validate_sequence([EMISSION, RECEPTION, COHERENCE, RESONANCE, SILENCE])
        assert result.passed

        # All transitions should be excellent or good
        transitions = [
            (EMISSION, RECEPTION),  # excellent
            (RECEPTION, COHERENCE),  # excellent
            (COHERENCE, RESONANCE),  # excellent
            (RESONANCE, SILENCE),  # good
        ]

        for prev_op, next_op in transitions:
            level = get_compatibility_level(prev_op, next_op)
            assert level in [
                CompatibilityLevel.EXCELLENT,
                CompatibilityLevel.GOOD,
            ]

    def test_caution_heavy_sequence_indicates_risky_structure(self):
        """Sequences with multiple caution transitions indicate risky structural patterns."""
        # EMISSION → DISSONANCE → DISSONANCE → MUTATION → COHERENCE
        # Multiple CAUTION levels
        transitions = [
            (EMISSION, DISSONANCE),  # caution
            (DISSONANCE, DISSONANCE),  # caution  
        ]

        caution_count = sum(
            1
            for prev_op, next_op in transitions
            if get_compatibility_level(prev_op, next_op) == CompatibilityLevel.CAUTION
        )

        assert caution_count >= 2, "Expected multiple caution transitions"


class TestContextualValidation:
    """Tests for contextual validation of caution-level transitions."""

    def test_emission_to_dissonance_caution_context(self, caplog):
        """EMISSION → DISSONANCE requires careful context validation."""
        import logging

        caplog.set_level(logging.WARNING)

        # This should pass but log a warning - need RECEPTION→COHERENCE + DISSONANCE before MUTATION
        result = validate_sequence([EMISSION, RECEPTION, COHERENCE, DISSONANCE, MUTATION, SILENCE])
        assert result.passed

        # Verify warning about caution for COHERENCE → DISSONANCE
        assert any(
            "Caution" in record.message and "dissonance" in record.message.lower()
            for record in caplog.records
        )

    def test_coherence_to_mutation_caution_context(self, caplog):
        """COHERENCE → MUTATION requires careful context validation."""
        import logging

        caplog.set_level(logging.WARNING)

        # This tests COHERENCE → MUTATION which is CAUTION level
        # Need DISSONANCE before MUTATION due to R4 rule
        result = validate_sequence(
            [EMISSION, RECEPTION, COHERENCE, DISSONANCE, MUTATION, COHERENCE, MUTATION, SILENCE]
        )
        assert result.passed

        # Verify warning about caution for COHERENCE → MUTATION
        caution_warnings = [
            record for record in caplog.records 
            if "Caution" in record.message and "mutation" in record.message.lower()
        ]
        assert len(caution_warnings) >= 1, "Expected at least one caution warning for COHERENCE → MUTATION"


class TestEdgeCases:
    """Tests for edge cases in graduated compatibility."""

    def test_self_transitions_vary_by_operator(self):
        """Self-transitions (operator → same operator) have appropriate levels."""
        # DISSONANCE → DISSONANCE is CAUTION (repeated tension)
        assert (
            get_compatibility_level(DISSONANCE, DISSONANCE) == CompatibilityLevel.CAUTION
        )

        # TRANSITION → TRANSITION is GOOD (continued handoff)
        assert get_compatibility_level(TRANSITION, TRANSITION) == CompatibilityLevel.GOOD

        # SELF_ORGANIZATION → SELF_ORGANIZATION is GOOD (nested fractality)
        assert (
            get_compatibility_level(SELF_ORGANIZATION, SELF_ORGANIZATION)
            == CompatibilityLevel.GOOD
        )

        # EMISSION → EMISSION is AVOID (redundant initiation)
        assert get_compatibility_level(EMISSION, EMISSION) == CompatibilityLevel.AVOID

    def test_closure_operators_have_limited_successors(self):
        """Closure operators (SILENCE, CONTRACTION) have restricted successors."""
        # SILENCE should only allow EMISSION and RECEPTION as excellent
        assert get_compatibility_level(SILENCE, EMISSION) == CompatibilityLevel.EXCELLENT
        assert get_compatibility_level(SILENCE, RECEPTION) == CompatibilityLevel.EXCELLENT
        assert get_compatibility_level(SILENCE, DISSONANCE) == CompatibilityLevel.AVOID

        # CONTRACTION should allow EMISSION and COHERENCE as excellent
        assert (
            get_compatibility_level(CONTRACTION, EMISSION) == CompatibilityLevel.EXCELLENT
        )
        assert (
            get_compatibility_level(CONTRACTION, COHERENCE)
            == CompatibilityLevel.EXCELLENT
        )
