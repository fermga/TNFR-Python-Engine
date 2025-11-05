"""Tests for semantic sequence validation."""

import pytest

from tnfr.validation import InvariantSeverity, SequenceSemanticValidator


class TestSequenceSemanticValidator:
    """Test semantic validation of operator sequences."""

    def test_valid_sequence_no_violations(self):
        """Test that a well-formed sequence has no violations."""
        sequence = ["emission", "reception", "coherence", "resonance", "silence"]

        validator = SequenceSemanticValidator()
        violations = validator.validate_semantic_sequence(sequence)

        # Should have no errors (may have warnings depending on rules)
        error_violations = [
            v
            for v in violations
            if v.severity
            in [InvariantSeverity.ERROR, InvariantSeverity.CRITICAL]
        ]
        assert len(error_violations) == 0

    def test_mutation_without_stabilization(self):
        """Test that mutation without stabilization triggers warning."""
        sequence = ["emission", "mutation", "dissonance"]

        validator = SequenceSemanticValidator()
        violations = validator.validate_semantic_sequence(sequence)

        warning_violations = [v for v in violations if v.severity == InvariantSeverity.WARNING]
        assert any("Mutation" in v.description for v in warning_violations)
        assert any("stabilization" in v.description for v in warning_violations)

    def test_mutation_with_stabilization_allowed(self):
        """Test that mutation followed by stabilization is allowed."""
        sequence = ["emission", "mutation", "coherence"]

        validator = SequenceSemanticValidator()
        violations = validator.validate_semantic_sequence(sequence)

        # Should not have mutation-specific warnings
        mutation_warnings = [
            v
            for v in violations
            if v.severity == InvariantSeverity.WARNING
            and "Mutation" in v.description
            and "stabilization" in v.description
        ]
        assert len(mutation_warnings) == 0

    def test_excessive_dissonance(self):
        """Test that excessive consecutive dissonance triggers error."""
        sequence = ["emission", "dissonance", "dissonance", "dissonance"]

        validator = SequenceSemanticValidator()
        violations = validator.validate_semantic_sequence(sequence)

        error_violations = [v for v in violations if v.severity == InvariantSeverity.ERROR]
        assert any("Excessive consecutive dissonance" in v.description for v in error_violations)

    def test_two_dissonance_allowed(self):
        """Test that two consecutive dissonances are allowed."""
        sequence = ["emission", "dissonance", "dissonance", "coherence"]

        validator = SequenceSemanticValidator()
        violations = validator.validate_semantic_sequence(sequence)

        # Should not trigger excessive dissonance error
        dissonance_errors = [
            v
            for v in violations
            if v.severity == InvariantSeverity.ERROR
            and "Excessive consecutive dissonance" in v.description
        ]
        assert len(dissonance_errors) == 0

    def test_transition_without_perturbation(self):
        """Test that transition without prior perturbation triggers error."""
        sequence = ["emission", "reception", "transition"]

        validator = SequenceSemanticValidator()
        violations = validator.validate_semantic_sequence(sequence)

        error_violations = [v for v in violations if v.severity == InvariantSeverity.ERROR]
        assert any("Transition" in v.description for v in error_violations)
        assert any("perturbation" in v.description for v in error_violations)

    def test_transition_with_dissonance_allowed(self):
        """Test that transition after dissonance is allowed."""
        sequence = ["emission", "dissonance", "transition"]

        validator = SequenceSemanticValidator()
        violations = validator.validate_semantic_sequence(sequence)

        # Should not have transition-specific errors
        transition_errors = [
            v
            for v in violations
            if v.severity == InvariantSeverity.ERROR
            and "Transition" in v.description
            and "perturbation" in v.description
        ]
        assert len(transition_errors) == 0

    def test_transition_with_mutation_allowed(self):
        """Test that transition after mutation is allowed."""
        sequence = ["emission", "mutation", "transition"]

        validator = SequenceSemanticValidator()
        violations = validator.validate_semantic_sequence(sequence)

        # Should not have transition-specific errors
        transition_errors = [
            v
            for v in violations
            if v.severity == InvariantSeverity.ERROR
            and "Transition" in v.description
            and "perturbation" in v.description
        ]
        assert len(transition_errors) == 0

    def test_resonance_without_coupling_warning(self):
        """Test that resonance without prior coupling triggers warning."""
        sequence = ["coherence", "resonance"]

        validator = SequenceSemanticValidator()
        violations = validator.validate_semantic_sequence(sequence)

        warning_violations = [v for v in violations if v.severity == InvariantSeverity.WARNING]
        assert any("Resonance" in v.description for v in warning_violations)

    def test_resonance_with_coupling_allowed(self):
        """Test that resonance after coupling is allowed."""
        sequence = ["emission", "coupling", "resonance"]

        validator = SequenceSemanticValidator()
        violations = validator.validate_semantic_sequence(sequence)

        # Should not have resonance-specific warnings
        resonance_warnings = [
            v
            for v in violations
            if v.severity == InvariantSeverity.WARNING
            and "Resonance" in v.description
            and "coupling" in v.description
        ]
        assert len(resonance_warnings) == 0

    def test_empty_sequence(self):
        """Test that empty sequence has no violations."""
        sequence = []

        validator = SequenceSemanticValidator()
        violations = validator.validate_semantic_sequence(sequence)

        assert len(violations) == 0

    def test_single_operator_sequence(self):
        """Test single operator sequences."""
        # Single emission should be fine
        sequence = ["emission"]

        validator = SequenceSemanticValidator()
        violations = validator.validate_semantic_sequence(sequence)

        # May have warnings but no errors
        error_violations = [
            v
            for v in violations
            if v.severity
            in [InvariantSeverity.ERROR, InvariantSeverity.CRITICAL]
        ]
        assert len(error_violations) == 0

    def test_multiple_patterns_in_sequence(self):
        """Test detection of multiple semantic issues in one sequence."""
        sequence = [
            "emission",
            "dissonance",
            "dissonance",
            "dissonance",
            "mutation",
            "transition",
        ]

        validator = SequenceSemanticValidator()
        violations = validator.validate_semantic_sequence(sequence)

        # Should have multiple violations
        assert len(violations) >= 2

        # Should have excessive dissonance error
        assert any("Excessive consecutive dissonance" in v.description for v in violations)

        # Should have mutation without stabilization warning
        # (transition doesn't count as stabilization)
        mutation_warnings = [
            v
            for v in violations
            if "Mutation" in v.description and "stabilization" in v.description
        ]
        assert len(mutation_warnings) > 0

    def test_complex_valid_sequence(self):
        """Test a complex but valid sequence."""
        sequence = [
            "emission",
            "reception",
            "coupling",
            "resonance",
            "coherence",
            "mutation",
            "silence",
            "dissonance",
            "transition",
        ]

        validator = SequenceSemanticValidator()
        violations = validator.validate_semantic_sequence(sequence)

        # Should have no errors
        error_violations = [
            v
            for v in violations
            if v.severity
            in [InvariantSeverity.ERROR, InvariantSeverity.CRITICAL]
        ]
        assert len(error_violations) == 0
