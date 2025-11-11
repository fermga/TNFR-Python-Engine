"""Integration tests for TNFR validation with structural operators.

This module tests the integration of the new validation system with
the structural operator execution in run_sequence.
"""

import math

import networkx as nx
import pytest

from tnfr.constants import EPI_PRIMARY, THETA_PRIMARY, VF_PRIMARY
from tnfr.structural import (
    Coherence,
    Dissonance,
    Emission,
    Mutation,
    Reception,
    Recursivity,
    Resonance,
    Silence,
    Transition,
    create_nfr,
    run_sequence,
)
from tnfr.validation import (
    InvariantSeverity,
    TNFRValidationError,
    TNFRValidator,
    configure_validation,
    validation_config,
)


class TestValidationIntegration:
    """Test validation system integration with structural operators."""

    def test_run_sequence_with_valid_graph(self):
        """Test that valid sequences run without validation errors."""
        G, node = create_nfr("test_node", epi=0.5, vf=1.0, theta=0.0)

        # Valid grammar sequence
        run_sequence(G, node, [Emission(), Reception(), Coherence(), Resonance(), Recursivity()])

        # Node should still exist and have valid attributes
        assert node in G.nodes()

    def test_run_sequence_detects_semantic_violations(self):
        """Test that semantic sequence violations are detected."""
        G, node = create_nfr("test_node", epi=0.5, vf=1.0, theta=0.0)

        # Excessive dissonance (3 in a row) triggers error - must follow grammar
        with pytest.raises(ValueError, match="Semantic sequence violations"):
            run_sequence(
                G,
                node,
                [
                    Emission(),
                    Reception(),
                    Coherence(),
                    Dissonance(),
                    Dissonance(),
                    Dissonance(),
                ],
            )

    def test_run_sequence_allows_valid_dissonance(self):
        """Test that two dissonances are allowed."""
        G, node = create_nfr("test_node", epi=0.5, vf=1.0, theta=0.0)

        # Two dissonances should be fine
        run_sequence(
            G,
            node,
            [
                Emission(),
                Reception(),
                Coherence(),
                Dissonance(),
                Dissonance(),
                Recursivity(),
            ],
        )

    def test_validation_can_be_disabled(self):
        """Test that validation can be disabled via config."""
        original_state = validation_config.validate_invariants

        try:
            # Disable validation
            configure_validation(validate_invariants=False)

            G, node = create_nfr("test_node", epi=0.5, vf=1.0, theta=0.0)

            # Valid sequence should still work
            run_sequence(G, node, [Emission(), Reception(), Coherence(), Recursivity()])

        finally:
            # Restore original state
            configure_validation(validate_invariants=original_state)

    def test_semantic_validation_can_be_disabled(self):
        """Test that semantic validation can be disabled."""
        original_state = validation_config.enable_semantic_validation

        try:
            # Disable semantic validation
            configure_validation(enable_semantic_validation=False)

            G, node = create_nfr("test_node", epi=0.5, vf=1.0, theta=0.0)

            # Excessive dissonance should not raise error when semantic validation is off
            run_sequence(
                G,
                node,
                [
                    Emission(),
                    Reception(),
                    Coherence(),
                    Dissonance(),
                    Dissonance(),
                    Dissonance(),
                ],
            )

        finally:
            # Restore original state
            configure_validation(enable_semantic_validation=original_state)

    def test_validation_with_invalid_transition_sequence(self):
        """Test transition without perturbation is caught."""
        G, node = create_nfr("test_node", epi=0.5, vf=1.0, theta=0.0)

        # Transition after COHERENCE without perturbation should trigger error
        # (COHERENCE → TRANSITION lacks dissonance/mutation/resonance)
        with pytest.raises(ValueError, match="Semantic sequence violations"):
            run_sequence(G, node, [Emission(), Reception(), Coherence(), Transition()])

    def test_validation_with_valid_transition_sequence(self):
        """Test transition after dissonance is allowed."""
        G, node = create_nfr("test_node", epi=0.5, vf=1.0, theta=0.0)

        # Transition after dissonance should be fine
        run_sequence(G, node, [Emission(), Reception(), Coherence(), Dissonance(), Transition()])

    def test_operator_tracking(self):
        """Test that operator tracking works for invariant 1."""
        G, node = create_nfr("test_node", epi=0.5, vf=1.0, theta=0.0)

        # After running sequence, graph should have operator tracking
        run_sequence(G, node, [Emission(), Reception(), Coherence(), Recursivity()])

        assert hasattr(G, "_last_operator_applied")
        # Last operator applied would be recursivity
        assert G._last_operator_applied is not None

    def test_direct_validator_on_graph(self):
        """Test validator directly on graph structure."""
        G, node = create_nfr("test_node", epi=0.5, vf=1.0, theta=0.0)

        # Run some operators
        run_sequence(G, node, [Emission(), Reception(), Coherence(), Recursivity()])

        # Directly validate the graph
        validator = TNFRValidator()
        violations = validator.validate_graph(G)

        # Should have no critical violations
        critical = [v for v in violations if v.severity == InvariantSeverity.CRITICAL]
        assert len(critical) == 0

    def test_validation_severity_levels(self):
        """Test that different severity levels are handled correctly."""
        original_severity = validation_config.min_severity

        try:
            # Set to WARNING level
            configure_validation(min_severity=InvariantSeverity.WARNING)

            G, node = create_nfr("test_node", epi=0.5, vf=1.0, theta=0.0)

            # Should still run (warnings don't block)
            run_sequence(G, node, [Emission(), Reception(), Coherence(), Recursivity()])

            # Set to ERROR level
            configure_validation(min_severity=InvariantSeverity.ERROR)

            # Should also run (no errors in valid graph)
            G2, node2 = create_nfr("test_node2", epi=0.5, vf=1.0, theta=0.0)
            run_sequence(G2, node2, [Emission(), Reception(), Coherence(), Recursivity()])

        finally:
            # Restore original severity
            configure_validation(min_severity=original_severity)
