"""Tests for TNFRValidator integration."""

import math

import networkx as nx
import pytest

from tnfr.constants import EPI_PRIMARY, THETA_PRIMARY, VF_PRIMARY
from tnfr.validation import (
    InvariantSeverity,
    TNFRValidationError,
    TNFRValidator,
)


class TestTNFRValidator:
    """Test integrated TNFR validator."""

    def test_valid_graph_passes(self):
        """Test that a valid graph passes all invariants (no ERROR or CRITICAL violations).

        A minimal graph with just node attributes may have WARNING violations
        for missing graph-level attributes (ΔNFR hook, history, seed, C(t)),
        but should have no ERROR or CRITICAL violations.
        """
        G = nx.Graph()
        G.add_node("node1", **{EPI_PRIMARY: 0.5, VF_PRIMARY: 1.0, THETA_PRIMARY: 0.0})

        validator = TNFRValidator()
        violations = validator.validate_graph(G)

        # Should have no ERROR or CRITICAL violations (warnings are acceptable)
        errors = [v for v in violations if v.severity.value in ("error", "critical")]
        assert len(errors) == 0

    def test_invalid_graph_detects_violations(self):
        """Test that invalid values are detected."""
        G = nx.Graph()
        G.add_node("node1", **{EPI_PRIMARY: 2.0, VF_PRIMARY: -1.0, THETA_PRIMARY: 0.0})

        validator = TNFRValidator()
        violations = validator.validate_graph(G)

        assert len(violations) > 0
        # Should have violations from multiple invariants
        invariant_ids = {v.invariant_id for v in violations}
        assert 1 in invariant_ids  # EPI out of range
        assert 2 in invariant_ids  # νf negative

    def test_validate_and_raise_on_errors(self):
        """Test that validate_and_raise raises on errors."""
        G = nx.Graph()
        G.add_node("node1", **{EPI_PRIMARY: 2.0, VF_PRIMARY: 1.0, THETA_PRIMARY: 0.0})

        validator = TNFRValidator()

        with pytest.raises(TNFRValidationError) as exc_info:
            validator.validate_and_raise(G, InvariantSeverity.ERROR)

        assert len(exc_info.value.violations) > 0
        assert "TNFR Invariant Violations" in str(exc_info.value)

    def test_validate_and_raise_ignores_warnings(self):
        """Test that validate_and_raise doesn't raise on warnings when min_severity is ERROR."""
        G = nx.Graph()
        G.add_node("node1", **{EPI_PRIMARY: 0.5, VF_PRIMARY: 1.0, THETA_PRIMARY: 10.0})

        validator = TNFRValidator()

        # Should not raise even though there's a phase warning
        validator.validate_and_raise(G, InvariantSeverity.ERROR)

    def test_severity_filter(self):
        """Test filtering violations by severity."""
        G = nx.Graph()
        G.add_node("node1", **{EPI_PRIMARY: 0.5, VF_PRIMARY: 1.0, THETA_PRIMARY: 10.0})

        validator = TNFRValidator()

        # Get all violations
        all_violations = validator.validate_graph(G)
        assert len(all_violations) > 0

        # Filter only warnings
        warnings = validator.validate_graph(
            G, severity_filter=InvariantSeverity.WARNING
        )
        assert len(warnings) > 0
        assert all(v.severity == InvariantSeverity.WARNING for v in warnings)

        # Filter only errors (should be none in this case)
        errors = validator.validate_graph(G, severity_filter=InvariantSeverity.ERROR)
        assert len(errors) == 0

    def test_generate_report_empty(self):
        """Test report generation for no violations."""
        validator = TNFRValidator()
        report = validator.generate_report([])

        assert "✅" in report
        assert "No TNFR invariant violations" in report

    def test_generate_report_with_violations(self):
        """Test report generation with violations."""
        G = nx.Graph()
        G.add_node(
            "node1", **{EPI_PRIMARY: 2.0, VF_PRIMARY: -1.0, THETA_PRIMARY: float("inf")}
        )

        validator = TNFRValidator()
        violations = validator.validate_graph(G)

        report = validator.generate_report(violations)

        assert "🚨" in report
        assert "TNFR Invariant Violations" in report
        assert "ERROR" in report or "CRITICAL" in report
        # Should have organized by severity
        assert "Invariant #" in report

    def test_custom_validator_integration(self):
        """Test adding custom validators."""
        from tnfr.validation import TNFRInvariant, InvariantViolation

        class CustomValidator(TNFRInvariant):
            invariant_id = 99
            description = "Custom test validator"

            def validate(self, graph):
                # Always return a test violation
                return [
                    InvariantViolation(
                        invariant_id=99,
                        severity=InvariantSeverity.WARNING,
                        description="Custom test violation",
                    )
                ]

        G = nx.Graph()
        G.add_node("node1", **{EPI_PRIMARY: 0.5, VF_PRIMARY: 1.0, THETA_PRIMARY: 0.0})

        validator = TNFRValidator()
        validator.add_custom_validator(CustomValidator())

        violations = validator.validate_graph(G)

        # Should have the custom violation
        custom_violations = [v for v in violations if v.invariant_id == 99]
        assert len(custom_violations) == 1
        assert custom_violations[0].description == "Custom test violation"

    def test_custom_phase_threshold(self):
        """Test custom phase coupling threshold."""
        G = nx.Graph()
        G.add_node("node1", **{EPI_PRIMARY: 0.5, VF_PRIMARY: 1.0, THETA_PRIMARY: 0.0})
        G.add_node(
            "node2", **{EPI_PRIMARY: 0.5, VF_PRIMARY: 1.0, THETA_PRIMARY: 2.0}
        )  # Larger phase diff
        G.add_edge("node1", "node2")

        # Default threshold (π/2 ≈ 1.57), phase diff of 2.0 should trigger warning
        validator_default = TNFRValidator()
        violations_default = validator_default.validate_graph(G)
        warnings_default = [
            v for v in violations_default if v.severity == InvariantSeverity.WARNING
        ]

        # Lenient threshold (π), phase diff of 2.0 should not trigger warning
        validator_lenient = TNFRValidator(phase_coupling_threshold=math.pi)
        violations_lenient = validator_lenient.validate_graph(G)
        warnings_lenient = [
            v for v in violations_lenient if v.severity == InvariantSeverity.WARNING
        ]

        # Lenient should have fewer warnings
        assert len(warnings_lenient) < len(warnings_default)

    def test_multiple_nodes(self):
        """Test validation with multiple nodes."""
        G = nx.Graph()
        G.add_node("node1", **{EPI_PRIMARY: 0.5, VF_PRIMARY: 1.0, THETA_PRIMARY: 0.0})
        G.add_node("node2", **{EPI_PRIMARY: 0.6, VF_PRIMARY: 2.0, THETA_PRIMARY: 1.0})
        G.add_node(
            "node3", **{EPI_PRIMARY: 2.0, VF_PRIMARY: 1.0, THETA_PRIMARY: 0.0}
        )  # Invalid EPI

        validator = TNFRValidator()
        violations = validator.validate_graph(G)

        # Should have violation for node3
        assert len(violations) > 0
        node3_violations = [v for v in violations if v.node_id == "node3"]
        assert len(node3_violations) > 0

    def test_validator_exception_handling(self):
        """Test that validator handles exceptions in custom validators."""
        from tnfr.validation import TNFRInvariant

        class FailingValidator(TNFRInvariant):
            invariant_id = 98
            description = "Validator that raises exception"

            def validate(self, graph):
                raise RuntimeError("Validator failed")

        G = nx.Graph()
        G.add_node("node1", **{EPI_PRIMARY: 0.5, VF_PRIMARY: 1.0, THETA_PRIMARY: 0.0})

        validator = TNFRValidator()
        validator.add_custom_validator(FailingValidator())

        violations = validator.validate_graph(G)

        # Should have a critical violation about validator failure
        critical_violations = [
            v for v in violations if v.severity == InvariantSeverity.CRITICAL
        ]
        assert any(
            "Validator execution failed" in v.description for v in critical_violations
        )
