"""Tests for TNFR invariant validators.

This module tests the core invariant validation system including:
- Invariant 1: EPI changes only through operators
- Invariant 2: νf stays in Hz_str units
- Invariant 5: Explicit phase checks
"""

import math

import networkx as nx
import pytest

from tnfr.constants import DNFR_PRIMARY, EPI_PRIMARY, THETA_PRIMARY, VF_PRIMARY
from tnfr.validation import (
    InvariantSeverity,
    InvariantViolation,
    Invariant1_EPIOnlyThroughOperators,
    Invariant2_VfInHzStr,
    Invariant5_ExplicitPhaseChecks,
)


class TestInvariant1_EPIOnlyThroughOperators:
    """Test Invariant 1: EPI changes only through structural operators."""

    def test_valid_epi_range(self):
        """Test that valid EPI values pass validation."""
        G = nx.Graph()
        G.add_node("node1", **{EPI_PRIMARY: 0.5, VF_PRIMARY: 1.0, THETA_PRIMARY: 0.0})

        validator = Invariant1_EPIOnlyThroughOperators()
        violations = validator.validate(G)

        assert len(violations) == 0

    def test_epi_below_minimum(self):
        """Test that EPI below minimum is flagged."""
        G = nx.Graph()
        # Set custom bounds to ensure violation
        G.graph["EPI_MIN"] = 0.0
        G.graph["EPI_MAX"] = 1.0
        G.add_node("node1", **{EPI_PRIMARY: -0.5, VF_PRIMARY: 1.0, THETA_PRIMARY: 0.0})

        validator = Invariant1_EPIOnlyThroughOperators()
        violations = validator.validate(G)

        assert len(violations) == 1
        assert violations[0].invariant_id == 1
        assert violations[0].severity == InvariantSeverity.ERROR
        assert "out of valid range" in violations[0].description

    def test_epi_above_maximum(self):
        """Test that EPI above maximum is flagged."""
        G = nx.Graph()
        G.add_node("node1", **{EPI_PRIMARY: 1.5, VF_PRIMARY: 1.0, THETA_PRIMARY: 0.0})

        validator = Invariant1_EPIOnlyThroughOperators()
        violations = validator.validate(G)

        assert len(violations) == 1
        assert violations[0].invariant_id == 1
        assert violations[0].severity == InvariantSeverity.ERROR

    def test_epi_not_finite(self):
        """Test that non-finite EPI values are flagged as critical."""
        G = nx.Graph()
        G.add_node("node1", **{EPI_PRIMARY: float("inf"), VF_PRIMARY: 1.0, THETA_PRIMARY: 0.0})

        validator = Invariant1_EPIOnlyThroughOperators()
        violations = validator.validate(G)

        critical_violations = [v for v in violations if v.severity == InvariantSeverity.CRITICAL]
        assert len(critical_violations) >= 1
        assert any("not a finite number" in v.description for v in critical_violations)

    def test_epi_change_without_operator(self):
        """Test that EPI changes without operator are detected."""
        G = nx.Graph()
        G.add_node("node1", **{EPI_PRIMARY: 0.5, VF_PRIMARY: 1.0, THETA_PRIMARY: 0.0})
        G._last_operator_applied = None  # type: ignore[attr-defined]

        validator = Invariant1_EPIOnlyThroughOperators()
        # First pass to establish baseline
        violations = validator.validate(G)

        # Change EPI without operator
        G.nodes["node1"][EPI_PRIMARY] = 0.8
        violations = validator.validate(G)

        critical_violations = [v for v in violations if v.severity == InvariantSeverity.CRITICAL]
        assert any("changed without operator" in v.description for v in critical_violations)

    def test_epi_change_with_operator_allowed(self):
        """Test that EPI changes with operator are allowed."""
        G = nx.Graph()
        G.add_node("node1", **{EPI_PRIMARY: 0.5, VF_PRIMARY: 1.0, THETA_PRIMARY: 0.0})
        G._last_operator_applied = "coherence"  # type: ignore[attr-defined]

        validator = Invariant1_EPIOnlyThroughOperators()
        # First pass to establish baseline
        violations = validator.validate(G)

        # Change EPI with operator marked
        G.nodes["node1"][EPI_PRIMARY] = 0.8
        violations = validator.validate(G)

        # Should not flag unauthorized change
        critical_violations = [
            v
            for v in violations
            if v.severity == InvariantSeverity.CRITICAL
            and "changed without operator" in v.description
        ]
        assert len(critical_violations) == 0


class TestInvariant2_VfInHzStr:
    """Test Invariant 2: νf stays in Hz_str units."""

    def test_valid_vf_range(self):
        """Test that valid νf values pass validation."""
        G = nx.Graph()
        G.add_node("node1", **{EPI_PRIMARY: 0.5, VF_PRIMARY: 1.0, THETA_PRIMARY: 0.0})

        validator = Invariant2_VfInHzStr()
        violations = validator.validate(G)

        assert len(violations) == 0

    def test_vf_below_minimum(self):
        """Test that νf below minimum is flagged."""
        G = nx.Graph()
        # Set custom bounds to ensure violation
        G.graph["VF_MIN"] = 0.001
        G.graph["VF_MAX"] = 1000.0
        G.add_node("node1", **{EPI_PRIMARY: 0.5, VF_PRIMARY: 0.0001, THETA_PRIMARY: 0.0})

        validator = Invariant2_VfInHzStr()
        violations = validator.validate(G)

        assert len(violations) >= 1
        assert any(v.invariant_id == 2 for v in violations)
        assert any("outside typical Hz_str range" in v.description for v in violations)

    def test_vf_above_maximum(self):
        """Test that νf above maximum is flagged."""
        G = nx.Graph()
        G.add_node("node1", **{EPI_PRIMARY: 0.5, VF_PRIMARY: 2000.0, THETA_PRIMARY: 0.0})

        validator = Invariant2_VfInHzStr()
        violations = validator.validate(G)

        assert len(violations) >= 1
        assert any(v.invariant_id == 2 for v in violations)

    def test_vf_not_finite(self):
        """Test that non-finite νf values are flagged as critical."""
        G = nx.Graph()
        G.add_node("node1", **{EPI_PRIMARY: 0.5, VF_PRIMARY: float("nan"), THETA_PRIMARY: 0.0})

        validator = Invariant2_VfInHzStr()
        violations = validator.validate(G)

        critical_violations = [v for v in violations if v.severity == InvariantSeverity.CRITICAL]
        assert len(critical_violations) >= 1
        assert any("not a finite number" in v.description for v in critical_violations)

    def test_vf_negative(self):
        """Test that negative νf is flagged."""
        G = nx.Graph()
        G.add_node("node1", **{EPI_PRIMARY: 0.5, VF_PRIMARY: -1.0, THETA_PRIMARY: 0.0})

        validator = Invariant2_VfInHzStr()
        violations = validator.validate(G)

        error_violations = [v for v in violations if v.severity == InvariantSeverity.ERROR]
        assert any("must be positive" in v.description for v in error_violations)

    def test_vf_zero(self):
        """Test that zero νf is flagged."""
        G = nx.Graph()
        G.add_node("node1", **{EPI_PRIMARY: 0.5, VF_PRIMARY: 0.0, THETA_PRIMARY: 0.0})

        validator = Invariant2_VfInHzStr()
        violations = validator.validate(G)

        error_violations = [v for v in violations if v.severity == InvariantSeverity.ERROR]
        assert any("must be positive" in v.description for v in error_violations)


class TestInvariant5_ExplicitPhaseChecks:
    """Test Invariant 5: Explicit phase checks for coupling."""

    def test_valid_phase(self):
        """Test that valid phase values pass validation."""
        G = nx.Graph()
        G.add_node("node1", **{EPI_PRIMARY: 0.5, VF_PRIMARY: 1.0, THETA_PRIMARY: 1.0})

        validator = Invariant5_ExplicitPhaseChecks()
        violations = validator.validate(G)

        critical_violations = [v for v in violations if v.severity == InvariantSeverity.CRITICAL]
        assert len(critical_violations) == 0

    def test_phase_not_finite(self):
        """Test that non-finite phase values are flagged as critical."""
        G = nx.Graph()
        G.add_node("node1", **{EPI_PRIMARY: 0.5, VF_PRIMARY: 1.0, THETA_PRIMARY: float("inf")})

        validator = Invariant5_ExplicitPhaseChecks()
        violations = validator.validate(G)

        critical_violations = [v for v in violations if v.severity == InvariantSeverity.CRITICAL]
        assert len(critical_violations) >= 1
        assert any("not a finite number" in v.description for v in critical_violations)

    def test_phase_outside_range_warning(self):
        """Test that phase outside [0, 2π] triggers warning."""
        G = nx.Graph()
        G.add_node("node1", **{EPI_PRIMARY: 0.5, VF_PRIMARY: 1.0, THETA_PRIMARY: 10.0})

        validator = Invariant5_ExplicitPhaseChecks()
        violations = validator.validate(G)

        warning_violations = [v for v in violations if v.severity == InvariantSeverity.WARNING]
        assert any("outside [0, 2π]" in v.description for v in warning_violations)

    def test_coupled_nodes_phase_difference(self):
        """Test phase difference detection in coupled nodes."""
        G = nx.Graph()
        G.add_node("node1", **{EPI_PRIMARY: 0.5, VF_PRIMARY: 1.0, THETA_PRIMARY: 0.0})
        G.add_node("node2", **{EPI_PRIMARY: 0.5, VF_PRIMARY: 1.0, THETA_PRIMARY: math.pi})
        G.add_edge("node1", "node2")

        validator = Invariant5_ExplicitPhaseChecks()
        violations = validator.validate(G)

        warning_violations = [v for v in violations if v.severity == InvariantSeverity.WARNING]
        assert any("Large phase difference" in v.description for v in warning_violations)

    def test_coupled_nodes_synchronized(self):
        """Test that synchronized coupled nodes pass validation."""
        G = nx.Graph()
        G.add_node("node1", **{EPI_PRIMARY: 0.5, VF_PRIMARY: 1.0, THETA_PRIMARY: 1.0})
        G.add_node("node2", **{EPI_PRIMARY: 0.5, VF_PRIMARY: 1.0, THETA_PRIMARY: 1.1})
        G.add_edge("node1", "node2")

        validator = Invariant5_ExplicitPhaseChecks()
        violations = validator.validate(G)

        # Small phase difference should not trigger warning
        phase_warnings = [
            v
            for v in violations
            if v.severity == InvariantSeverity.WARNING
            and "phase difference" in v.description
        ]
        assert len(phase_warnings) == 0

    def test_custom_phase_threshold(self):
        """Test custom phase coupling threshold."""
        G = nx.Graph()
        G.add_node("node1", **{EPI_PRIMARY: 0.5, VF_PRIMARY: 1.0, THETA_PRIMARY: 0.0})
        G.add_node("node2", **{EPI_PRIMARY: 0.5, VF_PRIMARY: 1.0, THETA_PRIMARY: 2.0})  # Larger phase difference
        G.add_edge("node1", "node2")

        # With default threshold (π/2 ≈ 1.57), phase diff of 2.0 should trigger warning
        validator = Invariant5_ExplicitPhaseChecks()
        violations = validator.validate(G)
        assert any(v.severity == InvariantSeverity.WARNING for v in violations)

        # With larger threshold (π), phase diff of 2.0 should not trigger warning
        validator_lenient = Invariant5_ExplicitPhaseChecks(phase_coupling_threshold=math.pi)
        violations_lenient = validator_lenient.validate(G)
        phase_warnings = [
            v
            for v in violations_lenient
            if v.severity == InvariantSeverity.WARNING
            and "phase difference" in v.description
        ]
        assert len(phase_warnings) == 0
