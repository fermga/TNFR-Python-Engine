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
            if v.severity == InvariantSeverity.WARNING and "phase difference" in v.description
        ]
        assert len(phase_warnings) == 0

    def test_custom_phase_threshold(self):
        """Test custom phase coupling threshold."""
        G = nx.Graph()
        G.add_node("node1", **{EPI_PRIMARY: 0.5, VF_PRIMARY: 1.0, THETA_PRIMARY: 0.0})
        G.add_node(
            "node2", **{EPI_PRIMARY: 0.5, VF_PRIMARY: 1.0, THETA_PRIMARY: 2.0}
        )  # Larger phase difference
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
            if v.severity == InvariantSeverity.WARNING and "phase difference" in v.description
        ]
        assert len(phase_warnings) == 0


class TestInvariant3_DNFRSemantics:
    """Test Invariant 3: ΔNFR semantics."""

    def test_valid_dnfr(self):
        """Test that valid ΔNFR values pass validation."""
        from tnfr.validation import Invariant3_DNFRSemantics

        G = nx.Graph()
        G.add_node(
            "node1",
            **{
                EPI_PRIMARY: 0.5,
                VF_PRIMARY: 1.0,
                THETA_PRIMARY: 0.0,
                DNFR_PRIMARY: 0.5,
            },
        )

        validator = Invariant3_DNFRSemantics()
        violations = validator.validate(G)

        assert len(violations) == 0

    def test_dnfr_not_finite(self):
        """Test that non-finite ΔNFR is flagged as critical."""
        from tnfr.validation import Invariant3_DNFRSemantics

        G = nx.Graph()
        G.add_node(
            "node1",
            **{
                EPI_PRIMARY: 0.5,
                VF_PRIMARY: 1.0,
                THETA_PRIMARY: 0.0,
                DNFR_PRIMARY: float("inf"),
            },
        )

        validator = Invariant3_DNFRSemantics()
        violations = validator.validate(G)

        critical_violations = [v for v in violations if v.severity == InvariantSeverity.CRITICAL]
        assert len(critical_violations) >= 1
        assert any("not a finite number" in v.description for v in critical_violations)

    def test_dnfr_unusually_large(self):
        """Test that unusually large ΔNFR triggers warning."""
        from tnfr.validation import Invariant3_DNFRSemantics

        G = nx.Graph()
        G.add_node(
            "node1",
            **{
                EPI_PRIMARY: 0.5,
                VF_PRIMARY: 1.0,
                THETA_PRIMARY: 0.0,
                DNFR_PRIMARY: 1500.0,
            },
        )

        validator = Invariant3_DNFRSemantics()
        violations = validator.validate(G)

        warning_violations = [v for v in violations if v.severity == InvariantSeverity.WARNING]
        assert any("unusually large" in v.description for v in warning_violations)


class TestInvariant4_OperatorClosure:
    """Test Invariant 4: Operator closure."""

    def test_valid_graph_structure(self):
        """Test that graph with all required attributes passes."""
        from tnfr.validation import Invariant4_OperatorClosure

        G = nx.Graph()
        G.add_node("node1", **{EPI_PRIMARY: 0.5, VF_PRIMARY: 1.0, THETA_PRIMARY: 0.0})
        G.graph["compute_delta_nfr"] = lambda g: None

        validator = Invariant4_OperatorClosure()
        violations = validator.validate(G)

        critical_violations = [v for v in violations if v.severity == InvariantSeverity.CRITICAL]
        assert len(critical_violations) == 0

    def test_missing_required_attributes(self):
        """Test that missing TNFR attributes are flagged."""
        from tnfr.validation import Invariant4_OperatorClosure

        G = nx.Graph()
        G.add_node("node1", **{EPI_PRIMARY: 0.5})  # Missing VF and THETA

        validator = Invariant4_OperatorClosure()
        violations = validator.validate(G)

        critical_violations = [v for v in violations if v.severity == InvariantSeverity.CRITICAL]
        assert len(critical_violations) >= 1
        assert any("missing required TNFR attributes" in v.description for v in critical_violations)

    def test_missing_dnfr_hook(self):
        """Test that missing ΔNFR hook triggers warning."""
        from tnfr.validation import Invariant4_OperatorClosure

        G = nx.Graph()
        G.add_node("node1", **{EPI_PRIMARY: 0.5, VF_PRIMARY: 1.0, THETA_PRIMARY: 0.0})

        validator = Invariant4_OperatorClosure()
        violations = validator.validate(G)

        warning_violations = [v for v in violations if v.severity == InvariantSeverity.WARNING]
        assert any("missing ΔNFR computation hook" in v.description for v in warning_violations)


class TestInvariant6_NodeBirthCollapse:
    """Test Invariant 6: Node birth/collapse conditions."""

    def test_valid_node_conditions(self):
        """Test that node with sufficient conditions passes."""
        from tnfr.validation import Invariant6_NodeBirthCollapse

        G = nx.Graph()
        G.add_node(
            "node1",
            **{
                EPI_PRIMARY: 0.5,
                VF_PRIMARY: 1.0,
                THETA_PRIMARY: 0.0,
                DNFR_PRIMARY: 0.5,
            },
        )

        validator = Invariant6_NodeBirthCollapse()
        violations = validator.validate(G)

        warning_violations = [v for v in violations if v.severity == InvariantSeverity.WARNING]
        assert len(warning_violations) == 0

    def test_insufficient_vf(self):
        """Test that low νf triggers collapse warning."""
        from tnfr.validation import Invariant6_NodeBirthCollapse

        G = nx.Graph()
        G.add_node(
            "node1",
            **{
                EPI_PRIMARY: 0.5,
                VF_PRIMARY: 0.0001,
                THETA_PRIMARY: 0.0,
                DNFR_PRIMARY: 0.0,
            },
        )

        validator = Invariant6_NodeBirthCollapse()
        violations = validator.validate(G)

        warning_violations = [v for v in violations if v.severity == InvariantSeverity.WARNING]
        assert any("insufficient νf" in v.description for v in warning_violations)

    def test_extreme_dissonance(self):
        """Test that extreme ΔNFR triggers collapse warning."""
        from tnfr.validation import Invariant6_NodeBirthCollapse

        G = nx.Graph()
        G.add_node(
            "node1",
            **{
                EPI_PRIMARY: 0.5,
                VF_PRIMARY: 1.0,
                THETA_PRIMARY: 0.0,
                DNFR_PRIMARY: 15.0,
            },
        )

        validator = Invariant6_NodeBirthCollapse()
        violations = validator.validate(G)

        warning_violations = [v for v in violations if v.severity == InvariantSeverity.WARNING]
        assert any("extreme dissonance" in v.description for v in warning_violations)


class TestInvariant7_OperationalFractality:
    """Test Invariant 7: Operational fractality."""

    def test_simple_epi_passes(self):
        """Test that simple EPI passes validation."""
        from tnfr.validation import Invariant7_OperationalFractality

        G = nx.Graph()
        G.add_node("node1", **{EPI_PRIMARY: 0.5, VF_PRIMARY: 1.0, THETA_PRIMARY: 0.0})

        validator = Invariant7_OperationalFractality()
        violations = validator.validate(G)

        assert len(violations) == 0

    def test_nested_epi_structure(self):
        """Test that properly nested EPI passes validation."""
        from tnfr.validation import Invariant7_OperationalFractality

        G = nx.Graph()
        nested_epi = {
            "continuous": ((0.5 + 0j), (0.6 + 0j)),
            "discrete": ((0.4 + 0j), (0.5 + 0j)),
            "grid": (0.0, 1.0),
        }
        G.add_node("node1", **{EPI_PRIMARY: nested_epi, VF_PRIMARY: 1.0, THETA_PRIMARY: 0.0})

        validator = Invariant7_OperationalFractality()
        violations = validator.validate(G)

        error_violations = [v for v in violations if v.severity == InvariantSeverity.ERROR]
        assert len(error_violations) == 0

    def test_nested_epi_with_invalid_values(self):
        """Test that nested EPI with non-finite values is flagged."""
        from tnfr.validation import Invariant7_OperationalFractality

        G = nx.Graph()
        nested_epi = {
            "continuous": ((float("inf") + 0j), (0.6 + 0j)),
            "discrete": ((0.4 + 0j), (0.5 + 0j)),
        }
        G.add_node("node1", **{EPI_PRIMARY: nested_epi, VF_PRIMARY: 1.0, THETA_PRIMARY: 0.0})

        validator = Invariant7_OperationalFractality()
        violations = validator.validate(G)

        error_violations = [v for v in violations if v.severity == InvariantSeverity.ERROR]
        assert any("non-finite values" in v.description for v in error_violations)


class TestInvariant8_ControlledDeterminism:
    """Test Invariant 8: Controlled determinism."""

    def test_graph_with_history_and_seed(self):
        """Test that graph with history and seed passes."""
        from tnfr.validation import Invariant8_ControlledDeterminism

        G = nx.Graph()
        G.add_node("node1", **{EPI_PRIMARY: 0.5, VF_PRIMARY: 1.0, THETA_PRIMARY: 0.0})
        G.graph["HISTORY_MAXLEN"] = 50
        G.graph["RANDOM_SEED"] = 42

        validator = Invariant8_ControlledDeterminism()
        violations = validator.validate(G)

        assert len(violations) == 0

    def test_missing_history_tracking(self):
        """Test that missing history tracking triggers warning."""
        from tnfr.validation import Invariant8_ControlledDeterminism

        G = nx.Graph()
        G.add_node("node1", **{EPI_PRIMARY: 0.5, VF_PRIMARY: 1.0, THETA_PRIMARY: 0.0})
        G.graph["RANDOM_SEED"] = 42

        validator = Invariant8_ControlledDeterminism()
        violations = validator.validate(G)

        warning_violations = [v for v in violations if v.severity == InvariantSeverity.WARNING]
        assert any("history tracking" in v.description for v in warning_violations)

    def test_missing_random_seed(self):
        """Test that missing random seed triggers warning."""
        from tnfr.validation import Invariant8_ControlledDeterminism

        G = nx.Graph()
        G.add_node("node1", **{EPI_PRIMARY: 0.5, VF_PRIMARY: 1.0, THETA_PRIMARY: 0.0})
        G.graph["HISTORY_MAXLEN"] = 50

        validator = Invariant8_ControlledDeterminism()
        violations = validator.validate(G)

        warning_violations = [v for v in violations if v.severity == InvariantSeverity.WARNING]
        assert any("random seed" in v.description for v in warning_violations)


class TestInvariant9_StructuralMetrics:
    """Test Invariant 9: Structural metrics."""

    def test_valid_structural_metrics(self):
        """Test that graph with valid metrics passes."""
        from tnfr.validation import Invariant9_StructuralMetrics

        G = nx.Graph()
        G.add_node(
            "node1",
            **{EPI_PRIMARY: 0.5, VF_PRIMARY: 1.0, THETA_PRIMARY: 0.0, "Si": 0.7},
        )
        G.graph["coherence"] = 0.8

        validator = Invariant9_StructuralMetrics()
        violations = validator.validate(G)

        warning_violations = [v for v in violations if v.severity == InvariantSeverity.WARNING]
        assert len(warning_violations) == 0

    def test_si_out_of_range(self):
        """Test that Si outside [0,1] triggers warning."""
        from tnfr.validation import Invariant9_StructuralMetrics

        G = nx.Graph()
        G.add_node(
            "node1",
            **{EPI_PRIMARY: 0.5, VF_PRIMARY: 1.0, THETA_PRIMARY: 0.0, "Si": 1.5},
        )

        validator = Invariant9_StructuralMetrics()
        violations = validator.validate(G)

        warning_violations = [v for v in violations if v.severity == InvariantSeverity.WARNING]
        assert any(
            "Sense index (Si) outside expected range" in v.description for v in warning_violations
        )

    def test_missing_coherence_metric(self):
        """Test that missing C(t) triggers warning."""
        from tnfr.validation import Invariant9_StructuralMetrics

        G = nx.Graph()
        G.add_node("node1", **{EPI_PRIMARY: 0.5, VF_PRIMARY: 1.0, THETA_PRIMARY: 0.0})

        validator = Invariant9_StructuralMetrics()
        violations = validator.validate(G)

        warning_violations = [v for v in violations if v.severity == InvariantSeverity.WARNING]
        assert any("coherence metric C(t)" in v.description for v in warning_violations)


class TestInvariant10_DomainNeutrality:
    """Test Invariant 10: Domain neutrality."""

    def test_domain_neutral_graph(self):
        """Test that domain-neutral graph passes."""
        from tnfr.validation import Invariant10_DomainNeutrality

        G = nx.Graph()
        G.add_node("node1", **{EPI_PRIMARY: 0.5, VF_PRIMARY: 1.0, THETA_PRIMARY: 0.0})

        validator = Invariant10_DomainNeutrality()
        violations = validator.validate(G)

        warning_violations = [v for v in violations if v.severity == InvariantSeverity.WARNING]
        assert len(warning_violations) == 0

    def test_domain_specific_keys(self):
        """Test that domain-specific keys trigger warning."""
        from tnfr.validation import Invariant10_DomainNeutrality

        G = nx.Graph()
        G.add_node("node1", **{EPI_PRIMARY: 0.5, VF_PRIMARY: 1.0, THETA_PRIMARY: 0.0})
        G.graph["temperature"] = 300
        G.graph["biology"] = True

        validator = Invariant10_DomainNeutrality()
        violations = validator.validate(G)

        warning_violations = [v for v in violations if v.severity == InvariantSeverity.WARNING]
        assert any("Domain-specific keys found" in v.description for v in warning_violations)

    def test_non_structural_units(self):
        """Test that non-structural units trigger error."""
        from tnfr.validation import Invariant10_DomainNeutrality

        G = nx.Graph()
        G.add_node(
            "node1",
            **{
                EPI_PRIMARY: 0.5,
                VF_PRIMARY: 1.0,
                THETA_PRIMARY: 0.0,
                "units": {"vf": "Hz"},  # Physical Hz, not Hz_str
            },
        )

        validator = Invariant10_DomainNeutrality()
        violations = validator.validate(G)

        error_violations = [v for v in violations if v.severity == InvariantSeverity.ERROR]
        assert any("Non-structural units" in v.description for v in error_violations)
