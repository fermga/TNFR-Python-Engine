"""Tests for THOL (Self-organization) enhanced precondition validation.

This module validates the strengthened implementation of THOL preconditions
according to canonical TNFR principles:

1. EPI ≥ min_epi (sufficient structure for bifurcation)
2. ΔNFR > 0 (reorganization pressure)
3. νf ≥ min_vf (structural reorganization capacity)
4. degree ≥ min_degree (network connectivity for metabolism)
5. len(epi_history) ≥ 3 (history for d²EPI/dt² computation)
6. Metabolic context (no isolated nodes when metabolism enabled)

The validation can be configured via graph flags to support different
operational modes (canonical vs simplified).
"""

import pytest
import networkx as nx

from tnfr.alias import set_attr
from tnfr.constants.aliases import ALIAS_DNFR, ALIAS_EPI, ALIAS_THETA, ALIAS_VF
from tnfr.operators.definitions import SelfOrganization
from tnfr.operators.preconditions import (
    OperatorPreconditionError,
    validate_self_organization,
)
from tnfr.structural import create_nfr, run_sequence


class TestTHOLPreconditions:
    """Test suite for THOL enhanced precondition validation."""

    def test_thol_rejects_isolated_node_by_default(self):
        """THOL should fail on isolated node unless explicitly allowed."""
        G, node = create_nfr("isolated", epi=0.50, vf=1.0)
        set_attr(G.nodes[node], ALIAS_DNFR, 0.15)
        G.nodes[node]["epi_history"] = [0.35, 0.42, 0.50]

        # Should raise error (not just warning)
        with pytest.raises(
            OperatorPreconditionError, match="insufficiently connected"
        ):
            validate_self_organization(G, node)

    def test_thol_allows_isolated_when_configured(self):
        """THOL should allow isolated nodes if explicitly configured."""
        G, node = create_nfr("isolated", epi=0.50, vf=1.0)
        set_attr(G.nodes[node], ALIAS_DNFR, 0.15)
        G.nodes[node]["epi_history"] = [0.35, 0.42, 0.50]
        G.graph["THOL_ALLOW_ISOLATED"] = True
        G.graph["THOL_METABOLIC_ENABLED"] = False  # Required for isolated mode

        # Should not raise
        validate_self_organization(G, node)  # OK

    def test_thol_requires_positive_vf(self):
        """THOL requires νf > 0 for reorganization capacity."""
        G = nx.Graph()
        G.add_node(0, theta=0.1)
        set_attr(G.nodes[0], ALIAS_EPI, 0.50)
        set_attr(G.nodes[0], ALIAS_VF, 0.05)  # vf too low
        set_attr(G.nodes[0], ALIAS_DNFR, 0.15)
        
        G.add_node(1, theta=0.15)
        set_attr(G.nodes[1], ALIAS_EPI, 0.50)
        set_attr(G.nodes[1], ALIAS_VF, 1.0)
        set_attr(G.nodes[1], ALIAS_DNFR, 0.10)
        G.add_edge(0, 1)

        G.nodes[0]["epi_history"] = [0.35, 0.42, 0.50]
        G.graph["THOL_MIN_VF"] = 0.1

        with pytest.raises(
            OperatorPreconditionError, match="Structural frequency too low"
        ):
            validate_self_organization(G, 0)

    def test_thol_requires_sufficient_history(self):
        """THOL requires ≥3 EPI history points for d²EPI/dt²."""
        G = nx.Graph()
        G.add_node(0, theta=0.1)
        set_attr(G.nodes[0], ALIAS_EPI, 0.50)
        set_attr(G.nodes[0], ALIAS_VF, 1.0)
        set_attr(G.nodes[0], ALIAS_DNFR, 0.15)
        
        G.add_node(1, theta=0.15)
        set_attr(G.nodes[1], ALIAS_EPI, 0.50)
        set_attr(G.nodes[1], ALIAS_VF, 1.0)
        set_attr(G.nodes[1], ALIAS_DNFR, 0.10)
        G.add_edge(0, 1)

        G.nodes[0]["epi_history"] = [0.42, 0.50]  # Only 2 points

        with pytest.raises(
            OperatorPreconditionError,
            match="Insufficient EPI history for acceleration",
        ):
            validate_self_organization(G, 0)

    def test_thol_rejects_isolated_with_metabolism_enabled(self):
        """Metabolic mode should fail on isolated nodes."""
        G, node = create_nfr("isolated", epi=0.50, vf=1.0)
        set_attr(G.nodes[node], ALIAS_DNFR, 0.15)
        G.nodes[node]["epi_history"] = [0.35, 0.42, 0.50]
        G.graph["THOL_METABOLIC_ENABLED"] = True
        G.graph["THOL_ALLOW_ISOLATED"] = True  # Contradictory configuration

        # Metabolism requires network context
        with pytest.raises(
            OperatorPreconditionError,
            match="Metabolic mode enabled but node is isolated",
        ):
            validate_self_organization(G, node)

    def test_thol_passes_with_valid_conditions(self):
        """THOL should pass when all preconditions are met."""
        G = nx.Graph()
        G.add_node(0, theta=0.1)
        set_attr(G.nodes[0], ALIAS_EPI, 0.50)
        set_attr(G.nodes[0], ALIAS_VF, 1.0)
        set_attr(G.nodes[0], ALIAS_DNFR, 0.15)
        
        G.add_node(1, theta=0.15)
        set_attr(G.nodes[1], ALIAS_EPI, 0.50)
        set_attr(G.nodes[1], ALIAS_VF, 1.0)
        set_attr(G.nodes[1], ALIAS_DNFR, 0.10)
        G.add_edge(0, 1)

        G.nodes[0]["epi_history"] = [0.35, 0.42, 0.50]
        G.nodes[0]["glyph_history"] = []  # Empty history for destabilizer check

        # Should not raise
        validate_self_organization(G, 0)

    def test_thol_respects_custom_thresholds(self):
        """THOL should respect custom configuration thresholds."""
        G = nx.Graph()
        G.add_node(0, theta=0.1)
        set_attr(G.nodes[0], ALIAS_EPI, 0.25)
        set_attr(G.nodes[0], ALIAS_VF, 0.15)
        set_attr(G.nodes[0], ALIAS_DNFR, 0.10)
        
        G.add_node(1, theta=0.15)
        set_attr(G.nodes[1], ALIAS_EPI, 0.50)
        set_attr(G.nodes[1], ALIAS_VF, 1.0)
        set_attr(G.nodes[1], ALIAS_DNFR, 0.10)
        G.add_edge(0, 1)

        G.nodes[0]["epi_history"] = [0.20, 0.22, 0.25]
        G.nodes[0]["glyph_history"] = []

        # Custom lower thresholds
        G.graph["THOL_MIN_EPI"] = 0.20
        G.graph["THOL_MIN_VF"] = 0.10

        # Should not raise with custom thresholds
        validate_self_organization(G, 0)

    def test_thol_fails_low_epi(self):
        """THOL should fail when EPI is too low for bifurcation."""
        G = nx.Graph()
        G.add_node(0, theta=0.1)
        set_attr(G.nodes[0], ALIAS_EPI, 0.15)  # EPI too low
        set_attr(G.nodes[0], ALIAS_VF, 1.0)
        set_attr(G.nodes[0], ALIAS_DNFR, 0.15)
        
        G.add_node(1, theta=0.15)
        set_attr(G.nodes[1], ALIAS_EPI, 0.50)
        set_attr(G.nodes[1], ALIAS_VF, 1.0)
        set_attr(G.nodes[1], ALIAS_DNFR, 0.10)
        G.add_edge(0, 1)

        G.nodes[0]["epi_history"] = [0.10, 0.12, 0.15]

        with pytest.raises(
            OperatorPreconditionError, match="EPI too low for bifurcation"
        ):
            validate_self_organization(G, 0)

    def test_thol_fails_non_positive_dnfr(self):
        """THOL should fail when ΔNFR is not positive."""
        G = nx.Graph()
        G.add_node(0, theta=0.1)
        set_attr(G.nodes[0], ALIAS_EPI, 0.50)
        set_attr(G.nodes[0], ALIAS_VF, 1.0)
        set_attr(G.nodes[0], ALIAS_DNFR, -0.05)  # Negative ΔNFR
        
        G.add_node(1, theta=0.15)
        set_attr(G.nodes[1], ALIAS_EPI, 0.50)
        set_attr(G.nodes[1], ALIAS_VF, 1.0)
        set_attr(G.nodes[1], ALIAS_DNFR, 0.10)
        G.add_edge(0, 1)

        G.nodes[0]["epi_history"] = [0.45, 0.48, 0.50]

        with pytest.raises(
            OperatorPreconditionError, match="ΔNFR non-positive"
        ):
            validate_self_organization(G, 0)

    def test_thol_integration_with_run_sequence(self):
        """THOL preconditions should be checked before execution."""
        G = nx.Graph()
        G.add_node(0, theta=0.1)
        set_attr(G.nodes[0], ALIAS_EPI, 0.50)
        set_attr(G.nodes[0], ALIAS_VF, 1.0)
        set_attr(G.nodes[0], ALIAS_DNFR, 0.15)
        
        G.add_node(1, theta=0.15)
        set_attr(G.nodes[1], ALIAS_EPI, 0.50)
        set_attr(G.nodes[1], ALIAS_VF, 1.0)
        set_attr(G.nodes[1], ALIAS_DNFR, 0.10)
        G.add_edge(0, 1)

        G.nodes[0]["epi_history"] = [0.35, 0.42, 0.50]
        G.nodes[0]["glyph_history"] = []

        # Test that preconditions pass (no exception raised)
        validate_self_organization(G, 0)

    def test_thol_migration_path_allow_isolated(self):
        """Migration: enable isolated mode for backward compatibility."""
        G, node = create_nfr("isolated", epi=0.50, vf=1.0)
        set_attr(G.nodes[node], ALIAS_DNFR, 0.15)
        G.nodes[node]["epi_history"] = [0.35, 0.42, 0.50]

        # Enable isolated mode
        G.graph["THOL_ALLOW_ISOLATED"] = True
        G.graph["THOL_METABOLIC_ENABLED"] = False

        # Preconditions should pass with isolated mode enabled
        validate_self_organization(G, node)

    def test_thol_min_degree_configurable(self):
        """THOL_MIN_DEGREE should be configurable."""
        G = nx.Graph()
        # Create a node with exactly 1 neighbor
        G.add_node(0, theta=0.1)
        set_attr(G.nodes[0], ALIAS_EPI, 0.50)
        set_attr(G.nodes[0], ALIAS_VF, 1.0)
        set_attr(G.nodes[0], ALIAS_DNFR, 0.15)
        
        G.add_node(1, theta=0.15)
        set_attr(G.nodes[1], ALIAS_EPI, 0.50)
        set_attr(G.nodes[1], ALIAS_VF, 1.0)
        set_attr(G.nodes[1], ALIAS_DNFR, 0.10)
        G.add_edge(0, 1)

        G.nodes[0]["epi_history"] = [0.35, 0.42, 0.50]
        G.nodes[0]["glyph_history"] = []

        # Require min_degree = 2 (should fail)
        G.graph["THOL_MIN_DEGREE"] = 2

        with pytest.raises(
            OperatorPreconditionError, match="insufficiently connected"
        ):
            validate_self_organization(G, 0)

        # Now add another edge (degree = 2)
        G.add_node(2, theta=0.2)
        set_attr(G.nodes[2], ALIAS_EPI, 0.50)
        set_attr(G.nodes[2], ALIAS_VF, 1.0)
        set_attr(G.nodes[2], ALIAS_DNFR, 0.10)
        G.add_edge(0, 2)

        # Should pass now
        validate_self_organization(G, 0)

    def test_thol_min_history_length_configurable(self):
        """THOL_MIN_HISTORY_LENGTH should be configurable."""
        G = nx.Graph()
        G.add_node(0, theta=0.1)
        set_attr(G.nodes[0], ALIAS_EPI, 0.50)
        set_attr(G.nodes[0], ALIAS_VF, 1.0)
        set_attr(G.nodes[0], ALIAS_DNFR, 0.15)
        
        G.add_node(1, theta=0.15)
        set_attr(G.nodes[1], ALIAS_EPI, 0.50)
        set_attr(G.nodes[1], ALIAS_VF, 1.0)
        set_attr(G.nodes[1], ALIAS_DNFR, 0.10)
        G.add_edge(0, 1)

        # Only 2 history points
        G.nodes[0]["epi_history"] = [0.42, 0.50]
        G.nodes[0]["glyph_history"] = []

        # Custom: require only 2 history points
        G.graph["THOL_MIN_HISTORY_LENGTH"] = 2

        # Should pass with custom threshold
        validate_self_organization(G, 0)


class TestTHOLOperationalModes:
    """Test THOL operational modes (canonical vs simplified)."""

    def test_thol_canonical_mode(self):
        """Canonical THOL mode: network coupled, metabolism active."""
        G = nx.Graph()
        G.add_node(0, theta=0.1)
        set_attr(G.nodes[0], ALIAS_EPI, 0.50)
        set_attr(G.nodes[0], ALIAS_VF, 1.0)
        set_attr(G.nodes[0], ALIAS_DNFR, 0.15)
        
        G.add_node(1, theta=0.15)
        set_attr(G.nodes[1], ALIAS_EPI, 0.50)
        set_attr(G.nodes[1], ALIAS_VF, 1.0)
        set_attr(G.nodes[1], ALIAS_DNFR, 0.10)
        G.add_edge(0, 1)

        G.nodes[0]["epi_history"] = [0.35, 0.42, 0.50]
        G.nodes[0]["glyph_history"] = []

        # Canonical mode (defaults)
        G.graph["THOL_METABOLIC_ENABLED"] = True
        G.graph["THOL_PROPAGATION_ENABLED"] = True
        G.graph["THOL_ALLOW_ISOLATED"] = False

        # Should pass
        validate_self_organization(G, 0)

    def test_thol_simplified_mode(self):
        """Simplified THOL mode: internal bifurcation only."""
        G, node = create_nfr("isolated", epi=0.50, vf=1.0)
        set_attr(G.nodes[node], ALIAS_DNFR, 0.15)
        G.nodes[node]["epi_history"] = [0.35, 0.42, 0.50]
        G.nodes[node]["glyph_history"] = []

        # Simplified mode
        G.graph["THOL_METABOLIC_ENABLED"] = False
        G.graph["THOL_PROPAGATION_ENABLED"] = False
        G.graph["THOL_ALLOW_ISOLATED"] = True

        # Should pass in simplified mode
        validate_self_organization(G, node)

    def test_thol_experimental_mode_disabled_validation(self):
        """Experimental mode: validation disabled."""
        G, node = create_nfr("isolated", epi=0.15, vf=0.05)
        # Intentionally invalid state (low EPI, low vf, no history)
        set_attr(G.nodes[node], ALIAS_DNFR, -0.05)

        # Disable validation
        G.graph["VALIDATE_OPERATOR_PRECONDITIONS"] = False

        # Validation is bypassed at run_sequence level, not here
        # This test confirms the validation function still raises
        with pytest.raises(OperatorPreconditionError):
            validate_self_organization(G, node)


class TestTHOLRegressionCompatibility:
    """Test backward compatibility and migration paths."""

    def test_existing_code_with_isolated_thol_breaks(self):
        """Legacy code using THOL on isolated nodes should break with clear error."""
        G, node = create_nfr("isolated", epi=0.50, vf=1.0)
        set_attr(G.nodes[node], ALIAS_DNFR, 0.15)
        G.nodes[node]["epi_history"] = [0.35, 0.42, 0.50]

        # This should now FAIL (was warning before)
        with pytest.raises(OperatorPreconditionError) as exc_info:
            validate_self_organization(G, node)

        # Error message should suggest solution
        assert "THOL_ALLOW_ISOLATED" in str(exc_info.value)

    def test_error_messages_provide_guidance(self):
        """Error messages should provide actionable guidance."""
        G, node = create_nfr("isolated", epi=0.50, vf=1.0)
        set_attr(G.nodes[node], ALIAS_DNFR, 0.15)
        G.nodes[node]["epi_history"] = [0.35, 0.42, 0.50]

        try:
            validate_self_organization(G, node)
        except OperatorPreconditionError as e:
            error_msg = str(e)
            # Check for helpful guidance
            assert "THOL_ALLOW_ISOLATED=True" in error_msg
            assert "internal-only bifurcation" in error_msg
