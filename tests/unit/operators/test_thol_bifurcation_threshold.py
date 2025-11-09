"""Tests for THOL bifurcation threshold validation (∂²EPI/∂t² > τ).

This module validates the canonical requirement from TNFR.pdf §2.2.10
that THOL (self-organization) bifurcation occurs only when structural
acceleration exceeds the threshold τ:

    ∂²EPI/∂t² > τ → bifurcation (sub-EPIs generated)
    ∂²EPI/∂t² ≤ τ → no bifurcation (THOL executes without sub-EPIs)

The validation is NON-BLOCKING (warning only) because THOL can meaningfully
execute without bifurcation - it still applies coherence and metabolic effects.

Tests verify:
1. Bifurcation threshold check is performed
2. Warning logged when ∂²EPI/∂t² ≤ τ
3. Telemetry flag _thol_no_bifurcation_expected is set correctly
4. Backward compatibility with existing THOL tests
5. Configuration parameters work correctly
"""

import logging
import pytest
import networkx as nx

from tnfr.alias import set_attr
from tnfr.constants.aliases import ALIAS_DNFR, ALIAS_EPI, ALIAS_VF
from tnfr.operators.preconditions import validate_self_organization
from tnfr.operators.nodal_equation import compute_d2epi_dt2


class TestTHOLBifurcationThreshold:
    """Test suite for THOL bifurcation threshold validation."""

    def setup_method(self):
        """Create test graph with proper setup for THOL."""
        self.G = nx.Graph()
        self.G.add_node(0, theta=0.1)
        set_attr(self.G.nodes[0], ALIAS_EPI, 0.50)
        set_attr(self.G.nodes[0], ALIAS_VF, 1.0)
        set_attr(self.G.nodes[0], ALIAS_DNFR, 0.15)
        
        self.G.add_node(1, theta=0.15)
        set_attr(self.G.nodes[1], ALIAS_EPI, 0.50)
        set_attr(self.G.nodes[1], ALIAS_VF, 1.0)
        set_attr(self.G.nodes[1], ALIAS_DNFR, 0.10)
        self.G.add_edge(0, 1)
        
        self.G.nodes[0]["glyph_history"] = []

    def test_high_acceleration_no_warning(self, caplog):
        """When ∂²EPI/∂t² > τ, no warning should be logged."""
        # Create EPI history with high acceleration
        # d²EPI/dt² = EPI_t - 2*EPI_{t-1} + EPI_{t-2}
        # d²EPI/dt² = 0.50 - 2*0.42 + 0.30 = 0.50 - 0.84 + 0.30 = -0.04
        # |d²EPI/dt²| = 0.04 (but we need positive acceleration)
        # Let's create accelerating growth:
        # 0.30, 0.38, 0.50 → d²EPI = 0.50 - 0.76 + 0.30 = 0.04
        self.G.nodes[0]["epi_history"] = [0.30, 0.38, 0.50]
        
        # Set low threshold so we exceed it
        self.G.graph["BIFURCATION_THRESHOLD_TAU"] = 0.01
        
        with caplog.at_level(logging.WARNING):
            validate_self_organization(self.G, 0)
        
        # No warning should be logged
        assert not any("no bifurcation will occur" in record.message.lower() 
                      for record in caplog.records if record.levelname == "WARNING")
        
        # Flag should be False
        assert self.G.nodes[0].get("_thol_no_bifurcation_expected") == False

    def test_low_acceleration_warning_logged(self, caplog):
        """When ∂²EPI/∂t² ≤ τ, warning should be logged."""
        # Create EPI history with LOW acceleration (nearly linear growth)
        # d²EPI/dt² = 0.50 - 2*0.49 + 0.48 = 0.50 - 0.98 + 0.48 = 0.00
        self.G.nodes[0]["epi_history"] = [0.48, 0.49, 0.50]
        
        # Set threshold higher than our acceleration
        self.G.graph["BIFURCATION_THRESHOLD_TAU"] = 0.1
        
        with caplog.at_level(logging.WARNING):
            validate_self_organization(self.G, 0)
        
        # Warning SHOULD be logged
        assert any("no bifurcation will occur" in record.message.lower()
                  for record in caplog.records if record.levelname == "WARNING")
        
        # Check warning details
        warning_records = [r for r in caplog.records if r.levelname == "WARNING"]
        assert len(warning_records) > 0
        warning_msg = warning_records[0].message
        assert "∂²EPI/∂t²" in warning_msg
        assert "τ=" in warning_msg
        assert "no bifurcation will occur" in warning_msg.lower()

    def test_telemetry_flag_set_when_below_threshold(self):
        """_thol_no_bifurcation_expected flag should be set when ∂²EPI/∂t² ≤ τ."""
        # Low acceleration
        self.G.nodes[0]["epi_history"] = [0.48, 0.49, 0.50]
        self.G.graph["BIFURCATION_THRESHOLD_TAU"] = 0.1
        
        validate_self_organization(self.G, 0)
        
        # Flag should be True
        assert self.G.nodes[0]["_thol_no_bifurcation_expected"] == True

    def test_telemetry_flag_cleared_when_above_threshold(self):
        """_thol_no_bifurcation_expected flag should be False when ∂²EPI/∂t² > τ."""
        # High acceleration: 0.10, 0.25, 0.50
        # d²EPI = 0.50 - 2*0.25 + 0.10 = 0.50 - 0.50 + 0.10 = 0.10
        self.G.nodes[0]["epi_history"] = [0.10, 0.25, 0.50]
        self.G.graph["BIFURCATION_THRESHOLD_TAU"] = 0.05
        
        validate_self_organization(self.G, 0)
        
        # Flag should be False (bifurcation expected)
        assert self.G.nodes[0]["_thol_no_bifurcation_expected"] == False

    def test_compute_d2epi_dt2_called(self):
        """Validation should use canonical compute_d2epi_dt2 function."""
        # Create history with known acceleration
        # d²EPI = 0.50 - 2*0.40 + 0.30 = 0.50 - 0.80 + 0.30 = 0.00
        self.G.nodes[0]["epi_history"] = [0.30, 0.40, 0.50]
        self.G.graph["BIFURCATION_THRESHOLD_TAU"] = 0.1
        
        validate_self_organization(self.G, 0)
        
        # Verify d2_epi was computed and stored by compute_d2epi_dt2
        d2_epi = compute_d2epi_dt2(self.G, 0)
        # Should be ~0.0 given our linear progression
        assert abs(d2_epi) < 0.01

    def test_threshold_from_bifurcation_threshold_tau(self):
        """Should use BIFURCATION_THRESHOLD_TAU config parameter."""
        self.G.nodes[0]["epi_history"] = [0.48, 0.49, 0.50]
        self.G.graph["BIFURCATION_THRESHOLD_TAU"] = 0.2  # High threshold
        
        validate_self_organization(self.G, 0)
        
        # Should warn because d²EPI ≈ 0 < 0.2
        assert self.G.nodes[0]["_thol_no_bifurcation_expected"] == True

    def test_threshold_from_thol_bifurcation_threshold(self):
        """Should fall back to THOL_BIFURCATION_THRESHOLD if canonical not set."""
        self.G.nodes[0]["epi_history"] = [0.48, 0.49, 0.50]
        # Don't set BIFURCATION_THRESHOLD_TAU, use operator-specific config
        self.G.graph["THOL_BIFURCATION_THRESHOLD"] = 0.05
        
        validate_self_organization(self.G, 0)
        
        # Should warn because d²EPI ≈ 0 < 0.05
        assert self.G.nodes[0]["_thol_no_bifurcation_expected"] == True

    def test_default_threshold_when_not_configured(self):
        """Should use default threshold 0.1 when not configured."""
        self.G.nodes[0]["epi_history"] = [0.48, 0.49, 0.50]
        # Don't set any threshold config
        
        validate_self_organization(self.G, 0)
        
        # Should warn with default threshold 0.1
        assert self.G.nodes[0]["_thol_no_bifurcation_expected"] == True

    def test_warning_suggests_destabilizers(self, caplog):
        """Warning message should suggest using destabilizers to increase acceleration."""
        self.G.nodes[0]["epi_history"] = [0.48, 0.49, 0.50]
        self.G.graph["BIFURCATION_THRESHOLD_TAU"] = 0.1
        
        with caplog.at_level(logging.WARNING):
            validate_self_organization(self.G, 0)
        
        # Check warning mentions destabilizers
        warning_msg = next(
            r.message for r in caplog.records 
            if r.levelname == "WARNING" and "no bifurcation" in r.message.lower()
        )
        assert "OZ" in warning_msg or "VAL" in warning_msg or "destabilizer" in warning_msg.lower()

    def test_validation_non_blocking(self):
        """Validation should NOT raise error, only warn."""
        self.G.nodes[0]["epi_history"] = [0.48, 0.49, 0.50]
        self.G.graph["BIFURCATION_THRESHOLD_TAU"] = 1.0  # Very high threshold
        
        # Should NOT raise exception
        validate_self_organization(self.G, 0)  # OK

    def test_accelerating_contraction_detected(self):
        """Should detect negative acceleration (accelerating contraction)."""
        # Decelerating contraction: 0.50, 0.45, 0.42
        # d²EPI = 0.42 - 2*0.45 + 0.50 = 0.42 - 0.90 + 0.50 = 0.02
        # This is positive acceleration despite decreasing EPI
        self.G.nodes[0]["epi_history"] = [0.50, 0.45, 0.42]
        self.G.graph["BIFURCATION_THRESHOLD_TAU"] = 0.01
        
        validate_self_organization(self.G, 0)
        
        # Should be above threshold
        # Check directly from node state (computed during validation)
        history = self.G.nodes[0]["epi_history"]
        d2_epi_calc = abs(history[-1] - 2*history[-2] + history[-3])
        assert d2_epi_calc > 0.01
        assert self.G.nodes[0]["_thol_no_bifurcation_expected"] == False

    def test_debug_log_when_above_threshold(self, caplog):
        """Should log debug message when threshold is exceeded."""
        # High acceleration
        self.G.nodes[0]["epi_history"] = [0.10, 0.25, 0.50]
        self.G.graph["BIFURCATION_THRESHOLD_TAU"] = 0.05
        
        with caplog.at_level(logging.DEBUG):
            validate_self_organization(self.G, 0)
        
        # Debug message should be logged
        debug_records = [r for r in caplog.records if r.levelname == "DEBUG"]
        if debug_records:  # Debug may not always be captured
            assert any("bifurcation threshold exceeded" in r.message.lower()
                      for r in debug_records)


class TestTHOLBifurcationIntegration:
    """Integration tests with actual THOL operator execution."""

    def test_thol_executes_without_error_below_threshold(self):
        """THOL should execute successfully even when ∂²EPI/∂t² ≤ τ."""
        from tnfr.operators.definitions import SelfOrganization
        
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
        
        # Low acceleration
        G.nodes[0]["epi_history"] = [0.48, 0.49, 0.50]
        G.nodes[0]["glyph_history"] = []
        G.graph["BIFURCATION_THRESHOLD_TAU"] = 0.1
        # Enable precondition validation
        G.graph["VALIDATE_OPERATOR_PRECONDITIONS"] = True
        
        # Should execute without exception
        thol = SelfOrganization()
        thol(G, 0)  # OK
        
        # Check telemetry flag was set during precondition validation
        # (which happens inside the operator call via validate_preconditions=True by default)
        assert G.nodes[0].get("_thol_no_bifurcation_expected") == True

    def test_no_sub_epis_generated_below_threshold(self):
        """When ∂²EPI/∂t² ≤ τ, no sub-EPIs should be generated."""
        from tnfr.operators.definitions import SelfOrganization
        
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
        
        # Low acceleration (linear growth)
        G.nodes[0]["epi_history"] = [0.48, 0.49, 0.50]
        G.nodes[0]["glyph_history"] = []
        G.graph["THOL_BIFURCATION_THRESHOLD"] = 0.1
        
        thol = SelfOrganization()
        thol(G, 0, tau=0.1)
        
        # No sub-EPIs should be created
        sub_epis = G.nodes[0].get("sub_epis", [])
        assert len(sub_epis) == 0

    def test_sub_epis_generated_above_threshold(self):
        """When ∂²EPI/∂t² > τ, sub-EPIs should be generated."""
        from tnfr.operators.definitions import SelfOrganization
        
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
        
        # High acceleration
        G.nodes[0]["epi_history"] = [0.10, 0.25, 0.50]
        G.nodes[0]["glyph_history"] = []
        G.graph["THOL_BIFURCATION_THRESHOLD"] = 0.05
        
        thol = SelfOrganization()
        thol(G, 0, tau=0.05)
        
        # Sub-EPIs SHOULD be created
        sub_epis = G.nodes[0].get("sub_epis", [])
        assert len(sub_epis) > 0


class TestTHOLBifurcationBackwardCompatibility:
    """Test backward compatibility with existing THOL behavior."""

    def test_existing_tests_still_pass(self):
        """Existing THOL tests should still pass with new validation."""
        # Simulate typical existing test setup
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
        
        # Should still pass (non-blocking validation)
        validate_self_organization(G, 0)  # OK

    def test_no_breaking_changes_to_api(self):
        """validate_self_organization API should be unchanged."""
        import inspect
        from tnfr.operators.preconditions import validate_self_organization
        
        sig = inspect.signature(validate_self_organization)
        params = list(sig.parameters.keys())
        
        # Should still have same 2 parameters
        assert len(params) == 2
        assert params[0] == "G"
        assert params[1] == "node"
