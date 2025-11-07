"""Tests for IL (Coherence) strict precondition validation.

This module validates the implementation of TNFR.pdf §2.2.1 precondition
requirements for the Coherence (IL) operator:

1. EPI > 0 (active structural form)
2. EPI < maximum (non-saturated, room for stabilization)
3. νf > 0 (active structural frequency)
4. ΔNFR present (reorganization pressure to stabilize - warning if zero)
5. ΔNFR not critical (manageable instability - warning if too high)
6. Network connections (phase locking capability - warning if isolated)

This follows the same pattern established by AL (Emission) and EN (Reception)
strict validation, providing consistent precondition enforcement across
structural operators.
"""

import warnings

import pytest

from tnfr.alias import get_attr
from tnfr.config.thresholds import DNFR_IL_CRITICAL, EPI_IL_MAX, EPI_IL_MIN, VF_IL_MIN
from tnfr.constants.aliases import ALIAS_DNFR, ALIAS_EPI, ALIAS_VF
from tnfr.operators.definitions import Coherence
from tnfr.operators.preconditions import diagnose_coherence_readiness
from tnfr.operators.preconditions.coherence import validate_coherence_strict
from tnfr.structural import create_nfr


class TestCoherenceStrictPreconditions:
    """Test suite for IL strict precondition validation."""

    def test_validate_coherence_strict_success_normal_node(self):
        """Validation passes for normal active node with ΔNFR."""
        G, node = create_nfr("active", epi=0.5, vf=0.9)
        G.nodes[node]["dnfr"] = 0.1

        # Should not raise
        validate_coherence_strict(G, node)

    def test_validate_coherence_strict_success_low_epi(self):
        """Validation passes for node with low but non-zero EPI."""
        G, node = create_nfr("low_epi", epi=0.05, vf=0.8)
        G.nodes[node]["dnfr"] = 0.05

        # Should not raise (0.05 > 0.0 default min)
        validate_coherence_strict(G, node)

    def test_validate_coherence_strict_success_high_epi(self):
        """Validation passes for node just below EPI maximum."""
        G, node = create_nfr("high_epi", epi=0.95, vf=1.0)
        G.nodes[node]["dnfr"] = 0.08

        # Should not raise (0.95 < 1.0 default max)
        validate_coherence_strict(G, node)

    def test_validate_coherence_strict_fails_zero_epi(self):
        """Validation fails when EPI <= 0."""
        G, node = create_nfr("inactive", epi=0.0, vf=0.9)
        G.nodes[node]["dnfr"] = 0.1

        with pytest.raises(ValueError) as exc_info:
            validate_coherence_strict(G, node)

        error_msg = str(exc_info.value)
        assert "IL precondition failed" in error_msg
        assert "EPI=0.000" in error_msg
        assert "active structural form" in error_msg
        assert "AL (Emission)" in error_msg

    def test_validate_coherence_strict_fails_negative_epi(self):
        """Validation fails when EPI is negative."""
        G, node = create_nfr("negative_epi", epi=-0.1, vf=0.9)
        G.nodes[node]["dnfr"] = 0.1

        with pytest.raises(ValueError) as exc_info:
            validate_coherence_strict(G, node)

        error_msg = str(exc_info.value)
        assert "IL precondition failed" in error_msg
        assert "EPI=-0.100" in error_msg

    def test_validate_coherence_strict_fails_saturated_epi(self):
        """Validation fails when EPI >= maximum."""
        G, node = create_nfr("saturated", epi=1.0, vf=1.0)
        G.nodes[node]["dnfr"] = 0.05

        with pytest.raises(ValueError) as exc_info:
            validate_coherence_strict(G, node)

        error_msg = str(exc_info.value)
        assert "IL precondition failed" in error_msg
        assert "EPI=1.000" in error_msg
        assert ">= 1.0" in error_msg
        assert "saturated" in error_msg
        assert "NUL (Contraction)" in error_msg

    def test_validate_coherence_strict_fails_zero_vf(self):
        """Validation fails when νf <= 0."""
        G, node = create_nfr("frozen", epi=0.5, vf=0.0)
        G.nodes[node]["dnfr"] = 0.1

        with pytest.raises(ValueError) as exc_info:
            validate_coherence_strict(G, node)

        error_msg = str(exc_info.value)
        assert "IL precondition failed" in error_msg
        assert "νf=0.000" in error_msg
        assert "Structural frequency too low" in error_msg
        assert "AL (Emission)" in error_msg or "NAV (Transition)" in error_msg

    def test_validate_coherence_strict_fails_negative_vf(self):
        """Validation fails when νf is negative."""
        G, node = create_nfr("negative_vf", epi=0.5, vf=0.5)
        # Manually set negative vf after creation
        from tnfr.constants.aliases import ALIAS_VF
        G.nodes[node][ALIAS_VF[0]] = -0.1

        with pytest.raises(ValueError) as exc_info:
            validate_coherence_strict(G, node)

        error_msg = str(exc_info.value)
        assert "IL precondition failed" in error_msg
        assert "νf=-0.100" in error_msg

    def test_validate_coherence_strict_warns_zero_dnfr(self):
        """Validation warns when ΔNFR == 0."""
        G, node = create_nfr("no_pressure", epi=0.5, vf=0.9)
        G.nodes[node]["dnfr"] = 0.0

        # Should warn but not fail
        with pytest.warns(UserWarning) as warning_info:
            validate_coherence_strict(G, node)

        assert len(warning_info) > 0
        warning_msg = str(warning_info[0].message)
        assert "IL warning" in warning_msg
        assert "ΔNFR=0" in warning_msg
        assert "redundant" in warning_msg.lower()

    def test_validate_coherence_strict_warns_critical_dnfr(self):
        """Validation warns when ΔNFR > critical threshold."""
        G, node = create_nfr("unstable", epi=0.5, vf=0.9)
        G.nodes[node]["dnfr"] = 0.85  # > 0.8 default critical

        # Should warn but not fail
        with pytest.warns(UserWarning) as warning_info:
            validate_coherence_strict(G, node)

        assert len(warning_info) > 0
        warning_msg = str(warning_info[0].message)
        assert "IL warning" in warning_msg
        assert "ΔNFR=0.850" in warning_msg
        assert "0.8" in warning_msg
        assert "OZ (Dissonance)" in warning_msg

    def test_validate_coherence_strict_warns_isolated_node(self):
        """Validation warns for isolated node in multi-node network."""
        G, node = create_nfr("isolated", epi=0.5, vf=0.9)
        G.nodes[node]["dnfr"] = 0.1

        # Add another node to make it multi-node network
        G.add_node("other")

        # Should warn but not fail
        with pytest.warns(UserWarning) as warning_info:
            validate_coherence_strict(G, node)

        assert len(warning_info) > 0
        warning_msg = str(warning_info[0].message)
        assert "IL warning" in warning_msg
        assert "isolated" in warning_msg.lower()
        assert "degree=0" in warning_msg
        assert "UM (Coupling)" in warning_msg

    def test_validate_coherence_strict_no_warning_single_node(self):
        """No isolation warning for single-node network."""
        G, node = create_nfr("solo", epi=0.5, vf=0.9)
        G.nodes[node]["dnfr"] = 0.1

        # Single node network - no isolation warning expected
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # Turn warnings into errors
            try:
                validate_coherence_strict(G, node)
            except UserWarning as w:
                if "isolated" in str(w).lower():
                    pytest.fail("Should not warn about isolation in single-node network")

    def test_validate_coherence_strict_multiple_warnings(self):
        """Validation can issue multiple warnings for different issues."""
        G, node = create_nfr("problematic", epi=0.5, vf=0.9)
        G.nodes[node]["dnfr"] = 0.0  # Will warn: zero ΔNFR

        # Make it isolated
        G.add_node("other")

        # Should get multiple warnings
        with pytest.warns(UserWarning) as warning_info:
            validate_coherence_strict(G, node)

        # Should have warnings about zero ΔNFR and isolation
        assert len(warning_info) >= 2
        warnings_text = " ".join(str(w.message) for w in warning_info)
        assert "ΔNFR=0" in warnings_text
        assert "isolated" in warnings_text.lower()

    def test_validate_coherence_strict_custom_thresholds_via_metadata(self):
        """Custom thresholds can be set via graph metadata."""
        G, node = create_nfr("custom", epi=0.7, vf=0.9)
        G.nodes[node]["dnfr"] = 0.1

        # Set custom EPI maximum threshold
        G.graph["IL_PRECONDITIONS"] = {
            "max_epi": 0.5,  # Lower than default 1.0
        }

        # Should fail with custom threshold
        with pytest.raises(ValueError) as exc_info:
            validate_coherence_strict(G, node)

        error_msg = str(exc_info.value)
        assert "EPI=0.700" in error_msg
        assert ">= 0.5" in error_msg

    def test_validate_coherence_strict_custom_min_epi(self):
        """Custom minimum EPI threshold via graph metadata."""
        G, node = create_nfr("custom_min", epi=0.05, vf=0.9)
        G.nodes[node]["dnfr"] = 0.1

        # Set custom minimum EPI
        G.graph["IL_PRECONDITIONS"] = {
            "min_epi": 0.1,  # Higher than default 0.0
        }

        # Should fail with custom threshold
        with pytest.raises(ValueError) as exc_info:
            validate_coherence_strict(G, node)

        error_msg = str(exc_info.value)
        assert "EPI=0.050" in error_msg
        assert "<= 0.1" in error_msg

    def test_validate_coherence_strict_can_disable_warnings(self):
        """Warnings can be disabled via graph metadata."""
        G, node = create_nfr("no_warnings", epi=0.5, vf=0.9)
        G.nodes[node]["dnfr"] = 0.0

        # Disable warnings
        G.graph["IL_PRECONDITIONS"] = {
            "warn_zero_dnfr": False,
            "warn_isolated": False,
        }

        # Should not warn
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # Turn warnings into errors
            validate_coherence_strict(G, node)  # Should pass without warnings

    def test_validate_coherence_strict_boundary_conditions(self):
        """Test boundary conditions for thresholds."""
        # EPI exactly at min (should pass)
        G1, node1 = create_nfr("at_min", epi=0.001, vf=0.9)
        G1.nodes[node1]["dnfr"] = 0.1
        validate_coherence_strict(G1, node1)  # Should not raise

        # EPI exactly at max (should fail)
        G2, node2 = create_nfr("at_max", epi=1.0, vf=0.9)
        G2.nodes[node2]["dnfr"] = 0.1
        with pytest.raises(ValueError):
            validate_coherence_strict(G2, node2)

        # νf exactly at min (should fail)
        G3, node3 = create_nfr("vf_at_min", epi=0.5, vf=0.0)
        G3.nodes[node3]["dnfr"] = 0.1
        with pytest.raises(ValueError):
            validate_coherence_strict(G3, node3)


class TestCoherenceDiagnostics:
    """Test suite for IL diagnostic function."""

    def test_diagnose_coherence_readiness_ready_node(self):
        """Diagnostic reports ready for valid node."""
        G, node = create_nfr("ready", epi=0.5, vf=0.9)
        G.nodes[node]["dnfr"] = 0.1

        report = diagnose_coherence_readiness(G, node)

        assert report["node"] == node
        assert report["ready"] is True
        assert report["checks"]["epi_active"] is True
        assert report["checks"]["epi_not_saturated"] is True
        assert report["checks"]["vf_active"] is True
        assert report["checks"]["dnfr_present"] is True
        assert report["values"]["epi"] == 0.5
        assert report["values"]["vf"] == 0.9
        assert report["values"]["dnfr"] == 0.1
        assert any("✓" in r for r in report["recommendations"])

    def test_diagnose_coherence_readiness_inactive_node(self):
        """Diagnostic reports not ready for inactive node."""
        G, node = create_nfr("inactive", epi=0.0, vf=0.9)
        G.nodes[node]["dnfr"] = 0.1

        report = diagnose_coherence_readiness(G, node)

        assert report["ready"] is False
        assert report["checks"]["epi_active"] is False
        assert "AL (Emission)" in " ".join(report["recommendations"])

    def test_diagnose_coherence_readiness_saturated_node(self):
        """Diagnostic reports not ready for saturated node."""
        G, node = create_nfr("saturated", epi=1.0, vf=0.9)
        G.nodes[node]["dnfr"] = 0.1

        report = diagnose_coherence_readiness(G, node)

        assert report["ready"] is False
        assert report["checks"]["epi_not_saturated"] is False
        assert "NUL (Contraction)" in " ".join(report["recommendations"])

    def test_diagnose_coherence_readiness_frozen_node(self):
        """Diagnostic reports not ready for frozen node (νf = 0)."""
        G, node = create_nfr("frozen", epi=0.5, vf=0.0)
        G.nodes[node]["dnfr"] = 0.1

        report = diagnose_coherence_readiness(G, node)

        assert report["ready"] is False
        assert report["checks"]["vf_active"] is False
        assert "NAV (Transition)" in " ".join(report["recommendations"])

    def test_diagnose_coherence_readiness_zero_dnfr_warning(self):
        """Diagnostic warns about zero ΔNFR."""
        G, node = create_nfr("no_pressure", epi=0.5, vf=0.9)
        G.nodes[node]["dnfr"] = 0.0

        report = diagnose_coherence_readiness(G, node)

        # Still technically ready (critical checks pass)
        assert report["ready"] is True
        assert report["checks"]["dnfr_present"] is False
        assert any("ΔNFR=0" in r for r in report["recommendations"])

    def test_diagnose_coherence_readiness_critical_dnfr_warning(self):
        """Diagnostic warns about critical ΔNFR."""
        G, node = create_nfr("unstable", epi=0.5, vf=0.9)
        G.nodes[node]["dnfr"] = 0.85

        report = diagnose_coherence_readiness(G, node)

        assert report["ready"] is True
        assert report["checks"]["dnfr_not_critical"] is False
        assert any("OZ (Dissonance)" in r for r in report["recommendations"])

    def test_diagnose_coherence_readiness_isolated_warning(self):
        """Diagnostic warns about isolated node."""
        G, node = create_nfr("isolated", epi=0.5, vf=0.9)
        G.nodes[node]["dnfr"] = 0.1
        G.add_node("other")

        report = diagnose_coherence_readiness(G, node)

        assert report["ready"] is True
        assert report["checks"]["has_connections"] is False
        assert any("UM (Coupling)" in r for r in report["recommendations"])

    def test_diagnose_coherence_readiness_multiple_issues(self):
        """Diagnostic reports multiple issues."""
        G, node = create_nfr("problematic", epi=0.0, vf=0.0)
        G.nodes[node]["dnfr"] = 0.0

        report = diagnose_coherence_readiness(G, node)

        assert report["ready"] is False
        assert report["checks"]["epi_active"] is False
        assert report["checks"]["vf_active"] is False
        assert len(report["recommendations"]) >= 2

    def test_diagnose_coherence_readiness_values_accurate(self):
        """Diagnostic reports accurate node values."""
        G, node = create_nfr("test", epi=0.42, vf=0.87)
        G.nodes[node]["dnfr"] = 0.23
        G.add_edge(node, "neighbor")

        report = diagnose_coherence_readiness(G, node)

        assert report["values"]["epi"] == 0.42
        assert report["values"]["vf"] == 0.87
        assert report["values"]["dnfr"] == 0.23
        assert report["values"]["degree"] == 1


class TestCoherenceOperatorIntegration:
    """Test integration of preconditions with Coherence operator."""

    def test_coherence_operator_calls_validation(self):
        """Coherence operator validates preconditions before execution."""
        G, node = create_nfr("invalid", epi=0.5, vf=0.9)
        # Manually set zero EPI after creation to test precondition
        from tnfr.constants.aliases import ALIAS_EPI
        G.nodes[node][ALIAS_EPI[0]] = 0.0
        
        # Enable precondition validation
        G.graph["VALIDATE_OPERATOR_PRECONDITIONS"] = True

        # Coherence should raise ValueError from precondition check
        with pytest.raises(ValueError) as exc_info:
            Coherence()(G, node)

        error_msg = str(exc_info.value)
        assert "IL precondition failed" in error_msg

    def test_coherence_operator_succeeds_with_valid_preconditions(self):
        """Coherence operator succeeds when preconditions are met."""
        G, node = create_nfr("valid", epi=0.5, vf=0.9)
        G.nodes[node]["dnfr"] = 0.1
        
        # Enable precondition validation
        G.graph["VALIDATE_OPERATOR_PRECONDITIONS"] = True

        # Should succeed
        Coherence()(G, node)

        # Verify IL was applied (ΔNFR should be reduced)
        dnfr_after = float(get_attr(G.nodes[node], ALIAS_DNFR, 0.0))
        assert dnfr_after < 0.1  # Should be reduced

    def test_coherence_operator_with_warnings(self):
        """Coherence operator issues warnings for suboptimal states."""
        G, node = create_nfr("suboptimal", epi=0.5, vf=0.9)
        G.nodes[node]["dnfr"] = 0.0  # Zero ΔNFR - should warn
        
        # Enable precondition validation
        G.graph["VALIDATE_OPERATOR_PRECONDITIONS"] = True

        # Should warn but not fail
        with pytest.warns(UserWarning) as warning_info:
            Coherence()(G, node)

        assert len(warning_info) > 0
        assert any("ΔNFR=0" in str(w.message) for w in warning_info)
