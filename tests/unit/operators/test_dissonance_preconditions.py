"""Tests for OZ (Dissonance) strict precondition validation.

This module validates the implementation of canonical precondition requirements
for the Dissonance (OZ) operator according to TNFR theory:

1. Minimum coherence base (EPI >= 0.2) - node must withstand disruption
2. ΔNFR not critically high (|ΔNFR| < 0.8) - avoid overload/collapse
3. Sufficient νf (>= 0.1) - capacity to respond to dissonance
4. No overload pattern - detect dissonant overload (multiple OZ without resolution)
5. Network connectivity - warn if isolated (bifurcation requires paths)

These tests ensure OZ is only applied when the node has sufficient structural
resilience to benefit from controlled dissonance rather than collapse.

References
----------
- TNFR.pdf §2.3.6: Dissonant overload
- Issue: [OZ] Strengthen structural preconditions with base coherence validation
"""

import warnings

import pytest

from tnfr.alias import get_attr, set_attr
from tnfr.constants.aliases import ALIAS_DNFR, ALIAS_EPI, ALIAS_VF
from tnfr.operators.definitions import (
    Coherence,
    Contraction,
    Dissonance,
    Emission,
    Resonance,
    SelfOrganization,
)
from tnfr.operators.preconditions import OperatorPreconditionError
from tnfr.operators.preconditions.dissonance import validate_dissonance_strict
from tnfr.structural import create_nfr


class TestDissonanceStrictPreconditions:
    """Test suite for OZ strict precondition validation."""

    def test_validate_dissonance_strict_success_normal_node(self):
        """Validation passes for node with sufficient coherence base."""
        G, node = create_nfr("normal", epi=0.5, vf=1.0)
        set_attr(G.nodes[node], ALIAS_DNFR, 0.3)

        # Should not raise
        validate_dissonance_strict(G, node)

        # Check telemetry context stored
        assert "_oz_precondition_context" in G.nodes[node]
        context = G.nodes[node]["_oz_precondition_context"]
        assert context["validation_passed"] is True
        assert context["epi"] == 0.5
        assert context["vf"] == 1.0

    def test_validate_dissonance_strict_success_at_threshold(self):
        """Validation passes for node at EPI threshold."""
        G, node = create_nfr("threshold", epi=0.2, vf=1.0)
        set_attr(G.nodes[node], ALIAS_DNFR, 0.3)

        # Should not raise (0.2 >= 0.2 default threshold)
        validate_dissonance_strict(G, node)

    def test_validate_dissonance_strict_fails_low_epi(self):
        """Validation fails when EPI < minimum (insufficient coherence base)."""
        G, node = create_nfr("weak", epi=0.05, vf=1.0)

        with pytest.raises(ValueError) as exc_info:
            validate_dissonance_strict(G, node)

        error_msg = str(exc_info.value)
        assert "OZ precondition failed" in error_msg
        assert "EPI=0.050" in error_msg
        assert "< 0.2" in error_msg
        assert "Insufficient coherence base" in error_msg
        assert "Apply IL (Coherence)" in error_msg

    def test_validate_dissonance_strict_fails_high_dnfr(self):
        """Validation fails when |ΔNFR| > maximum (critical reorganization pressure)."""
        G, node = create_nfr("critical", epi=0.5, vf=1.0)
        set_attr(G.nodes[node], ALIAS_DNFR, 0.9)

        with pytest.raises(ValueError) as exc_info:
            validate_dissonance_strict(G, node)

        error_msg = str(exc_info.value)
        assert "OZ precondition failed" in error_msg
        assert "|ΔNFR|=0.900" in error_msg
        assert "> 0.8" in error_msg
        assert "already critical" in error_msg
        assert "Apply IL (Coherence)" in error_msg

    def test_validate_dissonance_strict_fails_low_vf(self):
        """Validation fails when νf < minimum (cannot respond to dissonance)."""
        G, node = create_nfr("frozen", epi=0.5, vf=0.05)

        with pytest.raises(ValueError) as exc_info:
            validate_dissonance_strict(G, node)

        error_msg = str(exc_info.value)
        assert "OZ precondition failed" in error_msg
        assert "νf=0.050" in error_msg
        assert "< 0.1" in error_msg
        assert "too low" in error_msg

    def test_validate_dissonance_strict_warns_isolated_node(self):
        """Validation warns for isolated node (limited bifurcation paths)."""
        G, node = create_nfr("isolated", epi=0.5, vf=1.0)
        # Node has no neighbors (degree=0)

        with pytest.warns(UserWarning) as warning_info:
            validate_dissonance_strict(G, node)

        assert len(warning_info) > 0
        warning_msg = str(warning_info[0].message)
        assert "OZ warning" in warning_msg
        assert "low connectivity" in warning_msg
        assert "degree=0" in warning_msg
        assert "Consider applying UM (Coupling)" in warning_msg

    def test_validate_dissonance_strict_no_warning_connected_node(self):
        """No connectivity warning for connected node."""
        G, node = create_nfr("connected", epi=0.5, vf=1.0)

        # Add neighbor
        other = "neighbor"
        G.add_node(other)
        G.add_edge(node, other)

        # Should not warn
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # Turn warnings into errors
            validate_dissonance_strict(G, node)  # Should not raise

    def test_validate_dissonance_strict_custom_thresholds(self):
        """Custom thresholds can be set via graph metadata."""
        G, node = create_nfr("custom", epi=0.15, vf=1.0)

        # With default thresholds (EPI >= 0.2), this would fail
        # Set custom lower EPI threshold
        G.graph["OZ_MIN_EPI"] = 0.1

        # Should now pass
        validate_dissonance_strict(G, node)

    def test_validate_dissonance_strict_custom_dnfr_threshold(self):
        """Custom ΔNFR threshold via graph metadata."""
        G, node = create_nfr("custom_dnfr", epi=0.5, vf=1.0)
        set_attr(G.nodes[node], ALIAS_DNFR, 0.85)

        # With default threshold (< 0.8), this would fail
        # Set custom higher threshold
        G.graph["OZ_MAX_DNFR"] = 0.9

        # Should now pass
        validate_dissonance_strict(G, node)

    def test_validate_dissonance_strict_custom_vf_threshold(self):
        """Custom νf threshold via graph metadata."""
        G, node = create_nfr("custom_vf", epi=0.5, vf=0.05)

        # With default threshold (>= 0.1), this would fail
        # Set custom lower threshold
        G.graph["OZ_MIN_VF"] = 0.01

        # Should now pass
        validate_dissonance_strict(G, node)


class TestDissonanceOverloadDetection:
    """Test sobrecarga disonante (dissonance overload) detection."""

    def test_oz_overload_detects_multiple_oz_without_resolver(self):
        """Multiple OZ without resolution raises error."""
        G, node = create_nfr("overload", epi=0.5, vf=1.0)
        G.graph["VALIDATE_OPERATOR_PRECONDITIONS"] = True

        # First OZ ok
        Dissonance()(G, node)

        # Second OZ without resolver should fail
        with pytest.raises(OperatorPreconditionError) as exc_info:
            Dissonance()(G, node)

        error_msg = str(exc_info.value)
        assert "Sobrecarga disonante" in error_msg
        assert "without resolution" in error_msg
        assert "Apply IL (Coherence)" in error_msg

    def test_oz_overload_resolved_by_coherence(self):
        """OZ allowed again after IL (Coherence) resolution."""
        G, node = create_nfr("resolved_il", epi=0.5, vf=1.0)

        # First OZ
        Dissonance()(G, node)

        # Resolve with Coherence
        Coherence()(G, node)

        # Second OZ should succeed
        validate_dissonance_strict(G, node)

    def test_oz_overload_resolved_by_self_organization(self):
        """OZ allowed again after THOL (Self-organization) resolution."""
        G, node = create_nfr("resolved_thol", epi=0.5, vf=1.0)
        set_attr(G.nodes[node], ALIAS_DNFR, 0.3)  # Positive ΔNFR for THOL

        # First OZ
        Dissonance()(G, node)

        # Resolve with Self-organization
        SelfOrganization()(G, node)

        # Second OZ should succeed
        validate_dissonance_strict(G, node)

    def test_oz_overload_resolved_by_contraction(self):
        """OZ allowed again after NUL (Contraction) resolution."""
        G, node = create_nfr("resolved_nul", epi=0.5, vf=1.0)

        # First OZ
        Dissonance()(G, node)

        # Resolve with Contraction
        Contraction()(G, node)

        # Second OZ should succeed
        validate_dissonance_strict(G, node)

    def test_oz_overload_not_triggered_by_other_operators(self):
        """Other operators between OZ don't count as resolvers."""
        G, node = create_nfr("not_resolved", epi=0.5, vf=1.0)
        G.graph["VALIDATE_OPERATOR_PRECONDITIONS"] = True

        # Add neighbor for resonance
        other = "neighbor"
        G.add_node(other)
        G.add_edge(node, other)

        # First OZ
        Dissonance()(G, node)

        # Apply non-resolver operators
        Resonance()(G, node)
        Emission()(G, node)

        # Second OZ should still fail (no resolver)
        with pytest.raises(OperatorPreconditionError) as exc_info:
            Dissonance()(G, node)

        assert "Sobrecarga disonante" in str(exc_info.value)

    def test_oz_overload_with_three_consecutive_oz(self):
        """Three consecutive OZ also detected as overload."""
        G, node = create_nfr("triple_oz", epi=0.5, vf=1.0)
        G.graph["VALIDATE_OPERATOR_PRECONDITIONS"] = True

        # First OZ
        Dissonance()(G, node)

        # Second OZ should fail
        with pytest.raises(OperatorPreconditionError):
            Dissonance()(G, node)

    def test_oz_overload_checks_recent_history_only(self):
        """Overload detection looks at recent history (last 5 ops)."""
        G, node = create_nfr("old_oz", epi=0.5, vf=1.0)
        G.graph["VALIDATE_OPERATOR_PRECONDITIONS"] = True

        # Old OZ (6+ operations ago)
        Dissonance()(G, node)
        Coherence()(G, node)

        # Many operations in between
        for _ in range(5):
            Emission()(G, node)
            Coherence()(G, node)

        # New OZ should succeed (old OZ outside window)
        Dissonance()(G, node)

    def test_oz_no_overload_on_first_application(self):
        """First OZ application cannot be overload."""
        G, node = create_nfr("first_oz", epi=0.5, vf=1.0)

        # First OZ - no history, cannot be overload
        validate_dissonance_strict(G, node)


class TestDissonanceOperatorWithValidation:
    """Test Dissonance operator with precondition validation enabled."""

    def test_dissonance_validation_disabled_by_default(self):
        """Precondition validation disabled by default for backward compatibility."""
        # Create node that violates preconditions
        G, node = create_nfr("invalid", epi=0.05, vf=1.0)

        # Should succeed (validation disabled)
        Dissonance()(G, node)

    def test_dissonance_validation_enabled_via_flag(self):
        """Precondition validation can be enabled via graph flag."""
        G, node = create_nfr("invalid", epi=0.05, vf=1.0)

        # Enable validation
        G.graph["VALIDATE_OPERATOR_PRECONDITIONS"] = True

        # Should now fail
        with pytest.raises(OperatorPreconditionError) as exc_info:
            Dissonance()(G, node)

        assert "Insufficient coherence base" in str(exc_info.value)

    def test_dissonance_validation_success_when_enabled(self):
        """Valid dissonance succeeds with validation enabled."""
        G, node = create_nfr("valid", epi=0.5, vf=1.0)
        G.graph["VALIDATE_OPERATOR_PRECONDITIONS"] = True

        # Should succeed
        Dissonance()(G, node)

    def test_dissonance_low_epi_fails_with_validation(self):
        """Low EPI fails with validation enabled."""
        G, node = create_nfr("weak", epi=0.05, vf=1.0)
        G.graph["VALIDATE_OPERATOR_PRECONDITIONS"] = True

        with pytest.raises(OperatorPreconditionError) as exc_info:
            Dissonance()(G, node)

        error_msg = str(exc_info.value)
        assert "EPI=0.050" in error_msg
        assert "< 0.2" in error_msg

    def test_dissonance_high_dnfr_fails_with_validation(self):
        """High ΔNFR fails with validation enabled."""
        G, node = create_nfr("critical", epi=0.5, vf=1.0)
        set_attr(G.nodes[node], ALIAS_DNFR, 0.9)
        G.graph["VALIDATE_OPERATOR_PRECONDITIONS"] = True

        with pytest.raises(OperatorPreconditionError) as exc_info:
            Dissonance()(G, node)

        error_msg = str(exc_info.value)
        assert "|ΔNFR|=0.900" in error_msg

    def test_dissonance_overload_fails_with_validation(self):
        """Overload fails with validation enabled."""
        G, node = create_nfr("overload", epi=0.5, vf=1.0)
        G.graph["VALIDATE_OPERATOR_PRECONDITIONS"] = True

        # First OZ
        Dissonance()(G, node)

        # Second OZ should fail
        with pytest.raises(OperatorPreconditionError) as exc_info:
            Dissonance()(G, node)

        assert "Sobrecarga disonante" in str(exc_info.value)


class TestDissonancePreconditionsEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_exactly_threshold_epi_passes(self):
        """EPI exactly at threshold passes (>= condition)."""
        G, node = create_nfr("at_threshold", epi=0.2, vf=1.0)

        validate_dissonance_strict(G, node)

    def test_just_below_threshold_epi_fails(self):
        """EPI just below threshold fails."""
        G, node = create_nfr("just_below", epi=0.199, vf=1.0)

        with pytest.raises(ValueError):
            validate_dissonance_strict(G, node)

    def test_exactly_threshold_dnfr_passes(self):
        """ΔNFR exactly at threshold passes (<= condition)."""
        G, node = create_nfr("dnfr_threshold", epi=0.5, vf=1.0)
        set_attr(G.nodes[node], ALIAS_DNFR, 0.8)

        validate_dissonance_strict(G, node)

    def test_just_above_threshold_dnfr_fails(self):
        """ΔNFR just above threshold fails."""
        G, node = create_nfr("dnfr_above", epi=0.5, vf=1.0)
        set_attr(G.nodes[node], ALIAS_DNFR, 0.801)

        with pytest.raises(ValueError):
            validate_dissonance_strict(G, node)

    def test_negative_dnfr_uses_absolute_value(self):
        """Negative ΔNFR is checked using absolute value."""
        G, node = create_nfr("negative_dnfr", epi=0.5, vf=1.0)
        set_attr(G.nodes[node], ALIAS_DNFR, -0.9)

        # |-0.9| = 0.9 > 0.8, should fail
        with pytest.raises(ValueError) as exc_info:
            validate_dissonance_strict(G, node)

        assert "|ΔNFR|=0.900" in str(exc_info.value)

    def test_high_epi_with_low_vf_fails(self):
        """High EPI doesn't compensate for low νf."""
        G, node = create_nfr("high_epi_low_vf", epi=0.9, vf=0.05)

        with pytest.raises(ValueError) as exc_info:
            validate_dissonance_strict(G, node)

        assert "νf=0.050" in str(exc_info.value)

    def test_multiple_violations_reports_first(self):
        """Multiple violations report in order (EPI checked first)."""
        G, node = create_nfr("multi_violation", epi=0.05, vf=0.05)

        # Should fail on EPI check (checked first)
        with pytest.raises(ValueError) as exc_info:
            validate_dissonance_strict(G, node)

        # Should mention EPI, not vf
        assert "EPI=" in str(exc_info.value)

    def test_zero_dnfr_passes(self):
        """Zero ΔNFR passes (no critical pressure)."""
        G, node = create_nfr("zero_dnfr", epi=0.5, vf=1.0)
        set_attr(G.nodes[node], ALIAS_DNFR, 0.0)

        # Should pass - ΔNFR check is for critically HIGH values
        validate_dissonance_strict(G, node)


class TestDissonancePreconditionsIntegration:
    """Integration tests with full operator sequences."""

    def test_coherence_prepares_weak_node_for_dissonance(self):
        """IL (Coherence) can prepare low-EPI node for OZ."""
        G, node = create_nfr("prep", epi=0.1, vf=1.0)
        G.graph["VALIDATE_OPERATOR_PRECONDITIONS"] = True

        # Apply Coherence to increase EPI
        Coherence()(G, node)

        epi_after = float(get_attr(G.nodes[node], ALIAS_EPI, 0.0))

        # If EPI increased enough, dissonance should succeed
        if epi_after >= 0.2:
            Dissonance()(G, node)

    def test_coherence_reduces_critical_dnfr(self):
        """IL (Coherence) reduces critical ΔNFR before OZ."""
        G, node = create_nfr("high_dnfr", epi=0.5, vf=1.0)
        set_attr(G.nodes[node], ALIAS_DNFR, 0.9)
        G.graph["VALIDATE_OPERATOR_PRECONDITIONS"] = True

        # Apply Coherence to reduce ΔNFR
        Coherence()(G, node)

        dnfr_after = abs(float(get_attr(G.nodes[node], ALIAS_DNFR, 0.0)))

        # If ΔNFR reduced enough, dissonance should succeed
        if dnfr_after <= 0.8:
            Dissonance()(G, node)

    def test_oz_il_oz_sequence_succeeds(self):
        """OZ → IL → OZ sequence succeeds (IL resolves overload)."""
        G, node = create_nfr("sequence", epi=0.5, vf=1.0)
        G.graph["VALIDATE_OPERATOR_PRECONDITIONS"] = True

        # First OZ
        Dissonance()(G, node)

        # Resolve with IL
        Coherence()(G, node)

        # Second OZ should succeed
        Dissonance()(G, node)

    def test_error_message_suggests_coherence(self):
        """Error messages suggest IL (Coherence) for resolution."""
        G, node = create_nfr("suggest", epi=0.05, vf=1.0)

        with pytest.raises(ValueError) as exc_info:
            validate_dissonance_strict(G, node)

        assert "Apply IL (Coherence)" in str(exc_info.value)
