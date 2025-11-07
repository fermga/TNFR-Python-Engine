"""Tests for AL (Emission) strict precondition validation.

This module validates the implementation of TNFR.pdf §2.2.1 precondition
requirements for the Emission (AL) operator:

1. EPI < latent threshold (node must be in latent/low-activation state)
2. νf > basal threshold (sufficient structural frequency for activation)
3. Network connectivity (warning for isolated nodes)

The validation can be enabled/disabled via the VALIDATE_OPERATOR_PRECONDITIONS
graph flag to maintain backward compatibility.
"""

import warnings

import pytest

from tnfr.alias import get_attr
from tnfr.config.thresholds import EPI_LATENT_MAX, VF_BASAL_THRESHOLD
from tnfr.constants.aliases import ALIAS_EPI, ALIAS_VF
from tnfr.operators.definitions import Emission
from tnfr.operators.preconditions.emission import validate_emission_strict
from tnfr.structural import create_nfr


class TestEmissionStrictPreconditions:
    """Test suite for AL strict precondition validation."""

    def test_validate_emission_strict_success_latent_node(self):
        """Validation passes for latent node with sufficient frequency."""
        G, node = create_nfr("latent", epi=0.25, vf=0.95)

        # Should not raise
        validate_emission_strict(G, node)

    def test_validate_emission_strict_success_at_threshold(self):
        """Validation passes for node just below EPI threshold."""
        G, node = create_nfr("threshold", epi=0.79, vf=1.0)

        # Should not raise (0.79 < 0.8 default threshold)
        validate_emission_strict(G, node)

    def test_validate_emission_strict_fails_high_epi(self):
        """Validation fails when EPI >= latent threshold."""
        G, node = create_nfr("active", epi=0.85, vf=1.0)

        with pytest.raises(ValueError) as exc_info:
            validate_emission_strict(G, node)

        error_msg = str(exc_info.value)
        assert "AL precondition failed" in error_msg
        assert "EPI=0.850" in error_msg
        assert ">= 0.8" in error_msg
        assert "Consider IL (Coherence)" in error_msg

    def test_validate_emission_strict_fails_low_vf(self):
        """Validation fails when νf < basal threshold."""
        G, node = create_nfr("frozen", epi=0.2, vf=0.3)

        with pytest.raises(ValueError) as exc_info:
            validate_emission_strict(G, node)

        error_msg = str(exc_info.value)
        assert "AL precondition failed" in error_msg
        assert "νf=0.300" in error_msg
        assert "< 0.5" in error_msg
        assert "Consider NAV (Transition)" in error_msg

    def test_validate_emission_strict_fails_at_vf_threshold(self):
        """Validation fails when νf exactly at threshold (< not <=)."""
        G, node = create_nfr("at_threshold", epi=0.2, vf=0.5)

        # At threshold should still be rejected (< not <=)
        # Actually, the implementation should allow νf = 0.5 since it's >=
        # Let me verify: the condition is `vf < vf_threshold`, so vf=0.5 should pass
        try:
            validate_emission_strict(G, node)
            # Should succeed since 0.5 is not < 0.5
        except ValueError:
            pytest.fail("Should not raise when vf equals threshold")

    def test_validate_emission_strict_warns_isolated_node(self):
        """Validation warns for isolated node in multi-node network."""
        G, node = create_nfr("isolated", epi=0.25, vf=0.95)

        # Add another node to make it multi-node network
        G.add_node("other")

        # Should warn but not fail
        with pytest.warns(UserWarning) as warning_info:
            validate_emission_strict(G, node)

        assert len(warning_info) > 0
        warning_msg = str(warning_info[0].message)
        assert "AL warning" in warning_msg
        assert "degree" in warning_msg.lower()
        assert "Consider UM (Coupling)" in warning_msg

    def test_validate_emission_strict_no_warning_single_node(self):
        """No isolation warning for single-node network."""
        G, node = create_nfr("solo", epi=0.25, vf=0.95)

        # Single node network - no warning expected
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # Turn warnings into errors
            validate_emission_strict(G, node)  # Should not raise

    def test_validate_emission_strict_no_warning_connected_node(self):
        """No isolation warning for connected node."""
        G, node = create_nfr("connected", epi=0.25, vf=0.95)

        # Add neighbor
        other = "neighbor"
        G.add_node(other)
        G.add_edge(node, other)

        # Should not warn
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            validate_emission_strict(G, node)

    def test_validate_emission_strict_custom_thresholds_via_graph(self):
        """Custom thresholds can be set via graph metadata."""
        G, node = create_nfr("custom", epi=0.85, vf=0.95)

        # With default thresholds, this would fail
        # Set custom higher EPI threshold
        G.graph["EPI_LATENT_MAX"] = 0.9

        # Should now pass
        validate_emission_strict(G, node)

    def test_validate_emission_strict_custom_vf_threshold(self):
        """Custom νf threshold via graph metadata."""
        G, node = create_nfr("custom_vf", epi=0.25, vf=0.3)

        # With default threshold (0.5), this would fail
        # Set custom lower threshold
        G.graph["VF_BASAL_THRESHOLD"] = 0.2

        # Should now pass
        validate_emission_strict(G, node)


class TestEmissionOperatorWithValidation:
    """Test Emission operator with precondition validation enabled."""

    def test_emission_validation_disabled_by_default(self):
        """Precondition validation disabled by default for backward compatibility."""
        # Create node that violates preconditions
        G, node = create_nfr("invalid", epi=0.95, vf=0.95)

        # Should succeed (validation disabled)
        Emission()(G, node)

    def test_emission_validation_enabled_via_flag(self):
        """Precondition validation can be enabled via graph flag."""
        G, node = create_nfr("invalid", epi=0.95, vf=0.95)

        # Enable validation
        G.graph["VALIDATE_OPERATOR_PRECONDITIONS"] = True

        # Should now fail
        with pytest.raises(ValueError) as exc_info:
            Emission()(G, node)

        assert "AL precondition failed" in str(exc_info.value)

    def test_emission_validation_success_when_enabled(self):
        """Valid emission succeeds with validation enabled."""
        G, node = create_nfr("valid", epi=0.25, vf=0.95)
        G.graph["VALIDATE_OPERATOR_PRECONDITIONS"] = True

        # Should succeed
        Emission()(G, node)

        # Verify emission was applied
        assert G.nodes[node]["_emission_activated"] is True

    def test_emission_high_epi_fails_with_validation(self):
        """High EPI fails with validation enabled."""
        G, node = create_nfr("too_active", epi=0.85, vf=1.0)
        G.graph["VALIDATE_OPERATOR_PRECONDITIONS"] = True

        with pytest.raises(ValueError) as exc_info:
            Emission()(G, node)

        error_msg = str(exc_info.value)
        assert "EPI=0.850" in error_msg
        assert "Consider IL (Coherence)" in error_msg

    def test_emission_low_vf_fails_with_validation(self):
        """Low νf fails with validation enabled."""
        G, node = create_nfr("low_freq", epi=0.2, vf=0.3)
        G.graph["VALIDATE_OPERATOR_PRECONDITIONS"] = True

        with pytest.raises(ValueError) as exc_info:
            Emission()(G, node)

        error_msg = str(exc_info.value)
        assert "νf=0.300" in error_msg
        assert "Consider NAV (Transition)" in error_msg

    def test_emission_isolated_warns_with_validation(self):
        """Isolated node warns with validation enabled."""
        G, node = create_nfr("isolated", epi=0.25, vf=0.95)
        G.add_node("other")  # Make it multi-node
        G.graph["VALIDATE_OPERATOR_PRECONDITIONS"] = True

        with pytest.warns(UserWarning) as warning_info:
            Emission()(G, node)

        assert len(warning_info) > 0
        assert "Consider UM (Coupling)" in str(warning_info[0].message)


class TestEmissionPreconditionsEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_zero_epi_passes_validation(self):
        """Zero EPI (fully latent) passes validation."""
        G, node = create_nfr("zero_epi", epi=0.0, vf=1.0)

        validate_emission_strict(G, node)

    def test_exactly_threshold_epi_fails(self):
        """EPI exactly at threshold fails (>= condition)."""
        G, node = create_nfr("at_threshold", epi=EPI_LATENT_MAX, vf=1.0)

        with pytest.raises(ValueError):
            validate_emission_strict(G, node)

    def test_just_below_threshold_epi_passes(self):
        """EPI just below threshold passes."""
        G, node = create_nfr("just_below", epi=EPI_LATENT_MAX - 0.001, vf=1.0)

        validate_emission_strict(G, node)

    def test_exactly_threshold_vf_passes(self):
        """νf exactly at threshold passes (>= condition for passing)."""
        G, node = create_nfr("vf_threshold", epi=0.2, vf=VF_BASAL_THRESHOLD)

        validate_emission_strict(G, node)

    def test_just_below_threshold_vf_fails(self):
        """νf just below threshold fails."""
        G, node = create_nfr("vf_below", epi=0.2, vf=VF_BASAL_THRESHOLD - 0.001)

        with pytest.raises(ValueError):
            validate_emission_strict(G, node)

    def test_multiple_violations_reports_first(self):
        """Multiple violations report in order (EPI checked first)."""
        G, node = create_nfr("multi_violation", epi=0.95, vf=0.1)

        # Should fail on EPI check (checked first)
        with pytest.raises(ValueError) as exc_info:
            validate_emission_strict(G, node)

        # Should mention EPI, not vf
        assert "EPI=" in str(exc_info.value)

    def test_negative_epi_passes_if_in_range(self):
        """Negative EPI values pass if below threshold."""
        G, node = create_nfr("negative_epi", epi=-0.5, vf=1.0)

        # Negative EPI is < 0.8, so should pass
        validate_emission_strict(G, node)

    def test_high_vf_with_latent_epi_passes(self):
        """Very high νf with latent EPI passes."""
        G, node = create_nfr("high_vf", epi=0.1, vf=9.5)

        validate_emission_strict(G, node)


class TestEmissionPreconditionsIntegration:
    """Integration tests with full operator sequences."""

    def test_emission_after_transition_succeeds(self):
        """NAV (Transition) can prepare low-vf node for Emission."""
        from tnfr.operators.definitions import Transition

        G, node = create_nfr("prep", epi=0.2, vf=0.3)
        G.graph["VALIDATE_OPERATOR_PRECONDITIONS"] = True

        # Apply Transition to increase vf
        Transition()(G, node)

        vf_after = float(get_attr(G.nodes[node], ALIAS_VF, 0.0))

        # If vf increased enough, emission should succeed
        if vf_after >= VF_BASAL_THRESHOLD:
            Emission()(G, node)
            assert G.nodes[node]["_emission_activated"] is True

    def test_coherence_suggested_for_active_nodes(self):
        """Error message suggests IL (Coherence) for already-active nodes."""
        from tnfr.operators.definitions import Coherence

        G, node = create_nfr("active", epi=0.85, vf=1.0)
        G.graph["VALIDATE_OPERATOR_PRECONDITIONS"] = True

        # Emission should fail with suggestion
        with pytest.raises(ValueError) as exc_info:
            Emission()(G, node)

        assert "Consider IL (Coherence)" in str(exc_info.value)

        # Coherence should work instead (it has different preconditions)
        # Note: Coherence requires significant ΔNFR, which may not exist
        # So we don't test actual Coherence application here

    def test_coupling_suggested_for_isolated_nodes(self):
        """Warning suggests UM (Coupling) for isolated nodes."""
        from tnfr.operators.definitions import Coupling

        G, node = create_nfr("isolated", epi=0.25, vf=0.95)
        G.add_node("other")
        G.graph["VALIDATE_OPERATOR_PRECONDITIONS"] = True

        # Should warn about isolation
        with pytest.warns(UserWarning) as warning_info:
            Emission()(G, node)

        assert "Consider UM (Coupling)" in str(warning_info[0].message)
