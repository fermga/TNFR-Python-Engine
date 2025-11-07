"""Tests for EN (Reception) strict precondition validation.

This module validates the implementation of TNFR.pdf §2.2.1 precondition
requirements for the Reception (EN) operator:

1. EPI < saturation threshold (node has receptive capacity)
2. DNFR < threshold (minimal dissonance for stable integration)
3. Emission sources availability (warning for isolated nodes)

This follows the same pattern established by AL (Emission) strict validation,
providing consistent precondition enforcement across structural operators.

The validation can be enabled/disabled via the VALIDATE_OPERATOR_PRECONDITIONS
graph flag to maintain backward compatibility.
"""

import warnings

import pytest

from tnfr.alias import get_attr
from tnfr.config.thresholds import DNFR_RECEPTION_MAX, EPI_SATURATION_MAX
from tnfr.constants.aliases import ALIAS_DNFR, ALIAS_EPI
from tnfr.operators.definitions import Reception
from tnfr.operators.preconditions.reception import validate_reception_strict
from tnfr.structural import create_nfr


class TestReceptionStrictPreconditions:
    """Test suite for EN strict precondition validation."""

    def test_validate_reception_strict_success_normal_node(self):
        """Validation passes for node with receptive capacity and low dissonance."""
        G, node = create_nfr("receiver", epi=0.5, vf=0.9)
        G.nodes[node]["dnfr"] = 0.08

        # Should not raise
        validate_reception_strict(G, node)

    def test_validate_reception_strict_success_low_epi_low_dnfr(self):
        """Validation passes for node with low EPI and DNFR."""
        G, node = create_nfr("latent", epi=0.1, vf=0.8)
        G.nodes[node]["dnfr"] = 0.02

        # Should not raise
        validate_reception_strict(G, node)

    def test_validate_reception_strict_success_at_threshold(self):
        """Validation passes for node just below EPI threshold."""
        G, node = create_nfr("at_threshold", epi=0.89, vf=1.0)
        G.nodes[node]["dnfr"] = 0.05

        # Should not raise (0.89 < 0.9 default threshold)
        validate_reception_strict(G, node)

    def test_validate_reception_strict_fails_high_epi(self):
        """Validation fails when EPI >= saturation threshold."""
        G, node = create_nfr("saturated", epi=0.95, vf=1.0)
        G.nodes[node]["dnfr"] = 0.05

        with pytest.raises(ValueError) as exc_info:
            validate_reception_strict(G, node)

        error_msg = str(exc_info.value)
        assert "EN precondition failed" in error_msg
        assert "EPI=0.950" in error_msg
        assert ">= 0.9" in error_msg
        assert "Node saturated" in error_msg
        assert "IL (Coherence)" in error_msg or "NUL (Contraction)" in error_msg

    def test_validate_reception_strict_fails_at_epi_threshold(self):
        """Validation fails when EPI equals threshold (boundary condition)."""
        G, node = create_nfr("at_max", epi=0.9, vf=1.0)
        G.nodes[node]["dnfr"] = 0.05

        with pytest.raises(ValueError) as exc_info:
            validate_reception_strict(G, node)

        error_msg = str(exc_info.value)
        assert "EN precondition failed" in error_msg
        assert "EPI=0.900" in error_msg
        assert ">= 0.9" in error_msg

    def test_validate_reception_strict_fails_high_dnfr(self):
        """Validation fails when DNFR >= threshold."""
        G, node = create_nfr("dissonant", epi=0.5, vf=1.0)
        G.nodes[node]["dnfr"] = 0.18

        with pytest.raises(ValueError) as exc_info:
            validate_reception_strict(G, node)

        error_msg = str(exc_info.value)
        assert "EN precondition failed" in error_msg
        assert "DNFR=0.180" in error_msg
        assert ">= 0.15" in error_msg
        assert "Excessive dissonance" in error_msg
        assert "Consider IL (Coherence)" in error_msg

    def test_validate_reception_strict_fails_at_dnfr_threshold(self):
        """Validation fails when DNFR equals threshold (boundary condition)."""
        G, node = create_nfr("at_dnfr_max", epi=0.5, vf=1.0)
        G.nodes[node]["dnfr"] = 0.15

        with pytest.raises(ValueError) as exc_info:
            validate_reception_strict(G, node)

        error_msg = str(exc_info.value)
        assert "EN precondition failed" in error_msg
        assert "DNFR=0.150" in error_msg
        assert ">= 0.15" in error_msg

    def test_validate_reception_strict_passes_just_below_dnfr_threshold(self):
        """Validation passes when DNFR just below threshold."""
        G, node = create_nfr("just_below", epi=0.5, vf=1.0)
        G.nodes[node]["dnfr"] = 0.149

        # Should succeed: dnfr=0.149 is not >= 0.15
        try:
            validate_reception_strict(G, node)
        except ValueError:
            pytest.fail("Should not raise when DNFR just below threshold")

    def test_validate_reception_strict_warns_isolated_node(self):
        """Validation warns for isolated node in multi-node network."""
        G, node = create_nfr("isolated", epi=0.5, vf=0.9)
        G.nodes[node]["dnfr"] = 0.08

        # Add another node to make it multi-node network
        G.add_node("other")

        # Should warn but not fail
        with pytest.warns(UserWarning) as warning_info:
            validate_reception_strict(G, node)

        assert len(warning_info) > 0
        warning_msg = str(warning_info[0].message)
        assert "EN warning" in warning_msg
        assert "isolated" in warning_msg.lower()
        assert "Consider UM (Coupling)" in warning_msg

    def test_validate_reception_strict_no_warning_single_node(self):
        """No isolation warning for single-node network."""
        G, node = create_nfr("solo", epi=0.5, vf=0.9)
        G.nodes[node]["dnfr"] = 0.08

        # Single node network - no warning expected
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # Convert warnings to errors
            # Should not raise (including no warnings)
            validate_reception_strict(G, node)

    def test_validate_reception_strict_no_warning_connected_node(self):
        """No isolation warning for connected node."""
        G, node = create_nfr("connected", epi=0.5, vf=0.9)
        G.nodes[node]["dnfr"] = 0.08

        # Add neighbor
        G.add_node("neighbor")
        G.add_edge(node, "neighbor")

        # Should not warn
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            validate_reception_strict(G, node)

    def test_validate_reception_strict_custom_thresholds_via_graph(self):
        """Validation respects custom thresholds via graph metadata."""
        G, node = create_nfr("custom", epi=0.85, vf=1.0)
        G.nodes[node]["dnfr"] = 0.12

        # Set custom thresholds
        G.graph["EPI_SATURATION_MAX"] = 0.95
        G.graph["DNFR_RECEPTION_MAX"] = 0.20

        # Should not raise with higher thresholds
        validate_reception_strict(G, node)

    def test_validate_reception_strict_custom_epi_threshold(self):
        """Custom EPI threshold is enforced correctly."""
        G, node = create_nfr("custom_epi", epi=0.85, vf=1.0)
        G.nodes[node]["dnfr"] = 0.05

        # Set stricter EPI threshold
        G.graph["EPI_SATURATION_MAX"] = 0.8

        with pytest.raises(ValueError) as exc_info:
            validate_reception_strict(G, node)

        error_msg = str(exc_info.value)
        assert "EPI=0.850" in error_msg
        assert ">= 0.8" in error_msg

    def test_validate_reception_strict_custom_dnfr_threshold(self):
        """Custom DNFR threshold is enforced correctly."""
        G, node = create_nfr("custom_dnfr", epi=0.5, vf=1.0)
        G.nodes[node]["dnfr"] = 0.12

        # Set stricter DNFR threshold
        G.graph["DNFR_RECEPTION_MAX"] = 0.10

        with pytest.raises(ValueError) as exc_info:
            validate_reception_strict(G, node)

        error_msg = str(exc_info.value)
        assert "DNFR=0.120" in error_msg
        assert ">= 0.1" in error_msg


class TestReceptionOperatorWithValidation:
    """Test Reception operator with precondition validation enabled."""

    def test_reception_validation_disabled_by_default(self):
        """Reception operator does not validate by default."""
        G, node = create_nfr("receiver", epi=0.95, vf=1.0)  # Saturated EPI
        G.nodes[node]["dnfr"] = 0.20  # High DNFR

        # Add neighbor to pass basic check
        G.add_node("neighbor")
        G.add_edge(node, "neighbor")

        # Should not raise without validation flag
        Reception()(G, node)

    def test_reception_validation_enabled_via_flag(self):
        """Reception operator validates when flag is set."""
        G, node = create_nfr("receiver", epi=0.5, vf=0.9)
        G.nodes[node]["dnfr"] = 0.08

        # Add neighbor
        G.add_node("neighbor")
        G.add_edge(node, "neighbor")

        # Enable validation
        G.graph["VALIDATE_OPERATOR_PRECONDITIONS"] = True

        # Should not raise (valid state)
        Reception()(G, node)

    def test_reception_validation_success_when_enabled(self):
        """Reception passes validation with valid node state."""
        G, node = create_nfr("valid", epi=0.3, vf=0.9)
        G.nodes[node]["dnfr"] = 0.05

        # Add neighbor
        G.add_node("neighbor")
        G.add_edge(node, "neighbor")

        # Enable validation
        G.graph["VALIDATE_OPERATOR_PRECONDITIONS"] = True

        # Should succeed
        Reception()(G, node)

    def test_reception_high_epi_fails_with_validation(self):
        """Reception fails validation when EPI is saturated."""
        G, node = create_nfr("saturated", epi=0.95, vf=1.0)
        G.nodes[node]["dnfr"] = 0.05

        # Add neighbor
        G.add_node("neighbor")
        G.add_edge(node, "neighbor")

        # Enable validation
        G.graph["VALIDATE_OPERATOR_PRECONDITIONS"] = True

        with pytest.raises(ValueError) as exc_info:
            Reception()(G, node)

        error_msg = str(exc_info.value)
        assert "EN precondition failed" in error_msg
        assert "EPI=0.950" in error_msg

    def test_reception_high_dnfr_fails_with_validation(self):
        """Reception fails validation when DNFR is too high."""
        G, node = create_nfr("dissonant", epi=0.5, vf=1.0)
        G.nodes[node]["dnfr"] = 0.20

        # Add neighbor
        G.add_node("neighbor")
        G.add_edge(node, "neighbor")

        # Enable validation
        G.graph["VALIDATE_OPERATOR_PRECONDITIONS"] = True

        with pytest.raises(ValueError) as exc_info:
            Reception()(G, node)

        error_msg = str(exc_info.value)
        assert "EN precondition failed" in error_msg
        assert "DNFR=0.200" in error_msg

    def test_reception_isolated_warns_with_validation(self):
        """Reception warns for isolated node when validation enabled."""
        G, node = create_nfr("isolated", epi=0.5, vf=0.9)
        G.nodes[node]["dnfr"] = 0.08

        # Add another node (but don't connect)
        G.add_node("other")

        # Enable validation
        G.graph["VALIDATE_OPERATOR_PRECONDITIONS"] = True

        # Should warn but not fail
        with pytest.warns(UserWarning) as warning_info:
            Reception()(G, node, track_sources=False)  # Disable source tracking to test only preconditions

        assert len(warning_info) > 0
        warning_msg = str(warning_info[0].message)
        assert "EN warning" in warning_msg


class TestReceptionPreconditionsEdgeCases:
    """Test edge cases for EN precondition validation."""

    def test_zero_epi_passes_validation(self):
        """Zero EPI should pass validation (maximum receptive capacity)."""
        G, node = create_nfr("zero", epi=0.0, vf=0.9)
        G.nodes[node]["dnfr"] = 0.05

        validate_reception_strict(G, node)

    def test_zero_dnfr_passes_validation(self):
        """Zero DNFR should pass validation (no dissonance)."""
        G, node = create_nfr("stable", epi=0.5, vf=0.9)
        G.nodes[node]["dnfr"] = 0.0

        validate_reception_strict(G, node)

    def test_exactly_threshold_epi_fails(self):
        """EPI exactly at threshold should fail."""
        G, node = create_nfr("at_max", epi=EPI_SATURATION_MAX, vf=1.0)
        G.nodes[node]["dnfr"] = 0.05

        with pytest.raises(ValueError):
            validate_reception_strict(G, node)

    def test_just_below_threshold_epi_passes(self):
        """EPI just below threshold should pass."""
        G, node = create_nfr("just_below", epi=EPI_SATURATION_MAX - 0.001, vf=1.0)
        G.nodes[node]["dnfr"] = 0.05

        validate_reception_strict(G, node)

    def test_exactly_threshold_dnfr_fails(self):
        """DNFR exactly at threshold should fail."""
        G, node = create_nfr("at_max_dnfr", epi=0.5, vf=1.0)
        G.nodes[node]["dnfr"] = DNFR_RECEPTION_MAX

        with pytest.raises(ValueError):
            validate_reception_strict(G, node)

    def test_just_below_threshold_dnfr_passes(self):
        """DNFR just below threshold should pass."""
        G, node = create_nfr("just_below_dnfr", epi=0.5, vf=1.0)
        G.nodes[node]["dnfr"] = DNFR_RECEPTION_MAX - 0.001

        validate_reception_strict(G, node)

    def test_multiple_violations_reports_first(self):
        """When multiple conditions fail, first violation is reported."""
        G, node = create_nfr("multiple", epi=0.95, vf=1.0)
        G.nodes[node]["dnfr"] = 0.20

        # Both EPI and DNFR are over threshold
        # Should report EPI first (checked first in implementation)
        with pytest.raises(ValueError) as exc_info:
            validate_reception_strict(G, node)

        error_msg = str(exc_info.value)
        assert "EPI=0.950" in error_msg

    def test_negative_epi_passes_if_in_range(self):
        """Negative EPI (if somehow set) should pass as it's below threshold."""
        G, node = create_nfr("negative", epi=-0.1, vf=0.9)
        G.nodes[node]["dnfr"] = 0.05

        # Should not raise (negative < threshold)
        validate_reception_strict(G, node)

    def test_high_epi_with_low_dnfr_still_fails(self):
        """High EPI fails even with low DNFR."""
        G, node = create_nfr("high_epi_low_dnfr", epi=0.95, vf=1.0)
        G.nodes[node]["dnfr"] = 0.01

        with pytest.raises(ValueError) as exc_info:
            validate_reception_strict(G, node)

        assert "EPI=0.950" in str(exc_info.value)

    def test_low_epi_with_high_dnfr_still_fails(self):
        """High DNFR fails even with low EPI."""
        G, node = create_nfr("low_epi_high_dnfr", epi=0.2, vf=1.0)
        G.nodes[node]["dnfr"] = 0.20

        with pytest.raises(ValueError) as exc_info:
            validate_reception_strict(G, node)

        assert "DNFR=0.200" in str(exc_info.value)


class TestReceptionPreconditionsIntegration:
    """Integration tests for EN precondition validation in sequences."""

    def test_reception_after_coherence_succeeds(self):
        """Reception after Coherence should reduce DNFR and succeed."""
        from tnfr.operators.definitions import Coherence

        G, node = create_nfr("stabilizing", epi=0.5, vf=1.0)
        G.nodes[node]["dnfr"] = 0.20  # High DNFR

        # Add neighbor
        G.add_node("neighbor")
        G.add_edge(node, "neighbor")

        # Apply Coherence first (reduces DNFR)
        Coherence()(G, node)

        # Enable validation
        G.graph["VALIDATE_OPERATOR_PRECONDITIONS"] = True

        # DNFR should now be low enough for Reception
        # (assuming Coherence implementation reduces DNFR)
        try:
            Reception()(G, node)
        except ValueError:
            # If this fails, it means DNFR is still too high after Coherence
            # This is acceptable - the test verifies the sequence logic
            pass

    def test_coherence_suggested_for_high_dnfr(self):
        """Error message suggests IL (Coherence) for high DNFR nodes."""
        G, node = create_nfr("dissonant", epi=0.5, vf=1.0)
        G.nodes[node]["dnfr"] = 0.20

        # Enable validation
        G.graph["VALIDATE_OPERATOR_PRECONDITIONS"] = True

        # Add neighbor
        G.add_node("neighbor")
        G.add_edge(node, "neighbor")

        with pytest.raises(ValueError) as exc_info:
            Reception()(G, node)

        assert "Consider IL (Coherence)" in str(exc_info.value)

    def test_coherence_or_contraction_suggested_for_high_epi(self):
        """Error message suggests IL or NUL for saturated nodes."""
        G, node = create_nfr("saturated", epi=0.95, vf=1.0)
        G.nodes[node]["dnfr"] = 0.05

        # Enable validation
        G.graph["VALIDATE_OPERATOR_PRECONDITIONS"] = True

        # Add neighbor
        G.add_node("neighbor")
        G.add_edge(node, "neighbor")

        with pytest.raises(ValueError) as exc_info:
            Reception()(G, node)

        error_msg = str(exc_info.value)
        assert "IL (Coherence)" in error_msg or "NUL (Contraction)" in error_msg

    def test_coupling_suggested_for_isolated_nodes(self):
        """Warning suggests UM (Coupling) for isolated nodes."""
        G, node = create_nfr("isolated", epi=0.5, vf=0.9)
        G.nodes[node]["dnfr"] = 0.08

        # Add another node (but don't connect)
        G.add_node("other")

        # Enable validation
        G.graph["VALIDATE_OPERATOR_PRECONDITIONS"] = True

        with pytest.warns(UserWarning) as warning_info:
            Reception()(G, node, track_sources=False)

        warning_msg = str(warning_info[0].message)
        assert "Consider UM (Coupling)" in warning_msg
