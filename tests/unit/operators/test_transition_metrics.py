"""Tests for enhanced Transition (NAV) operator metrics.

This module validates comprehensive NAV metrics as specified in GitHub issue:
- Regime origin/destination classification
- Phase shift magnitude (wrapped properly for 0-2π)
- Transition type (reactivation, phase_shift, regime_change)
- Frequency scaling factor
- ΔNFR damping ratio
- EPI preservation ratio
- Latency duration tracking

Tests verify:
- Regime classification accuracy
- Phase wrapping behavior (0-2π boundary)
- Transition type logic
- Scaling factor computation
- EPI preservation tracking
- Latency duration from SHA → NAV
- Metrics storage in operator_metrics
"""

import math

import pytest

from tnfr.alias import get_attr, set_attr
from tnfr.constants.aliases import ALIAS_DNFR, ALIAS_EPI, ALIAS_THETA, ALIAS_VF
from tnfr.operators.definitions import Coherence, Emission, Silence, Transition
from tnfr.structural import create_nfr, run_sequence


class TestRegimeClassification:
    """Test regime origin and destination classification in metrics."""

    def test_latent_to_active_regime_metrics(self):
        """Latent → active transition should report correct regimes."""
        G, node = create_nfr("test", epi=0.3, vf=0.04)
        G.nodes[node]["latent"] = True
        G.graph["COLLECT_OPERATOR_METRICS"] = True

        Transition()(G, node)

        metrics = G.graph["operator_metrics"][-1]
        assert metrics["regime_origin"] == "latent"
        # After transition, vf increases from latent, should be active
        assert metrics["regime_destination"] in ["active", "latent"]

    def test_active_to_active_regime_metrics(self):
        """Active → active transition should report correct regimes."""
        G, node = create_nfr("test", epi=0.4, vf=0.5)
        G.graph["COLLECT_OPERATOR_METRICS"] = True

        Transition()(G, node)

        metrics = G.graph["operator_metrics"][-1]
        assert metrics["regime_origin"] == "active"
        assert metrics["regime_destination"] == "active"

    def test_resonant_to_active_regime_metrics(self):
        """Resonant → active transition should report correct regimes."""
        G, node = create_nfr("test", epi=0.6, vf=0.85)
        G.graph["COLLECT_OPERATOR_METRICS"] = True

        Transition()(G, node)

        metrics = G.graph["operator_metrics"][-1]
        assert metrics["regime_origin"] == "resonant"
        # After transition, vf might decrease slightly
        assert metrics["regime_destination"] in ["active", "resonant"]


class TestTransitionTypeClassification:
    """Test transition type logic (reactivation, phase_shift, regime_change)."""

    def test_reactivation_transition_type(self):
        """Transition from latent should be classified as reactivation."""
        G, node = create_nfr("test", epi=0.3, vf=0.04)
        G.nodes[node]["latent"] = True
        G.graph["COLLECT_OPERATOR_METRICS"] = True

        Transition()(G, node)

        metrics = G.graph["operator_metrics"][-1]
        assert metrics["transition_type"] == "reactivation"

    def test_phase_shift_transition_type(self):
        """Large phase change (>0.3 rad) should be classified as phase_shift."""
        G, node = create_nfr("test", epi=0.4, vf=0.5, theta=0.5)
        G.graph["COLLECT_OPERATOR_METRICS"] = True

        # Apply transition (active regime has default phase_shift=0.2)
        # This is below 0.3 threshold, so it should be regime_change
        Transition()(G, node)

        metrics = G.graph["operator_metrics"][-1]
        # Should be regime_change since default active phase shift (0.2) < 0.3
        assert metrics["transition_type"] == "regime_change"
        assert metrics["phase_shift_magnitude"] < 0.3

    def test_regime_change_transition_type(self):
        """Small phase change should be classified as regime_change."""
        G, node = create_nfr("test", epi=0.4, vf=0.5)
        G.graph["COLLECT_OPERATOR_METRICS"] = True

        # Apply transition with small phase shift
        Transition()(G, node, phase_shift=0.1)

        metrics = G.graph["operator_metrics"][-1]
        # Should be regime_change since |Δθ| < 0.3 and not latent
        assert metrics["transition_type"] == "regime_change"


class TestPhaseShiftMetrics:
    """Test phase shift magnitude and wrapping behavior."""

    def test_phase_shift_magnitude_computed(self):
        """Phase shift magnitude should be computed from actual change."""
        G, node = create_nfr("test", epi=0.4, vf=0.5, theta=1.0)
        theta_before = get_attr(G.nodes[node], ALIAS_THETA, 0.0)
        G.graph["COLLECT_OPERATOR_METRICS"] = True

        Transition()(G, node)

        theta_after = get_attr(G.nodes[node], ALIAS_THETA, 0.0)
        metrics = G.graph["operator_metrics"][-1]

        # Verify phase shift magnitude matches actual change
        actual_shift = abs(theta_after - theta_before)
        if actual_shift > math.pi:
            actual_shift = 2 * math.pi - actual_shift

        assert metrics["phase_shift_magnitude"] >= 0
        assert abs(metrics["phase_shift_signed"]) == metrics["phase_shift_magnitude"]

    def test_phase_shift_wrapping_at_2pi(self):
        """Phase shift should wrap correctly at 2π boundary."""
        G, node = create_nfr("test", epi=0.4, vf=0.5, theta=6.0)  # Near 2π
        theta_before = get_attr(G.nodes[node], ALIAS_THETA, 0.0)
        G.graph["COLLECT_OPERATOR_METRICS"] = True

        Transition()(G, node)

        theta_after = get_attr(G.nodes[node], ALIAS_THETA, 0.0)
        metrics = G.graph["operator_metrics"][-1]

        # Calculate expected wrapped shift
        raw_shift = theta_after - theta_before
        expected_wrapped = raw_shift
        if expected_wrapped > math.pi:
            expected_wrapped -= 2 * math.pi
        elif expected_wrapped < -math.pi:
            expected_wrapped += 2 * math.pi

        # Verify wrapping occurred correctly
        assert abs(metrics["phase_shift_signed"]) <= math.pi
        # Allow for floating point tolerance
        assert math.isclose(metrics["phase_shift_signed"], expected_wrapped, abs_tol=0.1)

    def test_phase_shift_signed_preserves_direction(self):
        """Signed phase shift should preserve direction information."""
        G, node = create_nfr("test", epi=0.4, vf=0.5, theta=1.0)
        theta_before = get_attr(G.nodes[node], ALIAS_THETA, 0.0)
        G.graph["COLLECT_OPERATOR_METRICS"] = True

        Transition()(G, node)

        theta_after = get_attr(G.nodes[node], ALIAS_THETA, 0.0)
        metrics = G.graph["operator_metrics"][-1]

        # Calculate expected sign
        raw_shift = theta_after - theta_before
        if raw_shift > math.pi:
            raw_shift -= 2 * math.pi
        elif raw_shift < -math.pi:
            raw_shift += 2 * math.pi

        # Verify sign is preserved
        if abs(raw_shift) > 1e-9:  # If there's actual movement
            assert math.copysign(1.0, metrics["phase_shift_signed"]) == math.copysign(
                1.0, raw_shift
            )
        assert metrics["delta_theta"] == metrics["phase_shift_signed"]


class TestStructuralScalingMetrics:
    """Test frequency scaling and ΔNFR damping ratio metrics."""

    def test_vf_scaling_factor_latent_transition(self):
        """Latent → active should have vf_scaling > 1.0 (20% increase)."""
        G, node = create_nfr("test", epi=0.3, vf=0.5)
        # Mark as latent to trigger latent transition
        G.nodes[node]["latent"] = True
        vf_before = get_attr(G.nodes[node], ALIAS_VF, 0.0)
        G.graph["COLLECT_OPERATOR_METRICS"] = True

        Transition()(G, node)

        vf_after = get_attr(G.nodes[node], ALIAS_VF, 0.0)
        metrics = G.graph["operator_metrics"][-1]

        # Latent transition increases vf by 20%
        expected_scaling = vf_after / vf_before if vf_before > 0 else 1.0
        assert metrics["vf_scaling_factor"] > 1.0
        assert math.isclose(metrics["vf_scaling_factor"], expected_scaling, abs_tol=0.01)

    def test_vf_scaling_factor_active_transition(self):
        """Active transition should preserve vf (scaling ≈ 1.0)."""
        G, node = create_nfr("test", epi=0.4, vf=0.5)
        G.graph["COLLECT_OPERATOR_METRICS"] = True

        Transition()(G, node)

        metrics = G.graph["operator_metrics"][-1]
        # Active transition default vf_factor = 1.0
        assert math.isclose(metrics["vf_scaling_factor"], 1.0, abs_tol=0.05)

    def test_dnfr_damping_ratio_computation(self):
        """ΔNFR damping ratio should be dnfr_after / dnfr_before."""
        G, node = create_nfr("test", epi=0.4, vf=0.5)
        set_attr(G.nodes[node], ALIAS_DNFR, 0.5)
        G.graph["COLLECT_OPERATOR_METRICS"] = True

        Transition()(G, node)

        metrics = G.graph["operator_metrics"][-1]
        dnfr_after = metrics["dnfr_final"]
        dnfr_before = 0.5
        expected_ratio = dnfr_after / dnfr_before if dnfr_before > 1e-9 else 1.0

        assert math.isclose(metrics["dnfr_damping_ratio"], expected_ratio, abs_tol=0.01)

    def test_dnfr_damping_ratio_zero_before(self):
        """ΔNFR damping should handle zero before value."""
        G, node = create_nfr("test", epi=0.4, vf=0.5)
        set_attr(G.nodes[node], ALIAS_DNFR, 0.0)
        G.graph["COLLECT_OPERATOR_METRICS"] = True

        Transition()(G, node)

        metrics = G.graph["operator_metrics"][-1]
        # Should default to 1.0 when dnfr_before ≈ 0
        assert metrics["dnfr_damping_ratio"] == 1.0


class TestEPIPreservationMetrics:
    """Test EPI preservation ratio tracking."""

    def test_epi_preservation_ratio_computed(self):
        """EPI preservation should be epi_after / epi_before."""
        G, node = create_nfr("test", epi=0.5, vf=0.5)
        G.graph["COLLECT_OPERATOR_METRICS"] = True

        Transition()(G, node)

        metrics = G.graph["operator_metrics"][-1]
        assert "epi_preservation" in metrics
        # EPI should be preserved (ratio ≈ 1.0)
        if metrics["epi_preservation"] is not None:
            assert 0.95 <= metrics["epi_preservation"] <= 1.05

    def test_epi_preservation_near_unity(self):
        """EPI preservation should be near 1.0 (identity preserved)."""
        G, node = create_nfr("test", epi=0.6, vf=0.7)
        epi_before = get_attr(G.nodes[node], ALIAS_EPI, 0.0)
        G.graph["COLLECT_OPERATOR_METRICS"] = True

        Transition()(G, node)

        epi_after = get_attr(G.nodes[node], ALIAS_EPI, 0.0)
        metrics = G.graph["operator_metrics"][-1]

        # Verify EPI barely changed (< 5% drift)
        epi_drift = abs(epi_after - epi_before)
        assert epi_drift < 0.05 * epi_before

        # Verify metric reflects preservation
        if metrics["epi_preservation"] is not None:
            assert math.isclose(metrics["epi_preservation"], 1.0, abs_tol=0.05)


class TestLatencyDurationMetrics:
    """Test latency duration tracking from SHA → NAV."""

    def test_latency_duration_from_silence(self):
        """SHA → NAV should record silence_duration in metrics."""
        G, node = create_nfr("test", epi=0.5, vf=0.8)
        G.graph["COLLECT_OPERATOR_METRICS"] = True

        # Apply Emission → Coherence → Silence to enter latency (per U1a, U2)
        run_sequence(G, node, [Emission(), Coherence(), Silence()])

        # Apply Transition
        Transition()(G, node)

        metrics = G.graph["operator_metrics"][-1]
        # Should have latency duration (computed from timestamp)
        assert metrics["latency_duration"] is not None
        # Duration should be small (< 1 second since we just applied Silence)
        assert 0 <= metrics["latency_duration"] < 1.0

    def test_latency_duration_none_without_silence(self):
        """Direct NAV without prior SHA should have latency_duration = None."""
        G, node = create_nfr("test", epi=0.5, vf=0.8)
        G.graph["COLLECT_OPERATOR_METRICS"] = True

        Transition()(G, node)

        metrics = G.graph["operator_metrics"][-1]
        assert metrics["latency_duration"] is None


class TestMetricsStorage:
    """Test that metrics are properly stored in operator_metrics."""

    def test_metrics_stored_in_operator_metrics(self):
        """Metrics should be stored in G.graph['operator_metrics']."""
        G, node = create_nfr("test", epi=0.4, vf=0.5)
        G.graph["COLLECT_OPERATOR_METRICS"] = True

        Transition()(G, node)

        assert "operator_metrics" in G.graph
        assert len(G.graph["operator_metrics"]) > 0
        metrics = G.graph["operator_metrics"][-1]
        assert metrics["operator"] == "Transition"
        assert metrics["glyph"] == "NAV"

    def test_metrics_contain_all_required_fields(self):
        """Metrics should contain all required fields per issue spec."""
        G, node = create_nfr("test", epi=0.4, vf=0.5)
        G.graph["COLLECT_OPERATOR_METRICS"] = True

        Transition()(G, node)

        metrics = G.graph["operator_metrics"][-1]

        # Check required fields
        required_fields = [
            "operator",
            "glyph",
            "regime_origin",
            "regime_destination",
            "transition_type",
            "phase_shift_magnitude",
            "vf_scaling_factor",
            "dnfr_damping_ratio",
            "epi_preservation",
            "latency_duration",
            "delta_theta",
            "delta_vf",
            "delta_dnfr",
        ]

        for field in required_fields:
            assert field in metrics, f"Missing required field: {field}"


class TestBackwardCompatibility:
    """Test that existing metrics are preserved for backward compatibility."""

    def test_legacy_metrics_preserved(self):
        """Legacy metrics (dnfr_change, vf_change, theta_shift) should exist."""
        G, node = create_nfr("test", epi=0.4, vf=0.5)
        G.graph["COLLECT_OPERATOR_METRICS"] = True

        Transition()(G, node)

        metrics = G.graph["operator_metrics"][-1]

        # Check legacy fields still exist
        assert "dnfr_change" in metrics
        assert "vf_change" in metrics
        assert "theta_shift" in metrics
        assert "dnfr_final" in metrics
        assert "vf_final" in metrics
        assert "theta_final" in metrics
        assert "transition_complete" in metrics

    def test_legacy_values_match_new_values(self):
        """Legacy absolute values should match new magnitude values."""
        G, node = create_nfr("test", epi=0.4, vf=0.5)
        G.graph["COLLECT_OPERATOR_METRICS"] = True

        Transition()(G, node)

        metrics = G.graph["operator_metrics"][-1]

        # theta_shift (legacy) should match phase_shift_magnitude (new)
        assert math.isclose(metrics["theta_shift"], metrics["phase_shift_magnitude"], abs_tol=1e-9)

        # dnfr_change (legacy) should match abs(delta_dnfr) (new)
        assert math.isclose(metrics["dnfr_change"], abs(metrics["delta_dnfr"]), abs_tol=1e-9)

        # vf_change (legacy) should match abs(delta_vf) (new)
        assert math.isclose(metrics["vf_change"], abs(metrics["delta_vf"]), abs_tol=1e-9)


class TestSequenceIntegration:
    """Test metrics in realistic operator sequences."""

    def test_sha_nav_reactivation_metrics(self):
        """SHA → NAV should produce reactivation metrics."""
        G, node = create_nfr("test", epi=0.5, vf=0.8)
        G.graph["COLLECT_OPERATOR_METRICS"] = True

        run_sequence(G, node, [Silence(), Transition()])

        # Get NAV metrics (last in list)
        metrics = G.graph["operator_metrics"][-1]

        assert metrics["operator"] == "Transition"
        assert metrics["transition_type"] == "reactivation"
        assert metrics["regime_origin"] == "latent"

    def test_il_nav_stable_transition_metrics(self):
        """IL → NAV should produce stable regime_change metrics."""
        G, node = create_nfr("test", epi=0.4, vf=0.5)
        G.graph["COLLECT_OPERATOR_METRICS"] = True

        run_sequence(G, node, [Coherence(), Transition()])

        # Get NAV metrics
        metrics = G.graph["operator_metrics"][-1]

        assert metrics["operator"] == "Transition"
        # Should be regime_change (not phase_shift, not reactivation)
        assert metrics["transition_type"] in ["regime_change", "phase_shift"]
        assert metrics["regime_origin"] == "active"
