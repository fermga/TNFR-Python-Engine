"""Unit tests for extended SHA (Silence) operator metrics.

This module tests the extended metrics functionality for the SHA operator
including EPI variance, preservation integrity, reactivation readiness,
and time-to-collapse estimation.

These metrics enable deep analysis of structural preservation effectiveness
as specified in the enhancement request.
"""

from __future__ import annotations

import math

import pytest

from tnfr.structural import create_nfr
from tnfr.operators.metrics import (
    silence_metrics,
    _compute_epi_variance,
    _compute_preservation_integrity,
    _compute_reactivation_readiness,
    _estimate_time_to_collapse,
)
from tnfr.alias import set_attr
from tnfr.constants.aliases import ALIAS_VF, ALIAS_EPI


class TestComputeEPIVariance:
    """Test _compute_epi_variance helper function."""

    def test_no_history_returns_zero(self):
        """Empty history should return 0.0 variance."""
        G, node = create_nfr("test", epi=0.5, vf=1.0)
        # No epi_history_during_silence attribute
        variance = _compute_epi_variance(G, node)
        assert variance == 0.0

    def test_single_value_returns_zero(self):
        """Single value in history should return 0.0 variance."""
        G, node = create_nfr("test", epi=0.5, vf=1.0)
        G.nodes[node]["epi_history_during_silence"] = [0.5]
        variance = _compute_epi_variance(G, node)
        assert variance == 0.0

    def test_constant_values_returns_zero(self):
        """Constant values should return 0.0 variance."""
        G, node = create_nfr("test", epi=0.5, vf=1.0)
        G.nodes[node]["epi_history_during_silence"] = [0.5, 0.5, 0.5, 0.5]
        variance = _compute_epi_variance(G, node)
        assert variance == pytest.approx(0.0, abs=1e-10)

    def test_varying_values_returns_positive_variance(self):
        """Varying values should return positive variance."""
        G, node = create_nfr("test", epi=0.5, vf=1.0)
        G.nodes[node]["epi_history_during_silence"] = [0.5, 0.6, 0.4, 0.55]
        variance = _compute_epi_variance(G, node)
        assert variance > 0.0
        # Verify it's reasonable (should be around 0.07 for these values)
        assert 0.05 < variance < 0.1


class TestComputePreservationIntegrity:
    """Test _compute_preservation_integrity helper function."""

    def test_perfect_preservation(self):
        """Identical values should give integrity = 1.0."""
        integrity = _compute_preservation_integrity(0.5, 0.5)
        assert integrity == pytest.approx(1.0)

    def test_zero_preserved_zero_after(self):
        """Both zero should give integrity = 1.0."""
        integrity = _compute_preservation_integrity(0.0, 0.0)
        assert integrity == 1.0

    def test_zero_preserved_nonzero_after(self):
        """Zero preserved with nonzero after should give integrity = 0.0."""
        integrity = _compute_preservation_integrity(0.0, 0.5)
        assert integrity == 0.0

    def test_small_drift(self):
        """Small drift should give high integrity."""
        integrity = _compute_preservation_integrity(0.5, 0.52)
        assert integrity > 0.95
        assert integrity < 1.0

    def test_large_drift(self):
        """Large drift should give low integrity."""
        integrity = _compute_preservation_integrity(0.5, 0.8)
        assert integrity < 0.5

    def test_complete_loss(self):
        """Complete loss should give integrity near 0."""
        integrity = _compute_preservation_integrity(0.5, 0.0)
        assert integrity == 0.0

    def test_negative_integrity_clamped_to_zero(self):
        """Negative integrity values should be clamped to 0.0."""
        # If after > preserved by more than preserved itself
        integrity = _compute_preservation_integrity(0.5, 1.5)
        assert integrity == 0.0


class TestComputeReactivationReadiness:
    """Test _compute_reactivation_readiness helper function."""

    def test_optimal_conditions_high_readiness(self):
        """Optimal conditions should give high readiness."""
        G, node = create_nfr("test", epi=0.5, vf=0.5)
        G.nodes[node]["silence_duration"] = 0.0

        # Add active neighbors
        for i in range(3):
            G.add_node(f"neighbor_{i}")
            set_attr(G.nodes[f"neighbor_{i}"], ALIAS_VF, 0.8)
            G.add_edge(node, f"neighbor_{i}")

        readiness = _compute_reactivation_readiness(G, node)
        assert readiness > 0.8
        assert readiness <= 1.0

    def test_low_vf_reduces_readiness(self):
        """Low νf should reduce readiness."""
        G, node = create_nfr("test", epi=0.5, vf=0.05)
        G.nodes[node]["silence_duration"] = 0.0
        readiness = _compute_reactivation_readiness(G, node)
        assert readiness < 0.6  # Adjusted: low vf reduces readiness

    def test_low_epi_reduces_readiness(self):
        """Low EPI should reduce readiness."""
        G, node = create_nfr("test", epi=0.05, vf=0.5)
        G.nodes[node]["silence_duration"] = 0.0
        readiness = _compute_reactivation_readiness(G, node)
        assert readiness < 0.6  # Adjusted: low EPI reduces readiness

    def test_long_duration_reduces_readiness(self):
        """Long silence duration should reduce readiness."""
        G, node = create_nfr("test", epi=0.5, vf=0.5)
        G.nodes[node]["silence_duration"] = 10.0  # Long duration
        readiness = _compute_reactivation_readiness(G, node)
        assert readiness < 0.8

    def test_no_active_neighbors_reduces_readiness(self):
        """No active neighbors should reduce readiness."""
        G, node = create_nfr("test", epi=0.5, vf=0.5)
        G.nodes[node]["silence_duration"] = 0.0

        # Add inactive neighbors
        for i in range(3):
            G.add_node(f"neighbor_{i}")
            set_attr(G.nodes[f"neighbor_{i}"], ALIAS_VF, 0.01)  # Inactive
            G.add_edge(node, f"neighbor_{i}")

        readiness = _compute_reactivation_readiness(G, node)
        # Should be lower than with active neighbors
        assert readiness < 0.8

    def test_readiness_bounded_zero_one(self):
        """Readiness should always be in [0, 1]."""
        # Extreme values
        G, node = create_nfr("test", epi=0.0, vf=0.0)
        G.nodes[node]["silence_duration"] = 100.0
        readiness = _compute_reactivation_readiness(G, node)
        assert 0.0 <= readiness <= 1.0


class TestEstimateTimeToCollapse:
    """Test _estimate_time_to_collapse helper function."""

    def test_no_drift_returns_infinity(self):
        """No drift should return infinite time."""
        G, node = create_nfr("test", epi=0.5, vf=0.5)
        G.nodes[node]["preserved_epi"] = 0.5
        G.nodes[node]["epi_drift_rate"] = 0.0
        time = _estimate_time_to_collapse(G, node)
        assert math.isinf(time)

    def test_positive_drift_finite_time(self):
        """Positive drift should give finite time."""
        G, node = create_nfr("test", epi=0.5, vf=0.5)
        G.nodes[node]["preserved_epi"] = 0.5
        G.nodes[node]["epi_drift_rate"] = 0.05  # 5% drift per step
        time = _estimate_time_to_collapse(G, node)
        assert not math.isinf(time)
        assert time == pytest.approx(10.0)  # 0.5 / 0.05 = 10

    def test_negative_drift_finite_time(self):
        """Negative drift should give finite time (abs value used)."""
        G, node = create_nfr("test", epi=0.5, vf=0.5)
        G.nodes[node]["preserved_epi"] = 0.5
        G.nodes[node]["epi_drift_rate"] = -0.05
        time = _estimate_time_to_collapse(G, node)
        assert not math.isinf(time)
        assert time == pytest.approx(10.0)

    def test_zero_preserved_epi_returns_zero(self):
        """Zero preserved EPI should return 0 time."""
        G, node = create_nfr("test", epi=0.0, vf=0.5)
        G.nodes[node]["preserved_epi"] = 0.0
        G.nodes[node]["epi_drift_rate"] = 0.05
        time = _estimate_time_to_collapse(G, node)
        assert time == 0.0

    def test_missing_drift_rate_returns_infinity(self):
        """Missing drift rate should default to 0.0 (infinity)."""
        G, node = create_nfr("test", epi=0.5, vf=0.5)
        G.nodes[node]["preserved_epi"] = 0.5
        # No drift_rate attribute
        time = _estimate_time_to_collapse(G, node)
        assert math.isinf(time)


class TestSilenceMetricsExtended:
    """Test silence_metrics function with extended metrics."""

    def test_extended_metrics_present(self):
        """Extended metrics should be present in output."""
        G, node = create_nfr("test", epi=0.5, vf=1.0)
        vf_before = 1.0
        epi_before = 0.5

        metrics = silence_metrics(G, node, vf_before, epi_before)

        # Core metrics (existing)
        assert "operator" in metrics
        assert "glyph" in metrics
        assert "vf_reduction" in metrics
        assert "vf_final" in metrics
        assert "epi_preservation" in metrics
        assert "epi_final" in metrics
        assert "is_silent" in metrics

        # Extended metrics (new)
        assert "epi_variance" in metrics
        assert "preservation_integrity" in metrics
        assert "reactivation_readiness" in metrics
        assert "time_to_collapse" in metrics

    def test_core_metrics_unchanged(self):
        """Core metrics should remain consistent."""
        G, node = create_nfr("test", epi=0.5, vf=0.05)
        vf_before = 1.0
        epi_before = 0.5

        metrics = silence_metrics(G, node, vf_before, epi_before)

        assert metrics["operator"] == "Silence"
        assert metrics["glyph"] == "SHA"
        assert metrics["vf_reduction"] == pytest.approx(0.95)
        assert metrics["vf_final"] == pytest.approx(0.05)
        assert metrics["is_silent"] is True

    def test_preservation_integrity_with_preserved_epi(self):
        """Preservation integrity should use preserved_epi when available."""
        G, node = create_nfr("test", epi=0.5, vf=0.05)
        G.nodes[node]["preserved_epi"] = 0.5
        vf_before = 1.0
        epi_before = 0.5

        metrics = silence_metrics(G, node, vf_before, epi_before)

        # Perfect preservation
        assert metrics["preservation_integrity"] == pytest.approx(1.0)

    def test_preservation_integrity_without_preserved_epi(self):
        """Preservation integrity should use epi_before when preserved_epi absent."""
        G, node = create_nfr("test", epi=0.5, vf=0.05)
        # No preserved_epi attribute
        vf_before = 1.0
        epi_before = 0.5

        metrics = silence_metrics(G, node, vf_before, epi_before)

        # Should compute from epi_before
        assert "preservation_integrity" in metrics
        assert metrics["preservation_integrity"] >= 0.0

    def test_epi_variance_computed(self):
        """EPI variance should be computed from history."""
        G, node = create_nfr("test", epi=0.5, vf=0.05)
        G.nodes[node]["epi_history_during_silence"] = [0.5, 0.52, 0.48, 0.51]
        vf_before = 1.0
        epi_before = 0.5

        metrics = silence_metrics(G, node, vf_before, epi_before)

        assert metrics["epi_variance"] > 0.0

    def test_reactivation_readiness_computed(self):
        """Reactivation readiness should be in [0, 1]."""
        G, node = create_nfr("test", epi=0.5, vf=0.05)
        vf_before = 1.0
        epi_before = 0.5

        metrics = silence_metrics(G, node, vf_before, epi_before)

        assert 0.0 <= metrics["reactivation_readiness"] <= 1.0

    def test_time_to_collapse_computed(self):
        """Time to collapse should be computed."""
        G, node = create_nfr("test", epi=0.5, vf=0.05)
        G.nodes[node]["preserved_epi"] = 0.5
        G.nodes[node]["epi_drift_rate"] = 0.05
        vf_before = 1.0
        epi_before = 0.5

        metrics = silence_metrics(G, node, vf_before, epi_before)

        assert metrics["time_to_collapse"] > 0.0
        assert metrics["time_to_collapse"] == pytest.approx(10.0)

    def test_latency_state_preserved(self):
        """Latency state tracking should still work."""
        G, node = create_nfr("test", epi=0.5, vf=0.05)
        G.nodes[node]["latent"] = True
        G.nodes[node]["silence_duration"] = 5.0
        vf_before = 1.0
        epi_before = 0.5

        metrics = silence_metrics(G, node, vf_before, epi_before)

        assert metrics["latent"] is True
        assert metrics["silence_duration"] == 5.0


class TestSilenceMetricsIntegration:
    """Integration tests for extended silence metrics."""

    def test_biomedical_use_case(self):
        """Test biomedical use case: sleep consolidation tracking."""
        # Simulate a node in sleep state with minimal drift
        G, node = create_nfr("hrv_signal", epi=0.6, vf=0.02)
        G.nodes[node]["preserved_epi"] = 0.6
        G.nodes[node]["silence_duration"] = 8.0  # 8 hours sleep
        G.nodes[node]["epi_history_during_silence"] = [0.6, 0.602, 0.598, 0.601]
        G.nodes[node]["epi_drift_rate"] = 0.001  # Very slow drift

        metrics = silence_metrics(G, node, vf_before=0.5, epi_before=0.6)

        # Should show high preservation quality
        assert metrics["preservation_integrity"] > 0.99
        # Low variance indicates stable sleep
        assert metrics["epi_variance"] < 0.01
        # Readiness should be moderate (long duration but stable)
        assert 0.3 < metrics["reactivation_readiness"] < 0.8
        # Long time to collapse due to slow drift
        assert metrics["time_to_collapse"] > 100.0

    def test_cognitive_use_case(self):
        """Test cognitive use case: memory consolidation."""
        # Simulate memory trace during incubation period
        G, node = create_nfr("memory_trace", epi=0.7, vf=0.03)
        G.nodes[node]["preserved_epi"] = 0.7
        G.nodes[node]["silence_duration"] = 2.0  # Brief pause
        G.nodes[node]["epi_history_during_silence"] = [0.7, 0.7, 0.7]  # Perfect stability

        # Add network support (other memory traces)
        for i in range(4):
            G.add_node(f"memory_{i}")
            set_attr(G.nodes[f"memory_{i}"], ALIAS_VF, 0.5)
            set_attr(G.nodes[f"memory_{i}"], ALIAS_EPI, 0.6)
            G.add_edge(node, f"memory_{i}")

        metrics = silence_metrics(G, node, vf_before=0.5, epi_before=0.7)

        # Perfect preservation
        assert metrics["preservation_integrity"] == pytest.approx(1.0)
        # Zero variance (perfect consolidation)
        assert metrics["epi_variance"] == pytest.approx(0.0, abs=1e-10)
        # High readiness (good conditions + network support)
        assert metrics["reactivation_readiness"] > 0.7

    def test_social_use_case(self):
        """Test social use case: strategic pause in conflict."""
        # Simulate conflict node paused for strategic reasons
        G, node = create_nfr("conflict_state", epi=0.4, vf=0.01)
        G.nodes[node]["preserved_epi"] = 0.45  # Slight EPI degradation
        G.nodes[node]["silence_duration"] = 15.0  # Long pause
        G.nodes[node]["epi_drift_rate"] = 0.01  # Slow degradation

        metrics = silence_metrics(G, node, vf_before=0.8, epi_before=0.45)

        # Some degradation expected
        assert 0.85 < metrics["preservation_integrity"] < 0.95
        # Long duration reduces readiness
        assert metrics["reactivation_readiness"] < 0.6
        # Finite time to collapse
        assert 30.0 < metrics["time_to_collapse"] < 50.0
