"""Tests for ZHIR bifurcation potential detection (∂²EPI/∂t² > τ).

This module validates the implementation of ZHIR bifurcation detection according
to AGENTS.md §U4a (Bifurcation Dynamics):

    ∂²EPI/∂t² > τ → bifurcation potential detected (telemetry flags set)
    ∂²EPI/∂t² ≤ τ → no bifurcation potential (no flags set)

The detection is NON-CREATING (Option B) - ZHIR detects bifurcation potential
but does not create structural variants. This enables validation of grammar U4a
requirement: "If {OZ, ZHIR}, include {THOL, IL}" for controlled bifurcation.

Tests verify:
1. Bifurcation potential detection when ∂²EPI/∂t² > τ
2. No detection when ∂²EPI/∂t² ≤ τ
3. Telemetry flags set correctly (_zhir_bifurcation_potential, _zhir_d2epi)
4. Bifurcation events recorded in graph
5. Configuration parameters work correctly
6. Integration with existing ZHIR functionality
7. Backward compatibility
"""

import logging
import pytest
import math

from tnfr.structural import create_nfr, run_sequence
from tnfr.operators.definitions import Mutation, Dissonance, Coherence
from tnfr.constants.aliases import ALIAS_EPI, ALIAS_VF, ALIAS_THETA


class TestZHIRBifurcationDetection:
    """Test suite for ZHIR bifurcation potential detection."""

    def test_high_acceleration_detects_bifurcation_potential(self, caplog):
        """When ∂²EPI/∂t² > τ, bifurcation potential should be detected."""
        G, node = create_nfr("test", epi=0.5, vf=1.0)
        
        # Create EPI history with high acceleration
        # d²EPI = 0.50 - 2*0.35 + 0.25 = 0.50 - 0.70 + 0.25 = 0.05
        G.nodes[node]["epi_history"] = [0.25, 0.35, 0.50]
        G.nodes[node]["glyph_history"] = []
        
        # Set low threshold so we exceed it
        G.graph["BIFURCATION_THRESHOLD_TAU"] = 0.03
        
        with caplog.at_level(logging.INFO):
            Mutation()(G, node)
        
        # Telemetry flag should be set
        assert G.nodes[node].get("_zhir_bifurcation_potential") == True
        assert "_zhir_d2epi" in G.nodes[node]
        assert "_zhir_tau" in G.nodes[node]
        
        # Log message should indicate detection
        assert any("bifurcation potential detected" in record.message.lower()
                  for record in caplog.records if record.levelname == "INFO")

    def test_low_acceleration_no_detection(self, caplog):
        """When ∂²EPI/∂t² ≤ τ, no bifurcation potential should be detected."""
        G, node = create_nfr("test", epi=0.5, vf=1.0)
        
        # Create EPI history with LOW acceleration (nearly linear)
        # d²EPI = 0.50 - 2*0.49 + 0.48 = 0.50 - 0.98 + 0.48 = 0.00
        G.nodes[node]["epi_history"] = [0.48, 0.49, 0.50]
        G.nodes[node]["glyph_history"] = []
        
        # Set threshold higher than our acceleration
        G.graph["BIFURCATION_THRESHOLD_TAU"] = 0.1
        
        with caplog.at_level(logging.INFO):
            Mutation()(G, node)
        
        # Telemetry flag should NOT be set
        assert G.nodes[node].get("_zhir_bifurcation_potential") != True
        
        # No detection log should be present
        assert not any("bifurcation potential detected" in record.message.lower()
                      for record in caplog.records if record.levelname == "INFO")

    def test_telemetry_flags_set_correctly(self):
        """Telemetry flags should contain correct acceleration and threshold values."""
        G, node = create_nfr("test", epi=0.5, vf=1.0)
        
        # High acceleration: 0.10, 0.30, 0.60
        # d²EPI = 0.60 - 2*0.30 + 0.10 = 0.60 - 0.60 + 0.10 = 0.10
        G.nodes[node]["epi_history"] = [0.10, 0.30, 0.60]
        G.nodes[node]["glyph_history"] = []
        
        tau = 0.05
        G.graph["BIFURCATION_THRESHOLD_TAU"] = tau
        
        Mutation()(G, node)
        
        # Check all telemetry flags
        assert G.nodes[node]["_zhir_bifurcation_potential"] == True
        
        # d2_epi should be approximately 0.10
        d2_epi = G.nodes[node]["_zhir_d2epi"]
        assert abs(d2_epi - 0.10) < 0.01, f"Expected d2_epi ≈ 0.10, got {d2_epi}"
        
        # tau should match config
        assert G.nodes[node]["_zhir_tau"] == tau

    def test_bifurcation_events_recorded_in_graph(self):
        """Bifurcation detection events should be recorded in graph for analysis."""
        G, node = create_nfr("test", epi=0.5, vf=1.0)
        
        G.nodes[node]["epi_history"] = [0.10, 0.30, 0.60]
        G.nodes[node]["glyph_history"] = []
        G.graph["BIFURCATION_THRESHOLD_TAU"] = 0.05
        
        # Should not have events initially
        assert "zhir_bifurcation_events" not in G.graph
        
        Mutation()(G, node)
        
        # Event should be recorded
        events = G.graph.get("zhir_bifurcation_events", [])
        assert len(events) == 1
        
        event = events[0]
        assert event["node"] == node
        assert event["d2_epi"] > 0
        assert event["tau"] == 0.05
        assert "timestamp" in event

    def test_threshold_from_bifurcation_threshold_tau(self):
        """Should use canonical BIFURCATION_THRESHOLD_TAU config parameter."""
        G, node = create_nfr("test", epi=0.5, vf=1.0)
        
        G.nodes[node]["epi_history"] = [0.10, 0.30, 0.60]
        G.nodes[node]["glyph_history"] = []
        
        # Set canonical threshold
        G.graph["BIFURCATION_THRESHOLD_TAU"] = 0.05
        
        Mutation()(G, node)
        
        # Should detect with canonical threshold
        assert G.nodes[node]["_zhir_bifurcation_potential"] == True

    def test_threshold_from_zhir_bifurcation_threshold(self):
        """Should fall back to ZHIR_BIFURCATION_THRESHOLD if canonical not set."""
        G, node = create_nfr("test", epi=0.5, vf=1.0)
        
        G.nodes[node]["epi_history"] = [0.10, 0.30, 0.60]
        G.nodes[node]["glyph_history"] = []
        
        # Don't set canonical, use operator-specific config
        G.graph["ZHIR_BIFURCATION_THRESHOLD"] = 0.05
        
        Mutation()(G, node)
        
        # Should detect with operator-specific threshold
        assert G.nodes[node]["_zhir_bifurcation_potential"] == True

    def test_default_threshold_when_not_configured(self):
        """Should use default threshold 0.5 when not configured."""
        G, node = create_nfr("test", epi=0.5, vf=1.0)
        
        # Very low acceleration
        G.nodes[node]["epi_history"] = [0.48, 0.49, 0.50]
        G.nodes[node]["glyph_history"] = []
        # Don't set any threshold config
        
        Mutation()(G, node)
        
        # Should not detect with default threshold 0.5
        assert G.nodes[node].get("_zhir_bifurcation_potential") != True

    def test_tau_parameter_overrides_config(self):
        """Explicit tau parameter should override graph configuration."""
        G, node = create_nfr("test", epi=0.5, vf=1.0)
        
        G.nodes[node]["epi_history"] = [0.10, 0.30, 0.60]
        G.nodes[node]["glyph_history"] = []
        
        # Set config to high value
        G.graph["BIFURCATION_THRESHOLD_TAU"] = 1.0
        
        # But pass low value explicitly
        Mutation()(G, node, tau=0.05)
        
        # Should detect with explicit parameter
        assert G.nodes[node]["_zhir_bifurcation_potential"] == True
        assert G.nodes[node]["_zhir_tau"] == 0.05

    def test_log_message_suggests_handlers(self, caplog):
        """Log message should suggest THOL or IL for controlled bifurcation."""
        G, node = create_nfr("test", epi=0.5, vf=1.0)
        
        G.nodes[node]["epi_history"] = [0.10, 0.30, 0.60]
        G.nodes[node]["glyph_history"] = []
        G.graph["BIFURCATION_THRESHOLD_TAU"] = 0.05
        
        with caplog.at_level(logging.INFO):
            Mutation()(G, node)
        
        # Check log mentions handlers
        info_msg = next(
            r.message for r in caplog.records
            if r.levelname == "INFO" and "bifurcation potential" in r.message.lower()
        )
        assert "THOL" in info_msg or "IL" in info_msg


class TestZHIRBifurcationIntegration:
    """Integration tests with ZHIR in canonical sequences."""

    def test_oz_zhir_detects_bifurcation(self):
        """OZ → ZHIR sequence should detect bifurcation if acceleration high."""
        G, node = create_nfr("test", epi=0.4, vf=1.0)
        
        # Build high acceleration history
        G.nodes[node]["epi_history"] = [0.20, 0.32, 0.50]
        G.nodes[node]["glyph_history"] = []
        G.graph["BIFURCATION_THRESHOLD_TAU"] = 0.05
        
        # Apply canonical destabilizer → mutation sequence
        run_sequence(G, node, [Dissonance(), Mutation()])
        
        # Should detect bifurcation potential
        assert G.nodes[node].get("_zhir_bifurcation_potential") == True

    def test_il_oz_zhir_il_no_structural_changes(self):
        """Full sequence with detection should not create extra nodes/edges."""
        G, node = create_nfr("test", epi=0.4, vf=1.0)
        
        G.nodes[node]["epi_history"] = [0.20, 0.32, 0.50]
        G.nodes[node]["glyph_history"] = []
        G.graph["BIFURCATION_THRESHOLD_TAU"] = 0.05
        
        nodes_before = set(G.nodes())
        edges_before = set(G.edges())
        
        # IL → OZ → ZHIR → IL
        run_sequence(G, node, [Coherence(), Dissonance(), Mutation(), Coherence()])
        
        nodes_after = set(G.nodes())
        edges_after = set(G.edges())
        
        # Detection should NOT create new nodes or edges (Option B)
        assert nodes_before == nodes_after, "Detection should not create nodes"
        assert edges_before == edges_after, "Detection should not create edges"

    def test_zhir_preserves_existing_functionality(self):
        """ZHIR with bifurcation detection should still apply phase transformation."""
        G, node = create_nfr("test", epi=0.4, vf=1.0)
        
        theta_before = G.nodes[node]["theta"]
        G.nodes[node]["delta_nfr"] = 0.3  # For phase transformation
        G.nodes[node]["epi_history"] = [0.20, 0.32, 0.50]
        G.nodes[node]["glyph_history"] = []
        G.graph["BIFURCATION_THRESHOLD_TAU"] = 0.05
        
        Mutation()(G, node)
        
        theta_after = G.nodes[node]["theta"]
        
        # Phase transformation should still occur
        assert theta_after != theta_before, "ZHIR should still transform phase"
        
        # AND bifurcation should be detected
        assert G.nodes[node].get("_zhir_bifurcation_potential") == True


class TestZHIRBifurcationEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_insufficient_history_no_detection(self):
        """With <3 EPI history points, cannot compute d²EPI/dt²."""
        G, node = create_nfr("test", epi=0.5, vf=1.0)
        
        # Only 2 points
        G.nodes[node]["epi_history"] = [0.48, 0.50]
        G.nodes[node]["glyph_history"] = []
        G.graph["BIFURCATION_THRESHOLD_TAU"] = 0.05
        
        Mutation()(G, node)
        
        # Should not detect (insufficient data)
        assert G.nodes[node].get("_zhir_bifurcation_potential") != True

    def test_exactly_at_threshold_no_detection(self):
        """When d²EPI = τ (not >, equal), should not detect."""
        G, node = create_nfr("test", epi=0.5, vf=1.0)
        
        # Create history where d²EPI = 0.10 exactly
        # We'll set tau = 0.10, so d²EPI = τ (not >)
        G.nodes[node]["epi_history"] = [0.10, 0.30, 0.60]
        G.nodes[node]["glyph_history"] = []
        
        tau = 0.10  # Exactly equal to d²EPI
        G.graph["BIFURCATION_THRESHOLD_TAU"] = tau
        
        Mutation()(G, node)
        
        # Should not detect (need >, not >=)
        assert G.nodes[node].get("_zhir_bifurcation_potential") != True

    def test_negative_acceleration_uses_magnitude(self):
        """Acceleration computation should use magnitude (abs value)."""
        G, node = create_nfr("test", epi=0.5, vf=1.0)
        
        # Decelerating: d²EPI negative but |d²EPI| large
        # 0.60, 0.30, 0.10 → d²EPI = 0.10 - 0.60 + 0.60 = 0.10
        # Actually that's positive. Let's try:
        # 0.10, 0.40, 0.60 → d²EPI = 0.60 - 0.80 + 0.10 = -0.10
        # |d²EPI| = 0.10
        G.nodes[node]["epi_history"] = [0.10, 0.40, 0.60]
        G.nodes[node]["glyph_history"] = []
        G.graph["BIFURCATION_THRESHOLD_TAU"] = 0.05
        
        Mutation()(G, node)
        
        # Should detect based on magnitude
        d2_epi = G.nodes[node].get("_zhir_d2epi", 0.0)
        assert d2_epi > 0, "Should store magnitude (positive)"
        assert G.nodes[node].get("_zhir_bifurcation_potential") == True

    def test_multiple_zhir_calls_accumulate_events(self):
        """Multiple ZHIR calls should accumulate events in graph."""
        G, node = create_nfr("test", epi=0.5, vf=1.0)
        
        G.nodes[node]["epi_history"] = [0.10, 0.30, 0.60]
        G.nodes[node]["glyph_history"] = []
        G.graph["BIFURCATION_THRESHOLD_TAU"] = 0.05
        
        # First call
        Mutation()(G, node)
        assert len(G.graph["zhir_bifurcation_events"]) == 1
        
        # Update history and call again
        G.nodes[node]["epi_history"].append(0.75)
        Mutation()(G, node)
        
        # Should have 2 events
        assert len(G.graph["zhir_bifurcation_events"]) == 2


class TestZHIRBifurcationBackwardCompatibility:
    """Test backward compatibility with existing ZHIR behavior."""

    def test_existing_zhir_tests_still_pass(self):
        """ZHIR should work without epi_history (backward compatibility)."""
        G, node = create_nfr("test", epi=0.5, vf=1.0)
        
        # No epi_history set (legacy test setup)
        G.nodes[node]["delta_nfr"] = 0.3
        G.nodes[node]["glyph_history"] = []
        
        # Should not raise error
        Mutation()(G, node)
        
        # Should not detect (no history)
        assert G.nodes[node].get("_zhir_bifurcation_potential") != True

    def test_no_config_changes_required(self):
        """ZHIR should work with default config (no breaking changes)."""
        G, node = create_nfr("test", epi=0.5, vf=1.0)
        
        G.nodes[node]["epi_history"] = [0.48, 0.49, 0.50]
        G.nodes[node]["glyph_history"] = []
        # No config set at all
        
        # Should not raise error
        Mutation()(G, node)  # OK

    def test_no_api_changes(self):
        """Mutation() call signature should be unchanged."""
        G, node = create_nfr("test", epi=0.5, vf=1.0)
        G.nodes[node]["glyph_history"] = []
        
        # All existing call patterns should work
        Mutation()(G, node)  # Standard
        Mutation()(G, node, validate_preconditions=False)  # With kwargs
        Mutation()(G, node, collect_metrics=True)  # With metrics


class TestZHIRBifurcationGrammarU4a:
    """Test grammar U4a validation support."""

    def test_detection_enables_u4a_validation(self):
        """Bifurcation detection should enable validation of U4a requirement."""
        G, node = create_nfr("test", epi=0.5, vf=1.0)
        
        G.nodes[node]["epi_history"] = [0.10, 0.30, 0.60]
        G.nodes[node]["glyph_history"] = []
        G.graph["BIFURCATION_THRESHOLD_TAU"] = 0.05
        
        Mutation()(G, node)
        
        # Telemetry should enable U4a check
        # Grammar validator can check: if ZHIR and _zhir_bifurcation_potential,
        # then sequence should have THOL or IL nearby
        assert G.nodes[node]["_zhir_bifurcation_potential"] == True
        assert "_zhir_d2epi" in G.nodes[node]

    def test_no_detection_no_u4a_requirement(self):
        """Without bifurcation detection, U4a not required for ZHIR."""
        G, node = create_nfr("test", epi=0.5, vf=1.0)
        
        # Low acceleration
        G.nodes[node]["epi_history"] = [0.48, 0.49, 0.50]
        G.nodes[node]["glyph_history"] = []
        G.graph["BIFURCATION_THRESHOLD_TAU"] = 0.5
        
        Mutation()(G, node)
        
        # No bifurcation potential, U4a not triggered
        assert G.nodes[node].get("_zhir_bifurcation_potential") != True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
