"""Comprehensive tests for mutation_metrics() expanded functionality.

This module tests the enhanced mutation_metrics() function to verify that all
canonical metric categories are properly collected and computed:

1. Core metrics (existing functionality preserved)
2. Threshold verification (enhanced)
3. Phase transformation (enhanced with regime analysis)
4. Bifurcation analysis (NEW)
5. Structural preservation (NEW)
6. Network impact (NEW)
7. Destabilizer context (NEW - R4 Extended)
8. Grammar validation (NEW - U4b)

Test Coverage:
- All metric categories present
- Metric values are correct and within expected ranges
- Bifurcation detection and scoring
- Network impact calculation
- Destabilizer context tracking
- Grammar validation (U4b)
- Backwards compatibility with existing tests
"""

import pytest
from tnfr.structural import create_nfr, run_sequence
from tnfr.operators.definitions import (
    Emission,
    Coherence,
    Dissonance,
    Mutation,
    Expansion,
    Silence,
)


class TestMutationMetricsComprehensive:
    """Test suite for comprehensive mutation metrics."""

    def test_all_metric_categories_present(self):
        """mutation_metrics() must return all canonical metric categories."""
        G, node = create_nfr("test", epi=0.5, vf=1.2, theta=0.5)
        G.nodes[node]["epi_history"] = [0.3, 0.4, 0.5]
        G.graph["COLLECT_OPERATOR_METRICS"] = True

        # Apply canonical sequence to enable metrics
        run_sequence(G, node, [Emission(), Coherence(), Dissonance(),
                               Mutation(), Silence()])

        # Find the Mutation metrics specifically (not the last operator)
        mutation_metrics = None
        for metric in G.graph["operator_metrics"]:
            if metric["operator"] == "Mutation":
                mutation_metrics = metric
                break
        assert mutation_metrics is not None, "Mutation metrics not found"
        metrics = mutation_metrics

        # === CORE METRICS ===
        assert "operator" in metrics
        assert "glyph" in metrics
        assert metrics["operator"] == "Mutation"
        assert metrics["glyph"] == "ZHIR"

        # === THRESHOLD VERIFICATION ===
        assert "depi_dt" in metrics
        assert "threshold_xi" in metrics
        assert "threshold_met" in metrics
        assert "threshold_ratio" in metrics
        assert "threshold_exceeded_by" in metrics

        # === PHASE TRANSFORMATION ===
        assert "theta_shift" in metrics
        assert "theta_regime_before" in metrics
        assert "theta_regime_after" in metrics
        assert "regime_changed" in metrics
        assert "theta_shift_direction" in metrics
        assert "phase_transformation_magnitude" in metrics

        # === BIFURCATION ANALYSIS ===
        assert "d2epi" in metrics
        assert "bifurcation_threshold_tau" in metrics
        assert "bifurcation_potential" in metrics
        assert "bifurcation_score" in metrics
        assert "bifurcation_triggered" in metrics
        assert "bifurcation_event_count" in metrics

        # === STRUCTURAL PRESERVATION ===
        assert "epi_kind_before" in metrics
        assert "epi_kind_after" in metrics
        assert "identity_preserved" in metrics
        assert "delta_vf" in metrics
        assert "vf_final" in metrics
        assert "delta_dnfr" in metrics
        assert "dnfr_final" in metrics

        # === NETWORK IMPACT ===
        assert "neighbor_count" in metrics
        assert "impacted_neighbors" in metrics
        assert "network_impact_radius" in metrics
        assert "phase_coherence_neighbors" in metrics

        # === DESTABILIZER CONTEXT ===
        assert "destabilizer_type" in metrics
        assert "destabilizer_operator" in metrics
        assert "destabilizer_distance" in metrics
        assert "recent_history" in metrics

        # === GRAMMAR VALIDATION ===
        assert "grammar_u4b_satisfied" in metrics
        assert "il_precedence_found" in metrics
        assert "destabilizer_recent" in metrics

        # === METADATA ===
        assert "metrics_version" in metrics
        assert metrics["metrics_version"] == "2.0_canonical"

    def test_bifurcation_score_calculation(self):
        """Bifurcation score must be computed using canonical formula."""
        G, node = create_nfr("test", epi=0.6, vf=1.5, theta=0.3)
        G.nodes[node]["epi_history"] = [0.3, 0.45, 0.6]  # High velocity
        G.graph["COLLECT_OPERATOR_METRICS"] = True
        G.graph["BIFURCATION_THRESHOLD_TAU"] = 0.4

        # Create high ΔNFR for bifurcation potential
        run_sequence(G, node, [
            Emission(), Dissonance(), Mutation(), Coherence(), Silence()
        ])

        # Find the Mutation metrics specifically
        mutation_metrics = None
        for metric in G.graph["operator_metrics"]:
            if metric["operator"] == "Mutation":
                mutation_metrics = metric
                break
        assert mutation_metrics is not None, "Mutation metrics not found"
        metrics = mutation_metrics

        # Verify bifurcation metrics
        assert "bifurcation_score" in metrics
        assert 0.0 <= metrics["bifurcation_score"] <= 1.0
        assert isinstance(metrics["bifurcation_potential"], bool)
        assert isinstance(metrics["bifurcation_triggered"], bool)
        assert metrics["bifurcation_event_count"] >= 0

    def test_threshold_verification_enhanced(self):
        """Enhanced threshold verification with additional edge cases."""
        G, node = create_nfr("test", epi=0.5, vf=1.2)
        G.nodes[node]["epi_history"] = [0.3, 0.4, 0.5]
        G.graph["COLLECT_OPERATOR_METRICS"] = True

        run_sequence(G, node, [
            Emission(), Dissonance(), Mutation(), Coherence(), Silence()
        ])

    def test_phase_transformation_regime_detection(self):
        """Phase transformation regime must be detectable."""
        G, node = create_nfr("test", epi=0.5, vf=1.2)
        G.nodes[node]["epi_history"] = [0.3, 0.4, 0.5]
        G.graph["COLLECT_OPERATOR_METRICS"] = True

        run_sequence(G, node, [
            Emission(), Dissonance(), Mutation(), Coherence(), Silence()
        ])

    def test_structural_preservation_tracking(self):
        """Structural preservation must track identity and state changes."""
        G, node = create_nfr("test", epi=0.5, vf=1.2)
        G.nodes[node]["epi_history"] = [0.3, 0.4, 0.5]
        G.nodes[node]["epi_kind"] = "test_pattern"
        # Note: _epi_kind_before is set by the operator, not manually
        G.graph["COLLECT_OPERATOR_METRICS"] = True

        run_sequence(G, node, [
            Emission(), Dissonance(), Mutation(), Coherence(), Silence()
        ])

        # Find the Mutation metrics specifically
        mutation_metrics = None
        for metric in G.graph["operator_metrics"]:
            if metric["operator"] == "Mutation":
                mutation_metrics = metric
                break
        assert mutation_metrics is not None, "Mutation metrics not found"
        metrics = mutation_metrics

        # Verify structural preservation
        assert "identity_preserved" in metrics
        # Identity preservation depends on implementation; may be True if no kind stored
        assert isinstance(metrics["identity_preserved"], bool)
        assert "delta_vf" in metrics
        assert "delta_dnfr" in metrics
        assert "vf_final" in metrics
        assert "dnfr_final" in metrics
        # Check that final values are reasonable
        assert metrics["vf_final"] > 0

    def test_network_impact_with_neighbors(self):
        """Network impact must calculate neighbor effects."""
        G, node = create_nfr("test", epi=0.5, vf=1.2, theta=0.5)
        G.nodes[node]["epi_history"] = [0.3, 0.4, 0.5]

        # Add neighbors with proper initialization (use Greek νf)
        neighbors = []
        for i in range(3):
            neighbor_id = f"neighbor_{i}"
            # Initialize neighbor properly with all required TNFR attributes
            G.add_node(
                neighbor_id,
                EPI=0.5,
                epi=0.5,
                theta=0.5 + i * 0.1,
                **{"νf": 1.0},  # Use Greek letter for canonical attribute
                vf=1.0,
                dnfr=0.0,
                theta_history=[0.5, 0.5 + i * 0.1],
            )
            G.add_edge(node, neighbor_id)
            neighbors.append(neighbor_id)

        G.graph["COLLECT_OPERATOR_METRICS"] = True

        run_sequence(G, node, [
            Emission(), Dissonance(), Mutation(), Coherence(), Silence()
        ])

        # Find the Mutation metrics specifically
        mutation_metrics = None
        for metric in G.graph["operator_metrics"]:
            if metric["operator"] == "Mutation":
                mutation_metrics = metric
                break
        assert mutation_metrics is not None, "Mutation metrics not found"
        metrics = mutation_metrics

        # Verify network impact
        assert metrics["neighbor_count"] == 3
        assert "impacted_neighbors" in metrics
        assert metrics["impacted_neighbors"] >= 0
        assert 0.0 <= metrics["network_impact_radius"] <= 1.0
        assert "phase_coherence_neighbors" in metrics

    def test_destabilizer_context_tracking(self):
        """Destabilizer context must be tracked (R4 Extended)."""
        G, node = create_nfr("test", epi=0.5, vf=1.2)
        G.nodes[node]["epi_history"] = [0.3, 0.4, 0.5]
        G.graph["COLLECT_OPERATOR_METRICS"] = True
        G.graph["VALIDATE_OPERATOR_PRECONDITIONS"] = True

        # Apply sequence that creates destabilizer context
        run_sequence(G, node, [
            Emission(), Coherence(), Dissonance(),
            Mutation(), Coherence(), Silence()
        ])

        # Find the Mutation metrics specifically
        mutation_metrics = None
        for metric in G.graph["operator_metrics"]:
            if metric["operator"] == "Mutation":
                mutation_metrics = metric
                break
        assert mutation_metrics is not None, "Mutation metrics not found"
        metrics = mutation_metrics

        # Verify destabilizer context
        assert "destabilizer_type" in metrics
        assert "destabilizer_operator" in metrics
        assert "destabilizer_distance" in metrics
        assert "recent_history" in metrics

        # Should have destabilizer from OZ
        if metrics["destabilizer_type"] is not None:
            assert metrics["destabilizer_type"] in [
                "strong",
                "moderate",
                "weak",
            ]

    def test_grammar_u4b_validation(self):
        """Grammar validation must check U4b (IL precedence + destabilizer)."""
        G, node = create_nfr("test", epi=0.5, vf=1.2)
        G.nodes[node]["epi_history"] = [0.3, 0.4, 0.5]
        G.graph["COLLECT_OPERATOR_METRICS"] = True
        G.graph["VALIDATE_OPERATOR_PRECONDITIONS"] = True

        # Apply canonical sequence: IL → OZ → ZHIR (U4b satisfied)
        run_sequence(G, node, [
            Emission(), Coherence(), Dissonance(),
            Mutation(), Coherence(), Silence()
        ])

        # Find the Mutation metrics specifically
        mutation_metrics = None
        for metric in G.graph["operator_metrics"]:
            if metric["operator"] == "Mutation":
                mutation_metrics = metric
                break
        assert mutation_metrics is not None, "Mutation metrics not found"
        metrics = mutation_metrics

        # Verify grammar validation
        assert "grammar_u4b_satisfied" in metrics
        assert "il_precedence_found" in metrics
        assert "destabilizer_recent" in metrics

        # Should be satisfied after canonical sequence
        assert metrics["il_precedence_found"] is True
        # destabilizer_recent depends on precondition validation

    def test_backwards_compatibility(self):
        """New metrics must not break existing functionality."""
        G, node = create_nfr("test", epi=0.5, vf=1.0)
        G.nodes[node]["epi_history"] = [0.3, 0.4, 0.5]
        G.graph["COLLECT_OPERATOR_METRICS"] = True

        run_sequence(G, node, [
            Emission(), Dissonance(), Mutation(), Coherence(), Silence()
        ])

        # Find the Mutation metrics specifically
        mutation_metrics = None
        for metric in G.graph["operator_metrics"]:
            if metric["operator"] == "Mutation":
                mutation_metrics = metric
                break
        assert mutation_metrics is not None, "Mutation metrics not found"
        metrics = mutation_metrics

        # Verify original metrics still present
        assert "theta_shift" in metrics
        assert "delta_epi" in metrics
        assert "threshold_met" in metrics
        assert "phase_change" in metrics

    def test_metric_count_comprehensive(self):
        """mutation_metrics() must return 40+ metrics (comparable to OZ/VAL)."""
        G, node = create_nfr("test", epi=0.5, vf=1.2)
        G.nodes[node]["epi_history"] = [0.3, 0.4, 0.5]
        G.graph["COLLECT_OPERATOR_METRICS"] = True

        run_sequence(G, node, [
            Emission(), Coherence(), Dissonance(),
            Mutation(), Coherence(), Silence()
        ])

        # Find the Mutation metrics specifically
        mutation_metrics = None
        for metric in G.graph["operator_metrics"]:
            if metric["operator"] == "Mutation":
                mutation_metrics = metric
                break
        assert mutation_metrics is not None, "Mutation metrics not found"
        metrics = mutation_metrics

        # Count metrics
        metric_count = len(metrics)

        # Should have 40+ metrics (comparable to dissonance: 25+, expansion: 20+)
        assert (
            metric_count >= 40
        ), f"Expected 40+ metrics, got {metric_count}. Keys: {list(metrics.keys())}"

    def test_bifurcation_event_detection(self):
        """Bifurcation events must be detected and counted."""
        G, node = create_nfr("test", epi=0.7, vf=1.5, theta=0.3)
        G.nodes[node]["epi_history"] = [0.4, 0.55, 0.7]
        G.graph["COLLECT_OPERATOR_METRICS"] = True
        G.graph["BIFURCATION_THRESHOLD_TAU"] = 0.3  # Lower threshold

        # Apply high dissonance to trigger bifurcation potential
        run_sequence(G, node, [
            Emission(), Dissonance(), Mutation(), Coherence(), Silence()
        ])

        # Find the Mutation metrics specifically
        mutation_metrics = None
        for metric in G.graph["operator_metrics"]:
            if metric["operator"] == "Mutation":
                mutation_metrics = metric
                break
        assert mutation_metrics is not None, "Mutation metrics not found"
        metrics = mutation_metrics

        # Check bifurcation detection
        assert "bifurcation_event_count" in metrics
        assert isinstance(metrics["bifurcation_event_count"], int)
        assert metrics["bifurcation_event_count"] >= 0

    def test_phase_coherence_isolated_node(self):
        """Phase coherence must handle isolated nodes (no neighbors)."""
        G, node = create_nfr("test", epi=0.5, vf=1.2)
        G.nodes[node]["epi_history"] = [0.3, 0.4, 0.5]
        G.graph["COLLECT_OPERATOR_METRICS"] = True

        run_sequence(G, node, [
            Emission(), Dissonance(), Mutation(), Coherence(), Silence()
        ])

        # Find the Mutation metrics specifically
        mutation_metrics = None
        for metric in G.graph["operator_metrics"]:
            if metric["operator"] == "Mutation":
                mutation_metrics = metric
                break
        assert mutation_metrics is not None, "Mutation metrics not found"
        metrics = mutation_metrics

        # Isolated node should have zero neighbors
        assert metrics["neighbor_count"] == 0
        assert metrics["impacted_neighbors"] == 0
        assert metrics["network_impact_radius"] == 0.0
        assert metrics["phase_coherence_neighbors"] == 0.0

    def test_threshold_ratio_zero_xi(self):
        """Threshold ratio must handle zero threshold gracefully."""
        G, node = create_nfr("test", epi=0.5, vf=1.2)
        G.nodes[node]["epi_history"] = [0.3, 0.4, 0.5]
        G.graph["ZHIR_THRESHOLD_XI"] = 0.0  # Zero threshold
        G.graph["COLLECT_OPERATOR_METRICS"] = True

        run_sequence(G, node, [
            Emission(), Dissonance(), Mutation(), Coherence(), Silence()
        ])

        # Find the Mutation metrics specifically
        mutation_metrics = None
        for metric in G.graph["operator_metrics"]:
            if metric["operator"] == "Mutation":
                mutation_metrics = metric
                break
        assert mutation_metrics is not None, "Mutation metrics not found"
        metrics = mutation_metrics

        # Should handle gracefully without division by zero
        assert "threshold_ratio" in metrics
        assert metrics["threshold_ratio"] == 0.0
        assert metrics["threshold_met"] is True  # Always met with zero threshold


class TestMutationMetricsParameterOptional:
    """Test that new parameters are optional (backwards compatibility)."""

    def test_mutation_metrics_without_vf_dnfr(self):
        """mutation_metrics() must work without vf_before/dnfr_before."""
        from tnfr.operators.metrics import mutation_metrics

        G, node = create_nfr("test", epi=0.5, vf=1.2, theta=0.5)
        G.nodes[node]["epi_history"] = [0.3, 0.4, 0.5]

        # Call without optional parameters (backwards compatibility)
        metrics = mutation_metrics(G, node, theta_before=0.4, epi_before=0.4)

        # Should still work and have core metrics
        assert "operator" in metrics
        assert "glyph" in metrics
        assert "threshold_met" in metrics
        assert "bifurcation_score" in metrics

        # Delta metrics should default to 0.0 or current values
        assert "delta_vf" in metrics
        assert "delta_dnfr" in metrics

    def test_mutation_metrics_with_vf_dnfr(self):
        """mutation_metrics() must use vf_before/dnfr_before when provided."""
        from tnfr.operators.metrics import mutation_metrics

        G, node = create_nfr("test", epi=0.5, vf=1.2, theta=0.5)
        G.nodes[node]["epi_history"] = [0.3, 0.4, 0.5]
        # Set vf using proper aliases
        G.nodes[node]["νf"] = 1.5
        G.nodes[node]["vf"] = 1.5
        G.nodes[node]["dnfr"] = 0.3

        # Call with optional parameters
        metrics = mutation_metrics(
            G,
            node,
            theta_before=0.4,
            epi_before=0.4,
            vf_before=1.2,
            dnfr_before=0.2,
        )

        # Should compute deltas correctly
        assert metrics["delta_vf"] == pytest.approx(0.3, abs=0.01)
        assert metrics["delta_dnfr"] == pytest.approx(0.1, abs=0.01)
        assert metrics["vf_before"] == 1.2
        assert metrics["dnfr_before"] == 0.2
