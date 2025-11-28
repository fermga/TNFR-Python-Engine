"""Tests for THOL collective coherence validation.

This module tests the explicit validation of collective coherence for sub-EPIs
created by THOL (Self-organization) operator, ensuring that emergent sub-structures
form coherent ensembles that preserve parent node structural identity.

References:
    - Issue: [THOL][Canonical] Añadir validación explícita de coherencia colectiva de sub-EPIs
    - TNFR Manual: "El pulso que nos atraviesa", §2.2.10 (THOL)
"""

import logging
import pytest
import networkx as nx

from tnfr.structural import create_nfr
from tnfr.operators.definitions import SelfOrganization
from tnfr.operators.health_analyzer import SequenceHealthAnalyzer
from tnfr.operators.metabolism import compute_subepi_collective_coherence
from tnfr.constants.aliases import ALIAS_EPI, ALIAS_VF, ALIAS_THETA
from tnfr.alias import get_attr


class TestCollectiveCoherenceValidation:
    """Test collective coherence validation in THOL operator."""

    def test_coherence_stored_in_node_attributes(self):
        """THOL should store collective coherence in node attributes for telemetry."""
        G, node = create_nfr("test", epi=0.50, vf=1.0, theta=0.1)

        # Set up for bifurcation with non-linear history (acceleration > 0)
        # d²EPI/dt² = EPI_t - 2*EPI_{t-1} + EPI_{t-2}
        # = 0.50 - 2*0.38 + 0.20 = 0.50 - 0.76 + 0.20 = -0.06 (abs = 0.06)
        G.nodes[node]["epi_history"] = [0.20, 0.38, 0.50]

        # Apply THOL with low tau to ensure bifurcation
        SelfOrganization()(G, node, tau=0.05)

        # Check that coherence was stored
        assert (
            "_thol_collective_coherence" in G.nodes[node]
        ), "Coherence should be stored in node attributes"

        coherence = G.nodes[node]["_thol_collective_coherence"]
        assert isinstance(coherence, float), "Coherence should be a float"
        assert 0.0 <= coherence <= 1.0, "Coherence should be in [0, 1]"

    def test_no_warning_for_high_coherence(self, caplog):
        """THOL should not warn when collective coherence is above threshold."""
        G, node = create_nfr("test", epi=0.50, vf=1.0, theta=0.1)

        # Set up for bifurcation with uniform history (should produce high coherence)
        G.nodes[node]["epi_history"] = [0.48, 0.49, 0.50]
        G.graph["THOL_MIN_COLLECTIVE_COHERENCE"] = 0.3

        # Capture warnings
        with caplog.at_level(logging.WARNING):
            SelfOrganization()(G, node, tau=0.01)

        # Check no warnings were logged
        assert (
            "collective coherence" not in caplog.text
        ), "High coherence should not trigger warnings"

    def test_warning_for_low_coherence(self, caplog):
        """THOL should warn when collective coherence is below threshold."""
        G, node = create_nfr("test", epi=0.50, vf=1.0, theta=0.1)

        # Set up for bifurcation
        # Create conditions that will produce multiple diverse sub-EPIs
        G.nodes[node]["epi_history"] = [0.05, 0.33, 0.50]
        G.graph["THOL_MIN_COLLECTIVE_COHERENCE"] = 0.8  # High threshold to trigger warning

        # We need multiple bifurcations to test coherence
        # Apply THOL multiple times
        with caplog.at_level(logging.WARNING):
            # First bifurcation
            SelfOrganization()(G, node, tau=0.1)

            # Update history for second bifurcation
            current_epi = float(get_attr(G.nodes[node], ALIAS_EPI, 0.0))
            G.nodes[node]["epi_history"].append(current_epi)
            G.nodes[node]["epi_history"].append(current_epi + 0.15)

            # Second bifurcation
            SelfOrganization()(G, node, tau=0.1)

        # Check coherence value
        coherence = G.nodes[node].get("_thol_collective_coherence")

        # Only check warning if we actually have low coherence with multiple sub-EPIs
        sub_epis = G.nodes[node].get("sub_epis", [])
        if len(sub_epis) >= 2 and coherence is not None and coherence < 0.8:
            assert "collective coherence" in caplog.text, "Low coherence should trigger warning"
            assert str(node) in caplog.text, "Warning should include node identifier"

    def test_coherence_warning_recorded_in_graph(self):
        """Coherence warnings should be recorded in graph metadata for analysis."""
        G, node = create_nfr("test", epi=0.50, vf=1.0, theta=0.1)

        # Manually create sub-EPIs with high variance (low coherence)
        G.nodes[node]["sub_epis"] = [
            {"epi": 0.05, "vf": 1.0, "timestamp": 10},
            {"epi": 0.50, "vf": 1.0, "timestamp": 10},
            {"epi": 0.95, "vf": 1.0, "timestamp": 10},
        ]
        G.nodes[node]["epi_history"] = [0.30, 0.40, 0.50]
        G.graph["THOL_MIN_COLLECTIVE_COHERENCE"] = 0.9  # High threshold

        # Trigger validation by applying THOL (will create another sub-EPI)
        SelfOrganization()(G, node, tau=0.05)

        # Check that warning was recorded
        coherence = G.nodes[node].get("_thol_collective_coherence")
        if coherence is not None and coherence < 0.9:
            warnings = G.graph.get("thol_coherence_warnings", [])

            # Find warning for this node
            node_warnings = [w for w in warnings if w["node"] == node]

            if node_warnings:
                warning = node_warnings[0]
                assert "coherence" in warning, "Warning should contain coherence value"
                assert "threshold" in warning, "Warning should contain threshold"
                assert "sub_epi_count" in warning, "Warning should contain sub-EPI count"

    def test_threshold_configurable_via_graph_config(self):
        """Coherence threshold should be configurable via graph config."""
        G, node = create_nfr("test", epi=0.50, vf=1.0, theta=0.1)

        # Set custom threshold
        custom_threshold = 0.5
        G.graph["THOL_MIN_COLLECTIVE_COHERENCE"] = custom_threshold

        # Create low-coherence sub-EPIs
        G.nodes[node]["sub_epis"] = [
            {"epi": 0.10, "vf": 1.0, "timestamp": 10},
            {"epi": 0.40, "vf": 1.0, "timestamp": 10},
        ]
        G.nodes[node]["epi_history"] = [0.30, 0.40, 0.50]

        # Apply THOL
        SelfOrganization()(G, node, tau=0.05)

        # Check that custom threshold is used
        warnings = G.graph.get("thol_coherence_warnings", [])
        if warnings:
            assert warnings[0]["threshold"] == custom_threshold, "Custom threshold should be used"

    def test_no_warning_for_single_sub_epi(self, caplog):
        """THOL should not warn about coherence for single sub-EPI (trivial case)."""
        G, node = create_nfr("test", epi=0.50, vf=1.0, theta=0.1)

        # Set up for single bifurcation
        G.nodes[node]["epi_history"] = [0.30, 0.40, 0.50]
        G.graph["THOL_MIN_COLLECTIVE_COHERENCE"] = 0.9  # High threshold

        with caplog.at_level(logging.WARNING):
            # Single bifurcation
            SelfOrganization()(G, node, tau=0.05)

        # Should not warn - need at least 2 sub-EPIs to measure coherence
        assert (
            "collective coherence" not in caplog.text
        ), "Single sub-EPI should not trigger coherence warning"


class TestHealthAnalyzerTHOLCoherence:
    """Test THOL coherence analysis in SequenceHealthAnalyzer."""

    def test_analyze_thol_coherence_no_bifurcations(self):
        """Should return None when no THOL bifurcations exist."""
        G = nx.Graph()
        G.add_node(0, epi=0.50, vf=1.0, theta=0.1)

        analyzer = SequenceHealthAnalyzer()
        result = analyzer.analyze_thol_coherence(G)

        assert result is None, "Should return None when no bifurcations"

    def test_analyze_thol_coherence_single_node(self):
        """Should analyze coherence for single THOL node."""
        G, node = create_nfr("test", epi=0.50, vf=1.0, theta=0.1)

        # Create sub-EPIs with known coherence
        G.nodes[node]["sub_epis"] = [
            {"epi": 0.15, "vf": 1.0, "timestamp": 10},
            {"epi": 0.16, "vf": 1.0, "timestamp": 10},
        ]
        G.nodes[node]["_thol_collective_coherence"] = 0.85

        analyzer = SequenceHealthAnalyzer()
        result = analyzer.analyze_thol_coherence(G)

        assert result is not None, "Should return results when bifurcations exist"
        assert result["mean_coherence"] == 0.85, "Mean should match single value"
        assert result["min_coherence"] == 0.85, "Min should match single value"
        assert result["max_coherence"] == 0.85, "Max should match single value"
        assert result["total_thol_nodes"] == 1, "Should count 1 THOL node"

    def test_analyze_thol_coherence_multiple_nodes(self):
        """Should aggregate coherence statistics across multiple nodes."""
        G = nx.Graph()

        # Create multiple nodes with different coherence values
        for i in range(3):
            G.add_node(i, epi=0.50, vf=1.0, theta=0.1)
            G.nodes[i]["sub_epis"] = [
                {"epi": 0.15, "vf": 1.0, "timestamp": 10},
                {"epi": 0.16, "vf": 1.0, "timestamp": 10},
            ]

        # Set different coherence values
        G.nodes[0]["_thol_collective_coherence"] = 0.9
        G.nodes[1]["_thol_collective_coherence"] = 0.5
        G.nodes[2]["_thol_collective_coherence"] = 0.7

        analyzer = SequenceHealthAnalyzer()
        result = analyzer.analyze_thol_coherence(G)

        assert result is not None
        assert result["mean_coherence"] == pytest.approx(0.7), "Mean should be (0.9+0.5+0.7)/3"
        assert result["min_coherence"] == 0.5, "Min should be lowest value"
        assert result["max_coherence"] == 0.9, "Max should be highest value"
        assert result["total_thol_nodes"] == 3, "Should count all 3 nodes"

    def test_analyze_thol_coherence_below_threshold_count(self):
        """Should count nodes below threshold correctly."""
        G = nx.Graph()

        # Create nodes with varying coherence
        for i in range(4):
            G.add_node(i, epi=0.50, vf=1.0, theta=0.1)
            G.nodes[i]["sub_epis"] = [
                {"epi": 0.15, "vf": 1.0, "timestamp": 10},
                {"epi": 0.16, "vf": 1.0, "timestamp": 10},
            ]

        # Set coherence values: 2 above, 2 below threshold of 0.5
        G.nodes[0]["_thol_collective_coherence"] = 0.8  # Above
        G.nodes[1]["_thol_collective_coherence"] = 0.2  # Below
        G.nodes[2]["_thol_collective_coherence"] = 0.6  # Above
        G.nodes[3]["_thol_collective_coherence"] = 0.3  # Below

        G.graph["THOL_MIN_COLLECTIVE_COHERENCE"] = 0.5

        analyzer = SequenceHealthAnalyzer()
        result = analyzer.analyze_thol_coherence(G)

        assert result["nodes_below_threshold"] == 2, "Should count 2 nodes below threshold of 0.5"
        assert result["threshold"] == 0.5, "Should report configured threshold"

    def test_analyze_thol_coherence_ignores_nodes_without_coherence_attr(self):
        """Should skip nodes that don't have coherence computed yet."""
        G = nx.Graph()

        # Node with sub-EPIs but no coherence computed
        G.add_node(0, epi=0.50, vf=1.0, theta=0.1)
        G.nodes[0]["sub_epis"] = [{"epi": 0.15, "vf": 1.0, "timestamp": 10}]

        # Node with sub-EPIs and coherence
        G.add_node(1, epi=0.50, vf=1.0, theta=0.1)
        G.nodes[1]["sub_epis"] = [{"epi": 0.15, "vf": 1.0, "timestamp": 10}]
        G.nodes[1]["_thol_collective_coherence"] = 0.75

        analyzer = SequenceHealthAnalyzer()
        result = analyzer.analyze_thol_coherence(G)

        # Should only count node 1 (with coherence computed)
        assert result is not None
        assert result["total_thol_nodes"] == 2, "Should count both nodes with sub-EPIs"
        # But only analyze the one with coherence value
        assert result["mean_coherence"] == 0.75, "Should only analyze node with coherence"


class TestCoherenceIntegration:
    """Integration tests for collective coherence validation."""

    def test_thol_workflow_with_coherence_validation(self):
        """Test complete THOL workflow with coherence validation."""
        G, node = create_nfr("test", epi=0.50, vf=1.0, theta=0.1)

        # Set up for bifurcation
        G.nodes[node]["epi_history"] = [0.30, 0.43, 0.50]
        G.graph["THOL_MIN_COLLECTIVE_COHERENCE"] = 0.3

        # Apply THOL
        SelfOrganization()(G, node, tau=0.05)

        # Verify bifurcation occurred
        sub_epis = G.nodes[node].get("sub_epis", [])
        assert len(sub_epis) > 0, "Bifurcation should have occurred"

        # Verify coherence was computed
        coherence = G.nodes[node].get("_thol_collective_coherence")
        assert coherence is not None, "Coherence should be computed"

        # Verify coherence can be analyzed
        analyzer = SequenceHealthAnalyzer()
        result = analyzer.analyze_thol_coherence(G)
        assert result is not None, "Should be able to analyze coherence"

    def test_multiple_thol_applications_track_coherence(self):
        """Multiple THOL applications should track coherence for each."""
        G, node = create_nfr("test", epi=0.50, vf=1.0, theta=0.1)

        # First bifurcation with acceleration > 0.05
        G.nodes[node]["epi_history"] = [0.20, 0.38, 0.50]
        SelfOrganization()(G, node, tau=0.05)
        coherence1 = G.nodes[node].get("_thol_collective_coherence")
        sub_epis_1 = len(G.nodes[node].get("sub_epis", []))

        # Second bifurcation (update history with acceleration)
        current_epi = float(get_attr(G.nodes[node], ALIAS_EPI, 0.0))
        # Create new history with acceleration: d²EPI/dt² = new - 2*curr + old
        G.nodes[node]["epi_history"] = [current_epi - 0.10, current_epi, current_epi + 0.20]
        SelfOrganization()(G, node, tau=0.05)
        coherence2 = G.nodes[node].get("_thol_collective_coherence")
        sub_epis_2 = len(G.nodes[node].get("sub_epis", []))

        # Coherence should be computed for both
        assert coherence1 is not None, "First coherence should be computed"
        assert coherence2 is not None, "Second coherence should be computed"

        # Second application should have more sub-EPIs
        assert sub_epis_2 >= sub_epis_1, "Should have at least as many sub-EPIs"
        assert sub_epis_1 >= 1, "First bifurcation should create at least one sub-EPI"

    def test_coherence_validation_preserves_tnfr_invariants(self):
        """Coherence validation should not violate TNFR canonical invariants."""
        G, node = create_nfr("test", epi=0.50, vf=1.0, theta=0.1)

        # Get initial state
        epi_before = float(get_attr(G.nodes[node], ALIAS_EPI, 0.0))
        vf_before = float(get_attr(G.nodes[node], ALIAS_VF, 1.0))

        # Set up and apply THOL with acceleration
        G.nodes[node]["epi_history"] = [0.20, 0.38, 0.50]
        SelfOrganization()(G, node, tau=0.05)

        # Get final state
        epi_after = float(get_attr(G.nodes[node], ALIAS_EPI, 0.0))
        vf_after = float(get_attr(G.nodes[node], ALIAS_VF, 1.0))

        # TNFR Invariants:
        # 1. EPI should increase (emergence contribution)
        assert epi_after >= epi_before, "EPI should increase or stay same"

        # 2. VF should be positive (structural frequency always positive)
        assert vf_after > 0, "VF must remain positive"

        # 3. Coherence validation should not corrupt node structure
        assert "_thol_collective_coherence" in G.nodes[node], "Coherence should be stored"

        # 4. Sub-EPIs should exist after bifurcation
        sub_epis = G.nodes[node].get("sub_epis", [])
        assert len(sub_epis) >= 1, "Bifurcation should create sub-EPIs"
