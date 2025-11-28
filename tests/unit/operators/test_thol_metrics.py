"""Tests for THOL (Self-organization) enhanced metrics.

This module tests the enriched telemetry for THOL operator including:
- Cascade depth computation
- Sub-EPI collective coherence
- Metabolic activity index
- Network emergence indicators

References:
    - Issue: #[THOL MÉTRICAS] Enriquecer telemetría con profundidad de cascada
    - TNFR Manual: "El pulso que nos atraviesa", §2.2.10 (THOL)
"""

import pytest
import networkx as nx

from tnfr.structural import create_nfr
from tnfr.operators.definitions import SelfOrganization
from tnfr.operators.metabolism import (
    compute_cascade_depth,
    compute_propagation_radius,
    compute_subepi_collective_coherence,
    compute_metabolic_activity_index,
)
from tnfr.constants.aliases import ALIAS_EPI, ALIAS_VF, ALIAS_THETA
from tnfr.alias import get_attr


class TestCascadeDepth:
    """Test cascade depth computation."""

    def test_no_bifurcation_zero_depth(self):
        """Node with no sub-EPIs should report depth = 0."""
        G, node = create_nfr("test", epi=0.50, vf=1.0)

        depth = compute_cascade_depth(G, node)

        assert depth == 0, "No bifurcation = depth 0"

    def test_single_level_bifurcation(self):
        """Single bifurcation should report depth = 1."""
        G, node = create_nfr("test", epi=0.50, vf=1.0)

        # Simulate bifurcation by adding sub-EPIs
        G.nodes[node]["sub_epis"] = [
            {"epi": 0.15, "vf": 1.1, "timestamp": 10},
            {"epi": 0.16, "vf": 1.1, "timestamp": 10},
        ]

        depth = compute_cascade_depth(G, node)

        assert depth == 1, "Single bifurcation = depth 1"

    def test_multi_level_cascade(self):
        """Nested bifurcations should report increasing depth."""
        G, node = create_nfr("test", epi=0.60, vf=1.0)

        # Simulate multi-level cascade
        G.nodes[node]["sub_epis"] = [
            {"epi": 0.18, "vf": 1.2, "timestamp": 10, "cascade_depth": 1},
            {"epi": 0.17, "vf": 1.1, "timestamp": 10, "cascade_depth": 0},
        ]

        depth = compute_cascade_depth(G, node)

        assert depth == 2, "Nested cascade should have depth 2"


class TestPropagationRadius:
    """Test propagation radius computation."""

    def test_no_propagations_zero_radius(self):
        """Network with no propagations should report radius = 0."""
        G = nx.Graph()
        G.add_node(0, epi=0.50, vf=1.0, theta=0.1)

        radius = compute_propagation_radius(G)

        assert radius == 0, "No propagations = radius 0"

    def test_single_propagation_event(self):
        """Single propagation should count source + targets."""
        G = nx.Graph()
        G.add_node(0, epi=0.50, vf=1.0, theta=0.1)
        G.add_node(1, epi=0.40, vf=1.0, theta=0.12)
        G.add_edge(0, 1)

        # Simulate propagation
        G.graph["thol_propagations"] = [
            {
                "source_node": 0,
                "propagations": [(1, 0.10)],
                "timestamp": 10,
            }
        ]

        radius = compute_propagation_radius(G)

        assert radius == 2, "Should count source + 1 target = 2 nodes"

    def test_multiple_propagation_events(self):
        """Multiple propagations should count unique nodes only."""
        G = nx.Graph()
        for i in range(5):
            G.add_node(i, epi=0.50, vf=1.0, theta=0.1)
            if i > 0:
                G.add_edge(0, i)

        # Simulate cascading propagations
        G.graph["thol_propagations"] = [
            {
                "source_node": 0,
                "propagations": [(1, 0.10), (2, 0.09)],
                "timestamp": 10,
            },
            {
                "source_node": 1,
                "propagations": [(3, 0.08)],
                "timestamp": 11,
            },
            {
                "source_node": 2,
                "propagations": [(4, 0.07)],
                "timestamp": 11,
            },
        ]

        radius = compute_propagation_radius(G)

        assert radius == 5, "Should count all 5 unique nodes"


class TestSubEPICollectiveCoherence:
    """Test sub-EPI collective coherence computation."""

    def test_no_sub_epis_zero_coherence(self):
        """Node with no sub-EPIs should report coherence = 0."""
        G, node = create_nfr("test", epi=0.50, vf=1.0)

        coherence = compute_subepi_collective_coherence(G, node)

        assert coherence == 0.0, "No sub-EPIs = coherence 0"

    def test_single_sub_epi_zero_coherence(self):
        """Single sub-EPI cannot measure coherence."""
        G, node = create_nfr("test", epi=0.50, vf=1.0)
        G.nodes[node]["sub_epis"] = [
            {"epi": 0.15, "vf": 1.1, "timestamp": 10},
        ]

        coherence = compute_subepi_collective_coherence(G, node)

        assert coherence == 0.0, "Single sub-EPI = coherence 0"

    def test_uniform_sub_epis_high_coherence(self):
        """Uniform sub-EPIs should have high coherence (low variance)."""
        G, node = create_nfr("test", epi=0.60, vf=1.0)

        # Create uniform sub-EPIs (very similar magnitudes)
        G.nodes[node]["sub_epis"] = [
            {"epi": 0.150, "vf": 1.1, "timestamp": 10},
            {"epi": 0.151, "vf": 1.1, "timestamp": 10},
            {"epi": 0.149, "vf": 1.1, "timestamp": 10},
        ]

        coherence = compute_subepi_collective_coherence(G, node)

        # High coherence: 1/(1 + low_variance) ≈ 1
        assert coherence > 0.9, f"Expected high coherence, got {coherence:.3f}"

    def test_varied_sub_epis_low_coherence(self):
        """Highly varied sub-EPIs should have lower coherence than uniform ones."""
        G, node = create_nfr("test", epi=0.60, vf=1.0)

        # Create highly varied sub-EPIs (very different magnitudes)
        G.nodes[node]["sub_epis"] = [
            {"epi": 0.05, "vf": 1.1, "timestamp": 10},
            {"epi": 0.50, "vf": 1.1, "timestamp": 10},
            {"epi": 0.95, "vf": 1.1, "timestamp": 10},
        ]

        coherence_varied = compute_subepi_collective_coherence(G, node)

        # Now compare to uniform case
        G.nodes[node]["sub_epis"] = [
            {"epi": 0.150, "vf": 1.1, "timestamp": 10},
            {"epi": 0.151, "vf": 1.1, "timestamp": 10},
            {"epi": 0.149, "vf": 1.1, "timestamp": 10},
        ]

        coherence_uniform = compute_subepi_collective_coherence(G, node)

        # Varied should have lower coherence than uniform
        assert coherence_varied < coherence_uniform, (
            f"Expected varied coherence ({coherence_varied:.3f}) < "
            f"uniform coherence ({coherence_uniform:.3f})"
        )


class TestMetabolicActivityIndex:
    """Test metabolic activity index computation."""

    def test_no_sub_epis_zero_activity(self):
        """Node with no sub-EPIs should report activity = 0."""
        G, node = create_nfr("test", epi=0.50, vf=1.0)

        activity = compute_metabolic_activity_index(G, node)

        assert activity == 0.0, "No sub-EPIs = activity 0"

    def test_isolated_node_zero_activity(self):
        """Isolated node should have zero metabolic activity."""
        G, node = create_nfr("isolated", epi=0.50, vf=1.0)

        # Add sub-EPIs but without metabolized flag
        G.nodes[node]["sub_epis"] = [
            {"epi": 0.15, "vf": 1.1, "timestamp": 10, "metabolized": False},
            {"epi": 0.16, "vf": 1.1, "timestamp": 10, "metabolized": False},
        ]

        activity = compute_metabolic_activity_index(G, node)

        assert activity == 0.0, "Isolated node should have zero metabolic activity"

    def test_full_network_metabolism(self):
        """Node with all metabolized sub-EPIs should report activity = 1.0."""
        G = nx.Graph()
        G.add_node(0, epi=0.50, vf=1.0, theta=0.1)
        G.add_node(1, epi=0.70, vf=1.0, theta=0.15)
        G.add_edge(0, 1)

        # All sub-EPIs metabolized network context
        G.nodes[0]["sub_epis"] = [
            {"epi": 0.15, "vf": 1.1, "timestamp": 10, "metabolized": True},
            {"epi": 0.16, "vf": 1.1, "timestamp": 10, "metabolized": True},
            {"epi": 0.14, "vf": 1.1, "timestamp": 10, "metabolized": True},
        ]

        activity = compute_metabolic_activity_index(G, 0)

        assert activity == 1.0, "All metabolized = activity 1.0"

    def test_partial_network_metabolism(self):
        """Node with mixed metabolism should report fractional activity."""
        G = nx.Graph()
        G.add_node(0, epi=0.50, vf=1.0, theta=0.1)
        G.add_node(1, epi=0.70, vf=1.0, theta=0.15)
        G.add_edge(0, 1)

        # 2 metabolized, 2 internal
        G.nodes[0]["sub_epis"] = [
            {"epi": 0.15, "vf": 1.1, "timestamp": 10, "metabolized": True},
            {"epi": 0.16, "vf": 1.1, "timestamp": 10, "metabolized": True},
            {"epi": 0.14, "vf": 1.1, "timestamp": 10, "metabolized": False},
            {"epi": 0.13, "vf": 1.1, "timestamp": 10, "metabolized": False},
        ]

        activity = compute_metabolic_activity_index(G, 0)

        assert activity == 0.5, "50% metabolized = activity 0.5"


class TestIntegratedTHOLMetrics:
    """Test integrated THOL metrics collection."""

    def test_metrics_with_bifurcation(self):
        """THOL metrics should include all new indicators when bifurcation occurs."""
        G, node = create_nfr("test", epi=0.60, vf=1.2)

        # Simulate bifurcation with sub-EPIs
        G.nodes[node]["sub_epis"] = [
            {"epi": 0.15, "vf": 1.2, "timestamp": 10, "metabolized": True},
            {"epi": 0.14, "vf": 1.2, "timestamp": 10, "metabolized": True},
        ]

        # Enable metrics collection
        G.graph["COLLECT_OPERATOR_METRICS"] = True

        # Apply THOL
        from tnfr.operators.metrics import self_organization_metrics

        metrics = self_organization_metrics(G, node, epi_before=0.50, vf_before=1.0)

        # Verify new metrics are present
        assert "cascade_depth" in metrics
        assert "propagation_radius" in metrics
        assert "subepi_coherence" in metrics
        assert "metabolic_activity_index" in metrics
        assert "network_emergence" in metrics

        # Verify values
        assert metrics["bifurcation_occurred"] is True
        assert metrics["cascade_depth"] >= 0
        assert metrics["subepi_coherence"] >= 0.0

    def test_network_emergence_indicator(self):
        """Network emergence should be True when cascade detected with high coherence."""
        G = nx.Graph()

        # Create network with multiple nodes
        for i in range(5):
            G.add_node(i, epi=0.50, vf=1.0, theta=0.1)
            if i > 0:
                G.add_edge(0, i)

        # Simulate cascade
        G.graph["thol_propagations"] = [
            {
                "source_node": 0,
                "propagations": [(1, 0.10), (2, 0.09), (3, 0.08)],
                "timestamp": 10,
            }
        ]

        # Add coherent sub-EPIs
        G.nodes[0]["sub_epis"] = [
            {"epi": 0.150, "vf": 1.2, "timestamp": 10, "metabolized": True},
            {"epi": 0.151, "vf": 1.2, "timestamp": 10, "metabolized": True},
            {"epi": 0.149, "vf": 1.2, "timestamp": 10, "metabolized": True},
        ]

        from tnfr.operators.metrics import self_organization_metrics

        metrics = self_organization_metrics(G, 0, epi_before=0.50, vf_before=1.0)

        # Should detect network emergence
        assert metrics["cascade_detected"] is True
        assert metrics["subepi_coherence"] > 0.5
        assert metrics["network_emergence"] is True


class TestMetricsBackwardCompatibility:
    """Test that new metrics don't break existing functionality."""

    def test_existing_metrics_still_present(self):
        """Original metrics should still be included."""
        G, node = create_nfr("test", epi=0.50, vf=1.0)

        from tnfr.operators.metrics import self_organization_metrics

        metrics = self_organization_metrics(G, node, epi_before=0.40, vf_before=0.9)

        # Original metrics
        assert "operator" in metrics
        assert "glyph" in metrics
        assert "delta_epi" in metrics
        assert "delta_vf" in metrics
        assert "epi_final" in metrics
        assert "vf_final" in metrics
        assert "d2epi" in metrics
        assert "dnfr_final" in metrics
        assert "nested_epi_count" in metrics

        assert metrics["operator"] == "Self-organization"
        assert metrics["glyph"] == "THOL"

    def test_metrics_with_empty_network(self):
        """Metrics should handle empty network gracefully."""
        G, node = create_nfr("test", epi=0.50, vf=1.0)

        from tnfr.operators.metrics import self_organization_metrics

        metrics = self_organization_metrics(G, node, epi_before=0.40, vf_before=0.9)

        # Should not crash
        assert metrics["cascade_depth"] == 0
        assert metrics["propagation_radius"] == 0
        assert metrics["subepi_coherence"] == 0.0
        assert metrics["metabolic_activity_index"] == 0.0
        assert metrics["network_emergence"] is False
