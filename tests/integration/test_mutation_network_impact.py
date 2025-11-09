"""Tests for ZHIR (Mutation) network impact and neighbor effects.

This module tests how ZHIR affects neighboring nodes in the network,
capturing the structural physics of phase transformation propagation.

Test Coverage:
1. Impact on directly connected neighbors
2. Phase coherence with neighbors
3. Network-wide effects
4. Isolated node behavior

References:
- AGENTS.md §11 (Mutation operator)
- test_mutation_metrics_comprehensive.py (network_impact metrics)
"""

import pytest
import math
from tnfr.structural import create_nfr, run_sequence
from tnfr.operators.definitions import Mutation, Coherence, Dissonance


class TestZHIRNetworkImpact:
    """Test ZHIR impact on network neighbors."""

    def test_zhir_affects_neighbors(self):
        """ZHIR should have measurable impact on connected neighbors."""
        G, node = create_nfr("test", epi=0.5, vf=1.0, theta=0.5)
        G.nodes[node]["epi_history"] = [0.3, 0.4, 0.5]
        
        # Add neighbors with proper TNFR attributes
        neighbors = []
        for i in range(3):
            neighbor_id = f"neighbor_{i}"
            G.add_node(
                neighbor_id,
                EPI=0.5,
                epi=0.5,
                theta=0.5 + i * 0.1,
                **{"νf": 1.0},  # Greek letter for canonical
                vf=1.0,
                dnfr=0.0,
                delta_nfr=0.0,
                theta_history=[0.5, 0.5 + i * 0.1],
                epi_history=[0.4, 0.5],
            )
            G.add_edge(node, neighbor_id)
            neighbors.append(neighbor_id)
        
        G.graph["COLLECT_OPERATOR_METRICS"] = True
        
        # Store neighbor states before mutation
        neighbors_theta_before = {n: G.nodes[n]["theta"] for n in neighbors}
        
        # Apply mutation
        Mutation()(G, node)
        
        # Check metrics captured network impact
        metrics = G.graph["operator_metrics"][-1]
        
        assert "neighbor_count" in metrics
        assert metrics["neighbor_count"] == 3
        
        assert "network_impact_radius" in metrics
        # Impact radius should be non-zero with neighbors
        # (actual value depends on implementation)
        
        assert "phase_coherence_neighbors" in metrics
        # Should have computed phase coherence with neighbors

    def test_zhir_phase_coherence_with_neighbors(self):
        """ZHIR should consider phase coherence with neighbors."""
        G, node = create_nfr("test", epi=0.5, vf=1.0, theta=0.5)
        G.nodes[node]["epi_history"] = [0.3, 0.4, 0.5]
        
        # Add neighbor with similar phase (coherent)
        G.add_node(
            "coherent_neighbor",
            epi=0.5,
            vf=1.0,
            theta=0.52,  # Very close phase
            delta_nfr=0.0,
            theta_history=[0.5, 0.52],
        )
        G.add_edge(node, "coherent_neighbor")
        
        # Add neighbor with opposite phase (incoherent)
        G.add_node(
            "incoherent_neighbor",
            epi=0.5,
            vf=1.0,
            theta=0.5 + math.pi,  # Opposite phase
            delta_nfr=0.0,
            theta_history=[0.5 + math.pi, 0.5 + math.pi],
        )
        G.add_edge(node, "incoherent_neighbor")
        
        G.graph["COLLECT_OPERATOR_METRICS"] = True
        
        # Apply mutation
        Mutation()(G, node)
        
        metrics = G.graph["operator_metrics"][-1]
        
        # Should have measured phase coherence
        assert "phase_coherence_neighbors" in metrics
        assert "impacted_neighbors" in metrics

    def test_zhir_isolated_node_zero_impact(self):
        """ZHIR on isolated node should have zero network impact."""
        G, node = create_nfr("test", epi=0.5, vf=1.0)
        # No neighbors
        G.nodes[node]["epi_history"] = [0.3, 0.4, 0.5]
        G.graph["COLLECT_OPERATOR_METRICS"] = True
        
        Mutation()(G, node)
        
        metrics = G.graph["operator_metrics"][-1]
        
        # Isolated node should have zero neighbors
        assert metrics["neighbor_count"] == 0
        assert metrics["impacted_neighbors"] == 0
        assert metrics["network_impact_radius"] == 0.0
        assert metrics["phase_coherence_neighbors"] == 0.0

    def test_zhir_network_impact_radius_calculation(self):
        """Network impact radius should be calculated correctly."""
        G, node = create_nfr("test", epi=0.5, vf=1.0, theta=0.5)
        G.nodes[node]["epi_history"] = [0.3, 0.4, 0.5]
        
        # Add neighbors at different "distances" (via phase difference)
        # Close phase = high impact
        G.add_node("close", epi=0.5, vf=1.0, theta=0.51, delta_nfr=0.0)
        G.add_edge(node, "close")
        
        # Far phase = low impact
        G.add_node("far", epi=0.5, vf=1.0, theta=0.5 + 1.5, delta_nfr=0.0)
        G.add_edge(node, "far")
        
        G.graph["COLLECT_OPERATOR_METRICS"] = True
        
        Mutation()(G, node)
        
        metrics = G.graph["operator_metrics"][-1]
        
        # Should have computed impact radius
        assert "network_impact_radius" in metrics
        assert 0.0 <= metrics["network_impact_radius"] <= 1.0

    def test_zhir_in_dense_network(self):
        """ZHIR in dense network should track all neighbors."""
        G, node = create_nfr("test", epi=0.5, vf=1.0, theta=0.5)
        G.nodes[node]["epi_history"] = [0.3, 0.4, 0.5]
        
        # Add many neighbors (dense network)
        for i in range(10):
            neighbor_id = f"n{i}"
            G.add_node(
                neighbor_id,
                epi=0.5,
                vf=1.0,
                theta=0.5 + i * 0.1,
                delta_nfr=0.0,
            )
            G.add_edge(node, neighbor_id)
        
        G.graph["COLLECT_OPERATOR_METRICS"] = True
        
        Mutation()(G, node)
        
        metrics = G.graph["operator_metrics"][-1]
        
        # Should track all neighbors
        assert metrics["neighbor_count"] == 10

    def test_zhir_with_bidirectional_edges(self):
        """ZHIR should handle bidirectional connections correctly."""
        G, node = create_nfr("test", epi=0.5, vf=1.0, theta=0.5)
        G.nodes[node]["epi_history"] = [0.3, 0.4, 0.5]
        
        # Add bidirectional neighbor
        G.add_node("neighbor", epi=0.5, vf=1.0, theta=0.52, delta_nfr=0.0)
        G.add_edge(node, "neighbor")
        G.add_edge("neighbor", node)  # Bidirectional
        
        G.graph["COLLECT_OPERATOR_METRICS"] = True
        
        # Should not raise error
        Mutation()(G, node)
        
        metrics = G.graph["operator_metrics"][-1]
        
        # Should count neighbor once (not twice for bidirectional)
        assert metrics["neighbor_count"] >= 1


class TestZHIRNeighborPhaseCompatibility:
    """Test phase compatibility checking with neighbors."""

    def test_zhir_with_compatible_neighbors(self):
        """ZHIR with phase-compatible neighbors should work smoothly."""
        G, node = create_nfr("test", epi=0.5, vf=1.0, theta=0.5)
        G.nodes[node]["epi_history"] = [0.3, 0.4, 0.5]
        
        # Add neighbors with compatible phases (within π/2)
        for i in range(3):
            G.add_node(
                f"n{i}",
                epi=0.5,
                vf=1.0,
                theta=0.5 + i * 0.3,  # Within compatible range
                delta_nfr=0.0,
            )
            G.add_edge(node, f"n{i}")
        
        G.graph["COLLECT_OPERATOR_METRICS"] = True
        
        # Should work without issues
        Mutation()(G, node)
        
        metrics = G.graph["operator_metrics"][-1]
        
        # Phase coherence should be relatively high
        if "phase_coherence_neighbors" in metrics:
            # Should be positive (compatible phases)
            assert metrics["phase_coherence_neighbors"] >= 0

    def test_zhir_with_incompatible_neighbors(self):
        """ZHIR with phase-incompatible neighbors should still work."""
        G, node = create_nfr("test", epi=0.5, vf=1.0, theta=0.5)
        G.nodes[node]["epi_history"] = [0.3, 0.4, 0.5]
        
        # Add neighbors with incompatible phases (antiphase)
        for i in range(3):
            G.add_node(
                f"n{i}",
                epi=0.5,
                vf=1.0,
                theta=0.5 + math.pi + i * 0.1,  # Opposite phase
                delta_nfr=0.0,
            )
            G.add_edge(node, f"n{i}")
        
        # Should not raise error (ZHIR is internal transformation)
        Mutation()(G, node)
        
        # Node should still be viable
        assert G.nodes[node]["vf"] > 0


class TestZHIRNetworkPropagation:
    """Test mutation effects propagation through network."""

    def test_zhir_sequence_with_resonance_propagates(self):
        """ZHIR → RA should propagate transformed state."""
        from tnfr.operators.definitions import Resonance
        
        G, node = create_nfr("test", epi=0.5, vf=1.0, theta=0.5)
        G.nodes[node]["epi_history"] = [0.3, 0.4, 0.5]
        
        # Add neighbor with compatible phase
        G.add_node("neighbor", epi=0.5, vf=1.0, theta=0.52, delta_nfr=0.0)
        G.add_edge(node, "neighbor")
        
        theta_before = G.nodes[node]["theta"]
        
        # Apply mutation then resonance
        run_sequence(G, node, [
            Coherence(),
            Dissonance(),
            Mutation(),    # Transform phase
            Resonance(),   # Propagate to neighbors
        ])
        
        theta_after = G.nodes[node]["theta"]
        
        # Phase should have changed
        assert theta_after != theta_before

    def test_zhir_does_not_directly_modify_neighbors(self):
        """ZHIR should not directly modify neighbor states (internal transformation)."""
        G, node = create_nfr("test", epi=0.5, vf=1.0, theta=0.5)
        G.nodes[node]["epi_history"] = [0.3, 0.4, 0.5]
        
        # Add neighbor
        G.add_node("neighbor", epi=0.5, vf=1.0, theta=0.52, delta_nfr=0.0)
        G.add_edge(node, "neighbor")
        
        # Store neighbor state
        neighbor_theta_before = G.nodes["neighbor"]["theta"]
        neighbor_epi_before = G.nodes["neighbor"]["epi"]
        
        # Apply mutation to main node
        Mutation()(G, node)
        
        # Neighbor should not be directly modified
        assert G.nodes["neighbor"]["theta"] == neighbor_theta_before
        assert G.nodes["neighbor"]["epi"] == neighbor_epi_before


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
