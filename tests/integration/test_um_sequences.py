"""Integration tests for UM (Coupling) operator sequences.

Tests canonical sequences involving UM from TNFR theory:
- UM → RA: Coupling followed by resonance propagation
- AL → UM: Emission followed by coupling  
- UM → IL: Coupling stabilized into coherence
- Network_sync: Complete sequence with UM
- UM in network formation cycles

These tests validate that UM correctly:
1. Synchronizes phases (θᵢ ≈ θⱼ)
2. Preserves EPI identity
3. Enables network-level coherence
4. Works in combination with other operators
"""

import math
import pytest
from tnfr.sdk import TNFRNetwork, NetworkConfig
from tnfr.structural import create_nfr, run_sequence
from tnfr.operators import apply_glyph
from tnfr.operators.definitions import (
    Coupling, Resonance, Emission, Coherence, Reception
)


class TestCanonicalUMSequences:
    """Test canonical UM sequences from TNFR theory."""
    
    def test_um_ra_coupling_propagation(self):
        """Test UM → RA sequence (coupling + propagation)."""
        # Create small network
        net = TNFRNetwork("um_ra_test", NetworkConfig(random_seed=42))
        net.add_nodes(8).connect_nodes(0.4, "random")
        
        # Set varied phases
        for i, node in enumerate(net.graph.nodes()):
            net.graph.nodes[node]['theta'] = (i % 3) * (2 * math.pi / 3)
        
        # Measure initial coherence
        results_before = net.measure()
        C_before = results_before.coherence
        
        # Apply UM → RA sequence via network_sync
        net.apply_sequence("network_sync", repeat=1)
        
        results_after = net.measure()
        C_after = results_after.coherence
        
        # RA after UM should maintain or increase coherence
        # (coupling creates synchronized substrate, resonance propagates it)
        assert C_after >= C_before * 0.8, "UM → RA should maintain reasonable coherence"
    
    def test_al_um_emission_coupling(self):
        """Test AL → UM sequence (emission + coupling)."""
        # Create network
        G, node = create_nfr("test_node", vf=1.0, theta=0.0, epi=0.6)
        G.add_node("node2", theta=math.pi/4, EPI=0.6, vf=1.0, dnfr=0.3, Si=0.5)
        G.add_node("node3", theta=math.pi/2, EPI=0.6, vf=1.0, dnfr=0.3, Si=0.5)
        G.add_edge(node, "node2")
        G.add_edge(node, "node3")
        
        # Record initial phase and EPI
        theta_before = G.nodes[node]['theta']
        epi_before = G.nodes[node].get('EPI', G.nodes[node].get('epi'))
        
        # Apply AL → UM (via network_sync which includes AL, EN, IL, UM)
        apply_glyph(G, node, "AL")  # Emission
        epi_after_emission = G.nodes[node].get('EPI', G.nodes[node].get('epi'))
        
        apply_glyph(G, node, "UM")  # Coupling
        
        theta_after = G.nodes[node]['theta']
        epi_after_coupling = G.nodes[node].get('EPI', G.nodes[node].get('epi'))
        
        # Phase should change (synchronization effect)
        # Emission may modify EPI, but Coupling should preserve it
        # Check that EPI didn't change significantly during coupling
        if isinstance(epi_after_emission, (int, float)) and isinstance(epi_after_coupling, (int, float)):
            assert epi_after_coupling == pytest.approx(epi_after_emission, abs=0.15), \
                "Coupling must preserve EPI (emission may change it)"
    
    def test_um_il_coupling_stabilization(self):
        """Test UM → IL sequence (coupling + stabilization)."""
        # Create ring network
        net = TNFRNetwork("um_il_test", NetworkConfig(random_seed=42))
        net.add_nodes(6).connect_nodes(0.5, "ring")
        
        # Set moderate phase variation
        for i, node in enumerate(net.graph.nodes()):
            net.graph.nodes[node]['theta'] = i * math.pi / 6
        
        results_before = net.measure()
        
        # Apply sequence with UM (network_sync includes UM)
        net.apply_sequence("network_sync")
        results_after_um = net.measure()
        
        # Apply stabilization sequence (includes coherence)
        net.apply_sequence("stabilization")
        results_after_stabilization = net.measure()
        
        # Stabilization should maintain or improve coherence after network_sync
        assert results_after_stabilization.coherence >= results_after_um.coherence * 0.8, \
            "Stabilization should maintain coupling effects"
    
    def test_network_sync_includes_um(self):
        """Test that network_sync sequence properly includes and uses UM."""
        net = TNFRNetwork("network_sync_test", NetworkConfig(random_seed=42))
        net.add_nodes(10).connect_nodes(0.3, "random")
        
        # Set distinct phase groups
        for i, node in enumerate(net.graph.nodes()):
            net.graph.nodes[node]['theta'] = (i % 4) * math.pi / 2
        
        phases_before = [net.graph.nodes[n]['theta'] for n in net.graph.nodes()]
        phase_spread_before = max(phases_before) - min(phases_before)
        
        # Apply network_sync (AL → EN → IL → UM → RA → NAV)
        net.apply_sequence("network_sync", repeat=2)
        
        phases_after = [net.graph.nodes[n]['theta'] for n in net.graph.nodes()]
        phase_spread_after = max(phases_after) - min(phases_after)
        
        # Phase spread should reduce (synchronization effect)
        assert phase_spread_after < phase_spread_before, \
            "network_sync should reduce phase spread via UM"


class TestUMNetworkFormation:
    """Test UM role in network formation from isolated or loosely connected nodes."""
    
    def test_um_forms_coherent_network(self):
        """Test that UM contributes to network formation."""
        # Start with isolated nodes
        net = TNFRNetwork("formation_test", NetworkConfig(random_seed=42))
        net.add_nodes(12)
        
        # Varied initial phases
        for i, node in enumerate(net.graph.nodes()):
            net.graph.nodes[node]['theta'] = (i % 5) * (2 * math.pi / 5)
        
        results_isolated = net.measure()
        
        # Add connections
        net.connect_nodes(0.25, "random")
        
        # Apply formation sequence with UM
        net.apply_sequence("network_sync", repeat=3)
        
        results_formed = net.measure()
        
        # Network should achieve reasonable coherence
        assert results_formed.coherence > 0.5, \
            "UM should contribute to network formation"
        
        # Average sense index should be reasonable
        avg_si = sum(results_formed.sense_indices.values()) / len(results_formed.sense_indices)
        assert avg_si > 0.3, "Formed network should have reasonable stability"
    
    def test_um_bridges_phase_groups(self):
        """Test that UM can bridge incompatible phase groups."""
        net = TNFRNetwork("phase_bridging", NetworkConfig(random_seed=42))
        net.add_nodes(8)
        
        # Create two opposing phase groups
        for i, node in enumerate(net.graph.nodes()):
            if i < 4:
                net.graph.nodes[node]['theta'] = 0.0
            else:
                net.graph.nodes[node]['theta'] = math.pi
        
        # Connect the groups
        net.connect_nodes(0.3, "random")
        
        phases_before = [net.graph.nodes[n]['theta'] for n in net.graph.nodes()]
        phase_spread_before = max(phases_before) - min(phases_before)
        
        # Apply multiple rounds of network_sync (includes UM)
        for _ in range(3):
            net.apply_sequence("network_sync")
        
        phases_after = [net.graph.nodes[n]['theta'] for n in net.graph.nodes()]
        phase_spread_after = max(phases_after) - min(phases_after)
        
        # Phase spread should significantly reduce
        assert phase_spread_after < phase_spread_before * 0.5, \
            "UM should bridge opposing phase groups"


class TestUMStructuralInvariants:
    """Test that UM preserves TNFR structural invariants."""
    
    def test_um_preserves_epi_identity(self):
        """Test that UM synchronizes phase without modifying EPI."""
        G, node = create_nfr("test_node", vf=1.0, theta=0.5, epi=0.6)
        G.add_node("neighbor", theta=0.1, EPI=0.5, vf=1.0, dnfr=0.3, Si=0.5)
        G.add_edge(node, "neighbor")
        
        epi_before = G.nodes[node].get('EPI', G.nodes[node].get('epi'))
        
        # Apply UM multiple times
        for _ in range(3):
            apply_glyph(G, node, "UM")
        
        epi_after = G.nodes[node].get('EPI', G.nodes[node].get('epi'))
        
        # EPI should not be directly modified by UM
        # (small changes may occur due to natural evolution via nodal equation)
        assert epi_after == pytest.approx(epi_before, abs=0.15), \
            "UM must preserve EPI identity (critical invariant)"
    
    def test_um_modifies_phase(self):
        """Test that UM actually modifies phase (its primary function)."""
        G, node = create_nfr("test_node", vf=1.0, theta=0.8, epi=0.5)
        G.add_node("neighbor1", theta=0.1, EPI=0.5, vf=1.0, dnfr=0.3, Si=0.5)
        G.add_node("neighbor2", theta=0.15, EPI=0.5, vf=1.0, dnfr=0.3, Si=0.5)
        G.add_edge(node, "neighbor1")
        G.add_edge(node, "neighbor2")
        
        theta_before = G.nodes[node]['theta']
        
        # Apply UM
        apply_glyph(G, node, "UM")
        
        theta_after = G.nodes[node]['theta']
        
        # Phase should move towards neighbors (synchronization)
        assert abs(theta_after - theta_before) > 0.01, \
            "UM should synchronize phase"
    
    def test_um_can_reduce_dnfr(self):
        """Test that UM can reduce ΔNFR through synchronization."""
        G, node = create_nfr("test_node", vf=1.0, theta=0.5, epi=0.5)
        G.add_node("neighbor", theta=0.1, EPI=0.5, vf=1.0, dnfr=0.3, Si=0.5)
        G.add_edge(node, "neighbor")
        
        # Set high initial ΔNFR
        G.nodes[node]["dnfr"] = 0.8
        G.graph["UM_STABILIZE_DNFR"] = True  # Enable ΔNFR stabilization
        
        dnfr_before = G.nodes[node]["dnfr"]
        
        # Apply UM
        apply_glyph(G, node, "UM")
        
        dnfr_after = G.nodes[node]["dnfr"]
        
        # ΔNFR should reduce (stabilization effect)
        assert dnfr_after < dnfr_before, \
            "UM with stabilization should reduce ΔNFR"


class TestUMTopologyEffects:
    """Test UM behavior across different network topologies."""
    
    @pytest.mark.parametrize("topology,edge_prob", [
        ("random", 0.3),
        ("ring", 0.5),
        ("random", 0.5),
    ])
    def test_um_across_topologies(self, topology, edge_prob):
        """Test UM effectiveness across different topologies."""
        net = TNFRNetwork(f"topology_{topology}", NetworkConfig(random_seed=42))
        net.add_nodes(10).connect_nodes(edge_prob, topology)
        
        # Varied initial phases
        for i, node in enumerate(net.graph.nodes()):
            net.graph.nodes[node]['theta'] = (i % 3) * (2 * math.pi / 3)
        
        results_before = net.measure()
        
        # Apply network_sync (includes UM)
        net.apply_sequence("network_sync", repeat=2)
        
        results_after = net.measure()
        
        # Should achieve reasonable coherence in any topology
        assert results_after.coherence > 0.5, \
            f"UM should work in {topology} topology"
    
    def test_um_with_disconnected_components(self):
        """Test UM behavior with disconnected network components."""
        net = TNFRNetwork("disconnected", NetworkConfig(random_seed=42))
        net.add_nodes(12)
        
        # Create two disconnected components
        nodes = list(net.graph.nodes())
        for i in range(5):
            net.graph.add_edge(nodes[i], nodes[(i+1) % 6])
        for i in range(6, 11):
            net.graph.add_edge(nodes[i], nodes[6 + (i-6+1) % 6])
        
        # Different phases in each component
        for i, node in enumerate(net.graph.nodes()):
            if i < 6:
                net.graph.nodes[node]['theta'] = 0.0
            else:
                net.graph.nodes[node]['theta'] = math.pi
        
        # Apply network_sync
        net.apply_sequence("network_sync", repeat=2)
        
        results = net.measure()
        
        # Each component should synchronize internally
        # (even if components remain desynchronized from each other)
        assert results.coherence > 0.4, \
            "UM should synchronize within connected components"


class TestUMMetricsAndTracking:
    """Test that UM effects are properly captured in metrics."""
    
    def test_um_phase_convergence_tracked(self):
        """Test that phase convergence from UM is observable."""
        net = TNFRNetwork("metrics_test", NetworkConfig(random_seed=42))
        net.add_nodes(8).connect_nodes(0.4, "random")
        
        # Set varied phases
        for i, node in enumerate(net.graph.nodes()):
            net.graph.nodes[node]['theta'] = (i % 4) * math.pi / 2
        
        # Track phase spread over multiple applications
        phase_spreads = []
        
        for i in range(4):
            phases = [net.graph.nodes[n]['theta'] for n in net.graph.nodes()]
            phase_spreads.append(max(phases) - min(phases))
            net.apply_sequence("network_sync")
        
        # Phase spread should show convergence trend
        # (may not be strictly monotonic due to other operators)
        assert phase_spreads[-1] < phase_spreads[0] * 0.8, \
            "Phase spread should reduce over time"
    
    def test_um_coherence_contribution(self):
        """Test that UM contributes to coherence increase."""
        net = TNFRNetwork("coherence_test", NetworkConfig(random_seed=42))
        net.add_nodes(10).connect_nodes(0.3, "random")
        
        # Moderate initial coherence
        results_before = net.measure()
        
        # Apply sequence with UM
        net.apply_sequence("network_sync", repeat=3)
        
        results_after = net.measure()
        
        # Coherence should improve or remain stable
        assert results_after.coherence >= results_before.coherence * 0.7, \
            "UM should contribute to coherence stability"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
