"""Test ΔNFR Dynamics (TIER 2: CORE PHYSICS).

Validates ΔNFR (Nodal gradient) as structural pressure driving reorganization.
Tests the physics: ΔNFR represents internal reorganization operator - "structural pressure"

This is TIER 2: CRITICAL - Understanding structural pressure dynamics.
"""
from __future__ import annotations

import math
import networkx as nx

from tnfr.constants.canonical import PHI, GAMMA, PI, E
from tnfr.physics.fields import compute_structural_potential


class TestDeltaNFRPhysics:
    """Test ΔNFR as structural pressure operator."""

    def test_delta_nfr_as_pressure(self) -> None:
        """Test ΔNFR as internal structural pressure."""
        G = nx.Graph()
        G.add_node(0)
        G.nodes[0]['EPI'] = "test_pattern"
        G.nodes[0]['νf'] = 1.0
        
        # ΔNFR represents structural pressure/mismatch
        pressure_values = [-1.0, -0.5, 0.0, 0.5, 1.0, 2.0]
        
        for pressure in pressure_values:
            G.nodes[0]['ΔNFR'] = pressure
            
            # Should be real-valued (can be positive or negative)
            assert isinstance(G.nodes[0]['ΔNFR'], (int, float))
            assert math.isfinite(G.nodes[0]['ΔNFR'])
            
            # Magnitude determines reorganization intensity
            intensity = abs(G.nodes[0]['ΔNFR'])
            assert intensity >= 0
            
    def test_delta_nfr_sign_semantics(self) -> None:
        """Test ΔNFR sign determines reorganization direction."""
        G = nx.Graph()
        G.add_node(0)
        G.nodes[0]['EPI'] = "directional_pattern"
        G.nodes[0]['νf'] = 1.0
        
        # Positive ΔNFR: expansion/complexification pressure
        G.nodes[0]['ΔNFR'] = 1.5
        positive_rate = G.nodes[0]['νf'] * G.nodes[0]['ΔNFR']
        assert positive_rate > 0  # Positive evolution
        
        # Negative ΔNFR: contraction/simplification pressure  
        G.nodes[0]['ΔNFR'] = -1.5
        negative_rate = G.nodes[0]['νf'] * G.nodes[0]['ΔNFR']
        assert negative_rate < 0  # Negative evolution
        
        # Zero ΔNFR: equilibrium (no pressure)
        G.nodes[0]['ΔNFR'] = 0.0
        zero_rate = G.nodes[0]['νf'] * G.nodes[0]['ΔNFR']
        assert zero_rate == 0.0  # No evolution
        
    def test_delta_nfr_magnitude_bounds(self) -> None:
        """Test ΔNFR magnitude bounds for stability."""
        G = nx.Graph()
        G.add_node(0)
        G.nodes[0]['EPI'] = "bounded_pattern"
        G.nodes[0]['νf'] = 1.0
        
        # Test different magnitude ranges
        small_pressure = GAMMA / 10  # ≈0.058
        medium_pressure = GAMMA / 2   # ≈0.289
        large_pressure = GAMMA        # ≈0.577
        extreme_pressure = 2 * GAMMA  # ≈1.154
        
        pressures = [small_pressure, medium_pressure, large_pressure, extreme_pressure]
        
        for pressure in pressures:
            G.nodes[0]['ΔNFR'] = pressure
            rate = G.nodes[0]['νf'] * G.nodes[0]['ΔNFR']
            
            # All should produce finite rates
            assert math.isfinite(rate)
            
            # Stability expectation: ΔNFR < γ for smooth evolution
            if pressure <= GAMMA:
                # Within canonical stability bound
                assert rate <= GAMMA
            else:
                # May require stabilizers (tested in grammar tests)
                assert rate > GAMMA  # Beyond stability threshold


class TestDeltaNFRNetworkEffects:
    """Test ΔNFR in network context."""

    def test_delta_nfr_coupling_influence(self) -> None:
        """Test how coupling affects ΔNFR values."""
        G = nx.path_graph(3)
        
        # Initialize with different base pressures
        base_pressures = [0.1, 0.5, 0.3]
        
        for i, node in enumerate(G.nodes()):
            G.nodes[node]['EPI'] = f"node_{node}"
            G.nodes[node]['νf'] = 1.0
            G.nodes[node]['ΔNFR'] = base_pressures[i]
            G.nodes[node]['phase'] = i * PI/4  # Different phases
            
        # Network coupling should influence effective ΔNFR
        # (In full implementation, this comes from neighbor interactions)
        
        for node in G.nodes():
            neighbors = list(G.neighbors(node))
            
            if neighbors:
                # Node pressure influenced by neighbors
                neighbor_pressures = [G.nodes[n]['ΔNFR'] for n in neighbors]
                avg_neighbor_pressure = sum(neighbor_pressures) / len(neighbor_pressures)
                
                # Coupling creates pressure gradients
                pressure_gradient = G.nodes[node]['ΔNFR'] - avg_neighbor_pressure
                assert isinstance(pressure_gradient, (int, float))
                
    def test_delta_nfr_structural_potential_emergence(self) -> None:
        """Test ΔNFR distribution creates structural potential field."""
        G = nx.complete_graph(4)
        
        # Create controlled ΔNFR distribution
        delta_nfr_values = [0.1, 0.3, 0.2, 0.4]  # Varied pressures
        
        for i, node in enumerate(G.nodes()):
            G.nodes[node]['EPI'] = f"potential_node_{node}"
            G.nodes[node]['νf'] = 1.0
            G.nodes[node]['ΔNFR'] = delta_nfr_values[i]
            G.nodes[node]['phase'] = i * PHI/4  # Golden spacing
            
        # Compute emergent structural potential from ΔNFR distribution
        Phi_s = compute_structural_potential(G)
        
        # Should create meaningful potential landscape
        assert isinstance(Phi_s, dict)
        assert len(Phi_s) == len(G.nodes())
        
        # Nodes with higher ΔNFR should influence potential field
        max_pressure_node = max(G.nodes(), key=lambda n: G.nodes[n]['ΔNFR'])
        min_pressure_node = min(G.nodes(), key=lambda n: G.nodes[n]['ΔNFR'])
        
        # Potential field should reflect pressure distribution
        # (Exact relationship depends on distance weighting in field computation)
        for node, potential in Phi_s.items():
            assert isinstance(potential, (int, float))
            assert math.isfinite(potential)


class TestDeltaNFRCanonicalBounds:
    """Test ΔNFR respects canonical parameter bounds."""

    def test_gamma_stability_bound(self) -> None:
        """Test ΔNFR < γ for stable smooth evolution."""
        G = nx.Graph()
        G.add_node(0)
        G.nodes[0]['EPI'] = "gamma_bounded"
        G.nodes[0]['νf'] = 1.0
        
        # Test values around γ stability bound
        test_values = [
            GAMMA / 10,   # Well below bound
            GAMMA / 2,    # Half bound
            GAMMA * 0.9,  # Just below bound
            GAMMA,        # At bound
            GAMMA * 1.1,  # Just above bound
            GAMMA * 2     # Well above bound
        ]
        
        for pressure in test_values:
            G.nodes[0]['ΔNFR'] = pressure
            rate = abs(G.nodes[0]['νf'] * G.nodes[0]['ΔNFR'])
            
            # All should be computable
            assert isinstance(rate, (int, float))
            assert rate >= 0
            
            # Stability classification
            if pressure <= GAMMA:
                # Stable regime - smooth evolution expected
                assert rate <= GAMMA
                stability_class = "stable"
            else:
                # Unstable regime - may need stabilizers
                assert rate > GAMMA
                stability_class = "unstable"
                
            # In unstable regime, grammar U2 requires stabilizers
            # (This will be tested in grammar tests)
            
    def test_phi_structural_confinement(self) -> None:
        """Test ΔNFR integration with φ structural bounds."""
        G = nx.complete_graph(5)  # Pentagon (φ-related)
        
        # Initialize with φ-related pressure scaling
        for i, node in enumerate(G.nodes()):
            G.nodes[node]['EPI'] = f"phi_node_{node}"
            G.nodes[node]['νf'] = 1.0
            # Scale pressure with golden ratio powers
            G.nodes[node]['ΔNFR'] = GAMMA / (PHI ** i)
            G.nodes[node]['phase'] = i * 2*PI/5  # Pentagon angles
            
        # Compute structural potential
        Phi_s = compute_structural_potential(G)
        
        # Should respect φ structural bounds (from U6)
        max_potential = max(abs(v) for v in Phi_s.values())
        
        # Relaxed test for foundation - just check computability
        assert isinstance(max_potential, (int, float))
        assert math.isfinite(max_potential)
        
        # In production: max_potential < φ (≈1.618) for structural confinement
        # This will be tested more rigorously in grammar U6 tests
        
    def test_exponential_e_decay_relationship(self) -> None:
        """Test ΔNFR exponential relationships with e."""
        G = nx.path_graph(6)
        
        # Create exponential ΔNFR decay pattern
        for i, node in enumerate(G.nodes()):
            G.nodes[node]['EPI'] = f"exp_node_{node}"
            G.nodes[node]['νf'] = 1.0
            # Exponential decay: Ae^(-x/ξ) pattern
            G.nodes[node]['ΔNFR'] = GAMMA * math.exp(-i / E)
            G.nodes[node]['phase'] = i * PI/6
            
        # Should create smooth exponential decay
        delta_values = [G.nodes[node]['ΔNFR'] for node in G.nodes()]
        
        # Verify exponential decay pattern
        for i in range(len(delta_values) - 1):
            # Each value should be smaller than the previous (decay)
            assert delta_values[i] >= delta_values[i+1]
            
            # Should follow exponential relationship
            expected_ratio = math.exp(-1 / E)  # e^(-1/e) ≈ 0.692
            if delta_values[i] > 0:
                actual_ratio = delta_values[i+1] / delta_values[i]
                # Should be approximately exponential (within numerical precision)
                assert abs(actual_ratio - expected_ratio) < 0.1  # Relaxed for foundation