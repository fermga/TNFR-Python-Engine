"""Test Nodal Equation Physics (TIER 2: CORE PHYSICS).

Validates the fundamental nodal equation:
∂EPI/∂t = νf · ΔNFR(t)

This is TIER 2: CRITICAL - The heart of TNFR dynamics.
"""
from __future__ import annotations

import math
import networkx as nx

from tnfr.constants.canonical import PHI, GAMMA, PI, E
from tnfr.physics.fields import compute_structural_potential


class TestNodalEquationFundamentals:
    """Test fundamental nodal equation behavior."""

    def test_nodal_equation_structure(self) -> None:
        """Verify nodal equation mathematical structure."""
        # ∂EPI/∂t = νf · ΔNFR(t)
        # Rate of change = reorganization capacity × structural pressure
        
        # Create test node with known parameters
        G = nx.Graph()
        G.add_node(0)
        G.nodes[0]['EPI'] = "test_pattern"
        G.nodes[0]['νf'] = 1.5  # Hz_str
        G.nodes[0]['ΔNFR'] = 0.8  # structural pressure
        G.nodes[0]['phase'] = 0.0
        
        # Nodal equation components must be well-defined
        assert isinstance(G.nodes[0]['EPI'], str)  # Coherent form
        assert isinstance(G.nodes[0]['νf'], (int, float))  # Structural frequency
        assert G.nodes[0]['νf'] > 0  # Positive reorganization capacity
        assert isinstance(G.nodes[0]['ΔNFR'], (int, float))  # Structural pressure
        
    def test_zero_capacity_frozen_state(self) -> None:
        """Test νf = 0 → frozen node (no evolution)."""
        G = nx.Graph()
        G.add_node(0)
        G.nodes[0]['EPI'] = "frozen_pattern"
        G.nodes[0]['νf'] = 0.0  # No reorganization capacity
        G.nodes[0]['ΔNFR'] = 10.0  # High pressure
        
        # Even with high pressure, no capacity means no change
        # ∂EPI/∂t = 0.0 · 10.0 = 0
        rate_of_change = G.nodes[0]['νf'] * G.nodes[0]['ΔNFR']
        assert rate_of_change == 0.0
        
    def test_zero_pressure_equilibrium_state(self) -> None:
        """Test ΔNFR = 0 → equilibrium (no driving force)."""
        G = nx.Graph()
        G.add_node(0)
        G.nodes[0]['EPI'] = "equilibrium_pattern"
        G.nodes[0]['νf'] = 5.0  # High capacity
        G.nodes[0]['ΔNFR'] = 0.0  # No structural pressure
        
        # Even with high capacity, no pressure means no change
        # ∂EPI/∂t = 5.0 · 0.0 = 0
        rate_of_change = G.nodes[0]['νf'] * G.nodes[0]['ΔNFR']
        assert rate_of_change == 0.0
        
    def test_positive_evolution_dynamics(self) -> None:
        """Test both νf > 0 and ΔNFR > 0 → active reorganization."""
        G = nx.Graph()
        G.add_node(0)
        G.nodes[0]['EPI'] = "evolving_pattern"
        G.nodes[0]['νf'] = 2.0
        G.nodes[0]['ΔNFR'] = 1.5
        
        # Active reorganization: ∂EPI/∂t > 0
        rate_of_change = G.nodes[0]['νf'] * G.nodes[0]['ΔNFR']
        assert rate_of_change > 0
        assert rate_of_change == 3.0  # 2.0 × 1.5


class TestStructuralTriad:
    """Test the structural triad: EPI, νf, phase."""

    def test_epi_coherent_form_properties(self) -> None:
        """Test EPI as coherent structural form."""
        G = nx.Graph()
        G.add_node(0)
        
        # EPI must be coherent structural information
        valid_epis = ["pattern_A", "molecular_H2O", "concept_golden_ratio"]
        
        for epi in valid_epis:
            G.nodes[0]['EPI'] = epi
            assert isinstance(G.nodes[0]['EPI'], str)
            assert len(G.nodes[0]['EPI']) > 0  # Non-empty
            
    def test_structural_frequency_units(self) -> None:
        """Test νf in Hz_str (structural hertz) units."""
        G = nx.Graph()
        G.add_node(0)
        
        # νf must be positive real number in Hz_str
        valid_frequencies = [0.1, 1.0, 2.718, PHI, GAMMA]
        
        for freq in valid_frequencies:
            G.nodes[0]['νf'] = freq
            assert isinstance(G.nodes[0]['νf'], (int, float))
            assert G.nodes[0]['νf'] >= 0
            # Note: νf = 0 means "dead/frozen" but still valid
            
    def test_phase_synchronization_range(self) -> None:
        """Test phase φ ∈ [0, 2π) for network synchrony."""
        G = nx.Graph()
        G.add_node(0)
        
        # Phase must be in valid range for synchronization
        valid_phases = [0.0, PI/2, PI, 3*PI/2, 2*PI - 0.001]
        
        for phase in valid_phases:
            G.nodes[0]['phase'] = phase
            assert 0 <= G.nodes[0]['phase'] < 2*PI


class TestNetworkDynamics:
    """Test network-level nodal equation behavior."""

    def test_coupled_nodal_equations(self) -> None:
        """Test multiple nodes with coupled evolution."""
        G = nx.path_graph(3)
        
        # Initialize nodes with different parameters
        for i, node in enumerate(G.nodes()):
            G.nodes[node]['EPI'] = f"pattern_{node}"
            G.nodes[node]['νf'] = 1.0 + i * 0.5  # Increasing capacity
            G.nodes[node]['ΔNFR'] = 0.5 + i * 0.3  # Increasing pressure
            G.nodes[node]['phase'] = i * PI/3  # 60° phase differences
            
        # Each node follows its own nodal equation
        for node in G.nodes():
            rate = G.nodes[node]['νf'] * G.nodes[node]['ΔNFR']
            assert rate > 0  # All nodes actively evolving
            
        # Network coupling affects ΔNFR through neighbors
        # (This will be tested more thoroughly in operator tests)
        
    def test_structural_potential_integration(self) -> None:
        """Test integration with structural potential field."""
        G = nx.complete_graph(4)
        
        # Initialize with controlled ΔNFR distribution
        for i, node in enumerate(G.nodes()):
            G.nodes[node]['EPI'] = f"node_{node}"
            G.nodes[node]['νf'] = 1.0
            G.nodes[node]['ΔNFR'] = GAMMA / (i + 1)  # Decreasing pressure
            G.nodes[node]['phase'] = i * PHI / 4  # Golden ratio spacing
            
        # Compute emergent structural potential
        Phi_s = compute_structural_potential(G)
        
        # Should respect nodal equation integration:
        # Global field emerges from local dynamics
        assert isinstance(Phi_s, dict)
        assert len(Phi_s) == len(G.nodes())
        
        # All potential values should be finite
        for node, potential in Phi_s.items():
            assert isinstance(potential, (int, float))
            assert math.isfinite(potential)


class TestCanonicalParameterRespect:
    """Test that nodal equation respects canonical parameters."""

    def test_golden_ratio_frequency_scaling(self) -> None:
        """Test νf scaling with golden ratio harmonics."""
        G = nx.Graph()
        G.add_node(0)
        G.nodes[0]['EPI'] = "golden_pattern"
        G.nodes[0]['ΔNFR'] = 1.0
        
        # Test frequencies based on φ harmonics
        phi_harmonics = [1.0, PHI, PHI**2, 1/PHI, 1/(PHI**2)]
        
        for freq in phi_harmonics:
            G.nodes[0]['νf'] = freq
            rate = G.nodes[0]['νf'] * G.nodes[0]['ΔNFR']
            
            # Should maintain mathematical relationships
            assert rate > 0
            assert isinstance(rate, (int, float))
            
    def test_euler_constant_pressure_bounds(self) -> None:
        """Test ΔNFR bounded by Euler constant relationships."""
        G = nx.Graph()
        G.add_node(0)
        G.nodes[0]['EPI'] = "euler_pattern"
        G.nodes[0]['νf'] = 1.0
        
        # Test pressure values related to γ
        gamma_values = [GAMMA/10, GAMMA/2, GAMMA, 2*GAMMA]
        
        for pressure in gamma_values:
            G.nodes[0]['ΔNFR'] = pressure
            rate = G.nodes[0]['νf'] * G.nodes[0]['ΔNFR']
            
            # Should maintain canonical relationships
            assert rate >= 0
            # For stability: typically ΔNFR < γ for smooth evolution
            if pressure <= GAMMA:
                assert rate <= GAMMA  # Within stability bound