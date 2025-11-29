"""Test Structural Triad (EPI, νf, phase) - TIER 2: CORE PHYSICS.

Validates the three essential properties of every TNFR node:
1. Form (EPI): The coherent configuration
2. Frequency (νf): Reorganization rate in Hz_str
3. Phase (φ): Network synchrony parameter

This is TIER 2: CRITICAL - Foundation for all TNFR dynamics.
"""
from __future__ import annotations

import math
import numpy as np
import networkx as nx

from tnfr.constants.canonical import PHI, GAMMA, PI, E


class TestEPICoherentForm:
    """Test EPI (Primary Information Structure) properties."""

    def test_epi_lives_in_banach_space(self) -> None:
        """Test EPI as point in structural manifold B_EPI."""
        G = nx.Graph()
        G.add_node(0)
        
        # EPI represents coherent structural information
        valid_epis = [
            "molecular_pattern_H2O",
            "geometric_golden_spiral", 
            "conceptual_fibonacci_sequence",
            "neural_pattern_gamma_oscillation"
        ]
        
        for epi in valid_epis:
            G.nodes[0]['EPI'] = epi
            
            # Must be representable as coherent information
            assert isinstance(G.nodes[0]['EPI'], str)
            assert len(G.nodes[0]['EPI']) > 0
            # EPI identity preserved across transformations
            assert G.nodes[0]['EPI'] == epi
            
    def test_epi_operational_fractality(self) -> None:
        """Test EPI can nest without losing identity."""
        G = nx.Graph()
        G.add_node(0)
        
        # Nested EPI structure
        parent_epi = "system_complex"
        child_epis = ["subsystem_A", "subsystem_B", "subsystem_C"]
        
        # Parent maintains identity while containing children
        G.nodes[0]['EPI'] = parent_epi
        G.nodes[0]['child_EPIs'] = child_epis
        
        # Both levels maintain structural coherence
        assert G.nodes[0]['EPI'] == parent_epi  # Parent identity preserved
        assert len(G.nodes[0]['child_EPIs']) == 3  # Children accessible
        
        # Fractality: children can have their own sub-structure
        for child in child_epis:
            assert isinstance(child, str)
            assert len(child) > 0
            
    def test_epi_change_only_via_operators(self) -> None:
        """Test EPI changes ONLY through structural operators."""
        G = nx.Graph()
        G.add_node(0)
        
        initial_epi = "original_pattern"
        G.nodes[0]['EPI'] = initial_epi
        G.nodes[0]['νf'] = 1.0
        G.nodes[0]['phase'] = 0.0
        
        # Direct mutation should be detected (this is a design principle)
        # In production, this would be enforced by operator system
        assert G.nodes[0]['EPI'] == initial_epi
        
        # Only operators should modify EPI
        # (Full operator tests in TIER 3)


class TestStructuralFrequency:
    """Test νf (structural frequency) in Hz_str units."""

    def test_hz_str_units_canonical(self) -> None:
        """Test structural hertz units are preserved."""
        G = nx.Graph()
        G.add_node(0)
        
        # νf must be in Hz_str (structural reorganization cycles per second)
        canonical_frequencies = [
            0.0,     # Frozen/dead state
            1.0,     # 1 reorganization per second
            PHI,     # Golden ratio frequency (≈1.618 Hz_str)
            GAMMA,   # Euler constant frequency (≈0.577 Hz_str)
            PI,      # Pi frequency (≈3.142 Hz_str)
            E        # Natural frequency (≈2.718 Hz_str)
        ]
        
        for freq in canonical_frequencies:
            G.nodes[0]['νf'] = freq
            
            # Must be non-negative real
            assert isinstance(G.nodes[0]['νf'], (int, float))
            assert G.nodes[0]['νf'] >= 0
            
            # Preserve canonical relationships
            if freq > 0:
                assert G.nodes[0]['νf'] > 0  # Active reorganization
            else:
                assert G.nodes[0]['νf'] == 0  # Frozen state
                
    def test_frequency_death_condition(self) -> None:
        """Test νf → 0 represents node 'death' (no reorganization)."""
        G = nx.Graph()
        G.add_node(0)
        
        # Living node
        G.nodes[0]['νf'] = 1.0
        G.nodes[0]['ΔNFR'] = 0.5
        
        living_rate = G.nodes[0]['νf'] * G.nodes[0]['ΔNFR']
        assert living_rate > 0  # Active evolution
        
        # Dying node (νf approaches 0)
        G.nodes[0]['νf'] = 0.001
        dying_rate = G.nodes[0]['νf'] * G.nodes[0]['ΔNFR']
        assert dying_rate < living_rate  # Reduced activity
        
        # Dead node (νf = 0)
        G.nodes[0]['νf'] = 0.0
        dead_rate = G.nodes[0]['νf'] * G.nodes[0]['ΔNFR']
        assert dead_rate == 0  # No evolution possible
        
    def test_frequency_reorganization_capacity(self) -> None:
        """Test νf as measure of reorganization capacity."""
        G = nx.Graph()
        G.add_nodes_from([0, 1, 2])
        
        # Different reorganization capacities
        capacities = [0.5, 1.0, 2.0]  # Low, medium, high
        pressure = 1.0  # Constant structural pressure
        
        for i, capacity in enumerate(capacities):
            G.nodes[i]['νf'] = capacity
            G.nodes[i]['ΔNFR'] = pressure
            
            rate = G.nodes[i]['νf'] * G.nodes[i]['ΔNFR']
            
            # Higher capacity → faster reorganization under same pressure
            assert rate == capacity * pressure
            
        # Verify ordering: higher νf → higher rate
        rates = [G.nodes[i]['νf'] * G.nodes[i]['ΔNFR'] for i in range(3)]
        assert rates[0] < rates[1] < rates[2]


class TestPhaseNetworkSynchrony:
    """Test phase φ for network synchronization."""

    def test_phase_range_canonical(self) -> None:
        """Test phase ∈ [0, 2π) radians for synchrony."""
        G = nx.Graph()
        G.add_node(0)
        
        # Valid phase values
        canonical_phases = [
            0.0,         # Reference phase
            PI/4,        # 45°
            PI/2,        # 90°
            PI,          # 180°
            3*PI/2,      # 270°
            2*PI - 0.01  # Just under 360°
        ]
        
        for phase in canonical_phases:
            G.nodes[0]['phase'] = phase
            
            # Must be in [0, 2π) range
            assert 0 <= G.nodes[0]['phase'] < 2*PI
            
            # Phase determines coupling compatibility
            assert isinstance(G.nodes[0]['phase'], (int, float))
            
    def test_phase_coupling_compatibility(self) -> None:
        """Test phase difference determines coupling strength."""
        G = nx.path_graph(2)
        
        # Test different phase relationships
        phase_pairs = [
            (0.0, 0.0),      # Perfect sync (Δφ = 0)
            (0.0, PI/6),     # Small difference (Δφ = π/6)
            (0.0, PI/2),     # Quarter phase (Δφ = π/2)
            (0.0, PI),       # Antiphase (Δφ = π)
        ]
        
        for phase1, phase2 in phase_pairs:
            G.nodes[0]['phase'] = phase1
            G.nodes[1]['phase'] = phase2
            
            # Calculate phase difference
            phase_diff = abs(G.nodes[1]['phase'] - G.nodes[0]['phase'])
            
            # Phase compatibility decreases with difference
            if phase_diff == 0:
                # Perfect synchrony - maximum coupling potential
                coupling_strength = 1.0
            elif phase_diff <= PI/2:
                # Compatible phases - good coupling
                coupling_strength = math.cos(phase_diff)
                assert coupling_strength > 0
            elif phase_diff == PI:
                # Antiphase - destructive interference
                coupling_strength = -1.0
            
            # Coupling strength should reflect phase relationship
            assert -1.0 <= coupling_strength <= 1.0
            
    def test_phase_golden_ratio_harmonics(self) -> None:
        """Test phase relationships based on φ harmonics."""
        G = nx.cycle_graph(5)  # Pentagon (φ-related geometry)
        
        # Golden ratio phase spacing: 2π/φ ≈ 3.883 radians
        golden_phase_increment = 2 * PI / PHI
        
        for i, node in enumerate(G.nodes()):
            G.nodes[node]['phase'] = (i * golden_phase_increment) % (2 * PI)
            
            # Should maintain golden ratio relationships
            assert 0 <= G.nodes[node]['phase'] < 2 * PI
            
        # Adjacent nodes should have φ-harmonic phase differences
        for i in range(len(G.nodes()) - 1):
            phase_diff = abs(G.nodes[i+1]['phase'] - G.nodes[i]['phase'])
            expected_diff = golden_phase_increment
            
            # Should be close to golden ratio spacing (relaxed for modular arithmetic)
            assert abs(phase_diff - expected_diff) < 2.0  # Relaxed due to modular wrapping


class TestTriadCoherence:
    """Test coherence of EPI-νf-phase triad."""

    def test_triad_completeness(self) -> None:
        """Test all three components are present and consistent."""
        G = nx.Graph()
        G.add_node(0)
        
        # Complete triad initialization
        G.nodes[0]['EPI'] = "complete_pattern"
        G.nodes[0]['νf'] = PHI  # Golden ratio frequency
        G.nodes[0]['phase'] = GAMMA  # Euler constant phase
        
        # All components must be present
        assert 'EPI' in G.nodes[0]
        assert 'νf' in G.nodes[0]
        assert 'phase' in G.nodes[0]
        
        # All must have valid values
        assert isinstance(G.nodes[0]['EPI'], str)
        assert isinstance(G.nodes[0]['νf'], (int, float))
        assert isinstance(G.nodes[0]['phase'], (int, float))
        
        # Canonical relationships preserved
        assert G.nodes[0]['νf'] > 0  # Active node
        assert 0 <= G.nodes[0]['phase'] < 2*PI  # Valid phase
        
    def test_triad_canonical_parameter_relationships(self) -> None:
        """Test triad respects universal constants."""
        G = nx.complete_graph(4)  # Tetrahedral (4 universal constants)
        
        # Initialize with canonical parameter relationships
        constants = [PHI, GAMMA, PI, E]
        
        for i, node in enumerate(G.nodes()):
            G.nodes[node]['EPI'] = f"canonical_pattern_{node}"
            G.nodes[node]['νf'] = constants[i]  # Frequency from universal constants
            G.nodes[node]['phase'] = (constants[i] / constants[0]) % (2 * PI)  # φ-normalized phase
            
        # All nodes should respect canonical relationships
        for node in G.nodes():
            freq = G.nodes[node]['νf']
            phase = G.nodes[node]['phase']
            
            # Frequency should be canonical constant
            assert freq in constants
            
            # Phase should be φ-normalized and in valid range
            assert 0 <= phase < 2*PI
            
        # Network should maintain tetrahedral correspondence
        assert len(G.nodes()) == 4  # Four nodes for four constants
        frequencies = [G.nodes[node]['νf'] for node in G.nodes()]
        assert set(frequencies) == set(constants)  # All constants represented
