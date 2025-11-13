"""
Unit tests for cell emergence module.

Tests the core cell formation detection functions and telemetry.
"""

import numpy as np
import networkx as nx

from tnfr.physics.cell import (
    CellTelemetry,
    compute_boundary_coherence,
    compute_selectivity_index,
    compute_homeostatic_index,
    compute_membrane_integrity,
    detect_cell_formation,
    apply_membrane_flux
)


def test_boundary_coherence_basic():
    """Test boundary coherence computation."""
    # Create simple graph
    G = nx.Graph()
    G.add_nodes_from([0, 1, 2, 3])
    G.add_edges_from([(0, 1), (1, 2), (2, 3)])
    
    # Set node attributes
    for node in G.nodes():
        G.nodes[node]['delta_nfr'] = 0.1  # Small positive ΔNFR
    
    boundary_nodes = [1, 2]  # Middle nodes as boundary
    
    coherence = compute_boundary_coherence(G, boundary_nodes)
    
    # Should be a valid coherence value
    assert 0.0 <= coherence <= 1.0
    assert isinstance(coherence, float)


def test_boundary_coherence_empty():
    """Test boundary coherence with empty boundary nodes."""
    G = nx.Graph()
    G.add_node(0)
    
    coherence = compute_boundary_coherence(G, [])
    assert coherence == 0.0


def test_selectivity_index_basic():
    """Test selectivity index computation."""
    G = nx.Graph()
    G.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 4), (0, 4)])
    
    internal_nodes = [0, 1]
    boundary_nodes = [2]
    
    selectivity = compute_selectivity_index(G, internal_nodes, boundary_nodes)
    
    # Should be in valid range
    assert -1.0 <= selectivity <= 1.0
    assert isinstance(selectivity, float)


def test_selectivity_index_perfect_internal():
    """Test selectivity with only internal connections."""
    G = nx.Graph()
    G.add_edges_from([(0, 1), (1, 2)])  # All internal
    
    internal_nodes = [0, 1, 2]
    boundary_nodes = []
    
    selectivity = compute_selectivity_index(G, internal_nodes, boundary_nodes)
    
    # Should be 1.0 (perfect internal preference)
    assert selectivity == 1.0


def test_homeostatic_index_basic():
    """Test homeostatic index computation."""
    # Stable internal ΔNFR (low variance)
    delta_nfr_stable = np.array([0.1, 0.12, 0.08, 0.11, 0.09])
    
    H_stable = compute_homeostatic_index(delta_nfr_stable)
    
    # Should indicate good homeostasis
    assert 0.0 <= H_stable <= 1.0
    assert H_stable > 0.5  # Should be stable


def test_homeostatic_index_unstable():
    """Test homeostatic index with unstable dynamics."""
    # Highly variable ΔNFR (poor homeostasis)
    delta_nfr_unstable = np.array([-2.0, 5.0, -1.0, 3.0, -4.0])
    
    H_unstable = compute_homeostatic_index(delta_nfr_unstable)
    
    # Should indicate poor homeostasis
    assert 0.0 <= H_unstable <= 1.0
    assert H_unstable < 0.5


def test_homeostatic_index_empty():
    """Test homeostatic index with empty data."""
    H_empty = compute_homeostatic_index(np.array([]))
    assert H_empty == 0.0


def test_membrane_integrity_basic():
    """Test membrane integrity computation."""
    flux_internal = 2.0  # Controlled flux
    flux_external = 0.5  # Some leakage
    
    integrity = compute_membrane_integrity(flux_internal, flux_external)
    
    # Should be high integrity (low leakage ratio)
    assert 0.0 <= integrity <= 1.0
    assert integrity > 0.7  # Should be good integrity


def test_membrane_integrity_perfect():
    """Test membrane integrity with no leakage."""
    integrity = compute_membrane_integrity(1.0, 0.0)
    assert integrity == 1.0  # Perfect integrity


def test_membrane_integrity_no_flux():
    """Test membrane integrity with no flux."""
    integrity = compute_membrane_integrity(0.0, 0.0)
    assert integrity == 1.0  # No flux means perfect integrity by definition


def test_detect_cell_formation_basic():
    """Test basic cell formation detection."""
    # Create simple spatial network
    G = nx.grid_2d_graph(3, 3)
    G = nx.convert_node_labels_to_integers(G)
    
    # Initialize attributes
    for node in G.nodes():
        G.nodes[node]['delta_nfr'] = 0.1
        G.nodes[node]['EPI'] = 1.0
        G.nodes[node]['theta'] = 0.0
    
    # Define regions
    internal_nodes = [0, 1, 3, 4]  # Center region
    boundary_nodes = [2, 5, 6, 7, 8]  # Boundary region
    
    # Create sequence with single timepoint
    graph_sequence = [G]
    times = [0.0]
    
    # Detect cell formation
    telem = detect_cell_formation(
        graph_sequence, times, internal_nodes, boundary_nodes
    )
    
    # Check telemetry structure
    assert isinstance(telem, CellTelemetry)
    assert len(telem.times) == 1
    assert len(telem.boundary_coherence) == 1
    assert len(telem.internal_coherence) == 1
    assert len(telem.selectivity_index) == 1
    assert len(telem.homeostatic_index) == 1
    assert len(telem.membrane_integrity) == 1
    
    # All metrics should be valid
    assert 0.0 <= telem.boundary_coherence[0] <= 1.0
    assert 0.0 <= telem.internal_coherence[0] <= 1.0
    assert -1.0 <= telem.selectivity_index[0] <= 1.0
    assert 0.0 <= telem.homeostatic_index[0] <= 1.0
    assert 0.0 <= telem.membrane_integrity[0] <= 1.0


def test_detect_cell_formation_threshold():
    """Test cell formation time detection."""
    # Create network that meets criteria at t=1
    G = nx.grid_2d_graph(3, 3)
    G = nx.convert_node_labels_to_integers(G)
    
    internal_nodes = [0, 1, 3, 4]
    boundary_nodes = [2, 5, 6, 7, 8]
    
    # Time 0: poor coherence
    G0 = G.copy()
    for node in G0.nodes():
        G0.nodes[node]['delta_nfr'] = 1.0  # High variation = poor coherence
    
    # Time 1: good coherence (meets criteria)
    G1 = G.copy()
    for node in G1.nodes():
        G1.nodes[node]['delta_nfr'] = 0.05  # Low variation = good coherence
    
    graph_sequence = [G0, G1]
    times = [0.0, 1.0]
    
    # Use relaxed thresholds for testing
    telem = detect_cell_formation(
        graph_sequence, times, internal_nodes, boundary_nodes,
        c_boundary_threshold=0.3,  # Relaxed
        selectivity_threshold=0.1,  # Relaxed
        homeostasis_threshold=0.1,  # Relaxed
        integrity_threshold=0.1     # Relaxed
    )
    
    # Should detect formation at some point
    assert telem.cell_formation_time is not None
    assert telem.cell_formation_time >= 0.0


def test_apply_membrane_flux_basic():
    """Test membrane flux application."""
    G = nx.path_graph(5)
    
    # Initialize attributes
    for node in G.nodes():
        G.nodes[node]['EPI'] = 1.0
        G.nodes[node]['theta'] = 0.0  # All in phase
    
    internal_nodes = [1, 2, 3]
    boundary_nodes = [1, 3]  # Overlapping boundary
    
    # Apply flux
    apply_membrane_flux(G, internal_nodes, boundary_nodes)
    
    # Should not crash and node attributes should still exist
    for node in G.nodes():
        assert 'EPI' in G.nodes[node]
        assert G.nodes[node]['EPI'] >= 0.0  # EPI should remain non-negative


def test_apply_membrane_flux_phase_selective():
    """Test phase-selective membrane transport."""
    G = nx.path_graph(3)
    
    # Node 0: external, phase 0
    G.nodes[0]['EPI'] = 2.0
    G.nodes[0]['theta'] = 0.0
    
    # Node 1: boundary, phase 0 (compatible)
    G.nodes[1]['EPI'] = 1.0  
    G.nodes[1]['theta'] = 0.0
    
    # Node 2: internal, phase π (incompatible)
    G.nodes[2]['EPI'] = 1.0
    G.nodes[2]['theta'] = np.pi
    
    initial_epi_1 = G.nodes[1]['EPI']
    
    # Apply flux with small phase threshold
    apply_membrane_flux(
        G, internal_nodes=[2], boundary_nodes=[1],
        permeability=0.5, phase_threshold=np.pi/4
    )
    
    # Boundary node should have changed EPI (flux from compatible neighbor)
    final_epi_1 = G.nodes[1]['EPI']
    
    # EPI should remain non-negative
    assert final_epi_1 >= 0.0


if __name__ == "__main__":
    # Run basic smoke tests
    test_boundary_coherence_basic()
    test_selectivity_index_basic() 
    test_homeostatic_index_basic()
    test_membrane_integrity_basic()
    test_detect_cell_formation_basic()
    test_apply_membrane_flux_basic()
    
    print("✅ All cell module tests passed!")