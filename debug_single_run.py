#!/usr/bin/env python3
"""Test single intensity run to see what's failing"""

import numpy as np
import networkx as nx
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from tnfr.physics.fields import (
    estimate_coherence_length, 
    compute_structural_potential, 
    compute_phase_gradient, 
    compute_phase_curvature,
    measure_phase_symmetry
)
from tnfr.operators.definitions import Dissonance, Coherence
from tnfr.dynamics.dnfr import default_compute_delta_nfr
from tnfr.constants import DNFR_PRIMARY
from tnfr.metrics.common import compute_coherence

def create_test_topology(topology: str, n_nodes: int, seed: int) -> nx.Graph:
    """Create test topology matching experiment"""
    np.random.seed(seed)
    
    if topology == "ws":
        G = nx.watts_strogatz_graph(n_nodes, 4, 0.3, seed=seed)
    elif topology == "scale_free":
        G = nx.barabasi_albert_graph(n_nodes, 2, seed=seed)
    elif topology == "grid":
        side = int(np.sqrt(n_nodes))
        G = nx.grid_2d_graph(side, side)
        G = nx.convert_node_labels_to_integers(G)
    else:
        raise ValueError(f"Unknown topology: {topology}")
    
    # Initialize all node attributes like in the experiment
    for node in G.nodes():
        G.nodes[node][DNFR_PRIMARY] = 0.0
        G.nodes[node]['nu_f'] = 1.0
        G.nodes[node]['theta'] = np.random.uniform(0, 2*np.pi)
        G.nodes[node]['EPI'] = np.random.normal(0, 0.1)
    
    return G

def test_single_run():
    """Test exactly one run to catch specific errors"""
    print("Testing single run like multi-topology experiment...")
    
    # Parameters matching the experiment
    I_C = 2.015
    intensity = 2.0  # Near critical
    n_nodes = 50
    topology = "ws"
    run = 0
    seed_base = 42
    
    try:
        seed = seed_base + 0 * 1000 + run  # intensity index=0
        np.random.seed(seed)
        
        # Create fresh topology
        G = create_test_topology(topology, n_nodes, seed)
        print(f"✓ Created {topology} network with {len(G.nodes)} nodes")
        
        # Create network-wide perturbation proportional to intensity
        for node in G.nodes():
            delta_I = intensity - I_C  # Distance from critical point
            base_dnfr = delta_I * np.random.normal(0, 0.5)
            G.nodes[node][DNFR_PRIMARY] = base_dnfr
        print(f"✓ Applied network perturbations")
        
        # Apply multiple dissonance operators for network evolution
        perturb_factor = 0.2 * n_nodes * abs(intensity - I_C)
        n_perturbed = max(1, int(perturb_factor))
        node_list = list(G.nodes())
        perturbed_nodes = np.random.choice(
            node_list,
            size=min(n_perturbed, len(node_list)),
            replace=False
        )
        
        dissonance = Dissonance()
        for node in perturbed_nodes:
            dissonance(G, node)
        print(f"✓ Applied dissonance to {len(perturbed_nodes)} nodes")
        
        # Let system evolve for several steps to develop correlations
        for step in range(10):  # Multiple evolution steps
            default_compute_delta_nfr(G)
            # Apply coherence to stabilize some nodes
            coherence_op = Coherence()
            node_list = list(G.nodes())
            stabilized_nodes = np.random.choice(
                node_list, size=max(1, n_nodes//4), replace=False
            )
            for node in stabilized_nodes:
                coherence_op(G, node)
        print(f"✓ Evolved network for 10 steps")
        
        # Final ΔNFR calculation after evolution
        default_compute_delta_nfr(G)
        
        # Add per-node coherence for coherence length calculation
        for node in G.nodes():
            dnfr = abs(G.nodes[node].get(DNFR_PRIMARY, 0.0))
            G.nodes[node]['coherence'] = 1.0 / (1.0 + dnfr)
        print(f"✓ Added per-node coherence")
        
        # Measure ξ_C coherence length
        print("Testing estimate_coherence_length...")
        xi_c = estimate_coherence_length(G, coherence_key="coherence")
        print(f"✓ xi_c = {xi_c}")
        
        # Measure phase symmetry
        print("Testing measure_phase_symmetry...")
        symmetry = measure_phase_symmetry(G)
        print(f"✓ symmetry = {symmetry}")
        
        # Measure global coherence
        print("Testing compute_coherence...")
        coherence = compute_coherence(G)
        print(f"✓ coherence = {coherence}")
        
        # NEW: Measure canonical fields for cross-validation
        print("Testing compute_structural_potential...")
        phi_s_dict = compute_structural_potential(G)
        print(f"✓ phi_s_dict type: {type(phi_s_dict)}, len: {len(phi_s_dict) if phi_s_dict else 0}")
        
        print("Testing compute_phase_gradient...")
        grad_phi_dict = compute_phase_gradient(G)
        print(f"✓ grad_phi_dict type: {type(grad_phi_dict)}, len: {len(grad_phi_dict) if grad_phi_dict else 0}")
        
        print("Testing compute_phase_curvature...")
        k_phi_dict = compute_phase_curvature(G)
        print(f"✓ k_phi_dict type: {type(k_phi_dict)}, len: {len(k_phi_dict) if k_phi_dict else 0}")
        
        # Convert dictionaries to scalar means
        phi_s = np.mean(list(phi_s_dict.values())) if phi_s_dict else 0.0
        grad_phi = np.mean(list(grad_phi_dict.values())) if grad_phi_dict else 0.0
        k_phi = np.mean(list(k_phi_dict.values())) if k_phi_dict else 0.0
        
        print(f"✓ Converted to scalars: phi_s={phi_s}, grad_phi={grad_phi}, k_phi={k_phi}")
        
        print("\n🎉 SUCCESS! All measurements completed without error")
        return True
        
    except Exception as e:
        import traceback
        print(f"\n❌ FAILED at some point: {e}")
        print("Full traceback:")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_single_run()
    print(f"\nResult: {'SUCCESS' if success else 'FAILED'}")