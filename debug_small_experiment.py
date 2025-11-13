#!/usr/bin/env python3
"""Quick test with smaller parameters to see exact error"""

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

def run_experiment_topology(topology: str, n_nodes: int = 50, n_runs: int = 3):
    """Run experiment for one topology with minimal parameters"""
    print(f"\n=== Testing topology: {topology} ===")
    
    # Parameters
    I_C = 2.015
    intensities = [2.0, 2.01]  # Just 2 intensities for speed
    seed_base = 12345
    
    xi_c_data = []
    
    for i, intensity in enumerate(intensities):
        print(f"  Intensity {intensity:.3f} ({i+1}/{len(intensities)})")
        
        xi_c_runs = []
        
        for run in range(n_runs):
            seed = seed_base + i * 1000 + run
            np.random.seed(seed)
            
            try:
                print(f"    Run {run+1}/{n_runs}...")
                
                # Create fresh topology
                G = create_test_topology(topology, n_nodes, seed)
                
                # Create network-wide perturbation proportional to intensity
                for node in G.nodes():
                    delta_I = intensity - I_C  # Distance from critical point
                    base_dnfr = delta_I * np.random.normal(0, 0.5)
                    G.nodes[node][DNFR_PRIMARY] = base_dnfr
                
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
                
                # Final ΔNFR calculation after evolution
                default_compute_delta_nfr(G)
                
                # Add per-node coherence for coherence length calculation
                for node in G.nodes():
                    dnfr = abs(G.nodes[node].get(DNFR_PRIMARY, 0.0))
                    G.nodes[node]['coherence'] = 1.0 / (1.0 + dnfr)
                
                # Measure ξ_C coherence length
                xi_c = estimate_coherence_length(G, coherence_key="coherence")
                xi_c_runs.append(xi_c)
                
                # Test other functions too
                symmetry = measure_phase_symmetry(G)
                coherence = compute_coherence(G)
                
                # NEW: Measure canonical fields for cross-validation
                phi_s_dict = compute_structural_potential(G)
                grad_phi_dict = compute_phase_gradient(G)
                k_phi_dict = compute_phase_curvature(G)
                
                # Convert dictionaries to scalar means
                phi_s = np.mean(list(phi_s_dict.values())) if phi_s_dict else 0.0
                grad_phi = np.mean(list(grad_phi_dict.values())) if grad_phi_dict else 0.0
                k_phi = np.mean(list(k_phi_dict.values())) if k_phi_dict else 0.0
                
                print(f"    ✓ Run {run+1} SUCCESS: xi_c={xi_c:.2f}, coherence={coherence:.3f}")
                
            except Exception as e:
                import traceback
                print(f"    ❌ Run {run+1} FAILED: {type(e).__name__}: {e}")
                print("    Traceback:")
                traceback.print_exc()
                xi_c_runs.append(0.0)
        
        print(f"  Intensity {intensity} results: {xi_c_runs}")
        xi_c_mean = np.mean(xi_c_runs)
        print(f"  Mean xi_c: {xi_c_mean}")
        
        xi_c_data.append({
            "intensity": intensity,
            "values": xi_c_runs,
            "mean": xi_c_mean,
        })
    
    return xi_c_data

if __name__ == "__main__":
    print("Quick experiment test with error tracking...")
    
    # Test just WS topology
    results = run_experiment_topology("ws", n_nodes=50, n_runs=3)
    
    print("\n=== FINAL RESULTS ===")
    for result in results:
        print(f"Intensity {result['intensity']}: mean xi_c = {result['mean']}")