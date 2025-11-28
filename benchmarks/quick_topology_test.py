#!/usr/bin/env python3
"""
Quick Multi-Topology Test for ξ_C Critical Exponents
Test the fixes with reduced runs to ensure functionality
"""

import json
import numpy as np
from pathlib import Path
import sys
import time

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from tnfr.physics.fields import (
        estimate_coherence_length,
        fit_correlation_length_exponent,
        measure_phase_symmetry,
        compute_structural_potential,
        compute_phase_gradient,
        compute_phase_curvature
    )
    from benchmark_utils import create_tnfr_topology

    from tnfr.operators.definitions import Dissonance, Coherence
    from tnfr.dynamics.dnfr import default_compute_delta_nfr
    from tnfr.metrics.common import compute_coherence
    from tnfr.config import DNFR_PRIMARY
    import networkx as nx
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running from the project root")
    sys.exit(1)

# Critical point constant
I_C = 2.015

def quick_test():
    """Quick test of multi-topology functionality"""
    
    print("="*50)
    print("QUICK MULTI-TOPOLOGY TEST")
    print("="*50)
    
    # Test parameters - reduced for speed
    topologies = ["ws"]  # Start with just one
    n_nodes = 20  # Smaller network
    n_runs = 5   # Fewer runs
    intensities = [1.8, 2.015, 2.2]  # Just 3 points
    
    for topology in topologies:
        print(f"\nTesting {topology} topology...")
        
        for intensity in intensities:
            print(f"  Intensity {intensity:.3f}")
            
            for run in range(n_runs):
                seed = 12345 + run
                np.random.seed(seed)
                
                try:
                    # Create test topology
                    G = create_tnfr_topology(topology, n_nodes, seed)
                    
                    # Apply perturbations
                    for node in G.nodes():
                        delta_I = intensity - I_C
                        base_dnfr = delta_I * np.random.normal(0, 0.5)
                        G.nodes[node][DNFR_PRIMARY] = base_dnfr
                    
                    # Simple evolution
                    dissonance = Dissonance()
                    coherence_op = Coherence()
                    
                    # Perturb some nodes
                    perturbed_nodes = list(G.nodes())[:3]
                    for node in perturbed_nodes:
                        dissonance(G, node)
                    
                    # Evolve system
                    for step in range(3):
                        default_compute_delta_nfr(G)
                        # Stabilize some nodes
                        stabilized_nodes = list(G.nodes())[:2]
                        for node in stabilized_nodes:
                            coherence_op(G, node)
                    
                    # Final calculation
                    default_compute_delta_nfr(G)
                    
                    # Add per-node coherence
                    for node in G.nodes():
                        dnfr = abs(G.nodes[node].get(DNFR_PRIMARY, 0.0))
                        G.nodes[node]['coherence'] = 1.0 / (1.0 + dnfr)
                    
                    # Measure ξ_C
                    xi_c = estimate_coherence_length(G, coherence_key="coherence")
                    
                    # Measure canonical fields (convert dict results to scalars)
                    phi_s_dict = compute_structural_potential(G)
                    grad_phi_dict = compute_phase_gradient(G) 
                    k_phi_dict = compute_phase_curvature(G)
                    
                    # Convert to scalar means for display
                    phi_s = np.mean(list(phi_s_dict.values())) if phi_s_dict else 0.0
                    grad_phi = np.mean(list(grad_phi_dict.values())) if grad_phi_dict else 0.0
                    k_phi = np.mean(list(k_phi_dict.values())) if k_phi_dict else 0.0
                    
                    print(f"    Run {run}: ξ_C={xi_c:.1f}, "
                          f"Φ_s={phi_s:.3f}, |∇φ|={grad_phi:.3f}, K_φ={k_phi:.3f}")
                    
                except Exception as e:
                    print(f"    Run {run} failed: {e}")
    
    print("\n✅ Quick test completed successfully!")

if __name__ == "__main__":
    quick_test()