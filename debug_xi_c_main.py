#!/usr/bin/env python3
"""
Debug estimate_coherence_length directly in main experiment context
"""
import sys
import os
sys.path.insert(0, os.path.abspath('src'))

import numpy as np
import networkx as nx

# Import all the same functions as the main experiment
from tnfr.physics.fields import estimate_coherence_length, measure_phase_symmetry
from tnfr.metrics.common import compute_coherence
from tnfr.operators.definitions import Dissonance, Coherence
from tnfr.dynamics.dnfr import default_compute_delta_nfr
from tnfr.constants import DNFR_PRIMARY

from benchmark_utils import create_tnfr_topology

def debug_xi_c_calculation():
    """Test xi_c calculation with exact same setup as main experiment"""
    print("=== Debugging ξ_C calculation ===\n")
    
    # Create exact same topology as main experiment
    G = create_tnfr_topology("ws", n_nodes=50, seed=42)
    print(f"Created WS topology: {len(G)} nodes, {len(G.edges())} edges")
    
    # Initialize with exact same pattern as main experiment
    for node in G.nodes():
        G.nodes[node]['EPI'] = np.random.normal(0, 0.1, 3)
        G.nodes[node]['theta'] = np.random.uniform(0, 2*np.pi)
        G.nodes[node]['nu_f'] = 1.0
    
    # Apply same intensity setup
    intensity = 2.015  # Critical point
    print(f"Using intensity: {intensity}")
    
    # Apply dissonance sequence like main experiment
    try:
        # Same sequence as main experiment
        dissonance_op = Dissonance()
        coherence_op = Coherence()
        
        # Apply operators
        for node in list(G.nodes())[:10]:  # Apply to first 10 nodes
            dissonance_op.apply(G, node)
            
        # Evolve for a few steps
        for step in range(5):
            for node in G.nodes():
                delta_nfr = default_compute_delta_nfr(G, node, DNFR_PRIMARY)
                current_epi = G.nodes[node]['EPI']
                nu_f = G.nodes[node]['nu_f']
                G.nodes[node]['EPI'] = current_epi + 0.1 * nu_f * delta_nfr
                
        # Apply coherence
        for node in list(G.nodes())[:10]:
            coherence_op.apply(G, node)
            
        print("Applied dissonance-evolution-coherence sequence")
        
        # Now test xi_c calculation
        print("\n--- Testing ξ_C calculation ---")
        
        # Check phases
        phases = [G.nodes[node]['theta'] for node in G.nodes()]
        print(f"Phase range: [{min(phases):.3f}, {max(phases):.3f}]")
        print(f"Phase std: {np.std(phases):.3f}")
        
        # Check coherence
        coherence = compute_coherence(G)
        print(f"Current coherence: {coherence:.3f}")
        
        # Check symmetry
        symmetry_result = measure_phase_symmetry(G)
        print(f"Symmetry result type: {type(symmetry_result)}")
        print(f"Symmetry result: {symmetry_result}")
        
        # Finally test xi_c
        xi_c = estimate_coherence_length(G)
        print(f"ξ_C result: {xi_c}")
        print(f"ξ_C type: {type(xi_c)}")
        
        if xi_c == 0.0:
            print("\n⚠️  ξ_C is zero - investigating why...")
            
            # Let's manually check the internals
            print("\nManual investigation:")
            print(f"Network size: {len(G)}")
            print(f"Network connectivity: {nx.is_connected(G)}")
            print(f"Average degree: {np.mean([d for n, d in G.degree()]):.2f}")
            
        return xi_c
        
    except Exception as e:
        print(f"❌ Exception in calculation: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    result = debug_xi_c_calculation()
    print(f"\nFinal result: {result}")