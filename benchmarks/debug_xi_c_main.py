#!/usr/bin/env python3
"""
Debug estimate_coherence_length directly in main experiment context
"""
import sys
import os
sys.path.insert(0, os.path.abspath('../src'))

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
    
    # Apply evolution sequence like successful experiment
    try:
        # Initialize DNFR perturbations (like successful experiment)
        for node in G.nodes():
            delta_I = intensity - 2.015  # Distance from critical point
            base_dnfr = delta_I * np.random.normal(0, 0.5)
            G.nodes[node][DNFR_PRIMARY] = base_dnfr
            
        # Apply operators like successful experiment  
        dissonance = Dissonance()
        coherence_op = Coherence()
        
        # Apply dissonance to some nodes
        node_list = list(G.nodes())
        perturbed_nodes = np.random.choice(
            node_list, size=min(10, len(node_list)), replace=False
        )
        for node in perturbed_nodes:
            dissonance(G, node)  # Call as function, not .apply()
            
        # Evolve for multiple steps like successful experiment
        for step in range(10):
            default_compute_delta_nfr(G)  # Full graph computation
            # Apply coherence to stabilize some nodes
            stabilized_nodes = np.random.choice(
                node_list, size=max(1, len(node_list)//4), replace=False
            )
            for node in stabilized_nodes:
                coherence_op(G, node)  # Call as function
                
        # Final ΔNFR calculation 
        default_compute_delta_nfr(G)
        
        # Add per-node coherence for coherence length calculation
        for node in G.nodes():
            dnfr = abs(G.nodes[node].get(DNFR_PRIMARY, 0.0))
            G.nodes[node]['coherence'] = 1.0 / (1.0 + dnfr)
            
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
        
        # Check if we have coherence values
        sample_node = list(G.nodes())[0]
        has_coherence = 'coherence' in G.nodes[sample_node]
        print(f"Has 'coherence' key: {has_coherence}")
        if has_coherence:
            coherence_sample = [G.nodes[n].get('coherence', 'MISSING') for n in list(G.nodes())[:5]]
            print(f"Coherence sample: {coherence_sample}")
        
        # Finally test xi_c
        xi_c = estimate_coherence_length(G)
        print(f"ξ_C result: {xi_c}")
        print(f"ξ_C type: {type(xi_c)}")
        
        # Also test with explicit coherence_key like main experiment
        xi_c_explicit = estimate_coherence_length(G, coherence_key="coherence")
        print(f"ξ_C with coherence_key='coherence': {xi_c_explicit}")
        
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