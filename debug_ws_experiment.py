#!/usr/bin/env python3
"""
Debug ws topology experiment - isolate the exact error location.
"""

import sys
import os
import numpy as np
import networkx as nx

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'benchmarks'))

from benchmark_utils import create_tnfr_topology
from tnfr.dynamics.dnfr import default_compute_delta_nfr
from tnfr.physics.fields import estimate_coherence_length
from tnfr.operators.definitions import Coherence

def test_ws_experiment():
    """Test exactly what the ws topology experiment does."""
    print("Creating ws topology...")
    
    # Exact same parameters as in the multi-topology experiment
    G = create_tnfr_topology("ws", 50, 42)
    print(f"Network created with {G.number_of_nodes()} nodes")
    
    print("Running evolution steps...")
    
    try:
        # Evolution loop - this is where the error occurs
        for step in range(10):
            print(f"  Evolution step {step+1}/10...")
            
            # This is the line that was failing
            default_compute_delta_nfr(G)
            
            # Apply coherence to stabilize some nodes
            coherence_op = Coherence()
            node_list = list(G.nodes())
            stabilized_nodes = np.random.choice(
                node_list, size=max(1, 50//4), replace=False
            )
            
            for node in stabilized_nodes:
                try:
                    coherence_op.apply(G, node)
                except Exception as e:
                    print(f"    Coherence failed on node {node}: {e}")
        
        print("  Evolution completed successfully!")
        
        # Now test coherence length estimation
        print("Testing coherence length estimation...")
        xi_c = estimate_coherence_length(G, coherence_key="coherence")
        print(f"  ξ_C = {xi_c}, type = {type(xi_c)}")
        
        # Test if it can be put into a list and then processed
        xi_c_runs = [xi_c]
        print(f"  xi_c_runs = {xi_c_runs}")
        
        # This is the exact line that was failing
        xi_c_means = [np.mean([xi_c])]  # Simulating the data structure
        print(f"  xi_c_means = {xi_c_means}")
        print(f"  min/max test: min={min(xi_c_means)}, max={max(xi_c_means)}")
        
        print("✅ All tests passed!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_ws_experiment()