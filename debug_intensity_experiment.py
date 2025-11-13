#!/usr/bin/env python3
"""
Debug multi-topology experiment with detailed logging to find the list vs float error.
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'benchmarks'))

from benchmark_utils import create_tnfr_topology
from tnfr.dynamics.dnfr import default_compute_delta_nfr
from tnfr.physics.fields import estimate_coherence_length
from tnfr.operators.definitions import Coherence

def debug_single_intensity():
    """Run one intensity to debug the comparison error."""
    print("=== DEBUGGING SINGLE INTENSITY EXPERIMENT ===")
    
    topology = "ws"
    n_nodes = 50
    n_runs = 3  # Small number for debugging
    intensity = 2.000
    seed_base = 42
    
    print(f"Testing topology={topology}, intensity={intensity}")
    
    xi_c_runs = []
    
    for run in range(n_runs):
        print(f"\n--- Run {run+1}/{n_runs} ---")
        try:
            # Create topology 
            seed = seed_base + run
            G = create_tnfr_topology(topology, n_nodes, seed)
            print(f"Created network with {G.number_of_nodes()} nodes")
            
            # Quick evolution (no coherence operators to avoid the API issue)
            for step in range(3):  # Fewer steps for debugging
                print(f"  Evolution step {step+1}/3...")
                default_compute_delta_nfr(G)
            
            # Estimate coherence length
            print("  Estimating coherence length...")
            xi_c = estimate_coherence_length(G, coherence_key="coherence")
            print(f"  xi_c = {xi_c}")
            print(f"  xi_c type = {type(xi_c)}")
            print(f"  xi_c repr = {repr(xi_c)}")
            
            # Check if it's really a scalar
            if hasattr(xi_c, '__len__') and not isinstance(xi_c, (str, bytes)):
                print(f"  ❌ WARNING: xi_c has length {len(xi_c)} - it's not a scalar!")
                print(f"  Contents: {list(xi_c) if hasattr(xi_c, '__iter__') else 'not iterable'}")
            else:
                print(f"  ✅ xi_c is a proper scalar")
            
            # Append to runs
            xi_c_runs.append(xi_c)
            print(f"  xi_c_runs now = {xi_c_runs}")
            print(f"  xi_c_runs types = {[type(x) for x in xi_c_runs]}")
            
        except Exception as e:
            print(f"  ❌ Run {run} failed: {e}")
            import traceback
            traceback.print_exc()
            xi_c_runs.append(0.0)
    
    print(f"\n=== FINAL DATA ANALYSIS ===")
    print(f"xi_c_runs = {xi_c_runs}")
    print(f"xi_c_runs types = {[type(x) for x in xi_c_runs]}")
    
    # Now test the exact operations that are failing
    print("\n--- Testing data aggregation ---")
    try:
        mean_val = np.mean(xi_c_runs)
        print(f"np.mean(xi_c_runs) = {mean_val}, type = {type(mean_val)}")
        
        xi_c_means = [mean_val]  # This is like the experiment does
        print(f"xi_c_means = {xi_c_means}")
        print(f"xi_c_means types = {[type(x) for x in xi_c_means]}")
        
        # This is the exact line that was failing
        print("Testing min/max operations...")
        min_val = min(xi_c_means)
        max_val = max(xi_c_means)
        print(f"min(xi_c_means) = {min_val}")
        print(f"max(xi_c_means) = {max_val}")
        print("✅ min/max operations succeeded!")
        
    except Exception as e:
        print(f"❌ Data aggregation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_single_intensity()