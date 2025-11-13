#!/usr/bin/env python3
"""
Simple debug for coherence length calculation
"""
import sys
import os
sys.path.insert(0, os.path.abspath('src'))

import numpy as np
import networkx as nx
from tnfr.dynamics import default_compute_delta_nfr
from tnfr.constants import DNFR_PRIMARY
from tnfr.operators import Dissonance, Coherence
from tnfr.physics.fields import estimate_coherence_length
from tnfr.metrics.common import compute_coherence


def simple_debug():
    """Simple debug what happens during coherence length calculation."""
    
    # Create simple network directly with NetworkX
    G = nx.watts_strogatz_graph(30, 4, 0.3, seed=42)
    
    # Initialize TNFR attributes
    for node in G.nodes():
        G.nodes[node]['EPI'] = np.array([0.5])  # Initial EPI
        G.nodes[node]['vf'] = 1.0  # Structural frequency  
        G.nodes[node]['theta'] = np.random.uniform(0, 2*np.pi)  # Phase
        G.nodes[node][DNFR_PRIMARY] = 0.0  # Initialize ΔNFR
    
    # Apply some dissonance to create variation
    dissonance_op = Dissonance()
    coherence_op = Coherence()
    
    print("=== Initial State ===")
    initial_dnfr = []
    for node in G.nodes():
        dnfr = abs(G.nodes[node].get(DNFR_PRIMARY, 0.0))
        initial_dnfr.append(dnfr)
    print(f"Initial ΔNFR: min={min(initial_dnfr):.6f}, max={max(initial_dnfr):.6f}, mean={np.mean(initial_dnfr):.6f}")
    
    # Apply some operators to create variation
    node_list = list(G.nodes())
    
    # Apply dissonance to some nodes
    dissonance_nodes = np.random.choice(node_list, size=10, replace=False)
    for node in dissonance_nodes:
        dissonance_op(G, node)
    
    # Evolve for a few steps
    for step in range(5):
        default_compute_delta_nfr(G)
        
        # Apply coherence to some nodes to create gradients
        coherence_nodes = np.random.choice(node_list, size=7, replace=False)
        for node in coherence_nodes:
            coherence_op(G, node)
    
    # Final ΔNFR calculation
    default_compute_delta_nfr(G)
    
    print("\n=== After Evolution ===")
    evolved_dnfr = []
    for node in G.nodes():
        dnfr = abs(G.nodes[node].get(DNFR_PRIMARY, 0.0))
        evolved_dnfr.append(dnfr)
    print(f"Evolved ΔNFR: min={min(evolved_dnfr):.6f}, max={max(evolved_dnfr):.6f}, mean={np.mean(evolved_dnfr):.6f}")
    
    # Calculate per-node coherence
    coherence_values = []
    for node in G.nodes():
        dnfr = abs(G.nodes[node].get(DNFR_PRIMARY, 0.0))
        node_coherence = 1.0 / (1.0 + dnfr)
        G.nodes[node]['coherence'] = node_coherence
        coherence_values.append(node_coherence)
    
    print(f"Per-node coherence: min={min(coherence_values):.6f}, max={max(coherence_values):.6f}, mean={np.mean(coherence_values):.6f}")
    
    # Test estimate_coherence_length step by step
    print(f"\n=== Coherence Length Calculation Debug ===")
    
    # Step 1: Select seed
    seed_node = max(G.nodes(), key=lambda n: G.nodes[n]['coherence'])
    max_coherence = G.nodes[seed_node]['coherence']
    print(f"Seed node: {seed_node}, max coherence: {max_coherence:.6f}")
    
    # Step 2: BFS layering
    layers = {}
    for n, dist in nx.single_source_shortest_path_length(G, seed_node).items():
        layers.setdefault(dist, []).append(n)
    
    print(f"Number of layers: {len(layers)}")
    for d in sorted(layers.keys())[:5]:  # Show first 5 layers
        print(f"  Layer {d}: {len(layers[d])} nodes")
    
    if len(layers) < 3:
        print("✗ Insufficient layers (< 3), returning 0.0")
        return
    
    # Step 3: Calculate layer coherence means
    d_vals = []
    c_vals = []
    for d, ns in sorted(layers.items()):
        mean_c = sum(G.nodes[x]['coherence'] for x in ns) / len(ns)
        d_vals.append(float(d))
        c_vals.append(mean_c)
        if d < 5:  # Show first few
            print(f"  Distance {d}: mean coherence = {mean_c:.6f}")
    
    # Step 4: Check filtering
    c_arr = np.array(c_vals, dtype=float)
    d_arr = np.array(d_vals, dtype=float)
    mask = c_arr > 1e-12
    valid_points = mask.sum()
    print(f"Valid points for fitting (coherence > 1e-12): {valid_points}")
    
    if valid_points < 3:
        print("✗ Insufficient valid points, returning 0.0")
        return
    
    # Step 5: Try the fit
    c_arr_filtered = c_arr[mask]
    d_arr_filtered = d_arr[mask]
    
    print(f"Filtered coherence range: {c_arr_filtered.min():.6f} to {c_arr_filtered.max():.6f}")
    
    try:
        y = np.log(c_arr_filtered)
        X = np.vstack([np.ones_like(d_arr_filtered), -d_arr_filtered]).T
        coeffs, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)
        a, b = coeffs
        
        print(f"Fit coefficients: a={a:.6f}, b={b:.6f}")
        if len(residuals) > 0:
            print(f"Fit residuals: {residuals[0]:.6f}")
        
        if b <= 0:
            print("✗ Invalid coefficient b <= 0, returning 0.0")
            return
        
        xi = 1.0 / b
        result = float(max(xi, 0.0))
        print(f"✓ Manual calculation xi_c: {result:.6f}")
        
    except np.linalg.LinAlgError as e:
        print(f"✗ Linear algebra error: {e}")
        return
    
    # Now test the actual function
    print(f"\n=== Function Result ===")
    actual_xi_c = estimate_coherence_length(G, coherence_key="coherence")
    print(f"estimate_coherence_length result: {actual_xi_c:.6f}")
    
    # Also check global coherence
    global_coherence = compute_coherence(G)
    print(f"Global coherence: {global_coherence:.6f}")

if __name__ == "__main__":
    simple_debug()