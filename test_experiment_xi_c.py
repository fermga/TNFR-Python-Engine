#!/usr/bin/env python3
"""
Quick test to reproduce the multi-topology experiment ξ_C = 0.0 issue
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

# Critical point constant
I_C = 2.015

def create_test_topology(topology_type: str, n_nodes: int = 50, seed: int = 42) -> nx.Graph:
    """Create test topology similar to multi_topology experiment."""
    np.random.seed(seed)
    
    if topology_type == "ws":
        return nx.watts_strogatz_graph(n_nodes, 4, 0.3, seed=seed)
    elif topology_type == "scale_free":
        return nx.barabasi_albert_graph(n_nodes, 2, seed=seed)
    elif topology_type == "grid":
        # Create grid-like structure
        side = int(np.sqrt(n_nodes))
        return nx.grid_2d_graph(side, side)
    else:
        raise ValueError(f"Unknown topology: {topology_type}")

def init_tnfr_attributes(G):
    """Initialize TNFR attributes for all nodes."""
    for node in G.nodes():
        G.nodes[node]['EPI'] = np.array([0.5])  # Initial EPI
        G.nodes[node]['vf'] = 1.0  # Structural frequency
        G.nodes[node]['theta'] = np.random.uniform(0, 2*np.pi)  # Phase
        G.nodes[node][DNFR_PRIMARY] = 0.0  # Initialize ΔNFR

def simulate_like_experiment(topology: str, intensity: float = 2.5):
    """Simulate exactly like multi_topology_critical_exponent.py"""
    
    n_nodes = 50
    G = create_test_topology(topology, n_nodes, seed=42)
    
    # Initialize TNFR attributes
    init_tnfr_attributes(G)
    
    print(f"\n=== Testing {topology} topology, intensity={intensity:.2f} ===")
    
    # Create network-wide perturbation proportional to intensity (EXACTLY like experiment)
    for node in G.nodes():
        delta_I = intensity - I_C  # Distance from critical point
        base_dnfr = delta_I * np.random.normal(0, 0.5)
        G.nodes[node][DNFR_PRIMARY] = base_dnfr
    
    print(f"Initial ΔNFR (from intensity): mean={np.mean([G.nodes[n][DNFR_PRIMARY] for n in G.nodes()]):.6f}")
    
    # Apply multiple dissonance operators (EXACTLY like experiment)
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
    
    print(f"After dissonance to {len(perturbed_nodes)} nodes")
    
    # Let system evolve (EXACTLY like experiment) 
    for step in range(10):
        default_compute_delta_nfr(G)
        
        # Apply coherence to stabilize some nodes
        coherence_op = Coherence()
        stabilized_nodes = np.random.choice(
            node_list, size=max(1, n_nodes//4), replace=False
        )
        for node in stabilized_nodes:
            coherence_op(G, node)
    
    # Final ΔNFR calculation after evolution
    default_compute_delta_nfr(G)
    
    dnfr_values = [abs(G.nodes[node].get(DNFR_PRIMARY, 0.0)) for node in G.nodes()]
    print(f"Final ΔNFR: min={min(dnfr_values):.6f}, max={max(dnfr_values):.6f}, mean={np.mean(dnfr_values):.6f}")
    
    # Add per-node coherence (EXACTLY like experiment)
    for node in G.nodes():
        dnfr = abs(G.nodes[node].get(DNFR_PRIMARY, 0.0))
        G.nodes[node]['coherence'] = 1.0 / (1.0 + dnfr)
    
    coherence_values = [G.nodes[node]['coherence'] for node in G.nodes()]
    print(f"Per-node coherence: min={min(coherence_values):.6f}, max={max(coherence_values):.6f}, mean={np.mean(coherence_values):.6f}")
    
    # Test BFS layering details
    seed_node = max(G.nodes(), key=lambda n: G.nodes[n]['coherence'])
    layers = {}
    for n, dist in nx.single_source_shortest_path_length(G, seed_node).items():
        layers.setdefault(dist, []).append(n)
    
    print(f"Seed: {seed_node} (coherence={G.nodes[seed_node]['coherence']:.6f})")
    print(f"BFS layers: {len(layers)}")
    
    if len(layers) < 3:
        print("✗ < 3 layers")
        return 0.0
    
    # Check layer coherence variation
    for d in sorted(layers.keys())[:3]:
        mean_c = sum(G.nodes[x]['coherence'] for x in layers[d]) / len(layers[d])
        print(f"  Layer {d}: {len(layers[d])} nodes, mean coherence = {mean_c:.6f}")
    
    # Measure ξ_C
    xi_c = estimate_coherence_length(G, coherence_key="coherence")
    print(f"ξ_C result: {xi_c:.6f}")
    
    return xi_c

if __name__ == "__main__":
    # Test with different intensities and topologies  
    for topology in ["ws", "scale_free"]:
        for intensity in [1.5, 2.0, 2.5, 3.0]:
            result = simulate_like_experiment(topology, intensity)