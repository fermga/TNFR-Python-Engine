#!/usr/bin/env python3
"""Quick debugging of xi_c calculation in multi-topology experiment context"""

import numpy as np
import networkx as nx
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from tnfr.physics.fields import estimate_coherence_length
from tnfr.operators.definitions import Dissonance, Coherence
from tnfr.dynamics.dnfr import default_compute_delta_nfr
from tnfr.constants import DNFR_PRIMARY

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

def test_xi_c_like_experiment():
    """Test xi_c calculation exactly like the experiment"""
    print("Testing xi_c calculation like multi-topology experiment...")
    
    # Parameters matching the experiment
    I_C = 2.015
    intensity = 2.0  # Near critical
    n_nodes = 50
    topology = "ws"
    seed = 42
    
    # Create and setup network exactly like experiment
    G = create_test_topology(topology, n_nodes, seed)
    
    print(f"Created {topology} network with {len(G.nodes)} nodes")
    
    # Apply perturbation like experiment
    for node in G.nodes():
        delta_I = intensity - I_C
        base_dnfr = delta_I * np.random.normal(0, 0.5)
        G.nodes[node][DNFR_PRIMARY] = base_dnfr
        print(f"Node {node}: initial DNFR = {base_dnfr:.3f}")
    
    # Apply dissonance like experiment
    perturb_factor = 0.2 * n_nodes * abs(intensity - I_C)
    n_perturbed = max(1, int(perturb_factor))
    
    print(f"Perturbation factor: {perturb_factor:.3f}, n_perturbed: {n_perturbed}")
    
    node_list = list(G.nodes())
    perturbed_nodes = np.random.choice(
        node_list,
        size=min(n_perturbed, len(node_list)),
        replace=False
    )
    
    dissonance = Dissonance()
    for node in perturbed_nodes:
        print(f"Applying dissonance to node {node}")
        dissonance(G, node)
    
    # Evolution like experiment
    print("Evolving network...")
    for step in range(10):
        default_compute_delta_nfr(G)
        
        # Apply coherence to stabilize some nodes
        coherence_op = Coherence()
        stabilized_nodes = np.random.choice(
            node_list, size=max(1, n_nodes//4), replace=False
        )
        for node in stabilized_nodes:
            coherence_op(G, node)
    
    # Final ΔNFR calculation
    default_compute_delta_nfr(G)
    
    # Check DNFR values before coherence calculation
    print("\nFinal DNFR values (sample):")
    for i, node in enumerate(list(G.nodes())[:5]):
        dnfr = G.nodes[node].get(DNFR_PRIMARY, 0.0)
        print(f"Node {node}: DNFR = {dnfr:.6f}")
    
    # Add per-node coherence like experiment
    for node in G.nodes():
        dnfr = abs(G.nodes[node].get(DNFR_PRIMARY, 0.0))
        G.nodes[node]['coherence'] = 1.0 / (1.0 + dnfr)
    
    # Check coherence values
    print("\nCoherence values (sample):")
    for i, node in enumerate(list(G.nodes())[:5]):
        coh = G.nodes[node]['coherence']
        print(f"Node {node}: coherence = {coh:.6f}")
    
    # Test estimate_coherence_length
    print(f"\n=== Testing estimate_coherence_length ===")
    xi_c = estimate_coherence_length(G, coherence_key="coherence")
    print(f"Estimated coherence length: {xi_c}")
    print(f"Type: {type(xi_c)}")
    
    return xi_c

if __name__ == "__main__":
    result = test_xi_c_like_experiment()
    print(f"\nFinal result: {result}")