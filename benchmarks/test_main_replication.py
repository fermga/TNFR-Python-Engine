#!/usr/bin/env python3
"""
Test exact replication of main experiment ξ_C calculation
"""
import sys
import os
sys.path.insert(0, os.path.abspath('../src'))

import numpy as np
import networkx as nx
from benchmark_utils import create_tnfr_topology

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

def replicate_main_experiment():
    """Replicate exact main experiment code"""
    print("=== Replicating Main Experiment ===\n")
    
    # Exact same setup
    topology = "ws"
    n_nodes = 50
    n_runs = 2  # Just 2 runs for quick test
    intensities = [2.025]  # Slightly above critical point
    I_C = 2.015
    seed_base = 42000
    
    for i, intensity in enumerate(intensities):
        print(f"Intensity {intensity} ({i+1}/{len(intensities)})")
        
        xi_c_runs = []
        
        for run in range(n_runs):
            seed = seed_base + i * 1000 + run
            np.random.seed(seed)
            
            try:
                print(f"    Run {run+1}/{n_runs}...")
                
                # Create fresh topology EXACTLY like main experiment
                G = create_tnfr_topology(topology, n_nodes, seed)
                
                # Create network-wide perturbation proportional to intensity 
                # EXACTLY like main experiment
                for node in G.nodes():
                    delta_I = intensity - I_C  # Distance from critical point
                    base_dnfr = delta_I * np.random.normal(0, 0.5)
                    G.nodes[node][DNFR_PRIMARY] = base_dnfr
                
                print(f"      Initial DNFR sample: {[G.nodes[n][DNFR_PRIMARY] for n in list(G.nodes())[:3]]}")
                
                # Apply multiple dissonance operators for network evolution
                perturb_factor = 0.2 * n_nodes * abs(intensity - I_C)
                n_perturbed = max(1, int(perturb_factor))
                node_list = list(G.nodes())
                perturbed_nodes = np.random.choice(
                    node_list,
                    size=min(n_perturbed, len(node_list)),
                    replace=False
                )
                
                print(f"      Perturbing {len(perturbed_nodes)} nodes")
                
                dissonance = Dissonance()
                for node in perturbed_nodes:
                    dissonance(G, node)
                
                # Let system evolve for several steps to develop correlations
                for step in range(10):  # Multiple evolution steps
                    # default_compute_delta_nfr(G)  # SKIP THIS
                    # Apply coherence to stabilize some nodes
                    coherence_op = Coherence()
                    node_list = list(G.nodes())
                    stabilized_nodes = np.random.choice(
                        node_list, size=max(1, n_nodes//4), replace=False
                    )
                    for node in stabilized_nodes:
                        coherence_op(G, node)
                
                # Final ΔNFR calculation after evolution
                # default_compute_delta_nfr(G)  # SKIP THIS
                
                print(f"      Final DNFR sample: {[G.nodes[n].get(DNFR_PRIMARY, 0) for n in list(G.nodes())[:3]]}")
                
                # Add per-node coherence for coherence length calculation
                # EXACTLY like main experiment
                for node in G.nodes():
                    dnfr = abs(G.nodes[node].get(DNFR_PRIMARY, 0.0))
                    G.nodes[node]['coherence'] = 1.0 / (1.0 + dnfr)
                
                coherence_sample = [G.nodes[n]['coherence'] for n in list(G.nodes())[:3]]
                print(f"      Coherence sample: {coherence_sample}")
                
                # Measure ξ_C coherence length EXACTLY like main experiment
                xi_c = estimate_coherence_length(G, coherence_key="coherence")
                print(f"      ξ_C result: {xi_c}")
                xi_c_runs.append(xi_c)
                
            except Exception as e:
                print(f"      ❌ Exception: {e}")
                import traceback
                traceback.print_exc()
                xi_c_runs.append(0.0)
                
        print(f"    Results for intensity {intensity}: {xi_c_runs}")
        mean_xi_c = np.mean(xi_c_runs)
        print(f"    Mean ξ_C: {mean_xi_c}")

if __name__ == "__main__":
    replicate_main_experiment()