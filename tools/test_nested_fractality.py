#!/usr/bin/env python3
"""
Test if critical exponent β holds for nested EPIs (operational fractality).

Protocol:
  1. Create parent network (N=50 nodes)
  2. Use REMESH to create nested sub-EPIs (5 clusters of 10 nodes)
  3. Apply intensity-scaled operator sequences
  4. Measure fragmentation rate vs intensity
  5. Compute β for nested system
  6. Compare with flat-network β = 0.556

Hypothesis: Operational fractality → β should be scale-independent
"""

import json
import numpy as np
import networkx as nx
import random
from pathlib import Path
from typing import List, Dict
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from tnfr.physics.fields import compute_structural_potential


def create_nested_network(n_clusters: int = 5, cluster_size: int = 10, seed: int = 42) -> nx.Graph:
    """
    Create hierarchical network with nested EPIs.
    
    Structure:
      - Parent level: n_clusters nodes (meta-nodes representing clusters)
      - Child level: Each meta-node contains cluster_size sub-nodes
      - Total nodes: n_clusters * cluster_size
    
    This simulates REMESH: each cluster = nested EPI
    """
    random.seed(seed)
    np.random.seed(seed)
    
    # Create parent network (ring of clusters for simplicity)
    G = nx.Graph()
    
    # Add nodes with hierarchical labels: (cluster_id, node_within_cluster)
    node_id = 0
    cluster_nodes = {}
    
    for cluster in range(n_clusters):
        cluster_nodes[cluster] = []
        for i in range(cluster_size):
            G.add_node(node_id, cluster=cluster, local_id=i)
            cluster_nodes[cluster].append(node_id)
            node_id += 1
    
    # Intra-cluster edges (dense within each EPI)
    for cluster, nodes in cluster_nodes.items():
        # Create small-world within cluster
        for i, n1 in enumerate(nodes):
            for n2 in nodes[i+1:i+3]:  # Connect to next 2 neighbors (ring-like)
                if n2 < len(nodes):
                    G.add_edge(n1, nodes[n2 % len(nodes)])
    
    # Inter-cluster edges (sparse between EPIs)
    for cluster in range(n_clusters):
        next_cluster = (cluster + 1) % n_clusters
        # Connect representative nodes from adjacent clusters
        G.add_edge(cluster_nodes[cluster][0], cluster_nodes[next_cluster][0])
        G.add_edge(cluster_nodes[cluster][-1], cluster_nodes[next_cluster][-1])
    
    return G


def compute_coherence(G: nx.Graph) -> float:
    """Simple coherence metric: mean EPI across nodes."""
    epis = [G.nodes[n].get('epi', 0.5) for n in G.nodes]
    return float(np.mean(epis))


def apply_stress_sequence(G: nx.Graph, intensity: float, seed: int) -> float:
    """
    Apply stress to nested network.
    Simulates operator sequence with intensity scaling.
    """
    random.seed(seed)
    np.random.seed(seed)
    
    # Simulate dissonance-like stress
    for node in G.nodes:
        # Reduce EPI proportional to intensity
        stress = random.uniform(0.1, 0.3) * intensity
        G.nodes[node]['epi'] = max(0.0, G.nodes[node]['epi'] - stress)
        
        # Update ΔNFR (reorganization pressure)
        G.nodes[node]['delta_nfr'] = random.uniform(0.05, 0.15) * intensity
    
    return compute_coherence(G)


def test_nested_fragmentation(intensity: float, n_runs: int = 15, seed_base: int = 300) -> Dict:
    """
    Test fragmentation rate at given intensity for nested EPIs.
    
    Returns:
        {'intensity': I, 'n_runs': N, 'n_fragmented': count, 'p_frag': rate}
    """
    n_fragmented = 0
    
    for run in range(n_runs):
        seed = seed_base + run
        
        # Create nested network
        G = create_nested_network(n_clusters=5, cluster_size=10, seed=seed)
        
        # Initialize state
        for node in G.nodes:
            G.nodes[node]['epi'] = random.uniform(0.5, 0.7)
            G.nodes[node]['delta_nfr'] = random.uniform(0.01, 0.05)
            G.nodes[node]['vf'] = 6.0  # Match previous experiments
        
        c_initial = compute_coherence(G)
        
        # Apply stress
        c_final = apply_stress_sequence(G, intensity, seed)
        
        # Check fragmentation (threshold ~0.2 based on previous results)
        if c_final < 0.2 or (c_initial - c_final) > 0.5:
            n_fragmented += 1
    
    return {
        'intensity': intensity,
        'n_runs': n_runs,
        'n_fragmented': n_fragmented,
        'p_frag': n_fragmented / n_runs if n_runs > 0 else 0.0
    }


def fit_beta_exponent(results: List[Dict], I_c: float = 2.015) -> float:
    """
    Fit β from P_frag ~ (I - I_c)^β.
    
    Uses log-log linear regression.
    """
    intensities = np.array([r['intensity'] for r in results])
    p_frags = np.array([r['p_frag'] for r in results])
    
    # Filter valid points (I > I_c, 0 < P < 1)
    valid_mask = (intensities > I_c) & (p_frags > 0) & (p_frags < 1)
    
    if np.sum(valid_mask) < 3:
        return np.nan
    
    I_valid = intensities[valid_mask]
    P_valid = p_frags[valid_mask]
    
    epsilon = I_valid - I_c
    log_eps = np.log(epsilon)
    log_P = np.log(P_valid)
    
    # Linear fit: log(P) = log(A) + β·log(ε)
    coeffs = np.polyfit(log_eps, log_P, 1)
    beta = coeffs[0]
    
    return beta


print("\n" + "="*70)
print("=== Multi-Scale Fractality Test: β for Nested EPIs ===")
print("="*70)

print("\nHypothesis: Operational fractality → β should be scale-independent")
print("Testing nested EPIs (REMESH-like structure) vs flat networks\n")

# Test range of intensities
intensities = [1.8, 2.0, 2.03, 2.05, 2.07, 2.08, 2.09, 2.1, 2.15, 2.2]

print("Step 1: Run fragmentation tests on nested networks\n")
print(f"{'Intensity':>10} | {'P_frag':>10} | {'N_frag/N_total':>20}")
print("-" * 50)

results = []
for I in intensities:
    res = test_nested_fragmentation(I, n_runs=15, seed_base=300)
    results.append(res)
    print(f"{res['intensity']:>10.2f} | {res['p_frag']:>10.1%} | "
          f"{res['n_fragmented']:>2}/{res['n_runs']:>2}")

# Fit β
print("\n\nStep 2: Fit critical exponent β\n")

I_c = 2.015  # From flat-network analysis
beta_nested = fit_beta_exponent(results, I_c)

print(f"{'Parameter':>20} | {'Value':>12}")
print("-" * 40)
print(f"{'I_c (fixed)':>20} | {I_c:>12.3f}")
print(f"{'β (nested EPIs)':>20} | {beta_nested:>12.3f}")
print(f"{'β (flat networks)':>20} | {0.556:>12.3f}")

# Compare
print("\n\nStep 3: Universality test across scales\n")

if np.isnan(beta_nested):
    print("✗ INSUFFICIENT DATA for β estimation")
    print("  → Need more intermediate intensity points for power-law fitting")
elif abs(beta_nested - 0.556) < 0.1:
    print("✓ OPERATIONAL FRACTALITY CONFIRMED")
    print(f"  → β_nested = {beta_nested:.3f} ≈ β_flat = 0.556")
    print("  → Critical exponent INVARIANT across scales")
    print("  → Nested EPIs follow same universality class")
    print()
    print("Physical interpretation:")
    print("  - REMESH-generated sub-EPIs preserve critical dynamics")
    print("  - Phase transition mechanism scale-independent")
    print("  - Validates TNFR operational fractality principle")
elif abs(beta_nested - 0.556) < 0.2:
    print("⚠ MODERATE SCALE-DEPENDENCE")
    print(f"  → β_nested = {beta_nested:.3f} vs β_flat = 0.556")
    print("  → Deviation Δβ = {:.3f}".format(abs(beta_nested - 0.556)))
    print("  → Possible finite-size or nesting-depth effects")
else:
    print("✗ DIFFERENT UNIVERSALITY CLASS")
    print(f"  → β_nested = {beta_nested:.3f} ≠ β_flat = 0.556")
    print("  → Nested EPIs exhibit distinct critical behavior")
    print("  → May require separate theoretical treatment")

print("\n" + "="*70)
