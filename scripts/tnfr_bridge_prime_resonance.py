"""
TNFR Bridge: Prime Resonance & Structural Potential
===================================================

This script links the Riemann Hypothesis investigation (Primes) with the
TNFR Engine's Structural Field physics.

It models the Prime Numbers as a TNFR Network and computes the
Canonical Structural Potential (Φ_s) to test for "Structural Stability".

Hypothesis:
    If the Primes form a "Rigid" structure (Hypothesis R), their
    TNFR Structural Potential should be minimized (Passive Equilibrium).

Dependencies:
    - src.tnfr.physics.fields (Canonical Fields)
    - networkx
    - numpy
"""

import sys
import os
import math
import numpy as np
import networkx as nx

# Ensure src is in path
sys.path.append(os.path.join(os.getcwd(), 'src'))

try:
    from tnfr.physics.fields import compute_structural_potential, compute_phase_gradient
except ImportError:
    print("Error: Could not import TNFR Engine. Run from repository root.")
    sys.exit(1)

def get_primes(n):
    """Sieve of Eratosthenes."""
    sieve = [True] * (n + 1)
    for x in range(2, int(n**0.5) + 1):
        if sieve[x]:
            for i in range(x*x, n + 1, x):
                sieve[i] = False
    return [x for x in range(2, n + 1) if sieve[x]]

def build_prime_network(limit=1000):
    """
    Builds a TNFR Graph where nodes are primes.
    
    - Node ID: Prime p
    - EPI: log(p)
    - Phase (phi): gamma_1 * log(p) mod 2pi (Resonance with 1st Zero)
    - Delta NFR: 1/sqrt(p) (The Adelic Weight)
    """
    primes = get_primes(limit)
    G = nx.Graph()
    
    # First Riemann Zero (approx)
    gamma_1 = 14.13472514173469
    
    print(f"Building Prime Network with {len(primes)} nodes...")
    
    for p in primes:
        # TNFR Node Attributes
        # phi: Phase synchronization with the first zero
        phi = (gamma_1 * math.log(p)) % (2 * math.pi)
        
        # Delta NFR: Structural Pressure
        # In Adelic theory, the weight is log(p)/sqrt(p) or 1/sqrt(p)
        # We use 1/sqrt(p) as the "pressure" to reorganize
        dnfr = 1.0 / math.sqrt(p)
        
        G.add_node(p, 
                   EPI=math.log(p),
                   phi=phi,
                   delta_nfr=dnfr,
                   vf=1.0) # Unit frequency
                   
    # Coupling: Connect primes that are "close" in log-space (Adelic metric)
    # This simulates the "interaction" in the Adelic product
    # We connect p_i to p_{i+1} (Linear Chain Topology)
    for i in range(len(primes) - 1):
        p1 = primes[i]
        p2 = primes[i+1]
        G.add_edge(p1, p2, weight=1.0)
        
    return G

def analyze_prime_structure():
    """Computes TNFR Fields for the Prime Network."""
    G = build_prime_network(limit=5000)
    
    # 1. Compute Structural Potential (Phi_s)
    # This measures the global "gravitational" stability of the structure
    phi_s_map = compute_structural_potential(G)
    
    # 2. Compute Phase Gradient (|grad phi|)
    # This measures local desynchronization
    grad_phi_map = compute_phase_gradient(G)
    
    # Aggregates
    avg_phi_s = np.mean(list(phi_s_map.values()))
    max_phi_s = np.max(list(phi_s_map.values()))
    
    avg_grad = np.mean(list(grad_phi_map.values()))
    
    print("\n--- TNFR Structural Analysis of Primes ---")
    print(f"Nodes: {len(G.nodes)}")
    print(f"Topology: Linear Chain (Adelic Ordering)")
    print(f"Phase Source: Gamma_1 = 14.1347...")
    print("-" * 40)
    print(f"Mean Structural Potential (Phi_s): {avg_phi_s:.6f}")
    print(f"Max Structural Potential (Phi_s):  {max_phi_s:.6f}")
    print(f"Mean Phase Gradient (|grad phi|):  {avg_grad:.6f}")
    
    # Interpretation
    print("-" * 40)
    print("INTERPRETATION:")
    if avg_grad < 1.0:
        print("[PASS] Phase Gradient is low. Primes are RESONANT with Gamma_1.")
    else:
        print("[FAIL] Phase Gradient is high. Decoherence detected.")
        
    if max_phi_s < 2.0:
        print("[PASS] Structural Potential < 2.0 (U6 Rule). System is BOUNDED.")
    else:
        print("[FAIL] Structural Potential > 2.0. System is UNSTABLE.")

if __name__ == "__main__":
    analyze_prime_structure()
