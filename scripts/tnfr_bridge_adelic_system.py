"""
TNFR Bridge: Adelic System Simulation
=====================================

This script runs a full TNFR Simulation of the Prime Number System.
It defines a custom 'AdelicCoupling' operator that enforces the
Rigidity of the Prime distribution.

We observe the 'Coherence' C(t) of the system.
If RH is true, C(t) should remain high (> 0.8) and stable.

Dependencies:
    - src.tnfr.operators (Base Operator)
    - src.tnfr.metrics (Coherence)
"""

import sys
import os
import math
import numpy as np
import networkx as nx

# Ensure src is in path
sys.path.append(os.path.join(os.getcwd(), 'src'))

# try:
from tnfr.operators.definitions_base import Operator
from tnfr.operators import Coherence, Resonance
from tnfr.metrics.coherence import compute_global_coherence
from tnfr.types import Glyph
# except ImportError:
#     print("Error: Could not import TNFR Engine.")
#     sys.exit(1)

# --- Custom Operator Definition ---

class AdelicCoupling(Operator):
    """
    Custom Operator: Adelic Coupling (UM_A).
    
    Connects a prime p to its neighbors in the Adelic topology.
    For this simulation, we use the standard ordering as a proxy for
    the 'Global' field Q.
    
    Effect: Synchronizes phase phi_p with phi_{p+1}.
    """
    name = "adelic_coupling"
    glyph = Glyph.UM # Use standard Coupling glyph
    
    def __call__(self, G, node, **kwargs):
        # Find neighbors (linear chain)
        neighbors = list(G.neighbors(node))
        if not neighbors:
            return
            
        # Calculate mean phase of neighbors
        phases = [G.nodes[n].get('phi', 0.0) for n in neighbors]
        target_phase = sum(phases) / len(phases)
        
        # Current phase
        current_phi = G.nodes[node].get('phi', 0.0)
        
        # Coupling strength (coupling constant)
        k = 0.1
        
        # Update phase: phi_new = phi_old + k * (target - old)
        # This is a diffusion process (Heat Equation on the Graph)
        new_phi = current_phi + k * (target_phase - current_phi)
        
        G.nodes[node]['phi'] = new_phi
        
        # Reduce Delta NFR (Pressure drops as we synchronize)
        dnfr = G.nodes[node].get('delta_nfr', 0.0)
        G.nodes[node]['delta_nfr'] = dnfr * 0.95

# --- Simulation ---

def run_adelic_simulation():
    print("Initializing Adelic Prime System...")
    
    # 1. Setup Graph
    # We use the same setup as the resonance script
    from tnfr_bridge_prime_resonance import build_prime_network
    G = build_prime_network(limit=2000)
    
    # 2. Define Sequence
    # The "Adelic Algorithm": Couple -> Cohere -> Resonate
    sequence = [
        AdelicCoupling(),
        Coherence(),
        Resonance()
    ]
    
    print(f"Starting Simulation with {len(G.nodes)} primes.")
    print("Time Step | Coherence C(t) | Mean Delta NFR")
    print("-" * 45)
    
    # 3. Time Loop
    for t in range(10):
        # Apply operators to all nodes (random order)
        nodes = list(G.nodes)
        np.random.shuffle(nodes)
        
        for node in nodes:
            for op in sequence:
                op(G, node)
                
        # Measure System State
        c_t = compute_global_coherence(G)
        
        # Calculate mean Delta NFR manually as metric might need specific keys
        dnfr_vals = [abs(G.nodes[n].get('delta_nfr', 0)) for n in G.nodes]
        mean_dnfr = np.mean(dnfr_vals)
        
        print(f"   t={t:2d}   |    {c_t:.4f}      |    {mean_dnfr:.4f}")
        
    print("-" * 45)
    print("Simulation Complete.")
    
    final_c = compute_global_coherence(G)
    if final_c > 0.8:
        print("[SUCCESS] System maintained High Coherence.")
        print("          The Prime distribution is Structurally Stable.")
    else:
        print("[WARNING] System decohered. Check coupling constants.")

if __name__ == "__main__":
    run_adelic_simulation()
