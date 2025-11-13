"""Debug: Verify that operators actually change ΔNFR
====================================================

Check if run_sequence() modifies ΔNFR values and if compute_global_coherence()
sees those changes.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import networkx as nx
from tnfr.operators.definitions import Emission, Coherence, Dissonance, Silence
from tnfr.structural import run_sequence
from tnfr.metrics.coherence import compute_global_coherence
from tnfr.constants import DNFR_PRIMARY
from benchmark_utils import create_tnfr_topology, initialize_tnfr_nodes


def main():
    """Test ΔNFR changes after operators."""
    print("="*70)
    print("DEBUGGING ΔNFR CHANGES")
    print("="*70)
    
    # Create small graph
    G = create_tnfr_topology('ring', 6, seed=42)
    initialize_tnfr_nodes(G, nu_f=1.0, epi_range=(0.2, 0.8), seed=42)
    
    # Initial state
    print("\n📊 INITIAL STATE:")
    C_initial = compute_global_coherence(G)
    print(f"  C(t) = {C_initial:.6f}")
    
    dnfr_initial = {}
    for node in G.nodes():
        dnfr = G.nodes[node].get(DNFR_PRIMARY, None)
        print(f"  Node {node}: ΔNFR = {dnfr}")
        dnfr_initial[node] = dnfr
    
    # Apply sequence to all nodes
    sequence = [Emission(), Coherence(), Dissonance(), Coherence(), Silence()]
    print(f"\n🔄 APPLYING SEQUENCE to ALL nodes: {[type(op).__name__ for op in sequence]}")
    
    for target_node in G.nodes():
        run_sequence(G, target_node, sequence)
    
    # Final state
    print("\n📊 FINAL STATE:")
    C_final = compute_global_coherence(G)
    print(f"  C(t) = {C_final:.6f}")
    print(f"  ΔC   = {C_final - C_initial:.6f}")
    
    dnfr_changed = False
    for node in G.nodes():
        dnfr = G.nodes[node].get(DNFR_PRIMARY, None)
        print(f"  Node {node}: ΔNFR = {dnfr} (was {dnfr_initial[node]})")
        if dnfr != dnfr_initial[node]:
            dnfr_changed = True
    
    print(f"\n{'='*70}")
    if dnfr_changed:
        print("✅ ΔNFR values CHANGED after operators")
    else:
        print("❌ ΔNFR values DID NOT CHANGE after operators")
    
    if abs(C_final - C_initial) > 1e-9:
        print(f"✅ C(t) changed: {C_initial:.6f} → {C_final:.6f}")
    else:
        print("❌ C(t) DID NOT CHANGE")
    
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
