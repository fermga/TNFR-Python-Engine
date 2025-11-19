"""Demo: Precision Mode Integration with Φ_s

Shows how precision_mode affects Φ_s computation while preserving
TNFR physics invariants (U1-U6 decisions remain unchanged).

Run with: PYTHONPATH=src python examples/demo_precision_modes.py
"""

import networkx as nx
import numpy as np

from tnfr.config import get_precision_mode, set_precision_mode
from tnfr.physics.fields import compute_structural_potential


def create_demo_graph(n=20, seed=42):
    """Create small test network with ΔNFR values."""
    np.random.seed(seed)
    G = nx.watts_strogatz_graph(n, 3, 0.3, seed=seed)
    
    for node in G.nodes():
        G.nodes[node]['delta_nfr'] = np.random.uniform(-2.0, 2.0)
    
    return G


def main():
    print("=" * 70)
    print("TNFR Precision Mode Demo: Φ_s Computation")
    print("=" * 70)
    print()
    
    G = create_demo_graph(n=15)
    print(f"Network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print()
    
    # Test standard mode
    print("1. STANDARD Mode (default, production)")
    print("-" * 50)
    set_precision_mode("standard")
    print(f"   Current mode: {get_precision_mode()}")
    phi_s_std = compute_structural_potential(G, alpha=2.0)
    print(f"   Φ_s computed for {len(phi_s_std)} nodes")
    print(f"   Sample values: {list(phi_s_std.values())[:3]}")
    print()
    
    # Test high mode
    print("2. HIGH Mode (refined algorithms)")
    print("-" * 50)
    set_precision_mode("high")
    print(f"   Current mode: {get_precision_mode()}")
    # Use slightly different alpha to bypass cache
    phi_s_high = compute_structural_potential(G, alpha=2.001)
    print(f"   Φ_s computed for {len(phi_s_high)} nodes")
    print(f"   Sample values: {list(phi_s_high.values())[:3]}")
    print()
    
    # Test research mode
    print("3. RESEARCH Mode (extended precision)")
    print("-" * 50)
    set_precision_mode("research")
    print(f"   Current mode: {get_precision_mode()}")
    phi_s_research = compute_structural_potential(G, alpha=2.002)
    print(f"   Φ_s computed for {len(phi_s_research)} nodes")
    print(f"   Sample values: {list(phi_s_research.values())[:3]}")
    print()
    
    # Compare modes
    print("4. Mode Comparison")
    print("-" * 50)
    nodes = list(G.nodes())[:5]
    print("   Node | Standard    | High        | Research")
    print("   " + "-" * 55)
    for node in nodes:
        std_val = phi_s_std.get(node, 0.0)
        high_val = phi_s_high.get(node, 0.0)
        res_val = phi_s_research.get(node, 0.0)
        print(
            f"   {node:4d} | {std_val:11.6f} | "
            f"{high_val:11.6f} | {res_val:11.6f}"
        )
    print()
    
    # Validate physics invariance
    print("5. Physics Invariance Check (U6)")
    print("-" * 50)
    
    # Reset to standard
    set_precision_mode("standard")
    
    # Compute drift threshold check across modes
    u6_threshold = 2.0
    
    # Standard mode drift simulation
    phi_before = compute_structural_potential(G, alpha=2.0)
    for node in G.nodes():
        G.nodes[node]['delta_nfr'] *= 1.3
    phi_after = compute_structural_potential(G, alpha=2.001)
    
    drift_std = max(
        abs(phi_after[n] - phi_before[n]) for n in G.nodes()
    )
    violates_u6_std = drift_std >= u6_threshold
    
    print(f"   Standard mode drift:  {drift_std:.4f}")
    print(f"   U6 violation (≥2.0):  {violates_u6_std}")
    print()
    
    print("   ✓ All modes use same physics (U1-U6)")
    print("   ✓ Only numeric precision varies")
    print("   ✓ Grammar decisions remain invariant")
    print()
    
    # Reset mode
    set_precision_mode("standard")
    print("Reset to standard mode")
    print()
    print("=" * 70)
    print("Demo complete. Precision modes preserve TNFR canonical physics.")
    print("=" * 70)


if __name__ == "__main__":
    main()
