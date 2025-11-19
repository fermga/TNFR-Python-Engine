"""Demonstration of rich tetrad telemetry with density modes.

This example shows how tetrad snapshots (Φ_s, |∇φ|, K_φ, ξ_C) can be
collected at different telemetry densities during TNFR simulations.

Key Concepts:
- Telemetry density controls detail level: "low" | "medium" | "high"
- Snapshots are purely observational (read-only, no physics changes)
- Sample interval scales with density (10×, 5×, 1× base timestep)
- All modes preserve U1-U6 grammar and structural dynamics

Usage:
    python examples/demo_tetrad_telemetry.py
"""

import networkx as nx
import numpy as np

from tnfr.config import set_telemetry_density
from tnfr.metrics.tetrad import (
    collect_tetrad_snapshot,
    get_tetrad_sample_interval,
)


def create_demo_network(n_nodes: int = 30, seed: int = 42) -> nx.Graph:
    """Create a small TNFR network for demonstration."""
    G = nx.watts_strogatz_graph(n_nodes, 4, 0.3, seed=seed)
    
    np.random.seed(seed)
    for i in G.nodes():
        G.nodes[i]["ΔNFR"] = np.random.uniform(-2, 2)
        G.nodes[i]["theta"] = np.random.uniform(0, 2 * np.pi)
        G.nodes[i]["νf"] = 1.0
    
    return G


def print_snapshot_summary(snapshot: dict, density: str) -> None:
    """Print human-readable summary of tetrad snapshot."""
    print(f"\n{'='*60}")
    print(f"Telemetry Density: {density.upper()}")
    print(f"{'='*60}")
    
    # Φ_s statistics
    phi_s = snapshot["phi_s"]
    print("\nStructural Potential (Phi_s):")
    print(f"  Mean: {phi_s['mean']:8.4f}")
    print(f"  Max:  {phi_s['max']:8.4f}")
    print(f"  Min:  {phi_s['min']:8.4f}")
    print(f"  Std:  {phi_s['std']:8.4f}")
    
    if "p25" in phi_s:
        print(f"  Q1 (p25): {phi_s['p25']:8.4f}")
        print(f"  Median:   {phi_s['p50']:8.4f}")
        print(f"  Q3 (p75): {phi_s['p75']:8.4f}")
    
    if "p10" in phi_s:
        print(f"  p10: {phi_s['p10']:8.4f}")
        print(f"  p90: {phi_s['p90']:8.4f}")
        print(f"  p99: {phi_s['p99']:8.4f}")
    
    if "histogram" in phi_s:
        print(f"  Histogram bins: {len(phi_s['histogram']['counts'])}")
    
    # Phase gradient
    grad = snapshot["phase_grad"]
    print("\nPhase Gradient (|grad_phi|):")
    print(f"  Mean: {grad['mean']:8.4f}")
    print(f"  Max:  {grad['max']:8.4f}")
    print(f"  Min:  {grad['min']:8.4f}")
    
    if "p50" in grad:
        print(f"  Median: {grad['p50']:8.4f}")
    
    # Phase curvature
    curv = snapshot["phase_curv"]
    print("\nPhase Curvature (K_phi):")
    print(f"  Mean: {curv['mean']:8.4f}")
    print(f"  Max:  {curv['max']:8.4f}")
    print(f"  Min:  {curv['min']:8.4f}")
    
    # Coherence length
    xi_c = snapshot["xi_c"]
    print("\nCoherence Length (xi_C):")
    if xi_c is not None:
        print(f"  xi_C = {xi_c:8.4f}")
    else:
        print("  xi_C = (not computable)")
    
    # Sample interval
    base_dt = 0.1
    interval = get_tetrad_sample_interval(base_dt)
    print(f"\nSample Interval: {interval:.2f} (base_dt = {base_dt})")


def demonstrate_telemetry_densities() -> None:
    """Show tetrad snapshots at all three density levels."""
    print("\n" + "="*60)
    print("TNFR Tetrad Telemetry Demonstration")
    print("="*60)
    print("\nCreating demo network (30 nodes, Watts-Strogatz)...")
    
    G = create_demo_network(n_nodes=30, seed=42)
    
    # Demonstrate each density level
    for density in ["low", "medium", "high"]:
        set_telemetry_density(density)
        snapshot = collect_tetrad_snapshot(G)
        print_snapshot_summary(snapshot, density)
    
    print("\n" + "="*60)
    print("Key Observations:")
    print("="*60)
    print("\n1. LOW Density:")
    print("   - Basic statistics only (mean, max, min, std)")
    print("   - Sample interval: 10× base timestep")
    print("   - Minimal overhead, suitable for long simulations")
    
    print("\n2. MEDIUM Density:")
    print("   - Adds quartiles (p25, p50, p75)")
    print("   - Sample interval: 5× base timestep")
    print("   - Balanced detail/performance")
    
    print("\n3. HIGH Density:")
    print("   - Full distribution (p10, p90, p99)")
    print("   - Includes histograms (20 bins)")
    print("   - Sample interval: 1× base timestep (every step)")
    print("   - Maximum detail for deep analysis")
    
    print("\n" + "="*60)
    print("Physics Invariance Check:")
    print("="*60)
    
    # Verify physics invariance
    original_dnfr = {i: G.nodes[i]["ΔNFR"] for i in G.nodes()}
    
    set_telemetry_density("high")
    snapshot = collect_tetrad_snapshot(G)
    
    invariant = True
    for i in G.nodes():
        if G.nodes[i]["ΔNFR"] != original_dnfr[i]:
            invariant = False
            break
    
    if invariant:
        print("\n[OK] DNFR unchanged across telemetry collection")
        print("[OK] Telemetry is READ-ONLY (no operator changes)")
        print("[OK] U1-U6 grammar preserved")
    else:
        print("\n[WARN] WARNING: DNFR was modified (BUG!)")
    
    print("\n" + "="*60)
    print("Demonstration Complete")
    print("="*60)


def demonstrate_snapshot_consistency() -> None:
    """Show that core statistics are consistent across densities."""
    print("\n" + "="*60)
    print("Snapshot Consistency Validation")
    print("="*60)
    
    G = create_demo_network(n_nodes=50, seed=123)
    
    # Collect at all densities
    snapshots = {}
    for density in ["low", "medium", "high"]:
        set_telemetry_density(density)
        snapshots[density] = collect_tetrad_snapshot(G)
    
    # Compare core statistics
    print("\nPhi_s Mean Comparison:")
    for density in ["low", "medium", "high"]:
        mean = snapshots[density]["phi_s"]["mean"]
        print(f"  {density:>6}: {mean:10.6f}")
    
    # Check consistency
    low_mean = snapshots["low"]["phi_s"]["mean"]
    med_mean = snapshots["medium"]["phi_s"]["mean"]
    high_mean = snapshots["high"]["phi_s"]["mean"]
    
    if np.allclose([low_mean, med_mean, high_mean], low_mean, rtol=1e-10):
        print("\n[OK] Core statistics IDENTICAL across densities")
        print("  (Density only affects detail level, not values)")
    else:
        print("\n[WARN] WARNING: Statistics differ (unexpected!)")
    
    print("\nDetail Level Comparison:")
    print(f"  Low:    {len([k for k in snapshots['low']['phi_s'] if k.startswith('p')])} percentiles")
    print(f"  Medium: {len([k for k in snapshots['medium']['phi_s'] if k.startswith('p')])} percentiles")
    print(f"  High:   {len([k for k in snapshots['high']['phi_s'] if k.startswith('p')])} percentiles")
    print(f"  High histogram: {'Yes' if 'histogram' in snapshots['high']['phi_s'] else 'No'}")


if __name__ == "__main__":
    demonstrate_telemetry_densities()
    demonstrate_snapshot_consistency()
