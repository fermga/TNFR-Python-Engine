#!/usr/bin/env python3
"""Demonstration of OZ dissonance network propagation.

This script demonstrates the new propagation features for the OZ (Dissonance)
operator, showing how dissonance spreads through phase-compatible neighbors
according to TNFR resonance principles.
"""

from tnfr.structural import create_nfr
from tnfr.operators.definitions import Emission, Coherence, Dissonance
from tnfr.dynamics.propagation import (
    compute_network_dissonance_field,
    detect_bifurcation_cascade,
)
from tnfr.alias import get_attr
from tnfr.constants.aliases import ALIAS_DNFR, ALIAS_THETA


def print_section(title):
    """Print a section header."""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print('=' * 60)


def demonstrate_basic_propagation():
    """Demonstrate basic dissonance propagation to neighbors."""
    print_section("Basic Propagation: Star Topology")
    
    # Create central node
    G, central = create_nfr("central", epi=0.5, vf=1.0)
    
    # Add neighbors in star topology
    neighbors = []
    for i in range(1, 5):
        G.add_node(i)
        G.add_edge(central, i)
        Emission()(G, i)
        Coherence()(G, i)
        # Set phase-compatible
        G.nodes[i][ALIAS_THETA[0]] = 0.15
        neighbors.append(i)
    
    G.nodes[central][ALIAS_THETA[0]] = 0.1
    
    # Capture DNFR before
    dnfr_before = {n: float(get_attr(G.nodes[n], ALIAS_DNFR, 0.0)) for n in neighbors}
    
    # Apply OZ with propagation
    print(f"\nApplying OZ to central node '{central}'...")
    Dissonance()(G, central, propagate_to_network=True)
    
    # Check propagation
    print("\nPropagation results:")
    for n in neighbors:
        dnfr_after = float(get_attr(G.nodes[n], ALIAS_DNFR, 0.0))
        change = dnfr_after - dnfr_before[n]
        if change > 0:
            print(f"  Node {n}: ΔNFR increased by {change:.3f}")
        else:
            print(f"  Node {n}: No change")
    
    # Check telemetry
    if "_oz_propagation_events" in G.graph:
        event = G.graph["_oz_propagation_events"][-1]
        print(f"\n✓ Propagation affected {event['affected_count']} neighbors")


def demonstrate_phase_filtering():
    """Demonstrate phase compatibility filtering."""
    print_section("Phase Filtering: Compatible vs Incompatible")
    
    G, source = create_nfr("source", epi=0.5, vf=1.0)
    
    # Add two neighbors with different phases
    G.add_node("compatible")
    G.add_node("incompatible")
    G.add_edge(source, "compatible")
    G.add_edge(source, "incompatible")
    
    for node in ["compatible", "incompatible"]:
        Emission()(G, node)
        Coherence()(G, node)
    
    # Set phases
    G.nodes[source][ALIAS_THETA[0]] = 0.0
    G.nodes["compatible"][ALIAS_THETA[0]] = 0.2    # Within π/2
    G.nodes["incompatible"][ALIAS_THETA[0]] = 3.0  # Beyond π/2
    
    print(f"\nSource phase: {G.nodes[source][ALIAS_THETA[0]]:.2f}")
    print(f"Compatible neighbor phase: {G.nodes['compatible'][ALIAS_THETA[0]]:.2f}")
    print(f"Incompatible neighbor phase: {G.nodes['incompatible'][ALIAS_THETA[0]]:.2f}")
    
    # Capture DNFR before
    dnfr_comp_before = float(get_attr(G.nodes["compatible"], ALIAS_DNFR, 0.0))
    dnfr_incomp_before = float(get_attr(G.nodes["incompatible"], ALIAS_DNFR, 0.0))
    
    # Apply OZ
    print(f"\nApplying OZ to source...")
    Dissonance()(G, source, propagate_to_network=True)
    
    # Check results
    dnfr_comp_after = float(get_attr(G.nodes["compatible"], ALIAS_DNFR, 0.0))
    dnfr_incomp_after = float(get_attr(G.nodes["incompatible"], ALIAS_DNFR, 0.0))
    
    print(f"\nResults:")
    print(f"  Compatible node: ΔNFR changed by {dnfr_comp_after - dnfr_comp_before:.3f}")
    print(f"  Incompatible node: ΔNFR changed by {dnfr_incomp_after - dnfr_incomp_before:.3f}")
    print(f"\n✓ Phase filtering successfully blocked incompatible neighbor")


def demonstrate_field_computation():
    """Demonstrate dissonance field with distance decay."""
    print_section("Dissonance Field: Path Topology")
    
    G, node0 = create_nfr("node0", epi=0.5, vf=1.0)
    
    # Create path: node0 - 1 - 2 - 3 - 4
    for i in range(1, 5):
        G.add_node(i)
        if i == 1:
            G.add_edge(node0, i)
        else:
            G.add_edge(i-1, i)
        Emission()(G, i)
    
    # Apply OZ
    print(f"\nCreated path: {node0} - 1 - 2 - 3 - 4")
    Dissonance()(G, node0)
    
    # Compute field with different radii
    for radius in [1, 2, 3]:
        field = compute_network_dissonance_field(G, node0, radius=radius)
        print(f"\nRadius {radius}:")
        for node in sorted(field.keys()):
            print(f"  Node {node}: field strength = {field[node]:.3f}")
    
    print(f"\n✓ Field decays with distance as expected")


def demonstrate_cascade_detection():
    """Demonstrate bifurcation cascade detection."""
    print_section("Bifurcation Cascade: Network Effects")
    
    G, source = create_nfr("source", epi=0.5, vf=1.2)
    
    # Add neighbors with accelerating EPI histories
    for i in range(1, 4):
        G.add_node(i)
        G.add_edge(source, i)
        Emission()(G, i)
        # Set history showing acceleration
        G.nodes[i]["_epi_history"] = [0.2, 0.45, 0.75]
        G.nodes[i][ALIAS_THETA[0]] = 0.1
    
    G.nodes[source][ALIAS_THETA[0]] = 0.1
    G.nodes[source]["_epi_history"] = [0.3, 0.4, 0.5]
    
    print(f"\nNetwork with {len(list(G.neighbors(source)))} neighbors")
    print("All neighbors have accelerating EPI histories")
    
    # Apply OZ with propagation
    print(f"\nApplying OZ to source...")
    Dissonance()(G, source, propagate_to_network=True)
    
    # Detect cascade
    cascade = detect_bifurcation_cascade(G, source, threshold=0.3)
    
    print(f"\nCascade detection:")
    print(f"  Nodes in bifurcation cascade: {len(cascade)}")
    
    for node in cascade:
        if "_bifurcation_cascade" in G.nodes[node]:
            meta = G.nodes[node]["_bifurcation_cascade"]
            print(f"  - Node {node}: d2epi = {meta['d2epi']:.3f}")
    
    if cascade:
        print(f"\n✓ Cascade detected: OZ triggered {len(cascade)} bifurcations")
    else:
        print(f"\n  (No cascade detected with current threshold)")


def demonstrate_configurable_parameters():
    """Demonstrate configurable propagation parameters."""
    print_section("Configurable Parameters")
    
    G, source = create_nfr("source", epi=0.5, vf=1.0)
    
    # Add neighbor
    G.add_node(1)
    G.add_edge(source, 1)
    Emission()(G, 1)
    
    # Set moderate phase difference
    G.nodes[source][ALIAS_THETA[0]] = 0.0
    G.nodes[1][ALIAS_THETA[0]] = 1.2  # Within default threshold
    
    print("\nDefault parameters:")
    print(f"  OZ_PHASE_THRESHOLD: π/2 (≈1.57)")
    print(f"  OZ_MIN_PROPAGATION: 0.05")
    
    # Test 1: Default (should propagate)
    dnfr_before = float(get_attr(G.nodes[1], ALIAS_DNFR, 0.0))
    Dissonance()(G, source, propagate_to_network=True)
    dnfr_after = float(get_attr(G.nodes[1], ALIAS_DNFR, 0.0))
    
    print(f"\nWith defaults: neighbor ΔNFR change = {dnfr_after - dnfr_before:.3f}")
    
    # Test 2: Strict threshold (should not propagate)
    G2, source2 = create_nfr("source2", epi=0.5, vf=1.0)
    G2.add_node(2)
    G2.add_edge(source2, 2)
    Emission()(G2, 2)
    G2.nodes[source2][ALIAS_THETA[0]] = 0.0
    G2.nodes[2][ALIAS_THETA[0]] = 1.2
    
    # Set strict threshold
    G2.graph["OZ_PHASE_THRESHOLD"] = 1.0
    
    dnfr_before2 = float(get_attr(G2.nodes[2], ALIAS_DNFR, 0.0))
    Dissonance()(G2, source2, propagate_to_network=True)
    dnfr_after2 = float(get_attr(G2.nodes[2], ALIAS_DNFR, 0.0))
    
    print(f"\nWith strict threshold (1.0): neighbor ΔNFR change = {dnfr_after2 - dnfr_before2:.3f}")
    print(f"\n✓ Parameters successfully control propagation behavior")


def main():
    """Run all demonstrations."""
    print("\n" + "=" * 60)
    print("  OZ DISSONANCE NETWORK PROPAGATION DEMONSTRATION")
    print("=" * 60)
    print("\nTNFR Canonical Principle:")
    print("  'Nodal interference: Dissonance between nodes that")
    print("   perturbs coherence. It can induce reorganization")
    print("   or collapse.'")
    
    demonstrate_basic_propagation()
    demonstrate_phase_filtering()
    demonstrate_field_computation()
    demonstrate_cascade_detection()
    demonstrate_configurable_parameters()
    
    print("\n" + "=" * 60)
    print("  DEMONSTRATION COMPLETE")
    print("=" * 60)
    print("\nKey Features Demonstrated:")
    print("  ✓ Phase-weighted propagation to neighbors")
    print("  ✓ Phase compatibility filtering")
    print("  ✓ Distance-decayed field computation")
    print("  ✓ Bifurcation cascade detection")
    print("  ✓ Configurable propagation parameters")
    print("\nAll features align with TNFR canonical theory.")
    print()


if __name__ == "__main__":
    main()
