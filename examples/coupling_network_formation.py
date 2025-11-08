"""Network Formation via Coupling (UM) - Canonical Usage Examples.

Demonstrates canonical UM (Coupling) operator usage to build coherent networks
through phase synchronization and structural link formation.

This example showcases:
1. network_sync sequence: Uses UM for phase synchronization
2. UM → RA effects: Coupling followed by resonance propagation
3. Complete network formation cycles
4. Phase convergence through repeated coupling

UM (Coupling) synchronizes phases between nodes (θᵢ ≈ θⱼ) to enable
resonant interaction while preserving each node's structural identity (EPI).
This is the foundation for all network-level coherence in TNFR.
"""

import warnings
warnings.filterwarnings("ignore")

from tnfr.sdk import TNFRNetwork, NetworkConfig
import math

print("\n" + "=" * 70)
print("COUPLING (UM) - Network Formation Examples")
print("=" * 70)
print("\nStructural Function: Phase synchronization (θᵢ ≈ θⱼ)")
print("Primary Effect: Creates coherent network links")
print("Invariant: Preserves EPI identity of each node")
print()


def example_1_network_sync():
    """Example 1: Network synchronization with UM."""
    print("\n" + "-" * 70)
    print("Example 1: Network Synchronization (network_sync sequence)")
    print("-" * 70)
    print("\nCanonical Sequence: AL → EN → IL → UM → RA → NAV")
    print("Purpose: Full network activation and synchronization cycle")
    print()
    
    # Create network with varied initial phases
    net = TNFRNetwork("network_sync_demo", NetworkConfig(random_seed=42))
    net.add_nodes(8)
    
    # Initialize with varied phases to show synchronization
    for i, node in enumerate(net.graph.nodes()):
        net.graph.nodes[node]['theta'] = (i % 4) * math.pi / 2  # Four phase groups
    
    print("Initial state:")
    print(f"  Nodes: {net.graph.number_of_nodes()}")
    phases_before = [net.graph.nodes[n]['theta'] for n in net.graph.nodes()]
    print(f"  Phase spread: {max(phases_before) - min(phases_before):.3f} rad")
    print(f"  Phase groups: 4 distinct phases (0, π/2, π, 3π/2)")
    
    results_before = net.measure()
    print(f"  Coherence C(t): {results_before.coherence:.3f}")
    
    # Apply network_sync sequence (includes UM)
    print("\nApplying 'network_sync' sequence (contains UM):")
    print("  AL → EN → IL → UM → RA → NAV")
    net.apply_sequence("network_sync", repeat=2)
    
    # Measure results
    results_after = net.measure()
    phases_after = [net.graph.nodes[n]['theta'] for n in net.graph.nodes()]
    
    print("\nResults:")
    print(f"  Coherence C(t): {results_after.coherence:.3f}")
    print(f"  Phase spread: {max(phases_after) - min(phases_after):.3f} rad")
    coherence_gain = results_after.coherence - results_before.coherence
    phase_reduction = max(phases_before) - min(phases_before) - (max(phases_after) - min(phases_after))
    print(f"  Coherence gain: {coherence_gain:+.3f}")
    print(f"  Phase spread reduction: {phase_reduction:.3f} rad")
    print("\n✓ UM synchronized phases, enabling network coherence")
    
    return net


def example_2_coupling_in_connected_network():
    """Example 2: Coupling effects in pre-connected network."""
    print("\n" + "-" * 70)
    print("Example 2: Coupling in Pre-Connected Network")
    print("-" * 70)
    print("\nDemonstrates: UM's effect on an existing network topology")
    print()
    
    # Create connected network
    net = TNFRNetwork("connected_coupling", NetworkConfig(random_seed=42))
    net.add_nodes(10).connect_nodes(0.3, "random")
    
    # Initialize with three phase clusters
    for i, node in enumerate(net.graph.nodes()):
        net.graph.nodes[node]['theta'] = (i % 3) * (2 * math.pi / 3)
    
    print("Initial state:")
    print(f"  Topology: Random (10 nodes, 30% connectivity)")
    print(f"  Edges: {net.graph.number_of_edges()}")
    print(f"  Phase clusters: 3 groups (0, 2π/3, 4π/3)")
    
    results_before = net.measure()
    print(f"  Coherence C(t): {results_before.coherence:.3f}")
    
    # Apply network_sync to engage coupling
    print("\nApplying network_sync with UM:")
    net.apply_sequence("network_sync", repeat=3)
    
    results_after = net.measure()
    phases_after = [net.graph.nodes[n]['theta'] for n in net.graph.nodes()]
    
    print("\nResults:")
    print(f"  Coherence C(t): {results_after.coherence:.3f}")
    print(f"  Phase convergence: {max(phases_after) - min(phases_after):.3f} rad")
    print(f"  Coherence gain: {results_after.coherence - results_before.coherence:+.3f}")
    print("\n✓ UM bridged phase clusters through network topology")
    
    return net


def example_3_ring_topology_coupling():
    """Example 3: Coupling in ring topology (high structural coherence)."""
    print("\n" + "-" * 70)
    print("Example 3: Ring Topology Coupling")
    print("-" * 70)
    print("\nDemonstrates: UM in highly structured topology")
    print()
    
    # Create ring network
    net = TNFRNetwork("ring_coupling", NetworkConfig(random_seed=42))
    net.add_nodes(12).connect_nodes(0.5, "ring")
    
    # Initialize with linear phase progression
    for i, node in enumerate(net.graph.nodes()):
        net.graph.nodes[node]['theta'] = i * math.pi / 6  # Gradual phase shift
    
    print("Initial state:")
    print(f"  Topology: Ring (12 nodes)")
    print(f"  Edges: {net.graph.number_of_edges()}")
    print(f"  Phase pattern: Linear progression (0 → 11π/6)")
    
    results_before = net.measure()
    print(f"  Coherence C(t): {results_before.coherence:.3f}")
    
    # Apply stabilization (includes coupling-like effects)
    print("\nApplying 'stabilization' sequence:")
    net.apply_sequence("stabilization", repeat=2)
    
    results_after = net.measure()
    
    print("\nResults:")
    print(f"  Coherence C(t): {results_after.coherence:.3f}")
    print(f"  Stability increase: {results_after.coherence - results_before.coherence:+.3f}")
    avg_si = sum(results_after.sense_indices.values()) / len(results_after.sense_indices)
    print(f"  Avg Sense Index: {avg_si:.3f}")
    print("\n✓ Ring topology enabled efficient phase propagation")
    
    return net


def example_4_building_from_isolated_nodes():
    """Example 4: Building coherent network from isolated nodes."""
    print("\n" + "-" * 70)
    print("Example 4: Network Formation from Isolated Nodes")
    print("-" * 70)
    print("\nDemonstrates: Creating coherent structure from scratch")
    print()
    
    # Create isolated nodes
    net = TNFRNetwork("formation", NetworkConfig(random_seed=42))
    net.add_nodes(15)
    
    # Highly varied initial state
    for i, node in enumerate(net.graph.nodes()):
        net.graph.nodes[node]['theta'] = (i % 5) * (2 * math.pi / 5)
    
    print("Initial state (isolated nodes):")
    print(f"  Nodes: {net.graph.number_of_nodes()}")
    print(f"  Edges: {net.graph.number_of_edges()}")
    results_init = net.measure()
    print(f"  Coherence C(t): {results_init.coherence:.3f}")
    
    # Add connections
    print("\nPhase 1: Add random connections")
    net.connect_nodes(0.25, "random")
    print(f"  Edges created: {net.graph.number_of_edges()}")
    
    # Apply formation sequence
    print("\nPhase 2: Apply network_sync sequence (with UM)")
    net.apply_sequence("network_sync", repeat=3)
    results_mid = net.measure()
    print(f"  Coherence after sync: {results_mid.coherence:.3f}")
    
    # Stabilize
    print("\nPhase 3: Consolidate structure")
    net.apply_sequence("consolidation", repeat=2)
    results_final = net.measure()
    
    print("\nFinal state (formed network):")
    print(f"  Coherence C(t): {results_final.coherence:.3f}")
    print(f"  Total gain: {results_final.coherence - results_init.coherence:+.3f}")
    avg_si = sum(results_final.sense_indices.values()) / len(results_final.sense_indices)
    print(f"  Avg Sense Index: {avg_si:.3f}")
    print("\n✓ UM enabled network-level coherence formation")
    
    return net


def example_5_phase_group_merging():
    """Example 5: Merging distinct phase groups through coupling."""
    print("\n" + "-" * 70)
    print("Example 5: Merging Phase Groups")
    print("-" * 70)
    print("\nDemonstrates: UM bridges incompatible phase groups")
    print()
    
    # Create two distinct phase groups
    net = TNFRNetwork("phase_merging", NetworkConfig(random_seed=42))
    net.add_nodes(10)
    
    # Two opposing phase groups
    for i, node in enumerate(net.graph.nodes()):
        if i < 5:
            net.graph.nodes[node]['theta'] = 0.0           # Group 1
        else:
            net.graph.nodes[node]['theta'] = math.pi       # Group 2 (opposite)
    
    print("Initial state:")
    print(f"  Group 1 (nodes 0-4): θ = 0.0 rad")
    print(f"  Group 2 (nodes 5-9): θ = π rad")
    print(f"  Phase difference: π rad (maximal incompatibility)")
    
    # Connect the groups
    net.connect_nodes(0.3, "random")
    results_before = net.measure()
    print(f"  Edges: {net.graph.number_of_edges()}")
    print(f"  Coherence C(t): {results_before.coherence:.3f}")
    
    # Apply repeated synchronization
    print("\nApplying network_sync repeatedly:")
    for i in range(3):
        net.apply_sequence("network_sync")
        phases = [net.graph.nodes[n]['theta'] for n in net.graph.nodes()]
        phase_spread = max(phases) - min(phases)
        results = net.measure()
        print(f"  Iteration {i+1}: phase spread = {phase_spread:.3f} rad, C(t) = {results.coherence:.3f}")
    
    results_final = net.measure()
    phases_final = [net.graph.nodes[n]['theta'] for n in net.graph.nodes()]
    
    print("\nResults:")
    print(f"  Final coherence: {results_final.coherence:.3f}")
    print(f"  Final phase spread: {max(phases_final) - min(phases_final):.3f} rad")
    print(f"  Coherence gain: {results_final.coherence - results_before.coherence:+.3f}")
    print("\n✓ UM successfully merged incompatible phase groups")
    
    return net


def run_all_examples():
    """Run all coupling examples."""
    example_1_network_sync()
    example_2_coupling_in_connected_network()
    example_3_ring_topology_coupling()
    example_4_building_from_isolated_nodes()
    example_5_phase_group_merging()
    
    print("\n" + "=" * 70)
    print("Summary: UM (Coupling) Operator")
    print("=" * 70)
    print("\nKey Principles:")
    print("  1. Phase Synchronization: θᵢ → θⱼ (primary mechanism)")
    print("  2. EPI Preservation: Each node keeps its structural identity")
    print("  3. Network Formation: Creates coherent relational structure")
    print("  4. Bidirectional: Enables mutual influence and shared coherence")
    print()
    print("Canonical Sequences Including UM:")
    print("  • network_sync: AL → EN → IL → UM → RA → NAV")
    print("    Purpose: Full network synchronization cycle")
    print("  • Combined with RA: Coupling + propagation = network coherence")
    print("  • Combined with IL: Coupling + stabilization = consolidated structure")
    print()
    print("Use Cases Demonstrated:")
    print("  • Phase group synchronization (Example 1)")
    print("  • Coupling in existing networks (Example 2)")
    print("  • High-structure topology coupling (Example 3)")
    print("  • Network formation from scratch (Example 4)")
    print("  • Bridging incompatible phases (Example 5)")
    print()
    print("TNFR Context:")
    print("  UM is essential for network-level coherence. Without phase")
    print("  synchronization, nodes cannot resonate (RA) or self-organize")
    print("  (THOL). UM creates the structural substrate for all collective")
    print("  dynamics in TNFR networks.")
    print()
    print("Real-World Applications:")
    print("  • Biomedical: Heart-brain coherence, respiratory coupling")
    print("  • Cognitive: Conceptual integration, memory association")
    print("  • Social: Team synchronization, cultural transmission")
    print("  • Neural: Network formation, synchronized firing patterns")
    print()
    print("=" * 70)
    print("\n✓ All coupling examples completed successfully!")
    print()


if __name__ == "__main__":
    run_all_examples()
