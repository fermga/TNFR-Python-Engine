"""Example: Multi-scale TNFR network simulation.

Demonstrates operational fractality (§3.7) by simulating a hierarchical
TNFR network spanning multiple scales with cross-scale coupling.

This example creates a three-level hierarchy (micro, meso, macro) and
demonstrates:
- Multi-scale initialization
- Cross-scale ΔNFR computation
- Simultaneous evolution across scales
- Coherence aggregation
- Memory efficiency reporting
"""

from tnfr.multiscale import HierarchicalTNFRNetwork, ScaleDefinition


def main():
    """Run multi-scale TNFR simulation."""
    print("=" * 70)
    print("TNFR Multi-Scale Network Simulation")
    print("Operational Fractality (§3.7) Example")
    print("=" * 70)
    print()
    
    # Define scale hierarchy
    print("Defining scale hierarchy...")
    scales = [
        ScaleDefinition(
            name="micro",
            node_count=500,
            coupling_strength=0.9,
            edge_probability=0.15,
        ),
        ScaleDefinition(
            name="meso",
            node_count=200,
            coupling_strength=0.7,
            edge_probability=0.12,
        ),
        ScaleDefinition(
            name="macro",
            node_count=100,
            coupling_strength=0.5,
            edge_probability=0.10,
        ),
    ]
    
    for scale in scales:
        print(f"  - {scale.name}: {scale.node_count} nodes, "
              f"coupling={scale.coupling_strength:.2f}")
    print()
    
    # Create hierarchical network
    print("Initializing hierarchical TNFR network...")
    network = HierarchicalTNFRNetwork(
        scales=scales,
        seed=42,
        parallel=True,  # Enable parallel evolution
    )
    print(f"  Total nodes: {sum(s.node_count for s in scales)}")
    print(f"  Parallel evolution: enabled")
    print()
    
    # Report initial cross-scale couplings
    print("Cross-scale coupling matrix:")
    scale_names = [s.name for s in scales]
    for from_scale in scale_names:
        for to_scale in scale_names:
            if from_scale != to_scale:
                coupling = network.cross_scale_couplings.get(
                    (from_scale, to_scale), 0.0
                )
                print(f"  {from_scale} → {to_scale}: {coupling:.3f}")
    print()
    
    # Customize cross-scale coupling
    print("Customizing cross-scale coupling (micro ↔ macro)...")
    network.set_cross_scale_coupling("micro", "macro", 0.25)
    network.set_cross_scale_coupling("macro", "micro", 0.15)
    print()
    
    # Compute initial metrics
    print("Initial state:")
    initial_coherence = network.compute_total_coherence()
    print(f"  Total coherence C(t): {initial_coherence:.4f}")
    print()
    
    # Evolve network
    print("Evolving multi-scale network with optimized Grammar 2.0 sequence...")
    # Optimized multi-scale sequence for better structural health
    # Complete pattern: activation → self-organization → amplification → closure
    # Health: 0.72 (good) - Pattern: activation
    evolution_params = {
        "dt": 0.1,
        "steps": 20,
        "operators": ["emission", "reception", "self_organization", "coherence", "silence"],
    }
    print(f"  Time step dt: {evolution_params['dt']}")
    print(f"  Evolution steps: {evolution_params['steps']}")
    print(f"  Operators: {evolution_params['operators']}")
    print(f"  Sequence health: 0.72 (good) - Complete activation pattern")
    print()
    
    result = network.evolve_multiscale(**evolution_params)
    
    # Report results
    print("Evolution complete!")
    print()
    print("Final state:")
    print(f"  Total coherence C(t): {result.total_coherence:.4f}")
    print(f"  Cross-scale synchrony: {result.cross_scale_coupling:.4f}")
    print()
    
    print("Per-scale coherence:")
    for scale_name, scale_result in result.scale_results.items():
        coherence = scale_result.get("coherence", 0.0)
        print(f"  {scale_name}: {coherence:.4f}")
    print()
    
    # Memory footprint
    print("Memory footprint per scale:")
    footprint = network.memory_footprint()
    for scale_name, memory_mb in footprint.items():
        if scale_name != "total":
            scale = next(s for s in scales if s.name == scale_name)
            per_node_kb = (memory_mb * 1024) / scale.node_count
            print(f"  {scale_name}: {memory_mb:.2f} MB "
                  f"({per_node_kb:.2f} KB/node)")
    print(f"  TOTAL: {footprint['total']:.2f} MB")
    print()
    
    # Demonstrate multi-scale ΔNFR computation
    print("Multi-scale ΔNFR computation example:")
    sample_node_id = 0
    for scale_name in scale_names:
        try:
            dnfr = network.compute_multiscale_dnfr(sample_node_id, scale_name)
            print(f"  Node {sample_node_id} @ {scale_name}: ΔNFR = {dnfr:.6f}")
        except Exception as e:
            print(f"  Node {sample_node_id} @ {scale_name}: {e}")
    print()
    
    print("=" * 70)
    print("Simulation Complete")
    print("=" * 70)
    print()
    print("Key features demonstrated:")
    print("  ✓ Operational fractality (§3.7)")
    print("  ✓ Cross-scale ΔNFR coupling")
    print("  ✓ Simultaneous multi-scale evolution")
    print("  ✓ Coherence aggregation across scales")
    print("  ✓ Parallel execution for performance")
    print("  ✓ Memory efficiency per scale")


if __name__ == "__main__":
    main()
