"""Example demonstrating the TNFR SDK for non-expert users.

This example shows how to use the simplified SDK to create and analyze
TNFR networks without requiring deep knowledge of the theory.
"""

from tnfr.sdk import TNFRNetwork, TNFRTemplates, TNFRExperimentBuilder

# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("TNFR SDK Example: Simplified Network Creation and Analysis")
print("=" * 70)

# Example 1: Basic fluent API usage
print("\n1. Creating a network with the fluent API:")
print("-" * 70)

network = TNFRNetwork("example_network")
results = (network
           .add_nodes(15, random_seed=42)
           .connect_nodes(0.3, "random")
           .apply_sequence("basic_activation", repeat=3)
           .measure())

print(results.summary())

# Example 2: Using pre-built templates
print("\n2. Using domain-specific templates:")
print("-" * 70)

print("\nA) Social Network Simulation:")
social_results = TNFRTemplates.social_network_simulation(
    people=25,
    simulation_steps=10,
    random_seed=42
)
print(f"  • Coherence: {social_results.coherence:.3f}")
print(f"  • Nodes: {len(social_results.sense_indices)}")
print(f"  • Avg Sense Index: {sum(social_results.sense_indices.values()) / len(social_results.sense_indices):.3f}")

print("\nB) Neural Network Model:")
neural_results = TNFRTemplates.neural_network_model(
    neurons=30,
    activation_cycles=15,
    random_seed=42
)
print(f"  • Coherence: {neural_results.coherence:.3f}")
print(f"  • Neurons: {len(neural_results.sense_indices)}")
print(f"  • Avg Sense Index: {sum(neural_results.sense_indices.values()) / len(neural_results.sense_indices):.3f}")

# Example 3: Using experiment builders
print("\n3. Using experiment builders:")
print("-" * 70)

print("\nA) Small-World Study:")
sw_results = TNFRExperimentBuilder.small_world_study(
    nodes=20,
    steps=5,
    random_seed=42
)
print(f"  • Coherence: {sw_results.coherence:.3f}")
print(f"  • Network size: {len(sw_results.sense_indices)} nodes")

print("\nB) Topology Comparison:")
comparison = TNFRExperimentBuilder.compare_topologies(
    node_count=20,
    steps=5,
    random_seed=42
)
for topology, results in comparison.items():
    print(f"  • {topology:12s}: C(t) = {results.coherence:.3f}")

# Example 4: Custom workflow
print("\n4. Custom workflow with method chaining:")
print("-" * 70)

custom_network = TNFRNetwork("custom_workflow")
custom_results = (custom_network
                  .add_nodes(12, vf_range=(0.4, 0.8), random_seed=42)
                  .connect_nodes(0.4, "ring")
                  .apply_sequence("network_sync", repeat=2)
                  .apply_sequence("consolidation", repeat=3)
                  .measure())

print(f"  • Final Coherence: {custom_results.coherence:.3f}")
print(f"  • Avg νf: {custom_results.avg_vf:.3f} Hz_str")
print(f"  • Avg Phase: {custom_results.avg_phase:.3f} rad")

# Example 5: Accessing detailed results
print("\n5. Accessing detailed results:")
print("-" * 70)

data_dict = custom_results.to_dict()
print(f"  • Total nodes: {data_dict['summary_stats']['node_count']}")
print(f"  • Avg ΔNFR: {data_dict['summary_stats']['avg_delta_nfr']:.3f}")

# Show a few node-level metrics
print("\n  Node-level Sense Indices (first 5):")
for i, (node_id, si_value) in enumerate(list(custom_results.sense_indices.items())[:5]):
    print(f"    - {node_id}: Si = {si_value:.3f}")

print("\n" + "=" * 70)
print("Example completed successfully!")
print("=" * 70)
print("\nKey advantages of the SDK:")
print("  ✓ No need to understand TNFR theory in depth")
print("  ✓ Fluent API with method chaining")
print("  ✓ Pre-configured templates for common use cases")
print("  ✓ Automatic validation of TNFR invariants")
print("  ✓ Easy access to coherence metrics")
print("=" * 70)
