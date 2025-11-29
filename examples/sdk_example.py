"""TNFR SDK Example - Modern Fluent API Demonstration

Shows how to use the current TNFR SDK for rapid network creation and analysis.
Demonstrates the fluent API, auto-optimization, and real-time telemetry monitoring.

Perfect for developers who want to use TNFR without deep theoretical knowledge.
"""

import networkx as nx
from tnfr.operators.definitions import Emission, Coherence, Resonance, Silence
from tnfr.operators.grammar import validate_sequence  
from tnfr.physics.fields import compute_coherence, compute_structural_potential
from tnfr.dynamics.self_optimizing_engine import TNFRSelfOptimizingEngine


class TNFRNetwork:
    """Simplified TNFR Network API for rapid development."""
    
    def __init__(self, name="tnfr_network"):
        self.name = name
        self.G = nx.Graph()
        self._measurement_results = {}
        
    def add_nodes(self, count, random_seed=None):
        """Add nodes with TNFR properties."""
        if random_seed:
            import numpy as np
            np.random.seed(random_seed)
            
        for i in range(count):
            self.G.add_node(i, 
                EPI=np.random.uniform(0.3, 0.7),
                nu_f=np.random.uniform(0.6, 1.4), 
                theta=np.random.uniform(0, 2*np.pi)
            )
        return self
        
    def connect_nodes(self, probability, method="random"):
        """Connect nodes with specified probability.""" 
        if method == "random":
            import numpy as np
            nodes = list(self.G.nodes())
            for i in range(len(nodes)):
                for j in range(i+1, len(nodes)):
                    if np.random.random() < probability:
                        self.G.add_edge(nodes[i], nodes[j])
        return self
        
    def apply_sequence(self, sequence_name, repeat=1):
        """Apply a named operator sequence."""
        sequences = {
            "basic_activation": [Emission(), Coherence(), Resonance(), Silence()],
            "stabilization": [Coherence(), Silence()],
            "exploration": [Emission(), Resonance(), Coherence()]
        }
        
        if sequence_name not in sequences:
            raise ValueError(f"Unknown sequence: {sequence_name}")
            
        sequence = sequences[sequence_name]
        
        # Validate sequence
        validation = validate_sequence(sequence)
        if not validation.is_valid:
            raise ValueError(f"Invalid sequence: {validation.error_message}")
            
        # Apply sequence to all nodes
        for _ in range(repeat):
            for node in self.G.nodes():
                for operator in sequence:
                    operator.apply(self.G, node)
                    
        return self
        
    def auto_optimize(self):
        """Apply auto-optimization if available."""
        try:
            engine = TNFRSelfOptimizingEngine(self.G)
            # Apply optimization to a random node
            if self.G.nodes():
                node = next(iter(self.G.nodes()))
                success, metrics = engine.step(node)
                self._measurement_results['auto_optimization'] = {
                    'success': success, 
                    'metrics': metrics
                }
        except Exception as e:
            self._measurement_results['auto_optimization'] = {
                'success': False,
                'error': str(e)
            }
        return self
        
    def measure(self):
        """Measure network properties and return results."""
        coherence = compute_coherence(self.G)
        phi_s = compute_structural_potential(self.G)
        
        results = TNFRResults({
            'name': self.name,
            'nodes': self.G.number_of_nodes(),
            'edges': self.G.number_of_edges(), 
            'coherence': coherence,
            'structural_potential': phi_s,
            'optimization': self._measurement_results.get('auto_optimization', {})
        })
        
        return results


class TNFRResults:
    """Results container with formatted output."""
    
    def __init__(self, data):
        self.data = data
        self.coherence = data['coherence']
        self.nodes = data['nodes']
        self.edges = data['edges']
        
    def summary(self):
        """Return formatted summary string."""
        lines = [
            f"TNFR Network: {self.data['name']}",
            f"  Nodes: {self.nodes}",
            f"  Edges: {self.edges}", 
            f"  Coherence C(t): {self.coherence:.3f}",
            f"  Structural Potential Φ_s: {self.data['structural_potential']:.3f}"
        ]
        
        if self.data['optimization']:
            opt = self.data['optimization']
            if opt.get('success'):
                lines.append("  Auto-optimization: ✅ Applied")
            else:
                lines.append("  Auto-optimization: ❌ Unavailable")
                
        return "\n".join(lines)


def main():
    """Run the complete SDK demonstration."""
    print("🌟 TNFR SDK Example - Modern Fluent API")
    print("="*70)
    print("Demonstrating rapid TNFR development with the simplified SDK")
    print()

    # Example 1: Basic fluent API usage
    print("1️⃣ Basic Fluent API Usage")
    print("-" * 50)
    
    network = TNFRNetwork("sdk_demo_network")
    results = (
        network
        .add_nodes(12, random_seed=42)
        .connect_nodes(0.35, "random")
        .apply_sequence("basic_activation", repeat=2)
        .measure()
    )
    
    print(results.summary())
    print()

    # Example 2: Auto-optimization showcase
    print("2️⃣ Auto-Optimization Showcase")  
    print("-" * 50)
    
    auto_network = TNFRNetwork("auto_optimizing_network")
    auto_results = (
        auto_network
        .add_nodes(8, random_seed=123)
        .connect_nodes(0.4, "random")
        .apply_sequence("exploration", repeat=1)
        .auto_optimize()  # Try to auto-optimize
        .measure()
    )
    
    print(auto_results.summary())
    print()

    # Example 3: Different sequence types
    print("3️⃣ Different Operator Sequences")
    print("-" * 50)
    
    sequences_to_test = ["basic_activation", "stabilization", "exploration"]
    
    for seq_name in sequences_to_test:
        test_network = TNFRNetwork(f"{seq_name}_network")
        test_results = (
            test_network
            .add_nodes(6, random_seed=789)
            .connect_nodes(0.5, "random")
            .apply_sequence(seq_name, repeat=1)
            .measure()
        )
        
        print(f"{seq_name}: C(t)={test_results.coherence:.3f}")
    
    print()
    print("🎯 SDK Features Demonstrated:")
    print("   • Fluent API with method chaining")
    print("   • Pre-validated operator sequences")
    print("   • Automatic grammar validation") 
    print("   • Auto-optimization capabilities")
    print("   • Simplified results reporting")
    print()
    print("🚀 Next Steps:")
    print("   • Create custom sequences for your domain")
    print("   • Integrate with your existing applications")
    print("   • Use structural field telemetry for monitoring")


if __name__ == "__main__":
    import numpy as np
    main()
print(f"  • Neurons: {len(neural_results.sense_indices)}")
print(
    f"  • Avg Sense Index: {sum(neural_results.sense_indices.values()) / len(neural_results.sense_indices):.3f}"
)

# Example 3: Using experiment builders
print("\n3. Using experiment builders:")
print("-" * 70)

print("\nA) Small-World Study:")
sw_results = TNFRExperimentBuilder.small_world_study(nodes=20, steps=5, random_seed=42)
print(f"  • Coherence: {sw_results.coherence:.3f}")
print(f"  • Network size: {len(sw_results.sense_indices)} nodes")

print("\nB) Topology Comparison:")
comparison = TNFRExperimentBuilder.compare_topologies(
    node_count=20, steps=5, random_seed=42
)
for topology, results in comparison.items():
    print(f"  • {topology:12s}: C(t) = {results.coherence:.3f}")

# Example 4: Custom workflow
print("\n4. Custom workflow with method chaining:")
print("-" * 70)

custom_network = TNFRNetwork("custom_workflow")
custom_results = (
    custom_network.add_nodes(12, vf_range=(0.4, 0.8), random_seed=42)
    .connect_nodes(0.4, "ring")
    .apply_sequence("network_sync", repeat=2)
    .apply_sequence("consolidation", repeat=3)
    .measure()
)

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
