# TNFR SDK - Simplified API for Non-Expert Users

The TNFR SDK provides a high-level, user-friendly interface for creating and simulating Resonant Fractal Networks while maintaining full theoretical fidelity to TNFR principles.

## Quick Start

```python
from tnfr.sdk import TNFRNetwork

# Create a network with fluent API
results = (TNFRNetwork("my_network")
           .add_nodes(20)
           .connect_nodes(0.3, "random")
           .apply_sequence("basic_activation", repeat=5)
           .measure())

print(results.summary())
```

## Core Components

### TNFRNetwork - Fluent API

Chainable interface for network creation and evolution:

```python
from tnfr.sdk import TNFRNetwork

network = TNFRNetwork("experiment")

# Add nodes with random TNFR properties
network.add_nodes(15, vf_range=(0.3, 0.8), random_seed=42)

# Connect with different topologies
network.connect_nodes(0.3, "random")      # Erdős-Rényi random
network.connect_nodes(pattern="ring")      # Ring lattice
network.connect_nodes(0.1, "small_world")  # Watts-Strogatz

# Apply operator sequences
network.apply_sequence("basic_activation", repeat=3)

# Measure and analyze
results = network.measure()
```

**Convenience Methods:**
```python
# Network statistics
print(f"Nodes: {network.get_node_count()}")
print(f"Edges: {network.get_edge_count()}")
print(f"Density: {network.get_density():.3f}")
print(f"Avg degree: {network.get_average_degree():.2f}")

# Clone network
cloned = network.clone()

# Export data
data = network.export_to_dict()

# Reset
network.reset()
```

### TNFRTemplates - Domain-Specific Patterns

Pre-configured templates for common use cases:

```python
from tnfr.sdk import TNFRTemplates

# Social network dynamics
social_results = TNFRTemplates.social_network_simulation(
    people=50,
    connections_per_person=6,
    simulation_steps=20,
    random_seed=42
)

# Neural network modeling
neural_results = TNFRTemplates.neural_network_model(
    neurons=100,
    connectivity=0.15,
    activation_cycles=30
)

# Ecosystem dynamics
ecosystem_results = TNFRTemplates.ecosystem_dynamics(
    species=25,
    evolution_steps=50
)

# Creative process modeling
creative_results = TNFRTemplates.creative_process_model(
    ideas=15,
    development_cycles=12
)

# Organizational networks
org_results = TNFRTemplates.organizational_network(
    agents=40,
    coordination_steps=25
)
```

### TNFRExperimentBuilder - Research Patterns

Builder patterns for standard experiments:

```python
from tnfr.sdk import TNFRExperimentBuilder

# Small-world network study
sw_results = TNFRExperimentBuilder.small_world_study(
    nodes=50,
    rewiring_prob=0.1,
    steps=10
)

# Synchronization analysis
sync_results = TNFRExperimentBuilder.synchronization_study(
    nodes=30,
    coupling_strength=0.5,
    steps=20
)

# Topology comparison
comparison = TNFRExperimentBuilder.compare_topologies(
    node_count=40,
    steps=10
)
for topology, results in comparison.items():
    print(f"{topology}: C(t) = {results.coherence:.3f}")

# Phase transition study
transition = TNFRExperimentBuilder.phase_transition_study(
    nodes=50,
    coupling_levels=5
)

# Resilience testing
resilience = TNFRExperimentBuilder.resilience_study(
    nodes=40,
    initial_steps=10,
    perturbation_steps=5,
    recovery_steps=10
)
```

## Utility Functions

### Analysis and Comparison

```python
from tnfr.sdk import (
    compare_networks,
    compute_network_statistics,
    format_comparison_table,
)

# Create multiple networks
results1 = TNFRNetwork("net1").add_nodes(20).connect_nodes(0.3).measure()
results2 = TNFRNetwork("net2").add_nodes(20).connect_nodes(0.5).measure()

# Compare
comparison = compare_networks({"net1": results1, "net2": results2})
print(format_comparison_table(comparison))

# Extended statistics
stats = compute_network_statistics(results1)
print(f"Coherence: {stats['coherence']:.3f}")
print(f"Avg Si: {stats['avg_si']:.3f} ± {stats['std_si']:.3f}")
print(f"Range: [{stats['min_si']:.3f}, {stats['max_si']:.3f}]")
```

### JSON Export/Import

```python
from tnfr.sdk import export_to_json, import_from_json

# Export network
network = TNFRNetwork("test").add_nodes(10).connect_nodes(0.3)
export_to_json(network, "network.json")

# Import data
data = import_from_json("network.json")
print(f"Loaded: {data['name']} with {data['metadata']['nodes']} nodes")
```

### Goal-Based Sequence Suggestions

```python
from tnfr.sdk import suggest_sequence_for_goal

# Get recommendations
seq, desc = suggest_sequence_for_goal("stabilize")
print(f"Goal: stabilize")
print(f"Sequence: {seq}")
print(f"Description: {desc}")

# Use directly
network = TNFRNetwork().add_nodes(15).connect_nodes(0.3)
network.apply_sequence(seq, repeat=5)
```

## Predefined Operator Sequences

All sequences follow TNFR grammar rules and maintain canonical invariants:

- **`basic_activation`**: `[emission, reception, coherence, resonance, silence]`
  - Initiates network with fundamental operators
  
- **`stabilization`**: `[emission, reception, coherence, resonance, recursivity]`
  - Establishes and maintains coherent structure
  
- **`creative_mutation`**: `[emission, dissonance, reception, coherence, mutation, resonance, silence]`
  - Generates variation through controlled mutation
  
- **`network_sync`**: `[emission, reception, coherence, coupling, resonance, silence]`
  - Synchronizes nodes through coupling
  
- **`exploration`**: `[emission, dissonance, reception, coherence, resonance, transition]`
  - Explores phase space with transitions
  
- **`consolidation`**: `[recursivity, reception, coherence, resonance, silence]`
  - Consolidates structure with recursive coherence

## Network Results

The `NetworkResults` dataclass provides structured access to metrics:

```python
results = network.measure()

# Direct access
print(f"Coherence: {results.coherence}")
print(f"Avg νf: {results.avg_vf} Hz_str")
print(f"Avg Phase: {results.avg_phase} rad")

# Node-level metrics
for node_id, si in results.sense_indices.items():
    print(f"{node_id}: Si = {si:.3f}")

# Convert to dict
data = results.to_dict()

# Human-readable summary
print(results.summary())
```

## TNFR Compliance

All SDK components maintain full TNFR theoretical fidelity:

- **Structural Invariants**: Preserved through validated operator sequences
- **Frequency Bounds**: All νf values ≤ 1.0 Hz_str (structural hertz)
- **Operator Grammar**: Sequences follow canonical TNFR rules
- **Metric Exposure**: C(t), Si, νf, phase exposed without abstraction loss
- **Nodal Equation**: ∂EPI/∂t = νf · ΔNFR(t) respected in all operations

## Type Safety

Type stubs (`.pyi` files) are provided for better IDE support:

```python
from tnfr.sdk import TNFRNetwork, NetworkResults

# Full type hints and autocomplete
network: TNFRNetwork = TNFRNetwork("typed")
results: NetworkResults = network.add_nodes(10).measure()
```

## Examples

See `examples/sdk_example.py` for comprehensive usage demonstrations.

## Testing

All SDK components are thoroughly tested:

```bash
pytest tests/unit/sdk/ -v
```

## Documentation

For detailed documentation on TNFR theory and canonical implementation:
- See `AGENTS.md` for TNFR fundamentals
- See `tnfr.pdf` for complete theoretical framework
- See `ARCHITECTURE.md` for system architecture and SDK integration

## Contributing

When extending the SDK, maintain TNFR canonicity:
1. Validate all operator sequences against TNFR grammar
2. Respect structural frequency bounds (νf ≤ 1.0 Hz_str)
3. Preserve nodal equation semantics
4. Expose canonical metrics without abstraction
5. Add tests for new functionality

## License

See repository LICENSE file.
