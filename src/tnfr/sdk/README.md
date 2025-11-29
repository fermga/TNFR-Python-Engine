# 🌊 TNFR SDK - Simplified & Powerful API ⭐ **OPTIMIZED**

The TNFR SDK provides an intuitive, production-ready interface for creating,
evolving, and analyzing Resonant Fractal Networks with complete theoretical
fidelity. **90% less code, 100% of the power.**

## 🚀 Quick Start (New API)

```python
from tnfr.sdk import TNFR

# One-line network creation and evolution
results = TNFR.create(20).random(0.3).evolve(5).results()
print(f'Coherence: {results.coherence:.3f}')

# Template-based approach
molecule = TNFR.template('molecule').auto_optimize()
print(molecule.summary())

# Ultra-compact with alias
from tnfr.sdk import T
net = T.create(10).complete().evolve(3)
```

**PHILOSOPHY**: Maximum power, minimum complexity.

## 📚 Core API (Simplified)

### TNFR - Static Factory

Instant network creation with zero boilerplate:

```python
from tnfr.sdk import TNFR

# Topology builders (chainable)
ring = TNFR.create(10).ring()              # Ring topology
star = TNFR.create(15).star()              # Star topology  
random = TNFR.create(20).random(0.3)       # Random connections
complete = TNFR.create(6).complete()       # All-to-all

# Templates for common patterns
molecule = TNFR.template('molecule')       # Molecular structure
small_net = TNFR.template('small')         # 5-node ring
large_net = TNFR.template('large')         # 50-node random

# Evolution and optimization
net.evolve(5)                              # TNFR dynamics
net.auto_optimize()                        # Self-optimization

# Metrics and analysis
result = net.results()                     # All metrics
coherence = net.coherence()                # Just coherence
summary = net.summary()                    # One-line overview
```

### Power User Shortcuts

```python
from tnfr.sdk import T  # Ultra-short alias

# Everything in one line
result = T.create(8).complete().evolve(2).results()

# Quick checks
if net.results().is_coherent():
    print("✅ Network is stable!")

# Comparison
comparison = TNFR.compare(net1, net2, net3)
print(f"Winner: {comparison['best']['name']}")7.5% reduction |
| **Learning Curve** | Steep | Gentle | Intuitive methods |
| **Import Complexity** | Multiple imports | Single import | Simplified |
| **Readability** | Technical | Natural English | Self-documenting |
| **Power** | Full TNFR | Full TNFR | No loss |
| **Performance** | Same | Same | Maintained |

## 🔄 Migration Guide

**OLD WAY (Complex):**
```python
from tnfr.sdk import TNFRNetwork

network = TNFRNetwork("test")
network.add_nodes(20, vf_range=(0.5, 2.0))
network.connect_nodes(0.3, "small_world")
network.apply_sequence("basic_activation", repeat=3)
results = network.measure()
```

**NEW WAY (Simple):**
```python
from tnfr.sdk import TNFR

results = TNFR.create(20).random(0.3).evolve(3).results()
```

**Backward Compatibility**: Old API still works! New code should use TNFR class.
print(f"Density: {network.get_density():.3f}")
print(f"Avg degree: {network.get_average_degree():.2f}")

# Clone network
cloned = network.clone()

# Export data
data = network.export_to_dict()

# Reset
network.reset()
```

### Advanced Examples

```python
# Molecular simulation
molecule = (TNFR.template('molecule')
           .evolve(10)
           .auto_optimize())
           
if molecule.results().is_stable():
    print("Molecule is stable!")

# Social network analysis
social_nets = {
    'family': TNFR.create(6).complete(),
    'friends': TNFR.create(15).ring().random(0.2),
    'community': TNFR.create(50).random(0.1)
}

for name, net in social_nets.items():
    evolved = net.evolve(5)
    print(f"{name}: {evolved.summary()}")

# Network comparison
comparison = TNFR.compare(*social_nets.values())
print(f"Most coherent: {comparison['best']['name']}")
```

### Legacy API (TNFRTemplates)

For backward compatibility, the old API is still available:

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
