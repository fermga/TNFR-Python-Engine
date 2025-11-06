# TNFR Quick Start Guide

**Get started with TNFR in 5 minutes!** 🚀

This guide will help you create your first TNFR network and understand the basics. No deep theoretical knowledge required - we'll explain as we go.

## What is TNFR?

**TNFR (Teoría de la Naturaleza Fractal Resonante)** is a paradigm for modeling systems as networks of resonating nodes. Think of it like this:

- 🎵 **Nodes** are like musical notes that vibrate at different frequencies
- 🔗 **Connections** allow nodes to influence each other
- 🎶 **Operators** are actions that change how nodes interact
- 📊 **Coherence** measures how well the network stays organized

**Key concepts** (don't worry, we'll explain these):
- **EPI**: The "shape" or state of a node
- **νf (nu-f)**: How fast a node vibrates (in Hz_str units)
- **C(t)**: Network coherence (stability measure)
- **Si**: Sense index (how well a node can adapt)

## Installation

```bash
pip install tnfr
```

That's it! You're ready to go.

## Your First TNFR Network (3 lines!)

```python
from tnfr.sdk import TNFRNetwork

# Create, connect, and measure a network
network = TNFRNetwork("my_first_network")
network.add_nodes(10).connect_nodes(0.3, "random")
results = network.apply_sequence("basic_activation", repeat=3).measure()

print(results.summary())
```

**Output:**
```
Network Results Summary
-----------------------
Coherence C(t): 0.XXX
Average Sense Index: 0.XXX
Number of nodes: 10
...
```

🎉 **Congratulations!** You just created, activated, and measured a TNFR network!

## What Just Happened?

Let's break down those 3 lines:

### Line 1: Create a Network
```python
network = TNFRNetwork("my_first_network")
```
- Creates an empty network with a name
- Like creating a new canvas for your system

### Line 2: Add Nodes and Connect
```python
network.add_nodes(10).connect_nodes(0.3, "random")
```
- `add_nodes(10)`: Creates 10 resonating nodes
- Each node gets a random frequency (νf) and phase (φ)
- `connect_nodes(0.3, "random")`: Connects nodes with 30% probability
- Think of it like neurons forming synapses

### Line 3: Apply Operators and Measure
```python
results = network.apply_sequence("basic_activation", repeat=3).measure()
```
- `apply_sequence("basic_activation", repeat=3)`: Activates the network 3 times
- This runs: emission → reception → coherence → resonance → silence
- `measure()`: Captures the final state and calculates metrics

## Understanding the Results

The `results` object contains:

```python
results.coherence        # C(t) - how stable is the network? (0-1)
results.sense_indices    # Si for each node - adaptability measure
results.delta_nfr       # ΔNFR - reorganization for each node
results.avg_vf          # Average frequency across nodes
results.avg_phase       # Average phase angle
```

**Interpreting coherence:**
- `C(t) > 0.6` → Highly coherent (very stable)
- `C(t) = 0.3-0.6` → Moderately coherent (some stability)
- `C(t) < 0.3` → Low coherence (loosely organized)

## Interactive Tutorial

Want a guided tour? Run the interactive tutorial:

```python
from tnfr.tutorials import hello_tnfr

hello_tnfr()
```

This 5-minute interactive tutorial explains concepts as you go!

## Pre-Built Operator Sequences

TNFR provides ready-to-use sequences for common patterns:

```python
# Stabilize a network
network.apply_sequence("stabilization", repeat=5)

# Introduce creative change
network.apply_sequence("creative_mutation", repeat=3)

# Synchronize network
network.apply_sequence("network_sync", repeat=10)

# Explore new states
network.apply_sequence("exploration", repeat=5)

# Consolidate structure
network.apply_sequence("consolidation", repeat=7)
```

## Domain-Specific Examples

### Biology: Cell Communication

```python
from tnfr.tutorials import biological_example

results = biological_example()
print(f"Tissue coherence: {results['coherence']:.3f}")
```

Models how cells coordinate through signaling.

### Sociology: Social Dynamics

```python
from tnfr.tutorials import social_network_example

results = social_network_example()
print(f"Group cohesion: {results['coherence']:.3f}")
```

Models how people reach consensus in groups.

### Technology: Distributed Systems

```python
from tnfr.tutorials import technology_example

results = technology_example()
print(f"System reliability: {results['coherence']:.3f}")
```

Models microservice coordination and resilience.

## Method Chaining (Fluent API)

Build complex workflows with clean, readable code:

```python
results = (
    TNFRNetwork("experiment")
    .add_nodes(20, vf_range=(0.4, 0.8))
    .connect_nodes(0.5, "ring")
    .apply_sequence("network_sync", repeat=5)
    .apply_sequence("consolidation", repeat=3)
    .measure()
)

print(f"Final coherence: {results.coherence:.3f}")
```

## Accessing Detailed Data

```python
# Get full results dictionary
data = results.to_dict()

# Iterate over node metrics
for node_id, si_value in results.sense_indices.items():
    print(f"Node {node_id}: Si = {si_value:.3f}")

# Access the underlying NetworkX graph
graph = results.graph
print(f"Number of edges: {graph.number_of_edges()}")
```

## Common Patterns

### Pattern 1: Quick Experiment

```python
from tnfr.sdk import TNFRNetwork

network = TNFRNetwork("quick_test")
results = (
    network
    .add_nodes(15)
    .connect_nodes(0.3, "random")
    .apply_sequence("basic_activation", repeat=5)
    .measure()
)
print(results.summary())
```

### Pattern 2: Multiple Configurations

```python
for connectivity in [0.1, 0.3, 0.5, 0.7]:
    network = TNFRNetwork(f"connectivity_{connectivity}")
    results = (
        network
        .add_nodes(20)
        .connect_nodes(connectivity, "random")
        .apply_sequence("network_sync", repeat=10)
        .measure()
    )
    print(f"Connectivity {connectivity}: C(t) = {results.coherence:.3f}")
```

### Pattern 3: Custom Frequency Range

```python
# Create nodes with specific frequency range
network = TNFRNetwork("high_frequency")
network.add_nodes(
    25,
    vf_range=(0.7, 1.5),  # Higher frequencies
    random_seed=42        # Reproducible
)
network.connect_nodes(0.4, "random")
results = network.apply_sequence("stabilization", repeat=10).measure()
```

## The 13 TNFR Operators

When you're ready to dive deeper, here are all the operators:

1. **emission** - Start broadcasting signals
2. **reception** - Receive signals from neighbors
3. **coherence** - Stabilize structure
4. **dissonance** - Introduce reorganization
5. **coupling** - Create dependencies between nodes
6. **resonance** - Amplify synchronized patterns
7. **silence** - Pause evolution
8. **expansion** - Grow network influence
9. **contraction** - Reduce network influence
10. **self_organization** - Let structure emerge
11. **mutation** - Change node phase/state
12. **transition** - Move to new structural state
13. **recursivity** - Apply structure to itself

## Error Messages with Hints

TNFR provides helpful error messages:

```python
from tnfr.errors import OperatorSequenceError

try:
    network.apply_sequence("emision")  # Typo!
except OperatorSequenceError as e:
    print(e)
```

Output:
```
======================================================================
TNFR Error: Invalid operator sequence: 'emision' cannot be applied
======================================================================

💡 Suggestion: Did you mean one of: emission, recursivity? Use one of the 13 canonical operators...

📊 Context:
   • invalid_operator: emision
   • sequence_so_far: empty
   • operator_count: 0

📚 Documentation: https://github.com/fermga/Teoria-de-la-naturaleza-fractal-resonante-TNFR-/blob/main/docs/source/api/operators.md
======================================================================
```

## Next Steps

### Learn More
- 📖 [Foundations](../foundations.md) - TNFR theory explained
- 🔧 [API Reference](../api/overview.md) - Detailed API documentation
- 📚 [Examples](../examples/README.md) - More code examples
- 🎯 [Operator Guide](../api/operators.md) - Deep dive into operators

### Explore Tutorials
```python
from tnfr.tutorials import (
    hello_tnfr,              # 5-minute introduction
    biological_example,      # Cell communication
    social_network_example,  # Social dynamics
    technology_example,      # Distributed systems
    run_all_tutorials        # Run all tutorials
)

# Try them all!
run_all_tutorials()
```

### Advanced Topics
- [Backend Performance](./math-backends.md) - GPU acceleration with JAX/Torch
- [Optional Dependencies](./optional-dependencies.md) - Caching and optimization
- [Telemetry](../api/telemetry.md) - Metrics and tracing

## Getting Help

**Found a bug or need help?**
- GitHub Issues: https://github.com/fermga/Teoria-de-la-naturaleza-fractal-resonante-TNFR-/issues
- Documentation: https://tnfr.readthedocs.io/
- Examples: Check the `examples/` directory

**Want to contribute?**
- See [CONTRIBUTING.md](../../../CONTRIBUTING.md) for guidelines
- Read [AGENTS.md](../../../AGENTS.md) for TNFR invariants

## Summary

You've learned:
- ✅ How to install TNFR
- ✅ How to create your first network (3 lines!)
- ✅ What the key concepts mean (EPI, νf, C(t), Si)
- ✅ How to use operator sequences
- ✅ How to interpret results
- ✅ Where to find domain examples (bio, social, tech)
- ✅ How to access detailed data

**Next:** Try the interactive tutorials or explore domain-specific examples!

```python
from tnfr.tutorials import hello_tnfr
hello_tnfr()  # Start learning interactively!
```

---

*This quick start guide gets you running in minutes while maintaining full TNFR theoretical fidelity. As you advance, you'll discover the deep structural principles that make TNFR a powerful paradigm for modeling complex systems.*
