# Getting Started with TNFR 🚀

This guide will get you up and running with the TNFR Python Engine in minutes.

## Installation

### Using pip (Recommended)

```bash
pip install tnfr
```

**Requirements**: Python ≥ 3.9

### Optional Dependencies

For enhanced performance and features:

```bash
# GPU acceleration with JAX
pip install tnfr[jax]

# PyTorch backend support
pip install tnfr[torch]

# Visualization tools
pip install tnfr[viz]

# All optional features
pip install tnfr[all]
```

### From Source

For development or to get the latest features:

```bash
git clone https://github.com/fermga/TNFR-Python-Engine.git
cd TNFR-Python-Engine
pip install -e ".[dev]"
```

## Quick Start

### Your First TNFR Network (3 Lines!)

```python
from tnfr.sdk import TNFRNetwork

# Create, activate, and measure a network
network = TNFRNetwork("hello_world")
results = (network
    .add_nodes(10)
    .connect_nodes(0.3, "random")
    .apply_sequence("basic_activation", repeat=3)
    .measure())

print(results.summary())
```

**Output:**
```
=== TNFR Network Results ===
Network: hello_world
Nodes: 10
Edges: 13
Coherence C(t): 0.782
Sense Index Si: 0.654
Status: STABLE
```

### What Just Happened?

Let's break down the code:

1. **`add_nodes(10)`**: Created 10 Resonant Fractal Nodes (NFRs)
   - Each node has an EPI (structural form), νf (frequency), and φ (phase)

2. **`connect_nodes(0.3, "random")`**: Connected nodes randomly
   - 30% connection probability
   - Creates couplings based on phase synchrony

3. **`apply_sequence("basic_activation", repeat=3)`**: Applied operator sequence
   - Emission → Coherence → Resonance
   - Repeated 3 times to stabilize the network

4. **`measure()`**: Calculated network metrics
   - C(t): Total coherence (pattern stability)
   - Si: Sense index (reorganization stability)

## Interactive Learning (5 Minutes)

TNFR includes guided tutorials to learn the concepts:

```python
from tnfr.tutorials import hello_tnfr

# Guided tour of TNFR fundamentals
hello_tnfr()
```

This interactive tutorial will:
- ✅ Explain core TNFR concepts
- ✅ Show you how to create and manipulate networks
- ✅ Demonstrate structural operators in action
- ✅ Teach you to interpret coherence metrics

## Core Workflow

Every TNFR application follows this pattern:

```python
from tnfr.sdk import TNFRNetwork

# 1. Initialize network
network = TNFRNetwork("my_network")

# 2. Build structure
network.add_nodes(20, epi_range=(0.8, 1.2))
network.connect_nodes(0.4, topology="scale_free")

# 3. Apply operators
network.apply_operator("emission", target_nodes=[0, 1, 2])
network.apply_operator("coherence")
network.apply_operator("resonance")

# 4. Measure & analyze
results = network.measure()
print(f"Coherence: {results.coherence:.3f}")
print(f"Sense Index: {results.sense_index:.3f}")

# 5. Visualize (optional)
network.visualize(show_metrics=True)
```

## Domain-Specific Examples

### Biological Network

Model cellular communication:

```python
from tnfr.sdk import TNFRNetwork

# Create a cellular network
cells = TNFRNetwork("cell_network")

results = (cells
    .add_nodes(20, epi_range=(0.8, 1.2))  # Biological variation
    .connect_nodes(0.3, "scale_free")      # Power-law connectivity
    .apply_sequence("therapeutic", repeat=5)  # Healing pattern
    .measure())

print(f"Network health: {results.coherence:.2%}")
```

### Social Network

Model information spreading:

```python
from tnfr.sdk import TNFRNetwork

# Create a social network
community = TNFRNetwork("social_community")

results = (community
    .add_nodes(50)
    .connect_nodes(0.25, "small_world")   # Small-world topology
    .apply_operator("emission", target_nodes=[0])  # Seed information
    .apply_sequence("propagation", repeat=10)  # Spread through network
    .measure())

print(f"Information reach: {results.active_nodes}/{results.total_nodes}")
```

### Technology/IoT Network

Model distributed system coordination:

```python
from tnfr.sdk import TNFRNetwork

# Create IoT sensor network
sensors = TNFRNetwork("iot_sensors")

results = (sensors
    .add_nodes(30)
    .connect_nodes(0.35, "lattice")        # Grid topology
    .apply_sequence("synchronization", repeat=8)  # Coordinate sensors
    .measure())

print(f"Sync quality: {results.sense_index:.3f}")
```

## Understanding Results

### Coherence Metrics

After measuring a network, you get these key metrics:

```python
results = network.measure()

# Total Coherence C(t) [0, 1]
print(f"C(t): {results.coherence:.3f}")
# > 0.7: Strong coherence (stable patterns)
# 0.3-0.7: Moderate coherence (evolving)
# < 0.3: Weak coherence (fragmentation risk)

# Sense Index Si [0, 1+]
print(f"Si: {results.sense_index:.3f}")
# > 0.8: Excellent stability
# 0.4-0.8: Moderate stability
# < 0.4: Unstable (changes may cause bifurcation)

# Reorganization Gradient ΔNFR
print(f"ΔNFR: {results.delta_nfr:.3f}")
# Positive: Structure expanding
# Negative: Structure contracting
# ~0: Equilibrium
```

### Network Status

```python
if results.coherence > 0.7 and results.sense_index > 0.6:
    print("✓ Network is STABLE and healthy")
elif results.coherence < 0.3:
    print("⚠ Network is FRAGMENTING - apply coherence operators")
else:
    print("◐ Network is EVOLVING - monitor closely")
```

## Built-in Operator Sequences

TNFR provides pre-configured operator sequences for common patterns:

```python
# Available sequences
sequences = [
    "basic_activation",    # Emission → Coherence → Resonance
    "therapeutic",         # Self-org → Coherence → Regeneration
    "exploration",         # Dissonance → Mutation → Coherence
    "propagation",         # Emission → Resonance → Coupling
    "synchronization",     # Reception → Coherence → Silence
    "stabilization",       # Coherence → Silence → Coherence
]

# Use any sequence
network.apply_sequence("therapeutic", repeat=5)
```

## Visualization

### Basic Network Plot

```python
network.visualize()
```

Shows:
- Nodes (colored by coherence level)
- Edges (thickness = coupling strength)
- Phase synchronization

### Interactive Dashboard

```python
from tnfr.viz import launch_dashboard

# Launch interactive web dashboard
launch_dashboard(network)
```

Features:
- Real-time metric updates
- Operator controls
- Time-series plots
- Network graph explorer

### Save Metrics Plot

```python
import matplotlib.pyplot as plt

# Plot coherence over time
results.plot_metrics()
plt.savefig("coherence_evolution.png")
```

## Advanced Configuration

### Custom Node Parameters

```python
# Precise control over node properties
network.add_nodes(
    count=15,
    epi_range=(0.5, 1.5),      # EPI magnitude range
    freq_range=(10, 50),        # νf in Hz_str
    phase_sync=True,            # Start with synchronized phases
    initial_coherence=0.8       # Target initial C(t)
)
```

### Custom Topology

```python
# Different network topologies
network.connect_nodes(0.3, topology="random")       # Erdős–Rényi
network.connect_nodes(0.3, topology="scale_free")   # Barabási–Albert
network.connect_nodes(0.3, topology="small_world")  # Watts–Strogatz
network.connect_nodes(0.3, topology="lattice")      # Grid
```

### Backend Selection

```python
from tnfr.sdk import TNFRNetwork

# Choose computational backend
network = TNFRNetwork("my_net", backend="jax")  # GPU-accelerated
# Options: "numpy" (default), "jax", "torch"
```

## Performance Tips

### For Large Networks (>1000 nodes)

```bash
# Use JAX backend with GPU
pip install tnfr[jax]
```

```python
network = TNFRNetwork("large_net", backend="jax")
```

### Caching

```python
# Enable intelligent caching for repeated operations
from tnfr.sdk import TNFRNetwork

network = TNFRNetwork("cached_net", enable_cache=True)
```

## Next Steps

Now that you're up and running:

1. **Learn the Theory**: Read [Core Concepts](Core-Concepts.md) to understand the paradigm
2. **Explore Examples**: Check out [Examples & Use Cases](Examples.md) for your domain
3. **API Deep Dive**: See the [API Reference](https://tnfr.netlify.app/api/overview/) for advanced features
4. **Join the Community**: Ask questions in [GitHub Discussions](https://github.com/fermga/TNFR-Python-Engine/discussions)

## Troubleshooting

### Import Errors

```bash
# If you get "No module named 'tnfr'"
pip install --upgrade tnfr

# Or for development install
pip install -e .
```

### Low Coherence

If your network has low coherence (<0.3):

```python
# Apply stabilization sequence
network.apply_sequence("stabilization", repeat=5)
results = network.measure()
```

### Installation Issues

See [Troubleshooting Guide](https://tnfr.netlify.app/user-guide/TROUBLESHOOTING/) for common issues.

## Getting Help

- 📖 **Documentation**: https://tnfr.netlify.app
- 💬 **Discussions**: https://github.com/fermga/TNFR-Python-Engine/discussions
- 🐛 **Issues**: https://github.com/fermga/TNFR-Python-Engine/issues
- 📧 **Email**: See [CONTRIBUTING.md](https://github.com/fermga/TNFR-Python-Engine/blob/main/CONTRIBUTING.md)

---

[← Back to Home](Home.md) | [Core Concepts →](Core-Concepts.md) | [Examples →](Examples.md)
