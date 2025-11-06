# Frequently Asked Questions (FAQ)

[Home](../index.rst) › [Getting Started](README.md) › FAQ

## General Questions

### What is TNFR?

TNFR (Teoría de la Naturaleza Fractal Resonante / Resonant Fractal Nature Theory) is a computational paradigm that models reality as **coherent patterns sustained by resonance**, not as isolated objects. It provides a framework for understanding and simulating complex systems across all scales.

### Why use TNFR instead of traditional modeling?

Traditional approaches model systems as collections of independent objects. TNFR takes a fundamentally different view:

- **Traditional**: Objects exist independently, interact through cause-and-effect
- **TNFR**: Patterns exist through resonance, co-organize through coherence

Benefits:
- 🎯 Operational fractality: patterns scale without losing structure
- 🔄 Complete reproducibility: same conditions → same outcomes
- 🔍 Full traceability: every reorganization is observable
- 🌐 Trans-scale: works from quantum to social systems

### What domains can TNFR model?

TNFR is **domain-neutral** and works across:
- **Biology**: Cell networks, neural systems, ecosystems
- **Social**: Communities, organizations, cultural dynamics
- **Technology**: Distributed systems, networks, AI
- **Physics**: Quantum systems, field dynamics
- **Economics**: Markets, resource flows
- And any system where patterns emerge through interaction

## Installation & Setup

### How do I install TNFR?

Basic installation:
```bash
pip install tnfr
```

With optional features:
```bash
# GPU acceleration
pip install tnfr[compute-jax]

# All features
pip install tnfr[compute-jax,viz-basic,yaml,orjson]
```

### What Python versions are supported?

Python 3.9 or newer is required.

### Do I need a GPU?

No! TNFR works with:
- **NumPy backend** (default, CPU-only)
- **JAX backend** (optional, GPU-accelerated)
- **PyTorch backend** (optional, GPU-accelerated)

GPUs are optional for performance optimization.

### How do I check if TNFR is installed correctly?

```python
import tnfr
print(tnfr.__version__)

# Create a simple network
G = tnfr.create_network(nodes=5)
print(f"Created network with {G.number_of_nodes()} nodes")
```

## Core Concepts

### What is an NFR (Resonant Fractal Node)?

An NFR is the minimum unit of structural coherence in TNFR. Think of it as a **tuning fork in a network** that:
- Has its own natural frequency (νf)
- Responds to nearby vibrations
- Maintains a coherent form (EPI)
- Synchronizes through phase (φ)

### What is EPI?

EPI (Estructura Primaria de Información / Primary Information Structure) is the coherent "form" or "identity" of a node. Like a musical chord has a specific structure, EPI defines the structural configuration of a node.

### What is νf (structural frequency)?

νf is the rate at which a node reorganizes its internal structure, measured in **Hz_str** (structural hertz). It's like a "heart rate for change":
- Higher νf = faster reorganization
- Lower νf = slower, more stable evolution
- νf → 0 = node collapse

**Important**: νf is NOT a physical frequency (like sound waves).

### What is ΔNFR?

ΔNFR is the internal reorganization gradient - the "pressure" driving structural change. It measures the difference between a node's current state and the network around it:
- Positive ΔNFR: expansion, growth
- Negative ΔNFR: contraction, simplification
- Large |ΔNFR|: intense reorganization

### What is the Nodal Equation?

The canonical equation governing TNFR:
```
∂EPI/∂t = νf · ΔNFR(t)
```

Translation: "A structure changes at a rate proportional to its reorganization capacity (νf) and the pressure to change (ΔNFR)."

### What are the 13 structural operators?

The operators are the **only valid way** to modify nodes in TNFR:

1. **Emission** - Initiate patterns
2. **Reception** - Receive external patterns
3. **Coherence** - Stabilize structure
4. **Dissonance** - Introduce controlled instability
5. **Coupling** - Create links between nodes
6. **Resonance** - Amplify and propagate patterns
7. **Silence** - Temporarily freeze evolution
8. **Expansion** - Increase complexity
9. **Contraction** - Reduce complexity
10. **Self-organization** - Spontaneous pattern formation
11. **Mutation** - Phase transformation
12. **Transition** - Navigate between states
13. **Recursivity** - Nested operations

See [Operators Guide](../user-guide/OPERATORS_GUIDE.md) for details.

## Using TNFR

### How do I create a network?

```python
import tnfr

# Simple network
G = tnfr.create_network(nodes=10, connectivity=0.3)

# Custom initialization
G = tnfr.create_network(
    nodes=20,
    connectivity=0.5,
    initial_frequency=1.0,  # Hz_str
    phase_distribution='uniform'
)
```

### How do I apply operators?

```python
from tnfr.operators import coherence, resonance, coupling

# Single operator
coherence(G)

# Sequence of operators
coupling(G, node1, node2)
resonance(G, source_node)
coherence(G)
```

### How do I measure network health?

```python
from tnfr.metrics import total_coherence, sense_index

# Total coherence C(t)
C_t = total_coherence(G)
print(f"Coherence: {C_t:.3f}")  # 0.0-1.0

# Sense index Si
Si = sense_index(G)
print(f"Sense index: {Si:.3f}")  # 0.0-1.0+

# Per-node metrics
for node in G.nodes():
    nf = G.nodes[node]['nf']  # structural frequency
    phase = G.nodes[node]['phase']
    print(f"Node {node}: νf={nf:.2f} Hz_str, φ={phase:.2f} rad")
```

### What metrics should I monitor?

Key metrics:
- **C(t)** (Total Coherence): Overall network stability (0.0-1.0)
- **Si** (Sense Index): Reorganization stability (0.0-1.0+)
- **νf** (Structural Frequency): Per-node reorganization rate
- **Phase**: Network synchronization (0-2π radians)

Healthy networks show:
- C(t) > 0.5 (moderate coherence)
- Si > 0.4 (stable reorganization)
- Phase coherence > 0.3

See [Metrics Interpretation Guide](../user-guide/METRICS_INTERPRETATION.md).

### Can I visualize TNFR networks?

Yes! With the visualization extra:
```bash
pip install tnfr[viz-basic]
```

```python
import tnfr.visualization as viz

# Visualize network
viz.plot_network(G, show_phase=True, show_frequency=True)

# Coherence over time
viz.plot_coherence_evolution(G, timesteps=100)
```

## Troubleshooting

### My network coherence is very low. What's wrong?

Low C(t) can result from:
- Insufficient coupling between nodes
- Phase desynchronization
- Excessive dissonance
- Very low structural frequencies

**Solutions**:
1. Apply `coherence()` operator
2. Increase coupling: `coupling(G, node1, node2)`
3. Check phase distribution
4. Verify νf values are > 0

See [Troubleshooting Guide](../user-guide/TROUBLESHOOTING.md).

### How do I debug operator sequences?

Enable telemetry:
```python
from tnfr.telemetry import enable_tracing

enable_tracing()  # Log all operator applications
```

Or use the validator:
```python
from tnfr.validation import validate_network

issues = validate_network(G)
for issue in issues:
    print(issue)
```

### My code is slow. How can I optimize?

1. **Use JAX backend for GPU acceleration**:
   ```python
   import tnfr
   tnfr.set_backend('jax')
   ```

2. **Enable caching**:
   ```bash
   pip install tnfr[orjson]
   ```

3. **Optimize network size**: Start small, scale gradually

See [Performance Optimization](../advanced/PERFORMANCE_OPTIMIZATION.md).

### Can I save/load networks?

Yes:
```python
import tnfr

# Save
tnfr.save_network(G, 'network.json')

# Load
G = tnfr.load_network('network.json')
```

YAML support:
```bash
pip install tnfr[yaml]
```

```python
tnfr.save_network(G, 'network.yaml', format='yaml')
```

## Advanced Topics

### What is operational fractality?

Operational fractality means patterns maintain their structure across scales. An EPI can contain nested sub-EPIs without losing functional identity. This is like Russian dolls: each level is complete in itself.

### What is the difference between coherence and coupling?

- **Coherence**: Internal stability of nodes (individual alignment)
- **Coupling**: Structural links between nodes (network connections)

Both are necessary: coherence stabilizes individual nodes, coupling creates network structure.

### Can TNFR handle large-scale networks?

Yes! TNFR scales through:
- **Sparse networks**: Only necessary connections
- **Hierarchical structure**: Nested EPIs
- **Efficient backends**: JAX/PyTorch for large networks
- **Caching**: Repeated computations cached

See [Scalability](../../SCALABILITY.md) and [Performance Optimization](../advanced/PERFORMANCE_OPTIMIZATION.md).

### How do I extend TNFR?

1. **Custom operators**: Compose existing operators
2. **Custom metrics**: Implement metric functions
3. **Custom backends**: Add new computational backends

See [Extending TNFR](../advanced/EXTENDING_TNFR.md) (coming soon).

### Where can I find the mathematical foundations?

- **Theory notebooks**: `docs/source/theory/`
- **TNFR.pdf**: Complete theoretical document
- **Foundations guide**: [foundations.md](../foundations.md)

## Contributing

### How can I contribute?

See [CONTRIBUTING.md](../../CONTRIBUTING.md) for:
- Code contributions
- Documentation improvements
- Bug reports
- Feature requests

### What are the TNFR invariants?

TNFR has 10 canonical invariants that must be preserved. See [AGENTS.md](../../AGENTS.md) for the complete list, including:
1. EPI changes only through structural operators
2. νf expressed in Hz_str units
3. ΔNFR semantics (not ML gradient)
4. Operator closure
5. Phase verification before coupling
6. And 5 more...

## Still Have Questions?

- **Documentation**: [Browse all docs](../../DOCUMENTATION_INDEX.md)
- **Examples**: [Example catalog](../examples/README.md)
- **GitHub Issues**: [Ask a question](https://github.com/fermga/TNFR-Python-Engine/issues)
- **API Reference**: [Complete API docs](../api/overview.md)

---

**See Also**:
- [Getting Started](README.md)
- [Quickstart Tutorial](quickstart.md)
- [TNFR Concepts](TNFR_CONCEPTS.md)
- [User Guide](../user-guide/OPERATORS_GUIDE.md)
