# Examples

[Home](../index.rst) › Examples

Runnable examples illustrate how the TNFR engine orchestrates canonical operators in realistic scenarios. Each script can be executed directly with Python 3.9+ after installing `tnfr` and optional extras.

## Quick Navigation

- **[Use Cases Guide](USE_CASES_GUIDE.md)** - **NEW!** Domain mappings and operator selection guide
- **[Getting Started Examples](#basic-examples)** - Simple, educational examples
- **[Domain Examples](#domain-examples)** - Biology, social, technical applications
- **[Advanced Examples](#advanced-examples)** - Complex operator sequences
- **[CLI Examples](#cli-examples)** - Command-line interface usage

> 📘 **New to TNFR applications?** Start with the **[Use Cases Guide](USE_CASES_GUIDE.md)** to understand when and how to use different operators for your domain!

---

## Basic Examples

### Controlled Dissonance with Re-coherence

**File**: [`controlled_dissonance.py`](controlled_dissonance.py)

**Summary**: Three-node ring where node C receives a controlled dissonance pulse, bifurcates, and re-stabilises while telemetry records C(t), ΔNFR, Si, and history windows.

**Key Concepts**:
- Dissonance operator
- Bifurcation detection
- Coherence recovery
- Telemetry and metrics

**Run**:
```bash
python docs/source/examples/controlled_dissonance.py
```

**Learn**: How to use controlled instability for exploration while maintaining network coherence.

---

### Optical Cavity Feedback Loop

**File**: [`optical_cavity_feedback.py`](optical_cavity_feedback.py)

**Summary**: Tabletop optical cavity (laser head, piezo mirror stage, detector array) realigns after a thermal drift using self-organisation, mutation, and resonance sequences.

**Key Concepts**:
- Self-organization
- Mutation operator
- Resonance propagation
- Feedback control

**Run**:
```bash
python docs/source/examples/optical_cavity_feedback.py
```

**Learn**: How TNFR models physical systems with feedback loops and self-correction.

---

## Domain Examples

### Biological Cell Communication

**File**: [`biological_coherence_example.py`](biological_coherence_example.py)

**Summary**: Models how cells in a tissue coordinate responses through chemical signaling. Demonstrates cell-to-cell communication using emission/reception operators, gap junction formation via coupling, and tissue organization through coherence metrics.

**Key Concepts**:
- Emission operator (chemical signal secretion)
- Reception operator (receptor binding)
- Coupling operator (gap junctions, cell contacts)
- Coherence as tissue organization
- Sense index as cell responsiveness

**Run**:
```bash
python docs/source/examples/biological_coherence_example.py
```

**Learn**: How TNFR models biological systems with cellular communication and tissue formation.

---

### Social Network Information Dynamics

**File**: [`social_network_dynamics.py`](social_network_dynamics.py)

**Summary**: Models information propagation and opinion formation in social networks. Demonstrates how individuals (nodes) interact, debate (dissonance), evolve opinions (mutation), and reach consensus (coherence).

**Key Concepts**:
- Nodes as individuals in a social network
- Resonance operator (consensus building, viral spread)
- Dissonance operator (conflict, debate, diverse opinions)
- Mutation operator (opinion evolution)
- Coherence as group cohesion

**Run**:
```bash
python docs/source/examples/social_network_dynamics.py
```

**Learn**: How TNFR analyzes social dynamics, consensus formation, and network stability vs. change.

---

### Adaptive AI Self-Organization

**File**: [`adaptive_ai_system.py`](adaptive_ai_system.py)

**Summary**: Models an AI system that learns through structural reorganization rather than gradient descent. Demonstrates how intelligent agents self-organize and adapt using TNFR operators.

**Key Concepts**:
- Nodes as intelligent agents
- SelfOrganization operator (THOL) for autonomous structuring
- Learning as coherence increase (not error minimization)
- Structural adaptation without backpropagation

**Run**:
```bash
python docs/source/examples/adaptive_ai_system.py
```

**Learn**: TNFR's alternative paradigm for learning and adaptation in AI systems.

---

### Supply Chain Resilience Under Disruption

**File**: [`supply_chain_resilience.py`](supply_chain_resilience.py)

**Summary**: Models an adaptive supply chain responding to external disruptions. Demonstrates how suppliers and distributors reorganize using mutation operators to maintain system coherence.

**Key Concepts**:
- Nodes as supply chain facilities (suppliers, warehouses, distributors)
- Mutation operator (ZHIR) for adaptive reconfiguration
- Dissonance as disruption/shock
- Resilience metrics through sense index
- Recovery patterns and system adaptation

**Run**:
```bash
python docs/source/examples/supply_chain_resilience.py
```

**Learn**: How TNFR models business resilience and adaptive logistics networks.

---

### Urban Traffic Flow Optimization

**File**: [`urban_traffic_flow.py`](urban_traffic_flow.py)

**Summary**: Models urban traffic grid with adaptive signal coordination. Demonstrates traffic flow optimization by treating intersections as nodes and using transition operators for signal timing.

**Key Concepts**:
- Intersections as nodes, traffic flow as coherence
- Transition operator (NAV) for coordinated signal changes
- Dissonance minimization (congestion reduction)
- Dynamic adaptation to traffic patterns
- Green wave propagation through resonance

**Run**:
```bash
python docs/source/examples/urban_traffic_flow.py
```

**Learn**: How TNFR applies to urban planning and intelligent transportation systems.

---

### Technical Systems

**File**: Available in main [`examples/`](../../../examples/) directory

Examples include:
- Distributed computing
- Network resilience
- Load balancing

**Key Concepts**:
- System synchronization
- Fault tolerance through coherence
- Adaptive reorganization

---

## Advanced Examples

### Multi-scale Network

**File**: Available in main [`examples/`](../../../examples/multiscale_network_demo.py)

**Summary**: Demonstrates operational fractality with nested EPIs at multiple scales.

**Key Concepts**:
- Recursivity operator
- Hierarchical coherence
- Multi-scale metrics

### Parallel Computation

**File**: Available in main [`examples/`](../../../examples/parallel_computation_demo.py)

**Summary**: Shows how to leverage JAX/PyTorch backends for GPU acceleration.

**Key Concepts**:
- Backend selection
- Parallel operator application
- Performance optimization

### Intelligent Caching

**File**: Available in main [`examples/`](../../../examples/intelligent_caching_demo.py)

**Summary**: Demonstrates caching strategies for large networks.

**Key Concepts**:
- Buffer management
- Cache configuration
- Performance monitoring

---

## CLI Examples

### Reproduce Optical Cavity with CLI

Use the TNFR CLI to reproduce the optical cavity workflow using canonical tokens:

**Files**:
- Configuration: [`config.json`](config.json)
- Sequence: [`sequence.json`](sequence.json)

**Command**:
```bash
tnfr sequence \
  --nodes 3 --topology ring --seed 1 \
  --sequence-file docs/source/examples/sequence.json \
  --config docs/source/examples/config.json \
  --save-history history.json
```

**Token Legend**:

| Token  | English Operator |
| ------ | ---------------- |
| `AL`   | Emission         |
| `EN`   | Reception        |
| `IL`   | Coherence        |
| `UM`   | Coupling         |
| `RA`   | Resonance        |
| `SHA`  | Silence          |
| `NAV`  | Transition       |
| `OZ`   | Dissonance       |
| `ZHIR` | Mutation         |

The CLI run writes telemetry to `history.json`, mirroring the metrics produced by Python scripts. Inspect `W_stats` and `nodal_diag` entries to correlate coherence spans with node states.

---

## More Examples

Complete collection of examples in the main repository:

**Location**: [`examples/`](../../../examples/) directory

**Includes**:
- `hello_world.py` - Simplest possible TNFR code
- `sdk_example.py` - Full SDK capabilities  
- `canonical_equation_demo.py` - Nodal equation demonstration
- `backend_performance_comparison.py` - Backend benchmarks
- `sparse_graph_demo.py` - Sparse network optimization
- And more!

**Browse all**: [GitHub Examples Directory](https://github.com/fermga/TNFR-Python-Engine/tree/main/examples)

---

## Example Categories

### By Learning Goal

**Understanding Basics**:
1. `controlled_dissonance.py` - Core operator sequence
2. `hello_world.py` - Minimal TNFR code
3. `canonical_equation_demo.py` - Nodal equation

**Domain Applications**:
1. Biological examples - Cell/neural networks
2. Social examples - Group dynamics
3. Technical examples - Distributed systems

**Performance Optimization**:
1. `backend_performance_comparison.py` - Backend selection
2. `parallel_computation_demo.py` - GPU acceleration
3. `intelligent_caching_demo.py` - Caching strategies

**Advanced Techniques**:
1. `multiscale_network_demo.py` - Recursivity and fractality
2. `optical_cavity_feedback.py` - Self-organization
3. `sparse_graph_demo.py` - Large networks

### By Complexity

**Beginner** (< 50 lines):
- `hello_world.py`
- `canonical_equation_demo.py`

**Intermediate** (50-200 lines):
- `controlled_dissonance.py`
- `sdk_example.py`
- `backend_performance_comparison.py`

**Advanced** (200+ lines):
- `optical_cavity_feedback.py`
- `multiscale_network_demo.py`
- `parallel_computation_demo.py`

---

## Running Examples

### Prerequisites

```bash
# Install TNFR
pip install tnfr

# Optional: Install extras for specific examples
pip install tnfr[viz-basic]        # Visualization
pip install tnfr[compute-jax]      # GPU acceleration
pip install tnfr[orjson]           # Fast caching
```

### General Pattern

```bash
# Run from repository root
cd /path/to/TNFR-Python-Engine

# Basic examples (in docs/source/examples/)
python docs/source/examples/controlled_dissonance.py

# Main examples (in examples/)
python examples/hello_world.py
python examples/sdk_example.py

# With visualization (requires viz-basic)
python examples/canonical_equation_demo.py
```

### CLI Examples

```bash
# Help
tnfr --help
tnfr sequence --help

# Run predefined sequence
tnfr sequence \
  --nodes 5 --topology ring \
  --sequence-file examples/sequence.json
```

---

## Creating Your Own Examples

Template for new examples:

```python
#!/usr/bin/env python3
"""
Example: Your Example Name

Description: Brief description of what this demonstrates.

Key Concepts:
- Concept 1
- Concept 2

Run:
    python your_example.py
"""

import tnfr
from tnfr.operators import Coherence, Resonance
from tnfr.metrics import total_coherence, sense_index

def main():
    # Create network
    G = tnfr.create_network(nodes=10, connectivity=0.3)
    
    # Apply operators
    Coherence()(G)
    Resonance()(G, list(G.nodes())[0])
    
    # Measure results
    C_t = total_coherence(G)
    Si = sense_index(G)
    
    print(f"Coherence: {C_t:.3f}")
    print(f"Sense Index: {Si:.3f}")

if __name__ == "__main__":
    main()
```

---

## See Also

- **[Quickstart Tutorial](../getting-started/quickstart.md)** - Basic usage
- **[Operators Guide](../user-guide/OPERATORS_GUIDE.md)** - Operator details
- **[API Reference](../api/overview.md)** - Complete API
- **[Performance Optimization](../advanced/PERFORMANCE_OPTIMIZATION.md)** - Speed tips

---

**Need help?** Check the [FAQ](../getting-started/FAQ.md) or [Troubleshooting Guide](../user-guide/TROUBLESHOOTING.md).
