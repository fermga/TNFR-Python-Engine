# TNFR Python Engine

<div align="center">

**Model reality as resonant patterns, not isolated objects**

[![PyPI](https://img.shields.io/pypi/v/tnfr)](https://pypi.org/project/tnfr/)
[![Python](https://img.shields.io/pypi/pyversions/tnfr)](https://pypi.org/project/tnfr/)
[![License](https://img.shields.io/github/license/fermga/TNFR-Python-Engine)](LICENSE.md)
[![Documentation](https://img.shields.io/badge/docs-latest-brightgreen)](https://tnfr.netlify.app)

[Quick Start](#-quick-start) • [Key Concepts](#-key-concepts) • [Documentation](#-documentation) • [Examples](#-examples) • [Contributing](#-contributing)

</div>

---

## 🌟 What is TNFR?

**TNFR** (**Resonant Fractal Nature Theory** / **Teoría de la Naturaleza Fractal Resonante**) is a paradigm shift in modeling complex systems. Instead of viewing reality as isolated "things" that interact through cause-and-effect, TNFR models it as **coherent patterns that persist through resonance**.

Think of a choir: each singer maintains their unique voice while synchronizing with others to create harmony. When voices resonate, they produce stable, beautiful structures. When they clash, patterns fragment. **TNFR captures this principle mathematically and makes it operational in code.**

### 🎯 Why TNFR?

| Traditional Approach | TNFR Paradigm |
|---------------------|---------------|
| Objects exist independently | Patterns exist through resonance |
| Causality: A causes B | Coherence: A and B co-organize |
| Static snapshots | Dynamic reorganization |
| Domain-specific models | Trans-scale, trans-domain |

**Key Advantages:**
- 🔄 **Operational Fractality**: Patterns scale without losing structure
- 📊 **Complete Traceability**: Every reorganization is observable
- 🎯 **Guaranteed Reproducibility**: Same conditions → same outcomes
- 🌐 **Domain Neutral**: Works from quantum to social systems

### 🚀 Use Cases

- 🧬 **Biology**: Cellular networks, neuronal synchronization, protein dynamics
- 🌐 **Social Systems**: Information spread, community formation, opinion dynamics
- 🤖 **AI**: Resonant symbolic systems, emergent learning
- 🔬 **Network Science**: Structural coherence, pattern detection
- 🏗️ **Distributed Systems**: Decentralized coordination, self-organization

---

## ⚡ Quick Start

### Installation

```bash
pip install tnfr
```

Requires Python ≥ 3.9

### Your First TNFR Network (3 Lines!)

```python
from tnfr.sdk import TNFRNetwork

# Create, activate, and measure a network
network = TNFRNetwork("hello_world")
results = network.add_nodes(10).connect_nodes(0.3, "random").apply_sequence("basic_activation", repeat=3).measure()
print(results.summary())
```

🎉 **That's it!** You just created a resonant network.

**What happened?**
- `add_nodes(10)`: Created 10 nodes that can synchronize
- `connect_nodes(0.3, "random")`: Connected them (30% probability)
- `apply_sequence("basic_activation", repeat=3)`: Applied Emission → Coherence → Resonance (3x)
- `measure()`: Calculated coherence C(t), sense index Si, and structural metrics

### 🎓 Interactive Learning (5 Minutes)

```python
from tnfr.tutorials import hello_tnfr
hello_tnfr()  # Guided tour of TNFR concepts
```

**Domain Examples:**
```python
from tnfr.tutorials import (
    biological_example,      # Cell communication
    social_network_example,  # Social dynamics
    technology_example,      # Distributed systems
    adaptive_ai_example,     # Learning through resonance
)
```

📘 **Structured Learning Path**: See our [**60-Minute Interactive Tutorial**](docs/source/getting-started/INTERACTIVE_TUTORIAL.md)

---

## 💡 Key Concepts

> **New to TNFR?** 👉 [**TNFR Fundamental Concepts Guide**](docs/source/getting-started/TNFR_CONCEPTS.md) - Understand the paradigm in 10 minutes!

### The Nodal Equation

At the heart of TNFR is one elegant equation:

```
∂EPI/∂t = νf · ΔNFR(t)
```

**What it means:**
- **EPI**: Primary Information Structure (the "shape" of a node)
- **νf**: Structural frequency (reorganization rate in Hz_str)
- **ΔNFR**: Internal reorganization operator (structural gradient)

*Structure changes proportionally to frequency and gradient*

### Three Essential Elements

**1. Resonant Fractal Node (NFR)**
- Minimum unit of structural coherence
- Has EPI (form), νf (frequency), φ (phase)

**2. Structural Operators** (13 canonical)
- **Emission/Reception**: Initiate & capture patterns
- **Coherence/Dissonance**: Stabilize or destabilize
- **Resonance**: Propagate without losing identity
- **Self-organization**: Create emergent structures
- [See all 13 operators →](GLOSSARY.md#structural-operators)

**3. Coherence Metrics**
- **C(t)**: Total network coherence [0,1]
- **Si**: Sense index (reorganization stability)
- **ΔNFR**: Evolution gradient

---

## 📚 Documentation

### 🎯 Single Source of Truth for Mathematics

**[Mathematical Foundations of TNFR](docs/source/theory/mathematical_foundations.md)** ⭐

This is THE ONLY place where TNFR mathematics is formally defined:
- Hilbert space H_NFR and Banach space B_EPI
- Coherence operator Ĉ (spectral theory, proofs)
- Frequency operator Ĵ and reorganization operator ΔNFR
- Complete nodal equation derivation
- **§3.1.1**: Implementation bridge (theory → code)

### 🎯 Classical Mechanics Emergence

**TNFR reveals how observable classical physics emerges from structural coherence dynamics**:

```
TNFR Nodal Equation (∂EPI/∂t = νf · ΔNFR)
           ↓ 
    Low-dissonance limit (ε → 0)
           ↓
Observable Classical Mechanics
```

**Key Emergent Phenomena**:
- **Mass**: `m = 1/νf` (inverse structural frequency) — mass is structural inertia
- **Force**: `F = -∇U(q)` (coherence potential gradient) — force is stability flow
- **Newton's Laws**: Natural consequences of the nodal equation at low dissonance
- **Action Principle**: Coherence optimization over time
- **Conservation Laws**: Network symmetries preserve structural quantities

**Documentation**:
- **[📘 N-Body Classical Mechanics Guide](docs/TNFR_CLASSICAL_NBODY.md)** — **Complete formal reference** (variable mappings, conservation laws, validation protocols, code examples)
- [Classical Mechanics from TNFR](docs/source/theory/07_emergence_classical_mechanics.md) — Complete derivation from nodal equation
- [Euler-Lagrange Correspondence](docs/source/theory/08_classical_mechanics_euler_lagrange.md) — Variational formulation
- [Numerical Validation](docs/source/theory/09_classical_mechanics_numerical_validation.md) — Computational verification

**Practical Examples**:
- `examples/domain_applications/nbody_gravitational.py` — Two-body orbits, three-body systems
- `examples/nbody_quantitative_validation.py` — Full validation suite (6 canonical experiments)
- `tests/validation/test_nbody_validation.py` — Automated test suite

This demonstrates **classical mechanics as a natural expression of coherent structural dynamics** in the observable, deterministic regime.

### 📖 Quick References

- **[GLOSSARY](GLOSSARY.md)** - Operational definitions for code use
- **[TNFR Concepts](docs/source/getting-started/TNFR_CONCEPTS.md)** - Paradigm introduction
- **[API Overview](docs/source/api/overview.md)** - Package architecture
- **[Operator Guide](docs/source/api/operators.md)** - Complete operator reference

### 🎨 Grammar 2.0 (Operator Sequences)

Enhanced validation system for operator sequences:

```python
from tnfr.operators.grammar import validate_sequence_with_health

result = validate_sequence_with_health(["emission", "reception", "coherence"])
print(f"Health: {result.health_metrics.overall_health:.2f}")
print(f"Pattern: {result.metadata['detected_pattern']}")
```

**Features:**
- 📊 Health metrics (7 dimensions, 0.0-1.0 scale)
- 🎨 18 structural patterns (Linear, Fractal, Regenerative, etc.)
- 🔄 Regenerative cycle detection
- ⚡ Structural frequencies (Hz_str units)

**Learn More**: [Operator Sequences Guide](GLYPH_SEQUENCES_GUIDE.md) • [Migration Guide 2.0](docs/MIGRATION_GUIDE_2.0.md)

### 🧪 Advanced Topics

- [ARCHITECTURE.md](ARCHITECTURE.md) - System design & invariants
- [Backend System](docs/backends.md) - NumPy/JAX/Torch backends
- [TESTING.md](TESTING.md) - Test strategy & validation
- [SECURITY.md](SECURITY.md) - Security practices
- [CONTRIBUTING.md](CONTRIBUTING.md) - Development workflow

---

## 🔬 Examples

### Hello World

```python
# examples/hello_world.py
from tnfr.sdk import TNFRNetwork

network = TNFRNetwork("simple_demo")
results = (network
    .add_nodes(5)
    .connect_nodes(0.5, "random")
    .apply_sequence("basic_activation")
    .measure())

print(f"Coherence: {results.coherence:.3f}")
print(f"Sense Index: {results.sense_index:.3f}")
```

### Biological Network

```python
# examples/biological_network.py
from tnfr.sdk import TNFRNetwork

# Model cellular communication
cells = TNFRNetwork("cell_network")
results = (cells
    .add_nodes(20, epi_range=(0.8, 1.2))  # Biological variation
    .connect_nodes(0.3, "scale_free")      # Power-law connectivity
    .apply_sequence("therapeutic", repeat=5)  # Healing pattern
    .measure())

print(f"Network health: {results.coherence:.2%}")
```

### More Examples

- [Dynamic Limits](examples/dynamic_limits_demo.py) - Adaptive thresholds
- [Multiscale Networks](examples/multiscale_network_demo.py) - Hierarchical structures
- [Regenerative Cycles](examples/regenerative_cycles.py) - Self-sustaining patterns
- [Performance Comparison](examples/backend_performance_comparison.py) - Backend benchmarks

📂 **Full Collection**: [examples/](examples/) directory

---

## 🛠️ Development

### Local Setup

```bash
# Clone repository
git clone https://github.com/fermga/TNFR-Python-Engine.git
cd TNFR-Python-Engine

# Install with development dependencies
pip install -e ".[dev,docs]"

# Run tests
./scripts/run_tests.sh

# Format code
./scripts/format.sh
```

### Documentation Build

```bash
# Install docs dependencies
pip install -r docs/requirements.txt

# Build documentation
make docs

# View locally
open docs/_build/html/index.html
```

### Configuration & Secrets

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your credentials (never commit this file!)
# Load with:
```

```python
from tnfr.secure_config import load_redis_config, get_cache_secret
redis_config = load_redis_config()
```

See [SECURITY.md](SECURITY.md) for best practices.

---

## 🤝 Contributing

We welcome contributions! Here's how to get started:

1. **Understand TNFR**: Read [Mathematical Foundations](docs/source/theory/mathematical_foundations.md)
2. **Check Invariants**: Follow [AGENTS.md](AGENTS.md) rules
3. **Write Tests**: Cover all invariants (see [TESTING.md](TESTING.md))
4. **Run QA**: Execute `./scripts/run_tests.sh`
5. **Submit PR**: See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines

**Key Principles:**
- ✅ Preserve canonical invariants
- ✅ Use structural operators only
- ✅ Document with references to Mathematical Foundations
- ✅ Test spectral properties

---

## 📊 CLI Tools

### Profiling Pipeline

```bash
tnfr profile-pipeline \
  --nodes 120 --edge-probability 0.28 --loops 3 \
  --si-chunk-sizes auto 48 --dnfr-chunk-sizes auto \
  --output-dir profiles/pipeline
```

Generates `.pstats` and JSON summaries for performance analysis.

---

## 📖 Learning Path

**Recommended Progression:**

1. **Newcomers** (10 min)
   - Read [TNFR Concepts](docs/source/getting-started/TNFR_CONCEPTS.md)
   - Run `hello_tnfr()` tutorial

2. **Beginners** (30 min)
   - Try [examples/hello_world.py](examples/hello_world.py)
   - Explore domain examples (biological, social, AI)

3. **Intermediate** (2 hours)
   - Study [Mathematical Foundations §1-3](docs/source/theory/mathematical_foundations.md)
   - Read [GLOSSARY](GLOSSARY.md)
   - Practice with [Interactive Tutorial](docs/source/getting-started/INTERACTIVE_TUTORIAL.md)

4. **Advanced** (ongoing)
   - Deep dive: [Mathematical Foundations (complete)](docs/source/theory/mathematical_foundations.md)
   - Architecture: [ARCHITECTURE.md](ARCHITECTURE.md)
   - Contribute: [CONTRIBUTING.md](CONTRIBUTING.md)

---

## 📜 License

Released under the [MIT License](LICENSE.md).

**Citation**: When publishing research or applications based on TNFR, please cite:
- This repository: `fermga/TNFR-Python-Engine`
- Theoretical foundations: [TNFR.pdf](TNFR.pdf)
- Mathematical formalization: [Mathematical Foundations](docs/source/theory/mathematical_foundations.md)

---

## 🔗 Links

- **Documentation**: https://tnfr.netlify.app
- **PyPI Package**: https://pypi.org/project/tnfr/
- **GitHub**: https://github.com/fermga/TNFR-Python-Engine
- **Issues**: https://github.com/fermga/TNFR-Python-Engine/issues

---

<div align="center">

**Made with ❤️ for researchers, developers, and explorers of complex systems**

*Reality is not made of things—it's made of resonance*

</div>
