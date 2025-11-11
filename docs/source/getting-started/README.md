# Getting Started with TNFR

[Home](../index.rst) › Getting Started

Welcome to TNFR (Teoría de la Naturaleza Fractal Resonante / Resonant Fractal Nature Theory)! This guide will help you get started with the TNFR Python Engine.

## What is TNFR?

TNFR is a computational paradigm that models reality as **coherent patterns that persist through resonance**, not as isolated objects. Instead of viewing systems as collections of independent entities, TNFR treats them as **resonant networks** where structures emerge and stabilize through synchronized vibration.

### Core Principle

**Reality is coherence, not substance.** What we perceive as "objects" or "structures" are actually **stable patterns of resonance** that persist because they synchronize with their environment.

### Key Concepts

- **NFR (Nodo Fractal Resonante)**: The fundamental unit of structural coherence
- **EPI (Estructura Primaria de Información)**: The coherent "form" or identity of a node
- **νf (Structural Frequency)**: The rate at which a node reorganizes (measured in Hz_str)
- **ΔNFR**: The internal reorganization gradient driving structural change
- **Phase (φ)**: Relative synchrony with the network

### The Nodal Equation

At the heart of TNFR is the canonical nodal equation:

```
∂EPI/∂t = νf · ΔNFR(t)
```

This equation describes how structures evolve: their rate of change depends on both their **capacity to reorganize** (νf) and the **pressure to change** (ΔNFR).

## Installation

### Quick Install

```bash
pip install tnfr
```

This installs the core engine with NumPy, NetworkX, and Cachetools.

### Optional Dependencies

For enhanced functionality:

```bash
# GPU acceleration with JAX
pip install tnfr[compute-jax]

# PyTorch backend
pip install tnfr[compute-torch]

# Visualization tools
pip install tnfr[viz-basic]

# All extras
pip install tnfr[compute-jax,compute-torch,viz-basic,yaml,orjson]
```

See [Optional Dependencies](optional-dependencies.md) for details.

## Your First TNFR Network

Create a simple resonant network in 3 lines:

```python
import tnfr

# Create a resonant network
G = tnfr.create_network(nodes=10, connectivity=0.3)

# Apply structural operators
tnfr.operators.coherence(G)

# Measure network coherence
C_t = tnfr.metrics.total_coherence(G)
print(f"Network coherence: {C_t:.3f}")
```

## Next Steps

### 🚀 **For Absolute Beginners** (5-10 minutes)
1. **[Quickstart Tutorial](quickstart.md)** - Run your first TNFR code
2. **[TNFR Concepts](TNFR_CONCEPTS.md)** - Understand the fundamentals
3. **[Interactive Tutorial](INTERACTIVE_TUTORIAL.md)** - Guided walkthrough

### 📚 **For Learning TNFR** (30-60 minutes)
1. **[Concepts Guide](TNFR_CONCEPTS.md)** - Deep dive into theory
2. **[User Guide](../user-guide/OPERATORS_GUIDE.md)** - Using the 13 operators
3. **[Examples](../examples/README.md)** - Practical examples

### 🔧 **For Building Applications**
1. **[API Overview](../api/overview.md)** - Package structure
2. **[Operators Reference](../api/operators.md)** - Complete operator documentation
3. **[Metrics Guide](../user-guide/METRICS_INTERPRETATION.md)** - Interpreting C(t), Si, etc.

### 🎓 **For Advanced Users**
1. **[Performance Optimization](../advanced/PERFORMANCE_OPTIMIZATION.md)** - Speed up your networks
2. **[Math Backends](math-backends.md)** - GPU acceleration
3. **[Mathematical Foundations](../theory/mathematical_foundations.md)** - ⭐ **CANONICAL MATH SOURCE** - Complete derivation

## Learning Paths

### Path 1: Quickest Start (10 minutes)
```
Quickstart → Interactive Tutorial → First Example
```

### Path 2: Comprehensive (1-2 hours)
```
Quickstart → TNFR Concepts → User Guide → Examples → API Reference
```

### Path 3: Theory-First (2-3 hours)
```
TNFR Concepts → Mathematical Foundations → Math Notebooks → Examples
```

## Key Resources

- **[Quickstart](quickstart.md)** - Get running in 5 minutes
- **[TNFR Concepts](TNFR_CONCEPTS.md)** - Core theory explained
- **[FAQ](FAQ.md)** - Common questions answered
- **[Glossary](../../GLOSSARY.md)** - Term definitions
- **[Examples](../examples/README.md)** - Runnable code samples

## Getting Help

- **Quick Questions**: Check the [FAQ](FAQ.md)
- **How-To Guides**: See [User Guide](../user-guide/OPERATORS_GUIDE.md)
- **API Details**: Read [API Overview](../api/overview.md)
- **Issues**: [Open a GitHub issue](https://github.com/fermga/TNFR-Python-Engine/issues)
- **Troubleshooting**: See [Troubleshooting Guide](../user-guide/TROUBLESHOOTING.md)

## Philosophy

TNFR is built on several core principles:

1. **Coherence First**: Structures exist through resonance, not substance
2. **Operational Fractality**: Patterns scale without losing structure
3. **Complete Traceability**: Every change is observable and reproducible
4. **Trans-Scale**: Works from quantum to social systems
5. **Domain Neutral**: No hard-coded assumptions about specific fields

---

**Ready to start?** Head to the [Quickstart Tutorial](quickstart.md) →

Or explore [TNFR Concepts](TNFR_CONCEPTS.md) to understand the theory first.
