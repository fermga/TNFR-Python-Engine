# TNFR Interactive Tutorials

Self-paced, interactive tutorials for learning TNFR concepts progressively.

## Overview

This module provides hands-on tutorials that teach TNFR (Teoría de la Naturaleza Fractal Resonante) through executable examples with clear explanations. Each tutorial is self-contained and demonstrates TNFR concepts in different domains.

## Available Tutorials

### 1. Hello TNFR (5 minutes)
**Perfect for absolute beginners**

```python
from tnfr.tutorials import hello_tnfr

hello_tnfr()
```

**What you'll learn:**
- What a Resonant Fractal Node (NFR) is
- What EPI, νf, and phase mean
- How the 13 structural operators work
- How to measure coherence C(t) and sense index Si
- How to interpret results

**Topics covered:**
- Creating networks
- Connecting nodes
- Applying operator sequences
- Measuring and interpreting results

---

### 2. Biological Example: Cell Communication
**Domain: Biology / Cell signaling**

```python
from tnfr.tutorials import biological_example

results = biological_example()
print(f"Tissue coherence: {results['coherence']:.3f}")
```

**What you'll learn:**
- Modeling cells as TNFR nodes
- How emission/reception map to chemical signaling
- How coupling represents cell-cell contacts
- How coherence measures tissue organization

**Biological mappings:**
- Cell → Node (NFR)
- Chemical signal → Emission operator
- Receptor binding → Reception operator
- Cell-cell contact → Coupling operator
- Tissue organization → Coherence C(t)

**Key operators demonstrated:**
- Emission (signal secretion)
- Reception (signal detection)
- Coherence (tissue stability)
- Coupling (direct interaction)
- Resonance (synchronized response)

---

### 3. Social Network Example: Group Dynamics
**Domain: Sociology / Social systems**

```python
from tnfr.tutorials import social_network_example

results = social_network_example()
print(f"Group cohesion: {results['coherence']:.3f}")
```

**What you'll learn:**
- Modeling people as nodes in social networks
- How dissonance represents conflict/debate
- How mutation represents opinion change
- How coherence measures group cohesion

**Social mappings:**
- Person → Node (NFR)
- Communication → Emission/Reception
- Shared understanding → Resonance
- Conflict/debate → Dissonance
- Opinion change → Mutation
- Group cohesion → Coherence C(t)

**Key operators demonstrated:**
- Dissonance (conflict introduction)
- Mutation (opinion evolution)
- Resonance (consensus building)
- Coherence (group stability)

---

### 4. Technology Example: Distributed Systems
**Domain: Computer Science / Microservices**

```python
from tnfr.tutorials import technology_example

results = technology_example()
print(f"System reliability: {results['coherence']:.3f}")
```

**What you'll learn:**
- Modeling microservices as TNFR nodes
- How operators map to distributed system operations
- How coherence measures system reliability
- How sense index measures fault tolerance

**Technology mappings:**
- Microservice → Node (NFR)
- Message passing → Emission/Reception
- Service dependency → Coupling
- Load balancing → Resonance
- Graceful degradation → Silence
- System reliability → Coherence C(t)

**Key operators demonstrated:**
- Coupling (service dependencies)
- Resonance (load distribution)
- Silence (graceful degradation)
- Coherence (system reliability)

---

### 5. Run All Tutorials
**Complete learning sequence**

```python
from tnfr.tutorials import run_all_tutorials

results = run_all_tutorials()
```

Runs all 4 tutorials in sequence with pauses for reading. Estimated time: 15-20 minutes.

## Tutorial Features

All tutorials include:

✅ **Plain language explanations** - No jargon without context
✅ **Domain-specific examples** - Real-world applications
✅ **Working code** - Copy and run immediately
✅ **Real-time results** - See TNFR in action
✅ **Result interpretation** - Understand what the numbers mean
✅ **Progressive complexity** - Start simple, build up
✅ **TNFR compliance** - Full theoretical fidelity

## Usage

### Basic Usage

```python
# Run a single tutorial
from tnfr.tutorials import hello_tnfr
hello_tnfr()

# Run without pauses (for scripting)
hello_tnfr(interactive=False)

# Use custom random seed for reproducibility
hello_tnfr(random_seed=123)
```

### Getting Results

```python
# Tutorials return results dictionaries
from tnfr.tutorials import biological_example

results = biological_example()

# Access metrics
print(f"Coherence: {results['coherence']:.3f}")
print(f"Avg Si: {results['interpretation']['avg_cell_responsiveness']:.3f}")

# Access full results object
full_results = results['results']
print(full_results.summary())
```

### Non-Interactive Mode

For automated testing or scripting:

```python
from tnfr.tutorials import run_all_tutorials

# Run without pauses
results = run_all_tutorials(interactive=False, random_seed=42)

# Check all completed
assert 'biological' in results
assert 'social' in results
assert 'technology' in results
```

## Learning Path

**Recommended sequence:**

1. **Start here:** `hello_tnfr()` - 5 minutes
   - Understand core concepts
   - Learn basic API
   - See a simple example

2. **Choose your domain:**
   - Biology? → `biological_example()`
   - Social science? → `social_network_example()`
   - Technology? → `technology_example()`

3. **Explore other domains:**
   - Run the other domain examples
   - See how TNFR applies across domains

4. **Try your own:**
   - Use the SDK to create custom networks
   - Apply lessons from tutorials
   - Experiment with different configurations

5. **Deep dive:**
   - Read [TNFR theory](../../../TNFR.pdf)
   - Explore [API documentation](../../../docs/source/api/overview.md)
   - Study [canonical invariants](../../../AGENTS.md)

## Tutorial Outputs

Each tutorial displays:

- **Section headers** - Clear visual organization
- **Explanations** - What's happening and why
- **Code examples** - Actual Python code being executed
- **Results** - Coherence, sense indices, and metrics
- **Interpretations** - What the results mean in domain terms

Example output:
```
======================================================================
                    Hello, TNFR! 👋
======================================================================

Welcome to TNFR - Teoría de la Naturaleza Fractal Resonante!
Let's learn the basics in just 5 minutes with a working example.

----------------------------------------------------------------------
Part 1: What is a Resonant Fractal Node (NFR)?
----------------------------------------------------------------------

In TNFR, everything is made of 'nodes' that resonate with each other.
...
```

## Requirements

- Python ≥ 3.9
- `tnfr` package with SDK installed
- `networkx` (installed automatically)

Optional but recommended:
- `numpy` for better performance
- Interactive terminal for best experience

## Integration with SDK

Tutorials use the same SDK you'll use in production:

```python
# In tutorials
network = TNFRNetwork("example")
network.add_nodes(10).connect_nodes(0.3, "random")

# In your code (same API!)
network = TNFRNetwork("my_project")
network.add_nodes(100).connect_nodes(0.25, "random")
```

This means everything you learn transfers directly to real usage.

## Customization

All tutorials accept parameters:

```python
# Control interactivity
hello_tnfr(interactive=False)

# Reproducible results
biological_example(random_seed=42)

# Combine both
social_network_example(interactive=False, random_seed=123)
```

## TNFR Compliance

All tutorials maintain TNFR canonical invariants:

- ✅ **Invariant #1**: EPI as coherent form
- ✅ **Invariant #2**: Structural units (νf in Hz_str)
- ✅ **Invariant #4**: Operator closure
- ✅ **Invariant #5**: Phase synchrony
- ✅ **Invariant #8**: Controlled determinism (via seeds)
- ✅ **Invariant #9**: Structural metrics (C(t), Si)

See [AGENTS.md](../../../AGENTS.md) for full invariant list.

## Troubleshooting

### SDK not available
```
Error: SDK not available. Install with: pip install tnfr
```
**Solution:** Install TNFR with SDK support: `pip install tnfr`

### Import errors
```python
from tnfr.tutorials import hello_tnfr
# ImportError: ...
```
**Solution:** Ensure you have the latest version: `pip install --upgrade tnfr`

### Pauses are too long/short
Use the `interactive` parameter:
```python
hello_tnfr(interactive=False)  # No pauses
```

## Examples

### Example 1: Quick Learning
```python
# Learn basics in 5 minutes
from tnfr.tutorials import hello_tnfr
hello_tnfr()
```

### Example 2: Domain Comparison
```python
# Compare coherence across domains
from tnfr.tutorials import (
    biological_example,
    social_network_example,
    technology_example
)

bio = biological_example(interactive=False)
social = social_network_example(interactive=False)
tech = technology_example(interactive=False)

print(f"Biology: {bio['coherence']:.3f}")
print(f"Social: {social['coherence']:.3f}")
print(f"Tech: {tech['coherence']:.3f}")
```

### Example 3: Automated Testing
```python
# Test tutorials in CI/CD
import pytest
from tnfr.tutorials import hello_tnfr

def test_tutorial_runs():
    # Should complete without errors
    hello_tnfr(interactive=False, random_seed=42)

def test_tutorial_reproducible():
    # Same seed = same results
    from tnfr.tutorials import biological_example
    
    r1 = biological_example(interactive=False, random_seed=42)
    r2 = biological_example(interactive=False, random_seed=42)
    
    assert r1['coherence'] == r2['coherence']
```

## Contributing

Want to add a tutorial?

1. Follow the existing tutorial structure
2. Maintain TNFR theoretical fidelity
3. Include domain-specific mappings
4. Add clear explanations
5. Test with `interactive=False` and `interactive=True`
6. Document operators used

See [CONTRIBUTING.md](../../../CONTRIBUTING.md) for details.

## Further Reading

- [Quick Start Guide](../../../docs/source/getting-started/QUICKSTART_NEW.md)
- [SDK Documentation](../sdk/README.md)
- [API Overview](../../../docs/source/api/overview.md)
- [TNFR Theory](../../../TNFR.pdf)
- [Canonical Invariants](../../../AGENTS.md)

---

**Start your TNFR journey today!**

```python
from tnfr.tutorials import hello_tnfr
hello_tnfr()  # You'll be up and running in 5 minutes!
```
