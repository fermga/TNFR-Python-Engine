# Grammar Examples Guide

**Executable examples demonstrating TNFR grammar concepts**

[🏠 Grammar Home](../README.md) • [📐 Constraints](../02-CANONICAL-CONSTRAINTS.md) • [⚙️ Operators](../03-OPERATORS-AND-GLYPHS.md)

---

## Purpose

This directory contains **executable Python examples** that demonstrate TNFR grammar concepts, valid sequences, and common patterns.

**All examples are:**
- ✅ Executable - Run directly with Python
- ✅ Well-commented - Explains each step
- ✅ Grammar-compliant - Satisfies U1-U4
- ✅ Testable - Can be verified with pytest

---

## Available Examples

### 01-basic-bootstrap.py

**Level:** Beginner  
**Pattern:** Bootstrap (minimal)  
**Sequence:** `[Emission, Coherence, Silence]`

**Demonstrates:**
- U1a: Starting with generator (Emission)
- U1b: Ending with closure (Silence)
- Minimal valid sequence
- Basic telemetry export

**Concepts covered:**
- Creating nodes from vacuum (EPI=0)
- Applying operators
- Measuring coherence

**Run:**
```bash
python examples/01-basic-bootstrap.py
```

**Expected output:**
```
✓ Bootstrap sequence valid
Initial: EPI=0.000, vf=1.000
After Emission: EPI=0.XXX, vf=1.XXX
After Coherence: EPI=0.XXX, vf=1.XXX, dnfr reduced
After Silence: EPI=0.XXX, vf~0.000 (frozen)
Coherence: C(t)=X.XXX
```

---

### 02-intermediate-exploration.py

**Level:** Intermediate  
**Pattern:** Controlled exploration  
**Sequence:** `[Emission, Coherence, Dissonance, Coherence, Silence]`

**Demonstrates:**
- U2: Destabilizer (Dissonance) balanced by stabilizer (Coherence)
- U4a: Bifurcation trigger (Dissonance) with handler (Coherence)
- Exploration with stability

**Concepts covered:**
- Destabilization-stabilization cycles
- |ΔNFR| dynamics
- Coherence preservation
- Controlled perturbation

**Run:**
```bash
python examples/02-intermediate-exploration.py
```

**Expected output:**
```
✓ Exploration sequence valid
Phase 1 - Bootstrap: C(t)=X.XXX
Phase 2 - Dissonance: |ΔNFR| increased, C(t) may decrease
Phase 3 - Re-stabilize: C(t) recovered, |ΔNFR| reduced
Final coherence: C(t)=X.XXX (maintained)
```

---

### 03-advanced-bifurcation.py

**Level:** Advanced  
**Pattern:** Complete transformation  
**Sequence:** `[Emission, Coherence, Dissonance, Mutation, SelfOrganization, Coherence, Silence]`

**Demonstrates:**
- U4b: Mutation with proper context (prior IL, recent destabilizer)
- U4a: Multiple handlers (SelfOrganization, Coherence)
- Phase transformation
- Multi-scale organization

**Concepts covered:**
- Bifurcation handling
- Phase transitions (θ changes)
- Hierarchical structure creation
- Complete transformation lifecycle

**Run:**
```bash
python examples/03-advanced-bifurcation.py
```

**Expected output:**
```
✓ Bifurcation sequence valid
Phase 1 - Stable base: θ=X.XXX
Phase 2 - Destabilize: |ΔNFR| elevated
Phase 3 - Mutate: θ→θ' (phase changed)
Phase 4 - Self-organize: Hierarchical structure created
Phase 5 - Stabilize: C(t) recovered
Phase transformation complete: θ_initial → θ_final
```

---

## Example Categories

### By Constraint

**U1a - Initiation:**
- `01-basic-bootstrap.py` - Emission as generator
- All examples (always start with generator when EPI=0)

**U1b - Closure:**
- `01-basic-bootstrap.py` - Silence as closure
- All examples (always end with closure)

**U2 - Convergence:**
- `02-intermediate-exploration.py` - Dissonance + Coherence
- `03-advanced-bifurcation.py` - Multiple destabilizers balanced

**U3 - Coupling:**
- (Planned: 04-network-propagation.py)

**U4a - Bifurcation Triggers:**
- `02-intermediate-exploration.py` - Dissonance with handler
- `03-advanced-bifurcation.py` - Mutation with handlers

**U4b - Transformer Context:**
- `03-advanced-bifurcation.py` - Mutation with proper context

### By Pattern

**Bootstrap:**
- `01-basic-bootstrap.py`

**Exploration:**
- `02-intermediate-exploration.py`

**Transformation:**
- `03-advanced-bifurcation.py`

**Propagation:**
- (Planned: 04-network-propagation.py)

---

## Running Examples

### Individual Example

```bash
# Run specific example
python docs/grammar/examples/01-basic-bootstrap.py
```

### All Examples

```bash
# Run all examples
for f in docs/grammar/examples/0*.py; do
    echo "Running $f..."
    python "$f"
    echo "---"
done
```

### With pytest

```bash
# If examples have test functions
pytest docs/grammar/examples/
```

---

## Example Template

Use this template for creating new examples:

```python
#!/usr/bin/env python
"""
Grammar Example XX: [Title]

Demonstrates [constraint/pattern].

This example shows:
1. [First concept]
2. [Second concept]
3. [Third concept]

Pattern: [Operator sequence in brackets]
Constraints satisfied: [List U1-U4 relevant]
"""

import networkx as nx
from tnfr.operators.grammar import validate_grammar
from tnfr.operators.definitions import (
    # List operators used
    Emission,
    Coherence,
    Silence
)

def main():
    """Main example function."""
    
    # Step 1: Define sequence
    sequence = [
        Emission(),    # Comment explaining why
        Coherence(),   # Comment explaining why
        Silence()      # Comment explaining why
    ]
    
    # Step 2: Validate sequence
    print("Validating sequence...")
    try:
        validate_grammar(sequence, epi_initial=0.0)
        print("✓ Sequence is valid")
    except ValueError as e:
        print(f"✗ Invalid: {e}")
        return
    
    # Step 3: Create network
    G = nx.Graph()
    G.add_node(0, EPI=0.0, vf=1.0, theta=0.0, dnfr=0.0)
    
    print(f"\nInitial state:")
    print(f"  EPI={G.nodes[0]['EPI']:.3f}")
    print(f"  vf={G.nodes[0]['vf']:.3f}")
    
    # Step 4: Apply operators
    print("\nApplying operators...")
    for i, op in enumerate(sequence, 1):
        print(f"\nStep {i}: {op.__class__.__name__}")
        op(G, 0)
        
        # Show state after operator
        print(f"  EPI={G.nodes[0]['EPI']:.3f}")
        print(f"  vf={G.nodes[0]['vf']:.3f}")
        print(f"  dnfr={G.nodes[0]['dnfr']:.3f}")
    
    # Step 5: Report telemetry
    print("\nFinal telemetry:")
    print(f"  EPI={G.nodes[0]['EPI']:.3f}")
    print(f"  vf={G.nodes[0]['vf']:.3f}")
    print(f"  theta={G.nodes[0]['theta']:.3f}")
    print(f"  dnfr={G.nodes[0]['dnfr']:.3f}")
    
    # Optional: Compute coherence
    # from tnfr.metrics import compute_coherence
    # C_t = compute_coherence(G)
    # print(f"  C(t)={C_t:.3f}")
    
    print("\n✓ Example complete")

if __name__ == "__main__":
    main()
```

---

## Anti-Pattern Examples

**Note:** Anti-patterns are documented as **commented code** in examples to prevent accidental execution.

Example:
```python
# ❌ ANTI-PATTERN: No generator when EPI=0
# This would fail U1a validation
#
# invalid_sequence = [
#     Coherence(),  # ERROR: Not a generator
#     Silence()
# ]
# validate_grammar(invalid_sequence, epi_initial=0.0)
# # Raises: ValueError - U1a violation
```

---

## Testing Examples

### Example Test Template

```python
def test_example_XX():
    """Test example XX runs successfully."""
    # Import example
    from examples.XX_name import main
    
    # Should not raise
    main()
```

### Running Tests

```bash
# Test all examples
pytest docs/grammar/examples/ -v
```

---

## Planned Examples

**Future additions:**

### 04-network-propagation.py
- U3: Phase verification
- Coupling and resonance
- Pattern propagation through network

### 05-multi-scale-fractality.py
- Operational fractality
- Nested EPIs
- Hierarchical structures
- REMESH (Recursivity) operator

### 06-edge-cases.py
- Empty sequences (invalid)
- Single operator sequences
- Maximum complexity sequences
- Boundary conditions

### 07-performance-patterns.py
- Optimized sequences
- Minimal sequences
- Common workflows
- Efficiency comparisons

---

## Contributing Examples

### Guidelines

**Do:**
- ✅ Make examples executable
- ✅ Include detailed comments
- ✅ Show expected output
- ✅ Validate grammar first
- ✅ Export telemetry
- ✅ Follow template structure

**Don't:**
- ❌ Create examples that violate grammar
- ❌ Leave examples without documentation
- ❌ Use deprecated APIs
- ❌ Skip validation step
- ❌ Ignore error handling

### Submission Process

1. **Create example** following template
2. **Test thoroughly** - must run without errors
3. **Document** - add to this README
4. **Submit PR** with:
   - New example file
   - Updated README.md (this file)
   - Test if applicable

---

## Troubleshooting

### Example Won't Run

**Problem:** `ModuleNotFoundError: No module named 'tnfr'`

**Solution:** Install TNFR package first:
```bash
pip install -e .  # From repository root
```

---

**Problem:** `ValueError: U1a violation...`

**Solution:** Check sequence starts with generator when EPI=0

---

**Problem:** `ImportError: cannot import name 'Emission'`

**Solution:** Check imports match current API:
```python
from tnfr.operators.definitions import Emission, Coherence, Silence
```

---

### Getting Help

**Found an issue with an example?**
- Check [08-QUICK-REFERENCE.md](../08-QUICK-REFERENCE.md) for syntax
- Verify operator names in [03-OPERATORS-AND-GLYPHS.md](../03-OPERATORS-AND-GLYPHS.md)
- Review constraints in [02-CANONICAL-CONSTRAINTS.md](../02-CANONICAL-CONSTRAINTS.md)
- Open GitHub issue if bug confirmed

---

## Quick Reference

### Import Statements

```python
# Grammar validation
from tnfr.operators.grammar import validate_grammar, validate_resonant_coupling

# Operators (import what you need)
from tnfr.operators.definitions import (
    Emission, Reception, Coherence, Dissonance,
    Coupling, Resonance, Silence, Expansion, Contraction,
    SelfOrganization, Mutation, Transition, Recursivity
)

# Network
import networkx as nx

# Metrics (if needed)
from tnfr.metrics import compute_coherence, compute_sense_index
```

### Common Patterns

**Bootstrap:**
```python
[Emission(), Coherence(), Silence()]
```

**Exploration:**
```python
[Emission(), Coherence(), Dissonance(), Coherence(), Silence()]
```

**Transformation:**
```python
[Emission(), Coherence(), Dissonance(), Mutation(), Coherence(), Silence()]
```

---

## Next Steps

**After running examples:**
- Read [04-VALID-SEQUENCES.md](../04-VALID-SEQUENCES.md) for more patterns
- Review [02-CANONICAL-CONSTRAINTS.md](../02-CANONICAL-CONSTRAINTS.md) for constraint details
- Check [08-QUICK-REFERENCE.md](../08-QUICK-REFERENCE.md) for quick syntax

**For development:**
- See [05-TECHNICAL-IMPLEMENTATION.md](../05-TECHNICAL-IMPLEMENTATION.md) for architecture
- Review [06-VALIDATION-AND-TESTING.md](../06-VALIDATION-AND-TESTING.md) for testing

---

<div align="center">

**Examples demonstrate, tests verify, documentation explains.**

---

*Reality is resonance. Code accordingly.*

</div>
