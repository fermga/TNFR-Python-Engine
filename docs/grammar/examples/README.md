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

### Constraint-Focused Examples (NEW)

#### u1-initiation-closure-examples.py

**Level:** Beginner  
**Constraint:** U1 (Structural Initiation & Closure)  
**Focus:** U1a (Initiation), U1b (Closure)

**Demonstrates:**
- Valid generator patterns (AL, NAV, REMESH)
- Valid closure patterns (SHA, NAV, REMESH, OZ)
- When U1a applies (EPI=0 vs EPI>0)
- Dual-role operators (NAV, REMESH)
- Common anti-patterns and mistakes

**Run:**
```bash
python docs/grammar/examples/u1-initiation-closure-examples.py
```

**Sections:**
- U1a valid examples (starting with generators)
- U1a invalid examples (missing generators)
- U1a context (when initiation applies)
- U1b valid examples (ending with closures)
- U1b invalid examples (missing closures)
- Dual role operators

**Key Learning:**
- Cannot evolve from EPI=0 without generator
- All sequences need closure operator
- Some operators serve multiple roles

---

#### u2-convergence-examples.py

**Level:** Intermediate  
**Constraint:** U2 (Convergence & Boundedness)  
**Focus:** Stabilizer-destabilizer balance

**Demonstrates:**
- Valid balanced sequences
- Invalid unbalanced sequences
- When U2 applies (has destabilizers)
- Operator classification
- Ordering importance
- Anti-patterns (masking, accumulation)

**Run:**
```bash
python docs/grammar/examples/u2-convergence-examples.py
```

**Sections:**
- U2 valid examples (balanced)
- U2 invalid examples (unbalanced)
- U2 not applicable (no destabilizers)
- Operator classification
- Ordering matters (stabilizer placement)
- Masking anti-pattern
- Interleaving pattern (best practice)

**Key Learning:**
- Without stabilizers: ∫νf·ΔNFR dt → ∞
- Stabilizer order matters
- Interleave for better control

---

#### u3-resonant-coupling-examples.py

**Level:** Intermediate  
**Constraint:** U3 (Resonant Coupling)  
**Focus:** Phase verification requirement

**Demonstrates:**
- Phase compatibility checking
- Coupling/resonance operators
- Sequence-level validation
- Wave interference physics
- Anti-patterns (no check, phase drift)

**Run:**
```bash
python docs/grammar/examples/u3-resonant-coupling-examples.py
```

**Sections:**
- Phase compatibility examples
- Coupling/resonance operator requirements
- Sequence-level validation (meta-rule)
- Anti-pattern: No phase check
- Anti-pattern: Phase drift
- Threshold considerations
- Wave interference physics

**Key Learning:**
- Phase check is MANDATORY (Invariant #5)
- |φᵢ - φⱼ| ≤ π/2 typically required
- Antiphase = destructive interference

---

#### u4-bifurcation-examples.py

**Level:** Advanced  
**Constraint:** U4 (Bifurcation Dynamics)  
**Focus:** U4a (Triggers need handlers), U4b (Transformers need context)

**Demonstrates:**
- Valid bifurcation sequences
- Invalid uncontrolled bifurcations
- Transformer context requirements
- ZHIR-specific requirements
- Anti-patterns (cascades, wrong handlers, window violations)

**Run:**
```bash
python docs/grammar/examples/u4-bifurcation-examples.py
```

**Sections:**
- U4a valid examples (triggers with handlers)
- U4a invalid examples (uncontrolled)
- U4b valid examples (transformers with context)
- U4b invalid examples (missing context)
- Operator classification
- ZHIR-specific requirements
- Anti-pattern: Bifurcation cascade
- Anti-pattern: Context window violation
- Handler selection best practices

**Key Learning:**
- Bifurcations need control (U4a)
- Transformers need energy (U4b)
- ZHIR needs stable base + recent destabilizer

---

### Pattern-Based Examples (EXISTING)

#### 01-basic-bootstrap.py

**Level:** Beginner  
**Pattern:** Bootstrap (minimal)  
**Sequence:** `[Emission, Coherence, Silence]`

**Demonstrates:**
- U1a: Starting with generator (Emission)
- U1b: Ending with closure (Silence)
- Minimal valid sequence
- Basic telemetry export

**Run:**
```bash
python docs/grammar/examples/01-basic-bootstrap.py
```

---

#### 02-intermediate-exploration.py

**Level:** Intermediate  
**Pattern:** Controlled exploration  
**Sequence:** `[Emission, Coherence, Dissonance, Coherence, Silence]`

**Demonstrates:**
- U2: Destabilizer (Dissonance) balanced by stabilizer (Coherence)
- U4a: Bifurcation trigger (Dissonance) with handler (Coherence)
- Exploration with stability

**Run:**
```bash
python docs/grammar/examples/02-intermediate-exploration.py
```

---

#### 03-advanced-bifurcation.py

**Level:** Advanced  
**Pattern:** Complete transformation  
**Sequence:** `[Emission, Coherence, Dissonance, Mutation, SelfOrganization, Coherence, Silence]`

**Demonstrates:**
- U4b: Mutation with proper context (prior IL, recent destabilizer)
- U4a: Multiple handlers (SelfOrganization, Coherence)
- Phase transformation

**Run:**
```bash
python docs/grammar/examples/03-advanced-bifurcation.py
```

---

## Example Categories

### By Constraint

**U1 - Initiation & Closure:**
- `u1-initiation-closure-examples.py` - Comprehensive U1 coverage
- `01-basic-bootstrap.py` - Emission as generator
- All examples (always start with generator when EPI=0)

**U2 - Convergence & Boundedness:**
- `u2-convergence-examples.py` - Comprehensive U2 coverage
- `02-intermediate-exploration.py` - Dissonance + Coherence
- `03-advanced-bifurcation.py` - Multiple destabilizers balanced

**U3 - Resonant Coupling:**
- `u3-resonant-coupling-examples.py` - Comprehensive U3 coverage
- (Planned: 04-network-propagation.py)

**U4 - Bifurcation Dynamics:**
- `u4-bifurcation-examples.py` - Comprehensive U4a/U4b coverage
- `02-intermediate-exploration.py` - Dissonance with handler
- `03-advanced-bifurcation.py` - Mutation with handlers

### By Pattern

**Bootstrap:**
- `01-basic-bootstrap.py`
- `u1-initiation-closure-examples.py` (simple sequences)

**Exploration:**
- `02-intermediate-exploration.py`
- `u2-convergence-examples.py` (balanced sequences)

**Transformation:**
- `03-advanced-bifurcation.py`
- `u4-bifurcation-examples.py` (transformer sequences)

**Anti-Patterns:**
- `u1-initiation-closure-examples.py` - U1 anti-patterns
- `u2-convergence-examples.py` - U2 anti-patterns
- `u3-resonant-coupling-examples.py` - U3 anti-patterns
- `u4-bifurcation-examples.py` - U4 anti-patterns

---

## Running Examples

### Individual Example

```bash
# Run specific example
python docs/grammar/examples/u1-initiation-closure-examples.py
```

### All Constraint Examples

```bash
# Run all U1-U4 examples
for constraint in u1 u2 u3 u4; do
    echo "Running ${constraint} examples..."
    python docs/grammar/examples/${constraint}-*-examples.py
    echo "---"
done
```

### All Pattern Examples

```bash
# Run all pattern examples
for f in docs/grammar/examples/0*.py; do
    echo "Running $f..."
    python "$f"
    echo "---"
done
```

### All Examples

```bash
# Run everything
python docs/grammar/examples/u1-initiation-closure-examples.py
python docs/grammar/examples/u2-convergence-examples.py
python docs/grammar/examples/u3-resonant-coupling-examples.py
python docs/grammar/examples/u4-bifurcation-examples.py
python docs/grammar/examples/01-basic-bootstrap.py
python docs/grammar/examples/02-intermediate-exploration.py
python docs/grammar/examples/03-advanced-bifurcation.py
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
