# Technical Implementation

**Architecture and implementation details of the TNFR grammar system**

[🏠 Home](README.md) • [📐 Constraints](02-CANONICAL-CONSTRAINTS.md) • [🔄 Sequences](04-VALID-SEQUENCES.md) • [🧪 Testing](06-VALIDATION-AND-TESTING.md)

---

## Purpose

This document details the **technical architecture** of the TNFR grammar validation system, including code structure, algorithms, and integration points.

**Prerequisites:** [02-CANONICAL-CONSTRAINTS.md](02-CANONICAL-CONSTRAINTS.md), [03-OPERATORS-AND-GLYPHS.md](03-OPERATORS-AND-GLYPHS.md)

**Audience:** Developers modifying core grammar system

**Reading time:** 45-60 minutes

---

## Architecture Overview

### Core Components

```
src/tnfr/operators/
├── grammar.py              # Grammar validation (U1-U5)
├── definitions.py          # Operator implementations
└── unified_grammar.py      # Legacy compatibility layer
```

### Key Files

**`grammar.py`** - Main grammar validation
- Defines operator sets (GENERATORS, CLOSURES, etc.)
- Implements `validate_grammar()` function
- Implements `validate_resonant_coupling()` for U3
- Contains all U1-U5 validation logic

**`definitions.py`** - Operator implementations
- Implements all 13 operators as classes
- Each operator modifies graph nodes
- Integrated with grammar validation

**`unified_grammar.py`** - Compatibility
- Bridges old and new grammar systems
- Provides legacy API
- Will be deprecated

---

## Operator Sets

### Definition

```python
# From src/tnfr/operators/grammar.py

# U1a: Generators (can start from EPI=0)
GENERATORS = {"emission", "transition", "recursivity"}

# U1b: Closures (can end sequences)
CLOSURES = {"silence", "transition", "recursivity", "dissonance"}

# U2: Stabilizers (negative feedback)
STABILIZERS = {"coherence", "selforganization"}

# U2: Destabilizers (positive feedback)
DESTABILIZERS = {"dissonance", "mutation", "expansion"}

# U3: Coupling/Resonance (phase-sensitive)
COUPLING_RESONANCE = {"coupling", "resonance"}

# U4a: Bifurcation triggers
BIFURCATION_TRIGGERS = {"dissonance", "mutation"}

# U4a: Bifurcation handlers
BIFURCATION_HANDLERS = {"selforganization", "coherence"}

# U4b: Transformers (need context)
TRANSFORMERS = {"mutation", "selforganization"}
```

### Set Operations

```python
def get_operator_name(operator):
    """Extract operator name from instance."""
    return operator.__class__.__name__.lower()

def is_generator(operator):
    """Check if operator is a generator."""
    return get_operator_name(operator) in GENERATORS

def is_closure(operator):
    """Check if operator is a closure."""
    return get_operator_name(operator) in CLOSURES

# Similar for other classifications
```

---

## Main Validation Functions

### Convenience Function: `validate_grammar()`

**Location**: `src/tnfr/operators/grammar.py`

```python
def validate_grammar(
    sequence: List[Operator],
    epi_initial: float = 0.0
) -> bool:
    """
    Validate operator sequence against U1-U5 constraints.
    
    Args:
        sequence: List of operator instances
        epi_initial: Initial EPI value (0.0 = vacuum)
    
    Returns:
        bool: True if sequence is valid, False otherwise
        
    Notes:
        This is a convenience wrapper around GrammarValidator.validate()
        that returns only the boolean result. For detailed validation
        messages, use GrammarValidator.validate() instead.
    """
```

**Implementation**:
```python
def validate_grammar(sequence: List[Operator], epi_initial: float = 0.0) -> bool:
    """Validate sequence using canonical TNFR grammar constraints."""
    is_valid, _ = GrammarValidator.validate(sequence, epi_initial)
    return is_valid
```

### Full Validator: `GrammarValidator.validate()`

**Location**: `src/tnfr/operators/grammar.py`

```python
@classmethod
def validate(
    cls,
    sequence: List[Operator],
    epi_initial: float = 0.0,
) -> tuple[bool, List[str]]:
    """
    Validate sequence using all unified canonical constraints.

    This validates pure TNFR physics:
    - U1: Structural initiation & closure
    - U2: Convergence & boundedness
    - U3: Resonant coupling
    - U4: Bifurcation dynamics

    Parameters
    ----------
    sequence : List[Operator]
        Sequence to validate
    epi_initial : float, optional
        Initial EPI value (default: 0.0)

    Returns
    -------
    tuple[bool, List[str]]
        (is_valid, messages)
        is_valid: True if all constraints satisfied
        messages: List of validation messages for each rule
    """
```

### Implementation Structure

```python
# In src/tnfr/operators/grammar.py

class GrammarValidator:
    """Validates sequences using canonical TNFR grammar constraints."""

    @classmethod
    def validate(cls, sequence, epi_initial=0.0):
        """Validate sequence against all U1-U5 rules."""
        messages = []
        all_valid = True

        # U1a: Initiation
        valid_init, msg_init = cls.validate_initiation(sequence, epi_initial)
        messages.append(f"U1a: {msg_init}")
        all_valid = all_valid and valid_init

        # U1b: Closure
        valid_closure, msg_closure = cls.validate_closure(sequence)
        messages.append(f"U1b: {msg_closure}")
        all_valid = all_valid and valid_closure

        # U2: Convergence
        valid_conv, msg_conv = cls.validate_convergence(sequence)
        messages.append(f"U2: {msg_conv}")
        all_valid = all_valid and valid_conv

        # U3: Resonant coupling
        valid_coupling, msg_coupling = cls.validate_resonant_coupling(sequence)
        messages.append(f"U3: {msg_coupling}")
        all_valid = all_valid and valid_coupling

        # U4a: Bifurcation triggers
        valid_triggers, msg_triggers = cls.validate_bifurcation_triggers(sequence)
        messages.append(f"U4a: {msg_triggers}")
        all_valid = all_valid and valid_triggers

        # U4b: Transformer context
        valid_context, msg_context = cls.validate_transformer_context(sequence)
        messages.append(f"U4b: {msg_context}")
        all_valid = all_valid and valid_context

        # U2-REMESH: Recursive amplification control
        valid_remesh, msg_remesh = cls.validate_remesh_amplification(sequence)
        messages.append(f"U2-REMESH: {msg_remesh}")
        all_valid = all_valid and valid_remesh

        return all_valid, messages


def validate_grammar(sequence, epi_initial=0.0):
    """Convenience function - returns only bool."""
    is_valid, _ = GrammarValidator.validate(sequence, epi_initial)
    return is_valid
```

### Individual Validation Methods

Each rule is implemented as a static method returning `tuple[bool, str]`:

#### U1a: `validate_initiation()`
```python
@staticmethod
def validate_initiation(sequence, epi_initial=0.0):
    """Check if sequence starts with generator when EPI=0."""
    if epi_initial > 0.0:
        return True, "U1a: EPI>0, initiation not required"
    
    if not sequence:
        return False, "U1a violated: Empty sequence with EPI=0"
    
    first_op = getattr(sequence[0], "canonical_name", sequence[0].name.lower())
    
    if first_op not in GENERATORS:
        return (
            False,
            f"U1a violated: EPI=0 requires generator (got '{first_op}'). "
            f"Valid: {sorted(GENERATORS)}",
        )
    
    return True, f"U1a satisfied: starts with generator '{first_op}'"
```

#### U1b: `validate_closure()`
```python
@staticmethod
def validate_closure(sequence):
    """Check if sequence ends with closure operator."""
    if not sequence:
        return False, "U1b violated: Empty sequence has no closure"
    
    last_op = getattr(sequence[-1], "canonical_name", sequence[-1].name.lower())
    
    if last_op not in CLOSURES:
        return (
            False,
            f"U1b violated: Sequence must end with closure (got '{last_op}'). "
            f"Valid: {sorted(CLOSURES)}",
        )
    
    return True, f"U1b satisfied: ends with closure '{last_op}'"
```

#### U2: `validate_convergence()`
```python
@staticmethod
def validate_convergence(sequence):
    """Check destabilizers have stabilizers for convergence."""
    destabilizers_present = [
        getattr(op, "canonical_name", op.name.lower())
        for op in sequence
        if getattr(op, "canonical_name", op.name.lower()) in DESTABILIZERS
    ]
    
    if not destabilizers_present:
        return True, "U2: not applicable (no destabilizers present)"
    
    stabilizers_present = [
        getattr(op, "canonical_name", op.name.lower())
        for op in sequence
        if getattr(op, "canonical_name", op.name.lower()) in STABILIZERS
    ]
    
    if not stabilizers_present:
        return (
            False,
            f"U2 violated: destabilizers {destabilizers_present} present "
            f"without stabilizer. Integral ∫νf·ΔNFR dt may diverge. "
            f"Add: {sorted(STABILIZERS)}",
        )
    
    return (
        True,
        f"U2 satisfied: stabilizers {stabilizers_present} "
        f"bound destabilizers {destabilizers_present}",
    )
```

See full implementation in `src/tnfr/operators/grammar.py` for U3, U4a, U4b, and U2-REMESH.

---

## Phase Validation (U3)

U3 (RESONANT COUPLING) is a **meta-rule** that documents the requirement for phase verification during coupling/resonance operations.

**Key Point**: Unlike U1, U2, U4 which validate sequences, U3 validates **runtime operations** when coupling/resonance operators are applied to specific nodes.

### Validation Approach

```python
@staticmethod
def validate_resonant_coupling(sequence):
    """Document U3 awareness for sequences with coupling/resonance.
    
    This method checks if sequence contains coupling/resonance operators
    and returns an awareness message. Actual phase verification happens
    at runtime in operator preconditions.
    """
    coupling_ops = [
        getattr(op, "canonical_name", op.name.lower())
        for op in sequence
        if getattr(op, "canonical_name", op.name.lower()) in COUPLING_RESONANCE
    ]
    
    if not coupling_ops:
        return True, "U3: not applicable (no coupling/resonance operators)"
    
    return (
        True,
        f"U3 awareness: operators {coupling_ops} require phase verification "
        f"(MANDATORY per Invariant #5). Enforced in preconditions.",
    )
```

### Runtime Phase Check

Phase compatibility is verified when operators are applied to nodes:

```python
# In operator preconditions (during application)
def check_phase_compatibility(G, node_i, node_j, delta_phi_max=np.pi/2):
    """Verify phase compatibility for coupling/resonance (U3).
    
    Called by Coupling and Resonance operators before creating links.
    """
    phi_i = G.nodes[node_i]['theta']
    phi_j = G.nodes[node_j]['theta']
    
    # Compute phase difference
    delta_phi = abs(phi_i - phi_j)
    
    # Normalize to [0, π] (considering periodicity)
    if delta_phi > np.pi:
        delta_phi = 2 * np.pi - delta_phi
    
    # Check compatibility
    if delta_phi > delta_phi_max:
        raise ValueError(
            f"U3 violation: Phase mismatch |φ_{node_i} - φ_{node_j}| = "
            f"{delta_phi:.3f} rad > Δφ_max = {delta_phi_max:.3f} rad"
        )
```

**Location**: Operator preconditions in `src/tnfr/operators/preconditions/`

---

## Integration with Operators

### Operator Base Structure

```python
# From definitions.py

class Emission:
    """
    AL - Emission operator.
    
    Physics: Creates EPI from vacuum via resonant emission.
    Grammar: Generator (U1a).
    """
    
    def __init__(self, **kwargs):
        """Initialize with parameters."""
        self.params = kwargs
    
    def __call__(self, G, node_id):
        """
        Apply operator to node.
        
        Args:
            G: NetworkX graph
            node_id: Target node
        """
        # Get current state
        current_epi = G.nodes[node_id].get('EPI', 0.0)
        current_vf = G.nodes[node_id].get('vf', 1.0)
        
        # Apply transformation
        new_epi = current_epi + 0.1  # Simplified
        new_vf = current_vf * 1.1
        
        # Update state
        G.nodes[node_id]['EPI'] = new_epi
        G.nodes[node_id]['vf'] = new_vf
```

### Validation Integration

```python
# Typical usage pattern

from tnfr.operators.grammar import validate_grammar, GrammarValidator
from tnfr.operators.definitions import Emission, Coherence, Silence

# 1. Define sequence
sequence = [Emission(), Coherence(), Silence()]

# 2. Validate BEFORE applying

# Option A: Simple boolean check
is_valid = validate_grammar(sequence, epi_initial=0.0)
if not is_valid:
    print("Sequence invalid!")
else:
    print("Sequence valid, proceed")

# Option B: Get detailed messages
is_valid, messages = GrammarValidator.validate(sequence, epi_initial=0.0)
print(f"Valid: {is_valid}")
for msg in messages:
    print(f"  {msg}")

# 3. Apply to network
G = nx.Graph()
G.add_node(0, EPI=0.0, vf=1.0, theta=0.0, dnfr=0.0)

for operator in sequence:
    operator(G, 0)

# 4. Check telemetry
print(f"Final EPI: {G.nodes[0]['EPI']:.3f}")
```

---

## Telemetry and Logging

### Essential Telemetry

```python
def export_telemetry(G):
    """Export essential TNFR metrics."""
    
    telemetry = {
        'timestamp': time.time(),
        'coherence': compute_coherence(G),  # C(t)
        'nodes': {}
    }
    
    for node in G.nodes():
        telemetry['nodes'][node] = {
            'EPI': G.nodes[node]['EPI'],
            'vf': G.nodes[node]['vf'],           # Hz_str
            'theta': G.nodes[node]['theta'],     # Phase
            'dnfr': G.nodes[node]['dnfr'],       # ΔNFR
            'sense_index': compute_sense_index(G, node)  # Si
        }
    
    return telemetry
```

### Operator Logging

```python
def log_operator_application(operator, node_id, telemetry_before, telemetry_after):
    """Log operator application for debugging."""
    
    log_entry = {
        'operator': operator.__class__.__name__,
        'node': node_id,
        'time': time.time(),
        'delta_EPI': telemetry_after['EPI'] - telemetry_before['EPI'],
        'delta_vf': telemetry_after['vf'] - telemetry_before['vf'],
        'delta_dnfr': telemetry_after['dnfr'] - telemetry_before['dnfr'],
    }
    
    return log_entry
```

---

## Performance Considerations

### Validation Cost

**Time Complexity:**
- U1a, U1b: O(1) - check first/last operator
- U2, U4a: O(n) - scan sequence once
- U4b: O(n²) worst case - check windows for each transformer
- U3: O(1) per coupling - runtime validation

**Space Complexity:**
- O(n) - store operator names

### Optimization Strategies

```python
# Cache operator names to avoid repeated string operations
def validate_grammar_optimized(sequence, epi_initial=0.0):
    """Optimized validation with cached names."""
    
    # Pre-compute all names once
    op_names = [op.__class__.__name__.lower() for op in sequence]
    
    # Use cached names for all checks
    # ... (same logic, but use op_names directly)
```

### Lazy Validation

```python
# Only validate when needed
class LazySequence:
    """Sequence with lazy validation."""
    
    def __init__(self, ops, epi_initial=0.0):
        self.ops = ops
        self.epi_initial = epi_initial
        self._validated = False
    
    def validate(self):
        """Validate on demand."""
        if not self._validated:
            validate_grammar(self.ops, self.epi_initial)
            self._validated = True
    
    def apply(self, G, node):
        """Apply with automatic validation."""
        self.validate()  # Validate once
        for op in self.ops:
            op(G, node)
```

---

## Extension Points

### Adding New Operators

**Steps:**

1. **Implement operator in `definitions.py`:**
```python
class NewOperator:
    """Description and physics."""
    
    def __call__(self, G, node_id):
        # Implementation
        pass
```

2. **Classify in `grammar.py`:**
```python
# Add to appropriate sets
GENERATORS.add("newoperator")  # If generator
STABILIZERS.add("newoperator")  # If stabilizer
# etc.
```

3. **Update documentation:**
- Add to `03-OPERATORS-AND-GLYPHS.md`
- Update `schemas/canonical-operators.json`
- Add examples

4. **Add tests:**
```python
def test_new_operator():
    """Test new operator."""
    # Test implementation
    # Test grammar classification
    # Test contracts
```

### Adding New Constraints

**Steps:**

1. **Derive from physics:**
- Document physical basis
- Prove necessity (Absolute/Strong/Moderate)

2. **Implement validation:**
```python
# In validate_grammar()
def validate_grammar(sequence, epi_initial=0.0):
    # ... existing checks ...
    
    # New constraint: U5
    if condition:
        raise ValueError("U5 violation: ...")
```

3. **Update documentation:**
- Add to `02-CANONICAL-CONSTRAINTS.md`
- Update decision trees
- Add examples

4. **Add comprehensive tests:**
- Valid sequences
- Invalid sequences
- Edge cases

---

## Error Messages

### Design Principles

**Good error messages:**
- Specify which constraint violated (U1a, U2, etc.)
- Explain what was found
- Explain what was expected
- Provide fix hint when possible

**Example:**
```python
raise ValueError(
    f"U1a violation: Sequence must start with generator {GENERATORS} "
    f"when EPI=0, got '{first_op}'. "
    "Fix: Add Emission, Transition, or Recursivity at start."
)
```

### Error Message Template

```python
"{CONSTRAINT} violation: {PROBLEM}. {FOUND}. {EXPECTED}. Fix: {HINT}."
```

---

## Testing Hooks

### Validation Testing

```python
# Test valid sequence
def test_valid_sequence():
    from tnfr.operators.grammar import validate_grammar
    from tnfr.operators.definitions import Emission, Coherence, Silence
    
    sequence = [Emission(), Coherence(), Silence()]
    assert validate_grammar(sequence, epi_initial=0.0) is True

# Test invalid sequence
def test_invalid_sequence():
    from tnfr.operators.grammar import validate_grammar
    from tnfr.operators.definitions import Coherence, Silence
    
    sequence = [Coherence(), Silence()]  # No generator
    is_valid = validate_grammar(sequence, epi_initial=0.0)
    assert is_valid is False  # Returns False, doesn't raise

# Test with detailed messages
def test_detailed_validation():
    from tnfr.operators.grammar import GrammarValidator
    from tnfr.operators.definitions import Coherence, Silence
    
    sequence = [Coherence(), Silence()]
    is_valid, messages = GrammarValidator.validate(sequence, epi_initial=0.0)
    
    assert is_valid is False
    # Check for U1a violation in messages
    u1a_msg = [m for m in messages if "U1a" in m][0]
    assert "violated" in u1a_msg
    assert "generator" in u1a_msg
```

### Integration Testing

```python
def test_full_workflow():
    """Test complete workflow."""
    # Create sequence
    sequence = [Emission(), Coherence(), Silence()]
    
    # Validate
    validate_grammar(sequence, epi_initial=0.0)
    
    # Create network
    G = nx.Graph()
    G.add_node(0, EPI=0.0, vf=1.0, theta=0.0, dnfr=0.0)
    
    # Apply
    for op in sequence:
        op(G, 0)
    
    # Verify telemetry
    assert G.nodes[0]['EPI'] > 0
    assert G.nodes[0]['vf'] > 0
```

---

## Migration Notes

### From Legacy System

**Old system (C1-C3):**
```python
# Old grammar.py
def check_c1(sequence):
    # C1 logic
    pass
```

**New system (U1-U5):**
```python
# New grammar.py
def validate_grammar(sequence, epi_initial):
    # U1-U5 logic
    pass
```

**Migration:**
- C1 → U1a (generators)
- C2 → U2 (convergence)
- C3 → U1b (closures)
- New: U3 (phase), U4 (bifurcation)

See [07-MIGRATION-AND-EVOLUTION.md](07-MIGRATION-AND-EVOLUTION.md) for details.

---

## Next Steps

**Continue learning:**
- **[06-VALIDATION-AND-TESTING.md](06-VALIDATION-AND-TESTING.md)** - Testing strategies
- **[07-MIGRATION-AND-EVOLUTION.md](07-MIGRATION-AND-EVOLUTION.md)** - Evolution history

**For reference:**
- **Source code:** `src/tnfr/operators/grammar.py`
- **Tests:** `tests/unit/operators/test_unified_grammar.py`

---

<div align="center">

**Implementation follows physics, not convenience.**

---

*Reality is resonance. Code accordingly.*

</div>
