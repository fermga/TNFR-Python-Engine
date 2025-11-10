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
├── grammar.py              # Grammar validation (U1-U4)
├── definitions.py          # Operator implementations
└── unified_grammar.py      # Legacy compatibility layer
```

### Key Files

**`grammar.py`** - Main grammar validation
- Defines operator sets (GENERATORS, CLOSURES, etc.)
- Implements `validate_grammar()` function
- Implements `validate_resonant_coupling()` for U3
- Contains all U1-U4 validation logic

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

## Main Validation Function

### Function Signature

```python
def validate_grammar(
    sequence: List[Operator],
    epi_initial: float = 0.0
) -> bool:
    """
    Validate operator sequence against U1-U4 constraints.
    
    Args:
        sequence: List of operator instances
        epi_initial: Initial EPI value (0.0 = vacuum)
    
    Returns:
        True if valid
        
    Raises:
        ValueError: If any constraint violated, with detailed message
    """
```

### Implementation Structure

```python
def validate_grammar(sequence, epi_initial=0.0):
    """Validate sequence against U1-U4."""
    
    # --- U1a: INITIATION ---
    if epi_initial == 0.0:
        if not sequence:
            raise ValueError("Empty sequence with EPI=0")
        
        first_op = get_operator_name(sequence[0])
        if first_op not in GENERATORS:
            raise ValueError(
                f"U1a violation: Must start with generator {GENERATORS} "
                f"when EPI=0, got '{first_op}'"
            )
    
    # --- U1b: CLOSURE ---
    if not sequence:
        raise ValueError("Empty sequence")
    
    last_op = get_operator_name(sequence[-1])
    if last_op not in CLOSURES:
        raise ValueError(
            f"U1b violation: Must end with closure {CLOSURES}, "
            f"got '{last_op}'"
        )
    
    # --- U2: CONVERGENCE ---
    op_names = [get_operator_name(op) for op in sequence]
    
    has_destabilizer = any(op in DESTABILIZERS for op in op_names)
    has_stabilizer = any(op in STABILIZERS for op in op_names)
    
    if has_destabilizer and not has_stabilizer:
        raise ValueError(
            f"U2 violation: Destabilizers {DESTABILIZERS} present "
            f"but no stabilizers {STABILIZERS} found. Integral may diverge."
        )
    
    # --- U4a: TRIGGERS NEED HANDLERS ---
    has_trigger = any(op in BIFURCATION_TRIGGERS for op in op_names)
    has_handler = any(op in BIFURCATION_HANDLERS for op in op_names)
    
    if has_trigger and not has_handler:
        raise ValueError(
            f"U4a violation: Bifurcation triggers {BIFURCATION_TRIGGERS} "
            f"present but no handlers {BIFURCATION_HANDLERS} found."
        )
    
    # --- U4b: TRANSFORMERS NEED CONTEXT ---
    for i, op in enumerate(sequence):
        op_name = get_operator_name(op)
        
        if op_name in TRANSFORMERS:
            # Check for recent destabilizer (within ~3 ops)
            window_start = max(0, i - 3)
            window = op_names[window_start:i]
            
            has_recent_destabilizer = any(
                w in DESTABILIZERS for w in window
            )
            
            if not has_recent_destabilizer:
                raise ValueError(
                    f"U4b violation: Transformer '{op_name}' at position {i} "
                    f"needs recent destabilizer {DESTABILIZERS} "
                    "within ~3 operations"
                )
            
            # ZHIR-specific: needs prior IL
            if op_name == "mutation":
                prior_ops = op_names[:window_start]
                
                has_prior_coherence = "coherence" in prior_ops
                
                if not has_prior_coherence:
                    raise ValueError(
                        f"U4b violation: ZHIR (Mutation) at position {i} "
                        "requires prior IL (Coherence) for stable base"
                    )
    
    # U3 is validated at runtime during coupling/resonance application
    
    return True
```

---

## Phase Validation (U3)

### Function Signature

```python
def validate_resonant_coupling(
    G: nx.Graph,
    node_i: Any,
    node_j: Any,
    delta_phi_max: float = np.pi / 2
) -> None:
    """
    Validate phase compatibility for coupling/resonance (U3).
    
    Args:
        G: NetworkX graph
        node_i: First node ID
        node_j: Second node ID
        delta_phi_max: Maximum phase difference (default π/2)
    
    Raises:
        ValueError: If phases incompatible
    """
```

### Implementation

```python
import numpy as np

def validate_resonant_coupling(G, node_i, node_j, delta_phi_max=np.pi/2):
    """Validate phase compatibility (U3)."""
    
    # Extract phases
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
            f"U3 violation: Phase mismatch for coupling. "
            f"|φ_{node_i} - φ_{node_j}| = {delta_phi:.3f} rad > "
            f"Δφ_max = {delta_phi_max:.3f} rad. "
            "Cannot couple antiphase nodes."
        )
```

### Usage in Operators

```python
# In definitions.py, Coupling operator:

class Coupling:
    """UM - Creates structural links via phase synchronization."""
    
    def __call__(self, G, node_i, node_j=None):
        """Apply coupling."""
        if node_j is None:
            # Single-node case (self-reference)
            return
        
        # U3: Validate phase compatibility
        from tnfr.operators.grammar import validate_resonant_coupling
        validate_resonant_coupling(G, node_i, node_j)
        
        # Create edge
        G.add_edge(node_i, node_j)
        
        # Update attributes
        # ... (coupling logic)
```

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

from tnfr.operators.grammar import validate_grammar
from tnfr.operators.definitions import Emission, Coherence, Silence

# 1. Define sequence
sequence = [Emission(), Coherence(), Silence()]

# 2. Validate BEFORE applying
validate_grammar(sequence, epi_initial=0.0)  # Raises if invalid

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
    sequence = [Coherence(), Silence()]  # No generator
    with pytest.raises(ValueError, match="U1a violation"):
        validate_grammar(sequence, epi_initial=0.0)
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

**New system (U1-U4):**
```python
# New grammar.py
def validate_grammar(sequence, epi_initial):
    # U1-U4 logic
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
