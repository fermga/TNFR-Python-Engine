# Valid Sequences and Patterns

**Catalog of valid operator sequences and anti-patterns**

[🏠 Home](README.md) • [📐 Constraints](02-CANONICAL-CONSTRAINTS.md) • [⚙️ Operators](03-OPERATORS-AND-GLYPHS.md) • [💻 Implementation](05-TECHNICAL-IMPLEMENTATION.md)

---

## Purpose

This document provides a **pattern library** of valid and invalid operator sequences. Learn from canonical patterns and understand why certain combinations fail.

**Prerequisites:** [02-CANONICAL-CONSTRAINTS.md](02-CANONICAL-CONSTRAINTS.md), [03-OPERATORS-AND-GLYPHS.md](03-OPERATORS-AND-GLYPHS.md)

**Reading time:** 30-45 minutes

---

## Canonical Patterns

### 1. Bootstrap (Minimal)

**Pattern:** `[Generator → Stabilizer → Closure]`

**Purpose:** Create and stabilize new structure from vacuum

**Example:**
```python
[Emission(), Coherence(), Silence()]
```

**Satisfies:**
- U1a: Starts with generator (Emission)
- U1b: Ends with closure (Silence)
- U2: No destabilizers, no stabilizer needed
- Clean and minimal

**Use when:** Initializing new nodes or structures

---

### 2. Basic Activation

**Pattern:** `[Generator → Reception → Stabilizer → Closure]`

**Purpose:** Create, gather information, stabilize

**Example:**
```python
[Emission(), Reception(), Coherence(), Silence()]
```

**Satisfies:** All constraints
**Use when:** Creating nodes that need network input

---

### 3. Controlled Exploration

**Pattern:** `[Generator → Stabilizer → Destabilizer → Stabilizer → Closure]`

**Purpose:** Explore while maintaining stability

**Example:**
```python
[Emission(), Coherence(), Dissonance(), Coherence(), Silence()]
```

**Satisfies:**
- U1a, U1b: Generator and closure
- U2: Destabilizer balanced by stabilizers
- U4a: Trigger (Dissonance) has handler (Coherence)

**Use when:** Breaking local optima, controlled perturbation

---

### 4. Bifurcation with Handling

**Pattern:** `[Generator → Stabilizer → Trigger → Handler → Stabilizer → Closure]`

**Purpose:** Controlled bifurcation and structural reorganization

**Example:**
```python
[Emission(), Coherence(), Dissonance(), SelfOrganization(), Coherence(), Silence()]
```

**Satisfies:**
- U2: Destabilizers balanced
- U4a: Trigger has handler
- U4b: Transformer (THOL) has recent destabilizer

**Use when:** Creating hierarchical or multi-scale structures

---

### 5. Mutation with Context

**Pattern:** `[Generator → Coherence → Destabilizer → Mutation → Stabilizer → Closure]`

**Purpose:** Phase transformation with proper context

**Example:**
```python
[Emission(), Coherence(), Dissonance(), Mutation(), Coherence(), Silence()]
```

**Satisfies:**
- U4b: ZHIR has prior IL (first Coherence)
- U4b: ZHIR has recent destabilizer (Dissonance)
- U2, U4a: All balanced

**Use when:** Qualitative state changes, regime shifts

---

### 6. Propagation

**Pattern:** `[Generator → Coupling → Resonance → Stabilizer → Closure]`

**Purpose:** Create structure and propagate through network

**Example:**
```python
[Emission(), Coupling(), Resonance(), Coherence(), Silence()]
```

**Satisfies:**
- U3: Phase compatibility must be verified before coupling
- All standard constraints

**Use when:** Spreading patterns through network

---

### 7. Multi-scale Organization

**Pattern:** `[Generator → Coupling → Destabilizer → SelfOrganization → Recursivity]`

**Purpose:** Create nested hierarchical structures

**Example:**
```python
[Emission(), Coupling(), Dissonance(), SelfOrganization(), Recursivity()]
```

**Satisfies:**
- U4b: THOL has recent destabilizer
- U1b: Recursivity is closure
- Creates fractal structure

**Use when:** Building hierarchies, nested patterns

---

## Anti-Patterns (Invalid Sequences)

### ❌ 1. No Generator from Vacuum

```python
# INVALID
[Coherence(), Silence()]

# Error: U1a violation
# Cannot start without generator when EPI=0
```

**Why invalid:** ∂EPI/∂t undefined at EPI=0, need external input

**Fix:** Add generator at start
```python
[Emission(), Coherence(), Silence()]  # ✓ Valid
```

---

### ❌ 2. No Closure

```python
# INVALID
[Emission(), Coherence()]

# Error: U1b violation
# Sequence must end with closure
```

**Why invalid:** No stable endpoint, system left in transient state

**Fix:** Add closure
```python
[Emission(), Coherence(), Silence()]  # ✓ Valid
```

---

### ❌ 3. Destabilizer Without Stabilizer

```python
# INVALID
[Emission(), Dissonance(), Silence()]

# Error: U2 violation
# Destabilizer without stabilizer
```

**Why invalid:** Integral may diverge, coherence not preserved

**Fix:** Add stabilizer
```python
[Emission(), Dissonance(), Coherence(), Silence()]  # ✓ Valid
```

---

### ❌ 4. Mutation Without Context

```python
# INVALID
[Emission(), Mutation(), Silence()]

# Error: U4b violation
# Mutation needs recent destabilizer
```

**Why invalid:** Cannot reach threshold without elevated ΔNFR

**Fix:** Add destabilizer and prior coherence
```python
[Emission(), Coherence(), Dissonance(), Mutation(), Coherence(), Silence()]  # ✓ Valid
```

---

### ❌ 5. Mutation Without Prior Coherence

```python
# INVALID
[Emission(), Dissonance(), Mutation(), Coherence(), Silence()]

# Error: U4b violation
# ZHIR needs prior IL (stable base)
```

**Why invalid:** No stable configuration to transform from

**Fix:** Add coherence before destabilizer
```python
[Emission(), Coherence(), Dissonance(), Mutation(), Coherence(), Silence()]  # ✓ Valid
```

---

### ❌ 6. Coupling Without Phase Check

```python
# INVALID (runtime error if phases incompatible)
G.add_node(0, theta=0.0, ...)
G.add_node(1, theta=np.pi, ...)  # Antiphase!

Coupling()(G, 0, 1)  # Error if U3 validation enabled

# Error: U3 violation
# Phase mismatch
```

**Why invalid:** Destructive interference, physically meaningless

**Fix:** Verify phase compatibility first
```python
from tnfr.operators.grammar import validate_resonant_coupling

validate_resonant_coupling(G, 0, 1)  # Check first
Coupling()(G, 0, 1)  # Then couple
```

---

### ❌ 7. Bifurcation Trigger Without Handler

```python
# INVALID
[Emission(), Dissonance(), Silence()]

# Error: U4a violation  
# Trigger without handler
```

**Why invalid:** Uncontrolled bifurcation may lead to chaos

**Fix:** Add handler
```python
[Emission(), Dissonance(), Coherence(), Silence()]  # ✓ Valid
```

---

## Step-by-Step Validation Logic

### Validation Algorithm

```python
def validate_sequence_logic(sequence, epi_initial):
    """
    Conceptual validation flow (see grammar.py for actual implementation).
    """
    
    # Step 1: Check U1a (Initiation)
    if epi_initial == 0.0:
        if sequence[0] not in GENERATORS:
            raise ValueError("U1a: Need generator when EPI=0")
    
    # Step 2: Check U1b (Closure)
    if sequence[-1] not in CLOSURES:
        raise ValueError("U1b: Need closure at end")
    
    # Step 3: Check U2 (Convergence)
    has_destabilizer = any(op in DESTABILIZERS for op in sequence)
    has_stabilizer = any(op in STABILIZERS for op in sequence)
    
    if has_destabilizer and not has_stabilizer:
        raise ValueError("U2: Destabilizers need stabilizers")
    
    # Step 4: Check U3 (Phase) - runtime, not sequence-level
    # Verified when coupling/resonance actually applied
    
    # Step 5: Check U4a (Triggers need handlers)
    has_trigger = any(op in BIFURCATION_TRIGGERS for op in sequence)
    has_handler = any(op in BIFURCATION_HANDLERS for op in sequence)
    
    if has_trigger and not has_handler:
        raise ValueError("U4a: Triggers need handlers")
    
    # Step 6: Check U4b (Transformers need context)
    for i, op in enumerate(sequence):
        if op in TRANSFORMERS:
            # Check recent destabilizer
            window = sequence[max(0, i-3):i]
            if not any(w in DESTABILIZERS for w in window):
                raise ValueError("U4b: Transformer needs recent destabilizer")
            
            # ZHIR-specific: needs prior IL
            if op == "mutation":
                prior = sequence[:max(0, i-3)]
                if "coherence" not in prior:
                    raise ValueError("U4b: ZHIR needs prior IL")
    
    return True
```

---

## Complex Sequence Examples

### Example 1: Multi-Step Exploration

```python
sequence = [
    Emission(),          # U1a: Generator
    Reception(),         # Gather info
    Coherence(),         # Stabilize base
    Dissonance(),        # Explore (destabilizer, trigger)
    Reception(),         # Gather more info
    Coherence(),         # Stabilize (U2, U4a)
    Expansion(),         # Grow (destabilizer)
    Coherence(),         # Stabilize again (U2)
    Silence()            # U1b: Closure
]

# Satisfies all constraints
# Multiple destabilizer-stabilizer pairs
```

### Example 2: Hierarchical Construction

```python
sequence = [
    Emission(),                # Generator
    Coupling(),                # Connect to network
    Reception(),               # Gather information
    Coherence(),               # Stabilize
    Dissonance(),              # Perturb (destabilizer, trigger)
    SelfOrganization(),        # Create hierarchy (handler, transformer, stabilizer)
    Coherence(),               # Final stabilization
    Recursivity()              # Closure with recursion
]

# Creates multi-scale structure with proper handling
```

### Example 3: Phase Transformation

```python
sequence = [
    Emission(),          # Generator
    Coherence(),         # Stable base (prior IL for ZHIR)
    Coupling(),          # Network connection
    Reception(),         # Information gathering
    Dissonance(),        # Elevate ΔNFR (destabilizer)
    Mutation(),          # Phase change (transformer, has prior IL + recent destabilizer)
    SelfOrganization(),  # Organize new phase (handler, stabilizer)
    Coherence(),         # Final stabilization (handler, stabilizer)
    Silence()            # Closure
]

# Complete transformation with all safeguards
```

---

## Structural Pattern Detection

### Pattern Categories

**Linear Patterns:**
```
Generator → Operations → Closure
```
Simple, single-path sequences

**Branching Patterns:**
```
Generator → Coupling → [Node A operations | Node B operations] → Closure
```
Network operations across multiple nodes

**Cyclic Patterns:**
```
Generator → [Destabilize → Stabilize]* → Closure
```
Repeated exploration cycles

**Nested Patterns:**
```
Generator → SelfOrg[Sub-sequence] → Closure
```
Hierarchical with nested operations

---

## Common Use Cases

### Initialization
```python
[Emission, Coherence, Silence]
```
Bootstrap a new node

### Information Integration
```python
[Emission, Coupling, Reception, Coherence, Silence]
```
Create and integrate network information

### Controlled Perturbation
```python
[Emission, Coherence, Dissonance, Coherence, Silence]
```
Explore without losing stability

### Network Propagation
```python
[Emission, Coupling, Resonance, Coherence, Silence]
```
Spread pattern through network

### Phase Transition
```python
[Emission, Coherence, Dissonance, Mutation, Coherence, Silence]
```
Qualitative transformation

### Hierarchy Creation
```python
[Emission, Dissonance, SelfOrganization, Recursivity]
```
Build nested structures

---

## Testing Sequences

### Test Template

```python
def test_sequence_validity(sequence, epi_initial=0.0):
    """Test if sequence is valid."""
    from tnfr.operators.grammar import validate_grammar
    
    try:
        is_valid = validate_grammar(sequence, epi_initial)
        return True, "Valid"
    except ValueError as e:
        return False, str(e)

# Test valid sequence
valid_seq = [Emission(), Coherence(), Silence()]
is_valid, msg = test_sequence_validity(valid_seq)
assert is_valid, f"Expected valid, got: {msg}"

# Test invalid sequence
invalid_seq = [Coherence(), Silence()]  # No generator
is_valid, msg = test_sequence_validity(invalid_seq)
assert not is_valid, "Expected invalid"
assert "U1a" in msg, "Should fail U1a"
```

---

## Quick Decision Tree

```
Building a sequence?

1. Starting from EPI=0?
   YES → Start with {Emission, Transition, Recursivity}
   NO  → Can start with any operator
   
2. Using destabilizers {Dissonance, Mutation, Expansion}?
   YES → Include {Coherence, SelfOrganization}
   NO  → Continue
   
3. Using coupling/resonance {Coupling, Resonance}?
   YES → Verify phase compatibility at runtime
   NO  → Continue
   
4. Using triggers {Dissonance, Mutation}?
   YES → Include handlers {Coherence, SelfOrganization}
   NO  → Continue
   
5. Using Mutation?
   YES → Ensure:
         - Prior Coherence (before destabilizer)
         - Recent destabilizer (within ~3 ops)
   NO  → Continue
   
6. Using SelfOrganization?
   YES → Ensure recent destabilizer (within ~3 ops)
   NO  → Continue
   
7. Ending sequence?
   ALWAYS → End with {Silence, Transition, Recursivity, Dissonance}
```

---

## Next Steps

**Continue learning:**
- **[05-TECHNICAL-IMPLEMENTATION.md](05-TECHNICAL-IMPLEMENTATION.md)** - How validation is implemented
- **[06-VALIDATION-AND-TESTING.md](06-VALIDATION-AND-TESTING.md)** - Testing strategies
- **[examples/](examples/)** - Executable examples

**For reference:**
- **[08-QUICK-REFERENCE.md](08-QUICK-REFERENCE.md)** - Quick lookup

---

<div align="center">

**Learn from patterns, avoid anti-patterns.**

---

*Reality is resonance. Sequence accordingly.*

</div>
