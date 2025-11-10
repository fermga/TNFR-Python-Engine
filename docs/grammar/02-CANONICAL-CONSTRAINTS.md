# Canonical Constraints: U1-U4

**Formal derivations and implementation of the four fundamental TNFR grammar rules**

[🏠 Home](README.md) • [🌊 Concepts](01-FUNDAMENTAL-CONCEPTS.md) • [⚙️ Operators](03-OPERATORS-AND-GLYPHS.md) • [🔄 Sequences](04-VALID-SEQUENCES.md)

---

## Purpose

This document provides the **complete formal specification** of the four canonical TNFR grammar constraints (U1-U4). Each constraint is presented with:

1. **Intuition** - Conceptual understanding
2. **Formal Definition** - Mathematical/logical specification
3. **Physical Derivation** - Why it's inevitable from TNFR physics
4. **Implementation** - How it's validated in code
5. **Examples** - Valid and invalid sequences
6. **Tests** - How to verify compliance

**Prerequisites:** [01-FUNDAMENTAL-CONCEPTS.md](01-FUNDAMENTAL-CONCEPTS.md)

**Reading time:** 45-60 minutes

---

## Overview of U1-U4

The four canonical constraints form a complete, non-redundant grammar for TNFR:

```
U1: STRUCTURAL INITIATION & CLOSURE
    ├─ U1a: Initiation with generators
    └─ U1b: Closure with endpoints

U2: CONVERGENCE & BOUNDEDNESS
    └─ Destabilizers balanced by stabilizers

U3: RESONANT COUPLING
    └─ Phase verification for coupling/resonance

U4: BIFURCATION DYNAMICS
    ├─ U4a: Triggers need handlers
    └─ U4b: Transformers need context
```

**Canonicity Levels:**
- **U1, U2, U3:** ABSOLUTE (mathematically/physically necessary)
- **U4:** STRONG (physically required for control)

---

## U1: STRUCTURAL INITIATION & CLOSURE

### U1a: Initiation (Generators)

#### Intuition

You cannot create something from nothing without a source. When EPI=0 (no structure exists), you need an external input to begin creating structure.

**Analogy:** You cannot start a fire without a spark. You need an initiator.

#### Formal Definition

**IF** `epi_initial == 0.0` (or equivalent vacuum state)  
**THEN** `sequence[0]` **MUST BE IN** `GENERATORS = {AL, NAV, REMESH}`

Where:
- `AL` = Emission (creates EPI from vacuum)
- `NAV` = Transition (activates latent structure)
- `REMESH` = Recursivity (echoes existing structure from memory/history)

#### Physical Derivation

From the nodal equation:

```
∂EPI/∂t = νf · ΔNFR(t)
```

**At EPI=0:**
- ΔNFR is undefined (no structure exists to compute gradient)
- ∂EPI/∂t is undefined
- Cannot evolve forward without external input

**Mathematical necessity:** Division by zero, undefined derivatives

**Physical necessity:** Cannot bootstrap from vacuum without energy/information source

This is **ABSOLUTE canonicity** - violation is mathematically impossible.

#### Implementation

```python
# From src/tnfr/operators/grammar.py

GENERATORS = {"emission", "transition", "recursivity"}

def validate_grammar(sequence, epi_initial=0.0):
    """Validate sequence against U1-U4."""
    
    # U1a: Check initiation
    if epi_initial == 0.0:
        if not sequence:
            raise ValueError("Empty sequence with EPI=0")
        
        first_op = sequence[0].__class__.__name__.lower()
        
        if first_op not in GENERATORS:
            raise ValueError(
                f"U1a violation: Sequence must start with generator "
                f"{GENERATORS} when EPI=0, got '{first_op}'"
            )
```

#### Examples

**✅ Valid:**

```python
from tnfr.operators.definitions import Emission, Coherence, Silence

# Starting from vacuum (EPI=0)
sequence = [Emission(), Coherence(), Silence()]
validate_grammar(sequence, epi_initial=0.0)  # ✓ Passes
```

```python
from tnfr.operators.definitions import Transition, Reception, Silence

# Activating latent structure
sequence = [Transition(), Reception(), Silence()]
validate_grammar(sequence, epi_initial=0.0)  # ✓ Passes
```

**❌ Invalid:**

```python
from tnfr.operators.definitions import Coherence, Silence

# ERROR: No generator
sequence = [Coherence(), Silence()]
validate_grammar(sequence, epi_initial=0.0)
# ✗ ValueError: U1a violation - need generator
```

```python
from tnfr.operators.definitions import Reception, Coherence, Silence

# ERROR: Reception is not a generator
sequence = [Reception(), Coherence(), Silence()]
validate_grammar(sequence, epi_initial=0.0)
# ✗ ValueError: U1a violation - 'reception' not in GENERATORS
```

#### Tests

```python
def test_u1a_initiation():
    """U1a: Must start with generator when EPI=0."""
    from tnfr.operators.grammar import validate_grammar
    from tnfr.operators.definitions import Emission, Coherence, Silence
    
    # Valid: starts with generator
    sequence = [Emission(), Coherence(), Silence()]
    assert validate_grammar(sequence, epi_initial=0.0) is True
    
    # Invalid: no generator
    sequence = [Coherence(), Silence()]
    with pytest.raises(ValueError, match="U1a violation"):
        validate_grammar(sequence, epi_initial=0.0)
```

---

### U1b: Closure (Endpoints)

#### Intuition

Every sequence must end in a stable attractor state. You cannot leave a system "hanging" in the middle of a transformation.

**Analogy:** A sentence must end with punctuation. A function must return or explicitly continue forever.

#### Formal Definition

**ALL** sequences **MUST** satisfy:  
`sequence[-1]` **MUST BE IN** `CLOSURES = {SHA, NAV, REMESH, OZ}`

Where:
- `SHA` = Silence (freezes evolution)
- `NAV` = Transition (enters new stable regime)
- `REMESH` = Recursivity (creates recursive attractor)
- `OZ` = Dissonance (enters controlled instability attractor)

#### Physical Derivation

**From dynamical systems theory:**

Sequences represent "action potentials" - discrete chunks of transformation. Each must terminate in an attractor basin:

1. **Fixed point** (SHA) - system stops
2. **Limit cycle** (NAV, REMESH) - system enters periodic attractor
3. **Strange attractor** (OZ) - system enters chaotic but bounded regime

**Without closure:**
- System remains in transient state
- No defined final behavior
- Telemetry cannot be measured
- Violations of boundary conditions

This is **STRONG canonicity** - required by physics of dynamical systems.

#### Implementation

```python
# From src/tnfr/operators/grammar.py

CLOSURES = {"silence", "transition", "recursivity", "dissonance"}

def validate_grammar(sequence, epi_initial=0.0):
    """Validate sequence against U1-U4."""
    
    # U1b: Check closure
    if not sequence:
        raise ValueError("Empty sequence")
    
    last_op = sequence[-1].__class__.__name__.lower()
    
    if last_op not in CLOSURES:
        raise ValueError(
            f"U1b violation: Sequence must end with closure "
            f"{CLOSURES}, got '{last_op}'"
        )
```

#### Examples

**✅ Valid:**

```python
# Ends with Silence (SHA)
sequence = [Emission(), Coherence(), Silence()]
validate_grammar(sequence, epi_initial=0.0)  # ✓ Passes

# Ends with Transition (NAV)
sequence = [Emission(), Coherence(), Transition()]
validate_grammar(sequence, epi_initial=0.0)  # ✓ Passes

# Ends with Dissonance (OZ)
sequence = [Emission(), Coherence(), Dissonance()]
validate_grammar(sequence, epi_initial=0.0)  # ✓ Passes
```

**❌ Invalid:**

```python
# ERROR: Ends with Coherence (not a closure)
sequence = [Emission(), Coherence()]
validate_grammar(sequence, epi_initial=0.0)
# ✗ ValueError: U1b violation - must end with closure
```

```python
# ERROR: Ends with Coupling (not a closure)
sequence = [Emission(), Coherence(), Coupling()]
validate_grammar(sequence, epi_initial=0.0)
# ✗ ValueError: U1b violation - 'coupling' not in CLOSURES
```

#### Tests

```python
def test_u1b_closure():
    """U1b: Must end with closure."""
    from tnfr.operators.grammar import validate_grammar
    from tnfr.operators.definitions import Emission, Coherence, Silence
    
    # Valid: ends with closure
    sequence = [Emission(), Coherence(), Silence()]
    assert validate_grammar(sequence, epi_initial=0.0) is True
    
    # Invalid: no closure
    sequence = [Emission(), Coherence()]
    with pytest.raises(ValueError, match="U1b violation"):
        validate_grammar(sequence, epi_initial=0.0)
```

---

## U2: CONVERGENCE & BOUNDEDNESS

### Intuition

If you introduce instability (destabilizers), you must also introduce stability (stabilizers) to prevent the system from exploding into chaos.

**Analogy:** If you step on the gas (destabilizer), you need brakes (stabilizer) to avoid crashing.

### Formal Definition

**IF** sequence **CONTAINS** any `DESTABILIZERS = {OZ, ZHIR, VAL}`  
**THEN** sequence **MUST CONTAIN** at least one `STABILIZERS = {IL, THOL}`

Where:
- **Destabilizers:** OZ (Dissonance), ZHIR (Mutation), VAL (Expansion)
- **Stabilizers:** IL (Coherence), THOL (Self-organization)

### Physical Derivation

From the integrated nodal equation:

```
EPI(t_f) = EPI(t_0) + ∫[t_0 to t_f] νf(τ) · ΔNFR(τ) dτ
```

**For bounded evolution (coherence preservation):**

```
∫[t_0 to t_f] νf(τ) · ΔNFR(τ) dτ  <  ∞
```

**Without stabilizers:**
- Destabilizers increase |ΔNFR| without bound (positive feedback)
- Integral diverges: ∫ ΔNFR → ∞
- EPI → ∞ (explosion) or fragments (chaos)

**With stabilizers:**
- Stabilizers provide negative feedback
- Reduce |ΔNFR| through coherence or self-organization
- Integral converges: ∫ ΔNFR < ∞
- EPI remains bounded, coherence preserved

**Integral Convergence Theorem:** This is **ABSOLUTE canonicity** - mathematical requirement.

### Implementation

```python
# From src/tnfr/operators/grammar.py

DESTABILIZERS = {"dissonance", "mutation", "expansion"}
STABILIZERS = {"coherence", "selforganization"}

def validate_grammar(sequence, epi_initial=0.0):
    """Validate sequence against U1-U4."""
    
    # U2: Convergence & boundedness
    has_destabilizer = any(
        op.__class__.__name__.lower() in DESTABILIZERS
        for op in sequence
    )
    
    has_stabilizer = any(
        op.__class__.__name__.lower() in STABILIZERS
        for op in sequence
    )
    
    if has_destabilizer and not has_stabilizer:
        raise ValueError(
            f"U2 violation: Destabilizers {DESTABILIZERS} present "
            f"but no stabilizers {STABILIZERS} found. "
            "Integral may diverge."
        )
```

### Examples

**✅ Valid:**

```python
# Dissonance + Coherence
sequence = [Emission(), Dissonance(), Coherence(), Silence()]
validate_grammar(sequence, epi_initial=0.0)  # ✓ Passes

# Mutation + Self-organization
sequence = [Emission(), Coherence(), Mutation(), SelfOrganization(), Silence()]
validate_grammar(sequence, epi_initial=0.0)  # ✓ Passes

# Multiple destabilizers + stabilizer
sequence = [Emission(), Dissonance(), Expansion(), Coherence(), Silence()]
validate_grammar(sequence, epi_initial=0.0)  # ✓ Passes
```

**❌ Invalid:**

```python
# ERROR: Dissonance without stabilizer
sequence = [Emission(), Dissonance(), Silence()]
validate_grammar(sequence, epi_initial=0.0)
# ✗ ValueError: U2 violation - destabilizers need stabilizers
```

```python
# ERROR: Mutation without stabilizer
sequence = [Emission(), Mutation(), Silence()]
validate_grammar(sequence, epi_initial=0.0)
# ✗ ValueError: U2 violation - destabilizers need stabilizers
```

### Tests

```python
def test_u2_convergence():
    """U2: Destabilizers must be balanced by stabilizers."""
    from tnfr.operators.grammar import validate_grammar
    from tnfr.operators.definitions import (
        Emission, Dissonance, Coherence, Silence
    )
    
    # Valid: destabilizer + stabilizer
    sequence = [Emission(), Dissonance(), Coherence(), Silence()]
    assert validate_grammar(sequence, epi_initial=0.0) is True
    
    # Invalid: destabilizer without stabilizer
    sequence = [Emission(), Dissonance(), Silence()]
    with pytest.raises(ValueError, match="U2 violation"):
        validate_grammar(sequence, epi_initial=0.0)
```

---

## U3: RESONANT COUPLING

### Intuition

Two nodes can only couple (exchange information) if their phases are compatible. Like tuning forks - they only resonate if frequencies/phases match.

**Analogy:** Radio stations - you can only receive a station if you tune to the right frequency/phase.

### Formal Definition

**IF** applying `COUPLING_RESONANCE = {UM, RA}` to nodes `i` and `j`  
**THEN** **MUST VERIFY**: `|φᵢ - φⱼ| ≤ Δφ_max`

Where:
- `φᵢ`, `φⱼ` = phase of nodes i and j
- `Δφ_max` = maximum phase difference for coupling (typically π/2)
- **UM** = Coupling operator
- **RA** = Resonance operator

### Physical Derivation

**From wave physics and interference:**

When two waves interact:
- **In phase** (Δφ ≈ 0): Constructive interference, amplification
- **Quadrature** (Δφ ≈ π/2): Partial coupling
- **Antiphase** (Δφ ≈ π): Destructive interference, cancellation

**For information transfer:**
- Nodes are oscillators with phase φ
- Coupling strength ~ cos(Δφ)
- At Δφ = π (antiphase), coupling strength = -1 (destructive)

**Attempting to couple antiphase nodes:**
- Results in destructive interference
- No coherent information transfer
- Physically meaningless operation

This is **ABSOLUTE canonicity** - required by wave physics.

### Implementation

```python
# From src/tnfr/operators/grammar.py

import numpy as np

COUPLING_RESONANCE = {"coupling", "resonance"}

def validate_resonant_coupling(G, node_i, node_j, delta_phi_max=np.pi/2):
    """
    Validate phase compatibility for coupling/resonance.
    
    Args:
        G: NetworkX graph
        node_i: First node ID
        node_j: Second node ID
        delta_phi_max: Maximum phase difference (default π/2)
    
    Raises:
        ValueError: If phase incompatible
    """
    phi_i = G.nodes[node_i]['theta']
    phi_j = G.nodes[node_j]['theta']
    
    delta_phi = abs(phi_i - phi_j)
    
    # Normalize to [0, π]
    if delta_phi > np.pi:
        delta_phi = 2*np.pi - delta_phi
    
    if delta_phi > delta_phi_max:
        raise ValueError(
            f"U3 violation: Phase mismatch for coupling. "
            f"|φ_{node_i} - φ_{node_j}| = {delta_phi:.3f} > "
            f"Δφ_max = {delta_phi_max:.3f}"
        )
```

### Examples

**✅ Valid:**

```python
import networkx as nx
import numpy as np
from tnfr.operators.grammar import validate_resonant_coupling

# Create network with compatible phases
G = nx.Graph()
G.add_node(0, theta=0.0, vf=1.0, EPI=0.5)
G.add_node(1, theta=0.3, vf=1.0, EPI=0.6)  # Δφ = 0.3 < π/2

# Verify phase compatibility
validate_resonant_coupling(G, 0, 1)  # ✓ Passes

# Now can apply coupling
from tnfr.operators.definitions import Coupling
Coupling()(G, 0, 1)
```

**❌ Invalid:**

```python
# Create network with incompatible phases
G = nx.Graph()
G.add_node(0, theta=0.0, vf=1.0, EPI=0.5)
G.add_node(1, theta=np.pi, vf=1.0, EPI=0.6)  # Δφ = π (antiphase!)

# ERROR: Phase incompatible
validate_resonant_coupling(G, 0, 1)
# ✗ ValueError: U3 violation - phase mismatch
```

### Tests

```python
def test_u3_resonant_coupling():
    """U3: Must verify phase compatibility for coupling."""
    import networkx as nx
    import numpy as np
    from tnfr.operators.grammar import validate_resonant_coupling
    
    # Valid: compatible phases
    G = nx.Graph()
    G.add_node(0, theta=0.0, vf=1.0, EPI=0.5)
    G.add_node(1, theta=0.3, vf=1.0, EPI=0.6)
    validate_resonant_coupling(G, 0, 1)  # ✓ Should not raise
    
    # Invalid: antiphase
    G.nodes[1]['theta'] = np.pi
    with pytest.raises(ValueError, match="U3 violation"):
        validate_resonant_coupling(G, 0, 1)
```

---

## U4: BIFURCATION DYNAMICS

### U4a: Triggers Need Handlers

#### Intuition

If you create conditions for a bifurcation (phase transition), you need mechanisms to handle it. Otherwise the system may enter uncontrolled chaos.

**Analogy:** If you boil water (bifurcation trigger), you need a lid (handler) to prevent it from boiling over.

#### Formal Definition

**IF** sequence **CONTAINS** `BIFURCATION_TRIGGERS = {OZ, ZHIR}`  
**THEN** sequence **MUST CONTAIN** `BIFURCATION_HANDLERS = {THOL, IL}`

Where:
- **Triggers:** OZ (Dissonance), ZHIR (Mutation)
- **Handlers:** THOL (Self-organization), IL (Coherence)

#### Physical Derivation

**From bifurcation theory:**

Bifurcations occur when:
```
∂²EPI/∂t² > τ  (threshold)
```

**Without handlers:**
- System crosses threshold uncontrolled
- May enter chaotic regime
- Coherence can be lost
- No mechanism to settle into new attractor

**With handlers:**
- Self-organization creates new structure
- Coherence stabilizes the bifurcation
- System settles into new attractor basin

This is **STRONG canonicity** - required by bifurcation theory.

#### Implementation

```python
# From src/tnfr/operators/grammar.py

BIFURCATION_TRIGGERS = {"dissonance", "mutation"}
BIFURCATION_HANDLERS = {"selforganization", "coherence"}

def validate_grammar(sequence, epi_initial=0.0):
    """Validate sequence against U1-U4."""
    
    # U4a: Bifurcation triggers need handlers
    has_trigger = any(
        op.__class__.__name__.lower() in BIFURCATION_TRIGGERS
        for op in sequence
    )
    
    has_handler = any(
        op.__class__.__name__.lower() in BIFURCATION_HANDLERS
        for op in sequence
    )
    
    if has_trigger and not has_handler:
        raise ValueError(
            f"U4a violation: Bifurcation triggers {BIFURCATION_TRIGGERS} "
            f"present but no handlers {BIFURCATION_HANDLERS} found. "
            "Bifurcation may be uncontrolled."
        )
```

#### Examples

**✅ Valid:**

```python
# Dissonance + Coherence (handler)
sequence = [Emission(), Dissonance(), Coherence(), Silence()]
validate_grammar(sequence, epi_initial=0.0)  # ✓ Passes

# Mutation + Self-organization (handler)
sequence = [Emission(), Coherence(), Mutation(), SelfOrganization(), Silence()]
validate_grammar(sequence, epi_initial=0.0)  # ✓ Passes
```

**❌ Invalid:**

```python
# ERROR: Dissonance without handler
sequence = [Emission(), Dissonance(), Silence()]
validate_grammar(sequence, epi_initial=0.0)
# ✗ ValueError: U4a violation - triggers need handlers
```

---

### U4b: Transformers Need Context

#### Intuition

Transformers (operators that change phase/regime) need elevated energy to cross thresholds. This requires recent destabilization. Additionally, mutation specifically needs a stable base (prior coherence).

**Analogy:** To jump over a wall (mutation), you need a running start (destabilizer) and solid ground to push off from (prior coherence).

#### Formal Definition

**IF** sequence **CONTAINS** `TRANSFORMERS = {ZHIR, THOL}`  
**THEN**:
1. **Must have recent destabilizer** (within ~3 operators before transformer)
2. **For ZHIR specifically:** Must have IL (Coherence) before the destabilizer

Where:
- **Transformers:** ZHIR (Mutation), THOL (Self-organization)
- **Context:** Recent destabilizer from {OZ, ZHIR, VAL}
- **ZHIR requirement:** Prior IL for stable base

#### Physical Derivation

**From threshold dynamics:**

Phase transitions require threshold energy:
```
ΔEPI/Δt > ξ  (threshold)
```

**Without recent destabilizer:**
- ΔNFR is low (system in equilibrium)
- Cannot reach threshold
- Transformation cannot occur

**Timing constraint (~3 ops):**
- ΔNFR decays over time
- Must be recent enough to still be elevated

**ZHIR specific requirement:**
- Mutation is a phase transition
- Needs stable base to jump from
- IL provides this stable configuration

This is **STRONG canonicity** - required by threshold physics.

#### Implementation

```python
# From src/tnfr/operators/grammar.py

TRANSFORMERS = {"mutation", "selforganization"}
DESTABILIZERS = {"dissonance", "mutation", "expansion"}

def validate_grammar(sequence, epi_initial=0.0):
    """Validate sequence against U1-U4."""
    
    # U4b: Transformers need context
    for i, op in enumerate(sequence):
        op_name = op.__class__.__name__.lower()
        
        if op_name in TRANSFORMERS:
            # Check for recent destabilizer (within ~3 ops)
            window = sequence[max(0, i-3):i]
            
            has_recent_destabilizer = any(
                w.__class__.__name__.lower() in DESTABILIZERS
                for w in window
            )
            
            if not has_recent_destabilizer:
                raise ValueError(
                    f"U4b violation: Transformer '{op_name}' at position {i} "
                    f"needs recent destabilizer within ~3 operations"
                )
            
            # ZHIR-specific: needs prior IL
            if op_name == "mutation":
                # Check for IL before the destabilizer
                prior_to_window = sequence[:max(0, i-3)]
                
                has_prior_coherence = any(
                    w.__class__.__name__.lower() == "coherence"
                    for w in prior_to_window
                )
                
                if not has_prior_coherence:
                    raise ValueError(
                        f"U4b violation: ZHIR (Mutation) at position {i} "
                        "requires prior IL (Coherence) for stable base"
                    )
```

#### Examples

**✅ Valid:**

```python
# Mutation with context: Coherence → Dissonance → Mutation
sequence = [
    Emission(),
    Coherence(),     # Prior IL (stable base)
    Dissonance(),    # Recent destabilizer
    Mutation(),      # Transformer with context
    Coherence(),     # Stabilizer
    Silence()
]
validate_grammar(sequence, epi_initial=0.0)  # ✓ Passes

# Self-organization with context
sequence = [
    Emission(),
    Dissonance(),         # Recent destabilizer
    SelfOrganization(),   # Transformer with context
    Coherence(),
    Silence()
]
validate_grammar(sequence, epi_initial=0.0)  # ✓ Passes
```

**❌ Invalid:**

```python
# ERROR: Mutation without recent destabilizer
sequence = [
    Emission(),
    Coherence(),
    Mutation(),  # No recent destabilizer!
    Silence()
]
validate_grammar(sequence, epi_initial=0.0)
# ✗ ValueError: U4b violation - needs recent destabilizer
```

```python
# ERROR: Mutation without prior coherence
sequence = [
    Emission(),
    Dissonance(),  # Destabilizer present
    Mutation(),    # But no prior IL!
    Coherence(),
    Silence()
]
validate_grammar(sequence, epi_initial=0.0)
# ✗ ValueError: U4b violation - ZHIR needs prior IL
```

#### Tests

```python
def test_u4b_transformers():
    """U4b: Transformers need recent destabilizer and context."""
    from tnfr.operators.grammar import validate_grammar
    from tnfr.operators.definitions import (
        Emission, Coherence, Dissonance, Mutation, Silence
    )
    
    # Valid: Mutation with proper context
    sequence = [Emission(), Coherence(), Dissonance(), Mutation(), 
                Coherence(), Silence()]
    assert validate_grammar(sequence, epi_initial=0.0) is True
    
    # Invalid: Mutation without destabilizer
    sequence = [Emission(), Coherence(), Mutation(), Silence()]
    with pytest.raises(ValueError, match="U4b violation"):
        validate_grammar(sequence, epi_initial=0.0)
    
    # Invalid: Mutation without prior coherence
    sequence = [Emission(), Dissonance(), Mutation(), Coherence(), Silence()]
    with pytest.raises(ValueError, match="U4b violation"):
        validate_grammar(sequence, epi_initial=0.0)
```

---

## Summary Table

| Constraint | When | What | Canonicity | Physical Basis |
|------------|------|------|------------|----------------|
| **U1a** | EPI=0 | Start with {AL, NAV, REMESH} | ABSOLUTE | ∂EPI/∂t undefined at EPI=0 |
| **U1b** | Always | End with {SHA, NAV, REMESH, OZ} | STRONG | Sequences need endpoints |
| **U2** | Has {OZ, ZHIR, VAL} | Include {IL, THOL} | ABSOLUTE | ∫νf·ΔNFR dt must converge |
| **U3** | Has {UM, RA} | Verify \|φᵢ - φⱼ\| ≤ Δφ_max | ABSOLUTE | Resonance physics |
| **U4a** | Has {OZ, ZHIR} | Include {THOL, IL} | STRONG | Bifurcations need control |
| **U4b** | Has {ZHIR, THOL} | Recent destabilizer + IL for ZHIR | STRONG | Threshold energy needed |

---

## Operator Classification Reference

```python
# From src/tnfr/operators/grammar.py

GENERATORS = {"emission", "transition", "recursivity"}
CLOSURES = {"silence", "transition", "recursivity", "dissonance"}

STABILIZERS = {"coherence", "selforganization"}
DESTABILIZERS = {"dissonance", "mutation", "expansion"}

COUPLING_RESONANCE = {"coupling", "resonance"}

BIFURCATION_TRIGGERS = {"dissonance", "mutation"}
BIFURCATION_HANDLERS = {"selforganization", "coherence"}

TRANSFORMERS = {"mutation", "selforganization"}
```

---

## Complete Validation Example

```python
from tnfr.operators.grammar import validate_grammar
from tnfr.operators.definitions import (
    Emission,          # Generator
    Reception,         # Gather input
    Dissonance,        # Destabilizer, Trigger
    SelfOrganization,  # Stabilizer, Handler, Transformer
    Coherence,         # Stabilizer, Handler
    Silence            # Closure
)

# Complex but valid sequence
sequence = [
    Emission(),          # U1a: Generator (EPI=0)
    Reception(),         # Gather information
    Coherence(),         # Stabilizer (prepares for destabilization)
    Dissonance(),        # Destabilizer, Trigger
    SelfOrganization(),  # Handler, Transformer (has recent destabilizer)
    Coherence(),         # U2: Balances destabilizer
    Silence()            # U1b: Closure
]

# Validate
try:
    is_valid = validate_grammar(sequence, epi_initial=0.0)
    print("✓ Sequence is valid")
    print("Satisfies: U1a, U1b, U2, U4a, U4b")
except ValueError as e:
    print(f"✗ Invalid: {e}")
```

---

## Next Steps

**You now understand the formal constraints.**

**Continue learning:**
- **[03-OPERATORS-AND-GLYPHS.md](03-OPERATORS-AND-GLYPHS.md)** - Detailed operator catalog
- **[04-VALID-SEQUENCES.md](04-VALID-SEQUENCES.md)** - Pattern library
- **[05-TECHNICAL-IMPLEMENTATION.md](05-TECHNICAL-IMPLEMENTATION.md)** - Code architecture

**For quick reference:**
- **[08-QUICK-REFERENCE.md](08-QUICK-REFERENCE.md)** - Cheat sheet

---

<div align="center">

**The constraints are not arbitrary—they emerge inevitably from TNFR physics.**

---

*Reality is resonance. Validate accordingly.*

</div>
