# Canonical Constraints: U1-U6

**Formal derivations and implementation of the six fundamental TNFR grammar rules**

[🏠 Home](README.md) • [🌊 Concepts](01-FUNDAMENTAL-CONCEPTS.md) • [⚙️ Operators](03-OPERATORS-AND-GLYPHS.md) • [🔄 Sequences](04-VALID-SEQUENCES.md)

---

## Purpose

This document provides the **complete formal specification** of the six canonical TNFR grammar constraints (U1-U6). Each constraint is presented with:

1. **Intuition** - Conceptual understanding
2. **Formal Definition** - Mathematical/logical specification
3. **Physical Derivation** - Why it's inevitable from TNFR physics
4. **Implementation** - How it's validated in code
5. **Examples** - Valid and invalid sequences
6. **Tests** - How to verify compliance

**Prerequisites:** [01-FUNDAMENTAL-CONCEPTS.md](01-FUNDAMENTAL-CONCEPTS.md)

**Reading time:** 45-60 minutes

---

## Overview of U1-U6

The six canonical constraints form a complete, non-redundant grammar for TNFR (temporal + spatial + multi-scale):

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

U5: MULTI-SCALE COHERENCE
    └─ Hierarchical REMESH (depth>1) requires scale stabilizers (IL / THOL)
       to conserve coherence across nested EPIs (C_parent ≥ α·ΣC_child)

U6: STRUCTURAL POTENTIAL CONFINEMENT
    └─ Monitor Δ Φ_s < 2.0 (escape threshold)
       Telemetry-based safety criterion for structural stability
```

**Canonicity Levels:**
- **U1, U2, U3, U5:** ABSOLUTE (mathematically/physically necessary)
- **U4, U6:** STRONG (extensive empirical validation, universal across topologies)

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
    """Validate sequence against U1-U5."""
    
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

#### Anti-Patterns

**⚠️ Common Mistakes:**

1. **Forgetting generator when reusing sequences**
   ```python
   # BAD: Reusing a subsequence without checking context
   def my_sequence():
       return [Coherence(), Silence()]  # Missing generator!
   
   # GOOD: Always start with generator if EPI could be 0
   def my_sequence():
       return [Emission(), Coherence(), Silence()]
   ```

2. **Assuming EPI exists**
   ```python
   # BAD: No check for initial state
   sequence = [Reception(), Coherence(), Silence()]
   
   # GOOD: Use generator or verify EPI > 0
   if epi_initial == 0.0:
       sequence = [Emission(), Reception(), Coherence(), Silence()]
   else:
       sequence = [Reception(), Coherence(), Silence()]
   ```

3. **Using Reception as initiator**
   - Reception gathers existing EPI, cannot create it
   - Only {AL, NAV, REMESH} can generate from vacuum

#### Tests

**Implementation**: `src/tnfr/operators/grammar.py::GrammarValidator.validate_initiation()`

**Test Suite**:
- `tests/unit/operators/test_unified_grammar.py::TestU1Initiation`
- `tests/integration/test_mutation_sequences.py::test_u1a_satisfied_with_emission`

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

**Related Documentation**:
- [UNIFIED_GRAMMAR_RULES.md § U1a](https://github.com/fermga/TNFR-Python-Engine/blob/main/UNIFIED_GRAMMAR_RULES.md#u1-structural-initiation--closure)
- [AGENTS.md § Invariant #1](https://github.com/fermga/TNFR-Python-Engine/blob/main/AGENTS.md#canonical-invariants-never-break)

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
    """Validate sequence against U1-U5."""
    
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

#### Anti-Patterns

**⚠️ Common Mistakes:**

1. **Ending with Coherence**
   ```python
   # BAD: Coherence is not a closure operator
   sequence = [Emission(), Reception(), Coherence()]
   # ✗ System left in transient state
   
   # GOOD: Add closure after coherence
   sequence = [Emission(), Reception(), Coherence(), Silence()]
   ```

2. **Ending with data gathering operations**
   ```python
   # BAD: Reception doesn't stabilize endpoint
   sequence = [Emission(), Coherence(), Reception()]
   
   # GOOD: Close with attractor state
   sequence = [Emission(), Coherence(), Reception(), Silence()]
   ```

3. **Confusing closure with stabilization**
   - Coherence (IL) **stabilizes** but doesn't close
   - Silence (SHA) both stabilizes **and closes**
   - Not all stabilizers are closures

#### Tests

**Implementation**: `src/tnfr/operators/grammar.py::GrammarValidator.validate_closure()`

**Test Suite**:
- `tests/unit/operators/test_unified_grammar.py::TestU1Closure`
- `tests/integration/test_mutation_sequences.py::test_u1b_closure_satisfied`
- `tests/unit/operators/test_remesh_operator_integration.py::test_remesh_as_closure_U1b`

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

**Related Documentation**:
- [UNIFIED_GRAMMAR_RULES.md § U1b](https://github.com/fermga/TNFR-Python-Engine/blob/main/UNIFIED_GRAMMAR_RULES.md#u1-structural-initiation--closure)
- [AGENTS.md § Operator Closure Invariant #4](https://github.com/fermga/TNFR-Python-Engine/blob/main/AGENTS.md#canonical-invariants-never-break)

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
    """Validate sequence against U1-U5."""
    
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

### Anti-Patterns

**⚠️ Common Mistakes:**

1. **"Masking" with weak stabilizers**
   ```python
   # QUESTIONABLE: Multiple destabilizers, single stabilizer
   sequence = [
       Emission(),
       Dissonance(),  # +ΔNFR
       Expansion(),   # ++ΔNFR
       Mutation(),    # +++ΔNFR
       Coherence(),   # -ΔNFR (may not be sufficient!)
       Silence()
   ]
   # Technically passes U2, but integral may still be large
   # Better: Add more stabilizers or reduce destabilizers
   ```

2. **Assuming order doesn't matter**
   ```python
   # BAD: Stabilizer before destabilizer provides no protection
   sequence = [Emission(), Coherence(), Dissonance(), Silence()]
   # Coherence has no effect on later dissonance
   
   # GOOD: Stabilizer after destabilizer bounds growth
   sequence = [Emission(), Dissonance(), Coherence(), Silence()]
   ```

3. **Ignoring accumulation effects**
   ```python
   # BAD: Long sequence of destabilizers with stabilizer at end
   sequence = [Emission(), Dissonance(), Dissonance(), 
               Expansion(), Mutation(), Coherence(), Silence()]
   # ΔNFR may diverge before coherence is applied
   
   # GOOD: Interleave stabilizers with destabilizers
   sequence = [Emission(), Dissonance(), Coherence(), 
               Expansion(), Coherence(), Mutation(), 
               Coherence(), Silence()]
   ```

### Tests

**Implementation**: `src/tnfr/operators/grammar.py::GrammarValidator.validate_convergence()`

**Test Suite**:
- `tests/unit/operators/test_unified_grammar.py::TestU2Convergence`
- `tests/integration/test_mutation_sequences.py::test_u2_satisfied_with_stabilizers`
- `tests/unit/operators/test_canonical_grammar_legacy.py::test_rc2_maps_to_u2`
- `tests/unit/operators/test_grammar_c1_c3_deprecation.py::test_validate_c2_boundedness_*`

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

**Related Documentation**:
- [UNIFIED_GRAMMAR_RULES.md § U2](https://github.com/fermga/TNFR-Python-Engine/blob/main/UNIFIED_GRAMMAR_RULES.md#u2-convergence--boundedness)
- [AGENTS.md § Convergence & Boundedness](https://github.com/fermga/TNFR-Python-Engine/blob/main/AGENTS.md)
- [TNFR_FORCES_EMERGENCE.md § Integrated Dynamics](../TNFR_FORCES_EMERGENCE.md)

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

### Anti-Patterns

**⚠️ Common Mistakes:**

1. **Coupling nodes without phase check**
   ```python
   # BAD: Direct coupling without verification
   import networkx as nx
   from tnfr.operators.definitions import Coupling
   
   G = nx.Graph()
   G.add_node(0, theta=0.0, vf=1.0, EPI=0.5)
   G.add_node(1, theta=3.0, vf=1.0, EPI=0.6)  # May be antiphase!
   
   Coupling()(G, 0, 1)  # ERROR: No phase check
   
   # GOOD: Verify phase compatibility first
   from tnfr.operators.grammar import validate_resonant_coupling
   validate_resonant_coupling(G, 0, 1)  # Raises if incompatible
   Coupling()(G, 0, 1)
   ```

2. **Assuming small phase differences are always OK**
   ```python
   # PROBLEMATIC: Phase difference near threshold
   G.nodes[0]['theta'] = 0.0
   G.nodes[1]['theta'] = 1.5  # Close to π/2 threshold
   
   # May pass but creates weak coupling
   # Better: Adjust phases or use different nodes
   ```

3. **Ignoring phase drift during sequences**
   ```python
   # BAD: Coupling after operators that change phase
   sequence = [
       Emission(),
       Mutation(),  # Changes theta!
       Coupling(),  # Phase may no longer be compatible
       Silence()
   ]
   
   # GOOD: Verify phase after transformations
   # Or: Couple before phase-changing operators
   ```

### Tests

**Implementation**: `src/tnfr/operators/grammar.py::GrammarValidator.validate_resonant_coupling()`

**Test Suite**:
- `tests/unit/operators/test_unified_grammar.py::TestU3ResonantCoupling`
- `tests/unit/operators/test_coupling_preconditions.py::test_um_phase_compatibility_*`
- `tests/unit/metrics/test_phase_compatibility.py::test_grammar_u3_compliance`
- `tests/unit/operators/test_canonical_grammar_legacy.py::test_rc3_maps_to_u3`

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

**Related Documentation**:
- [UNIFIED_GRAMMAR_RULES.md § U3](https://github.com/fermga/TNFR-Python-Engine/blob/main/UNIFIED_GRAMMAR_RULES.md#u3-resonant-coupling)
- [AGENTS.md § Invariant #5: Phase Verification](https://github.com/fermga/TNFR-Python-Engine/blob/main/AGENTS.md)
- [03-OPERATORS-AND-GLYPHS.md § Coupling (UM)](03-OPERATORS-AND-GLYPHS.md)

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
    """Validate sequence against U1-U5."""
    
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

#### Anti-Patterns

**⚠️ Common Mistakes:**

1. **Uncontrolled bifurcation cascades**
   ```python
   # BAD: Multiple triggers without handlers
   sequence = [
       Emission(),
       Dissonance(),  # Trigger 1
       Mutation(),    # Trigger 2 - bifurcation cascade!
       Silence()
   ]
   # System may enter chaotic regime
   
   # GOOD: Handler between triggers
   sequence = [
       Emission(),
       Dissonance(),      # Trigger 1
       Coherence(),       # Handler
       Mutation(),        # Trigger 2
       SelfOrganization(), # Handler
       Silence()
   ]
   ```

2. **Wrong handler for trigger type**
   ```python
   # SUBOPTIMAL: Coherence after Mutation
   # Mutation creates new structure, Self-organization better handles it
   sequence = [Emission(), Coherence(), Dissonance(), 
               Mutation(), Coherence(), Silence()]
   
   # BETTER: Self-organization after Mutation
   sequence = [Emission(), Coherence(), Dissonance(), 
               Mutation(), SelfOrganization(), Silence()]
   ```

3. **Assuming handler proximity doesn't matter**
   ```python
   # PROBLEMATIC: Handler too far from trigger
   sequence = [
       Emission(),
       Dissonance(),  # Trigger
       Reception(),
       Reception(),
       Reception(),
       Coherence(),  # Handler too late
       Silence()
   ]
   # Bifurcation may complete before handler acts
   ```

#### Tests

**Implementation**: `src/tnfr/operators/grammar.py::GrammarValidator.validate_bifurcation_triggers()`

**Test Suite**:
- `tests/unit/operators/test_unified_grammar.py::TestU4aBifurcationTriggers`
- `tests/unit/operators/test_controlled_bifurcation.py::test_multiple_bifurcations_*`
- `tests/unit/operators/test_bifurcation.py::test_bifurcation_above_threshold`

**Related Documentation**:
- [UNIFIED_GRAMMAR_RULES.md § U4a](https://github.com/fermga/TNFR-Python-Engine/blob/main/UNIFIED_GRAMMAR_RULES.md#u4-bifurcation-dynamics)
- [AGENTS.md § Contract OZ](https://github.com/fermga/TNFR-Python-Engine/blob/main/AGENTS.md)
- [03-OPERATORS-AND-GLYPHS.md § Dissonance (OZ)](03-OPERATORS-AND-GLYPHS.md)

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
    """Validate sequence against U1-U5."""
    
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

#### Anti-Patterns

**⚠️ Common Mistakes:**

1. **Transformer without sufficient energy**
   ```python
   # BAD: Mutation too soon after destabilizer
   sequence = [
       Emission(),
       Coherence(),
       Dissonance(),  # Destabilizer
       Reception(),   # ΔNFR starts decaying
       Reception(),   # More decay
       Mutation(),    # Insufficient ΔNFR for threshold!
       Silence()
   ]
   
   # GOOD: Mutation close to destabilizer
   sequence = [
       Emission(),
       Coherence(),
       Dissonance(),  # Destabilizer
       Mutation(),    # Within ~3 ops window
       Coherence(),
       Silence()
   ]
   ```

2. **ZHIR without stable base**
   ```python
   # BAD: Mutation without prior Coherence
   sequence = [
       Emission(),
       Dissonance(),  # Destabilizer present
       Mutation(),    # But no stable base!
       Coherence(),
       Silence()
   ]
   
   # GOOD: Coherence before destabilizer-transformer pair
   sequence = [
       Emission(),
       Coherence(),   # Stable base
       Dissonance(),  # Destabilizer
       Mutation(),    # Transformer
       Coherence(),
       Silence()
   ]
   ```

3. **Confusing context window**
   ```python
   # UNCLEAR: Which destabilizer provides context?
   sequence = [
       Emission(),
       Dissonance(),  # Too far (position 1)
       Reception(),
       Reception(),
       Reception(),
       Mutation(),    # Position 5 - no recent destabilizer!
       Silence()
   ]
   
   # Window is ~3 ops, so destabilizer at position 1
   # is NOT recent for transformer at position 5
   ```

#### Tests

**Implementation**: `src/tnfr/operators/grammar.py::GrammarValidator.validate_transformer_context()`

**Test Suite**:
- `tests/unit/operators/test_unified_grammar.py::TestU4bTransformerContext`
- `tests/integration/test_mutation_sequences.py::test_u4b_satisfied_in_canonical_sequence`
- `tests/unit/operators/test_controlled_bifurcation.py::test_transformer_at_sequence_start_fails`
- `tests/unit/operators/test_zhir_u4b_validation.py`
- `tests/unit/operators/test_mutation_metrics_comprehensive.py::test_grammar_u4b_validation`

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

**Related Documentation**:
- [UNIFIED_GRAMMAR_RULES.md § U4b](https://github.com/fermga/TNFR-Python-Engine/blob/main/UNIFIED_GRAMMAR_RULES.md#u4-bifurcation-dynamics)
- [AGENTS.md § Contract OZ + ZHIR Requirements](https://github.com/fermga/TNFR-Python-Engine/blob/main/AGENTS.md)
- [03-OPERATORS-AND-GLYPHS.md § Mutation (ZHIR)](03-OPERATORS-AND-GLYPHS.md)
- [U4B_AUDIT_REPORT.md](https://github.com/fermga/TNFR-Python-Engine/blob/main/U4B_AUDIT_REPORT.md) - Complete U4b implementation analysis

---

## U5: MULTI-SCALE COHERENCE

### Intuition

When recursivity operates at deep nesting levels, hierarchical coherence must be preserved. Parent coherence must dominate the sum of child coherences—otherwise fractal fragmentation occurs.

**Analogy:** A Russian nesting doll. The outer doll must be strong enough to contain all inner dolls. If the outer shell cracks (parent coherence too low), all nested structures spill out.

### Formal Definition

**IF** sequence **CONTAINS** `RECURSIVE_GENERATORS = {REMESH}` **with depth > 1**  
**THEN** sequence **MUST CONTAIN** `SCALE_STABILIZERS = {IL, THOL}` **within ±3 operators**

**Physical constraint:**
```
C_parent ≥ α · Σ C_child
```

Where:
- **α ≥ 1**: Conservation factor (typically 1.0-1.2)
- **C_parent**: Coherence of parent EPI
- **Σ C_child**: Sum of coherences of all nested sub-EPIs

### Physical Derivation

**From hierarchical coherence conservation:**

In multi-scale TNFR systems:
```
C_total = C_parent + Σ C_child
```

For stable nesting:
```
C_parent ≥ α · Σ C_child
```

**Without scale stabilizers:**
- Child coherences can grow unbounded
- Parent coherence insufficient to contain them
- Fractal fragmentation: nested structures break containment
- Loss of hierarchical integrity

**With scale stabilizers:**
- IL (Coherence) strengthens parent structure
- THOL (Self-organization) balances multi-scale dynamics
- Hierarchical coherence preserved across scales

**Timing constraint (±3 ops):**
- Stabilizers must act close to deep REMESH
- Too distant → fragmentation already initiated

This is **STRONG canonicity** - required by hierarchical physics.

### Implementation

```python
# From src/tnfr/operators/grammar.py

RECURSIVE_GENERATORS = frozenset({"recursivity"})
SCALE_STABILIZERS = frozenset({"coherence", "selforganization"})

def validate_grammar(sequence, epi_initial=0.0):
    """Validate sequence against U1-U5."""
    
    # U5: Multi-scale coherence
    for i, op in enumerate(sequence):
        op_name = op.__class__.__name__.lower()
        
        if op_name in RECURSIVE_GENERATORS:
            # Check if deep recursion (depth > 1)
            depth = getattr(op, 'depth', 1)
            
            if depth > 1:
                # Check for scale stabilizers within ±3 operators
                window_start = max(0, i - 3)
                window_end = min(len(sequence), i + 4)
                window = sequence[window_start:window_end]
                
                has_scale_stabilizer = any(
                    w.__class__.__name__.lower() in SCALE_STABILIZERS
                    for w in window
                )
                
                if not has_scale_stabilizer:
                    raise ValueError(
                        f"U5 violation: Deep REMESH (depth={depth}) at position {i} "
                        f"requires scale stabilizers {SCALE_STABILIZERS} within ±3 operations. "
                        "Without stabilizers, hierarchical coherence may fragment."
                    )
```

### Examples

**✅ Valid:**

```python
# Deep REMESH with Coherence stabilizer
sequence = [
    Emission(),
    Coherence(),        # Scale stabilizer
    Recursivity(depth=2),  # Deep REMESH - within window of IL
    Silence()
]
validate_grammar(sequence, epi_initial=0.0)  # ✓ Passes

# Deep REMESH with Self-organization stabilizer
sequence = [
    Emission(),
    Recursivity(depth=3),     # Deep REMESH
    SelfOrganization(),       # Scale stabilizer within ±3
    Coherence(),
    Silence()
]
validate_grammar(sequence, epi_initial=0.0)  # ✓ Passes

# Shallow REMESH (depth=1) - no stabilizer needed
sequence = [
    Emission(),
    Recursivity(depth=1),  # Shallow - U5 not triggered
    Silence()
]
validate_grammar(sequence, epi_initial=0.0)  # ✓ Passes
```

**❌ Invalid:**

```python
# ERROR: Deep REMESH without scale stabilizer
sequence = [
    Emission(),
    Recursivity(depth=2),  # Deep REMESH
    Silence()              # No IL or THOL within ±3!
]
validate_grammar(sequence, epi_initial=0.0)
# ✗ ValueError: U5 violation - deep REMESH needs scale stabilizers
```

```python
# ERROR: Stabilizer too far from deep REMESH
sequence = [
    Emission(),
    Coherence(),          # Position 1
    Reception(),
    Reception(),
    Reception(),
    Recursivity(depth=2), # Position 5 - stabilizer at position 1 is outside ±3 window!
    Silence()
]
validate_grammar(sequence, epi_initial=0.0)
# ✗ ValueError: U5 violation - stabilizer outside window
```

### Anti-Patterns

**⚠️ Common Mistakes:**

1. **Deep recursion without hierarchical control**
   ```python
   # BAD: Nested fractality without stabilization
   sequence = [
       Emission(),
       Recursivity(depth=3),  # Creates deep hierarchy
       Reception(),           # No stabilizers!
       Silence()
   ]
   # Child coherences may fragment parent structure
   
   # GOOD: Stabilize hierarchical structure
   sequence = [
       Emission(),
       SelfOrganization(),    # Prepares multi-scale organization
       Recursivity(depth=3),  # Deep hierarchy
       Coherence(),           # Stabilizes across scales
       Silence()
   ]
   ```

2. **Assuming shallow depth rules apply**
   ```python
   # BAD: Treating all REMESH equally
   sequence = [
       Emission(),
       Recursivity(depth=1),  # Shallow - OK without stabilizer
       Recursivity(depth=2),  # Deep - NEEDS stabilizer!
       Silence()
   ]
   
   # GOOD: Check depth and apply appropriate stabilization
   sequence = [
       Emission(),
       Recursivity(depth=1),  # Shallow
       Coherence(),           # Stabilizer for upcoming deep REMESH
       Recursivity(depth=2),  # Deep - now covered
       Silence()
   ]
   ```

3. **Stabilizer placement outside window**
   ```python
   # BAD: Stabilizer too early
   sequence = [
       Emission(),
       Coherence(),           # Position 1
       Reception(),
       Reception(),
       Reception(),
       Reception(),           # Many operations between
       Recursivity(depth=2),  # Position 6 - stabilizer outside ±3!
       Silence()
   ]
   
   # GOOD: Stabilizer within ±3 window
   sequence = [
       Emission(),
       Reception(),
       Reception(),
       Coherence(),           # Position 3
       Recursivity(depth=2),  # Position 4 - within window!
       Silence()
   ]
   ```

4. **Ignoring coherence conservation inequality**
   ```python
   # BAD: Creating many nested levels without checking C_parent
   sequence = [
       Emission(),
       Recursivity(depth=5),  # Very deep nesting
       Coherence(),           # Single stabilizer may be insufficient!
       Silence()
   ]
   # C_parent may not satisfy C_parent ≥ α·ΣC_child for depth=5
   
   # BETTER: Multiple stabilizers or depth limit
   sequence = [
       Emission(),
       Coherence(),           # Strengthen parent
       Recursivity(depth=3),  # Moderate depth
       SelfOrganization(),    # Balance multi-scale
       Coherence(),           # Additional stabilization
       Silence()
   ]
   ```

### Tests

**Implementation**: `src/tnfr/operators/grammar.py::GrammarValidator.validate_multiscale_coherence()`

**Test Suite**:
- `tests/unit/operators/test_unified_grammar.py::TestU5MultiScaleCoherence`
- `tests/unit/operators/test_u5_multiscale_coherence.py`
- `tests/integration/test_deep_remesh_sequences.py`
- `tests/unit/operators/test_recursive_operators.py::test_u5_deep_remesh_validation`

```python
def test_u5_multiscale_coherence():
    """U5: Deep REMESH requires scale stabilizers within ±3 operators."""
    from tnfr.operators.grammar import validate_grammar
    from tnfr.operators.definitions import (
        Emission, Recursivity, Coherence, SelfOrganization, Silence
    )
    
    # Valid: Deep REMESH with nearby Coherence
    sequence = [
        Emission(),
        Coherence(),
        Recursivity(depth=2),
        Silence()
    ]
    assert validate_grammar(sequence, epi_initial=0.0) is True
    
    # Valid: Deep REMESH with nearby Self-organization
    sequence = [
        Emission(),
        Recursivity(depth=3),
        SelfOrganization(),
        Silence()
    ]
    assert validate_grammar(sequence, epi_initial=0.0) is True
    
    # Valid: Shallow REMESH without stabilizer
    sequence = [
        Emission(),
        Recursivity(depth=1),
        Silence()
    ]
    assert validate_grammar(sequence, epi_initial=0.0) is True
    
    # Invalid: Deep REMESH without stabilizer
    sequence = [
        Emission(),
        Recursivity(depth=2),
        Silence()
    ]
    with pytest.raises(ValueError, match="U5 violation"):
        validate_grammar(sequence, epi_initial=0.0)
    
    # Invalid: Stabilizer outside ±3 window
    sequence = [
        Emission(),
        Coherence(),        # Position 1
        Reception(),
        Reception(),
        Reception(),
        Recursivity(depth=2),  # Position 5 - outside window!
        Silence()
    ]
    with pytest.raises(ValueError, match="U5 violation"):
        validate_grammar(sequence, epi_initial=0.0)
```

**Physical Verification:**
```python
def test_u5_coherence_conservation():
    """Verify C_parent ≥ α·ΣC_child for deep REMESH."""
    import networkx as nx
    from tnfr.metrics.coherence import compute_coherence
    
    # Create hierarchical graph with deep nesting
    G = nx.Graph()
    parent_node = 0
    child_nodes = [1, 2, 3]
    
    # Deep REMESH creates nested structure
    G.add_node(parent_node, EPI=1.0, vf=1.0, theta=0.0, depth=2)
    for child in child_nodes:
        G.add_node(child, EPI=0.5, vf=1.0, theta=0.1, parent=parent_node)
        G.add_edge(parent_node, child)
    
    # Compute coherences
    C_parent = compute_coherence(G, nodes=[parent_node])
    C_children = sum(compute_coherence(G, nodes=[c]) for c in child_nodes)
    
    # Verify conservation inequality
    alpha = 1.0
    assert C_parent >= alpha * C_children, \
        f"U5 violation: C_parent={C_parent} < α·ΣC_child={alpha*C_children}"
```

**Related Documentation**:
- [UNIFIED_GRAMMAR_RULES.md § U5](https://github.com/fermga/TNFR-Python-Engine/blob/main/UNIFIED_GRAMMAR_RULES.md#u5-multi-scale-coherence)
- [AGENTS.md § Invariant #7: Operational Fractality](https://github.com/fermga/TNFR-Python-Engine/blob/main/AGENTS.md)
- [03-OPERATORS-AND-GLYPHS.md § Recursivity (REMESH)](03-OPERATORS-AND-GLYPHS.md)
- [SHA_ALGEBRA_PHYSICS.md § Multi-Scale Coherence](https://github.com/fermga/TNFR-Python-Engine/blob/main/SHA_ALGEBRA_PHYSICS.md)

---

## U6: STRUCTURAL POTENTIAL CONFINEMENT

### Intuition

The structural potential field (Φ_s) acts like a "gravitational well" for network dynamics. Sequences that respect grammar naturally stay near potential minima (stable equilibrium). Large displacements from equilibrium indicate fragmentation risk.

**Analogy:** Like a marble in a bowl. Small disturbances keep it near the bottom (stable). Large kicks can eject it (fragmentation).

### Formal Definition

**FOR ALL** sequences (telemetry-based safety criterion):

**COMPUTE** structural potential field before and after sequence:
```
Φ_s(i) = Σ_{j≠i} ΔNFR_j / d(i,j)²
```

**VERIFY** displacement within escape threshold:
```
Δ Φ_s = |Φ_s_after - Φ_s_before| < 2.0
```

Where:
- `Φ_s(i)` = structural potential at node i
- `ΔNFR_j` = reorganization gradient at node j
- `d(i,j)` = network distance between nodes i and j
- `Δ Φ_s` = change in mean structural potential
- `2.0` = escape threshold (empirically calibrated)

**Note**: U6 is **telemetry-based** (read-only safety check), NOT a sequence constraint like U1-U5.

### Physical Derivation

From the nodal equation and network topology:

**Step 1: Structural potential emerges from ΔNFR distribution**
```
Each node contributes "structural pressure" to its neighbors
Weight by inverse-square of distance (like gravitational field)

Φ_s(i) = Σ_{j≠i} ΔNFR_j / d(i,j)²
```

**Step 2: Relationship to coherence**
```
From 2,400+ experiments across 5 topologies:

corr(Δ Φ_s, ΔC) = -0.822 (R² ≈ 0.68)

Strong negative correlation: displacement → coherence loss
```

**Step 3: Passive equilibrium mechanism**
```
Φ_s minima = passive equilibrium states (potential wells)

Grammar-valid sequences: Δ Φ_s ≈ 0.6 (30% of threshold)
Grammar-violating: Δ Φ_s ≈ 3.9 (195% of threshold)

Reduction factor: 0.15× (85% safer)
```

**Step 4: Universal validation**
```
Tested topologies: ring, scale_free, small-world, tree, grid
Coefficient of variation: CV = 0.1% (perfect universality)

→ Φ_s dynamics fundamental to TNFR, not topology artifact
```

**Physical interpretation**:
- Φ_s creates passive equilibrium landscape
- Grammar (U1-U5) acts as **confinement mechanism** (not active attractor)
- Valid sequences naturally maintain small Δ Φ_s
- Violations push system toward fragmentation boundary

This is **STRONG canonicity** - extensive empirical validation (2,400+ experiments), universal across topologies.

### Implementation

```python
# From src/tnfr/physics/fields.py

def compute_structural_potential(G, alpha=2.0):
    """Compute Φ_s field: Σ_{j≠i} ΔNFR_j / d(i,j)^α
    
    Parameters
    ----------
    G : nx.Graph
        TNFR network with 'dnfr' node attributes
    alpha : float
        Distance exponent (default: 2.0 for inverse-square)
    
    Returns
    -------
    dict
        {node_id: Φ_s value}
    """
    from networkx import shortest_path_length
    
    phi_s = {}
    for i in G.nodes():
        potential = 0.0
        for j in G.nodes():
            if i == j:
                continue
            dnfr_j = G.nodes[j].get('dnfr', 0.0)
            try:
                dist = shortest_path_length(G, i, j)
            except:
                dist = float('inf')
            
            if dist > 0 and dist < float('inf'):
                potential += dnfr_j / (dist ** alpha)
        
        phi_s[i] = potential
    
    return phi_s


# From src/tnfr/operators/grammar.py

def validate_structural_potential_confinement(
    G, 
    phi_s_before, 
    phi_s_after, 
    threshold=2.0
):
    """Validate U6: Structural potential confinement.
    
    Parameters
    ----------
    G : nx.Graph
        TNFR network
    phi_s_before : dict
        Φ_s field before sequence
    phi_s_after : dict
        Φ_s field after sequence
    threshold : float
        Escape threshold (default: 2.0)
    
    Returns
    -------
    tuple[bool, float, str]
        (is_valid, drift, message)
    """
    import numpy as np
    
    # Compute mean displacement
    nodes = list(G.nodes())
    phi_before_vals = [phi_s_before[n] for n in nodes]
    phi_after_vals = [phi_s_after[n] for n in nodes]
    
    delta_phi = abs(np.mean(phi_after_vals) - np.mean(phi_before_vals))
    
    is_valid = delta_phi < threshold
    
    if is_valid:
        msg = f"✓ U6: Δ Φ_s = {delta_phi:.3f} < {threshold} (safe)"
    else:
        msg = f"✗ U6: Δ Φ_s = {delta_phi:.3f} ≥ {threshold} (fragmentation risk)"
    
    return is_valid, delta_phi, msg
```

### Usage Example

```python
from tnfr.physics.fields import compute_structural_potential
from tnfr.operators.grammar import validate_structural_potential_confinement
from tnfr.operators.definitions import Emission, Dissonance, Coherence, Silence

# Setup network
G = create_test_network(topology='ring', n=50)

# Compute Φ_s before sequence
phi_s_before = compute_structural_potential(G, alpha=2.0)

# Apply sequence
sequence = [Emission(), Dissonance(), Coherence(), Silence()]
for op in sequence:
    apply_operator(G, target_node, op)
    step(G, dt=1.0)

# Compute Φ_s after sequence
phi_s_after = compute_structural_potential(G, alpha=2.0)

# Validate U6
is_valid, drift, msg = validate_structural_potential_confinement(
    G, phi_s_before, phi_s_after, threshold=2.0
)

print(msg)
# ✓ U6: Δ Φ_s = 0.583 < 2.0 (safe)

# For telemetry
print(f"Drift as % of threshold: {drift/2.0*100:.1f}%")
# Drift as % of threshold: 29.2%
```

### Valid vs Invalid Examples

**✓ VALID: Grammar-respecting sequence**
```python
sequence = [
    Emission(),        # Generator
    Coherence(),       # Stabilizer
    Dissonance(),      # Destabilizer
    Coherence(),       # Balances destabilizer
    Silence()          # Closure
]

# Expected: Δ Φ_s ≈ 0.6 (safe, ~30% of threshold)
```

**✗ INVALID: Grammar-violating sequence**
```python
sequence = [
    Emission(),        # Generator
    Dissonance(),      # Destabilizer
    Dissonance(),      # Another destabilizer (no stabilizer!)
    Dissonance(),      # Yet another!
    Mutation()         # Transformer without context
    # Missing closure, no stabilizers
]

# Expected: Δ Φ_s ≈ 3.9 (unsafe, ~195% of threshold)
```

### Tests

```python
def test_u6_structural_potential_confinement():
    """Test U6: Valid sequences maintain small Δ Φ_s."""
    G = create_test_network('ring', n=50)
    
    # Valid sequence
    phi_before = compute_structural_potential(G)
    apply_valid_sequence(G, node='n0')
    phi_after = compute_structural_potential(G)
    
    is_valid, drift, _ = validate_structural_potential_confinement(
        G, phi_before, phi_after, threshold=2.0
    )
    
    assert is_valid, "Valid sequence should pass U6"
    assert drift < 1.0, "Valid sequence should have small drift"
    
    # Invalid sequence (violates U2 - no stabilizers)
    G2 = create_test_network('ring', n=50)
    phi_before2 = compute_structural_potential(G2)
    apply_invalid_sequence(G2, node='n0')  # Multiple destabilizers, no stabilizers
    phi_after2 = compute_structural_potential(G2)
    
    is_valid2, drift2, _ = validate_structural_potential_confinement(
        G2, phi_before2, phi_after2, threshold=2.0
    )
    
    assert not is_valid2, "Invalid sequence should fail U6"
    assert drift2 > 2.0, "Invalid sequence should have large drift"


def test_u6_universality():
    """Test U6 works across topologies."""
    topologies = ['ring', 'scale_free', 'ws', 'tree', 'grid']
    
    for topology in topologies:
        G = create_test_network(topology, n=50)
        
        phi_before = compute_structural_potential(G)
        apply_valid_sequence(G, node='n0')
        phi_after = compute_structural_potential(G)
        
        is_valid, drift, _ = validate_structural_potential_confinement(
            G, phi_before, phi_after
        )
        
        assert is_valid, f"U6 should work for {topology}"
        assert drift < 1.0, f"Drift should be small for {topology}"
```

### Key Differences from U1-U5

| Aspect | U1-U5 | U6 |
|--------|-------|-----|
| **Type** | Sequence constraints | Telemetry criterion |
| **Enforcement** | Hard requirement | Safety monitoring |
| **Timing** | Check during validation | Check after execution |
| **Failure** | Reject sequence | Warning (fragmentation risk) |
| **Usage** | Mandatory for all sequences | Optional but recommended |

**Relationship to U2**:
- **U2**: Temporal integral convergence (∫νf·ΔNFR dt < ∞)
- **U6**: Spatial potential confinement (Δ Φ_s < 2.0)
- **Independence**: U2 prevents divergence over time, U6 prevents escape in structural space

### Related Documentation

- **[U6_STRUCTURAL_POTENTIAL_CONFINEMENT.md](U6_STRUCTURAL_POTENTIAL_CONFINEMENT.md)** - Complete specification ⭐
- **[UNIFIED_GRAMMAR_RULES.md § U6](../../UNIFIED_GRAMMAR_RULES.md)** - Physics derivation
- **[TNFR_FORCES_EMERGENCE.md § 14-15](../TNFR_FORCES_EMERGENCE.md)** - Empirical validation
- **[AGENTS.md § U6](../../AGENTS.md)** - Quick reference
- **[src/tnfr/physics/fields.py](../../src/tnfr/physics/fields.py)** - Implementation

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
| **U5** | REMESH depth>1 | Include {IL, THOL} within ±3 ops | STRONG | Hierarchical coherence conservation |
| **U6** | All sequences | Monitor Δ Φ_s < 2.0 (telemetry) | STRONG | Structural potential confinement (2,400+ experiments) |

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

# U5: Multi-scale coherence
RECURSIVE_GENERATORS = {"recursivity"}
SCALE_STABILIZERS = {"coherence", "selforganization"}
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

# Example with U5 (multi-scale)
from tnfr.operators.definitions import Recursivity

sequence_multiscale = [
    Emission(),              # U1a: Generator
    Coherence(),             # U5: Scale stabilizer
    Recursivity(depth=2),    # Deep REMESH
    Silence()                # U1b: Closure
]

try:
    is_valid = validate_grammar(sequence_multiscale, epi_initial=0.0)
    print("✓ Multi-scale sequence valid")
    print("Satisfies: U1a, U1b, U5")
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
