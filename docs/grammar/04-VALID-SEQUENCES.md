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

### 8. Dimension Reduction

**Pattern:** `[Generator → Stabilizer → Contraction → Stabilizer → Closure]`

**Purpose:** Simplify structure while maintaining coherence

**Example:**
```python
[Emission(), Coherence(), Contraction(), Coherence(), Silence()]
```

**Satisfies:**
- U2: Contraction (no destabilizer classification, but structural change) balanced by stabilizers
- U1a, U1b: Generator and closure
- Reduces dimensional complexity

**Use when:** Simplifying over-complex structures, dimension reduction

---

### 9. Network Bootstrap

**Pattern:** `[Generator → Coupling → Reception → Coherence → Closure]`

**Purpose:** Create node connected to network and integrate information

**Example:**
```python
[Emission(), Coupling(), Reception(), Coherence(), Silence()]
```

**Satisfies:**
- All constraints
- Establishes network connectivity early

**Use when:** Creating nodes that need immediate network integration

---

### 10. Controlled Expansion

**Pattern:** `[Generator → Stabilizer → Expansion → Stabilizer → Closure]`

**Purpose:** Increase structural complexity safely

**Example:**
```python
[Emission(), Coherence(), Expansion(), Coherence(), Silence()]
```

**Satisfies:**
- U2: Expansion (destabilizer) balanced by Coherence (stabilizer)
- Safe complexity increase

**Use when:** Growing structure dimensionality, adding degrees of freedom

---

### 11. Deep Exploration Cycle

**Pattern:** `[Generator → [Coherence → Dissonance]* → Coherence → Closure]`

**Purpose:** Multiple rounds of exploration with stabilization

**Example:**
```python
[Emission(), Coherence(), Dissonance(), Coherence(), Dissonance(), Coherence(), Silence()]
```

**Satisfies:**
- Multiple U2-balanced destabilizer-stabilizer pairs
- Repeated exploration cycles

**Use when:** Thorough exploration of structural space, optimization

---

### 12. Fractal Replication

**Pattern:** `[Generator → Coupling → SelfOrganization → Recursivity]`

**Purpose:** Create self-similar hierarchical structures

**Example:**
```python
[Emission(), Coherence(), Coupling(), Dissonance(), SelfOrganization(), Recursivity()]
```

**Satisfies:**
- U4b: THOL has recent destabilizer (Dissonance)
- U1b: Recursivity provides closure with nested structure
- Operational fractality

**Use when:** Creating multi-scale self-similar patterns

---

### 13. Transition-Based Reorganization

**Pattern:** `[Transition → Operations → Transition]`

**Purpose:** Move between regime states

**Example:**
```python
[Transition(), Coupling(), Reception(), Coherence(), Transition()]
```

**Satisfies:**
- U1a: Transition is generator
- U1b: Transition is closure
- Regime shift semantics

**Use when:** Switching between behavioral modes or attractor states

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

### ❌ 8. Expansion Without Stabilization

```python
# INVALID
[Emission(), Expansion(), Silence()]

# Error: U2 violation
# Destabilizer (Expansion) without stabilizer
```

**Why invalid:** Unbounded growth, integral diverges

**Fix:** Add stabilizer
```python
[Emission(), Expansion(), Coherence(), Silence()]  # ✓ Valid
```

---

### ❌ 9. Self-Organization Without Recent Destabilizer

```python
# INVALID
[Emission(), Coherence(), SelfOrganization(), Silence()]

# Error: U4b violation
# Transformer needs recent destabilizer (within ~3 ops)
```

**Why invalid:** Cannot organize without elevated ΔNFR

**Fix:** Add destabilizer before THOL
```python
[Emission(), Coherence(), Dissonance(), SelfOrganization(), Silence()]  # ✓ Valid
```

---

### ❌ 10. Resonance on Antiphase Nodes

```python
# INVALID (runtime error)
G.add_node(0, theta=0.0, ...)
G.add_node(1, theta=np.pi, ...)  # 180° phase difference

# Sequence is valid, but runtime fails U3
[Emission(), Coupling(), Resonance(), Silence()]
Resonance()(G, 0, 1)  # Error: antiphase nodes

# Error: U3 violation at runtime
# |θ₀ - θ₁| = π > Δθ_max
```

**Why invalid:** Destructive interference, physically meaningless coupling

**Fix:** Ensure phase compatibility
```python
G.add_node(0, theta=0.0, ...)
G.add_node(1, theta=0.1, ...)  # Compatible phase

# Now valid
Resonance()(G, 0, 1)  # ✓ Works
```

---

### ❌ 11. Multiple Destabilizers Without Adequate Stabilization

```python
# INVALID
[Emission(), Dissonance(), Expansion(), Coherence(), Silence()]

# Error: U2 violation
# Two destabilizers with only one stabilizer may not be sufficient
```

**Why invalid:** Cumulative instability may exceed single stabilizer capacity

**Fix:** Balance with multiple stabilizers
```python
[Emission(), Dissonance(), Coherence(), Expansion(), Coherence(), Silence()]  # ✓ Valid
```

---

### ❌ 12. Recursivity Without Proper Context

```python
# INVALID (semantic)
[Recursivity(), Silence()]  # Technically passes grammar

# Error: Logical violation
# Recursivity references prior structure but none exists
```

**Why invalid:** Cannot echo structure that doesn't exist yet

**Fix:** Build structure first
```python
[Emission(), Coherence(), Dissonance(), SelfOrganization(), Recursivity()]  # ✓ Valid
```

---

### ❌ 13. Reception Without Network Connectivity

```python
# INVALID (runtime/semantic)
# Node has no edges, yet using Reception
G.add_node(0, EPI=np.array([0.1]), ...)
# No edges!

[Emission(), Reception(), Silence()]
Reception()(G, 0)  # Warning: no neighbors

# Error: Semantic violation
# Reception with no information sources
```

**Why invalid:** Cannot integrate information from non-existent network

**Fix:** Establish connectivity first
```python
[Emission(), Coupling(), Reception(), Coherence(), Silence()]  # ✓ Valid
```

---

### ❌ 14. Transition Without State Change Logic

```python
# INVALID (semantic)
[Transition(), Transition()]  # Passes grammar but illogical

# Error: Semantic violation
# Transition to what? No operations between states
```

**Why invalid:** Transition without operations is meaningless

**Fix:** Add meaningful operations
```python
[Transition(), Coupling(), Reception(), Coherence(), Transition()]  # ✓ Valid
```

---

## 13×13 Operator Transition Matrix

This matrix shows the validity of transitions between operators. Read as "Can operator X be followed by operator Y?"

**Legend:**
- ✓ = Generally valid transition
- ⚠️ = Valid but needs additional context (see notes)
- ❌ = Generally invalid or problematic
- ⭐ = Recommended pattern

**Notes:**
- All transitions must still satisfy U1-U4 constraints
- Phase compatibility required for UM/RA (U3)
- Stabilizers needed for destabilizers (U2)
- Handlers needed for triggers (U4a)
- Transformers need recent destabilizers (U4b)

### Matrix

```
           │ AL │ EN │ IL │ OZ │ UM │ RA │SHA│VAL│NUL│THO│ZHI│NAV│REM│
───────────┼────┼────┼────┼────┼────┼────┼───┼───┼───┼───┼───┼───┼───┤
AL  Emiss  │ ⚠️ │ ⭐ │ ⭐ │ ⚠️ │ ⭐ │ ⚠️ │ ✓ │ ⚠️ │ ✓ │ ⚠️ │ ❌ │ ✓ │ ⚠️ │
EN  Recept │ ⚠️ │ ⚠️ │ ⭐ │ ⚠️ │ ✓ │ ⚠️ │ ✓ │ ⚠️ │ ✓ │ ⭐ │ ❌ │ ✓ │ ✓ │
IL  Coher  │ ⚠️ │ ✓ │ ⚠️ │ ⭐ │ ✓ │ ✓ │ ⭐ │ ✓ │ ✓ │ ⚠️ │ ⚠️ │ ⭐ │ ⭐ │
OZ  Disson │ ⚠️ │ ✓ │ ⭐ │ ❌ │ ✓ │ ✓ │ ⚠️ │ ⚠️ │ ✓ │ ⭐ │ ⭐ │ ✓ │ ✓ │
UM  Coupl  │ ⚠️ │ ⭐ │ ⭐ │ ⚠️ │ ⚠️ │ ⭐ │ ✓ │ ⚠️ │ ✓ │ ⚠️ │ ❌ │ ✓ │ ✓ │
RA  Reson  │ ⚠️ │ ⭐ │ ⭐ │ ⚠️ │ ✓ │ ⚠️ │ ✓ │ ⚠️ │ ✓ │ ⚠️ │ ❌ │ ✓ │ ✓ │
SHA Silenc │ ─  │ ─  │ ─  │ ─  │ ─  │ ─  │ ─ │ ─ │ ─ │ ─ │ ─ │ ─ │ ─ │
VAL Expan  │ ⚠️ │ ✓ │ ⭐ │ ⚠️ │ ✓ │ ✓ │ ⚠️ │ ❌ │ ⚠️ │ ⭐ │ ❌ │ ✓ │ ✓ │
NUL Contra │ ⚠️ │ ✓ │ ⭐ │ ⚠️ │ ✓ │ ✓ │ ✓ │ ⚠️ │ ⚠️ │ ⚠️ │ ❌ │ ✓ │ ✓ │
THO SelfO  │ ⚠️ │ ✓ │ ⭐ │ ⚠️ │ ✓ │ ✓ │ ✓ │ ⚠️ │ ✓ │ ⚠️ │ ⚠️ │ ✓ │ ⭐ │
ZHI Mutat  │ ⚠️ │ ✓ │ ⭐ │ ❌ │ ✓ │ ✓ │ ⚠️ │ ⚠️ │ ✓ │ ⭐ │ ❌ │ ✓ │ ✓ │
NAV Trans  │ ⭐ │ ⭐ │ ✓ │ ⚠️ │ ⭐ │ ⚠️ │ ─ │ ⚠️ │ ✓ │ ⚠️ │ ❌ │ ⚠️ │ ⚠️ │
REM Recurs │ ─  │ ─  │ ─  │ ─  │ ─  │ ─  │ ─ │ ─ │ ─ │ ─ │ ─ │ ─ │ ─ │
```

### Transition Notes

**Emission (AL):**
- ⭐ → EN, IL, UM: Bootstrap patterns
- ⚠️ → AL: Redundant unless transitioning regimes
- ❌ → ZHIR: Cannot mutate immediately after creation
- ⚠️ → OZ, VAL, THOL: Needs balancing

**Reception (EN):**
- ⭐ → IL, THOL: Integrate then stabilize
- ⚠️ → AL, EN: Redundant activation/reception
- ❌ → ZHIR: Cannot mutate immediately after reception

**Coherence (IL):**
- ⭐ → OZ, SHA, NAV, REM: Common stabilize-then patterns
- ⚠️ → IL: Redundant unless needed
- ⚠️ → AL: Unusual to emit after coherence

**Dissonance (OZ):**
- ⭐ → IL, THOL, ZHIR: Destabilize then handle/transform
- ❌ → OZ: Compounding instability dangerous
- ⚠️ → SHA: Should stabilize before silence

**Coupling (UM):**
- ⭐ → EN, RA, IL: Connect then integrate/propagate/stabilize
- ⚠️ → UM: Multiple couplings need phase checks
- ❌ → ZHIR: Cannot mutate immediately after coupling

**Resonance (RA):**
- ⭐ → EN, IL: Resonate then integrate/stabilize
- ⚠️ → RA: Cascading resonance needs control
- ❌ → ZHIR: Cannot mutate immediately after resonance

**Silence (SHA):**
- ─ = Closure operator, ends sequence (no followers)

**Expansion (VAL):**
- ⭐ → IL, THOL: Grow then stabilize/organize
- ❌ → VAL, ZHIR: Compounding expansion or immediate mutation
- ⚠️ → SHA: Should stabilize first

**Contraction (NUL):**
- ⭐ → IL: Contract then stabilize
- ⚠️ → NUL: Excessive contraction risky
- ❌ → ZHIR: Cannot mutate immediately

**Self-Organization (THOL):**
- ⭐ → IL, REM: Organize then stabilize/nest
- ⚠️ → THOL: Nested organization needs context
- ⚠️ → ZHIR: Sequential transforms need careful context

**Mutation (ZHIR):**
- ⭐ → IL, THOL: Transform then stabilize/organize
- ❌ → OZ, ZHIR: Cannot immediately destabilize/mutate again
- ⚠️ → SHA: Should stabilize first

**Transition (NAV):**
- ⭐ → AL, EN, UM: Regime shift then activate/connect
- ⚠️ → NAV: Multiple transitions need purpose
- ❌ → ZHIR: Transition to mutation without context

**Recursivity (REM):**
- ─ = Closure operator, ends sequence (no followers)

### Reading the Matrix

**Example 1:** Can Emission (AL) → Coherence (IL)?
- Row: AL, Column: IL → ⭐ (Recommended)
- Common bootstrap pattern

**Example 2:** Can Dissonance (OZ) → Mutation (ZHIR)?
- Row: OZ, Column: ZHIR → ⭐ (Recommended)
- Classic destabilize-then-transform pattern

**Example 3:** Can Mutation (ZHIR) → Mutation (ZHIR)?
- Row: ZHIR, Column: ZHIR → ❌ (Invalid)
- Cannot perform consecutive mutations

**Example 4:** Can Coherence (IL) → Dissonance (OZ)?
- Row: IL, Column: OZ → ⭐ (Recommended)
- Stabilize-then-explore pattern

---

## Step-by-Step Validation Logic

### Complete Validation Algorithm

This section documents the precise algorithms used to validate U1-U4 constraints. This is the canonical implementation that matches `tnfr.operators.grammar.validate_grammar()`.

```python
def validate_sequence_comprehensive(sequence, epi_initial=0.0):
    """
    Complete validation algorithm for TNFR sequences.
    
    Parameters
    ----------
    sequence : List[Operator]
        List of operator instances
    epi_initial : float
        Initial EPI magnitude (0.0 for vacuum state)
    
    Returns
    -------
    is_valid : bool
        True if sequence passes all constraints
    violations : List[str]
        List of violation messages (empty if valid)
    """
    violations = []
    
    # ========================================
    # STEP 1: U1a (Initiation)
    # ========================================
    # When EPI=0, must start with generator
    if epi_initial == 0.0 or is_effectively_zero(epi_initial):
        first_op = get_operator_name(sequence[0])
        if first_op not in GENERATORS:
            violations.append(
                f"U1a: When EPI=0, must start with generator "
                f"{{emission, transition, recursivity}}. "
                f"Found: '{first_op}'"
            )
    
    # ========================================
    # STEP 2: U1b (Closure)
    # ========================================
    # All sequences must end with closure operator
    if len(sequence) == 0:
        violations.append("U1b: Empty sequence has no closure")
    else:
        last_op = get_operator_name(sequence[-1])
        if last_op not in CLOSURES:
            violations.append(
                f"U1b: Sequence must end with closure "
                f"{{silence, transition, recursivity, dissonance}}. "
                f"Found: '{last_op}'"
            )
    
    # ========================================
    # STEP 3: U2 (Convergence & Boundedness)
    # ========================================
    # Destabilizers must be balanced by stabilizers
    # for integral ∫νf·ΔNFR dt to converge
    has_destabilizer = any(
        get_operator_name(op) in DESTABILIZERS 
        for op in sequence
    )
    has_stabilizer = any(
        get_operator_name(op) in STABILIZERS 
        for op in sequence
    )
    
    if has_destabilizer and not has_stabilizer:
        destabilizers_found = [
            get_operator_name(op) for op in sequence 
            if get_operator_name(op) in DESTABILIZERS
        ]
        violations.append(
            f"U2: Destabilizers {destabilizers_found} require "
            f"stabilizers {{coherence, self_organization}} "
            f"to ensure integral convergence (∫νf·ΔNFR dt < ∞)"
        )
    
    # ========================================
    # STEP 4: U3 (Resonant Coupling)
    # ========================================
    # Phase verification is runtime, not sequence-level
    # We note that phase checks will be needed
    has_coupling_resonance = any(
        get_operator_name(op) in COUPLING_RESONANCE 
        for op in sequence
    )
    
    # This is informational only - actual phase check happens at runtime
    # when operators are applied to specific nodes
    
    # ========================================
    # STEP 5: U4a (Bifurcation Triggers Need Handlers)
    # ========================================
    # When ∂²EPI/∂t² > τ, need handlers to control reorganization
    has_trigger = any(
        get_operator_name(op) in BIFURCATION_TRIGGERS 
        for op in sequence
    )
    has_handler = any(
        get_operator_name(op) in BIFURCATION_HANDLERS 
        for op in sequence
    )
    
    if has_trigger and not has_handler:
        triggers_found = [
            get_operator_name(op) for op in sequence 
            if get_operator_name(op) in BIFURCATION_TRIGGERS
        ]
        violations.append(
            f"U4a: Bifurcation triggers {triggers_found} require "
            f"handlers {{coherence, self_organization}} "
            f"to control reorganization when ∂²EPI/∂t² > τ"
        )
    
    # ========================================
    # STEP 6: U4b (Transformers Need Context)
    # ========================================
    # Transformers need recent destabilizer for elevated ΔNFR
    # ZHIR specifically needs prior IL (stable base)
    TRANSFORMER_WINDOW = 3  # Recent = within 3 operators
    
    for i, op in enumerate(sequence):
        op_name = get_operator_name(op)
        
        if op_name not in TRANSFORMERS:
            continue
        
        # Check recent destabilizer (within window)
        window_start = max(0, i - TRANSFORMER_WINDOW)
        recent_window = sequence[window_start:i]
        
        has_recent_destabilizer = any(
            get_operator_name(w) in DESTABILIZERS 
            for w in recent_window
        )
        
        if not has_recent_destabilizer:
            violations.append(
                f"U4b: Transformer '{op_name}' at position {i} "
                f"needs recent destabilizer (within {TRANSFORMER_WINDOW} ops). "
                f"Window: {[get_operator_name(w) for w in recent_window]}"
            )
        
        # ZHIR-specific: needs prior IL (stable base before destabilization)
        if op_name == "mutation":
            # Check for coherence BEFORE the destabilizer window
            prior_to_window = sequence[:window_start]
            has_prior_coherence = any(
                get_operator_name(p) == "coherence" 
                for p in prior_to_window
            )
            
            if not has_prior_coherence:
                violations.append(
                    f"U4b: Mutation at position {i} needs prior "
                    f"coherence (stable base before destabilization). "
                    f"Prior ops: {[get_operator_name(p) for p in prior_to_window]}"
                )
    
    return len(violations) == 0, violations
```

### Validation Decision Tree

Visual representation of validation logic flow:

```
START
  │
  ├─> Is EPI = 0?
  │   ├─> YES: First op in GENERATORS? → NO: FAIL U1a
  │   └─> NO: Continue
  │
  ├─> Last op in CLOSURES?
  │   ├─> NO: FAIL U1b
  │   └─> YES: Continue
  │
  ├─> Any DESTABILIZERS in sequence?
  │   ├─> YES: Any STABILIZERS? → NO: FAIL U2
  │   └─> NO: Continue
  │
  ├─> Any COUPLING_RESONANCE in sequence?
  │   └─> YES: Note runtime phase check needed (U3)
  │
  ├─> Any BIFURCATION_TRIGGERS in sequence?
  │   ├─> YES: Any HANDLERS? → NO: FAIL U4a
  │   └─> NO: Continue
  │
  ├─> For each TRANSFORMER at position i:
  │   ├─> Recent DESTABILIZER (i-3 to i)?
  │   │   ├─> NO: FAIL U4b
  │   │   └─> YES: Continue
  │   │
  │   └─> If MUTATION:
  │       └─> Prior COHERENCE (before i-3)?
  │           ├─> NO: FAIL U4b (ZHIR-specific)
  │           └─> YES: Continue
  │
  └─> PASS (all constraints satisfied)
```

### Runtime Validation: U3 Phase Compatibility

Phase validation happens when operators are actually applied, not during sequence validation:

```python
def validate_phase_compatibility_runtime(G, node_i, node_j, delta_theta_max=np.pi/2):
    """
    Runtime validation for U3 (Resonant Coupling).
    
    Called when Coupling or Resonance operators are applied.
    
    Parameters
    ----------
    G : TNFRGraph
        Network graph
    node_i, node_j : NodeId
        Nodes to couple
    delta_theta_max : float
        Maximum allowed phase difference (default π/2)
    
    Raises
    ------
    PhaseCompatibilityError
        If phase difference exceeds threshold
    """
    theta_i = G.nodes[node_i]['theta']
    theta_j = G.nodes[node_j]['theta']
    
    # Phase difference
    delta_theta = abs(theta_i - theta_j)
    
    # Normalize to [0, π] (symmetry: θ and θ+π are equivalent)
    if delta_theta > np.pi:
        delta_theta = 2 * np.pi - delta_theta
    
    # Check threshold
    if delta_theta > delta_theta_max:
        raise PhaseCompatibilityError(
            f"U3: Cannot couple nodes {node_i} and {node_j}. "
            f"Phase difference |θᵢ - θⱼ| = {delta_theta:.3f} rad "
            f"exceeds Δθ_max = {delta_theta_max:.3f} rad. "
            f"Antiphase coupling causes destructive interference."
        )
    
    return True
```

### Validation Timing

**Sequence-Level Validation** (U1a, U1b, U2, U4a, U4b):
- Performed BEFORE sequence execution
- Static analysis of operator list
- Fast, no graph required
- Use: `validate_grammar(sequence, epi_initial)`

**Runtime Validation** (U3):
- Performed DURING operator application
- Requires actual graph state (phase values)
- Dynamic check per coupling/resonance
- Automatic in operator implementations

### Validation Strictness Levels

```python
class ValidationMode(Enum):
    """Validation strictness levels."""
    STRICT = "strict"      # Fail on any violation
    WARNING = "warning"    # Warn but allow
    DISABLED = "disabled"  # No validation
```

**STRICT** (default):
- All violations cause errors
- Recommended for production
- Ensures physics integrity

**WARNING**:
- Violations logged but not fatal
- Useful for experimentation
- Risk: may violate physics

**DISABLED**:
- No validation
- Only for testing/debugging
- Not recommended

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

## "How Do I...?" Lookup Guide

This section provides quick answers to common questions. Find your goal, get the sequence.

### Creation & Initialization

**Q: How do I create a new node from scratch?**
```python
[Emission(), Coherence(), Silence()]
```
Minimal bootstrap pattern.

**Q: How do I create a node connected to a network?**
```python
[Emission(), Coupling(), Reception(), Coherence(), Silence()]
```
Creates node, connects it, integrates network info, stabilizes.

**Q: How do I initialize multiple nodes at once?**
```python
# For each node i:
[Emission(), Coherence(), Silence()]
# Then couple them:
[Coupling(), Resonance(), Silence()]
```
Bootstrap individually, then connect.

---

### Exploration & Perturbation

**Q: How do I explore alternative states safely?**
```python
[Emission(), Coherence(), Dissonance(), Coherence(), Silence()]
```
Stabilize-explore-restabilize pattern.

**Q: How do I perform deep exploration with multiple iterations?**
```python
[Emission(), Coherence(), Dissonance(), Coherence(), 
 Dissonance(), Coherence(), Silence()]
```
Multiple stabilize-explore cycles.

**Q: How do I break out of a local optimum?**
```python
[Emission(), Coherence(), Dissonance(), SelfOrganization(), Silence()]
```
Destabilize then reorganize into new configuration.

---

### Network Operations

**Q: How do I connect two existing nodes?**
```python
# Verify phase compatibility first, then:
[Coupling(), Coherence(), Silence()]
```
Note: Must check |θᵢ - θⱼ| ≤ Δθ_max (U3).

**Q: How do I propagate a pattern through the network?**
```python
[Emission(), Coupling(), Resonance(), Coherence(), Silence()]
```
Create, connect, resonate, stabilize.

**Q: How do I integrate information from neighbors?**
```python
[Reception(), Coherence(), Silence()]
```
Requires existing network connectivity.

---

### Structural Transformation

**Q: How do I change a node's phase (qualitative transformation)?**
```python
[Emission(), Coherence(), Dissonance(), Mutation(), Coherence(), Silence()]
```
Must have prior IL, recent destabilizer for ZHIR.

**Q: How do I increase structural complexity?**
```python
[Emission(), Coherence(), Expansion(), Coherence(), Silence()]
```
Balanced growth with stabilization.

**Q: How do I simplify an over-complex structure?**
```python
[Emission(), Coherence(), Contraction(), Coherence(), Silence()]
```
Dimension reduction with stabilization.

---

### Hierarchical Structures

**Q: How do I create a hierarchical/nested structure?**
```python
[Emission(), Coherence(), Dissonance(), SelfOrganization(), Recursivity()]
```
Destabilize to enable reorganization, organize into hierarchy, nest.

**Q: How do I build multi-scale fractal patterns?**
```python
[Emission(), Coupling(), Coherence(), Dissonance(), 
 SelfOrganization(), Recursivity()]
```
Connect, stabilize, perturb, self-organize, nest recursively.

**Q: How do I organize chaotic structure?**
```python
[Emission(), Dissonance(), SelfOrganization(), Coherence(), Silence()]
```
Perturb, let self-organize, stabilize result.

---

### Regime Changes

**Q: How do I switch between behavioral modes?**
```python
[Transition(), Coupling(), Reception(), Coherence(), Transition()]
```
Transition operators for regime shifts.

**Q: How do I pause evolution temporarily?**
```python
[Emission(), Coherence(), Silence()]
```
Silence freezes evolution (νf → 0).

**Q: How do I resume evolution after silence?**
```python
# New sequence starting from existing EPI:
[Emission(), Coherence(), ...]  # epi_initial > 0
```
Or use Transition to activate latent EPI.

---

### Complex Workflows

**Q: How do I implement a learning cycle?**
```python
[Emission(), Coupling(), Reception(), Coherence(),      # Gather info
 Dissonance(), Coherence(),                              # Explore
 SelfOrganization(), Coherence(), Silence()]             # Consolidate
```
Integrate → Explore → Organize pattern.

**Q: How do I perform iterative optimization?**
```python
[Emission(), Coherence(),
 Dissonance(), Coherence(),  # Iteration 1
 Dissonance(), Coherence(),  # Iteration 2
 Dissonance(), Coherence(),  # Iteration 3
 Silence()]
```
Repeated explore-stabilize cycles.

**Q: How do I implement adaptive reorganization?**
```python
[Emission(), Coupling(), Reception(), Coherence(),
 Dissonance(), Mutation(), SelfOrganization(),
 Coherence(), Recursivity()]
```
Sense environment → Transform → Reorganize → Nest.

---

### Debugging & Validation

**Q: My sequence fails U1a. What's wrong?**
- **Check:** Starting from EPI=0? Must begin with {Emission, Transition, Recursivity}
- **Fix:** Add generator at start

**Q: My sequence fails U1b. What's wrong?**
- **Check:** Does sequence end with {Silence, Transition, Recursivity, Dissonance}?
- **Fix:** Add closure operator at end

**Q: My sequence fails U2. What's wrong?**
- **Check:** Using {Dissonance, Mutation, Expansion}? Need {Coherence, SelfOrganization}
- **Fix:** Add stabilizer after destabilizers

**Q: My sequence fails U3 at runtime. What's wrong?**
- **Check:** Phase difference |θᵢ - θⱼ| ≤ Δθ_max?
- **Fix:** Only couple phase-compatible nodes

**Q: My sequence fails U4a. What's wrong?**
- **Check:** Using {Dissonance, Mutation}? Need handlers {Coherence, SelfOrganization}
- **Fix:** Add handler after triggers

**Q: My sequence fails U4b. What's wrong?**
- **Check 1:** Transformer has recent destabilizer (within 3 ops)?
- **Check 2:** If Mutation, has prior Coherence?
- **Fix:** Add destabilizer before transformer, ensure prior IL for ZHIR

---

### Performance & Efficiency

**Q: What's the shortest valid sequence?**
```python
[Emission(), Silence()]
```
Just creation and closure (only valid from EPI=0).

**Q: What's the most stable pattern?**
```python
[Emission(), Coherence(), Coherence(), Silence()]
```
Extra stabilization for maximum C(t).

**Q: What's the most exploratory pattern?**
```python
[Emission(), Coherence(), Dissonance(), Coherence(),
 Dissonance(), Coherence(), Dissonance(), Coherence(), Silence()]
```
Maximum exploration with safety (multiple cycles).

---

### Domain-Specific Patterns

**Q: How do I model neural learning?**
```python
[Emission(), Coupling(), Reception(), Coherence(),      # Input integration
 Dissonance(), Mutation(), Coherence(),                 # Weight adjustment
 Resonance(), Coherence(), Silence()]                   # Output propagation
```

**Q: How do I model biological growth?**
```python
[Emission(), Coherence(), Expansion(), Coherence(),     # Growth
 SelfOrganization(), Coherence(), Silence()]            # Differentiation
```

**Q: How do I model social consensus formation?**
```python
[Emission(), Coupling(), Resonance(), Reception(),      # Information sharing
 Coherence(), Dissonance(), SelfOrganization(),         # Debate & alignment
 Coherence(), Silence()]                                 # Consensus reached
```

**Q: How do I model quantum decoherence?**
```python
[Emission(), Coherence(), Dissonance(),                 # Perturbation
 Contraction(), Silence()]                              # Collapse to eigenstate
```

---

### Quick Pattern Finder

**Goal → Pattern Type → Sequence**

| Goal | Pattern | Sequence |
|------|---------|----------|
| Create new | Bootstrap | `[AL, IL, SHA]` |
| Explore | Exploration | `[AL, IL, OZ, IL, SHA]` |
| Connect | Network | `[AL, UM, EN, IL, SHA]` |
| Propagate | Resonance | `[AL, UM, RA, IL, SHA]` |
| Transform | Mutation | `[AL, IL, OZ, ZHIR, IL, SHA]` |
| Organize | Hierarchy | `[AL, OZ, THOL, REM]` |
| Expand | Growth | `[AL, IL, VAL, IL, SHA]` |
| Contract | Reduction | `[AL, IL, NUL, IL, SHA]` |
| Transition | Regime shift | `[NAV, UM, EN, IL, NAV]` |
| Learn | Adaptive | `[AL, UM, EN, IL, OZ, IL, THOL, SHA]` |

**Legend:**
- AL=Emission, EN=Reception, IL=Coherence, OZ=Dissonance
- UM=Coupling, RA=Resonance, SHA=Silence, VAL=Expansion
- NUL=Contraction, THOL=SelfOrganization, ZHIR=Mutation
- NAV=Transition, REM=Recursivity

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
