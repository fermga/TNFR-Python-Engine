# Master Index - TNFR Grammar System

**Global conceptual map and relationship diagram**

[🏠 Home](README.md) • [🌊 Concepts](01-FUNDAMENTAL-CONCEPTS.md) • [📚 Glossary](GLOSSARY.md)

---

## Purpose

This document provides a **high-level conceptual map** of the entire TNFR grammar system, showing relationships between concepts, constraints, and operators.

**Audience:** Developers planning large changes, system architects

**Reading time:** 15-20 minutes

---

## System Overview Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    TNFR GRAMMAR SYSTEM                          │
│                                                                 │
│  Physical Foundation                                            │
│  ├─ Nodal Equation: ∂EPI/∂t = νf · ΔNFR(t)                    │
│  ├─ Structural Triad: (EPI, νf, φ)                            │
│  └─ Integral Convergence: ∫νf·ΔNFR dt < ∞                      │
│                                                                 │
│  ↓ Derives                                                      │
│                                                                 │
│  Grammar Constraints (U1-U4)                                    │
│  ├─ U1: INITIATION & CLOSURE (generators, closures)           │
│  ├─ U2: CONVERGENCE (stabilizers, destabilizers)              │
│  ├─ U3: COUPLING (phase compatibility)                        │
│  └─ U4: BIFURCATION (triggers, handlers, transformers)        │
│                                                                 │
│  ↓ Governs                                                      │
│                                                                 │
│  Canonical Operators (13)                                       │
│  ├─ Initialization: AL, NAV, REMESH                           │
│  ├─ Information: EN                                            │
│  ├─ Stabilization: IL, THOL                                   │
│  ├─ Destabilization: OZ, ZHIR, VAL                           │
│  ├─ Propagation: UM, RA                                       │
│  └─ Control: SHA, NUL                                         │
│                                                                 │
│  ↓ Compose into                                                 │
│                                                                 │
│  Valid Sequences                                                │
│  ├─ Bootstrap: [AL, IL, SHA]                                  │
│  ├─ Exploration: [AL, IL, OZ, IL, SHA]                       │
│  ├─ Transformation: [AL, IL, OZ, ZHIR, IL, SHA]              │
│  └─ Propagation: [AL, UM, RA, IL, SHA]                       │
│                                                                 │
│  ↓ Validated by                                                 │
│                                                                 │
│  Validation System                                              │
│  ├─ validate_grammar(sequence, epi_initial)                   │
│  ├─ validate_resonant_coupling(G, i, j)                       │
│  └─ Test suite (unit, integration, property)                  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Conceptual Hierarchy

### Level 1: Physical Principles

**Nodal Equation:**
```
∂EPI/∂t = νf · ΔNFR(t)
```

**Derives:**
- U1a: Cannot start from EPI=0 without generator
- U2: Integral must converge
- U4: Bifurcations need control

**Structural Triad:**
- Form (EPI)
- Frequency (νf)
- Phase (φ)

**Derives:**
- U3: Phase compatibility for coupling

---

### Level 2: Grammar Constraints

```
U1: INITIATION & CLOSURE
    ├─ U1a: Generators ─→ {AL, NAV, REMESH}
    └─ U1b: Closures ─→ {SHA, NAV, REMESH, OZ}

U2: CONVERGENCE & BOUNDEDNESS
    ├─ Stabilizers ─→ {IL, THOL}
    └─ Destabilizers ─→ {OZ, ZHIR, VAL}

U3: RESONANT COUPLING
    └─ Phase check ─→ |φᵢ - φⱼ| ≤ Δφ_max

U4: BIFURCATION DYNAMICS
    ├─ U4a: Triggers ─→ {OZ, ZHIR} need Handlers {IL, THOL}
    └─ U4b: Transformers ─→ {ZHIR, THOL} need recent destabilizer
                           + ZHIR needs prior IL
```

---

### Level 3: Operators

```
13 Canonical Operators

├─ Initialization (3)
│   ├─ AL (Emission) ──────────┐
│   ├─ NAV (Transition) ────────┤─→ Generators (U1a)
│   └─ REMESH (Recursivity) ────┘

├─ Information (1)
│   └─ EN (Reception)

├─ Stabilization (2)
│   ├─ IL (Coherence) ─────────┐
│   └─ THOL (Self-org) ─────────┤─→ Stabilizers (U2), Handlers (U4a)
│                               │   Transformers (U4b for THOL)
│                               └─→ Handler: IL, THOL

├─ Destabilization (3)
│   ├─ OZ (Dissonance) ────────┐
│   ├─ ZHIR (Mutation) ─────────┤─→ Destabilizers (U2), Triggers (U4a)
│   └─ VAL (Expansion) ─────────┘   Transformers (U4b)

├─ Propagation (2)
│   ├─ UM (Coupling) ──────────┐
│   └─ RA (Resonance) ──────────┤─→ Requires phase check (U3)

└─ Control (2)
    ├─ SHA (Silence) ──────────┐
    └─ NUL (Contraction)       │─→ SHA is Closure (U1b)
                               └─→ OZ also Closure (U1b)
```

---

## Constraint Dependencies

```
U1a (Initiation)
  ↓
Requires: GENERATORS = {emission, transition, recursivity}
  ↓
When: epi_initial == 0.0

U1b (Closure)
  ↓
Requires: CLOSURES = {silence, transition, recursivity, dissonance}
  ↓
Always (end of every sequence)

U2 (Convergence)
  ↓
If: DESTABILIZERS present
  ↓
Then: STABILIZERS required
  ↓
Ensures: ∫νf·ΔNFR dt < ∞

U3 (Coupling)
  ↓
If: {coupling, resonance} applied
  ↓
Then: Verify |φᵢ - φⱼ| ≤ Δφ_max
  ↓
At: Runtime (during operator application)

U4a (Triggers)
  ↓
If: BIFURCATION_TRIGGERS present
  ↓
Then: BIFURCATION_HANDLERS required

U4b (Transformers)
  ↓
If: TRANSFORMERS present
  ↓
Then: Recent destabilizer (~3 ops)
  ↓
And (for ZHIR): Prior IL (Coherence)
```

---

## Operator Classification Matrix

|Operator|Generator|Closure|Stabilizer|Destabilizer|Trigger|Handler|Transformer|Coupling|
|--------|---------|-------|----------|------------|-------|-------|-----------|--------|
|AL      |✓        |       |          |            |       |       |           |        |
|EN      |         |       |          |            |       |       |           |        |
|IL      |         |       |✓         |            |       |✓      |           |        |
|OZ      |         |✓      |          |✓           |✓      |       |           |        |
|UM      |         |       |          |            |       |       |           |✓       |
|RA      |         |       |          |            |       |       |           |✓       |
|SHA     |         |✓      |          |            |       |       |           |        |
|VAL     |         |       |          |✓           |       |       |           |        |
|NUL     |         |       |          |            |       |       |           |        |
|THOL    |         |       |✓         |            |       |✓      |✓          |        |
|ZHIR    |         |       |          |✓           |✓      |       |✓          |        |
|NAV     |✓        |✓      |          |            |       |       |           |        |
|REMESH  |✓        |✓      |          |            |       |       |           |        |

---

## Validation Flow

```
Input: sequence, epi_initial

   ↓
   
Step 1: Check U1a (Initiation)
   │
   ├─ IF epi_initial == 0.0
   │   └─ sequence[0] ∈ GENERATORS?
   └─ PASS/FAIL
   
   ↓
   
Step 2: Check U1b (Closure)
   │
   └─ sequence[-1] ∈ CLOSURES?
   
   ↓
   
Step 3: Check U2 (Convergence)
   │
   ├─ Has DESTABILIZERS?
   │   └─ Has STABILIZERS? (must have)
   └─ PASS/FAIL
   
   ↓
   
Step 4: Check U4a (Triggers)
   │
   ├─ Has BIFURCATION_TRIGGERS?
   │   └─ Has BIFURCATION_HANDLERS? (must have)
   └─ PASS/FAIL
   
   ↓
   
Step 5: Check U4b (Transformers)
   │
   └─ For each TRANSFORMER:
       ├─ Has recent destabilizer? (~3 ops)
       └─ IF ZHIR: Has prior IL?
   
   ↓
   
Step 6: U3 verified at runtime
   │
   └─ During coupling/resonance application
       └─ validate_resonant_coupling(G, i, j)
   
   ↓
   
Output: VALID or ValueError with explanation
```

---

## Data Flow

### Operator Application

```
Input State (t₀)
  ├─ EPI(t₀)
  ├─ νf(t₀)
  ├─ θ(t₀)
  └─ ΔNFR(t₀)
  
  ↓ Apply Operator
  
Transformation
  ├─ Compute effect on ∂EPI/∂t
  ├─ Update νf (if applicable)
  ├─ Update θ (if applicable)
  └─ Update ΔNFR (if applicable)
  
  ↓ Integration
  
Output State (t₁)
  ├─ EPI(t₁)
  ├─ νf(t₁)
  ├─ θ(t₁)
  └─ ΔNFR(t₁)
  
  ↓ Telemetry
  
Metrics
  ├─ C(t) - Coherence
  ├─ Si - Sense Index
  ├─ νf - Frequency
  ├─ θ - Phase
  └─ ΔNFR - Gradient
```

---

## Document Relationships

```
TNFR Grammar Documentation

├─ README.md (You start here)
│   └─ Navigation hub
│
├─ 01-FUNDAMENTAL-CONCEPTS.md
│   ├─ Paradigm shift
│   ├─ Nodal equation
│   ├─ Structural triad
│   └─ Referenced by: All other docs
│
├─ 02-CANONICAL-CONSTRAINTS.md
│   ├─ U1: Initiation & Closure
│   ├─ U2: Convergence
│   ├─ U3: Coupling
│   ├─ U4: Bifurcation
│   └─ Referenced by: 04, 05, 06, 07
│
├─ 03-OPERATORS-AND-GLYPHS.md
│   ├─ 13 operator definitions
│   ├─ Grammar classification
│   └─ Referenced by: 04, 05, 06
│
├─ 04-VALID-SEQUENCES.md
│   ├─ Canonical patterns
│   ├─ Anti-patterns
│   └─ References: 02, 03
│
├─ 05-TECHNICAL-IMPLEMENTATION.md
│   ├─ Code architecture
│   ├─ Validation algorithms
│   └─ References: 02, 03
│
├─ 06-VALIDATION-AND-TESTING.md
│   ├─ Test strategy
│   ├─ Test examples
│   └─ References: 02, 03, 05
│
├─ 07-MIGRATION-AND-EVOLUTION.md
│   ├─ Version history
│   ├─ Migration guide
│   └─ References: 02, 05
│
├─ 08-QUICK-REFERENCE.md
│   ├─ Cheat sheet
│   ├─ Quick lookup
│   └─ References: All docs
│
├─ GLOSSARY.md (You are here)
│   ├─ Operational definitions
│   └─ Referenced by: All docs
│
└─ MASTER-INDEX.md (This document)
    ├─ Conceptual map
    ├─ Relationship diagrams
    └─ System overview
```

---

## Implementation Files

```
Source Code Structure

src/tnfr/operators/
├─ grammar.py
│   ├─ Operator sets (GENERATORS, CLOSURES, etc.)
│   ├─ validate_grammar(sequence, epi_initial)
│   └─ validate_resonant_coupling(G, i, j)
│
├─ definitions.py
│   ├─ 13 operator classes
│   │   ├─ Emission, Reception, Coherence, etc.
│   │   └─ Each implements __call__(G, node)
│   └─ Integrated with grammar.py
│
└─ unified_grammar.py
    └─ Legacy compatibility (deprecated)

tests/
├─ unit/operators/test_unified_grammar.py
│   ├─ U1a, U1b tests
│   ├─ U2, U3, U4 tests
│   └─ Operator tests
│
├─ integration/
│   └─ Full workflow tests
│
└─ property/
    └─ Invariant tests

docs/grammar/
├─ All documentation files
├─ examples/
│   ├─ 01-basic-bootstrap.py
│   ├─ 02-intermediate-exploration.py
│   └─ 03-advanced-bifurcation.py
│
└─ schemas/
    ├─ canonical-operators.json
    └─ constraints-u1-u4.json
```

---

## Dependency Graph

```
Physical Principles (TNFR.pdf)
    ↓
Nodal Equation
    ↓
Grammar Constraints (U1-U4)
    ↓
Operator Classification
    ↓
Validation Logic
    ↓
Implementation (grammar.py, definitions.py)
    ↓
Tests (test_unified_grammar.py)
    ↓
Documentation (01-08, GLOSSARY, MASTER-INDEX)
    ↓
Examples (examples/*.py)
    ↓
User Applications
```

---

## Key Relationships

### Operators → Constraints

```
Emission ──┐
Transition ├─→ GENERATORS ──→ U1a
Recursivity┘

Silence ───┐
Transition ├─→ CLOSURES ──→ U1b
Recursivity│
Dissonance ┘

Coherence ─────┐
Self-org ──────┤─→ STABILIZERS ──→ U2
               │
               └─→ HANDLERS ──→ U4a

Dissonance ─┐
Mutation ───├─→ DESTABILIZERS ──→ U2
Expansion ──┘  └─→ TRIGGERS ──→ U4a

Mutation ───┐
Self-org ───┴─→ TRANSFORMERS ──→ U4b

Coupling ───┐
Resonance ──┴─→ Phase check ──→ U3
```

### Constraints → Physics

```
U1a ─→ ∂EPI/∂t undefined at EPI=0
U1b ─→ Attractor dynamics
U2  ─→ ∫νf·ΔNFR dt < ∞
U3  ─→ Wave interference physics
U4a ─→ Bifurcation theory
U4b ─→ Threshold dynamics
```

---

## Navigation Paths

### For New Users

```
START
  ↓
README.md (Navigation)
  ↓
01-FUNDAMENTAL-CONCEPTS.md
  ↓
GLOSSARY.md (Reference as needed)
  ↓
03-OPERATORS-AND-GLYPHS.md
  ↓
examples/01-basic-bootstrap.py
  ↓
08-QUICK-REFERENCE.md (Keep open)
```

### For Developers

```
START
  ↓
02-CANONICAL-CONSTRAINTS.md
  ↓
05-TECHNICAL-IMPLEMENTATION.md
  ↓
src/tnfr/operators/grammar.py (Read code)
  ↓
06-VALIDATION-AND-TESTING.md
  ↓
tests/unit/operators/test_unified_grammar.py
  ↓
MASTER-INDEX.md (This document, for big picture)
```

### For Maintainers

```
START
  ↓
07-MIGRATION-AND-EVOLUTION.md
  ↓
MASTER-INDEX.md (This document)
  ↓
02-CANONICAL-CONSTRAINTS.md (Review constraints)
  ↓
05-TECHNICAL-IMPLEMENTATION.md (Architecture)
  ↓
Code changes as needed
```

---

## Cross-Reference Index

### By Concept

**Nodal Equation:**
- [01-FUNDAMENTAL-CONCEPTS.md](01-FUNDAMENTAL-CONCEPTS.md) § Nodal Equation
- [02-CANONICAL-CONSTRAINTS.md](02-CANONICAL-CONSTRAINTS.md) § Physical Derivation
- [GLOSSARY.md](GLOSSARY.md) § N

**Operators:**
- [03-OPERATORS-AND-GLYPHS.md](03-OPERATORS-AND-GLYPHS.md) § All 13 operators
- [05-TECHNICAL-IMPLEMENTATION.md](05-TECHNICAL-IMPLEMENTATION.md) § Operator Sets
- [GLOSSARY.md](GLOSSARY.md) § O

**Constraints:**
- [02-CANONICAL-CONSTRAINTS.md](02-CANONICAL-CONSTRAINTS.md) § U1-U4
- [08-QUICK-REFERENCE.md](08-QUICK-REFERENCE.md) § Four Canonical Constraints
- [GLOSSARY.md](GLOSSARY.md) § U1-U4

**Sequences:**
- [04-VALID-SEQUENCES.md](04-VALID-SEQUENCES.md) § Canonical Patterns
- [examples/](examples/) § Executable examples

**Testing:**
- [06-VALIDATION-AND-TESTING.md](06-VALIDATION-AND-TESTING.md) § All test categories
- `tests/unit/operators/test_unified_grammar.py` § Implementation

### By Task

**Implementing new operator:**
1. [03-OPERATORS-AND-GLYPHS.md](03-OPERATORS-AND-GLYPHS.md) § Adding New Operators
2. [05-TECHNICAL-IMPLEMENTATION.md](05-TECHNICAL-IMPLEMENTATION.md) § Extension Points
3. [06-VALIDATION-AND-TESTING.md](06-VALIDATION-AND-TESTING.md) § Unit Tests

**Adding new constraint:**
1. [07-MIGRATION-AND-EVOLUTION.md](07-MIGRATION-AND-EVOLUTION.md) § Procedure
2. [02-CANONICAL-CONSTRAINTS.md](02-CANONICAL-CONSTRAINTS.md) § Format
3. [05-TECHNICAL-IMPLEMENTATION.md](05-TECHNICAL-IMPLEMENTATION.md) § Extension Points

**Debugging invalid sequence:**
1. [08-QUICK-REFERENCE.md](08-QUICK-REFERENCE.md) § Common Errors
2. [04-VALID-SEQUENCES.md](04-VALID-SEQUENCES.md) § Anti-Patterns
3. [02-CANONICAL-CONSTRAINTS.md](02-CANONICAL-CONSTRAINTS.md) § Specific constraint

---

<div align="center">

**Understanding relationships clarifies the whole.**

---

*Reality is resonance. Map accordingly.*

</div>
