# TNFR Grammar Glossary

**Operational definitions of key TNFR grammar terms**

[🏠 Home](README.md) • [🌊 Concepts](01-FUNDAMENTAL-CONCEPTS.md) • [📐 Constraints](02-CANONICAL-CONSTRAINTS.md) • [⚙️ Operators](03-OPERATORS-AND-GLYPHS.md)

---

## Purpose

This glossary provides **precise operational definitions** of terms used in TNFR grammar documentation. Each entry includes context, notation, and references.

---

## A

### Attractor
**Definition:** Stable state or regime in structural space toward which dynamics converge.

**Context:** Sequences must end in attractor states (closures).

**Types:** Fixed point (Silence), limit cycle (Transition, Recursivity), strange attractor (Dissonance).

**References:** [01-FUNDAMENTAL-CONCEPTS.md](01-FUNDAMENTAL-CONCEPTS.md), dynamical systems theory

---

## B

### Banach Space (B_EPI)
**Definition:** Complete normed vector space where EPI values reside.

**Notation:** B_EPI

**Context:** EPI is a point in this infinite-dimensional function space.

**Properties:** Supports convergence, continuity, operator theory.

**References:** [01-FUNDAMENTAL-CONCEPTS.md](01-FUNDAMENTAL-CONCEPTS.md)

### Bifurcation
**Definition:** Qualitative change in system dynamics when parameter crosses threshold.

**Equation:** Occurs when ∂²EPI/∂t² > τ (threshold)

**Context:** Triggers (OZ, ZHIR) can cause bifurcations; handlers (IL, THOL) control them.

**Related:** U4 (BIFURCATION DYNAMICS)

**References:** [02-CANONICAL-CONSTRAINTS.md](02-CANONICAL-CONSTRAINTS.md)

### Boundedness
**Definition:** Property that integral ∫νf·ΔNFR dt remains finite.

**Context:** Required for coherence preservation.

**Related:** U2 (CONVERGENCE & BOUNDEDNESS)

**References:** [01-FUNDAMENTAL-CONCEPTS.md](01-FUNDAMENTAL-CONCEPTS.md)

---

## C

### C(t) - Coherence
**Definition:** Global network stability measure.

**Range:** [0, 1]

**Interpretation:**
- C(t) > 0.7: Strong coherence
- C(t) < 0.3: Fragmentation risk

**Contract:** Coherence operator must not reduce C(t) (except in controlled dissonance).

**References:** [01-FUNDAMENTAL-CONCEPTS.md](01-FUNDAMENTAL-CONCEPTS.md)

### Canonicity
**Definition:** Level of inevitability of a constraint from TNFR physics.

**Levels:**
- **ABSOLUTE:** Mathematically or physically necessary (violation impossible)
- **STRONG:** Physically required (violation leads to non-physical behavior)
- **MODERATE:** Best practice (violation leads to suboptimal behavior)

**Current system:** U1, U2, U3 are ABSOLUTE; U4 is STRONG.

**References:** [02-CANONICAL-CONSTRAINTS.md](02-CANONICAL-CONSTRAINTS.md)

### Closure
**Definition:** Operator that can terminate a sequence, leaving system in stable attractor.

**Set:** CLOSURES = {SHA (Silence), NAV (Transition), REMESH (Recursivity), OZ (Dissonance)}

**Related:** U1b

**References:** [02-CANONICAL-CONSTRAINTS.md](02-CANONICAL-CONSTRAINTS.md), [03-OPERATORS-AND-GLYPHS.md](03-OPERATORS-AND-GLYPHS.md)

### Coherence (concept)
**Definition:** Property of patterns that maintain structural integrity through reorganization.

**Context:** Central concept in TNFR - reality is coherence, not substance.

**Measured by:** C(t), Si

**References:** [01-FUNDAMENTAL-CONCEPTS.md](01-FUNDAMENTAL-CONCEPTS.md)

### Convergence
**Definition:** Property that a sequence or integral approaches a finite limit.

**Context:** ∫νf·ΔNFR dt must converge for bounded evolution.

**Related:** U2 (CONVERGENCE & BOUNDEDNESS)

**References:** [01-FUNDAMENTAL-CONCEPTS.md](01-FUNDAMENTAL-CONCEPTS.md), [02-CANONICAL-CONSTRAINTS.md](02-CANONICAL-CONSTRAINTS.md)

### Coupling
**Definition:** Structural link between nodes via phase synchronization.

**Operator:** UM (Coupling)

**Requirement:** Must verify phase compatibility (U3).

**Effect:** Enables information exchange, creates graph edge.

**References:** [03-OPERATORS-AND-GLYPHS.md](03-OPERATORS-AND-GLYPHS.md)

---

## D

### Destabilizer
**Definition:** Operator that increases |ΔNFR|, introducing instability.

**Set:** DESTABILIZERS = {OZ (Dissonance), ZHIR (Mutation), VAL (Expansion)}

**Requirement:** Must be balanced by stabilizers (U2).

**Context:** Creates exploration, growth, or phase transitions.

**References:** [02-CANONICAL-CONSTRAINTS.md](02-CANONICAL-CONSTRAINTS.md), [03-OPERATORS-AND-GLYPHS.md](03-OPERATORS-AND-GLYPHS.md)

### ΔNFR (Delta NFR) - Reorganization Gradient
**Definition:** Structural pressure driving change; mismatch between node and environment.

**Notation:** ΔNFR(t)

**Sign:**
- Positive: Expansion pressure
- Negative: Contraction pressure

**Magnitude:** Intensity of reorganization drive

**Equation:** Appears in ∂EPI/∂t = νf · ΔNFR(t)

**NOT:** An ML "error gradient" or "loss"

**References:** [01-FUNDAMENTAL-CONCEPTS.md](01-FUNDAMENTAL-CONCEPTS.md)

---

## E

### EPI (Estructura Primaria de Información)
**Definition:** Coherent structural form of a node.

**Space:** Lives in Banach space B_EPI

**Properties:**
- Changes ONLY via structural operators
- Can nest (fractality)
- Preserves identity through reorganization

**Range:** Typically [0, ∞) for scalar, complex structure for hierarchical

**Equation:** ∂EPI/∂t = νf · ΔNFR(t)

**References:** [01-FUNDAMENTAL-CONCEPTS.md](01-FUNDAMENTAL-CONCEPTS.md)

---

## F

### Fractality (Operational)
**Definition:** Property where EPIs can nest within EPIs without losing identity.

**Context:** THOL (Self-organization) and REMESH (Recursivity) create fractal structures.

**Invariant:** Nested EPIs maintain functional identity.

**References:** [01-FUNDAMENTAL-CONCEPTS.md](01-FUNDAMENTAL-CONCEPTS.md), [03-OPERATORS-AND-GLYPHS.md](03-OPERATORS-AND-GLYPHS.md)

---

## G

### Generator
**Definition:** Operator that can create EPI from vacuum (EPI=0).

**Set:** GENERATORS = {AL (Emission), NAV (Transition), REMESH (Recursivity)}

**Requirement:** Must start sequence with generator when EPI=0 (U1a).

**References:** [02-CANONICAL-CONSTRAINTS.md](02-CANONICAL-CONSTRAINTS.md), [03-OPERATORS-AND-GLYPHS.md](03-OPERATORS-AND-GLYPHS.md)

### Glyph
**Definition:** Short symbolic name for an operator (e.g., AL for Emission).

**Origin:** From phonetic/linguistic encoding of operators.

**Usage:** Concise notation in sequences.

**References:** [03-OPERATORS-AND-GLYPHS.md](03-OPERATORS-AND-GLYPHS.md)

### Grammar
**Definition:** Set of rules (U1-U5) governing valid operator sequences.

**Purpose:** Ensure sequences respect TNFR physics.

**Components:** U1 (INITIATION & CLOSURE), U2 (CONVERGENCE), U3 (COUPLING), U4 (BIFURCATION), U5 (MULTI-SCALE)

**References:** [02-CANONICAL-CONSTRAINTS.md](02-CANONICAL-CONSTRAINTS.md)

---

## H

### Handler
**Definition:** Operator that controls bifurcations triggered by other operators.

**Set:** BIFURCATION_HANDLERS = {IL (Coherence), THOL (Self-organization)}

**Requirement:** Triggers need handlers (U4a).

**References:** [02-CANONICAL-CONSTRAINTS.md](02-CANONICAL-CONSTRAINTS.md)

### Hz_str (Structural Hertz)
**Definition:** Unit of structural frequency (νf).

**Context:** Rate of reorganization capacity, analogous to frequency in Hz but for structure.

**NOT:** Regular Hz (cycles per second).

**References:** [01-FUNDAMENTAL-CONCEPTS.md](01-FUNDAMENTAL-CONCEPTS.md)

---

## I

### Integral Convergence Theorem
**Definition:** Mathematical requirement that ∫νf·ΔNFR dt < ∞ for bounded evolution.

**Physical Basis:** Without convergence, EPI → ∞ (explosion) or fragments.

**Grammar Implication:** U2 (destabilizers need stabilizers).

**References:** [01-FUNDAMENTAL-CONCEPTS.md](01-FUNDAMENTAL-CONCEPTS.md), [02-CANONICAL-CONSTRAINTS.md](02-CANONICAL-CONSTRAINTS.md)

### Invariant
**Definition:** Property that must be preserved by all valid operations.

**Examples:**
- EPI changes only via operators
- Coherence not reduced (except controlled)
- Phase compatibility for coupling

**References:** [../../AGENTS.md](../../AGENTS.md), [06-VALIDATION-AND-TESTING.md](06-VALIDATION-AND-TESTING.md)

---

## N

### Nodal Equation
**Definition:** ∂EPI/∂t = νf · ΔNFR(t)

**Meaning:** Rate of structural change = Reorganization capacity × Structural pressure

**Centrality:** All grammar rules derive from this equation.

**References:** [01-FUNDAMENTAL-CONCEPTS.md](01-FUNDAMENTAL-CONCEPTS.md)

---

## O

### Operator
**Definition:** Resonant transformation applied to nodes. Only way to modify EPI.

**Count:** 13 canonical operators

**Properties:**
- Physically grounded (not arbitrary)
- Classified by grammar role
- Has preconditions and postconditions

**References:** [03-OPERATORS-AND-GLYPHS.md](03-OPERATORS-AND-GLYPHS.md)

---

## P

### Phase (φ or θ)
**Definition:** Network synchrony parameter, timing of reorganization cycles.

**Range:** [0, 2π) radians

**Role:** Determines coupling compatibility.

**Coupling condition:** |φᵢ - φⱼ| ≤ Δφ_max (typically π/2)

**Related:** U3 (RESONANT COUPLING)

**References:** [01-FUNDAMENTAL-CONCEPTS.md](01-FUNDAMENTAL-CONCEPTS.md), [02-CANONICAL-CONSTRAINTS.md](02-CANONICAL-CONSTRAINTS.md)

---

## R

### Reorganization
**Definition:** Structural change while maintaining identity. Core dynamic in TNFR.

**Rate:** Determined by νf · ΔNFR

**Context:** Nodes don't "move" or "change state" - they reorganize.

**References:** [01-FUNDAMENTAL-CONCEPTS.md](01-FUNDAMENTAL-CONCEPTS.md)

### Resonance
**Definition:** Amplification and propagation of patterns through phase-compatible coupling.

**Operator:** RA (Resonance)

**Requirement:** Phase compatibility (U3).

**Effect:** Increases effective coupling, pattern spreads without identity loss.

**References:** [03-OPERATORS-AND-GLYPHS.md](03-OPERATORS-AND-GLYPHS.md)

---

## S

### Sense Index (Si)
**Definition:** Node-level capacity for stable reorganization.

**Range:** [0, 1+]

**Interpretation:**
- Si > 0.8: Excellent stability
- Si < 0.4: Changes may cause bifurcation

**References:** [01-FUNDAMENTAL-CONCEPTS.md](01-FUNDAMENTAL-CONCEPTS.md)

### Sequence
**Definition:** Ordered list of operators to be applied.

**Validation:** Must satisfy U1-U5 constraints.

**Examples:** [Emission, Coherence, Silence]

**References:** [04-VALID-SEQUENCES.md](04-VALID-SEQUENCES.md)

### Stabilizer
**Definition:** Operator that reduces |ΔNFR| through negative feedback.

**Set:** STABILIZERS = {IL (Coherence), THOL (Self-organization)}

**Requirement:** Required to balance destabilizers (U2).

**References:** [02-CANONICAL-CONSTRAINTS.md](02-CANONICAL-CONSTRAINTS.md), [03-OPERATORS-AND-GLYPHS.md](03-OPERATORS-AND-GLYPHS.md)

### Structural Triad
**Definition:** The three essential properties of every node: Form (EPI), Frequency (νf), Phase (φ).

**Context:** Complete specification of node state.

**References:** [01-FUNDAMENTAL-CONCEPTS.md](01-FUNDAMENTAL-CONCEPTS.md)

---

## T

### Telemetry
**Definition:** Measurement and export of essential TNFR metrics.

**Essential metrics:** C(t), Si, νf, φ, ΔNFR

**Purpose:** Verification, debugging, reproducibility.

**References:** [01-FUNDAMENTAL-CONCEPTS.md](01-FUNDAMENTAL-CONCEPTS.md), [05-TECHNICAL-IMPLEMENTATION.md](05-TECHNICAL-IMPLEMENTATION.md)

### Transformer
**Definition:** Operator that changes phase or regime, requiring elevated energy.

**Set:** TRANSFORMERS = {ZHIR (Mutation), THOL (Self-organization)}

**Requirement:** Needs recent destabilizer (~3 ops) and, for ZHIR, prior coherence (U4b).

**References:** [02-CANONICAL-CONSTRAINTS.md](02-CANONICAL-CONSTRAINTS.md)

### Trigger
**Definition:** Operator that can initiate bifurcation.

**Set:** BIFURCATION_TRIGGERS = {OZ (Dissonance), ZHIR (Mutation)}

**Requirement:** Needs handler (U4a).

**References:** [02-CANONICAL-CONSTRAINTS.md](02-CANONICAL-CONSTRAINTS.md)

---

## U

### U1: STRUCTURAL INITIATION & CLOSURE
**Definition:** Constraint ensuring sequences have valid start (U1a) and end (U1b).

**U1a:** Start with generator when EPI=0
**U1b:** End with closure

**Canonicity:** ABSOLUTE

**References:** [02-CANONICAL-CONSTRAINTS.md](02-CANONICAL-CONSTRAINTS.md)

### U2: CONVERGENCE & BOUNDEDNESS
**Definition:** Constraint requiring destabilizers to be balanced by stabilizers.

**Physical Basis:** Integral convergence theorem

**Canonicity:** ABSOLUTE

**References:** [02-CANONICAL-CONSTRAINTS.md](02-CANONICAL-CONSTRAINTS.md)

### U3: RESONANT COUPLING
**Definition:** Constraint requiring phase compatibility for coupling/resonance.

**Condition:** |φᵢ - φⱼ| ≤ Δφ_max

**Canonicity:** ABSOLUTE

**References:** [02-CANONICAL-CONSTRAINTS.md](02-CANONICAL-CONSTRAINTS.md)

### U4: BIFURCATION DYNAMICS
**Definition:** Constraint governing bifurcation control and transformer context.

**U4a:** Triggers need handlers
**U4b:** Transformers need recent destabilizer (+ ZHIR needs prior IL)

**Canonicity:** STRONG

**References:** [02-CANONICAL-CONSTRAINTS.md](02-CANONICAL-CONSTRAINTS.md)

---

## V

### νf (Nu-f) - Structural Frequency
**Definition:** Rate of reorganization capacity.

**Units:** Hz_str (structural hertz)

**Range:** ℝ⁺ (positive reals)

**Equation:** ∂EPI/∂t = νf · ΔNFR(t)

**Death condition:** νf → 0 means node cannot reorganize

**References:** [01-FUNDAMENTAL-CONCEPTS.md](01-FUNDAMENTAL-CONCEPTS.md)

---

## Quick Lookup

**Most Important Terms:**
- EPI: Structural form
- νf: Reorganization rate
- ΔNFR: Structural pressure
- φ (theta): Phase
- C(t): Coherence
- Si: Sense index

**Constraints:**
- U1: Initiation & closure
- U2: Convergence
- U3: Phase compatibility
- U4: Bifurcation control

**Operator Classes:**
- Generators: Create from vacuum
- Closures: End sequences
- Stabilizers: Reduce ΔNFR
- Destabilizers: Increase ΔNFR

---

## References

**Full Documentation:**
- [01-FUNDAMENTAL-CONCEPTS.md](01-FUNDAMENTAL-CONCEPTS.md) - Conceptual foundations
- [02-CANONICAL-CONSTRAINTS.md](02-CANONICAL-CONSTRAINTS.md) - Formal constraints
- [03-OPERATORS-AND-GLYPHS.md](03-OPERATORS-AND-GLYPHS.md) - Operator catalog

**Repository:**
- [../../GLOSSARY.md](../../GLOSSARY.md) - General project glossary
- [../../UNIFIED_GRAMMAR_RULES.md](../../UNIFIED_GRAMMAR_RULES.md) - Mathematical derivations
- [../../TNFR.pdf](../../TNFR.pdf) - Complete theory

---

<div align="center">

**Precise definitions enable precise thinking.**

---

*Reality is resonance. Define accordingly.*

</div>
