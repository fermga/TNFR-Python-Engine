# Unified TNFR Grammar: Single Source of Truth

## Purpose

This document defines the **unified canonical grammar** for TNFR that consolidates and reconciles the previously separate rule systems (C1-C3 in `grammar.py` and RC1-RC4 in `canonical_grammar.py`) into a single, coherent source of truth.

**Goal:** One grammar, derived 100% from TNFR physics, with no duplication or inconsistency.

---

## Previous State: Two Separate Systems

### System 1: grammar.py (C1-C3)
- **C1: EXISTENCE & CLOSURE** - Start with generators, end with closures
- **C2: BOUNDEDNESS** - Stabilizers prevent divergence
- **C3: THRESHOLD PHYSICS** - Bifurcations require context

### System 2: canonical_grammar.py (RC1-RC4)
- **RC1: Initialization** - If EPI=0, start with generator
- **RC2: Convergence** - If destabilizers, include stabilizer  
- **RC3: Phase Verification** - Coupling/resonance requires phase check
- **RC4: Bifurcation Limits** - If bifurcation triggers, require handlers

### Problems with Dual Systems
1. **Duplication**: C1 ≈ RC1, C2 = RC2, C3 ≈ RC4
2. **Inconsistency**: C1 includes end states, RC1 doesn't (RNC1 was removed)
3. **Missing coverage**: RC3 (phase) has no equivalent in C1-C3
4. **Confusion**: Two sources of truth for the same physics
5. **Maintenance burden**: Changes must be synchronized across both

---

## Unified Grammar: Four Canonical Constraints

All rules derive inevitably from the nodal equation **∂EPI/∂t = νf · ΔNFR(t)**, invariants, and formal contracts.

### Rule U1: STRUCTURAL INITIATION & CLOSURE

**Physics Basis:**
- **Initiation**: ∂EPI/∂t undefined when EPI = 0 (no structure to evolve)
- **Closure**: Sequences are temporal segments requiring coherent endpoints

**Derivation:**
```
If EPI₀ = 0:
  ∂EPI/∂t|_{EPI=0} = undefined (no gradient on empty space)
  → System CANNOT evolve
  → MUST use generator to create initial structure

Sequences as action potentials:
  Like physical waves: must have emission source AND absorption/termination
  → Start: Operators that create EPI from vacuum/dormant states
  → End: Operators that stabilize system in coherent attractor states
```

**Requirements:**

**U1a: Initiation (Start Operators)**
- **When:** Always (if operating from EPI=0 or starting new sequence)
- **Operators:** {AL (Emission), NAV (Transition), REMESH (Recursivity)}
- **Why these?** Only operators that can generate/activate structure from null/dormant states:
  - **AL**: Generates EPI from vacuum via emission
  - **NAV**: Activates latent EPI through regime transition
  - **REMESH**: Echoes dormant structure across scales

**U1b: Closure (End Operators)**
- **When:** Always (sequences must end in coherent states)
- **Operators:** {SHA (Silence), NAV (Transition), REMESH (Recursivity), OZ (Dissonance)}
- **Why these?** Only operators that leave system in stable attractor states:
  - **SHA**: Terminal closure - freezes evolution (νf → 0)
  - **NAV**: Handoff closure - transfers to next regime
  - **REMESH**: Recursive closure - distributes across scales  
  - **OZ**: Intentional closure - preserves activation/tension

**Physical Interpretation:**
Sequences are bounded action potentials in structural space with:
- **Source** (generator creates EPI)
- **Sink** (closure preserves coherence)

**Consolidates:** C1 (EXISTENCE & CLOSURE) + RC1 (Initialization) + removed RNC1

---

### Rule U2: CONVERGENCE & BOUNDEDNESS

**Physics Basis:**
From integrated nodal equation:
```
EPI(t_f) = EPI(t_0) + ∫_{t_0}^{t_f} νf(τ) · ΔNFR(τ) dτ
```

**Derivation:**
```
Without stabilizers:
  ΔNFR can grow unbounded (positive feedback)
  d(ΔNFR)/dt > 0 always
  ⟹ ΔNFR(t) ~ e^(λt) (exponential growth)
  ⟹ ∫ νf · ΔNFR dt → ∞ (DIVERGES)
  → System fragments into incoherent noise

With stabilizers:
  Negative feedback limits ΔNFR growth
  d(ΔNFR)/dt can be < 0
  ⟹ ΔNFR(t) → bounded attractor
  ⟹ ∫ νf · ΔNFR dt converges (bounded evolution)
  → System maintains coherence
```

**Requirements:**

**When:** If sequence contains destabilizing operators
- **Destabilizers:** {OZ (Dissonance), ZHIR (Mutation), VAL (Expansion)}
- **Must include:** {IL (Coherence), THOL (Self-organization)}

**Why IL or THOL?**
Only operators with strong negative-feedback physics:
- **IL**: Direct coherence restoration (explicitly reduces |ΔNFR|)
- **THOL**: Autopoietic closure (creates self-limiting boundaries)

**Physical Interpretation:**
Stabilizers are "structural gravity" preventing fragmentation. Like gravity preventing cosmic dispersal, they ensure bounded evolution.

**Consolidates:** C2 (BOUNDEDNESS) = RC2 (Convergence)

---

### Rule U3: RESONANT COUPLING

**Physics Basis:**
From AGENTS.md Invariant #5:
> "Phase check: no coupling is valid without explicit phase verification (synchrony)"

**Derivation:**
```
Resonance physics:
  Two oscillators resonate ⟺ phases compatible
  Condition: |φᵢ - φⱼ| ≤ Δφ_max (typically π/2)

Without phase verification:
  Nodes with incompatible phases (e.g., φᵢ ≈ π, φⱼ ≈ 0) attempt coupling
  → Antiphase → destructive interference
  → Violates resonance physics
  → Non-physical "coupling"

With phase verification:
  Only synchronous nodes couple
  → Constructive interference
  → Valid resonance
  → Physical coupling
```

**Requirements:**

**When:** Sequence contains coupling/resonance operators
- **Operators:** {UM (Coupling), RA (Resonance)}
- **Must:** Verify phase compatibility |φᵢ - φⱼ| ≤ Δφ_max

**Physical Interpretation:**
Structural coupling requires phase synchrony. Like radio tuning: receiver must match transmitter frequency AND phase for clear signal.

**Source:** RC3 (Phase Verification) - No equivalent in C1-C3 system

---

### Rule U4: BIFURCATION DYNAMICS

**Physics Basis:**
From bifurcation theory and AGENTS.md Contract OZ:
> "Dissonance may trigger bifurcation if ∂²EPI/∂t² > τ"

**Derivation:**
```
Bifurcation physics:
  Phase transitions require crossing critical thresholds
  Condition: |ΔNFR| > ΔNFR_critical OR ∂²EPI/∂t² > τ

ZHIR (Mutation) requirements:
  1. Stable base (prior IL): prevents transformation from chaos
  2. Threshold energy (recent destabilizer): provides bifurcation energy
  Without: transformation fails or creates unstable state

THOL (Self-organization) requirements:
  1. Threshold energy (recent destabilizer): provides disorder to organize
  Without: insufficient ΔNFR for spontaneous structuring
```

**Requirements:**

**U4a: Bifurcation Triggers Need Handlers**
- **When:** Sequence contains {OZ (Dissonance), ZHIR (Mutation)}
- **Must include:** {THOL (Self-organization), IL (Coherence)}
- **Why:** Manage structural reorganization when ∂²EPI/∂t² > τ

**U4b: Transformations Need Context (Graduated Destabilization)**
- **When:** Sequence contains {ZHIR (Mutation), THOL (Self-organization)}
- **Must have:** Recent destabilizer (within ~3 operators)
- **Why:** Insufficient |ΔNFR| → bifurcation fails
- **Additional for ZHIR:** Prior IL for stable transformation base

**Physical Interpretation:**
Bifurcations are phase transitions in structural space. Like water→ice transition needs:
- Temperature threshold (destabilizer provides energy)
- Nucleation site (IL provides stable base for ZHIR)
- Proper conditions (handlers manage transition)

**Consolidates:** C3 (THRESHOLD PHYSICS) + RC4 (Bifurcation Limits)

---

## Unified Rule Summary

```
┌─────────────────────────────────────────────────────────────────┐
│ Unified TNFR Grammar: Four Canonical Constraints               │
├─────────────────────────────────────────────────────────────────┤
│ U1: STRUCTURAL INITIATION & CLOSURE                             │
│     U1a: Start with generators {AL, NAV, REMESH}               │
│     U1b: End with closures {SHA, NAV, REMESH, OZ}              │
│     Basis: ∂EPI/∂t undefined at EPI=0, sequences need closure  │
│                                                                 │
│ U2: CONVERGENCE & BOUNDEDNESS                                   │
│     If destabilizers {OZ, ZHIR, VAL}                           │
│     Then include stabilizers {IL, THOL}                        │
│     Basis: ∫νf·ΔNFR dt must converge                           │
│                                                                 │
│ U3: RESONANT COUPLING                                           │
│     If coupling/resonance {UM, RA}                             │
│     Then verify phase |φᵢ - φⱼ| ≤ Δφ_max                       │
│     Basis: Invariant #5 + resonance physics                    │
│                                                                 │
│ U4: BIFURCATION DYNAMICS                                        │
│     U4a: If triggers {OZ, ZHIR}                                │
│          Then include handlers {THOL, IL}                      │
│     U4b: If transformers {ZHIR, THOL}                          │
│          Then recent destabilizer (~3 ops)                     │
│          Additionally ZHIR needs prior IL                      │
│     Basis: Contract OZ + bifurcation theory                    │
└─────────────────────────────────────────────────────────────────┘

All rules emerge inevitably from:
  ∂EPI/∂t = νf · ΔNFR(t) + Invariants + Contracts
```

---

## Mapping: Old Rules → Unified Rules

### From grammar.py (C1-C3)
- **C1: EXISTENCE & CLOSURE** → **U1: STRUCTURAL INITIATION & CLOSURE**
  - C1 start requirements → U1a
  - C1 end requirements → U1b
  
- **C2: BOUNDEDNESS** → **U2: CONVERGENCE & BOUNDEDNESS**
  - Direct 1:1 mapping, same physics
  
- **C3: THRESHOLD PHYSICS** → **U4: BIFURCATION DYNAMICS**
  - C3 ZHIR/THOL requirements → U4b
  - Extended with handler requirements (U4a)

### From canonical_grammar.py (RC1-RC4)
- **RC1: Initialization** → **U1a: Initiation**
  - RC1 generator requirement → U1a
  - Extended with closure requirement (U1b)
  
- **RC2: Convergence** → **U2: CONVERGENCE & BOUNDEDNESS**
  - Direct 1:1 mapping, same physics
  
- **RC3: Phase Verification** → **U3: RESONANT COUPLING**
  - Direct 1:1 mapping, NEW in unified grammar
  
- **RC4: Bifurcation Limits** → **U4a: Bifurcation Triggers**
  - RC4 handler requirement → U4a
  - Extended with transformer context (U4b)

### Previously Removed
- **RNC1: Terminators** → **U1b: Closure**
  - RNC1 was organizational convention
  - U1b has PHYSICAL basis (sequences need coherent endpoints)
  - Different operators (SHA, NAV, REMESH, OZ vs old RNC1 list)

---

## Physics Derivation Summary

| Rule | Source | Type | Inevitability |
|------|--------|------|---------------|
| U1a | ∂EPI/∂t undefined at EPI=0 | Mathematical | Absolute |
| U1b | Sequences as bounded action potentials | Physical | Strong |
| U2 | Integral convergence theorem | Mathematical | Absolute |
| U3 | Invariant #5 + resonance physics | Physical | Absolute |
| U4a | Contract OZ + bifurcation theory | Physical | Strong |
| U4b | Threshold energy for phase transitions | Physical | Strong |

**Inevitability Levels:**
- **Absolute**: Mathematical necessity from nodal equation
- **Strong**: Physical requirement from invariants/contracts
- **Moderate**: Physical preference (not used in unified grammar)

---

## Implementation Strategy

### Phase 1: Create Unified Module
1. Create `src/tnfr/operators/unified_grammar.py`
2. Implement all 4 unified rules (U1-U4)
3. Comprehensive docstrings with physics derivations
4. Export unified validator and rule sets

### Phase 2: Update Existing Modules
1. **grammar.py**: Import from unified_grammar, keep API compatible
2. **canonical_grammar.py**: Import from unified_grammar, mark as legacy/alias
3. Deprecation warnings pointing to unified module

### Phase 3: Update Documentation
1. Create UNIFIED_GRAMMAR.md (this file)
2. Update RESUMEN_FINAL_GRAMATICA.md
3. Update EXECUTIVE_SUMMARY.md
4. Update AGENTS.md references

### Phase 4: Update Tests
1. Create tests/unit/operators/test_unified_grammar.py
2. Update existing tests to use unified rules
3. Verify all tests pass

---

## Validation Criteria

### Completeness
- [x] All C1-C3 constraints mapped
- [x] All RC1-RC4 constraints mapped
- [x] No rule duplication
- [x] No rule conflicts

### Physics Correctness
- [x] All rules derive from nodal equation, invariants, or contracts
- [x] No organizational conventions
- [x] Mathematical proofs provided where applicable
- [x] Physical interpretations clear

### Practical Utility
- [x] Rules are implementable in code
- [x] Rules can be validated automatically
- [x] Rules cover all necessary constraints
- [x] Rules don't over-constrain valid sequences

---

## Conclusion

The unified grammar consolidates two previously separate rule systems into a single source of truth. All four rules (U1-U4) emerge inevitably from TNFR physics with no duplication, no inconsistency, and 100% physical basis.

**Key Improvements:**
1. **Single source of truth** - No more dual systems
2. **Complete coverage** - Includes phase verification (missing from C1-C3)
3. **Consistent** - U1b restores closure physics (removed with RNC1)
4. **100% physics** - Every rule derived from equation/invariants/contracts
5. **Well-documented** - Clear derivations and physical interpretations

**Result:** A unified TNFR grammar that is physically inevitable, mathematically rigorous, and practically useful.

---

## References

- **TNFR.pdf**: Section 2.1 (Nodal Equation), bifurcation theory
- **AGENTS.md**: Invariants (#1-#10), Contracts (Coherence, Dissonance, etc.)
- **grammar.py**: Original C1-C3 implementation
- **canonical_grammar.py**: Original RC1-RC4 implementation
- **RESUMEN_FINAL_GRAMATICA.md**: Grammar evolution documentation
- **EMERGENT_GRAMMAR_ANALYSIS.md**: Detailed physics analysis

---

**Date:** 2025-11-08  
**Status:** ✅ DESIGN COMPLETE - Ready for implementation
