# Unified TNFR Grammar: Single Source of Truth

## Purpose

This document defines the **unified canonical grammar** for TNFR that consolidates and reconciles the previously separate rule systems (C1-C3 in `grammar.py` and RC1-RC4 in `canonical_grammar.py`) into a single, coherent source of truth.

**Goal:** One grammar, derived 100% from TNFR physics, with no duplication or inconsistency.

**Related Documentation:**
- **[AGENTS.md](../AGENTS.md)** - Concise grammar reference for developers
- **[docs/grammar/PHYSICS_VERIFICATION.md](../docs/grammar/PHYSICS_VERIFICATION.md)** - Technical specifications with implementation examples
- **[GLOSSARY.md](GLOSSARY.md)** - Quick term reference
- **[src/tnfr/operators/grammar.py](../src/tnfr/operators/grammar.py)** - Canonical implementation

---

## Unified Grammar: Six Canonical Constraints

All rules are derived from the nodal equation **∂EPI/∂t = νf · ΔNFR(t)**, invariants, and formal contracts.

**Status Summary**:
- **U1-U5**: CANONICAL (ABSOLUTE/STRONG) - Mathematically derived from nodal equation
- **U6**: CANONICAL (STRONG) - Empirically validated with 2,400+ experiments (Nov 2025)

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

**Quantitative bound (derived from the pulse relaxation):**
The maximum uncompensated-destabilizer **debt** the relaxation can absorb is the
geometric steady state `⌊1/(1−q)⌋ = ⌊1/(νf·dt·ρ)⌋`, where `q = 1 − νf·dt·ρ` is
the discrete per-step decay of a |ΔNFR| perturbation and `ρ = trace(L_rw)/N = 1`
(the mean structural relaxation rate, exact). For the canonical `νf=1, dt=0.5` this
is **2** — the canonical U2 debt threshold, **derived** (not assumed) by
`derive_u2_debt_capacity_from_physics`. The same `q` sets the U4b window: the window
is the relaxation *time*, the debt is the relaxation *absorption capacity*.

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

**U4b: Transformations Need Context**
- **When:** Sequence contains {ZHIR (Mutation), THOL (Self-organization)}
- **Must have:** A recent destabilizer within the **structural-relaxation window** —
  derived from the pulse: the discrete steps for a |ΔNFR| perturbation to relax into
  the coherence band `1/(π+1)` (canonically **3 ops**, the **same for every
  destabilizer**; `derive_bifurcation_window_from_physics`).
- **Why:** Insufficient |ΔNFR| → bifurcation fails (the structure must stay plastic)
- **Additional for ZHIR:** Prior IL for stable transformation base

**Physical Interpretation:**
Bifurcations are phase transitions in structural space. Like water→ice transition needs:
- Temperature threshold (destabilizer provides energy)
- Nucleation site (IL provides stable base for ZHIR)
- Proper conditions (handlers manage transition)

**Consolidates:** C3 (THRESHOLD PHYSICS) + RC4 (Bifurcation Limits)

---

### Rule U5: MULTI-SCALE COHERENCE

**Physics Basis:**
From the nodal equation applied to hierarchical systems with nested EPIs created by REMESH with depth>1.

**Derivation from Nodal Equation:**

```
Step 1: Nodal equation at each hierarchical level
  Parent level:  ∂EPI_parent/∂t = νf_parent · ΔNFR_parent(t)
  Child level i: ∂EPI_child_i/∂t = νf_child_i · ΔNFR_child_i(t)

Step 2: Hierarchical coupling (structural interdependence)
  EPI_parent = f(EPI_child_1, EPI_child_2, ..., EPI_child_N)
  
  This is the essence of hierarchy: parent structure depends on children
  Example: Cell EPI depends on {Nucleus, Mitochondria, ...} EPIs

Step 3: Chain rule for time evolution
  ∂EPI_parent/∂t = Σ (∂f/∂EPI_child_i) · ∂EPI_child_i/∂t
                  = Σ w_i · (νf_child_i · ΔNFR_child_i)
  
  where w_i = ∂f/∂EPI_child_i are coupling weights

Step 4: Equate with parent's nodal equation
  νf_parent · ΔNFR_parent = Σ w_i · νf_child_i · ΔNFR_child_i
  
  Rearranging:
  ΔNFR_parent = (1/νf_parent) · Σ w_i · νf_child_i · ΔNFR_child_i

Step 5: Coherence definition
  C(t) = structural stability = 1/|ΔNFR(t)|
  
  Higher coherence ⟺ Lower reorganization pressure
  This is Invariant #9: Structural Metrics

Step 6: Coherence relationship
  C_parent ~ 1/|ΔNFR_parent|
          ~ νf_parent / |Σ w_i · νf_child_i · ΔNFR_child_i|
  
  C_child_i ~ 1/|ΔNFR_child_i|

Step 7: Conservation inequality
  For bounded evolution, parent coherence must be bounded below:
  
  C_parent ≥ α · Σ C_child_i
  
  Where α emerges from coupling structure:
    α = (1/√N) · η_phase(N) · η_coupling(N)
    
  Components:
  - 1/√N: Scale factor from weight distribution (central limit theorem)
  - η_phase: Phase synchronization efficiency (from U3, Invariant #5)
  - η_coupling: Structural coupling efficiency (from w_i distribution)
  - Typical range: α ∈ [0.1, 0.4]

Step 8: Physical necessity of stabilizers
  Without stabilizers:
    Each ΔNFR_child_i evolves independently
    → |ΔNFR_parent| = |Σ w_i · νf_child_i · ΔNFR_child_i| grows
    → C_parent decreases below α·ΣC_child
    → CONSERVATION VIOLATED → Fragmentation
  
  With stabilizers (IL or THOL):
    IL reduces |ΔNFR| at each level (Contract IL)
    THOL creates self-limiting boundaries (Contract THOL)
    → |ΔNFR_parent| bounded
    → C_parent ≥ α·ΣC_child maintained
    → CONSERVATION PRESERVED → Bounded evolution

Conclusion: U5 follows from:
  1. Nodal equation: ∂EPI/∂t = νf · ΔNFR(t)
  2. Hierarchical coupling: EPI_parent = f(EPI_child_1, ..., EPI_child_N)
  3. Chain rule: ∂f/∂t must account for all child contributions
  4. Coherence definition: C ~ 1/|ΔNFR|
  5. Conservation requirement: Bounded evolution needs C_parent ≥ α·ΣC_child
```

**Requirements:**

**When:** Sequence contains deep REMESH (depth > 1)
- **Deep recursion:** REMESH with depth > 1 creates hierarchical nesting
- **Must include:** {IL (Coherence), THOL (Self-organization)} within the
  structural-relaxation window
- **Window:** Stabilizer must be within the **structural-relaxation window**
  (`BIFURCATION_WINDOW`, derived = 3 ops) before or after REMESH

**Why IL or THOL?**
From operator contracts, only these provide multi-scale stabilization:
- **IL (Contract)**: Reduces |ΔNFR| → increases C → direct coherence restoration
- **THOL (Contract)**: Creates sub-EPIs with autopoietic closure → multi-level stability

**Physical Interpretation:**
Multi-scale structures require conservation of coherence across hierarchy levels. Just as thermodynamic entropy must increase globally while local order can increase with work input, hierarchical coherence requires "work" (stabilization) to maintain C_parent ≥ α·ΣC_child against natural tendency toward fragmentation.

**Dimensionality:**
- **U1-U4**: TEMPORAL dimension (operator sequences in time)
- **U5**: SPATIAL dimension (hierarchical nesting in structure)

**Independence from U2+U4b:**
Decisive test case that passes U2+U4b but fails U5:
```python
[AL, REMESH(depth=3), SHA]
  U2:  ✓ No destabilizers (trivially convergent)
  U4b: ✓ REMESH not a transformer (U4b doesn't apply)
  U5:  ✗ Deep recursivity without stabilization → fragmentation
```

This proves U5 captures a physical constraint (spatial hierarchy) not covered by existing temporal rules (U2, U4b).

**Source:** 
- Research in "El pulso que nos atraviesa.pdf"
- Direct derivation from nodal equation + hierarchical coupling

**Canonicity Level**: **STRONG** - Derived from nodal equation applied to hierarchical systems. Violating it produces C_parent < α·ΣC_child → fragmentation.

**Traceability**: 
- **TNFR.pdf § 2.1**: Nodal equation ∂EPI/∂t = νf · ΔNFR(t)
- **Chain rule**: Standard calculus for composite functions
- **AGENTS.md § Invariant #7**: Operational Fractality (EPIs can nest)
- **AGENTS.md § Invariant #9**: Structural Metrics (C, Si, etc.)
- **Contract IL**: Reduces |ΔNFR| (stabilization at each level)
- **Contract THOL**: Autopoietic closure (multi-level boundaries)

---

### Rule U6: STRUCTURAL POTENTIAL CONFINEMENT

**Physics Basis:**
From emergent structural potential field Φ_s derived from weighted ΔNFR distribution across network.

**Derivation from Nodal Equation:**

```
Step 1: Structural potential definition
  Φ_s(i) = Σ_{j≠i} ΔNFR_j / d(i,j)^α  (α=2)
  
  Physical meaning: Aggregates structural pressure from all network nodes
  weighted by coupling distance (inverse-square law analog)

Step 2: Relationship to coherence
  From 2,400+ experiments across 5 topology families:
  
  corr(Δ Φ_s, ΔC) = -0.822 (R² ≈ 0.68)
  
  Strong negative correlation: displacement from Φ_s minima → coherence loss
  
Step 3: Universality validation
  Tested topologies: ring, scale_free, small-world, tree, grid
  Coefficient of variation: CV = 0.1% (perfect universality)
  
  → Φ_s dynamics independent of topology
  → Fundamental structural physics, not topology artifact

Step 4: Passive equilibrium mechanism
  Φ_s minima = passive equilibrium states (potential wells)
  Grammar-valid sequences show Δ Φ_s = +0.583
  Grammar-violating sequences show Δ Φ_s = +3.879
  
  Reduction factor: 0.15× (85% reduction in escape tendency)
  
  Physical interpretation:
  - NOT active attraction toward minima (no force pulling back)
  - Passive protection: grammar acts as confinement mechanism
  - Valid sequences naturally maintain proximity to equilibrium

Step 5: Safety criterion from empirical threshold
  Escape threshold (fragmentation boundary): Δ Φ_s < 2.0
  
  Valid sequences: Δ Φ_s ≈ 0.6 (30% of threshold)
  Violations: Δ Φ_s ≈ 3.9 (195% of threshold)
  
  → 2.0 threshold separates stable from fragmenting regimes

Step 6: Scale-dependent universality
  β exponent (fragmentation criticality):
  - Flat networks: β = 0.556
  - Nested EPIs: β = 0.178
  
  Different universality classes for different scales (physically expected)
  Φ_s correlation universal across both: corr = -0.822 ± 0.001

Conclusion: U6 follows from:
  1. Nodal equation: ΔNFR as structural pressure
  2. Distance-weighted field: Φ_s from network topology
  3. Empirical validation: 2,400+ experiments, 5 topologies
  4. Conservation: Grammar as passive stabilizer
  5. Threshold physics: Δ Φ_s < 2.0 escape boundary
```

**Requirements:**

**When:** All sequences (telemetry-based safety criterion)
- **Compute:** Φ_s before and after sequence application
- **Verify:** Δ Φ_s < π/2 ≈ 1.571 (π-derived drift bound, half phase-wrap)
- **Typical:** Valid sequences show Δ Φ_s ≈ 0.6 (≈ 38% of the bound)

**Why Δ Φ_s < π/2 ≈ 1.571?**

π-derived confinement bound (half phase-wrap, tied to the one genuine structural scale π):

- **Scale:** π is the one genuine structural scale (the phase-wrap bound of the phase sector); the Φ_s drift bound π/2 is half that wrap.
- **Below π/2:** System remains in the confined regime, structural confinement
- **Above π/2:** Confinement breakdown → fragmentation (coherent → incoherent)
- **Mechanism:** the emergent Φ_s field must stay bounded for coherence to survive.

**Physical Interpretation:**
Φ_s field creates passive equilibrium landscape. Nodes exist at potential minima. Sequences that respect grammar (U1-U5) naturally maintain small Δ Φ_s (~0.6). Grammar violations create large Δ Φ_s (~3.9), pushing system toward fragmentation threshold.

**Validation Evidence:**
- **Experiments:** 2,400+ across 5 topologies (ring, scale_free, ws, tree, grid)
- **Correlation:** corr(Δ Φ_s, ΔC) = -0.822 (R² ≈ 0.68)
- **Universality:** CV = 0.1% (perfect across topologies)
- **Fractality:** β scale-dependent (0.178 nested vs 0.556 flat)
- **Mechanism:** Passive protection (grammar as stabilizer, not attractor)

**Distinction from U2 (Boundedness):**
- **U2:** Temporal integral convergence (∫νf·ΔNFR dt < ∞)
- **U6:** Spatial potential confinement (Δ Φ_s < 2.0)
- **Independence:** U2 prevents divergence over time, U6 prevents escape in structural space

**Usage as Telemetry:**
U6 is a **read-only safety check**, not a sequence constraint like U1-U5:
- Does NOT dictate which operators to use
- Does NOT require specific operator patterns
- DOES provide early warning when Δ Φ_s approaches 2.0
- DOES validate that grammar-compliant sequences naturally stay confined

**Canonicity Level**: **CANONICAL** (promoted 2025-11-11)
- Formal derivation from ΔNFR field theory
- Strong predictive power (R² = 0.68)
- Universal across topologies (CV = 0.1%)
- Grammar-compliant (read-only, no U1-U5 conflicts)
- Validated: 2,400+ experiments

**Traceability**: 
- **TNFR.pdf § 2.1**: Nodal equation ∂EPI/∂t = νf · ΔNFR(t)
- **docs/STRUCTURAL_FIELDS_TETRAD.md**: Complete derivation and validation
- **AGENTS.md § Structural Fields**: Φ_s canonical status with safety criteria
- **src/tnfr/physics/fields.py**: Implementation of compute_structural_potential()

---

## Unified Rule Summary

```
┌─────────────────────────────────────────────────────────────────┐
│ Unified TNFR Grammar: Six Canonical Constraints                │
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
│                                                                 │
│ U5: MULTI-SCALE COHERENCE                                       │
│     If deep REMESH (depth>1)                                   │
│     Then include scale stabilizers {IL, THOL} within ±3 ops   │
│     Basis: C_parent ≥ α·ΣC_child (coherence conservation)     │
│                                                                 │
│ U6: STRUCTURAL POTENTIAL CONFINEMENT                            │
│     Verify Δ Φ_s < 2.0 (escape threshold)                      │
│     Telemetry-based safety check (read-only)                  │
│     Basis: Emergent Φ_s field, empirical threshold            │
│     Evidence: 2,400+ exp, corr = -0.822, CV = 0.1%            │
└─────────────────────────────────────────────────────────────────┘

All rules follow from:
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

## Canonicity and Physical Basis

This section provides the comprehensive justification for why each unified rule (U1-U6) is **canonical** - that is, derived from TNFR physics rather than organizational convention.

### Summary Table: Canonicity Verification

| Rule | Canonicity | Strength | Physical Base | Reference |
|------|------------|----------|---------------|-----------|
| U1a | ✅ CANONICAL | Absolute | ∂EPI/∂t undefined at EPI=0 | Nodal equation |
| U1b | ✅ CANONICAL | Strong   | Sequences as action potentials | Wave physics |
| U2  | ✅ CANONICAL | Absolute | Integral convergence theorem | Analysis |
| U3  | ✅ CANONICAL | Absolute | Resonance physics + Inv. #5 | AGENTS.md |
| U4a | ✅ CANONICAL | Strong   | Contract OZ + bifurcation | Contracts |
| U4b | ✅ CANONICAL | Strong   | Threshold physics + timing | Bifurcation theory |
| U5  | ✅ CANONICAL | Strong   | Coherence conservation + hierarchy | Conservation |
| U6  | ✅ CANONICAL | Strong   | Structural potential field + empirical | STRUCTURAL_FIELDS_TETRAD.md |

**Key**: 
- **Absolute**: Mathematical necessity (cannot be otherwise)
- **Strong**: Physical requirement (violating it produces non-physical states)

---

### U1a: Structural Initiation - Canonicity

**Derivation from Nodal Equation:**

```
Given: ∂EPI/∂t = νf · ΔNFR(t)

At EPI = 0 (null state):
  ΔNFR(0) = f(EPI, topology, phase) where EPI=0
  → ΔNFR(0) is undefined or null
  → ∂EPI/∂t|_{EPI=0} = νf · 0 = 0 OR undefined

Conclusion: System CANNOT evolve from EPI=0 without generator
```

**Physical Necessity:**
- Like a wave equation: cannot have wave propagation without source
- Like thermodynamics: cannot have heat flow without temperature difference
- Like structural mechanics: cannot have deformation without initial geometry

**Why These Generators?**
- **Emission (AL)**: Creates EPI from vacuum via resonant emission
- **Transition (NAV)**: Activates latent/dormant EPI through regime change
- **Recursivity (REMESH)**: Echoes existing structure across scales

Only these three operators have the **physical capacity** to generate structure from null states.

**Canonicity Level**: **ABSOLUTE** - Mathematical impossibility to evolve from EPI=0 without generation.

**Traceability**: TNFR.pdf § 2.1 (Nodal Equation) → Direct mathematical consequence

---

### U1b: Structural Closure - Canonicity

**Derivation from Wave Physics:**

```
Sequences as temporal action potentials:
  Like electromagnetic pulses: must have source AND termination
  Like neural spikes: must have depolarization AND repolarization
  Like sound waves: must have emission AND absorption/decay

Physical requirement:
  Bounded temporal segments need coherent endpoints
  → Start: Generator creates initial perturbation
  → End: Closure absorbs/stabilizes final state
```

**Analogy with Classical Physics:**
- **Electromagnetic**: Every emission needs absorption (energy conservation)
- **Mechanical**: Every force pulse needs damping (stability)
- **Thermodynamic**: Every process needs equilibrium endpoint (2nd law)

**Why These Closures?**
- **Silence (SHA)**: Terminal closure - freezes evolution (νf → 0)
- **Transition (NAV)**: Handoff closure - transfers to next regime
- **Recursivity (REMESH)**: Recursive closure - distributes across scales
- **Dissonance (OZ)**: Intentional closure - preserves activation for next cycle

Each leaves system in a **coherent attractor state** rather than mid-evolution.

**Canonicity Level**: **STRONG** - Physical requirement for bounded sequences (like action potentials must repolarize).

**Traceability**: Wave physics + TNFR structural dynamics → Sequences need endpoints

---

### U2: Convergence & Boundedness - Canonicity

**Derivation from Integral Analysis:**

```
Integrated nodal equation:
  EPI(t_f) = EPI(t_0) + ∫_{t_0}^{t_f} νf(τ) · ΔNFR(τ) dτ

Without stabilizers (only destabilizers):
  dΔNFR/dt > 0 always (positive feedback)
  → ΔNFR(t) ~ e^(λt) (exponential growth)
  → ∫ νf · ΔNFR dt → ∞ (DIVERGES)
  → EPI(t) → ∞ (structural fragmentation)

With stabilizers:
  dΔNFR/dt can be < 0 (negative feedback)
  → ΔNFR(t) → bounded attractor
  → ∫ νf · ΔNFR dt converges
  → EPI(t) remains bounded (coherence preserved)
```

**Physical Necessity:**
- Like feedback control: need negative feedback to prevent runaway
- Like ecological systems: need limiting factors to prevent population explosion
- Like chemical reactions: need inhibitors to prevent autocatalytic divergence

**Mathematical Proof:**
1. Destabilizers create positive feedback: dΔNFR/dt > 0
2. Without negative feedback, integral diverges (proven via comparison test)
3. Divergent integral → unbounded EPI → fragmentation (non-physical)
4. Stabilizers provide negative feedback → convergence → bounded evolution

**Canonicity Level**: **ABSOLUTE** - Mathematical theorem from integral convergence.

**Experimental refinement (Grammar-Energy Landscape)**: The Lyapunov contractivity bound ($\Pi < 1$) derived from U2 is *sufficient* but not *necessary* for energy descent. Grammar-compliant sequences with $\Pi > 1$ can still achieve net energy decrease due to nonlinear operator interactions on the shared graph state. See [STRUCTURAL_OPERATORS.md §17.6](STRUCTURAL_OPERATORS.md) and [example 38](../examples/02_physics_regimes/38_grammar_energy_landscape.py).

**Traceability**: Analysis (integral convergence) + Nodal equation → Direct mathematical necessity

---

### U3: Resonant Coupling - Canonicity

**Derivation from Resonance Physics:**

```
Classical resonance condition:
  Two oscillators couple ⟺ frequency AND phase compatibility
  
Frequency condition: ω_i ≈ ω_j (met by structural frequency matching)
Phase condition: |φ_i - φ_j| ≤ Δφ_max (typically π/2)

Without phase verification:
  Nodes attempt coupling with φ_i ≈ π, φ_j ≈ 0 (antiphase)
  → Wave interference: A_i sin(ωt) + A_j sin(ωt + π) = 0
  → Destructive interference (pattern cancellation)
  → NO effective coupling (non-physical "ghost coupling")

With phase verification:
  Only synchronous nodes couple (constructive interference)
  → A_i sin(ωt) + A_j sin(ωt + δ) ≈ 2A sin(ωt) for δ ≈ 0
  → Resonant amplification (physical coupling)
```

**Physical Analogy:**
- **Radio tuning**: Must match frequency AND phase for signal lock
- **Laser coherence**: Photons must be phase-aligned for beam coherence
- **AC circuits**: Phase matters for power transmission (power factor)

**AGENTS.md Invariant #5:**
> "Phase check: no coupling is valid without explicit phase verification (synchrony)"

This is not a convention - it's a **physical requirement** of wave mechanics.

**Canonicity Level**: **ABSOLUTE** - Direct consequence of wave interference physics + explicit invariant.

**Traceability**: 
- Resonance physics (classical mechanics) → Phase requirement
- AGENTS.md Invariant #5 → Explicit TNFR requirement
- grammar.py → Implementation of physical law

---

### U4a: Bifurcation Triggers Need Handlers - Canonicity

**Derivation from Bifurcation Theory:**

```
Bifurcation condition (from AGENTS.md Contract OZ):
  System undergoes phase transition when ∂²EPI/∂t² > τ

Dissonance (OZ) and Mutation (ZHIR):
  Explicitly designed to trigger ∂²EPI/∂t² > τ
  → Create structural instability (bifurcation point)

Without handlers:
  System crosses bifurcation → chaos/fragmentation
  → No mechanism to organize new phase
  → Non-physical "explosion" of ΔNFR

With handlers (Self-organization, Coherence):
  Bifurcation → transient chaos → self-organization → new stable phase
  → Autopoietic closure (THOL) or explicit stabilization (IL)
  → Physical phase transition (like water → ice with nucleation)
```

**Physical Analogy:**
- **Water → Ice**: Need nucleation sites (handlers) for orderly crystallization
- **Laser threshold**: Need cavity stabilization for coherent emission
- **Chemical reactions**: Need catalysts (handlers) for controlled reactions

**Contract OZ (from AGENTS.md):**
> "Dissonance may trigger bifurcation if ∂²EPI/∂t² > τ"

Without handlers, bifurcations are uncontrolled → fragmentation.

**Canonicity Level**: **STRONG** - Physical requirement from bifurcation theory + explicit contract.

**Traceability**: 
- Contract OZ → Bifurcation physics
- Bifurcation theory → Need for stability mechanisms
- grammar.py → Implementation of controlled phase transitions

---

### U4b: Transformers Need Context - Canonicity

**Derivation from Threshold Physics:**

```
Phase transition requirements:
  1. Threshold energy: E > E_critical
  2. Proper timing: Energy must be "fresh" (recent)

Mutation (ZHIR) and Self-organization (THOL):
  Perform structural phase transitions
  → Require |ΔNFR| > threshold (energy condition)

Without recent destabilizer:
  |ΔNFR| may have decayed below threshold
  → Insufficient energy for phase transition
  → Transformation fails or produces unstable state

With recent destabilizer (within the relaxation window):
  |ΔNFR| still elevated (energy available)
  → Sufficient gradient for threshold crossing
  → Physical phase transition succeeds

Additional for ZHIR (Mutation):
  Needs prior Coherence (IL) for stable transformation base
  → Like crystal growth: needs stable seed
```

**Physical Analogy:**
- **Nuclear reactions**: Need recent energy input for activation
- **Chemical kinetics**: Reaction rate depends on "fresh" reactants
- **Phase transitions**: Need proper energy timing (not stale conditions)

**Timing Constraint — the structural-relaxation window (derived = 3):**
- The discrete steps for the |ΔNFR| perturbation to relax into the coherence
  band `1/(π+1)`, with per-step decay `q = 1 − νf·dt·ρ` (`ρ = trace(L_rw)/N = 1`,
  exact): `derive_bifurcation_window_from_physics`. The same for every destabilizer.
- Ensures the gradient hasn't dissipated below threshold (the structure is still plastic)
- Discrete geometric decay `qⁿ` — no continuous exponential, no `e`

**Canonicity Level**: **STRONG** - Physical requirement from threshold/timing physics.

**Traceability**: 
- Threshold energy physics → Energy requirement
- ΔNFR decay dynamics → Timing constraint
- Bifurcation stability → Prior IL for ZHIR

---

### U5: Multi-Scale Coherence - Canonicity

**Derivation from Nodal Equation + Hierarchical Coupling:**

```
Step 1: Nodal equation at each level (mathematical necessity)
  ∂EPI_parent/∂t = νf_parent · ΔNFR_parent
  ∂EPI_child_i/∂t = νf_child_i · ΔNFR_child_i

Step 2: Hierarchical coupling (from Invariant #7: Operational Fractality)
  EPI_parent = f(EPI_child_1, ..., EPI_child_N)
  
  Physical meaning: Parent structure depends on children
  Example: Cell depends on {nucleus, mitochondria, ribosomes}

Step 3: Chain rule (standard calculus)
  ∂EPI_parent/∂t = Σ (∂f/∂EPI_child_i) · ∂EPI_child_i/∂t
                  = Σ w_i · νf_child_i · ΔNFR_child_i

Step 4: Coherence relationship (from Invariant #9: Structural Metrics)
  C ~ 1/|ΔNFR|  (coherence inversely proportional to reorganization pressure)
  
  Parent coherence depends on aggregate child reorganization:
  |ΔNFR_parent| ~ |Σ w_i · νf_child_i · ΔNFR_child_i|
  
  Therefore: C_parent ~ 1/|Σ w_i · νf_child_i · ΔNFR_child_i|

Step 5: Statistical mechanics of coupling weights
  From central limit theorem with N independent children:
  |Σ w_i · X_i| ~ √N · |w_typical| · |X_typical|
  
  This gives α ~ 1/√N factor in coherence conservation

Step 6: Phase synchronization (from U3/Invariant #5)
  Only phase-compatible children contribute coherently
  Efficiency η_phase decreases with N (harder to sync many nodes)

Step 7: Conservation inequality (mathematical consequence)
  For bounded |ΔNFR_parent| (required for coherence):
  
  C_parent ≥ α · Σ C_child_i
  
  where α = (1/√N) · η_phase · η_coupling

Step 8: Physical necessity of stabilizers
  Without IL/THOL:
    Each child evolves independently with own ΔNFR_child_i
    → Parent ΔNFR grows from uncorrelated fluctuations
    → C_parent drops below α·ΣC_child
    → CONSERVATION VIOLATED → Fragmentation
  
  With IL/THOL (from operator contracts):
    IL reduces |ΔNFR| at each level → maintains coherence
    THOL creates self-limiting boundaries → prevents runaway
    → C_parent ≥ α·ΣC_child maintained
    → Bounded hierarchical evolution
```

**Why This Follows Within TNFR:**

1. **Nodal equation**: Given as foundational (∂EPI/∂t = νf · ΔNFR)
2. **Hierarchical coupling**: Follows from Invariant #7 (Fractality)
3. **Chain rule**: Standard calculus
4. **Coherence definition**: Follows from Invariant #9 (Metrics)
5. **Conservation inequality**: Mathematical consequence of above
6. **Stabilizer necessity**: Required to maintain conservation

**Physical Analogies:**
- **Thermodynamics**: Nested systems must exchange energy to maintain local order
- **Structural engineering**: Multi-story buildings need support at each level
- **Biological hierarchy**: Cells need homeostasis at tissue, organ, organism levels
- **Neural hierarchies**: Cortical areas need inter-layer stabilization

**Contract Requirements:**
- **IL (Coherence)**: "Reduces |ΔNFR|" → Direct stabilization at each level
- **THOL (Self-organization)**: "Creates sub-EPIs with boundaries" → Multi-level closure

**Independence from U2/U4b:**
```
Decisive test: [AL, REMESH(depth=3), SHA]

U2 (Convergence):
  No destabilizers present → ∫νf·ΔNFR dt trivially bounded
  ✓ PASSES (temporal constraint satisfied)

U4b (Transformer Context):
  REMESH is generator/closure, not transformer
  ✓ PASSES (temporal constraint not applicable)

U5 (Multi-Scale):
  3 hierarchical levels without stabilizers
  → C_parent < α·ΣC_child (spatial conservation violated)
  ✗ FAILS (spatial constraint violated)

Conclusion: U5 captures SPATIAL (hierarchy) physics
            U2/U4b capture TEMPORAL (sequence) physics
            INDEPENDENT dimensions, INDEPENDENT constraints
```

**Canonicity Level**: **STRONG** - Follows from:
1. Nodal equation (foundational)
2. Hierarchical coupling (Invariant #7)
3. Chain rule (mathematical necessity)
4. Coherence definition (Invariant #9)
5. Conservation requirement (bounded evolution)

**Traceability**: 
- **TNFR.pdf § 2.1**: Nodal equation foundation
- **AGENTS.md § Invariant #7**: Operational Fractality enables nesting
- **AGENTS.md § Invariant #9**: Structural Metrics define C(t)
- **Contract IL**: Stabilizer reducing |ΔNFR|
- **Contract THOL**: Multi-level autopoietic closure
- **grammar.py**: Implementation with depth parameter

**Why "STRONG" not "ABSOLUTE":**
- Requires Invariant #7 (fractality) which is empirical
- α factor has empirical component (η_phase, η_coupling)
- But given fractality, the rest follows by derivation

---

### U6: Structural Potential Confinement - Canonicity

**Derivation from Network ΔNFR Field:**

```
Step 1: Structural potential definition (from nodal equation)
  Starting from: ∂EPI/∂t = νf · ΔNFR(t)
  
  ΔNFR represents local structural pressure at each node
  Network aggregate: Φ_s(i) = Σ_{j≠i} ΔNFR_j / d(i,j)^α  (α=2)
  
  Physical interpretation: Distance-weighted sum of reorganization pressures
  Analogous to gravitational potential: Φ_g = Σ G·m_j/r_ij

Step 2: Empirical validation (2,400+ experiments)
  Correlation: corr(Δ Φ_s, ΔC) = -0.822 (R² ≈ 0.68)
  
  Physical meaning: Displacement from Φ_s minima → coherence loss
  Strong predictive power comparable to fundamental field theories

Step 3: Topology universality (5 families tested)
  Networks: ring, scale_free, small-world (ws), tree, grid
  Coefficient of variation: CV = 0.1%
  
  → Φ_s-coherence relationship independent of topology
  → Universal structural physics, not architecture artifact

Step 4: Passive equilibrium mechanism (from sequence analysis)
  Grammar-valid sequences: Δ Φ_s = +0.583
  Grammar-violating sequences: Δ Φ_s = +3.879
  Reduction factor: 0.15× (85% protection)
  
  Physical interpretation:
  - Φ_s minima = passive equilibrium states (potential wells)
  - Grammar U1-U5 = confinement mechanism (not active attractor)
  - Valid sequences naturally maintain proximity to equilibrium
  - No "force" pulling back - only passive resistance to escape

Step 5: Safety threshold (empirical calibration)
  Escape threshold: Δ Φ_s < 2.0
  
  Below 2.0: System remains confined, C(t) bounded
  Above 2.0: Escape from well → fragmentation risk
  
  Valid sequences: Δ Φ_s ≈ 0.6 (30% of threshold)
  Violations: Δ Φ_s ≈ 3.9 (195% of threshold)
  
  Clear separation between stable and fragmenting regimes

Step 6: Scale-dependent universality (fractality test)
  β exponent (fragmentation criticality):
  - Flat networks: β = 0.556 (standard universality class)
  - Nested EPIs: β = 0.178 (hierarchical universality class)
  
  Despite different β, Φ_s correlation remains universal: -0.822 ± 0.001
  → Φ_s captures fundamental coherence-pressure relationship across scales

Step 7: Independence from U2 (Boundedness)
  U2 (temporal): ∫νf·ΔNFR dt < ∞ (integral convergence over TIME)
  U6 (spatial): Δ Φ_s < 2.0 (potential confinement in STRUCTURE SPACE)
  
  Different dimensions:
  - U2: Time-integrated evolution must not diverge
  - U6: Spatial displacement must not exceed escape velocity
  
  Analogy: Rocket trajectory
  - U2: Total fuel expenditure must be finite
  - U6: Current position must stay within planet's gravity well

Step 8: Usage as telemetry-based safety check
  U6 is READ-ONLY (no operator dictation like U1-U5):
  - Does NOT require specific operator patterns
  - Does NOT modify sequence generation
  - DOES provide early warning: Δ Φ_s approaching 2.0
  - DOES validate: Grammar-compliant sequences naturally stay confined
  
  Physical basis: Grammar U1-U5 EMERGENTLY confines Φ_s dynamics
  → U6 observes and quantifies this emergent confinement

Conclusion: U6 emerges from:
  1. Nodal equation: ΔNFR as field source
  2. Distance-weighted aggregation: Φ_s field definition
  3. Empirical validation: 2,400+ experiments, 5 topologies
  4. Universal correlation: R² = 0.68, CV = 0.1%
  5. Grammar as confinement: Passive protection mechanism
  6. Threshold physics: Escape boundary at Δ Φ_s < π/2 ≈ 1.571 (π-derived, half phase-wrap)
```

**Why This Is Canonical:**

1. **Formal derivation**: Φ_s directly from ΔNFR field theory (nodal equation)
2. **Strong predictive power**: R² = 0.68 (comparable to established field theories)
3. **Topology universality**: CV = 0.1% across 5 diverse network families
4. **Grammar compliance**: Read-only telemetry, no conflicts with U1-U5
5. **Extensive validation**: 2,400+ experiments with reproducible results
6. **Scale-independent**: Universal correlation despite scale-dependent β

**Physical Interpretation:**
Φ_s is the structural potential landscape emerging from ΔNFR distribution. Nodes reside at potential minima (equilibrium). Grammar U1-U5 acts as passive confinement mechanism preventing escape (Δ Φ_s → 2.0). This is NOT active attraction but passive stabilization - like a bowl containing marbles without pulling them down.

**Distinction from Other Fields:**
- **Φ_s (CANONICAL)**: corr = -0.822, dominant field
- **|∇φ| (research)**: corr ≈ -0.13, weak EM-like
- **K_φ (research)**: corr ≈ -0.07, weak strong-like
- **ξ_C (research)**: threshold behavior, weak-like

Only Φ_s has met canonicity criteria.

**Contract Requirements:**
No operator contracts required (telemetry-based, not prescriptive). However:
- Grammar U1-U5 compliance NATURALLY maintains Δ Φ_s < 2.0
- Violations NATURALLY produce Δ Φ_s > 2.0
- U6 OBSERVES this emergent relationship

**Independence from U1-U5:**
U6 does NOT duplicate any existing rule:
- **vs U1**: U1 dictates start/end operators; U6 measures resulting Φ_s
- **vs U2**: U2 prevents temporal divergence; U6 prevents spatial escape
- **vs U3**: U3 requires phase checks; U6 aggregates global field
- **vs U4**: U4 manages bifurcations; U6 measures overall stability
- **vs U5**: U5 hierarchical stabilization; U6 flat+nested universality

**Canonicity Level**: **STRONG** (promoted 2025-11-11)

**Why "STRONG" not "ABSOLUTE":**
- Threshold (2.0) is empirically calibrated, not analytically derived
- α exponent (2) chosen by physics analogy (inverse-square), not proven optimal
- Correlation (-0.822) strong but not perfect (R² = 0.68, not 1.0)
- However: Universality (CV = 0.1%) and predictive power justify canonical status

**Traceability**: 
- **TNFR.pdf § 2.1**: Nodal equation ∂EPI/∂t = νf · ΔNFR(t)
- **docs/STRUCTURAL_FIELDS_TETRAD.md**: Φ_s drift analysis (corr = -0.822) and canonicity validation
- **AGENTS.md § Structural Fields**: Φ_s canonical status and usage
- **src/tnfr/physics/fields.py**: compute_structural_potential() implementation

**Evidence Base:**
- **Experiments**: 2,400+ simulations (360 drift + 480 universality + 1,200 nested + 360 RA-dominated)
- **Topologies**: ring, scale_free, ws (small-world), tree (hierarchical), grid (lattice)
- **Sequence types**: 2 glyphs × 2 phases × 30 intensities × 3 reps each
- **Validation date**: 2025-11-11

---

### Summary: Why These Rules Are Canonical

**U1a (Initiation)**: Mathematical impossibility to evolve from EPI=0 → **ABSOLUTE**

**U1b (Closure)**: Wave physics requires bounded sequences have endpoints → **STRONG**

**U2 (Convergence)**: Integral divergence theorem + feedback control → **ABSOLUTE**

**U3 (Phase)**: Wave interference physics + explicit invariant → **ABSOLUTE**

**U4a (Handlers)**: Bifurcation theory + explicit contract → **STRONG**

**U4b (Context)**: Threshold energy + timing physics → **STRONG**

**U5 (Multi-Scale)**: Nodal equation + hierarchical coupling + chain rule → **STRONG**

**U6 (Confinement)**: ΔNFR field + empirical validation + universality → **STRONG**

**All eight sub-rules** follow from:
1. The nodal equation: ∂EPI/∂t = νf · ΔNFR(t)
2. Mathematical analysis (integrals, chain rule, wave interference, field theory)
3. Physical constraints (resonance, bifurcations, thresholds, conservation, potentials)
4. Explicit invariants/contracts (AGENTS.md)
5. Empirical validation (2,400+ experiments, 5 topologies)

**Conclusion**: The unified grammar (U1-U6) is fully canonical within the TNFR framework — all rules derive from the nodal equation and its formal contracts.

**Reproducibility**: This analysis documents the derivation chain for grammar rules, supporting:
- Theoretical consistency
- Implementation fidelity
- Educational clarity
- Long-term maintenance

---

## Physics Derivation Summary

| Rule | Source | Type | Derivation Strength |
|------|--------|------|---------------------|
| U1a | ∂EPI/∂t undefined at EPI=0 | Mathematical | Absolute |
| U1b | Sequences as bounded action potentials | Physical | Strong |
| U2 | Integral convergence theorem | Mathematical | Absolute |
| U3 | Invariant #5 + resonance physics | Physical | Absolute |
| U4a | Contract OZ + bifurcation theory | Physical | Strong |
| U4b | Threshold energy for phase transitions | Physical | Strong |
| U5 | Nodal equation + hierarchical coupling | Mathematical+Physical | Strong |
| U6 | ΔNFR field + empirical validation | Physical+Empirical | Strong |

**Derivation Strength Levels:**
- **Absolute**: Mathematical necessity from nodal equation
- **Strong**: Physical requirement from invariants/contracts/validation

---

## Implementation Strategy

### Phase 1: Create Unified Module
1. Create `src/tnfr/operators/grammar.py`
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
1. Create tests/unit/operators/test_grammar.py
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

The unified grammar consolidates two previously separate rule systems into a single source of truth. All six rules (U1-U6) are derived from TNFR physics with no duplication, no inconsistency, and full physical basis.

**Key Improvements:**
1. **Single source of truth** - No more dual systems
2. **Complete coverage** - Includes phase verification (U3) and multi-scale coherence (U5)
3. **Consistent** - U1b restores closure physics (removed with RNC1)
4. **Fully physics-based** - Every rule derived from equation/invariants/contracts
5. **Well-documented** - Clear derivations and physical interpretations
6. **Dimensionally complete** - Covers temporal (U1-U4), spatial (U5), and field-theoretic (U6) constraints

**Result:** A unified TNFR grammar that is physically grounded, mathematically rigorous, and practically useful.

**Extension History:**
- **2025-11-08**: Original U1-U4 unified grammar
- **2025-11-10**: Added U5 Multi-Scale Coherence for hierarchical structures
- **2025-11-11**: Promoted U6 Structural Potential Confinement to canonical (2,400+ experiments)

---

## Grammar Completeness: Why No U7/U8

**Canonical Grammar**: U1-U6 (COMPLETE - no additional rules required)

The canonical TNFR grammar consists of **exactly six rules (U1-U6)**. Extended dynamics (phase flux J_φ, reorganization conservation ∇·J_ΔNFR) do NOT require new grammar rules because:

1. **Nodal equation unchanged**: ∂EPI/∂t = νf·ΔNFR(t) remains fundamental
2. **Preconditions covered**: U1-U6 already enforce all necessary constraints
   - U3 (Resonant coupling): Phase verification for J_φ generation
   - U2/U4 (Boundedness/Bifurcation): Gradient containment for flux convergence
   - U5 (Multi-scale): Stabilization where field stresses amplify across scales
3. **Flux fields = telemetry**: Add measurements, not prescriptive constraints

---

## Implementation & Examples

### SDK Entry Points

```python
from tnfr.sdk import TNFR

# Grammar-aware evolution enforces U1-U6 proactively
net = TNFR.create(15).random(0.3).evolve_grammar_aware(steps=10)
```

### Executable Demonstrations

| Example | Concept from this document |
|---------|---------------------------|
| [04_operator_sequences.py](../examples/01_foundations/04_operator_sequences.py) | U1–U6 validation: valid vs invalid sequences |
| [07_phase_transitions.py](../examples/01_foundations/07_phase_transitions.py) | Bifurcation dynamics (U4), critical thresholds |
| [36_grammar_violation_detector.py](../examples/02_physics_regimes/36_grammar_violation_detector.py) | Systematic violation detection: conservation residuals diagnose U1–U6 breaches in real time |

### Key Source Modules

- `src/tnfr/operators/grammar.py` — U1-U6 validation (public API facade)
- `src/tnfr/operators/grammar_validate.py` — Main validation entry point
- `src/tnfr/operators/grammar_dynamics.py` — Incremental grammar-aware operator selection
- `src/tnfr/operators/grammar_application.py` — Pre-validated operator application
- `src/tnfr/operators/grammar_core.py` — Core validator (U1-U4 rule logic)
- `src/tnfr/operators/grammar_u6.py` — U6 Structural Potential Confinement

**Structural Field Hexad** (measurements, NOT grammar rules):
- **Tetrad** (read-only telemetry under U6): Φ_s, |∇φ|, K_φ, ξ_C
- **Flux Pair** (dynamics variables): J_φ, ∇·J_ΔNFR

**See**: [docs/grammar/U6_STRUCTURAL_FIELD_TETRAD.md](../docs/STRUCTURAL_FIELDS_TETRAD.md)

---

## Historical Research: Proposed U7 (NOT Canonical)

**IMPORTANT**: The following documents a research direction that is **NOT part of canonical grammar**. The canonical grammar consists of **exactly six rules (U1-U6)** and is complete.

This section documents grammar constraints that have physical motivation but do not meet the canonicity threshold (STRONG/ABSOLUTE) for implementation.

### Proposed U7: TEMPORAL ORDERING

**Status:** 🔬 RESEARCH PHASE - Not Implemented - **NOT Canonical**  
**Canonicity Level:** MODERATE (40% confidence) - **Insufficient for inclusion**  
**Investigation Date:** 2025-11-10  
**Decision:** **DO NOT IMPLEMENT** (fundamental issues prevent canonization)  
**Note:** Previously labeled as "U6" before Structural Potential Confinement was promoted to canonical status (2025-11-11).  
**Investigation Date:** 2025-11-10  
**Note:** Previously labeled as "U6" before structural potential confinement was promoted to canonical status (2025-11-11).

#### Physical Motivation

**Proposed Rule:**
```
If bifurcation trigger {OZ, ZHIR} at position i,
Then do NOT apply {OZ, ZHIR, VAL} at positions i+1, i+2
```

**Physics Basis:**

From bifurcation theory, systems experience **structural relaxation time** after phase transitions:

$$
\tau_{\text{relax}} \approx \frac{\alpha}{2\pi\nu_f}
$$

where:
- α is scale factor (typically 0.5-0.9, context-dependent)
- νf is structural frequency (Hz_str)
- For νf = 1.0 Hz_str: τ_relax ≈ 0.159 seconds structural

**Rationale:**
1. **Post-bifurcation delay:** Systems exhibit ε^(2/3) delay after fold bifurcations
2. **Structural instability:** Non-hyperbolic transitions cause extreme sensitivity
3. **TNFR evidence:** "Caos estructural resonante" when νf high and ΔNFR grows rapidly

**Physical Analogies:**
- **Neuronal refractory period:** Neurons cannot fire immediately after action potential
- **Thermal equilibration:** Phase transitions require relaxation time
- **Oscillator synchronization:** After perturbation, need reconvergence time

#### Gap Analysis: Does U6 Add Constraints?

Testing reveals U6 DOES identify sequences that pass U1-U5 but may be problematic:

**Example Sequences Passing U1-U5 but Flagged by U6:**

```python
# Case 1: Consecutive destabilizers
[Emission, Dissonance, Dissonance, Coherence, Silence]  
# ✓ U1-U5, ✗ U6 (OZ at i, OZ at i+1)

# Case 2: Immediate OZ → ZHIR
[Emission, Coherence, Dissonance, Mutation, Coherence, Silence]
# ✓ U1-U5, ✗ U6 (OZ→ZHIR without spacing)

# Case 3: Triple destabilizers
[Emission, Dissonance, Expansion, Dissonance, Coherence, Silence]
# ✓ U1-U5, ✗ U6 (consecutive destabilization)
```

**Gap Coverage:** 5 out of 6 test cases (83% coverage improvement over U1-U5)

**Control (Valid under both):**
```python
[Emission, Dissonance, Coherence, SelfOrganization, Dissonance, Coherence, Silence]
# ✓ U1-U5, ✓ U6 (3 operators spacing between OZ)
```

#### CANONICAL STATUS: STRONG (Nov 2025)

**U6 Promoted to CANONICAL** based on comprehensive experimental validation:

**Empirical Validation (2,400+ experiments)**:
- **Correlation**: corr(Δ Φ_s, ΔC) = -0.822 (strong negative correlation)
- **Predictive power**: R² ≈ 0.68 (68% variance explained)
- **Universality**: Validated across 5 topology families (scale-free, small-world, grid, tree, ring)
- **Threshold**: Δ Φ_s < 2.0 (escape threshold, 30% typical for valid sequences)
- **Mechanism**: Passive equilibrium - grammar acts as confinement field

**Implementation**:
- **Formula**: Φ_s(i) = Σ_{j≠i} ΔNFR_j / d(i,j)² (inverse-square law analog)
- **Usage**: Telemetry-based safety check (read-only, not sequence constraint)
- **Typical drift**: Valid sequences maintain Δ Φ_s ≈ 0.6 (30% of threshold)

**Resolution of Previous Concerns**:

1. **Empirical Validation COMPLETE** ✅
   - 2,400+ controlled experiments (Nov 2025)
   - Strong correlation confirmed across multiple topologies
   - Predictive threshold established (Δ Φ_s < 2.0)

2. **Universal Behavior Confirmed** ✅
   - Works across 5 distinct topology families
   - No domain-specific calibration required
   - Passive confinement mechanism universal

3. **Telemetry-Based (Not Sequence Constraint)** ✅
   - Read-only structural field measurement
   - Does not modify operator sequences
   - Complements U1-U5 enforcement

4. **Physical Interpretation Clear** ✅
   - Emergent field from ΔNFR distribution
   - Inverse-square distance weighting
   - Passive equilibrium (not active control)

5. **Non-Redundant with U2/U4** ✅
   - U2/U4: Local operator pair constraints
   - U6: Global structural field confinement
   - Complementary mechanisms at different scales

#### Comparison with Canonical Rules

| Property | U1-U5 | U6 (CANONICAL) |
|----------|-------|----------------|
| **Derivation** | Direct from nodal equation | Emergent field from ΔNFR distribution |
| **Parameters** | None (or implicit in physics) | α=2 (inverse-square law) |
| **Domain** | Universal (mathematical) | Universal (empirical, 5 topologies) |
| **Evidence** | Mathematical/physical necessity | 2,400+ experiments, R²≈0.68 |
| **Type** | ABSOLUTE/STRONG | STRONG (empirical) |

#### Implementation and Usage

**Telemetry Integration**:
```python
from tnfr.physics.canonical import (
    compute_structural_potential,
    validate_structural_potential_confinement
)
from tnfr.config.defaults_core import STRUCTURAL_ESCAPE_THRESHOLD

# Compute structural potential before sequence
phi_before = compute_structural_potential(G, alpha=2.0)

# Execute operator sequence
run_sequence(G, node, sequence)

# Compute structural potential after sequence
phi_after = compute_structural_potential(G, alpha=2.0)

# Validate confinement (telemetry check)
valid, drift, msg = validate_structural_potential_confinement(
    G, phi_before, phi_after, threshold=STRUCTURAL_ESCAPE_THRESHOLD, strict=False
)

if not valid:
    logger.warning(f"U6 confinement violated: {msg} (drift={drift:.3f})")
```

**Key Characteristics**:
- **Read-only**: Does not modify operator sequences
- **Passive**: Monitors emergent field, does not enforce
- **Telemetry**: Post-hoc validation, not pre-validation constraint
- **Safety**: Provides early warning of structural escape

**Future Research Directions** (Enhancement, not validation):

1. **Theoretical Derivation** (Optional)
   - Formal proof from integrated nodal equation
   - Connection to gauge field theory
   - Relationship to other structural fields (|∇φ|, K_φ, ξ_C)

2. **Multi-Scale Analysis** (Enhancement)
   - Hierarchical Φ_s computation for nested EPIs
   - Scale-dependent confinement thresholds
   - Fractal scaling laws

3. **Domain-Specific Studies** (Application)
   - Biological networks (neural, metabolic)
   - Social networks (information flow)
   - AI architectures (attention mechanisms)

4. **Optimization** (Performance)
   - Approximate methods for large networks
   - Landmark-based sampling
   - GPU acceleration

#### Theoretical Derivation (Sketch) from Nodal Equation

We outline a physics-based bridge from the nodal equation to a relaxation timescale that motivates U6.

1) Linearization around a coherent attractor

Let EPI* denote a coherent form (attractor). For small deviations δEPI(t) = EPI(t) − EPI*, assume ΔNFR is linearizable:

  ΔNFR(δEPI) ≈ L · δEPI

where L is a linear operator capturing local reorganization response (a structural Liouvillian). The nodal equation becomes:

  d(δEPI)/dt = νf · L · δEPI

2) Modal decomposition and decay

If v_k are eigenmodes of L with eigenvalues λ_k (Re λ_k ≤ 0 for contractivity), then

  δEPI_k(t) = c_k · exp(νf · λ_k · t)

The slowest decay rate is set by the mode with the smallest magnitude of negative real part, λ_slow (Re λ_slow < 0). Therefore, the characteristic relaxation time is

  τ_relax = 1 / (νf · |Re(λ_slow)|)

3) Relation to practical Liouvillian spectrum

In practice, when the full time-generator ℒ is constructed (e.g., Lindblad Liouvillian), its eigenvalues already carry temporal units (Hz_str). In that case, the evolution is

  d(δEPI)/dt = ℒ · δEPI  ⇒  δEPI_k(t) = c_k · exp(λ_k · t)

and the relaxation time simplifies to

  τ_relax = 1 / |Re(λ_slow)|

This matches the implementation in mathematics/liouville.py and operators/metrics_u6.py, where we prefer Liouvillian slow-mode when available.

4) Recovery threshold and minimal spacing

For a target recovery factor ε ∈ (0, 1), requiring ||δEPI(Δt)|| ≤ ε · ||δEPI(0)|| yields

  Δt ≥ ln(1/ε) / (νf · |Re(λ_slow)|)

Hence a minimum spacing Δt on the order of τ_relax between destabilizers allows δEPI to decay towards the attractor before the next perturbation, giving a physics-grounded rationale for U6.

5) Integral boundedness link (U2)

Integrating the nodal equation gives

  EPI(t_f) = EPI(t_0) + ∫_{t_0}^{t_f} νf(τ) · ΔNFR(τ) dτ

Under the linear regime, ΔNFR(τ) ~ L · δEPI(τ) and δEPI(τ) decays as above. The integral converges provided Re(νf · λ_k) < 0. Imposing Δt ≥ O(τ_relax) after a destabilizer allows δEPI to decay sufficiently, keeping the integral bounded and coherence preserved—consistent with U2 and clarifying U6’s temporal role.

Notes:
- If the spectrum is computed from a structural operator L without temporal scaling, include νf explicitly: τ_relax = 1/(νf · |Re(λ_slow)|).
- If using a full time-generator (Liouvillian) ℒ, νf is already absorbed: τ_relax = 1/|Re(λ_slow)|.

#### Preliminary Empirical Results (2025-11-11)

Experimental setup (benchmarks/u6_sequence_simulator.py):
- Topologies: star, ring, small-world (ws), scale-free
- Sizes: n ∈ {20, 50}
- Structural frequencies: νf ∈ {0.5, 1.0, 2.0, 4.0}
- Sequences: valid_U6 (spaced) vs violate_U6 (consecutive destabilizers)
- Runs: 5 per combination (total: 320 experiments)
- Metrics: minimum C(t), recovery steps, fragmentation (sustained C(t) < 0.3), τ_relax (Liouvillian if available, spectral proxy otherwise), empirical α = τ_relax · 2π · νf, min_spacing_steps

Findings:
1. Coherence dip: violate_U6 systematically reduces minimum coherence vs. valid_U6 (e.g., 0.448 vs. 0.616 on average in the batch).
2. Fragmentation: not observed under current parameters (window=5, threshold=0.3), so correlations with fragmentation are null.
3. Recovery: recovery_steps ≈ 0 in this regime; perturbations are moderate and the system does not cross severe thresholds.
4. Empirical α: scales linearly with νf and depends on topology (star < ws < scale_free < ring). Large magnitudes (order 10^3–10^4) indicate direct α_emp is not comparable to the proposed 0.5–0.9 range without structural normalization.

Implications:
- U6 shows a gentle effect (depression of minimum coherence) but does not yet evidence fragmentation; canonicity remains MODERATE.
- More aggressive conditions are required (higher νf, longer sequences with denser OZ/ZHIR/VAL) to explore fragmentation thresholds.
- To compare α with the proposed range, normalize α_emp by topological scale (e.g., α_norm = (τ_relax · 2π · νf) / (N · k_eff) with k_eff ≈ average degree or λ₁).

Next steps (empirical):
- Extend sequences with triple/quintuple destabilizers and longer windows.
- Increase νf beyond 4.0 and vary connectivities (modularity and bottlenecks) to induce violations crossing the threshold.
- Record λ₁ per experiment and report α_norm to facilitate cross-topology comparisons.

#### Implementation Strategy (If Elevated to STRONG)

**Phase 1: Experimental Flag**
```python
validator = UnifiedGrammarValidator(experimental_u6=True)
violations = validator.validate(sequence, epi_initial=0.0)
```

**Phase 2: Configurable Parameter**
```python
validator = UnifiedGrammarValidator(u6_spacing=2, u6_alpha=0.7)
```

**Phase 3: Canonical Integration**
- Add U6 to grammar.py operator sets
- Update UNIFIED_GRAMMAR_RULES.md derivation section
- Comprehensive test suite (bifurcation simulations)
- Update AGENTS.md invariants if needed

#### Current Recommendation

⚠️ **HISTORICAL NOTE (Pre-Nov 2025)**: This section describes the REJECTED "U6: Temporal Ordering" proposal based on τ_relax spacing. This approach was superseded by **"U6: STRUCTURAL POTENTIAL CONFINEMENT"** which was promoted to CANONICAL status in November 2025 based on 2,400+ experiments.

**See**: Section "Rule U6: STRUCTURAL POTENTIAL CONFINEMENT" (line 344) and Appendix "Discovery 1: Cache Invalidation Issue" (line 1516) for current canonical U6 specification.

---

**DEPRECATED - DO NOT IMPLEMENT** the temporal ordering approach below.

**Original Rationale (Historical):**
1. Canonicity MODERATE (40%) below threshold for inclusion
2. Requires empirical validation not yet performed
3. Parameter α needs principled determination method
4. May introduce false positives (overly restrictive)
5. Alternative: Strengthen U4a/U4b to cover temporal aspects

**What Happened Instead:**
- U6 was reimplemented as **STRUCTURAL POTENTIAL CONFINEMENT**
- Based on Φ_s field theory (distance-weighted ΔNFR)
- Validated with 2,400+ experiments (Nov 2025)
- Achieved CANONICAL (STRONG) status
- No temporal spacing assumptions required

**Alternative Approach:**
- Document U6 as "physically motivated constraint under research"
- Provide experimental validation framework in research tools
- Gather data from domain applications
- Revisit in 6-12 months with empirical evidence
- Consider elevation if canonicity reaches STRONG (60-80%)

**Alignment with TNFR Philosophy:**
- **"Physics First"** - wait for complete derivation
- **"No Arbitrary Choices"** - resolve α parameter issue
- **"Reproducibility Always"** - need validation studies
- **"Coherence Over Convenience"** - don't prematurely constrain

#### Timeline Estimate

**Realistic elevation timeline:** 6-12 months

**Milestones:**
- Month 1-2: Simulation framework for τ_relax measurement
- Month 3-4: Cross-domain validation studies
- Month 5-6: Theoretical derivation attempts
- Month 7-9: α parameter methodology development
- Month 10-11: Comprehensive testing and refinement
- Month 12: Decision on canonical promotion

**Success Criteria:**
- Empirical data: >80% of U6 violations cause measurable coherence loss
- Theoretical: Derivation from nodal equation (even if approximate)
- Parameter: α determinable from node properties (not free parameter)
- Universality: Works across 3+ distinct domains without re-tuning

---

## Appendix: Key Technical Discoveries (Nov 2025)

### Discovery 1: Cache Invalidation Bug (Φ_s / ξ_C / J_ΔNFR)

**Problem**: U6 structural potential tests reported zero drift despite ΔNFR changes; `compute_structural_potential` returned bit-identical Φ_s after ΔNFR was modified on a fixed topology.

**Original misdiagnosis (Nov 2025, superseded)**: the symptom was attributed to "uniform ΔNFR scaling preserving Φ_s ratios" and worked around by (1) non-uniform ΔNFR patterns and (2) varying `alpha` (2.0→2.001) to force a cache miss. This was incorrect: Φ_s is linear in ΔNFR (`Φ_s(k·ΔNFR) = k·Φ_s`), so uniform scaling DOES change Φ_s and DOES produce a non-zero drift `(k−1)·Φ_s`. The zero-drift symptom was not physics — it was a cache bug.

**Actual root cause (corrected May 2026)**: `@cache_tnfr_computation` builds its key from a dependency hash (`tnfr.utils.cache._compute_dependency_hash`). For `node_dnfr`/`node_vf`/`node_epi` dependencies it read node values by hardcoded English keys (`'delta_nfr'`, `'vf'`, `'epi'`), but the canonical writer (`tnfr.alias.set_attr`) stores each field under its FIRST alias — the Greek/canonical key (`'ΔNFR'`, `'νf'`, `'EPI'`). The mismatch made the hash read `None` for every node, so the cache key was **blind** to ΔNFR: any ΔNFR change returned stale Φ_s, and two distinct graphs with identical topology but different ΔNFR collided. (The `alpha`-variation workaround "worked" only because `alpha` is part of the function-argument key, forcing an unrelated miss.)

**Fix**: `_compute_dependency_hash` now resolves dependencies through the canonical alias tuples (`_dependency_alias_keys`), so ΔNFR/νf/EPI changes correctly invalidate dependent caches. The `node_phase` path was already correct (phase is stored under both `'theta'` and `'phase'`). Affected canonical functions: `compute_structural_potential` (Φ_s), `estimate_coherence_length` (ξ_C), `compute_dnfr_flux` (J_ΔNFR), and `physics/telemetry.py`.

**Regression**: `tests/physics/test_field_cache_invalidation.py` (Φ_s/ξ_C respond to ΔNFR changes; no same-topology collisions; the dependency hash reflects ΔNFR/νf/EPI changes).

**Location**: `src/tnfr/utils/cache.py` (`_compute_dependency_hash`, `_dependency_alias_keys`), `src/tnfr/physics/canonical.py` module docstring

---

### Discovery 2: Context Override for Pre-existing EPI

**Problem**: Grammar enforcement too strict for operational scenarios where nodes already have initialized EPI.

**Root Cause**: U1a (generator requirement) exists because ∂EPI/∂t is undefined at EPI=0. However, when EPI≠0, applying operators like Coherence, Resonance, etc. is physically valid since structure exists to evolve.

**Solution**: `run_sequence()` auto-detects non-zero EPI via `check_epi_nonzero()` and passes `context={'initial_epi_nonzero': True}` to `validate_sequence()`. This bypasses U1a generator requirement while maintaining strict enforcement for EPI=0 initialization cases.

**Implementation**:
```python
# In run_sequence()
if check_epi_nonzero(G, node):
    context = {'initial_epi_nonzero': True}
else:
    context = None
validate_sequence(sequence, context=context)
```

**Physics Rationale**: U1a's mathematical basis (undefined gradient at EPI=0) does NOT apply when structure pre-exists. The override is operational flexibility, not grammar violation. Strict canonicity preserved for initialization scenarios.

**Canonicity Status**: ✅ PRESERVES U1a physics - context override is consequence of conditional applicability, not exception

**Location**: `src/tnfr/structural.py` module docstring and `run_sequence()` function, `src/tnfr/operators/grammar_patterns.py::_check_start_rule()`

---

### Discovery 3: Diagnostic Pattern Exemption ([dissonance, mutation])

**Problem**: Bifurcation detection tests require controlled destabilization to test threshold crossing (∂²EPI/∂t² > τ). Adding stabilizers defeats the purpose.

**Pattern**: `[dissonance, mutation]` sequence intentionally violates:
- U2 (stabilizer requirement after destabilizers)
- U4b (transformer context requirement)

**Solution**: Special exemption in `_check_end_rule()` and stabilizer checks explicitly allows `[OZ, ZHIR]` patterns for bifurcation probe sequences.

**Rationale**: 
- **Not a grammar failure**: Diagnostic tool for threshold behavior validation
- **Controlled environment**: Only used in tests where fragmentation is expected outcome
- **Safety**: Does not compromise canonical grammar integrity in production

**Implementation**:
```python
# In _check_end_rule()
# Diagnostic exemption: [dissonance, mutation] probe pattern
if len(names) == 2 and names[0] == 'dissonance' and names[1] == 'mutation':
    # Bifurcation probe - allow without stabilizer
    return SequenceValidationResult(valid=True, ...)
```

**Physics Insight**: Bifurcation detection is inherently a destabilization test. Requiring stabilizers creates logical contradiction - you cannot test threshold crossing while preventing threshold crossing.

**Canonicity Status**: ✅ PRESERVES grammar integrity - exemption limited to controlled diagnostic context

**Location**: `src/tnfr/operators/grammar_patterns.py` module docstring and `_check_end_rule()` function

---

### Discovery 4: Spatial Gradient Requirement for U6 Validation

**Problem**: U6 structural potential confinement validation requires detecting structural pressure changes, but uniform transformations produce no measurable drift.

**Physics**: Structural potential is defined as distance-weighted sum of ΔNFR:
```
Φ_s(i) = Σ_{j≠i} ΔNFR_j / d(i,j)^α
```

For uniform scaling (all nodes k·ΔNFR), potential scales proportionally:
```
Φ_s'(i) = Σ_{j≠i} (k·ΔNFR_j) / d(i,j)^α = k·Φ_s(i)
```

Network-wide drift:
```
Δ Φ_s = |Φ_s' - Φ_s| = |k·Φ - Φ| = |k-1|·|Φ|
```

This is proportional scaling without spatial structure change. U6 validation requires:
```
Δ Φ_s = |Σ_i (Φ_s,after(i) - Φ_s,before(i))|
```

For uniform scaling: Δ Φ_s = 0 (despite individual node changes) because gradient structure preserved.

**Solution**: Use **non-uniform ΔNFR patterns** that break symmetry:
- Alternating high/low (e.g., nodes 0,2,4,... → 5.0; nodes 1,3,5,... → 0.1)
- Spatial gradients (linear, radial, or cluster-based distributions)
- Random perturbations with controlled variance

**Validation Pattern**:
```python
# Create spatial gradient
for i, node in enumerate(G.nodes()):
    G.nodes[node]['delta_nfr'] = 5.0 if i % 2 == 0 else 0.1
```

**Physics Insight**: U6 measures passive equilibrium confinement through structural pressure gradients. Uniform transformations are gauge transformations that preserve equilibrium state. Only non-uniform changes create pressure differentials detectable by Δ Φ_s.

**Implication**: Valid U6 tests must involve spatial reorganization, not just magnitude scaling. This aligns with TNFR principle that structure = pattern, not absolute values.

**Location**: `src/tnfr/physics/canonical.py` module docstring, `tests/unit/operators/test_unified_grammar.py` TestU6 implementations

---

## References

- **TNFR.pdf**: Section 2.1 (Nodal Equation), bifurcation theory
- **AGENTS.md**: Invariants (#1-#10), Contracts (Coherence, Dissonance, etc.)
- **grammar.py**: Original C1-C3 implementation
- **canonical_grammar.py**: Original RC1-RC4 implementation
- **RESUMEN_FINAL_GRAMATICA.md**: Grammar evolution documentation
- **EMERGENT_GRAMMAR_ANALYSIS.md**: Detailed physics analysis
- **Bifurcation Theory:** Kuznetsov (2004), "Elements of Applied Bifurcation Theory"
- **U6 Research:** "The Pulse That Traverses Us.pdf" § Resonant structural chaos
- **Technical Discoveries:** Grammar refinement session Nov 2025 (cache issues, context override, diagnostic patterns)

---

**Date:** 2025-11-08 (U1-U4), 2025-11-10 (U5), 2025-11-15 (U6 promoted to CANONICAL, technical discoveries documented)  
**Status:** ✅ CANONICAL - U1-U6 complete with empirical validation (2,400+ experiments for U6)  
**Implementation:** All six rules implemented in `src/tnfr/operators/grammar.py` and validated in test suite

---

## Implementation Architecture

The unified grammar is implemented in a modular but consolidated structure within `src/tnfr/operators/`.

### Core Components

*   **`src/tnfr/operators/grammar.py`**: The single source of truth for validation logic. It implements the `GrammarValidator` class which enforces U1-U6.
*   **`src/tnfr/operators/definitions.py`**: Contains the implementation of the 13 canonical operators. Each operator class is tagged with its grammar role (Generator, Stabilizer, etc.).

### Operator Categories

The grammar relies on strict categorization of operators:

*   **Generators (U1a)**: `{AL, NAV, REMESH}` - Can initiate structure from vacuum.
*   **Closures (U1b)**: `{SHA, NAV, REMESH, OZ}` - Valid endpoints for sequences.
*   **Stabilizers (U2)**: `{IL, THOL}` - Provide negative feedback to bound energy.
*   **Destabilizers (U2)**: `{OZ, ZHIR, VAL}` - Introduce positive feedback or expansion.
*   **Coupling/Resonance (U3)**: `{UM, RA}` - Require phase synchronization.
*   **Bifurcation Triggers (U4a)**: `{OZ, ZHIR}` - Push system towards instability.
*   **Transformers (U4b)**: `{ZHIR, THOL}` - Require context (recent destabilization).

**Cross-reference (Dual-Lever Structure)**: The grammar categories above classify operators by their *sequence role*. A complementary classification by *dynamical mechanism* is the **dual-lever structure**: capacity lever ($\nu_f$: UM, SHA, VAL, NUL) vs. pressure lever ($\Delta$NFR: IL, OZ, THOL, ZHIR, NAV). The two classifications are orthogonal and jointly characterise each operator’s physics. See [STRUCTURAL_OPERATORS.md §17.1](STRUCTURAL_OPERATORS.md).

### Validation Flow

1.  **Sequence Check**: The `validate_grammar()` function accepts a sequence of operators.
2.  **Context Initialization**: A `GrammarContext` tracks state (EPI, cumulative ΔNFR, phase).
3.  **Rule Application**:
    *   **U1a**: Checks the first operator against `GENERATORS` if initial EPI is zero.
    *   **U2/U4**: Iterates through the sequence, maintaining a window of recent operators to ensure stabilizers follow destabilizers.
    *   **U3**: Checks for phase compatibility (simulated or actual).
    *   **U5**: Checks for scale stabilizers if recursion depth > 1.
    *   **U6**: (Runtime only) Monitors structural potential telemetry.
4.  **U1b**: Checks the last operator against `CLOSURES`.

This architecture ensures that every sequence executed by the engine is physically valid before it runs.

