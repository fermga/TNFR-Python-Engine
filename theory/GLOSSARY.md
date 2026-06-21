# TNFR Glossary

**Purpose**: Operational quick reference for the Resonant Fractal Nature Theory (TNFR) v0.0.3  
**Status**: Complete reference for current implementation  
**Version**: March 2026  
**Authority**: Aligned with [AGENTS.md](../AGENTS.md) as single source of truth  

**Scope**: This glossary provides **API-focused definitions** for developers implementing TNFR networks. For complete theoretical foundations, see [AGENTS.md](../AGENTS.md) and [UNIFIED_GRAMMAR_RULES.md](UNIFIED_GRAMMAR_RULES.md).

---

## Core Variables

### Primary Information Structure (EPI)

**Code:** `G.nodes[n]['EPI']`, `ALIAS_EPI`  
**Symbol:** \(\text{EPI}\) or \(E\)  
**What:** Coherent structural form of a node  
**Space:** \(B_{\text{EPI}}\) (Banach space)  
**Rules:** Modified only via structural operators, never directly  
**API:** `tnfr.structural` operators  
**Math:** [FUNDAMENTAL_THEORY.md §2.2 (Structural Triad — Banach space B_EPI)](FUNDAMENTAL_THEORY.md)

### Structural Frequency (νf)

**Code:** `G.nodes[n]['vf']`, `ALIAS_VF`  
**Symbol:** \(\nu_f\)  
**Units:** Hz_str (structural hertz)  
**Range:** \(\mathbb{R}^+\) (positive reals; node collapse when \(\nu_f \to 0\))  
**What:** Rate of structural reorganization  
**API:** `adapt_vf_by_coherence()`, operators  
**Math:** [FUNDAMENTAL_THEORY.md §2 (Governing Dynamics)](FUNDAMENTAL_THEORY.md)

### Internal Reorganization Operator (ΔNFR)

**Code:** `G.nodes[n]['dnfr']`, `ALIAS_DNFR`  
**Symbol:** \(\Delta\text{NFR}\)  
**What:** Structural evolution gradient (drives reorganization)  
**Sign:** Positive = expansion, Negative = contraction  
**Compute:** Via `default_compute_delta_nfr` hook, automatic in `step()`  
**Math:** [FUNDAMENTAL_THEORY.md §2.1 (Nodal Equation)](FUNDAMENTAL_THEORY.md)

### Phase (φ, θ)

**Code:** `G.nodes[n]['theta']`, `collect_theta_attr()`  
**Symbol:** \(\theta\) or \(\phi\)  
**Range:** \([0, 2\pi)\) or \([-\pi, \pi)\) radians  
**What:** Network synchrony parameter (relative timing)  
**Phase difference:** \(\Delta\theta = \theta_i - \theta_j\)  
**API:** Phase adaptation in dynamics  
**Math:** [FUNDAMENTAL_THEORY.md §2.2 (Structural Triad — phase)](FUNDAMENTAL_THEORY.md)

### Total Coherence (C(t))

**Code:** `compute_coherence(G)` → float ∈ [0,1]  
**Symbol:** \(C(t)\)  
**Formula:** \(C(t) = 1/(1 + \overline{|\Delta\text{NFR}|} + \overline{|d\text{EPI}|})\) (canonical; derived from the nodal equation — equilibrium \(\Delta\text{NFR}\to 0 \wedge d\text{EPI}\to 0 \Rightarrow C\to 1\))  
**Range:** \([0, 1]\) where 1 = perfect coherence, 0 = total fragmentation  
**What:** Global network stability measure (primary canonical metric, recorded in `history['C_steps']`)  
**Math:** [FUNDAMENTAL_THEORY.md §5.1 (Total Coherence)](FUNDAMENTAL_THEORY.md)

### Coherence Operator (Ĉ)

**Code:** `coherence_matrix(G)` → (nodes, W)  
**Symbol:** \(\hat{C}\)  
**Matrix element:** \(w_{ij} \approx \langle i | \hat{C} | j \rangle\)  
**Properties:** Hermitian (\(\hat{C}^\dagger = \hat{C}\)), positive semi-definite  
**What:** Operator measuring structural stability between nodes  
**Math:** [src/tnfr/metrics/coherence.py](../src/tnfr/metrics/coherence.py) (`coherence_matrix`)

### Sense Index (Si)

**Code:** `G.nodes[n]['Si']`, `ALIAS_SI`, `compute_Si_node()`  
**Symbol:** \(\text{Si}\) (global) or \(S_i\) (node i)  
**Formula:** \(\text{Si} = \alpha \cdot \nu_{f,\text{norm}} + \beta \cdot (1 - \text{disp}_\theta) + \gamma \cdot (1 - |\Delta\text{NFR}|_{\text{norm}})\)  
**Range:** \([0, 1^+]\) typically, higher = more stable reorganization  
**What:** Capacity for stable structural reorganization  
**Weights:** canonical defaults \(\alpha = \varphi/(\varphi+\gamma) \approx 0.737\), \(\beta = \gamma/(\pi+\gamma) \approx 0.155\), \(\gamma_w = \gamma/(\varphi\pi) \approx 0.114\) (`SI_WEIGHTS` in `config/defaults_core.py`; sum \(\approx 1\))  
**Math:** [Mathematical Foundations - Metrics](MATHEMATICAL_DYNAMICS_BASIS.md)

### Phase Gradient (|∇φ|) - CANONICAL

**Code:** `compute_phase_gradient(G)` → Dict[NodeId, float]  
**Symbol:** \(|\nabla\phi|(i)\)  
**Formula:** \(|\nabla\phi|(i) = \text{mean}_{j \in N(i)} |\theta_i - \theta_j|\) (circular mean)  
**What:** Local phase desynchronization / stress proxy field  
**Status:** **CANONICAL** (Nov 2025)  
**Physics:** Captures dynamics C(t) misses due to scaling invariance  
**Threshold:** Kinematic bound |∇φ| ≤ π (phase wrap — same as K_φ); γ/π ≈ 0.1837 is only a heuristic early-warning level, NOT a derived bound (measured sync-onset ≈ 0.29, σ-dependent)  
**API:** `tnfr.physics.fields.compute_phase_gradient()`  
**Usage:** Stress detection, local instability prediction  
**Documentation:** [docs/STRUCTURAL_FIELDS_TETRAD.md](../docs/STRUCTURAL_FIELDS_TETRAD.md)

### Phase Curvature (K_φ) - CANONICAL

**Code:** `compute_phase_curvature(G)` → Dict[NodeId, float]  
**Symbol:** \(K_\phi(i)\)  
**Formula:** \(K_\phi = \text{wrap\_angle}(\phi_i - \text{circular\_mean}(\text{neighbors}))\)  
**What:** Phase torsion and geometric confinement field  
**Status:** **CANONICAL** (Nov 2025)  
**Physics:** Flags mutation-prone loci via geometric constraints  
**Threshold:** Classical bound |K_φ| < 2.8274 (90% of π theoretical maximum)  
**API:** `tnfr.physics.fields.compute_phase_curvature()`  
**Usage:** Geometric confinement monitoring, bifurcation prediction  
**Documentation:** [docs/STRUCTURAL_FIELDS_TETRAD.md](../docs/STRUCTURAL_FIELDS_TETRAD.md)

### Coherence Length (ξ_C) - CANONICAL

**Code:** `estimate_coherence_length(G)` → float  
**Symbol:** \(\xi_C\)  
**Formula:** Spatial correlation function \(C(r) = A \exp(-r/\xi_C)\)  
**What:** Spatial correlation scale of local coherence  
**Status:** **CANONICAL** (Nov 2025)  
**Physics:** Critical phenomena and finite-size scaling analysis  
**Thresholds:**  
- Critical: ξ_C > 1.0 × diameter (finite-size scaling dominates)  
- Watch: ξ_C > π ≈ 3.14 × mean_distance (RG scaling)  
- Stable: ξ_C < mean_distance (bulk behavior)  
**API:** `tnfr.physics.fields.estimate_coherence_length()`  
**Usage:** Critical point detection, correlation analysis  
**Documentation:** [docs/STRUCTURAL_FIELDS_TETRAD.md](../docs/STRUCTURAL_FIELDS_TETRAD.md)

---

### Structural Potential (Φ_s) - CANONICAL

**Code:** `compute_structural_potential(G, alpha=2.0)` → Dict[NodeId, float]  
**Symbol:** \(\Phi_s(i)\)  
**Formula:** \(\Phi_s(i) = \sum_{j \neq i} \frac{\Delta\text{NFR}_j}{d(i,j)^\alpha}\) where \(\alpha = 2\)  
**What:** Global structural potential field from ΔNFR distribution  
**Status:** **CANONICAL** (Nov 2025)  
**Validation:** 2,041 tests across 5 topologies  
**Physics:** Passive equilibrium confinement landscape  
**Grammar:** U6 STRUCTURAL POTENTIAL CONFINEMENT (Δ Φ_s < φ ≈ 1.618 canonical confinement; ceiling 2.0 binary escape)  
**API:** `tnfr.physics.fields.compute_structural_potential()`  
**Threshold:** Per-node bound |Φ_s| < 0.7711 (empirically validated; no closed-form derivation — the "von Koch / Γ(4/3)/Γ(1/3)" derivation is incorrect, that ratio = 1/3)  
**Documentation:** [docs/STRUCTURAL_FIELDS_TETRAD.md](../docs/STRUCTURAL_FIELDS_TETRAD.md)
- [src/tnfr/physics/fields.py](../src/tnfr/physics/fields.py) - Implementation

**Interpretation:**
- Φ_s minima = passive equilibrium states
- Δ Φ_s < φ ≈ 1.618 = canonical confinement (safe regime)
- Δ Φ_s ≥ 2.0 = binary escape threshold (fragmentation risk)
- Valid sequences: Δ Φ_s ≈ 0.6 (37% of φ threshold)
- Violations: Δ Φ_s ≈ 3.9 (240% of φ threshold)

**Mechanism:** Grammar U1-U5 acts as passive confinement (NOT active attractor). Reduces escape drift by 85%.

---

## The Nodal Equation

**The fundamental equation of TNFR** governs structural evolution:

\[
\frac{\partial \text{EPI}}{\partial t} = \nu_f \cdot \Delta\text{NFR}(t)
\]

**Where:**
- \(\frac{\partial \text{EPI}}{\partial t}\): Rate of change of structure
- \(\nu_f\): Structural frequency (reorganization rate) in Hz_str
- \(\Delta\text{NFR}(t)\): Reorganization gradient (driving pressure)

**Interpretation:**
- Structure changes **only when** both \(\nu_f > 0\) (capacity) and \(\Delta\text{NFR} \neq 0\) (pressure) exist
- Rate of change is **proportional** to both frequency and gradient
- When \(\nu_f \to 0\), evolution freezes (node collapse)
- When \(\Delta\text{NFR} = 0\), structure reaches equilibrium

**Implementation:** See `src/tnfr/dynamics/` for numerical integration  
**Theory:** [Nodal equation](FUNDAMENTAL_THEORY.md) §2

---

## Structural Operators

The 13 canonical operators are the **only way** to modify nodes in TNFR. They're not arbitrary functions—they're **resonant transformations** with rigorous physics.

For complete specifications with physics derivations, contracts, and usage examples, see **[AGENTS.md § The 13 Canonical Operators](../AGENTS.md#-the-13-canonical-operators)**.

### Quick Reference

| Symbol | Name | Physics | Grammar Sets | When to Use |
|--------|------|---------|-------------|-------------|
| **AL** | Emission | Creates EPI from vacuum via resonant emission | Generator (U1a) | Starting new patterns, initializing from EPI=0 |
| **EN** | Reception | Captures and integrates incoming resonance | - | Information gathering, listening phase |
| **IL** | Coherence | Stabilizes form through negative feedback | Stabilizer (U2) | After changes, consolidation |
| **OZ** | Dissonance | Introduces controlled instability | Destabilizer (U2), Bifurcation trigger (U4a), Closure (U1b) | Breaking local optima, exploration |
| **UM** | Coupling | Creates structural links via phase synchronization | Requires phase verification (U3) | Network formation, connecting nodes |
| **RA** | Resonance | Amplifies and propagates patterns coherently | Requires phase verification (U3) | Pattern reinforcement, spreading coherence |
| **SHA** | Silence | Freezes evolution temporarily (νf → 0) | Closure (U1b) | Observation windows, pause for synchronization |
| **VAL** | Expansion | Increases structural complexity (dim ↑) | Destabilizer (U2) | Adding degrees of freedom |
| **NUL** | Contraction | Reduces structural complexity (dim ↓) | - | Simplification, dimensionality reduction |
| **THOL** | Self-organization | Spontaneous autopoietic pattern formation | Stabilizer (U2), Handler (U4a), Transformer (U4b) | Emergent organization, fractal structuring |
| **ZHIR** | Mutation | Phase transformation at threshold | Bifurcation trigger (U4a), Transformer (U4b) | Qualitative state changes |
| **NAV** | Transition | Regime shift, activates latent EPI | Generator (U1a), Closure (U1b) | Switching between attractor states |
| **REMESH** | Recursivity | Echoes structure across scales | Generator (U1a), Closure (U1b) | Multi-scale operations, memory |

### Operator Composition

Operators combine into **sequences** that implement complex behaviors:

- **Bootstrap** = [Emission, Coupling, Coherence]
- **Stabilize** = [Coherence, Silence]
- **Explore** = [Dissonance, Mutation, Coherence]
- **Propagate** = [Resonance, Coupling]

**Critical**: All sequences must satisfy unified grammar (U1-U6).

**API:** 
- `tnfr.structural.<OperatorName>()` - Individual operators
- `run_sequence(G, node, ops)` - Execute operator sequences
- `validate_sequence(ops)` - Check grammar compliance

**Grammar:** See [UNIFIED_GRAMMAR_RULES.md](UNIFIED_GRAMMAR_RULES.md) for complete rules  
**Detailed Specs:** See [AGENTS.md § The 13 Canonical Operators](../AGENTS.md#-the-13-canonical-operators)  
**Math:** [Mathematical Foundations](MATHEMATICAL_DYNAMICS_BASIS.md)

---

## Canonical Invariants (Optimized Set)

From [AGENTS.md](../AGENTS.md) - Optimized from 10 to 6 invariants based on mathematical derivation:

1. **Nodal Equation Integrity**: EPI evolution only via ∂EPI/∂t = νf · ΔNFR(t)
2. **Phase-Coherent Coupling**: |φᵢ - φⱼ| ≤ Δφ_max required for resonant operations
3. **Multi-Scale Fractality**: Operational fractality and nested EPIs maintained
4. **Grammar Compliance**: All operator sequences must satisfy U1-U6 validation
5. **Structural Metrology**: Units consistency (νf in Hz_str) and telemetry exposure
6. **Reproducible Dynamics**: Deterministic evolution with seed-based control

---

## Quick Reference Tables

### Variable Summary

| Symbol | Mathematical | Code Attribute | Units | Range | Type |
|--------|--------------|----------------|-------|-------|------|
| \(\text{EPI}\) | Primary Information Structure | `'EPI'` | dimensionless | \(B_{\text{EPI}}\) | Coherent form |
| \(\nu_f\) | Structural frequency | `'vf'` | Hz_str | \(\mathbb{R}^+\) | Reorganization rate |
| \(\Delta\text{NFR}\) | Reorganization operator | `'dnfr'` | dimensionless | \(\mathbb{R}\) | Evolution gradient |
| \(\theta\), \(\phi\) | Phase angle | `'theta'` | radians | \([0, 2\pi)\) | Network synchrony |
| \(C(t)\) | Total coherence | `compute_coherence()` | dimensionless | \([0, 1]\) | Global stability |
| \(\text{Si}\) | Sense Index | `'Si'` | dimensionless | \([0, 1^+]\) | Reorganization stability |

### Common API Patterns

```python
# Access node attributes
epi = G.nodes[node_id]['EPI']
vf = G.nodes[node_id]['vf']
theta = G.nodes[node_id]['theta']

# Compute metrics
C_t = compute_coherence(G)
nodes, W = coherence_matrix(G)
Si = compute_Si_node(G, node_id)

# Apply operators
from tnfr.structural import Emission, Coherence, Resonance
run_sequence(G, node_id, [Emission(), Coherence(), Resonance()])

# Evolution step
from tnfr.dynamics import step
step(G, use_Si=True, apply_glyphs=True)
```

---

## Telemetry & Traces

Expose in telemetry:
- `C(t)` - Total coherence
- `νf` per node - Structural frequency
- `phase` per node - Synchrony state
- `Si` per node/network - Sense index
- `ΔNFR` per node - Reorganization gradient
- Operator history - Applied transformations
- Events - Birth, bifurcation, collapse

**API:** `tnfr.utils.callback_manager`, history tracking in `G.graph['_hist']`

---

## Domain Neutrality

TNFR is designed to be **domain-neutral**:
- Applicable across multiple domains (network science, number theory, chemistry applications)
- No built-in assumptions about specific domains
- Structural operators apply to graph-coupled networks

**Guideline:** Avoid domain-specific hard-coding in core engine

---

## Reproducibility

All simulations must be:
1. **Seeded:** Explicit RNG seeds
2. **Traceable:** Log operators, parameters, states
3. **Deterministic:** Same seed → same trajectory

**Tools:** RNG scaffolding, structural history, telemetry caches

---

## Unified Grammar Terms

### Unified Grammar

The consolidated TNFR grammar system (**U1-U6**) that replaces the old C1-C3 and RC1-RC4 systems.

**Source of Truth:** [UNIFIED_GRAMMAR_RULES.md](UNIFIED_GRAMMAR_RULES.md)  
**Quick Reference:** [AGENTS.md § Unified Grammar (U1-U6)](../AGENTS.md#-unified-grammar-u1-u6)  
**Implementation:** `src/tnfr/operators/grammar.py`

**Grammar Completeness**: The canonical TNFR grammar consists of **exactly six rules (U1-U6)** and is **COMPLETE**. No additional rules (U7, U8, etc.) are required or planned. Extended dynamics (flux fields) add telemetry, not prescriptive constraints.

**Six Canonical Constraints:**

| Rule | Name | Physics Basis | Requirement | Canonicity |
|------|------|---------------|-------------|------------|
| **U1** | STRUCTURAL INITIATION & CLOSURE | ∂EPI/∂t undefined at EPI=0 | Start with generator {AL, NAV, REMESH}, End with closure {SHA, NAV, REMESH, OZ} | ABSOLUTE |
| **U2** | CONVERGENCE & BOUNDEDNESS | ∫νf·ΔNFR dt must converge | If destabilizer {OZ, ZHIR, VAL}, then include stabilizer {IL, THOL} | ABSOLUTE |
| **U3** | RESONANT COUPLING | Phase compatibility required for resonance | If coupling {UM, RA}, verify \|φᵢ - φⱼ\| ≤ Δφ_max | ABSOLUTE |
| **U4** | BIFURCATION DYNAMICS | ∂²EPI/∂t² > τ requires control | Triggers {OZ, ZHIR} need handlers {THOL, IL}; Transformers need context | STRONG |
| **U5** | MULTI-SCALE COHERENCE | Hierarchical coupling + chain rule | Nested EPIs require stabilizers {IL, THOL} at each level | ABSOLUTE |
| **U6** | STRUCTURAL POTENTIAL CONFINEMENT | Emergent Φ_s field: Φ_s(i) = Σ ΔNFR_j/d(i,j)² | Monitor Δ Φ_s < φ ≈ 1.618 (canonical confinement); ceiling 2.0 | STRONG |

**Canonicity Levels:**
- **ABSOLUTE**: Mathematical necessity (direct consequence of nodal equation)
- **STRONG**: Strong empirical/theoretical support (2,041 tests for U6)

**Recent Updates:**
- U5 added 2025-11-10 (hierarchical REMESH stabilization)
- U6 promoted to canonical 2025-11-11 (Φ_s field validation complete)
  - Replaces experimental "Temporal Ordering" research proposal
  - Validated across 5 topologies: ring, scale_free, small-world, tree, grid
  - Correlation: corr(Δ Φ_s, ΔC) = -0.822 (R² ≈ 0.68)
- 2025-11-15: Grammar declared COMPLETE (U1-U6) - no U7/U8 required

**Not Part of Grammar** (telemetry/dynamics, NOT rules):
- **Structural Field Hexad**: Tetrad (Φ_s, |∇φ|, K_φ, ξ_C) + Flux Pair (J_φ, ∇·J_ΔNFR)
- **"Proposed U7"**: Historical research direction (Temporal Ordering) - NOT canonical, NOT implemented

**See Also:**
- [UNIFIED_GRAMMAR_RULES.md](UNIFIED_GRAMMAR_RULES.md) - Complete derivations from physics
- [AGENTS.md § Unified Grammar](../AGENTS.md#-unified-grammar-u1-u6) - Quick reference
- [docs/grammar/U6_STRUCTURAL_POTENTIAL_CONFINEMENT.md](../docs/grammar/PHYSICS_VERIFICATION.md) - U6 complete specification
- [docs/grammar/U6_STRUCTURAL_FIELD_TETRAD.md](../docs/STRUCTURAL_FIELDS_TETRAD.md) - Why no U7/U8
- [STRUCTURAL_FIELDS_TETRAD.md](../docs/STRUCTURAL_FIELDS_TETRAD.md) - U6 validation details
- [src/tnfr/physics/fields.py](../src/tnfr/physics/fields.py) - Φ_s implementation

---

### Generator Operator

Operator that can create EPI from null/dormant states.

**Set:** GENERATORS = {emission, transition, recursivity}

**Physics:** Only these operators can initialize when EPI=0

**Grammar Rule:** U1a (STRUCTURAL INITIATION)

**See:** UNIFIED_GRAMMAR_RULES.md § U1a

---

### Closure Operator

Operator that leaves system in coherent attractor state.

**Set:** CLOSURES = {silence, transition, recursivity, dissonance}

**Physics:** Terminal states preserving coherence

**Grammar Rule:** U1b (STRUCTURAL CLOSURE)

**See:** UNIFIED_GRAMMAR_RULES.md § U1b

---

### Stabilizer Operator

Operator that provides negative feedback for convergence.

**Set:** STABILIZERS = {coherence, self_organization}

**Physics:** Ensures ∫νf·ΔNFR dt converges (bounded evolution)

**Grammar Rule:** U2 (CONVERGENCE & BOUNDEDNESS)

**See:** UNIFIED_GRAMMAR_RULES.md § U2

---

### Destabilizer Operator

Operator that increases |ΔNFR| through positive feedback.

**Set:** DESTABILIZERS = {dissonance, mutation, expansion}

**Physics:** Without stabilizers, leads to divergence

**Grammar Rule:** U2 (CONVERGENCE & BOUNDEDNESS)

**See:** UNIFIED_GRAMMAR_RULES.md § U2

---

### Coupling/Resonance Operator

Operators that require phase verification for valid coupling.

**Set:** COUPLING_RESONANCE = {coupling, resonance}

**Physics:** Resonance requires |φᵢ - φⱼ| ≤ Δφ_max

**Grammar Rule:** U3 (RESONANT COUPLING)

**See:** UNIFIED_GRAMMAR_RULES.md § U3

---

## The Structural-Field Tetrad

**Theory:** The four structural fields are the minimal derivative tower (DERIVED). They are *associated* with four constants as a notational label; only π is a genuine structural scale (the phase-wrap bound shared by |∇φ| and K_φ).

### The constants

The engine uses four mathematical constants (φ, γ, π, e) as notational labels for parameters. Only **π** is a genuine structural scale — it bounds the phase sector (|∇φ| ≤ π and |K_φ| ≤ π). φ, γ, e are notational; the bounds they label are empirical or heuristic, not derived from the nodal equation.

### Structural Fields and their bounds

1. **Φ_s** (0th order): empirical confinement Δ Φ_s < φ ≈ 1.618 (no closed form; φ is motivation only)
2. **|∇φ|** (1st order): bound |∇φ| ≤ π (phase wrap); γ/π ≈ 0.184 is a heuristic early-warning only
3. **K_φ** (2nd order): bound |K_φ| < 0.9×π ≈ 2.827 (phase wrap — GENUINE); K_φ = L_rw·φ
4. **ξ_C** (correlation): scale set by the spectral gap, ξ_C ∝ 1/√λ₂ (not base e)

**Documentation:** [Structural-field tetrad](FUNDAMENTAL_THEORY.md)

---

### Bifurcation Trigger

Operators that may trigger phase transitions.

**Set:** BIFURCATION_TRIGGERS = {dissonance, mutation}

**Physics:** Can cause ∂²EPI/∂t² > τ (bifurcation)

**Grammar Rule:** U4a (requires handlers)

**See:** UNIFIED_GRAMMAR_RULES.md § U4a

---

### Bifurcation Handler

Operators that manage structural reorganization during bifurcations.

**Set:** BIFURCATION_HANDLERS = {self_organization, coherence}

**Physics:** Provide stability during phase transitions

**Grammar Rule:** U4a (BIFURCATION DYNAMICS)

**See:** UNIFIED_GRAMMAR_RULES.md § U4a

---

### Transformer Operator

Operators that perform graduated destabilization for phase transitions.

**Set:** TRANSFORMERS = {mutation, self_organization}

**Physics:** Require recent destabilizer for threshold energy

**Grammar Rule:** U4b (requires context + prior IL for ZHIR)

**See:** UNIFIED_GRAMMAR_RULES.md § U4b

---

## Related Documentation

### Core References (Essential)
- **[AGENTS.md](../AGENTS.md)** ⭐ - Single source of truth for TNFR agent guidance, invariants, and philosophy
- **[UNIFIED_GRAMMAR_RULES.md](UNIFIED_GRAMMAR_RULES.md)** ⭐ - Grammar single source of truth (U1-U6 complete derivations)
- **[Mathematical Foundations](MATHEMATICAL_DYNAMICS_BASIS.md)** ⭐ - **SINGLE SOURCE FOR ALL MATH** (formalization, proofs, spectral theory)

### Theory & Physics
- [TNFR.pdf](TNFR.pdf) - Original theoretical companion (paradigm, nodal equation, foundational physics)
- [docs/grammar/U6_STRUCTURAL_POTENTIAL_CONFINEMENT.md](../docs/grammar/PHYSICS_VERIFICATION.md) - U6 complete specification
- [STRUCTURAL_FIELDS_TETRAD.md](../docs/STRUCTURAL_FIELDS_TETRAD.md) - Structural fields validation (Φ_s, phase gradients)


### Implementation & API
- [ARCHITECTURE.md](../ARCHITECTURE.md) - System design and architecture patterns
- [Foundations](../AGENTS.md) - Runtime/API guide
- [API Overview](../AGENTS.md) - Package architecture
- [Structural Operators](STRUCTURAL_OPERATORS.md) - Operator implementation details
- [Examples](../examples/README.md) - Runnable scenarios across domains

### Grammar & Migration
- [docs/grammar/](../docs/grammar/) - Grammar documentation directory (U6, fundamental concepts, etc.)

### Testing & Development
- [TESTING.md](../TESTING.md) - Test conventions and invariant verification
- [CONTRIBUTING.md](../CONTRIBUTING.md) - Detailed contribution guidelines
- [REPRODUCIBILITY.md](../AGENTS.md) - Determinism requirements

### Cross-References and Documentation Hub

**Primary Sources:**  
- **[AGENTS.md](../AGENTS.md)** - Single source of truth for TNFR theory  
- **[UNIFIED_GRAMMAR_RULES.md](UNIFIED_GRAMMAR_RULES.md)** - Complete U1-U6 grammar derivations  
- **[Structural Fields and the Tetrad](FUNDAMENTAL_THEORY.md)** - Mathematical foundations

**Implementation References:**  
- **[src/tnfr/physics/fields.py](../src/tnfr/physics/fields.py)** - Unified Structural Field Tetrad (Canonical)  
- **[src/tnfr/dynamics/self_optimizing_engine.py](../src/tnfr/dynamics/self_optimizing_engine.py)** - Self-optimization & auto-optimization  
- **[docs/STRUCTURAL_FIELDS_TETRAD.md](../docs/STRUCTURAL_FIELDS_TETRAD.md)** - Technical field specifications  
- **[docs/grammar/PHYSICS_VERIFICATION.md](../docs/grammar/PHYSICS_VERIFICATION.md)** - Grammar physics verification

**Development Resources:**  
- **[src/tnfr/sdk/](../src/tnfr/sdk/)** - Simplified & Fluent API  
- **[examples/](../examples/)** - Complete tutorial suite  
- **[ARCHITECTURE.md](../ARCHITECTURE.md)** - System design patterns

---

## Molecular Chemistry from TNFR

**Technical approach**: Chemistry modeled via TNFR nodal dynamics applied to atomic-scale graph networks.

### Element Signatures

**Code:** `tnfr.physics.signatures`  
**What:** Structural field-based classification of coherent patterns  
**Metrics:** ξ_C, |∇φ|, |K_φ|, ΔΦ_s drift, stability classification  
**API:** `compute_element_signature(G)`, `compute_au_like_signature(G)`  
**Physics:** Elements as coherent attractors in structural space  

### Au-like Patterns

**Symbol:** Au (from Latin 'aurum')  
**What:** Complex coherent patterns exhibiting metallic properties  
**Criteria:** Extended ξ_C, phase synchrony (|∇φ| < 2.0), evolution stability  
**Detection:** `compute_au_like_signature()["is_au_like"]`  
**Physics:** Optimal multi-scale coordination under nodal dynamics  

### Chemical Bonds (TNFR Structural Analogy)

**Traditional:** Force between atoms  
**TNFR:** Phase synchronization with U3 verification: |φᵢ - φⱼ| ≤ Δφ_max  
**API:** Coupling operators with phase compatibility check  
**Strength:** Determined by phase coherence and coupling stability  

### Chemical Reactions (TNFR Structural Analogy)

**Traditional:** Collision/transition state theory  
**TNFR:** Operator sequences: [Dissonance→Mutation→Coupling→Coherence]  
**Grammar:** Must satisfy U1-U6 constraints  
**API:** Sequence validation via `grammar.py`  
**Example:** Bond formation = [OZ, ZHIR, UM, IL] sequence  

### Molecular Geometry (TNFR Structural Analogy)

**Traditional:** VSEPR, orbital hybridization  
**TNFR:** ΔNFR minimization in coupled network topology  
**Prediction:** Stable configurations minimize reorganization pressure  
**API:** Network topology analysis after coupling sequences  

**Complete Theory:** [MOLECULAR_CHEMISTRY_FROM_NODAL_DYNAMICS.md](../examples/07_number_theory/emergent_chemistry_particles_demo.py)  
**Implementation:** [Physics README § 9-10](../src/tnfr/physics/README.md)

## Self-Optimizing Engine

**Self-Optimization:** The TNFR engine includes self-optimization capabilities using unified field telemetry.

### Core Components

**TNFRSelfOptimizingEngine:** `src/tnfr/dynamics/self_optimizing_engine.py`  
**Purpose:** Closes feedback loop via unified field monitoring  
**Monitors:** Complex Geometric Field (Ψ), Chirality (χ), Symmetry Breaking (𝒮), Coherence Coupling (𝒞)  
**Detects:** Inefficiencies via tensor invariants (Energy Density ℰ, Topological Charge 𝒬)  
**Usage:** `engine = TNFRSelfOptimizingEngine(G); success, metrics = engine.step(node_id)`

### Auto-Optimization API

**Fluent Integration:** `TNFRNetwork(G).focus(node).auto_optimize().execute()`  
**Field Analysis:** `analyze_optimization_potential(G)` - Mathematical structure analysis  
**Strategy Recommendations:** `recommend_field_optimization_strategy(G)` - Optimization strategies  
**Automatic Execution:** `auto_optimize_field_computation(G)` - Self-optimizing computation

## Unified Field Framework (Nov 2025)

**Mathematical Unification:** Discovery of complex field relationships and conservation laws.

### Complex Geometric Field (Ψ)

**Definition:** Ψ = K_φ + i·J_φ (unifies geometry + transport)  
**Evidence:** r(K_φ, J_φ) = -0.854 to -0.997 (near-perfect anticorrelation)  
**API:** `compute_complex_geometric_field(G)`  
**Usage:** Unified geometry-transport analysis

### Emergent Fields

**Chirality (χ):** `χ = |∇φ|·K_φ - J_φ·J_ΔNFR` - Handedness detection  
**Symmetry Breaking (𝒮):** Phase transition indicator  
**Coherence Coupling (𝒞):** Multi-scale connector field  
**API:** `compute_emergent_fields(G)`

### Tensor Invariants

**Energy Density (ℰ):** `ℰ = Φ_s² + |∇φ|² + K_φ² + J_φ² + J_ΔNFR²`  
**Topological Charge (𝒬):** `𝒬 = |∇φ|·J_φ - K_φ·J_ΔNFR`  
**Conservation Law:** ∂ρ/∂t + ∇·𝐉 = S_grammar where S_grammar → 0 under U1-U6  
**API:** `compute_tensor_invariants(G)`

**Unified Telemetry:** `compute_unified_telemetry(G)` - Complete field suite

---

## Operator-Tetrad Synergies (Experimental, March 2026)

Six experimentally validated results connecting canonical operators to the structural field tetrad.
Reference: [STRUCTURAL_OPERATORS.md §17](STRUCTURAL_OPERATORS.md), examples 37-39.

### Dual-Lever Structure

**What:** Operators modify the nodal equation through exactly one of two channels (or both, or neither):
- **Capacity lever** (vf): UM, SHA, VAL, NUL adjust the reorganization rate.
- **Pressure lever** (DNFR): IL, OZ, THOL, ZHIR, NAV adjust the structural pressure.
- **Neutral**: AL, EN, RA, REMESH do not directly modify either lever at the single-node level.
**Evidence:** Classification from `examples/02_physics_regimes/39_nodal_equation_decomposition.py`.

### Operator-Tetrad Fingerprint Matrix

**What:** Each operator produces a unique signature across the four tetrad fields. The fingerprint matrix tabulates relative changes (dPhi_s, d|grad_phi|, dK_phi, dxi_C) per operator.
**Example:** UM modifies all four fields (strongest Phi_s at -73.7%); SHA is tetrad-neutral.
**Evidence:** `examples/02_physics_regimes/37_operator_tetrad_synergy.py`.

### IL-OZ Tetrad Symmetry

**What:** Coherence (IL) and Dissonance (OZ) produce identical tetrad perturbation magnitudes and identical energy changes (dE = -0.011) despite opposite physics. They share the same |d(DNFR)| = 0.0096, differing only in sign.
**Interpretation:** IL and OZ are structural mirrors on the energy manifold, explaining U2 balancing.

### Linear Response of Phi_s

**What:** Phi_s responds linearly to DNFR perturbations with |r| = 1.000 (Pearson correlation). This confirms its 0th-order position in the operator-derivative tower.
**Contrast:** xi_C transitions to strongly nonlinear behaviour above DNFR ~ 0.3.
**Evidence:** `examples/02_physics_regimes/39_nodal_equation_decomposition.py`.

### Complete Causal Chain

**What:** The information flow is strictly unidirectional:
`Operator -> (vf, DNFR) -> dEPI/dt -> Tetrad -> (E, Q)`.
Tetrad fields are diagnostic outputs, not independent dynamical variables. They are fully determined by the nodal equation state.

### Grammar-Energy Landscape

**What:** Lyapunov contractivity (cumulative multiplier Pi < 1) is sufficient but not necessary for energy descent. Experimentally: lambda = 1.288 (non-contractive) yet net dE = -9.59 (energy descent).
**Interpretation:** The Lyapunov bound is conservative; actual grammar-compliant sequences may descend more steeply than the bound predicts.
**Evidence:** `examples/02_physics_regimes/38_grammar_energy_landscape.py`.

---

## Structural Conservation Theorem

**Main Result:** Grammar symmetry (U1-U6) implies an approximate Noether-like structural conservation law.

**Charge Density:** ρ = Φ_s + K_φ (potential + geometric sectors)  
**Current:** 𝐉 = (J_φ, J_ΔNFR) (transport channels)  
**Conservation:** ∂ρ/∂t + ∇·𝐉 = S_grammar where S_grammar → 0 under U1-U6  
**Two Sectors:** Potential (Φ_s ↔ J_ΔNFR) and Geometric (K_φ ↔ J_φ), coupled through Ψ = K_φ + i·J_φ  
**Lyapunov:** E = ½Σ(Φ_s² + |∇φ|² + K_φ² + J_φ² + J_ΔNFR²) ≥ 0 with dE/dt ≤ 0 observed under grammar (proof sketch; complete proof open)  
**Validation:** 62 tests, charge drift < 0.03% across tested topologies and seeds  
**API:** `tnfr.physics.conservation` — Noether charge Q, energy functional E, Ward identities, spectral decomposition  
**Documentation:** [theory/STRUCTURAL_CONSERVATION_THEOREM.md](STRUCTURAL_CONSERVATION_THEOREM.md)

---

## Integrity Monitor

**What:** Closed-loop postcondition verification ensuring all 13 canonical operators satisfy their structural contracts after execution.  
**API:** `tnfr.physics.integrity`  
**Coverage:** 13/13 operators verified — monotonicity (IL), ΔNFR increase (OZ), phase preservation (SHA), EPI creation (AL), coupling validity (UM/RA), etc.  
**Usage:** Automatic contract verification in `apply_glyph_with_grammar()` pipeline.  
**Documentation:** [src/tnfr/physics/integrity.py](../src/tnfr/physics/integrity.py)

---

## Grammar-Aware Dynamics

**What:** Incremental U1-U6 enforcement during step-by-step operator selection, bridging grammar validation with the dynamic operator selection layer.  
**API:** `tnfr.operators.grammar_dynamics.GrammarAwareDynamics`  
**Checks:** U1a initiation, U2 destabilizer/stabilizer debt tracking, U3 phase compatibility for UM/RA, U4a/U4b bifurcation context  
**Physics:** Proactive enforcement prevents grammar violations *before* they corrupt graph state.  
**Documentation:** [src/tnfr/operators/grammar_dynamics.py](../src/tnfr/operators/grammar_dynamics.py)

---

## Grammar Application

**What:** Pre-validated, grammar-enforced operator application at runtime.  
**API:** `tnfr.operators.grammar_application.apply_glyph_with_grammar()`  
**Pipeline:** Grammar check → operator application → postcondition verification (integrity monitor)  
**Physics:** Ensures every structural mutation passes U1-U6 before modifying graph state.  
**Documentation:** [src/tnfr/operators/grammar_application.py](../src/tnfr/operators/grammar_application.py)

---

## Von Koch Threshold

**Value:** PHI_S_VON_KOCH_THRESHOLD = 0.7711  
**What:** Per-node safety threshold for structural potential |Φ_s|.  
**Derivation:** Empirically validated, confirmed across 5 topologies. **No closed-form first-principles derivation is currently established** (open problem). The constant name retains "VON_KOCH" for code-compatibility, but the previously claimed identity Γ(4/3)/Γ(1/3) ≈ 0.7711 is **incorrect**: Γ(4/3)/Γ(1/3) = 1/3, not 0.7711.  
**Usage:** |Φ_s(i)| < 0.7711 indicates safe per-node structural potential.  
**API:** `tnfr.constants.canonical.PHI_S_VON_KOCH_THRESHOLD`  
**Relation to U6:** Part of three-tier Φ_s monitoring: 0.7711 (per-node) → φ ≈ 1.618 (drift confinement) → 2.0 (escape ceiling).

---

## Classical-Quantum Regime Emergence

**Theory:** Classical and quantum mechanics emerge as different structural regimes of nodal dynamics, not distinct sets of laws.

### Classical Limit (High Coherence)
**Condition:** C(t) → 1, |∇φ| → 0  
**Correspondence:** m = 1/νf (mass ↔ inverse frequency), F = ΔNFR (force ↔ structural pressure)  
**API:** `tnfr.physics.classical_mechanics`

### Quantum Regime (High Dissonance)
**Condition:** |∇φ| ~ π, near phase singularities  
**Emergent:** Discrete states (resonant eigenmodes), uncertainty (Fourier ΔEPi·Δνf ≥ K), superposition  
**API:** `tnfr.physics.quantum_mechanics`

---

## TNFR-Riemann Program

**What:** Theoretical framework connecting discrete TNFR operators to the Riemann Hypothesis through structural coherence.  
**Core Operator:** H^(k)(σ) = L_k + V_σ where L_k = graph Laplacian, V_σ = diagonal potential  
**Discovery:** Critical parameter σ_c^(k) → 1/2 as k → ∞  
**Convergence:** σ_c^(k) = 1/2 + O(log⁻¹ k) (universal across topologies)  
**Implementation:** `src/tnfr/riemann/` — 14 modules (operator, spectral_proof, complex_extension, spectral_zeta, topology, spectral_conservation, analytical_convergence, etc.)  
**Documentation:** [theory/TNFR_RIEMANN_RESEARCH_NOTES.md](TNFR_RIEMANN_RESEARCH_NOTES.md)

---

## Key Canonical Thresholds

Quick reference for canonical threshold values from `src/tnfr/constants/canonical.py`:

| Threshold | Value | Derivation | Usage |
|-----------|-------|------------|-------|
| PHI_S_VON_KOCH_THRESHOLD | 0.7711 | Empirical (no closed-form derivation; Γ(4/3)/Γ(1/3)=1/3, not 0.7711) | Per-node Φ_s safety |
| PHASE_GRADIENT_THRESHOLD | γ/π ≈ 0.1837 | Heuristic early-warning (not derived; bound is π) | \|∇φ\| stability |
| K_PHI_CANONICAL_THRESHOLD | 0.9×π ≈ 2.8274 | 90% of wrap_angle π bound (genuine) | K_φ fault zone detection |
| U6 canonical confinement | φ ≈ 1.618 | Empirical/notational (φ is motivation, no closed form) | ΔΦ_s drift safety |
| STRUCTURAL_ESCAPE_THRESHOLD | e^ln(2) = 2.0 | Binary escape theory | ΔΦ_s absolute ceiling |
| MIN_BUSINESS_COHERENCE | (e×φ)/(π+e) ≈ 0.7506 | Notational combination | Strong coherence threshold |
| THOL_MIN_COLLECTIVE_COHERENCE | 1/(π+1) ≈ 0.2415 | Geometric series bound | Fragmentation risk threshold |

---

## Contributing Guidelines

When adding new functionality:

1. **Verify theoretical foundation**: Align with [AGENTS.md](../AGENTS.md) physics  
2. **Preserve canonical invariants**: Follow optimized 6-invariant set  
3. **Use established terminology**: Reference this glossary for consistency  
4. **Map to canonical operators**: All functions must correspond to 13 canonical operators  
5. **Validate grammar compliance**: Ensure U1-U6 satisfaction  
6. **Maintain English-only policy**: All documentation in English for canonical terminology  
7. **Write comprehensive tests**: Cover invariants and operator contracts

**Development Workflow:**  
1. Read [AGENTS.md](../AGENTS.md) completely - **SINGLE SOURCE OF TRUTH**  
2. Study [UNIFIED_GRAMMAR_RULES.md](UNIFIED_GRAMMAR_RULES.md) for physics foundations  
3. Follow [CONTRIBUTING.md](../CONTRIBUTING.md) for detailed guidelines  
4. Test with [TESTING.md](../TESTING.md) requirements

**Version**: 0.0.3.3 (March 2026)  
**Status**: Complete operational reference for current TNFR implementation  
**Language**: English only (canonical documentation policy)
