# TNFR Glossary

**Purpose**: Operational quick reference for the Resonant Fractal Nature Theory (TNFR) v9.7.0  
**Status**: Complete reference for current implementation  
**Version**: November 29, 2025  
**Authority**: Aligned with [AGENTS.md](AGENTS.md) as single source of truth  

**Scope**: This glossary provides **API-focused definitions** for developers implementing TNFR networks. For complete theoretical foundations, see [AGENTS.md](AGENTS.md) and [UNIFIED_GRAMMAR_RULES.md](UNIFIED_GRAMMAR_RULES.md).

---

## Core Variables

### Primary Information Structure (EPI)

**Code:** `G.nodes[n]['EPI']`, `ALIAS_EPI`  
**Symbol:** \(\text{EPI}\) or \(E\)  
**What:** Coherent structural form of a node  
**Space:** \(B_{\text{EPI}}\) (Banach space)  
**Rules:** Modified only via structural operators, never directly  
**API:** `tnfr.structural` operators  
**Math:** [§2.2 Banach Space B_EPI](docs/source/theory/mathematical_foundations.md#22-banach-space-b_epi)

### Structural Frequency (νf)

**Code:** `G.nodes[n]['vf']`, `ALIAS_VF`  
**Symbol:** \(\nu_f\)  
**Units:** Hz_str (structural hertz)  
**Range:** \(\mathbb{R}^+\) (positive reals; node collapse when \(\nu_f \to 0\))  
**What:** Rate of structural reorganization  
**API:** `adapt_vf_by_coherence()`, operators  
**Math:** [§3.2 Frequency Operator Ĵ](docs/source/theory/mathematical_foundations.md#32-frequency-operator-ĵ)

### Internal Reorganization Operator (ΔNFR)

**Code:** `G.nodes[n]['dnfr']`, `ALIAS_DNFR`  
**Symbol:** \(\Delta\text{NFR}\)  
**What:** Structural evolution gradient (drives reorganization)  
**Sign:** Positive = expansion, Negative = contraction  
**Compute:** Via `default_compute_delta_nfr` hook, automatic in `step()`  
**Math:** [§3.3 Reorganization Operator](docs/source/theory/mathematical_foundations.md#33-reorganization-operator-δnfr)

### Phase (φ, θ)

**Code:** `G.nodes[n]['theta']`, `collect_theta_attr()`  
**Symbol:** \(\theta\) or \(\phi\)  
**Range:** \([0, 2\pi)\) or \([-\pi, \pi)\) radians  
**What:** Network synchrony parameter (relative timing)  
**Phase difference:** \(\Delta\theta = \theta_i - \theta_j\)  
**API:** Phase adaptation in dynamics  
**Math:** [§4 Nodal Equation](docs/source/theory/mathematical_foundations.md#4-the-nodal-equation-complete-derivation)

### Total Coherence (C(t))

**Code:** `compute_coherence(G)` → float ∈ [0,1]  
**Symbol:** \(C(t)\)  
**Formula:** \(C(t) = \text{Tr}(\hat{C}\rho)\) where \(\hat{C}\) is the coherence operator  
**Range:** \([0, 1]\) where 1 = perfect coherence, 0 = total fragmentation  
**What:** Global network stability measure  
**Math:** [§3.1 Coherence Operator Ĉ](docs/source/theory/mathematical_foundations.md#31-coherence-operator-ĉ)

### Coherence Operator (Ĉ)

**Code:** `coherence_matrix(G)` → (nodes, W)  
**Symbol:** \(\hat{C}\)  
**Matrix element:** \(w_{ij} \approx \langle i | \hat{C} | j \rangle\)  
**Properties:** Hermitian (\(\hat{C}^\dagger = \hat{C}\)), positive semi-definite  
**What:** Operator measuring structural stability between nodes  
**Math:** [§3.1 Theory](docs/source/theory/mathematical_foundations.md#31-coherence-operator-ĉ) + [§3.1.1 Implementation](docs/source/theory/mathematical_foundations.md#311-implementation-bridge-theory-to-code)

### Sense Index (Si)

**Code:** `G.nodes[n]['Si']`, `ALIAS_SI`, `compute_Si_node()`  
**Symbol:** \(\text{Si}\) (global) or \(S_i\) (node i)  
**Formula:** \(\text{Si} = \alpha \cdot \nu_{f,\text{norm}} + \beta \cdot (1 - \text{disp}_\theta) + \gamma \cdot (1 - |\Delta\text{NFR}|_{\text{norm}})\)  
**Range:** \([0, 1^+]\) typically, higher = more stable reorganization  
**What:** Capacity for stable structural reorganization  
**Weights:** \(\alpha + \beta + \gamma = 1\) (default: 0.4, 0.3, 0.3)  
**Math:** [Mathematical Foundations - Metrics](docs/source/theory/mathematical_foundations.md)

### Phase Gradient (|∇φ|) - CANONICAL

**Code:** `compute_phase_gradient(G)` → Dict[NodeId, float]  
**Symbol:** \(|\nabla\phi|(i)\)  
**Formula:** \(|\nabla\phi|(i) = \text{mean}_{j \in N(i)} |\theta_i - \theta_j|\) (circular mean)  
**What:** Local phase desynchronization / stress proxy field  
**Status:** **CANONICAL** (Nov 2025)  
**Physics:** Captures dynamics C(t) misses due to scaling invariance  
**Threshold:** Classical bound |∇φ| < 0.2904 (harmonic oscillator stability)  
**API:** `tnfr.physics.fields.compute_phase_gradient()`  
**Usage:** Stress detection, local instability prediction  
**Documentation:** [docs/STRUCTURAL_FIELDS_TETRAD.md](docs/STRUCTURAL_FIELDS_TETRAD.md)

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
**Documentation:** [docs/STRUCTURAL_FIELDS_TETRAD.md](docs/STRUCTURAL_FIELDS_TETRAD.md)

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
**Documentation:** [docs/STRUCTURAL_FIELDS_TETRAD.md](docs/STRUCTURAL_FIELDS_TETRAD.md)

---

### Structural Potential (Φ_s) - CANONICAL

**Code:** `compute_structural_potential(G, alpha=2.0)` → Dict[NodeId, float]  
**Symbol:** \(\Phi_s(i)\)  
**Formula:** \(\Phi_s(i) = \sum_{j \neq i} \frac{\Delta\text{NFR}_j}{d(i,j)^\alpha}\) where \(\alpha = 2\)  
**What:** Global structural potential field from ΔNFR distribution  
**Status:** **CANONICAL** (Nov 2025)  
**Validation:** 2,400+ experiments across 5 topologies  
**Physics:** Passive equilibrium confinement landscape  
**Grammar:** U6 STRUCTURAL POTENTIAL CONFINEMENT (Δ Φ_s < 2.0 escape threshold)  
**API:** `tnfr.physics.fields.compute_structural_potential()`  
**Threshold:** Classical bound |Φ_s| < 0.771 (von Koch fractal theory)  
**Documentation:** [docs/STRUCTURAL_FIELDS_TETRAD.md](docs/STRUCTURAL_FIELDS_TETRAD.md)
- [src/tnfr/physics/fields.py](src/tnfr/physics/fields.py) - Implementation

**Interpretation:**
- Φ_s minima = passive equilibrium states
- Δ Φ_s < 2.0 = system confined (safe regime)
- Δ Φ_s ≥ 2.0 = escape threshold (fragmentation risk)
- Valid sequences: Δ Φ_s ≈ 0.6 (30% of threshold)
- Violations: Δ Φ_s ≈ 3.9 (195% of threshold)

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
**Theory:** [§4 The Nodal Equation](docs/source/theory/mathematical_foundations.md#4-the-nodal-equation-complete-derivation)

---

## Structural Operators

The 13 canonical operators are the **only way** to modify nodes in TNFR. They're not arbitrary functions—they're **resonant transformations** with rigorous physics.

For complete specifications with physics derivations, contracts, and usage examples, see **[AGENTS.md § The 13 Canonical Operators](AGENTS.md#-the-13-canonical-operators)**.

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
**Detailed Specs:** See [AGENTS.md § The 13 Canonical Operators](AGENTS.md#-the-13-canonical-operators)  
**Math:** [Mathematical Foundations §5](docs/source/theory/mathematical_foundations.md)

---

## Canonical Invariants (Optimized Set)

From [AGENTS.md](AGENTS.md) - Optimized from 10 to 6 invariants based on mathematical derivation:

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

TNFR is **trans-scale** and **trans-domain**:
- Works from quantum to social systems
- No built-in assumptions about specific domains
- Structural operators apply universally

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
**Quick Reference:** [AGENTS.md § Unified Grammar (U1-U6)](AGENTS.md#-unified-grammar-u1-u6)  
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
| **U6** | STRUCTURAL POTENTIAL CONFINEMENT | Emergent Φ_s field: Φ_s(i) = Σ ΔNFR_j/d(i,j)² | Monitor Δ Φ_s < 2.0 (telemetry-based safety) | STRONG |

**Canonicity Levels:**
- **ABSOLUTE**: Mathematical necessity (direct consequence of nodal equation)
- **STRONG**: Strong empirical/theoretical support (2,400+ experiments for U6)

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
- [AGENTS.md § Unified Grammar](AGENTS.md#-unified-grammar-u1-u6) - Quick reference
- [docs/grammar/U6_STRUCTURAL_POTENTIAL_CONFINEMENT.md](docs/grammar/U6_STRUCTURAL_POTENTIAL_CONFINEMENT.md) - U6 complete specification
- [docs/grammar/U6_STRUCTURAL_FIELD_TETRAD.md](docs/grammar/U6_STRUCTURAL_FIELD_TETRAD.md) - Why no U7/U8
- [TNFR_FORCES_EMERGENCE.md § 14-15](docs/TNFR_FORCES_EMERGENCE.md) - U6 validation details
- [src/tnfr/physics/fields.py](src/tnfr/physics/fields.py) - Φ_s implementation

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

## Universal Tetrahedral Correspondence

**Theory:** Central TNFR discovery establishing exact correspondence between four universal mathematical constants and four structural fields.

### Mathematical Constants

| Constant | Value | Role | Domain |
|----------|-------|------|--------|
| **φ** (Golden Ratio) | 1.618034... | Harmonic proportion | Global/Harmonic |
| **γ** (Euler Constant) | 0.577216... | Harmonic growth rate | Local/Dynamic |
| **π** (Pi) | 3.141593... | Geometric relations | Geometric/Spatial |
| **e** (Euler Number) | 2.718282... | Exponential base | Correlational/Temporal |

### Structural Field Correspondences

1. **φ ↔ Φ_s**: Global harmonic confinement (Δ Φ_s < φ ≈ 1.618)
2. **γ ↔ |∇φ|**: Local dynamic evolution (|∇φ| < γ/π ≈ 0.184)
3. **π ↔ K_φ**: Geometric spatial constraints (|K_φ| < φ×π ≈ 5.083)
4. **e ↔ ξ_C**: Correlational memory decay (C(r) ~ exp(-r/ξ_C))

**Documentation:** [FUNDAMENTAL_TNFR_THEORY_UNIVERSAL_TETRAHEDRAL_CORRESPONDENCE.md](FUNDAMENTAL_TNFR_THEORY_UNIVERSAL_TETRAHEDRAL_CORRESPONDENCE.md)

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
- **[AGENTS.md](AGENTS.md)** ⭐ - Single source of truth for TNFR agent guidance, invariants, and philosophy
- **[UNIFIED_GRAMMAR_RULES.md](UNIFIED_GRAMMAR_RULES.md)** ⭐ - Grammar single source of truth (U1-U6 complete derivations)
- **[Mathematical Foundations](docs/source/theory/mathematical_foundations.md)** ⭐ - **SINGLE SOURCE FOR ALL MATH** (formalization, proofs, spectral theory)

### Theory & Physics
- [TNFR.pdf](TNFR.pdf) - Original theoretical companion (paradigm, nodal equation, foundational physics)
- [docs/grammar/U6_STRUCTURAL_POTENTIAL_CONFINEMENT.md](docs/grammar/U6_STRUCTURAL_POTENTIAL_CONFINEMENT.md) - U6 complete specification
- [TNFR_FORCES_EMERGENCE.md](docs/TNFR_FORCES_EMERGENCE.md) - Structural fields validation (Φ_s, phase gradients)
- [SHA_ALGEBRA_PHYSICS.md](SHA_ALGEBRA_PHYSICS.md) - Silence operator physical basis

### Implementation & API
- [ARCHITECTURE.md](ARCHITECTURE.md) - System design and architecture patterns
- [Foundations](docs/source/foundations.md) - Runtime/API guide
- [API Overview](docs/source/api/overview.md) - Package architecture
- [Structural Operators](docs/source/api/operators.md) - Operator implementation details
- [Examples](docs/source/examples/README.md) - Runnable scenarios across domains

### Grammar & Migration
- [MIGRATION_GUIDE.md](MIGRATION_GUIDE.md) - Migration from C1-C3/RC1-RC4 to U1-U6
- [docs/grammar/](docs/grammar/) - Grammar documentation directory (U6, fundamental concepts, etc.)
- [GLYPH_SEQUENCES_GUIDE.md](GLYPH_SEQUENCES_GUIDE.md) - Operator sequence patterns

### Testing & Development
- [TESTING.md](TESTING.md) - Test conventions and invariant verification
- [CONTRIBUTING.md](CONTRIBUTING.md) - Detailed contribution guidelines
- [REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) - Determinism requirements

### Cross-References and Documentation Hub

**Primary Sources:**  
- **[AGENTS.md](AGENTS.md)** - Single source of truth for TNFR theory  
- **[UNIFIED_GRAMMAR_RULES.md](UNIFIED_GRAMMAR_RULES.md)** - Complete U1-U6 grammar derivations  
- **[FUNDAMENTAL_TNFR_THEORY_UNIVERSAL_TETRAHEDRAL_CORRESPONDENCE.md](FUNDAMENTAL_TNFR_THEORY_UNIVERSAL_TETRAHEDRAL_CORRESPONDENCE.md)** - Mathematical foundations

**Implementation References:**  
- **[src/tnfr/physics/fields.py](src/tnfr/physics/fields.py)** - Unified Structural Field Tetrad (Canonical)  
- **[src/tnfr/dynamics/self_optimizing_engine.py](src/tnfr/dynamics/self_optimizing_engine.py)** - Intrinsic agency & auto-optimization  
- **[docs/STRUCTURAL_FIELDS_TETRAD.md](docs/STRUCTURAL_FIELDS_TETRAD.md)** - Technical field specifications  
- **[docs/grammar/PHYSICS_VERIFICATION.md](docs/grammar/PHYSICS_VERIFICATION.md)** - Grammar physics verification

**Development Resources:**  
- **[src/tnfr/sdk/](src/tnfr/sdk/)** - Simplified & Fluent API  
- **[examples/](examples/)** - Complete tutorial suite  
- **[ARCHITECTURE.md](ARCHITECTURE.md)** - System design patterns

---

## Molecular Chemistry from TNFR ⭐ **BREAKTHROUGH**

**Revolutionary paradigm**: Chemistry emerges from TNFR nodal dynamics without additional postulates.

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

### Chemical Bonds (TNFR Redefinition)

**Traditional:** Force between atoms  
**TNFR:** Phase synchronization with U3 verification: |φᵢ - φⱼ| ≤ Δφ_max  
**API:** Coupling operators with phase compatibility check  
**Strength:** Determined by phase coherence and coupling stability  

### Chemical Reactions (TNFR Redefinition)

**Traditional:** Collision/transition state theory  
**TNFR:** Operator sequences: [Dissonance→Mutation→Coupling→Coherence]  
**Grammar:** Must satisfy U1-U6 constraints  
**API:** Sequence validation via `grammar.py`  
**Example:** Bond formation = [OZ, ZHIR, UM, IL] sequence  

### Molecular Geometry (TNFR Redefinition)

**Traditional:** VSEPR, orbital hybridization  
**TNFR:** ΔNFR minimization in coupled network topology  
**Prediction:** Stable configurations minimize reorganization pressure  
**API:** Network topology analysis after coupling sequences  

**Complete Theory:** [MOLECULAR_CHEMISTRY_FROM_NODAL_DYNAMICS.md](docs/examples/MOLECULAR_CHEMISTRY_FROM_NODAL_DYNAMICS.md)  
**Implementation:** [Physics README § 9-10](src/tnfr/physics/README.md)

## Self-Optimizing Engine (v9.7.0)

**Intrinsic Agency:** The TNFR engine possesses self-optimization capabilities using unified field telemetry.

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
**Conservation Law:** ∂ρ/∂t + ∇·𝐉 = 0  
**API:** `compute_tensor_invariants(G)`

**Unified Telemetry:** `compute_unified_telemetry(G)` - Complete field suite

---

## Contributing Guidelines

When adding new functionality:

1. **Verify theoretical foundation**: Align with [AGENTS.md](AGENTS.md) physics  
2. **Preserve canonical invariants**: Follow optimized 6-invariant set  
3. **Use established terminology**: Reference this glossary for consistency  
4. **Map to canonical operators**: All functions must correspond to 13 canonical operators  
5. **Validate grammar compliance**: Ensure U1-U6 satisfaction  
6. **Maintain English-only policy**: All documentation in English for canonical terminology  
7. **Write comprehensive tests**: Cover invariants and operator contracts

**Development Workflow:**  
1. Read [AGENTS.md](AGENTS.md) completely - **SINGLE SOURCE OF TRUTH**  
2. Study [UNIFIED_GRAMMAR_RULES.md](UNIFIED_GRAMMAR_RULES.md) for physics foundations  
3. Follow [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines  
4. Test with [TESTING.md](TESTING.md) requirements

**Version**: 9.7.0 (November 29, 2025)  
**Status**: Complete operational reference for current TNFR implementation  
**Language**: English only (canonical documentation policy)
