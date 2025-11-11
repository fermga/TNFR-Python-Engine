# TNFR Glossary

Quick operational reference for the Resonant Fractal Nature Theory (TNFR). This document provides **API-focused definitions for code use only**.

> **📐 SINGLE SOURCE OF TRUTH FOR MATHEMATICS**: 
> 
> ### [Mathematical Foundations of TNFR](docs/source/theory/mathematical_foundations.md)
>
> **All mathematical formalization lives there**: rigorous definitions, derivations, axioms, proofs, spectral theory, operator algebra, Hilbert spaces, and theoretical foundations.
>
> **This glossary** contains only **operational quick reference** for developers implementing TNFR networks.

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

---

### Structural Potential (Φ_s)

**Code:** `compute_structural_potential(G, alpha=2.0)` → Dict[NodeId, float]  
**Symbol:** \(\Phi_s(i)\)  
**Formula:** \(\Phi_s(i) = \sum_{j \neq i} \frac{\Delta\text{NFR}_j}{d(i,j)^\alpha}\) where \(\alpha = 2\)  
**What:** Emergent potential field from network ΔNFR distribution  
**Status:** ✅ **CANONICAL** (promoted 2025-11-11)  
**Validation:** 2,400+ experiments, corr(Δ Φ_s, ΔC) = -0.822, CV = 0.1%  
**Physics:** Passive equilibrium landscape (minima = potential wells)  
**Grammar:** U6 STRUCTURAL POTENTIAL CONFINEMENT (Δ Φ_s < 2.0)  
**API:** `tnfr.physics.fields.compute_structural_potential()`  
**Validation:** `tnfr.operators.grammar.validate_structural_potential_confinement()`  
**Math/Physics:** 
- [UNIFIED_GRAMMAR_RULES.md § U6](UNIFIED_GRAMMAR_RULES.md) - Complete derivation
- [TNFR_FORCES_EMERGENCE.md § 14-15](docs/TNFR_FORCES_EMERGENCE.md) - Empirical validation
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

## Invariants (Must Preserve)

From [AGENTS.md](AGENTS.md):

1. **EPI changes only via operators** (no ad-hoc mutations)
2. **Structural units**: νf in Hz_str only
3. **ΔNFR semantics**: not a classic ML gradient
4. **Operator closure**: compositions yield valid states
5. **Phase check**: explicit verification before coupling
6. **Node lifecycle**: birth/collapse conditions maintained
7. **Operational fractality**: EPIs nest without loss of identity
8. **Controlled determinism**: reproducible (seeds + logs)
9. **Structural metrics**: C(t), Si exposed in telemetry
10. **Domain neutrality**: trans-scale, trans-domain

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

**See Also:**
- [UNIFIED_GRAMMAR_RULES.md](UNIFIED_GRAMMAR_RULES.md) - Complete derivations from physics
- [AGENTS.md § Unified Grammar](AGENTS.md#-unified-grammar-u1-u6) - Quick reference
- [docs/grammar/U6_STRUCTURAL_POTENTIAL_CONFINEMENT.md](docs/grammar/U6_STRUCTURAL_POTENTIAL_CONFINEMENT.md) - U6 complete specification
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
- [GRAMMAR_MIGRATION_GUIDE.md](GRAMMAR_MIGRATION_GUIDE.md) - Migration from C1-C3/RC1-RC4 to U1-U6
- [docs/grammar/](docs/grammar/) - Grammar documentation directory (U6, fundamental concepts, etc.)
- [GLYPH_SEQUENCES_GUIDE.md](GLYPH_SEQUENCES_GUIDE.md) - Operator sequence patterns

### Testing & Development
- [TESTING.md](TESTING.md) - Test conventions and invariant verification
- [CONTRIBUTING.md](CONTRIBUTING.md) - Detailed contribution guidelines
- [REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) - Determinism requirements

### See Also Cross-References
This glossary is bidirectionally linked with:
- **AGENTS.md** references this glossary for term definitions
- **UNIFIED_GRAMMAR_RULES.md** references this glossary for quick lookups
- **ARCHITECTURE.md** references this glossary for technical terms
- This glossary references all above documents for complete specifications

---

## Contributing

When adding new functionality:

1. **Verify math**: Check [Mathematical Foundations](docs/source/theory/mathematical_foundations.md)
2. **Preserve invariants**: Follow [AGENTS.md](AGENTS.md) rules
3. **Use canonical terms**: Reference this glossary
4. **Update docs**: If introducing new concepts
5. **Write tests**: Cover invariants (see [TESTING.md](TESTING.md))

For detailed contribution guidelines, see [CONTRIBUTING.md](CONTRIBUTING.md).
