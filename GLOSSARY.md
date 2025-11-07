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

13 canonical operators that modify EPI through resonant interactions. Each operator has preconditions and postconditions defined in the grammar.

| Symbol | Name | Effect | Usage |
|--------|------|--------|-------|
| AL | Emission | Initiate pattern | Start trajectories |
| EN | Reception | Integrate external | Network listening |
| IL | Coherence | Stabilize form | Consolidation |
| OZ | Dissonance | Controlled instability | Exploration |
| UM | Coupling | Create links | Network formation |
| RA | Resonance | Amplify/propagate | Pattern reinforcement |
| SHA | Silence | Freeze evolution | Observation windows |
| VAL | Expansion | Increase complexity | Add degrees of freedom |
| NUL | Contraction | Reduce complexity | Simplification |
| THOL | Self-organization | Emergent structure | Fractalization |
| ZHIR | Mutation | Phase transformation | State changes |
| NAV | Transition | Controlled movement | Trajectory navigation |
| REMESH | Recursivity | Nested operations | Multi-scale ops |

**API:** `tnfr.structural.<OperatorName>()`, `run_sequence(G, node, ops)`  
**Grammar:** See [CANONICAL_BOUNDARY_PATTERN.md](CANONICAL_BOUNDARY_PATTERN.md)  
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

## Related Documentation

### Mathematical Theory
- **[Mathematical Foundations](docs/source/theory/mathematical_foundations.md)** ⭐ **SINGLE SOURCE FOR ALL MATH**
- [TNFR.pdf](TNFR.pdf) - Original theoretical companion

### Implementation
- [AGENTS.md](AGENTS.md) - AI agent guidelines and invariants
- [Foundations](docs/source/foundations.md) - Runtime/API guide
- [API Overview](docs/source/api/overview.md) - Package architecture
- [Structural Operators](docs/source/api/operators.md) - Operator details
- [Examples](docs/source/examples/README.md) - Runnable scenarios

### Canonical Patterns
- [CANONICAL_BOUNDARY_PATTERN.md](CANONICAL_BOUNDARY_PATTERN.md) - Operator grammar
- [CANONICITY_VERIFICATION.md](CANONICITY_VERIFICATION.md) - Validation rules
- [TESTING.md](TESTING.md) - Test conventions

---

## Contributing

When adding new functionality:

1. **Verify math**: Check [Mathematical Foundations](docs/source/theory/mathematical_foundations.md)
2. **Preserve invariants**: Follow [AGENTS.md](AGENTS.md) rules
3. **Use canonical terms**: Reference this glossary
4. **Update docs**: If introducing new concepts
5. **Write tests**: Cover invariants (see [TESTING.md](TESTING.md))

For detailed contribution guidelines, see [CONTRIBUTING.md](CONTRIBUTING.md).
