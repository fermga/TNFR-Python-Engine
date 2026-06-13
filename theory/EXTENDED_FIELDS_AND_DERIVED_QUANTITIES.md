# Extended Fields and Derived Quantities

Beyond the core structural field tetrad $(\Phi_s, |\nabla\phi|, K_\phi, \xi_C)$, the nodal equation $\partial\mathrm{EPI}/\partial t = \nu_f \cdot \Delta\mathrm{NFR}(t)$ gives rise to **transport fields** and **derived invariants** that complete the multi-scale characterisation of TNFR network state.

**Status**: CANONICAL — All fields promoted November 2025 after multi-topology validation.

---

## 1. Extended Canonical Fields

Two flux fields complement the core tetrad by adding directed transport dynamics.

### 1.1 Phase Current ($J_\phi$)

$$
J_\phi(i) = \frac{1}{|\mathcal{N}(i)|}\sum_{j \in \mathcal{N}(i)} \sin(\phi_j - \phi_i)
$$

| Property | Value |
|----------|-------|
| **Physics** | Geometric phase confinement drives directed transport |
| **Sign convention** | Positive = net inward flow; negative = net outward flow; zero = equilibrium |
| **Promotion evidence** | 48 samples across WS, BA, Grid topologies; anticorrelation $r(J_\phi, K_\phi) \approx -0.854$ to $-0.997$ (see §2.2); 100% sign consistency |
| **Canonical status** | Promoted November 12, 2025 |

### 1.2 $\Delta$NFR Flux ($J_{\Delta\mathrm{NFR}}$)

$$
J_{\Delta\mathrm{NFR}}(i) = \frac{1}{|\mathcal{N}(i)|}\sum_{j \in \mathcal{N}(i)} \left(\Delta\mathrm{NFR}_j - \Delta\mathrm{NFR}_i\right)
$$

| Property | Value |
|----------|-------|
| **Physics** | Potential-driven reorganisation transport |
| **Sign convention** | Positive = net inward reorganisation pressure; negative = net outward |
| **Canonical status** | Promoted November 12, 2025 |

### 1.3 Research-Level Fields

Additional fields under investigation (not yet canonical):

| Field | Definition | Status |
|-------|-----------|--------|
| Phase strain | Second-order phase gradient tensor | Research |
| Phase vorticity | Curl analogue of phase field | Research |
| Reorganisation strain | Second-order $\Delta$NFR gradient | Research |

**Implementation**: `src/tnfr/physics/extended.py` — `compute_phase_current()`, `compute_dnfr_flux()`.

---

## 2. Complex Geometric Field

### 2.1 Definition

The complex geometric field unifies curvature and transport into a single quantity:

$$
\Psi = K_\phi + i \cdot J_\phi
$$

| Property | Expression |
|----------|-----------|
| Magnitude | $|\Psi| = \sqrt{K_\phi^2 + J_\phi^2}$ |
| Phase | $\arg(\Psi) = \arctan(J_\phi / K_\phi)$ |

### 2.2 Anticorrelation Evidence

Systematic validation reveals near-perfect anticorrelation between the real and imaginary components:

$$
r(K_\phi, J_\phi) \approx -0.854 \text{ to } -0.997
$$

This anticorrelation is a structural consequence: increasing phase curvature (confinement) suppresses phase current (transport) and vice versa.

### 2.3 Physical Interpretation

$\Psi$ unifies two complementary aspects of the geometric sector:
- $\operatorname{Re}(\Psi) = K_\phi$: Static geometric confinement (how much the phase field bends)
- $\operatorname{Im}(\Psi) = J_\phi$: Dynamic transport (how much phase flows through the node)

The unified field enables gauge structure (§ [GAUGE_SYMMETRY_AND_UNIFICATION.md](GAUGE_SYMMETRY_AND_UNIFICATION.md)).

**Implementation**: `src/tnfr/physics/unified.py` — `compute_complex_geometric_field()`.

---

## 3. Derived Invariants

Seven derived fields emerge from the 6D field space $(\Phi_s, |\nabla\phi|, K_\phi, J_\phi, J_{\Delta\mathrm{NFR}}, \xi_C)$.

### 3.1 Chirality ($\chi$)

$$
\chi(i) = |\nabla\phi|(i) \cdot K_\phi(i) - J_\phi(i) \cdot J_{\Delta\mathrm{NFR}}(i)
$$

Detects structural handedness. Nonzero chirality signals broken parity in the network — the field configuration has a preferred rotational direction.

### 3.2 Symmetry Breaking ($\mathcal{S}$)

$$
\mathcal{S}(i) = \left(|\nabla\phi|^2 - K_\phi^2\right) + \left(J_\phi^2 - J_{\Delta\mathrm{NFR}}^2\right)
$$

Order parameter for phase transitions. $\mathcal{S} \approx 0$ indicates balanced sectors; large $|\mathcal{S}|$ signals structural asymmetry between geometric and transport fields.

### 3.3 Coherence Coupling ($\mathcal{C}$)

$$
\mathcal{C}(i) = \Phi_s(i) \cdot |\Psi(i)|
$$

Multi-scale connector that measures how strongly the global potential field $\Phi_s$ couples to the local geometric-transport intensity $|\Psi|$.

### 3.4 Energy Density ($\mathcal{E}$)

$$
\mathcal{E}(i) = \Phi_s^2 + |\nabla\phi|^2 + K_\phi^2 + J_\phi^2 + J_{\Delta\mathrm{NFR}}^2
$$

Total structural energy per node. This is a **gauge-invariant** scalar (see [GAUGE_SYMMETRY_AND_UNIFICATION.md](GAUGE_SYMMETRY_AND_UNIFICATION.md) §1.2).

### 3.5 Action Density ($\mathcal{A}$)

$$
\mathcal{A}(i) = \Phi_s \cdot |\nabla\phi| + K_\phi \cdot J_\phi + |\nabla\phi| \cdot J_{\Delta\mathrm{NFR}}
$$

Mixed-sector coupling density. Connects potential, geometric, and transport contributions through cross-terms.

### 3.6 Topological Charge ($\mathcal{Q}$)

$$
\mathcal{Q}(i) = |\nabla\phi|(i) \cdot J_\phi(i) - K_\phi(i) \cdot J_{\Delta\mathrm{NFR}}(i)
$$

Conserved topological quantity under grammar-compliant evolution. Forms a U(1) doublet $(\mathcal{Q}, \tilde{\mathcal{Q}})$ whose norm $|\mathcal{T}|^2 = \mathcal{Q}^2 + \tilde{\mathcal{Q}}^2$ is gauge-invariant.

### 3.7 Dual Charges

$$
\tilde{\mathcal{Q}}(i) = K_\phi \cdot |\nabla\phi| + J_\phi \cdot J_{\Delta\mathrm{NFR}}
$$

$$
\tilde{\chi}(i) = |\nabla\phi| \cdot J_\phi + K_\phi \cdot J_{\Delta\mathrm{NFR}}
$$

**Implementation**: `src/tnfr/physics/unified.py` — `compute_chirality()`, `compute_symmetry_breaking()`, `compute_coherence_coupling()`, `compute_energy_density()`, `compute_action_density()`, `compute_topological_charge()`.

---

## 4. Energy Decomposition

The Lyapunov energy functional decomposes into kinetic and potential sectors:

$$
E = \frac{1}{2}\sum_i \mathcal{E}(i) = T + V
$$

where:

$$
T = \frac{1}{2}\sum_i \left(J_\phi(i)^2 + J_{\Delta\mathrm{NFR}}(i)^2\right) \qquad\text{(kinetic — transport sector)}
$$

$$
V = \frac{1}{2}\sum_i \left(\Phi_s(i)^2 + |\nabla\phi|(i)^2 + K_\phi(i)^2\right) \qquad\text{(potential — geometric sector)}
$$

Per-node Hamiltonian density: $H(i) = \frac{1}{2}\mathcal{E}(i)$.

### 4.1 Two-Sector Structure

| Sector | Fields | Conjugate pair | Physical role |
|--------|--------|---------------|---------------|
| **Geometric** | $K_\phi$, $J_\phi$ | $(K_\phi, J_\phi)$ via $\Psi$ | Confinement vs transport |
| **Potential** | $\Phi_s$, $J_{\Delta\mathrm{NFR}}$ | $(\Phi_s, J_{\Delta\mathrm{NFR}})$ | Stability vs reorganisation |

The phase gradient $|\nabla\phi|$ mediates between sectors, appearing in the potential energy but driving transport through its gradient.

---

## 5. Structural Signatures

Element-specific structural signatures are constructed from the field tetrad and extended fields:

| Threshold | Value | Derivation |
|-----------|-------|-----------|
| $|\nabla\phi|$ canonical | $\gamma/\pi \approx 0.1837$ | Kuramoto critical coupling |
| $K_\phi$ canonical | $0.9\pi \approx 2.827$ | 90% of wrap-angle maximum |
| Permissive curvature (Au) | $(\varphi + 1)\pi/e \approx 3.025$ | Golden-ratio adjusted |

**Implementation**: `src/tnfr/physics/signatures.py` — `compute_element_signature()`, spectral coherence metrics.

---

## 6. Field Hierarchy Summary

```text
Core Tetrad (CANONICAL)          Extended Transport (CANONICAL)
├── Φ_s  (structural potential)  ├── J_φ       (phase current)
├── |∇φ| (phase gradient)       └── J_ΔNFR    (ΔNFR flux)
├── K_φ  (phase curvature)
└── ξ_C  (coherence length)
                     ↓
            Complex Unification
            Ψ = K_φ + i·J_φ
                     ↓
         Derived Invariants (CANONICAL)
         ├── χ  (chirality)
         ├── S  (symmetry breaking)
         ├── C  (coherence coupling)
         ├── E  (energy density)
         ├── A  (action density)
         └── Q  (topological charge)
```

**Causal chain**: The hierarchy above is *diagnostic*, not dynamical. The
experimentally confirmed causal chain is:
Operator → (ν_f, ΔNFR) → dEPI/dt → Tetrad → (ℰ, Q).
All fields and invariants are uniquely determined by the nodal equation state;
they do not feed back into the dynamics. See [STRUCTURAL_OPERATORS.md
§17.5](STRUCTURAL_OPERATORS.md) and [example 39](../examples/02_physics_regimes/39_nodal_equation_decomposition.py).

---

## Implementation Reference

| Module | Content |
|--------|---------|
| `src/tnfr/physics/extended.py` | $J_\phi$, $J_{\Delta\mathrm{NFR}}$ computation |
| `src/tnfr/physics/unified.py` | $\Psi$, $\chi$, $\mathcal{S}$, $\mathcal{C}$, $\mathcal{E}$, $\mathcal{A}$, $\mathcal{Q}$ |
| `src/tnfr/physics/signatures.py` | Element signatures, spectral coherence |
| `src/tnfr/physics/telemetry.py` | Optimised computation pipeline |

**Tests**: `tests/test_extended_fields.py`, `tests/test_unified_fields.py`

---

## Implementation & Examples

### SDK Entry Points

```python
from tnfr.sdk import TNFR

net = TNFR.create(20).ring().evolve(5)
invariants = net.tensor_invariants()    # energy_density, topological_charge
emergent = net.emergent_fields()        # chirality, symmetry_breaking, coherence_coupling
```

### Executable Demonstrations

| Example | Concept from this document |
|---------|---------------------------|
| [20_eigenmode_tetrad.py](../examples/03_riemann_zeta/20_eigenmode_tetrad.py) | Per-eigenmode structural field tetrad |
| [33_complex_field_unification.py](../examples/02_physics_regimes/33_complex_field_unification.py) | Ψ = K_φ + i·J_φ anticorrelation, emergent fields χ/𝒮/𝒞, energy decomposition |
| [unified_fields_showcase.py](../examples/08_emergent_geometry/unified_fields_showcase.py) | Ψ = K_φ + i·J_φ, emergent fields χ/𝒮/𝒰, tensor invariants |

### Key Source Modules

- `src/tnfr/physics/fields.py` — `compute_tensor_invariants()`, `compute_emergent_fields()`, `compute_complex_geometric_field_arrays()`

---

## Cross-References

- Core tetrad: [FUNDAMENTAL_THEORY.md](FUNDAMENTAL_THEORY.md) §2
- Gauge symmetry of $\Psi$: [GAUGE_SYMMETRY_AND_UNIFICATION.md](GAUGE_SYMMETRY_AND_UNIFICATION.md)
- Conservation laws: [STRUCTURAL_CONSERVATION_THEOREM.md](STRUCTURAL_CONSERVATION_THEOREM.md)
- Variational principle: [TNFR_VARIATIONAL_PRINCIPLE.md](TNFR_VARIATIONAL_PRINCIPLE.md)
- Lyapunov stability: [STRUCTURAL_STABILITY_AND_DYNAMICS.md](STRUCTURAL_STABILITY_AND_DYNAMICS.md)
