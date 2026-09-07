# Extended Fields and Derived Quantities

Beyond the core structural field tetrad $(\Phi_s, |\nabla\phi|, K_\phi, \xi_C)$, the nodal equation $\partial\mathrm{EPI}/\partial t = \nu_f \cdot \Delta\mathrm{NFR}(t)$ supports **transport fields** and **derived diagnostic contractions**. They add directional and interaction read-outs to a declared graph state; they do not make the tetrad an injective or complete representation of arbitrary TNFR states.

**Status**: CANONICAL DIAGNOSTIC DEFINITIONS — finite multi-topology validation supports the implemented definitions; completeness and unrestricted conservation claims remain open.

---

## 1. Extended transport diagnostics

Two flux fields complement the core tetrad by adding directed transport dynamics.

### 1.1 Phase Current ($J_\phi$)

$$
J_\phi(i) = \frac{1}{|\mathcal{N}(i)|}\sum_{j \in \mathcal{N}(i)} \sin(\phi_j - \phi_i)
$$

| Property | Value |
|----------|-------|
| **Physics** | Geometric phase confinement drives directed transport |
| **Sign convention** | Positive = net inward flow; negative = net outward flow; zero = equilibrium |
| **Finite evidence** | 48 historical samples across WS, BA and grid topologies reported anticorrelation $r(J_\phi, K_\phi) \approx -0.854$ to $-0.997$ under that protocol (see §2.2) |
| **Engine status** | Canonical diagnostic definition; no universal correlation theorem |

### 1.2 $\Delta$NFR Flux ($J_{\Delta\mathrm{NFR}}$)

$$
J_{\Delta\mathrm{NFR}}(i) = \frac{1}{|\mathcal{N}(i)|}\sum_{j \in \mathcal{N}(i)} \left(\Delta\mathrm{NFR}_j - \Delta\mathrm{NFR}_i\right)
$$

| Property | Value |
|----------|-------|
| **Physics** | Potential-driven reorganisation transport |
| **Sign convention** | Positive = net inward reorganisation pressure; negative = net outward |
| **Engine status** | Canonical diagnostic definition |

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
| Phase | $\arg(\Psi) = \operatorname{atan2}(J_\phi, K_\phi)$ |

### 2.2 Anticorrelation Evidence

The cited finite protocol measured strong anticorrelation between the real and
imaginary components:

$$
r(K_\phi, J_\phi) \approx -0.854 \text{ to } -0.997
$$

This is a protocol-specific relationship, not a universal consequence of the
definitions. Graph topology, phase distribution and weighting can change it.

### 2.3 Physical Interpretation

$\Psi$ unifies two complementary aspects of the geometric sector:
- $\operatorname{Re}(\Psi) = K_\phi$: Static geometric confinement (how much the phase field bends)
- $\operatorname{Im}(\Psi) = J_\phi$: Dynamic transport (how much phase flows through the node)

The pair can be embedded in the auxiliary rotation/gauge comparison described in
[GAUGE_SYMMETRY_AND_UNIFICATION.md](GAUGE_SYMMETRY_AND_UNIFICATION.md); forming a
complex number alone does not derive a gauge law.

**Implementation**: `src/tnfr/physics/unified.py` — `compute_complex_geometric_field()`.

---

## 3. Derived diagnostic contractions

Seven named read-outs are computed from the six-field tuple
$(\Phi_s, |\nabla\phi|, K_\phi, J_\phi, J_{\Delta\mathrm{NFR}}, \xi_C)$.
They are algebraic contractions; names such as “charge” or “action” do not by
themselves establish conservation, topology or a variational principle.

### 3.1 Chirality ($\chi$)

$$
\chi(i) = |\nabla\phi|(i) \cdot K_\phi(i) - J_\phi(i) \cdot J_{\Delta\mathrm{NFR}}(i)
$$

Signed handedness diagnostic in the declared field convention. A nonzero value
does not by itself prove graph-level parity breaking.

### 3.2 Symmetry Breaking ($\mathcal{S}$)

$$
\mathcal{S}(i) = \left(|\nabla\phi|^2 - K_\phi^2\right) + \left(J_\phi^2 - J_{\Delta\mathrm{NFR}}^2\right)
$$

Candidate order parameter used by the finite transition diagnostics.
$\mathcal{S} \approx 0$ means that these squared terms balance; a phase
transition requires an independently specified finite-size protocol.

### 3.3 Coherence Coupling ($\mathcal{C}$)

$$
\mathcal{C}(i) = \Phi_s(i) \cdot |\Psi(i)|
$$

Multi-scale connector that measures how strongly the global potential field $\Phi_s$ couples to the local geometric-transport intensity $|\Psi|$.

### 3.4 Energy Density ($\mathcal{E}$)

$$
\mathcal{E}(i) = \Phi_s^2 + |\nabla\phi|^2 + K_\phi^2 + J_\phi^2 + J_{\Delta\mathrm{NFR}}^2
$$

Nonnegative quadratic energy diagnostic per node. It is invariant under the
specified auxiliary orthogonal rotations; this scoped algebraic fact is not a
general gauge-conservation theorem.

### 3.5 Action Density ($\mathcal{A}$)

$$
\mathcal{A}(i) = \Phi_s \cdot |\nabla\phi| + K_\phi \cdot J_\phi + |\nabla\phi| \cdot J_{\Delta\mathrm{NFR}}
$$

Mixed-sector coupling density. Connects potential, geometric, and transport contributions through cross-terms.

### 3.6 Topological Charge ($\mathcal{Q}$)

$$
\mathcal{Q}(i) = |\nabla\phi|(i) \cdot J_\phi(i) - K_\phi(i) \cdot J_{\Delta\mathrm{NFR}}(i)
$$

The public API retains the historical name “topological charge”. It is a
bilinear diagnostic. Grammar compliance alone does not conserve it, and no
topological quantization theorem is claimed. Together with
$\tilde{\mathcal{Q}}$, its Euclidean norm is invariant only under the declared
auxiliary U(1) rotation.

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

The quadratic energy diagnostic, used as a Lyapunov candidate, decomposes by
definition into transport and static-field sectors:

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
| **Geometric** | $K_\phi$, $J_\phi$ | Auxiliary pair $(K_\phi, J_\phi)$ via $\Psi$ | Curvature vs transport |
| **Potential** | $\Phi_s$, $J_{\Delta\mathrm{NFR}}$ | Auxiliary pair $(\Phi_s, J_{\Delta\mathrm{NFR}})$ | Source aggregation vs reorganisation transport |

The phase gradient $|\nabla\phi|$ mediates between sectors, appearing in the potential energy but driving transport through its gradient.

---

## 5. Graph-field probe signatures

`compute_element_signature()` is a compatibility-named diagnostic over the
supplied graph. It reports fitted $\xi_C$, phase-gradient and phase-curvature
summaries, and the response of $\Phi_s$ to an optional declared probe. The
default probe runs the valid word
`[Emission, Coherence, Silence]` once on a detached graph copy and defines

$$
\Delta_{\mathrm{probe}}\Phi_s
= \max_i |\Phi_s^{\mathrm{after}}(i)-\Phi_s^{\mathrm{before}}(i)|.
$$

The input graph is not evolved. The returned `synthetic_probe_word` and
`synthetic_step_applied` fields identify the intervention. If the probe is
disabled, its zero drift is a sentinel for “not measured,” not evidence of
stability. A resulting `signature_class="stable"` then reflects only the phase
gates because the implementation treats the absent drift check as satisfied.

| Diagnostic policy | Value | Status |
|-------------------|-------|--------|
| Mean $|\nabla\phi|$ gate | $\pi/16 \approx 0.196$ | Selected early-warning threshold; the exact kinematic bound is $\pi$ |
| Maximum $|K_\phi|$ gate | $0.9\pi \approx 2.827$ | Selected margin within the exact wrapped bound $\pi$ |
| Probe $\Phi_s$ drift gate | $\pi/2$ | Selected U6 policy, applied only when the probe runs |
| Fitted $\xi_C$ category | relative to $\sqrt{|V|}$ | Heuristic: localized below $0.3\sqrt{|V|}$, extended above $1.2\sqrt{|V|}$ |
| Legacy Au-like curvature gate | $0.95\pi \approx 2.985$ | Compatibility policy |

These outputs are a response signature for a specified graph and intervention.
They are not spectral-coherence metrics, physical-element signatures, chemical
stability tests, or evidence of autonomous restoring dynamics. The historical
`compute_au_like_signature()` name adds a compatibility boolean from declared
thresholds; it does not establish an Au or metallic correspondence.

**Implementation**: `src/tnfr/physics/signatures.py` —
`compute_element_signature()`, `compute_au_like_signature()`.

---

## 6. Field Hierarchy Summary

```text
Core Tetrad (DIAGNOSTIC)         Extended Transport (DIAGNOSTIC)
├── Φ_s  (structural potential)  ├── J_φ       (phase current)
├── |∇φ| (phase gradient)       └── J_ΔNFR    (ΔNFR flux)
├── K_φ  (phase curvature)
└── ξ_C  (coherence length)
                     ↓
            Complex Unification
            Ψ = K_φ + i·J_φ
                     ↓
         Derived Diagnostic Contractions
         ├── χ  (chirality)
         ├── S  (symmetry breaking)
         ├── C  (coherence coupling)
         ├── E  (energy density)
         ├── A  (action density)
         └── Q  (topological charge)
```

**Dependency chain**: The hierarchy above is *diagnostic*, not a general causal
identification theorem:
Operator → updated nodal/graph state → structural read-outs → derived contractions.
For a fully specified state and graph these functions are deterministic. The
read-outs alone need not reconstruct that state or predict its future. See
[STRUCTURAL_OPERATORS.md
§17.5](STRUCTURAL_OPERATORS.md) and [example 39](../examples/02_physics_regimes/39_nodal_equation_decomposition.py).

---

## Implementation Reference

| Module | Content |
|--------|---------|
| `src/tnfr/physics/extended.py` | $J_\phi$, $J_{\Delta\mathrm{NFR}}$ computation |
| `src/tnfr/physics/unified.py` | $\Psi$, $\chi$, $\mathcal{S}$, $\mathcal{C}$, $\mathcal{E}$, $\mathcal{A}$, $\mathcal{Q}$ |
| `src/tnfr/physics/signatures.py` | Graph-field summaries and declared-probe response signatures |
| `src/tnfr/physics/telemetry.py` | Optimised computation pipeline |

**Tests**: `tests/physics/test_field_readout_consistency.py`,
`tests/physics/test_gauge.py`, `tests/physics/test_readout_reuse.py`

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
