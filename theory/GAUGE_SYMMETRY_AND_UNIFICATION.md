# Gauge Symmetry and Conservation-Gauge Unification

The complex geometric field $\Psi = K_\phi + i \cdot J_\phi$ admits a **local U(1) gauge symmetry** with structural consequences within the TNFR formalism. This document derives the gauge structure, identifies gauge-invariant observables, establishes the conservation-gauge unification, and extends conservation laws to the spectral domain.

**Status**: CANONICAL — Derived from the nodal equation and U(1) representation theory.

---

## 1. Local U(1) Gauge Symmetry

### 1.1 Gauge Transformation

The gauge transformation acts on the geometric-transport sector $(K_\phi, J_\phi)$ while leaving the remaining fields as gauge singlets:

$$
\Psi(i) \;\to\; e^{i\alpha(i)}\,\Psi(i)
$$

In component form:

$$
K_\phi'(i) = K_\phi(i)\cos\alpha(i) - J_\phi(i)\sin\alpha(i)
$$

$$
J_\phi'(i) = K_\phi(i)\sin\alpha(i) + J_\phi(i)\cos\alpha(i)
$$

The fields $\Phi_s$, $|\nabla\phi|$, $J_{\Delta\mathrm{NFR}}$, and $\xi_C$ are **gauge singlets** (invariant under the transformation).

### 1.2 Gauge-Invariant Observables

Five quantities are **exactly gauge-invariant**:

| Quantity | Definition | Physical meaning |
|----------|-----------|------------------|
| Energy density $\mathcal{E}(i)$ | $\Phi_s^2 + |\nabla\phi|^2 + |\Psi|^2 + J_{\Delta\mathrm{NFR}}^2$ | Total structural energy per node |
| Field magnitude $|\Psi(i)|^2$ | $K_\phi^2 + J_\phi^2$ | Geometric-transport intensity |
| Coherence $C(t)$ | Depends on $\Delta\mathrm{NFR}$/phase, not $\arg(\Psi)$ | Network stability |
| Topological norm $|\mathcal{T}|^2$ | $\mathcal{Q}^2 + \tilde{\mathcal{Q}}^2$ | Topological charge magnitude |
| Chirality norm $|\mathcal{X}|^2$ | $\chi^2 + \tilde{\chi}^2$ | Chirality magnitude |

where:
- $\tilde{\mathcal{Q}} = K_\phi \cdot |\nabla\phi| + J_\phi \cdot J_{\Delta\mathrm{NFR}}$ (dual topological charge)
- $\tilde{\chi} = |\nabla\phi| \cdot J_\phi + K_\phi \cdot J_{\Delta\mathrm{NFR}}$ (dual chirality)

### 1.3 U(1) Multiplets (NOT Invariant)

The following transform as 2D rotation doublets under $\alpha$:

- $(\mathcal{Q},\;\tilde{\mathcal{Q}})$ — topological charge doublet
- $(\chi,\;\tilde{\chi})$ — chirality doublet
- Noether charge $Q = \sum(\Phi_s + K_\phi)$ — NOT invariant
- Symmetry breaking $\mathcal{S} = (|\nabla\phi|^2 - K_\phi^2) + (J_\phi^2 - J_{\Delta\mathrm{NFR}}^2)$ — NOT invariant

Their **norms** $|\mathcal{T}|^2$ and $|\mathcal{X}|^2$ are invariant because quadratic sums are preserved under rotation.

### 1.4 Derivation Outline

The proof follows from representation theory of U(1) on the 6D field space $(\Phi_s, |\nabla\phi|, K_\phi, J_\phi, J_{\Delta\mathrm{NFR}}, \xi_C)$:

1. **Step 1**: Gauge transformation acts only on the geometric-transport sector $(K_\phi, J_\phi)$, leaving 4 fields as singlets.
2. **Step 2**: Component rotation formulas (§1.1).
3. **Step 3**: Bilinear forms involving one $\Psi$-component and one singlet transform as 2D doublets; their quadratic norms are invariant.

**Implementation**: `src/tnfr/physics/gauge.py` — `verify_gauge_invariance()`, `GaugeInvarianceResult` dataclass.

---

## 2. Gauge Connection and Curvature

### 2.1 Gauge Connection

The natural gauge connection on edges emerges from the $\Psi$ phase gradient:

$$
A_{ij} = \arg(\Psi_j) - \arg(\Psi_i) \quad\text{(wrapped to } [-\pi,\pi]\text{)}
$$

### 2.2 Covariant Derivative

The discrete covariant derivative along edge $(i,j)$:

$$
D_{ij}\Psi = \Psi(j) - e^{iA_{ij}}\,\Psi(i)
$$

Under gauge transformation $\Psi \to e^{i\alpha}\Psi$:

$$
D_{ij}\Psi \;\to\; e^{i\alpha(j)}\,D_{ij}\Psi \quad\text{(covariant)}
$$

Therefore $|D_{ij}\Psi|$ is gauge-invariant.

### 2.3 Gauge Curvature (Field Strength)

The gauge curvature on a cycle $C$ is the discrete holonomy:

$$
F_C = \sum_{(i,j) \in C} A_{ij} \quad\text{(wrapped to } [-\pi,\pi]\text{)}
$$

Non-zero $F_C$ indicates **gauge vortices** — topological defects analogous to magnetic flux tubes.

### 2.4 Yang-Mills Action

The Yang-Mills action functional:

$$
S_{\mathrm{YM}} = \frac{1}{2}\sum_C F_C^2
$$

The Yang-Mills field equations with matter current follow from $\delta S_{\mathrm{YM}} / \delta A_{ij} = 0$.

**Implementation**: `src/tnfr/physics/gauge.py` — `compute_gauge_curvature()`, `compute_covariant_derivative_magnitude()`, `compute_yang_mills_action()`.

---

## 3. Interaction Regimes

Four interaction regimes emerge from the gauge structure, classified by which sector dominates:

| Regime | Condition | Physical analogy |
|--------|-----------|-----------------|
| **em_like** | $\arg(\Psi) \approx 0$ | Geometric-dominant, weak coupling |
| **weak_like** | $\arg(\Psi) \approx \pi/2$ | Transport-dominant, chiral asymmetry |
| **strong_like** | $|F_C| \gg 0$ | Gauge confinement, strong curvature |
| **gravity_like** | $\Phi_s \gg |\Psi|$ | Potential-dominant, long-range |

**Constants**:
- Regime dominance threshold: $1/\varphi \approx 0.618$
- Strong regime threshold: $\gamma/\pi \approx 0.1837$

**Implementation**: `src/tnfr/physics/gauge.py` — `classify_interaction_regime()`, `GaugeSnapshot` dataclass.

---

## 4. Operator-Gauge Correspondence

Each canonical operator has a specific gauge interpretation:

| Operator | Gauge role |
|----------|-----------|
| **UM** (Coupling) | Creates gauge links — establishes the connection field $A_{ij}$ |
| **IL** (Coherence) | Acts as covariant derivative operator — reduces gauge-variant fluctuations |
| **OZ** (Dissonance) | Sources gauge curvature $F_C$ — creates topological defects |
| **RA** (Resonance) | Propagates gauge-invariant quantities along links |

Grammar rule **U3** constrains the *external* phase $\phi$ via $|\phi_i - \phi_j| \le \Delta\phi_{\max}$, while the *internal* gauge phase $\arg(\Psi)$ provides an independent degree of freedom.

---

## 5. Conservation-Gauge Unification

### 5.1 The TNFR Action Functional

The central theoretical result: grammar rules, symmetries, conservation laws, and gauge structure are four projections of a single mathematical structure — the stationarity of the TNFR action functional:

$$
S_{\mathrm{TNFR}} = \sum_n \Delta t \cdot \sum_i \left[\frac{1}{2}(J_\phi^2 + J_{\Delta\mathrm{NFR}}^2) - \frac{1}{2}(\Phi_s^2 + |\nabla\phi|^2 + K_\phi^2)\right]
$$

This encodes the nodal equation $\partial\mathrm{EPI}/\partial t = \nu_f \cdot \Delta\mathrm{NFR}(t)$ as its Euler-Lagrange equation.

### 5.2 Four Symmetries

Under grammar-compliant evolution (U1–U6), $S_{\mathrm{TNFR}}$ possesses:

| Symmetry | Conservation law | Via |
|----------|-----------------|-----|
| Time-translation | Energy conservation $H = T + V = \text{const}$ | Noether's theorem |
| Internal U(1) | Gauge structure on $\Psi$ | $S[e^{i\alpha}\Psi] = S[\Psi]$ |
| Grammar symmetry | Structural continuity $\partial\rho/\partial t + \nabla\cdot\mathbf{J} = S_{\text{grammar}}$ | Ward identities |
| Symplectic structure | Phase space geometry | Canonical form $\omega$ |

### 5.3 Unification Structure

$$
S_{\mathrm{TNFR}} \begin{cases} \delta S/\delta\Phi = 0 &\to \text{Euler-Lagrange} = \text{Nodal equation} \\ \partial S/\partial t = 0 &\to \text{Noether} \to \text{Energy conservation} \\ S[e^{i\alpha}\Psi] = S[\Psi] &\to \text{U(1) gauge structure} \\ \text{U1–U6} \subset \mathrm{Aut}(S) &\to \text{Structural continuity } (\rho, \mathbf{J}) \end{cases}
$$

### 5.4 Grammar → Conservation Mapping

Each grammar rule protects a specific conservation law:

| Grammar rule | Symmetry protected | Conservation consequence |
|--------------|--------------------|------------------------|
| **U1** (Initiation/Closure) | Boundary conditions | Energy finiteness (action endpoints) |
| **U2** (Convergence) | Lyapunov stability | $dH/dt \le 0$ (stability) |
| **U3** (Resonant Coupling) | Gauge connection | $A_{ij}$ regularity (smooth connection) |
| **U4** (Bifurcation) | Topological charge | Quantisation of $\mathcal{Q}$ |
| **U5** (Multi-Scale) | Hierarchical action | Action factorisation across scales |
| **U6** (Confinement) | Potential energy | $V < \frac{1}{2}\varphi^2 \cdot N$ (boundedness) |

### 5.5 Symplectic Structure

The symplectic 2-form on the structural phase space:

$$
\omega = \sum_i \left[dK_\phi(i) \wedge dJ_\phi(i) + d\Phi_s(i) \wedge dJ_{\Delta\mathrm{NFR}}(i)\right]
$$

Two conjugate pairs:
- **Geometric sector**: $(K_\phi, J_\phi)$ — curvature and current
- **Potential sector**: $(\Phi_s, J_{\Delta\mathrm{NFR}})$ — potential and flux

Canonical operators preserve $\omega$ (Liouville theorem analogue).

**Implementation**: `src/tnfr/physics/conservation_gauge_unification.py` — `ConservationGaugeUnification`, `GrammarSymmetryMapping`, `SymplecticGaugeCompatibility` dataclasses.

---

## 6. Spectral Conservation

### 6.1 Graph Fourier Transform

The structural conservation equations are lifted to the spectral domain using the Graph Fourier Transform (GFT) in the Laplacian eigenbasis $\{\psi_k\}$.

Given the discrete structural continuity equation:

$$
\frac{\Delta\rho(i)}{\Delta t} + \operatorname{div}\mathbf{J}(i) = S_{\text{grammar}}(i)
$$

Projecting onto eigenvectors $\psi_k$:

$$
\frac{d\hat{\rho}_k}{dt} + \lambda_k \cdot \hat{J}_k = \hat{S}_k
$$

where:
- $\hat{\rho}_k = \langle\psi_k | \rho\rangle$ — charge density in mode $k$
- $\hat{J}_k = \langle\psi_k | \operatorname{div}\mathbf{J}\rangle$ — current divergence in mode $k$
- $\hat{S}_k = \langle\psi_k | S\rangle$ — source term in mode $k$
- $\lambda_k$ — Laplacian eigenvalue (mode frequency)

### 6.2 Physical Interpretation

| Mode regime | Condition | Behaviour |
|-------------|-----------|-----------|
| Low-frequency (small $\lambda_k$) | Global coherence modes | Near-exact conservation ($\hat{S}_k \approx 0$); corresponds to U5 multi-scale structure |
| High-frequency (large $\lambda_k$) | Local fluctuation modes | Rapid equilibration via $\lambda_k \cdot \hat{J}_k$ dissipation; grammar violations manifest here |

### 6.3 Parseval Conservation

Energy in the spatial domain equals energy in the spectral domain:

$$
\|\rho\|^2 = \sum_k |\hat{\rho}_k|^2
$$

Drift in this identity signals numerical or structural inconsistency.

### 6.4 Spectral Energy Decomposition

The Lyapunov energy decomposes mode-by-mode:

$$
E_k = \frac{1}{2}\left(|\hat{\Phi}_{s,k}|^2 + |\widehat{\nabla\phi}_k|^2 + |\hat{K}_{\phi,k}|^2 + |\hat{J}_{\phi,k}|^2 + |\hat{J}_{\Delta\mathrm{NFR},k}|^2\right)
$$

Under grammar compliance (U2): $dE_k/dt \le 0$ for stabiliser-dominated modes.

**Implementation**: `src/tnfr/physics/spectral_conservation.py` — `SpectralConservationBalance`, `SpectralWardIdentity`, `SpectralLyapunovResult` dataclasses.

---

## 7. Spectral-Gauge Consequence

The gauge-conservation unification implies that the TNFR-Riemann operator $H^{(k)}(\sigma) = L_k + V_\sigma$ has spectral properties constrained by both:

- **Conservation**: Eigenvalues satisfy sum rules from $E = \text{const}$
- **Gauge**: U(1)-symmetric spectrum at $\sigma = 1/2$ (self-dual point)
- **Together**: Critical parameter $\sigma_c^{(k)} \to 1/2$ as $k \to \infty$

This provides the structural basis for the convergence proved in `src/tnfr/riemann/convergence_proof.py`.

---

## Implementation Reference

| Module | Content |
|--------|---------|
| `src/tnfr/physics/gauge.py` | U(1) gauge symmetry, connection, curvature, Yang-Mills, interaction regimes |
| `src/tnfr/physics/conservation_gauge_unification.py` | Action functional, four symmetries, grammar mapping, symplectic structure |
| `src/tnfr/physics/spectral_conservation.py` | Spectral continuity, Parseval identity, per-mode Lyapunov |

**Tests**: `tests/test_gauge.py`, `tests/test_conservation_gauge_unification.py`, `tests/test_spectral_conservation.py`

**Example**: `examples/02_physics_regimes/26_gauge_structure_demo.py`

---

## Implementation & Examples

### Executable Demonstrations

| Example | Concept from this document |
|---------|---------------------------|
| [26_gauge_structure_demo.py](../examples/02_physics_regimes/26_gauge_structure_demo.py) | U(1) gauge symmetry of Ψ, connections, curvature, Wilson loops |

### Key Source Modules

- `src/tnfr/physics/gauge.py` — Gauge field structure and symmetries
- `src/tnfr/physics/fields.py` — Complex geometric field Ψ = K_φ + i·J_φ

---

## Cross-References

- Nodal equation: [FUNDAMENTAL_THEORY.md](FUNDAMENTAL_THEORY.md) §1
- Conservation laws: [STRUCTURAL_CONSERVATION_THEOREM.md](STRUCTURAL_CONSERVATION_THEOREM.md)
- Variational principle: [TNFR_VARIATIONAL_PRINCIPLE.md](TNFR_VARIATIONAL_PRINCIPLE.md)
- Grammar rules: [UNIFIED_GRAMMAR_RULES.md](UNIFIED_GRAMMAR_RULES.md)
- Extended fields ($J_\phi$, $J_{\Delta\mathrm{NFR}}$): [EXTENDED_FIELDS_AND_DERIVED_QUANTITIES.md](EXTENDED_FIELDS_AND_DERIVED_QUANTITIES.md)
