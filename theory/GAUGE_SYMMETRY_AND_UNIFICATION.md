# Gauge Symmetry and Conservation-Gauge Unification

The complex geometric field $\Psi = K_\phi + i \cdot J_\phi$ admits a
nodewise **U(1) field-coordinate action** in the auxiliary model. This document
derives its algebraic invariants and records diagnostic comparisons with
conservation, grammar and spectral projections.

**Status**: HISTORICAL RESEARCH MODEL — algebraic U(1) identities retained; dynamical claims scoped below.

> **Current scope and supersession (2026-09).** The rotation identities,
> gauge-invariant norms, Parseval identity and the exact harmonic substrate flow
> are valid within their stated auxiliary models. They do not derive the full
> nodal equation from the displayed action, make U1–U6 symmetries of that
> action, prove that engine operators are symplectomorphisms, or establish a
> conservation/Lyapunov theorem for arbitrary operator trajectories. The
> implemented connection is the exact one-form
> $A_{ij}=d(\arg\Psi)_{ij}$. Its cycle holonomy is zero analytically, so the
> current implementation has no independent gauge curvature, vortices, flux,
> confinement sector or non-trivial Yang-Mills dynamics. The corresponding
> public names are retained as compatibility diagnostics. The
> structural energy is a non-negative finite-snapshot diagnostic and Lyapunov
> candidate. The separate exact decrease theorem concerns the Dirichlet energy
> of restricted pure-EPI diffusion; see
> [STRUCTURAL_CONSERVATION_THEOREM.md](STRUCTURAL_CONSERVATION_THEOREM.md),
> [TNFR_VARIATIONAL_PRINCIPLE.md](TNFR_VARIATIONAL_PRINCIPLE.md), and
> [TNFR_DIFFUSION_STABILITY_THEOREM.md](TNFR_DIFFUSION_STABILITY_THEOREM.md).

---

## 1. Nodewise U(1) field-coordinate action

### 1.1 Gauge Transformation

The auxiliary transformation rotates the geometric-transport coordinates
$(K_\phi,J_\phi)$ at each node while holding the other snapshot fields fixed:

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

The fields $\Phi_s$, $|\nabla\phi|$, $J_{\Delta\mathrm{NFR}}$, and $\xi_C$ are
singlets by definition of this analysis transform. This transformation is not
an engine operator and is not shown to map one reachable TNFR state or
trajectory to another.

### 1.2 Gauge-Invariant Observables

The following quantities are invariant under this auxiliary rotation:

| Quantity | Definition | Established statement |
|----------|-----------|-----------------------|
| Energy density $\mathcal{E}(i)$ | $\Phi_s^2 + |\nabla\phi|^2 + |\Psi|^2 + J_{\Delta\mathrm{NFR}}^2$ | The $K_\phi,J_\phi$ quadratic sum is rotation invariant |
| Field magnitude $|\Psi(i)|^2$ | $K_\phi^2 + J_\phi^2$ | Euclidean norm identity |
| Coherence $C(t)$ | Depends on the unchanged graph fields | Unchanged because the analysis transform does not mutate the graph |
| Legacy topological norm $|\mathcal{T}|^2$ | $\mathcal{Q}^2 + \tilde{\mathcal{Q}}^2$ | Norm of an algebraic doublet |
| Legacy chirality norm $|\mathcal{X}|^2$ | $\chi^2 + \tilde{\chi}^2$ | Norm of an algebraic doublet |

where:
- $\tilde{\mathcal{Q}} = K_\phi \cdot |\nabla\phi| + J_\phi \cdot J_{\Delta\mathrm{NFR}}$ (dual topological charge)
- $\tilde{\chi} = |\nabla\phi| \cdot J_\phi + K_\phi \cdot J_{\Delta\mathrm{NFR}}$ (dual chirality)

### 1.3 Rotation doublets and variant quantities

The following pairs transform as 2D rotation doublets under $\alpha$:

- $(\mathcal{Q},\;\tilde{\mathcal{Q}})$ — topological charge doublet
- $(\chi,\;\tilde{\chi})$ — chirality doublet

Two further historical quantities are gauge-frame dependent but are not
members of those doublets:

- Legacy quantity named Noether charge $Q = \sum(\Phi_s + K_\phi)$ — not invariant
- Symmetry breaking $\mathcal{S} = (|\nabla\phi|^2 - K_\phi^2) + (J_\phi^2 - J_{\Delta\mathrm{NFR}}^2)$ — NOT invariant

Their **norms** $|\mathcal{T}|^2$ and $|\mathcal{X}|^2$ are invariant because quadratic sums are preserved under rotation.

### 1.4 Derivation Outline

The proof is the Euclidean norm identity for a two-dimensional rotation:

1. **Step 1**: Rotate only $(K_\phi,J_\phi)$ and hold four fields fixed.
2. **Step 2**: Component rotation formulas (§1.1).
3. **Step 3**: Bilinear forms involving one $\Psi$-component and one singlet transform as 2D doublets; their quadratic norms are invariant.

**Implementation**: `src/tnfr/physics/gauge.py` — `verify_gauge_invariance()`, `GaugeInvarianceResult` dataclass.

---

## 2. Exact connection and cycle closure

### 2.1 Gauge Connection

The implementation defines the edge connection from a vertex scalar:

$$
A_{ij} = \arg(\Psi_j) - \arg(\Psi_i) \quad\text{(wrapped to } [-\pi,\pi]\text{)}
$$

Let $\theta_i=\arg\Psi_i$, with the implementation's deterministic zero phase
when $|\Psi_i|$ is numerically zero. For an integer $k_{ij}$,

$$
A_{ij}=\theta_j-\theta_i-2\pi k_{ij}.
$$

Thus $A=d\theta$ modulo the branch convention: it is a **pure-gauge exact
one-form**, not an independent edge variable.

### 2.2 Covariant Derivative

The discrete covariant derivative along edge $(i,j)$:

$$
D_{ij}\Psi = \Psi(j) - e^{iA_{ij}}\,\Psi(i)
$$

At nonzero field endpoints, under gauge transformation
$\Psi \to e^{i\alpha}\Psi$ and reconstruction of the link:

$$
D_{ij}\Psi \;\to\; e^{i\alpha(j)}\,D_{ij}\Psi \quad\text{(covariant)}
$$

Therefore $|D_{ij}\Psi|$ is gauge-invariant. At a zero of $\Psi$, its phase is
undefined and the implementation selects phase zero; the complex covariance
formula then has no intrinsic meaning there, while the magnitude identity
below remains valid.

More strongly, writing $\Psi_i=r_i e^{i\theta_i}$ gives

$$
D_{ij}\Psi=e^{i\theta_j}(r_j-r_i),\qquad
|D_{ij}\Psi|=|r_j-r_i|.
$$

The implemented covariant difference therefore measures endpoint amplitude
contrast after the derived phase alignment. It does not introduce independent
parallel-transport dynamics.

### 2.3 Cycle-closure residual

The legacy-named curvature function sums the exact connection around a cycle:

$$
F_C = \sum_{(i,j) \in C} A_{ij} \quad\text{(wrapped to } [-\pi,\pi]\text{)}
$$

For $C=(v_0,\ldots,v_{m-1},v_0)$ the vertex-phase terms telescope:

$$
F_C
= \sum_{r=0}^{m-1}(\theta_{v_{r+1}}-\theta_{v_r}-2\pi k_r)
= -2\pi\sum_r k_r
\equiv 0 \pmod {2\pi}.
$$

Accordingly, any nonzero returned value is a floating-point wrapping or
accumulation residual. It is not evidence of an independent field strength,
vortex, topological defect, magnetic flux or confinement. Such phenomena
would require independently specified edge degrees of freedom with nonzero
cycle holonomy, which this API does not expose.

### 2.4 Yang-Mills Action

The compatibility diagnostic

$$
S_{\mathrm{YM}} = \frac{1}{2}\sum_C F_C^2
$$

is therefore zero analytically and reports squared numerical closure error in
floating arithmetic. Deriving field equations with a matter current would
require an independently varied connection, a coupled matter action and
boundary conditions; those are not supplied by this implementation. The
``compute_yang_mills_equations`` result is a legacy-named snapshot consistency
evaluation, not an equation of motion for TNFR.

**Implementation**: `src/tnfr/physics/gauge.py` — `compute_gauge_curvature()`, `compute_covariant_derivative_magnitude()`, `compute_yang_mills_action()`.

---

## 3. Historical four-label heuristic

The API retains four historical labels. They summarize selected coordinates;
they are not derived fundamental interactions:

| Legacy label | Input coordinate | Valid scope |
|--------------|------------------|-------------|
| **em_like** | $\arg(\Psi) \approx 0$ | Projection on the selected $K_\phi$ axis; gauge-frame dependent |
| **weak_like** | $\arg(\Psi) \approx \pi/2$ | Projection on the selected $J_\phi$ axis; gauge-frame dependent |
| **strong_like** | numerical $|F_C|$ | Pure-gauge closure-error slot; zero within tolerance in a valid computation |
| **gravity_like** | $|\Phi_s| \gg |\Psi|$ | Gauge-invariant potential-to-field magnitude comparison |

The four scores share a unit reporting budget. The API marks a label active
when its normalized score exceeds the equal-share reference
$1/N_\text{labels}=1/4$. This is a compatibility convention, not a derived
TNFR threshold. In particular, numerical closure noise is suppressed so the
``strong_like`` slot cannot be promoted into a physical claim.

**Implementation**: `src/tnfr/physics/gauge.py` — `classify_interaction_regime()`, `classify_interaction_regime_formal()`, `GaugeSnapshot` dataclass.

---

## 4. Operator and grammar boundary

Earlier versions assigned gauge roles to four canonical operators. Those roles
are not derived by the implementation:

| Historical claim | Current status |
|------------------|----------------|
| UM creates $A_{ij}$ | Unsupported: $A$ is computed after the fact from $\Psi$ on every existing graph edge |
| IL minimizes $|D\Psi|$ | Unsupported without before/after operator telemetry or a theorem |
| OZ sources $F_C$ | Incompatible with the current exact connection, whose cycle sum is always zero |
| RA transports gauge invariants | Unsupported without an induced-map covariance check |

Grammar rule **U3** constrains the node phase $\phi$ before coupling via
$|\phi_i-\phi_j|\le\Delta\phi_{\max}$. It is not a gauge-fixing rule for this
auxiliary construction. Since $\Psi$ itself is extracted from graph fields,
$\arg\Psi$ has not been established as an independent dynamical degree of
freedom.

---

## 5. Auxiliary Conservation-Gauge Comparison

### 5.1 Legacy auxiliary action functional

The historical model introduced the finite field functional

$$
S_{\mathrm{TNFR}} = \sum_n \Delta t \cdot \sum_i \left[\frac{1}{2}(J_\phi^2 + J_{\Delta\mathrm{NFR}}^2) - \frac{1}{2}(\Phi_s^2 + |\nabla\phi|^2 + K_\phi^2)\right]
$$

The same five squared fields also appear in the structural energy diagnostic,
so the corresponding Hamiltonian and energy read-outs agree algebraically
when evaluated on one snapshot. A variational derivation of
$\partial\mathrm{EPI}/\partial t = \nu_f \cdot \Delta\mathrm{NFR}(t)$ from
this functional has not been established: EPI and $\nu_f\Delta\mathrm{NFR}$
are not supplied here as an Euler-Lagrange coordinate/force pair.

### 5.2 Scoped symmetry statements

The auxiliary model and the runtime diagnostics support different statements:

| Structure | What is established | Boundary |
|----------|---------------------|----------|
| Time translation | Conserves the auxiliary Hamiltonian along the exact autonomous auxiliary flow | Does not imply conservation along engine trajectories |
| Internal U(1) | The isotropic $(K_\phi,J_\phi)$ quadratic sector and listed norms are rotation invariant | This is an auxiliary field-space identity |
| Structural balance | $\partial_t\rho+\nabla\cdot\mathbf J=S_{\text{grammar}}$ defines a measurable residual | Grammar validity does not force $S_{\text{grammar}}=0$ |
| Symplectic form | Defines the ambient two-pair phase space; its exact harmonic flow preserves $\omega$ | Each engine operator needs its own Jacobian-level check |

### 5.3 Unification Structure

The historical unification diagram joined four arrows: stationarity to the
nodal equation, time translation to energy conservation, U(1) invariance to
the gauge construction, and U1–U6 to structural continuity. Only the U(1)
algebra and the auxiliary-flow symmetry statements are established as exact
identities here. The first and fourth arrows remain correspondence questions,
and trajectory-level energy change must be measured.

### 5.4 Grammar → Conservation Mapping

The legacy implementation exposes the following diagnostic associations. They
are useful labels for measurements; they are not Noether correspondences or
conservation consequences of sequence validity.

| Grammar rule | Diagnostic association | What must be checked separately |
|--------------|------------------------|---------------------------------|
| **U1** (Initiation/Closure) | Sequence boundary conditions | Finite values on the actual trajectory |
| **U2** (Stabilization/debt) | Energy and coherence trend | The sign of measured $\Delta E$ |
| **U3** (Resonant Coupling) | External phase-admissibility gate | Gauge-link behaviour of the induced map |
| **U4** (Bifurcation) | Trigger/handler and transformer context | Any proposed topological invariant |
| **U5** (Multi-Scale) | Hierarchical coherence checks | Factorisation for a specified nesting model |
| **U6** (Confinement) | Selected drift policy $\Delta\Phi_s<\pi/2$ | It is not a magnitude or potential-energy bound |

### 5.5 Symplectic Structure

The symplectic 2-form on the structural phase space:

$$
\omega = \sum_i \left[dK_\phi(i) \wedge dJ_\phi(i) + d\Phi_s(i) \wedge dJ_{\Delta\mathrm{NFR}}(i)\right]
$$

Two conjugate pairs:
- **Geometric sector**: $(K_\phi, J_\phi)$ — curvature and current
- **Potential sector**: $(\Phi_s, J_{\Delta\mathrm{NFR}})$ — potential and flux

The exact harmonic flow of the auxiliary substrate preserves $\omega$. The 13
engine operators have not been proved to preserve this form; a before/after
snapshot or the fact that an operator is canonical in the TNFR registry is not
a symplecticity certificate.

**Implementation**: `src/tnfr/physics/conservation_gauge_unification.py` —
`ConservationGaugeUnification`, `GrammarSymmetryMapping`, and
`SymplecticGaugeCompatibility` are legacy-named diagnostic dataclasses; their
names do not promote the reported checks to dynamical theorems.

---

## 6. Spectral Balance Diagnostics

### 6.1 Graph Fourier Transform

The measured structural-balance residual can be represented in the spectral
domain using the Graph Fourier Transform (GFT) in the Laplacian eigenbasis
$\{\psi_k\}$.

Given the discrete structural continuity equation:

$$
\frac{\Delta\rho(i)}{\Delta t} + \operatorname{div}\mathbf{J}(i) = S_{\text{grammar}}(i)
$$

Projecting the measured balance onto eigenvectors $\psi_k$:

$$
\frac{d\hat{\rho}_k}{dt} + \widehat{\operatorname{div}\mathbf J}_k = \hat{S}_k
$$

where:
- $\hat{\rho}_k = \langle\psi_k | \rho\rangle$ — charge density in mode $k$
- $\widehat{\operatorname{div}\mathbf J}_k = \langle\psi_k | \operatorname{div}\mathbf{J}\rangle$ — current divergence in mode $k$
- $\hat{S}_k = \langle\psi_k | S\rangle$ — source term in mode $k$
- $\lambda_k$ — Laplacian eigenvalue (mode frequency)

A factor $\lambda_k$ appears only after separately representing the divergence
as a Laplacian acting on a declared scalar current potential; it must not be
multiplied into a quantity already defined as the projected divergence.

### 6.2 Physical Interpretation

| Mode regime | Condition | Behaviour |
|-------------|-----------|-----------|
| Low-frequency (small $\lambda_k$) | Global graph modes | Measure $\hat S_k$; neither small residual nor a U5 interpretation follows from frequency alone |
| High-frequency (large $\lambda_k$) | Local graph modes | Measure redistribution and residuals; rapid dissipation requires a specified dissipative evolution |

### 6.3 Parseval Conservation

Energy in the spatial domain equals energy in the spectral domain:

$$
\|\rho\|^2 = \sum_k |\hat{\rho}_k|^2
$$

For one field and one fixed orthonormal basis this is an exact coordinate
identity. Apparent drift signals numerical error, a changed graph/basis, or a
mismatched normalization; it is not by itself a grammar violation.

### 6.4 Spectral Energy Decomposition

The non-negative structural energy diagnostic decomposes mode-by-mode:

$$
E_k = \frac{1}{2}\left(|\hat{\Phi}_{s,k}|^2 + |\widehat{\nabla\phi}_k|^2 + |\hat{K}_{\phi,k}|^2 + |\hat{J}_{\phi,k}|^2 + |\hat{J}_{\Delta\mathrm{NFR},k}|^2\right)
$$

For two supplied snapshots, the implementation reports finite differences
$\Delta E_k/\Delta t$. Their sign is empirical. U2 grammar compliance alone
does not imply $dE_k/dt\le0$, even for a word containing stabilizers.

**Implementation**: `src/tnfr/physics/spectral_conservation.py` — `SpectralConservationBalance`, `SpectralWardIdentity`, `SpectralLyapunovResult` dataclasses.

---

## 7. Historical Spectral-Gauge Conjecture

The original note proposed that the TNFR-Riemann operator
$H^{(k)}(\sigma) = L_k + V_\sigma$ might have spectral properties constrained
by both:

- **Conservation**: Eigenvalues satisfy sum rules from $E = \text{const}$
- **Gauge**: U(1)-symmetric spectrum at $\sigma = 1/2$ (self-dual point)
- **Together**: Critical parameter $\sigma_c^{(k)} \to 1/2$ as $k \to \infty$

These bullets are a research conjecture, not a consequence of the U(1)
rotation identities or of Parseval's theorem. In particular, neither
$\sigma_c^{(k)}\to1/2$ nor the Riemann critical line follows from the results
in this note. The nodal-pulse implementation can supply numerical evidence for
a declared experiment, but it does not turn the correspondence into a proof.

---

## Implementation Reference

| Module | Content |
|--------|---------|
| `src/tnfr/physics/gauge.py` | Auxiliary U(1) rotations, exact connection, closure residuals and legacy four-label heuristic |
| `src/tnfr/physics/conservation_gauge_unification.py` | Auxiliary action and finite-snapshot compatibility diagnostics |
| `src/tnfr/physics/spectral_conservation.py` | Spectral residual, Parseval and finite-difference energy diagnostics |

**Tests**: [test_gauge.py](../tests/physics/test_gauge.py),
[test_conservation_gauge_unification.py](../tests/physics/test_conservation_gauge_unification.py),
[test_spectral_conservation.py](../tests/physics/test_spectral_conservation.py)

**Example**: `examples/02_physics_regimes/26_gauge_structure_demo.py`

---

## Implementation & Examples

### Executable Demonstrations

| Example | Concept from this document |
|---------|---------------------------|
| [26_gauge_structure_demo.py](../examples/02_physics_regimes/26_gauge_structure_demo.py) | U(1) rotation identities, exact connection and numerical cycle closure |

### Key Source Modules

- `src/tnfr/physics/gauge.py` — Auxiliary U(1) field-coordinate diagnostics
- `src/tnfr/physics/fields.py` — Complex geometric field Ψ = K_φ + i·J_φ

---

## Cross-References

- Nodal equation: [FUNDAMENTAL_THEORY.md](FUNDAMENTAL_THEORY.md) §1
- Conservation laws: [STRUCTURAL_CONSERVATION_THEOREM.md](STRUCTURAL_CONSERVATION_THEOREM.md)
- Variational principle: [TNFR_VARIATIONAL_PRINCIPLE.md](TNFR_VARIATIONAL_PRINCIPLE.md)
- Grammar rules: [UNIFIED_GRAMMAR_RULES.md](UNIFIED_GRAMMAR_RULES.md)
- Extended fields ($J_\phi$, $J_{\Delta\mathrm{NFR}}$): [EXTENDED_FIELDS_AND_DERIVED_QUANTITIES.md](EXTENDED_FIELDS_AND_DERIVED_QUANTITIES.md)
