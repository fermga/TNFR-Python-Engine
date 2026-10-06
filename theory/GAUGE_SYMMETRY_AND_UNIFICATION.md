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
> $A_{ij}=d(\arg\Psi)_{ij}$ modulo angle wrapping. Its U(1) cycle holonomy is
> the identity analytically, so the
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
| Energy density $\mathcal{E}(i)$ | $\Phi_s^2 + \vert \nabla\phi\vert ^2 + \vert \Psi\vert ^2 + J_{\Delta\mathrm{NFR}}^2$ | The $K_\phi,J_\phi$ quadratic sum is rotation invariant |
| Field magnitude $\vert \Psi(i)\vert ^2$ | $K_\phi^2 + J_\phi^2$ | Euclidean norm identity |
| Coherence $C(t)$ | Depends on the unchanged graph fields | Unchanged because the analysis transform does not mutate the graph |
| Legacy topological norm $\vert \mathcal{T}\vert ^2$ | $\mathcal{Q}^2 + \tilde{\mathcal{Q}}^2$ | Norm of an algebraic doublet |
| Legacy chirality norm $\vert \mathcal{X}\vert ^2$ | $\chi^2 + \tilde{\chi}^2$ | Norm of an algebraic doublet |

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

The norms of the two doublets, $|\mathcal{T}|^2$ and $|\mathcal{X}|^2$, are
invariant because their quadratic sums are preserved under rotation. No such
norm-invariance statement is made for the separate $Q$ or $\mathcal S$ above.

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

Thus the exponentiated links satisfy $e^{iA_{ij}}=e^{i\theta_j}e^{-i\theta_i}$:
they are **pure gauge**, not independent edge variables. The unwrapped
vertex difference is an exact real one-form. The wrapped real edge values
can instead sum to a nonzero integer multiple of $2\pi$ around a cycle.

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
would require independent edge curvature or an additional specified defect
model, which this API does not expose. This does not rule out integer winding
of a supplied nodal phase field: the final wrap discards that integer. The
separate [declared-cycle winding interface](../docs/STRUCTURAL_FIELDS_TETRAD.md#appendix-topological-winding)
retains branch and orientation evidence and is not this gauge-curvature diagnostic.

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
| **strong_like** | numerical $\vert F_C\vert $ | Pure-gauge closure-error slot; zero within tolerance in a valid computation |
| **gravity_like** | $\vert \Phi_s\vert  \gg \vert \Psi\vert $ | Gauge-invariant potential-to-field magnitude comparison |

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
| IL minimizes $\vert D\Psi\vert $ | Unsupported without before/after operator telemetry or a theorem |
| OZ sources $F_C$ | Incompatible with a nonzero analytic final wrapped closure residual; the underlying wrapped-edge sum can be a nonzero integer multiple of $2\pi$ |
| RA transports gauge invariants | Unsupported without an induced-map covariance check |

Grammar rule **U3** constrains the node phase $\phi$ before coupling via
$|\operatorname{wrap}(\phi_i-\phi_j)|\le\Delta\phi_{\max}$. It is not a gauge-fixing rule for this
auxiliary construction. Since $\Psi$ itself is extracted from graph fields,
$\arg\Psi$ has not been established as an independent dynamical degree of
freedom.

---

## 5. Auxiliary Conservation-Gauge Comparison

<a id="51-legacy-auxiliary-action-functional"></a>
<a id="52-scoped-symmetry-statements"></a>
<a id="53-unification-structure"></a>
The [variational owner](TNFR_VARIATIONAL_PRINCIPLE.md#2-the-tnfr-lagrangian)
defines the shared quadratic action/energy readouts and
[harmonic coordinate model](TNFR_VARIATIONAL_PRINCIPLE.md#3-harmonic-model-and-the-unresolved-nodal-correspondence).
Agreement of the five squared-field sums is a snapshot algebraic identity.
Auxiliary Hamiltonian conservation applies to its declared autonomous flow,
not automatically to engine trajectories. The
[conservation owner](STRUCTURAL_CONSERVATION_THEOREM.md#main-result)
defines the measured balance residual, including its normalization and units;
grammar validity does not force that residual to vanish.

<a id="54-grammar-conservation-mapping"></a>
### 5.4 Availability of the grammar diagnostic

`compute_grammar_symmetry_mapping` in
[conservation_gauge_unification.py](../src/tnfr/physics/conservation_gauge_unification.py)
reports the evidence it can assess, rather than deriving conservation from
historical rule labels:

| Rules | Actual assessment boundary |
| --- | --- |
| U1, U2, U4 | Always `not_assessed` by this snapshot API; validation needs ordered operator/trigger/debt history and execution context |
| U5 | Always `not_assessed` here; validation needs declared parent/child hierarchy and normalization |
| U3 | Current-edge phase compatibility requires explicit finite endpoint phases and the shared U3 gate |
| U6 | Mean absolute potential drift requires a declared earlier graph or captured potential field; missing reference remains unavailable |

These observations do not execute a grammar word, prove a Noether law or supply
an all-time storage bound. [Grammar](UNIFIED_GRAMMAR_RULES.md) owns sequence
admission. The different `analyze_grammar_stationarity` snapshot heuristics
retain their [separate variational scope](TNFR_VARIATIONAL_PRINCIPLE.md#4-grammar-labelled-heuristics).

<a id="55-symplectic-structure"></a>
### 5.5 Constant oscillator rotation

`verify_symplectic_gauge_compatibility` checks one constant global `SO(2)`
rotation of the auxiliary `(K_phi,J_phi)` pair. It computes the `2x2` pullback
`R.T @ omega @ R` and determinant residual, with
`transformation_scope="global_constant_oscillator_rotation"` and
`local_gauge_assessed=False`. The legacy product/volume fields are snapshot
statistics, not geometric certificates. This check does not supply local gauge
dynamics or the derivative of any of the 13 engine operators. A proposed
operator map needs the [separate Jacobian-level test](TNFR_VARIATIONAL_PRINCIPLE.md#5-operator-energy-heuristics-and-local-symplectic-checks).

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
- $\lambda_k$ — graph Laplacian eigenvalue; a temporal frequency requires a separate evolution law and clock

A factor $\lambda_k$ appears only after separately representing the divergence
as a Laplacian acting on a declared scalar current potential; it must not be
multiplied into a quantity already defined as the projected divergence.

### 6.2 Physical Interpretation

| Mode regime | Condition | Behaviour |
|-------------|-----------|-----------|
| Small $\lambda_k$ | Slowly varying graph modes | Measure $\hat S_k$; neither a small residual nor a U5 interpretation follows from the eigenvalue alone |
| Large $\lambda_k$ | Rapidly varying graph modes | Measure redistribution and residuals; rapid temporal dissipation requires a specified dissipative evolution |

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

<a id="7-historical-spectral-gauge-conjecture"></a>
## 7. Spectral claims require a separate model

U(1) coordinate rotations and Parseval identities do not select a critical
parameter, an analytic zeta representation or a nodal pulse law. The
[Riemann scope](TNFR_RIEMANN_RESEARCH_NOTES.md) retains the inserted-threshold
counterexample and finite arithmetic comparisons. Those independent models
cannot acquire a dynamical bridge from the gauge diagnostics defined here.

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
