# TNFR Structural Conservation Diagnostics

## Noether-Like Laws from the Nodal Equation

**Status**: CANONICAL DIAGNOSTIC REFERENCE — algebraic balance plus measured residuals
**Date**: March 2026
**Version**: 0.0.3.5
**Prerequisite**: [AGENTS.md](../AGENTS.md) §Foundational Physics, [UNIFIED_GRAMMAR_RULES.md](UNIFIED_GRAMMAR_RULES.md) §U2, §U6

---

## Table of Contents

1. [Scope and Motivation](#1-scope-and-motivation)
2. [Governing Dynamics Recap](#2-governing-dynamics-recap)
3. [Structural Charge and Current Definitions](#3-structural-charge-and-current-definitions)
4. [Construction of the Balance Equation](#4-derivation-of-the-continuity-equation)
5. [Two-Sector Decomposition](#5-two-sector-decomposition)
6. [Noether-Like Grammar/Conservation Correspondences](#6-noether-correspondence-grammar-conservation)
7. [Ward-Like Diagnostics for Operator Sequences](#7-ward-identities-for-operator-sequences)
8. [Lyapunov Candidate and Restricted Stability](#8-lyapunov-stability-from-the-energy-functional)
9. [Discrete Formulation on Graphs](#9-discrete-formulation-on-graphs)
10. [Numerical Validation](#10-numerical-validation)
11. [Physical Interpretation and Analogies](#11-physical-interpretation-and-analogies)
12. [Applications](#12-applications)
13. [Implementation Reference](#13-implementation-reference)
14. [Summary of Main Results](#14-summary-of-main-results)

---

## 1. Scope and Motivation

Every physical theory with continuous symmetries possesses conservation laws
(Noether, 1918). TNFR, however, operates on *discrete* graphs with *discrete*
operator sequences constrained by grammar rules U1–U6. The question is:

> **Do the grammar constraints play the role of continuous symmetries and
> generate conservation laws?**

The implemented answer is deliberately narrower. The balance below defines a
measurable source term, while grammar validation supplies separate constraints
on operator histories. A valid history does not by itself prove a vanishing
source, conservation of the tetrad charge, or monotonicity of the structural
energy candidate.

### Main Result

**Structural balance identity**: for two snapshots of a declared trajectory,
define

$$\frac{\partial \rho}{\partial t} + \nabla \cdot \mathbf{J} = \mathcal{S}_{\text{grammar}}$$

where $\rho = \Phi_s + K_\phi$ is the **structural charge density**,
$\mathbf{J} = (J_\phi, J_{\Delta\text{NFR}})$ is the **structural current**,
and measure $\mathcal{S}$ as the residual. The equation is exact by this
definition of $\mathcal{S}$; its smallness must be demonstrated on the actual
trajectory. The legacy name $\mathcal{S}_{\text{grammar}}$ records the intended
diagnostic use and is not an implication from U1–U6.

---

## 2. Governing Dynamics Recap

### 2.1 Nodal Equation

Every node $i$ in a TNFR network evolves according to:

$$\frac{\partial \text{EPI}_i}{\partial t} = \nu_{f,i} \cdot \Delta\text{NFR}_i(t) \quad \text{(NE)}$$

where:
- $\text{EPI}_i$ is the Primary Information Structure at node $i$
- $\nu_{f,i} \in \mathbb{R}^+$ is the structural frequency (Hz_str)
- $\Delta\text{NFR}_i(t)$ is the nodal reorganization gradient

### 2.2 Phase Dynamics

Phase evolves through coupling with the nodal equation:

$$\frac{\partial \phi_i}{\partial t} = \nu_{f,i} \cdot h(\Delta\text{NFR}_i, \phi_i, \{\phi_j\}_{j \in \mathcal{N}(i)})$$

where $h$ is the phase coupling function determined by the operator sequence.
For Coupling (UM) and Resonance (RA) operators, $h$ drives synchronization:
$h \to \sin(\phi_j - \phi_i)$ (Kuramoto-type).

### 2.3 Grammar Constraints

The evolution is restricted to operator sequences satisfying U1–U6:

- **U2** (stabilization/debt policy): destabilizers require stabilizers within
  the configured finite-history policy
- **U3** (Coupling): $|\phi_i - \phi_j| \leq \Delta\phi_{\max}$ for interactions
- **U6** (Confinement): monitor $\Delta\Phi_s < \pi/2 \approx 1.571$ relative
  to a declared reference

These are execution and telemetry contracts. They do not define a proved
smooth manifold or a general convergence theorem for arbitrary pressure laws.

---

## 3. Structural Charge and Current Definitions

### 3.1 Structural Charge Density

$$\rho(i, t) = \Phi_s(i, t) + K_\phi(i, t)$$

where:

**Structural Potential** (global, ΔNFR-driven):
$$\Phi_s(i) = \sum_{j \neq i} \frac{\Delta\text{NFR}_j}{d(i,j)^\alpha}, \quad \alpha = 2$$

**Phase Curvature** (local, phase-driven):
$$K_\phi(i) = \text{wrap}\!\left(\phi_i - \text{circmean}_{j \in \mathcal{N}(i)} \phi_j\right)$$

The charge $\rho$ couples the *global potential landscape* with the *local
geometric curvature*. This is the natural conserved quantity because:

1. $\Phi_s$ aggregates the reorganization pressure field (information about
   the entire network projected onto node $i$)
2. $K_\phi$ captures the local geometric mismatch (how much node $i$ deviates
   from its neighborhood's mean phase)
3. Their sum represents the total *structural stress* at node $i$

### 3.2 Structural Current Vector

$$\mathbf{J}(i, t) = \big(J_\phi(i, t),\; J_{\Delta\text{NFR}}(i, t)\big)$$

where:

**Phase Current** (transport of phase coherence):
$$J_\phi(i) = \frac{1}{|\mathcal{N}(i)|} \sum_{j \in \mathcal{N}(i)} \sin(\phi_j - \phi_i)$$

**Reorganization Flux** (transport of structural pressure):
$$J_{\Delta\text{NFR}}(i) = \frac{1}{|\mathcal{N}(i)|} \sum_{j \in \mathcal{N}(i)} \big(\Delta\text{NFR}_j - \Delta\text{NFR}_i\big)$$

The current $\mathbf{J}$ carries two types of structural information:

- $J_\phi$ transports *phase coherence* — how synchronization flows through
  the network
- $J_{\Delta\text{NFR}}$ transports *reorganization pressure* — how structural
  stress redistributes

### 3.3 Why This Pairing?

The charge–current pairing $(\rho, \mathbf{J})$ is not arbitrary. It arises
from the **sector structure** of TNFR fields:

| Sector | Charge Component | Current Component | Driving Physics |
|--------|-----------------|-------------------|-----------------|
| **Potential** | $\Phi_s$ | $J_{\Delta\text{NFR}}$ | ΔNFR distribution & redistribution |
| **Geometric** | $K_\phi$ | $J_\phi$ | Phase dynamics & curvature transport |

These sectors are coupled through the complex geometric field
$\Psi = K_\phi + i J_\phi$, discovered via the
$r(K_\phi, J_\phi) \approx -0.997$ anticorrelation (see
[AGENTS.md](../AGENTS.md) §Mathematical Unification Discoveries).

---

## 4. Construction of the Balance Equation {#4-derivation-of-the-continuity-equation}

### 4.1 Time Derivative of Structural Potential

$$\frac{\partial \Phi_s(i)}{\partial t} = \sum_{j \neq i} \frac{1}{d(i,j)^\alpha} \frac{\partial \Delta\text{NFR}_j}{\partial t}$$

By the nodal equation, $\Delta\text{NFR}_j$ changes through operator
applications. If a specific model supplies an integrable pressure derivative,
then

$$\sum_{j \neq i} \frac{|\partial \Delta\text{NFR}_j / \partial t|}{d(i,j)^\alpha} < \infty$$

the fixed-kernel potential derivative is bounded. U2 sequence validity alone
does not supply this analytic hypothesis; topology changes add a kernel-change
term and must be handled separately.

### 4.2 Time Derivative of Phase Curvature

$$\frac{\partial K_\phi(i)}{\partial t} = \frac{\partial \phi_i}{\partial t} - \sum_{j \in \mathcal{N}(i)} w_j \frac{\partial \phi_j}{\partial t}$$

where $w_j$ are the circular mean weights. U3 checks the wrapped phase
difference before a coupling/resonance action. A phase-velocity bound requires
an additional, specified phase-evolution law; it does not follow from U3 alone.

For a bounded Kuramoto comparison model one may derive a model-specific
$C_{\text{phase}}$; that auxiliary result is not a universal grammar bound.

### 4.3 Discrete Divergence of Current

The graph divergence at node $i$:

$$(\nabla \cdot \mathbf{J})(i) = \frac{1}{|\mathcal{N}(i)|} \sum_{j \in \mathcal{N}(i)} \Big[\big(J_\phi(j) - J_\phi(i)\big) + \big(J_{\Delta\text{NFR}}(j) - J_{\Delta\text{NFR}}(i)\big)\Big]$$

### 4.4 The Balance Equation

Combining §4.1–4.3, the rate of change of charge is:

$$\frac{\Delta \rho(i)}{\Delta t} = \frac{\Delta \Phi_s(i)}{\Delta t} + \frac{\Delta K_\phi(i)}{\Delta t}$$

The **structural source term** is:

$$\mathcal{S}(i) = \frac{\Delta \rho(i)}{\Delta t} + (\nabla \cdot \mathbf{J})(i)$$

### 4.5 Measured conservation residual and conditional bounds

For a finite graph and two recorded snapshots, the implementation computes
$\mathcal S$ directly. A decay estimate such as

$$\|\mathcal{S}\|_{\ell^2} \;\leq\; \frac{C_{\text{net}}}{\sqrt{N}}$$

is a **hypothesis requiring a specified graph sequence and uniform analytical
bounds**. It is not established by U1–U6, by one finite-size fit, or by the
identity defining $\mathcal S$. No continuum-limit theorem is claimed.

*Historical heuristic and numerical scope.*

The following steps document the heuristic that motivated the diagnostics.
They do not establish the displayed asymptotic bound. The residual must be
reported with the actual topology, pressure law, discretization, and finite
trajectory; topology changes and infinite-horizon behavior require separate
analysis.

**Step 1. Operational multipliers do not prove a trajectory bound.**

The implementation assigns nominal pressure-role multipliers to IL, OZ and
other operators. For example, the configured OZ–IL product can exceed one.
U2 requires stabilizer coverage and limits uncompensated debt; it does not
prove absolute integrability of $\nu_f\Delta\mathrm{NFR}$ for every pressure
law or every valid word.

If a particular model independently establishes

$$M_{\mathrm{rate}}=
\sup_t\sum_j|\partial_t\Delta\mathrm{NFR}_j|<\infty,$$

then a fixed potential kernel $B_G$ gives the conditional estimate

$$\|\partial_t\Phi_s\|_\infty
\leq\|B_G\|_\infty M_{\mathrm{rate}}.$$

U6 monitors potential drift; it does not invert a signed aggregation into a
bound on absolute pressure. A topology change additionally contributes the
kernel-change term documented by `structural_potential_change_terms`.

**Step 2. Phase-rate bounds require a phase model.**

U3 checks the wrapped phase difference before Coupling or Resonance. It does
not by itself bound $\dot\phi$. For an explicitly selected Kuramoto law with
bounded capacity, $|\sin(\phi_j-\phi_i)|\leq1$ supplies a model-specific
phase-rate estimate. Such an estimate is not a theorem about every canonical
operator trajectory.

**Step 3. The residual is observed, not cancelled by definition.**

$\Phi_s$ uses a non-local inverse-distance kernel, whereas
$J_{\Delta\mathrm{NFR}}$ and $J_\phi$ use local neighbour differences. Their
finite-difference balance need not vanish. The implementation therefore
records potential and geometric residuals separately instead of asserting an
exact continuum correspondence.

**Step 4. Linearization and scaling are conditional.**

The approximation $\sin(\Delta\phi)\approx\Delta\phi$ requires a measured
small-branch margin. Likewise, diameter or density scaling depends on a
specified graph family. U3 and U6 alone imply neither condition.

**Step 5. Finite-size evidence remains measured.**

The historical experiments fitted

$$q(N)\sim1-\frac{C}{\sqrt N}$$

with $C\approx2.1$ over their sampled topologies. This is a finite empirical
fit, not a continuum-limit theorem or a graph-uniform bound.

**Step 6. Residuals are alerts, not validators.**

Large potential or geometric residuals can accompany U2, U3 or U6 risks, but
they may also arise from discretization, topology changes or an external
pressure law. The actual operator history must be validated independently.
`compute_grammar_conservation_bounds` therefore exposes compatibility alert
levels, not admitted analytical bounds on every valid trajectory.

---

## 5. Two-Sector Decomposition

The full conservation law decomposes into two coupled sub-equations:

### 5.1 Potential Sector

$$\frac{\partial \Phi_s}{\partial t} + \nabla \cdot J_{\Delta\text{NFR}} = \mathcal{S}_{\text{pot}}$$

- **Physics**: Global ΔNFR landscape evolves via operator-driven redistribution
- **Grammar context**: U2 tracks stabilizer debt; U6 supplies a potential-drift
  alert. Neither bounds $\mathcal{S}_{\text{pot}}$ by itself.
- **Monitoring**: Track via $|\Delta\Phi_s| < \pi/2$ per U6

### 5.2 Geometric Sector

$$\frac{\partial K_\phi}{\partial t} + \nabla \cdot J_\phi = \mathcal{S}_{\text{geo}}$$

- **Physics**: Local phase curvature evolves via synchronization dynamics
- **Grammar context**: U3 gates coupling and U4 checks bifurcation context;
  neither proves a bound on curvature transport by itself.
- **Monitoring**: Track via $|K_\phi| < 2.8274$ (hotspot threshold)

### 5.3 Cross-Sector Coupling

The two sectors are not independent. The **coupling strength** is:

$$\kappa = \text{corr}\!\left(\mathcal{S}_{\text{pot}}, \mathcal{S}_{\text{geo}}\right)$$

The recorded finite experiments found $\kappa \approx 0.6$–$0.7$. This is a
correlation between residual channels in that protocol, not evidence of a
causal mechanism or a proof of the algebraic complex-field construction
$\Psi = K_\phi + i J_\phi$.

---

## 6. Noether-Like Grammar/Conservation Correspondences {#6-noether-correspondence-grammar-conservation}

### 6.1 The Correspondence Table

> **Note**: Every row below is an interpretive/diagnostic correspondence. The
> harmonic substrate has genuine continuous symmetries and Noether charges;
> the discrete grammar rules alone do not generate these conservation laws.

| Grammar Rule | Enforced contract | Associated diagnostic analogy |
|-------------|-------------------|-------------------------------|
| **U1** | generator/closure history | source/closure balance |
| **U2** | stabilizer coverage and bounded debt | energy/coherence trend |
| **U3** | wrapped phase compatibility | phase-current admissibility |
| **U4** | bifurcation context and recency | curvature-transition context |
| **U5** | nested identity and per-level stabilization | hierarchical charge readout |
| **U6** | potential drift telemetry | confinement alert |

### 6.2 Diagnostic hierarchy

The reported quantities form an evidence hierarchy:

1. **Exact model identities:** for example, substrate charges along the exact
  harmonic flow or the degree-weighted EPI total in its fixed symmetric
  diffusion regime.
2. **Finite-trajectory measurements:** charge drift and continuity residuals
  computed from recorded snapshots.
3. **Statistical summaries:** residual distributions over declared operators,
  graph families and seeds.

### 6.3 Symmetry Breaking

Grammar violations may correlate with characteristic residual patterns:

- **U2 violation** (uncompensated destabilizer): inspect pressure, C(t), and
  energy trends; unbounded growth is not inferred from the label alone.
- **U3 violation** (coupling without a phase check): the action is inadmissible;
  any current residual is supporting telemetry only.
- **U6 alert** (potential drift exceeds $\pi/2$): report the kernel, reference,
  norm, and finite observation window.

This provides a heuristic *diagnostic tool*. It does not replace the canonical
grammar validator or uniquely infer which rule was broken.

---

## 7. Ward-Like Diagnostics for Operator Sequences {#7-ward-identities-for-operator-sequences}

### 7.1 Definition

A **Ward identity** constrains the expectation value of observables
between operator applications. For a TNFR operator $\mathcal{O}_k$ applied
at step $k$:

$$\langle \Delta \rho \rangle_k + \langle \nabla \cdot \mathbf{J} \rangle_k = \langle \mathcal{S}_k \rangle$$

where $\langle \cdot \rangle_k$ denotes the network average at step $k$.

### 7.2 Operator-specific measurements

The contract source fixes each operator's primary channel and U2 role; it does
not fix a universal sign for $\Delta\rho$ or $\Delta E$:

| Operators | Primary channel | U2 role | $\Delta\rho$, $\Delta E$ |
|-----------|-----------------|---------|---------------------------|
| AL, EN, RA, REMESH | EPI form | neutral | measured per trajectory |
| SHA, NUL | capacity (NUL also pressure) | neutral | measured per trajectory |
| VAL | capacity | destabilizer | measured per trajectory |
| UM | phase | neutral | measured per trajectory; U3-gated |
| ZHIR | phase | destabilizer | measured per trajectory; U4 context |
| IL, THOL | pressure | stabilizer | measured per trajectory |
| OZ | pressure | destabilizer | measured per trajectory |
| NAV | pressure | neutral | measured per trajectory |

### 7.3 Sequence Ward residual

For a recorded sequence $\sigma = [\mathcal{O}_1, \ldots, \mathcal{O}_N]$,
the diagnostic aggregates

$$\sum_{k=1}^{N} \langle \mathcal{S}_k \rangle \approx 0$$

Whether this value is small is measured. U1 closure and U2 debt balance do not
prove cancellation of the independently defined tetrad residual.

**Experimental note (causal readout chain)**: The operator-specific measurements
in §7.2 are consistent with the recorded chain:
Operator → (ν_f, ΔNFR) → dEPI/dt → Tetrad → (ℰ, Q). Each operator
produced a tetrad fingerprint in the stated fixtures (see [STRUCTURAL_OPERATORS.md
§17.2](STRUCTURAL_OPERATORS.md) and [example 37](../examples/02_physics_regimes/37_operator_tetrad_synergy.py)).
The IL-OZ symmetry (ΔE = −0.011 for both, despite opposite physics)
shows why a fixed sign cannot be assigned from perturbation magnitude alone.

---

## 8. Lyapunov Candidate and Restricted Stability {#8-lyapunov-stability-from-the-energy-functional}

### 8.1 Energy Functional

Define the **structural energy functional** from the five canonical fields:

$$E[G] = \frac{1}{2} \sum_{i \in V} \left[\Phi_s(i)^2 + |\nabla\phi|(i)^2 + K_\phi(i)^2 + J_\phi(i)^2 + J_{\Delta\text{NFR}}(i)^2\right]$$

This is the half-sum of the **energy density invariant** $\mathcal{E}$ defined
in [AGENTS.md](../AGENTS.md) §Tensor Invariants.  All five tetrad fields
contribute — omitting $|\nabla\phi|^2$ would break the Noether correspondence
because phase gradient stress is the local driver of K_φ transport.

$E \geq 0$ always (sum of squares).

### 8.2 Lyapunov candidate for observed evolution

For a trajectory whose measured energy is non-increasing,

$$\frac{dE}{dt} \leq 0$$

The following is a motivation, not a proof:

1. Coherence (IL) has a pressure-reducing contract, but cancellation in
  $\Phi_s$ means this does not fix the sign of every energy term.
2. Self-organization (THOL) preserves global form while producing sub-EPIs;
  its tetrad-energy change remains trajectory-dependent.
3. U2 requires stabilizer coverage for destabilizer debt; it does not quantify
  absorption of tetrad energy.
4. The net energy change must therefore be computed from actual snapshots.

Therefore $E$ is a **candidate Lyapunov function**. Grammar validity alone does
not fix the sign of its finite-step change. A complete formal proof of
asymptotic stability would require analytic bounds on the nonlinear operator
interactions; §8.4 records only an operational multiplier model.

**Refinement (Grammar-Energy Landscape)**: The nominal multiplier indicator
($\Pi < 1$) is neither a necessary nor a general sufficient theorem for energy
descent.
Experimental evidence ([example 38](../examples/02_physics_regimes/38_grammar_energy_landscape.py))
shows sequences with $\Pi \approx 1.288$ (non-contractive) that still achieve
net energy descent ($\Delta E = -9.59$). The nominal model is conservative in
that fixture;
the measured sequence descended more steeply than the nominal product
suggested, because operators interacted nonlinearly on that graph state.

### 8.3 Energy Dissipation Rate

The dissipation rate $\dot{E}$ has physical meaning:

$$\mathcal{D}[G] = \max\!\left(0,-\frac{dE}{dt}\right) \geq 0$$

where $\mathcal{D}$ is a nonnegative **dissipation readout** for steps with
$dE/dt\leq0$. This measures the observed change; it does not prove attraction.

High $\mathcal{D}$ → fast convergence to coherence (heavy stabilization)
Low $\mathcal{D}$ → slow convergence (exploration phase)
Energy increase → inspect the trajectory and grammar independently

### 8.4 Per-operator nominal multiplier model

The compatibility API assigns each operator a nominal multiplier from its U2
role and configured factor. These values are not proved bounds on the tetrad
energy for every state.

The canonical U2 partition is derived from
`config.physics_derivation`: stabilizers `{IL, THOL}`, destabilizers
`{OZ, ZHIR, VAL}`, and all other operators neutral for debt accounting. The
compatibility class `OperatorLyapunovBound` maps this partition to nominal
multipliers; despite its historical name, it is not a proof of a bound on the
tetrad energy. Operator names, channels and postconditions remain owned by
`operator_contracts.py` and must not be duplicated here.

**Nominal sequence product (not a U2 consequence)**

For any supplied sequence $\{O_1, O_2, \ldots, O_n\}$, the compatibility
model defines a nominal multiplier per operator:

$$m_i = \begin{cases}
1 - \rho_i & \text{stabiliser} \\
1 + \kappa_i & \text{destabiliser} \\
1 & \text{neutral}
\end{cases}$$

The cumulative product $\Pi = \prod_{i=1}^{n} m_i$ satisfies:

- $\Pi < 1$ → contractive in this nominal multiplier model
- $\Pi \geq 1$ → non-contractive in this nominal multiplier model

Neither result establishes or refutes grammar validity; U2 is checked by the
grammar/debt owner and actual energy change is read from the trajectory.

*Example*: OZ followed by 4×IL:
$(1 + 3.0) \times (1 - 0.438)^4 = 4.0 \times 0.0997 \approx 0.40 < 1$ ✓

### 8.5 Spectral Gap Characterisation

The **diffusive relaxation time-scale** is controlled by the canonical TNFR
diffusion operator $L_{\mathrm{rw}} = I - D^{-1}W$ (the EPI channel of the nodal
equation; see `structural_diffusion`): the field relaxes as
$e^{-\nu_f \lambda_2 t}$, where $\lambda_2$ is the spectral gap of the symmetric
normalized Laplacian $L_{\mathrm{sym}} = I - D^{-1/2} W D^{-1/2}$ (same spectrum as
$L_{\mathrm{rw}}$, orthonormal eigenbasis).  The **combinatorial algebraic
connectivity** $\lambda_1$ of $L = D - A$ is the related graph-topology measure;
the two coincide only up to the degree normalisation ($\lambda_1/d$ on a
$d$-regular graph) and differ on irregular graphs.  The convergence rate below
uses the **normalized** (canonical) gap; `analyze_spectral_gap` exposes it as
`diffusion_gap`.

**Spectral Quantities**

| Quantity | Symbol | Formula | Physical Meaning |
|----------|--------|---------|-----------------|
| Spectral gap | $\lambda_1$ | $\min(\lambda_k : \lambda_k > 0)$ | Algebraic connectivity |
| Relaxation time | $\tau_{\text{relax}}$ | $1/\lambda_1$ | Time for slowest non-trivial mode to decay by $e$ |
| Mixing time | $t_{\text{mix}}$ | $\ln(N)/\lambda_1$ | Upper bound on mixing time |
| Cheeger bound | $h$ | $\sqrt{2\,d_{\max}\,\lambda_1}$ | Isoperimetric lower bound |
| Spectral ratio | $r$ | $\lambda_{\max}/\lambda_1$ | Condition number of dynamics |

**Nominal comparison rate**

For a stabilizer, the compatibility API reports

$$r_{\text{eff}} = \min(\rho, \lambda_1)$$

and an associated nominal half-life:

$$t_{1/2} = \frac{\ln 2}{r_{\text{eff}}}$$

This is a policy-model summary, not a theorem that an arbitrary operator step
follows the linear diffusion eigenmodes. The spectral ratio remains a graph
diagnostic; actual convergence must be established for the declared dynamics.

**Implementation**: `src/tnfr/physics/lyapunov.py` — nominal per-operator
multipliers, spectral gap analysis, and sequence policy diagnostics.
**Validation**: the Lyapunov test suite in
`tests/core_physics/test_lyapunov_operators.py`.

### 8.6 Relaxation-rate identity and the partial Lyapunov reduction

The structural H-theorem of the EPI diffusion channel — the Dirichlet energy
$F = \tfrac12\sum_{ij} A_{ij}(\mathrm{EPI}_i - \mathrm{EPI}_j)^2$ is non-increasing
under $\partial_t\mathrm{EPI} = -\nu_f L_{\mathrm{rw}}\mathrm{EPI}$, a *proven*
fact (Lyapunov functional of the heat semigroup; see
`examples/08_emergent_geometry/135_arrow_of_time_h_theorem.py`) — decays at the
**canonical rate**

$$F(t) \sim e^{-2\nu_f \lambda_2 t},\qquad \lambda_2 = \lambda_2(L_{\mathrm{sym}}) = \texttt{diffusion\_gap},$$

verified to machine precision in a mode-isolated test ($|{\rm err}| \sim 10^{-15}$
across regular and irregular graphs; the combinatorial $\lambda_2(L=D-A)$ gives
the *wrong* rate, off by the degree factor $1/d$ on regular graphs and more on
irregular ones). This is the $\lambda_2$ that §8.5 uses for the **diffusion**
relaxation rate. It does not establish the decay rate of the full tetrad energy
or of arbitrary operator trajectories.

**Partial reduction of the open $dE/dt\le 0$ proof.** The energy functional
$E = \tfrac12\sum(\Phi_s^2 + |\nabla\phi|^2 + K_\phi^2 + J_\phi^2 + J_{\Delta\mathrm{NFR}}^2)$
splits into (i) **gradient / Dirichlet sectors** ($|\nabla\phi|^2$ and the EPI
diffusion sector), for which a fixed symmetric pure-diffusion model has an exact
Dirichlet balance, and (ii) the remaining tetrad terms and operator-dependent
sources. U2 does not bound the latter by itself. This isolates rather than
solves the open monotonicity and asymptotic-stability questions.

A separate Parry/Markov model is explored in
`examples/08_emergent_geometry/150_emergent_grammatical_pattern_parry.py`.
Its relaxation must not be identified with the graph-diffusion clock without
an explicit model bridge.

### 8.7 The two conservation/Lyapunov structures

TNFR exposes one restricted exact diffusion conservation law and a distinct
tetrad conservation diagnostic:

| | EPI channel (diffusion) | Tetrad / grammar |
|---|---|---|
| **Field** | the scalar form EPI | the tetrad $(\Phi_s, \lvert\nabla\phi\rvert, K_\phi, \dots)$ |
| **Quantity** | degree-weighted total $\sum_i \deg(i)\,\mathrm{EPI}_i$ (left null vector of $L_{\mathrm{rw}}$) | Noether-like charge $Q=\sum_i(\Phi_s+K_\phi)$ |
| **Status** | conserved for fixed symmetric pure diffusion with the stated capacity assumptions | drift measured on the supplied trajectory; not implied by U1–U6 |
| **Lyapunov functional** | Dirichlet energy $F=\tfrac12\sum A_{ij}(\mathrm{EPI}_i-\mathrm{EPI}_j)^2$ | $E=\tfrac12\sum(\Phi_s^2+\lvert\nabla\phi\rvert^2+K_\phi^2+J_\phi^2+J_{\Delta\mathrm{NFR}}^2)$ |
| **Equilibrium** | uniform EPI on each connected component | no general attractor theorem |
| **Implementation** | `structural_diffusion.degree_weighted_total`, `stationary_distribution` | `conservation.compute_noether_charge`, `compute_energy_functional` |

The two are **independent**: the degree-weighted total acts on EPI, while $Q$
is a tetrad diagnostic. The exact fixed-graph diffusion theorem does not imply
conservation or one-clock relaxation of the full tetrad trajectory.

---

## 9. Discrete Formulation on Graphs

### 9.1 Graph Laplacian Connection

The discrete divergence used in TNFR conservation is the **random-walk graph
Laplacian** $L_{\mathrm{rw}} = I - D^{-1}W$ — the $1/d_i$ normalisation turns the
combinatorial Laplacian $L = D - A$ into $L_{\mathrm{rw}}$:

$$(\nabla \cdot \mathbf{J})(i) \approx \frac{1}{d_i} \sum_{j \sim i} [J(j) - J(i)] = \frac{1}{d_i} (L \cdot \mathbf{J})_i = (L_{\mathrm{rw}} \mathbf{J})_i$$

so the relaxation spectrum governing conservation is that of $L_{\mathrm{rw}}$
(equivalently the symmetric $L_{\mathrm{sym}}$), consistent with §8.5.

This connects conservation on graphs to spectral graph theory.

### 9.2 Spectral Decomposition

Expanding in the eigenbasis of the graph Laplacian $L \psi_k = \lambda_k \psi_k$:

$$\rho(i) = \sum_k \hat{\rho}_k \psi_k(i), \quad J(i) = \sum_k \hat{J}_k \psi_k(i)$$

The continuity equation mode-by-mode:

$$\frac{d\hat{\rho}_k}{dt} + \lambda_k \hat{J}_k = \hat{\mathcal{S}}_k$$

Low-frequency modes ($\lambda_k$ small): charge changes slowly, mainly
transported → *conservation regime*

High-frequency modes ($\lambda_k$ large): rapid transport, potential
dissipation → *relaxation regime*

### 9.3 Residual resolution by scale

The eigenvalue spectrum supplies a basis in which to report the measured
balance residual by scale:

- **Global modes** ($k = 0, 1$): inspect aggregate charge drift
- **Mesoscale modes**: inspect sector-level residual coupling
- **Local modes** ($k \to N$): inspect rapidly varying sources and sinks

This is an observational decomposition; it does not certify U5.

---

## 10. Numerical Validation

### 10.1 Protocol

Conservation validated across:
- **Topologies**: Watts-Strogatz, Barabási-Albert, Grid, Complete
- **Sizes**: $N = 10$ to $N = 500$
- **Dynamics**: Nodal equation integration with $\Delta t = 0.01$
- **Duration**: 20–100 steps per experiment
- **Discretization**: Crank-Nicolson (trapezoidal) divergence averaging
  $\frac{1}{2}[\nabla\!\cdot\!\mathbf{J}_{\text{before}} + \nabla\!\cdot\!\mathbf{J}_{\text{after}}]$
  for $\mathcal{O}(\Delta t^2)$ accuracy

### 10.2 Key Results

| Metric | WS(30,4,0.3) | BA(30,3) | Grid(5×5) |
|--------|-------------|----------|-----------|
| Charge drift (20 steps) | $2.0 \times 10^{-4}$ | $1.8 \times 10^{-4}$ | $2.3 \times 10^{-4}$ |
| Conservation quality | 0.65 | 0.63 | 0.61 |
| Sector asymmetry | 1.03 | 1.12 | 1.08 |
| Cross-coupling $\kappa$ | 0.65 | 0.58 | 0.71 |
| Energy monotonicity | Yes | Yes | Yes |

### 10.3 Interpretation

- **Charge drift < 0.03%** in this finite protocol
- **Conservation quality ≈ 0.6** reflects the discrete approximation; improves
  with smaller $\Delta t$ and denser networks
- **Cross-coupling** ≈ 0.6–0.7 is a measured correlation in these runs
- **Energy monotonically decreasing** in these runs supports continued study
  of the Lyapunov candidate

### 10.4 Measured finite-size fit

Conservation quality scales as:

$$q(N) \sim 1 - \frac{C}{\sqrt{N}}$$

where $C \approx 2.1$ was fitted on the recorded finite topologies. No
continuum-limit convergence or exact conservation theorem follows from that
finite sample.

---

## 11. Physical Interpretation and Analogies

> **Note**: The tables in this section draw structural analogies between TNFR conservation quantities and established physical theories. These parallels serve as intuition aids and naming conventions; they are not claims that TNFR derives or replaces those physical theories.

### 11.1 Electrodynamics Analogy

| TNFR | Electrodynamics |
|------|-----------------|
| $\rho = \Phi_s + K_\phi$ | $\rho = \text{charge density}$ |
| $\mathbf{J} = (J_\phi, J_{\Delta\text{NFR}})$ | $\mathbf{J} = \text{current density}$ |
| Grammar U-rules | Gauge symmetry U(1) |
| Operator sequences | Gauge transformations |
| $\mathcal{S}_{\text{grammar}}$ | Gauge anomaly |
| Energy functional $E$ | Field energy $\frac{1}{2}(E^2 + B^2)$ |

### 11.2 Fluid Dynamics Analogy

| TNFR | Fluid Dynamics |
|------|---------------|
| $\rho$ → structural charge | $\rho$ → mass density |
| $\mathbf{J}$ → structural flow | $\rho\mathbf{v}$ → momentum density |
| Grammar → incompressibility | $\nabla \cdot \mathbf{v} = 0$ |
| Coherence (IL) → viscosity | Energy dissipation |

### 11.3 Thermodynamic Analogy

| TNFR | Thermodynamics |
|------|---------------|
| $E$ → structural energy | Internal energy $U$ |
| $\mathcal{D}$ → dissipation rate | Entropy production $\dot{S}$ |
| Grammar evolution → irreversibility | Second law |
| Coherent attractor → equilibrium | Thermal equilibrium |

---

## 12. Applications

### 12.1 Grammar Violation Detection

Conservation residuals serve as a **real-time anomaly indicator** alongside
the canonical grammar validator:

```python
tracker = ConservationTracker(G)
tracker.record(t=0.0)
apply_operator_sequence(G, sequence)
tracker.record(t=1.0)

balance = tracker.latest_balance
if balance.grammar_violation_index > 0.5:
    violations = detect_grammar_violations_from_conservation(balance)
    # Heuristic classification; validate the actual history separately.
```

### 12.2 Self-Optimization via Conservation Monitoring

The `ConservationTracker` can guide the self-optimizing engine:

1. **Monitor** conservation quality during optimization
2. **Flag** operator choices for review when residuals rise
3. **Correct** by selecting operators that restore conservation
4. **Verify** improvement after correction

### 12.3 Network Health Telemetry

Conservation quality serves as an aggregate health metric:

- $q > 0.9$: Excellent structural coherence
- $0.5 < q < 0.9$: Active dynamics, normal operation
- $q < 0.5$: Possible grammar violation or fragmentation risk

### 12.4 Diagnostic associations

The sector decomposition flags patterns for investigation:

- **Potential sector dominant**: ΔNFR imbalance → U2/U6 risk
- **Geometric sector dominant**: Phase decoherence → U3 risk
- **Both elevated**: Cascading bifurcation → U4/U5 risk

### 12.5 Operator-Tetrad Fingerprinting

The per-operator Ward identities (§7.2) are experimentally confirmed by the
**operator-tetrad fingerprint matrix** ([example 37](../examples/02_physics_regimes/37_operator_tetrad_synergy.py)).
Each operator produced a signature across (Φ_s, |∇φ|, K_φ, ξ_C) in the
recorded protocol. Such fingerprints can assist runtime diagnosis, but they do
not uniquely identify an operator outside that finite model and state family.

---

## 13. Implementation Reference

### 13.1 Core Module

**File**: `src/tnfr/physics/conservation.py`

| Component | Purpose |
|-----------|---------|
| `ConservationSnapshot` | Frozen state capture at time $t$ |
| `ConservationBalance` | Two-snapshot continuity verification |
| `ConservationTimeSeries` | Multi-step diagnostics |
| `ConservationTracker` | Live tracking across operator sequences |
| `compute_charge_density(G)` | $\rho(i) = \Phi_s(i) + K_\phi(i)$ |
| `compute_current_divergence(G)` | $\nabla \cdot \mathbf{J}$ |
| `compute_noether_charge(G)` | $Q = \sum_i \rho(i)$ |
| `compute_energy_functional(G)` | $E = \frac{1}{2}\sum(\Phi_s^2 + K_\phi^2 + J_\phi^2 + J_{\Delta\text{NFR}}^2)$ |
| `verify_conservation_balance(...)` | Continuity equation residual (Crank-Nicolson, O(Δt²)) |
| `decompose_conservation_residual(...)` | Sector decomposition (Crank-Nicolson) |
| `analyze_sector_coupling(...)` | Cross-sector correlation |
| `compute_grammar_conservation_bounds(G)` | Legacy policy-scaled alert levels |
| `detect_grammar_violations_from_conservation(...)` | Heuristic alert classification |

| `WardIdentity` | Per-operator conservation signature |
| `LyapunovResult` | Lyapunov dE/dt analysis |
| `SpectralConservation` | Graph Laplacian eigendecomposition |
| `compute_ward_identity(...)` | Single-step Ward identity |
| `verify_sequence_ward_identity(...)` | Sequence Σ⟨S_k⟩ ≈ 0 |
| `compute_lyapunov_derivative(...)` | dE/dt and dissipation D[G] |
| `compute_spectral_conservation(...)` | Spectral mode analysis |
| `compute_conservation_scaling(...)` | q(N) ~ 1 − C/√N fit |

**Per-Operator Lyapunov Module** (`src/tnfr/physics/lyapunov.py`):

| Component | Purpose |
|-----------|---------|
| `EnergyClass` | Enum: STABILISER, DESTABILISER, NEUTRAL, MIXED |
| `OperatorLyapunovBound` | Legacy-named nominal U2-role multiplier |
| `OPERATOR_LYAPUNOV_BOUNDS` | Registry of all 13 operator bounds |
| `get_bound(name_or_glyph)` | Lookup by operator name or glyph |
| `compute_operator_energy_bound(...)` | Nominal ΔE allowance per step |
| `compute_sequence_energy_bound(...)` | Cumulative nominal multiplier model |
| `verify_operator_lyapunov(...)` | Observed change versus nominal model |
| `analyze_spectral_gap(G)` | Full Laplacian eigendecomposition: λ₁, τ_relax, t_mix, Cheeger |
| `analyze_operator_convergence(G, name)` | Combined Lyapunov + spectral rate |
| `prove_sequence_lyapunov(operators)` | Legacy-named nominal contractivity check |

### 13.2 Tests

**File**: `tests/core_physics/test_conservation_laws.py` — 62 tests
**File**: `tests/core_physics/test_lyapunov_operators.py` — compatibility multipliers, spectral gap and sequence diagnostics

### 13.3 Benchmark

**File**: `benchmarks/conservation_law_validation.py`

### 13.4 Example

**File**: `examples/02_physics_regimes/17_conservation_law_demo.py`

---

## 14. Summary of Main Results

1. **Structural balance identity**: $\partial\rho/\partial t + \nabla \cdot \mathbf{J} = \mathcal{S}$, where $\mathcal S$ is measured from the actual trajectory.

2. **Noether-like correspondence**: Grammar rules organize diagnostic analogies; only a specified continuous symmetry and model establish a Noether charge.

3. **Two-Sector Structure**: Conservation decomposes into potential ($\Phi_s \leftrightarrow J_{\Delta\text{NFR}}$) and geometric ($K_\phi \leftrightarrow J_\phi$) sectors coupled through $\Psi = K_\phi + i J_\phi$.

4. **Ward diagnostics**: Operators have measured conservation signatures; the aggregate residual is reported rather than assumed to vanish.

5. **Lyapunov candidate**: $E = \frac{1}{2}\sum(\Phi_s^2 + |\nabla\phi|^2 + K_\phi^2 + J_\phi^2 + J_{\Delta\text{NFR}}^2)$ is nonnegative. Its observed change and the nominal per-operator multipliers are diagnostics; a complete monotonicity or asymptotic-stability proof remains open. The separate Dirichlet energy decreases exactly in its fixed symmetric diffusion regime.

6. **Numerical validation**: The recorded finite protocol measured small charge drift and a size trend on selected topologies; no continuum extrapolation is claimed.

7. **Diagnostic application**: Conservation residuals flag patterns for review; canonical history validation decides grammar compliance.

---

---

## Implementation & Examples

### SDK Entry Points

```python
from tnfr.sdk import TNFR

net = TNFR.create(20).ring().evolve(5)
cons = net.conservation()            # ConservationReport
print(cons.summary())                # Q, E, dE/dt, stability
```

### Executable Demonstrations

| Example | Concept from this document |
|---------|---------------------------|
| [17_conservation_law_demo.py](../examples/02_physics_regimes/17_conservation_law_demo.py) | Noether charge, energy functional, Lyapunov stability, Ward identities |
| [34_conservation_protocol_suite.py](../examples/02_physics_regimes/34_conservation_protocol_suite.py) | Multi-topology conservation protocol: charge drift, q(N) scaling, sector decomposition (§10) |
| [36_grammar_violation_detector.py](../examples/02_physics_regimes/36_grammar_violation_detector.py) | Grammar violation detection via conservation residuals (§12.1), violation classification |

### Key Source Modules

- `src/tnfr/physics/conservation.py` — Canonical conservation implementation
- `src/tnfr/sdk/simple.py` — `ConservationReport` dataclass

---

**Status**: CANONICAL DIAGNOSTIC REFERENCE
**Derived from**: field definitions and the measured balance identity; exact diffusion results retain their stated hypotheses
**Validated by**: finite numerical experiments and the current test suite
**Implementation**: `src/tnfr/physics/conservation.py`
