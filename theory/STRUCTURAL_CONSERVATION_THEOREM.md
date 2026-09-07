# TNFR Structural Conservation Diagnostics

## Noether-Like Balance Candidates and Restricted Exact Laws

**Status**: CANONICAL DIAGNOSTIC REFERENCE — algebraic balance plus measured residuals
**Date**: September 2026
**Version**: 0.0.3.5
**Prerequisite**: [AGENTS.md](../AGENTS.md) §Foundational Physics, [UNIFIED_GRAMMAR_RULES.md](UNIFIED_GRAMMAR_RULES.md) §U2, §U6

---

## Table of Contents

1. [Scope and Motivation](#1-scope-and-motivation)
2. [Governing Dynamics Recap](#2-governing-dynamics-recap)
3. [Structural Charge and Current Definitions](#3-structural-charge-and-current-definitions)
4. [Construction of the Balance Equation](#4-derivation-of-the-continuity-equation)
5. [Two-Sector Decomposition](#5-two-sector-decomposition)
6. [Grammar/Balance Correspondences](#6-noether-correspondence-grammar-conservation)
7. [Ward-Like Diagnostics for Operator Sequences](#7-ward-identities-for-operator-sequences)
8. [Lyapunov Candidate and Restricted Stability](#8-lyapunov-stability-from-the-energy-functional)
9. [Discrete Formulation on Graphs](#9-discrete-formulation-on-graphs)
10. [Numerical Evaluation](#10-numerical-evaluation)
11. [Physical Interpretation and Analogies](#11-physical-interpretation-and-analogies)
12. [Applications](#12-applications)
13. [Implementation Reference](#13-implementation-reference)
14. [Summary of Main Results](#14-summary-of-main-results)

---

## 1. Scope and Motivation

Noether's theorem relates continuous symmetries of a specified action to
conserved currents under its mathematical hypotheses. TNFR operates on
*discrete* graphs with *discrete* operator sequences constrained by grammar
rules U1–U6. The question is therefore:

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
- $\Delta\text{NFR}_i(t)$ is the nodal reorganization pressure

### 2.2 Phase Dynamics

The nodal equation specifies the EPI channel; it does not by itself supply a
universal differential equation for phase. Canonical phase-changing operators
define their own updates and UM/RA require the U3 compatibility check. An
auxiliary Kuramoto comparison may separately posit

$$\frac{\partial \phi_i}{\partial t}
= \nu_{f,i}\,h_i(\phi,\Delta\mathrm{NFR}),$$

often with sine coupling. Results derived from that auxiliary phase law retain
its assumptions and do not automatically apply to every operator trajectory.

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

The diagnostic $\rho$ combines the *global potential landscape* with the
*local geometric curvature*. This selection is useful because:

1. $\Phi_s$ aggregates the reorganization pressure field (information about
   the entire network projected onto node $i$)
2. $K_\phi$ captures the local geometric mismatch (how much node $i$ deviates
   from its neighborhood's mean phase)
3. Their sum gives one scalar structural-stress readout at node $i$

These observations define a charge **candidate**. They do not establish that
its graph sum is conserved under the nodal equation or canonical operators.

### 3.2 Structural Current Vector

$$\mathbf{J}(i, t) = \big(J_\phi(i, t),\; J_{\Delta\text{NFR}}(i, t)\big)$$

where:

**Phase Current** (transport of phase coherence):
$$J_\phi(i) = \frac{1}{|\mathcal{N}(i)|} \sum_{j \in \mathcal{N}(i)} \sin(\phi_j - \phi_i)$$

**Reorganization Flux** (transport of structural pressure):
$$J_{\Delta\text{NFR}}(i) = \frac{1}{|\mathcal{N}(i)|} \sum_{j \in \mathcal{N}(i)} \big(\Delta\text{NFR}_j - \Delta\text{NFR}_i\big)$$

The selected current readout contains two types of structural information:

- $J_\phi$ is a signed local phase-mismatch average
- $J_{\Delta\text{NFR}}$ is a signed local pressure-difference average

Calling these quantities currents does not establish a transport equation;
the measured balance residual tests that proposed interpretation on a declared
trajectory.

### 3.3 Why This Pairing?

The charge–current pairing $(\rho, \mathbf{J})$ is the selected two-sector
diagnostic construction:

| Sector | Charge Component | Current Component | Driving Physics |
|--------|-----------------|-------------------|-----------------|
| **Potential** | $\Phi_s$ | $J_{\Delta\text{NFR}}$ | ΔNFR distribution & redistribution |
| **Geometric** | $K_\phi$ | $J_\phi$ | Phase dynamics & curvature transport |

The complex geometric field $\Psi = K_\phi + iJ_\phi$ packages the geometric
pair algebraically. Finite correlations between its components depend on the
graph ensemble and trajectory; they neither derive this definition nor prove
that the two residual sectors are dynamically conjugate.

---

## 4. Construction of the Balance Equation {#4-derivation-of-the-continuity-equation}

### 4.1 Time Derivative of Structural Potential

$$\frac{\partial \Phi_s(i)}{\partial t} = \sum_{j \neq i} \frac{1}{d(i,j)^\alpha} \frac{\partial \Delta\text{NFR}_j}{\partial t}$$

The nodal equation determines the EPI rate from the supplied pressure; it does
not determine $\partial_t\Delta\text{NFR}_j$. If a specific operator trajectory
or pressure model supplies a pressure derivative satisfying

$$\sum_{j \neq i} \frac{|\partial \Delta\text{NFR}_j / \partial t|}{d(i,j)^\alpha} < \infty,$$

then the fixed-kernel potential derivative is bounded. U2 sequence validity alone
does not supply this analytic hypothesis; topology changes add a kernel-change
term and must be handled separately.

### 4.2 Time Derivative of Phase Curvature

Away from a wrap discontinuity and where the neighborhood circular resultant
is nonzero, differentiating the circular mean gives the local form

$$\frac{\partial K_\phi(i)}{\partial t}
= \frac{\partial \phi_i}{\partial t}
- \sum_{j \in \mathcal{N}(i)} w_j(\phi)
  \frac{\partial \phi_j}{\partial t},$$

with state-dependent circular-mean derivative weights $w_j(\phi)$. This is not
a global identity at branch cuts or undefined circular means. U3 checks the
wrapped phase difference before a coupling/resonance action. A phase-velocity
bound requires an additional, specified phase-evolution law; it does not follow
from U3 alone.

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

The historical program proposed

$$q(N)\sim1-\frac{C}{\sqrt N}$$

and reported $C\approx2.1$ for an earlier finite sample. The current seeded
protocol in example 34 does not reproduce the required coefficient direction.
Neither result is a continuum-limit theorem or a graph-uniform bound.

**Step 6. Residuals are alerts, not validators.**

Large potential or geometric residuals can accompany U2, U3 or U6 risks, but
they may also arise from discretization, topology changes or an external
pressure law. The actual operator history must be validated independently.
`compute_grammar_conservation_bounds` therefore exposes compatibility alert
levels, not admitted analytical bounds on every valid trajectory.

---

## 5. Two-Sector Decomposition

The full measured balance decomposes into two residual channels:

### 5.1 Potential Sector

$$\frac{\partial \Phi_s}{\partial t} + \nabla \cdot J_{\Delta\text{NFR}} = \mathcal{S}_{\text{pot}}$$

- **Physics**: Global ΔNFR landscape evolves via operator-driven redistribution
- **Grammar context**: U2 tracks stabilizer debt; U6 supplies a potential-drift
  alert. Neither bounds $\mathcal{S}_{\text{pot}}$ by itself.
- **Monitoring**: Evaluate the declared-reference U6 statistic
  $\operatorname{mean}_i|\Delta\Phi_s(i)| < \pi/2$

### 5.2 Geometric Sector

$$\frac{\partial K_\phi}{\partial t} + \nabla \cdot J_\phi = \mathcal{S}_{\text{geo}}$$

- **Physics**: Local phase curvature evolves via synchronization dynamics
- **Grammar context**: U3 gates coupling and U4 checks bifurcation context;
  neither proves a bound on curvature transport by itself.
- **Monitoring**: compare per-node $|K_\phi|$ with the selected
  $0.9\pi\approx2.827$ safety margin; the exact wrap bound is $\pi$

### 5.3 Cross-Sector Coupling

The diagnostic reports their finite **cross-sector correlation** as

$$\kappa = \text{corr}\!\left(\mathcal{S}_{\text{pot}}, \mathcal{S}_{\text{geo}}\right)$$

The recorded example-17 fixture gives a value in the range
$\kappa \approx 0.6$–$0.7$. This single-protocol association neither proves
that the sectors are dynamically dependent nor measures a causal coupling.
It also does not prove the algebraic complex-field construction
$\Psi = K_\phi + i J_\phi$, which is defined independently.

---

## 6. Grammar/Balance Correspondences {#6-noether-correspondence-grammar-conservation}

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

### 6.3 Comparing independent evidence

When a grammar validator separately reports a rule failure, the balance data
can be inspected for association:

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

The implementation retains **WardIdentity** as the historical name for a
per-step, network-averaged charge/energy diagnostic. It does not compute a
field-theoretic expectation-value identity. For a TNFR operator
$\mathcal{O}_k$ observed over an interval $\Delta t_k$, its mean residual is

$$\overline{\mathcal S}_k = \frac{1}{N}\sum_i\left[
\frac{\rho_i^{k+1}-\rho_i^k}{\Delta t_k}
+\frac{(\nabla\cdot\mathbf J)_i^k+(\nabla\cdot\mathbf J)_i^{k+1}}{2}
\right].$$

The same record stores the distinct total changes
$\Delta Q_k=Q_{k+1}-Q_k$ and $\Delta E_k=E_{k+1}-E_k$. It does not identify
the total charge change with the mean residual.

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
the compatibility diagnostic defines the aggregate residual score

$$\mathcal{S}_{\mathrm{agg}}=\sum_{k=1}^{N} \overline{\mathcal{S}}_k.$$

This unweighted sum is not a time integral when interval lengths differ, so it
should be compared only within a declared sampling protocol. Whether it is
small is measured. U1 closure and U2 debt balance do not prove cancellation
of the independently defined tetrad residual.

**Experimental note (observational readout chain)**: The operator-specific
measurements in §7.2 can be organized as
Operator → (ν_f, ΔNFR) → dEPI/dt → Tetrad → (ℰ, Q). Each operator
produced a tetrad fingerprint in the stated fixtures (see [STRUCTURAL_OPERATORS.md
§17.2](STRUCTURAL_OPERATORS.md) and [example 37](../examples/02_physics_regimes/37_operator_tetrad_synergy.py)).
Those finite fingerprints do not identify this diagram as a causal model.
Equal or similar energy changes for operators with different contracts also
show why perturbation magnitude cannot assign a universal sign or operator.

---

## 8. Lyapunov Candidate and Restricted Stability {#8-lyapunov-stability-from-the-energy-functional}

### 8.1 Energy Functional

Define the **structural energy candidate** from five included readouts:

$$E[G] = \frac{1}{2} \sum_{i \in V} \left[\Phi_s(i)^2 + |\nabla\phi|(i)^2 + K_\phi(i)^2 + J_\phi(i)^2 + J_{\Delta\text{NFR}}(i)^2\right]$$

This is the half-sum of the quadratic **energy-density diagnostic**
$\mathcal{E}$. All five listed fields contribute by definition. Including
$|\nabla\phi|^2$ records phase-gradient stress; its inclusion does not create a
Noether theorem for the graph dynamics.

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

- High $\mathcal{D}$: large candidate-energy decrease on the sampled interval
- Low $\mathcal{D}$: small or zero candidate-energy decrease on that interval
- Energy increase: inspect the trajectory and grammar independently

The size of $\mathcal D$ alone does not determine convergence speed or the
limiting state.

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

### 8.5 Independent spectral context

For fixed connected symmetric nonnegative weights and homogeneous capacity,
the isolated EPI channel uses
$L_{\mathrm{rw}}=I-D^{-1}W$. Its nonzero decay rates are
$\nu_f\lambda_k(L_{\mathrm{sym}})$, where
$L_{\mathrm{sym}}=I-D^{-1/2}WD^{-1/2}$ has the same spectrum. The smallest
positive normalized eigenvalue therefore controls the slowest nonuniform EPI
mode in this restricted model.

`analyze_spectral_gap` reports two different gaps and keeps their roles
separate:

| Returned quantity | Definition | Scope |
|-------------------|------------|-------|
| `fiedler_value` / `spectral_gap` | second combinatorial eigenvalue of $D-W$ | topology diagnostic |
| `diffusion_gap` | second normalized eigenvalue of $L_{\mathrm{sym}}$ | unit-capacity pure-EPI decay |
| `relaxation_time` | $1/\texttt{diffusion_gap}$ | restricted continuous-time scale |
| `mixing_time_bound` | $\log N/\texttt{diffusion_gap}$ | library comparison estimate |
| `cheeger_lower` | $\texttt{diffusion_gap}/2$ | normalized Cheeger lower estimate |

The operator policy multiplier acts per history position, whereas the
diffusion gap acts per continuous-time unit. The implementation leaves their
combined rate undefined because no canonical map between those clocks has
been established.

**Implementation**: `src/tnfr/physics/lyapunov.py` — nominal per-operator
multipliers, spectral gap analysis, and sequence policy diagnostics.
**Validation**: the Lyapunov test suite in
`tests/core_physics/test_lyapunov_operators.py`.

### 8.6 Relaxation-rate identity and the partial Lyapunov reduction

The structural H-theorem of the EPI diffusion channel — the Dirichlet energy
$F = \tfrac14\sum_{ij} W_{ij}(\mathrm{EPI}_i - \mathrm{EPI}_j)^2$
is non-increasing
under $\partial_t\mathrm{EPI} = -\nu_f L_{\mathrm{rw}}\mathrm{EPI}$, a *proven*
fact (Lyapunov functional of the heat semigroup; see
`examples/08_emergent_geometry/135_arrow_of_time_h_theorem.py`) — decays at the
**canonical rate**

$$F(t) = F(0)e^{-2\nu_f\lambda_k t}$$

for an isolated normalized eigenmode $k$. The slowest nonuniform asymptotic
exponent uses $\lambda_2=\texttt{diffusion\_gap}$ when that mode has nonzero
amplitude. This is not an exact single exponential for an arbitrary mixture of
modes. The isolated-mode exponent was
verified to machine precision ($|{\rm err}| \sim 10^{-15}$
across regular and irregular graphs; the combinatorial $\lambda_2(L=D-A)$ gives
the *wrong* rate, off by the degree factor $1/d$ on regular graphs and more on
irregular ones). This is the $\lambda_2$ that §8.5 uses for the **diffusion**
relaxation rate. It does not establish the decay rate of the full tetrad energy
or of arbitrary operator trajectories.

**Partial reduction of the open $dE/dt\le 0$ proof.** The exact fixed-graph
Dirichlet balance controls the separate EPI-diffusion energy. It does not by
itself control the phase-gradient term $|\nabla\phi|^2$ or the remaining tetrad
terms in
$E = \tfrac12\sum(\Phi_s^2 + |\nabla\phi|^2 + K_\phi^2 + J_\phi^2 + J_{\Delta\mathrm{NFR}}^2)$.
Those terms require model-specific evolution identities and bounds on their
operator-dependent sources. U2 supplies neither automatically. The exact EPI
sector therefore isolates one proved component without solving the open
monotonicity and asymptotic-stability questions for $E$.

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
| **Quantity** | $\sum_i d_i\,\mathrm{EPI}_i$ for common capacity; $\sum_i(d_i/\nu_{f,i})\,\mathrm{EPI}_i$ for fixed positive heterogeneous capacity | Noether-like charge $Q=\sum_i(\Phi_s+K_\phi)$ |
| **Status** | conserved on a fixed connected symmetric pure-diffusion graph under the respective capacity assumptions | drift measured on the supplied trajectory; not implied by U1–U6 |
| **Lyapunov functional** | Dirichlet energy $F=\tfrac14\sum_{ij}W_{ij}(\mathrm{EPI}_i-\mathrm{EPI}_j)^2$ | candidate $E=\tfrac12\sum(\Phi_s^2+\lvert\nabla\phi\rvert^2+K_\phi^2+J_\phi^2+J_{\Delta\mathrm{NFR}}^2)$ |
| **Equilibrium** | uniform EPI on each connected component | no general attractor theorem |
| **Implementation** | `structural_diffusion.degree_weighted_total`, `stationary_distribution` | `conservation.compute_noether_charge`, `compute_energy_functional` |

The two are **independent**: the weighted totals act on EPI, while $Q$ is a
tetrad diagnostic. The exact fixed-graph diffusion theorem does not imply
conservation or one-clock relaxation of the full tetrad trajectory.

---

## 9. Discrete Formulation on Graphs

### 9.1 Graph Laplacian Connection

For one scalar current component, the implemented graph divergence uses an
unweighted neighbor-minus-center average. It therefore has the sign of the
negative random-walk Laplacian of the unweighted topology:

$$(\nabla\cdot J)(i)=\frac{1}{d_i}\sum_{j\sim i}[J(j)-J(i)]
=-(L_{\mathrm{rw}}J)_i.$$

The implemented two-sector divergence sums this construction for $J_\phi$ and
$J_{\Delta\mathrm{NFR}}$. This algebraic use of the graph operator does not
make the residual follow the pure-EPI diffusion semigroup of §8.5.

It nevertheless permits a spectral decomposition of the measured residual.

### 9.2 Spectral Decomposition

For a fixed snapshot, `compute_spectral_conservation` projects the charge and
already-computed divergence into the orthonormal $L_{\mathrm{sym}}$ basis
returned by the shared diffusion helper:

$$\rho=\sum_k\hat\rho_k\psi_k,\qquad
\operatorname{div}J=\sum_k\widehat{\operatorname{div}J}_k\psi_k.$$

Its canonical `modal_divergence_magnitude` value is the static activity
readout

$$a_k=\left|\widehat{\operatorname{div}J}_k\right|.$$

The divergence has already been computed in node space, so another factor of
$\lambda_k$ would apply an unintended second graph derivative. Because one
snapshot contains no $d\hat\rho_k/dt$, $a_k$ is not a modal continuity
residual. `low_divergence_activity_modes` is a descriptive median split; the
legacy aliases `conservation_by_mode` and `dominant_conservation_modes` expose
the same readouts and must not be read as a conservation theorem or U5 result.
On a weighted graph, this basis may differ from the unweighted operator used
to construct the stored divergence; the projection remains a basis expansion,
not a diagonalization of that divergence rule.

### 9.3 Residual resolution by scale

The eigenvalue spectrum supplies a basis in which to report spatial variation:

- **Low-eigenvalue modes**: spatially smooth components on the graph
- **Intermediate modes**: mesoscale variation relative to that graph
- **High-eigenvalue modes**: rapidly alternating graph components

These labels are relative to the supplied topology. A temporal projection is
still required to form a modal continuity residual, and the decomposition does
not certify U5.

---

## 10. Numerical Evaluation

### 10.1 Protocol

Examples 17 and 34 run a seeded auxiliary phase/pressure smoothing rule on
several finite graph fixtures. They record two-snapshot trapezoidal residuals,
charge drift, candidate energy and sector correlations. The auxiliary rule is
not a canonical operator word, so these runs do not test an implication from
grammar compliance.

The trapezoidal endpoint average has second-order quadrature accuracy for a
smooth divergence signal. The complete balance residual need not converge at
that order unless the sampled fields and underlying trajectory satisfy the
additional regularity and model assumptions.

### 10.2 Historical targets and reproducible outcome

The original protocol advertised fixed targets: relative charge drift below
0.03%, mean quality in $[0.60,0.65]$, sector ratio in $[1.0,1.2]$, a negative
$1/\sqrt N$ fit coefficient, and non-increasing candidate energy at every
sample. Example 34 now evaluates each target and prints `PASS` or `FAIL`; it
retains failures as negative evidence instead of treating them as warnings.

### 10.3 Interpretation policy

- Report the actual topology, seed, time step and auxiliary update rule.
- Treat quality $q=1/(1+\mathrm{RMS})$ as a normalized residual score, not a
  direct coherence, grammar or convergence metric.
- Treat cross-sector correlation as a finite association.
- Treat sampled energy descent as evidence only for the recorded intervals.

### 10.4 Finite-size fit

The historical ansatz asks whether a supplied finite family follows

$$q(N) \sim 1 - \frac{C}{\sqrt{N}}$$

The implementation reports an unconstrained finite fit and checks its
direction. Its intercept and coefficient describe only the supplied graph
family and auxiliary rule. No continuum-limit convergence or exact
conservation theorem follows from that finite sample.

---

## 11. Physical Interpretation and Analogies

The notation resembles continuity, field-energy and dissipation constructions
from established models. The comparison is limited to algebraic form:

| TNFR diagnostic | Familiar comparison | Boundary of the comparison |
|-----------------|---------------------|----------------------------|
| $\rho=\Phi_s+K_\phi$ | a density variable | not electric charge or mass density |
| $\mathbf J=(J_\phi,J_{\Delta\mathrm{NFR}})$ | a flux variable | no transport law follows from its name |
| $\partial_t\rho+\nabla\cdot\mathbf J=\mathcal S$ | a sourced continuity equation | $\mathcal S$ is the measured residual by definition |
| quadratic $E$ | a non-negative field energy | only a Lyapunov candidate for TNFR trajectories |
| $\mathcal D=\max(0,-dE/dt)$ | a dissipation-rate readout | not entropy production or a second-law result |

Grammar rules have no demonstrated identification with gauge symmetry,
incompressibility or thermodynamic irreversibility. The auxiliary harmonic
substrate has its own explicitly specified continuous symmetries and conserved
charges; those belong to that model alone.

---

## 12. Applications

### 12.1 Balance alerts and independent grammar validation

Conservation residuals serve as a **real-time anomaly indicator** alongside
the canonical grammar validator:

```python
from tnfr.operators import apply_glyph
from tnfr.operators.grammar import validate_sequence_incremental
from tnfr.physics.conservation import (
    ConservationTracker,
    detect_grammar_violations_from_conservation,
)

# G, node and planned_sequence are supplied by the experiment.
grammar_steps = validate_sequence_incremental(G, node, planned_sequence)
if not all(step.allowed for step in grammar_steps):
    raise ValueError("inadmissible operator history")

tracker = ConservationTracker(G)
tracker.record(t=0.0)
for glyph in planned_sequence:
    apply_glyph(G, node, glyph)
tracker.record(t=1.0)

balance = tracker.latest_balance
alerts = detect_grammar_violations_from_conservation(balance)
if alerts["alerts_detected"]:
    inspect(alerts["alert_types"])
```

The historical helper name and `violations_*` keys remain for compatibility;
those fields are always false/empty because the helper does not assess grammar.

### 12.2 Self-Optimization via Conservation Monitoring

The `ConservationTracker` can inform the self-optimizing engine:

1. **Monitor** finite balance quality during optimization.
2. **Flag** intervals for review when residuals rise.
3. **Diagnose** the cause from topology, sources, discretization and actual
   operator history.
4. **Select** any structural operator only after its own contract and grammar
   context justify it.

The SDK exposes the sampled mapping as `balance_feedback` while retaining
`conservation_feedback` as an alias. Review prompts are available through
`balance_alert_reviews`; they remain outside `recommended_strategies`, so a
balance alert cannot become an executable optimization choice.
`balance_sample_count = 0` suppresses balance, charge and candidate-energy
reviews because no observed interval exists; operator-postcondition alerts
remain independently reportable. The optimizer compares the mean absolute
charge drift per sampled interval; cumulative drift remains available under
`total_structural_charge_drift` and the legacy `charge_drift` key.

### 12.3 Network Health Telemetry

Balance quality is a normalized finite-residual score. Selected operational
bands may be used for dashboards, but they are not canonical coherence bands:

- $q > 0.9$: small RMS residual under this normalization
- $0.5 < q < 0.9$: intermediate RMS residual
- $q < 0.5$: RMS residual greater than one in the chosen units

### 12.4 Diagnostic associations

The sector decomposition locates which measured channel dominates:

- **Potential sector dominant**: inspect pressure changes and the potential
  kernel.
- **Geometric sector dominant**: inspect phase changes and curvature.
- **Both elevated**: inspect both sources and any topology change.

No sector pattern uniquely identifies a U-rule failure.

### 12.5 Operator-Tetrad Fingerprinting

The **operator-tetrad fingerprint matrix**
([example 37](../examples/02_physics_regimes/37_operator_tetrad_synergy.py))
measures finite signatures across (Φ_s, |∇φ|, K_φ, ξ_C). Such fingerprints can
assist runtime diagnosis, but they do not establish a Ward identity or uniquely
identify an operator outside that finite model and state family.

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
| `compute_noether_charge(G)` | Historical name for charge candidate $Q = \sum_i \rho(i)$ |
| `compute_energy_functional(G)` | Candidate $E = \frac{1}{2}\sum(\Phi_s^2 + |\nabla\phi|^2 + K_\phi^2 + J_\phi^2 + J_{\Delta\text{NFR}}^2)$ |
| `verify_conservation_balance(...)` | Two-snapshot trapezoidal balance residual |
| `decompose_conservation_residual(...)` | Sector decomposition (Crank-Nicolson) |
| `analyze_sector_coupling(...)` | Cross-sector correlation |
| `compute_grammar_conservation_bounds(G)` | Legacy policy-scaled alert levels |
| `detect_grammar_violations_from_conservation(...)` | Legacy-named balance alerts; never a grammar verdict |
| `WardIdentity` | Per-step charge/energy diagnostic |
| `LyapunovResult` | Finite candidate-energy change |
| `SpectralConservation` | Static normalized-Laplacian decomposition |
| `compute_ward_identity(...)` | Legacy-named single-step charge/energy diagnostic |
| `verify_sequence_ward_identity(...)` | Finite-sequence aggregate and legacy threshold alert |
| `compute_lyapunov_derivative(...)` | Sampled candidate $dE/dt$ and descent readout $D[G]$ |
| `compute_spectral_conservation(...)` | Static spectral activity proxy |
| `compute_conservation_scaling(...)` | Historical finite $q(N)$ ansatz fit |

**U2 Policy and Spectral Context Module** (`src/tnfr/physics/lyapunov.py`;
legacy names retained):

| Component | Purpose |
|-----------|---------|
| `EnergyClass` | Enum: STABILISER, DESTABILISER, NEUTRAL, MIXED |
| `OperatorLyapunovBound` | Legacy-named nominal U2-role multiplier |
| `OPERATOR_LYAPUNOV_BOUNDS` | Legacy alias for the 13 nominal U2 policy multipliers |
| `get_bound(name_or_glyph)` | Lookup by operator name or glyph |
| `compute_operator_energy_bound(...)` | Legacy wrapper returning a nominal change in an abstract policy score |
| `compute_sequence_energy_bound(...)` | Legacy wrapper returning the final abstract policy score after multiplier composition |
| `verify_operator_lyapunov(...)` | Observed energy change compared with the nominal policy model; not a Lyapunov certificate |
| `analyze_spectral_gap(G)` | Combinatorial λ₂ and normalized diffusion λ₂, with separate relaxation diagnostics |
| `analyze_operator_convergence(G, name)` | Legacy wrapper reporting policy and spectral context side by side; the combined rate is undefined (`nan`) |
| `prove_sequence_lyapunov(operators)` | Legacy-named nominal policy-product check; not a trajectory proof |

The compatibility names above preserve import paths, not their historical
physical interpretation. Policy multipliers act per operator position, while
the normalized diffusion gap acts per continuous-time unit; no effective rate
is returned without a declared map between those clocks.

### 13.2 Tests

- `tests/core_physics/test_conservation_laws.py`
- `tests/core_physics/test_lyapunov_operators.py` — compatibility multipliers,
  spectral gap and sequence diagnostics

### 13.3 Benchmark

**File**: `benchmarks/conservation_law_validation.py`

### 13.4 Example

**File**: `examples/02_physics_regimes/17_conservation_law_demo.py`

---

## 14. Summary of Main Results

1. **Structural balance identity**: $\partial\rho/\partial t + \nabla \cdot \mathbf{J} = \mathcal{S}$, where $\mathcal S$ is measured from the actual trajectory.

2. **Noether-like correspondence**: Grammar rules organize diagnostic analogies; only a specified continuous symmetry and model establish a Noether charge.

3. **Two-sector diagnostic**: the measured residual separates potential ($\Phi_s$, $J_{\Delta\text{NFR}}$) and geometric ($K_\phi$, $J_\phi$) channels; finite correlation does not prove dynamic conjugacy.

4. **Per-step diagnostics**: A labeled observed step has a measured charge and candidate-energy signature; the aggregate residual is reported rather than assumed to vanish.

5. **Lyapunov candidate**: $E = \frac{1}{2}\sum(\Phi_s^2 + |\nabla\phi|^2 + K_\phi^2 + J_\phi^2 + J_{\Delta\text{NFR}}^2)$ is nonnegative. Its observed change and the nominal per-operator multipliers are diagnostics; a complete monotonicity or asymptotic-stability proof remains open. The separate Dirichlet energy decreases exactly in its fixed symmetric diffusion regime.

6. **Numerical evaluation**: The current seeded protocol rejects the advertised charge-drift, sector-ratio and scaling-direction targets while its mean-quality band and sampled candidate-energy target pass. These finite outcomes do not establish an asymptotic law.

7. **Diagnostic application**: Conservation residuals flag patterns for review; canonical history validation decides grammar compliance.

---

## Implementation & Examples

### SDK Entry Points

```python
from tnfr.sdk import TNFR

net = TNFR.create(20).ring().evolve(5)
cons = net.conservation()            # ConservationReport
print(cons.summary())                # baseline; call again after real evolution
net.evolve(1)
cons = net.conservation(dt=1.0)     # sampled balance + candidate-energy trend
alerts = net.balance_alerts(dt=1.0) # residual alerts, no grammar verdict
```

### Executable Demonstrations

| Example | Concept from this document |
|---------|---------------------------|
| [17_conservation_law_demo.py](../examples/02_physics_regimes/17_conservation_law_demo.py) | Charge candidate, balance residuals, candidate-energy trend and static spectral proxy |
| [34_conservation_protocol_suite.py](../examples/02_physics_regimes/34_conservation_protocol_suite.py) | Finite reproduction of historical charge, quality, scaling and energy targets (§10) |
| [36_grammar_violation_detector.py](../examples/02_physics_regimes/36_grammar_violation_detector.py) | Limits of residual alerts and separate grammar validation (§12.1) |

### Key Source Modules

- `src/tnfr/physics/conservation.py` — structural-balance diagnostics and restricted helpers
- `src/tnfr/physics/integrity.py` — operator postconditions plus finite monitor alerts
- `src/tnfr/sdk/simple.py` — scoped `ConservationReport` and balance-alert API
- `src/tnfr/dynamics/self_optimizing_engine.py` — review-only consumption of
  balance and candidate-energy feedback

---

- **Status**: CANONICAL DIAGNOSTIC REFERENCE
- **Derived from**: field definitions and the measured balance identity; exact
  diffusion results retain their stated hypotheses
- **Checked by**: finite numerical experiments and the current test suite
- **Implementation**: `src/tnfr/physics/conservation.py`
