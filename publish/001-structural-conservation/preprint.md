# A Structural Continuity Law for Grammar-Constrained Dynamics in TNFR

**F. F. Martinez Gamo**

*Preprint — Version 1.0 — March 2026*

---

## Abstract

We present a structural continuity result for grammar-constrained dynamics
on finite graphs within the TNFR (Resonant Fractal Nature Theory) formalism.
Within TNFR, admissible operator sequences are restricted by a finite grammar
(rules U1–U6), and the resulting dynamics induce a charge-like structural
quantity together with an associated current. We formulate a Noether-like
continuity statement internal to the framework and derive a Lyapunov-type
energetic corollary for grammar-compliant operator evolution. To support the
formal result we provide a compact computational benchmark across five
fixed graph topologies with seeded initial conditions, comparing
grammar-compliant dynamics against perturbed and invalid regimes. The
experiments track structural charge drift, energy evolution, and stability
diagnostics, showing bounded conservation behaviour under valid sequences
and measurable degradation when grammatical constraints are violated. The
contribution of this work is intentionally narrow: it does not claim a
universal physical law, but rather establishes a mathematically explicit and
computationally reproducible conservation principle inside the TNFR
operator-grammar formalism. All code, scripts, and figure-generation steps
are included in a minimal Zenodo-ready package.

**Keywords**: structural conservation, grammar-constrained dynamics,
Noether-like theorem, Lyapunov stability, graph dynamics, TNFR.

---

## 1. Introduction

Conservation laws occupy a central role in mathematical physics. Noether's
theorem [2] establishes that every continuous symmetry of a Lagrangian system
gives rise to a conserved quantity. On continuous manifolds the result is
well understood; on discrete structures such as finite graphs the situation
is less clear, because the usual differential machinery must be replaced by
combinatorial analogues.

TNFR (Resonant Fractal Nature Theory) [1] is a computational formalism for
coherent pattern dynamics on finite graphs. In TNFR, every node carries a
phase $\phi_i$, a structural frequency $\nu_f$, and a pressure term
$\Delta\text{NFR}$, and evolves according to the nodal equation
$\partial\text{EPI}/\partial t = \nu_f \cdot \Delta\text{NFR}(t)$.
Admissible operator sequences are restricted by six grammar rules (U1–U6)
that enforce initiation/closure, convergence, phase-compatibility,
bifurcation control, multi-scale coherence, and potential confinement.

A natural question arises: **does the grammar constraint itself induce a
conservation-like structure?**  That is, if we define a charge-like quantity
from the structural fields and track it under grammar-compliant evolution,
does it remain approximately conserved—and does it degrade measurably when
the grammar is violated?

This paper answers affirmatively within the scope of the TNFR formalism.
We define a structural charge density $\rho = \Phi_s + K_\phi$ from two
tetrad fields (structural potential and phase curvature) and an associated
current $\mathbf{J} = (J_\phi, J_{\Delta\text{NFR}})$. We formulate a
continuity statement analogous to $\partial\rho/\partial t + \nabla_G \cdot
\mathbf{J} = \varepsilon_t$ and show computationally that:

- Under grammar-compliant evolution, $|\varepsilon_t|$ remains small
  (relative charge drift below 0.5 % on four of five test topologies).
- Under grammar violation, drift increases by approximately two to five
  orders of magnitude.
- An associated Lyapunov energy functional $E \geq 0$ is non-increasing
  on path, cycle, grid, and Erdős–Rényi topologies under valid dynamics.

**Scope.** This is an internal result of the TNFR formalism. We do not
claim equivalence with physical conservation laws, nor do we assert that the
structural charge corresponds to any quantity in conventional physics. The
contribution is narrowly defined: a computationally verifiable
conservation-like property that emerges from grammar constraints on
graph-coupled dynamics. All code, data, and reproduction instructions are
included.

## 2. Minimal Formalism

We summarise only the definitions required for the continuity statement.
The full TNFR specification is documented in [1].

### 2.1 TNFR Graph State

Let $G = (V, E)$ be a finite connected graph. Each node $i \in V$ carries:

- **Phase** $\phi_i \in [0, 2\pi)$: synchronisation parameter.
- **Structural frequency** $\nu_{f,i} > 0$: reorganisation rate (units: Hz_str).
- **Structural pressure** $\Delta\text{NFR}_i$: mismatch with coupled
  environment.
- **Primary information structure** $\text{EPI}_i$: coherent structural
  configuration.

The nodal equation governs all evolution:

$$\frac{\partial\,\text{EPI}_i}{\partial t} = \nu_{f,i} \cdot \Delta\text{NFR}_i(t)$$

### 2.2 Structural Field Tetrad

Four scalar fields characterise the structural state at every node. They
are computed from $\phi_i$ and $\Delta\text{NFR}_i$ on $G$:

| Field | Symbol | Order | Description |
|-------|--------|-------|-------------|
| Structural potential | $\Phi_s(i)$ | 0th | $\sum_{j \neq i} \Delta\text{NFR}_j / d(i,j)^2$ |
| Phase gradient | $\lvert\nabla\phi\rvert(i)$ | 1st | Local phase mismatch with neighbours |
| Phase curvature | $K_\phi(i)$ | 2nd | Discrete Laplacian of phase |
| Coherence length | $\xi_C$ | Non-local | Exponential decay scale of correlations |

Two transport currents complete the picture:
- $J_\phi$: geometric current (phase transport between neighbours).
- $J_{\Delta\text{NFR}}$: potential current ($\Delta\text{NFR}$ flux).

### 2.3 Unified Grammar U1–U6

TNFR dynamics are mediated by 13 canonical operators. Not every operator
sequence is admissible; six grammar rules constrain composition:

- **U1** (Initiation/Closure): sequences must begin with a generator
  $\{$AL, NAV, REMESH$\}$ and end with a closure
  $\{$SHA, NAV, REMESH, OZ$\}$.
- **U2** (Convergence): destabilisers $\{$OZ, ZHIR, VAL$\}$ require
  accompanying stabilisers $\{$IL, THOL$\}$ so that
  $\int |\nu_f \cdot \Delta\text{NFR}|\,dt$ converges.
- **U3** (Resonant Coupling): coupling operators $\{$UM, RA$\}$ require
  $|\phi_i - \phi_j| \leq \Delta\phi_{\max}$.
- **U4** (Bifurcation): triggers need handlers (U4a); transformers need
  recent destabiliser context (U4b).
- **U5** (Multi-scale Coherence): nested structures require stabilisers
  at each hierarchical level.
- **U6** (Potential Confinement): $|\Delta\Phi_s| < \varphi \approx 1.618$.

### 2.4 Charge and Current Definitions

We define the structural charge density and current as follows:

$$\rho(i, t) \;=\; \Phi_s(i, t) + K_\phi(i, t)$$

$$\mathbf{J}(i, t) \;=\; \bigl(J_\phi(i, t),\; J_{\Delta\text{NFR}}(i, t)\bigr)$$

The discrete divergence $\nabla_G \cdot \mathbf{J}$ is computed via the
graph Laplacian. The global Noether charge is
$Q(t) = \sum_{i \in V} \rho(i, t)$.

## 3. Structural Continuity Theorem

**Theorem 1** (Structural Continuity). *Let $G = (V, E)$ be a finite
connected graph with TNFR nodal state $\{\phi_i, \nu_{f,i},
\Delta\text{NFR}_i, \text{EPI}_i\}$. Let the evolution be governed by
grammar-compliant operator sequences satisfying U1–U6. Define the charge
density $\rho(i,t) = \Phi_s(i,t) + K_\phi(i,t)$ and current
$\mathbf{J}(i,t) = (J_\phi(i,t),\, J_{\Delta\text{NFR}}(i,t))$. Then:*

$$\frac{\partial\rho}{\partial t} + \nabla_G \cdot \mathbf{J} = \varepsilon_t$$

*where*

*(a) In the ideal grammar regime, $\varepsilon_t = 0$.*

*(b) In the computational grammar-compliant regime, $|\varepsilon_t| \leq
C \cdot \Delta t$ for a topology-dependent constant $C$.*

*(c) When grammar constraints are violated, $\varepsilon_t$ is unbounded.*

The theorem decomposes into two independent sector balances:

$$\frac{\partial\Phi_s}{\partial t} + \operatorname{div}(J_{\Delta\text{NFR}}) \approx 0
\qquad\text{(potential sector)}$$

$$\frac{\partial K_\phi}{\partial t} + \operatorname{div}(J_\phi) \approx 0
\qquad\text{(geometric sector)}$$

The total continuity equation is the sum of these two balances.

**Corollary 1** (Lyapunov energy). *Define the energy functional*

$$E(t) = \tfrac{1}{2}\sum_{i \in V}\bigl[\Phi_s(i)^2 + |\nabla\phi|(i)^2
+ K_\phi(i)^2 + J_\phi(i)^2 + J_{\Delta\text{NFR}}(i)^2\bigr]$$

*Then $E \geq 0$ by construction, and under grammar-compliant evolution
$dE/dt \leq 0$ (modulo numerical tolerance of order $\Delta t$).*

The corollary asserts that grammar-compliant dynamics dissipate structural
energy rather than amplify it. This is consistent with U2 (convergence),
which requires stabilisers to bound the integral of $\nu_f \cdot
\Delta\text{NFR}$.

## 4. Proof Sketch

We outline the argument in four steps. A fully formal proof would require
analytic bounds on the discrete Laplacian residuals, which we leave to
future work; the present treatment demonstrates the mechanism and is
supported by computational evidence (§ 6).

**Step 1: Two-sector decomposition.** The charge density
$\rho = \Phi_s + K_\phi$ splits into a *potential sector*
($\Phi_s$, $J_{\Delta\text{NFR}}$) and a *geometric sector*
($K_\phi$, $J_\phi$). Because $\Phi_s$ is computed from
$\Delta\text{NFR}$ via inverse-square summation and $K_\phi$ from the
discrete Laplacian of $\phi$, each sector couples to its own current
through the graph topology.

**Step 2: Grammar as a constraint symmetry.** The grammar rules U1–U6 do
not constitute a continuous symmetry in the Lagrangian sense. However, they
act as *structural constraints* on the admissible evolution:

- U2 bounds $\int |\nu_f \cdot \Delta\text{NFR}|\,dt$ by requiring
  stabilisers. This prevents unbounded growth of $\Phi_s$.
- U6 confines $|\Delta\Phi_s| < \varphi$, bounding the potential sector
  directly.
- U3 restricts coupling to phase-compatible nodes, limiting the rate at
  which $K_\phi$ can change through network interactions.

Together, these constraints limit the rate of change of both $\Phi_s$ and
$K_\phi$ at each time step.

**Step 3: Approximate continuity.** Given bounded $\partial\Phi_s/\partial t$
and $\partial K_\phi/\partial t$, and given that the currents $J_\phi$,
$J_{\Delta\text{NFR}}$ transport the same quantities through the graph
edges, the discrete balance

$$\frac{\partial\rho}{\partial t} + \nabla_G \cdot \mathbf{J} = \varepsilon_t$$

has residual $\varepsilon_t$ controlled by the grammar constraints. Under
exact grammar compliance, the source term vanishes identically. Under
computational (finite $\Delta t$) grammar compliance, the residual is
$O(\Delta t)$. When grammar is violated, the bounds from U2 and U6 no
longer hold and $\varepsilon_t$ can grow without limit.

**Step 4: Lyapunov corollary.** The energy functional
$E = \frac{1}{2}\sum_i [\Phi_s^2 + |\nabla\phi|^2 + K_\phi^2 + J_\phi^2
+ J_{\Delta\text{NFR}}^2]$ is non-negative by construction. Under
grammar-compliant evolution, U2 ensures that destabilisers are
counterbalanced by stabilisers, and U6 prevents potential escape. The net
effect is that the sum of squared field amplitudes does not increase,
yielding $dE/dt \leq 0$. This constitutes a Lyapunov stability condition
for the structural state.

**Caveat.** This argument is internal to TNFR. The "symmetry" is the
grammar constraint itself, not a continuous Lie symmetry. The analogy with
Noether's theorem is structural, not formal: grammar invariance plays the
role that Lagrangian invariance plays in classical mechanics. We make no
claim that this extends beyond the formalism.

## 5. Experimental Protocol

### 5.1 Graph Topologies

We test on five fixed graphs covering a range of structural properties:

| Label | Type | Nodes | Edges | Key property |
|-------|------|-------|-------|--------------|
| path  | Path graph | 20 | 19 | Linear, no cycles |
| cycle | Cycle graph | 20 | 20 | Single cycle, uniform degree |
| grid  | 2D grid | 25 (5×5) | 40 | Planar, regular lattice |
| tree  | Binary tree | 31 (depth 4) | 30 | Acyclic, hierarchical |
| erdos | Erdős–Rényi | 25 | 105 | Random, high connectivity |

All graphs are constructed with a fixed global seed (42) for the
random-graph case.

### 5.2 Experimental Arms

Each topology is tested under three regimes:

**(a) Valid.** A simplified grammar-compliant nodal evolution step:
phase advances by $\Delta t \cdot \nu_f \cdot \Delta\text{NFR} \cdot 0.1$
and $\Delta\text{NFR}$ relaxes diffusively toward the neighbourhood mean.
This is not a full 13-operator sequence but captures the core nodal
equation dynamics under U2-safe (stabilising) coupling.

**(b) Perturbed.** Valid evolution plus i.i.d. Gaussian noise
($\sigma = 0.01$) added to each node's phase after each step. The grammar
remains nominally satisfied but numerical precision is stressed.

**(c) Invalid.** Grammar rules U2 and U3 are deliberately violated:
$\Delta\text{NFR}$ is amplified at each step without stabilisation
(breaking U2 convergence), and 10 % of nodes receive random full-range
phase resets per step (breaking U3 phase compatibility).

### 5.3 Parameters

All runs use $\Delta t = 0.01$, $N_{\text{steps}} = 40$, and global
seed 42. Total experiment: $5 \times 3 = 15$ runs.

### 5.4 Metrics

For each run we record:

- **Relative charge drift**: $|Q(t_f) - Q(0)| / |Q(0)|$.
- **Mean conservation quality**: time-average of the RMS balance
  residual diagnostic (0 = poor, 1 = perfect).
- **Lyapunov stability %**: fraction of time steps with $dE/dt \leq 0$.
- **Mean $dE/dt$**: time-average of the Lyapunov derivative.

### 5.5 Reproduction

The entire experiment is executed by a single deterministic script:

```
python src/run_conservation_experiment.py
```

It produces `results/metrics.csv` and four figures in `results/figures/`.
Claim-level tests are run with `python -m pytest tests/ -v`.

## 6. Results

### Table 1: Per-topology summary

| Topology | Regime | Nodes | Rel. drift | Quality | Stable % | Mean $dE/dt$ |
|----------|--------|-------|-----------|---------|----------|-------------|
| path | valid | 20 | 0.13 % | 0.40 | 100 % | −0.079 |
| path | perturbed | 20 | 0.10 % | 0.36 | 57.5 % | −3.27 |
| path | invalid | 20 | 39.1 % | 0.07 | 40.0 % | +12.4 |
| cycle | valid | 20 | < 0.001 % | 0.42 | 100 % | −0.082 |
| cycle | perturbed | 20 | < 0.001 % | 0.37 | 60.0 % | −2.52 |
| cycle | invalid | 20 | 69.5 % | 0.07 | 42.5 % | +10.9 |
| grid | valid | 25 | 0.44 % | 0.51 | 100 % | −0.889 |
| grid | perturbed | 25 | 273 % | 0.34 | 67.5 % | −10.3 |
| grid | invalid | 25 | 833 % | 0.02 | 47.5 % | +28.7 |
| tree | valid | 31 | 3.38 % | 0.45 | 0 % | +0.691 |
| tree | perturbed | 31 | 6.74 % | 0.20 | 45.0 % | +8.31 |
| tree | invalid | 31 | 289 % | 0.01 | 47.5 % | −74.1 |
| erdos | valid | 25 | 0.41 % | 0.68 | 100 % | −0.505 |
| erdos | perturbed | 25 | 2.73 % | 0.34 | 35.0 % | +5.64 |
| erdos | invalid | 25 | 273 % | 0.02 | 50.0 % | +45.5 |

### Figures

- **Figure 1** (`charge_vs_time.png`): Noether charge $Q(t)$ over 40
  steps for each topology under valid dynamics. All traces remain within a
  narrow band around $Q(0)$.
- **Figure 2** (`energy_vs_time.png`): Lyapunov energy $E(t)$ under
  valid dynamics. Non-increasing on path, cycle, grid, and erdos; slightly
  increasing on tree.
- **Figure 3** (`valid_vs_invalid.png`): Side-by-side comparison of
  $Q(t)$ and $E(t)$ for the Erdős–Rényi topology in valid vs. invalid
  regimes.
- **Figure 4** (`topology_summary.png`): Bar chart of relative drift per
  topology, valid vs. invalid.

### Claim Assessment

**C1 (Charge conservation).** Relative drift under valid evolution is below
0.5 % on path, cycle, grid, and erdos. On tree the drift is 3.4 %, higher
but still roughly two orders of magnitude below the invalid regime
(289 %, ratio $\approx 85\times$). We therefore state:
*grammar-compliant dynamics preserve the structural charge to within a few
percent*. The tree topology represents a partial exception discussed in
§ 7.

**C2 (Lyapunov stability).** Under valid dynamics, 100 % of time steps
satisfy $dE/dt \leq 0$ on path, cycle, grid, and erdos. Tree shows 0 %
stability with a positive mean $dE/dt = +0.691$. The Lyapunov corollary
thus holds on four of five tested topologies; tree is excluded from this
claim.

**C3 (Conservation quality).** Mean quality ranges from 0.40 (path) to 0.68
(erdos) in the valid regime. All values exceed the test threshold of 0.30.

**C4 (Invalid degradation).** In every topology, the invalid regime shows
dramatically worse drift than the valid regime: factors of $\times$85
(tree) to $\times$420,000 (cycle). This confirms that grammar violation
destroys conservation.

**C5 (Reproducibility).** Re-running with the same seed produces identical
CSV output, verified by the test suite.

## 7. Scope and Limitations

### What this paper establishes

1. A charge-like quantity $\rho = \Phi_s + K_\phi$ and an associated current
   $\mathbf{J}$ satisfy an approximate continuity equation under
   grammar-compliant TNFR dynamics on finite graphs.
2. The approximation is tight on path, cycle, grid, and Erdős–Rényi
   topologies (drift $< 0.5\%$) and moderate on tree topologies
   (drift $\approx 3.4\%$).
3. Violation of grammar rules destroys conservation by approximately two to
   five orders of magnitude, confirming that the grammar is the operative
   constraint.
4. A Lyapunov energy functional is non-increasing under valid dynamics on
   four of five tested topologies.
5. The experiment is fully deterministic and reproducible from a single
   script.

### What this paper does NOT claim

- **Not a physical conservation law.** The structural charge $\rho$ is
  defined within the TNFR formalism. We make no assertion that it
  corresponds to energy, momentum, or any quantity in conventional physics.
- **Not a proof of Noether's theorem on graphs.** The analogy with Noether
  is structural: grammar constraints play the role of symmetries. A formal
  Noether-type proof for discrete grammar-constrained systems is an open
  problem.
- **No connection to the Riemann Hypothesis, cosmology, or other external
  conjectures.** The result is entirely internal to TNFR.
- **Not universal across all graph topologies.** The tree topology shows
  anomalous Lyapunov behaviour (positive $dE/dt$), indicating that acyclic
  hierarchical structures may require different treatment. We report the
  anomaly without attempting to explain it.

### Open questions

- **Scale dependence.** All tests use graphs with $N \leq 31$ nodes. The
  behaviour on larger graphs ($N > 100$) is untested.
- **Full operator sequences.** The experiment uses nodal evolution only, not
  arbitrary compositions of the 13 canonical operators. Conservation under
  full operator sequences remains to be verified.
- **Formal $O(\Delta t)$ bound.** The proof sketch argues that the
  continuity residual is $O(\Delta t)$ under grammar compliance; a rigorous
  analytical bound has not been derived.
- **Tree topology anomaly.** The positive Lyapunov derivative on acyclic
  topologies may reflect a genuine limitation of the current continuity
  formulation for graphs without cycles. Further investigation is needed.

## 8. Reproducibility

All materials required to reproduce the results are included in the
publication package and the public repository.

**Software:**
- Repository: https://github.com/fermga/TNFR-Python-Engine
- Package: `pip install tnfr` (version ≥ 0.0.3.3)
- Zenodo DOI: *to be assigned upon deposit*

**Reproduction in three commands:**

```bash
pip install -r requirements.txt
python src/run_conservation_experiment.py
python -m pytest tests/ -v
```

The experiment script produces `results/metrics.csv` (the data behind
Table 1) and four PNG figures in `results/figures/`. The test suite
verifies all five claims (C1–C5) against the generated data.

**Claim-to-test mapping:**

| Claim | Test class | Assertion |
|-------|-----------|-----------|
| C1 | `TestC1ChargeDrift` | Drift $< 5\%$ per topology |
| C2 | `TestC2Lyapunov` | Stable $\geq 80\%$ (path, cycle, grid, erdos) |
| C3 | `TestC3Quality` | Quality $> 0.30$ per topology |
| C4 | `TestC4InvalidDegradation` | Invalid drift $>$ valid drift |
| C5 | `TestC5Reproducibility` | Identical output across re-runs |

## References

[1] F. F. Martinez Gamo, "TNFR-Python-Engine," 2024–2026.
    https://github.com/fermga/TNFR-Python-Engine

[2] E. Noether, "Invariante Variationsprobleme," *Nachrichten von der
    Gesellschaft der Wissenschaften zu Göttingen*, pp. 235–257, 1918.

[3] Y. Kuramoto, *Chemical Oscillations, Waves, and Turbulence*,
    Springer-Verlag, 1984.

[4] A. Hagberg, D. Schult, and P. Swart, "Exploring network structure,
    dynamics, and function using NetworkX," in *Proceedings of the 7th
    Python in Science Conference*, 2008.

---

## Appendix A: Numbered Contributions

1. Explicit formulation of a **structural continuity law** within TNFR.
2. Operational definition of **structural charge density** ρ and **structural current** J.
3. **Lyapunov-type energy corollary** for grammar-compliant dynamics.
4. Minimal **reproducible benchmark** across five topologies with fixed seeds.
5. Explicit **valid vs invalid comparison** demonstrating conservation degradation.

## Appendix B: Notation Summary

| Symbol | Definition |
|--------|-----------|
| G = (V, E) | Finite connected graph |
| φ_i | Phase of node i |
| ν_f | Structural frequency (Hz_str) |
| ΔNFR | Structural pressure (nodal gradient) |
| EPI | Primary information structure |
| Φ_s | Structural potential field |
| \|∇φ\| | Phase gradient field |
| K_φ | Phase curvature field |
| ξ_C | Coherence length field |
| J_φ | Phase current (geometric transport) |
| J_ΔNFR | ΔNFR flux (potential transport) |
| ρ = Φ_s + K_φ | Structural charge density |
| J = (J_φ, J_ΔNFR) | Structural current vector |
| E | Lyapunov energy functional |
| C(t) | Total coherence |
| U1–U6 | Unified grammar rules |
