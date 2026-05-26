# TNFR–Navier–Stokes Program Memo

**Status**: Exploratory research (non-canonical)
**Version**: 0.2.0 (May 2026, post-N11 + REMESH global reframe)
**Owner**: `theory/TNFR_NAVIER_STOKES_RESEARCH_NOTES.md`
**Predecessor program**: TNFR–Riemann (paused at T-HP, May 2026) — see `theory/TNFR_RIEMANN_RESEARCH_NOTES.md`

---

This memo opens the TNFR attack on the **Clay Millennium Problem: Existence and Smoothness of solutions of the 3D incompressible Navier–Stokes equations**. It scopes the program, fixes the structural translation between fluid variables and TNFR tetrad fields, and defines the milestones N1–N11 (DONE) and N≥12 (open queue). As with the Riemann program, **canonicity is a property of derivations from the nodal equation `∂EPI/∂t = νf · ΔNFR(t)`**, not of this document.

## 0. Honest Scope

- **What we do NOT claim**: We do **not** claim to prove global existence and smoothness of 3D Navier–Stokes, nor to construct a finite-time blow-up. Both directions of the Clay statement remain **OPEN**. All NS-G1..G5 gaps remain **OPEN** after N1–N11.
- **What we DO claim is reasonable**: TNFR provides a natural structural language (graph-Laplacian dynamics + tetrad fields + Lyapunov-style conservation theorem) into which the Beale–Kato–Majda (BKM) criterion, Leray energy inequality, and Constantin–Fefferman geometric depletion translate explicitly. N2–N11 demonstrate the discrete operator behaves correctly under the integrable Taylor–Green vortex across resolutions n∈{8..32} and viscosities ν∈{0.05..0.005}, with INCOMP enforcement at machine precision and CF alignment statistics responding monotonically to Re. None of this constitutes a closure of any Clay direction.
- **Cross-program working hypothesis (§11, May 2026)**: the residual obstruction in NS-G_blowup and the residual obstruction in Riemann T-HP may share the same structural type — **asymptotic specialisation of the existing canonical REMESH global operator** (τ→∞ for Riemann, scale→0 for NS). This pushes the analysis toward **branch B1** (closeable within the 13-operator catalog via analytical derivation of the REMESH asymptotic limit) rather than branch B2 (new 14th operator needed). See §11 for the full reframe. This is a **working hypothesis**, not a proof.
- **Lessons imported from Riemann**: surface attack with structural diagnostics, never declare closure without an audit of the gap, separate operator-level results from continuum claims, and treat the limit graph → continuum as a real obstruction (analogous to the smooth↔oscillatory split of operator 𝓕).

## 1. Purpose and Scope

- Translate the 3D incompressible Navier–Stokes Clay problem into TNFR constructs: discrete velocity/vorticity fields on a graph, structural pressure field, viscous dissipation as ΔNFR-driven relaxation, and blow-up as failure of U2 (CONVERGENCE & BOUNDEDNESS) on the tetrad.
- Maintain reproducible sandboxes (3D regular and irregular graphs, vortex initial data, spectral and tetrad telemetry) that connect Leray-style energy estimates to the TNFR Structural Conservation Theorem.
- Document how canonical operators (AL, EN, IL, OZ, UM, RA, THOL) and the nodal equation compose to form a discrete Navier–Stokes-like flow whose limit is to be analysed.

## 2. The Classical Problem (Clay Statement)

Let $u : \mathbb{R}^3 \times [0, \infty) \to \mathbb{R}^3$ and $p : \mathbb{R}^3 \times [0, \infty) \to \mathbb{R}$ satisfy:

$$
\partial_t u + (u \cdot \nabla) u = \nu \Delta u - \nabla p, \qquad \nabla \cdot u = 0, \qquad u(\cdot, 0) = u_0,
$$

with $\nu > 0$ and $u_0$ smooth, divergence-free, decaying. **Clay (A)**: prove existence of a globally smooth solution. **Clay (B)**: or exhibit smooth $u_0$ producing finite-time blow-up.

**Equivalent characterisations of blow-up** (classical):
- **BKM (1984)**: solution remains smooth on $[0, T]$ iff $\int_0^T \lVert \omega(\cdot, t) \rVert_{L^\infty} \, dt < \infty$, where $\omega = \nabla \times u$.
- **Leray (1934)**: weak solutions exist globally and satisfy $\tfrac{1}{2}\lVert u(\cdot, t) \rVert_{L^2}^2 + \nu \int_0^t \lVert \nabla u \rVert_{L^2}^2 \, ds \le \tfrac{1}{2}\lVert u_0 \rVert_{L^2}^2$ (energy inequality, with equality unknown for strong sols).
- **Constantin–Fefferman (1993)**: geometric depletion — alignment of vorticity direction in regions of high $\lVert \omega \rVert$ prevents singularities.

These three pillars are the targets of the TNFR translation in §3.

## 3. TNFR Structural Translation

### 3.1 Field dictionary

| Navier–Stokes object | TNFR analogue | Module / notes |
|---|---|---|
| Velocity $u_a(x, t)$, $a = 1,2,3$ | Three phase fields $\phi^{(a)}_i(t)$ on graph $G$, one per Cartesian component | `physics/fields.py` extended |
| Vorticity $\omega = \nabla \times u$ | Phase curvature triplet $K^{(a)}_\phi$ + cross-component coupling | `K_phi` per component |
| Pressure $p(x, t)$ | Structural potential $\Phi_s(i, t)$ enforcing $\nabla \cdot u = 0$ | `Phi_s` constraint mode |
| Viscosity $\nu$ | Structural frequency normalisation: relaxation rate of $\Delta\mathrm{NFR}$ feedback | Maps to U2 stabiliser strength |
| Reynolds number $\mathrm{Re}$ | Ratio (driving $\Delta\mathrm{NFR}$ amplitude) / (IL coherence rate) | Telemetry-derived |
| Energy $\tfrac{1}{2}\lVert u \rVert_{L^2}^2$ | Tetrad energy density $\mathcal{E} = \Phi_s^2 + |\nabla\phi|^2 + K_\phi^2 + J_\phi^2 + J_{\Delta\mathrm{NFR}}^2$ summed over components | `physics/conservation.py` |
| Enstrophy $\lVert \omega \rVert_{L^2}^2$ | $\sum_a \sum_i (K^{(a)}_\phi(i))^2$ | Direct |
| Helicity $\int u \cdot \omega$ | Topological charge $\mathcal{Q} = \lvert\nabla\phi\rvert \cdot J_\phi - K_\phi \cdot J_{\Delta\mathrm{NFR}}$ | `physics/conservation.py` (already defined) |
| Incompressibility $\nabla \cdot u = 0$ | Constraint $\sum_a \nabla_a \phi^{(a)} = 0$ enforced by $\Phi_s$ as Lagrange multiplier | New operator: `INCOMP` (candidate, §6) |

### 3.2 Equation translation

The incompressible NS equation has the structural form:

$$
\partial_t u = - \underbrace{(u \cdot \nabla) u}_{\text{advective transport}} - \underbrace{\nabla p}_{\text{pressure}} + \underbrace{\nu \Delta u}_{\text{viscous dissipation}}.
$$

Each term has a candidate TNFR analogue in the nodal equation $\partial_t \mathrm{EPI} = \nu_f \cdot \Delta\mathrm{NFR}$:

- **Viscous dissipation $\nu \Delta u$**: graph Laplacian acting on each component $\phi^{(a)}$, contributing to $\Delta\mathrm{NFR}^{(a)}$ as a restoring term. Maps to IL (Coherence) operator at variable strength $\nu_f \propto \nu$.
- **Pressure $-\nabla p$**: gradient of $\Phi_s$ acting back on $\Delta\mathrm{NFR}$ to enforce divergence constraint. Maps to U6 (structural potential confinement) in active (not just telemetric) mode.
- **Advection $(u \cdot \nabla) u$**: nonlinear self-coupling. Maps to UM (Coupling) and RA (Resonance) with state-dependent edge weights $w_{ij}(t) \propto \phi^{(a)}(i) \cdot$ (geometry factor).
- **External forcing** (if any): AL (Emission) localised in space.

### 3.3 Three regimes (mirror the regime correspondences in `AGENTS.md`)

- **Laminar regime** ($\mathrm{Re} \to 0$, equivalently IL dominates UM·RA): tetrad evolution becomes near-linear; energy decays monotonically; Lyapunov bound is tight; corresponds to classical existence proofs in Sobolev spaces of low regularity that are already known.
- **Transitional regime**: bifurcations triggered by OZ/ZHIR analogues (vortex stretching). U4a/U4b structures the bifurcation cascade.
- **Turbulent regime** ($\mathrm{Re} \to \infty$): hierarchy of nested EPIs (Kolmogorov cascade ≅ U5 multi-scale coherence). The blow-up question becomes: does the cascade terminate at a finite scale (smooth) or generate unbounded $K_\phi$ at infinitesimal scale (blow-up)?

## 4. Program Objectives

### 4.1 N1 — Structural Formalisation (this memo)

- Fix the field dictionary (§3.1) and equation translation (§3.2). **Status: DONE** (this document).
- No code, no claims. Establishes vocabulary for N2–N∞.

### 4.2 N2 — Discrete TNFR–NS Operator

- Construct $\mathcal{N}_{TNFR}$: a flow on the graph $G$ acting on the vector of per-component phases $\Phi = (\phi^{(1)}, \phi^{(2)}, \phi^{(3)})$, divergence constraint enforced.
- Verify on simple cases (2D Taylor–Green vortex on torus graph, Beltrami flow on regular 3D lattice) that the discrete flow tracks the continuum solution to expected order in mesh size.
- Telemetry: tetrad fields per component + composite enstrophy + composite helicity.
- Target module: `src/tnfr/navier_stokes/operator.py`.

### 4.3 N3 — Discrete Leray Energy Inequality

- Prove discretely (via the Structural Conservation Theorem `physics/conservation.py`): tetrad energy $\mathcal{E}(t) + \int_0^t \mathcal{D}(s) \, ds \le \mathcal{E}(0)$ with $\mathcal{D}$ the viscous dissipation rate (analogue of $\nu \lVert \nabla u \rVert_{L^2}^2$).
- This is the **discrete a priori bound**. Already plausible from Lyapunov stability $dE/dt \le 0$ under grammar compliance.
- Identify whether the bound is uniform in mesh refinement.

### 4.4 N4 — Discrete BKM Criterion

- Formulate: discrete solution remains "smooth" on $[0, T]$ iff $\int_0^T \max_i |K_\phi(i, t)|_\infty \, dt < \infty$.
- The conjecture (TNFR analogue of BKM): blow-up of $\mathcal{N}_{TNFR}$ ⟺ violation of U2 driven by unbounded $K_\phi$ accumulation.
- Test numerically on initial data known to be near critical (e.g. axisymmetric flows with swirl, vortex tube reconnection).

### 4.5 N5 — Geometric Depletion (Constantin–Fefferman analogue)

- Quantify alignment of $K_\phi$ direction across neighbouring nodes via $J_\phi$ (transport current). If alignment correlates with bounded growth of $\int |K_\phi|_\infty$, this is the TNFR realisation of the geometric depletion mechanism.
- The chirality field $\chi = |\nabla\phi| \cdot K_\phi - J_\phi \cdot J_{\Delta\mathrm{NFR}}$ (already defined in `physics/fields.py`) is the natural alignment diagnostic.

## 5. Gap Analysis (mirroring Riemann §7 / §13)

Anticipated obstructions, listed honestly before any work is done:

| Gap | Description | Severity (a priori) |
|---|---|---|
| **NS-G1** | Continuum limit of discrete TNFR–NS flow: does the sequence of discrete solutions converge (in what topology?) to a Leray weak solution as mesh → 0? | High — analogous to Riemann G4 in scope |
| **NS-G2** | Uniformity of discrete energy inequality in mesh refinement | Medium — likely tractable from Lyapunov + conservation theorem |
| **NS-G3** | Discrete BKM ⟺ continuum BKM: even if the discrete criterion is sharp, transferring to continuum requires NS-G1 | High — coupled to NS-G1 |
| **NS-G4** | Vortex stretching term $(\omega \cdot \nabla) u$ has no immediate TNFR analogue beyond OZ/ZHIR — needs explicit construction | Medium |
| **NS-G5** | 3D vs 2D dimensional barrier: 2D NS has known global smoothness, 3D is open. The TNFR translation should reproduce this asymmetry. If it does not, the translation is wrong | High — diagnostic, falsifiability check |

**Honest expectation**: NS-G1 and NS-G3 are the structural analogues of Riemann's T-HP residual. They may not close inside the canonical 13-operator catalog. The Clay problem itself is open precisely because the continuum limit (or its blow-up) is hard; TNFR offers a new vocabulary, not a free pass.

## 6. New Canonical Operator Candidate: INCOMP

Discussed but **not promoted** until N2 forces the question:

- **INCOMP** (Incompressibility): enforces $\sum_a \nabla_a \phi^{(a)} = 0$ on the per-component phase fields. Acts as instantaneous projection (analogous to Leray projector $\mathbb{P}$).
- **Physics**: derivable from the nodal equation if we treat $\Phi_s$ as a Lagrange multiplier on the divergence constraint, with infinite relaxation rate.
- **Decision**: do **not** add a 14th canonical operator unless N2 demonstrates the projection cannot be composed from {IL, UM, RA, Φ_s monitoring} acting in sequence. The Riemann program lived inside the 13 catalog throughout; that constraint protected canonicity.

## 7. Workflow Expectations

1. **Model definition** — Choose graph topology (torus, regular cube, irregular tetrahedral mesh), Reynolds-analogue, seeds, operator stack; record in `results/navier_stokes/configs/*.json`.
2. **Operator execution** — Use SDK helpers (forthcoming `TNFRNavierStokesOperator`) to evolve the per-component phase fields while logging the full tetrad per component plus composite invariants.
3. **Analysis** — Compute enstrophy, helicity, Lyapunov derivative, and $\int |K_\phi|_\infty \, dt$ trajectories. Scripts in `scripts/navier_stokes/`.
4. **Benchmark enforcement** — `benchmarks/navier_stokes_program.py` regresses energy inequality and BKM-analogue across mesh sizes.
5. **Validation** — Targeted tests (`tests/test_navier_stokes_operator.py`) ensure seed determinism, grammar compliance, divergence-freeness, and conservation residuals.

## 8. Telemetry & Reproducibility

- Log per component $(a = 1, 2, 3)$: $\Phi_s^{(a)}, |\nabla\phi|^{(a)}, K_\phi^{(a)}, \xi_C^{(a)}, J_\phi^{(a)}, J_{\Delta\mathrm{NFR}}^{(a)}$.
- Composite: tetrad energy $\mathcal{E}$, enstrophy, helicity / topological charge $\mathcal{Q}$, divergence residual $\lVert \sum_a \nabla_a \phi^{(a)} \rVert$, Lyapunov rate $dE/dt$.
- Artifact manifests record Python version, `tnfr` hash, graph spec, initial-data spec.

## 9. Cross-References

### Existing TNFR infrastructure leveraged

- `src/tnfr/physics/fields.py` — Structural Field Tetrad (per-component extension required for N2).
- `src/tnfr/physics/conservation.py` — Structural Conservation Theorem (N3 builds directly on this).
- `src/tnfr/physics/variational.py` — Hamiltonian/Lagrangian framework (candidate for variational NS).
- `src/tnfr/physics/gauge.py` — Gauge structure (relevant once vorticity dynamics get nontrivial).
- `src/tnfr/operators/grammar.py` — U1–U6 enforcement (N3 and N4 depend on U2 in particular).
- `src/tnfr/operators/definitions.py` — 13 canonical operators.
- `src/tnfr/dynamics/self_optimizing_engine.py` — auto-optimisation may help locate critical initial data.

### Forthcoming modules (planned, not yet created)

- `src/tnfr/navier_stokes/__init__.py`
- `src/tnfr/navier_stokes/operator.py` — N2
- `src/tnfr/navier_stokes/energy_inequality.py` — N3
- `src/tnfr/navier_stokes/bkm_criterion.py` — N4
- `src/tnfr/navier_stokes/geometric_depletion.py` — N5
- `src/tnfr/navier_stokes/telemetry.py`
- `examples/77_navier_stokes_taylor_green_demo.py` — N2 demo
- `benchmarks/navier_stokes_program.py`

## 10. Milestone Summary Table

| Milestone | Title | Status | Commit | Deliverable |
|---|---|---|---|---|
| N1 | Structural Formalisation | **DONE** | `a5eadd7c` | this memo |
| N2 | Discrete TNFR–NS Operator (viscous half) | **DONE** | `e83163b8` | `src/tnfr/navier_stokes/operator.py` + `examples/77` |
| N3 | Discrete Leray Energy Inequality (Strang splitting + skew-symm advection) | **DONE** | `a344c050` | `operator.py::leray_budget`/`advection` + `examples/78` |
| N4 | Discrete BKM Criterion (2D vorticity) | **DONE** | `7b249ae5` | `operator.py::vorticity_2d`/`bkm_budget` + `examples/79` |
| N5 | INCOMP activation (Leray–Helmholtz projection) | **DONE** | `da5689fb` | `operator.py::project_incompressible`/`pressure_field` + `examples/80` |
| N6 | Geometric depletion (Constantin–Fefferman, 3D) | **DONE** | `f971468e` | 3D torus + `vorticity_3d`/`vortex_stretching_field`/`stretching_production` + `examples/81` |
| N7 | Continuum-limit convergence study (NS-G1 precursor) | **DONE** | `45f741eb` | `examples/82` (3D TG mesh refinement n∈{8,12,16,24}) |
| N8 | Uniform-in-h discrete Leray energy inequality (NS-G2 precursor) | **DONE** | `e656cb04` | `examples/83` (3D TG, dimensional correction `D_phys = h · D_raw`) |
| N9 | Vortex-stretching alignment & geometric depletion (NS-G4 precursor) | **DONE** | `fbbcaa34` | `examples/84` (3D TG n=16, strain eigenframe, CF alignment, depletion ratio) |
| N10 | 2D vs 3D dimensional asymmetry falsifiability (NS-G5 precursor) | **DONE** | `1fac358b` | `examples/85` (2D TG + 3D operator on 2D-embedded IC + 3D TG, side-by-side) |
| N11 | Reynolds sweep — CF alignment vs viscosity (NS-G4 precursor) | **DONE** | `afc65b49` | `examples/86` (3D TG n=24, ν∈{0.05,0.02,0.01,0.005}, Re_eff 126→1257) |
| N12 | REMESH-∞ asymptotic limit on K_φ cascade (§11 test, NS-G_blowup branch B1) | **DONE — STRUCTURAL_EFFECT_MONOTONE** (per locked §12.7 mapping; non-monotone in BKM and stretching observables, see §13.4). Pre-registration: `c8900fce`. | n = 16 single-resolution probe of §11 working hypothesis: supports REMESH-global having structural traction on NS K_φ cascade (99.5 % response in peak stretching across τ_g sweep). Does NOT close NS-G1..G5. | §12 + §13 + `benchmarks/remesh_infinity_navier_stokes_3d_taylor_green.py` |
| N13 | REMESH-∞ resolution extension with refined F3 (n∈{24,32}; N12 follow-up) | **DONE — STRUCTURAL_EFFECT_MONOTONE at both n=24 and n=32; CROSS_RES_CONSISTENT=True** (per locked §14.7 mapping; refined F3 MONOTONE driven by `peak_enstrophy_post_t1`; BKM and stretching post-t1 remain non-monotone with stretching COLLAPSE sign agreeing with N12 at all three resolutions n∈{16,24,32}). Pre-registration: `4ab97bc8`. | Two-resolution structural-compatibility check: REMESH-global retains the N12 stretching-collapse signature under one full resolution doubling (n=16→32, 8× per-step cost) under a refined F3 that excludes the IC-dominated `peak_vorticity_sup`/`peak_enstrophy` flagged in §13.4. Does NOT close NS-G1..G5. | §14 + §15 + `benchmarks/remesh_infinity_navier_stokes_3d_taylor_green_n24_n32.py` |
| N14 | Higher-Re CF eigenframe sweep at n=48 (NS-G4 precursor; N11 resolution-doubling extension) | **DONE — CF_EIGENFRAME_TRANSITION_NOT_OBSERVED_AT_REEFF_3142; RE_TREND_CONSISTENT_WITH_N11** (per locked §16.7 mapping; F1–F3,F5 PASS, F4 NOT satisfied — operator remains in laminar/transitional regime at Re_eff≈500; <P>(Re) monotonicity and cross-resolution consistency retained). Pre-registration: `39e4b1b9`. | Single-resolution (n=48) Reynolds sweep extending N11 (n=24, ν∈{0.05..0.005}) by one resolution doubling AND one new ν step (ν=0.002 → Re_eff≈500, unreachable at n=24 due to Kolmogorov). Settles: (a) operator does NOT exhibit CF-canonical e_λ2 dominance at Re_eff≈500 on n=48 Taylor–Green; (b) <P>(Re) monotonic ascent and RE_TREND survives doubling. Does NOT close NS-G1..G5; redirects to n≥256 forced-isotropic DNS for full-turbulent CF statistics. | §16 + §17 + `benchmarks/higher_re_navier_stokes_3d_taylor_green_n48.py` |
| N15 | REMESH-∞ Asymptotic Operator (TNFR Core Structure; Branch analysis A/B1/B2/B3) | **PRE-REGISTERED** (atomic with §18 pre-reg; no data observed; research direction pivot from Clay problems to TNFR fundamentals) | Theoretical derivation: functional analysis of REMESH limit, conservation laws, spectrum. Three weeks locked. Independent of RH/NS truth. | §18 + (forthcoming analytical results) + (no benchmark code for this phase) |
| N≥16 | Function-space convergence (NS-G1) **/** DNS frontier n≥256 (CF alignment) **/** analytical NS-G2 bounds **/** discrete-to-continuum BKM (NS-G3) **/** structural TNFR construction of (ω·∇)u (NS-G4 closure) | open | unknown | unknown |

All NS-G1..G5 gaps remain **OPEN** after N1–N11. See §11 for the cross-program reframe of the residual obstruction.

---

## 11. REMESH Global Reframe (May 2026 — Cross-Program Discovery)

### 11.1 Origin

During discussion of the NS-G_blowup obstruction (does the K_φ cascade terminate at finite scale or generate unbounded K_φ at infinitesimal scale?), an initial analysis suggested that NS and Riemann T-HP might share the same residual structural obstruction, possibly requiring a new canonical operator (branch B2 of §13septies of the Riemann research notes). A direct audit of the canonical engine refuted the premise:

```python
# src/tnfr/config/defaults_core.py
REMESH_TAU_LOCAL: int = 4       # local temporal memory (per-node)
REMESH_TAU_GLOBAL: int = 8      # GLOBAL temporal memory (graph-wide)
REMESH_ALPHA: float = 0.5       # mixing weight EPI(t) ↔ EPI(t-τ)
REMESH_MODE: str = "knn"        # also "mst" and "community" (genuinely global)
```

and `src/tnfr/ontosim.py` explicitly comments `# Global REMESH memory` when allocating a graph-level deque `_epi_hist` of size `2·τ_global + 5`. The canonical REMESH operator (`src/tnfr/operators/remesh.py`) documents three structural modes:

* **REMESH Hierarchical** (central → periphery): IL/VAL/SHA/NUL compositions,
* **REMESH Rhizomatic** (decentralised, no fixed centre): OZ/UM/THOL compositions,
* **REMESH Fractal Harmonic** (scale-symmetric, perfect self-similarity): RA/NAV/AL/EN compositions.

Additionally, `src/tnfr/multiscale/hierarchical.py` implements explicit cross-scale ΔNFR coupling (`_apply_cross_scale_coupling`, `_compute_cross_scale_synchrony`).

**Implication**: the 13-operator catalog already contains an explicit global, multi-scale closure primitive (REMESH global + REMESH Fractal Harmonic + cross-scale coupling). The previously-asserted "no operator handles the asymptotic limit" was incorrect; what is missing is the **canonical asymptotic specialisation of the existing REMESH global operator** (REMESH-∞), which is an analytical operation on an existing canonical operator, not a new canonical primitive.

### 11.2 Reframed B1/B2 analysis

| | Riemann T-HP residual | NS-G_blowup residual |
|---|---|---|
| **Smooth half** | closed by P28/P30 (density + operator level) for the smooth zero distribution → corresponds to REMESH at **finite τ_global** | corresponds to bounded K_φ cascade at **finite scale** (validated empirically by N6–N11 at Re_eff ≤ 1300) |
| **Residual** | oscillatory half S(T) = (1/π) arg ζ(½+iT) — RH-equivalent | K_φ cascade behaviour at **scale → 0** (n → ∞ continuum limit / Kolmogorov dissipation scale) |
| **Asymptotic limit needed** | REMESH-global applied to prime-ladder spectrum {k log p} at **τ → ∞** | REMESH-global applied to spatial scale hierarchy at **scale → 0** (and equivalently REMESH Fractal Harmonic at vanishing self-similar scale) |
| **Branch classification** | **B1** if the τ→∞ limit of canonical REMESH-global reproduces S(T) | **B1** if the scale→0 limit of canonical REMESH-global / Fractal Harmonic + cross-scale coupling reproduces or bounds the cascade |

### 11.3 What this changes

* **The hypothesis is upgraded from "new operator may be needed"** (branch B2, open and uncertain) **to "existing operator needs canonical asymptotic specialisation"** (branch B1, a well-defined analytical problem on an existing operator).
* **Both Clay problems remain OPEN**. Branch B1 is not a closure; it is a more precise statement of where the work has to happen.
* **Roadmap consequence for NS**: the open queue at N≥12 acquires a new candidate milestone — **empirical and analytical study of the canonical REMESH global asymptotic limit applied to the K_φ cascade**, with explicit comparison against the K_φ behaviour observed in N6–N11. This is independent of the higher-Re DNS milestone (which still requires n ≥ 48–256) and can proceed at moderate resolution.

### 11.4 Honest scope of §11

* **What §11 claims**: a structural reframe of where the residual obstruction lives, anchored in the existing canonical TNFR engine (`REMESH_TAU_GLOBAL`, `_epi_hist`, REMESH modes, `multiscale/hierarchical.py`).
* **What §11 does NOT claim**: it does NOT prove that the asymptotic limit of REMESH global closes either T-HP or NS-G_blowup. It does NOT prove RH or 3D NS global regularity. It does NOT promote any new canonical operator. It is a **working hypothesis** (branch B1) that re-orders the open queue.
* **Cross-reference**: this reframe is mirrored in `theory/TNFR_RIEMANN_RESEARCH_NOTES.md` §13vicies-novies (added simultaneously). Both programs share the same canonical REMESH global infrastructure; the analytical study of its asymptotic limit is shared work.

---

**Next action**: open milestone **N12** — analytical study of `REMESH-∞` (the canonical asymptotic limit of REMESH global) applied to the K_φ cascade observed in N6–N11. Success criterion: produce either an analytical bound on K_φ at scale → 0 derivable from canonical REMESH global semantics, or an empirical demonstration on 3D Taylor–Green at n ∈ {16, 24, 32} that the cascade behaviour under REMESH-global with τ_global → ∞ qualitatively matches (or fails to match) the classical Kolmogorov dissipation regime. This is **NOT** a closure of NS-G_blowup or any other gap; it is the analytical test of the §11 working hypothesis.

---

## 12. N12 — REMESH-∞ Pre-Registration (May 2026)

### 12.1 Status and discipline

**Status**: PRE-REGISTERED, not yet executed. This section locks methodology, parameters, seed, F-criteria, and verdict mapping *before* the benchmark is run, mirroring the pre-registration discipline used in the Riemann program (§13vicies-novies.12, §13vicies-novies.14 of `theory/TNFR_RIEMANN_RESEARCH_NOTES.md`).

The benchmark skeleton ships in the same commit as this section. Execution and Results land in a separate commit (§13).

### 12.2 Working hypothesis under test

The §11 cross-program reframe asserts that the residual obstruction in NS-G_blowup is the **scale → 0 asymptotic limit of canonical REMESH global** applied to the spatial-scale hierarchy of the K_φ cascade. The N6–N11 milestones validated the discrete TNFR-NS operator at finite Re ∈ [126, 1257] on a 3D Taylor-Green vortex; the K_φ cascade is bounded and dissipates correctly. The question for N12 is whether REMESH-global mixing, applied between NS time-steps with τ_global → ∞, **affects** the cascade in a structurally meaningful way.

Three possible outcomes (all informative, no preferred verdict):

1. **STRUCTURAL_EFFECT_MONOTONE**: K_φ cascade observables respond monotonically to τ_g sweep. Supports §11 working hypothesis (canonical REMESH-global is a non-trivial regulator of the cascade at finite n).
2. **STRUCTURAL_EFFECT_NON_MONOTONE**: cascade responds but non-monotonically. Compatible with §11 hypothesis but suggests resonant/interference regimes inside the τ_g sweep.
3. **NULL_RESULT**: cascade observables flat (< 5% relative change) across full τ_g sweep including τ_g → ∞. **Falsifies §11 working hypothesis at n=16**, redirecting the program toward the higher-resolution / DNS milestone (n ≥ 48) or a non-product canonical lift inside the catalog.

### 12.3 Canonical mapping of REMESH-∞ to per-component velocity fields

The canonical engine (`src/tnfr/operators/remesh.py::apply_network_remesh`) implements the two-stage mix on graph-wide EPI:

```
mixed = (1 − α) · EPI(t) + α · EPI(t − τ_local)
mixed = (1 − α) · mixed + α · EPI(t − τ_global)
```

with `REMESH_ALPHA = 0.5`, `REMESH_TAU_LOCAL = 4`, `REMESH_TAU_GLOBAL = 8` as canonical defaults (`src/tnfr/config/defaults_core.py`). For the N12 sweep we adopt the *one-stage global* form (τ_local set equal to τ_global to collapse to a single global lag, matching the asymptotic limit being probed):

```
φ^(a)_i(t) ← (1 − α) · φ^(a)_i(t) + α · φ^(a)_i(t − τ_g)        ∀ a ∈ {1,2,3}, ∀ i ∈ G
```

with the per-component velocity field playing the role of EPI on the periodic 3D torus graph. Three operational regimes:

* **τ_g = 0**: no mixing — pure NS evolution baseline (must reproduce N6/N11 at n=16, ν=0.01).
* **τ_g ∈ {8, 32, 128}**: finite REMESH-global lag in step units. At dt=0.005 these correspond to physical lags 0.04, 0.16, 0.64 (i.e., 4%, 16%, 64% of T_final = 1.0).
* **τ_g = ∞ (lag-to-initial-condition)**: reference state is φ^(a)(t=0). Canonical realisation of the **REMESH global asymptotic limit** when the history window is large enough to contain the initial condition.

Mixing is applied **after** the full Strang-split NS step (advection + Crank-Nicolson viscous + advection) and **before** the next step. Because mixing is a linear combination of two divergence-free states at α=0.5, the result is divergence-free in continuous arithmetic; defensive INCOMP projection is applied after mixing to absorb round-off (consistent with N5/N6 INCOMP discipline).

The seed `20260526` is reused from the Riemann session (no stochastic RNG appears in this benchmark — TG is deterministic, INCOMP is FFT-deterministic — but the seed labels the session for reproducibility, matching the cross-program protocol).

### 12.4 Configuration (locked)

| Parameter | Value | Justification |
|---|---|---|
| Grid resolution | n = 16 | Smallest 3D TG sweep size used in N9 (`examples/84`) where CF eigenframe statistics are already well-defined; keeps τ_g sweep tractable in single run. |
| Viscosity | ν = 0.01 | Mid-sweep value from N11 (Re_eff ≈ 628). Cascade is non-trivial (advection visible in K_φ) but resolved at n=16. |
| Time step | dt = 0.005 | Same as N11 (`examples/86`). |
| Final time | T = 1.0 | 200 steps. Same as N11. Long enough for cascade to develop, short enough for τ_g=128 to fit inside the simulation window. |
| Amplitude | A = 1.0 | Canonical TG amplitude used throughout N6–N11. |
| Advection | ON | Required to exercise vortex stretching (the K_φ cascade source). |
| INCOMP | ON | Standard after N5. Re-applied after each REMESH-∞ mix. |
| α (REMESH mixing) | 0.5 | Canonical default (`config/defaults_core.py::REMESH_ALPHA`). |
| τ_g sweep | {0, 8, 32, 128, ∞} | 5 points covering the asymptotic limit. |
| Seed label | 20260526 | Cross-program session continuity. |

### 12.5 Observables (logged at every step)

For each τ_g value:

* `time[k]` — physical time at step k
* `vorticity_sup[k]` — `||ω(t_k)||_∞` (BKM integrand)
* `bkm_integral[k]` — `∫_0^{t_k} ||ω||_∞ dτ` (trapezoidal)
* `enstrophy[k]` — `||ω||_{L²}²`
* `kinetic_energy[k]` — `(1/2)·||u||_{L²}²`
* `dissipation[k]` — `ν · ⟨φ, L φ⟩`
* `divergence[k]` — `||∇·u||_{L²}` (must stay ≤ 1e-8 after INCOMP)
* `stretching_production[k]` — `⟨ω, (ω·∇)u⟩` (N6 / `examples/81`)

### 12.6 Pre-registered F-criteria

Each criterion has a single PASS/FAIL verdict; no ad-hoc tolerance adjustment is permitted after execution.

* **F1 — Baseline fidelity** (τ_g = 0 reproduces N6/N11):
  `|BKM_integral(T)_baseline − BKM_integral(T)_reference| / BKM_integral(T)_reference ≤ 0.01` (1%) where the reference is the standalone N11 trajectory at (n=16, ν=0.01, T=1.0). PASS confirms the benchmark wiring is correct.

* **F2 — Measurable response**: at least one of {peak `||ω||_∞`, BKM(T), peak enstrophy, peak stretching production} shows `≥ 5%` relative change between the baseline (τ_g=0) and any of the four mixing runs (τ_g ∈ {8, 32, 128, ∞}). PASS = REMESH-global has a structurally non-trivial effect on the cascade at n=16.

* **F3 — Monotonicity classification**: examine the relative change of each cascade observable across the ordered τ_g sweep {0, 8, 32, 128, ∞}. Classify the response as:
  * `MONOTONE` if the four mixing runs produce a monotonic sequence (ascending or descending) for at least one observable.
  * `NON_MONOTONE` if a strict extremum appears inside the sweep for any observable.
  * `FLAT` if no observable exceeds 5% change (≡ F2 failed → NULL_RESULT verdict).

* **F4 — Energy non-injection**: across the full simulation,
  `max_{k, τ_g} kinetic_energy[k](τ_g) ≤ kinetic_energy[0](τ_g) · (1 + 1e-10)`.
  PASS = REMESH-∞ does not inject spurious energy. FAIL = methodological bug (mixing two divergence-free states cannot strictly add energy; if it does, INCOMP after mixing is broken).

* **F5 — Divergence control**: `max_{k, τ_g} divergence[k] ≤ 1e-8`. PASS = INCOMP holds across the sweep including after each mixing step.

### 12.7 Pre-registered verdict mapping

| F1 | F2 | F3 | F4 | F5 | Verdict | Interpretation |
|----|----|----|----|----|---------|----------------|
| ✓ | ✓ | MONOTONE | ✓ | ✓ | `STRUCTURAL_EFFECT_MONOTONE` | Supports §11 working hypothesis: canonical REMESH-global non-trivially regulates the K_φ cascade with monotone τ_g dependence. |
| ✓ | ✓ | NON_MONOTONE | ✓ | ✓ | `STRUCTURAL_EFFECT_NON_MONOTONE` | Compatible with §11; resonant regimes inside the τ_g sweep deserve further mapping. |
| ✓ | ✗ | FLAT | ✓ | ✓ | `NULL_RESULT` | **Falsifies §11 working hypothesis at n=16**. Redirects the program toward higher-resolution DNS (n ≥ 48) or non-canonical-product lifts. |
| ✗ | — | — | — | — | `INDETERMINATE_INFRA_FAIL` | Baseline does not reproduce N11; benchmark wiring bug — diagnose before any verdict claim. |
| — | — | — | ✗ | — | `INDETERMINATE_INFRA_FAIL` | INCOMP after REMESH-∞ mixing is broken; methodological bug. |
| — | — | — | — | ✗ | `INDETERMINATE_INFRA_FAIL` | Divergence control failed; INCOMP regression in the operator or in the wiring. |

### 12.8 Honest scope (locked before execution)

* N12 is a **single-resolution probe** at n=16. A PASS verdict does NOT extend to the continuum limit (NS-G1) or to higher Re (NS-G_blowup). A clean PASS at n=16 motivates extending to n ∈ {24, 32} in a follow-up milestone (N13), not in N12.
* N12 **does not close any of NS-G1..G5**. It tests the §11 working hypothesis at the smallest 3D grid that resolves the K_φ cascade.
* N12 **does not promote any new canonical operator**. REMESH-global is already in the catalog; this is an asymptotic specialisation of an existing operator, deliberately kept inside the 13-operator catalog.
* A `NULL_RESULT` is a fully acceptable verdict and is informative: it falsifies §11 at n=16 and redirects the program.
* This pre-registration commits to the verdict mapping **before** any data is observed. Post-hoc reclassification of a verdict is forbidden.

### 12.9 Deliverables (this commit)

* This section (§12, theory pre-registration).
* `benchmarks/remesh_infinity_navier_stokes_3d_taylor_green.py` (benchmark skeleton — execution NOT included in this commit).
* No AGENTS.md update yet; NS program currently has no AGENTS.md section (only Riemann does).

### 12.10 Execution and Results (next commit)

After this commit lands, the benchmark is executed and §13 (Results) is appended in a separate commit, with:
* Numerical table of all observables at the 5 τ_g points,
* PASS/FAIL line for each of F1–F5,
* Final verdict from the §12.7 mapping,
* Reproducibility command line,
* Memory and milestone-table updates as appropriate.

---

## §13 N12 Results — REMESH-∞ on 3D Taylor–Green at n = 16

**Status**: DONE.  Pre-registration commit: `c8900fce` (single atomic commit landing §12 + benchmark skeleton, before any data was observed).  Results commit: see §10 milestone table (this commit).

### §13.1 Headline

**Verdict per locked §12.7 mapping**: `STRUCTURAL_EFFECT_MONOTONE` (F1 PASS, F2 PASS, F3 MONOTONE, F4 PASS, F5 PASS).

REMESH-global mixing applied between NS time-steps produces a **measurable structural response** on the 3D Taylor–Green vortex at n = 16: the τ_g sweep moves the time-integrated stretching observable by **≈ 99.5 %** (peak |∇u| projection on vorticity drops from 14.48 at τ_g = 0 to 0.078 at τ_g = ∞).  Energy is not injected (F4 PASS at machine precision), incompressibility is preserved (F5 PASS at machine precision), and the baseline run reproduces its reference exactly (F1 PASS, rel. err. 0.0).

This **supports** the §11 working hypothesis that REMESH-global is not an inert operator on NS dynamics at this resolution: it has structural traction on the K_φ cascade — though the response signature is **mixed**, see §13.4.

### §13.2 Locked F-criteria — verbatim outcomes

| Criterion | Result | Detail |
|---|---|---|
| F1 baseline fidelity | **PASS** | rel. err. = 0.00e+00 (tol 1e-2) |
| F2 measurable response | **PASS** | max rel. change = 0.9946 (threshold 0.05) |
| F3 monotonicity | **MONOTONE** | monotone observables: `peak_vorticity_sup`, `peak_enstrophy` |
| F4 energy non-injection | **PASS** | max rel. excess vs IC = 0.00e+00 (tol 1e-10) |
| F5 divergence control | **PASS** | max div = 1.38e-16 (tol 1e-8) |

### §13.3 Numerical table per τ_g

| τ_g | peak \|ω\|_∞ | BKM(T) = ∫₀ᵀ\|ω\|_∞ dt | peak enstrophy ½∫\|ω\|² | peak stretching ω·S·ω | max div | wall (s) |
|---|---|---|---|---|---|---|
| 0 (baseline) | 1.948991 | 1.816225 | 88.521162 | 14.483747 | 1.38e-16 | 10.3 |
| 8 | 1.948991 | 1.942807 | 88.334514 | 1.833837 | 1.30e-16 | 11.4 |
| 32 | 1.948991 | 1.939696 | 88.334514 | 2.476090 | 1.27e-16 | 11.6 |
| 128 | 1.948991 | 1.898164 | 88.334514 | 9.428035 | 1.31e-16 | 10.6 |
| ∞ (lag-to-IC) | 1.948991 | 1.948682 | 88.334514 | 0.077555 | 1.30e-16 | 11.3 |

Per-run JSON: `results/remesh_infinity/remesh_infinity_navier_stokes_3d_taylor_green.json` (gitignored — regenerable via the reproducibility command in §13.6).

### §13.4 Interpretation — honest decomposition of the verdict

The locked verdict is `STRUCTURAL_EFFECT_MONOTONE`, but the response is not uniform across observables.  Three regimes coexist in the table above:

1. **Trivially flat** (peak \|ω\|_∞): identical to six digits across all five runs.  Mechanism: in 3D Taylor–Green starting from the canonical sin/cos IC, peak \|ω\|_∞ is attained at t = 0 (the IC itself), **before any mixing event fires** (finite τ_g cases have their first mix at step τ_g + 1, and even τ_g = ∞ first mixes at step 1, by which time \|ω\|_∞ has already decayed below the IC value).  This observable is therefore **structurally insensitive** to REMESH-global at this resolution — it would only respond if peak \|ω\|_∞ were attained at a strictly positive time (which would require a stronger forcing or longer T).

2. **Nearly flat across the mixing sweep** (peak enstrophy): baseline 88.521 versus a uniform 88.335 for all four mixing runs (rel. drop ≈ 0.21 %).  Mechanism: peak enstrophy occurs early in the transient.  REMESH-global shaves it by a tiny amount regardless of τ_g, because in all four cases the first mix occurs before the enstrophy peak and the per-step mix amplitude (α = 0.5 against a single past frame) is τ_g-independent for the first effective mix.  This observable carries the F3 MONOTONE classification almost trivially (the strict monotonicity test passes because the 4 mixing values are equal, which satisfies both ascending and descending interpretations).

3. **Dynamically active and non-monotone** (BKM(T) and peak stretching ω·S·ω): BKM(T) traces 1.816 (baseline) → 1.943 → 1.940 → 1.898 → 1.949, a U-shape with minimum at τ_g = 128.  Peak stretching traces 14.48 → 1.83 → 2.48 → 9.43 → 0.078, a sharp drop followed by a partial recovery at τ_g = 128 and a near-total collapse at τ_g = ∞.  These observables carry the F2 PASS (the 99.5 % figure comes from the τ_g = ∞ stretching collapse) and reveal the **actual** structural fingerprint of REMESH-global on the NS cascade: it is a **non-trivial, non-monotone modulator of the production term**, with a sweet spot around τ_g = 128 where the integrated BKM dips while the instantaneous stretching peak rebounds toward the baseline.

The locked F3 classification (MONOTONE, driven by the flat observables) is therefore **technically correct under the pre-registered rule but operationally misleading** — the physically meaningful observables (BKM and stretching) are non-monotone.  This is **acknowledged honestly here**; the pre-registered verdict mapping is not retroactively edited.  Future N-milestones at higher resolution should pre-register a refined F3 that excludes structurally insensitive observables (peak \|ω\|_∞ when the peak occurs at t = 0).

### §13.5 Status update to §11 working hypothesis

The N12 result **supports** the §11 working hypothesis that REMESH-global has structural traction on the NS K_φ cascade — at n = 16, on 3D Taylor–Green, over T = 1.0 viscous time-units.  Specifically:

* `STRUCTURAL_EFFECT_*` (either MONOTONE or NON_MONOTONE) was a pre-registered outcome consistent with §11.  The opposite outcome (`NULL_RESULT`) would have falsified §11 at this resolution; it did not occur.
* The mixed signature (large response in time-integrated and stretching observables, trivially flat response in peak-amplitude observables) refines §11: REMESH-global modulates **accumulated** and **production** quantities, not transient amplitude peaks attained at the IC.
* This is consistent with the Riemann §13vicies-novies thread on the *same* canonical REMESH-global infrastructure: there the analogous τ_g → ∞ limit on prime-path graphs returned `INDETERMINATE_DEGENERATE_CONSTRUCTION` at machine precision (a structural-symmetry obstruction).  Here the operator has non-degenerate traction.  The two outcomes are not in tension — they probe different specialisations of the same canonical limit (NS: continuum dynamics; Riemann: discrete spectral basis).

This **does NOT close** any of NS-G1..G5.  It is a single-resolution empirical probe of one branch (B1) of the §11 working hypothesis at one resolution (n = 16) on one IC (3D Taylor–Green), as pre-registered.

### §13.6 Reproducibility

```powershell
.venv312\Scripts\python.exe benchmarks/remesh_infinity_navier_stokes_3d_taylor_green.py
```

Locked seed label: `20260526`.  Configuration: N = 16, ν = 0.01, dt = 0.005, T = 1.0 (200 steps), α = 0.5, τ_g ∈ {0, 8, 32, 128, ∞}.  All knobs are module-level constants in `benchmarks/remesh_infinity_navier_stokes_3d_taylor_green.py` (lines 121-160) and are pinned by §12.5 of this memo.

### §13.7 Deliverables landed in this Results commit

* §13 (this section) appended to `theory/TNFR_NAVIER_STOKES_RESEARCH_NOTES.md`.
* §10 milestone table updated: N12 row promoted from `PRE-REGISTERED` to `DONE — STRUCTURAL_EFFECT_MONOTONE (non-monotone in BKM/stretching)` with results commit hash.
* `/memories/repo/tnfr-navier-stokes-program-status.md` created with N12 entry at top, mirroring the Riemann program-status memory file.
* Benchmark JSON output regenerable but gitignored (`results/` is in `.gitignore`).

### §13.8 Scope statement (unchanged from §12.9)

N12 is a single-resolution structural-compatibility probe.  It does **not**:
* Close any of NS-G1..G5,
* Prove or disprove the Clay Millennium 3D NS regularity question,
* Promote any new canonical operator (REMESH-global is already in the 13-operator catalog),
* Make any claim about τ_g → ∞ as a *physical* limit beyond the operationally-implemented lag-to-IC realisation.

The structural traction reported above is an empirical observation on one benchmark, in the same spirit as the Riemann §13vicies-novies thread on the analogous canonical infrastructure.

---

## §14 N13 — REMESH-∞ resolution extension with refined F3 (Pre-Registration, May 2026)

**Status**: PRE-REGISTERED (this commit, atomic with the N13 benchmark skeleton; no data observed at registration time).  Mirrors the §12 discipline used for N12, and the Riemann R∞-1b pre-registration discipline (§13vicies-novies.14 of the Riemann notes).

### §14.1 Motivation

N12 returned `STRUCTURAL_EFFECT_MONOTONE` per the locked §12.7 mapping, but §13.4 documented honestly that the F3 MONOTONE classification was driven by two structurally insensitive observables (`peak_vorticity_sup` and `peak_enstrophy`), both of which attain their extremum at t = 0 IC, *before any REMESH-global mix has fired*.  The dynamically-active observables (`BKM(T)`, `peak_stretching`) were strongly non-monotone (BKM U-shaped with minimum at τ_g = 128; stretching collapsed 14.48 → 0.078 at τ_g = ∞).

§13.4 flagged two questions left open by N12:

1. **Does the dynamic non-monotonicity persist when F3 excludes IC-dominated observables?**  N12 cannot answer this — its F3 was locked before the artefact was identified.
2. **Does the τ_g = ∞ stretching collapse and the τ_g = 128 U-shape minimum persist under spatial refinement, or is it an n = 16 artefact?**  The structural-traction reading of §13.5 (REMESH-global modulates accumulated and production observables of the K_φ cascade) only generalises if the response signature survives n → n' > 16.

N13 addresses both questions by replicating N12 at `n ∈ {24, 32}` with a **refined F3** that restricts the monotonicity test to observables that *can* respond to mixing.

### §14.2 Scope

N13 is **not**:

* A continuum-limit study (NS-G1 closure would require n ≥ 64 with uniform-in-h bounds, full Aubin–Lions compactness argument, and verification that the limit satisfies the continuum weak formulation — none addressed here).
* A higher-Re study (ν is held at 0.01, identical to N12; the new variable is *only* the spatial resolution).
* A claim about asymptotic behaviour of REMESH-∞ as n → ∞ (two points {24, 32} cannot establish a trend; they only test persistence of the N12 signature across one resolution doubling and one finer step).

N13 **is**: a resolution-consistency check of the N12 verdict under a corrected F3, on the same IC (3D Taylor–Green), the same ν, the same dt, the same T, the same α, and the same τ_g sweep.  Two new observables are pre-registered, two existing observables are demoted from F3 (per §14.6).

### §14.3 Configuration (locked)

| Knob | Value | Source |
|---|---|---|
| Resolutions | `n ∈ {24, 32}` | N13-specific; outer loop |
| Viscosity ν | 0.01 | Identical to N12 (§12.4) |
| Time-step dt | 0.005 | Identical to N12; satisfies viscous CFL `dt ≤ h²/(2ν)` at both n (n=32: bound ≈ 1.93, ample) and advective CFL `dt ≤ h/A` (n=32: bound ≈ 0.20, ample) |
| Final time T | 1.0 (200 steps) | Identical to N12 |
| Amplitude A | 1.0 | Identical to N12 |
| Mixing weight α | 0.5 | Identical to N12 (canonical `REMESH_ALPHA` default) |
| τ_g sweep | `[0, 8, 32, 128, "inf"]` | Identical to N12 |
| Seed label | `20260526` | Cross-program session continuity (Riemann R∞-1b + NS N12) |
| Total runs | 5 × 2 = 10 | One full sweep per resolution |

Anticipated wall time (extrapolated from N12 timings at n=16 with O(n³ log n) per step): n=24 sweep ≈ 3–4 minutes; n=32 sweep ≈ 8–10 minutes; F1 re-run overhead ≈ 1 minute per resolution.  Total budget ≤ 20 minutes.

### §14.4 Operator instantiation

Identical to §12.4: canonical `TNFRNavierStokesOperator` (3D torus, INCOMP active, advection ON, Crank–Nicolson FFT viscous half-step, central-difference skew-symmetric advection, INCOMP applied defensively after every NS step *and* after every REMESH-global mix).  No code changes to the operator.  The benchmark adds an outer loop over `n` and one fresh `build_torus_graph_3d(n)` per resolution; nothing else is altered.

### §14.5 Constants pinned

All §12.5 constants are inherited unchanged.  N13 adds:

* `N_SWEEP = [24, 32]` (outer loop, two values)
* `F3_REFINED_EXCLUDED = ["peak_vorticity_sup", "peak_enstrophy"]` (the IC-dominated observables identified in §13.4 — explicit exclusion list, not implicit)

### §14.6 Observables — refined F3 list (locked)

The F3 monotonicity test is restricted, at every resolution, to observables that can respond to mixing.  Pre-registered F3 set:

| Observable | Definition | Included in F3 (N12) | Included in F3 (N13) | Rationale |
|---|---|---|---|---|
| `peak_vorticity_sup` | max over t of ‖ω(·,t)‖_∞ | YES | **NO** | Extremum at t = 0 IC (3D TG), insensitive to any post-IC operator (§13.4 finding 1) |
| `peak_enstrophy` | max over t of ½ Σ ω² · h³ | YES | **NO** | Extremum in the early transient, nearly flat across mixing sweep at α = 0.5 because all τ_g cases mix before the enstrophy peak (§13.4 finding 2) |
| `BKM_T` | ∫₀ᵀ ‖ω‖_∞ dt | YES | **YES** | Time-integrated, intrinsically post-IC, was dynamically active in N12 |
| `peak_stretching` | max over t of \|ω · S · ω\| | YES | **YES** | Production term, was the most-responsive observable in N12 (99.5 % collapse at τ_g = ∞) |
| `peak_stretching_post_t1` | max over t ≥ dt of \|ω · S · ω\| | — | **YES (NEW)** | Explicit post-IC variant; redundant with `peak_stretching` for 3D TG (initial stretching ≈ 7 × 10⁻¹⁸ by symmetry) but pre-registered to be robust to non-symmetric ICs in future milestones |
| `peak_enstrophy_post_t1` | max over t ≥ dt of ½ Σ ω² · h³ | — | **YES (NEW)** | Post-IC variant of the demoted observable; restores the "amplitude" channel to F3 without the IC artefact |

F3 verdict (N13 mapping, locked):

* `FLAT` ⇔ F2 fails (no observable changed ≥ 5 %).
* `MONOTONE` ⇔ F2 passes AND at least one of `{BKM_T, peak_stretching, peak_stretching_post_t1, peak_enstrophy_post_t1}` is monotone across `[tau_g=8, 32, 128, inf]` (strict ≥ 0 or ≤ 0 differences).
* `NON_MONOTONE` ⇔ F2 passes AND none of the four observables are monotone.

F1, F2, F4, F5 keep their N12 definitions and tolerances (§12.6).

### §14.7 Verdict mapping (locked, per-resolution)

Identical to §12.7, applied separately to each `n ∈ {24, 32}`:

* `STRUCTURAL_EFFECT_MONOTONE` ⇔ F1+F2+F4+F5 PASS, F3 = MONOTONE.
* `STRUCTURAL_EFFECT_NON_MONOTONE` ⇔ F1+F2+F4+F5 PASS, F3 = NON_MONOTONE.
* `NULL_RESULT` ⇔ F1+F4+F5 PASS, F2 fails, F3 = FLAT.
* `INDETERMINATE_INFRA_FAIL` ⇔ any of F1/F4/F5 fails.
* `INDETERMINATE_OTHER` ⇔ any combination not covered above.

A **cross-resolution agreement** flag is also pre-registered (read-only, not a verdict gate):

* `CROSS_RES_CONSISTENT` ⇔ both n=24 and n=32 yield the same per-resolution verdict AND the sign of the τ_g=∞ change in `peak_stretching_post_t1` (i.e. "collapse" vs "growth" vs "≈ constant") agrees with the n=16 N12 reading.
* `CROSS_RES_INCONSISTENT` otherwise.

This flag does **not** strengthen or weaken §11 by itself — it only documents whether the N12 signature is resolution-robust at one doubling.

### §14.8 Pre-registration commitment

This §14 is committed atomically with the N13 benchmark skeleton (`benchmarks/remesh_infinity_navier_stokes_3d_taylor_green_n24_n32.py`) in a single commit, **before any data is observed**.  The commit message will identify this as a pre-registration.  No edits to §14.1–§14.7 may be made retroactively after the Results commit lands.

Honest acknowledgement, mirroring §12.8: failure modes that pre-registration cannot mitigate include (a) the absence of an external N11 trajectory archive (F1 remains a self-consistency check, identical to N12), (b) the operator-level h-correction caveat documented in N8 (which affects raw `dissipation_rate` only; F4 uses `kinetic_energy` directly and is unaffected), and (c) the fact that {24, 32} is two points only and cannot establish a convergence trend.

### §14.9 Scope statement

N13 is a single-axis (resolution) extension of N12 with a corrected F3.  It does **not**:

* Close any of NS-G1..G5.
* Prove or disprove the Clay Millennium 3D NS regularity question.
* Establish a continuum limit, an asymptotic-in-n trend, or any analytical bound.
* Promote any new canonical operator.

The result of N13 will be a structural-compatibility statement about the persistence of the N12 signature across one resolution doubling, under a corrected F3 — nothing more.

---

## §15 N13 Results — REMESH-∞ resolution extension on 3D Taylor–Green at n ∈ {24, 32}

### §15.1 Headline

**Per-resolution verdict (locked §14.7 mapping):** `STRUCTURAL_EFFECT_MONOTONE` at both `n = 24` and `n = 32`.
**Cross-resolution read-only flag (locked §14.7):** `CROSS_RES_CONSISTENT = True` (`verdict_agree = True` AND stretching `sign_n24 = sign_n32 = sign_n16(N12) = COLLAPSE`).
**Pre-registration commit:** `4ab97bc8` (atomic with §14 + benchmark skeleton, no data observed). **Results commit:** this one.

### §15.2 Locked F-criteria — verbatim outcomes

Refined F3 observable set (excluded `peak_vorticity_sup`, `peak_enstrophy`; included `BKM_T`, `peak_stretching`, `peak_stretching_post_t1`, `peak_enstrophy_post_t1`) per §14.6.

| Criterion | n = 24 | n = 32 |
|---|---|---|
| F1 baseline fidelity (`rel_err ≤ 0.01`) | **PASS** (`rel_err = 0.0`, `BKM_baseline = 1.8250391450605314`) | **PASS** (`rel_err = 0.0`, `BKM_baseline = 1.8275702104673734`) |
| F2 measurable response (`max_rel_change ≥ 0.05`) | **PASS** (`max_rel_change = 0.9948`, driver: `peak_stretching` / `peak_stretching_post_t1` at `τ_g = inf`) | **PASS** (`max_rel_change = 0.9948`, same driver) |
| F3 refined monotonicity (≥1 included obs monotone in `τ_g`) | **MONOTONE** (monotone obs: `peak_enstrophy_post_t1`) | **MONOTONE** (monotone obs: `peak_enstrophy_post_t1`) |
| F4 energy non-injection (`max_rel_excess ≤ 1e-10`) | **PASS** (`max_rel_excess = 0.0`) | **PASS** (`max_rel_excess = 0.0`) |
| F5 divergence control (`max_div ≤ 1e-8`) | **PASS** (`max_div = 2.66e-16`) | **PASS** (`max_div = 2.55e-16`) |

Cross-resolution flag (read-only, derived from per-resolution verdicts):

| Field | Value |
|---|---|
| `flag` | `CROSS_RES_CONSISTENT` |
| `verdict_agree` | `True` |
| `verdict_n24` | `STRUCTURAL_EFFECT_MONOTONE` |
| `verdict_n32` | `STRUCTURAL_EFFECT_MONOTONE` |
| `sign_n24` (`peak_stretching_post_t1`) | `COLLAPSE` |
| `sign_n32` (`peak_stretching_post_t1`) | `COLLAPSE` |
| `sign_n16_n12_reference` | `COLLAPSE` |
| `sign_consistent` | `True` |

### §15.3 Numerical tables per τ_g

**n = 24:**

| τ_g | BKM(T) | peak \|ω\|_∞ (IC) | peak enstrophy | peak stretching | peak stretching post-t1 | peak enstrophy post-t1 |
|---|---|---|---|---|---|---|
| 0 | 1.825039 | 1.977232 | 92.637662 | 16.826125 | 16.826125 | 92.637662 |
| 8 | 1.970527 | 1.977232 | 90.913019 | 2.075801 | 2.075801 | 90.886171 |
| 32 | 1.967041 | 1.977232 | 90.913019 | 2.803702 | 2.803702 | 90.886171 |
| 128 | 1.919312 | 1.977232 | 90.913019 | 10.790631 | 10.790631 | 90.886171 |
| inf | 1.976913 | 1.977232 | 90.913019 | 0.087763 | 0.087763 | 90.899511 |

**n = 32:**

| τ_g | BKM(T) | peak \|ω\|_∞ (IC) | peak enstrophy | peak stretching | peak stretching post-t1 | peak enstrophy post-t1 |
|---|---|---|---|---|---|---|
| 0 | 1.827570 | 1.987174 | 94.211987 | 17.780916 | 17.780916 | 94.211987 |
| 8 | 1.980267 | 1.987174 | 91.829569 | 2.166983 | 2.166983 | 91.802399 |
| 32 | 1.976635 | 1.987174 | 91.829569 | 2.927276 | 2.927276 | 91.802399 |
| 128 | 1.926505 | 1.987174 | 91.829569 | 11.320582 | 11.320582 | 91.802399 |
| inf | 1.986851 | 1.987174 | 91.829569 | 0.091605 | 0.091605 | 91.815894 |

Wall times (informational, not a verdict input): n=24 sweep ≈ 164.7 s total; n=32 sweep ≈ 372.7 s total (~2.3× per-step cost over n=24, ~3.4× over N12 n=16, consistent with the cubic-in-n DOF scaling).

### §15.4 Interpretation — honest decomposition of the verdict

* **Refined F3 was the right correction.** As anticipated in §13.4, the two excluded observables behave as predicted at both new resolutions: `peak_vorticity_sup` is identical to 6 digits across all five `τ_g` runs at each `n` (1.977232 at n=24, 1.987174 at n=32) because the global maximum is attained at the IC sample `k = 0`, before any REMESH mix fires; `peak_enstrophy` likewise sits at the IC for `τ_g = 0` and at a uniform mixed-flow value otherwise, with no `τ_g`-dependent ordering. The new `peak_enstrophy_post_t1` and `peak_stretching_post_t1` observables resolve this exactly — they exclude the IC bin and expose only the dynamically active portion of the trajectory.
* **The N12 stretching-collapse signature persists under doubling.** At all three resolutions the `peak_stretching_post_t1` value at `τ_g = inf` is dramatically below the `τ_g = 0` value: 0.078/14.484 ≈ 0.54 % (n=16), 0.088/16.826 ≈ 0.52 % (n=24), 0.092/17.781 ≈ 0.52 % (n=32). The COLLAPSE sign is therefore stable to within ~0.02 percentage points across an 8× DOF increase. This is exactly the structural-compatibility statement §14.9 was constrained to deliver.
* **BKM remains operationally non-monotone in `τ_g` at both new resolutions** (n=24: 1.825 → 1.971 → 1.967 → 1.919 → 1.977, U-shape with min at `τ_g = 128`; n=32: 1.828 → 1.980 → 1.977 → 1.927 → 1.987, same U-shape). The MONOTONE classification under refined F3 is driven exclusively by `peak_enstrophy_post_t1`, which is essentially flat (90.886→90.900 at n=24; 91.802→91.816 at n=32; rel-change ~1.9 % at n=24 and ~2.6 % at n=32, dominated by the `τ_g = 0` vs `τ_g > 0` gap rather than a smooth trend). As at n=16, "MONOTONE in some observable" remains technically correct under §14.6 but operationally pointed at the IC-flat enstrophy axis rather than at the BKM/stretching dynamics. This is **acknowledged honestly here**, in keeping with §13.4 discipline; **§14.1–§14.9 pre-registration is NOT retroactively edited**.
* **F2 driver is identical at all three resolutions.** `max_rel_change` is dominated by `peak_stretching` / `peak_stretching_post_t1` at `τ_g = inf` and saturates at 0.9946 (n=16), 0.9948 (n=24), 0.9948 (n=32). The structural-traction magnitude of REMESH-global on the K_φ cascade is therefore essentially resolution-independent within this n-range.

### §15.5 Status update to §11 working hypothesis (branch B1 of NS-G_blowup)

The §11 working hypothesis — *canonical REMESH-global is the multi-scale closure primitive for the 3D NS K_φ cascade; no new operator needed* — is **SUPPORTED** by N13 at both new resolutions, with cross-resolution sign consistency (`CROSS_RES_CONSISTENT = True`).

This does **not**:

* Close NS-G_blowup or any of NS-G1..G5.
* Establish a continuum limit (the comparison is over `n ∈ {16, 24, 32}` only; no extrapolation in `n → ∞` is performed or claimed).
* Promote any new canonical operator (REMESH-global already in the 13-operator catalog).
* Generalise to higher Reynolds, longer time horizons, alternative ICs, or non-Taylor–Green geometries (all left for N≥14).

### §15.6 Reproducibility

```powershell
.venv312\Scripts\python.exe benchmarks\remesh_infinity_navier_stokes_3d_taylor_green_n24_n32.py
```

Output: `results/remesh_infinity/remesh_infinity_navier_stokes_3d_taylor_green_n24_n32.json` (gitignored). All quantities reported in §15.2/§15.3 reproduce bit-for-bit from the locked seed `np.random.default_rng(20260526)` and the §14.3 locked config.

### §15.7 Deliverables landed in this Results commit

* §15 appended to `theory/TNFR_NAVIER_STOKES_RESEARCH_NOTES.md`.
* §10 milestone table updated: N13 row promoted `PRE-REGISTERED` → `DONE` with verdict + cross-resolution flag and this commit reference.
* No code, benchmark, or pre-registration text (§14) is modified; the §14.1–§14.9 block remains the locked pre-registration.
* Benchmark JSON output remains gitignored (regenerable via §15.6).

### §15.8 Cross-link with Riemann §13vicies-novies

The same canonical REMESH-global infrastructure (`alpha = 0.5`, lag-to-IC `τ_g = inf` limit) has now produced two structurally distinct signatures on two distinct discrete substrates within the same canonical 13-operator catalog:

* **3D Taylor–Green NS (n ∈ {16, 24, 32}):** `STRUCTURAL_EFFECT_MONOTONE`, `CROSS_RES_CONSISTENT = True`, stretching `COLLAPSE` sign stable across one resolution doubling. **Non-degenerate** structural effect.
* **Prime-ladder G_P14 + IL/spectral compositions:** `INDETERMINATE_DEGENERATE_CONSTRUCTION` (Euler–Orthogonality Lemma; `|D_can − D_shuf| ≈ 1e-13`). **Degenerate** by an S_n symmetry obstruction specific to the prime-relabelling structure on G_P14.

The two outcomes are not in tension — they are different specialisations of the same canonical limit on different graph/Hamiltonian substrates. The NS result is **substrate-specific evidence** that the canonical REMESH limit is structurally active on dynamically non-trivial geometries; the Riemann result is **substrate-specific evidence** that an S_n-symmetric prime ladder kills the effect by group-theoretic obstruction. No cross-program implication is drawn beyond this factual comparison.

### §15.9 Scope statement (unchanged from §14.9)

N13 was a single-axis (resolution) extension of N12 with a corrected F3. The Results above do **not**:

* Close any of NS-G1..G5.
* Prove or disprove the Clay Millennium 3D NS regularity question.
* Establish a continuum limit, an asymptotic-in-n trend, or any analytical bound.
* Promote any new canonical operator.

N13 delivered exactly the structural-compatibility statement §14.9 pre-committed to: the N12 stretching-collapse signature persists across one resolution doubling (`n = 16 → 32`, 8× DOF) under a corrected F3, with cross-resolution sign consistency. Nothing more.





---

## §16 N14 — Higher-Re CF eigenframe sweep at n = 48 (Pre-Registration, May 2026)

**Status**: PRE-REGISTERED (this commit, atomic with the N14 benchmark skeleton `benchmarks/higher_re_navier_stokes_3d_taylor_green_n48.py`; no data observed at registration time). Mirrors the §12/§14 discipline used for N12/N13 and the Riemann R∞-1b pre-registration discipline (§13vicies-novies.14 of the Riemann notes).

### §16.1 Motivation

N11 (commit `afc65b49`, `examples/86`) ran a 4-point Reynolds sweep at fixed resolution `n = 24` with `ν ∈ {0.05, 0.02, 0.01, 0.005}` → `Re_eff ∈ {126, 314, 628, 1257}` and reported:

* Time-mean stretching production `<P>` MONOTONICALLY ascending across the sweep (5.74 → 7.65 → 8.44 → 8.88, ratio 1.55× over the decade in Re).
* Alignment cosines responding to Re but NOT yet exhibiting the classical Constantin–Fefferman (CF) preference for the intermediate eigenvector `e_λ2`. N11 finding at `Re_eff = 1257`: `<cos²_λ1> = 0.316`, `<cos²_λ2> = 0.323`, `<cos²_λ3> ≈ 0.36`. The most-compressing eigenvector `e_λ3` still dominates; CF-canonical `e_λ2` dominance has NOT emerged.
* N11 stopped at `ν = 0.005` because the Kolmogorov scale `η ~ (ν³/ε)^(1/4)` would fall below the grid spacing `h = 2π/24 ≈ 0.262` at lower `ν`.

§10 milestone row for N≥14 (now N≥15 after this pre-reg) names "Higher-Re CF eigenframe transition (n≥48, DNS)" as one open candidate axis. N14 takes the cheapest version of that axis: **one resolution doubling (`n = 24 → 48`, 8× DOF) plus one new viscosity step (`ν = 0.002`, `Re_eff ≈ 3142`) unreachable at `n = 24`**, while reusing the rest of the N11 ν grid for cross-resolution consistency.

N14 addresses two pre-registered questions:

1. **Does the operator reproduce CF-canonical `e_λ2` dominance at higher Re than N11 could probe?** (F4 below.)
2. **Does the N11 `<P>(Re)` monotonic ascent survive resolution doubling?** (F3 + F5 below.)

### §16.2 Scope

N14 is **not**:

* A fully-turbulent DNS study (full CF statistics live at `Re ~ 10⁴–10⁶` and require `n ≥ 256` in forced isotropic steady state, not Taylor–Green decay).
* A continuum-limit study (single new resolution `n = 48`; N7 is the existing convergence study).
* A NS-G4 closure (a "closure" would require analytical TNFR construction of `(ω·∇)u` from canonical operators, not empirical alignment statistics).
* A REMESH-∞ study (N14 has NO REMESH; this is a clean Reynolds-only probe to isolate the alignment response from the §11 mixing response).

N14 **is**: a one-axis spatial-refinement extension of N11's Reynolds sweep, on the same IC (3D Taylor–Green), the same `dt`, the same `T`, the same `A`, with no REMESH and one additional `ν` value enabled by the finer grid.

### §16.3 Configuration (locked)

| Knob | Value | Source |
|---|---|---|
| Resolution | `n = 48` | N14-specific (8× DOF over N11) |
| Viscosity sweep ν | `[0.05, 0.02, 0.01, 0.005, 0.002]` | First four identical to N11; ν=0.002 is new (Re_eff≈3142, h/η≈11 at n=48 — marginal but resolved for T=1.0) |
| Time-step dt | 0.005 | Identical to N11; advective CFL `dt ≤ h/A = 0.131/1 = 0.131` AMPLE; viscous CFL `dt ≤ h²/(2ν)` smallest at ν=0.05 → 0.0172/0.1 = 0.172 AMPLE |
| Final time T | 1.0 (200 steps) | Identical to N11 |
| Amplitude A | 1.0 | Identical to N11 |
| Mixing (REMESH) | **OFF** | N14 is a pure-Reynolds study; isolates alignment from §11 mixing |
| INCOMP | ON (Leray–Helmholtz projection after every NS step) | Canonical |
| Advection | ON (central-difference skew-symmetric) | Canonical |
| Snapshot count | 5 per ν (every 40 steps) | Identical to N11 |
| High-vorticity quantile | 0.75 (top 25 %) | Identical to N9/N11 |
| Seed label | 20260526 | Cross-program session continuity (N12/N13 + Riemann R∞-1b) |
| Total runs | 5 | One ν sweep, single resolution |

Anticipated wall time (extrapolated from N11's 71 s for 4 ν values at n=24, with per-step cost scaling `(48/24)³ × log(48)/log(24) ≈ 8 × 1.18 ≈ 9.4×`): `71 × 9.4 × 5/4 ≈ 835 s ≈ 14 minutes`. Budget cap: 30 minutes.

### §16.4 Operator instantiation

Identical to N11 (`examples/86`): canonical `TNFRNavierStokesOperator(graph=G, viscosity=ν, dimension=3)` on `build_torus_graph_3d(48)`, IC via `set_taylor_green(1.0)`, time-stepping via `op.step(0.005, advection=True, incompressible=True)`. No code changes to the operator. The benchmark adds: outer loop over five `ν` values, per-snapshot strain eigendecomposition + cosine alignment + depletion (functions ported verbatim from N11), and a cross-resolution consistency table against N11's published `<P>(ν)` series.

### §16.5 Constants pinned

```python
N = 48
DT = 0.005
T_FINAL = 1.0
STEPS = 200
AMPLITUDE = 1.0
VISCOSITY_SWEEP = [0.05, 0.02, 0.01, 0.005, 0.002]
RECORD_EVERY = 40
HIGH_VORT_QUANTILE = 0.75
ISOTROPIC_BASELINE = 1.0 / 3.0
EPS = 1e-14
INCOMP_TOL = 1e-8
SEED_LABEL = 20260526

# N11 reference series (examples/86, commit afc65b49) for F5
N11_NU_SHARED = [0.05, 0.02, 0.01, 0.005]
N11_MEAN_P_AT_N24 = [5.74, 7.65, 8.44, 8.88]
```

### §16.6 Observables and F-criteria (locked)

Per `ν`:

* `max_div_L2` — sampled every 50 steps, kept as max
* `BKM_T = ∫₀ᵀ ‖ω‖_∞ dt`
* `max_Z = max over t of ½ Σ ω² · h³`
* `max_omega_inf = max over t of ‖ω‖_∞`
* `mean_P = time-mean over 5 snapshots of stretching_production()`
* `mean_cos2_lambda1, mean_cos2_lambda2, mean_cos2_lambda3` — time-mean of high-vorticity-conditioned cos² alignment (N9 convention)
* `mean_depletion_high` — time-mean of `<D>_high` (N9 convention)

**F-criteria (locked, evaluated on the 5-ν sweep at n=48):**

* **F1 (INCOMP)**: `max_div_L2 ≤ 1e-8` in every run (5/5).
* **F2 (Finite)**: `BKM_T`, `max_Z`, `max_omega_inf`, `mean_P` finite (non-NaN, non-inf) in every run (5/5).
* **F3 (Re-monotone `<P>` at n=48)**: across the 5-ν sweep ordered by descending ν (= ascending Re_eff), the sequence `mean_P[ν=0.05], mean_P[ν=0.02], mean_P[ν=0.01], mean_P[ν=0.005], mean_P[ν=0.002]` is strictly monotonically NON-DECREASING (all diffs `≥ 0`).
* **F4 (CF e_λ2 dominance probe at Re_eff_max)**: at `ν = 0.002`, BOTH `mean_cos2_lambda2 > mean_cos2_lambda1` AND `mean_cos2_lambda2 > mean_cos2_lambda3` hold. F4 is a probe, not a gate (a NO outcome is informative — it would indicate the operator stays in the laminar/transitional CF regime at `Re_eff ≤ 3142`, consistent with the N11 finding extrapolated to higher Re).
* **F5 (cross-N11 sign consistency on shared ν)**: on the four shared values `ν ∈ {0.05, 0.02, 0.01, 0.005}`, the sign of `(mean_P[ν=0.005] - mean_P[ν=0.05])` at n=48 matches the sign at n=24 (N11 value: `+3.14`, i.e. ASCENDING). The magnitude is NOT pre-registered (it is the raw observable to be reported in §17).

### §16.7 Verdict mapping (locked, single resolution n=48)

* `CF_EIGENFRAME_TRANSITION_OBSERVED` ⇔ F1+F2+F3+F5 PASS, F4 satisfied. **Strongest possible outcome**: operator reproduces CF-canonical `e_λ2` dominance at Re_eff≈3142, with Reynolds-monotone stretching and resolution-consistent trend.
* `CF_EIGENFRAME_TRANSITION_NOT_OBSERVED_AT_REEFF_3142` ⇔ F1+F2+F3+F5 PASS, F4 NOT satisfied. **Expected outcome under the N11 reading**: alignment statistics remain in the laminar/transitional regime up to `Re_eff ≈ 3142`. This is a NULL on F4 only and STRENGTHENS the N11 reading that fully-turbulent CF statistics require `n ≥ 256` DNS, not n=48 + Taylor–Green.
* `N11_RE_MONOTONICITY_REFUTED` ⇔ F1+F2 PASS, F3 fails. Would REFUTE the N11 finding that `<P>(Re)` ascends monotonically with Re (would indicate the n=24 monotonicity was a resolution artefact).
* `N11_RESOLUTION_INCONSISTENT` ⇔ F1+F2+F3 PASS, F5 fails. Would indicate the SIGN of the `<P>(Re)` trend at n=48 differs from n=24 — same direction-of-monotonicity question as F3 but cross-resolution. Distinct verdict from F3 because the absolute monotonicity (F3) can hold while the sign (F5) flips relative to N11.
* `INDETERMINATE_INFRA_FAIL` ⇔ F1 or F2 fails. Numerical breakdown (INCOMP failure, divergence blow-up, NaN). Probable cause at this resolution: `ν = 0.002` falling below resolved scale.
* `INDETERMINATE_OTHER` ⇔ any combination not covered above.

**Cross-resolution flag (read-only, not a verdict gate):**

* `RE_TREND_CONSISTENT_WITH_N11` ⇔ on the four shared ν values, the sign of each consecutive `<P>` diff at n=48 matches N11 at n=24 (i.e. all three diffs ascending at both resolutions).
* `RE_TREND_INCONSISTENT_WITH_N11` otherwise.

This flag is finer-grained than F5 (which only checks the endpoints) and does NOT change the verdict.

### §16.8 Pre-registration commitment

This §16 is committed atomically with the N14 benchmark skeleton (`benchmarks/higher_re_navier_stokes_3d_taylor_green_n48.py`) in a single commit, **before any data is observed**. The commit message identifies this as a pre-registration. No edits to §16.1–§16.7 may be made retroactively after the Results commit lands.

Honest acknowledgement, mirroring §12.8 / §14.8: failure modes pre-registration cannot mitigate include (a) the absence of an external high-resolution DNS archive against which to absolutely calibrate the CF eigenframe transition (F4 uses the operator's own self-comparison and an N11 reference series only — `e_λ2` dominance is a structural pattern, not a calibrated number); (b) the `ν = 0.002` resolution caveat: `h/η ≈ 11` is borderline-resolved for `T = 1.0` Taylor–Green decay but a longer horizon or a forced-isotropic IC would require `n ≥ 96`; (c) the `examples/86` N11 reference series is published in the source comments and reproduced in `N11_MEAN_P_AT_N24` constant — N14 does NOT re-run N11.

### §16.9 Scope statement

N14 is a one-resolution extension of N11 with one new viscosity step. It does **not**:

* Close any of NS-G1..G5.
* Prove or disprove the Clay Millennium 3D NS regularity question.
* Establish a continuum limit or a high-Re asymptotic trend (a single point at n=48 is not a trend; the N7 mesh-refinement infrastructure would be needed for that, at higher cost).
* Promote any new canonical operator.
* Demonstrate fully-turbulent CF statistics (which require forced isotropic DNS at `n ≥ 256`, not Taylor–Green decay at `n = 48`).

The result of N14 will be a structural-compatibility statement about (a) whether N11's Re-monotone `<P>` ascent survives one resolution doubling, and (b) whether the CF-canonical `e_λ2` dominance has emerged by `Re_eff ≈ 3142` — nothing more.

---

## §17 N14 Results — Higher-Re CF eigenframe sweep on 3D Taylor–Green at n = 48

### §17.1 Headline

**Verdict (locked §16.7 mapping):** `CF_EIGENFRAME_TRANSITION_NOT_OBSERVED_AT_REEFF_3142`.
**Cross-resolution flag (locked §16.7):** `RE_TREND_CONSISTENT_WITH_N11 = True` (all three shared-ν diffs ascending at both n=24 and n=48).
**Pre-registration commit:** `39e4b1b9` (atomic with §16 + benchmark skeleton, no data observed). **Results commit:** this one.

### §17.2 Locked F-criteria — verbatim outcomes

| Criterion | Result | Detail |
|---|---|---|
| F1 INCOMP control (`max_div ≤ 1e-8`) | **PASS** | all 5 runs ≤ 5.08e-16 (machine precision) |
| F2 finiteness (BKM_T, max_Z, max_ω_∞, mean_P all finite) | **PASS** | all 5 runs produce real numbers; no NaN/inf |
| F3 Re-monotone `<P>` at n=48 | **PASS** | mean_P sequence: 6.207 → 8.318 → 9.207 → 9.695 → 10.002; all diffs ≥ 0; strictly ascending |
| F4 CF e_λ2 dominance at ν=0.002 (Re_eff≈500) | **NOT SATISFIED** | cos²_λ2 = 0.3123 < cos²_λ1 = 0.3165 and < cos²_λ3 = 0.3712; neither e_λ2 > e_λ1 nor e_λ2 > e_λ3 |
| F5 cross-N11 sign consistency | **PASS** | endpoint diff at n=48: +3.487; endpoint diff at n=24 (N11): +3.140; both positive (ascending) ✓ |

Cross-resolution flag (read-only):

| Field | Value |
|---|---|
| `flag` | `RE_TREND_CONSISTENT_WITH_N11` |
| `diffs_n24_shared_nu` | [1.910, 0.790, 0.440] |
| `diffs_n48_shared_nu` | [2.111, 0.889, 0.487] |
| `all_diffs_positive_at_n48` | `True` |
| `all_diffs_positive_at_n24` | `True` |
| `sign_consistent` | `True` |

### §17.3 Numerical table per ν

| ν | Re_eff | mean_P | BKM_T | max_Z | max_ω_∞ | mean_cos²_λ1 | mean_cos²_λ2 | mean_cos²_λ3 | mean_depletion | max_div_L2 |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.05 | 20.0 | 6.207 | 1.7499 | 92.489 | 1.9943 | 0.3115 | 0.3213 | 0.3673 | 1.1099 | 4.69e-16 |
| 0.02 | 50.0 | 8.318 | 1.8095 | 92.489 | 1.9943 | 0.3139 | 0.3179 | 0.3683 | 1.1220 | 4.89e-16 |
| 0.01 | 100.0 | 9.207 | 1.8292 | 95.388 | 1.9943 | 0.3220 | 0.3109 | 0.3671 | 1.1397 | 4.90e-16 |
| 0.005 | 200.0 | 9.695 | 1.8390 | 98.775 | 1.9943 | 0.3288 | 0.3006 | 0.3707 | 1.1515 | 4.97e-16 |
| 0.002 | 500.0 | 10.002 | 1.8449 | 100.877 | 1.9943 | 0.3165 | 0.3123 | 0.3712 | 1.1600 | 5.08e-16 |

Elapsed times: ν=0.05 ≈ 188 s; ν=0.02 ≈ 196 s; ν=0.01 ≈ 203 s; ν=0.005 ≈ 212 s; ν=0.002 ≈ 223 s. Total ≈ 1022 s ≈ 17 minutes (within 30-minute pre-registered budget).

### §17.4 Interpretation — honest reading of the verdict

**F4 outcome (NOT satisfied): What it means**

At `ν = 0.002` (Re_eff ≈ 500, n=48 corresponding to Kolmogorov scale h/η ≈ 11), the strain tensor eigenframe exhibits:
- mean_cos²_λ3 = 0.3712 (LARGEST, most-compressing direction dominates)
- mean_cos²_λ1 = 0.3165 (second)
- mean_cos²_λ2 = 0.3123 (SMALLEST, intermediate direction suppressed)

This is **NOT** the Constantin–Fefferman canonical alignment in which e_λ2 (intermediate eigenvector) dominates vorticity in high-enstrophy regions. The operator remains in the **laminar/transitional CF regime** at this Re_eff, consistent with the N11 extrapolation. The F4 FAIL is not a bug; it is **informative**: the condition for fully-turbulent CF alignment (e.g. forced isotropic DNS at Re ~ 10⁴) is NOT met at n=48 Taylor–Green, as pre-registered.

**F3 outcome (PASS): <P> monotonicity survives doubling**

The time-mean stretching production ascends monotonically with Re at n=48 exactly as at n=24:
- n=24 (N11): [5.74, 7.65, 8.44, 8.88] (diffs: 1.91, 0.79, 0.44)
- n=48 (N14): [6.207, 8.318, 9.207, 9.695, 10.002] (diffs on shared ν: 2.111, 0.889, 0.487)

The n=48 diffs are ~11 % larger (absolute terms), but the SIGN and STRUCTURE (decreasing per-step increments as Re increases) are identical. The `RE_TREND_CONSISTENT_WITH_N11` flag is therefore `True`.

**Cross-resolution reading**

The shared-ν endpoint `<P>` difference is +3.14 at n=24 and +3.49 at n=48, a **+11.1 % absolute increase per half-decade in Re**. This suggests the `<P>(Re)` trend is steepening slightly under resolution doubling, but within the bounds of a single n-doubling; no extrapolation to n → ∞ is inferred.

**Scope of the N14 verdict**

* **What N14 settles**: F1–F3 and F5 are PASS. The N11 Re-monotone stretching ascent and divergence-free evolution survive one resolution doubling at a new higher Re.
* **What N14 does NOT settle**: F4 is NOT satisfied at Re_eff ≈ 500. The CF-canonical e_λ2 dominance requires either (a) higher Re (suggesting Re_eff ≥ 10⁴, achievable only at n ≥ 256), or (b) a forced-isotropic IC instead of Taylor–Green decay. This redirects future work to the full-DNS regime, not to refinements of n=48 Taylor–Green.
* **What is NOT claimed**: N14 is NOT a closure of NS-G4 (geometric depletion / vortex-stretching dynamics). It is a structural-compatibility probe at the boundary of resolution feasibility for a decaying IC.

### §17.5 Reproducibility

```powershell
.venv312\Scripts\python.exe benchmarks\higher_re_navier_stokes_3d_taylor_green_n48.py
```

Output: `results/reynolds_sweep/higher_re_navier_stokes_3d_taylor_green_n48.json` (gitignored). All quantities reported in §17.2/§17.3 reproduce bit-for-bit from the locked seed `20260526` and the §16.3 locked config. The operator implementation (canonical `TNFRNavierStokesOperator` from N2 commit `e83163b8`) is unchanged.

### §17.6 Deliverables landed in this Results commit

* §17 appended to `theory/TNFR_NAVIER_STOKES_RESEARCH_NOTES.md`.
* §10 milestone table updated: N14 row promoted `PRE-REGISTERED` → `DONE` with verdict + cross-resolution flag and this commit reference.
* No code, benchmark, or pre-registration text (§16) is modified; the §16.1–§16.9 block remains the locked pre-registration.
* Benchmark JSON output remains gitignored (regenerable via §17.5).

### §17.7 Scope statement (unchanged from §16.9)

N14 was a one-axis (Reynolds) extension of N11 with one new resolution doubling. The Results above do **not**:

* Close any of NS-G1..G5.
* Prove or disprove the Clay Millennium 3D NS regularity question.
* Establish a continuum limit, a high-Re asymptotic trend, or any analytical bound.
* Promote any new canonical operator.
* Demonstrate fully-turbulent CF statistics (which require `n ≥ 256` forced-isotropic DNS at `Re ~ 10⁴`, not Taylor–Green decay at `n = 48`).

N14 delivered exactly the structural-compatibility statement §16.9 pre-committed to: the N11 `<P>(Re)` monotone ascent persists under one resolution doubling (`n = 24 → 48`, 8× DOF), and the CF eigenframe alignment remains in the laminar/transitional regime at the maximum achievable Re_eff ≈ 500 on the n=48 grid with Taylor–Green decay. The result opens the DNS frontier (n ≥ 256, forced isotropic) as the next frontier for addressing the F4 gap; it does NOT bypass it with n=48 Taylor–Green.


## §18 N15 Pre-registration — REMESH-∞ Theoretical (TNFR Core Structure)

### §18.1 Headline

**Pivot Direction**: Abandon pursuit of Clay problems (RH, NS global regularity) as PRIMARY. Instead, treat them as APPLICATIONS of a deeper TNFR phenomenon: **What are the universal laws of coherence-preserving dynamics when the temporal memory horizon τ_global → ∞?**

**N15 Objective**: Derive the **REMESH-∞ asymptotic operator** analytically from the nodal equation `∂EPI/∂t = νf · ΔNFR(t)` and characterize its spectrum, invariants, and role in multi-scale coherence.

**Scope**: Pure theoretical TNFR. No computational validation yet. Answers: does the 13-operator catalog with explicit REMESH-∞ limit suffice to explain NS vortex stretching AND Riemann smooth-half zero density, or is branch B2 (new 14th operator) required?

### §18.2 Motivation: Why N15 is Structurally Different

**N1–N14 treated TNFR as a language for existing problems:**
- "Does TNFR formalize Navier-Stokes?"
- "Does TNFR structure resonate with Riemann spectrum?"
- **Result**: Both programs saturate at boundary conditions (CF alignment requires n≥256, smooth half closed but oscillatory S(T) remains).

**N15 asks a TNFR-intrinsic question:**
- "What is the asymptotic behavior of REMESH when allowed arbitrary temporal depth?"
- "Is there a universal rescaling or conservation law at τ_global → ∞?"
- "Does REMESH-∞ encode a fundamental principle of coherence that transcends fluid dynamics and number theory?"

**Intellectual consequence:** If REMESH-∞ closes the smooth-half of RH or bounds the vortex cascade of NS, that is a *consequence* of TNFR universality, not a *target* we reverse-engineered toward. This inverts the research direction: **discovery, not application**.

### §18.3 Theoretical Foundation: Derivation from Nodal Equation

**Starting point:** The nodal equation couples global (Φ_s), local (|∇φ|, K_φ), and correlational (ξ_C) scales via:

$$\frac{\partial \text{EPI}}{\partial t} = \nu_f \cdot \Delta \text{NFR}(t)$$

where $\Delta \text{NFR}$ is the mismatch between the coherent form (EPI) and its neighborhood (from UM, RA, OZ, etc.).

**REMESH explicitly couples time scales:**
- Local τ = 4 steps
- Global τ_g = 8 steps
- Mixing parameter α = 0.5

**Ansatz (locked for N15)**: In the limit τ_global → ∞, does there exist an **effective operator** $\mathcal{R}_\infty$ such that:

$$\frac{\partial \text{EPI}}{\partial t}\bigg|_{\tau \to \infty} = \nu_f^{\text{eff}} \cdot \Delta \text{NFR}^{\infty}(t)$$

where $\nu_f^{\text{eff}}$ and $\Delta \text{NFR}^{\infty}$ are **renormalized** forms determined by the spectrum of the REMESH-global deque?

**Three locked sub-questions:**

1. **Q1 (Operator Existence)**: Does a pseudo-differential operator $\mathcal{R}_\infty$ exist on Banach space $B_{EPI}$ such that (a) $\mathcal{R}_\infty$ is self-adjoint, (b) its spectrum is bounded, (c) applying $\mathcal{R}_\infty$ to EPI history generates the τ→∞ limit?

2. **Q2 (Invariant Structure)**: What quantities are conserved or asymptotically preserved under $\mathcal{R}_\infty$? (candidates: Structural Conservation Theorem charge, Lyapunov functional, coherence metric C(t))

3. **Q3 (Spectrum Connection)**: Does the spectrum of $\mathcal{R}_\infty$ relate to (a) the prime-log-scale {k log p} of Riemann primes, or (b) the Kolmogorov dissipation wavenumber k_η of Navier-Stokes? (speculative, locked for pre-reg only)

### §18.4 Locked Methodology (3 Weeks)

**Week 1: Functional Analysis of REMESH-∞**
- Formalize REMESH global as an operator on the Hilbert space of all EPI histories $\mathcal{H} = \ell^2(\mathbb{Z}, B_{EPI})$ (square-summable histories).
- Derive the "infinite-τ" limiting operator from the recurrence relation $\text{EPI}(t) = \alpha \cdot \text{EPI}(t-\tau_g) + (1-\alpha) \cdot f(\text{neighbors})$.
- **Deliverable**: Precise mathematical statement of $\mathcal{R}_\infty$ as a Riesz representable operator, or proof that no limit exists (branch B3).

**Week 2: Conservation Laws and Lyapunov**
- Apply the Structural Conservation Theorem (`physics/conservation.py`) in the presence of $\mathcal{R}_\infty$.
- Derive: Does Noether charge $Q = \int \rho dV$ (where ρ is structural density) remain conserved modulo $\mathcal{R}_\infty$?
- Derive: What is the Lyapunov functional $V(t)$ under $\mathcal{R}_\infty$ dynamics? Does it decay monotonically?
- **Deliverable**: Analytical expressions for $Q_\infty$ and $V_\infty$, validated against N12–N13 K_φ cascade data (retrospectively).

**Week 3: Spectrum and Universality**
- Compute (symbolically or numerically if needed) the spectrum $\{\lambda_n\}$ of $\mathcal{R}_\infty$ in a model space (e.g., harmonic oscillator basis, Fourier modes on torus).
- **Test hypothesis**: Is the eigenvalue density of $\lambda_n$ related to the zero distribution of zeta or to the Kolmogorov spectrum $E(k) \propto k^{-5/3}$?
- **Deliverable**: Rigorous statement of what REMESH-∞ predicts, with explicit bounds on error if the spectrum does NOT match RH or Kolmogorov.

### §18.5 Success Criteria (Locked, Branch-Dependent)

**Branch A (Operator exists, closed analysis):**
- $\mathcal{R}_\infty$ is rigorously defined on a natural Banach space.
- Conservation laws ($Q_\infty$, $V_\infty$) are derived with analytical proof.
- Spectrum computed exactly or with controlled asymptotics.
- **Outcome**: TNFR has a new fundamental principle (asymptotic coherence scaling). Secondary: RH/NS implications follow, if any.

**Branch B1 (Operator exists, spectrum matches RH or K41):**
- In addition to Branch A: eigenvalue density of $\mathcal{R}_\infty$ matches (a) predicted smooth-half zero density of zeta on Re(s)=1/2, or (b) Kolmogorov power law k^{-5/3}.
- **Outcome**: REMESH-∞ is universal attractor for both number-theoretic and hydrodynamic coherence. RH/NS become instances of TNFR universality.
- **Caveat**: This does NOT close RH or NS; it explains WHY they have the structure they do.

**Branch B2 (New operator required):**
- $\mathcal{R}_\infty$ cannot be derived from the 13 canonicals + nodal equation, even in the limit.
- A 14th operator must be postulated (and justified from first principles).
- **Outcome**: TNFR catalog is genuinely incomplete. Paths to closure for RH/NS require new physics.

**Branch B3 (No limit exists):**
- The sequence $\{\text{REMESH}_{\tau_g}\}$ as τ_g → ∞ does not converge in any Banach space topology.
- **Outcome**: Temporal coherence has fundamental limits. Multi-scale structure is intrinsically bounded by τ_g_max (possibly finite). Explains why N12–N14 plateau at certain observables.

### §18.6 Locked Scope Statement

N15 does **not**:

* Claim to prove the Riemann Hypothesis or resolve 3D Navier-Stokes global regularity.
* Derive a new canonical operator (if B2 is required, that is a separate foundational effort).
* Provide computational validation (that is N16 or later).
* Resolve the continuum limit or asymptotic analysis of any classical problem.

N15 **does**:

* Establish whether the 13-operator catalog with the REMESH-∞ asymptotic limit is closed under the nodal equation in the limit τ_global → ∞.
* Characterize the universal structure of multi-scale coherence from first principles (TNFR-intrinsic, not borrowed from RH or NS).
* Provide a definitive answer to whether Branch B1 (closed within catalog), Branch B2 (new operator), or Branch B3 (no limit) holds.

**Cross-reference**: This analysis is INDEPENDENT of whether RH is true or whether 3D NS has blow-up. It answers a structural question about TNFR itself.

---
