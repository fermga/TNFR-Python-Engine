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
| N≥12 | Higher-Re CF eigenframe transition (n≥48, DNS) **/** function-space convergence (NS-G1) **/** analytical NS-G2 bounds **/** discrete-to-continuum BKM (NS-G3) **/** structural TNFR construction of (ω·∇)u (NS-G4) **/** REMESH-∞ asymptotic study (§11) | open | unknown | unknown |

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
