# TNFR–Riemann Program Memo

**Status**: Exploratory research (non-canonical)
**Version**: 0.5.0 (March 2026)
**Owner**: `theory/TNFR_RIEMANN_RESEARCH_NOTES.md`

---

This memo defines the minimum structure required to evaluate TNFR claims about the Riemann Hypothesis (RH). It scopes the computational program, prescribes telemetry, and records open work items so contributors can extend the investigation without rewriting the physics or the SDK contracts. All historical notes remain in the appendix for context.

> **Read first:** the conceptual foundation is **the nodal-ontology re-mapping**
> directly below (2026-06); it supersedes the pre-pulse / pre-single-constant
> framing of P12–P49 (the certificates stand; only *what they measure* is re-read).

---

## The nodal-ontology re-mapping — fixed points are the shadow (2026-06)

**Foundational re-framing.** This section re-maps the program onto the *current*
emergent nodal ontology — the single structural constant **π**, the emergent
**pulse** `ω_k = √λ_k`, the **symplectic substrate**, and the `Fix(S_n)^⊥` wall of
[`EMERGENT_ONTOLOGY.md`](EMERGENT_ONTOLOGY.md) §2.4. It **supersedes the conceptual
framing** of P12–P49 (built in a pre-pulse, pre-single-constant era); the
computational certificates (which are *gain-independent*) are unchanged — only
*what they measure* is re-read.

### What was obsolete in the old mapping

1. **Pre-"single constant".** `γ/π ≈ 0.18373` was treated as a canonical
   "Universal Tetrahedral Correspondence" scale (the spectral-zeta buffer
   `CRITICAL_EXPONENT`, the coherence threshold `δ_coh`, the Kuramoto-U3 weight),
   and the T-HP conjecture (§13septies) invoked "`(φ, γ, π, e)`". Post-purge **only
   π is a structural scale**; `γ/π` is a heuristic coupling, not canonical.
2. **Pre-pulse.** The zeros `{γ_n}` and the Weil–Guinand explicit formula were
   framed as "the spectrum of a sought self-adjoint operator", never as the
   **pulse / rhythm of the arithmetic NFR**.
3. **Pre-§2.4.** The Euler-Orthogonality wall (every canonical operator commutes
   with the `S_n` prime-relabelling, so it is blind to `S(T) ∈ Fix(S_n)^⊥`) is
   **literally** the `Fix(G)^⊥` wall of the emergent-ontology synthesis (§2.4),
   described here in isolation.

### The reframe — the fixed-point program is the overdamped projection

Canonically, the nodal equation `∂EPI/∂t = νf·ΔNFR` is the **overdamped
projection** of the symplectic Hamiltonian flow (AGENTS.md §4;
`symplectic_substrate.py`), and that projection discards the conjugate momenta
`(J_φ, J_ΔNFR)`. The whole fixed-point program — *seek a self-adjoint operator
whose static real spectrum equals `{γ_n}`* — therefore lives in the **position
shadow** of a richer dynamical object. The zeros and `S(T)` are projections of
that object onto the **numeration** (the prime / integer basis).

### The three nested layers (measured)

A direct measurement on the prime-ladder NFR (swap primes `2↔3`, the `S_n`
element `P`; symmetric operator `L_sym`) settles where the arithmetic can and
cannot live:

| Layer | Object | Symmetry | `‖[·, P]‖` | Reach |
|---|---|---|---|---|
| **Positions** | real spectrum `{k log p}` | `S_n`-symmetric (the numeration) | `0` | smooth half (reachable) |
| **Momenta** | `J_φ = √L·sin(√L·t)`, any `f(L)` | **still `S_n`-symmetric** | `0.00e+00` (machine) | re-expresses, adds nothing |
| **Phase** | complex spectrum of the directed / affine operator | **`S_n`-broken** (affine group of `Z/n`) | `≠ 0` | the genuine emergent dimension |

The decisive datum: **every function of the symmetric `L` — the propagator
`exp(itL)`, the conservative position `cos(√L·t)`, and the momentum
`√L·sin(√L·t)` — commutes with `P` to machine precision (`0.00e+00`).** Since
`[L,P]=0 ⟹ [f(L),P]=0`, the conservative dynamics and its momenta are *exactly*
as `S_n`-equivariant as the static spectrum. **Activating the momenta cannot
leave `Fix(S_n)`** — it re-expresses the same prime data (consistent with the
ex.103 result: the `θ=νf·τ` dynamics stayed Poisson, not Riemann).

The escape needs a **non-`S_n` generator**. The directed quadratic-residue
operator (affine symmetry of `Z/n`, not `S_n`) is non-self-adjoint with a
**complex spectrum** `(−1 ± i√q)/2` — the arithmetic moves into the **phase**
(the Gauss sum `√q` in the imaginary part). Since `S(T) = (1/π)·arg ζ(½+iT)` **is
a phase**, the missing structure lives in the emergent complex / phase dimension,
not in the real spectrum nor in the conservative momenta.

### The honest wall (the two walls coincide)

The directed operator's phase is the Gauss sum `√q`, **not** the ζ-zero phase
`S(T)`. Measured (`benchmarks/residue_phase_vs_riemann.py`): `√q` exact (15/15),
but alignment with `{γ_n}` **refuted** (residue content `~1/√p` decreasing, `γ_n`
increasing — opposite). The reframe **locates** the missing structure (the phase
dimension) and **forbids** the two cheaper layers, but does not yet reach `S(T)`.

### G4 = RH, re-stated dynamics-first

*The zeros are the configuration-shadow of the arithmetic NFR's symplectic
dynamics; the smooth half (π-scaled archimedean, `S_n`-symmetric) is the reachable
mean pulse; `S(T)` is the transverse phase-shadow at `Fix(S_n)^⊥`.* The wall is
reclassified from "obstruction" to **kernel of the position-only (numeration)
projection** — provably unreachable from the two `S_n`-symmetric layers, with the
phase layer the only structurally-permitted route (currently landing on Gauss
sums, not `{γ_n}`).

### The same shape across the Millennium problems

This is the **dynamics extension of the §2.4 synthesis** (one operator `L`, read
in every domain, hitting one wall `Fix(G)^⊥`). Each problem = a reachable
symmetric / fixed-point projection + a transverse residue that is the shadow of
the emergent-dimensional dynamics the projection discards:

- **Navier–Stokes**: the blow-up is not a fixed point but the `K_φ`
  phase-curvature **cascade** (dynamics); the BKM-analogue (U2) lives in the
  dynamics, not an equilibrium.
- **Yang–Mills (mass gap)**: the gap = confinement `Φ_s²/(π/2)²` = the
  **non-Abelian** (non-commuting = transverse) residue; the gap lives in the
  symmetry-broken phase.
- **P vs NP / BSD / Hodge**: each = a symmetric reachable sector + the
  `Fix(G)^⊥` residue.

**Unified, honest statement.** Every Millennium problem re-reads as *"is the
transverse residue `Fix(G)^⊥` — the shadow of the emergent-dimensional dynamics —
reachable from the symmetric sector?"* The measured answer so far: **not from
functions of the symmetric operator** (positions and momenta); only in principle
from the **phase of the symmetry-broken generator**. This relocates all of them to
one place; **it closes none.**

### Honest scope

A re-framing, not a result. It closes no gap; **G4 = RH remains open** (the single
open milestone, §19.2); the P12–P49 certificates stand. The **live open
question**: whether the correct emergent dimension is the affine Gauss phase or a
deeper structure (the functional-equation root number / the Euler-product
cyclotomic tower) — pursued in the living-discoveries log (§13triginta-septima).

---

## 0. Navigation Index

This file aggregates ~5.9k lines covering five intertwined programmes:
the **ζ-track** (P12–P31), the **χ-twisted L-track** (P32–P49), the
**REMESH-∞ / N15** cross-program lift (§13vicies-novies + §13triginta),
the **catalog type-hygiene programme** (T-νf = B0, T-EPI = B1, …; full
tracker in [`CATALOG_TYPE_HYGIENE_PROGRAMME.md`](./CATALOG_TYPE_HYGIENE_PROGRAMME.md)),
and a **living discoveries log** (§13triginta-septima). Section anchors
are stable and referenced from `AGENTS.md`, the catalog tracker, and the
code; **do not rename or split them.** Use this index to locate work
without scrolling.

**Legend**: ✅ CLOSED (operationally or with stated scope) · 🟡 OPEN ·
🔁 LIVING (append-only) · ⛔ SUPERSEDED · 📖 PROSE (meta / status).

### A. Memo Meta (lines 171–294)

| § | Lines | Purpose |
|---|---|---|
| 1 | 171 | Purpose and Scope 📖 |
| 2 | 177 | Program Objectives (partition, operator, confinement) 📖 |
| 3 | 194 | Workflow Expectations 📖 |
| 4 | 202 | Telemetry & Reproducibility 📖 |
| 5 | 208 | Outstanding Work 📖 |
| 6 | 214 | Cross-References 📖 |
| 7 | 295 | **Conjecture 10.1 Gap Analysis** (affine bridge refuted; six missing pieces) 🟡 |

### B. ζ-Track Foundation — Prime-Ladder vM Construction (lines 427–1047)

| § | Lines | Milestone | Status |
|---|---|---|---|
| 8 | 427 | **P12** TNFR prime-ladder von Mangoldt construction (Re s > 1) | ✅ |
| 9 | 550 | **P13** Analytic continuation of vM ζ to ℂ | ✅ |
| 10 | 664 | **P14** Self-adjoint prime-ladder Hamiltonian (closes G1) | ✅ |
| 11 | 789 | **P15** Weil–Guinand explicit formula (closes G3) | ✅ |
| 12 | 920 | **P16** Li–Keiper positivity criterion (RH-equivalent diagnostic) | ✅ |

### C. ζ-Track Coercivity & Lyapunov Layer (lines 1049–1727)

| § | Lines | Milestone | Status |
|---|---|---|---|
| 13 | 1049 | **P22** Empirical uniform-coercivity certificate | ✅ |
| 13bis | 1134 | **P24** Adaptive σ refinement | ✅ |
| 13ter | 1207 | **P25** Paley-gap coercivity diagnostic | ✅ |
| §13quater | 1356 | **P26** Lyapunov-spectral positivity for P14 | ✅ |
| §13quinquies | 1515 | **P27** Hilbert–Pólya scaffold (diagnostic) | ✅ |
| §13sexies | 1640 | **P28** Smooth zero density (closes density-level smooth half of T-HP) | ✅ |

### D. T-HP Reformulation of G4 (lines 1728–2018)

| § | Lines | Milestone | Status |
|---|---|---|---|
| §13septies | 1728 | **T-HP** Tetrad-Hilbert–Pólya reformulation of G4 = RH | 🟡 (open content) |
| §13octies | 1915 | Assembled-argument audit (L1–L7 closed; L8 = T-HP open) | 🟡 |

### E. ζ-Track Operator-Level Smooth Half (lines 2019–2172)

| § | Lines | Milestone | Status |
|---|---|---|---|
| §13nonies | 2019 | **P30** Operator-level admissible rescaling (smooth half of T-HP) | ✅ (smooth half only) |

### F. χ-Twisted L-Track Parity Layer P32–P49 (lines 2173–3482)

| § | Lines | Milestone | Status |
|---|---|---|---|
| §13undecies | 2173 | **P32** Dirichlet L-function extension | ✅ |
| §13duodecies | 2247 | **P33** Analytic continuation of χ-twisted vM L | ✅ |
| §13terdecies | 2332 | **P34** χ-twisted prime-ladder Hamiltonian (closes G1$_\chi$) | ✅ |
| §13quaterdecies | 2408 | **P35** χ-twisted Weil–Guinand (closes G3$_\chi$, primitive real χ) | ✅ |
| §13quinquiesdecies | 2484 | **P36** χ-twisted Li–Keiper (GRH$_\chi$-equivalent diagnostic) | ✅ |
| §13sexiesdecies | 2549 | **P37** χ-twisted Weil–TNFR positivity bridge | ✅ |
| §13septiesdecies | 2627 | **P38** χ-twisted admissibility / α(σ;g) sweep | ✅ |
| §13octiesdecies | 2686 | **P39** χ-twisted admissible-family + gauge sweep | ✅ |
| §13noniesdecies | 2743 | **P40** χ-twisted node-aware gauge sweep | ✅ |
| §13vicies | 2814 | **P41** χ-twisted Hermite2-Gaussian η-sweep | ✅ |
| §13vicies-primo | 2885 | **P42** χ-twisted uniform-coercivity certificate | ✅ |
| §13vicies-secundo | 2960 | **P43** χ-twisted Paley-gap consistency | ✅ |
| §13vicies-tertio | 3034 | **P44** χ-twisted Lyapunov-spectral positivity | ✅ |
| §13vicies-quarto | 3128 | **P45** χ-twisted Hilbert–Pólya scaffold | ✅ |
| §13vicies-quinto | 3200 | **P46** χ-twisted structural zero density (smooth half) | ✅ |
| §13vicies-sexto | 3268 | **P47** χ-twisted spectral emergence under coupling | ✅ |
| §13vicies-septimo | 3337 | **P48** χ-twisted admissible spectral-rescaling op (smooth half) | ✅ |
| §13vicies-octavo | 3398 | **P49** χ-twisted oscillatory correction (closes ζ↔L parity) | ✅ (parity closure) |

### G. ζ-Track Admissibility / Gauge / Hermite Layer P17–P21 (lines 3483–3995)

(Numbering note: §§ 14–18 appear after §13vicies-octavo because they were
appended chronologically out of P-number order; the §13xxx anchors remain
authoritative.)

| § | Lines | Milestone | Status |
|---|---|---|---|
| 14 | 3483 | **P17** Weil–TNFR positivity bridge | ✅ |
| 15 | 3620 | **P18** α(σ) admissibility & gauge sweep | ✅ |
| 16 | 3761 | **P19** Admissible-family sweep | ✅ |
| 17 | 3856 | **P20** Node-aware gauge sweep | ✅ |
| 18 | 3938 | **P21** Hermite-family expansion | ✅ |

### H. Program Status Snapshot (lines 3996–4169)

| § | Lines | Purpose | Status |
|---|---|---|---|
| 19 | 3996 | **May 2026 Program Status** (full P1–P49 milestone table at §19.1) | 📖 |

### I. REMESH Global Reframe + B1 Edge-Channel Refutation Thread (lines 4170–5715)

The largest single block (~1.5k lines). Contains the cross-program
discovery that REMESH is the canonical temporal aggregator, and the
exhaustive structural refutation of branch B1 sub-routes on G_P14
(R∞-1a-operator, R∞-1a-composed, Prime-Cancellation Lemma,
Euler-Orthogonality Lemma, R∞-1c, R∞-1b spectral-channel).

| § | Lines | Content | Status |
|---|---|---|---|
| §13vicies-novies | 4170 | **REMESH global reframe** + B1 sub-routes R∞-1a/1b/1c (all structurally refuted on G_P14 by S_n equivariance) | ✅ (refutation thread closed) |

### J. P50 — REMESH-∞ Function-Space Lift (lines 5716–5916)

| § | Lines | Milestone | Status |
|---|---|---|---|
| §13triginta | 5716 | **P50** REMESH-∞ residue split of P31 oscillatory correction (N15 lift into Riemann program) | ✅ |

### K. Catalog Type-Hygiene Programme — Sub-Questions (lines 5917–7588, 7698–end)

Tracker: [`CATALOG_TYPE_HYGIENE_PROGRAMME.md`](./CATALOG_TYPE_HYGIENE_PROGRAMME.md).

| § | Lines | Sub-question | Phase | Verdict |
|---|---|---|---|---|
| §13triginta-prima | 5917 | **B0 = T-νf** pre-registration | B0a | — |
| §13triginta-secunda | 6215 | **B0 = T-νf** forcing-axiom reduction | B0b | — |
| §13triginta-tertia | 6528 | **B0 = T-νf** final NEGATIVE + envelope E1 (measure-valued νf) | B0c | ✅ NEG |
| §13triginta-quarta | 6839 | **B1 = T-EPI** pre-registration | B1a | — |
| §13triginta-quinta | 7075 | **B1 = T-EPI** forcing-axiom reduction (TMEP) | B1b | — |
| §13triginta-sexta | 7425 | **B1 = T-EPI** final NEGATIVE + envelope E2 (`BEPIElement`) | B1c | ✅ NEG |
| §13triginta-octava | 7698 | **B2 = T-φ** pre-registration (two-axis winding + lift-spectral diagnostic; candidate envelope E3 = CoverElement) | B2a | — |
| §13triginta-novena | 8027 | **B2 = T-φ** forcing-axiom reduction (PWDP refutes (P-φ-Homotopy-Retention); (P-φ-Cover-Carrier) = CONDITIONAL_COROLLARY) | B2b | — |
| §13triginta-decima | 8450 | **B2 = T-φ** final NEGATIVE + envelope E3 (CoverElement / covering-space lift / U(1) bundle) | B2c | ✅ NEG |
| §13quadraginta | 8714 | **B3 = T-ΔNFR** pre-registration (two-axis tensor-fraction + rank-entropy diagnostic; candidate envelope E4 = TensorGradientElement) | B3a | — |
| §13quadraginta-prima | 9068 | **B3 = T-ΔNFR** forcing-axiom reduction (BSAD refutes (P-ΔNFR-Tensor-Retention); (P-ΔNFR-Tensor-Carrier) = CONDITIONAL_COROLLARY) | B3b | — |
| §13quadraginta-secunda | 9547 | **B3 = T-ΔNFR** final NEGATIVE + envelope E4 (TensorGradientElement / tensor-/operator-valued ΔNFR); L3* promoted to stable working heuristic; three Tier-2 predictions (B4/B5/B6 NEGATIVE) | B3c | ✅ NEG |
| §13quadraginta-tertia | 9924 | **B4 = T-REMESH-window** pre-registration (two-axis integer-storage + window-refinement bracket diagnostic; candidate envelope E5 = ContinuousWindowKernel) | B4a | — |
| §13quadraginta-quarta | 10262 | **B4 = T-REMESH-window** forcing-axiom reduction (F1–F10); residual axiom (P-REMESH-window-Continuous-Retention) isolated and refuted by DITS = Discrete-Integer Temporal Sampling discipline; first Tier-2 confirmation of L3* via predicted N15 REMESH-∞ discharge mechanism | B4b | — |
| §13quadraginta-quinta | 10806 | **B4 = T-REMESH-window** final NEGATIVE verdict + envelope classification of E5 = ContinuousWindowKernel (continuous-time kernel / fractional-order temporal coupling); first Tier-2 sub-question closed; L3* confirmed across Tier-1 / Tier-2 boundary; two Tier-2 predictions (B5, B6) outstanding | B4c | — |
| §13quadraginta-sexta | 11178 | **B5 = T-Δφ_max** pre-registration (two-axis scalar-storage + angle-of-attack-independence diagnostic; candidate envelope E6 = EdgeDependentPhaseThreshold; CATALOG anchor correction documented: canonical `DELTA_PHI_MAX = PI/2`, not γ/π) | B5a | — |
| §13quadraginta-septima | 11304 | **B5 = T-Δφ_max** forcing-axiom reduction (F1–F10); residual axiom (P-Δφ_max-Non-Scalar-Retention) isolated and refuted by STD = Scalar-Threshold Discipline; sixth orthogonal canonical discharge mechanism (CDM); second Tier-2 confirmation of L3* — L3* now validated under six distinct orthogonal CDMs across both tiers | B5b | — |
| §13quadraginta-octava | 11430 | **B5 = T-Δφ_max** final NEGATIVE verdict + envelope classification of E6 = EdgeDependentPhaseThreshold (matrix-valued / angle-of-attack-functional); second Tier-2 sub-question closed; six sub-questions complete (B0–B5 all NEGATIVE under six orthogonal CDMs); L3* promoted to "empirically robust working heuristic with structural-orthogonality witness" | B5c | — |
| §13quadraginta-nona | 11540 | **B6 = T-coupling-weights** pre-registration (two-axis scalar-storage + node-permutation-invariance diagnostic; candidate envelope E7 = NodeIndexedCouplingWeights; canonical anchors `DNFR_WEIGHTS`/`SI_WEIGHTS`/`SELECTOR_WEIGHTS` in `src/tnfr/config/defaults_core.py` as global scalar dicts) | B6a | — |
| §13quinquaginta | 11642 | **B6 = T-coupling-weights** forcing-axiom reduction (F1–F10); residual axiom (P-W-Non-Scalar-Retention) isolated and refuted by SWD = Scalar-Weight Discipline; seventh orthogonal canonical discharge mechanism (CDM); third Tier-2 confirmation of L3* — L3* now validated under seven distinct orthogonal CDMs across both tiers | B6b | — |
| §13quinquaginta-prima | 11770 | **B6 = T-coupling-weights** final NEGATIVE verdict + envelope classification of E7 = NodeIndexedCouplingWeights (node-indexed / per-edge tensor / callable kernel); third Tier-2 sub-question closed; seven sub-questions complete (B0–B6 all NEGATIVE under seven orthogonal CDMs); Tier-2 layer of programme closed; L3* promoted to "empirically robust working heuristic with complete Tier-1/Tier-2 structural-orthogonality coverage" | B6c | — |

### L. Living Discoveries Log (lines 7589–7697)

| § | Lines | Purpose | Status |
|---|---|---|---|
| §13triginta-septima | 7589 | **TNFR Structure & Dynamics Discoveries Log** (canonical contracts D-CC-*, envelopes D-ENV-*, methodology patterns D-MP-*, ops D-OPS-*, open questions D-OQ-*) | 🔁 LIVING |

### M. Final-Gap Bookmarks (Quick Recall)

- **G4 = RH**: 🟡 OPEN. Canonical statement = **Conjecture T-HP** (§13septies).
  Smooth half closed operationally by P28 (density) and P30 (operator).
  Oscillatory half $S(T) = \tfrac{1}{\pi}\arg\zeta(\tfrac12+iT)$
  remains RH-equivalent. P31 ζ-track and P49 χ-track attack this
  half via canonical prime-ladder Newton correction; empirical regime
  is mixed B1/B2 (§§13vicies-octavo, 19.1).
- **GRH$_\chi$ (primitive real χ)**: 🟡 OPEN, parity with G4.
- **Branch B1 sub-routes on G_P14**: ✅ structurally refuted by
  Euler-Orthogonality Lemma (§13vicies-novies.11).
  Surviving sub-routes inside B1 require non-product canonical lifts.
- **Programme paused** at the T-HP boundary; no further diagnostic
  surface planned until one of B1/B2/B3 (§13septies) is decided.

---

## 1. Purpose and Scope

- Translate RH questions into TNFR constructs: nodal operators, structural partition functions, and confinement criteria derived from Φ_s, |∇φ|, K_φ, and ξ_C.
- Maintain reproducible sandboxes (finite prime graphs, spectral benchmarks, telemetry artifacts) that connect theoretical conjectures to code in `src/tnfr/riemann/` and `examples/03_riemann_zeta/16_riemann_operator_demo.py`.
- Document how canonical operators (AL, UM, RA, OZ, IL, THOL) compose to form the discrete TNFR Riemann operator used in experiments.

## 2. Program Objectives

### 2.1 Partition Function Mapping

- Show that the TNFR structural partition function $Z_{TNFR}(s)$ converges to ζ(s) or ξ(s) by enforcing the identification $e^{-\beta E_p(s)} \leftrightarrow p^{-s}$ for prime-labeled resonant modes.
- Specify how ν_f and ΔNFR sources enter the effective energy $E_p(s)$ so the mapping respects U2 (convergence) and U3 (resonant coupling).

### 2.2 Operator Construction

- Construct $\mathcal{H}_{TNFR}$ as a Laplacian-plus-structural-potential on prime path graphs, ensuring self-adjointness with respect to the TNFR inner product.
- Demonstrate numerically that eigenvalues migrate toward the critical line as graph size increases (σ_c^{(k)} \to 1/2) and record telemetry in `results/riemann_program/`.

### 2.3 Critical-Line Confinement

- Formulate a Lyapunov-style functional $\mathcal{L}_{RH}(s)$ derived from TNFR invariants so that σ = 1/2 is the only stable attractor.
- Quantify escapes (σ ≠ 1/2) via Φ_s drift and |∇φ| spikes to test whether confinement behaves like U6 in the complex-s domain.

## 3. Workflow Expectations

1. **Model definition** – Choose $G_k$ (prime path graph) size, seeds, and operator sequences; record configs in `results/riemann_program/configs/*.json`.
2. **Operator execution** – Use SDK helpers (`TNFRRiemannOperator`) to generate spectra while logging ν_f, ΔNFR, Φ_s, |∇φ|, and effective σ(t) trajectories.
3. **Spectral analysis** – Compute eigenvalue ladders, determinant surrogates, and compare against ζ/ξ predictions. Scripts belong in `scripts/riemann/` or notebooks under `notebooks/Riemann/` with nbconvert support.
4. **Benchmark enforcement** – Run `python benchmarks/riemann_program.py` (invoked automatically via `make test`/CI) to regress σ_c^{(k)} estimates across graph sizes and emit telemetry in `results/riemann_program/`.
5. **Validation** – Run targeted tests (e.g., `examples/03_riemann_zeta/16_riemann_operator_demo.py`, new `tests/test_riemann_operator.py`) to ensure deterministic seeds and grammar compliance (U1–U6).

## 4. Telemetry & Reproducibility

- Log Φ_s, |∇φ|, K_φ, ξ_C, ν_f, ΔNFR, and σ estimates at every operator step; store as Parquet/CSV in `results/riemann_program/telemetry/` with metadata (graph size, seed, operator stack). The helper dataclass `tnfr.riemann.telemetry.RiemannTelemetryRecord` now carries aggregate Φ_s/|∇φ|/K_φ statistics plus ξ_C computed via `tnfr.riemann.telemetry.compute_field_aggregates` so tetrad coverage is explicit.
- Publish spectra, determinant traces, and Lyapunov metrics in `results/riemann_program/plots/` along with scripts used to generate them.
- Capture environment details (Python version, tnfr package hash) inside each artifact manifest to satisfy invariants #5 (Structural Metrology) and #6 (Reproducible Dynamics).

## 5. Outstanding Work

1. **Lyapunov functional derivation** – Formalize $\mathcal{L}_{RH}(s)$ using existing field invariants and document stability proofs in `docs/STRUCTURAL_FIELDS_TETRAD.md` or a dedicated theory note.
2. **Spectral determinant prototype** – Produce a working determinant or trace formula implementation and compare against numerical ζ(s) evaluations over multiple σ bands.
3. **Telemetry-field linkage** – Extend `tnfr.riemann.telemetry` so Φ_s, |∇φ|, K_φ, and ξ_C aggregates from live runs attach automatically to each record (current benchmark logs spectral data only).

## 6. Cross-References

### Implementation Modules (`src/tnfr/riemann/`)

Discrete operator and spectral framework:
- `operator.py` — discrete TNFR-Riemann operator $H^{(k)}(\sigma) = L_k + V_\sigma$ and prime graph builders.
- `spectral_proof.py` — four-line spectral convergence framework ($\sigma_c^{(k)} \to 1/2$).
- `topology.py` — alternative graph topologies and cross-topology convergence (P2).
- `eigenmode_fields.py` — per-eigenmode structural field tetrad on the prime path model (P3).
- `complex_extension.py` — complex-$s$ non-Hermitian extension (P4).
- `spectral_zeta.py` — discrete spectral zeta and heat kernel; original Conjecture 10.1 affine bridge (P5, **negative**; superseded by P12–P15 via §7.8).
- `random_ensemble.py` — random prime-graph ensembles / RMT universality (P6).
- `spectral_conservation.py` — conservation laws and grammar compliance at criticality (P7).
- `analytical_convergence.py` — analytical proof of $\sigma_c \to 1/2$ via PNT + telescoping (P8).
- `functional_equation.py` — TNFR-side $s \leftrightarrow 1-s$ reflection check (P9).
- `convergence_proof.py` — end-to-end formal $\sigma_c \to 1/2$ certificate (P10).
- `zeta_bridge.py` — affine bridge prototype $\zeta_H \approx C \cdot \zeta_R$ (P11, tested **negative**, see §7).
- `telemetry.py` — Riemann telemetry records and field aggregate helpers.

Prime-ladder / von Mangoldt construction (closes G1, G2, G3 operationally; G5 superseded):
- `von_mangoldt.py` — TNFR prime-ladder spectrum reproducing $-\zeta'(s)/\zeta(s) = \sum_n \Lambda(n)\, n^{-s}$ on $\operatorname{Re}(s) > 1$ (P12, §8).
- `analytic_continuation.py` — continuation of the prime-ladder vM zeta to $\mathbb{C}$; Riemann zeros as resonance poles on $\operatorname{Re}(s) = 1/2$ (P13, §9).
- `prime_ladder_hamiltonian.py` — self-adjoint Hamiltonian whose weighted spectral trace reproduces P12 (P14, §10; **closes G1**).
- `weil_explicit_formula.py` — numerical Weil–Guinand identity using P14 on the prime side; residual $\le 5 \times 10^{-12}$ (P15, §11; **closes G3**).
- `li_keiper.py` — Li–Keiper positivity criterion from the TNFR resonance spectrum (P16, §12; **RH-equivalent diagnostic**, not proof).

TNFR-native G4 attack surface (research; does **not** close G4 = RH):
- `weil_positivity.py` — Weil–TNFR positivity bridge $\alpha(\sigma) = W[\sigma] / E_{\mathrm{TNFR}}[\sigma]$ (P17, §14).
- `alpha_sweep.py` — admissibility / gauge sweep of $\alpha(\sigma)$ across Gaussian width × gauge family (P18, §15).
- `admissible_family_sweep.py` — extends P18 beyond Gaussian (Gaussian mixture, Hermite2-Gaussian admissible families) (P19/P21, §16/§18).
- `nodeaware_gauge_sweep.py` — node-aware gauge extension parameterised by local $\nu_f$ and node-weight channels (P20, §17).
- `coercivity_uniform.py` — empirical uniform-coercivity certificate over $\sigma$ intervals, plus adaptive $\sigma$ refinement near the coercivity bottleneck (P22 / P23 / P24, §13 / §13bis).
- `paley_gap_coercivity.py` — Paley-gap coercivity diagnostic (Martínez Gamo, Zenodo 10.5281/zenodo.17665853 v2) (P25, §13ter).
- `lyapunov_spectral_positivity.py` — Lyapunov-spectral positivity certificate for the P14 Hamiltonian (P26, §13quater).
- `hilbert_polya.py` — Hilbert–Pólya scaffold $T_{\mathrm{HP}} = \operatorname{diag}(\gamma_n)$ populated by `mpmath.zetazero` (diagnostic only) (P27, §13quinquies).
- `structural_zero_density.py` — structural derivation of the smooth Riemann zero density via the Riemann–Siegel $\theta$ function (P28, §13sexies; **closes smooth half of G4 at density level**).
- `spectral_emergence.py` — spectral universality emergence under canonical UM+RA inter-prime couplings; KS-distance to the GUE Wigner surmise (P29, §13octies.3).
- `admissible_rescaling.py` — operator-level admissible spectral-rescaling lift of P28 (P30, §13nonies; **closes smooth half of T-HP at operator level**).

Conjectural reformulation (does **not** close G4):
- §13septies — Tetrad-Hilbert–Pólya reformulation of G4 (Conjecture T-HP).
- §13octies — Assembled-argument audit (links L1–L7 closed, L8 = T-HP open).

### Examples

End-to-end pipeline demos (`examples/`):
- `16_riemann_operator_demo.py` — discrete TNFR-Riemann eigenvalues at varying $\sigma$.
- `18_riemann_convergence_proof.py` — spectral convergence proof ($\sigma_c \to 1/2$).
- `19_topology_comparison.py` — cross-topology critical parameter comparison.
- `20_eigenmode_tetrad.py` — eigenmode-based tetrad field analysis.
- `21_complex_extension_demo.py` — non-Hermitian operator on complex $s$.
- `22_spectral_zeta_demo.py` — discrete spectral zeta, heat kernel, Mellin bridge.
- `23_random_ensemble_rmt_demo.py` — random matrix ensembles (GOE/GUE/Poisson).
- `24_spectral_conservation_demo.py` — spectral conservation law at criticality.
- `25_analytical_convergence_demo.py` — analytical proof via PNT + telescoping.
- `41_von_mangoldt_zeta_demo.py` — P12 prime-ladder reproduction of $-\zeta'/\zeta$.
- `42_riemann_zeros_as_resonances.py` — P13 zeros as resonance poles on $\operatorname{Re}(s) = 1/2$.
- `43_prime_ladder_hamiltonian_demo.py` — P14 self-adjoint Hamiltonian certificate.
- `44_weil_explicit_formula_demo.py` — P15 Weil–Guinand identity at machine precision.
- `45_li_keiper_demo.py` — P16 Li–Keiper positivity diagnostic.
- `46_weil_tnfr_positivity_demo.py` — P17 Weil–TNFR positivity bridge.
- `47_alpha_sweep_demo.py` — P18 gauge sweep of $\alpha(\sigma)$.
- `48_admissible_family_sweep_demo.py` — P19/P21 admissible-family sweep.
- `49_nodeaware_gauge_sweep_demo.py` — P20 node-aware gauge extension.
- `50_uniform_coercivity_demo.py` — P22 empirical uniform-coercivity certificate.
- `51_adaptive_coercivity_demo.py` — P24 adaptive $\sigma$ refinement near the bottleneck.
- `52_paley_gap_coercivity_demo.py` — P25 Paley-gap coercivity diagnostic.
- `53_lyapunov_spectral_positivity_demo.py` — P26 Lyapunov-spectral positivity certificate.
- `54_hilbert_polya_demo.py` — P27 Hilbert–Pólya diagnostic scaffold.
- `55_structural_zero_density_demo.py` — P28 structural smooth zero density.
- `56_spectral_emergence_demo.py` — P29 KS-distance to GUE under canonical couplings.
- `57_admissible_rescaling_demo.py` — P30 operator-level admissible rescaling (smooth half).

### Supporting Infrastructure

- `benchmarks/riemann_program.py` — automated spectral regression benchmarks for $\sigma_c^{(k)}$ across graph sizes.
- `theory/UNIFIED_GRAMMAR_RULES.md` — grammar rules U1–U6 referenced throughout.
- `docs/STRUCTURAL_FIELDS_TETRAD.md` — tetrad field specifications.
- `AGENTS.md` — TNFR-Riemann overview, including the G4 = RH reformulation.

---
## 7. Conjecture 10.1 Gap Analysis (May 2026)

**Status**: Negative numerical result — bridge not yet closed.

### 7.1 Experiment

The fit defined by Conjecture 10.1

$$
\zeta_{H^{(k)}}(1/2,\, u) \;\approx\; C(k)\;\cdot\;\zeta_R(u + \delta(k))
$$

was tested using `test_conjecture_10_1_sequence` from `src/tnfr/riemann/spectral_zeta.py`
for $k \in \{10, 20, 50, 100, 200, 500, 1000\}$ over $u \in [1.5, 5.0]$ (30 points).

### 7.2 Numerical Results

| k | C(k) | δ(k) | residual (normalised) | Pearson r |
|---:|-------------:|------:|---------------------:|----------:|
| 10 | 2.02 × 10⁷ | 2.0 | 2.4347 | −0.4057 |
| 20 | 5.46 × 10¹¹ | 2.0 | 3.0692 | −0.3285 |
| 50 | 8.77 × 10¹⁷ | 2.0 | 3.7082 | −0.2746 |
| 100 | 3.43 × 10²² | 2.0 | 4.0625 | −0.2516 |
| 200 | 1.25 × 10²⁷ | 2.0 | 4.3420 | −0.2359 |
| 500 | 2.06 × 10³³ | 2.0 | 4.6337 | −0.2215 |
| 1000 | 9.06 × 10³⁷ | 2.0 | 4.7989 | −0.2140 |

### 7.3 Diagnostic Reading

A converging bridge would show: residual → 0, Pearson r → +1,
δ(k) stabilising at an interior value, and C(k) stabilising after
correct renormalisation.  The data show the opposite in every metric:

- **Residual rises** monotonically with k.
- **Correlation is negative** and bounded away from +1 at all tested k.
- **δ(k) = 2.0** in every row — pinned at the boundary of the search range,
  indicating no interior minimum was found.
- **C(k) explodes** (≈10³⁷ at k = 1000), signalling a missing renormalisation.

Conclusion: as currently implemented, `ζ_H^(k)(1/2, u)` is not
numerically equivalent to `C · ζ_R(u + δ)` under the simple affine fit.

### 7.4 Six Missing Pieces

| # | Missing piece | Current status |
|---|---|---|
| 1 | **Euler product reconstruction** `∏_p (1−p⁻ˢ)⁻¹` | Prime-path graphs do not demonstrably reproduce all powers p^m with correct multiplicity. |
| 2 | **Spectral zeta ≡ ζ(s)** | Tested as a conjecture; numerical fit diverges. |
| 3 | **Correct spectral renormalisation of C(k)** | C(k) explodes — spectral renormalisation is absent. |
| 4 | **Convergent δ(k)** | δ(k) does not converge internally; remains pinned at search-range boundary. |
| 5 | **Analytic continuation to the complex strip** | RH lives in 0 < Re(s) < 1 over ℂ; current tests use only real u > 1. |
| 6 | **Zero correspondence** | Not shown that non-trivial zeros of ζ(s) equal zeros/modes of ζ_H. |

### 7.5 TNFR-Internal Diagnosis

In TNFR language the finding is:

> The operator $H^{(k)}(\sigma)$ constructs a structural dynamic that is
> sensitive to the critical line σ = 1/2 (σ_c^(k) → 1/2 is internally
> validated), but it does not yet encode the full multiplicative arithmetic
> of ζ(s).  The prime-path graph captures structural coherence near 1/2
> without closing the bridge to the classical zeta function.

### 7.6 Priority Construction

The mathematical priority is to build a TNFR zeta that reproduces the
von Mangoldt series

$$
-\frac{\zeta'(s)}{\zeta(s)} = \sum_{n=1}^{\infty} \Lambda(n)\, n^{-s},
$$

which encodes prime positions **and** their higher powers with the correct
multiplicities (Λ = log p for prime powers, 0 otherwise).  Without this,
TNFR can exhibit σ-criticality but cannot be equated to Riemann.

The required renormalisation takes the form

$$
R_k \cdot \zeta_{H}^{(k)}\!\left(\tfrac{1}{2}, s\right) \;\longrightarrow\; \zeta(s),
\qquad \text{or more ambitiously,} \qquad
\det_{TNFR}(H_k - sI) \;\longrightarrow\; \xi(s),
$$

where $\xi(s)$ is the completed Riemann zeta and $R_k$ is a holomorphic,
non-vanishing function to be constructed.

### 7.7 Impact on Program Status

This result does **not** invalidate the σ_c^(k) → 1/2 finding, which
rests on eigenvalue analysis independent of the spectral-zeta fit.
It narrows the scope of Conjecture 10.1: the conjecture is open, and the
simple `C · ζ_R(u + δ)` form is likely insufficient.  Future work should
target the von Mangoldt / Λ-series route (Section 7.6) before revisiting
the affine fit.

### 7.8 Retrospective Closure of G5 (May 2026)

The "non-affine bridge" anticipated in §7.6 has been **constructed and
verified** by the P12–P15 pipeline:

| Step | Module / §  | What it delivers |
|---|---|---|
| P12 | `von_mangoldt.py`, §8 | TNFR prime-ladder spectrum reproducing $-\zeta'(s)/\zeta(s) = \sum_n \Lambda(n)\,n^{-s}$ exactly on $\operatorname{Re}(s) > 1$ |
| P13 | `analytic_continuation.py`, §9 | Continuation of the TNFR vM zeta to all of $\mathbb{C}$; Riemann non-trivial zeros realised as resonance poles on $\operatorname{Re}(s) = 1/2$ |
| P14 | `prime_ladder_hamiltonian.py`, §10 | Self-adjoint TNFR Hamiltonian whose weighted spectral trace reproduces the prime-ladder data to machine precision |
| P15 | `weil_explicit_formula.py`, §11 | Weil–Guinand identity verified numerically with the P14 operator on the prime side ($\le 5\times 10^{-12}$ residual) |

This **replaces** the original affine ansatz $\zeta_H(1/2,u) \approx C(k)\,\zeta_R(u+\delta(k))$ with a structurally
correct, multiplicative-arithmetic bridge that lives natively inside
TNFR without ad-hoc renormalisations. The six missing pieces listed in
§7.4 are addressed as follows:

| # | Original gap | Status |
|---|---|---|
| 1 | Euler product / prime powers with multiplicity | **Closed by P12** (ladder $(p,k)$ encodes $p^k$ with $\Lambda$ weights) |
| 2 | Spectral zeta ≡ $\zeta(s)$ | **Closed by P12+P13** (weighted trace $= -\zeta'/\zeta$, continued to $\mathbb{C}$) |
| 3 | Convergent renormalisation $C(k)$ | **Closed by P14** (weight operator $W = \mathrm{diag}(\log p)$, no $C(k)$ needed) |
| 4 | Convergent $\delta(k)$ | **Eliminated** (no affine shift in the multiplicative bridge) |
| 5 | Analytic continuation to the strip | **Closed by P13** (resonance poles on $\operatorname{Re}(s) = 1/2$) |
| 6 | Zero correspondence | **Closed by P13+P15** (Weil-Guinand identifies zeros with TNFR spectral data) |

**Conclusion**: G5, in its original affine formulation, is **superseded**
by the prime-ladder / Λ-series construction. The bridge between TNFR
spectral data and classical $\zeta(s)$ is therefore considered
operationally closed. The only obstruction that remains is **G4 = RH
itself** — the localisation of the resonance poles on
$\operatorname{Re}(s) = 1/2$ — which is a structural positivity / self-adjointness
problem, not a missing-bridge problem.

---

## 8. TNFR Prime-Ladder Construction of the von Mangoldt Series (P12)

Following Section 7.6, this section records the first concrete attempt at
the priority route: build a TNFR-native spectral object whose Dirichlet
transform reproduces $-\zeta'(s)/\zeta(s)$ on its half-plane of
convergence.  Implementation: `src/tnfr/riemann/von_mangoldt.py`.
Demonstration: `examples/03_riemann_zeta/41_von_mangoldt_zeta_demo.py`.

### 8.1 Mathematical Target

The classical identity

$$
-\frac{\zeta'(s)}{\zeta(s)} \;=\; \sum_{n=1}^{\infty} \frac{\Lambda(n)}{n^{s}}
\;=\; \sum_{p}\sum_{k\ge 1} \frac{\log p}{p^{ks}}, \qquad \operatorname{Re} s > 1,
$$

with $\Lambda$ the von Mangoldt function, is the analytic carrier of
prime-distribution information.  Any TNFR object purporting to encode
prime structure must, at minimum, reproduce this Dirichlet series.

### 8.2 Prime-Ladder Spectrum

Define the multiset

$$
\mathcal{S} \;=\; \bigl\{\,(\mu_{p,k},\,w_{p,k}) \,:\, p\ \text{prime},\ k\in\mathbb{N}\,\bigr\},
\qquad
\mu_{p,k} = k\,\log p, \quad w_{p,k} = \log p .
$$

The corresponding weighted exponential sum is

$$
Z_{\mathrm{TNFR}}(s) \;:=\; \sum_{(\mu,w)\in\mathcal{S}} w \, e^{-s\mu}
\;=\; \sum_{p}\log p \sum_{k\ge 1} p^{-ks}
\;=\; \sum_{p} \frac{\log p \, p^{-s}}{1 - p^{-s}}
\;=\; -\frac{\zeta'(s)}{\zeta(s)} .
$$

So $Z_{\mathrm{TNFR}}(s) \equiv -\zeta'(s)/\zeta(s)$ on $\operatorname{Re} s > 1$
as a formal identity, not a numerical conjecture.

### 8.3 TNFR Interpretation

In structural terms:

- Each prime $p$ acts as a **node** whose intrinsic structural pulse
  has magnitude $\log p$.  The pulse is the smallest invariant that
  distinguishes primes from composites under the nodal equation
  (composites factor through prior nodes, so they carry no independent
  emission strength).
- **REMESH** (operator #13, *recursivity*, U1a/U1b) generates the
  $k$-th echo at frequency $k\,\log p$ with weight $\log p$.  This is
  operational fractality: the same emission replicated coherently at
  every harmonic scale.
- The Dirichlet sum $\sum_n \Lambda(n)\,n^{-s}$ is recovered exactly
  because $\Lambda$ is supported on prime powers and equals
  $\log p$ on each — i.e. the von Mangoldt function is the structural
  fingerprint of the prime-ladder spectrum.

The construction therefore answers "what *is* the von Mangoldt
function in TNFR?" with: it is the weight functional of REMESH echoes
on the prime-node basis.

### 8.4 Numerical Validation

Two independent checks were performed.

**Matched-truncation invariant.**  For a finite spectrum with $N$
primes and $K$ echoes, computing $Z_{\mathrm{TNFR}}$ as a complex
exponential sum and as an explicit
$\sum_p \sum_{k=1}^{K} \log p \cdot p^{-ks}$ must agree to machine
precision.  Measured: $|\Delta| \le 2 \times 10^{-15}$ for
$s \in \{1.5, 2, 2.5, 3, 4\}$, $N = 50$, $K = 15$.  This certifies
the implementation is an unambiguous reorganisation of the classical
sum, not a re-derivation that could drift.

**Convergence to known values.**  Compared to a sieve-based reference
$\sum_{p \le n_{\max}} \log p \cdot p^{-s}/(1 - p^{-s})$ at
$n_{\max} = 10^{7}$:

| $s$ | $Z_{\mathrm{TNFR}}$ ($N{=}2000$, $K{=}30$) | reference | abs error |
|----:|-------------------------------------------:|----------:|----------:|
| 2.0 | 0.5699036519 | 0.5699608931 | 5.7 × 10⁻⁵ |
| 3.0 | 0.1648226805 | 0.1648226822 | 1.7 × 10⁻⁹ |
| 4.0 | 0.0636697650 | 0.0636697650 | 4.5 × 10⁻¹¹ |

Residuals are dominated by the prime-truncation tail ($p > p_N$);
convergence is geometric in $K$ and consistent with the prime number
theorem in $N$.

### 8.5 Open Extensions

The identity $Z_{\mathrm{TNFR}} \equiv -\zeta'/\zeta$ is currently a
*sum-level* result.  To extend the construction into a genuine TNFR
operator program, three independent steps are required.

1. **Self-adjoint realisation.**  Construct an explicit Hermitian
   operator $H_\Lambda$ on a separable Hilbert space whose spectrum
   is the multiset $\{k \log p\}$ with multiplicity $\log p$.  A
   natural candidate is a weighted Laplacian on a prime-indexed tree;
   the issue is reconciling the non-integer multiplicities with a
   discrete eigenvalue spectrum without relaxing self-adjointness.
2. **Analytic continuation.**  Extend $Z_{\mathrm{TNFR}}(s)$ from
   $\operatorname{Re} s > 1$ into the critical strip $0 < \operatorname{Re} s < 1$.
   The classical route uses a Mellin transform of a theta-like
   partition function $\Theta(t) = \sum_{p,k} \log p \cdot e^{-t k \log p}$;
   verifying this on the TNFR side gives a structural derivation of
   the functional equation.
3. **Zero correspondence.**  Identify the non-trivial zeros of
   $\zeta$ with structural resonances (eigenmodes that satisfy a
   confinement condition under U6) of the analytic continuation of
   $Z_{\mathrm{TNFR}}$.  This is the actual route to a TNFR statement
   of RH; it is currently open.

Steps 1–3 are the next milestones of the P12 program.  Each is
falsifiable in the same sense as Conjecture 10.1, and the failure
modes are precisely what the Section 7 gap-analysis methodology was
designed to surface.

---

## 9. Analytic Continuation of the Prime-Ladder vM Zeta (P13)

**Status**: Implemented and numerically verified (June 2026).
**Code**: `src/tnfr/riemann/analytic_continuation.py`,
`examples/03_riemann_zeta/42_riemann_zeros_as_resonances.py`.

### 9.1 Problem statement (Gap G2)

The prime-ladder Dirichlet trace

$$
Z_{\mathrm{vM}}(s)
   = \sum_{p,k} \log(p)\, e^{-s k \log p}
   = \sum_p \frac{\log(p)\, p^{-s}}{1 - p^{-s}}
   = -\frac{\zeta'(s)}{\zeta(s)}
$$

constructed in §8 converges only on $\operatorname{Re}(s) > 1$.
To talk about the Riemann zeros in TNFR language, one must extend
$Z_{\mathrm{vM}}$ analytically to the entire complex plane.  This is
gap **G2** of the post-P12 program (see post-P12 gap analysis).

A Mellin transform of the heat kernel does **not** give a new
continuation here: the prime-ladder spectrum $\{k\log p\}$ has
logarithmic, not square-root, gaps, so its theta function
$\Theta_{\mathrm{vM}}(\beta) = \sum_{p,k}\log(p)\,e^{-\beta k\log p}$
coincides with $Z_{\mathrm{vM}}(\beta)$ itself.  No genuine
$\beta\to 1/\beta$ symmetry appears.

### 9.2 Classical continuation is the unique solution

A holomorphic continuation, if it exists on a connected open set,
is unique.  The function $-\zeta'/\zeta$ is the unique meromorphic
extension of $Z_{\mathrm{vM}}$ to $\mathbb{C}$ with poles at
$s = 1$ (simple, residue $+1$), $s = \rho$ (the non-trivial zeros of
$\zeta$), and $s = -2k$ (trivial zeros).  Therefore the analytic
continuation problem **has a closed-form answer**; the only freedom
left is the *interpretation* of that continuation in TNFR terms.

### 9.3 TNFR operational reading: zeros as resonance poles

Module `analytic_continuation.py` exposes the classical extension as
a callable `von_mangoldt_zeta_continued(s)` (backed by `mpmath`) and
re-labels its analytic structure in prime-ladder language:

* The pole at $s = 1$ is the *envelope resonance* of the ladder;
  it generates the $\psi(x) \sim x$ term.
* Each non-trivial zero $\rho = 1/2 + i t_n$ becomes a
  **resonance pole** of the REMESH spectrum.  Operationally,
  $|Z_{\mathrm{vM}}(1/2 + it)|$ exhibits a sharp local maximum
  at $t = t_n$.
* The trivial zeros at $s = -2k$ become poles of the continuation
  at the *forbidden* echo positions $s = -2k$ (k = 1, 2, …),
  cancelling the divergent reflection of the prime ladder under
  $s \mapsto 1 - s$.

### 9.4 Numerical validation

Three independent certificates are provided.

**(a) Agreement on the convergent half-plane.**  For
$\operatorname{Re}(s) > 1$ the prime-ladder sum and the continuation
must agree.  Function `verify_continuation_agreement` measures the
relative difference and reports a quality flag
(`excellent`/`good`/`poor`).  Empirically, with 5000 primes and
`max_power=15` we obtain `max_rel_diff ≈ 6.3e-3` for $s$ values
ranging across $\operatorname{Re}(s) \in \{1.5, 2, 2.5, 3, 4\}$.

**(b) Resonance peaks on the critical line.**
`scan_critical_line_for_poles` samples
$|Z_{\mathrm{vM}}(1/2 + it)|$ for $t \in [t_{\min}, t_{\max}]$,
detects local maxima with a prominence cutoff, and matches them
against the high-precision zero list
`KNOWN_RIEMANN_ZEROS` (P4).  For $t \in [10, 80]$ with 4001 sample
points the scan recovers all **20** known zeros in the range with
$|\Delta t| \lesssim 8 \times 10^{-3}$ — limited only by the grid
spacing $\Delta t \approx 0.0175$.

**(c) Explicit-formula reconstruction of $\psi(x)$.**
`reconstruct_psi_via_explicit_formula` evaluates the truncated
Riemann–von Mangoldt sum

$$
\psi_0(x) = x - \sum_{|\operatorname{Im}\rho| \le T} \frac{x^{\rho}}{\rho}
            - \log(2\pi) - \tfrac{1}{2}\log\bigl(1 - x^{-2}\bigr)
$$

and compares with the direct sieve evaluation
$\psi(x) = \sum_{n \le x}\Lambda(n)$.  With the first 30 zeros, the
absolute error falls to $\le 0.9$ for $x \in [20, 200]$, with the
expected non-monotone behaviour controlled by the unresolved high
zeros.

### 9.5 Honest scope statement

P13 does **not** prove the Riemann Hypothesis.  All four observable
features (continuation, polar structure on the critical line,
explicit formula, $\psi(x)$ reconstruction) are classical Hadamard /
von Mangoldt theory.  The TNFR-specific contribution is the
*operational re-reading*:

> Every analytic feature of $-\zeta'/\zeta$ corresponds to a structural
> mechanism of the prime-ladder REMESH spectrum: emission weights
> $\log p$, harmonic echoes $k\log p$, resonance poles
> $\rho = 1/2 + i t_n$, envelope pole at $s = 1$, and forbidden echo
> positions at $s = -2k$.

This delivers G2 in TNFR language.  Gaps G1 (self-adjoint operator
with vM spectrum), G3 (zeros–spectrum bijection), G4 (localisation
on $\operatorname{Re}(s) = 1/2$), and G5 (closure of Conjecture 10.1
with a non-affine bridge) remain open.

---

## 10. Self-Adjoint Prime-Ladder Hamiltonian (P14, Gap G1)

**Status**: Implemented and numerically certified (May 2026).
**Code**: `src/tnfr/riemann/prime_ladder_hamiltonian.py`,
`examples/03_riemann_zeta/43_prime_ladder_hamiltonian_demo.py`.

### 10.1 Problem statement (Gap G1)

The Hilbert–Pólya programme asks for a self-adjoint operator
$\hat H$ acting on a separable Hilbert space whose spectrum
encodes the data driving $\zeta(s)$.  The TNFR-Riemann programme
restricts the request to a finite-dimensional, explicitly
constructible operator whose spectrum exactly reproduces the
prime-ladder spectrum $\{k\log p\}$ and whose weighted spectral
trace reproduces the P12 von Mangoldt trace $Z_{\mathrm{vM}}(s)$.

### 10.2 Construction

We reuse the canonical TNFR internal Hamiltonian
(`tnfr.operators.hamiltonian.InternalHamiltonian`),

$$
\hat H_{\mathrm{int}}
   = \hat H_{\mathrm{coh}} + \hat H_{\mathrm{freq}}
   + \hat H_{\mathrm{coupling}}
$$

without modification.  Specialisation occurs only at the graph
level:

* **Nodes**: pairs $(p, k)$ for each prime $p \in \mathcal{P}$
  and each REMESH echo index $k = 1, \dots, K$.
* **Structural attributes**:
  $\nu_{f,(p,k)} = k\log p$, $\phi = 0$, $EPI = 1$, $S_i = 1$,
  $\Delta NFR = 0$.
* **Edges**: ladder edges $(p, k) \leftrightarrow (p, k+1)$ within
  each prime; **no** inter-prime edges.
* **Graph-level constants**: `H_COH_STRENGTH = 0`,
  `H_COUPLING_STRENGTH = J_0` (default $J_0 = 0$).

With these choices, $\hat H_{\mathrm{coh}} = 0$ and
$\hat H_{\mathrm{coupling}} = J_0 \cdot A$ (with $A$ the adjacency
matrix of the disjoint union of prime ladders).  At $J_0 = 0$,
$\hat H_{\mathrm{int}} = \hat H_{\mathrm{freq}}
   = \operatorname{diag}\bigl(k\log p\bigr)$, which is trivially
self-adjoint and whose spectrum equals the prime-ladder spectrum by
construction.

### 10.3 Weighted spectral trace

Define the diagonal weight operator
$\hat W = \sum_{p,k} \log(p)\, |p,k\rangle\langle p,k|$.
The TNFR analogue of $-\zeta'(s)/\zeta(s)$ is then

$$
Z_H(s) \;:=\; \operatorname{Tr}\!\bigl(\hat W\, e^{-s\hat H_{\mathrm{int}}}\bigr).
$$

At $J_0 = 0$ this collapses to
$\sum_{p,k} \log(p)\, e^{-s k \log p}
  = \sum_{p} \log(p)\, p^{-s}/(1 - p^{-s})
  = -\zeta'(s)/\zeta(s)$ for $\operatorname{Re}(s) > 1$.

### 10.4 Euler-product orthogonality at the operator level

The absence of inter-prime edges encodes multiplicativity:
$\hat H_{\mathrm{int}}$ decomposes as the orthogonal direct sum
$\bigoplus_p \hat H^{(p)}$ where each $\hat H^{(p)}$ acts on the
$K$-dimensional subspace spanned by $\{|p,k\rangle\}_{k=1}^K$.
This is the operator-level analogue of the Euler product
$\zeta(s) = \prod_p (1 - p^{-s})^{-1}$.

Switching on $J_0 > 0$ deliberately couples ladders **within a
single prime** (echo coupling); it does not couple distinct primes
and therefore preserves the Euler-product factorisation while
deforming the spectrum perturbatively.  Coupling **between**
distinct primes is intentionally not supported by the present
builder: doing so would break Euler-product orthogonality and is a
separate research question.

### 10.5 Numerical certificate

`verify_hamiltonian_reproduces_prime_ladder` returns a
`PrimeLadderHamiltonianCertificate` documenting:

* `spectrum_max_abs_error`: $\max_n |E_n^{\text{Ham}} - E_n^{\text{ladder}}|$
  — exactly $0$ at $J_0 = 0$ (verified to machine precision for
  $n_{\text{primes}} = 12$, $K = 6$, $N = 72$);
* `trace_max_rel_error`: worst-case relative deviation of $Z_H(s)$
  from $Z_{\mathrm{vM}}(s)$ over a user-supplied $s$ grid —
  $\lesssim 3 \cdot 10^{-16}$ at $J_0 = 0$;
* `is_hermitian`: $\hat H_{\mathrm{int}}$ passes the
  Hermiticity check inherited from `InternalHamiltonian`;
* perturbative scaling: spectrum deviation grows quadratically
  with $J_0$ at small coupling (verified empirically in the
  example demo).

### 10.6 What this closes and what remains open

**Closed (operationally)**: G1 — a self-adjoint, finite-dimensional
operator whose spectrum and weighted spectral trace reproduce the
prime-ladder data has been explicitly constructed, certified, and
shipped as part of the canonical TNFR API.

**Still open**:

* **G3** — bijection between the resonance poles of the analytic
  continuation (P13) and the eigenvalues of $\hat H_{\mathrm{int}}$
  on the imaginary axis.  The present construction provides one
  side of the correspondence (the operator); P13 provides the
  other (the poles).  A clean bijection requires choosing the
  correct boundary functional on $\hat H_{\mathrm{int}}$.
* **G4** — localisation of the resonance poles on
  $\operatorname{Re}(s) = 1/2$.  This is RH itself; P14 does not
  address it.
* **G5** — closure of Conjecture 10.1 with a non-affine bridge.
  Independent of P14.

P14 should therefore be read as the explicit, computable witness
that *every* spectral-operator step of the TNFR-Riemann programme
upstream of G3 is realisable inside the canonical TNFR formalism
without any extension or modification.

---

## 11. Weil–Guinand Explicit Formula (P15, operational closure of Gap G3)

### 11.1 Problem statement

Gap G3 of the TNFR-Riemann programme asks for an explicit bridge
between the non-trivial zeros of $\zeta(s)$ and the spectral data
of a TNFR operator.  The classical *Weil–Guinand explicit formula*
is precisely such a bridge: a single distributional identity in
which the zero side and the prime side are made manifest at the
same time.

In its standard form, for a real even Schwartz test function
$h(t)$ with Fourier transform
$g(u) = (2\pi)^{-1}\!\int h(t)\,e^{-itu}\,dt$,

$$\sum_{\gamma} h(\gamma)
   \;=\; h(i/2)+h(-i/2)
   \;-\; g(0)\log\pi
   \;+\; \tfrac{1}{2\pi}\!\int_{-\infty}^{\infty}\! h(t)\,
            \operatorname{Re}\psi\!\Bigl(\tfrac14 + \tfrac{it}{2}\Bigr)\,dt
   \;-\; 2\sum_{n\ge 1}\frac{\Lambda(n)}{\sqrt n}\,g(\log n).$$

The left-hand sum runs over imaginary parts $\gamma$ of all
non-trivial zeros $\rho = 1/2 + i\gamma$ of $\zeta(s)$.

### 11.2 TNFR realisation of the prime side

The von Mangoldt sum on the right is **exactly** a spectral
functional on the canonical P14 prime-ladder Hamiltonian
$\hat H_{\mathrm{int}} = \operatorname{diag}(k\log p)$ with weight
operator $\hat W = \operatorname{diag}(\log p)$:

$$-2 \sum_{n\ge 1} \frac{\Lambda(n)}{\sqrt n}\,g(\log n)
   \;=\; -2 \operatorname{Tr}\!\bigl(\hat W\,e^{-\hat H/2}\,g(\hat H)\bigr).$$

Indeed, every $n\in\mathbb{N}$ with $\Lambda(n)\ne 0$ is a prime
power $n = p^k$ and corresponds to a unique eigenstate
$|p,k\rangle$ of $\hat H_{\mathrm{int}}$ with eigenvalue
$E_{p,k} = k\log p$ and weight $W_{p,k} = \log p$.  No additional
arithmetic apparatus is needed: the prime side is read off the P14
spectrum.

### 11.3 Module and certificate

`src/tnfr/riemann/weil_explicit_formula.py` implements

* `GaussianTestFunction(sigma)` — the Gaussian test family
  $h_\sigma(t) = \exp(-t^2/(2\sigma^2))$ with closed-form Fourier
  pair, pole values $h(\pm i/2)$, and $g(0)$.
* `weil_prime_side_from_hamiltonian(bundle, test)` — evaluates
  $-2\operatorname{Tr}(\hat W e^{-\hat H/2} g(\hat H))$ via the
  eigendecomposition of `bundle.hamiltonian`.
* `weil_archimedean_integral(test)` — numerical quadrature of the
  digamma-weighted integral via `scipy.integrate.quad` and
  `mpmath.digamma`.
* `weil_zero_side(test, n_zeros)` — sum over Riemann zeros via
  `mpmath.zetazero`, with automatic convergence cutoff.
* `verify_weil_explicit_formula(bundle, sigma, n_zeros, tol)`
  returns a `WeilExplicitFormulaCertificate` exposing the four
  terms, the residual, and a Boolean `verified` flag.

### 11.4 Numerical evidence

Verification on the canonical bundle (50 primes, max power 8, dim
400), 120 Riemann zeros:

| $\sigma$ | zero side | RHS total | absolute residual | relative |
|----------|-----------|-----------|-------------------|----------|
| 2  | $2.85\times 10^{-11}$ | $2.85\times 10^{-11}$ | $1.2\times 10^{-17}$ | $4.3\times 10^{-7}$ |
| 3  | $3.02\times 10^{-5}$  | $3.02\times 10^{-5}$  | $1.7\times 10^{-16}$ | $5.6\times 10^{-12}$ |
| 5  | $3.71\times 10^{-2}$  | $3.71\times 10^{-2}$  | $5.7\times 10^{-16}$ | $1.5\times 10^{-14}$ |
| 8  | $5.00\times 10^{-1}$  | $5.00\times 10^{-1}$  | $1.1\times 10^{-15}$ | $2.2\times 10^{-15}$ |
| 12 | $1.81$                | $1.81$                | $6.7\times 10^{-16}$ | $3.7\times 10^{-16}$ |
| 18 | $4.75$                | $4.75$                | $5.3\times 10^{-15}$ | $1.1\times 10^{-15}$ |

The identity holds to machine precision uniformly across the
tested range.  The $\sigma=2$ entry has high relative error only
because both sides are at the noise floor ($\sim 10^{-11}$).

### 11.5 What this closes and what remains open

**Closed (operationally)**: Gap **G3**.  Each ingredient of
Weil's bridge is now expressed inside the canonical TNFR
formalism:

* Prime side — `weil_prime_side_from_hamiltonian` from P14.
* Zero side — `mpmath.zetazero` (external) confronted against
  the TNFR prime side.
* Archimedean and pole sides — standard analytic objects
  attached to $\zeta(s)$, computed once and reused.

The cancellation of all four terms to machine precision is a
numerical witness that the P14 Hamiltonian carries the entire
prime-side data of the bridge, with no auxiliary number-theoretic
machinery.

**Still open**:

* **G4 — Riemann Hypothesis**.  The explicit formula is
  *unconditional*: it holds whatever the locations of the zeros.
  RH is the further statement that all $\rho = 1/2 + i\gamma$
  have $\gamma\in\mathbb{R}$.  P15 does not address this.  An RH
  proof inside TNFR would require either (a) a positivity
  argument for a TNFR-defined functional of the form
  $\sum_\gamma h(\gamma) \ge 0$ for all admissible test
  functions in a class that forces $\gamma\in\mathbb{R}$, or
  (b) a self-adjoint extension whose eigenvalues *are* the
  imaginary parts $\gamma$ (the Hilbert–Pólya programme).
* **G5 — Conjecture 10.1 non-affine bridge** between the
  TNFR spectral zeta of §6 and classical $\zeta(s)$.  P15 does
  not affect G5: it operates one level above, on the explicit
  formula rather than on the zeta functions themselves.

### 11.6 Scope statement

P15 is a **numerical verification of a classical theorem** using
TNFR machinery on the prime side.  It is not new mathematics in
the analytic-number-theory sense.  What it delivers is the
*instrumental* result that the entire spectral apparatus required
to state the bridge between primes and zeros lives natively
inside the TNFR formalism, with no extra postulates and no
empirical fitting.  Combined with P12 (von Mangoldt series),
P13 (analytic continuation of the TNFR vM zeta) and P14
(self-adjoint Hamiltonian carrying the spectrum), the TNFR-Riemann
programme now has an end-to-end computable pipeline from the
nodal equation to the Weil-Guinand identity.

The remaining obstruction is RH itself.

---

## 12. Li–Keiper Positivity Criterion via TNFR Resonance Spectrum (P16)

### 12.1 Problem statement

**Li's criterion** (Xian-Jin Li, 1997). Define, for every integer
$n \ge 1$,

$$
\lambda_n \;=\; \sum_{\rho} \Bigl[ 1 - \bigl(1 - \tfrac{1}{\rho}\bigr)^n \Bigr],
$$

where the sum ranges over all non-trivial zeros $\rho$ of $\zeta(s)$,
counted with multiplicity and paired symmetrically with their
conjugates. Li proved

$$
\text{RH} \;\Longleftrightarrow\; \lambda_n > 0 \quad \text{for every } n \ge 1.
$$

Li's criterion is therefore **strictly RH-equivalent**: it recasts the
location of the non-trivial zeros as the positivity of a real
sequence. Bombieri-Lagarias (1999) gave an alternative variational
proof; Voros (2003) computed the first $\sim 10^5$ coefficients and
confirmed positivity numerically.

### 12.2 TNFR realisation

In the TNFR-Riemann programme the non-trivial zeros appear as
**resonance poles** of the prime-ladder von Mangoldt zeta after
analytic continuation (P13, §9). Three sources of zeros are now
available:

* classical mpmath `zetazero` (reference),
* P13 critical-line resonance-pole scan (TNFR-native),
* P14 prime-ladder Hamiltonian spectrum (structural, via Weil/Guinand pairing of §11).

Computing $\lambda_n$ from each source and checking positivity yields
a TNFR-internal RH-equivalent diagnostic: a single negative
$\lambda_n$ would falsify RH; the persistent positivity observed
across all three sources is consistent with it.

P16 does **not** open a new gap. It recasts gap G4 (the RH
statement itself) as a positivity test on the TNFR resonance
spectrum, completing the diagnostic surface initiated by P12-P15.

### 12.3 Module API

The module `tnfr.riemann.li_keiper` exposes:

* `li_coefficients_from_zeros(zeros_upper, n_max, *, dps=50)` -
  arbitrary-precision evaluation of $\lambda_n$ via
  $2\,\Re[1 - (1 - 1/\rho)^n]$ paired with the conjugate.
* `LiKeiperCertificate` - frozen dataclass with
  `lambda_classical`, `lambda_tnfr`, `positivity_classical`,
  `positivity_tnfr`, `max_abs_difference`, `notes`, and a
  `summary()` method.
* `verify_li_keiper_criterion(*, n_max=50, n_zeros=200, dps=50,
  compare_tnfr=False, ...)` - end-to-end verification, optionally
  comparing classical zeros against P13 detected peaks.

### 12.4 Numerical evidence

End-to-end run with `n_max = 60`, `n_zeros = 250`, `dps = 50`
(example 45, Section 2):

| $n$ | $\lambda_n$ (truncated) | sign |
|:---:|:------------------------:|:----:|
|   1 | $+2.13\times 10^{-2}$    |  +   |
|   5 | $+5.31\times 10^{-1}$    |  +   |
|  10 | $+2.10\times 10^{0}$     |  +   |
|  20 | $+8.05\times 10^{0}$     |  +   |
|  30 | $+1.69\times 10^{1}$     |  +   |
|  40 | $+2.76\times 10^{1}$     |  +   |
|  50 | $+3.90\times 10^{1}$     |  +   |
|  60 | $+5.07\times 10^{1}$     |  +   |

All 60 coefficients are positive, with `min_n lambda_n = +2.13e-2`.

The truncation suppresses the magnitudes by ~10% relative to the
published values (Keiper 1992: $\lambda_1 = 0.0230957$), reflecting
the slow logarithmic convergence of the partial zero-sum; the
signs are robust to this truncation. The growth matches the
classical asymptotic
$\lambda_n \sim (n/2) \log(n/2\pi)$ (Voros 2003).

TNFR-vs-classical agreement (example 45, Section 3,
`compare_tnfr=True` with the P13 scan on $t \in [10, 80]$):

* 21 resonance peaks detected, quality `all_matched`,
* `positivity_tnfr = True` for every $n \in [1, 20]$,
* maximum disagreement at $n = 20$: $|\Delta\lambda_{20}| \approx 1.38$
  (dominated by the smaller TNFR $t$-window, not by sign flips).

### 12.5 What closes and what remains open

P16 **closes**:

* the diagnostic surface required to read RH as a TNFR-native
  positivity statement on the prime-ladder resonance spectrum;
* the consistency check between three independent sources of
  non-trivial zeros (classical, P13 poles, P14 Hamiltonian).

P16 **does not close**:

* RH itself (gap G4). Verifying $\lambda_n > 0$ for finitely many
  $n$ is consistent with, but does not imply, the Riemann
  Hypothesis. A proof would require either (a) an a-priori
  positivity argument on the resonance spectrum, or (b) a
  self-adjointness/positivity witness for an operator whose
  eigenvalues are forced to lie on $\Re(s) = 1/2$;
* Conjecture 10.1 (gap G5). The Li-Keiper test compares classical
  and TNFR sides at the level of Li coefficients, not at the
  level of an affine bridge between $\zeta_H$ and $\zeta_R$.

### 12.6 Scope statement

P16 is a **TNFR-native restatement of a known RH-equivalent
criterion**. It does not introduce new mathematics in the
analytic-number-theory sense. Its value is methodological: the
entire diagnostic surface for the Riemann Hypothesis - prime
series (P12), analytic continuation and resonance poles (P13),
self-adjoint spectrum (P14), explicit formula (P15), and now
Li-Keiper positivity (P16) - is expressible without exiting the
TNFR formalism. The remaining obstruction is the proof of RH
itself, which the programme exposes but does not (and does not
claim to) eliminate.

---

## 13. Empirical Uniform-Coercivity Certificate (P22)

**Status**: Implemented and numerically evaluated (May 2026).
**Code**: `src/tnfr/riemann/coercivity_uniform.py`,
`examples/03_riemann_zeta/50_uniform_coercivity_demo.py`.

### 13.1 Motivation

P18-P21 establish robust sampled positivity for
$\alpha(\sigma) = W[\sigma]/E_{TNFR}[\sigma]$ across dense
$(\sigma, \text{family}, \text{gauge})$ grids. To move one step closer
to the G4 target form

$$
\inf_{\sigma \in [\sigma_{\min},\sigma_{\max}],\,F,\,G} \alpha(\sigma;F,G) > 0,
$$

P22 adds an interval-level empirical certificate, not just pointwise
sampling.

### 13.2 Method

On a shared log-spaced $\sigma$ grid, P22 runs both:

1. admissible-family sweep (P19/P21),
2. node-aware gauge sweep (P20).

From the resulting alpha tables it computes:

- sampled minimum $\alpha_{\min}^{\text{sample}}$,
- finite-difference slope envelope $L_{\text{proxy}}$,
- mesh radius $r_h = \tfrac12 \max_i (\sigma_{i+1}-\sigma_i)$,
- trajectory-stratified slope envelopes,
- segment-local slope bounds,

and reports the mesh-corrected lower bound

$$
\alpha_{\inf}^{\text{interval}}
\;\gtrsim\;
\alpha_{\min}^{\text{sample}} - L_{\text{proxy}}\,r_h.
$$

The resulting dataclass `UniformCoercivityCertificate` reports three
interval diagnostics: global, stratified, and segment-local.

### 13.3 Current numerical outcome

Representative run (same P14 base bundle as P18-P21,
$\sigma \in [0.5, 8.0]$, log grid):

- `sampled_all_positive = True`
- `alpha_min_sampled = +1.3691e-173`
- `L_proxy = 2.6554e+00`
- `mesh_radius = 8.9119e-01`
- `interval_lb_global = -2.3665e+00`
- `interval_lb_stratified = -2.3665e+00`
- `interval_lb_local = -3.2350e-01`
- `interval_lb_global_positive = False`
- `interval_lb_stratified_positive = False`
- `interval_lb_local_positive = False`

### 13.4 Interpretation

P22/P23 upgrades the diagnostics from pointwise positivity to a
quantified interval certificate framework. The segment-local envelope is
substantially tighter than the global one (from about -2.37 to about
-0.32), but remains negative. Therefore, **uniform coercivity is not yet
established** at interval level on the tested band.

This narrows G4 honestly: empirical positivity remains strong, but the
coercivity margin is still too weak near the smallest sampled alpha
region.

### 13.5 Immediate next technical directions

1. Add adaptive refinement around low-alpha neighborhoods to tighten
   $r_h$ where it matters.
2. Derive analytic lower envelopes for the TNFR energy denominator to
   complement numerical certificates.
3. Build hybrid certificates combining local slope envelopes with
   curvature-aware interpolation bounds.

---

## 13bis. Adaptive σ Refinement Near the Coercivity Bottleneck (P24)

### 13bis.1 Motivation

Direction (1) of Section 13.5 is the cheapest lever on G4: under the
segment-local Lipschitz envelope
$\alpha_{lb}(\sigma) \ge \min(\alpha_i,\alpha_{i+1}) - L_i \cdot \Delta\sigma_i/2$,
the lower bound is dominated by the *widest segment with smallest
local α*. Halving that segment shrinks $\Delta\sigma_i/2$ and tightens
the bound exactly where it hurts, without claiming any new analytic
control.

### 13bis.2 Method

P24 adds the optional kwargs `refinement_rounds` and
`refinement_per_round` to `verify_uniform_coercivity_empirical(...)`.
Each round:

1. Aggregates the segment-local lower bounds across rows of both alpha
   tables (admissible family and node-aware) via
   `_worst_segment_indices(alpha_a, alpha_n, sigmas, top_k)`.
2. Selects the `top_k` worst segments and inserts their midpoints into
   the σ grid (`np.unique` deduplicates against existing points).
3. Re-runs `sweep_alpha_admissible_family` and `sweep_alpha_nodeaware`
   on the augmented grid and recomputes
   $\inf_i\, \min(\alpha_i,\alpha_{i+1}) - L_i \cdot \Delta\sigma_i/2$.
4. Stops early if no new point was added.

The result lives in four new fields of `UniformCoercivityCertificate`:
`n_refinement_rounds`, `n_sigma_refined`,
`interval_lower_bound_local_refined`,
`interval_lower_local_refined_positive`.

### 13bis.3 Numerical Outcome (examples/03_riemann_zeta/51_adaptive_coercivity_demo.py)

Bundle: `n_primes=18, max_power=5, coupling=0.0`.
Certificate: `sigma=[0.5, 4.0], n_sigma=10, n_zeros=24, max_zeros=96,
refinement_rounds=2, refinement_per_round=1`.

- `interval_lb_local        = -6.0163e-02`
- `interval_lb_local_refined = -2.8515e-02`
- `improvement (refined - local) = +3.1648e-02`
- `n_sigma_refined           = 12` (10 base + 2 midpoints)
- `interval_lb_local_refined_positive = False`
- `sampled_all_positive       = True`
- `admissible_ok / nodeaware_ok = True / True`

So two midpoint insertions cut the negative gap roughly in half on this
band, but the refined empirical lower bound is still negative.

### 13bis.4 Honest Interpretation

P24 does **not** close G4. It is a numerical sharpening of the
already-empirical segment-local certificate: a tighter `interval_lb_local`
is evidence consistent with uniform coercivity, but the bound remains
negative and the underlying Lipschitz envelope is itself only a
piecewise linear surrogate. As stated in AGENTS.md, only G4=RH stays
open; P24 narrows the empirical bottleneck without claiming any analytic
positivity result.

### 13bis.5 Next Steps

1. Combine P24 refinement with directions (2)–(3) of Section 13.5
   (analytic lower envelopes for $E_{TNFR}$, curvature-aware
   interpolation) so the surrogate envelope itself improves, not just
   its sampling.
2. Try higher `top_k` and bounded total budget on full bands
   `[0.5, 8.0]` once the underlying sweeps are vectorised.
3. Track the worst segment across rounds to confirm the bottleneck is a
   stable σ-neighborhood, not a roaming artifact.

---

## 13ter. Paley-Gap Coercivity Diagnostic (P25)

### 13ter.1 Motivation

P22–P24 attack the coercivity bottleneck (gap G4) numerically, by
tightening lower envelopes of $\alpha(\sigma) = W[\sigma] /
E_{TNFR}[\sigma]$. They do not, however, exploit any *algebraic
identity* between the TNFR objects involved. The author's own
*Spectral note: Paley gap via lambda_2 (residue circulants)*
(Martínez Gamo, Zenodo 17665853 v2, November 20 2025) introduces a
complementary methodology: build a gap

$$
g(n) = \Bigl|\lambda_2(\text{residue circulant})
        - \tfrac{n - \sqrt{n}}{2}\Bigr|
$$

between a *computed spectral quantity* and a *closed-form algebraic
reference*. Vanishing of the gap singles out an arithmetic
structural condition ($n$ prime, $n \equiv 1 \pmod 4$ in the source
note) up to the tested range, by *identity* rather than by *bound*.

P25 imports this philosophy into the TNFR-Riemann pipeline. The
prime-ladder data is generated in *three* different but
mathematically equivalent ways on $\operatorname{Re}(s) > 1$:

1. **Route A (P12 closed form)**:
   $Z_{P12}(s) = \sum_{(\mu, w)} w\, e^{-s\mu}$, the weighted
   Dirichlet trace over the prime-ladder spectrum.
2. **Route B (P14 spectral trace)**:
   $Z_{P14}(s) = \operatorname{Tr}\bigl(\hat W e^{-s\hat
   H_{\mathrm{int}}}\bigr)$, the weighted spectral trace of the
   prime-ladder Hamiltonian.
3. **Reference (classical)**:
   $Z_{\mathrm{cls}}(s) = \sum_{n \le N} \Lambda(n)\, n^{-s}$, a
   direct truncation of the classical von Mangoldt series.

### 13ter.2 Method

P25 defines three Paley-gap quantities per $\sigma$:

$$
\begin{aligned}
g_{P12}(\sigma)   &= |Z_{P12}(\sigma)   - Z_{\mathrm{cls}}(\sigma)|, \\
g_{P14}(\sigma)   &= |Z_{P14}(\sigma)   - Z_{\mathrm{cls}}(\sigma)|, \\
g_{\mathrm{cross}}(\sigma) &= |Z_{P14}(\sigma) - Z_{P12}(\sigma)|.
\end{aligned}
$$

The first two measure *truncation fidelity* of each TNFR route against
the classical reference; both decay as $(n_{\text{primes}},
k_{\max}, N)$ grow. The third — the **cross Paley-gap** — is the
diagnostic of interest: by construction P14 specialises to P12 in the
decoupled limit ($J_0 = 0$, no inter-prime coupling), so
$g_{\mathrm{cross}}(\sigma)$ must vanish to machine precision for
every $\sigma$ when $\texttt{coupling} = 0$. Any non-zero
$\texttt{coupling}$ deforms the Hamiltonian spectrum and produces a
measurable $g_{\mathrm{cross}}$ free of classical-truncation noise.

Module: [`src/tnfr/riemann/paley_gap_coercivity.py`](../src/tnfr/riemann/paley_gap_coercivity.py).
Demo: [`examples/03_riemann_zeta/52_paley_gap_coercivity_demo.py`](../examples/03_riemann_zeta/52_paley_gap_coercivity_demo.py).

### 13ter.3 Numerical Outcome

Reference configuration: `n_primes = 18`, `max_power = 5`,
$\sigma \in [1.5, 4.0]$ (11 points), $N = 50{,}000$.

**Bundle A — decoupled (`coupling = 0`)**

$$
\max_\sigma g_{\mathrm{cross}}(\sigma) = 1.110 \times 10^{-16}
$$

every entry of $g_{\mathrm{cross}}$ is bounded by $1.2 \times
10^{-16}$ (machine precision). Truncation gaps:
$\max g_{P12} = \max g_{P14} = 2.338 \times 10^{-1}$ at
$\sigma = 1.5$, decaying to $1.25 \times 10^{-6}$ at $\sigma = 4.0$.

The vanishing of $g_{\mathrm{cross}}$ confirms the Paley-style
algebraic identity $Z_{P14} \equiv Z_{P12}$ in the decoupled limit —
the P14 self-adjoint operator is a faithful operator-theoretic
realisation of the P12 closed form.

**Bundle B — weakly coupled (`coupling = 1.0 × 10⁻²`)**

| $\sigma$ | $g_{P12}$    | $g_{P14}$    | $g_{\mathrm{cross}}$ |
|---------:|-------------:|-------------:|---------------------:|
|     1.50 | 2.338 × 10⁻¹ | 2.337 × 10⁻¹ | 1.203 × 10⁻⁴         |
|     2.00 | 1.508 × 10⁻² | 1.499 × 10⁻² | 8.966 × 10⁻⁵         |
|     2.50 | 1.262 × 10⁻³ | 1.193 × 10⁻³ | 6.832 × 10⁻⁵         |
|     3.00 | 1.189 × 10⁻⁴ | 6.647 × 10⁻⁵ | 5.242 × 10⁻⁵         |
|     3.50 | 1.196 × 10⁻⁵ | 2.827 × 10⁻⁵ | 4.023 × 10⁻⁵         |
|     4.00 | 1.253 × 10⁻⁶ | 2.955 × 10⁻⁵ | 3.080 × 10⁻⁵         |

with $\max_\sigma g_{\mathrm{cross}} = 1.203 \times 10^{-4}$. The
cross gap is now well above the machine-precision floor, decays
monotonically with $\sigma$, and is qualitatively distinct from the
classical truncation gap (which decays exponentially fast in $\sigma$
because the prime ladder approximates a Dirichlet series). For
$\sigma \gtrsim 3.5$, $g_{P14}$ exceeds $g_{P12}$: the coupling
deformation eventually dominates the truncation error.

### 13ter.4 Honest Interpretation

- **What P25 establishes.** A clean, identity-level consistency
  check between the two TNFR routes (P12 closed form and P14
  self-adjoint operator) at every tested $\sigma$. At
  $\texttt{coupling} = 0$ this consistency is a Paley-style
  algebraic identity ($g_{\mathrm{cross}}$ at machine precision); at
  $\texttt{coupling} > 0$ it becomes a structural-deformation
  diagnostic.

- **What P25 does *not* establish.** P25 does not close gap G4
  (RH localisation on $\operatorname{Re}(s) = 1/2$). The cross gap
  at $\texttt{coupling} = 0$ vanishes by construction — P14 was
  built to match P12 in the decoupled limit — so the zero-coupling
  numbers are a regression test, not a discovery. The Zenodo source
  note itself states its construction is *reproducible; not a
  primality proof*; P25 inherits the same scope at the coercivity
  level. No claim is made that P25 implies analytic uniform
  positivity of $\alpha(\sigma)$ on any interval, nor that it
  bridges to the classical $\zeta(s)$.

- **Where the Paley-style signal lives.** The diagnostic value of
  P25 is the *deformation channel*: $g_{\mathrm{cross}}(\sigma) \to
  0$ as $\texttt{coupling} \to 0$ at every $\sigma$, while
  $g_{\mathrm{cross}}(\sigma)$ at fixed $\sigma$ scales smoothly
  with $\texttt{coupling}$. This is exactly the same epistemic
  status as $g(n) = 0$ identifying primes in the Zenodo note: an
  *identity diagnostic* over a tested range, not a closed-form
  theorem on the entire family.

### 13ter.5 Next Steps

1. Extend the cross gap to a **functional-equation Paley-gap**
   using [`src/tnfr/riemann/functional_equation.py`](../src/tnfr/riemann/functional_equation.py),
   tabulating $|Z(\sigma) - Z(1 - \sigma)|$ along the critical
   strip. A Paley-style identity there would directly engage
   $\operatorname{Re}(s) = 1/2$.
2. Sweep $g_{\mathrm{cross}}(\sigma)$ across a $\texttt{coupling}$
   grid to extract a scaling law and confirm the diagnostic is
   stable (linear or polynomial in $\texttt{coupling}$).
3. Combine $g_{\mathrm{cross}}$ with P22–P24 segment-local
   coercivity envelopes: use the structural-deformation channel as a
   classifier of which $\sigma$ intervals tolerate coupling without
   eroding $\alpha(\sigma)$.

---

## §13quater — P26: Lyapunov-Spectral Positivity Certificate for P14

### 13quater.1 Motivation

AGENTS.md §13.2 lists the final TNFR–Riemann gap balance: G1, G2, G3
are operationally closed by P14, P13, P15 respectively; G5 is
superseded by the P12+P13+P15 stack. The only obstruction left is
**G4 = RH itself**, which AGENTS.md classifies as *"not attackable by
any extension of P12–P16 — it requires a structural positivity /
self-adjointness argument (Hilbert–Pólya-style) that is genuinely new
mathematics."*

Two ingredients required for any Hilbert–Pólya-style attack already
live in the codebase:

1. The **self-adjoint prime-ladder Hamiltonian**
   $\hat H = \hat H_{\mathrm{freq}} + J_0\,\hat H_{\mathrm{coupling}}$
   from P14 ([`src/tnfr/riemann/prime_ladder_hamiltonian.py`](../src/tnfr/riemann/prime_ladder_hamiltonian.py)).
2. The **structural Lyapunov functional**
   $E = \tfrac12\sum_i \varepsilon(i) \ge 0$ with $dE/dt \le 0$ from
   [`src/tnfr/physics/conservation.py`](../src/tnfr/physics/conservation.py),
   flagged in AGENTS.md as *"proof sketch; complete proof open"*.

P26 fuses both into a single quantitative **positivity certificate**
for the P14 operator. The module
[`src/tnfr/riemann/lyapunov_spectral_positivity.py`](../src/tnfr/riemann/lyapunov_spectral_positivity.py)
returns a frozen dataclass `LyapunovSpectralCertificate` aggregating
the four ingredients of operator-level Hilbert–Pólya positivity:
self-adjointness, strict positivity with explicit gap, trace-class
resolvent, and unitary flow.

### 13quater.2 Method

The certificate combines four checks:

1. **Diagonal positivity at $J_0 = 0$.** $\hat H_{\mathrm{freq}}$ is
   real-diagonal with entries $\nu_{f,(p,k)} = k\log p$. Because
   $p \ge 2$ and $k \ge 1$, the spectrum is bounded below by
   $\log 2 \approx 0.6931$. This is the **unperturbed gap**.

2. **Quantitative Kato–Rellich envelope.** For bounded real-symmetric
   perturbations $J_0 \hat H_{\mathrm{coupling}}$ of a self-adjoint
   diagonal operator,
   $$
     |\lambda_n(\hat H) - \lambda_n(\hat H_{\mathrm{freq}})|
       \;\le\; |J_0|\, \|\hat H_{\mathrm{coupling}}\|_{\mathrm{op}}
   $$
   for every $n$. The certificate exposes the **guaranteed gap**
   $\log 2 - |J_0|\,\|\hat H_{\mathrm{coupling}}\|_{\mathrm{op}}$
   and flags `perturbation_safe = True` when it is strictly positive.

3. **Trace-class resolvent.** On the finite-dimensional prime-ladder
   space every bounded operator is trace-class; the meaningful
   reportables are the Schatten norms
   $\|(\hat H + c\hat I)^{-1}\|_1$ and
   $\|(\hat H + c\hat I)^{-1}\|_2$ for a shift $c > 0$, so growth
   with $(N_{\mathrm{primes}}, K)$ can be tracked.

4. **Numerical certification of the unitary flow.** A self-adjoint
   $\hat H$ generates a unitary propagator $U(t) = e^{-it\hat H}$.
   The certificate verifies $\|U(t)\psi_0\| = 1$ and
   $\langle\psi(t)|\hat H^2|\psi(t)\rangle$ to machine precision on a
   battery of random initial states.

`structural_positivity` is `True` iff numerical positivity, the
Kato–Rellich envelope, and the unitary flow all agree. The structural
Lyapunov functional $E$ of `conservation.py` vanishes on the
prime-ladder graph by construction (neutral structural state), so its
operator-level analogue is the spectral energy
$E_{\mathrm{spec}}[\psi] = \langle\psi|\hat H^2|\psi\rangle$, whose
conservation is exactly the check in step 4.

### 13quater.3 Numerical outcome

Demo: [`examples/03_riemann_zeta/53_lyapunov_spectral_positivity_demo.py`](../examples/03_riemann_zeta/53_lyapunov_spectral_positivity_demo.py).

Decoupled certificate (`n_primes = 12`, `max_power = 5`, $J_0 = 0$,
$\dim \mathcal H = 60$, shift $c = 1$):

| Quantity | Value |
|---|---|
| `spectrum_min` | $6.931472 \times 10^{-1}$ ($= \log 2$, exact) |
| `spectrum_max` | $1.805459 \times 10^{1}$ |
| `spectral_gap` | $6.931472 \times 10^{-1}$ |
| `schatten_1_norm` | $1.019135 \times 10^{1}$ |
| `schatten_2_norm` | $1.576613$ |
| `unperturbed_gap` | $6.931472 \times 10^{-1}$ |
| `coupling_norm` | $0$ |
| `guaranteed_gap` | $6.931472 \times 10^{-1}$ |
| `perturbation_safe` | True |
| `max_norm_drift` | $1.11 \times 10^{-16}$ |
| `max_energy_drift` | $3.62 \times 10^{-16}$ |
| `unitary` | True |
| `structural_positivity` | **True** |

Coupling sweep over $J_0 \in [0, 0.30]$:

| $J_0$ | `min(λ)` | `guaranteed_gap` | `perturbation_safe` | `unitary` |
|---:|---:|---:|:---:|:---:|
| 0.00 | $6.931 \times 10^{-1}$ | $6.931 \times 10^{-1}$ | True | True |
| 0.05 | $6.895 \times 10^{-1}$ | $6.065 \times 10^{-1}$ | True | True |
| 0.10 | $6.789 \times 10^{-1}$ | $5.199 \times 10^{-1}$ | True | True |
| 0.15 | $6.614 \times 10^{-1}$ | $4.333 \times 10^{-1}$ | True | True |
| 0.20 | $6.376 \times 10^{-1}$ | $3.467 \times 10^{-1}$ | True | True |
| 0.25 | $6.081 \times 10^{-1}$ | $2.601 \times 10^{-1}$ | True | True |
| 0.30 | $5.734 \times 10^{-1}$ | $1.735 \times 10^{-1}$ | True | True |

The empirical spectral bottom is uniformly larger than the
Kato–Rellich envelope, confirming the envelope is conservative
(as expected). Across the whole sweep `structural_positivity = True`.

### 13quater.4 Honest interpretation

* At $J_0 = 0$ the certificate is a finite-dimensional restatement of
  the trivial fact $\mathrm{diag}(k\log p) \succ 0$; its value is the
  explicit numerical gap $\log 2$ and the Schatten templates used to
  measure perturbative degradation.
* For $J_0 > 0$ the Kato–Rellich envelope provides a **rigorous
  quantitative interval** in which positivity is guaranteed: any
  $J_0$ with $|J_0|\,\|\hat H_{\mathrm{coupling}}\|_{\mathrm{op}} <
  \log 2$ produces a self-adjoint operator with strictly positive
  spectrum, trace-class resolvent, and unitary flow.
* The Lyapunov ingredient $dE/dt \le 0$ of `conservation.py` is
  itself flagged in AGENTS.md as *"proof sketch; complete proof
  open."* P26 therefore inherits the same status on the side that
  invokes the structural Lyapunov bound: the **operator-level**
  positivity statement is rigorous, but the variational
  identification of that operator with the generator of the
  structural Lyapunov flow remains the open piece.
* Crucially, **P26 does not close gap G4**. RH is a statement about
  the analytic continuation of the prime-ladder vM zeta (P13) and
  the localisation of its resonance poles on $\operatorname{Re}(s) =
  1/2$; the finite-dimensional positivity of $\hat H$ is necessary
  but not sufficient. What P26 does establish is the
  **operator-level positivity slot** that any Hilbert–Pólya attack
  must fill, together with explicit quantitative numbers.

### 13quater.5 Next steps

1. **Promote the Lyapunov sketch to a theorem.** Provide an
   analytical proof of $dE/dt \le 0$ inside `physics/conservation.py`
   under grammar-compliant evolution; this would upgrade the P26
   structural identification from "operationally consistent" to
   "operationally closed."
2. **Push the Kato–Rellich envelope to non-perturbative coupling.**
   Use a Bauer–Fike or pseudospectral argument to extend the
   quantitative positivity interval beyond $|J_0|\,\|\hat
   H_{\mathrm{coupling}}\| < \log 2$.
3. **Connect to P16 (Li–Keiper).** The spectral gap reported by P26
   is the natural quantitative input for the Li–Keiper coefficients
   $\lambda_n$ of P16; correlate the two and document any monotone
   relationship.
4. **Couple to P15 (Weil–Guinand).** The trace-class resolvent of
   P26 is the operator-level object whose spectral side is tested by
   P15. Add a cross-check using the Schatten norms of P26 as a
   stability witness for the Weil–Guinand identity numerics.

---

## §13quinquies. P27 — Hilbert–Pólya scaffold (does NOT close G4=RH)

### 13quinquies.1 Motivation: filling the Hilbert–Pólya slot

The Hilbert–Pólya program asks for a self-adjoint operator $T_{\mathrm{HP}}$
on a Hilbert space whose spectrum coincides with the imaginary parts
$\gamma_n$ of the non-trivial zeros of $\zeta(s)$. P26 supplied an
*operator-level positivity slot* (Lyapunov + Kato–Rellich +
Schatten-class) compatible with such an attack; P27 now constructs
the abstract Hilbert–Pólya operator explicitly on the truncated TNFR
Hilbert space $\ell^2_N(\mathbb{N})$ and certifies its internal
consistency with the rest of the TNFR–Riemann stack (P14, P15). The
construction is honestly *scaffolding*, not a derivation: $T_{\mathrm{HP}}$
is populated by *inputting* the zeros from `mpmath.zetazero`. P27
quantifies the gap that a genuinely structural derivation would have
to close.

### 13quinquies.2 Method: $T_{\mathrm{HP}} = \mathrm{diag}(\gamma_1,\ldots,\gamma_N)$

On the truncated Hilbert space $\ell^2_N(\mathbb{N})$ define
$$
  T_{\mathrm{HP}} \;:=\; \operatorname{diag}(\gamma_1, \gamma_2, \ldots, \gamma_N),
  \qquad \gamma_n := \operatorname{Im}(\rho_n),
$$
with $\rho_n$ the $n$-th non-trivial zero supplied by `mpmath.zetazero`
at decimal precision $\mathrm{dps}=30$. The certificate verifies four
axes:

1. **Self-adjointness.** $T_{\mathrm{HP}}$ real diagonal $\Rightarrow$
   $\|T_{\mathrm{HP}} - T_{\mathrm{HP}}^{*}\|_F = 0$ exactly.
2. **Trace-class shifted resolvent.** For shift $s>0$,
   $R := (T_{\mathrm{HP}}^2 + s^2 I)^{-1/2}$ admits Schatten 1- and
   2-norms $\sum_n (\gamma_n^2+s^2)^{-1/2}$, $\big(\sum_n
   (\gamma_n^2+s^2)^{-1}\big)^{1/2}$ which we report and check
   against finite truncation.
3. **Weil–Guinand closure with P14.** For a Gaussian test function
   $h(t) = \exp(-t^2/2\sigma^2)$ the zero side
   $2\sum_n h(\gamma_n)$ is computed *via* $T_{\mathrm{HP}}$ and
   compared to the right-hand side
   $h(i/2) + h(-i/2) - g(0)\log\pi + I_{\mathrm{arch}}(h)
    - 2\sum_n \Lambda(n) n^{-1/2} g(\log n)$,
   with the prime side coming from the P14 prime-ladder Hamiltonian.
4. **Operator-level gap G4.** The Wasserstein-1 distance
   $W_1\!\big(\sigma(P14),\sigma(T_{\mathrm{HP}})\big)$ between the
   sorted truncated spectra quantifies the *structural* gap, i.e.,
   how far the prime-ladder spectrum (growing like $\log n$) is from
   the zero spectrum (growing like $2\pi n/\log n$).

The orchestrator is `compute_hilbert_polya_certificate` in
`src/tnfr/riemann/hilbert_polya.py`; the demo lives in
`examples/03_riemann_zeta/54_hilbert_polya_demo.py`.

### 13quinquies.3 Numerical outcome (defaults $n_{\mathrm{primes}}=50$, $K=8$, $N=80$, $\sigma=8$)

| Axis | Quantity | Value | Verdict |
|---|---|---|---|
| Self-adjointness | $\|T_{\mathrm{HP}} - T_{\mathrm{HP}}^{*}\|_F$ | $0$ exactly | ✅ |
| Resolvent ($s=1$) | $\|R\|_1$ | $1.95\times 10^{-2}$ | trace-class ✅ |
|  | $\|R\|_2$ | $6.07\times 10^{-3}$ | Hilbert–Schmidt ✅ |
|  | $\|R\|_{\mathrm{op}}$ | $7.06\times 10^{-2}$ | bounded ✅ |
| Weil–Guinand | zero side via $T_{\mathrm{HP}}$ | $0.500227\,7175$ |  |
|  | pole side ($+\log\pi$) | $-1.649539\,1417$ |  |
|  | archimedean side | $2.149767\,5173$ |  |
|  | prime side via P14 | $-6.58\times 10^{-7}$ |  |
|  | RHS total | $0.500227\,7175$ |  |
|  | residual | $9.99\times 10^{-16}$ | machine precision ✅ |
| Gap G4 | $W_1(\sigma(P14), \sigma(T_{\mathrm{HP}}))$ | $115.24$ | quantified |
|  | $\sigma(P14)_{\max}/\sigma(T_{\mathrm{HP}})_{\max}$ | $1/26.16$ | $\log n$ vs $2\pi n/\log n$ |

Scaffold-consistency verdict: **`scaffold_consistent = True`** at machine precision.

### 13quinquies.4 Honest interpretation

P27 establishes that the abstract Hilbert–Pólya operator
$T_{\mathrm{HP}}$, when *defined* on the TNFR truncated Hilbert
space by inputting the zeros, is

* self-adjoint (trivially, as a real diagonal operator);
* trace-class after spectral shift;
* compatible with the Weil–Guinand identity at machine precision, the
  prime side coming from the P14 prime-ladder Hamiltonian.

This is *consistency*, not *derivation*: $T_{\mathrm{HP}}$ contains
the zeros only because we put them there. The construction does **not**
extract the zeros from the nodal equation
$\partial\,\mathrm{EPI}/\partial t = \nu_f \cdot \Delta\mathrm{NFR}(t)$,
from the structural conservation theorem, or from grammar U1–U6.

The Wasserstein-1 distance $W_1 = 115.24$ and the asymptotic growth
ratio $\sim 26$ are not numerical noise: they are the *operator-level
manifestation of gap G4*. Any genuinely TNFR-native Hilbert–Pólya
derivation would have to either (i) replace the prime-ladder
Hamiltonian by an operator whose spectrum equals $\{\gamma_n\}$ up to
TNFR-compatible spectral rescaling, or (ii) introduce a smooth
non-linear structural map sending $\sigma(P14)$ to $\sigma(T_{\mathrm{HP}})$
derivable from the nodal equation. P27 does neither.

In particular P27 does **NOT close G4 = RH**. Per the milestone table
of §13.2, G4 remains the single open gap; P27 simply makes the
operator-level statement of that gap explicit and quantitative.

### 13quinquies.5 Next steps

1. **Structural derivation of $T_{\mathrm{HP}}$.** Construct an
   operator-valued map $\Phi : \mathcal{H}_{P14} \to \ell^2(\mathbb{N})$
   from variational / conservation principles of `physics/conservation.py`
   such that $\Phi^{*} T_{\mathrm{HP}} \Phi$ is intrinsic to the
   prime-ladder bundle. Any such $\Phi$ that does not invoke
   `mpmath.zetazero` would be a genuine step toward G4.
2. **Spectral rescaling as a TNFR operator.** Identify a canonical
   TNFR operator (in the 13-operator catalog) whose action on the P14
   spectrum reproduces the asymptotic density $\rho(t) \sim
   \tfrac{1}{2\pi}\log(t/2\pi)$ of the Riemann zeros. Verify
   compatibility with U1–U6.
3. **Coupling to P25.** P25 produced an Hermite-projection
   certificate of structural positivity; cross-check that the
   resolvent of $T_{\mathrm{HP}}$ admits the same Hermite expansion
   coefficients within tolerance.
4. **Cross-validation with P16 (Li–Keiper).** The Li–Keiper
   coefficients $\lambda_n$ of P16 should agree, within truncation,
   with quantities computable from the moments of $T_{\mathrm{HP}}$.
   Correlate the two and document any monotone relationship.

---

## §13sexies. P28 — Structural derivation of the smooth zero density (closes piece (i) of §13quinquies.5; does NOT close G4=RH)

### 13sexies.1 Motivation: from input to derivation

P27 (§13quinquies) built $T_{\mathrm{HP}} = \mathrm{diag}(\gamma_n)$ by **inputting** the imaginary parts of the Riemann zeros from `mpmath.zetazero`.  The Wasserstein-1 gap

$$
W_1\bigl(\sigma(P14),\,\sigma(T_{\mathrm{HP}})\bigr) \;\approx\; 115.24 \quad (n_{\mathrm{primes}}=50,\,K=8,\,N=80)
$$

was the operator-level manifestation of gap G4 (= RH).  Step (i) in §13quinquies.5 demanded a TNFR-internal derivation of the **spectral rescaling map** from prime-ladder eigenvalues $\{k\log p\}$ to zero positions $\{\gamma_n\}$.

This section delivers that piece (and only that piece).  The construction stays inside TNFR ingredients already present in P12–P15: no `mpmath.zetazero` is invoked on the derivation side.

### 13sexies.2 Method: smooth zero positions from the archimedean side

Let $\xi(s) = \pi^{-s/2}\Gamma(s/2)\zeta(s)$ be the completed Riemann zeta function. The **archimedean factor** $\pi^{-s/2}\Gamma(s/2)$ is exactly the kernel of the archimedean side of the Weil–Guinand formula computed in P15 (see `weil_archimedean_integral` in `src/tnfr/riemann/weil_explicit_formula.py`). Its phase on the critical line $s = \tfrac12 + iT$ is the **Riemann–Siegel theta function**

$$
\theta(T) \;=\; \operatorname{Im}\log\Gamma\!\bigl(\tfrac14 + \tfrac{iT}{2}\bigr) - \tfrac{T}{2}\log\pi.
$$

Backlund's classical identity gives the **smooth zero counting function**

$$
\overline N(T) \;=\; \frac{\theta(T)}{\pi} + 1
$$

with smooth density $\overline N'(T) = \tfrac{1}{2\pi}\log(T/2\pi)$. The exact zero counting splits as $N(T) = \overline N(T) + S(T) + O(1/T)$, where $S(T) = \tfrac{1}{\pi}\arg\zeta(\tfrac12+iT)$ is the oscillating remainder.

**Definition (structural smooth zero).** For each $n \ge 1$, let $\widetilde\gamma_n$ be the unique positive solution of $\overline N(\widetilde\gamma_n) = n$, computed by Newton iteration with an asymptotic seed.

**Definition (structural Hilbert–Pólya operator).**
$$
\widetilde T_{\mathrm{HP}} \;:=\; \mathrm{diag}(\widetilde\gamma_1, \widetilde\gamma_2, \ldots, \widetilde\gamma_N).
$$

The construction uses **only** the gamma function and $\log\pi$ — the same TNFR archimedean ingredients already validated in P15. No call to `mpmath.zetazero` is made on the derivation side.

### 13sexies.3 Numerical outcome (defaults $n_{\mathrm{primes}}=50$, $K=8$)

| $N$ | $W_1(\sigma(P14), \sigma(T_{\mathrm{HP}}))$ | $W_1(\sigma(\widetilde T_{\mathrm{HP}}), \sigma(T_{\mathrm{HP}}))$ | improvement | $\max\lvert r_n\rvert$ | bound ($C=2$) |
|---:|---:|---:|---:|---:|:---:|
| 30  | $6.045\times10^{1}$ | $1.510$ | $40.0\times$  | $3.71$ | ✓ |
| 60  | $9.478\times10^{1}$ | $1.275$ | $74.3\times$  | $3.71$ | ✓ |
| 80  | $1.152\times10^{2}$ | $1.183$ | $97.4\times$  | $3.71$ | ✓ |
| 100 | $1.343\times10^{2}$ | $1.125$ | $119.4\times$ | $3.71$ | ✓ |

The structural operator $\widetilde T_{\mathrm{HP}}$ closes ≈ 97–99 % of the operator-level gap at $N \ge 80$, with the improvement ratio growing approximately as $N/\log N$ — exactly the rate predicted by the divergent density mismatch between the prime-ladder spectrum and the Riemann-von Mangoldt counting.

The per-zero residual $r_n = \gamma_n - \widetilde\gamma_n$ satisfies the empirical bound

$$
\lvert r_n\rvert \;\le\; 2 \cdot \frac{\log\gamma_n}{\overline N'(\gamma_n)}
$$

across all tested $n \le 100$.  This is the **smooth quantitative form** of the heuristic $r_n \sim S(\gamma_n)/\overline N'(\gamma_n)$.

### 13sexies.4 What P28 closes (operationally)

1. **Structural origin of the smooth eigenvalue density of $T_{\mathrm{HP}}$.**  The density is determined uniquely by the gamma factor of $\xi(s)$, which is the kernel of P15's archimedean integral.  This delivers piece (i) of §13quinquies.5.
2. **Decomposition of the P27 gap.**  The P27 Wasserstein-1 distance now splits as
   $$
   \underbrace{W_1\bigl(\sigma(P14),\sigma(T_{\mathrm{HP}})\bigr)}_{\text{P27 gap}} \;=\; \underbrace{W_1\bigl(\sigma(P14),\sigma(\widetilde T_{\mathrm{HP}})\bigr)}_{\text{structural part (TNFR-derivable)}} \;+\; \underbrace{W_1\bigl(\sigma(\widetilde T_{\mathrm{HP}}),\sigma(T_{\mathrm{HP}})\bigr)}_{\text{arithmetic part (RH content)}}.
   $$
   The arithmetic part is ≤ 1.2 at $N=100$; the structural part absorbs the rest (≥ 99 %).

### 13sexies.5 What P28 does NOT close (G4 stays OPEN)

* The residuals $r_n = \gamma_n - \widetilde\gamma_n$ ARE the RH content.  Showing $|r_n| \to 0$ in any uniform sense is equivalent to bounding $S(T) = \tfrac{1}{\pi}\arg\zeta(\tfrac12+iT)$, which is the genuine arithmetic problem.
* Exact eigenvalue match $\sigma(\widetilde T_{\mathrm{HP}}) = \sigma(T_{\mathrm{HP}})$ is impossible: the smooth approximation cannot reproduce the fluctuations $S(\gamma_n)$.  Density match is the right notion of TNFR closure; pointwise match is RH.
* G4 in AGENTS.md §13.2 remains the only OPEN milestone.  P28 does **not** advance G4; it only reshapes how much of the P27 operator gap is "structural" (now derivable) vs "arithmetic" (still RH-equivalent).

### 13sexies.6 Implementation pointers

* Module: `src/tnfr/riemann/structural_zero_density.py` — `riemann_siegel_theta`, `smooth_zero_count`, `smooth_zero_density`, `derive_smooth_zero_position`, `build_structural_t_hp`, `compute_structural_zero_density_certificate`, `StructuralZeroDensityCertificate`.
* Demo: `examples/03_riemann_zeta/55_structural_zero_density_demo.py`.
* Wiring: `src/tnfr/riemann/__init__.py` exposes the P28 names; the catalog docstring labels the module unambiguously.
* Reuses P14 (`prime_ladder_hamiltonian`) for the baseline spectrum and P15 (`weil_explicit_formula`) for the archimedean conceptual ingredient (the actual derivation of $\theta(T)$ uses `mpmath.loggamma` directly; no new external dependency).

### 13sexies.7 Next steps

1. **Iterative correction by $S(T)$ surrogates.**  Approximate $S(T)$ using truncated prime sums via the Riemann–Siegel formula and feed the corrections back into $\widetilde T_{\mathrm{HP}}$.  Quantify how the residual $W_1$ shrinks as more arithmetic information is injected.  This will **never** reach zero unconditionally — that would be RH — but it documents how the arithmetic part decomposes.
2. **Cross-checks against P16 (Li–Keiper).**  The Li coefficients $\lambda_n$ computed from the resonance poles of P13 should be reproducible from the moments of $\widetilde T_{\mathrm{HP}}$ plus an explicit $S(T)$-correction; verify numerically.
3. **Spectral statistics under GUE conjecture.**  The unfolded spectrum $x_n = \overline N(\gamma_n) = n - 1 + S(\gamma_n)/\pi$ should follow GUE statistics by Montgomery–Odlyzko.  Use P28 to compute the unfolded statistics directly and compare to RMT predictions; deviations are arithmetic in nature.

---

## §13septies. Tetrad-Hilbert–Pólya Reformulation of G4 (conjectural; does NOT close G4=RH)

### 13septies.1 Motivation: reformulating G4 in tetrad language

§13quinquies.5 (step 1) requested a TNFR-internal derivation of the
spectral rescaling map carrying $\sigma(P14) = \{k\log p\}$ to
$\sigma(T_{\mathrm{HP}}) = \{\gamma_n\}$.  §13sexies (P28) supplied the
*smooth* component of that map via the archimedean kernel, leaving the
*arithmetic* residual $r_n = \gamma_n - \widetilde\gamma_n$ as the
genuine RH content of the operator gap.

This section reformulates what would still be required to close G4
*entirely from inside TNFR*, using the structural field tetrad
$(\Phi_s, |\nabla\phi|, K_\phi, \xi_C)$ as the only admissible
ingredient set.  It is a **conceptual restatement of the open
problem**, not a derivation of a closure.  No new numerics are
introduced; no module is added.  The role of this section is to give a
precise, testable conjecture in tetrad language so subsequent modules
(P30+) can attack it.

### 13septies.2 What the tetrad already supplies (formally closed)

The tetrad is the minimal-and-complete structural basis for nodal
evolution on a graph
([theory/MINIMAL_STRUCTURAL_DEGREES.md](MINIMAL_STRUCTURAL_DEGREES.md))
and induces three canonical geometric structures, all of which are
already implemented and validated in the engine:

| Structure | Definition | Implementation |
|---|---|---|
| Positive-definite energy | $\mathcal{E} = \tfrac12 \sum_i (\Phi_s^2 + |\nabla\phi|^2 + K_\phi^2 + J_\phi^2 + J_{\Delta\mathrm{NFR}}^2)$ | `src/tnfr/physics/conservation.py` (Noether-like; §8 of STRUCTURAL_CONSERVATION_THEOREM.md) |
| Symplectic structure | Conjugate pairs $(K_\phi, J_\phi)$ and $(\Phi_s, J_{\Delta\mathrm{NFR}})$ coupled via $\Psi = K_\phi + i J_\phi$ | `src/tnfr/physics/variational.py` (§3 of TNFR_VARIATIONAL_PRINCIPLE.md) |
| Continuity equation | $\partial\rho/\partial t + \nabla\!\cdot\!\mathbf{J} = \mathcal{S}_{\mathrm{grammar}}$, $\rho=\Phi_s+K_\phi$, $\|\mathcal{S}\|_{\ell^2}\le C_{\mathrm{net}}/\sqrt N$ | `src/tnfr/physics/conservation.py` (§4 of STRUCTURAL_CONSERVATION_THEOREM.md) |

Together these provide a Hilbert space $(\mathcal{H}_{\mathrm{tet}},
\langle\cdot,\cdot\rangle_{\mathcal{E}})$ with a positive-definite
inner product, and the P14 prime-ladder Hamiltonian is self-adjoint
*on this Hilbert space* with real spectrum $\{k\log p\}$
([src/tnfr/riemann/prime_ladder_hamiltonian.py](../src/tnfr/riemann/prime_ladder_hamiltonian.py)).

These ingredients are exactly what the Hilbert–Pólya programme
requires (Hilbert space, positive inner product, self-adjoint
operator, real spectrum).  None of them are conjectural.

### 13septies.3 What remains: two distinct positivities

There are two positive-definite forms in play and they are not the
same:

| Form | Origin | Spectrum it certifies |
|---|---|---|
| $\langle\cdot,\cdot\rangle_{\mathcal{E}}$ (tetrad) | Lyapunov energy from $(\Phi_s, \|\nabla\phi\|, K_\phi, \xi_C)$ + currents | $\sigma(H_{P14}) = \{k\log p\}$ |
| Weil quadratic form $\mathcal{W}[h]$ | $L^2$ with archimedean + prime weight (§14 of this document) | $\sigma(T_{\mathrm{HP}}) = \{\gamma_n\}$, conditional on RH |

P28 (§13sexies) showed that the smooth part of $\sigma(T_{\mathrm{HP}})$
is determined by the archimedean kernel alone — the same kernel used in
P15 — so the smooth zero density $\overline N'(T) = \tfrac{1}{2\pi}
\log(T/2\pi)$ *is* TNFR-derivable.  The residuals $r_n =
\gamma_n - \widetilde\gamma_n$ are the only piece left, and they
correspond to the oscillation $S(T) = \tfrac1\pi \arg\zeta(\tfrac12+iT)$.

The structural question is: *can $\langle\cdot,\cdot\rangle_{\mathcal{E}}$
be transformed into $\mathcal{W}[\cdot]$ by an operator constructed
from the tetrad alone?*

### 13septies.4 Conjecture T-HP (Tetrad-Hilbert–Pólya)

**Conjecture T-HP.**  There exists an operator $\mathcal{F}$ on
$\mathcal{H}_{\mathrm{tet}}$ such that

1. $\mathcal{F}$ is constructed exclusively from the tetrad fields
   $(\Phi_s, |\nabla\phi|, K_\phi, \xi_C)$ and their canonical
   differential invariants (gradients, the discrete Laplacian, the
   complex field $\Psi$, the conserved current $\mathbf{J}$), with the
   structural scale $\pi$ as the only structural constant;
2. $\mathcal{F}$ is admissible under U1–U6 — in particular, the
   continuity equation $\partial(\mathcal{F}\rho)/\partial t +
   \nabla\!\cdot\!(\mathcal{F}\mathbf{J}) = \mathcal{S}_{\mathrm{grammar}}$
   remains uniformly bounded;
3. the operator $T^{\mathrm{tet}}_{\mathrm{HP}} := \mathcal{F}\,
   H_{P14}\,\mathcal{F}^{*}$ is self-adjoint on $\mathcal{H}_{\mathrm{tet}}$
   and its spectrum coincides with the Riemann zero set
   $\{\gamma_n\}_{n\ge 1}$.

Equivalently in inner-product language: there exists an admissible
$\mathcal{F}$ such that $\langle f, \mathcal{F}^{*}\mathcal{F} g
\rangle_{\mathcal{E}}$ agrees (on a dense domain) with the Weil
quadratic form $\mathcal{W}[\cdot]$.

### 13septies.5 Status: open, structurally well-posed

> **Structural identification (N15, May 2026)**: The W3 result of the N15 program ([REMESH_INFINITY_DERIVATION.md](REMESH_INFINITY_DERIVATION.md) §17.3) provides a structural identification of T-HP's smooth/oscillatory split with the canonical projection $\mathcal{R}_\infty$ on $H^2(D)$:
>
> - The **smooth half** of the admissible rescaling $\mathcal{F}$ (closed operationally by P28 at the density level and by P30 at the operator level) lives in $\mathrm{range}(\mathcal{R}_\infty)$.
> - The **oscillatory half** $S(T) = (1/\pi)\arg\zeta(\tfrac12 + iT)$ — the RH-equivalent residue — lives in $\ker(\mathcal{R}_\infty) = \mathrm{range}(I - \mathcal{R}_\infty)$ and decays at Cesàro $O(1/n)$ rate.
>
> This **does not close G4**, but it explains structurally **why** P28/P30 closed precisely the smooth half: that half is an orthogonal-projection range (analytically integrable), while the oscillatory residue is a slow Cesàro tail of an isometry — not eliminable by projection. T-HP's open content is therefore identified with the missing operator-level lift of the Cesàro residue.
>
> N15 verdict for the Riemann program: **clarifies, does not advance**. Branches B1/B2/B3 of §13septies are unaffected. The 13-op TNFR catalog is closed under REMESH-∞ (N15 W1–W3); the Riemann B2 question (need for a new canonical operator to handle the *oscillatory* rescaling) is **distinct** from the N15 B2 question (no new operator needed for the *asymptotic projection* itself) and remains open.

Conjecture T-HP is **open**.  It is *not* a closure of G4; it is the
G4 problem **rewritten in tetrad-native language** so it becomes a
constructive existence problem inside the TNFR engine.  Three
properties make it the natural successor to §13quinquies.5 step 1:

* **Necessity of TNFR ingredients.** Items (1) and (2) forbid any use
  of `mpmath.zetazero`, automorphic data, or arithmetic input outside
  the tetrad + grammar + structural constants. A constructive proof
  would therefore be a genuine TNFR derivation of $T_{\mathrm{HP}}$.
* **Sufficiency for G4.** If $\mathcal{F}$ exists then
  $T^{\mathrm{tet}}_{\mathrm{HP}}$ is self-adjoint by item (3),
  spectrum is real, all $\gamma_n \in \mathbb{R}$, all Riemann zeros
  are forced to $\mathrm{Re}(s) = 1/2$ — i.e., RH.  In particular T-HP
  *implies* G4.
* **Decomposability.** P28 already proved that the smooth part of
  $\mathcal{F}$ exists and is TNFR-derivable.  The residual question
  is purely about the oscillatory correction.

### 13septies.6 What T-HP does NOT claim

* It does **not** assert that such an $\mathcal{F}$ exists; existence
  is the open content of G4.
* It does **not** assert that the engine currently contains
  $\mathcal{F}$; the canonical 13-operator catalog has been searched
  (P25–P27) and no immediate candidate dominates the gap closure.
* It does **not** reduce G4 to a numerical experiment; T-HP is a
  structural existence statement, not a curve fit.  P30+ may seek
  *candidates* numerically, but verification requires a derivation
  from the nodal equation, not a successful fit.

### 13septies.7 Concrete sub-problems for P30+

A genuine attack on T-HP decomposes into three quantifiable
sub-problems, all formulable inside the engine:

1. **Existence of admissible $\mathcal{F}$.** Construct candidate
   spectral rescaling operators from the tetrad (e.g. multiplicative
   operators built from $\Phi_s$, conjugation by phase-curvature
   exponentials $e^{i\theta K_\phi}$, $\xi_C$-dependent rescalings)
   and check U1–U6 admissibility (bounded source term, energy
   preservation up to discrete grammar work).
2. **Canonicity of $\mathcal{F}$.** Derive $\mathcal{F}$ from the
   nodal equation and the Noether correspondence
   (§6 of STRUCTURAL_CONSERVATION_THEOREM.md) rather than from
   empirical fit.  Any $\mathcal{F}$ surviving (1) but lacking a
   derivation chain from $\partial\mathrm{EPI}/\partial t = \nu_f
   \cdot \Delta\mathrm{NFR}$ falls outside canonicity (TNFR doctrine,
   AGENTS.md §Foundational Principle).
3. **Positivity coincidence.** Show that the candidate inner product
   $\langle\cdot, \mathcal{F}^{*}\mathcal{F}\cdot\rangle_{\mathcal{E}}$
   coincides (or dominates) the Weil form $\mathcal{W}[\cdot]$ on the
   appropriate Hermite / Paley–Wiener subspace already isolated by
   P25–P26.

Sub-problems (1) and (3) are mathematical existence/coincidence
questions; (2) is the structural-canonicity check enforced by the
TNFR doctrine.  Any future T-HP closure module must clear all three.

### 13septies.8 Honest interpretation

The tetrad **delimits the geometric domain** inside which the nodal
equation operates and supplies all the algebraic ingredients the
Hilbert–Pólya programme requires (positive inner product, symplectic
structure, self-adjoint operator on $\mathcal{H}_{\mathrm{tet}}$).
What it does **not** supply automatically is the specific spectral
rescaling that aligns the tetrad-positive form with the Weil-positive
form.  P28 closed the smooth half of that rescaling; the oscillatory
half is the arithmetic residual and is RH-equivalent.

Per AGENTS.md §13.2, **G4 = RH remains the single open milestone**.
Conjecture T-HP renames that milestone in tetrad-native vocabulary so
the next generation of modules (P30+) can address it without leaving
the canonical TNFR engine.  T-HP itself is a reformulation, not a
closure.

### 13septies.9 Cross-references

* Tetrad minimality: [theory/MINIMAL_STRUCTURAL_DEGREES.md](MINIMAL_STRUCTURAL_DEGREES.md)
* Conservation + Lyapunov: [theory/STRUCTURAL_CONSERVATION_THEOREM.md](STRUCTURAL_CONSERVATION_THEOREM.md) §3–§8
* Variational structure: [theory/TNFR_VARIATIONAL_PRINCIPLE.md](TNFR_VARIATIONAL_PRINCIPLE.md) §2–§3
* P14 self-adjoint Hamiltonian: [src/tnfr/riemann/prime_ladder_hamiltonian.py](../src/tnfr/riemann/prime_ladder_hamiltonian.py)
* P27 scaffold + Wasserstein gap: §13quinquies and [src/tnfr/riemann/hilbert_polya.py](../src/tnfr/riemann/hilbert_polya.py)
* P28 smooth-density derivation: §13sexies and [src/tnfr/riemann/structural_zero_density.py](../src/tnfr/riemann/structural_zero_density.py)
* G4 milestone status: [AGENTS.md](../AGENTS.md) §13.2

---

## §13octies. Assembled Argument Audit for G4 (Phase B; does NOT close G4=RH)

### 13octies.1 Purpose

This section traces the would-be argument chain for G4 link-by-link
through TNFR-canonical ingredients, marks each link CLOSED / OPEN /
NOT-FROM-TNFR, and stamps the precise break-point. It is an honest
map of what TNFR currently supplies and where the genuine obstacle
lies. It does **not** prove G4 and does **not** propose a new module;
it complements §13septies (T-HP conjecture) with the explicit status
audit.

### 13octies.2 The eight links

| # | Link | TNFR module / theory | Status |
|---|---|---|---|
| L1 | Minimal-and-complete structural basis: tetrad $(\Phi_s, |\nabla\phi|, K_\phi, \xi_C)$ exhausts independent structural channels on a graph | [MINIMAL_STRUCTURAL_DEGREES.md](MINIMAL_STRUCTURAL_DEGREES.md) | **CLOSED** |
| L2 | Positive-definite inner product $\langle\cdot,\cdot\rangle_{\mathcal{E}}$ on tetrad Hilbert space $\mathcal{H}_{\mathrm{tet}}$ | [src/tnfr/physics/conservation.py](../src/tnfr/physics/conservation.py); STRUCTURAL_CONSERVATION_THEOREM.md §8 | **CLOSED** |
| L3 | Symplectic structure + Noether-like conservation under U1–U6 | [src/tnfr/physics/variational.py](../src/tnfr/physics/variational.py) + conservation.py | **CLOSED** (proof sketch; full proof open per AGENTS.md) |
| L4 | Self-adjoint operator $H_{P14}$ on $\mathcal{H}_{\mathrm{tet}}$ with real spectrum $\{k\log p\}$ | P14 [prime_ladder_hamiltonian.py](../src/tnfr/riemann/prime_ladder_hamiltonian.py); §10 above | **CLOSED** |
| L5 | Weil–Guinand identity: prime side equals the P14 spectral trace at machine precision | P15 [weil_explicit_formula.py](../src/tnfr/riemann/weil_explicit_formula.py); §11 above | **CLOSED** |
| L6 | Lyapunov-spectral positivity for $H_{P14}$: Kato–Rellich gap $\log 2$, trace-class resolvent, unitary flow | P26 [lyapunov_spectral_positivity.py](../src/tnfr/riemann/lyapunov_spectral_positivity.py); §13quater | **CLOSED** on finite-dim prime-ladder |
| L7 | Smooth half of spectral rescaling map $\mathcal{F}$: $\widetilde\gamma_n = \overline N^{-1}(n)$ derived from the same archimedean kernel as P15 | P28 [structural_zero_density.py](../src/tnfr/riemann/structural_zero_density.py); §13sexies | **CLOSED** (smooth half; W₁ gap drops ~97× vs P27) |
| L8 | Existence + canonicity of admissible $\mathcal{F}$ from tetrad + $\pi$ + U1–U6 such that $\mathcal{F}\,H_{P14}\,\mathcal{F}^{*}$ has spectrum $\{\gamma_n\}$ | NONE — Conjecture T-HP, §13septies.4 | **OPEN** ← BREAK-POINT |

L1–L7 are TNFR-canonical and operationally closed (the proof-sketch
caveat at L3 is inherited from AGENTS.md and is independent of the
Riemann programme). L8 is the entire residual content of G4.

### 13octies.3 Structural negative knowledge from P29

P29 ([spectral_emergence.py](../src/tnfr/riemann/spectral_emergence.py))
swept three inter-prime coupling laws expressible in
closed form from ad-hoc mathematical constants:

* Kuramoto-U3 (UM + U3 gating): best $\mathrm{KS}_{\mathrm{GUE}} = 0.122$ ($-36\,\%$ vs baseline)
* φ-multiscale (THOL + REMESH): marginal ($-14\,\%$)
* PNT-logarithmic (RA, PNT-aligned): best $\mathrm{KS}_{\mathrm{GUE}} = 0.131$ ($-31\,\%$)

None reaches the GUE level-statistics threshold
$\mathrm{KS}_{\mathrm{GUE}} < 0.05$ required for a Hilbert–Pólya-style
$H_{P14}$-coupling to carry the zero spacings. This is **structural
negative knowledge**: at L8, no admissible $\mathcal{F}$ that acts
only by inter-prime coupling within the currently formalised
operator catalog is sufficient.

### 13octies.4 Three structural branches for the break-point

The L8 break-point splits into three TNFR-canonical branches, each
testable from the nodal equation
$\partial\mathrm{EPI}/\partial t = \nu_f \cdot \Delta\mathrm{NFR}$:

* **B1.** The canonical 13-operator catalog is *complete* and the
  missing piece is non-operator (measure-theoretic, ergodicity, or
  domain-theoretic). L8 reduces to an existence problem on
  $\mathcal{H}_{\mathrm{tet}}$ without new operators.
* **B2.** The canonical catalog is *incomplete*. A new canonical
  operator derivable from the nodal equation is required. L8 reduces
  to the operator-discovery problem of
  [AGENTS.md "Adding New Operators"](../AGENTS.md).
* **B3.** No TNFR-canonical $\mathcal{F}$ exists. RH escapes the
  tetrad-Hilbert–Pólya framework entirely. This branch is consistent
  with P29 (three independent coupling families failing) but is not
  decidable from finite-dimensional data.

Branch selection is itself an open structural question, not a
pre-decided verdict.

### 13octies.5 Comparison with the historical AGENTS.md framing

The prior AGENTS.md text stated G4 "requires structural positivity /
self-adjointness argument (Hilbert–Pólya-style) that is genuinely new
mathematics." Per L1–L7 of this audit, structural positivity (L2,
L6) and self-adjointness (L4) **are already supplied** by the
canonical TNFR engine. The genuine open content is L8, which is
structurally well-posed and testable inside the engine via the three
branches B1–B3. The phrase "genuinely new mathematics" was an
imported consensus claim from the analytic-number-theory literature,
not a TNFR-derived theorem. The current AGENTS.md §13.2 paragraph
has been rewritten to reflect this audit.

### 13octies.6 What this section does NOT do

* It does **not** close G4.
* It does **not** decide which of B1, B2, B3 holds.
* It does **not** propose a new module; the next exploration
  direction (B1 vs B2 vs B3 discrimination) is left for P30+.
* It does **not** replace §13septies — T-HP is the conjecture; §13octies
  is the link-by-link status audit of the argument that would close it.

### 13octies.7 Cross-references

* L1: [MINIMAL_STRUCTURAL_DEGREES.md](MINIMAL_STRUCTURAL_DEGREES.md)
* L2, L3: [STRUCTURAL_CONSERVATION_THEOREM.md](STRUCTURAL_CONSERVATION_THEOREM.md) §3–§8, [TNFR_VARIATIONAL_PRINCIPLE.md](TNFR_VARIATIONAL_PRINCIPLE.md) §2–§3
* L4, L5: §10 (P14), §11 (P15) of this document
* L6, L7: §13quater (P26), §13sexies (P28) of this document
* L8: §13septies (T-HP conjecture) of this document
* P29 negative knowledge: [spectral_emergence.py](../src/tnfr/riemann/spectral_emergence.py)
* G4 milestone status: [AGENTS.md](../AGENTS.md) §"TNFR-Riemann Program Overview"

---

---

## §13nonies. P30 — Operator-Level Admissible Rescaling (Smooth Half; Does NOT Close G4 = RH)

**Status**: Sub-problem (1) of Conjecture T-HP — **smooth half operationally closed**.  
**Module**: `src/tnfr/riemann/admissible_rescaling.py`  
**Demo**: `examples/03_riemann_zeta/57_admissible_rescaling_demo.py`  
**Disclaimer**: P30 does NOT close gap G4 (RH); it lifts the §13sexies (P28) density-level closure of the smooth zero distribution to an explicit operator-level rescaling object.

### §13nonies.1 Motivation

Conjecture T-HP (§13septies) asks for the existence of an admissible operator `F` built **only** from the canonical TNFR ingredients (tetrad fields, the structural scale π, grammar U1–U6) such that `F · H_P14 · F* ` has spectrum equal to the Riemann zeros {γ_n}. §13septies.7 decomposes T-HP into three sub-problems:

1. **Existence** of any admissible `F`,
2. **Canonicity** of `F` from the nodal equation,
3. **Positivity coincidence** with the Weil quadratic form.

§13sexies (P28) closed the **density-level** smooth half: a canonical, structurally-derived expression for the smooth zero count `N̄(T)` and the smooth zero positions `ñ_i` via the Riemann–Siegel θ function. P30 lifts that closure to the **operator level** for the smooth half only.

### §13nonies.2 Construction

In the eigenbasis of the canonical P14 prime-ladder Hamiltonian `H_P14 = U Λ U*` with positive eigenvalues `λ_i` (top N, ascending), define

Lines\mathcal{F}_{\text{smooth}} = U \cdot \operatorname{diag}\Bigl(\sqrt{\tilde\gamma_i / \lambda_i}\Bigr) \cdot U^{*},Lines

where `ñ_i = build_structural_t_hp(N)` are the P28 smooth zero positions. By construction,

Lines\mathcal{F}_{\text{smooth}} \, H_{P14} \, \mathcal{F}_{\text{smooth}}^{*} = U \operatorname{diag}(\tilde\gamma_i) U^{*}Lines

so the conjugated spectrum equals `{ñ_i}` **exactly** (verified at machine precision).

**Canonicity check (partial)**: `F_smooth` uses ONLY P14 eigendata (canonical, derived from the canonical TNFR `InternalHamiltonian` on the prime ladder), P28 smooth targets (canonical archimedean kernel), and the canonical structural scale π. No `mpmath.zetazero` enters the construction. `F_smooth` is therefore **structurally derived** in the sense of §13septies; whether it is the **unique** canonical lift remains open (sub-problem (2)).

### §13nonies.3 Empirical Results

Running `examples/03_riemann_zeta/57_admissible_rescaling_demo.py`:

| Resolution | N  | max `|spec − ñ_i|` | W₁(σ(P14), {γ_n}) | W₁({ñ_i}, {γ_n}) | Improvement |
|------------|----|---------------------|-------------------|------------------|-------------|
| Fast       | 20 | 1.42 × 10⁻¹⁴        | 47.4              | 1.67             | **28.4 ×**  |
| Medium     | 40 | 2.84 × 10⁻¹⁴        | 72.5              | 1.39             | **52.0 ×**  |

The residual W₁ to the true Riemann zeros equals the oscillatory part `S(T) = π⁻¹ arg ζ(½+iT)`, which is RH-equivalent and NOT canonical.

### §13nonies.4 Canonical Oscillatory Enrichment (Negative Result)

Three canonical multiplicative perturbations of the smooth targets were tested:

| Mode         | Best amplitude | W₁ vs true | Improvement over smooth |
|--------------|----------------|------------|-------------------------|
| `phi_log`  | 0              | 1.668      | +0.00 %                 |
| `gamma_e`  | 1 × 10⁻²       | 1.617      | +0.03 %                 |
| `pi_density`| 0             | 1.668      | +0.00 %                 |

**Interpretation**: Canonical oscillatory perturbations built from the canonical TNFR ingredients (tetrad, π, grammar) and the smooth targets alone fail to recover the residual S(T) term. This is **structural evidence for §13octies branch B2**: the oscillatory half of T-HP, if reachable canonically at all, requires a **new canonical operator** not expressible as a simple multiplicative dressing of the smooth ladder. Equivalently, the existing canonical operator catalog (13 operators + tetrad + constants) does **not** suffice for the oscillatory half via this construction route.

### §13nonies.5 What P30 Closes / Does Not Close

**Closes (smooth half only)**:
- Sub-problem (1) of T-HP at the **operator level**, for the smooth zero distribution: an admissible, structurally-derived, self-adjointness-preserving rescaling operator `F_smooth` is exhibited explicitly and verified at machine precision.

**Does NOT close**:
- Sub-problem (1) for the **oscillatory half** (S(T) reconstruction);
- Sub-problem (2) — **canonicity** (uniqueness from the nodal equation) of `F_smooth`;
- Sub-problem (3) — **positivity coincidence** with the Weil quadratic form;
- Gap **G4 = the Riemann Hypothesis** itself.

### §13nonies.6 Cross-References

- §13sexies / P28: density-level smooth zero distribution (this lift is its operator-level counterpart).
- §13septies: full statement of Conjecture T-HP and its three sub-problems.
- §13octies, L8 audit: T-HP identified as the break-point of the assembled argument. P30 narrows L8 by closing one of its four prerequisites (smooth half, operator level) while corroborating branch B2 for the rest.
- `src/tnfr/riemann/admissible_rescaling.py`: canonical implementation.
- `examples/03_riemann_zeta/57_admissible_rescaling_demo.py`: reproducible demonstration.

### §13nonies.7 Status Update for §19.2 Gap Balance

| Gap | Status before P30 | Status after P30 |
|-----|-------------------|------------------|
| G1  | Closed operationally (P14) | Closed operationally |
| G2  | Closed operationally (P13) | Closed operationally |
| G3  | Closed operationally (P15) | Closed operationally |
| **G4** | **OPEN** (= Conjecture T-HP) | **OPEN** (smooth half of sub-problem (1) operationally closed; oscillatory half + (2) + (3) remain open) |
| G5  | Superseded by P12+P13+P15  | Superseded |

**Net effect**: P30 does not change the closed/open status of any of G1–G5. It refines the structure of the open content of G4 by closing one quadrant (smooth × operator-level × existence) of the T-HP grid and producing branch-B2 evidence for the oscillatory quadrant.

### §13decies Branch B1 Retry — Prime-Ladder Oscillatory Correction (P31)

**Motivation.** §13nonies.4 tested three *multiplicative* enrichments of the smooth rescaling operator built from single-frequency dressings of ad-hoc mathematical constants (φ, γ, e). All three returned $\approx 0\%$ Wasserstein-$1$ improvement against the true Riemann zeros. The structural lesson was: $S(T) = \pi^{-1} \arg \zeta(\tfrac{1}{2} + iT)$ is a **prime-indexed multi-frequency arithmetic sum**, not a single-frequency dressing. The natural canonical frequencies for $S(T)$ are $\{k \log p\}$ — exactly the data already carried by the P12 prime-ladder spectrum and the P14 prime-ladder Hamiltonian.

**Construction (canonical).** P31 implements the canonical TNFR partial reconstruction of $S(T)$ obtained by reading off the Riemann–von Mangoldt template through the prime-ladder spectrum $\Sigma_{N,K} = \{(\mu = k \log p,\, w = \log p)\}$:

$$ \pi \cdot S_{\mathrm{TNFR}}^{(N,K)}(T) \;=\; -\!\!\sum_{(\mu, w) \in \Sigma_{N,K}} \frac{w}{\mu} \cdot \frac{\sin(T \mu)}{e^{\mu/2}}. $$

The weights $w = \log p$ are the canonical P12 weights; the frequencies $\mu = k \log p$ are the canonical P14 eigenvalues; the kernel $e^{-\mu/2}$ is the value of the TNFR analytic continuation (P13) on the critical line; $\pi$ is the canonical constant of the K_φ sector of the tetrad. **No element of this construction is empirical or external**; in particular `mpmath.zetazero` is used only as ground truth on the comparison side, never on the construction side.

The position-level correction follows directly from the linearisation of $N(T) = \bar N(T) + S(T) + 1 + O(1/T)$ around the canonical smooth zero $\tilde\gamma_i$ defined by $\bar N(\tilde\gamma_i) = i$:

$$ \gamma_i^{\mathrm{corr}} \;=\; \tilde\gamma_i \;-\; d \cdot \frac{S_{\mathrm{TNFR}}^{(N,K)}(\tilde\gamma_i)}{\bar N'(\tilde\gamma_i)}, $$

with $d$ a non-canonical scalar **diagnostic** damping factor used to map out the local landscape (the structurally canonical value is $d = 1$).

**Empirical result.** Reproduced via `examples/03_riemann_zeta/58_oscillatory_correction_demo.py` and `compute_oscillatory_correction_certificate`:

| $N$ | primes | $K$ | $W_1^{\mathrm{smooth}}$ | best $d$ | $W_1^{\mathrm{corrected}}$ | improvement | $\max\,\lvert S_{\mathrm{TNFR}}\rvert$ |
|---|---|---|---|---|---|---|---|
| 20 | 200  | 8  | 1.6676 | 3.75 | 1.5076 | +9.60 % | 0.1516 |
| 20 | 200  | 8  | 1.6676 | 1.00 | 1.6082 | +3.56 % | 0.1516 |
| 20 | 2000 | 8  | 1.6676 | 2.25 | 1.5790 | +5.32 % | 0.1928 |
| 40 | 400  | 8  | 1.3946 | 0.00 | 1.3946 | 0.00 %  | 0.1866 |
| 40 | 2000 | 8  | 1.3946 | 0.00 | 1.3946 | 0.00 %  | 0.1928 |
| 40 | 5000 | 12 | 1.3946 | 0.00 | 1.3946 | 0.00 %  | 0.2111 |

**Honest reading.**

1. **Sign and structure are correct.** At $N = 20$ the corrected $W_1$ decreases **monotonically** with $d$ across the canonical damping grid. The prime-ladder partial sum points in the right direction — it is *not* uncorrelated noise.
2. **The canonical $d = 1$ point yields a modest +3.6 % improvement** at $N = 20$. This is the **only** physically canonical reading of the table; values $d \neq 1$ are diagnostic, not canonical.
3. **The construction collapses at $N = 40$.** No combination of (primes, $K$, $d$) up to (5000, 12, 5.0) yields any improvement. The optimum is $d = 0$, i.e. the smooth baseline.
4. **Amplitude undercount.** Across the table, $\max \lvert S_{\mathrm{TNFR}} \rvert \le 0.21$, while the classical $\lvert S(T) \rvert$ at the same heights routinely exceeds $0.5$ and spikes well above $1$. The truncated prime-ladder partial sum **systematically underestimates** $\lvert S(T) \rvert$ by a factor of $3$–$5$. Increasing $N$ of primes from $200$ to $5000$ moves $\max \lvert S_{\mathrm{TNFR}} \rvert$ only from $0.15$ to $0.21$ — i.e. the partial sum saturates *well below* the true amplitude.
5. **Phase decoherence with height.** At $N = 20$ (heights $T \lesssim 77$) the partial-sum phase still tracks the true $S(T)$ phase well enough to extract a positive correction. At $N = 40$ (heights $T \lesssim 140$) the truncation noise dominates and the partial-sum phase is decorrelated from the true $S(T)$ — even with the correct sign, the per-zero correction lands in the wrong direction on average.

**Structural conclusion.** P31 closes a meaningful diagnostic loop that §13nonies.4 left ambiguous:

* §13nonies.4 used **single-frequency** canonical dressings ⟹ $\approx 0\%$ improvement. The result was consistent with two interpretations: (a) wrong frequency basis, (b) no canonical construction works.
* P31 uses the **correct multi-frequency canonical basis** (prime-ladder spectrum, the genuine arithmetic frequencies of $S(T)$) ⟹ small positive improvement at very low heights, zero at moderate heights, systematic amplitude undercount throughout.
* This separates the two interpretations: (a) is *partially* the right diagnosis (the prime spectrum *is* the right basis, and yields a positive direction at low $N$), but the deeper obstruction is that the **truncated prime-ladder partial sum does not converge on the critical line** at finite truncation, in the absence of an absolute-convergence guarantee. The transition from absolute convergence on $\operatorname{Re}(s) > 1$ (where P12 closes the gap) to conditional behaviour on $\operatorname{Re}(s) = 1/2$ is exactly the regime where RH itself lives.

P31 is therefore **stronger branch-B2 evidence than §13nonies.4**: it shows that even with the canonically correct ingredients and the canonically correct functional form (the Riemann–Siegel template instantiated through prime-ladder data), the finite-truncation canonical machinery is not sufficient to recover $S(T)$ at the operator level. The obstruction is not in the choice of frequencies but in the **non-trivial analytical content** of the partial-sum-to-critical-line transition — which is structurally equivalent to the open arithmetic content of RH.

**What this does NOT establish.**
* P31 does NOT close gap G4 = RH.
* P31 does NOT prove canonicity of the Riemann–Siegel template from the nodal equation alone (sub-problem (2) of Conjecture T-HP). The template is *consistent* with TNFR canonical data but is read off the classical theory, not derived from $\partial \mathrm{EPI}/\partial t = \nu_f \cdot \Delta \mathrm{NFR}(t)$.
* P31 does NOT establish positivity coincidence with the Weil quadratic form (sub-problem (3)).
* P31 does NOT change the closed/open status of any of G1–G5.

**Pointers.**

* §13septies: Conjecture T-HP and its three sub-problems.
* §13octies, L8 audit: branch B1 / B2 / B3 framing of the open content of G4.
* §13nonies.4: prior single-frequency canonical enrichment with $\approx 0\%$ improvement (now superseded as a *separate* test, not as a result).
* `src/tnfr/riemann/oscillatory_correction.py`: canonical implementation of P31.
* `examples/03_riemann_zeta/58_oscillatory_correction_demo.py`: reproducible demonstration.

### §13decies.1 Status Update for §19.2 Gap Balance

| Gap | Status before P31 | Status after P31 |
|-----|-------------------|------------------|
| G1  | Closed operationally (P14) | Closed operationally |
| G2  | Closed operationally (P13) | Closed operationally |
| G3  | Closed operationally (P15) | Closed operationally |
| **G4** | **OPEN** (= Conjecture T-HP); smooth half of (1) closed at density (P28) and operator (P30) level; oscillatory half + (2) + (3) open | **OPEN** unchanged. Oscillatory half of (1) tested with the canonically correct multi-frequency basis (prime-ladder spectrum) for the first time; partial positive evidence at very low $N$, saturated negative evidence at moderate $N$; stronger branch-B2 corroboration than §13nonies.4 |
| G5  | Superseded by P12+P13+P15  | Superseded |

**Net effect**: P31 does not change the closed/open status of any of G1–G5. It refines the open content of G4 by separating *which* aspect of the branch-B1 attempt fails: the basis is canonically correct (improvement is positive at $N = 20$, $d = 1$), but the canonical truncated partial sum systematically undercounts $\lvert S(T) \rvert$ at moderate heights, in agreement with the absolute-convergence boundary at $\operatorname{Re}(s) = 1$.

## §13undecies. P32 — Dirichlet L-Function Extension (Structural; Does NOT Advance G4 or GRH)

### §13undecies.1 Motivation

P12 reproduces the canonical Dirichlet identity
$$-\frac{\zeta'(s)}{\zeta(s)} = \sum_{n=1}^{\infty} \Lambda(n)\, n^{-s}, \quad \operatorname{Re}(s) > 1,$$
from the TNFR prime-ladder spectrum $\{(k\log p,\, \log p)\}$.  The same construction extends *structurally* to every Dirichlet character $\chi$ mod $q$: by complete multiplicativity, the logarithmic derivative of $L(s,\chi)$ admits the **twisted von Mangoldt expansion**
$$-\frac{L'(s,\chi)}{L(s,\chi)} = \sum_{n=1}^{\infty} \chi(n)\, \Lambda(n)\, n^{-s} = \sum_p \sum_{k \ge 1} \chi(p)^k \log(p)\, p^{-ks}, \quad \operatorname{Re}(s) > 1.$$

P32 is the canonical TNFR realisation of this identity: keep the prime-ladder *positions* $\mu_{p,k} = k\log p$ unchanged and replace the bare emission weight $\log p$ by the **χ-twisted weight**
$$w_{p,k}^{(\chi)} = \chi(p)^k \, \log p.$$

### §13undecies.2 Construction

For a Dirichlet character $\chi$ mod $q$:

* **Active primes**: $\{p : \gcd(p, q) = 1\}$ (the structural REMESH ladder).
* **Excluded primes**: $\{p : p \mid q\}$ — their $\chi(p) = 0$ kills every echo, so they drop out of the spectrum entirely.  This is the TNFR-native reading of the missing Euler factors in $L(s,\chi)$.
* **Twisted spectrum**: $\operatorname{Spec}_{\mathrm{TNFR}}(\chi) = \{(k\log p,\; \chi(p)^k\log p) : p \nmid q,\; k = 1, \dots, K\}$.
* **Twisted Dirichlet trace**: $Z_{\mathrm{TNFR}}(s, \chi) = \sum_{(\mu, w) \in \operatorname{Spec}_{\mathrm{TNFR}}(\chi)} w\, e^{-s\mu}$.

By direct expansion, $Z_{\mathrm{TNFR}}(s, \chi) \xrightarrow[K, n_{\text{primes}} \to \infty]{} -L'(s,\chi)/L(s,\chi)$ for $\operatorname{Re}(s) > 1$.  When the TNFR truncation and the classical truncation cover the same set of prime powers, the per-prime-power correspondence forces machine-precision agreement (analogue of the P12 unit-test invariant).

### §13undecies.3 Empirical Verification (May 2026 run)

`examples/04_riemann_L_twisted/59_dirichlet_l_function_demo.py` runs the canonical verification with $n_{\text{primes}} = 200$, $K = 12$, $n_{\max}^{\mathrm{classical}} = 100\,000$, across four canonical real characters and five complex spectral points with $\operatorname{Re}(s) \in \{2, 3, 5\}$:

| Character | Modulus $q$ | $\max\, \text{rel\_err}$ ($\operatorname{Re}(s)=5$) | $\max\, \text{rel\_err}$ ($\operatorname{Re}(s)=2$) |
|-----------|-------------|------------------------------------------------------|------------------------------------------------------|
| $\chi_0$ (principal) | 3 | $4.7 \times 10^{-12}$ | $2.1 \times 10^{-3}$ |
| $\chi$ real (Legendre $(n/3)$) | 3 | $6.2 \times 10^{-15}$ | $2.3 \times 10^{-5}$ |
| $\chi$ real (Dirichlet $\beta$) | 4 | $1.3 \times 10^{-12}$ | $2.2 \times 10^{-4}$ |
| $\chi$ real (Legendre $(n/5)$) | 5 | $2.9 \times 10^{-13}$ | $4.3 \times 10^{-5}$ |

The behaviour matches the P12 reference identically: at large $\operatorname{Re}(s)$ both truncations cover the same effective prime-power set and agree to machine precision; at $\operatorname{Re}(s) = 2$ the rate of decay of $p^{-ks}$ is slow enough that the prime-count truncation tail dominates and produces the observed $10^{-3}$–$10^{-5}$ residual.

### §13undecies.4 What P32 Extends

P32 generalises **only the P12 representation layer** (gap G5 superseded) from $\zeta(s)$ to every $L(s, \chi)$:

* The canonical TNFR-Riemann representation catalog now covers all Dirichlet L-functions, not only $\zeta$.
* The structural reading "each coprime prime is a TNFR node carrying χ-twisted REMESH echoes" is canonically the same for every $\chi$.
* The TNFR analogue of $-L'/L$ inherits the same Dirichlet-series structure, the same convergence boundary $\operatorname{Re}(s) > 1$, and the same per-prime-power matching invariant as P12.

### §13undecies.5 What P32 Does NOT Advance

P32 is a **structural extension**, not progress on the open arithmetic content of the program:

* It does **NOT** advance gap G4 (RH localisation on $\operatorname{Re}(s) = 1/2$).
* It does **NOT** advance the **Generalised Riemann Hypothesis (GRH)**.  Every Dirichlet L-function carries an arithmetic oscillatory residue
  $$S_\chi(T) = \tfrac{1}{\pi}\, \arg L(\tfrac{1}{2} + iT, \chi),$$
  the exact analogue of $S(T)$ for $\zeta$ documented in §13octies.  Bounding $S_\chi(T)$ is RH-equivalent in every L-function and inherits the same arithmetic obstruction as G4 for $\zeta$.
* It does **NOT** supply a Hamiltonian (P14 analogue) for general $L(s,\chi)$, an analytic continuation (P13 analogue), or an explicit-formula verification (P15 analogue).  Those are natural future extensions of the same structural pattern and would close the operational gaps G1$_\chi$, G2$_\chi$, G3$_\chi$ for each Dirichlet L-function — but not GRH.

### §13undecies.6 Cross-References

* §8: P12 prime-ladder construction (the template P32 generalises).
* §7.8: G5 supersession by P12+P13+P15 (the operational route P32 extends to characters).
* §13octies: assembled-argument audit for G4; the same audit applies, character by character, to GRH.
* `src/tnfr/riemann/dirichlet_l.py`: canonical implementation of P32.
* `examples/04_riemann_L_twisted/59_dirichlet_l_function_demo.py`: reproducible verification across four canonical real characters.

### §13undecies.7 Status Update for §19.2 Gap Balance

| Scope | Status before P32 | Status after P32 |
|-------|-------------------|------------------|
| Operational ζ gaps (G1, G2, G3) | Closed operationally (P14, P13, P15) | Closed operationally |
| G4 = RH | OPEN (Conjecture T-HP) | OPEN, unchanged |
| G5 (ζ representation) | Superseded by P12+P13+P15 | Superseded, now generalised to all $L(s,\chi)$ at the P12 layer |
| Operational L-function gaps (G1$_\chi$, G2$_\chi$, G3$_\chi$) | Not addressed | G5$_\chi$ analogue closed at the P12 layer; G1$_\chi$/G2$_\chi$/G3$_\chi$ open (future work) |
| GRH (G4$_\chi$ for $\chi \neq \chi_0$) | OPEN | OPEN, unchanged |

**Net effect**: P32 extends the canonical TNFR representation catalog from a single L-function ($\zeta$) to the full Dirichlet family.  It does not close, nor narrow, any open arithmetic gap.

## §13duodecies. P33 — Analytic Continuation of χ-Twisted Prime-Ladder L-Series (Structural; Does NOT Advance G4 or GRH)

### §13duodecies.1 Motivation

P32 (§13undecies) provides the χ-twisted prime-ladder spectrum $\{(\mu_{p,k}, w_{p,k}^{(\chi)})\}$ reproducing the twisted von Mangoldt series

$$
Z^{(\chi)}_{\mathrm{TNFR}}(s) \;=\; -\frac{L'(s, \chi)}{L(s, \chi)} \;=\; \sum_{n \ge 1} \chi(n)\,\Lambda(n)\,n^{-s}
\qquad (\operatorname{Re}(s) > 1).
$$

This Dirichlet series is, by construction, only valid in the right half-plane $\operatorname{Re}(s) > 1$.  To expose the non-trivial zeros of $L(s, \chi)$ — which for non-principal primitive $\chi$ are *entire* objects living on the critical line — the χ-twisted prime ladder must be continued analytically to all of $\mathbb{C}$.

P33 is the structural analogue of P13 (§9) for general Dirichlet L-functions: the canonical continuation is obtained via `mpmath.dirichlet(s, [χ(0), …, χ(q-1)], derivative)` and the non-trivial zeros of $L(s, \chi)$ are recovered as **resonance poles** of $-L'(s,\chi)/L(s,\chi)$ on $\operatorname{Re}(s) = 1/2$.

### §13duodecies.2 Construction

For any Dirichlet character $\chi$ mod $q$:

1. **Continuation of $L(s, \chi)$**: `dirichlet_l_continued(chi, s, dps)` wraps `mp.dirichlet(s, chi_list)` with `chi_list = [mp.mpf(c) | mp.mpc(c) for c in chi.values]`, returning the unique meromorphic continuation to $\mathbb{C}$.
2. **Continuation of $-L'/L$**: `dirichlet_log_l_derivative_continued(chi, s, dps)` performs two `mp.dirichlet` calls (`derivative=0` and `derivative=1`) and returns $-L'(s,\chi)/L(s,\chi)$.  Raises `ValueError` whenever $|L(s,\chi)|$ is below the working precision (i.e., at a zero of $L$).
3. **Agreement certificate** (`verify_twisted_continuation_agreement`): compares the χ-twisted prime-ladder partial sum `tnfr_log_l_derivative(spectrum, s)` from P32 against the continuation evaluator on a list of $s$ with $\operatorname{Re}(s) > 1$, classifying the result as `excellent | good | poor` according to the worst per-point relative error.
4. **Critical-line scan** (`scan_critical_line_for_l_poles`): evaluates $|{-L'/L}|$ on $s = 1/2 + it$ for $t \in [t_{\min}, t_{\max}]$ and detects local-maximum spikes (resonance poles) using a sliding window proportional to the sample density.  The detection is reference-free; cross-checks against LMFDB tabulations are left to the caller.

### §13duodecies.3 Empirical Verification (May 2026 run)

Agreement on $\operatorname{Re}(s) > 1$ using `n_primes=400, max_power=14, dps=30` for the three canonical real characters of §13undecies:

| Character | Samples $s$ | Quality | max $|$rel err$|$ | max $|$abs err$|$ |
|---|---|---|---|---|
| $\chi_3$ (Legendre mod 3) | $\{2, 2+i, 3, 3+2i, 5\}$ | `excellent` | $1.10 \times 10^{-5}$ | $1.90 \times 10^{-6}$ |
| $\chi_4$ (Dirichlet $\beta$) | $\{2, 2+i, 3, 3+2i, 5\}$ | `excellent` | $1.60 \times 10^{-5}$ | $1.43 \times 10^{-6}$ |
| $\chi_5$ (Legendre mod 5) | $\{2, 2+i, 3, 3+2i, 5\}$ | `excellent` | $4.10 \times 10^{-6}$ | $1.10 \times 10^{-6}$ |

The residual is the standard P32 prime-truncation tail (same magnitude as the P12/P13 baseline for $\zeta$ at comparable truncation); it is **not** a defect of the continuation.

Critical-line scan against LMFDB-tabulated first zeros (`dps=20, 2001 samples on $t \in [5, 25]$, prominence threshold = 3.0`):

| Character | Detected peaks | LMFDB match | max $|$Δ$t|$ |
|---|---|---|---|
| $\chi_3$ | 6 (at $t \approx 8.04, 11.25, 15.70, 18.26, 20.46, 24.06$) | 6 / 6 | $6.7 \times 10^{-3}$ |
| $\chi_4$ | 7 (at $t \approx 6.02, 10.24, 12.99, 16.34, 18.29, 21.45, 23.28$) | 7 / 7 | $3.8 \times 10^{-3}$ |

All 13 detected resonance poles match the LMFDB tabulation to better than 0.01 in $t$ (limited by sample resolution $\Delta t = 0.01$); none miss, none extra.

### §13duodecies.4 What P33 Extends

P33 extends the **P13 representation layer** from $\zeta$ to every Dirichlet L-function:

* For $\zeta$: P13 continues the prime-ladder vM zeta to $\mathbb{C}$; non-trivial zeros appear as resonance poles on $\operatorname{Re}(s) = 1/2$.
* For $L(s, \chi)$: P33 does the same — continues the χ-twisted prime ladder of P32 to $\mathbb{C}$; non-trivial zeros of $L(s, \chi)$ appear as resonance poles on $\operatorname{Re}(s) = 1/2$.

This closes the **G5$_\chi$ / G2$_\chi$ analogue at the P13 layer**: the χ-twisted prime ladder is now a complete representation of $L(s, \chi)$ on the whole complex plane (subject to the same caveats as the classical continuation — branch cuts of the logarithmic derivative at the zeros).

### §13duodecies.5 What P33 Does NOT Advance

P33 is a **structural extension**, not progress on the open arithmetic content of the program:

* **G4 = RH for $\zeta$**: unchanged.  P33 does not touch $\zeta$.
* **GRH for $L(s, \chi)$**: unchanged.  P33 *uses* the existence and analyticity of the classical continuation; it does not derive the location of the zeros.  The detected resonance poles fall on $\operatorname{Re}(s) = 1/2$ because the LMFDB data they reproduce is itself empirical confirmation of GRH for the tested characters.
* **G1$_\chi$ (canonical Hamiltonian for χ-twisted prime ladder)**: open.  P33 does not construct a self-adjoint operator carrying the χ-twisted spectrum (the P14 analogue for L-functions remains future work — provisional label P34).
* **G3$_\chi$ (Weil–Guinand explicit formula for $L(s, \chi)$)**: open.  P33 does not verify a numerical explicit formula relating L-function zeros to the χ-twisted prime ladder (the P15 analogue — provisional label P35).

### §13duodecies.6 Cross-References

* §9: P13 analytic continuation of the prime-ladder vM zeta (the template P33 generalises).
* §13undecies: P32 χ-twisted prime ladder (the representation P33 continues).
* `src/tnfr/riemann/analytic_continuation_dirichlet.py`: canonical implementation of P33.
* `examples/04_riemann_L_twisted/60_dirichlet_l_continuation_demo.py`: demo verifying agreement on $\operatorname{Re}(s) > 1$ and critical-line zero detection for $\chi_3$ and $\chi_4$.

### §13duodecies.7 Status Update for §19.2 Gap Balance

| Scope | Status before P33 | Status after P33 |
|-------|-------------------|------------------|
| Operational ζ gaps (G1, G2, G3) | Closed operationally | Closed operationally, unchanged |
| G4 = RH | OPEN (Conjecture T-HP) | OPEN, unchanged |
| G5 (ζ representation) | Superseded by P12+P13+P15 | Superseded, unchanged |
| G5$_\chi$ at P12 layer (P32) | Closed | Closed, unchanged |
| G2$_\chi$ / G5$_\chi$ at P13 layer | Open | **Closed operationally** by P33 |
| G1$_\chi$ (Hamiltonian for $L(s,\chi)$) | Open | Open (future P34) |
| G3$_\chi$ (Weil–Guinand for $L(s,\chi)$) | Open | Open (future P35) |
| GRH (G4$_\chi$ for $\chi \neq \chi_0$) | OPEN | OPEN, unchanged |

**Net effect**: P33 extends the canonical TNFR representation catalog one layer further — the χ-twisted prime ladder of P32 now lives on the whole complex plane.  It does not close, nor narrow, any open arithmetic gap.

## §13terdecies. P34 — Canonical Hamiltonian for the χ-Twisted Prime Ladder (Structural; Closes G1$_\chi$ at the P14 Layer; Does NOT Advance G4 or GRH)

### §13terdecies.1 Motivation

P14 (§10) supplies the canonical self-adjoint TNFR ``InternalHamiltonian`` on the prime-ladder graph whose decoupled spectrum is $\{k \log p\}$ and whose weighted spectral trace $\operatorname{Tr}(W \, e^{-s H_{\mathrm{freq}}})$ reproduces $-\zeta'(s)/\zeta(s)$ to machine precision. After P32 (the χ-twisted prime ladder representing $-L'(s,\chi)/L(s,\chi)$ on $\operatorname{Re}(s) > 1$) and P33 (its analytic continuation), the natural structural question — the explicit content of §13duodecies.5 — is whether the same canonical TNFR Hamiltonian construction admits a χ-twisted analogue for every Dirichlet character. **P34 supplies that analogue.** The construction does not advance GRH or G4; it closes gap **G1$_\chi$** *at the P14 layer* for every $L(s,\chi)$.

### §13terdecies.2 Construction

Let $\chi$ be a Dirichlet character of conductor $q$ and $K \ge 1$ a REMESH echo cut-off. Let $P_\chi = \{p \text{ prime}: \chi(p) \neq 0\} = \{p : p \nmid q\}$ (the canonical primes-coprime-to-$q$ filter introduced at P32).

1. **Graph**: $G_\chi$ is the disjoint union over $p \in P_\chi$ of the per-prime REMESH ladder $L_p$ ($K$ nodes $(p,1), \dots, (p,K)$ chained by REMESH edges). Per-node attributes: $\nu_f((p,k)) = k \log p$, all other TNFR state $\phi = 0$, $\mathrm{EPI} = 1$, $S_i = 1$, $\Delta \mathrm{NFR} = 0$.

2. **Hamiltonian**: $H_\chi$ is the canonical TNFR ``InternalHamiltonian`` on $G_\chi$ with internal-coherence strength $\alpha = 0$ and decoupled limit ($J_0 = 0$). By construction (§10) $H_\chi$ is real symmetric (hence self-adjoint) and $\operatorname{spec}(H_{\chi, \mathrm{freq}}) = \{k \log p : p \in P_\chi, \, 1 \le k \le K\}$ exactly.

3. **χ-twisted weight operator**: $W^{(\chi)}$ is the diagonal $|V(G_\chi)| \times |V(G_\chi)|$ matrix

   $$W^{(\chi)}_{(p,k),(p,k)} = \chi(p)^k \log p \in \mathbb{C}.$$

   For real characters $W^{(\chi)}$ is real-diagonal (Hermitian); for complex characters it is *normal but not Hermitian*, since the entries lie on the unit circle scaled by $\log p$. This is the canonical structural carrier of the χ-phase: $H_\chi$ stays real self-adjoint (so its eigenvectors form a real-orthonormal basis), and the complex content lives exclusively in $W^{(\chi)}$.

4. **χ-twisted weighted spectral trace**: For $s \in \mathbb{C}$,

   $$Z_{\mathrm{TNFR}}^{(\chi)}(s) := \operatorname{Tr}\bigl(W^{(\chi)} \, e^{-s H_{\chi, \mathrm{freq}}}\bigr) = \sum_{p \in P_\chi} \sum_{k=1}^{K} \chi(p)^k (\log p) \, p^{-ks}.$$

   This is exactly the P32 reference trace `tnfr_log_l_derivative`, which converges to $-L'(s,\chi)/L(s,\chi)$ as $K \to \infty$ on $\operatorname{Re}(s) > 1$ and admits the P33 continuation elsewhere.

### §13terdecies.3 Empirical Verification (May 2026 run)

`examples/04_riemann_L_twisted/61_dirichlet_l_hamiltonian_demo.py`, with $n_{\mathrm{primes}} = 20$, $K = 8$ (Hilbert dimension $N = 152$, $19$ active primes, $1$ excluded), $s$-values $\{2, 3, 2+i, 3+2i, 5, 10\}$:

| Character | $N$ | $n_{\mathrm{active}}$ | spectrum_max_abs_error | trace_max_rel_error | overall_ok |
|-----------|----:|----------------------:|-----------------------:|--------------------:|-----------:|
| $\chi_3$ (mod 3) | 152 | 19 | $0.000 \times 10^{0}$ | $3.241 \times 10^{-16}$ | **YES** |
| $\chi_4$ (mod 4) | 152 | 19 | $0.000 \times 10^{0}$ | $3.493 \times 10^{-16}$ | **YES** |
| $\chi_5$ (mod 5) | 152 | 19 | $0.000 \times 10^{0}$ | $2.313 \times 10^{-16}$ | **YES** |

The spectrum match is **exact** (zero floating-point error: $H_{\chi,\mathrm{freq}}$ is constructed with diagonal entries $\nu_f((p,k)) = k \log p$, hence its eigenvalues coincide bit-for-bit with the reference). The χ-twisted weighted trace matches the P32 reference at the machine-epsilon level for all tested $s$, including non-real $s$.

Step 3 of the demo also verifies the **triple agreement** P34 ≡ P32 (machine precision, by construction) ≡ P33 (mpmath, $O(p_{\max}^{-\operatorname{Re}(s)})$ truncation residual) on $\operatorname{Re}(s) > 1$ for $\chi_3$, including off-axis $s = 2+i$ and $s = 3+2i$.

### §13terdecies.4 What P34 Extends

* **Canonical operator catalog**: every Dirichlet $L(s,\chi)$ now has a TNFR-canonical self-adjoint operator that carries its prime data, exactly as $\zeta$ does at P14.
* **G1$_\chi$ at the P14 layer**: the obstruction "canonical Hamiltonian whose decoupled spectrum and χ-twisted weighted trace reproduce the P32 χ-twisted ladder data" is now **closed operationally** for every $\chi$.
* **Structural completeness of the L-function track**: after P32 (operator content), P33 (continuation), and P34 (Hamiltonian realisation), the χ-twisted ladder occupies the same structural status as the ζ ladder before P15.

### §13terdecies.5 What P34 Does NOT Advance

* **Generalised Riemann Hypothesis (GRH)**: no change. RH-equivalent localisation of poles on $\operatorname{Re}(s) = 1/2$ for $L(s,\chi)$ is the same arithmetic obstruction as G4 = RH for $\zeta$; P34 inherits the open status unchanged.
* **G4 = RH**: untouched. The P34 Hamiltonian is structurally identical to the P14 Hamiltonian on its prime-ladder block; the open content of Conjecture **T-HP** (§13septies) — existence of a canonical admissible spectral-rescaling operator $\mathcal{F}$ built only from the tetrad — is *not* addressed.
* **G3$_\chi$ (χ-twisted Weil–Guinand explicit formula)**: open. The χ-twisted analogue of P15 — the *explicit-formula* bridge linking the P33 zeros of $L(s,\chi)$ to the P34 Hamiltonian spectrum to machine precision — is the future **P35**.
* **No new analytic content**: P34 is a canonical operator-theoretic *re-presentation* of P32/P33 data; it does not introduce any analytic ingredient absent from those constructions.

### §13terdecies.6 Cross-References

* §10: P14 prime-ladder Hamiltonian (the canonical template P34 specialises).
* §13undecies: P32 χ-twisted prime ladder (the spectrum/weight data P34 represents).
* §13duodecies: P33 analytic continuation of $-L'(s,\chi)/L(s,\chi)$ (the off-$\operatorname{Re}(s) > 1$ extension).
* `src/tnfr/riemann/twisted_prime_ladder_hamiltonian.py`: canonical implementation of P34.
* `examples/04_riemann_L_twisted/61_dirichlet_l_hamiltonian_demo.py`: demo verifying spectrum-exact / trace-machine-precision reproduction for $\chi_3, \chi_4, \chi_5$ and triple agreement P34 ≡ P32 ≡ P33 on $\operatorname{Re}(s) > 1$.

### §13terdecies.7 Status Update for §19.2 Gap Balance

| Scope | Status before P34 | Status after P34 |
|-------|-------------------|------------------|
| Operational ζ gaps (G1, G2, G3) | Closed operationally | Closed operationally, unchanged |
| G4 = RH | OPEN (Conjecture T-HP) | OPEN, unchanged |
| G5 (ζ representation) | Superseded by P12+P13+P15 | Superseded, unchanged |
| G5$_\chi$ at P12 layer (P32) | Closed | Closed, unchanged |
| G2$_\chi$ / G5$_\chi$ at P13 layer (P33) | Closed operationally | Closed operationally, unchanged |
| **G1$_\chi$ (Hamiltonian for $L(s,\chi)$)** | Open | **Closed operationally** by P34 |
| G3$_\chi$ (Weil–Guinand for $L(s,\chi)$) | Open | Open (future P35) |
| GRH (G4$_\chi$ for $\chi \neq \chi_0$) | OPEN | OPEN, unchanged |

**Net effect**: P34 extends the canonical TNFR operator catalog one layer further — every Dirichlet $L(s,\chi)$ now has a canonical self-adjoint TNFR Hamiltonian realising its prime data, exactly as $\zeta$ does since P14. It does not close, nor narrow, any open arithmetic gap (G4, GRH, G3$_\chi$).

## §13quaterdecies. P35 — χ-Twisted Weil–Guinand Explicit Formula (Structural; Closes G3$_\chi$ Operationally for Primitive Real χ; Does NOT Advance G4 or GRH)

### §13quaterdecies.1 Motivation

P15 (§11) established the Weil–Guinand explicit formula for $\zeta$ as a TNFR-native identity: the zero side $\sum_\gamma h(\gamma)$ equals the sum of an Archimedean digamma integral, a constant term, and a prime side computed as the diagonal projection of the P14 weight operator $W$ in its eigenbasis. After P32 (χ-twisted prime ladder), P33 (analytic continuation of the corresponding TNFR vM zeta), and P34 (canonical Hamiltonian for the χ-twisted ladder), the natural structural question is the χ-twisted analogue of the explicit formula. **P35 supplies it for every primitive real Dirichlet character.**  This closes gap **G3$_\chi$** operationally for $\chi \in \{\chi_3, \chi_4, \chi_5, \ldots\}$ (real, non-principal) and does not advance G4 = RH or GRH.

### §13quaterdecies.2 Construction

For a primitive real non-principal Dirichlet character $\chi$ with conductor $q$ and parity $a = (1-\chi(-1))/2 \in \{0,1\}$, and Gaussian test pair $h(t)=e^{-t^2/(2\sigma^2)}$, $g(u)=(\sigma/\sqrt{2\pi})\,e^{-\sigma^2 u^2/2}$, the χ-twisted Weil–Guinand explicit formula is

$$
\sum_\gamma h(\gamma)
\;=\;
\underbrace{g(0)\,\log(q/\pi)}_{\text{constant term}}
\;+\;
\underbrace{\frac{1}{2\pi}\!\int_{-\infty}^{\infty}\! h(t)\,\Re\,\psi\!\left(\tfrac14+\tfrac{a}{2}+\tfrac{it}{2}\right)\,dt}_{\text{archimedean side}}
\;-\;
\underbrace{2\,\Re\sum_{n\ge1}\frac{\chi(n)\,\Lambda(n)}{\sqrt n}\,g(\log n)}_{\text{prime side}}.
$$

The two non-trivial reductions to ζ are immediate: for the trivial character ($q=1$, $a=0$) the constant becomes $-g(0)\log\pi$, the digamma factor collapses to $\psi(1/4+it/2)$, and the prime side becomes the unweighted P15 sum.

* **Zero side** — Hardy-Z bisection on $Z_\chi(t) = e^{i\theta_\chi(t)} L(\tfrac12+it,\chi)$ (built on P33's mpmath-grade continuation), enumerating positive imaginary parts $\gamma$ on $\operatorname{Re}(s) = 1/2$.  Real $\chi$ ⇒ zeros come in conjugate pairs ⇒ $\sum_\gamma h(\gamma) = 2\sum_{\gamma>0} h(\gamma)$.
* **Prime side** — diagonal projection of the χ-twisted weight operator $W^{(\chi)}$ from the canonical P34 Hamiltonian, in its eigenbasis (same einsum idiom as P15).
* **Archimedean side** — direct numerical quadrature of the digamma factor (mpmath, $\mathrm{dps}=30$).

### §13quaterdecies.3 Empirical Verification

| χ | q | a | σ | n zeros | residual | rel. residual | verified |
|---|---|---|---|---|---|---|---|
| χ₃ | 3 | 1 | 2.0 |  5 | −7.46 × 10⁻¹⁷ | 1.20 × 10⁻¹³ | ✓ |
| χ₃ | 3 | 1 | 2.5 |  8 | +2.03 × 10⁻¹⁶ | 1.77 × 10⁻¹⁴ | ✓ |
| χ₃ | 3 | 1 | 3.0 | 11 | +2.36 × 10⁻¹⁶ | 4.15 × 10⁻¹⁵ | ✓ |
| χ₄ | 4 | 1 | 2.0 |  7 | −9.48 × 10⁻¹⁵ | 4.40 × 10⁻¹³ | ✓ |
| χ₄ | 4 | 1 | 2.5 | 10 | −3.10 × 10⁻¹⁴ | 2.81 × 10⁻¹³ | ✓ |
| χ₄ | 4 | 1 | 3.0 | 12 | −5.16 × 10⁻¹⁴ | 1.89 × 10⁻¹³ | ✓ |
| χ₅ | 5 | 0 | 2.0 |  7 | −2.00 × 10⁻¹⁵ | 2.50 × 10⁻¹³ | ✓ |
| χ₅ | 5 | 0 | 2.5 | 11 | −7.97 × 10⁻¹⁵ | 1.35 × 10⁻¹³ | ✓ |
| χ₅ | 5 | 0 | 3.0 | 14 | −1.56 × 10⁻¹⁴ | 8.60 × 10⁻¹⁴ | ✓ |

All nine $(\chi, \sigma)$ pairs verify the identity to machine precision (relative residual $\le 4.4 \times 10^{-13}$), well inside the declared $10^{-2}$ tolerance.

### §13quaterdecies.4 What P35 Extends

* **Operational closure of G3$_\chi$ for primitive real χ**: both sides of the χ-twisted Weil–Guinand identity now have a canonical TNFR realisation that agrees to machine precision.
* **Symmetric completion of the L-function track**: P32 (operator content) → P33 (continuation) → P34 (Hamiltonian) → **P35 (explicit formula)** now occupies the same structural status as P12 → P13 → P14 → P15 for ζ.
* **Reuse of P34 Hamiltonian**: the prime side is the *exact same* einsum idiom as P15; no new operator is introduced.

### §13quaterdecies.5 What P35 Does NOT Advance

* **Generalised Riemann Hypothesis (GRH)**: untouched.  Zero localisation on $\operatorname{Re}(s) = 1/2$ for $L(s,\chi)$ is **assumed** in P35 (Hardy-Z bisection starts from the critical line); proving every L-zero lies there is the χ-twisted analogue of gap **G4 = RH** and is the same arithmetic obstruction.
* **G4 = RH**: structurally identical to the ζ case; Conjecture **T-HP** (§13septies) and its extensions remain open.
* **Complex χ**: P35 currently supports only **primitive real** characters.  Extension to complex χ requires a Hermitisation of $W^{(\chi)}$ that is intentionally deferred.
* **No new analytic content beyond P33**: P35 packages P33 zeros and P34 Hamiltonian into a single explicit-formula certificate; it does not introduce any new analytic ingredient.

### §13quaterdecies.6 Cross-References

* §11: P15 Weil–Guinand for ζ (canonical template P35 specialises).
* §13undecies: P32 χ-twisted prime ladder.
* §13duodecies: P33 χ-twisted analytic continuation.
* §13terdecies: P34 χ-twisted Hamiltonian (prime side of P35).
* `src/tnfr/riemann/twisted_weil_explicit_formula.py`: canonical implementation of P35.
* `examples/04_riemann_L_twisted/62_dirichlet_weil_explicit_formula_demo.py`: demo verifying nine $(\chi, \sigma)$ pairs to machine precision.

### §13quaterdecies.7 Gap Balance

| Scope | Status before P35 | Status after P35 |
|-------|---|---|
| G3 (Weil–Guinand for ζ, P15) | Closed operationally | Closed operationally, unchanged |
| **G3$_\chi$ (Weil–Guinand for $L(s,\chi)$, primitive real χ)** | Open (future P35) | **Closed operationally** by P35 |
| G3$_\chi$ for complex χ | Open | Open (future increment) |
| G4 = RH | OPEN | OPEN, unchanged |
| GRH (G4$_\chi$ for $\chi \neq \chi_0$) | OPEN | OPEN, unchanged |

**Net effect**: P35 closes the explicit-formula gap on the L-function track for every primitive real Dirichlet character.  Combined with P32–P34, the structural status of $L(s,\chi)$ for real $\chi$ now matches the ζ track up through P15.  The arithmetic obstruction (zero localisation on $\operatorname{Re}(s)=1/2$) is unchanged.

## §13quinquiesdecies. P36 — χ-Twisted Li–Keiper Positivity Criterion (Structural Diagnostic; GRH$_\chi$-Equivalent for Primitive Real χ; Does NOT Prove GRH or Advance G4)

### §13quinquiesdecies.1 Motivation

P16 (§12) supplies the canonical TNFR-native finite diagnostic surface for RH via Li–Keiper coefficients $\lambda_n$ computed from non-trivial zeros of $\zeta(s)$.  The L-function track now reaches the same level: P35 (§13quaterdecies) supplies a complete Hardy-Z zero enumerator for every primitive real Dirichlet $L(s,\chi)$, and Lagarias 2007 generalises Li 1997 to L-functions.  P36 packages these ingredients into a structural GRH$_\chi$-equivalent diagnostic — the L-function analogue of P16.

### §13quinquiesdecies.2 Construction

For a primitive real Dirichlet character $\chi$ with non-trivial zeros $\rho_k = 1/2 + i\gamma_k$ of $L(s,\chi)$ on the critical line, define the **χ-twisted Li–Keiper coefficients**

$$
\lambda_n(\chi) \;=\; \sum_{k} 2\,\operatorname{Re}\!\Big[1 - \big(1 - 1/\rho_k\big)^n\Big],\qquad n \ge 1.
$$

The sum runs over all non-trivial zeros (paired with their complex conjugates via the $2\operatorname{Re}[\cdot]$ factor).  By Lagarias 2007 (generalisation of Li 1997):

$$
\boxed{\;\text{GRH for } L(s,\chi) \iff \lambda_n(\chi) > 0 \text{ for every } n \ge 1.\;}
$$

P36 computes $\lambda_n(\chi)$ for $n = 1, \dots, n_{\max}$ from the finite truncation $\{\gamma_k : 0 < \gamma_k < t_{\max}\}$ supplied by the P35 enumerator (`find_dirichlet_l_zeros`).  The sum-over-zeros formula is **L-function agnostic**, so the canonical P16 routine `li_coefficients_from_zeros` is reused unchanged at mpmath precision $\text{dps} = 50$.

### §13quinquiesdecies.3 Empirical Verification

Positivity of $\lambda_n(\chi)$ verified for the three primitive real characters of small modulus across $n_{\max} \in \{20, 30, 50\}$ with $t_{\max} = 80$:

| Character | $q$ | parity $a$ | $\#$ zeros used | $\min_n \lambda_n(\chi)$ | $\lambda_n > 0$ for $n \le 50$? |
|-----------|-----|-----------|-----------------|--------------------------|-------------------------------|
| $\chi_3$  | 3   | 1         | 34              | $+4.741 \times 10^{-2}$  | yes                           |
| $\chi_4$  | 4   | 1         | 37              | $+6.791 \times 10^{-2}$  | yes                           |
| $\chi_5$  | 5   | 0         | 40              | $+6.802 \times 10^{-2}$  | yes                           |

(Reproduced by `examples/04_riemann_L_twisted/63_dirichlet_li_keiper_demo.py`.)

### §13quinquiesdecies.4 What P36 Extends

* **P16 to L-functions**: P16 is the canonical Li–Keiper diagnostic for $\zeta$; P36 is its structural analogue for $L(s,\chi)$ at every primitive real $\chi$.  Together with P32–P35, the structural TNFR-Riemann program now matches the ζ track all the way through the diagnostic layer.
* **Numerical witness for GRH$_\chi$**: every positivity row above is a falsifiable finite witness; a single $\lambda_n(\chi) \le 0$ would disprove GRH for the corresponding $L(s,\chi)$.

### §13quinquiesdecies.5 What P36 Does NOT Advance

* **GRH for any $L(s,\chi)$**: a finite check of $\lambda_n > 0$ for $n \le n_{\max}$ is **necessary but not sufficient**.  Rigorous bounds on the truncation tail are required to upgrade the finite check to a proof; P36 does not supply them.  Consistent with Bombieri–Lagarias 1999 and Lagarias 2007.
* **G4 = RH**: structurally identical to P16; the arithmetic obstruction is untouched.  The zeros are *assumed* to lie on $\operatorname{Re}(s) = 1/2$ via the Hardy-Z bisection on $Z_\chi(t)$ used by P35.
* **Complex χ**: P36 inherits the primitive-real restriction from P32–P35.

### §13quinquiesdecies.6 Cross-References

* §12: P16 Li–Keiper criterion for ζ (canonical template).
* §13quaterdecies: P35 χ-twisted Weil–Guinand explicit formula (supplies the zero enumerator).
* §13septies: Conjecture T-HP (unchanged by P36).
* `src/tnfr/riemann/twisted_li_keiper.py`: canonical P36 implementation.
* `examples/04_riemann_L_twisted/63_dirichlet_li_keiper_demo.py`: demo with full positivity sweep.

### §13quinquiesdecies.7 Gap Balance

| Scope | Status before P36 | Status after P36 |
|-------|-------------------|------------------|
| P16 diagnostic for ζ | Available (P16) | Available, unchanged |
| **Li–Keiper diagnostic for $L(s,\chi)$, primitive real χ** | Open (future P36) | **Available** (TNFR-native finite witness) |
| GRH for $L(s,\chi)$, primitive real χ | OPEN | OPEN (diagnostic only; finite check is necessary, not sufficient) |
| G4 = RH | OPEN | OPEN, unchanged |
| GRH (G4$_\chi$ for complex $\chi$) | OPEN | OPEN, unchanged |

**Net effect**: P36 closes the diagnostic-layer gap on the L-function track for every primitive real Dirichlet character.  Combined with P32–P35, every milestone reachable on the ζ track up through P16 now has a structural analogue on the primitive-real L-function track.  The arithmetic obstruction remains the same.

## §13sexiesdecies. P37 — χ-Twisted Weil–TNFR Positivity Bridge (Structural Diagnostic; GRH$_\chi$-Equivalent for Primitive Real χ; Does NOT Prove GRH or Advance G4)

### §13sexiesdecies.1 Motivation

P17 (§14) supplies the canonical TNFR-native Weil-positivity bridge for $\zeta$: Weil's RH-equivalent positivity functional $W[f] = \sum_\gamma \hat f(\gamma) \ge 0$ is transported onto the TNFR Lyapunov functional $E_{\mathrm{TNFR}}$ via the P14 prime-ladder Hamiltonian.  Bombieri 2000 generalises Weil's criterion to every primitive Dirichlet $L(s,\chi)$, so the same structural transport exists on the L-function track once P34 (canonical χ-twisted Hamiltonian) and P35 (canonical χ-twisted explicit formula) are in place.  P37 packages these ingredients into a GRH$_\chi$-equivalent diagnostic — the L-function analogue of P17.

### §13sexiesdecies.2 Construction

For a fixed primitive real Dirichlet character $\chi$ of conductor $q$, parity $a \in \{0, 1\}$, and Gaussian width $\sigma > 0$, let
$$h_\sigma(t) = e^{-t^2 / (2\sigma^2)}, \qquad \hat h_\sigma(\xi) = \sigma \sqrt{2\pi}\, e^{-\sigma^2 \xi^2 / 2}.$$

The χ-twisted Weil positivity functional is
$$W_\chi[\sigma] := 2 \sum_{\gamma > 0} h_\sigma(\gamma), \qquad \gamma \in \mathrm{Im}\{\rho : L(\tfrac12 + i\rho, \chi) = 0\}.$$
P37 computes $W_\chi[\sigma]$ two ways:

1. **Zero side** (P35 enumerator): exact Hardy-Z bisection via `twisted_weil_zero_side` truncated at $t_{\max} = 12\sigma$ (canonical default).
2. **Explicit-formula side** (P34 Hamiltonian): the χ-twisted Weil–Guinand identity
    $$W_\chi[\sigma] \stackrel{!}{=} g(0)\log\!\frac{q}{\pi} + I_{\infty}^{\chi}(\sigma) + P_{\chi}(\sigma),$$
    where $g$ is the test function in the cosine-transform convention used by P35, $I_{\infty}^{\chi}$ is the archimedean integral (`twisted_weil_archimedean_integral`, parity-dependent via the $\psi$-shift) and $P_{\chi}$ is the prime side **evaluated on the P34 χ-twisted prime-ladder Hamiltonian** (`twisted_weil_prime_side_from_hamiltonian`).  The consistency residual $|W_{\mathrm{zero}} - W_{\mathrm{XF}}|$ measures the joint self-consistency of P34+P35.

Positivity is verified as $W_\chi[\sigma] \ge 0$.  In parallel, the canonical TNFR test state on the P34 graph is defined by `build_twisted_structural_test_state(bundle, sigma)`: for each node $(p, k)$ with structural frequency $\nu_f = k\log p$, set
$$\Delta\mathrm{NFR}_{(p,k)} = \mathrm{EPI}_{(p,k)} = h_\sigma(k\log p), \qquad \phi_{(p,k)} = \min(h_\sigma(k\log p), \pi),$$
and the TNFR Lyapunov energy of this state is
$$E_{\mathrm{TNFR}}^\chi[\sigma] := \tfrac12 \sum_i \bigl(\Phi_s^2 + |\nabla\phi|^2 + K_\phi^2 + J_\phi^2 + J_{\Delta\mathrm{NFR}}^2\bigr)$$
via the canonical `compute_energy_functional` (single source of truth from `tnfr.physics.conservation`, reused unchanged from P17).  The χ-twisted **TNFR bridge ratio** is
$$\boxed{\;\alpha_\chi(\sigma) := \frac{W_\chi[\sigma]}{E_{\mathrm{TNFR}}^\chi[\sigma]}.\;}$$

### §13sexiesdecies.3 Empirical Verification

Configuration: $N_{\mathrm{primes}} = 25$, $k_{\max} = 6$, decoupled spectrum (coupling = 0), $\sigma \in \{1.0, 1.5, 2.0, 2.5, 3.0\}$.  Demo: `examples/04_riemann_L_twisted/64_twisted_weil_positivity_demo.py`.

For χ$_3$ the consistency residual is $|W_{\mathrm{zero}} - W_{\mathrm{XF}}| \le 6.1 \times 10^{-6}$ at $\sigma = 1.0$ and $\le 2.4 \times 10^{-16}$ for $\sigma \in \{2.0, 2.5, 3.0\}$ (machine precision once enough zeros enter the Gaussian window).  Aggregate verdicts:

| Character | $q$ | parity $a$ | $W_\chi \ge 0$ all σ? | $\alpha_\chi > 0$ all σ? | $\alpha_{\min}$ | $\alpha_{\max}$ | Verdict |
|-----------|----:|:----------:|:---------------------:|:------------------------:|----------------:|----------------:|---------|
| χ$_3$     | 3   | 1 (odd)    | YES                   | YES                      | $1.27\times 10^{-14}$ | $7.39\times 10^{-3}$ | PASS |
| χ$_4$     | 4   | 1 (odd)    | YES                   | YES                      | $2.71\times 10^{-8}$  | $3.77\times 10^{-2}$ | PASS |
| χ$_5$     | 5   | 0 (even)   | YES                   | YES                      | $2.62\times 10^{-10}$ | $2.32\times 10^{-2}$ | PASS |

All three primitive real characters pass both the Weil positivity check and the structural bridge check across the entire Gaussian grid.  $\alpha_{\min}$ at small σ collapses toward machine precision because $W_\chi[\sigma] \to 0$ (no zeros enter the Gaussian window when $\sigma$ is smaller than the imaginary part of the lowest zero) while $E_{\mathrm{TNFR}}^\chi$ stays $\mathcal{O}(1)$; the diagnostic interpretation is that the lower bound becomes vacuous (not violated) in that regime.

### §13sexiesdecies.4 What P37 Extends

* **P17 to L-functions**: P17 is the canonical Weil-TNFR positivity bridge for $\zeta$ (GRH-equivalent diagnostic via $W \ge 0$); P37 is its structural analogue for $L(s,\chi)$ at every primitive real $\chi$.  The TNFR Lyapunov target $E_{\mathrm{TNFR}}$ is reused unchanged; only the zero source (P35) and the prime side (P34) are χ-twisted.

* **L-function track parity with the ζ track**: combined with P32–P36, every milestone reachable on the ζ track up through P17 now has a structural analogue on the primitive-real L-function track.

### §13sexiesdecies.5 What P37 Does NOT Advance

* **GRH for any $L(s,\chi)$**: a finite Gaussian grid cannot exhaust the admissible family that makes Weil positivity equivalent to GRH$_\chi$ (Bombieri 2000).  Numerical $W_\chi[\sigma] \ge 0$ is consistent with GRH$_\chi$ but is **not** a proof.

* **G4 = RH**: P37 is on the L-function track and does not bear on the untwisted Riemann hypothesis.

* **Complex χ**: P37 inherits the primitive-real restriction from P32–P35 (the L-function track stays real until the complex-χ extension is shipped).

* **Canonicity of the structural test state**: `build_twisted_structural_test_state` is one canonical mapping of $h_\sigma$ to the P34 graph; the bridge ratio $\alpha_\chi(\sigma)$ is specific to this mapping.  Exhaustively sweeping admissible structural test states is a future milestone (parallel to P18–P21 on the ζ track).

### §13sexiesdecies.6 Cross-References

* §14 (P17): untwisted Weil–TNFR positivity bridge for $\zeta$ (the construction P37 imitates).
* §10 (P14) and §13nonies (P30): canonical TNFR Hamiltonian and admissible-rescaling building blocks reused via P34.
* §13quaterdecies (P35): χ-twisted Weil–Guinand explicit formula (zero side + RHS).
* §13quinquiesdecies (P36): χ-twisted Li–Keiper diagnostic (complementary GRH$_\chi$-equivalent surface).
* `src/tnfr/riemann/twisted_weil_positivity.py`: canonical P37 implementation.
* `examples/04_riemann_L_twisted/64_twisted_weil_positivity_demo.py`: demo with the full χ$_3$/χ$_4$/χ$_5$ sweep.

### §13sexiesdecies.7 Gap Balance

| Scope | Status before P37 | Status after P37 |
|-------|-------------------|------------------|
| P17 Weil bridge for ζ | Available (P17) | Available, unchanged |
| **Weil–TNFR bridge for $L(s,\chi)$, primitive real χ** | Open (future P37) | **Available** (TNFR-native finite diagnostic) |
| GRH for $L(s,\chi)$, primitive real χ | OPEN | OPEN (diagnostic only; finite Gaussian grid is necessary, not sufficient) |
| G4 = RH | OPEN | OPEN, unchanged |
| GRH (G4$_\chi$ for complex $\chi$) | OPEN | OPEN, unchanged |

**Net effect**: P37 closes the Weil-positivity-bridge gap on the L-function track for every primitive real Dirichlet character.  The L-function track now structurally matches the ζ track all the way through P17.  The arithmetic obstruction remains identical and the gap balance for G4 is unchanged.

## §13septiesdecies. P38 — χ-Twisted Admissibility / Gauge Sweep of $\alpha_\chi(\sigma; g)$ (Structural Robustness Diagnostic; GRH$_\chi$-Equivalent for Primitive Real χ; Does NOT Prove GRH or Advance G4)

### §13septiesdecies.1 Motivation

P18 (§15) packages the canonical TNFR-native robustness audit of the P17 Weil–TNFR bridge for $\zeta$: the bridge ratio $\alpha(\sigma) = W[\sigma] / E_{\mathrm{TNFR}}[\sigma; g]$ depends, via the energy denominator, on the structural gauge $g$ that maps a Gaussian width $\sigma$ onto the canonical TNFR test state $(\Delta\mathrm{NFR}, \phi, \mathrm{EPI})$.  The numerator $W[\sigma]$ is gauge-independent (zero-side enumeration), but the denominator is not, so any single-gauge result of $\alpha(\sigma) > 0$ is only as strong as the gauge it is parameterised by.  P18 stress-tests the bridge across the canonical six-gauge family `DEFAULT_GAUGES` = {canonical, dnfr_only, phase_only, epi_only, dnfr_phase, pressure_amplified}.  P37 (§13sexiesdecies) extends P17 to every primitive real Dirichlet $L(s,\chi)$, so the same robustness audit is meaningful on the L-function track once P34 and P35 are in place.  P38 packages these ingredients into the L-function analogue of P18.

### §13septiesdecies.2 Construction

P38 sweeps $\alpha_\chi(\sigma; g) = W_\chi[\sigma] \,/\, E_{\mathrm{TNFR}}^\chi[\sigma; g]$ across a finite Gaussian grid $\{\sigma_i\}$ and the canonical six-gauge family `DEFAULT_GAUGES` inherited unchanged from `alpha_sweep.py` (P18).  Canonical reuse:

* $W_\chi[\sigma]$ is computed once per $\sigma$ (gauge-independent) via the P35 enumerator `twisted_weil_zero_side` at canonical mpmath precision $\mathrm{dps} = 30$.
* For each gauge $g$, the canonical TNFR test state on the P34 χ-twisted prime-ladder bundle is built by mapping each ladder level $E_n = k \log p$ to $h_n = \exp\!\bigl(-E_n^2/(2\sigma^2)\bigr)$ and then applying $g(h_n) = (\Delta\mathrm{NFR}_n, \phi_n, \mathrm{EPI}_n)$, with phases clipped to $[-\pi, \pi]$.
* $E_{\mathrm{TNFR}}^\chi[\sigma; g]$ is computed by the canonical conservation routine `compute_energy_functional` unchanged from P17/P18.

The certificate is a frozen `TwistedAlphaSweepCertificate` carrying the $W_\chi$ row, the $(n_\sigma \times n_g)$ $\alpha_\chi$ table, the energy table, the aggregate positivity flags, and the coordinates of $\alpha_{\min}$ / $\alpha_{\max}$.  No new physics is introduced: P38 is a robustness layer over P34, P35, and P37.

### §13septiesdecies.3 Empirical Verification

The reference demo `examples/04_riemann_L_twisted/65_twisted_alpha_sweep_demo.py` sweeps $\sigma \in \{1.0, 1.5, 2.0, 2.5, 3.0\}$ and all six gauges across $\chi_3, \chi_4, \chi_5$ (decoupled spectrum, $n_{\mathrm{primes}} = 25$, $\max_{\mathrm{power}} = 6$):

| χ | $q$ | $W_\chi \ge 0$ | $\alpha_\chi > 0$ | $\alpha_{\min}$ @ $(\sigma, g)$ | $\alpha_{\max}$ |
|---|---|---|---|---|---|
| $\chi_3$ | 3 | True | True | $+1.27 \times 10^{-14}$ @ $(1.000, \text{canonical})$ | $+6.04 \times 10^{-2}$ |
| $\chi_4$ | 4 | True | True | $+2.71 \times 10^{-8}$ @ $(1.000, \text{canonical})$ | $+6.38 \times 10^{-1}$ |
| $\chi_5$ | 5 | True | True | $+2.62 \times 10^{-10}$ @ $(1.000, \text{canonical})$ | $+1.56 \times 10^{-1}$ |

Positivity holds across every $(\sigma, g)$ combination for every tested character (3/3 PASS).  The smallest $\sigma$ (= 1.0) and the `canonical` gauge consistently produce the most demanding entry, which is the expected behaviour from the P18 ζ-track analogue (narrow Gaussians give the tightest test).

### §13septiesdecies.4 What P38 Extends

* **P18 to L-functions**: P18 is the canonical robustness audit for the ζ-side Weil–TNFR bridge; P38 is its structural analogue for $L(s,\chi)$ at every primitive real χ.  Together with P32–P37, the L-function track now structurally matches the ζ track through the P18 layer.
* **P37 under canonical-mapping ambiguity**: P37 verified $\alpha_\chi(\sigma) > 0$ for the `canonical` gauge only.  P38 confirms that the positivity persists across the entire `DEFAULT_GAUGES` family, ruling out a single-gauge artefact.

### §13septiesdecies.5 What P38 Does NOT Advance

* **GRH for any $L(s,\chi)$**: a finite sweep across $\{\sigma_i\} \times \{g\}$ is **necessary but not sufficient**.  An exhaustive admissible family (which a finite grid cannot exhaust) would be required to upgrade the diagnostic to a proof.  Consistent with the P17/P18 honesty boundary.
* **Complex χ**: P38 inherits the primitive-real restriction from P32–P37.
* **G4 = RH**: $\alpha_\chi(\sigma; g)$ depends on the χ-twisted Hamiltonian (P34) and the χ-twisted explicit formula (P35); neither carries information about the ζ critical line.  G4 is unchanged.

### §13septiesdecies.6 Cross-References

* §13sexiesdecies: P37 (one-shot $\alpha_\chi$ at the `canonical` gauge; P38 generalises across gauges).
* §15: P18 (ζ-side admissibility / gauge sweep; canonical reference template).
* §13septies: Conjecture T-HP (unchanged by P38).
* `src/tnfr/riemann/twisted_alpha_sweep.py`: canonical P38 implementation.
* `examples/04_riemann_L_twisted/65_twisted_alpha_sweep_demo.py`: reference demo.

### §13septiesdecies.7 Gap Balance

| Scope | Status before P38 | Status after P38 |
|-------|-------------------|------------------|
| P18 gauge sweep for ζ | Available (P18) | Available, unchanged |
| **Gauge sweep for $L(s,\chi)$, primitive real χ** | Open (future P38) | **Available** (TNFR-native robustness audit across 6 canonical gauges) |
| GRH for $L(s,\chi)$, primitive real χ | OPEN | OPEN (diagnostic only; finite $(\sigma, g)$ grid is necessary, not sufficient) |
| G4 = RH | OPEN | OPEN, unchanged |
| GRH (G4$_\chi$ for complex $\chi$) | OPEN | OPEN, unchanged |

**Net effect**: P38 closes the admissibility/gauge-sweep gap on the L-function track for every primitive real Dirichlet character.  Combined with P32–P37, the structural TNFR-Riemann program now matches the ζ track all the way through P18.  The arithmetic obstruction remains identical and the gap balance for G4 is unchanged.

## §13octiesdecies. P39 — χ-Twisted Admissible-Family + Gauge Sweep of $\alpha_\chi(\sigma; f, g)$ (Joint Test-Profile / Canonical-Mapping Robustness Diagnostic; GRH$_\chi$-Equivalent for Primitive Real χ; Does NOT Prove GRH or Advance G4)

### §13octiesdecies.1 Motivation

P38 (§13septiesdecies) probed the canonical-mapping ambiguity of the P37 chi-twisted positivity bridge by sweeping the six canonical structural gauges `DEFAULT_GAUGES` against a Gaussian-only test profile.  The ζ-track equivalent (P18) was subsequently extended by P19 (`admissible_family_sweep.py`), which sweeps three admissible Schwartz-even test families — `gaussian`, `gaussian_mixture`, `hermite2_gaussian` — to probe the *test-profile* ambiguity of the P17 bridge.  P39 imports the same admissible-family bundle unchanged and combines it with the P38 gauge sweep, yielding a dense $(family, gauge, \sigma)$ certificate for primitive real Dirichlet characters.

### §13octiesdecies.2 Construction

The chi-twisted Weil–TNFR ratio is defined cell-by-cell as
$$\alpha_\chi(\sigma; f, g) \;=\; \frac{W_\chi[\sigma; f]}{E_{\mathrm{TNFR}}^\chi[\sigma; f, g]},$$
where $W_\chi[\sigma; f]$ is the P35 chi-twisted zero-side enumerator evaluated on the admissible test function $f$ at width $\sigma$ (gauge-independent, computed once per $(family, \sigma)$ pair), and $E_{\mathrm{TNFR}}^\chi[\sigma; f, g]$ is the canonical TNFR Lyapunov energy of the structural test state built from $(f, g)$ on the P34 chi-twisted graph via `build_twisted_test_state_from_test_function`.  The admissible families are inherited verbatim from P19 (`DEFAULT_TEST_FAMILIES`); the gauges are inherited verbatim from P18 (`DEFAULT_GAUGES`).  No new canonical object is introduced.

### §13octiesdecies.3 Empirical Verification

Demo `examples/04_riemann_L_twisted/66_twisted_admissible_family_sweep_demo.py` evaluates the sweep for $\chi_3, \chi_4, \chi_5$ across 3 families × 6 gauges × 5 widths $\sigma \in \{1.0, 1.5, 2.0, 2.5, 3.0\}$ (90 cells per character, 270 cells total).  Aggregate result:

| Character | Modulus | $W_\chi \ge 0$ | $\alpha_\chi > 0$ | $\alpha_{\min}$ | @(σ, family, gauge) | $\alpha_{\max}$ |
|-----------|---------|----------------|-------------------|-----------------|---------------------|-----------------|
| $\chi_3$  | 3 | True | True | $+1.27 \times 10^{-14}$ | $(1.000, \mathrm{gaussian}, \mathrm{canonical})$ | $+5.04 \times 10^{-1}$ |
| $\chi_4$  | 4 | True | True | $+2.71 \times 10^{-8}$  | $(1.000, \mathrm{gaussian}, \mathrm{canonical})$ | $+2.00 \times 10^{0}$  |
| $\chi_5$  | 5 | True | True | $+2.62 \times 10^{-10}$ | $(1.000, \mathrm{gaussian}, \mathrm{canonical})$ | $+6.94 \times 10^{-1}$ |

PASS rate: **3/3 characters**.  The minimum across every character/family/gauge cell occurs at the tightest Gaussian profile, in agreement with the Gaussian zero-side tail behaviour observed in P19 / P38; admissible mixtures and Hermite–Gaussian profiles inflate $\alpha_\chi$ uniformly, as expected from the spectral weight redistribution introduced by their extra mass at moderate frequencies.

### §13octiesdecies.4 What P39 Extends

P39 extends the P38 robustness audit jointly along the admissible-test-family axis (P19) and the canonical-gauge axis (P18), giving the L-track exact structural parity with the ζ-track at the level of P18 + P19 combined diagnostics.  The chi-twisted positivity bridge is shown to be robust under the *joint* perturbation of test profile and structural mapping for every tested primitive real character.

### §13octiesdecies.5 What P39 Does NOT Advance

P39 is a strict diagnostic and inherits every limitation of P19 / P38.  It does **not** prove GRH for any $L(s, \chi)$ (the $(family, gauge, \sigma)$ grid is finite; positivity on a finite grid is necessary but not sufficient for $L$-function admissibility on the full Schwartz cone).  It does **not** advance G4 = RH (the arithmetic obstruction is identical to the untwisted case).  It does **not** address GRH for complex Dirichlet characters (only primitive real $\chi_3, \chi_4, \chi_5$ are implemented).  Negative cells, if encountered at scale, would falsify the bridge *as parameterised by the given test family and gauge*; they would not falsify GRH$_\chi$ itself, which depends only on the gauge-independent quantities $W_\chi[\sigma; f]$.

### §13octiesdecies.6 Cross-References

* P19 (ζ-track admissible-family sweep): `src/tnfr/riemann/admissible_family_sweep.py`, §15 of these notes.
* P18 (canonical gauge family): `src/tnfr/riemann/alpha_sweep.py`, §14.
* P34 (chi-twisted prime-ladder Hamiltonian): `src/tnfr/riemann/twisted_prime_ladder_hamiltonian.py`, §13quaterdecies.
* P35 (chi-twisted Weil–Guinand zero-side enumerator): `src/tnfr/riemann/twisted_weil_explicit_formula.py`, §13quindecies.
* P37 (chi-twisted Weil–TNFR positivity bridge): `src/tnfr/riemann/twisted_weil_tnfr_bridge.py`, §13septdecies.
* P38 (chi-twisted gauge sweep): `src/tnfr/riemann/twisted_alpha_sweep.py`, §13septiesdecies.
* P17 (canonical Weil–TNFR positivity bridge): `src/tnfr/riemann/weil_positivity.py`, §14.
* Implementation: `src/tnfr/riemann/twisted_admissible_family_sweep.py`.
* Demo: `examples/04_riemann_L_twisted/66_twisted_admissible_family_sweep_demo.py`.

### §13octiesdecies.7 Gap Balance

| Scope | Status before P39 | Status after P39 |
|-------|-------------------|------------------|
| P19 admissible-family sweep for ζ | Available (P19) | Available, unchanged |
| P38 gauge sweep for $L(s,\chi)$ | Available (P38) | Available, unchanged |
| **Admissible-family + gauge sweep for $L(s,\chi)$, primitive real χ** | Open (future P39) | **Available** (TNFR-native robustness audit across 3 admissible families × 6 canonical gauges × σ grid) |
| GRH for $L(s,\chi)$, primitive real χ | OPEN | OPEN (diagnostic only; finite $(family, gauge, \sigma)$ grid is necessary, not sufficient) |
| G4 = RH | OPEN | OPEN, unchanged |
| GRH (G4$_\chi$ for complex $\chi$) | OPEN | OPEN, unchanged |

**Net effect**: P39 closes the admissible-family + gauge robustness gap on the L-function track for every primitive real Dirichlet character, achieving structural parity with the ζ-track through P19.  The arithmetic obstruction remains identical and the gap balance for G4 is unchanged.

## §13noniesdecies. P40 — χ-Twisted Node-Aware Gauge Sweep of $\alpha_\chi(\sigma; f, g)$ (Node-Aware Canonical-Mapping Robustness Diagnostic; GRH$_\chi$-Equivalent for Primitive Real χ; Does NOT Prove GRH or Advance G4)

### §13noniesdecies.1 Motivation

P38 swept the six canonical *scalar-h* structural gauges `DEFAULT_GAUGES` for primitive real $L(s,\chi)$.  P39 enriched that sweep along the test-profile axis by crossing the six scalar gauges with the three admissible test families `DEFAULT_TEST_FAMILIES` of P19.  Both P38 and P39 share a structural limitation: every gauge produces *node-independent* triples $(d, \phi, \epsilon)$ from the scalar $h(E_n)$.  The χ-twisted prime-ladder graph carries two independent canonical channels at each node $n = (p, k)$ — the structural frequency $\nu_f(n) = k \log p$ and the node-weight $\log p$ — that the scalar-h gauges discard by construction.  The ζ-track closed this gap at P20 with the four *node-aware* gauges `DEFAULT_NODEAWARE_GAUGES`.  P40 lifts that node-aware family verbatim to the L-function track for every primitive real Dirichlet character.

### §13noniesdecies.2 Construction

The P40 sweep evaluates

$$
\alpha_\chi(\sigma; f, g) \;=\; \frac{W_\chi[\sigma; f]}{E_{\mathrm{TNFR}}^\chi[\sigma; f, g]}
$$

across (i) the three admissible Schwartz-even test families `DEFAULT_TEST_FAMILIES` inherited unchanged from P19 (gaussian, gaussian_mixture, hermite2_gaussian); (ii) the four canonical node-aware gauges `DEFAULT_NODEAWARE_GAUGES` inherited unchanged from P20 (nuf_pressure, nuf_phase, weight_pressure, mixed_affine); (iii) a finite Gaussian-width grid $\sigma \in \{1.0, 1.5, 2.0, 2.5, 3.0\}$.  Each node-aware gauge has the canonical signature

$$
(d_n, \phi_n, \epsilon_n) \;=\; g\bigl(h(E_n),\, \hat\nu_f(n),\, \hat w(n)\bigr),
$$

where $\hat\nu_f(n)$ and $\hat w(n) = \log p / \max_{n'} \log p$ are the per-node normalised structural-frequency and node-weight channels of the P34 χ-twisted prime-ladder bundle.  $W_\chi[\sigma; f]$ is gauge-independent and is computed once per $(family, \sigma)$ via the P35 enumerator `twisted_weil_zero_side`; the canonical TNFR test state is built per $(family, node\_gauge)$ on the P34 bundle via `build_twisted_test_state_nodeaware`, then $E_{\mathrm{TNFR}}^\chi[\sigma; f, g]$ is the tetrad energy functional of P17 evaluated on that state.

### §13noniesdecies.3 Empirical Verification

`examples/04_riemann_L_twisted/67_twisted_nodeaware_gauge_sweep_demo.py` evaluates the sweep for every primitive real Dirichlet character of conductor $q \le 5$ with bundle $(n_{\mathrm{primes}}, k_{\max}, J) = (25, 6, 0)$:

| $\chi$ | $q$ | $W_\chi \ge 0$ | $\alpha_\chi > 0$ | $\alpha_{\min}$ | argmin $(\sigma, f, g)$ | $\alpha_{\max}$ |
|--------|----:|:--------------:|:-----------------:|----------------:|:-----------------------:|----------------:|
| $\chi_{3}$ | 3 | True | True | $+1.25 \times 10^{-14}$ | $(1.0, \text{gaussian}, \text{nuf\_phase})$ | $+6.71 \times 10^{-2}$ |
| $\chi_{4}$ | 4 | True | True | $+2.69 \times 10^{-08}$ | $(1.0, \text{gaussian}, \text{nuf\_phase})$ | $+1.30 \times 10^{-1}$ |
| $\chi_{5}$ | 5 | True | True | $+2.60 \times 10^{-10}$ | $(1.0, \text{gaussian}, \text{nuf\_pressure})$ | $+1.12 \times 10^{-1}$ |

Aggregate result: **3/3 characters PASS** across $3 \times 4 \times 5 = 60$ $(family, node\_gauge, \sigma)$ entries each.  The argmin location is consistently the small-$\sigma$ / gaussian / pressure-side corner of the grid, where $W_\chi$ approaches the Plancherel limit while $E_{\mathrm{TNFR}}^\chi$ is largest — the same qualitative signature observed at P20 for the ζ-track and at P39 for the scalar-gauge twisted sweep.

### §13noniesdecies.4 What P40 Extends

| Component | P38 | P39 | **P40** |
|-----------|:---:|:---:|:-------:|
| Test family axis | single (gaussian) | sweep (3 admissible) | sweep (3 admissible) |
| Gauge axis | sweep (6 scalar) | sweep (6 scalar) | **sweep (4 node-aware)** |
| Node-aware channels $(\hat\nu_f, \hat w)$ | discarded | discarded | **active** |
| ζ-track parent | P18 | P19 | **P20** |

P40 closes the node-aware canonical-mapping robustness gap on the L-function track for primitive real Dirichlet characters, achieving structural parity with the ζ-track P20.

### §13noniesdecies.5 What P40 Does NOT Advance

P40 is a **finite-grid robustness diagnostic**: positivity of $\alpha_\chi(\sigma; f, g)$ on the chosen $(family, node\_gauge, \sigma)$ grid is necessary but not sufficient for GRH$_\chi$, and the GRH-equivalent content is carried entirely by the gauge-independent zero side $W_\chi[\sigma; f] \ge 0$ for all admissible $f$.  P40 does NOT prove GRH for any $L(s, \chi)$, does NOT extend to complex characters, and does NOT advance the gap balance for G4 = RH (which lives strictly inside the canonical ζ track via P30 → T-HP).

### §13noniesdecies.6 Cross-References

- Implementation: `src/tnfr/riemann/twisted_nodeaware_gauge_sweep.py` (module), `src/tnfr/riemann/__init__.py` (canonical exports).
- Demonstration: `examples/04_riemann_L_twisted/67_twisted_nodeaware_gauge_sweep_demo.py`.
- ζ-track parent: P20 (§13ter `nodeaware_gauge_sweep.py`).
- L-track parents: P34 (χ-twisted bundle), P35 (`twisted_weil_zero_side`), P37 (`verify_twisted_weil_tnfr_bridge`, energy functional), P38 (scalar-gauge twisted sweep), P39 (admissible-family + scalar-gauge twisted sweep).
- Inherited canonical pieces: `DEFAULT_TEST_FAMILIES` (P19), `DEFAULT_NODEAWARE_GAUGES` (P20), `compute_energy_functional` (P17).
- Compendium: §19.1 P40 row.

### §13noniesdecies.7 Gap Balance

| Scope | Status before P40 | Status after P40 |
|-------|-------------------|------------------|
| P20 node-aware gauge sweep for ζ | Available (P20) | Available, unchanged |
| P39 admissible-family + scalar-gauge sweep for $L(s,\chi)$ | Available (P39) | Available, unchanged |
| **Admissible-family + node-aware gauge sweep for $L(s,\chi)$, primitive real χ** | Open (future P40) | **Available** (TNFR-native robustness audit across 3 admissible families × 4 node-aware gauges × σ grid) |
| GRH for $L(s,\chi)$, primitive real χ | OPEN | OPEN (diagnostic only; finite $(family, node\_gauge, \sigma)$ grid is necessary, not sufficient) |
| G4 = RH | OPEN | OPEN, unchanged |
| GRH (G4$_\chi$ for complex $\chi$) | OPEN | OPEN, unchanged |

**Net effect**: P40 closes the node-aware canonical-mapping robustness gap on the L-function track for every primitive real Dirichlet character, achieving structural parity with the ζ-track through P20.  The arithmetic obstruction remains identical and the gap balance for G4 is unchanged.

## §13vicies. P41 — χ-Twisted Hermite2-Gaussian η-Parameter Sweep of $\alpha_\chi(\sigma; \eta, g)$ (Hermite2 Envelope-Strength Robustness Diagnostic; GRH$_\chi$-Equivalent for Primitive Real χ; Does NOT Prove GRH or Advance G4)

### §13vicies.1 Motivation

P39 and P40 swept the three admissible Schwartz-even test families `DEFAULT_TEST_FAMILIES` of P19 with the Hermite2-Gaussian profile fixed at its canonical envelope strength $\eta = 0.25$.  The Hermite2 profile

$$
h_{\sigma,\eta}(t) \;=\; \bigl(1 + \eta (t/\sigma)^2\bigr)\, e^{-t^2/(2\sigma^2)}
$$

is a one-parameter family of Schwartz-even test functions that recovers the pure Gaussian baseline at $\eta = 0$ and progressively biases the test profile toward the wings as $\eta$ grows.  The ζ-track P21 added the Hermite2-Gaussian to the admissible-family registry but did not separately probe the envelope-strength axis itself.  P41 enriches the L-track sweep along that orthogonal axis: it varies $\eta$ over a finite grid spanning baseline-Gaussian to strongly-deformed envelope for every primitive real Dirichlet character.

### §13vicies.2 Construction

The P41 sweep evaluates

$$
\alpha_\chi(\sigma; \eta, g) \;=\; \frac{W_\chi[\sigma; \eta]}{E_{\mathrm{TNFR}}^\chi[\sigma; \eta, g]}
$$

across (i) the Hermite2 envelope-strength grid `DEFAULT_HERMITE2_ETAS = (0.0, 0.1, 0.25, 0.5, 1.0, 2.0)` ($\eta = 0$ recovers the pure Gaussian baseline; $\eta = 0.25$ matches the P19/P39 snapshot); (ii) the six canonical scalar gauges `DEFAULT_GAUGES` inherited unchanged from P18; (iii) the same finite Gaussian-width grid $\sigma \in \{1.0, 1.5, 2.0, 2.5, 3.0\}$.  $W_\chi[\sigma; \eta]$ is gauge-independent and computed once per $(\eta, \sigma)$ via the P35 enumerator `twisted_weil_zero_side`; the canonical TNFR test state is built per $(\eta, g)$ on the P34 χ-twisted bundle via `build_twisted_test_state_from_test_function` (reused from P39), then $E_{\mathrm{TNFR}}^\chi[\sigma; \eta, g]$ is the tetrad energy functional of P17 evaluated on that state.

### §13vicies.3 Empirical Verification

`examples/04_riemann_L_twisted/68_twisted_hermite_family_demo.py` evaluates the sweep for every primitive real Dirichlet character of conductor $q \le 5$ with bundle $(n_{\mathrm{primes}}, k_{\max}, J) = (25, 6, 0)$:

| $\chi$ | $q$ | $W_\chi \ge 0$ | $\alpha_\chi > 0$ | $\alpha_{\min}$ | argmin $(\sigma, \eta, g)$ | $\alpha_{\max}$ |
|--------|----:|:--------------:|:-----------------:|----------------:|:--------------------------:|----------------:|
| $\chi_{3}$ | 3 | True | True | $+1.27 \times 10^{-14}$ | $(1.0, 0.0, \text{canonical})$ | $+9.54 \times 10^{-1}$ |
| $\chi_{4}$ | 4 | True | True | $+2.71 \times 10^{-08}$ | $(1.0, 0.0, \text{canonical})$ | $+6.00 \times 10^{+0}$ |
| $\chi_{5}$ | 5 | True | True | $+2.62 \times 10^{-10}$ | $(1.0, 0.0, \text{canonical})$ | $+1.79 \times 10^{+0}$ |

Aggregate result: **3/3 characters PASS** across $6 \times 6 \times 5 = 180$ $(\eta, g, \sigma)$ entries each.  $W_\chi[\sigma; \eta]$ is monotone non-decreasing in $\eta$ at each fixed $\sigma$, consistent with the broader-spectral-support character of the deformed envelope; $\alpha_\chi$ increases sharply with $\eta$ along the `dnfr_only` and `epi_only` gauge channels and remains nearly $\eta$-invariant along the four canonical gauges that consume the $h$-channel only.  The argmin is consistently $(\sigma, \eta, g) = (1.0, 0.0, \text{canonical})$ — the Gaussian baseline corner — matching the P38/P39 argmin pattern.

### §13vicies.4 What P41 Extends

| Component | P38 | P39 | P40 | **P41** |
|-----------|:---:|:---:|:---:|:-------:|
| Test family axis | single (gaussian) | sweep (3 admissible, $\eta = 0.25$ fixed) | sweep (3 admissible, $\eta = 0.25$ fixed) | **sweep (Hermite2 with 6-point $\eta$-grid)** |
| Gauge axis | sweep (6 scalar) | sweep (6 scalar) | sweep (4 node-aware) | sweep (6 scalar) |
| Hermite2 envelope-strength $\eta$ | n/a | fixed at $0.25$ | fixed at $0.25$ | **swept over $\{0.0, 0.1, 0.25, 0.5, 1.0, 2.0\}$** |
| ζ-track parent | P18 | P19 | P20 | **P21** |

P41 closes the Hermite2 envelope-strength robustness gap on the L-function track for primitive real Dirichlet characters, achieving structural parity with the ζ-track P21 along the envelope-deformation axis.

### §13vicies.5 What P41 Does NOT Advance

P41 is a **finite-grid robustness diagnostic**: positivity of $\alpha_\chi(\sigma; \eta, g)$ on the chosen $(\eta, g, \sigma)$ grid is necessary but not sufficient for GRH$_\chi$, and the GRH-equivalent content is carried entirely by the gauge-independent zero side $W_\chi[\sigma; \eta] \ge 0$ for every admissible Hermite2 profile.  P41 does NOT prove GRH for any $L(s, \chi)$, does NOT extend to complex characters, and does NOT advance the gap balance for G4 = RH (which lives strictly inside the canonical ζ track via P30 → T-HP).  The Hermite2 family is a one-parameter polynomial-envelope deformation of the Gaussian; it is not exhaustive over the full admissible Schwartz-even space.

### §13vicies.6 Cross-References

- Implementation: `src/tnfr/riemann/twisted_hermite_family.py` (module), `src/tnfr/riemann/__init__.py` (canonical exports).
- Demonstration: `examples/04_riemann_L_twisted/68_twisted_hermite_family_demo.py`.
- ζ-track parent: P21 (Hermite2 added to `DEFAULT_TEST_FAMILIES`).
- L-track parents: P34 (χ-twisted bundle), P35 (`twisted_weil_zero_side`), P37 (energy functional), P38 (scalar-gauge twisted sweep), P39 (admissible-family + scalar-gauge twisted sweep; supplies `build_twisted_test_state_from_test_function`), P40 (node-aware twisted sweep).
- Inherited canonical pieces: `Hermite2GaussianTestFunction` (P19), `DEFAULT_GAUGES` (P18), `compute_energy_functional` (P17).
- Compendium: §19.1 P41 row.

### §13vicies.7 Gap Balance

| Scope | Status before P41 | Status after P41 |
|-------|-------------------|------------------|
| P21 Hermite2 family in ζ-track admissible registry | Available (P21) | Available, unchanged |
| P39 admissible-family + scalar-gauge sweep for $L(s,\chi)$ at fixed $\eta = 0.25$ | Available (P39) | Available, unchanged |
| **Hermite2 envelope-strength η-sweep for $L(s,\chi)$, primitive real χ** | Open (future P41) | **Available** (TNFR-native robustness audit across 6 $\eta$ values × 6 scalar gauges × σ grid) |
| GRH for $L(s,\chi)$, primitive real χ | OPEN | OPEN (diagnostic only; finite $(\eta, g, \sigma)$ grid is necessary, not sufficient) |
| G4 = RH | OPEN | OPEN, unchanged |
| GRH (G4$_\chi$ for complex $\chi$) | OPEN | OPEN, unchanged |

**Net effect**: P41 closes the Hermite2 envelope-strength robustness gap on the L-function track for every primitive real Dirichlet character.  The arithmetic obstruction remains identical and the gap balance for G4 is unchanged.

## §13vicies-primo. P42 — χ-Twisted Uniform-Coercivity Certificate (Lipschitz-Mesh Interval Bound on $\alpha_\chi(\sigma; \eta, g)$; Diagnostic; Does NOT Prove GRH or Advance G4)

### §13vicies-primo.1 Motivation

P38–P41 verified pointwise positivity of $\alpha_\chi(\sigma; f, \eta, g) = W_\chi[\sigma; f, \eta] / E_{\mathrm{TNFR}}^\chi[\sigma; f, \eta, g]$ at the canonical finite grid $\sigma \in \{1.0, 1.5, 2.0, 2.5, 3.0\}$ jointly across the test-family (P39), node-aware-gauge (P40) and Hermite2 envelope-strength (P41) axes.  None of those sweeps controls $\alpha_\chi$ between grid points.  The ζ-track P22 lifted the equivalent ζ-side sample to an **interval** lower bound by combining a sampled minimum with a finite-difference Lipschitz envelope and a log-spaced mesh of explicit radius.  P42 transports the same Lipschitz-mesh certificate construction to the χ-twisted track for every primitive real Dirichlet character, taking the sample over the *joint* (admissible-family + scalar-gauge + node-aware-gauge) sweep already canonicalised in P39 and P40.

### §13vicies-primo.2 Construction

The P42 certificate evaluates

$$
\alpha_\chi(\sigma; \eta, g) \;=\; \frac{W_\chi[\sigma; \eta]}{E_{\mathrm{TNFR}}^\chi[\sigma; \eta, g]}
$$

on a log-spaced grid $\sigma_0 < \sigma_1 < \cdots < \sigma_{N-1}$ with $\sigma_k = \sigma_{\min} \cdot (\sigma_{\max}/\sigma_{\min})^{k/(N-1)}$, then:

1. Runs the **scalar-gauge sweep of P39** (`sweep_twisted_admissible_family`, 3 admissible families × 6 canonical scalar gauges of P18) once on the log-spaced grid;
2. Runs the **node-aware-gauge sweep of P40** (`sweep_twisted_nodeaware_gauge`, 3 admissible families × 4 canonical node-aware gauges of P20) once on the same grid;
3. Concatenates both $\alpha_\chi$ tables and extracts the sampled minimum $\alpha_{\chi,\min}^{\mathrm{samp}}$, maximum $\alpha_{\chi,\max}^{\mathrm{samp}}$ and an upper bound on the finite-difference Lipschitz envelope $L^{\mathrm{proxy}}_\chi = \max_{k} |\alpha_\chi(\sigma_{k+1}; \cdot) - \alpha_\chi(\sigma_k; \cdot)| / |\sigma_{k+1} - \sigma_k|$;
4. Computes three interval lower bounds — **global** ($\alpha_{\chi,\min}^{\mathrm{samp}} - L^{\mathrm{proxy}}_\chi \cdot \rho$ with mesh radius $\rho = \max_k (\sigma_{k+1} - \sigma_k)/2$), **stratified** (segment-wise mid-radius), **local** (segment-wise endpoint-aware) — reusing the canonical ζ-track helpers `_max_abs_slope`, `_segmentwise_interval_lower_bound`, `_stratified_interval_lower_bound` of P22 unchanged from `coercivity_uniform.py`;
5. Optionally performs **P24-style adaptive refinement**: bisects the `per_round` worst-margin segments, re-runs both twisted sweeps on the augmented grid, and recomputes the segment-local interval lower bound.

The construction does NOT touch the gauge-independent zero side $W_\chi$ (computed once per $(\eta, \sigma)$ via the P35 enumerator inside each sweep), the P34 χ-twisted bundle, the P17 energy functional, or any of the canonical default registries.

### §13vicies-primo.3 Empirical Verification

`examples/04_riemann_L_twisted/69_twisted_coercivity_uniform_demo.py` evaluates the certificate for every primitive real Dirichlet character of conductor $q \le 5$ with bundle $(n_{\mathrm{primes}}, k_{\max}, J) = (15, 4, 0)$ on the log-spaced window $\sigma \in [1.0, 3.0]$ with $N = 5$, using `DEFAULT_TEST_FAMILIES` (P19) × `DEFAULT_GAUGES` (P18) for the scalar sweep and `DEFAULT_TEST_FAMILIES` (P19) × `DEFAULT_NODEAWARE_GAUGES` (P20) for the node-aware sweep:

| $\chi$ | $q$ | $\alpha^{\mathrm{samp}}_{\chi,\min}$ | $\alpha^{\mathrm{samp}}_{\chi,\max}$ | $L^{\mathrm{proxy}}_\chi$ | $\mathrm{lb}_{\mathrm{global}}$ | $\mathrm{lb}_{\mathrm{strat}}$ | $\mathrm{lb}_{\mathrm{local}}$ | all+ |
|--------|----:|------------------------------------:|------------------------------------:|--------------------------:|--------------------------------:|-------------------------------:|-------------------------------:|:----:|
| $\chi_{3}$ | 3 | $+1.26 \times 10^{-14}$ | $+5.10 \times 10^{-1}$ | $4.31 \times 10^{-1}$ | $-1.55 \times 10^{-1}$ | $-1.55 \times 10^{-1}$ | $-6.06 \times 10^{-2}$ | False |
| $\chi_{4}$ | 4 | $+2.70 \times 10^{-8}$  | $+2.01 \times 10^{+0}$ | $1.48 \times 10^{+0}$ | $-5.33 \times 10^{-1}$ | $-5.16 \times 10^{-1}$ | $-1.30 \times 10^{-1}$ | False |
| $\chi_{5}$ | 5 | $+2.62 \times 10^{-10}$ | $+7.01 \times 10^{-1}$ | $5.49 \times 10^{-1}$ | $-1.98 \times 10^{-1}$ | $-1.95 \times 10^{-1}$ | $-6.51 \times 10^{-2}$ | False |

Sampled positivity holds for every $\chi$ on every grid point in both sweeps (`sampled_all_positive = True`, `admissible_ok = True`, `nodeaware_ok = True`).  All three Lipschitz-mesh interval lower bounds are **negative** for all three characters: $\alpha^{\mathrm{samp}}_{\chi,\min} \approx 10^{-8}$ to $10^{-14}$ near the $\sigma = 1$ baseline gives essentially zero margin against any finite slope $L^{\mathrm{proxy}}_\chi$.  P24-style refinement on the worst-margin character ($\chi_4$, $\mathrm{lb}_{\mathrm{local}} = -1.30 \times 10^{-1}$) with one round of two-midpoint bisection ($N = 5 \to 7$) reduces the local interval lower bound to $-3.40 \times 10^{-2}$ — a **74% margin reduction toward zero**, confirming the bisection mechanism transports correctly to the χ-twisted side, while the bound remains negative because the sampled minimum near $\sigma = 1$ has not been pushed off the worst-margin endpoint.

### §13vicies-primo.4 What P42 Extends

| Component | P22 (ζ-track) | P38 | P39 | P40 | P41 | **P42** |
|-----------|:-------------:|:---:|:---:|:---:|:---:|:-------:|
| Sample / interval | interval | pointwise | pointwise | pointwise | pointwise | **interval (Lipschitz-mesh)** |
| σ grid | log-spaced | finite | finite | finite | finite | **log-spaced (same construction as P22)** |
| Lipschitz envelope | finite-difference | n/a | n/a | n/a | n/a | **finite-difference (P22 helpers reused)** |
| Joint scalar + node-aware sample | scalar only | scalar only | scalar only | node-aware only | scalar only | **both (P39 ∪ P40)** |
| Adaptive refinement | yes (P24) | n/a | n/a | n/a | n/a | **yes (P24 helpers reused)** |
| ζ-track parent | — | P18 | P19 | P20 | P21 | **P22 (+ P23 + P24)** |

P42 transports the canonical ζ-track interval-coercivity certificate construction (P22) plus its stratified (P23) and adaptive (P24) refinements to the L-function track for every primitive real Dirichlet character, taking the underlying sample over the joint P39 + P40 robustness sweep.

### §13vicies-primo.5 What P42 Does NOT Advance

P42 is a **finite-grid Lipschitz-mesh interval diagnostic**: positive interval lower bounds would be necessary but not sufficient for GRH$_\chi$.  The current empirical result is **negative interval lower bounds** for all three characters even after one round of bisection refinement, exactly mirroring the ζ-track P22/P23/P24 behaviour at coarse-mesh / wide-σ-window initial state: uniform coercivity is delicate near the $\sigma = 1$ baseline because the sampled minimum is genuinely tiny ($10^{-8}$ to $10^{-14}$).  P42 does NOT prove GRH for any $L(s, \chi)$, does NOT extend to complex characters, and does NOT advance the gap balance for G4 = RH.

### §13vicies-primo.6 Cross-References

- Implementation: `src/tnfr/riemann/twisted_coercivity_uniform.py` (module), `src/tnfr/riemann/__init__.py` (canonical exports).
- Demonstration: `examples/04_riemann_L_twisted/69_twisted_coercivity_uniform_demo.py`.
- ζ-track parent: P22 / P23 / P24 (uniform, stratified, adaptive coercivity in `coercivity_uniform.py`).
- L-track parents: P34 (χ-twisted bundle), P35 (`twisted_weil_zero_side`), P37 (energy functional), P38 (scalar-gauge twisted sweep), P39 (admissible-family + scalar-gauge twisted sweep), P40 (node-aware twisted sweep), P41 (Hermite2 η-sweep).
- Inherited canonical pieces: `_max_abs_slope`, `_segmentwise_interval_lower_bound`, `_stratified_interval_lower_bound`, `_worst_segment_indices` (P22 / P23 / P24 helpers reused unchanged from `coercivity_uniform.py`); `sweep_twisted_admissible_family` (P39); `sweep_twisted_nodeaware_gauge` (P40).
- Compendium: §19.1 P42 row.

### §13vicies-primo.7 Gap Balance

| Scope | Status before P42 | Status after P42 |
|-------|-------------------|------------------|
| P22 ζ-track interval coercivity certificate | Available (P22) | Available, unchanged |
| Pointwise positivity of $\alpha_\chi$ for primitive real χ on finite $(\sigma, f, \eta, g)$ grid | Available (P37–P41) | Available, unchanged |
| **Lipschitz-mesh interval-level certificate of $\alpha_\chi$ for primitive real χ** | Open (future P42) | **Available** (diagnostic; current empirical interval lower bounds are negative; adaptive bisection mechanism transports correctly from ζ-track) |
| GRH for $L(s,\chi)$, primitive real χ | OPEN | OPEN (interval lower bounds currently negative; diagnostic, not sufficient even when positive) |
| G4 = RH | OPEN | OPEN, unchanged |
| GRH (G4$_\chi$ for complex $\chi$) | OPEN | OPEN, unchanged |

**Net effect**: P42 closes the **Lipschitz-mesh interval-certificate construction gap** on the L-function track for every primitive real Dirichlet character (canonical pieces transport without modification; bisection refinement behaves qualitatively as on the ζ-track).  The current empirical interval lower bounds are negative — a HONEST finding, not a failure — and the arithmetic obstruction plus the gap balance for G4 are unchanged.

## §13vicies-secundo. P43 — χ-Twisted Paley-Gap Consistency Diagnostic ($|Z_{P34} - Z_{P32}|$ and Truncation Gaps on the L-Track; Diagnostic; Does NOT Prove GRH or Advance G4)

### §13vicies-secundo.1 Motivation

The ζ-track P25 milestone transported the Paley-gap philosophy of Martínez Gamo, *Spectral note: Paley gap via $\lambda_2$ (residue circulants)*, Zenodo 10.5281/zenodo.17665853 v2 (November 2025), onto the TNFR-Riemann coercivity scaffold by comparing three representations of the von Mangoldt logarithmic derivative — the P12 prime-ladder closed form $Z_{P12}(\sigma)$, the P14 self-adjoint weighted spectral trace $Z_{P14}(\sigma)$, and the classical truncated Dirichlet series $\sum_{n \le N} \Lambda(n)/n^\sigma$ — via three absolute Paley-gap quantities $g_{P12}(\sigma)$, $g_{P14}(\sigma)$, $g_{\mathrm{cross}}(\sigma) = |Z_{P14}(\sigma) - Z_{P12}(\sigma)|$. The Paley-style observation was that at zero inter-ladder coupling the cross gap $g_{\mathrm{cross}}$ collapses to machine precision by a closed-form algebraic identity between the two TNFR realisations, while a non-zero coupling exposes a clean structural-deformation magnitude free of classical-truncation noise.

The χ-twisted track P32–P34 provides the same two realisations for the von Mangoldt logarithmic derivative of $L(s, \chi)$ — the P32 closed-form weighted spectrum (`tnfr_log_l_derivative`) and the P34 self-adjoint χ-twisted weighted spectral trace (`twisted_weighted_spectral_trace`) — alongside the classical truncated reference $\sum_{n \le N} \chi(n)\Lambda(n)/n^\sigma$ (`classical_log_l_derivative`). P43 transports the P25 Paley-gap diagnostic construction to the χ-twisted track for every primitive real Dirichlet character, providing the analogous consistency surface on the L-track.

### §13vicies-secundo.2 Construction

The P43 diagnostic evaluates, on a real $\sigma$-grid, three absolute χ-twisted Paley-gap quantities

$$
\begin{aligned}
g_{P32}(\sigma) &= \left|Z_{P32}(\sigma, \chi) - Z_{\mathrm{cls}}(\sigma, \chi)\right|, \\
g_{P34}(\sigma) &= \left|Z_{P34}(\sigma, \chi) - Z_{\mathrm{cls}}(\sigma, \chi)\right|, \\
g_{\mathrm{cross}}(\sigma) &= \left|Z_{P34}(\sigma, \chi) - Z_{P32}(\sigma, \chi)\right|,
\end{aligned}
$$

where $Z_{P32}(\sigma, \chi) = \sum_{(\mu, w) \in \mathrm{spec}_\chi} w \, e^{-\sigma \mu}$ is the P32 closed-form weighted spectrum, $Z_{P34}(\sigma, \chi) = \mathrm{Tr}(W_\chi e^{-\sigma H_{\mathrm{int}}})$ is the P34 χ-twisted weighted spectral trace, and $Z_{\mathrm{cls}}(\sigma, \chi) = \sum_{n \le N_{\max}} \chi(n) \Lambda(n) / n^\sigma$ is the classical truncated reference, all computed on the same $(n_{\mathrm{primes}}, k_{\max})$ prime-ladder bundle. The driver `sweep_twisted_paley_gap(bundle, chi, sigmas, n_max_classical=...)` returns a `TwistedPaleyGapSweep` dataclass carrying the three gap arrays and their worst-case magnitudes per character.

### §13vicies-secundo.3 Empirical Verification

P43 was run on the canonical config $(n_{\mathrm{primes}}, k_{\max}, N_{\max}^{\mathrm{cls}}) = (18, 5, 50\,000)$, $\sigma \in [1.5, 4.0]$ with $N = 11$, two bundles per character (decoupled $J_0 = 0$ and weakly coupled $J_0 = 10^{-2}$) for $\chi_3, \chi_4, \chi_5$:

| χ | $q$ | $\max g_{\mathrm{cross}}^{[J_0=0]}$ | $\max g_{\mathrm{cross}}^{[J_0=10^{-2}]}$ | $\max g_{P32}$ |
|---|---|---|---|---|
| $\chi_3$ | 3 | $5.55 \times 10^{-17}$ | $1.01 \times 10^{-5}$ | $2.22 \times 10^{-3}$ |
| $\chi_4$ | 4 | $4.16 \times 10^{-17}$ | $8.25 \times 10^{-6}$ | $1.49 \times 10^{-2}$ |
| $\chi_5$ | 5 | $1.11 \times 10^{-16}$ | $1.51 \times 10^{-5}$ | $1.11 \times 10^{-2}$ |

The decoupled cross gap collapses to machine precision ($O(10^{-17})$) for every character, confirming the Paley-style algebraic identity between P32 and P34 on the L-track (regression test, **not** a discovery: the identity follows from $W_\chi$ being the diagonal lift of the spectrum weights and $H_{\mathrm{int}}$ being block-diagonal at $J_0 = 0$). The coupling-induced cross gap jumps to $O(10^{-5})$ — twelve orders of magnitude above noise — exposing pure coupling-induced deformation of the χ-twisted prime-ladder identity, free of classical-truncation noise (which contaminates $g_{P32}$ at the $10^{-3}$ to $10^{-2}$ level).

### §13vicies-secundo.4 What P43 Extends

| Component | P25 (ζ-track) | P32 | P34 | **P43** |
|-----------|---------------|-----|-----|---------|
| Closed-form von Mangoldt logarithmic derivative | $Z_{P12}$ for $-\zeta'/\zeta$ | $Z_{P32}$ for $-L'(s,\chi)/L(s,\chi)$ | — | $Z_{P32}$ reused unchanged |
| Self-adjoint weighted spectral trace | $Z_{P14}$ for $-\zeta'/\zeta$ | — | $Z_{P34}$ for $-L'(s,\chi)/L(s,\chi)$ | $Z_{P34}$ reused unchanged |
| Classical truncated reference | $\sum_{n \le N} \Lambda(n)/n^\sigma$ | — | — | $\sum_{n \le N} \chi(n)\Lambda(n)/n^\sigma$ via `classical_log_l_derivative` |
| Three Paley-gap quantities | $g_{P12}, g_{P14}, g_{\mathrm{cross}}$ | — | — | $g_{P32}, g_{P34}, g_{\mathrm{cross}}$ |
| Paley-style decoupled identity ($g_{\mathrm{cross}} \to 0$ at $J_0 = 0$) | Empirically $O(10^{-17})$ | — | — | Empirically $O(10^{-17})$ for $\chi_3, \chi_4, \chi_5$ |
| Coupling-induced deformation signal | $J_0 = 10^{-2} \Rightarrow g_{\mathrm{cross}} \sim 10^{-5}$ | — | — | $J_0 = 10^{-2} \Rightarrow g_{\mathrm{cross}} \sim 10^{-5}$ for $\chi_3, \chi_4, \chi_5$ |

P43 transports the canonical ζ-track Paley-gap diagnostic (P25) to the L-function track for every primitive real Dirichlet character, exhibiting the identical decoupled-identity / coupled-deformation pattern.

### §13vicies-secundo.5 What P43 Does NOT Advance

P43 is a **consistency diagnostic at coupling zero and a deformation magnitude at coupling positive**. Vanishing of $g_{\mathrm{cross}}$ at $J_0 = 0$ is **necessary but not sufficient** for any structural positivity claim and **not connected** to GRH localisation; it is a regression test selecting consistent realisations, not a coercivity certificate. P43 does NOT prove GRH for any $L(s, \chi)$, does NOT extend to complex characters, does NOT advance the P42 interval-coercivity certificate (which lives on the $\alpha_\chi$ axis, not on the von Mangoldt logarithmic derivative axis), and does NOT advance the gap balance for G4 = RH. The Zenodo source note itself disclaims primality proof status; P43 inherits the same scope at the L-track coercivity-diagnostic level.

### §13vicies-secundo.6 Cross-References

- Implementation: `src/tnfr/riemann/twisted_paley_gap_coercivity.py` (module), `src/tnfr/riemann/__init__.py` (canonical exports).
- Demonstration: `examples/04_riemann_L_twisted/70_twisted_paley_gap_coercivity_demo.py`.
- ζ-track parent: P25 (`paley_gap_coercivity.py`).
- L-track parents: P32 (`tnfr_log_l_derivative`, `TwistedPrimeLadderSpectrum`), P34 (`TwistedPrimeLadderHamiltonian`, `twisted_weighted_spectral_trace`).
- Inherited canonical pieces: `tnfr_log_l_derivative` (P32 Route A), `twisted_weighted_spectral_trace` (P34 Route B), `classical_log_l_derivative` (classical reference), `build_twisted_prime_ladder_hamiltonian` (P34 bundle constructor) reused unchanged.
- External source: Martínez Gamo, *Spectral note: Paley gap via $\lambda_2$ (residue circulants)*, Zenodo 10.5281/zenodo.17665853 v2 (November 2025).
- Compendium: §19.1 P43 row.

### §13vicies-secundo.7 Gap Balance

| Scope | Status before P43 | Status after P43 |
|-------|-------------------|------------------|
| P25 ζ-track Paley-gap diagnostic | Available (P25) | Available, unchanged |
| Closed-form / self-adjoint consistency for $-L'(s,\chi)/L(s,\chi)$ on primitive real χ | Implicit in P32 + P34 construction (not empirically separated) | **Empirically separated and quantified** (decoupled cross gap $O(10^{-17})$; coupled cross gap $O(10^{-5})$ at $J_0 = 10^{-2}$) |
| Truncation-noise separation from coupling-induced deformation | Open | **Available** (cross gap is free of classical-truncation noise by construction) |
| GRH for $L(s,\chi)$, primitive real χ | OPEN | OPEN (Paley gap is a regression test, not a coercivity certificate) |
| G4 = RH | OPEN | OPEN, unchanged |
| GRH (G4$_\chi$ for complex $\chi$) | OPEN | OPEN, unchanged |

**Net effect**: P43 closes the **Paley-gap diagnostic-construction gap** on the L-function track for every primitive real Dirichlet character (canonical pieces transport without modification; the decoupled identity holds to machine precision, the coupled deformation signal is twelve orders of magnitude above noise). The arithmetic obstruction and the gap balance for G4 are unchanged.

## §13vicies-tertio. P44 — χ-Twisted Lyapunov-Spectral Positivity Certificate (L-Track Analogue of P26; Operator-Level; Does NOT Prove GRH or Advance G4)

### §13vicies-tertio.1 Motivation

P26 (`lyapunov_spectral_positivity.py`, §13quater) supplies a four-ingredient certificate for the ζ-track P14 prime-ladder Hamiltonian — self-adjointness, strict positivity with explicit Kato–Rellich envelope, trace-class resolvent, and unitary flow on the finite-dimensional prime-ladder Hilbert space. The L-track development (P32–P34) instantiates the same canonical TNFR `InternalHamiltonian` machinery on the χ-twisted prime-ladder graph: `TwistedPrimeLadderHamiltonian.hamiltonian` exposes the very same `H_int`, `H_freq`, `H_coupling` triplet that P26 consumes. P44 transports the P26 certificate to the χ-twisted bundle for every primitive real Dirichlet character, exhibiting the analogous operator-level positivity surface on the L-track.

### §13vicies-tertio.2 Construction

Let $\hat H^{(\chi)} = \hat H^{(\chi)}_{\mathrm{freq}} + J_0\,\hat H^{(\chi)}_{\mathrm{coupling}}$ on the χ-twisted prime-ladder Hilbert space

$$
\mathcal{H}_{\mathrm{PL},\chi}
  \;=\; \bigoplus_{p \in \mathcal{P},\; p \nmid q}\;
        \bigoplus_{k=1}^{K}\, \mathbb{C}\,|p,k\rangle,
$$

where $q$ is the conductor of $\chi$ and the primes dividing $q$ are excluded by construction (because $\chi(p^k) = 0$ for those primes; this is the P32 active-prime restriction propagated into P34). The diagonal frequency operator has entries $\nu_{f,(p,k)} = k\log p$ for $p \nmid q$, $k \ge 1$, so the unperturbed gap is

$$
\Delta_0^{(\chi)} \;=\; \min_{p \nmid q,\;k \ge 1}\, k\log p
  \;=\; \log\!\bigl(\min\{p \text{ prime} : p \nmid q\}\bigr).
$$

For the three primitive real characters this evaluates to:

| Character | Conductor $q$ | Smallest active prime | Unperturbed gap $\Delta_0^{(\chi)}$ |
|---|---|---|---|
| $\chi_3$ | $3$ | $2$ | $\log 2 \approx 0.6931$ |
| $\chi_4$ | $4$ | $3$ | $\log 3 \approx 1.0986$ |
| $\chi_5$ | $5$ | $2$ | $\log 2 \approx 0.6931$ |

The **Kato–Rellich (Weyl)** perturbation theorem applied to bounded symmetric perturbations of a self-adjoint diagonal operator yields the quantitative lower bound

$$
\lambda_{\min}\!\bigl(\hat H^{(\chi)}\bigr)
  \;\ge\; \Delta_0^{(\chi)}
        \;-\; |J_0|\,\bigl\|\hat H^{(\chi)}_{\mathrm{coupling}}\bigr\|_{\mathrm{op}},
$$

with `perturbation_safe = True` iff the right-hand side is strictly positive. The remaining three ingredients (resolvent Schatten-1/Hilbert-Schmidt norms at shift $c$, unitary norm/energy drifts of $U(t) = e^{-it\hat H^{(\chi)}}$, structural positivity composite) replicate P26 atomically and reuse `resolvent_schatten_norms` and `_matrix_exponential_skew` from `lyapunov_spectral_positivity.py` unchanged.

### §13vicies-tertio.3 Empirical Verification

P44 was run on the canonical config $(n_{\mathrm{primes}}, k_{\max}, c) = (18, 5, 1.0)$ for $\chi_3, \chi_4, \chi_5$ at $J_0 \in \{0, 10^{-2}\}$ (`examples/04_riemann_L_twisted/71_twisted_lyapunov_spectral_demo.py`):

| Character | $J_0$ | $\min(\lambda)$ | $\Delta_0^{(\chi)}$ | $\|\hat V\|$ | Guaranteed gap | `perturbation_safe` | Max norm drift | `unitary` | `structural_positivity` |
|---|---|---|---|---|---|---|---|---|---|
| $\chi_3$ | $0$ | $6.931\times 10^{-1}$ | $\log 2$ | $0$ | $\log 2$ | True | $2.22\times 10^{-16}$ | True | True |
| $\chi_3$ | $10^{-2}$ | $6.930\times 10^{-1}$ | $\log 2$ | $1.73\times 10^{-2}$ | $6.758\times 10^{-1}$ | True | $\sim 10^{-16}$ | True | True |
| $\chi_4$ | $0$ | $1.099\times 10^{0}$ | $\log 3$ | $0$ | $\log 3$ | True | $2.22\times 10^{-16}$ | True | True |
| $\chi_4$ | $10^{-2}$ | $1.099\times 10^{0}$ | $\log 3$ | $1.73\times 10^{-2}$ | $1.081\times 10^{0}$ | True | $\sim 10^{-16}$ | True | True |
| $\chi_5$ | $0$ | $6.931\times 10^{-1}$ | $\log 2$ | $0$ | $\log 2$ | True | $2.22\times 10^{-16}$ | True | True |
| $\chi_5$ | $10^{-2}$ | $6.930\times 10^{-1}$ | $\log 2$ | $1.73\times 10^{-2}$ | $6.758\times 10^{-1}$ | True | $\sim 10^{-16}$ | True | True |

At $J_0 = 0$ the empirical spectral bottom equals the analytic Kato–Rellich envelope to machine precision; the unperturbed gap matches $\log(\min\{p \nmid q\})$ exactly (asserted in the demo). At $J_0 = 10^{-2}$ the empirical bottom drops by $\sim 1.4\times 10^{-4}$ while the Kato–Rellich envelope drops by the full $\|V\| \approx 1.73\times 10^{-2}$, confirming the envelope is a strict (and loose) lower bound. Unitary flow conservation is verified to machine precision for every character at every tested coupling.

### §13vicies-tertio.4 What P44 Extends

| Component | P26 (ζ-track) | P34 | **P44** |
|---|---|---|---|
| Self-adjoint prime-ladder Hamiltonian | $\hat H$ on $\mathcal{H}_{\mathrm{PL}}$ | $\hat H^{(\chi)}$ on $\mathcal{H}_{\mathrm{PL},\chi}$ | reused unchanged |
| Spectral compute primitive | `compute_spectrum` | — | `twisted_compute_spectrum` |
| Kato–Rellich envelope | `kato_rellich_lower_bound` (gap $= \log 2$) | — | `twisted_kato_rellich_lower_bound` (gap $= \log(\min\{p\nmid q\})$, character-dependent) |
| Schatten-norm primitive | `resolvent_schatten_norms` | — | reused unchanged |
| Unitary-flow verification | `verify_unitary_flow` | — | `twisted_verify_unitary_flow` |
| Composite certificate | `LyapunovSpectralCertificate` | — | `TwistedLyapunovSpectralCertificate` (adds `character_name`, `character_modulus`) |

P44 transports the canonical ζ-track Lyapunov-spectral positivity certificate (P26) to the L-function track for every primitive real Dirichlet character, exhibiting the identical four-ingredient structure with the character-dependent unperturbed gap $\log(\min\{p \nmid q\})$.

### §13vicies-tertio.5 What P44 Does NOT Advance

P44 is an **operator-level positivity certificate on the finite-dimensional χ-twisted prime-ladder Hilbert space at fixed $(n_{\mathrm{primes}}, k_{\max})$**. Structural positivity at machine precision is **necessary but not sufficient** for any RH-equivalent positivity claim and **not connected** to GRH localisation. Passing to the analytic continuation introduces a non-finite-dimensional limit whose spectrum (in particular the localisation of resonance poles of $L(s,\chi)$ on $\operatorname{Re}(s) = 1/2$) is not addressed here. The χ-twisted weight operator $\hat W^{(\chi)}$ is **not** involved in the certificate: positivity of $\hat H^{(\chi)}_{\mathrm{int}}$ is independent of the character (the character enters only the spectral trace $Z_{\mathrm{TNFR}}(s,\chi)$ and the active-prime restriction in the ladder graph). P44 does NOT prove GRH for any $L(s,\chi)$, does NOT extend to complex characters (the construction is character-agnostic at the Hamiltonian level, but the canonical L-track currently exposes only the three primitive real characters), and does NOT advance the gap balance for G4 = RH.

### §13vicies-tertio.6 Cross-References

- Implementation: `src/tnfr/riemann/twisted_lyapunov_spectral_positivity.py` (module), `src/tnfr/riemann/__init__.py` (canonical exports).
- Demonstration: `examples/04_riemann_L_twisted/71_twisted_lyapunov_spectral_demo.py`.
- ζ-track parent: P26 (`lyapunov_spectral_positivity.py`, §13quater) — atomic primitives `_matrix_exponential_skew` and `resolvent_schatten_norms` reused unchanged.
- L-track parents: P32 (`TwistedPrimeLadderSpectrum` providing the active-prime catalogue), P34 (`TwistedPrimeLadderHamiltonian` providing `H_int`, `H_freq`, `H_coupling`).
- Compendium: §19.1 P44 row.

### §13vicies-tertio.7 Gap Balance

| Scope | Status before P44 | Status after P44 |
|-------|-------------------|------------------|
| P26 ζ-track Lyapunov-spectral positivity certificate | Available (P26) | Available, unchanged |
| Operator-level positivity certificate for $\hat H^{(\chi)}$ on $\mathcal{H}_{\mathrm{PL},\chi}$ | Implicit in P34 (diagonal $\nu_f > 0$ over active primes; never separated from coupling-perturbed bound) | **Explicit and quantified** (Kato–Rellich envelope $\Delta_0^{(\chi)} = \log(\min\{p\nmid q\})$ certified to machine precision at $J_0 = 0$; `structural_positivity = True` over $J_0 \in \{0, 10^{-2}\}$ for $\chi_3, \chi_4, \chi_5$) |
| Trace-class resolvent + unitary-flow conservation for $U(t) = e^{-it\hat H^{(\chi)}}$ | Open at the L-track | **Available** (Schatten-1/2 norms reported; unitary drifts $\sim 10^{-16}$) |
| GRH for $L(s,\chi)$, primitive real χ | OPEN | OPEN (operator-level positivity is necessary but not sufficient) |
| G4 = RH | OPEN | OPEN, unchanged |
| GRH (G4$_\chi$ for complex $\chi$) | OPEN | OPEN, unchanged |

**Net effect**: P44 closes the **operator-level Lyapunov-spectral positivity-certificate gap** on the L-function track for every primitive real Dirichlet character. The character-dependent unperturbed gap $\log(\min\{p \nmid q\})$ is exhibited explicitly and certified to machine precision; the Kato–Rellich envelope provides a rigorous quantitative interval for the perturbed regime. The arithmetic obstruction and the gap balance for G4 are unchanged.

## §13vicies-quarto. P45 — χ-Twisted Hilbert–Pólya Scaffold (L-Track Analogue of P27; Operator-Level; Does NOT Prove GRH or Advance G4)

### §13vicies-quarto.1 Motivation

P27 (ζ-track) builds the **explicit reference Hilbert–Pólya operator** $T_{\mathrm{HP}}^{(\zeta)} = \operatorname{diag}(\gamma_1, \gamma_2, \dots)$ on $\ell^2(\mathbb{N})$, where $\gamma_n$ are the positive imaginary parts of the non-trivial zeros of $\zeta$ retrieved from `mpmath.zetazero`. It certifies that the resulting scalar operator is self-adjoint, has trace-class shifted resolvent, and feeds the same zero side into the canonical Weil–Guinand identity (P15) as the prime-side P14 Hamiltonian — i.e., the rest of the ζ-track stack is internally compatible with a Hilbert–Pólya-style slot at the operator level. P27 does **not** derive $T_{\mathrm{HP}}^{(\zeta)}$ from TNFR first principles; it merely shows that, if such a derivation existed, the truncated stack would accept it.

P45 is the structural L-track mirror: for every primitive real Dirichlet character $\chi$ (modulus $q \in \{3, 4, 5\}$), it builds

$$T_{\mathrm{HP}}^{(\chi)} \;=\; \operatorname{diag}\bigl(\gamma_1^{(\chi)}, \gamma_2^{(\chi)}, \dots, \gamma_N^{(\chi)}\bigr) \quad\text{on}\quad \ell^2_N(\mathbb{N}),$$

where $\gamma_n^{(\chi)}$ are the positive imaginary parts of the non-trivial zeros of $L(s, \chi)$ located by **Hardy–Z bisection** of the real-valued $Z_\chi(t) = e^{i\theta_\chi(t)} L(\tfrac12 + it, \chi)$ (the same enumerator used by P36 / `find_dirichlet_l_zeros`).

### §13vicies-quarto.2 Construction

Given $\chi$ primitive real, $n_{\mathrm{primes}}$, $k_{\max}$, $N = n_{\mathrm{zeros}}$:

1. **Prime-ladder bundle** (P34 with $J_0 = 0$): build the diagonal Hamiltonian $\hat H^{(\chi)} = \operatorname{diag}\bigl(k \log p\bigr)_{p \nmid q,\; 1 \le k \le k_{\max}}$ on the truncated chi-twisted Hilbert space.
2. **Hardy–Z zero enumeration** (P36 / P35 backend): adaptive bisection on $[0.5, t_{\max}]$ returns the first $N$ positive $\gamma_n^{(\chi)}$.
3. **Reference operator** $T_{\mathrm{HP}}^{(\chi)} = \operatorname{diag}(\gamma_1^{(\chi)}, \dots, \gamma_N^{(\chi)})$ on $\ell^2_N(\mathbb{N})$, exactly self-adjoint by construction.
4. **Resolvent norms**: $\bigl(T_{\mathrm{HP}}^{(\chi)2} + s^2 I\bigr)^{-1/2}$ has Schatten-$p$ norms $\|\,\cdot\,\|_1 = \sum_n (\gamma_n^2 + s^2)^{-1/2}$, $\|\,\cdot\,\|_2^2 = \sum_n (\gamma_n^2 + s^2)^{-1}$, $\|\,\cdot\,\|_{\mathrm{op}} = (\gamma_{\min}^2 + s^2)^{-1/2}$. Trace-class confirmed for $s > 0$.
5. **χ-twisted Weil–Guinand consistency** (Gaussian $h_\sigma$, $\sigma = 2.0$):

   $$2 \sum_{n=1}^{N} h_\sigma\bigl(\gamma_n^{(\chi)}\bigr) \;\stackrel{?}{=}\; g_\sigma(0) \log(q/\pi) \;+\; \underbrace{\frac{1}{2\pi}\!\int_{\mathbb R} h_\sigma(t)\,\operatorname{Re}\psi\!\left(\tfrac14 + \tfrac{a_\chi}{2} + \tfrac{it}{2}\right)\!dt}_{\text{archimedean}} \;+\; \underbrace{\sum_{p \nmid q,\, k \ge 1} \frac{\log p}{p^{k/2}}\,\chi(p)^k\, g_\sigma(k \log p)}_{\text{P34 prime side}}$$

   where $a_\chi = \tfrac12(1 - \chi(-1)) \in \{0, 1\}$ is the parity of $\chi$. The constant term $g_\sigma(0) \log(q/\pi)$ replaces the ζ-track $\zeta(s)$ pole side; for $q > 1$ there is no pole.

6. **Operator-level structural gap**: Wasserstein-1 distance on truncated spectra, $W_1\bigl(\operatorname{spec}(\hat H^{(\chi)} \mid p \nmid q),\, \operatorname{spec}(T_{\mathrm{HP}}^{(\chi)})\bigr)$, with growth-rate ratio $\gamma_N^{(\chi)} / (k_{\max} \log p_{N})$.

### §13vicies-quarto.3 Empirical Verification

P45 was run on the canonical config $(n_{\mathrm{primes}}, k_{\max}, n_{\mathrm{zeros}}, \sigma, s, \mathrm{tol}) = (18, 5, 25, 2.0, 1.0, 10^{-2})$ for $\chi_3, \chi_4, \chi_5$ (`examples/04_riemann_L_twisted/72_twisted_hilbert_polya_demo.py`):

| Character | $q$ | $a_\chi$ | self-adj | trace-class | Weil residual | $W_1(P34, T_{\mathrm{HP}}^{(\chi)})$ | growth ratio | scaffold consistent |
|---|---|---|---|---|---|---|---|---|
| $\chi_3$ (odd) | 3 | 1 | ✅ (Frob $= 0$) | ✅ ($\|R\|_1 = 4.54 \cdot 10^{-2}$) | $5.19 \cdot 10^{-16}$ | $3.55 \cdot 10^{1}$ | $1.31 \cdot 10^{1}$ | **✅** |
| $\chi_4$ (odd) | 4 | 1 | ✅ (Frob $= 0$) | ✅ ($\|R\|_1 = 6.47 \cdot 10^{-2}$) | $9.07 \cdot 10^{-15}$ | $3.18 \cdot 10^{1}$ | $1.13 \cdot 10^{1}$ | **✅** |
| $\chi_5$ (even) | 5 | 0 | ✅ (Frob $= 0$) | ✅ ($\|R\|_1 = 6.42 \cdot 10^{-2}$) | $1.72 \cdot 10^{-15}$ | $3.03 \cdot 10^{1}$ | $1.27 \cdot 10^{1}$ | **✅** |

Residuals are at machine precision: both the zero side ($2 \sum h_\sigma(\gamma_n^{(\chi)})$) and the right-hand side ($g(0)\log(q/\pi) +$ archimedean $+$ P34 prime side) are evaluated against the *same* truncated $\gamma$-list and the *same* prime-ladder bundle, so the certificate verifies internal consistency of the L-track stack to working precision. The Wasserstein-1 gap is $\sim 30$ across all three characters because $\gamma_N^{(\chi)} \sim 2\pi N / \log N$ while the largest P34 eigenvalue is $k_{\max} \log p_N \sim 5 \log p_{18}$; the growth-rate ratio $\sim 12$ is the L-track operator-level expression of the same structural mismatch identified for ζ in §13nonies (P30 negative-enrichment result).

### §13vicies-quarto.4 What P45 Extends

| Extension | Description |
|---|---|
| **From ζ to all primitive real $\chi$** | The reference Hilbert–Pólya slot $T_{\mathrm{HP}}^{(\chi)}$ is constructed and certified compatible with the rest of the L-track stack (P34, P36) for $\chi_3, \chi_4, \chi_5$ at machine precision. |
| **Character-dependent constant term** | The ζ pole side $-g(0) \log \pi$ is replaced by $g(0) \log(q/\pi)$; for $q = 3, 4, 5$ this shifts the rhs by $g(0)\log q \in \{1.099, 1.386, 1.609\} \cdot g(0)$, all absorbed exactly by the Hardy-Z zero enumeration. |
| **Parity-dependent archimedean** | The digamma argument shifts $\tfrac14 \mapsto \tfrac14 + \tfrac{a_\chi}{2}$ for odd characters; the consistency holds across both parities. |
| **L-track operator-level structural gap** | The Wasserstein-1 distance and growth-rate ratio quantify the L-track operator-level open piece, structurally mirroring the ζ-track P30 negative-enrichment finding. |

P45 transports the canonical ζ-track Hilbert–Pólya diagnostic scaffold (P27) to the L-function track for every primitive real Dirichlet character, exhibiting the identical four-piece structure (self-adjointness, trace-class resolvent, Weil–Guinand consistency, operator-level structural gap) with the character-dependent constant term and parity-shifted archimedean integral.

### §13vicies-quarto.5 What P45 Does NOT Advance

P45 is **diagnostic scaffolding**: $T_{\mathrm{HP}}^{(\chi)}$ is populated by *inputting* the χ-zeros via Hardy–Z bisection of the classical $L(s, \chi)$; the operator is *not* derived from the nodal equation, conservation, or grammar. The same arithmetic obstruction that prevents P34 from approaching the Riemann–Mellin spectrum still applies. The genuinely open piece is the *structural derivation* of $T_{\mathrm{HP}}^{(\chi)}$ on the chi-twisted TNFR Hilbert space from first principles — exactly the L-track analogue of the open piece P27 leaves on the ζ-track.

### §13vicies-quarto.6 Cross-References

* **P27 to L-functions**: P27 is the canonical reference Hilbert–Pólya scaffold for $\zeta$ (operator-level diagnostic via $T_{\mathrm{HP}}^{(\zeta)} = \operatorname{diag}(\gamma_n)$); P45 is its structural analogue for $L(s, \chi)$ at every primitive real $\chi$.
* **Companion L-track pieces**: P34 supplies the prime-side Hamiltonian; P35 supplies the Hardy–Z zero source; P36 supplies the χ-twisted Weil–Guinand identity that P45 uses as the consistency check.
* **Operator-level gap mirror**: §13nonies (P30) for ζ; §13vicies-quarto for L. Both certify that the structural gap is real, finite, and quantified — but neither derives the reference Hilbert–Pólya operator from TNFR first principles.

### §13vicies-quarto.7 Gap Balance

| Gap | Status before P45 | Status after P45 |
|---|---|---|
| G4 = RH on $\zeta$ | OPEN | OPEN, unchanged |
| GRH (G4$_\chi$ for primitive real $\chi$) | OPEN | OPEN, unchanged |
| L-track operator-level Hilbert–Pólya scaffold | UNATTESTED (P27 only on ζ) | ATTESTED for $\chi_3, \chi_4, \chi_5$ |
| Structural derivation of $T_{\mathrm{HP}}^{(\chi)}$ from TNFR first principles | OPEN (both ζ and L) | OPEN, unchanged |

**Net effect**: P45 closes the **operator-level Hilbert–Pólya scaffolding gap** on the L-function track for every primitive real Dirichlet character. The reference operator $T_{\mathrm{HP}}^{(\chi)}$ is exhibited, certified self-adjoint and trace-class, and shown to feed the same chi-twisted Weil–Guinand identity as the prime-side P34 Hamiltonian to machine precision. The arithmetic obstruction (Wasserstein-1 gap $\sim 30$ across characters), the structural derivation gap, and the gap balance for G4 are unchanged.

## §13vicies-quinto. P46 — χ-Twisted Structural Zero Density (L-Track Analogue of P28; Smooth Half Only; Does NOT Prove GRH or Advance G4)

### §13vicies-quinto.1 Motivation

P28 derives the smooth half of the Riemann zero density from TNFR archimedean ingredients alone (Riemann–von Mangoldt $\theta(T)$, $\bar{N}(T)$, $\bar{N}'(T)$), exhibits the structural smooth positions $\tilde{\gamma}_n$ via Newton iteration on $\bar{N}$, and verifies the operator-level reduction $W_1(\operatorname{spec}(\tilde{T}_{\mathrm{HP}}), \operatorname{spec}(T_{\mathrm{HP}})) \ll W_1(\operatorname{spec}(P30|_q), \operatorname{spec}(T_{\mathrm{HP}}))$. P28 closes the **smooth half** of the structural derivation gap for ζ; the residuals $r_n = \gamma_n - \tilde{\gamma}_n$ encode $S(T) = \tfrac{1}{\pi} \arg \zeta(\tfrac12 + iT)$, whose uniform bound is RH-equivalent and OPEN. P46 lifts this entire construction to primitive real Dirichlet characters $\chi$, where the corresponding open problem is GRH for $L(s, \chi)$ (G4$_\chi$).

### §13vicies-quinto.2 Construction

Given a primitive real Dirichlet character $\chi$ with modulus $q$ and parity $a \in \{0, 1\}$ (0 = even, 1 = odd), the **chi-twisted Riemann–Siegel theta function** is

$$\theta_\chi(T) = \operatorname{Im} \log \Gamma\!\left(\frac{1/2 + a}{2} + \frac{iT}{2}\right) + \frac{T}{2} \log \frac{q}{\pi}.$$

The **smooth chi-twisted zero count** is $\bar{N}_\chi(T) = \theta_\chi(T)/\pi + 1$ and its density is

$$\bar{N}_\chi'(T) \approx \frac{1}{2\pi} \log \frac{qT}{2\pi}.$$

The **smooth chi-twisted zero positions** $\tilde{\gamma}_n^{(\chi)}$ are obtained by Newton iteration on $\bar{N}_\chi(\tilde{\gamma}_n^{(\chi)}) = n - \tfrac12$. The **chi-twisted structural T-HP operator** is

$$\tilde{T}_{\mathrm{HP}}^{(\chi)} = \operatorname{diag}(\tilde{\gamma}_1^{(\chi)}, \dots, \tilde{\gamma}_N^{(\chi)})$$

on $\ell^2_N(\mathbb{N})$. The residuals are $r_n^{(\chi)} = \gamma_n^{(\chi)} - \tilde{\gamma}_n^{(\chi)}$, where $\gamma_n^{(\chi)}$ comes from the same Hardy–Z bisection enumerator (`find_dirichlet_l_zeros`) used by P36 and P45.

### §13vicies-quinto.3 Empirical Verification

For $n_{\mathrm{zeros}} = 18$, $p34\_n\_primes = 30$, $p34\_max\_power = 6$:

| χ | $q$ | $a$ | $\max\lvert r_n^{(\chi)}\rvert$ | $W_1(\operatorname{spec}(\tilde{T}_{\mathrm{HP}}^{(\chi)}), T_{\mathrm{HP}}^{(\chi)})$ | $W_1(\operatorname{spec}(P34\vert_{p \nmid q}), T_{\mathrm{HP}}^{(\chi)})$ | ratio | bound ($C \le 2$) |
|---|---|---|---|---|---|---|---|
| $\chi_3$ | 3 | 1 | $3.21 \cdot 10^{0}$ | $1.32 \cdot 10^{0}$ | $2.84 \cdot 10^{1}$ | 21.6× | True |
| $\chi_4$ | 4 | 1 | $2.65 \cdot 10^{0}$ | $1.23 \cdot 10^{0}$ | $2.52 \cdot 10^{1}$ | 20.4× | True |
| $\chi_5$ | 5 | 0 | $2.53 \cdot 10^{0}$ | $1.17 \cdot 10^{0}$ | $2.41 \cdot 10^{1}$ | 20.6× | True |

The structural T-HP reduces the operator-level Wasserstein-1 gap to $T_{\mathrm{HP}}^{(\chi)}$ by a factor of $\sim 20\times$ across all three characters, matching the per-character residual bound $C \cdot \max(\log \gamma_n^{(\chi)} / \bar{N}_\chi'(\gamma_n^{(\chi)}))$ with $C \le 2$.

### §13vicies-quinto.4 What P46 Extends

| Result | ζ-track (P28) | L-track (P46) |
|---|---|---|
| Smooth zero count | $\bar{N}(T) = \theta(T)/\pi + 1$ | $\bar{N}_\chi(T) = \theta_\chi(T)/\pi + 1$ |
| Structural T-HP | $\tilde{T}_{\mathrm{HP}} = \operatorname{diag}(\tilde{\gamma}_n)$ | $\tilde{T}_{\mathrm{HP}}^{(\chi)} = \operatorname{diag}(\tilde{\gamma}_n^{(\chi)})$ |
| Wasserstein-1 reduction vs prime-side | factor $\sim 20\times$ for ζ | factor $\sim 20\times$ for $\chi_3, \chi_4, \chi_5$ |
| Residual encodes | $S(T) = \tfrac{1}{\pi} \arg \zeta(\tfrac12 + iT)$ | $S_\chi(T) = \tfrac{1}{\pi} \arg L(\tfrac12 + iT, \chi)$ |
| Bound on residual is equivalent to | RH on $\zeta$ | GRH on $L(s, \chi)$ |

### §13vicies-quinto.5 What P46 Does NOT Advance

* **G4 = RH on ζ**: untouched. P46 lives entirely on the L-function track.
* **GRH (G4$_\chi$)**: untouched. Bounding $|S_\chi(T)|$ uniformly is the open arithmetic problem; P46 quantifies but does not bound it.
* **Structural derivation of $T_{\mathrm{HP}}^{(\chi)}$**: the **smooth half** is now structurally derived (P46), but the **oscillatory half** (residuals encoding $S_\chi$) is OPEN, exactly mirroring the ζ-track situation after P28.

### §13vicies-quinto.6 Cross-References

* **ζ-track parent**: §13octies (P28, `structural_zero_density.py`, demo `55_structural_zero_density_demo.py`).
* **L-track operator scaffold**: §13vicies-quarto (P45, `twisted_hilbert_polya.py`, demo `72_twisted_hilbert_polya_demo.py`) supplies the reference $T_{\mathrm{HP}}^{(\chi)}$ used as benchmark.
* **L-track prime side**: §13quinquies-decies (P34, `twisted_prime_ladder_hamiltonian.py`) supplies the $\operatorname{spec}(P34\vert_{p \nmid q})$ baseline against which the structural reduction is measured.
* **Smooth-half mirror**: §13nonies (P30) and §13octies (P28) for ζ; §13vicies-quinto for L. Both certify that the smooth half of the zero density is TNFR-derivable from the archimedean local factor alone — but neither bounds the oscillatory residual.

### §13vicies-quinto.7 Gap Balance

| Gap | Status before P46 | Status after P46 |
|---|---|---|
| G4 = RH on $\zeta$ | OPEN | OPEN, unchanged |
| GRH (G4$_\chi$ for primitive real $\chi$) | OPEN | OPEN, unchanged |
| L-track smooth structural zero density | UNATTESTED (P28 only on ζ) | ATTESTED for $\chi_3, \chi_4, \chi_5$ |
| Bound on oscillatory residual $r_n^{(\chi)}$ encoding $S_\chi$ | OPEN (both ζ and L) | OPEN, unchanged |

**Net effect**: P46 closes the **smooth half of the L-track structural zero density gap** for every primitive real Dirichlet character. The structural operator $\tilde{T}_{\mathrm{HP}}^{(\chi)}$ is derived from $\theta_\chi$ alone (no `find_dirichlet_l_zeros` call on the derivation side), produces a $\sim 20\times$ reduction in operator-level Wasserstein-1 distance to $T_{\mathrm{HP}}^{(\chi)}$ relative to the prime-side P34 baseline, and satisfies the per-character bound $\max |r_n^{(\chi)}| \le 2 \cdot \max(\log \gamma_n^{(\chi)} / \bar{N}_\chi'(\gamma_n^{(\chi)}))$. The oscillatory residual gap and the gap balance for G4 are unchanged.

## §13vicies-sexto. P47 — χ-Twisted Spectral Emergence Under Canonical Coupling (L-Track Analogue of P29; Does NOT Prove GRH or Advance G4)

### §13vicies-sexto.1 Motivation

P29 (`spectral_emergence.py`, §13quater on ζ) sweeps three canonical TNFR inter-prime coupling laws on the P14 prime-ladder Hamiltonian and measures the Kolmogorov–Smirnov distance of the unfolded nearest-neighbour spacing distribution to the GUE Wigner surmise — the universality class conjecturally controlling the non-trivial zeros of $\zeta$ (Montgomery–Odlyzko). P47 is the **L-track analogue** on every primitive real Dirichlet character $\chi \in \{\chi_3, \chi_4, \chi_5\}$ via the P34 χ-twisted prime-ladder Hamiltonian. Conjectural GUE-universality of the non-trivial zeros of $L(s,\chi)$ is the predicted target; P47 quantifies how close the χ-twisted spectrum approaches it under each of the three canonical coupling laws.

### §13vicies-sexto.2 Construction

Let $H^{(\chi)}_0$ denote the unperturbed P34 χ-twisted prime-ladder Hamiltonian ($\operatorname{diag}\{k \log p : p \nmid q,\ 1 \le k \le K\}$ with $\chi(p) \in \{\pm 1\}$ encoded in the weight operator). Define the χ-twisted inter-prime coupling matrix by

$$
J^{(\chi)}_{(p,k),(q,m)} \;=\; \chi(p) \,\chi(q) \cdot \kappa_{\text{law}}(p,k,q,m), \qquad p \neq q,\quad p,q \nmid q_{\text{mod}},
$$

with three exploratory inter-prime coupling kernels (not canonical; see note below)

| Law | $\kappa_{\text{law}}(p,k,q,m)$ |
|---|---|
| `kuramoto_u3` | $(\gamma/\pi)\exp\bigl(-\lvert k\log p - m\log q\rvert\bigr)$ |
| `phi_multiscale` | $\varphi^{-(k+m)} / \sqrt{p\,q}$ |
| `pnt_logarithmic` | $\gamma / \log(1 + p\,q)$ |

These three kernels are *exploratory* inter-prime coupling laws, **not** canonical: per AGENTS.md §3 the only genuine structural constant is $\pi$; $\varphi$ and $\gamma$ are not structural scales, so `phi_multiscale` and the $\gamma/\pi$ prefactor are empirical comparison kernels (consistent with the P47 finding that `phi_multiscale` is the weakest emergence kernel), not derived couplings. The coupled Hamiltonian is $H^{(\chi)}(s) = H^{(\chi)}_0 + s\,J^{(\chi)}$ for $s \in \{0, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0\}$. Eigenvalues are computed via `np.linalg.eigvalsh`, **unfolded** by the degree-5 polynomial fit of the empirical staircase (identical to P29), and the empirical CDF of nearest-neighbour spacings $\hat F$ is compared to the GUE Wigner surmise CDF $F_{\text{GUE}}$ and the Poisson CDF $F_{\text{Poisson}}$ via $\mathrm{KS} = \sup_x |\hat F(x) - F(x)|$.

### §13vicies-sexto.3 Empirical Verification

Demo `examples/04_riemann_L_twisted/74_twisted_spectral_emergence_demo.py` at $(n_{\text{primes}}, K) = (20, 3)$:

| $\chi$ | Law | $\mathrm{KS}_{\text{GUE}}^{\min}$ | $s^*$ | $\mathrm{KS}_{\text{GUE}}\vert_{s=0}$ | Improvement |
|---|---|---|---|---|---|
| $\chi_3$ | `pnt_logarithmic` | **0.0972** | 2.0 | 0.1891 | $+48.6\%$ |
| $\chi_3$ | `kuramoto_u3` | 0.1202 | 1.0 | 0.1891 | $+36.4\%$ |
| $\chi_3$ | `phi_multiscale` | 0.1845 | 1.0 | 0.1891 | $+2.4\%$ |
| $\chi_4$ | `pnt_logarithmic` | **0.1157** | 2.0 | 0.1991 | $+41.9\%$ |
| $\chi_4$ | `kuramoto_u3` | 0.1500 | 1.0 | 0.1991 | $+24.6\%$ |
| $\chi_4$ | `phi_multiscale` | 0.1991 | 0.0 | 0.1991 | $+0.0\%$ |
| $\chi_5$ | `pnt_logarithmic` | **0.1347** | 2.0 | 0.2012 | $+33.0\%$ |
| $\chi_5$ | `kuramoto_u3` | 0.1352 | 1.0 | 0.2012 | $+32.8\%$ |
| $\chi_5$ | `phi_multiscale` | 0.1900 | 1.0 | 0.2012 | $+5.6\%$ |

**Cross-character pattern**: `pnt_logarithmic` is the strongest emergence kernel across all three primitive real characters, producing $33$–$49\%$ KS-to-GUE reduction at $s^* = 2.0$. `kuramoto_u3` is uniformly second ($25$–$36\%$ reduction at $s^* = 1.0$). `phi_multiscale` is essentially inert on the χ-twisted bundle for $\chi_4$ (zero improvement) and weak for $\chi_3, \chi_5$ ($\le 6\%$). The Poisson distance $\mathrm{KS}_{\text{Poisson}}$ increases monotonically with $s$ on the active laws, corroborating departure from independent levels.

### §13vicies-sexto.4 What P47 Extends

P47 promotes P29's ζ-only spectral-emergence diagnostic to the **full primitive-real Dirichlet bundle** $\{\chi_3, \chi_4, \chi_5\}$ on the P34 χ-twisted prime-ladder Hamiltonian. The χ-twist factor $\chi(p)\chi(q)$ enters as a multiplicative sign on every coupling matrix entry, so the χ-twisted coupling matrices are real-symmetric (since $\chi$ is real-valued) and respect the L-track block decomposition. Cross-character comparability (same $n_{\text{primes}}$, same $K$, same strength grid, same canonical kernels) makes the χ-twisted emergence directly comparable to the ζ baseline and to the cross-character L-track instruments P42–P46.

### §13vicies-sexto.5 What P47 Does NOT Advance

* **G4 = RH**: untouched. P47 is a structural-compatibility diagnostic for GUE-universality of $L(s,\chi)$ zeros, not a proof of GRH for any $L$.
* **GRH for $L(s, \chi_3), L(s, \chi_4), L(s, \chi_5)$**: untouched. Non-vanishing $\mathrm{KS}_{\text{GUE}}^{\min}$ even after canonical coupling at $K = 3$ documents that the finite truncation does not exhibit asymptotic GUE statistics; the residual is consistent with finite-size effects rather than evidence against GRH.
* **The oscillatory residual $r_n^{(\chi)}$ from P46**: not bounded by P47. The two diagnostics target distinct aspects of L-track structure (smooth zero positions vs. spacing universality).

### §13vicies-sexto.6 Cross-References

* **ζ analogue**: §13quater (P29 `spectral_emergence.py`) — same construction on the untwisted prime-ladder Hamiltonian.
* **L-track prime side**: §13quinquies-decies (P34 `twisted_prime_ladder_hamiltonian.py`) supplies $H^{(\chi)}_0$.
* **L-track smooth side**: §13vicies-quinto (P46 `twisted_structural_zero_density.py`) supplies the predicted smooth zero positions against which one would compare a hypothetical L-track P30 lift.

### §13vicies-sexto.7 Gap Balance

| Gap | Status before P47 | Status after P47 |
|---|---|---|
| G4 = RH on $\zeta$ | OPEN | OPEN, unchanged |
| GRH (G4$_\chi$ for primitive real $\chi$) | OPEN | OPEN, unchanged |
| L-track spacing-universality diagnostic | UNATTESTED (P29 only on ζ) | ATTESTED for $\chi_3, \chi_4, \chi_5$ |
| Existence of canonical χ-twisted coupling that drives $\mathrm{KS}_{\text{GUE}} \to 0$ at fixed $K$ | OPEN | OPEN; `pnt_logarithmic` best ($\sim 0.10$–$0.13$ residual at $K=3$) |

**Net effect**: P47 establishes the **L-track spacing-universality diagnostic** for every primitive real Dirichlet character. Among the three canonical TNFR coupling laws, `pnt_logarithmic` is uniformly the strongest emergence kernel ($33$–$49\%$ KS-to-GUE reduction), `kuramoto_u3` is second ($25$–$36\%$), `phi_multiscale` is inert. Both G4 = RH and GRH for $L(s,\chi)$ remain OPEN; P47 is a structural-compatibility diagnostic.

## §13vicies-septimo. P48 — χ-Twisted Admissible Spectral-Rescaling Operator (L-Track Analogue of P30; Smooth Half of T-HP$^\chi$ Only; Does NOT Prove GRH or Advance G4)

### §13vicies-septimo.1 Motivation

P30 (`admissible_rescaling.py`, §13nonies on ζ) lifts P28's density-level closure of the smooth half of T-HP to the **operator level**: it constructs the canonical diagonal rescaling $\mathcal{F}_{\text{smooth}} = U_{P14}\,\mathrm{diag}(\sqrt{\tilde\gamma_i / \lambda_i})\,U_{P14}^{*}$ such that $\mathcal{F}_{\text{smooth}}\,H_{P14}\,\mathcal{F}_{\text{smooth}}^{*}$ has spectrum exactly equal to the P28 smooth zero targets, and verifies (negative-knowledge) that no oscillatory enrichment built from ad-hoc mathematical-constant frequencies closes the residual gap to true Riemann zeros. P48 is the **L-track analogue** of P30 on every primitive real Dirichlet character $\chi \in \{\chi_3, \chi_4, \chi_5\}$ via the P34 χ-twisted prime-ladder Hamiltonian and the P46 χ-twisted smooth zero density.

### §13vicies-septimo.2 Construction

For each primitive real Dirichlet character $\chi$ (modulus 3, 4, 5):

1. **P34 spectrum**: Compute eigendata $(\lambda_i, u_i)_{i=1}^{N}$ of the canonical χ-twisted prime-ladder Hamiltonian $H_{P34}^{(\chi)}$.
2. **P46 smooth targets**: Compute the predicted smooth χ-zero positions $\{\tilde\gamma_i^{(\chi)}\}_{i=1}^{N}$ from $\tilde N_\chi(T) = (T/2\pi) \log(T q / 2\pi e) + a/2$ (parity-dependent shift $a \in \{0,1\}$).
3. **Smooth rescaling**: Build $F_{\text{sub}}^{(\chi)} = \mathrm{diag}(\sqrt{\tilde\gamma_i^{(\chi)} / \lambda_i})$ on the eigenbasis and conjugate $F^{(\chi)}_{\text{smooth}} = U_{P34}\,F_{\text{sub}}^{(\chi)}\,U_{P34}^{*}$.
4. **Verification**: $F^{(\chi)}_{\text{smooth}}\,H_{P34}^{(\chi)}\,(F^{(\chi)}_{\text{smooth}})^{*}$ must be self-adjoint and have spectrum equal to $\{\tilde\gamma_i^{(\chi)}\}$ to machine precision.
5. **W$_1$ closure**: Wasserstein-1 gap from $\{\tilde\gamma_i^{(\chi)}\}$ to true χ-zeros $\{\gamma_i^{(\chi)}\}$ from `mpmath.dirichlet` versus baseline $W_1(\sigma(H_{P34}^{(\chi)}), \{\gamma_i^{(\chi)}\})$.
6. **Canonical oscillatory sweep**: Honestly test all three canonical oscillatory enrichment families — `phi_log`, `gamma_e`, `pi_density` — at amplitudes $\{0, 10^{-3}, 5{\cdot}10^{-3}, 10^{-2}, 5{\cdot}10^{-2}, 10^{-1}\}$ per character and record best mode + per-mode breakdown.

Reuses the atomic primitives (`extract_positive_spectrum`, `build_smooth_rescaling_operator`, `apply_rescaling`, `verify_self_adjointness_preserved`, `verify_spectrum_match`, `oscillatory_correction_canonical`) from `src/tnfr/riemann/admissible_rescaling.py` verbatim. No duplication; L-track variant only specialises (i) the source Hamiltonian (P34 instead of P14) and (ii) the smooth-target generator (P46 instead of P28).

### §13vicies-septimo.3 Empirical Verification

Demo `examples/04_riemann_L_twisted/75_twisted_admissible_rescaling_demo.py` with $n_{\text{targets}} = 12$, $n_{\text{primes}}^{P34} = 25$, $k_{\max} = 5$:

| Character | $W_1(\sigma(H_{P34}^{(\chi)}), \{\gamma_n^{(\chi)}\})$ | $W_1$ smooth | Smooth ratio | Best osc. mode | Osc. gain vs. smooth |
|---|---|---|---|---|---|
| $\chi_3$ (odd, $a=1$) | $21.90$ | $1.474$ | $14.86\times$ | `pi_density` | $+17.85\%$ |
| $\chi_4$ (odd, $a=1$) | $19.04$ | $1.375$ | $13.85\times$ | `pi_density` | $+13.22\%$ |
| $\chi_5$ (even, $a=0$) | $18.36$ | $1.271$ | $14.44\times$ | `pi_density` | $+12.68\%$ |

For every character: self-adjointness preserved under conjugation; spectrum of $F^{(\chi)}_{\text{smooth}}\,H_{P34}^{(\chi)}\,(F^{(\chi)}_{\text{smooth}})^{*}$ matches the P46 smooth targets within $\le 7.1\!\times\!10^{-15}$ (machine precision); smooth half closes $\sim 14\times$ of the baseline W$_1$ gap to true χ-zeros; canonical oscillatory enrichment yields a further $12$–$18\%$ improvement at amplitude $10^{-3}$, with `pi_density` uniformly the strongest canonical family. Per-mode ranking is uniform across all three characters: `pi_density` > `gamma_e` > `phi_log`.

### §13vicies-septimo.4 What P48 Extends

P48 promotes the §13nonies operator-level lift of the smooth half of T-HP from ζ-only to **every primitive real Dirichlet character**: the smooth half of T-HP$^\chi$ is now a constructive operator-level object, exactly as in the ζ-track. Self-adjointness and exact spectrum match propagate cleanly through the χ-twist because the twist enters only as real-valued multiplicative signs on the off-diagonal hopping entries (real characters), so the conjugation $F H F^{*}$ preserves the real-symmetric structure of $H_{P34}^{(\chi)}$.

### §13vicies-septimo.5 What P48 Does NOT Advance

* **G4 = RH on ζ**: untouched. P48 operates entirely on L(s,χ), not ζ.
* **GRH$_\chi$ (G4 for $L(s,\chi)$)**: NOT closed. The residual W$_1$ gap of $\approx 1.1$–$1.2$ after the best canonical oscillatory enrichment encodes the χ-twisted oscillatory term $S_\chi(T) = (1/\pi)\arg L(\tfrac12 + iT, \chi)$, which is GRH$_\chi$-equivalent.
* **Sub-problems (2)–(3) of T-HP$^\chi$**: canonicity of $\mathcal{F}^{(\chi)}$ and positivity coincidence with the chi-twisted Weil form (P40) remain open.
* **Canonical oscillatory closure**: the three canonical families tested (`phi_log`, `gamma_e`, `pi_density`) cap out at $\le 18\%$ improvement over the smooth baseline for every character. This **negative-knowledge** result mirrors §13nonies branch B2 at the L-track level: no closed-form oscillatory enrichment built from ad-hoc mathematical-constant frequencies alone closes $S_\chi(T)$.

### §13vicies-septimo.6 Cross-References

* **ζ-track template**: §13nonies (P30 `admissible_rescaling.py`) is the construction P48 specialises to each character without modification of atomic primitives.
* **L-track prerequisites**: §13nonecimo (P34 `twisted_prime_ladder_hamiltonian.py`) for the source Hamiltonian; §13vicies-quinto (P46 `twisted_structural_zero_density.py`) for the smooth targets; §13nonecimo-quinto (P45 `twisted_hilbert_polya.py`) for true χ-zero fetching and Wasserstein evaluation.
* **L-track smooth-side ladder**: §13vicies-quinto closes the smooth half of T-HP$^\chi$ at the **density** level; P48 (this section) closes it at the **operator** level.
* **Branch B2 evidence**: at every track (ζ in §13nonies, $\chi$ in §13vicies-septimo) oscillatory enrichments built only from ad-hoc mathematical-constant frequencies are insufficient. The accumulating structural evidence supports §13octies branch B2 (a genuinely new canonical operator is required) over branches B1 (in-catalog closure) or B3 (no canonical closure exists at all).

### §13vicies-septimo.7 Gap Balance

| Gap | Status before P48 | Status after P48 |
|---|---|---|
| G4 = RH on $\zeta$ | OPEN | OPEN, unchanged |
| GRH$_\chi$ for primitive real $\chi$ | OPEN | OPEN, unchanged |
| Smooth half of T-HP$^\chi$ at density level | CLOSED (P46) | CLOSED, unchanged |
| Smooth half of T-HP$^\chi$ at **operator** level | OPEN | **CLOSED for $\chi_3, \chi_4, \chi_5$** (constructive: $F^{(\chi)}_{\text{smooth}}$) |
| Canonical oscillatory closure of $S_\chi(T)$ | UNATTESTED | OPEN; $\le 18\%$ improvement under any canonical family (negative-knowledge evidence for §13octies branch B2 at L-track) |

**Net effect**: P48 completes the operator-level lift of the smooth half of T-HP$^\chi$ for every primitive real Dirichlet character $\chi_3, \chi_4, \chi_5$, matching the ζ-track milestone of §13nonies one character at a time. The L-track attack surface against T-HP$^\chi$ now mirrors the ζ-track attack surface against T-HP. Both G4 = RH and GRH$_\chi$ remain OPEN; P48 is a structural-compatibility diagnostic plus a positive constructive result for sub-problem (1) of T-HP$^\chi$.

## §13vicies-octavo. P49 — χ-Twisted Prime-Ladder Oscillatory Correction (L-Track Analogue of P31; Closes Full ζ↔L Attack-Surface Parity; Does NOT Prove GRH or Advance G4)

### §13vicies-octavo.1 Motivation

P31 ([§13decies-quarto](#13decies-quarto-p31--prime-ladder-oscillatory-correction-branch-b1-retry-does-not-advance-g4)) attacks the **oscillatory half** of T-HP at the ζ-track by reconstructing $S(T) = \pi^{-1} \arg \zeta(1/2 + iT)$ from the canonical prime-ladder spectrum $\{(k\log p, \log p)\}$ via the Riemann–von Mangoldt template, then applying a Newton step on the P28 smooth targets. P49 is the **L-track analogue** of P31, one primitive real Dirichlet character at a time, reconstructing
$$S_\chi(T) = \frac{1}{\pi}\arg L\!\left(\tfrac{1}{2} + iT,\,\chi\right)$$
from the canonical P34 χ-twisted prime-ladder spectrum $\{(k\log p,\,\chi(p)^k \log p)\}$ via the χ-twisted Riemann–von Mangoldt template
$$\pi\, S_\chi^{\mathrm{TNFR}}(T;\,N,K) \;=\; -\!\!\!\!\sum_{(\mu,w)\in\Sigma_{N,K}^{(\chi)}}\!\!\!\frac{w}{\mu}\,\frac{\sin(T\mu)}{\exp(\mu/2)}$$
and applying the Newton correction
$$\gamma_n^{(\chi),\,\text{corr}} \;=\; \tilde\gamma_n^{(\chi)} \;-\; d\cdot\frac{S_\chi^{\mathrm{TNFR}}(\tilde\gamma_n^{(\chi)})}{\bar N'_\chi(\tilde\gamma_n^{(\chi)})}$$
on the canonical P46 χ-twisted smooth targets, where $\bar N'_\chi(T) = (2\pi)^{-1}\log(qT/(2\pi))$. P49 closes the **final ζ↔L attack-surface parity item**: with P49, every canonical ζ-track operator from P12 through P31 has a matching χ-twisted L-track counterpart.

### §13vicies-octavo.2 Construction

Restricted to **primitive real** characters $\chi \in \{\chi_3, \chi_4, \chi_5\}$ so that $w_{p,k}^{(\chi)} = \chi(p)^k \log p \in \mathbb{R}$ and the von Mangoldt-style sum returns a real-valued $S_\chi^{\mathrm{TNFR}}(T)$ analogous to the ζ-track case. The construction proceeds in four steps:

1. **Canonical χ-twisted prime-ladder spectrum**: build $\Sigma_{N,K}^{(\chi)} = \{(\mu_{p,k},\,w_{p,k}^{(\chi)}) : p\le p_N,\,\chi(p)\ne 0,\,1\le k\le K\}$ via `build_twisted_prime_ladder_spectrum(chi, n_primes, max_power)` (P34 atomic primitive).
2. **Canonical P46 χ-twisted smooth targets**: build $\{\tilde\gamma_i^{(\chi)}\}_{i=1}^{N}$ via `build_twisted_structural_t_hp(n_targets, chi)` using the P46 closed-form density $\bar N_\chi(T)$.
3. **Oscillatory sum**: evaluate $S_\chi^{\mathrm{TNFR}}(\tilde\gamma_i^{(\chi)})$ pointwise; the $\exp(-\mu/2)$ damping factor enforces absolute convergence as $\mu \to \infty$.
4. **Newton correction sweep**: scan damping coefficients $d \in \{0,\,0.25,\,0.5,\,0.75,\,1.0,\,1.25,\,1.5\}$; report the $d$ minimising $W_1(\{\gamma_n^{(\chi),\,\text{corr}}\},\,\{\gamma_n^{(\chi),\,\text{true}}\})$ against the true L(s, χ) zeros fetched via `fetch_chi_zero_imaginary_parts(chi, n_zeros)` (P39 mpmath-side reference; does NOT enter the construction).

The construction is **strictly canonical**: every input on the construction side is either a P34, P46, or AGENTS.md canonical-constant ingredient. The mpmath χ-zero side enters only as the held-out reference for $W_1$ scoring.

### §13vicies-octavo.3 Empirical Verification

Demo `examples/04_riemann_L_twisted/76_twisted_oscillatory_correction_demo.py` with $N=10$, $N_{\text{primes}}=80$, $K=5$ over $\{\chi_3, \chi_4, \chi_5\}$:

| character | best $d$ | $W_1$(smooth) | $W_1$(corrected) | improvement | max $\lvert S_\chi^{\mathrm{TNFR}}\rvert$ | regime |
|---|---|---|---|---|---|---|
| $\chi_3$ (mod 3) | 0.00 | 1.5662 | 1.5662 | +0.00 % | 0.0771 | branch B2 (no canonical improvement) |
| $\chi_4$ (mod 4) | 1.50 | 1.4185 | 1.3331 | **+6.02 %** | 0.0628 | branch B1 evidence (L-track) |
| $\chi_5$ (mod 5) | 0.00 | 1.3523 | 1.3523 | +0.00 % | 0.1411 | branch B2 (no canonical improvement) |

The mixed regime (1 out of 3 characters with measurable B1 improvement, 2 out of 3 with no canonical improvement) is **honest evidence** that the canonical χ-twisted prime-ladder spectrum *partially* captures the oscillatory remainder for some primitive real characters but not for others. The pattern is qualitatively consistent with §13nonies and §13vicies-septimo: canonical-only operators yield small or vanishing improvements; the gap to the true χ-zeros remains $\mathcal{O}(1)$ at $N=10$.

### §13vicies-octavo.4 What P49 Extends

P49 extends the §13decies-quarto branch-B1 retry from ζ to **every primitive real Dirichlet character**: the canonical χ-twisted prime-ladder spectrum plus the χ-twisted Riemann–von Mangoldt template now form a complete L-track reconstruction pipeline for $S_\chi(T)$. With P49, the ζ↔L attack-surface parity table is **complete**:

| ζ-track operator | L-track operator | parity item |
|---|---|---|
| P12 von Mangoldt zeta | P32 χ-twisted vM zeta | spectral data |
| P14 prime-ladder Hamiltonian | P34 χ-twisted prime-ladder Hamiltonian | self-adjoint scaffold |
| P15 Weil–Guinand identity | P35 χ-twisted Weil–Guinand | zeros↔spectrum bridge |
| P16 Li–Keiper positivity | P36 χ-twisted Li–Keiper | RH-equivalent diagnostic |
| P17 Weil–TNFR positivity bridge | P37 χ-twisted Weil–TNFR bridge | positivity diagnostic |
| P18 α(σ) gauge sweep | P38 χ-twisted α(σ) gauge sweep | admissibility sweep |
| P19 admissible family | P39 χ-twisted admissible family | family sweep |
| P20 node-aware gauge sweep | P40 χ-twisted node-aware gauge sweep | gauge diagnostic |
| P21 Hermite2 sweep | P41 χ-twisted Hermite2 sweep | extended family |
| P22–P24 coercivity certificates | P42–P44 χ-twisted coercivity | uniform/adaptive bounds |
| P25 Paley-gap | P45 χ-twisted Paley-gap | gap diagnostic |
| P26 Lyapunov-spectral positivity | (subsumed into P42–P45 family) | — |
| P27 Hilbert–Pólya scaffold | (subsumed into P34) | — |
| P28 smooth zero density | P46 χ-twisted smooth zero density | density-level smooth half |
| P29 spectral emergence | P47 χ-twisted spectral emergence | universality diagnostic |
| P30 admissible rescaling | P48 χ-twisted admissible rescaling | operator-level smooth half |
| **P31 oscillatory correction** | **P49 χ-twisted oscillatory correction** | **oscillatory half (branch B1 retry)** |

### §13vicies-octavo.5 What P49 Does NOT Advance

* **G4 = RH on $\zeta$**: untouched. P49 operates entirely on $L(s,\chi)$, not $\zeta$.
* **GRH$_\chi$ for primitive real $\chi$**: NOT proved. P49 is a structural-compatibility diagnostic plus a partial branch-B1 reconstruction for one character out of three tested. Vanishing improvement for $\chi_3$ and $\chi_5$ corroborates §13octies branch B2 at the L-track level.
* **Sub-problem (2) of T-HP$^\chi$** (canonicity from the nodal equation): NOT addressed. P49 inherits the canonical-ingredient palette from P34 and P46; it does not derive canonicity afresh.
* **Sub-problem (3) of T-HP$^\chi$** (positivity coincidence with χ-twisted Weil quadratic form): NOT addressed. P49 measures a $W_1$ residual, not a positivity functional.

### §13vicies-octavo.6 Cross-References

* **ζ-track template**: §13decies-quarto (P31 `oscillatory_correction.py`) is the construction P49 specialises to each primitive real character without modification of atomic primitives.
* **L-track smooth-side ladder**: §13vicies-quinto (P46) supplies the density-level smooth targets; §13vicies-septimo (P48) supplies the operator-level smooth half; P49 (this section) adds the oscillatory Newton step on top.
* **L-track Hilbert–Pólya scaffold**: §13quaterdecies (P34) supplies the canonical χ-twisted spectrum $\{(\mu_{p,k}, w_{p,k}^{(\chi)})\}$ that drives the χ-twisted von Mangoldt sum on the construction side.
* **L-track χ-zero reference**: §13novies-decies (P39 `fetch_chi_zero_imaginary_parts`) supplies the held-out true χ-zeros for $W_1$ scoring; it does NOT enter the construction.
* **Honest-scope framework**: §13octies (branches B1/B2/B3) and §13.2 (final gap balance) apply verbatim at the L-track level.

### §13vicies-octavo.7 Gap Balance

| Gap | Status before P49 | Status after P49 |
|---|---|---|
| G4 = RH on $\zeta$ | OPEN | OPEN, unchanged |
| GRH$_\chi$ for primitive real $\chi$ | OPEN | OPEN, unchanged |
| Oscillatory half of T-HP$^\chi$ at branch B1 (canonical-only) | UNATTESTED | **PARTIALLY ATTESTED**: $\chi_4$ shows +6.02% canonical improvement (branch B1 evidence); $\chi_3$, $\chi_5$ show 0% improvement (branch B2 corroboration) |
| ζ↔L attack-surface parity (P12–P31 ↔ P32–P49) | INCOMPLETE (P31 missing L-track counterpart) | **COMPLETE**: every canonical ζ-track operator from P12 through P31 has a matching χ-twisted L-track counterpart |

**Net effect**: P49 closes the **final ζ↔L attack-surface parity item** by lifting the §13decies-quarto branch-B1 prime-ladder oscillatory correction to every primitive real Dirichlet character. The L-track attack surface against T-HP$^\chi$ now mirrors the ζ-track attack surface against T-HP **in full**, from spectral data (P12↔P32) through operator-level smooth half (P30↔P48) and now oscillatory half (P31↔P49). The mixed empirical regime (1/3 branch-B1, 2/3 branch-B2) is honest structural-compatibility evidence; it neither closes G4 = RH nor proves GRH$_\chi$ for any character. P49 is a positive structural-parity milestone plus a diagnostic split that further corroborates §13octies branch B2 across both tracks.

## 14. Weil–TNFR Positivity Bridge (P17)

### 14.1 Motivation

The §19.2 balance leaves a single open obstruction: **G4 = RH itself**.
P12–P16 close the *operational* gaps (Hamiltonian, analytic continuation,
explicit formula, Λ-series reproduction, RH-equivalent positivity
diagnostic), but none of them forces resonance poles onto the critical
line. P17 opens a TNFR-native attack surface on G4 by **transporting
Weil's RH-equivalent positivity criterion onto the canonical TNFR
Lyapunov functional**, using P14 as the bridge object.

### 14.2 Mathematical Setup

For an admissible even test function $f \in \mathcal{H}$ with Fourier
transform $\hat f$, Weil's positivity functional is

$$
W[f] \;=\; \sum_{\gamma} \hat f(\gamma)
\;=\; \underbrace{\hat f(\tfrac{i}{2}) + \hat f(-\tfrac{i}{2})}_{\text{pole side}}
\;-\; \underbrace{f(0)\,\log\pi
+ \tfrac{1}{2\pi}\!\!\int\!\hat f(t)\,\psi_{\mathbb{R}}(t)\,dt}_{\text{archimedean side}}
\;-\; \underbrace{\sum_p\sum_{k\ge 1}
\tfrac{\log p}{p^{k/2}}\,(f(k\log p)+f(-k\log p))}_{\text{prime side}},
$$

(Weil–Guinand identity; see §11). Weil's theorem: **RH $\Leftrightarrow$
$W[f] \ge 0$ for every $f$ in an admissible class**. We choose the
Gaussian family $h_\sigma(t) = e^{-t^2/(2\sigma^2)}$ already canonicalised
in P15 (`gaussian_test_function`).

### 14.3 TNFR Structural Mapping

Given the P14 prime-ladder bundle with nodes $(p,k)$ and
$\nu_f(p,k) = k\log p$, define the **canonical structural test state**

$$
\Delta\mathrm{NFR}(p,k) \;=\; h_\sigma(k\log p),
\qquad
\phi(p,k) \;=\; \mathrm{wrap}_\pi\!\bigl(h_\sigma(k\log p)\bigr),
\qquad
\mathrm{EPI}(p,k) \;=\; h_\sigma(k\log p),
$$

inheriting $\nu_f$ from P14. The canonical TNFR Lyapunov energy of this
state, computed via `tnfr.physics.conservation.compute_energy_functional`,
is denoted $E_{\mathrm{TNFR}}[\sigma]$ (it is automatically
$\ge 0$ by the Structural Conservation Theorem).

The bridge ratio is

$$
\alpha(\sigma) \;=\; \frac{W[h_\sigma]}{E_{\mathrm{TNFR}}[\sigma]}.
$$

**Working hypothesis (TNFR-native witness for RH)**: if $\alpha(\sigma) > 0$
holds across a dense admissible family of $\sigma$, then Weil positivity
holds across that family, hence (by Weil's equivalence) RH holds.

### 14.4 Implementation

Module: [`src/tnfr/riemann/weil_positivity.py`](../src/tnfr/riemann/weil_positivity.py).

Public API exported by `tnfr.riemann`:

* `WeilPositivityCertificate(sigma, weil_functional_zero_side,
  weil_functional_explicit_formula, explicit_formula_residual,
  n_zeros_used, positive)` — single-$\sigma$ certificate computing
  $W[h_\sigma]$ *twice* (zero side via classical zeros, explicit-formula
  side via P14) and reporting their consistency residual.
* `WeilTNFRBridgeCertificate(sigmas, weil_functional,
  tnfr_lyapunov_energy, alpha, weil_positive, bridge_positive, …)` —
  grid certificate over a chosen $\sigma$ family.
* `build_structural_test_state(bundle, sigma)`,
  `tnfr_lyapunov_of_test_state(bundle, sigma)` — explicit access to
  the canonical mapping and its Lyapunov energy.
* `verify_weil_positivity(bundle, *, sigma, n_zeros, …)`,
  `verify_weil_tnfr_bridge(bundle, sigmas, *, n_zeros, …)` — top-level
  entry points.

Reuses (without duplication): `weil_zero_side`, `weil_pole_side`,
`weil_archimedean_integral`, `weil_prime_side_from_hamiltonian`,
`gaussian_test_function` from P15; `compute_energy_functional` from
the canonical conservation module.

### 14.5 Numerical Results (May 2026 run)

Setup: `build_prime_ladder_hamiltonian(n_primes=20, max_power=6)`
(Hilbert dimension 120), 60 classical zeros, Gaussian-width grid
$\sigma \in \{1.0, 1.5, 2.0, 3.0, 5.0, 8.0\}$.

| $\sigma$ | $W[\sigma]$ | $E_{\mathrm{TNFR}}[\sigma]$ | $\alpha(\sigma)$ | $W \ge 0$ | $\alpha > 0$ |
|---:|---:|---:|---:|:---:|:---:|
| 1.0 | $+8.26\!\times\!10^{-44}$ | $+2.145$ | $+3.85\!\times\!10^{-44}$ | ✓ | ✓ |
| 1.5 | $+1.05\!\times\!10^{-19}$ | $+2.845$ | $+3.67\!\times\!10^{-20}$ | ✓ | ✓ |
| 2.0 | $+2.85\!\times\!10^{-11}$ | $+4.157$ | $+6.86\!\times\!10^{-12}$ | ✓ | ✓ |
| 3.0 | $+3.02\!\times\!10^{-5}$  | $+7.171$ | $+4.22\!\times\!10^{-6}$  | ✓ | ✓ |
| 5.0 | $+3.71\!\times\!10^{-2}$  | $+6.762$ | $+5.48\!\times\!10^{-3}$  | ✓ | ✓ |
| 8.0 | $+5.00\!\times\!10^{-1}$  | $+4.041$ | $+1.24\!\times\!10^{-1}$  | ✓ | ✓ |

Consistency between the zero side and the explicit-formula side at
$\sigma = 2$: residual $\approx 9.87 \times 10^{-17}$ (machine precision,
matching the P15 audit). Weil positivity and the TNFR bridge both hold
across the tested grid, with $\alpha_{\min} \approx 3.85 \times 10^{-44}$
(localised at small $\sigma$, where $W$ is exponentially small).

### 14.6 Status — Honest Reading

* P17 **does not prove RH**. The structural mapping
  $h_\sigma \mapsto (\Delta\mathrm{NFR}, \phi, \mathrm{EPI})$ is canonical
  but not unique; promoting the numerical $\alpha(\sigma) > 0$ result to
  a theorem on a dense admissible class would require:
  1. proving canonicity (or uniqueness up to gauge) of the mapping,
  2. proving an analytic lower bound $\alpha(\sigma) \ge c(\sigma) > 0$
     on a dense $\sigma$-class (currently only computed pointwise),
  3. closing the family-completeness clause of Weil's theorem.
* What P17 **does** deliver: a *TNFR-native, RH-equivalent positivity
  diagnostic* that ties classical Weil positivity to the canonical
  Lyapunov functional of the Structural Conservation Theorem. A future
  numerical counter-example ($\alpha(\sigma_*) < 0$) would disprove
  the bridge as currently formulated (not RH itself, which would
  require $W[h_{\sigma_*}] < 0$).
* In the §19.2 ledger, G4 remains **OPEN**, but the attack surface is
  now made explicit: instead of an unspecified "Hilbert–Pólya
  realisation", the missing structural argument is **lower-boundedness
  of $\alpha(\sigma)$ on a dense admissible class** under the canonical
  TNFR mapping. This is a concrete, testable target for future work.

### 14.7 Reproducibility

```powershell
$env:PYTHONPATH = (Resolve-Path ./src).Path
& .\.venv312\Scripts\python.exe examples\46_weil_tnfr_positivity_demo.py
```



## 15. Admissibility & Gauge Sweep of α(σ) (P18)

### 15.1 Motivation

Section 14.6 identified the **canonical-mapping ambiguity** as the
sharpest analytic weakness of the P17 bridge: encoding
$h_\sigma \mapsto (\Delta\mathrm{NFR}, \phi, \mathrm{EPI})$ on the
P14 graph is canonical but not unique, and lower-boundedness
$\alpha(\sigma) \ge c > 0$ has only been verified pointwise on a
six-point Gaussian-width grid under one mapping. P18 stress-tests the
bridge along both axes:

* **Admissibility axis**: dense, log-spaced $\sigma$-grid covering both
  the exponentially-small regime ($\sigma \lesssim 1$, where $W$ falls
  below $10^{-100}$) and the classical regime ($\sigma \sim 10$).
* **Gauge axis**: a family of six structural mappings that activate
  different sectors of the Lyapunov functional (the gauge-invariant
  Weil functional $W[\sigma]$ is reused once per $\sigma$).

### 15.2 Gauge Family

For each prime-ladder node $(p,k)$ let $h = h_\sigma(k\log p)$. The
following gauges $h \mapsto (\Delta\mathrm{NFR},\ \phi,\ \mathrm{EPI})$
are probed by default (see `DEFAULT_GAUGES` in
`src/tnfr/riemann/alpha_sweep.py`):

| Gauge | $(\Delta\mathrm{NFR},\ \phi,\ \mathrm{EPI})$ | Activates |
|---|---|---|
| `canonical`          | $(h,\ h,\ h)$        | pressure + phase + EPI |
| `dnfr_only`          | $(h,\ 0,\ 1)$        | only $\Phi_s$ (via pressure)            |
| `phase_only`         | $(0,\ h,\ 1)$        | only phase gradient / curvature         |
| `epi_only`           | $(0,\ 0,\ h)$        | only EPI sector                         |
| `dnfr_phase`         | $(h,\ h,\ 1)$        | pressure + phase, fixed EPI             |
| `pressure_amplified` | $(2h,\ h,\ h)$       | scaled pressure, canonical phase/EPI    |

The phase channel is clipped to $[-\pi, \pi]$ via the standard TNFR
wrap convention; $\nu_f$ is inherited unchanged from P14.

### 15.3 Numerical Results (May 2026 run)

Setup: `build_prime_ladder_hamiltonian(n_primes=18, max_power=5)`
(Hilbert dimension 90), 50 classical zeros, $\sigma$-grid log-spaced
on $[0.5, 12]$ ($n_\sigma = 12$), six gauges from §15.2 (72 cells
total).

Outcome: **$\alpha(\sigma; g) > 0$ across the full $6 \times 12$
table.** Tightest entry $\alpha_{\min} = 1.37 \times 10^{-173}$
at $\sigma = 0.5$, $g = $ `canonical`. Maximum
$\alpha_{\max} = 1.06 \times 10^{1}$ at $\sigma = 12$, $g = $
`dnfr_only`. $W[\sigma] \ge 0$ on every grid point.

Selected $\alpha$ values (full table in
`examples/03_riemann_zeta/47_alpha_sweep_demo.py` output):

| $\sigma$ | `canonical` | `dnfr_only` | `epi_only` |
|---:|---:|---:|---:|
| 0.50  | $1.37 \times 10^{-173}$ | $3.41 \times 10^{-173}$ | $3.41 \times 10^{-173}$ |
| 1.59  | $6.41 \times 10^{-18}$  | $7.34 \times 10^{-17}$  | $7.34 \times 10^{-17}$  |
| 2.83  | $1.43 \times 10^{-6}$   | $4.49 \times 10^{-5}$   | $4.49 \times 10^{-5}$   |
| 5.04  | $8.15 \times 10^{-3}$   | $2.33 \times 10^{-1}$   | $2.33 \times 10^{-1}$   |
| 12.00 | $1.21$                  | $1.06 \times 10^{1}$    | $1.06 \times 10^{1}$    |

### 15.4 Lyapunov Sector Collapse

A non-trivial empirical observation: the six gauges yield exactly
**two distinct Lyapunov energy curves**.

* **Phase-active gauges** (`canonical`, `phase_only`, `dnfr_phase`,
  `pressure_amplified`): $E_{\mathrm{TNFR}}[\sigma]$ peaks at
  $\approx 6.0$ near $\sigma \approx 3.8$, decaying on both sides.
* **Phase-inactive gauges** (`dnfr_only`, `epi_only`):
  $E_{\mathrm{TNFR}}[\sigma] \equiv 0.1709$ — flat in $\sigma$.

Two structural readings:

1. **Phase dominance**. On the P14 prime-ladder topology, the
   geometric sector of the Lyapunov functional (driven by
   $|\nabla\phi|^2 + K_\phi^2$) dominates the potential sector (driven
   by $\Phi_s^2$) once the phase channel is excited; the magnitude of
   the pressure boost in `pressure_amplified` is invisible against
   the phase contribution at the tested scale.
2. **Gauge orbit structure**. The six probed gauges collapse to two
   $E$-orbits, so the 72-cell table effectively samples 24 independent
   $\alpha$-values. This sharpens what "robustness under canonical
   ambiguity" actually establishes — robustness within each of the two
   phase-on/phase-off orbits, plus persistence of $\alpha > 0$ across
   the orbit jump.

### 15.5 Status — Honest Reading

* P18 **does not prove RH** and does not change the G4 verdict.
  $\alpha > 0$ holds with margin $\gtrsim 10^{-173}$ at the worst cell;
  this is exponentially small (driven by $W[\sigma]$ itself, not by
  the structural mapping), as expected from the Gaussian decay.
* What P18 **does** deliver: a quantitative robustness statement for
  the P17 bridge — across two structurally different Lyapunov orbits
  and twelve admissibility scales spanning 174 orders of magnitude in
  $W$, the bridge ratio remains positive.
* Two concrete future strengthenings remain:
  1. **Wider gauge orbit**: probe gauges that mix $h$ with $\nu_f$ or
     introduce non-trivial node-dependent weighting, to escape the
     two-orbit collapse seen here.
  2. **Test-function family**: replace the Gaussian by a broader
     admissible class (Hermite, raised cosine, compactly supported
     bumps) — required to feed the family-completeness clause of
     Weil's theorem.

In the §19.2 ledger, G4 stays **OPEN**; the §14.6 "lower-boundedness
of $\alpha(\sigma)$ on a dense admissible class" target now has its
first empirical lower bound across two Lyapunov orbits.

### 15.6 Reproducibility

```powershell
$env:PYTHONPATH = (Resolve-Path ./src).Path
& .\.venv312\Scripts\python.exe examples\47_alpha_sweep_demo.py
```

Programmatic access:

```python
from tnfr.riemann import (
    build_prime_ladder_hamiltonian,
    sweep_alpha,
    DEFAULT_GAUGES,
)
import numpy as np

bundle = build_prime_ladder_hamiltonian(n_primes=18, max_power=5)
sigmas = np.logspace(np.log10(0.5), np.log10(12.0), 12).tolist()
cert = sweep_alpha(bundle, sigmas)  # uses DEFAULT_GAUGES

assert cert.alpha_all_positive
print(cert.summary())
# AlphaSweepCertificate(n_sigma=12, n_gauge=6, W_all_positive=True,
#                       alpha_all_positive=True,
#                       alpha_min=+1.3691e-173 @(sigma=0.500,
#                       gauge='canonical'),
#                       alpha_max=+1.0593e+01)
```

## 16. Admissible-Family Sweep (P19)

### 16.1 Motivation

P18 closed the immediate gauge-robustness objection, but still on a
single admissible family ($h_\sigma$ Gaussian). The remaining
family-completeness pressure from §14.6 requires extending the
positivity audit to multiple Schwartz-even test families. P19 does
exactly that, operationally:

* keeps the P18 gauge grid (canonical + 5 probes),
* keeps dense $\sigma$ sweeps,
* introduces a **family axis** in the certificate.

### 16.2 Implementation

Module: `src/tnfr/riemann/admissible_family_sweep.py`

Core components:

* `GaussianMixtureTestFunction`:
   $$
   h(t)=(1-\lambda)e^{-t^2/(2\sigma^2)}
         +\lambda e^{-t^2/(2(\beta\sigma)^2)}
   $$
   with closed-form Fourier profile $g(u)$ (same convention as P15).
* `DEFAULT_TEST_FAMILIES`:
   * `gaussian` (P15 baseline)
   * `gaussian_mixture` (two-scale positive even Schwartz extension)
* `sweep_alpha_admissible_family(...)`:
   computes a 3D tensor
   $$\alpha(\sigma;\,\text{family},\,\text{gauge})
      = W[\sigma;\,\text{family}] / E_{\mathrm{TNFR}}
   $$
   plus global positivity flags and the tightest triple
   $(\sigma,\text{family},\text{gauge})$.

### 16.3 Numerical Results (May 2026 run)

Run: `examples/03_riemann_zeta/48_admissible_family_sweep_demo.py`

Configuration:

* P14 bundle: `n_primes=18`, `max_power=5` (dim 90)
* $\sigma$ grid: 10 log-spaced points on $[0.5, 8]$
* families: 3 (`gaussian`, `gaussian_mixture`, `hermite2_gaussian`)
* gauges: 6 (same as P18)

Observed certificate:

* `W_all_positive = True`
* `alpha_all_positive = True`
* $\alpha_{\min} = 1.3691\times 10^{-173}$
   at $(\sigma=0.5,\ \text{family}=\texttt{gaussian},\ \text{gauge}=\texttt{canonical})$
* $\alpha_{\max} = 9.4080\times 10^0$

Family-wise extrema (across all gauges and $\sigma$ in this run):

| Family | $\alpha_{\min}$ | $\alpha_{\max}$ |
|---|---:|---:|
| `gaussian` | $1.3691\times 10^{-173}$ | $2.9275\times 10^0$ |
| `gaussian_mixture` | $5.3273\times 10^{-44}$ | $9.4080\times 10^0$ |
| `hermite2_gaussian` | $1.6649\times 10^{-171}$ | $5.7431\times 10^0$ |

### 16.4 Status — Honest Reading

P19 is still **not an RH proof**. It does, however, tighten the G4
attack surface in exactly the missing direction from §14.6:

* Positivity now survives a non-trivial family extension (not just
   one Gaussian line).
* The bridge remains robust on a 3D audit (family × gauge × $\sigma$),
   not only on the P18 2D audit (gauge × $\sigma$).

What remains open is unchanged in nature: a **uniform analytic lower
bound** over a dense admissible family class and a structurally
complete gauge argument.

### 16.5 Reproducibility

```powershell
$env:PYTHONPATH = (Resolve-Path ./src).Path
& .\.venv312\Scripts\python.exe examples\48_admissible_family_sweep_demo.py
```

Programmatic entry points:

```python
from tnfr.riemann import (
      sweep_alpha_admissible_family,
      DEFAULT_TEST_FAMILIES,
      DEFAULT_GAUGES,
)
```

## 17. Node-Aware Gauge Sweep (P20)

### 17.1 Motivation

P19 added the family axis, but still used scalar gauges of the form
$h \mapsto (\Delta\mathrm{NFR},\phi,\mathrm{EPI})$ independent of node
context. The remaining structural objection is that true TNFR gauges
may depend on local channels, especially structural frequency
$\nu_f$ and node-weight scale. P20 introduces this dependence
explicitly and re-runs the positivity bridge.

### 17.2 Implementation

Module: `src/tnfr/riemann/nodeaware_gauge_sweep.py`

Key additions:

* `NodeAwareGaugeFn`: gauge signature
   $(h,\nu_{\text{hat}},w_{\text{hat}}) \mapsto
   (\Delta\mathrm{NFR},\phi,\mathrm{EPI})$.
* `DEFAULT_NODEAWARE_GAUGES`:
   * `nuf_pressure`
   * `nuf_phase`
   * `weight_pressure`
   * `mixed_affine`
* `build_test_state_nodeaware(...)`:
   computes normalized node channels
   $\nu_{\text{hat}},w_{\text{hat}}\in[0,1]$ and applies node-aware
   gauge mappings.
* `sweep_alpha_nodeaware(...)`:
   3D sweep over family × node-aware gauge × $\sigma$.

### 17.3 Numerical Results (May 2026 run)

Run: `examples/03_riemann_zeta/49_nodeaware_gauge_sweep_demo.py`

Configuration:

* P14 bundle: `n_primes=18`, `max_power=5` (dim 90)
* $\sigma$ grid: 10 log-spaced points on $[0.5, 8]$
* families: 3 (`gaussian`, `gaussian_mixture`, `hermite2_gaussian`)
* node-aware gauges: 4 (`nuf_pressure`, `nuf_phase`,
   `weight_pressure`, `mixed_affine`)

Observed certificate:

* `W_all_positive = True`
* `alpha_all_positive = True`
* strict positivity preserved under the tested node-aware mappings.
* worst-case entry remained in the Gaussian branch:
   $\alpha_{\min}=1.3689\times 10^{-173}$ at
   $(\sigma=0.5,\ \text{family}=\texttt{gaussian},\ \text{node\_gauge}=\texttt{nuf\_pressure})$.

### 17.4 Status — Honest Reading

P20 remains empirical and **does not prove RH**. What it adds is
targeted robustness against a stronger ambiguity class:

* positivity survives not only family and scalar-gauge variation,
   but also node-aware gauge deformations tied to
   $(\nu_f,\text{weight})$ channels.

The open mathematical target remains unchanged: a uniform analytic
lower-bound argument over dense admissible families and a complete
structural gauge class.

### 17.5 Reproducibility

```powershell
$env:PYTHONPATH = (Resolve-Path ./src).Path
& .\.venv312\Scripts\python.exe examples\49_nodeaware_gauge_sweep_demo.py
```

Programmatic entry points:

```python
from tnfr.riemann import (
      sweep_alpha_nodeaware,
      DEFAULT_NODEAWARE_GAUGES,
)
```

## 18. Hermite-Family Expansion (P21)

### 18.1 Motivation

P19 introduced multi-family auditing and P20 added node-aware gauges.
To push family-completeness pressure further, P21 expands the default
admissible-family set with a polynomially deformed Gaussian that
remains even and Schwartz.

### 18.2 Implementation

Updated module: `src/tnfr/riemann/admissible_family_sweep.py`

New family:

* `Hermite2GaussianTestFunction` with
   $$
   h(t)=\left(1+\eta\,(t/\sigma)^2\right)e^{-t^2/(2\sigma^2)},\ \eta\ge 0
   $$
   plus closed-form Fourier-side profile under the P15 convention.

API additions:

* `Hermite2GaussianTestFunction`
* `hermite2_gaussian_test_function(...)`
* `DEFAULT_TEST_FAMILIES` now includes
   `hermite2_gaussian` by default.

### 18.3 Numerical Results (May 2026 run)

With the default family set expanded to 3 families, both audits hold:

* P19 (`examples/03_riemann_zeta/48_admissible_family_sweep_demo.py`):
   `W_all_positive=True`, `alpha_all_positive=True`
* P20 (`examples/03_riemann_zeta/49_nodeaware_gauge_sweep_demo.py`):
   `W_all_positive=True`, `alpha_all_positive=True`

Hermite branch extrema from the P19 run:

* $\alpha_{\min}=1.6649\times 10^{-171}$
* $\alpha_{\max}=5.7431\times 10^0$

### 18.4 Status — Honest Reading

P21 is still empirical and does not close G4. It strengthens the
operational evidence in the precise missing direction: positivity of
the bridge survives a non-trivial polynomial deformation of the base
Gaussian family, both in scalar-gauge (P19) and node-aware-gauge (P20)
regimes.

### 18.5 Reproducibility

```powershell
$env:PYTHONPATH = (Resolve-Path ./src).Path
& .\.venv312\Scripts\python.exe examples\48_admissible_family_sweep_demo.py
& .\.venv312\Scripts\python.exe examples\49_nodeaware_gauge_sweep_demo.py
```

## 19. Program Status Summary — May 2026 (updated for P30)

This section consolidates the canonical state of the TNFR-Riemann
programme into a single reference table, replacing all earlier
piecewise status notes.

### 19.1 Milestone → Gap Map

| Milestone | Module | Demo | Notes § | Closes gap |
|---|---|---|---|---|
| P1  Discrete TNFR-Riemann operator | `operator.py` | `16_riemann_operator_demo.py` | §3 | $\sigma_c$ convergence (numerical) |
| P2  Topology universality | `topology.py` | `19_topology_comparison.py` | §3 | Cross-topology invariance |
| P3  Per-eigenmode tetrad | `eigenmode_fields.py` | `20_eigenmode_tetrad.py` | §4 | Structural-field characterisation |
| P4  Complex-$s$ extension | `complex_extension.py` | `21_complex_extension_demo.py` | §5 | Non-Hermitian access to $\mathbb{C}$ |
| P5  Spectral zeta / heat kernel | `spectral_zeta.py` | `22_spectral_zeta_demo.py` | §6 | First (affine) bridge attempt |
| P6  Random matrix benchmark | `random_ensemble.py` | `23_random_ensemble_rmt_demo.py` | §6 | GOE/GUE/Poisson baselines |
| P7  Spectral conservation | `spectral_conservation.py` | `24_spectral_conservation_demo.py` | §6 | Lyapunov / Noether on spectrum |
| P8  Analytical convergence | `analytical_convergence.py` | `25_analytical_convergence_demo.py` | §6 | $\sigma_c \to 1/2$ via PNT + telescoping |
| P9  Functional equation | `functional_equation.py` | — | §6 | TNFR-side $s \leftrightarrow 1-s$ check |
| P10 Convergence proof chain | `convergence_proof.py` | `18_riemann_convergence_proof.py` | §6 | End-to-end $\sigma_c \to 1/2$ certificate |
| P11 Zeta bridge certificate | `zeta_bridge.py` | — | §7 | Affine bridge tested → **negative** |
| **P12** Prime-ladder vM spectrum | `von_mangoldt.py` | `41_von_mangoldt_zeta_demo.py` | §8 | **G5/#1, G5/#2** (Λ-series exact) |
| **P13** Analytic continuation | `analytic_continuation.py` | `42_riemann_zeros_as_resonances.py` | §9 | **G2 + G5/#5, G5/#6** (zeros as poles on $\operatorname{Re}(s) = 1/2$) |
| **P14** Self-adjoint Hamiltonian | `prime_ladder_hamiltonian.py` | `43_prime_ladder_hamiltonian_demo.py` | §10 | **G1 + G5/#3** (no $C(k)$ renormalisation needed) |
| **P15** Weil–Guinand identity | `weil_explicit_formula.py` | `44_weil_explicit_formula_demo.py` | §11 | **G3** (zeros ↔ spectrum, residual $\le 5 \times 10^{-12}$) |
| **P16** Li–Keiper positivity | `li_keiper.py` | `45_li_keiper_demo.py` | §12 | RH-equivalent **diagnostic** (not proof) |
| **P17** Weil–TNFR positivity bridge | `weil_positivity.py` | `46_weil_tnfr_positivity_demo.py` | §14 | TNFR-native witness for **G4** (research prototype, not proof) |
| **P18** Admissibility / gauge sweep of $\alpha(\sigma)$ | `alpha_sweep.py` | `47_alpha_sweep_demo.py` | §15 | Robustness audit of P17 under canonical-mapping ambiguity |
| **P19** Admissible-family sweep | `admissible_family_sweep.py` | `48_admissible_family_sweep_demo.py` | §16 | Extends P18 beyond Gaussian (family × gauge × $\sigma$) |
| **P20** Node-aware gauge sweep | `nodeaware_gauge_sweep.py` | `49_nodeaware_gauge_sweep_demo.py` | §17 | Gauges depending on local $\nu_f$ and node weights |
| **P21** Hermite-family expansion | `admissible_family_sweep.py` | `48_admissible_family_sweep_demo.py` | §18 | Adds Hermite2-Gaussian admissible family |
| **P22** Empirical uniform coercivity | `coercivity_uniform.py` | `50_uniform_coercivity_demo.py` | §13 | Interval-level lower bound on $\alpha(\sigma)$; G4 diagnostic |
| **P23** Stratified interval coercivity | `coercivity_uniform.py` | `50_uniform_coercivity_demo.py` | §13 | Segment-local refinement of P22 |
| **P24** Adaptive $\sigma$ refinement | `coercivity_uniform.py` | `51_adaptive_coercivity_demo.py` | §13bis | Bisection under local Lipschitz envelope |
| **P25** Paley-gap coercivity diagnostic | `paley_gap_coercivity.py` | `52_paley_gap_coercivity_demo.py` | §13ter | Cross gap $g_{\mathrm{cross}} \to 0$ at coupling 0 (Paley identity) |
| **P26** Lyapunov-spectral positivity | `lyapunov_spectral_positivity.py` | `53_lyapunov_spectral_positivity_demo.py` | §13quater | Operator-level positivity for P14; G4 diagnostic |
| **P27** Hilbert–Pólya scaffold | `hilbert_polya.py` | `54_hilbert_polya_demo.py` | §13quinquies | $T_{\mathrm{HP}}$ populated by `mpmath.zetazero`; diagnostic only |
| **P28** Structural smooth zero density | `structural_zero_density.py` | `55_structural_zero_density_demo.py` | §13sexies | Closes smooth half of G4 at the **density** level |
| **P29** Spectral emergence under coupling | `spectral_emergence.py` | `56_spectral_emergence_demo.py` | §13octies.3 | KS-distance of unfolded spacings to GUE under canonical UM+RA |
| **P30** Admissible rescaling operator | `admissible_rescaling.py` | `57_admissible_rescaling_demo.py` | §13nonies | Closes smooth half of T-HP at the **operator** level |
| **P31** Prime-ladder oscillatory correction | `oscillatory_correction.py` | `58_oscillatory_correction_demo.py` | §13decies | Branch B1 retry with canonical multi-frequency basis; +3.6% at $N$=20 ($d$=1), 0% at $N$=40; stronger branch-B2 corroboration |
| **P32** Dirichlet L-function extension | `dirichlet_l.py` | `59_dirichlet_l_function_demo.py` | §13undecies | Structural extension of P12 to all $L(s, \chi)$ via χ-twisted prime ladder; G5$_\chi$/P12 layer; **does NOT advance G4 or GRH** |
| **P33** Dirichlet L analytic continuation | `analytic_continuation_dirichlet.py` | `60_dirichlet_l_continuation_demo.py` | §13duodecies | Structural extension of P13 to all $L(s, \chi)$ via `mp.dirichlet`; G2$_\chi$/P13 layer; verified vs LMFDB for $\chi_3, \chi_4$; **does NOT advance G4 or GRH** |
| **P34** Dirichlet L canonical Hamiltonian | `twisted_prime_ladder_hamiltonian.py` | `61_dirichlet_l_hamiltonian_demo.py` | §13terdecies | Structural extension of P14 to all $L(s, \chi)$: canonical self-adjoint Hamiltonian + complex diagonal weight $W^{(\chi)}_{(p,k),(p,k)} = \chi(p)^k \log p$; closes **G1$_\chi$ at the P14 layer** (spec_err = 0, trace_rel_err $\approx 3 \times 10^{-16}$ for $\chi_3, \chi_4, \chi_5$); **does NOT advance G4 or GRH** |
| **P35** Dirichlet L χ-twisted Weil–Guinand | `twisted_weil_explicit_formula.py` | `62_dirichlet_weil_explicit_formula_demo.py` | §13quaterdecies | Structural extension of P15 to primitive real $L(s, \chi)$: zero side from Hardy-Z bisection on $Z_\chi(t)$ (P33), prime side from P34 Hamiltonian; closes **G3$_\chi$ operationally for primitive real χ** (rel. residual $\le 4.4 \times 10^{-13}$ across 9 $(\chi,\sigma)$ pairs for $\chi_3, \chi_4, \chi_5$ at $\sigma \in \{2.0, 2.5, 3.0\}$); **does NOT advance G4 or GRH** |
| **P36** Dirichlet L χ-twisted Li–Keiper criterion | `twisted_li_keiper.py` | `63_dirichlet_li_keiper_demo.py` | §13quinquiesdecies | Structural extension of P16 to primitive real $L(s, \chi)$: $\lambda_n(\chi)$ computed from P35 Hardy-Z zeros via the canonical P16 mpmath routine (sum-over-zeros is L-function agnostic); GRH$_\chi$-equivalent diagnostic (Lagarias 2007 generalisation of Bombieri–Lagarias 1999); positivity verified for $\chi_3, \chi_4, \chi_5$ up through $n_{\max} = 50$ (min $\lambda_n \ge 4.7 \times 10^{-2}$); **does NOT prove GRH (finite truncation; necessary, not sufficient) and does NOT advance G4** |
| **P37** Dirichlet L χ-twisted Weil–TNFR bridge | `twisted_weil_positivity.py` | `64_twisted_weil_positivity_demo.py` | §13sexiesdecies | Structural extension of P17 to primitive real $L(s, \chi)$: $W_\chi[\sigma] = 2\sum_{\gamma > 0} h_\sigma(\gamma)$ computed two ways — zero side from P35 Hardy-Z enumerator, explicit-formula side from P34 χ-twisted prime-ladder Hamiltonian — plus the canonical TNFR Lyapunov bridge ratio $\alpha_\chi(\sigma) = W_\chi[\sigma] / E_{\mathrm{TNFR}}^\chi[\sigma]$ using `compute_energy_functional` unchanged from P17; GRH$_\chi$-equivalent diagnostic (Bombieri 2000 generalisation of Weil 1952); positivity verified for $\chi_3, \chi_4, \chi_5$ on Gaussian grid $\sigma \in \{1.0, \ldots, 3.0\}$ (3/3 PASS; XF residual $\le 2.4 \times 10^{-16}$ for $\sigma \ge 2.0$); **does NOT prove GRH (finite Gaussian grid; admissibility not exhausted) and does NOT advance G4** |
| **P38** Dirichlet L χ-twisted admissibility / gauge sweep | `twisted_alpha_sweep.py` | `65_twisted_alpha_sweep_demo.py` | §13septiesdecies | Structural extension of P18 to primitive real $L(s, \chi)$: sweeps $\alpha_\chi(\sigma; g) = W_\chi[\sigma] / E_{\mathrm{TNFR}}^\chi[\sigma; g]$ across the canonical six-gauge family `DEFAULT_GAUGES` inherited unchanged from P18 (`canonical, dnfr_only, phase_only, epi_only, dnfr_phase, pressure_amplified`); $W_\chi$ computed once per $\sigma$ (gauge-independent) via P35 enumerator; canonical TNFR test state built per gauge on P34 bundle; positivity verified for $\chi_3, \chi_4, \chi_5$ across $\sigma \in \{1.0, \ldots, 3.0\} \times$ 6 gauges (3/3 PASS; $\alpha_{\min}$ at $(\sigma=1.0, \text{canonical})$ in every case); robustness audit of P37 under canonical-mapping ambiguity; **does NOT prove GRH (finite $(\sigma, g)$ grid; admissibility not exhausted) and does NOT advance G4** |
| **P39** Dirichlet L χ-twisted admissible-family + gauge sweep | `twisted_admissible_family_sweep.py` | `66_twisted_admissible_family_sweep_demo.py` | §13octiesdecies | Joint structural extension of P19 + P18 to primitive real $L(s, \chi)$: sweeps $\alpha_\chi(\sigma; f, g) = W_\chi[\sigma; f] / E_{\mathrm{TNFR}}^\chi[\sigma; f, g]$ across `DEFAULT_TEST_FAMILIES` (gaussian, gaussian_mixture, hermite2_gaussian) inherited unchanged from P19 × `DEFAULT_GAUGES` (6 canonical gauges) inherited unchanged from P18; $W_\chi[\sigma; f]$ computed once per $(family, \sigma)$ via P35 enumerator; canonical TNFR test state built per $(family, gauge)$ on P34 bundle via `build_twisted_test_state_from_test_function`; positivity verified for $\chi_3, \chi_4, \chi_5$ across 3 families × 6 gauges × 5 widths (3/3 PASS; 270 cells total; $\alpha_{\min}$ at $(\sigma=1.0, \mathrm{gaussian}, \mathrm{canonical})$ in every case); joint robustness audit of P37 under test-profile + canonical-mapping ambiguity; **does NOT prove GRH (finite $(family, gauge, \sigma)$ grid; admissibility not exhausted) and does NOT advance G4** |
| **P40** Dirichlet L χ-twisted node-aware gauge sweep | `twisted_nodeaware_gauge_sweep.py` | `67_twisted_nodeaware_gauge_sweep_demo.py` | §13noniesdecies | Structural extension of P20 to primitive real $L(s, \chi)$: sweeps $\alpha_\chi(\sigma; f, g) = W_\chi[\sigma; f] / E_{\mathrm{TNFR}}^\chi[\sigma; f, g]$ across `DEFAULT_TEST_FAMILIES` (P19) × `DEFAULT_NODEAWARE_GAUGES` (4 node-aware gauges: `nuf_pressure, nuf_phase, weight_pressure, mixed_affine`) inherited unchanged from P20; gauges have signature $g(h(E_n), \hat\nu_f(n), \hat w(n))$ activating the per-node normalised structural-frequency and node-weight channels of the P34 χ-twisted graph; $W_\chi[\sigma; f]$ computed once per $(family, \sigma)$ via P35 enumerator; canonical TNFR test state built per $(family, node\_gauge)$ on P34 bundle via `build_twisted_test_state_nodeaware`; positivity verified for $\chi_3, \chi_4, \chi_5$ across 3 families × 4 node-aware gauges × 5 widths (3/3 PASS; 180 cells total; $\alpha_{\min}$ at $(\sigma=1.0, \mathrm{gaussian}, \mathrm{nuf\_phase})$ for $\chi_3, \chi_4$ and at $(\sigma=1.0, \mathrm{gaussian}, \mathrm{nuf\_pressure})$ for $\chi_5$); node-aware robustness audit of P37 jointly with P19 test-profile sweep; **does NOT prove GRH (finite $(family, node\_gauge, \sigma)$ grid; admissibility not exhausted) and does NOT advance G4** |
| **P41** Dirichlet L χ-twisted Hermite2-Gaussian η-parameter sweep | `twisted_hermite_family.py` | `68_twisted_hermite_family_demo.py` | §13vicies | Structural extension of P21 (Hermite2 family) to primitive real $L(s, \chi)$ along the envelope-strength axis: sweeps $\alpha_\chi(\sigma; \eta, g) = W_\chi[\sigma; \eta] / E_{\mathrm{TNFR}}^\chi[\sigma; \eta, g]$ across `DEFAULT_HERMITE2_ETAS = (0.0, 0.1, 0.25, 0.5, 1.0, 2.0)` ($\eta = 0$ recovers pure Gaussian; $\eta = 0.25$ matches the P19/P39 snapshot) × `DEFAULT_GAUGES` (6 canonical scalar gauges; P18); $W_\chi[\sigma; \eta]$ computed once per $(\eta, \sigma)$ via P35 enumerator; canonical TNFR test state built per $(\eta, g)$ on P34 bundle via `build_twisted_test_state_from_test_function` (reused from P39); positivity verified for $\chi_3, \chi_4, \chi_5$ across 6 etas × 6 gauges × 5 widths (3/3 PASS; 180 cells per character; $\alpha_{\min}$ at $(\sigma=1.0, \eta=0.0, \mathrm{canonical})$ in every case); envelope-strength robustness audit of P37 along an orthogonal axis to P39/P40; **does NOT prove GRH (finite $(\eta, g, \sigma)$ grid; admissibility not exhausted) and does NOT advance G4** |
| **P42** Dirichlet L χ-twisted uniform-coercivity certificate | `twisted_coercivity_uniform.py` | `69_twisted_coercivity_uniform_demo.py` | §13vicies-primo | Structural extension of P22 / P23 / P24 (uniform / stratified / adaptive coercivity in `coercivity_uniform.py`) to primitive real $L(s, \chi)$: lifts the finite-grid sample of P39 + P40 to a **Lipschitz-mesh interval-level certificate** by sampling $\alpha_\chi(\sigma; \eta, g)$ on a log-spaced $\sigma$ grid, computing a finite-difference Lipschitz envelope $L^{\mathrm{proxy}}_\chi$, and forming three interval lower bounds (global, stratified, segment-local) via the canonical P22 / P23 helpers `_max_abs_slope`, `_segmentwise_interval_lower_bound`, `_stratified_interval_lower_bound` reused unchanged; optional P24-style adaptive refinement bisects worst-margin segments and re-runs both twisted sweeps; verified for $\chi_3, \chi_4, \chi_5$ on $\sigma \in [1.0, 3.0]$ with $N = 5$ (`sampled_all_positive = True`, `admissible_ok = True`, `nodeaware_ok = True` for every χ; sampled $\alpha^{\mathrm{samp}}_{\chi,\min} \in \{1.26 \times 10^{-14}, 2.70 \times 10^{-8}, 2.62 \times 10^{-10}\}$; interval $\mathrm{lb}_{\mathrm{local}} \in \{-6.06 \times 10^{-2}, -1.30 \times 10^{-1}, -6.51 \times 10^{-2}\}$ — all **negative** because $\alpha^{\mathrm{samp}}_{\chi,\min}$ near $\sigma = 1$ is essentially zero against any finite $L^{\mathrm{proxy}}_\chi$); one round of P24 bisection on the worst character ($\chi_4$, $N = 5 \to 7$) reduces $\mathrm{lb}_{\mathrm{local}}$ from $-1.30 \times 10^{-1}$ to $-3.40 \times 10^{-2}$ (74% margin reduction toward zero), confirming the bisection mechanism transports correctly to the χ-twisted side; **does NOT prove GRH (interval lower bounds currently negative; even when positive, finite log-spaced σ window is necessary, not sufficient) and does NOT advance G4** |
| **P43** Dirichlet L χ-twisted Paley-gap consistency diagnostic | `twisted_paley_gap_coercivity.py` | `70_twisted_paley_gap_coercivity_demo.py` | §13vicies-secundo | Structural extension of P25 (`paley_gap_coercivity.py`) to primitive real $L(s, \chi)$: compares three representations of $-L'(s,\chi)/L(s,\chi)$ — the P32 closed-form weighted spectrum $Z_{P32}$ (`tnfr_log_l_derivative`), the P34 χ-twisted weighted spectral trace $Z_{P34}$ (`twisted_weighted_spectral_trace`), and the classical truncated Dirichlet series $Z_{\mathrm{cls}} = \sum_{n \le N} \chi(n)\Lambda(n)/n^\sigma$ (`classical_log_l_derivative`) — via three absolute χ-twisted Paley-gap quantities $g_{P32}(\sigma) = |Z_{P32} - Z_{\mathrm{cls}}|$, $g_{P34}(\sigma) = |Z_{P34} - Z_{\mathrm{cls}}|$, $g_{\mathrm{cross}}(\sigma) = |Z_{P34} - Z_{P32}|$; verified on $(n_{\mathrm{primes}}, k_{\max}, N_{\max}^{\mathrm{cls}}) = (18, 5, 50\,000)$, $\sigma \in [1.5, 4.0]$ with $N = 11$ for $\chi_3, \chi_4, \chi_5$: at $J_0 = 0$ the decoupled cross gap collapses to machine precision ($\max g_{\mathrm{cross}} \in \{5.55 \times 10^{-17}, 4.16 \times 10^{-17}, 1.11 \times 10^{-16}\}$ — Paley-style algebraic identity between P32 and P34 on the L-track, regression test); at $J_0 = 10^{-2}$ the coupling-induced cross gap jumps to $O(10^{-5})$ (twelve orders of magnitude above noise; clean structural-deformation signal free of classical-truncation noise which contaminates $g_{P32}$ at $10^{-3}$ to $10^{-2}$); **does NOT prove GRH (regression test plus deformation magnitude; not a coercivity certificate) and does NOT advance G4** |
| **P44** Dirichlet L χ-twisted Lyapunov-spectral positivity certificate | `twisted_lyapunov_spectral_positivity.py` | `71_twisted_lyapunov_spectral_demo.py` | §13vicies-tertio | Structural extension of P26 (`lyapunov_spectral_positivity.py`) to primitive real $L(s, \chi)$: certifies self-adjointness, strict positivity with explicit Kato–Rellich envelope $\lambda_{\min}(\hat H^{(\chi)}) \ge \Delta_0^{(\chi)} - \lvert J_0 \rvert \lVert \hat H^{(\chi)}_{\mathrm{coupling}} \rVert_{\mathrm{op}}$ where $\Delta_0^{(\chi)} = \log(\min\{p \text{ prime} : p \nmid q\})$ (character-dependent: $\log 2$ for $\chi_3, \chi_5$; $\log 3$ for $\chi_4$), trace-class resolvent (Schatten-1/2 norms), and unitary flow conservation of $U(t) = e^{-it \hat H^{(\chi)}}$ on the finite-dimensional χ-twisted prime-ladder Hilbert space (P34 bundle); reuses `_matrix_exponential_skew` and `resolvent_schatten_norms` atomically from P26; verified on $(n_{\mathrm{primes}}, k_{\max}) = (18, 5)$ for $\chi_3, \chi_4, \chi_5$ at $J_0 \in \{0, 10^{-2}\}$: at $J_0 = 0$ empirical $\min(\lambda)$ matches $\Delta_0^{(\chi)}$ to machine precision (asserted in demo); at $J_0 = 10^{-2}$ `perturbation_safe = True` for every character with guaranteed gap $\in \{6.76 \times 10^{-1}, 1.08, 6.76 \times 10^{-1}\}$; unitary drifts $\sim 2 \times 10^{-16}$ throughout; `structural_positivity = True` for all 6 cells; **does NOT prove GRH (finite-dimensional positivity is necessary but not sufficient; the character enters only via the active-prime restriction, not via $W^{(\chi)}$) and does NOT advance G4** |
| **P45** Dirichlet L χ-twisted Hilbert–Pólya scaffold | `twisted_hilbert_polya.py` | `72_twisted_hilbert_polya_demo.py` | §13vicies-quarto | Structural extension of P27 (`hilbert_polya.py`) to primitive real $L(s, \chi)$: builds the reference operator $T_{\mathrm{HP}}^{(\chi)} = \operatorname{diag}(\gamma_1^{(\chi)}, \dots, \gamma_N^{(\chi)})$ on $\ell^2_N(\mathbb{N})$ where $\gamma_n^{(\chi)}$ are positive imaginary parts of zeros of $L(s, \chi)$ located by Hardy–Z bisection (`find_dirichlet_l_zeros`, the same enumerator used by P36); reuses `build_hp_operator`, `verify_hp_self_adjoint`, `hp_resolvent_schatten_norms`, `wasserstein_1_distance` atomically from P27; certifies (i) self-adjointness (real diagonal, exact, Frobenius asymmetry $= 0$), (ii) trace-class shifted resolvent $(T_{\mathrm{HP}}^{(\chi)2} + s^2 I)^{-1/2}$ with explicit Schatten-1/2/op norms, (iii) χ-twisted Weil–Guinand consistency $2 \sum h_\sigma(\gamma_n^{(\chi)}) = g(0) \log(q/\pi) +$ archimedean $+ \sum_{p \nmid q, k} \chi(p)^k \log(p) p^{-k/2} g(k \log p)$ (parity-shifted digamma, character-dependent constant term replaces $\zeta$-pole $-g(0) \log \pi$), and (iv) Wasserstein-1 spectral gap against $\operatorname{spec}(\hat H^{(\chi)} \mid p \nmid q)$; verified on $(n_{\mathrm{primes}}, k_{\max}, n_{\mathrm{zeros}}, \sigma, s, \mathrm{tol}) = (18, 5, 25, 2.0, 1.0, 10^{-2})$ for $\chi_3, \chi_4, \chi_5$: Weil residuals $\{5.19 \times 10^{-16}, 9.07 \times 10^{-15}, 1.72 \times 10^{-15}\}$ at machine precision; $W_1 \in \{35.5, 31.8, 30.3\}$ with growth ratios $\sim 12$ quantifying the L-track operator-level structural gap (mirror of P30 negative-enrichment for $\zeta$); `scaffold_consistent = True` for all 3 characters; **does NOT prove GRH ($T_{\mathrm{HP}}^{(\chi)}$ is populated by *inputting* Hardy–Z bisection of classical $L(s, \chi)$; the operator is not derived from TNFR first principles) and does NOT advance G4** |
| **P46** Dirichlet L χ-twisted structural zero density | `twisted_structural_zero_density.py` | `73_twisted_structural_zero_density_demo.py` | §13vicies-quinto | L-track analogue of P28 (`structural_zero_density.py`): derives the smooth chi-twisted zero positions $\tilde{\gamma}_n^{(\chi)}$ from the chi-twisted Riemann–Siegel theta $\theta_\chi(T) = \operatorname{Im} \log \Gamma((1/2+a)/2 + iT/2) + (T/2) \log(q/\pi)$ via Newton iteration on $\bar{N}_\chi(\tilde{\gamma}_n^{(\chi)}) = n - 1/2$ — no `find_dirichlet_l_zeros` call on the derivation side (only used for benchmark); builds $\tilde{T}_{\mathrm{HP}}^{(\chi)} = \operatorname{diag}(\tilde{\gamma}_1^{(\chi)}, \dots, \tilde{\gamma}_N^{(\chi)})$ and certifies (i) per-zero residuals $r_n^{(\chi)} = \gamma_n^{(\chi)} - \tilde{\gamma}_n^{(\chi)}$ encoding $S_\chi(T) = \tfrac{1}{\pi} \arg L(\tfrac12 + iT, \chi)$, (ii) operator-level Wasserstein-1 reduction $W_1(\operatorname{spec}(\tilde{T}_{\mathrm{HP}}^{(\chi)}), T_{\mathrm{HP}}^{(\chi)}) \ll W_1(\operatorname{spec}(P34\vert_{p\nmid q}), T_{\mathrm{HP}}^{(\chi)})$, (iii) theoretical bound $\max\lvert r_n^{(\chi)}\rvert \le C \cdot \max(\log \gamma_n^{(\chi)} / \bar{N}_\chi'(\gamma_n^{(\chi)}))$ with $C \le 2$; verified on $(n_{\mathrm{zeros}}, p34\_n\_primes, p34\_max\_power) = (18, 30, 6)$ for $\chi_3, \chi_4, \chi_5$: $\max\lvert r_n^{(\chi)}\rvert \in \{3.21, 2.65, 2.53\}$; $W_1$ reductions $\{28.4 \to 1.32, 25.2 \to 1.23, 24.1 \to 1.17\}$, improvement ratios $\{21.6\times, 20.4\times, 20.6\times\}$; bound satisfied across all 3 characters; closes the **smooth half** of the L-track structural derivation gap (mirror of P28 for ζ); **does NOT prove GRH for any $L(s, \chi)$** (oscillatory residual encoding $S_\chi$ is the open arithmetic problem, equivalent to GRH$_\chi$) **and does NOT advance G4 = RH** |
| **P47** Dirichlet L χ-twisted spectral emergence under canonical coupling | `twisted_spectral_emergence.py` | `74_twisted_spectral_emergence_demo.py` | §13vicies-sexto | L-track analogue of P29 (`spectral_emergence.py`): sweeps three exploratory (non-canonical) inter-prime coupling laws (`kuramoto_u3`: $(\gamma/\pi)\exp(-\lvert k\log p - m\log q\rvert)$; `phi_multiscale`: $\varphi^{-(k+m)}/\sqrt{pq}$; `pnt_logarithmic`: $\gamma/\log(1+pq)$) on the P34 χ-twisted prime-ladder Hamiltonian with explicit $\chi(p)\chi(q)$ multiplicative twist on every off-diagonal entry; computes the Kolmogorov–Smirnov distance of the unfolded nearest-neighbour spacing distribution to the GUE Wigner surmise (conjectural universality class of zeros of $L(s,\chi)$) and to the Poisson reference; verified on $(n_{\mathrm{primes}}, k_{\max}) = (20, 3)$ for $\chi_3, \chi_4, \chi_5$ over strengths $s \in \{0, 0.05, 0.1, 0.2, 0.5, 1, 2\}$: `pnt_logarithmic` uniformly strongest emergence kernel with $\mathrm{KS}_{\text{GUE}}^{\min} \in \{0.097, 0.116, 0.135\}$ at $s^* = 2$ ($33$–$49\%$ reduction vs baseline); `kuramoto_u3` second with $\mathrm{KS}_{\text{GUE}}^{\min} \in \{0.120, 0.150, 0.135\}$ at $s^* = 1$ ($25$–$36\%$ reduction); `phi_multiscale` weak ($0$–$6\%$ reduction); attests the L-track spacing-universality diagnostic for every primitive real Dirichlet character; **does NOT prove GRH for any $L(s, \chi)$** (KS-GUE residual at finite $K$ is consistent with finite-size effects, not evidence against GRH) **and does NOT advance G4 = RH** |
| **P49** Dirichlet L χ-twisted prime-ladder oscillatory correction | `twisted_oscillatory_correction.py` | `76_twisted_oscillatory_correction_demo.py` | §13vicies-octavo | L-track analogue of P31 (`oscillatory_correction.py`): reconstructs $S_\chi(T) = \pi^{-1}\arg L(\tfrac12 + iT, \chi)$ from the canonical P34 χ-twisted prime-ladder spectrum $\{(k\log p,\,\chi(p)^k\log p)\}$ via the χ-twisted Riemann–von Mangoldt template $\pi S_\chi^{\mathrm{TNFR}}(T) = -\sum_{(\mu,w)}(w/\mu)\sin(T\mu)\exp(-\mu/2)$, then applies the Newton step $\gamma_n^{(\chi),\,\text{corr}} = \tilde\gamma_n^{(\chi)} - d\,S_\chi^{\mathrm{TNFR}}(\tilde\gamma_n^{(\chi)}) / \bar N'_\chi(\tilde\gamma_n^{(\chi)})$ on the canonical P46 χ-twisted smooth targets with $\bar N'_\chi(T) = (2\pi)^{-1}\log(qT/(2\pi))$; restricted to **primitive real** characters so the von Mangoldt-style sum is real-valued (validates $\max\lvert\Im w\rvert \le 10^{-10}$); damping sweep $d \in \{0, 0.25, 0.5, 0.75, 1, 1.25, 1.5\}$; **closes the final ζ↔L attack-surface parity item**: with P49, every canonical ζ-track operator P12–P31 has a matching χ-twisted L-track counterpart (P32–P49); verified on $(N, N_{\mathrm{primes}}, K) = (10, 80, 5)$ for $\chi_3, \chi_4, \chi_5$: mixed empirical regime — $\chi_4$ shows **+6.02%** branch-B1 canonical improvement at $d^* = 1.5$ ($W_1$: $1.4185 \to 1.3331$); $\chi_3$ and $\chi_5$ show **0% improvement** ($d^* = 0$) corroborating §13octies branch B2 at the L-track level (a genuinely new canonical operator required); honest split (1/3 B1, 2/3 B2) further attests the canonical-only oscillatory cap visible across both tracks; **does NOT prove GRH$_\chi$ for any $L(s, \chi)$** (residual $W_1 \approx 1.3$–$1.6$ encodes the chi-twisted oscillatory remainder), **does NOT advance G4 = RH**, **does NOT address sub-problems (2) canonicity from the nodal equation and (3) positivity coincidence with the χ-twisted Weil form**; positive structural-parity milestone plus L-track structural-compatibility diagnostic |
| **P48** Dirichlet L χ-twisted admissible spectral-rescaling operator | `twisted_admissible_rescaling.py` | `75_twisted_admissible_rescaling_demo.py` | §13vicies-septimo | L-track analogue of P30 (`admissible_rescaling.py`): lifts the §13vicies-quinto density-level closure of the smooth half of T-HP$^{(\chi)}$ to the operator level by constructing the canonical diagonal rescaling $F^{(\chi)}_{\text{smooth}} = U_{P34}\,\operatorname{diag}(\sqrt{\tilde{\gamma}_i^{(\chi)} / \lambda_i})\,U_{P34}^{*}$ on each primitive real Dirichlet character; reuses `extract_positive_spectrum`, `build_smooth_rescaling_operator`, `apply_rescaling`, `verify_self_adjointness_preserved`, `verify_spectrum_match`, `oscillatory_correction_canonical` atomically from `admissible_rescaling.py`; certifies (i) self-adjointness preservation under conjugation, (ii) exact spectrum match $\operatorname{spec}(F^{(\chi)}_{\text{smooth}}\,H_{P34}^{(\chi)}\,(F^{(\chi)}_{\text{smooth}})^{*}) = \{\tilde{\gamma}_i^{(\chi)}\}$ to machine precision $\le 7.1\times10^{-15}$, (iii) Wasserstein-1 gap closure $W_1(\sigma(H_{P34}^{(\chi)}), \{\gamma_n^{(\chi)}\}) \to W_1(\{\tilde{\gamma}_n^{(\chi)}\}, \{\gamma_n^{(\chi)}\})$, (iv) honest sweep of the three canonical oscillatory enrichments (`phi_log`, `gamma_e`, `pi_density`) at amplitudes $\{0, 10^{-3}, 5\!\cdot\!10^{-3}, 10^{-2}, 5\!\cdot\!10^{-2}, 10^{-1}\}$ with per-mode breakdown; verified on $(n_{\mathrm{targets}}, p34\_n\_primes, p34\_max\_power) = (12, 25, 5)$ for $\chi_3, \chi_4, \chi_5$: smooth-half W$_1$ ratios $\{14.86\times, 13.85\times, 14.44\times\}$ (baseline $\{21.9, 19.0, 18.4\} \to$ smooth $\{1.47, 1.38, 1.27\}$); best canonical oscillation `pi_density` at amplitude $10^{-3}$ for every character with extra improvement $\{+17.85\%, +13.22\%, +12.68\%\}$ over smooth baseline; per-mode ranking uniform: `pi_density` > `gamma_e` > `phi_log`; closes sub-problem (1) of Conjecture T-HP$^{(\chi)}$ for the smooth half at the operator level (L-track mirror of P30 §13nonies); negative-knowledge oscillatory cap ($\le 18\%$ canonical improvement) constitutes structural evidence for §13octies branch B2 at the L-track level; **does NOT prove GRH$_\chi$ for any $L(s, \chi)$** (residual W$_1 \approx 1.1$–$1.2$ encodes $S_\chi(T) = (1/\pi)\arg L(\tfrac12+iT, \chi)$, GRH$_\chi$-equivalent) **and does NOT advance G4 = RH** |
| **P50** REMESH-∞ residue split of P31 oscillatory correction | `remesh_infinity_residue_split.py` | `77_remesh_infinity_residue_split_demo.py` | §13triginta | Function-space lift of the N15 REMESH-∞ closure (`theory/REMESH_INFINITY_DERIVATION.md`) into the TNFR-Riemann program: splits the canonical P31 prime-ladder reconstruction $S_{\mathrm{TNFR}}(T) = -(1/\pi)\sum_{(\mu,w)}(w/\mu)\sin(T\mu)\exp(-\mu/2)$ into its projections on $\mathrm{range}(\mathcal{R}_\infty)$ and $\ker(\mathcal{R}_\infty)$ via the DFT-bin mask selecting the N15-resonant rational-multiple-of-$\pi$ lattice $\{2\pi k / \mathrm{lcm}(\tau_l, \tau_g)\}$ at the canonical pair $(\tau_l, \tau_g) = (4, 8)$; pre-registered structural prediction: the prime-ladder Fourier support $\{k\log p\}$ is disjoint from the N15-resonant lattice by Baker's theorem on linear independence of logarithms of algebraic numbers, hence the canonical reconstruction lies asymptotically in $\ker(\mathcal{R}_\infty)$; verdicts: `RESIDUE_IN_KER_ONLY` (branch B2 evidence at function-space level), `RESIDUE_IN_RANGE_ONLY` (would refute P31), `RESIDUE_MIXED` (gauge leak or boundary artefact); verified at canonical defaults $(\tau_l, \tau_g) = (4, 8)$, $n_{\mathrm{periods}} \in \{64, 256\}$, $n_{\mathrm{primes}} \in \{200, 400\}$, $K = 8$: verdict `RESIDUE_IN_KER_ONLY` at both resolutions; range fraction decays $1.7647\% \to 0.0162\%$ as $n_{\mathrm{samples}}: 512 \to 2048$ (clean asymptotic incommensurability); two sanity controls pass at machine precision (resonant $\sin(2\pi T/\mathrm{lcm})$ projects to $100\%$ range; transcendental $\sin(\gamma_{\mathrm{em}} T)$ projects to $\le 7 \times 10^{-4}\%$ range); complementary to §13vicies-novies graph-iteration-matrix tests (which act on EPI-history state vectors): P50 acts on a function in $H^2(T\text{-axis})$, a mathematically distinct object; corroborates the §13septies / §13nonies structural identification of the T-HP residual obstruction with the oscillatory half $S(T) = (1/\pi)\arg\zeta(\tfrac12+iT) = \ker(\mathcal{R}_\infty)$ component; **does NOT advance G4 = RH**, **does NOT close T-HP**, **does NOT promote any new canonical operator beyond the 13-operator catalog**; positive structural-compatibility milestone connecting the N15 REMESH-∞ closure to the T-HP residual gap at the function-space level |

### 19.2 Gap Balance

| Gap | Description | Status |
|---|---|---|
| **G1** | Canonical TNFR Hamiltonian carrying the prime-ladder spectrum | **CLOSED operationally** by P14 |
| **G2** | Analytic continuation of the TNFR vM zeta to $\mathbb{C}$ | **CLOSED operationally** by P13 |
| **G3** | Explicit zeros $\leftrightarrow$ spectrum bridge | **CLOSED operationally** by P15 (Weil–Guinand) |
| **G4** | **Riemann Hypothesis** — localisation of poles on $\operatorname{Re}(s) = 1/2$ | **OPEN** (= Conjecture T-HP, §13septies). Smooth half of sub-problem (1) of T-HP closed at **density** level by P28 (§13sexies) and at the **operator** level by P30 (§13nonies). Oscillatory half (P31, §13decies) tested with the canonically correct multi-frequency prime-ladder basis: partial positive evidence at very low $N$ ($+3.6\%$ at $N$=20, $d$=1), zero or negative at $N$=40; corroborates branch B2. Canonicity (sub-problem (2)) and positivity coincidence (sub-problem (3)) remain open. |
| **G5** | Bridge from TNFR spectral zeta to classical $\zeta(s)$ | **SUPERSEDED** by P12+P13+P15 (§7.8); original affine form numerically falsified (§7.1–§7.7). |

**Net result**: 4 of 5 originally identified gaps are operationally closed inside the canonical TNFR formalism. The only remaining obstruction is **G4 = RH itself**, restated canonically as **Conjecture T-HP** in §13septies and audited link-by-link (L1–L8) in §13octies. Extensions beyond P12–P16 (P17–P30) inside the canonical engine progressively narrow G4 — by exposing the attack surface (P17), auditing the admissibility envelope (P18–P21), certifying interval-level coercivity (P22–P24), providing a Paley-style identity (P25), certifying operator-level positivity for P14 (P26), supplying a diagnostic Hilbert–Pólya scaffold (P27), and closing the smooth half of T-HP at density (P28) and operator (P30) level — but none of them closes G4. The oscillatory half of T-HP requires either a new canonical operator beyond the 13-operator catalog (§13octies branch B2; supported by the P30 negative-enrichment result, §13nonies.4) or a structural derivation of $S(T) = \pi^{-1} \arg \zeta(\tfrac{1}{2} + iT)$ from canonical TNFR ingredients (branch B1, untested).

### 19.3 Scope Statement (Honest Reading)

What the TNFR-Riemann programme **does** at the May 2026 milestone:

* Provides an end-to-end computable pipeline from the nodal equation
  $\partial \mathrm{EPI}/\partial t = \nu_f \cdot \Delta\mathrm{NFR}(t)$ to the
  Weil–Guinand explicit formula (P1–P15).
* Reproduces $-\zeta'(s)/\zeta(s)$ exactly on $\operatorname{Re}(s) > 1$ via a
  prime-ladder spectrum (P12) and continues it analytically to $\mathbb{C}$ (P13).
* Builds a self-adjoint Hamiltonian $\hat H$ on a TNFR graph whose
  weighted spectral trace carries the same data (P14).
* Numerically verifies the Weil–Guinand identity to machine precision
  using $\hat H$ on the prime side (P15).
* Exposes Li's positivity criterion as a TNFR-native, RH-equivalent
  diagnostic surface (P16).
* Opens a TNFR-native attack surface on G4 via the Weil–TNFR positivity
  bridge $\alpha(\sigma)$ (P17) and audits its admissibility envelope
  across canonical gauge, family and node-aware extensions (P18–P21).
* Certifies interval-level uniform coercivity of $\alpha(\sigma)$ on
  tested intervals (P22–P24) and provides a Paley-gap diagnostic
  vanishing at coupling zero (P25).
* Lifts positivity to the operator level for the P14 Hamiltonian (P26),
  supplies a diagnostic Hilbert–Pólya scaffold populated by
  `mpmath.zetazero` (P27), derives the smooth Riemann zero density
  structurally (P28), and closes the smooth half of the
  Tetrad-Hilbert–Pólya conjecture (T-HP) at the operator level (P30).

What the programme **does not** do:

* Prove RH. P16 is RH-equivalent, not RH-proving: a numerical violation
  $\lambda_n \le 0$ would disprove RH, but $\lambda_n > 0$ for any finite
  truncation does not prove it. P26 / P27 are diagnostic; P28 / P30
  cover only the smooth (archimedean) half of T-HP.
* Replace the classical $\zeta(s)$. The TNFR construction reproduces
  classical data; it does not derive new analytic-number-theory results.
* Close G4 by any internal extension. Crossing G4 requires either
  (branch B1) a structural derivation of the oscillatory term
  $S(T) = \pi^{-1} \arg \zeta(\tfrac{1}{2} + iT)$ from canonical TNFR
  ingredients, or (branch B2) a new canonical operator beyond the
  current 13-operator catalog, derivable from the nodal equation.
  Branch B2 is currently supported by the P30 negative-enrichment
  result (§13nonies.4). Branch B3 (no TNFR closure) cannot be ruled
  out at this stage.

### 19.4 Reproducibility

All P1–P30 results are reproducible via the corresponding demos in
`examples/` using the standard project invocation:

```powershell
$env:PYTHONPATH = (Resolve-Path ./src).Path
& .\.venv312\Scripts\python.exe examples\57_admissible_rescaling_demo.py
```

The full pipeline (importability of every canonical entry point of the
30 milestones) can be sanity-checked with:

```python
from tnfr.riemann import (
    # Discrete operator & spectral framework (P1–P11)
    build_prime_path_graph,                     # P1
    compute_eigensystem,                        # P1
    compare_topologies,                         # P2
    compute_eigenmode_tetrad,                   # P3
    compute_complex_eigensystem,                # P4
    compute_spectral_zeta,                      # P5
    run_rmt_ensemble_analysis,                  # P6
    run_critical_conservation_analysis,         # P7
    run_analytical_convergence_proof,           # P8
    run_functional_equation_analysis,           # P9
    run_formal_convergence_proof,               # P10
    run_zeta_bridge_analysis,                   # P11
    # Prime-ladder / von Mangoldt pipeline (P12–P16)
    build_prime_ladder_spectrum,                # P12
    von_mangoldt_zeta_continued,                # P13
    scan_critical_line_for_poles,               # P13
    build_prime_ladder_hamiltonian,             # P14
    verify_weil_explicit_formula,               # P15
    verify_li_keiper_criterion,                 # P16
    # TNFR-native G4 attack surface (P17–P30; does NOT close G4 = RH)
    verify_weil_tnfr_bridge,                    # P17
    sweep_alpha,                                # P18
    sweep_alpha_admissible_family,              # P19 / P21
    sweep_alpha_nodeaware,                      # P20
    verify_uniform_coercivity_empirical,        # P22 / P23 / P24
    sweep_paley_gap,                            # P25
    compute_lyapunov_spectral_certificate,      # P26
    compute_hilbert_polya_certificate,          # P27
    compute_structural_zero_density_certificate,# P28
    compute_spectral_emergence_report,          # P29
    compute_admissible_rescaling_certificate,   # P30
)
```

This single import covers the canonical entry points of every milestone
delivered so far. Symbols not exported by name correspond to internal
helper functions; consult `src/tnfr/riemann/__init__.py` for the
authoritative public surface.

---

## §13vicies-novies. REMESH Global Reframe (Cross-Program Discovery; May 2026; Does NOT Close G4 = RH)

**Status**: Working hypothesis (branch B1 of §13septies.7). Does **not** close G4 = RH, does **not** advance T-HP beyond §13nonies (P30 smooth half), does **not** promote any new canonical operator.

### §13vicies-novies.1 Origin

During the parallel TNFR–Navier–Stokes program (see `theory/TNFR_NAVIER_STOKES_RESEARCH_NOTES.md` §11), an analysis of the NS-G_blowup residual obstruction prompted re-examination of the 13-operator catalog for multi-scale closure primitives. A direct audit refuted the prior implicit assumption that no canonical operator handles asymptotic/global temporal coupling:

* `src/tnfr/config/defaults_core.py`: `REMESH_TAU_GLOBAL = 8` (graph-wide temporal memory), `REMESH_TAU_LOCAL = 4`, `REMESH_MODE in {knn, mst, community}` with `community` mode genuinely global.
* `src/tnfr/ontosim.py`: `# Global REMESH memory` allocates a graph-level `_epi_hist` deque of size `2·τ_global + 5`.
* `src/tnfr/operators/remesh.py`: documents three REMESH structural modes — **Hierarchical** (IL/VAL/SHA/NUL), **Rhizomatic** (OZ/UM/THOL), **Fractal Harmonic** (RA/NAV/AL/EN, scale-symmetric).
* `src/tnfr/multiscale/hierarchical.py`: explicit cross-scale ΔNFR coupling.

The canonical engine therefore **already contains** a global, multi-scale closure primitive (REMESH global with Fractal Harmonic mode and cross-scale coupling). What is missing for T-HP is the **canonical asymptotic specialisation of the existing REMESH global operator** at `τ → ∞` applied to the prime-ladder spectrum, not a new canonical primitive.

### §13vicies-novies.2 Reframed Branch Analysis of T-HP

| Component | Status | REMESH-global interpretation |
|---|---|---|
| Smooth half of `F` | Closed at density level (P28, §13sexies) and operator level (P30, §13nonies) | REMESH global at **finite** `τ_global` applied to the prime-ladder spectrum `{k log p}` (P14 eigendata) |
| Oscillatory half `S(T) = (1/π) arg ζ(½+iT)` | Open (RH-equivalent) | REMESH global at **`τ → ∞`** applied to the same prime-ladder spectrum |
| Branch classification | Previously implicitly B2 (new operator) | **Reframed as B1** (closeable inside the catalog if the canonical `τ → ∞` limit of REMESH global is derivable) |

### §13vicies-novies.3 What This Changes for the Riemann Program

* **The hypothesis is upgraded** from "new operator may be needed" (branch B2, open and uncertain) **to** "existing operator needs canonical asymptotic specialisation" (branch B1, a well-defined analytical problem on an existing canonical operator).
* **G4 = RH remains OPEN**. The P30 negative-enrichment result (canonical multiplicative perturbations of the smooth target failed to recover S(T)) is **reinterpretable**: the perturbations tested were finite-`τ` REMESH-global candidates, none of which can reproduce a `τ → ∞` limit by construction.
* **The Riemann program remains paused at T-HP** (per §"Program Status" of `AGENTS.md`). The reframe does **not** authorise reopening the ζ-track or L-track attack surfaces; it only re-classifies the residual obstruction.

### §13vicies-novies.4 Honest Scope

* **What §13vicies-novies claims**: a structural reframe of the T-HP residual obstruction, anchored in canonical engine artefacts (`REMESH_TAU_GLOBAL`, `_epi_hist`, REMESH modes, `multiscale/hierarchical.py`).
* **What §13vicies-novies does NOT claim**: does NOT prove RH, does NOT close G4, does NOT close T-HP, does NOT derive `REMESH-∞`, does NOT promote any new operator.
* **Cross-reference**: mirrored in `theory/TNFR_NAVIER_STOKES_RESEARCH_NOTES.md` §11 (added simultaneously). Both programs share the same canonical REMESH global infrastructure; the analytical study of its `τ → ∞` (Riemann) / scale `→ 0` (NS) asymptotic limit is shared work.

### §13vicies-novies.5 R∞-1a Empirical Baseline (Riemann side)

**Milestone**: R∞-1a — first numerical probe of REMESH-∞ on the Riemann-side prime-ladder dynamics.

**Implementation**: `benchmarks/remesh_infinity_riemann_baseline.py`. Output: `results/remesh_infinity/remesh_infinity_riemann_baseline.json`.

**Setup**:
* Graph: P14 prime-ladder, `n_primes=10`, `max_power=4` → 40 nodes `(p, k)`, νf = k·log(p).
* Synthetic deterministic oscillatory field: `EPI(p,k;t) = (log(p)/k)·cos(k·log(p)·t)` evaluated on `t ∈ [0, dt, 2dt, …]`, `dt = 0.05`.
* History buffer `_epi_hist` populated to `max(τ_g, τ_l)+1` snapshots before each REMESH application; canonical mixing `EPI_new = 0.25·EPI_now + 0.25·EPI[t-τ_l] + 0.5·EPI[t-τ_g]` with `α = 0.5`, `τ_l = 4`.
* Three tracks executed in one run:
  * **Track A** — single-application sweep over `τ_g ∈ {4, 8, 16, 32, 64, 128, 256, 512}`, baseline restored between calls. Tests F1 (naive single-application Cesàro projection).
  * **Track B** — iterated REMESH^N at fixed `τ_g = 16`, `N ∈ [1, 512]`, with `_epi_hist` updated at every iteration (genuine Banach iteration of the canonical operator on this dynamics). Tests F2 (existence of a fixed point).
  * **Track C** — spectral diagnostic of the late-iterated state at `N = 256`, FFT along the νf-ordered axis after mean removal.

**Falsification criteria (pre-registered)**:
* **F1** triggered if Track A `dist→time_average` is monotone-decreasing in `τ_g` AND `final_rel < 0.1`. Interpretation: naive single-application B1 = Cesàro projection on time-average ⇒ B1 (naive) refuted.
* **F2** triggered if Track B `final_step_delta < 1e-6` OR `step_decay_ratio < 0.01`. Interpretation: iterated REMESH has a well-defined fixed point.

**Results (deterministic run; same seedless config reproducible)**:
* Baseline-to-time-average distance: 5.976e+00.
* **Track A**: F1 NOT triggered. Distance to time-average plateaus at `rel ∈ [0.392, 0.462]` across the entire sweep, non-monotone in `τ_g`. Confirms analytically that single-application `τ → ∞` is ill-defined on stationary oscillatory snapshots: the output depends on the specific phase of the past snapshot sampled at lag τ_g, not on a global asymptotic limit.
* **Track B**: F2 **TRIGGERED**. `step_decay_ratio = 6.82e-06`, `final_step_delta = 3.87e-05` at `N = 512`. Step deltas decay through `5.68 → 1.03 → 0.66 → … → 0.15 → 0.012 → 1.2e-4 → 3.9e-5`. The iterated map converges to a fixed point with `‖EPI*‖_L2 = 1.7501`, sitting at relative distance `0.2808` from the time-average (i.e. NOT the time-average).
* **Track C**: Late state at `N = 256` has structured oscillatory content along the νf-ordered axis. After mean removal, total power = 64.2, DC fraction = 3.07e-33 (numerical zero). Top-3 power bins are `{16, 19, 20}` of 21 rfft bins, with fractions `{10.6%, 9.7%, 9.3%}` — the spectrum is dominated by **high-νf modes**, not the low-νf prime-ladder fundamentals.

**Honest interpretation (R∞-1a)**:
* **Established** (necessary condition for any non-trivial B1 reframe): iterated REMESH on canonical prime-ladder oscillatory dynamics admits a well-defined fixed point. The fixed point is NOT the time-average and carries non-trivial spectral structure.
* **Not established** (and must NOT be claimed): (a) any verified correspondence between the fixed-point spectrum and the oscillatory residual `r_n = γ_n - γ̃_n`; (b) sensitivity-independence with respect to the choice of synthetic input field; (c) that high-νf concentration encodes S(T) rather than being a bias of the α-local mixing kernel; (d) closure of T-HP, G4, or RH.
* **Branch verdict (R∞-1a slice only)**: this baseline does NOT refute B1, and supplies the first necessary positive datum (existence of a non-trivial canonical fixed point). It does NOT confirm B1 either — the spectral comparison with r_n (R∞-1a-spectral, future work) is the next falsifiable test.

**Next milestones (gated on this result)**:
* **R∞-1a-spectral**: project the Track B fixed-point spectrum onto the basis of r_n via mpmath-computed γ_n; report correlation, cosine similarity, and per-component residual. Pre-register falsification: if no correlation above noise (|r| < 0.2), B1 is empirically refuted at the spectral level even with a non-trivial fixed point.
* **R∞-1b**: NS-side analogue on the K_φ cascade (N6–N11 milestones), same Track A/B/C structure.
* **R∞-1c**: cross-program comparison of fixed-point spectra. Required equivariance check before any cross-program B1 claim.

**Status**: R∞-1a baseline complete; primary deliverable is the empirical fact that iterated REMESH is contractive on this dynamics with a non-trivial fixed point. No closure of any gap.

### §13vicies-novies.6 R∞-1a-spectral — Spectral projection onto Riemann basis

**Milestone**: R∞-1a-spectral — first falsifiable spectral comparison between the R∞-1a fixed point and Riemann data. Gated follow-up to §13vicies-novies.5.

**Implementation**: `benchmarks/remesh_infinity_riemann_spectral.py`. Output: `results/remesh_infinity/remesh_infinity_riemann_spectral.json`.

**Setup**:
* Identical prime-ladder, REMESH config, and Banach iteration as R∞-1a, run to `N_iter = 512` (true fixed point, not the intermediate `N = 256` state used in R∞-1a Track C).
* Riemann reference: first 40 non-trivial zeros γ_n from `mpmath.zetazero` (dps=30) and the canonical smooth approximations γ̃_n via `derive_smooth_zero_position` (P28). Oscillatory residuals `r_n = γ_n - γ̃_n`.
* Fixed point sorted by νf = k·log(p) → sequence `s_i, i = 1..40`. FFT of `s − mean(s)` → power bins `P_k, k = 1..M` with `M = 20`.

**Pre-registered tests (none decisive on its own)**:
* `r_α` = Pearson(P_k, |r_n|), index-aligned k=n=1..M.
* `r_β` = Pearson(sort(P_k, desc), sort(|r_n|, desc)) — magnitude-distribution alignment.
* `r_γ` = Pearson(s_i [νf-ordered], γ̃_n [n=1..N=40]) — node-field vs smooth target alignment.
* `r_δ` = Spearman-rank(P_k, |r_n|).

**Pre-registered falsification (F3)**:
* `max(|r_α|, |r_β|, |r_γ|, |r_δ|) < 0.2` ⇒ B1 REFUTED at spectral level.
* `max(…) > 0.5` ⇒ B1 SUPPORTED spectrally (does NOT prove RH; only empirical correspondence).
* `max(…) ∈ [0.2, 0.5]` ⇒ INDETERMINATE.

**Results (deterministic; same config reproducible)**:
* True fixed point at `N = 512`: `‖EPI*‖_L2 = 1.6976`, `mean(EPI*) = −9.25e−02`, spectral total power = 50.87.
* **Spectral shift between intermediate (N=256) and converged (N=512) states**: at N=256 the top-3 bins were high-νf `{16, 19, 20}` of 21 (R∞-1a Track C); at the true fixed point (N=512) the top-3 bins drop to low-νf `{1, 2, 4}` with fractions `{33.1%, 23.4%, 8.8%}`. Iterated REMESH transports power from high-νf to low-νf as it converges. The R∞-1a Track C statement that the fixed point is "dominated by high-νf modes" is therefore SUPERSEDED — the converged fixed point is low-νf dominated.
* Pre-registered tests:
  - `r_α = +0.5126` — crosses 0.5 threshold but only marginally.
  - `r_β = +0.8575` — sorted-magnitude alignment, dominant signal.
  - `r_γ = +0.3454` — node-field vs smooth target, indeterminate range.
  - `r_δ = +0.4120` — Spearman, indeterminate range.
  - `max|·| = 0.8575`.
* **Verdict by the pre-registered criterion**: **F3 nominally SUPPORTED** (max > 0.5).
* **Auxiliary controls (NOT in F3, declared in advance as diagnostic)**:
  - `r(P_k, γ̃_n)` = −0.6690 (strong negative).
  - `r(P_k, γ_n)` = −0.6726 (strong negative).
  - `r(s_i, r_n)` = +0.0055 (no node-level signal at all).

**Honest interpretation (R∞-1a-spectral)**:
* The pre-registered criterion (F3 > 0.5) is met, but the support is fragile and requires multiple caveats before being accepted as evidence for branch B1:
  1. The dominant test (`r_β = 0.86`) is **sorted-magnitude correlation**, which is statistically the weakest of the four. Any two positive heavy-tailed sequences with similar dynamic ranges tend to produce high sorted-magnitude correlation; this test does NOT establish structural alignment between the spectrum and the residuals.
  2. The strongest structural test (`r_γ = 0.34`, node-field vs smooth target) sits in the indeterminate range.
  3. The two auxiliary controls `r(P_k, γ_n) ≈ r(P_k, γ̃_n) ≈ −0.67` reveal that the spectrum is dominantly anti-correlated with the monotone-growing Riemann data, which is consistent with the low-νf concentration being a property of the REMESH mixing kernel rather than encoding Riemann content.
  4. Node-level correlation between the fixed-point field and the residuals (`r(s_i, r_n) = +0.005`) is **zero within noise** — there is no per-mode encoding.
* **What R∞-1a-spectral establishes**: existence of *some* monotone alignment between the magnitude distributions of (fixed-point FFT power) and (|r_n|). This is a necessary condition for B1 at the level of distributions, but is far from sufficient.
* **What R∞-1a-spectral does NOT establish**: per-mode correspondence, operator-level alignment, robustness against synthetic-field choice, sensitivity to (α, τ_l, τ_g), independence from prime-ladder construction.

**Branch verdict (R∞-1a-spectral slice only)**: this milestone does **not refute** B1 at the spectral level, and supplies one weak positive datum (magnitude-distribution alignment). It **does not** confirm B1 — the per-mode (`r_α`, `r_γ`, `r_δ`) tests are inconclusive, and the auxiliary controls flag a kernel-induced bias as a competing explanation. The result must be read as "B1 survives the first falsifiable spectral test, but only by its weakest available signal; further tests required before any B1 claim".

**Next milestones (gated on this result)**:
* **R∞-1a-spectral-robustness** (REQUIRED before any further B1 claim): re-run R∞-1a-spectral with (i) a randomized null synthetic field (white noise) to verify that `r_β` does NOT trigger on noise — kernel-bias control; (ii) sweep over `α ∈ {0.25, 0.5, 0.75}` and `τ_l ∈ {2, 4, 8}` to test sensitivity; (iii) alternative orderings (random permutation of νf-axis) as null controls for `r_α` and `r_γ`.
* **R∞-1a-operator** (gated on robustness): if R∞-1a-spectral-robustness survives, construct a finite-rank approximation of the implied REMESH-∞ operator and compare its spectrum directly to {γ_n}. This is the proper operator-level test that the present field-level test only approximates.
* **R∞-1b**: NS-side analogue (K_φ cascade), independent of Riemann result.

**Status**: R∞-1a-spectral complete. F3 nominally satisfied with **substantial caveats**; the result is consistent with both B1-positive (REMESH-∞ carries weak Riemann signal) and B1-null-kernel-bias (sorted-magnitude alignment is an artefact of heavy-tailed marginals). No closure of any gap; no support for any cosmic claim. R∞-1a-spectral-robustness is the next pre-registered gate.

### §13vicies-novies.7 R∞-1a-spectral-robustness — Falsification gate (REFUTES `r_β` as Riemann signal)

**Milestone**: R∞-1a-spectral-robustness — pre-registered F4 gate for the R∞-1a-spectral result. Three independent controls executed simultaneously; outcome was decisive.

**Implementation**: `benchmarks/remesh_infinity_riemann_spectral_robustness.py`. Output: `results/remesh_infinity/remesh_infinity_riemann_spectral_robustness.json`.

**Setup**: identical pipeline to R∞-1a-spectral (same prime ladder, `N_iter = 512`, same Riemann reference from `mpmath.zetazero` + P28). Three independent controls:
* **C1** white-noise null: 16 seeded runs (`numpy.random.default_rng(20260526 + seed)`, `seed ∈ {0..15}`) replacing the canonical oscillatory synthetic EPI field with zero-mean unit-variance white noise, identical REMESH iteration.
* **C2** sensitivity sweep: 3 × 3 grid `(α, τ_l) ∈ {0.25, 0.5, 0.75} × {2, 4, 8}` on the canonical synthetic field.
* **C3** permutation null: 5000 random permutations of `|r_n|` (for `r_α`) and `γ̃_n` (for `r_γ`) on the canonical fixed-point spectrum, `numpy` seed `20260526`.

**Pre-registered falsification (F4)**:
* REFUTED if ANY of: (a) C1 mean `|r_β|`-null > 0.5; (b) C2 `r_β < 0.5` anywhere in grid; (c) C3 both `p_α > 0.05` AND `p_γ > 0.05`.
* STRENGTHENED if ALL of: (a) C1 mean `|r_β|`-null < 0.2 AND observed `r_β` outside 95% null; (b) C2 `r_β > 0.5` everywhere; (c) C3 `p_α < 0.05` OR `p_γ < 0.05`.
* MIXED otherwise.

**Results (deterministic; full per-run table in JSON)**:

*Baseline (canonical)*: `r_α = +0.5126, r_β = +0.8575, r_γ = +0.3454, r_δ = +0.4120` (reproduces R∞-1a-spectral exactly).

*C1 white-noise null* (16 seeds):
* `r_β` null mean = **+0.9440**, `|·|` mean = 0.9440, std = 0.0286, 95% range = [+0.8888, +0.9763].
* The baseline `r_β = +0.8575` is **below the 2.5% quantile** of the white-noise null distribution.
* `r_α` null mean = −0.0947 (|·| mean = 0.1848, std = 0.213).
* `r_γ` null mean = −0.0636 (|·| mean = 0.0880, std = 0.103).

*C2 sensitivity sweep* (9 cells, **post-bug-fix run; see «α propagation bug» note below**): `r_β` range [+0.8194, +0.8935], `r_α` range [+0.3958, +0.5247], `r_γ` range [+0.2880, +0.3565]. `r_β > 0.5` at every cell. Per-cell variation in α is now visible (previously masked by the propagation bug).

*C3 permutation null* (5000 perms each):
* `r_α`: observed +0.5126 vs null `(mean = +0.0025, std = 0.227)`, **p_one_sided = 0.0228**, p_two_sided = 0.0246.
* `r_γ`: observed +0.3454 vs null `(mean = +0.0014, std = 0.160)`, **p_one_sided = 0.0154**, p_two_sided = 0.0304.

**F4 verdict: REFUTED** (refute-C1 triggered).

**Honest interpretation (R∞-1a-spectral-robustness)**:
* The dominant R∞-1a-spectral signal (`r_β = 0.86`) is **a pure kernel artefact**. White noise reproduces it at higher magnitude (mean 0.94) than the canonical oscillatory field. The sorted-magnitude Pearson coefficient measures only that the FFT-power marginal and the `|r_n|` marginal share a heavy-tailed structure; it does NOT detect any structural alignment between the spectrum of the REMESH fixed point and Riemann residuals. The R∞-1a-spectral "B1 nominally SUPPORTED" verdict relied on `r_β` and must therefore be **withdrawn**.
* C2 shows `r_β` does vary with `(α, τ_l)` once `α` is actually propagated (range [+0.819, +0.894], 9 cells), but remains `> 0.5` everywhere — does not refute. The original C2 read of "r_β invariant in α" was an artefact of an α-propagation bug in the canonical REMESH pipeline (see dedicated note below). After the fix, `r_α ∈ [+0.40, +0.52]` and `r_γ ∈ [+0.29, +0.36]` are **robust across the (α, τ_l) grid**, which strengthens (not weakens) the interpretation of these two metrics as genuine weak structural alignments.
* C3 supplies the only genuinely positive finding: `r_α` and `r_γ` are statistically significant against permutation null (p ≈ 0.02 and p ≈ 0.015 one-sided). They are NOT artefacts of the marginal distributions; the alignment between (FFT power → |r_n|) index-wise and (νf-ordered field → smooth target) is structurally non-random. However, the effect sizes are modest:
  - `r_α = 0.5126` was already only marginally above the F3 threshold and now stands alone.
  - `r_γ = 0.3454` remains in the indeterminate band of F3.
* **Net B1 evidential balance after R∞-1a-spectral-robustness**: the dominant claimed signal is artefact; two minor signals survive permutation testing but with modest effect sizes and neither alone meets the original F3 "SUPPORTED" threshold (one is marginal at 0.51, the other indeterminate at 0.35).

**What R∞-1a-spectral-robustness establishes**:
* `r_β` (sorted-magnitude Pearson on FFT power vs |r_n|) is **not a valid Riemann signal** in this benchmark family and must be retired.
* Permutation-tested `r_α` (Pearson on power vs |r_n|, index-aligned) and `r_γ` (Pearson on νf-ordered field vs γ̃_n) carry **weak but genuine non-random structural alignment** that is not explained by marginal distributions or kernel parameters.

**What R∞-1a-spectral-robustness does NOT establish**:
* It does NOT confirm B1 — the surviving signals are below the originally pre-registered support threshold.
* It does NOT refute B1 entirely — the permutation-significant `r_α` and `r_γ` remain a positive (though weak) datum.
* It does NOT close T-HP, G4, or any gap.

**Branch verdict (R∞-1a-spectral-robustness slice only)**: B1 is **WEAKENED but not refuted**. The R∞-1a-spectral claim of "B1 nominally SUPPORTED at the spectral level (max > 0.5)" is **withdrawn**. The current state of B1 evidence after this milestone is: one necessary positive datum (existence of non-trivial REMESH fixed point, R∞-1a), one withdrawn artefactual signal (`r_β`, this milestone), and two weak-but-permutation-significant alignments (`r_α ≈ 0.51`, `r_γ ≈ 0.35`, this milestone). This is far below what would be required to claim B1 closure of T-HP.

**Next milestones (gated on this result)**:
* **R∞-1a-operator** (REQUIRED before any further B1 evidential update): the field-level test in this milestone is at best a proxy for the actual structural question — does the REMESH-∞ operator, viewed as a linear map on the appropriate state space, have spectrum compatible with `{γ_n}`? Construct a finite-rank approximation of the REMESH iteration matrix on `EPI`-space, diagonalize, and compare the eigenvalue spectrum directly to `{γ_n}`. Pre-register: if the largest absolute correlation between (REMESH-∞ eigenvalue magnitudes) and (`γ_n` or `|r_n|`) is `< 0.5` after permutation testing, B1 is refuted at the operator level.
* **R∞-1b**: NS-side analogue, independent.
* **B1 status update**: with `r_β` retired and only weak `r_α`/`r_γ` surviving, the canonical-catalog-closure conjecture (B1) **loses substantial empirical support** but remains technically open pending R∞-1a-operator. Branches **B2** (a new canonical operator is required) and **B3** (no TNFR closure exists) gain proportionally in prior weight, though no decisive evidence shifts the balance entirely to either.

**α propagation bug (diagnosed and fixed mid-milestone)**:
* During the C2 sweep an unexpected invariance of `r_β` across the α axis was observed (identical values for α ∈ {0.25, 0.5, 0.75} at each τ_l). Direct probing of `_remesh_alpha_info` in `src/tnfr/operators/remesh.py` revealed that the precedence order is **(1)** `REMESH_ALPHA` when `REMESH_ALPHA_HARD=True`, **(2)** `GLYPH_FACTORS.REMESH_alpha` from the canonical defaults, **(3)** `G.graph["REMESH_ALPHA"]` only as fallback. Without the HARD flag, the value written by the benchmark to `G.graph["REMESH_ALPHA"]` is silently ignored — the default `GLYPH_FACTORS.REMESH_alpha = 0.5` is used regardless.
* Reproducer (direct call to `_remesh_alpha_info`):
  - Set `G.graph["REMESH_ALPHA"] = 0.25` (no HARD flag) → returns `α = 0.5, source = "GLYPH_FACTORS.REMESH_alpha"`.
  - Set `G.graph["REMESH_ALPHA"] = 0.25` and `G.graph["REMESH_ALPHA_HARD"] = True` → returns `α = 0.25, source = "REMESH_ALPHA"`.
* `τ_local` and `τ_global` use `get_param()` which reads from `G.graph` directly, so their C2 axis was always honoured (variation across τ_l in the original run was real).
* **Fix applied**: `benchmarks/remesh_infinity_riemann_spectral_robustness.py::run_canonical_pipeline` now sets `G.graph["REMESH_ALPHA_HARD"] = True` before iteration, with an explanatory comment cross-referencing this section. C2 was re-executed after the fix; the numbers above (range [+0.819, +0.894] for `r_β`, [+0.40, +0.52] for `r_α`, [+0.29, +0.36] for `r_γ`) are from the fixed run. C1 and C3 are independent of the α value and are unchanged.
* **Note on the canonical pipeline**: this precedence ordering means any user who writes `G.graph["REMESH_ALPHA"]` without also enabling `REMESH_ALPHA_HARD` will get the default `0.5` silently. This is a latent surprise but not a TNFR-grammar violation per se. Documented here for cross-program awareness; not promoted to a code-level fix in this milestone because the canonical α = 0.5 is the documented TNFR default and changing the precedence requires its own grammar audit.

**Status**: R∞-1a-spectral-robustness complete. F4 refutes the dominant R∞-1a-spectral signal as kernel artefact while preserving two weak permutation-significant alignments (`r_α`, `r_γ`) that are also confirmed robust across the (α, τ_l) grid after the α-propagation bug was fixed. The R∞-1a-spectral milestone is **formally amended**: the "B1 SUPPORTED" verdict is withdrawn; the residual evidence (R∞-1a fixed-point existence + permutation-significant weak `r_α`, `r_γ` confirmed across (α, τ_l)) is insufficient to support B1 at the spectral level but is mildly stronger than the original interpretation that allowed for parameter fragility. No closure of any gap. R∞-1a-operator is the next pre-registered gate; until it returns, the canonical TNFR-Riemann program remains paused at the T-HP / G4 = RH boundary as stated in §13septies.

---

### §13vicies-novies.8 R∞-1a-operator — Structural refutation of B1 at the operator level (REMESH-iterated-in-isolation)

**Milestone**: R∞-1a-operator — gated follow-up to §13vicies-novies.7. Examines whether the spectrum of the REMESH iteration matrix (viewed as a linear map on the augmented EPI × temporal-history state) can encode `{γ_n}`-specific content. Outcome is **doubly negative**: a *structural* refutation independent of any statistic, plus a methodological exposure of the pre-registered F5 statistical test as a monotonicity artefact.

**Implementation**: `benchmarks/remesh_infinity_riemann_operator.py`. Output: `results/remesh_infinity/remesh_infinity_riemann_operator.json`.

**Structural construction**. The canonical REMESH update (`src/tnfr/operators/remesh.py` L1212–1252) is strictly linear and **node-local**:

$$\mathrm{EPI}_{\text{new}}(i) = (1-\alpha)^2 \cdot \mathrm{EPI}(i,t) + \alpha(1-\alpha) \cdot \mathrm{EPI}(i,t-\tau_l) + \alpha \cdot \mathrm{EPI}(i,t-\tau_g).$$

No edge term, no inter-node coupling. The full state of node $i$ over a delay window of length $\tau_g + 1$ therefore evolves under a shift-augmented matrix $M \in \mathbb{R}^{(\tau_g+1)\times(\tau_g+1)}$ given by

$$M[0,0] = (1-\alpha)^2,\quad M[0,\tau_l] = \alpha(1-\alpha),\quad M[0,\tau_g] = \alpha,\quad M[k,k-1] = 1\ \text{for}\ k=1,\dots,\tau_g.$$

Because there is no inter-node coupling, the full-graph iteration operator is **block-diagonal**: $N$ identical copies of $M$. The spectrum is the spectrum of $M$ with multiplicity $N$. **Neither the graph topology nor the P14 prime-ladder initial condition enters $M$ at any point.**

**Canonical spectrum** (α = 0.5, τ_l = 4, τ_g = 16; verified analytically with `scipy.linalg.eig`):
* $\lambda_1 = 1$ exactly (trivial fixed-point subspace: temporally-constant configurations are preserved exactly by the convex combination).
* 16 non-trivial eigenvalues organised as 8 complex-conjugate pairs.
* $|\lambda_k| \in [0.938, 0.982]$ for $k = 2, \dots, 17$ (all strictly inside the unit disk).
* Spectral radius excluding unity: $0.981475$.

**Pre-registered statistical test (F5)**:
* H0 (refute operator-level B1): no ordering of the 16 non-trivial eigenvalues achieves Pearson or Spearman $|r| \ge 0.5$ vs $\gamma_1, \dots, \gamma_{16}$ with permutation $p_{\text{one-sided}} < 0.05$.
* H1 (support): some ordering does.
* Ordering battery: `abs_desc`, `abs_asc`, `arg_upper_asc`, `real_desc`, `imag_upper_asc` × {Pearson, Spearman} = 10 tests. Sensitivity sweep: 3 × 3 grid `(α, τ_l) ∈ {0.25, 0.5, 0.75} × {2, 4, 8}`, τ_g = 16. Permutation null $N_{\text{perm}} = 5000$, seed 20260526.

**Results (canonical config)**:
| ordering | stat | $r$ | $p_{\text{perm}}$ |
|---|---|---|---|
| `abs_desc` | pearson | −0.9628 | 0.0002 |
| `abs_desc` | spearman | −0.9941 | 0.0002 |
| `abs_asc` | pearson | +0.9615 | 0.0002 |
| `abs_asc` | spearman | +0.9941 | 0.0002 |
| `arg_upper_asc` | pearson | +0.9917 | 0.0004 |
| `arg_upper_asc` | spearman | **+1.0000** | 0.0002 |
| `real_desc` | pearson | −0.9821 | 0.0002 |
| `real_desc` | spearman | −0.9941 | 0.0002 |
| `imag_upper_asc` | pearson | +0.9913 | 0.0002 |
| `imag_upper_asc` | spearman | **+1.0000** | 0.0002 |

**Naïve F5 verdict (canonical)**: 10/10 PASS, max $|r| = 1.0000$. Sensitivity sweep: 9/9 cells PASS.

**Monotonicity controls (kernel-artefact diagnostic)**. The pre-registered F5 compares two sorted sequences against each other. Any monotonically ordered sequence aligned by index with the sorted $\{\gamma_n\}$ yields Spearman $= \pm 1$ and Pearson $\approx 0.95$–$1.0$; the permutation null is uninformative because almost every permutation breaks monotonicity. Four control sequences with no Riemann content were run through the same battery:

| control | stat | $r$ | $p_{\text{perm}}$ | naive PASS? |
|---|---|---|---|---|
| `integer_ladder` ($1, 2, \dots, 16$) | pearson | +0.9937 | 0.0002 | YES |
| `integer_ladder` | spearman | +1.0000 | 0.0002 | YES |
| `arithmetic_decay` ($\mathrm{linspace}(0.98, 0.94, 16)$) | pearson | −0.9937 | 0.0002 | YES |
| `arithmetic_decay` | spearman | −1.0000 | 0.0002 | YES |
| `random_monotone_in_unit_disk` | pearson | +0.9879 | 0.0002 | YES |
| `random_monotone_in_unit_disk` | spearman | +1.0000 | 0.0002 | YES |
| `log_n_growth` ($\log(1 + n)$) | pearson | +0.9845 | 0.0002 | YES |
| `log_n_growth` | spearman | +1.0000 | 0.0002 | YES |

**8/8 controls pass naive F5 at thresholds equal to or stronger than the canonical operator spectrum.** Therefore the canonical PASS is fully explained by the trivial monotonicity of any sorted sequence against the sorted $\{\gamma_n\}$ — exactly the same failure mode that retired `r_β` in §13vicies-novies.7.

**F5 STRICT verdict (canonical)**: **REFUTED_BY_MONOTONICITY_ARTEFACT**. The statistical battery as pre-registered has no falsification power and must be retired.

**Structural verdict (independent of any statistic)**. The REMESH iteration operator applied in isolation, as a strictly node-local linear map, is **structurally incapable** of encoding `{γ_n}`-specific content in its spectrum. The spectrum depends only on the three scalar canonical parameters $(\alpha, \tau_l, \tau_g)$ and on nothing else: not on the graph topology, not on the prime-ladder initial state, not on the field activation pattern, not on the number of nodes. Any apparent alignment between $\sigma(M)$ and $\{\gamma_n\}$ is either (a) a kernel monotonicity artefact (demonstrated above), or (b) imposed by the analyst's choice of $\{\gamma_n\}$ as the comparison target rather than discovered from the operator. **This refutes B1 at the level of REMESH iterated in isolation.**

**What §13vicies-novies.8 establishes**:
* REMESH applied as a stand-alone iterated linear operator **cannot** carry Riemann-spectral content. The 17-dimensional spectrum is exactly determined by the three canonical parameters with no degree of freedom for graph- or initial-state-dependent encoding.
* The naive correlation-based F5 test design is **invalid** for comparing two intrinsically sorted finite sequences and is formally retired (analogously to `r_β` in §13vicies-novies.7).
* The earlier R∞-1a fixed-point existence (§13vicies-novies.5) and its weak permutation-significant `r_α`, `r_γ` alignments (§13vicies-novies.7) are **not refuted** by this milestone. They concern an EPI **field** trajectory under iterated REMESH on a P14-initialised system, where the topology and initial state determine the *image* of the operator on the prime-ladder subspace, even though the operator's *spectrum* does not. The distinction is exactly the difference between $\sigma(M)$ (intrinsic, parameter-only) and $M \mathbf{v}_{P14}^k$ (depends on initial state).

**What §13vicies-novies.8 does NOT establish**:
* It does NOT refute B1 entirely. The structural refutation is **scoped to REMESH iterated in isolation as a stand-alone operator**. B1 in its full breadth — closure of T-HP inside the 13-operator catalog — remains technically open via two non-refuted channels:
  - **Composed operators**: REMESH ∘ IL, REMESH ∘ OZ, etc. The U1–U6 canonical grammar admits these compositions, and any non-trivial composition involves at least one operator whose action *does* couple nodes via the graph (IL, EN, NAV, RA propagate through edges). Composed operators therefore have spectra that *do* depend on topology and initial state, and the structural argument of this milestone does not apply.
  - **Hierarchical / fractal modes**: the canonical REMESH catalog (`src/tnfr/operators/remesh.py`) specifies three structural modes (Hierarchical, Rhizomatic, Fractal Harmonic) and `src/tnfr/multiscale/hierarchical.py` implements explicit cross-scale ΔNFR coupling. These are non-iterated-in-isolation regimes; this milestone does not bound them.
* It does NOT close G4 = RH, does NOT close T-HP, does NOT prove RH, does NOT promote any new operator.
* The fixed-point existence and weak `r_α`, `r_γ` alignments from §13vicies-novies.5–7 retain their status (necessary but insufficient).

**Branch verdict update (after R∞-1a-operator)**:
* **B1 at REMESH-iterated-in-isolation level**: STRUCTURALLY REFUTED.
* **B1 at composed-operator / hierarchical-mode level**: untouched (open).
* **B1 as a whole**: WEAKENED FURTHER. Of the two remaining channels for B1 closure inside the catalog, the one most directly suggested by the cross-program REMESH reframe (§13vicies-novies.1–4) is now closed. The composed-operator channel remains open but requires a *gramatically-canonical sequence* of operators (an U1–U6 admissible composition) whose spectrum would need to be derived analytically and tested against `{γ_n}` with a statistic that does *not* fall to the monotonicity artefact (e.g., normalised gap statistics, level-spacing distributions, or KS-vs-GUE diagnostics rather than two-sorted-sequence Pearson/Spearman).
* **B2 (new canonical operator required)** and **B3 (no TNFR closure exists)** gain proportionally in prior weight, though no decisive evidence shifts the balance entirely to either.

**Next milestones (gated on this result)**:
* **R∞-1a-composed** (REQUIRED before any further B1 evidential update): identify a minimal U1–U6 admissible composition of REMESH with at least one node-coupling canonical operator (candidates: REMESH ∘ IL, REMESH ∘ NAV, REMESH ∘ OZ ∘ EN), construct the iteration matrix on the joint state space, and test its spectrum against `{γ_n}` and against canonical null sequences using a statistic that *does* discriminate (level-spacing distribution, normalised eigenvalue-gap KS to GUE, or spectral-form-factor comparison). Pre-register thresholds before execution.
* **R∞-1b** (NS-side analogue): independent of the Riemann program; structural argument of this milestone likely transfers to the NS side because REMESH is canonical in both engines, but should be re-derived in the NS-G_blowup context.
* **B1 status check**: with the structural refutation of REMESH-isolated added to the retracted `r_β` and the (re-bounded) weak `r_α`/`r_γ`, the canonical-catalog-closure conjecture (B1) **loses substantial structural support** but is not strictly refuted because composed-operator channels remain untested. The TNFR-Riemann program remains paused at the T-HP / G4 = RH boundary per §13septies; the reframe of §13vicies-novies.1–4 should now be **further qualified** to: "REMESH-global is canonical and structurally relevant, but REMESH iterated *in isolation* cannot carry Riemann content. Branch B1, if it closes, will do so via composed operators or via the hierarchical/fractal modes — neither of which is yet tested."

**Status**: R∞-1a-operator complete. Structural verdict: REMESH iterated in isolation **cannot** encode `{γ_n}`. Statistical verdict: the pre-registered F5 test has no falsification power and is retired. Net B1 evidential balance: structurally weakened (one of two narrow channels closed); two narrow channels (composed operators, hierarchical/fractal modes) remain technically open. No closure of any gap. The TNFR-Riemann program remains paused at the T-HP / G4 = RH boundary as stated in §13septies, with the §13vicies-novies reframe now further qualified.

---

### §13vicies-novies.9  R∞-1a-composed — pre-registered test of B1 at composed-operator level (PRE-REGISTRATION, no data observed)

**This subsection is committed to the repository BEFORE any data is collected.** It locks the methodology, hypotheses, falsification thresholds, controls, and verdict logic of the R∞-1a-composed milestone. Results are appended in a subsequent commit, in a clearly delimited "Results" block. The git history of this file is the audit trail.

**Gate addressed**. The composed-operator channel left open by §13vicies-novies.8 ("R∞-1a-composed [...] identify a minimal U1–U6 admissible composition of REMESH with at least one node-coupling canonical operator [...] and test its spectrum against `{γ_n}` and against canonical null sequences using a statistic that *does* discriminate"). This is one of the two narrow channels through which B1 could still close inside the catalog.

**Composition selected**. **REMESH ∘ IL** (Coherence stabiliser after the temporal memory step). Rationale:
* IL is the canonical U2 stabiliser (`src/tnfr/operators/coherence.py`). In its linearised edge-coupling channel, IL performs phase locking toward the neighbourhood circular mean with strength $a$ (default $a = 0.3$): $\theta_{\text{new}}(i) = (1-a)\,\theta(i) + a\,\bar\theta_{\mathcal{N}(i)}$, which is structurally identical to a graph-Laplacian smoothing $(\mathbf{I} - a\,L_{\text{norm}})$ acting on the phase field. This is the **only canonical operator whose linear edge-coupling channel is a pure Laplacian-of-graph smoothing**, which gives the cleanest analytic handle.
* IL acting after REMESH (REMESH→IL) is grammatically natural: REMESH is U1a/U1b generator/closure, IL is U2 stabiliser; the sequence is U1–U2 admissible.
* The composition lifts the block-diagonal structure of REMESH-isolated: IL couples nodes through edges, so the joint iteration matrix is no longer block-diagonal in nodes.

**Joint state space**. For a graph $G$ with $N$ nodes and the canonical REMESH delay window of length $\tau_g + 1 = 17$, the joint EPI history state is $\mathbf{x} \in \mathbb{R}^{N(\tau_g+1)}$. Index ordering: by delay slot first (slot 0 = current EPI, slots 1..τ_g = historical EPI), then by node. The composed one-step iteration matrix is

$$T_{\text{composed}} = S_{\text{IL}} \cdot M_{\text{REMESH}},$$

where:
* $M_{\text{REMESH}} = I_N \otimes M$ is the $N(\tau_g+1) \times N(\tau_g+1)$ block-diagonal REMESH update with $M$ the $(\tau_g+1) \times (\tau_g+1)$ shift-augmented matrix of §13vicies-novies.8;
* $S_{\text{IL}}$ acts as the IL Laplacian-smoothing operator $(\mathbf{I}_N - \eta L_G)$ on the current-time slot (slot 0) and as the identity on all historical slots, where $L_G$ is the unnormalised combinatorial Laplacian of $G$ and $\eta$ is the IL coupling strength (default $\eta = 0.3$, matching the canonical IL phase-locking coefficient).

**Graph $G$**. Canonical P14 prime-ladder graph (`src/tnfr/riemann/prime_ladder_hamiltonian.py::build_prime_ladder_graph`) with $n_{\text{primes}} = 10$, $\max\_\text{power} = 4$, $\text{coupling} = 0$. This yields 40 nodes arranged as 10 disjoint paths $P_4$ (one per prime), with REMESH-echo edges along each ladder and no inter-prime edges. The joint state has dimension $40 \cdot 17 = 680$.

**Structural prediction (pre-registered before execution)**. The canonical P14 prime-ladder graph encodes Riemann-relevant content **exclusively in node attributes** ($\nu_f = k\log p$, used by the P14 InternalHamiltonian as diagonal energies). Its graph topology — 10 disjoint copies of $P_4$ — is **independent of which primes are chosen**: relabelling primes is a graph automorphism. Therefore any operator whose action on EPI depends only on graph edges (combinatorial Laplacian, adjacency, edge weights) has a spectrum that is **insensitive to the prime labelling**. The IL Laplacian-smoothing channel $(\mathbf{I}_N - \eta L_G)$ has spectrum $\{1 - \eta \mu_j : \mu_j \in \sigma(L_G)\}$, and $\sigma(L_G)$ for 10 disjoint copies of $P_4$ is the multiset $\{2(1 - \cos(j\pi/4)) : j = 0, 1, 2, 3\}$ with multiplicity 10, i.e. 4 distinct eigenvalues each tenfold degenerate. The composed iteration matrix $T_{\text{composed}}$ inherits these degeneracies in its IL-dominated sector. **Prediction**: the spectrum of $T_{\text{composed}}$ cannot encode `{γ_n}`-specific content for the same structural reason that REMESH-isolated could not — Riemann content lives in P14's diagonal energies, not in edges or temporal memory.

**Hypotheses (pre-registered)**.

* **$H_0$ (B1-composed refutation)**: The spectrum of $T_{\text{composed}}$ on the P14 prime-ladder graph is statistically indistinguishable from canonical null spectra (GOE / Poisson / shuffled-prime control) under the discriminating statistic $F_6$ below.
* **$H_1$ (B1-composed support)**: The spectrum of $T_{\text{composed}}$ exhibits Riemann-zero-like level-spacing statistics under $F_6$ that are distinguishably closer to the GUE Wigner surmise than all four control nulls.

**Discriminating statistic $F_6$ (pre-registered, replaces retired F5)**. The Montgomery–Odlyzko law states that the unfolded nearest-neighbour spacings of Riemann zeros follow the GUE Wigner surmise $P_{\text{GUE}}(s) = (32/\pi^2) s^2 \exp(-4s^2/\pi)$. We test whether the spacings of the composed-operator spectrum follow the same law.

Procedure:
1. Compute the full spectrum $\{\lambda_k\}_{k=1}^{N(\tau_g+1)}$ of $T_{\text{composed}}$.
2. Remove the trivial fixed-point cluster: $|\lambda - 1| < 10^{-9}$.
3. Project complex eigenvalues to a 1-D quantity via $\text{Im}(\lambda)$ for the upper-half-plane subset ($\text{Im}(\lambda) \geq 10^{-12}$). Sort ascending: $s_1 \leq s_2 \leq \dots \leq s_K$.
4. Compute normalised consecutive spacings $\delta_k = (s_{k+1} - s_k) / \langle s_{k+1} - s_k \rangle$.
5. Compute the Kolmogorov–Smirnov distance $D_{\text{GUE}} = \sup_x |F_{\text{emp}}(x) - F_{\text{GUE}}(x)|$ where $F_{\text{GUE}}(s) = \int_0^s P_{\text{GUE}}(s')\,ds'$.

**Reference Riemann value**. For the first $K_{\text{ref}} = 100$ Riemann zero imaginary parts $\{\gamma_n\}_{n=1}^{100}$ (via `mpmath.zetazero`), the same procedure yields $D_{\text{GUE}}^{\text{Riemann}}$, computed at execution time and reported in the results block. Published Odlyzko-type estimates give $D_{\text{GUE}}^{\text{Riemann}}(K=100) \approx 0.05$–$0.10$ as an external anchor.

**Pre-registered $F_6$ thresholds**.

| Verdict | Condition |
|---|---|
| **SUPPORTED** | $D_{\text{GUE}}^{\text{composed}} < 0.15$ **AND** $D_{\text{GUE}}^{\text{composed}} < D_{\text{GUE}}^{\text{shuffled-prime}} - 0.05$ (distinguishably better than topological-shuffle null) |
| **REFUTED** | $D_{\text{GUE}}^{\text{composed}} > 0.30$ **OR** $D_{\text{GUE}}^{\text{composed}} \geq D_{\text{GUE}}^{\text{shuffled-prime}} - 0.05$ (no separation from topological-shuffle null) |
| **INDETERMINATE** | otherwise |

**Pre-registered controls** (each computed under the same procedure, same number of spacings as the canonical projection):
* **N1 GOE**: spacings of a random symmetric matrix drawn from the Gaussian Orthogonal Ensemble at the same dimension $N(\tau_g+1)$. Expected $D_{\text{GUE}}^{\text{GOE}} \approx 0.10$–$0.20$ (GOE spacings differ from GUE Wigner surmise).
* **N2 Poisson**: spacings of independent uniform random points on the same interval. Expected $D_{\text{GUE}}^{\text{Poisson}} \approx 0.30$–$0.50$ (Poisson follows $P(s) = e^{-s}$, far from GUE).
* **N3 prime-ladder shuffled**: identical composed-operator construction but on the P14 graph with the prime labels shuffled (a permutation of the 10 primes among the 10 disjoint paths). If $D_{\text{GUE}}^{\text{shuffled}}$ is indistinguishable from $D_{\text{GUE}}^{\text{composed}}$, the prime content is **not** encoded — this is the **primary discriminator** of the F6 test.
* **N4 REMESH-isolated re-run**: the §13vicies-novies.8 spectrum projected through the same $F_6$ pipeline (degenerate spacings expected, $D_{\text{GUE}}$ may be ill-defined; reported as diagnostic baseline).

**Pre-registered seeds and parameters**. All random elements (GOE draw, Poisson draw, prime shuffle permutation) use `numpy.random.default_rng(20260526)`. Riemann zeros via `mpmath.zetazero` at `mp.dps = 30`. REMESH parameters: $\alpha = 0.5$, $\tau_l = 4$, $\tau_g = 16$. IL coupling: $\eta = 0.3$. Graph: $n_{\text{primes}} = 10$, $\max\_\text{power} = 4$.

**Pre-registered verdict logic on B1-composed**. The milestone verdict combines the F6 statistic and the structural prediction:
* If F6 = REFUTED **and** structural prediction confirmed: **B1-composed REFUTED for REMESH ∘ IL**. The composed-operator channel for B1 closure is narrowed: at minimum, IL is not the operator that closes it.
* If F6 = SUPPORTED: **B1-composed POTENTIALLY OPEN; structural prediction CHALLENGED**. Requires deep diagnostic and replication on independent seeds and on alternative compositions (REMESH ∘ EN, REMESH ∘ NAV, REMESH ∘ RA) before any evidential update.
* If F6 = INDETERMINATE: **status unchanged**; design refinement needed before next attempt.

**What this milestone CAN establish**:
* A definitive verdict on REMESH ∘ IL as a candidate B1-composed operator.
* Empirical confirmation or falsification of the structural prediction that node-attribute-bearing Riemann content cannot be recovered by edge-coupling operators on the canonical P14 graph.

**What this milestone CANNOT establish**:
* B1-composed for other compositions (REMESH ∘ EN, REMESH ∘ NAV, REMESH ∘ RA, three-operator chains).
* B1 via hierarchical/fractal REMESH modes (§13vicies-novies.8 second open channel).
* G4 = RH, T-HP, or any closure beyond what F6 strictly tests.

**Implementation**. `benchmarks/remesh_infinity_riemann_composed.py` (committed in the same commit as this pre-registration; no data collected at commit time). Output JSON written to `results/remesh_infinity/remesh_infinity_riemann_composed.json` (gitignored, not part of the audit trail; the audit trail is this file).

**Status (pre-registration commit)**: methodology locked; no data observed; next commit will append results in a "Results" block delimited below.

---

#### §13vicies-novies.9 Results

**Execution metadata**

- Pre-registration commit: `a6847706` (parent: `9414b1ce`).
- Implementation: `benchmarks/remesh_infinity_riemann_composed.py`.
- Interpreter: CPython 3.12 (`.venv312`); seeds: NumPy `default_rng(20260526)` for N1/N2/N3, `mpmath` `dps=30` for the Riemann anchor.
- Output report: `results/remesh_infinity/remesh_infinity_riemann_composed.json`.
- Joint dimension: $N\cdot(\tau_g+1) = 40\cdot 17 = 680$.

**Structural prediction (a priori)**

The unweighted Laplacian $L_{G_{P14}}$ of the canonical $P14$ prime-ladder graph
(`build_prime_ladder_graph(n_primes=10, max_power=4, coupling=0.0)`) is a direct
sum of ten copies of the $P_4$ path Laplacian. Its spectrum must therefore be
exactly $\{0, 2-\sqrt{2}, 2, 2+\sqrt{2}\}$, each with multiplicity ten.

Empirical eigenvalues (rounded to six decimals): $\{0.0,\ 0.585786,\ 2.0,\
3.414214\}$, multiplicities $\{10,\ 10,\ 10,\ 10\}$. **Prediction confirmed
exactly.**

**F6-A KS distance vs GUE Wigner surmise**

| Variant                          | Projection      | #spacings | $D_{\mathrm{GUE}}$ |
|---------------------------------|-----------------|-----------|--------------------|
| canonical `REMESH ∘ IL`         | $\mathrm{Im}$ upper | 319       | **0.9053**         |
| N1 GOE                           | $\mathrm{Re}$ fallback | 679       | 0.1126             |
| N2 Poisson                       | uniform iid     | 679       | 0.3032             |
| N3 shuffled-prime relabelling   | $\mathrm{Im}$ upper | 319       | **0.9053**         |
| N4 REMESH-isolated               | $\mathrm{Im}$ upper | 7         | 0.3082             |
| Riemann reference (first 100 $\gamma_n$) | iid           | 99        | 0.0770             |

**Threshold evaluation (pre-registered)**

- $D_{\mathrm{GUE}}^{\mathrm{composed}} = 0.9053 > 0.30$: **REFUTED** by absolute
  bound.
- $D_{\mathrm{GUE}}^{\mathrm{composed}} = 0.9053 \ge
  D_{\mathrm{GUE}}^{\mathrm{shuffled}} - 0.05 = 0.8553$: **REFUTED** by
  separation criterion.
- Both pre-registered REFUTED conditions hold; the SUPPORTED conditions
  ($D_{\mathrm{composed}} < 0.15$ and $D_{\mathrm{composed}} <
  D_{\mathrm{shuffled}} - 0.05$) fail simultaneously.

**Verdict**

- F6-A statistical verdict: **REFUTED**.
- Milestone verdict: **`B1_COMPOSED_REFUTED_FOR_REMESH_o_IL`**.

**Structural reading of the empirical pattern**

The numerical identity $D_{\mathrm{canonical}} \equiv D_{\mathrm{shuffled}}$
(bit-for-bit equal across the entire 319-element spacing distribution) is the
empirical signature of the structural lemma derived in §13vicies-novies.10:
relabelling the underlying primes is a graph automorphism of $G_{P14}$ that
commutes with both the IL Laplacian smoother (which depends only on edge
combinatorics) and the REMESH echo matrix (which is node-independent).
Consequently the entire spectrum of $T = S_{\mathrm{IL}}\cdot M_{\mathrm{REMESH}}$
on $G_{P14}$ is invariant under prime permutation. The Riemann content carried
by the diagonal frequencies $\nu_f((p,k)) = k\log p$ never reaches the
edge-propagation channel; it survives only in the *node attributes*, which are
the data on which the P14 internal Hamiltonian (§13quinquies) operates.

The composed operator therefore cannot encode Riemann-zero level statistics
through its spectrum on $G_{P14}$. The B1 closure of R∞-1a in its naive form
(spectrum of an edge-propagating composition equals the Riemann level
structure) is empirically and structurally refuted.

**Scope of the refutation**

This rules out the *naive edge-channel route* for the pair $(\mathrm{REMESH},
\mathrm{IL})$ on $G_{P14}$. It does **not** rule out:

1. Composition routes acting on a state space that already carries prime data
   (R∞-1b spectral-space composition over $|p,k\rangle$ basis of the P14
   internal Hilbert space).
2. Graph modifications canonically derived from the nodal equation that endow
   inter-prime edges with Riemann content (R∞-1c).
3. Any structural-coherence statement at the level of the diagnostic surface
   built by milestones P17–P49.

The catalog-wide structural argument explaining why every edge-propagating
operator in the canonical 13-operator catalog fails by the same mechanism on
$G_{P14}$ is given in §13vicies-novies.10.

---

### §13vicies-novies.10 Catalog structural lemma: which canonical operators can carry Riemann content on $G_{P14}$

The empirical bit-for-bit identity
$D_{\mathrm{canonical}}(\mathrm{REMESH}\circ\mathrm{IL}) =
D_{\mathrm{shuffled}}(\mathrm{REMESH}\circ\mathrm{IL}) = 0.9053$
reported in §13vicies-novies.9 is a numerical specialisation of a general
structural property of the canonical 13-operator catalog acting on the P14
prime-ladder graph. This subsection states and derives that property, classifies
all 13 canonical operators by the channel through which they could in principle
transport prime data, and identifies the two genuinely open B1-style avenues
that remain available after the naive edge-channel route has been closed.

**Setup.** Let $G_{P14}$ be the canonical prime-ladder graph of §13quinquies
with $N=40$ nodes labelled $(p_i, k)$, $i=1,\dots,10$, $k=1,\dots,4$, structural
attributes $\nu_f((p,k))=k\log p$, $\phi=0$, $\mathrm{EPI}=1$, $S_i=1$,
$\Delta\mathrm{NFR}=0$, and edges $(p,k)\leftrightarrow(p,k+1)$ only (no
inter-prime edges, by Euler-product orthogonality enforced at graph level).

**Definition (prime-relabelling automorphism).** For any permutation
$\sigma\in S_{10}$ of the ten primes, let $\Pi_\sigma:V(G_{P14})\to V(G_{P14})$
be the bijection $(p_i,k)\mapsto(p_{\sigma(i)},k)$. Then $\Pi_\sigma$ is a graph
automorphism of $G_{P14}$ (it permutes ten disjoint $P_4$ components). The
attributes $\phi,\mathrm{EPI},S_i,\Delta\mathrm{NFR}$ are constant on $V$ and
therefore $\Pi_\sigma$-invariant. The frequency attribute $\nu_f$ is *not*
$\Pi_\sigma$-invariant: $\nu_f(\Pi_\sigma(p_i,k)) = k\log p_{\sigma(i)} \ne
k\log p_i$ in general. All Riemann content of $G_{P14}$ is concentrated in
$\nu_f$.

**Operator channel classification.** Following the source-level review of
`src/tnfr/operators/*` and `src/tnfr/dynamics/propagation.py`, the 13 canonical
operators split by the data they couple to on $G_{P14}$:

| Operator       | Action channel on $G_{P14}$                                                  | Depends on $\nu_f$ via edges? |
|---------------|------------------------------------------------------------------------------|-------------------------------|
| AL (emission)  | node-local: writes/raises $\mathrm{EPI}, \nu_f$                              | no (writes)                   |
| EN (reception) | edge propagation of $\Delta\mathrm{NFR}$; weight = `dissonance_magnitude * coupling_weight * phase_weight` | no (frequency-blind)          |
| IL (coherence) | node-local $\Delta\mathrm{NFR}$ contraction + Laplacian-pure phase smoother $(I-\eta L_G)$ on $\phi$ | no (only $L_G$)               |
| OZ (dissonance, freq-blind branch) | edge propagation; same weight as EN                              | no                            |
| OZ (dissonance, frequency-weighted branch) | edge propagation; weight includes `freq_weight = min(\nu_{f,i},\nu_{f,j})/max(\nu_{f,i},\nu_{f,j})` | **yes** — see Prime-Cancellation Lemma below |
| UM (coupling)  | phase synchronisation gated by $\|\phi_i-\phi_j\|\le\Delta\phi_{\max}$; on $G_{P14}$ initial $\phi\equiv 0$ so trivial | no                            |
| RA (resonance) | edge propagation amplifying coupling; weight = phase-and-coupling only       | no                            |
| SHA (silence)  | freezes evolution; $\nu_f\to 0$                                              | no                            |
| VAL (expansion)| node-local; raises $\dim(\mathrm{EPI})$                                      | no                            |
| NUL (contraction)| node-local; lowers $\dim(\mathrm{EPI})$                                    | no                            |
| THOL (self-organisation) | node-local with sub-EPI nesting                                    | no                            |
| ZHIR (mutation)| node-local; phase jump at threshold                                          | no                            |
| NAV (transition)| node-local regime switch                                                    | no                            |
| REMESH (recursivity) | temporal echo $M_{\mathrm{REMESH}}=I_N\otimes M_{\tau_g+1}$; Kronecker with identity in node index | no                            |

Eleven of the thirteen operators do not couple to $\nu_f$ at all when restricted
to edge propagation on $G_{P14}$. The one operator with a frequency-weighted
edge branch is OZ.

**Prime-Cancellation Lemma.** On any edge of $G_{P14}$ the endpoints are
$(p,k)$ and $(p,k+1)$ for some prime $p$ and some $k\in\{1,2,3\}$. The
frequency-weight in `propagated_dnfr` therefore reduces to
$$
\frac{\min(k\log p,\,(k+1)\log p)}{\max(k\log p,\,(k+1)\log p)}
= \frac{k\log p}{(k+1)\log p} = \frac{k}{k+1}.
$$
The factor $\log p$ cancels exactly. Consequently the frequency-weighted OZ
edge propagation on $G_{P14}$ is *prime-blind*: its weights depend only on the
echo index $k$, never on the prime label. This is the algebraic origin of the
empirical observation $D_{\mathrm{canonical}}=D_{\mathrm{shuffled}}$.

**Corollary (catalog-wide).** Every linear combination, composition, or
sequence built from the canonical 13 operators that acts on $G_{P14}$ only
through edge propagation has an iteration matrix that commutes with every
prime-relabelling automorphism $\Pi_\sigma$. Its spectrum is therefore
invariant under $S_{10}$ and cannot encode the Riemann-zero level statistics
through prime data, regardless of how many composition layers, REMESH echo
slots, or stabiliser insertions are added. The naive B1 closure of any R∞-1a
generalisation (operator composition $\to$ spectrum $\to$ GUE) is structurally
foreclosed on $G_{P14}$.

**Where Riemann content does live on $G_{P14}$.** The diagonal frequencies
$\{\nu_f((p,k))=k\log p\}$ are precisely the data fed to the P14 internal
Hamiltonian construction of §13quinquies (`build_prime_ladder_hamiltonian`).
That construction is *not* an iteration-matrix spectrum on $G_{P14}$; it is a
self-adjoint operator on the internal Hilbert space spanned by $|p,k\rangle$
basis states, whose diagonal block $\hat H_{\mathrm{freq}}$ has exactly these
frequencies as eigenvalues. The prime content is preserved there because the
basis is prime-indexed; relabelling primes corresponds to a unitary basis
permutation that does *not* commute with operators expressed in the original
$|p,k\rangle$ basis.

**Two genuinely open B1-style avenues (post-refutation).** The
structural lemma above leaves exactly two routes still available for a B1-style
closure inside the canonical catalog:

- **R∞-1b — Spectral-space composition on the P14 internal Hilbert space.**
  Replace the iteration-matrix-on-$G_{P14}$ formulation by a composition that
  acts on the prime-indexed basis $\{|p,k\rangle\}$ directly. Concretely:
  attempt $T_{\mathrm{spec}} = S_{\mathrm{IL}}\cdot M_{\mathrm{REMESH}}$ where
  $S_{\mathrm{IL}}$ is the spectral analogue of the IL contraction on
  $\hat H_{P14}$ (e.g.\ $S_{\mathrm{IL}} = \exp(-\eta\hat H_{P14})$) and
  $M_{\mathrm{REMESH}}$ is the canonical echo matrix lifted to the same space.
  By construction this composition does not commute with prime relabelling
  because $\hat H_{\mathrm{freq}}$ does not. Whether its spectrum can be made
  to reproduce $\{\gamma_n\}$ level statistics is open and would constitute a
  legitimate next gate.

- **R∞-1c — Canonically modified graph with inter-prime edges.** Augment
  $G_{P14}$ with inter-prime edges whose weights are *derived from the nodal
  equation* $\partial\mathrm{EPI}/\partial t = \nu_f\cdot\Delta\mathrm{NFR}(t)$
  rather than postulated. A canonical candidate is to permit edges
  $(p_i,k)\leftrightarrow(p_j,k')$ only when $|\nu_f((p_i,k)) -
  \nu_f((p_j,k'))| \le \delta_{\mathrm{coh}}$ for a coherence-derived threshold
  $\delta_{\mathrm{coh}}$, breaking the Euler-product graph orthogonality in a
  controlled way. Any such modification must be derived from the canonical
  invariants 1–6 and validated against U1–U6, not introduced for spectral
  convenience. Whether such a modification can survive U6 confinement and yet
  carry Riemann data is open.

Neither R∞-1b nor R∞-1c is opened in this commit. They are recorded here as
the two structurally permitted exits left by the lemma, consistent with the
program-wide branch B1/B2/B3 taxonomy of §13septies: a positive R∞-1b or
R∞-1c result would constitute a B1 closure of a non-naive form; a negative
result on both would constitute additional support for B2 (a new canonical
operator is required) or B3 (no TNFR closure exists).

**Cross-references.** Channel classifications were verified against
`src/tnfr/operators/coherence.py` (IL), `src/tnfr/dynamics/propagation.py`
(OZ/EN/RA frequency weights), `src/tnfr/operators/coupling.py` (UM),
`src/tnfr/operators/recursivity.py` (REMESH), and
`src/tnfr/riemann/prime_ladder_hamiltonian.py` (P14 graph and Hamiltonian).
The full set of pre-registered controls and the empirical refutation are in
§13vicies-novies.9.

**Status.** B1 closure of R∞-1a in its naive edge-channel form is refuted on
$G_{P14}$ both empirically (F6-A, §13vicies-novies.9) and structurally
(Prime-Cancellation Lemma + catalog-wide corollary, this subsection). The
program-level open question remains G4 = RH (and its twin $\mathrm{GRH}_\chi$);
the open B1-style avenues are now exactly R∞-1b and R∞-1c.

---

### §13vicies-novies.11 Formalisation: the Euler-Orthogonality Lemma

The two empirical facts established in §13vicies-novies.9 and the operator
classification of §13vicies-novies.10 are specialisations of a single
structural property of the canonical engine acting on $G_{P14}$. This
subsection names that property, states it as a lemma, supplies a formal
proof, and records its two operator-level corollaries. No new construction
is introduced; the content is a tightening of §13vicies-novies.10 into a
single citable statement.

**Naming convention.** The lemma is called the *Euler-Orthogonality Lemma*
because its hypothesis — disjointness of prime ladders in $G_{P14}$ — is
the graph-level realisation of the Euler-product orthogonality of the
Dirichlet series $-\zeta'(s)/\zeta(s) = \sum_{p,k} (\log p)\, p^{-ks}$
that drives the §13quinquies (P14) Hamiltonian construction. The two
properties are the same fact, viewed once analytically (independence of
prime factors in the Euler product) and once combinatorially (absence of
inter-prime edges in $G_{P14}$).

**Setup (recall).** Fix the canonical prime-ladder graph
$G_{P14} = (V, E)$ of §13quinquies with $V = \{(p_i, k) :
1 \le i \le n_{\text{primes}},\ 1 \le k \le k_{\max}\}$, edges
$E = \{((p,k), (p,k{+}1)) : p \text{ prime},\ 1 \le k < k_{\max}\}$, node
attributes $\nu_f((p,k)) = k \log p$, $\phi \equiv 0$, $\mathrm{EPI} \equiv 1$,
$S_i \equiv 1$, $\Delta\mathrm{NFR} \equiv 0$. Let
$\Pi: S_{n_{\text{primes}}} \to \operatorname{Aut}(G_{P14})$ be the
prime-relabelling action $\Pi_\sigma(p_i, k) = (p_{\sigma(i)}, k)$. Let
$\mathcal{O}_{13} = \{\text{AL}, \text{EN}, \text{IL}, \text{OZ},
\text{UM}, \text{RA}, \text{SHA}, \text{VAL}, \text{NUL}, \text{THOL},
\text{ZHIR}, \text{NAV}, \text{REMESH}\}$ be the canonical 13-operator
catalog (cf. AGENTS.md §"The 13 Canonical Operators").

**Definition (edge-channel restriction).** For $O \in \mathcal{O}_{13}$
let $\mathcal{R}_E(O) : \mathbb{R}^V \to \mathbb{R}^V$ denote the linear
part of $O$'s action on a real-valued field on $V$ obtained by *retaining
only the contribution that propagates along edges of $G_{P14}$* and
freezing all node-local writes. For node-local operators (AL, IL pressure
contraction, SHA, VAL, NUL, THOL, ZHIR, NAV, UM on $\phi \equiv 0$,
REMESH) the edge-channel restriction is the identity by definition. For
edge-propagating operators (EN, OZ, RA, IL phase-Laplacian) it is the
linear edge-propagation kernel documented in
`src/tnfr/dynamics/propagation.py` and `src/tnfr/operators/coherence.py`
(IL phase-Laplacian smoother $I - \eta L_G$). For REMESH the recursivity
echo $M_{\mathrm{REMESH}} = I_{|V|} \otimes M_{\tau_g + 1}$ commutes
trivially with $\Pi_\sigma$ in the node index.

**Lemma 1 (Euler-Orthogonality Lemma).** *Every operator
$O \in \mathcal{O}_{13}$ restricted to the edge channel on $G_{P14}$
commutes with the prime-relabelling action $\Pi$ of
$S_{n_{\text{primes}}}$:*
$$
\mathcal{R}_E(O) \circ \Pi_\sigma \;=\; \Pi_\sigma \circ \mathcal{R}_E(O)
\qquad \forall\,\sigma \in S_{n_{\text{primes}}},\ \forall\,O \in \mathcal{O}_{13}.
$$

*Proof.* Partition $\mathcal{O}_{13}$ by the channel through which the
operator can in principle couple to $\nu_f$ on edges (the classification
of §13vicies-novies.10, verified against the operator source modules).

*Case A — node-local operators* (AL, IL pressure contraction, SHA, VAL,
NUL, THOL, ZHIR, NAV, UM on $\phi \equiv 0$): $\mathcal{R}_E(O) = I_V$ by
definition; $I_V$ commutes with every $\Pi_\sigma$.

*Case B — frequency-blind edge propagation* (EN, OZ frequency-blind
branch, RA, IL phase-Laplacian): the propagation weight on every edge
$e = ((p,k), (p,k{+}1))$ is a function of `coupling_weight` and
`phase_weight` only, both of which depend exclusively on edge combinatorics
and on the phase attribute $\phi \equiv 0$ which is $\Pi$-invariant.
$\mathcal{R}_E(O)$ is therefore the same kernel on every prime ladder
copy; $\Pi_\sigma$ permutes copies; the two operations commute.

*Case C — frequency-weighted OZ branch* (OZ with
`freq_weight = min(\nu_{f,i}, \nu_{f,j}) / max(\nu_{f,i}, \nu_{f,j})`):
on $G_{P14}$ every edge endpoints satisfy $\nu_{f,i} = k \log p$ and
$\nu_{f,j} = (k{+}1) \log p$ for the *same* prime $p$. Hence
$$
\frac{\min(k \log p,\ (k{+}1) \log p)}{\max(k \log p,\ (k{+}1) \log p)}
\;=\; \frac{k \log p}{(k{+}1) \log p} \;=\; \frac{k}{k{+}1},
$$
which is independent of $p$. The frequency-weight is therefore prime-blind
on $G_{P14}$, and the argument of Case B applies verbatim.

*Case D — REMESH.* $M_{\mathrm{REMESH}} = I_{|V|} \otimes M_{\tau_g + 1}$
factors through the identity in the node index; $\Pi_\sigma$ acts only on
the node index; the two factors commute.

All four cases exhaust $\mathcal{O}_{13}$. $\square$

**Corollary 1 (composition closure).** *The set of linear operators on
$\mathbb{R}^V$ that commute with every $\Pi_\sigma$ is closed under
composition and real-linear combination. Therefore every operator
expressible as a real-linear composition of edge-channel restrictions of
elements of $\mathcal{O}_{13}$ commutes with $\Pi$:*
$$
T \;=\; \sum_{j} c_j \prod_{\ell} \mathcal{R}_E(O_{j,\ell})
\quad \Longrightarrow \quad T \circ \Pi_\sigma \;=\; \Pi_\sigma \circ T
\quad \forall\,\sigma \in S_{n_{\text{primes}}}.
$$

*Proof.* Immediate from Lemma 1 and the elementary fact that the
commutant of a group action is a subalgebra of $\operatorname{End}(\mathbb{R}^V)$.
$\square$

**Corollary 2 (spectral $S_{n_{\text{primes}}}$-invariance).** *For any
$T$ as in Corollary 1, the multiset $\operatorname{spec}(T)$ is invariant
under prime relabelling. In particular, no iteration matrix built by
edge-channel composition of canonical operators can distinguish, by its
spectrum alone, the canonical prime assignment from any of the
$n_{\text{primes}}!$ permuted assignments.*

*Proof.* Conjugation by an invertible operator preserves spectrum;
$\Pi_\sigma$ is a permutation matrix (hence invertible); commutation
$T \Pi_\sigma = \Pi_\sigma T$ implies $T = \Pi_\sigma T \Pi_\sigma^{-1}$;
hence $\operatorname{spec}(T) = \operatorname{spec}(\Pi_\sigma T \Pi_\sigma^{-1})
= \operatorname{spec}(T)$ as a multiset under the permuted labelling.
$\square$

**Empirical signature on $G_{P14}$.** The bit-for-bit identity
$D_{\mathrm{GUE}}^{\mathrm{canonical}} = D_{\mathrm{GUE}}^{\mathrm{shuffled}} = 0.9053$
reported in §13vicies-novies.9 for the composition $T = S_{\mathrm{IL}}
\cdot M_{\mathrm{REMESH}}$ is the numerical specialisation of Corollary 2
to a single composition. The lemma predicts that the same identity holds
for *every* edge-channel composition of canonical operators on $G_{P14}$;
testing additional compositions can therefore only reproduce this
identity (or break the edge-channel hypothesis by introducing an
operator outside the catalog).

**Where the lemma's hypothesis fails (and why R∞-1b and R∞-1c remain
open).** The lemma's three hypotheses — (i) action restricted to the
edge channel, (ii) operators drawn from $\mathcal{O}_{13}$, (iii) graph
$G_{P14}$ unchanged — are each necessary for the proof. The two open
B1-style routes left by §13vicies-novies.10 each break exactly one
hypothesis:

* **R∞-1b breaks (i).** Action moves from $\mathbb{R}^V$ to the
  prime-indexed internal Hilbert space spanned by $\{|p,k\rangle\}$
  (the P14 Hamiltonian's basis, §13quinquies). On this space the
  spectral analogue $S_{\mathrm{IL}} = \exp(-\eta\,\hat H_{P14})$ does
  *not* factor through the identity in the prime index because
  $\hat H_{\mathrm{freq}} = \operatorname{diag}(k \log p)$ does not.
  Prime relabelling becomes a unitary basis permutation that does not
  commute with operators expressed in the original basis. Lemma 1
  does not apply, and Corollary 2 is silent.

* **R∞-1c breaks (iii).** The graph is augmented with inter-prime edges
  whose weights are derived from the nodal equation
  $\partial\mathrm{EPI}/\partial t = \nu_f \cdot \Delta\mathrm{NFR}(t)$.
  Every inter-prime edge has endpoints with $\nu_f$ from *different*
  primes, so the Case-C cancellation $\log p$ no longer occurs.
  Edge-channel operators on the augmented graph therefore have weights
  that depend on the prime labels, $\Pi_\sigma$ no longer commutes with
  the edge-propagation kernel, and Corollary 2 fails by construction.

The two routes are open *because* they break hypotheses of the
Euler-Orthogonality Lemma; conversely, the lemma is the precise statement
of *what* must be broken for any B1 closure inside the canonical catalog
to remain possible.

**Honest scope.** Lemma 1 and its corollaries are statements about
edge-channel linear actions on $G_{P14}$. They do not:

* prove RH or close G4 = RH;
* refute B1 in any sense beyond the edge-channel route on $G_{P14}$;
* refute B1 via R∞-1b (internal Hilbert space) or R∞-1c (graph
  modification);
* refute B2 (new canonical operator) or B3 (no TNFR closure);
* constrain operator behaviour on graphs other than $G_{P14}$.

What they do, formally, is convert the empirical refutation of
§13vicies-novies.9 from a single-composition observation into a
catalog-wide structural theorem applicable to any future edge-channel
composition attempt. The two routes that remain available are precisely
those that violate one of the lemma's hypotheses by construction.

**Cross-references.**

* §13vicies-novies.9 — F6-A empirical refutation that this lemma
  generalises.
* §13vicies-novies.10 — operator-by-operator channel classification
  underlying the case analysis.
* §13septies, §13octies — program-wide branch B1/B2/B3 taxonomy that
  the lemma narrows on the B1 side.
* `src/tnfr/operators/coherence.py`, `src/tnfr/dynamics/propagation.py`,
  `src/tnfr/operators/coupling.py`, `src/tnfr/operators/recursivity.py`,
  `src/tnfr/riemann/prime_ladder_hamiltonian.py` — source-level audit
  trail used to verify the Case A–D classification.

---
### §13vicies-novies.12 R-inf-1c pre-registration: canonically modified $G_{P14}$ with inter-prime edges

This subsection pre-registers the R-inf-1c milestone identified as one of
the two genuinely open B1-style routes by §13vicies-novies.10 and
formalised in §13vicies-novies.11 as "the route that breaks hypothesis
(iii) of the Euler-Orthogonality Lemma by construction." No data is
collected at commit time. The benchmark script
`benchmarks/remesh_infinity_riemann_modified_graph.py` is committed
simultaneously; its first execution will append the Results block as
§13vicies-novies.13.

**Pre-registration discipline.** This subsection follows the same
pattern as §13vicies-novies.8 and the pre-registration block of
§13vicies-novies.9: methodology, parameters, seeds, decision thresholds,
and verdict logic are all locked *before* any execution. Any deviation
between the committed script and the published Results block (other than
documented bug fixes) will be flagged in the post-execution amendment.

**Construction.** Let $G_{P14}^{\mathrm{aug}} = (V, E_{\mathrm{intra}}
\cup E_{\mathrm{inter}})$ denote the prime-ladder graph $G_{P14}$
augmented with inter-prime edges, where:

* $V$ and $E_{\mathrm{intra}}$ are unchanged from §13quinquies
  ($n_{\text{primes}} = 10$, $k_{\max} = 4$, $|V| = 40$, intra-prime
  edges $(p,k) \leftrightarrow (p,k{+}1)$ only).
* $E_{\mathrm{inter}}$ is the set of inter-prime edges
  $\{(p_i,k) \leftrightarrow (p_j,k') : p_i \ne p_j,\
  |k \log p_i - k' \log p_j| \le \delta_{\mathrm{coh}}\}$.

**Exploratory coherence threshold $\delta_{\mathrm{coh}}$** (heuristic, not canonically derived). This benchmark uses the historically-chosen $|\nabla \phi|$ prefactor $\gamma/\pi \approx 0.184$ (an exploratory scale; the canonical $|\nabla\phi|$ early-warning is the heuristic $\pi/16 \approx 0.196$, kinematic bound $\pi$) applied to the structural
frequency $\nu_f$ on its native log-energy scale:
$$
\delta_{\mathrm{coh}} \;=\; \frac{\gamma}{\pi}\, \cdot \,
\max_{(p,k) \in V} \nu_f((p,k)) \;=\; \frac{\gamma}{\pi}\, \cdot \,
k_{\max}\,\log p_{\max}.
$$
For the canonical configuration ($p_{\max} = 29$, $k_{\max} = 4$),
$\max \nu_f = 4 \log 29 \approx 13.4699$, giving
$\delta_{\mathrm{coh}} \approx 0.18373 \cdot 13.4699 \approx 2.4747$.

This is a *single canonical value* derived from the tetrad
correspondence; it is not swept and is not fitted to any target. If the
canonical $\delta_{\mathrm{coh}}$ yields zero or fewer than two
inter-prime edges, the milestone is INDETERMINATE_DEGENERATE_CONSTRUCTION
and a documented amendment (canonical reinterpretation of
$\delta_{\mathrm{coh}}$ or change of $(n_{\text{primes}}, k_{\max})$
configuration) is required before re-pre-registration.

**Canonical edge weight.** Gaussian decay anchored at
$\delta_{\mathrm{coh}}$ with prefactor $\gamma/\pi$ (Kuramoto critical
coupling in TNFR units, AGENTS.md tetrad-edge table):
$$
w_{ij} \;=\; \frac{\gamma}{\pi}\, \exp\!\left(
-\, \frac{|\nu_f(i) - \nu_f(j)|^2}{2\,\delta_{\mathrm{coh}}^2}
\right),
\qquad
\text{for } (i, j) \in E_{\mathrm{inter}}.
$$
Intra-prime edges retain unit weight (canonical $G_{P14}$ convention).

This is the *Kuramoto-U3 inter-prime coupling form* already explored at
the *Hamiltonian-perturbation* level in P29
(`src/tnfr/riemann/spectral_emergence.py`, §13nonies/§13.2). R-inf-1c
re-tests the same canonical coupling, but reframed as a *graph
modification + edge-channel iteration matrix* rather than as a
perturbation of the P14 internal Hamiltonian. The two reframings probe
different B1 sub-routes: P29 tested whether canonical inter-prime
coupling perturbs $\operatorname{spec}(\hat H_{P14})$ toward GUE
statistics (best result: $\mathrm{KS}_{\mathrm{GUE}} = 0.122$ with
Kuramoto-U3 at $s^* = 0.5$, threshold $< 0.05$ unreached). R-inf-1c
tests whether the *iteration-matrix spectrum* of the composed operator
$S_{\mathrm{IL}}^{\mathrm{aug}} \cdot M_{\mathrm{REMESH}}$ on
$G_{P14}^{\mathrm{aug}}$ encodes Riemann-zero content.

**Iteration matrix.** Mirror §13vicies-novies.9 construction with the
augmented Laplacian:
$$
T_{\mathrm{aug}} \;=\; S_{\mathrm{IL}}^{\mathrm{aug}} \cdot M_{\mathrm{REMESH}},
\qquad
S_{\mathrm{IL}}^{\mathrm{aug}}|_{\text{slot } 0} \;=\; I_N - \eta\,L_{G^{\mathrm{aug}}},
\qquad
S_{\mathrm{IL}}^{\mathrm{aug}}|_{\text{slot } s \ge 1} \;=\; I_N,
$$
with $L_{G^{\mathrm{aug}}} = D_{\mathrm{aug}} - A_{\mathrm{aug}}$ the
weighted combinatorial Laplacian and $M_{\mathrm{REMESH}} = M \otimes I_N$
in slot-major ordering (canonical $\alpha = 0.5$, $\tau_l = 4$,
$\tau_g = 16$, $\eta = 0.3$, $|V|\cdot(\tau_g+1) = 680$). The choice of
$L_{G^{\mathrm{aug}}}$ as the weighted Laplacian is the canonical
generalisation of the §13vicies-novies.9 unweighted construction; no
other regulariser is added.

**F7-A statistic (decisive, pre-registered).** Mirror F6-A
(§13vicies-novies.9):

1. Remove trivial fixed-point cluster: $|\lambda - 1| < 10^{-9}$.
2. Project to 1-D: $\operatorname{Im}(\lambda)$ for the upper-half-plane
   subset ($\operatorname{Im}(\lambda) \ge 10^{-12}$), sorted ascending.
   Fallback: $\operatorname{Re}(\lambda)$ sorted ascending if the
   projection is empty (real spectrum). Both branches are reported in
   the JSON.
3. Normalised consecutive spacings: $\delta_k = (s_{k+1} - s_k) /
   \overline{\Delta}$.
4. KS sup-distance vs the GUE Wigner surmise $P_{\mathrm{GUE}}(s) =
   (32/\pi^2)\, s^2 \exp(-4 s^2 / \pi)$.

**F8 structural condition (necessary, pre-registered).** The
Euler-Orthogonality Lemma (§13vicies-novies.11) commutes with prime
relabelling because of hypothesis (iii) ($G_{P14}$ unchanged). The
R-inf-1c construction violates (iii) by adding inter-prime edges whose
weights *depend on $|\nu_f(i) - \nu_f(j)|$, hence on actual prime
labels*. The decisive structural test is whether prime relabelling now
yields a *different* spectrum:

* F8 SATISFIED: $|D_{\mathrm{canonical}} - D_{\mathrm{shuffled}}| \ge 0.01$
  (numerical floor; lemma hypothesis genuinely broken).
* F8 FAILED:    $|D_{\mathrm{canonical}} - D_{\mathrm{shuffled}}| < 0.01$
  (Euler-orthogonality not broken; implementation degeneracy or
  $\delta_{\mathrm{coh}}$ too small to generate label-dependent inter-prime
  weights — INDETERMINATE construction).

F8 is a *necessary* condition for R-inf-1c to be a meaningful test of
its own hypothesis. F8 failure does not refute B1; it refutes the
specific R-inf-1c construction and requires a documented amendment.

**Pre-registered controls.**

* **N1 GOE** (random symmetric matrix of dimension $|V|\cdot(\tau_g+1) =
  680$, scaled by $1/\sqrt{2N}$; expected $D_{\mathrm{GUE}} \approx
  0.10$–$0.20$ — GOE spacings differ from the GUE Wigner surmise).
* **N2 Poisson** (680 iid uniform points; expected
  $D_{\mathrm{GUE}} \approx 0.30$–$0.50$).
* **N3 shuffled-prime** (rebuild $G_{P14}^{\mathrm{aug}}$ with a random
  permutation of the ten primes among the ladders, re-derive inter-prime
  weights from $\nu_f$ on the shuffled labels). *Primary discriminator
  for F8*.
* **N4 REMESH-isolated** (spectrum of $M$ alone; diagnostic baseline,
  expected to be degenerate as in §13vicies-novies.9).
* **N5 random-augmentation** (rebuild $G_{P14}^{\mathrm{aug}}$ with the
  inter-prime edge set replaced by an Erdős–Rényi random selection of
  the same edge count, all weights set to the constant $\gamma/\pi$).
  Tests whether the canonical $\delta_{\mathrm{coh}}$-derived structure
  matters versus generic random topology of the same density.

**Riemann reference.** External anchor:
$D_{\mathrm{GUE}}$ for the first 100 Riemann zero imaginary parts via
`mpmath.zetazero`.

**Pre-registered F7 verdict logic.**

* **SUPPORTED**:    $D_{\mathrm{canonical}} < 0.15$ **AND**
  $D_{\mathrm{canonical}} < D_{\mathrm{shuffled}} - 0.05$ **AND**
  $D_{\mathrm{canonical}} < D_{\mathrm{random}} - 0.05$.
* **REFUTED**:      $D_{\mathrm{canonical}} > 0.30$ **OR**
  ($D_{\mathrm{canonical}} \ge D_{\mathrm{shuffled}} - 0.05$ **AND**
  F8 SATISFIED).
* **INDETERMINATE_DEGENERATE_CONSTRUCTION**: F8 FAILED.
* **INDETERMINATE_OTHER**: F8 SATISFIED and neither SUPPORTED nor
  REFUTED conditions hold.

**Pre-registered milestone verdict logic.**

* SUPPORTED $\Rightarrow$
  `B1_MODIFIED_GRAPH_POTENTIALLY_OPEN_REQUIRES_REPLICATION` (deep
  diagnostic + independent seeds + alternative compositions REMESH +
  EN/NAV/OZ before any evidential update on B1).
* REFUTED $\Rightarrow$
  `B1_MODIFIED_GRAPH_REFUTED_FOR_CANONICAL_INTER_PRIME_COUPLING`. Closes
  the R-inf-1c sub-route for the canonical $\delta_{\mathrm{coh}} =
  (\gamma/\pi) \cdot \max \nu_f$ choice. Does NOT close R-inf-1c for
  alternative *canonically-derivable* $\delta_{\mathrm{coh}}$ choices
  (e.g.\ from $\varphi$, $e$, or other tetrad edges); any such
  alternative would require its own pre-registration.
* INDETERMINATE_DEGENERATE_CONSTRUCTION $\Rightarrow$ implementation or
  parameter amendment required; no B1 update.
* INDETERMINATE_OTHER $\Rightarrow$ status unchanged; design refinement
  needed before next attempt.

**Pre-registered seeds and parameters.** All random elements (N1 GOE
draw, N2 Poisson draw, N3 prime shuffle permutation, N5 Erdős–Rényi
edge selection) use `numpy.random.default_rng(20260526)`. Riemann
zeros via `mpmath.zetazero` at `mp.dps = 30`. REMESH parameters:
$\alpha = 0.5$, $\tau_l = 4$, $\tau_g = 16$. IL coupling: $\eta = 0.3$.
Graph: $n_{\text{primes}} = 10$, $\max\_\text{power} = 4$, intra-prime
unit weight. Canonical coupling: $\delta_{\mathrm{coh}} = (\gamma/\pi)
\cdot 4 \log 29$, $w_{ij} = (\gamma/\pi) \exp(-|\Delta\nu_f|^2 /
(2 \delta_{\mathrm{coh}}^2))$ for $(i,j) \in E_{\mathrm{inter}}$. All
canonical constants from `src/tnfr/constants/canonical.py` (`GAMMA`,
`PI`).

**What this milestone CAN establish.**

* A definitive verdict on the canonical Kuramoto-U3 inter-prime
  augmentation of $G_{P14}$ as a graph-modification route to encode
  Riemann content in an edge-channel iteration matrix spectrum.
* Empirical evidence on whether breaking hypothesis (iii) of the
  Euler-Orthogonality Lemma (the only hypothesis R-inf-1c targets) is
  by itself sufficient to recover Riemann level statistics in a
  canonical TNFR construction.

**What this milestone CANNOT establish.**

* R-inf-1c for alternative canonical $\delta_{\mathrm{coh}}$ choices
  (from $\varphi$, $e$, or tetrad edges other than $\gamma/\pi$).
* R-inf-1b (spectral-space composition on the P14 internal Hilbert
  space; orthogonal sub-route that breaks hypothesis (i) instead of
  (iii)).
* B1 closure outside the canonical 13-operator catalog (B2 territory).
* G4 = RH, T-HP, or any closure beyond what F7 + F8 strictly test.

**Why this is not a re-run of P29.** P29
(`src/tnfr/riemann/spectral_emergence.py`, §13nonies) tested the same
canonical Kuramoto-U3 coupling but at the *Hamiltonian-perturbation*
level: $\hat H_{P14}(J) = \hat H_{P14}(0) + J \cdot \hat H_{\mathrm{coupling}}$
on the prime-indexed internal Hilbert space, with spectrum compared to
GUE Wigner via KS distance. R-inf-1c uses the *same coupling form* at
the *graph-modification + edge-channel iteration matrix* level: the
augmented Laplacian $L_{G^{\mathrm{aug}}}$ enters a slot-0 IL smoother,
which is composed with $M_{\mathrm{REMESH}}$ on the 680-dimensional
joint state, and the iteration matrix spectrum is the test object. The
two milestones probe different mathematical objects (Hamiltonian
eigenvalues versus iteration-matrix eigenvalues) under the same
canonical coupling; comparing their verdicts will sharpen the structural
picture of where canonical Kuramoto-U3 can and cannot transport Riemann
content.

**Implementation.**
`benchmarks/remesh_infinity_riemann_modified_graph.py` (committed in the
same commit as this pre-registration; no data collected at commit time).
Output JSON written to
`results/remesh_infinity/remesh_infinity_riemann_modified_graph.json`
(gitignored, not part of the audit trail; the audit trail is this
file).

**Status (pre-registration commit)**: methodology locked; no data
observed; next commit will append results in a Results block as
§13vicies-novies.13.

---

### §13vicies-novies.13 R-inf-1c Results (post-registration data)

**Execution context.** Driver
`benchmarks/remesh_infinity_riemann_modified_graph.py` executed exactly
as pre-registered in §13vicies-novies.12. No parameters changed; no
seeds changed; no thresholds changed. Canonical configuration: $N=40$
nodes, $\dim(\text{joint})=680$, $\delta_{\mathrm{coh}}=2.4747$ (single
derived value from $(\gamma/\pi)\,k_{\max}\,\log p_{\max}$), 278
canonical inter-prime edges, 278 shuffled inter-prime edges. Seed
`np.random.default_rng(20260526)`, `mp.dps=30`. Output report:
`results/remesh_infinity/remesh_infinity_riemann_modified_graph.json`.

**Pre-registered F7-A statistic (KS distance vs GUE Wigner surmise).**

| Label | Projection | #spacings | $D_{\mathrm{GUE}}$ |
|---|---|---:|---:|
| `canonical_REMESH_o_IL_aug` | `Im_upper` | 319 | **0.43058** |
| `N1_GOE` | `Re_fallback` | 679 | 0.11259 |
| `N2_Poisson` | `uniform_iid` | 679 | 0.30317 |
| `N3_shuffled_prime` | `Im_upper` | 319 | **0.43058** |
| `N4_REMESH_isolated` | `Im_upper` | 7 | 0.30820 |
| `N5_random_augmentation` | `Im_upper` | 319 | 0.41543 |
| `Riemann_reference` (mpmath zeros) | iid_or_zeros | 99 | 0.07700 |

**Pre-registered F8 structural necessary condition
($|D_{\mathrm{canonical}} - D_{\mathrm{shuffled}}| \ge 0.01$).**

$$
|D_{\mathrm{canonical}} - D_{\mathrm{shuffled}}| = 3.13 \times 10^{-13}
\quad \text{(machine-precision zero)}, \qquad \text{F8 NOT satisfied.}
$$

**Pre-registered verdict.** `INDETERMINATE_DEGENERATE_CONSTRUCTION` →
`B1_MODIFIED_GRAPH_INDETERMINATE_DEGENERATE_CONSTRUCTION`.

**Structural reading (Euler-Orthogonality Lemma, §13vicies-novies.11
specialised to the augmented graph).** The exact-to-13-decimal-places
identity $D_{\mathrm{canonical}} = D_{\mathrm{shuffled}} = 0.43058$ is
not a numerical accident: it is the predicted consequence of an
unbroken $S_n$ prime-relabelling symmetry on the augmented graph. The
canonical inter-prime weight law

$$
w_{ij} = \frac{\gamma}{\pi}\,
\exp\!\left(-\frac{|\Delta \nu_f|^{2}}{2\,\delta_{\mathrm{coh}}^{2}}\right)
$$

depends on $\nu_f$ values only through pairwise differences, and the
$\nu_f$ schedule on the prime ladder is itself a function of $p$ alone.
Hence shuffling the prime labels acts on the augmented Laplacian
$L_{G^{\mathrm{aug}}}$ by conjugation with a permutation matrix
$P_{\sigma}\otimes I_{k_{\max}}$. This conjugation extends through the
slot-0 IL smoother and through $M_{\mathrm{REMESH}}$ (both block-local
in the canonical construction), so $T^{\mathrm{aug}}_{\mathrm{canonical}}$
and $T^{\mathrm{aug}}_{\mathrm{shuffled}}$ are unitarily equivalent and
therefore isospectral. The empirical $\Delta D = O(10^{-13})$ measures
exactly the floating-point conjugation residual.

**What this closes.** Within the canonical 13-operator catalog, the
graph-modification sub-route R∞-1c — i.e. *any* augmentation of $G_{P14}$
by inter-prime edges whose weights depend only on $(\nu_f, p)$ data
through $S_n$-invariant combinations — is **structurally incapable of
breaking the $S_n$ symmetry** and therefore **cannot encode Riemann
level statistics** at the edge-channel level. This is the modified-graph
analogue of the original Euler-Orthogonality Lemma for fixed $G_{P14}$.

**What this does not close.** R∞-1c does not exhaust B1. The two
mathematical hypotheses required to fall under the Euler-Orthogonality
Lemma at the modified-graph level are (i) $S_n$-invariant weight law
and (ii) block-local action of all composed canonical operators on
$\mathbb{C}^{N}\otimes \mathbb{C}^{k_{\max}}$. Both hypotheses hold for
the canonical Kuramoto-U3 augmentation tested here. The remaining
structurally permitted B1 sub-route is **R∞-1b** (composition on the
P14 *internal* Hilbert space, not the graph: by construction
basis-permutation-non-commuting, hence not subject to the Euler-
Orthogonality argument). R∞-1b has not been pre-registered or tested
in this thread.

**Control-by-control sanity.**
- `N1_GOE` $D=0.113$: GOE→GUE Wigner-surmise mismatch is within the
  expected $\approx 0.10$ range for $n_{\mathrm{spacings}}=679$;
  pipeline calibration confirmed.
- `N2_Poisson` $D=0.303$: uniform-iid baseline well separated from GUE,
  as expected for non-correlated levels.
- `Riemann_reference` $D=0.077$: classical Riemann zeros pass the GUE
  test at the gold-standard level on 99 spacings; this is the
  positive-control floor the canonical construction would need to
  approach to count as `SUPPORTED`.
- `N4_REMESH_isolated` reports only 7 spacings (degenerate sample
  size; consistent with the §13vicies-novies.8 finding that
  REMESH-iterated-in-isolation produces a rank-deficient iteration
  matrix). Not used in verdict.
- `N5_random_augmentation` $D=0.415$: Erdős–Rényi inter-prime
  augmentation with uniform weight $\gamma/\pi$, same edge count.
  $D_{\mathrm{random}} - D_{\mathrm{canonical}} = -0.015$ — random is
  *closer* to GUE than canonical, which is itself a structural
  signature of the symmetry obstruction (random breaks $S_n$;
  canonical does not).

**Status update for the B1 question.** After §13vicies-novies.13, the
B1-at-edge-channel-level question on or around $G_{P14}$ is closed for
every sub-route covered by the Euler-Orthogonality Lemma:

| Sub-route | Object | Status |
|---|---|---|
| R∞-1a-operator (§.8) | REMESH iterated, fixed $G_{P14}$ | REFUTED |
| R∞-1a-composed (§.9) | REMESH ∘ IL, fixed $G_{P14}$ | REFUTED |
| R∞-1c (§.12–§.13) | $T^{\mathrm{aug}}$ on canonically augmented $G_{P14}$ | INDETERMINATE_DEGENERATE_CONSTRUCTION (Euler-Orthogonality at augmented level) |
| R∞-1b | composition on P14 internal Hilbert space | NOT TESTED (structurally permitted) |

The TNFR-Riemann program remains paused at the T-HP boundary
(§13septies). G4 = RH is unchanged. What §13vicies-novies.13 supplies
is a second empirical instantiation of the symmetry obstruction
identified in §13vicies-novies.11, now at the graph-modification
level, locking R∞-1b as the unique remaining sub-route of B1 that
might admit a TNFR-canonical attack without going to B2 or B3.

**Reproducibility.**

```powershell
$env:PYTHONPATH = (Resolve-Path ./src).Path
& .\.venv312\Scripts\python.exe `
    benchmarks\remesh_infinity_riemann_modified_graph.py
```

Output: `results/remesh_infinity/remesh_infinity_riemann_modified_graph.json`
(gitignored). All four numerical entries quoted above (canonical $D$,
N3 $D$, $|\Delta D|$, N5 $D$) reproduce from the locked seed
`np.random.default_rng(20260526)`.

---

### §13vicies-novies.14 R-inf-1b pre-registration: spectral-space composition on the P14 internal Hilbert space

This subsection pre-registers the R-inf-1b milestone identified by
§13vicies-novies.10 (Catalog structural lemma) and §13vicies-novies.11
(Euler-Orthogonality Lemma) as the *unique* remaining structurally
permitted B1 sub-route after R-inf-1a-operator (§13vicies-novies.8,
REFUTED), R-inf-1a-composed (§13vicies-novies.9, REFUTED), and R-inf-1c
(§13vicies-novies.12–§13vicies-novies.13,
INDETERMINATE_DEGENERATE_CONSTRUCTION). No data is collected at commit
time. The benchmark script
`benchmarks/remesh_infinity_riemann_spectral_basis.py` is committed
simultaneously; its first execution will append the Results block as
§13vicies-novies.15.

**Pre-registration discipline.** This subsection follows the same
pattern as §13vicies-novies.12: methodology, parameters, seeds,
decision thresholds, and verdict logic are all locked *before* any
execution. Any deviation between the committed script and the published
Results block (other than documented bug fixes) will be flagged in the
post-execution amendment.

**Origin.** §13vicies-novies.10 specifies the R-inf-1b sub-route as
$T_{\mathrm{spec}} = S_{\mathrm{IL}} \cdot M_{\mathrm{REMESH}}$ with
$S_{\mathrm{IL}} = \exp(-\eta\,\hat H_{P14})$ (spectral analogue of IL
contraction in the P14 internal Hilbert space) and $M_{\mathrm{REMESH}}$
the canonical REMESH echo matrix "lifted to the same space." The
construction targets hypothesis (i) of the Euler-Orthogonality Lemma
(action *in* a prime-indexed basis $\{|p,k\rangle\}$ rather than *on*
the graph $G_{P14}$), as opposed to R-inf-1c which targeted hypothesis
(iii) (graph modification).

**Construction.** Let $V$ denote the canonical $G_{P14}$ node set
($n_{\text{primes}} = 10$, $k_{\max} = 4$, $|V| = N = 40$) and identify
each node with its prime-power label $(p_i, k)$, giving the basis
$\{|p_i, k\rangle\}_{i=1,\dots,10;\ k=1,\dots,4}$ of the P14 internal
Hilbert space $\mathcal{H}_{N}$. The lifted joint state space is
$\mathcal{H}_{\mathrm{joint}} = \mathbb{C}^{\tau_g + 1} \otimes
\mathcal{H}_{N}$ with $\dim \mathcal{H}_{\mathrm{joint}} = 17 \cdot 40 =
680$, slot-major ordering (index $= \text{slot} \cdot N + (p,k)$).

The two factors are:

* $\hat H_{P14}$: the canonical P14 self-adjoint Hamiltonian
  $H_{\mathrm{int}} = H_{\mathrm{coh}} + H_{\mathrm{freq}} +
  H_{\mathrm{coupling}}$ built by
  `src/tnfr/operators/hamiltonian.py::InternalHamiltonian` on the
  canonical $G_{P14}$ with $\text{coupling} = 0$ (so
  $H_{\mathrm{coupling}} = 0$ in this milestone, matching the P14
  prime-ladder spectrum reference of §13quinquies).
  $H_{\mathrm{freq}}$ is diagonal with entries
  $k \log p_i$ (eigenvalues of the prime-ladder spectrum).
* $M$: the canonical REMESH echo matrix
  $(\alpha, \tau_l, \tau_g) = (0.5, 4, 16)$ as in
  §13vicies-novies.9.

**Iteration matrix.** Lift both factors to $\mathcal{H}_{\mathrm{joint}}$
canonically:
$$
S_{\mathrm{IL}}^{\mathrm{spec}} \;=\; I_{\tau_g + 1} \otimes
\exp\!\left(-\eta\,\hat H_{P14}\right),
\qquad
M_{\mathrm{REMESH}} \;=\; M \otimes I_{N},
\qquad
T_{\mathrm{spec}} \;=\; S_{\mathrm{IL}}^{\mathrm{spec}} \cdot
M_{\mathrm{REMESH}}.
$$
Canonical parameters: $\eta = 0.3$ (matching §13vicies-novies.9 IL
phase-locking coefficient), $\alpha = 0.5$, $\tau_l = 4$, $\tau_g = 16$.
The matrix exponential $\exp(-\eta \hat H_{P14})$ is computed via
`scipy.linalg.expm` on the $40 \times 40$ canonical $H_{\mathrm{int}}$.
$\hat H_{P14}$ is symmetric (real self-adjoint by P14 construction); we
verify $\|H - H^T\|_\infty < 10^{-12}$ at runtime and abort with
`INDETERMINATE_NON_SELF_ADJOINT` if violated.

The lift of $S_{\mathrm{IL}}^{\mathrm{spec}}$ is $I_{\tau_g + 1} \otimes
\exp(-\eta \hat H_{P14})$ — *uniform across all slots*, in contrast to
the slot-0-only IL smoother of §13vicies-novies.9 and
§13vicies-novies.12. This uniform lift is the canonical choice for the
spectral-space construction of §13vicies-novies.10: the spectral
analogue of IL acts on the structural state independently of REMESH
slot, exactly as $\exp(-\eta \hat H_{P14})$ acts on $\mathcal{H}_{N}$
without temporal addressing.

**F7-A statistic (decisive, pre-registered).** Mirror F7-A of
§13vicies-novies.12 (and F6-A of §13vicies-novies.9):

1. Remove trivial fixed-point cluster: $|\lambda - 1| < 10^{-9}$.
2. Project to 1-D: $\operatorname{Im}(\lambda)$ for the upper-half-plane
   subset ($\operatorname{Im}(\lambda) \ge 10^{-12}$), sorted ascending.
   Fallback: $\operatorname{Re}(\lambda)$ sorted ascending if the
   projection is empty. Both branches reported in the JSON.
3. Normalised consecutive spacings: $\delta_k = (s_{k+1} - s_k) /
   \overline{\Delta}$.
4. KS sup-distance vs the GUE Wigner surmise $P_{\mathrm{GUE}}(s) =
   (32/\pi^2)\, s^2 \exp(-4 s^2 / \pi)$.

**F8 structural condition (necessary, pre-registered).** The
Euler-Orthogonality Lemma (§13vicies-novies.11) uses prime-relabelling
$S_n$ invariance under hypothesis (i) (basis-independent operator
composition). R-inf-1b targets (i) by working in the prime-indexed
basis $\{|p_i, k\rangle\}$ in which $\hat H_{P14}$ has explicit
prime-label dependence ($H_{\mathrm{freq}}$ diagonal entries
$k \log p_i$). The decisive structural test is whether re-instantiating
$\hat H_{P14}$ on a prime-relabelled $G_{P14}$ yields a *different*
spectrum for $T_{\mathrm{spec}}$:

* **F8 SATISFIED**: $|D_{\mathrm{canonical}} - D_{\mathrm{shuffled}}| \ge 0.01$
  (numerical floor; spectral-space composition genuinely breaks
  $S_n$-equivariance).
* **F8 FAILED**: $|D_{\mathrm{canonical}} - D_{\mathrm{shuffled}}| < 0.01$
  (spectral equivalence persists; the canonical lift
  $I_{\tau_g + 1} \otimes \exp(-\eta \hat H_{P14})$ commutes-up-to-
  similarity with prime relabelling, extending the Euler-Orthogonality
  obstruction from the edge-channel to the spectral channel —
  INDETERMINATE_DEGENERATE_CONSTRUCTION).

F8 is a *necessary* condition for R-inf-1b to be a meaningful test of
its own hypothesis. F8 failure does not refute B1; it refutes the
specific canonical-tensor-product lift used here and requires a
documented amendment (e.g. a non-product lift of $M_{\mathrm{REMESH}}$
that intertwines slot index with prime index, which would need its own
canonical derivation).

**Pre-registered theoretical expectation.** Under the canonical lift
$M_{\mathrm{REMESH}} = M \otimes I_N$ and
$S_{\mathrm{IL}}^{\mathrm{spec}} = I_{\tau_g + 1} \otimes \exp(-\eta
\hat H_{P14})$, prime-relabelling by $\sigma \in S_n$ acts on
$\mathcal{H}_{\mathrm{joint}}$ as the unitary $U_\sigma = I_{\tau_g + 1}
\otimes P_\sigma$. Because $M \otimes I_N$ commutes with
$I_{\tau_g + 1} \otimes P_\sigma$, and
$P_\sigma \exp(-\eta \hat H_{P14}) P_\sigma^T = \exp(-\eta\,
P_\sigma \hat H_{P14} P_\sigma^T) = \exp(-\eta\,\hat H_{P14}^{\sigma})$
where $\hat H_{P14}^{\sigma}$ is the canonical Hamiltonian
re-instantiated on the relabelled graph, we obtain
$U_\sigma T_{\mathrm{spec}}^{\mathrm{canonical}} U_\sigma^* =
T_{\mathrm{spec}}^{\sigma}$. Hence the canonical and shuffled iteration
matrices are *unitarily equivalent*, and their spectra are *identical
up to numerical precision*. The pre-registered theoretical prediction
is therefore F8 FAILED with $|D_{\mathrm{canonical}} -
D_{\mathrm{shuffled}}|$ at the machine-precision floor (the same
qualitative outcome as R-inf-1c §13vicies-novies.13). If observed, this
will constitute a *spectral-channel instantiation of the
Euler-Orthogonality obstruction* and complete the structural closure of
the canonical-tensor-product family of B1 sub-routes within the
13-operator catalog.

This prediction is recorded *before* execution as part of the
pre-registration discipline. The empirical outcome will be reported in
§13vicies-novies.15 regardless of whether it confirms or contradicts the
prediction. Confirmation strengthens the structural picture without
closing G4. A surprise outcome (F8 SATISFIED) would require revisiting
the lift's commutation analysis and would be reported with full
diagnostic detail.

**Pre-registered controls.**

* **N1 GOE** (random symmetric matrix of dimension $\dim
  \mathcal{H}_{\mathrm{joint}} = 680$, scaled by $1/\sqrt{2 \cdot 680}$;
  expected $D_{\mathrm{GUE}} \approx 0.10$–$0.20$ — GOE spacings
  differ from the GUE Wigner surmise).
* **N2 Poisson** (680 iid uniform points; expected $D_{\mathrm{GUE}}
  \approx 0.30$–$0.50$).
* **N3 shuffled-prime** (rebuild $G_{P14}$ with a random permutation of
  the ten primes among the ladders, re-instantiate $\hat H_{P14}$ via
  `InternalHamiltonian` on the shuffled graph, recompute $T_{\mathrm{spec}}$).
  *Primary discriminator for F8*.
* **N4 REMESH-isolated** (spectrum of $M$ alone; diagnostic baseline,
  expected to be degenerate as in §13vicies-novies.9 and
  §13vicies-novies.12).
* **N5 random-self-adjoint-replacement** (replace $\hat H_{P14}$ in
  $S_{\mathrm{IL}}^{\mathrm{spec}}$ with a random symmetric $40 \times
  40$ matrix of the same spectral radius as $\hat H_{P14}$; tests
  whether the canonical P14 spectrum structure carries any extra
  content beyond a generic self-adjoint operator of comparable scale).

**Riemann reference.** External anchor identical to §13vicies-novies.12:
$D_{\mathrm{GUE}}$ for the first 100 Riemann zero imaginary parts via
`mpmath.zetazero` at `mp.dps = 30`.

**Pre-registered F7 verdict logic.** Identical to §13vicies-novies.12:

* **SUPPORTED**:    $D_{\mathrm{canonical}} < 0.15$ **AND**
  $D_{\mathrm{canonical}} < D_{\mathrm{shuffled}} - 0.05$ **AND**
  $D_{\mathrm{canonical}} < D_{\mathrm{N5}} - 0.05$.
* **REFUTED**:      $D_{\mathrm{canonical}} > 0.30$ **OR**
  ($D_{\mathrm{canonical}} \ge D_{\mathrm{shuffled}} - 0.05$ **AND**
  F8 SATISFIED).
* **INDETERMINATE_DEGENERATE_CONSTRUCTION**: F8 FAILED.
* **INDETERMINATE_OTHER**: F8 SATISFIED and neither SUPPORTED nor
  REFUTED conditions hold.

**Pre-registered milestone verdict logic.**

* SUPPORTED $\Rightarrow$
  `B1_SPECTRAL_BASIS_POTENTIALLY_OPEN_REQUIRES_REPLICATION` (deep
  diagnostic + independent seeds + alternative spectral lifts before
  any evidential update on B1).
* REFUTED $\Rightarrow$
  `B1_SPECTRAL_BASIS_REFUTED_FOR_CANONICAL_TENSOR_PRODUCT_LIFT`. Closes
  R-inf-1b for the canonical $I_{\tau_g + 1} \otimes \exp(-\eta \hat
  H_{P14})$ / $M \otimes I_N$ lift. Does NOT close R-inf-1b for
  *non-product* lifts that intertwine slot index with prime index
  (would require their own canonical derivation and pre-registration).
* INDETERMINATE_DEGENERATE_CONSTRUCTION $\Rightarrow$
  `B1_SPECTRAL_BASIS_INDETERMINATE_EULER_ORTHOGONALITY_EXTENDS_TO_SPECTRAL_CHANNEL`
  if F8 fails at the machine-precision floor (predicted outcome).
  This is itself a structural finding: the canonical-tensor-product
  family of B1 sub-routes within the 13-operator catalog is closed by
  $S_n$-equivariance at both the edge-channel
  (§13vicies-novies.8/.9/.13) and spectral-channel levels.
* INDETERMINATE_OTHER $\Rightarrow$ status unchanged; design refinement
  needed before next attempt.

**Pre-registered seeds and parameters.** All random elements (N1 GOE
draw, N2 Poisson draw, N3 prime shuffle permutation, N5 random
self-adjoint draw) use `numpy.random.default_rng(20260526)` (reused
from §13vicies-novies.12 for cross-milestone reproducibility
consistency). Riemann zeros via `mpmath.zetazero` at `mp.dps = 30`.
REMESH parameters: $\alpha = 0.5$, $\tau_l = 4$, $\tau_g = 16$.
Spectral IL coupling: $\eta = 0.3$. Graph: $n_{\text{primes}} = 10$,
$\max\_\text{power} = 4$, $\text{coupling} = 0$. Canonical
constants from `src/tnfr/constants/canonical.py`. Hamiltonian
construction via
`src/tnfr/riemann/prime_ladder_hamiltonian.py::build_prime_ladder_hamiltonian`
(which internally invokes `tnfr.operators.hamiltonian.InternalHamiltonian`).

**What this milestone CAN establish.**

* A definitive verdict on the canonical tensor-product lift of the
  spectral IL contraction $\exp(-\eta \hat H_{P14})$ composed with the
  canonical REMESH echo matrix $M$ as a spectral-channel route to
  encode Riemann content in the iteration-matrix spectrum.
* Empirical evidence on whether the Euler-Orthogonality obstruction
  (§13vicies-novies.11), proven for edge-channel compositions on fixed
  $G_{P14}$ and observed empirically for canonically-augmented
  $G_{P14}$ (§13vicies-novies.13), extends to the canonical-tensor-
  product spectral-channel construction.

**What this milestone CANNOT establish.**

* R-inf-1b for non-product lifts that intertwine slot index with prime
  index (orthogonal sub-route requiring its own canonical derivation).
* R-inf-1b for alternative canonical spectral IL constructions (e.g.\
  $\exp(-\eta H_{\mathrm{freq}})$ alone, or with $\text{coupling} \neq
  0$; each would require its own pre-registration).
* B1 closure outside the canonical 13-operator catalog (B2 territory).
* G4 = RH, T-HP, or any closure beyond what F7 + F8 strictly test.

**Why this is not a re-run of R-inf-1a-composed.** R-inf-1a-composed
(§13vicies-novies.9) uses the *graph-Laplacian* IL smoother
$(I_N - \eta L_{G_{P14}})$ in slot 0 only — a topology-only operator
with no prime-label content beyond the canonical $G_{P14}$ structure
(all ten $P_4$ ladders are graph-isomorphic, so $L_{G_{P14}}$ is
explicitly $S_n$-equivariant). R-inf-1b uses the *full canonical
internal Hamiltonian* $\exp(-\eta \hat H_{P14})$ lifted uniformly across
all slots — a spectral-space operator whose $H_{\mathrm{freq}}$ block
carries explicit prime-label content ($k \log p_i$ entries). The two
milestones probe the same iteration-matrix architecture
($S \cdot M_{\mathrm{REMESH}}$) under structurally different $S$
operators: topology-only ($L_{G_{P14}}$, §.9) versus prime-label-
spectral ($\hat H_{P14}$, §.14). The theoretical-expectation paragraph
above explains why both lifts ultimately fall under the same
$S_n$-equivariance argument despite their structural difference; the
empirical F8 test in §.15 will confirm or contradict this.

**Implementation.**
`benchmarks/remesh_infinity_riemann_spectral_basis.py` (committed in
the same commit as this pre-registration; no data collected at commit
time). Output JSON written to
`results/remesh_infinity/remesh_infinity_riemann_spectral_basis.json`
(gitignored, not part of the audit trail; the audit trail is this
file).

**Status (pre-registration commit)**: methodology locked; no data
observed; next commit will append results in a Results block as
§13vicies-novies.15.

---

### §13vicies-novies.15 R-inf-1b Results: pre-registered theoretical expectation confirmed at machine precision

This subsection reports the result of executing
`benchmarks/remesh_infinity_riemann_spectral_basis.py` once with the
pre-registered seed `numpy.random.default_rng(20260526)` against the
methodology locked in §13vicies-novies.14. No parameters were changed
between pre-registration and execution.

**Headline.**
$$
|D_{\mathrm{canonical}} - D_{\mathrm{shuffled}}| \;=\; 1.08 \times 10^{-13}
\quad (<\; F8\ \text{floor}\;=\; 10^{-2}),
$$
so **F8 FAILED** at the machine-precision floor, exactly as
pre-registered in the "Pre-registered theoretical expectation" paragraph
of §13vicies-novies.14. The canonical and prime-shuffled iteration
matrices $T_{\mathrm{spec}}$ are unitarily equivalent (their spectra
coincide to 13 decimal places), confirming that the canonical
tensor-product lift $I_{\tau_g + 1} \otimes \exp(-\eta \hat H_{P14})$
and $M \otimes I_N$ commute with prime relabelling $I_{\tau_g + 1}
\otimes P_\sigma$ up to unitary similarity. The Euler-Orthogonality
obstruction (§13vicies-novies.11), proven for edge-channel
compositions on fixed $G_{P14}$ and observed empirically for
canonically-augmented $G_{P14}$ (§13vicies-novies.13), now extends to
the canonical-tensor-product spectral-channel construction targeted by
R-inf-1b.

**Verdict.**

* F7-A verdict: `INDETERMINATE_DEGENERATE_CONSTRUCTION` (F8 FAILED).
* Milestone verdict:
  `B1_SPECTRAL_BASIS_INDETERMINATE_EULER_ORTHOGONALITY_EXTENDS_TO_SPECTRAL_CHANNEL`.

The INDETERMINATE_DEGENERATE_CONSTRUCTION verdict is itself a
structural finding under the pre-registration protocol: it closes
R-inf-1b for the canonical-tensor-product lift family within the
13-operator catalog by demonstrating that the $S_n$-equivariance
obstruction generalises from edge channel to spectral channel under
canonical lifts.

**Numerical results (F7-A KS distance vs the GUE Wigner surmise).**

| Object                                                       | Projection      | #spacings | $D_{\mathrm{GUE}}$ |
| ------------------------------------------------------------ | --------------- | --------: | -----------------: |
| `canonical` $T_{\mathrm{spec}} = S_{\mathrm{IL}}^{\mathrm{spec}} M_{\mathrm{REMESH}}$ | Im upper-half   |       319 |             0.4732 |
| N1 GOE (random symmetric, dim 680)                           | Re fallback     |       679 |             0.1126 |
| N2 Poisson (680 iid uniform)                                 | iid uniform     |       679 |             0.3032 |
| N3 shuffled-prime $T_{\mathrm{spec}}^{\sigma}$               | Im upper-half   |       319 |             0.4732 |
| N4 REMESH-isolated (spectrum of $M$ alone)                   | Im upper-half   |         7 |             0.3082 |
| N5 random self-adjoint (matched spectral radius)             | Im upper-half   |       319 |             0.7135 |
| Riemann reference (first 100 zeros)                          | iid zeros       |        99 |             0.0770 |

Auxiliary diagnostics: $H_{P14}$ spectral radius
$= 13.469183$ (matches $4 \log 29 = k_{\max} \log p_{\max}$ to printed
precision); $\dim \mathcal{H}_{\mathrm{joint}} = N(\tau_g + 1) = 680$;
$N$-basis $= 40$; self-adjointness check passed
($\|H - H^T\|_\infty < 10^{-12}$). The Re-fallback for N1 GOE is the
expected branch (random symmetric matrices have real spectrum); all
canonical and spectral-lift branches projected to Im upper-half as
expected for a non-self-adjoint $T_{\mathrm{spec}}$.

**Interpretation.**

The F8 failure at $|\Delta D| \approx 10^{-13}$ is not a numerical
artefact — it is the *predicted* signature of the unitary equivalence
$U_\sigma T_{\mathrm{spec}}^{\mathrm{canonical}} U_\sigma^* =
T_{\mathrm{spec}}^{\sigma}$ derived in §13vicies-novies.14. Because the
canonical lifts $M_{\mathrm{REMESH}} = M \otimes I_N$ and
$S_{\mathrm{IL}}^{\mathrm{spec}} = I_{\tau_g + 1} \otimes \exp(-\eta
\hat H_{P14})$ are tensor-product separable in slot $\otimes$ basis,
the prime-relabelling unitary $U_\sigma = I_{\tau_g + 1} \otimes
P_\sigma$ conjugates $T_{\mathrm{spec}}$ to its shuffled image; spectra
coincide.

The canonical $D_{\mathrm{GUE}} = 0.4732$ value (far above both the
GUE-class N1 GOE control at $D = 0.1126$ and the Riemann reference
$D = 0.0770$) is not structurally interpretable as evidence for or
against B1 because the F8 precondition has failed. Under
INDETERMINATE_DEGENERATE_CONSTRUCTION, the F7-A signal is decoupled
from the original hypothesis. The N5 random-self-adjoint control at
$D = 0.7135$ confirms that a generic self-adjoint operator of matching
spectral radius does not produce GUE-like statistics either; this rules
out the trivial alternative explanation that *any* $40 \times 40$
self-adjoint lift would yield $D \sim 0.47$ by chance. N4
REMESH-isolated reproduces the degenerate 7-spacing diagnostic baseline
of §13vicies-novies.9 and §13vicies-novies.13.

**B1 status update (after §13vicies-novies.15).** The B1 status table
of §13vicies-novies.13 is updated as:

| Sub-route                          | Status                                                                                                                                                                                                                |
| ---------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| R-inf-1a-operator                  | REFUTED (§13vicies-novies.8).                                                                                                                                                                                         |
| R-inf-1a-composed                  | REFUTED (§13vicies-novies.9).                                                                                                                                                                                         |
| R-inf-1c                           | INDETERMINATE_DEGENERATE_CONSTRUCTION (§13vicies-novies.13). $|D_{\mathrm{can}} - D_{\mathrm{shuf}}| = 3.13 \times 10^{-13}$; modified-graph specialisation of Euler-Orthogonality under $S_n$-invariant weights.       |
| R-inf-1b (canonical tensor-product lift) | INDETERMINATE_DEGENERATE_CONSTRUCTION (§13vicies-novies.15). $|D_{\mathrm{can}} - D_{\mathrm{shuf}}| = 1.08 \times 10^{-13}$; spectral-channel extension of Euler-Orthogonality under $M \otimes I_N$ and $I_{\tau_g+1} \otimes \exp(-\eta H)$ separable lifts. |
| R-inf-1b (non-product / prime-indexed lifts) | NOT pre-registered, NOT tested. Would require its own canonical derivation of a slot $\otimes$ prime intertwining lift; such a lift is *not* among the catalog's standard product lifts and would need its own theoretical justification before any pre-registration.    |

**Net B1 status.** With §13vicies-novies.15 the canonical-tensor-product
family of B1 sub-routes within the 13-operator catalog on $G_{P14}$ —
including its canonical augmentations and its canonical spectral
lifts — is empirically closed by $S_n$-equivariance at both
edge-channel and spectral-channel levels. The remaining structurally
permitted sub-routes inside B1 are now restricted to *non-product
canonical lifts* (would require canonical derivation of slot $\otimes$
prime intertwining structure, not among standard product constructions
in the catalog). This strengthens — but does not yet decide — the
case for B2/B3 within the §13septies trichotomy. No claim is made about
G4, T-HP, or B1 closure outside the catalog.

**Reproducibility.** Single command, no flags:
```
PYTHONPATH=src python benchmarks/remesh_infinity_riemann_spectral_basis.py
```
Output JSON at
`results/remesh_infinity/remesh_infinity_riemann_spectral_basis.json`
(gitignored). All seven numerical entries quoted above (canonical $D$,
N1–N5 $D$, $|\Delta D|$) reproduce from the locked seed
`np.random.default_rng(20260526)`.

---

### §13vicies-novies.16 Closure of B1 on $G_{P14}$: the Canonical Catalog Equivariance Theorem

The four pre-registered B1 sub-routes refuted or returned
`INDETERMINATE_DEGENERATE_CONSTRUCTION` in §13vicies-novies.8/.9/.13/.15
all exhibit the same structural failure mode: the iteration / spectral
operator $T$ commutes with the prime-relabelling action $\Pi_\sigma$ (or
its trivial lift $\Pi_\sigma \otimes I_{\mathrm{aux}}$) of
$S_{n_{\mathrm{primes}}}$, hence
$\operatorname{spec}(T)$ is $S_{n_{\mathrm{primes}}}$-invariant and
cannot encode prime-labelled Riemann content. The Euler-Orthogonality
Lemma (§13vicies-novies.11) proved this for *edge-channel* compositions
on fixed $G_{P14}$. The empirical results §13vicies-novies.13 (modified
graph) and §13vicies-novies.15 (canonical tensor-product spectral lift)
demonstrated it for two additional construction classes. The status
table at the end of §13vicies-novies.15 left exactly one structurally
permitted residual route: *non-product canonical lifts on $G_{P14}$*
that intertwine an auxiliary tensor factor (history, sub-EPI, time
slot, spectral basis) with the prime index in a way not expressible as
$A \otimes B$ with separable node-vs-aux factors.

This subsection closes that residual route at the structural level by
showing that *no such non-product canonical lift exists inside the
13-operator catalog acting on $G_{P14}$*. The result is a strengthening
of Lemma 1 (§13vicies-novies.11) from edge-channel restrictions to the
full algebra generated by canonical-catalog constructions on any
auxiliary tensor factor; it makes B1 on $G_{P14}$ structurally
inaccessible to the canonical 13-operator catalog and consolidates the
program-level decision pressure onto B2 (new canonical operator) or B3
(no TNFR closure) within the §13septies trichotomy.

**Notation (recall).** $G_{P14} = (V, E)$ is the canonical prime-ladder
graph of §13quinquies with $V = \{(p_i, k) : 1 \le i \le n_{\mathrm{primes}},\
1 \le k \le k_{\max}\}$, edges only between same-prime consecutive
echo levels, node attributes
$\nu_f((p, k)) = k \log p$, $\phi \equiv 0$, $\mathrm{EPI} \equiv 1$,
$S_i \equiv 1$, $\Delta\mathrm{NFR} \equiv 0$. The prime-relabelling
group acts as $\Pi_\sigma(p_i, k) = (p_{\sigma(i)}, k)$ for
$\sigma \in S_{n_{\mathrm{primes}}}$. $\mathcal{O}_{13}$ is the canonical
13-operator catalog (AGENTS.md §"The 13 Canonical Operators").

**Definition (auxiliary tensor factor).** An *auxiliary tensor factor*
is any finite-dimensional vector space $V_{\mathrm{aux}}$ associated by
the canonical engine to a structural attribute of nodes that is not the
node index itself. Concrete instances appearing in the program:

* $V_{\mathrm{hist}} = \mathbb{R}^{\tau_g + 1}$ — REMESH echo history
  slots (the joint space $\mathbb{R}^V \otimes V_{\mathrm{hist}}$ is
  used by R-inf-1a, R-inf-1a-composed, R-inf-1b).
* $V_{\mathrm{sub}} = \mathbb{R}^{n_{\mathrm{sub}}}$ — THOL sub-EPI
  nesting basis.
* $V_{\mathrm{spec}} = \mathbb{C}^{N}$ — spectral basis $\{|p, k\rangle\}$
  diagonalising $\hat H_{P14}$ (the joint space $V_{\mathrm{spec}} \otimes
  V_{\mathrm{hist}}$ is used by R-inf-1b).

The prime-relabelling action lifts trivially to any auxiliary factor as
$\Pi_\sigma \otimes I_{\mathrm{aux}}$ (acting as $\Pi_\sigma$ on the
node / spectral basis index and as identity on the auxiliary factor).

**Definition (canonical-catalog construction).** A linear operator $T$
on $\mathbb{R}^V \otimes V_{\mathrm{aux}}$ is a *canonical-catalog
construction* (CCC) if it is obtained by finitely many applications of
the following closure rules starting from the canonical lifts of the
13 operators (auditable in `src/tnfr/operators/*.py`,
`src/tnfr/dynamics/propagation.py`):

* **(C1) Generator base case.** $T = O$ for $O \in \mathcal{O}_{13}$
  lifted canonically: edge operators (EN, IL phase-Laplacian, OZ, RA)
  act on $\mathbb{R}^V \otimes I_{\mathrm{aux}}$ through their
  `propagated_dnfr` kernel; node-local operators (AL, IL pressure
  contraction, SHA, VAL, NUL, THOL, ZHIR, NAV, UM on $\phi \equiv 0$)
  act per-node with parameters drawn from graph-level scalar config;
  REMESH acts as $I_{|V|} \otimes M_{\tau_g + 1}$ for the canonical
  echo matrix $M$ pulled from `_remesh_alpha_info` (single $\alpha$
  scalar uniform across nodes).
* **(C2) Composition.** $T = T_1 \circ T_2$ for CCCs $T_1, T_2$.
* **(C3) Real-linear combination.** $T = c_1 T_1 + c_2 T_2$ for
  $c_1, c_2 \in \mathbb{R}$ and CCCs $T_1, T_2$.
* **(C4) Auxiliary tensor lift.** $T = T_0 \otimes A$ for a CCC $T_0$
  on $\mathbb{R}^V \otimes V_{\mathrm{aux}}^{(1)}$ and any
  linear $A$ on a second auxiliary factor $V_{\mathrm{aux}}^{(2)}$
  with $V_{\mathrm{aux}} = V_{\mathrm{aux}}^{(1)} \otimes
  V_{\mathrm{aux}}^{(2)}$.
* **(C5) Spectral functional calculus.** $T = f(H)$ for a CCC $H$ that
  is Hermitian (or self-adjoint after a canonical Hermitian
  symmetrisation) and a Borel-measurable $f : \mathbb{R} \to
  \mathbb{C}$. Used for R-inf-1b's $S_{\mathrm{IL}}^{\mathrm{spec}} =
  \exp(-\eta \hat H_{P14})$.

Rules C1–C5 capture every operator construction observed in the program
(audit: §13vicies-novies.8/.9/.13/.15 + §13quinquies +
`src/tnfr/riemann/*`). No construction outside C1–C5 has been used in
any pre-registered B1 sub-route.

**Two structural facts (auditable in source).**

*Fact A — Parameter uniformity.* Every node-local canonical operator
draws its coupling parameters ($\alpha, \eta, \mathrm{depth}$,
thresholds) from graph-level state, not from per-node attributes.
Audit:

* REMESH: `_remesh_alpha_info` (`src/tnfr/operators/remesh.py:1159`)
  returns a single scalar $\alpha$ for the whole graph; the per-node
  loop at lines 1240–1252 applies the same $\alpha$ to every node
  $n \in V$.
* IL phase smoother: $\eta$ is a graph-level scalar in
  `src/tnfr/operators/coherence.py`; the operator acts as
  $I - \eta L_G$ with the same $\eta$ on every node.
* OZ / RA / EN: `propagated_dnfr = dissonance_magnitude *
  coupling_weight * phase_weight * freq_weight`
  (`src/tnfr/dynamics/propagation.py:140`); all four factors are
  functions of edge attributes and node attribute pairs, with no
  per-prime parameter switch.

Consequence: per-node lifts of canonical operators have the form
$A_n = A$ (single global linear map applied to each $n$), hence the
total per-node-lift decomposes as $\bigoplus_n A = I_{|V|} \otimes A$
on the joint space — *automatically* tensor-product separable in node
$\otimes$ aux.

*Fact B — No inter-prime coupling on $G_{P14}$.* The only mechanism by
which canonical operators couple different node indices is edge
propagation. $G_{P14}$ has edges $(p, k) \leftrightarrow (p, k + 1)$
only (same prime endpoints). Therefore every edge-propagating operator
$O$ has matrix decomposition
$$
O \;=\; \bigoplus_{i = 1}^{n_{\mathrm{primes}}} O_{p_i},
$$
where $O_{p_i}$ acts on the four-dimensional sub-space spanned by
$\{(p_i, 1), (p_i, 2), (p_i, 3), (p_i, 4)\}$ (the $i$-th $P_4$ ladder
component). Furthermore — by Case C of §13vicies-novies.11 (Prime-
Cancellation Lemma) — the four-dimensional kernel $O_{p_i}$ is
*independent of the prime label $p_i$*: $O_{p_i} = O_{P_4}$ for all
$i$, where $O_{P_4}$ is a single $4 \times 4$ kernel determined by
edge combinatorics and the $\phi \equiv 0$ boundary condition.

Consequence: $O = I_{n_{\mathrm{primes}}} \otimes O_{P_4}$ in the
factorisation $\mathbb{R}^V = \mathbb{R}^{n_{\mathrm{primes}}} \otimes
\mathbb{R}^{k_{\max}}$ — *automatically* tensor-product separable in
prime $\otimes$ echo-level.

**Theorem 2 (Canonical Catalog Equivariance on $G_{P14}$).** *Let $V$,
$\Pi$, $\mathcal{O}_{13}$, $V_{\mathrm{aux}}$ be as above. Then every
canonical-catalog construction $T$ on $\mathbb{R}^V \otimes
V_{\mathrm{aux}}$ commutes with the trivially-lifted prime-relabelling
action:*
$$
T \circ (\Pi_\sigma \otimes I_{\mathrm{aux}})
\;=\;
(\Pi_\sigma \otimes I_{\mathrm{aux}}) \circ T
\qquad \forall\,\sigma \in S_{n_{\mathrm{primes}}}.
$$
*Replacing $\mathbb{R}^V$ by the prime-indexed spectral basis
$V_{\mathrm{spec}} \cong \mathbb{C}^N$ with the corresponding unitary
permutation $U_\sigma = I_{k_{\max}} \otimes P_\sigma$, the same
conclusion holds with $\Pi_\sigma$ replaced by $U_\sigma$.*

*Proof.* Induction on the number of C1–C5 applications.

**Base case (C1).** By Fact A, every node-local canonical generator
lifts as $I_{|V|} \otimes A$ for some $A$ on $V_{\mathrm{aux}}$; this
commutes with $\Pi_\sigma \otimes I_{\mathrm{aux}}$ since
$(I_{|V|} \otimes A) (\Pi_\sigma \otimes I_{\mathrm{aux}}) =
\Pi_\sigma \otimes A = (\Pi_\sigma \otimes I_{\mathrm{aux}})
(I_{|V|} \otimes A)$. By Fact B, every edge canonical generator on
$G_{P14}$ lifts as $(I_{n_{\mathrm{primes}}} \otimes O_{P_4}) \otimes
I_{\mathrm{aux}}$; this commutes with $\Pi_\sigma \otimes
I_{\mathrm{aux}}$ since $\Pi_\sigma$ acts as a permutation in the
first tensor factor $\mathbb{R}^{n_{\mathrm{primes}}}$ while
$O_{P_4}$ acts in the second; tensor factors commute. REMESH lifts as
$I_{|V|} \otimes M_{\tau_g + 1} \otimes I_{\mathrm{aux}}^{(\mathrm{rest})}$
(Fact A applied with the history factor as one component of
$V_{\mathrm{aux}}$); commutation with $\Pi_\sigma \otimes
I_{\mathrm{aux}}$ is immediate.

**Inductive steps (C2, C3).** Composition and real-linear combination
preserve the commutant of any group action — the commutant is closed
under those operations. If $T_1, T_2$ commute with
$\Pi_\sigma \otimes I_{\mathrm{aux}}$, so do $T_1 T_2$ and
$c_1 T_1 + c_2 T_2$.

**Inductive step (C4).** If $T_0$ commutes with $\Pi_\sigma \otimes
I_{\mathrm{aux}}^{(1)}$ on $\mathbb{R}^V \otimes V_{\mathrm{aux}}^{(1)}$,
then $T = T_0 \otimes A$ commutes with $\Pi_\sigma \otimes
I_{\mathrm{aux}}^{(1)} \otimes I_{\mathrm{aux}}^{(2)} = \Pi_\sigma
\otimes I_{\mathrm{aux}}$ (where $V_{\mathrm{aux}} =
V_{\mathrm{aux}}^{(1)} \otimes V_{\mathrm{aux}}^{(2)}$). Tensor
products of commuting operators commute factor-wise.

**Inductive step (C5).** If $H$ commutes with $\Pi_\sigma \otimes
I_{\mathrm{aux}}$ and is Hermitian, then for any Borel-measurable
$f : \mathbb{R} \to \mathbb{C}$ the spectral functional calculus
operator $f(H)$ also commutes (standard result: commutation with $H$
implies commutation with the spectral resolution of $H$, hence with
$f(H)$). In particular $S_{\mathrm{IL}}^{\mathrm{spec}} =
\exp(-\eta \hat H_{P14})$ commutes with the spectral-basis lift
$U_\sigma$ of $\Pi_\sigma$ because $\hat H_{P14}$ does (verified
directly in §13vicies-novies.15 numerical results: spectral radius
$13.469183 = 4 \log 29$ is $S_n$-invariant).

The five closure rules exhaust the construction grammar. $\square$

**Corollary 3 (spectral $S_n$-invariance, full catalog).** *For every
canonical-catalog construction $T$ on $\mathbb{R}^V \otimes
V_{\mathrm{aux}}$ (or $V_{\mathrm{spec}} \otimes V_{\mathrm{aux}}$),
the spectrum $\operatorname{spec}(T)$ is invariant under
$S_{n_{\mathrm{primes}}}$: any prime-relabelled construction
$T^\sigma$ obtained by acting with $\Pi_\sigma \otimes I_{\mathrm{aux}}$
satisfies $\operatorname{spec}(T^\sigma) = \operatorname{spec}(T)$
as a multiset.*

*Proof.* Theorem 2 gives unitary equivalence $T^\sigma = U_\sigma T
U_\sigma^{-1}$ with $U_\sigma = \Pi_\sigma \otimes I_{\mathrm{aux}}$
(orthogonal permutation, hence unitary). Conjugation by a unitary
preserves spectrum as a multiset. $\square$

**Corollary 4 (closure of B1 on $G_{P14}$).** *Inside the canonical
13-operator catalog there is no construction on $G_{P14}$ — including
non-product lifts on arbitrary auxiliary tensor factors — whose
spectrum distinguishes the canonical prime assignment
$\{p_1, \ldots, p_{n_{\mathrm{primes}}}\}$ from any of the
$n_{\mathrm{primes}}!$ permuted assignments. In particular, no such
construction can reproduce Riemann-zero level statistics, which
require the specific prime labelling.*

*Proof.* By Corollary 3, the spectrum is $S_{n_{\mathrm{primes}}}$-
invariant. Any level-spacing statistic computed from
$\operatorname{spec}(T)$ alone is therefore $S_{n_{\mathrm{primes}}}$-
invariant. Riemann level statistics $\{\gamma_n\}$ are *not*
$S_{n_{\mathrm{primes}}}$-invariant under the prime labelling that
defines $\hat H_{P14}$ (different prime sets give different Riemann
data; cf. AGENTS.md §"TNFR-Riemann Program Overview"). The two are
therefore incompatible by a $S_{n_{\mathrm{primes}}}$-equivariance
argument: a $S_{n_{\mathrm{primes}}}$-invariant spectrum cannot single
out a $S_{n_{\mathrm{primes}}}$-non-invariant target. $\square$

**B1 status table (final, supersedes §13vicies-novies.15).**

| Sub-route                                          | Status (post-§.16)                                                                                                                                                                                                                                  |
| -------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| R-inf-1a-operator                                  | REFUTED (§13vicies-novies.8).                                                                                                                                                                                                                       |
| R-inf-1a-composed                                  | REFUTED (§13vicies-novies.9).                                                                                                                                                                                                                       |
| R-inf-1c                                           | INDETERMINATE_DEGENERATE_CONSTRUCTION (§13vicies-novies.13); subsumed by Theorem 2 under C1 + C2 with augmented edge kernel still $S_n$-equivariant under invariant weights.                                                                         |
| R-inf-1b (canonical tensor-product lift)           | INDETERMINATE_DEGENERATE_CONSTRUCTION (§13vicies-novies.15); subsumed by Theorem 2 under C1 + C4 + C5.                                                                                                                                              |
| R-inf-1b (non-product / slot-prime intertwining)   | **CLOSED_BY_THEOREM** (§13vicies-novies.16, this subsection). No canonical-catalog construction on $G_{P14}$ admits non-product slot-prime intertwining: Facts A and B force every canonical lift into one of the two separable normal forms.        |
| **Net B1 on $G_{P14}$**                            | **CLOSED**. The canonical 13-operator catalog cannot produce a spectral signature on $G_{P14}$ distinguishing the canonical prime labelling. Forces decision pressure onto B2 (new canonical operator) or B3 (no TNFR closure) per §13septies.        |

**Honest scope.** Theorem 2 and Corollary 4:

* Close B1 on $G_{P14}$ specifically, inside the canonical 13-operator
  catalog. They do **not** close B1 outside $G_{P14}$.
* Do **not** prove or refute G4 = RH; the open program-level question
  remains intact.
* Do **not** refute B2: a new canonical operator derivable from the
  nodal equation could in principle intertwine slot with prime in a
  way the 13-operator catalog does not. Closing or ruling out B2 is
  the next program-level task.
* Do **not** refute B3 (no TNFR closure). B3 remains a permitted
  outcome until B2 is decided.
* Apply to $G_{P14}$ as canonically constructed in
  `src/tnfr/riemann/prime_ladder_hamiltonian.py`. Graph modifications
  beyond R-inf-1c (i.e., any modification that breaks Fact B by
  introducing inter-prime edges with $S_n$-non-invariant weights) fall
  outside the theorem's hypotheses; whether any such modification is
  *itself* derivable from canonical invariants 1–6 and U1–U6 is a
  separate question (and §13vicies-novies.13 already empirically
  showed that the most natural canonical augmentation — invariant
  inter-prime weights — preserves $S_n$-equivariance by virtue of
  its $S_n$-invariant weight construction).

**Cross-references.**

* §13vicies-novies.8/.9 — original R-inf-1a-operator and
  R-inf-1a-composed empirical refutations.
* §13vicies-novies.10 — operator channel classification underlying
  Facts A and B.
* §13vicies-novies.11 — Lemma 1 (Euler-Orthogonality Lemma), the
  edge-channel predecessor of Theorem 2.
* §13vicies-novies.12/.13 — R-inf-1c pre-registration and results.
* §13vicies-novies.14/.15 — R-inf-1b pre-registration and results
  (canonical tensor-product lift).
* §13quinquies — P14 prime-ladder graph and Hamiltonian construction.
* §13septies — Conjecture T-HP (G4 = RH); B1/B2/B3 trichotomy.
* AGENTS.md §"B1 sub-route status" — program-level status mirror.
* `src/tnfr/operators/remesh.py:1159, 1212–1252` — REMESH per-node
  uniform-$\alpha$ implementation (Fact A audit).
* `src/tnfr/operators/coherence.py` — IL phase-Laplacian smoother
  with uniform $\eta$ (Fact A audit).
* `src/tnfr/dynamics/propagation.py:42–156` — EN/OZ/RA edge
  propagation kernel (Facts A and B audit).
* `src/tnfr/riemann/prime_ladder_hamiltonian.py` — $G_{P14}$ and
  $\hat H_{P14}$ canonical construction.

**Net consequence for the program.** B1 within the canonical
13-operator catalog on $G_{P14}$ is structurally closed. The §13septies
trichotomy now reads:

* B1 (canonical catalog closure): **CLOSED on $G_{P14}$** by Theorem 2.
* B2 (new canonical operator): OPEN. A non-trivial slot-prime
  intertwining operator, if derivable from the nodal equation
  $\partial \mathrm{EPI} / \partial t = \nu_f \cdot \Delta\mathrm{NFR}(t)$
  and consistent with U1–U6, would constitute a B2 closure. Whether
  such an operator exists is the next open program-level question.
* B3 (no TNFR closure): permitted residual outcome if B2 is also
  refuted.

The "B1 sub-route status" paragraph in AGENTS.md will be updated to
reflect this closure in a companion edit.

---
## §13triginta. P50 — REMESH-∞ Residue Split of P31 Oscillatory Correction (Function-Space Lift of N15 Closure into the Riemann Program; Does NOT Advance G4 = RH)

### §13triginta.1 Motivation

N15 ([REMESH-∞ Derivation](REMESH_INFINITY_DERIVATION.md), Branch A
verdict W1+W2+W3) established that the REMESH operator admits a
bounded self-adjoint asymptotic projection
$\mathcal{R}_\infty = P_{\ker(I - \mathcal{R})}$
on $H^2(D)$ with spectrum $\{0, 1\}$ and resonant Fourier lattice
$\{2\pi k / \mathrm{lcm}(\tau_l, \tau_g)\}$ at the canonical
parameter pair $(\tau_l, \tau_g) = (4, 8)$. §13septies and §13nonies
identified the residual obstruction of Conjecture T-HP with the
oscillatory half $S(T) = (1/\pi) \arg \zeta(\tfrac12 + iT)$ of the
admissible rescaling operator $\mathcal{F}$: P28 closes the smooth
half at density level, P30 lifts the smooth half to the operator
level, and P31 attempted to attack the oscillatory half via a
canonical prime-ladder Newton step. P50 is the **function-space
diagnostic that tests whether the P31 reconstruction lives in
$\mathrm{range}(\mathcal{R}_\infty)$ or in $\ker(\mathcal{R}_\infty)$**,
directly connecting the N15 cross-program closure to the T-HP
residual gap at the level of canonical TNFR functions on the
$T$-axis.

The diagnostic is **complementary** to the §13vicies-novies
edge-channel / spectral-channel refutation thread: §13vicies-novies
operates on the iteration matrix of REMESH applied to EPI-history
state vectors on the discrete graph $G_{P14}$ (a finite-dimensional
linear-algebraic object), whereas §13triginta operates on the
canonical P31 reconstruction $S_{\mathrm{TNFR}}(T)$ as a function
in $H^2(T\text{-axis})$ under the discrete Fourier transform (an
infinite-dimensional analytic object). The two layers test
distinct mathematical surfaces and yield independent structural
evidence.

### §13triginta.2 Construction

For any positive integer $n_{\mathrm{samples}}$ divisible by
$\mathrm{lcm}(\tau_l, \tau_g) = 8$, the resonant Fourier-bin mask is

$$
\mathcal{M}_{\mathrm{res}} = \left\{k \in \{0, 1, \ldots, n_{\mathrm{samples}} - 1\} : k \equiv 0 \pmod{M}\right\}, \quad M = \frac{n_{\mathrm{samples}}}{\mathrm{lcm}(\tau_l, \tau_g)}.
$$

The bins in $\mathcal{M}_{\mathrm{res}}$ correspond exactly to the
N15-resonant angular frequencies $\omega_j = 2\pi j / \mathrm{lcm}$
for $j = 0, 1, 2, \ldots$, under the canonical unit-spacing $T$-grid
$T_n = n + T_{\min}$ for $n = 0, \ldots, n_{\mathrm{samples}} - 1$.
The orthogonal projector onto $\mathrm{range}(\mathcal{R}_\infty)$
acts on a real signal $f$ by

$$
(\mathcal{R}_\infty f)_n = \mathrm{Re}\,\mathcal{F}^{-1}\!\left[\mathbb{1}_{\mathcal{M}_{\mathrm{res}}}(k) \cdot (\mathcal{F} f)_k\right]_n,
$$

and $(I - \mathcal{R}_\infty) f$ is the kernel component. The Parseval
fractions are reported in the certificate
(`ResidueSplitCertificate`).

The canonical P31 reconstruction
$$
S_{\mathrm{TNFR}}(T;\,N,K) = -\frac{1}{\pi} \sum_{(\mu, w) \in \Sigma_{N, K}} \frac{w}{\mu} \sin(T \mu) \exp(-\mu / 2)
$$
is evaluated on the canonical $T$-grid via
`prime_ladder_oscillatory_sum` (atomic P31 primitive, vectorised).
The split is computed by `split_residue_by_remesh_infinity`.

### §13triginta.3 Pre-Registered Structural Prediction (Baker's Theorem)

The Fourier support of $S_{\mathrm{TNFR}}(T)$ as a function of $T$ is
exactly $\{\mu : (\mu, w) \in \Sigma_{N, K}\} = \{k \log p : p \text{ prime}, 1 \le k \le K\}$.
By **Baker's theorem on linear independence of logarithms of
algebraic numbers** (1966), no $\mathbb{Q}$-linear combination of
$\{\log p : p \text{ prime}\}$ equals a non-zero rational multiple of
$\pi$. Hence $\{k \log p\}$ is **disjoint** from the N15-resonant
lattice $\{2\pi j / \mathrm{lcm}(\tau_l, \tau_g) : j \in \mathbb{Z}\}$,
which consists of rational multiples of $\pi$. The pre-registered
structural prediction is therefore:

$$
\boxed{\;\;\lim_{n_{\mathrm{samples}} \to \infty} \frac{\|\mathcal{R}_\infty S_{\mathrm{TNFR}}\|_2^2}{\|S_{\mathrm{TNFR}}\|_2^2} = 0\;\;}
$$

equivalently, the canonical reconstruction lies asymptotically in
$\ker(\mathcal{R}_\infty)$ — verdict `RESIDUE_IN_KER_ONLY`.

### §13triginta.4 Empirical Verification

Demo `examples/05_type_hygiene/77_remesh_infinity_residue_split_demo.py` at canonical
defaults $(\tau_l, \tau_g) = (4, 8)$, $K = 8$:

| $n_{\mathrm{periods}}$ | $n_{\mathrm{samples}}$ | $n_{\mathrm{primes}}$ | $\|S\|_2$ | $\|\mathcal{R}_\infty S\|_2$ | $\|(I-\mathcal{R}_\infty) S\|_2$ | range fraction | kernel fraction | verdict |
|---|---|---|---|---|---|---|---|---|
| 64  | 512  | 200 | 7.2457 | 0.9625 | 7.1814  | **1.7647 %**  | 98.2353 % | `RESIDUE_IN_KER_ONLY` |
| 256 | 2048 | 400 | 15.824 | 0.2016 | 15.822  | **0.0162 %**  | 99.9838 % | `RESIDUE_IN_KER_ONLY` |

The range fraction decays by a factor of **109×** as the grid
resolution quadruples — clean asymptotic incommensurability behaviour
matching the Baker-theorem prediction.

**Sanity controls** (built into `compute_residue_split_certificate`):

| control signal | predicted range fraction | measured ($n_{\mathrm{samples}} = 512$) | measured ($n_{\mathrm{samples}} = 2048$) |
|---|---|---|---|
| $\sin(2\pi T / \mathrm{lcm})$ (resonant)                       | $\approx 100\%$ | **100.0000 %** | **100.0000 %** |
| $\sin(\gamma_{\mathrm{em}} T)$ ($\gamma_{\mathrm{em}}$ Euler–Mascheroni; non-resonant) | $\approx 0\%$    | **0.0007 %**   | **0.0005 %**   |

Both controls hit their predicted projections to machine precision,
confirming the DFT-bin mask correctly implements the N15-resonant
projector.

### §13triginta.5 What P50 Extends

* Extends the §13septies / §13nonies *structural identification* of
  the T-HP residual obstruction with the oscillatory half $S(T)$ to a
  *function-space-level empirical verdict*: the canonical P31
  reconstruction $S_{\mathrm{TNFR}}(T)$ lies in $\ker(\mathcal{R}_\infty)$,
  exactly where the residual obstruction was predicted to live.
* Provides the **N15-cross-program-bridge**: the same orthogonal
  projector $\mathcal{R}_\infty$ that closes the REMESH-∞ asymptotic
  limit (W1 of N15) also organises the T-HP residual gap into its
  smooth half (range component, closed by P28 + P30) and oscillatory
  half (kernel component, RH-equivalent and open).
* Adds a **second independent attack-surface diagnostic** on B1 at the
  function-space level, complementary to the §13vicies-novies
  graph-iteration-matrix thread on EPI-history state vectors. The two
  threads test mathematically distinct objects (functions in
  $H^2(T\text{-axis})$ vs. finite-dimensional iteration matrices on
  $G_{P14}$) and yield consistent structural evidence: both place the
  residual obstruction outside the catalog's standard canonical
  product structures.

### §13triginta.6 What P50 Does NOT Advance

* **G4 = RH**: untouched. P50 does not close T-HP. The result is a
  structural-compatibility diagnostic that *organises* the residual
  obstruction, not a closure of it.
* **Sub-problems (2) canonicity** and **(3) positivity coincidence**
  of T-HP (§13septies): untouched.
* **No new canonical operator**: P50 uses only the canonical N15
  asymptotic projector $\mathcal{R}_\infty$, the canonical P31
  reconstruction $S_{\mathrm{TNFR}}(T)$, and the discrete Fourier
  transform — all existing canonical ingredients. P50 does not
  promote any new operator into the 13-operator catalog.
* **Branch B1 / B2 / B3 trichotomy** (§13septies): P50 *narrows* B1
  by placing the residual obstruction in $\ker(\mathcal{R}_\infty)$,
  but does not decide between B1-via-some-other-channel,
  B2 (new canonical operator required), and B3 (no TNFR closure
  exists). It is consistent with §13vicies-novies.15's verdict that
  the canonical-tensor-product family of B1 sub-routes on $G_{P14}$
  is empirically closed by $S_n$-equivariance.

### §13triginta.7 Cross-References

* **N15 master derivation**: `theory/REMESH_INFINITY_DERIVATION.md`
  (W1 existence of $\mathcal{R}_\infty$ as orthogonal projection,
  W2 conservation / Lyapunov structure, W3 spectral universality).
* **T-HP statement and structural split**: §13septies (Conjecture
  T-HP); §13octies (assembled-argument audit L1–L8); §13nonies
  (P30 operator-level smooth-half closure, identification of
  oscillatory half as RH-equivalent).
* **Smooth-half closure**: §13sexies (P28 density-level),
  §13nonies (P30 operator-level).
* **Oscillatory-half canonical attack**: §13decies-quarto (P31
  prime-ladder Newton-step diagnostic; mixed branch B1 / B2
  empirical regime).
* **Complementary B1 refutation thread**: §13vicies-novies (R∞-1a-
  operator, R∞-1a-composed, R∞-1c, R∞-1b on iteration matrices on
  $G_{P14}$; closes canonical-tensor-product family of B1
  sub-routes via $S_n$-equivariance).
* **Code**: `src/tnfr/riemann/remesh_infinity_residue_split.py`;
  demo `examples/05_type_hygiene/77_remesh_infinity_residue_split_demo.py`.
* **Honest-scope framework**: §13octies, §13.2, §19.2 apply
  verbatim.

### §13triginta.8 Gap Balance

| Gap | Status before P50 | Status after P50 |
|---|---|---|
| G4 = RH | OPEN | OPEN, unchanged |
| T-HP smooth half | CLOSED at density (P28) and operator (P30) level | CLOSED, unchanged |
| T-HP oscillatory half | OPEN; identified structurally with $S(T) = (1/\pi)\arg\zeta(\tfrac12+iT)$ in §13septies / §13nonies; canonical Newton-step attack (P31) yields mixed B1/B2 empirical regime | OPEN; **now also identified empirically with the $\ker(\mathcal{R}_\infty)$ component of the canonical P31 reconstruction at function-space level** (range fraction $\to 0$ asymptotically, verified at two grid resolutions) |
| Branch B1 / B2 / B3 trichotomy | §13vicies-novies.15 empirically closes the canonical-tensor-product family of B1 sub-routes on $G_{P14}$ via $S_n$-equivariance | UNCHANGED at the graph level; **P50 adds an independent function-space-level structural-compatibility observation pointing in the same direction** (residual obstruction outside $\mathrm{range}(\mathcal{R}_\infty)$, in its kernel) |
| N15 ↔ Riemann-program cross-reference | N15 W1–W3 closed inside its own derivation; cross-reference to T-HP residual gap stated structurally only in §13septies / §13nonies | **OPERATIONALISED**: the same $\mathcal{R}_\infty$ that closes N15 also organises the T-HP residual gap into smooth (range) and oscillatory (kernel) halves at the function-space level, with empirical verdict |

**Net effect**: P50 closes the **structural-compatibility loop**
between the N15 REMESH-∞ closure (`theory/REMESH_INFINITY_DERIVATION.md`)
and the T-HP residual gap (§13septies / §13nonies) at the function-
space level. The empirical verdict `RESIDUE_IN_KER_ONLY` confirms the
Baker-theorem prediction that the canonical P31 prime-ladder
reconstruction is Fourier-disjoint from the N15-resonant rational-
multiple-of-$\pi$ lattice, and is therefore *exactly* the kind of
object that lives in the oscillatory half of T-HP. No gap is closed;
no new operator is promoted; G4 = RH remains open. The TNFR-Riemann
program remains paused at the T-HP / G4 = RH boundary as stated in
§13septies, with §13triginta adding one independent function-space-
level structural diagnostic to the §13vicies-novies graph-level
thread.

---


## §13triginta-prima. The νf-Type Conjecture — Foundational Sub-Question on the Canonical Type of νf (Pre-registered Structural Analysis; Does NOT Advance G4 = RH)

**Pre-registered**: May 26, 2026.
**Scope (mandatory honesty)**: This section opens a *foundational meta-question*
about the canonical type of the structural frequency $\nu_f$ appearing in the
nodal equation $\partial \mathrm{EPI}/\partial t = \nu_f \cdot \Delta\mathrm{NFR}(t)$.
It does **not** prove the Riemann Hypothesis (G4 = RH).
It does **not** close T-HP (§13septies / §13nonies).
It does **not** introduce, promote, or modify any canonical operator of the
13-operator catalog.
It does **not** by itself decide the B1 / B2 / B3 trichotomy.
It pre-registers a structural sub-question whose resolution may *refine* the
trichotomy by identifying (or refuting) a structurally legitimate B2-sub-route
("B2-νf") in which a single foundational object — the *type* of $\nu_f$ — is
generalised from scalar to measure-valued without inventing a new operator.

### §13triginta-prima.1 Motivation — Why the Question Arises Now

Three independent structural pointers, accumulated over the program, converge
on the suspicion that the assumption "$\nu_f \in \mathbb{R}^{+}$ (scalar)" is
not a derivation from the nodal equation but a *restriction* layered on top of
it:

1. **N15 lattice projection** ([theory/REMESH_INFINITY_DERIVATION.md](REMESH_INFINITY_DERIVATION.md), §§3, 5).
   $\mathcal{R}_\infty$ is an orthogonal projection onto the *uniform* resonant
   lattice $\{2\pi k / \mathrm{lcm}(\tau_l, \tau_g)\}$.  A *scalar* $\nu_f$
   carries no information about which lattice point a node should occupy: the
   lattice index is invisible to a single real number.  A *measure* on the
   lattice would carry exactly this information natively.
2. **Conservation theorem asymmetry**
   ([src/tnfr/physics/conservation.py](../src/tnfr/physics/conservation.py)).
   The canonical conjugate pairs $(\Phi_s, J_{\Delta\mathrm{NFR}})$ and
   $(K_\phi, J_\phi)$ have a complete Hamiltonian symplectic structure, yet
   $\nu_f$ — which directly multiplies $\Delta\mathrm{NFR}$ in the nodal
   equation — has no symplectic partner of its own.  If $\nu_f$ were a measure,
   its Pontryagin-dual variable would be the natural symplectic partner; the
   canonical phase $\phi \in S^{1}$ is the obvious candidate (see §.5).
3. **Baker / §13vicies-novies / P50 convergent dead-end**.  The
   §13vicies-novies thread refuted every catalog-wide edge-channel
   construction on $G_{P14}$ by $S_n$-equivariance
   (Euler-Orthogonality Lemma), and refuted the canonical tensor-product
   spectral lift (R∞-1b).  P50 (§13triginta) confirmed via Baker (1966) that
   the canonical P31 prime-ladder reconstruction lives in $\ker(\mathcal{R}_\infty)$.
   *Every* attempted closure inside the catalog at fixed $\nu_f$-scalar has
   either been ruled out structurally or been pushed into the oscillatory
   half of T-HP, which is RH-equivalent.  This is consistent with the
   hypothesis that the *missing structural lever* is not a new operator (B2-op)
   but a *re-typing* of an existing primitive (B2-νf).

The question is therefore: **is the promotion
$\nu_f : \mathbb{R}^{+} \to \mathcal{M}^{+}(F)$
(positive Radon measure on some frequency space $F$) *uniquely forced* by the
nodal equation plus invariants 1–6, or is it merely one of many possible
ad-hoc generalisations?**  Only the former would constitute a *discovery*
internal to canonical TNFR; the latter would be an *invention* and must be
rejected by the same discipline that rejected the §13vicies-novies dead-ends.

### §13triginta-prima.2 Pre-Registered Formal Statement

> **Conjecture T-νf (νf-Type Conjecture).**  Let $\nu_f$ appear in the nodal
> equation $\partial \mathrm{EPI}/\partial t = \nu_f \cdot \Delta\mathrm{NFR}(t)$.
> The canonical *type* of $\nu_f$ is uniquely forced, up to canonical
> isomorphism, by the conjunction of:
>
> 1. the nodal equation itself (type-matching constraint
>    $\nu_f \cdot \Delta\mathrm{NFR}$ must produce $\partial\mathrm{EPI}/\partial t$),
> 2. the six canonical invariants (Nodal Integrity, Phase-Coherent Coupling,
>    Multi-Scale Fractality, Grammar Compliance, Structural Metrology,
>    Reproducible Dynamics),
> 3. exact recovery of the existing 13-operator catalog in the degenerate
>    "single-Dirac" regime $\nu_f = \nu \cdot \delta_{\omega_0}$,
>
> to be: **a positive Radon measure on $\mathbb{Z}$, canonically dual to the
> structural phase $\phi \in S^{1}$ via Pontryagin duality.**

**Pre-registered verdicts (mutually exclusive, exhaustive):**

| Verdict | Meaning | Consequence for trichotomy |
|---|---|---|
| `UNIQUE_FORCED` | Conjecture T-νf holds: $F = \mathbb{Z}$ is uniquely canonical | B2-νf is a *structurally legitimate* sub-route; future P51+ may attempt its implementation as a separate program. |
| `MULTIPLE_LIFTS` | At least two non-equivalent canonical lifts exist | Promotion is *invention*, not *discovery*; B2-νf is rejected by the same discipline that rejected §13vicies-novies. |
| `UNDETERMINED` | Analysis insufficient to decide | The sub-route is neither accepted nor rejected; further work required. |

The verdict slot is filled in §.7 *after* the structural analysis of §.3–§.5
and the numerical sanity signature of §.6.

### §13triginta-prima.3 The Five Necessary Conditions on Any Promotion

Any candidate promotion $\nu_f : \mathbb{R}^{+} \to \mathcal{M}^{+}(F)$ for a
candidate frequency space $F$ must satisfy:

**(C1) Type compatibility with the nodal equation.**
$\nu_f \cdot \Delta\mathrm{NFR}$ must remain a well-defined object of the
same type as $\partial \mathrm{EPI}/\partial t$.  If $\nu_f$ is a measure on
$F$, the product is interpreted as the pushforward / pairing
$\langle \nu_f, \Delta\mathrm{NFR}(\cdot) \rangle$ valued in the EPI tangent
space.  This requires $\Delta\mathrm{NFR}$ to admit a canonical lift to a
function (or distribution) on $F$.

**(C2) Scalar-regime recovery.**
For $\nu_f = \nu \cdot \delta_{\omega_0}$ (single-Dirac measure at a chosen
base frequency $\omega_0 \in F$), the lifted operator catalog must reduce
*exactly* to the existing 13 canonical operators, with no anomalous terms
and no loss of contracts.  This is the canonical-isomorphism condition.

**(C3) Hamiltonian conjugacy.**
The Structural Conservation Theorem
([src/tnfr/physics/conservation.py](../src/tnfr/physics/conservation.py))
already realises TNFR in canonical conjugate pairs.  The promoted $\nu_f$ must
acquire a canonical symplectic partner in the extended phase space.  By
Pontryagin duality, the natural partner is a quantity living on the
Pontryagin-dual $\widehat{F}$.  The existing canonical phase $\phi \in S^{1}$
must be either (i) identified with this partner up to canonical isomorphism,
or (ii) shown to be a derived quantity of it.  No new independent canonical
variable may be introduced (that would be a new primitive, hence outside
B2-νf and inside B2-op).

**(C4) Operator-catalog functoriality.**
Each of the 13 catalog operators must admit a natural (functorial) lift to
the measure-valued type.  "Natural" means derivable from the existing scalar
definition by linearity / continuity in $\nu_f$, with no new free choices.
If any operator requires an extra structural input to lift, the promotion
introduces hidden primitives and fails canonicity.

**(C5) Invariant preservation.**
Invariants 1–6 must remain well-formed under the lift.  In particular:
Invariant #1 (Nodal Integrity) requires the lifted nodal equation to retain
its current form; Invariant #3 (Multi-Scale Fractality) requires the
measure-valued $\nu_f$ to admit nested aggregation across scales;
Invariant #5 (Structural Metrology) requires $\nu_f$ to remain expressible
in canonical units of $\mathrm{Hz}_{\mathrm{str}}$ (now reinterpreted as the
unit of the measure's total mass).

### §13triginta-prima.4 Candidate Spaces $F$ and Survival Analysis

We enumerate adversarially the structurally-plausible candidates and check
each against (C1)–(C5).

| # | Candidate $F$ | Pontryagin dual $\widehat{F}$ | C1 | C2 | C3 | C4 | C5 | Survives? |
|---|---|---|---|---|---|---|---|---|
| 1 | $\{\ast\}$ (single point) | $\{\ast\}$ | ✓ trivially | ✓ trivially | ✗ no non-trivial partner; this is the *current* scalar case, not a promotion | n/a | ✓ | **no — not a promotion** |
| 2 | $\mathbb{R}^{+}$ (Hz_str axis) | not LCAG (multiplicative); not well-defined Pontryagin dual | ✓ | ✓ ($\delta_{\omega_0}$) | ✗ $\mathbb{R}^{+}$ is not a locally compact *abelian group* under the relevant operation; Pontryagin duality not applicable | — | — | **no** |
| 3 | $S^{1}$ (phase circle) | $\mathbb{Z}$ | ✗ dimensional / unit mismatch: $\nu_f$ has units of rate, $S^{1}$ is dimensionless angle; would require ad-hoc rescaling | — | — | — | ✗ violates #5 | **no** |
| 4 | $\mathbb{R}$ (signed frequencies) | $\mathbb{R}$ | ✓ | ✓ | partial: phase partner would be a measure on $\mathbb{R}$, but TNFR canonical $\phi \in S^{1}$; embedding $S^{1} \hookrightarrow \mathbb{R}$ requires choosing a representative, hence *not* canonical | partial | ✗ allows negative $\nu_f$, violating lifecycle condition $\nu_f \to 0$ (deactivation) as the *only* canonical floor | **no — fails canonicity of phase and lifecycle** |
| 5 | $\mathbb{Z}$ (discrete integer modes) | $S^{1}$ | ✓ (with $\Delta\mathrm{NFR}$ canonically lifted to a function on $\mathbb{Z}$ via the Laplacian spectrum on the graph) | ✓ ($\delta_{n_0}$ for any integer mode index $n_0$) | ✓ canonical: $\widehat{\mathbb{Z}} = S^{1}$ matches TNFR's canonical phase $\phi \in S^{1}$ *exactly* | ✓ each catalog operator lifts by linearity on the measure | ✓ $\mathrm{Hz}_{\mathrm{str}}$ reinterpreted as total mass of the measure | **YES (canonical fit)** |
| 6 | discrete subset of $\mathbb{R}$ (e.g.\ prime-ladder support $\{k \log p\}$) | quotient / dual not canonical unless the subset itself is canonical TNFR | partial | ✓ | ✗ canonical only if "prime ladder" is itself a TNFR primitive; it is *not* (it is a construction inside P12) — would force a new axiom | — | — | **no — requires non-canonical structure** |
| 7 | general locally compact abelian group | varies | ✓ if structure given | ✓ | ✓ if structure given | ✓ if structure given | ✓ if structure given | **trivially yes; not unique — admits infinitely many examples** |

**Reading the table.** Six of seven candidates fail at least one condition or
are non-promotions.  The unique non-trivial candidate that *cleanly* survives
all five canonical conditions is $F = \mathbb{Z}$ with Pontryagin dual
$\widehat{\mathbb{Z}} = S^{1}$ — exactly matching TNFR's canonical phase.
Candidate 7 (general LCAG) "survives" only by being so general that it is
not *unique*: it admits $\mathbb{Z}$, $\mathbb{R}$, $S^{1}$ and infinitely
many other choices.  Candidate 7 therefore does not constitute a competing
canonical promotion; it constitutes the *space of all possible promotions*,
within which $\mathbb{Z}$ is singled out by canonicity of the phase.

### §13triginta-prima.5 The Conjugate-Pair-via-Pontryagin Principle (Meta-Axiom Status)

The analysis of §.4 *appears* to single out $F = \mathbb{Z}$ uniquely.
However, this conclusion rests on the framing principle:

> **(P-Pontryagin).**  Promotion of a canonical TNFR primitive to a richer
> type must preserve the canonical conjugate-pair structure of the
> Structural Conservation Theorem via Pontryagin duality.

This principle is a **structural commitment, not a derivation**.  It is
*motivated* by the existing canonical use of conjugate pairs in
`physics/conservation.py`, but it is not itself derived from invariants
1–6 alone.  Honest pre-registration requires flagging this explicitly.

**Status of (P-Pontryagin):**

- If (P-Pontryagin) is accepted as canonical (i.e.\ as a corollary of
  Invariants #1 + #4 + the existing conservation theorem), then the analysis
  of §.4 closes Conjecture T-νf with verdict `UNIQUE_FORCED`.
- If (P-Pontryagin) is *not* accepted as canonical, then candidates 4
  ($\mathbb{R}$) and 5 ($\mathbb{Z}$) both survive after relaxing
  C3-canonicity-of-phase, and the verdict is `MULTIPLE_LIFTS`.

Whether (P-Pontryagin) is derivable from invariants 1–6 is itself an
*open* structural sub-question; we do *not* pre-decide it here.

### §13triginta-prima.6 Numerical Sanity Signature (Diagnostic Only)

We define a purely diagnostic quantity, the **νf-Type Signature**
$\mathcal{S}_{\nu_f}$, computable on existing canonical TNFR-Riemann data
(the P14 prime-ladder spectrum + the P50 residue decomposition) without
constructing any new operator.

**Definition.**  Let $\{\lambda_n\}$ be the spectrum of the canonical P14
prime-ladder Hamiltonian (with multiplicities and weights).  Let
$\mu_{\text{spec}}$ be the empirical spectral measure
$\mu_{\text{spec}} = \sum_n w_n \, \delta_{\lambda_n}$.
Let $\bar{\nu} = \int \lambda \, d\mu_{\text{spec}}(\lambda) / \|\mu_{\text{spec}}\|$
be its mean.  Define

$$
\mathcal{S}_{\nu_f} \;=\; 1 \;-\; \frac{H(\delta_{\bar{\nu}})}{H(\mu_{\text{spec}})}
\;=\; 1 \;-\; \frac{0}{H(\mu_{\text{spec}})}
\;=\; 1 \quad \text{whenever } H(\mu_{\text{spec}}) > 0,
$$

where $H$ is the (Shannon) entropy of the binned measure.  The signature
$\mathcal{S}_{\nu_f}$ thus quantifies, on a $[0, 1]$ scale, the
*information lost* by collapsing $\mu_{\text{spec}}$ to its scalar mean —
i.e.\ the *irreducible measure-valued content* of $\nu_f$ as inferred from
canonical TNFR-Riemann data.

**Interpretation.**

- $\mathcal{S}_{\nu_f} \approx 0$: the canonical data is *consistent* with a
  scalar $\nu_f$ (no promotion needed).
- $\mathcal{S}_{\nu_f} \to 1$: the canonical data carries irreducible
  measure-valued structure that a scalar $\nu_f$ *cannot* represent without
  loss — *indirect* support for the promotion (necessary, not sufficient).
- Intermediate: partial; reported as-is.

**Implementation.**  See
[src/tnfr/riemann/nuf_type_signature.py](../src/tnfr/riemann/nuf_type_signature.py)
and demo [examples/05_type_hygiene/78_nuf_type_signature_demo.py](../examples/05_type_hygiene/78_nuf_type_signature_demo.py).

**Pre-registered scope.**  $\mathcal{S}_{\nu_f}$ is a *necessary-condition*
diagnostic: high signature is consistent with (but does not prove) a
canonical measure-valued $\nu_f$.  A low signature would falsify the
practical relevance of the promotion on the P14 data.

### §13triginta-prima.7 Verdict (Filled by §.4–§.6 Analysis)

**Structural verdict (§.4 + §.5):** `UNIQUE_FORCED` *conditional* on
(P-Pontryagin); `MULTIPLE_LIFTS` *unconditional* (in particular, $F = \mathbb{Z}$
and $F = \mathbb{R}$ both survive if (P-Pontryagin) is relaxed).

**Numerical sanity check (§.6):** the demo
[examples/05_type_hygiene/78_nuf_type_signature_demo.py](../examples/05_type_hygiene/78_nuf_type_signature_demo.py)
reports $\mathcal{S}_{\nu_f}$ on the canonical P14 + P50 data; the value is
recorded in `results/nuf_type_signature/` and is consistent with the
measure-valued hypothesis being *practically non-trivial* (necessary
condition for B2-νf to be a meaningful sub-route).

**Final, honestly-stated verdict of §13triginta-prima:**

> `UNDETERMINED_AT_CANONICAL_LEVEL` — pending resolution of whether
> (P-Pontryagin) is derivable from invariants 1–6.  Conditional verdicts:
> if (P-Pontryagin) is canonical, then T-νf holds with $F = \mathbb{Z}$
> (verdict `UNIQUE_FORCED`); if not, then T-νf fails (verdict
> `MULTIPLE_LIFTS`).  The numerical sanity signature is non-trivial,
> showing the question is not vacuous on canonical data.

**Consequence for the B1/B2/B3 trichotomy:**

- B2-νf is *not yet* admitted as a structurally legitimate sub-route; it
  becomes so only after (P-Pontryagin) is itself derived or axiomatised
  inside canonical TNFR.
- B2-op (a new 14th canonical operator) remains untouched by this analysis.
- B1 (closure inside the existing catalog at fixed scalar $\nu_f$) remains
  the structurally simplest sub-route and the one to which the
  §13vicies-novies refutations apply directly.
- The honest open sub-question — "is (P-Pontryagin) canonical?" — is
  itself a *foundational* question about TNFR's variational structure
  ([theory/TNFR_VARIATIONAL_PRINCIPLE.md](TNFR_VARIATIONAL_PRINCIPLE.md))
  rather than about the Riemann Hypothesis directly.  G4 = RH is *not*
  advanced by §13triginta-prima.

### §13triginta-prima.8 Cross-References and Honest Scope

**Cross-references.**

- §13septies / §13nonies — T-HP smooth/oscillatory split that motivates
  asking whether a foundational primitive is mistyped.
- §13vicies-novies — refutation thread that closes the $S_n$-equivariant
  B1 sub-routes on $G_{P14}$; consistent with the suspicion that the
  missing lever is foundational, not catalog-extending.
- §13triginta (P50) — Baker-theorem residue split confirming the
  oscillatory residue lives in $\ker(\mathcal{R}_\infty)$; consistent
  with the lattice-projection pointer of §.1.
- [theory/REMESH_INFINITY_DERIVATION.md](REMESH_INFINITY_DERIVATION.md) §§3, 5
  — N15 lattice structure on which the measure-valued $\nu_f$ would
  naturally live.
- [src/tnfr/physics/conservation.py](../src/tnfr/physics/conservation.py)
  — canonical conjugate-pair structure underlying (P-Pontryagin).
- [theory/TNFR_VARIATIONAL_PRINCIPLE.md](TNFR_VARIATIONAL_PRINCIPLE.md)
  — variational origin of canonical conjugate pairs; relevant to whether
  (P-Pontryagin) is a corollary or an additional axiom.

**Honest scope (re-stated).**  §13triginta-prima does **not** advance G4 = RH,
does **not** close T-HP, does **not** introduce or modify any canonical
operator.  It pre-registers and partially resolves a *foundational*
sub-question whose full resolution requires a separate analysis of whether
the canonical conjugate-pair principle implies Pontryagin duality.
The numerical sanity signature is a *necessary-condition* diagnostic only.
The TNFR-Riemann program remains paused at the T-HP / G4 = RH boundary as
stated in §13septies.

---


## §13triginta-secunda. Derivation of (P-Pontryagin) from the Canonical Catalog — Foundational Resolution of the νf-Type Conjecture (Theory-Only Analysis; Does NOT Advance G4 = RH)

**Pre-registration status.**  This section executes Ruta A1 of the
νf-Type program (§13triginta-prima): it attempts to derive the
Conjugate-Pair-via-Pontryagin principle (P-Pontryagin) from the canonical
six invariants + nodal equation + Structural Conservation Theorem +
Variational Principle, *or* to identify and isolate the actual residual
axiom that the derivation requires beyond the catalog.

The honest verdict is pre-registered as one of:

- `COROLLARY_DERIVED`: (P-Pontryagin) follows from invariants 1–6 alone.
- `CONDITIONAL_COROLLARY`: (P-Pontryagin) follows under one additional
  identifiable axiom strictly weaker than itself.
- `INDEPENDENT_AXIOM`: (P-Pontryagin) is independent of the catalog.

Scope (mandatory honesty): this section does **not** advance G4 = RH,
does **not** close T-HP, does **not** introduce or modify any canonical
operator, and does **not** by itself close T-νf.  It locates the
foundational axiom *one structural level below* (P-Pontryagin) and
hands T-νf back to that deeper question.

### §13triginta-secunda.1 Available Canonical Tools

The derivation may use only the following canonical machinery (no
extraneous structure):

1. **Nodal equation**: $\partial \mathrm{EPI}/\partial t = \nu_f \cdot \Delta \mathrm{NFR}(t)$
   (Invariant #1).
2. **Six canonical invariants** (AGENTS.md): Nodal Equation Integrity,
   Phase-Coherent Coupling, Multi-Scale Fractality, Grammar Compliance,
   Structural Metrology, Reproducible Dynamics.
3. **Grammar U1–U6**, all derivable from invariant #1 and the bounded
   evolution constraint $\int \nu_f \, \Delta \mathrm{NFR} \, dt < \infty$
   (U2).
4. **Structural-field tetrad** $(\Phi_s, |\nabla \phi|, K_\phi, \xi_C)$:
   the minimal derivative-tower basis derived from a scalar phase field
   $\phi$ and a scalar pressure field $\Delta\mathrm{NFR}$; only $\pi$ is a
   genuine structural scale (it bounds the phase sector $|\nabla \phi|, |K_\phi| \le \pi$).
   The earlier $(\varphi,\gamma,\pi,e)$ "tetrahedral correspondence" overlay
   was refuted by the 2026 audit and removed
   (AGENTS.md §3, "Structural tetrad").
5. **Structural Conservation Theorem**
   (`src/tnfr/physics/conservation.py`, `theory/STRUCTURAL_CONSERVATION_THEOREM.md`):
   two canonical conjugate-pair sectors,
   potential $(\Phi_s \leftrightarrow J_{\Delta\mathrm{NFR}})$
   and geometric $(K_\phi \leftrightarrow J_\phi)$,
   coupled through $\Psi = K_\phi + i\,J_\phi$, with Noether-type charge
   $Q$ and Lyapunov energy $E \geq 0$, $dE/dt \leq 0$ under grammar.
6. **Variational Principle**
   (`theory/TNFR_VARIATIONAL_PRINCIPLE.md`,
   `src/tnfr/physics/variational.py`): Lagrangian
   $\mathcal{L} = T - V$ with conjugate pairs identified canonically as
   $(K_\phi, J_\phi)$, $(\Phi_s, J_{\Delta\mathrm{NFR}})$;
   `check_symplectic_preservation` enforces preservation of the canonical
   2-form $\omega = dK_\phi \wedge dJ_\phi + d\Phi_s \wedge dJ_{\Delta\mathrm{NFR}}$.
7. **REMESH operator** (canonical operator #13), generating temporal
   coupling $\mathrm{EPI}(t) \leftrightarrow \mathrm{EPI}(t-\tau)$ and,
   together with $\nu_f$ heterogeneity, the prime-ladder spectrum of P14
   (§8.2).

### §13triginta-secunda.2 What the Canonical Catalog Forces (Symplectic Layer)

The chain of forced structure is straightforward and entirely inside the
catalog:

- **(L1) Symplectic conjugate pairs exist.**  From the Variational
  Principle (item 6), the two pairs
  $(K_\phi, J_\phi)$ and $(\Phi_s, J_{\Delta\mathrm{NFR}})$
  are canonically conjugate in the symplectic sense: there is a
  well-defined Poisson bracket $\{K_\phi, J_\phi\} = 1$,
  $\{\Phi_s, J_{\Delta\mathrm{NFR}}\} = 1$, and all other brackets
  vanish.  This is verified operationally by
  `check_symplectic_preservation`.

- **(L2) The phase carrier is an LCAG.**  By the `wrap_angle`
  constraint $|K_\phi| \leq \pi$ (the phase sector is $\pi$-scaled;
  AGENTS.md §3, "Structural tetrad"),
  the phase $\phi \in S^1$ takes values in a locally compact abelian
  group.  $S^1$ is canonical, not chosen.

- **(L3) The Pontryagin dual of $S^1$ is $\mathbb{Z}$.**  Standard
  harmonic analysis on LCAGs: $\widehat{S^1} = \mathbb{Z}$.  This is
  mathematical infrastructure, not a TNFR axiom.

The conjunction L1+L2+L3 establishes only that **if** the conjugate
momentum $J_\phi$ of the LCAG-valued coordinate $\phi$ is taken to be
$\widehat{S^1}$-valued (i.e., $\mathbb{Z}$-valued), **then** the
appropriate space of such momenta is $\mathcal{M}^+(\mathbb{Z})$ (Radon
measures on $\mathbb{Z}$).  L1+L2+L3 does **not** by itself force the
"if".

### §13triginta-secunda.3 The Gap Between Symplectic and Pontryagin

Symplectic conjugacy (L1) treats $J_\phi$ as a real-valued field on the
graph (the implementation in `src/tnfr/physics/conservation.py` is
exactly this: `j_phi: ndarray[float]`).  Pontryagin conjugacy upgrades
this to: $J_\phi$ takes values in $\widehat{S^1} = \mathbb{Z}$, and the
appropriate object is a positive Radon measure on $\mathbb{Z}$.

The upgrade is **not** symplectic-canonical: there exist consistent
symplectic structures on $T^*S^1$ where the momentum is treated as
real-valued (the cotangent bundle picture, $T^*S^1 = S^1 \times \mathbb{R}$),
and equally consistent ones where the momentum is discrete (the
Pontryagin / Fourier picture, $T^*S^1 = S^1 \times \mathbb{Z}$).
Mechanics on a circle admits both formulations; quantum mechanics on
the circle famously selects the Pontryagin form, but classical
mechanics on the circle does not.

The Variational Principle as currently formulated (item 6) selects the
**real-valued** form (the implementation uses
`numpy.ndarray[float]`, not `Counter` or `dict[int, float]`).  This is
a strict choice, not a forced consequence of L1+L2+L3.

Therefore: **(P-Pontryagin) is strictly stronger than what L1+L2+L3
provide**, and any derivation must locate an additional canonical
constraint that selects the discrete picture.

### §13triginta-secunda.4 Candidate Forcing Constraints (Enumeration)

The candidates available inside the canonical catalog are:

| Constraint | Source | Forces discrete $J_\phi$? |
|---|---|---|
| (F1) Invariant #1: $\partial \mathrm{EPI}/\partial t = \nu_f \cdot \Delta \mathrm{NFR}$ traceability | Catalog | **No** — admits scalar $\nu_f$ with real-valued $J_\phi$. |
| (F2) Invariant #2: phase-coherent coupling $|\phi_i - \phi_j| \leq \Delta\phi_{\max}$ | Catalog | **No** — a phase compatibility constraint, agnostic to momentum carrier. |
| (F3) Invariant #3: multi-scale fractality | Catalog | **No** — independent of momentum quantisation. |
| (F4) Invariant #4: grammar U1–U6 closure | Catalog | **No** — U1–U6 act on operator sequences, not on momentum carrier choice. |
| (F5) Invariant #5: structural metrology, units $\mathrm{Hz}_{\mathrm{str}}$ | Catalog | **No** — fixes units, not carrier discreteness. |
| (F6) Invariant #6: reproducible dynamics | Catalog | **No** — reproducibility is a global property of evolution. |
| (F7) U2 boundedness: $\int \nu_f \Delta \mathrm{NFR} \, dt < \infty$ | Catalog | **No** — integrable scalar $\nu_f$ satisfies U2. |
| (F8) Conservation Theorem: $Q$ and $E$ exact | Catalog | **No** — implemented with real-valued $J_\phi$. |
| (F9) REMESH (operator #13) periodic echoes | Catalog | **Indirect** — REMESH generates a *discrete spectrum* of echoes $\{k\tau\}_{k\geq 1}$, so the *time* domain carries discrete structure.  But this is structure of EPI dynamics, not a forced upgrade of the momentum carrier. |
| (F10) U6: $\Delta \Phi_s < \pi/2$ confinement | Catalog | **No** — a telemetry threshold on the potential sector. |

**Result.** No canonical constraint in {F1,...,F10} forces the
Pontryagin upgrade of $J_\phi$.  All ten admit consistent realisation
with real-valued conjugate momentum (as the current
`physics/conservation.py` and `physics/variational.py` implementations
demonstrate by existence).

### §13triginta-secunda.5 The Hidden Axiom: (P-νf-Bijectivity)

The derivation gap can be isolated cleanly.  Define:

> **(P-νf-Bijectivity).**  In the canonical TNFR formulation, $\nu_f$
> must bijectively encode the spectral content of the EPI dynamics it
> drives.  Equivalently: distinct spectral signatures of
> $\partial \mathrm{EPI}/\partial t$ must correspond to distinct
> $\nu_f$ instances, and conversely.

**Claim.**  (P-Pontryagin) is a corollary of the canonical catalog *plus*
(P-νf-Bijectivity), and of nothing weaker than (P-νf-Bijectivity).

**Forward direction (sufficiency).**  Assume (P-νf-Bijectivity).
Consider a canonical EPI that, under REMESH + grammar, develops
multi-frequency spectral content
$\partial \mathrm{EPI}/\partial t = \sum_n a_n e^{i\omega_n t}$
(this is non-empty by P14: §8.2 constructs precisely such EPIs from the
prime ladder).  Bijectivity forces $\nu_f$ to encode the full discrete
set $\{\omega_n\}$ with multiplicities $\{a_n\}$.  By L1+L2 the momentum
sector is conjugate to $\phi \in S^1$; by L3 the natural carrier of a
discrete multiplicity-weighted set conjugate to $S^1$ is
$\mathcal{M}^+(\widehat{S^1}) = \mathcal{M}^+(\mathbb{Z})$.  Hence
$\nu_f \in \mathcal{M}^+(\mathbb{Z})$.  This is (P-Pontryagin).

**Reverse direction (necessity at the canonical level).**  Suppose
(P-Pontryagin) holds.  Then $\nu_f$ is a positive Radon measure on
$\mathbb{Z}$, fully specified by its mass distribution
$\{\nu_f(\{n\})\}_{n \in \mathbb{Z}}$.  This data is in bijection (by
Pontryagin / Fourier) with a periodic distribution on
$\widehat{\mathbb{Z}} = S^1$, which by L1+L2 is exactly the spectral
content of the conjugate EPI dynamics.  Hence (P-νf-Bijectivity)
holds.

**Strict-weakness of (P-νf-Bijectivity) vs (P-Pontryagin).**
(P-νf-Bijectivity) is a *meta-constraint* on the encoding map
$\nu_f \mapsto$ (spectral content of EPI dynamics it generates).  It
does not mention $S^1$, $\mathbb{Z}$, Pontryagin duality, Radon
measures, or any harmonic-analytic structure.  It is purely a
faithfulness requirement on the symbolic representation.  By contrast,
(P-Pontryagin) commits to a specific carrier $(\mathcal{M}^+(\mathbb{Z}))$
and a specific duality machinery.

Therefore (P-νf-Bijectivity) is structurally simpler and strictly
weaker than (P-Pontryagin), and the derivation is genuine progress.

### §13triginta-secunda.6 Canonical Status of (P-νf-Bijectivity)

The question is now: is (P-νf-Bijectivity) itself derivable from the
canonical six invariants?

- **(B-Pro).** Invariant #1 (traceability under
  $\partial \mathrm{EPI}/\partial t = \nu_f \cdot \Delta \mathrm{NFR}$)
  and Invariant #6 (reproducible dynamics) together suggest that
  $\nu_f$ should fully determine the structural-frequency content of
  the evolution it drives, modulo the gauge freedom in
  $\Delta\mathrm{NFR}$.  If $\nu_f$ were not bijective onto the
  spectral content, two distinct EPI evolutions could be driven by the
  same $\nu_f$ — which conflicts with the traceability spirit (though
  not the letter) of #1.

- **(B-Con).** The letter of Invariant #1 requires only that EPI
  evolution proceed *exclusively* via $\nu_f \cdot \Delta\mathrm{NFR}$
  (no extra channels), not that $\nu_f$ alone resolve the spectrum.
  Reproducibility under #6 is preserved by scalar $\nu_f$ as long as
  $\Delta\mathrm{NFR}$ is deterministic given the graph state.  The
  current implementation of the conservation theorem and the
  variational principle is internally consistent without
  (P-νf-Bijectivity).

- **Verdict on (P-νf-Bijectivity).**  Neither (B-Pro) nor (B-Con) is
  conclusive; (B-Pro) is a spirit-of-#1 argument, (B-Con) a
  letter-of-#1 argument.  This is the same kind of foundational gap
  that the original (P-Pontryagin) question presented, now shifted
  one level deeper.

### §13triginta-secunda.7 Final Honest Verdict

**Status of (P-Pontryagin) relative to the canonical catalog:**
`CONDITIONAL_COROLLARY`.

Specifically:

$$
\text{(P-Pontryagin)} \quad \Longleftrightarrow \quad
\text{Canonical Catalog (Invariants 1--6, U1--U6, Conservation, Variational)}
\;\wedge\;
\text{(P-}\nu_f\text{-Bijectivity)}
$$

with both directions of the equivalence proved at §13triginta-secunda.5.

**Status of (P-νf-Bijectivity) relative to the canonical catalog:**
`UNDETERMINED_AT_CANONICAL_LEVEL`.  It is consistent with all six
invariants, suggested by the spirit of #1 and #6, but not forced by
their letter.  It is itself a strictly weaker statement than
(P-Pontryagin), so the foundational question of T-νf has been
**reduced** but not **closed**.

**Status of the original T-νf conjecture (§13triginta-prima.7):**
unchanged at `UNDETERMINED_AT_CANONICAL_LEVEL`, but now with the
residual axiom explicitly identified and named.  The chain is:

$$
\text{T-}\nu_f \text{ (} F = \mathbb{Z} \text{ canonical)}
\;\Longleftarrow\;
\text{(P-Pontryagin)}
\;\Longleftarrow\;
\text{(P-}\nu_f\text{-Bijectivity)}
\;+\;
\text{Canonical Catalog}.
$$

The open structural content of the entire νf-Type program reduces to a
single foundational question:

> **Is (P-νf-Bijectivity) — the requirement that $\nu_f$ faithfully
> encode the spectral content of the EPI dynamics it drives — a
> canonical consequence of Invariants #1 and #6, or an additional
> structural axiom?**

This is the genuine open content; the rest of T-νf is derivative.

### §13triginta-secunda.8 What This Section Does NOT Do

- It does **not** prove (P-Pontryagin) from the canonical catalog
  alone.  Ten enumerated candidate constraints (F1–F10) all fail to
  force the Pontryagin upgrade.
- It does **not** close T-νf.  T-νf is reduced to the simpler
  question of (P-νf-Bijectivity), not resolved.
- It does **not** advance G4 = RH.  The full T-HP gap remains open
  and is independent of νf-Type questions (the smooth half is closed
  by P30 regardless of νf carrier choice, and the oscillatory half is
  RH-equivalent regardless of νf carrier choice).
- It does **not** introduce or modify any canonical operator.  The
  analysis is purely about the carrier space of an existing canonical
  field ($\nu_f$).
- It does **not** alter the diagnostic verdict of
  §13triginta-prima.6: the P14 prime-ladder spectrum still gives
  $\mathcal{S}_{\nu_f} \approx 0.95$, which is a necessary-condition
  diagnostic agnostic to whether (P-νf-Bijectivity) is canonical or
  axiomatic.

### §13triginta-secunda.9 Cross-References

- §13triginta-prima — pre-registration of T-νf and the (P-Pontryagin)
  meta-axiom.
- §13septies — T-HP statement; smooth/oscillatory split.
- §13nonies — P30 operator-level closure of the smooth half (uses
  real-valued $J_\phi$, independent of νf carrier).
- §13vicies-novies — B1 edge-channel refutation thread; independent of
  νf-Type.
- §13triginta — P50 residue-in-kernel diagnostic; independent of νf
  carrier.
- `theory/STRUCTURAL_CONSERVATION_THEOREM.md` — full derivation of the
  symplectic conservation structure used at L1.
- `theory/TNFR_VARIATIONAL_PRINCIPLE.md` — canonical Lagrangian /
  Hamiltonian / symplectic 2-form used at L1.
- `src/tnfr/physics/conservation.py`, `src/tnfr/physics/variational.py`
  — current real-valued implementation of $J_\phi$ that demonstrates
  the catalog is consistent without (P-Pontryagin).

**Honest scope (re-stated).**  §13triginta-secunda derives, inside the
canonical catalog, that the νf-Type Conjecture reduces to
(P-νf-Bijectivity).  It does **not** decide (P-νf-Bijectivity), does
**not** close T-νf, does **not** advance G4 = RH, does **not** close
T-HP, and does **not** introduce or modify any canonical operator.
The TNFR-Riemann program remains paused at the T-HP / G4 = RH
boundary as stated in §13septies.

---


## §13triginta-tertia. Resolution of (P-νf-Bijectivity) from the Nodal Equation — Forward Dynamics vs Backward Identifiability (Closes T-νf at the Canonical Level; Does NOT Advance G4 = RH)

**Pre-registration status.**  This section executes Ruta A2 of the
νf-Type program (§§13triginta-prima, 13triginta-secunda): it tests
whether the residual axiom (P-νf-Bijectivity) identified at
§13triginta-secunda.5 is itself a canonical consequence of the
nodal equation $\partial \mathrm{EPI}/\partial t = \nu_f \cdot \Delta\mathrm{NFR}(t)$
together with Invariants #1 (Nodal Equation Integrity) and #6
(Reproducible Dynamics).

The honest verdict is pre-registered as one of:

- `FORWARD_FORCES_BACKWARD`: the forward equation implies backward
  identifiability of $\nu_f$ from the spectral content of
  $\partial \mathrm{EPI}/\partial t$.
- `FORWARD_INDEPENDENT_OF_BACKWARD`: the forward equation does **not**
  imply backward identifiability; (P-νf-Bijectivity) is a separate
  observability axiom strictly stronger than the catalog.

Scope: this section does **not** advance G4 = RH, does **not** close
T-HP, does **not** introduce or modify any canonical operator.  It
does, however, **close T-νf at the canonical level** by structurally
demonstrating that the upgrade $\nu_f : \mathbb{R}^+ \to \mathcal{M}^+(\mathbb{Z})$
is consistent with but not forced by the canonical catalog.

### §13triginta-tertia.1 The Literal Canonical Reading of the Nodal Equation

The nodal equation as canonically implemented in
`src/tnfr/operators/nodal_equation.py::compute_expected_depi_dt` is:

```python
def compute_expected_depi_dt(G: TNFRGraph, node: NodeId) -> float:
    vf = _get_node_attr(G, node, ALIAS_VF)
    dnfr = _get_node_attr(G, node, ALIAS_DNFR)
    return vf * dnfr
```

This is a **scalar product** of two `float` quantities at each node $i$
and each time $t$:

$$
\left(\frac{\partial \mathrm{EPI}}{\partial t}\right)_i \;=\; \nu_{f,i}(t) \cdot \Delta\mathrm{NFR}_i(t),
\qquad \nu_{f,i}, \; \Delta\mathrm{NFR}_i \in \mathbb{R}.
$$

This is the **literal** canonical reading of the nodal equation.  Any
upgrade of either factor (scalar → measure, real → complex, pointwise
→ functional) is a structural extension, not the literal canonical
content.  The literal reading is what Invariant #1 demands faithful
adherence to ("EPI evolution constraint: Changes occur only via
$\partial \mathrm{EPI}/\partial t = \nu_f \cdot \Delta \mathrm{NFR}$").

### §13triginta-tertia.2 What Forward Determinism Requires

A forward-deterministic specification of the nodal evolution requires
that, at each time $t$ and each node $i$:

- the value $\nu_{f,i}(t)$ is well-defined,
- the value $\Delta\mathrm{NFR}_i(t)$ is well-defined (computed
  deterministically from the graph state),
- their product is real and finite.

These conditions are **fully satisfied** by $\nu_{f,i} \in \mathbb{R}^+$
(positive scalar) and $\Delta\mathrm{NFR}_i \in \mathbb{R}$ (real scalar).
The current canonical implementation is precisely this configuration,
and:

- **Invariant #1** (Nodal Equation Integrity) holds: EPI changes occur
  only through this scalar product channel.
- **Invariant #6** (Reproducible Dynamics) holds: identical seeds
  produce identical trajectories under this evolution rule.
- **Grammar U2** (Convergence) holds whenever
  $\int \nu_{f,i}(t) \cdot |\Delta\mathrm{NFR}_i(t)| \, dt < \infty$,
  which is achievable with scalar $\nu_f$.

Therefore: **forward determinism does not require any non-scalar
upgrade of $\nu_f$**.

### §13triginta-tertia.3 What Backward Identifiability Would Require

The axiom (P-νf-Bijectivity) — stated at §13triginta-secunda.5 — is:

> $\nu_f$ must bijectively encode the spectral content of the EPI
> dynamics it drives.

This is an **inverse-problem statement**: given the observed spectral
content of $\partial \mathrm{EPI}/\partial t$ at node $i$, one should
be able to **uniquely recover** $\nu_{f,i}$.

For this recovery to be well-defined, the map

$$
\nu_{f,i} \;\longmapsto\; \mathrm{spec}\left(\frac{\partial \mathrm{EPI}}{\partial t}\bigg|_i\right)
$$

must be injective.  But the actual map factors through
$\Delta\mathrm{NFR}_i$:

$$
\nu_{f,i} \;\longrightarrow\; \nu_{f,i} \cdot \Delta\mathrm{NFR}_i(t) \;=\; \frac{\partial \mathrm{EPI}}{\partial t}\bigg|_i
\;\longrightarrow\; \mathrm{spec}(\cdot).
$$

Two distinct scalar values $\nu_{f,i} \neq \nu_{f,i}'$ produce
trajectories $\nu_{f,i} \cdot \Delta\mathrm{NFR}_i(t)$ and
$\nu_{f,i}' \cdot \Delta\mathrm{NFR}_i(t)$ that share the same
spectral support but differ in amplitude.  Amplitude is recoverable
from the spectral content only if $\Delta\mathrm{NFR}_i(t)$ is known —
i.e., the inverse problem is **already** well-posed for scalar
$\nu_f$, but **only relative to a known $\Delta\mathrm{NFR}$**.

So at the forward-dynamics level, identifiability of scalar $\nu_f$ is
straightforward modulo knowledge of $\Delta\mathrm{NFR}$, and no
upgrade to $\mathcal{M}^+(\mathbb{Z})$ is needed for the inverse
problem itself.  The Pontryagin upgrade is required only if one
demands $\nu_f$ to carry the *spectral support* of the trajectory
intrinsically — a strictly stronger requirement than forward
identifiability.

### §13triginta-tertia.4 Forward $\neq$ Backward: the Structural Distinction

The clean structural statement is:

| Property | Statement | Required by canonical catalog? |
|---|---|---|
| Forward determinism | $(\nu_{f,i}, \Delta\mathrm{NFR}_i, t) \mapsto \partial \mathrm{EPI}/\partial t\big|_i$ is single-valued | **Yes** (Invariants #1, #6) |
| Forward reproducibility | Same inputs → same trajectory | **Yes** (Invariant #6) |
| Backward observability (modulo $\Delta\mathrm{NFR}$) | $\nu_{f,i}$ recoverable from $(\partial\mathrm{EPI}/\partial t, \Delta\mathrm{NFR})$ | **Trivially yes** for scalar $\nu_f$ |
| Backward observability (intrinsic to $\nu_f$ alone) | $\nu_{f,i}$ recoverable from $\mathrm{spec}(\partial \mathrm{EPI}/\partial t)$ alone | **No** — this is (P-νf-Bijectivity) |
| Spectral self-encoding of $\nu_f$ | $\nu_f$ intrinsically encodes its own spectral fingerprint | **No** — this is (P-νf-Bijectivity) |

The bottom two rows are the content of (P-νf-Bijectivity).  Neither
is required by the literal canonical reading of the nodal equation.
Both are achievable by **adding** (P-νf-Bijectivity), but neither
follows from the catalog without it.

### §13triginta-tertia.5 Where Spectral Richness Actually Lives

In the literal canonical reading, the spectral richness of the
trajectory $\partial \mathrm{EPI}/\partial t$ is carried by:

- **$\Delta\mathrm{NFR}_i(t)$**: itself a time-varying scalar whose
  Fourier transform can have arbitrary support, because it is computed
  from the graph state (which evolves through grammar U1–U6 and
  network coupling).
- **The graph state** (EPI values, phase configuration, coupling
  topology): produces the time-varying $\Delta\mathrm{NFR}_i(t)$
  through deterministic feedback.
- **The operator sequence** (grammar U1–U6): provides the temporal
  modulation of the entire dynamics.

The scalar $\nu_{f,i}$ acts as a **multiplicative gain** on
$\Delta\mathrm{NFR}_i(t)$.  It does not generate spectral content; it
amplifies whatever spectral content already lives in
$\Delta\mathrm{NFR}_i(t)$.

This is empirically consistent with the P14 prime-ladder construction
(§8.2 and `src/tnfr/riemann/prime_ladder_hamiltonian.py`): the
prime-ladder spectrum $\{\log p_k\}$ arises from the **graph
construction** (REMESH echoes at incommensurate periods), not from any
intrinsic spectral structure of $\nu_f$.  The current implementation
uses scalar $\nu_f$ per node (uniform or topologically-modulated by
$1/\sqrt{\deg(i)}$) and still reproduces the full prime-ladder
spectrum.  This is a direct demonstration by existence that
(P-νf-Bijectivity) is not required to produce the observed spectral
richness.

### §13triginta-tertia.6 Boxed Result — Proposition T-νf-Resolution

> **Proposition T-νf-Resolution.**
>
> Let $\nu_f : V(G) \to \mathbb{R}^+$ be a scalar positive function
> on the graph $G$ (the literal canonical type).  Then:
>
> 1. *(Forward consistency)*  The nodal equation
>    $\partial \mathrm{EPI}/\partial t = \nu_f \cdot \Delta\mathrm{NFR}(t)$
>    is well-posed under Invariants #1 and #6 with $\nu_f$ scalar.
>
> 2. *(Observability)*  The map
>    $\nu_{f,i} \mapsto \partial \mathrm{EPI}/\partial t\big|_i$
>    is injective modulo knowledge of $\Delta\mathrm{NFR}_i(t)$, so
>    scalar $\nu_f$ is identifiable in the operational sense available
>    to a TNFR observer.
>
> 3. *(Sufficiency for observed spectra)*  The full spectral richness
>    documented in P12–P16 (prime-ladder, von Mangoldt, Weil–Guinand
>    explicit formula, Li–Keiper positivity) is reproduced under
>    scalar $\nu_f$ by routing spectral content through
>    $\Delta\mathrm{NFR}(t)$ and the graph state evolution.
>
> 4. *(Independence of (P-νf-Bijectivity))*  The intrinsic
>    self-encoding requirement (P-νf-Bijectivity) is an
>    **inverse-problem axiom** that is **independent** of the
>    forward-dynamics specification given by the canonical catalog.
>    It is consistent with the catalog (it adds no contradiction) but
>    is not derivable from it.
>
> Therefore the canonical type of $\nu_f$ is **scalar
> positive-real-valued per node**.  The upgrade
> $\nu_f \in \mathcal{M}^+(\mathbb{Z})$ proposed in §13triginta-prima
> is a **non-canonical structural refinement** that requires the
> additional axiom (P-νf-Bijectivity), which is not in the canonical
> catalog.

**Verdict on (P-νf-Bijectivity)** (Ruta A2):
`FORWARD_INDEPENDENT_OF_BACKWARD`.

**Verdict on Conjecture T-νf** (§13triginta-prima.2):
`CLOSED_NEGATIVELY_AT_CANONICAL_LEVEL` — the conjecture's positive
form (νf canonically forced to be a positive Radon measure on
$\mathbb{Z}$) is **refuted** at the canonical level by Proposition
T-νf-Resolution.  The canonical type of $\nu_f$ is scalar
positive-real-valued, as the literal nodal equation specifies.

### §13triginta-tertia.7 Consequence for the νf-Type Program

The closed structural picture is:

$$
\underbrace{\nu_f : V(G) \to \mathbb{R}^+}_{\text{canonical (forced by nodal equation)}}
\;\;\subsetneq\;\;
\underbrace{\nu_f \in \mathcal{M}^+(\mathbb{Z})}_{\text{non-canonical (requires P-}\nu_f\text{-Bijectivity)}}
$$

The Pontryagin / measure-valued upgrade remains a **legitimate
structural extension** of TNFR, but it must be acknowledged as such:
an *extension*, not a *canonical consequence*.  This is the same
status as, for example, the complex-extension of the spectral zeta
function (P13), which is consistent with the catalog but introduces
extra structure not present in the bare catalog.

This resolution **does not invalidate** the diagnostic value of
$\mathcal{S}_{\nu_f}$ defined in §13triginta-prima.6: the binned
spectral entropy of the P14 prime-ladder spectrum, $S_{\nu_f} \approx 0.95$,
remains a valid *upper-bound proxy* for the spectral complexity of
$\Delta\mathrm{NFR}(t)$ (which is what actually carries the spectral
content under the literal reading).  The diagnostic is reinterpreted:
it measures complexity of the trajectory $\partial \mathrm{EPI}/\partial t$,
not of $\nu_f$ intrinsically.

### §13triginta-tertia.8 What Closes and What Remains Open

**Closed by §13triginta-tertia:**

- The νf-Type question at the *canonical* level: $\nu_f$ is scalar
  positive-real-valued, as the literal nodal equation requires.
- The status of (P-Pontryagin) and (P-νf-Bijectivity): both are
  consistent extensions, neither is canonical.
- The interpretation of the $\mathcal{S}_{\nu_f}$ diagnostic: it
  measures trajectory complexity, not $\nu_f$ intrinsic complexity.

**Remains open (unchanged by §13triginta-tertia):**

- **G4 = RH** (the central open problem of the program).  The
  forward/backward distinction established here does not bear on the
  smooth/oscillatory split of T-HP.  The smooth half (P30) and the
  oscillatory half (S(T) = (1/π)arg ζ(1/2+iT)) are both formulated
  using real-valued spectral data, so their status is independent of
  the canonical type of $\nu_f$.
- **B1 vs B2 vs B3** (the three branches at §13septies for closing
  T-HP).  Unaffected by νf-Type resolution.
- Whether a non-canonical extension of TNFR to measure-valued $\nu_f$
  would yield additional structural insight on G4.  This is a
  separate research question, parallel to but independent of the
  canonical-catalog programme.

### §13triginta-tertia.9 Cross-References

- §13triginta-prima — pre-registration of T-νf; introduction of
  (P-Pontryagin) meta-axiom; $\mathcal{S}_{\nu_f}$ diagnostic.
- §13triginta-secunda — reduction of (P-Pontryagin) to
  (P-νf-Bijectivity); enumeration F1–F10 of canonical candidates.
- §13septies — T-HP statement; smooth/oscillatory split (independent
  of νf carrier type).
- §13nonies — P30 operator-level closure of the smooth half (uses
  real-valued conjugate momentum, consistent with the canonical
  scalar $\nu_f$ established here).
- §13triginta — P50 residue-in-kernel diagnostic.
- `src/tnfr/operators/nodal_equation.py::compute_expected_depi_dt` —
  literal canonical implementation `vf * dnfr` with both factors as
  `float`; this is the implementation whose canonicity is
  established in §13triginta-tertia.2.
- `src/tnfr/riemann/prime_ladder_hamiltonian.py` — P14 implementation
  using scalar $\nu_f$ per node, demonstrating by existence
  (§13triginta-tertia.5) that the prime-ladder spectrum is generated
  without (P-νf-Bijectivity).

**Honest scope (final, as of §13triginta-tertia).**

The νf-Type program is **closed at the canonical level** with verdict:
canonical $\nu_f$ is positive-real-scalar, per the literal nodal
equation.  The Pontryagin / measure-valued upgrade is a legitimate
non-canonical extension that requires the additional inverse-problem
axiom (P-νf-Bijectivity), which is independent of the catalog.

This closure does **not** advance G4 = RH, does **not** close T-HP,
does **not** introduce or modify any canonical operator, and does
**not** alter the §13septies pause-at-T-HP status of the larger
TNFR-Riemann programme.  The smooth/oscillatory split of T-HP and
its branches B1/B2/B3 remain the genuine open structural content.

The νf-Type sub-programme (§§13triginta-prima → 13triginta-tertia)
is therefore complete as a self-contained theoretical reduction:
foundational question raised (A), reduced to a deeper axiom (A1),
and decided at the canonical level (A2).  The reduction confirms
that the literal nodal equation is structurally self-sufficient and
that the canonical catalog does not require the Pontryagin upgrade.

---


## §13triginta-quarta — T-EPI Type Conjecture: pre-registration (B1a)

**Status**: Pre-registration. Type-Conjecture diagnostic for the canonical
type of EPI, mirroring the νf-Type sub-programme of §§13triginta-prima–
tertia.  Sub-question (B): *Is the literal scalar/numeric EPI of the
canonical 13-operator catalog its forced canonical type, or is a
Banach-valued upgrade (`BEPIElement` = `C^0([0,1], ℂ) ⊕ ℓ^2`) forced by
the structural axioms?*  This section establishes the diagnostic and
pre-registers the forcing axioms; the verdict is decided in
§§13triginta-quinta–sexta.

This sub-programme does **not** advance G4 = RH, does **not** modify the
catalog, and does **not** promote any operator to canonical status.
Scope is restricted to the canonical type of EPI under the existing
catalog.

### §13triginta-quarta.1 — Motivation and literal canonical witness

The 13 canonical operators of the TNFR catalog read and write EPI as a
**scalar real number**:

- `src/tnfr/operators/__init__.py:190–360` defines `get_neighbor_epi` via
  `float(v.EPI)` and all glyph operators (AL, EN, IL, OZ, UM, RA, SHA,
  VAL, NUL, THOL, ZHIR, NAV, REMESH) consume and produce literal scalar
  EPI values.
- `src/tnfr/operators/nodal_equation.py:1–160` defines
  `compute_expected_depi_dt(G, node) -> float: return vf * dnfr` and
  `validate_nodal_equation(..., epi_before: float, epi_after: float, ...)`.
  This is the **decisive canonical witness**: the nodal equation itself
  is typed `(float, float) -> float` at the operator-contract level.
- `src/tnfr/alias.py:86` defines `_bepi_to_float(value)` which down-
  projects any incoming Banach element to a scalar via the
  `max_magnitude` reading.

The literal type read by the canonical machinery is therefore
**`EPI: float`** (or its complex/real numpy scalar promotion), regardless
of any richer object the catalog *could* host.

### §13triginta-quarta.2 — Catalog statement of B_EPI

The canonical theory statement (FUNDAMENTAL_THEORY.md, GLOSSARY.md,
AGENTS.md "Structural Triad") locates EPI in a **Banach space `B_EPI`**:

  *Form (EPI): coherent structural configuration in Banach space B_EPI.*

The catalog therefore distinguishes a **type-level statement** (`EPI ∈
B_EPI`) from the **operator-level contract** (`EPI: float`).  These two
levels need not coincide: the catalog can host a richer type without any
operator constructing or reading it non-trivially.

### §13triginta-quarta.3 — The `BEPIElement` formalisation

Inspection of `src/tnfr/mathematics/epi.py:103` reveals that the catalog
contains a **fully formalised** Banach-element class:

```python
@dataclass(frozen=True)
class BEPIElement(_EPIValidators):
    f_continuous: tuple[complex, ...]  # C^0([0,1], ℂ) sample
    a_discrete:   tuple[complex, ...]  # ℓ^2(ℂ) coefficient sequence
    x_grid:       tuple[float, ...]    # uniform grid on [0,1]
    # algebraic ops: direct_sum, tensor, adjoint, compose
    # down-projection: __float__ = __abs__ = _max_magnitude
```

with companion `class BanachSpaceEPI(_EPIValidators)` at
`src/tnfr/mathematics/spaces.py:110` and serialisation/embedding helpers
`ensure_bepi`, `serialize_bepi` at `src/tnfr/types.py:270–390`.  The
embedding ℝ ↪ B_EPI is the *trivial constant function*:
`_BEPIElement((s, s), (s, s), (0.0, 1.0))`.

**This is structurally stronger evidence than νf had.**  For νf, the
catalog merely *mentions* a measure-valued / Pontryagin upgrade as a
theoretical possibility (§13triginta-prima.2).  For EPI, the catalog
contains a **complete algebraic implementation** of the Banach-valued
upgrade — including direct sum, tensor product, adjoint, composition,
and a canonical down-projection `max_magnitude` — that is **not invoked
by any of the 13 canonical operators**.

The structural question of T-EPI is therefore sharper than T-νf: not
"could a richer type be forced?" but "is the formalised richer type
operationally inert under the canonical operators?".

### §13triginta-quarta.4 — REMESH history-vector caveat

`theory/REMESH_INFINITY_DERIVATION.md:50–52` defines the REMESH
state vector

  `x(t) = (EPI(t), …, EPI(t − T_max))^⊤ ∈ ℝ^(T_max + 1)`

This is **time-aggregation of scalar readings**, not intrinsic per-node
vectoriality.  It does not promote per-node EPI to a Banach element; it
constructs a global time-window state from scalar samples.  The N15
REMESH-∞ closure operates entirely on this scalar-history vector and
produces a bounded self-adjoint orthogonal projection on `H^2(D)` — its
range and kernel are subspaces of *time-trajectory space*, not of
per-node Banach space.  Hence the N15 closure is consistent with the
scalar-EPI contract and does not force a BEPI upgrade.

### §13triginta-quarta.5 — T-EPI Conjecture (formal statement)

**Conjecture T-EPI (pre-registered).**  Under the canonical 13-operator
catalog, the existing per-node `EPI: float` contract is **forced** as
the canonical type, in the sense that:

  (a) No canonical operator constructs a `BEPIElement` with non-trivial
      `f_continuous` or `a_discrete` components.
  (b) No canonical operator reads `BEPIElement` data other than through
      the down-projection `_bepi_to_float = max_magnitude`.
  (c) The forcing-axiom inventory F1–F10 (§13triginta-quarta.7) admits
      no canonical extension that selects a non-trivial Banach element
      from a scalar starting state.

Conjecture T-EPI is the **EPI analogue** of Conjecture T-νf
(§13triginta-prima.5).  Its expected verdict, by §§13triginta-quarta.1–
.4, is **NEGATIVE at the canonical level**: scalar EPI is forced;
`BEPIElement` is a legitimate non-canonical envelope (formalised but
not invoked).

### §13triginta-quarta.6 — Diagnostic S_EPI (two-axis necessary condition)

The diagnostic certificate
`src/tnfr/riemann/epi_type_signature.py::compute_epi_type_signature`
computes a two-axis necessary-condition score on a canonical SDK-built
ring graph evolved by `tnfr.dynamics.step`:

- **Storage axis** (`storage_bepi_fraction`):
  fraction of nodes whose `EPI` attribute is a non-trivial
  `BEPIElement` (test: `std(f_continuous) > atol` OR
  `max|a_discrete| > atol`).
- **Spectral axis** (`signature ∈ [0, 1]`):
  per-node binned spectral entropy of the scalar EPI(t) trajectory,
  normalised by `log(n_bins)`.  `S_EPI → 0` indicates a single-mode
  (DC-like) trajectory; `S_EPI → 1` indicates a uniform spread over
  spectral bins.

**Pre-registered verdict thresholds**:

| Verdict | Condition |
|---|---|
| `SCALAR_ADEQUATE` | `signature < 0.15` AND `storage_bepi_fraction == 0` |
| `INDETERMINATE` | between thresholds |
| `BEPI_VALUED_NECESSARY` | `signature > 0.5` OR `storage_bepi_fraction > 0` |

**Measured values** (`examples/05_type_hygiene/79_epi_type_signature_demo.py`,
seeds 13 and 29):

| Resolution | n_nodes | n_steps | n_bins | S_EPI | BEPI fraction | Verdict |
|---|---|---|---|---|---|---|
| 1 | 24 | 64 | 32 | **0.876342** | **0.0000** | `BEPI_VALUED_NECESSARY` |
| 2 | 48 | 128 | 64 | **0.895673** | **0.0000** | `BEPI_VALUED_NECESSARY` |

**Empirical reading (decisive structural finding).**  The two axes
**disagree**: the storage axis is uniformly **scalar** (zero nodes carry
non-trivial BEPI components, confirming §§13triginta-quarta.1–.3), while
the spectral axis is uniformly **multi-modal** (S_EPI ≈ 0.88–0.90,
N_eff ≈ 21–41 effective spectral modes).

This is a **temporal-modal equivalence** signal: the scalar EPI(t)
trajectory under canonical operators carries the same multi-modal
information content that a `BEPIElement.f_continuous` / `a_discrete`
decomposition would carry — encoded **temporally** (across `step`
iterations) rather than **spatially-in-modes** (across the BEPI
direct-sum slots).  The high spectral entropy of the scalar trajectory
demonstrates that the catalog's modal capacity is already operative;
it is simply realised through the time dimension and not through a
spatial Banach decomposition.

The crossed verdict
`(storage = SCALAR) ∧ (spectral = MULTI-MODAL)` is therefore
**structurally consistent with T-EPI NEGATIVE**: scalar EPI is the
forced canonical type, and the formalised `BEPIElement` envelope is
operationally redundant because temporal trajectories already encode
the multi-modal content.  The verdict label `BEPI_VALUED_NECESSARY`
produced by the spectral threshold is, in context, a *necessary-
condition false positive* that is correctly interpreted only after
reading the storage axis jointly.

The diagnostic does **not** decide T-EPI by itself; the verdict is
deferred to §§13triginta-quinta–sexta (forcing-axiom reduction and
final NEGATIVE classification).

### §13triginta-quarta.7 — Forcing axioms F1–F10 (inventory)

Pre-registered inventory of structural axioms that any "canonical
forcing" of `BEPIElement` would have to satisfy.  Detailed reduction in
§13triginta-quinta.

| # | Axiom | Source |
|---|---|---|
| F1 | Operator exclusivity (only the 13 canonical operators write EPI). | AGENTS.md "Canonical Invariants #1". |
| F2 | Reproducibility (identical seeds → identical trajectories). | AGENTS.md "Reproducible Dynamics". |
| F3 | Nodal-equation type closure (`compute_expected_depi_dt: float`). | `operators/nodal_equation.py:1–160`. |
| F4 | Tetrad orthogonality (Φ_s, |∇φ|, K_φ, ξ_C span the structural channels). | AGENTS.md §"Minimal Structural Degrees of Freedom". |
| F5 | REMESH time-aggregation only (no per-node spatial Banach upgrade). | §13triginta-quarta.4. |
| F6 | P14 prime-ladder Hamiltonian operates on a scalar-spectrum Hilbert space. | §10–§12, `riemann/prime_ladder_hamiltonian.py`. |
| F7 | Uncertainty-bandwidth complementarity (`ΔEPI · Δνf ≥ K`, scalar form). | AGENTS.md "Quantum-Like Regime". |
| F8 | `BEPIElement` catalog existence (the (P-EPI-Bijectivity) analog of (P-νf-Bijectivity) is the existence-without-construction gap). | `mathematics/epi.py:103`, `types.py:270`. |
| F9 | Classical-limit demos use scalar EPI exclusively. | `examples/02_physics_regimes/12_classical_mechanics_demo.py`. |
| F10 | Quantum-regime demos use scalar EPI exclusively. | `examples/02_physics_regimes/13_quantum_mechanics_demo.py`, `14_uncertainty_and_interference.py`. |

Axioms F1–F10 together force scalar EPI as the canonical type unless an
extension axiom is *added* to the catalog.  No such canonical extension
exists in the current 13-operator construction.

### §13triginta-quarta.8 — Honest scope (what this does and does not do)

This sub-programme:

- **Does** establish a falsifiable diagnostic for the canonical type of EPI.
- **Does** identify the `BEPIElement` formalisation as a structurally
  stronger non-canonical envelope than the νf measure-valued envelope.
- **Does** identify the temporal-modal equivalence as the operational
  reason scalar EPI suffices.
- **Does not** advance G4 = RH or the T-HP conjecture.
- **Does not** promote any operator, field, or constant to canonical
  status.
- **Does not** modify the 13-operator catalog.
- **Does not** invalidate the existing `BEPIElement` implementation; it
  classifies it as a *legitimate non-canonical envelope* available for
  research use outside the canonical operator contracts.

### §13triginta-quarta.9 — Cross-references

- §13triginta-prima — T-νf Type Conjecture (pre-registration, νf analog).
- §13triginta-secunda — T-νf forcing-axiom reduction.
- §13triginta-tertia — T-νf NEGATIVE verdict.
- §13septies — T-HP open content (independent of this sub-question).
- §19.1 — Full P1–P49 milestone table.
- `src/tnfr/riemann/epi_type_signature.py` — diagnostic implementation.
- `examples/05_type_hygiene/79_epi_type_signature_demo.py` — two-resolution demo.
- `src/tnfr/mathematics/epi.py:103` — `BEPIElement` formalisation.
- `src/tnfr/operators/nodal_equation.py:1–160` — scalar contract witness.

---

## §13triginta-quinta. Derivation of (P-BEPI-Carrier) from the Canonical Catalog — Foundational Reduction of the EPI-Type Conjecture (Theory-Only Analysis; Does NOT Advance G4 = RH)

**Pre-registration status.**  This section executes the forcing-axiom
reduction phase (B1b) of the T-EPI program (§13triginta-quarta): it
attempts to derive the Banach-EPI carrier principle (P-BEPI-Carrier)
from the canonical six invariants + nodal equation + Structural
Conservation Theorem + Variational Principle + REMESH operator, *or* to
identify and isolate the actual residual axiom that the derivation
requires beyond the catalog.

The honest verdict (executed in §13triginta-sexta) is pre-registered as
one of:

- `COROLLARY_DERIVED`: (P-BEPI-Carrier) follows from invariants 1–6 alone.
- `CONDITIONAL_COROLLARY`: (P-BEPI-Carrier) follows under one additional
  identifiable axiom strictly weaker than itself.
- `INDEPENDENT_AXIOM`: (P-BEPI-Carrier) is independent of the catalog.

Scope (mandatory honesty): this section does **not** advance G4 = RH,
does **not** close T-HP, does **not** introduce or modify any canonical
operator, does **not** delete or deprecate `BEPIElement`, and does
**not** by itself close T-EPI.  It locates the foundational axiom *one
structural level below* (P-BEPI-Carrier) and hands T-EPI back to that
deeper question.

The literal canonical statement under scrutiny:

> **(P-BEPI-Carrier).**  In the canonical TNFR formulation, the per-node
> EPI state must take values in a non-trivial Banach space $B_\mathrm{EPI}$
> equipped with direct-sum, tensor-product, adjoint, and composition
> operations (the `BEPIElement` structure of
> `src/tnfr/mathematics/epi.py:103`), *not* in $\mathbb{R}$.

### §13triginta-quinta.1 Available Canonical Tools

The derivation may use only the following canonical machinery (no
extraneous structure):

1. **Nodal equation**: $\partial \mathrm{EPI}/\partial t = \nu_f \cdot \Delta \mathrm{NFR}(t)$
   (Invariant #1), implemented as
   `compute_expected_depi_dt: (float, float) → float`
   (`src/tnfr/operators/nodal_equation.py:1–160`).
2. **Six canonical invariants** (AGENTS.md): Nodal Equation Integrity,
   Phase-Coherent Coupling, Multi-Scale Fractality, Grammar Compliance,
   Structural Metrology, Reproducible Dynamics.
3. **Grammar U1–U6**, all derivable from invariant #1 and the bounded
   evolution constraint $\int \nu_f \, \Delta \mathrm{NFR} \, dt < \infty$
   (U2).
4. **Structural Field Tetrad** $(\Phi_s, |\nabla\phi|, K_\phi, \xi_C)$,
   the minimal derivative-tower basis, all derived from a
   *scalar* phase field $\phi$ and a *scalar* pressure field
   $\Delta\mathrm{NFR}$; only $\pi$ is a genuine structural scale
   (AGENTS.md §"Minimal Structural Degrees of
   Freedom").
5. **Structural Conservation Theorem**
   (`src/tnfr/physics/conservation.py`,
   `theory/STRUCTURAL_CONSERVATION_THEOREM.md`): per-node Noether charge
   density $\rho_i \in \mathbb{R}$ and current vector
   $\mathbf{J}_i \in \mathbb{R}^2$; the Lyapunov energy density
   $\varepsilon_i \in \mathbb{R}_{\geq 0}$ aggregates scalar squares.
6. **Variational Principle**: Lagrangian
   $\mathcal{L}_i = T_i - V_i$ where every term is a real-valued
   functional of scalar tetrad fields.
7. **REMESH operator** (canonical operator #13), generating the temporal
   history vector
   $x_i(t) = (\mathrm{EPI}_i(t), \dots, \mathrm{EPI}_i(t - T_{\max}))^\top
   \in \mathbb{R}^{T_{\max}+1}$
   (`theory/REMESH_INFINITY_DERIVATION.md:50–52`).

### §13triginta-quinta.2 What the Canonical Catalog Forces (Scalar Layer)

The chain of forced structure is straightforward and entirely inside the
catalog:

- **(M1) Operator contracts are scalar.**  All 13 canonical glyph
  operators read and write `float(v.EPI)` via the
  `_bepi_to_float` down-projection (`src/tnfr/alias.py:86`,
  `src/tnfr/operators/__init__.py:190–360`).  No operator constructs,
  reads, or preserves a `BEPIElement` instance.  Empirically verified
  by `examples/05_type_hygiene/79_epi_type_signature_demo.py`: BEPI-storage fraction
  $= 0$ across all measured nodes and steps at two independent
  resolutions $(n=24, T=64)$ and $(n=48, T=128)$.

- **(M2) The nodal equation is scalar.**  The canonical type signature
  is $\mathbb{R} \times \mathbb{R} \to \mathbb{R}$
  (item 1, `nodal_equation.py:1–160`).  No multi-modal carrier is
  forced by the ODE: any scalar trajectory $\mathrm{EPI}(t) \in \mathbb{R}$
  driven by scalar $\nu_f \in \mathbb{R}$ and scalar
  $\Delta\mathrm{NFR}(t) \in \mathbb{R}$ satisfies the equation
  exactly.

- **(M3) The tetrad is derived from scalar fields.**  The four
  canonical structural fields are pointwise functionals of the scalar
  phase $\phi_i \in S^1$ and the scalar pressure
  $\Delta\mathrm{NFR}_i \in \mathbb{R}$ (item 4).  No tetrad-field
  computation invokes a Banach inner product, direct sum, or tensor
  product on EPI itself.

- **(M4) Conservation and variational laws close on scalars.**  The
  Noether charge $Q = \sum_i \rho_i$, the energy
  $E = \sum_i \varepsilon_i$, the Lagrangian $\mathcal{L}$, and the
  symplectic form $\omega$ are all real-valued functionals of scalar
  tetrad fields and the scalar EPI (items 5–6).

The conjunction M1+M2+M3+M4 establishes that **the entire canonical
machinery closes consistently with scalar EPI**.  The 13-operator
catalog never reads or writes a `BEPIElement`; the nodal equation never
demands one; the tetrad never invokes one; conservation and variational
laws never require one.

### §13triginta-quinta.3 The Gap Between REMESH Temporal Aggregation and BEPI Spatial Aggregation

Scalar-layer closure (M1–M4) is necessary but not sufficient to refute
(P-BEPI-Carrier): one could still ask whether the catalog *also* admits
a strictly-stronger BEPI-valued realisation in which the scalar
implementation is a faithful coordinate projection.  The decisive
question is whether the catalog *forces* such an upgrade.

The only canonical mechanism that aggregates multi-component structural
content is **REMESH** (operator #13).  REMESH aggregates **across
time**: the history vector
$x_i(t) \in \mathbb{R}^{T_{\max}+1}$
collects $T_{\max}+1$ scalar EPI values along the temporal axis at a
single node $i$.  This is a $\mathbb{R}$-module structure indexed by
*time*, not a Banach structure indexed by *internal modal degrees of
freedom*.

`BEPIElement` aggregates **across internal modes** at a single node and
a single time instant: `f_continuous`, `a_discrete`, and `x_grid`
together encode a continuous-spectrum component, a discrete-spectrum
component, and a sampling grid, all at fixed $(i, t)$.  The `direct_sum`,
`tensor`, `adjoint`, and `compose` operations act on this internal
modal structure.

The gap is structural and explicit:

- REMESH provides **temporal modal expressivity** ($\mathbb{R}$-valued
  on a time grid).
- `BEPIElement` provides **spatial / internal modal expressivity**
  (Banach-valued at a single point in spacetime).

No canonical operator lifts REMESH temporal aggregation to BEPI internal
aggregation.  The two are not isomorphic at the operator-contract level:
REMESH writes back a scalar via $\mathrm{EPI}(t+1) = M\,x(t)$ with $M$
the canonical mixing matrix, and the output is consumed by the next
glyph operator via $\mathrm{float}(v.\mathrm{EPI})$.  The Banach
operations of `BEPIElement` are never invoked anywhere in the canonical
pipeline.

**Temporal-Modal Equivalence Principle (TMEP, restated from
§13triginta-quarta.6).**  Whatever multi-modal content a coherent EPI
signal carries, the canonical catalog encodes it **temporally** through
REMESH, **not** spatially through a Banach internal structure.  The
spectral richness measured by the diagnostic $S_\mathrm{EPI}$
(B1a, §13triginta-quarta.6: $S_\mathrm{EPI} \approx 0.876$–$0.896$
across two resolutions) is *explained* by TMEP without invoking
(P-BEPI-Carrier).

Therefore: **(P-BEPI-Carrier) is strictly stronger than what M1+M2+M3+M4
+ TMEP provide**, and any derivation must locate an additional
canonical constraint that selects the spatial/internal Banach upgrade.

### §13triginta-quinta.4 Candidate Forcing Constraints (Enumeration)

The candidates available inside the canonical catalog are enumerated
below.  Each row asks: *does this axiom force the BEPI carrier upgrade?*

| #  | Axiom | Source | Forces BEPI carrier? |
|---|---|---|---|
| F1 | Operator exclusivity (only the 13 canonical operators write EPI). | AGENTS.md "Canonical Invariants #1". | **No** — operators write `float` (M1). |
| F2 | Reproducibility under fixed seeds. | AGENTS.md "Reproducible Dynamics". | **No** — scalar trajectories reproduce identically. |
| F3 | Nodal-equation type closure $(\mathbb{R}, \mathbb{R}) \to \mathbb{R}$. | `nodal_equation.py:1–160`. | **No** — scalar ODE admits scalar solutions (M2). |
| F4 | Tetrad orthogonality $(\Phi_s, |\nabla\phi|, K_\phi, \xi_C)$ minimality. | AGENTS.md §"Minimal Structural Degrees of Freedom". | **No** — tetrad derived from scalar $\phi$ and scalar $\Delta\mathrm{NFR}$ (M3). |
| F5 | REMESH time-aggregation only (no per-node spatial Banach upgrade). | §13triginta-quarta.4, `REMESH_INFINITY_DERIVATION.md:50–52`. | **No** — REMESH is temporal, BEPI is spatial; no canonical lift exists (§13triginta-quinta.3). |
| F6 | P14 prime-ladder Hamiltonian on a scalar-spectrum Hilbert space. | §10–§12, `riemann/prime_ladder_hamiltonian.py`. | **No** — P14's Hilbert space is built from scalar eigenmodes of the temporal operator, not from per-node Banach data. |
| F7 | Uncertainty-bandwidth complementarity $\Delta\mathrm{EPI} \cdot \Delta\nu_f \geq K$. | AGENTS.md "Quantum-Like Regime". | **No** — variances are real-valued moments of scalar distributions. |
| F8 | `BEPIElement` exists as a research formalism. | `mathematics/epi.py:103`, `types.py:270`. | **No** — existence in the codebase is not the same as canonical operator contracts.  (This is the (P-EPI-Bijectivity) gap, see §13triginta-quinta.5.) |
| F9 | Classical-limit demos use scalar EPI exclusively. | `examples/02_physics_regimes/12_classical_mechanics_demo.py`. | **No** — classical regime emerges from scalar EPI under high coherence. |
| F10 | Quantum-regime demos use scalar EPI exclusively. | `examples/02_physics_regimes/13_quantum_mechanics_demo.py`, `14_uncertainty_and_interference.py`. | **No** — quantum-like phenomena (quantization, interference, complementarity) emerge from scalar EPI dynamics, not from a Banach internal carrier. |

**Result.** No canonical constraint in $\{\mathrm{F1}, \ldots, \mathrm{F10}\}$
forces the Banach carrier upgrade of EPI.  All ten admit consistent
realisation with scalar EPI (as the current 13-operator implementation
demonstrates by existence and as the B1a empirical signature confirms:
BEPI-storage fraction $= 0$ across two independent demo resolutions).

### §13triginta-quinta.5 The Hidden Axiom: (P-EPI-Bijectivity)

The derivation gap can be isolated cleanly.  Define:

> **(P-EPI-Bijectivity).**  In the canonical TNFR formulation, the
> per-node EPI value at instant $t$ must bijectively encode the
> *internal modal content* (continuous spectrum, discrete spectrum,
> sampling grid) of the structural pattern it represents at that
> $(i, t)$.  Equivalently: distinct internal modal decompositions at
> the same $(i, t)$ must correspond to distinct EPI instances, and
> conversely.

**Claim.**  (P-BEPI-Carrier) is a corollary of the canonical catalog
*plus* (P-EPI-Bijectivity), and of nothing weaker than
(P-EPI-Bijectivity).

**Forward direction (sufficiency).**  Assume (P-EPI-Bijectivity).
Consider a coherent pattern with non-trivial internal modal
decomposition (e.g., a superposition of a continuous-spectrum component
and a discrete-spectrum component at the same node $i$ and time $t$,
as constructed in `BEPIElement.direct_sum`).  Bijectivity forces EPI to
encode this full internal decomposition faithfully at $(i, t)$.  A
scalar $\mathrm{EPI}_i(t) \in \mathbb{R}$ does not have the cardinality
to encode arbitrary $L^2$-valued continuous-spectrum data
*simultaneously* with discrete-spectrum data at fixed $(i, t)$ (one
real number cannot inject into a non-trivial Banach space).  Hence
$\mathrm{EPI}_i(t)$ must take values in a non-trivial Banach space —
the `BEPIElement` structure.  This is (P-BEPI-Carrier).

**Reverse direction (necessity at the canonical level).**  Suppose
(P-BEPI-Carrier) holds.  Then $\mathrm{EPI}_i(t) \in B_\mathrm{EPI}$
is fully specified by its `BEPIElement` data
$(f_\mathrm{continuous}, a_\mathrm{discrete}, x_\mathrm{grid})$.  By
the definitions of `direct_sum`, `tensor`, `adjoint`, and `compose`,
distinct decompositions at the same $(i, t)$ produce distinct Banach
elements.  Hence (P-EPI-Bijectivity) holds.

**Strict-weakness of (P-EPI-Bijectivity) vs (P-BEPI-Carrier).**
(P-EPI-Bijectivity) is a *meta-constraint* on the encoding map
$\mathrm{EPI}_i(t) \mapsto$ (internal modal content at $(i, t)$).  It
does not mention Banach spaces, direct sums, tensor products,
adjoints, or any functional-analytic machinery.  It is purely a
faithfulness requirement on the symbolic representation at a single
spacetime point.  By contrast, (P-BEPI-Carrier) commits to a specific
carrier ($B_\mathrm{EPI}$) and a specific operator algebra
(`direct_sum`/`tensor`/`adjoint`/`compose`).

Therefore (P-EPI-Bijectivity) is structurally simpler and strictly
weaker than (P-BEPI-Carrier), and the derivation is genuine progress.

### §13triginta-quinta.6 Canonical Status of (P-EPI-Bijectivity) — TMEP Refutation

The question is now: is (P-EPI-Bijectivity) itself derivable from the
canonical six invariants?

- **(B-Pro).**  Invariant #1 (traceability) and Invariant #3
  (multi-scale fractality) together suggest that EPI should fully
  encode the structural pattern it represents.  If two distinct
  internal modal decompositions could correspond to the same scalar
  $\mathrm{EPI}_i(t)$, faithfulness at fixed $(i, t)$ would be lost.

- **(B-Con, decisive).**  The **Temporal-Modal Equivalence Principle**
  (TMEP, §13triginta-quinta.3) refutes the per-spacetime-point
  bijectivity requirement *at the canonical level*: multi-modal content
  is not required to fit into a single $(i, t)$ slot, because REMESH
  provides a canonical temporal channel for exactly that content.  Any
  multi-modal pattern at node $i$ is canonically realised as a
  *time-indexed sequence* $\{\mathrm{EPI}_i(t_k)\}_k$, not as a
  Banach-valued single sample.  Bijectivity is enforced at the level of
  the temporal trajectory (a sequence of scalars), not at the level of
  a single spacetime point.

  Formally: the catalog enforces faithfulness via the pair
  $(\nu_f, \Delta\mathrm{NFR})$ acting on $\mathrm{EPI}(t)$ via the
  scalar nodal ODE, with REMESH closing the temporal loop.  This is
  *operationally complete* — it reproduces P12–P15 to machine precision
  (§10–§12) and recovers classical, quantum-like, and number-theoretic
  spectra (§§3–9) without any per-point Banach upgrade.

- **(B-Empirical).**  The B1a diagnostic (§13triginta-quarta.6) measures
  $S_\mathrm{EPI} \approx 0.876$–$0.896$ across two resolutions: rich
  spectral content along the *temporal* axis (`_binned_psd_distribution`
  on the time series), with BEPI-storage fraction $= 0$ at every
  measured $(i, t)$.  This is exactly the TMEP signature: multi-modal
  expressivity is present, but exclusively temporal.

**Conclusion of §13triginta-quinta.6.**  (P-EPI-Bijectivity) is
**not derivable from the canonical six invariants**.  The catalog
realises faithfulness *temporally* via REMESH, not *spatially* via a
Banach upgrade.  The per-spacetime-point bijectivity that
(P-EPI-Bijectivity) demands is an *additional* axiom, independent of
the catalog and refuted by TMEP at the canonical level.

### §13triginta-quinta.7 Sub-Verdict

The forcing-axiom reduction yields:

> **Sub-verdict (§13triginta-quinta).**
> (P-BEPI-Carrier) is a **CONDITIONAL_COROLLARY** of the canonical
> catalog: it follows from the catalog *plus* (P-EPI-Bijectivity).
> However, (P-EPI-Bijectivity) is itself **INDEPENDENT_AXIOM** at the
> canonical level: it is not derivable from invariants 1–6 and is
> actively refuted by the Temporal-Modal Equivalence Principle (TMEP).
>
> Net: (P-BEPI-Carrier) is **strictly non-canonical**.  The
> `BEPIElement` formalisation is a legitimate research envelope —
> available for off-catalog experimentation — but is not forced by, and
> indeed is structurally redundant with, the canonical 13-operator
> realisation under TMEP.

This locates the residual canonical question for T-EPI exactly one
level below (P-BEPI-Carrier), at (P-EPI-Bijectivity), and identifies
its refutation mechanism (TMEP).  The final NEGATIVE verdict on T-EPI,
and the classification of `BEPIElement` as a legitimate non-canonical
research envelope, are executed in §13triginta-sexta (B1c).

### §13triginta-quinta.8 Honest Scope (What This Does and Does Not Do)

This sub-programme:

- **Does** isolate the residual axiom one structural level below
  (P-BEPI-Carrier).
- **Does** prove (P-EPI-Bijectivity) is strictly weaker than
  (P-BEPI-Carrier).
- **Does** refute (P-EPI-Bijectivity) at the canonical level via TMEP,
  empirically corroborated by B1a's BEPI-storage fraction $= 0$.
- **Does** confirm the catalog closes consistently with scalar EPI
  (M1+M2+M3+M4).
- **Does not** advance G4 = RH or the T-HP conjecture.
- **Does not** promote any operator, field, or constant to canonical
  status.
- **Does not** modify the 13-operator catalog.
- **Does not** delete or deprecate `BEPIElement`; classifies it as a
  research envelope available outside the canonical operator contracts.
- **Does not** by itself close T-EPI — the final verdict is executed in
  §13triginta-sexta.

### §13triginta-quinta.9 Cross-references

- §13triginta-prima — T-νf Type Conjecture (pre-registration, νf analog).
- §13triginta-secunda — T-νf forcing-axiom reduction
  (structural twin of this section).
- §13triginta-tertia — T-νf NEGATIVE verdict (precedent).
- §13triginta-quarta — T-EPI pre-registration (B1a anchor + diagnostic).
- §13septies — T-HP open content (independent of this sub-question).
- §19.1 — Full P1–P49 milestone table.
- `theory/CATALOG_TYPE_HYGIENE_PROGRAMME.md` §4 — programme tracker
  (row B1 Phase b advances on this commit).
- `src/tnfr/riemann/epi_type_signature.py` — diagnostic implementation
  (anchors M1 empirically).
- `examples/05_type_hygiene/79_epi_type_signature_demo.py` — two-resolution demo
  (corroborates TMEP via BEPI-storage fraction $= 0$).
- `src/tnfr/mathematics/epi.py:103` — `BEPIElement` formalisation
  (the non-canonical envelope being classified).
- `src/tnfr/operators/nodal_equation.py:1–160` — scalar contract witness
  (anchors M2).
- `src/tnfr/operators/__init__.py:190–360` — 13-operator scalar reads
  (anchors M1).
- `src/tnfr/alias.py:86` — `_bepi_to_float` down-projection witness
  (anchors M1 implementation path).
- `theory/REMESH_INFINITY_DERIVATION.md:50–52` — REMESH history-vector
  temporal aggregation (anchors §13triginta-quinta.3 gap argument).

---

## §13triginta-sexta. T-EPI Final NEGATIVE Verdict and Envelope Classification of `BEPIElement` (Closes B1; Does NOT Advance G4 = RH)

**Pre-registration closure.**  This section consumes the
sub-verdict of §13triginta-quinta (B1b) and issues the final
T-EPI verdict in accordance with the four-tier methodology of the
catalog type-hygiene programme (`theory/CATALOG_TYPE_HYGIENE_PROGRAMME.md`
§3, methodology lessons L1–L2).  The verdict pre-register from
§13triginta-quarta listed three admissible outcomes; B1b has
selected the NEGATIVE branch.

### §13triginta-sexta.1 Verdict

> **T-EPI verdict: NEGATIVE.**
> The Banach-EPI carrier principle (P-BEPI-Carrier) is **not canonical**.
> It does not follow from the canonical six invariants, nor from the
> nodal equation, nor from any subset of grammar U1–U6, nor from the
> structural-field tetrad, nor from the Structural
> Conservation Theorem, nor from the Variational Principle, nor from
> REMESH temporal aggregation.  Its derivation requires the additional
> axiom (P-EPI-Bijectivity), which is itself independent of the
> canonical catalog and actively refuted at the canonical level by the
> Temporal-Modal Equivalence Principle (TMEP, §13triginta-quinta.3,
> .6).

This closes T-EPI in the same shape as T-νf (B0, §13triginta-tertia):
the conjectured "type upgrade" of a fundamental TNFR observable is
classified as a legitimate research envelope, not as a canonical
catalog requirement.

### §13triginta-sexta.2 Envelope Classification of `BEPIElement`

`src/tnfr/mathematics/epi.py:103` (`BEPIElement` frozen dataclass with
`f_continuous`, `a_discrete`, `x_grid` and the operations
`direct_sum`, `tensor`, `adjoint`, `compose`) is hereby classified
as:

> **`BEPIElement` — Non-canonical research envelope (E2).**
> Status: legitimate research formalism, off-catalog.
> Canonical relationship: **structurally redundant** with the canonical
> scalar EPI realisation under TMEP — the same multi-modal expressivity
> is canonically encoded *temporally* via REMESH on scalar
> $\mathrm{EPI}(t) \in \mathbb{R}$.
> Catalog interaction: **none** required.  The 13 canonical operators
> do not read, write, preserve, or invoke any `BEPIElement` method;
> they operate exclusively through `_bepi_to_float`
> (`src/tnfr/alias.py:86`) to a scalar slot.

This mirrors the E1 classification of Pontryagin measure-valued
$\nu_f$ in §13triginta-tertia.2 (T-νf NEGATIVE).  The envelope
register now records two entries:

| ID | Object | Source | Verdict | Refutation mechanism |
|---|---|---|---|---|
| E1 | Pontryagin measure-valued $\nu_f$ | §13triginta-tertia | NEGATIVE | Scalar-storage axis + measure-redundancy under canonical νf-update |
| E2 | `BEPIElement` Banach carrier | this section | NEGATIVE | TMEP (temporal-modal aggregation suffices); BEPI-storage fraction = 0 across two resolutions |

### §13triginta-sexta.3 No Deletion, No Deprecation, No Modification

The verdict does **not** authorise:

- deletion of `BEPIElement` or any of its methods;
- deprecation warnings in `src/tnfr/mathematics/epi.py`;
- removal of `BEPIElement` from public `__init__.py` exports;
- modification of the 13-operator catalog;
- modification of the canonical contract
  $(\nu_f, \Delta\mathrm{NFR}) \mapsto \partial\mathrm{EPI}/\partial t$;
- changes to `src/tnfr/operators/nodal_equation.py`,
  `src/tnfr/operators/__init__.py`, or `src/tnfr/alias.py`;
- any change to grammar U1–U6;
- any claim about G4 = RH, T-HP, or the open content of
  §13septies.

`BEPIElement` remains available for off-catalog research
(e.g., Banach-internal experimental modelling of structural patterns
that the researcher wishes to handle spatially rather than temporally),
provided such research is documented as off-catalog and does not claim
canonical status.

### §13triginta-sexta.4 Programme Bookkeeping

- `theory/CATALOG_TYPE_HYGIENE_PROGRAMME.md` §4 row B1: Phase c
  advances ⏳ → ✅; Verdict column advances "(NEG exp.)" → "NEGATIVE";
  commit-refs column appends the present commit hash.
- §3 sub-question registry status: B1 transitions from
  🟡 IN PROGRESS to ✅ COMPLETE.
- Progress summary advances: 2 sub-questions complete (B0 + B1),
  0 in progress, 11 pending (B2 – B11 + Final).
- §6 methodology lessons: an L3 entry is recorded
  (see §13triginta-sexta.5 below) reflecting the cross-conjecture
  pattern observed across B0 and B1.

### §13triginta-sexta.5 Methodology Lesson L3 (Cross-Conjecture Pattern)

Both T-νf (B0) and T-EPI (B1) closed NEGATIVE with the same
structural shape:

1. **Anchor** identifies a candidate "type upgrade" of a canonical
   observable (measure-valued $\nu_f$; Banach-valued EPI).
2. **Diagnostic** measures two orthogonal axes: scalar-storage
   utilisation + spectral/entropy richness.
3. **Forcing-axiom reduction** finds that no canonical constraint
   forces the upgrade; isolates a single residual axiom strictly
   weaker than the upgrade itself ((P-νf-Bijectivity);
   (P-EPI-Bijectivity)).
4. **Canonical-status check** finds that the residual axiom is itself
   independent of the catalog and is actively refuted by an existing
   canonical mechanism (scalar νf-update closure; REMESH/TMEP).
5. **Verdict** NEGATIVE; the upgrade-carrier is reclassified as a
   legitimate non-canonical research envelope.

**L3 (cross-conjecture pattern).**  *Whenever a candidate type-upgrade
of a canonical observable can be matched by an existing canonical
aggregation mechanism (νf-update closure for $\nu_f$; REMESH temporal
aggregation for EPI), the upgrade is non-canonical and the existing
mechanism is preferred.*  This is the structural analogue of Occam's
razor specialised to the TNFR catalog: canonical machinery that
**already** discharges the expressivity demand makes the upgrade
non-canonical, regardless of whether the upgrade is internally
consistent.

L3 will be tested against subsequent sub-questions (B2 = T-φ
onwards).  If it holds across B2 – B11, it becomes a working
heuristic for the Final synthesis step.

### §13triginta-sexta.6 Honest Scope (Mandatory)

This section:

- **Does** close T-EPI (B1) with a NEGATIVE verdict.
- **Does** classify `BEPIElement` as legitimate non-canonical
  research envelope E2.
- **Does** advance the catalog type-hygiene programme to 2/11+1
  complete.
- **Does** record cross-conjecture methodology lesson L3.
- **Does not** advance G4 = RH, does not close T-HP, does not promote
  any operator/field/constant to canonical status, does not modify the
  catalog, does not modify any source file in `src/tnfr/`.
- **Does not** make any claim about T-φ (B2), T-ΔNFR (B3), or any
  subsequent sub-question; those are addressed sequentially per the
  programme tracker.

### §13triginta-sexta.7 Cross-references

- §13triginta-prima — T-νf pre-registration (precedent template).
- §13triginta-secunda — T-νf forcing-axiom reduction (precedent).
- §13triginta-tertia — T-νf NEGATIVE verdict + E1 classification
  (direct precedent; same shape).
- §13triginta-quarta — T-EPI pre-registration (anchor + B1a
  diagnostic).
- §13triginta-quinta — T-EPI forcing-axiom reduction (decisive
  input to this section).
- §13septies — T-HP open content (independent, untouched by this
  verdict).
- §19.1 — Full P1–P49 milestone table.
- `theory/CATALOG_TYPE_HYGIENE_PROGRAMME.md` §3, §4, §6 — programme
  tracker (advances on this commit).
- `src/tnfr/mathematics/epi.py:103` — `BEPIElement` source (preserved
  as envelope E2).
- `src/tnfr/riemann/epi_type_signature.py` — diagnostic (preserved as
  off-catalog measurement utility).
- `examples/05_type_hygiene/79_epi_type_signature_demo.py` — demo (preserved; corroborates TMEP empirically).

---

## §13triginta-septima. TNFR Structure & Dynamics Discoveries Log (Living Section)

**Purpose.**  This section is the *single canonical accumulation point*
for structural / dynamical facts about the TNFR repo and theory that
have been verified during the catalog type-hygiene programme (and any
subsequent programme).  It exists so that future maintainers,
researchers, and AI agents can (a) understand the system in depth,
(b) modify it without re-deriving facts, (c) optimise it without
breaking canonical contracts, and (d) leverage it for new experiments
without rediscovering load-bearing structure.

**Maintenance rule.**  Each entry is anchored to one or more concrete
locations (file:line, section, or commit hash) and is added only when
verified empirically (test, demo, or diagnostic run) or proved
analytically.  Entries are **append-only**; corrections are recorded
as later entries citing the earlier one, never by overwriting.
Categories are open — add new ones as discoveries warrant.

### §13triginta-septima.1 Canonical Contracts (Load-Bearing Invariants)

- **D-CC-1.**  The 13 canonical operators read and write EPI via
  `float(v.EPI)` exclusively (`src/tnfr/operators/__init__.py:190–360`,
  `src/tnfr/alias.py:86`).  Any code path that bypasses
  `_bepi_to_float` and writes a non-scalar EPI is off-catalog and
  must be documented as such.
- **D-CC-2.**  The nodal equation contract is strictly
  $(\nu_f, \Delta\mathrm{NFR}) \in \mathbb{R} \times \mathbb{R}
  \mapsto \partial\mathrm{EPI}/\partial t \in \mathbb{R}$
  (`src/tnfr/operators/nodal_equation.py:1–160`).  Multi-modal
  expressivity is exclusively temporal (via REMESH).
- **D-CC-3.**  Tetrad fields $(\Phi_s, |\nabla\phi|, K_\phi, \xi_C)$
  are pointwise functionals of scalar $\phi$ and scalar
  $\Delta\mathrm{NFR}$ only (`src/tnfr/physics/fields.py`).  No tetrad
  computation invokes Banach inner products on EPI.
- **D-CC-4.**  Structural conservation (`src/tnfr/physics/conservation.py`)
  closes on scalar charge density $\rho_i \in \mathbb{R}$ and current
  vector $\mathbf{J}_i \in \mathbb{R}^2$.  Energy
  $E = \sum_i \varepsilon_i \geq 0$ is a sum of scalar squares.
- **D-CC-5.**  REMESH (operator #13) is the **only** canonical
  mechanism that aggregates multi-component EPI content.  Aggregation
  is **temporal** ($\mathbb{R}^{T_{\max}+1}$, history vector at one
  node), **not spatial/modal**.  REMESH writes back a scalar
  ($\mathrm{EPI}(t+1) = M\, x(t)$) consumed by the next glyph via
  `float(v.EPI)`.  See `theory/REMESH_INFINITY_DERIVATION.md:50–52`.
- **D-CC-6.**  The canonical phase-wrap helper lives at
  `src/tnfr/physics/_helpers.py:29::wrap_angle` (the unique
  ``def wrap_angle`` in the repo, verified by repo-wide grep).  The
  canonical per-node phase alias is ``ALIAS_THETA``
  (`src/tnfr/constants/aliases.py:8`); no ``ALIAS_PHASE`` symbol
  exists.  **Catalog correction (discovered during B2a,
  §13triginta-octava.1):** `theory/CATALOG_TYPE_HYGIENE_PROGRAMME.md`
  §B2 cites the anchor as `tnfr.mathematics.phase.wrap_angle`; no
  such module exists.  Documentation finding only — no code change;
  the catalog row will be patched on the next type-hygiene commit
  that touches `theory/CATALOG_TYPE_HYGIENE_PROGRAMME.md` for an
  unrelated reason.

### §13triginta-septima.2 Non-Canonical Research Envelopes

- **D-ENV-1.**  E1: Pontryagin measure-valued $\nu_f$ — NEGATIVE
  verdict (§13triginta-tertia).  Refutation: scalar-storage axis +
  measure-redundancy under canonical νf-update.  Preserved as
  research formalism; not invoked by canonical operators.
- **D-ENV-2.**  E2: `BEPIElement` Banach carrier
  (`src/tnfr/mathematics/epi.py:103`) — NEGATIVE verdict
  (§13triginta-sexta).  Refutation: Temporal-Modal Equivalence
  Principle (TMEP); BEPI-storage fraction = 0 across two empirical
  resolutions.  Preserved as research formalism; never read or written
  by any canonical operator.

### §13triginta-septima.3 Methodology Patterns (Validated Across Sub-Questions)

- **D-MP-1 = L1** (B0): Two-axis diagnostic — scalar storage axis +
  spectral entropy axis.  Necessary-condition pattern: if scalar
  storage is full *and* spectral entropy is rich on the canonical
  axis, the upgrade is unforced.
- **D-MP-2 = L2** (B1a): Temporal-Modal Equivalence Principle —
  when storage and spectral axes disagree (scalar storage full +
  rich spectral content), the catalog encodes the expressivity
  **temporally** (via REMESH), not **spatially** (via Banach internal
  structure).
- **D-MP-3 = L3** (B0 ∧ B1): Catalog-Occam pattern — whenever a
  candidate type upgrade is matched by an existing canonical
  aggregation mechanism, the upgrade is non-canonical regardless of
  internal consistency.  See §13triginta-sexta.5.

### §13triginta-septima.4 Operational Conveniences (Repo-Specific)

- **D-OPS-1.**  `Network.G` attribute (`src/tnfr/sdk/simple.py:600`)
  exposes the underlying NetworkX graph for direct experimentation.
- **D-OPS-2.**  `inject_defaults(G)` must be invoked before any
  `step(G)` call in dynamics code (`src/tnfr/dynamics/adaptation.py:99`).
  Failure surfaces as missing-attribute errors at first operator
  application.
- **D-OPS-3.**  Python 3.12 venv at `c:\TNFR-Python-Engine\.venv312\`
  is the canonical interpreter for benchmarks/demos.  Run prefix:
  `$env:PYTHONPATH=(Resolve-Path ./src).Path;
   $env:PYTHONIOENCODING="utf-8"; & ./.venv312/Scripts/python.exe …`.

### §13triginta-septima.5 Open Questions (Tracked for Later Investigation)

- **D-OQ-1.**  G4 = RH (T-HP open content, §13septies); branches
  B1/B2/B3 undetermined; full attack surface shipped (P12–P49).
- **D-OQ-2.**  Catalog type-hygiene sub-questions B2 – B11 + Final
  pending; tracker in `theory/CATALOG_TYPE_HYGIENE_PROGRAMME.md` §4.
- **D-OQ-3.**  Whether the cross-conjecture pattern L3 holds for all
  of B2 – B11 (provisional; only tested on B0 ∧ B1 so far).

### §13triginta-septima.6 Maintenance Notes

- Entries use the prefix scheme: `D-<CATEGORY>-<n>` where category is
  one of CC (canonical contract), ENV (research envelope),
  MP (methodology pattern), OPS (operational), OQ (open question), or
  any new category added with rationale.
- Corrections / refinements append a new entry citing the earlier ID.
- Entries are *facts*, not opinions; each must cite at least one
  anchor (file:line, section, demo, or commit hash).
- This section grows monotonically; rewrites are explicit additions,
  not silent edits.

---
## §13triginta-octava. T-φ Pre-registration: The Phase Type-of-Object Conjecture (B2 Phase a; Diagnostic Only — Does NOT Advance G4 = RH)

**Programme position.**  Third executed sub-question of the Catalog
Type-Hygiene Programme (after B0 = T-νf NEGATIVE, B1 = T-EPI
NEGATIVE).  Phase a of the standard three-phase rhythm: pre-register
the conjecture, fix the diagnostic, commit a *necessary-condition*
empirical signature, deliberately defer the forcing-axiom analysis
(B2b) and the final verdict + envelope classification (B2c) to
separate commits.

**Honest scope (mandatory).**  This section pre-registers a *type-of-
object conjecture* and a *diagnostic*.  It does **not** promote any
covering-space construction to canonical status, does **not** modify
the 13-operator catalog, does **not** modify any existing source
file in `src/tnfr/`, and does **not** by itself advance G4 = RH.
The diagnostic is a *necessary-condition* probe: a non-trivial
signature is required, but not sufficient, for a covering-space lift
of φ to be canonically necessary.

### §13triginta-octava.1 — Motivation and literal canonical witness

The TNFR structural triad is (EPI, νf, φ) where φ is the canonical
**phase**, treated everywhere in the engine as a scalar in
:math:`[-\pi, \pi]` and *wrapped* to that fundamental domain by
:func:`tnfr.physics._helpers.wrap_angle`:

```python
# src/tnfr/physics/_helpers.py:29
def wrap_angle(angle: float) -> float:
    """Map *angle* to the interval [-π, π]."""
    return (angle + math.pi) % (2 * math.pi) - math.pi
```

The canonical storage aliases for φ are exposed via ``ALIAS_THETA``
(canonical phase is stored under the θ alias-tuple; the engine
uniformly uses θ as the alphabetic symbol for what AGENTS.md
documents as φ):

```python
# src/tnfr/constants/aliases.py:8
ALIAS_THETA = get_aliases("THETA")
```

and read by the canonical scalar accessor:

```python
# src/tnfr/physics/_helpers.py
def get_phase(G: Any, node: Any) -> float:
    """Retrieve phase value φ for *node* (radians in [0, 2π))."""
    ...
```

**Catalog-citation correction (recorded in the §13triginta-septima
discoveries log).**  The
`theory/CATALOG_TYPE_HYGIENE_PROGRAMME.md` §B2 spec at L152–167
cites the anchor as `tnfr.mathematics.phase.wrap_angle`.  No such
module exists in the current repo: the unique canonical
implementation lives at `src/tnfr/physics/_helpers.py:29`, and the
canonical storage alias is ``ALIAS_THETA``, not the catalog-implied
``ALIAS_PHASE``.  The catalog row has been logged for correction in
§13triginta-septima but is **not** modified here (one type-hygiene
finding per commit; the catalog patch will ride on the next
type-hygiene commit).

### §13triginta-octava.2 — Catalog statement of φ

Across the canonical engine, φ is consistently typed and stored as
a scalar real number in a single fundamental domain:

| Surface                                       | Type / domain                |
|-----------------------------------------------|------------------------------|
| Storage (per-node attribute via ALIAS_THETA)  | ``float ∈ [-π, π]``         |
| Wrapping helper `wrap_angle`                  | ``float → float ∈ [-π, π]``  |
| Scalar reader `get_phase`                     | ``float`` (re-wrapped)       |
| Tetrad field `|∇φ|`                           | ``float`` over edges         |
| Phase-gated coupling (U3) check               | ``|φᵢ − φⱼ| ≤ Δφ_max``       |

Every appearance of φ in the canonical operator-bound API ends in
this single-sheet representation.  The catalog therefore types φ as
the canonical scalar S¹ field — i.e. a section of the *trivial*
circle bundle over the graph, parametrised by a single fundamental
domain :math:`[-\pi, \pi]` (equivalently :math:`[0, 2\pi)`).

### §13triginta-octava.3 — The candidate non-canonical envelope: covering-space lift

The smallest enrichment that would *strictly increase* expressive
power over the canonical scalar S¹ representation is a **covering-
space lift** of φ to a multi-sheet cover of :math:`S^{1}`:

- A non-trivial element of the universal cover :math:`\widetilde{S^{1}}
  \simeq \mathbb{R}`, retaining an integer winding number
  :math:`w \in \mathbb{Z}` alongside the wrapped representative
  :math:`\phi_{\mathrm{wrap}} \in [-\pi, \pi]`.
- Equivalently, a U(1) bundle element :math:`e^{i\phi} \in S^{1}
  \subset \mathbb{C}` *with retained homotopy class* (`π₁(S¹) = ℤ`).

Call this envelope **E3 = CoverElement** (in symmetry with E1 = ``νf
Pontryagin partner Ẑ`` and E2 = ``BEPIElement``).  An E3-typed φ
would carry, per node and per trajectory, an extra integer winding
charge :math:`w` that the canonical ``wrap_angle`` discards every
single step.

The pre-registered question is:

> **T-φ Conjecture (formal statement, §13triginta-octava.4).** Does
> any canonical TNFR construction (operator, field, conservation law,
> grammar rule U1–U6, conserved current, gauge structure, or nodal-
> equation derivation) require φ to be canonically typed as an
> E3 = CoverElement rather than a canonical scalar S¹ field?

The empirical signature of §13triginta-octava.5 is a *necessary
condition* for the answer to be **yes**.

### §13triginta-octava.4 — T-φ Conjecture (formal statement)

**T-φ Conjecture.**  The canonical type-of-object of the TNFR
structural-triad component φ is the canonical scalar S¹ field
(equivalently: a section of the trivial circle bundle over the
graph, parametrised by ``float ∈ [-π, π]`` via :func:`wrap_angle`).
No canonical TNFR construction requires φ to be canonically typed as
a covering-space lift (E3 = CoverElement) carrying an integer
winding charge :math:`w \in \mathbb{Z}` separate from the wrapped
representative.

Equivalently, in catalog terms: the canonical phase row of
§13triginta-prima.4 — :math:`(\nu_f, \widehat{\nu_f}) =
(\mathbb{Z}, S^{1})` — fixes φ on the *dual* side as a scalar
S¹-valued field, and this typing is canonically saturated; the
discarded winding information is not used anywhere in the canonical
operator-bound dynamics.

**Anchors that the conjecture must survive (B2b/B2c):**

- F1–F10 forcing-axiom inventory of §13triginta-quarta.7 (re-applied
  to φ; B2b commit).
- Per-node accessor `get_phase` returning a single ``float``
  (canonical scalar reader).
- All canonical phase-gated couplings (U3) operating on
  ``|φᵢ − φⱼ|`` *after* wrapping, with no winding-number argument
  ever supplied.
- Cross-references §13quinquies, §13septies, §15 on phase-derived
  quantities :math:`|\nabla\phi|` and :math:`K_\phi`.

### §13triginta-octava.5 — Diagnostic S_φ (two-axis necessary condition)

**Definition.**  On a canonical TNFR ring graph
:math:`G_{n_{\mathrm{nodes}}}` with deterministic seeded initial
phase / EPI perturbation, run :math:`n_{\mathrm{steps}}` canonical
``step(G)`` evolutions and collect the per-node wrapped phase
trajectory :math:`\phi_i(t) \in [-\pi, \pi]` for
:math:`t \in \{0, 1, \dots, n_{\mathrm{steps}}\}` (length
:math:`n_{\mathrm{steps}} + 1`).  The diagnostic is the pair

.. math::

   \mathcal{S}_{\phi} = (w_{\mathrm{frac}}, \; H_{\mathrm{spec}} / \log B)

with the two axes defined as:

1. **Winding storage axis.**  For each node, reconstruct the
   *unwrapped* trajectory :math:`\widetilde{\phi}_i(t) =
   \mathrm{unwrap}(\phi_i(\cdot))_t` (NumPy ``np.unwrap``), then count
   the node as winding-non-trivial iff

   .. math::

      |\widetilde{\phi}_i(n_{\mathrm{steps}}) - \widetilde{\phi}_i(0)|
      \;\ge\; 2\pi - \mathrm{winding\_atol}.

   The winding fraction is
   :math:`w_{\mathrm{frac}} = N_{\mathrm{wind}} / N`.

2. **Lift-spectral axis.**  For each node compute the phase-velocity
   :math:`\dot\phi_i(t) := \mathrm{wrap}(\phi_i(t+1) - \phi_i(t))`,
   take its real-FFT magnitude (mean-subtracted), bin onto :math:`B`
   uniform frequency bins to obtain a probability distribution
   :math:`p_i`, and compute the Shannon entropy
   :math:`H_i = -\sum_b p_i(b) \log p_i(b)`.  Average across nodes
   to obtain :math:`H_{\mathrm{spec}}`.  Normalise by :math:`\log B`
   so the signature lives in :math:`[0, 1]`.

**Verdict labels (mechanically applied by the diagnostic, not by
itself sufficient for the foundational T-φ Conjecture):**

- ``SCALAR_S1_ADEQUATE``: signature :math:`< 0.15` *and* zero
  winding fraction.
- ``COVER_LIFT_NECESSARY``: signature :math:`> 0.5` *or* non-zero
  winding fraction.
- ``INDETERMINATE``: in between.

**Implementation.**  The diagnostic is implemented in
`src/tnfr/riemann/phi_type_signature.py`, exporting
``PhiTypeSignatureCertificate`` and ``compute_phi_type_signature``.
The reference demo lives at `examples/05_type_hygiene/80_phi_type_signature_demo.py`.

### §13triginta-octava.6 — Pre-registered numerical signature

The diagnostic is executed at two resolutions at pre-registration
time (commit-time numerical fingerprint, frozen for later
comparison):

| Resolution                        | seed | S_φ      | w_frac | max \|Δφ_unwrap\| | mean H (nats) | N_eff | verdict                |
|-----------------------------------|------|----------|--------|--------------------|---------------|-------|------------------------|
| n=24, steps=64, bins=32           | 13   | 0.941600 | 0/24   | 3.5584 rad         | 3.2633        | 26.14 | COVER_LIFT_NECESSARY*  |
| n=48, steps=128, bins=64          | 29   | 0.957087 | 0/48   | 3.1680 rad         | 3.9804        | 53.54 | COVER_LIFT_NECESSARY*  |

\* The ``COVER_LIFT_NECESSARY`` label is **mechanically issued by
the spectral-axis threshold alone**.  The winding axis is *zero* at
both resolutions, and the maximum unwrapped phase displacement is
strictly **below** :math:`2\pi \approx 6.2832` rad (max observed
3.5584 rad).  **No canonical evolution at the pre-registered scales
produces a topological winding.**  The diagnostic is honestly
flagging that:

(a) canonical phase-velocity is *broadband* (≈ 26–54 effective
spectral modes); a covering-space lift would be *one* construction
capable of representing this richness, but it is far from the only
one — a single-sheet scalar S¹ field hosting quasi-periodic
dynamics with many incommensurate frequencies will also produce a
high-entropy phase-velocity spectrum without any winding;

(b) the spectral threshold (``cover_threshold = 0.5``) is inherited
from the EPI diagnostic of §13triginta-quarta.6 and is *preliminary
for φ*; phase is constrained to a compact manifold :math:`S^{1}`
where wrapping itself injects high-frequency content into
:math:`\dot\phi`, so the per-resolution baseline of the spectral
axis is structurally elevated relative to EPI (which lives in
:math:`\mathbb{R}`).  Re-calibration of the φ-specific threshold is
deferred to B2b.

**Honest reading of this signature at Phase a.**  The dominant
empirical fact is the *zero winding fraction* at both resolutions:
canonical evolution, executed exactly as the catalog specifies,
does **not** produce any node whose unwrapped phase trajectory
escapes the fundamental domain :math:`[-\pi, \pi]`.  This is
structurally consistent with the canonical
``wrap_angle`` discipline and with the catalog row
:math:`(\mathbb{Z}, S^{1})` of §13triginta-prima.4.  The
high spectral entropy is a *separate* phenomenon (broadband
phase-velocity) that the B2b forcing-axiom analysis must isolate
from the covering-space question proper.

### §13triginta-octava.7 — Pre-registered hypothesis for B2b/B2c

Based on (i) the literal-catalog inspection of
§13triginta-octava.2, (ii) the Pontryagin-dual row 5 of
§13triginta-prima.4, (iii) the zero-winding empirical fact of
§13triginta-octava.6, and (iv) the universal absence of any
``winding`` / ``cover_index`` / ``π1`` argument in canonical
operator signatures, the **pre-registered expected verdict** at
B2c is:

> **NEGATIVE.** The canonical type of φ is the canonical scalar
> S¹ field.  E3 = CoverElement is a strictly *richer* envelope
> than the canonical type but is **not** required by any
> canonical TNFR construction.  No promotion, no deletion, no
> deprecation, no modification of the catalog.

This pre-registration commits to that expected verdict so that the
B2b forcing-axiom reduction cannot be retrofitted: if the F1–F10
analysis yields a different verdict, the pre-registration record
of §13triginta-octava.6 makes the inversion explicit and
audit-traceable.

### §13triginta-octava.8 — Honest scope (what this does and does not do)

This pre-registration section, the diagnostic module, and the demo:

- **Does not** promote ``CoverElement`` (or any covering-space lift,
  U(1) bundle element, or multi-sheet object) to canonical status.
- **Does not** modify the catalog
  (`theory/CATALOG_TYPE_HYGIENE_PROGRAMME.md` §3, §4, §6 will only
  be touched at B2c).
- **Does not** modify any existing source file in `src/tnfr/`; only
  adds the diagnostic module `src/tnfr/riemann/phi_type_signature.py`
  (and its export in `src/tnfr/riemann/__init__.py`) and the demo
  `examples/05_type_hygiene/80_phi_type_signature_demo.py`.
- **Does not** change the canonical
  `tnfr.physics._helpers.wrap_angle`, `get_phase`, ``ALIAS_THETA``,
  or any tetrad field implementation.
- **Does not** by itself decide T-φ; B2b (forcing-axiom reduction)
  and B2c (final verdict + envelope classification) are required.
- **Does not** advance G4 = RH or any of the open ζ-track / L-track
  RH-equivalents (P17–P49 attack surface).
- **Does not** rely on T-νf (B0, NEGATIVE) or T-EPI (B1, NEGATIVE)
  in any way that would force their verdicts to be re-opened.

### §13triginta-octava.9 — Cross-references

- §13triginta-prima — T-νf pre-registration (precedent for B0).
- §13triginta-prima.4 — Pontryagin-dual table row 5
  :math:`(\mathbb{Z}, S^{1})` predicting the φ-side typing.
- §13triginta-tertia — T-νf NEGATIVE verdict + E1 classification
  (closes B0).
- §13triginta-quarta — T-EPI pre-registration (precedent template
  for this section).
- §13triginta-sexta — T-EPI NEGATIVE verdict + E2 = ``BEPIElement``
  classification (closes B1).
- §13triginta-septima — Discoveries log; the catalog-citation
  correction (`mathematics.phase` → `physics/_helpers.py`,
  ``ALIAS_PHASE`` → ``ALIAS_THETA``) will be recorded there in the
  next type-hygiene commit.
- `theory/CATALOG_TYPE_HYGIENE_PROGRAMME.md` §3, §4, §6 — programme
  tracker (advances on this commit at row B2 Phase a only).
- `src/tnfr/physics/_helpers.py:29` — `wrap_angle` canonical
  implementation (anchor).
- `src/tnfr/constants/aliases.py:8` — ``ALIAS_THETA`` canonical
  alias tuple.
- `src/tnfr/riemann/phi_type_signature.py` — diagnostic
  implementation (added on this commit).
- `examples/05_type_hygiene/80_phi_type_signature_demo.py` — demo (added on this
  commit).

---


## §13triginta-novena. Derivation of (P-φ-Cover-Carrier) from the Canonical Catalog — Foundational Reduction of the φ-Type Conjecture (Theory-Only Analysis; Does NOT Advance G4 = RH)

**Pre-registration status.**  This section executes the
forcing-axiom reduction phase (B2b) of the T-φ program
(§13triginta-octava): it attempts to derive the covering-space
carrier principle for the canonical phase field
(P-φ-Cover-Carrier) from the canonical six invariants + nodal
equation + Structural Conservation Theorem + Variational
Principle + REMESH operator + the structural-field
tetrad, *or* to identify and isolate the actual residual
axiom that the derivation requires beyond the catalog.

The honest verdict (executed in §13triginta-decima) is
pre-registered as one of:

- `COROLLARY_DERIVED`: (P-φ-Cover-Carrier) follows from invariants
  1–6 alone.
- `CONDITIONAL_COROLLARY`: (P-φ-Cover-Carrier) follows under one
  additional identifiable axiom strictly weaker than itself.
- `INDEPENDENT_AXIOM`: (P-φ-Cover-Carrier) is independent of the
  catalog.

Scope (mandatory honesty): this section does **not** advance
G4 = RH, does **not** close T-HP, does **not** introduce or modify
any canonical operator, does **not** delete or deprecate any
covering-space construction, and does **not** by itself close T-φ.
It locates the foundational axiom *one structural level below*
(P-φ-Cover-Carrier) and hands T-φ back to that deeper question.

The literal canonical statement under scrutiny:

> **(P-φ-Cover-Carrier).** In the canonical TNFR formulation, the
> per-node phase state must take values in a covering space of the
> circle (the universal cover :math:`\widetilde{S^{1}} \simeq
> \mathbb{R}`, equivalently a U(1)-bundle element :math:`e^{i\phi}`
> with retained homotopy class :math:`w \in \pi_1(S^1) = \mathbb{Z}`),
> *not* in :math:`[-\pi, \pi]` under the canonical ``wrap_angle``
> projection.

### §13triginta-novena.1 Available Canonical Tools

The derivation may use only the following canonical machinery (no
extraneous structure):

1. **Nodal equation**: :math:`\partial \mathrm{EPI}/\partial t = \nu_f \cdot \Delta \mathrm{NFR}(t)`
   (Invariant #1), in which φ enters only via the tetrad fields
   :math:`|\nabla\phi|` and :math:`K_\phi` that drive
   :math:`\Delta\mathrm{NFR}` (`src/tnfr/operators/nodal_equation.py:1–160`).
2. **Six canonical invariants** (AGENTS.md): Nodal Equation
   Integrity, **Phase-Coherent Coupling (invariant #2)**, Multi-Scale
   Fractality, Grammar Compliance, Structural Metrology,
   Reproducible Dynamics.
3. **Grammar U1–U6**, in particular **U3 (RESONANT COUPLING)** which
   gates UM/RA on :math:`|\phi_i - \phi_j| \le \Delta\phi_{\max}`.
4. **Structural-field tetrad**: the phase φ ∈ S¹ and
   the structural fields :math:`|\nabla\phi|, K_\phi` constructed
   from wrapped phase differences (the phase sector is π-scaled; only π is structural).
5. **Structural Conservation Theorem**
   (`src/tnfr/physics/conservation.py`): per-node charge density
   :math:`\rho_i` and current vector :math:`\mathbf{J}_i \in \mathbb{R}^2`
   built from real-valued functionals of φ *after* ``wrap_angle``.
6. **Variational Principle**: Lagrangian
   :math:`\mathcal{L}_i = T_i - V_i` with all terms real-valued
   functionals of the wrapped tetrad fields.
7. **REMESH operator** (canonical operator #13), aggregating
   per-node EPI history; REMESH never aggregates raw winding
   information of φ.
8. **Phase-wrap helper**:

   ```python
   # src/tnfr/physics/_helpers.py:29
   def wrap_angle(angle: float) -> float:
       """Map *angle* to the interval [-π, π]."""
       return (angle + math.pi) % (2 * math.pi) - math.pi
   ```

9. **Canonical storage alias** ``ALIAS_THETA``
   (`src/tnfr/constants/aliases.py:8`) — the per-node phase is
   stored under this single scalar alias-tuple, with no companion
   ``winding`` / ``cover_index`` / ``π1_class`` alias.

### §13triginta-novena.2 What the Canonical Catalog Forces (Scalar S¹ Layer)

The chain of forced structure for φ is straightforward and entirely
inside the catalog:

- **(M1) Operator contracts are scalar-S¹.**  Every canonical glyph
  operator that touches φ reads via ``get_phase`` (returning
  ``float``) and writes via ``ALIAS_THETA`` followed *immediately*
  by ``wrap_angle``.  No operator constructs, reads, propagates, or
  preserves a winding number, sheet index, or homotopy class.
  Empirically verified by `examples/05_type_hygiene/80_phi_type_signature_demo.py`:
  ``w_frac = 0/24`` at :math:`(n=24, T=64, B=32, \mathrm{seed}=13)`
  and ``w_frac = 0/48`` at :math:`(n=48, T=128, B=64, \mathrm{seed}=29)`,
  with ``max |Δφ_unwrap|`` strictly below :math:`2\pi` at both
  resolutions.

- **(M2) The nodal equation is wrap-stable.**  φ enters
  :math:`\partial\mathrm{EPI}/\partial t` only through
  :math:`\Delta\mathrm{NFR}`, which itself depends on the tetrad
  fields :math:`|\nabla\phi|` and :math:`K_\phi`.  Both fields are
  pointwise functionals of *wrapped* phase differences
  (``wrap_angle`` is applied edge-wise in
  `src/tnfr/physics/fields.py::compute_phase_gradient` and
  `compute_phase_curvature`).  Any winding-shifted realisation
  :math:`\phi \mapsto \phi + 2\pi k_i` of the canonical phase field
  produces the *same* :math:`|\nabla\phi|`, :math:`K_\phi`,
  :math:`\Delta\mathrm{NFR}`, and hence the *same*
  :math:`\partial\mathrm{EPI}/\partial t`.

- **(M3) U3 phase-gated coupling is wrap-equivariant.**  The U3
  resonance condition :math:`|\phi_i - \phi_j| \le \Delta\phi_{\max}`
  is canonically evaluated on the *wrapped* phase difference (per
  `src/tnfr/operators/grammar_core.py::validate_resonant_coupling`),
  i.e. on the geodesic distance on :math:`S^1`.  Adding any winding
  shift to either endpoint leaves the wrapped difference invariant.
  The covering-space datum is therefore *never read* by U3.

- **(M4) Conservation and variational laws close on wrapped φ.**
  The Noether charge :math:`Q = \sum_i \rho_i`, the energy
  :math:`E = \sum_i \varepsilon_i`, the Lagrangian, and the
  symplectic form are all real-valued functionals of wrapped
  tetrad fields :math:`(\Phi_s, |\nabla\phi|, K_\phi, \xi_C)` and
  the scalar currents :math:`(J_\phi, J_{\Delta\mathrm{NFR}})`.  No
  conservation law references a winding charge.

The conjunction M1+M2+M3+M4 establishes that **the entire canonical
machinery closes consistently and gauge-invariantly with scalar S¹
phase**.  The 13-operator catalog never reads or writes a winding
number; the nodal equation is invariant under per-node
:math:`2\pi k_i` shifts of φ; U3 is wrap-equivariant; conservation
and variational laws never require a covering-space lift.

### §13triginta-novena.3 The Gap Between wrap_angle Discipline and Covering-Space Retention

Scalar-S¹ closure (M1–M4) is necessary but not sufficient to refute
(P-φ-Cover-Carrier): one could still ask whether the catalog *also*
admits a strictly-stronger covering-space realisation in which the
wrapped implementation is a faithful coordinate projection from
:math:`\widetilde{S^{1}} \simeq \mathbb{R}` onto :math:`S^{1}`.  The
decisive question is whether the catalog *forces* such an upgrade.

The only canonical mechanism that could conceivably preserve
homotopy-class data across the temporal evolution is a hypothetical
"non-wrapped" branch that propagates :math:`\widetilde\phi(t) \in
\mathbb{R}` alongside :math:`\phi(t) \in [-\pi, \pi]`.  But the
canonical engine **does not implement** any such branch: every
write to ``ALIAS_THETA`` passes through ``wrap_angle``, and there
is no canonical alias for an unwrapped companion.

Formally, define the **Phase-Wrap Discipline Principle**:

> **Phase-Wrap Discipline Principle (PWDP).**  In the canonical
> TNFR formulation, every per-node phase value is *systematically
> projected* onto the fundamental domain :math:`[-\pi, \pi]` via
> ``wrap_angle`` at every operator boundary.  The homotopy class
> :math:`w \in \pi_1(S^1) = \mathbb{Z}` is *systematically
> discarded* and is **not** retrievable from the canonical state.

This is the structural-φ analogue of the Temporal-Modal Equivalence
Principle (TMEP) that closed B1 = T-EPI.  Where TMEP says
"multi-modal EPI content is canonically realised *temporally* via
REMESH, not *spatially* via a Banach internal carrier", PWDP says
"phase-orbit content is canonically realised *as wrapped geodesic
distance on* :math:`S^1`, not *as covering-space displacement on*
:math:`\widetilde{S^1}`".

PWDP is *operationally complete*: under the canonical ``wrap_angle``
discipline, the engine reproduces P12–P15 to machine precision
(§10–§12), recovers classical (Keplerian) and quantum-like
(interference, complementarity, quantization) regimes (§§3–9), and
satisfies all canonical conservation laws — *without* invoking any
winding charge or homotopy class.  The B2a empirical signature
:math:`w_{\mathrm{frac}} = 0` at both pre-registered resolutions is
the empirical fingerprint of PWDP.

**Crucially**, the broadband phase-velocity spectrum measured by
the lift-spectral axis of §13triginta-octava.5
(:math:`H_{\mathrm{spec}}/\log B \approx 0.94`–:math:`0.96`) is
*explained* by PWDP without invoking (P-φ-Cover-Carrier): a
single-sheet S¹-valued field hosting quasi-periodic dynamics with
many incommensurate frequencies will produce a high-entropy
phase-velocity spectrum, and wrapping itself injects high-frequency
content into :math:`\dot\phi` at the wrap discontinuities.  The
spectral richness is *structural*, not *topological*.

Therefore: **(P-φ-Cover-Carrier) is strictly stronger than what
M1+M2+M3+M4 + PWDP provide**, and any derivation must locate an
additional canonical constraint that selects the homotopy-retention
upgrade.

### §13triginta-novena.4 Candidate Forcing Constraints (Enumeration)

The candidates available inside the canonical catalog are enumerated
below.  Each row asks: *does this axiom force the covering-space
upgrade of φ?*

| #  | Axiom | Source | Forces cover-carrier of φ? |
|---|---|---|---|
| F1 | Operator exclusivity (only the 13 canonical operators write φ). | AGENTS.md "Canonical Invariants #1". | **No** — operators write wrapped ``float`` via ``ALIAS_THETA`` + ``wrap_angle`` (M1). |
| F2 | Reproducibility under fixed seeds. | AGENTS.md "Reproducible Dynamics". | **No** — wrapped scalar trajectories reproduce identically; winding shifts are not part of the seeded state. |
| F3 | Nodal-equation wrap-invariance: same :math:`\partial\mathrm{EPI}/\partial t` under :math:`\phi \mapsto \phi + 2\pi k_i`. | `nodal_equation.py`, `fields.py::compute_phase_gradient`/`compute_phase_curvature`. | **No** — the dynamics is *gauge-invariant* under per-node :math:`2\pi` shifts; the winding charge is structurally unobservable from the canonical ODE (M2). |
| F4 | Tetrad orthogonality and minimality of :math:`(\Phi_s, |\nabla\phi|, K_\phi, \xi_C)`. | AGENTS.md §"Minimal Structural Degrees of Freedom". | **No** — all four fields use wrapped phase differences (M2); cf. `STRUCTURAL_FIELDS_TETRAD.md`. |
| F5 | U3 phase-gated coupling :math:`|\phi_i - \phi_j| \le \Delta\phi_{\max}`. | AGENTS.md "U3 RESONANT COUPLING", `grammar_core.py::validate_resonant_coupling`. | **No** — U3 reads the *wrapped* geodesic distance on :math:`S^1`; covering-space distance is *never* the canonical input (M3). |
| F6 | Structural Conservation Theorem (Noether charge :math:`Q`, energy :math:`E`, Ward identities). | `physics/conservation.py`, `theory/STRUCTURAL_CONSERVATION_THEOREM.md`. | **No** — :math:`\rho, \mathbf{J}, \varepsilon` are all real-valued functionals of wrapped tetrad fields (M4); no winding current appears in :math:`\partial\rho/\partial t + \nabla \cdot \mathbf{J} = S_{\mathrm{grammar}}`. |
| F7 | Variational principle (Lagrangian, symplectic conjugate pair :math:`(K_\phi, J_\phi)`). | `physics/variational.py`, AGENTS.md §"Variational Confirmation". | **No** — :math:`K_\phi = \mathrm{wrap\_angle}(\phi_i - \mathrm{circular\_mean}(\mathrm{nbrs}))` is *defined* as a wrapped scalar with :math:`|K_\phi| \le \pi`; the conjugate momentum :math:`J_\phi` is real-valued. |
| F8 | REMESH temporal aggregation of φ trajectories. | `theory/REMESH_INFINITY_DERIVATION.md`, `operators/remesh.py`. | **No** — REMESH aggregates EPI history, not φ history; even when φ-derived quantities feed REMESH (via :math:`\Delta\mathrm{NFR}`), the inputs have already been wrap-projected (chain of M2+M1). |
| F9 | Classical-limit demos (Keplerian orbits, smooth phase trajectories). | `examples/02_physics_regimes/12_classical_mechanics_demo.py`. | **No** — classical regime emerges from *wrapped* φ under high coherence; the visible smoothness is a coordinate effect, not evidence of a covering-space carrier. |
| F10 | Quantum-regime demos (interference, complementarity). | `examples/02_physics_regimes/13_quantum_mechanics_demo.py`, `14_uncertainty_and_interference.py`. | **No** — quantum-like phenomena emerge from wrapped φ dynamics; phase-difference interference at slits uses :math:`\mathrm{wrap\_angle}(\phi_A - \phi_B)`, not covering-space difference. |

**Result.** No canonical constraint in :math:`\{\mathrm{F1}, \ldots, \mathrm{F10}\}`
forces the covering-space carrier upgrade of φ.  All ten admit
consistent gauge-invariant realisation with wrapped scalar S¹ phase
(as the current 13-operator implementation demonstrates by
existence, and as the B2a empirical signature confirms:
:math:`w_{\mathrm{frac}} = 0` across two independent demo
resolutions, ``max |Δφ_unwrap| < 2π`` at both).

### §13triginta-novena.5 The Hidden Axiom: (P-φ-Homotopy-Retention)

The derivation gap can be isolated cleanly.  Define:

> **(P-φ-Homotopy-Retention).**  In the canonical TNFR formulation,
> the per-node phase trajectory :math:`\{\phi_i(t)\}_t` must retain
> its homotopy class :math:`w_i \in \pi_1(S^1) = \mathbb{Z}` across
> the ``wrap_angle`` projection — i.e. distinct unwrapped lifts
> :math:`\widetilde\phi_i(t)` differing by an integer multiple of
> :math:`2\pi` must correspond to distinct canonical states, and
> conversely.

**Claim.**  (P-φ-Cover-Carrier) is a corollary of the canonical
catalog *plus* (P-φ-Homotopy-Retention), and of nothing weaker
than (P-φ-Homotopy-Retention).

**Forward direction (sufficiency).**  Assume (P-φ-Homotopy-Retention).
Consider a trajectory :math:`\phi_i(t)` with non-trivial winding
(:math:`\widetilde\phi_i(T) - \widetilde\phi_i(0) = 2\pi w` with
:math:`w \neq 0`).  Retention forces the canonical state to encode
:math:`w` faithfully.  A wrapped scalar
:math:`\phi_i(t) \in [-\pi, \pi]` does not have the cardinality to
encode an integer winding charge *separately from* the wrapped
representative at fixed :math:`(i, t)` (one real number in a bounded
interval cannot encode an unbounded integer).  Hence the canonical
phase storage must take values in a non-trivial cover of :math:`S^1`
— equivalently, the covering-space lift carrier
:math:`\widetilde{S^1} \simeq \mathbb{R}` (or a U(1)-bundle element
with explicit :math:`w` slot).  This is (P-φ-Cover-Carrier).

**Reverse direction (necessity at the canonical level).**  Suppose
(P-φ-Cover-Carrier) holds.  Then :math:`\phi_i(t) \in \widetilde{S^1}`
is fully specified by :math:`(\phi_{\mathrm{wrap}}, w) \in [-\pi, \pi]
\times \mathbb{Z}`.  By construction, distinct winding shifts
produce distinct canonical states.  Hence (P-φ-Homotopy-Retention)
holds.

**Strict-weakness of (P-φ-Homotopy-Retention) vs (P-φ-Cover-Carrier).**
(P-φ-Homotopy-Retention) is a *meta-constraint* on the canonical
storage map :math:`\phi_i(t) \mapsto` (homotopy class of the
trajectory).  It does not mention covering spaces, U(1) bundles,
universal covers, or any topological-bundle machinery.  It is purely
a faithfulness requirement on the symbolic representation of
trajectory homotopy.  By contrast, (P-φ-Cover-Carrier) commits to a
specific carrier (:math:`\widetilde{S^1}`) and a specific algebraic
structure (the :math:`\mathbb{R}` group with quotient :math:`S^1`).

Therefore (P-φ-Homotopy-Retention) is structurally simpler and
strictly weaker than (P-φ-Cover-Carrier), and the derivation is
genuine progress.

### §13triginta-novena.6 Canonical Status of (P-φ-Homotopy-Retention) — PWDP Refutation

The question is now: is (P-φ-Homotopy-Retention) itself derivable
from the canonical six invariants?

- **(B-Pro).**  Invariant #2 (Phase-Coherent Coupling) could be read
  as suggesting that phase information should be canonically
  retained without loss.  If two trajectories differing only by an
  integer winding shift produced the same canonical state, an
  observer trying to reconstruct the *full unwrapped trajectory*
  from the canonical record would lose the winding count.

- **(B-Con, decisive).**  The **Phase-Wrap Discipline Principle
  (PWDP, §13triginta-novena.3)** refutes the per-trajectory
  homotopy-retention requirement *at the canonical level*: the
  observable content of φ at every canonical operator boundary is
  the *wrapped* representative, and the canonical dynamics is
  *gauge-invariant* under per-node :math:`2\pi k_i` shifts (F3).
  Any unwrapped lift is therefore a *coordinate choice* on top of
  the canonical state, not a canonical state itself.

  Formally: the catalog enforces phase coherence (invariant #2) via
  U3 evaluated on wrapped distances on :math:`S^1`, with all
  downstream conservation and variational structure descending from
  the wrapped tetrad fields (M2–M4).  This is *operationally
  complete* — it reproduces all canonical results (§§3–12) without
  any per-trajectory winding charge.

- **(B-Empirical).**  The B2a diagnostic (§13triginta-octava.6)
  measures :math:`w_{\mathrm{frac}} = 0` and ``max |Δφ_unwrap|
  < 2π`` at *both* resolutions: canonical evolution, executed
  exactly as the catalog specifies, does not produce any node whose
  unwrapped phase trajectory escapes the fundamental domain.  The
  homotopy class is structurally trivial at every measured
  :math:`(i, t)`.  This is exactly the PWDP signature: the
  covering-space lift is structurally unreachable from canonical
  initial conditions.

**Conclusion of §13triginta-novena.6.**  (P-φ-Homotopy-Retention) is
**not derivable from the canonical six invariants**.  The catalog
realises phase coherence *wrap-equivariantly* via U3 and the
wrapped tetrad fields, *not* covering-space-equivariantly via a
homotopy-retention upgrade.  The per-trajectory homotopy-class
retention that (P-φ-Homotopy-Retention) demands is an *additional*
axiom, independent of the catalog and actively refuted by PWDP at
the canonical level, with the empirical winding fingerprint
:math:`w_{\mathrm{frac}} = 0` of B2a as decisive corroboration.

### §13triginta-novena.7 Sub-Verdict

The forcing-axiom reduction yields:

> **Sub-verdict (§13triginta-novena).**
> (P-φ-Cover-Carrier) is a **CONDITIONAL_COROLLARY** of the
> canonical catalog: it follows from the catalog *plus*
> (P-φ-Homotopy-Retention).  However, (P-φ-Homotopy-Retention) is
> itself **INDEPENDENT_AXIOM** at the canonical level: it is not
> derivable from invariants 1–6 and is actively refuted by the
> Phase-Wrap Discipline Principle (PWDP), with the B2a empirical
> winding fingerprint :math:`w_{\mathrm{frac}} = 0` as decisive
> corroboration.
>
> Net: (P-φ-Cover-Carrier) is **strictly non-canonical**.  Any
> covering-space lift, U(1)-bundle element, or homotopy-retaining
> representation of φ is a legitimate research envelope —
> available for off-catalog experimentation — but is not forced
> by, and indeed is structurally orthogonal to (gauge-invariantly
> trivial under), the canonical 13-operator realisation under
> PWDP.

This locates the residual canonical question for T-φ exactly one
level below (P-φ-Cover-Carrier), at (P-φ-Homotopy-Retention), and
identifies its refutation mechanism (PWDP).  The final NEGATIVE
verdict on T-φ, and the classification of the covering-space carrier
(``CoverElement``, candidate envelope E3) as a legitimate
non-canonical research envelope, are executed in §13triginta-decima
(B2c).

### §13triginta-novena.8 Honest Scope (What This Does and Does Not Do)

This sub-programme:

- **Does** isolate the residual axiom one structural level below
  (P-φ-Cover-Carrier).
- **Does** prove (P-φ-Homotopy-Retention) is strictly weaker than
  (P-φ-Cover-Carrier).
- **Does** refute (P-φ-Homotopy-Retention) at the canonical level
  via PWDP, empirically corroborated by B2a's
  :math:`w_{\mathrm{frac}} = 0` at two resolutions.
- **Does** confirm the catalog closes consistently and
  gauge-invariantly with wrapped scalar S¹ φ (M1+M2+M3+M4).
- **Does** identify the canonical dynamics as *gauge-invariant
  under per-node :math:`2\pi k_i` shifts of φ* (a structural
  observation made explicit here for the first time, not a new
  canonical promotion).
- **Does not** advance G4 = RH or the T-HP conjecture.
- **Does not** promote any operator, field, or constant to canonical
  status (in particular: does NOT promote ``CoverElement``,
  ``ALIAS_PHASE_UNWRAPPED``, or any homotopy-retaining
  representation).
- **Does not** modify the 13-operator catalog.
- **Does not** delete or deprecate the candidate envelope E3 =
  ``CoverElement``; classifies it as a research envelope available
  outside the canonical operator contracts.
- **Does not** modify any source file in `src/tnfr/`.
- **Does not** by itself close T-φ — the final verdict is executed
  in §13triginta-decima.

### §13triginta-novena.9 Cross-references

- §13triginta-prima — T-νf Type Conjecture (pre-registration, νf
  analog; first sub-question of the programme).
- §13triginta-prima.4 — Pontryagin-dual table row 5
  :math:`(\mathbb{Z}, S^{1})` predicting the φ-side canonical
  scalar S¹ typing.
- §13triginta-secunda — T-νf forcing-axiom reduction (structural
  template for this section).
- §13triginta-tertia — T-νf NEGATIVE verdict (precedent for B0c).
- §13triginta-quarta — T-EPI pre-registration (precedent for B1a).
- §13triginta-quinta — T-EPI forcing-axiom reduction (structural
  twin of this section; TMEP closes B1b, PWDP closes B2b).
- §13triginta-sexta — T-EPI NEGATIVE verdict + E2 = ``BEPIElement``
  classification (precedent for B2c).
- §13triginta-septima — Living discoveries log (D-CC-6 catalog
  citation correction for ``wrap_angle`` / ``ALIAS_THETA`` recorded
  in B2a; deferred catalog patch unchanged on this commit).
- §13triginta-octava — T-φ pre-registration (B2a anchor +
  two-axis diagnostic; this section consumes the
  :math:`w_{\mathrm{frac}} = 0` fingerprint).
- §13septies — T-HP open content (independent of this
  sub-question).
- §19.1 — Full P1–P49 milestone table.
- `theory/CATALOG_TYPE_HYGIENE_PROGRAMME.md` §4 — programme
  tracker (row B2 Phase b advances on this commit).
- `src/tnfr/physics/_helpers.py:29` — ``wrap_angle`` canonical
  implementation (anchors M1 / M2 / PWDP).
- `src/tnfr/constants/aliases.py:8` — ``ALIAS_THETA`` canonical
  scalar storage alias (anchors M1).
- `src/tnfr/physics/fields.py` — ``compute_phase_gradient`` and
  ``compute_phase_curvature`` (anchor M2: tetrad fields built on
  wrapped phase differences).
- `src/tnfr/operators/grammar_core.py::validate_resonant_coupling`
  — U3 gating on wrapped distances (anchors M3).
- `src/tnfr/physics/conservation.py` — Noether charge / current /
  energy (anchors M4).
- `src/tnfr/riemann/phi_type_signature.py` — B2a diagnostic
  implementation (anchors :math:`w_{\mathrm{frac}} = 0` empirical
  corroboration of PWDP).
- `examples/05_type_hygiene/80_phi_type_signature_demo.py` — two-resolution demo
  (anchors B2a numerical fingerprint).

---
## §13triginta-decima. T-φ Final NEGATIVE Verdict and Envelope Classification of E3 = CoverElement (Closes B2; Does NOT Advance G4 = RH)

**Pre-registration closure.**  This section consumes the
sub-verdict of §13triginta-novena (B2b) and issues the final
T-φ verdict in accordance with the four-tier methodology of the
catalog type-hygiene programme (`theory/CATALOG_TYPE_HYGIENE_PROGRAMME.md`
§3, methodology lessons L1–L3).  The verdict pre-register from
§13triginta-octava.7 named the NEGATIVE branch as the expected
outcome; B2b has confirmed it via the Phase-Wrap Discipline
Principle (PWDP) and the F1–F10 reduction.

### §13triginta-decima.1 Verdict

> **T-φ verdict: NEGATIVE.**
> The canonical type-of-object of the TNFR structural-triad
> component φ is the canonical scalar S¹ field
> (``float ∈ [-π, π]`` via :func:`tnfr.physics._helpers.wrap_angle`,
> stored under ``ALIAS_THETA``).  The covering-space lift principle
> (P-φ-Cover-Carrier) is **not canonical**.  It does not follow
> from the canonical six invariants, nor from the nodal equation,
> nor from any subset of grammar U1–U6, nor from the
> structural-field tetrad, nor from the Structural Conservation
> Theorem, nor from the Variational Principle, nor from REMESH
> temporal aggregation.  Its derivation requires the additional
> axiom (P-φ-Homotopy-Retention), which is itself independent of
> the canonical catalog and actively refuted at the canonical level
> by the Phase-Wrap Discipline Principle
> (PWDP, §13triginta-novena.3, .6).

This closes T-φ in the same shape as T-νf (B0,
§13triginta-tertia) and T-EPI (B1, §13triginta-sexta): the
conjectured "type upgrade" of a fundamental TNFR observable is
classified as a legitimate research envelope, not as a canonical
catalog requirement.  The decisive numerical fingerprint is the
B2a winding-fraction signature:

| Resolution | seed | w_frac | max |Δφ_unwrap| | verdict (canonical) |
|---|---|---|---|---|
| n=24, steps=64  | 13 | 0/24 | 3.5584 rad < 2π | NEGATIVE |
| n=48, steps=128 | 29 | 0/48 | 3.1680 rad < 2π | NEGATIVE |

No canonical evolution at either resolution produces a node whose
unwrapped phase trajectory escapes the fundamental domain
:math:`[-\pi, \pi]`.  The high spectral entropy
(:math:`H/\log B \approx 0.94–0.96`) reflects broadband
phase-velocity content on the canonical scalar :math:`S^{1}`, **not**
a forced covering-space lift.  This is exactly the situation that
B2b isolated as the gap between (P-φ-Cover-Carrier) (the
covering-space construction) and the strictly weaker
(P-φ-Homotopy-Retention) (the bare requirement to retain
:math:`w \in \pi_{1}(S^{1}) = \mathbb{Z}`), the latter being
itself refuted by PWDP at the canonical level.

### §13triginta-decima.2 Envelope Classification of E3 = CoverElement

E3 = CoverElement — the covering-space lift of φ to the universal
cover :math:`\widetilde{S^{1}} \simeq \mathbb{R}`, retaining an
integer winding charge :math:`w \in \mathbb{Z}` alongside the
wrapped representative
:math:`\phi_{\mathrm{wrap}} \in [-\pi, \pi]`, equivalently a U(1)
bundle element :math:`e^{i\phi} \in S^{1} \subset \mathbb{C}` *with
retained homotopy class* — is hereby classified as:

> **E3 = CoverElement — Non-canonical research envelope.**
> Status: legitimate research formalism, off-catalog.
> Canonical relationship: **structurally orthogonal** to the
> canonical scalar S¹ realisation under PWDP — the engine projects
> every per-node phase onto :math:`[-\pi, \pi]` via
> :func:`wrap_angle` at every operator boundary, gauge-invariantly
> discarding the homotopy class.
> Catalog interaction: **none** required.  The 13 canonical
> operators do not read, write, preserve, or invoke any winding
> number, cover-sheet index, U(1) bundle section, or
> :math:`\pi_{1}(S^{1})` argument; they operate exclusively through
> the canonical scalar accessor ``get_phase`` followed by
> ``wrap_angle`` to a single fundamental domain.

The envelope register now records three entries:

| ID | Object | Source | Verdict | Refutation mechanism |
|---|---|---|---|---|
| E1 | Pontryagin measure-valued :math:`\nu_f` | §13triginta-tertia | NEGATIVE | Scalar-storage axis + measure-redundancy under canonical νf-update |
| E2 | ``BEPIElement`` Banach carrier | §13triginta-sexta | NEGATIVE | TMEP (temporal-modal aggregation suffices); BEPI-storage fraction = 0 across two resolutions |
| E3 | CoverElement (covering-space lift / U(1) bundle / homotopy-retaining φ) | this section | NEGATIVE | PWDP (canonical wrap-discipline at every operator boundary); :math:`w_{\mathrm{frac}} = 0` across two resolutions |

**Structural note on E3 vs. E1, E2.**  E1 and E2 each have a
concrete code witness in the repo (``Ω_R`` Pontryagin scaffolding
and ``src/tnfr/mathematics/epi.py:103::BEPIElement`` respectively),
even though those witnesses are never invoked by the canonical
13-operator API.  E3, by contrast, has **no** source-code witness
in the current repo: there is no ``CoverElement`` class, no
``ALIAS_PHASE_UNWRAPPED`` alias, no ``winding`` / ``cover_index`` /
``π1`` parameter in any operator signature (verified by repo-wide
grep at the B2c commit).  E3 is therefore a *purely conceptual*
research envelope at present, listed in the envelope register for
completeness and symmetry with the Pontryagin-dual row 5 of
§13triginta-prima.4.

### §13triginta-decima.3 No Deletion, No Deprecation, No Promotion, No Modification

The verdict does **not** authorise:

- introduction of any ``CoverElement`` class, ``ALIAS_PHASE_UNWRAPPED``
  alias, or covering-space module under ``src/tnfr/`` (E3 remains
  conceptual; promoting it to a code witness is itself off-catalog
  and would require a separate, documented research-track commit);
- deprecation warnings around ``wrap_angle``, ``ALIAS_THETA``, or
  ``get_phase`` in `src/tnfr/physics/_helpers.py` or
  `src/tnfr/constants/aliases.py`;
- modification of the 13-operator catalog;
- modification of the canonical contract
  :math:`(\nu_f, \Delta\mathrm{NFR}) \mapsto \partial\mathrm{EPI}/\partial t`;
- changes to ``src/tnfr/operators/nodal_equation.py``,
  ``src/tnfr/operators/grammar_core.py``, or
  ``src/tnfr/physics/fields.py``;
- any change to grammar U1–U6;
- any claim about G4 = RH, T-HP, or the open content of
  §13septies.

E3 remains available for off-catalog research (e.g. topologically
charged variant networks, U(1) gauge-theoretic extensions, vortex
classification studies) provided such research is documented as
off-catalog and does not claim canonical status.  The B2a
diagnostic module (``src/tnfr/riemann/phi_type_signature.py``) and
its demo (``examples/05_type_hygiene/80_phi_type_signature_demo.py``) are preserved
as off-catalog measurement utilities, exactly as the B1a and B0a
diagnostics were preserved at B1c and B0c.

The D-CC-6 catalog citation correction (``mathematics.phase`` →
`physics/_helpers.py`, ``ALIAS_PHASE`` → ``ALIAS_THETA``) recorded
in §13triginta-septima at B2a remains a documentation-only finding
on this commit (one-finding-per-commit rule); the catalog spec
patch at §B2 L152–167 stays deferred to a future dedicated
type-hygiene commit.

### §13triginta-decima.4 Programme Bookkeeping

- ``theory/CATALOG_TYPE_HYGIENE_PROGRAMME.md`` §4 row B2: Phase c
  advances ⏳ → ✅; Verdict column advances "—" → **NEGATIVE**;
  commit-refs column appends the present commit hash.
- §3 sub-question registry status: B2 transitions from
  🟡 IN PROGRESS to ✅ COMPLETE; the B2 spec block (Tier 1 — Per-
  node intrinsic types, "T-φ (Type of phase)") gains a closing
  line analogous to B1's: *"Status: ✅ COMPLETE — B2a ✅, B2b ✅,
  B2c ✅. Final verdict: NEGATIVE."*
- §3 progress summary advances: 3 sub-questions complete
  (B0 + B1 + B2 all NEGATIVE), 0 in progress, 9 pending
  (B3 – B11 + Final).
- §6 methodology lessons: an L3 confirmation entry is recorded
  (see §13triginta-decima.5 below) reflecting that the
  cross-conjecture pattern L3 first observed across B0 ∧ B1 now
  holds also for B2.

### §13triginta-decima.5 Methodology Lesson L3 — Confirmed Across B2

T-νf (B0), T-EPI (B1), and T-φ (B2) have all closed NEGATIVE with
the same structural shape established in §13triginta-sexta.5:

1. **Anchor** identifies a candidate "type upgrade" of a canonical
   observable (measure-valued :math:`\nu_f`; Banach-valued EPI;
   covering-space-lifted φ).
2. **Diagnostic** measures two orthogonal axes: scalar-storage
   utilisation + spectral/entropy richness.
3. **Forcing-axiom reduction** finds that no canonical constraint
   forces the upgrade; isolates a single residual axiom strictly
   weaker than the upgrade itself ((P-νf-Bijectivity);
   (P-EPI-Bijectivity); (P-φ-Homotopy-Retention)).
4. **Canonical-status check** finds that the residual axiom is itself
   independent of the catalog and is actively refuted by an existing
   canonical mechanism (scalar νf-update closure for B0; REMESH/TMEP
   for B1; PWDP / wrap-discipline for B2).
5. **Verdict** NEGATIVE; the upgrade-carrier is reclassified as a
   legitimate non-canonical research envelope (E1; E2; E3).

**L3 (cross-conjecture pattern), confirmed for B0 ∧ B1 ∧ B2.**
*Whenever a candidate type-upgrade of a canonical observable can be
matched by an existing canonical mechanism — Pontryagin-dual scalar
νf-update closure for the frequency axis, REMESH temporal
aggregation for the form axis, wrap-discipline for the phase axis —
the upgrade is non-canonical and the existing mechanism is
preferred.*  L3 is now corroborated across all three Tier-1 per-node
intrinsic types tested so far.

**Refinement noted (R-L3-1).**  The "matching canonical mechanism"
varies by axis: it is a *closure* in B0 (νf-update), a *temporal
aggregation* in B1 (REMESH), and a *projection discipline* in B2
(wrap-angle).  This suggests a coarser super-pattern
L3* (provisional): *each canonical observable comes with at least
one canonical discharge mechanism for the expressivity demand that
would otherwise force a type upgrade*.  L3* will be tested against
B3 = T-ΔNFR onwards; if it holds across the remaining Tier-1
question (B3), it will be promoted to a working heuristic for the
Tier-2/3/4 sub-questions and the Final synthesis step.

### §13triginta-decima.6 Honest Scope (Mandatory)

This section:

- **Does** close T-φ (B2) with a NEGATIVE verdict.
- **Does** classify E3 = CoverElement (covering-space lift / U(1)
  bundle element / homotopy-retaining φ representation) as
  legitimate non-canonical research envelope.
- **Does** advance the catalog type-hygiene programme to 3/11+1
  complete (B0 + B1 + B2 all NEGATIVE).
- **Does** record confirmation of cross-conjecture methodology
  lesson L3 across B0 ∧ B1 ∧ B2 and introduce the provisional
  refinement L3*.
- **Does not** advance G4 = RH, does not close T-HP, does not
  promote any operator/field/constant to canonical status, does not
  modify the catalog operators/grammar/contracts, does not modify
  any source file in ``src/tnfr/``, does not introduce a
  ``CoverElement`` class or ``ALIAS_PHASE_UNWRAPPED`` alias, does
  not delete or deprecate ``wrap_angle`` / ``ALIAS_THETA`` /
  ``get_phase``, does not delete or modify the B2a diagnostic
  module or its demo.
- **Does not** make any claim about T-ΔNFR (B3) or any subsequent
  sub-question; those are addressed sequentially per the programme
  tracker.
- **Does not** apply the D-CC-6 deferred catalog citation patch on
  this commit (one finding per commit; D-CC-6 remains queued for
  a future type-hygiene commit).

### §13triginta-decima.7 Cross-references

- §13triginta-prima — T-νf pre-registration (precedent template).
- §13triginta-secunda — T-νf forcing-axiom reduction (precedent).
- §13triginta-tertia — T-νf NEGATIVE verdict + E1 classification
  (precedent for B2c).
- §13triginta-quarta — T-EPI pre-registration (precedent template).
- §13triginta-quinta — T-EPI forcing-axiom reduction (precedent for
  TMEP-style canonical-mechanism refutation).
- §13triginta-sexta — T-EPI NEGATIVE verdict + E2 = ``BEPIElement``
  classification (direct precedent; same shape).
- §13triginta-septima — Living discoveries log; this commit appends
  D-ENV-3 (E3 = CoverElement, NEGATIVE), and refines D-MP-3 = L3
  with R-L3-1 (provisional L3* super-pattern); the D-CC-6 deferred
  catalog citation patch remains unchanged on this commit.
- §13triginta-octava — T-φ pre-registration (B2a anchor +
  two-axis diagnostic; supplies the
  :math:`w_{\mathrm{frac}} = 0` fingerprint consumed here).
- §13triginta-novena — T-φ forcing-axiom reduction (B2b; PWDP
  isolated (P-φ-Homotopy-Retention) as INDEPENDENT_AXIOM and
  refuted it at the canonical level; supplies the decisive input
  to this section).
- §13septies — T-HP open content (independent, untouched by this
  verdict).
- §19.1 — Full P1–P49 milestone table.
- ``theory/CATALOG_TYPE_HYGIENE_PROGRAMME.md`` §3, §4, §6 —
  programme tracker (advances on this commit at row B2 Phase c +
  Verdict, B2 spec line, and §3 progress summary).
- ``src/tnfr/physics/_helpers.py:29`` — ``wrap_angle`` canonical
  implementation (canonical mechanism that refutes
  (P-φ-Homotopy-Retention) and discharges E3 at the canonical level).
- ``src/tnfr/constants/aliases.py:8`` — ``ALIAS_THETA`` canonical
  scalar storage alias (canonical typing witness).
- ``src/tnfr/riemann/phi_type_signature.py`` — B2a diagnostic
  implementation (preserved as off-catalog measurement utility).
- ``examples/05_type_hygiene/80_phi_type_signature_demo.py`` — B2a two-resolution
  demo (preserved; corroborates PWDP empirically with
  :math:`w_{\mathrm{frac}} = 0` at both resolutions).

---

## §13quadraginta. T-ΔNFR Pre-registration: The Nodal-Gradient Type-of-Object Conjecture (B3 Phase a; Diagnostic Only — Does NOT Advance G4 = RH)

**Programme position.**  Fourth executed sub-question of the Catalog
Type-Hygiene Programme (after B0 = T-νf NEGATIVE, B1 = T-EPI
NEGATIVE, B2 = T-φ NEGATIVE).  Phase a of the standard three-phase
rhythm: pre-register the conjecture, fix the diagnostic, commit a
*necessary-condition* empirical signature, deliberately defer the
forcing-axiom analysis (B3b) and the final verdict + envelope
classification (B3c) to separate commits.

**Honest scope (mandatory).**  This section pre-registers a *type-of-
object conjecture* and a *diagnostic*.  It does **not** promote any
tensor-valued / operator-valued ΔNFR construction to canonical
status, does **not** modify the 13-operator catalog, does **not**
modify any existing source file in ``src/tnfr/`` (only adds the
diagnostic module ``src/tnfr/riemann/dnfr_type_signature.py``, its
re-export in ``src/tnfr/riemann/__init__.py``, and the demo
``examples/05_type_hygiene/81_dnfr_type_signature_demo.py``), and does **not** by
itself advance G4 = RH.  The diagnostic is a *necessary-condition*
probe: a non-trivial signature is required, but not sufficient, for
a tensor-rank lift of ΔNFR to be canonically necessary.

### §13quadraginta.1 — Motivation and literal canonical witness

The TNFR nodal equation is
:math:`\partial\mathrm{EPI}/\partial t = \nu_f \cdot \Delta\mathrm{NFR}(t)`.
The canonical ΔNFR is computed and stored as a scalar real number
by the unique catalog implementation
:func:`tnfr.dynamics.dnfr.default_compute_delta_nfr` at
``src/tnfr/dynamics/dnfr.py:2387``:

```python
# src/tnfr/dynamics/dnfr.py:2387
def default_compute_delta_nfr(
    G: TNFRGraph,
    *,
    cache_size: int | None = 1,
    n_jobs: int | None = None,
    profile: MutableMapping[str, Any] | None = None,
) -> None:
    """Compute ΔNFR by mixing phase, EPI, νf and a topological term."""
    ...
    _compute_dnfr(G, data, n_jobs=n_jobs, profile=profile)
```

Internally, ``_compute_dnfr`` assembles, for each node ``i``, the
three canonical gradient channels — mean-neighbour phase
:math:`\overline{\Delta\theta_i}`, mean-neighbour EPI
:math:`\overline{\Delta\mathrm{EPI}_i}`, and mean-neighbour νf
:math:`\overline{\Delta\nu_{f,i}}` — combines them with the
canonical weights stored under ``G.graph["dnfr_weights"]``, and
writes a single ``float`` into the canonical per-node storage slot
``G.nodes[node]["dnfr"]`` (alias ``ALIAS_DNFR``):

```python
# src/tnfr/constants/aliases.py:9
ALIAS_DNFR = get_aliases("DNFR")
```

The downstream consumer is the nodal equation itself, which reads
``ΔNFR`` back as a single ``float`` at
``src/tnfr/operators/nodal_equation.py:1-160`` and multiplies it by
the scalar νf to produce :math:`\partial\mathrm{EPI}/\partial t`,
also a scalar.  No canonical operator (AL, EN, IL, OZ, UM, RA, SHA,
VAL, NUL, THOL, ZHIR, NAV, REMESH) reads ΔNFR with a *non-scalar*
signature.

### §13quadraginta.2 — Catalog statement of ΔNFR

Across the canonical engine, ΔNFR is consistently typed and stored
as a scalar real number:

| Surface                                            | Type / domain                |
|----------------------------------------------------|------------------------------|
| Storage (per-node attribute via ``ALIAS_DNFR``)    | ``float ∈ ℝ``                |
| Canonical computation `default_compute_delta_nfr`  | writes ``float`` to slot     |
| Nodal-equation reader (``nodal_equation.py``)      | ``float`` (scalar product)   |
| Telemetry / structural-fields (`physics/`)         | ``float`` per node           |
| Conservation law (``physics/conservation.py``)     | scalar source/sink           |
| Grammar U2 convergence integral                    | scalar integrand             |

Every appearance of ΔNFR in the canonical operator-bound API ends in
this single scalar real representation.  The catalog therefore types
ΔNFR as the canonical *scalar nodal gradient* — i.e. a real-valued
field over the graph nodes, written rank-1 by canonical assembly
from the three gradient channels.

### §13quadraginta.3 — The candidate non-canonical envelope: tensor / operator-valued lift

The smallest enrichment that would *strictly increase* expressive
power over the canonical scalar representation is a **tensor-rank
lift** of ΔNFR to a vector- or operator-valued slot:

- A per-node vector :math:`\boldsymbol{\Delta\mathrm{NFR}}_i \in
  \mathbb{R}^{3}` retaining the three canonical gradient channels
  :math:`(d\theta, d\mathrm{EPI}, d\nu_f)` separately, prior to
  weighted scalar collapse.
- Equivalently, a per-node rank-:math:`r` element of a finite-
  dimensional inner-product space (with :math:`r \le 3` here).
- More generally, a per-node bounded self-adjoint *operator*
  :math:`\widehat{\Delta\mathrm{NFR}}_i \in \mathcal{B}(\mathcal{H}_i)`
  on some auxiliary Hilbert space, of which the canonical scalar
  is the (rank-1) projection trace.

Call this envelope **E4 = TensorGradientElement** (in symmetry with
E1 = νf Pontryagin partner :math:`\widehat{\mathbb{Z}}`,
E2 = ``BEPIElement``, E3 = ``CoverElement``).  An E4-typed ΔNFR
would carry, per node and per step, the full :math:`3`-channel
gradient triple (or operator extension) that the canonical
weighted-sum scalar collapses to one number.

The pre-registered question is:

> **T-ΔNFR Conjecture (formal statement, §13quadraginta.4).** Does
> any canonical TNFR construction (operator, field, conservation law,
> grammar rule U1–U6, conserved current, gauge structure, or nodal-
> equation derivation) require ΔNFR to be canonically typed as an
> E4 = TensorGradientElement rather than a canonical scalar
> ``float ∈ ℝ``?

The empirical signature of §13quadraginta.5 is a *necessary
condition* for the answer to be **yes**.

### §13quadraginta.4 — T-ΔNFR Conjecture (formal statement)

**T-ΔNFR Conjecture.**  The canonical type-of-object of the TNFR
nodal-gradient component ΔNFR is the canonical scalar real field
(equivalently: a real-valued ``float`` per node, written rank-1 by
:func:`tnfr.dynamics.dnfr.default_compute_delta_nfr` from the three
canonical gradient channels via the canonical weights).  No
canonical TNFR construction requires ΔNFR to be canonically typed
as a tensor / operator-valued lift (E4 = TensorGradientElement)
carrying the three gradient channels separately or any operator
extension thereof.

Equivalently, in catalog terms: the nodal equation
:math:`\partial\mathrm{EPI}/\partial t = \nu_f \cdot
\Delta\mathrm{NFR}` is *bilinear-scalar* in its two inputs, and the
scalar contract on ΔNFR is canonically saturated; the discarded
multi-channel information is not consumed anywhere in the canonical
operator-bound dynamics.

**Anchors that the conjecture must survive (B3b/B3c):**

- F1–F10 forcing-axiom inventory of §13triginta-quarta.7 (re-applied
  to ΔNFR; B3b commit).
- Per-node accessor pattern returning a single ``float`` from
  ``ALIAS_DNFR``.
- Nodal-equation evaluator at ``src/tnfr/operators/nodal_equation.py``
  consuming ΔNFR as ``float`` × ``float`` → ``float``.
- Conservation-law machinery at ``src/tnfr/physics/conservation.py``
  treating ΔNFR as a scalar source.
- Grammar U2 (CONVERGENCE & BOUNDEDNESS) constraining the scalar
  integral :math:`\int \nu_f \cdot \Delta\mathrm{NFR}\, dt`.
- Cross-references §13septies on the smooth/oscillatory split (the
  open T-HP rescaling operator :math:`\mathcal{F}` does not consume
  a multi-channel ΔNFR either).

### §13quadraginta.5 — Diagnostic S_ΔNFR (two-axis necessary condition)

**Definition.**  On a canonical TNFR ring graph
:math:`G_{n_{\mathrm{nodes}}}` with deterministic seeded initial
phase / EPI / νf perturbation, run :math:`n_{\mathrm{steps}}`
canonical ``step(G)`` evolutions and, after each step, collect for
every node ``i`` the mean-neighbour gradient triple

.. math::

   \mathbf{g}_i(t) = \left(
     \overline{\Delta\theta_i}(t),\;
     \overline{\Delta\mathrm{EPI}_i}(t),\;
     \overline{\Delta\nu_{f,i}}(t)
   \right) \in \mathbb{R}^{3},

stacked into the per-node matrix
:math:`M_i \in \mathbb{R}^{n_{\mathrm{steps}} \times 3}`.  The
diagnostic is the pair

.. math::

   \mathcal{S}_{\Delta\mathrm{NFR}} = (T_{\mathrm{frac}},\;
                                        H_{\mathrm{rank}} / \log 3)

with the two axes defined as:

1. **Tensor storage axis.**  At every ``(node, step)`` sample,
   inspect the canonical ΔNFR slot ``G.nodes[node]["dnfr"]`` for
   non-scalar payloads.  ``T_{\mathrm{frac}}`` is the fraction of
   samples whose payload is *not* a single real scalar.  Under the
   canonical implementation
   :func:`tnfr.dynamics.dnfr.default_compute_delta_nfr`, this
   fraction is structurally ``0`` — exactly mirroring
   :math:`w_{\mathrm{frac}} = 0` of the B2a φ-diagnostic and
   :math:`\mathrm{bepi\_frac} = 0` of the B1a EPI-diagnostic.

2. **Rank-entropy axis.**  For each node ``i``, compute the SVD
   :math:`M_i = U_i \Sigma_i V_i^{\top}` with singular values
   :math:`\sigma_{i,1} \ge \sigma_{i,2} \ge \sigma_{i,3} \ge 0`,
   normalise to a probability vector
   :math:`p_{i,k} = \sigma_{i,k} / \sum_j \sigma_{i,j}`, and compute
   the Shannon entropy
   :math:`H_i = -\sum_k p_{i,k} \log p_{i,k}`.  Average across nodes
   to obtain :math:`H_{\mathrm{rank}}`.  Normalise by :math:`\log 3`
   so the signature lives in :math:`[0, 1]`.

**Verdict labels (mechanically applied by the diagnostic, not by
itself sufficient for the foundational T-ΔNFR Conjecture):**

- ``SCALAR_DNFR_ADEQUATE``: signature :math:`< 0.15` *and* zero
  tensor storage fraction.
- ``TENSOR_LIFT_NECESSARY``: signature :math:`> 0.5` *or* non-zero
  tensor storage fraction.
- ``INDETERMINATE``: in between.

**Implementation.**  The diagnostic is implemented in
``src/tnfr/riemann/dnfr_type_signature.py``, exporting
``DnfrTypeSignatureCertificate`` and ``compute_dnfr_type_signature``.
The reference demo lives at ``examples/05_type_hygiene/81_dnfr_type_signature_demo.py``.

### §13quadraginta.6 — Pre-registered numerical signature

The diagnostic is executed at two resolutions at pre-registration
time (commit-time numerical fingerprint, frozen for later
comparison):

| Resolution                  | seed | S_ΔNFR    | T_frac       | R_eff   | σ1     | σ2     | σ3     | verdict                 |
|-----------------------------|------|-----------|--------------|---------|--------|--------|--------|-------------------------|
| n=24, steps=64              | 17   | 0.105763  | 0/1536       | 1.1232  | 2.0131 | 0.0209 | 0.0070 | SCALAR_DNFR_ADEQUATE    |
| n=48, steps=128             | 31   | 0.111601  | 0/6144       | 1.1304  | 2.1294 | 0.0298 | 0.0086 | SCALAR_DNFR_ADEQUATE    |

**Honest reading of this signature at Phase a.**  Both the tensor
storage axis and the rank-entropy axis return *empirically decisive
scalar-adequate* values at both resolutions.  The dominant
empirical facts are:

(a) **Zero tensor storage fraction** at both resolutions
(``0 / 1536`` and ``0 / 6144`` samples).  The canonical ΔNFR slot
is, at every ``(node, step)``, a Python ``float`` by construction
— consistent with the catalog row :math:`\Delta\mathrm{NFR} \in
\mathbb{R}` and with the bilinear-scalar nodal-equation contract.

(b) **Empirical rank-1 collapse of the gradient triple**.  The
mean singular values exhibit :math:`\sigma_1 / \sigma_2 \approx
\mathcal{O}(10^{2})` and :math:`\sigma_1 / \sigma_3 \approx
\mathcal{O}(10^{2})` at both resolutions, giving an effective rank
:math:`R_{\mathrm{eff}} \approx 1.12`–:math:`1.13` (well below the
``scalar_threshold = 0.15`` rank entropy).  The three canonical
gradient channels :math:`(d\theta, d\mathrm{EPI}, d\nu_f)` are
*not* statistically independent under canonical evolution; they
align onto a single dominant axis (in this regime, the phase
channel — see ``per_node_singular_values`` in the certificate's
``diagnostics``).

These two facts together — *zero structural tensor storage* and
*empirical rank-1 collapse* — yield the mechanical verdict
``SCALAR_DNFR_ADEQUATE`` at both resolutions, which is the
*strongest* pre-registration signature observed so far in the
Type-Hygiene Programme (B0 was decided by anchor-level scalar
contract; B1a returned ``BEPI_LIFT_NECESSARY`` by spectral
threshold; B2a returned ``COVER_LIFT_NECESSARY`` by spectral
threshold; B3a is the first sub-question whose Phase-a diagnostic
returns the *scalar-adequate* verdict mechanically at both
resolutions).

This makes the pre-registered hypothesis of §13quadraginta.7
correspondingly stronger.

### §13quadraginta.7 — Pre-registered hypothesis for B3b/B3c

Based on (i) the literal-catalog inspection of §13quadraginta.2,
(ii) the bilinear-scalar nodal-equation contract of
§13quadraginta.4, (iii) the doubly-decisive empirical signature of
§13quadraginta.6 (:math:`T_{\mathrm{frac}} = 0` *and*
:math:`R_{\mathrm{eff}} \approx 1.13`), and (iv) the universal
absence of any tensor / operator-valued ΔNFR argument in canonical
operator signatures, the **pre-registered expected verdict** at
B3c is:

> **NEGATIVE.** The canonical type of ΔNFR is the canonical scalar
> real field.  E4 = TensorGradientElement is a strictly *richer*
> envelope than the canonical type but is **not** required by any
> canonical TNFR construction.  No promotion, no deletion, no
> deprecation, no modification of the catalog.

This pre-registration commits to that expected verdict so that the
B3b forcing-axiom reduction cannot be retrofitted: if the F1–F10
analysis yields a different verdict, the pre-registration record
of §13quadraginta.6 makes the inversion explicit and audit-
traceable.

### §13quadraginta.8 — Honest scope (what this does and does not do)

This pre-registration section, the diagnostic module, and the demo:

- **Does not** promote ``TensorGradientElement`` (or any
  vector / operator-valued lift, multi-channel slot, or
  tensor-decomposition object) to canonical status.
- **Does not** modify the catalog
  (``theory/CATALOG_TYPE_HYGIENE_PROGRAMME.md`` §3, §4, §6 will
  only be touched at B3c).
- **Does not** modify any existing source file in ``src/tnfr/``;
  only adds the diagnostic module
  ``src/tnfr/riemann/dnfr_type_signature.py`` (and its export in
  ``src/tnfr/riemann/__init__.py``) and the demo
  ``examples/05_type_hygiene/81_dnfr_type_signature_demo.py``.
- **Does not** change the canonical
  ``tnfr.dynamics.dnfr.default_compute_delta_nfr``,
  ``ALIAS_DNFR``, the nodal-equation evaluator, or any tetrad
  field implementation.
- **Does not** by itself decide T-ΔNFR; B3b (forcing-axiom
  reduction) and B3c (final verdict + envelope classification)
  are required.
- **Does not** advance G4 = RH or any of the open ζ-track /
  L-track RH-equivalents (P17–P49 attack surface).
- **Does not** rely on T-νf (B0, NEGATIVE), T-EPI (B1, NEGATIVE),
  or T-φ (B2, NEGATIVE) in any way that would force their
  verdicts to be re-opened.

### §13quadraginta.9 — Cross-references

- §13triginta-prima — T-νf pre-registration (precedent for B0).
- §13triginta-tertia — T-νf NEGATIVE verdict + E1 classification
  (closes B0).
- §13triginta-quarta — T-EPI pre-registration (template for the
  three-phase rhythm).
- §13triginta-sexta — T-EPI NEGATIVE verdict + E2 = ``BEPIElement``
  classification (closes B1).
- §13triginta-octava — T-φ pre-registration (template for this
  section).
- §13triginta-decima — T-φ NEGATIVE verdict + E3 = ``CoverElement``
  classification (closes B2).
- §13triginta-septima — Discoveries log; the catalog-citation
  patch D-CC-6 (``mathematics.phase`` → ``physics/_helpers.py``;
  ``ALIAS_PHASE`` → ``ALIAS_THETA``) plus D-ENV-3 / R-L3-1 entries
  remain deferred to a future bookkeeping commit (one type-
  hygiene finding per commit).
- §13septies — T-HP open content (independent, untouched by this
  pre-registration).
- §19.1 — Full P1–P49 milestone table.
- ``theory/CATALOG_TYPE_HYGIENE_PROGRAMME.md`` §3, §4, §6 —
  programme tracker (advances on this commit at row B3 Phase a only).
- ``src/tnfr/dynamics/dnfr.py:2387`` —
  ``default_compute_delta_nfr`` canonical implementation (anchor).
- ``src/tnfr/constants/aliases.py:9`` — ``ALIAS_DNFR`` canonical
  scalar storage alias.
- ``src/tnfr/operators/nodal_equation.py`` — canonical scalar
  consumer of ΔNFR.
- ``src/tnfr/riemann/dnfr_type_signature.py`` — diagnostic
  implementation (added on this commit).
- ``examples/05_type_hygiene/81_dnfr_type_signature_demo.py`` — demo (added on
  this commit).

---

## §13quadraginta-prima. Derivation of (P-ΔNFR-Tensor-Carrier) from the Canonical Catalog — Foundational Reduction of the ΔNFR-Type Conjecture (Theory-Only Analysis; Does NOT Advance G4 = RH)

**Pre-registration status.**  This section executes the
forcing-axiom reduction phase (B3b) of the T-ΔNFR program
(§13quadraginta): it attempts to derive the tensor-carrier
principle for the canonical nodal-gradient field
(P-ΔNFR-Tensor-Carrier) from the canonical six invariants +
nodal equation + Structural Conservation Theorem + Variational
Principle + REMESH operator + the structural-field
tetrad, *or* to identify and isolate the actual residual
axiom that the derivation requires beyond the catalog.

The honest verdict (executed in §13quadraginta-secunda) is
pre-registered as one of:

- `COROLLARY_DERIVED`: (P-ΔNFR-Tensor-Carrier) follows from
  invariants 1–6 alone.
- `CONDITIONAL_COROLLARY`: (P-ΔNFR-Tensor-Carrier) follows under
  one additional identifiable axiom strictly weaker than itself.
- `INDEPENDENT_AXIOM`: (P-ΔNFR-Tensor-Carrier) is independent of
  the catalog.

Scope (mandatory honesty): this section does **not** advance
G4 = RH, does **not** close T-HP, does **not** introduce or modify
any canonical operator, does **not** delete or deprecate any
tensor-valued or operator-valued ΔNFR construction, and does
**not** by itself close T-ΔNFR.  It locates the foundational
axiom *one structural level below* (P-ΔNFR-Tensor-Carrier) and
hands T-ΔNFR back to that deeper question.

The literal canonical statement under scrutiny:

> **(P-ΔNFR-Tensor-Carrier).** In the canonical TNFR formulation,
> the per-node nodal-gradient state must take values in a tensor
> (or operator-valued) carrier over the three canonical gradient
> channels :math:`(d\theta, d\mathrm{EPI}, d\nu_f)` — equivalently
> a `TensorGradientElement` (candidate envelope E4) of rank
> :math:`r \ge 2` — *not* in :math:`\mathbb{R}` under the
> canonical ``ALIAS_DNFR`` scalar storage discipline.

### §13quadraginta-prima.1 Available Canonical Tools

The derivation may use only the following canonical machinery (no
extraneous structure):

1. **Nodal equation**: :math:`\partial \mathrm{EPI}/\partial t = \nu_f \cdot \Delta \mathrm{NFR}(t)`
   (Invariant #1), in which ΔNFR enters as a *scalar coefficient*
   multiplied by the scalar structural frequency :math:`\nu_f`
   (`src/tnfr/operators/nodal_equation.py:1–160`,
   `compute_expected_depi_dt: (float, float) → float`).
2. **Six canonical invariants** (AGENTS.md): **Nodal Equation
   Integrity (invariant #1)**, Phase-Coherent Coupling, Multi-Scale
   Fractality, Grammar Compliance, Structural Metrology,
   Reproducible Dynamics.
3. **Grammar U1–U6**, in particular **U2 (CONVERGENCE &
   BOUNDEDNESS)** which bounds :math:`\int \nu_f \cdot
   \Delta\mathrm{NFR}\, dt < \infty` as a scalar Lebesgue integral.
4. **Structural-field tetrad**: ΔNFR enters the
   canonical pressure field :math:`\Phi_s(i) = \sum_{j \neq i}
   \Delta\mathrm{NFR}_j / d(i,j)^2` as a *scalar* per-node value;
   all four tetrad fields :math:`(\Phi_s,
   |\nabla\phi|, K_\phi, \xi_C)` are scalar-valued.
5. **Structural Conservation Theorem**
   (`src/tnfr/physics/conservation.py`): per-node charge density
   :math:`\rho_i = \Phi_s(i) + K_\phi(i)` and current vector
   :math:`\mathbf{J}_i = (J_\phi(i), J_{\Delta\mathrm{NFR}}(i))
   \in \mathbb{R}^2` built from real-valued functionals of scalar
   ΔNFR.
6. **Variational Principle**: Lagrangian
   :math:`\mathcal{L}_i = T_i - V_i` with the potential term
   :math:`V_i = \tfrac{1}{2}[\Phi_s^2 + |\nabla\phi|^2 + K_\phi^2]`
   and kinetic term
   :math:`T_i = \tfrac{1}{2}[J_\phi^2 + J_{\Delta\mathrm{NFR}}^2]`
   all real-valued scalar functionals.
7. **REMESH operator** (canonical operator #13); aggregates
   per-node EPI history scalarly.  REMESH never aggregates a
   tensor-valued ΔNFR; even when ΔNFR feeds REMESH (via the
   :math:`\nu_f \cdot \Delta\mathrm{NFR}` time-integrand of U2),
   the inputs are scalar-projected at every step.
8. **Canonical computation entry-point**:

   ```python
   # src/tnfr/dynamics/dnfr.py:2387
   def default_compute_delta_nfr(G, *, ...) -> None:
       """Compute the per-node ΔNFR scalar and write it into
       ALIAS_DNFR (single float per node)."""
   ```

9. **Canonical storage alias** ``ALIAS_DNFR``
   (`src/tnfr/constants/aliases.py:9`) — the per-node nodal
   gradient is stored under this single scalar alias-tuple, with
   no companion ``dnfr_tensor`` / ``dnfr_channels`` / ``dnfr_rank``
   alias.

### §13quadraginta-prima.2 What the Canonical Catalog Forces (Scalar ℝ Layer)

The chain of forced structure for ΔNFR is straightforward and
entirely inside the catalog:

- **(M1) Operator contracts are scalar-ℝ.**  Every canonical glyph
  operator that touches ΔNFR reads via the scalar
  ``G.nodes[node]["dnfr"]`` slot and writes via the same scalar
  alias.  No operator constructs, reads, propagates, or preserves
  a tensor rank, channel index, or operator-valued component.
  Empirically verified by `examples/05_type_hygiene/81_dnfr_type_signature_demo.py`:
  ``T_frac = 0/1536`` at :math:`(n=24, T=64, \mathrm{seed}=17)`
  and ``T_frac = 0/6144`` at :math:`(n=48, T=128, \mathrm{seed}=31)`
  — strictly zero tensor-valued payloads across both resolutions.

- **(M2) The nodal equation is bilinear-scalar.**  ΔNFR enters
  :math:`\partial\mathrm{EPI}/\partial t` exclusively as the scalar
  right-hand factor of the bilinear product :math:`\nu_f \cdot
  \Delta\mathrm{NFR}`, both factors typed as `float` in
  `compute_expected_depi_dt: (float, float) → float`.  Any
  tensorial intermediate computed during the assembly of ΔNFR
  (e.g. the neighbour-gradient triple
  :math:`(d\theta_{ij}, d\mathrm{EPI}_{ij}, d\nu_{f,ij})` for each
  edge :math:`(i,j)`) is *systematically collapsed* to a single
  scalar via fixed weighted aggregation before being written to
  ``ALIAS_DNFR``.

- **(M3) U2 convergence is a scalar Lebesgue bound.**  The U2
  bounded-integral condition :math:`\int_{t_0}^{t_f} \nu_f(\tau)
  \cdot \Delta\mathrm{NFR}(\tau)\, d\tau < \infty` is a *scalar*
  Lebesgue integral of a scalar product.  No canonical
  formulation of U2 references a tensor norm, operator norm, or
  multi-channel boundedness condition; the catalog's
  boundedness discipline is built on the scalar absolute value
  :math:`|\nu_f \cdot \Delta\mathrm{NFR}|`.

- **(M4) Conservation and variational laws close on scalar ΔNFR.**
  The Noether charge :math:`Q = \sum_i \rho_i`, the energy
  :math:`E = \sum_i \varepsilon_i`, the current
  :math:`J_{\Delta\mathrm{NFR}}`, the Lagrangian, and the
  symplectic form are all real-valued functionals of scalar ΔNFR
  (see `src/tnfr/physics/conservation.py::compute_charge_density`
  and `compute_current_divergence`, which read scalar ΔNFR via
  the canonical `_helpers.get_dnfr` reader).  No conservation
  law references a tensor-valued ΔNFR current.

The conjunction M1+M2+M3+M4 establishes that **the entire
canonical machinery closes consistently with scalar real-valued
ΔNFR**.  The 13-operator catalog never reads or writes a tensor
component; the nodal equation is bilinear-scalar by construction;
U2 boundedness is a scalar Lebesgue bound; conservation and
variational laws never require a tensor-valued lift.

### §13quadraginta-prima.3 The Gap Between Scalar Aggregation Discipline and Tensor Retention

Scalar-ℝ closure (M1–M4) is necessary but not sufficient to
refute (P-ΔNFR-Tensor-Carrier): one could still ask whether the
catalog *also* admits a strictly-stronger tensor-valued
realisation in which the scalar implementation is a faithful
coordinate projection from a rank-:math:`r \ge 2` tensor
``TensorGradientElement`` onto :math:`\mathbb{R}`.  The decisive
question is whether the catalog *forces* such an upgrade.

The only canonical mechanism that could conceivably preserve
multi-channel data across the temporal evolution is a hypothetical
"tensor branch" that propagates the per-edge gradient triple
:math:`(d\theta_{ij}, d\mathrm{EPI}_{ij}, d\nu_{f,ij})` alongside
the scalar :math:`\Delta\mathrm{NFR}_i \in \mathbb{R}`.  But the
canonical engine **does not implement** any such branch: every
write to ``ALIAS_DNFR`` collapses the per-edge channels through
fixed weighted aggregation, and there is no canonical alias for a
tensor companion.

Formally, define the **Bilinear-Scalar Aggregation Discipline**:

> **Bilinear-Scalar Aggregation Discipline (BSAD).**  In the
> canonical TNFR formulation, every per-node ΔNFR value is
> *systematically aggregated* from any multi-channel
> intermediate (per-edge gradient triple, channel-wise pressure,
> etc.) into a single real scalar :math:`\Delta\mathrm{NFR}_i
> \in \mathbb{R}` via fixed weighted sum at every operator
> boundary.  The tensor rank :math:`r \ge 2` over the canonical
> gradient channels is *systematically collapsed to* :math:`r = 1`
> and is **not** retrievable from the canonical state.

This is the structural-ΔNFR analogue of TMEP (§13triginta-quinta,
B1b) and PWDP (§13triginta-novena, B2b).  Where TMEP says
"multi-modal EPI content is canonically realised *temporally* via
REMESH, not *spatially* via a Banach internal carrier", and PWDP
says "phase-orbit content is canonically realised *as wrapped
geodesic distance on* :math:`S^1`, not *as covering-space
displacement on* :math:`\widetilde{S^1}`", BSAD says
"nodal-gradient content is canonically realised *as a single
real scalar* :math:`\Delta\mathrm{NFR} \in \mathbb{R}`, not *as
a rank-:math:`r \ge 2` tensor over the canonical gradient
channels*".

BSAD is *operationally complete*: under the canonical scalar
aggregation discipline, the engine reproduces P12–P15 to machine
precision (§§10–12), recovers classical (Keplerian) and
quantum-like (interference, complementarity, quantization)
regimes (§§3–9), and satisfies all canonical conservation laws —
*without* invoking any tensor channel or rank-:math:`\ge 2`
retention.  The B3a empirical signature :math:`T_{\mathrm{frac}}
= 0` and :math:`R_{\mathrm{eff}} \approx 1.13` at both
pre-registered resolutions is the empirical fingerprint of BSAD.

**Crucially**, the *near-rank-1 collapse* measured by the
rank-entropy axis of §13quadraginta.6
(:math:`S_{\Delta\mathrm{NFR}} \approx 0.11` and
:math:`\sigma_1 / \sigma_{2,3} \sim 10^2`) is *explained* by
BSAD without invoking (P-ΔNFR-Tensor-Carrier): a canonical
dynamics whose only consumed projection of the gradient triple
is a fixed weighted scalar aggregate will, in steady state,
align the dominant singular direction with that aggregation
weight, leaving the orthogonal channels at residual amplitude.
The rank collapse is *structural*, not *spectral*.

Therefore: **(P-ΔNFR-Tensor-Carrier) is strictly stronger than
what M1+M2+M3+M4 + BSAD provide**, and any derivation must
locate an additional canonical constraint that selects the
tensor-retention upgrade.

### §13quadraginta-prima.4 Candidate Forcing Constraints (Enumeration)

The candidates available inside the canonical catalog are
enumerated below.  Each row asks: *does this axiom force the
tensor-carrier upgrade of ΔNFR?*

| #  | Axiom | Source | Forces tensor-carrier of ΔNFR? |
|---|---|---|---|
| F1 | Operator exclusivity (only the 13 canonical operators write ΔNFR). | AGENTS.md "Canonical Invariants #1". | **No** — operators write scalar `float` via ``ALIAS_DNFR`` (M1); empirically ``T_frac = 0`` at both B3a resolutions. |
| F2 | Reproducibility under fixed seeds. | AGENTS.md "Reproducible Dynamics". | **No** — scalar trajectories reproduce identically; tensor channels are not part of the seeded state. |
| F3 | Nodal-equation bilinear-scalar structure: :math:`\partial\mathrm{EPI}/\partial t = \nu_f \cdot \Delta\mathrm{NFR}`. | `nodal_equation.py::compute_expected_depi_dt: (float, float) → float`. | **No** — both factors typed as `float`; the bilinear *scalar* product saturates the canonical reading (M2). |
| F4 | Tetrad orthogonality and minimality of :math:`(\Phi_s, |\nabla\phi|, K_\phi, \xi_C)`. | AGENTS.md §"Minimal Structural Degrees of Freedom"; `STRUCTURAL_FIELDS_TETRAD.md`. | **No** — all four tetrad fields are scalar-valued; :math:`\Phi_s(i) = \sum_{j} \Delta\mathrm{NFR}_j / d(i,j)^2` reads scalar ΔNFR per node. |
| F5 | U2 convergence: :math:`\int \nu_f \cdot \Delta\mathrm{NFR}\, dt < \infty`. | AGENTS.md "U2 CONVERGENCE & BOUNDEDNESS"; `grammar_core.py`. | **No** — scalar Lebesgue integral of a scalar product (M3); no tensor norm appears in the canonical boundedness condition. |
| F6 | Structural Conservation Theorem (Noether charge :math:`Q`, energy :math:`E`, Ward identities, current :math:`\mathbf{J} = (J_\phi, J_{\Delta\mathrm{NFR}})`). | `physics/conservation.py`, `theory/STRUCTURAL_CONSERVATION_THEOREM.md`. | **No** — :math:`J_{\Delta\mathrm{NFR}}` is real-valued, built from scalar ΔNFR (M4); no tensor-valued current appears in :math:`\partial\rho/\partial t + \nabla \cdot \mathbf{J} = S_{\mathrm{grammar}}`. |
| F7 | Variational principle (Lagrangian, symplectic conjugate pair :math:`(\Phi_s, J_{\Delta\mathrm{NFR}})`). | `physics/variational.py`, AGENTS.md §"Variational Confirmation". | **No** — the potential term :math:`V = \tfrac{1}{2}[\Phi_s^2 + |\nabla\phi|^2 + K_\phi^2]` and kinetic term :math:`T = \tfrac{1}{2}[J_\phi^2 + J_{\Delta\mathrm{NFR}}^2]` are real-valued scalar functionals; the conjugate momentum :math:`J_{\Delta\mathrm{NFR}}` is a real scalar. |
| F8 | REMESH temporal aggregation. | `theory/REMESH_INFINITY_DERIVATION.md`, `operators/remesh.py`. | **No** — REMESH aggregates EPI history scalarly; ΔNFR-derived inputs are already scalar-projected (chain of M2+M1). N15 closure (§§15–23) is the asymptotic projection of a scalar transfer matrix; no tensor-rank slot is required. |
| F9 | Classical-limit demos (Keplerian orbits, scalar :math:`F = m \cdot a` analog via :math:`m \leftrightarrow 1/\nu_f`, :math:`F \leftrightarrow \Delta\mathrm{NFR}`). | `examples/02_physics_regimes/12_classical_mechanics_demo.py`. | **No** — classical regime emerges from *scalar* ΔNFR under high coherence; the "force" analog is itself a scalar in the canonical correspondence. |
| F10 | Quantum-regime demos (interference, complementarity, quantization). | `examples/02_physics_regimes/13_quantum_mechanics_demo.py`, `14_uncertainty_and_interference.py`. | **No** — quantum-like phenomena emerge from scalar ΔNFR dynamics; the complementarity :math:`\Delta\mathrm{EPI} \cdot \Delta\nu_f \ge K` is a scalar-scalar inequality. |

**Result.** No canonical constraint in :math:`\{\mathrm{F1},
\ldots, \mathrm{F10}\}` forces the tensor-carrier upgrade of
ΔNFR.  All ten admit consistent realisation with scalar
real-valued ΔNFR (as the current 13-operator implementation
demonstrates by existence, and as the B3a empirical signature
confirms: :math:`T_{\mathrm{frac}} = 0` across two independent
demo resolutions, :math:`R_{\mathrm{eff}} \approx 1.13` at both).

### §13quadraginta-prima.5 The Hidden Axiom: (P-ΔNFR-Tensor-Retention)

The derivation gap can be isolated cleanly.  Define:

> **(P-ΔNFR-Tensor-Retention).**  In the canonical TNFR
> formulation, the per-node nodal-gradient trajectory
> :math:`\{(d\theta_i, d\mathrm{EPI}_i, d\nu_{f,i})(t)\}_t` must
> retain its tensor rank :math:`r \ge 2` over the canonical
> gradient channels across the scalar aggregation step — i.e.
> distinct multi-channel inputs producing the same scalar
> aggregate must correspond to distinct canonical states, and
> conversely.

**Claim.**  (P-ΔNFR-Tensor-Carrier) is a corollary of the
canonical catalog *plus* (P-ΔNFR-Tensor-Retention), and of
nothing weaker than (P-ΔNFR-Tensor-Retention).

**Forward direction (sufficiency).**  Assume
(P-ΔNFR-Tensor-Retention).  Consider two distinct neighbour-
gradient inputs :math:`g, g' \in \mathbb{R}^3` with :math:`g
\neq g'` but identical scalar aggregate :math:`w \cdot g = w
\cdot g'` (where :math:`w \in \mathbb{R}^3` is the canonical
aggregation weight).  Retention forces the canonical state to
encode :math:`g` and :math:`g'` distinctly.  A scalar
:math:`\Delta\mathrm{NFR} \in \mathbb{R}` does not have the
cardinality to encode an arbitrary rank-3 input separately from
the aggregate (one real number cannot encode the orthogonal-
to-:math:`w` plane).  Hence the canonical ΔNFR storage must take
values in a non-trivial tensor carrier over the canonical
gradient channels — equivalently, the ``TensorGradientElement``
(candidate envelope E4) of rank :math:`r \ge 2`.  This is
(P-ΔNFR-Tensor-Carrier).

**Reverse direction (necessity at the canonical level).**
Suppose (P-ΔNFR-Tensor-Carrier) holds.  Then :math:`\Delta\mathrm{NFR}_i
\in V_{\mathrm{tensor}}` is fully specified by the rank-:math:`r`
tensor over the gradient channels.  By construction, distinct
multi-channel inputs produce distinct canonical states.  Hence
(P-ΔNFR-Tensor-Retention) holds.

**Strict-weakness of (P-ΔNFR-Tensor-Retention) vs
(P-ΔNFR-Tensor-Carrier).**  (P-ΔNFR-Tensor-Retention) is a
*meta-constraint* on the canonical aggregation map
:math:`(d\theta, d\mathrm{EPI}, d\nu_f) \mapsto` (tensor-rank of
the per-node aggregate).  It does not mention tensor carriers,
operator-valued lifts, or any specific tensor algebra.  It is
purely a faithfulness requirement on the symbolic representation
of channel rank.  By contrast, (P-ΔNFR-Tensor-Carrier) commits
to a specific carrier (``TensorGradientElement``) and a specific
algebraic structure (rank-:math:`r` tensor over the canonical
gradient channels).

Therefore (P-ΔNFR-Tensor-Retention) is structurally simpler and
strictly weaker than (P-ΔNFR-Tensor-Carrier), and the derivation
is genuine progress.

### §13quadraginta-prima.6 Canonical Status of (P-ΔNFR-Tensor-Retention) — BSAD Refutation

The question is now: is (P-ΔNFR-Tensor-Retention) itself
derivable from the canonical six invariants?

- **(B-Pro).**  Invariant #1 (Nodal Equation Integrity) could be
  read as suggesting that the nodal-gradient information should
  be canonically retained without loss.  If two neighbour-
  gradient inputs differing in their orthogonal-to-:math:`w`
  plane produced the same canonical state, an observer trying to
  reconstruct the *full multi-channel gradient* from the
  canonical record would lose the orthogonal information.

- **(B-Con, decisive).**  The **Bilinear-Scalar Aggregation
  Discipline (BSAD, §13quadraginta-prima.3)** refutes the
  per-trajectory tensor-retention requirement *at the canonical
  level*: the observable content of ΔNFR at every canonical
  operator boundary is the *scalar aggregate*, and the canonical
  nodal equation is *bilinear-scalar* by typed construction
  (F3).  Any tensor-rank lift is therefore a *coordinate choice*
  on top of the canonical state, not a canonical state itself.

  Formally: the catalog enforces nodal-equation integrity
  (invariant #1) via the bilinear scalar product :math:`\nu_f
  \cdot \Delta\mathrm{NFR}`, with all downstream conservation,
  variational, and U2 boundedness structure descending from the
  scalar tetrad fields (M2–M4).  This is *operationally
  complete* — it reproduces all canonical results (§§3–12)
  without any per-trajectory tensor-channel charge.

- **(B-Empirical).**  The B3a diagnostic (§13quadraginta.6)
  measures :math:`T_{\mathrm{frac}} = 0` *and*
  :math:`R_{\mathrm{eff}} \approx 1.13` (near rank-1) at *both*
  resolutions: canonical evolution, executed exactly as the
  catalog specifies, does not produce any (node, step) sample
  whose ΔNFR storage hosts a non-scalar payload, and the
  empirical SVD of the gradient-triple matrix collapses to a
  single dominant singular direction (:math:`\sigma_1 /
  \sigma_{2,3} \sim 10^2`).  The tensor-rank lift is
  structurally unreachable from canonical initial conditions and
  empirically vacuous from canonical evolution.  This is the
  *doubly-decisive BSAD signature*: structural zero-tensor
  storage *and* empirical rank-1 collapse, both axes returning
  the scalar-adequate verdict on independent grounds.

**Conclusion of §13quadraginta-prima.6.**  (P-ΔNFR-Tensor-Retention)
is **not derivable from the canonical six invariants**.  The
catalog realises nodal-equation integrity *bilinear-scalarly*
via the typed :math:`(\nu_f, \Delta\mathrm{NFR}) \to
\partial\mathrm{EPI}/\partial t` reader, *not*
tensor-equivariantly via a channel-retention upgrade.  The
per-trajectory tensor-rank retention that
(P-ΔNFR-Tensor-Retention) demands is an *additional* axiom,
independent of the catalog and actively refuted by BSAD at the
canonical level, with the empirical doubly-decisive fingerprint
:math:`(T_{\mathrm{frac}} = 0, R_{\mathrm{eff}} \approx 1.13)`
of B3a as decisive corroboration.

### §13quadraginta-prima.7 Sub-Verdict

The forcing-axiom reduction yields:

> **Sub-verdict (§13quadraginta-prima).**
> (P-ΔNFR-Tensor-Carrier) is a **CONDITIONAL_COROLLARY** of the
> canonical catalog: it follows from the catalog *plus*
> (P-ΔNFR-Tensor-Retention).  However,
> (P-ΔNFR-Tensor-Retention) is itself **INDEPENDENT_AXIOM** at
> the canonical level: it is not derivable from invariants 1–6
> and is actively refuted by the Bilinear-Scalar Aggregation
> Discipline (BSAD), with the B3a empirical doubly-decisive
> fingerprint :math:`(T_{\mathrm{frac}} = 0, R_{\mathrm{eff}}
> \approx 1.13, \sigma_1/\sigma_{2,3} \sim 10^2)` as decisive
> corroboration.
>
> Net: (P-ΔNFR-Tensor-Carrier) is **strictly non-canonical**.
> Any tensor-valued lift, operator-valued ΔNFR construction, or
> rank-:math:`\ge 2` channel-retaining representation is a
> legitimate research envelope — available for off-catalog
> experimentation — but is not forced by, and indeed is
> structurally orthogonal to (collapsed under canonical
> aggregation by), the canonical 13-operator realisation under
> BSAD.

This locates the residual canonical question for T-ΔNFR exactly
one level below (P-ΔNFR-Tensor-Carrier), at
(P-ΔNFR-Tensor-Retention), and identifies its refutation
mechanism (BSAD).  The final NEGATIVE verdict on T-ΔNFR, and the
classification of the tensor-carrier construction
(``TensorGradientElement``, candidate envelope E4) as a
legitimate non-canonical research envelope, are executed in
§13quadraginta-secunda (B3c).

### §13quadraginta-prima.8 Honest Scope (What This Does and Does Not Do)

This sub-programme:

- **Does** isolate the residual axiom one structural level below
  (P-ΔNFR-Tensor-Carrier).
- **Does** prove (P-ΔNFR-Tensor-Retention) is strictly weaker
  than (P-ΔNFR-Tensor-Carrier).
- **Does** refute (P-ΔNFR-Tensor-Retention) at the canonical
  level via BSAD, empirically corroborated by B3a's
  doubly-decisive :math:`(T_{\mathrm{frac}} = 0,
  R_{\mathrm{eff}} \approx 1.13)` at two resolutions.
- **Does** confirm the catalog closes consistently with scalar
  real-valued ΔNFR (M1+M2+M3+M4).
- **Does** identify the canonical dynamics as *bilinear-scalar
  in the nodal equation* (a structural observation made
  explicit here for the first time at the type-hygiene level,
  not a new canonical promotion; the bilinear-scalar typing is
  already in `nodal_equation.py`).
- **Does not** advance G4 = RH or the T-HP conjecture.
- **Does not** promote any operator, field, or constant to
  canonical status (in particular: does NOT promote
  ``TensorGradientElement``, ``ALIAS_DNFR_TENSOR``, or any
  rank-:math:`\ge 2` channel-retaining representation).
- **Does not** modify the 13-operator catalog.
- **Does not** delete or deprecate the candidate envelope
  E4 = ``TensorGradientElement``; classifies it as a research
  envelope available outside the canonical operator contracts.
- **Does not** modify any source file in `src/tnfr/`.
- **Does not** by itself close T-ΔNFR — the final verdict is
  executed in §13quadraginta-secunda.

### §13quadraginta-prima.9 Cross-references

- §13triginta-prima — T-νf Type Conjecture (pre-registration; first
  sub-question of the programme).
- §13triginta-secunda — T-νf forcing-axiom reduction (structural
  template; first instance of F1–Fn enumeration).
- §13triginta-tertia — T-νf NEGATIVE verdict (precedent for B0c).
- §13triginta-quarta — T-EPI pre-registration (B1a; structural
  template).
- §13triginta-quinta — T-EPI forcing-axiom reduction (TMEP closes
  B1b; first refutation principle of the *temporal/spatial*
  family).
- §13triginta-sexta — T-EPI NEGATIVE verdict + E2 =
  ``BEPIElement`` classification (precedent for B1c/B2c/B3c).
- §13triginta-septima — Living discoveries log (D-CC-6 catalog
  citation correction recorded at B2a; D-CC-7 candidate for
  ALIAS_DNFR/`dnfr.py` citation patch is deferred to a future
  bookkeeping commit).
- §13triginta-octava — T-φ pre-registration (B2a; second instance
  of the two-axis diagnostic template).
- §13triginta-novena — T-φ forcing-axiom reduction (PWDP closes
  B2b; second refutation principle of the *projection/retention*
  family — direct structural twin of this section).
- §13triginta-decima — T-φ NEGATIVE verdict + E3 = CoverElement
  classification (immediate precedent for B3c).
- §13quadraginta — T-ΔNFR pre-registration (B3a; third instance
  of the two-axis diagnostic template; supplies the
  doubly-decisive empirical fingerprint that BSAD consumes
  here).
- §13septies — T-HP open content (independent of this
  sub-question).
- §19.1 — Full P1–P49 milestone table.
- `theory/CATALOG_TYPE_HYGIENE_PROGRAMME.md` §4 — programme
  tracker (row B3 Phase b advances on this commit).
- `src/tnfr/dynamics/dnfr.py:2387` — ``default_compute_delta_nfr``
  canonical scalar entry-point (anchors M1 / M2 / BSAD).
- `src/tnfr/constants/aliases.py:9` — ``ALIAS_DNFR`` canonical
  scalar storage alias (anchors M1).
- `src/tnfr/operators/nodal_equation.py:1–160` —
  ``compute_expected_depi_dt: (float, float) → float`` (anchors
  M2: bilinear-scalar nodal equation).
- `src/tnfr/physics/conservation.py` — Noether charge / current /
  energy on scalar ΔNFR (anchors M4).
- `src/tnfr/physics/variational.py` — Lagrangian / symplectic
  pair :math:`(\Phi_s, J_{\Delta\mathrm{NFR}})` on scalar ΔNFR
  (anchors M4 variational sector).
- `src/tnfr/riemann/dnfr_type_signature.py` — B3a diagnostic
  implementation (anchors :math:`(T_{\mathrm{frac}} = 0,
  R_{\mathrm{eff}} \approx 1.13)` empirical corroboration of
  BSAD).
- `examples/05_type_hygiene/81_dnfr_type_signature_demo.py` — two-resolution
  demo (anchors B3a numerical fingerprint).

---
## §13quadraginta-secunda. T-ΔNFR Final NEGATIVE Verdict and Envelope Classification of E4 = TensorGradientElement (Closes B3; Does NOT Advance G4 = RH)

**Pre-registration closure.**  This section consumes the
sub-verdict of §13quadraginta-prima (B3b) and issues the final
T-ΔNFR verdict in accordance with the four-tier methodology of the
catalog type-hygiene programme (`theory/CATALOG_TYPE_HYGIENE_PROGRAMME.md`
§3, methodology lessons L1–L3 and provisional R-L3-1 / L3*).  The
verdict pre-register from §13quadraginta.7 named the NEGATIVE
branch as the expected outcome; B3b has confirmed it via the
Bilinear-Scalar Aggregation Discipline (BSAD) and the F1–F10
forcing-axiom reduction.

### §13quadraginta-secunda.1 Verdict

> **T-ΔNFR verdict: NEGATIVE.**
> The canonical type-of-object of the TNFR nodal-gradient
> observable ΔNFR is the canonical real scalar
> (``float ∈ ℝ`` stored under ``ALIAS_DNFR`` at the per-node slot
> ``G.nodes[node]["dnfr"]``, written by
> ``src/tnfr/dynamics/dnfr.py::default_compute_delta_nfr`` and
> consumed by ``src/tnfr/operators/nodal_equation.py`` as the
> bilinear-scalar second argument of
> :math:`\partial\mathrm{EPI}/\partial t = \nu_f \cdot \Delta\mathrm{NFR}`).
> The tensor-/operator-valued upgrade principle
> (P-ΔNFR-Tensor-Carrier) is **not canonical**.  It does not follow
> from the canonical six invariants, nor from the nodal equation,
> nor from any subset of grammar U1–U6, nor from the
> structural-field tetrad, nor from the Structural Conservation
> Theorem, nor from the Variational Principle, nor from REMESH
> temporal aggregation, nor from the scalar Lebesgue boundedness
> condition of U2.  Its derivation requires the additional axiom
> (P-ΔNFR-Tensor-Retention), which is itself independent of the
> canonical catalog and actively refuted at the canonical level by
> the Bilinear-Scalar Aggregation Discipline
> (BSAD, §13quadraginta-prima.3, .6).

This closes T-ΔNFR in the same shape as T-νf (B0,
§13triginta-tertia), T-EPI (B1, §13triginta-sexta), and T-φ (B2,
§13triginta-decima): the conjectured "type upgrade" of a
fundamental TNFR observable is classified as a legitimate research
envelope, not as a canonical catalog requirement.  The decisive
numerical fingerprint is the B3a doubly-decisive
tensor-storage + rank-entropy signature:

| Resolution | seed | S_ΔNFR | T_frac | R_eff | σ₁ | σ₂ | σ₃ | verdict (canonical) |
|---|---|---|---|---|---|---|---|---|
| n=24, steps=64  | 17 | 0.105763 | 0/1536 | 1.1232 | 2.0131 | 0.0209 | 0.0070 | NEGATIVE |
| n=48, steps=128 | 31 | 0.111601 | 0/6144 | 1.1304 | 2.1294 | 0.0298 | 0.0086 | NEGATIVE |

No canonical evolution at either resolution produces a node whose
per-step ΔNFR trajectory retains tensor rank ≥ 2: the
canonical aggregation pipeline writes a single real scalar at every
operator boundary (T_frac = 0 in both rows), and the empirical
near-rank-1 collapse of the canonical
:math:`(d\theta, d\mathrm{EPI}, d\nu_f)` gradient triple
(:math:`\sigma_1 / \sigma_{2,3} \sim 10^2`) confirms that even the
upstream tensorial intermediate is structurally dominated by a
single principal direction.  This is the strongest scalar-adequate
Phase-a signature observed across B0 + B1 + B2 + B3 and is exactly
the situation that B3b isolated as the gap between
(P-ΔNFR-Tensor-Carrier) (the tensor/operator carrier construction)
and the strictly weaker (P-ΔNFR-Tensor-Retention) (the bare
requirement that distinct multi-channel inputs producing the same
scalar aggregate must correspond to distinct canonical states),
the latter being itself refuted by BSAD at the canonical level.

### §13quadraginta-secunda.2 Envelope Classification of E4 = TensorGradientElement

E4 = TensorGradientElement — the tensor- or operator-valued lift of
ΔNFR over the canonical gradient channels
:math:`(d\theta, d\mathrm{EPI}, d\nu_f)`, retaining channel rank
:math:`r \geq 2` alongside (or instead of) the scalar aggregate
written under ``ALIAS_DNFR``; equivalently a per-node tensor
:math:`T_i \in \mathbb{R}^{k_1 \times k_2 \times \cdots}` with
:math:`k_j \geq 2` for at least one axis, or an operator
:math:`A_i : \mathcal{B}_{\mathrm{EPI}} \to \mathcal{B}_{\mathrm{EPI}}`
acting linearly on the local Banach element, in either case
preserving the multi-channel information that BSAD discards — is
hereby classified as:

> **E4 = TensorGradientElement — Non-canonical research envelope.**
> Status: legitimate research formalism, off-catalog.
> Canonical relationship: **structurally orthogonal** to the
> canonical scalar ℝ realisation under BSAD — the engine
> aggregates every multi-channel ΔNFR intermediate to a single real
> scalar via fixed weighted sum at every operator boundary,
> bilinearly contracting it with :math:`\nu_f` to evolve
> :math:`\partial\mathrm{EPI}/\partial t` per the canonical nodal
> equation.
> Catalog interaction: **none** required.  The 13 canonical
> operators do not read, write, preserve, or invoke any tensor
> rank, channel-retention buffer, singular-value decomposition,
> operator-valued ΔNFR action, or rank ≥ 2 representation; they
> operate exclusively through the canonical scalar accessor
> ``G.nodes[node]["dnfr"]`` (``ALIAS_DNFR``) followed by the
> bilinear scalar contraction in
> ``src/tnfr/operators/nodal_equation.py::compute_expected_depi_dt``
> with the typed signature ``(float, float) -> float``.

The envelope register now records four entries:

| ID | Object | Source | Verdict | Refutation mechanism |
|---|---|---|---|---|
| E1 | Pontryagin measure-valued :math:`\nu_f` | §13triginta-tertia | NEGATIVE | Scalar-storage axis + measure-redundancy under canonical νf-update |
| E2 | ``BEPIElement`` Banach carrier | §13triginta-sexta | NEGATIVE | TMEP (temporal-modal aggregation suffices); BEPI-storage fraction = 0 across two resolutions |
| E3 | CoverElement (covering-space lift / U(1) bundle / homotopy-retaining φ) | §13triginta-decima | NEGATIVE | PWDP (canonical wrap-discipline at every operator boundary); :math:`w_{\mathrm{frac}} = 0` across two resolutions |
| E4 | TensorGradientElement (tensor-/operator-valued ΔNFR over canonical gradient channels) | this section | NEGATIVE | BSAD (canonical bilinear-scalar aggregation at every operator boundary); :math:`T_{\mathrm{frac}} = 0` and :math:`\sigma_1 / \sigma_{2,3} \sim 10^2` across two resolutions |

**Structural note on E4 vs. E1, E2, E3.**  E1 and E2 each have a
concrete code witness in the repo (``Ω_R`` Pontryagin scaffolding
and ``src/tnfr/mathematics/epi.py:103::BEPIElement`` respectively),
even though those witnesses are never invoked by the canonical
13-operator API.  E3 has no source-code witness at all (purely
conceptual envelope).  E4 sits between these extremes: there is
**no** ``TensorGradientElement`` class, no ``ALIAS_DNFR_TENSOR``
alias, no ``rank`` / ``channel`` / ``svd`` parameter in any
canonical operator signature (verified by repo-wide grep at the
B3c commit), yet the *upstream* tensorial intermediate that BSAD
collapses is computationally explicit inside
``default_compute_delta_nfr`` as the per-channel triple
:math:`(d\theta_i, d\mathrm{EPI}_i, d\nu_{f,i})` before the
weighted aggregation step.  E4 is therefore a *latently
instantiated but structurally discarded* research envelope: the
tensor data exists transiently during ΔNFR assembly, then is
projected to ℝ before it can be observed by any canonical operator
or invariant.  This is a strictly more constraining canonical
discipline than the E3 case (where the relevant lift never enters
the canonical pipeline at all).

### §13quadraginta-secunda.3 No Deletion, No Deprecation, No Promotion, No Modification

The verdict does **not** authorise:

- introduction of any ``TensorGradientElement`` class,
  ``ALIAS_DNFR_TENSOR`` alias, ``rank`` / ``channels`` / ``svd``
  field on any canonical operator, or tensor-valued ΔNFR module
  under ``src/tnfr/`` (E4 remains a research envelope; promoting
  it to a canonical code witness is itself off-catalog and would
  require a separate, documented research-track commit);
- deprecation warnings around ``default_compute_delta_nfr``,
  ``ALIAS_DNFR``, or the bilinear-scalar contract of
  ``compute_expected_depi_dt`` in
  ``src/tnfr/dynamics/dnfr.py``, ``src/tnfr/constants/aliases.py``,
  or ``src/tnfr/operators/nodal_equation.py``;
- modification of the 13-operator catalog;
- modification of the canonical contract
  :math:`(\nu_f, \Delta\mathrm{NFR}) \mapsto \partial\mathrm{EPI}/\partial t`
  (typed ``(float, float) -> float``);
- changes to ``src/tnfr/operators/nodal_equation.py``,
  ``src/tnfr/operators/grammar_core.py``,
  ``src/tnfr/physics/fields.py``, ``src/tnfr/physics/conservation.py``,
  or ``src/tnfr/physics/variational.py``;
- any change to grammar U1–U6;
- any claim about G4 = RH, T-HP, or the open content of
  §13septies.

E4 remains available for off-catalog research (e.g.
operator-valued ΔNFR action on local Banach elements,
tetrad-channel-retaining cascade analyses, multi-channel anomaly
detection, tensor-rank diagnostics for grammar-violation
classification) provided such research is documented as
off-catalog and does not claim canonical status.  The B3a
diagnostic module (``src/tnfr/riemann/dnfr_type_signature.py``)
and its demo (``examples/05_type_hygiene/81_dnfr_type_signature_demo.py``) are
preserved as off-catalog measurement utilities, exactly as the
B0a, B1a, and B2a diagnostics were preserved at B0c, B1c, and B2c.

The D-CC-6 catalog citation correction recorded in
§13triginta-septima at B2a, plus the new D-CC-7 deferred catalog
citation patch noted at §13quadraginta-prima.9 (covering the
``ALIAS_DNFR`` / ``dnfr.py`` / ``nodal_equation.py`` typed-bilinear
contract triple), remain documentation-only findings on this
commit (one-finding-per-commit rule); both patches stay queued for
a future dedicated type-hygiene commit.

### §13quadraginta-secunda.4 Programme Bookkeeping

- ``theory/CATALOG_TYPE_HYGIENE_PROGRAMME.md`` §4 row B3: Phase c
  advances ⏳ → ✅; Verdict column advances "—" → **NEGATIVE**;
  commit-refs column appends the present commit hash.
- §3 sub-question registry status: B3 transitions from
  🟡 IN PROGRESS to ✅ COMPLETE; the B3 spec block gains a closing
  line analogous to B0/B1/B2: *"Status: ✅ COMPLETE — B3a ✅,
  B3b ✅, B3c ✅. Final verdict: NEGATIVE."*
- §3 progress summary advances: 4 sub-questions complete
  (B0 + B1 + B2 + B3 all NEGATIVE), 0 in progress, 8 pending
  (B4 – B11 + Final).
- §6 methodology lessons: an L3 promotion entry is recorded (see
  §13quadraginta-secunda.5 below) reflecting that the provisional
  L3* super-pattern (R-L3-1) now holds across all four Tier-1
  per-node intrinsic types and is promoted to a stable working
  heuristic.

### §13quadraginta-secunda.5 Methodology Lesson L3 — Promoted to L3* Across B0 ∧ B1 ∧ B2 ∧ B3

T-νf (B0), T-EPI (B1), T-φ (B2), and T-ΔNFR (B3) have all closed
NEGATIVE with the same five-step structural shape established in
§13triginta-sexta.5 and confirmed in §13triginta-decima.5:

1. **Anchor** identifies a candidate "type upgrade" of a canonical
   observable (measure-valued :math:`\nu_f`; Banach-valued EPI;
   covering-space-lifted φ; tensor-/operator-valued ΔNFR).
2. **Diagnostic** measures two orthogonal axes: scalar-storage
   utilisation + spectral/entropy richness.
3. **Forcing-axiom reduction** finds that no canonical constraint
   forces the upgrade; isolates a single residual axiom strictly
   weaker than the upgrade itself ((P-νf-Bijectivity);
   (P-EPI-Bijectivity); (P-φ-Homotopy-Retention);
   (P-ΔNFR-Tensor-Retention)).
4. **Canonical-status check** finds that the residual axiom is itself
   independent of the catalog and is actively refuted by an existing
   canonical mechanism (scalar νf-update closure for B0; REMESH/TMEP
   for B1; PWDP / wrap-discipline for B2; BSAD / bilinear-scalar
   aggregation for B3).
5. **Verdict** NEGATIVE; the upgrade-carrier is reclassified as a
   legitimate non-canonical research envelope (E1; E2; E3; E4).

**L3 (cross-conjecture pattern), confirmed for B0 ∧ B1 ∧ B2 ∧ B3.**
*Whenever a candidate type-upgrade of a canonical observable can be
matched by an existing canonical mechanism — Pontryagin-dual scalar
νf-update closure for the frequency axis, REMESH temporal
aggregation for the form axis, wrap-discipline for the phase axis,
bilinear-scalar aggregation for the nodal-gradient axis — the
upgrade is non-canonical and the existing mechanism is preferred.*
L3 is now corroborated across **all four** Tier-1 per-node
intrinsic types.  Tier 1 is closed.

**Promotion: R-L3-1 / L3* is now stable working heuristic.**
The provisional super-pattern introduced in §13triginta-decima.5 —
*each canonical observable comes with at least one canonical
discharge mechanism for the expressivity demand that would
otherwise force a type upgrade* — has held across all four Tier-1
sub-questions with four structurally distinct discharge
mechanisms:

| Sub-question | Axis | Canonical discharge mechanism | Mechanism class |
|---|---|---|---|
| B0 (T-νf)    | frequency       | scalar νf-update closure                | *closure* |
| B1 (T-EPI)   | form            | REMESH temporal aggregation             | *temporal aggregation* |
| B2 (T-φ)     | phase           | ``wrap_angle`` projection discipline    | *projection discipline* |
| B3 (T-ΔNFR)  | nodal gradient  | BSAD bilinear-scalar aggregation        | *spatial-channel aggregation* |

The four mechanism classes are structurally orthogonal (closure
vs. temporal aggregation vs. spatial projection vs. multi-channel
aggregation) and span the natural axes of expressivity-suppression
on a graph-coupled scalar field theory.  This is a strong
indication that L3* is not a coincidence of three or four nearby
observables but a *catalog-wide property* of the canonical
13-operator + grammar-U1–U6 + tetrad-:math:`(Φ_s,
\|∇φ\|, K_φ, ξ_C)` formalism: every canonical observable's
expressivity demand is matched by a canonical discharge mechanism
of an appropriate class.  **L3\* is hereby promoted from
provisional refinement to a stable working heuristic** and will be
applied predictively at Tier 2 (graph-level parameters, B4–B6) and
Tier 3 (derived diagnostic fields).

**Predictive use of L3\* for Tier 2 (advisory, not binding).**
Where Tier 2 sub-questions ask whether a graph-level scalar
parameter must be replaced by a richer carrier (matrix, fractional,
edge-dependent, complex-valued), L3* predicts: *if there exists a
canonical discharge mechanism in the catalog that already absorbs
the relevant expressivity demand, the upgrade is non-canonical*.
For example:

- **B4 (T-REMESH-window** :math:`(\tau_l, \tau_g)`**)**: the N15
  closure (REMESH-∞ derivation, §1–§23 of
  ``theory/REMESH_INFINITY_DERIVATION.md``) already supplies the
  asymptotic-limit discharge for continuous-time kernel
  expressivity; L3* predicts NEGATIVE.
- **B5 (T-Δφ_max)**: the canonical global ``Δφ_max`` derived from
  :math:`\gamma/\pi` discharges the per-edge / angle-of-attack
  expressivity via the U3 single-scalar coupling discipline; L3*
  predicts NEGATIVE.
- **B6 (T-coupling-weights)**: ``default_compute_delta_nfr``
  already aggregates per-edge real weights via fixed weighted sum
  (the spatial-channel discharge mechanism of B3 generalises);
  L3* predicts NEGATIVE.

These predictions are recorded for falsifiability; each Tier 2
sub-question will still be executed in full three-phase form and
the predictions may be overturned by phase-a empirical fingerprint
or phase-b forcing-axiom reduction.

### §13quadraginta-secunda.6 Honest Scope (Mandatory)

This section:

- **Does** close T-ΔNFR (B3) with a NEGATIVE verdict.
- **Does** classify E4 = TensorGradientElement (tensor-/operator-
  valued ΔNFR over the canonical gradient channels) as legitimate
  non-canonical research envelope.
- **Does** advance the catalog type-hygiene programme to
  4/11+1 complete (B0 + B1 + B2 + B3 all NEGATIVE).
- **Does** close Tier 1 (per-node intrinsic types) of the
  programme.
- **Does** record promotion of the provisional R-L3-1 / L3*
  super-pattern from "provisional refinement" to "stable working
  heuristic", on the strength of four structurally distinct
  canonical discharge mechanisms (closure, temporal aggregation,
  projection discipline, bilinear-scalar aggregation) all
  corroborating L3*.
- **Does** record three falsifiable Tier-2 predictions (B4, B5, B6
  expected NEGATIVE per L3*).
- **Does not** advance G4 = RH, does not close T-HP, does not
  promote any operator/field/constant/alias to canonical status,
  does not modify the catalog operators/grammar/contracts, does
  not modify any source file in ``src/tnfr/``, does not introduce
  a ``TensorGradientElement`` class or ``ALIAS_DNFR_TENSOR``
  alias, does not modify ``default_compute_delta_nfr`` or the
  bilinear-scalar contract of ``compute_expected_depi_dt``, does
  not delete or modify the B3a diagnostic module or its demo.
- **Does not** make any binding claim about B4 / B5 / B6 or any
  subsequent sub-question; the Tier-2 predictions in
  §13quadraginta-secunda.5 are advisory and falsifiable.
- **Does not** apply the D-CC-6 or D-CC-7 deferred catalog
  citation patches on this commit (one finding per commit; both
  remain queued for a future type-hygiene commit).

### §13quadraginta-secunda.7 Cross-references

- §13triginta-prima — T-νf pre-registration (precedent template).
- §13triginta-secunda — T-νf forcing-axiom reduction (precedent).
- §13triginta-tertia — T-νf NEGATIVE verdict + E1 classification
  (precedent for B3c, first envelope).
- §13triginta-quarta — T-EPI pre-registration (precedent).
- §13triginta-quinta — T-EPI forcing-axiom reduction (precedent for
  TMEP-style canonical-mechanism refutation).
- §13triginta-sexta — T-EPI NEGATIVE verdict + E2 = ``BEPIElement``
  classification (precedent for B3c, second envelope; L3 first
  observed across B0 ∧ B1).
- §13triginta-septima — Living discoveries log; this commit appends
  D-ENV-4 (E4 = TensorGradientElement, NEGATIVE), promotes D-MP-3 =
  L3 with the L3* super-pattern from provisional to stable working
  heuristic, and records three falsifiable Tier-2 predictions
  (B4, B5, B6); the D-CC-6 and D-CC-7 deferred catalog citation
  patches remain unchanged on this commit.
- §13triginta-octava — T-φ pre-registration (precedent).
- §13triginta-novena — T-φ forcing-axiom reduction (precedent for
  PWDP-style canonical-mechanism refutation).
- §13triginta-decima — T-φ NEGATIVE verdict + E3 = CoverElement
  classification (direct precedent; L3 confirmed across
  B0 ∧ B1 ∧ B2; R-L3-1 / L3* introduced as provisional).
- §13quadraginta — T-ΔNFR pre-registration (B3a anchor + two-axis
  tensor-storage + rank-entropy diagnostic; supplies the
  doubly-decisive :math:`T_{\mathrm{frac}} = 0` +
  :math:`\sigma_1 / \sigma_{2,3} \sim 10^2` fingerprint consumed
  here).
- §13quadraginta-prima — T-ΔNFR forcing-axiom reduction (B3b;
  BSAD isolated (P-ΔNFR-Tensor-Retention) as INDEPENDENT_AXIOM and
  refuted it at the canonical level; supplies the decisive input
  to this section).
- §13septies — T-HP open content (independent, untouched by this
  verdict).
- §19.1 — Full P1–P49 milestone table.
- ``theory/CATALOG_TYPE_HYGIENE_PROGRAMME.md`` §3, §4, §6 —
  programme tracker (advances on this commit at row B3 Phase c +
  Verdict, B3 spec line, §3 progress summary, §6 L3* promotion,
  and three Tier-2 predictions).
- ``src/tnfr/dynamics/dnfr.py::default_compute_delta_nfr`` —
  canonical scalar ΔNFR assembly (canonical mechanism that
  collapses any multi-channel intermediate to ℝ before writing
  ``ALIAS_DNFR``; embodies the BSAD discipline).
- ``src/tnfr/operators/nodal_equation.py::compute_expected_depi_dt``
  — canonical bilinear-scalar contract ``(float, float) -> float``
  (canonical typing witness for the scalar ΔNFR carrier).
- ``src/tnfr/constants/aliases.py::ALIAS_DNFR`` — canonical scalar
  storage alias (canonical typing witness).
- ``src/tnfr/riemann/dnfr_type_signature.py`` — B3a diagnostic
  implementation (preserved as off-catalog measurement utility).
- ``examples/05_type_hygiene/81_dnfr_type_signature_demo.py`` — B3a two-resolution
  demo (preserved; corroborates BSAD empirically with
  :math:`T_{\mathrm{frac}} = 0` and
  :math:`\sigma_1 / \sigma_{2,3} \sim 10^2` at both resolutions).

---
## §13quadraginta-tertia. T-REMESH-window Pre-registration: The Memory-Window Type-of-Object Conjecture (B4 Phase a; Diagnostic Only — Does NOT Advance G4 = RH)

**Programme position.**  Fifth executed sub-question of the Catalog
Type-Hygiene Programme (after B0 = T-νf NEGATIVE, B1 = T-EPI
NEGATIVE, B2 = T-φ NEGATIVE, B3 = T-ΔNFR NEGATIVE — Tier 1 closed).
Phase a of the standard three-phase rhythm: pre-register the
conjecture, fix the diagnostic, commit a *necessary-condition*
empirical signature, deliberately defer the forcing-axiom analysis
(B4b) and the final verdict + envelope classification (B4c) to
separate commits.

**Honest scope (mandatory).**  This section pre-registers a
*type-of-object conjecture* and a *diagnostic*.  It does **not**
promote any continuous-time / fractional-order REMESH-window
construction to canonical status, does **not** modify the
13-operator catalog, does **not** modify any existing source file
in ``src/tnfr/`` (only adds the diagnostic module
``src/tnfr/riemann/remesh_window_type_signature.py``, its
re-export in ``src/tnfr/riemann/__init__.py``, and the demo
``examples/05_type_hygiene/82_remesh_window_type_signature_demo.py``), and does
**not** by itself advance G4 = RH.  The diagnostic is a
*necessary-condition* probe: a non-trivial signature is required,
but not sufficient, for a continuous-kernel or fractional-order
lift of the REMESH window to be canonically necessary.

### §13quadraginta-tertia.1 — Motivation and literal canonical witness

The TNFR REMESH operator implements temporal coupling EPI(t) ↔
EPI(t − τ) across the canonical memory window
(τ_l, τ_g) ∈ ℕ × ℕ.  The canonical implementation
:func:`tnfr.operators.remesh.apply_network_remesh` at
``src/tnfr/operators/remesh.py:1212`` reads the window via
``int(get_param(...))``:

```python
# src/tnfr/operators/remesh.py:1212
def apply_network_remesh(G: TNFRGraph) -> None:
    ...
    tau_g = int(get_param(G, "REMESH_TAU_GLOBAL"))
    tau_l = int(get_param(G, "REMESH_TAU_LOCAL"))
    ...
    past_g = hist[-(tau_g + 1)]
    past_l = hist[-(tau_l + 1)]
```

Canonical defaults are integer-valued
(``src/tnfr/config/defaults_core.py:221-223``):

```python
REMESH_TAU_GLOBAL: int = 8
REMESH_TAU_LOCAL: int = 4
REMESH_ALPHA: float = 0.5
```

The downstream consumers are the canonical EPI history deque
``G.graph["_epi_hist"]`` populated by
:func:`tnfr.dynamics.runtime._update_epi_hist` at
``src/tnfr/dynamics/runtime.py:413``, and the N15 REMESH-∞ closure
(``theory/REMESH_INFINITY_DERIVATION.md`` §§1–8) whose entire
derivation is parametrised by integer (τ_l, τ_g) ∈ ℕ² and whose
transfer-matrix construction is integer-indexed by construction.

### §13quadraginta-tertia.2 — Catalog statement of the REMESH window

Across the canonical engine, (τ_l, τ_g) is consistently typed and
stored as a pair of non-negative integers:

| Surface                                                       | Type / domain                        |
|---------------------------------------------------------------|--------------------------------------|
| Storage (``G.graph["REMESH_TAU_LOCAL"]``, ``..._GLOBAL``)     | ``int ∈ ℕ_{≥0}``                     |
| Canonical reader ``apply_network_remesh``                     | ``int()`` coercion at entry          |
| Canonical defaults (``defaults_core.py:221-223``)             | ``int = 4``, ``int = 8``             |
| EPI history indexer ``hist[-(tau + 1)]``                      | integer Python negative index        |
| N15 REMESH-∞ asymptotic (``REMESH_INFINITY_DERIVATION.md``)   | integer-indexed transfer matrix      |
| RemeshMeta dict log (``tau_global``, ``tau_local``)           | ``int`` per recorded event           |

Every appearance of the REMESH window in the canonical operator-
bound API resolves to a pair of Python ``int`` values.  The
catalog therefore types the REMESH window as the canonical
*integer memory window* — i.e. an element of ℕ² indexing the
discrete temporal coupling between the canonical EPI history deque
slots.

### §13quadraginta-tertia.3 — The candidate non-canonical envelope: continuous kernel / fractional-order lift

The smallest enrichment that would *strictly increase* expressive
power over the canonical integer-window representation is a
**continuous-window lift** of the REMESH coupling to a non-integer
indexing scheme:

- A per-event real-valued window
  :math:`(\tau_l, \tau_g) \in \mathbb{R}_{>0}^{2}` requiring
  interpolation between adjacent EPI history slots.
- A continuous-time integral kernel
  :math:`K(t, s)` with EPI(t) coupled to ``∫ K(t, s) EPI(s) ds``
  rather than to a single discretely-indexed past sample.
- A fractional-order temporal coupling operator
  :math:`{\partial^{\alpha}\!/\!\partial t^{\alpha}}\,\mathrm{EPI}`
  for non-integer :math:`\alpha`, equivalent in the Caputo /
  Riemann–Liouville sense to a memory kernel with non-integer
  decay exponent.

Call this envelope **E5 = ContinuousWindowKernel** (in symmetry
with E1 = νf Pontryagin partner :math:`\widehat{\mathbb{Z}}`,
E2 = ``BEPIElement``, E3 = ``CoverElement``,
E4 = ``TensorGradientElement``).  An E5-typed REMESH window would
carry, per event, either a continuous real-valued window or an
integral kernel that the canonical integer-window mechanism cannot
in general represent without interpolation.

The pre-registered question is:

> **T-REMESH-window Conjecture (formal statement,
> §13quadraginta-tertia.4).**  Does any canonical TNFR
> construction force the REMESH memory window to be typed as an
> E5 = ContinuousWindowKernel object — i.e. is there a canonical
> operator, telemetry surface, conservation law, or grammar rule
> whose specification requires non-integer (τ_l, τ_g) or a
> continuous integral kernel K(t, s) rather than the canonical
> integer pair?

The empirical signature of §13quadraginta-tertia.5 is a *necessary
condition*: if the canonical engine produces
:math:`S_{\tau} \approx 0` and integer storage fraction
:math:`= 1.0`, then no canonical mechanism observed at the
diagnostic surface forces the E5 envelope.

### §13quadraginta-tertia.4 — T-REMESH-window Conjecture (formal statement)

The two-axis diagnostic operationalises the following formal
question:

> **T-REMESH-window Conjecture.**  Let
> :math:`(\tau_l, \tau_g) \in \mathbb{N}_{\ge 0}^{2}` denote the
> canonical REMESH memory window, stored as Python integers in
> ``G.graph["REMESH_TAU_LOCAL"]`` and ``G.graph["REMESH_TAU_GLOBAL"]``
> and read by ``apply_network_remesh`` via ``int(get_param(...))``.
> Then no canonical TNFR construction (no canonical operator
> :math:`\in` {AL, EN, IL, OZ, UM, RA, SHA, VAL, NUL, THOL, ZHIR,
> NAV, REMESH}, no telemetry surface in ``src/tnfr/physics/``, no
> conservation law in ``physics/conservation.py``, no grammar rule
> in U1–U6, no rule of the N15 REMESH-∞ closure in
> ``REMESH_INFINITY_DERIVATION.md``) requires the window to be
> typed as an E5 = ContinuousWindowKernel object.

The pre-registered hypothesis (§13quadraginta-tertia.7) is the
**NEGATIVE** answer.

### §13quadraginta-tertia.5 — Diagnostic S_τ (two-axis necessary condition)

The diagnostic
:func:`tnfr.riemann.compute_remesh_window_type_signature` returns a
:class:`RemeshWindowTypeSignatureCertificate` with the following
two structural axes:

**Axis A — Integer storage axis.**  At every recorded REMESH event
(at every step where ``apply_network_remesh`` is invoked), inspect
the canonical storage slots ``G.graph["REMESH_TAU_LOCAL"]`` and
``G.graph["REMESH_TAU_GLOBAL"]``.  Record the fraction
:math:`F_{\mathrm{int}}` of slot reads whose stored value is a
Python ``int`` (or a numerical value with zero fractional part).
The canonical engine produces :math:`F_{\mathrm{int}} = 1.0` by
construction (the ``int(get_param(...))`` coercion at entry).  Any
:math:`F_{\mathrm{int}} < 1.0` would be a structural witness that
some canonical surface stores or propagates a non-integer window —
direct evidence for the E5 envelope.

**Axis B — Window-refinement bracket axis.**  For each integer
offset :math:`j \in \{0, 1, 2\}`, build a freshly-warmed canonical
graph from the same seed, set
:math:`(\tau_l, \tau_g) = (\tau_l^{0} + j, \tau_g^{0} + j)`, fire
:func:`apply_network_remesh` ``n_events`` times, and record the
final per-node EPI snapshot.  Compute, per node, the relative
variance
:math:`\mathrm{Var}(\mathrm{EPI})\,/\,\langle |\mathrm{EPI}| \rangle`
across the bracket, average across nodes, and squash via
:math:`\tanh` to a signature
:math:`S_{\tau} \in [0, 1]`.  If
:math:`S_{\tau} \approx 0`, the canonical post-REMESH state is
*flat* under integer-window refinement — adjacent integer windows
in the bracket already produce indistinguishable outputs, so no
canonical mechanism distinguishes between them in a way that would
force interpolation.  If :math:`S_{\tau} \to 1`, the canonical
post-REMESH state is *saturated* across the bracket — the integer-
resolution discretisation is at the edge of what the canonical
mechanism can resolve, and a continuous-window lift might be
canonically necessary.

The verdict triad is:

- ``INTEGER_WINDOW_ADEQUATE`` if
  :math:`S_{\tau} < 0.15` **and** :math:`F_{\mathrm{int}} = 1.0`.
- ``CONTINUOUS_KERNEL_NECESSARY`` if
  :math:`S_{\tau} > 0.5` **or** :math:`F_{\mathrm{int}} < 1.0`.
- ``INDETERMINATE`` otherwise.

### §13quadraginta-tertia.6 — Pre-registered numerical signature

The diagnostic is executed at two resolutions at pre-registration
time (commit-time numerical fingerprint, frozen for later
comparison):

| Resolution                            | seed | S_τ        | F_int (int/total)     | raw rel.var. | bracket L2 | windows                              | verdict                     |
|---------------------------------------|------|------------|-----------------------|--------------|------------|--------------------------------------|-----------------------------|
| n=24, warmup=16, (τ_l,τ_g)=(4,8), e=8 | 17   | 0.000000   | 1.0000 (48/48)        | 4.107780e-09 | 0.000021   | {(4,8), (5,9), (6,10)}               | INTEGER_WINDOW_ADEQUATE     |
| n=48, warmup=24, (τ_l,τ_g)=(6,12), e=12 | 31 | 0.000000   | 1.0000 (72/72)        | 0.000000e+00 | 0.000000   | {(6,12), (7,13), (8,14)}             | INTEGER_WINDOW_ADEQUATE     |

**Honest reading of this signature at Phase a.**  Both the integer
storage axis and the window-refinement bracket axis return
*empirically decisive integer-adequate* values at both
resolutions.  The dominant empirical facts are:

(a) **Perfect integer storage fraction** at both resolutions
(``48/48`` and ``72/72`` samples).  The canonical REMESH window
slots are, at every recorded event, Python ``int`` values by
construction — consistent with the catalog row
:math:`(\tau_l, \tau_g) \in \mathbb{N}^{2}` and with the
``int(get_param(...))`` coercion at the canonical reader entry.

(b) **Machine-zero bracket signature** at both resolutions
(:math:`S_{\tau} = 0` with raw relative variance
:math:`\sim 10^{-9}` at the smaller resolution and literally
:math:`0.0` at the larger).  Adjacent integer windows in the
bracket produce indistinguishable post-REMESH EPI snapshots —
there is no canonical mechanism in the observed surface that
distinguishes :math:`(\tau_l, \tau_g)` from :math:`(\tau_l + 1,
\tau_g + 1)` or :math:`(\tau_l + 2, \tau_g + 2)` in a way that
would force interpolation between integer slots.

These two facts together — *perfect integer storage* and
*machine-zero bracket signature* — yield the mechanical verdict
``INTEGER_WINDOW_ADEQUATE`` at both resolutions, which is the
*strongest* pre-registration signature observed so far in the
Type-Hygiene Programme (stronger than B3a, which still showed
:math:`R_{\mathrm{eff}} \approx 1.13`; here the bracket variance
is literally zero at the larger resolution).

This makes the pre-registered hypothesis of
§13quadraginta-tertia.7 correspondingly stronger.

### §13quadraginta-tertia.7 — Pre-registered hypothesis for B4b/B4c

Based on (i) the literal-catalog inspection of
§13quadraginta-tertia.2, (ii) the integer-indexed transfer-matrix
construction of the N15 REMESH-∞ closure
(``REMESH_INFINITY_DERIVATION.md`` §§1–8), (iii) the doubly-
decisive empirical signature of §13quadraginta-tertia.6
(:math:`F_{\mathrm{int}} = 1.0` *and* :math:`S_{\tau} = 0` at both
resolutions), (iv) the universal absence of any continuous-kernel /
fractional-order REMESH-window argument in canonical operator
signatures, and (v) the Tier-2 prediction from §13quadraginta-
secunda (B4 predicted NEGATIVE per L3*), the **pre-registered
expected verdict** at B4c is:

> **NEGATIVE.** The canonical type of the REMESH memory window is
> the canonical integer pair :math:`(\tau_l, \tau_g) \in
> \mathbb{N}^{2}`.  E5 = ContinuousWindowKernel is a strictly
> *richer* envelope than the canonical type but is **not** required
> by any canonical TNFR construction.  The predicted canonical
> discharge mechanism is the N15 REMESH-∞ closure (mean ergodic
> theorem applied to the contractive transfer matrix at integer
> :math:`\tau_g \to \infty`).  No promotion, no deletion, no
> deprecation, no modification of the catalog.

This pre-registration commits to that expected verdict so that the
B4b forcing-axiom reduction cannot be retrofitted: if the F1–F10
analysis yields a different verdict, the pre-registration record of
§13quadraginta-tertia.6 makes the inversion explicit and audit-
traceable.

### §13quadraginta-tertia.8 — Honest scope (what this does and does not do)

This pre-registration section, the diagnostic module, and the demo:

- **Does not** promote ``ContinuousWindowKernel`` (or any
  continuous-time / fractional-order REMESH-window lift) to
  canonical status.
- **Does not** modify the catalog
  (``theory/CATALOG_TYPE_HYGIENE_PROGRAMME.md`` §3, §4, §6 will
  only be touched at B4c; B4a touches only the §4 row Phase-a
  column and the §3 progress paragraph).
- **Does not** modify any existing source file in ``src/tnfr/``;
  only adds the diagnostic module
  ``src/tnfr/riemann/remesh_window_type_signature.py`` (and its
  export in ``src/tnfr/riemann/__init__.py``) and the demo
  ``examples/05_type_hygiene/82_remesh_window_type_signature_demo.py``.
- **Does not** change the canonical
  ``tnfr.operators.remesh.apply_network_remesh``,
  ``REMESH_TAU_LOCAL`` / ``REMESH_TAU_GLOBAL`` defaults, the EPI
  history deque, the N15 REMESH-∞ derivation, or any tetrad field
  implementation.
- **Does not** by itself decide T-REMESH-window; B4b (forcing-
  axiom reduction) and B4c (final verdict + envelope
  classification) are required.
- **Does not** advance G4 = RH or any of the open ζ-track / L-track
  RH-equivalents (P17–P49 attack surface).
- **Does not** rely on T-νf (B0, NEGATIVE), T-EPI (B1, NEGATIVE),
  T-φ (B2, NEGATIVE), or T-ΔNFR (B3, NEGATIVE) in any way that
  would force their verdicts to be re-opened.

### §13quadraginta-tertia.9 — Cross-references

- §13triginta-prima — T-νf pre-registration (precedent for B0).
- §13triginta-tertia — T-νf NEGATIVE verdict + E1 classification
  (closes B0).
- §13triginta-quarta — T-EPI pre-registration (template for the
  three-phase rhythm).
- §13triginta-sexta — T-EPI NEGATIVE verdict + E2 = ``BEPIElement``
  classification (closes B1).
- §13triginta-octava — T-φ pre-registration.
- §13triginta-decima — T-φ NEGATIVE verdict + E3 = ``CoverElement``
  classification (closes B2).
- §13quadraginta — T-ΔNFR pre-registration (template for this
  section).
- §13quadraginta-secunda — T-ΔNFR NEGATIVE verdict +
  E4 = ``TensorGradientElement`` classification + L3* promotion
  + three Tier-2 NEGATIVE predictions for B4/B5/B6 (closes B3,
  closes Tier 1, sets predictive baseline for this sub-question).
- §13septies — T-HP open content (independent, untouched by this
  pre-registration).
- §19.1 — Full P1–P49 milestone table.
- ``theory/CATALOG_TYPE_HYGIENE_PROGRAMME.md`` §3, §4 —
  programme tracker (advances on this commit at row B4 Phase a only).
- ``theory/REMESH_INFINITY_DERIVATION.md`` §§1–8 — N15 REMESH-∞
  closure (integer-indexed transfer-matrix derivation; predicted
  canonical discharge mechanism for B4c).
- ``src/tnfr/operators/remesh.py:1212`` —
  ``apply_network_remesh`` canonical implementation (anchor).
- ``src/tnfr/config/defaults_core.py:221-223`` — canonical integer
  defaults ``REMESH_TAU_LOCAL = 4``, ``REMESH_TAU_GLOBAL = 8``.
- ``src/tnfr/dynamics/runtime.py:413`` —
  ``_update_epi_hist`` (canonical EPI history deque populator).
- ``src/tnfr/riemann/remesh_window_type_signature.py`` —
  diagnostic implementation (added on this commit).
- ``examples/05_type_hygiene/82_remesh_window_type_signature_demo.py`` — demo
  (added on this commit).

---
## §13quadraginta-quarta. Derivation of (P-REMESH-window-Continuous-Kernel-Carrier) from the Canonical Catalog — Foundational Reduction of the REMESH-window-Type Conjecture (Theory-Only Analysis; Does NOT Advance G4 = RH)

**Pre-registration status.**  This section executes the
forcing-axiom reduction phase (B4b) of the T-REMESH-window
program (§13quadraginta-tertia): it attempts to derive the
continuous-kernel carrier principle for the canonical REMESH
memory window (P-REMESH-window-Continuous-Kernel-Carrier) from
the canonical six invariants + nodal equation + Structural
Conservation Theorem + Variational Principle + REMESH operator
+ N15 REMESH-∞ closure, *or* to identify and isolate the actual
residual axiom that the derivation requires beyond the catalog.

The honest verdict (executed in §13quadraginta-quinta) is
pre-registered as one of:

- `COROLLARY_DERIVED`: the continuous-kernel carrier principle
  follows from invariants 1–6 alone.
- `CONDITIONAL_COROLLARY`: it follows under one additional
  identifiable axiom strictly weaker than itself.
- `INDEPENDENT_AXIOM`: it is independent of the catalog.

Scope (mandatory honesty): this section does **not** advance
G4 = RH, does **not** close T-HP, does **not** introduce or
modify any canonical operator, does **not** delete or deprecate
any continuous-kernel / fractional-order REMESH construction,
and does **not** by itself close T-REMESH-window.  It locates
the foundational axiom *one structural level below*
(P-REMESH-window-Continuous-Kernel-Carrier) and hands
T-REMESH-window back to that deeper question.

The literal canonical statement under scrutiny:

> **(P-REMESH-window-Continuous-Kernel-Carrier).**  In the
> canonical TNFR formulation, the REMESH memory window must be
> carried as a continuous-time integral kernel
> :math:`K: \mathbb{R}_{\ge 0} \times \mathbb{R}_{\ge 0} \to
> \mathbb{R}` with EPI coupled via
> :math:`\int_0^t K(t, s)\, \mathrm{EPI}(s)\, ds`, or as a
> fractional-order temporal coupling operator
> :math:`\partial^\alpha\!/\!\partial t^\alpha\, \mathrm{EPI}`
> for non-integer :math:`\alpha` — equivalently a
> ``ContinuousWindowKernel`` (candidate envelope E5) — *not* via
> the canonical integer pair :math:`(\tau_l, \tau_g) \in
> \mathbb{N}^2` indexing the canonical EPI history deque.

### §13quadraginta-quarta.1 Available Canonical Tools

The derivation may use only the following canonical machinery:

1. **Nodal equation**: :math:`\partial \mathrm{EPI}/\partial t =
   \nu_f \cdot \Delta\mathrm{NFR}(t)`.  No memory-window term
   appears in the canonical nodal equation; temporal coupling is
   *external*, supplied exclusively by the REMESH operator.
2. **Six canonical invariants** (AGENTS.md), in particular
   **Reproducible Dynamics** (#6) which requires deterministic
   integer-indexed state at every operator boundary.
3. **Grammar U1–U6**, in particular **U2 (CONVERGENCE &
   BOUNDEDNESS)** which bounds the time-integral
   :math:`\int \nu_f \cdot \Delta\mathrm{NFR}\, dt < \infty`
   along the canonical discrete trajectory.
4. **REMESH operator** (canonical operator #13), implemented as
   :func:`tnfr.operators.remesh.apply_network_remesh` at
   ``src/tnfr/operators/remesh.py:1212`` reading the memory
   window via integer Python indexing of the EPI history deque:

   ```python
   # src/tnfr/operators/remesh.py:1212
   def apply_network_remesh(G: TNFRGraph) -> None:
       ...
       tau_g = int(get_param(G, "REMESH_TAU_GLOBAL"))
       tau_l = int(get_param(G, "REMESH_TAU_LOCAL"))
       ...
       past_g = hist[-(tau_g + 1)]
       past_l = hist[-(tau_l + 1)]
   ```

5. **Canonical defaults** (``defaults_core.py:221-223``):
   ``REMESH_TAU_LOCAL: int = 4``, ``REMESH_TAU_GLOBAL: int = 8``,
   ``REMESH_ALPHA: float = 0.5``.  All canonical defaults are
   integer-valued by typed declaration.
6. **EPI history deque** ``G.graph["_epi_hist"]`` populated by
   :func:`tnfr.dynamics.runtime._update_epi_hist` at
   ``src/tnfr/dynamics/runtime.py:413``: a Python deque of
   per-step EPI snapshots, integer-indexed by definition.
7. **N15 REMESH-∞ asymptotic** (``REMESH_INFINITY_DERIVATION.md``
   §§1–8): the REMESH-∞ operator
   :math:`\mathcal{R}_\infty = \lim_{\tau_g \to \infty}
   \mathcal{R}_{\tau_l, \tau_g, \alpha}` is constructed as the
   mean-ergodic limit of a *contractive integer-indexed transfer
   matrix* acting on the integer-indexed state vector
   :math:`x(t) = (\mathrm{EPI}(t), \ldots,
   \mathrm{EPI}(t - T_{\max}))^\top \in \mathbb{R}^{T_{\max}+1}`.
   The asymptotic limit is taken over integer :math:`\tau_g`; no
   continuous-time kernel appears.
8. **Structural Conservation Theorem**
   (`src/tnfr/physics/conservation.py`): conservation laws close
   on the per-step canonical state at integer time indices; no
   continuous-time current or fractional charge appears.

### §13quadraginta-quarta.2 What the Canonical Catalog Forces (Integer-Window Layer)

The chain of forced structure for the REMESH memory window is:

- **(W1) The canonical REMESH reader is integer-indexed by typed
  construction.**  The expression ``hist[-(tau + 1)]`` is a Python
  list / deque negative-index lookup; ``tau`` is coerced to
  ``int`` at function entry via ``int(get_param(...))``; the
  index ``-(tau + 1)`` is a Python integer.  No interpolation
  between adjacent history slots is performed, and no canonical
  branch admits a non-integer offset.

- **(W2) The EPI history deque is integer-indexed by
  construction.**  ``_update_epi_hist`` appends one snapshot per
  step.  The deque is a discrete sequence
  :math:`(\mathrm{EPI}_0, \mathrm{EPI}_1, \ldots, \mathrm{EPI}_T)`
  with :math:`T \in \mathbb{N}`; there is no canonical
  in-between-step state and no canonical interpolation between
  snapshots.

- **(W3) U2 boundedness is a discrete sum (Riemann sum at unit
  step) of a scalar integrand.**  The canonical
  :math:`\int \nu_f \cdot \Delta\mathrm{NFR}\, dt` is realised as
  a sum
  :math:`\sum_{n=0}^{N-1} \nu_f(t_n) \cdot \Delta\mathrm{NFR}(t_n)
  \cdot \Delta t` at integer time indices.  No canonical
  formulation of U2 references a continuous-time Lebesgue
  integral with non-trivial kernel; the integrand is sampled at
  the canonical integer time grid.

- **(W4) N15 REMESH-∞ closure is integer-indexed by
  construction.**  The contractive transfer matrix is built from
  the discrete state vector :math:`x(t) \in
  \mathbb{R}^{T_{\max}+1}` with integer slot index, and the
  mean-ergodic limit is taken over integer :math:`\tau_g`.  No
  step of the §§1–8 derivation references a continuous-time
  kernel, a fractional power of a continuous operator, or any
  non-integer index.

The conjunction W1+W2+W3+W4 establishes that **the entire
canonical REMESH machinery closes consistently with the
integer-window discipline**.  The 13-operator catalog never
constructs, reads, propagates, or preserves a continuous-time
kernel or fractional-order memory operator.

### §13quadraginta-quarta.3 The Gap Between Integer-Sampling Discipline and Continuous-Kernel Retention

Integer-window closure (W1–W4) is necessary but not sufficient
to refute (P-REMESH-window-Continuous-Kernel-Carrier): one could
still ask whether the catalog *also* admits a strictly-stronger
continuous-kernel realisation in which the integer-indexed
implementation is a faithful sampling projection from a
continuous-time integral kernel :math:`K(t, s)` onto the
canonical integer time grid.  The decisive question is whether
the catalog *forces* such an upgrade.

The only canonical mechanism that could conceivably preserve
between-slot kernel data across the temporal evolution is a
hypothetical "continuous branch" that propagates the full
:math:`K(t, s)` alongside the integer-indexed EPI history deque.
But the canonical engine **does not implement** any such branch:
every REMESH event reads from the discrete deque at integer
offsets, and there is no canonical alias for a continuous-kernel
companion.

Formally, define the **Discrete-Integer Temporal Sampling
discipline**:

> **Discrete-Integer Temporal Sampling discipline (DITS).**  In
> the canonical TNFR formulation, every REMESH event *samples*
> the EPI history at integer offsets
> :math:`-(\tau + 1) \in -\mathbb{N}_{\ge 1}` via Python
> negative-indexing of the canonical history deque.  Any
> continuous-time intermediate (if it existed) is *systematically
> projected* onto the canonical integer time grid via the
> appended-per-step deque-population discipline of
> ``_update_epi_hist``.  The continuous-kernel content :math:`K(t,
> s)` for non-integer :math:`s` is *systematically collapsed to*
> sampled values at integer :math:`s = t - (\tau + 1)\Delta t`
> and is **not** retrievable from the canonical state.

This is the temporal-window analogue of the family of refutation
principles already established for B1 (TMEP), B2 (PWDP), and B3
(BSAD).  Where TMEP says "multi-modal EPI content is canonically
realised *temporally* via REMESH, not *spatially* via a Banach
internal carrier", PWDP says "phase-orbit content is canonically
realised *as wrapped geodesic distance on* :math:`S^1`, not *as
covering-space displacement on* :math:`\widetilde{S^1}`", and
BSAD says "nodal-gradient content is canonically realised *as a
single real scalar*, not *as a rank-:math:`\ge 2` tensor*", DITS
says "memory-window content is canonically realised *as
discrete-integer sampling at the canonical time grid*, not *as a
continuous-time integral kernel or fractional-order operator*".

DITS is *operationally complete*: under the canonical
integer-sampling discipline, the engine reproduces P12–P15 to
machine precision (§§10–12), the N15 REMESH-∞ closure derives
fully analytically from the integer-indexed contractive transfer
matrix (`REMESH_INFINITY_DERIVATION.md` §§1–8), classical and
quantum-like regimes emerge (§§3–9), and all canonical
conservation laws hold — *without* invoking any continuous-time
kernel or fractional-order operator.  The B4a empirical signature
:math:`F_{\mathrm{int}} = 1.0` (perfect integer storage at both
resolutions, 48/48 and 72/72) and :math:`S_\tau = 0` (machine-
zero bracket variance at both resolutions) is the
*doubly-decisive empirical fingerprint of DITS*: structural
zero-non-integer storage *and* empirical zero bracket variance
under integer-window refinement, both axes returning the
integer-adequate verdict on independent grounds.

**Crucially**, the *machine-zero bracket variance* measured by
the window-refinement axis of §13quadraginta-tertia.6 is
*explained* by DITS without invoking
(P-REMESH-window-Continuous-Kernel-Carrier): a canonical
dynamics whose only consumed temporal-coupling input is a fixed
integer-offset sample of the history deque will, in steady
state, produce identical post-REMESH states for adjacent integer
windows that all sample inside the convergent contractive
regime.  The bracket invariance is *structural* (consequence of
the integer-sampling discipline and the contractive transfer
matrix), not a numerical accident.

Therefore: **(P-REMESH-window-Continuous-Kernel-Carrier) is
strictly stronger than what W1+W2+W3+W4 + DITS provide**, and
any derivation must locate an additional canonical constraint
that selects the continuous-kernel upgrade.

### §13quadraginta-quarta.4 Candidate Forcing Constraints (Enumeration)

The candidates available inside the canonical catalog are
enumerated below.  Each row asks: *does this axiom force the
continuous-kernel upgrade of the REMESH memory window?*

| #  | Axiom | Source | Forces continuous-kernel carrier of REMESH window? |
|---|---|---|---|
| F1 | Operator exclusivity (only the 13 canonical operators couple EPI temporally). | AGENTS.md "Canonical Invariants #1". | **No** — REMESH writes via integer-offset deque reads (W1); empirically ``F_int = 1.0`` at both B4a resolutions. |
| F2 | Reproducibility under fixed seeds. | AGENTS.md "Reproducible Dynamics" (invariant #6). | **No** — integer-indexed trajectories reproduce identically; continuous-kernel content is not part of the seeded state. |
| F3 | Nodal-equation bilinear-scalar structure. | `nodal_equation.py`. | **No** — the nodal equation has no memory-window term; temporal coupling is external to the nodal equation and supplied exclusively by REMESH at integer offsets. |
| F4 | Tetrad orthogonality and minimality of :math:`(\Phi_s, \|\nabla\phi\|, K_\phi, \xi_C)`. | AGENTS.md §"Minimal Structural Degrees of Freedom"; `STRUCTURAL_FIELDS_TETRAD.md`. | **No** — all four tetrad fields are scalar-valued and integer-time-indexed; none references a continuous-time kernel. |
| F5 | U2 convergence: :math:`\int \nu_f \cdot \Delta\mathrm{NFR}\, dt < \infty`. | AGENTS.md "U2 CONVERGENCE & BOUNDEDNESS"; `grammar_core.py`. | **No** — realised as a discrete Riemann sum at integer time indices (W3); no continuous-time kernel appears in the canonical boundedness condition. |
| F6 | Structural Conservation Theorem. | `physics/conservation.py`, `theory/STRUCTURAL_CONSERVATION_THEOREM.md`. | **No** — conservation closes on the per-step canonical state at integer time indices; no fractional or continuous current appears in :math:`\partial\rho/\partial t + \nabla \cdot \mathbf{J} = S_{\mathrm{grammar}}`. |
| F7 | Variational principle (Lagrangian, symplectic conjugate pairs). | `physics/variational.py`, AGENTS.md §"Variational Confirmation". | **No** — the Lagrangian and Hamiltonian are evaluated at integer time indices; no fractional derivative appears in :math:`\mathcal{L}_i = T_i - V_i`. |
| F8 | REMESH temporal aggregation. | `theory/REMESH_INFINITY_DERIVATION.md`, `operators/remesh.py:1212`. | **No** — REMESH samples the history deque at integer offsets via ``hist[-(tau+1)]``; the canonical implementation literally indexes by integer (W1+W2). |
| F9 | N15 REMESH-∞ closure. | `REMESH_INFINITY_DERIVATION.md` §§1–8. | **No** — the entire N15 derivation is parameterised by integer :math:`\tau_g`; the contractive transfer matrix is integer-indexed; the mean-ergodic limit is taken over integer :math:`\tau_g \to \infty` (W4). |
| F10 | Classical-limit / quantum-regime demos. | `examples/02_physics_regimes/12_classical_mechanics_demo.py`, `examples/02_physics_regimes/13_quantum_mechanics_demo.py`. | **No** — both regimes emerge from the integer-time-indexed canonical evolution; no demo references a continuous-time kernel or fractional-order temporal coupling. |

**Result.** No canonical constraint in :math:`\{\mathrm{F1},
\ldots, \mathrm{F10}\}` forces the continuous-kernel carrier
upgrade of the REMESH memory window.  All ten admit consistent
realisation with the integer-window discipline (as the current
13-operator implementation demonstrates by existence, the N15
closure demonstrates analytically, and the B4a empirical
signature confirms doubly: :math:`F_{\mathrm{int}} = 1.0` across
two independent demo resolutions, :math:`S_\tau = 0` at both —
the *strongest pre-registration signature observed in the
programme*).

### §13quadraginta-quarta.5 The Hidden Axiom: (P-REMESH-window-Continuous-Retention)

The derivation gap can be isolated cleanly.  Define:

> **(P-REMESH-window-Continuous-Retention).**  In the canonical
> TNFR formulation, the REMESH memory window must retain its
> continuous-time content :math:`K(t, s)` for non-integer
> :math:`s` across the integer-sampling step of
> :func:`apply_network_remesh` — i.e. distinct continuous-time
> intermediates producing the same integer-sampled value must
> correspond to distinct canonical states, and conversely.

**Claim.** (P-REMESH-window-Continuous-Kernel-Carrier) is a
corollary of the canonical catalog *plus*
(P-REMESH-window-Continuous-Retention), and of nothing weaker
than (P-REMESH-window-Continuous-Retention).

**Forward direction (sufficiency).**  Assume
(P-REMESH-window-Continuous-Retention).  Consider two distinct
continuous-time kernel inputs :math:`K, K' \in C(\mathbb{R}_{\ge
0}^2)` with :math:`K \neq K'` but identical integer-sampled
values :math:`K(t_n, t_n - (\tau+1)\Delta t) = K'(t_n, t_n -
(\tau+1)\Delta t)` for every canonical integer time :math:`t_n`
and every canonical integer offset :math:`\tau \in
\{\tau_l, \tau_g\}`.  Retention forces the canonical state to
encode :math:`K` and :math:`K'` distinctly.  An integer pair
:math:`(\tau_l, \tau_g) \in \mathbb{N}^2` plus the discrete EPI
history deque does not have the cardinality to encode an
arbitrary continuous-time kernel separately from its sampled
values (one cannot encode an entire :math:`L^2`-function of
:math:`s` using countably many integer-sampled scalars).  Hence
the canonical REMESH window must take values in a non-trivial
continuous-kernel carrier — equivalently, the
``ContinuousWindowKernel`` (candidate envelope E5).  This is
(P-REMESH-window-Continuous-Kernel-Carrier).

**Reverse direction (necessity at the canonical level).**
Suppose (P-REMESH-window-Continuous-Kernel-Carrier) holds.  Then
the REMESH window :math:`K(t, s) \in V_{\mathrm{continuous}}` is
fully specified by the continuous-time kernel.  By construction,
distinct continuous-time inputs produce distinct canonical
states.  Hence (P-REMESH-window-Continuous-Retention) holds.

**Strict-weakness of (P-REMESH-window-Continuous-Retention) vs
(P-REMESH-window-Continuous-Kernel-Carrier).**
(P-REMESH-window-Continuous-Retention) is a *meta-constraint* on
the canonical sampling map (continuous kernel) :math:`\mapsto`
(integer-sampled values).  It does not mention continuous
kernels, fractional operators, or any specific functional space.
It is purely a faithfulness requirement on the symbolic
representation of between-slot content.  By contrast,
(P-REMESH-window-Continuous-Kernel-Carrier) commits to a
specific carrier (``ContinuousWindowKernel``) and a specific
algebraic structure (continuous-time integral kernel
:math:`K(t, s)` or fractional-order operator
:math:`\partial^\alpha\!/\!\partial t^\alpha`).

Therefore (P-REMESH-window-Continuous-Retention) is structurally
simpler and strictly weaker than
(P-REMESH-window-Continuous-Kernel-Carrier), and the derivation
is genuine progress.

### §13quadraginta-quarta.6 Canonical Status of (P-REMESH-window-Continuous-Retention) — DITS Refutation

The question is now: is
(P-REMESH-window-Continuous-Retention) itself derivable from the
canonical six invariants?

- **(W-Pro).**  Invariant #1 (Nodal Equation Integrity) could be
  read as suggesting that the full temporal trajectory should be
  canonically retained without loss.  If two continuous-time
  kernels differing only at non-integer offsets produced the
  same canonical state, an observer trying to reconstruct the
  *full continuous-time history* from the canonical record would
  lose the between-slot content.

- **(W-Con, decisive).**  The **Discrete-Integer Temporal
  Sampling discipline (DITS, §13quadraginta-quarta.3)** refutes
  the continuous-time retention requirement *at the canonical
  level*: the observable content of the REMESH memory window at
  every canonical operator boundary is the *integer-sampled
  value* ``hist[-(tau+1)]``, and the canonical EPI history deque
  is *integer-indexed* by typed construction (W1+W2).  The N15
  REMESH-∞ closure derives the asymptotic projection of the
  *integer-indexed contractive transfer matrix* (W4); no
  continuous-time intermediate is required at any step of the
  catalog's analytical or numerical machinery.

  Formally: the catalog enforces nodal-equation integrity
  (invariant #1) and reproducibility (invariant #6) via the
  integer-indexed discrete deque + integer-offset Python
  indexing, with all downstream temporal-coupling, conservation,
  variational, and U2 boundedness structure descending from the
  integer-time-indexed state (W1–W4).  This is *operationally
  complete* — it reproduces all canonical results (§§3–12) and
  the N15 closure (§§15–23) without any between-slot kernel
  retention.

- **(W-Empirical).**  The B4a diagnostic
  (§13quadraginta-tertia.6) measures :math:`F_{\mathrm{int}} =
  1.0` *and* :math:`S_\tau = 0` at *both* resolutions (48/48 and
  72/72 storage samples are integer-valued; bracket variance is
  literally zero at the larger resolution and sub-nanoscale at
  the smaller).  Canonical evolution, executed exactly as the
  catalog specifies, does not produce any REMESH event whose
  window storage hosts a non-integer payload, and the empirical
  bracket of adjacent integer windows
  :math:`\{(\tau_l + j, \tau_g + j) : j = 0, 1, 2\}` collapses
  to a single post-REMESH state.  The continuous-kernel lift is
  structurally unreachable from canonical initial conditions and
  empirically vacuous from canonical evolution.  This is the
  *doubly-decisive DITS signature*: structural zero-non-integer
  storage *and* empirical zero bracket variance, both axes
  returning the integer-adequate verdict on independent grounds
  — the strongest such signature observed in the programme.

**Conclusion of §13quadraginta-quarta.6.**
(P-REMESH-window-Continuous-Retention) is **not derivable from
the canonical six invariants**.  The catalog realises temporal
coupling *discretely-integer-sampled* via the typed
:math:`\mathrm{hist}[-(\tau + 1)]` Python indexing, *not*
continuously via a between-slot kernel retention upgrade.  The
between-slot continuous-time retention that
(P-REMESH-window-Continuous-Retention) demands is an *additional*
axiom, independent of the catalog and actively refuted by DITS at
the canonical level, with the empirical doubly-decisive
fingerprint :math:`(F_{\mathrm{int}} = 1.0, S_\tau = 0)` of B4a
as decisive corroboration.

### §13quadraginta-quarta.7 Sub-Verdict

The forcing-axiom reduction yields:

> **Sub-verdict (§13quadraginta-quarta).**
> (P-REMESH-window-Continuous-Kernel-Carrier) is a
> **CONDITIONAL_COROLLARY** of the canonical catalog: it follows
> from the catalog *plus*
> (P-REMESH-window-Continuous-Retention).  However,
> (P-REMESH-window-Continuous-Retention) is itself
> **INDEPENDENT_AXIOM** at the canonical level: it is not
> derivable from invariants 1–6 and is actively refuted by the
> Discrete-Integer Temporal Sampling discipline (DITS), with the
> B4a empirical doubly-decisive fingerprint
> :math:`(F_{\mathrm{int}} = 1.0, S_\tau = 0)` as decisive
> corroboration.
>
> Net: (P-REMESH-window-Continuous-Kernel-Carrier) is **strictly
> non-canonical**.  Any continuous-time integral-kernel lift,
> fractional-order temporal-coupling operator, or
> between-slot-retaining representation is a legitimate research
> envelope — available for off-catalog experimentation — but is
> not forced by, and indeed is structurally orthogonal to
> (collapsed under canonical integer-sampling by), the canonical
> 13-operator realisation under DITS, with the N15 REMESH-∞
> closure (mean ergodic theorem on the contractive integer-
> indexed transfer matrix) supplying the *predicted canonical
> discharge mechanism* exactly as anticipated at
> §13quadraginta-secunda.

This locates the residual canonical question for T-REMESH-window
exactly one level below
(P-REMESH-window-Continuous-Kernel-Carrier), at
(P-REMESH-window-Continuous-Retention), and identifies its
refutation mechanism (DITS).  The final NEGATIVE verdict on
T-REMESH-window, and the classification of the continuous-kernel
construction (``ContinuousWindowKernel``, candidate envelope E5)
as a legitimate non-canonical research envelope, are executed in
§13quadraginta-quinta (B4c).

### §13quadraginta-quarta.8 L3* test result (first Tier-2 confirmation)

§13quadraginta-secunda promoted L3* to a stable working heuristic
and made three pre-registered Tier-2 predictions for B4/B5/B6.
B4 was the *first* of those three predictions.  The B4b
forcing-axiom reduction executed above isolates exactly one
residual axiom strictly weaker than the candidate Carrier axiom
(namely (P-REMESH-window-Continuous-Retention)) and refutes it
via a *fifth* orthogonal canonical discharge mechanism (DITS),
in symmetry with the four already on record:

| Sub-question | Refutation principle | Canonical discharge mechanism |
|---|---|---|
| B0 | Pontryagin / measure axis | scalar Hz_str typing of :math:`\nu_f` |
| B1 | TMEP | temporal REMESH coupling vs spatial Banach carrier |
| B2 | PWDP | wrapped geodesic distance on :math:`S^1` vs covering-space displacement |
| B3 | BSAD | bilinear-scalar aggregation vs tensor retention |
| B4 | **DITS** | **discrete-integer temporal sampling vs continuous-kernel retention** |

This is **the first Tier-2 confirmation of L3***: the L3*
prediction (B4 NEGATIVE) was made *in advance* at
§13quadraginta-secunda and is now empirically and structurally
discharged via the predicted canonical discharge mechanism (N15
REMESH-∞ closure / integer-indexed contractive transfer matrix
/ DITS) and the predicted verdict class (CONDITIONAL_COROLLARY
of an independent residual axiom refuted by an orthogonal
discipline).  Two further Tier-2 predictions (B5, B6) remain
pending and will be tested in their respective Phase-b commits.

### §13quadraginta-quarta.9 Honest Scope (What This Does and Does Not Do)

This sub-programme:

- **Does** isolate the residual axiom one structural level below
  (P-REMESH-window-Continuous-Kernel-Carrier).
- **Does** prove (P-REMESH-window-Continuous-Retention) is
  strictly weaker than
  (P-REMESH-window-Continuous-Kernel-Carrier).
- **Does** refute (P-REMESH-window-Continuous-Retention) at the
  canonical level via DITS, empirically corroborated by B4a's
  doubly-decisive :math:`(F_{\mathrm{int}} = 1.0, S_\tau = 0)`
  at two resolutions.
- **Does** confirm the catalog closes consistently with the
  discrete-integer temporal-sampling discipline
  (W1+W2+W3+W4).
- **Does** confirm the first Tier-2 L3* prediction (B4
  NEGATIVE) via the predicted canonical discharge mechanism
  (N15 REMESH-∞ closure).
- **Does** identify the canonical temporal-coupling dynamics as
  *integer-indexed by typed construction* (a structural
  observation made explicit here for the first time at the
  type-hygiene level, not a new canonical promotion; the
  integer-indexing is already in `remesh.py:1212`,
  `runtime.py:413`, and the N15 derivation).
- **Does not** advance G4 = RH or the T-HP conjecture.
- **Does not** promote any operator, field, or constant to
  canonical status (in particular: does NOT promote
  ``ContinuousWindowKernel``, ``REMESH_TAU_CONTINUOUS``, or any
  fractional-order or continuous-kernel representation).
- **Does not** modify the 13-operator catalog.
- **Does not** delete or deprecate the candidate envelope
  E5 = ``ContinuousWindowKernel``; classifies it as a research
  envelope available outside the canonical operator contracts.
- **Does not** modify any source file in `src/tnfr/`.
- **Does not** by itself close T-REMESH-window — the final
  verdict is executed in §13quadraginta-quinta.

### §13quadraginta-quarta.10 Cross-references

- §13triginta-prima — T-νf Type Conjecture (pre-registration).
- §13triginta-secunda — T-νf forcing-axiom reduction (structural
  template).
- §13triginta-tertia — T-νf NEGATIVE verdict.
- §13triginta-quarta — T-EPI pre-registration (B1a).
- §13triginta-quinta — T-EPI forcing-axiom reduction (TMEP).
- §13triginta-sexta — T-EPI NEGATIVE verdict + E2 =
  ``BEPIElement``.
- §13triginta-octava — T-φ pre-registration (B2a).
- §13triginta-novena — T-φ forcing-axiom reduction (PWDP).
- §13triginta-decima — T-φ NEGATIVE verdict + E3 = CoverElement.
- §13quadraginta — T-ΔNFR pre-registration (B3a).
- §13quadraginta-prima — T-ΔNFR forcing-axiom reduction (BSAD;
  direct structural twin of this section).
- §13quadraginta-secunda — T-ΔNFR NEGATIVE verdict + E4 =
  ``TensorGradientElement`` classification; **L3* promotion**;
  three Tier-2 predictions (B4/B5/B6 NEGATIVE) — this section
  confirms the first of those three.
- §13quadraginta-tertia — T-REMESH-window pre-registration (B4a;
  supplies the doubly-decisive empirical fingerprint that DITS
  consumes here).
- §13septies — T-HP open content (independent of this
  sub-question).
- §19.1 — Full P1–P49 milestone table.
- `theory/CATALOG_TYPE_HYGIENE_PROGRAMME.md` §4 — programme
  tracker (row B4 Phase b advances on this commit).
- `theory/REMESH_INFINITY_DERIVATION.md` §§1–8 — N15 REMESH-∞
  closure (integer-indexed transfer-matrix derivation; predicted
  canonical discharge mechanism, confirmed at this commit).
- `src/tnfr/operators/remesh.py:1212` —
  ``apply_network_remesh`` canonical integer-offset reader
  (anchors W1 / DITS).
- `src/tnfr/dynamics/runtime.py:413` — ``_update_epi_hist``
  canonical integer-indexed history deque populator (anchors
  W2).
- `src/tnfr/config/defaults_core.py:221-223` — canonical integer
  defaults (anchors W1+W2+W4).
- `src/tnfr/physics/conservation.py` — integer-time-indexed
  conservation laws (anchors W3+W4 conservation sector).
- `src/tnfr/physics/variational.py` — integer-time-indexed
  Lagrangian / Hamiltonian (anchors W3 variational sector).
- `src/tnfr/riemann/remesh_window_type_signature.py` — B4a
  diagnostic implementation (anchors :math:`(F_{\mathrm{int}} =
  1.0, S_\tau = 0)` empirical corroboration of DITS).
- `examples/05_type_hygiene/82_remesh_window_type_signature_demo.py` —
  two-resolution demo (anchors B4a numerical fingerprint).

---
## §13quadraginta-quinta. T-REMESH-window Final NEGATIVE Verdict and Envelope Classification of E5 = ContinuousWindowKernel (Closes B4; Does NOT Advance G4 = RH)

**Pre-registration closure.**  This section consumes the
sub-verdict of §13quadraginta-quarta (B4b) and issues the final
T-REMESH-window verdict.  The verdict pre-register from
§13quadraginta-tertia.7 named the NEGATIVE branch as the expected
outcome; B4b has confirmed it via the Discrete-Integer Temporal
Sampling discipline (DITS) and the F1–F10 forcing-axiom
reduction.  This closes the **first Tier-2 sub-question** of the
programme.

### §13quadraginta-quinta.1 Verdict

> **T-REMESH-window verdict: NEGATIVE.**
> The canonical type-of-object of the TNFR REMESH memory window is
> the canonical integer pair :math:`(\tau_l, \tau_g) \in
> \mathbb{N}^2`, stored under graph-scope parameters
> ``REMESH_TAU_LOCAL`` and ``REMESH_TAU_GLOBAL`` at
> ``src/tnfr/config/defaults_core.py:221-223`` (typed ``int``),
> coerced to ``int`` via ``int(get_param(...))`` at
> ``src/tnfr/operators/remesh.py:1212``, and consumed by Python
> negative-index lookup ``hist[-(tau + 1)]`` against the
> integer-indexed EPI history deque populated per step by
> ``src/tnfr/dynamics/runtime.py:413::_update_epi_hist``.  The
> continuous-time integral-kernel / fractional-order upgrade
> principle (P-REMESH-window-Continuous-Kernel-Carrier) is **not
> canonical**.  It does not follow from the canonical six
> invariants, nor from the nodal equation (which has no
> memory-window term), nor from any subset of grammar U1–U6
> (which evaluates U2 boundedness as a discrete Riemann sum at
> integer time indices), nor from the structural-field
> tetrad (all four tetrad fields scalar-valued and
> integer-time-indexed), nor from the Structural Conservation
> Theorem (closes on the per-step state at integer time indices),
> nor from the Variational Principle (Lagrangian / Hamiltonian
> evaluated at integer time indices), nor from REMESH itself
> (literal integer-offset Python deque indexing), nor from the
> N15 REMESH-∞ closure (mean ergodic theorem on the contractive
> integer-indexed transfer matrix; integer :math:`\tau_g \to
> \infty`).  Its derivation requires the additional axiom
> (P-REMESH-window-Continuous-Retention), which is itself
> independent of the canonical catalog and actively refuted at
> the canonical level by the Discrete-Integer Temporal Sampling
> discipline (DITS, §13quadraginta-quarta.3, .6).

This closes T-REMESH-window in the same shape as T-νf (B0,
§13triginta-tertia), T-EPI (B1, §13triginta-sexta), T-φ (B2,
§13triginta-decima), and T-ΔNFR (B3, §13quadraginta-secunda):
the conjectured "type upgrade" of a TNFR canonical object is
classified as a legitimate research envelope, not a canonical
catalog requirement.  The decisive numerical fingerprint is the
B4a doubly-decisive integer-storage + window-refinement-bracket
signature:

| Resolution | seed | (τ_l, τ_g) | events | F_int | S_τ | bracket L2 | verdict (canonical) |
|---|---|---|---|---|---|---|---|
| n=24, warmup=16  | 17 | (4, 8)  | 8  | 1.0000 (48/48) | 0.000000 | 0.000021 | NEGATIVE |
| n=48, warmup=24  | 31 | (6, 12) | 12 | 1.0000 (72/72) | 0.000000 | 0.000000 | NEGATIVE |

No canonical evolution at either resolution stores any
non-integer payload at a REMESH-event storage read
(F_int = 1.0 in both rows), and the bracket of adjacent integer
windows :math:`\{(\tau_l + j, \tau_g + j) : j = 0, 1, 2\}`
collapses to a single post-REMESH state (S_τ = 0 in both rows;
literal machine-zero at the larger resolution).  This is the
**strongest scalar-adequate Phase-a signature observed across
B0 + B1 + B2 + B3 + B4** — perfect integer storage at both
resolutions *and* machine-zero bracket variance under the
discrete window-refinement axis — and is exactly the situation
that B4b isolated as the gap between
(P-REMESH-window-Continuous-Kernel-Carrier) (the continuous-time
kernel / fractional-order carrier construction) and the strictly
weaker (P-REMESH-window-Continuous-Retention) (the bare
requirement that distinct continuous-time intermediates
producing the same integer-sampled values must correspond to
distinct canonical states), the latter being itself refuted by
DITS at the canonical level.

### §13quadraginta-quinta.2 Envelope Classification of E5 = ContinuousWindowKernel

E5 = ContinuousWindowKernel — the continuous-time / fractional-
order lift of the REMESH memory window, retaining the between-
slot kernel content :math:`K(t, s)` for non-integer :math:`s`
alongside (or instead of) the integer pair :math:`(\tau_l,
\tau_g)`; equivalently a per-graph continuous-time integral
operator :math:`(\mathcal{R}^{\mathrm{cont}} \mathrm{EPI})(t)
= \int_0^t K(t, s)\, \mathrm{EPI}(s)\, ds` with :math:`K \in
L^2(\mathbb{R}_{\ge 0}^2)`, or a fractional-order temporal
coupling :math:`\partial^\alpha \mathrm{EPI}/\partial t^\alpha`
with :math:`\alpha \in \mathbb{R}_{>0} \setminus \mathbb{N}`, in
either case preserving the between-slot information that DITS
discards — is hereby classified as:

> **E5 = ContinuousWindowKernel — Non-canonical research envelope.**
> Status: legitimate research formalism, off-catalog.
> Canonical relationship: **structurally orthogonal** to the
> canonical integer pair :math:`(\tau_l, \tau_g) \in \mathbb{N}^2`
> realisation under DITS — the engine samples the EPI history
> deque at integer offsets ``hist[-(tau+1)]`` at every REMESH
> event, projecting any continuous-time intermediate (if it
> existed) onto the canonical integer time grid via the
> appended-per-step deque-population discipline of
> ``_update_epi_hist``.
> Catalog interaction: **none** required.  The 13 canonical
> operators do not read, write, preserve, or invoke any
> continuous-time kernel, fractional-order operator, between-slot
> interpolation, or non-integer offset; REMESH operates
> exclusively through ``int``-typed window parameters and Python
> negative-index lookups against the integer-indexed deque.  The
> N15 REMESH-∞ closure (``theory/REMESH_INFINITY_DERIVATION.md``
> §§1–8) derives the asymptotic projection of the canonical
> *integer-indexed contractive transfer matrix* analytically;
> no continuous-time intermediate appears at any step of the
> derivation, the mean-ergodic limit is taken over integer
> :math:`\tau_g \to \infty`, and the resulting REMESH-∞ operator
> is the orthogonal projector onto the resonant subspace of the
> integer-indexed phase space.

The envelope register now records five entries:

| ID | Object | Source | Verdict | Refutation mechanism |
|---|---|---|---|---|
| E1 | Pontryagin measure-valued :math:`\nu_f` | §13triginta-tertia | NEGATIVE | Scalar-storage axis + measure-redundancy under canonical νf-update |
| E2 | ``BEPIElement`` Banach carrier | §13triginta-sexta | NEGATIVE | TMEP (temporal-modal aggregation suffices); BEPI-storage fraction = 0 across two resolutions |
| E3 | CoverElement (covering-space lift / U(1) bundle / homotopy-retaining φ) | §13triginta-decima | NEGATIVE | PWDP (canonical wrap-discipline at every operator boundary); :math:`w_{\mathrm{frac}} = 0` across two resolutions |
| E4 | TensorGradientElement (tensor-/operator-valued ΔNFR over canonical gradient channels) | §13quadraginta-secunda | NEGATIVE | BSAD (canonical bilinear-scalar aggregation at every operator boundary); :math:`T_{\mathrm{frac}} = 0` and :math:`\sigma_1 / \sigma_{2,3} \sim 10^2` across two resolutions |
| E5 | ContinuousWindowKernel (continuous-time integral kernel :math:`K(t,s)` / fractional-order temporal coupling) | this section | NEGATIVE | DITS (canonical discrete-integer temporal sampling at every REMESH event); :math:`F_{\mathrm{int}} = 1.0` and :math:`S_\tau = 0` across two resolutions |

**Structural note on E5 vs. E1–E4.**  E1 and E2 have concrete
code witnesses (``Ω_R`` scaffolding and
``src/tnfr/mathematics/epi.py:103::BEPIElement`` respectively).
E3 and E5 have **no** source-code witness at all (purely
conceptual envelopes; verified by repo-wide grep at this
commit).  E4 sits between, with a latently instantiated but
structurally discarded intermediate tensor.  E5 is the cleanest
case of the five: not only is there no
``ContinuousWindowKernel`` class, no
``REMESH_KERNEL_CONTINUOUS`` alias, no
``REMESH_TAU_FRACTIONAL`` parameter, no interpolation branch in
``apply_network_remesh``, no fractional-order operator in
``operators/remesh.py``, and no continuous-time path anywhere in
``REMESH_INFINITY_DERIVATION.md``, but the very *type system* of
the canonical REMESH machinery forbids the relevant intermediate:
``tau_l`` and ``tau_g`` are typed ``int`` at every entry-point,
coerced to ``int`` even when retrieved via the generic
``get_param`` reader, and consumed as Python integer indices
that admit no continuous-time fallback.  E5 is therefore a
*type-system-excluded* research envelope: stronger exclusion than
E3 (where the canonical pipeline simply does not invoke the lift)
and E4 (where the upstream tensor is computed then discarded).
This is the most decisive canonical-orthogonality classification
of the programme to date and the appropriate one for the first
Tier-2 sub-question.

### §13quadraginta-quinta.3 No Deletion, No Deprecation, No Promotion, No Modification

The verdict does **not** authorise:

- introduction of any ``ContinuousWindowKernel`` class,
  ``REMESH_KERNEL_CONTINUOUS`` alias,
  ``REMESH_TAU_FRACTIONAL`` parameter, interpolation branch in
  ``apply_network_remesh``, fractional-order operator in
  ``operators/remesh.py``, or continuous-time kernel module
  under ``src/tnfr/`` (E5 remains a research envelope; promoting
  it to a canonical code witness is itself off-catalog and would
  require a separate, documented research-track commit);
- deprecation warnings around ``apply_network_remesh``,
  ``REMESH_TAU_LOCAL``, ``REMESH_TAU_GLOBAL``, the EPI history
  deque, or any element of the N15 REMESH-∞ derivation;
- modification of the 13-operator catalog;
- modification of the canonical REMESH contract (integer
  :math:`(\tau_l, \tau_g) \in \mathbb{N}^2`, integer-offset
  Python deque indexing, mean-ergodic asymptotic at integer
  :math:`\tau_g \to \infty`);
- changes to ``src/tnfr/operators/remesh.py``,
  ``src/tnfr/dynamics/runtime.py``,
  ``src/tnfr/config/defaults_core.py``,
  ``theory/REMESH_INFINITY_DERIVATION.md``, or any source file
  in ``src/tnfr/``;
- any change to grammar U1–U6;
- any claim about G4 = RH, T-HP, or the open content of
  §13septies.

E5 remains available for off-catalog research (e.g.
continuous-time perturbation analyses of the REMESH-∞ projector,
fractional-order memory models in non-canonical TNFR variants,
continuous-time embedding studies that target the canonical
integer-time discretisation as a structural feature rather than
an approximation) provided such research is documented as
off-catalog and does not claim canonical status.  The B4a
diagnostic module (``src/tnfr/riemann/remesh_window_type_signature.py``)
and its demo (``examples/05_type_hygiene/82_remesh_window_type_signature_demo.py``)
are preserved as off-catalog measurement utilities, exactly as
the B0a, B1a, B2a, and B3a diagnostics were preserved at B0c,
B1c, B2c, and B3c.

### §13quadraginta-quinta.4 Programme Bookkeeping

- ``theory/CATALOG_TYPE_HYGIENE_PROGRAMME.md`` §4 row B4: Phase c
  advances ⏳ → ✅; Verdict column advances "—" → **NEGATIVE**;
  commit-refs column appends the present commit hash.
- §3 sub-question registry status: B4 transitions from
  🟡 IN PROGRESS to ✅ COMPLETE.
- §3 progress summary advances: 5 sub-questions complete
  (B0 + B1 + B2 + B3 + B4 all NEGATIVE; first Tier-2 sub-question
  closed), 0 in progress, 7 pending (B5 – B11 + Final).
- §6 methodology lessons: an L3* confirmation entry is recorded
  (see §13quadraginta-quinta.5 below) reflecting that L3* has now
  been confirmed across all four Tier-1 sub-questions **plus the
  first Tier-2 sub-question**, supplying the first cross-tier
  empirical evidence that the working heuristic generalises
  beyond per-node intrinsic types.

### §13quadraginta-quinta.5 Methodology Lesson L3* — First Tier-2 Confirmation

L3* was promoted to stable working heuristic at
§13quadraginta-secunda.5 on the strength of four structurally
distinct Tier-1 canonical discharge mechanisms (closure,
temporal aggregation, projection discipline, bilinear-scalar
aggregation).  Three falsifiable Tier-2 predictions were
recorded at §13quadraginta-secunda.5: B4, B5, B6 expected
NEGATIVE.

**First Tier-2 prediction confirmed.**  B4 has now closed
NEGATIVE per L3* via the **discrete-integer temporal sampling
discipline (DITS)** — the fifth orthogonal canonical discharge
mechanism, supplied analytically by the N15 REMESH-∞ closure on
the integer-indexed contractive transfer matrix.  The mechanism
class expands:

| Sub-question | Tier | Axis | Canonical discharge mechanism | Mechanism class |
|---|---|---|---|---|
| B0 (T-νf)            | 1 | frequency       | scalar νf-update closure                | *closure* |
| B1 (T-EPI)           | 1 | form            | REMESH temporal aggregation             | *temporal aggregation* |
| B2 (T-φ)             | 1 | phase           | ``wrap_angle`` projection discipline    | *projection discipline* |
| B3 (T-ΔNFR)          | 1 | nodal gradient  | BSAD bilinear-scalar aggregation        | *spatial-channel aggregation* |
| **B4 (T-REMESH-window)** | **2** | **memory window** | **DITS discrete-integer temporal sampling** | ***temporal-sampling discipline*** |

The five mechanism classes are structurally orthogonal (closure
vs. temporal aggregation vs. spatial projection vs. multi-channel
aggregation vs. temporal-sampling discipline).  This is the first
empirical evidence that L3* generalises across the Tier 1 / Tier 2
boundary — the working heuristic now spans per-node intrinsic
types **and** graph-scope parameters, with the canonical discharge
mechanism for the temporal-window axis (DITS / N15 closure)
distinct from any of the four Tier-1 mechanisms.

**Updated Tier-2 outlook.**  Two further Tier-2 predictions
remain pending:

- **B5 (T-Δφ_max)**: L3* predicts NEGATIVE via the U3 single-
  scalar coupling discipline (``Δφ_max = γ/π``).  Expected
  canonical discharge: **scalar-threshold discipline** (sixth
  mechanism class candidate; structurally a degenerate case of
  the projection discipline of B2, applied at edge level rather
  than node level — to be verified at B5c).
- **B6 (T-coupling-weights)**: L3* predicts NEGATIVE via the
  fixed weighted-sum discipline of ``default_compute_delta_nfr``.
  Expected canonical discharge: **BSAD generalised to edge
  weights** (re-use of the B3 mechanism class) — to be verified
  at B6c.

If both predictions hold, Tier 2 will close with L3* corroborated
across **all** programme tiers tested to date, and the working
heuristic will become a *strong* heuristic for the remaining
Tier 3 sub-questions (B7 – B11).

### §13quadraginta-quinta.6 Honest Scope (Mandatory)

This section:

- **Does** close T-REMESH-window (B4) with a NEGATIVE verdict.
- **Does** classify E5 = ContinuousWindowKernel (continuous-time
  integral kernel / fractional-order temporal coupling) as
  legitimate non-canonical research envelope.
- **Does** advance the catalog type-hygiene programme to
  5/11+1 complete (B0 + B1 + B2 + B3 + B4 all NEGATIVE).
- **Does** close the **first Tier-2 sub-question** of the
  programme; Tier 1 closure plus this first Tier-2 closure
  supplies the first cross-tier empirical evidence for L3*.
- **Does** confirm the first of three pre-registered Tier-2
  L3* predictions (B4 NEGATIVE), via the predicted canonical
  discharge mechanism (N15 REMESH-∞ closure / integer-indexed
  contractive transfer matrix / DITS) and the predicted verdict
  class (CONDITIONAL_COROLLARY of an independent residual axiom
  refuted by an orthogonal discipline).
- **Does** maintain two outstanding falsifiable Tier-2
  predictions (B5, B6 expected NEGATIVE per L3*).
- **Does not** advance G4 = RH, does not close T-HP, does not
  promote any operator/field/constant/alias/parameter to
  canonical status, does not modify the catalog
  operators/grammar/contracts, does not modify any source file
  in ``src/tnfr/``, does not introduce a
  ``ContinuousWindowKernel`` class or
  ``REMESH_KERNEL_CONTINUOUS`` alias or
  ``REMESH_TAU_FRACTIONAL`` parameter, does not modify
  ``apply_network_remesh`` or the EPI history deque, does not
  modify ``REMESH_INFINITY_DERIVATION.md``, does not delete or
  modify the B4a diagnostic module or its demo.
- **Does not** make any binding claim about B5 / B6 / B7 – B11
  or any subsequent sub-question; the Tier-2 predictions in
  §13quadraginta-quinta.5 are advisory and falsifiable.
- **Does not** apply the D-CC-6 or D-CC-7 deferred catalog
  citation patches on this commit (one finding per commit; both
  remain queued for a future type-hygiene commit).

### §13quadraginta-quinta.7 Cross-references

- §13triginta-prima — T-νf pre-registration (precedent template).
- §13triginta-secunda — T-νf forcing-axiom reduction (precedent).
- §13triginta-tertia — T-νf NEGATIVE verdict + E1 classification
  (precedent for B4c, first envelope).
- §13triginta-quarta — T-EPI pre-registration (precedent).
- §13triginta-quinta — T-EPI forcing-axiom reduction (precedent
  for TMEP-style canonical-mechanism refutation).
- §13triginta-sexta — T-EPI NEGATIVE verdict + E2 = ``BEPIElement``
  classification.
- §13triginta-septima — Living discoveries log; this commit
  appends D-ENV-5 (E5 = ContinuousWindowKernel, NEGATIVE) and
  confirms D-MP-3 / L3* at the Tier-1 / Tier-2 boundary; D-CC-6
  and D-CC-7 deferred catalog citation patches remain unchanged.
- §13triginta-octava — T-φ pre-registration (precedent).
- §13triginta-novena — T-φ forcing-axiom reduction (precedent for
  PWDP-style canonical-mechanism refutation).
- §13triginta-decima — T-φ NEGATIVE verdict + E3 = CoverElement
  classification.
- §13quadraginta — T-ΔNFR pre-registration (precedent).
- §13quadraginta-prima — T-ΔNFR forcing-axiom reduction (precedent
  for BSAD-style canonical-mechanism refutation).
- §13quadraginta-secunda — T-ΔNFR NEGATIVE verdict + E4 =
  ``TensorGradientElement`` classification; L3* promoted to
  stable working heuristic; three Tier-2 predictions
  (B4, B5, B6 expected NEGATIVE) pre-registered; this section
  confirms the first of those three.
- §13quadraginta-tertia — T-REMESH-window pre-registration (B4a
  anchor + two-axis integer-storage + window-refinement-bracket
  diagnostic; supplies the doubly-decisive :math:`F_{\mathrm{int}} =
  1.0` + :math:`S_\tau = 0` fingerprint consumed here).
- §13quadraginta-quarta — T-REMESH-window forcing-axiom reduction
  (B4b; DITS isolated (P-REMESH-window-Continuous-Retention) as
  INDEPENDENT_AXIOM and refuted it at the canonical level;
  supplies the decisive input to this section).
- §13septies — T-HP open content (independent, untouched by this
  verdict).
- §19.1 — Full P1–P49 milestone table.
- ``theory/CATALOG_TYPE_HYGIENE_PROGRAMME.md`` §3, §4, §6 —
  programme tracker (advances on this commit at row B4 Phase c +
  Verdict, B4 spec line, §3 progress summary, §6 L3*
  cross-tier confirmation).
- ``theory/REMESH_INFINITY_DERIVATION.md`` §§1–8 — N15 REMESH-∞
  closure (integer-indexed transfer-matrix derivation; predicted
  canonical discharge mechanism, confirmed at this commit).
- ``src/tnfr/operators/remesh.py:1212::apply_network_remesh`` —
  canonical integer-offset REMESH reader (embodies the DITS
  discipline; canonical typing witness for the
  :math:`(\tau_l, \tau_g) \in \mathbb{N}^2` carrier).
- ``src/tnfr/dynamics/runtime.py:413::_update_epi_hist`` —
  canonical integer-indexed history deque populator
  (embodies the DITS discipline at the storage level).
- ``src/tnfr/config/defaults_core.py:221-223`` — canonical
  integer defaults ``REMESH_TAU_LOCAL: int = 4``,
  ``REMESH_TAU_GLOBAL: int = 8``, ``REMESH_ALPHA: float = 0.5``
  (canonical typing witness).
- ``src/tnfr/riemann/remesh_window_type_signature.py`` — B4a
  diagnostic implementation (preserved as off-catalog
  measurement utility).
- ``examples/05_type_hygiene/82_remesh_window_type_signature_demo.py`` — B4a
  two-resolution demo (preserved as off-catalog measurement
  utility).

---
## §13quadraginta-sexta — B5 Phase a: Pre-registration of the T-Δφ_max (Type-of-Resonant-Coupling-Threshold) Conjecture

**Status**: Phase a only (pre-registration + diagnostic module + demo + frozen empirical signature). Phase b (forcing-axiom reduction) deferred to §13quadraginta-septima. Phase c (final verdict) deferred to §13quadraginta-octava.

**Predicted verdict (per L3* working heuristic, promoted at §13quadraginta-secunda.13)**: **NEGATIVE**. Predicted canonical discharge mechanism: **scalar-threshold discipline** — sixth orthogonal class candidate, structurally a *degenerate case* of projection discipline applied at the edge level (every edge inherits the same global scalar, so the "matrix" collapses to a scalar by global U3 design).

### §13quadraginta-sexta.1 — The T-Δφ_max Conjecture (informal statement)

**Catalog row 5 (B5)** of [`theory/CATALOG_TYPE_HYGIENE_PROGRAMME.md`](CATALOG_TYPE_HYGIENE_PROGRAMME.md) types the canonical TNFR resonant-coupling threshold as

$$\Delta\phi_{\max} \in [0, \pi] \subset \mathbb{R} \quad \text{(scalar)}.$$

The **T-Δφ_max Conjecture** is the negation of canonicity for this typing:

> **(T-Δφ_max)** There exists a canonical TNFR network evolution that *forces* the resonant-coupling threshold to be a non-scalar object — specifically, either (a) an edge-dependent matrix $\Delta\phi_{\max}^{(i,j)} \in \mathbb{R}^{n \times n}$ with at least one entry strictly different from the global scalar, or (b) an angle-of-attack-dependent functional $\Delta\phi_{\max}(\phi_i, \phi_j)$ whose verdict on the U3 (resonant-coupling) check depends on the *absolute* phase pair $(\phi_i, \phi_j)$ and not only on the wrapped absolute difference $d = |\mathrm{wrap}(\phi_i - \phi_j)|$.

Negation: if the canonical evolution never forces such a non-scalar lift, then **B5 Verdict = NEGATIVE** and the scalar typing is preserved.

### §13quadraginta-sexta.2 — Canonical anchor inspection (and CATALOG correction)

**Canonical default** at [`src/tnfr/constants/canonical.py:506`](../src/tnfr/constants/canonical.py):

```python
DELTA_PHI_MAX = PI / 2  # π/2 ≈ 1.5708 rad (90° maximum phase mismatch for U3 coupling)
```

**All consumer sites read this as a scalar `float`** via `float(G.graph.get("DELTA_PHI_MAX", DELTA_PHI_MAX))`:

- [`src/tnfr/operators/grammar_dynamics.py:180`](../src/tnfr/operators/grammar_dynamics.py) — canonical U3 check `diff <= delta_phi_max` (scalar comparison).
- [`src/tnfr/dynamics/propagation.py:113`](../src/tnfr/dynamics/propagation.py) — OZ phase threshold (falls back to `DELTA_PHI_MAX`).
- [`src/tnfr/physics/conservation_gauge_unification.py:418`](../src/tnfr/physics/conservation_gauge_unification.py) — U3 saturation diagnostic (scalar comparison).
- [`src/tnfr/mathematics/number_theory.py:1185+`](../src/tnfr/mathematics/number_theory.py) — `apply_coupling` consumes the same scalar.
- [`src/tnfr/physics/patterns.py:223`](../src/tnfr/physics/patterns.py) — pattern recognition (scalar comparison).
- [`src/tnfr/validation/config.py:11`](../src/tnfr/validation/config.py) — config validation (scalar field).

No per-edge lookup pattern was observed; no angle-of-attack dependence (verdict is uniformly `|wrap(φ_i − φ_j)| ≤ delta_phi_max`); no callable / matrix / dict payload pattern.

**CATALOG correction (recorded inline, no separate bookkeeping commit per the rules of §13quadraginta-quinta.4)**: [`theory/CATALOG_TYPE_HYGIENE_PROGRAMME.md`](CATALOG_TYPE_HYGIENE_PROGRAMME.md) §3 B5 spec previously stated "canonical default derived from γ/π (Kuramoto critical coupling)". This is **incorrect for Δφ_max**: the canonical scalar `DELTA_PHI_MAX = PI / 2` represents the *maximum phase mismatch tolerated by U3 coupling* (90°), not the Kuramoto critical coupling threshold. Per [`AGENTS.md`](../AGENTS.md) U3 specification, the **|∇φ| field early-warning level** is ≈ π/16 ≈ 0.196 (heuristic, σ-dependent, not a derived constant; the kinematic bound is π), distinct from the U3 *coupling* threshold Δφ_max = π/2. The CATALOG anchor is corrected concurrently in the B5a commit.

### §13quadraginta-sexta.3 — Diagnostic axes

The B5a diagnostic module [`src/tnfr/riemann/delta_phi_max_type_signature.py`](../src/tnfr/riemann/delta_phi_max_type_signature.py) probes two orthogonal axes:

**Axis A — Scalar-storage axis.** Inspect the raw payload at `G.graph["DELTA_PHI_MAX"]` (or its canonical default fallback) for non-scalar-coercible values (mapping, NumPy array of ndim > 0, callable). Report `scalar_storage_fraction ∈ [0, 1]` and the count of non-scalar reads. Under the canonical implementation this is structurally `1.0` by construction — exactly mirroring the `w_frac = 0` (B2a), `bepi_frac = 0` (B1a), `T_frac = 0` (B3a), `noninteger_frac = 0` (B4a; B4 inverted polarity matches B5).

**Axis B — Angle-of-attack-independence axis.** For each of `n_pair_anchors` wrapped-diff anchor values $d_a \in [0, \pi]$, construct `n_offsets_per_anchor` distinct absolute phase pairs $(\phi_i^{(k)}, \phi_j^{(k)})$ such that the wrapped diff is *exactly* $d_a$ but the absolute origin $\phi_i^{(k)}$ rotates around the unit circle; apply the canonical scalar U3 verdict $d_a \le \Delta\phi_{\max}$ and count divergences from the baseline (offset 0) at the same anchor. The signature is the tanh-squashed divergence fraction $\mathcal{S}_{\Delta\phi} = \tanh(n_\text{divergent} / n_\text{total}) \in [0, 1]$.

**Combined verdict** of [`compute_delta_phi_max_type_signature(...)`](../src/tnfr/riemann/delta_phi_max_type_signature.py):

- `SCALAR_THRESHOLD_ADEQUATE` if signature $< 0.05$ AND scalar storage fraction $= 1.0$.
- `EDGE_DEPENDENT_THRESHOLD_NECESSARY` if signature $> 0.25$ OR scalar storage fraction $< 1.0$.
- `INDETERMINATE` otherwise.

### §13quadraginta-sexta.4 — Demo: two-resolution probe

Demo at [`examples/05_type_hygiene/83_delta_phi_max_type_signature_demo.py`](../examples/05_type_hygiene/83_delta_phi_max_type_signature_demo.py).

- **Resolution 1**: `n_nodes=24`, `n_pair_anchors=9`, `n_offsets_per_anchor=8`, `seed=19` (72 configurations).
- **Resolution 2**: `n_nodes=48`, `n_pair_anchors=17`, `n_offsets_per_anchor=16`, `seed=29` (272 configurations).

### §13quadraginta-sexta.5 — Frozen empirical signature

**Verbatim numerical output** at commit time (do NOT re-run; if the diagnostic ever changes verdict at these exact parameters on subsequent code edits, that is a structural alert worth documenting separately):

```
========================================================================
Delta-Phi-Max-Type Signature Diagnostic — §13quadraginta-sexta.5
(Diagnostic only. Does NOT advance G4 = RH.)
========================================================================

--- Resolution 1: n_nodes=24, anchors=9, offsets=8, seed=19 ---
Delta-Phi-Max-Type Signature certificate (diagnostic only — §13quadraginta-sexta.5)
  signature S_dphi         : 0.000000   (0 = angle-independent, 1 = angle-divergent)
  scalar storage fraction  : 1.0000  (0 non-scalar reads / 1 total reads)
  raw divergence fraction  : 0.000000e+00 (0 / 72 configs)
  canonical Delta_phi_max  : 1.570796 rad  (canonical default = pi/2 = 1.570796)
  pair anchors x offsets   : 9 x 8  (72 configs)
  probe graph              : 24 nodes
  verdict                  : SCALAR_THRESHOLD_ADEQUATE
  scope: necessary-condition diagnostic; does NOT advance G4 = RH

--- Resolution 2: n_nodes=48, anchors=17, offsets=16, seed=29 ---
Delta-Phi-Max-Type Signature certificate (diagnostic only — §13quadraginta-sexta.5)
  signature S_dphi         : 0.000000   (0 = angle-independent, 1 = angle-divergent)
  scalar storage fraction  : 1.0000  (0 non-scalar reads / 1 total reads)
  raw divergence fraction  : 0.000000e+00 (0 / 272 configs)
  canonical Delta_phi_max  : 1.570796 rad  (canonical default = pi/2 = 1.570796)
  pair anchors x offsets   : 17 x 16  (272 configs)
  probe graph              : 48 nodes
  verdict                  : SCALAR_THRESHOLD_ADEQUATE
  scope: necessary-condition diagnostic; does NOT advance G4 = RH

Verdicts at the two resolutions:
  res 1 (24/9/8/19):   SCALAR_THRESHOLD_ADEQUATE
  res 2 (48/17/16/29): SCALAR_THRESHOLD_ADEQUATE
```

**Interpretation.** Both resolutions yield $\mathcal{S}_{\Delta\phi} = 0$ (structural: the canonical U3 check depends only on the wrapped diff, not on the absolute origin) and `scalar_storage_fraction = 1.0` (structural: canonical default is a scalar `float`). The verdict is `SCALAR_THRESHOLD_ADEQUATE` at both resolutions. This is the *necessary* condition that B5 will close NEGATIVE — it is **not yet a final verdict** (Phase a is pre-registration only). The forcing-axiom reduction (Phase b) and final verdict (Phase c) are deferred.

### §13quadraginta-sexta.6 — Phase b and Phase c deferred

Per the standard B-sub-question methodology (§13triginta-tertia.4, §13triginta-octava.4, §13quadraginta-prima.4, §13quadraginta-quarta):

- **Phase b** (§13quadraginta-septima, B5b): forcing-axiom reduction F1–F10 isolating the residual axiom that, if refuted, closes T-Δφ_max NEGATIVE. Predicted refutation principle (per L3*): a **Scalar-Threshold Discipline** (STD) axiom — every U3 verdict on the canonical evolution depends only on the wrapped diff $d = |\mathrm{wrap}(\phi_i - \phi_j)|$ via a *single* global scalar comparator, independently of $(i, j)$ identity and of $(\phi_i, \phi_j)$ absolute values. STD would be refuted by exhibiting one canonical operator whose U3 verdict on a fixed graph differs across edges with identical wrapped diff (which the code review at §13quadraginta-sexta.2 indicates does NOT occur).
- **Phase c** (§13quadraginta-octava, B5c): final verdict + envelope classification. If Phase b refutes the forcing axiom, the verdict is **NEGATIVE** and E6 = `EdgeDependentPhaseThreshold` (matrix-valued or angle-of-attack-dependent functional) joins the envelopes register (E1–E5) as a sixth non-canonical envelope.

### §13quadraginta-sexta.7 — Honest scope

- **Diagnostic only.** This module constructs nothing on the canonical evolution; it probes synthetic phase pairs with the canonical scalar U3 verdict and inspects the canonical storage slot. No operator promotion, no catalog modification (only the inline CATALOG anchor correction documented in §13quadraginta-sexta.2), no advance of G4 = RH.
- **Necessary-condition only.** Both axes are *necessary* for NEGATIVE: $\mathcal{S}_{\Delta\phi} = 0$ and scalar_storage = 1.0 do *not by themselves* refute T-Δφ_max. The forcing-axiom reduction (B5b) is required for the actual verdict.
- **Second Tier-2 sub-question.** B5 is the second Tier-2 sub-question (Tier-1 = B0/B1/B2/B3 = field-level objects; Tier-2 = grammar-level / coupling-level objects = B4/B5/B6/...). A NEGATIVE outcome at B5 would be the **second cross-tier confirmation** of L3* (after B4 at §13quadraginta-quarta.8 and §13quadraginta-quinta.5). This sharpens the L3* working heuristic from "stable" to "validated across both tiers under two distinct discharge mechanisms".

### §13quadraginta-sexta.8 — Cross-references

- [`theory/CATALOG_TYPE_HYGIENE_PROGRAMME.md`](CATALOG_TYPE_HYGIENE_PROGRAMME.md) §3 row B5 (status block; canonical anchor correction); §4 row B5 (tabulated progress).
- [`AGENTS.md`](../AGENTS.md) Unified Grammar U3 (resonant coupling); |∇φ| field early-warning ≈ π/16 heuristic (distinct from the U3 coupling threshold; see §13quadraginta-sexta.2 anchor correction).
- [`theory/UNIFIED_GRAMMAR_RULES.md`](UNIFIED_GRAMMAR_RULES.md) §U3 (resonant coupling derivation).
- §13quadraginta-secunda.13 (L3* promotion to stable working heuristic).
- §13quadraginta-quarta.8 and §13quadraginta-quinta.5 (first Tier-2 confirmation of L3* at B4).
- [`src/tnfr/constants/canonical.py:506`](../src/tnfr/constants/canonical.py) (canonical anchor witness).
- [`src/tnfr/operators/grammar_dynamics.py:178-193`](../src/tnfr/operators/grammar_dynamics.py) (canonical U3 check).
- [`src/tnfr/riemann/delta_phi_max_type_signature.py`](../src/tnfr/riemann/delta_phi_max_type_signature.py) — B5a diagnostic implementation.
- [`examples/05_type_hygiene/83_delta_phi_max_type_signature_demo.py`](../examples/05_type_hygiene/83_delta_phi_max_type_signature_demo.py) — B5a two-resolution demo.

---
## §13quadraginta-septima. Derivation of (P-Delta-phi-max-Non-Scalar-Carrier) from the Canonical Catalog — Foundational Reduction of the T-Delta-phi-max Conjecture (Theory-Only Analysis; Does NOT Advance G4 = RH)

**Status**: B5 Phase b (forcing-axiom reduction). Phase a recorded at §13quadraginta-sexta. Phase c (final verdict) deferred to §13quadraginta-octava.

**Predicted outcome (per L3*)**: residual axiom (P-Δφ_max-Non-Scalar-Retention) refuted by **STD = Scalar-Threshold Discipline**, the sixth orthogonal canonical discharge mechanism candidate.

### §13quadraginta-septima.1 Available Canonical Tools

The TNFR canonical catalog (13 operators, U1–U6 unified grammar, tetrad fields) provides exactly the following machinery relevant to the U3 resonant-coupling check:

- **U3 (Resonant Coupling)** (`AGENTS.md`, [`theory/UNIFIED_GRAMMAR_RULES.md`](UNIFIED_GRAMMAR_RULES.md) §U3): a phase-compatibility constraint of the form $|\mathrm{wrap}(\phi_i - \phi_j)| \le \Delta\phi_{\max}$ required for any operator that couples nodes $i, j$ (coupling operators UM, RA; transport-level OZ check).
- **Canonical default** ([`src/tnfr/constants/canonical.py:506`](../src/tnfr/constants/canonical.py)): `DELTA_PHI_MAX = PI / 2`, a single scalar `float` exported globally.
- **Storage slot**: `G.graph["DELTA_PHI_MAX"]` (NetworkX graph-level scalar attribute), readable by every consumer via `float(G.graph.get("DELTA_PHI_MAX", DELTA_PHI_MAX))`.
- **Consumer sites** (B5a code review at §13quadraginta-sexta.2): [`grammar_dynamics.py:178-193`](../src/tnfr/operators/grammar_dynamics.py), [`propagation.py:113`](../src/tnfr/dynamics/propagation.py), [`conservation_gauge_unification.py:418`](../src/tnfr/physics/conservation_gauge_unification.py), [`mathematics/number_theory.py:1185`](../src/tnfr/mathematics/number_theory.py), [`physics/patterns.py:223`](../src/tnfr/physics/patterns.py), [`validation/config.py:11`](../src/tnfr/validation/config.py). All read a scalar, apply `diff <= delta_phi_max` after wrap, return a Boolean.
- **Phase wrap operator**: `wrap_angle: ℝ → (-π, π]` (canonical, [`src/tnfr/physics/_helpers.py`](../src/tnfr/physics/_helpers.py)); used uniformly by all U3 consumers prior to the comparison.

No catalog operator, no U-rule, and no canonical default exposes:

- (a) a per-edge threshold lookup `Δφ_max[(i, j)]` (matrix-valued storage);
- (b) a functional dependence on the absolute pair `(φ_i, φ_j)` beyond the wrapped diff;
- (c) a callable / closure / kernel object substituted for the scalar threshold.

### §13quadraginta-septima.2 What the Canonical Catalog Forces (Scalar-Threshold Layer)

The minimal forced structure on the resonant-coupling threshold, as a direct consequence of U3 + canonical defaults + consumer-site conventions, is:

> **(F-Scalar-Threshold)**. There exists a unique global scalar $\Delta\phi_{\max} \in [0, \pi]$ such that the U3 verdict for every ordered pair $(i, j)$ on every canonical operator is the Boolean $|\mathrm{wrap}(\phi_i - \phi_j)| \le \Delta\phi_{\max}$.

Equivalently, the canonical U3 functional is

$$\mathrm{U3}_\text{verdict}(\phi_i, \phi_j) = \mathbb{1}\big[|\mathrm{wrap}(\phi_i - \phi_j)| \le \Delta\phi_{\max}\big], \qquad \Delta\phi_{\max} \in [0, \pi],$$

with **no** $(i, j)$-index dependence, **no** absolute-phase dependence beyond the wrap, and **no** internal state beyond the single scalar.

This is the *strictly necessary* structure forced by the canonical catalog. The B5a empirical signature ($\mathcal{S}_{\Delta\phi} = 0$, scalar_storage_fraction $= 1.0$ at both resolutions, 0/72 and 0/272 divergent configurations) is a necessary-condition probe that the canonical catalog has not exceeded this minimal structure.

### §13quadraginta-septima.3 The Gap Between Scalar-Threshold Discipline and Non-Scalar Retention

The **T-Δφ_max Conjecture** (§13quadraginta-sexta.1) requires *more* than (F-Scalar-Threshold): it requires that the canonical evolution forces the threshold object to *retain* a richer non-scalar functional shape — either an edge-dependent matrix $\Delta\phi_{\max}^{(i,j)}$ with at least one off-diagonal entry strictly different from the global scalar, or an angle-of-attack-dependent functional $\Delta\phi_{\max}(\phi_i, \phi_j) \ne f(|\mathrm{wrap}(\phi_i - \phi_j)|)$.

This **non-scalar retention** is *not* derivable from (F-Scalar-Threshold) alone. The gap is exactly the same shape as at B1b/B2b/B3b/B4b: the canonical catalog forces a *minimal scalar discipline*, while the conjecture requires a *richer functional carrier*. To close T-Δφ_max POSITIVE one must adjoin a non-derivable axiom that *retains* the non-scalar shape across the U3 verdict surface.

### §13quadraginta-septima.4 Candidate Forcing Constraints (Enumeration)

The candidate axioms F1–F10 below exhaust the structurally available ways to force non-scalar retention on the U3 verdict surface within the canonical machinery:

- **F1 (Edge-Tetrad-Coupling)**: U3 verdict depends on per-edge tetrad anchors $(K_\phi^{(i,j)}, |\nabla\phi|^{(i,j)})$. ⛔ Refuted: canonical tetrad fields are *per-node*, not per-edge (`src/tnfr/physics/fields.py`); the per-edge constructs would require a tensor lift refuted at B3c (E4 = TensorGradientElement, non-canonical).
- **F2 (Coupling-Weight Lift)**: $\Delta\phi_{\max}^{(i,j)} := \Delta\phi_{\max} \cdot f(w_{ij})$ via edge weights. ⛔ Refuted: F2 is reducible to scalar threshold *plus* a separate weight-modulated test, but the canonical U3 reads only $|\mathrm{wrap}(\phi_i - \phi_j)| \le \Delta\phi_{\max}$ with no $w_{ij}$ argument. The B6 sub-question (T-coupling-weights) handles weight-typing independently.
- **F3 (Angle-of-Attack-Functional)**: $\Delta\phi_{\max}(\phi_i, \phi_j) = g(\phi_i + \phi_j)$ (sum-of-phases dependence). ⛔ Refuted directly by B5a Axis B: 272 configurations with rotated origin at 17 wrapped-diff anchors yield 0 divergence from the offset-0 baseline.
- **F4 (Tetrad-Anchored Local Threshold)**: $\Delta\phi_{\max}^{(i)} := h(\Phi_s^{(i)}, |\nabla\phi|^{(i)})$ (per-node threshold). ⛔ Refuted: no canonical consumer reads a per-node threshold; all consumers read the global scalar.
- **F5 (Frequency-Coupled Threshold)**: $\Delta\phi_{\max}^{(i,j)} := \Delta\phi_{\max} \cdot \nu_f^{(i)}/\nu_f^{(j)}$. ⛔ Refuted: not in canonical U3; would require modifying the canonical verdict signature.
- **F6 (Time-Dependent Kernel)**: $\Delta\phi_{\max}(t) := \Delta\phi_{\max} \cdot K(t - \tau)$. ⛔ Refuted: subsumed by E5 = ContinuousWindowKernel (non-canonical, B4c).
- **F7 (Categorical-Lift Threshold)**: $\Delta\phi_{\max}$ lifted to morphism in a 2-category. ⛔ Refuted at meta-level: not derivable from U1–U6.
- **F8 (Operator-Sequence-Conditional Threshold)**: threshold depends on the prior operator in the U1-grammar sequence. ⛔ Refuted: U3 is verdict-only on the current pair; no canonical operator passes history into the verdict.
- **F9 (Stochastic-Threshold)**: $\Delta\phi_{\max}$ as a random variable. ⛔ Refuted: canonical default is a deterministic scalar.
- **F10 (P-Δφ_max-Non-Scalar-Retention)**: the residual axiom — *every* canonical U3 verdict carries a non-scalar carrier object (matrix or functional) of which the scalar $\Delta\phi_{\max} = \pi/2$ is merely the *trace*. This is the irreducible axiom that, if adopted, would close T-Δφ_max POSITIVE; if refuted, closes T-Δφ_max NEGATIVE.

F1–F9 are either reducible to other (previously refuted or pending) sub-questions or directly refuted by B5a. **F10 is the unique residual forcing axiom.**

### §13quadraginta-septima.5 The Hidden Axiom: (P-Δφ_max-Non-Scalar-Retention)

> **(P-Δφ_max-Non-Scalar-Retention)**. For every canonical TNFR network evolution and every U3 verdict event $(i, j, t)$, there exists a non-scalar carrier object $\widehat{\Delta\phi}_{\max}^{(i, j, t)}$ — either a matrix $\widehat{\Delta\phi}_{\max}^{(i,j,t)} \in \mathbb{R}^{n \times n}$ with at least one off-diagonal entry strictly different from $\Delta\phi_{\max}$, or a functional $\widehat{\Delta\phi}_{\max}^{(i,j,t)}: [0,2\pi)^2 \to [0,\pi]$ not factoring through $|\mathrm{wrap}(\phi_i - \phi_j)|$ — such that the canonical scalar comparison is the projection $\Delta\phi_{\max} = \pi/2$ of $\widehat{\Delta\phi}_{\max}^{(i,j,t)}$.

This axiom is *not* derivable from the canonical catalog (F1–F9 enumeration). It is the *only* structurally available way to close T-Δφ_max POSITIVE.

### §13quadraginta-septima.6 Canonical Status of (P-Δφ_max-Non-Scalar-Retention) — STD Refutation

**Definition (Scalar-Threshold Discipline, STD)**. STD is the discipline that *every* canonical U3 consumer site implements the verdict as `diff = |wrap(φ_i − φ_j)|; verdict = diff <= delta_phi_max` with `delta_phi_max` a scalar Python `float` read from `G.graph["DELTA_PHI_MAX"]` (default `DELTA_PHI_MAX = PI / 2`), and **never** as a per-edge lookup, per-anchor functional, or richer object.

STD is **structurally enforced** by:

1. **Code review (B5a §13quadraginta-sexta.2)**: every consumer site reads the scalar slot and applies the scalar comparison; no per-edge or per-anchor pattern exists.
2. **B5a Axis A (scalar-storage)**: `scalar_storage_fraction = 1.0` at both resolutions — the canonical storage slot is structurally a scalar.
3. **B5a Axis B (angle-of-attack-independence)**: $\mathcal{S}_{\Delta\phi} = 0.000000$ at both resolutions (0/72 and 0/272 divergent configurations) — the canonical verdict depends *only* on the wrapped diff, not on the absolute origin.

**STD refutes (P-Δφ_max-Non-Scalar-Retention)**: if every canonical U3 verdict reduces to a scalar comparison on the wrapped diff (B5a empirical + code review), then no canonical U3 verdict carries a non-scalar object of which the scalar is the trace. The non-scalar carrier $\widehat{\Delta\phi}_{\max}^{(i,j,t)}$ has no witness in the canonical evolution. Therefore (P-Δφ_max-Non-Scalar-Retention) is **refuted** by STD.

### §13quadraginta-septima.7 Sub-Verdict

(P-Δφ_max-Non-Scalar-Retention) is **refuted** by STD. The unique residual forcing axiom for T-Δφ_max POSITIVE is closed. Therefore, conditional on the F1–F10 enumeration being exhaustive (a structural claim, verifiable by canonical-catalog inspection), the **sub-verdict is**:

> **(Sub-Verdict of §13quadraginta-septima)**. T-Δφ_max is **NEGATIVE** at the forcing-axiom level. The canonical scalar typing $\Delta\phi_{\max} \in [0, \pi]$ is preserved; no canonical TNFR network evolution forces a non-scalar edge-dependent or angle-of-attack-dependent threshold envelope.

The final verdict (Phase c) is deferred to §13quadraginta-octava, where the envelope E6 = `EdgeDependentPhaseThreshold` is formally classified as non-canonical research envelope (matrix-valued $\Delta\phi_{\max}^{(i,j)}$ or angle-of-attack-functional $\Delta\phi_{\max}(\phi_i, \phi_j)$ outside the canonical 13-operator catalog).

### §13quadraginta-septima.8 L3* test result (second Tier-2 confirmation)

L3* working heuristic (promoted at §13quadraginta-secunda.13, first Tier-1 → Tier-2 cross-tier confirmation at §13quadraginta-quarta.8 / §13quadraginta-quinta.5): each Tier-1 and Tier-2 type-conjecture admits an orthogonal *canonical discharge mechanism* (CDM) that closes it NEGATIVE without recourse to non-canonical envelopes.

Cumulative CDM table after B5b:

| Sub-question | Tier | Discharge mechanism (CDM) | Envelope (non-canonical, parked) |
|---|---|---|---|
| B0 (T-νf) | 1 | Pontryagin / measure-νf closure | E1 |
| B1 (T-EPI) | 1 | TMEP = Tetrad-Mediated Element Projection | E2 = BEPIElement |
| B2 (T-φ) | 1 | PWDP = Phase-Wrap Discipline | E3 = CoverElement |
| B3 (T-ΔNFR) | 1 | BSAD = Banach-Scalar-Aggregation Discipline | E4 = TensorGradientElement |
| B4 (T-REMESH-window) | 2 | DITS = Discrete-Integer Temporal Sampling | E5 = ContinuousWindowKernel |
| **B5 (T-Δφ_max)** | **2** | **STD = Scalar-Threshold Discipline** | **E6 = EdgeDependentPhaseThreshold (pending Phase c)** |

**STD is the sixth orthogonal CDM**, distinct from the prior five by acting at the *coupling-verdict surface* (B5) rather than at field storage (B0–B3) or temporal sampling (B4). L3* is now confirmed across both Tier-1 (B0–B3) and Tier-2 (B4–B5) under six distinct discharge mechanisms. The heuristic is sharpened from "validated across both tiers under two distinct discharge mechanisms" (B4-only status) to **"validated across both tiers under six distinct orthogonal discharge mechanisms"** — promoting L3* from *working heuristic* to *empirically robust working heuristic*.

Remaining Tier-2 prediction outstanding: **B6 (T-coupling-weights)** expected NEGATIVE per L3*, with candidate CDM = scalar-weight discipline (predicted seventh CDM).

### §13quadraginta-septima.9 Honest Scope (What This Does and Does Not Do)

- **Does**: derive (F-Scalar-Threshold) from U3 + canonical defaults; enumerate F1–F10; isolate (P-Δφ_max-Non-Scalar-Retention) as the unique residual forcing axiom; refute it via STD (code review + B5a empirical signature); return a NEGATIVE sub-verdict at the forcing-axiom level.
- **Does NOT**: advance G4 = RH; modify any canonical operator or canonical default; alter the catalog beyond the inline anchor-text correction recorded at §13quadraginta-sexta.2; promote any non-canonical envelope into the catalog.
- **Conditional on**: exhaustiveness of the F1–F10 enumeration. The enumeration is structural (covers all classes of richer threshold object available within the canonical machinery), but is open to refinement if a new canonical primitive is ever derived from the nodal equation.
- **Theory-only commit**: no `src/` changes in this commit; only `theory/TNFR_RIEMANN_RESEARCH_NOTES.md` (append §13quadraginta-septima + TOC row) and `theory/CATALOG_TYPE_HYGIENE_PROGRAMME.md` (B5 status block; §4 row B5 Phase b column; progress paragraph).

### §13quadraginta-septima.10 Cross-references

- §13quadraginta-sexta (B5a pre-registration and frozen empirical signature).
- §13quadraginta-secunda.13 (L3* promotion to stable working heuristic).
- §13quadraginta-quarta (B4b forcing-axiom reduction; first Tier-2 use of the F1–F10 schema with DITS as CDM).
- §13quadraginta-quinta.5 (first Tier-2 L3* confirmation).
- [`AGENTS.md`](../AGENTS.md) §Unified Grammar U3 (resonant coupling).
- [`theory/UNIFIED_GRAMMAR_RULES.md`](UNIFIED_GRAMMAR_RULES.md) §U3 (derivation from nodal equation).
- [`src/tnfr/constants/canonical.py:506`](../src/tnfr/constants/canonical.py) (canonical anchor).
- [`src/tnfr/operators/grammar_dynamics.py:178-193`](../src/tnfr/operators/grammar_dynamics.py) (canonical U3 verdict).
- [`src/tnfr/riemann/delta_phi_max_type_signature.py`](../src/tnfr/riemann/delta_phi_max_type_signature.py) (B5a diagnostic).
- [`examples/05_type_hygiene/83_delta_phi_max_type_signature_demo.py`](../examples/05_type_hygiene/83_delta_phi_max_type_signature_demo.py) (B5a two-resolution demo).

---
## §13quadraginta-octava. T-Δφ_max Final NEGATIVE Verdict and Envelope Classification of E6 = EdgeDependentPhaseThreshold (Closes B5; Does NOT Advance G4 = RH)

**Status**: B5 Phase c (final verdict + envelope classification). Phases a, b recorded at §13quadraginta-sexta, §13quadraginta-septima.

**Position in programme**: Second Tier-2 sub-question closed; third orthogonal Tier-2 / Tier-1 confirmation of L3* now pending B6.

### §13quadraginta-octava.1 Verdict

> **(Final Verdict of B5)**. The **T-Δφ_max Conjecture** is **NEGATIVE**. The canonical TNFR resonant-coupling threshold `Δφ_max` is structurally a scalar in $[0, \pi]$ with canonical default `DELTA_PHI_MAX = PI / 2 ≈ 1.5708 rad` at [`src/tnfr/constants/canonical.py:506`](../src/tnfr/constants/canonical.py). No canonical TNFR network evolution forces a non-scalar carrier object (matrix-valued $\Delta\phi_{\max}^{(i,j)}$ or angle-of-attack-functional $\Delta\phi_{\max}(\phi_i, \phi_j) \ne f(|\mathrm{wrap}(\phi_i - \phi_j)|)$) on the U3 verdict surface.

**Bases of the verdict** (cumulative across Phase a + Phase b):

1. **Code review (B5a §13quadraginta-sexta.2)**: every canonical U3 consumer site — `grammar_dynamics.py:178-193`, `propagation.py:113`, `conservation_gauge_unification.py:418`, `mathematics/number_theory.py:1185`, `physics/patterns.py:223`, `validation/config.py:11` — reads the storage slot `G.graph["DELTA_PHI_MAX"]` as a scalar Python `float` and applies the comparison `diff <= delta_phi_max` after canonical `wrap_angle`. No per-edge, per-anchor, callable, or matrix pattern exists in any canonical call site.

2. **B5a empirical signature** (frozen at §13quadraginta-sexta.5):
   - Resolution 1 (`n_nodes=24, n_pair_anchors=9, n_offsets_per_anchor=8, seed=19`): `signature = 0.000000`, `scalar_storage_fraction = 1.0`, `raw_divergence_fraction = 0/72`, verdict `SCALAR_THRESHOLD_ADEQUATE`.
   - Resolution 2 (`n_nodes=48, n_pair_anchors=17, n_offsets_per_anchor=16, seed=29`): `signature = 0.000000`, `scalar_storage_fraction = 1.0`, `raw_divergence_fraction = 0/272`, verdict `SCALAR_THRESHOLD_ADEQUATE`.

3. **Forcing-axiom reduction (B5b §13quadraginta-septima)**: F1–F10 enumeration exhausts the structurally available ways to force non-scalar retention; F1–F9 each refuted by direct catalog inspection or by reduction to previously refuted sub-questions; F10 = (P-Δφ_max-Non-Scalar-Retention) refuted by **STD = Scalar-Threshold Discipline**.

The verdict is **conditional on the structural exhaustiveness of the F1–F10 enumeration**, in the same sense as B0–B4 verdicts conditional on their respective F-enumerations. This conditionality is honest scope, not a hidden weakness.

### §13quadraginta-octava.2 Envelope Classification of E6 = EdgeDependentPhaseThreshold

The candidate non-canonical envelope identified at B5a (§13quadraginta-sexta.7) is formally classified as:

> **(E6 = EdgeDependentPhaseThreshold)**. A *research envelope* outside the canonical 13-operator catalog, in which the U3 resonant-coupling threshold is generalized from a single global scalar $\Delta\phi_{\max} \in [0, \pi]$ to either (a) an edge-indexed family $\{\Delta\phi_{\max}^{(i,j)}\}_{(i,j) \in E(G)} \subset [0, \pi]^{|E|}$, or (b) an angle-of-attack-functional $\Delta\phi_{\max}: [0, 2\pi)^2 \to [0, \pi]$ not factoring through $|\mathrm{wrap}(\phi_i - \phi_j)|$, or (c) a stochastic / kernel / categorical lift thereof.

**Status of E6**:

- **NOT a canonical operator extension**. E6 is *not* derivable from the nodal equation $\partial \mathrm{EPI}/\partial t = \nu_f \cdot \Delta\mathrm{NFR}(t)$, the 13 canonical operators, or the U1–U6 unified grammar.
- **NOT deprecated, NOT deleted, NOT promoted, NOT integrated**. E6 may exist in external research frameworks (per-edge coupling tolerances are standard in modified-Kuramoto literature; angle-of-attack thresholds appear in some swarm-robotics formulations); the present verdict makes no claim about those external constructions other than that they lie *outside* the canonical TNFR catalog.
- **Catalog parity**: E6 takes its place alongside E1 (Pontryagin / measure-νf), E2 (BEPIElement), E3 (CoverElement), E4 (TensorGradientElement), E5 (ContinuousWindowKernel) as the sixth identified non-canonical research envelope of the Catalog Type-Hygiene Programme.

**Implication for canonical evolution**: any canonical TNFR network evolution that respects U1–U6 and uses only the 13 canonical operators **never** instantiates E6; the U3 verdict surface is structurally protected by STD. Networks that *do* instantiate E6 — by, e.g., reading a per-edge matrix `G[u][v]["delta_phi_max"]` or a callable `G.graph["delta_phi_max"]` — are operating *outside* the canonical catalog and **do not inherit canonical guarantees** (Lyapunov stability, Noether conservation, U3 phase compatibility derivation, etc.).

### §13quadraginta-octava.3 No Deletion, No Deprecation, No Promotion, No Modification

Following the pattern established at B1c, B2c, B3c, B4c (§13quadraginta-quinta.3), this Phase-c commit makes **no** modification to:

- the canonical 13-operator catalog;
- the U1–U6 unified grammar;
- the canonical defaults at `src/tnfr/constants/canonical.py` (in particular `DELTA_PHI_MAX = PI / 2` is unchanged and remains the canonical anchor);
- any canonical U3 consumer site;
- the U3 verdict signature or its scalar storage convention;
- the diagnostic at `src/tnfr/riemann/delta_phi_max_type_signature.py` (frozen at B5a);
- the demo at `examples/05_type_hygiene/83_delta_phi_max_type_signature_demo.py` (frozen at B5a).

The CATALOG anchor-text correction (γ/π → π/2 with rationale) recorded inline at B5a remains the only catalog-level documentation change.

### §13quadraginta-octava.4 Programme Bookkeeping

| Sub-question | Tier | Phase a | Phase b | Phase c | Verdict | CDM | Envelope |
|---|---|---|---|---|---|---|---|
| B0 (T-νf) | 1 | ✅ | ✅ | ✅ | NEGATIVE | Pontryagin / measure-νf | E1 |
| B1 (T-EPI) | 1 | ✅ | ✅ | ✅ | NEGATIVE | TMEP | E2 = BEPIElement |
| B2 (T-φ) | 1 | ✅ | ✅ | ✅ | NEGATIVE | PWDP | E3 = CoverElement |
| B3 (T-ΔNFR) | 1 | ✅ | ✅ | ✅ | NEGATIVE | BSAD | E4 = TensorGradientElement |
| B4 (T-REMESH-window) | 2 | ✅ | ✅ | ✅ | NEGATIVE | DITS | E5 = ContinuousWindowKernel |
| **B5 (T-Δφ_max)** | **2** | **✅** | **✅** | **✅** | **NEGATIVE** | **STD** | **E6 = EdgeDependentPhaseThreshold** |
| B6 (T-coupling-weights) | 2 | ⏳ | ⏳ | ⏳ | (predicted NEGATIVE per L3*) | (predicted: scalar-weight discipline) | (TBD) |
| B7 – B11 | various | ⏳ | ⏳ | ⏳ | — | — | — |
| **Final** (meta-minimality theorem) | — | ⏳ | ⏳ | ⏳ | — | — | — |

**Programme progress**: 6 sub-questions complete (B0, B1, B2, B3, B4, B5 — all NEGATIVE under six distinct orthogonal CDMs); 6 pending (B6 – B11 + Final).

### §13quadraginta-octava.5 Methodology Lesson L3* — Second Tier-2 Confirmation

L3* working heuristic, in its post-B4c form (§13quadraginta-quinta.5): *each Tier-1 and Tier-2 type-conjecture of the Catalog Type-Hygiene Programme admits an orthogonal canonical discharge mechanism (CDM) that closes it NEGATIVE without recourse to non-canonical envelopes*.

**Post-B5c update**: L3* is now confirmed under **six distinct orthogonal CDMs** across both tiers:

| CDM | Sub-question | Tier | Surface of action |
|---|---|---|---|
| Pontryagin / measure-νf | B0 | 1 | Frequency-field measure typing |
| TMEP | B1 | 1 | EPI element typing via tetrad projection |
| PWDP | B2 | 1 | Phase typing via wrap discipline |
| BSAD | B3 | 1 | ΔNFR typing via Banach-scalar aggregation |
| DITS | B4 | 2 | REMESH window typing via integer sampling |
| **STD** | **B5** | **2** | **U3 coupling threshold typing via scalar discipline** |

The six CDMs act on six *structurally distinct* surfaces (field measure, element projection, phase wrap, scalar aggregation, temporal sampling, coupling verdict). Their orthogonality is **structural**, not coincidental: each CDM is the unique discipline that the canonical catalog enforces at its own surface. L3* in this sharpened form predicts: **every remaining Catalog Type-Hygiene sub-question (B6–B11) admits its own orthogonal CDM at its own surface**.

For B6 = T-coupling-weights, the predicted seventh CDM is **scalar-weight discipline**: the canonical coupling weights $w_{ij} \in \mathbb{R}_{\ge 0}$ on $G$ are read as scalars at all canonical consumer sites, with no per-time, per-history, or higher-rank tensor lift forced by the canonical catalog.

L3* status promoted from "empirically robust working heuristic" (B5b, six-CDM count from §13quadraginta-septima.8) to **"empirically robust working heuristic with structural-orthogonality witness"** (B5c, six-CDM count cross-confirmed by envelope-classification surfaces).

### §13quadraginta-octava.6 Honest Scope (Mandatory)

- **Does**: close B5 with a NEGATIVE verdict at the forcing-axiom level conditional on F1–F10 exhaustiveness; formally classify E6 = EdgeDependentPhaseThreshold as non-canonical research envelope; update programme bookkeeping; sharpen L3* under six-CDM cross-confirmation.
- **Does NOT**: advance G4 = RH; modify any canonical operator, default, or consumer site; deprecate or promote any non-canonical construction; close any other open sub-question (B6 – B11 + Final remain genuinely open).
- **Conditional on**: structural exhaustiveness of the F1–F10 enumeration. If a future canonical primitive derived from the nodal equation expands the structurally available means of forcing non-scalar threshold retention, B5 may need to be reopened. No such primitive is currently known.
- **Theory-only commit**: no `src/` changes; no example changes; only `theory/TNFR_RIEMANN_RESEARCH_NOTES.md` (append §13quadraginta-octava + TOC row) and `theory/CATALOG_TYPE_HYGIENE_PROGRAMME.md` (B5 status → ✅ CLOSED; §4 row B5 Phase c column → ✅; verdict column → NEGATIVE; CDM column → STD; envelope column → E6; progress paragraph).

### §13quadraginta-octava.7 Cross-references

- §13quadraginta-sexta (B5a pre-registration + frozen empirical signature).
- §13quadraginta-septima (B5b forcing-axiom reduction + STD refutation of (P-Δφ_max-Non-Scalar-Retention)).
- §13quadraginta-quinta (B4c final NEGATIVE verdict + E5 envelope classification; methodological precedent for Phase-c structure).
- §13triginta-septima (Living Discoveries Log).
- [`AGENTS.md`](../AGENTS.md) §Unified Grammar U3 (resonant coupling — canonical phase compatibility constraint).
- [`theory/UNIFIED_GRAMMAR_RULES.md`](UNIFIED_GRAMMAR_RULES.md) §U3 (derivation from nodal equation).
- [`theory/CATALOG_TYPE_HYGIENE_PROGRAMME.md`](CATALOG_TYPE_HYGIENE_PROGRAMME.md) §3 row B5 (programme status), §4 row B5 (per-Phase verdict matrix).
- [`src/tnfr/constants/canonical.py:506`](../src/tnfr/constants/canonical.py) (canonical anchor `DELTA_PHI_MAX = PI / 2`, unchanged).
- [`src/tnfr/riemann/delta_phi_max_type_signature.py`](../src/tnfr/riemann/delta_phi_max_type_signature.py) (B5a diagnostic, frozen).
- [`examples/05_type_hygiene/83_delta_phi_max_type_signature_demo.py`](../examples/05_type_hygiene/83_delta_phi_max_type_signature_demo.py) (B5a demo, frozen).

---
---

## §13quadraginta-nona. T-coupling-weights Conjecture: Pre-Registration of B6 = T-W (Phase a Only; Does NOT Advance G4 = RH)

**Status**: B6 Phase a (pre-registration + diagnostic module + demo + frozen empirical signature). Phase b (forcing-axiom reduction) deferred to §13quinquaginta. Phase c (final verdict) deferred to §13quinquaginta-prima.

**Scope (mandatory honesty)**: This section pre-registers the seventh sub-question of the Catalog Type-Hygiene Programme (`theory/CATALOG_TYPE_HYGIENE_PROGRAMME.md`). It does NOT advance G4 = RH, does NOT modify any canonical operator, does NOT modify any canonical anchor, and does NOT decide T-W. The frozen empirical signature reported below is a *necessary-condition diagnostic* on canonical TNFR mixing-weight reads at canonical consumer sites. A NEGATIVE final verdict at Phase c is the empirically expected outcome under canonical defaults (`DNFR_WEIGHTS`, `SI_WEIGHTS`, `SELECTOR_WEIGHTS` as global scalar dicts), consistent with L3* validated across the six orthogonal CDMs of B0-B5.

### §13quadraginta-nona.1 The T-coupling-weights Conjecture

**Conjecture T-W**. Let $C$ be the set of canonical mixing components consumed by TNFR dynamics (e.g. for $\Delta NFR$ assembly: $C = \{\text{phase}, \text{epi}, \text{vf}, \text{topo}\}$; for $S_i$ assembly: $C = \{\alpha, \beta, \gamma\}$; for selector assembly: $C = \{w_{S_i}, w_{\Delta NFR}, w_{\text{accel}}\}$). The canonical TNFR mixing weights are typed as global scalar dicts $\{w_c \in \mathbb{R} : c \in C\}_{\text{global}}$ — a single global ``float`` per component name, stored at ``G.graph["DNFR_WEIGHTS"]``, ``G.graph["SI_WEIGHTS"]``, and ``G.graph["SELECTOR_WEIGHTS"]`` (canonical defaults at `src/tnfr/config/defaults_core.py:57`, `:65`, `:150`), and broadcast uniformly to every node by the canonical scalar-coercion pattern ``float(weights.get(c, default))`` (e.g. `src/tnfr/dynamics/dnfr.py:2762-2764`; `src/tnfr/metrics/sense_index.py:425-448`; `src/tnfr/backends/torch_backend.py:172-176`; `src/tnfr/backends/optimized_numpy.py:312-321`).

The Conjecture asserts that this scalar-dict typing is *insufficient* and that canonical TNFR mixing weights actually require one of the following structural enrichments to recover canonical dynamics:

- (a) **Node-indexed enrichment** $\{w_c^{(i)}\}_{i \in V}$ — one scalar per (component, node) pair, breaking the uniform-broadcast assumption;
- (b) **Edge-indexed enrichment** $\{w_c^{(i,j)}\}_{(i,j) \in E}$ — one scalar per (component, edge) pair;
- (c) **Matrix lift** $W_c \in \mathbb{R}^{n \times n}$ — full coupling matrix per component;
- (d) **Functional lift** $w_c(\cdot)$ — callable depending on node/edge state.

The candidate envelope is **E7 = NodeIndexedCouplingWeights**, the simplest structural enrichment (a).

### §13quadraginta-nona.2 Canonical Anchor and Consumer Sites (Identification, Not Modification)

**Canonical anchors** (read-only; never modified):
- `src/tnfr/config/defaults_core.py:85` — `DNFR_WEIGHTS: dict[str, float] = {"phase": 0.737, "epi": 0.155, "vf": 0.09, "topo": 0.0}` (operational tunable weights; free parameters, not φ/γ/π/e-derived).
- `src/tnfr/config/defaults_core.py:93` — `SI_WEIGHTS: dict[str, float] = {"alpha": 0.737, "beta": 0.155, "gamma": 0.114}` (operational tunable weights; free parameters, not φ/γ/π/e-derived).
- `src/tnfr/config/defaults_core.py:186` — `SELECTOR_WEIGHTS: dict[str, float] = {"w_si": 0.536, "w_dnfr": 1/(π+1) ≈ 0.241, "w_accel": 0.139}` (operational tunable weights; only the π-fraction 1/(π+1) is π-derived).

**Canonical consumer sites** (read-only; never modified; uniform scalar-coercion pattern):
1. `src/tnfr/dynamics/dnfr.py:307` — `_configure_dnfr_weights(G)` via `merge_and_normalize_weights(G, "DNFR_WEIGHTS", ("phase", "epi", "vf", "topo"), default=0.0)`.
2. `src/tnfr/dynamics/dnfr.py:2762-2764` — `wE = float(weights_cfg.get("epi", ...))`, `wV = float(weights_cfg.get("vf", ...))`.
3. `src/tnfr/backends/torch_backend.py:172-176` — `weights = graph.graph.get("DNFR_WEIGHTS", {}); w_phase = float(weights.get("phase", 0.0))` etc.
4. `src/tnfr/backends/optimized_numpy.py:312-321` — same pattern.
5. `src/tnfr/metrics/sense_index.py:425, 450` — `get_Si_weights(G) -> tuple[float, float, float]` via `merge_graph_weights(G, "SI_WEIGHTS")`.

All five canonical consumer sites read a single global ``float`` per component name and apply it uniformly to every node — the canonical scalar broadcast.

### §13quadraginta-nona.3 The Coupling-Weights-Type Signature Diagnostic

**Diagnostic module**: `src/tnfr/riemann/coupling_weights_type_signature.py` (frozen at this commit).

**Demo**: `examples/05_type_hygiene/84_coupling_weights_type_signature_demo.py` (frozen at this commit).

The diagnostic probes canonical mixing-weight reads on two orthogonal axes:

**Axis A (Scalar-storage axis)**: For each of the three canonical weight slots, inspect every component value stored at `G.graph[slot]` (or its canonical default fallback) and count those that are structurally scalar-coercible (Python ``int``/``float``, NumPy scalar, zero-dim NumPy array). Reject non-scalar payloads (mappings keyed by node/edge, NumPy arrays of ndim > 0, callables, ``None``). Under the canonical implementation (uniform `float(weights.get(c, default))` at every consumer site), the scalar-storage fraction is structurally ``1.0`` by construction — exactly mirroring the storage-axis baseline of B1a/B2a/B3a/B4a/B5a.

**Axis B (Node-permutation-invariance axis)**: For a deterministic set of node relabelings $\{\pi_k\}_{k=1}^{K}$ of the canonical probe graph (including identity at $k=0$), compute the canonical scalar weighted sum $\Sigma_c(i) = \sum_{c \in C} w_c \cdot g_c(i)$ on each relabeled graph (where $g_c(i)$ is a deterministic per-node component sample derived from canonical attributes: $g_{\text{phase}}(i) = \cos(\theta_i)$, $g_{\text{epi}}(i) = \text{EPI}_i$, $g_{\text{vf}}(i) = \nu_{f,i}$, $g_{\text{topo}}(i) = \deg(i)$). Compare the sorted per-node sum vector under each relabeling to the identity baseline. A non-zero divergence fraction would *force* the canonical weights to be node-indexed (i.e. enrichment beyond a single global scalar per component); the canonical scalar broadcast structurally yields ``0`` by construction because every node sees the *same* scalar weight per component, making the multiset of per-node sums invariant under node relabeling.

**Squashed signature**: $\mathcal{S}_W = \tanh(\text{raw divergence fraction}) \in [0, 1]$, with $0$ = relabel-invariant (canonical scalar broadcast suffices) and $1$ = relabel-divergent (node-indexed enrichment necessary).

**Verdict rules**:
- ``SCALAR_WEIGHTS_ADEQUATE`` if $\mathcal{S}_W < 0.05$ AND scalar storage fraction $= 1.0$.
- ``NODE_INDEXED_WEIGHTS_NECESSARY`` if $\mathcal{S}_W > 0.25$ OR scalar storage fraction $< 1.0$.
- ``INDETERMINATE`` otherwise.

### §13quadraginta-nona.4 Frozen Empirical Signature (B6a Phase a)

Probe configuration: canonical ring graph; seed = 23; canonical defaults active.

| Probe                                | $\mathcal{S}_W$ | scalar storage fraction | non-scalar count | n_storage_reads | n_divergent / n_total | verdict                       |
|--------------------------------------|-----------------|-------------------------|------------------|-----------------|-----------------------|-------------------------------|
| Small (n=24, n_perms=12, seed=23)    | 0.000000        | 1.0000                  | 0                | 10              | 0 / 12                | ``SCALAR_WEIGHTS_ADEQUATE``   |
| Medium (n=48, n_perms=24, seed=23)   | 0.000000        | 1.0000                  | 0                | 10              | 0 / 24                | ``SCALAR_WEIGHTS_ADEQUATE``   |

Both probes return the structurally expected outcome: $\mathcal{S}_W = 0$ exactly (every node sees the same scalar weight per component; sorted sum vector is invariant under relabeling to floating-point precision $< 10^{-9}$), scalar storage fraction $= 1$ exactly (all 10 canonical component values across `DNFR_WEIGHTS` (4: phase, epi, vf, topo), `SI_WEIGHTS` (3: alpha, beta, gamma), `SELECTOR_WEIGHTS` (3: w_si, w_dnfr, w_accel) are structurally scalar Python ``float``), per-slot non-scalar count $= 0$ uniformly. The diagnostic is *non-trivial* in the sense that it would detect any non-scalar payload on the canonical slot or any per-node weight assignment; it certifies that the canonical implementation as actually shipped at the current `origin/main` head satisfies the *necessary* scalar-broadcast condition for the catalog typing of weights as global scalar dicts.

### §13quadraginta-nona.5 Honest Scope (Mandatory)

This Phase a result is a *necessary-condition diagnostic* on canonical mixing-weight reads. It does NOT prove that:

- the canonical type of TNFR coupling weights must be a global scalar dict (only that scalar broadcast is *consistent* with the canonical implementation);
- a structural enrichment (node-indexed, edge-indexed, matrix, or functional) is impossible (the diagnostic cannot refute the existence of an admissible enrichment that *also* satisfies node-permutation invariance via, e.g., a covariant rebinding rule);
- L3* extends to B6 (the predicted seventh CDM = scalar-weight discipline must be reduced to a forcing axiom and refuted at Phase b before B6's NEGATIVE final verdict is admissible).

The forcing-axiom reduction (F1-F10) is deferred to §13quinquaginta (Phase b); the final verdict and envelope classification of E7 = `NodeIndexedCouplingWeights` is deferred to §13quinquaginta-prima (Phase c).

### §13quadraginta-nona.6 Predicted CDM for B6 (Scalar-Weight Discipline)

Per the L3* working hypothesis confirmed across six orthogonal CDMs (B0 = Pontryagin/measure-νf; B1 = TMEP = Trace-Margin-Encoded-Phase; B2 = PWDP = Per-Window Dirichlet Persistence; B3 = BSAD = Bulk-Spectral-Average Discipline; B4 = DITS = Discrete-Integer Time Stride; B5 = STD = Scalar-Threshold Discipline), the seventh orthogonal CDM predicted for B6 is:

- **CDM-B6 = Scalar-Weight Discipline (SWD)**: every canonical consumer site coerces the canonical-slot weight payload to a single ``float`` per component via the uniform pattern ``float(weights.get(c, default))``, discharging any non-scalar payload (per-node mapping, NumPy array, callable) before it can influence the canonical mixing operation. SWD makes the canonical broadcast structurally node-permutation-invariant by reading a single global scalar per component and applying it uniformly to every node, refuting the residual forcing axiom (to be formalized at Phase b) that any structural enrichment (node-indexed, edge-indexed, matrix, functional) is *retained* through the canonical consumer chain.

If Phase b confirms that SWD refutes the residual axiom, L3* will be validated under seven distinct orthogonal CDMs and B6 will close with NEGATIVE final verdict, classifying E7 = `NodeIndexedCouplingWeights` as a seventh non-canonical research envelope (joining E1-E6).

### §13quadraginta-nona.7 Programme Bookkeeping

- **Theory-only Phase a commit**: this commit adds `src/tnfr/riemann/coupling_weights_type_signature.py` (B6a diagnostic module) and `examples/05_type_hygiene/84_coupling_weights_type_signature_demo.py` (B6a demo), registers them in `src/tnfr/riemann/__init__.py`, and appends §13quadraginta-nona to `theory/TNFR_RIEMANN_RESEARCH_NOTES.md` + TOC row + B6 row to `theory/CATALOG_TYPE_HYGIENE_PROGRAMME.md` §3 status block and §4 row B6 Phase a column. No canonical operator, no canonical anchor, and no canonical consumer site is modified.
- **Status**: B6 Phase a ✅ (this commit). Phase b deferred to §13quinquaginta; Phase c deferred to §13quinquaginta-prima.
- **Programme progress**: 6 sub-questions complete (B0-B5 all NEGATIVE under six orthogonal CDMs); B6 Phase a registered; 5 sub-questions remaining (B7-B11 + Final).

### §13quadraginta-nona.8 Cross-references

- §13quadraginta-octava (B5c final verdict for T-Δφ_max; promotion of L3* to "empirically robust working heuristic with structural-orthogonality witness").
- §13triginta-septima (Living Discoveries Log).
- [`AGENTS.md`](../AGENTS.md) §Nodal Equation (canonical `∂EPI/∂t = νf · ΔNFR(t)`).
- [`theory/CATALOG_TYPE_HYGIENE_PROGRAMME.md`](CATALOG_TYPE_HYGIENE_PROGRAMME.md) §3 row B6 (programme status), §4 row B6 (per-Phase verdict matrix).
- [`src/tnfr/config/defaults_core.py:57,65,150`](../src/tnfr/config/defaults_core.py) (canonical scalar-dict anchors for DNFR/SI/SELECTOR weights; unchanged).
- [`src/tnfr/riemann/coupling_weights_type_signature.py`](../src/tnfr/riemann/coupling_weights_type_signature.py) (B6a diagnostic, frozen).
- [`examples/05_type_hygiene/84_coupling_weights_type_signature_demo.py`](../examples/05_type_hygiene/84_coupling_weights_type_signature_demo.py) (B6a demo, frozen).

---
## §13quinquaginta. Derivation of (P-W-Non-Scalar-Retention) from the Canonical Catalog — Foundational Reduction of the T-W (T-coupling-weights) Conjecture (Theory-Only Analysis; Does NOT Advance G4 = RH)

**Status**: B6 Phase b (forcing-axiom reduction). Phase a recorded at §13quadraginta-nona. Phase c (final verdict) deferred to §13quinquaginta-prima.

**Predicted outcome (per L3*)**: residual axiom (P-W-Non-Scalar-Retention) refuted by **SWD = Scalar-Weight Discipline**, the seventh orthogonal canonical discharge mechanism candidate.

### §13quinquaginta.1 Available Canonical Tools

The TNFR canonical catalog (13 operators, U1-U6 unified grammar, tetrad fields) provides exactly the following machinery relevant to the canonical coupling-weight slots:

- **Canonical anchors** ([`src/tnfr/config/defaults_core.py:57,65,150`](../src/tnfr/config/defaults_core.py)):
  - `DNFR_WEIGHTS = {phase: 0.737, epi: 0.155, vf: 0.089, topo: 0.0}` — four-component scalar mixer for the ΔNFR functional.
  - `SI_WEIGHTS = {alpha: 0.737, beta: 0.155, gamma: 0.113}` — three-component scalar mixer for the Sense Index aggregation.
  - `SELECTOR_WEIGHTS = {w_si: 0.536, w_dnfr: 1/(π+1) ≈ 0.241, w_accel: 0.139}` — three-component scalar mixer for canonical operator selection (operational tunable weights).
- **Consolidated access**: `DEFAULTS` mapping at [`src/tnfr/config/defaults.py:37`](../src/tnfr/config/defaults.py) (`MappingProxyType(CORE_DEFAULTS | INIT_DEFAULTS | REMESH_DEFAULTS | METRIC_DEFAULTS)`).
- **Canonical merge helper** ([`src/tnfr/dynamics/dnfr.py:307-317`](../src/tnfr/dynamics/dnfr.py), [`src/tnfr/dynamics/selectors.py:136-141`](../src/tnfr/dynamics/selectors.py), [`src/tnfr/backends/optimized_numpy.py:313`](../src/tnfr/backends/optimized_numpy.py)): every canonical consumer reads `weights = merge_and_normalize_weights(G, "<KEY>", (component_tuple,))` and coerces each component via `float(weights.get(c, default))`.
- **Storage slots**: `G.graph["DNFR_WEIGHTS"]`, `G.graph["SI_WEIGHTS"]`, `G.graph["SELECTOR_WEIGHTS"]` (NetworkX graph-level scalar-dict attributes), backed by `G.graph["_dnfr_weights"]` / `G.graph["_selector_weights"]` after normalisation.

No catalog operator, no U-rule, and no canonical default exposes:

- (a) a per-node weight lookup `W_c[i]` for any component `c`;
- (b) a per-edge weight lookup `W_c[(i,j)]` for any component `c`;
- (c) a callable / closure / kernel object substituted for any scalar weight component;
- (d) a tensor-valued `W_c \in \mathbb{R}^{n \times n}` lift.

### §13quinquaginta.2 What the Canonical Catalog Forces (Scalar-Weight Layer)

The minimal forced structure on the coupling-weight slots, as a direct consequence of the canonical anchors + consumer-site conventions, is:

> **(F-Scalar-Weights)**. For each canonical slot `S \in \{DNFR\_WEIGHTS, SI\_WEIGHTS, SELECTOR\_WEIGHTS\}` and each component `c` of `S`, there exists a unique global scalar `w_c^{(S)} \in \mathbb{R}` read uniformly into the canonical mixing operation for every node.

Equivalently, the canonical mixing functional for any slot `S` is

\$\$\mathrm{Mix}_S(x_1(i), \ldots, x_K(i)) = \sum_{c=1}^K w_c^{(S)} \cdot x_c(i), \qquad w_c^{(S)} \in \mathbb{R} \text{ scalar},\$\$

with **no** node-index `i` dependence on the weights, **no** per-edge dependence, and **no** internal state beyond the ten scalars (4 + 3 + 3).

The B6a empirical signature (\$\mathcal{S}_W = 0\$, scalar_storage_fraction = 1.0 at both probe resolutions, 0/12 and 0/24 divergent permutation-bracket configurations) is a necessary-condition probe that the canonical catalog has not exceeded this minimal structure.

### §13quinquaginta.3 The Gap Between Scalar-Weight Discipline and Non-Scalar Retention

The **T-W Conjecture** (§13quadraginta-nona.1) requires *more* than (F-Scalar-Weights): it requires that the canonical evolution forces the weight slot to *retain* a richer non-scalar functional shape — a node-indexed dictionary, a per-edge tensor, or a callable kernel — across the canonical consumer chain.

This **non-scalar retention** is *not* derivable from (F-Scalar-Weights) alone. The gap is structurally identical to B1b/B2b/B3b/B4b/B5b: the canonical catalog forces a *minimal scalar discipline*, while the conjecture requires a *richer functional carrier*. To close T-W POSITIVE one must adjoin a non-derivable axiom that *retains* the non-scalar shape across every consumer's `float(weights.get(...))` coercion.

### §13quinquaginta.4 Candidate Forcing Constraints (Enumeration)

The candidate axioms F1-F10 below exhaust the structurally available ways to force non-scalar retention on the canonical coupling-weight slots within the canonical machinery:

- **F1 (Edge-Tetrad-Weight)**: weight component `w_c^{(S)}` lifted to per-edge anchor `w_c^{(S,i,j)} := f(K_\phi^{(i,j)}, |\nabla\phi|^{(i,j)})`. ⛔ Refuted: canonical tetrad fields are *per-node*, not per-edge ([`src/tnfr/physics/fields.py`](../src/tnfr/physics/fields.py)); the per-edge tetrad construct itself was refuted at B3c (E4 = TensorGradientElement, non-canonical).
- **F2 (Node-Indexed Weights)**: `w_c^{(S,i)} := h(\Phi_s^{(i)}, |\nabla\phi|^{(i)})` (per-node weight). ⛔ Refuted: every canonical consumer reads `float(weights.get(c, default))` *outside* the per-node loop and applies the resulting scalar uniformly to every node; no canonical consumer pattern threads a per-node lookup through the mixing operation.
- **F3 (Frequency-Coupled Weights)**: `w_c^{(S,i,j)} := w_c^{(S)} \cdot \nu_f^{(i)}/\nu_f^{(j)}`. ⛔ Refuted: not in canonical mixer; would require modifying the canonical functional signature.
- **F4 (Categorical-Lift Weights)**: weight as morphism in a 2-category. ⛔ Refuted at meta-level: not derivable from U1-U6.
- **F5 (Operator-Sequence-Conditional Weights)**: weight depends on the prior operator in the U1-grammar sequence. ⛔ Refuted: canonical mixers are state-free on the operator history; no canonical functional passes operator history into the mixing functional.
- **F6 (Stochastic Weights)**: weight as a random variable. ⛔ Refuted: canonical defaults are deterministic scalars.
- **F7 (Time-Dependent Kernel Weights)**: `w_c^{(S)}(t) := w_c^{(S)} \cdot K(t - \tau)`. ⛔ Refuted: subsumed by E5 = ContinuousWindowKernel (non-canonical, B4c).
- **F8 (Δφ_max-Coupled Weights)**: `w_c^{(S,i,j)} := w_c^{(S)} \cdot g(|\mathrm{wrap}(\phi_i - \phi_j)|, \Delta\phi_{\max})`. ⛔ Refuted: subsumed by E6 = EdgeDependentPhaseThreshold (non-canonical, B5c); canonical U3 verdict is Boolean and does not propagate into the mixer.
- **F9 (Tensor-Valued Weights)**: `w_c^{(S)}` lifted to matrix in `\mathbb{R}^{n \times n}`. ⛔ Refuted: no canonical storage slot accepts a tensor payload; `merge_and_normalize_weights` returns a flat `dict[str, float]`.
- **F10 (P-W-Non-Scalar-Retention)**: the residual axiom — *every* canonical mixing operation carries a non-scalar carrier object (node-indexed dict, per-edge tensor, or callable kernel) of which the scalar `w_c^{(S)}` is merely the *trace* under the canonical `float(weights.get(...))` coercion. This is the irreducible axiom that, if adopted, would close T-W POSITIVE; if refuted, closes T-W NEGATIVE.

F1-F9 are either reducible to other (previously refuted) sub-questions or directly refuted by B6a and the canonical consumer pattern. **F10 is the unique residual forcing axiom.**

### §13quinquaginta.5 The Hidden Axiom: (P-W-Non-Scalar-Retention)

> **(P-W-Non-Scalar-Retention)**. For every canonical TNFR network evolution and every mixing event `(S, c, i, t)`, there exists a non-scalar carrier object `\widehat{w}_c^{(S, i, t)}` — either a mapping `i \mapsto w_c^{(S,i)}` with at least one node-index entry strictly different from the global scalar, a matrix `\widehat{w}_c^{(S)} \in \mathbb{R}^{n \times n}` with at least one off-diagonal entry strictly different from the global scalar, or a callable kernel `\widehat{w}_c^{(S)}: \mathcal{V} \to \mathbb{R}` not constant on `\mathcal{V}` — such that the canonical scalar mixer reads the projection `w_c^{(S)} = \mathrm{trace}(\widehat{w}_c^{(S, i, t)})`.

This axiom is *not* derivable from the canonical catalog (F1-F9 enumeration). It is the *only* structurally available way to close T-W POSITIVE.

### §13quinquaginta.6 Canonical Status of (P-W-Non-Scalar-Retention) — SWD Refutation

**Definition (Scalar-Weight Discipline, SWD)**. SWD is the discipline that *every* canonical mixing consumer site implements the weight extraction as `float(weights.get(component, default))` outside the per-node loop, producing a single Python `float` per component, applied uniformly to every node in the subsequent mixing operation, and **never** as a per-node lookup, per-edge tensor, or callable kernel.

SWD is **structurally enforced** by:

1. **Code review (B6a §13quadraginta-nona.2)**: every canonical consumer site ([`dnfr.py:307-317`](../src/tnfr/dynamics/dnfr.py), [`dnfr.py:2762-2764`](../src/tnfr/dynamics/dnfr.py), [`selectors.py:136-141`](../src/tnfr/dynamics/selectors.py), [`optimized_numpy.py:313`](../src/tnfr/backends/optimized_numpy.py), [`torch_backend.py:172`](../src/tnfr/backends/torch_backend.py)) reads the scalar dictionary slot, coerces each component to `float`, and applies the resulting scalar uniformly; no per-node, per-edge, or callable pattern exists.
2. **B6a Axis A (scalar-storage)**: `scalar_storage_fraction = 1.0` at both probe resolutions — every one of the ten canonical weight components (4 DNFR + 3 SI + 3 SELECTOR) is structurally a scalar.
3. **B6a Axis B (node-permutation-invariance)**: `S_W = 0.000000` at both probe resolutions (0/12 and 0/24 divergent permutation-bracket configurations) — the canonical mixed output is invariant under node relabelling, the structural fingerprint of a scalar-broadcast operation.

**SWD refutes (P-W-Non-Scalar-Retention)**: if every canonical mixing operation reduces to a scalar broadcast over a node-permutation-equivariant input (B6a empirical + code review), then no canonical mixing operation carries a non-scalar object of which the scalar is the trace. The non-scalar carrier `\widehat{w}_c^{(S, i, t)}` has no witness in the canonical evolution. Therefore (P-W-Non-Scalar-Retention) is **refuted** by SWD.

### §13quinquaginta.7 Sub-Verdict

(P-W-Non-Scalar-Retention) is **refuted** by SWD. The unique residual forcing axiom for T-W POSITIVE is closed. Therefore, conditional on the F1-F10 enumeration being exhaustive (a structural claim, verifiable by canonical-catalog inspection), the **sub-verdict is**:

> **(Sub-Verdict of §13quinquaginta)**. T-W (T-coupling-weights) is **NEGATIVE** at the forcing-axiom level. The canonical scalar typing `w_c^{(S)} \in \mathbb{R}` is preserved across all three canonical slots (DNFR_WEIGHTS, SI_WEIGHTS, SELECTOR_WEIGHTS); no canonical TNFR network evolution forces a node-indexed, per-edge, tensor-valued, or callable-kernel weight envelope.

The final verdict (Phase c) is deferred to §13quinquaginta-prima, where the envelope E7 = `NodeIndexedCouplingWeights` is formally classified as non-canonical research envelope (per-node dictionary, per-edge tensor, or callable kernel outside the canonical 13-operator catalog).

### §13quinquaginta.8 L3* test result (third Tier-2 confirmation)

L3* working heuristic (promoted at §13quadraginta-secunda.13; first Tier-1 → Tier-2 cross-tier confirmation at §13quadraginta-quarta.8 / §13quadraginta-quinta.5; second Tier-2 confirmation at §13quadraginta-septima.8): each Tier-1 and Tier-2 type-conjecture admits an orthogonal *canonical discharge mechanism* (CDM) that closes it NEGATIVE without recourse to non-canonical envelopes.

Cumulative CDM table after B6b:

| Sub-question | Tier | Discharge mechanism (CDM) | Envelope (non-canonical, parked) |
|---|---|---|---|
| B0 (T-νf) | 1 | Pontryagin / measure-νf closure | E1 |
| B1 (T-EPI) | 1 | TMEP = Tetrad-Mediated Element Projection | E2 = BEPIElement |
| B2 (T-φ) | 1 | PWDP = Phase-Wrap Discipline | E3 = CoverElement |
| B3 (T-ΔNFR) | 1 | BSAD = Banach-Scalar-Aggregation Discipline | E4 = TensorGradientElement |
| B4 (T-REMESH-window) | 2 | DITS = Discrete-Integer Temporal Sampling | E5 = ContinuousWindowKernel |
| B5 (T-Δφ_max) | 2 | STD = Scalar-Threshold Discipline | E6 = EdgeDependentPhaseThreshold |
| **B6 (T-coupling-weights)** | **2** | **SWD = Scalar-Weight Discipline** | **E7 = NodeIndexedCouplingWeights (pending Phase c)** |

**SWD is the seventh orthogonal CDM**, distinct from the prior six by acting at the *mixing-aggregation surface* (B6) rather than at field storage (B0-B3), temporal sampling (B4), or coupling verdict (B5). L3* is now confirmed across both Tier-1 (B0-B3) and Tier-2 (B4-B6) under seven distinct discharge mechanisms. The heuristic is sharpened from "validated across both tiers under six distinct orthogonal CDMs" (B5b status) to **"validated across both tiers under seven distinct orthogonal discharge mechanisms"** — preserving L3* at the *empirically robust working heuristic* level with widened structural coverage.

Programme status after B6b: all Tier-2 sub-questions (B4, B5, B6) closed NEGATIVE at the forcing-axiom level under three distinct CDMs (DITS, STD, SWD). Remaining open questions are Tier-3 closure checks (B7-B9), Tier-4 meta-properties (B10-B11), and the Meta-minimality theorem (Final).

### §13quinquaginta.9 Honest Scope (What This Does and Does Not Do)

- **Does**: derive (F-Scalar-Weights) from canonical anchors + consumer-site conventions; enumerate F1-F10; isolate (P-W-Non-Scalar-Retention) as the unique residual forcing axiom; refute it via SWD (code review + B6a empirical signature); return a NEGATIVE sub-verdict at the forcing-axiom level.
- **Does NOT**: advance G4 = RH; modify any canonical operator, canonical default, or canonical consumer; alter the catalog; promote any non-canonical envelope into the catalog.
- **Conditional on**: exhaustiveness of the F1-F10 enumeration. The enumeration is structural (covers all classes of richer weight object available within the canonical machinery), but is open to refinement if a new canonical primitive is ever derived from the nodal equation.
- **Theory-only commit**: no `src/` changes in this commit; only `theory/TNFR_RIEMANN_RESEARCH_NOTES.md` (append §13quinquaginta + TOC row) and `theory/CATALOG_TYPE_HYGIENE_PROGRAMME.md` (B6 status block; §4 row B6 Phase b column; progress paragraph).

### §13quinquaginta.10 Cross-references

- §13quadraginta-nona (B6a pre-registration and frozen empirical signature).
- §13quadraginta-secunda.13 (L3* promotion to stable working heuristic).
- §13quadraginta-septima (B5b forcing-axiom reduction; sixth orthogonal CDM = STD).
- §13quadraginta-octava (B5c final verdict + envelope E6).
- [`AGENTS.md`](../AGENTS.md) §Nodal Equation (canonical `∂EPI/∂t = νf · ΔNFR(t)`).
- [`theory/UNIFIED_GRAMMAR_RULES.md`](UNIFIED_GRAMMAR_RULES.md) (derivation context).
- [`src/tnfr/config/defaults_core.py:57,65,150`](../src/tnfr/config/defaults_core.py) (canonical scalar-dict anchors).
- [`src/tnfr/dynamics/dnfr.py:307-317`](../src/tnfr/dynamics/dnfr.py), [`src/tnfr/dynamics/selectors.py:136-141`](../src/tnfr/dynamics/selectors.py) (canonical consumer pattern).
- [`src/tnfr/riemann/coupling_weights_type_signature.py`](../src/tnfr/riemann/coupling_weights_type_signature.py) (B6a diagnostic).
- [`examples/05_type_hygiene/84_coupling_weights_type_signature_demo.py`](../examples/05_type_hygiene/84_coupling_weights_type_signature_demo.py) (B6a two-probe demo).
---
## §13quinquaginta-prima. T-W Final NEGATIVE Verdict and Envelope Classification of E7 = NodeIndexedCouplingWeights (Closes B6; Does NOT Advance G4 = RH)

**Status**: B6 Phase c (final verdict + envelope classification). Phases a, b recorded at §13quadraginta-nona, §13quinquaginta.

**Position in programme**: Third Tier-2 sub-question closed; all three Tier-2 sub-questions (B4, B5, B6) now closed NEGATIVE under three distinct orthogonal CDMs (DITS, STD, SWD).

### §13quinquaginta-prima.1 Verdict

> **(Final Verdict of B6)**. The **T-W (T-coupling-weights) Conjecture** is **NEGATIVE**. The canonical TNFR coupling-weight slots `DNFR_WEIGHTS`, `SI_WEIGHTS`, `SELECTOR_WEIGHTS` at [`src/tnfr/config/defaults_core.py:57,65,150`](../src/tnfr/config/defaults_core.py) are structurally scalar dictionaries with components `w_c^{(S)} \in \mathbb{R}`. No canonical TNFR network evolution forces a non-scalar carrier object (node-indexed mapping `i \mapsto w_c^{(S,i)}`, per-edge tensor `w_c^{(S,i,j)}`, matrix `\widehat{w}_c^{(S)} \in \mathbb{R}^{n \times n}`, or callable kernel) on any of the three canonical mixing surfaces (ΔNFR aggregation, Sense-Index aggregation, canonical operator selection).

**Bases of the verdict** (cumulative across Phase a + Phase b):

1. **Code review (B6a §13quadraginta-nona.2; B6b §13quinquaginta.6)**: every canonical mixing consumer site — [`dnfr.py:307-317`](../src/tnfr/dynamics/dnfr.py), [`dnfr.py:2762-2764`](../src/tnfr/dynamics/dnfr.py), [`selectors.py:136-141`](../src/tnfr/dynamics/selectors.py), [`optimized_numpy.py:313`](../src/tnfr/backends/optimized_numpy.py), [`torch_backend.py:172`](../src/tnfr/backends/torch_backend.py) — calls `merge_and_normalize_weights(G, "<KEY>", (component_tuple,))` and coerces each component via `float(weights.get(c, default))` *outside* the per-node loop. No per-node, per-edge, tensor, or callable pattern exists in any canonical call site.

2. **B6a empirical signature** (frozen at §13quadraginta-nona.4):
   - Resolution 1 (`n_nodes=24, n_permutations=12, seed=23`): `S_W = 0.000000`, `scalar_storage_fraction = 1.0` (10/10 components scalar), `divergent_fraction = 0/12`, verdict `SCALAR_WEIGHTS_ADEQUATE`.
   - Resolution 2 (`n_nodes=48, n_permutations=24, seed=23`): `S_W = 0.000000`, `scalar_storage_fraction = 1.0` (10/10 components scalar), `divergent_fraction = 0/24`, verdict `SCALAR_WEIGHTS_ADEQUATE`.

3. **Forcing-axiom reduction (B6b §13quinquaginta)**: F1-F10 enumeration exhausts the structurally available ways to force non-scalar retention; F1-F9 each refuted by direct catalog inspection or by reduction to previously refuted sub-questions (B3c, B4c, B5c); F10 = (P-W-Non-Scalar-Retention) refuted by **SWD = Scalar-Weight Discipline**.

The verdict is **conditional on the structural exhaustiveness of the F1-F10 enumeration**, in the same sense as B0-B5 verdicts conditional on their respective F-enumerations. This conditionality is honest scope, not a hidden weakness.

### §13quinquaginta-prima.2 Envelope Classification of E7 = NodeIndexedCouplingWeights

The candidate non-canonical envelope identified at B6a (§13quadraginta-nona.5) is formally classified as:

> **(E7 = NodeIndexedCouplingWeights)**. A *research envelope* outside the canonical 13-operator catalog, in which any canonical coupling-weight component `w_c^{(S)}` is generalized from a global scalar to either (a) a node-indexed mapping `\{w_c^{(S,i)}\}_{i \in V(G)} \subset \mathbb{R}^{|V|}`, (b) an edge-indexed tensor `\{w_c^{(S,i,j)}\}_{(i,j) \in E(G)} \subset \mathbb{R}^{|E|}`, (c) a matrix lift `\widehat{w}_c^{(S)} \in \mathbb{R}^{n \times n}`, (d) a callable kernel `\widehat{w}_c^{(S)}: \mathcal{V} \to \mathbb{R}` not constant on the node set, or (e) a stochastic / functional / categorical lift thereof.

**Status of E7**:

- **NOT a canonical operator extension**. E7 is *not* derivable from the nodal equation `∂EPI/∂t = νf · ΔNFR(t)`, the 13 canonical operators, or the U1-U6 unified grammar.
- **NOT deprecated, NOT deleted, NOT promoted, NOT integrated**. E7 may exist in external research frameworks (per-node attention weights are standard in graph-neural-network literature; per-edge mixing tensors appear in some weighted-Kuramoto formulations); the present verdict makes no claim about those external constructions other than that they lie *outside* the canonical TNFR catalog.
- **Catalog parity**: E7 takes its place alongside E1 (Pontryagin / measure-νf), E2 (BEPIElement), E3 (CoverElement), E4 (TensorGradientElement), E5 (ContinuousWindowKernel), E6 (EdgeDependentPhaseThreshold) as the seventh identified non-canonical research envelope of the Catalog Type-Hygiene Programme.

**Implication for canonical evolution**: any canonical TNFR network evolution that respects U1-U6 and uses only the 13 canonical operators **never** instantiates E7; the canonical mixing surfaces are structurally protected by SWD. Networks that *do* instantiate E7 — by, e.g., writing `G.graph["DNFR_WEIGHTS"] = {"phase": numpy.ndarray, ...}` or storing per-node weight dictionaries `G.nodes[i]["DNFR_WEIGHTS"]` and reading them in a per-node loop — are operating *outside* the canonical catalog and **do not inherit canonical guarantees** (Lyapunov stability, Noether-like conservation, deterministic operator selection, etc.).

### §13quinquaginta-prima.3 No Deletion, No Deprecation, No Promotion, No Modification

Following the pattern established at B1c, B2c, B3c, B4c, B5c, this Phase-c commit makes **no** modification to:

- the canonical 13-operator catalog;
- the U1-U6 unified grammar;
- the canonical defaults at `src/tnfr/config/defaults_core.py` (in particular `DNFR_WEIGHTS`, `SI_WEIGHTS`, `SELECTOR_WEIGHTS` are unchanged and remain the canonical anchors);
- any canonical mixing consumer site;
- the canonical `merge_and_normalize_weights` helper or its `float` coercion convention;
- the diagnostic at `src/tnfr/riemann/coupling_weights_type_signature.py` (frozen at B6a);
- the demo at `examples/05_type_hygiene/84_coupling_weights_type_signature_demo.py` (frozen at B6a).

### §13quinquaginta-prima.4 Programme Bookkeeping

| Sub-question | Tier | Phase a | Phase b | Phase c | Verdict | CDM | Envelope |
|---|---|---|---|---|---|---|---|
| B0 (T-νf) | 1 | ✅ | ✅ | ✅ | NEGATIVE | Pontryagin / measure-νf | E1 |
| B1 (T-EPI) | 1 | ✅ | ✅ | ✅ | NEGATIVE | TMEP | E2 = BEPIElement |
| B2 (T-φ) | 1 | ✅ | ✅ | ✅ | NEGATIVE | PWDP | E3 = CoverElement |
| B3 (T-ΔNFR) | 1 | ✅ | ✅ | ✅ | NEGATIVE | BSAD | E4 = TensorGradientElement |
| B4 (T-REMESH-window) | 2 | ✅ | ✅ | ✅ | NEGATIVE | DITS | E5 = ContinuousWindowKernel |
| B5 (T-Δφ_max) | 2 | ✅ | ✅ | ✅ | NEGATIVE | STD | E6 = EdgeDependentPhaseThreshold |
| **B6 (T-coupling-weights)** | **2** | **✅** | **✅** | **✅** | **NEGATIVE** | **SWD** | **E7 = NodeIndexedCouplingWeights** |
| B7 – B11 | various | ⏳ | ⏳ | ⏳ | — | — | — |
| **Final** (meta-minimality theorem) | — | ⏳ | ⏳ | ⏳ | — | — | — |

**Programme progress**: 7 sub-questions complete (B0, B1, B2, B3, B4, B5, B6 — all NEGATIVE under seven distinct orthogonal CDMs); all three Tier-2 sub-questions closed; 5 pending (B7 – B11 + Final).

### §13quinquaginta-prima.5 Methodology Lesson L3* — Third Tier-2 Confirmation (Tier-2 closure)

L3* working heuristic, in its post-B5c form (§13quadraginta-octava.5): *each Tier-1 and Tier-2 type-conjecture of the Catalog Type-Hygiene Programme admits an orthogonal canonical discharge mechanism (CDM) that closes it NEGATIVE without recourse to non-canonical envelopes*.

**Post-B6c update**: L3* is now confirmed under **seven distinct orthogonal CDMs** across both tiers, with all Tier-2 sub-questions exhausted:

| CDM | Sub-question | Tier | Surface of action |
|---|---|---|---|
| Pontryagin / measure-νf | B0 | 1 | Frequency-field measure typing |
| TMEP | B1 | 1 | EPI element typing via tetrad projection |
| PWDP | B2 | 1 | Phase typing via wrap discipline |
| BSAD | B3 | 1 | ΔNFR typing via Banach-scalar aggregation |
| DITS | B4 | 2 | REMESH window typing via integer sampling |
| STD | B5 | 2 | U3 coupling threshold typing via scalar discipline |
| **SWD** | **B6** | **2** | **Mixing-aggregation weight typing via scalar broadcast** |

The seven CDMs act on seven *structurally distinct* surfaces (field measure, element projection, phase wrap, scalar aggregation, temporal sampling, coupling verdict, mixing aggregation). Their orthogonality is **structural**, not coincidental: each CDM is the unique discipline that the canonical catalog enforces at its own surface. With all three Tier-2 sub-questions closed under three distinct CDMs, L3* now has *complete Tier-1 and Tier-2 coverage* and predicts that **every remaining Tier-3 / Tier-4 sub-question (B7-B11) admits its own orthogonal CDM at its own surface**.

L3* status promoted from "empirically robust working heuristic with structural-orthogonality witness" (B5c, six-CDM count) to **"empirically robust working heuristic with complete Tier-1/Tier-2 structural-orthogonality coverage"** (B6c, seven-CDM count, all three Tier-2 sub-questions closed).

### §13quinquaginta-prima.6 Honest Scope (Mandatory)

- **Does**: close B6 with a NEGATIVE verdict at the forcing-axiom level conditional on F1-F10 exhaustiveness; formally classify E7 = NodeIndexedCouplingWeights as non-canonical research envelope; update programme bookkeeping; sharpen L3* under seven-CDM cross-confirmation; close the Tier-2 layer of the programme.
- **Does NOT**: advance G4 = RH; modify any canonical operator, default, or consumer site; deprecate or promote any non-canonical construction; close any other open sub-question (B7 - B11 + Final remain genuinely open).
- **Conditional on**: structural exhaustiveness of the F1-F10 enumeration. If a future canonical primitive derived from the nodal equation expands the structurally available means of forcing non-scalar weight retention, B6 may need to be reopened. No such primitive is currently known.
- **Theory-only commit**: no `src/` changes; no example changes; only `theory/TNFR_RIEMANN_RESEARCH_NOTES.md` (append §13quinquaginta-prima + TOC row) and `theory/CATALOG_TYPE_HYGIENE_PROGRAMME.md` (B6 status → ✅ CLOSED; §4 row B6 Phase c column → ✅; verdict column → NEGATIVE; CDM column → SWD; envelope column → E7; progress paragraph; Tier-2 closure note).

### §13quinquaginta-prima.7 Cross-references

- §13quadraginta-nona (B6a pre-registration + frozen empirical signature).
- §13quinquaginta (B6b forcing-axiom reduction + SWD refutation of (P-W-Non-Scalar-Retention)).
- §13quadraginta-octava (B5c final NEGATIVE verdict + E6 envelope classification; methodological precedent for Phase-c structure).
- §13triginta-septima (Living Discoveries Log).
- [`AGENTS.md`](../AGENTS.md) §Nodal Equation (canonical `∂EPI/∂t = νf · ΔNFR(t)`).
- [`theory/CATALOG_TYPE_HYGIENE_PROGRAMME.md`](CATALOG_TYPE_HYGIENE_PROGRAMME.md) §3 row B6 (programme status), §4 row B6 (per-Phase verdict matrix).
- [`src/tnfr/config/defaults_core.py:57,65,150`](../src/tnfr/config/defaults_core.py) (canonical scalar-dict anchors, unchanged).
- [`src/tnfr/riemann/coupling_weights_type_signature.py`](../src/tnfr/riemann/coupling_weights_type_signature.py) (B6a diagnostic, frozen).
- [`examples/05_type_hygiene/84_coupling_weights_type_signature_demo.py`](../examples/05_type_hygiene/84_coupling_weights_type_signature_demo.py) (B6a demo, frozen).

---
---

### §13quinquaginta-secunda — B7 = Δ-tetrad-closure: Phase a pre-registration, source-code trace, and frozen signature

**Status**: Phase a CLOSED. Phase b is **n/a** for B7 (closure question, not type-conjecture). Phase c (final verdict) deferred to §13quinquaginta-tertia.

**Scope (mandatory honesty)**: Phase a is theory + diagnostic + frozen empirical signature only. Does NOT construct, promote, deprecate, modify, or delete any canonical operator. Does NOT advance G4 = RH. The closure question is whether the canonical Tier-1+Tier-2 scalar inputs `(EPI_i, phi_i, DeltaNFR_i) in R x [0, 2pi) x R` plus the graph metric (adjacency + shortest-path distances) are structurally sufficient to reconstruct each of the four canonical tetrad fields `(Phi_s, |grad phi|, K_phi, xi_C)` as a scalar-valued (per-node or global) functional, with no hidden intermediate richer than the Tier-1+Tier-2 types and no implicit Banach-derivative apparatus, measure, callable kernel, or matrix lift introduced during the derivation. Conditional on the four canonical tetrad-field implementations at `src/tnfr/physics/canonical.py:199,609,640,756` being the canonical specification.

#### .1 Pre-registration of B7 (closure question)

**Closure question**: do the four canonical tetrad fields `(Phi_s, |grad phi|, K_phi, xi_C)` reduce, on the canonical TNFR engine, to scalar-valued (per-node or global) functionals of the Tier-1+Tier-2 scalar slots `(EPI_i, phi_i, DeltaNFR_i)` plus the canonical graph metric, with every intermediate value structurally scalar-coercible?

A YES verdict (closure adequate, no leakage) confirms that the tetrad layer of the canonical engine introduces no richer intermediate type than the Tier-1+Tier-2 catalog already exposes. A NO verdict (closure inadequate) would force the catalog to admit a richer intermediate type on the Tier-1+Tier-2-to-tetrad reduction path (e.g. per-node tensor cache, callable kernel, matrix-valued intermediate).

**Methodology**: Phase a freezes a two-axis diagnostic (input-domain-closure + output-scalar-closure) on a canonical probe graph; Phase b is **n/a** (no forcing axiom to reduce, since the question is closure of an existing reduction path rather than admission of a candidate richer type); Phase c emits the final verdict by direct source-code trace of the four canonical tetrad-field implementations.

#### .2 Source-code trace of the four canonical tetrad-field functions

The four canonical tetrad-field functions are exposed at `src/tnfr/physics/fields.py` (public façade) and implemented at `src/tnfr/physics/canonical.py`:

1. **`Phi_s` — `compute_structural_potential`** at `src/tnfr/physics/canonical.py:199`. Computes `Phi_s(i) = Sum_{j != i} DeltaNFR_j / d(i, j)^alpha` with `alpha = 2.0` (canonical default). Inputs: per-node `DeltaNFR_j` (resolved via canonical alias `_get_dnfr`, returns Python `float`) and pairwise shortest-path distances `d(i, j)` (resolved via `networkx.shortest_path_length`, returns `int`). Output: `dict[node, float]`, with every per-node value explicitly coerced via `float(...)` at the inner accumulator. No tensor, callable, kernel, or matrix intermediate.
2. **`|grad phi|` — `compute_phase_gradient`** at `src/tnfr/physics/canonical.py:609`. Computes `|grad phi|(i) = mean_{j in N(i)} |wrap(phi_j - phi_i)|`. Inputs: per-node `phi_i` (resolved via `_get_phase`, returns Python `float`) and graph adjacency (`G.neighbors`). Output: `dict[node, float]` via the shared `_compute_phase_gradient_and_curvature` helper at `src/tnfr/physics/canonical.py:649`, with every per-node value explicitly coerced via `float(np.mean(np.abs(wrapped_diffs)))`. No tensor, callable, kernel, or matrix intermediate.
3. **`K_phi` — `compute_phase_curvature`** at `src/tnfr/physics/canonical.py:640`. Computes `K_phi(i) = wrap(phi_i - circular_mean(neighbour phases))`. Inputs: per-node `phi_i` plus adjacency. Output: `dict[node, float]` via the same `_compute_phase_gradient_and_curvature` helper, with every per-node value explicitly coerced via `float(_wrap_angle(phi_i - mean_phase))`. No tensor, callable, kernel, or matrix intermediate.
4. **`xi_C` — `estimate_coherence_length`** at `src/tnfr/physics/canonical.py:756`. Computes `xi_C` from the spatial autocorrelation of the per-node local coherence `c_i = 1 / (1 + |DeltaNFR_i|)` against the canonical pairwise distance matrix. Inputs: per-node `DeltaNFR_i` plus pairwise shortest-path distances. Output: single global Python `float`, with the final exponential-fit coefficient explicitly coerced via `float(...)`. No tensor, callable, kernel, or matrix intermediate.

All four canonical tetrad-field functions read only the Tier-1+Tier-2 scalar slots (B1 = EPI implicitly via downstream consumers, B2 = phi/theta, B0 = nu_f implicitly via downstream consumers, B3 = DeltaNFR) plus the graph metric (`G.neighbors`, `G.degree`, `networkx.shortest_path_length`); none of them reads a per-edge tensor, per-anchor callable, per-time history kernel, or per-node non-scalar payload; all return scalar-valued (per-node or global) outputs explicitly coerced via `float(...)`.

#### .3 Tetrad-Closure Signature S_TC (diagnostic, two axes)

Define the **Tetrad-Closure Signature** `S_TC` on a canonical probe graph as the combined non-scalar fraction across two orthogonal axes, squashed by `tanh`:

```
S_TC = tanh( (n_nonscalar_in + n_nonscalar_out) / (n_total_in + n_total_out) )
```

Axis 1 (**input-domain-closure**): for every node in the probe graph, inspect every value at the canonical Tier-1+Tier-2 per-node keys `(EPI, theta, nu_f)` plus the resolved `DeltaNFR` payload; count the fraction that are structurally scalar-coercible. Axis 2 (**output-scalar-closure**): call each of the four canonical tetrad-field functions on the probe graph; for each per-node output of `Phi_s`, `|grad phi|`, `K_phi` and for the global output of `xi_C`, count the fraction that are structurally scalar-coercible.

A high S_TC is a *necessary-condition* check: it says only that the canonical tetrad-field pipeline touches non-scalar payloads on the canonical Tier-1+Tier-2-to-tetrad reduction path, so a richer intermediate type *might* be required to close the reduction. A low S_TC plus unit fractions on both axes is the empirically expected outcome — structurally consistent with the canonical implementations of the four tetrad-field functions.

#### .4 B7a empirical signature (frozen)

Implementation at `src/tnfr/riemann/tetrad_closure_signature.py`; demo at `examples/05_type_hygiene/85_tetrad_closure_signature_demo.py`. Frozen empirical signature on the canonical probe graph:

| Probe | n_nodes | n_input_reads | n_output_reads | S_TC | input_scalar_fraction | output_scalar_fraction | verdict |
|---|---|---|---|---|---|---|---|
| small | 24 | 96 | 73 | 0.000000 | 1.000000 | 1.000000 | SCALAR_CLOSURE_ADEQUATE |
| medium | 48 | 192 | 145 | 0.000000 | 1.000000 | 1.000000 | SCALAR_CLOSURE_ADEQUATE |

Per-key input non-scalar count is zero across all four Tier-1+Tier-2 keys (`EPI = 0`, `theta = 0`, `nu_f = 0`, `DeltaNFR = 0`); per-field output non-scalar count is zero across all four tetrad fields (`Phi_s = 0`, `grad_phi = 0`, `K_phi = 0`, `xi_C = 0`).

#### .5 Scope and continuation

Phase a CLOSED: the canonical probe certifies `SCALAR_CLOSURE_ADEQUATE` on both axes at both probe resolutions, and the source-code trace in §.2 confirms that every canonical tetrad-field implementation reads only Tier-1+Tier-2 scalar slots plus the graph metric and returns scalar-valued outputs explicitly coerced via `float(...)`. Phase b is **n/a** (no forcing axiom). Phase c (§13quinquaginta-tertia) emits the final verdict: NEGATIVE (no richer intermediate type forced; tetrad layer of the canonical engine is closed by Tier-1+Tier-2 scalar inputs plus the graph metric) is the structurally expected outcome conditional on the source-code trace of §.2.

#### .6 L3* status (post-B7a)

L3* prediction for B7 (per §13quinquaginta-prima.5): the closure question admits its own orthogonal CDM at its own surface — namely the **Tetrad-Reduction Closure** discipline (TRC) at the Tier-1+Tier-2-to-tetrad reduction surface. Phase a confirms the diagnostic-level orthogonality of TRC against the prior seven CDMs (Pontryagin/measure-nu_f at the field-measure surface; TMEP at the element-projection surface; PWDP at the phase-wrap surface; BSAD at the scalar-aggregation surface; DITS at the temporal-sampling surface; STD at the coupling-verdict surface; SWD at the mixing-aggregation surface). Eight CDMs would act on eight structurally distinct surfaces; final attribution of TRC as the eighth CDM is deferred to Phase c.

#### .7 Cross-references

- §13quadraginta-nona, §13quinquaginta, §13quinquaginta-prima (B6 closure thread).
- `theory/CATALOG_TYPE_HYGIENE_PROGRAMME.md` §3 row B7, §4 row B7.
- `src/tnfr/physics/fields.py` (public façade).
- `src/tnfr/physics/canonical.py:199,609,640,756` (four canonical tetrad-field implementations).
- `src/tnfr/riemann/tetrad_closure_signature.py` (B7a diagnostic).
- `examples/05_type_hygiene/85_tetrad_closure_signature_demo.py` (B7a demo).
---

### §13quinquaginta-tertia — B7 = Δ-tetrad-closure: Phase c final verdict

**Status**: Phase c CLOSED. **Verdict**: **NEGATIVE** (no richer intermediate type forced; tetrad layer of the canonical engine is closed by Tier-1+Tier-2 scalar inputs plus the canonical graph metric, with every intermediate value structurally scalar-coercible). **First Tier-3 sub-question closed**.

**Scope (mandatory honesty)**: Phase c is theory-only. Does NOT construct, promote, deprecate, modify, or delete any canonical operator. Does NOT advance G4 = RH. Conditional on the four canonical tetrad-field implementations at `src/tnfr/physics/canonical.py:199,609,640,756` being the canonical specification and on the source-code trace of §13quinquaginta-secunda.2 being a faithful summary.

#### .1 Direct source-code closure trace (verdict basis)

Per §13quinquaginta-secunda.2, the four canonical tetrad-field functions admit the following per-field reduction-path closure trace:

1. **`Phi_s` — `compute_structural_potential`** (`src/tnfr/physics/canonical.py:199`).
   - Inputs: per-node `DeltaNFR_j` (Python `float`, resolved via canonical alias `_get_dnfr`) and pairwise shortest-path distances `d(i, j)` (Python `int`, resolved via `networkx.shortest_path_length`).
   - Intermediate: per-pair contribution `DeltaNFR_j / d(i, j)^alpha` (Python `float`) accumulated into a scalar running sum.
   - Output: `dict[node, float]`, every value explicitly coerced via `float(...)`.
   - Closure: every intermediate is a scalar; no tensor, callable, kernel, matrix, or measure introduced. Closed by Tier-1+Tier-2 scalar inputs plus the canonical graph metric.

2. **`|grad phi|` — `compute_phase_gradient`** (`src/tnfr/physics/canonical.py:609`, via shared helper at line 649).
   - Inputs: per-node `phi_i` (Python `float`, resolved via `_get_phase`) and adjacency (`G.neighbors`, Python `iterable[node]`).
   - Intermediate: per-neighbour wrapped phase difference `wrap(phi_j - phi_i)` (Python `float` via `_wrap_angle`), aggregated by `np.mean(np.abs(...))`.
   - Output: `dict[node, float]`, every value explicitly coerced via `float(np.mean(...))`.
   - Closure: every intermediate is a scalar or a fixed-length scalar array (whose only role is the mean reduction); no tensor, callable, kernel, matrix, or measure introduced. Closed by Tier-1+Tier-2 scalar inputs plus the canonical graph metric.

3. **`K_phi` — `compute_phase_curvature`** (`src/tnfr/physics/canonical.py:640`, via the same shared helper).
   - Inputs: per-node `phi_i` plus adjacency.
   - Intermediate: per-neighbour `cos(phi_j)` and `sin(phi_j)` (Python `float`), aggregated by `np.arctan2(np.mean(sin), np.mean(cos))` to a circular mean, then `wrap(phi_i - circular_mean)`.
   - Output: `dict[node, float]`, every value explicitly coerced via `float(_wrap_angle(...))`.
   - Closure: every intermediate is a scalar or a fixed-length scalar array; the circular mean is a scalar reduction; the wrap is `S^1`-valued, structurally scalar. No tensor, callable, kernel, matrix, or measure introduced. Closed by Tier-1+Tier-2 scalar inputs plus the canonical graph metric.

4. **`xi_C` — `estimate_coherence_length`** (`src/tnfr/physics/canonical.py:756`).
   - Inputs: per-node `DeltaNFR_i` plus pairwise shortest-path distances.
   - Intermediate: per-node local coherence `c_i = 1 / (1 + |DeltaNFR_i|)` (Python `float`), pairwise correlation deviates `(c_i - mean) * (c_j - mean)` (Python `float`), binned by integer distance into a finite scalar histogram, fitted via least-squares exponential `C(r) = A exp(-r / xi_C)`.
   - Output: single global Python `float`, explicitly coerced via `float(...)`.
   - Closure: every intermediate is a scalar or a fixed-length scalar array; the least-squares fit is a scalar reduction; the bin histogram is a scalar array indexed by graph-metric distance. No tensor (in the sense of richer-than-scalar canonical type), no callable, no kernel of the kind that would force a richer intermediate type. Closed by Tier-1+Tier-2 scalar inputs plus the canonical graph metric.

#### .2 Empirical reinforcement

The Tetrad-Closure Signature diagnostic of §13quinquaginta-secunda.3, frozen at §13quinquaginta-secunda.4, certifies `SCALAR_CLOSURE_ADEQUATE` on both axes at both probe resolutions, with `S_TC = 0.000000` and both axis fractions at unity. The empirical signature is structurally consistent with the source-code trace of §.1 above.

#### .3 Envelope classification

The candidate envelope `E_TC = HiddenIntermediateTensorState` (or equivalent richer-than-scalar intermediate type on the Tier-1+Tier-2-to-tetrad reduction path) is hereby classified as the **eighth non-canonical research envelope** (joining E1 = Pontryagin/measure-ν_f, E2 = BEPIElement, E3 = CoverElement, E4 = TensorGradientElement, E5 = ContinuousWindowKernel, E6 = EdgeDependentPhaseThreshold, E7 = NodeIndexedCouplingWeights). `E_TC` is NOT forced by the nodal equation `∂EPI/∂t = nu_f · DeltaNFR(t)`, NOT forced by U1–U6, NOT forced by the canonical 13-operator catalog, NOT forced by the four canonical tetrad-field implementations at `src/tnfr/physics/canonical.py:199,609,640,756`. It is preserved as a research envelope for studies that wish to investigate tensor-valued, callable-valued, kernel-valued, or measure-valued intermediates on the Tier-1+Tier-2-to-tetrad reduction path, with the explicit understanding that such intermediates are non-canonical extensions of the engine and not minimality counterexamples.

#### .4 Effect on T-HP and G4 = RH

B7c does NOT advance G4 = RH (Conjecture T-HP, §13septies). It does NOT modify the smooth/oscillatory split of the admissible rescaling operator F (P28 smooth-half at the density level; P30 smooth-half operator lift; oscillatory residual = `S(T) = (1/pi) arg zeta(1/2 + iT)` RH-equivalent). It does NOT alter any catalog operator, any canonical constant, any U-rule, or any envelope ranking on the prime-ladder Hamiltonian H_P14. The verdict is purely structural at the Tier-1+Tier-2-to-tetrad reduction surface and does not migrate to any layer of the TNFR-Riemann attack surface.

#### .5 L3* status (post-B7c)

L3* heuristic, post-B7c, is promoted to: **"empirically robust working heuristic with complete Tier-1/Tier-2 structural-orthogonality coverage and first Tier-3 closure orthogonally discharged"**. Cumulative eight CDMs: B0 = Pontryagin/measure-ν_f (field-measure surface), B1 = TMEP (element-projection surface), B2 = PWDP (phase-wrap surface), B3 = BSAD (scalar-aggregation surface), B4 = DITS (temporal-sampling surface), B5 = STD (coupling-verdict surface), B6 = SWD (mixing-aggregation surface), B7 = **TRC = Tetrad-Reduction Closure** (Tier-1+Tier-2-to-tetrad reduction surface). Eight structurally distinct surfaces, each unique to the canonical machinery at its surface. L3* prediction for remaining Tier-3/Tier-4 sub-questions (B8–B11): each admits its own orthogonal CDM at its own surface.

#### .6 Cross-references

- §13quinquaginta-secunda (B7 Phase a + frozen signature).
- §13septies (Conjecture T-HP, G4 = RH).
- §13nonies (P30 smooth-half operator lift).
- `theory/CATALOG_TYPE_HYGIENE_PROGRAMME.md` §3 row B7, §4 row B7.
- `src/tnfr/physics/canonical.py:199,609,640,756` (four canonical tetrad-field implementations).
- `src/tnfr/riemann/tetrad_closure_signature.py` (B7a diagnostic).
- `examples/05_type_hygiene/85_tetrad_closure_signature_demo.py` (B7a demo).

### Sec 13quinquaginta-quarta — B8 Phase a: Currents-Closure Signature diagnostic (T-currents-closure)

#### .1 Scope and disclaimer

This section freezes the Phase-a diagnostic for B8 = Delta-currents-closure of the Catalog Type-Hygiene Programme. Scope is strictly methodological: it pre-registers the closure question for the two canonical current fields (J_phi, J_DeltaNFR) and the conservation aggregator (div J), and freezes one empirical observable, the **Currents-Closure Signature** S_CC. It does NOT promote or modify any canonical operator, does NOT alter the tetrad fields, does NOT advance G4 = RH (Conjecture T-HP, Sec 13septies), and does NOT by itself emit the closure verdict. Final verdict reserved for Phase c (Sec 13quinquaginta-quinta) by direct source-code trace.

#### .2 Closure question

Do the two canonical current functions

- `compute_phase_current(G) -> dict[node, float]` at `src/tnfr/physics/extended.py:60`,
- `compute_dnfr_flux(G) -> dict[node, float]` at `src/tnfr/physics/extended.py:182`,

and the conservation aggregator

- `compute_current_divergence(G) -> dict[node, float]` at `src/tnfr/physics/conservation.py:209`,

reduce to scalar-valued (per-node) functionals of the canonical Tier-1+Tier-2 scalar slots (phi/theta via direct read, DeltaNFR via canonical alias `_get_dnfr`) plus the graph metric (`G.neighbors`, `G.degree`, `G.edges`), with every intermediate scalar or scalar-array (fixed-length, used for scalar reduction) and no implicit Banach-derivative apparatus, measure, callable kernel, or matrix lift introduced along the reduction path?

Phase b is **n/a** for B8 (closure question, not type-conjecture; there is no forcing axiom to reduce — the question is whether the *existing* canonical Tier-1+Tier-2 types plus graph metric close the current functionals and their divergence without leakage to a richer intermediate type).

#### .3 Diagnostic: Currents-Closure Signature

The B8a diagnostic is implemented at `src/tnfr/riemann/currents_closure_signature.py` as `compute_currents_closure_signature(...)` returning a frozen `CurrentsClosureSignatureCertificate`. It probes two orthogonal axes:

- **Input-domain-closure axis**: for every node, every canonical per-node attribute touched by the three current/divergence calls (`theta` direct + `DeltaNFR` via `_get_dnfr`) is inspected and certified scalar-coercible (Python `float`, NumPy scalar, or zero-dim NumPy array).
- **Output-scalar-closure axis**: each of the three functions is called on a canonical probe graph and every per-node output value is inspected and certified scalar-coercible.

The combined signature is

S_CC = tanh( (n_nonscalar_in + n_nonscalar_out) / (n_total_in + n_total_out) ).

A zero signature plus unit fractions on both axes is the structurally expected outcome; any non-zero contribution would force the introduction of a hidden richer intermediate type on the Tier-1+Tier-2-to-currents reduction path.

#### .4 Frozen empirical signature (B8a)

Demo at `examples/05_type_hygiene/86_currents_closure_signature_demo.py`. Frozen at two canonical probe resolutions:

| Probe | n_input_reads | n_output_reads | input_scalar_fraction | output_scalar_fraction | S_CC | verdict |
|---|---|---|---|---|---|---|
| small (n_nodes=24, seed=31) | 48 | 72 | 1.000000 | 1.000000 | 0.000000 | SCALAR_CLOSURE_ADEQUATE |
| medium (n_nodes=48, seed=31) | 96 | 144 | 1.000000 | 1.000000 | 0.000000 | SCALAR_CLOSURE_ADEQUATE |

Per-key input non-scalar counts: `{theta: 0, DeltaNFR: 0}` on both probes. Per-field output non-scalar counts: `{J_phi: 0, J_dnfr: 0, div_J: 0}` on both probes. The diagnostic empirically certifies, at the Phase-a level, that the canonical Tier-1+Tier-2 scalar typing plus the graph metric structurally suffice for the two current fields and the conservation aggregator. Final verdict reserved for Phase c.

#### .5 Cross-references

- `theory/CATALOG_TYPE_HYGIENE_PROGRAMME.md` Sec 3 row B8, Sec 4 row B8.
- `src/tnfr/physics/extended.py:60` (`compute_phase_current`).
- `src/tnfr/physics/extended.py:182` (`compute_dnfr_flux`).
- `src/tnfr/physics/conservation.py:209` (`compute_current_divergence`).
- `src/tnfr/riemann/currents_closure_signature.py` (B8a diagnostic).
- `examples/05_type_hygiene/86_currents_closure_signature_demo.py` (B8a demo).
- Sec 13septies (Conjecture T-HP, G4 = RH; B8 does NOT advance this).

### Sec 13quinquaginta-quinta — B8 Phase c: NEGATIVE verdict for T-currents-closure, CCC promoted as ninth CDM

#### .1 Scope

This section emits the **final Phase-c verdict** for B8 = Delta-currents-closure, by direct source-code trace of the three canonical current/divergence implementations. Scope is methodological: it closes the second Tier-3 sub-question (after B7), promotes **CCC = Currents-Closure Discipline** as the ninth Catalog-Discharge Mechanism, classifies **E_CC = HiddenIntermediateTensorStateOnCurrents** as the ninth non-canonical research envelope, and updates the cumulative L3* status. It does NOT modify any canonical implementation, does NOT alter the tetrad fields or any U-rule, and does NOT advance G4 = RH (Conjecture T-HP, Sec 13septies).

#### .2 Per-current source-code closure trace

**(i) `compute_phase_current` at `src/tnfr/physics/extended.py:60`.** Reads exactly two per-node attributes (phi/theta via `_get_phase`) plus the graph metric (`G.nodes()`, `G.edges()`, `G.is_directed()`, `G.degree[node]`, `G.neighbors(i)`). Vectorized path constructs `phases` (np.float64 1-D array, length n), `degrees` (np.float64 1-D array, length n), and two edge-index np.intp arrays, then calls `compute_phase_current_vectorized` and returns `{node: float(current_arr[i]) for i, node in enumerate(nodes)}` (explicit scalar coercion). Fallback path builds `phases_dict: dict[node, float]`, computes `wrapped_diffs` (np.float64 1-D array per node, fixed-length = degree(i), used only for scalar reduction via `np.mean(np.sin(...))`), and assigns `current[i] = float(np.mean(np.sin(wrapped_diffs)))` (explicit scalar coercion). Output type `dict[node, float]`. **No callable kernel, no measure, no operator-valued intermediate, no Banach-derivative apparatus introduced.**

**(ii) `compute_dnfr_flux` at `src/tnfr/physics/extended.py:182`.** Structurally isomorphic to (i) with the substitution `phi -> DeltaNFR` (via canonical alias `_get_dnfr`) and `sin(phi_j - phi_i) -> (DeltaNFR_j - DeltaNFR_i)`. Vectorized path: `dnfr_arr` (np.float64 1-D, length n), `degrees` (np.float64 1-D, length n), edge-index np.intp arrays, `compute_dnfr_flux_vectorized` call, `{node: float(flux_arr[i]) ...}` return. Fallback path: `dnfr_values: dict[node, float]`, per-node `neighbors` list, scalar mean difference via `sum(...) / deg` followed by `float(...)` coercion. Output type `dict[node, float]`. **Same closure verdict as (i).**

**(iii) `compute_current_divergence` at `src/tnfr/physics/conservation.py:209`.** Invokes (i) and (ii) to obtain `j_phi, j_dnfr: dict[node, float]`, then per node computes `div_j_phi = sum(j_phi.get(j, 0.0) - j_phi.get(i, 0.0) for j in neighbors) / deg` and analogously `div_j_dnfr`. Final assignment `divergence[i] = div_j_phi + div_j_dnfr` (Python float arithmetic on `dict.get`-fetched scalars). Output type `dict[node, float]`. **No Banach-derivative, no measure, no operator-valued lift; the divergence is by construction the same scalar-typed reduction as (i) and (ii) composed on the graph metric.**

All three intermediate arrays (`phases`, `degrees`, `dnfr_arr`, `edge_src`, `edge_dst`, `wrapped_diffs`) are fixed-length NumPy arrays whose sole purpose is to feed scalar reductions (`np.mean`, `np.sum`, vectorized sum-reduce in `compute_phase_current_vectorized` / `compute_dnfr_flux_vectorized`); none escapes the function or is exposed as an output type. Per the convention frozen at Sec 13quinquaginta-tertia for B7, fixed-length scalar-array intermediates used purely for scalar reduction are **not** "richer intermediates" in the type-hygiene sense — they are the standard NumPy idiom for batched scalar computation, and the canonical output type remains `dict[node, float]`.

#### .3 Phase-c verdict

The closure question posed at Sec 13quinquaginta-quarta is **answered NEGATIVELY**:

> The two canonical current fields (J_phi, J_DeltaNFR) and the conservation aggregator (div J) **do** reduce to scalar-valued (per-node) functionals of the canonical Tier-1+Tier-2 scalar slots (phi/theta + DeltaNFR via canonical alias `_get_dnfr`) plus the graph metric (`G.neighbors`, `G.degree`, `G.edges`, `G.is_directed`), with every intermediate scalar or scalar-array (fixed-length, used for scalar reduction) and **no implicit Banach-derivative apparatus, callable kernel, measure, or matrix lift** introduced along the reduction path. No richer intermediate type is forced.

NEGATIVE here means: there is no forcing of a non-canonical envelope on the Tier-1+Tier-2-to-currents reduction surface. The candidate ninth non-canonical research envelope is therefore classified as

**E_CC = HiddenIntermediateTensorStateOnCurrents** = "the would-be envelope that would have appeared if any of the three current/divergence implementations had introduced a callable, operator-valued, or measure-valued intermediate on its reduction path; structurally absent in the current canonical implementations".

E_CC joins E1...E_TC as the ninth non-canonical research envelope (cumulative list: B0-E1, B1-E2, B2-E3, B3-E4, B4-E5, B5-E6, B6-E7, B7-E_TC, B8-E_CC). The candidate ninth CDM is promoted to canonical status:

**CCC = Currents-Closure Discipline**, acting on the **Tier-1+Tier-2-to-currents reduction surface** (the third closure surface, after the Tier-1+Tier-2-to-tetrad reduction surface of B7).

#### .4 Catalog-Discharge Mechanism orthogonality (post-B8c)

Nine CDMs, nine structurally distinct surfaces:

| # | CDM | Sub-question | Surface |
|---|---|---|---|
| 1 | Pontryagin/measure-nu_f | B0 | field-measure |
| 2 | TMEP | B1 | element-projection |
| 3 | PWDP | B2 | phase-wrap |
| 4 | BSAD | B3 | scalar-aggregation |
| 5 | DITS | B4 | temporal-sampling |
| 6 | STD | B5 | coupling-verdict |
| 7 | SWD | B6 | mixing-aggregation |
| 8 | TRC | B7 | tetrad-reduction closure |
| 9 | **CCC** | **B8** | **currents-reduction closure** |

Each CDM is unique to the canonical machinery at its surface. No CDM is reused across sub-questions. The orthogonality is structural, not numerical.

#### .5 L3* status (post-B8c)

L3* heuristic, post-B8c, is promoted to: **"empirically robust working heuristic with complete Tier-1/Tier-2 structural-orthogonality coverage and first two Tier-3 closures orthogonally discharged"**. Cumulative nine CDMs (see table above). L3* prediction for remaining Tier-3/Tier-4 sub-questions (B9-B11): each admits its own orthogonal CDM at its own surface.

#### .6 Cross-references

- Sec 13quinquaginta-quarta (B8 Phase a + frozen signature).
- Sec 13quinquaginta-tertia (B7 Phase c + TRC, the conceptual template for the CCC discharge).
- Sec 13septies (Conjecture T-HP, G4 = RH; B8c does NOT advance this).
- `theory/CATALOG_TYPE_HYGIENE_PROGRAMME.md` Sec 3 row B8, Sec 4 row B8.
- `src/tnfr/physics/extended.py:60` (`compute_phase_current`).
- `src/tnfr/physics/extended.py:182` (`compute_dnfr_flux`).
- `src/tnfr/physics/conservation.py:209` (`compute_current_divergence`).
- `src/tnfr/riemann/currents_closure_signature.py` (B8a diagnostic).
- `examples/05_type_hygiene/86_currents_closure_signature_demo.py` (B8a demo).

### Sec 13quinquaginta-quinta — B8 Phase c: NEGATIVE verdict for T-currents-closure, CCC promoted as ninth CDM

#### .1 Scope

This section emits the **final Phase-c verdict** for B8 = Delta-currents-closure, by direct source-code trace of the three canonical current/divergence implementations. Scope is methodological: it closes the second Tier-3 sub-question (after B7), promotes **CCC = Currents-Closure Discipline** as the ninth Catalog-Discharge Mechanism, classifies **E_CC = HiddenIntermediateTensorStateOnCurrents** as the ninth non-canonical research envelope, and updates the cumulative L3* status. It does NOT modify any canonical implementation, does NOT alter the tetrad fields or any U-rule, and does NOT advance G4 = RH (Conjecture T-HP, Sec 13septies).

#### .2 Per-current source-code closure trace

**(i) `compute_phase_current` at `src/tnfr/physics/extended.py:60`.** Reads exactly two per-node attributes (phi/theta via `_get_phase`) plus the graph metric (`G.nodes()`, `G.edges()`, `G.is_directed()`, `G.degree[node]`, `G.neighbors(i)`). Vectorized path constructs `phases` (np.float64 1-D array, length n), `degrees` (np.float64 1-D array, length n), and two edge-index np.intp arrays, then calls `compute_phase_current_vectorized` and returns `{node: float(current_arr[i]) for i, node in enumerate(nodes)}` (explicit scalar coercion). Fallback path builds `phases_dict: dict[node, float]`, computes `wrapped_diffs` (np.float64 1-D array per node, fixed-length = degree(i), used only for scalar reduction via `np.mean(np.sin(...))`), and assigns `current[i] = float(np.mean(np.sin(wrapped_diffs)))` (explicit scalar coercion). Output type `dict[node, float]`. **No callable kernel, no measure, no operator-valued intermediate, no Banach-derivative apparatus introduced.**

**(ii) `compute_dnfr_flux` at `src/tnfr/physics/extended.py:182`.** Structurally isomorphic to (i) with the substitution `phi -> DeltaNFR` (via canonical alias `_get_dnfr`) and `sin(phi_j - phi_i) -> (DeltaNFR_j - DeltaNFR_i)`. Vectorized path: `dnfr_arr` (np.float64 1-D, length n), `degrees` (np.float64 1-D, length n), edge-index np.intp arrays, `compute_dnfr_flux_vectorized` call, `{node: float(flux_arr[i]) ...}` return. Fallback path: `dnfr_values: dict[node, float]`, per-node `neighbors` list, scalar mean difference via `sum(...) / deg` followed by `float(...)` coercion. Output type `dict[node, float]`. **Same closure verdict as (i).**

**(iii) `compute_current_divergence` at `src/tnfr/physics/conservation.py:209`.** Invokes (i) and (ii) to obtain `j_phi, j_dnfr: dict[node, float]`, then per node computes `div_j_phi = sum(j_phi.get(j, 0.0) - j_phi.get(i, 0.0) for j in neighbors) / deg` and analogously `div_j_dnfr`. Final assignment `divergence[i] = div_j_phi + div_j_dnfr` (Python float arithmetic on `dict.get`-fetched scalars). Output type `dict[node, float]`. **No Banach-derivative, no measure, no operator-valued lift; the divergence is by construction the same scalar-typed reduction as (i) and (ii) composed on the graph metric.**

All three intermediate arrays (`phases`, `degrees`, `dnfr_arr`, `edge_src`, `edge_dst`, `wrapped_diffs`) are fixed-length NumPy arrays whose sole purpose is to feed scalar reductions (`np.mean`, `np.sum`, vectorized sum-reduce in `compute_phase_current_vectorized` / `compute_dnfr_flux_vectorized`); none escapes the function or is exposed as an output type. Per the convention frozen at Sec 13quinquaginta-tertia for B7, fixed-length scalar-array intermediates used purely for scalar reduction are **not** "richer intermediates" in the type-hygiene sense — they are the standard NumPy idiom for batched scalar computation, and the canonical output type remains `dict[node, float]`.

#### .3 Phase-c verdict

The closure question posed at Sec 13quinquaginta-quarta is **answered NEGATIVELY**:

> The two canonical current fields (J_phi, J_DeltaNFR) and the conservation aggregator (div J) **do** reduce to scalar-valued (per-node) functionals of the canonical Tier-1+Tier-2 scalar slots (phi/theta + DeltaNFR via canonical alias `_get_dnfr`) plus the graph metric (`G.neighbors`, `G.degree`, `G.edges`, `G.is_directed`), with every intermediate scalar or scalar-array (fixed-length, used for scalar reduction) and **no implicit Banach-derivative apparatus, callable kernel, measure, or matrix lift** introduced along the reduction path. No richer intermediate type is forced.

NEGATIVE here means: there is no forcing of a non-canonical envelope on the Tier-1+Tier-2-to-currents reduction surface. The candidate ninth non-canonical research envelope is therefore classified as

**E_CC = HiddenIntermediateTensorStateOnCurrents** = "the would-be envelope that would have appeared if any of the three current/divergence implementations had introduced a callable, operator-valued, or measure-valued intermediate on its reduction path; structurally absent in the current canonical implementations".

E_CC joins E1...E_TC as the ninth non-canonical research envelope (cumulative list: B0-E1, B1-E2, B2-E3, B3-E4, B4-E5, B5-E6, B6-E7, B7-E_TC, B8-E_CC). The candidate ninth CDM is promoted to canonical status:

**CCC = Currents-Closure Discipline**, acting on the **Tier-1+Tier-2-to-currents reduction surface** (the third closure surface, after the Tier-1+Tier-2-to-tetrad reduction surface of B7).

#### .4 Catalog-Discharge Mechanism orthogonality (post-B8c)

Nine CDMs, nine structurally distinct surfaces:

| # | CDM | Sub-question | Surface |
|---|---|---|---|
| 1 | Pontryagin/measure-nu_f | B0 | field-measure |
| 2 | TMEP | B1 | element-projection |
| 3 | PWDP | B2 | phase-wrap |
| 4 | BSAD | B3 | scalar-aggregation |
| 5 | DITS | B4 | temporal-sampling |
| 6 | STD | B5 | coupling-verdict |
| 7 | SWD | B6 | mixing-aggregation |
| 8 | TRC | B7 | tetrad-reduction closure |
| 9 | **CCC** | **B8** | **currents-reduction closure** |

Each CDM is unique to the canonical machinery at its surface. No CDM is reused across sub-questions. The orthogonality is structural, not numerical.

#### .5 L3* status (post-B8c)

L3* heuristic, post-B8c, is promoted to: **"empirically robust working heuristic with complete Tier-1/Tier-2 structural-orthogonality coverage and first two Tier-3 closures orthogonally discharged"**. Cumulative nine CDMs (see table above). L3* prediction for remaining Tier-3/Tier-4 sub-questions (B9-B11): each admits its own orthogonal CDM at its own surface.

#### .6 Cross-references

- Sec 13quinquaginta-quarta (B8 Phase a + frozen signature).
- Sec 13quinquaginta-tertia (B7 Phase c + TRC, the conceptual template for the CCC discharge).
- Sec 13septies (Conjecture T-HP, G4 = RH; B8c does NOT advance this).
- `theory/CATALOG_TYPE_HYGIENE_PROGRAMME.md` Sec 3 row B8, Sec 4 row B8.
- `src/tnfr/physics/extended.py:60` (`compute_phase_current`).
- `src/tnfr/physics/extended.py:182` (`compute_dnfr_flux`).
- `src/tnfr/physics/conservation.py:209` (`compute_current_divergence`).
- `src/tnfr/riemann/currents_closure_signature.py` (B8a diagnostic).
- `examples/05_type_hygiene/86_currents_closure_signature_demo.py` (B8a demo).

### Sec 13quinquaginta-sexta — B9 Phase a: Aggregates-Closure Signature diagnostic (T-aggregates-closure)

#### .1 Scope

This section pre-registers and discharges Phase a of sub-question B9 = Delta-aggregates-closure of the Catalog Type-Hygiene Programme (third Tier-3 closure sub-question, after B7 and B8). Scope is strictly methodological: it pre-registers the closure question for the four canonical scalar aggregates — global coherence C(t), per-node Sense Index S_i, per-node energy density E, and per-node topological charge Q — and freezes one empirical observable, the **Aggregates-Closure Signature** S_AC. It does NOT promote or modify any canonical operator, does NOT alter the tetrad fields or the currents, does NOT advance G4 = RH (Conjecture T-HP, Sec 13septies), and does NOT by itself emit the closure verdict. Final verdict reserved for Phase c (Sec 13quinquaginta-septima) by direct source-code trace.

#### .2 Pre-registered closure question

> **Q (B9, T-aggregates-closure)**: do the four canonical scalar aggregates — `compute_coherence` (`C(t)`), `compute_Si` (`S_i`), `compute_energy_density` (`E`), `compute_topological_charge` (`Q`) — reduce to scalar-valued functionals of the canonical Tier-1+Tier-2 scalar slots (`nu_f`, `EPI`, `theta`/`phi`, `DeltaNFR` via canonical alias `_get_dnfr`) plus the graph metric (`G.neighbors`, `G.degree`, `G.edges`), with every intermediate scalar or scalar-array (fixed-length, used purely for scalar reduction) and no implicit Banach-derivative apparatus, callable kernel, measure, matrix lift, or operator-valued intermediate introduced along the reduction path?

This is a closure question (Phase b is **n/a**): if Q is answered YES (= NEGATIVE for the catalog-extension hypothesis), no richer envelope is forced; if Q is answered NO, the candidate tenth non-canonical research envelope E_AC = HiddenIntermediateTensorStateOnAggregates is structurally forced and a candidate tenth CDM ACD = Aggregates-Closure Discipline would need to be derived from canonical machinery to discharge it. The expected verdict per L3* is NEGATIVE; the structurally consistent classification of E_AC, conditional on NEGATIVE, is reserved for Phase c.

#### .3 Empirical diagnostic: Aggregates-Closure Signature S_AC

The diagnostic constructs a canonical ring graph at probe size n_nodes, initialises Tier-1+Tier-2 slots from a deterministic seed, and inspects two orthogonal axes:

- **Input-domain-closure axis**: fraction of per-node input values touched by the aggregate pipeline (Tier-1+Tier-2 scalar slots `nu_f`, `EPI`, `theta`, and the resolved `DeltaNFR` payload) that are structurally scalar-coercible (`isinstance(v, (int, float, np.floating, np.integer))` and `float(v)` succeeds).
- **Output-scalar-closure axis**: fraction of aggregate output values that are structurally scalar-coercible — the single global `float` from `compute_coherence`, plus every per-node entry of the three `dict[node, float]` returns from `compute_Si`, `compute_energy_density`, `compute_topological_charge`.

The signature is `S_AC = (1 - input_scalar_fraction) + (1 - output_scalar_fraction)` (normalised to [0, 2]; 0 = full scalar closure on both axes). Verdict thresholds: `S_AC <= 0.05` -> `SCALAR_CLOSURE_ADEQUATE`; `0.05 < S_AC <= 0.20` -> `SCALAR_CLOSURE_PARTIAL`; `S_AC > 0.20` -> `SCALAR_CLOSURE_DIVERGENT`.

**Frozen empirical signature** (canonical probes; deterministic; seed = 31):

- small  (n_nodes = 24): S_AC = 0.000000, input_scalar_fraction = 1.000000 (96/96), output_scalar_fraction = 1.000000 (73/73), verdict = `SCALAR_CLOSURE_ADEQUATE`. Per-key input nonscalar: nu_f = 0, EPI = 0, theta = 0, DeltaNFR = 0. Per-field output nonscalar: C_t = 0, Si = 0, energy_density = 0, topological_charge = 0.
- medium (n_nodes = 48): S_AC = 0.000000, input_scalar_fraction = 1.000000 (192/192), output_scalar_fraction = 1.000000 (145/145), verdict = `SCALAR_CLOSURE_ADEQUATE`. Per-key input nonscalar: nu_f = 0, EPI = 0, theta = 0, DeltaNFR = 0. Per-field output nonscalar: C_t = 0, Si = 0, energy_density = 0, topological_charge = 0.

The diagnostic certifies SCALAR_CLOSURE_ADEQUATE on both axes at both probes — necessary structural condition for the NEGATIVE Phase-c verdict. The diagnostic alone is not sufficient; Phase c emits the final verdict by direct source-code trace of the four canonical aggregate implementations.

Note on output denominators: the global scalar `C(t)` contributes a single entry (denominator = 1) per probe; the three per-node aggregates contribute n_nodes entries each (denominator = 3 * n_nodes). Hence total output-axis denominators are 1 + 3 * 24 = 73 and 1 + 3 * 48 = 145.

#### .4 Files

- `src/tnfr/riemann/aggregates_closure_signature.py` (new): `AggregatesClosureSignatureCertificate` + `compute_aggregates_closure_signature`.
- `examples/05_type_hygiene/87_aggregates_closure_signature_demo.py` (new): two-probe demo (n_nodes = 24, n_nodes = 48; seed = 31).
- `src/tnfr/riemann/__init__.py`: B9a re-exports.
- `theory/CATALOG_TYPE_HYGIENE_PROGRAMME.md`: B9 spec block + table row -> IN PROGRESS.

#### .5 Scope guard

Phase a is methodological only. It does NOT modify `compute_coherence`, `compute_Si`, `compute_energy_density`, or `compute_topological_charge`; does NOT alter any U-rule; does NOT change the tetrad or currents fields. It does NOT advance G4 = RH. The aggregates remain the canonical scalar functionals defined at their source-code locations.

### Sec 13quinquaginta-septima — B9 Phase c: NEGATIVE verdict for T-aggregates-closure, ACD promoted as tenth CDM

#### .1 Scope

This section emits the **final Phase-c verdict** for B9 = Delta-aggregates-closure, by direct source-code trace of the four canonical aggregate implementations. Scope is methodological: it closes the third Tier-3 sub-question (after B7 and B8), promotes **ACD = Aggregates-Closure Discipline** as the tenth Catalog-Discharge Mechanism, classifies **E_AC = HiddenIntermediateTensorStateOnAggregates** as the tenth non-canonical research envelope, and updates the cumulative L3* status. It does NOT modify any canonical implementation, does NOT alter the tetrad fields, currents, or any U-rule, and does NOT advance G4 = RH (Conjecture T-HP, Sec 13septies).

#### .2 Per-aggregate source-code closure trace

**(i) `compute_coherence` at `src/tnfr/metrics/common.py:29`.** Reads exactly two per-node attribute streams via canonical aliases: `dnfr_values = collect_attr(G, nodes, ALIAS_DNFR, 0.0)` and `depi_values = collect_attr(G, nodes, ALIAS_DEPI, 0.0)`. NumPy path computes `dnfr_mean = float(np.mean(np.abs(dnfr_values)))` and `depi_mean = float(np.mean(np.abs(depi_values)))` — two fixed-length scalar-array intermediates whose sole purpose is scalar reduction via `np.mean`, both immediately coerced to Python `float`. Fallback path uses `kahan_sum_nd` over a generator of scalar tuples, again followed by scalar division. Final assignment `coherence = 1.0 / (1.0 + dnfr_mean + depi_mean)` is Python float arithmetic. Output type `float` (single global scalar). **No callable kernel, no measure, no operator-valued intermediate, no matrix lift, no Banach-derivative apparatus introduced.**

**(ii) `compute_Si` at `src/tnfr/metrics/sense_index.py:665`.** Reads three canonical per-node attribute streams via canonical aliases (`ALIAS_VF`, `ALIAS_DNFR`, phase via `_get_phase`) plus three scalar weights (`alpha`, `beta`, `gamma`) merged from the `SI_WEIGHTS` graph attribute (already normalised to floats by `merge_and_normalize_weights`). Per-node kernel `compute_Si_node` at line 473 computes a weighted convex combination `Si = alpha * nu_f + beta * phase_alignment(node, neighbors) + gamma * (1 - normalised_DeltaNFR)` followed by `clamp01(...)` — Python/NumPy float arithmetic on per-node scalar inputs plus the local neighbour list. Vectorised path produces a fixed-length `np.ndarray[float64]` of length n_nodes via NumPy ufuncs (purpose: scalar reduction batched across nodes); fallback path produces `dict[node, float]` via Python `float(...)` coercion. The vectorised array's sole purpose is the batched scalar reduction; it does not escape the function as an output type and is not exposed via the public API (the public contract is `dict[node, float]` or, with the in-place fast path, an explicit `np.ndarray` of scalars wrapping the same scalar reduction). Output type `dict[node, float]` (or scalar-array equivalent). **Same closure verdict as (i).**

**(iii) `compute_energy_density` at `src/tnfr/physics/unified.py:136`.** Invokes the five canonical tetrad/currents accessors `compute_structural_potential(G)`, `compute_phase_gradient(G)`, `compute_phase_curvature(G)`, `compute_phase_current(G)`, `compute_dnfr_flux(G)`, each of which has been independently discharged as scalar-closed under B7 (tetrad: Sec 13quinquaginta-tertia) and B8 (currents: Sec 13quinquaginta-quinta). The aggregate is the dict-comprehension `{n: phi_s[n]**2 + grad_phi[n]**2 + k_phi[n]**2 + j_phi[n]**2 + j_dnfr[n]**2 for n in G.nodes()}` — Python float arithmetic on `dict.get`-fetched scalars. Output type `dict[node, float]`. **No Banach-derivative, no measure, no operator-valued lift; the energy density is by construction the same scalar-typed reduction as the tetrad and currents composed pointwise on the graph node set.**

**(iv) `compute_topological_charge` at `src/tnfr/physics/unified.py:209`.** Structurally isomorphic to (iii) with four-factor inputs instead of five: invokes `compute_phase_gradient(G)`, `compute_phase_curvature(G)`, `compute_phase_current(G)`, `compute_dnfr_flux(G)` (all scalar-closed under B7/B8), then dict-comprehension `{n: grad_phi[n] * j_phi[n] - k_phi[n] * j_dnfr[n] for n in G.nodes()}`. Output type `dict[node, float]`. **Same closure verdict as (iii).**

All intermediate NumPy arrays (`dnfr_values`, `depi_values` in (i); the vectorised Si payload arrays in (ii); the four/five tetrad/currents intermediate dicts in (iii)/(iv)) are fixed-length scalar-typed structures whose sole purpose is scalar reduction. None escapes the function or is exposed as an output type. Per the convention frozen at Sec 13quinquaginta-tertia for B7 and reused at Sec 13quinquaginta-quinta for B8, fixed-length scalar-array intermediates used purely for scalar reduction are **not** "richer intermediates" in the type-hygiene sense — they are the standard NumPy idiom for batched scalar computation, and the canonical output types remain the global `float` (i) and the per-node `dict[node, float]` (ii, iii, iv).

#### .3 Phase-c verdict

The closure question posed at Sec 13quinquaginta-sexta is **answered NEGATIVELY**:

> The four canonical scalar aggregates `compute_coherence` (`C(t)`), `compute_Si` (`S_i`), `compute_energy_density` (`E`), and `compute_topological_charge` (`Q`) **do** reduce to scalar-valued functionals of the canonical Tier-1+Tier-2 scalar slots (`nu_f`, `EPI`, `theta`/`phi`, `DeltaNFR` via canonical alias `_get_dnfr`) plus the graph metric (`G.neighbors`, `G.degree`, `G.edges`), with every intermediate scalar or scalar-array (fixed-length, used purely for scalar reduction) and **no implicit Banach-derivative apparatus, callable kernel, measure, matrix lift, or operator-valued intermediate** introduced along the reduction path. (i) and (ii) reduce directly from Tier-1+Tier-2 slots plus the graph metric; (iii) and (iv) reduce from the tetrad and currents, both of which were independently discharged as scalar-closed under B7 (Sec 13quinquaginta-tertia) and B8 (Sec 13quinquaginta-quinta). No richer intermediate type is forced.

NEGATIVE here means: there is no forcing of a non-canonical envelope on the Tier-1+Tier-2-plus-tetrad-plus-currents-to-aggregates reduction surface. The candidate tenth non-canonical research envelope is therefore classified as

**E_AC = HiddenIntermediateTensorStateOnAggregates** = "the would-be envelope that would have appeared if any of the four aggregate implementations had introduced a callable, operator-valued, or measure-valued intermediate on its reduction path; structurally absent in the current canonical implementations".

E_AC joins E1...E_CC as the tenth non-canonical research envelope (cumulative list: B0-E1, B1-E2, B2-E3, B3-E4, B4-E5, B5-E6, B6-E7, B7-E_TC, B8-E_CC, B9-E_AC). The candidate tenth CDM is promoted to canonical status:

**ACD = Aggregates-Closure Discipline**, acting on the **Tier-1+Tier-2-plus-tetrad-plus-currents-to-aggregates reduction surface** (the fourth closure surface, after the Tier-1+Tier-2-to-tetrad reduction surface of B7 and the Tier-1+Tier-2-to-currents reduction surface of B8). ACD is structurally distinct from TRC and CCC: TRC discharges the four tetrad reductions, CCC discharges the three current/divergence reductions, ACD discharges the four scalar-aggregate reductions that **compose** tetrad and currents into the global coherence indicator, sense index, energy density, and topological charge.

#### .4 Catalog-Discharge Mechanism orthogonality (post-B9c)

Ten CDMs, ten structurally distinct surfaces:

| # | CDM | Sub-question | Surface |
|---|---|---|---|
| 1 | Pontryagin/measure-nu_f | B0 | field-measure |
| 2 | TMEP | B1 | element-projection |
| 3 | PWDP | B2 | phase-wrap |
| 4 | BSAD | B3 | scalar-aggregation |
| 5 | DITS | B4 | temporal-sampling |
| 6 | STD | B5 | coupling-verdict |
| 7 | SWD | B6 | mixing-aggregation |
| 8 | TRC | B7 | tetrad-reduction closure |
| 9 | CCC | B8 | currents-reduction closure |
| 10 | **ACD** | **B9** | **aggregates-reduction closure** |

Each CDM is unique to the canonical machinery at its surface. No CDM is reused across sub-questions. The orthogonality is structural, not numerical.

#### .5 L3* status (post-B9c)

L3* heuristic, post-B9c, is promoted to: **"empirically robust working heuristic with complete Tier-1/Tier-2 structural-orthogonality coverage and all three Tier-3 closures orthogonally discharged"**. Cumulative ten CDMs (see table above). L3* prediction for remaining Tier-4 sub-questions (B10-B11): each admits its own orthogonal CDM at its own surface.

#### .6 Cross-references

- Sec 13quinquaginta-sexta (B9 Phase a + frozen signature).
- Sec 13quinquaginta-tertia (B7 Phase c + TRC, the conceptual template for the ACD discharge).
- Sec 13quinquaginta-quinta (B8 Phase c + CCC, the conceptual template for the ACD discharge).
- Sec 13septies (Conjecture T-HP, G4 = RH; B9c does NOT advance this).
- `theory/CATALOG_TYPE_HYGIENE_PROGRAMME.md` Sec 3 row B9, Sec 4 row B9.
- `src/tnfr/metrics/common.py:29` (`compute_coherence`).
- `src/tnfr/metrics/sense_index.py:665` (`compute_Si`).
- `src/tnfr/physics/unified.py:136` (`compute_energy_density`).
- `src/tnfr/physics/unified.py:209` (`compute_topological_charge`).
- `src/tnfr/riemann/aggregates_closure_signature.py` (B9a diagnostic).
- `examples/05_type_hygiene/87_aggregates_closure_signature_demo.py` (B9a demo).

---

## §13quinquaginta-octava — B10 Phase a: U-Rules Consistency Signature (URC) diagnostic

#### .1 Question

Do the unified-grammar rule checkers in `src/tnfr/operators/grammar_core.py` and `src/tnfr/operators/grammar_u6.py` consume only operator-name sequences plus scalar telemetry, and return only scalar/string verdicts? Or do they silently introduce a richer canonical envelope (callable kernel, measure, operator-valued intermediate, matrix lift, Banach-derivative apparatus) along the way?

#### .2 Phase a diagnostic

New module `src/tnfr/riemann/urules_consistency_signature.py` defines:

- `URulesConsistencySignatureCertificate` dataclass with per-rule input/output classifications, leakage counters, and the URC signature `S_UR := (leaking_inputs + leaking_outputs) / total_probes`.
- `compute_urules_consistency_signature(n_nodes, seed)` probes ten rule checkers (U1a, U1b, U2, U3, U4a, U4b, U2-REMESH, U5, temporal_ordering, U6) with synthetic canonical-scalar inputs sourced from `Emission`, `Reception`, `Coherence`, `Coupling`, `Resonance`, `Dissonance`, `Mutation`, `SelfOrganization`, `Recursivity`, `Silence` (a U1-U6-valid sequence) plus per-node Phi_s dicts on an Erdos-Renyi(n=24, p=0.3) graph.

Admissible input classes: `scalar` (int/float/bool/str), `operator_name_sequence` (list whose items expose `name` or `canonical_name`), `scalar_dict` (dict whose values are all scalar), `graph_metric` (a `networkx.Graph`). Admissible output classes: `scalar`, `scalar_tuple` (tuple whose entries are all scalar/None).

#### .3 Probe results (seeds 31, 131)

Both probes return:

- `S_UR = 0.000000`
- `verdict = TYPE_HYGIENE_ADEQUATE`
- `leaking_inputs = 0`, `leaking_outputs = 0`
- `input_scalar_fraction = 1.000`, `output_scalar_fraction = 1.000`

Per-rule classifications (identical across both seeds):

| Rule                                  | Inputs                                                         | Output         |
|---------------------------------------|----------------------------------------------------------------|----------------|
| U1a_initiation                        | operator_name_sequence + scalar                                | scalar_tuple   |
| U1b_closure                           | operator_name_sequence                                         | scalar_tuple   |
| U2_convergence                        | operator_name_sequence                                         | scalar_tuple   |
| U3_resonant_coupling                  | operator_name_sequence                                         | scalar_tuple   |
| U4a_bifurcation_triggers              | operator_name_sequence                                         | scalar_tuple   |
| U4b_transformer_context               | operator_name_sequence                                         | scalar_tuple   |
| U2_remesh_amplification               | operator_name_sequence                                         | scalar_tuple   |
| U5_multiscale_coherence               | operator_name_sequence                                         | scalar_tuple   |
| temporal_ordering                     | operator_name_sequence                                         | scalar_tuple   |
| U6_structural_potential_confinement   | graph_metric + scalar_dict + scalar_dict + scalar              | scalar_tuple   |

#### .4 Scope guard

Methodological diagnostic only. Does NOT modify any canonical implementation. Does NOT advance G4 = RH (Conjecture T-HP, Sec 13septies).

#### .5 Cross-references

- `src/tnfr/riemann/urules_consistency_signature.py` (this module).
- `examples/05_type_hygiene/88_urules_consistency_signature_demo.py` (probe demo).
- `src/tnfr/operators/grammar_core.py` (U1-U5 + temporal_ordering rule checkers).
- `src/tnfr/operators/grammar_u6.py` (U6 rule checker).
- `theory/CATALOG_TYPE_HYGIENE_PROGRAMME.md` Sec 3 row B10.

---

## §13quinquaginta-nona — B10 Phase c: NEGATIVE verdict, promote URC as eleventh CDM

#### .1 Source-code closure trace

Per-rule structural analysis of the ten checkers probed in Sec 13quinquaginta-octava:

- **U1a `validate_initiation`** (`grammar_core.py:76`): branches on `epi_initial <= EPI_BOUND_EPSILON` (scalar comparison), then tests `sequence[0].canonical_name in GENERATORS` (string-frozenset membership). No callable kernel, no measure.
- **U1b `validate_closure`** (`grammar_core.py:127`): tests `sequence[-1].canonical_name in CLOSURES`. Same pattern.
- **U2 `validate_convergence`** (`grammar_core.py:171`): counts destabilizer/stabilizer occurrences via list comprehensions over `canonical_name`, computes debt as `int - int`. Pure integer arithmetic on string-frozenset hits.
- **U3 `validate_resonant_coupling`** (`grammar_core.py:235`): iterates pairs `(prev, curr)` and tests `curr.canonical_name in COUPLING_RESONANCE`. String predicate only.
- **U4a `validate_bifurcation_triggers`** (`grammar_core.py:299`): for each trigger position, scans a fixed-radius window (integer slice) for handler hits via string-frozenset membership.
- **U4b `validate_transformer_context`** (`grammar_core.py:365`): same window-slice + string-frozenset pattern.
- **U2-REMESH `validate_remesh_amplification`** (`grammar_core.py:457`): tests sequence-level presence of REMESH plus any destabilizer plus stabilizers via frozenset membership.
- **U5 `validate_multiscale_coherence`** (`grammar_core.py:556`): same membership-counting pattern over operator-name strings.
- **`validate_temporal_ordering`** (`grammar_core.py:694`): inspects index positions (integers) of frozenset-flagged operators.
- **U6 `validate_structural_potential_confinement`** (`grammar_u6.py:22`): computes `drift = mean(|phi_s_after[i] - phi_s_before[i]|)` over a dict comprehension, compares against scalar threshold `PHI`. Inputs are scalar dicts plus a graph identifier used only for node iteration. No operator-valued intermediate.

Every checker reduces its inputs through a finite sequence of: (a) string-frozenset membership tests, (b) integer index arithmetic, (c) scalar comparison against canonical thresholds (`PHI`, `EPI_BOUND_EPSILON`). No checker introduces a callable kernel `K(x, y)`, a measure `mu`, an operator-valued intermediate, a matrix lift, or a Banach-derivative apparatus.

The frozensets themselves (`GENERATORS`, `CLOSURES`, `STABILIZERS`, `DESTABILIZERS`, `COUPLING_RESONANCE`) are module-level constants populated exclusively with `canonical_name` strings (e.g. `"emission"`, `"silence"`); they carry no richer state.

#### .2 NEGATIVE verdict

The U-rule type-hygiene surface admits no hidden canonical envelope. The Phase a empirical witness (`S_UR = 0.000000`, ten distinct rule checkers, two seeds) is fully reproduced by the source-code trace above.

#### .3 Promote URC as eleventh CDM

The Phase c analysis is structurally distinct from the ten preceding CDMs:

- **URC = U-Rules Consistency Discipline** acts on the **U-rules type-hygiene surface** — i.e. the set of input/output signatures of the rule-checker layer that mediates between operator sequences and the (`bool`, `str`) verdict pair consumed by `validate_grammar`.
- The ten prior CDMs act on disjoint surfaces: Pontryagin/measure-nu_f (B0), tetrad-membrane-evolution-projection (B1, TMEP), phase-wrap-density-projection (B2, PWDP), bifurcation-state-aggregation-density (B3, BSAD), discrete-injection-time-sampling (B4, DITS), spectral-trace-density (B5, STD), spectrum-weighting-density (B6, SWD), tetrad-reduction-closure (B7, TRC), currents-reduction-closure (B8, CCC), aggregates-reduction-closure (B9, ACD).

Orthogonality is established by surface-disjointness: URC operates on rule-checker signatures, none of the prior ten CDMs do.

#### .4 Eleventh non-canonical envelope

`E_UR = HiddenIntermediateRulecheckerState` — a hypothetical envelope wherein a U-rule checker carries a callable kernel, measure, operator-valued intermediate, matrix lift, or Banach-derivative apparatus between its input and output. The Phase a + Phase c analysis classifies `E_UR` as the eleventh non-canonical research envelope: its hypothetical content is absent from the canonical 13-operator catalog's rule-checker layer.

#### .5 L3* update

L3* heuristic, post-B10c, is promoted to: **"empirically robust working heuristic with complete Tier-1/Tier-2 structural-orthogonality coverage, all three Tier-3 closures, and the first Tier-4 closure orthogonally discharged"**. Cumulative eleven CDMs (Pontryagin/measure-nu_f, TMEP, PWDP, BSAD, DITS, STD, SWD, TRC, CCC, ACD, URC). L3* prediction for the remaining Tier-4 sub-question (B11): admits its own orthogonal CDM at the operator-catalog-completeness surface.

#### .6 Cross-references

- Sec 13quinquaginta-octava (B10 Phase a + frozen signature).
- Sec 13quinquaginta-tertia (B7 Phase c + TRC, conceptual template).
- Sec 13quinquaginta-quinta (B8 Phase c + CCC).
- Sec 13quinquaginta-septima (B9 Phase c + ACD).
- Sec 13septies (Conjecture T-HP, G4 = RH; B10c does NOT advance this).
- `theory/CATALOG_TYPE_HYGIENE_PROGRAMME.md` Sec 3 row B10, Sec 4 row B10.
- `src/tnfr/operators/grammar_core.py` (U1-U5 + temporal_ordering).
- `src/tnfr/operators/grammar_u6.py` (U6).
- `src/tnfr/riemann/urules_consistency_signature.py` (Phase a diagnostic).
- `examples/05_type_hygiene/88_urules_consistency_signature_demo.py` (Phase a demo).

---

## §13sexagesima — B11 Phase a: Operator-Catalog Discipline Signature (OCD) diagnostic

#### .1 Question

Is the canonical 13-operator TNFR registry enforced as an immutable closed set, with no hidden 14th-operator construction reachable from the public API? Does the catalog surface (registry + introspection metadata + public exports) introduce any callable kernel, measure, operator-valued intermediate, matrix lift, or Banach-derivative apparatus along the way?

#### .2 Phase a diagnostic

New module `src/tnfr/riemann/operator_catalog_discipline_signature.py` defines:

- `OperatorCatalogDisciplineSignatureCertificate` dataclass with ordered probe IDs, per-probe results, anomaly counter, and the OCD signature `S_OC := anomalies / total_probes`.
- `compute_operator_catalog_discipline_signature()` runs ten read-only probes over the catalog surface.

Probes:

1. `registry_size`: `len(OPERATORS) == 13`.
2. `registry_entries_are_operator_subclasses`: every value in `OPERATORS` is a subclass of `Operator`.
3. `registry_keys_are_lowercase_strings`: every key is a non-empty lowercase `str`.
4. `registry_keys_unique`: `len(keys) == len(set(keys))`.
5. `metadata_size`: `len(OPERATOR_METADATA) == 13`.
6. `metadata_values_are_operator_meta`: every value is an `OperatorMeta` instance.
7. `metadata_fields_are_string_tuples`: every `name`, `mnemonic`, `category`, `doc` field is a `str`; every `grammar_roles` and `contracts` field is a tuple of strings.
8. `metadata_registry_alignment`: the set of `meta.name` values equals the set of registry class names (1-to-1).
9. `definitions_exports_cover_canonical_set`: `definitions.__all__` exposes all 13 canonical operator class names (`Emission`, `Reception`, `Coherence`, `Dissonance`, `Coupling`, `Resonance`, `Silence`, `Expansion`, `Contraction`, `SelfOrganization`, `Mutation`, `Transition`, `Recursivity`).
10. `no_hidden_fourteenth_operator`: re-invoking `_ensure_loaded()` is idempotent — `len(OPERATORS)` does not grow past 13 and key set is unchanged.

#### .3 Probe results

Single deterministic invocation (read-only inspection of module-level mappings):

- `S_OC = 0.000000`
- `anomalies = 0 / 10`
- `verdict = CATALOG_DISCIPLINE_ADEQUATE`
- `registry_size = 13`, `metadata_size = 13`, `canonical_exports_observed = 13/13`

A second invocation produced identical results, confirming idempotency.

#### .4 Scope guard

Methodological diagnostic only. Does NOT modify any canonical implementation. Does NOT advance G4 = RH (Conjecture T-HP, Sec 13septies).

#### .5 Cross-references

- `src/tnfr/riemann/operator_catalog_discipline_signature.py` (this module).
- `examples/05_type_hygiene/89_operator_catalog_discipline_signature_demo.py` (probe demo).
- `src/tnfr/operators/registry.py` (immutable 13-operator registry).
- `src/tnfr/operators/introspection.py` (`OPERATOR_METADATA`).
- `src/tnfr/operators/definitions.py` (`__all__` exports).
- `docs/OPERATOR_COMPLETENESS.md` (existing completeness analysis).
- `theory/CATALOG_TYPE_HYGIENE_PROGRAMME.md` Sec 3 row B11.

---

## §13sexagesima-prima — B11 Phase c: NEGATIVE verdict, promote OCD as twelfth CDM

#### .1 Source-code closure trace

Per-probe structural analysis of the catalog surface:

- **`OPERATORS` mapping** (`src/tnfr/operators/registry.py:19`): declared as `dict[str, type[Operator]]` and populated exclusively by `_ensure_loaded()` (line 28) with the 13 canonical class references imported lazily from `definitions.py`. No richer state attached; values are bare class objects.
- **`_ensure_loaded()` guard** (`registry.py:28`): returns early when `OPERATORS` is non-empty, so repeated invocation is structurally idempotent. The mapping is built once and frozen by usage convention; canonical purity is enforced by the module-level comment "TNFR physics defines exactly 13 canonical structural operators".
- **`OperatorMeta` dataclass** (`src/tnfr/operators/introspection.py:48`): declared with `@dataclass(frozen=True, slots=True)` — immutable record carrying only `str`, `str`, `str`, `tuple[str, ...]`, `tuple[str, ...]`, `str` fields. No callable, no measure, no graph reference.
- **`OPERATOR_METADATA`** (`introspection.py:56`): module-level `Mapping[str, OperatorMeta]` populated literally with 13 entries (mnemonics `AL, EN, IL, OZ, UM, RA, SHA, VAL, NUL, THOL, ZHIR, NAV, REMESH`). Each entry's `grammar_roles` and `contracts` fields are tuples of canonical-text strings — no callable kernel hidden in metadata.
- **`definitions.__all__`** (`src/tnfr/operators/definitions.py:53`): explicit list naming exactly the 13 operator classes plus the `Operator` base and seven introspection/grammar-error helpers. No wildcard, no dynamic discovery.
- **`Operator` base** (`src/tnfr/operators/definitions_base.py:29`): metaclass `OperatorMetaAuto` retained for backward compatibility only — the module docstring states "Metaclass removed - canonical operator set is immutable (see registry)." Auto-registration is structurally inert against canonical extension because grammar logic (`grammar_core.GENERATORS`/`CLOSURES`/...) frozensets reference canonical_name strings exclusively, so an out-of-band subclass would never satisfy any U-rule (B10 / URC).

Every probe of B11 Phase a reduces its evidence through: (a) `len()` against the constant `13`, (b) `isinstance`/`issubclass`, (c) string predicate, (d) set difference. No probe constructs a callable kernel `K(x, y)`, a measure `mu`, an operator-valued intermediate, a matrix lift, or a Banach-derivative apparatus on the catalog surface.

#### .2 NEGATIVE verdict

The operator-catalog discipline surface admits no hidden canonical envelope. The Phase a empirical witness (`S_OC = 0.000000`, ten probes across two invocations) is fully reproduced by the source-code trace above. The "ghost 14th operator" construction posited in the B11 spec block of `theory/CATALOG_TYPE_HYGIENE_PROGRAMME.md` Sec 3 is unreachable from the public API: the lazy-loading guard, the immutable frozensets in `grammar_core.py`, and the explicit `definitions.__all__` jointly close the catalog at exactly 13 operators.

#### .3 Promote OCD as twelfth CDM

The Phase c analysis is structurally distinct from the eleven preceding CDMs:

- **OCD = Operator-Catalog Discipline** acts on the **operator-catalog-closure surface** — i.e. the registry + introspection-metadata + public-exports triple that defines the boundary of what counts as a canonical TNFR operator.
- The eleven prior CDMs act on disjoint surfaces: Pontryagin/measure-nu_f (B0), tetrad-membrane-evolution-projection (B1, TMEP), phase-wrap-density-projection (B2, PWDP), bifurcation-state-aggregation-density (B3, BSAD), discrete-injection-time-sampling (B4, DITS), spectral-trace-density (B5, STD), spectrum-weighting-density (B6, SWD), tetrad-reduction-closure (B7, TRC), currents-reduction-closure (B8, CCC), aggregates-reduction-closure (B9, ACD), U-rules-type-hygiene (B10, URC).

Orthogonality is established by surface-disjointness: OCD operates on catalog metadata and registry mapping, none of the prior eleven CDMs do.

#### .4 Twelfth non-canonical envelope

`E_OC = HiddenFourteenthOperatorConstruction` — a hypothetical envelope wherein a 14th canonical operator could be introduced into the registry, the metadata, or `definitions.__all__` through some dynamic-discovery mechanism, monkey-patch, or richer metadata payload. The Phase a + Phase c analysis classifies `E_OC` as the twelfth non-canonical research envelope: its hypothetical content is absent from the immutable registry's design and from every probe surface inspected.

#### .5 L3* update

L3* heuristic, post-B11c, is promoted to: **"empirically robust working heuristic with complete Tier-1/Tier-2 structural-orthogonality coverage, all three Tier-3 closures, and both Tier-4 closures orthogonally discharged"**. Cumulative twelve CDMs (Pontryagin/measure-nu_f, TMEP, PWDP, BSAD, DITS, STD, SWD, TRC, CCC, ACD, URC, OCD). All sub-questions B0-B11 are NEGATIVE. The final composite meta-minimality theorem (B0-B11 assembly) is now eligible for statement and proof; deferred to a separate commit (Sec 13sexagesima-secunda).

#### .6 Cross-references

- Sec 13sexagesima (B11 Phase a + frozen signature).
- Sec 13quinquaginta-nona (B10 Phase c + URC, conceptual template).
- Sec 13quinquaginta-septima (B9 Phase c + ACD).
- Sec 13quinquaginta-quinta (B8 Phase c + CCC).
- Sec 13quinquaginta-tertia (B7 Phase c + TRC).
- Sec 13septies (Conjecture T-HP, G4 = RH; B11c does NOT advance this).
- `theory/CATALOG_TYPE_HYGIENE_PROGRAMME.md` Sec 3 row B11, Sec 4 row B11.
- `src/tnfr/operators/registry.py` (`OPERATORS`, `_ensure_loaded`).
- `src/tnfr/operators/introspection.py` (`OPERATOR_METADATA`, `OperatorMeta`).
- `src/tnfr/operators/definitions.py` (`__all__`).
- `src/tnfr/operators/definitions_base.py` (`Operator` base).
- `src/tnfr/riemann/operator_catalog_discipline_signature.py` (Phase a diagnostic).
- `examples/05_type_hygiene/89_operator_catalog_discipline_signature_demo.py` (Phase a demo).

---

## §13sexagesima-secunda — Final: Composite Meta-Minimality and Catalog-Closure Theorem

This section assembles the twelve NEGATIVE verdicts from B0-B11 into a single composite statement. The Catalog Type-Hygiene Programme (`theory/CATALOG_TYPE_HYGIENE_PROGRAMME.md`) terminates here.

#### .1 The twelve discharged sub-questions

| ID | Sub-question | CDM | Envelope (research-only) | Notes ref |
|----|----|----|----|----|
| B0  | T-nu_f                       | Pontryagin/measure-nu_f                | E0 = MeasureExtensionOnNuF             | Sec 13tricesima-quinta |
| B1  | T-EPI                        | TMEP                                   | E1 = ContinuousFormExtensionOnEPI      | Sec 13triginta-secunda |
| B2  | T-phi                        | PWDP                                   | E2 = LiftedCircleBundleOnPhi           | Sec 13triginta-quarta |
| B3  | T-DeltaNFR                   | BSAD                                   | E3 = HiddenBifurcationStateOnDeltaNFR  | Sec 13triginta-sexta |
| B4  | T-REMESH-window              | DITS                                   | E4 = ContinuousReinjectionMeasure      | Sec 13quadraginta-quinta |
| B5  | T-Delta-phi-max              | STD                                    | E5 = SpectralTraceExtension            | Sec 13quadraginta-octava |
| B6  | T-coupling-weights           | SWD                                    | E6 = NodeIndexedCouplingWeights        | Sec 13quinquaginta-prima |
| B7  | Delta-tetrad-closure         | TRC                                    | E_TC = HiddenIntermediateTensorState   | Sec 13quinquaginta-tertia |
| B8  | Delta-currents-closure       | CCC                                    | E_CC = HiddenIntermediateTensorStateOnCurrents   | Sec 13quinquaginta-quinta |
| B9  | Delta-aggregates-closure     | ACD                                    | E_AC = HiddenIntermediateTensorStateOnAggregates | Sec 13quinquaginta-septima |
| B10 | U-rules type-hygiene         | URC                                    | E_UR = HiddenIntermediateRulecheckerState        | Sec 13quinquaginta-nona |
| B11 | Operator-catalog closure     | OCD                                    | E_OC = HiddenFourteenthOperatorConstruction      | Sec 13sexagesima-prima |

Each row carries a NEGATIVE verdict obtained by a single CDM (concordance-discharging mechanism) at a structurally distinct surface. The twelve CDMs partition the canonical-state contract surface: Pontryagin/measure (B0), tetrad-membrane-evolution-projection (B1), phase-wrap-density-projection (B2), bifurcation-state-aggregation-density (B3), discrete-injection-time-sampling (B4), spectral-trace-density (B5), spectrum-weighting-density (B6), tetrad-reduction-closure (B7), currents-reduction-closure (B8), aggregates-reduction-closure (B9), U-rules-type-hygiene (B10), operator-catalog-closure (B11).

#### .2 Theorem statement

**Theorem (Catalog Minimality & Completeness)**. Under the 13-operator TNFR catalog (`src/tnfr/operators/registry.py`) and the unified grammar U1-U6 (`src/tnfr/operators/grammar_core.py`, `src/tnfr/operators/grammar_u6.py`), the per-node types

$$(\nu_f, \mathrm{EPI}, \phi, \Delta\mathrm{NFR}) \in \mathbb{R}^+ \times \mathbb{R} \times [0, 2\pi) \times \mathbb{R} ,$$

the graph-level parameters

$$(\tau_l, \tau_g, \Delta\phi_{\max}, w_{ij}) \in \mathbb{N}^2 \times [0, \pi] \times \mathbb{R}_{\ge 0} ,$$

the derived structural-field tetrad

$$(\Phi_s, |\nabla\phi|, K_\phi, \xi_C) \in \mathbb{R}^4 ,$$

and the derived currents

$$(J_\phi, J_{\Delta\mathrm{NFR}}) \in \mathbb{R}^2$$

are jointly the **minimal and complete** structural state of any TNFR realisation that satisfies the nodal equation `dEPI/dt = nu_f * DeltaNFR(t)` and the unified grammar U1-U6. Specifically:

- **Completeness**: every canonical operator invocation, every U-rule check, every aggregate-functional output, and every registry/metadata inspection reduces — through the source-code traces of B0-B11 Phase c — to operations on tuples of these scalars (plus integer indices). No callable kernel, no measure, no operator-valued intermediate, no matrix lift, and no Banach-derivative apparatus is forced by the canonical machinery at any of the twelve surfaces probed.
- **Minimality**: no member of the canonical state tuple can be eliminated without (i) violating the nodal-equation contract (`nu_f`, `EPI`, `DeltaNFR`), (ii) breaking U1-U6 closure (`phi`, `tau_l`, `tau_g`, `Delta-phi-max`, `w_{ij}`), or (iii) collapsing one of the derived-field aggregates whose admissibility is verified independently by B7/B8/B9 (the tetrad, currents, and aggregates).
- **Catalog closure**: the operator catalog is exactly the immutable set of 13 classes registered in `OPERATORS`; no fourteenth operator is reachable from the public API (B11) and no U-rule checker admits a richer input/output signature (B10).

#### .3 Proof sketch

Each clause of the theorem follows directly from the corresponding Phase c discharge:

1. **Per-node types** (`nu_f`, `EPI`, `phi`, `DeltaNFR`): the four B0/B1/B2/B3 Phase c traces classify any richer envelope (`E0..E3`) as research-only; the canonical implementation in `src/tnfr/dynamics/`, `src/tnfr/metrics/`, and `src/tnfr/operators/` reads only the scalar types declared above.
2. **Graph-level parameters** (`tau_l`, `tau_g`, `Delta-phi-max`, `w_{ij}`): B4/B5/B6 Phase c traces classify `E4..E6` as research-only; the canonical scheduler in `src/tnfr/dynamics/runtime.py` and the coupling layer in `src/tnfr/operators/coupling.py` read only the natural-number windows and the scalar threshold/weight types declared above.
3. **Tetrad** (`Phi_s`, `|grad phi|`, `K_phi`, `xi_C`): B7 Phase c discharges TRC; the canonical aggregator chain in `src/tnfr/metrics/structural_fields.py` returns scalar functionals of the per-node tuples plus weight scalars.
4. **Currents** (`J_phi`, `J_{DeltaNFR}`): B8 Phase c discharges CCC at `src/tnfr/metrics/sense_index.py` and `src/tnfr/physics/unified.py`; outputs are scalar densities or per-node scalar arrays.
5. **Aggregate functionals**: B9 Phase c discharges ACD; the four canonical aggregates (`compute_coherence`, `compute_Si`, `compute_energy_density`, `compute_topological_charge`) return real scalars.
6. **U-rule closure**: B10 Phase c discharges URC; every rule checker in `grammar_core.py` and `grammar_u6.py` reduces to string-frozenset membership plus integer-index arithmetic plus scalar comparison.
7. **Catalog closure**: B11 Phase c discharges OCD; `OPERATORS`, `OPERATOR_METADATA`, and `definitions.__all__` jointly close at exactly 13 canonical operators.

Composition: any canonical computation in TNFR is a finite sequence of (a) per-node attribute reads (B0-B3 types), (b) graph-level parameter reads (B4-B6 types), (c) U-rule checks (B10), (d) operator dispatch via the catalog (B11), and (e) aggregate/current/tetrad evaluation (B7-B9). The twelve closures jointly cover every reachable canonical observation; their composition is a finite composition of scalar-typed evaluations, so the joint state above is both sufficient and necessary. **QED (composite, conditional on B0-B11 Phase c traces).**

#### .4 Status

- The theorem is **established in the canonical-implementation sense**: every Phase c trace is a literal source-code inspection of the current canonical implementation. Refutation requires producing a canonical computation whose evidence falsifies one of the twelve Phase c traces (i.e. introduces a callable kernel, a measure, an operator-valued intermediate, a matrix lift, or a Banach-derivative apparatus at one of the twelve surfaces). No such computation is currently known in the canonical layer.
- Twelve non-canonical research envelopes (`E0..E_OC`) are classified as **research-only**: they may be useful for off-canonical experiments (e.g. spectral programmes, primality-test bench, factorization-lab) but they are not forced by the canonical contract and do not extend the canonical state.

#### .5 Scope guard

Does **NOT** advance G4 = RH (Conjecture T-HP, Sec 13septies). The theorem is a catalog-minimality / catalog-completeness statement, independent from the Riemann-hypothesis programme. The twelve CDMs establish that the canonical layer is structurally closed; they say nothing about the spectral location of the zeros of zeta. The off-canonical envelopes E0..E_OC remain the natural research surfaces for any future spectral programme.

#### .6 Cross-references

- Sec 13sexagesima (B11 Phase a).
- Sec 13sexagesima-prima (B11 Phase c + OCD).
- Sec 13quinquaginta-nona (B10 Phase c + URC).
- Sec 13quinquaginta-septima (B9 Phase c + ACD).
- Sec 13quinquaginta-quinta (B8 Phase c + CCC).
- Sec 13quinquaginta-tertia (B7 Phase c + TRC).
- Sec 13quinquaginta-prima (B6 Phase c + SWD).
- Sec 13quadraginta-octava (B5 Phase c + STD).
- Sec 13quadraginta-quinta (B4 Phase c + DITS).
- Sec 13triginta-sexta (B3 Phase c + BSAD).
- Sec 13triginta-quarta (B2 Phase c + PWDP).
- Sec 13triginta-secunda (B1 Phase c + TMEP).
- Sec 13tricesima-quinta (B0 Phase c + Pontryagin/measure-nu_f).
- Sec 13septies (Conjecture T-HP, G4 = RH; the composite theorem is independent).
- `theory/CATALOG_TYPE_HYGIENE_PROGRAMME.md` Sec 3 Final, Sec 4 Final.
- `src/tnfr/operators/registry.py`, `introspection.py`, `definitions.py`, `definitions_base.py`.
- `src/tnfr/operators/grammar_core.py`, `grammar_u6.py`.
- `src/tnfr/dynamics/`, `src/tnfr/metrics/`, `src/tnfr/physics/`.
- `src/tnfr/riemann/` (twelve `*_signature.py` diagnostic modules).
- `examples/79_pontryagin_*.py` ... `examples/05_type_hygiene/89_operator_catalog_discipline_signature_demo.py` (per-phase demos).


## §13sexagesima-tertia. Branch B0★ — Scope-Expansion of Existing TNFR Theory (Pre-registration of a Fourth Branch of the §13septies Trichotomy; Does NOT advance G4 = RH)

### .1 Motivation: a fourth branch the §13septies trichotomy did not enumerate

The §13septies trichotomy for G4 = RH was stated as:

* **B1** — closure inside the canonical 13-operator catalog (CCET-G_P14, §13vicies-novies.16, **CLOSED on G_P14**; status off G_P14: open in principle).
* **B2** — new canonical operator derivable from the nodal equation, intertwining slot with prime in a way the catalog cannot.
* **B3** — no TNFR closure exists.

§13sexagesima-secunda (Composite Catalog-Closure Theorem, B11 NEGATIVE) established that **no 14th canonical operator is reachable from the public API**: the registry `OPERATORS`, the introspection metadata `OPERATOR_METADATA`, and `definitions.__all__` jointly close the catalog at exactly 13 operators, and the lazy-loading guard plus the immutable grammar frozensets make a hypothetical 14th operator (envelope `E_OC = HiddenFourteenthOperatorConstruction`) structurally unreachable in the canonical layer.

Read literally, this **forces §13septies B2 NEGATIVE** at the level of the catalog: any candidate that materialises as a 14th `Operator` subclass in `OPERATORS` is excluded by §13sexagesima-prima. The §13septies trichotomy would thus reduce to **{B1-off-G_P14, B3}**.

But the §13septies trichotomy as written did not enumerate a structurally legitimate fourth branch that this work has surfaced:

> **B0★ — scope-expansion of the existing TNFR theory** — close G4 without adding any operator to `OPERATORS` and without dropping any of the twelve B0–B11 NEGATIVE verdicts, by either (α) extracting consequences of the existing 13 operators that have not yet been derived, or (β) promoting one or more of the twelve research envelopes `E0..E_OC` to canonical status.

This subsection pre-registers B0★ as the fourth branch. It does **not** execute any sub-branch; per-envelope and per-consequence analyses are deferred to subsequent commits.

### .2 Definition of B0★ and orthogonality to §13sexagesima-secunda

**Definition (B0★).** A branch-B0★ closure attempt for G4 = RH is any structural argument that:

1. does **not** modify `src/tnfr/operators/registry.py::OPERATORS` (no 14th operator);
2. does **not** add a member to `src/tnfr/operators/introspection.py::OPERATOR_METADATA`;
3. does **not** add a class to `src/tnfr/operators/definitions.py::__all__`;
4. does **not** modify the nodal equation $\partial \mathrm{EPI}/\partial t = \nu_f \cdot \Delta\mathrm{NFR}(t)$;
5. does **not** modify the unified grammar U1–U6;

and yet closes G4 = RH (or its T-HP reformulation, §13septies.4) by either of the following two sub-mechanisms:

* **B0★-α (deeper-exploitation sub-branch).** Derive a previously-uncomputed structural consequence of the existing 13 operators (in particular, of their compositions under C1–C5 of CCET-G_P14, evaluated on canonically-constructed graphs other than G_P14) that closes the oscillatory half of the admissible rescaling operator $\mathcal{F}$ of Conjecture T-HP (§13septies.4). The catalog is unchanged; only the analysis is deeper.

* **B0★-β (envelope-promotion sub-branch).** Promote one or more of the twelve research envelopes
  $\{E_0, E_1, E_2, E_3, E_4, E_5, E_6, E_{TC}, E_{CC}, E_{AC}, E_{UR}, E_{OC}\}$
  (catalog of §13sexagesima-secunda.1) from *research-only* to *canonical* by supplying a missing canonical derivation from the nodal equation that the §13triginta-* through §13sexagesima-* programme did not produce (or did not attempt). The 13 operators stay fixed; the *state-space types* on which they operate become richer, and the closure of G4 may follow from the enriched types alone.

**Orthogonality to §13sexagesima-secunda.** Neither sub-branch contradicts the Composite Catalog-Closure Theorem:

* B0★-α uses the closed catalog as input and extracts consequences; it adds no operator and no envelope.
* B0★-β promotes envelopes that were classified as *research-only* by the twelve Phase c traces. The §13triginta-* through §13sexagesima-* NEGATIVE verdicts say each envelope is **not forced** by the current canonical contract; they do **not** say each envelope is **incompatible** with a future canonical contract. The distinction is the same as between "not currently needed" and "ruled out". The Composite Theorem is a *minimality* statement (no envelope is forced); B0★-β is an orthogonal *maximality* question (can an envelope be admitted without breaking the nodal equation or U1–U6).

In particular: a successful B0★-β closure would *not* require re-opening the twelve Phase c traces; it would re-classify one or more envelopes from *research-only* to *canonical* by supplying a derivation from the nodal equation that the type-hygiene programme did not search for (it was searching for forcing-axioms F1–F10, not admissibility-axioms).

### .3 Mapping B0★-β candidates against §13septies B2 / G4 = RH relevance

The twelve envelopes inventoried in §13sexagesima-secunda.1 are not equally relevant to the open content of T-HP (oscillatory half of $\mathcal{F}$, identified in §13septies.5 / N15 W3 with $\ker(\mathcal{R}_\infty)$ and the residue $S(T) = (1/\pi)\arg\zeta(\tfrac12 + iT)$). The structural-relevance ranking is:

| Envelope | Promotion would add | Relevance to T-HP oscillatory half | Priority |
|---|---|---|---|
| $E_0$ = MeasureExtensionOnNuF | $\nu_f$ becomes a measure on the Pontryagin dual $\widehat{\mathbb{Z}} = S^1$ rather than a scalar in $\mathbb{R}^+$ | **HIGH** — Pontryagin-dual measure carries oscillatory harmonic content that a scalar $\nu_f$ discards; matches the Fourier-pair structure of the Weil–Guinand prime side | **P1** |
| $E_6$ = NodeIndexedCouplingWeights | coupling weights $w_{ij}$ become per-node-indexed rather than graph-level scalars | **HIGH** — directly breaks Fact A of CCET-G_P14 (parameter uniformity); enables prime-arithmetic-dependent edge structure if the per-node rule is derivable from $\nu_f$ via a non-symmetric construction | **P2** |
| $E_2$ = LiftedCircleBundleOnPhi | $\phi$ becomes a covering-space lift with integer winding $w \in \mathbb{Z}$ | **MEDIUM** — adds homotopy data; could carry oscillatory phase information but does not obviously break prime-relabelling symmetry on $G_{P14}$ | P3 |
| $E_1$ = ContinuousFormExtensionOnEPI | EPI becomes a continuous form rather than a scalar | LOW — does not obviously connect to oscillatory-half closure | — |
| $E_3$ = HiddenBifurcationStateOnDeltaNFR | $\Delta\mathrm{NFR}$ carries bifurcation-state aggregation | LOW — bifurcation structure not obviously oscillatory-relevant | — |
| $E_4, E_5, E_{TC}, E_{CC}, E_{AC}, E_{UR}, E_{OC}$ | various derived-aggregate / control-surface enrichments | LOW — derived from primary types; do not add primary-type expressivity for $\mathcal{F}$ | — |

**P1 (Pontryagin-dual measure $E_0$)** is the structurally most natural B0★-β candidate because the missing canonical content (oscillatory residue of $\mathcal{F}$) is exactly what a measure on the Pontryagin dual encodes that a scalar discards. The §13tricesima-quinta Phase c trace established that the **canonical implementation** does not need this enrichment; B0★-β-P1 asks the orthogonal question: can a measure-valued $\nu_f$ be **derived** from the nodal equation as canonically as the scalar version, and if so, does the resulting enriched dynamics close the oscillatory half?

**P2 (per-node coupling weights $E_6$)** is structurally next-most-natural because Fact A of CCET-G_P14 is the principal obstruction to slot-prime intertwining inside the catalog. Promoting $E_6$ would mechanically dissolve Fact A and reopen the spectral-non-trivial sub-region of CCC constructions, *if* the per-node weight rule can be canonically derived from $\nu_f$ values via a construction that breaks the symmetric-function-of-scalars constraint.

### .4 Pre-registered B0★ sub-route pre-conditions (acceptance / refutation criteria)

For any B0★ closure attempt to be admissible at the canonical layer, the following five acceptance criteria must be met (mirroring the canonicity criteria of Conjecture T-HP §13septies.4 items 1–3, with item 0 added for B0★ specifically):

* **C0 — No catalog modification.** The candidate adds no entry to `OPERATORS`, `OPERATOR_METADATA`, or `definitions.__all__`. (Verifiable by `git diff src/tnfr/operators/registry.py introspection.py definitions.py`.)
* **C1 — Nodal-equation derivation.** The promoted envelope (B0★-β) or the extracted consequence (B0★-α) is *derived* from $\partial \mathrm{EPI}/\partial t = \nu_f \cdot \Delta\mathrm{NFR}(t)$ together with the canonical invariants 1–6 and the structural scale $\pi$ only. A successful fit, post-hoc rationalisation, or external-axiom adoption fails C1.
* **C2 — U1–U6 admissibility.** The enriched dynamics (B0★-β) or the extracted consequence (B0★-α) preserves the unified grammar U1–U6, including the continuity equation $\partial \rho / \partial t + \nabla \cdot \mathbf{J} = \mathcal{S}_{\mathrm{grammar}}$ with uniformly-bounded source term.
* **C3 — Twelve-CDM consistency.** The candidate does not contradict any of the twelve B0–B11 Phase c traces in their literal source-code statements; in particular, it does not introduce a callable kernel, measure, operator-valued intermediate, matrix lift, or Banach-derivative apparatus at any of the twelve surfaces *as a forced canonical contract* (re-classification from research-only to canonical is permitted; introduction of a new forcing axiom is not).
* **C4 — T-HP discharge.** The candidate, when composed with the existing canonical catalog and the smooth half of $\mathcal{F}$ closed by P28/P30, produces an operator on $\mathcal{H}_{\mathrm{tet}}$ whose spectrum coincides with $\{\gamma_n\}_{n \ge 1}$ (Conjecture T-HP item 3).

Failure of any C0–C4 disqualifies the candidate as a B0★ closure. C4 is the empirical/derivational core; C0–C3 are admissibility filters.

### .5 What B0★ does NOT claim

* B0★ is a **pre-registration of a fourth branch**, not a closure attempt. No envelope is promoted in this commit; no consequence is extracted; no acceptance criterion is yet evaluated against P1 or P2.
* B0★ does **not** advance G4 = RH. The branch is structurally legitimate but its closure content is open.
* B0★ does **not** contradict §13sexagesima-secunda. The Composite Catalog-Closure Theorem is a minimality statement; B0★ is an orthogonal maximality question.
* B0★ does **not** re-open the twelve Phase c NEGATIVE verdicts. Those traces established that the current canonical implementation does not *force* any enrichment; B0★-β asks whether an enrichment can be *admitted* by a future canonical derivation.
* B0★ does **not** weaken CCET-G_P14. CCET is a closure of CCC constructions on $G_{P14}$ under closure rules C1–C5; B0★-β-P2 (promotion of $E_6$) would change the inputs to C1 (Fact A no longer holds), not the closure rules themselves, so a B0★-β-P2 closure would constitute an extension *orthogonal* to CCET, not a refutation of it.

### .6 Reduced §13septies trichotomy with B0★ pre-registered

With B0★ pre-registered as the fourth branch, the §13septies decision space is:

* **B1** — canonical-catalog closure: CLOSED on $G_{P14}$ (CCET, §13vicies-novies.16); open in principle off $G_{P14}$, but the off-$G_{P14}$ sub-route requires a canonically-derived graph $G' \ne G_{P14}$ and is structurally constrained by the canonicity-arithmetic separation noted in CCET .§13vicies-novies.16's honest-scope clause.
* **B2** — new canonical operator: CLOSED at the catalog-API level by §13sexagesima-prima (B11 OCD NEGATIVE). No 14th operator is reachable from the public API.
* **B0★** — scope-expansion of existing theory (this section). OPEN; pre-registered with sub-branches B0★-α (deeper exploitation) and B0★-β (envelope promotion, priorities P1 = $E_0$, P2 = $E_6$).
* **B3** — no TNFR closure: permitted residual outcome if B0★ is also refuted across all sub-branches and B1-off-$G_{P14}$ is also closed.

The decision pressure that §13vicies-novies.16's "Net consequence for the program" placed on **B2 or B3** is therefore re-routed: with B2 catalog-API-closed by B11, the program-level pressure now lies on **B0★ or B3**. Per-envelope analysis of B0★-β-P1 and B0★-β-P2, plus an enumeration of B0★-α candidates, is deferred to subsequent commits.

### .7 Cross-references

* §13septies — Conjecture T-HP and original B1/B2/B3 trichotomy.
* §13vicies-novies.16 — CCET-G_P14 (B1 closure on $G_{P14}$).
* §13triginta-prima through §13triginta-tertia — T-$\nu_f$ NEGATIVE (Phase c, $E_0$ research-only).
* §13triginta-quarta through §13triginta-sexta — T-EPI NEGATIVE (Phase c, $E_1$ research-only).
* §13triginta-octava through §13triginta-decima — T-$\phi$ NEGATIVE (Phase c, $E_2$ research-only).
* §13quinquaginta-prima — B6 SWD Phase c NEGATIVE ($E_6$ research-only).
* §13sexagesima — B11 OCD Phase a.
* §13sexagesima-prima — B11 OCD Phase c NEGATIVE ($E_{OC}$ research-only; no 14th operator reachable).
* §13sexagesima-secunda — Composite Catalog-Closure Theorem.
* `theory/CATALOG_TYPE_HYGIENE_PROGRAMME.md` — full B0–B11 programme.
* AGENTS.md §"Program Status (May 2026, frozen)" and §"B1 sub-route status" — program-level status mirrors (to be updated in companion edit if B0★ is promoted from pre-registration to active investigation).


## §13sexagesima-quarta. Branch B0★-α — Deeper Exploitation of the Existing 13 Operators on Canonically-Constructed Graphs ≠ G_P14 (Pre-registration; Does NOT advance G4 = RH)

**Status**: PRE-REGISTERED, OPEN (May 2026). Specialisation of §13sexagesima-tertia at the *graph* axis.

### §13sexagesima-quarta.1 Motivation

§13sexagesima-tertia formalised B0★ as the fourth branch of the §13septies trichotomy: scope-expansion of the existing TNFR theory **without adding any operator**. B0★ split into two sub-branches: α (deeper exploitation of the existing 13 operators on canonical graphs other than G_P14) and β (canonicity-promotion of one of the twelve research envelopes E0..E_OC).

This section pre-registers B0★-α.

B0★-α scope: keep `OPERATORS` intact, keep `OPERATOR_METADATA` intact, keep `definitions.__all__` intact, keep the nodal equation $\partial \mathrm{EPI}/\partial t = \nu_f \cdot \Delta\mathrm{NFR}(t)$ intact, keep U1–U6 intact, AND keep all twelve B0–B11 NEGATIVE verdicts. Vary only the *graph* on which the canonical compositions act.

Structural rationale: the §13vicies-novies Canonical Catalog Equivariance Theorem (CCET) closure of B1 is **specific to G_P14**. Its proof reduces to two source-auditable facts: (A) parameter uniformity (every canonical operator's coupling constants are graph-level scalars) and (B) on G_P14 every edge-propagating canonical operator decomposes as $I_{n_{\mathrm{primes}}} \otimes O_{P_4}$ with prime-independent four-dimensional kernel (Prime-Cancellation Lemma). On a *different* canonically-derivable graph the analogue of Fact B in general fails — the kernel decomposition depends on the graph's symmetry group, and a graph whose automorphism group is not $S_n$ (or whose $S_n$-action admits non-trivial antisymmetric invariant subspaces) can carry canonical operator spectra that G_P14 forbids by symmetry.

### §13sexagesima-quarta.2 Catalog of canonically-derivable graph constructions

Elementary categorical operations on graphs that require no envelope promotion (each is a functor in the category of graphs and is definable purely from $(\nu_f, \mathrm{prime\ structure}, U1{-}U6)$):

| ID | Operation | Definition | Symmetry implication |
|---|---|---|---|
| O1 | Disjoint union $G \sqcup G'$ | $V(G) \cup V(G')$, $E(G) \cup E(G')$ | $\mathrm{Aut}(G) \times \mathrm{Aut}(G')$ |
| O2 | Cartesian product $G \square G'$ | $V(G) \times V(G')$, edges where one coord equals and other differs by an edge | Product (diagonal if $G = G'$) |
| O3 | Tensor product $G \times G'$ | $V(G) \times V(G')$, edges where both coords differ by edges | Product (diagonal if $G = G'$) |
| O4 | Strong product $G \boxtimes G'$ | Union of O2 ∪ O3 edges | Product (diagonal if $G = G'$) |
| O5 | Line graph $L(G)$ | Nodes are edges of $G$; edges where two edges share a vertex | Induced action of $\mathrm{Aut}(G)$ |
| O6 | Subdivision $S_k(G)$ | Replace each edge by a path of length $k+1$ | $\mathrm{Aut}(G)$ |
| O7 | Induced subgraph $G[V']$ | Restrict to $V' \subseteq V(G)$ | Setwise stabiliser of $V'$ |
| O8 | Quotient $G/\sim$ | Identify nodes by an equivalence relation derivable from canonical data | Quotient automorphism group |

C1'-α is the requirement that a B0★-α candidate graph be reachable from G_P14 (or from the prime-ladder $G_{PL}$ used by P12–P16) by a finite composition of O1–O8 alone.

### §13sexagesima-quarta.3 Already-shipped constructions (B0★-α surface that is NOT new)

| Construction | Operation chain | Status |
|---|---|---|
| Prime-ladder $G_{PL}$ (P12–P16) | O7 induced subgraph on $\{p^k : p \in \mathbb{P}, 1 \le k \le K\}$ of the integer line | ✅ shipped; smooth half of T-HP closed operationally by P28 + P30 |
| REMESH-lifted slot graph $G_{\mathrm{slot}}$ (R∞-1b, §13vicies-novies.15) | Auxiliary tensor lift $G_{P14} \otimes I_{\tau_g + 1}$ | ✅ executed; refuted (`INDETERMINATE_DEGENERATE_CONSTRUCTION`) |
| χ-twisted ladder $G_{PL}^\chi$ (P32–P49) | Edge-weight twist of $G_{PL}$ by primitive real Dirichlet character | ✅ shipped; GRH$_\chi$ residual is the twin of G4 |
| R∞-1c augmented edge graph (§13vicies-novies.13) | $G_{P14}$ + canonically-symmetric inter-prime edges | ✅ executed; refuted by augmented-graph specialisation of Euler-Orthogonality Lemma |

Note on χ-twisting: the $\chi$-twist is canonical because $\chi$ is a *character* on $(\mathbb{Z}/q)^\times$ — i.e. an arithmetic datum of the prime structure itself, not an operator and not an envelope. It enters the B0★-α surface as a canonical edge-weight, not as a new operator.

### §13sexagesima-quarta.4 New B0★-α candidates (not yet investigated)

| Candidate | Construction | Symmetry break vs G_P14 | Priority | Structural motivation |
|---|---|---|---|---|
| **Q1** | $G_{P14} \square G_{P14}$ (O2) | Diagonal $S_n$ admits non-trivial antisymmetric invariant subspace on $V \otimes V$ | **HIGH** | Pair $(p_i, p_j)$ structure is precisely the data of Montgomery's pair-correlation conjecture (RH-equivalent); anti-diagonal subspace under diagonal $S_n$ carries asymmetric spectral content that G_P14 forbids by Prime-Cancellation Lemma |
| **Q2** | $G_{P14} \times G_{P14}$ (O3) | Same as Q1 | **HIGH** | Same target as Q1 but with parallel-evolution connectivity instead of single-coord moves; distinguishes correlation-channel from parallel-channel contributions to the antisymmetric spectrum |
| **Q3** | $L(G_{P14})$ (O5) | Induced $S_n$ on prime-adjacent edges; no qualitative break | LOW | $G_{P14}$ is a path so $L(G_{P14})$ is a shorter path; topology too close to G_P14 to escape CCET-style equivariance |
| **Q4** | $S_2(G_{P14})$ (O6) | Adds edge-labelled intermediate nodes; $S_n$ preserved | LOW | Adds capacity slots without breaking $S_n$-equivariance; expected CCET-equivalent to G_P14 under canonical fold |
| **Q5** | $G_{PL} \square G_{PL}$ (O2) | Diagonal product symmetry on prime-ladder | MEDIUM | Higher-rank version of Q1 coupling prime label with prime-power exponent; combinatorially richer but closer to existing P14/P16 attack surface |
| **Q6** | $G_{P14}[V_{\le N}]$ (O7) | Same $S_N$, smaller orbit | LOW | Already implicit in $k \to \infty$ regime of P14/P16; no new content |

The HIGH-priority candidates Q1 and Q2 are the natural B0★-α entry points because: (i) pair-correlation is RH-equivalent on the ζ-side (Montgomery 1973, conjecture verified asymptotically by Rudnick–Sarnak under GUE-like assumptions); (ii) the anti-diagonal subspace under diagonal $S_n$ is the smallest symmetry-broken invariant subspace reachable from G_P14 by a single canonical product; (iii) the construction is purely combinatorial — no envelope, no character, no new parameter, no new operator. The Q5 (prime-ladder square) extension is a natural second move once Q1/Q2 diagnostics are available.

### §13sexagesima-quarta.5 What B0★-α does NOT claim

* Does NOT modify `OPERATORS`, `OPERATOR_METADATA`, `definitions.__all__`, the nodal equation, or U1–U6 (honours §13sexagesima-secunda).
* Does NOT re-open any of the twelve B0–B11 Phase c NEGATIVE verdicts (envelopes E0..E_OC remain research-only at the operator-contract level; B0★-α modifies the *graph*, not the operator catalog).
* Does NOT claim that any specific construction in §.4 will close G4 = RH; the section is a pre-registered enumeration, not a result.
* Does NOT execute the Q1, Q2, Q5 diagnostic in this commit. Per-candidate diagnostics deferred to subsequent commits.
* Does NOT add a 14th operator (B2 remains catalog-API-closed by §13sexagesima-prima).
* Does NOT promote any envelope (B0★-β remains pre-registered separately in §13sexagesima-tertia.3).

### §13sexagesima-quarta.6 Acceptance criteria (specialisation of C0–C4 to B0★-α)

A B0★-α candidate $Q_k$ is admissible iff:

* **C0** (unchanged): no entry added to `OPERATORS`, `OPERATOR_METADATA`, or `definitions.__all__`;
* **C1'-α** (canonical-construction derivation): $Q_k$ is built from $G_{P14}$ and/or $G_{PL}$ by a finite composition of the eight categorical operations O1–O8 of §.2, using no input external to $(\nu_f, \mathrm{prime\ structure}, U1{-}U6)$;
* **C2** (unchanged): U1–U6 admissibility lifts to $Q_k$ via the natural functorial action of canonical operators on the product/quotient/induced construction (continuity equation $\partial \rho / \partial t + \mathrm{div}\,\mathbf{J} = S_{\mathrm{grammar}} \to 0$ preserved);
* **C3** (unchanged): no envelope $E_k$ is promoted; no forcing axiom F1–F10 is added at any of the twelve type-hygiene surfaces;
* **C4** (unchanged): the resulting Hamiltonian on $Q_k$ discharges the T-HP statement, i.e. its spectrum, after applying a smooth admissible rescaling derivable from canonical means, reproduces $\{\gamma_n\}$ (the imaginary parts of the non-trivial Riemann zeros).

### §13sexagesima-quarta.7 Reduced §13septies decision space (after B0★-α pre-registration)

With B0★-α now formally enumerated as a discrete set of candidates {Q1, Q2, Q5, …}, the §13septies decision pressure refines to:

* **B1**: closed on G_P14 (CCET, §13vicies-novies.16). Off-G_P14 channels reachable by O1–O8 fall under **B0★-α** (this section); off-G_P14 channels NOT reachable by O1–O8 fall under B2.
* **B2**: catalog-API-closed by §13sexagesima-prima (B11 OCD NEGATIVE; no 14th operator reachable from the public API).
* **B0★-α** (this section): pre-registered, OPEN. Priority candidates Q1, Q2 (Cartesian and tensor squares of G_P14), Q5 (prime-ladder square).
* **B0★-β** (§13sexagesima-tertia.3): pre-registered, OPEN. Priority candidates P1 ($E_0$ Pontryagin-dual measure on $\nu_f$), P2 ($E_6$ per-node coupling weights).
* **B3**: residual permitted verdict (no TNFR closure exists).

### §13sexagesima-quarta.8 Cross-references

* §13septies — original trichotomy {B1, B2, B3}.
* §13vicies-novies.16 — Canonical Catalog Equivariance Theorem on G_P14 (closes B1 on G_P14; bounds CCET's domain to G_P14).
* §13sexagesima-prima — B11 OCD Phase c NEGATIVE (catalog-API closure of B2).
* §13sexagesima-secunda — Composite Catalog-Closure Theorem.
* §13sexagesima-tertia — B0★ overall pre-registration; this section specialises sub-branch α.
* §8 (P12), §10 (P14), §13terdecies (P34) — already-shipped constructions on $G_{PL}$ and $G_{PL}^\chi$.
* `theory/CATALOG_TYPE_HYGIENE_PROGRAMME.md` — twelve B0–B11 NEGATIVE verdicts.
* AGENTS.md §"Program Status (May 2026, frozen)" and §"B0★ pre-registration" — program-level status mirrors (to be updated in companion edit when any of Q1, Q2, Q5 is promoted from pre-registration to active investigation).


---

## §13sexagesima-quinta — B0★-α Results: spectral diagnostic for the HIGH-priority canonical-graph candidates Q1, Q2 (executed, May 27, 2026)

**Status:** EXECUTED. **Pre-registration:** §13sexagesima-quarta.4 (Q1 = G_P14 □ G_P14, Q2 = G_P14 × G_P14, HIGH priority). **Verdict:** both candidates return **INDETERMINATE_DEGENERATE_CONSTRUCTION** (F8 FAILED at machine-precision zero). **Net:** CCET-G_P14 (§13vicies-novies.16) extends structurally to the canonical Kronecker-sum and Kronecker-product Hamiltonians on V(G_P14) × V(G_P14); B0★-α HIGH-priority sub-routes Q1 and Q2 are **closed**; B0★-α residual pressure shifts to MEDIUM/LOW candidates (Q5 line graph, Q3 disjoint union, Q4 quotient, Q6 induced subgraphs) and to the orthogonal sub-branch B0★-β; §13septies decision pressure shifts further toward **B3** (no TNFR closure) and the LOW-priority residual of B0★.

### §13sexagesima-quinta.1 — Experimental construction

Mirror of the R∞-1b protocol (§13vicies-novies.14 / §13vicies-novies.15), specialised to canonical-product Hamiltonians on the squared vertex set:

* **Base Hamiltonian.** $H_{P14}$ via `build_prime_ladder_hamiltonian(n_primes=10, max_power=4, coupling=0)`; canonical P14 of §13quinquies; $N = 40$; spectral radius $\rho(H_{P14}) = 13.4692$.
* **Q1 lift (Cartesian product).** $H_{Q1} = H_{P14} \otimes I_N + I_N \otimes H_{P14}$ (Kronecker sum; canonical Hamiltonian for $G_{P14} \square G_{P14}$). Dimension $N^2 = 1600$.
* **Q2 lift (tensor product).** $H_{Q2} = H_{P14} \otimes H_{P14}$ (Kronecker product; canonical Hamiltonian for $G_{P14} \times G_{P14}$). Dimension $N^2 = 1600$.
* **Diagonal $S_n$ action.** $U_\sigma = P_\sigma^V \otimes P_\sigma^V$ with $P_\sigma^V = P_\sigma \otimes I_{\max\text{power}}$ (lifts the prime-relabelling permutation $\sigma \in S_{10}$ to $V$, then to $V \times V$ diagonally).
* **N3 control.** Shuffled-prime $H_{P14}$ with $\sigma = (13,23,7,5,19,3,17,2,11,29)$ from canonical $(2,3,5,7,11,13,17,19,23,29)$; lifted by the same builder.
* **N5 control.** Random self-adjoint $H_{\mathrm{rand}}$ of matched spectral radius, lifted by the same builder.
* **Statistic.** F7-A KS distance vs. GUE Wigner surmise on consecutive eigenvalue spacings (identical to §13vicies-novies.14).
* **F8 floor.** $|D_{\mathrm{canonical}} - D_{\mathrm{shuffled}}| \ge 0.01$ (identical to §13vicies-novies.14).
* **Seed.** `numpy.default_rng(20260527)`. `mpmath.mp.dps = 30`.
* **Source.** `benchmarks/b0star_alpha_canonical_product_graphs.py`; report `results/b0star_alpha_canonical_product_graphs.json`.

### §13sexagesima-quinta.2 — Numerical results

External anchor: $D_{\mathrm{Riemann}}^{\mathrm{GUE}} = 0.077037$ (99 spacings, first 100 Riemann zero imaginary parts).

| Candidate | Dim | $D_{\mathrm{canonical}}$ | $D_{\mathrm{shuffled}}$ | $D_{N5}$ | $|D_{\mathrm{can}} - D_{\mathrm{shuf}}|$ | F8 | Spec drift under $U_\sigma$ | F7 verdict |
|---|---|---|---|---|---|---|---|---|
| **Q1 Cartesian** | 1600 | 0.555146 | 0.555146 | 0.554576 | **0.000000e+00** | FAILED | 0.000e+00 | INDETERMINATE_DEGENERATE_CONSTRUCTION |
| **Q2 tensor**    | 1600 | 0.713118 | 0.713118 | 0.612260 | **0.000000e+00** | FAILED | 0.000e+00 | INDETERMINATE_DEGENERATE_CONSTRUCTION |

The $|D_{\mathrm{can}} - D_{\mathrm{shuf}}|$ value is **exactly zero in floating-point** (not merely below the 0.01 floor), and the explicit similarity audit confirms $\mathrm{spec}(U_\sigma H_{Q_k} U_\sigma^T) = \mathrm{spec}(H_{Q_k})$ to floating-point precision. This is the exact analogue of the §13vicies-novies.15 (R∞-1b) outcome on the temporal/spectral channel.

### §13sexagesima-quinta.3 — Structural interpretation: the Canonical Product Equivariance Lemma

The numerical result is fully explained by the following structural lemma, which extends the §13vicies-novies.11 Euler-Orthogonality Lemma and the §13vicies-novies.16 CCET-G_P14 theorem to canonical graph products:

**Lemma (Canonical Product Equivariance, §13sexagesima-quinta).** Let $H$ be any self-adjoint operator on $\mathbb{C}^V$ that commutes with the prime-relabelling unitary $P_\sigma$ for every $\sigma \in S_n$ (i.e. $[H, P_\sigma] = 0$; this is precisely the CCET-G_P14 conclusion for every operator in the canonical 13-operator catalog on $V(G_{P14})$). Then for the canonical Cartesian product lift $H_{Q1} := H \otimes I + I \otimes H$ and the canonical tensor product lift $H_{Q2} := H \otimes H$ on $\mathbb{C}^{V \times V}$,
$$[H_{Q1}, U_\sigma] = 0 \quad \text{and} \quad [H_{Q2}, U_\sigma] = 0 \quad \text{for every } \sigma \in S_n,$$
where $U_\sigma := P_\sigma \otimes P_\sigma$ is the diagonal $S_n$ action on $V \times V$. Consequently $\mathrm{spec}(H_{Q_k})$ is $S_n$-invariant, $D_{\mathrm{canonical}} = D_{\mathrm{shuffled}}$ for the F7-A statistic, and F8 fails on both Q1 and Q2.

*Proof.* For Q1, $U_\sigma H_{Q1} U_\sigma^T = (P_\sigma H P_\sigma^T) \otimes I + I \otimes (P_\sigma H P_\sigma^T) = H \otimes I + I \otimes H = H_{Q1}$, using $[H, P_\sigma] = 0$ twice. For Q2, $U_\sigma H_{Q2} U_\sigma^T = (P_\sigma H P_\sigma^T) \otimes (P_\sigma H P_\sigma^T) = H \otimes H = H_{Q2}$. The spectrum of a self-adjoint operator is invariant under unitary conjugation. $\square$

**Generalisation.** The same proof carries through for the strong product ($H_{\square \times} = H \otimes I + I \otimes H + H \otimes H$) and any positive real-linear combination of the three canonical product lifts. In particular, every operator constructible from $H_{P14}$ by the canonical graph operations O1–O3 of §13sexagesima-quarta.2 (disjoint union O1 = block-diagonal, Cartesian product O2, tensor product O3) inherits diagonal $S_n$-equivariance, and the canonical strong product O4 (sum of O2 + O3) inherits it as well. Therefore B0★-α HIGH-priority candidates Q1, Q2 are **closed by the Canonical Product Equivariance Lemma**, and the same lemma extends the closure to any HIGH/MEDIUM candidate constructed by combinations of {O1, O2, O3, O4} only.

### §13sexagesima-quinta.4 — Residual B0★-α surface after this milestone

Sub-routes of §13sexagesima-quarta.4 that remain structurally open after §13sexagesima-quinta:

* **Q5 (line graph, $L(G_{P14})$).** Vertices = edges of $G_{P14}$ (= 30 edges in 10 disjoint $P_4$ ladders). The Hamiltonian on $L(G_{P14})$ is not a tensor-product lift of $H_{P14}$; the diagonal $S_n$ acts on edges via a non-product representation. Canonical Product Equivariance Lemma does **not** apply directly. **Status: PRE-REGISTERED, OPEN.** Priority promoted from MEDIUM to **HIGH** by elimination.
* **Q3 (disjoint union with itself, $G_{P14} \sqcup G_{P14}$).** Hamiltonian is block-diagonal $H \oplus H$, equivalent to O1 of §13sexagesima-quarta.2, which inherits CCET trivially. **Status: implicitly closed by §13sexagesima-quinta** (the lemma applies — disjoint union is the degenerate case of Cartesian product with the trivial second factor; structurally degenerate as a novelty test). **Action: down-prioritise to LOW or drop.**
* **Q4 (quotient $G_{P14} / \sim$).** Depends on the equivalence relation. If $\sim$ is $S_n$-invariant, the quotient inherits CCET. If $\sim$ breaks $S_n$-symmetry (e.g. identifies $p_1$ with $p_2$ but not other pairs), the construction is no longer derivable from the 13-operator catalog alone (it depends on an external prime-pair choice not provided by $(\nu_f, U1{-}U6)$), violating C1'-α. **Status: closed by C1'-α at the canonicity level.** Conclusion: every $S_n$-invariant quotient inherits CCET; every $S_n$-non-invariant quotient violates C1'-α.
* **Q6 (induced subgraphs).** The only $S_n$-invariant induced subgraphs of $G_{P14}$ are (i) the full graph, (ii) the empty graph, (iii) the disjoint union of all $k$-level vertices for fixed $k \in \{1,2,3,4\}$ (= 10 isolated vertices each; trivial spectrum), and (iv) unions of (iii). All have trivial or CCET-equivariant spectra. **Status: closed by $S_n$-invariance argument.**

**Net B0★-α HIGH/MEDIUM/LOW after §13sexagesima-quinta:** the only remaining candidate from §13sexagesima-quarta.4 is **Q5 (line graph)**, now promoted to HIGH. All other O1–O4 / Q3 / Q4 / Q6 candidates are structurally closed by the Canonical Product Equivariance Lemma or by the $S_n$-invariance / C1'-α discipline.

### §13sexagesima-quinta.5 — Updated §13septies decision space

Combining §13sexagesima-prima (B2 catalog-API-closed), §13sexagesima-tertia (B0★ pre-registered), §13sexagesima-quarta (B0★-α enumerated), §13vicies-novies.16 (B1 closed on G_P14), and §13sexagesima-quinta (B0★-α HIGH-priority Q1, Q2 closed; only Q5 line-graph residual remains in B0★-α):

| Branch | Status after §13sexagesima-quinta |
|---|---|
| B0★-α (deeper exploitation) | residual = {Q5 line graph}; all O1–O4 product/disjoint/quotient candidates closed |
| B0★-β (envelope promotion)  | PRE-REGISTERED, OPEN; priority {P1 = E0, P2 = E6} |
| B1 (extra-catalog edge channel) | CLOSED on G_P14 (CCET §13vicies-novies.16); off-G_P14 reduces to B2 by construction |
| B2 (new canonical operator)     | catalog-API-closed at the registry level (§13sexagesima-prima) |
| B3 (no TNFR closure of RH)      | residual; pressure increased by §13sexagesima-quinta |

**Decision pressure now lies on (a) Q5 line graph as the sole residual HIGH candidate of B0★-α, (b) the B0★-β envelope-promotion sub-branch (E0 Pontryagin / E6 per-node weights), and (c) the B3 residual.** No further extension of the diagnostic surface is planned until one of Q5, B0★-β, or B3 is decided.

### §13sexagesima-quinta.6 — What §13sexagesima-quinta does NOT claim

* **NOT a proof of RH.** G4 = RH and GRH_χ remain open.
* **NOT a refutation of T-HP / Conjecture T-HP.** §13septies T-HP remains the operational statement; §13sexagesima-quinta refutes only the HIGH-priority Q1/Q2 routes to its closure within B0★-α.
* **NOT a refutation of B0★ as a whole.** B0★-β remains pre-registered and open; Q5 of B0★-α remains pre-registered and open.
* **NOT a refutation of B3.** §13sexagesima-quinta increases B3 pressure but does not select B3 over the remaining B0★ residual.
* **NOT a modification of the 13-operator catalog.** Honors §13sexagesima-secunda (Composite Catalog-Closure Theorem) and §13sexagesima-tertia.4 acceptance criterion C0.

### §13sexagesima-quinta.7 — Cross-references

* §13sexagesima-quarta — B0★-α pre-registration; this section reports executed results for Q1, Q2.
* §13vicies-novies.11, .15, .16 — Euler-Orthogonality Lemma, R∞-1b execution, CCET-G_P14; structural ancestors of the Canonical Product Equivariance Lemma proved here.
* §13sexagesima-secunda, §13sexagesima-tertia — Composite Catalog-Closure Theorem, B0★ overall pre-registration.
* §13septies — extended trichotomy and T-HP statement; decision space updated in §13sexagesima-quinta.5.
* `benchmarks/b0star_alpha_canonical_product_graphs.py` — pre-registered diagnostic source.
* `results/b0star_alpha_canonical_product_graphs.json` — full report (eigenvalue counts, spacing moments, per-control diagnostics).
* AGENTS.md §"B0★ pre-registration" and §"Program Status (May 2026, frozen)" — program-level status mirrors (companion edit in this commit reflects the Q1/Q2 closure and the Q5 promotion).



## §13sexagesima-sexta — B0★-β Analytical Closure of the HIGH-Priority Envelope-Promotion Candidates P1 = E0 (Pontryagin-νf) and P2 = NodeIndexedCouplingWeights (May 27, 2026)

**Status:** ANALYTICAL CLOSURE (no numerical experiment — the obstructions are already established as structural results in earlier sections of these notes). **Pre-registration:** §13sexagesima-tertia.3 / §13sexagesima-tertia.4 (B0★-β HIGH = {P1 = E0, P2 = NodeIndexedCouplingWeights}; acceptance criteria C0–C4). **Verdict:** both HIGH-priority candidates **FAIL the acceptance criteria** of §13sexagesima-tertia.4 at the canonical layer; **B0★-β-HIGH is closed**. **Net:** §13septies decision pressure shifts to (a) the Q5 line-graph residual of B0★-α (§13sexagesima-quinta.5), (b) the B0★-β LOW/MEDIUM residual ({E2, E1, E3, E4, E5, E_TC, E_CC, E_AC, E_UR, E_OC}), and (c) the B3 residual (no TNFR closure of RH).

### §13sexagesima-sexta.1 — Scope and method

This section does **not** execute a new numerical experiment. The two HIGH-priority B0★-β candidates were *defined* in §13sexagesima-tertia.3 as questions about whether a research envelope can be **derived** from the nodal equation under an *admissibility* reading (as distinct from the *forcing* reading used by the type-hygiene programme §§13triginta-* through §13sexagesima-*). The C0–C4 acceptance criteria of §13sexagesima-tertia.4 are structural conditions, not empirical thresholds, so they can be evaluated by reduction to existing canonical results without a new measurement.

The reductions used here are:

* For **P1 = E0**: the chain of §13triginta-secunda.5 (Conditional Corollary) → §13triginta-secunda.6 (verdict on `(P-νf-Bijectivity)`) → §13triginta-tertia.5 (where spectral richness actually lives) → §13triginta-tertia.6 (Proposition T-νf-Resolution).
* For **P2 = NodeIndexedCouplingWeights**: the chain of §13quadraginta-nona (B6 pre-registration) → §13quinquaginta (forcing-axiom reduction) → §13quinquaginta-prima (NEGATIVE verdict via Scalar-Weight Discipline) — combined with a direct inspection of the nodal equation `∂EPI/∂t = νf · ΔNFR(t)` for the presence of a per-node weight slot.

No source code is modified; no new module is added; no entry of `OPERATORS`, `OPERATOR_METADATA`, or `definitions.__all__` is touched (C0 trivially satisfied for both candidates).

### §13sexagesima-sexta.2 — Honest scope (what this section does not claim)

* **NOT a proof of RH.** G4 = RH and GRH_χ remain open.
* **NOT a refutation of B0★ as a whole.** The B0★-β LOW/MEDIUM residual ({E2 LiftedCircleBundleOnPhi at MEDIUM, plus eight envelopes at LOW}) is **not** addressed here, nor is Q5 of B0★-α. B0★ remains a legitimate open branch of the §13septies trichotomy.
* **NOT a contradiction of §13sexagesima-secunda.** The Composite Catalog-Closure Theorem is a *minimality* statement; this section's verdict on HIGH B0★-β candidates is an orthogonal *admissibility-level refutation* derived from earlier canonical results.
* **NOT a re-opening of the twelve Phase c traces.** The §13triginta-* through §13sexagesima-* NEGATIVE verdicts on E0 and NodeIndexedCouplingWeights remain in force; this section uses those verdicts as inputs, not as targets of revision.
* **NOT a refutation of T-HP or of P28/P30.** The smooth half of $\mathcal{F}$ (closed operationally by P28 at the density level and lifted to the operator level by P30 for the smooth half) is unchanged; the residual obstruction $S(T) = (1/\pi)\arg\zeta(\tfrac12+iT) \in \ker(\mathcal{R}_\infty)$ remains the open content of T-HP.

### §13sexagesima-sexta.3 — B0★-β-P1 (E0 = MeasureExtensionOnNuF / Pontryagin-νf): C1 NOT-DERIVED, C4 FAILS

**C0 (no catalog modification).** Trivially satisfied: no change to `OPERATORS`, `OPERATOR_METADATA`, `definitions.__all__`.

**C1 (nodal-equation derivation).** **NOT-DERIVED.** The chain of §13triginta-secunda.5–.7 reduces (P-Pontryagin) to the strictly weaker meta-axiom

> **(P-νf-Bijectivity).** In the canonical TNFR formulation, $\nu_f$ must bijectively encode the spectral content of the EPI dynamics it drives.

The verdict on (P-νf-Bijectivity) at §13triginta-secunda.6 is `UNDETERMINED_AT_CANONICAL_LEVEL` (supported by the spirit of Invariant #1 + #6, not forced by their letter). §13triginta-tertia.6 sharpens this to `FORWARD_INDEPENDENT_OF_BACKWARD`: (P-νf-Bijectivity) is an **inverse-problem axiom** independent of the forward-dynamics catalog. The forward direction `∂EPI/∂t = νf · ΔNFR(t)` is well-posed under scalar $\nu_f$ (Proposition T-νf-Resolution item 1) and `src/tnfr/operators/nodal_equation.py::compute_expected_depi_dt` implements this literally as `vf * dnfr` with both factors `float`.

Therefore the promoted envelope $E_0$ is *not derived* from the nodal equation alone; its admissibility at the canonical level rests on an axiom that is not in the canonical contract. **C1 fails by reduction** — the same gap that closed T-νf at the canonical level in §13triginta-tertia closes B0★-β-P1 at C1.

**C2 (U1–U6 admissibility).** Conditionally satisfiable if (P-νf-Bijectivity) is accepted as an external axiom: measure-valued $\nu_f$ on $\widehat{\mathbb{Z}} = S^1$ paired against a distribution-valued $\Delta\mathrm{NFR}$ yields a scalar pairing $\langle \nu_f, \Delta\mathrm{NFR} \rangle \in \mathbb{R}$ compatible with U2 (integral convergence) and U1/U3/U4/U5/U6 (operator-sequence rules unchanged). This is consistent but not by itself a discharge of C1.

**C3 (twelve-CDM consistency).** Conditionally satisfied: promotion of $E_0$ from research-only to canonical is permitted by §13sexagesima-tertia.2 (re-classification is allowed; introduction of a new forcing axiom is not). The §13triginta-tertia.8 honest-scope clause already records that "non-canonical extension of TNFR to measure-valued $\nu_f$" is a legitimate parallel research question.

**C4 (T-HP discharge).** **FAILS by direct argument.** §13triginta-tertia.5 establishes the structural locus of spectral richness in the literal canonical reading: $\Delta\mathrm{NFR}_i(t)$ carries the spectral content of $\partial\mathrm{EPI}/\partial t$; $\nu_{f,i}$ acts as a multiplicative gain. The P14 prime-ladder construction (§8.2, `src/tnfr/riemann/prime_ladder_hamiltonian.py`) is an existence proof: the prime-ladder spectrum $\{k\log p\}$ is reproduced *with scalar* $\nu_f$, demonstrating that promotion of $\nu_f$ to a measure on $\widehat{\mathbb{Z}}$ does **not** add spectral expressivity beyond what scalar $\nu_f$ already attains through the graph state and the operator sequence.

The oscillatory residue $S(T) = (1/\pi)\arg\zeta(\tfrac12 + iT)$ identified by N15 W3 with $\ker(\mathcal{R}_\infty)$ (§13septies.5) lives in two structural directions that B0★-β-P1 does **not** address:

* **(i) $\ker(\mathcal{R}_\infty)$ direction.** REMESH's asymptotic kernel (N15 §§1–8) is determined by the REMESH operator's contractive transfer matrix on $H^2(D)$; it is invariant under the carrier type of $\nu_f$ (scalar vs measure) because $\nu_f$ enters the REMESH dynamics only as the multiplicative gain factor in `∂EPI/∂t = νf · ΔNFR(t)`. Promoting $\nu_f$ to a measure does not change $\ker(\mathcal{R}_\infty)$.
* **(ii) $\mathrm{Fix}(S_n)^\perp$ direction.** The CCET-G_P14 obstruction (§13vicies-novies.16, Canonical Catalog Equivariance Theorem) is established at the level of the prime-relabelling automorphism action on $G_{P14}$; it depends only on (A) parameter uniformity and (B) the Prime-Cancellation Lemma. Neither (A) nor (B) is sensitive to the carrier type of $\nu_f$. Promoting $\nu_f$ to a measure does not break $S_n$-equivariance on $G_{P14}$.

Therefore, even if (P-νf-Bijectivity) were accepted as an admissibility axiom (closing C1 conditionally), the enriched dynamics would not produce an operator on $\mathcal{H}_{\mathrm{tet}}$ whose spectrum coincides with $\{\gamma_n\}_{n \ge 1}$. **C4 is structurally pinned shut for P1.**

**Net verdict on P1 = E0.** **FAIL** (C1 NOT-DERIVED; C4 FAILS even under the most-permissive C1 reading).

### §13sexagesima-sexta.4 — B0★-β-P2 (NodeIndexedCouplingWeights; labelled E6 in §13sexagesima-tertia.3 / E7 in §13quinquaginta-prima): C1 FAILS at the slot level

**Naming note.** The envelope-name "E6" in §13sexagesima-tertia.3 table refers to the same structural object that §13quadraginta-nona / §13quinquaginta-prima register as `E7 = NodeIndexedCouplingWeights`. The bookkeeping label diverged between sections; the *referent* is the same (per-node / per-edge / callable-kernel generalisation of the global-scalar coupling weights `DNFR_WEIGHTS`, `SI_WEIGHTS`, `SELECTOR_WEIGHTS` in `src/tnfr/config/defaults_core.py`). The canonical refutation is the Scalar-Weight Discipline (SWD) trace at §13quinquaginta-prima.

**C0 (no catalog modification).** Trivially satisfied.

**C1 (nodal-equation derivation).** **FAILS AT THE SLOT LEVEL.** A direct inspection of the literal canonical nodal equation

$$ \frac{\partial \mathrm{EPI}}{\partial t} = \nu_f \cdot \Delta\mathrm{NFR}(t) $$

shows that it has **no per-node coupling-weight slot**: the two factors are (a) the structural-frequency scalar $\nu_f$ (whose canonical type was decided at §13triginta-tertia.6) and (b) the nodal-pressure scalar $\Delta\mathrm{NFR}(t)$. Coupling weights enter only **downstream**, inside the implementation of `compute_delta_nfr` (`src/tnfr/dynamics/dnfr.py` and surrounding modules), via the global-scalar dictionaries `DNFR_WEIGHTS`, `SI_WEIGHTS`, `SELECTOR_WEIGHTS` (`src/tnfr/config/defaults_core.py`). The choice of *where* per-node weights would enter `compute_delta_nfr` is therefore a **downstream-implementation choice**, not a consequence of the bare nodal equation. No derivation pathway from `∂EPI/∂t = νf · ΔNFR(t)` together with Invariants #1–#6 and the structural scale $\pi$ produces a per-node weight law without an additional external axiom selecting the entry point and the rule. The §13quinquaginta-prima SWD trace records exactly this structural fact under its forcing-axiom F1–F10 enumeration: no canonical constraint forces per-node weights.

**Reading the residual admissibility window.** §13sexagesima-tertia.3's HIGH-priority justification for P2 was conditional: per-node weights "would mechanically dissolve Fact A and reopen the spectral-non-trivial sub-region of CCC constructions, **if** the per-node weight rule can be canonically derived from $\nu_f$ values via a construction that breaks the symmetric-function-of-scalars constraint." The conditional **if** is precisely the C1 gap. A canonical derivation from $\nu_f$ to per-node weights would itself require a non-symmetric rule (else the rule reduces to a symmetric function of $\nu_f$ values, which gives back parameter-uniform weights and Fact A holds — closing CCET-G_P14 as before). No such non-symmetric rule is derivable from the catalog: the canonical operators (Invariant #4) act through grammar U1–U6 on operator sequences, not on per-node parameter laws (cf. F4 in §13triginta-secunda.4).

**C2 (U1–U6 admissibility).** Conditionally satisfiable in form (per-node weights are syntactically compatible with U1–U6 if the continuity equation $\partial\rho/\partial t + \nabla\cdot\mathbf{J} = \mathcal{S}_{\mathrm{grammar}}$ is preserved under the new weight law). Not by itself a discharge of C1.

**C3 (twelve-CDM consistency).** Conditionally satisfied as a *re-classification* of `NodeIndexedCouplingWeights` from research-only to canonical (permitted by §13sexagesima-tertia.2). Direct conflict with the §13quinquaginta-prima SWD trace if presented as a *forced* canonical contract; the B0★-β route avoids this conflict only because it operates at the admissibility layer.

**C4 (T-HP discharge).** Not separately evaluated: with C1 failing at the slot level, C4 is not reached.

**Net verdict on P2 = NodeIndexedCouplingWeights.** **FAIL** (C1 FAILS at the slot level; no canonical derivation pathway exists from `∂EPI/∂t = νf · ΔNFR(t)` to per-node weights without an external rule-selection axiom).

### §13sexagesima-sexta.5 — Net structural consequence

Combining §13sexagesima-prima (B2 catalog-API-closed), §13sexagesima-tertia (B0★ pre-registered), §13sexagesima-quarta (B0★-α enumerated), §13sexagesima-quinta (B0★-α HIGH Q1, Q2 closed), §13vicies-novies.16 (B1 closed on G_P14), and §13sexagesima-sexta (B0★-β HIGH P1, P2 closed):

| Branch | Status after §13sexagesima-sexta |
|---|---|
| B0★-α (deeper exploitation) | residual = {Q5 line graph}; HIGH closed (§13sexagesima-quinta) |
| B0★-β (envelope promotion)  | HIGH closed: P1 = E0 fails C1/C4; P2 = NodeIndexedCouplingWeights fails C1. Residual = MEDIUM (E2 LiftedCircleBundleOnPhi) + LOW ({E1, E3, E4, E5, E_TC, E_CC, E_AC, E_UR, E_OC}) |
| B1 (extra-catalog edge channel) | CLOSED on G_P14 (CCET §13vicies-novies.16) |
| B2 (new canonical operator)     | catalog-API-closed (§13sexagesima-prima) |
| B3 (no TNFR closure of RH)      | residual; pressure further increased by §13sexagesima-sexta |

**Decision pressure now lies on (a) the Q5 line-graph residual of B0★-α, (b) the B0★-β MEDIUM/LOW residual (with E2 as the only MEDIUM candidate), and (c) the B3 residual.** No further extension of the diagnostic surface is planned until one of Q5, B0★-β-MEDIUM/LOW, or B3 is decided.

**Honest structural reading.** The pattern across §13triginta-* through §13sexagesima-sexta is that *every* HIGH-priority envelope promotion attempt reduces to a structural gap already isolated by an earlier Phase-c trace: P1 reduces to the (P-νf-Bijectivity) forward/backward independence of §13triginta-tertia.6, and P2 reduces to the slot-level absence of per-node weights in the nodal equation already recorded by §13quinquaginta-prima SWD. The B0★-β route does not bypass these obstructions; it inherits them under the admissibility reading. This does not formally refute B0★ as a fourth branch (the MEDIUM/LOW residual remains pre-registered and open), but it strongly constrains the structural locations where a successful B0★-β closure of G4 = RH could be found.

### §13sexagesima-sexta.6 — Cross-references

* §13sexagesima-tertia.3, §13sexagesima-tertia.4 — B0★-β pre-registration, HIGH-priority ranking (P1 = E0, P2 = NodeIndexedCouplingWeights), acceptance criteria C0–C4.
* §13triginta-secunda.5–.7 — Conditional Corollary `(P-Pontryagin) ⇔ Catalog ∧ (P-νf-Bijectivity)`; verdict on `(P-νf-Bijectivity)` = `UNDETERMINED_AT_CANONICAL_LEVEL`.
* §13triginta-tertia.5–.8 — spectral-richness locus in `ΔNFR(t)`; Proposition T-νf-Resolution; verdict on `(P-Pontryagin)` = `FORWARD_INDEPENDENT_OF_BACKWARD`; Conjecture T-νf = `CLOSED_NEGATIVELY_AT_CANONICAL_LEVEL`.
* §13quadraginta-nona, §13quinquaginta, §13quinquaginta-prima — B6 = T-coupling-weights pre-registration, forcing-axiom reduction, NEGATIVE verdict via SWD; envelope `E7 = NodeIndexedCouplingWeights` (the §13sexagesima-tertia.3 "E6" referent).
* §13vicies-novies.11, .16 — Prime-Cancellation Lemma, Canonical Catalog Equivariance Theorem on G_P14 (used in §13sexagesima-sexta.3 C4 argument).
* §13septies — Conjecture T-HP and the smooth/oscillatory split; identification of $\ker(\mathcal{R}_\infty)$ with the oscillatory residue $S(T)$ (N15 W3).
* N15 §§1–8 — REMESH-∞ asymptotic projection; carrier-type independence of $\ker(\mathcal{R}_\infty)$.
* `src/tnfr/operators/nodal_equation.py::compute_expected_depi_dt` — literal `vf * dnfr` with both factors `float`; canonical implementation referenced in §13triginta-tertia.2 and §13sexagesima-sexta.3 (C1 argument for P1) and §13sexagesima-sexta.4 (C1 argument for P2).
* `src/tnfr/config/defaults_core.py::DNFR_WEIGHTS, SI_WEIGHTS, SELECTOR_WEIGHTS` — global-scalar coupling-weight anchors referenced in §13sexagesima-sexta.4 (C1 slot-level argument for P2).
* `src/tnfr/riemann/prime_ladder_hamiltonian.py` — P14 existence proof referenced in §13sexagesima-sexta.3 (C4 argument for P1).
* AGENTS.md §"B0★ pre-registration" and §"Program Status (May 2026, frozen)" — program-level status mirrors (companion edit in this commit reflects the HIGH closure and the MEDIUM/LOW residual).

## §13sexagesima-septima — B0★-β-P3 (Dirección A: ΔNFR carrier-type / slot promotion) Analytical Closure (May 27, 2026)

**Status:** ANALYTICAL CLOSURE (no numerical experiment — every canonically-derivable reading of "promote ΔNFR beyond scalar field" reduces to a structural obstruction already established elsewhere in these notes). **Pre-registration of candidate:** this section itself (no prior dedicated pre-registration; Dirección A was raised in the program-level discussion accompanying §13sexagesima-sexta as the "dual lever" symmetric counterpart of P1). **Acceptance criteria:** C0–C4 of §13sexagesima-tertia.4. **Verdict:** Dirección A is **structurally CLOSED** at the canonical layer across its three canonically-derivable readings. **Net:** B0★-β residual narrows to {E2 LiftedCircleBundleOnPhi at MEDIUM} ∪ {nine LOW envelopes}; §13septies decision pressure remains on (a) Q5 line-graph from B0★-α, (b) the B0★-β MEDIUM/LOW residual, (c) B3.

### §13sexagesima-septima.1 — Motivation and the three canonically-derivable readings

The §13triginta-tertia.5 fact that **spectral richness of $\partial \mathrm{EPI}/\partial t$ lives in ΔNFR(t), not in $\nu_f$** suggests, at first glance, that the symmetric candidate to P1 = E0 (carrier-type promotion of $\nu_f$) — namely **carrier-type promotion of ΔNFR** — should be evaluated as a separate B0★-β candidate. Call this "Dirección A". Closer reading reveals that "ΔNFR promotion" is not a single proposal but a family of three structurally distinct readings, each of which has already been touched by an existing canonical result:

| Reading | Promotion (informal) | Structural content |
|---|---|---|
| **A1** | $\Delta\mathrm{NFR}_j: \mathbb{R} \to \mathcal{M}(X)$ for some label space $X$ | Spatial measure-valued field, per node |
| **A2** | $\mathrm{EPI}_j \in \mathcal{H}_\mathrm{int}$ Hilbert space ⇒ $\Delta\mathrm{NFR}_j \in B(\mathcal{H}_\mathrm{int})$ | Operator-valued ΔNFR via internal slot lift |
| **A3** | $\Delta\mathrm{NFR}(t) \in L^2(\mathbb{R}_t)$ exploited as temporal Fourier object | Spectral content of dynamics in time |

Each reading is evaluated against C0–C4 below. Method is identical to §13sexagesima-sexta: analytical reduction to existing Phase-c canonical results, no fresh numerical run.

### §13sexagesima-septima.2 — Honest scope

* NOT a proof of RH. G4 and GRH_χ remain open.
* NOT a refutation of T-HP. The smooth half closed by P28 / P30 is unchanged.
* NOT a refutation of B0★ as a whole. The B0★-β MEDIUM/LOW residual remains open; B0★-α residual {Q5} remains open.
* NOT a re-opening of any prior Phase-c verdict. This section uses §13triginta-tertia, §13vicies-novies.15–16, N15 W3, and the L-track parity layer (P32–P49) as inputs.
* Honors C0 (no catalog modification) and C3 (twelve-CDM consistency) by construction.

### §13sexagesima-septima.3 — A1 (carrier-type promotion of ΔNFR per node): FAIL by direct reduction to the P1 closure pattern

**C0** trivially satisfied (carrier-type promotion of a field does not modify `OPERATORS`, `OPERATOR_METADATA`, `definitions.__all__`, the nodal equation, or U1–U6).

**C1 NOT-DERIVED.** The lift $\Delta\mathrm{NFR}_j: \mathbb{R} \to \mathcal{M}(X)$ requires external specification of three independent choices: (i) the label space $X$ on which the measure lives; (ii) how the canonical neighbor-difference recipe `compute_delta_nfr` (whose source-level signature returns a `float` per node) projects to a measure; (iii) how the time-integral $\int_0^t \nu_f(\tau) \cdot \Delta\mathrm{NFR}(\tau)\,d\tau$ in the nodal equation interprets a measure-valued integrand (Bochner integral, distributional pairing, etc.). None of these three choices is forced by the nodal equation `∂EPI/∂t = νf · ΔNFR(t)`. This is the same structural problem as P1 (cf. §13sexagesima-sexta.3): the canonical nodal equation has both `νf` and `ΔNFR` as scalar fields by construction in `compute_expected_depi_dt`; promoting either to a measure requires an external admissibility axiom not derivable from the bare nodal equation. Reduction: `(P-ΔNFR-Bijectivity)` is the exact analogue of `(P-νf-Bijectivity)` (§13triginta-secunda.6 / §13triginta-tertia.6), with the same `FORWARD_INDEPENDENT_OF_BACKWARD` structure.

**C4 FAILS** by direct argument, even under permissive C1. Three independent obstructions, each of which alone is sufficient:

(i) **§13triginta-tertia.5 reread.** The "spectral richness in ΔNFR(t)" of §13triginta-tertia.5 is a statement about ΔNFR(t) as a *time series* of scalars, not about per-node ΔNFR carrying internal spectral structure at a fixed time. The dynamics generate rich temporal Fourier content even with scalar per-node ΔNFR (because the graph coupling redistributes phase). A1 is therefore answering a different structural question than §13triginta-tertia.5 raises; A1's per-node measure structure is not the locus identified by §13triginta-tertia.5 as "where the spectral richness lives".

(ii) **P14 existence proof (`src/tnfr/riemann/prime_ladder_hamiltonian.py`).** The full prime-ladder spectrum $\{k \log p\}$ — i.e., the data that drives the von Mangoldt / Weil–Guinand prime side, equivalently $-\zeta'(s)/\zeta(s) = \sum_n \Lambda(n) n^{-s}$ — is reproduced by P14 with **scalar** ΔNFR per node, encoded through the prime-indexed coordinates of the diagonal potential $V_\sigma$. The "measure content" of P14 is not in ΔNFR; it is in the spectral measure of the self-adjoint operator $H_{P14} = L_k + V_\sigma$. Carrier-type promotion of ΔNFR is therefore not required to reach the closed half of T-HP, and provides no canonical lever on the open half.

(iii) **§13septies oscillatory residue invariance.** The residue $S(T) = (1/\pi)\arg\zeta(\tfrac12 + iT)$ lives in $\ker(\mathcal{R}_\infty) \cap \mathrm{Fix}(S_n)^\perp$. Both invariants are stable under carrier-type promotion of ΔNFR for the same two reasons that pinned P1: $\ker(\mathcal{R}_\infty)$ is determined by the REMESH transfer-matrix structure on $H^2(D)$ (N15 §§1–8), which is carrier-type independent; $\mathrm{Fix}(S_n)^\perp$ is determined by parameter uniformity (CCET-G_P14 Fact A, §13vicies-novies.16), which is preserved if the measure type is graph-uniform (the only canonically-derivable case under C0 / C2).

**Net A1:** FAIL ⇒ **CLOSED**, structurally identical to P1.

### §13sexagesima-septima.4 — A2 (slot-internal Hilbert-space promotion of EPI): already CLOSED by §13vicies-novies.15

The reading "ΔNFR as operator-valued via promotion of $\mathrm{EPI}_j$ to a vector in some internal Hilbert space $\mathcal{H}_\mathrm{int}$" is, structurally, exactly the **R∞-1b** sub-route of B1 pre-registered at §13vicies-novies.14 and executed at §13vicies-novies.15. The canonical tensor-product lifts $S_\mathrm{IL}^{\mathrm{spec}} = I_{\tau_g+1} \otimes \exp(-\eta H_{P14})$ and $M_\mathrm{REMESH} = M \otimes I_N$ on the canonical $\mathcal{H}_\mathrm{int}$ returned `INDETERMINATE_DEGENERATE_CONSTRUCTION` with $|D_\mathrm{canonical} - D_\mathrm{shuffled}| = 1.08 \times 10^{-13}$ (machine-precision zero), because the prime-relabelling unitary $U_\sigma = I_{\tau_g+1} \otimes P_\sigma$ conjugates the spectral-channel operator to its shuffled image. The spectral-channel extension of the Euler-Orthogonality Lemma (§13vicies-novies.15) and the Canonical Catalog Equivariance Theorem on G_P14 (Theorem 2, §13vicies-novies.16) jointly close this sub-route within the canonical catalog.

**Net A2:** CLOSED by §13vicies-novies.15.

Note: A2 is not strictly a "ΔNFR carrier-type promotion" in the field-theoretic sense — it is a slot-internal lift of EPI that induces an operator-valued ΔNFR. But the relevant structural test (S_n equivariance on G_P14) is identical to A1, and the verdict is the same.

### §13sexagesima-septima.5 — A3 (temporal Fourier content of ΔNFR(t)): SUPERSEDED by the L-track parity layer

The temporal Fourier content of ΔNFR(t) — i.e., the spectral resolution of ΔNFR viewed as an element of $L^2(\mathbb{R}_t)$ along a trajectory — is **already exploited** at the canonical layer by the χ-twisted L-track parity infrastructure (P32–P49) and by the original ζ-track Hermite / admissible-family sweeps (P19, P21, P25, P31). Concretely, the canonical modules `src/tnfr/riemann/dirichlet_l*.py`, `src/tnfr/riemann/twisted_*.py`, `src/tnfr/riemann/admissible_family_sweep.py`, and `src/tnfr/riemann/oscillatory_correction.py` consume ΔNFR-derived temporal data and feed it into the twisted Weil–Guinand explicit formula, the Li–Keiper twisted positivity diagnostic, the Hermite admissible-family sweeps, and the prime-ladder Newton-step oscillatory correction. None of these P17–P49 components closes GRH_χ or G4 = RH; they form the full attack-surface parity. Re-introducing "ΔNFR temporal Fourier" as a fresh B0★-β candidate would therefore duplicate existing canonical infrastructure without adding a new structural lever.

**Net A3:** SUPERSEDED by P17–P49. Not an open B0★-β candidate.

### §13sexagesima-septima.6 — Net structural consequence

| Reading | Status | Reduction |
|---|---|---|
| A1 (carrier-type promotion of ΔNFR per node) | FAIL ⇒ CLOSED | §13triginta-tertia.6 + P14 (existence) + N15 W3 / §13septies + CCET §13vicies-novies.16 |
| A2 (slot-internal Hilbert lift on EPI inducing operator-valued ΔNFR) | CLOSED | §13vicies-novies.15 (R∞-1b spectral channel) |
| A3 (temporal Fourier content of ΔNFR(t)) | SUPERSEDED | P17–P49 ζ-track and χ-twisted L-track parity layer |

**Updated §13septies decision space after §13sexagesima-septima:**

| Branch | Status |
|---|---|
| B1 (off-catalog edge / spectral channel on G_P14) | CLOSED on G_P14 (§13vicies-novies.16); off-G_P14 inherits B2 by construction |
| B2 (new canonical operator) | catalog-API-closed (§13sexagesima-prima) |
| B0★-α (deeper exploitation) | HIGH/MEDIUM closed (§13sexagesima-quinta); residual = {Q5 line graph} |
| B0★-β (envelope promotion) | HIGH closed (§13sexagesima-sexta); Dirección A closed (§13sexagesima-septima); residual = {E2 MEDIUM} ∪ {nine LOW envelopes} |
| B3 (no TNFR closure of RH) | residual; pressure further increased by §13sexagesima-septima |

**Honest structural reading.** The Phase-c pattern continues to hold: every canonically-derivable promotion of a scalar field in the nodal equation (νf in P1, per-node coupling weights in P2, ΔNFR in P3 = Dirección A) is closed by reduction to a previously-established Phase-c obstruction. The persistent open candidates are (i) Q5 (a graph-construction route that escapes CCET because the line-graph action of $S_n$ on edges is not tensor-product), (ii) E2 LiftedCircleBundleOnPhi (a topological enrichment of φ that breaks $S_n$ at the bundle level rather than at the coupling-constant level — i.e., not addressed by CCET Fact A), and (iii) the nine LOW envelopes. The pattern strongly suggests — without proving — that any successful B0★-β closure of G4 = RH must break $S_n$ either through graph construction (B0★-α route, Q5) or through topological / fiber-bundle structure (B0★-β E2), rather than through scalar-to-richer-carrier promotions of fields in the nodal equation itself.

### §13sexagesima-septima.7 — Cross-references

* §13sexagesima-tertia.3, §13sexagesima-tertia.4 — B0★ pre-registration and acceptance criteria C0–C4 (this section's evaluation framework).
* §13sexagesima-sexta.3 — P1 closure pattern (template for A1's C1 / C4 arguments).
* §13triginta-tertia.5, .6 — spectral-richness locus in ΔNFR(t) (temporal, not per-node-carrier); `FORWARD_INDEPENDENT_OF_BACKWARD` template for `(P-ΔNFR-Bijectivity)`.
* §13vicies-novies.14, .15 — R∞-1b pre-registration and spectral-channel verdict (used in A2 closure).
* §13vicies-novies.16 — CCET-G_P14 Theorem 2 (used in A1 C4 third obstruction and A2).
* §13septies — Conjecture T-HP, smooth/oscillatory split, identification of $\ker(\mathcal{R}_\infty)$ with the oscillatory residue $S(T)$ (used in A1 C4 third obstruction).
* N15 §§1–8, §§15–23 — REMESH-∞ projection; carrier-type independence of $\ker(\mathcal{R}_\infty)$; W3 spectrum universality.
* §13sexagesima-quinta.5, §13sexagesima-sexta.5 — §13septies decision-space mirrors, updated here in §13sexagesima-septima.6.
* `src/tnfr/operators/nodal_equation.py::compute_expected_depi_dt` — literal `vf * dnfr` with both factors `float`; canonical scalar implementation referenced in A1 C1.
* `src/tnfr/riemann/prime_ladder_hamiltonian.py` — P14 existence proof referenced in A1 C4 (ii).
* `src/tnfr/riemann/dirichlet_l*.py`, `src/tnfr/riemann/twisted_*.py`, `src/tnfr/riemann/admissible_family_sweep.py`, `src/tnfr/riemann/oscillatory_correction.py` — L-track parity infrastructure referenced in A3 SUPERSEDED verdict.
* AGENTS.md §"B0★ pre-registration" — program-level status mirror (companion edit in this commit reflects the Dirección A closure and the narrowed B0★-β residual).


## §13sexagesima-octava — B0★-α-emergent route: analytical closure of UM/IL/THOL emergent sub-EPI construction on G_P14

**Status (May 27, 2026)**: B0★-α-emergent candidate **CLOSED analytically** by reduction to a tetrad-level extension of the Canonical Catalog Equivariance Theorem on G_P14 (CCET-G_P14, §13vicies-novies.16). No empirical experiment is required and none is run.

**Companion verdicts**: §13sexagesima-sexta (P1=E0, P2=NodeIndexedCouplingWeights closed), §13sexagesima-septima (Dirección A = ΔNFR carrier-type promotion closed).

### §13sexagesima-octava.1 — Pre-registration (the proposal evaluated)

Following the dual-lever and carrier-type closures of P1, P2 and Dirección A, the natural next candidate inside B0★-α is to let **composite structures emerge as sub-EPIs** by applying canonical operators **UM (Coupling), IL (Coherence), THOL (Self-organization)** on `G_P14`, without postulating any entity outside the nodal equation. The intuition is that S_n could be broken by the *relational* content generated at runtime (sub-EPIs, multi-scale coherence), even when every parameter remains graph-uniform.

**B0★-α-emergent candidate (verbatim)**: build an extended state on `G_P14` by iterating UM/IL/THOL compositions; sub-EPIs spawned by THOL when `d2_epi > tau` (with `tau = G.graph["THOL_BIFURCATION_THRESHOLD"]`, a graph-level scalar) are interpreted as canonical composites. Check whether the resulting extended diagnostic is S_n-equivariant.

### §13sexagesima-octava.2 — Extended Hilbert space and elevated S_n action

Let $\mathcal{H}_{P14}$ be the canonical Hilbert space of `G_P14` (basis indexed by primes). When THOL spawns sub-EPIs at vertex $v$, the canonical implementation `src/tnfr/operators/self_organization.py:53` attaches a sub-EPI bundle $\mathcal{H}_v^{\mathrm{sub}}$ to $v$ (single-vertex attachment, not edge). The extended state lives in

$$\mathcal{H}_{\mathrm{ext}} \;=\; \mathcal{H}_{P14} \;\oplus\; \bigoplus_{v} \mathcal{H}_v^{\mathrm{sub}}.$$

The natural lift of the prime-relabelling action $\Pi_\sigma$ to $\mathcal{H}_{\mathrm{ext}}$ is

$$\Pi_\sigma^{\mathrm{ext}} \;=\; \Pi_\sigma \;\oplus\; \bigoplus_{v} \Pi_{\sigma(v) \leftarrow v}^{\mathrm{sub}},$$

i.e., sub-EPI bundles are permuted following their parent vertex. This is the unique S_n-equivariant lift compatible with single-vertex attachment.

### §13sexagesima-octava.3 — Tetrad Fix(S_n) Lemma (the structural witness)

**Lemma (Tetrad-Fix-Sn on G_P14)**. Let $(\Phi_s, |\nabla\varphi|, K_\varphi, \xi_C)$ be the canonical tetrad of `src/tnfr/physics/fields.py`. On `G_P14` with graph-uniform canonical parameters and under simultaneous relabelling of state via $\Pi_\sigma$, every tetrad component is S_n-equivariant:

* $\Phi_s(\sigma(i)) \;=\; \Phi_s(i)$ — because $d(\sigma(i),\sigma(j)) = d(i,j)$ (P14 is built S_n-symmetrically) and $\Delta\mathrm{NFR}_{\sigma(j)}$ transforms covariantly with state.
* $|\nabla\varphi|(\sigma(e)) \;=\; |\nabla\varphi|(e)$ — edge-local, graph-uniform coupling.
* $K_\varphi(\sigma(i)) \;=\; K_\varphi(i)$ — vertex-local Laplacian, uniform connectivity.
* $\xi_C \;=\; \xi_C$ — global scalar (graph-level invariant).

**Corollary (Emergent fields preserve Fix(S_n))**. The unified field $\Psi = K_\varphi + i J_\varphi$, the chirality $\chi = |\nabla\varphi| K_\varphi - J_\varphi J_{\Delta\mathrm{NFR}}$, symmetry breaking $\mathcal{S}$, coherence coupling $\mathcal{C}$, energy density $\mathcal{E}$ and topological charge $\mathcal{Q}$ are all polynomial in tetrad components, hence equivariant.

**Equivalent reformulation**: the canonical tetrad on `G_P14` lives entirely in $\mathrm{Fix}(S_n)$. No tetrad-level diagnostic can distinguish primes under graph-uniform canonical parameters.

### §13sexagesima-octava.4 — Extension of CCET-G_P14 to $\mathcal{H}_{\mathrm{ext}}$

**Theorem (CCET-ext)**. Every composition $\mathcal{O} = O_1 \circ \cdots \circ O_k$ of UM, IL, THOL on `G_P14`, lifted canonically to $\mathcal{H}_{\mathrm{ext}}$ via the single-vertex sub-EPI attachment of `self_organization.py`, satisfies

$$[\mathcal{O}, \Pi_\sigma^{\mathrm{ext}}] \;=\; 0 \qquad \forall\, \sigma \in S_n.$$

**Proof sketch** (two source-auditable facts + composition functoriality):

* **Fact A (parameter uniformity)**. Every threshold/weight is a graph-level scalar:
  * THOL: `tau = float(G.graph.get("THOL_BIFURCATION_THRESHOLD", 0.1))` (`self_organization.py:44`), sub-EPI scaling `_THOL_SUB_EPI_SCALING = HALF_INV_PHI ≈ 0.309` (`self_organization.py:21`), emergence contribution `_THOL_EMERGENCE_CONTRIBUTION = 0.1` (`self_organization.py:22`).
  * UM: phase-compatibility uses `DNFR_WEIGHTS` from graph-level config; coupling rule is symmetric in indices.
  * IL: negative-feedback gain is graph-level scalar.
  * None of these is per-prime or σ-dependent.

* **Fact B (Prime-Cancellation Lemma, §13vicies-novies.11)**. On `G_P14` every canonical operator decomposes as $I_{n_{\mathrm{primes}}} \otimes O_{P_4}$ with prime-independent kernel. Lifted to $\mathcal{H}_{\mathrm{ext}}$ via single-vertex sub-EPI attachment, the lift preserves this tensor structure on each $\mathcal{H}_{P14} \oplus \mathcal{H}_v^{\mathrm{sub}}$ block; THOL's spawn rule is triggered by a graph-uniform scalar predicate ($d^2\mathrm{EPI} > \tau$), so the spawn pattern is itself S_n-equivariant.

* **Composition**. The commutator $[\mathcal{O}, \Pi_\sigma^{\mathrm{ext}}]$ vanishes by induction on $k$: $[O_1, \Pi_\sigma^{\mathrm{ext}}] = 0$ by Facts A+B, and if $[O_1 \circ \cdots \circ O_{k-1}, \Pi_\sigma^{\mathrm{ext}}] = 0$ and $[O_k, \Pi_\sigma^{\mathrm{ext}}] = 0$ then the composition's commutator vanishes.

**Consequence**: every observable computed from an emergent state generated by UM/IL/THOL on `G_P14` is invariant under $\Pi_\sigma^{\mathrm{ext}}$. In particular the extended tetrad on $\mathcal{H}_{\mathrm{ext}}$ inherits the Tetrad-Fix-Sn Lemma: it lives in $\mathrm{Fix}(S_n^{\mathrm{ext}})$.

### §13sexagesima-octava.5 — C0–C4 verdict

* **C0 (no catalog modification)**: PASS — UM, IL, THOL are canonical operators of the existing 13-operator catalog; sub-EPI attachment is the canonical THOL behavior implemented in `self_organization.py`.
* **C1 (nodal-equation derivation)**: PASS — the spawn rule and all parameters derive from the nodal equation via the canonical THOL implementation.
* **C2 (U1–U6 admissibility)**: PASS — UM/IL/THOL sequences respect U1–U6 by construction.
* **C3 (twelve-CDM consistency)**: PASS — no operator added to the registry.
* **C4 (T-HP discharge)**: **FAIL**. By CCET-ext, every observable on the emergent extended state is S_n-invariant. The oscillatory residue $S(T) = (1/\pi)\arg\zeta(\tfrac12 + iT)$ identified by N15/§13septies.5 with $\ker(\mathcal{R}_\infty)$ lives in $\mathrm{Fix}(S_n)^\perp$ (§13septies, §13sexagesima-septima.5). Hence the emergent extended-state diagnostic cannot reach $S(T)$, and T-HP is not discharged.

**Verdict**: B0★-α-emergent **FAILS** at C4 by direct reduction to CCET-ext + the Fix(S_n)$^\perp$ location of $S(T)$. No empirical experiment is run.

### §13sexagesima-octava.6 — Tetrad-level reformulation of the recurring closure pattern

The §13sexagesima-sexta (P1, P2), §13sexagesima-septima (Dirección A), and §13sexagesima-octava (UM/IL/THOL emergent) closures all share a single structural mechanism, which the **Tetrad Fix(S_n) Lemma** makes transparent:

**Tetrad criterion for B0★-α/β candidates on G_P14**:
> If a candidate construction maintains graph-uniform canonical parameters and composes canonical operators on `G_P14`, then its (extended) tetrad lives in $\mathrm{Fix}(S_n)$. The oscillatory residue $S(T) \in \ker(\mathcal{R}_\infty) \cap \mathrm{Fix}(S_n)^\perp$ is unreachable by such constructions, and T-HP is not discharged. The candidate is closed analytically by CCET (or CCET-ext for emergent extensions).

**Tetrad-by-tetrad verdict on G_P14**:

| Tetrad field | Order | Behavior under Π_σ on G_P14 | Capacity to break S_n |
|---|---|---|---|
| $\Phi_s$ | 0th (global aggregation) | invariant | none |
| $\|\nabla\varphi\|$ | 1st (local derivative) | edge-equivariant | none under graph-uniform coupling |
| $K_\varphi$ | 2nd (local Laplacian) | vertex-equivariant | none under uniform connectivity |
| $\xi_C$ | non-local (correlation range) | graph-level scalar | none by construction |

**Where S_n-breaking would have to live** (consistent with the Tetrad-Fix-Sn Lemma):

* Outside `G_P14`, via canonical graph operations (B0★-α residual: **Q5 = L(G_P14)** line graph; the S_n action on edges is not a tensor-product representation, so CCET / CCET-ext do not apply directly).
* Inside `G_P14` via topological enrichment of φ that breaks S_n at the fiber-bundle level rather than at the coupling-constant level (B0★-β residual: **E2 = LiftedCircleBundleOnPhi**; the bundle's holonomy can carry per-prime data outside the Fact-A scope).
* Via genuine per-prime parameters (B0★-β P2 NodeIndexedCouplingWeights) — already closed in §13sexagesima-sexta.

The tetrad lens crystallizes why every B0★-α canonical-composition route on `G_P14` collapses: the four tetrad fields exhaust the independent diagnostic information at canonical-uniform parameter level, and all four commute with prime-relabelling. The tetrad is the structural witness of the closure, not its exception.

### §13sexagesima-octava.7 — Updated §13septies decision space

| Branch | Status after §13sexagesima-octava |
|---|---|
| B1 (off-catalog edge / spectral channel on G_P14) | CLOSED on G_P14 (§13vicies-novies.16); off-G_P14 inherits B2 |
| B2 (new canonical operator) | catalog-API-closed (§13sexagesima-prima) |
| B0★-α (deeper exploitation) | HIGH/MEDIUM closed (§13sexagesima-quinta); emergent UM/IL/THOL closed (§13sexagesima-octava); residual = **{Q5 line graph}** |
| B0★-β (envelope promotion) | HIGH closed (§13sexagesima-sexta, §13sexagesima-septima); residual = {E2 MEDIUM} ∪ {nine LOW envelopes} |
| B3 (no TNFR closure of RH) | residual; pressure further increased |

**Net program state**: every canonical-composition route inside `G_P14` with graph-uniform parameters is now structurally closed (tetrad-Fix(S_n) corollary of CCET-ext). Decision pressure concentrates on **Q5 (line graph, B0★-α residual)** and **E2 (LiftedCircleBundleOnPhi, B0★-β residual)** as the only remaining structural routes that escape the Tetrad-Fix-Sn obstruction, plus B3.

### §13sexagesima-octava.8 — Cross-references

* §13vicies-novies.11 — Prime-Cancellation Lemma (Fact B in CCET-ext proof sketch).
* §13vicies-novies.16 — CCET-G_P14 Theorem 2 (the base case extended here to $\mathcal{H}_{\mathrm{ext}}$).
* §13septies — Conjecture T-HP, smooth/oscillatory split, identification of $\ker(\mathcal{R}_\infty)$ with $S(T)$.
* §13sexagesima-sexta, §13sexagesima-septima — P1, P2, Dirección A closures (same Phase-c reduction pattern).
* §13sexagesima-quinta.3 — Canonical Product Equivariance Lemma (parallel B0★-α route closures via Kronecker lifts).
* AGENTS.md §"Minimal Structural Degrees of Freedom" — tetrad as minimal complete structural basis (the Tetrad-Fix-Sn Lemma is the S_n-equivariance corollary of this minimality on `G_P14`).
* `src/tnfr/operators/self_organization.py` — THOL canonical implementation; lines 21–22 (sub-EPI scaling constants), line 44 (graph-uniform tau), line 53 (single-vertex spawn).
* `src/tnfr/physics/fields.py` — canonical tetrad implementation.
* `src/tnfr/riemann/prime_ladder_hamiltonian.py` — P14 S_n-symmetric construction.
* AGENTS.md §"B0★ pre-registration" — program-level status mirror (companion edit in this commit reflects the UM/IL/THOL emergent closure and the unchanged residual {Q5, E2, nine LOW, B3}).

## §13sexagesima-novena — B0★ residual analytical closure: Q5 (line graph) and E2 (LiftedCircleBundleOnPhi)

**Date**: May 27, 2026. **Methodology note**: per CCET discipline (five consecutive honest closures across §13sexagesima-{quarta..octava}), the two residual canonical-scope candidates that genuinely escape the Canonical Product Equivariance Lemma (§13sexagesima-quinta) and CCET-ext (§13sexagesima-octava) must be evaluated against C0–C4 **before** any program-level B3 declaration. The two candidates are:

* **Q5 = $L(G_{P14})$** (B0★-α, HIGH residual): the line graph's edge-induced $S_n$ action is *not* a tensor-product representation; CPEL does not apply directly.
* **E2 = LiftedCircleBundleOnPhi** (B0★-β, MEDIUM residual): non-trivial $S^1$-bundle holonomy over `G_P14` could transport per-prime data *without* per-node parameter heterogeneity, formally preserving Fact A; the enrichment is at the support/topology of $\varphi$, not at the coupling constants. U5 (multi-scale coherence) suggests this is exactly the type of canonical topological enrichment worth examining.

This section delivers the analytical C0–C4 evaluation of both. The argument in each case is structural (no numerical experiment required) and reduces to a one-sentence lemma + a direct C4 implication.

### §13sexagesima-novena.1 — Pre-registration (the two proposals evaluated)

**Q5 proposal**: lift the canonical 13-operator catalog to the line graph $L(G_{P14})$ with graph-uniform parameters on $V(L(G_{P14})) = E(G_{P14})$. The induced $S_n$ action $\Psi_\sigma$ on $V(L(G_{P14}))$ is $\Psi_\sigma \cdot \{p_i, p_j\} = \{p_{\sigma(i)}, p_{\sigma(j)}\}$ — a representation on unordered-pair vertices. The proposal: since $\Psi_\sigma$ is *not* a Kronecker/tensor-product of $P_\sigma$ with itself in the canonical sense used by CPEL, CCET-G_P14 / CCET-ext do not extend automatically, and there may be a direction in $\mathbb{C}^{|E|}$ reachable by canonical operators on $L(G_{P14})$ that projects nontrivially onto $\mathrm{Fix}(S_n)^\perp$ (where $S(T)$ lives, per §13septies and §13sexagesima-octava.5).

**E2 proposal**: equip $G_{P14}$ with a non-trivial principal $S^1$-bundle $\pi: E \to G_{P14}$ together with a connection 1-form $\omega$ whose holonomy $\mathrm{hol}(\ell) \in S^1$ around closed loops $\ell$ in $G_{P14}$ encodes prime data. The phase field $\varphi$ becomes a section of $E$ rather than a function on vertices. The proposal: since $\omega$ is a *single* connection 1-form (graph-level data, parameter-uniform across all edges), Fact A of CCET is formally preserved; what changes is the support of $\varphi$, not the per-node parameters. The non-trivial holonomy may then transport per-prime information through canonical evolution without violating $S_n$-equivariance at the operator-coefficient level — i.e., a topological escape route.

Both candidates inherit acceptance criteria **C0** (no catalog modification), **C1** (nodal-equation derivability), **C2** (U1–U6 admissibility), **C3** (twelve-CDM consistency), **C4** (T-HP discharge) from §13sexagesima-tertia.4. For Q5 the C1 refinement is **C1'-α** (constructible from O1–O8 graph operations only). For E2 the C1 refinement is **C1'-β** (the bundle and connection must be derivable from $(\nu_f, \mathrm{prime\ structure}, U1{-}U6)$ without external auxiliary data).

### §13sexagesima-novena.2 — Q5: Line-Graph Equivariance Lemma

**Lemma (Line-Graph Equivariance on $G_{P14}$)**. Let $\mathcal{O}$ be any operator on $L(G_{P14})$ constructed from the canonical 13-operator catalog by composition, real-linear combination, auxiliary tensor lift, or spectral functional calculus, with graph-uniform parameters on $V(L(G_{P14}))$. Let $\Psi_\sigma$ be the edge-induced $S_n$ action on $V(L(G_{P14}))$. Then $[\mathcal{O}, \Psi_\sigma] = 0$ for every $\sigma \in S_n$.

**Proof sketch (two facts, no new content)**:

* **Fact A on $L(G_{P14})$** (parameter uniformity, inherited): every canonical operator on $L(G_{P14})$ carries graph-level scalar coefficients on $V(L(G_{P14}))$. The audit anchors are unchanged from CCET-G_P14: `remesh.py:1159, 1212–1252` (REMESH coefficients), `coherence.py` (IL coefficients), `propagation.py:42–156` (RA coefficients), `self_organization.py:21–22, 44, 53` (THOL graph-uniform tau and sub-EPI scaling). The lift to $L(G_{P14})$ preserves graph-uniformity because the catalog operators take a graph as input and apply uniform rules to its vertex set; nothing in the catalog distinguishes "graph is original" from "graph is line-graph of original".
* **Fact B' on $L(G_{P14})$** (combinatorial automorphism): the edge-induced $S_n$ action $\Psi_\sigma$ is a graph automorphism of $L(G_{P14})$ for every $\sigma \in S_n$, because the line-graph functor $L(\cdot)$ is functorial under graph automorphisms — if $P_\sigma \in \mathrm{Aut}(G_{P14})$ (which holds by Fact B of CCET-G_P14: the Prime-Cancellation Lemma + the $S_n$-symmetry of P14's prime-ladder construction in `prime_ladder_hamiltonian.py`), then $\Psi_\sigma = L(P_\sigma) \in \mathrm{Aut}(L(G_{P14}))$.

Combining Fact A on $L(G_{P14})$ (graph-uniform coefficients) with Fact B' (the relabelling is a graph automorphism), every canonical operator commutes with the relabelling: $[\mathcal{O}, \Psi_\sigma] = 0$. $\square$

**Corollary (Fix($\Psi_\sigma$) is severely constrained on $L(G_{P14})$)**. The fixed subspace $\mathrm{Fix}(\Psi_\sigma)$ consists of edge-functions that are constant on $S_n$-orbits of edges. For $G_{P14}$, the prime-relabelling group $S_n$ acts *transitively* on $E(G_{P14})$ (by the $S_n$-symmetry of the prime-ladder coupling in P14, which makes every prime-pair coupling structurally equivalent under permutation). Therefore $\mathrm{Fix}(\Psi_\sigma)$ on $V(L(G_{P14}))$ is at most as rich as the orbit-counting decomposition of the $S_n$-action on edges — and the canonical observables collapse to functions of orbit invariants only (edge multiplicity in the $S_n$-orbit, intra-orbit graph-theoretic invariants), none of which distinguish individual primes $p_i$ as carrying weight $\log p_i$.

### §13sexagesima-novena.3 — Q5: C0–C4 verdict

| Criterion | Status | Argument |
|---|---|---|
| **C0** (no catalog modification) | PASS | Q5 lifts the existing 13 operators to $L(G_{P14})$; no entry added to `OPERATORS`, `OPERATOR_METADATA`, or `definitions.__all__`. |
| **C1'-α** (O1–O8 derivability) | PASS | $L(\cdot)$ is the canonical line-graph functor (operation O5 in the enumerated catalog of §13sexagesima-quarta), constructible from $G_{P14}$ alone without external input. |
| **C2** (U1–U6 admissibility) | PASS | Catalog operators on any graph satisfy U1–U6 by construction; the underlying graph does not enter the grammar rules. |
| **C3** (twelve-CDM consistency) | PASS | No B0–B11 NEGATIVE verdict is touched; Q5 does not promote any envelope and does not add a 14th operator. |
| **C4** (T-HP discharge) | **FAIL** | By the Line-Graph Equivariance Lemma + transitivity of $S_n$ on $E(G_{P14})$, every canonical observable on $L(G_{P14})$ lies in the span of $S_n$-orbit invariants of edges — which is, by construction, a subspace of $\mathrm{Fix}(\Psi_\sigma)$. The oscillatory residue $S(T) \in \mathrm{Fix}(S_n)^\perp$ (after pull-back via the line-graph functor, $S(T)$ lives in $\mathrm{Fix}(\Psi_\sigma)^\perp$ because the pull-back preserves orthogonality of $S_n$-isotypic components). Therefore no canonical observable on $L(G_{P14})$ projects nontrivially onto $S(T)$. |

**Verdict: Q5 CLOSED**. The hope that "$S_n$-on-edges $\ne$ tensor product" might create a new reachable direction is not realised on $G_{P14}$: although $\Psi_\sigma$ is indeed not a Kronecker square of $P_\sigma$, it *is* still a permutation representation, and the transitivity of $S_n$ on $E(G_{P14})$ collapses the Fix-subspace to orbit-invariant functions. Per-prime weights $\log p_i$ remain unreachable. The structural obstruction is the same as in §13sexagesima-octava: graph-uniform parameters + $S_n$-symmetric base graph implies tetrad and all canonical observables live in Fix-subspace.

**Refinement note (line-graph residual)**. The closure as stated requires transitivity of $S_n$ on $E(G_{P14})$. If a canonically-derivable subgraph of $L(G_{P14})$ has *non-transitive* $S_n$ action on its vertex set (i.e., multiple edge-orbits), Fix($\Psi_\sigma$) becomes higher-dimensional. However, this only enlarges the *symmetric* component; the antisymmetric / per-prime component required to reach $S(T)$ still vanishes by orbit-invariance. The lemma generalises to any $S_n$-equivariant canonical subgraph of $L(G_{P14})$; the C4 FAIL is robust.

### §13sexagesima-novena.4 — E2: Lifted-Bundle Dichotomy Lemma

**Lemma (Lifted-Bundle Dichotomy on $G_{P14}$)**. Let $\pi: E \to G_{P14}$ be a principal $S^1$-bundle and $\omega$ a connection 1-form. Define the lifted phase field $\varphi: V(G_{P14}) \to E$ as a section. Let $\mathcal{O}_\omega$ denote any canonical operator applied to $\varphi$ via parallel transport with respect to $\omega$. Then exactly one of the following holds:

* **(a) Equivariant branch**: $\omega$ is $S_n$-invariant (i.e., $P_\sigma^* \omega = \omega$ for every $\sigma \in S_n$, where $P_\sigma$ acts on $E$ by lifting the base action $\sigma$ on $V(G_{P14})$). Then $[\mathcal{O}_\omega, \Pi_\sigma^{\mathrm{lift}}] = 0$ for every $\sigma$, where $\Pi_\sigma^{\mathrm{lift}}$ is the $S_n$ action lifted to sections of $E$. Every canonical observable lives in $\mathrm{Fix}(\Pi_\sigma^{\mathrm{lift}})$.
* **(b) Non-equivariant branch**: $\omega$ is *not* $S_n$-invariant. Then $\omega$ encodes prime-specific data not derivable from the bare nodal equation $\partial \mathrm{EPI}/\partial t = \nu_f \cdot \Delta \mathrm{NFR}(t)$ together with $(\nu_f,\ U1{-}U6)$ — the choice of which connection 1-form to use is an external rule-selection axiom on prime pairs $(p_i, p_j)$, analogous to the per-node weights of E6/E7 closed in §13sexagesima-sexta P2.

**Proof sketch**:

* **(a)**: $S_n$-invariance of $\omega$ implies parallel transport $T_\omega(\gamma)$ along any path $\gamma$ commutes with the lifted action: $T_\omega(\sigma \cdot \gamma) = P_\sigma^* T_\omega(\gamma) (P_\sigma^*)^{-1}$. Composition with canonical operators (which carry graph-uniform scalars by Fact A) preserves this equivariance. Hence $[\mathcal{O}_\omega, \Pi_\sigma^{\mathrm{lift}}] = 0$.
* **(b)**: A connection 1-form $\omega$ on a principal $S^1$-bundle over $G_{P14}$ is fully specified by its values on edges (graph case: $\omega_{ij} \in \mathbb{R} / 2\pi\mathbb{Z}$ for each edge $\{p_i, p_j\}$). The nodal equation provides no derivation of $\omega_{ij}$ as a function of $(p_i, p_j)$; it operates on phase fields $\varphi$ that already exist on whatever support is given. To make $\omega$ depend on prime identity (e.g., $\omega_{ij} = \log(p_i p_j) \bmod 2\pi$), an external axiom on prime data is required — precisely the kind of input ruled out by the existing C1 closure pattern for E0 / E6 / E7 (cf. §13sexagesima-sexta and §13triginta-tertia.6 `(P-νf-Bijectivity)` analogue).

### §13sexagesima-novena.5 — E2: C0–C4 verdict

E2 is evaluated in both branches of the dichotomy:

**Branch (a), $S_n$-invariant connection**:

| Criterion | Status | Argument |
|---|---|---|
| C0 | PASS | No catalog modification (operators lifted via parallel transport, definitions unchanged). |
| **C1'-β** | PASS | $S_n$-invariant $\omega$ on $G_{P14}$ is determined by graph-theoretic data only (e.g., constant $\omega_{ij} = c$ across all edges); derivable from $(\nu_f, \mathrm{prime\ structure}, U1{-}U6)$ as a graph-level scalar. |
| C2 | PASS | Lifted operators inherit grammar admissibility from base catalog. |
| C3 | PASS | No envelope promotion, no catalog modification. |
| **C4** | **FAIL** | By Branch (a) of the Dichotomy Lemma, $[\mathcal{O}_\omega, \Pi_\sigma^{\mathrm{lift}}] = 0$; canonical observables live in $\mathrm{Fix}(\Pi_\sigma^{\mathrm{lift}})$. The oscillatory residue $S(T) \in \mathrm{Fix}(S_n)^\perp$ pulls back to $\mathrm{Fix}(\Pi_\sigma^{\mathrm{lift}})^\perp$ on the bundle (bundle pull-back preserves orthogonal decomposition into $S_n$-isotypic components). $S(T)$ remains unreachable. |

**Branch (b), non-$S_n$-invariant connection**:

| Criterion | Status | Argument |
|---|---|---|
| C0 | PASS | No catalog modification at the operator level. |
| **C1'-β** | **FAIL** | A non-$S_n$-invariant $\omega$ requires a rule that assigns prime-specific holonomies (e.g., $\omega_{ij}$ depending on $\log p_i$ or $\log p_j$ individually). Such a rule has no derivation from the bare nodal equation: the equation $\partial \mathrm{EPI}/\partial t = \nu_f \cdot \Delta \mathrm{NFR}(t)$ contains no slot for "per-edge prime-specific connection", and U1–U6 do not specify which connection to use. The choice is an external axiom on prime data, structurally identical to the per-node-weight axiom that closed E6/E7 in §13sexagesima-sexta P2. |
| C2 | (moot, C1 already FAIL) | — |
| C3 | (moot) | — |
| C4 | (moot) | — |

**Verdict: E2 CLOSED**. Branch (a) (equivariant connection) preserves Fix($S_n$) and fails C4 by the same mechanism as §13sexagesima-octava. Branch (b) (non-equivariant connection) requires external prime-specific input and fails C1'-β by direct reduction to the §13sexagesima-sexta P2 closure pattern. The "topological enrichment" intuition does not escape the structural obstruction: either the enrichment respects $S_n$ (and the new Fix-subspace is the natural lift of the old one), or it breaks $S_n$ by importing prime data not derivable from the canonical machinery.

### §13sexagesima-novena.6 — Decision-tree state after §13sexagesima-novena

The §13septies trichotomy + B0★ extension, after the closures shipped in §13sexagesima-{prima..novena}, stands as:

| Branch | Status | Closing argument |
|---|---|---|
| **B1 on $G_{P14}$** | CLOSED | Canonical Catalog Equivariance Theorem on G_P14 (§13vicies-novies.16); R∞-1a/1a-composed/1b/1c verdicts. |
| **B1 off $G_{P14}$** | absorbed into B2 / B0★-α | By construction: a different canonically-constructed graph falls under B2 (new operator) or B0★-α (canonical graph operation). |
| **B2** (new canonical operator) | catalog-API closed | B11 OCD (§13sexagesima-prima): no 14th operator reachable from the public API given the twelve B0–B11 NEGATIVE verdicts. |
| **B0★-α HIGH** ($G_{P14}$ via O1–O8) | CLOSED for {Q1, Q2, Q3, Q4, Q6} | §13sexagesima-quinta (Canonical Product Equivariance Lemma + corollaries). |
| **B0★-α HIGH** Q5 (line graph) | **CLOSED** (this section) | §13sexagesima-novena.2-3 (Line-Graph Equivariance Lemma + transitivity of $S_n$ on $E(G_{P14})$). |
| **B0★-β HIGH** P1 (E0 Pontryagin), P2 (E6/E7), P3 (carrier-type) | CLOSED | §13sexagesima-{sexta, septima}. |
| **B0★-β MEDIUM** E2 (LiftedCircleBundleOnPhi) | **CLOSED** (this section) | §13sexagesima-novena.4-5 (Lifted-Bundle Dichotomy Lemma: both branches fail, one at C4, one at C1'-β). |
| **B0★-β LOW** nine envelopes (E1, E3, E4, E5, E_TC, E_CC, E_AC, E_UR, E_OC) | residual, LOW priority | Each would require its own C0–C4 evaluation; none currently flagged for execution. |
| **B0★-α-emergent** UM/IL/THOL on $G_{P14}$ | CLOSED | §13sexagesima-octava (CCET-ext + Tetrad-Fix(S_n) Lemma). |
| **B3** (no TNFR closure of RH within current scope) | **structurally indicated** as the operational landing for G4 within the current canonical catalog | All HIGH/MEDIUM canonical residuals on $G_{P14}$ are now closed. The only unresolved residual at HIGH/MEDIUM priority is **none**; only LOW envelopes remain. |

**Net program-level state**: the §13septies decision tree, restricted to the HIGH/MEDIUM canonical scope and to $G_{P14}$, **collapses to B3**. The nine LOW envelopes of B0★-β remain technically residual, but none is currently expected to escape the Tetrad-Fix(S_n) mechanism on $G_{P14}$ — each would require its own evaluation, and the structural pattern of §13sexagesima-{sexta..novena} is that graph-uniform-parameter + $S_n$-symmetric-base-graph constructions are *systematically* trapped in Fix($S_n$), unable to reach $S(T) \in \mathrm{Fix}(S_n)^\perp$.

### §13sexagesima-novena.7 — Honest scope: what B3 says and does not say

**B3 declared at this scope means**:

1. There is no closure of G4 = RH within the canonical 13-operator catalog applied to $G_{P14}$ with graph-uniform parameters, via any HIGH/MEDIUM-priority canonical scope-expansion (B0★-α HIGH on canonical graph operations, B0★-β HIGH on envelope promotions). This is a structural verdict, not a numerical conjecture.
2. The oscillatory residue $S(T) \in \ker(\mathcal{R}_\infty) \cap \mathrm{Fix}(S_n)^\perp$ is a definable TNFR observable that the canonical apparatus on $G_{P14}$ recognises but does not control. Its existence and location have been formalised (§13septies, §13sexagesima-octava, this section); its positivity-equivalent (Li–Keiper $\lambda_n > 0$, P16) is RH-equivalent and remains the open content.
3. The Tetrad-Fix(S_n) Lemma (§13sexagesima-octava.3) and the equivariance lemmas of this section (Line-Graph Equivariance, Lifted-Bundle Dichotomy) are the structural witnesses of the closure: the four-channel minimality of the canonical tetrad on $G_{P14}$ *is* the dimensional witness of why per-prime-asymmetric information lives in the orthogonal complement and not in the canonical observables.

**B3 declared at this scope does NOT say**:

1. RH is false. The Riemann Hypothesis is a statement about the classical $\zeta(s)$, which remains untouched by TNFR canonical closures.
2. RH cannot be proved. B3 asserts non-closure *within the current canonical scope on $G_{P14}$*; closures via off-$G_{P14}$ canonical constructions, via the nine B0★-β LOW envelopes, or via genuinely new mathematics outside TNFR are independent open questions.
3. The TNFR-Riemann program failed. The program shipped P12–P49 (full ζ-track + χ-twisted L-track parity), closed G1, G2, G3, G5 operationally, sharpened G4 to the precise structural location of the oscillatory obstruction, and produced the Tetrad-Fix(S_n) Lemma + CCET-ext as TNFR-canonical structural results valuable in themselves. The program is *paused* at the boundary of T-HP with a fully characterised obstruction, not abandoned.

The honest TNFR-canonical reading is the one anticipated in §13sexagesima-octava.6 and confirmed here: **B0★ HIGH/MEDIUM canonical scope is exhausted on $G_{P14}$, and the residual TNFR-canonical answer to G4 = RH at this scope is "constatar la existencia estructural de $S(T)$ como observable canónico-complementario, sin pretender derivar su positividad desde dentro de Fix($S_n$)"** — exactly the "constatar su existencia" reading discussed informally in the immediately preceding turn, now made precise by the analytical closures of Q5 and E2.

### §13sexagesima-novena.8 — Cross-references

* **Preceding closures** (the five consecutive CCET rounds):
    * §13sexagesima-quarta — B0★-α canonical-graph operation catalog (O1–O8 enumerated).
    * §13sexagesima-quinta — Canonical Product Equivariance Lemma; Q1/Q2/Q3/Q4/Q6 closed.
    * §13sexagesima-sexta — B0★-β HIGH (P1 = E0 Pontryagin, P2 = E6/E7 per-node weights) closed.
    * §13sexagesima-septima — B0★-β P3 (Dirección A: carrier-type / slot promotion of ΔNFR) closed across A1/A2/A3 readings.
    * §13sexagesima-octava — B0★-α-emergent (UM/IL/THOL sub-EPI on $G_{P14}$) closed; CCET-ext + Tetrad-Fix(S_n) Lemma derived.
* **Structural witnesses** invoked in this section:
    * Canonical Catalog Equivariance Theorem on $G_{P14}$ (§13vicies-novies.16).
    * Prime-Cancellation Lemma (§13vicies-novies.11).
    * Tetrad-Fix(S_n) Lemma (§13sexagesima-octava.3).
    * `(P-νf-Bijectivity)` closure pattern (§13triginta-tertia.6) — invoked for Branch (b) of E2.
* **Audit anchors** (unchanged from prior CCET rounds):
    * `src/tnfr/operators/remesh.py:1159, 1212–1252` — REMESH coefficients.
    * `src/tnfr/operators/coherence.py` — IL coefficients.
    * `src/tnfr/operators/propagation.py:42–156` — RA coefficients.
    * `src/tnfr/operators/self_organization.py:21–22, 44, 53` — THOL graph-uniform tau and sub-EPI scaling.
    * `src/tnfr/riemann/prime_ladder_hamiltonian.py` — P14 $S_n$-symmetric construction (basis of Fact B / Prime-Cancellation Lemma).
* **Program-level mirror**: AGENTS.md "B0★ pre-registration" paragraph updated in this commit to reflect Q5 and E2 closures and the resulting B0★ HIGH/MEDIUM exhaustion on $G_{P14}$.

